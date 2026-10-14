"""Anthropic Claude adapter (Messages API, beta surface).

Request shape and why:

* ``output_config.format`` (json_schema, strict) — the model can only emit our schema (P5).
* ``output_config.effort`` — the only depth/cost control on Claude Opus 5.5. Sampling params,
  thinking budgets and ``thinking: disabled`` are rejected by that model, so none are sent;
  thinking stays adaptive by default.
* ``fallbacks="default"`` + beta ``server-side-fallback-2026-07-01`` — when a safety classifier
  declines (often a false positive on incident runbooks that mention attacks), the API re-runs
  the request on a recommended fallback model instead of failing the step.
* ``cache_control`` on the system prompt — the safety preamble + task instructions are
  identical for every chunk of a document, so later chunks read them from the prompt cache.
* No assistant prefill and no forced ``tool_choice`` (both rejected on this model).

``anthropic`` is imported lazily so the package imports (and LLM-OFF mode works) without the
optional extra installed (P4). Missing package or credentials surface as
:class:`LLMDisabledError` on the first call, which callers turn into the deterministic fallback.
"""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, Protocol, TypeVar, get_args

from pydantic import BaseModel

from jettae.llm.base import (
    LLMDisabledError,
    LLMError,
    LLMOutputError,
    LLMRefusalError,
    LLMResult,
    LLMUsage,
    UntrustedDocument,
)
from jettae.llm.prompting import Message, Reply, build_system, complete_with_repair, render_user_content
from jettae.llm.schema import JsonSchema, to_strict_json_schema

_T = TypeVar("_T", bound=BaseModel)

Effort = Literal["low", "medium", "high", "xhigh", "max"]

DEFAULT_MODEL: Final = "claude-opus-5-5"
FALLBACK_BETA: Final = "server-side-fallback-2026-07-01"
_FINISHED: Final = "end_turn"
_MISSING_CREDENTIALS_HINT: Final = "authentication method"
_KEY_HINT: Final = "API 키 확인 필요"

_log = logging.getLogger(__name__)


class _MessagesAPI(Protocol):
    async def create(self, *args: Any, **kwargs: Any) -> Any: ...


class _BetaAPI(Protocol):
    @property
    def messages(self) -> _MessagesAPI: ...


class AnthropicClientLike(Protocol):
    """The slice of ``anthropic.AsyncAnthropic`` this adapter uses (lets tests inject fakes)."""

    @property
    def beta(self) -> _BetaAPI: ...


@dataclass
class _Totals:
    """Usage summed over the first attempt and the optional repair attempt."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    served_model: str | None = None

    def add(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        self.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        self.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
        self.cache_read_tokens += int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        model = getattr(response, "model", None)
        if isinstance(model, str) and model:
            self.served_model = model


class AnthropicLLM:
    """:class:`~jettae.llm.base.LLMClient` backed by Claude structured outputs."""

    provider: str = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: Effort = "medium",
        *,
        client: AnthropicClientLike | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        use_fallbacks: bool = True,
    ) -> None:
        if effort not in get_args(Effort):
            raise ValueError(f"지원하지 않는 effort 값입니다: {effort!r}")
        self.model = model
        self.effort: Effort = effort
        self.use_fallbacks = use_fallbacks
        self._unavailable: str | None = None
        self._client: AnthropicClientLike | None = client
        if client is None:
            self._client, self._unavailable = _default_client(timeout=timeout, max_retries=max_retries)
        self._accepted_params = (
            _named_params(self._client.beta.messages.create) if self._client is not None else None
        )

    async def structured(
        self,
        *,
        task: str,
        instructions: str,
        schema: type[_T],
        documents: tuple[UntrustedDocument, ...] = (),
        context: str = "",
        max_tokens: int = 16000,
    ) -> LLMResult[_T]:
        client = self._require_client(task)
        strict_schema = to_strict_json_schema(schema)
        system_text = build_system(instructions)
        totals = _Totals()
        started = time.perf_counter()

        async def send(history: list[Message]) -> Reply:
            response = await self._create(
                client,
                history,
                system_text=system_text,
                strict_schema=strict_schema,
                max_tokens=max_tokens,
                task=task,
            )
            totals.add(response)
            return Reply(
                text=_response_text(response, task=task, max_tokens=max_tokens),
                assistant_turn={"role": "assistant", "content": _echo_content(response.content)},
            )

        data = await complete_with_repair(
            send, user_content=render_user_content(documents, context), schema=schema, task=task
        )
        usage = LLMUsage(
            provider=self.provider,
            model=totals.served_model or self.model,
            task=task,
            input_tokens=totals.input_tokens,
            output_tokens=totals.output_tokens,
            cache_read_tokens=totals.cache_read_tokens,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
        return LLMResult(data=data, usage=usage)

    def request_kwargs(
        self, system_text: str, history: Sequence[Message], strict_schema: JsonSchema, max_tokens: int
    ) -> dict[str, Any]:
        """Keyword arguments for ``client.beta.messages.create`` (public for inspection/tests).

        ``output_config``/``fallbacks`` go to ``extra_body`` when the installed SDK does not
        name them, so an older 1.x SDK still sends the same wire request.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}],
            "messages": list(history),
        }
        optional: dict[str, Any] = {
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": strict_schema},
            }
        }
        if self.use_fallbacks:
            kwargs["betas"] = [FALLBACK_BETA]
            optional["fallbacks"] = "default"
        accepted = self._accepted_params
        extra_body = {name: value for name, value in optional.items() if not _accepts(accepted, name)}
        kwargs.update({name: value for name, value in optional.items() if _accepts(accepted, name)})
        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs

    def _require_client(self, task: str) -> AnthropicClientLike:
        if self._client is None:
            raise LLMDisabledError(f"{self._unavailable or _KEY_HINT} (task={task})")
        return self._client

    async def _create(
        self,
        client: AnthropicClientLike,
        history: list[Message],
        *,
        system_text: str,
        strict_schema: JsonSchema,
        max_tokens: int,
        task: str,
    ) -> Any:
        kwargs = self.request_kwargs(system_text, history, strict_schema, max_tokens)
        try:
            response = await client.beta.messages.create(**kwargs)
        except Exception as exc:
            mapped = _map_error(exc, task)
            if mapped is None:
                raise
            _log.warning(
                "anthropic call failed task=%s error=%s request_id=%s",
                task,
                type(exc).__name__,
                getattr(exc, "request_id", None),
            )
            raise mapped from exc
        _log_response(response, task=task, requested_model=self.model)
        return response


# --------------------------------------------------------------------------- helpers


def _default_client(*, timeout: float, max_retries: int) -> tuple[AnthropicClientLike | None, str | None]:
    """Build the real SDK client, or explain (lazily, on first call) why the LLM is unavailable."""
    try:
        import anthropic
    except ImportError:
        return None, "anthropic 패키지가 설치되어 있지 않습니다 (pip install 'jettae-ops[anthropic]')"
    try:
        return anthropic.AsyncAnthropic(timeout=timeout, max_retries=max_retries), None
    except anthropic.AnthropicError as exc:  # credential/profile resolution happens in the constructor
        _log.warning("anthropic client unavailable: %s", type(exc).__name__)
        return None, f"{_KEY_HINT}: Anthropic 자격 증명을 찾지 못했습니다"


def _named_params(function: Callable[..., Any]) -> frozenset[str] | None:
    """Named parameters of ``function``; ``None`` means "accepts any keyword" (``**kwargs``)."""
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return None
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters):
        return None
    return frozenset(param.name for param in parameters)


def _accepts(accepted: frozenset[str] | None, name: str) -> bool:
    return accepted is None or name in accepted


def _response_text(response: Any, *, task: str, max_tokens: int) -> str:
    """Branch on ``stop_reason`` before touching ``content`` (a refusal's content is not ours)."""
    _check_stop_reason(response, task=task, max_tokens=max_tokens)
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return str(block.text)
    raise LLMOutputError(f"응답에 텍스트 블록이 없습니다 (task={task})")


def _check_stop_reason(response: Any, *, task: str, max_tokens: int) -> None:
    """Only a natural ``end_turn`` is a finished answer (P3: unknown is not pass).

    Truncations raise without a repair turn — repeating the request with a longer history would
    hit the same limit. Endings this adapter never asks for (``pause_turn``, ``tool_use``,
    ``compaction`` or a future value) are failures even if the partial text happens to parse.
    """
    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason == _FINISHED:
        return
    if stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None) if details is not None else None
        raise LLMRefusalError(f"모델이 요청을 거절했습니다 (task={task}, 분류={category or '미상'})")
    if stop_reason == "max_tokens":
        raise LLMOutputError(
            f"응답이 max_tokens={max_tokens}에서 잘렸습니다 (task={task}). "
            "문서를 더 작게 나누거나 max_tokens를 늘리세요"
        )
    if stop_reason == "model_context_window_exceeded":
        raise LLMOutputError(
            f"컨텍스트 창이 가득 차 응답이 잘렸습니다 (task={task}). 문서를 더 작게 나누세요"
        )
    raise LLMOutputError(f"응답이 정상적으로 끝나지 않았습니다 (task={task}, stop_reason={stop_reason})")


def _echo_content(content: Sequence[Any]) -> list[Any]:
    """Blocks to echo as the assistant turn of a repair request.

    Echoed unchanged (thinking blocks included — stripping them can break thinking-block
    validation), except that before the last ``fallback`` marker only ``text`` blocks are kept,
    per the server-side fallback replay rules.
    """
    kinds = [getattr(block, "type", None) for block in content]
    boundary = max((i for i, kind in enumerate(kinds) if kind == "fallback"), default=0)
    return [block for i, block in enumerate(content) if i >= boundary or kinds[i] == "text"]


def _map_error(exc: Exception, task: str) -> LLMError | None:
    """Translate SDK errors; ``None`` means "not an SDK error, re-raise unchanged"."""
    if isinstance(exc, TypeError) and _MISSING_CREDENTIALS_HINT in str(exc):
        # The SDK raises a bare TypeError when no api_key / auth_token / profile resolves.
        return LLMDisabledError(f"{_KEY_HINT}: Anthropic 자격 증명이 없습니다 (task={task})")
    return _map_sdk_error(exc, task)


def _map_sdk_error(exc: Exception, task: str) -> LLMError | None:
    """Most specific first: credentials → rate limit → other HTTP status → connection → any SDK error.

    The final ``AnthropicError`` branch matters: credential refresh (workload identity), response
    validation and middleware errors are SDK errors too, and letting them escape untyped would
    crash the caller instead of triggering its :class:`LLMError` fallback.
    """
    try:
        import anthropic
    except ImportError:
        return None
    if not isinstance(exc, anthropic.AnthropicError):
        return None
    # ``CredentialsError``/``WorkloadIdentityError`` only exist in newer 1.x SDKs.
    credentials_errors = tuple(
        cls
        for cls in (
            getattr(anthropic, "CredentialsError", None),
            getattr(anthropic, "WorkloadIdentityError", None),
        )
        if isinstance(cls, type)
    )
    if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError, *credentials_errors)):
        return LLMDisabledError(f"{_KEY_HINT} (task={task})")
    return LLMError(_sdk_error_message(exc, task))


def _sdk_error_message(exc: Exception, task: str) -> str:
    """User-facing text for a non-credential SDK error (the caller already imported the SDK)."""
    import anthropic

    if isinstance(exc, anthropic.RateLimitError):
        return f"Anthropic 요청 한도를 초과했습니다. 잠시 후 다시 시도하세요 (task={task})"
    if isinstance(exc, anthropic.APIStatusError):
        return _status_message("Anthropic", exc.status_code, exc.message, task)
    if isinstance(exc, anthropic.APIConnectionError):
        return f"Anthropic API에 연결하지 못했습니다 (task={task})"
    return f"Anthropic SDK 오류입니다 ({type(exc).__name__}, task={task})"


def _status_message(vendor: str, status_code: int, detail: str, task: str) -> str:
    if status_code >= 500:
        return f"{vendor} 서버 오류입니다 (HTTP {status_code}, task={task})"
    return f"{vendor} 요청이 거부되었습니다 (HTTP {status_code}, task={task}): {detail}"


def _log_response(response: Any, *, task: str, requested_model: str) -> None:
    served = getattr(response, "model", None)
    _log.info(
        "anthropic call task=%s model=%s request_id=%s stop_reason=%s",
        task,
        served,
        getattr(response, "_request_id", None),
        getattr(response, "stop_reason", None),
    )
    if isinstance(served, str) and served and served != requested_model:
        _log.info("anthropic fallback served task=%s requested=%s served=%s", task, requested_model, served)
