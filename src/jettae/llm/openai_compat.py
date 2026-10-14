"""OpenAI-compatible adapter for vLLM / on-prem gateways / other model families.

Why it exists: some customers cannot send runbooks to an external API, and the evaluation
harness re-phrases inputs with a different model family. Any server that speaks the
``/v1/chat/completions`` protocol with ``response_format: json_schema`` (vLLM guided decoding,
OpenAI, most gateways) works.

The prompt is identical to the Anthropic adapter (same safety preamble, same nonce-bounded
documents, same strict schema), so the two are interchangeable behind
:class:`~jettae.llm.base.LLMClient`. ``openai`` is imported lazily (optional extra, P4).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Final, Protocol, TypeVar

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

#: placeholder key for servers without auth (vLLM's default); the SDK refuses an empty key
EMPTY_API_KEY: Final = "EMPTY"
_SCHEMA_NAME_INVALID: Final = re.compile(r"[^A-Za-z0-9_-]")
_SCHEMA_NAME_MAX: Final = 64
_KEY_HINT: Final = "API 키 확인 필요"

_log = logging.getLogger(__name__)


class _CompletionsAPI(Protocol):
    async def create(self, *args: Any, **kwargs: Any) -> Any: ...


class _ChatAPI(Protocol):
    @property
    def completions(self) -> _CompletionsAPI: ...


class OpenAIClientLike(Protocol):
    """The slice of ``openai.AsyncOpenAI`` this adapter uses (lets tests inject fakes)."""

    @property
    def chat(self) -> _ChatAPI: ...


@dataclass
class _Totals:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0

    def add(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        self.input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        details = getattr(usage, "prompt_tokens_details", None)
        self.cache_read_tokens += int(getattr(details, "cached_tokens", 0) or 0)


class OpenAICompatLLM:
    """:class:`~jettae.llm.base.LLMClient` for OpenAI-protocol chat completion servers."""

    provider: str = "openai"

    def __init__(
        self,
        base_url: str | None,
        model: str,
        api_key: str | None = None,
        *,
        client: OpenAIClientLike | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self._unavailable: str | None = None
        self._client: OpenAIClientLike | None = client
        if client is None:
            self._client, self._unavailable = _default_client(
                base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries
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
        response_format = json_schema_response_format(schema, strict_schema)
        totals = _Totals()
        started = time.perf_counter()

        async def send(history: list[Message]) -> Reply:
            response = await self._create(
                client,
                messages=[{"role": "system", "content": system_text}, *history],
                response_format=response_format,
                max_tokens=max_tokens,
                task=task,
            )
            totals.add(response)
            text = _response_text(response, task=task, max_tokens=max_tokens)
            return Reply(text=text, assistant_turn={"role": "assistant", "content": text})

        data = await complete_with_repair(
            send, user_content=render_user_content(documents, context), schema=schema, task=task
        )
        usage = LLMUsage(
            provider=self.provider,
            model=self.model,
            task=task,
            input_tokens=totals.input_tokens,
            output_tokens=totals.output_tokens,
            cache_read_tokens=totals.cache_read_tokens,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
        return LLMResult(data=data, usage=usage)

    def _require_client(self, task: str) -> OpenAIClientLike:
        if self._client is None:
            raise LLMDisabledError(f"{self._unavailable or _KEY_HINT} (task={task})")
        return self._client

    async def _create(
        self,
        client: OpenAIClientLike,
        *,
        messages: list[Message],
        response_format: dict[str, Any],
        max_tokens: int,
        task: str,
    ) -> Any:
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format=response_format,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            mapped = _map_error(exc, task)
            if mapped is None:
                raise
            _log.warning("openai-compatible call failed task=%s error=%s", task, type(exc).__name__)
            raise mapped from exc
        _log.info(
            "openai-compatible call task=%s model=%s request_id=%s",
            task,
            getattr(response, "model", self.model),
            getattr(response, "_request_id", None),
        )
        return response


def json_schema_response_format(schema: type[BaseModel], strict_schema: JsonSchema) -> dict[str, Any]:
    """``response_format`` payload; the name must match ``^[A-Za-z0-9_-]{1,64}$``."""
    name = _SCHEMA_NAME_INVALID.sub("_", schema.__name__)[:_SCHEMA_NAME_MAX] or "output"
    return {"type": "json_schema", "json_schema": {"name": name, "schema": strict_schema, "strict": True}}


def _default_client(
    *, base_url: str | None, api_key: str | None, timeout: float, max_retries: int
) -> tuple[OpenAIClientLike | None, str | None]:
    try:
        import openai
    except ImportError:
        return None, "openai 패키지가 설치되어 있지 않습니다 (pip install 'jettae-ops[openai]')"
    client = openai.AsyncOpenAI(
        base_url=base_url, api_key=api_key or EMPTY_API_KEY, timeout=timeout, max_retries=max_retries
    )
    return client, None


def _response_text(response: Any, *, task: str, max_tokens: int) -> str:
    choices = getattr(response, "choices", None) or ()
    if not choices:
        raise LLMOutputError(f"응답에 선택지(choices)가 없습니다 (task={task})")
    choice = choices[0]
    message = choice.message
    refusal = getattr(message, "refusal", None)
    if refusal:
        raise LLMRefusalError(f"모델이 요청을 거절했습니다 (task={task}): {refusal}")
    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason == "content_filter":
        raise LLMRefusalError(f"콘텐츠 필터가 응답을 차단했습니다 (task={task})")
    if finish_reason == "length":
        raise LLMOutputError(
            f"응답이 max_tokens={max_tokens}에서 잘렸습니다 (task={task}). "
            "문서를 더 작게 나누거나 max_tokens를 늘리세요"
        )
    content = getattr(message, "content", None)
    return content if isinstance(content, str) else ""


def _map_error(exc: Exception, task: str) -> LLMError | None:
    """Most specific first; ``None`` means "not an SDK error, re-raise unchanged".

    The final ``OpenAIError`` branch keeps response-validation and other SDK errors typed, so a
    misbehaving gateway triggers the caller's :class:`LLMError` fallback instead of a crash.
    """
    try:
        import openai
    except ImportError:
        return None
    if not isinstance(exc, openai.OpenAIError):
        return None
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return LLMDisabledError(f"{_KEY_HINT} (task={task})")
    return LLMError(_sdk_error_message(exc, task))


def _sdk_error_message(exc: Exception, task: str) -> str:
    """User-facing text for a non-credential SDK error (the caller already imported the SDK)."""
    import openai

    if isinstance(exc, openai.RateLimitError):
        return f"LLM 요청 한도를 초과했습니다. 잠시 후 다시 시도하세요 (task={task})"
    if isinstance(exc, openai.APIStatusError):
        return _status_message(exc.status_code, exc.message, task)
    if isinstance(exc, openai.APIConnectionError):
        return f"LLM 서버에 연결하지 못했습니다 (task={task})"
    return f"LLM SDK 오류입니다 ({type(exc).__name__}, task={task})"


def _status_message(status_code: int, detail: str, task: str) -> str:
    if status_code >= 500:
        return f"LLM 서버 오류입니다 (HTTP {status_code}, task={task})"
    return f"LLM 요청이 거부되었습니다 (HTTP {status_code}, task={task}): {detail}"
