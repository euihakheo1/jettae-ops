"""Paid-call plumbing for the real-data studies: spend tracking, the budget gate, the providers.

Why it is shaped this way:

* **Spend is metered at the SDK boundary.** :class:`RecordingAnthropic` wraps the Anthropic SDK
  client that :class:`~jettae.llm.anthropic_client.AnthropicLLM` calls, so every response —
  the repair turn included — is priced from its own ``usage`` (input, cache writes, cache reads,
  output with thinking). With server-side fallbacks the top-level ``usage`` covers only the
  attempt that produced the message, so ``usage.iterations`` (one entry per attempt, each with
  its model — a declined hop is billed too) is priced instead whenever present.
  :class:`OpenAIResponsesLLM` records the same way. A call cancelled in flight returns no usage
  but may still be billed, so its reservation is booked as spent (conservative).
* **The budget stops a run, it never degrades one.** :class:`BudgetedLLM` reserves the expected
  cost of a call before sending it and refuses (:class:`BudgetExhausted`) once spend plus
  reservations would pass ``--max-usd`` — a *cumulative* cap of the output folder: what earlier
  runs spent there (``prior_usd``, from the spend ledger of :mod:`~jettae.evalkit.realdata.resume`)
  counts against it — or the optional per-invocation cap ``--max-usd-run``. The exception says
  which cap stopped the run (``scope``). ``BudgetExhausted`` is deliberately *not* an
  :class:`~jettae.llm.base.LLMError`: the product pipeline turns an ``LLMError`` into the
  rule-parser fallback, which would quietly record a degraded item as a finished one. It
  propagates instead, the study stops and writes what it has, and a later run resumes.
  :func:`gather_jobs` runs a study's own concurrent calls so that nothing keeps sending
  requests behind a failure (``asyncio.gather`` would leave the siblings running).
* **Back-translation is Korean-only.** The product renders every response edge back into a Korean
  sentence and asks the model whether it matches the source quote; with an English quote that
  question compares two languages. :class:`SkipNonKoreanBacktranslation` answers such requests
  with :class:`BacktranslationSkippedError` (an ``LLMError``, which the product's judge treats as
  "no verdict"), so the deterministic slot check alone decides — exactly the product's LLM-off
  behaviour — and Korean documents pass through untouched.
* **Retries are the SDKs' own.** Both official SDKs retry 408/409/429/5xx and connection errors
  with exponential back-off (honouring ``retry-after``); the studies raise ``max_retries``.
* **No sampling parameters.** Claude Opus 5.5 rejects ``temperature``; OpenAI reasoning models
  reject it too. None is sent anywhere.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal, Protocol, TypeVar, cast

from pydantic import BaseModel, ConfigDict

from jettae.evalkit.realdata.pricing import (
    CallKind,
    ModelRef,
    call_usd,
    supports_server_fallbacks,
    thinking_allowance,
    visible_output_tokens,
)
from jettae.llm.anthropic_client import AnthropicClientLike, AnthropicLLM, Effort
from jettae.llm.base import (
    LLMClient,
    LLMDisabledError,
    LLMError,
    LLMOutputError,
    LLMRefusalError,
    LLMResult,
    LLMUsage,
    UntrustedDocument,
)
from jettae.llm.prompting import Message, Reply, build_system, complete_with_repair, render_user_content
from jettae.llm.schema import to_strict_json_schema

__all__ = [
    "BACKTRANSLATE_TASK",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_TIMEOUT_SECONDS",
    "UNMETERED_LINE",
    "BacktranslationSkippedError",
    "BudgetExhausted",
    "BudgetScope",
    "BudgetedLLM",
    "OpenAIResponsesLLM",
    "RecordingAnthropic",
    "SkipNonKoreanBacktranslation",
    "SpendLine",
    "SpendReport",
    "SpendTracker",
    "claude_llm",
    "expected_call_usd",
    "gather_jobs",
    "new_anthropic_sdk",
    "new_openai_sdk",
]

_T = TypeVar("_T", bound=BaseModel)
_R = TypeVar("_R")

#: ``task`` of the product's back-translation judge (``jettae.extract.backtranslate``).
BACKTRANSLATE_TASK: Final = "backtranslate"
DEFAULT_MAX_RETRIES: Final = 6
#: The SDKs' own default. A shorter timeout can cut off a long (16K-token, thinking) answer; the
#: SDK then retries, and a timed-out attempt may still be billed without returning usage.
DEFAULT_TIMEOUT_SECONDS: Final = 600.0
_SCHEMA_NAME_INVALID: Final = re.compile(r"[^A-Za-z0-9_-]")
#: Input-size heuristic of the pre-send reservation. ``count_tokens`` measured ≈ 1.7 characters
#: per token on the real study-1 requests (English runbooks + Korean instructions) and ≈ 1.07 on
#: Korean runbook text, so 1.0 errs on the side of reserving too much.
_RESERVATION_CHARS_PER_TOKEN: Final = 1.0
#: Spend line of calls cancelled in flight (no usage came back; the reservation is booked).
UNMETERED_LINE: Final = "(취소된 호출: 예약액으로 계상)"

_log = logging.getLogger(__name__)


#: Which cap stopped a run: ``total`` = ``--max-usd`` (cumulative over the output folder),
#: ``run`` = ``--max-usd-run`` (this invocation only).
BudgetScope = Literal["total", "run"]


class BudgetExhausted(RuntimeError):  # noqa: N818 - a stop signal, not an error condition
    """Spend (plus calls in flight) reached a cap: stop, keep what is done, resume later."""

    def __init__(self, message: str, *, scope: BudgetScope = "total") -> None:
        super().__init__(message)
        self.scope: BudgetScope = scope


class _SkippedError(Exception):
    """A job of :func:`gather_jobs` not started because a sibling had already failed."""


async def gather_jobs(jobs: Sequence[Callable[[], Awaitable[_R]]], *, concurrency: int) -> list[_R]:
    """Run ``jobs`` (at most ``concurrency`` at a time) and return their results in order.

    Unlike ``asyncio.gather``, nothing is left running behind a failure: once a job fails, jobs
    that have not started are skipped (no new paid request), jobs already in flight finish (so
    their usage is metered rather than lost to a cancellation), and then one error is raised —
    :class:`BudgetExhausted` first (the run must stop), otherwise the first failure.
    """
    gate = asyncio.Semaphore(max(1, concurrency))
    failures: list[BaseException] = []

    async def run(job: Callable[[], Awaitable[_R]]) -> _R:
        async with gate:
            if failures:
                raise _SkippedError
            try:
                return await job()
            except BaseException as exc:
                failures.append(exc)
                raise

    outcomes = await asyncio.gather(*(run(job) for job in jobs), return_exceptions=True)
    if failures:
        raise next((e for e in failures if isinstance(e, BudgetExhausted)), failures[0])
    return [cast(_R, outcome) for outcome in outcomes]


# --------------------------------------------------------------------------- spend


class SpendLine(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    calls: int
    input_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    output_tokens: int
    usd: float


class SpendReport(BaseModel):
    """Spend of one run. ``max_usd`` is the cumulative cap of the output folder, ``prior_usd`` what
    earlier runs had spent there when this one started, ``max_usd_run`` this run's own cap."""

    model_config = ConfigDict(frozen=True)

    max_usd: float | None
    spent_usd: float
    by_model: tuple[SpendLine, ...]
    max_usd_run: float | None = None
    prior_usd: float = 0.0


@dataclass
class _Line:
    calls: int = 0
    input_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0


@dataclass
class SpendTracker:
    """Actual spend of one study run, plus reservations of calls in flight.

    ``max_usd`` caps ``prior_usd`` (spent in the output folder by earlier runs) plus this run;
    ``max_usd_run`` caps this run alone. ``on_change`` is called after every change of spend or
    reservations (the spend ledger persists them), outside the lock.
    """

    max_usd: float | None = None
    max_usd_run: float | None = None
    prior_usd: float = 0.0
    on_change: Callable[[], None] | None = field(default=None, repr=False)
    _lines: dict[str, _Line] = field(default_factory=dict)
    _reserved: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def spent_usd(self) -> float:
        """Spent by this run."""
        with self._lock:
            return sum(line.usd for line in self._lines.values())

    @property
    def reserved_usd(self) -> float:
        """Reserved by calls in flight."""
        with self._lock:
            return self._reserved

    @property
    def used(self) -> bool:
        """Has anything been booked or reserved yet?"""
        with self._lock:
            return bool(self._lines) or self._reserved > 0

    def _changed(self) -> None:
        if self.on_change is not None:
            self.on_change()

    def reserve(self, usd: float, *, what: str) -> None:
        """Book ``usd`` for a call about to start, or raise :class:`BudgetExhausted`."""
        with self._lock:
            spent = sum(line.usd for line in self._lines.values())
            if self.max_usd is not None and self.prior_usd + spent + self._reserved + usd > self.max_usd:
                raise BudgetExhausted(
                    f"누적 예산 상한 --max-usd ${self.max_usd:.2f}(이 결과 폴더의 모든 실행 합계)에 닿아 "
                    f"멈춥니다 (이전 실행 ${self.prior_usd:.4f} + 이번 실행 ${spent:.4f}, "
                    f"진행 중 예약 ${self._reserved:.4f}, 다음 호출 {what} 예상 ${usd:.4f})",
                    scope="total",
                )
            if self.max_usd_run is not None and spent + self._reserved + usd > self.max_usd_run:
                raise BudgetExhausted(
                    f"이번 실행 예산 상한 --max-usd-run ${self.max_usd_run:.2f}에 닿아 멈춥니다 "
                    f"(이번 실행 ${spent:.4f}, 진행 중 예약 ${self._reserved:.4f}, "
                    f"다음 호출 {what} 예상 ${usd:.4f})",
                    scope="run",
                )
            self._reserved += usd
        self._changed()

    def release(self, usd: float) -> None:
        with self._lock:
            self._reserved = max(0.0, self._reserved - usd)
        self._changed()

    def record_unmetered(self, usd: float) -> None:
        """Book a call that was cancelled in flight at its reservation (it returned no usage but
        may have been billed), under :data:`UNMETERED_LINE` — the cap must never under-count."""
        with self._lock:
            line = self._lines.setdefault(UNMETERED_LINE, _Line())
            line.calls += 1
            line.usd += usd
        self._changed()

    def record(
        self,
        model: str,
        *,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float:
        """Add one response's usage (priced by the model that served it); returns its USD."""
        usd = call_usd(
            model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )
        with self._lock:
            line = self._lines.setdefault(model, _Line())
            line.calls += 1
            line.input_tokens += input_tokens
            line.output_tokens += output_tokens
            line.cache_read_tokens += cache_read_tokens
            line.cache_write_tokens += cache_write_tokens
            line.usd += usd
        self._changed()
        return usd

    def report(self) -> SpendReport:
        with self._lock:
            lines = tuple(
                SpendLine(
                    model=model,
                    calls=line.calls,
                    input_tokens=line.input_tokens,
                    cache_write_tokens=line.cache_write_tokens,
                    cache_read_tokens=line.cache_read_tokens,
                    output_tokens=line.output_tokens,
                    usd=round(line.usd, 6),
                )
                for model, line in sorted(self._lines.items())
            )
        return SpendReport(
            max_usd=self.max_usd,
            spent_usd=round(sum(x.usd for x in lines), 6),
            by_model=lines,
            max_usd_run=self.max_usd_run,
            prior_usd=round(self.prior_usd, 6),
        )


def expected_call_usd(
    ref: ModelRef,
    effort: Effort,
    kind: CallKind,
    *,
    input_chars: int,
    doc_chars: int = 0,
    items: int = 0,
    korean: bool = False,
    max_tokens: int = 16_000,
) -> float:
    """What one call is expected to cost — the reservation the budget gate books before sending."""
    input_tokens = math.ceil(input_chars / _RESERVATION_CHARS_PER_TOKEN)
    visible = visible_output_tokens(kind, chars=doc_chars, items=items, korean=korean)
    output = min(max_tokens, visible + thinking_allowance(ref.provider, ref.model, effort))
    return call_usd(ref.model, input_tokens=input_tokens, output_tokens=output)


#: ``(task, documents, instructions + context chars, max_tokens) -> expected USD``
Estimator = Callable[[str, tuple[UntrustedDocument, ...], int, int], float]


class BudgetedLLM:
    """:class:`LLMClient` that books each call's expected cost before sending it (see module doc)."""

    def __init__(self, inner: LLMClient, tracker: SpendTracker, estimate: Estimator) -> None:
        self.inner = inner
        self.tracker = tracker
        self.estimate = estimate
        self.provider = inner.provider
        self.model = inner.model

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
        expected = self.estimate(task, documents, len(instructions) + len(context), max_tokens)
        self.tracker.reserve(expected, what=task)
        try:
            return await self.inner.structured(
                task=task,
                instructions=instructions,
                schema=schema,
                documents=documents,
                context=context,
                max_tokens=max_tokens,
            )
        except asyncio.CancelledError:
            # e.g. the product extractor's TaskGroup cancels sibling chunks when one fails
            self.tracker.record_unmetered(expected)
            raise
        finally:
            self.tracker.release(expected)


class BacktranslationSkippedError(LLMError):
    """The source is not Korean, so the Korean-only back-translation judge is not asked."""


class SkipNonKoreanBacktranslation:
    """Pass every request through, except back-translation of a non-Korean source (module doc)."""

    def __init__(self, inner: LLMClient, non_korean_doc_ids: frozenset[str]) -> None:
        self.inner = inner
        self.non_korean_doc_ids = non_korean_doc_ids
        self.provider = inner.provider
        self.model = inner.model
        self.skipped = 0

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
        if task == BACKTRANSLATE_TASK and any(d.doc_id in self.non_korean_doc_ids for d in documents):
            self.skipped += 1
            raise BacktranslationSkippedError(
                "역번역 LLM 대조 생략: 한국어가 아닌 원문 (backtranslation skipped)"
            )
        return await self.inner.structured(
            task=task,
            instructions=instructions,
            schema=schema,
            documents=documents,
            context=context,
            max_tokens=max_tokens,
        )


# --------------------------------------------------------------------------- Anthropic


class _RecordingMessages:
    def __init__(self, inner: Any, tracker: SpendTracker) -> None:
        self._inner = inner
        self._tracker = tracker

    async def create(self, **kwargs: Any) -> Any:
        response = await self._inner.create(**kwargs)
        usage = getattr(response, "usage", None)
        served = getattr(response, "model", None)
        served_model = served if isinstance(served, str) and served else str(kwargs.get("model", ""))
        # One entry per attempt when the API itemises them (server-side fallback: the declined hop
        # is billed at its own model's rate; top-level usage covers only the serving attempt).
        attempts = list(getattr(usage, "iterations", None) or ())
        if not attempts:
            self._record(usage, served_model)
        for attempt in attempts:
            model = getattr(attempt, "model", None)
            # an attempt without a model is priced like the served model (the dearer one on a
            # fallback to an older Opus, the requested one otherwise)
            self._record(attempt, model if isinstance(model, str) and model else served_model)
        return response

    def _record(self, usage: Any, model: str) -> None:
        self._tracker.record(
            model,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            cache_read_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            cache_write_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        )


class _RecordingBeta:
    def __init__(self, inner: Any, tracker: SpendTracker) -> None:
        self._messages = _RecordingMessages(inner.messages, tracker)

    @property
    def messages(self) -> _RecordingMessages:
        return self._messages


class RecordingAnthropic:
    """An Anthropic SDK client (``.beta.messages.create``) whose every response is priced."""

    def __init__(self, sdk: Any, tracker: SpendTracker) -> None:
        self._beta = _RecordingBeta(sdk.beta, tracker)

    @property
    def beta(self) -> _RecordingBeta:
        return self._beta


def new_anthropic_sdk(
    api_key: str, *, timeout: float = DEFAULT_TIMEOUT_SECONDS, max_retries: int = DEFAULT_MAX_RETRIES
) -> Any:
    """``anthropic.AsyncAnthropic`` with an explicit key (never read back from the environment)."""
    import anthropic

    return anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout, max_retries=max_retries)


def claude_llm(model: str, effort: Effort, sdk: Any, tracker: SpendTracker) -> AnthropicLLM:
    """The product's Anthropic adapter (structured outputs, effort, server-side fallbacks where the
    model has them) over a recording SDK client."""
    client: AnthropicClientLike = RecordingAnthropic(sdk, tracker)
    return AnthropicLLM(model, effort, client=client, use_fallbacks=supports_server_fallbacks(model))


# --------------------------------------------------------------------------- OpenAI Responses


class _ResponsesAPI(Protocol):
    async def create(self, *args: Any, **kwargs: Any) -> Any: ...


class OpenAIResponsesClientLike(Protocol):
    """The slice of ``openai.AsyncOpenAI`` this adapter uses (lets tests inject fakes)."""

    @property
    def responses(self) -> _ResponsesAPI: ...


def new_openai_sdk(
    api_key: str, *, timeout: float = DEFAULT_TIMEOUT_SECONDS, max_retries: int = DEFAULT_MAX_RETRIES
) -> Any:
    """``openai.AsyncOpenAI`` with an explicit key."""
    import openai

    return openai.AsyncOpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)


class OpenAIResponsesLLM:
    """:class:`LLMClient` over the official OpenAI SDK's ``responses.create`` with a strict JSON
    schema (``text.format``). Same safety preamble, nonce-bounded documents and one repair turn
    as the product adapters (:mod:`jettae.llm.prompting`). ``store=False``: nothing is kept
    server-side for later retrieval."""

    provider: str = "openai"

    def __init__(
        self, model: str, *, client: OpenAIResponsesClientLike, tracker: SpendTracker | None = None
    ) -> None:
        self.model = model
        self._client = client
        self._tracker = tracker

    def request_kwargs(
        self, system_text: str, history: list[Message], text_format: dict[str, Any], max_tokens: int
    ) -> dict[str, Any]:
        """Keyword arguments for ``client.responses.create`` (public for the dry run and tests)."""
        return {
            "model": self.model,
            "instructions": system_text,
            "input": list(history),
            "text": {"format": text_format},
            "max_output_tokens": max_tokens,
            "store": False,
        }

    @staticmethod
    def text_format(schema: type[BaseModel]) -> dict[str, Any]:
        name = _SCHEMA_NAME_INVALID.sub("_", schema.__name__)[:64] or "output"
        return {"type": "json_schema", "name": name, "schema": to_strict_json_schema(schema), "strict": True}

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
        system_text = build_system(instructions)
        text_format = self.text_format(schema)
        totals = {"input": 0, "output": 0, "cached": 0}
        started = time.perf_counter()

        async def send(history: list[Message]) -> Reply:
            kwargs = self.request_kwargs(system_text, history, text_format, max_tokens)
            try:
                response = await self._client.responses.create(**kwargs)
            except Exception as exc:
                mapped = _map_openai_error(exc, task)
                if mapped is None:
                    raise
                _log.warning("openai responses call failed task=%s error=%s", task, type(exc).__name__)
                raise mapped from exc
            self._record(response, totals)
            text = _responses_text(response, task=task, max_tokens=max_tokens)
            return Reply(text=text, assistant_turn={"role": "assistant", "content": text})

        data = await complete_with_repair(
            send, user_content=render_user_content(documents, context), schema=schema, task=task
        )
        usage = LLMUsage(
            provider=self.provider,
            model=self.model,
            task=task,
            input_tokens=totals["input"],
            output_tokens=totals["output"],
            cache_read_tokens=totals["cached"],
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
        return LLMResult(data=data, usage=usage)

    def _record(self, response: Any, totals: dict[str, int]) -> None:
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        details = getattr(usage, "input_tokens_details", None)
        cached = int(getattr(details, "cached_tokens", 0) or 0)
        totals["input"] += input_tokens
        totals["output"] += output_tokens
        totals["cached"] += cached
        if self._tracker is not None:
            # cached input is billed at the full input price here (conservative, see pricing)
            self._tracker.record(self.model, input_tokens=input_tokens, output_tokens=output_tokens)
        _log.info(
            "openai responses call model=%s request_id=%s status=%s",
            getattr(response, "model", self.model),
            getattr(response, "_request_id", None),
            getattr(response, "status", None),
        )


def _responses_text(response: Any, *, task: str, max_tokens: int) -> str:
    """Branch on ``status`` and refusals before reading the text (an unfinished reply is no answer)."""
    status = getattr(response, "status", None)
    if status == "incomplete":
        reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
        if reason == "content_filter":
            raise LLMRefusalError(f"콘텐츠 필터가 응답을 차단했습니다 (task={task})")
        raise LLMOutputError(
            f"응답이 max_output_tokens={max_tokens}에서 잘렸습니다 (task={task}, 사유={reason})"
        )
    if status not in (None, "completed"):
        raise LLMOutputError(f"응답이 정상적으로 끝나지 않았습니다 (task={task}, status={status})")
    for item in getattr(response, "output", None) or ():
        for part in getattr(item, "content", None) or ():
            if getattr(part, "type", None) == "refusal":
                raise LLMRefusalError(f"모델이 요청을 거절했습니다 (task={task})")
    text = getattr(response, "output_text", None)
    if not isinstance(text, str) or not text.strip():
        raise LLMOutputError(f"응답에 텍스트가 없습니다 (task={task})")
    return text


def _map_openai_error(exc: Exception, task: str) -> LLMError | None:  # noqa: PLR0911 - one per SDK error
    """Most specific first; ``None`` = not an SDK error (re-raise unchanged)."""
    try:
        import openai
    except ImportError:
        return None
    if not isinstance(exc, openai.OpenAIError):
        return None
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return LLMDisabledError(f"API 키 확인 필요: OpenAI 인증 실패 (task={task})")
    if isinstance(exc, openai.RateLimitError):
        return LLMError(f"OpenAI 요청 한도를 초과했습니다(재시도 후) (task={task})")
    if isinstance(exc, openai.APIStatusError):
        return LLMError(f"OpenAI 요청 실패 (HTTP {exc.status_code}, task={task})")
    if isinstance(exc, openai.APIConnectionError):
        return LLMError(f"OpenAI API에 연결하지 못했습니다 (task={task})")
    return LLMError(f"OpenAI SDK 오류입니다 ({type(exc).__name__}, task={task})")
