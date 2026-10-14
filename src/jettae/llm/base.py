"""LLM port. Every generative step in JETTAE goes through :class:`LLMClient`.

Design rules (see docs/ARCHITECTURE.md §5):
* Structured output only — callers pass a Pydantic schema and get a validated object back.
* Untrusted document text is passed in ``documents`` (datamarked by the adapter), never mixed
  into ``instructions`` — the instruction/data separation behind our prompt-injection defense.
* Every call is metered (:class:`UsageMeter`) so each diagnosis can publish calls/tokens.
* ``NullLLM`` raises :class:`LLMDisabledError`; callers MUST fall back to deterministic code.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class LLMDisabledError(LLMError):
    """LLM mode is OFF (or no credentials). Use the deterministic fallback."""


class LLMRefusalError(LLMError):
    pass


class LLMOutputError(LLMError):
    """The model returned output that does not validate against the schema."""


@dataclass(frozen=True)
class LLMUsage:
    provider: str
    model: str
    task: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    latency_ms: float = 0.0
    cached_replay: bool = False


@dataclass(frozen=True)
class LLMResult(Generic[T]):
    data: T
    usage: LLMUsage


@dataclass(frozen=True)
class UntrustedDocument:
    """A piece of untrusted text (runbook chunk) handed to the model as data."""

    doc_id: str
    title: str
    text: str
    offset: int = 0  # offset of ``text`` inside the full document


@runtime_checkable
class LLMClient(Protocol):
    provider: str
    model: str

    async def structured(
        self,
        *,
        task: str,
        instructions: str,
        schema: type[T],
        documents: tuple[UntrustedDocument, ...] = (),
        context: str = "",
        max_tokens: int = 16000,
    ) -> LLMResult[T]:
        """Run one schema-constrained generation.

        ``task``          short stable name used for metering & cassette keys ("extract", "backtranslate" …)
        ``instructions``  trusted system-level instructions (ours)
        ``documents``     untrusted text, datamarked/spotlighted by the adapter
        ``context``       trusted structured context (e.g. roster names, node ids) as text
        """
        ...


@dataclass
class UsageMeter:
    """Thread-safe accumulator of LLM usage for one run."""

    records: list[LLMUsage] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, usage: LLMUsage) -> None:
        with self._lock:
            self.records.append(usage)

    @property
    def calls(self) -> int:
        return len(self.records)

    @property
    def input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.records)

    @property
    def output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.records)

    @property
    def cache_read_tokens(self) -> int:
        return sum(r.cache_read_tokens for r in self.records)

    def by_task(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.records:
            out[r.task] = out.get(r.task, 0) + 1
        return dict(sorted(out.items()))


class MeteredLLM:
    """Decorator that records usage of any :class:`LLMClient` into a :class:`UsageMeter`."""

    def __init__(self, inner: LLMClient, meter: UsageMeter) -> None:
        self.inner = inner
        self.meter = meter
        self.provider = inner.provider
        self.model = inner.model

    async def structured(
        self,
        *,
        task: str,
        instructions: str,
        schema: type[T],
        documents: tuple[UntrustedDocument, ...] = (),
        context: str = "",
        max_tokens: int = 16000,
    ) -> LLMResult[T]:
        result = await self.inner.structured(
            task=task,
            instructions=instructions,
            schema=schema,
            documents=documents,
            context=context,
            max_tokens=max_tokens,
        )
        self.meter.add(result.usage)
        return result


class NullLLM:
    """LLM OFF. Every call raises :class:`LLMDisabledError` (callers fall back)."""

    provider = "none"
    model = "none"

    async def structured(
        self,
        *,
        task: str,
        instructions: str,
        schema: type[T],
        documents: tuple[UntrustedDocument, ...] = (),
        context: str = "",
        max_tokens: int = 16000,
    ) -> LLMResult[T]:
        raise LLMDisabledError(f"LLM is disabled (task={task})")
