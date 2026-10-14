"""In-process fake LLM for tests: a plain function decides each response.

It mirrors the real adapters' contract where it matters for callers: the request schema must be
expressible as a strict schema (so a test passing here cannot fail later against a live
provider), dict outputs go through the same null-pruning validation, and invalid output raises
:class:`LLMOutputError`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import pydantic
from pydantic import BaseModel

from jettae.llm.base import LLMOutputError, LLMResult, LLMUsage, UntrustedDocument
from jettae.llm.schema import to_strict_json_schema, validate_output

_T = TypeVar("_T", bound=BaseModel)

FakeOutcome = BaseModel | dict[str, Any] | Exception
#: ``handler(task, schema, documents, context)`` → model instance, plain dict, or an exception to raise
FakeHandler = Callable[[str, type[BaseModel], tuple[UntrustedDocument, ...], str], FakeOutcome]


@dataclass(frozen=True)
class FakeCall:
    """One recorded request, for assertions in tests."""

    task: str
    instructions: str
    schema: type[BaseModel]
    documents: tuple[UntrustedDocument, ...]
    context: str
    max_tokens: int


class FakeLLM:
    """:class:`~jettae.llm.base.LLMClient` whose responses come from ``handler``."""

    def __init__(self, handler: FakeHandler, provider: str = "fake", model: str = "fake-1") -> None:
        self.handler = handler
        self.provider = provider
        self.model = model
        self.calls: list[FakeCall] = []

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
        self.calls.append(FakeCall(task, instructions, schema, documents, context, max_tokens))
        to_strict_json_schema(schema)  # fail like a live adapter would on an unsupported schema
        outcome = self.handler(task, schema, documents, context)
        if isinstance(outcome, Exception):
            raise outcome
        data = _coerce(schema, outcome, task)
        return LLMResult(data=data, usage=LLMUsage(provider=self.provider, model=self.model, task=task))


def _coerce(schema: type[_T], outcome: BaseModel | dict[str, Any], task: str) -> _T:
    if isinstance(outcome, schema):
        return outcome
    if isinstance(outcome, BaseModel):
        raise LLMOutputError(
            f"가짜 LLM 응답 타입이 스키마와 다릅니다 (task={task}): "
            f"{type(outcome).__name__} ≠ {schema.__name__}"
        )
    try:
        return validate_output(schema, outcome)
    except pydantic.ValidationError as exc:
        raise LLMOutputError(f"가짜 LLM 응답이 스키마 검증을 통과하지 못했습니다 (task={task})") from exc
