"""FakeLLM — the test double other modules use instead of a live model (jettae.llm.fake)."""

from __future__ import annotations

from collections.abc import Coroutine
from typing import Any, TypeVar

import pytest
from pydantic import BaseModel

from jettae.llm.base import LLMClient, LLMDisabledError, LLMOutputError, LLMResult, UntrustedDocument
from jettae.llm.fake import FakeCall, FakeLLM
from jettae.llm.schema import StrictSchemaError

R = TypeVar("R")

DOCS = (UntrustedDocument(doc_id="doc-1", title="런북", text="담당: 김대리"),)


def run(coro: Coroutine[Any, Any, R]) -> R:
    """Drive a coroutine that never suspends (FakeLLM is synchronous underneath) to completion."""
    try:
        coro.send(None)
    except StopIteration as done:
        result: R = done.value
        return result
    coro.close()
    raise AssertionError("coroutine suspended; this test needs a real event loop")


class Person(BaseModel):
    name: str
    title: str | None = None
    aliases: tuple[str, ...] = ()


class Other(BaseModel):
    value: int


class Mapping(BaseModel):
    table: dict[str, int]


def ask(llm: FakeLLM, schema: type[Any] = Person, task: str = "extract") -> LLMResult[Any]:
    return run(
        llm.structured(
            task=task, instructions="지시", schema=schema, documents=DOCS, context="ctx", max_tokens=99
        )
    )


def test_model_instances_are_returned_as_is() -> None:
    person = Person(name="김대리")

    result = ask(FakeLLM(lambda *_: person))

    assert result.data is person
    assert (result.usage.provider, result.usage.model, result.usage.task) == ("fake", "fake-1", "extract")
    assert result.usage.input_tokens == result.usage.output_tokens == 0


def test_dicts_are_validated_with_null_as_absent() -> None:
    result = ask(FakeLLM(lambda *_: {"name": "김대리", "title": None, "aliases": None}))

    assert result.data == Person(name="김대리")


def test_handler_sees_the_request_and_calls_are_recorded() -> None:
    seen: list[tuple[str, type[BaseModel], tuple[UntrustedDocument, ...], str]] = []

    def handler(
        task: str, schema: type[BaseModel], documents: tuple[UntrustedDocument, ...], context: str
    ) -> Person:
        seen.append((task, schema, documents, context))
        return Person(name=documents[0].text.split(": ")[1])

    llm = FakeLLM(handler, provider="anthropic", model="claude-opus-5-5")
    result = ask(llm, task="owners")

    assert result.data.name == "김대리"
    assert seen == [("owners", Person, DOCS, "ctx")]
    assert llm.calls == [FakeCall("owners", "지시", Person, DOCS, "ctx", 99)]
    assert (result.usage.provider, result.usage.model) == ("anthropic", "claude-opus-5-5")


def test_returned_exceptions_are_raised() -> None:
    llm = FakeLLM(lambda *_: LLMDisabledError("off"))

    with pytest.raises(LLMDisabledError, match="off"):
        ask(llm)
    assert len(llm.calls) == 1


def test_raising_handlers_propagate() -> None:
    def handler(*_: Any) -> Person:
        raise KeyError("boom")

    with pytest.raises(KeyError, match="boom"):
        ask(FakeLLM(handler))


def test_wrong_model_type_is_an_output_error() -> None:
    with pytest.raises(LLMOutputError, match="Other ≠ Person"):
        ask(FakeLLM(lambda *_: Other(value=1)))


def test_invalid_dict_is_an_output_error() -> None:
    with pytest.raises(LLMOutputError, match="task=extract"):
        ask(FakeLLM(lambda *_: {"title": "대리"}))


def test_schemas_a_live_provider_cannot_take_are_rejected() -> None:
    llm = FakeLLM(lambda *_: {"table": {"a": 1}})

    with pytest.raises(StrictSchemaError):
        ask(llm, schema=Mapping)


def test_fake_satisfies_the_llm_port() -> None:
    assert isinstance(FakeLLM(lambda *_: Person(name="x")), LLMClient)
