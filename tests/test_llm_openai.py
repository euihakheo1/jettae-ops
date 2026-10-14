"""OpenAI-compatible adapter with an injected fake SDK client — no network (jettae.llm.openai_compat)."""

from __future__ import annotations

import sys
from collections.abc import Coroutine
from types import SimpleNamespace
from typing import Any, TypeVar

import httpx2
import openai
import pytest
from pydantic import BaseModel, create_model

from jettae.llm.base import (
    LLMClient,
    LLMDisabledError,
    LLMError,
    LLMOutputError,
    LLMRefusalError,
    LLMResult,
    UntrustedDocument,
)
from jettae.llm.openai_compat import EMPTY_API_KEY, OpenAICompatLLM, json_schema_response_format
from jettae.llm.prompting import SAFETY_PREAMBLE, render_user_content
from jettae.llm.schema import to_strict_json_schema

DOCS = (UntrustedDocument(doc_id="doc-1", title="DB 런북", text="장애 시 롤백 스크립트 실행"),)
VALID = '{"action": "롤백 스크립트 실행", "quote": "롤백 스크립트 실행"}'
_REQUEST = httpx2.Request("POST", "http://127.0.0.1:9/v1/chat/completions")
R = TypeVar("R")


def run(coro: Coroutine[Any, Any, R]) -> R:
    """Drive a coroutine to completion without an event loop.

    Every fake here completes without suspending, so no loop (and no OS self-pipe socket) is
    needed; the tests stay deterministic and resource-free.
    """
    try:
        coro.send(None)
    except StopIteration as done:
        result: R = done.value
        return result
    coro.close()
    raise AssertionError("coroutine suspended; this test needs a real event loop")


class Mitigation(BaseModel):
    action: str
    quote: str


def completion(
    content: str | None,
    *,
    finish_reason: str = "stop",
    refusal: str | None = None,
    prompt_tokens: int = 80,
    completion_tokens: int = 12,
    cached: int | None = None,
    usage: bool = True,
) -> SimpleNamespace:
    message = SimpleNamespace(content=content, refusal=refusal, role="assistant")
    details = SimpleNamespace(cached_tokens=cached) if cached is not None else None
    return SimpleNamespace(
        choices=[SimpleNamespace(index=0, message=message, finish_reason=finish_reason)],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, prompt_tokens_details=details
        )
        if usage
        else None,
        model="served-model",
    )


class FakeCompletions:
    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        assert not args
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeOpenAI:
    def __init__(self, *outcomes: Any) -> None:
        self.completions = FakeCompletions(*outcomes)
        self.chat = SimpleNamespace(completions=self.completions)


def make_llm(client: FakeOpenAI) -> OpenAICompatLLM:
    return OpenAICompatLLM("http://127.0.0.1:9/v1", "qwen3-32b", client=client)


def run_structured(llm: OpenAICompatLLM, **overrides: Any) -> LLMResult[Mitigation]:
    request: dict[str, Any] = {
        "task": "mitigation",
        "instructions": "완화 조치를 추출하세요.",
        "schema": Mitigation,
        "documents": DOCS,
        "context": "",
    }
    return run(llm.structured(**(request | overrides)))


# --------------------------------------------------------------------------- happy path


def test_success_returns_validated_data_and_usage() -> None:
    llm = make_llm(FakeOpenAI(completion(VALID, cached=30)))

    result = run_structured(llm)

    assert result.data == Mitigation(action="롤백 스크립트 실행", quote="롤백 스크립트 실행")
    usage = result.usage
    assert (usage.provider, usage.model, usage.task) == ("openai", "qwen3-32b", "mitigation")
    assert (usage.input_tokens, usage.output_tokens, usage.cache_read_tokens) == (80, 12, 30)
    assert usage.latency_ms >= 0.0
    assert isinstance(llm, LLMClient)


def test_request_shape_is_strict_json_schema_with_system_and_user_turns() -> None:
    client = FakeOpenAI(completion(VALID))

    run_structured(make_llm(client), max_tokens=2048)

    (kwargs,) = client.completions.calls
    assert kwargs["model"] == "qwen3-32b"
    assert kwargs["max_tokens"] == 2048
    system, user = kwargs["messages"]
    assert system["role"] == "system"
    assert system["content"].startswith(SAFETY_PREAMBLE)
    assert system["content"].endswith("완화 조치를 추출하세요.")
    assert user == {"role": "user", "content": render_user_content(DOCS, "")}
    assert kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "Mitigation", "schema": to_strict_json_schema(Mitigation), "strict": True},
    }
    assert set(kwargs) == {"model", "messages", "response_format", "max_tokens"}


def test_schema_names_are_sanitised_for_the_api() -> None:
    weird = create_model("Weird Name!/v2", value=(int, ...))
    long = create_model("L" * 80, value=(int, ...))

    assert json_schema_response_format(weird, {})["json_schema"]["name"] == "Weird_Name__v2"
    assert len(json_schema_response_format(long, {})["json_schema"]["name"]) == 64


def test_missing_usage_counts_as_zero() -> None:
    result = run_structured(make_llm(FakeOpenAI(completion(VALID, usage=False))))

    assert (result.usage.input_tokens, result.usage.output_tokens, result.usage.cache_read_tokens) == (
        0,
        0,
        0,
    )


# --------------------------------------------------------------------------- finish reasons


def test_refusal_field_raises_refusal() -> None:
    client = FakeOpenAI(completion(None, refusal="I can't help with that"))

    with pytest.raises(LLMRefusalError, match="I can't help with that"):
        run_structured(make_llm(client))
    assert len(client.completions.calls) == 1


def test_content_filter_is_a_refusal() -> None:
    with pytest.raises(LLMRefusalError, match="콘텐츠 필터"):
        run_structured(make_llm(FakeOpenAI(completion(None, finish_reason="content_filter"))))


def test_length_is_an_output_error_without_repair() -> None:
    client = FakeOpenAI(completion('{"action": "롤', finish_reason="length"))

    with pytest.raises(LLMOutputError, match="max_tokens=16000"):
        run_structured(make_llm(client))
    assert len(client.completions.calls) == 1


def test_no_choices_is_an_output_error() -> None:
    reply = completion(VALID)
    reply.choices = []

    with pytest.raises(LLMOutputError, match="choices"):
        run_structured(make_llm(FakeOpenAI(reply)))


# --------------------------------------------------------------------------- repair


def test_invalid_json_is_repaired_once_with_append_only_history() -> None:
    client = FakeOpenAI(completion("```json\n{}\n```", prompt_tokens=50), completion(VALID, prompt_tokens=70))

    result = run_structured(make_llm(client))

    assert result.data.action == "롤백 스크립트 실행"
    initial, repair = client.completions.calls
    assert len(initial["messages"]) == 2
    assert repair["messages"][:2] == initial["messages"]
    assert repair["messages"][2] == {"role": "assistant", "content": "```json\n{}\n```"}
    assert repair["messages"][3]["role"] == "user"
    assert "JSON" in repair["messages"][3]["content"]
    assert result.usage.input_tokens == 120


def test_empty_content_goes_through_repair() -> None:
    client = FakeOpenAI(completion(None), completion(VALID))

    result = run_structured(make_llm(client))

    assert result.data.quote == "롤백 스크립트 실행"
    assert client.completions.calls[1]["messages"][2] == {"role": "assistant", "content": ""}


def test_second_invalid_reply_raises_output_error() -> None:
    client = FakeOpenAI(completion('{"action": 1}'), completion('{"action": 2}'))

    with pytest.raises(LLMOutputError, match="task=mitigation"):
        run_structured(make_llm(client))


# --------------------------------------------------------------------------- errors


def _status(cls: type[openai.APIStatusError], code: int) -> openai.APIStatusError:
    return cls("boom", response=httpx2.Response(code, request=_REQUEST), body=None)


@pytest.mark.parametrize(
    ("error", "expected", "fragment"),
    [
        (_status(openai.AuthenticationError, 401), LLMDisabledError, "API 키 확인 필요"),
        (_status(openai.PermissionDeniedError, 403), LLMDisabledError, "API 키 확인 필요"),
        (_status(openai.RateLimitError, 429), LLMError, "한도"),
        (_status(openai.InternalServerError, 503), LLMError, "서버 오류"),
        (_status(openai.BadRequestError, 400), LLMError, "HTTP 400"),
        (openai.APIConnectionError(request=_REQUEST), LLMError, "연결"),
        (openai.APITimeoutError(request=_REQUEST), LLMError, "연결"),
    ],
    ids=["auth", "permission", "rate", "5xx", "4xx", "conn", "timeout"],
)
def test_sdk_errors_are_mapped(error: Exception, expected: type[LLMError], fragment: str) -> None:
    with pytest.raises(expected, match=fragment) as caught:
        run_structured(make_llm(FakeOpenAI(error)))

    assert type(caught.value) is expected
    assert caught.value.__cause__ is error


def test_non_sdk_errors_propagate_unchanged() -> None:
    with pytest.raises(KeyError):
        run_structured(make_llm(FakeOpenAI(KeyError("bug"))))


# --------------------------------------------------------------------------- client construction


def test_default_client_targets_base_url_with_placeholder_key() -> None:
    llm = OpenAICompatLLM("http://127.0.0.1:9/v1", "local-model")

    client = llm._client
    assert isinstance(client, openai.AsyncOpenAI)
    assert client.api_key == EMPTY_API_KEY
    assert str(client.base_url).startswith("http://127.0.0.1:9/v1")


def test_explicit_api_key_is_used() -> None:
    client = OpenAICompatLLM("http://127.0.0.1:9/v1", "m", "sk-test")._client

    assert isinstance(client, openai.AsyncOpenAI)
    assert client.api_key == "sk-test"


def test_missing_package_disables_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", None)

    llm = OpenAICompatLLM("http://127.0.0.1:9/v1", "m")

    with pytest.raises(LLMDisabledError, match="openai 패키지"):
        run_structured(llm)


# --------------------------------------------------------------------------- review


@pytest.mark.parametrize(
    "error",
    [
        openai.APIResponseValidationError(response=httpx2.Response(200, request=_REQUEST), body=None),
        openai.OpenAIError("gateway returned garbage"),
    ],
    ids=["response-validation", "base-sdk-error"],
)
def test_other_sdk_errors_stay_typed(error: Exception) -> None:
    """Any SDK error must surface as LLMError so callers can fall back instead of crashing."""
    with pytest.raises(LLMError, match="LLM SDK 오류") as caught:
        run_structured(make_llm(FakeOpenAI(error)))

    assert type(caught.value) is LLMError
    assert caught.value.__cause__ is error


def test_korean_refusal_text_is_surfaced() -> None:
    client = FakeOpenAI(completion(None, refusal="해당 요청은 도와드릴 수 없습니다"))

    with pytest.raises(LLMRefusalError, match="해당 요청은 도와드릴 수 없습니다"):
        run_structured(make_llm(client))


def test_gateway_ignoring_strict_mode_cannot_crash_the_parser() -> None:
    """A server that ignores ``response_format`` may return anything; it ends as LLMOutputError."""
    deep = "[" * 50_000 + "]" * 50_000
    client = FakeOpenAI(completion(deep), completion("여기 있습니다: " + VALID))

    with pytest.raises(LLMOutputError, match="1회 재요청 후"):
        run_structured(make_llm(client))
    assert len(client.completions.calls) == 2


def test_empty_request_sends_system_and_user_turns() -> None:
    client = FakeOpenAI(completion(VALID))

    run_structured(make_llm(client), documents=(), instructions="")

    (kwargs,) = client.completions.calls
    system, user = kwargs["messages"]
    assert system["content"].startswith(SAFETY_PREAMBLE)
    assert user["content"].strip()


# --------------------------------------------------------------------------- review 2: hostile text


def test_gateway_infinity_reply_goes_through_one_repair() -> None:
    """Servers that ignore strict mode can emit ``Infinity``; Python's json would accept it."""
    client = FakeOpenAI(completion('{"action": Infinity, "quote": "q"}'), completion(VALID))

    result = run_structured(make_llm(client))

    assert result.data.action == "롤백 스크립트 실행"
    assert "Infinity" in client.completions.calls[1]["messages"][3]["content"]


def test_lone_surrogates_reach_the_sdk_as_encodable_text() -> None:
    client = FakeOpenAI(completion(VALID))
    broken = (UntrustedDocument(doc_id="scan.pdf", title="DB\udfff 런북", text="장애 시 롤백\ud800 실행"),)

    run_structured(make_llm(client), documents=broken, context="담당\udc00")

    (kwargs,) = client.completions.calls
    for turn in kwargs["messages"]:
        turn["content"].encode("utf-8")
    assert "롤백� 실행" in kwargs["messages"][1]["content"]
