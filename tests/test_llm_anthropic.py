"""Anthropic adapter with an injected fake SDK client — no network (jettae.llm.anthropic_client)."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Coroutine
from types import SimpleNamespace
from typing import Any, TypeVar

import anthropic
import httpx2
import pytest
from pydantic import BaseModel

from jettae.llm.anthropic_client import FALLBACK_BETA, AnthropicLLM, _echo_content
from jettae.llm.base import (
    LLMClient,
    LLMDisabledError,
    LLMError,
    LLMOutputError,
    LLMRefusalError,
    LLMResult,
    UntrustedDocument,
)
from jettae.llm.prompting import SAFETY_PREAMBLE, render_user_content
from jettae.llm.schema import to_strict_json_schema

DOCS = (UntrustedDocument(doc_id="doc-1", title="결제 런북", text="1차 담당: 김대리"),)
FORBIDDEN_PARAMS = {"temperature", "top_p", "top_k", "thinking", "tool_choice", "tools", "stop_sequences"}
_REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
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


class Owner(BaseModel):
    name: str
    quote: str
    escalation_minutes: int | None = None


def block(kind: str, **fields: Any) -> SimpleNamespace:
    return SimpleNamespace(type=kind, **fields)


def message(
    *texts: str,
    stop_reason: str = "end_turn",
    model: str = "claude-opus-5-5",
    input_tokens: int = 100,
    output_tokens: int = 20,
    cache_read: int | None = 60,
    stop_details: Any = None,
    content: Any = None,
) -> SimpleNamespace:
    usage = SimpleNamespace(
        input_tokens=input_tokens, output_tokens=output_tokens, cache_read_input_tokens=cache_read
    )
    blocks = content if content is not None else [block("text", text=text) for text in texts]
    return SimpleNamespace(
        content=blocks,
        stop_reason=stop_reason,
        stop_details=stop_details,
        model=model,
        usage=usage,
        _request_id="req_test",
    )


class FakeMessages:
    """Stands in for ``client.beta.messages``; records every kwargs dict it receives."""

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


class FakeAnthropic:
    def __init__(self, *outcomes: Any) -> None:
        self.messages = FakeMessages(*outcomes)
        self.beta = SimpleNamespace(messages=self.messages)


class LegacyMessages:
    """An SDK whose ``create`` predates ``output_config``/``fallbacks`` as named parameters."""

    def __init__(self, reply: Any) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    async def create(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[Any],
        system: Any = None,
        betas: list[str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> Any:
        self.calls.append(
            {"model": model, "max_tokens": max_tokens, "messages": messages, "system": system}
            | {"betas": betas, "extra_body": extra_body}
        )
        return self.reply


def run_structured(llm: AnthropicLLM, **overrides: Any) -> LLMResult[Owner]:
    request: dict[str, Any] = {
        "task": "extract",
        "instructions": "담당자를 추출하세요.",
        "schema": Owner,
        "documents": DOCS,
        "context": "명부: 김대리",
    }
    return run(llm.structured(**(request | overrides)))


VALID = '{"name": "김대리", "quote": "1차 담당: 김대리", "escalation_minutes": null}'


# --------------------------------------------------------------------------- happy path


def test_success_returns_validated_data_and_usage() -> None:
    client = FakeAnthropic(message(VALID))
    llm = AnthropicLLM(client=client)

    result = run_structured(llm)

    assert result.data == Owner(name="김대리", quote="1차 담당: 김대리")
    usage = result.usage
    assert (usage.provider, usage.model, usage.task) == ("anthropic", "claude-opus-5-5", "extract")
    assert (usage.input_tokens, usage.output_tokens, usage.cache_read_tokens) == (100, 20, 60)
    assert usage.latency_ms >= 0.0
    assert usage.cached_replay is False
    assert isinstance(llm, LLMClient)


def test_request_carries_fallbacks_effort_schema_and_cached_system() -> None:
    client = FakeAnthropic(message(VALID))

    run_structured(AnthropicLLM(client=client), max_tokens=4000)

    (kwargs,) = client.messages.calls
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["max_tokens"] == 4000
    assert kwargs["betas"] == [FALLBACK_BETA] == ["server-side-fallback-2026-07-01"]
    assert kwargs["fallbacks"] == "default"
    assert kwargs["output_config"] == {
        "effort": "medium",
        "format": {"type": "json_schema", "schema": to_strict_json_schema(Owner)},
    }
    (system,) = kwargs["system"]
    assert system["type"] == "text"
    assert system["cache_control"] == {"type": "ephemeral"}
    assert system["text"].startswith(SAFETY_PREAMBLE)
    assert system["text"].endswith("담당자를 추출하세요.")
    assert kwargs["messages"] == [{"role": "user", "content": render_user_content(DOCS, "명부: 김대리")}]
    assert not FORBIDDEN_PARAMS & set(kwargs)
    assert "extra_body" not in kwargs


def test_instructions_never_leak_into_the_user_turn() -> None:
    client = FakeAnthropic(message(VALID))

    run_structured(AnthropicLLM(client=client), instructions="SECRET-INSTRUCTION")

    (kwargs,) = client.messages.calls
    assert "SECRET-INSTRUCTION" not in kwargs["messages"][0]["content"]


def test_effort_and_model_are_configurable() -> None:
    client = FakeAnthropic(message(VALID))

    run_structured(AnthropicLLM("claude-sonnet-5-5", "high", client=client))

    (kwargs,) = client.messages.calls
    assert kwargs["model"] == "claude-sonnet-5-5"
    assert kwargs["output_config"]["effort"] == "high"


def test_invalid_effort_is_rejected_early() -> None:
    with pytest.raises(ValueError, match="effort"):
        AnthropicLLM(effort="extreme", client=FakeAnthropic())  # type: ignore[arg-type]


def test_fallbacks_can_be_disabled() -> None:
    client = FakeAnthropic(message(VALID))

    run_structured(AnthropicLLM(client=client, use_fallbacks=False))

    (kwargs,) = client.messages.calls
    assert "betas" not in kwargs
    assert "fallbacks" not in kwargs
    assert kwargs["output_config"]["effort"] == "medium"


def test_unnamed_sdk_params_are_sent_via_extra_body() -> None:
    legacy = LegacyMessages(message(VALID))
    client = SimpleNamespace(beta=SimpleNamespace(messages=legacy))

    result = run_structured(AnthropicLLM(client=client))

    assert result.data.name == "김대리"
    (call,) = legacy.calls
    assert call["betas"] == [FALLBACK_BETA]
    assert call["extra_body"] == {
        "output_config": {
            "effort": "medium",
            "format": {"type": "json_schema", "schema": to_strict_json_schema(Owner)},
        },
        "fallbacks": "default",
    }


def test_installed_sdk_names_output_config_and_fallbacks() -> None:
    llm = AnthropicLLM(client=anthropic.AsyncAnthropic(api_key="test-key", max_retries=0))

    kwargs = llm.request_kwargs("system", [{"role": "user", "content": "u"}], {"type": "object"}, 100)

    assert kwargs["fallbacks"] == "default"
    assert kwargs["output_config"]["format"]["schema"] == {"type": "object"}
    assert "extra_body" not in kwargs


def test_fallback_served_model_is_reported_in_usage() -> None:
    reply = message(content=[block("fallback"), block("text", text=VALID)], model="claude-opus-4-8")

    result = run_structured(AnthropicLLM(client=FakeAnthropic(reply)))

    assert result.usage.model == "claude-opus-4-8"
    assert result.data.name == "김대리"


def test_missing_cache_read_counts_as_zero() -> None:
    result = run_structured(AnthropicLLM(client=FakeAnthropic(message(VALID, cache_read=None))))

    assert result.usage.cache_read_tokens == 0


# --------------------------------------------------------------------------- stop reasons


def test_refusal_is_raised_before_content_is_read() -> None:
    details = SimpleNamespace(type="refusal", category="cyber", explanation="declined")
    client = FakeAnthropic(message(stop_reason="refusal", stop_details=details, content=None))
    client.messages.outcomes[0].content = None  # reading content would crash

    with pytest.raises(LLMRefusalError, match=r"task=extract, 분류=cyber"):
        run_structured(AnthropicLLM(client=client))
    assert len(client.messages.calls) == 1


def test_refusal_without_details_has_unknown_category() -> None:
    client = FakeAnthropic(message(stop_reason="refusal"))

    with pytest.raises(LLMRefusalError, match="분류=미상"):
        run_structured(AnthropicLLM(client=client))


def test_max_tokens_is_an_output_error_without_repair() -> None:
    client = FakeAnthropic(message('{"name": "김', stop_reason="max_tokens"))

    with pytest.raises(LLMOutputError, match="max_tokens=16000"):
        run_structured(AnthropicLLM(client=client))
    assert len(client.messages.calls) == 1


def test_reply_without_text_block_is_an_output_error() -> None:
    client = FakeAnthropic(message(content=[block("thinking", thinking="", signature="sig")]))

    with pytest.raises(LLMOutputError, match="텍스트 블록"):
        run_structured(AnthropicLLM(client=client))


# --------------------------------------------------------------------------- repair


def test_invalid_json_is_repaired_once_with_append_only_history() -> None:
    first = message("죄송합니다, JSON이 아닙니다", input_tokens=100, output_tokens=10, cache_read=0)
    second = message(VALID, input_tokens=150, output_tokens=20, cache_read=90)
    client = FakeAnthropic(first, second)

    result = run_structured(AnthropicLLM(client=client))

    assert result.data.name == "김대리"
    initial, repair = client.messages.calls
    assert len(initial["messages"]) == 1
    assert repair["messages"][0] == initial["messages"][0]
    assert repair["messages"][1] == {"role": "assistant", "content": first.content}
    assert repair["messages"][2]["role"] == "user"
    assert "JSON 구문 오류" in repair["messages"][2]["content"]
    assert repair["system"] == initial["system"]
    assert (result.usage.input_tokens, result.usage.output_tokens, result.usage.cache_read_tokens) == (
        250,
        30,
        90,
    )


def test_schema_violation_is_repaired() -> None:
    client = FakeAnthropic(message('{"name": "김대리"}'), message(VALID))

    result = run_structured(AnthropicLLM(client=client))

    assert result.data.quote == "1차 담당: 김대리"
    assert "- quote: Field required" in client.messages.calls[1]["messages"][2]["content"]


def test_second_invalid_reply_raises_output_error() -> None:
    client = FakeAnthropic(message("nope"), message("still nope"))

    with pytest.raises(LLMOutputError, match="1회 재요청 후"):
        run_structured(AnthropicLLM(client=client))
    assert len(client.messages.calls) == 2


def test_refusal_during_repair_is_still_a_refusal() -> None:
    client = FakeAnthropic(message("nope"), message(stop_reason="refusal"))

    with pytest.raises(LLMRefusalError):
        run_structured(AnthropicLLM(client=client))


def test_echo_keeps_only_text_before_the_last_fallback_marker() -> None:
    early_thinking = block("thinking", thinking="", signature="s1")
    early_text = block("text", text="partial")
    marker = block("fallback")
    late_thinking = block("thinking", thinking="", signature="s2")
    late_text = block("text", text="final")

    echoed = _echo_content([early_thinking, early_text, marker, late_thinking, late_text])

    assert echoed == [early_text, marker, late_thinking, late_text]


def test_echo_without_fallback_keeps_everything_unchanged() -> None:
    blocks = [block("thinking", thinking="", signature="s"), block("text", text="x")]

    assert _echo_content(blocks) == blocks


# --------------------------------------------------------------------------- errors


def _status(cls: type[anthropic.APIStatusError], code: int) -> anthropic.APIStatusError:
    return cls("boom", response=httpx2.Response(code, request=_REQUEST), body=None)


@pytest.mark.parametrize(
    ("error", "expected", "fragment"),
    [
        (_status(anthropic.AuthenticationError, 401), LLMDisabledError, "API 키 확인 필요"),
        (_status(anthropic.PermissionDeniedError, 403), LLMDisabledError, "API 키 확인 필요"),
        (anthropic.CredentialsError("no profile"), LLMDisabledError, "API 키 확인 필요"),
        (
            TypeError('"Could not resolve authentication method. Expected one of api_key"'),
            LLMDisabledError,
            "자격",
        ),
        (_status(anthropic.RateLimitError, 429), LLMError, "한도"),
        (_status(anthropic.InternalServerError, 500), LLMError, "서버 오류"),
        (_status(anthropic.APIStatusError, 529), LLMError, "HTTP 529"),
        (_status(anthropic.BadRequestError, 400), LLMError, "HTTP 400"),
        (anthropic.APIConnectionError(request=_REQUEST), LLMError, "연결"),
        (anthropic.APITimeoutError(request=_REQUEST), LLMError, "연결"),
    ],
    ids=[
        "auth",
        "permission",
        "credentials",
        "no-auth-type-error",
        "rate",
        "5xx",
        "overloaded",
        "4xx",
        "conn",
        "timeout",
    ],
)
def test_sdk_errors_are_mapped(error: Exception, expected: type[LLMError], fragment: str) -> None:
    client = FakeAnthropic(error)

    with pytest.raises(expected, match=fragment) as caught:
        run_structured(AnthropicLLM(client=client))

    assert type(caught.value) is expected  # rate limits etc. must NOT look like "LLM disabled"
    assert caught.value.__cause__ is error


def test_non_sdk_errors_propagate_unchanged() -> None:
    client = FakeAnthropic(RuntimeError("bug in caller"))

    with pytest.raises(RuntimeError, match="bug in caller"):
        run_structured(AnthropicLLM(client=client))


# --------------------------------------------------------------------------- lazy availability


def test_missing_package_disables_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)

    llm = AnthropicLLM()

    with pytest.raises(LLMDisabledError, match="anthropic 패키지"):
        run_structured(llm)


def test_missing_credentials_disable_lazily(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    for name in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_PROFILE",
        "ANTHROPIC_IDENTITY_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("ANTHROPIC_FEDERATION_RULE_ID", raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path / "no-config"))
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9")  # never leaves the machine

    llm = AnthropicLLM(max_retries=0)

    with pytest.raises(LLMDisabledError, match="API 키 확인 필요"):
        run_structured(llm)


# --------------------------------------------------------------------------- review: stop reasons (P3)


def test_context_window_exceeded_is_an_output_error_without_repair() -> None:
    client = FakeAnthropic(message('{"name": "김', stop_reason="model_context_window_exceeded"))

    with pytest.raises(LLMOutputError, match="컨텍스트 창"):
        run_structured(AnthropicLLM(client=client))
    assert len(client.messages.calls) == 1  # a repair turn would only lengthen the context


@pytest.mark.parametrize("stop_reason", ["pause_turn", "tool_use", "compaction", None, "brand_new_reason"])
def test_unexpected_stop_reasons_are_not_success(stop_reason: str | None) -> None:
    """P3: a reply that did not end naturally is never accepted, even if its text parses."""
    client = FakeAnthropic(message(VALID, stop_reason="end_turn"))
    client.messages.outcomes[0].stop_reason = stop_reason

    with pytest.raises(LLMOutputError, match=f"stop_reason={stop_reason}"):
        run_structured(AnthropicLLM(client=client))
    assert len(client.messages.calls) == 1


def test_refusal_category_is_reported_in_korean_message() -> None:
    details = SimpleNamespace(type="refusal", category="bio", explanation=None)
    client = FakeAnthropic(message(stop_reason="refusal", stop_details=details))

    with pytest.raises(LLMRefusalError, match=r"모델이 요청을 거절했습니다 \(task=extract, 분류=bio\)"):
        run_structured(AnthropicLLM(client=client))


def test_korean_text_with_unicode_escapes_validates() -> None:
    escaped = r'{"name": "\uae40\ub300\ub9ac", "quote": "1\ucc28 \ub2f4\ub2f9: \uae40\ub300\ub9ac"}'
    assert "\uae40" not in escaped  # the model sent JSON escapes, not raw Hangul

    result = run_structured(AnthropicLLM(client=FakeAnthropic(message(escaped))))

    assert result.data == Owner(name="김대리", quote="1차 담당: 김대리")


def test_pathologically_nested_reply_is_repaired_then_an_output_error() -> None:
    deep = "[" * 50_000 + "]" * 50_000
    client = FakeAnthropic(message(deep), message(deep))

    with pytest.raises(LLMOutputError, match="1회 재요청 후"):
        run_structured(AnthropicLLM(client=client))
    assert "JSON 중첩" in client.messages.calls[1]["messages"][2]["content"]


def test_empty_request_still_sends_a_non_empty_user_turn() -> None:
    client = FakeAnthropic(message(VALID))

    run_structured(AnthropicLLM(client=client), documents=(), context="", instructions="")

    (kwargs,) = client.messages.calls
    assert kwargs["messages"][0]["content"].strip()
    assert kwargs["system"][0]["text"].startswith(SAFETY_PREAMBLE)


# --------------------------------------------------------------------------- review: SDK error coverage


def test_workload_identity_failure_means_no_credentials() -> None:
    error = anthropic.WorkloadIdentityError("token exchange failed")
    client = FakeAnthropic(error)

    with pytest.raises(LLMDisabledError, match="API 키 확인 필요") as caught:
        run_structured(AnthropicLLM(client=client))
    assert caught.value.__cause__ is error


@pytest.mark.parametrize(
    "error",
    [
        anthropic.APIResponseValidationError(response=httpx2.Response(200, request=_REQUEST), body=None),
        anthropic.AnthropicError("middleware failure"),
    ],
    ids=["response-validation", "base-sdk-error"],
)
def test_other_sdk_errors_stay_typed(error: Exception) -> None:
    """Any SDK error must surface as LLMError so callers can fall back instead of crashing."""
    client = FakeAnthropic(error)

    with pytest.raises(LLMError, match="Anthropic SDK 오류") as caught:
        run_structured(AnthropicLLM(client=client))
    assert type(caught.value) is LLMError
    assert caught.value.__cause__ is error


# --------------------------------------------------------------------------- review: real SDK on the wire


def _event_loop() -> asyncio.AbstractEventLoop:
    """New event loop, retried: some sandboxed Windows hosts intermittently fail the
    ``socket.socketpair()`` every loop needs for its self-pipe (WinError 10014)."""
    for _ in range(9):
        try:
            return asyncio.new_event_loop()
        except OSError:
            continue
    return asyncio.new_event_loop()


def _wire_reply(content: list[dict[str, Any]], **usage: int) -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": content,
        "stop_reason": "end_turn",
        "stop_details": None,
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0} | usage,
    }


def test_real_sdk_wire_request_and_repair_echo() -> None:
    """End to end through the installed SDK with an in-memory transport (no socket, no network):
    the bytes on the wire carry the beta header, fallbacks, effort + strict schema, a cached
    system prompt, no forbidden sampling/thinking params, and a repair turn that echoes the
    thinking block unchanged."""
    seen: list[tuple[httpx2.Headers, dict[str, Any]]] = []
    replies = [
        _wire_reply(
            [{"type": "thinking", "thinking": "", "signature": "sig-1"}, {"type": "text", "text": "아님"}],
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=4,
        ),
        _wire_reply([{"type": "text", "text": VALID}], input_tokens=20, output_tokens=6),
    ]

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append((request.headers, json.loads(request.content)))
        return httpx2.Response(200, json=replies.pop(0), headers={"request-id": "req_wire"})

    sdk = anthropic.AsyncAnthropic(
        api_key="test-key",
        base_url="http://sdk.test",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    llm = AnthropicLLM(client=sdk, effort="low")

    with asyncio.Runner(loop_factory=_event_loop) as runner:
        result = runner.run(
            llm.structured(
                task="extract",
                instructions="담당자를 추출하세요.",
                schema=Owner,
                documents=DOCS,
                context="명부",
            )
        )

    assert result.data == Owner(name="김대리", quote="1차 담당: 김대리")
    assert (result.usage.input_tokens, result.usage.output_tokens, result.usage.cache_read_tokens) == (
        30,
        11,
        4,
    )
    (headers, first), (_, repair) = seen
    assert FALLBACK_BETA in headers["anthropic-beta"]
    assert first["fallbacks"] == "default"
    assert first["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": to_strict_json_schema(Owner)},
    }
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert not FORBIDDEN_PARAMS & set(first)
    assert "betas" not in first  # a header, never a body field
    assert repair["messages"][:1] == first["messages"]
    assert repair["messages"][1] == {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "sig-1"},
            {"type": "text", "text": "아님"},
        ],
    }
    assert repair["messages"][2]["role"] == "user"


# --------------------------------------------------------------------------- review 2: hostile text


def test_non_json_number_reply_goes_through_one_repair() -> None:
    client = FakeAnthropic(
        message('{"name": "김대리", "quote": "1차 담당: 김대리", "escalation_minutes": NaN}'), message(VALID)
    )

    result = run_structured(AnthropicLLM(client=client))

    assert result.data.escalation_minutes is None
    assert "NaN" in client.messages.calls[1]["messages"][2]["content"]


def test_lone_surrogates_reach_the_sdk_as_encodable_text() -> None:
    client = FakeAnthropic(message(VALID))
    broken = (UntrustedDocument(doc_id="d\ud800", title="결제\udfff", text="1차 담당: 김\ud800대리"),)

    run_structured(
        AnthropicLLM(client=client), documents=broken, context="명부\udc00", instructions="지시\ud834"
    )

    (kwargs,) = client.messages.calls
    json.dumps(kwargs["messages"], ensure_ascii=False).encode("utf-8")
    kwargs["system"][0]["text"].encode("utf-8")
    assert "김�대리" in kwargs["messages"][0]["content"]


def test_real_sdk_sends_a_runbook_with_lone_surrogates() -> None:
    """Before the fix the installed SDK raised a bare UnicodeEncodeError while encoding the body —
    not an LLMError, so the extractor crashed instead of falling back."""
    bodies: list[bytes] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(request.content)
        return httpx2.Response(200, json=_wire_reply([{"type": "text", "text": VALID}], input_tokens=3))

    sdk = anthropic.AsyncAnthropic(
        api_key="test-key",
        base_url="http://sdk.test",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    broken = (UntrustedDocument(doc_id="scan.pdf", title="스캔 런북", text="1차 담당: 김\ud800대리"),)

    with asyncio.Runner(loop_factory=_event_loop) as runner:
        result = runner.run(
            AnthropicLLM(client=sdk).structured(
                task="extract", instructions="담당자를 추출하세요.", schema=Owner, documents=broken
            )
        )

    assert result.data.name == "김대리"
    (body,) = bodies
    sent = json.loads(body)["messages"][0]["content"]
    assert "김�대리" in sent
