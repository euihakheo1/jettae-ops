"""Paid-call plumbing with fake SDKs: spend recording, the budget gate, the OpenAI Responses adapter."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from jettae.evalkit.realdata.clients import budgeted, build_study_clients
from jettae.evalkit.realdata.pricing import ModelRef, call_usd
from jettae.evalkit.realdata.providers import (
    UNMETERED_LINE,
    BacktranslationSkippedError,
    BudgetedLLM,
    BudgetExhausted,
    OpenAIResponsesLLM,
    SkipNonKoreanBacktranslation,
    SpendTracker,
    claude_llm,
    gather_jobs,
)
from jettae.llm.base import LLMDisabledError, LLMError, LLMOutputError, LLMRefusalError, UntrustedDocument
from jettae.llm.fake import FakeLLM
from support_realdata import FakeAnthropicSDK, FakeOpenAISDK


class Verdict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ok: bool
    reason: str


DOC = UntrustedDocument(doc_id="d1", title="t", text="The payments API is owned by the payments team.")


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


# --------------------------------------------------------------------------- tracker & gate


def test_tracker_records_and_refuses_beyond_the_cap() -> None:
    tracker = SpendTracker(max_usd=1.0)
    usd = tracker.record("claude-opus-5-5", input_tokens=100_000, output_tokens=10_000)
    assert usd == pytest.approx(0.6)
    tracker.reserve(0.3, what="extract")
    with pytest.raises(BudgetExhausted, match="예산 상한"):
        tracker.reserve(0.2, what="extract")  # 0.6 spent + 0.3 in flight + 0.2 > 1.0
    tracker.release(0.3)
    tracker.reserve(0.2, what="extract")
    report = tracker.report()
    assert report.spent_usd == pytest.approx(0.6)
    assert report.by_model[0].model == "claude-opus-5-5"
    assert report.by_model[0].calls == 1
    assert not issubclass(BudgetExhausted, LLMError)  # never mistaken for an LLM failure (fallback)


def test_budget_gate_refuses_before_sending() -> None:
    inner = FakeLLM(lambda *_: Verdict(ok=True, reason="x"))
    tracker = SpendTracker(max_usd=0.0001)
    gate = BudgetedLLM(inner, tracker, lambda task, docs, chars, max_tokens: 0.01)
    with pytest.raises(BudgetExhausted):
        _run(gate.structured(task="extract", instructions="i", schema=Verdict, documents=(DOC,)))
    assert inner.calls == []  # nothing was sent
    roomy = BudgetedLLM(inner, SpendTracker(max_usd=10.0), lambda *_: 0.01)
    result = _run(roomy.structured(task="extract", instructions="i", schema=Verdict, documents=(DOC,)))
    assert result.data.ok
    assert roomy.tracker._reserved == 0.0  # reservation released after the call


def test_budgeted_estimator_prices_by_task() -> None:
    tracker = SpendTracker(max_usd=None)
    seen: list[float] = []

    class Spy(BudgetedLLM):
        async def structured(self, **kwargs: Any) -> Any:
            seen.append(
                self.estimate(kwargs["task"], kwargs.get("documents", ()), 0, kwargs.get("max_tokens", 16000))
            )
            return await super().structured(**kwargs)

    inner = FakeLLM(lambda *_: Verdict(ok=True, reason="x"))
    base = budgeted(inner, ModelRef("anthropic", "claude-opus-5-5"), "medium", tracker)
    spy = Spy(inner, tracker, base.estimate)
    _run(spy.structured(task="extract", instructions="", schema=Verdict, documents=(DOC,)))
    _run(
        spy.structured(
            task="backtranslate", instructions="", schema=Verdict, documents=(DOC,), max_tokens=1024
        )
    )
    assert seen[0] > seen[1] > 0  # extraction is the expensive call kind; output capped at max_tokens


def test_backtranslation_gate_skips_only_non_korean_sources() -> None:
    inner = FakeLLM(lambda *_: Verdict(ok=True, reason="x"))
    gate = SkipNonKoreanBacktranslation(inner, frozenset({"d1"}))
    with pytest.raises(BacktranslationSkippedError):
        _run(gate.structured(task="backtranslate", instructions="i", schema=Verdict, documents=(DOC,)))
    assert issubclass(BacktranslationSkippedError, LLMError)  # the product's judge reads it as "no verdict"
    korean = UntrustedDocument(doc_id="k1", title="t", text="결제 서비스 담당은 김민수 대리다.")
    _run(gate.structured(task="backtranslate", instructions="i", schema=Verdict, documents=(korean,)))
    _run(gate.structured(task="extract", instructions="i", schema=Verdict, documents=(DOC,)))
    assert [c.task for c in inner.calls] == ["backtranslate", "extract"]
    assert gate.skipped == 1


# --------------------------------------------------------------------------- Anthropic


def test_claude_adapter_over_recording_sdk_prices_every_response() -> None:
    sdk = FakeAnthropicSDK(input_tokens=2000, output_tokens=1000)
    tracker = SpendTracker()
    llm = claude_llm("claude-opus-5-5", "medium", sdk, tracker)
    result = _run(
        llm.structured(task="extract", instructions="추출", schema=_extraction_schema(), documents=(DOC,))
    )
    assert result.data.entities  # the fake answered from the schema
    request = sdk.created[0]
    assert request["model"] == "claude-opus-5-5"
    assert request["output_config"]["effort"] == "medium"
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert request["fallbacks"] == "default"
    assert request["betas"] == ["server-side-fallback-2026-07-01"]
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    for forbidden in ("temperature", "top_p", "top_k", "thinking"):
        assert forbidden not in request
    assert tracker.spent_usd == pytest.approx(
        call_usd("claude-opus-5-5", input_tokens=2000, output_tokens=1000)
    )


def test_haiku_is_sent_without_server_fallbacks() -> None:
    sdk = FakeAnthropicSDK()
    llm = claude_llm("claude-haiku-5-5", "low", sdk, SpendTracker())
    _run(llm.structured(task="extract", instructions="추출", schema=_extraction_schema(), documents=(DOC,)))
    assert "fallbacks" not in sdk.created[0]
    assert "betas" not in sdk.created[0]
    assert sdk.created[0]["output_config"]["effort"] == "low"


def _extraction_schema() -> type[BaseModel]:
    from jettae.extract.llm_extractor import LlmGraphPayload

    return LlmGraphPayload


# --------------------------------------------------------------------------- OpenAI Responses


class _ScriptedResponses:
    def __init__(self, replies: list[Any]) -> None:
        self.replies = replies
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.replies.pop(0)


def _reply(text: str, *, status: str = "completed", reason: str | None = None, refusal: bool = False) -> Any:
    part = (
        SimpleNamespace(type="refusal", refusal="no")
        if refusal
        else SimpleNamespace(type="output_text", text=text)
    )
    return SimpleNamespace(
        status=status,
        incomplete_details=SimpleNamespace(reason=reason) if reason else None,
        output=[SimpleNamespace(type="message", content=[part])],
        output_text=text,
        usage=SimpleNamespace(
            input_tokens=100, output_tokens=50, input_tokens_details=SimpleNamespace(cached_tokens=40)
        ),
        model="gpt-6-luna",
    )


def test_openai_responses_request_shape_and_usage() -> None:
    responses = _ScriptedResponses([_reply(json.dumps({"ok": True, "reason": "같음"}))])
    tracker = SpendTracker()
    llm = OpenAIResponsesLLM("gpt-6-luna", client=SimpleNamespace(responses=responses), tracker=tracker)
    result = _run(llm.structured(task="code_findings", instructions="판정", schema=Verdict, documents=(DOC,)))
    assert result.data == Verdict(ok=True, reason="같음")
    request = responses.calls[0]
    assert request["model"] == "gpt-6-luna"
    assert request["store"] is False
    assert request["text"]["format"]["type"] == "json_schema"
    assert request["text"]["format"]["strict"] is True
    assert request["text"]["format"]["name"] == "Verdict"
    assert request["text"]["format"]["schema"]["additionalProperties"] is False
    assert request["input"][0]["role"] == "user"
    assert 'nonce="' in request["input"][0]["content"]  # untrusted text travels inside the boundary
    assert "Safety rules" in request["instructions"]
    assert "temperature" not in request
    assert result.usage.cache_read_tokens == 40
    assert tracker.spent_usd == pytest.approx(call_usd("gpt-6-luna", input_tokens=100, output_tokens=50))


def test_openai_responses_repairs_once_then_fails() -> None:
    responses = _ScriptedResponses([_reply("not json"), _reply(json.dumps({"ok": False, "reason": "다름"}))])
    llm = OpenAIResponsesLLM("gpt-6-luna", client=SimpleNamespace(responses=responses))
    result = _run(llm.structured(task="t", instructions="i", schema=Verdict))
    assert result.data.ok is False
    assert [m["role"] for m in responses.calls[1]["input"]] == ["user", "assistant", "user"]  # append-only

    broken = _ScriptedResponses([_reply("x"), _reply("y")])
    with pytest.raises(LLMOutputError):
        _run(
            OpenAIResponsesLLM("m", client=SimpleNamespace(responses=broken)).structured(
                task="t", instructions="i", schema=Verdict
            )
        )


def test_openai_responses_unfinished_replies() -> None:
    cut = _ScriptedResponses([_reply("{", status="incomplete", reason="max_output_tokens")])
    with pytest.raises(LLMOutputError, match="max_output_tokens"):
        _run(
            OpenAIResponsesLLM("m", client=SimpleNamespace(responses=cut)).structured(
                task="t", instructions="i", schema=Verdict
            )
        )
    filtered = _ScriptedResponses([_reply("", status="incomplete", reason="content_filter")])
    with pytest.raises(LLMRefusalError):
        _run(
            OpenAIResponsesLLM("m", client=SimpleNamespace(responses=filtered)).structured(
                task="t", instructions="i", schema=Verdict
            )
        )
    refused = _ScriptedResponses([_reply("", refusal=True)])
    with pytest.raises(LLMRefusalError):
        _run(
            OpenAIResponsesLLM("m", client=SimpleNamespace(responses=refused)).structured(
                task="t", instructions="i", schema=Verdict
            )
        )


def test_openai_sdk_errors_are_mapped() -> None:
    import httpx
    import openai

    request = httpx.Request("POST", "https://api.openai.com/v1/responses")

    class Failing:
        def __init__(self, exc: Exception) -> None:
            self.exc = exc

        async def create(self, **kwargs: Any) -> Any:
            raise self.exc

    auth = openai.AuthenticationError("bad key", response=httpx.Response(401, request=request), body=None)
    with pytest.raises(LLMDisabledError):
        _run(
            OpenAIResponsesLLM("m", client=SimpleNamespace(responses=Failing(auth))).structured(
                task="t", instructions="i", schema=Verdict
            )
        )
    limited = openai.RateLimitError("slow down", response=httpx.Response(429, request=request), body=None)
    with pytest.raises(LLMError, match="한도"):
        _run(
            OpenAIResponsesLLM("m", client=SimpleNamespace(responses=Failing(limited))).structured(
                task="t", instructions="i", schema=Verdict
            )
        )


# --------------------------------------------------------------------------- clients


def test_study_clients_need_two_model_families_and_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    a, b = ModelRef("anthropic", "claude-opus-5-5"), ModelRef("openai", "gpt-6-luna")
    with pytest.raises(ValueError, match="계열"):
        build_study_clients(a, a, a, effort="medium", tracker=SpendTracker())
    with pytest.raises(ValueError, match="Claude"):
        build_study_clients(b, a, b, effort="medium", tracker=SpendTracker())
    monkeypatch.setattr("jettae.evalkit.realdata.clients.load_api_key", lambda name: None)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        build_study_clients(a, a, b, effort="medium", tracker=SpendTracker())
    clients = build_study_clients(
        a,
        a,
        b,
        effort="medium",
        tracker=SpendTracker(),
        anthropic_sdk=FakeAnthropicSDK(),
        openai_sdk=FakeOpenAISDK(),
    )
    assert clients.owned == ()  # injected SDKs belong to the caller
    assert isinstance(clients.coder_b, BudgetedLLM)


# --------------------------------------------------------------------------- review fixes


def _fallback_response(iterations: list[Any] | None) -> Any:
    text = json.dumps({"ok": True, "reason": "x"})
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
        stop_details=None,
        usage=SimpleNamespace(
            input_tokens=1000,
            output_tokens=500,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            iterations=iterations,
        ),
        model="claude-opus-5",
        _request_id="req_test",
    )


class _OneReplySDK:
    def __init__(self, response: Any) -> None:
        outer = self
        self.response = response

        class _Messages:
            async def create(self, **_: Any) -> Any:
                return outer.response

        self.beta = SimpleNamespace(messages=_Messages())


def test_a_fallback_served_reply_is_priced_per_attempt() -> None:
    """Top-level usage covers only the serving attempt; the declined hop is billed too."""
    declined = SimpleNamespace(
        type="message",
        model="claude-opus-5-5",
        input_tokens=1000,
        output_tokens=200,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    served = SimpleNamespace(
        type="fallback_message",
        model="claude-opus-5",
        input_tokens=1000,
        output_tokens=500,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    tracker = SpendTracker()
    llm = claude_llm(
        "claude-opus-5-5", "medium", _OneReplySDK(_fallback_response([declined, served])), tracker
    )
    _run(llm.structured(task="extract", instructions="i", schema=Verdict, documents=(DOC,)))
    expected = call_usd("claude-opus-5-5", input_tokens=1000, output_tokens=200) + call_usd(
        "claude-opus-5", input_tokens=1000, output_tokens=500
    )
    assert tracker.spent_usd == pytest.approx(expected)
    assert {line.model for line in tracker.report().by_model} == {"claude-opus-5-5", "claude-opus-5"}
    # without an itemisation the top-level usage is priced at the served model
    plain = SpendTracker()
    llm = claude_llm("claude-opus-5-5", "medium", _OneReplySDK(_fallback_response(None)), plain)
    _run(llm.structured(task="extract", instructions="i", schema=Verdict, documents=(DOC,)))
    assert plain.spent_usd == pytest.approx(call_usd("claude-opus-5", input_tokens=1000, output_tokens=500))


def test_gather_jobs_sends_nothing_new_after_a_failure() -> None:
    started: list[str] = []

    def job(name: str, outcome: Any) -> Any:
        async def run() -> str:
            started.append(name)
            await asyncio.sleep(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return str(outcome)

        return run

    assert _run(gather_jobs([job("a", 1), job("b", 2)], concurrency=2)) == ["1", "2"]
    started.clear()
    with pytest.raises(BudgetExhausted):
        _run(
            gather_jobs(
                [job("a", LLMError("x")), job("b", BudgetExhausted("cap")), job("c", 3), job("d", 4)],
                concurrency=2,
            )
        )
    assert started == ["a", "b"]  # c and d never started; the budget stop wins over the LLM error
    started.clear()
    with pytest.raises(LLMError):
        _run(gather_jobs([job("a", LLMError("x")), job("b", 2)], concurrency=1))
    assert started == ["a"]


def test_a_call_cancelled_in_flight_is_booked_at_its_reservation() -> None:
    class Hanging:
        provider = "anthropic"
        model = "claude-opus-5-5"

        async def structured(self, **_: Any) -> Any:
            await asyncio.sleep(3600)

    tracker = SpendTracker(max_usd=1.0)
    gate = BudgetedLLM(Hanging(), tracker, lambda *_: 0.25)  # type: ignore[arg-type]

    async def cancel_it() -> None:
        task = asyncio.ensure_future(gate.structured(task="extract", instructions="i", schema=Verdict))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    _run(cancel_it())
    assert tracker.spent_usd == pytest.approx(0.25)
    assert tracker._reserved == 0.0
    assert [line.model for line in tracker.report().by_model] == [UNMETERED_LINE]
