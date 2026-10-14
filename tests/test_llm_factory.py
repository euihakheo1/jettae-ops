"""build_llm modes + MeteredLLM accounting (jettae.llm.factory)."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, TypeVar

import openai
import pytest
from pydantic import BaseModel, SecretStr

from jettae.config import Settings
from jettae.domain.inputs import LlmMode
from jettae.llm import (
    AnthropicLLM,
    CassetteLLM,
    FakeLLM,
    LLMClient,
    LLMDisabledError,
    LLMResult,
    LLMUsage,
    MeteredLLM,
    NullLLM,
    OpenAICompatLLM,
    UntrustedDocument,
    UsageMeter,
    build_llm,
)

R = TypeVar("R")
T = TypeVar("T", bound=BaseModel)


def run(coro: Coroutine[Any, Any, R]) -> R:
    """Drive a coroutine that never suspends (all clients here are fakes) to completion."""
    try:
        coro.send(None)
    except StopIteration as done:
        result: R = done.value
        return result
    coro.close()
    raise AssertionError("coroutine suspended; this test needs a real event loop")


class Note(BaseModel):
    text: str


class TokenLLM:
    """Inner client reporting fixed token usage, to check what the meter accumulates."""

    provider = "anthropic"
    model = "claude-opus-5-5"

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
        usage = LLMUsage(
            provider=self.provider,
            model=self.model,
            task=task,
            input_tokens=100,
            output_tokens=10,
            cache_read_tokens=40,
        )
        return LLMResult(data=schema.model_validate({"text": task}), usage=usage)


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {"cassette_path": tmp_path / "cassettes" / "llm.jsonl"} | overrides
    return Settings(_env_file=None, **values)


def ask(llm: LLMClient, task: str = "extract") -> LLMResult[Note]:
    return run(llm.structured(task=task, instructions="i", schema=Note))


@pytest.fixture(autouse=True)
def _hermetic_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Live clients are constructed (never called): give them a dummy key, keep them local."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9")


def test_off_mode_is_the_null_llm(tmp_path: Path) -> None:
    llm = build_llm(make_settings(tmp_path), LlmMode.OFF)

    assert isinstance(llm, NullLLM)
    with pytest.raises(LLMDisabledError):
        ask(llm)


def test_replay_mode_reads_the_configured_cassette(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)

    llm = build_llm(settings, LlmMode.REPLAY)

    assert isinstance(llm, CassetteLLM)
    assert (llm.mode, llm.inner, llm.path) == ("replay", None, settings.cassette_path)
    with pytest.raises(LLMDisabledError, match="카세트에 없는 요청"):
        ask(llm)


def test_live_anthropic_uses_model_and_effort_from_settings(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, anthropic_model="claude-opus-5-5", anthropic_effort="high")

    llm = build_llm(settings, LlmMode.LIVE)

    assert isinstance(llm, AnthropicLLM)
    assert (llm.model, llm.effort, llm.use_fallbacks) == ("claude-opus-5-5", "high", True)


def test_live_openai_uses_base_url_model_and_secret_key(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path,
        llm_provider="openai",
        openai_base_url="http://127.0.0.1:9/v1",
        openai_model="qwen3-32b",
        openai_api_key=SecretStr("sk-local"),
    )

    llm = build_llm(settings, LlmMode.LIVE)

    assert isinstance(llm, OpenAICompatLLM)
    assert (llm.base_url, llm.model) == ("http://127.0.0.1:9/v1", "qwen3-32b")
    assert isinstance(llm._client, openai.AsyncOpenAI)
    assert llm._client.api_key == "sk-local"


def test_live_with_record_wraps_the_live_client_in_a_recording_cassette(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)

    llm = build_llm(settings, LlmMode.LIVE, record=True)

    assert isinstance(llm, CassetteLLM)
    assert llm.mode == "record"
    assert isinstance(llm.inner, AnthropicLLM)
    assert llm.path == settings.cassette_path


def test_meter_wraps_every_mode(tmp_path: Path) -> None:
    meter = UsageMeter()

    for mode in LlmMode:
        llm = build_llm(make_settings(tmp_path), mode, meter)
        assert isinstance(llm, MeteredLLM)
        assert isinstance(llm, LLMClient)


def test_metered_llm_counts_calls_tokens_and_tasks() -> None:
    meter = UsageMeter()
    llm = MeteredLLM(TokenLLM(), meter)

    for task in ("extract", "verify", "extract"):
        ask(llm, task)

    assert (llm.provider, llm.model) == ("anthropic", "claude-opus-5-5")
    assert meter.calls == 3
    assert (meter.input_tokens, meter.output_tokens, meter.cache_read_tokens) == (300, 30, 120)
    assert meter.by_task() == {"extract": 2, "verify": 1}


def test_failed_calls_are_not_metered(tmp_path: Path) -> None:
    meter = UsageMeter()
    llm = build_llm(make_settings(tmp_path), LlmMode.OFF, meter)

    with pytest.raises(LLMDisabledError):
        ask(llm)
    assert meter.calls == 0


def test_replayed_calls_are_metered_as_cached(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    ask(CassetteLLM(TokenLLM(), settings.cassette_path, "record"))
    meter = UsageMeter()

    ask(build_llm(settings, LlmMode.REPLAY, meter))

    (usage,) = meter.records
    assert usage.cached_replay is True
    assert (usage.provider, usage.model, usage.input_tokens) == ("cassette", "claude-opus-5-5", 100)


def test_metered_fake_reports_fake_identity() -> None:
    meter = UsageMeter()

    ask(MeteredLLM(FakeLLM(lambda *_: {"text": "x"}), meter))

    assert [(record.provider, record.model) for record in meter.records] == [("fake", "fake-1")]


# --------------------------------------------------------------------------- review: mode coercion (P4)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("off", NullLLM), ("replay", CassetteLLM), ("live", AnthropicLLM)],
)
def test_raw_string_modes_map_like_the_enum(tmp_path: Path, raw: str, expected: type[object]) -> None:
    """A plain string from an API payload or CLI flag must select the same client as the enum;
    in particular "off" must never fall through to a live model."""
    llm = build_llm(make_settings(tmp_path), raw)  # type: ignore[arg-type]

    assert type(llm) is expected


def test_unknown_mode_is_rejected_not_treated_as_live(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="LlmMode"):
        build_llm(make_settings(tmp_path), "online")  # type: ignore[arg-type]


def test_replay_mode_never_constructs_a_live_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY")

    llm = build_llm(make_settings(tmp_path), LlmMode.REPLAY)

    assert isinstance(llm, CassetteLLM)
    assert llm.inner is None


def test_package_imports_without_optional_sdks() -> None:
    """P4: ``import jettae.llm`` and LLM-OFF work with neither ``anthropic`` nor ``openai`` installed."""
    script = (
        "import sys; sys.modules['anthropic'] = None; sys.modules['openai'] = None\n"
        "import asyncio, jettae.llm as llm\n"
        "missing = [n for n in llm.__all__ if not hasattr(llm, n)]\n"
        "assert not missing, missing\n"
        "assert type(llm.AnthropicLLM()._client) is type(None)\n"
        "assert 'anthropic' not in [m for m in sys.modules if sys.modules[m] is not None]\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
