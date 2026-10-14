"""Cassette record/replay — byte-stable files, deterministic misses (jettae.llm.replay)."""

from __future__ import annotations

import asyncio
import json
import threading
import unicodedata
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, TypeVar

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from jettae.llm.base import (
    LLMClient,
    LLMDisabledError,
    LLMOutputError,
    LLMResult,
    LLMUsage,
    UntrustedDocument,
)
from jettae.llm.fake import FakeLLM
from jettae.llm.replay import CassetteLLM, cassette_key

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")

DOCS = (UntrustedDocument(doc_id="doc-1", title="결제 런북", text="1차 담당: 김대리", offset=10),)


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


class Summary(BaseModel):
    text: str
    count: int = 0


# Same strict schema as Summary (no docstring → no description), stricter validator:
# models a cassette recorded before the validation code changed.
class StrictSummary(BaseModel):
    text: str
    count: int = 0

    @field_validator("count")
    @classmethod
    def _positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("count must be positive")
        return value


class Detailed(BaseModel):
    text: str
    count: int = 0
    detail: str | None = None


class CountingLLM:
    """Deterministic inner client that reports token usage and counts calls."""

    provider = "anthropic"
    model = "claude-opus-5-5"

    def __init__(self, *, suspend: bool = False) -> None:
        self.calls = 0
        self.suspend = suspend

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
        self.calls += 1
        if self.suspend:
            await asyncio.sleep(0)  # yield to the loop so concurrent calls genuinely interleave
        data = schema.model_validate({"text": f"{task}:{context}", "count": len(context)})
        usage = LLMUsage(
            provider=self.provider,
            model=self.model,
            task=task,
            input_tokens=100 + len(context),
            output_tokens=7,
            cache_read_tokens=50,
            latency_ms=123.4,
        )
        return LLMResult(data=data, usage=usage)


def call(llm: LLMClient, context: str = "명부: 김대리", task: str = "extract") -> LLMResult[Summary]:
    return run(
        llm.structured(task=task, instructions="요약하세요.", schema=Summary, documents=DOCS, context=context)
    )


def record_all(path: Path, contexts: list[str]) -> CountingLLM:
    inner = CountingLLM()
    cassette = CassetteLLM(inner, path, "record")
    for context in contexts:
        call(cassette, context)
    return inner


# --------------------------------------------------------------------------- record → replay


def test_record_then_replay_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "llm.jsonl"
    recorded = call(CassetteLLM(CountingLLM(), path, "record"))

    replayed = call(CassetteLLM(None, path))

    assert replayed.data == recorded.data == Summary(text="extract:명부: 김대리", count=7)
    assert replayed.usage == LLMUsage(
        provider="cassette",
        model="claude-opus-5-5",
        task="extract",
        input_tokens=107,
        output_tokens=7,
        cache_read_tokens=50,
        latency_ms=0.0,
        cached_replay=True,
    )


def test_recording_returns_the_inner_result_unchanged(tmp_path: Path) -> None:
    result = call(CassetteLLM(CountingLLM(), tmp_path / "c.jsonl", "record"))

    assert result.usage.provider == "anthropic"
    assert result.usage.cached_replay is False
    assert result.usage.latency_ms == 123.4


def test_cassette_files_are_byte_stable(tmp_path: Path) -> None:
    contexts = ["a", "b", "명부: 김대리"]
    record_all(tmp_path / "first.jsonl", contexts)
    record_all(tmp_path / "second.jsonl", contexts)

    first = (tmp_path / "first.jsonl").read_bytes()
    assert first == (tmp_path / "second.jsonl").read_bytes()
    assert b"\r\n" not in first
    lines = first.decode("utf-8").splitlines()
    assert len(lines) == 3
    for line in lines:
        record = json.loads(line)
        assert list(record) == sorted(record)
        assert set(record) == {"key", "task", "provider", "model", "response", "usage"}
        assert "latency_ms" not in record["usage"]  # wall-clock never reaches the cassette
        assert line == json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def test_replay_miss_raises_disabled_for_deterministic_fallback(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    record_all(path, ["known"])

    with pytest.raises(LLMDisabledError, match=r"카세트에 없는 요청 \(task=extract, key=[0-9a-f]{12}\)"):
        call(CassetteLLM(None, path), "unknown")


def test_missing_cassette_file_is_an_empty_cassette(tmp_path: Path) -> None:
    with pytest.raises(LLMDisabledError):
        call(CassetteLLM(None, tmp_path / "absent.jsonl"))
    assert not (tmp_path / "absent.jsonl").exists()


def test_auto_mode_replays_hits_and_records_misses(tmp_path: Path) -> None:
    inner = CountingLLM()
    cassette = CassetteLLM(inner, tmp_path / "c.jsonl", "auto")

    first = call(cassette, "x")
    second = call(cassette, "x")
    third = call(cassette, "y")

    assert inner.calls == 2
    assert (first.usage.cached_replay, second.usage.cached_replay, third.usage.cached_replay) == (
        False,
        True,
        False,
    )
    assert second.data == first.data
    assert len((tmp_path / "c.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_record_mode_always_calls_inner_and_last_recording_wins(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    inner = CountingLLM()
    cassette = CassetteLLM(inner, path, "record")
    call(cassette, "x")
    call(cassette, "x")
    lines = path.read_text(encoding="utf-8").splitlines()
    newer = json.loads(lines[1])
    newer["response"]["text"] = "newer answer"
    path.write_text(lines[0] + "\n" + json.dumps(newer, ensure_ascii=False) + "\n", encoding="utf-8")

    assert inner.calls == 2
    assert call(CassetteLLM(None, path), "x").data.text == "newer answer"


def test_unreadable_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    record_all(path, ["x"])
    good = path.read_bytes()
    path.write_bytes(b'{"truncated": \n\xff\xfe\n{"no_key": 1}\n\n' + good)

    assert call(CassetteLLM(None, path), "x").data.text == "extract:x"


def test_stale_recording_that_fails_validation_is_an_output_error(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    call(CassetteLLM(FakeLLM(lambda *_: {"text": "t", "count": 0}), path, "record"))
    cassette = CassetteLLM(None, path)

    with pytest.raises(LLMOutputError, match="다시 녹화"):
        run(
            cassette.structured(
                task="extract",
                instructions="요약하세요.",
                schema=StrictSummary,
                documents=DOCS,
                context="명부: 김대리",
            )
        )


def test_constructor_validates_mode_and_inner(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inner"):
        CassetteLLM(None, tmp_path / "c.jsonl", "record")
    with pytest.raises(ValueError, match="inner"):
        CassetteLLM(None, tmp_path / "c.jsonl", "auto")
    with pytest.raises(ValueError, match="모드"):
        CassetteLLM(CountingLLM(), tmp_path / "c.jsonl", "rewind")  # type: ignore[arg-type]


def test_identity_attributes(tmp_path: Path) -> None:
    replay_only = CassetteLLM(None, tmp_path / "c.jsonl")
    recording = CassetteLLM(CountingLLM(), tmp_path / "c.jsonl", "auto")

    assert (replay_only.provider, replay_only.model) == ("cassette", "cassette")
    assert (recording.provider, recording.model) == ("cassette", "claude-opus-5-5")
    assert isinstance(replay_only, LLMClient)


# --------------------------------------------------------------------------- concurrency


def _event_loop() -> asyncio.AbstractEventLoop:
    """New event loop, retried: some sandboxed Windows hosts intermittently fail the
    ``socket.socketpair()`` every loop needs for its self-pipe (WinError 10014)."""
    for _ in range(9):
        try:
            return asyncio.new_event_loop()
        except OSError:
            continue
    return asyncio.new_event_loop()


def test_concurrent_async_recording_keeps_every_line_intact(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    inner = CountingLLM(suspend=True)
    cassette = CassetteLLM(inner, path, "record")

    async def burst() -> list[LLMResult[Summary]]:
        return await asyncio.gather(
            *(
                cassette.structured(task="t", instructions="i", schema=Summary, context=f"ctx-{i}")
                for i in range(30)
            )
        )

    with asyncio.Runner(loop_factory=_event_loop) as runner:
        results = runner.run(burst())

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert inner.calls == 30
    assert [result.data.count for result in results] == [len(f"ctx-{i}") for i in range(30)]
    assert len({record["key"] for record in records}) == 30


def test_concurrent_threads_share_one_cassette_safely(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    cassette = CassetteLLM(CountingLLM(), path, "auto")

    def worker(offset: int) -> None:
        for i in range(10):
            call(cassette, f"thread-{offset}-{i}")

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 40
    assert call(CassetteLLM(None, path), "thread-3-9").data.text == "extract:thread-3-9"


# --------------------------------------------------------------------------- keys


def _key(**overrides: Any) -> str:
    request: dict[str, Any] = {
        "task": "extract",
        "instructions": "요약하세요.",
        "schema": Summary,
        "documents": DOCS,
        "context": "ctx",
    }
    return cassette_key(**(request | overrides))


@pytest.mark.parametrize(
    "override",
    [
        {"task": "backtranslate"},
        {"instructions": "다르게 요약하세요."},
        {"schema": Detailed},
        {"documents": (UntrustedDocument("doc-1", "결제 런북", "1차 담당: 김대리", offset=11),)},
        {"documents": ()},
        {"context": "ctx2"},
    ],
    ids=["task", "instructions", "schema", "offset", "no-docs", "context"],
)
def test_key_changes_with_request_content(override: dict[str, Any]) -> None:
    assert _key(**override) != _key()


def test_key_depends_on_schema_shape_not_class_identity() -> None:
    assert _key(schema=StrictSummary) == _key()


@settings(max_examples=50, deadline=None)
@given(st.text(max_size=40), st.text(max_size=40))
def test_property_key_is_a_pure_function_of_the_request(context: str, text: str) -> None:
    docs = (UntrustedDocument("d", "t", text),)

    assert _key(context=context, documents=docs) == _key(context=context, documents=docs)
    assert len(_key(context=context, documents=docs)) == 64


# --------------------------------------------------------------------------- review: hermetic replay


def test_replay_mode_never_calls_an_attached_inner_client(tmp_path: Path) -> None:
    """P6: ``replay`` must stay hermetic even when a live client is attached — a miss falls back,
    it never reaches the model (nor writes to the cassette)."""
    path = tmp_path / "c.jsonl"
    inner = CountingLLM()

    with pytest.raises(LLMDisabledError, match="카세트에 없는 요청"):
        call(CassetteLLM(inner, path, "replay"))

    assert inner.calls == 0
    assert not path.exists()


def test_replay_mode_with_inner_still_serves_hits(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    record_all(path, ["명부: 김대리"])
    inner = CountingLLM()

    result = call(CassetteLLM(inner, path, "replay"))

    assert result.usage.cached_replay is True
    assert inner.calls == 0


# --------------------------------------------------------------------------- review: what a line stores


class Aliased(BaseModel):
    """A schema whose wire names differ from its attribute names."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(alias="from")
    target: str = Field(alias="to")


class WithComputed(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minutes: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def seconds(self) -> int:
        return self.minutes * 60


def _record_and_replay(path: Path, schema: type[T], payload: dict[str, Any]) -> tuple[T, T]:
    recorder = CassetteLLM(FakeLLM(lambda *_: payload), path, "record")
    recorded = run(recorder.structured(task="t", instructions="i", schema=schema))
    replayed = run(CassetteLLM(None, path).structured(task="t", instructions="i", schema=schema))
    return recorded.data, replayed.data


def test_aliased_fields_round_trip(tmp_path: Path) -> None:
    recorded, replayed = _record_and_replay(tmp_path / "c.jsonl", Aliased, {"from": "결제", "to": "DB"})

    assert replayed == recorded == Aliased.model_validate({"from": "결제", "to": "DB"})
    line = json.loads((tmp_path / "c.jsonl").read_text(encoding="utf-8"))
    assert line["response"] == {"from": "결제", "to": "DB"}  # the names the model emits


def test_computed_fields_are_not_recorded(tmp_path: Path) -> None:
    recorded, replayed = _record_and_replay(tmp_path / "c.jsonl", WithComputed, {"minutes": 10})

    assert replayed == recorded
    assert replayed.seconds == 600
    line = json.loads((tmp_path / "c.jsonl").read_text(encoding="utf-8"))
    assert line["response"] == {"minutes": 10}


def test_hangul_is_stored_as_utf8_and_replays_exactly(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    record_all(path, ["담당: 김대리 → 박팀장 (10분)"])

    raw = path.read_bytes()
    assert "김대리".encode() in raw  # canonical JSON keeps Hangul readable (ensure_ascii=False)
    assert b"\\u" not in raw
    replayed = call(CassetteLLM(None, path), "담당: 김대리 → 박팀장 (10분)")
    assert replayed.data.text == "extract:담당: 김대리 → 박팀장 (10분)"


def test_nfc_and_nfd_hangul_are_different_requests() -> None:
    """Keys are byte-level: normalisation is ingest's job, not something replay guesses at."""
    nfc = unicodedata.normalize("NFC", "김대리")
    nfd = unicodedata.normalize("NFD", "김대리")
    assert nfc != nfd

    assert _key(context=nfc) != _key(context=nfd)


def test_empty_request_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    recorder = CassetteLLM(FakeLLM(lambda *_: {"text": ""}), path, "record")
    run(recorder.structured(task="", instructions="", schema=Summary))

    replayed = run(CassetteLLM(None, path).structured(task="", instructions="", schema=Summary))

    assert replayed.data == Summary(text="")


# --------------------------------------------------------------------------- review: damaged cassettes


@pytest.mark.parametrize(
    "damage",
    [
        {"usage": "many"},
        {"usage": {"input_tokens": "100"}},
        {"usage": {"output_tokens": -1}},
        {"usage": {"cache_read_tokens": True}},
        {"response": ["not", "an", "object"]},
        {"response": None},
        {"key": 42},
    ],
    ids=[
        "usage-str",
        "tokens-str",
        "tokens-negative",
        "tokens-bool",
        "response-list",
        "response-null",
        "key-int",
    ],
)
def test_damaged_lines_are_misses_not_crashes(tmp_path: Path, damage: dict[str, Any]) -> None:
    path = tmp_path / "c.jsonl"
    record_all(path, ["x"])
    record = json.loads(path.read_text(encoding="utf-8")) | damage
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(LLMDisabledError):
        call(CassetteLLM(None, path), "x")


def test_null_usage_replays_with_zero_tokens(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    record_all(path, ["x"])
    record = json.loads(path.read_text(encoding="utf-8")) | {"usage": None}
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    usage = call(CassetteLLM(None, path), "x").usage

    assert (usage.input_tokens, usage.output_tokens, usage.cache_read_tokens) == (0, 0, 0)
    assert usage.cached_replay is True


# --------------------------------------------------------------------------- review: properties


_hangul_text = st.text(
    alphabet=st.one_of(
        st.characters(min_codepoint=0xAC00, max_codepoint=0xD7A3), st.sampled_from(" :→\n\"'\\")
    ),
    max_size=30,
)


@settings(max_examples=40, deadline=None)
@given(contexts=st.lists(_hangul_text, min_size=1, max_size=4, unique=True))
def test_property_record_replay_is_lossless_and_byte_stable(
    tmp_path_factory: Any, contexts: list[str]
) -> None:
    folder = tmp_path_factory.mktemp("cassettes")
    first, second = folder / "a.jsonl", folder / "b.jsonl"
    record_all(first, contexts)
    record_all(second, contexts)

    assert first.read_bytes() == second.read_bytes()
    replay = CassetteLLM(None, first)
    for context in contexts:
        assert call(replay, context).data == Summary(text=f"extract:{context}", count=len(context))


# --------------------------------------------------------------------------- review 2: hostile bytes


def test_lone_surrogate_requests_get_a_key_and_miss_deterministically(tmp_path: Path) -> None:
    """A PDF with a broken ToUnicode map yields lone surrogates; replay must fall back
    (LLMDisabledError), never crash with UnicodeEncodeError."""
    broken = (UntrustedDocument("doc-1", "결제 런북", "담당: 김\ud800대리"),)
    replaced = (UntrustedDocument("doc-1", "결제 런북", "담당: 김�대리"),)

    key = _key(documents=broken, context="명부\udfff")

    assert len(key) == 64
    assert key == _key(documents=broken, context="명부\udfff")
    assert key != _key(documents=replaced, context="명부\udfff")
    with pytest.raises(LLMDisabledError, match="카세트에 없는 요청"):
        run(
            CassetteLLM(None, tmp_path / "c.jsonl").structured(
                task="extract", instructions="i", schema=Summary, documents=broken
            )
        )


def test_lone_surrogate_requests_record_and_replay(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    broken = (UntrustedDocument("doc-1", "t", "김\ud800대리"),)
    recorder = CassetteLLM(FakeLLM(lambda *_: {"text": "김대리"}), path, "record")
    run(recorder.structured(task="extract", instructions="i", schema=Summary, documents=broken))

    replayed = run(
        CassetteLLM(None, path).structured(task="extract", instructions="i", schema=Summary, documents=broken)
    )

    assert replayed.data == Summary(text="김대리")
    path.read_bytes().decode("utf-8")  # the file itself stays valid UTF-8


def test_bom_prefixed_cassette_keeps_its_first_recording(tmp_path: Path) -> None:
    """Windows PowerShell 5.1 writes a BOM when a cassette is concatenated or re-saved."""
    path = tmp_path / "c.jsonl"
    record_all(path, ["first", "second"])
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())

    replay = CassetteLLM(None, path)

    assert call(replay, "first").data.text == "extract:first"
    assert call(replay, "second").data.text == "extract:second"


def test_crlf_cassette_from_a_windows_checkout_replays(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    record_all(path, ["담당: 김대리", "x"])
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))

    replay = CassetteLLM(None, path)

    assert call(replay, "담당: 김대리").data.text == "extract:담당: 김대리"
    assert call(replay, "x").data.count == 1


def test_line_separators_inside_recorded_text_do_not_split_lines(tmp_path: Path) -> None:
    """U+2028/U+0085 survive ``ensure_ascii=False`` unescaped; lines split on LF only."""
    context = "담당 김대리\x85박팀장 "
    path = tmp_path / "c.jsonl"
    record_all(path, [context])

    assert path.read_bytes().count(b"\n") == 1
    assert call(CassetteLLM(None, path), context).data.text == f"extract:{context}"


def test_non_json_number_in_a_recording_is_an_output_error(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    record_all(path, ["x"])
    path.write_bytes(path.read_bytes().replace(b'"count":1', b'"count":NaN'))

    with pytest.raises(LLMOutputError, match="다시 녹화"):
        call(CassetteLLM(None, path), "x")


def test_concatenated_bom_files_keep_every_recording(tmp_path: Path) -> None:
    first, second, merged = tmp_path / "a.jsonl", tmp_path / "b.jsonl", tmp_path / "merged.jsonl"
    record_all(first, ["first"])
    record_all(second, ["second"])
    merged.write_bytes(b"\xef\xbb\xbf" + first.read_bytes() + b"\xef\xbb\xbf" + second.read_bytes())

    replay = CassetteLLM(None, merged)

    assert call(replay, "first").data.text == "extract:first"
    assert call(replay, "second").data.text == "extract:second"


def test_recording_after_an_unterminated_last_line_keeps_both_records(tmp_path: Path) -> None:
    """A hand-edited cassette often loses its final newline; appending must not merge lines."""
    path = tmp_path / "c.jsonl"
    record_all(path, ["edited by hand"])
    path.write_bytes(path.read_bytes().rstrip(b"\n"))

    call(CassetteLLM(CountingLLM(), path, "auto"), "new")

    replay = CassetteLLM(None, path)
    assert call(replay, "edited by hand").data.text == "extract:edited by hand"
    assert call(replay, "new").data.text == "extract:new"
    assert path.read_bytes().count(b"\n") == 2


_json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats() | st.text(max_size=6),
    lambda children: (
        st.lists(children, max_size=3)
        | st.dictionaries(st.sampled_from(["text", "count", "input_tokens", "x"]), children, max_size=3)
    ),
    max_leaves=12,
)


@settings(max_examples=80, deadline=None)
@given(response=_json_values, usage=_json_values, model=_json_values)
def test_property_tampered_recordings_never_crash_replay(
    tmp_path_factory: Any, response: Any, usage: Any, model: Any
) -> None:
    """Whatever a hand edit leaves on a line with a matching key, replay answers, misses
    (fallback) or reports a stale recording — it never raises an untyped error."""
    path = tmp_path_factory.mktemp("tampered") / "c.jsonl"
    record_all(path, ["x"])
    record = json.loads(path.read_text(encoding="utf-8")) | {
        "response": response,
        "usage": usage,
        "model": model,
    }
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    try:
        result = call(CassetteLLM(None, path), "x")
    except (LLMDisabledError, LLMOutputError):
        return
    assert isinstance(result.data, Summary)
    assert result.usage.cached_replay is True
