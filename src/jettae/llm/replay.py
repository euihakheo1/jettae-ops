"""Cassette record/replay of LLM calls — deterministic demos and regression tests (P6).

A cassette is a JSONL file; each line is one recorded call::

    {"key": ..., "model": ..., "provider": ..., "response": {...}, "task": ..., "usage": {...}}

The key is the SHA-256 of the canonical request (task, instructions, strict schema, documents,
context). It deliberately excludes provider/model so a cassette recorded with one provider
replays under any configuration, and excludes ``max_tokens`` (a budget, not content). Lines are
canonical JSON (sorted keys, LF endings, no wall-clock latency), so recording the same responses
in the same order twice yields byte-identical files. Concurrent recording appends in completion
order; replay is unaffected because lookups go by key, never by position.

The recorded ``response`` is the validated object dumped the way the model would have to emit
it (field aliases, no computed fields), so every line replays through the same parser as a live
reply.

In ``replay`` mode a miss raises :class:`LLMDisabledError` — the same signal as LLM OFF — so
callers take their deterministic fallback instead of silently calling a live model. That holds
even when an ``inner`` client is attached: replay never reaches the network.
"""

from __future__ import annotations

import codecs
import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, BinaryIO, Final, Literal, TypeVar, get_args

from pydantic import BaseModel

from jettae.llm.base import (
    LLMClient,
    LLMDisabledError,
    LLMOutputError,
    LLMResult,
    LLMUsage,
    UntrustedDocument,
)
from jettae.llm.schema import parse_output, to_strict_json_schema

_T = TypeVar("_T", bound=BaseModel)

CassetteMode = Literal["replay", "record", "auto"]

CASSETTE_PROVIDER: Final = "cassette"

_log = logging.getLogger(__name__)


def canonical_json(value: Any) -> str:
    """The one serialisation used for keys and cassette lines (stable across runs and OSes)."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def cassette_key(
    *,
    task: str,
    instructions: str,
    schema: type[BaseModel],
    documents: tuple[UntrustedDocument, ...],
    context: str,
) -> str:
    """SHA-256 of the canonical request; identical requests share one recording.

    ``surrogatepass`` keeps the key a total, injective function of the request: a lone
    surrogate from a damaged PDF must give a cassette miss (deterministic fallback), not a crash.
    """
    request = {
        "task": task,
        "instructions": instructions,
        "schema": to_strict_json_schema(schema),
        "documents": [
            {"doc_id": doc.doc_id, "title": doc.title, "text": doc.text, "offset": doc.offset}
            for doc in documents
        ],
        "context": context,
    }
    return hashlib.sha256(canonical_json(request).encode("utf-8", "surrogatepass")).hexdigest()


class CassetteLLM:
    """Replays recorded responses; records calls to ``inner`` in ``record``/``auto`` mode.

    * ``replay`` — hit → recorded response (``usage.cached_replay=True``); miss → ``LLMDisabledError``.
    * ``record`` — always call ``inner`` and append the result (a later line wins on replay).
    * ``auto``   — replay on hit, otherwise call ``inner`` and record.
    """

    provider: str = CASSETTE_PROVIDER

    def __init__(self, inner: LLMClient | None, path: Path, mode: CassetteMode = "replay") -> None:
        if mode not in get_args(CassetteMode):
            raise ValueError(f"지원하지 않는 카세트 모드입니다: {mode!r}")
        if inner is None and mode != "replay":
            raise ValueError(f"카세트 '{mode}' 모드에는 실제 LLM 클라이언트(inner)가 필요합니다")
        self.inner = inner
        self.path = Path(path)
        self.mode: CassetteMode = mode
        self.model = inner.model if inner is not None else CASSETTE_PROVIDER
        self._lock = threading.Lock()
        self._index: dict[str, dict[str, Any]] | None = None

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
        key = cassette_key(
            task=task, instructions=instructions, schema=schema, documents=documents, context=context
        )
        if self.mode != "record":
            record = self._lookup(key)
            if record is not None:
                return _replayed(record, schema=schema, task=task)
        if self.mode == "replay" or self.inner is None:
            # Replay must stay hermetic (P6): a miss never falls through to a live model, even
            # when one is attached. The constructor guarantees ``inner`` for record/auto.
            _log.info("cassette miss task=%s key=%s", task, key[:12])
            raise LLMDisabledError(f"카세트에 없는 요청 (task={task}, key={key[:12]})")
        result = await self.inner.structured(
            task=task,
            instructions=instructions,
            schema=schema,
            documents=documents,
            context=context,
            max_tokens=max_tokens,
        )
        self._append(key, task=task, result=result)
        return result

    def _lookup(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            return self._load_index().get(key)

    def _append(self, key: str, *, task: str, result: LLMResult[Any]) -> None:
        record = {
            "key": key,
            "task": task,
            "provider": result.usage.provider,
            "model": result.usage.model,
            # Aliases + round_trip (which drops computed fields) = the shape the strict schema
            # asks the model for, so the line validates again on replay.
            "response": result.data.model_dump(mode="json", by_alias=True, round_trip=True),
            "usage": {
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "cache_read_tokens": result.usage.cache_read_tokens,
            },
        }
        line = (canonical_json(record) + "\n").encode("utf-8")
        with self._lock:
            index = self._load_index()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a+b") as handle:
                handle.write(_line_start(handle) + line)
            index[key] = record

    def _load_index(self) -> dict[str, dict[str, Any]]:
        """Read the cassette once (caller holds the lock). Unreadable lines are skipped."""
        if self._index is None:
            self._index = _read_cassette(self.path)
        return self._index


def _line_start(handle: BinaryIO) -> bytes:
    """``b"\\n"`` when the file does not end with one (a hand-edited cassette), else ``b""``.

    Appending straight after an unterminated last line would merge two records into one
    invalid line, silently losing both on the next replay.
    """
    size = handle.seek(0, os.SEEK_END)
    if size == 0:
        return b""
    handle.seek(size - 1)
    return b"" if handle.read(1) == b"\n" else b"\n"


def _read_cassette(path: Path) -> dict[str, dict[str, Any]]:
    """Index a cassette by key (a later line wins).

    A UTF-8 BOM at the start of a line is ignored: Windows PowerShell 5.1 (``Set-Content
    -Encoding utf8``, ``Out-File``) writes one when a cassette is re-saved, and concatenating
    such files leaves one mid-file. Either would otherwise silently turn a recording into a miss.
    """
    index: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return index
    for number, raw in enumerate(path.read_bytes().splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw.removeprefix(codecs.BOM_UTF8).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            _log.warning("cassette line skipped (invalid JSON) path=%s line=%d", path, number)
            continue
        if _is_replayable(record):
            index[record["key"]] = record
        else:
            _log.warning("cassette line skipped (malformed record) path=%s line=%d", path, number)
    return index


def _is_replayable(record: Any) -> bool:
    """String key, object response, and integer token counts (when present).

    Checked at load time so a hand-edited or truncated line becomes a miss (deterministic
    fallback) instead of a crash in the middle of a run.
    """
    if not isinstance(record, dict):
        return False
    if not (isinstance(record.get("key"), str) and isinstance(record.get("response"), dict)):
        return False
    usage = record.get("usage")
    if usage is None:
        return True
    return isinstance(usage, dict) and all(
        _is_count(usage.get(name, 0)) for name in ("input_tokens", "output_tokens", "cache_read_tokens")
    )


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _replayed(record: dict[str, Any], *, schema: type[_T], task: str) -> LLMResult[_T]:
    try:
        data = parse_output(schema, canonical_json(record["response"]))
    except ValueError as exc:
        raise LLMOutputError(
            f"카세트 응답이 현재 스키마와 맞지 않습니다 (task={task}). 카세트를 다시 녹화하세요"
        ) from exc
    usage_record: dict[str, int] = record.get("usage") or {}
    usage = LLMUsage(
        provider=CASSETTE_PROVIDER,
        model=str(record.get("model") or CASSETTE_PROVIDER),
        task=task,
        input_tokens=usage_record.get("input_tokens", 0),
        output_tokens=usage_record.get("output_tokens", 0),
        cache_read_tokens=usage_record.get("cache_read_tokens", 0),
        latency_ms=0.0,
        cached_replay=True,
    )
    return LLMResult(data=data, usage=usage)
