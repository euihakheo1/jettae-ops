"""Resuming and extending a study in one output folder: settings, inputs and the spend ledger.

One ``--out`` folder holds one study. Its files make continuing — or growing — it safe and exact:

* ``run.json`` — the settings the results belong to (models, effort …). A run with other settings
  is refused: it would mix two studies. It is written when a paid run starts, never by a dry run
  or a refused start.
* ``results.jsonl`` — the finished items (the resume point). An item there whose content is
  unchanged is never processed or paid for again.
* ``manifest.json`` — every input folder used and every item seen (id, content hash, where it was
  first seen). A run may point at another folder — typically the full set after a subset — when
  every *finished* item is still in it with the same content (a superset): only the new and
  unfinished items are processed, and the metrics are recomputed over all of them. A finished
  item that disappeared or changed is refused unless ``allow_input_change``: then a changed item
  is processed again and the superseded results move to ``results.superseded.jsonl`` (paid
  results are never deleted).
* ``spend.json`` — the spend ledger: actual spend per model (input / output / cache tokens, USD),
  per item (with its last status) and per run, for every run in this folder. It is rewritten
  after every metered call and every reservation, so a killed run still leaves its spend behind;
  calls that were in flight when a run died are booked at their reservation on the next load
  (never under-count). ``--max-usd`` is a **cumulative** cap over this ledger; ``--max-usd-run``
  optionally caps a single invocation.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from jettae.evalkit.realdata.common import write_json
from jettae.evalkit.realdata.providers import SpendLine, SpendReport, SpendTracker

__all__ = [
    "INTERRUPTED_LINE",
    "LEDGER_FILE",
    "LEGACY_LINE",
    "MANIFEST_FILE",
    "SETTINGS_FILE",
    "SUPERSEDED_FILE",
    "Finished",
    "InputChangeError",
    "InputCheck",
    "ItemSpend",
    "LedgerData",
    "Manifest",
    "ResumeState",
    "RunSpend",
    "SpendLedger",
    "StopKind",
    "check_inputs",
    "check_settings",
    "combine_lines",
    "now_iso",
    "prepare_resume",
    "read_manifest",
    "record_inputs",
    "resume_lines",
    "save_settings",
    "shown_dir",
    "start_run",
    "stop_kind_of",
]

SETTINGS_FILE: Final = "run.json"
MANIFEST_FILE: Final = "manifest.json"
LEDGER_FILE: Final = "spend.json"
SUPERSEDED_FILE: Final = "results.superseded.jsonl"
#: Spend line of calls that were in flight when a run was killed (booked at their reservation).
INTERRUPTED_LINE: Final = "(중단된 실행의 진행 중 호출: 예약액으로 계상)"
#: Spend line imported from ``results.jsonl`` of a folder written before the ledger existed.
LEGACY_LINE: Final = "(원장 이전 실행: results.jsonl의 완료 항목 비용)"
_LEGACY_NOTE: Final = (
    "spend.json이 없던 이전 실행의 지출은 results.jsonl에 적힌 완료 항목 비용만 가져왔습니다 "
    "(그때 실패·중단된 항목의 지출은 알 수 없어 빠져 있을 수 있음)."
)
_SHOWN_IDS: Final = 5
_STRICT_RETRIES: Final = 5

#: Why a run stopped early: a cap (``budget_total`` = cumulative ``--max-usd``, ``budget_run`` =
#: ``--max-usd-run``) or too many failures in a row.
StopKind = Literal["budget_total", "budget_run", "failures"]
ItemStatus = Literal["done", "failed", "stopped", "running", "interrupted"]

_log = logging.getLogger(__name__)


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True)


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def shown_dir(path: Path) -> str:
    """An input folder as recorded and compared: absolute, forward slashes."""
    try:
        return path.resolve().as_posix()
    except OSError:
        return path.absolute().as_posix()


def stop_kind_of(scope: str) -> StopKind:
    return "budget_run" if scope == "run" else "budget_total"


def _ids(ids: Sequence[str]) -> str:
    head = ", ".join(ids[:_SHOWN_IDS])
    return head + (f" 외 {len(ids) - _SHOWN_IDS}건" if len(ids) > _SHOWN_IDS else "")


# --------------------------------------------------------------------------- settings


def check_settings(out: Path, settings: Mapping[str, Any]) -> None:
    """Refuse a folder whose results were made with other settings (models, effort …)."""
    path = out / SETTINGS_FILE
    if not path.is_file():
        return
    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored != dict(settings):
        changed = sorted(k for k in set(stored) | set(settings) if stored.get(k) != settings.get(k))
        raise ValueError(
            f"이 결과 폴더는 다른 설정({', '.join(changed)})으로 만든 결과입니다. "
            "이어서 실행하려면 같은 설정을 쓰거나 --out을 바꾸세요."
        )


def save_settings(out: Path, settings: Mapping[str, Any]) -> None:
    check_settings(out, settings)
    path = out / SETTINGS_FILE
    if not path.is_file():
        write_json(path, dict(settings))


# --------------------------------------------------------------------------- inputs


class InputUse(_Model):
    dir: str
    first_used: str
    last_used: str
    runs: int = 1
    items: int
    commit: str | None = None


class SeenItem(_Model):
    sha256: str
    path: str
    #: the input folder it was (last) taken from
    input_dir: str


class Manifest(_Model):
    study: str
    #: input folders, least recently used first (the last one is the previous run's)
    inputs: tuple[InputUse, ...] = ()
    items: dict[str, SeenItem] = Field(default_factory=dict)
    #: how the study is set up beyond its model settings (jurisdiction, dev/test split, coding
    #: rubric) as the last paid run recorded it — the report's "what was measured, under which rules"
    design: dict[str, Any] = Field(default_factory=dict)


class InputChangeError(ValueError):
    """Finished items are missing from, or changed in, this run's input (Korean message)."""


def read_manifest(out: Path, study: str) -> Manifest:
    path = out / MANIFEST_FILE
    if not path.is_file():
        return Manifest(study=study)
    manifest = Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    if manifest.study != study:
        raise ValueError(f"이 결과 폴더는 다른 실증({manifest.study})의 결과입니다. --out을 바꾸세요.")
    return manifest


@dataclass(frozen=True)
class Finished:
    """A finished item as ``results.jsonl`` records it."""

    sha256: str
    usd: float


@dataclass(frozen=True)
class InputCheck:
    input_dir: str
    #: input folders of earlier runs (manifest order), this one included if used before
    previous_dirs: tuple[str, ...]
    #: finished items present with the same content (never processed again)
    done: tuple[str, ...]
    #: items this run processes, in input order
    todo: tuple[str, ...]
    #: items of ``todo`` never seen in this folder before
    new: tuple[str, ...]
    #: finished items not in this input / with other content (only with ``allow_input_change``)
    vanished: tuple[str, ...]
    changed: tuple[str, ...]

    @property
    def dir_changed(self) -> bool:
        return bool(self.previous_dirs) and self.previous_dirs[-1] != self.input_dir


def check_inputs(
    manifest: Manifest,
    *,
    input_dir: Path,
    current: Sequence[tuple[str, str]],
    finished: Mapping[str, Finished],
    allow_change: bool,
    item_word: str,
) -> InputCheck:
    """Which items to process; refuse a finished item that disappeared or changed (module doc)."""
    shown = shown_dir(input_dir)
    content = dict(current)
    vanished = sorted(i for i in finished if i not in content)
    changed = sorted(i for i, f in finished.items() if i in content and content[i] != f.sha256)
    previous = tuple(use.dir for use in manifest.inputs)
    if (vanished or changed) and not allow_change:
        problems = []
        if vanished:
            problems.append(f"이번 입력에 없음 {len(vanished)}건({_ids(vanished)})")
        if changed:
            problems.append(f"내용이 바뀜 {len(changed)}건({_ids(changed)})")
        earlier = ", ".join(previous) or "기록 없음"
        raise InputChangeError(
            f"이 결과 폴더(--out)에서 이미 끝낸 {item_word}이 이번 입력과 맞지 않습니다: "
            f"{'; '.join(problems)}. 이전 입력: {earlier} · 이번 입력: {shown}. "
            "이어서 하거나 늘리려면 이전에 끝낸 항목을 모두 같은 내용으로 담은 폴더(상위 집합)를 지정하세요"
            "(--limit을 줄였다면 되돌리세요). 의도한 변경이면 --allow-input-change를 붙이세요 — "
            "바뀐 항목은 다시 처리(과금)하고, 빠지거나 바뀐 항목의 이전 결과는 "
            f"{SUPERSEDED_FILE}로 옮겨 지표에서 뺍니다."
        )
    done = tuple(i for i, sha in current if i in finished and finished[i].sha256 == sha)
    done_set = set(done)
    todo = tuple(i for i, _ in current if i not in done_set)
    new = tuple(i for i in todo if i not in manifest.items)
    return InputCheck(
        input_dir=shown,
        previous_dirs=previous,
        done=done,
        todo=todo,
        new=new,
        vanished=tuple(vanished),
        changed=tuple(changed),
    )


def record_inputs(
    out: Path,
    manifest: Manifest,
    *,
    input_dir: Path,
    items: Sequence[tuple[str, str, str]],
    when: str,
    commit: str | None = None,
    design: Mapping[str, Any] | None = None,
) -> Manifest:
    """Add this run's input folder and its items ``(id, sha256, path)`` to ``manifest.json``
    (and, when given, the study design the run used)."""
    shown = shown_dir(input_dir)
    uses = [use for use in manifest.inputs if use.dir != shown]
    old = next((use for use in manifest.inputs if use.dir == shown), None)
    if old is None:
        uses.append(InputUse(dir=shown, first_used=when, last_used=when, items=len(items), commit=commit))
    else:  # moved to the end: the list stays in order of last use
        uses.append(
            old.model_copy(
                update={"last_used": when, "runs": old.runs + 1, "items": len(items), "commit": commit}
            )
        )
    seen = dict(manifest.items)
    for item_id, sha, path in items:
        known = seen.get(item_id)
        if known is None or known.sha256 != sha:
            seen[item_id] = SeenItem(sha256=sha, path=path, input_dir=shown)
    update: dict[str, Any] = {"inputs": tuple(uses), "items": seen}
    if design is not None:
        update["design"] = dict(design)
    updated = manifest.model_copy(update=update)
    write_json(out / MANIFEST_FILE, updated.model_dump(mode="json"))
    return updated


# --------------------------------------------------------------------------- spend ledger


_COUNT_FIELDS: Final = ("calls", "input_tokens", "cache_write_tokens", "cache_read_tokens", "output_tokens")


def combine_lines(*groups: Iterable[SpendLine], subtract: Iterable[SpendLine] = ()) -> tuple[SpendLine, ...]:
    """Per-model sum of ``groups`` minus ``subtract`` (models left with nothing are dropped)."""
    counts: dict[str, dict[str, int]] = {}
    usd: dict[str, float] = {}

    def add(line: SpendLine, sign: int) -> None:
        slot = counts.setdefault(line.model, dict.fromkeys(_COUNT_FIELDS, 0))
        for name in _COUNT_FIELDS:
            slot[name] += sign * int(getattr(line, name))
        usd[line.model] = usd.get(line.model, 0.0) + sign * line.usd

    for group in groups:
        for line in group:
            add(line, 1)
    for line in subtract:
        add(line, -1)
    return tuple(
        SpendLine(model=model, usd=round(usd[model], 6), **slot)
        for model, slot in sorted(counts.items())
        if any(slot.values()) or round(usd[model], 6) != 0
    )


def _usd(lines: Iterable[SpendLine]) -> float:
    return round(sum(line.usd for line in lines), 6)


class ItemSpend(_Model):
    usd: float = 0.0
    calls: int = 0
    by_model: tuple[SpendLine, ...] = ()
    #: runs that started this item
    attempts: int = 0
    status: ItemStatus = "running"


class RunSpend(_Model):
    run: int
    started_at: str
    finished_at: str | None = None
    input_dir: str
    max_usd: float | None
    max_usd_run: float | None
    #: spent in this folder before this run (what the cumulative cap counted against it)
    prior_usd: float
    spent_usd: float = 0.0
    #: reservations of calls in flight at the last write (booked if the run was killed)
    in_flight_usd: float = 0.0
    items_done: int = 0
    stop_reason: str | None = None
    interrupted: bool = False


class LedgerData(_Model):
    study: str
    version: int = 1
    #: every dollar spent by every run in this folder (cumulative; what ``--max-usd`` caps)
    spent_usd: float = 0.0
    by_model: tuple[SpendLine, ...] = ()
    items: dict[str, ItemSpend] = Field(default_factory=dict)
    runs: tuple[RunSpend, ...] = ()
    notes: tuple[str, ...] = ()


def _close_interrupted(data: LedgerData, when: str) -> LedgerData:
    """A run that never finished was killed: book its calls in flight at their reservation."""
    open_runs = [r for r in data.runs if r.finished_at is None]
    if not open_runs:
        return data
    booked = round(sum(r.in_flight_usd for r in open_runs), 6)
    runs = tuple(
        r.model_copy(
            update={"finished_at": when, "interrupted": True, "spent_usd": r.spent_usd + r.in_flight_usd}
        )
        if r.finished_at is None
        else r
        for r in data.runs
    )
    items = {
        key: item.model_copy(update={"status": "interrupted"}) if item.status == "running" else item
        for key, item in data.items.items()
    }
    by_model = data.by_model
    if booked > 0:
        line = SpendLine(
            model=INTERRUPTED_LINE,
            calls=0,
            input_tokens=0,
            cache_write_tokens=0,
            cache_read_tokens=0,
            output_tokens=0,
            usd=booked,
        )
        by_model = combine_lines(by_model, [line])
    return data.model_copy(
        update={"runs": runs, "items": items, "by_model": by_model, "spent_usd": _usd(by_model)}
    )


def _legacy(study: str, legacy: Mapping[str, float]) -> LedgerData:
    def line(usd: float) -> SpendLine:
        return SpendLine(
            model=LEGACY_LINE,
            calls=0,
            input_tokens=0,
            cache_write_tokens=0,
            cache_read_tokens=0,
            output_tokens=0,
            usd=round(usd, 6),
        )

    items = {
        key: ItemSpend(usd=round(usd, 6), by_model=(line(usd),), attempts=1, status="done")
        for key, usd in legacy.items()
    }
    total = round(sum(legacy.values()), 6)
    return LedgerData(
        study=study,
        spent_usd=total,
        by_model=(line(total),) if total else (),
        items=items,
        notes=(_LEGACY_NOTE,),
    )


class SpendLedger:
    """``spend.json`` of one output folder: earlier runs (loaded) plus the running one (live).

    A run is attached with :meth:`begin_run`; from then on every change of its
    :class:`~jettae.evalkit.realdata.providers.SpendTracker` rewrites the file, and
    :meth:`begin_item` / :meth:`end_item` attribute spend to items.
    """

    def __init__(self, path: Path, data: LedgerData, *, legacy: bool = False) -> None:
        self.path = path
        self.legacy = legacy
        self._base = data
        self._tracker: SpendTracker | None = None
        self._run: RunSpend | None = None
        self._run_items: dict[str, tuple[tuple[SpendLine, ...], ItemStatus]] = {}
        self._current: tuple[str, SpendReport] | None = None
        self._items_done = 0
        self._lock = threading.RLock()

    @classmethod
    def load(
        cls, out: Path, *, study: str, legacy: Mapping[str, float] | None = None, when: str | None = None
    ) -> SpendLedger:
        """The ledger of ``out`` (``legacy`` = item costs from ``results.jsonl``, used only when a
        folder has results but no ledger yet)."""
        path = out / LEDGER_FILE
        if path.is_file():
            data = LedgerData.model_validate_json(path.read_text(encoding="utf-8"))
            if data.study != study:
                raise ValueError(f"이 결과 폴더는 다른 실증({data.study})의 결과입니다. --out을 바꾸세요.")
            return cls(path, _close_interrupted(data, when or now_iso()))
        if legacy:
            return cls(path, _legacy(study, legacy), legacy=True)
        return cls(path, LedgerData(study=study))

    # ----------------------------------------------------------------- reading

    @property
    def prior_usd(self) -> float:
        """Spent in this folder before the attached run."""
        return self._base.spent_usd

    @property
    def spent_usd(self) -> float:
        """Spent in this folder so far, the attached run included."""
        return self.data().spent_usd

    def data(self) -> LedgerData:
        """The ledger as it stands (what :meth:`flush` writes)."""
        with self._lock:
            base = self._base
            tracker = self._tracker
            if tracker is None or self._run is None:
                return base
            report = tracker.report()
            by_model = combine_lines(base.by_model, report.by_model)
            items = dict(base.items)
            attempts: dict[str, tuple[tuple[SpendLine, ...], ItemStatus]] = dict(self._run_items)
            if self._current is not None:
                key, start = self._current
                attempts[key] = (combine_lines(report.by_model, subtract=start.by_model), "running")
            for key, (lines, status) in attempts.items():
                old = items.get(key, ItemSpend())
                merged = combine_lines(old.by_model, lines)
                items[key] = ItemSpend(
                    usd=_usd(merged),
                    calls=sum(line.calls for line in merged),
                    by_model=merged,
                    attempts=old.attempts + 1,
                    status=status,
                )
            run = self._run.model_copy(
                update={
                    "spent_usd": report.spent_usd,
                    "in_flight_usd": round(tracker.reserved_usd, 6),
                    "items_done": self._items_done,
                }
            )
            return base.model_copy(
                update={
                    "spent_usd": _usd(by_model),
                    "by_model": by_model,
                    "items": items,
                    "runs": (*base.runs, run),
                }
            )

    # ----------------------------------------------------------------- the running run

    def begin_run(self, tracker: SpendTracker, *, input_dir: Path, when: str | None = None) -> None:
        """Attach this run's (fresh) tracker. Its ``max_usd`` now counts what this folder already
        spent (``prior_usd``): the cap is cumulative over the folder, ``max_usd_run`` per run."""
        if tracker.used:
            raise ValueError(
                "이미 사용한 SpendTracker로는 새 실행을 시작할 수 없습니다(지출 이중 계산 방지)."
            )
        with self._lock:
            tracker.prior_usd = self._base.spent_usd
            self._tracker = tracker
            self._run = RunSpend(
                run=len(self._base.runs) + 1,
                started_at=when or now_iso(),
                input_dir=shown_dir(input_dir),
                max_usd=tracker.max_usd,
                max_usd_run=tracker.max_usd_run,
                prior_usd=self._base.spent_usd,
            )
        tracker.on_change = self._on_change
        self.flush(strict=True)

    def begin_item(self, item_id: str) -> None:
        with self._lock:
            if self._tracker is None:
                return
            self._current = (item_id, self._tracker.report())
        self.flush(strict=False)

    def end_item(self, item_id: str, status: ItemStatus) -> None:
        """Close the current item (``done`` / ``failed`` / ``stopped``) and persist."""
        with self._lock:
            if self._tracker is None or self._current is None or self._current[0] != item_id:
                return
            start = self._current[1]
            lines = combine_lines(self._tracker.report().by_model, subtract=start.by_model)
            self._run_items[item_id] = (lines, status)
            self._current = None
            if status == "done":
                self._items_done += 1
        self.flush(strict=True)

    def finish_run(self, *, stop_reason: str | None, when: str | None = None) -> None:
        """Close the run (an item still open was interrupted) and persist."""
        with self._lock:
            if self._tracker is None or self._run is None:
                return
            if self._current is not None:
                key, start = self._current
                lines = combine_lines(self._tracker.report().by_model, subtract=start.by_model)
                self._run_items[key] = (lines, "interrupted")
                self._current = None
            self._run = self._run.model_copy(
                update={"finished_at": when or now_iso(), "stop_reason": stop_reason}
            )
        self.flush(strict=True)
        with self._lock:
            if self._tracker is not None:
                self._tracker.on_change = None

    def _on_change(self) -> None:
        self.flush(strict=False)

    def flush(self, *, strict: bool) -> None:
        """Rewrite ``spend.json`` atomically. Per-call writes (``strict=False``) only log a failure
        (the next write catches up); item boundaries retry briefly, then raise — a run must not go
        on spending what it cannot record."""
        with self._lock:
            payload = self.data().model_dump(mode="json")
            attempts = _STRICT_RETRIES if strict else 1
            for attempt in range(attempts):
                try:
                    write_json(self.path, payload)
                    return
                except OSError:
                    if attempt + 1 >= attempts:
                        if strict:
                            raise
                        _log.warning(
                            "spend ledger write failed path=%s (retried at the next write)", self.path
                        )
                        return
                    time.sleep(0.1 * (attempt + 1))


# --------------------------------------------------------------------------- one call for both studies


@dataclass(frozen=True)
class ResumeState:
    """What a run of a study in ``out`` starts from (shared by the guard and the run)."""

    inputs: InputCheck
    manifest: Manifest
    ledger: SpendLedger
    finished: Mapping[str, Finished]


def prepare_resume(
    out: Path,
    *,
    study: str,
    settings: Mapping[str, Any],
    input_dir: Path,
    current: Sequence[tuple[str, str]],
    finished: Mapping[str, Finished],
    allow_input_change: bool,
    item_word: str,
) -> ResumeState:
    """Check settings and inputs (raises ``ValueError`` / :class:`InputChangeError`) and load the
    ledger. Writes nothing: a dry run or a refused start leaves the folder as it was."""
    check_settings(out, settings)
    manifest = read_manifest(out, study)
    inputs = check_inputs(
        manifest,
        input_dir=input_dir,
        current=current,
        finished=finished,
        allow_change=allow_input_change,
        item_word=item_word,
    )
    legacy = {key: f.usd for key, f in finished.items()}
    ledger = SpendLedger.load(out, study=study, legacy=legacy)
    return ResumeState(inputs=inputs, manifest=manifest, ledger=ledger, finished=finished)


def start_run(
    state: ResumeState,
    out: Path,
    *,
    settings: Mapping[str, Any],
    input_dir: Path,
    items: Sequence[tuple[str, str, str]],
    tracker: SpendTracker,
    commit: str | None = None,
    design: Mapping[str, Any] | None = None,
) -> SpendLedger:
    """A paid run starts: pin the settings, record the input folder and its items ``(id, sha256,
    path)`` (and the study ``design``) in the manifest, attach ``tracker`` to the ledger (cumulative
    cap). Returns the ledger."""
    started = now_iso()
    save_settings(out, settings)
    record_inputs(
        out, state.manifest, input_dir=input_dir, items=items, when=started, commit=commit, design=design
    )
    state.ledger.begin_run(tracker, input_dir=input_dir, when=started)
    return state.ledger


def resume_lines(
    state: ResumeState,
    *,
    item_word: str,
    estimate_usd: float,
    max_usd: float | None,
    max_usd_run: float | None,
) -> list[str]:
    """The Korean lines a run (or dry run) prints before it starts: what is done, what is left,
    what was spent in this folder, the estimate for what is left and the budget that remains."""
    inputs = state.inputs
    lines: list[str] = []
    if inputs.done:
        lines.append(
            f"이어서 실행(같은 --out): 이미 끝난 {item_word} {len(inputs.done)}건은 results.jsonl에 있어 "
            f"다시 하지 않습니다. 남은 {len(inputs.todo)}건(처음 보는 {len(inputs.new)}건 포함)만 "
            "처리·과금합니다."
        )
    if inputs.dir_changed:
        lines.append(
            f"입력 폴더가 이전 실행과 다릅니다: 이전 {inputs.previous_dirs[-1]} → 이번 {inputs.input_dir}. "
            f"끝난 {len(inputs.done)}건이 모두 같은 내용으로 들어 있어(상위 집합) 계속하며, "
            f"두 폴더를 모두 {MANIFEST_FILE}에 기록합니다. 지표는 이번 입력 전체로 다시 계산합니다."
        )
    if inputs.vanished:
        lines.append(
            f"주의(--allow-input-change): 끝난 {item_word} {len(inputs.vanished)}건이 이번 입력에 없어 "
            f"그 결과를 {SUPERSEDED_FILE}로 옮기고 지표에서 뺍니다: {_ids(inputs.vanished)}"
        )
    if inputs.changed:
        lines.append(
            f"주의(--allow-input-change): 내용이 바뀐 {item_word} {len(inputs.changed)}건을 "
            f"다시 처리(과금)합니다: {_ids(inputs.changed)}"
        )
    if state.ledger.legacy:
        lines.append(f"주의: {_LEGACY_NOTE}")
    spent = state.ledger.prior_usd
    head = (
        f"지출(이 --out 폴더 누적, {LEDGER_FILE}): ${spent:,.4f} · "
        f"남은 {len(inputs.todo)}건 추정: ${estimate_usd:,.2f} · "
    )
    if max_usd is None:
        lines.append(head + "남은 예산: --max-usd(누적 상한) 미지정 — 남은 예산 = --max-usd − 누적 지출")
    else:
        remaining = max_usd - spent
        lines.append(
            head + f"남은 예산: ${remaining:,.2f} "
            f"(= 누적 상한 --max-usd ${max_usd:,.2f} − 누적 지출 ${spent:,.4f})"
        )
    if max_usd_run is not None:
        lines.append(
            f"이번 실행 상한 --max-usd-run ${max_usd_run:,.2f}: 이번 실행은 이 금액에서도 멈춥니다"
            "(다시 실행하면 끝난 항목을 건너뛰고 이어서 함)."
        )
    return lines
