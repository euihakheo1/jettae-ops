"""In-memory :class:`~jettae.store.base.Store` for tests, demos and ``memory://`` runs.

It deliberately mirrors :class:`~jettae.store.sql.SqlStore` rather than just keeping objects
in dicts: every record that the SQL store keeps as a JSON payload (inputs, patches, scenarios,
rehearsal results, reports, audit entries, event props) is kept here as the very same JSON
*text* and decoded and re-validated on every read. Serialisation bugs therefore surface in the
fast backend too, and no caller can mutate stored state through an object it got back: frozen
models still hold mutable dicts and lists (``FixHint.params``, audit payloads), so keeping the
decoded objects themselves would let a caller rewrite the append-only audit log.

A single re-entrant lock makes every operation atomic — in particular the head check and the
insert in :meth:`MemoryGraphRepository.append_version` — so the store is safe to share between
threads (e.g. FastAPI's thread pool).
"""

from __future__ import annotations

import json
import threading
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TypeVar

from jettae.domain import DiagnosisReport, Document, GraphVersion, OpsGraph, Patch, RehearsalResult, Scenario
from jettae.store.base import (
    AuditEntry,
    Clock,
    JsonObject,
    NotFoundError,
    Signup,
    StoredDocument,
    StoredInputs,
    Workspace,
    as_utc,
    check_documents,
    check_key_digest,
    check_owner,
    check_patch,
    check_report,
    check_result,
    check_scenario,
    check_workspace_key,
    create_workspace_with,
    decode_graph,
    deletion_cutoff,
    document_order,
    json_text,
    model_json_text,
    patch_order,
    prepare_append,
    rehearsal_id,
    report_order,
    restore_document,
    result_order,
    scenario_order,
    signup_key,
    signup_order,
    stored_document,
    to_json_object,
    utc_now,
    validate_audit,
    validate_event_name,
    version_conflict,
    workspace_order,
)

_K = TypeVar("_K")
_V = TypeVar("_V")


@dataclass
class _State:
    """All tables of one in-memory store, guarded by ``lock``.

    Only immutable values are kept: frozen models without mutable members, strings, and JSON
    text for everything the SQL backend stores in a JSON column.
    """

    lock: threading.RLock = field(default_factory=threading.RLock)
    workspaces: dict[str, Workspace] = field(default_factory=dict)
    documents: dict[tuple[str, str], tuple[StoredDocument, str | None]] = field(default_factory=dict)
    inputs: dict[str, str] = field(default_factory=dict)
    blobs: dict[str, str] = field(default_factory=dict)
    versions: dict[str, list[GraphVersion]] = field(default_factory=dict)
    patches: dict[str, tuple[str, str]] = field(default_factory=dict)
    scenarios: dict[tuple[str, str], str] = field(default_factory=dict)
    results: dict[str, tuple[str, str]] = field(default_factory=dict)
    reports: dict[str, tuple[str, datetime, str]] = field(default_factory=dict)
    #: (workspace id, seq, entry JSON) in append order
    audit: list[tuple[str, int, str]] = field(default_factory=list)
    signups: dict[str, Signup] = field(default_factory=dict)
    events: list[tuple[str, str, datetime]] = field(default_factory=list)
    access_keys: dict[str, str] = field(default_factory=dict)


def _decode(text: str) -> JsonObject:
    """Fresh plain-JSON objects for one read, exactly what the SQL backend's JSON columns return."""
    decoded: JsonObject = json.loads(text)
    return decoded


def _drop_where(table: dict[_K, _V], doomed: Callable[[_K, _V], bool]) -> bool:
    """Remove the entries ``doomed`` selects (caller holds the lock); True when any was removed."""
    keys = [key for key, value in table.items() if doomed(key, value)]
    for key in keys:
        del table[key]
    return bool(keys)


def _delete_workspace(state: _State, workspace_id: str) -> bool:
    """Remove every record of ``workspace_id`` (caller holds the lock); see
    :meth:`~jettae.store.base.WorkspaceRepository.delete_workspace`."""
    history = state.versions.pop(workspace_id, [])
    kept_audit = [entry for entry in state.audit if entry[0] != workspace_id]
    removed = [
        bool(history),
        len(kept_audit) != len(state.audit),
        state.workspaces.pop(workspace_id, None) is not None,
        state.inputs.pop(workspace_id, None) is not None,
        state.access_keys.pop(workspace_id, None) is not None,
        _drop_where(state.documents, lambda key, _: key[0] == workspace_id),
        _drop_where(state.scenarios, lambda key, _: key[0] == workspace_id),
        _drop_where(state.patches, lambda _, entry: entry[0] == workspace_id),
        _drop_where(state.results, lambda _, entry: entry[0] == workspace_id),
        _drop_where(state.reports, lambda _, entry: entry[0] == workspace_id),
    ]
    state.audit[:] = kept_audit
    # Blobs are shared by content: only those no remaining version references go with it.
    in_use = {version.graph_hash for versions in state.versions.values() for version in versions}
    for graph_hash in {version.graph_hash for version in history} - in_use:
        state.blobs.pop(graph_hash, None)
    return any(removed)


class MemoryWorkspaceRepository:
    def __init__(self, state: _State, clock: Clock) -> None:
        self._state = state
        self._clock = clock

    def create_workspace(self, name: str, *, workspace_id: str | None = None) -> Workspace:
        with self._state.lock:
            return create_workspace_with(self._insert, self._clock, name, workspace_id)

    def _insert(self, workspace: Workspace) -> bool:
        if workspace.id in self._state.workspaces:
            return False
        self._state.workspaces[workspace.id] = workspace
        return True

    def get_workspace(self, workspace_id: str) -> Workspace | None:
        with self._state.lock:
            return self._state.workspaces.get(workspace_id)

    def list_workspaces(self) -> list[Workspace]:
        with self._state.lock:
            return sorted(self._state.workspaces.values(), key=workspace_order)

    def delete_workspace(self, workspace_id: str) -> bool:
        check_workspace_key(workspace_id)
        with self._state.lock:
            return _delete_workspace(self._state, workspace_id)


class MemoryDocumentRepository:
    def __init__(self, state: _State) -> None:
        self._state = state

    def save_documents(self, workspace_id: str, docs: Sequence[Document], *, store_raw: bool) -> None:
        # Build every entry before touching the table, so a batch is saved whole or not at all.
        entries = {
            (workspace_id, doc.doc_id): (
                stored_document(workspace_id, doc, raw_stored=store_raw),
                doc.text if store_raw else None,
            )
            for doc in check_documents(workspace_id, docs)
        }
        with self._state.lock:
            self._state.documents.update(entries)

    def list_documents(self, workspace_id: str) -> list[StoredDocument]:
        with self._state.lock:
            metas = [meta for (ws, _), (meta, _) in self._state.documents.items() if ws == workspace_id]
        return sorted(metas, key=document_order)

    def get_document(self, workspace_id: str, doc_id: str) -> Document | None:
        with self._state.lock:
            entry = self._state.documents.get((workspace_id, doc_id))
        if entry is None or entry[1] is None:
            return None
        return restore_document(entry[0], entry[1])


class MemoryInputRepository:
    def __init__(self, state: _State) -> None:
        self._state = state

    def save_inputs(self, workspace_id: str, inputs: StoredInputs) -> None:
        check_workspace_key(workspace_id)
        payload = model_json_text(inputs, "입력 자료")
        with self._state.lock:
            self._state.inputs[workspace_id] = payload

    def load_inputs(self, workspace_id: str) -> StoredInputs:
        with self._state.lock:
            payload = self._state.inputs.get(workspace_id)
        return StoredInputs() if payload is None else StoredInputs.model_validate(_decode(payload))


class MemoryGraphRepository:
    def __init__(self, state: _State) -> None:
        self._state = state

    def head(self, workspace_id: str) -> GraphVersion | None:
        with self._state.lock:
            history = self._state.versions.get(workspace_id)
            return history[-1] if history else None

    def get_version(self, workspace_id: str, number: int) -> GraphVersion:
        with self._state.lock:
            for version in self._state.versions.get(workspace_id, []):
                if version.number == number:
                    return version
        raise NotFoundError(f"워크스페이스 {workspace_id}에 그래프 버전 {number}이(가) 없습니다.")

    def list_versions(self, workspace_id: str) -> list[GraphVersion]:
        with self._state.lock:
            return list(self._state.versions.get(workspace_id, []))

    def load_graph(self, graph_hash: str) -> OpsGraph:
        with self._state.lock:
            blob = self._state.blobs.get(graph_hash)
        if blob is None:
            raise NotFoundError(f"그래프 {graph_hash[:12]}…을(를) 찾을 수 없습니다.")
        return decode_graph(graph_hash, blob)

    def append_version(self, version: GraphVersion, graph: OpsGraph, *, expected_head: int | None) -> None:
        normalized, blob = prepare_append(version, graph, expected_head)
        with self._state.lock:
            history = self._state.versions.setdefault(version.workspace_id, [])
            current = history[-1].number if history else None
            if current != expected_head:
                raise version_conflict(version.workspace_id, expected_head, current)
            self._state.blobs.setdefault(normalized.graph_hash, blob)
            history.append(normalized)


class MemoryPatchRepository:
    def __init__(self, state: _State) -> None:
        self._state = state

    def save_patch(self, patch: Patch) -> None:
        check_patch(patch)
        payload = model_json_text(patch, "수정안")
        with self._state.lock:
            existing = self._state.patches.get(patch.id)
            check_owner("수정안", patch.id, None if existing is None else existing[0], patch.workspace_id)
            self._state.patches[patch.id] = (patch.workspace_id, payload)

    def get_patch(self, patch_id: str) -> Patch:
        with self._state.lock:
            entry = self._state.patches.get(patch_id)
        if entry is None:
            raise NotFoundError(f"수정안 {patch_id}을(를) 찾을 수 없습니다.")
        return Patch.model_validate(_decode(entry[1]))

    def list_patches(self, workspace_id: str) -> list[Patch]:
        with self._state.lock:
            payloads = [payload for ws, payload in self._state.patches.values() if ws == workspace_id]
        return sorted((Patch.model_validate(_decode(p)) for p in payloads), key=patch_order)


class MemoryRehearsalRepository:
    def __init__(self, state: _State) -> None:
        self._state = state

    def save_scenario(self, workspace_id: str, scenario: Scenario) -> None:
        check_scenario(workspace_id, scenario)
        payload = model_json_text(scenario, "시나리오")
        with self._state.lock:
            self._state.scenarios[(workspace_id, scenario.id)] = payload

    def list_scenarios(self, workspace_id: str) -> list[Scenario]:
        with self._state.lock:
            payloads = [p for (ws, _), p in self._state.scenarios.items() if ws == workspace_id]
        return sorted((Scenario.model_validate(_decode(p)) for p in payloads), key=scenario_order)

    def save_result(self, workspace_id: str, result: RehearsalResult) -> str:
        check_result(workspace_id, result)
        payload = model_json_text(result, "리허설 결과")
        rid = rehearsal_id(workspace_id, result)
        with self._state.lock:
            self._state.results[rid] = (workspace_id, payload)
        return rid

    def list_results(self, workspace_id: str, graph_hash: str | None = None) -> list[RehearsalResult]:
        with self._state.lock:
            entries = [(rid, p) for rid, (ws, p) in self._state.results.items() if ws == workspace_id]
        decoded = [(rid, RehearsalResult.model_validate(_decode(p))) for rid, p in entries]
        selected = [item for item in decoded if graph_hash is None or item[1].graph_hash == graph_hash]
        return [result for _, result in sorted(selected, key=result_order)]


class MemoryReportRepository:
    def __init__(self, state: _State) -> None:
        self._state = state

    def save_report(self, report: DiagnosisReport) -> None:
        created_at = check_report(report)
        payload = model_json_text(report, "리포트")
        with self._state.lock:
            existing = self._state.reports.get(report.run_id)
            owner = None if existing is None else existing[0]
            check_owner("리포트", report.run_id, owner, report.workspace_id)
            self._state.reports[report.run_id] = (report.workspace_id, created_at, payload)

    def latest_report(self, workspace_id: str) -> DiagnosisReport | None:
        with self._state.lock:
            candidates = [
                (report_order(created_at, run_id), payload)
                for run_id, (ws, created_at, payload) in self._state.reports.items()
                if ws == workspace_id
            ]
        if not candidates:
            return None
        _, payload = max(candidates, key=lambda item: item[0])
        return DiagnosisReport.model_validate(_decode(payload))


class MemoryAuditRepository:
    def __init__(self, state: _State, clock: Clock) -> None:
        self._state = state
        self._clock = clock

    def record(self, workspace_id: str, actor: str, action: str, payload: dict[str, object]) -> None:
        validate_audit(workspace_id, actor, action)
        body = to_json_object(payload, "감사 로그 내용")
        with self._state.lock:
            # The clock is read under the lock, so append order (seq) and time never disagree.
            # seq follows the last entry, as SQLite's rowid does (the two agree even after a
            # workspace's entries were deleted).
            audit = self._state.audit
            entry = AuditEntry(
                seq=audit[-1][1] + 1 if audit else 1,
                workspace_id=workspace_id,
                actor=actor,
                action=action,
                payload=body,
                recorded_at=as_utc(self._clock(), "clock()의 반환값"),
            )
            audit.append((workspace_id, entry.seq, model_json_text(entry, "감사 로그")))

    def entries(self, workspace_id: str) -> list[AuditEntry]:
        with self._state.lock:
            rows = [row for ws, _, row in self._state.audit if ws == workspace_id]
        return [AuditEntry.model_validate(_decode(row)) for row in rows]


class MemorySignupRepository:
    def __init__(self, state: _State) -> None:
        self._state = state

    def add_signup(self, signup: Signup) -> None:
        with self._state.lock:
            self._state.signups[signup.email] = signup

    def add_signup_if_absent(self, signup: Signup) -> bool:
        with self._state.lock:
            if signup.email in self._state.signups:
                return False
            self._state.signups[signup.email] = signup
            return True

    def delete_signup(self, email: str) -> bool:
        key = signup_key(email)
        if key is None:
            return False
        with self._state.lock:
            return self._state.signups.pop(key, None) is not None

    def delete_signups_before(self, cutoff: datetime) -> int:
        limit = deletion_cutoff(cutoff)
        with self._state.lock:
            expired = [email for email, signup in self._state.signups.items() if signup.created_at < limit]
            for email in expired:
                del self._state.signups[email]
        return len(expired)

    def get_signup(self, email: str) -> Signup | None:
        key = signup_key(email)
        if key is None:
            return None
        with self._state.lock:
            return self._state.signups.get(key)

    def count_signups(self) -> int:
        with self._state.lock:
            return len(self._state.signups)

    def list_signups(self) -> list[Signup]:
        with self._state.lock:
            signups = list(self._state.signups.values())
        return sorted(signups, key=signup_order)


class MemoryEventRepository:
    def __init__(self, state: _State) -> None:
        self._state = state

    def record_event(self, name: str, props: Mapping[str, object], ts: datetime) -> None:
        entry = (validate_event_name(name), json_text(props, "이벤트 속성"), as_utc(ts, "이벤트 시각"))
        with self._state.lock:
            self._state.events.append(entry)

    def event_counts(self) -> dict[str, int]:
        with self._state.lock:
            counts = Counter(name for name, _, _ in self._state.events)
        return dict(sorted(counts.items()))


class MemoryWorkspaceAccessRepository:
    def __init__(self, state: _State) -> None:
        self._state = state

    def set_access_key_digest(self, workspace_id: str, digest: str) -> None:
        checked = check_key_digest(workspace_id, digest)
        with self._state.lock:
            self._state.access_keys[workspace_id] = checked

    def access_key_digest(self, workspace_id: str) -> str | None:
        with self._state.lock:
            return self._state.access_keys.get(workspace_id)


class MemoryStore:
    """Process-local store implementing every repository protocol (see :class:`~jettae.store.base.Store`)."""

    def __init__(self, *, clock: Clock = utc_now) -> None:
        state = _State()
        self.workspaces = MemoryWorkspaceRepository(state, clock)
        self.documents = MemoryDocumentRepository(state)
        self.inputs = MemoryInputRepository(state)
        self.graphs = MemoryGraphRepository(state)
        self.patches = MemoryPatchRepository(state)
        self.rehearsals = MemoryRehearsalRepository(state)
        self.reports = MemoryReportRepository(state)
        self.audit = MemoryAuditRepository(state, clock)
        self.signups = MemorySignupRepository(state)
        self.events = MemoryEventRepository(state)
        self.access = MemoryWorkspaceAccessRepository(state)

    def close(self) -> None:
        """Nothing to release; present so callers can treat every :class:`Store` alike."""


__all__ = ["MemoryStore"]
