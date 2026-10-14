"""Tests for ``jettae.improve.versioning`` against the in-memory store."""

from __future__ import annotations

import itertools
from datetime import UTC, datetime, timedelta

import pytest

from jettae.check.registry import run_checks
from jettae.domain import GraphVersion, OpsGraph, Patch, PatchStatus, PatchValidation
from jettae.domain.ports import NotFoundError, VersionConflictError
from jettae.improve import (
    PatchStateError,
    TemplateDrafter,
    VersionService,
    apply_ops,
    graph_diff,
    utc_now,
    validate_patch,
)
from jettae.store import MemoryStore
from test_improve_support import PaymentOrg, payment_org

WS = "ws-demo"
START = datetime(2026, 10, 13, 9, 0, tzinfo=UTC)


class Clock:
    """Deterministic clock: every call is one minute later."""

    def __init__(self) -> None:
        self._ticks = itertools.count()

    def __call__(self) -> datetime:
        return START + timedelta(minutes=next(self._ticks))


@pytest.fixture
def store() -> MemoryStore:
    memory = MemoryStore(clock=Clock())
    memory.workspaces.create_workspace("데모", workspace_id=WS)
    return memory


@pytest.fixture
def service(store: MemoryStore) -> VersionService:
    return VersionService(store.graphs, store.patches, store.audit, clock=Clock())


@pytest.fixture
def po() -> PaymentOrg:
    return payment_org()


def drafted(po: PaymentOrg, graph: OpsGraph, version: int, rule: str) -> Patch:
    ctx = po.org.ctx().with_graph(graph)
    patches = TemplateDrafter().draft(ctx, run_checks(ctx), workspace_id=WS, base_version=version)
    return next(p for p in patches if any(f.startswith(rule) for f in p.finding_ids))


def validated(po: PaymentOrg, graph: OpsGraph, version: int, rule: str) -> Patch:
    patch = drafted(po, graph, version, rule)
    result = validate_patch(po.org.ctx().with_graph(graph), patch)
    assert result.passed
    return patch.model_copy(update={"status": PatchStatus.VALIDATED, "validation": result})


def actions(store: MemoryStore) -> list[str]:
    return [e.action for e in store.audit.entries(WS)]


def test_commit_appends_only_new_content(service: VersionService, store: MemoryStore, po: PaymentOrg) -> None:
    graph = po.org.graph()
    first = service.commit_graph(WS, graph, actor="system", note="초기 추출")
    assert (first.number, first.parent, first.graph_hash) == (1, None, graph.content_hash())
    assert service.commit_graph(WS, graph, actor="system") == first
    owner = drafted(po, graph, 1, "JT-PPL-001")
    second_graph, _ = apply_ops(graph, owner.ops)
    second = service.commit_graph(WS, second_graph, actor="system", note="재진단")
    assert (second.number, second.parent, second.revert_of) == (2, 1, None)
    assert [v.number for v in service.history(WS)] == [1, 2]
    assert actions(store) == ["graph.committed", "graph.committed"]
    assert service.head(WS) == (second, second_graph)


def test_approve_appends_a_version_and_records_the_decision(
    service: VersionService, store: MemoryStore, po: PaymentOrg
) -> None:
    graph = po.org.graph()
    service.commit_graph(WS, graph, actor="system")
    patch = validated(po, graph, 1, "JT-PPL-001")
    store.patches.save_patch(patch)
    version = service.approve(patch.id, actor="박재현", role="Lead ")
    expected, _ = apply_ops(graph, patch.ops)
    assert (version.number, version.parent, version.patch_id) == (2, 1, patch.id)
    assert version.graph_hash == expected.content_hash()
    assert version.note == patch.runbook_text
    assert version.created_by == "박재현"
    assert version.created_at.tzinfo is not None
    saved = store.patches.get_patch(patch.id)
    assert saved.status is PatchStatus.APPROVED
    assert saved.decided_by == "박재현"
    assert saved.decided_at is not None
    approved = store.audit.entries(WS)[-1]
    assert approved.action == "patch.approved"
    assert approved.payload["patch_id"] == patch.id
    assert approved.payload["version"] == 2
    assert approved.payload["finding_ids"] == list(patch.finding_ids)


def test_only_leads_and_admins_decide(service: VersionService, store: MemoryStore, po: PaymentOrg) -> None:
    graph = po.org.graph()
    service.commit_graph(WS, graph, actor="system")
    patch = validated(po, graph, 1, "JT-PPL-001")
    store.patches.save_patch(patch)
    with pytest.raises(PermissionError, match="승인 권한이 없습니다"):
        service.approve(patch.id, actor="강현우", role="member")
    with pytest.raises(PermissionError, match="승인 권한이 없습니다"):
        service.reject(patch.id, actor="강현우", role="viewer", reason="아님")
    with pytest.raises(PermissionError, match="되돌리기 권한이 없습니다"):
        service.revert(WS, 1, actor="강현우", role="")
    assert service.approve(patch.id, actor="관리자", role="admin").number == 2


@pytest.mark.parametrize(
    ("status", "passed"),
    [(PatchStatus.DRAFT, None), (PatchStatus.REJECTED_BY_VALIDATOR, False), (PatchStatus.VALIDATED, False)],
)
def test_only_validated_patches_are_approved(
    service: VersionService, store: MemoryStore, po: PaymentOrg, status: PatchStatus, passed: bool | None
) -> None:
    graph = po.org.graph()
    service.commit_graph(WS, graph, actor="system")
    patch = drafted(po, graph, 1, "JT-PPL-001")
    validation = None if passed is None else PatchValidation(passed=passed)
    store.patches.save_patch(patch.model_copy(update={"status": status, "validation": validation}))
    with pytest.raises(PatchStateError, match="검증을 통과한 수정안만"):
        service.approve(patch.id, actor="박재현", role="lead")
    assert len(service.history(WS)) == 1


def test_stale_base_version_supersedes_the_patch(
    service: VersionService, store: MemoryStore, po: PaymentOrg
) -> None:
    graph = po.org.graph()
    service.commit_graph(WS, graph, actor="system")
    owner = validated(po, graph, 1, "JT-PPL-001")
    approver = validated(po, graph, 1, "JT-PATH-020")
    executor = validated(po, graph, 1, "JT-PATH-041")
    for patch in (owner, approver, executor):
        store.patches.save_patch(patch)
    rejected = service.reject(executor.id, actor="박재현", role="lead", reason="실행자는 팀 회의 후 정함")
    service.approve(owner.id, actor="박재현", role="lead")
    # the other pending patch was checked against version 1 only: it is marked stale at once
    assert store.patches.get_patch(approver.id).status is PatchStatus.SUPERSEDED
    assert store.patches.get_patch(executor.id) == rejected
    with pytest.raises(PatchStateError):
        service.approve(approver.id, actor="박재현", role="lead")
    late = validated(po, graph, 1, "JT-PATH-020").model_copy(update={"id": "patch-late"})
    store.patches.save_patch(late)
    with pytest.raises(VersionConflictError, match="최신이 아닙니다"):
        service.approve(late.id, actor="박재현", role="lead")
    assert store.patches.get_patch(late.id).status is PatchStatus.SUPERSEDED
    assert actions(store).count("patch.superseded") == 2


def test_lost_race_supersedes_the_patch(store: MemoryStore, po: PaymentOrg) -> None:
    class Racing:
        """A graph repository where another approver wins between our head read and append."""

        def __init__(self) -> None:
            self.inner = store.graphs

        def head(self, workspace_id: str) -> GraphVersion | None:
            return self.inner.head(workspace_id)

        def get_version(self, workspace_id: str, number: int) -> GraphVersion:
            return self.inner.get_version(workspace_id, number)

        def list_versions(self, workspace_id: str) -> list[GraphVersion]:
            return self.inner.list_versions(workspace_id)

        def load_graph(self, graph_hash: str) -> OpsGraph:
            return self.inner.load_graph(graph_hash)

        def append_version(
            self, version: GraphVersion, graph: OpsGraph, *, expected_head: int | None
        ) -> None:
            raise VersionConflictError("다른 승인이 먼저 반영됨")

    racing = VersionService(Racing(), store.patches, store.audit, clock=Clock())
    graph = po.org.graph()
    store.graphs.append_version(
        GraphVersion(
            workspace_id=WS,
            number=1,
            parent=None,
            graph_hash=graph.content_hash(),
            created_by="system",
            created_at=START,
        ),
        graph,
        expected_head=None,
    )
    patch = validated(po, graph, 1, "JT-PPL-001")
    store.patches.save_patch(patch)
    with pytest.raises(VersionConflictError, match="그사이 다른 변경"):
        racing.approve(patch.id, actor="박재현", role="lead")
    assert store.patches.get_patch(patch.id).status is PatchStatus.SUPERSEDED


def test_reject_needs_a_reason_and_a_pending_patch(
    service: VersionService, store: MemoryStore, po: PaymentOrg
) -> None:
    graph = po.org.graph()
    service.commit_graph(WS, graph, actor="system")
    patch = drafted(po, graph, 1, "JT-PATH-020")
    store.patches.save_patch(patch)
    with pytest.raises(ValueError, match="거절 사유"):
        service.reject(patch.id, actor="박재현", role="lead", reason="  ")
    rejected = service.reject(patch.id, actor="박재현", role="lead", reason=" CTO 승인으로 바꿀 것 ")
    assert rejected.status is PatchStatus.REJECTED
    assert rejected.decided_by == "박재현"
    entry = store.audit.entries(WS)[-1]
    assert (entry.action, entry.payload["reason"]) == ("patch.rejected", "CTO 승인으로 바꿀 것")
    with pytest.raises(PatchStateError, match="이미 결정된"):
        service.reject(patch.id, actor="박재현", role="lead", reason="다시")


def test_revert_appends_the_old_graph_as_a_new_version(
    service: VersionService, store: MemoryStore, po: PaymentOrg
) -> None:
    graph = po.org.graph()
    first = service.commit_graph(WS, graph, actor="system")
    patch = validated(po, graph, 1, "JT-PPL-001")
    store.patches.save_patch(patch)
    service.approve(patch.id, actor="박재현", role="lead")
    reverted = service.revert(WS, 1, actor="박재현", role="lead")
    assert (reverted.number, reverted.parent, reverted.revert_of) == (3, 2, 1)
    assert reverted.graph_hash == first.graph_hash
    assert reverted.note == "버전 1로 되돌림"
    assert service.revert(WS, 1, actor="박재현", role="lead") == reverted  # nothing left to revert
    assert [v.number for v in service.history(WS)] == [1, 2, 3]  # history is never rewritten
    assert actions(store)[-1] == "graph.reverted"
    with pytest.raises(NotFoundError):
        service.revert(WS, 9, actor="박재현", role="lead")


def test_diff_and_graph_reads(service: VersionService, store: MemoryStore, po: PaymentOrg) -> None:
    graph = po.org.graph()
    service.commit_graph(WS, graph, actor="system")
    patch = validated(po, graph, 1, "JT-PPL-001")
    store.patches.save_patch(patch)
    service.approve(patch.id, actor="박재현", role="lead")
    diff = service.diff(WS, 1, 2)
    assert diff == graph_diff(service.graph(WS, 1), service.graph(WS, 2))
    assert diff.removed_edges
    assert diff.added_edges
    assert not diff.removed_nodes
    assert service.diff(WS, 2, 2).is_empty


def test_empty_workspace(service: VersionService) -> None:
    with pytest.raises(NotFoundError, match="아직 그래프 버전이 없습니다"):
        service.head(WS)
    with pytest.raises(NotFoundError):
        service.revert(WS, 1, actor="박재현", role="lead")
    assert service.supersede_pending(WS) == []
    assert service.history(WS) == []


def test_default_clock_is_aware_utc() -> None:
    now = utc_now()
    assert now.tzinfo is UTC
