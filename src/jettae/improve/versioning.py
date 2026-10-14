"""Graph versions (그래프 버전): approve, reject and revert — every change is an appended event.

Why a service object owns these transitions (ADR-0003, P7): the graph is an immutable value and
the only way to change it is an approved patch, so "who changed what, when, which break it closed
and which rehearsal proved it" can always be answered from the history. The rules enforced here:

* **Only a lead or an admin decides** (``X-Jettae-Role``): approving, rejecting and reverting
  raise ``PermissionError('승인 권한이 없습니다')`` for anyone else (ARCHITECTURE §5 RBAC).
* **Only validated patches are approved**, and only against the version they were validated on.
  A patch whose ``base_version`` is no longer the head is marked SUPERSEDED and refused with
  :class:`~jettae.domain.ports.VersionConflictError` — its validation said nothing about the newer
  graph. The repository's optimistic concurrency (``expected_head``) closes the race between two
  approvers in the same way. One race needs care: two leads approving the *same* patch at once.
  The loser sees a moved head, but the version that moved it is that very patch — so the history
  (append-only, hence authoritative) is consulted first, and the loser gets
  :class:`PatchStateError` instead of overwriting the winner's APPROVED with SUPERSEDED.
* **History is append-only.** Reverting appends a new version whose graph equals the target's
  (``revert_of``); nothing is deleted. Committing a graph identical to the head appends nothing.
* **Every decision is audited** (actor, role, patch, finding ids, version, graph hash). The actor
  and a rejection reason are checked *before* anything is written: the audit log refuses a blank
  actor or unstorable text, and finding that out after the version was appended or the patch was
  saved would leave an unaudited change behind.
* After the head moves, still-pending patches drafted on an older head are marked SUPERSEDED, so
  nobody approves a change that was checked against a graph that no longer exists. Each patch is
  re-read just before it is marked, so a decision taken meanwhile (a rejection) is not overwritten.

Timestamps come from an injected clock (UTC by default) so tests and demos are reproducible.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Final

from jettae.check.elements import josa
from jettae.domain import GraphVersion, OpsGraph, Patch, PatchStatus
from jettae.domain.ports import (
    AuditLog,
    GraphRepository,
    NotFoundError,
    PatchRepository,
    VersionConflictError,
)
from jettae.improve.patching import GraphDiff, apply_ops, graph_diff

__all__ = [
    "DECIDER_ROLES",
    "SYSTEM_ACTOR",
    "Clock",
    "PatchStateError",
    "VersionService",
    "utc_now",
]

#: A clock returns the current instant, timezone-aware (it stamps metadata, never judgement).
Clock = Callable[[], datetime]

#: Roles allowed to approve, reject and revert (compared case-insensitively).
DECIDER_ROLES: Final[frozenset[str]] = frozenset({"lead", "admin"})
#: Actor recorded for bookkeeping the service does on its own (marking stale patches).
SYSTEM_ACTOR: Final = "jettae"

_PENDING: Final = frozenset({PatchStatus.DRAFT, PatchStatus.VALIDATED})
_REJECTABLE: Final = frozenset({PatchStatus.DRAFT, PatchStatus.VALIDATED, PatchStatus.REJECTED_BY_VALIDATOR})
_STATUS_KO: Final[dict[PatchStatus, str]] = {
    PatchStatus.DRAFT: "초안",
    PatchStatus.VALIDATED: "검증 통과",
    PatchStatus.REJECTED_BY_VALIDATOR: "검증 실패",
    PatchStatus.APPROVED: "승인됨",
    PatchStatus.REJECTED: "거절됨",
    PatchStatus.SUPERSEDED: "기준 버전 변경으로 무효",
}


def utc_now() -> datetime:
    """Default :data:`Clock`: the wall clock in UTC."""
    return datetime.now(UTC)


class PatchStateError(ValueError):
    """The patch is not in a state that allows the requested decision (message in Korean)."""


def _storable(text: str, what: str) -> str:
    """Refuse text no store can keep (NUL, lone surrogates) *before* anything is written.

    The audit log refuses such text; finding that out after the patch or version was saved would
    leave a decision without its audit record.
    """
    if "\x00" in text or any("\ud800" <= ch <= "\udfff" for ch in text):
        raise ValueError(f"{what}에 저장할 수 없는 문자(NUL·짝 없는 서로게이트)가 있습니다.")
    return text


def _actor(actor: str) -> str:
    """The acting person, stripped; a blank or unstorable actor is refused before anything is written."""
    name = _storable(actor.strip(), "행위자")
    if not name:
        raise ValueError("누가 결정했는지(행위자)를 적어 주세요. 감사 로그에 남길 수 없습니다.")
    return name


class VersionService:
    """Approve / reject / revert patches as append-only graph versions (see the module docstring)."""

    def __init__(
        self,
        graphs: GraphRepository,
        patches: PatchRepository,
        audit: AuditLog,
        clock: Clock = utc_now,
    ) -> None:
        self.graphs = graphs
        self.patches = patches
        self.audit = audit
        self.clock = clock

    # ------------------------------------------------------------------ reads
    def history(self, workspace_id: str) -> list[GraphVersion]:
        """Every version of the workspace, oldest first."""
        return self.graphs.list_versions(workspace_id)

    def head(self, workspace_id: str) -> tuple[GraphVersion, OpsGraph]:
        """The latest version and its graph (:class:`NotFoundError` if nothing was committed yet)."""
        head = self.graphs.head(workspace_id)
        if head is None:
            raise NotFoundError(f"워크스페이스 {workspace_id}에 아직 그래프 버전이 없습니다.")
        return head, self.graphs.load_graph(head.graph_hash)

    def graph(self, workspace_id: str, number: int) -> OpsGraph:
        return self.graphs.load_graph(self.graphs.get_version(workspace_id, number).graph_hash)

    def diff(self, workspace_id: str, a: int, b: int) -> GraphDiff:
        """What changed from version ``a`` to version ``b``."""
        return graph_diff(self.graph(workspace_id, a), self.graph(workspace_id, b))

    # ------------------------------------------------------------------ writes
    def commit_graph(self, workspace_id: str, graph: OpsGraph, *, actor: str, note: str = "") -> GraphVersion:
        """Append ``graph`` as a new version — unless it equals the head, which is then returned.

        Used for a diagnosis' extracted graph; patches go through :meth:`approve`.
        """
        actor = _actor(actor)
        graph.validate_integrity()
        head = self.graphs.head(workspace_id)
        graph_hash = graph.content_hash()
        if head is not None and head.graph_hash == graph_hash:
            return head
        version = self._next_version(workspace_id, head, graph_hash, actor=actor, note=note)
        self.graphs.append_version(version, graph, expected_head=head.number if head else None)
        self.audit.record(
            workspace_id,
            actor,
            "graph.committed",
            {"version": version.number, "graph_hash": graph_hash, "note": note},
        )
        self.supersede_pending(workspace_id, actor=actor)
        return version

    def approve(self, patch_id: str, *, actor: str, role: str) -> GraphVersion:
        """Apply a VALIDATED patch to the head it was validated on and append the result."""
        self._authorize(role, "승인 권한이 없습니다")
        actor = _actor(actor)
        patch = self.patches.get_patch(patch_id)
        validated = patch.validation is not None and patch.validation.passed
        if patch.status is not PatchStatus.VALIDATED or not validated:
            raise PatchStateError(
                f"검증을 통과한 수정안만 승인할 수 있습니다(수정안 {patch.id}: {_STATUS_KO[patch.status]})."
            )
        ws = patch.workspace_id
        head, base = self.head(ws)
        if patch.base_version != head.number:
            raise self._conflict(patch, actor, head.number)
        graph, _applied = apply_ops(base, patch.ops)
        now = self.clock()
        version = GraphVersion(
            workspace_id=ws,
            number=head.number + 1,
            parent=head.number,
            graph_hash=graph.content_hash(),
            patch_id=patch.id,
            created_by=actor,
            created_at=now,
            note=patch.runbook_text,
        )
        try:
            self.graphs.append_version(version, graph, expected_head=head.number)
        except VersionConflictError:
            raise self._conflict(patch, actor, None) from None
        approved = patch.model_copy(
            update={"status": PatchStatus.APPROVED, "decided_by": actor, "decided_at": now}
        )
        self.patches.save_patch(approved)
        self.audit.record(
            ws,
            actor,
            "patch.approved",
            {
                "patch_id": patch.id,
                "role": role,
                "finding_ids": list(patch.finding_ids),
                "base_version": patch.base_version,
                "version": version.number,
                "graph_hash": version.graph_hash,
            },
        )
        self.supersede_pending(ws, except_ids=(patch.id,), actor=actor)
        return version

    def reject(self, patch_id: str, *, actor: str, role: str, reason: str) -> Patch:
        """A human turns a pending patch down; the reason goes to the audit log."""
        self._authorize(role, "승인 권한이 없습니다")
        actor = _actor(actor)
        if not _storable(reason, "거절 사유").strip():
            raise ValueError("거절 사유를 적어 주세요.")
        patch = self.patches.get_patch(patch_id)
        if patch.status not in _REJECTABLE:
            raise PatchStateError(
                f"이미 결정된 수정안은 거절할 수 없습니다(수정안 {patch.id}: {_STATUS_KO[patch.status]})."
            )
        rejected = patch.model_copy(
            update={"status": PatchStatus.REJECTED, "decided_by": actor, "decided_at": self.clock()}
        )
        self.patches.save_patch(rejected)
        self.audit.record(
            patch.workspace_id,
            actor,
            "patch.rejected",
            {
                "patch_id": patch.id,
                "role": role,
                "reason": reason.strip(),
                "finding_ids": list(patch.finding_ids),
            },
        )
        return rejected

    def revert(self, workspace_id: str, to_number: int, *, actor: str, role: str) -> GraphVersion:
        """Append a version whose graph equals version ``to_number``'s (history is never rewritten).

        When the head already has that content nothing is appended and the head is returned.
        """
        self._authorize(role, "되돌리기 권한이 없습니다")
        actor = _actor(actor)
        head = self.graphs.head(workspace_id)
        if head is None:
            raise NotFoundError(f"워크스페이스 {workspace_id}에 아직 그래프 버전이 없습니다.")
        target = self.graphs.get_version(workspace_id, to_number)
        if target.graph_hash == head.graph_hash:
            return head
        graph = self.graphs.load_graph(target.graph_hash)
        note = f"{josa(f'버전 {to_number}', '으로', '로')} 되돌림"
        version = self._next_version(
            workspace_id, head, target.graph_hash, actor=actor, note=note, revert_of=to_number
        )
        self.graphs.append_version(version, graph, expected_head=head.number)
        self.audit.record(
            workspace_id,
            actor,
            "graph.reverted",
            {"role": role, "to": to_number, "version": version.number, "graph_hash": version.graph_hash},
        )
        self.supersede_pending(workspace_id, actor=actor)
        return version

    def supersede_pending(
        self, workspace_id: str, *, except_ids: Iterable[str] = (), actor: str = SYSTEM_ACTOR
    ) -> list[Patch]:
        """Mark DRAFT/VALIDATED patches based on an older head SUPERSEDED; return them."""
        actor = _actor(actor)
        head = self.graphs.head(workspace_id)
        if head is None:
            return []
        keep = set(except_ids)
        stale: list[Patch] = []
        for listed in self.patches.list_patches(workspace_id):
            if listed.status not in _PENDING or listed.base_version >= head.number or listed.id in keep:
                continue
            marked = self._mark_superseded(listed.id)
            if marked is not None:
                stale.append(marked)
        if stale:
            self.audit.record(
                workspace_id,
                actor,
                "patch.superseded",
                {"patch_ids": [p.id for p in stale], "head": head.number},
            )
        return stale

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _authorize(role: str, message: str) -> None:
        if role.strip().casefold() not in DECIDER_ROLES:
            raise PermissionError(message)

    def _next_version(
        self,
        workspace_id: str,
        head: GraphVersion | None,
        graph_hash: str,
        *,
        actor: str,
        note: str,
        revert_of: int | None = None,
    ) -> GraphVersion:
        return GraphVersion(
            workspace_id=workspace_id,
            number=1 if head is None else head.number + 1,
            parent=None if head is None else head.number,
            graph_hash=graph_hash,
            revert_of=revert_of,
            created_by=actor,
            created_at=self.clock(),
            note=note,
        )

    def _mark_superseded(self, patch_id: str) -> Patch | None:
        """Re-read the patch and mark it SUPERSEDED if it is still pending; return what was saved.

        The re-read keeps a decision taken since the caller looked (an approval or a rejection by
        someone else) from being overwritten with SUPERSEDED.
        """
        current = self.patches.get_patch(patch_id)
        if current.status not in _PENDING:
            return None
        marked = current.model_copy(update={"status": PatchStatus.SUPERSEDED})
        self.patches.save_patch(marked)
        return marked

    def _approved_as(self, patch: Patch) -> int | None:
        """The version number this patch already produced, if any (the history is authoritative)."""
        return next(
            (v.number for v in self.graphs.list_versions(patch.workspace_id) if v.patch_id == patch.id),
            None,
        )

    def _conflict(self, patch: Patch, actor: str, head: int | None) -> PatchStateError | VersionConflictError:
        """The error to raise when ``patch`` no longer targets the head.

        If the head moved because this very patch was approved concurrently, that approval stands
        and :class:`PatchStateError` is returned. Otherwise the patch is marked SUPERSEDED (when
        still pending), audited, and a :class:`VersionConflictError` is returned.
        """
        produced = self._approved_as(patch)
        if produced is not None:
            return PatchStateError(
                f"이미 승인되어 {josa(f'버전 {produced}', '이', '가')} 된 수정안입니다(수정안 {patch.id})."
            )
        if self._mark_superseded(patch.id) is not None:
            self.audit.record(
                patch.workspace_id,
                actor,
                "patch.superseded",
                {"patch_ids": [patch.id], "head": head, "base_version": patch.base_version},
            )
        where = f"현재 최신 버전은 {head}입니다" if head is not None else "그사이 다른 변경이 승인되었습니다"
        return VersionConflictError(
            f"수정안 {patch.id}의 기준 버전({patch.base_version})이 최신이 아닙니다. {where}. "
            "최신 버전에서 다시 진단해 수정안을 새로 만드세요."
        )
