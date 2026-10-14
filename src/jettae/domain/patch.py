"""Patches (수정안) and graph versions (그래프 버전).

The graph changes ONLY through an approved patch. Every approved patch produces a new
content-addressed :class:`GraphVersion`; revert is just another patch (the inverse ops), so the
history is append-only and auditable (event sourcing).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from jettae.domain.graph import Attr, Edge, Node


class AddNode(BaseModel):
    model_config = ConfigDict(frozen=True)
    op: Literal["add_node"] = "add_node"
    node: Node


class AddEdge(BaseModel):
    model_config = ConfigDict(frozen=True)
    op: Literal["add_edge"] = "add_edge"
    edge: Edge


class RemoveEdge(BaseModel):
    model_config = ConfigDict(frozen=True)
    op: Literal["remove_edge"] = "remove_edge"
    edge_id: str
    #: snapshot of the removed edge so the op is invertible without the base graph
    snapshot: Edge | None = None


class RemoveNode(BaseModel):
    """Only allowed for nodes with no remaining edges (validator enforces)."""

    model_config = ConfigDict(frozen=True)
    op: Literal["remove_node"] = "remove_node"
    node_id: str
    snapshot: Node | None = None


class SetAttr(BaseModel):
    model_config = ConfigDict(frozen=True)
    op: Literal["set_attr"] = "set_attr"
    target_id: str  # node or edge id
    key: str
    value: Attr | None  # None = remove attribute
    previous: Attr | None = None  # filled by patching.apply for invertibility


PatchOp = Annotated[AddNode | AddEdge | RemoveEdge | RemoveNode | SetAttr, Field(discriminator="op")]


class PatchAuthor(StrEnum):
    AI = "ai"  # 생성형 AI 초안
    TEMPLATE = "template"  # 결정적 템플릿 초안 (LLM OFF)
    HUMAN = "human"


class PatchStatus(StrEnum):
    DRAFT = "draft"
    VALIDATED = "validated"  # 재검사·과거 리허설 재실행 통과
    REJECTED_BY_VALIDATOR = "rejected_by_validator"
    APPROVED = "approved"  # 팀 리드 승인 -> 새 버전 생성됨
    REJECTED = "rejected"  # 사람이 거절
    SUPERSEDED = "superseded"  # 기준 버전이 바뀌어 무효


class PatchValidation(BaseModel):
    model_config = ConfigDict(frozen=True)

    passed: bool
    resolved_finding_ids: tuple[str, ...] = ()
    unresolved_finding_ids: tuple[str, ...] = ()
    new_finding_ids: tuple[str, ...] = ()  # 회귀 (수정으로 새로 생긴 끊김)
    rehearsals_fixed: tuple[str, ...] = ()  # FAIL -> PASS 된 시나리오
    rehearsals_regressed: tuple[str, ...] = ()  # PASS -> FAIL (no-regression 위반)
    unknown_people: tuple[str, ...] = ()  # 명부 외 인물 (생성 금지 KPI: 0건)
    integrity_errors: tuple[str, ...] = ()
    score_before: float | None = None
    score_after: float | None = None
    notes: tuple[str, ...] = ()


class Patch(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    workspace_id: str
    base_version: int
    finding_ids: tuple[str, ...]
    ops: tuple[PatchOp, ...]
    runbook_text: str  # 런북에 추가/수정할 한국어 문장 (사람이 읽고 승인하는 대상)
    rationale: str
    author: PatchAuthor
    status: PatchStatus = PatchStatus.DRAFT
    validation: PatchValidation | None = None
    decided_by: str | None = None
    decided_at: datetime | None = None
    created_at: datetime | None = None
    llm_usage: dict[str, int] = Field(default_factory=dict)


class GraphVersion(BaseModel):
    model_config = ConfigDict(frozen=True)

    workspace_id: str
    number: int  # 1, 2, 3 …
    parent: int | None
    graph_hash: str
    patch_id: str | None = None  # None for the initial extraction
    revert_of: int | None = None  # set when this version reverts a previous one
    rehearsal_ids: tuple[str, ...] = ()  # rehearsals that validated this version
    created_by: str
    created_at: datetime
    note: str = ""
