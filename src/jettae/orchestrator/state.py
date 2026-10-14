"""What flows into, through and out of one diagnosis run (진단 1회의 입력·상태·결과).

Why three separate shapes:

* :class:`DiagnosisOptions` is everything the caller decides *besides* the documents — which
  workspace and head version the run builds on, which approved patches must survive a
  re-extraction, which hand-written drills to replay. It is a frozen, validated value so a run
  can be described (and reproduced) by its inputs alone (P6).
* :class:`DiagnosisState` is the LangGraph state: a ``TypedDict`` whose values are immutable
  domain objects. Every node returns only the keys it produced; LangGraph merges them. Two keys
  accumulate instead of being replaced — ``notes`` (Korean remarks such as "LLM 추출이 실패해
  규칙 파서로 진행") and ``timings`` (milliseconds per node) — through the reducers declared
  in their ``Annotated`` types.
* :class:`DiagnosisOutcome` is what :func:`jettae.orchestrator.pipeline.run_diagnosis` returns:
  the graph and every verdict about it, plus the bookkeeping the service needs to persist the
  run (extraction hash for the version lineage, re-applied patch ids, the version number the
  drafts were made against).

Nothing here performs I/O or judges anything; it only names the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import TypedDict

from jettae.check.context import CheckContext
from jettae.domain import (
    Document,
    Extraction,
    Finding,
    GroundingStatus,
    InputBundle,
    LlmUsageSummary,
    OpsGraph,
    Patch,
    PatchStatus,
    ReadinessReport,
    RehearsalResult,
    Scenario,
    ScenarioOrigin,
    SecurityNotice,
    SimConfig,
)
from jettae.extract import AssemblyResult, VerificationResult

__all__ = [
    "DEFAULT_WORKSPACE_ID",
    "MAX_LLM_CONCURRENCY",
    "MIN_LLM_CONCURRENCY",
    "DiagnosisOptions",
    "DiagnosisOutcome",
    "DiagnosisState",
    "ExtractorPolicy",
    "append_notes",
    "merge_timings",
]

#: Workspace id used for patch ids when a run is not tied to a stored workspace (one-shot scans).
DEFAULT_WORKSPACE_ID = "local"
#: Bounds of :attr:`DiagnosisOptions.llm_concurrency` (concurrent model calls within one run).
MIN_LLM_CONCURRENCY = 1
MAX_LLM_CONCURRENCY = 64

#: ``rule`` — Korean rule parser only (LLM OFF, or the LLM path failed);
#: ``hybrid`` — rule parser *and* LLM extractions fed to one assembly (the product default when
#: an LLM is configured): the rules guarantee recall of the patterns they know, the LLM adds the
#: free-form sentences they miss, and the assembler grounds both the same way.
ExtractorPolicy = Literal["rule", "hybrid"]


def append_notes(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    """LangGraph reducer for ``notes``: keep every node's remarks, in execution order."""
    return (*left, *right)


def merge_timings(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    """LangGraph reducer for ``timings``: one entry per node (a later write of a key wins)."""
    return {**left, **right}


class DiagnosisOptions(BaseModel):
    """Caller decisions for one :func:`~jettae.orchestrator.pipeline.run_diagnosis` call.

    ``base_version`` is the head version number of the ``base_graph`` passed alongside (both or
    neither); drafts are made against the version the resulting graph will be committed as, so
    the pipeline must know where it stands. ``base_extraction_hash`` and ``approved_patches``
    describe how the head came to be — the extraction it started from and the approved patches
    applied on top, in approval order — so a re-extraction can keep human decisions.
    ``carried_scenarios`` are previously resolved human/LLM drills (kept when their nodes still
    exist); ``scenario_sources`` are ``scenarios.yaml`` sources to resolve onto the new graph, as
    :func:`~jettae.rehearse.scenarios_from_yaml` reads them: ``bytes`` is file content, a ``str`` is
    YAML text or — one line ending in ``.yaml`` — a path. Services pass uploaded *bytes* only, so a
    request can never name a server file.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str = Field(default=DEFAULT_WORKSPACE_ID, min_length=1)
    base_version: int | None = Field(default=None, ge=1)
    base_extraction_hash: str | None = None
    approved_patches: tuple[Patch, ...] = ()
    carried_scenarios: tuple[Scenario, ...] = ()
    scenario_sources: tuple[str | bytes, ...] = ()
    rehearse: bool = True
    draft: bool = True
    #: restrict drafting to these finding ids (``None`` = every finding)
    draft_finding_ids: tuple[str, ...] | None = None
    llm_scenario_count: int = Field(default=3, ge=0, le=20)
    sim_config: SimConfig = Field(default_factory=SimConfig)
    #: on-call checks only look at shifts not yet over at ``now`` (``None`` = the whole table)
    now: datetime | None = None
    llm_concurrency: int = Field(default=4, ge=MIN_LLM_CONCURRENCY, le=MAX_LLM_CONCURRENCY)
    max_chunk_chars: int = Field(default=6000, ge=500)

    @model_validator(mode="after")
    def _consistent(self) -> DiagnosisOptions:
        if self.base_version is None and (self.base_extraction_hash or self.approved_patches):
            raise ValueError("기준 버전(base_version) 없이 추출 이력이나 승인된 수정안을 넘길 수 없습니다.")
        if self.now is not None and self.now.utcoffset() is None:
            raise ValueError("now에는 시간대 정보가 있어야 합니다(예: Asia/Seoul).")
        unapproved = [p.id for p in self.approved_patches if p.status is not PatchStatus.APPROVED]
        if unapproved:
            raise ValueError(f"승인되지 않은 수정안은 다시 적용할 수 없습니다: {', '.join(unapproved)}")
        templates = [s.id for s in self.carried_scenarios if s.origin is ScenarioOrigin.TEMPLATE]
        if templates:
            raise ValueError(
                "템플릿 시나리오는 그래프마다 새로 만들므로 이어받을 수 없습니다: " + ", ".join(templates)
            )
        return self


class DiagnosisState(TypedDict, total=False):
    """LangGraph state of one run (see the module docstring). Keys appear as nodes produce them."""

    bundle: InputBundle
    #: the current head graph the run builds on (``None`` for a first diagnosis or a one-shot scan)
    base_graph: OpsGraph | None
    documents: tuple[Document, ...]
    security: tuple[SecurityNotice, ...]
    rule_extractions: tuple[Extraction, ...]
    llm_extractions: tuple[Extraction, ...]
    llm_failure: str | None
    extractor: ExtractorPolicy
    assembly: AssemblyResult
    verification: VerificationResult
    extraction_hash: str
    graph: OpsGraph
    graph_reused: bool
    reapplied_patch_ids: tuple[str, ...]
    ctx: CheckContext
    findings: tuple[Finding, ...]
    readiness: ReadinessReport
    scenarios: tuple[Scenario, ...]
    rehearsals: tuple[RehearsalResult, ...]
    draft_base_version: int
    patches: tuple[Patch, ...]
    grounding: GroundingStatus
    held_ids: tuple[str, ...]
    catalog_gaps: tuple[str, ...]
    usage: LlmUsageSummary
    notes: Annotated[tuple[str, ...], append_notes]
    timings: Annotated[dict[str, float], merge_timings]


@dataclass(frozen=True)
class DiagnosisOutcome:
    """Everything one diagnosis produced (immutable; the service persists it as a report).

    ``patches`` are the drafts after validation: VALIDATED (approvable), REJECTED_BY_VALIDATOR
    (they would not close their gap or would break something), or DRAFT for review memos without
    graph ops, which only a human can resolve. ``timings`` are wall-clock milliseconds per node
    plus ``total`` — metadata, the only part of the outcome that is not reproducible.
    """

    graph: OpsGraph
    findings: tuple[Finding, ...]
    readiness: ReadinessReport
    scenarios: tuple[Scenario, ...]
    rehearsals: tuple[RehearsalResult, ...]
    patches: tuple[Patch, ...]
    grounding: GroundingStatus
    security: tuple[SecurityNotice, ...]
    held_ids: tuple[str, ...]
    catalog_gaps: tuple[str, ...]
    timings: dict[str, float]
    usage: LlmUsageSummary
    extractor: ExtractorPolicy
    extraction_hash: str
    graph_reused: bool
    reapplied_patch_ids: tuple[str, ...]
    draft_base_version: int
    notes: tuple[str, ...]
