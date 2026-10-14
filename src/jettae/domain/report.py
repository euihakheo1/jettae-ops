"""The diagnosis report — what one ``diagnose`` run returns to API/CLI/UI."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from jettae.domain.extraction import GroundingStatus
from jettae.domain.findings import Finding, ReadinessReport
from jettae.domain.inputs import LlmMode
from jettae.domain.patch import Patch
from jettae.domain.rehearsal import RehearsalResult, Scenario


class LlmUsageSummary(BaseModel):
    """KPI: 1건당 LLM 호출 수·토큰 공개. The judgement path always reports 0 calls."""

    model_config = ConfigDict(frozen=True)

    mode: LlmMode
    provider: str | None = None
    model: str | None = None
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    by_task: dict[str, int] = Field(default_factory=dict)  # task -> calls
    judgement_path_calls: int = 0  # 판정·점수·재생 경로 LLM 호출 (항상 0)


class SecurityNotice(BaseModel):
    """Prompt-injection attempts found in documents (quarantined, never obeyed)."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    start: int
    end: int
    excerpt: str
    pattern: str


class DiagnosisReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    workspace_id: str
    graph_version: int
    graph_hash: str
    created_at: datetime
    findings: tuple[Finding, ...]
    readiness: ReadinessReport
    scenarios: tuple[Scenario, ...] = ()
    rehearsals: tuple[RehearsalResult, ...] = ()
    patches: tuple[Patch, ...] = ()
    grounding: GroundingStatus = GroundingStatus()
    security: tuple[SecurityNotice, ...] = ()
    llm: LlmUsageSummary
    held_element_ids: tuple[str, ...] = ()  # 판단 보류 요소
    catalog_gaps: tuple[str, ...] = ()  # 카탈로그엔 있으나 담당·런북 없는 서비스
    timings_ms: dict[str, float] = Field(default_factory=dict)
