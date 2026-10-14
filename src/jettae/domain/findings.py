"""Findings (끊김) and readiness scores (준비도 점수). Produced only by fixed rules."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from jettae.domain.graph import SourceSpan


class Severity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class FindingStatus(StrEnum):
    BROKEN = "broken"  # 끊김 확인
    UNKNOWN = "unknown"  # 확인 불가 — 정상으로 취급하지 않음 (확인 불가 ≠ 정상)


class Element(StrEnum):
    """The five required response-path elements per critical service (필수 경로 5요소)."""

    DETECTION = "detection"  # 감지
    OWNER = "owner"  # 담당 (상향 포함)
    APPROVER = "approver"  # 승인자
    FALLBACK_CHANNEL = "fallback_channel"  # 대체 채널
    ROLLBACK = "rollback"  # 롤백/완화 조치


ELEMENT_LABEL_KO: dict[Element, str] = {
    Element.DETECTION: "감지",
    Element.OWNER: "담당",
    Element.APPROVER: "승인자",
    Element.FALLBACK_CHANNEL: "대체 채널",
    Element.ROLLBACK: "롤백",
}


class FixKind(StrEnum):
    """Structured remediation intents. The drafter turns these into patch ops + a sentence."""

    ASSIGN_OWNER = "assign_owner"
    REPLACE_DEPARTED = "replace_departed"
    ADD_ESCALATION = "add_escalation"
    BREAK_ESCALATION_CYCLE = "break_escalation_cycle"
    ASSIGN_APPROVER = "assign_approver"
    ADD_DEPUTY = "add_deputy"
    ADD_FALLBACK_CHANNEL = "add_fallback_channel"
    ADD_ROLLBACK = "add_rollback"
    ADD_DETECTION = "add_detection"
    FIX_DEADLINE = "fix_deadline"
    ADD_REPORTING_STEP = "add_reporting_step"
    ADJUST_ONCALL = "adjust_oncall"
    ADD_SERVICE_RUNBOOK = "add_service_runbook"
    CONFIRM_IDENTITY = "confirm_identity"
    ASSIGN_EXECUTOR = "assign_executor"
    ADD_FAILOVER = "add_failover"
    REVIEW_ELEMENT = "review_element"


class FixHint(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: FixKind
    params: dict[str, Any] = Field(default_factory=dict)


class Finding(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    rule_id: str
    severity: Severity
    status: FindingStatus = FindingStatus.BROKEN
    title: str  # 짧은 한국어 제목
    message: str  # 사람이 읽는 설명 (무엇이 왜 끊겼나)
    service_id: str | None = None
    element: Element | None = None
    subject_ids: tuple[str, ...] = ()  # 관련 node/edge id (그래프에서 빨간색으로 표시)
    evidence: tuple[SourceSpan, ...] = ()
    fix: FixHint | None = None
    legal_basis: str | None = None

    @staticmethod
    def make_id(rule_id: str, *keys: str) -> str:
        h = hashlib.sha256("\x00".join((rule_id, *keys)).encode()).hexdigest()[:10]
        return f"{rule_id}:{h}"


class RuleInfo(BaseModel):
    """Catalog entry for a fixed rule (published so customers can audit the judge)."""

    model_config = ConfigDict(frozen=True)

    rule_id: str
    title: str
    severity: Severity
    element: Element | None
    rationale: str
    legal_basis: str | None = None


class ElementStatus(StrEnum):
    PASS = "pass"
    PARTIAL = "partial"
    FAIL = "fail"
    UNKNOWN = "unknown"  # 확인 불가 — scored as FAIL


class ElementScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    element: Element
    score: float = Field(ge=0.0, le=1.0)
    status: ElementStatus
    reasons: tuple[str, ...] = ()
    finding_ids: tuple[str, ...] = ()


class ServiceReadiness(BaseModel):
    model_config = ConfigDict(frozen=True)

    service_id: str
    service_name: str
    tier: int
    weight: float
    score: float = Field(ge=0.0, le=100.0)
    elements: tuple[ElementScore, ...]


class ReadinessReport(BaseModel):
    """준비도 점수(0~100). Weights are versioned and published (docs/scoring.md)."""

    model_config = ConfigDict(frozen=True)

    overall: float = Field(ge=0.0, le=100.0)
    services: tuple[ServiceReadiness, ...]
    weights_version: str
    element_weights: dict[str, float]
    tier_weights: dict[int, float]
    graph_hash: str
