"""Mock-incident rehearsal (모의 장애 리허설) contracts.

Scenarios may be written by templates, humans, or an LLM — but they are *data*: the
deterministic replayer (``jettae.rehearse.simulator``) is the only judge, and it never
touches real systems.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from jettae.domain.findings import Element
from jettae.domain.graph import IncidentType


class PerturbationKind(StrEnum):
    PERSON_UNAVAILABLE = "person_unavailable"  # 담당자 부재 (휴가·수면·퇴사 미반영)
    CHANNEL_DOWN = "channel_down"  # 메신저 다운
    SYSTEM_DOWN = "system_down"  # 의존 시스템 장애
    LOCATION_DOWN = "location_down"  # 센터/리전 장애 (화재 등)
    CONCURRENT_INCIDENT = "concurrent_incident"  # 동시 장애 (같은 사람이 두 곳에 필요)


class Perturbation(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: PerturbationKind
    target_id: str  # node id (person/channel/system/location/service)
    start_minute: float = 0.0
    end_minute: float | None = None  # None = 끝날 때까지


class ScenarioOrigin(StrEnum):
    TEMPLATE = "template"
    LLM = "llm"
    HUMAN = "human"


class Scenario(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    description: str = ""
    service_id: str
    incident_type: IncidentType = IncidentType.OUTAGE
    perturbations: tuple[Perturbation, ...] = ()
    deadline_minutes: float | None = None  # None -> policy tier SLO
    origin: ScenarioOrigin = ScenarioOrigin.TEMPLATE

    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json", exclude={"title", "description", "origin"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


class SimConfig(BaseModel):
    """Published replay assumptions. Changing them changes results -> part of result hash."""

    model_config = ConfigDict(frozen=True)

    detection_delay_minutes: float = 5.0  # 감지 신호가 있을 때 기본 감지 지연
    undetected_discovery_minutes: float | None = None  # None = 감지 경로 없으면 실패 처리
    ack_minutes: float = 5.0  # 연락 가능한 사람이 응답하는 시간
    default_escalation_after_minutes: float | None = None  # None = 상향 규칙 없으면 상향 안 함
    approval_minutes: float = 5.0
    default_action_minutes: float = 10.0
    max_hops: int = 12  # 무한 상향 방지


class SimEventKind(StrEnum):
    INCIDENT_START = "incident_start"
    DETECTED = "detected"
    UNDETECTED = "undetected"
    PAGE = "page"  # 호출 시도 (사람 + 채널)
    CHANNEL_UNAVAILABLE = "channel_unavailable"
    NO_ANSWER = "no_answer"
    ACK = "ack"
    ESCALATE = "escalate"
    DEAD_END = "dead_end"  # 더 이상 상향할 곳 없음
    APPROVAL_REQUESTED = "approval_requested"
    APPROVED = "approved"
    APPROVAL_BLOCKED = "approval_blocked"
    ACTION_STARTED = "action_started"
    MITIGATED = "mitigated"
    NO_MITIGATION = "no_mitigation"
    REPORT_FILED = "report_filed"
    REPORT_MISSED = "report_missed"
    DEADLINE_MISSED = "deadline_missed"


class SimEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    t: float  # minutes since incident start
    kind: SimEventKind
    actor_id: str | None = None
    target_id: str | None = None
    edge_id: str | None = None
    note: str = ""  # 한국어 설명 (타임라인 표시용)


class Outcome(StrEnum):
    PASS = "pass"  # 시한 안 종료 = 작동
    FAIL = "fail"


class RehearsalResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    scenario_id: str
    scenario_fingerprint: str
    graph_hash: str
    config_hash: str
    outcome: Outcome
    time_to_mitigate: float | None
    deadline_minutes: float
    failed_element: Element | None = None
    failed_at_id: str | None = None  # node/edge where the path broke
    reason: str = ""
    events: tuple[SimEvent, ...]
    path_ids: tuple[str, ...] = Field(default=(), description="nodes/edges traversed (for highlighting)")

    def result_hash(self) -> str:
        """Reproducibility KPI: same graph + scenario + config -> identical hash (100%)."""
        payload = self.model_dump(mode="json")
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
