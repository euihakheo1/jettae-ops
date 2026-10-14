"""Tests for the deterministic rehearsal replayer (jettae.rehearse.simulator)."""

from __future__ import annotations

import ast
import json
import math
import unicodedata
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from jettae.check.context import CheckContext
from jettae.domain import (
    EDGE_SIGNATURES,
    Attr,
    Edge,
    EdgeKind,
    Element,
    Extractor,
    GraphBuilder,
    IncidentType,
    Node,
    NodeKind,
    ObligationRule,
    OpsGraph,
    Outcome,
    PersonStatus,
    Perturbation,
    PerturbationKind,
    Policy,
    Provenance,
    RehearsalResult,
    RosterEntry,
    Scenario,
    SimConfig,
    SimEventKind,
    SourceKind,
    Verification,
    edge_id,
    node_id,
)
from jettae.rehearse import config_hash, replay, replay_all
from jettae.rehearse.simulator import (
    _SUBJECT,
    _TOWARD,
    _attach,
    applicable_obligation_rules,
    can_respond,
    choose_mitigation,
    effective_status,
    first_contacts,
    first_responders,
    paging_channels,
    requires_approval,
    trusted_view,
)

PROV = (Provenance(source=SourceKind.CATALOG, extractor=Extractor.IMPORT, ref="test"),)

SERVICE = node_id(NodeKind.SERVICE, "결제")
TRIGGER = node_id(NodeKind.TRIGGER, "결제 오류율 알람")
KIM = node_id(NodeKind.PERSON, "김대리")
PARK = node_id(NodeKind.PERSON, "박팀장")
LEAD = node_id(NodeKind.ROLE, "팀장")
SLACK = node_id(NodeKind.CHANNEL, "슬랙")
PHONE = node_id(NodeKind.CHANNEL, "전화")
ROLLBACK = node_id(NodeKind.ACTION, "결제 롤백")
OBLIGATION = node_id(NodeKind.OBLIGATION, "KISA 침해사고 신고")
SECURITY = node_id(NodeKind.PERSON, "최보안")
IDC = node_id(NodeKind.LOCATION, "IDC-A")
MESSENGER_DB = node_id(NodeKind.SYSTEM, "메신저 DB")
ORDER = node_id(NodeKind.SERVICE, "주문")


# --------------------------------------------------------------------------- graph DSL


def _attrs(values: dict[str, Any]) -> dict[str, Attr]:
    return {k: Attr(value=v, source=SourceKind.CATALOG) for k, v in values.items()}


class Demo:
    """Tiny builder for the competition demo graph (결제 서비스)."""

    def __init__(self) -> None:
        self.b = GraphBuilder()

    def node(self, kind: NodeKind, name: str, **attrs: Any) -> str:
        nid = node_id(kind, name)
        self.b.add_node(Node(id=nid, kind=kind, name=name, attrs=_attrs(attrs), prov=PROV))
        return nid

    def edge(self, kind: EdgeKind, src: str, dst: str, **attrs: Any) -> str:
        eid = edge_id(kind, src, dst)
        self.b.add_edge(Edge(id=eid, kind=kind, src=src, dst=dst, attrs=_attrs(attrs), prov=PROV))
        return eid

    def build(self) -> OpsGraph:
        return self.b.build()


def base(kim_status: str = "active", *, notify_channel: bool = True) -> Demo:
    """결제 → 알람(3분) → 김대리(슬랙) ; 롤백(10분)."""
    d = Demo()
    d.node(NodeKind.SERVICE, "결제", tier=1)
    d.node(NodeKind.TRIGGER, "결제 오류율 알람", delay_minutes=3)
    d.node(NodeKind.PERSON, "김대리", status=kim_status, title="대리")
    d.node(NodeKind.CHANNEL, "슬랙", medium="slack")
    d.node(NodeKind.ACTION, "결제 롤백", action_type="rollback", duration_minutes=10)
    d.edge(EdgeKind.DETECTED_BY, SERVICE, TRIGGER)
    if notify_channel:
        d.edge(EdgeKind.NOTIFIES, TRIGGER, KIM, channel=SLACK)
    else:
        d.edge(EdgeKind.NOTIFIES, TRIGGER, KIM)
    d.edge(EdgeKind.OWNED_BY, SERVICE, KIM)
    d.edge(EdgeKind.REACHABLE_VIA, KIM, SLACK)
    d.edge(EdgeKind.MITIGATED_BY, SERVICE, ROLLBACK)
    return d


def add_lead(d: Demo, *, park_status: str = "active", escalate: bool = True, approver: bool = True) -> Demo:
    """팀장 역할(박팀장, 전화) + 김대리→팀장 상향(5분) + 롤백 승인자 팀장."""
    d.node(NodeKind.ROLE, "팀장")
    d.node(NodeKind.PERSON, "박팀장", status=park_status, title="팀장")
    d.node(NodeKind.CHANNEL, "전화", medium="phone")
    d.edge(EdgeKind.HOLDS_ROLE, PARK, LEAD)
    d.edge(EdgeKind.REACHABLE_VIA, PARK, PHONE)
    if escalate:
        d.edge(EdgeKind.ESCALATES_TO, KIM, LEAD, after_minutes=5)
    if approver:
        d.edge(EdgeKind.APPROVED_BY, ROLLBACK, LEAD)
    return d


def scenario(*perturbations: Perturbation, **kw: Any) -> Scenario:
    kw.setdefault("id", "s1")
    kw.setdefault("title", "테스트")
    kw.setdefault("service_id", SERVICE)
    return Scenario(perturbations=perturbations, **kw)


def kinds(result: RehearsalResult) -> list[SimEventKind]:
    return [e.kind for e in result.events]


def event(result: RehearsalResult, kind: SimEventKind) -> Any:
    return next(e for e in result.events if e.kind is kind)


# --------------------------------------------------------------------------- competition demo


def test_demo_departed_owner_without_escalation_fails_at_owner() -> None:
    graph = add_lead(base("departed"), escalate=False).build()
    result = replay(graph, scenario())
    assert result.outcome is Outcome.FAIL
    assert result.failed_element is Element.OWNER
    assert result.failed_at_id == KIM
    assert result.time_to_mitigate is None
    assert kinds(result) == [
        SimEventKind.INCIDENT_START,
        SimEventKind.DETECTED,
        SimEventKind.NO_ANSWER,
        SimEventKind.DEAD_END,
    ]
    assert "퇴사" in event(result, SimEventKind.NO_ANSWER).note
    assert "상향 경로 없음" in result.reason


def test_demo_escalation_to_active_lead_with_lead_approval_passes() -> None:
    graph = add_lead(base("departed")).build()
    result = replay(graph, scenario())
    assert result.outcome is Outcome.PASS, result.reason
    assert result.failed_element is None
    # detect 3 → escalate 8 → ack 13 → self-approval 18 → rollback 10 min → 28
    assert result.time_to_mitigate == 28.0
    assert result.deadline_minutes == 60.0  # tier 1 SLO from Policy
    assert kinds(result) == [
        SimEventKind.INCIDENT_START,
        SimEventKind.DETECTED,
        SimEventKind.NO_ANSWER,
        SimEventKind.ESCALATE,
        SimEventKind.PAGE,
        SimEventKind.ACK,
        SimEventKind.APPROVAL_REQUESTED,
        SimEventKind.APPROVED,
        SimEventKind.ACTION_STARTED,
        SimEventKind.MITIGATED,
    ]
    page = event(result, SimEventKind.PAGE)
    assert (page.t, page.actor_id, page.target_id) == (8.0, PHONE, PARK)
    assert page.note == "담당 호출: 박팀장(팀장)에게 전화로 호출"
    escalate = event(result, SimEventKind.ESCALATE)
    assert escalate.edge_id == edge_id(EdgeKind.ESCALATES_TO, KIM, LEAD)
    assert "팀장으로 상향" in escalate.note
    assert "직접 승인" in event(result, SimEventKind.APPROVAL_REQUESTED).note
    assert "응답자가 실행(가정)" in event(result, SimEventKind.ACTION_STARTED).note
    assert result.reason == "목표 복구 시간 60분 안에 완화(28분)"
    assert result.path_ids[:4] == (
        SERVICE,
        edge_id(EdgeKind.DETECTED_BY, SERVICE, TRIGGER),
        TRIGGER,
        edge_id(EdgeKind.NOTIFIES, TRIGGER, KIM),
    )
    assert {LEAD, PARK, PHONE, ROLLBACK, edge_id(EdgeKind.HOLDS_ROLE, PARK, LEAD)} <= set(result.path_ids)


def test_demo_only_channel_down_fails() -> None:
    graph = base().build()
    result = replay(graph, scenario(Perturbation(kind=PerturbationKind.CHANNEL_DOWN, target_id=SLACK)))
    assert result.outcome is Outcome.FAIL
    assert result.failed_element is Element.OWNER
    assert result.failed_at_id == KIM
    unavailable = event(result, SimEventKind.CHANNEL_UNAVAILABLE)
    assert (unavailable.actor_id, unavailable.target_id) == (SLACK, KIM)
    assert "모든 연락 수단 사용 불가" in event(result, SimEventKind.NO_ANSWER).note


def test_demo_fallback_channel_passes_when_primary_down() -> None:
    d = base()
    d.node(NodeKind.CHANNEL, "전화", medium="phone")
    fallback = d.edge(EdgeKind.FALLBACK_TO, SLACK, PHONE)
    graph = d.build()
    result = replay(graph, scenario(Perturbation(kind=PerturbationKind.CHANNEL_DOWN, target_id=SLACK)))
    # the owner is reached over the fallback; the rollback then needs an approver nobody named
    assert result.failed_element is Element.APPROVER
    page = event(result, SimEventKind.PAGE)
    assert (page.actor_id, page.target_id) == (PHONE, KIM)
    assert fallback in result.path_ids

    d2 = base()
    d2.node(NodeKind.CHANNEL, "전화", medium="phone")
    d2.edge(EdgeKind.FALLBACK_TO, SLACK, PHONE)
    d2.node(
        NodeKind.ACTION, "결제 롤백", action_type="rollback", duration_minutes=10, requires_approval=False
    )
    result2 = replay(d2.build(), scenario(Perturbation(kind=PerturbationKind.CHANNEL_DOWN, target_id=SLACK)))
    assert result2.outcome is Outcome.PASS, result2.reason
    assert result2.time_to_mitigate == 3 + 5 + 10


def test_escalation_cycle_is_a_dead_end() -> None:
    d = add_lead(base("departed"), park_status="departed")
    d.edge(EdgeKind.ESCALATES_TO, LEAD, KIM, after_minutes=5)
    result = replay(d.build(), scenario())
    assert result.outcome is Outcome.FAIL
    assert result.failed_element is Element.OWNER
    assert result.failed_at_id == LEAD
    dead_end = event(result, SimEventKind.DEAD_END)
    assert "상향 순환" in dead_end.note
    assert dead_end.t == 8.0
    assert "상향 순환" in result.reason


def test_self_loop_escalation_is_a_cycle() -> None:
    d = base("departed")
    d.edge(EdgeKind.ESCALATES_TO, KIM, KIM, after_minutes=1)
    result = replay(d.build(), scenario())
    assert "상향 순환" in event(result, SimEventKind.DEAD_END).note


def test_approval_blocked_when_only_approver_is_unreachable() -> None:
    d = add_lead(base("active"), park_status="on_leave", escalate=False)
    result = replay(d.build(), scenario())
    assert result.outcome is Outcome.FAIL
    assert result.failed_element is Element.APPROVER
    assert result.failed_at_id == LEAD
    blocked = event(result, SimEventKind.APPROVAL_BLOCKED)
    assert blocked.target_id == ROLLBACK
    assert any("장기 부재" in e.note for e in result.events if e.kind is SimEventKind.NO_ANSWER)
    assert SimEventKind.ACTION_STARTED not in kinds(result)


def test_rollback_without_any_approver_is_blocked_at_action() -> None:
    result = replay(base().build(), scenario())
    assert result.failed_element is Element.APPROVER
    assert result.failed_at_id == ROLLBACK
    assert "승인자 미지정" in result.reason


def test_approver_paged_with_escalation_when_responder_is_not_approver() -> None:
    d = add_lead(base("active"), escalate=False)
    result = replay(d.build(), scenario())
    assert result.outcome is Outcome.PASS, result.reason
    requested = event(result, SimEventKind.APPROVAL_REQUESTED)
    # 김대리 acks at 8, pages 팀장 at 8 → ack 13 → approved 18 → mitigated 28
    assert (requested.t, requested.actor_id, requested.target_id) == (13.0, KIM, PARK)
    assert event(result, SimEventKind.APPROVED).t == 18.0
    assert result.time_to_mitigate == 28.0


def test_requires_approval_false_overrides_rollback_default() -> None:
    d = base()
    d.node(NodeKind.ACTION, "결제 롤백", action_type="rollback", duration_minutes=10, requires_approval=False)
    result = replay(d.build(), scenario())
    assert result.outcome is Outcome.PASS
    assert SimEventKind.APPROVAL_REQUESTED not in kinds(result)
    assert result.time_to_mitigate == 18.0


def _restart_named_approver(**attrs: Any) -> OpsGraph:
    """Payment mitigated only by a restart whose (departed) approver the runbook names."""
    d = add_lead(base("active"), escalate=False, approver=False, park_status="departed")
    d.node(NodeKind.ACTION, "재시작", action_type="restart", **attrs)
    restart = node_id(NodeKind.ACTION, "재시작")
    d.b.edges.pop(edge_id(EdgeKind.MITIGATED_BY, SERVICE, ROLLBACK))
    d.edge(EdgeKind.MITIGATED_BY, SERVICE, restart)
    d.edge(EdgeKind.APPROVED_BY, restart, LEAD)
    return d.build()


def test_approver_edge_alone_does_not_gate_a_restart() -> None:
    """Single rule (CheckContext.requires_approval): an APPROVED_BY edge says who, never whether."""
    result = replay(_restart_named_approver(), scenario())
    timeline = kinds(result)
    assert SimEventKind.APPROVAL_REQUESTED not in timeline
    assert SimEventKind.APPROVAL_BLOCKED not in timeline
    assert result.failed_element is not Element.APPROVER


def test_restart_the_runbook_gates_waits_for_its_departed_approver() -> None:
    result = replay(_restart_named_approver(requires_approval=True), scenario())
    assert result.failed_element is Element.APPROVER


def test_breach_report_missing_fails_citing_the_law() -> None:
    graph = add_lead(base()).build()
    result = replay(graph, scenario(incident_type=IncidentType.BREACH))
    assert result.outcome is Outcome.FAIL
    assert result.failed_element is None
    assert result.failed_at_id == SERVICE
    assert result.time_to_mitigate == 28.0  # mitigation itself worked
    missed = event(result, SimEventKind.REPORT_MISSED)
    assert missed.t == 3.0
    assert "정보통신망법 제48조의3" in result.reason
    assert "정보통신망법 제48조의3" in missed.note


def breach_graph(
    *, reporter_status: str = "active", deadline: int | None = 1440, reporter: bool = True
) -> OpsGraph:
    d = add_lead(base())
    attrs: dict[str, Any] = {"incident_type": "breach", "recipient": "KISA"}
    if deadline is not None:
        attrs["deadline_minutes"] = deadline
    d.node(NodeKind.OBLIGATION, "KISA 침해사고 신고", **attrs)
    d.edge(EdgeKind.MUST_REPORT, SERVICE, OBLIGATION)
    d.node(NodeKind.PERSON, "최보안", status=reporter_status)
    d.edge(EdgeKind.REACHABLE_VIA, SECURITY, PHONE)
    if reporter:
        d.edge(EdgeKind.REPORTED_BY, OBLIGATION, SECURITY)
    return d.build()


def test_breach_report_filed_in_time_passes() -> None:
    result = replay(breach_graph(), scenario(incident_type=IncidentType.BREACH))
    assert result.outcome is Outcome.PASS, result.reason
    filed = event(result, SimEventKind.REPORT_FILED)
    assert (filed.t, filed.actor_id, filed.target_id) == (8.0, SECURITY, OBLIGATION)
    assert "신고 1건" in result.reason
    # events stay chronologically ordered although reporting is simulated after mitigation
    assert [e.t for e in result.events] == sorted(e.t for e in result.events)


def test_breach_reporter_missing_or_unreachable_fails() -> None:
    no_reporter = replay(breach_graph(reporter=False), scenario(incident_type=IncidentType.BREACH))
    assert no_reporter.outcome is Outcome.FAIL
    assert no_reporter.failed_at_id == OBLIGATION
    assert "신고 담당 미지정" in no_reporter.reason
    assert "정보통신망법" in no_reporter.reason

    departed = replay(breach_graph(reporter_status="departed"), scenario(incident_type=IncidentType.BREACH))
    assert departed.outcome is Outcome.FAIL
    assert departed.failed_at_id == SECURITY
    assert departed.failed_element is None


def test_report_deadline_is_stricter_of_runbook_and_law() -> None:
    lenient = replay(breach_graph(deadline=48 * 60), scenario(incident_type=IncidentType.BREACH))
    assert lenient.outcome is Outcome.PASS  # filed at 8 min, well within 24h

    # report must reach the reporter only through a long escalation → late against the 24h law
    d = add_lead(base())
    d.node(NodeKind.OBLIGATION, "KISA 침해사고 신고", incident_type="breach", deadline_minutes=48 * 60)
    d.edge(EdgeKind.MUST_REPORT, SERVICE, OBLIGATION)
    d.node(NodeKind.PERSON, "최보안", status="active")
    d.node(NodeKind.ROLE, "보안 담당")
    security_role = node_id(NodeKind.ROLE, "보안 담당")
    d.edge(EdgeKind.REPORTED_BY, OBLIGATION, security_role)
    d.edge(EdgeKind.ESCALATES_TO, security_role, SECURITY, after_minutes=30 * 60)
    d.edge(EdgeKind.REACHABLE_VIA, SECURITY, PHONE)
    late = replay(d.build(), scenario(incident_type=IncidentType.BREACH))
    assert late.outcome is Outcome.FAIL
    missed = event(late, SimEventKind.REPORT_MISSED)
    assert missed.t == 3 + 24 * 60
    assert "24시간 초과" in missed.note
    filed = event(late, SimEventKind.REPORT_FILED)
    assert filed.t == 3 + 30 * 60 + 5
    assert late.events.index(missed) < late.events.index(filed)


def test_obligation_without_any_deadline_cannot_be_confirmed() -> None:
    policy = Policy(obligations=())
    result = replay(breach_graph(deadline=None), scenario(incident_type=IncidentType.BREACH), policy=policy)
    assert result.outcome is Outcome.FAIL
    assert "신고 시한 미기재" in result.reason


def test_out_of_scope_service_needs_no_report() -> None:
    policy = Policy(regulated_services=("service:다른-서비스",))
    graph = add_lead(base()).build()
    result = replay(graph, scenario(incident_type=IncidentType.BREACH), policy=policy)
    assert result.outcome is Outcome.PASS
    assert SimEventKind.REPORT_MISSED not in kinds(result)


def test_privacy_leak_uses_privacy_rule() -> None:
    graph = add_lead(base()).build()
    result = replay(graph, scenario(incident_type=IncidentType.PRIVACY_LEAK))
    assert result.outcome is Outcome.FAIL
    assert "개인정보 보호법 제34조" in result.reason


def test_deadline_missed_is_emitted_at_the_deadline() -> None:
    graph = add_lead(base("departed")).build()
    result = replay(graph, scenario(deadline_minutes=20))
    assert result.outcome is Outcome.FAIL
    assert result.failed_element is None
    assert result.failed_at_id == SERVICE
    missed = event(result, SimEventKind.DEADLINE_MISSED)
    assert missed.t == 20.0
    assert result.time_to_mitigate == 28.0
    assert result.deadline_minutes == 20.0
    assert "목표 복구 시간 20분 초과" in result.reason


def test_mitigation_exactly_at_deadline_passes() -> None:
    graph = add_lead(base("departed")).build()
    assert replay(graph, scenario(deadline_minutes=28)).outcome is Outcome.PASS


# --------------------------------------------------------------------------- detection


def test_no_trigger_is_undetected_failure() -> None:
    d = add_lead(base())
    d.b.edges.pop(edge_id(EdgeKind.DETECTED_BY, SERVICE, TRIGGER))
    result = replay(d.build(), scenario())
    assert result.outcome is Outcome.FAIL
    assert result.failed_element is Element.DETECTION
    assert result.failed_at_id == SERVICE
    assert kinds(result) == [SimEventKind.INCIDENT_START, SimEventKind.UNDETECTED]


def test_undetected_discovery_assumption_pages_owners() -> None:
    d = add_lead(base())
    d.b.edges.pop(edge_id(EdgeKind.DETECTED_BY, SERVICE, TRIGGER))
    result = replay(d.build(), scenario(), config=SimConfig(undetected_discovery_minutes=30))
    detected = event(result, SimEventKind.DETECTED)
    assert detected.t == 30.0
    assert "고객 문의로 인지(가정)" in detected.note
    assert "담당자 직접 연락(가정)" in detected.note
    page = event(result, SimEventKind.PAGE)
    assert page.edge_id == edge_id(EdgeKind.OWNED_BY, SERVICE, KIM)
    assert result.time_to_mitigate == 30 + 5 + 5 + 5 + 10


def test_trigger_without_notifies_pages_owner_with_assumption() -> None:
    d = add_lead(base())
    d.b.edges.pop(edge_id(EdgeKind.NOTIFIES, TRIGGER, KIM))
    result = replay(d.build(), scenario())
    assert "호출 대상 미지정 → 담당자 직접 연락(가정)" in event(result, SimEventKind.DETECTED).note
    assert result.outcome is Outcome.PASS


def test_trigger_delay_defaults_to_config_and_earliest_trigger_wins() -> None:
    d = add_lead(base())
    d.node(NodeKind.TRIGGER, "고객 문의")
    inquiry = node_id(NodeKind.TRIGGER, "고객 문의")
    d.edge(EdgeKind.DETECTED_BY, SERVICE, inquiry)
    result = replay(d.build(), scenario(), config=SimConfig(detection_delay_minutes=1))
    detected = event(result, SimEventKind.DETECTED)
    assert (detected.t, detected.actor_id) == (1.0, inquiry)
    # the inquiry trigger notifies nobody, the alarm pages 김대리 when it fires at 3
    assert event(result, SimEventKind.PAGE).t == 3.0


def test_no_owner_and_no_notifies_is_owner_dead_end_at_service() -> None:
    d = Demo()
    d.node(NodeKind.SERVICE, "결제", tier=1)
    d.node(NodeKind.TRIGGER, "결제 오류율 알람", delay_minutes=3)
    d.edge(EdgeKind.DETECTED_BY, SERVICE, TRIGGER)
    result = replay(d.build(), scenario())
    assert result.failed_element is Element.OWNER
    assert result.failed_at_id == SERVICE
    assert "호출 대상·담당자 미지정" in result.reason


def test_unknown_service_fails_without_raising() -> None:
    result = replay(base().build(), scenario(service_id="service:없음"))
    assert result.outcome is Outcome.FAIL
    assert result.failed_element is Element.DETECTION
    assert result.deadline_minutes == 240.0  # default tier 2 SLO
    assert result.path_ids == ()


# --------------------------------------------------------------------------- escalation details


def test_missing_after_minutes_is_not_invented() -> None:
    d = add_lead(base("departed"), escalate=False)
    d.edge(EdgeKind.ESCALATES_TO, KIM, LEAD)
    graph = d.build()
    result = replay(graph, scenario())
    assert result.failed_element is Element.OWNER
    assert "상향 대기시간 미기재로 상향 시점 불명" in result.reason
    with_default = replay(graph, scenario(), config=SimConfig(default_escalation_after_minutes=10))
    assert with_default.outcome is Outcome.PASS
    assert event(with_default, SimEventKind.ESCALATE).t == 13.0


def test_max_hops_bounds_escalation() -> None:
    d = add_lead(base("departed"))
    graph = d.build()
    blocked = replay(graph, scenario(), config=SimConfig(max_hops=0))
    assert blocked.failed_element is Element.OWNER
    assert "최대 상향 단계(0) 초과" in blocked.reason
    assert replay(graph, scenario(), config=SimConfig(max_hops=1)).outcome is Outcome.PASS


def test_role_without_holders_then_team_reached_on_shared_channel() -> None:
    d = base("departed")
    d.node(NodeKind.TEAM, "결제팀")
    team = node_id(NodeKind.TEAM, "결제팀")
    d.node(NodeKind.ROLE, "팀장")
    d.edge(EdgeKind.ESCALATES_TO, KIM, LEAD, after_minutes=5)  # role with no holder
    d.edge(EdgeKind.ESCALATES_TO, LEAD, team, after_minutes=5)
    d.node(NodeKind.PERSON, "박팀장", status="active")
    d.node(NodeKind.CHANNEL, "전화")
    d.edge(EdgeKind.MEMBER_OF, PARK, team)
    d.edge(EdgeKind.REACHABLE_VIA, team, PHONE)  # team-level shared channel
    d.node(NodeKind.ACTION, "결제 롤백", action_type="rollback", duration_minutes=10, requires_approval=False)
    result = replay(d.build(), scenario())
    assert result.outcome is Outcome.PASS, result.reason
    no_answers = [e for e in result.events if e.kind is SimEventKind.NO_ANSWER]
    assert any(e.target_id == LEAD and "사람이 없음" in e.note for e in no_answers)
    page = event(result, SimEventKind.PAGE)
    assert (page.t, page.actor_id, page.target_id) == (13.0, PHONE, PARK)
    assert edge_id(EdgeKind.MEMBER_OF, PARK, team) in result.path_ids


def test_escalation_from_a_resolved_person_is_followed() -> None:
    d = Demo()
    d.node(NodeKind.SERVICE, "결제", tier=1)
    d.node(NodeKind.TRIGGER, "결제 오류율 알람", delay_minutes=3)
    d.node(NodeKind.ROLE, "팀장")
    d.node(NodeKind.PERSON, "박팀장", status="on_leave")
    d.node(NodeKind.PERSON, "최부장", status="active")
    choi = node_id(NodeKind.PERSON, "최부장")
    d.node(NodeKind.CHANNEL, "전화")
    d.edge(EdgeKind.DETECTED_BY, SERVICE, TRIGGER)
    d.edge(EdgeKind.NOTIFIES, TRIGGER, LEAD)
    d.edge(EdgeKind.HOLDS_ROLE, PARK, LEAD)
    personal = d.edge(EdgeKind.ESCALATES_TO, PARK, choi, after_minutes=10)  # 박팀장's own deputy
    d.edge(EdgeKind.REACHABLE_VIA, choi, PHONE)
    result = replay(d.build(), scenario())
    escalate = event(result, SimEventKind.ESCALATE)
    assert (escalate.t, escalate.actor_id, escalate.target_id, escalate.edge_id) == (
        13.0,
        PARK,
        choi,
        personal,
    )
    assert event(result, SimEventKind.ACK).actor_id == choi
    assert result.failed_element is Element.ROLLBACK  # no mitigation in this graph


def test_parallel_notifies_earliest_escalation_wins() -> None:
    d = add_lead(base("departed"))
    d.node(NodeKind.PERSON, "이주임", status="departed")
    lee = node_id(NodeKind.PERSON, "이주임")
    d.edge(EdgeKind.NOTIFIES, TRIGGER, lee)
    d.edge(EdgeKind.ESCALATES_TO, lee, LEAD, after_minutes=2)  # reaches 팀장 earlier than 김대리's 5
    result = replay(d.build(), scenario())
    assert result.outcome is Outcome.PASS
    escalations = [e for e in result.events if e.kind is SimEventKind.ESCALATE]
    assert len(escalations) == 1
    assert (escalations[0].t, escalations[0].actor_id) == (5.0, lee)


def test_person_unavailable_window_is_half_open() -> None:
    graph = add_lead(base()).build()
    absent_until_3 = scenario(
        Perturbation(kind=PerturbationKind.PERSON_UNAVAILABLE, target_id=KIM, start_minute=0, end_minute=3)
    )
    assert event(replay(graph, absent_until_3), SimEventKind.PAGE).target_id == KIM
    absent_until_4 = scenario(
        Perturbation(kind=PerturbationKind.PERSON_UNAVAILABLE, target_id=KIM, start_minute=0, end_minute=4)
    )
    result = replay(graph, absent_until_4)
    assert event(result, SimEventKind.PAGE).target_id == PARK
    assert "시나리오상 부재" in event(result, SimEventKind.NO_ANSWER).note
    assert "김대리 부재(0~4분)" in event(result, SimEventKind.INCIDENT_START).note


# --------------------------------------------------------------------------- perturbations


def test_location_down_takes_down_dependent_channels() -> None:
    d = add_lead(base(), escalate=False)
    d.node(NodeKind.LOCATION, "IDC-A")
    d.node(NodeKind.SYSTEM, "메신저 DB")
    d.edge(EdgeKind.DEPENDS_ON, SLACK, MESSENGER_DB)
    d.edge(EdgeKind.RUNS_IN, MESSENGER_DB, IDC)
    d.edge(EdgeKind.RUNS_IN, SERVICE, IDC)
    graph = d.build()
    result = replay(graph, scenario(Perturbation(kind=PerturbationKind.LOCATION_DOWN, target_id=IDC)))
    assert result.failed_element is Element.OWNER  # 김대리 only has 슬랙, which lives in IDC-A
    start = event(result, SimEventKind.INCIDENT_START).note
    assert "영향 채널: 슬랙" in start
    assert "영향 서비스: 결제" in start
    assert "IDC-A 장애 영향" in event(result, SimEventKind.CHANNEL_UNAVAILABLE).note
    system = replay(graph, scenario(Perturbation(kind=PerturbationKind.SYSTEM_DOWN, target_id=MESSENGER_DB)))
    assert system.failed_element is Element.OWNER
    recovered = replay(
        graph,
        scenario(Perturbation(kind=PerturbationKind.SYSTEM_DOWN, target_id=MESSENGER_DB, end_minute=3)),
    )
    assert recovered.outcome is Outcome.PASS


def test_concurrent_incident_occupies_shared_first_responder() -> None:
    d = add_lead(base())
    d.node(NodeKind.SERVICE, "주문", tier=1)
    d.edge(EdgeKind.OWNED_BY, ORDER, KIM)
    graph = d.build()
    result = replay(graph, scenario(Perturbation(kind=PerturbationKind.CONCURRENT_INCIDENT, target_id=ORDER)))
    assert "주문 동시 장애 — 김대리 투입" in event(result, SimEventKind.INCIDENT_START).note
    no_answer = event(result, SimEventKind.NO_ANSWER)
    assert no_answer.target_id == KIM
    assert "주문 동시 장애 대응 중" in no_answer.note
    assert result.outcome is Outcome.PASS  # 팀장 takes over via escalation
    assert event(result, SimEventKind.PAGE).target_id == PARK


def test_concurrent_incident_without_free_responder_is_noted() -> None:
    d = add_lead(base())
    d.node(NodeKind.SERVICE, "주문", tier=1)
    graph = d.build()
    result = replay(graph, scenario(Perturbation(kind=PerturbationKind.CONCURRENT_INCIDENT, target_id=ORDER)))
    assert "투입될 1차 대응자 없음" in event(result, SimEventKind.INCIDENT_START).note


def test_roster_status_overrides_graph_status() -> None:
    graph = add_lead(base("active"), escalate=False).build()
    roster = (RosterEntry(name="김대리", status=PersonStatus.DEPARTED),)
    result = replay(graph, scenario(), roster=roster)
    assert result.failed_element is Element.OWNER
    assert "퇴사" in event(result, SimEventKind.NO_ANSWER).note
    assert result.config_hash != replay(graph, scenario()).config_hash


def test_missing_status_is_unknown_and_not_available() -> None:
    d = Demo()
    d.node(NodeKind.SERVICE, "결제", tier=1)
    d.node(NodeKind.TRIGGER, "결제 오류율 알람", delay_minutes=3)
    d.node(NodeKind.PERSON, "김대리")  # no status attribute
    d.edge(EdgeKind.DETECTED_BY, SERVICE, TRIGGER)
    d.edge(EdgeKind.NOTIFIES, TRIGGER, KIM)
    result = replay(d.build(), scenario())
    assert "재직 여부 확인 불가" in event(result, SimEventKind.NO_ANSWER).note


def test_person_without_channels_cannot_be_reached() -> None:
    d = base(notify_channel=False)
    d.b.edges.pop(edge_id(EdgeKind.REACHABLE_VIA, KIM, SLACK))
    result = replay(d.build(), scenario())
    assert "연락 수단 미기재" in event(result, SimEventKind.NO_ANSWER).note


def test_hold_elements_are_excluded() -> None:
    d = add_lead(base("departed"))
    esc = edge_id(EdgeKind.ESCALATES_TO, KIM, LEAD)
    d.b.edges[esc] = d.b.edges[esc].model_copy(update={"verification": Verification.HOLD})
    graph = d.build()
    result = replay(graph, scenario())
    assert result.outcome is Outcome.FAIL
    assert result.failed_element is Element.OWNER
    assert "판단 보류 요소 1개" in event(result, SimEventKind.INCIDENT_START).note
    assert result.graph_hash == graph.content_hash()


# --------------------------------------------------------------------------- mitigation details


def test_no_mitigation_fails_rollback() -> None:
    d = base()
    d.b.edges.pop(edge_id(EdgeKind.MITIGATED_BY, SERVICE, ROLLBACK))
    result = replay(d.build(), scenario())
    assert result.failed_element is Element.ROLLBACK
    assert result.failed_at_id == SERVICE
    assert event(result, SimEventKind.NO_MITIGATION).t == 8.0


def test_preferred_action_and_designated_executor() -> None:
    d = add_lead(base())
    d.b.edges.pop(edge_id(EdgeKind.MITIGATED_BY, SERVICE, ROLLBACK))
    d.node(NodeKind.ACTION, "가동 재개", action_type="restart", requires_approval=False)
    d.node(
        NodeKind.ACTION,
        "기능 플래그 끄기",
        action_type="feature_flag",
        requires_approval=False,
        duration_minutes=2,
    )
    restart = node_id(NodeKind.ACTION, "가동 재개")
    flag = node_id(NodeKind.ACTION, "기능 플래그 끄기")
    assert restart < flag  # id order alone would pick the restart
    d.edge(EdgeKind.MITIGATED_BY, SERVICE, restart)
    d.edge(EdgeKind.MITIGATED_BY, SERVICE, flag)
    d.edge(EdgeKind.EXECUTED_BY, flag, LEAD)
    graph = d.build()
    chosen = choose_mitigation(CheckContext(graph), SERVICE)
    assert chosen is not None
    assert chosen.dst == flag  # feature_flag is preferred over restart
    result = replay(graph, scenario())
    started = event(result, SimEventKind.ACTION_STARTED)
    assert (started.actor_id, started.target_id) == (PARK, flag)
    assert started.edge_id == edge_id(EdgeKind.EXECUTED_BY, flag, LEAD)
    assert result.time_to_mitigate == 3 + 5 + 2


def test_unavailable_designated_executor_falls_back_to_responder() -> None:
    d = add_lead(base(), park_status="departed")
    d.node(NodeKind.ACTION, "결제 롤백", action_type="rollback", requires_approval=False)
    d.edge(EdgeKind.EXECUTED_BY, ROLLBACK, LEAD)
    result = replay(d.build(), scenario())
    started = event(result, SimEventKind.ACTION_STARTED)
    assert started.actor_id == KIM
    assert "지정 실행자 응답 불가 → 응답자가 실행(가정)" in started.note
    mitigated = event(result, SimEventKind.MITIGATED)
    assert "소요 시간 미기재로 기본값 가정" not in mitigated.note  # duration 10 kept from first writer


def test_default_action_duration_is_noted() -> None:
    d = Demo()
    d.node(NodeKind.SERVICE, "결제", tier=1)
    d.node(NodeKind.TRIGGER, "결제 오류율 알람", delay_minutes=3)
    d.node(NodeKind.PERSON, "김대리", status="active")
    d.node(NodeKind.CHANNEL, "슬랙")
    d.node(NodeKind.ACTION, "재시작", action_type="restart")
    restart = node_id(NodeKind.ACTION, "재시작")
    d.edge(EdgeKind.DETECTED_BY, SERVICE, TRIGGER)
    d.edge(EdgeKind.NOTIFIES, TRIGGER, KIM)
    d.edge(EdgeKind.REACHABLE_VIA, KIM, SLACK)
    d.edge(EdgeKind.MITIGATED_BY, SERVICE, restart)
    result = replay(d.build(), scenario(), config=SimConfig(default_action_minutes=7))
    assert result.time_to_mitigate == 3 + 5 + 7
    assert "기본값 가정" in event(result, SimEventKind.MITIGATED).note


# --------------------------------------------------------------------------- hashes & helpers


def test_hashes_and_replay_all_order() -> None:
    graph = add_lead(base("departed")).build()
    s1 = scenario()
    s2 = scenario(Perturbation(kind=PerturbationKind.CHANNEL_DOWN, target_id=PHONE), id="s2")
    results = replay_all(graph, [s2, s1], policy=Policy(), config=SimConfig())
    assert [r.scenario_id for r in results] == ["s2", "s1"]
    assert results[1] == replay(graph, s1)
    assert results[1].graph_hash == graph.content_hash()
    assert results[1].scenario_fingerprint == s1.fingerprint()
    assert results[1].config_hash == config_hash(SimConfig(), Policy(), ())
    assert config_hash(SimConfig(ack_minutes=1)) != config_hash(SimConfig())
    assert config_hash(SimConfig(), Policy(tier_slo_minutes={1: 30})) != config_hash(SimConfig())
    assert config_hash(SimConfig(), Policy(fatigue=Policy().fatigue)) == config_hash(SimConfig())


def test_result_hash_is_reproducible() -> None:
    graph = breach_graph()
    s = scenario(
        Perturbation(kind=PerturbationKind.PERSON_UNAVAILABLE, target_id=KIM, end_minute=10),
        incident_type=IncidentType.BREACH,
    )
    hashes = {replay(graph, s).result_hash() for _ in range(3)}
    assert len(hashes) == 1


def test_shared_helpers() -> None:
    graph = add_lead(base("departed")).build()
    ctx = CheckContext(graph)
    kim = graph.nodes[KIM]
    assert effective_status(ctx, kim) is PersonStatus.DEPARTED
    assert not can_respond(ctx, kim)
    assert can_respond(ctx, graph.nodes[PARK])
    assert not can_respond(ctx.with_unavailable([PARK]), graph.nodes[PARK])
    assert first_responders(ctx, SERVICE) == []
    roster_ctx = CheckContext(graph, roster=(RosterEntry(name="김대리", status=PersonStatus.ACTIVE),))
    assert [p.id for p in first_responders(roster_ctx, SERVICE)] == [KIM]
    rules = applicable_obligation_rules(ctx, SERVICE, IncidentType.BREACH)
    assert [r.key for r in rules] == ["kr-breach-24h"]
    assert applicable_obligation_rules(ctx, "service:없음", IncidentType.BREACH) == []
    scoped = CheckContext(graph, policy=Policy(regulated_services=("결제",)))
    assert applicable_obligation_rules(scoped, SERVICE, IncidentType.PRIVACY_LEAK)
    not_applicable = Policy(
        obligations=(
            ObligationRule(
                key="x",
                incident_type=IncidentType.BREACH,
                recipient="r",
                deadline_minutes=1,
                legal_basis="b",
                applies=False,
            ),
        )
    )
    assert (
        applicable_obligation_rules(CheckContext(graph, policy=not_applicable), SERVICE, IncidentType.BREACH)
        == []
    )


def test_simulator_never_imports_llm_code() -> None:
    from jettae.rehearse import simulator

    tree = ast.parse(Path(simulator.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for stmt in ast.walk(tree):
        if isinstance(stmt, ast.ImportFrom) and stmt.module:
            imported.add(stmt.module)
        elif isinstance(stmt, ast.Import):
            imported.update(alias.name for alias in stmt.names)
    assert not any(
        m.startswith(("jettae.llm", "anthropic", "openai", "jettae.rehearse.scenarios")) for m in imported
    )
    assert {m for m in imported if m.startswith("jettae.check")} == {"jettae.check.context"}


# --------------------------------------------------------------------------- property tests

_POOL: dict[NodeKind, list[str]] = {
    NodeKind.SERVICE: ["s0", "s1"],
    NodeKind.SYSTEM: ["y0"],
    NodeKind.LOCATION: ["l0"],
    NodeKind.PERSON: ["p0", "p1", "p2"],
    NodeKind.ROLE: ["r0"],
    NodeKind.TEAM: ["t0"],
    NodeKind.CHANNEL: ["c0", "c1"],
    NodeKind.TRIGGER: ["g0", "g1"],
    NodeKind.ACTION: ["a0", "a1"],
    NodeKind.OBLIGATION: ["o0"],
}
_IDS: dict[NodeKind, list[str]] = {k: [node_id(k, n) for n in names] for k, names in _POOL.items()}
_ALL_IDS = sorted(i for ids in _IDS.values() for i in ids)
_CANDIDATE_EDGES = sorted(
    (kind, src, dst)
    for kind, (src_kinds, dst_kinds) in EDGE_SIGNATURES.items()
    for sk in src_kinds
    for dk in dst_kinds
    for src in _IDS[sk]
    for dst in _IDS[dk]
)

_minutes = st.one_of(
    st.none(), st.integers(0, 60), st.floats(0, 120, allow_nan=False), st.just("5분"), st.just(-3)
)
_STATUS = st.one_of(st.none(), st.sampled_from([s.value for s in PersonStatus]))


def _node_attrs(kind: NodeKind) -> st.SearchStrategy[dict[str, Any]]:
    def without_none(**fields: st.SearchStrategy[Any]) -> st.SearchStrategy[dict[str, Any]]:
        return st.fixed_dictionaries(fields).map(lambda d: {k: v for k, v in d.items() if v is not None})

    if kind is NodeKind.SERVICE:
        return without_none(
            tier=st.one_of(st.none(), st.integers(1, 3)),
            slo_minutes=st.one_of(st.none(), st.integers(5, 120)),
        )
    if kind is NodeKind.PERSON:
        return without_none(status=_STATUS)
    if kind is NodeKind.TRIGGER:
        return without_none(delay_minutes=_minutes)
    if kind is NodeKind.ACTION:
        return without_none(
            action_type=st.one_of(
                st.none(), st.sampled_from(["rollback", "failover", "restart", "feature_flag", "bogus"])
            ),
            requires_approval=st.one_of(st.none(), st.booleans()),
            duration_minutes=_minutes,
        )
    if kind is NodeKind.OBLIGATION:
        return without_none(
            incident_type=st.one_of(st.none(), st.sampled_from([t.value for t in IncidentType])),
            deadline_minutes=_minutes,
        )
    return st.just({})


@st.composite
def graphs(draw: st.DrawFn) -> OpsGraph:
    d = Demo()
    for kind, names in _POOL.items():
        for name in names:
            d.node(kind, name, **draw(_node_attrs(kind)))
    for ekind, src, dst in draw(st.lists(st.sampled_from(_CANDIDATE_EDGES), unique=True, max_size=30)):
        attrs: dict[str, Any] = {}
        if ekind is EdgeKind.ESCALATES_TO:
            after = draw(_minutes)
            if after is not None:
                attrs["after_minutes"] = after
        if ekind is EdgeKind.NOTIFIES and draw(st.booleans()):
            attrs["channel"] = draw(st.sampled_from([*_IDS[NodeKind.CHANNEL], "channel:없음", "person:p0"]))
        held = draw(st.integers(0, 9)) == 0
        eid = d.edge(ekind, src, dst, **attrs)
        if held:
            d.b.edges[eid] = d.b.edges[eid].model_copy(update={"verification": Verification.HOLD})
    return d.build()


_windows = st.tuples(
    st.floats(0, 60, allow_nan=False), st.one_of(st.none(), st.floats(0, 120, allow_nan=False))
)


@st.composite
def scenarios(draw: st.DrawFn) -> Scenario:
    perturbations = []
    for kind, target, (start, end) in draw(
        st.lists(
            st.tuples(
                st.sampled_from(list(PerturbationKind)), st.sampled_from([*_ALL_IDS, "x:없음"]), _windows
            ),
            max_size=4,
        )
    ):
        perturbations.append(Perturbation(kind=kind, target_id=target, start_minute=start, end_minute=end))
    return Scenario(
        id="prop",
        title="속성 시험",
        service_id=draw(st.sampled_from([*_IDS[NodeKind.SERVICE], "service:없음"])),
        incident_type=draw(st.sampled_from(list(IncidentType))),
        perturbations=tuple(perturbations),
        deadline_minutes=draw(st.one_of(st.none(), st.floats(0, 200, allow_nan=False))),
    )


configs = st.builds(
    SimConfig,
    detection_delay_minutes=st.floats(0, 10, allow_nan=False),
    undetected_discovery_minutes=st.one_of(st.none(), st.floats(0, 60, allow_nan=False)),
    ack_minutes=st.floats(0, 10, allow_nan=False),
    default_escalation_after_minutes=st.one_of(st.none(), st.floats(0, 30, allow_nan=False)),
    approval_minutes=st.floats(0, 10, allow_nan=False),
    default_action_minutes=st.floats(0, 30, allow_nan=False),
    max_hops=st.integers(0, 4),
)


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(graph=graphs(), sc=scenarios(), config=configs)
def test_replay_never_raises_and_is_reproducible(graph: OpsGraph, sc: Scenario, config: SimConfig) -> None:
    first = replay(graph, sc, config=config)
    second = replay(graph, sc, config=config)
    assert first.result_hash() == second.result_hash()
    assert first == second

    times = [e.t for e in first.events]
    assert times == sorted(times)
    assert first.events[0].kind is SimEventKind.INCIDENT_START
    assert len(set(first.path_ids)) == len(first.path_ids)
    assert set(first.path_ids) <= set(graph.nodes) | set(graph.edges)
    assert first.graph_hash == graph.content_hash()
    if first.outcome is Outcome.PASS:
        assert first.time_to_mitigate is not None
        assert first.time_to_mitigate <= first.deadline_minutes
        assert first.failed_element is None
        assert first.failed_at_id is None
        assert SimEventKind.MITIGATED in kinds(first)
    else:
        assert first.reason
        assert first.failed_at_id is not None
    if first.failed_element is Element.OWNER:
        assert SimEventKind.DEAD_END in kinds(first)


@settings(max_examples=40, deadline=None)
@given(graph=graphs(), sc=scenarios())
def test_perturbation_order_does_not_change_the_verdict(graph: OpsGraph, sc: Scenario) -> None:
    reordered = sc.model_copy(update={"perturbations": tuple(reversed(sc.perturbations))})
    a, b = replay(graph, sc), replay(graph, reordered)
    assert a.model_dump(exclude={"scenario_fingerprint"}) == b.model_dump(exclude={"scenario_fingerprint"})


@pytest.mark.parametrize(
    ("word", "expected"),
    [("김대리", "김대리가"), ("박팀장", "박팀장이"), ("Slack", "Slack이(가)")],
)
def test_korean_subject_particle(word: str, expected: str) -> None:
    assert _attach(word, _SUBJECT) == expected
    assert _attach("전화", _TOWARD) == "전화로"
    assert _attach("슬랙", _TOWARD) == "슬랙으로"
    assert _attach("메일", _TOWARD) == "메일로"


def test_escalating_into_a_node_another_branch_already_tried_is_not_called_a_cycle() -> None:
    d = base("departed")
    d.node(NodeKind.PERSON, "이주임", status="departed")
    lee = node_id(NodeKind.PERSON, "이주임")
    d.edge(EdgeKind.NOTIFIES, TRIGGER, lee)
    d.edge(EdgeKind.ESCALATES_TO, lee, KIM, after_minutes=1)  # 김대리 was paged in parallel already
    result = replay(d.build(), scenario())
    dead_end = event(result, SimEventKind.DEAD_END)
    assert dead_end.target_id == lee
    assert "이미 호출에 실패한 대상으로 상향" in dead_end.note
    assert "상향 순환" not in dead_end.note


def test_unrecognised_action_and_incident_types_stay_unknown() -> None:
    d = add_lead(base(), escalate=False)
    d.b.edges.pop(edge_id(EdgeKind.MITIGATED_BY, SERVICE, ROLLBACK))
    d.node(NodeKind.ACTION, "가속 조치", action_type="warp-speed", duration_minutes=4)
    odd = node_id(NodeKind.ACTION, "가속 조치")
    d.edge(EdgeKind.MITIGATED_BY, SERVICE, odd)
    d.node(NodeKind.OBLIGATION, "KISA 침해사고 신고", incident_type="meteor", deadline_minutes=60)
    d.edge(EdgeKind.MUST_REPORT, SERVICE, OBLIGATION)
    graph = d.build()
    outage = replay(graph, scenario())
    assert outage.outcome is Outcome.PASS  # unknown type → no default approval, no approver edge
    assert outage.time_to_mitigate == 3 + 5 + 4
    breach = replay(graph, scenario(incident_type=IncidentType.BREACH))
    # the "meteor" obligation does not count — but the runbook does have it, so the note must
    # not claim "no procedure"; it says applicability cannot be confirmed (P2/P3)
    assert breach.outcome is Outcome.FAIL
    assert "신고 절차가 런북에 없음" not in breach.reason
    assert "신고 의무(KISA 침해사고 신고)의 사고 유형이 미기재·인식 불가" in breach.reason
    assert "정보통신망법 제48조의3" in breach.reason
    assert SimEventKind.REPORT_FILED not in kinds(breach)


def test_window_descriptions_in_incident_start() -> None:
    graph = add_lead(base()).build()
    result = replay(
        graph,
        scenario(
            Perturbation(kind=PerturbationKind.CHANNEL_DOWN, target_id=PHONE, start_minute=10),
            Perturbation(kind=PerturbationKind.PERSON_UNAVAILABLE, target_id=PARK, end_minute=float("inf")),
            Perturbation(kind=PerturbationKind.PERSON_UNAVAILABLE, target_id=KIM, start_minute=float("nan")),
        ),
    )
    note = event(result, SimEventKind.INCIDENT_START).note
    assert "전화 다운(10분부터)" in note
    assert "박팀장 부재(0~∞분)" in note
    assert "김대리 부재(?분부터)" in note
    assert "2.5" in replay(graph, scenario(deadline_minutes=2.5)).reason


def test_a_node_reached_twice_is_paged_only_once() -> None:
    d = add_lead(base("departed"), park_status="departed")
    d.node(NodeKind.PERSON, "이주임", status="departed")
    lee = node_id(NodeKind.PERSON, "이주임")
    d.edge(EdgeKind.NOTIFIES, TRIGGER, lee)
    d.edge(EdgeKind.ESCALATES_TO, lee, LEAD, after_minutes=2)  # 팀장 at 5 (via 이주임) and 8 (via 김대리)
    result = replay(d.build(), scenario())
    escalations = [e for e in result.events if e.kind is SimEventKind.ESCALATE]
    assert [(e.t, e.actor_id) for e in escalations] == [(5.0, lee)]
    assert result.failed_at_id == LEAD
    assert event(result, SimEventKind.DEAD_END).t == 5.0


# --------------------------------------------------------------------------- reviewer: regressions


def _second_alarm(d: Demo, *, delay: float = 10, channel: str | None = None) -> str:
    """Add '지연 알람' on 결제 that also notifies 김대리 (optionally over ``channel``)."""
    d.node(NodeKind.TRIGGER, "지연 알람", delay_minutes=delay)
    slow = node_id(NodeKind.TRIGGER, "지연 알람")
    d.edge(EdgeKind.DETECTED_BY, SERVICE, slow)
    if channel is None:
        d.edge(EdgeKind.NOTIFIES, slow, KIM)
    else:
        d.edge(EdgeKind.NOTIFIES, slow, KIM, channel=channel)
    return slow


def _no_approval(d: Demo) -> Demo:
    d.node(NodeKind.ACTION, "결제 롤백", action_type="rollback", duration_minutes=10, requires_approval=False)
    return d


def test_nan_scenario_deadline_falls_back_to_the_service_slo() -> None:
    """A NaN deadline used to make every comparison false → PASS for any mitigation time (P3)."""
    graph = add_lead(base("departed")).build()
    result = replay(graph, scenario(deadline_minutes=float("nan")))
    assert result.deadline_minutes == 60.0  # tier-1 SLO, not NaN
    assert result.outcome is Outcome.PASS  # mitigated at 28 ≤ 60
    assert "시나리오 목표 시간이 숫자가 아님 → 서비스 목표 복구 시간 적용" in result.events[0].note

    d = add_lead(base("departed"))
    d.node(NodeKind.SERVICE, "결제", slo_minutes=20)  # merged into the existing service node
    strict = replay(d.build(), scenario(deadline_minutes=float("nan")))
    assert strict.outcome is Outcome.FAIL
    assert strict.deadline_minutes == 20.0
    assert event(strict, SimEventKind.DEADLINE_MISSED).t == 20.0
    assert strict.result_hash() == replay(d.build(), scenario(deadline_minutes=float("nan"))).result_hash()


def test_empty_policy_slo_table_never_raises_and_never_passes() -> None:
    """``CheckContext.slo_minutes`` cannot handle an empty SLO table; the replay must."""
    no_slo = Policy(tier_slo_minutes={})
    graph = add_lead(base("departed")).build()
    result = replay(graph, scenario(), policy=no_slo)
    assert result.deadline_minutes == 0.0
    assert result.outcome is Outcome.FAIL
    assert result.time_to_mitigate == 28.0
    assert "목표 복구 시간이 서비스·정책 어디에도 없음" in result.events[0].note
    unknown = replay(graph, scenario(service_id="service:없음"), policy=no_slo)
    assert unknown.deadline_minutes == 0.0
    assert unknown.failed_element is Element.DETECTION
    # a deadline stated by the service itself, or by the scenario, still applies
    d = add_lead(base("departed"))
    d.node(NodeKind.SERVICE, "결제", slo_minutes=45)
    assert replay(d.build(), scenario(), policy=no_slo).outcome is Outcome.PASS
    assert replay(graph, scenario(deadline_minutes=30), policy=no_slo).outcome is Outcome.PASS


def test_negative_deadline_is_taken_as_given_but_never_precedes_the_incident() -> None:
    graph = add_lead(base("departed")).build()
    result = replay(graph, scenario(deadline_minutes=-1))
    assert result.outcome is Outcome.FAIL
    assert result.deadline_minutes == -1.0
    assert result.events[0].kind is SimEventKind.INCIDENT_START  # used to be DEADLINE_MISSED at t=-1
    assert min(e.t for e in result.events) == 0.0
    assert event(result, SimEventKind.DEADLINE_MISSED).t == 0.0
    assert "목표 복구 시간 -1분 초과" in result.reason


@pytest.mark.parametrize("infinite", [float("inf"), float("-inf")])
def test_infinite_deadline_states_no_limit_and_falls_back_to_the_slo(infinite: float) -> None:
    """∞ used to be taken literally: every mitigation passed and the result was not storable JSON."""
    d = add_lead(base("departed"))
    d.node(NodeKind.SERVICE, "결제", slo_minutes=20)  # strict SLO; mitigation happens at 28
    graph = d.build()
    result = replay(graph, scenario(deadline_minutes=infinite))
    assert result.deadline_minutes == 20.0
    assert result.outcome is Outcome.FAIL
    assert event(result, SimEventKind.DEADLINE_MISSED).t == 20.0
    assert "무한대(기한 없음)" in result.events[0].note
    json.dumps(result.model_dump(mode="json"), allow_nan=False)  # storable as portable JSON
    assert replay(add_lead(base("departed")).build(), scenario(deadline_minutes=infinite)).outcome is (
        Outcome.PASS
    )  # tier-1 SLO (60) applies instead


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (  # open end vs explicit infinity used to compare equal → input order leaked into notes
            Perturbation(kind=PerturbationKind.PERSON_UNAVAILABLE, target_id=KIM),
            Perturbation(kind=PerturbationKind.PERSON_UNAVAILABLE, target_id=KIM, end_minute=float("inf")),
        ),
        (  # NaN start compares false both ways
            Perturbation(kind=PerturbationKind.PERSON_UNAVAILABLE, target_id=KIM, start_minute=float("nan")),
            Perturbation(kind=PerturbationKind.PERSON_UNAVAILABLE, target_id=KIM, start_minute=5),
        ),
        (
            Perturbation(kind=PerturbationKind.CHANNEL_DOWN, target_id=SLACK, end_minute=float("nan")),
            Perturbation(kind=PerturbationKind.CHANNEL_DOWN, target_id=SLACK, end_minute=3),
        ),
    ],
)
def test_perturbation_order_is_irrelevant_even_for_odd_bounds(
    first: Perturbation, second: Perturbation
) -> None:
    graph = add_lead(base()).build()
    a = replay(graph, scenario(first, second))
    b = replay(graph, scenario(second, first))
    assert a.model_dump(exclude={"scenario_fingerprint"}) == b.model_dump(exclude={"scenario_fingerprint"})


def test_second_alarm_repages_someone_back_from_a_short_absence() -> None:
    """Two alarms paging the same on-call is the norm; the later one must re-evaluate them.

    Before the fix the 10-minute page was dropped as 'already visited' → false FAIL at OWNER.
    """
    d = _no_approval(base())
    slow = _second_alarm(d)
    graph = d.build()
    absent = Perturbation(kind=PerturbationKind.PERSON_UNAVAILABLE, target_id=KIM, end_minute=5)
    result = replay(graph, scenario(absent))
    assert result.outcome is Outcome.PASS, result.reason
    assert [(e.t, e.kind) for e in result.events if e.target_id == KIM] == [
        (3.0, SimEventKind.NO_ANSWER),
        (10.0, SimEventKind.PAGE),
        (15.0, SimEventKind.ACK),
    ]
    ack = event(result, SimEventKind.ACK)
    assert (ack.t, ack.actor_id, ack.edge_id) == (15.0, KIM, edge_id(EdgeKind.NOTIFIES, slow, KIM))
    assert result.time_to_mitigate == 25.0
    assert SimEventKind.DEAD_END not in kinds(result)
    # the second alarm is highlighted end to end (service → detected_by → trigger → notifies)
    path = result.path_ids
    assert path.index(edge_id(EdgeKind.DETECTED_BY, SERVICE, slow)) < path.index(slow)
    assert path.index(slow) < path.index(edge_id(EdgeKind.NOTIFIES, slow, KIM))


def test_identical_pages_in_the_same_minute_are_not_repeated() -> None:
    d = base("departed", notify_channel=False)
    _second_alarm(d, delay=3)
    result = replay(d.build(), scenario())
    no_answers = [e for e in result.events if e.kind is SimEventKind.NO_ANSWER and e.target_id == KIM]
    assert len(no_answers) == 1
    assert result.failed_element is Element.OWNER


def test_same_minute_page_on_another_channel_is_tried() -> None:
    """Alarm A pages over 슬랙 (down), alarm B over 전화 in the same minute → B gets through."""
    d = _no_approval(base())
    d.node(NodeKind.CHANNEL, "전화")
    _second_alarm(d, delay=3, channel=PHONE)
    result = replay(d.build(), scenario(Perturbation(kind=PerturbationKind.CHANNEL_DOWN, target_id=SLACK)))
    assert result.outcome is Outcome.PASS, result.reason
    page = event(result, SimEventKind.PAGE)
    assert (page.t, page.actor_id, page.target_id) == (3.0, PHONE, KIM)


def test_unpaged_later_alarm_is_not_highlighted() -> None:
    """An alarm that fires after the owner already answered was never traversed."""
    d = _no_approval(base())
    slow = _second_alarm(d, delay=30)
    result = replay(d.build(), scenario())
    assert result.outcome is Outcome.PASS
    assert slow not in result.path_ids
    assert edge_id(EdgeKind.DETECTED_BY, SERVICE, slow) not in result.path_ids


@pytest.mark.parametrize("target", ["service:없음", KIM, SLACK])
def test_target_that_is_not_a_confirmed_service_is_reported_as_such(target: str) -> None:
    graph = add_lead(base()).build()
    config = SimConfig(undetected_discovery_minutes=1)  # no assumption can rescue an unknown target
    result = replay(graph, scenario(service_id=target), config=config)
    assert result.outcome is Outcome.FAIL
    assert result.failed_element is Element.DETECTION
    assert result.failed_at_id == target
    assert kinds(result) == [SimEventKind.INCIDENT_START, SimEventKind.UNDETECTED]
    assert "확인된 서비스·시스템이 아님" in result.reason
    assert "감지 신호(알람·모니터)가 없어" not in result.reason  # P2: do not claim what we do not know
    assert result.path_ids == ()


def test_held_service_cannot_pass() -> None:
    d = add_lead(base())
    d.b.nodes[SERVICE] = d.b.nodes[SERVICE].model_copy(update={"verification": Verification.HOLD})
    result = replay(d.build(), scenario())
    assert result.outcome is Outcome.FAIL
    assert result.failed_element is Element.DETECTION
    assert "판단 보류" in result.events[0].note
    assert result.deadline_minutes == 240.0  # tier unknown in the trusted view → default tier


def test_system_node_can_be_rehearsed() -> None:
    d = _no_approval(base())
    db = d.node(NodeKind.SYSTEM, "결제 DB")
    d.edge(EdgeKind.DETECTED_BY, db, TRIGGER)
    d.edge(EdgeKind.MITIGATED_BY, db, ROLLBACK)
    result = replay(d.build(), scenario(service_id=db))
    assert result.outcome is Outcome.PASS, result.reason


def test_untyped_obligation_with_reporter_still_cannot_satisfy_the_law() -> None:
    d = add_lead(base())
    d.node(NodeKind.OBLIGATION, "KISA 침해사고 신고", deadline_minutes=60)  # incident_type missing
    d.edge(EdgeKind.MUST_REPORT, SERVICE, OBLIGATION)
    d.node(NodeKind.PERSON, "최보안", status="active")
    d.edge(EdgeKind.REACHABLE_VIA, SECURITY, PHONE)
    d.edge(EdgeKind.REPORTED_BY, OBLIGATION, SECURITY)
    result = replay(d.build(), scenario(incident_type=IncidentType.BREACH))
    assert result.outcome is Outcome.FAIL
    assert "사고 유형이 미기재·인식 불가" in result.reason
    assert SimEventKind.REPORT_FILED not in kinds(result)


def test_first_responders_follow_paging_time_not_trigger_id() -> None:
    d = add_lead(base())
    d.node(NodeKind.TRIGGER, "가 느린 알람", delay_minutes=30)
    slow = node_id(NodeKind.TRIGGER, "가 느린 알람")
    assert slow < TRIGGER  # id order alone would put 박팀장 (slow alarm) first
    d.edge(EdgeKind.DETECTED_BY, SERVICE, slow)
    d.edge(EdgeKind.NOTIFIES, slow, PARK)
    graph = d.build()
    ctx = CheckContext(graph)
    assert [p.id for p in first_responders(ctx, SERVICE)] == [KIM, PARK]
    assert event(replay(graph, scenario()), SimEventKind.ACK).actor_id == KIM


def test_first_responders_order_uses_the_configured_default_delay() -> None:
    d = add_lead(base())
    d.node(NodeKind.TRIGGER, "고객 문의")  # no delay attribute → SimConfig.detection_delay_minutes
    inquiry = node_id(NodeKind.TRIGGER, "고객 문의")
    d.edge(EdgeKind.DETECTED_BY, SERVICE, inquiry)
    d.edge(EdgeKind.NOTIFIES, inquiry, PARK)
    graph = d.build()
    ctx = CheckContext(graph)
    early, late = SimConfig(detection_delay_minutes=1), SimConfig(detection_delay_minutes=10)
    assert [p.id for p in first_responders(ctx, SERVICE, config=early)] == [PARK, KIM]
    assert [p.id for p in first_responders(ctx, SERVICE, config=late)] == [KIM, PARK]
    for config, who in ((early, PARK), (late, KIM)):
        assert event(replay(graph, scenario(), config=config), SimEventKind.ACK).actor_id == who


def test_first_contacts_skip_people_who_cannot_be_paged() -> None:
    d = add_lead(base())
    d.node(NodeKind.PERSON, "가무연락", status="active")  # sorts before 김대리, has no channel at all
    nobody = node_id(NodeKind.PERSON, "가무연락")
    d.edge(EdgeKind.NOTIFIES, TRIGGER, nobody)
    graph = d.build()
    ctx = CheckContext(graph)
    contacts = first_contacts(ctx, SERVICE)
    assert [(c.person.id, c.via_id, c.channel_ids) for c in contacts] == [(KIM, KIM, (SLACK,))]
    assert event(replay(graph, scenario()), SimEventKind.ACK).actor_id == KIM
    assert paging_channels(graph, nobody, nobody) == ()
    assert paging_channels(graph, PARK, LEAD) == (PHONE,)


def test_team_channel_makes_members_reachable_first_contacts() -> None:
    d = Demo()
    d.node(NodeKind.SERVICE, "결제", tier=1)
    team = d.node(NodeKind.TEAM, "결제팀")
    d.node(NodeKind.PERSON, "김대리", status="active")
    d.node(NodeKind.CHANNEL, "전화")
    d.edge(EdgeKind.MEMBER_OF, KIM, team)
    d.edge(EdgeKind.OWNED_BY, SERVICE, team)
    d.edge(EdgeKind.REACHABLE_VIA, team, PHONE)
    contacts = first_contacts(CheckContext(d.build()), SERVICE)
    assert [(c.person.id, c.via_id, c.channel_ids) for c in contacts] == [(KIM, team, (PHONE,))]


def test_concurrent_incident_never_pulls_an_unreachable_person() -> None:
    d = add_lead(base())
    d.node(NodeKind.SERVICE, "주문", tier=1)
    d.node(NodeKind.PERSON, "가무연락", status="active")
    nobody = node_id(NodeKind.PERSON, "가무연락")
    d.edge(EdgeKind.OWNED_BY, ORDER, nobody)  # listed first, but could never be called to 주문
    d.edge(EdgeKind.OWNED_BY, ORDER, KIM)
    result = replay(
        d.build(), scenario(Perturbation(kind=PerturbationKind.CONCURRENT_INCIDENT, target_id=ORDER))
    )
    assert "주문 동시 장애 — 김대리 투입" in result.events[0].note
    assert event(result, SimEventKind.PAGE).target_id == PARK


def test_trusted_view_drops_held_elements_and_counts_them() -> None:
    d = add_lead(base())
    d.b.nodes[PARK] = d.b.nodes[PARK].model_copy(update={"verification": Verification.HOLD})
    esc = edge_id(EdgeKind.ESCALATES_TO, KIM, LEAD)
    d.b.edges[esc] = d.b.edges[esc].model_copy(update={"verification": Verification.HOLD})
    graph = d.build()
    view, held = trusted_view(graph)
    assert held == 2
    assert PARK not in view.nodes
    assert esc not in view.edges
    assert edge_id(EdgeKind.HOLDS_ROLE, PARK, LEAD) not in view.edges  # touches a held node
    assert trusted_view(view) == (view, 0)


def test_empty_inputs() -> None:
    empty = OpsGraph()
    result = replay(empty, scenario())
    assert result.outcome is Outcome.FAIL
    assert result.failed_element is Element.DETECTION
    assert replay_all(empty, []) == []
    assert first_contacts(CheckContext(empty), SERVICE) == []
    no_slo = Policy(tier_slo_minutes={})
    assert replay(empty, scenario(), policy=no_slo).deadline_minutes == 0.0


@pytest.mark.parametrize(
    ("word", "particle", "expected"),
    [
        ("", _SUBJECT, "이(가)"),
        ("메일", _SUBJECT, "메일이"),  # ㄹ final takes 이 for the subject…
        ("서울", _TOWARD, "서울로"),  # …but 로 for direction
        ("IDC-2", _TOWARD, "IDC-2(으)로"),
        ("카카오톡", _TOWARD, "카카오톡으로"),
        ("팀장 (대행)", _SUBJECT, "팀장 (대행)이(가)"),
    ],
)
def test_korean_particles_edge_cases(word: str, particle: Any, expected: str) -> None:
    assert _attach(word, particle) == expected


_odd_bounds = st.one_of(
    st.none(),
    st.just(float("nan")),
    st.just(float("inf")),
    st.just(-0.0),
    st.floats(0, 30, allow_nan=False),
)


@settings(max_examples=60, deadline=None)
@given(
    graph=graphs(),
    raw=st.lists(
        st.tuples(
            st.sampled_from(list(PerturbationKind)),
            st.sampled_from(_ALL_IDS),
            st.one_of(st.just(float("nan")), st.floats(0, 30, allow_nan=False)),
            _odd_bounds,
        ),
        max_size=4,
    ),
    data=st.data(),
)
def test_perturbation_order_never_matters_with_odd_bounds(
    graph: OpsGraph, raw: list[tuple[PerturbationKind, str, float, float | None]], data: st.DataObject
) -> None:
    perturbations = [Perturbation(kind=k, target_id=t, start_minute=s, end_minute=e) for k, t, s, e in raw]
    shuffled = data.draw(st.permutations(perturbations))
    service = data.draw(st.sampled_from(_IDS[NodeKind.SERVICE]))
    a = replay(graph, scenario(*perturbations, service_id=service))
    b = replay(graph, scenario(*shuffled, service_id=service))
    assert a.model_dump(exclude={"scenario_fingerprint"}) == b.model_dump(exclude={"scenario_fingerprint"})


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(graph=graphs(), service=st.sampled_from(_IDS[NodeKind.SERVICE]), config=configs)
def test_first_responder_is_who_answers_the_first_page_at_baseline(
    graph: OpsGraph, service: str, config: SimConfig
) -> None:
    """Templates aim at ``first_responders``; this pins it to what the replay really does."""
    result = replay(graph, scenario(service_id=service), config=config)
    seed_edges = {eid for eid, e in graph.edges.items() if e.kind in (EdgeKind.NOTIFIES, EdgeKind.OWNED_BY)}
    seed_acks = [e for e in result.events if e.kind is SimEventKind.ACK and e.edge_id in seed_edges]
    assert len(seed_acks) <= 1  # the first owner ACK wins; nobody else is paged via a seed edge
    responders = first_responders(CheckContext(trusted_view(graph)[0]), service, config=config)
    if seed_acks:
        assert responders
        assert seed_acks[0].actor_id == responders[0].id
    for nid in result.path_ids:
        edge = graph.edges.get(nid)
        if edge is not None and edge.kind is EdgeKind.NOTIFIES:
            # every alarm that paged is highlighted from the service on
            assert edge_id(EdgeKind.DETECTED_BY, service, edge.src) in result.path_ids


# --------------------------------------------------------------------------- reviewer 2: adversarial


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ack_minutes", float("nan")),
        ("ack_minutes", float("inf")),
        ("ack_minutes", -100.0),
        ("approval_minutes", float("nan")),
        ("default_action_minutes", -1.0),
        ("detection_delay_minutes", float("nan")),
        ("undetected_discovery_minutes", float("nan")),
        ("default_escalation_after_minutes", -5.0),
        ("default_escalation_after_minutes", float("inf")),
        ("max_hops", -1),
    ],
)
def test_broken_sim_config_is_refused_not_judged(field: str, value: float) -> None:
    """A NaN ack/approval time used to PASS every scenario with time_to_mitigate=NaN, and a
    negative ack put the ACK 92 minutes *before* the incident started."""
    config = SimConfig.model_validate({field: value})
    graph = add_lead(base("departed")).build()
    with pytest.raises(ValueError, match=field):
        replay(graph, scenario(), config=config)
    with pytest.raises(ValueError, match="SimConfig"):
        replay_all(graph, [], config=config)


def test_zero_timings_are_valid_assumptions() -> None:
    config = SimConfig(
        detection_delay_minutes=0,
        undetected_discovery_minutes=0,
        ack_minutes=0,
        default_escalation_after_minutes=0,
        approval_minutes=0,
        default_action_minutes=0,
        max_hops=0,
    )
    result = replay(add_lead(base("active"), escalate=False).build(), scenario(), config=config)
    assert result.outcome is Outcome.PASS, result.reason
    # alarm at 3 (graph attr) → 김대리 acks at 3 → 팀장 paged and acks at 3 → approved 3 → rollback 10
    assert result.time_to_mitigate == 13.0


def test_negative_slo_and_negative_legal_deadline_never_precede_the_incident() -> None:
    d = add_lead(base("departed"))
    d.node(NodeKind.SERVICE, "결제", slo_minutes=-5)
    result = replay(d.build(), scenario())
    assert result.outcome is Outcome.FAIL
    assert event(result, SimEventKind.DEADLINE_MISSED).t == 0.0
    assert result.events[0].kind is SimEventKind.INCIDENT_START

    rule = ObligationRule(
        key="broken",
        incident_type=IncidentType.BREACH,
        recipient="KISA",
        deadline_minutes=-60,
        legal_basis="시험용 규정",
    )
    policy = Policy(obligations=(rule,))
    breach = replay(breach_graph(), scenario(incident_type=IncidentType.BREACH), policy=policy)
    assert breach.outcome is Outcome.FAIL
    missed = event(breach, SimEventKind.REPORT_MISSED)
    assert missed.t == 3.0  # due at detection, not at 3 - 60
    assert "시험용 규정" in missed.note
    assert min(e.t for e in breach.events) == 0.0


def test_negative_infinite_window_keeps_its_sign_in_notes() -> None:
    graph = add_lead(base()).build()
    absent = Perturbation(kind=PerturbationKind.PERSON_UNAVAILABLE, target_id=KIM, start_minute=float("-inf"))
    result = replay(graph, scenario(absent))
    assert "김대리 부재(-∞분부터)" in result.events[0].note
    assert event(result, SimEventKind.NO_ANSWER).target_id == KIM  # the window does cover t=3


@pytest.mark.parametrize(
    ("attrs", "approver", "expected"),
    [
        ({"action_type": "restart", "requires_approval": True}, False, True),
        ({"action_type": "rollback", "requires_approval": False}, True, False),
        ({"action_type": "rollback"}, False, True),
        ({"action_type": "failover"}, False, True),
        ({"action_type": "feature_flag"}, False, False),
        ({"action_type": "restart"}, True, False),  # a named approver alone gates nothing
        ({"action_type": "warp-speed"}, False, False),
        ({"requires_approval": "아니오"}, False, False),  # not a bool → treated as absent
    ],
)
def test_requires_approval_follows_the_single_published_rule(
    attrs: dict[str, Any], approver: bool, expected: bool
) -> None:
    d = add_lead(base(), escalate=False, approver=approver)
    d.b.nodes.pop(ROLLBACK)
    d.node(NodeKind.ACTION, "결제 롤백", **attrs)
    graph = d.build()
    assert requires_approval(CheckContext(graph), graph.nodes[ROLLBACK]) is expected
    timeline = kinds(replay(graph, scenario()))
    asked = SimEventKind.APPROVAL_REQUESTED in timeline or SimEventKind.APPROVAL_BLOCKED in timeline
    assert asked is expected


def test_roster_alias_wins_over_the_graph_status() -> None:
    graph = add_lead(base("active"), escalate=False).build()
    roster = (RosterEntry(name="김민수", aliases=("김대리",), status=PersonStatus.ON_LEAVE),)
    result = replay(graph, scenario(), roster=roster)
    assert result.failed_element is Element.OWNER
    assert "장기 부재" in event(result, SimEventKind.NO_ANSWER).note
    without_alias = (RosterEntry(name="김민수", status=PersonStatus.ON_LEAVE),)
    assert config_hash(SimConfig(), Policy(), roster) != config_hash(SimConfig(), Policy(), without_alias)
    assert replay(graph, scenario(), roster=without_alias).outcome is Outcome.PASS


def test_replay_all_consumes_generators_once_and_runs_are_independent() -> None:
    graph = add_lead(base("active"), escalate=False).build()
    down = scenario(Perturbation(kind=PerturbationKind.CHANNEL_DOWN, target_id=SLACK), id="down")
    roster = (RosterEntry(name="박팀장", status=PersonStatus.DEPARTED),)
    batch = replay_all(graph, (s for s in [down, scenario(), down]), roster=(r for r in roster))
    assert [r.scenario_id for r in batch] == ["down", "s1", "down"]
    assert batch[0] == batch[2]  # no state leaks from one run into the next
    assert batch[1] == replay(graph, scenario(), roster=roster)
    assert batch[1].failed_element is Element.APPROVER  # the roster reached every run


def test_odd_korean_names_get_natural_fallback_particles() -> None:
    d = Demo()
    d.node(NodeKind.SERVICE, "결제", tier=1)
    d.node(NodeKind.TRIGGER, "결제 오류율 알람", delay_minutes=3)
    who = d.node(NodeKind.PERSON, "박 팀장(대행)", status="active")
    bell = d.node(NodeKind.CHANNEL, "카카오톡 🔔")
    action = d.node(
        NodeKind.ACTION, "롤백 v2", action_type="rollback", requires_approval=False, duration_minutes=1
    )
    d.edge(EdgeKind.DETECTED_BY, SERVICE, TRIGGER)
    d.edge(EdgeKind.NOTIFIES, TRIGGER, who)
    d.edge(EdgeKind.REACHABLE_VIA, who, bell)
    d.edge(EdgeKind.MITIGATED_BY, SERVICE, action)
    result = replay(d.build(), scenario())
    assert result.outcome is Outcome.PASS, result.reason
    assert event(result, SimEventKind.PAGE).note == "담당 호출: 박 팀장(대행)에게 카카오톡 🔔(으)로 호출"
    assert event(result, SimEventKind.ACK).note == "담당 호출: 박 팀장(대행)이(가) 응답"
    assert event(result, SimEventKind.MITIGATED).note.startswith("롤백 v2 완료")


@settings(max_examples=200)
@given(
    word=st.text(
        alphabet=st.one_of(st.characters(min_codepoint=0xAC00, max_codepoint=0xD7A3), st.characters()),
        max_size=6,
    )
)
def test_particles_follow_the_final_sound(word: str) -> None:
    """Independent oracle: an NFD-decomposed syllable has 3 jamo iff it has a final consonant."""
    last = word[-1:]
    for particle in (_SUBJECT, _TOWARD):
        out = _attach(word, particle)
        assert out.startswith(word)
        suffix = out[len(word) :]
        if last and "가" <= last <= "힣":
            jamo = unicodedata.normalize("NFD", last)
            vowel_form = len(jamo) == 2 or (jamo[-1] == "\u11af" and particle is _TOWARD)
            assert suffix == (particle.after_vowel if vowel_form else particle.after_consonant)
        else:
            assert suffix == particle.unknown


_odd_deadlines = st.one_of(
    st.none(),
    st.just(float("nan")),
    st.just(float("inf")),
    st.just(float("-inf")),
    st.floats(-100, 200, allow_nan=False),
)
_NEGATIVE_RULE = ObligationRule(
    key="neg", incident_type=IncidentType.BREACH, recipient="KISA", deadline_minutes=-30, legal_basis="시험"
)
_policies = st.sampled_from(
    [
        Policy(),
        Policy(tier_slo_minutes={}),
        Policy(regulated_services=("service:s1",)),
        Policy(obligations=(_NEGATIVE_RULE,)),
    ]
)


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(graph=graphs(), sc=scenarios(), deadline=_odd_deadlines, policy=_policies, config=configs)
def test_every_result_is_storable_and_its_timeline_starts_at_the_incident(
    graph: OpsGraph, sc: Scenario, deadline: float | None, policy: Policy, config: SimConfig
) -> None:
    odd = sc.model_copy(update={"deadline_minutes": deadline})
    result = replay(graph, odd, policy=policy, config=config)
    json.dumps(result.model_dump(mode="json"), allow_nan=False)  # the store refuses NaN/∞
    assert result.events[0].kind is SimEventKind.INCIDENT_START
    assert result.events[0].t == 0.0
    assert all(math.isfinite(e.t) and e.t >= 0 for e in result.events)
    assert all(e.note for e in result.events)  # every timeline entry explains itself
    assert math.isfinite(result.deadline_minutes)
    assert result.time_to_mitigate is None or math.isfinite(result.time_to_mitigate)
    if result.outcome is Outcome.PASS:
        assert result.time_to_mitigate is not None
        assert result.time_to_mitigate <= result.deadline_minutes
        assert event(result, SimEventKind.MITIGATED).t == result.time_to_mitigate
        assert not {SimEventKind.DEADLINE_MISSED, SimEventKind.REPORT_MISSED} & set(kinds(result))
    else:
        assert result.failed_at_id in graph.nodes or result.failed_at_id == odd.service_id


def test_configured_defaults_are_marked_as_assumptions_in_the_timeline() -> None:
    """P2: a delay or waiting time the runbook never stated must not read like a runbook fact."""
    d = add_lead(base("departed"), escalate=False)
    d.edge(EdgeKind.ESCALATES_TO, KIM, LEAD)  # no after_minutes
    inquiry = d.node(NodeKind.TRIGGER, "고객 문의")  # no delay_minutes
    d.edge(EdgeKind.DETECTED_BY, SERVICE, inquiry)
    config = SimConfig(detection_delay_minutes=1, default_escalation_after_minutes=10)
    result = replay(d.build(), scenario(), config=config)
    detected = event(result, SimEventKind.DETECTED)
    assert detected.actor_id == inquiry
    assert detected.note == "고객 문의로 감지(장애 후 1분 — 감지 지연 미기재로 기본값 가정)"
    escalate = event(result, SimEventKind.ESCALATE)
    assert escalate.t == 13.0
    assert (
        escalate.note == "담당 호출: 김대리 미응답 10분 → 팀장으로 상향(상향 대기시간 미기재로 기본값 가정)"
    )
    stated = replay(add_lead(base("departed")).build(), scenario(), config=config)
    assert event(stated, SimEventKind.DETECTED).note == "결제 오류율 알람으로 감지(장애 후 3분)"
    assert "가정" not in event(stated, SimEventKind.ESCALATE).note


# --------------------------------------------------------------------------- org-wide duties


def _org_breach_graph(*, link_from: str | None = None, held: bool = False) -> OpsGraph:
    """결제 + a KISA step that names no service (org-wide) unless ``link_from`` links it."""
    d = add_lead(base())
    d.node(NodeKind.SERVICE, "주문", tier=1)
    d.node(
        NodeKind.OBLIGATION,
        "KISA 침해사고 신고",
        incident_type="breach",
        recipient="KISA",
        deadline_minutes=1440,
    )
    d.node(NodeKind.PERSON, "최보안", status="active")
    d.edge(EdgeKind.REACHABLE_VIA, SECURITY, PHONE)
    d.edge(EdgeKind.REPORTED_BY, OBLIGATION, SECURITY)
    if link_from is not None:
        link = d.edge(EdgeKind.MUST_REPORT, link_from, OBLIGATION)
        if held:
            d.b.edges[link] = d.b.edges[link].model_copy(
                update={"verification": Verification.HOLD, "hold_reason": "역번역 대조 불일치"}
            )
    return d.build()


def test_org_wide_obligation_is_filed_for_any_service() -> None:
    result = replay(_org_breach_graph(), scenario(incident_type=IncidentType.BREACH))
    assert result.outcome is Outcome.PASS, result.reason
    filed = event(result, SimEventKind.REPORT_FILED)
    assert (filed.actor_id, filed.target_id) == (SECURITY, OBLIGATION)
    assert OBLIGATION in result.path_ids
    assert not any(pid.startswith("must_report:") for pid in result.path_ids)


def test_obligation_of_another_service_does_not_bind_this_one() -> None:
    result = replay(_org_breach_graph(link_from=ORDER), scenario(incident_type=IncidentType.BREACH))
    assert result.outcome is Outcome.FAIL
    assert "(KISA) 신고 절차가 런북에 없음" in result.reason


@pytest.mark.parametrize("link_from", [SERVICE, ORDER])
def test_obligation_tied_only_by_a_held_link_is_never_org_wide(link_from: str) -> None:
    """Dropping the held link must not turn an unconfirmed step into everyone's duty (P3)."""
    graph = _org_breach_graph(link_from=link_from, held=True)
    view, excluded = trusted_view(graph)
    assert OBLIGATION not in view.nodes
    assert excluded == 2  # the held link and the obligation it alone tied to a service
    result = replay(graph, scenario(incident_type=IncidentType.BREACH))
    assert result.outcome is Outcome.FAIL
    assert SimEventKind.REPORT_FILED not in kinds(result)
