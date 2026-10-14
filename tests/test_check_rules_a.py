"""Tests for the core fixed rules: JT-PATH-*, JT-PPL-*, JT-ESC-*, JT-CHN-*."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from jettae.check import elements, registry
from jettae.check.context import CheckContext
from jettae.check.elements import MAX_ESCALATION_HOPS, analyze_owner, evaluate_service
from jettae.check.registry import Rule, sort_findings
from jettae.check.rules import channels, escalation, path, people
from jettae.check.rules.escalation import canonical_cycle, escalation_cycles
from jettae.check.scoring import score
from jettae.domain import (
    ActionType,
    Attr,
    ChannelMedium,
    EdgeKind,
    Element,
    ElementStatus,
    Extractor,
    Finding,
    FindingStatus,
    FixKind,
    GraphBuilder,
    Node,
    NodeKind,
    OpsGraph,
    PersonStatus,
    Provenance,
    Severity,
    SourceKind,
    SourceSpan,
    Verification,
    edge_id,
    node_id,
)
from jettae.domain.graph import AttrValue, Edge

# --------------------------------------------------------------------------- graph DSL

DOC = "doc-runbook"
ROSTER = Provenance(source=SourceKind.ROSTER, extractor=Extractor.IMPORT, ref="roster.csv")
CATALOG = Provenance(source=SourceKind.CATALOG, extractor=Extractor.IMPORT, ref="catalog.yaml")


def rb(quote: str, start: int = 0) -> Provenance:
    span = SourceSpan(doc_id=DOC, start=start, end=start + len(quote), quote=quote, line=1)
    return Provenance(source=SourceKind.RUNBOOK, extractor=Extractor.RULE, spans=(span,))


class Net:
    """GraphBuilder DSL: roster facts carry ROSTER provenance, runbook facts an exact span."""

    def __init__(self, base: OpsGraph | None = None) -> None:
        self.b = GraphBuilder(base)
        self._offset = 0

    def _prov(self, quote: str) -> Provenance:
        self._offset += len(quote) + 1
        return rb(quote, self._offset)

    def node(
        self,
        kind: NodeKind,
        name: str,
        prov: Provenance | None = None,
        *,
        hold: bool = False,
        **attrs: AttrValue | None,
    ) -> str:
        p = prov or self._prov(name)
        nid = node_id(kind, name)
        values = {k: Attr(value=v, source=p.source) for k, v in attrs.items() if v is not None}
        verification = Verification.HOLD if hold else Verification.VERIFIED
        self.b.add_node(
            Node(id=nid, kind=kind, name=name, attrs=values, prov=(p,), verification=verification)
        )
        return nid

    def edge(
        self,
        kind: EdgeKind,
        src: str,
        dst: str,
        *,
        hold: bool = False,
        prov: Provenance | None = None,
        **attrs: AttrValue | None,
    ) -> str:
        p = prov or self._prov(f"{src} {kind.value} {dst}")
        eid = edge_id(kind, src, dst)
        values = {k: Attr(value=v, source=p.source) for k, v in attrs.items() if v is not None}
        verification = Verification.HOLD if hold else Verification.VERIFIED
        self.b.add_edge(
            Edge(id=eid, kind=kind, src=src, dst=dst, attrs=values, prov=(p,), verification=verification)
        )
        return eid

    def service(self, name: str, tier: int = 1, *, location: str | None = None) -> str:
        sid = self.node(NodeKind.SERVICE, name, tier=tier)
        if location:
            self.edge(EdgeKind.RUNS_IN, sid, location, prov=CATALOG)
        return sid

    def person(self, name: str, status: PersonStatus = PersonStatus.ACTIVE, title: str | None = None) -> str:
        return self.node(NodeKind.PERSON, name, ROSTER, status=status.value, title=title)

    def stranger(self, name: str) -> str:
        return self.node(NodeKind.PERSON, name)

    def role(self, name: str, *holders: str) -> str:
        rid = self.node(NodeKind.ROLE, name, ROSTER)
        for h in holders:
            self.edge(EdgeKind.HOLDS_ROLE, h, rid, prov=ROSTER)
        return rid

    def team(self, name: str, *members: str) -> str:
        tid = self.node(NodeKind.TEAM, name, ROSTER)
        for m in members:
            self.edge(EdgeKind.MEMBER_OF, m, tid, prov=ROSTER)
        return tid

    def channel(self, name: str, medium: ChannelMedium, *, located: str | None = None) -> str:
        cid = self.node(NodeKind.CHANNEL, name, ROSTER, medium=medium.value)
        if located:
            self.edge(EdgeKind.RUNS_IN, cid, located, prov=CATALOG)
        return cid

    def reach(self, who: str, *chs: str) -> None:
        for c in chs:
            self.edge(EdgeKind.REACHABLE_VIA, who, c, prov=ROSTER)

    def trigger(self, sid: str, name: str, *, hold: bool = False) -> str:
        tid = self.node(NodeKind.TRIGGER, name, hold=hold)
        self.edge(EdgeKind.DETECTED_BY, sid, tid)
        return tid

    def action(self, sid: str, name: str, kind: ActionType | None = ActionType.ROLLBACK) -> str:
        aid = self.node(NodeKind.ACTION, name, action_type=kind.value if kind else None)
        self.edge(EdgeKind.MITIGATED_BY, sid, aid)
        return aid

    def build(self) -> OpsGraph:
        return self.b.build()

    def ctx(self) -> CheckContext:
        return CheckContext(graph=self.build())


# --------------------------------------------------------------------------- running my rules

RULES: tuple[type, ...] = (
    path.DetectionMissing,
    path.DetectionNotifiesNobody,
    path.OwnerMissing,
    path.OwnerUnreachable,
    path.ApproverMissing,
    path.FallbackChannelMissing,
    path.RollbackMissing,
    path.RollbackExecutorMissing,
    people.DepartedResponder,
    people.UnconfirmedPerson,
    people.OnLeaveResponder,
    escalation.EscalationMissing,
    escalation.EscalationCycle,
    escalation.EscalationDeadEnd,
    escalation.EscalationWaitMissing,
    channels.ChannelSharesFailureDomain,
    channels.FallbackChannelUnused,
)
INSTANCES: dict[str, Rule] = {cls().info.rule_id: cls() for cls in RULES}


def raw(ctx: CheckContext, *rule_ids: str) -> list[Finding]:
    selected = rule_ids or tuple(sorted(INSTANCES))
    return [f for rid in selected for f in INSTANCES[rid].check(ctx)]


def run(ctx: CheckContext, *rule_ids: str) -> list[Finding]:
    return sort_findings(raw(ctx, *rule_ids))


def ids(findings: list[Finding]) -> list[str]:
    return [f.rule_id for f in findings]


# --------------------------------------------------------------------------- catalog


EXPECTED_CATALOG = {
    "JT-PATH-001": ("감지 경로 없음", Severity.CRITICAL, Element.DETECTION),
    "JT-PATH-002": ("감지 후 호출 대상 없음", Severity.HIGH, Element.DETECTION),
    "JT-PATH-010": ("담당자 없음", Severity.CRITICAL, Element.OWNER),
    "JT-PATH-011": ("담당자 연락 수단 없음", Severity.HIGH, Element.OWNER),
    "JT-PATH-020": ("승인자 없음", Severity.HIGH, Element.APPROVER),
    "JT-PATH-030": ("대체 채널 없음", Severity.HIGH, Element.FALLBACK_CHANNEL),
    "JT-PATH-040": ("롤백/완화 조치 없음", Severity.HIGH, Element.ROLLBACK),
    "JT-PATH-041": ("롤백 실행자 미지정", Severity.MEDIUM, Element.ROLLBACK),
    "JT-PPL-001": ("퇴사자에게 대응이 걸려 있음", Severity.CRITICAL, None),
    "JT-PPL-002": ("명부에서 확인되지 않는 인물", Severity.MEDIUM, None),
    "JT-PPL-003": ("장기 부재자에게 대응이 걸려 있음", Severity.HIGH, None),
    "JT-ESC-001": ("상향 경로 없음", Severity.HIGH, Element.OWNER),
    "JT-ESC-002": ("상향 순환", Severity.HIGH, Element.OWNER),
    "JT-ESC-003": ("상향 막다른 길", Severity.HIGH, Element.OWNER),
    "JT-ESC-004": ("상향 대기시간 미기재", Severity.LOW, Element.OWNER),
    "JT-CHN-001": ("연락 채널이 서비스와 함께 죽음", Severity.HIGH, Element.FALLBACK_CHANNEL),
    "JT-CHN-002": ("대체 채널 연결 누락", Severity.LOW, Element.FALLBACK_CHANNEL),
}


def test_rule_catalog_metadata() -> None:
    assert set(INSTANCES) == set(EXPECTED_CATALOG)
    for rid, (title, severity, element) in EXPECTED_CATALOG.items():
        info = INSTANCES[rid].info
        assert (info.title, info.severity, info.element) == (title, severity, element)
        assert len(info.rationale) > 30


def test_rules_are_registered() -> None:
    for rid, rule in INSTANCES.items():
        assert isinstance(registry.get_rule(rid), type(rule))


# --------------------------------------------------------------------------- demo scenario


def demo() -> tuple[Net, dict[str, str]]:
    net = Net()
    sid = net.service("결제")
    kim = net.person("김대리", PersonStatus.DEPARTED, title="대리")
    lead = net.person("박팀장", title="팀장")
    net.team("결제팀", kim, lead)
    tid = net.trigger(sid, "결제 실패율 5% 초과 알람")
    net.edge(EdgeKind.NOTIFIES, tid, kim)
    net.edge(EdgeKind.OWNED_BY, sid, kim)
    aid = net.action(sid, "이전 버전으로 롤백")
    return net, {"service": sid, "kim": kim, "lead": lead, "action": aid, "trigger": tid}


def test_competition_demo_findings_and_low_readiness() -> None:
    net, n = demo()
    ctx = net.ctx()
    findings = run(ctx)
    rule_ids = set(ids(findings))
    assert {"JT-PPL-001", "JT-ESC-001", "JT-PATH-020", "JT-PATH-030"} <= rule_ids
    assert {"JT-PATH-002", "JT-PATH-010", "JT-PATH-041"} <= rule_ids
    assert "JT-PATH-001" not in rule_ids
    assert "JT-PATH-040" not in rule_ids
    assert findings[0].severity is Severity.CRITICAL

    esc = next(f for f in findings if f.rule_id == "JT-ESC-001")
    assert esc.fix is not None
    assert esc.fix.kind is FixKind.ADD_ESCALATION
    assert esc.fix.params["from_id"] == n["kim"]
    assert esc.fix.params["candidates"] == [n["lead"]]

    ppl = [f for f in findings if f.rule_id == "JT-PPL-001"]
    assert {f.element for f in ppl} == {Element.OWNER, Element.DETECTION}
    assert all(n["kim"] in f.subject_ids and f.service_id == n["service"] for f in ppl)
    assert all("퇴사자 '김대리'" in f.message for f in ppl)

    report = score(ctx, findings)
    assert report.overall < 30
    owner = next(e for e in report.services[0].elements if e.element is Element.OWNER)
    assert {"JT-PATH-010", "JT-ESC-001"} <= {f.split(":")[0] for f in owner.finding_ids}


def test_competition_demo_fixes_clear_findings_and_raise_score() -> None:
    net, n = demo()
    before = score(net.ctx(), run(net.ctx())).overall
    net.edge(EdgeKind.ESCALATES_TO, n["kim"], n["lead"], after_minutes=30)
    net.edge(EdgeKind.APPROVED_BY, n["action"], n["lead"])
    net.reach(n["lead"], net.channel("박팀장 휴대폰", ChannelMedium.PHONE))
    net.reach(n["lead"], net.channel("결제팀 Slack", ChannelMedium.SLACK))
    ctx = net.ctx()
    findings = run(ctx)
    rule_ids = set(ids(findings))
    assert not rule_ids & {"JT-ESC-001", "JT-PATH-020", "JT-PATH-030"}
    assert "JT-PPL-001" in rule_ids  # 김대리 is still listed as owner
    assert score(ctx, findings).overall > before


# --------------------------------------------------------------------------- JT-PATH


def test_path_001_no_detection() -> None:
    net = Net()
    sid = net.service("결제")
    net.service("사내 위키", tier=3)  # not critical → never flagged
    (f,) = run(net.ctx(), "JT-PATH-001")
    assert f.service_id == sid
    assert f.element is Element.DETECTION
    assert f.status is FindingStatus.BROKEN
    assert f.fix is not None
    assert f.fix.kind is FixKind.ADD_DETECTION
    assert "감지 신호" in f.message
    net.trigger(sid, "알람")
    assert run(net.ctx(), "JT-PATH-001") == []


def test_path_001_held_detection_is_unknown() -> None:
    net = Net()
    sid = net.service("결제")
    tid = net.trigger(sid, "알람", hold=True)
    (f,) = run(net.ctx(), "JT-PATH-001")
    assert f.status is FindingStatus.UNKNOWN
    assert f.title == "감지 경로 확인 불가"
    assert tid in f.subject_ids
    assert f.evidence
    assert run(net.ctx(), "JT-PATH-002") == []


def test_path_002_trigger_pages_nobody() -> None:
    net = Net()
    sid = net.service("결제")
    tid = net.trigger(sid, "알람")
    (f,) = run(net.ctx(), "JT-PATH-002")
    assert "호출 대상이 적혀 있지 않습니다" in f.message
    gone = net.person("김대리", PersonStatus.DEPARTED)
    net.edge(EdgeKind.NOTIFIES, tid, gone)
    (f,) = run(net.ctx(), "JT-PATH-002")
    assert gone in f.subject_ids
    assert "'김대리'(퇴사)" in f.message
    net.edge(EdgeKind.NOTIFIES, tid, net.person("박과장"))
    assert run(net.ctx(), "JT-PATH-002") == []


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (PersonStatus.DEPARTED, FindingStatus.BROKEN),
        (PersonStatus.ON_LEAVE, FindingStatus.BROKEN),
        (None, FindingStatus.UNKNOWN),
    ],
)
def test_path_010_owner_not_active(status: PersonStatus | None, expected: FindingStatus) -> None:
    net = Net()
    sid = net.service("결제")
    owner = net.person("김대리", status) if status else net.stranger("김대리")
    net.edge(EdgeKind.OWNED_BY, sid, owner)
    (f,) = run(net.ctx(), "JT-PATH-010")
    assert f.status is expected
    assert owner in f.subject_ids
    assert f.fix is not None
    assert f.fix.kind is FixKind.ASSIGN_OWNER


def test_path_010_no_owner_and_empty_role() -> None:
    net = Net()
    sid = net.service("결제")
    (f,) = run(net.ctx(), "JT-PATH-010")
    assert "지정되어 있지 않습니다" in f.message
    net.edge(EdgeKind.OWNED_BY, sid, net.role("결제 담당"))
    (f,) = run(net.ctx(), "JT-PATH-010")
    assert "맡은 사람이 없습니다" in f.message
    net.edge(EdgeKind.OWNED_BY, sid, net.person("박과장"))
    assert run(net.ctx(), "JT-PATH-010") == []


def test_path_011_owner_unreachable() -> None:
    net = Net()
    sid = net.service("결제")
    owner = net.person("박과장")
    net.edge(EdgeKind.OWNED_BY, sid, owner)
    (f,) = run(net.ctx(), "JT-PATH-011")
    assert f.fix is not None
    assert f.fix.params["person_ids"] == [owner]
    net.reach(owner, net.channel("휴대폰", ChannelMedium.PHONE))
    assert run(net.ctx(), "JT-PATH-011") == []


def test_path_020_one_finding_per_action_needing_approval() -> None:
    net = Net()
    sid = net.service("결제")
    rollback = net.action(sid, "롤백")
    failover = net.action(sid, "DB 페일오버", ActionType.FAILOVER)
    net.action(sid, "재시작", ActionType.RESTART)  # no approval needed
    found = run(net.ctx(), "JT-PATH-020")
    assert sorted(f.fix.params["action_id"] for f in found if f.fix) == sorted([rollback, failover])
    approver = net.person("박팀장")
    net.reach(approver, net.channel("휴대폰", ChannelMedium.PHONE))
    net.edge(EdgeKind.APPROVED_BY, rollback, approver)
    (f,) = run(net.ctx(), "JT-PATH-020")
    assert f.fix is not None
    assert f.fix.params["action_id"] == failover


def test_path_020_unconfirmed_or_inactive_approver() -> None:
    net = Net()
    sid = net.service("결제")
    aid = net.action(sid, "롤백")
    ghost = net.stranger("최주임")
    net.edge(EdgeKind.APPROVED_BY, aid, ghost)
    (f,) = run(net.ctx(), "JT-PATH-020")
    assert f.status is FindingStatus.UNKNOWN
    assert "명부에서 확인되지 않습니다" in f.message

    net2 = Net()
    sid2 = net2.service("결제")
    aid2 = net2.action(sid2, "롤백")
    net2.edge(EdgeKind.APPROVED_BY, aid2, net2.person("박팀장", PersonStatus.ON_LEAVE))
    (f2,) = run(net2.ctx(), "JT-PATH-020")
    assert f2.status is FindingStatus.BROKEN
    assert "활동 중이고 연락 가능한 사람이 없습니다" in f2.message


def test_path_020_ignores_a_restart_that_only_names_an_approver() -> None:
    """Single approval rule: an approver edge says who approves, not whether approval is needed."""
    net = Net()
    sid = net.service("결제")
    restart = net.action(sid, "결제 서버 재시작", ActionType.RESTART)
    net.edge(EdgeKind.APPROVED_BY, restart, net.person("김대리", PersonStatus.DEPARTED))
    assert run(net.ctx(), "JT-PATH-020") == []


def test_path_020_gated_restart_whose_named_approver_left() -> None:
    """'김대리 승인 후 재시작' (requires_approval=True) waits for 김대리: the replay blocks, so must
    the rule."""
    net = Net()
    sid = net.service("결제")
    restart = net.node(NodeKind.ACTION, "결제 서버 재시작", action_type="restart", requires_approval=True)
    net.edge(EdgeKind.MITIGATED_BY, sid, restart)
    gone = net.person("김대리", PersonStatus.DEPARTED)
    net.edge(EdgeKind.APPROVED_BY, restart, gone)
    (f,) = run(net.ctx(), "JT-PATH-020")
    assert f.status is FindingStatus.BROKEN
    assert f.fix is not None
    assert f.fix.params["action_id"] == restart
    assert f.fix.params["current_approver_ids"] == [gone]
    assert "'결제 서버 재시작'은 승인을 받도록 적혀 있지만" in f.message
    lead = net.person("박팀장")
    net.reach(lead, net.channel("휴대폰", ChannelMedium.PHONE))
    net.edge(EdgeKind.APPROVED_BY, restart, lead)
    assert run(net.ctx(), "JT-PATH-020") == []


def test_path_030_fallback_channel() -> None:
    net = Net()
    sid = net.service("결제")
    owner = net.person("박과장")
    net.edge(EdgeKind.OWNED_BY, sid, owner)
    (f,) = run(net.ctx(), "JT-PATH-030")
    assert f.fix is not None
    assert f.fix.params["person_ids"] == [owner]
    net.reach(owner, net.channel("Slack", ChannelMedium.SLACK))
    (f,) = run(net.ctx(), "JT-PATH-030")
    assert "하나뿐" in f.message
    net.reach(owner, net.channel("휴대폰", ChannelMedium.PHONE))
    assert run(net.ctx(), "JT-PATH-030") == []


def test_path_040_rollback_missing_and_restart_only() -> None:
    net = Net()
    sid = net.service("결제")
    (f,) = run(net.ctx(), "JT-PATH-040")
    assert f.severity is Severity.HIGH
    net.action(sid, "재시작", ActionType.RESTART)
    (f,) = run(net.ctx(), "JT-PATH-040")
    assert f.severity is Severity.MEDIUM
    assert "되돌리는 조치가 없습니다" in f.message
    net.action(sid, "기능 플래그 끄기", ActionType.FEATURE_FLAG)
    assert run(net.ctx(), "JT-PATH-040") == []


def test_path_041_rollback_executor() -> None:
    net = Net()
    sid = net.service("결제")
    first = net.action(sid, "롤백")
    second = net.action(sid, "페일오버", ActionType.FAILOVER)
    net.edge(EdgeKind.EXECUTED_BY, second, net.person("김대리", PersonStatus.DEPARTED))
    found = run(net.ctx(), "JT-PATH-041")
    assert len(found) == 2
    assert all(f.fix and f.fix.params["edge_kind"] == "executed_by" for f in found)
    assert any("'김대리'(퇴사)" in f.message for f in found)
    net.edge(EdgeKind.EXECUTED_BY, first, net.person("박과장"))
    assert run(net.ctx(), "JT-PATH-041") == []


# --------------------------------------------------------------------------- JT-PPL


def test_ppl_001_groups_duties_per_service_and_element() -> None:
    net = Net()
    sid = net.service("결제")
    gone = net.person("김대리", PersonStatus.DEPARTED)
    owner = net.person("박과장")
    owned = net.edge(EdgeKind.OWNED_BY, sid, owner)
    esc = net.edge(EdgeKind.ESCALATES_TO, owner, gone, after_minutes=10)
    aid = net.action(sid, "롤백")
    appr = net.edge(EdgeKind.APPROVED_BY, aid, gone)
    obligation = net.node(NodeKind.OBLIGATION, "KISA 24시간 신고")
    net.edge(EdgeKind.MUST_REPORT, sid, obligation)
    report = net.edge(EdgeKind.REPORTED_BY, obligation, gone)
    orphan_trigger = net.node(NodeKind.TRIGGER, "고아 알람")
    orphan = net.edge(EdgeKind.NOTIFIES, orphan_trigger, gone)
    found = run(net.ctx(), "JT-PPL-001")
    by_key = {(f.service_id, f.element): f for f in found}
    assert set(by_key) == {
        (sid, Element.OWNER),
        (sid, Element.APPROVER),
        (sid, None),
        (None, Element.DETECTION),
    }
    assert by_key[(sid, Element.OWNER)].subject_ids == (gone, esc)
    assert by_key[(sid, Element.APPROVER)].subject_ids == (gone, appr)
    assert by_key[(sid, None)].subject_ids == (gone, report)
    assert by_key[(None, Element.DETECTION)].subject_ids == (gone, orphan)
    fix = by_key[(sid, Element.OWNER)].fix
    assert fix is not None
    assert fix.kind is FixKind.REPLACE_DEPARTED
    assert fix.params["person_id"] == gone
    assert fix.params["edge_ids"] == [esc]
    assert owned not in by_key[(sid, Element.OWNER)].subject_ids
    assert all(f.severity is Severity.CRITICAL and f.status is FindingStatus.BROKEN for f in found)
    assert "서비스에 직접 이어지지 않은 대응 경로" in by_key[(None, Element.DETECTION)].message


def test_ppl_001_ignores_departed_people_without_duties() -> None:
    net = Net()
    net.service("결제")
    net.person("김대리", PersonStatus.DEPARTED)
    assert run(net.ctx(), "JT-PPL-001") == []


def test_ppl_002_unconfirmed_person() -> None:
    net = Net()
    sid = net.service("결제")
    ghost = net.stranger("최주임")
    net.edge(EdgeKind.OWNED_BY, sid, ghost)
    net.edge(EdgeKind.OWNED_BY, net.service("정산"), net.person("박과장"))
    (f,) = run(net.ctx(), "JT-PPL-002")
    assert f.status is FindingStatus.UNKNOWN
    assert f.severity is Severity.MEDIUM
    assert f.fix is not None
    assert f.fix.kind is FixKind.CONFIRM_IDENTITY
    assert f.fix.params["name"] == "최주임"
    assert f.evidence  # the runbook mention is the evidence


def test_ppl_003_on_leave_person() -> None:
    net = Net()
    sid = net.service("결제")
    away = net.person("이과장", PersonStatus.ON_LEAVE)
    lead = net.person("박팀장", title="팀장")
    net.team("결제팀", away, lead)
    net.edge(EdgeKind.OWNED_BY, sid, away)
    (f,) = run(net.ctx(), "JT-PPL-003")
    assert f.fix is not None
    assert f.fix.kind is FixKind.ADD_DEPUTY
    assert f.fix.params["candidates"] == [lead]
    assert "장기 부재 중인 '이과장'" in f.message


def test_ppl_held_person_is_reported_as_unknown() -> None:
    net = Net()
    sid = net.service("결제")
    gone = net.node(NodeKind.PERSON, "김대리", ROSTER, hold=True, status="departed")
    net.edge(EdgeKind.OWNED_BY, sid, gone)
    (f,) = run(net.ctx(), "JT-PPL-001")
    assert f.status is FindingStatus.UNKNOWN


# --------------------------------------------------------------------------- JT-ESC


def _owned(net: Net, name: str = "결제") -> tuple[str, str]:
    sid = net.service(name)
    owner = net.person("박과장")
    net.edge(EdgeKind.OWNED_BY, sid, owner)
    return sid, owner


def test_esc_001_missing_escalation() -> None:
    net = Net()
    sid, owner = _owned(net)
    (f,) = run(net.ctx(), "JT-ESC-001")
    assert f.service_id == sid
    assert f.fix is not None
    assert f.fix.params["from_id"] == owner
    net.edge(EdgeKind.ESCALATES_TO, owner, net.person("박팀장"), after_minutes=10)
    assert run(net.ctx(), "JT-ESC-001") == []


def test_esc_001_needs_an_owner_and_reports_held_edges_as_unknown() -> None:
    net = Net()
    net.service("결제")
    assert run(net.ctx(), "JT-ESC-001") == []
    net2 = Net()
    _, owner = _owned(net2)
    net2.edge(EdgeKind.ESCALATES_TO, owner, net2.person("박팀장"), hold=True)
    (f,) = run(net2.ctx(), "JT-ESC-001")
    assert f.status is FindingStatus.UNKNOWN


def test_esc_002_cycle_without_exit() -> None:
    net = Net()
    sid, owner = _owned(net)
    peer = net.person("이과장")
    e1 = net.edge(EdgeKind.ESCALATES_TO, owner, peer, after_minutes=10)
    e2 = net.edge(EdgeKind.ESCALATES_TO, peer, owner, after_minutes=10)
    (f,) = run(net.ctx(), "JT-ESC-002")
    assert f.service_id == sid
    assert f.fix is not None
    assert f.fix.kind is FixKind.BREAK_ESCALATION_CYCLE
    assert f.fix.params["cycle"] == list(canonical_cycle([peer, owner]))
    assert f.fix.params["edge_ids"] == sorted([e1, e2])
    assert "→" in f.message


def test_esc_002_cycle_with_exit_to_terminal_role_is_fine() -> None:
    net = Net()
    _, owner = _owned(net)
    peer = net.person("이과장")
    net.edge(EdgeKind.ESCALATES_TO, owner, peer, after_minutes=10)
    net.edge(EdgeKind.ESCALATES_TO, peer, owner, after_minutes=10)
    net.edge(EdgeKind.ESCALATES_TO, peer, net.role("CTO", net.person("정CTO")), after_minutes=30)
    assert run(net.ctx(), "JT-ESC-002") == []


def test_esc_002_cycle_through_active_terminal_role_is_fine() -> None:
    net = Net()
    _, owner = _owned(net)
    cto = net.role("CTO", net.person("정CTO"))
    net.edge(EdgeKind.ESCALATES_TO, owner, cto, after_minutes=10)
    net.edge(EdgeKind.ESCALATES_TO, cto, owner, after_minutes=10)
    assert run(net.ctx(), "JT-ESC-002") == []


def test_esc_002_self_loop() -> None:
    net = Net()
    _, owner = _owned(net)
    net.edge(EdgeKind.ESCALATES_TO, owner, owner, after_minutes=5)
    (f,) = run(net.ctx(), "JT-ESC-002")
    assert f.fix is not None
    assert f.fix.params["cycle"] == [owner]


def test_canonical_cycle() -> None:
    assert canonical_cycle(["c", "a", "b"]) == ("a", "b", "c")
    assert canonical_cycle(["b", "a"]) == ("a", "b")
    assert canonical_cycle(["x"]) == ("x",)


def test_esc_003_dead_end() -> None:
    net = Net()
    sid, owner = _owned(net)
    gone = net.person("이부장", PersonStatus.DEPARTED)
    net.edge(EdgeKind.ESCALATES_TO, owner, gone, after_minutes=10)
    (f,) = run(net.ctx(), "JT-ESC-003")
    assert f.service_id == sid
    assert f.fix is not None
    assert f.fix.params["from_id"] == gone
    assert run(net.ctx(), "JT-ESC-001") == []
    net.edge(EdgeKind.ESCALATES_TO, gone, net.person("박팀장"), after_minutes=10)
    assert run(net.ctx(), "JT-ESC-003") == []


def test_esc_004_missing_wait_time() -> None:
    net = Net()
    sid, owner = _owned(net)
    untimed = net.edge(EdgeKind.ESCALATES_TO, owner, net.person("박팀장"))
    (f,) = run(net.ctx(), "JT-ESC-004")
    assert f.severity is Severity.LOW
    assert f.subject_ids == (sid, untimed)
    assert f.fix is not None
    assert f.fix.kind is FixKind.FIX_DEADLINE
    assert f.fix.params == {"service_id": sid, "edge_id": untimed, "attr": "after_minutes"}


def test_esc_004_timed_escalation_is_fine() -> None:
    net = Net()
    _, owner = _owned(net)
    net.edge(EdgeKind.ESCALATES_TO, owner, net.person("박팀장"), after_minutes=15)
    assert run(net.ctx(), "JT-ESC-004") == []


# --------------------------------------------------------------------------- JT-CHN


def test_chn_001_channels_die_with_the_service() -> None:
    net = Net()
    idc = net.node(NodeKind.LOCATION, "판교 IDC", CATALOG)
    sid = net.service("결제", location=idc)
    owner = net.person("박과장")
    net.edge(EdgeKind.OWNED_BY, sid, owner)
    messenger = net.channel("사내 메신저", ChannelMedium.INTERNAL_MESSENGER, located=idc)
    net.reach(owner, messenger)
    (f,) = run(net.ctx(), "JT-CHN-001")
    assert idc in f.subject_ids
    assert messenger in f.subject_ids
    assert "'판교 IDC'" in f.message
    assert f.fix is not None
    assert f.fix.params["independent_of"] == [idc]
    net.reach(owner, net.channel("휴대폰", ChannelMedium.PHONE))
    assert run(net.ctx(), "JT-CHN-001") == []


def test_chn_001_requires_some_channel() -> None:
    net = Net()
    _owned(net)
    assert run(net.ctx(), "JT-CHN-001") == []


def test_chn_002_fallback_channel_nobody_uses() -> None:
    net = Net()
    owner = net.person("박과장")
    slack = net.channel("Slack", ChannelMedium.SLACK)
    kakao = net.channel("카카오톡", ChannelMedium.KAKAOTALK)
    net.reach(owner, slack)
    fb = net.edge(EdgeKind.FALLBACK_TO, slack, kakao)
    (f,) = run(net.ctx(), "JT-CHN-002")
    assert f.subject_ids == (kakao, fb, slack)
    assert f.fix is not None
    assert f.fix.params["person_ids"] == [owner]
    assert f.service_id is None
    net.reach(owner, kakao)
    assert run(net.ctx(), "JT-CHN-002") == []


# --------------------------------------------------------------------------- held items → 확인 필요


def _held_role_holder(hold: bool) -> CheckContext:
    net = Net()
    sid = net.service("결제")
    role = net.node(NodeKind.ROLE, "1차 당직", ROSTER)
    net.edge(EdgeKind.HOLDS_ROLE, net.person("박과장"), role, prov=ROSTER, hold=hold)
    net.edge(EdgeKind.NOTIFIES, net.trigger(sid, "알람"), role)
    return net.ctx()


def _held_owner_edge(hold: bool) -> CheckContext:
    net = Net()
    net.edge(EdgeKind.OWNED_BY, net.service("결제"), net.person("박과장"), hold=hold)
    return net.ctx()


def _held_owner_channel(hold: bool) -> CheckContext:
    net = Net()
    _, owner = _owned(net)
    phone = net.channel("휴대폰", ChannelMedium.PHONE)
    net.edge(EdgeKind.REACHABLE_VIA, owner, phone, prov=ROSTER, hold=hold)
    return net.ctx()


def _held_approver(hold: bool) -> CheckContext:
    net = Net()
    aid = net.action(net.service("결제"), "롤백")
    lead = net.person("박팀장")
    net.reach(lead, net.channel("휴대폰", ChannelMedium.PHONE))
    net.edge(EdgeKind.APPROVED_BY, aid, lead, hold=hold)
    return net.ctx()


def _held_second_channel(hold: bool) -> CheckContext:
    net = Net()
    _, owner = _owned(net)
    net.reach(owner, net.channel("휴대폰", ChannelMedium.PHONE))
    slack = net.channel("Slack", ChannelMedium.SLACK)
    net.edge(EdgeKind.REACHABLE_VIA, owner, slack, prov=ROSTER, hold=hold)
    return net.ctx()


def _held_mitigation(hold: bool) -> CheckContext:
    net = Net()
    sid = net.service("결제")
    aid = net.node(NodeKind.ACTION, "롤백", action_type=ActionType.ROLLBACK.value)
    net.edge(EdgeKind.MITIGATED_BY, sid, aid, hold=hold)
    return net.ctx()


def _held_executor(hold: bool) -> CheckContext:
    net = Net()
    aid = net.action(net.service("결제"), "롤백")
    net.edge(EdgeKind.EXECUTED_BY, aid, net.person("박과장"), hold=hold)
    return net.ctx()


def _held_cycle_exit(hold: bool) -> CheckContext:
    net = Net()
    _, owner = _owned(net)
    peer = net.person("이과장")
    net.edge(EdgeKind.ESCALATES_TO, owner, peer, after_minutes=10)
    net.edge(EdgeKind.ESCALATES_TO, peer, owner, after_minutes=10)
    cto = net.role("CTO", net.person("정CTO"))
    net.edge(EdgeKind.ESCALATES_TO, peer, cto, after_minutes=30, hold=hold)
    return net.ctx()


def _held_dead_end_exit(hold: bool) -> CheckContext:
    net = Net()
    _, owner = _owned(net)
    gone = net.person("이부장", PersonStatus.DEPARTED)
    net.edge(EdgeKind.ESCALATES_TO, owner, gone, after_minutes=10)
    net.edge(EdgeKind.ESCALATES_TO, gone, net.person("박팀장"), after_minutes=10, hold=hold)
    return net.ctx()


def _held_independent_channel(hold: bool) -> CheckContext:
    net = Net()
    idc = net.node(NodeKind.LOCATION, "판교 IDC", CATALOG)
    sid = net.service("결제", location=idc)
    owner = net.person("박과장")
    net.edge(EdgeKind.OWNED_BY, sid, owner)
    net.reach(owner, net.channel("사내 메신저", ChannelMedium.INTERNAL_MESSENGER, located=idc))
    phone = net.channel("휴대폰", ChannelMedium.PHONE)
    net.edge(EdgeKind.REACHABLE_VIA, owner, phone, prov=ROSTER, hold=hold)
    return net.ctx()


def _held_fallback_member(hold: bool) -> CheckContext:
    net = Net()
    owner = net.person("박과장")
    slack = net.channel("Slack", ChannelMedium.SLACK)
    kakao = net.channel("카카오톡", ChannelMedium.KAKAOTALK)
    net.reach(owner, slack)
    net.edge(EdgeKind.FALLBACK_TO, slack, kakao)
    net.edge(EdgeKind.REACHABLE_VIA, owner, kakao, prov=ROSTER, hold=hold)
    return net.ctx()


HELD_CASES = {
    "JT-PATH-002": _held_role_holder,
    "JT-PATH-010": _held_owner_edge,
    "JT-PATH-011": _held_owner_channel,
    "JT-PATH-020": _held_approver,
    "JT-PATH-030": _held_second_channel,
    "JT-PATH-040": _held_mitigation,
    "JT-PATH-041": _held_executor,
    "JT-ESC-002": _held_cycle_exit,
    "JT-ESC-003": _held_dead_end_exit,
    "JT-CHN-001": _held_independent_channel,
    "JT-CHN-002": _held_fallback_member,
}


@pytest.mark.parametrize("rule_id", sorted(HELD_CASES))
def test_gap_that_a_held_item_would_close_is_unknown_not_broken(rule_id: str) -> None:
    """P3: a held link is not evidence, but it is not proof of absence either."""
    build = HELD_CASES[rule_id]
    (f,) = run(build(True), rule_id)
    assert f.status is FindingStatus.UNKNOWN
    assert "판단 보류(HOLD)" in f.message
    assert run(build(False), rule_id) == []  # confirming the held item closes the gap


def test_unrelated_held_items_do_not_soften_a_broken_verdict() -> None:
    net = Net()
    unowned = net.service("결제")
    other = net.service("정산")
    net.edge(EdgeKind.OWNED_BY, other, net.person("박과장"), hold=True)
    found = {f.service_id: f for f in run(net.ctx(), "JT-PATH-010")}
    assert found[unowned].status is FindingStatus.BROKEN
    assert "판단 보류" not in found[unowned].message
    assert found[other].status is FindingStatus.UNKNOWN


# --------------------------------------------------------------------------- dense escalation loops


def _escalation_clique(net: Net, size: int = 8) -> tuple[str, str]:
    """An owner plus departed colleagues who all escalate to each other (K_size, no way out)."""
    sid, owner = _owned(net)
    everyone = [owner, *(net.person(f"퇴사자{i}", PersonStatus.DEPARTED) for i in range(size - 1))]
    for src in everyone:
        for dst in everyone:
            if src != dst:
                net.edge(EdgeKind.ESCALATES_TO, src, dst, after_minutes=10)
    return sid, owner


def test_esc_002_dense_loops_are_bounded() -> None:
    net = Net()
    sid, owner = _escalation_clique(net)
    ctx = net.ctx()
    esc = analyze_owner(ctx, sid).escalation
    cycles = escalation_cycles(ctx, esc)
    assert len(cycles) == escalation.MAX_CYCLES_EXAMINED  # K8 has 16,064 simple cycles
    assert cycles == sorted(set(cycles))
    assert all(c == canonical_cycle(c) for c in cycles)
    assert escalation_cycles(ctx, esc, limit=0) == []
    found = run(ctx, "JT-ESC-002")
    assert len(found) == escalation.MAX_CYCLE_FINDINGS
    # loops without the owner lead back to the (active) owner; only loops through the owner are stuck
    assert all(f.fix is not None and owner in f.fix.params["cycle"] for f in found)


@pytest.mark.parametrize(("hops", "flagged"), [(MAX_ESCALATION_HOPS, False), (MAX_ESCALATION_HOPS + 1, True)])
def test_esc_002_exit_must_lie_within_the_escalation_hop_bound(hops: int, flagged: bool) -> None:
    net = Net()
    _, owner = _owned(net)
    peer = net.person("이과장")
    net.edge(EdgeKind.ESCALATES_TO, owner, peer, after_minutes=10)
    net.edge(EdgeKind.ESCALATES_TO, peer, owner, after_minutes=10)
    chain = [peer, *(net.person(f"퇴사자{i}", PersonStatus.DEPARTED) for i in range(hops - 1))]
    cto = net.role("CTO", net.person("정CTO"))
    for src, dst in zip(chain, [*chain[1:], cto], strict=True):
        net.edge(EdgeKind.ESCALATES_TO, src, dst, after_minutes=10)
    assert bool(run(net.ctx(), "JT-ESC-002")) is flagged


def test_canonical_cycle_of_nothing_is_empty() -> None:
    assert canonical_cycle([]) == ()


_ESC_002_SCRIPT = """
import sys
from pathlib import Path
from jettae.check.context import CheckContext
from jettae.check.rules.escalation import EscalationCycle
from jettae.domain import OpsGraph
graph = OpsGraph.model_validate_json(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(",".join(f.id for f in EscalationCycle().check(CheckContext(graph=graph))))
"""


def test_esc_002_is_identical_across_hash_seeds(tmp_path: Path) -> None:
    """P6: string hashing differs per process; the bounded cycle sample must not."""
    net = Net()
    _escalation_clique(net)
    graph = net.build()
    graph_file = tmp_path / "graph.json"
    graph_file.write_text(graph.canonical_json(), encoding="utf-8")
    assert OpsGraph.model_validate_json(graph.canonical_json()).content_hash() == graph.content_hash()
    expected = ",".join(f.id for f in escalation.EscalationCycle().check(CheckContext(graph=graph)))
    for seed in ("0", "1", "4242"):
        result = subprocess.run(
            [sys.executable, "-c", _ESC_002_SCRIPT, str(graph_file)],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        assert result.stdout.strip() == expected


# --------------------------------------------------------------------------- properties


def _pairs(
    draw: st.DrawFn, srcs: list[str], dsts: list[str], odds: int, *, allow_self: bool = False
) -> list[tuple[str, str]]:
    """Each (src, dst) pair independently with probability 1/odds."""
    return [
        (a, b) for a in srcs for b in dsts if (allow_self or a != b) and draw(st.integers(0, odds - 1)) == 0
    ]


def _add_service(draw: st.DrawFn, net: Net, index: int, idc: str, who: list[str]) -> None:
    sid = net.service(
        f"서비스{index}", tier=draw(st.integers(1, 3)), location=idc if draw(st.booleans()) else None
    )
    for _, owner in _pairs(draw, [sid], who, 3):
        net.edge(EdgeKind.OWNED_BY, sid, owner, hold=draw(st.integers(0, 9)) == 0)
    for t in range(draw(st.integers(0, 2))):
        trig = net.trigger(sid, f"서비스{index} 알람{t}", hold=draw(st.integers(0, 9)) == 0)
        for _, target in _pairs(draw, [trig], who, 3):
            net.edge(EdgeKind.NOTIFIES, trig, target)
    for a in range(draw(st.integers(0, 2))):
        aid = net.action(sid, f"서비스{index} 조치{a}", draw(st.sampled_from([*ActionType, None])))
        for _, approver in _pairs(draw, [aid], who, 3):
            net.edge(EdgeKind.APPROVED_BY, aid, approver)
        for _, doer in _pairs(draw, [aid], who, 4):
            net.edge(EdgeKind.EXECUTED_BY, aid, doer)


@st.composite
def graphs(draw: st.DrawFn) -> OpsGraph:
    net = Net()
    idc = net.node(NodeKind.LOCATION, "IDC", CATALOG)
    people_ids = []
    for i in range(draw(st.integers(1, 4))):
        status = draw(st.sampled_from([*PersonStatus, None]))
        people_ids.append(net.person(f"사람{i}", status) if status else net.stranger(f"사람{i}"))
    roles = [net.role(f"역할{i}") for i in range(draw(st.integers(0, 2)))]
    if draw(st.booleans()):
        roles.append(net.role("CTO"))
    for person, role in _pairs(draw, people_ids, roles, 2):
        net.edge(EdgeKind.HOLDS_ROLE, person, role, prov=ROSTER)
    who = [*people_ids, *roles]
    media = st.sampled_from(list(ChannelMedium))
    chans = [
        net.channel(f"채널{i}", draw(media), located=idc if draw(st.booleans()) else None)
        for i in range(draw(st.integers(0, 3)))
    ]
    for person, channel in _pairs(draw, who, chans, 3):
        net.reach(person, channel)
    for primary, backup in _pairs(draw, chans, chans, 5):
        net.edge(EdgeKind.FALLBACK_TO, primary, backup)
    for index in range(draw(st.integers(1, 2))):
        _add_service(draw, net, index, idc, who)
    for src, dst in _pairs(draw, who, who, 4, allow_self=True):
        net.edge(EdgeKind.ESCALATES_TO, src, dst, after_minutes=draw(st.sampled_from([None, 10])))
    return net.build()


PROPERTY = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@PROPERTY
@given(graphs())
def test_findings_are_sorted_unique_and_well_formed(graph: OpsGraph) -> None:
    ctx = CheckContext(graph=graph)
    produced = raw(ctx)
    assert len({f.id for f in produced}) == len(produced)
    ordered = run(ctx)
    assert ordered == sort_findings(list(reversed(produced)))
    for f in ordered:
        assert f.id.startswith(f"{f.rule_id}:")
        assert f.title
        assert f.message
        assert f.service_id is None or graph.nodes[f.service_id].kind is NodeKind.SERVICE
        assert all(i in graph.nodes or i in graph.edges for i in f.subject_ids)
        if f.fix is not None:
            json.dumps(f.fix.params, ensure_ascii=False)


@PROPERTY
@given(graphs())
def test_rules_are_deterministic_under_rebuild(graph: OpsGraph) -> None:
    rebuilt = GraphBuilder()
    for n in reversed(list(graph.nodes.values())):
        rebuilt.add_node(n)
    for e in reversed(list(graph.edges.values())):
        rebuilt.add_edge(e)
    first = run(CheckContext(graph=graph))
    second = run(CheckContext(graph=rebuilt.build()))
    assert [f.model_dump_json() for f in first] == [f.model_dump_json() for f in second]


@PROPERTY
@given(graphs())
def test_path_rules_agree_with_element_scores(graph: OpsGraph) -> None:
    """A JT-PATH finding exists exactly when the matching element is broken for that reason."""
    ctx = CheckContext(graph=graph)
    findings = run(ctx, *[r for r in INSTANCES if r.startswith("JT-PATH")])
    flagged = {(f.rule_id, f.service_id) for f in findings}
    broken = (ElementStatus.FAIL, ElementStatus.UNKNOWN)
    for svc in ctx.critical_services():
        evals = evaluate_service(ctx, svc.id)
        assert (("JT-PATH-001", svc.id) in flagged) == (evals[Element.DETECTION].score == 0.0)
        assert (("JT-PATH-010", svc.id) in flagged) == (evals[Element.OWNER].status in broken)
        assert (("JT-PATH-030", svc.id) in flagged) == (evals[Element.FALLBACK_CHANNEL].score <= 0.25)
        assert (("JT-PATH-040", svc.id) in flagged) == (evals[Element.ROLLBACK].score < 0.6)
        approver = evals[Element.APPROVER]
        has_actions = approver.reasons[0] != "조치가 없어 승인 경로를 평가할 수 없음"
        assert (("JT-PATH-020", svc.id) in flagged) == (approver.status in broken and has_actions)
        if ("JT-PATH-020", svc.id) in flagged:
            assert approver.score <= 0.2


# --------------------------------------------------------------------------- review: rule order & holds


def _warm(ctx: CheckContext) -> CheckContext:
    """Query every node once so the graph's lazy adjacency index exists (as after earlier rules)."""
    for nid in ctx.graph.nodes:
        ctx.graph.out_edges(nid)
        ctx.graph.in_edges(nid)
    return ctx


def _fresh(graph: OpsGraph) -> OpsGraph:
    """An equal graph object whose adjacency index has not been built yet."""
    return GraphBuilder(graph).build()


@pytest.mark.parametrize("rule_id", sorted(HELD_CASES))
def test_held_verdict_survives_earlier_rules_having_queried_the_graph(rule_id: str) -> None:
    """Regression (P3): with a pre-built index the 'confirmed' view still saw the HOLD edges."""
    (alone,) = run(_warm(HELD_CASES[rule_id](True)), rule_id)
    assert alone.status is FindingStatus.UNKNOWN
    everything = [f for f in registry.run_checks(HELD_CASES[rule_id](True)) if f.rule_id == rule_id]
    assert [f.status for f in everything] == [FindingStatus.UNKNOWN]
    assert everything == [alone]


@st.composite
def held_graphs(draw: st.DrawFn) -> OpsGraph:
    """Random graphs from :func:`graphs` with a few more nodes and edges put on HOLD."""
    graph = draw(graphs())
    held_edges = draw(st.sets(st.sampled_from(sorted(graph.edges)), max_size=4)) if graph.edges else set()
    candidates = sorted(
        nid
        for nid, n in graph.nodes.items()
        if n.kind in (NodeKind.PERSON, NodeKind.ROLE, NodeKind.CHANNEL, NodeKind.ACTION)
    )
    held_nodes = draw(st.sets(st.sampled_from(candidates), max_size=2)) if candidates else set()
    builder = GraphBuilder()
    for nid, n in graph.nodes.items():
        builder.add_node(n.model_copy(update={"verification": Verification.HOLD}) if nid in held_nodes else n)
    for eid, e in graph.edges.items():
        builder.add_edge(e.model_copy(update={"verification": Verification.HOLD}) if eid in held_edges else e)
    return builder.build()


@PROPERTY
@given(held_graphs())
def test_each_rule_gives_the_same_findings_alone_or_after_the_others(graph: OpsGraph) -> None:
    """P6: a rule's verdict must not depend on which rules happened to run before it."""
    forward = raw(CheckContext(graph=_fresh(graph)))
    shared = _warm(CheckContext(graph=_fresh(graph)))
    backward = [f for rid in sorted(INSTANCES, reverse=True) for f in INSTANCES[rid].check(shared)]
    assert sort_findings(forward) == sort_findings(backward)
    for rid, rule in INSTANCES.items():
        alone = list(rule.check(CheckContext(graph=_fresh(graph))))
        assert alone == [f for f in forward if f.rule_id == rid], rid


#: Rules whose finding means the matching element cannot be at full score.
_NOT_PASS_RULES = frozenset(
    {
        "JT-PATH-001",
        "JT-PATH-002",
        "JT-PATH-010",
        "JT-PATH-011",
        "JT-PATH-020",
        "JT-PATH-030",
        "JT-PATH-040",
        "JT-PATH-041",
        "JT-ESC-001",
        "JT-ESC-003",
        "JT-CHN-001",
    }
)


def _spans(graph: OpsGraph, item_id: str) -> tuple[SourceSpan, ...]:
    item: Node | Edge | None = graph.nodes.get(item_id) or graph.edges.get(item_id)
    return item.spans if item is not None else ()


@PROPERTY
@given(held_graphs())
def test_findings_and_element_scores_never_contradict(graph: OpsGraph) -> None:
    """Rules and score share one evaluator: a gap finding never sits next to a passing element."""
    ctx = CheckContext(graph=graph)
    findings = run(ctx)
    report = score(ctx, findings)
    elements = {(s.service_id, e.element): e for s in report.services for e in s.elements}
    for f in findings:
        if f.service_id is not None and f.element is not None:
            assert f.id in elements[(f.service_id, f.element)].finding_ids
        if f.rule_id in _NOT_PASS_RULES:
            assert f.service_id is not None
            assert f.element is not None
            assert elements[(f.service_id, f.element)].status is not ElementStatus.PASS, f.rule_id
        # P2: evidence is exactly the source spans of what the finding points at, nothing invented
        assert set(f.evidence) <= {s for i in f.subject_ids for s in _spans(graph, i)}


# --------------------------------------------------------------------------- review: wording & fixes


def test_path_041_asks_for_an_executor_not_an_owner() -> None:
    """The drafter must add EXECUTED_BY, so the hint must say ASSIGN_EXECUTOR (not ASSIGN_OWNER)."""
    net = Net()
    sid = net.service("결제")
    aid = net.action(sid, "롤백")
    (f,) = run(net.ctx(), "JT-PATH-041")
    assert f.fix is not None
    assert f.fix.kind is FixKind.ASSIGN_EXECUTOR
    assert f.fix.params["action_id"] == aid
    assert f.fix.params["edge_kind"] == EdgeKind.EXECUTED_BY.value


def test_particles_agree_with_the_name_not_the_status_in_brackets() -> None:
    net = Net()
    sid = net.service("결제")
    away = net.person("박과장", PersonStatus.ON_LEAVE)
    net.edge(EdgeKind.OWNED_BY, sid, away)
    tid = net.trigger(sid, "알람")
    net.edge(EdgeKind.NOTIFIES, tid, net.person("최과장", PersonStatus.DEPARTED))
    rollback = net.action(sid, "롤백")
    net.edge(EdgeKind.EXECUTED_BY, rollback, net.person("정과장", PersonStatus.ON_LEAVE))
    ctx = net.ctx()
    (owner,) = run(ctx, "JT-PATH-010")
    assert "지정된 담당자 '박과장'(장기 부재)은 활동 중이 아닙니다" in owner.message
    (notify,) = run(ctx, "JT-PATH-002")
    assert "호출 대상 '최과장'(퇴사)이 활동 중인 대응자로" in notify.message
    (executor,) = run(ctx, "JT-PATH-041")
    assert "실행자 '정과장'(장기 부재)이 활동 중인 대응자로" in executor.message


def test_messages_keep_multiline_runbook_names_on_one_line() -> None:
    net = Net()
    sid = net.service("결제\n  서비스")
    (f,) = run(net.ctx(), "JT-PATH-001")
    assert f.service_id == sid
    assert f.message.startswith("'결제 서비스'는 장애를 알아챌")
    assert all("\n" not in g.message for g in run(net.ctx()))


def test_parallel_edges_do_not_repeat_names_in_messages() -> None:
    """Edge ids may carry a discriminator (``#disc``, e.g. one NOTIFIES per channel)."""
    net = Net()
    sid = net.service("결제")
    gone = net.person("김대리", PersonStatus.DEPARTED)
    tid = net.trigger(sid, "알람")
    aid = net.action(sid, "롤백")
    sources = {
        EdgeKind.OWNED_BY: sid,
        EdgeKind.NOTIFIES: tid,
        EdgeKind.APPROVED_BY: aid,
        EdgeKind.EXECUTED_BY: aid,
    }
    for disc in ("slack", "phone"):
        for kind, src in sources.items():
            eid = edge_id(kind, src, gone, disc)
            net.b.add_edge(Edge(id=eid, kind=kind, src=src, dst=gone, prov=(rb(f"{kind.value} {disc}"),)))
    ctx = net.ctx()
    found = run(ctx)
    assert {"JT-PATH-002", "JT-PATH-010", "JT-PATH-020", "JT-PATH-041", "JT-ESC-001", "JT-PPL-001"} <= set(
        ids(found)
    )
    for f in found:
        assert f.message.count("'김대리'") <= 1, (f.rule_id, f.message)
        if f.fix is not None:
            for value in f.fix.params.values():
                if isinstance(value, list):
                    assert len(value) == len(set(value)), (f.rule_id, value)
    for element in evaluate_service(ctx, sid).values():
        assert all(r.count("'김대리'") <= 1 for r in element.reasons), element.reasons


def test_empty_graph_and_non_critical_services_yield_no_core_findings() -> None:
    assert run(CheckContext(graph=OpsGraph())) == []
    net = Net()
    wiki = net.service("사내 위키", tier=3)
    gone = net.person("김대리", PersonStatus.DEPARTED)
    net.edge(EdgeKind.OWNED_BY, wiki, gone)
    found = run(net.ctx())
    # only the people rules look beyond critical services: a departed owner is still a dead page
    assert ids(found) == ["JT-PPL-001"]
    assert found[0].service_id == wiki


def test_people_rules_rank_replacements_once_per_person(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ranking the roster per duty made PPL-001/003 take ~20 s on a 300-person organisation."""
    net = Net()
    gone = net.person("김대리", PersonStatus.DEPARTED)
    net.person("최과장", PersonStatus.DEPARTED)  # a leaver with no duty: nothing to rank for
    for name in ("결제", "정산", "배송"):
        sid = net.service(name)
        net.edge(EdgeKind.OWNED_BY, sid, gone)
        net.edge(EdgeKind.NOTIFIES, net.trigger(sid, f"{name} 알람"), gone)
        net.edge(EdgeKind.APPROVED_BY, net.action(sid, f"{name} 롤백"), gone)
    calls: list[tuple[str, ...]] = []
    real = elements.suggest_candidates

    def counting(ctx: CheckContext, anchor_ids: tuple[str, ...]) -> list[str]:
        calls.append(tuple(anchor_ids))
        return real(ctx, anchor_ids)

    monkeypatch.setattr(people, "suggest_candidates", counting)
    found = run(net.ctx(), "JT-PPL-001")
    assert len(found) == 9  # 3 services × (owner, detection, approver)
    assert calls == [(gone,)]
    params = [f.fix.params["candidates"] for f in found if f.fix is not None]
    assert len(params) == 9
    assert len({id(p) for p in params}) == 9  # no list object shared between findings


# --------------------------------------------------------------------------- review: P6 across processes


def _mixed_graph() -> OpsGraph:
    """Every core rule family at once: departed/on-leave/unknown people, holds, loops, channels."""
    net = Net()
    idc = net.node(NodeKind.LOCATION, "판교 IDC", CATALOG)
    pay = net.service("결제", location=idc)
    settle = net.service("정산", tier=2)
    net.service("사내 위키", tier=3)
    kim = net.person("김대리", PersonStatus.DEPARTED, title="대리")
    lead = net.person("박팀장", title="팀장")
    away = net.person("이과장", PersonStatus.ON_LEAVE)
    ghost = net.stranger("최주임")
    held = net.node(NodeKind.PERSON, "한대리", ROSTER, hold=True, status="active")
    cto = net.role("CTO", net.person("정CTO"))
    net.team("결제팀", kim, lead, away, held)
    oncall = net.node(NodeKind.ROLE, "1차 당직", ROSTER)
    net.edge(EdgeKind.HOLDS_ROLE, lead, oncall, prov=ROSTER, hold=True)
    messenger = net.channel("사내 메신저", ChannelMedium.INTERNAL_MESSENGER, located=idc)
    slack = net.channel("Slack", ChannelMedium.SLACK, located=idc)
    kakao = net.channel("카카오톡", ChannelMedium.KAKAOTALK, located=idc)
    net.reach(lead, messenger, slack)
    net.edge(EdgeKind.FALLBACK_TO, slack, kakao)
    net.edge(EdgeKind.OWNED_BY, pay, kim)
    net.edge(EdgeKind.OWNED_BY, pay, oncall)
    net.edge(EdgeKind.OWNED_BY, settle, ghost)
    net.edge(EdgeKind.NOTIFIES, net.trigger(pay, "결제 실패율 5% 초과"), kim)
    net.trigger(settle, "정산 지연 알람", hold=True)
    rollback = net.action(pay, "이전 버전으로 롤백")
    net.edge(EdgeKind.APPROVED_BY, rollback, ghost)
    net.edge(EdgeKind.EXECUTED_BY, rollback, away)
    net.action(settle, "배치 재시작", ActionType.RESTART)
    net.edge(EdgeKind.ESCALATES_TO, kim, away, after_minutes=10)
    net.edge(EdgeKind.ESCALATES_TO, away, kim)
    net.edge(EdgeKind.ESCALATES_TO, oncall, lead, after_minutes=5)
    net.edge(EdgeKind.ESCALATES_TO, ghost, cto, hold=True)
    return net.build()


_ALL_CORE_RULES_SCRIPT = """
import json
import sys
from pathlib import Path
from jettae.check.context import CheckContext
from jettae.check.registry import sort_findings
from jettae.check.rules import channels, escalation, path, people
from jettae.check.scoring import score
from jettae.domain import OpsGraph
graph = OpsGraph.model_validate_json(Path(sys.argv[1]).read_text(encoding="utf-8"))
ctx = CheckContext(graph=graph)
rules = sorted(
    (obj() for mod in (channels, escalation, path, people) for obj in vars(mod).values()
     if isinstance(obj, type) and obj.__module__ == mod.__name__ and hasattr(obj, "info")),
    key=lambda r: r.info.rule_id,
)
findings = sort_findings(f for r in rules for f in r.check(ctx))
payload = [f.model_dump(mode="json") for f in findings] + [score(ctx, findings).model_dump(mode="json")]
print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
"""


def test_all_core_rules_and_the_score_are_identical_across_hash_seeds(tmp_path: Path) -> None:
    graph = _mixed_graph()
    graph_file = tmp_path / "graph.json"
    graph_file.write_text(graph.canonical_json(), encoding="utf-8")
    outputs = {
        seed: subprocess.run(
            [sys.executable, "-c", _ALL_CORE_RULES_SCRIPT, str(graph_file)],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        ).stdout
        for seed in ("0", "7", "4242")
    }
    assert len(set(outputs.values())) == 1
    payload = json.loads(outputs["0"])
    findings = payload[:-1]
    rule_ids = {f["rule_id"] for f in findings}
    assert {"JT-PPL-001", "JT-PPL-002", "JT-PPL-003", "JT-ESC-002", "JT-PATH-020", "JT-CHN-001"} <= rule_ids
    assert {f["status"] for f in findings} == {"broken", "unknown"}
    assert [f["id"] for f in findings] == [f.id for f in run(CheckContext(graph=graph))]
