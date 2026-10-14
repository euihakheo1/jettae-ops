"""Tests for the required-path element evaluator (jettae.check.elements, SCORING_V1 semantics)."""

from __future__ import annotations

import unicodedata

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jettae.check.context import CheckContext
from jettae.check.elements import (
    MAX_ESCALATION_HOPS,
    ApprovalGate,
    ElementEval,
    action_type,
    active_responders,
    analyze_channels,
    analyze_escalation,
    approval_gate,
    channel_medium,
    confirmed_view,
    evaluate_service,
    is_terminal,
    josa,
    label,
    needs_approval,
    person_label,
    reachable_channels,
    service_label,
    spans_of,
    suggest_candidates,
)
from jettae.check.spof import GraphIndex, gated_actions
from jettae.domain import (
    ActionType,
    Attr,
    ChannelMedium,
    EdgeKind,
    Element,
    ElementStatus,
    Extractor,
    GraphBuilder,
    Node,
    NodeKind,
    OpsGraph,
    PersonStatus,
    Policy,
    Provenance,
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
    """Tiny DSL over GraphBuilder: roster facts carry ROSTER provenance, runbook facts a span."""

    def __init__(self) -> None:
        self.b = GraphBuilder()
        self._offset = 0

    def _prov(self, quote: str) -> Provenance:
        self._offset += len(quote) + 1
        return rb(quote, self._offset)

    def node(
        self,
        kind: NodeKind,
        name: str,
        prov: Provenance,
        *,
        hold: bool = False,
        **attrs: AttrValue | None,
    ) -> str:
        nid = node_id(kind, name)
        self.b.add_node(
            Node(
                id=nid,
                kind=kind,
                name=name,
                attrs={k: Attr(value=v, source=prov.source) for k, v in attrs.items() if v is not None},
                prov=(prov,),
                verification=Verification.HOLD if hold else Verification.VERIFIED,
            )
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
        self.b.add_edge(
            Edge(
                id=eid,
                kind=kind,
                src=src,
                dst=dst,
                attrs={k: Attr(value=v, source=p.source) for k, v in attrs.items() if v is not None},
                prov=(p,),
                verification=Verification.HOLD if hold else Verification.VERIFIED,
            )
        )
        return eid

    def service(self, name: str, tier: int | None = 1, *, location: str | None = None) -> str:
        sid = self.node(NodeKind.SERVICE, name, self._prov(name), tier=tier)
        if location:
            self.edge(EdgeKind.RUNS_IN, sid, location, prov=CATALOG)
        return sid

    def location(self, name: str) -> str:
        return self.node(NodeKind.LOCATION, name, CATALOG, location_kind="datacenter")

    def person(
        self,
        name: str,
        status: PersonStatus = PersonStatus.ACTIVE,
        *,
        title: str | None = None,
        hold: bool = False,
    ) -> str:
        return self.node(NodeKind.PERSON, name, ROSTER, hold=hold, status=status.value, title=title)

    def stranger(self, name: str) -> str:
        """A person mentioned only in the runbook (no roster status → UNKNOWN)."""
        return self.node(NodeKind.PERSON, name, self._prov(name))

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

    def channel(self, name: str, medium: ChannelMedium | None, *, located: str | None = None) -> str:
        cid = self.node(NodeKind.CHANNEL, name, ROSTER, medium=medium.value if medium else None)
        if located:
            self.edge(EdgeKind.RUNS_IN, cid, located, prov=CATALOG)
        return cid

    def reach(self, who: str, *channels: str) -> None:
        for c in channels:
            self.edge(EdgeKind.REACHABLE_VIA, who, c, prov=ROSTER)

    def trigger(self, sid: str, name: str, *, hold: bool = False, edge_hold: bool = False) -> str:
        tid = self.node(NodeKind.TRIGGER, name, self._prov(name), hold=hold)
        self.edge(EdgeKind.DETECTED_BY, sid, tid, hold=edge_hold)
        return tid

    def action(
        self,
        sid: str,
        name: str,
        kind: ActionType | None = ActionType.ROLLBACK,
        *,
        requires_approval: bool | None = None,
    ) -> str:
        aid = self.node(
            NodeKind.ACTION,
            name,
            self._prov(name),
            action_type=kind.value if kind else None,
            requires_approval=requires_approval,
        )
        self.edge(EdgeKind.MITIGATED_BY, sid, aid)
        return aid

    def build(self) -> OpsGraph:
        return self.b.build()

    def ctx(self, **kwargs: object) -> CheckContext:
        return CheckContext(graph=self.build(), **kwargs)  # type: ignore[arg-type]


def ev(net: Net, sid: str, element: Element, **ctx_kwargs: object) -> ElementEval:
    return evaluate_service(net.ctx(**ctx_kwargs), sid)[element]


# --------------------------------------------------------------------------- detection


def test_detection_without_trigger_fails() -> None:
    net = Net()
    sid = net.service("결제")
    e = ev(net, sid, Element.DETECTION)
    assert (e.score, e.status) == (0.0, ElementStatus.FAIL)
    assert "감지 신호" in e.reasons[0]


def test_detection_trigger_without_notify_is_partial() -> None:
    net = Net()
    sid = net.service("결제")
    net.trigger(sid, "결제 실패율 알람")
    e = ev(net, sid, Element.DETECTION)
    assert (e.score, e.status) == (0.5, ElementStatus.PARTIAL)


def test_detection_notifying_departed_person_is_partial() -> None:
    net = Net()
    sid = net.service("결제")
    kim = net.person("김대리", PersonStatus.DEPARTED)
    tid = net.trigger(sid, "결제 실패율 알람")
    net.edge(EdgeKind.NOTIFIES, tid, kim)
    assert ev(net, sid, Element.DETECTION).score == 0.5


def test_detection_notifying_role_with_active_holder_passes() -> None:
    net = Net()
    sid = net.service("결제")
    park = net.person("박과장")
    oncall = net.role("1차 당직", park)
    tid = net.trigger(sid, "결제 실패율 알람")
    nid = net.edge(EdgeKind.NOTIFIES, tid, oncall)
    e = ev(net, sid, Element.DETECTION)
    assert (e.score, e.status) == (1.0, ElementStatus.PASS)
    assert {tid, nid, oncall} <= set(e.subject_ids)
    assert e.evidence
    assert all(s.doc_id == DOC for s in e.evidence)
    assert "'박과장'" in e.reasons[0]


def test_detection_notify_over_unavailable_channel_does_not_deliver() -> None:
    net = Net()
    sid = net.service("결제")
    park = net.person("박과장")
    slack = net.channel("Slack", ChannelMedium.SLACK)
    tid = net.trigger(sid, "알람")
    net.edge(EdgeKind.NOTIFIES, tid, park, channel=slack)
    assert ev(net, sid, Element.DETECTION).score == 1.0
    assert ev(net, sid, Element.DETECTION, unavailable=frozenset({slack})).score == 0.5


@pytest.mark.parametrize("where", ["trigger", "detect_edge", "notify_edge"])
def test_detection_with_any_held_item_is_unknown(where: str) -> None:
    net = Net()
    sid = net.service("결제")
    park = net.person("박과장")
    ok = net.trigger(sid, "확실한 알람")
    net.edge(EdgeKind.NOTIFIES, ok, park)
    held = net.trigger(sid, "보류 알람", hold=where == "trigger", edge_hold=where == "detect_edge")
    net.edge(EdgeKind.NOTIFIES, held, park, hold=where == "notify_edge")
    e = ev(net, sid, Element.DETECTION)
    assert (e.score, e.status) == (0.0, ElementStatus.UNKNOWN)
    assert "판단 보류" in e.reasons[0]


def test_detection_ignores_unavailable_trigger() -> None:
    net = Net()
    sid = net.service("결제")
    tid = net.trigger(sid, "알람")
    assert ev(net, sid, Element.DETECTION, unavailable=frozenset({tid})).score == 0.0


# --------------------------------------------------------------------------- owner


def _owner_net(
    owner_status: PersonStatus = PersonStatus.ACTIVE, *, channel: bool = True
) -> tuple[Net, str, str]:
    net = Net()
    sid = net.service("결제")
    owner = net.person("김대리", owner_status, title="대리")
    if channel:
        net.reach(owner, net.channel("김대리 휴대폰", ChannelMedium.PHONE))
    net.edge(EdgeKind.OWNED_BY, sid, owner)
    return net, sid, owner


def test_owner_missing_fails() -> None:
    net = Net()
    sid = net.service("결제")
    e = ev(net, sid, Element.OWNER)
    assert (e.score, e.status) == (0.0, ElementStatus.FAIL)
    assert "지정되어 있지 않음" in e.reasons[0]


def test_owner_reachable_without_escalation_is_partial() -> None:
    net, sid, _ = _owner_net()
    e = ev(net, sid, Element.OWNER)
    assert (e.score, e.status) == (0.6, ElementStatus.PARTIAL)


def test_owner_without_channel_gets_half_owner_part() -> None:
    net, sid, _ = _owner_net(channel=False)
    assert ev(net, sid, Element.OWNER).score == 0.3


def test_owner_with_timed_escalation_passes() -> None:
    net, sid, owner = _owner_net()
    lead = net.person("박팀장", title="팀장")
    eid = net.edge(EdgeKind.ESCALATES_TO, owner, lead, after_minutes=15)
    e = ev(net, sid, Element.OWNER)
    assert (e.score, e.status) == (1.0, ElementStatus.PASS)
    assert eid in e.subject_ids
    assert lead in e.subject_ids


def test_owner_with_untimed_escalation() -> None:
    net, sid, owner = _owner_net()
    net.edge(EdgeKind.ESCALATES_TO, owner, net.person("박팀장"))
    e = ev(net, sid, Element.OWNER)
    assert (e.score, e.status) == (0.9, ElementStatus.PARTIAL)
    assert "대기 시간" in e.reasons[1]


def test_departed_owner_keeps_only_escalation_part() -> None:
    net, sid, owner = _owner_net(PersonStatus.DEPARTED)
    net.edge(EdgeKind.ESCALATES_TO, owner, net.person("박팀장"), after_minutes=30)
    e = ev(net, sid, Element.OWNER)
    assert (e.score, e.status) == (0.4, ElementStatus.FAIL)
    assert "퇴사" in e.reasons[0]


def test_on_leave_owner_fails() -> None:
    net, sid, _ = _owner_net(PersonStatus.ON_LEAVE)
    e = ev(net, sid, Element.OWNER)
    assert (e.score, e.status) == (0.0, ElementStatus.FAIL)
    assert "장기 부재" in e.reasons[0]


def test_unconfirmed_owner_is_unknown_even_with_escalation() -> None:
    net = Net()
    sid = net.service("결제")
    ghost = net.stranger("최주임")
    net.edge(EdgeKind.OWNED_BY, sid, ghost)
    net.edge(EdgeKind.ESCALATES_TO, ghost, net.person("박팀장"), after_minutes=10)
    e = ev(net, sid, Element.OWNER)
    assert (e.score, e.status) == (0.0, ElementStatus.UNKNOWN)
    assert "명부에서 확인되지 않아" in e.reasons[0]


def test_escalation_to_departed_only_is_a_dead_end() -> None:
    net, sid, owner = _owner_net()
    net.edge(EdgeKind.ESCALATES_TO, owner, net.person("이부장", PersonStatus.DEPARTED), after_minutes=5)
    e = ev(net, sid, Element.OWNER)
    assert e.score == 0.6
    assert "막다른 길" in e.reasons[1]


def test_escalation_back_to_the_same_person_through_a_role_earns_nothing() -> None:
    net, sid, owner = _owner_net()
    self_role = net.role("결제 담당", owner)
    net.edge(EdgeKind.ESCALATES_TO, owner, self_role, after_minutes=5)
    assert ev(net, sid, Element.OWNER).score == 0.6


def test_team_owner_escalating_to_its_own_lead_counts() -> None:
    net = Net()
    sid = net.service("결제")
    member = net.person("김대리")
    lead = net.person("박팀장", title="팀장")
    team = net.team("결제팀", member, lead)
    net.reach(member, net.channel("결제팀 Slack", ChannelMedium.SLACK))
    net.edge(EdgeKind.OWNED_BY, sid, team)
    net.edge(EdgeKind.ESCALATES_TO, team, net.role("팀장", lead), after_minutes=20)
    assert ev(net, sid, Element.OWNER).score == 1.0


def test_multi_hop_escalation_timing_is_checked_on_every_hop() -> None:
    net, sid, owner = _owner_net()
    empty = net.role("결제 리드")
    cto = net.role("CTO", net.person("정CTO"))
    net.edge(EdgeKind.ESCALATES_TO, owner, empty, after_minutes=10)
    net.edge(EdgeKind.ESCALATES_TO, empty, cto, after_minutes=20)
    assert ev(net, sid, Element.OWNER).score == 1.0

    net2, sid2, owner2 = _owner_net()
    empty2 = net2.role("결제 리드")
    cto2 = net2.role("CTO", net2.person("정CTO"))
    net2.edge(EdgeKind.ESCALATES_TO, owner2, empty2, after_minutes=10)
    net2.edge(EdgeKind.ESCALATES_TO, empty2, cto2)
    assert ev(net2, sid2, Element.OWNER).score == 0.9


def test_owner_reachable_through_role_channel() -> None:
    net = Net()
    sid = net.service("결제")
    role = net.role("1차 당직", net.person("박과장"))
    net.reach(role, net.channel("당직 호출기", ChannelMedium.PAGER))
    net.edge(EdgeKind.OWNED_BY, sid, role)
    assert ev(net, sid, Element.OWNER).score == 0.6


def test_held_owner_edge_is_not_evidence() -> None:
    net = Net()
    sid = net.service("결제")
    net.edge(EdgeKind.OWNED_BY, sid, net.person("박과장"), hold=True)
    e = ev(net, sid, Element.OWNER)
    assert (e.score, e.status) == (0.0, ElementStatus.FAIL)
    assert any("판단 보류" in r for r in e.reasons)


def test_unavailable_owner_what_if() -> None:
    net, sid, owner = _owner_net()
    assert ev(net, sid, Element.OWNER, unavailable=frozenset({owner})).status is ElementStatus.FAIL


@pytest.mark.parametrize(("hops", "reached"), [(MAX_ESCALATION_HOPS, True), (MAX_ESCALATION_HOPS + 1, False)])
def test_escalation_reach_is_bounded_by_max_hops(hops: int, reached: bool) -> None:
    """Same bound as ``CheckContext.escalation_closure``: a lead 13 hops away is not a plan."""
    net, sid, owner = _owner_net()
    chain = [owner, *(net.person(f"퇴사자{i}", PersonStatus.DEPARTED) for i in range(hops - 1))]
    lead = net.person("박팀장")
    for src, dst in zip(chain, [*chain[1:], lead], strict=True):
        net.edge(EdgeKind.ESCALATES_TO, src, dst, after_minutes=5)
    assert (analyze_escalation(net.ctx(), [owner]).responder_ids == (lead,)) is reached
    assert ev(net, sid, Element.OWNER).score == (1.0 if reached else 0.6)


def test_active_responders_of_non_people_and_held_nodes() -> None:
    net = Net()
    pager = net.channel("호출기", ChannelMedium.PAGER)
    held = net.person("박과장", hold=True)
    ctx = net.ctx()
    assert active_responders(ctx, "person:없음") == []
    assert active_responders(ctx, pager) == []
    assert active_responders(ctx, held) == []


def test_escalation_analysis_reports_dead_ends_and_untimed_edges() -> None:
    net, _, owner = _owner_net()
    gone = net.person("이부장", PersonStatus.DEPARTED)
    e1 = net.edge(EdgeKind.ESCALATES_TO, owner, gone)
    esc = analyze_escalation(net.ctx(), [owner])
    assert esc.edge_ids == (e1,)
    assert esc.dead_end_ids == (gone,)
    assert esc.untimed_edge_ids == (e1,)
    assert esc.first_hop_ids == (gone,)
    assert not esc.reaches_responder


# --------------------------------------------------------------------------- approver


def _approver_net() -> tuple[Net, str]:
    net = Net()
    sid = net.service("결제")
    return net, sid


def _ready(net: Net, name: str, status: PersonStatus = PersonStatus.ACTIVE) -> str:
    pid = net.person(name, status)
    net.reach(pid, net.channel(f"{name} 휴대폰", ChannelMedium.PHONE))
    return pid


def test_approver_without_actions_fails_with_reason() -> None:
    net, sid = _approver_net()
    e = ev(net, sid, Element.APPROVER)
    assert (e.score, e.status) == (0.0, ElementStatus.FAIL)
    assert e.reasons[0] == "조치가 없어 승인 경로를 평가할 수 없음"


def test_approver_not_needed_for_plain_restart() -> None:
    net, sid = _approver_net()
    net.action(sid, "파드 재시작", ActionType.RESTART)
    assert ev(net, sid, Element.APPROVER).score == 1.0


def test_rollback_without_approver_fails() -> None:
    net, sid = _approver_net()
    net.action(sid, "이전 버전 롤백")
    e = ev(net, sid, Element.APPROVER)
    assert (e.score, e.status) == (0.0, ElementStatus.FAIL)
    assert "승인자가 적혀 있지 않음" in e.reasons[0]


def test_single_and_double_approvers() -> None:
    net, sid = _approver_net()
    aid = net.action(sid, "이전 버전 롤백")
    net.edge(EdgeKind.APPROVED_BY, aid, _ready(net, "박팀장"))
    assert ev(net, sid, Element.APPROVER).score == 0.8
    net.edge(EdgeKind.APPROVED_BY, aid, _ready(net, "정CTO"))
    e = ev(net, sid, Element.APPROVER)
    assert (e.score, e.status) == (1.0, ElementStatus.PASS)


def test_role_with_two_reachable_holders_counts_twice() -> None:
    net, sid = _approver_net()
    aid = net.action(sid, "DB 페일오버", ActionType.FAILOVER)
    net.edge(EdgeKind.APPROVED_BY, aid, net.role("팀장", _ready(net, "박팀장"), _ready(net, "이팀장")))
    assert ev(net, sid, Element.APPROVER).score == 1.0


def test_same_person_reached_twice_counts_once() -> None:
    net, sid = _approver_net()
    aid = net.action(sid, "이전 버전 롤백")
    park = _ready(net, "박팀장")
    net.edge(EdgeKind.APPROVED_BY, aid, park)
    net.edge(EdgeKind.APPROVED_BY, aid, net.role("팀장", park))
    assert ev(net, sid, Element.APPROVER).score == 0.8


@pytest.mark.parametrize("reachable", [True, False])
def test_approver_named_but_not_ready(reachable: bool) -> None:
    net, sid = _approver_net()
    aid = net.action(sid, "이전 버전 롤백")
    approver = (
        _ready(net, "박팀장", PersonStatus.DEPARTED)
        if reachable
        else net.person("박팀장")  # active, no channel
    )
    net.edge(EdgeKind.APPROVED_BY, aid, approver)
    e = ev(net, sid, Element.APPROVER)
    assert (e.score, e.status) == (0.2, ElementStatus.FAIL)


def test_explicit_no_approval_flag_wins_over_rollback_default() -> None:
    net, sid = _approver_net()
    net.action(sid, "기능 플래그 롤백", ActionType.ROLLBACK, requires_approval=False)
    assert ev(net, sid, Element.APPROVER).score == 1.0


def test_restart_requiring_approval_without_approver() -> None:
    net, sid = _approver_net()
    net.action(sid, "전체 재시작", ActionType.RESTART, requires_approval=True)
    assert ev(net, sid, Element.APPROVER).score == 0.0


def test_approver_element_is_min_over_actions() -> None:
    net, sid = _approver_net()
    ok = net.action(sid, "이전 버전 롤백")
    net.edge(EdgeKind.APPROVED_BY, ok, _ready(net, "박팀장"))
    net.edge(EdgeKind.APPROVED_BY, ok, _ready(net, "정CTO"))
    net.action(sid, "DB 페일오버", ActionType.FAILOVER)
    e = ev(net, sid, Element.APPROVER)
    assert (e.score, e.status) == (0.0, ElementStatus.FAIL)
    assert len(e.reasons) == 2


def test_unconfirmed_approver_is_unknown() -> None:
    net, sid = _approver_net()
    aid = net.action(sid, "이전 버전 롤백")
    net.edge(EdgeKind.APPROVED_BY, aid, net.stranger("최주임"))
    e = ev(net, sid, Element.APPROVER)
    assert (e.score, e.status) == (0.0, ElementStatus.UNKNOWN)


def test_named_approver_alone_does_not_gate_a_restart() -> None:
    """Single rule (CheckContext.requires_approval): an approver edge says who, never whether."""
    net, sid = _approver_net()
    restart = net.action(sid, "재시작", ActionType.RESTART)
    net.edge(EdgeKind.APPROVED_BY, restart, net.person("김대리", PersonStatus.DEPARTED))
    ctx = net.ctx()
    action = ctx.graph.nodes[restart]
    assert approval_gate(ctx, action) is None
    assert not needs_approval(action)
    assert not ctx.requires_approval(action)
    ungated = evaluate_service(ctx, sid)[Element.APPROVER]
    assert (ungated.score, ungated.status) == (1.0, ElementStatus.PASS)
    # adding a reachable approver on top can never lower the score (monotonicity)
    net.edge(EdgeKind.APPROVED_BY, restart, _ready(net, "박팀장"))
    assert ev(net, sid, Element.APPROVER).score == 1.0


def test_restart_the_runbook_gates_is_judged_like_a_rollback() -> None:
    """'박팀장 승인 후 재시작' carries requires_approval=True: one ready approver is 0.8, not 1.0."""
    net, sid = _approver_net()
    restart = net.action(sid, "재시작", ActionType.RESTART, requires_approval=True)
    net.edge(EdgeKind.APPROVED_BY, restart, _ready(net, "박팀장"))
    e = ev(net, sid, Element.APPROVER)
    assert (e.score, e.status) == (0.8, ElementStatus.PARTIAL)
    assert "1명뿐" in e.reasons[0]


def test_explicit_no_approval_flag_wins_over_a_named_approver() -> None:
    net, sid = _approver_net()
    restart = net.action(sid, "재시작", ActionType.RESTART, requires_approval=False)
    net.edge(EdgeKind.APPROVED_BY, restart, net.person("김대리", PersonStatus.DEPARTED))
    ctx = net.ctx()
    assert approval_gate(ctx, ctx.graph.nodes[restart]) is None
    assert ev(net, sid, Element.APPROVER).score == 1.0


def test_approval_gate_kinds() -> None:
    net, sid = _approver_net()
    stated = net.action(sid, "재시작", ActionType.RESTART, requires_approval=True)
    default = net.action(sid, "롤백")
    named = net.action(sid, "확장", ActionType.SCALE)
    net.edge(EdgeKind.APPROVED_BY, named, _ready(net, "박팀장"))
    held = net.action(sid, "캐시 비우기", ActionType.OTHER)
    net.edge(EdgeKind.APPROVED_BY, held, _ready(net, "정CTO"), hold=True)
    ctx = net.ctx()
    nodes = ctx.graph.nodes
    assert approval_gate(ctx, nodes[stated]) is ApprovalGate.STATED
    assert approval_gate(ctx, nodes[default]) is ApprovalGate.DEFAULT
    assert approval_gate(ctx, nodes[named]) is None  # naming an approver creates no requirement
    assert approval_gate(ctx, nodes[held]) is None


def test_not_needed_reason_names_the_single_rule() -> None:
    net, sid = _approver_net()
    restart = net.action(sid, "재시작", ActionType.RESTART)
    net.edge(EdgeKind.APPROVED_BY, restart, _ready(net, "박팀장"))
    e = ev(net, sid, Element.APPROVER)
    assert e.reasons[0].startswith("승인이 필요한 조치가 없음")
    assert "승인 필요로 적힌 조치" in e.reasons[0]


def test_unconfirmed_approver_on_a_gated_restart_is_unknown() -> None:
    net, sid = _approver_net()
    restart = net.action(sid, "재시작", ActionType.RESTART, requires_approval=True)
    net.edge(EdgeKind.APPROVED_BY, restart, net.stranger("최주임"))
    e = ev(net, sid, Element.APPROVER)
    assert (e.score, e.status) == (0.0, ElementStatus.UNKNOWN)


@pytest.mark.parametrize(("kind", "flag"), [(ActionType.ROLLBACK, None), (ActionType.RESTART, True)])
def test_approver_switched_off_by_a_what_if_is_named_but_not_ready(
    kind: ActionType, flag: bool | None
) -> None:
    """Knocking the only approver out must fail the requirement, not make it disappear."""
    net, sid = _approver_net()
    aid = net.action(sid, "조치", kind, requires_approval=flag)
    lead = _ready(net, "박팀장")
    net.edge(EdgeKind.APPROVED_BY, aid, lead)
    assert ev(net, sid, Element.APPROVER).score == 0.8
    e = ev(net, sid, Element.APPROVER, unavailable=frozenset({lead}))
    assert (e.score, e.status) == (0.2, ElementStatus.FAIL)
    assert lead in e.subject_ids
    assert "'박팀장' 중 활동 중이고 연락 가능한 사람이 없음" in e.reasons[0]


_FLAGS = st.sampled_from([True, False, None])
_KINDS = st.sampled_from([*ActionType, None])


@settings(deadline=None)  # the first example imports the simulator
@given(_KINDS, _FLAGS, st.lists(st.sampled_from(list(PersonStatus)), max_size=3))
def test_one_approval_rule_for_score_spof_and_rehearsal(
    kind: ActionType | None, flag: bool | None, approver_statuses: list[PersonStatus]
) -> None:
    """Score, JT-PATH-020, SPOF and the replay agree on which steps wait for an approver, and the
    answer depends on the action alone (explicit flag, else rollback/failover) — never on approvers."""
    simulator = pytest.importorskip("jettae.rehearse.simulator")
    net, sid = _approver_net()
    aid = net.action(sid, "조치", kind, requires_approval=flag)
    for i, status in enumerate(approver_statuses):
        net.edge(EdgeKind.APPROVED_BY, aid, net.person(f"승인자{i}", status))
    ctx = net.ctx()
    action = ctx.graph.nodes[aid]
    expected = flag if flag is not None else kind in (ActionType.ROLLBACK, ActionType.FAILOVER)
    assert ctx.requires_approval(action) is expected
    assert needs_approval(action) is expected
    assert (approval_gate(ctx, action) is not None) is expected
    assert simulator.requires_approval(ctx, action) is expected
    gated = [a.id for _, a in gated_actions(GraphIndex(ctx.graph), sid)]
    assert (aid in gated) is expected


# --------------------------------------------------------------------------- fallback channel


def _channel_net() -> tuple[Net, str, str, str]:
    net = Net()
    idc = net.location("판교 IDC")
    sid = net.service("결제", location=idc)
    owner = net.person("박과장")
    net.edge(EdgeKind.OWNED_BY, sid, owner)
    return net, sid, owner, idc


def test_fallback_without_active_owner_is_zero() -> None:
    net = Net()
    sid = net.service("결제")
    owner = net.person("김대리", PersonStatus.DEPARTED)
    net.reach(owner, net.channel("휴대폰", ChannelMedium.PHONE))
    net.edge(EdgeKind.OWNED_BY, sid, owner)
    e = ev(net, sid, Element.FALLBACK_CHANNEL)
    assert (e.score, e.status) == (0.0, ElementStatus.FAIL)


def test_fallback_channel_counts() -> None:
    net, sid, owner, _ = _channel_net()
    assert ev(net, sid, Element.FALLBACK_CHANNEL).score == 0.0
    net.reach(owner, net.channel("Slack", ChannelMedium.SLACK))
    e = ev(net, sid, Element.FALLBACK_CHANNEL)
    assert (e.score, e.status) == (0.25, ElementStatus.FAIL)
    net.reach(owner, net.channel("Slack 비상", ChannelMedium.SLACK))
    e = ev(net, sid, Element.FALLBACK_CHANNEL)
    assert (e.score, e.status) == (0.5, ElementStatus.PARTIAL)
    assert "같은 매체" in e.reasons[0]


def test_fallback_all_channels_share_failure_domain() -> None:
    net, sid, owner, idc = _channel_net()
    net.reach(owner, net.channel("사내 메신저", ChannelMedium.INTERNAL_MESSENGER, located=idc))
    net.reach(owner, net.channel("사내 메일", ChannelMedium.EMAIL, located=idc))
    e = ev(net, sid, Element.FALLBACK_CHANNEL)
    assert e.score == 0.5
    assert "장애 영역" in e.reasons[0]


def test_fallback_with_independent_distinct_medium_passes() -> None:
    net, sid, owner, idc = _channel_net()
    net.reach(owner, net.channel("사내 메신저", ChannelMedium.INTERNAL_MESSENGER, located=idc))
    phone = net.channel("휴대폰", ChannelMedium.PHONE)
    net.reach(owner, phone)
    e = ev(net, sid, Element.FALLBACK_CHANNEL)
    assert (e.score, e.status) == (1.0, ElementStatus.PASS)
    assert analyze_channels(net.ctx(), sid).independent_ids == (phone,)


def test_channel_depending_on_shared_system_shares_domain() -> None:
    net, sid, owner, _ = _channel_net()
    mq = net.node(NodeKind.SYSTEM, "공용 MQ", CATALOG)
    net.edge(EdgeKind.DEPENDS_ON, sid, mq, prov=CATALOG)
    alert = net.channel("알림봇", ChannelMedium.OTHER)
    net.edge(EdgeKind.DEPENDS_ON, alert, mq, prov=CATALOG)
    net.reach(owner, alert)
    net.reach(owner, net.channel("Slack", ChannelMedium.SLACK))
    ch = analyze_channels(net.ctx(), sid)
    assert alert not in ch.independent_ids


def test_fallback_includes_first_escalation_hop_only() -> None:
    net, sid, owner, idc = _channel_net()
    net.reach(owner, net.channel("사내 메신저", ChannelMedium.INTERNAL_MESSENGER, located=idc))
    lead = net.person("박팀장")
    net.edge(EdgeKind.ESCALATES_TO, owner, lead, after_minutes=10)
    cto = net.person("정CTO")
    net.edge(EdgeKind.ESCALATES_TO, lead, cto, after_minutes=10)
    net.reach(cto, net.channel("CTO 휴대폰", ChannelMedium.PHONE))
    assert ev(net, sid, Element.FALLBACK_CHANNEL).score == 0.25
    net.reach(lead, net.channel("팀장 휴대폰", ChannelMedium.PHONE))
    assert ev(net, sid, Element.FALLBACK_CHANNEL).score == 1.0


def test_fallback_to_closure_and_unavailable_channels() -> None:
    net, sid, owner, idc = _channel_net()
    messenger = net.channel("사내 메신저", ChannelMedium.INTERNAL_MESSENGER, located=idc)
    kakao = net.channel("카카오톡 단체방", ChannelMedium.KAKAOTALK)
    net.reach(owner, messenger)
    net.edge(EdgeKind.FALLBACK_TO, messenger, kakao)
    assert ev(net, sid, Element.FALLBACK_CHANNEL).score == 1.0
    assert ev(net, sid, Element.FALLBACK_CHANNEL, unavailable=frozenset({kakao})).score == 0.25


def test_channels_without_medium_do_not_prove_diversity() -> None:
    net, sid, owner, _ = _channel_net()
    net.reach(owner, net.channel("연락처 A", None), net.channel("연락처 B", None))
    assert ev(net, sid, Element.FALLBACK_CHANNEL).score == 0.5


# --------------------------------------------------------------------------- rollback


def test_rollback_scores() -> None:
    net = Net()
    sid = net.service("결제")
    assert ev(net, sid, Element.ROLLBACK).score == 0.0
    aid = net.action(sid, "이전 버전 롤백")
    e = ev(net, sid, Element.ROLLBACK)
    assert (e.score, e.status) == (0.6, ElementStatus.PARTIAL)
    assert "실행자" in e.reasons[0]
    net.edge(EdgeKind.EXECUTED_BY, aid, net.person("박과장"))
    e = ev(net, sid, Element.ROLLBACK)
    assert (e.score, e.status) == (1.0, ElementStatus.PASS)


def test_rollback_with_departed_executor_gets_no_bonus() -> None:
    net = Net()
    sid = net.service("결제")
    aid = net.action(sid, "이전 버전 롤백")
    net.edge(EdgeKind.EXECUTED_BY, aid, net.person("김대리", PersonStatus.DEPARTED))
    assert ev(net, sid, Element.ROLLBACK).score == 0.6


@pytest.mark.parametrize(
    ("kind", "with_executor", "expected"),
    [
        (ActionType.RESTART, False, 0.3),
        (ActionType.RESTART, True, 0.5),
        (ActionType.SCALE, True, 0.5),
        (None, False, 0.3),
        (ActionType.FEATURE_FLAG, False, 0.6),
        (ActionType.FAILOVER, True, 1.0),
    ],
)
def test_mitigation_types(kind: ActionType | None, with_executor: bool, expected: float) -> None:
    net = Net()
    sid = net.service("결제")
    aid = net.action(sid, "조치", kind)
    if with_executor:
        net.edge(EdgeKind.EXECUTED_BY, aid, net.team("결제팀", net.person("박과장")))
    assert ev(net, sid, Element.ROLLBACK).score == expected


def test_rollback_takes_best_action() -> None:
    net = Net()
    sid = net.service("결제")
    restart = net.action(sid, "재시작", ActionType.RESTART)
    net.edge(EdgeKind.EXECUTED_BY, restart, net.person("박과장"))
    net.action(sid, "롤백", ActionType.ROLLBACK)
    assert ev(net, sid, Element.ROLLBACK).score == 0.6


# --------------------------------------------------------------------------- helpers & invariants


def test_evaluate_service_returns_all_elements_in_order() -> None:
    net = Net()
    sid = net.service("결제")
    result = evaluate_service(net.ctx(), sid)
    assert list(result) == list(Element)
    assert all(v.element is k for k, v in result.items())


def test_evaluation_is_independent_of_insertion_order() -> None:
    net = Net()
    sid = net.service("결제")
    owner = net.person("박과장")
    net.edge(EdgeKind.OWNED_BY, sid, owner)
    net.reach(owner, net.channel("Slack", ChannelMedium.SLACK), net.channel("휴대폰", ChannelMedium.PHONE))
    net.edge(EdgeKind.ESCALATES_TO, owner, net.person("박팀장"), after_minutes=5)
    graph = net.build()
    rebuilt = GraphBuilder()
    for n in reversed(list(graph.nodes.values())):
        rebuilt.add_node(n)
    for e in reversed(list(graph.edges.values())):
        rebuilt.add_edge(e)
    other = rebuilt.build()
    assert other.content_hash() == graph.content_hash()
    assert evaluate_service(CheckContext(graph=graph), sid) == evaluate_service(
        CheckContext(graph=other), sid
    )


@pytest.mark.parametrize(
    ("word", "pair", "expected"),
    [
        ("'결제'", ("이", "가"), "'결제'가"),
        ("'박팀장'", ("이", "가"), "'박팀장'이"),
        ("'김대리'", ("은", "는"), "'김대리'는"),
        ("'알람'", ("을", "를"), "'알람'을"),
        ("'서울'", ("으로", "로"), "'서울'로"),
        ("'부산'", ("으로", "로"), "'부산'으로"),
        ("'Slack'", ("이", "가"), "'Slack'이(가)"),
        ("Slack", ("으로", "로"), "Slack(으)로"),
        ("", ("이", "가"), "이(가)"),
    ],
)
def test_josa(word: str, pair: tuple[str, str], expected: str) -> None:
    assert josa(word, *pair) == expected


@given(st.integers(min_value=0xAC00, max_value=0xD7A3))
def test_josa_agrees_with_final_consonant(code: int) -> None:
    syllable = chr(code)
    has_final = (code - 0xAC00) % 28 != 0
    assert josa(syllable, "이", "가") == syllable + ("이" if has_final else "가")


def test_attribute_parsers() -> None:
    net = Net()
    sid = net.service("결제")
    weird = net.action(sid, "이상한 조치", None)
    graph = net.build()
    action = graph.nodes[weird]
    assert action_type(action) is ActionType.OTHER
    assert not needs_approval(action)
    bogus = action.model_copy(update={"attrs": {"action_type": Attr(value="teleport")}})
    assert action_type(bogus) is ActionType.OTHER
    flagged = action.model_copy(update={"attrs": {"requires_approval": Attr(value=True)}})
    assert needs_approval(flagged)
    channel = Node(id="channel:x", kind=NodeKind.CHANNEL, name="x", prov=(ROSTER,))
    assert channel_medium(channel) is None
    assert channel_medium(channel.model_copy(update={"attrs": {"medium": Attr(value="fax")}})) is None
    assert channel_medium(channel.model_copy(update={"attrs": {"medium": Attr(value="sms")}})) is (
        ChannelMedium.SMS
    )


def test_spans_of_is_sorted_and_deduplicated() -> None:
    net = Net()
    sid = net.service("결제")
    tid = net.trigger(sid, "알람")
    spans = spans_of(net.ctx(), [tid, sid, tid, "missing"])
    assert [s.start for s in spans] == sorted(s.start for s in spans)
    assert len(spans) == len(set(spans)) == 2


def test_reachable_channels_via_role() -> None:
    net = Net()
    park = net.person("박과장")
    role = net.role("당직", park)
    pager = net.channel("호출기", ChannelMedium.PAGER)
    net.reach(role, pager)
    ctx = net.ctx()
    assert reachable_channels(ctx, park) == []
    assert [c.id for c in reachable_channels(ctx, park, role)] == [pager]
    assert reachable_channels(ctx, "person:없음") == []


def test_suggest_candidates_ranks_leads_then_terminal_roles_then_teammates() -> None:
    net = Net()
    kim = net.person("김대리", PersonStatus.DEPARTED, title="대리")
    lead = net.person("박팀장", title="팀장")
    mate = net.person("이사원", title="사원")
    gone = net.person("최과장", PersonStatus.DEPARTED, title="과장")
    net.team("결제팀", kim, lead, mate, gone)
    cto = net.person("정CTO")
    net.role("CTO", cto)
    net.person("외부인", title="팀장")  # other team's lead: not suggested
    net.reach(mate, net.channel("휴대폰", ChannelMedium.PHONE))
    ctx = net.ctx()
    assert suggest_candidates(ctx, [kim]) == [lead, cto, mate]
    assert suggest_candidates(ctx, [kim], limit=1) == [lead]
    assert suggest_candidates(ctx, [kim], exclude=[lead]) == [cto, mate]


def test_suggest_candidates_uses_roster_team_when_graph_has_no_membership() -> None:
    from jettae.domain import RosterEntry

    net = Net()
    kim = net.person("김대리", PersonStatus.DEPARTED)
    lead = net.person("박팀장")
    roster = (
        RosterEntry(name="김대리", status=PersonStatus.DEPARTED, team="결제팀"),
        RosterEntry(name="박팀장", team="결제팀", title="팀장"),
    )
    ctx = CheckContext(graph=net.build(), roster=roster)
    assert suggest_candidates(ctx, [kim]) == [lead]


def test_is_terminal_matches_policy_roles_case_insensitively() -> None:
    net = Net()
    cto_role = net.role("cto")
    ceo = net.person("대표님", title="CEO")
    plain = net.person("박과장", title="과장")
    team = net.team("CTO", ceo)
    ctx = net.ctx(policy=Policy())
    nodes = ctx.graph.nodes
    assert is_terminal(ctx, nodes[cto_role])
    assert is_terminal(ctx, nodes[ceo])
    assert not is_terminal(ctx, nodes[plain])
    assert not is_terminal(ctx, nodes[team])  # a team is a pool, never a terminal


# --------------------------------------------------------------------------- review: adversarial cases


def _held_everything_net() -> tuple[Net, str, str, str]:
    """A service whose owner link, role membership and contact link are all on HOLD."""
    net = Net()
    sid = net.service("결제")
    park = net.person("박과장")
    role = net.node(NodeKind.ROLE, "1차 당직", ROSTER)
    net.edge(EdgeKind.HOLDS_ROLE, park, role, prov=ROSTER, hold=True)
    owned = net.edge(EdgeKind.OWNED_BY, sid, role, hold=True)
    phone = net.channel("휴대폰", ChannelMedium.PHONE)
    net.edge(EdgeKind.REACHABLE_VIA, park, phone, prov=ROSTER, hold=True)
    return net, sid, owned, park


def test_confirmed_view_confirms_even_after_the_adjacency_index_was_built() -> None:
    """Regression: a model copy kept the cached index, so 'confirmed' edges were still on HOLD.

    In ``run_checks`` some earlier rule has always queried the graph already, which silently
    turned every 확인 필요 (UNKNOWN) finding into a settled 끊김 (BROKEN).
    """
    net, sid, owned, park = _held_everything_net()
    ctx = net.ctx()
    for nid in ctx.graph.nodes:  # warm the lazily built adjacency index first
        ctx.graph.out_edges(nid)
        ctx.graph.in_edges(nid)
    alt = confirmed_view(ctx)
    assert alt is not ctx
    walked = [e for nid in alt.graph.nodes for e in alt.graph.out_edges(nid)]
    walked += [e for nid in alt.graph.nodes for e in alt.graph.in_edges(nid)]
    assert walked
    assert all(e.verification is Verification.VERIFIED for e in walked)
    assert all(n.verification is Verification.VERIFIED for n in alt.graph.nodes.values())
    assert alt.graph.docs == ctx.graph.docs
    assert set(alt.graph.nodes) == set(ctx.graph.nodes)
    assert set(alt.graph.edges) == set(ctx.graph.edges)
    # the original context is a value: still on HOLD, still not evidence
    assert ctx.graph.edges[owned].verification is Verification.HOLD
    assert evaluate_service(ctx, sid)[Element.OWNER].score == 0.0
    confirmed = evaluate_service(alt, sid)[Element.OWNER]
    assert confirmed.score == 0.6
    assert park in confirmed.subject_ids


def test_confirmed_view_is_identity_without_held_items() -> None:
    net, _, _ = _owner_net()
    ctx = net.ctx()
    assert confirmed_view(ctx) is ctx


@pytest.mark.parametrize(
    ("word", "pair", "expected"),
    [
        # the particle agrees with the word *before* a parenthetical (person labels carry status)
        ("'박과장'(장기 부재)", ("은", "는"), "'박과장'(장기 부재)은"),
        ("'김대리'(퇴사)", ("이", "가"), "'김대리'(퇴사)가"),
        ("'최과장'(퇴사)", ("이", "가"), "'최과장'(퇴사)이"),
        ("'이서울'(명부 미확인)", ("으로", "로"), "'이서울'(명부 미확인)로"),
        ("'결제(신)'", ("은", "는"), "'결제(신)'는"),
        ("'정산'(a)(b)", ("을", "를"), "'정산'(a)(b)을"),
        ("(퇴사)", ("이", "가"), "(퇴사)가"),
        ("퇴사)", ("이", "가"), "퇴사)가"),
        ("'Slack'(퇴사)", ("이", "가"), "'Slack'(퇴사)이(가)"),
        # digits are read in Sino-Korean: 일·이·삼·사·오·육·칠·팔·구, 0 → 영/십/백/천
        ("'서비스2'", ("이", "가"), "'서비스2'가"),
        ("'알람1'", ("으로", "로"), "'알람1'로"),
        ("'알람6'", ("으로", "로"), "'알람6'으로"),
        ("'3'", ("을", "를"), "'3'을"),
        ("'v10'", ("이", "가"), "'v10'이"),
        ("'9'", ("으로", "로"), "'9'로"),
        ("'CPU 90%'", ("이", "가"), "'CPU 90%'이(가)"),
        ("'결제'」", ("은", "는"), "'결제'」는"),
    ],
)
def test_josa_korean_edge_cases(word: str, pair: tuple[str, str], expected: str) -> None:
    assert josa(word, *pair) == expected


def test_josa_on_decomposed_hangul_keeps_the_original_text() -> None:
    nfd = unicodedata.normalize("NFD", "'박팀장'")
    assert nfd != "'박팀장'"
    assert josa(nfd, "은", "는") == nfd + "은"


_NOT_A_CLOSER = st.characters(blacklist_characters="()'\"’”]」』 ")


@given(st.integers(min_value=0xAC00, max_value=0xD7A3), st.text(alphabet=_NOT_A_CLOSER, max_size=6))
def test_josa_ignores_any_trailing_parenthetical(code: int, inside: str) -> None:
    syllable = chr(code)
    has_final = (code - 0xAC00) % 28 != 0
    word = f"'{syllable}'({inside})"
    assert josa(word, "을", "를") == word + ("을" if has_final else "를")


@pytest.mark.parametrize(
    ("digit", "with_final", "is_rieul"),
    [
        ("0", True, False),
        ("1", True, True),
        ("2", False, False),
        ("3", True, False),
        ("4", False, False),
        ("5", False, False),
        ("6", True, False),
        ("7", True, True),
        ("8", True, True),
        ("9", False, False),
    ],
)
def test_josa_after_each_digit(digit: str, with_final: bool, is_rieul: bool) -> None:
    assert josa(digit, "이", "가") == digit + ("이" if with_final else "가")
    assert josa(digit, "으로", "로") == digit + ("으로" if with_final and not is_rieul else "로")


def test_labels_are_single_line_and_never_empty() -> None:
    net = Net()
    sid = net.node(NodeKind.SERVICE, "결제\n\t 서비스 ", rb("결제 서비스"), tier=1)
    blank = net.node(NodeKind.TRIGGER, " 　 ", rb("알람"))
    ctx = net.ctx()
    assert label(ctx, sid) == "'결제 서비스'"
    assert service_label(ctx, sid) == "'결제 서비스'"  # not "'결제 서비스' 서비스"
    assert label(ctx, blank) == f"'{blank}'"
    assert label(ctx, "service:없음") == "'service:없음'"
    assert person_label(ctx, "person:없음") == "'person:없음'"


def test_bare_service_fails_every_element_with_korean_reasons() -> None:
    net = Net()
    sid = net.service("결제")
    result = evaluate_service(net.ctx(), sid)
    assert [e.score for e in result.values()] == [0.0] * len(Element)
    assert all(e.status is ElementStatus.FAIL for e in result.values())
    for e in result.values():
        assert e.reasons
        assert all(any("가" <= ch <= "힣" for ch in r) for r in e.reasons)
        assert e.subject_ids[0] == sid
        assert e.evidence  # the runbook mention of the service


def test_evaluating_an_unknown_service_id_does_not_crash() -> None:
    ctx = CheckContext(graph=OpsGraph())
    result = evaluate_service(ctx, "service:없음")
    assert all(e.score == 0.0 for e in result.values())
    assert all(e.evidence == () for e in result.values())


def test_suggest_candidates_reads_the_roster_once_with_roster_entry_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jettae.domain import RosterEntry

    net = Net()
    kim = net.person("김대리", PersonStatus.DEPARTED)
    lead = net.person("박팀장")
    other = net.person("최팀장")
    roster = (
        RosterEntry(name="김대리", status=PersonStatus.DEPARTED, team="결제팀"),
        # first entry in roster order wins, by name *or* alias — exactly like ctx.roster_entry
        RosterEntry(name="박지훈", aliases=("박팀장",), team="결제팀", title="팀장"),
        RosterEntry(name="박팀장", team="다른팀", title="사원"),
        RosterEntry(name="최팀장", team="정산팀", title="팀장"),
    )
    ctx = CheckContext(graph=net.build(), roster=roster)
    expected = ctx.roster_entry(ctx.graph.nodes[lead])
    assert expected is not None
    assert expected.name == "박지훈"

    def no_scan(self: CheckContext, person: Node) -> None:
        raise AssertionError("suggest_candidates must not scan the roster per person")

    monkeypatch.setattr(CheckContext, "roster_entry", no_scan)
    assert suggest_candidates(ctx, [kim]) == [lead]
    assert other not in suggest_candidates(ctx, [kim])


def test_suggest_candidates_skips_held_unavailable_and_inactive_people() -> None:
    net = Net()
    kim = net.person("김대리", PersonStatus.DEPARTED)
    held = net.person("박팀장", title="팀장", hold=True)
    away = net.person("이팀장", PersonStatus.ON_LEAVE, title="팀장")
    off = net.person("최팀장", title="팀장")
    ok = net.person("정팀장", title="팀장")
    net.team("결제팀", kim, held, away, off, ok)
    ctx = net.ctx(unavailable=frozenset({off}))
    assert suggest_candidates(ctx, [kim]) == [ok]
    assert suggest_candidates(ctx, []) == []
    assert suggest_candidates(ctx, ["person:없음"]) == []
