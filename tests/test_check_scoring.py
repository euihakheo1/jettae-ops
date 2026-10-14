"""Tests for the readiness score (jettae.check.scoring, SCORING_V1)."""

from __future__ import annotations

import decimal
from dataclasses import dataclass

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from jettae.check.context import CheckContext
from jettae.check.elements import evaluate_service
from jettae.check.scoring import (
    SCORING_V1,
    element_weight,
    overall_score,
    score,
    scoring_table_markdown,
    service_readiness,
    tier_weight,
)
from jettae.domain import (
    ActionType,
    Attr,
    ChannelMedium,
    EdgeKind,
    Element,
    ElementStatus,
    Extractor,
    Finding,
    GraphBuilder,
    Node,
    NodeKind,
    OpsGraph,
    PersonStatus,
    Policy,
    Provenance,
    ReadinessReport,
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
    """Minimal GraphBuilder DSL (roster facts → ROSTER provenance, runbook facts → span)."""

    def __init__(self, base: OpsGraph | None = None) -> None:
        self.b = GraphBuilder(base)
        self._offset = 0

    def _prov(self, quote: str) -> Provenance:
        self._offset += len(quote) + 1
        return rb(quote, self._offset)

    def node(
        self, kind: NodeKind, name: str, prov: Provenance | None = None, **attrs: AttrValue | None
    ) -> str:
        p = prov or self._prov(name)
        nid = node_id(kind, name)
        values = {k: Attr(value=v, source=p.source) for k, v in attrs.items() if v is not None}
        self.b.add_node(Node(id=nid, kind=kind, name=name, attrs=values, prov=(p,)))
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

    def person(self, name: str, status: PersonStatus = PersonStatus.ACTIVE, title: str | None = None) -> str:
        return self.node(NodeKind.PERSON, name, ROSTER, status=status.value, title=title)

    def channel(self, name: str, medium: ChannelMedium) -> str:
        return self.node(NodeKind.CHANNEL, name, ROSTER, medium=medium.value)

    def build(self) -> OpsGraph:
        return self.b.build()


def ready_service(net: Net, name: str, tier: int) -> str:
    """A service whose five elements all pass."""
    sid = net.node(NodeKind.SERVICE, name, tier=tier)
    owner = net.person(f"{name} 담당")
    lead = net.person(f"{name} 팀장")
    for who in (owner, lead):
        net.edge(EdgeKind.REACHABLE_VIA, who, net.channel(f"{who} 휴대폰", ChannelMedium.PHONE), prov=ROSTER)
        net.edge(EdgeKind.REACHABLE_VIA, who, net.channel(f"{who} Slack", ChannelMedium.SLACK), prov=ROSTER)
    net.edge(EdgeKind.OWNED_BY, sid, owner)
    net.edge(EdgeKind.ESCALATES_TO, owner, lead, after_minutes=10)
    trigger = net.node(NodeKind.TRIGGER, f"{name} 알람")
    net.edge(EdgeKind.DETECTED_BY, sid, trigger)
    net.edge(EdgeKind.NOTIFIES, trigger, owner)
    action = net.node(NodeKind.ACTION, f"{name} 롤백", action_type=ActionType.ROLLBACK.value)
    net.edge(EdgeKind.MITIGATED_BY, sid, action)
    net.edge(EdgeKind.APPROVED_BY, action, owner)
    net.edge(EdgeKind.APPROVED_BY, action, lead)
    net.edge(EdgeKind.EXECUTED_BY, action, owner)
    return sid


@dataclass(frozen=True)
class Demo:
    graph: OpsGraph
    service: str
    kim: str
    lead: str
    action: str


def demo() -> Demo:
    """The competition demo: 결제, alarm → 김대리(퇴사), rollback without approver, nothing else."""
    net = Net()
    sid = net.node(NodeKind.SERVICE, "결제", tier=1)
    kim = net.person("김대리", PersonStatus.DEPARTED, title="대리")
    lead = net.person("박팀장", title="팀장")
    team = net.node(NodeKind.TEAM, "결제팀", ROSTER)
    net.edge(EdgeKind.MEMBER_OF, kim, team, prov=ROSTER)
    net.edge(EdgeKind.MEMBER_OF, lead, team, prov=ROSTER)
    trigger = net.node(NodeKind.TRIGGER, "결제 실패율 5% 초과 알람")
    net.edge(EdgeKind.DETECTED_BY, sid, trigger)
    net.edge(EdgeKind.NOTIFIES, trigger, kim)
    net.edge(EdgeKind.OWNED_BY, sid, kim)
    action = net.node(NodeKind.ACTION, "이전 버전으로 롤백", action_type=ActionType.ROLLBACK.value)
    net.edge(EdgeKind.MITIGATED_BY, sid, action)
    return Demo(graph=net.build(), service=sid, kim=kim, lead=lead, action=action)


def fixed_demo(d: Demo) -> OpsGraph:
    """Demo after the human-approved fixes: escalation to 박팀장, approver, two channels."""
    net = Net(d.graph)
    net.edge(EdgeKind.ESCALATES_TO, d.kim, d.lead, after_minutes=30)
    net.edge(EdgeKind.APPROVED_BY, d.action, d.lead)
    net.edge(EdgeKind.REACHABLE_VIA, d.lead, net.channel("박팀장 휴대폰", ChannelMedium.PHONE), prov=ROSTER)
    net.edge(EdgeKind.REACHABLE_VIA, d.lead, net.channel("결제팀 Slack", ChannelMedium.SLACK), prov=ROSTER)
    return net.build()


# --------------------------------------------------------------------------- published constants


def test_scoring_v1_constants_are_published_as_specified() -> None:
    assert SCORING_V1["version"] == "v1"
    assert SCORING_V1["element_weights"] == {
        "detection": 0.20,
        "owner": 0.25,
        "approver": 0.20,
        "fallback_channel": 0.15,
        "rollback": 0.20,
    }
    assert SCORING_V1["tier_weights"] == {1: 3.0, 2: 2.0, 3: 1.0}
    assert sum(element_weight(e) for e in Element) == pytest.approx(1.0)
    assert set(SCORING_V1["element_weights"]) == {e.value for e in Element}


def test_tier_weight_falls_back_to_lowest_weight() -> None:
    assert tier_weight(1) == 3.0
    assert tier_weight(4) == 1.0
    assert tier_weight(0) == 1.0


def test_scoring_table_markdown_renders_weights_and_semantics_in_korean() -> None:
    md = scoring_table_markdown()
    assert md.startswith("# 준비도 점수 산식 (SCORING_V1)")
    assert "| 담당 | `owner` | 0.25 |" in md
    assert "| 대체 채널 | `fallback_channel` | 0.15 |" in md
    assert "| 1 | 핵심 | 3.0 |" in md
    assert "조치가 없어 승인 경로를 평가할 수 없음" in md
    assert md.count("### ") == len(Element)
    assert md.endswith("\n")
    assert not md.endswith("\n\n")
    assert scoring_table_markdown() == md


# --------------------------------------------------------------------------- scoring


def test_empty_graph_scores_zero() -> None:
    report = score(CheckContext(graph=OpsGraph()), [])
    assert report.overall == 0.0
    assert report.services == ()
    assert report.weights_version == "v1"
    assert report.graph_hash == OpsGraph().content_hash()


def test_fully_ready_service_scores_100() -> None:
    net = Net()
    sid = ready_service(net, "결제", tier=1)
    report = score(CheckContext(graph=net.build()), [])
    (svc,) = report.services
    assert svc.service_id == sid
    assert svc.service_name == "결제"
    assert svc.score == 100.0
    assert report.overall == 100.0
    assert all(e.status is ElementStatus.PASS for e in svc.elements)
    assert [e.element for e in svc.elements] == list(Element)


def test_demo_graph_scores_low_and_fixes_raise_it() -> None:
    d = demo()
    before = score(CheckContext(graph=d.graph), [])
    (svc,) = before.services
    by_element = {e.element: e for e in svc.elements}
    assert by_element[Element.DETECTION].score == 0.5
    assert by_element[Element.OWNER].score == 0.0
    assert by_element[Element.APPROVER].score == 0.0
    assert by_element[Element.FALLBACK_CHANNEL].score == 0.0
    assert by_element[Element.ROLLBACK].score == 0.6
    assert svc.score == 22.0  # 100 × (0.2×0.5 + 0.2×0.6)
    assert before.overall == 22.0

    after = score(CheckContext(graph=fixed_demo(d)), [])
    (fixed,) = after.services
    fixed_by_element = {e.element: e for e in fixed.elements}
    assert fixed_by_element[Element.OWNER].score == 0.4
    assert fixed_by_element[Element.APPROVER].score == 0.8
    assert fixed_by_element[Element.FALLBACK_CHANNEL].score == 1.0
    assert fixed.score == 63.0
    assert after.overall > before.overall
    assert after.graph_hash != before.graph_hash


def test_service_score_matches_weighted_element_sum() -> None:
    d = demo()
    ctx = CheckContext(graph=d.graph)
    evals = evaluate_service(ctx, d.service)
    expected = round(100 * sum(element_weight(e) * v.score for e, v in evals.items()), 1)
    assert service_readiness(ctx, d.service).score == expected


def test_overall_is_tier_weighted_over_critical_services() -> None:
    net = Net()
    ready_service(net, "결제", tier=1)  # 100
    net.node(NodeKind.SERVICE, "장바구니", tier=2)  # 0
    net.node(NodeKind.SERVICE, "사내 위키", tier=3)  # not critical → ignored in overall
    report = score(CheckContext(graph=net.build()), [])
    assert [s.service_name for s in report.services] == ["결제", "사내 위키", "장바구니"]
    assert [s.weight for s in report.services] == [3.0, 1.0, 2.0]
    assert report.overall == round((3.0 * 100 + 2.0 * 0) / 5.0, 1)


def test_overall_uses_all_services_when_none_is_critical() -> None:
    net = Net()
    ready_service(net, "사내 위키", tier=3)
    net.node(NodeKind.SERVICE, "사내 메일", tier=3)
    report = score(CheckContext(graph=net.build()), [])
    assert report.overall == 50.0


def test_overall_score_edge_cases() -> None:
    assert overall_score([], []) == 0.0


def test_service_tier_defaults_follow_policy() -> None:
    net = Net()
    sid = net.node(NodeKind.SERVICE, "결제")
    report = score(CheckContext(graph=net.build(), policy=Policy(default_tier=3)), [])
    (svc,) = report.services
    assert svc.service_id == sid
    assert svc.tier == 3
    assert svc.weight == 1.0


def test_finding_ids_link_to_matching_service_and_element() -> None:
    d = demo()
    ctx = CheckContext(graph=d.graph)

    def make(rule: str, element: Element | None, service: str | None) -> Finding:
        return Finding(
            id=Finding.make_id(rule, service or "-"),
            rule_id=rule,
            severity=Severity.HIGH,
            title="t",
            message="m",
            service_id=service,
            element=element,
        )

    owner_finding = make("JT-PATH-010", Element.OWNER, d.service)
    unrelated = make("JT-PATH-010", Element.OWNER, "service:다른")
    no_element = make("JT-PPL-001", None, d.service)
    report = score(ctx, [unrelated, owner_finding, no_element])
    elements = {e.element: e for e in report.services[0].elements}
    assert elements[Element.OWNER].finding_ids == (owner_finding.id,)
    assert elements[Element.DETECTION].finding_ids == ()


def test_service_score_rounds_half_up_exactly() -> None:
    """11.25 must publish as 11.3 — what a reader recomputing by hand gets, on any decimal context."""
    net = Net()
    net.node(NodeKind.SERVICE, "결제", tier=1)
    sid = node_id(NodeKind.SERVICE, "결제")
    kim = net.person("김대리", PersonStatus.DEPARTED)
    lead = net.person("박팀장")
    net.edge(EdgeKind.OWNED_BY, sid, kim)
    net.edge(EdgeKind.ESCALATES_TO, kim, lead)  # no after_minutes → escalation part 0.3
    net.edge(EdgeKind.REACHABLE_VIA, lead, net.channel("박팀장 휴대폰", ChannelMedium.PHONE), prov=ROSTER)
    ctx = CheckContext(graph=net.build())
    report = score(ctx, [])
    (svc,) = report.services
    assert {e.element: e.score for e in svc.elements} == {
        Element.DETECTION: 0.0,
        Element.OWNER: 0.3,
        Element.APPROVER: 0.0,
        Element.FALLBACK_CHANNEL: 0.25,
        Element.ROLLBACK: 0.0,
    }
    assert round(100 * (0.25 * 0.3 + 0.15 * 0.25), 1) == 11.2  # binary float + banker's rounding
    assert svc.score == 11.3
    assert report.overall == 11.3
    with decimal.localcontext() as hostile:
        hostile.prec = 2
        hostile.rounding = decimal.ROUND_DOWN
        assert score(ctx, []) == report


def test_overall_rounds_the_weighted_mean_half_up() -> None:
    net = Net()
    ready_service(net, "결제", tier=1)  # 100 × 3
    net.node(NodeKind.SERVICE, "장바구니", tier=2)  # 0 × 2
    net.node(NodeKind.SERVICE, "정산", tier=2)  # 0 × 2
    assert score(CheckContext(graph=net.build()), []).overall == 42.9  # 300 / 7 = 42.857…


def test_unknown_element_counts_as_zero() -> None:
    net = Net()
    sid = net.node(NodeKind.SERVICE, "결제", tier=1)
    trigger = net.node(NodeKind.TRIGGER, "알람")
    net.edge(EdgeKind.DETECTED_BY, sid, trigger, hold=True)
    report = score(CheckContext(graph=net.build()), [])
    det = report.services[0].elements[0]
    assert det.status is ElementStatus.UNKNOWN
    assert det.score == 0.0


# --------------------------------------------------------------------------- property tests


@dataclass(frozen=True)
class World:
    graph: OpsGraph
    people: tuple[str, ...]
    responders: tuple[str, ...]
    actions: tuple[str, ...]


def _pairs(
    draw: st.DrawFn, srcs: list[str], dsts: list[str], odds: int, *, allow_self: bool = False
) -> list[tuple[str, str]]:
    """Each (src, dst) pair independently with probability 1/odds."""
    return [
        (a, b) for a in srcs for b in dsts if (allow_self or a != b) and draw(st.integers(0, odds - 1)) == 0
    ]


def _rare(draw: st.DrawFn) -> bool:
    return draw(st.integers(0, 9)) == 0


def _add_action(
    draw: st.DrawFn, net: Net, sid: str, name: str, *, approvers: list[str], doers: list[str]
) -> str:
    kind = draw(st.sampled_from([*ActionType, None]))
    aid = net.node(
        NodeKind.ACTION,
        name,
        action_type=kind.value if kind else None,
        requires_approval=draw(st.sampled_from([True, False, None])),
    )
    net.edge(EdgeKind.MITIGATED_BY, sid, aid)
    for _, who in _pairs(draw, [aid], approvers, 3):
        net.edge(EdgeKind.APPROVED_BY, aid, who)
    for _, who in _pairs(draw, [aid], doers, 4):
        net.edge(EdgeKind.EXECUTED_BY, aid, who)
    return aid


def _add_service(
    draw: st.DrawFn, net: Net, index: int, idc: str, *, approvers: list[str], responders: list[str]
) -> list[str]:
    sid = net.node(NodeKind.SERVICE, f"서비스{index}", tier=draw(st.integers(1, 3)))
    if draw(st.booleans()):
        net.edge(EdgeKind.RUNS_IN, sid, idc, prov=CATALOG)
    for _, who in _pairs(draw, [sid], responders, 3):
        net.edge(EdgeKind.OWNED_BY, sid, who, hold=_rare(draw))
    for t in range(draw(st.integers(0, 2))):
        trig = net.node(NodeKind.TRIGGER, f"서비스{index} 알람{t}")
        net.edge(EdgeKind.DETECTED_BY, sid, trig, hold=_rare(draw))
        for _, who in _pairs(draw, [trig], responders, 3):
            net.edge(EdgeKind.NOTIFIES, trig, who)
    return [
        _add_action(draw, net, sid, f"서비스{index} 조치{a}", approvers=approvers, doers=responders)
        for a in range(draw(st.integers(0, 2)))
    ]


@st.composite
def worlds(draw: st.DrawFn) -> World:
    """Small random operation graphs covering every edge kind the evaluator reads."""
    net = Net()
    idc = net.node(NodeKind.LOCATION, "IDC", CATALOG)
    statuses = st.sampled_from(list(PersonStatus))
    people = [net.person(f"사람{i}", draw(statuses)) for i in range(draw(st.integers(1, 4)))]
    roles = [net.node(NodeKind.ROLE, f"역할{i}", ROSTER) for i in range(draw(st.integers(0, 2)))]
    teams = [net.node(NodeKind.TEAM, f"팀{i}", ROSTER) for i in range(draw(st.integers(0, 1)))]
    for person, role in _pairs(draw, people, roles, 2):
        net.edge(EdgeKind.HOLDS_ROLE, person, role, prov=ROSTER)
    for person, team in _pairs(draw, people, teams, 2):
        net.edge(EdgeKind.MEMBER_OF, person, team, prov=ROSTER)
    media = st.sampled_from(list(ChannelMedium))
    channels = [net.channel(f"채널{i}", draw(media)) for i in range(draw(st.integers(0, 3)))]
    for channel in channels:
        if draw(st.booleans()):
            net.edge(EdgeKind.RUNS_IN, channel, idc, prov=CATALOG)
    responders = [*people, *roles, *teams]
    for who, channel in _pairs(draw, responders, channels, 3):
        net.edge(EdgeKind.REACHABLE_VIA, who, channel, prov=ROSTER, hold=_rare(draw))
    for primary, backup in _pairs(draw, channels, channels, 4):
        net.edge(EdgeKind.FALLBACK_TO, primary, backup)
    actions = [
        aid
        for index in range(draw(st.integers(1, 2)))
        for aid in _add_service(draw, net, index, idc, approvers=[*people, *roles], responders=responders)
    ]
    for src, dst in _pairs(draw, responders, responders, 4, allow_self=True):
        net.edge(EdgeKind.ESCALATES_TO, src, dst, after_minutes=draw(st.sampled_from([None, 5, 30])))
    return World(net.build(), tuple(people), tuple(responders), tuple(actions))


def _scores(report: ReadinessReport) -> dict[str, float]:
    return {s.service_id: s.score for s in report.services}


def _assert_not_lower(before: ReadinessReport, after: ReadinessReport) -> None:
    old, new = _scores(before), _scores(after)
    for sid, value in old.items():
        assert new[sid] >= value, (sid, value, new[sid])
    assert after.overall >= before.overall


PROPERTY = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@PROPERTY
@given(worlds())
def test_scores_are_within_bounds(world: World) -> None:
    report = score(CheckContext(graph=world.graph), [])
    assert 0.0 <= report.overall <= 100.0
    for svc in report.services:
        assert 0.0 <= svc.score <= 100.0
        for e in svc.elements:
            assert 0.0 <= e.score <= 1.0
            if e.status is ElementStatus.UNKNOWN:
                assert e.score == 0.0
            if e.status is ElementStatus.PASS:
                assert e.score == 1.0


@PROPERTY
@given(worlds(), st.data())
def test_adding_an_active_approver_never_lowers_scores(world: World, data: st.DataObject) -> None:
    if not world.actions:
        return
    action = data.draw(st.sampled_from(world.actions))
    before = score(CheckContext(graph=world.graph), [])
    net = Net(world.graph)
    helper = net.person("추가 승인자")
    net.edge(
        EdgeKind.REACHABLE_VIA, helper, net.channel("추가 승인자 휴대폰", ChannelMedium.PHONE), prov=ROSTER
    )
    net.edge(EdgeKind.APPROVED_BY, action, helper)
    _assert_not_lower(before, score(CheckContext(graph=net.build()), []))


@PROPERTY
@given(worlds(), st.data())
def test_adding_a_channel_never_lowers_scores(world: World, data: st.DataObject) -> None:
    who = data.draw(st.sampled_from(world.responders))
    medium = data.draw(st.sampled_from(list(ChannelMedium)))
    before = score(CheckContext(graph=world.graph), [])
    net = Net(world.graph)
    net.edge(EdgeKind.REACHABLE_VIA, who, net.channel("새 연락 수단", medium), prov=ROSTER)
    _assert_not_lower(before, score(CheckContext(graph=net.build()), []))


@PROPERTY
@given(worlds(), st.data())
def test_adding_an_escalation_never_lowers_scores(world: World, data: st.DataObject) -> None:
    source = data.draw(st.sampled_from(world.responders))
    minutes = data.draw(st.sampled_from([None, 10]))
    before = score(CheckContext(graph=world.graph), [])
    net = Net(world.graph)
    target = net.person("새 상향 대상")
    if data.draw(st.booleans()):
        net.edge(
            EdgeKind.REACHABLE_VIA, target, net.channel("상향 대상 휴대폰", ChannelMedium.PHONE), prov=ROSTER
        )
    net.edge(EdgeKind.ESCALATES_TO, source, target, after_minutes=minutes)
    _assert_not_lower(before, score(CheckContext(graph=net.build()), []))


@PROPERTY
@given(worlds())
def test_scoring_is_deterministic_and_order_independent(world: World) -> None:
    first = score(CheckContext(graph=world.graph), [])
    rebuilt = GraphBuilder()
    for n in reversed(list(world.graph.nodes.values())):
        rebuilt.add_node(n)
    for e in reversed(list(world.graph.edges.values())):
        rebuilt.add_edge(e)
    second = score(CheckContext(graph=rebuilt.build()), [])
    assert first.model_dump_json() == second.model_dump_json()
    assert first == score(CheckContext(graph=world.graph), [])


# --------------------------------------------------------------------------- review: stronger invariants


@PROPERTY
@given(worlds(), st.data())
def test_any_escalation_between_existing_responders_never_lowers_scores(
    world: World, data: st.DataObject
) -> None:
    """Stronger than adding a fresh target: any ESCALATES_TO edge (self-loops, loops, departed
    targets, untimed hops) only adds reach, so it can never cost points."""
    source = data.draw(st.sampled_from(world.responders))
    target = data.draw(st.sampled_from(world.responders))
    minutes = data.draw(st.sampled_from([None, 10]))
    before = score(CheckContext(graph=world.graph), [])
    net = Net(world.graph)
    net.edge(EdgeKind.ESCALATES_TO, source, target, after_minutes=minutes)
    _assert_not_lower(before, score(CheckContext(graph=net.build()), []))


@PROPERTY
@given(worlds(), st.data())
def test_adding_a_reachable_notify_target_or_executor_never_lowers_scores(
    world: World, data: st.DataObject
) -> None:
    before = score(CheckContext(graph=world.graph), [])
    net = Net(world.graph)
    helper = net.person("추가 대응자")
    net.edge(
        EdgeKind.REACHABLE_VIA, helper, net.channel("추가 대응자 휴대폰", ChannelMedium.PHONE), prov=ROSTER
    )
    triggers = [n.id for n in world.graph.nodes_of(NodeKind.TRIGGER)]
    if triggers:
        net.edge(EdgeKind.NOTIFIES, data.draw(st.sampled_from(triggers)), helper)
    if world.actions:
        net.edge(EdgeKind.EXECUTED_BY, data.draw(st.sampled_from(world.actions)), helper)
    _assert_not_lower(before, score(CheckContext(graph=net.build()), []))


@PROPERTY
@given(worlds(), st.data())
def test_scores_do_not_depend_on_the_findings_passed(world: World, data: st.DataObject) -> None:
    """Findings only annotate elements (finding_ids); they never move a number."""
    ctx = CheckContext(graph=world.graph)
    services = sorted(world.graph.nodes_of(NodeKind.SERVICE), key=lambda n: n.id)
    sid = data.draw(st.sampled_from([s.id for s in services]))
    element = data.draw(st.sampled_from(list(Element)))
    noise = Finding(
        id=Finding.make_id("JT-TEST-001", sid),
        rule_id="JT-TEST-001",
        severity=Severity.CRITICAL,
        title="t",
        message="m",
        service_id=sid,
        element=element,
    )
    plain, annotated = score(ctx, []), score(ctx, [noise, noise])
    assert _scores(plain) == _scores(annotated)
    assert plain.overall == annotated.overall
    linked = {
        (s.service_id, e.element): e.finding_ids
        for s in annotated.services
        for e in s.elements
        if e.finding_ids
    }
    assert linked == {(sid, element): (noise.id,)}


def test_published_approver_semantics_match_the_evaluator() -> None:
    """An unconfirmed approver next to a departed one makes the action UNKNOWN (P3), as published."""
    net = Net()
    sid = net.node(NodeKind.SERVICE, "결제", tier=1)
    action = net.node(NodeKind.ACTION, "롤백", action_type=ActionType.ROLLBACK.value)
    net.edge(EdgeKind.MITIGATED_BY, sid, action)
    net.edge(EdgeKind.APPROVED_BY, action, net.person("김대리", PersonStatus.DEPARTED))
    ghost = net.node(NodeKind.PERSON, "최주임")  # runbook-only mention, no roster status
    ctx = CheckContext(graph=net.build())
    assert evaluate_service(ctx, sid)[Element.APPROVER].score == 0.2
    net.edge(EdgeKind.APPROVED_BY, action, ghost)
    approver = evaluate_service(CheckContext(graph=net.build()), sid)[Element.APPROVER]
    assert (approver.score, approver.status) == (0.0, ElementStatus.UNKNOWN)
    md = scoring_table_markdown()
    assert "승인자 중 명부에서 확인되지 않는 인물이 있으면 확인 불가(UNKNOWN), 0" in md
    assert "명부 미확인 인물뿐이면" not in md
