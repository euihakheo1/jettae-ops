"""Response-path lineage and minimal cut sets (SPOF) — scenarios + LDFI-style properties."""

from __future__ import annotations

import itertools
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jettae.check.context import CheckContext
from jettae.check.spof import (
    RESPONDER_KEY,
    GraphIndex,
    escalation_edges,
    minimal_cut_sets,
    response_cut_sets,
    response_lineage,
    response_path_ok,
)
from jettae.domain import (
    ActionType,
    Attr,
    Edge,
    EdgeKind,
    Extractor,
    GraphBuilder,
    Node,
    NodeKind,
    OpsGraph,
    PersonStatus,
    Provenance,
    SourceKind,
    SourceSpan,
    Verification,
    edge_id,
    node_id,
)


def _span(quote: str) -> SourceSpan:
    return SourceSpan(doc_id="doc-test", start=0, end=len(quote), quote=quote, line=1)


RUNBOOK = Provenance(source=SourceKind.RUNBOOK, extractor=Extractor.RULE, spans=(_span("런북 근거"),))
ROSTER = Provenance(source=SourceKind.ROSTER, extractor=Extractor.IMPORT, ref="roster.csv")


class G:
    """Small fluent builder for response-path graphs."""

    def __init__(self) -> None:
        self.b = GraphBuilder()

    def node(self, kind: NodeKind, name: str, prov: Provenance = RUNBOOK, **attrs: Any) -> str:
        nid = node_id(kind, name)
        self.b.add_node(
            Node(
                id=nid,
                kind=kind,
                name=name,
                attrs={k: Attr(value=v, spans=prov.spans, source=prov.source) for k, v in attrs.items()},
                prov=(prov,),
            )
        )
        return nid

    def edge(self, kind: EdgeKind, src: str, dst: str, **attrs: Any) -> str:
        eid = edge_id(kind, src, dst)
        self.b.add_edge(
            Edge(
                id=eid,
                kind=kind,
                src=src,
                dst=dst,
                attrs={k: Attr(value=v, spans=RUNBOOK.spans) for k, v in attrs.items()},
                prov=(RUNBOOK,),
            )
        )
        return eid

    def service(self, name: str, tier: int = 1) -> str:
        return self.node(NodeKind.SERVICE, name, tier=tier)

    def channel(self, name: str) -> str:
        return self.node(NodeKind.CHANNEL, name)

    def person(self, name: str, *channels: str, status: PersonStatus = PersonStatus.ACTIVE) -> str:
        pid = self.node(NodeKind.PERSON, name, prov=ROSTER, status=status.value)
        for ch in channels:
            self.edge(EdgeKind.REACHABLE_VIA, pid, ch)
        return pid

    def action(self, name: str, **attrs: Any) -> str:
        return self.node(NodeKind.ACTION, name, **attrs)

    def hold(self, element_id: str, reason: str = "역번역 대조 불일치") -> str:
        """Put an existing node or edge on HOLD (판단 보류)."""
        table: dict[str, Any] = self.b.nodes if element_id in self.b.nodes else self.b.edges
        table[element_id] = table[element_id].model_copy(
            update={"verification": Verification.HOLD, "hold_reason": reason}
        )
        return element_id

    def build(self) -> OpsGraph:
        return self.b.build()


def ctx_of(g: G) -> CheckContext:
    return CheckContext(graph=g.build())


# ---------------------------------------------------------------------------- scenarios


def test_single_owner_single_channel_is_two_single_points() -> None:
    g = G()
    svc, slack = g.service("결제"), g.channel("Slack")
    kim = g.person("김대리", slack)
    g.edge(EdgeKind.OWNED_BY, svc, kim)
    ctx = ctx_of(g)
    assert response_path_ok(ctx, svc)
    assert response_cut_sets(ctx, svc) == [frozenset({slack}), frozenset({kim})]


def test_escalation_deputy_removes_single_points_and_pairs_are_minimal() -> None:
    g = G()
    svc, slack, phone = g.service("결제"), g.channel("Slack"), g.channel("전화")
    kim, park = g.person("김대리", slack), g.person("박과장", phone)
    g.edge(EdgeKind.OWNED_BY, svc, kim)
    g.edge(EdgeKind.ESCALATES_TO, kim, park, after_minutes=30)
    ctx = ctx_of(g)
    cuts = response_cut_sets(ctx, svc)
    assert all(len(c) == 2 for c in cuts)
    assert set(cuts) == {
        frozenset({slack, phone}),
        frozenset({kim, park}),
        frozenset({kim, phone}),
        frozenset({slack, park}),
    }
    assert cuts == sorted(cuts, key=lambda c: (len(c), sorted(c)))


def test_departed_owner_is_covered_by_active_escalation_target() -> None:
    g = G()
    svc, slack = g.service("결제"), g.channel("Slack")
    gone = g.person("이퇴사", slack, status=PersonStatus.DEPARTED)
    lead = g.person("박팀장", slack)
    g.edge(EdgeKind.OWNED_BY, svc, gone)
    g.edge(EdgeKind.ESCALATES_TO, gone, lead)
    ctx = ctx_of(g)
    assert response_path_ok(ctx, svc)
    assert response_cut_sets(ctx, svc, max_size=1) == [frozenset({slack}), frozenset({lead})]


def test_broken_path_returns_the_empty_cut() -> None:
    g = G()
    svc, slack = g.service("결제"), g.channel("Slack")
    gone = g.person("이퇴사", slack, status=PersonStatus.DEPARTED)
    g.edge(EdgeKind.OWNED_BY, svc, gone)
    ctx = ctx_of(g)
    assert not response_path_ok(ctx, svc)
    assert response_cut_sets(ctx, svc) == [frozenset()]


def test_unknown_status_is_not_a_responder() -> None:
    g = G()
    svc, slack = g.service("결제"), g.channel("Slack")
    g.edge(EdgeKind.OWNED_BY, svc, g.person("미확인", slack, status=PersonStatus.UNKNOWN))
    assert not response_path_ok(ctx_of(g), svc)


def test_person_without_any_channel_cannot_respond() -> None:
    g = G()
    svc = g.service("결제")
    g.edge(EdgeKind.OWNED_BY, svc, g.person("김대리"))
    assert not response_path_ok(ctx_of(g), svc)


def test_ctx_unavailable_is_respected() -> None:
    g = G()
    svc, slack = g.service("결제"), g.channel("Slack")
    kim = g.person("김대리", slack)
    g.edge(EdgeKind.OWNED_BY, svc, kim)
    ctx = ctx_of(g)
    assert not response_path_ok(ctx.with_unavailable([kim]), svc)
    assert not response_path_ok(ctx.with_unavailable([slack]), svc)
    assert response_cut_sets(ctx.with_unavailable([slack]), svc) == [frozenset()]


def test_fallback_channel_closure_counts_as_a_way_in() -> None:
    g = G()
    svc, slack, phone = g.service("결제"), g.channel("Slack"), g.channel("전화")
    kim = g.person("김대리", slack)
    g.edge(EdgeKind.FALLBACK_TO, slack, phone)
    g.edge(EdgeKind.OWNED_BY, svc, kim)
    ctx = ctx_of(g)
    assert response_path_ok(ctx.with_unavailable([slack]), svc)
    assert response_cut_sets(ctx, svc) == [frozenset({kim}), frozenset({phone, slack})]


def test_rollback_needs_an_approver_by_default() -> None:
    g = G()
    svc, slack = g.service("결제"), g.channel("Slack")
    g.edge(EdgeKind.OWNED_BY, svc, g.person("김대리", slack))
    rollback = g.action("롤백", action_type=ActionType.ROLLBACK.value)
    g.edge(EdgeKind.MITIGATED_BY, svc, rollback)
    ctx = ctx_of(g)
    assert not response_path_ok(ctx, svc)
    broken = response_lineage(ctx, svc).broken()
    assert [r.key for r in broken] == [rollback]


def test_explicit_no_approval_overrides_rollback_default() -> None:
    g = G()
    svc, slack = g.service("결제"), g.channel("Slack")
    g.edge(EdgeKind.OWNED_BY, svc, g.person("김대리", slack))
    g.edge(EdgeKind.MITIGATED_BY, svc, g.action("롤백", action_type="rollback", requires_approval=False))
    assert response_path_ok(ctx_of(g), svc)


def test_restart_without_approval_edge_needs_no_approver() -> None:
    g = G()
    svc, slack = g.service("결제"), g.channel("Slack")
    g.edge(EdgeKind.OWNED_BY, svc, g.person("김대리", slack))
    g.edge(EdgeKind.MITIGATED_BY, svc, g.action("재시작", action_type=ActionType.RESTART.value))
    lineage = response_lineage(ctx_of(g), svc)
    assert [r.key for r in lineage.requirements] == [RESPONDER_KEY]


def test_approver_through_role_is_a_single_point() -> None:
    g = G()
    svc, slack, phone = g.service("결제"), g.channel("Slack"), g.channel("전화")
    kim, park = g.person("김대리", slack), g.person("박과장", phone)
    lead = g.person("최팀장", phone)
    g.edge(EdgeKind.OWNED_BY, svc, kim)
    g.edge(EdgeKind.ESCALATES_TO, kim, park)
    role = g.node(NodeKind.ROLE, "결제팀장")
    g.edge(EdgeKind.HOLDS_ROLE, lead, role)
    rollback = g.action("롤백", action_type="rollback")
    g.edge(EdgeKind.MITIGATED_BY, svc, rollback)
    g.edge(EdgeKind.APPROVED_BY, rollback, role)
    ctx = ctx_of(g)
    assert response_path_ok(ctx, svc)
    singles = response_cut_sets(ctx, svc, max_size=1)
    assert frozenset({lead}) in singles
    lineage = response_lineage(ctx, svc)
    assert [r.label for r in lineage.broken([lead])] == ["'롤백' 승인"]
    assert edge_id(EdgeKind.HOLDS_ROLE, lead, role) in lineage.path_ids


def test_trigger_notify_channel_and_role_pager_are_extra_ways_in() -> None:
    g = G()
    svc, slack, pager = g.service("결제"), g.channel("Slack"), g.channel("PagerDuty")
    oncall_role = g.node(NodeKind.ROLE, "1차 당직")
    kim = g.person("김대리", slack)
    g.edge(EdgeKind.HOLDS_ROLE, kim, oncall_role)
    alarm = g.node(NodeKind.TRIGGER, "결제 오류율 알람")
    g.edge(EdgeKind.DETECTED_BY, svc, alarm)
    g.edge(EdgeKind.NOTIFIES, alarm, oncall_role, channel=pager)
    ctx = ctx_of(g)
    assert response_path_ok(ctx.with_unavailable([slack]), svc)
    assert response_cut_sets(ctx, svc) == [frozenset({kim}), frozenset({pager, slack})]


def test_escalation_from_role_follows_person_level_edges() -> None:
    g = G()
    svc, slack = g.service("결제"), g.channel("Slack")
    role = g.node(NodeKind.ROLE, "결제 담당")
    kim = g.person("김대리", status=PersonStatus.DEPARTED)
    g.edge(EdgeKind.HOLDS_ROLE, kim, role)
    park = g.person("박과장", slack)
    g.edge(EdgeKind.OWNED_BY, svc, role)
    g.edge(EdgeKind.ESCALATES_TO, kim, park)
    assert response_path_ok(ctx_of(g), svc)


def test_escalation_cycle_is_safe() -> None:
    g = G()
    svc, slack = g.service("결제"), g.channel("Slack")
    a = g.person("가", status=PersonStatus.DEPARTED)
    b = g.person("나", status=PersonStatus.DEPARTED)
    g.edge(EdgeKind.OWNED_BY, svc, a)
    g.edge(EdgeKind.ESCALATES_TO, a, b)
    g.edge(EdgeKind.ESCALATES_TO, b, a)
    g.channel("미사용")
    assert not response_path_ok(ctx_of(g), svc)
    assert slack not in response_lineage(ctx_of(g), svc).candidates


def test_max_size_zero_and_negative() -> None:
    g = G()
    svc, slack = g.service("결제"), g.channel("Slack")
    g.edge(EdgeKind.OWNED_BY, svc, g.person("김대리", slack))
    ctx = ctx_of(g)
    assert response_cut_sets(ctx, svc, max_size=0) == []
    with pytest.raises(ValueError, match="max_size"):
        response_cut_sets(ctx, svc, max_size=-1)


def test_lineage_rejects_non_service_nodes() -> None:
    g = G()
    pid = g.person("김대리")
    with pytest.raises(ValueError, match="대응 경로"):
        response_lineage(ctx_of(g), pid)
    with pytest.raises(ValueError, match="대응 경로"):
        response_lineage(ctx_of(g), "service:없음")


def test_minimal_cut_sets_on_top_of_existing_unavailability() -> None:
    g = G()
    svc, slack, phone = g.service("결제"), g.channel("Slack"), g.channel("전화")
    kim, park = g.person("김대리", slack), g.person("박과장", phone)
    g.edge(EdgeKind.OWNED_BY, svc, kim)
    g.edge(EdgeKind.OWNED_BY, svc, park)
    lineage = response_lineage(ctx_of(g), svc)
    assert minimal_cut_sets(lineage, unavailable=frozenset({kim}), max_size=1) == [
        frozenset({phone}),
        frozenset({park}),
    ]


# ---------------------------------------------------------------------------- properties


@st.composite
def response_graphs(draw: st.DrawFn) -> tuple[CheckContext, str]:
    """Random owners/escalations/approvers over 2-6 people and 1-3 channels.

    Kept within the subset of semantics the naive oracle below re-implements with plain
    ``CheckContext`` queries (no role-level channels, no alert channels).
    """
    g, svc = _draw_response_graph(draw)
    return ctx_of(g), svc


def _draw_response_graph(draw: st.DrawFn, *, approval_flag: bool | None = True) -> tuple[G, str]:
    g = G()
    svc = g.service("svc")
    channels = [g.channel(f"ch{i}") for i in range(draw(st.integers(1, 3)))]
    people: list[str] = []
    for i in range(draw(st.integers(2, 6))):
        status = draw(
            st.sampled_from([PersonStatus.ACTIVE] * 3 + [PersonStatus.DEPARTED, PersonStatus.UNKNOWN])
        )
        mine = draw(st.lists(st.sampled_from(channels), max_size=2, unique=True))
        people.append(g.person(f"p{i}", *mine, status=status))
    role = g.node(NodeKind.ROLE, "role")
    for p in draw(st.lists(st.sampled_from(people), max_size=2, unique=True)):
        g.edge(EdgeKind.HOLDS_ROLE, p, role)
    owners = draw(st.lists(st.sampled_from([*people, role]), min_size=1, max_size=2, unique=True))
    for o in owners:
        g.edge(EdgeKind.OWNED_BY, svc, o)
    for a, b in draw(st.lists(st.tuples(st.sampled_from(people), st.sampled_from(people)), max_size=5)):
        if a != b:
            g.edge(EdgeKind.ESCALATES_TO, a, b)
    if draw(st.booleans()):
        flag = {} if approval_flag is None else {"requires_approval": approval_flag}
        action = g.action("act", **flag)
        g.edge(EdgeKind.MITIGATED_BY, svc, action)
        for p in draw(st.lists(st.sampled_from([*people, role]), max_size=2, unique=True)):
            g.edge(EdgeKind.APPROVED_BY, action, p)
    return g, svc


def _naive_ok(ctx: CheckContext, sid: str) -> bool:
    """Oracle using only CheckContext queries."""

    def reachable(nid: str) -> bool:
        return any(ctx.is_active(p) and ctx.available_channels(p.id) for p in ctx.resolve_people(nid))

    entries = [o.id for o in ctx.owners(sid)]
    starts = {*entries, *(p.id for e in entries for p in ctx.resolve_people(e))}
    closure = {n.id for s in starts for n in ctx.escalation_closure(s)}
    if not any(reachable(n) for n in (*entries, *closure)):
        return False
    gated = [a for a in ctx.mitigations(sid) if ctx.requires_approval(a)]
    return all(any(reachable(ap.id) for ap in ctx.approvers(a.id)) for a in gated)


def _universe(ctx: CheckContext) -> list[str]:
    return [n.id for n in ctx.graph.nodes_of(NodeKind.PERSON, NodeKind.CHANNEL)]


@settings(max_examples=200, deadline=None)
@given(response_graphs(), st.data())
def test_lineage_agrees_with_naive_oracle(case: tuple[CheckContext, str], data: st.DataObject) -> None:
    ctx, sid = case
    down = data.draw(st.lists(st.sampled_from(_universe(ctx)), max_size=3))
    what_if = ctx.with_unavailable(down)
    assert response_path_ok(what_if, sid) == _naive_ok(what_if, sid)


@settings(max_examples=120, deadline=None)
@given(response_graphs())
def test_cut_sets_are_cuts_minimal_and_complete(case: tuple[CheckContext, str]) -> None:
    ctx, sid = case
    cuts = response_cut_sets(ctx, sid, max_size=2)
    if not response_path_ok(ctx, sid):
        assert cuts == [frozenset()]
        return
    for cut in cuts:
        assert 1 <= len(cut) <= 2
        assert not response_path_ok(ctx.with_unavailable(cut), sid)  # it is a cut
        for x in cut:
            assert response_path_ok(ctx.with_unavailable(cut - {x}), sid)  # and minimal
    assert cuts == sorted(cuts, key=lambda c: (len(c), sorted(c)))
    # completeness: every breaking subset (of ANY person/channel, size <= 2) contains a returned cut
    for size in (1, 2):
        for combo in itertools.combinations(_universe(ctx), size):
            if not response_path_ok(ctx.with_unavailable(combo), sid):
                assert any(cut <= frozenset(combo) for cut in cuts)


@settings(max_examples=50, deadline=None)
@given(response_graphs())
def test_cut_sets_are_deterministic(case: tuple[CheckContext, str]) -> None:
    ctx, sid = case
    assert response_cut_sets(ctx, sid) == response_cut_sets(ctx, sid)
    assert response_lineage(ctx, sid) == response_lineage(ctx, sid)


@settings(max_examples=60, deadline=None)
@given(response_graphs())
def test_graph_index_mirrors_graph_and_context_queries(case: tuple[CheckContext, str]) -> None:
    ctx, _ = case
    index = GraphIndex(ctx.graph)
    for nid in ctx.graph.nodes:
        assert index.resolve_people(nid) == ctx.resolve_people(nid)
        for kind in EdgeKind:
            assert index.out_edges(nid, kind) == ctx.graph.out_edges(nid, kind)
        expected = {
            e.id for src in {nid, *(p.id for p in ctx.resolve_people(nid))} for e in ctx.escalations(src)
        }
        assert [e.id for e in escalation_edges(index, nid)] == sorted(expected)


# ---------------------------------------------------------------------------- HOLD (판단 보류) — reviewer
# P3: an unconfirmed fact never makes the response path pass (same stance as the scorer and the
# rehearsal simulator); an unconfirmed fact that *adds* an approval requirement is still kept.


def _owned(g: G) -> tuple[str, str, str]:
    svc, slack = g.service("결제"), g.channel("Slack")
    kim = g.person("김대리", slack)
    return svc, slack, kim


def test_held_owner_edge_does_not_make_the_path_ok() -> None:
    g = G()
    svc, _, kim = _owned(g)
    g.hold(g.edge(EdgeKind.OWNED_BY, svc, kim))
    ctx = ctx_of(g)
    assert not response_path_ok(ctx, svc)
    assert response_cut_sets(ctx, svc) == [frozenset()]
    assert kim not in response_lineage(ctx, svc).candidates


def test_held_person_node_is_not_a_responder() -> None:
    g = G()
    svc, _, kim = _owned(g)
    g.edge(EdgeKind.OWNED_BY, svc, kim)
    g.hold(kim, "동명이인: 김대리 2명")
    assert not response_path_ok(ctx_of(g), svc)


@pytest.mark.parametrize("held", ["channel", "reachable_via"])
def test_held_channel_or_contact_link_is_not_a_way_in(held: str) -> None:
    g = G()
    svc, slack, kim = _owned(g)
    g.edge(EdgeKind.OWNED_BY, svc, kim)
    g.hold(slack if held == "channel" else edge_id(EdgeKind.REACHABLE_VIA, kim, slack))
    assert not response_path_ok(ctx_of(g), svc)


def test_held_fallback_link_does_not_add_a_channel() -> None:
    g = G()
    svc, slack, kim = _owned(g)
    phone = g.channel("전화")
    g.edge(EdgeKind.OWNED_BY, svc, kim)
    g.hold(g.edge(EdgeKind.FALLBACK_TO, slack, phone))
    ctx = ctx_of(g)
    assert not response_path_ok(ctx.with_unavailable([slack]), svc)
    assert frozenset({slack}) in response_cut_sets(ctx, svc, max_size=1)


def test_held_escalation_and_role_membership_do_not_extend_the_closure() -> None:
    g = G()
    svc, slack = g.service("결제"), g.channel("Slack")
    gone = g.person("이퇴사", slack, status=PersonStatus.DEPARTED)
    lead = g.person("박팀장", slack)
    g.edge(EdgeKind.OWNED_BY, svc, gone)
    escalation = g.edge(EdgeKind.ESCALATES_TO, gone, lead)
    assert response_path_ok(ctx_of(g), svc)
    g.hold(escalation)
    assert not response_path_ok(ctx_of(g), svc)

    g = G()
    svc, slack = g.service("결제"), g.channel("Slack")
    role = g.node(NodeKind.ROLE, "1차 당직")
    kim = g.person("김대리", slack)
    g.hold(g.edge(EdgeKind.HOLDS_ROLE, kim, role))
    g.edge(EdgeKind.OWNED_BY, svc, role)
    assert not response_path_ok(ctx_of(g), svc)


def test_held_notify_channel_attribute_is_not_a_way_in() -> None:
    g = G()
    svc, slack, kim = _owned(g)
    pager = g.channel("PagerDuty")
    alarm = g.node(NodeKind.TRIGGER, "결제 오류율 알람")
    g.edge(EdgeKind.DETECTED_BY, svc, alarm)
    g.edge(EdgeKind.NOTIFIES, alarm, kim, channel=pager)
    assert response_path_ok(ctx_of(g).with_unavailable([slack]), svc)
    g.hold(pager)
    assert not response_path_ok(ctx_of(g).with_unavailable([slack]), svc)


def test_held_approver_cannot_approve_a_gated_step() -> None:
    g = G()
    svc, _, kim = _owned(g)
    g.edge(EdgeKind.OWNED_BY, svc, kim)
    restart = g.action("재시작", action_type=ActionType.RESTART.value, requires_approval=True)
    g.edge(EdgeKind.MITIGATED_BY, svc, restart)
    approval = g.edge(EdgeKind.APPROVED_BY, restart, kim)
    assert response_path_ok(ctx_of(g), svc)
    g.hold(approval)
    lineage = response_lineage(ctx_of(g), svc)
    assert [r.key for r in lineage.broken()] == [restart]  # requirement kept, no usable approver


def test_approver_edge_alone_adds_no_approval_requirement() -> None:
    """Single rule (CheckContext.requires_approval): naming an approver of a restart gates nothing,
    so a departed approver named on it is not a single point of failure."""
    g = G()
    svc, _, kim = _owned(g)
    g.edge(EdgeKind.OWNED_BY, svc, kim)
    restart = g.action("재시작", action_type=ActionType.RESTART.value)
    g.edge(EdgeKind.MITIGATED_BY, svc, restart)
    g.edge(EdgeKind.APPROVED_BY, restart, g.person("김대리", status=PersonStatus.DEPARTED))
    lineage = response_lineage(ctx_of(g), svc)
    assert [r.key for r in lineage.requirements] == [RESPONDER_KEY]
    assert response_path_ok(ctx_of(g), svc)


def test_held_gated_mitigation_still_requires_an_approver() -> None:
    g = G()
    svc, _, kim = _owned(g)
    g.edge(EdgeKind.OWNED_BY, svc, kim)
    rollback = g.action("롤백", action_type=ActionType.ROLLBACK.value)
    g.hold(g.edge(EdgeKind.MITIGATED_BY, svc, rollback))
    assert [r.key for r in response_lineage(ctx_of(g), svc).broken()] == [rollback]


def test_index_built_from_another_graph_is_rejected() -> None:
    g = G()
    svc, _, kim = _owned(g)
    g.edge(EdgeKind.OWNED_BY, svc, kim)
    stale = GraphIndex(G().build())
    with pytest.raises(ValueError, match="GraphIndex"):
        response_lineage(ctx_of(g), svc, index=stale)


def test_korean_and_mixed_script_ids_are_ordered_deterministically() -> None:
    g = G()
    svc = g.service("결제")
    channels = [g.channel(n) for n in ("카카오톡", "Slack", "전화", "SMS")]
    people = [g.person(n, ch) for n, ch in zip(("김대리", "Kim", "박과장", "이주임"), channels, strict=True)]
    for p in people:
        g.edge(EdgeKind.OWNED_BY, svc, p)
    ctx = ctx_of(g)
    assert response_cut_sets(ctx, svc, max_size=2) == []  # four independent responders
    fours = response_cut_sets(ctx, svc, max_size=4)
    assert len(fours) == 2**4  # one of (person, own channel) per responder
    assert fours[0] == frozenset(channels)  # "channel:…" sorts before "person:…"
    assert fours[-1] == frozenset(people)
    assert response_lineage(ctx, svc).candidates == tuple(sorted([*channels, *people]))


@st.composite
def held_response_graphs(draw: st.DrawFn) -> tuple[CheckContext, str]:
    """``response_graphs`` plus random HOLD on people, channels, the role, the action and any edge.

    The approval flag may also be absent, so an approver edge (held or not) alone gates the action.
    """
    g, svc = _draw_response_graph(draw, approval_flag=draw(st.sampled_from([True, None])))
    candidates = sorted(nid for nid in g.b.nodes if nid != svc) + sorted(g.b.edges)
    for element in draw(st.lists(st.sampled_from(candidates), max_size=4, unique=True)):
        g.hold(element)
    return ctx_of(g), svc


def _reachable(ctx: CheckContext, nid: str, held: set[str]) -> bool:
    if nid in held:
        return False
    return any(ctx.is_active(p) and ctx.available_channels(p.id) for p in ctx.resolve_people(nid))


def _naive_ok_with_holds(ctx: CheckContext, sid: str) -> bool:
    """Independent oracle: plain CheckContext queries over a graph with held facts removed,
    while approval requirements are read from the full graph."""
    full = ctx.graph
    held = {nid for nid, n in full.nodes.items() if n.verification is Verification.HOLD}
    usable_edges = {
        eid: e
        for eid, e in full.edges.items()
        if e.verification is Verification.VERIFIED and e.src not in held and e.dst not in held
    }
    view = ctx.with_graph(OpsGraph(nodes=full.nodes, edges=usable_edges, docs=full.docs))
    entries = [o.id for o in view.owners(sid)]
    starts = {*entries, *(p.id for e in entries for p in view.resolve_people(e))}
    closure = {n.id for s in starts for n in view.escalation_closure(s)}
    if not any(_reachable(view, n, held) for n in (*entries, *closure)):
        return False
    gated = [a for a in ctx.mitigations(sid) if ctx.requires_approval(a)]
    return all(any(_reachable(view, ap.id, held) for ap in view.approvers(a.id)) for a in gated)


@settings(max_examples=200, deadline=None)
@given(held_response_graphs(), st.data())
def test_lineage_with_holds_agrees_with_naive_oracle(
    case: tuple[CheckContext, str], data: st.DataObject
) -> None:
    ctx, sid = case
    down = data.draw(st.lists(st.sampled_from(_universe(ctx)), max_size=3))
    what_if = ctx.with_unavailable(down)
    assert response_path_ok(what_if, sid) == _naive_ok_with_holds(what_if, sid)


@settings(max_examples=120, deadline=None)
@given(held_response_graphs(), st.data())
def test_losing_more_never_repairs_the_path(case: tuple[CheckContext, str], data: st.DataObject) -> None:
    """Monotonicity: if the path survives losing X and Y, it survives losing X alone."""
    ctx, sid = case
    universe = _universe(ctx)
    x = data.draw(st.lists(st.sampled_from(universe), max_size=3))
    y = data.draw(st.lists(st.sampled_from(universe), max_size=3))
    if response_path_ok(ctx.with_unavailable([*x, *y]), sid):
        assert response_path_ok(ctx.with_unavailable(x), sid)


@settings(max_examples=80, deadline=None)
@given(held_response_graphs())
def test_cut_sets_with_holds_are_cuts_and_minimal(case: tuple[CheckContext, str]) -> None:
    ctx, sid = case
    cuts = response_cut_sets(ctx, sid, max_size=2)
    if not response_path_ok(ctx, sid):
        assert cuts == [frozenset()]
        return
    held = {nid for nid, n in ctx.graph.nodes.items() if n.verification is Verification.HOLD}
    for cut in cuts:
        assert not cut & held  # a held person/channel is never a fault-injection target
        assert not response_path_ok(ctx.with_unavailable(cut), sid)
        for x in cut:
            assert response_path_ok(ctx.with_unavailable(cut - {x}), sid)
