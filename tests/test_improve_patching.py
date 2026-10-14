"""Tests for ``jettae.improve.patching``: strict application, inversion and diffs."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from jettae.domain import (
    AddEdge,
    AddNode,
    Attr,
    AttrKey,
    Edge,
    EdgeKind,
    Extractor,
    Node,
    NodeKind,
    OpsGraph,
    PatchOp,
    Provenance,
    RemoveEdge,
    RemoveNode,
    SetAttr,
    SourceKind,
    edge_id,
    node_id,
)
from jettae.improve.patching import GraphDiff, PatchError, apply_ops, graph_diff, invert_ops
from test_improve_support import ROSTER, Org

PATCH = Provenance(source=SourceKind.PATCH, extractor=Extractor.HUMAN, ref="patch-test")


def person(name: str) -> Node:
    return Node(id=node_id(NodeKind.PERSON, name), kind=NodeKind.PERSON, name=name, prov=(PATCH,))


def link(kind: EdgeKind, src: str, dst: str, **attrs: int) -> Edge:
    values = {k: Attr(value=v, source=SourceKind.PATCH) for k, v in attrs.items()}
    return Edge(id=edge_id(kind, src, dst), kind=kind, src=src, dst=dst, attrs=values, prov=(PATCH,))


@pytest.fixture
def base() -> OpsGraph:
    org = Org()
    a = org.person("가나다", title="사원", team="결제팀")
    b = org.person("라마바", title="팀장", team="결제팀")
    svc = org.service("결제")
    org.owner(svc, a)
    org.escalate(a, b, 10)
    return org.graph()


A = node_id(NodeKind.PERSON, "가나다")
B = node_id(NodeKind.PERSON, "라마바")
SVC = node_id(NodeKind.SERVICE, "결제")
OWNER = edge_id(EdgeKind.OWNED_BY, SVC, A)
ESC = edge_id(EdgeKind.ESCALATES_TO, A, B)


# --------------------------------------------------------------------------- application


def test_add_node_and_edge_then_identical_repeats_are_dropped(base: OpsGraph) -> None:
    new = person("사아자")
    ops: list[PatchOp] = [
        AddNode(node=new),
        AddEdge(edge=link(EdgeKind.OWNED_BY, SVC, new.id)),
        AddNode(node=new),
        AddEdge(edge=link(EdgeKind.OWNED_BY, SVC, new.id)),
    ]
    graph, applied = apply_ops(base, ops)
    assert new.id in graph.nodes
    assert edge_id(EdgeKind.OWNED_BY, SVC, new.id) in graph.edges
    assert applied == tuple(ops[:2])


def test_existing_identical_element_is_a_no_op_and_a_different_one_is_refused(base: OpsGraph) -> None:
    graph, applied = apply_ops(base, [AddNode(node=base.nodes[A]), AddEdge(edge=base.edges[OWNER])])
    assert applied == ()
    assert graph.content_hash() == base.content_hash()
    clash = base.nodes[A].model_copy(update={"name": "다른 이름"})
    with pytest.raises(PatchError, match="같은 id의 다른 노드"):
        apply_ops(base, [AddNode(node=clash)])
    other = base.edges[ESC].model_copy(update={"attrs": {}})
    with pytest.raises(PatchError, match="같은 id의 다른 연결"):
        apply_ops(base, [AddEdge(edge=other)])


def test_add_edge_checks_endpoints_signature_and_id(base: OpsGraph) -> None:
    with pytest.raises(PatchError, match="끝 노드가 없습니다"):
        apply_ops(base, [AddEdge(edge=link(EdgeKind.OWNED_BY, SVC, "person:없는사람"))])
    with pytest.raises(PatchError, match="둘 수 없습니다"):
        apply_ops(base, [AddEdge(edge=link(EdgeKind.OWNED_BY, A, SVC))])
    forged = link(EdgeKind.OWNED_BY, SVC, B).model_copy(update={"id": "owned_by:service:x->person:y"})
    with pytest.raises(PatchError, match="형식이 아닙니다"):
        apply_ops(base, [AddEdge(edge=forged)])
    tagged = link(EdgeKind.OWNED_BY, SVC, B).model_copy(
        update={"id": f"{edge_id(EdgeKind.OWNED_BY, SVC, B)}#2"}
    )
    graph, _ = apply_ops(base, [AddEdge(edge=tagged)])
    assert tagged.id in graph.edges


def test_node_id_must_match_its_kind(base: OpsGraph) -> None:
    wrong = person("사아자").model_copy(update={"id": "role:사아자"})
    with pytest.raises(PatchError, match="종류"):
        apply_ops(base, [AddNode(node=wrong)])


def test_remove_edge_fills_snapshot_and_refuses_missing_or_stale(base: OpsGraph) -> None:
    graph, applied = apply_ops(base, [RemoveEdge(edge_id=ESC)])
    assert ESC not in graph.edges
    assert applied == (RemoveEdge(edge_id=ESC, snapshot=base.edges[ESC]),)
    with pytest.raises(PatchError, match="그래프에 없습니다"):
        apply_ops(base, [RemoveEdge(edge_id="escalates_to:x->y")])
    stale = base.edges[ESC].model_copy(update={"attrs": {}})
    with pytest.raises(PatchError, match="바뀌었습니다"):
        apply_ops(base, [RemoveEdge(edge_id=ESC, snapshot=stale)])


def test_remove_node_needs_no_remaining_edges(base: OpsGraph) -> None:
    with pytest.raises(PatchError, match=r"연결 \d+건이 남아"):
        apply_ops(base, [RemoveNode(node_id=B)])
    touching = sorted(e.id for e in base.edges.values() if B in (e.src, e.dst))
    graph, applied = apply_ops(base, [*(RemoveEdge(edge_id=e) for e in touching), RemoveNode(node_id=B)])
    assert B not in graph.nodes
    assert applied[-1] == RemoveNode(node_id=B, snapshot=base.nodes[B])
    with pytest.raises(PatchError, match="그래프에 없습니다"):
        apply_ops(base, [RemoveNode(node_id="person:없음")])
    stale = base.nodes[B].model_copy(update={"name": "옛 이름"})
    with pytest.raises(PatchError, match="바뀌었습니다"):
        apply_ops(
            graph.model_copy(update={"nodes": {**graph.nodes, B: base.nodes[B]}}),
            [RemoveNode(node_id=B, snapshot=stale)],
        )


def test_set_attr_sets_replaces_removes_and_records_previous(base: OpsGraph) -> None:
    fifteen = Attr(value=15, source=SourceKind.PATCH)
    graph, applied = apply_ops(base, [SetAttr(target_id=ESC, key=AttrKey.AFTER_MINUTES.value, value=fifteen)])
    assert graph.edges[ESC].attr(AttrKey.AFTER_MINUTES) == 15
    previous = base.edges[ESC].attrs[AttrKey.AFTER_MINUTES.value]
    assert applied == (SetAttr(target_id=ESC, key="after_minutes", value=fifteen, previous=previous),)
    removed, applied = apply_ops(base, [SetAttr(target_id=A, key=AttrKey.TITLE.value, value=None)])
    assert removed.nodes[A].attr(AttrKey.TITLE) is None
    assert isinstance(applied[0], SetAttr)
    assert applied[0].previous == base.nodes[A].attrs["title"]


def test_set_attr_no_ops_are_dropped(base: OpsGraph) -> None:
    same = base.edges[ESC].attrs[AttrKey.AFTER_MINUTES.value]
    graph, applied = apply_ops(
        base,
        [
            SetAttr(target_id=ESC, key="after_minutes", value=same),
            SetAttr(target_id=ESC, key="never_there", value=None),
        ],
    )
    assert applied == ()
    assert graph.content_hash() == base.content_hash()


def test_set_attr_refuses_missing_ambiguous_and_stale_targets(base: OpsGraph) -> None:
    attr = Attr(value=1, source=SourceKind.PATCH)
    with pytest.raises(PatchError, match="그래프에 없습니다"):
        apply_ops(base, [SetAttr(target_id="person:없음", key="x", value=attr)])
    twin = Edge(id=A, kind=EdgeKind.OWNED_BY, src=SVC, dst=B, prov=(ROSTER,))
    ambiguous = OpsGraph(nodes=base.nodes, edges={**base.edges, A: twin}, docs=base.docs)
    with pytest.raises(PatchError, match="모두의 id"):
        apply_ops(ambiguous, [SetAttr(target_id=A, key="x", value=attr)])
    wrong_previous = Attr(value=99, source=SourceKind.PATCH)
    with pytest.raises(PatchError, match="값이 바뀌었습니다"):
        apply_ops(base, [SetAttr(target_id=ESC, key="after_minutes", value=attr, previous=wrong_previous)])


def test_result_must_pass_integrity(base: OpsGraph) -> None:
    orphan = OpsGraph(
        nodes=base.nodes,
        edges={**base.edges, "x": link(EdgeKind.OWNED_BY, SVC, "person:유령").model_copy(update={"id": "x"})},
    )
    with pytest.raises(PatchError, match="무결성"):
        apply_ops(orphan, [])


def test_input_graph_is_never_modified(base: OpsGraph) -> None:
    before = base.content_hash()
    apply_ops(base, [RemoveEdge(edge_id=ESC), AddNode(node=person("사아자"))])
    assert base.content_hash() == before


# --------------------------------------------------------------------------- inversion & diff


def test_invert_restores_the_graph_exactly(base: OpsGraph) -> None:
    new = person("사아자")
    ops: list[PatchOp] = [
        RemoveEdge(edge_id=OWNER),
        AddNode(node=new),
        AddEdge(edge=link(EdgeKind.OWNED_BY, SVC, new.id)),
        AddEdge(edge=link(EdgeKind.ESCALATES_TO, new.id, B, after_minutes=5)),
        SetAttr(target_id=ESC, key="after_minutes", value=Attr(value=3, source=SourceKind.PATCH)),
        SetAttr(target_id=SVC, key="slo_minutes", value=Attr(value=30, source=SourceKind.PATCH)),
    ]
    changed, applied = apply_ops(base, ops)
    restored, _ = apply_ops(changed, invert_ops(applied))
    assert restored.content_hash() == base.content_hash()


def test_invert_needs_snapshots() -> None:
    with pytest.raises(PatchError, match="스냅숏"):
        invert_ops([RemoveEdge(edge_id="e")])
    with pytest.raises(PatchError, match="스냅숏"):
        invert_ops([RemoveNode(node_id="person:x")])


def test_graph_diff_lists_sorted_changes(base: OpsGraph) -> None:
    assert graph_diff(base, base) == GraphDiff()
    assert graph_diff(base, base).is_empty
    new = person("사아자")
    changed, _ = apply_ops(
        base,
        [
            RemoveEdge(edge_id=OWNER),
            AddNode(node=new),
            AddEdge(edge=link(EdgeKind.OWNED_BY, SVC, new.id)),
            SetAttr(target_id=ESC, key="after_minutes", value=Attr(value=3, source=SourceKind.PATCH)),
            SetAttr(target_id=A, key="title", value=None),
        ],
    )
    diff = graph_diff(base, changed)
    assert diff == GraphDiff(
        added_nodes=(new.id,),
        changed_nodes=(A,),
        added_edges=(edge_id(EdgeKind.OWNED_BY, SVC, new.id),),
        removed_edges=(OWNER,),
        changed_edges=(ESC,),
    )
    assert not diff.is_empty
    assert graph_diff(changed, base).removed_nodes == (new.id,)


# --------------------------------------------------------------------------- properties

NAMES = ("가", "나", "다", "라", "마", "바")


def _people(graph: OpsGraph) -> list[str]:
    return [n.id for n in graph.nodes.values() if n.kind is NodeKind.PERSON]


def _draw_add_person(draw: st.DrawFn, graph: OpsGraph, _svc: str) -> PatchOp | None:
    name = draw(st.sampled_from(["사", "아", "자"]))
    return None if node_id(NodeKind.PERSON, name) in graph.nodes else AddNode(node=person(name))


def _draw_link(draw: st.DrawFn, graph: OpsGraph, svc: str) -> PatchOp | None:
    people = _people(graph)
    if not people:
        return None
    target = draw(st.sampled_from(people))
    if draw(st.booleans()):
        candidate = link(EdgeKind.OWNED_BY, svc, target)
    else:
        source = draw(st.sampled_from(people))
        candidate = link(EdgeKind.ESCALATES_TO, source, target, after_minutes=draw(st.integers(1, 30)))
    if candidate.id in graph.edges or candidate.src == candidate.dst:
        return None
    return AddEdge(edge=candidate)


def _draw_remove_edge(draw: st.DrawFn, graph: OpsGraph, _svc: str) -> PatchOp | None:
    return RemoveEdge(edge_id=draw(st.sampled_from(sorted(graph.edges)))) if graph.edges else None


def _draw_remove_node(draw: st.DrawFn, graph: OpsGraph, _svc: str) -> PatchOp | None:
    linked = {n for e in graph.edges.values() for n in (e.src, e.dst)}
    isolated = [p for p in _people(graph) if p not in linked]
    return RemoveNode(node_id=draw(st.sampled_from(isolated))) if isolated else None


def _draw_set_attr(draw: st.DrawFn, graph: OpsGraph, svc: str) -> PatchOp | None:
    values = st.one_of(st.none(), st.integers(1, 99).map(lambda v: Attr(value=v, source=SourceKind.PATCH)))
    return SetAttr(
        target_id=draw(st.sampled_from([svc, *sorted(graph.edges)])),
        key=draw(st.sampled_from(["slo_minutes", "after_minutes"])),
        value=draw(values),
    )


_DRAWERS = (_draw_add_person, _draw_link, _draw_remove_edge, _draw_remove_node, _draw_set_attr)


@st.composite
def graph_and_ops(draw: st.DrawFn) -> tuple[OpsGraph, tuple[PatchOp, ...]]:
    """A small graph and a sequence of ops that is valid by construction (applied step by step)."""
    org = Org()
    people = [org.person(n, team="팀") for n in NAMES[: draw(st.integers(2, len(NAMES)))]]
    svc = org.service("서비스")
    for p in people:
        if draw(st.booleans()):
            org.owner(svc, p)
    graph = current = org.graph()
    ops: list[PatchOp] = []
    for _ in range(draw(st.integers(0, 8))):
        op = draw(st.sampled_from(_DRAWERS))(draw, current, svc)
        if op is not None:
            current, _ = apply_ops(current, [op])
            ops.append(op)
    return graph, tuple(ops)


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(graph_and_ops())
def test_property_inverse_restores_content_hash(case: tuple[OpsGraph, tuple[PatchOp, ...]]) -> None:
    graph, ops = case
    changed, applied = apply_ops(graph, ops)
    restored, _ = apply_ops(changed, invert_ops(applied))
    assert restored.content_hash() == graph.content_hash()


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(graph_and_ops())
def test_property_apply_is_deterministic_and_reapplying_applied_ops_is_idempotent(
    case: tuple[OpsGraph, tuple[PatchOp, ...]],
) -> None:
    graph, ops = case
    first, applied = apply_ops(graph, ops)
    second, applied_again = apply_ops(graph, ops)
    assert first.content_hash() == second.content_hash()
    assert applied == applied_again
    replayed, _ = apply_ops(graph, applied)
    assert replayed.content_hash() == first.content_hash()


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(graph_and_ops())
def test_property_diff_matches_the_applied_ops(case: tuple[OpsGraph, tuple[PatchOp, ...]]) -> None:
    graph, ops = case
    changed, _ = apply_ops(graph, ops)
    diff = graph_diff(graph, changed)
    assert set(diff.added_edges) == changed.edges.keys() - graph.edges.keys()
    assert set(diff.removed_nodes) == graph.nodes.keys() - changed.nodes.keys()
    assert all(list(t) == sorted(t) for t in diff.model_dump().values())
    assert diff.is_empty == (changed.content_hash() == graph.content_hash())
