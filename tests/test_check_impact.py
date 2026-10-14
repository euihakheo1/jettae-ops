"""Blast radius (영향 범위) — correctness, cycles, location failure, determinism and speed."""

from __future__ import annotations

import itertools
import time

import networkx as nx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jettae.check.impact import PROPAGATING_EDGES, blast_radius
from jettae.domain import (
    Edge,
    EdgeKind,
    Extractor,
    GraphBuilder,
    Node,
    NodeKind,
    OpsGraph,
    Provenance,
    SourceKind,
    Verification,
    edge_id,
    node_id,
)

CATALOG = Provenance(source=SourceKind.CATALOG, extractor=Extractor.IMPORT, ref="catalog.yaml")


class Builder:
    """Minimal catalog-style graph builder (no runbook spans needed for structure tests)."""

    def __init__(self) -> None:
        self.b = GraphBuilder()

    def node(self, kind: NodeKind, name: str) -> str:
        nid = node_id(kind, name)
        self.b.add_node(Node(id=nid, kind=kind, name=name, prov=(CATALOG,)))
        return nid

    def edge(self, kind: EdgeKind, src: str, dst: str) -> str:
        eid = edge_id(kind, src, dst)
        self.b.add_edge(Edge(id=eid, kind=kind, src=src, dst=dst, prov=(CATALOG,)))
        return eid

    def build(self) -> OpsGraph:
        return self.b.build()


def _shop() -> tuple[OpsGraph, dict[str, str]]:
    """결제 → 주문DB → IDC-A; 장바구니 → 결제; 검색 runs in IDC-B; Slack → 사내IdP → IDC-A."""
    g = Builder()
    ids = {
        "idc_a": g.node(NodeKind.LOCATION, "IDC-A"),
        "idc_b": g.node(NodeKind.LOCATION, "IDC-B"),
        "db": g.node(NodeKind.SYSTEM, "주문DB"),
        "idp": g.node(NodeKind.SYSTEM, "사내IdP"),
        "pay": g.node(NodeKind.SERVICE, "결제"),
        "cart": g.node(NodeKind.SERVICE, "장바구니"),
        "search": g.node(NodeKind.SERVICE, "검색"),
        "slack": g.node(NodeKind.CHANNEL, "Slack"),
        "person": g.node(NodeKind.PERSON, "김대리"),
    }
    g.edge(EdgeKind.RUNS_IN, ids["db"], ids["idc_a"])
    g.edge(EdgeKind.RUNS_IN, ids["idp"], ids["idc_a"])
    g.edge(EdgeKind.DEPENDS_ON, ids["pay"], ids["db"])
    g.edge(EdgeKind.DEPENDS_ON, ids["cart"], ids["pay"])
    g.edge(EdgeKind.RUNS_IN, ids["search"], ids["idc_b"])
    g.edge(EdgeKind.DEPENDS_ON, ids["slack"], ids["idp"])
    g.edge(EdgeKind.OWNED_BY, ids["pay"], ids["person"])
    return g.build(), ids


def test_location_failure_takes_down_everything_running_in_it() -> None:
    graph, ids = _shop()
    result = blast_radius(graph, [ids["idc_a"]])
    assert result.failed == (ids["idc_a"],)
    assert result.impacted_services == tuple(sorted([ids["pay"], ids["cart"]]))
    assert result.impacted_systems == tuple(sorted([ids["db"], ids["idp"]]))
    assert result.impacted_channels == (ids["slack"],)
    assert ids["search"] not in result.paths
    assert result.paths[ids["cart"]] == (ids["cart"], ids["pay"], ids["db"], ids["idc_a"])
    assert result.via_edges[ids["cart"]] == (
        edge_id(EdgeKind.DEPENDS_ON, ids["cart"], ids["pay"]),
        edge_id(EdgeKind.DEPENDS_ON, ids["pay"], ids["db"]),
        edge_id(EdgeKind.RUNS_IN, ids["db"], ids["idc_a"]),
    )


def test_failed_nodes_are_not_listed_as_impacted_and_is_down_covers_both() -> None:
    graph, ids = _shop()
    result = blast_radius(graph, [ids["pay"]])
    assert result.impacted_services == (ids["cart"],)
    assert ids["pay"] not in result.paths
    assert result.is_down(ids["pay"])
    assert result.is_down(ids["cart"])
    assert not result.is_down(ids["db"])


def test_people_do_not_propagate_failure() -> None:
    graph, ids = _shop()
    result = blast_radius(graph, [ids["person"]])
    assert result.impacted_services == ()
    assert result.paths == {}


def test_dependency_cycles_terminate_and_report_shortest_path() -> None:
    g = Builder()
    a, b, c = (g.node(NodeKind.SERVICE, n) for n in ("A", "B", "C"))
    g.edge(EdgeKind.DEPENDS_ON, a, b)
    g.edge(EdgeKind.DEPENDS_ON, b, a)
    g.edge(EdgeKind.DEPENDS_ON, b, c)
    graph = g.build()
    from_a = blast_radius(graph, [a])
    assert from_a.impacted_services == (b,)
    from_c = blast_radius(graph, [c])
    assert from_c.impacted_services == (a, b)
    assert from_c.paths[a] == (a, b, c)


def test_input_order_and_duplicates_do_not_change_result() -> None:
    graph, ids = _shop()
    one = blast_radius(graph, [ids["idc_a"], ids["idc_b"]])
    two = blast_radius(graph, [ids["idc_b"], ids["idc_a"], ids["idc_b"]])
    assert one == two
    assert one.model_dump_json() == two.model_dump_json()


def test_a_bare_string_is_one_failed_id_not_its_characters() -> None:
    graph, ids = _shop()
    assert blast_radius(graph, ids["idc_a"]) == blast_radius(graph, [ids["idc_a"]])


def test_held_dependency_still_propagates_failure() -> None:
    """Over-reporting impact on an unconfirmed dependency beats silently dropping a service (P3)."""
    g = Builder()
    idc = g.node(NodeKind.LOCATION, "판교 IDC")
    svc = g.node(NodeKind.SERVICE, "결제")
    eid = g.edge(EdgeKind.RUNS_IN, svc, idc)
    g.b.edges[eid] = g.b.edges[eid].model_copy(
        update={"verification": Verification.HOLD, "hold_reason": "역번역 대조 불일치"}
    )
    assert blast_radius(g.build(), [idc]).impacted_services == (svc,)


def test_multiple_failures_explain_each_node_by_its_nearest_failure() -> None:
    graph, ids = _shop()
    result = blast_radius(graph, [ids["idc_a"], ids["pay"]])
    assert result.paths[ids["cart"]] == (ids["cart"], ids["pay"])  # 1 hop, not via IDC-A
    assert ids["pay"] not in result.impacted_services  # failed, not collateral
    assert result.paths[ids["db"]] == (ids["db"], ids["idc_a"])


def test_channels_and_systems_running_in_a_failed_location_are_down_too() -> None:
    g = Builder()
    idc = g.node(NodeKind.LOCATION, "IDC-A")
    kakao = g.node(NodeKind.CHANNEL, "카카오톡")
    g.edge(EdgeKind.RUNS_IN, kakao, idc)
    result = blast_radius(g.build(), [idc])
    assert result.impacted_channels == (kakao,)
    assert result.is_down(kakao)
    assert result.impacted_services == ()


def test_unknown_failed_id_is_an_error_not_an_empty_impact() -> None:
    graph, _ = _shop()
    with pytest.raises(ValueError, match="그래프에 없는 노드"):
        blast_radius(graph, ["location:없는-센터"])


def test_empty_failure_set_has_no_impact() -> None:
    graph, _ = _shop()
    result = blast_radius(graph, [])
    assert result.failed == ()
    assert result.impacted_services == ()


# ---------------------------------------------------------------------------- property


def _oracle(graph: OpsGraph, failed: set[str]) -> dict[str, int]:
    """Independent check with networkx: node -> hop distance to the nearest failed node."""
    dg = nx.DiGraph()
    dg.add_nodes_from(graph.nodes)
    dg.add_edges_from((e.src, e.dst) for e in graph.edges.values() if e.kind in PROPAGATING_EDGES)
    out: dict[str, int] = {}
    for f in failed:
        # networkx < 3.5 yields pairs, newer versions return a dict; dict() accepts both.
        for nid, dist in dict(nx.single_target_shortest_path_length(dg, f)).items():
            if nid not in failed:
                out[nid] = min(out.get(nid, dist), dist)
    return out


@st.composite
def dependency_graphs(draw: st.DrawFn) -> tuple[OpsGraph, set[str]]:
    n_services = draw(st.integers(1, 8))
    n_systems = draw(st.integers(0, 5))
    n_locations = draw(st.integers(1, 3))
    g = Builder()
    services = [g.node(NodeKind.SERVICE, f"svc{i}") for i in range(n_services)]
    systems = [g.node(NodeKind.SYSTEM, f"sys{i}") for i in range(n_systems)]
    locations = [g.node(NodeKind.LOCATION, f"loc{i}") for i in range(n_locations)]
    deps = services + systems
    for src in deps:
        for dst in draw(st.lists(st.sampled_from(deps), max_size=3, unique=True)):
            if dst != src:
                g.edge(EdgeKind.DEPENDS_ON, src, dst)
        if draw(st.booleans()):
            g.edge(EdgeKind.RUNS_IN, src, draw(st.sampled_from(locations)))
    graph = g.build()
    failed = set(draw(st.lists(st.sampled_from(sorted(graph.nodes)), min_size=1, max_size=3)))
    return graph, failed


@settings(max_examples=120, deadline=None)
@given(dependency_graphs())
def test_blast_radius_matches_reverse_reachability(case: tuple[OpsGraph, set[str]]) -> None:
    graph, failed = case
    result = blast_radius(graph, failed)
    expected = _oracle(graph, failed)
    assert set(result.paths) == set(expected)
    impacted = (*result.impacted_services, *result.impacted_systems, *result.impacted_channels)
    assert sorted(impacted) == sorted(expected)
    for nid, path in result.paths.items():
        assert path[0] == nid
        assert path[-1] in failed
        assert len(path) - 1 == expected[nid]  # shortest
        for (src, dst), eid in zip(itertools.pairwise(path), result.via_edges[nid], strict=True):
            edge = graph.edges[eid]
            assert (edge.src, edge.dst) == (src, dst)
            assert edge.kind in PROPAGATING_EDGES


# ---------------------------------------------------------------------------- KPI benchmark


def _synthetic(n_services: int = 200, n_systems: int = 50, n_locations: int = 3) -> OpsGraph:
    """Deterministic synthetic estate: every system runs in a location; services fan out."""
    g = Builder()
    locations = [g.node(NodeKind.LOCATION, f"센터{i}") for i in range(n_locations)]
    systems = [g.node(NodeKind.SYSTEM, f"시스템{i}") for i in range(n_systems)]
    services = [g.node(NodeKind.SERVICE, f"서비스{i}") for i in range(n_services)]
    for i, sys_id in enumerate(systems):
        g.edge(EdgeKind.RUNS_IN, sys_id, locations[i % n_locations])
        if i >= 5:
            g.edge(EdgeKind.DEPENDS_ON, sys_id, systems[(i * 7) % i])
    for i, svc in enumerate(services):
        for k in (i, i * 3 + 1, i * 11 + 2):
            g.edge(EdgeKind.DEPENDS_ON, svc, systems[k % n_systems])
        if i >= 10:
            g.edge(EdgeKind.DEPENDS_ON, svc, services[(i * 13) % i])
    return g.build()


def test_kpi_single_center_failure_on_200_services_under_half_a_second() -> None:
    graph = _synthetic()
    assert len(graph.nodes_of(NodeKind.SERVICE)) == 200
    target = node_id(NodeKind.LOCATION, "센터0")
    started = time.perf_counter()
    result = blast_radius(graph, [target])
    elapsed = time.perf_counter() - started
    assert elapsed < 0.5, f"blast radius took {elapsed:.3f}s"
    assert set(result.paths) == set(_oracle(graph, {target}))
    assert len(result.impacted_services) > 0
