"""Pure builders behind the API's DTOs: graph view, version diff view, download names.

Hypothesis builds random (valid) operations graphs and checks the views against the graphs they
came from — every element exactly once and in id order, values equal to their attributes, findings
attached only to the graph they judged (P3) — and that the same input always gives the same JSON
(P6).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from urllib.parse import unquote

from hypothesis import given, settings
from hypothesis import strategies as st

from jettae.api.routers.reports import content_disposition, download_names
from jettae.api.schemas import graph_diff_view, graph_view
from jettae.domain import (
    Attr,
    DiagnosisReport,
    Edge,
    EdgeKind,
    Extractor,
    Finding,
    GraphBuilder,
    GraphVersion,
    LlmMode,
    LlmUsageSummary,
    Node,
    NodeKind,
    OpsGraph,
    Provenance,
    ReadinessReport,
    Severity,
    SourceKind,
    SourceSpan,
    edge_id,
    node_id,
)
from jettae.improve import graph_diff

CLOCK = datetime(2026, 10, 28, 14, 0, tzinfo=UTC)
_NAMES = ["결제", "주문", "환율", "김서연", "박재현", "이수진"]
_ATTR_KEYS = ["tier", "status", "title", "aliases"]


def _provenance(runbook: bool) -> Provenance:
    if runbook:
        span = SourceSpan(doc_id="doc-1", start=0, end=4, quote="결제 장애", line=1)
        return Provenance(source=SourceKind.RUNBOOK, extractor=Extractor.RULE, spans=(span,))
    return Provenance(source=SourceKind.ROSTER, extractor=Extractor.IMPORT, ref="roster.csv")


_attr_values = st.one_of(
    st.integers(min_value=1, max_value=3),
    st.sampled_from(["active", "departed", "대리", "팀장"]),
    st.lists(st.sampled_from(["김대리", "pay"]), max_size=2),
)


@st.composite
def graphs(draw: st.DrawFn) -> OpsGraph:
    builder = GraphBuilder()
    services = draw(st.lists(st.sampled_from(_NAMES[:3]), unique=True, max_size=3))
    people = draw(st.lists(st.sampled_from(_NAMES[3:]), unique=True, max_size=3))
    for kind, names in ((NodeKind.SERVICE, services), (NodeKind.PERSON, people)):
        for name in names:
            attrs = {
                key: Attr(value=draw(_attr_values), source=SourceKind.ROSTER)
                for key in draw(st.lists(st.sampled_from(_ATTR_KEYS), unique=True, max_size=3))
            }
            prov = tuple(_provenance(flag) for flag in draw(st.lists(st.booleans(), min_size=1, max_size=2)))
            builder.add_node(Node(id=node_id(kind, name), kind=kind, name=name, attrs=attrs, prov=prov))
    for service in services:
        for person in people:
            if draw(st.booleans()):
                src, dst = node_id(NodeKind.SERVICE, service), node_id(NodeKind.PERSON, person)
                attrs = {"after_minutes": Attr(value=draw(st.integers(1, 30)))} if draw(st.booleans()) else {}
                builder.add_edge(
                    Edge(
                        id=edge_id(EdgeKind.OWNED_BY, src, dst),
                        kind=EdgeKind.OWNED_BY,
                        src=src,
                        dst=dst,
                        attrs=attrs,
                        prov=(_provenance(draw(st.booleans())),),
                    )
                )
    return builder.build()


def _version(graph: OpsGraph, number: int = 1) -> GraphVersion:
    return GraphVersion(
        workspace_id="ws-1",
        number=number,
        parent=number - 1 if number > 1 else None,
        graph_hash=graph.content_hash(),
        created_by="test",
        created_at=CLOCK,
    )


def _report(graph_hash: str, subjects: list[list[str]]) -> DiagnosisReport:
    findings = tuple(
        Finding(
            id=f"JT-TEST-{index:03d}:0",
            rule_id="JT-TEST",
            severity=Severity.HIGH,
            title="테스트",
            message="테스트 끊김",
            subject_ids=tuple(ids),
        )
        for index, ids in enumerate(subjects)
    )
    readiness = ReadinessReport(
        overall=0.0,
        services=(),
        weights_version="v1",
        element_weights={},
        tier_weights={},
        graph_hash=graph_hash,
    )
    return DiagnosisReport(
        run_id="run-000001-abc",
        workspace_id="ws-1",
        graph_version=1,
        graph_hash=graph_hash,
        created_at=CLOCK,
        findings=findings,
        readiness=readiness,
        llm=LlmUsageSummary(mode=LlmMode.OFF),
    )


# --------------------------------------------------------------------------- graph view


@settings(max_examples=60)
@given(
    graphs(),
    st.lists(st.lists(st.sampled_from([*_NAMES, "service:결제", "ghost:x"]), max_size=3), max_size=4),
)
def test_graph_view_lists_every_element_once_with_its_values(
    graph: OpsGraph, raw_subjects: list[list[str]]
) -> None:
    subjects = [[f"service:{s}" if ":" not in s else s for s in ids] for ids in raw_subjects]
    version = _version(graph)
    report = _report(graph.content_hash(), subjects)
    view = graph_view(version, graph, report)
    assert [n.id for n in view.nodes] == sorted(graph.nodes)
    assert [e.id for e in view.edges] == sorted(graph.edges)
    for item in (*view.nodes, *view.edges):
        element = graph.nodes.get(item.id) or graph.edges[item.id]
        assert item.attrs == element.attrs
        assert item.values == {key: attr.value for key, attr in element.attrs.items()}
        assert list(item.values) == sorted(item.values)
        assert item.spans == element.spans
        assert list(item.sources) == sorted({p.source for p in element.prov}, key=lambda s: s.value)
    expected: dict[str, set[str]] = {}
    for finding in report.findings:
        for subject in finding.subject_ids:
            expected.setdefault(subject, set()).add(finding.id)
    assert {k: set(v) for k, v in view.findings_by_element.items()} == expected
    assert all(list(v) == sorted(v) for v in view.findings_by_element.values())
    assert view.findings_run_id == report.run_id
    assert view.model_dump_json() == graph_view(version, graph, report).model_dump_json()


@given(graphs())
def test_findings_of_another_graph_are_never_attached(graph: OpsGraph) -> None:
    report = _report("0" * 64, [sorted(graph.nodes)])
    view = graph_view(_version(graph), graph, report)
    assert view.findings_by_element == {}
    assert view.findings_run_id is None
    assert graph_view(_version(graph), graph, None).findings_by_element == {}


# --------------------------------------------------------------------------- diff view


@settings(max_examples=60)
@given(graphs(), graphs())
def test_diff_view_resolves_exactly_the_diffed_elements(before: OpsGraph, after: OpsGraph) -> None:
    ids = graph_diff(before, after)
    view = graph_diff_view(1, 2, ids, before, after)
    assert (view.a, view.b) == (1, 2)
    assert tuple(n.id for n in view.added_nodes) == ids.added_nodes
    assert tuple(n.id for n in view.removed_nodes) == ids.removed_nodes
    assert tuple(e.id for e in view.added_edges) == ids.added_edges
    assert tuple(e.id for e in view.removed_edges) == ids.removed_edges
    assert all(after.nodes[n.id] == n for n in view.added_nodes)
    assert all(before.nodes[n.id] == n for n in view.removed_nodes)
    for change in (*view.changed_nodes, *view.changed_edges):
        old = before.nodes.get(change.id) or before.edges[change.id]
        new = after.nodes.get(change.id) or after.edges[change.id]
        assert (change.before, change.after) == (old, new)
        differing = sorted(k for k in set(old.attrs) | set(new.attrs) if old.attrs.get(k) != new.attrs.get(k))
        assert [c.key for c in change.attrs] == differing
        for attr_change in change.attrs:
            assert attr_change.before == old.attrs.get(attr_change.key)
            assert attr_change.after == new.attrs.get(attr_change.key)
    assert tuple(c.id for c in view.changed_nodes) == ids.changed_nodes
    assert tuple(c.id for c in view.changed_edges) == ids.changed_edges


@given(graphs())
def test_diff_of_a_graph_with_itself_is_empty(graph: OpsGraph) -> None:
    view = graph_diff_view(1, 1, graph_diff(graph, graph), graph, graph)
    for part in (view.added_nodes, view.removed_nodes, view.changed_nodes):
        assert part == ()
    for part in (view.added_edges, view.removed_edges, view.changed_edges):
        assert part == ()


# --------------------------------------------------------------------------- download names

_FORBIDDEN = re.compile(r'[\\/:*?"<>|\x00-\x1f\x7f]')


@settings(max_examples=150)
@given(
    st.text(min_size=1, max_size=300),
    st.text(max_size=80),
    st.integers(min_value=1, max_value=10_000),
    st.sampled_from(["md", "html", "sarif"]),
)
def test_download_names_are_safe_file_names_and_headers(
    name: str, workspace_id: str, version: int, kind: str
) -> None:
    ascii_name, display = download_names(workspace_id, name, version, kind)  # type: ignore[arg-type]
    assert re.fullmatch(rf"jettae-report-[A-Za-z0-9._-]+-v{version}\.{kind}", ascii_name)
    assert display.startswith("제때옵스-증적리포트-")
    assert display.endswith(f"-v{version}.{kind}")
    assert not _FORBIDDEN.search(display)
    assert len(display) <= 160
    header = content_disposition("attachment", ascii_name, display)
    header.encode("latin-1")  # a header value must be Latin-1 (here: ASCII)
    star = re.search(r"filename\*=UTF-8''(\S+)$", header)
    assert star is not None
    assert unquote(star.group(1)) == display


def test_download_names_keep_korean_names_readable() -> None:
    ascii_name, display = download_names("ws-b7212db3a067507c", "별빛상점 (DMC 데모)", 3, "md")
    assert ascii_name == "jettae-report-ws-b7212db3a067507c-v3.md"
    assert display == "제때옵스-증적리포트-별빛상점 (DMC 데모)-v3.md"
    assert download_names("ws-1", "a/b:c\n\td", 1, "html")[1] == "제때옵스-증적리포트-a_b_c d-v1.html"
    assert (
        download_names("ws-1", "\u200b\u3164", 1, "sarif")[1] == "제때옵스-증적리포트-워크스페이스-v1.sarif"
    )
