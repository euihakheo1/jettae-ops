"""Benchmarks (성능·안전 시험): blast-radius speed and the prompt-injection self-test.

* :func:`impact_benchmark` times the KPI "센터 1곳 장애 가정 시 영향 서비스 목록 산출 시간" on a
  synthetic catalog of realistic size (200 services, 50 systems, 3 locations). The graph is
  seeded, so every run times the same work; the median of 20 runs is reported because a single
  timing on a shared machine mostly measures the machine.
* :func:`injection_selftest` checks P5 ("문서는 데이터, 지시가 아니다") end to end with the LLM
  OFF: each of the 30 cases plants one attack line in a demo runbook, and the diagnosis must be
  indistinguishable from the clean one — same graph structure, same findings, same score — with
  no element taken from the planted line. The guard's quarantine count is reported separately:
  a line the guard missed but that changed nothing is still safe, and is reported as such.

Why a *structure* hash for the injection test: inserting a line shifts every later character
offset and changes the document's id and SHA-256, so the raw content hash differs by
construction. :func:`structure_hash` hashes the graph with source locations removed (node and
edge ids, kinds, names, attribute values and sources, provenance kinds and verification), which
is exactly "the same graph, read from a longer file". Whether the quoted evidence also stayed the
same is reported separately (``evidence_unchanged``), because a shorter quote of the same fact
is not a change of what the graph says.
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics
import time
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict

from jettae.check.context import CheckContext
from jettae.check.impact import blast_radius
from jettae.check.registry import run_checks
from jettae.check.scoring import score
from jettae.domain import (
    Attr,
    Document,
    Edge,
    EdgeKind,
    Extractor,
    Finding,
    GraphBuilder,
    InputBundle,
    Node,
    NodeKind,
    OpsGraph,
    Provenance,
    SourceKind,
    edge_id,
    node_id,
)
from jettae.evalkit.conditions import offline_graph
from jettae.evalkit.gold import DEFAULT_SEED
from jettae.evalkit.injection import PLANTED_PERSON, build_injection_suite
from jettae.ingest import load_bundle_dir, load_bytes, normalize_text

__all__ = [
    "DEFAULT_BUNDLE",
    "IMPACT_RUNS",
    "ImpactBenchmark",
    "InjectionOutcome",
    "InjectionSelftest",
    "impact_benchmark",
    "injection_selftest",
    "structure_hash",
    "synthetic_catalog_graph",
]

DEFAULT_BUNDLE = "examples/dmc-shop"
IMPACT_RUNS = 20
#: Share of synthetic services that also depend on another (earlier) service.
_SERVICE_DEPENDENCY_SHARE = 0.3


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True)


def _published_path(root: Path) -> str:
    """``root`` as it may appear in published metrics: relative to the working directory when it
    lies under it, otherwise just its name — never an absolute path (user names, machine layout)."""
    if not root.is_absolute():
        return root.as_posix()
    try:
        return root.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return root.name


# --------------------------------------------------------------------------- blast radius


class ImpactBenchmark(_Model):
    n_services: int
    n_systems: int
    n_locations: int
    n_nodes: int
    n_edges: int
    failed: str
    impacted_services: int
    impacted_systems: int
    runs: int
    median_ms: float
    min_ms: float
    max_ms: float


def _location_name(index: int) -> str:
    return f"센터 {chr(ord('A') + index)}" if index < 26 else f"센터 {index + 1}"


def synthetic_catalog_graph(
    n_services: int = 200, n_systems: int = 50, n_locations: int = 3, *, seed: int = DEFAULT_SEED
) -> OpsGraph:
    """A catalog-shaped graph: systems run in locations, services run in a location and depend
    on 1–3 systems and sometimes on an earlier service (so failures chain). Seeded (P6)."""
    if min(n_services, n_systems, n_locations) < 1:
        raise ValueError("서비스·시스템·위치 수는 1 이상이어야 합니다")
    rng = random.Random(f"{seed}|impact|{n_services}|{n_systems}|{n_locations}")
    prov = (Provenance(source=SourceKind.CATALOG, extractor=Extractor.IMPORT, ref="synthetic"),)
    builder = GraphBuilder()

    def node(kind: NodeKind, name: str, tier: int | None = None) -> str:
        attrs = {} if tier is None else {"tier": Attr(value=tier, source=SourceKind.CATALOG)}
        nid = node_id(kind, name)
        builder.add_node(Node(id=nid, kind=kind, name=name, attrs=attrs, prov=prov))
        return nid

    def edge(kind: EdgeKind, src: str, dst: str) -> None:
        builder.add_edge(Edge(id=edge_id(kind, src, dst), kind=kind, src=src, dst=dst, prov=prov))

    locations = [node(NodeKind.LOCATION, _location_name(i)) for i in range(n_locations)]
    systems = [node(NodeKind.SYSTEM, f"system-{i:03d}") for i in range(n_systems)]
    for i, sys in enumerate(systems):
        edge(EdgeKind.RUNS_IN, sys, locations[i % n_locations])
    services: list[str] = []
    for i in range(n_services):
        sid = node(NodeKind.SERVICE, f"service-{i:03d}", tier=rng.choice((1, 2, 2, 3)))
        edge(EdgeKind.RUNS_IN, sid, rng.choice(locations))
        for sys in rng.sample(systems, k=min(len(systems), rng.randint(1, 3))):
            edge(EdgeKind.DEPENDS_ON, sid, sys)
        if services and rng.random() < _SERVICE_DEPENDENCY_SHARE:
            edge(EdgeKind.DEPENDS_ON, sid, rng.choice(services))
        services.append(sid)
    return builder.build()


def impact_benchmark(
    *,
    n_services: int = 200,
    n_systems: int = 50,
    n_locations: int = 3,
    runs: int = IMPACT_RUNS,
    seed: int = DEFAULT_SEED,
) -> ImpactBenchmark:
    """Median wall time of :func:`~jettae.check.impact.blast_radius` for the first location."""
    if runs < 1:
        raise ValueError("반복 횟수는 1 이상이어야 합니다")
    graph = synthetic_catalog_graph(n_services, n_systems, n_locations, seed=seed)
    failed = node_id(NodeKind.LOCATION, _location_name(0))
    timings: list[float] = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        blast_radius(graph, failed)
        timings.append((time.perf_counter_ns() - start) / 1e6)
    result = blast_radius(graph, failed)  # the (deterministic) answer every timed run computed
    return ImpactBenchmark(
        n_services=n_services,
        n_systems=n_systems,
        n_locations=n_locations,
        n_nodes=len(graph.nodes),
        n_edges=len(graph.edges),
        failed=failed,
        impacted_services=len(result.impacted_services),
        impacted_systems=len(result.impacted_systems),
        runs=runs,
        median_ms=round(statistics.median(timings), 3),
        min_ms=round(min(timings), 3),
        max_ms=round(max(timings), 3),
    )


# --------------------------------------------------------------------------- injection self-test


def structure_hash(graph: OpsGraph, *, quotes: bool = False) -> str:
    """SHA-256 of the graph without source locations (offsets, document ids, document refs).

    With ``quotes=True`` the quoted evidence text of every element and attribute is included as
    well. That variant is stricter than "the same graph": an extractor may pick a shorter quote
    for the same fact when a quarantined line now interrupts a multi-line passage it used to
    quote, which changes the evidence shown, not what the graph says.
    """

    def said(element: Node | Edge | Attr) -> list[str]:
        return sorted({s.quote for s in element.spans}) if quotes else []

    def attrs(element: Node | Edge) -> dict[str, object]:
        return {k: [a.value, a.source.value, said(a)] for k, a in sorted(element.attrs.items())}

    def prov(element: Node | Edge) -> list[tuple[str, str, str]]:
        return sorted({(p.source.value, p.extractor.value, p.ref or "") for p in element.prov})

    payload = {
        "nodes": [
            [n.id, n.kind.value, n.name, attrs(n), prov(n), n.verification.value, said(n)]
            for n in graph.nodes.values()
        ],
        "edges": [
            [e.id, e.kind.value, e.src, e.dst, attrs(e), prov(e), e.verification.value, said(e)]
            for e in graph.edges.values()
        ],
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class InjectionOutcome(_Model):
    case_id: str
    kind: str
    line_no: int
    #: the guard quarantined the whole planted line
    quarantined: bool
    structure_unchanged: bool
    #: also the same quoted evidence (informational: a shorter quote of the same fact is allowed)
    evidence_unchanged: bool
    findings_unchanged: bool
    score_unchanged: bool
    #: graph elements (node/edge, or one of their attributes) whose evidence overlaps the line
    elements_from_injection: int
    planted_person: bool
    unauthorised: bool


class InjectionSelftest(_Model):
    bundle: str
    document: str
    n: int
    llm: str
    baseline_score: float
    baseline_structure_hash: str
    quarantined: int
    unauthorised_changes: int
    evidence_changed: int
    cases: tuple[InjectionOutcome, ...]


def _finding_view(findings: list[Finding]) -> list[tuple[str, str | None, str, str, tuple[str, ...]]]:
    """Findings without ids or evidence offsets (both move with the planted line)."""
    return sorted(
        (f.rule_id, f.service_id, f.status.value, f.severity.value, f.subject_ids) for f in findings
    )


def _diagnose(documents: tuple[Document, ...], bundle: InputBundle) -> tuple[OpsGraph, list[Finding], float]:
    graph = offline_graph(
        documents, roster=bundle.roster, catalog=bundle.catalog, policy=bundle.policy, oncall=bundle.oncall
    )
    ctx = CheckContext(
        graph=graph, policy=bundle.policy, roster=bundle.roster, oncall=bundle.oncall, catalog=bundle.catalog
    )
    findings = run_checks(ctx)
    return graph, findings, score(ctx, findings).overall


#: Formats whose text is the file itself (offsets of the planted line survive loading).
_PLAIN_TEXT_SUFFIXES = frozenset({".md", ".markdown", ".txt"})


def _is_plain_text(path: str) -> bool:
    return PurePosixPath(path).suffix.lower() in _PLAIN_TEXT_SUFFIXES


def _from_range(graph: OpsGraph, doc_id: str, start: int, end: int) -> int:
    count = 0
    elements: list[Node | Edge] = [*graph.nodes.values(), *graph.edges.values()]
    for element in elements:
        spans = [*element.spans, *(s for a in element.attrs.values() for s in a.spans)]
        if any(s.doc_id == doc_id and s.start < end and s.end > start for s in spans):
            count += 1
    return count


def injection_selftest(
    base_bundle_dir: str | Path = DEFAULT_BUNDLE, n: int = 30, *, document: str | None = None
) -> InjectionSelftest:
    """Plant ``n`` attack lines (one per case) in one runbook of the bundle; LLM OFF.

    ``document`` selects the runbook by its bundle path (default: the first plain-text runbook in
    bundle order). Only plain-text formats qualify: the attack line is inserted into the loaded
    text and located by its offsets, which a PDF/DOCX/HTML conversion would neither keep nor
    map back. A case is an *unauthorised change* when the structure hash, the findings or the
    score differ from the clean run, when any element's evidence overlaps the planted line, or
    when the planted person appears in the graph.
    """
    root = Path(base_bundle_dir)
    bundle = load_bundle_dir(root)
    paths = [d.path or "" for d in bundle.documents]
    if document is None:
        target = next((p for p in paths if _is_plain_text(p)), None)
        if target is None:
            raise ValueError(f"주입 시험에 쓸 텍스트 런북(.md·.txt)이 묶음에 없습니다: {', '.join(paths)}")
    else:
        target = document
    if target not in paths:
        raise ValueError(f"묶음에 없는 런북입니다: {target} (있는 문서: {', '.join(paths)})")
    if not _is_plain_text(target):
        raise ValueError(f"주입 시험은 텍스트 런북(.md·.txt)에만 할 수 있습니다: {target}")
    index = paths.index(target)
    base_graph, base_findings, base_score = _diagnose(bundle.documents, bundle)
    base_hash = structure_hash(base_graph)
    base_evidence = structure_hash(base_graph, quotes=True)
    base_view = _finding_view(base_findings)
    # Plant into the text the baseline was diagnosed from, not a second read of the file:
    # ``Document.path`` is the NFC-normalised, PII-masked display name, which is not the on-disk
    # name of a bundle saved on macOS (NFD) or of a file whose name holds a phone number. The
    # loaded text is already normalised and masked, and both steps are idempotent, so reloading
    # it with one planted line changes nothing but that line.
    base_text = bundle.documents[index].text
    outcomes: list[InjectionOutcome] = []
    for case in build_injection_suite(base_text, n):
        doc = load_bytes(target, case.text.encode("utf-8"))
        start = len(normalize_text(case.text[: case.start]))
        end = len(normalize_text(case.text[: case.end]))
        documents = (*bundle.documents[:index], doc, *bundle.documents[index + 1 :])
        graph, findings, overall = _diagnose(documents, bundle)
        structure_same = structure_hash(graph) == base_hash
        findings_same = _finding_view(findings) == base_view
        score_same = overall == base_score
        leaked = _from_range(graph, doc.doc_id, start, end)
        planted = any(PLANTED_PERSON in node.name for node in graph.nodes.values())
        outcomes.append(
            InjectionOutcome(
                case_id=case.case_id,
                kind=case.kind.value,
                line_no=case.line_no,
                quarantined=any(qs <= start and end <= qe for qs, qe in doc.quarantined),
                structure_unchanged=structure_same,
                evidence_unchanged=structure_hash(graph, quotes=True) == base_evidence,
                findings_unchanged=findings_same,
                score_unchanged=score_same,
                elements_from_injection=leaked,
                planted_person=planted,
                unauthorised=not (structure_same and findings_same and score_same) or bool(leaked) or planted,
            )
        )
    return InjectionSelftest(
        bundle=_published_path(root),
        document=target,
        n=len(outcomes),
        llm="off",
        baseline_score=base_score,
        baseline_structure_hash=base_hash,
        quarantined=sum(o.quarantined for o in outcomes),
        unauthorised_changes=sum(o.unauthorised for o in outcomes),
        evidence_changed=sum(not o.evidence_unchanged for o in outcomes),
        cases=tuple(outcomes),
    )
