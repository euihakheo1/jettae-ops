"""GraphBuilder merges: linear provenance dedupe with exactly the original builder's results (P6).

The original builder re-ran :func:`merge_provenance` over everything an element had accumulated on
every merge — quadratic in the mentions of one entity. :class:`_ReferenceBuilder` keeps that code
as the specification; the current builder must agree with it after every single operation
(stored elements, their key order, the returned element, provenance objects and order) and on
whole assembled bundles (the demo, a long repeated runbook, evalkit-generated organisations).
"""

from __future__ import annotations

import importlib
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from jettae.domain import CatalogService, Document, OnCallShift, Policy, RosterEntry
from jettae.domain.graph import (
    Attr,
    Edge,
    EdgeKind,
    Extractor,
    GraphBuilder,
    Node,
    NodeKind,
    OpsGraph,
    Provenance,
    SourceKind,
    SourceSpan,
    Verification,
    edge_id,
    merge_provenance,
)
from jettae.evalkit import DEFAULT_SEED, STYLES, generate_gold_orgs, gold_facts, oracle_graph, render
from jettae.extract import RuleExtractor, assemble
from jettae.ingest import load_bundle_dir, load_bytes, load_catalog, load_policy, load_roster

DEMO = Path(__file__).resolve().parents[1] / "examples" / "dmc-shop"
#: ``jettae.extract.assemble`` the module (the package re-exports a function of the same name)
ASSEMBLE_MODULE = importlib.import_module("jettae.extract.assemble")
FACTS_MODULE = importlib.import_module("jettae.evalkit.facts")


class _ReferenceBuilder(GraphBuilder):
    """The builder's merge before the provenance index, kept verbatim as the specification."""

    def add_node(self, node: Node) -> Node:
        cur = self.nodes.get(node.id)
        if cur is None:
            self.nodes[node.id] = node
            return node
        merged = cur.model_copy(
            update={
                "attrs": self._merge_attrs(cur.attrs, node.attrs),
                "prov": merge_provenance(cur.prov, node.prov),
                "verification": Verification.HOLD
                if Verification.HOLD in (cur.verification, node.verification)
                else Verification.VERIFIED,
                "hold_reason": cur.hold_reason or node.hold_reason,
            }
        )
        self.nodes[node.id] = merged
        return merged

    def add_edge(self, edge: Edge) -> Edge:
        cur = self.edges.get(edge.id)
        if cur is None:
            self.edges[edge.id] = edge
            return edge
        merged = cur.model_copy(
            update={
                "attrs": self._merge_attrs(cur.attrs, edge.attrs),
                "prov": merge_provenance(cur.prov, edge.prov),
                "verification": Verification.HOLD
                if Verification.HOLD in (cur.verification, edge.verification)
                else Verification.VERIFIED,
                "hold_reason": cur.hold_reason or edge.hold_reason,
            }
        )
        self.edges[edge.id] = merged
        return merged

    def remove_edge(self, eid: str) -> None:
        self.edges.pop(eid, None)


# --------------------------------------------------------------------------- comparisons


def _same(got: Node | Edge, want: Node | Edge) -> None:
    """Equal down to the bytes, and the provenance entries are the very same objects."""
    assert got.model_dump_json() == want.model_dump_json()
    assert got.model_fields_set == want.model_fields_set
    assert len(got.prov) == len(want.prov)
    assert all(a is b for a, b in zip(got.prov, want.prov, strict=True))


def _same_state(got: GraphBuilder, want: GraphBuilder) -> None:
    assert list(got.nodes) == list(want.nodes)
    assert list(got.edges) == list(want.edges)
    for key, node in want.nodes.items():
        _same(got.nodes[key], node)
    for key, edge in want.edges.items():
        _same(got.edges[key], edge)


def _same_graph(got: OpsGraph, want: OpsGraph) -> None:
    """Byte-identical including element order (``canonical_json`` alone sorts the keys)."""
    assert got.model_dump_json() == want.model_dump_json()
    assert got.content_hash() == want.content_hash()


# --------------------------------------------------------------------------- random operations

_NODE_IDS = ("service:결제", "person:김대리", "role:팀장", "channel:slack")
_EDGE_ENDS = (
    ("service:결제", "person:김대리"),
    ("service:결제", "role:팀장"),
    ("person:김대리", "channel:slack"),
)
_ATTR_KEYS = ("status", "tier", "description")
_ATTR_SOURCES = (SourceKind.RUNBOOK, SourceKind.ROSTER, SourceKind.PATCH, SourceKind.POLICY)


def _span(start: int) -> SourceSpan:
    return SourceSpan(doc_id="d", start=start, end=start + 3, quote="김대리", line=start // 40 + 1)


def _runbook(start: int) -> Provenance:
    return Provenance(source=SourceKind.RUNBOOK, extractor=Extractor.RULE, spans=(_span(start),))


def _imported(ref: str) -> Provenance:
    return Provenance(source=SourceKind.ROSTER, extractor=Extractor.IMPORT, ref=ref)


class _Ops:
    """Random elements whose provenance repeats: a pooled object, or a fresh *equal* copy of one."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.makers: list[Callable[[], Provenance]] = []
        for _ in range(10):
            start, ref = self.rng.randint(0, 400), f"row-{self.rng.randint(0, 3)}"
            self.makers.append(
                (lambda s=start: _runbook(s)) if self.rng.random() < 0.6 else (lambda r=ref: _imported(r))
            )
        self.pooled = [make() for make in self.makers]

    def prov(self) -> tuple[Provenance, ...]:
        out: list[Provenance] = []
        for _ in range(self.rng.randint(0, 4)):
            i = self.rng.randrange(len(self.makers))
            out.append(self.pooled[i] if self.rng.random() < 0.6 else self.makers[i]())
        return tuple(out)

    def attrs(self) -> dict[str, Attr]:
        keys = self.rng.sample(_ATTR_KEYS, k=self.rng.randint(0, 2))
        return {
            k: Attr(value=self.rng.choice(["a", "b", 1]), source=self.rng.choice(_ATTR_SOURCES)) for k in keys
        }

    def flags(self) -> tuple[Verification, str | None]:
        if self.rng.random() < 0.2:
            return Verification.HOLD, self.rng.choice([None, "동명이인", "역번역 불일치"])
        return Verification.VERIFIED, None

    def node(self) -> Node:
        nid = self.rng.choice(_NODE_IDS)
        kind, name = nid.split(":", 1)
        verification, reason = self.flags()
        return Node(
            id=nid,
            kind=NodeKind(kind),
            name=name,
            attrs=self.attrs(),
            prov=self.prov(),
            verification=verification,
            hold_reason=reason,
        )

    def edge(self) -> Edge:
        src, dst = self.rng.choice(_EDGE_ENDS)
        verification, reason = self.flags()
        return Edge(
            id=edge_id(EdgeKind.OWNED_BY, src, dst),
            kind=EdgeKind.OWNED_BY,
            src=src,
            dst=dst,
            attrs=self.attrs(),
            prov=self.prov(),
            verification=verification,
            hold_reason=reason,
        )

    def step(self, new: GraphBuilder, ref: GraphBuilder) -> None:
        """One random operation, applied identically to both builders."""
        roll = self.rng.random()
        if roll < 0.4:
            node = self.node()
            _same(new.add_node(node), ref.add_node(node))
        elif roll < 0.75:
            edge = self.edge()
            _same(new.add_edge(edge), ref.add_edge(edge))
        elif roll < 0.82:  # read-modify-write of stored elements, the way tests flip a flag
            pick = self.rng.randrange(1_000)
            for b in (new, ref):
                if b.nodes:
                    key = list(b.nodes)[pick % len(b.nodes)]
                    b.nodes[key] = b.nodes[key].model_copy(update={"verification": Verification.HOLD})
                if b.edges:
                    key = list(b.edges)[pick % len(b.edges)]
                    b.edges[key] = b.edges[key].model_copy(update={"hold_reason": "수동 보류"})
        elif roll < 0.88:  # a different element stored over one: its own provenance is the truth
            node = self.node()
            for b in (new, ref):
                b.nodes[node.id] = node
        elif roll < 0.94:
            eid = self.edge().id
            for b in (new, ref):
                b.remove_edge(eid)
        else:  # direct deletes bypass the builder's methods
            nid, eid = self.rng.choice(_NODE_IDS), self.edge().id
            for b in (new, ref):
                b.nodes.pop(nid, None)
                b.edges.pop(eid, None)


@pytest.mark.parametrize("seed", range(40))
def test_every_operation_matches_the_original_builder(seed: int) -> None:
    ops = _Ops(seed)
    base: OpsGraph | None = None
    if seed % 2:  # also start from a built graph, as patching and rebasing do
        warmup = _ReferenceBuilder()
        for _ in range(30):
            warmup.add_node(ops.node())
            warmup.add_edge(ops.edge())
        base = warmup.build(validate=False)
    new, ref = GraphBuilder(base), _ReferenceBuilder(base)
    for _ in range(250):
        ops.step(new, ref)
        _same_state(new, ref)
    _same_graph(new.build(validate=False), ref.build(validate=False))


def _person(*prov: Provenance) -> Node:
    return Node(id="person:김대리", kind=NodeKind.PERSON, name="김대리", prov=prov)


def test_a_read_modify_write_keeps_the_accumulated_provenance() -> None:
    a, b, c = _imported("row-1"), _imported("row-2"), _imported("row-3")
    builder = GraphBuilder()
    builder.add_node(_person(a))
    builder.add_node(_person(b))
    builder.nodes["person:김대리"] = builder.nodes["person:김대리"].model_copy(
        update={"verification": Verification.HOLD}
    )
    merged = builder.add_node(_person(c, a))
    assert merged.prov == (a, b, c)
    assert merged.verification is Verification.HOLD
    assert builder.nodes["person:김대리"] is merged


def test_a_replaced_element_starts_over_from_its_own_provenance() -> None:
    a, b, c = _imported("row-1"), _imported("row-2"), _imported("row-3")
    builder = GraphBuilder()
    builder.add_node(_person(a))
    builder.add_node(_person(b))
    builder.nodes["person:김대리"] = _person(c, c)  # stored as given, duplicates and all
    assert builder.add_node(_person(a)).prov == (c, a)
    builder.remove_edge("missing")  # a no-op, like before
    assert builder.build(validate=False).nodes["person:김대리"].prov == (c, a)


# --------------------------------------------------------------------------- assembled bundles


@dataclass(frozen=True)
class _Inputs:
    documents: Sequence[Document]
    roster: Sequence[RosterEntry] = ()
    catalog: Sequence[CatalogService] = ()
    oncall: Sequence[OnCallShift] = ()
    policy: Policy = Policy()  # noqa: RUF009 - an immutable value object

    def assembled(self) -> OpsGraph:
        extractor = RuleExtractor()
        extractions = [extractor.extract(d, roster=self.roster, catalog=self.catalog) for d in self.documents]
        return assemble(
            extractions,
            self.documents,
            roster=self.roster,
            catalog=self.catalog,
            oncall=self.oncall,
            policy=self.policy,
        ).graph


@pytest.fixture
def use_reference_builder(monkeypatch: pytest.MonkeyPatch) -> Callable[[], None]:
    """Switch the assembler and the evalkit oracle to the original builder (undone after the test)."""

    def use() -> None:
        monkeypatch.setattr(ASSEMBLE_MODULE, "GraphBuilder", _ReferenceBuilder)
        monkeypatch.setattr(FACTS_MODULE, "GraphBuilder", _ReferenceBuilder)

    return use


def _demo(copies: int) -> _Inputs:
    bundle = load_bundle_dir(DEMO)
    documents: Sequence[Document] = bundle.documents
    if copies > 1:  # one long runbook naming the same people in every copy: many merges per element
        text = "\n\n".join(d.text.replace("## ", f"## {i}-") for i in range(copies) for d in bundle.documents)
        documents = (load_bytes("repeated.md", text.encode("utf-8")),)
    return _Inputs(documents, bundle.roster, bundle.catalog, bundle.oncall, bundle.policy)


@pytest.mark.parametrize("copies", [1, 6])
def test_demo_graphs_are_byte_identical_to_the_original_builder(
    copies: int, use_reference_builder: Callable[[], None]
) -> None:
    inputs = _demo(copies)
    current = inputs.assembled()
    use_reference_builder()
    reference = inputs.assembled()
    _same_graph(current, reference)
    assert max(len(n.prov) for n in current.nodes.values()) >= 2 * copies  # merges actually happened
    if copies == 1:  # the published demo graph (docs/demo-baseline.md)
        assert current.content_hash().startswith("883b5d40d60c")


def test_generated_organisations_are_byte_identical_to_the_original_builder(
    use_reference_builder: Callable[[], None],
) -> None:
    orgs = generate_gold_orgs(3)
    cases: list[_Inputs] = []
    for org in orgs:
        for style in STYLES:
            case = render(org, style=style, position="front", focus_service=None, seed=DEFAULT_SEED)
            cases.append(
                _Inputs(
                    (load_bytes("runbook.md", case.text.encode("utf-8")),),
                    roster=load_roster(case.roster_csv.encode("utf-8")),
                    catalog=load_catalog(case.catalog_yaml.encode("utf-8")),
                    policy=load_policy(case.policy_yaml.encode("utf-8")),
                )
            )

    def graphs() -> list[OpsGraph]:
        assembled = [case.assembled() for case in cases]
        oracles = [
            oracle_graph(gold_facts(org), roster=case.roster, catalog=case.catalog)
            for org, case in zip(orgs, cases[:: len(STYLES)], strict=True)
        ]
        return assembled + oracles

    current = graphs()
    use_reference_builder()
    reference = graphs()
    assert len(current) == len(reference) == 3 * len(STYLES) + 3
    for got, want in zip(current, reference, strict=True):
        _same_graph(got, want)
    assert any(len(n.prov) > 1 for g in current for n in g.nodes.values())


# --------------------------------------------------------------------------- cost


def test_merges_hash_each_incoming_provenance_a_bounded_number_of_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linear dedupe: k merges into one element hash O(k) provenance (the old merge: ~k²/2)."""
    k = 2_000
    entries = [_runbook(i) for i in range(k)]
    calls = 0
    original = Provenance.__hash__

    def counting(self: Provenance) -> int:
        nonlocal calls
        calls += 1
        return original(self)

    monkeypatch.setattr(Provenance, "__hash__", counting)
    builder = GraphBuilder()
    for entry in entries:
        builder.add_node(_person(entry, entries[0]))  # one new entry and one repeat per mention
    assert builder.nodes["person:김대리"].prov == tuple(entries)
    assert calls <= 4 * k


def test_ten_thousand_mentions_of_one_person_merge_quickly() -> None:
    entries = [_runbook(i) for i in range(10_000)]
    mentions = [_person(entry) for entry in entries]
    builder = GraphBuilder()
    started = time.perf_counter()
    for mention in mentions:
        builder.add_node(mention)
    elapsed = time.perf_counter() - started
    assert builder.build(validate=False).nodes["person:김대리"].prov == tuple(entries)
    assert elapsed < 5.0  # about 0.3 s here; the re-merging builder took about 40 s
