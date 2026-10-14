"""merge_provenance: one linear pass over hashable frozen models, with exactly the result of the
original JSON-keyed implementation (same entries, same objects, same order — P6)."""

from __future__ import annotations

import random
import time
from collections.abc import Iterable
from pathlib import Path

import pytest

from jettae.domain import graph as graph_module
from jettae.domain.graph import Extractor, Provenance, SourceKind, SourceSpan, merge_provenance
from jettae.extract import RuleExtractor, assemble
from jettae.ingest import load_bundle_dir

DEMO = Path(__file__).resolve().parents[1] / "examples" / "dmc-shop"


def _merge_by_json(a: Iterable[Provenance], b: Iterable[Provenance]) -> tuple[Provenance, ...]:
    """The original implementation, kept as the specification."""
    seen: dict[str, Provenance] = {}
    for p in [*a, *b]:
        seen.setdefault(p.model_dump_json(), p)
    return tuple(seen.values())


def _provenance(rng: random.Random) -> Provenance:
    source = rng.choice(list(SourceKind))
    spans = tuple(
        SourceSpan(
            doc_id=rng.choice(["d1", "d2"]),
            start=(start := rng.randint(0, 5)),
            end=start + rng.randint(0, 3),
            quote=rng.choice(["김대리", "결제 서비스", ""]),
            line=rng.choice([None, 1, 2]),
        )
        for _ in range(rng.randint(1 if source is SourceKind.RUNBOOK else 0, 2))
    )
    return Provenance(
        source=source,
        extractor=rng.choice(list(Extractor)),
        spans=spans,
        ref=rng.choice([None, "row-1", "row-2"]),
    )


def test_same_entries_objects_and_order_as_the_json_keyed_original() -> None:
    rng = random.Random(20261014)
    for _ in range(300):
        a = [_provenance(rng) for _ in range(rng.randint(0, 12))]
        b = [_provenance(rng) for _ in range(rng.randint(0, 12))]
        b += rng.sample(a, k=min(len(a), 3))  # duplicates across the two inputs
        expected = _merge_by_json(a, b)
        merged = merge_provenance(a, b)
        assert merged == expected
        assert all(got is want for got, want in zip(merged, expected, strict=True))


def test_equal_values_built_separately_are_one_entry() -> None:
    def make() -> Provenance:
        span = SourceSpan(doc_id="d", start=1, end=4, quote="김대리", line=1)
        return Provenance(source=SourceKind.RUNBOOK, extractor=Extractor.RULE, spans=(span,))

    first, second = make(), make()
    assert first is not second
    assert merge_provenance([first], [second]) == (first,)
    assert merge_provenance([first], [second])[0] is first


def test_demo_graph_is_identical_with_the_json_keyed_original(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = load_bundle_dir(DEMO)
    extractor = RuleExtractor()
    extractions = [
        extractor.extract(d, roster=bundle.roster, catalog=bundle.catalog) for d in bundle.documents
    ]

    def build() -> graph_module.OpsGraph:
        return assemble(
            extractions,
            bundle.documents,
            roster=bundle.roster,
            catalog=bundle.catalog,
            oncall=bundle.oncall,
            policy=bundle.policy,
        ).graph

    current = build()
    monkeypatch.setattr(graph_module, "merge_provenance", _merge_by_json)
    reference = build()
    assert current.content_hash() == reference.content_hash()
    assert [n.prov for n in current.nodes.values()] == [n.prov for n in reference.nodes.values()]
    assert [e.prov for e in current.edges.values()] == [e.prov for e in reference.edges.values()]
    assert any(len(n.prov) > 1 for n in current.nodes.values())  # merges actually happened


def test_twenty_thousand_entries_merge_quickly() -> None:
    def span(i: int) -> SourceSpan:
        return SourceSpan(
            doc_id="doc", start=i, end=i + 10, quote="결제 서비스 담당 김대리", line=i // 80 + 1
        )

    entries = [
        Provenance(source=SourceKind.RUNBOOK, extractor=Extractor.RULE, spans=(span(i),))
        for i in range(20_000)
    ]
    started = time.perf_counter()
    merged = merge_provenance(entries[:12_000], [*entries[8_000:], *entries[:5_000]])
    assert time.perf_counter() - started < 2.0
    assert merged == tuple(entries)
