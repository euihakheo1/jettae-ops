"""Statements a runbook makes for a group of services or for the whole document, and evidence hygiene.

Every runbook here is written for these tests (no third-party text). They pin generic principles
that hold for any organisation's runbooks:

* a step stated for a list of services ("A, B and C ... move their traffic") is a step of *each*
  of them; a step stated for no service in particular (a resolution section) is a step of every
  service the document covers;
* a name the document only ever writes inside a fenced code block (a host in a command, a label in
  sample output) is not a service;
* the evidence of a service points at a line that names it, not at an unrelated relation sentence.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from jettae.check.context import CheckContext
from jettae.check.registry import run_checks
from jettae.domain import (
    CatalogService,
    Document,
    EdgeKind,
    EntityRef,
    ExtractedAttr,
    ExtractedEntity,
    ExtractedRelation,
    ExtractedSharedRelation,
    Extraction,
    Extractor,
    NodeKind,
    OpsGraph,
)
from jettae.extract import assemble
from jettae.extract.llm_extractor import (
    EXTRACTION_INSTRUCTIONS,
    FEW_SHOT_EXAMPLES,
    SHARED_RELATION_KINDS,
    LlmGraphPayload,
    LlmSharedRelation,
    NameScope,
    payload_to_extraction,
)
from jettae.extract.markdown import fenced_ranges, in_ranges
from jettae.ingest import load_bytes
from jettae.llm.schema import to_strict_json_schema

ZONE_RUNBOOK = """\
# Zone loss in the east region

The catalog service, the cart service and the checkout service run in the east region.
If a zone is lost, move their traffic to the remaining zones.

## Recovery
- Restore the image cache from the nightly snapshot.
- If the outage followed a deploy, roll the deploy back.
"""
TRIO = ("catalog service", "cart service", "checkout service")
MOVE = "move their traffic to the remaining zones"
MOVE_QUOTE = "If a zone is lost, move their traffic to the remaining zones."
ROLL = "roll the deploy back"
ROLL_QUOTE = "If the outage followed a deploy, roll the deploy back."

CODE_RUNBOOK = """\
# Cache warmup

The pricing service reads from the price cache.

```
$ warmup --target ledger-proxy --verbose
# the pricing service is warmed last
ok: ledger-proxy warmed
```
"""


def _doc(name: str, text: str) -> Document:
    return load_bytes(name, text.encode("utf-8"))


def _services(*names: str) -> list[ExtractedEntity]:
    return [ExtractedEntity(kind=NodeKind.SERVICE, name=n, quote=n) for n in names]


def _actions() -> list[ExtractedEntity]:
    return [
        ExtractedEntity(
            kind=NodeKind.ACTION,
            name=MOVE,
            quote=MOVE,
            attrs=(ExtractedAttr(key="action_type", value="failover", quote="move their traffic"),),
        ),
        ExtractedEntity(
            kind=NodeKind.ACTION,
            name=ROLL,
            quote=ROLL,
            attrs=(ExtractedAttr(key="action_type", value="rollback", quote=ROLL),),
        ),
    ]


def _shared(
    kind: EdgeKind, dst_kind: NodeKind, dst: str, quote: str, services: Sequence[str] = ()
) -> ExtractedSharedRelation:
    return ExtractedSharedRelation(
        kind=kind, dst=EntityRef(kind=dst_kind, name=dst), services=tuple(services), quote=quote
    )


def _graph(doc: Document, extraction: Extraction, *, catalog: Sequence[CatalogService] = ()) -> OpsGraph:
    return assemble([extraction], [doc], catalog=catalog).graph


def _zone(shared: Sequence[ExtractedSharedRelation] = (), extractor: Extractor = Extractor.LLM) -> OpsGraph:
    doc = _doc("recovery/zone.md", ZONE_RUNBOOK)
    extraction = Extraction(
        doc_id=doc.doc_id,
        extractor=extractor,
        entities=(*_services(*TRIO), *_actions()),
        shared=tuple(shared),
    )
    return _graph(doc, extraction)


def _mitigations(graph: OpsGraph) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for edge in graph.edges_of(EdgeKind.MITIGATED_BY):
        found.setdefault(graph.nodes[edge.src].name, set()).add(graph.nodes[edge.dst].name)
    return found


GROUP = _shared(EdgeKind.MITIGATED_BY, NodeKind.ACTION, MOVE, MOVE_QUOTE, TRIO)
WHOLE = _shared(EdgeKind.MITIGATED_BY, NodeKind.ACTION, ROLL, ROLL_QUOTE)


# --------------------------------------------------------------------------- group statements (1)


def test_a_step_stated_for_a_list_of_services_mitigates_each_of_them() -> None:
    graph = _zone([GROUP])
    assert _mitigations(graph) == {key: {MOVE} for key in TRIO}
    assert all(
        [s.quote for s in e.spans] == [MOVE_QUOTE] for e in graph.edges_of(EdgeKind.MITIGATED_BY)
    )  # every link is grounded on the stating sentence


def test_a_step_stated_for_no_service_mitigates_every_service_of_the_document() -> None:
    graph = _zone([GROUP, WHOLE])
    assert _mitigations(graph) == {key: {MOVE, ROLL} for key in TRIO}


def test_shared_mitigations_close_the_rollback_gap_that_would_otherwise_be_reported() -> None:
    def rollback_gaps(graph: OpsGraph) -> list[str]:
        findings = run_checks(CheckContext(graph), ["JT-PATH-040"])
        return sorted(graph.nodes[f.service_id].name for f in findings if f.service_id)

    assert rollback_gaps(_zone()) == sorted(TRIO)  # without the group statements: "없음" × 3
    assert rollback_gaps(_zone([GROUP, WHOLE])) == []


def test_a_document_wide_step_links_only_the_services_of_its_own_document() -> None:
    zone = _doc("recovery/zone.md", ZONE_RUNBOOK)
    other = _doc("services/search.md", "# Search\nThe search service answers queries.\n")
    extractions = [
        Extraction(
            doc_id=zone.doc_id,
            extractor=Extractor.LLM,
            entities=(*_services(*TRIO), *_actions()),
            shared=(WHOLE,),
        ),
        Extraction(doc_id=other.doc_id, extractor=Extractor.LLM, entities=tuple(_services("search service"))),
    ]
    graph = assemble(extractions, [zone, other]).graph
    assert set(_mitigations(graph)) == set(TRIO)


def test_a_document_wide_step_in_a_document_without_services_links_nothing() -> None:
    doc = _doc("recovery/zone.md", ZONE_RUNBOOK)
    extraction = Extraction(
        doc_id=doc.doc_id, extractor=Extractor.LLM, entities=tuple(_actions()), shared=(WHOLE,)
    )
    result = assemble([extraction], [doc])
    assert result.graph.edges_of(EdgeKind.MITIGATED_BY) == []
    assert result.grounding.dropped_invalid == 0  # nothing wrong was proposed; there was nothing to link


def test_a_group_member_the_document_names_as_a_system_stays_a_system() -> None:
    doc = _doc("recovery/zone.md", ZONE_RUNBOOK)
    runs_in = _shared(
        EdgeKind.RUNS_IN,
        NodeKind.LOCATION,
        "east region",
        "The catalog service, the cart service and the checkout service run in the east region.",
        ("catalog service", "image cache"),
    )
    extraction = Extraction(
        doc_id=doc.doc_id,
        extractor=Extractor.LLM,
        entities=(
            *_services("catalog service"),
            ExtractedEntity(kind=NodeKind.SYSTEM, name="image cache", quote="image cache"),
        ),
        shared=(runs_in,),
    )
    graph = _graph(doc, extraction)
    assert {n.name for n in graph.nodes_of(NodeKind.SERVICE)} == {"catalog service"}
    located = {graph.nodes[e.src].id for e in graph.edges_of(EdgeKind.RUNS_IN)}
    assert located == {"service:catalog-service", "system:image-cache"}


def test_shared_detection_signals_link_every_service_like_any_detection() -> None:
    doc = _doc("recovery/zone.md", ZONE_RUNBOOK)
    signal = _shared(EdgeKind.DETECTED_BY, NodeKind.TRIGGER, "zone is lost", MOVE_QUOTE)
    extraction = Extraction(
        doc_id=doc.doc_id,
        extractor=Extractor.LLM,
        entities=(
            *_services(*TRIO),
            ExtractedEntity(kind=NodeKind.TRIGGER, name="zone is lost", quote="zone is lost"),
        ),
        shared=(signal,),
    )
    graph = _graph(doc, extraction)
    assert sorted(graph.nodes[e.src].name for e in graph.edges_of(EdgeKind.DETECTED_BY)) == sorted(TRIO)
    assert run_checks(CheckContext(graph), ["JT-PATH-001"]) == []


# --------------------------------------------------------------------------- payload conversion


def _scope(doc: Document) -> NameScope:
    return NameScope.build(f"{doc.title}\n{doc.text}")


def _convert(*shared: LlmSharedRelation) -> tuple[ExtractedSharedRelation, ...]:
    doc = _doc("recovery/zone.md", ZONE_RUNBOOK)
    payload = LlmGraphPayload(shared_relations=shared)
    return payload_to_extraction(payload, doc_id=doc.doc_id, scope=_scope(doc)).shared


def _llm_shared(
    kind: EdgeKind, dst_kind: NodeKind, dst: str, services: tuple[str, ...] = ()
) -> LlmSharedRelation:
    return LlmSharedRelation(kind=kind, dst_kind=dst_kind, dst_name=dst, services=services, quote=MOVE_QUOTE)


def test_conversion_keeps_the_named_members_and_drops_unread_ones() -> None:
    [kept] = _convert(
        _llm_shared(EdgeKind.MITIGATED_BY, NodeKind.ACTION, MOVE, ("cart service", "payments gateway", " "))
    )
    assert kept.services == ("cart service",)  # the unnamed gateway and the blank are left out
    assert kept.dst == EntityRef(kind=NodeKind.ACTION, name=MOVE)


def test_conversion_never_widens_a_group_of_unread_services_to_the_whole_document() -> None:
    assert _convert(_llm_shared(EdgeKind.MITIGATED_BY, NodeKind.ACTION, MOVE, ("payments gateway",))) == ()


def test_conversion_drops_kinds_whose_source_is_not_a_service_and_wrong_targets() -> None:
    assert EdgeKind.ESCALATES_TO not in SHARED_RELATION_KINDS
    assert EdgeKind.MITIGATED_BY in SHARED_RELATION_KINDS
    assert _convert(_llm_shared(EdgeKind.ESCALATES_TO, NodeKind.ROLE, "on-call")) == ()
    assert _convert(_llm_shared(EdgeKind.MITIGATED_BY, NodeKind.TRIGGER, "zone is lost")) == ()
    assert (
        _convert(_llm_shared(EdgeKind.MITIGATED_BY, NodeKind.ACTION, "page the vendor")) == ()
    )  # unread label


def test_every_example_group_statement_names_entities_of_the_same_payload() -> None:
    for document, payload in FEW_SHOT_EXAMPLES:
        declared = {(e.kind, e.name) for e in payload.entities}
        for shared in payload.shared_relations:
            assert shared.quote in document
            assert (shared.dst_kind, shared.dst_name) in declared
            assert all((NodeKind.SERVICE, s) in declared for s in shared.services)


def test_instructions_describe_scopes_signals_and_templates_structurally() -> None:
    text = EXTRACTION_INSTRUCTIONS
    for phrase in (
        "shared_relations",
        "an empty list means the whole document",
        "Detection signals are not only paging alerts",
        "trigger's channel attribute",
        "a password-manager or vault entry",
        "a status-page or customer-notice template",
        "including a workaround",
    ):
        assert phrase in text
    # rules are stated as structure, not as phrases lifted from particular runbooks — in the
    # instructions and in the output schema, whose field and enum descriptions reach the model too
    schema = json.dumps(to_strict_json_schema(LlmGraphPayload), ensure_ascii=False)
    for seen_by_model in (text, schema):
        lowered = seen_by_model.lower()
        for lifted in (
            "'a specific service'",
            "'whichever service'",
            "refer to the service catalog",
            "refer to the service catalogue",
            "service catalogue",
        ):
            assert lifted not in lowered


# --------------------------------------------------------------------------- code blocks (5)


def _code(
    entities: Sequence[ExtractedEntity],
    relations: Sequence[ExtractedRelation] = (),
    *,
    extractor: Extractor = Extractor.LLM,
    catalog: Sequence[CatalogService] = (),
) -> tuple[OpsGraph, int]:
    doc = _doc("cache/warmup.md", CODE_RUNBOOK)
    extraction = Extraction(
        doc_id=doc.doc_id, extractor=extractor, entities=tuple(entities), relations=tuple(relations)
    )
    result = assemble([extraction], [doc], catalog=catalog)
    return result.graph, result.grounding.dropped_invalid


def test_fenced_ranges_cover_whole_blocks_and_unclosed_fences() -> None:
    text = "a\n```\nx\n```\nb\n~~~~\ny\n"
    ranges = fenced_ranges(text)
    assert [text[lo:hi] for lo, hi in ranges] == ["```\nx\n```", "~~~~\ny\n"]
    assert in_ranges(text.index("x"), text.index("x") + 1, ranges)
    assert not in_ranges(text.index("b"), text.index("b") + 1, ranges)


def test_a_name_written_only_inside_code_is_not_a_service() -> None:
    graph, dropped = _code(
        _services("pricing service", "ledger-proxy"),
        [
            ExtractedRelation(
                kind=EdgeKind.DEPENDS_ON,
                src=EntityRef(kind=NodeKind.SERVICE, name="pricing service"),
                dst=EntityRef(kind=NodeKind.SERVICE, name="ledger-proxy"),
                quote="$ warmup --target ledger-proxy --verbose",
            )
        ],
    )
    assert {n.name for n in graph.nodes_of(NodeKind.SERVICE)} == {"pricing service"}
    assert graph.edges == {}
    assert dropped == 2  # the mention and the relation


def test_a_name_also_written_in_prose_stays_a_service() -> None:
    graph, dropped = _code(_services("pricing service"))  # it is also in a code comment
    assert [n.name for n in graph.nodes_of(NodeKind.SERVICE)] == ["pricing service"]
    assert dropped == 0


def test_catalog_services_and_the_rule_parser_are_not_affected_by_code_blocks() -> None:
    catalog = (CatalogService(name="ledger-proxy", tier=1),)
    graph, _ = _code(_services("ledger-proxy"), catalog=catalog)
    assert "ledger-proxy" in {n.name for n in graph.nodes_of(NodeKind.SERVICE)}
    graph, dropped = _code(_services("ledger-proxy"), extractor=Extractor.RULE)
    assert [n.name for n in graph.nodes_of(NodeKind.SERVICE)] == ["ledger-proxy"]
    assert dropped == 0


# --------------------------------------------------------------------------- evidence (6)

OWNER_RUNBOOK = """\
# Billing

The billing service sends invoices at midnight.

## Ownership
It is owned by the ledger team.
"""
OWNER_QUOTE = "It is owned by the ledger team."


def _owner_graph(extractor: Extractor) -> OpsGraph:
    doc = _doc("billing.md", OWNER_RUNBOOK)
    relation = ExtractedRelation(
        kind=EdgeKind.OWNED_BY,
        src=EntityRef(kind=NodeKind.SERVICE, name="billing service"),
        dst=EntityRef(kind=NodeKind.TEAM, name="ledger team"),
        quote=OWNER_QUOTE,
    )
    return _graph(doc, Extraction(doc_id=doc.doc_id, extractor=extractor, relations=(relation,)))


def test_an_llm_endpoint_the_relation_quote_does_not_name_cites_its_own_mention() -> None:
    graph = _owner_graph(Extractor.LLM)
    [service] = graph.nodes_of(NodeKind.SERVICE)
    assert [(s.quote, s.line) for s in service.spans] == [("billing service", 3)]
    [edge] = graph.edges_of(EdgeKind.OWNED_BY)
    assert [s.quote for s in edge.spans] == [OWNER_QUOTE]  # the relation keeps its own sentence


def test_rule_parser_endpoints_keep_the_relation_quote() -> None:
    [service] = _owner_graph(Extractor.RULE).nodes_of(NodeKind.SERVICE)
    assert [s.quote for s in service.spans] == [OWNER_QUOTE]


def test_group_members_cite_the_line_that_names_them() -> None:
    graph = _zone([GROUP])
    for name in TRIO:
        [node] = [n for n in graph.nodes_of(NodeKind.SERVICE) if n.name == name]
        assert {s.line for s in node.spans} == {3}  # the sentence listing the three, not line 4
