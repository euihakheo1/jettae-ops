"""English runbooks: the rule parser's markdown pitfalls, document subjects and delegations.

Every runbook here is written for these tests (no third-party text). They pin generic principles
that hold for any organisation's English runbooks:

* a list of links to other pages (past incidents, related runbooks) names pages, not services;
* an identifier with "Service" glued inside ("OrderServiceLatency") is not the service "Order";
* an alert runbook *is* the statement "this alert detects the services it covers";
* "owners: see the service registry" hands the element off — it is neither stated nor missing.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from jettae.domain import (
    DelegatedElement,
    Document,
    EdgeKind,
    ExtractedDelegation,
    ExtractedEntity,
    ExtractedSubject,
    Extraction,
    Extractor,
    NodeKind,
    SubjectKind,
    delegation_key,
)
from jettae.extract import RuleExtractor, assemble
from jettae.extract.llm_extractor import (
    EXTRACTION_INSTRUCTIONS,
    FEW_SHOT_EXAMPLES,
    LlmDelegation,
    LlmGraphPayload,
    LlmSubject,
    NameScope,
    build_extraction_context,
    payload_to_extraction,
)
from jettae.ingest import load_bytes
from jettae.llm.schema import parse_output, to_strict_json_schema

LINKS_RUNBOOK = """\
# DiskFillingUp

This alert fires when a volume is more than 90% full.

## Past incidents
- [OrderServiceLatency during the sale](https://example.com/incidents/1)
- [Payments System outage](https://example.com/incidents/2)
- [Search Service][search-postmortem]

[search-postmortem]: https://example.com/incidents/3
"""

ALERT_RUNBOOK = """\
# CheckoutLatencyHigh

This alert fires when the p99 latency of the checkout service exceeds 2 seconds.
For ownership and escalation, refer to the service registry.
"""

ANY_SERVICE_RUNBOOK = """\
# QueueStalled

This alert fires for whichever service owns a stalled queue.
Known consumers are the billing service and the shipping service.
Owners are listed in the service registry.
"""


def _doc(name: str, text: str) -> Document:
    return load_bytes(name, text.encode("utf-8"))


def _services(extraction: Extraction) -> set[str]:
    return {e.name for e in extraction.entities if e.kind in (NodeKind.SERVICE, NodeKind.SYSTEM)}


# --------------------------------------------------------------------------- rule parser (b)


def test_link_titles_in_a_past_incidents_list_name_no_service() -> None:
    extraction = RuleExtractor().extract(_doc("alerts/disk.md", LINKS_RUNBOOK))
    assert _services(extraction) == set()


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        ("## KubeServiceApiserverDown", set()),  # an identifier, not "Kube" + "Service"
        ("## OrderServiceLatency", set()),
        ("## Billing Service", {"Billing"}),  # a separate word still names a service
        ("## 결제 서비스", {"결제"}),  # Korean headings are untouched
        ("## 결제서비스", {"결제"}),
    ],
)
def test_a_latin_service_suffix_must_be_a_word_of_its_own(heading: str, expected: set[str]) -> None:
    text = f"{heading}\n담당: 김민수\n"
    assert _services(RuleExtractor().extract(_doc("x.md", text))) == expected


def test_a_korean_bracket_header_still_opens_a_service_scope() -> None:
    text = "[결제 서비스]\n담당: 김민수 대리\n"
    extraction = RuleExtractor().extract(_doc("x.md", text))
    assert "결제" in _services(extraction)
    assert any(r.kind is EdgeKind.OWNED_BY for r in extraction.relations)


# --------------------------------------------------------------------------- payload conversion


def _scope(doc: Document) -> NameScope:
    return NameScope.build(f"{doc.title}\n{doc.text}")


def test_subject_and_delegations_survive_conversion_when_the_chunk_names_them() -> None:
    doc = _doc("alerts/checkout.md", ALERT_RUNBOOK)
    payload = LlmGraphPayload(
        document_subject=LlmSubject(
            kind=SubjectKind.ALERT,
            name="CheckoutLatencyHigh",
            services=("checkout service", "inventory service"),  # the second is not in the text
            quote="This alert fires when the p99 latency of the checkout service exceeds 2 seconds.",
        ),
        delegations=(
            LlmDelegation(
                element=DelegatedElement.OWNER,
                target="service registry",
                quote="For ownership and escalation, refer to the service registry.",
            ),
            LlmDelegation(  # names a target the text never mentions
                element=DelegatedElement.ESCALATION, target="PagerDuty schedule", quote="refer to"
            ),
            LlmDelegation(  # names a service the text never mentions
                element=DelegatedElement.ROLLBACK,
                target="service registry",
                applies_to="payments API",
                quote="refer to the service registry",
            ),
        ),
    )
    extraction = payload_to_extraction(payload, doc_id=doc.doc_id, scope=_scope(doc))
    assert extraction.subjects == (
        ExtractedSubject(
            kind=SubjectKind.ALERT,
            name="CheckoutLatencyHigh",
            services=("checkout service",),
            quote="This alert fires when the p99 latency of the checkout service exceeds 2 seconds.",
        ),
    )
    assert extraction.delegations == (
        ExtractedDelegation(
            element=DelegatedElement.OWNER,
            target="service registry",
            quote="For ownership and escalation, refer to the service registry.",
        ),
    )


def test_a_subject_the_chunk_does_not_name_is_dropped() -> None:
    doc = _doc("alerts/checkout.md", ALERT_RUNBOOK)
    payload = LlmGraphPayload(
        document_subject=LlmSubject(kind=SubjectKind.ALERT, name="DiskFillingUp", quote="This alert fires")
    )
    assert payload_to_extraction(payload, doc_id=doc.doc_id, scope=_scope(doc)).subjects == ()


def test_the_schema_admits_subject_and_delegations_under_strict_decoding() -> None:
    strict = to_strict_json_schema(LlmGraphPayload)
    assert {"document_subject", "delegations", "entities", "relations"} <= set(strict["properties"])
    _document, example = FEW_SHOT_EXAMPLES[2]
    assert parse_output(LlmGraphPayload, example.model_dump_json()) == example
    # older answers without the new keys still validate (absent = not stated)
    assert parse_output(LlmGraphPayload, '{"entities": [], "relations": []}') == LlmGraphPayload()


def test_the_english_example_quotes_its_document_and_converts_losslessly() -> None:
    document, payload = FEW_SHOT_EXAMPLES[2]
    assert payload.document_subject is not None
    assert payload.document_subject.quote in document
    assert all(d.quote in document and d.target in document for d in payload.delegations)
    extraction = payload_to_extraction(payload, doc_id="example", scope=NameScope.build(document))
    assert len(extraction.subjects) == 1
    assert len(extraction.delegations) == len(payload.delegations)
    assert len(extraction.entities) == len(payload.entities)
    assert len(extraction.relations) == len(payload.relations)
    # the link title in "Past incidents" is not extracted as a service
    assert not any(e.kind in (NodeKind.SERVICE, NodeKind.SYSTEM) for e in payload.entities)
    # a monitoring system that pages the on-call is a trigger -> role relation
    assert any(r.kind is EdgeKind.NOTIFIES and r.src_kind is NodeKind.TRIGGER for r in payload.relations)


def test_instructions_and_context_cover_english_runbooks() -> None:
    text = EXTRACTION_INSTRUCTIONS
    for phrase in (
        "document_subject",
        "delegations",
        "any_service",
        "link titles",
        "never translate",
        "the source of notifies is always a trigger",
    ):
        assert phrase in text
    context = build_extraction_context()
    assert "detection signal" in context
    assert "whom the signal pages first" in context


# --------------------------------------------------------------------------- assembly (c), (d)


def _llm(
    doc: Document,
    *,
    subjects: Sequence[ExtractedSubject] = (),
    delegations: Sequence[ExtractedDelegation] = (),
    entities: Sequence[ExtractedEntity] = (),
) -> Extraction:
    return Extraction(
        doc_id=doc.doc_id,
        extractor=Extractor.LLM,
        entities=tuple(entities),
        subjects=tuple(subjects),
        delegations=tuple(delegations),
    )


def test_an_alert_runbook_detects_the_services_it_names() -> None:
    doc = _doc("alerts/checkout.md", ALERT_RUNBOOK)
    subject = ExtractedSubject(
        kind=SubjectKind.ALERT,
        name="CheckoutLatencyHigh",
        services=("checkout service",),
        quote="This alert fires when the p99 latency of the checkout service exceeds 2 seconds.",
    )
    graph = assemble([_llm(doc, subjects=[subject])], [doc]).graph
    [edge] = graph.edges_of(EdgeKind.DETECTED_BY)
    assert graph.nodes[edge.src].name == "checkout service"
    assert graph.nodes[edge.dst].kind is NodeKind.TRIGGER
    assert graph.nodes[edge.dst].name == "CheckoutLatencyHigh"
    assert [s.quote for s in edge.spans] == [subject.quote]  # grounded on the stating sentence


def test_an_any_service_alert_detects_every_service_of_its_own_runbook_only() -> None:
    doc = _doc("alerts/queue.md", ANY_SERVICE_RUNBOOK)
    other = _doc("services/search.md", "# Search\nThe search service is owned by the search team.\n")
    subject = ExtractedSubject(
        kind=SubjectKind.ALERT,
        name="QueueStalled",
        any_service=True,
        quote="This alert fires for whichever service owns a stalled queue.",
    )
    mentions = [
        ExtractedEntity(kind=NodeKind.SERVICE, name="billing service", quote="billing service"),
        ExtractedEntity(kind=NodeKind.SERVICE, name="shipping service", quote="shipping service"),
    ]
    search = Extraction(
        doc_id=other.doc_id,
        extractor=Extractor.LLM,
        entities=(ExtractedEntity(kind=NodeKind.SERVICE, name="search service", quote="search service"),),
    )
    graph = assemble([_llm(doc, subjects=[subject], entities=mentions), search], [doc, other]).graph
    detected = sorted(graph.nodes[e.src].name for e in graph.edges_of(EdgeKind.DETECTED_BY))
    assert detected == ["billing service", "shipping service"]


def test_an_any_service_alert_without_services_leaves_a_note_not_a_link() -> None:
    doc = _doc("alerts/queue.md", ANY_SERVICE_RUNBOOK)
    subject = ExtractedSubject(
        kind=SubjectKind.ALERT,
        name="QueueStalled",
        any_service=True,
        quote="This alert fires for whichever service owns a stalled queue.",
    )
    result = assemble([_llm(doc, subjects=[subject])], [doc])
    assert result.graph.edges_of(EdgeKind.DETECTED_BY) == []
    assert [n.name for n in result.graph.nodes_of(NodeKind.TRIGGER)] == ["QueueStalled"]
    assert len(result.notes) == 1
    assert "'QueueStalled'" in result.notes[0]
    assert "검사 대상 없음 ≠ 정상" in result.notes[0]


def test_a_document_wide_delegation_lands_on_every_service_of_the_runbook() -> None:
    doc = _doc("alerts/queue.md", ANY_SERVICE_RUNBOOK)
    mentions = [
        ExtractedEntity(kind=NodeKind.SERVICE, name="billing service", quote="billing service"),
        ExtractedEntity(kind=NodeKind.SERVICE, name="shipping service", quote="shipping service"),
    ]
    delegation = ExtractedDelegation(
        element=DelegatedElement.OWNER,
        target="service registry",
        quote="Owners are listed in the service registry.",
    )
    graph = assemble([_llm(doc, entities=mentions, delegations=[delegation])], [doc]).graph
    key = delegation_key(DelegatedElement.OWNER)
    services = graph.nodes_of(NodeKind.SERVICE)
    assert [s.attr(key) for s in services] == ["service registry", "service registry"]
    assert all(s.attrs[key].spans[0].quote == delegation.quote for s in services)


def test_a_delegation_for_one_service_lands_on_that_service_only() -> None:
    doc = _doc("alerts/queue.md", ANY_SERVICE_RUNBOOK)
    mentions = [
        ExtractedEntity(kind=NodeKind.SERVICE, name="billing service", quote="billing service"),
        ExtractedEntity(kind=NodeKind.SERVICE, name="shipping service", quote="shipping service"),
    ]
    delegation = ExtractedDelegation(
        element=DelegatedElement.OWNER,
        target="service registry",
        applies_to="billing service",
        quote="Owners are listed in the service registry.",
    )
    graph = assemble([_llm(doc, entities=mentions, delegations=[delegation])], [doc]).graph
    key = delegation_key(DelegatedElement.OWNER)
    owners = {s.name: s.attr(key) for s in graph.nodes_of(NodeKind.SERVICE)}
    assert owners == {"billing service": "service registry", "shipping service": None}


def test_an_ungrounded_delegation_or_subject_is_dropped_and_counted() -> None:
    doc = _doc("alerts/queue.md", ANY_SERVICE_RUNBOOK)
    delegation = ExtractedDelegation(
        element=DelegatedElement.OWNER, target="service registry", quote="Owners live in the wiki."
    )
    subject = ExtractedSubject(kind=SubjectKind.ALERT, name="QueueStalled", any_service=True, quote="Nope.")
    result = assemble([_llm(doc, subjects=[subject], delegations=[delegation])], [doc])
    assert result.grounding.dropped_ungrounded == 2
    assert result.graph.nodes == {}


def test_rule_parser_extractions_carry_no_document_level_statements() -> None:
    extraction = RuleExtractor().extract(_doc("alerts/checkout.md", ALERT_RUNBOOK))
    assert extraction.subjects == ()
    assert extraction.delegations == ()
