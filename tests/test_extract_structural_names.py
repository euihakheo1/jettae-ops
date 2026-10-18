"""Structural readings of service-like names, alert hand-offs without services, and no-service notes.

Every runbook here is written for these tests (no third-party text). They pin generic principles
that hold for any organisation's runbooks read by the LLM extractor:

* a name a document gives both as a service and as a system is one service: steps stated for the
  system twin are steps of the service the findings are about (service wins, every span kept);
* a title that describes a state of an already-named component ("<X> is down", "<X> outage") is
  that component, never a second service; a heading that is a sentence is no service at all;
* the runbook of an alert in a document that names no service keeps what it hands off to an
  external source, whatever the extractor's ``any_service`` flag says;
* a document whose graph ends with no service always says so, with the reason — never silently
  empty.

The rule parser's output is sealed: the same extraction labelled as the rule parser's is assembled
exactly as before.
"""

from __future__ import annotations

from collections.abc import Sequence

from jettae.check.context import CheckContext
from jettae.check.registry import run_checks
from jettae.domain import (
    ANY_SERVICE_ALERT,
    UNNAMED_SERVICE_ALERT,
    CatalogService,
    DelegatedElement,
    Document,
    EdgeKind,
    EntityRef,
    ExtractedAttr,
    ExtractedDelegation,
    ExtractedEntity,
    ExtractedRelation,
    ExtractedSubject,
    Extraction,
    Extractor,
    Finding,
    FindingStatus,
    NodeKind,
    SubjectKind,
    edge_id,
    node_id,
)
from jettae.extract import AssemblyResult, assemble
from jettae.extract.markdown import heading_texts
from jettae.extract.surfaces import deployment_base, is_sentence, is_state_phrase, state_base, words
from jettae.ingest import load_bytes

NO_SERVICE = "검사 대상 없음 ≠ 정상"

TWIN_RUNBOOK = """\
# Inventory API

The Inventory API serves stock levels to the storefront.

## Recovery
- If the Inventory API stops answering, fail it over to the second zone.
"""
TWIN_SERVICE_QUOTE = "The Inventory API serves stock levels to the storefront."
TWIN_STEP = "fail it over to the second zone"
TWIN_STEP_QUOTE = "If the Inventory API stops answering, fail it over to the second zone."

STATE_RUNBOOK = """\
# Ledger API is down

The Ledger API records every payment.

## Mitigation
- Roll back the last ledger deploy.
"""
STATE_STEP = "Roll back the last ledger deploy"

SENTENCE_RUNBOOK = """\
# Checkout is not loading for anyone

Open the storefront in a private window and note the error page.
"""

ALERT_RUNBOOK = """\
# QueueDepthHigh

This alert fires when a worker queue grows faster than it drains.

Owners and escalation for the affected service are listed in the team directory.
"""
ALERT_SUBJECT_QUOTE = "This alert fires when a worker queue grows faster than it drains."
ALERT_HANDOFF_QUOTE = "Owners and escalation for the affected service are listed in the team directory."

ROLES_RUNBOOK = """\
# Incident roles

The incident lead coordinates the response.
"""

TOOLS_RUNBOOK = """\
# Build runner maintenance

The build runner pulls images from the artifact cache.
"""


def _doc(name: str, text: str) -> Document:
    return load_bytes(name, text.encode("utf-8"))


def _entity(kind: NodeKind, name: str, quote: str | None = None, **attrs: str) -> ExtractedEntity:
    extracted = tuple(ExtractedAttr(key=k, value=v, quote=quote or name) for k, v in attrs.items())
    return ExtractedEntity(kind=kind, name=name, quote=quote or name, attrs=extracted)


def _relation(
    kind: EdgeKind, src: tuple[NodeKind, str], dst: tuple[NodeKind, str], quote: str
) -> ExtractedRelation:
    return ExtractedRelation(
        kind=kind,
        src=EntityRef(kind=src[0], name=src[1]),
        dst=EntityRef(kind=dst[0], name=dst[1]),
        quote=quote,
    )


def _assemble(
    doc: Document,
    *,
    entities: Sequence[ExtractedEntity] = (),
    relations: Sequence[ExtractedRelation] = (),
    subjects: Sequence[ExtractedSubject] = (),
    delegations: Sequence[ExtractedDelegation] = (),
    extractor: Extractor = Extractor.LLM,
    catalog: Sequence[CatalogService] = (),
) -> AssemblyResult:
    extraction = Extraction(
        doc_id=doc.doc_id,
        extractor=extractor,
        entities=tuple(entities),
        relations=tuple(relations),
        subjects=tuple(subjects),
        delegations=tuple(delegations),
    )
    return assemble([extraction], [doc], catalog=catalog)


def _findings(result: AssemblyResult, catalog: Sequence[CatalogService] = ()) -> list[Finding]:
    return list(run_checks(CheckContext(graph=result.graph, catalog=tuple(catalog))))


# --------------------------------------------------------------------------- text helpers


def test_state_phrases_and_sentences_are_read_structurally() -> None:
    assert is_state_phrase(words("is down"))
    assert is_state_phrase(words("outage"))
    assert is_state_phrase(words("is not responding"))
    assert is_state_phrase(words("partial outage of the"))
    assert not is_state_phrase(words("is working"))
    assert not is_state_phrase(words("runbook"))
    assert not is_state_phrase(words("outage runbook"))
    assert not is_state_phrase([])
    known = {"ledger api"}.__contains__

    def is_known(name: str) -> bool:
        return known(name.casefold())

    assert state_base("Ledger API is down", is_known) == "Ledger API"
    assert state_base("Ledger API outage", is_known) == "Ledger API"
    assert state_base("Partial outage of the Ledger API", is_known) == "Ledger API"
    assert state_base("Ledger API outage runbook", is_known) is None  # the remainder is no state
    assert state_base("Search API is down", is_known) is None  # not a name the document gives
    assert state_base("Ledger API", is_known) is None
    assert is_sentence("Checkout is not loading for anyone")
    assert is_sentence("What to do when the queue has stalled")
    assert not is_sentence("Inventory API")
    assert not is_sentence("Ledger API outage")
    assert deployment_base("Ledger API nodes", is_known) == "Ledger API"
    assert deployment_base("Ledger API clusters", is_known) == "Ledger API"
    assert deployment_base("Ledger API cluster manager", is_known) is None
    assert deployment_base("Search API nodes", is_known) is None


def test_heading_texts_skip_fenced_comments() -> None:
    text = "# Title\n\n```\n# not a heading\n```\n## Section ##\nbody # not a heading\n"
    assert heading_texts(text) == ("Title", "Section")


# --------------------------------------------------------------------------- (1) service/system twins


def _twin(extractor: Extractor = Extractor.LLM) -> AssemblyResult:
    doc = _doc("inventory/api.md", TWIN_RUNBOOK)
    return _assemble(
        doc,
        entities=[
            _entity(NodeKind.SERVICE, "Inventory API", TWIN_SERVICE_QUOTE),
            _entity(NodeKind.SYSTEM, "Inventory API", "If the Inventory API stops answering"),
            _entity(NodeKind.ACTION, TWIN_STEP, action_type="failover"),
        ],
        relations=[
            _relation(
                EdgeKind.MITIGATED_BY,
                (NodeKind.SYSTEM, "Inventory API"),
                (NodeKind.ACTION, TWIN_STEP),
                TWIN_STEP_QUOTE,
            )
        ],
        extractor=extractor,
    )


def test_a_service_and_system_twin_is_one_service_that_keeps_the_systems_steps() -> None:
    result = _twin()
    graph = result.graph
    service = node_id(NodeKind.SERVICE, "Inventory API")
    assert service in graph.nodes
    assert node_id(NodeKind.SYSTEM, "Inventory API") not in graph.nodes
    action = node_id(NodeKind.ACTION, TWIN_STEP)
    assert edge_id(EdgeKind.MITIGATED_BY, service, action) in graph.edges
    # provenance kept: the service carries the line of its system twin as well
    lines = {s.line for s in graph.nodes[service].spans}
    assert {3, 6} <= lines
    rules = {f.rule_id for f in _findings(result) if f.service_id == service}
    assert "JT-PATH-040" not in rules  # the failover stated for the twin is the service's mitigation


def test_a_system_named_as_a_services_deployment_is_that_service() -> None:
    doc = _doc("inventory/api.md", TWIN_RUNBOOK.replace("stops answering", "nodes stop answering"))
    result = _assemble(
        doc,
        entities=[
            _entity(NodeKind.SERVICE, "Inventory API", TWIN_SERVICE_QUOTE),
            _entity(NodeKind.SYSTEM, "Inventory API nodes", "If the Inventory API nodes stop answering"),
            _entity(NodeKind.ACTION, TWIN_STEP, action_type="failover"),
        ],
        relations=[
            _relation(
                EdgeKind.MITIGATED_BY,
                (NodeKind.SYSTEM, "Inventory API nodes"),
                (NodeKind.ACTION, TWIN_STEP),
                TWIN_STEP_QUOTE.replace("stops answering", "nodes stop answering"),
            )
        ],
    )
    graph = result.graph
    service = node_id(NodeKind.SERVICE, "Inventory API")
    assert [n.id for n in graph.nodes_of(NodeKind.SERVICE, NodeKind.SYSTEM)] == [service]
    assert edge_id(EdgeKind.MITIGATED_BY, service, node_id(NodeKind.ACTION, TWIN_STEP)) in graph.edges


def test_a_system_without_a_service_of_its_name_stays_a_system() -> None:
    doc = _doc("inventory/api.md", TWIN_RUNBOOK)
    result = _assemble(
        doc,
        entities=[
            _entity(NodeKind.SERVICE, "storefront", "storefront"),
            _entity(NodeKind.SYSTEM, "Inventory API", "If the Inventory API stops answering"),
        ],
    )
    assert node_id(NodeKind.SYSTEM, "Inventory API") in result.graph.nodes
    assert node_id(NodeKind.SERVICE, "Inventory API") not in result.graph.nodes


def test_twins_from_the_rule_parser_stay_as_they_were() -> None:
    graph = _twin(Extractor.RULE).graph
    assert node_id(NodeKind.SERVICE, "Inventory API") in graph.nodes
    assert node_id(NodeKind.SYSTEM, "Inventory API") in graph.nodes


def test_a_catalog_name_keeps_the_catalog_kind() -> None:
    doc = _doc("inventory/api.md", TWIN_RUNBOOK)
    catalog = [CatalogService(name="Inventory API", kind="system")]
    result = _assemble(
        doc,
        entities=[
            _entity(NodeKind.SERVICE, "Inventory API", TWIN_SERVICE_QUOTE),
            _entity(NodeKind.SYSTEM, "Inventory API", "If the Inventory API stops answering"),
        ],
        catalog=catalog,
    )
    assert node_id(NodeKind.SYSTEM, "Inventory API") in result.graph.nodes
    assert node_id(NodeKind.SERVICE, "Inventory API") not in result.graph.nodes


# --------------------------------------------------------------------------- (2) state titles


def _state(title_kind: NodeKind = NodeKind.SERVICE, base_kind: NodeKind = NodeKind.SERVICE) -> AssemblyResult:
    doc = _doc("ledger/down.md", STATE_RUNBOOK)
    return _assemble(
        doc,
        entities=[
            _entity(title_kind, "Ledger API is down"),
            _entity(base_kind, "Ledger API", "The Ledger API records every payment."),
            _entity(NodeKind.ACTION, STATE_STEP, action_type="rollback"),
        ],
        relations=[
            _relation(
                EdgeKind.MITIGATED_BY,
                (title_kind, "Ledger API is down"),
                (NodeKind.ACTION, STATE_STEP),
                f"- {STATE_STEP}.",
            )
        ],
    )


def test_a_title_stating_a_components_state_is_that_component() -> None:
    result = _state()
    graph = result.graph
    services = [n.name for n in graph.nodes_of(NodeKind.SERVICE)]
    assert services == ["Ledger API"]
    service = node_id(NodeKind.SERVICE, "Ledger API")
    assert edge_id(EdgeKind.MITIGATED_BY, service, node_id(NodeKind.ACTION, STATE_STEP)) in graph.edges
    assert all("is down" not in (f.message + f.title) for f in _findings(result))


def test_a_state_title_never_turns_a_system_into_a_service() -> None:
    result = _state(base_kind=NodeKind.SYSTEM)
    graph = result.graph
    assert graph.nodes_of(NodeKind.SERVICE) == []
    system = node_id(NodeKind.SYSTEM, "Ledger API")
    assert edge_id(EdgeKind.MITIGATED_BY, system, node_id(NodeKind.ACTION, STATE_STEP)) in graph.edges
    [note] = result.notes
    assert NO_SERVICE in note
    assert "'Ledger API'" in note  # the reason names what the document did name


def test_a_sentence_heading_is_no_service_and_the_note_says_why() -> None:
    doc = _doc("checkout/blank.md", SENTENCE_RUNBOOK)
    result = _assemble(doc, entities=[_entity(NodeKind.SERVICE, "Checkout is not loading for anyone")])
    assert result.graph.nodes_of(NodeKind.SERVICE) == []
    assert result.grounding.dropped_invalid == 1
    assert any("문장으로 된 제목" in ex for ex in result.grounding.dropped_examples)
    [note] = result.notes
    assert NO_SERVICE in note
    assert "'Checkout is not loading for anyone'" in note
    assert _findings(result) == []


def test_a_heading_that_is_a_name_stays_a_service() -> None:
    doc = _doc("inventory/api.md", TWIN_RUNBOOK)
    result = _assemble(doc, entities=[_entity(NodeKind.SERVICE, "Inventory API")])
    assert [n.name for n in result.graph.nodes_of(NodeKind.SERVICE)] == ["Inventory API"]
    assert result.notes == ()


# --------------------------------------------------------------------------- (3) alert hand-offs


def _alert(any_service: bool) -> AssemblyResult:
    doc = _doc("alerts/queue-depth.md", ALERT_RUNBOOK)
    return _assemble(
        doc,
        entities=[_entity(NodeKind.TRIGGER, "QueueDepthHigh")],
        subjects=[
            ExtractedSubject(
                kind=SubjectKind.ALERT,
                name="QueueDepthHigh",
                any_service=any_service,
                quote=ALERT_SUBJECT_QUOTE,
            )
        ],
        delegations=[
            ExtractedDelegation(
                element=DelegatedElement.OWNER, target="team directory", quote=ALERT_HANDOFF_QUOTE
            ),
            ExtractedDelegation(
                element=DelegatedElement.ESCALATION, target="team directory", quote=ALERT_HANDOFF_QUOTE
            ),
        ],
    )


def test_an_alert_in_a_document_without_services_keeps_its_hand_offs_without_the_flag() -> None:
    result = _alert(any_service=False)
    trigger = result.graph.nodes[node_id(NodeKind.TRIGGER, "QueueDepthHigh")]
    assert trigger.attr(UNNAMED_SERVICE_ALERT) is True
    assert trigger.attr(ANY_SERVICE_ALERT) is None
    [note] = result.notes
    assert NO_SERVICE in note
    assert "감시하는 서비스가 적혀 있지 않아" in note
    assert "적용할 서비스나 알람이 문서에 없어" not in note
    findings = {f.rule_id: f for f in _findings(result)}
    assert {"JT-PATH-010", "JT-ESC-001"} <= set(findings)
    owner = findings["JT-PATH-010"]
    assert owner.status is FindingStatus.UNKNOWN
    assert owner.service_id is None
    assert "감시하는 서비스를 적지 않았습니다" in owner.message
    assert "두루 적용됩니다" not in owner.message
    assert "'team directory'" in owner.message


def test_a_flagged_any_service_alert_keeps_its_wording() -> None:
    result = _alert(any_service=True)
    trigger = result.graph.nodes[node_id(NodeKind.TRIGGER, "QueueDepthHigh")]
    assert trigger.attr(ANY_SERVICE_ALERT) is True
    owner = next(f for f in _findings(result) if f.rule_id == "JT-PATH-010")
    assert "특정 서비스가 아니라 경보를 낸 서비스에 두루 적용됩니다" in owner.message
    [note] = result.notes
    assert "적용된다고만 적혀 있어" in note


def test_an_unflagged_alert_in_a_document_with_services_is_not_generic() -> None:
    doc = _doc("alerts/queue-depth.md", ALERT_RUNBOOK + "\nThe worker queue service runs the jobs.\n")
    result = _assemble(
        doc,
        entities=[
            _entity(NodeKind.TRIGGER, "QueueDepthHigh"),
            _entity(NodeKind.SERVICE, "worker queue service"),
        ],
        subjects=[ExtractedSubject(kind=SubjectKind.ALERT, name="QueueDepthHigh", quote=ALERT_SUBJECT_QUOTE)],
    )
    trigger = result.graph.nodes[node_id(NodeKind.TRIGGER, "QueueDepthHigh")]
    assert trigger.attr(UNNAMED_SERVICE_ALERT) is None
    assert trigger.attr(ANY_SERVICE_ALERT) is None
    assert result.notes == ()


# --------------------------------------------------------------------------- (4) never silently empty


def test_a_document_of_roles_only_says_it_has_no_service() -> None:
    doc = _doc("process/roles.md", ROLES_RUNBOOK)
    result = _assemble(doc, entities=[_entity(NodeKind.ROLE, "incident lead")])
    [note] = result.notes
    assert NO_SERVICE in note
    assert "이유:" in note
    assert "역할·팀·연락 수단" in note


def test_a_document_of_systems_only_names_them_in_the_note() -> None:
    doc = _doc("tools/runner.md", TOOLS_RUNBOOK)
    result = _assemble(
        doc, entities=[_entity(NodeKind.SYSTEM, "build runner"), _entity(NodeKind.SYSTEM, "artifact cache")]
    )
    [note] = result.notes
    assert NO_SERVICE in note
    assert "시스템·도구('artifact cache', 'build runner')" in note


def test_an_empty_reading_still_leaves_a_note() -> None:
    doc = _doc("process/roles.md", ROLES_RUNBOOK)
    [note] = _assemble(doc).notes
    assert NO_SERVICE in note
    assert "하나도 읽지 못했습니다" in note


def test_rule_parser_runs_without_services_keep_their_sealed_notes() -> None:
    doc = _doc("process/roles.md", ROLES_RUNBOOK)
    result = _assemble(doc, entities=[_entity(NodeKind.ROLE, "incident lead")], extractor=Extractor.RULE)
    assert result.notes == ()


def test_a_catalog_service_means_there_is_something_to_check() -> None:
    doc = _doc("process/roles.md", ROLES_RUNBOOK)
    result = _assemble(
        doc,
        entities=[_entity(NodeKind.ROLE, "incident lead")],
        catalog=[CatalogService(name="storefront", tier=1)],
    )
    assert result.notes == ()
