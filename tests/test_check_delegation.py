"""Elements a runbook hands off to an external source are 확인 불가, never 없음 and never met.

"Owners and escalation: see the service registry" states where a fact lives, not the fact. The
fixed rules must not call that a missing owner (the runbook did not forget it) nor count it as an
owner (nothing in the input proves one): the finding is UNKNOWN, cites the delegating sentence,
and the score is exactly what it would be without the sentence.
"""

from __future__ import annotations

from collections.abc import Sequence

from jettae.check.context import CheckContext
from jettae.check.elements import evaluate_service
from jettae.check.registry import run_checks
from jettae.check.scoring import score
from jettae.domain import (
    CatalogService,
    DelegatedElement,
    Document,
    EdgeKind,
    Element,
    ElementStatus,
    ExtractedDelegation,
    ExtractedEntity,
    ExtractedRelation,
    ExtractedSubject,
    Extraction,
    Extractor,
    Finding,
    FindingStatus,
    NodeKind,
    OpsGraph,
    RosterEntry,
    SubjectKind,
)
from jettae.domain.extraction import EntityRef
from jettae.extract import assemble
from jettae.ingest import load_bytes

RUNBOOK = """\
# CheckoutLatencyHigh

This alert fires when the p99 latency of the checkout service exceeds 2 seconds.
For ownership and escalation, refer to the service registry.
Contacts for the on-call rotation are in the team directory.
"""
OWNER_QUOTE = "For ownership and escalation, refer to the service registry."
CONTACT_QUOTE = "Contacts for the on-call rotation are in the team directory."
SUBJECT = ExtractedSubject(
    kind=SubjectKind.ALERT,
    name="CheckoutLatencyHigh",
    services=("checkout service",),
    quote="This alert fires when the p99 latency of the checkout service exceeds 2 seconds.",
)


def _doc() -> Document:
    return load_bytes("alerts/checkout.md", RUNBOOK.encode("utf-8"))


def _owner(element: DelegatedElement = DelegatedElement.OWNER) -> ExtractedDelegation:
    return ExtractedDelegation(element=element, target="service registry", quote=OWNER_QUOTE)


def _graph(
    delegations: Sequence[ExtractedDelegation] = (),
    *,
    relations: Sequence[ExtractedRelation] = (),
    entities: Sequence[ExtractedEntity] = (),
    roster: Sequence[RosterEntry] = (),
    catalog: Sequence[CatalogService] = (),
) -> OpsGraph:
    doc = _doc()
    extraction = Extraction(
        doc_id=doc.doc_id,
        extractor=Extractor.LLM,
        entities=tuple(entities),
        relations=tuple(relations),
        subjects=(SUBJECT,),
        delegations=tuple(delegations),
    )
    return assemble([extraction], [doc], roster=roster, catalog=catalog).graph


def _findings(graph: OpsGraph, rule_id: str, **ctx: object) -> list[Finding]:
    context = CheckContext(graph, **ctx)  # type: ignore[arg-type]
    return [f for f in run_checks(context) if f.rule_id == rule_id]


def _service_id(graph: OpsGraph) -> str:
    [service] = graph.nodes_of(NodeKind.SERVICE)
    return service.id


def test_the_alert_of_an_alert_runbook_is_the_services_detection_signal() -> None:
    graph = _graph()
    assert _findings(graph, "JT-PATH-001") == []  # 감지 경로 없음 no longer fires
    [edge] = graph.edges_of(EdgeKind.DETECTED_BY)
    assert graph.nodes[edge.dst].name == "CheckoutLatencyHigh"


def test_a_delegated_owner_is_unknown_not_missing() -> None:
    [finding] = _findings(_graph([_owner()]), "JT-PATH-010")
    assert finding.status is FindingStatus.UNKNOWN
    assert finding.title == "담당자 확인 불가"
    assert "외부 참조('service registry')" in finding.message
    assert OWNER_QUOTE in {s.quote for s in finding.evidence}


def test_without_the_delegation_the_owner_is_missing() -> None:
    [finding] = _findings(_graph(), "JT-PATH-010")
    assert finding.status is FindingStatus.BROKEN
    assert finding.title == "담당자 없음"


def test_a_delegation_never_earns_points() -> None:
    plain, delegated = _graph(), _graph([_owner(), _owner(DelegatedElement.FALLBACK_CHANNEL)])
    sid = _service_id(delegated)
    before = evaluate_service(CheckContext(plain), sid)
    after = evaluate_service(CheckContext(delegated), sid)
    assert after[Element.OWNER].status is ElementStatus.UNKNOWN
    assert after[Element.OWNER].score == before[Element.OWNER].score == 0.0
    assert after[Element.OWNER].reasons[0].startswith("담당자를 외부 참조('service registry')로 위임")
    assert OWNER_QUOTE in {s.quote for s in after[Element.OWNER].evidence}
    assert [e.score for e in after.values()] == [e.score for e in before.values()]
    plain_ctx, delegated_ctx = CheckContext(plain), CheckContext(delegated)
    assert (
        score(delegated_ctx, run_checks(delegated_ctx)).overall
        == score(plain_ctx, run_checks(plain_ctx)).overall
    )


def test_channels_of_a_service_whose_owner_is_delegated_are_unknown_too() -> None:
    [finding] = _findings(_graph([_owner()]), "JT-PATH-030")
    assert finding.status is FindingStatus.UNKNOWN
    assert finding.message.startswith(
        "'checkout service' 서비스: 연락 수단을 확인할 대응자를 알 수 없습니다."
    )
    element = evaluate_service(CheckContext(_graph([_owner()])), _service_id(_graph()))[
        Element.FALLBACK_CHANNEL
    ]
    assert element.status is ElementStatus.UNKNOWN


def test_delegated_contacts_make_the_fallback_finding_unknown() -> None:
    contacts = ExtractedDelegation(
        element=DelegatedElement.FALLBACK_CHANNEL, target="team directory", quote=CONTACT_QUOTE
    )
    [finding] = _findings(_graph([contacts]), "JT-PATH-030")
    assert finding.status is FindingStatus.UNKNOWN
    assert "외부 참조('team directory')" in finding.message
    assert CONTACT_QUOTE in {s.quote for s in finding.evidence}


def test_a_stated_owner_wins_over_a_delegation() -> None:
    owner = ExtractedRelation(
        kind=EdgeKind.OWNED_BY,
        src=EntityRef(kind=NodeKind.SERVICE, name="checkout service"),
        dst=EntityRef(kind=NodeKind.TEAM, name="checkout team"),
        quote="the checkout service",
    )
    graph = _graph([_owner()], relations=[owner])
    findings = _findings(graph, "JT-PATH-010")
    # the team resolves to nobody: the gap is the stated owner, reported as before (not "위임")
    assert [f.title for f in findings] == ["담당자 없음"]
    assert all("외부 참조" not in f.message for f in findings)


def test_an_imported_catalog_owner_resolves_the_delegation() -> None:
    roster = (RosterEntry(name="Dana Kim", team="checkout team"),)
    catalog = (CatalogService(name="checkout service", tier=1, owner="Dana Kim"),)
    graph = _graph([_owner()], roster=roster, catalog=catalog)
    assert _findings(graph, "JT-PATH-010", roster=roster, catalog=catalog) == []


def test_a_delegated_escalation_is_unknown_when_owners_exist() -> None:
    roster = (RosterEntry(name="Dana Kim"),)
    catalog = (CatalogService(name="checkout service", tier=1, owner="Dana Kim"),)
    plain = _findings(_graph(roster=roster, catalog=catalog), "JT-ESC-001", roster=roster, catalog=catalog)
    assert [f.status for f in plain] == [FindingStatus.BROKEN]
    delegated = _graph([_owner(DelegatedElement.ESCALATION)], roster=roster, catalog=catalog)
    [finding] = _findings(delegated, "JT-ESC-001", roster=roster, catalog=catalog)
    assert finding.status is FindingStatus.UNKNOWN
    assert finding.title == "상향 경로 확인 불가"
    assert finding.id == plain[0].id  # the same gap, only the verdict changes


def test_a_delegated_rollback_procedure_is_unknown() -> None:
    rollback = ExtractedDelegation(
        element=DelegatedElement.ROLLBACK, target="service registry", quote=OWNER_QUOTE
    )
    [finding] = _findings(_graph([rollback]), "JT-PATH-040")
    assert finding.status is FindingStatus.UNKNOWN
    assert finding.title == "롤백/완화 조치 확인 불가"


def test_delegated_paging_makes_a_trigger_without_targets_unknown() -> None:
    paging = ExtractedDelegation(
        element=DelegatedElement.DETECTION, target="team directory", quote=CONTACT_QUOTE
    )
    plain = _findings(_graph(), "JT-PATH-002")
    assert [f.status for f in plain] == [FindingStatus.BROKEN]
    [finding] = _findings(_graph([paging]), "JT-PATH-002")
    assert finding.status is FindingStatus.UNKNOWN
    assert finding.title == "감지 후 호출 대상 확인 불가"
