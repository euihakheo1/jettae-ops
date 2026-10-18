"""Detection signals that are not paging alerts, paging tools without a named recipient, and the
hand-offs of an alert runbook that names no service.

Every runbook here is written for these tests (no third-party text). The principles they pin hold
for any organisation's runbooks:

* a signal is any stated way a failure becomes visible — a notification posted to a channel
  counts as detection, but it pages nobody, and the finding says where it goes;
* "raised as a page in <tool>" with nobody named leaves the recipients to the tool's routing:
  확인 불가, never 호출 대상 없음 — a named recipient still decides the verdict;
* an alert runbook that applies to whichever service raised it has no service to check (a note
  says so), but what it hands off to an external source is reported as 확인 불가, not dropped.
"""

from __future__ import annotations

from collections.abc import Sequence

from jettae.check.context import CheckContext
from jettae.check.elements import evaluate_service, generic_alerts
from jettae.check.registry import run_checks
from jettae.check.scoring import score
from jettae.domain import (
    ANY_SERVICE_ALERT,
    DelegatedElement,
    Document,
    EdgeKind,
    Element,
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
    OpsGraph,
    SubjectKind,
    delegation_key,
)
from jettae.extract import AssemblyResult, assemble
from jettae.ingest import load_bytes

SYNC_RUNBOOK = """\
# Inventory sync

The inventory service copies stock levels every minute.
SyncLagWarning posts to the #inventory-alerts channel.
"""
SYNC_DETECT = "SyncLagWarning posts to the #inventory-alerts channel."

WEBHOOK_RUNBOOK = """\
# Webhook delivery

The webhook service delivers order events to partners.
WebhookFailuresHigh is raised as an Opsgenie page.
It pages the integrations on-call.
"""
WEBHOOK_DETECT = "WebhookFailuresHigh is raised as an Opsgenie page."

DISK_RUNBOOK = """\
# DiskPressure

This alert fires when a node's disk is more than 90% full.
It is not tied to one service; the labels name the affected host.
It is raised as an Opsgenie page.
Owners and escalation contacts are kept in the infrastructure wiki.
"""
DISK_ANY = "It is not tied to one service; the labels name the affected host."
DISK_WIKI = "Owners and escalation contacts are kept in the infrastructure wiki."


def _doc(name: str, text: str) -> Document:
    return load_bytes(name, text.encode("utf-8"))


def _assemble(
    doc: Document,
    *,
    entities: Sequence[ExtractedEntity] = (),
    relations: Sequence[ExtractedRelation] = (),
    subjects: Sequence[ExtractedSubject] = (),
    delegations: Sequence[ExtractedDelegation] = (),
) -> AssemblyResult:
    extraction = Extraction(
        doc_id=doc.doc_id,
        extractor=Extractor.LLM,
        entities=tuple(entities),
        relations=tuple(relations),
        subjects=tuple(subjects),
        delegations=tuple(delegations),
    )
    return assemble([extraction], [doc])


def _findings(graph: OpsGraph, *rule_ids: str) -> list[Finding]:
    return run_checks(CheckContext(graph), rule_ids or None)


def _trigger(name: str, channel: str | None) -> ExtractedEntity:
    attrs = (ExtractedAttr(key="channel", value=channel, quote=channel),) if channel else ()
    return ExtractedEntity(kind=NodeKind.TRIGGER, name=name, quote=name, attrs=attrs)


def _detects(service: str, trigger: str, quote: str) -> ExtractedRelation:
    return ExtractedRelation(
        kind=EdgeKind.DETECTED_BY,
        src=EntityRef(kind=NodeKind.SERVICE, name=service),
        dst=EntityRef(kind=NodeKind.TRIGGER, name=trigger),
        quote=quote,
    )


# --------------------------------------------------------------------------- channel notifications (2)


def _sync(channel: str | None = "#inventory-alerts") -> OpsGraph:
    doc = _doc("inventory/sync.md", SYNC_RUNBOOK)
    entities = [
        ExtractedEntity(kind=NodeKind.SERVICE, name="inventory service", quote="inventory service"),
        _trigger("SyncLagWarning", channel),
    ]
    relations = [_detects("inventory service", "SyncLagWarning", SYNC_DETECT)]
    return _assemble(doc, entities=entities, relations=relations).graph


def test_a_notification_posted_to_a_channel_is_a_detection_signal() -> None:
    graph = _sync()
    assert _findings(graph, "JT-PATH-001") == []
    [service] = graph.nodes_of(NodeKind.SERVICE)
    detection = evaluate_service(CheckContext(graph), service.id)[Element.DETECTION]
    assert detection.score == 0.5  # a signal, but nobody is paged


def test_a_channel_only_signal_pages_nobody_and_the_finding_says_where_it_goes() -> None:
    [finding] = _findings(_sync(), "JT-PATH-002")
    assert finding.status is FindingStatus.BROKEN
    assert finding.title == "감지 후 호출 대상 없음"
    assert finding.message == (
        "'inventory service' 서비스의 감지 신호 'SyncLagWarning'은(는) '#inventory-alerts'에 게시될 뿐 "
        "호출받는 대응자가 적혀 있지 않습니다. 그 채널을 지켜보는 사람이 없으면 아무도 알아채지 못합니다."
    )
    assert "channel:inventory-alerts" in finding.subject_ids


def test_without_a_stated_channel_the_message_is_unchanged() -> None:
    [finding] = _findings(_sync(channel=None), "JT-PATH-002")
    assert finding.message == (
        "'inventory service' 서비스의 감지 신호 'SyncLagWarning'이(가) 울려도 호출 대상이 적혀 있지 "
        "않습니다. 알람이 아무에게도 닿지 않습니다."
    )


# --------------------------------------------------------------------------- paging tools (3)


def _webhook(*, notify: bool = False, channel: str | None = "Opsgenie") -> OpsGraph:
    doc = _doc("partners/webhooks.md", WEBHOOK_RUNBOOK)
    entities = [
        ExtractedEntity(kind=NodeKind.SERVICE, name="webhook service", quote="webhook service"),
        _trigger("WebhookFailuresHigh", channel),
    ]
    relations = [_detects("webhook service", "WebhookFailuresHigh", WEBHOOK_DETECT)]
    if notify:
        relations.append(
            ExtractedRelation(
                kind=EdgeKind.NOTIFIES,
                src=EntityRef(kind=NodeKind.TRIGGER, name="WebhookFailuresHigh"),
                dst=EntityRef(kind=NodeKind.ROLE, name="integrations on-call"),
                quote="It pages the integrations on-call.",
            )
        )
    return _assemble(doc, entities=entities, relations=relations).graph


def test_a_paging_tool_without_a_named_recipient_is_unknown_not_missing() -> None:
    graph = _webhook()
    [trigger] = graph.nodes_of(NodeKind.TRIGGER)
    assert trigger.attr(delegation_key(DelegatedElement.DETECTION)) == "Opsgenie"
    [finding] = _findings(graph, "JT-PATH-002")
    assert finding.status is FindingStatus.UNKNOWN
    assert finding.title == "감지 후 호출 대상 확인 불가"
    assert (
        "'WebhookFailuresHigh'은(는) 호출 도구로 발송되지만 받는 사람이 적혀 있지 않습니다."
        in finding.message
    )
    assert "외부 참조('Opsgenie')" in finding.message
    assert "Opsgenie" in {s.quote for s in finding.evidence}


def test_the_hand_off_to_a_paging_tool_never_earns_points() -> None:
    delegated, plain = _webhook(), _webhook(channel=None)
    [service] = delegated.nodes_of(NodeKind.SERVICE)
    after = evaluate_service(CheckContext(delegated), service.id)
    before = evaluate_service(CheckContext(plain), service.id)
    assert [e.score for e in after.values()] == [e.score for e in before.values()]
    assert after[Element.DETECTION].score == 0.5


def test_a_named_recipient_decides_the_verdict_over_the_paging_tool() -> None:
    [finding] = _findings(_webhook(notify=True), "JT-PATH-002")
    assert finding.status is FindingStatus.BROKEN  # the stated on-call resolves to nobody (no roster)
    assert "'integrations on-call'" in finding.message
    assert "외부 참조" not in finding.message


def test_a_chat_channel_is_not_a_paging_tool() -> None:
    [trigger] = _sync().nodes_of(NodeKind.TRIGGER)
    assert trigger.attr(delegation_key(DelegatedElement.DETECTION)) is None


def test_rule_parser_triggers_get_no_paging_hand_off() -> None:
    doc = _doc("partners/webhooks.md", WEBHOOK_RUNBOOK)
    extraction = Extraction(
        doc_id=doc.doc_id, extractor=Extractor.RULE, entities=(_trigger("WebhookFailuresHigh", "Opsgenie"),)
    )
    [trigger] = assemble([extraction], [doc]).graph.nodes_of(NodeKind.TRIGGER)
    assert trigger.attr(delegation_key(DelegatedElement.DETECTION)) is None


# --------------------------------------------------------------------------- service-agnostic alerts (9)


def _disk(
    delegations: Sequence[DelegatedElement] = (DelegatedElement.OWNER, DelegatedElement.ESCALATION),
) -> AssemblyResult:
    doc = _doc("alerts/disk.md", DISK_RUNBOOK)
    subject = ExtractedSubject(kind=SubjectKind.ALERT, name="DiskPressure", any_service=True, quote=DISK_ANY)
    entities = [_trigger("DiskPressure", "Opsgenie")]
    handed = [
        ExtractedDelegation(element=e, target="infrastructure wiki", quote=DISK_WIKI) for e in delegations
    ]
    return _assemble(doc, entities=entities, subjects=[subject], delegations=handed)


def test_an_alert_that_names_no_service_still_leaves_its_note() -> None:
    result = _disk()
    assert result.graph.nodes_of(NodeKind.SERVICE) == []
    [note] = result.notes
    assert note.startswith("'DiskPressure' 알람 런북은")
    assert "검사 대상 없음 ≠ 정상" in note
    assert "그 요소만 확인 불가로 따로 알립니다" in note


def test_the_hand_offs_of_a_service_agnostic_alert_are_reported_unknown() -> None:
    graph = _disk().graph
    [alert] = generic_alerts(CheckContext(graph))
    assert alert.attr(ANY_SERVICE_ALERT) is True
    assert alert.attr(delegation_key(DelegatedElement.OWNER)) == "infrastructure wiki"
    findings = {f.rule_id: f for f in _findings(graph)}
    assert set(findings) == {"JT-PATH-002", "JT-PATH-010", "JT-PATH-030", "JT-ESC-001"}
    assert all(f.status is FindingStatus.UNKNOWN and f.service_id is None for f in findings.values())
    owner = findings["JT-PATH-010"]
    assert owner.title == "담당자 확인 불가"
    assert owner.message.startswith(
        "'DiskPressure' 알람 런북은 특정 서비스가 아니라 경보를 낸 서비스에 두루 적용됩니다."
    )
    assert "외부 참조('infrastructure wiki')" in owner.message
    assert DISK_WIKI in {s.quote for s in owner.evidence}
    assert findings["JT-ESC-001"].title == "상향 경로 확인 불가"
    # channels need someone to reach; the owner being handed off leaves them unknown too
    assert "연락 수단을 확인할 대응자를 알 수 없습니다." in findings["JT-PATH-030"].message
    assert "외부 참조('Opsgenie')" in findings["JT-PATH-002"].message


def test_an_unlinked_trigger_of_an_ordinary_runbook_is_not_a_service_agnostic_alert() -> None:
    doc = _doc("partners/webhooks.md", WEBHOOK_RUNBOOK)
    entities = [
        ExtractedEntity(kind=NodeKind.SERVICE, name="webhook service", quote="webhook service"),
        _trigger("WebhookFailuresHigh", "Opsgenie"),  # carries the paging hand-off, detects nothing
    ]
    graph = _assemble(doc, entities=entities).graph
    assert generic_alerts(CheckContext(graph)) == []
    findings = _findings(graph, "JT-PATH-001", "JT-PATH-002")
    assert [(f.rule_id, f.service_id) for f in findings] == [("JT-PATH-001", "service:webhook-service")]


def test_elements_a_service_agnostic_alert_does_not_hand_off_are_not_reported() -> None:
    graph = _disk(delegations=()).graph
    rules = {f.rule_id for f in _findings(graph)}
    assert rules == {"JT-PATH-002"}  # only the paging tool's hand-off; no invented gaps


def test_findings_without_a_service_leave_the_score_alone() -> None:
    graph = _disk().graph
    ctx = CheckContext(graph)
    report = score(ctx, run_checks(ctx))
    assert report.services == ()


def test_a_document_wide_hand_off_with_nothing_to_hold_it_is_noted() -> None:
    doc = _doc(
        "process/handbook.md",
        "# Handbook\n\nOwners and escalation contacts are kept in the infrastructure wiki.\n",
    )
    delegation = ExtractedDelegation(
        element=DelegatedElement.OWNER, target="infrastructure wiki", quote=DISK_WIKI
    )
    result = _assemble(doc, delegations=[delegation])
    assert result.graph.nodes == {}
    handed_off, empty = result.notes
    assert "담당자 항목을 외부 자료('infrastructure wiki')로 넘겼지만" in handed_off
    assert "검사 대상 없음 ≠ 정상" in handed_off
    # the document itself is never silently empty either: why it left nothing to check
    assert "검사 대상 없음 ≠ 정상" in empty
    assert "이유: 문서에서 서비스·시스템을 하나도 읽지 못했습니다" in empty
    assert _findings(result.graph) == []
