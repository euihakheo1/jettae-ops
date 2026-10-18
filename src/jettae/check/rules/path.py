"""JT-PATH-* — required response-path rules (필수 경로 규칙) for critical services.

Why these rules read :mod:`jettae.check.elements` instead of walking the graph themselves: the
scorer uses the very same analyses, so a JT-PATH finding appears exactly when the matching
element loses points for that reason, and the finding's subject ids are the same nodes/edges the
score reasons about. Non-critical services are scored but not flagged — the published scope of
the required-path check is ``ctx.critical_services()``.

Held items (판단 보류) are never evidence that a requirement is met. When a gap would close if the
held items turned out to be right (checked on :func:`~jettae.check.elements.confirmed_view`), the
finding is reported as UNKNOWN ("확인 필요") instead of BROKEN and its message says so: the runbook
may well state what extraction could not yet confirm, so the report must not present the gap as
settled (P3).

The same holds for an element the runbook hands off to an external source instead of stating it
("owners and escalation: see the team directory",:func:`~jettae.check.elements.delegation_of`):
nothing is written here, but nothing was forgotten either — the finding is UNKNOWN ("확인 불가"),
cites the delegating sentence, and asks for that source.
"""

from __future__ import annotations

from collections.abc import Iterator

from jettae.check.context import CheckContext
from jettae.check.elements import (
    ActionApproval,
    ActionReadiness,
    ApprovalGate,
    analyze_approval,
    analyze_channels,
    analyze_detection,
    analyze_owner,
    analyze_rollback,
    channel_delegation,
    confirmed_view,
    delegated_message,
    delegated_title,
    delegation_of,
    delivery_channel,
    generic_alert_message,
    generic_alerts,
    josa,
    label,
    labels,
    merge_spans,
    owner_delegation,
    person_label,
    policy_tier_basis,
    service_label,
    spans_of,
    suggest_candidates,
    trigger_delegation,
)
from jettae.check.registry import finding, register
from jettae.domain import (
    DelegatedElement,
    EdgeKind,
    Element,
    Finding,
    FindingStatus,
    FixHint,
    FixKind,
    Node,
    RuleInfo,
    Severity,
)

HOLD_TAIL = (
    " 다만 판단 보류(HOLD) 중인 항목이 맞다고 확인되면 해소되므로, 끊김 확정이 아니라 확인 필요로 표시합니다."
)


def _subjects(*groups: tuple[str, ...] | list[str] | str) -> tuple[str, ...]:
    out: list[str] = []
    for g in groups:
        out.extend([g] if isinstance(g, str) else g)
    return tuple(dict.fromkeys(out))


def _verdict(unsure: bool) -> tuple[FindingStatus, str]:
    """(status, message tail): UNKNOWN when confirming held items would close the gap."""
    return (FindingStatus.UNKNOWN, HOLD_TAIL) if unsure else (FindingStatus.BROKEN, "")


#: Why the channels are unknown when the owner is delegated: there is nobody whose channels to count.
CHANNEL_LEAD = "연락 수단을 확인할 대응자를 알 수 없습니다. "


def generic_alert_findings(
    ctx: CheckContext,
    info: RuleInfo,
    element: DelegatedElement,
    *,
    via_owner: bool = False,
) -> Iterator[Finding]:
    """확인 불가 findings for what the runbook of a service-agnostic alert hands off.

    Such a runbook names no service, so no service is checked (the assembler's note says so), yet
    "owners: see the catalog" is a stated fact about every service the alert fires for. Reporting
    it keeps the element visible instead of silently unchecked; it is never a gap (nothing in the
    input says the element is missing) and never met. ``via_owner``: the element also depends on
    a delegated owner (channels need someone to reach).
    """
    title = delegated_title(info.title)
    for trigger in generic_alerts(ctx):
        delegated, lead = delegation_of(ctx, trigger.id, element), ""
        if delegated is None and via_owner:
            delegated = delegation_of(ctx, trigger.id, DelegatedElement.OWNER)
            lead = CHANNEL_LEAD
        if delegated is None:
            continue
        subjects = (trigger.id,)
        yield finding(
            info,
            keys=subjects,
            title=title,
            message=generic_alert_message(ctx, trigger.id, delegated, lead=lead),
            subject_ids=subjects,
            evidence=merge_spans(spans_of(ctx, subjects), delegated.evidence),
            status=FindingStatus.UNKNOWN,
        )


@register
class DetectionMissing:
    info = RuleInfo(
        rule_id="JT-PATH-001",
        title="감지 경로 없음",
        severity=Severity.CRITICAL,
        element=Element.DETECTION,
        rationale=(
            "장애를 알아챌 신호(알람·모니터·고객 문의 경로)가 없으면 대응이 시작되지 않습니다. 누군가 "
            "우연히 발견할 때까지 장애가 계속됩니다. 감지 경로가 판단 보류 상태여도 정상으로 보지 않습니다. "
            "호출 알람이 아니어도 문서가 장애가 드러나는 방법으로 적은 것(상태 점검 실패, 채널에 게시되는 "
            "알림, 장애를 보여 주는 곳으로 지목한 대시보드, 공급자의 장애 공지)은 감지 신호로 셉니다."
        ),
    )

    def check(self, ctx: CheckContext) -> Iterator[Finding]:
        for svc in ctx.critical_services():
            det = analyze_detection(ctx, svc.id)
            if det.held_ids:
                subjects = _subjects(svc.id, det.held_ids)
                yield finding(
                    self.info,
                    keys=subjects,
                    title="감지 경로 확인 불가",
                    message=(
                        f"{service_label(ctx, svc.id)}의 감지 경로 "
                        f"{josa(labels(ctx, det.held_ids), '이', '가')} 판단 보류 상태입니다. 원문과 추출 "
                        "결과가 어긋나 실제로 알람이 누구에게 가는지 확인할 수 없으므로 정상으로 보지 "
                        "않습니다."
                    ),
                    service_id=svc.id,
                    subject_ids=subjects,
                    evidence=spans_of(ctx, subjects),
                    status=FindingStatus.UNKNOWN,
                )
            elif not det.routes:
                own = analyze_owner(ctx, svc.id)
                delegated = delegation_of(ctx, svc.id, DelegatedElement.DETECTION)
                message = (
                    f"{josa(service_label(ctx, svc.id), '은', '는')} 장애를 알아챌 감지 "
                    "신호(알람·모니터)가 런북에 없습니다. 누군가 우연히 발견하기 전까지 대응이 시작되지 "
                    "않습니다."
                )
                yield finding(
                    self.info,
                    keys=(svc.id,),
                    title=self.info.title if delegated is None else "감지 경로 확인 불가",
                    message=message if delegated is None else delegated_message(ctx, svc.id, delegated),
                    service_id=svc.id,
                    subject_ids=(svc.id,),
                    evidence=merge_spans(spans_of(ctx, (svc.id,)), delegated.evidence if delegated else ()),
                    status=FindingStatus.BROKEN if delegated is None else FindingStatus.UNKNOWN,
                    fix=FixHint(
                        kind=FixKind.ADD_DETECTION,
                        params={
                            "service_id": svc.id,
                            "trigger_ids": [],
                            "notify_candidates": list(own.active_ids)
                            or suggest_candidates(ctx, own.owner_ids),
                        },
                    ),
                )


@register
class DetectionNotifiesNobody:
    info = RuleInfo(
        rule_id="JT-PATH-002",
        title="감지 후 호출 대상 없음",
        severity=Severity.HIGH,
        element=Element.DETECTION,
        rationale=(
            "알람이 울려도 받을 사람이 없거나 퇴사·부재자에게만 간다면 감지는 아무 소용이 없습니다. 감지 "
            "신호마다 활동 중인 대응자에게 이어지는 호출 경로가 있어야 합니다. 채널에 게시만 되는 알림은 "
            "호출이 아닙니다. 호출 도구로 발송된다고만 적혀 받는 사람을 알 수 없으면, 받는 사람은 그 도구의 "
            "설정이 정하므로 확인 불가로 알립니다."
        ),
    )

    def check(self, ctx: CheckContext) -> Iterator[Finding]:
        alt = confirmed_view(ctx)
        for svc in ctx.critical_services():
            det = analyze_detection(ctx, svc.id)
            if det.held_ids or not det.routes or det.notified:
                continue
            status, tail = _verdict(alt is not ctx and analyze_detection(alt, svc.id).notified)
            trigger_ids = list(dict.fromkeys(r.trigger_id for r in det.routes))
            target_ids = sorted({t for r in det.routes for t in r.target_ids})
            notify_ids = [e for r in det.routes for e in r.notify_edge_ids]
            channel_ids = list(
                dict.fromkeys(c.id for c in (delivery_channel(ctx, t) for t in trigger_ids) if c is not None)
            )
            triggers = labels(ctx, trigger_ids)
            if target_ids:
                targets = ", ".join(person_label(ctx, t) for t in target_ids)
                head = f"{josa(triggers, '이', '가')} 울려도"
                why = f"호출 대상 {josa(targets, '이', '가')} 활동 중인 대응자로 이어지지 않습니다"
                outcome = "알람이 아무에게도 닿지 않습니다"
            elif channel_ids:
                # posted to a channel, nobody named: the signal exists, but no responder is paged
                head = josa(triggers, "은", "는")
                why = f"{labels(ctx, channel_ids)}에 게시될 뿐 호출받는 대응자가 적혀 있지 않습니다"
                outcome = "그 채널을 지켜보는 사람이 없으면 아무도 알아채지 못합니다"
            else:
                head = f"{josa(triggers, '이', '가')} 울려도"
                why = "호출 대상이 적혀 있지 않습니다"
                outcome = "알람이 아무에게도 닿지 않습니다"
            subjects = _subjects(svc.id, trigger_ids, notify_ids, target_ids, channel_ids)
            own = analyze_owner(ctx, svc.id)
            message = f"{service_label(ctx, svc.id)}의 감지 신호 {head} {why}. {outcome}.{tail}"
            evidence = spans_of(ctx, subjects)
            title = self.info.title
            delegated, lead = None, ""
            if not target_ids:
                delegated = delegation_of(ctx, svc.id, DelegatedElement.DETECTION)
                if delegated is None:
                    delegated = trigger_delegation(ctx, trigger_ids)
                    lead = (
                        f"감지 신호 {josa(triggers, '은', '는')} 호출 도구로 발송되지만 받는 사람이 "
                        "적혀 있지 않습니다. "
                        if delegated is not None
                        else ""
                    )
            if delegated is not None:
                status, title = FindingStatus.UNKNOWN, delegated_title(self.info.title)
                message = delegated_message(ctx, svc.id, delegated, lead=lead)
                evidence = merge_spans(evidence, delegated.evidence)
            yield finding(
                self.info,
                keys=subjects,
                title=title,
                message=message,
                service_id=svc.id,
                subject_ids=subjects,
                evidence=evidence,
                status=status,
                fix=FixHint(
                    kind=FixKind.ADD_DETECTION,
                    params={
                        "service_id": svc.id,
                        "trigger_ids": trigger_ids,
                        "notify_candidates": list(own.active_ids)
                        or suggest_candidates(ctx, (*own.owner_ids, *target_ids)),
                    },
                ),
            )
        yield from generic_alert_findings(ctx, self.info, DelegatedElement.DETECTION)


@register
class OwnerMissing:
    info = RuleInfo(
        rule_id="JT-PATH-010",
        title="담당자 없음",
        severity=Severity.CRITICAL,
        element=Element.OWNER,
        rationale=(
            "서비스를 맡은 활동 중인 사람이 없으면 장애가 나도 책임지고 움직일 사람이 없습니다. 담당자가 "
            "퇴사·장기 부재이거나 명부에서 확인되지 않는 경우도 포함합니다."
        ),
    )

    def check(self, ctx: CheckContext) -> Iterator[Finding]:
        alt = confirmed_view(ctx)
        for svc in ctx.critical_services():
            own = analyze_owner(ctx, svc.id)
            if own.active_ids:
                continue
            unknown = bool(own.unknown_ids)
            status, hold_tail = _verdict(alt is not ctx and bool(analyze_owner(alt, svc.id).active_ids))
            if not own.owner_ids:
                why = "담당자가 지정되어 있지 않습니다"
            elif own.people_ids:
                people = ", ".join(person_label(ctx, p) for p in own.people_ids)
                why = (
                    f"지정된 담당자 {josa(people, '은', '는')} 활동 중이 아닙니다"
                    if len(own.people_ids) == 1
                    else f"지정된 담당자 {people} 중 활동 중인 사람이 없습니다"
                )
            else:
                why = f"담당으로 적힌 {josa(labels(ctx, own.owner_ids), '을', '를')} 맡은 사람이 없습니다"
            tail = (
                " 명부에서 확인되지 않는 사람은 정상으로 보지 않습니다."
                if unknown
                else " 장애가 나면 책임지고 움직일 사람이 없습니다."
            ) + hold_tail
            subjects = _subjects(svc.id, own.owner_edge_ids, own.owner_ids, own.people_ids)
            message = f"{service_label(ctx, svc.id)}: {why}.{tail}"
            evidence = spans_of(ctx, subjects)
            title = self.info.title
            delegated = None if own.owner_ids else owner_delegation(ctx, svc.id)
            if delegated is not None:
                unknown, title = True, delegated_title(self.info.title)
                message = delegated_message(ctx, svc.id, delegated)
                evidence = merge_spans(evidence, delegated.evidence)
            yield finding(
                self.info,
                keys=subjects,
                title=title,
                message=message,
                service_id=svc.id,
                subject_ids=subjects,
                evidence=evidence,
                status=FindingStatus.UNKNOWN if unknown else status,
                fix=FixHint(
                    kind=FixKind.ASSIGN_OWNER,
                    params={
                        "service_id": svc.id,
                        "current_owner_ids": list(own.owner_ids),
                        "candidates": suggest_candidates(ctx, (*own.owner_ids, *own.people_ids)),
                    },
                ),
            )
        yield from generic_alert_findings(ctx, self.info, DelegatedElement.OWNER)


@register
class OwnerUnreachable:
    info = RuleInfo(
        rule_id="JT-PATH-011",
        title="담당자 연락 수단 없음",
        severity=Severity.HIGH,
        element=Element.OWNER,
        rationale=(
            "담당자가 있어도 연락할 수단(전화·메신저·호출 도구)이 등록되어 있지 않으면 장애 때 부를 수 "
            "없습니다."
        ),
    )

    def check(self, ctx: CheckContext) -> Iterator[Finding]:
        alt = confirmed_view(ctx)
        for svc in ctx.critical_services():
            own = analyze_owner(ctx, svc.id)
            if not own.active_ids or own.reachable_ids:
                continue
            status, tail = _verdict(alt is not ctx and bool(analyze_owner(alt, svc.id).reachable_ids))
            subjects = _subjects(svc.id, own.owner_edge_ids, own.active_ids)
            yield finding(
                self.info,
                keys=subjects,
                title=self.info.title,
                message=(
                    f"{service_label(ctx, svc.id)}의 담당자 {labels(ctx, own.active_ids)}에게 등록된 연락 "
                    f"수단이 없습니다. 장애가 나도 담당자를 부를 방법이 없습니다.{tail}"
                ),
                service_id=svc.id,
                subject_ids=subjects,
                evidence=spans_of(ctx, subjects),
                status=status,
                fix=FixHint(
                    kind=FixKind.ADD_FALLBACK_CHANNEL,
                    params={"service_id": svc.id, "person_ids": list(own.active_ids), "channel_ids": []},
                ),
            )


def _ready_if_confirmed(alt: CheckContext, sid: str) -> set[str]:
    """Actions that would have a ready approver once held items are confirmed."""
    return {a.action_id for a in analyze_approval(alt, sid).approvals if a.ready_ids}


#: Why approval is needed when the policy, not the document, says so (default tier + default gate).
POLICY_APPROVAL_TAIL = (
    " 승인이 필요하다는 전제는 문서가 아니라 정책에서 왔습니다(등급이 적히지 않은 서비스는 핵심 등급으로 "
    "보고, 핵심 서비스의 롤백·페일오버는 승인을 받는 것을 기본으로 봅니다)."
)


def _approver_message(ctx: CheckContext, svc: Node, approval: ActionApproval) -> str:
    """Approver gap message. When the gate is the policy default *and* the service is critical only
    by the policy's default tier, the premise "approval is needed" is the policy's, and the message
    says so (:func:`~jettae.check.elements.policy_tier_basis`): the document is judged on whether
    it names an approver, not on whether it says approval is needed."""
    action = label(ctx, approval.action_id)
    basis = policy_tier_basis(ctx, svc.id) if approval.gate is ApprovalGate.DEFAULT else None
    if approval.gate is ApprovalGate.STATED:
        need = "승인을 받도록 적혀 있지만"
    elif basis is not None:
        need = f"{basis}으로 승인이 필요하지만"
    else:
        need = "승인이 필요하지만"
    head = f"{service_label(ctx, svc.id)}의 조치 {josa(action, '은', '는')} {need}"
    tail = POLICY_APPROVAL_TAIL if basis is not None else ""
    if approval.unknown_ids:
        who = josa(labels(ctx, approval.unknown_ids), "이", "가")
        return (
            f"{head} 승인자 {who} 명부에서 확인되지 않습니다. "
            f"실제로 승인할 수 있는지 확인할 수 없습니다.{tail}"
        )
    if approval.approver_ids:
        people = labels(ctx, approval.approver_ids)
        return (
            f"{head} 승인자 {people} 중 활동 중이고 연락 가능한 사람이 없습니다. 장애 중에 승인을 받지 못해 "
            f"조치가 멈춥니다.{tail}"
        )
    return (
        f"{head} 승인자가 적혀 있지 않습니다. 장애 중에 누구에게 승인을 받아야 하는지 몰라 조치가 "
        f"늦어집니다.{tail}"
    )


@register
class ApproverMissing:
    info = RuleInfo(
        rule_id="JT-PATH-020",
        title="승인자 없음",
        severity=Severity.HIGH,
        element=Element.APPROVER,
        rationale=(
            "롤백·페일오버처럼 승인이 필요한 조치(원문에 승인 필요로 적힌 조치 포함)에 활동 중이고 "
            "연락 가능한 승인자가 없으면, 조치 방법을 알아도 실행하지 못하고 기다리게 됩니다."
        ),
    )

    def check(self, ctx: CheckContext) -> Iterator[Finding]:
        alt = confirmed_view(ctx)
        for svc in ctx.critical_services():
            appr = analyze_approval(ctx, svc.id)
            own = analyze_owner(ctx, svc.id)
            ready_if_confirmed = _ready_if_confirmed(alt, svc.id) if alt is not ctx else set()
            for approval in appr.approvals:
                if approval.ready_ids:
                    continue
                status, tail = _verdict(approval.action_id in ready_if_confirmed)
                subjects = _subjects(svc.id, approval.action_id, approval.edge_ids, approval.approver_ids)
                message = _approver_message(ctx, svc, approval) + tail
                evidence = spans_of(ctx, subjects)
                title = self.info.title
                stated = approval.approver_ids or approval.unknown_ids
                delegated = None if stated else delegation_of(ctx, svc.id, DelegatedElement.APPROVER)
                if delegated is not None:
                    status, title = FindingStatus.UNKNOWN, delegated_title(self.info.title)
                    message = delegated_message(ctx, svc.id, delegated)
                    evidence = merge_spans(evidence, delegated.evidence)
                yield finding(
                    self.info,
                    keys=subjects,
                    title=title,
                    message=message,
                    service_id=svc.id,
                    subject_ids=subjects,
                    evidence=evidence,
                    status=FindingStatus.UNKNOWN if approval.unknown_ids else status,
                    fix=FixHint(
                        kind=FixKind.ASSIGN_APPROVER,
                        params={
                            "service_id": svc.id,
                            "action_id": approval.action_id,
                            "current_approver_ids": list(approval.approver_ids),
                            "candidates": suggest_candidates(ctx, (*own.owner_ids, *approval.approver_ids)),
                        },
                    ),
                )
        yield from generic_alert_findings(ctx, self.info, DelegatedElement.APPROVER)


@register
class FallbackChannelMissing:
    info = RuleInfo(
        rule_id="JT-PATH-030",
        title="대체 채널 없음",
        severity=Severity.HIGH,
        element=Element.FALLBACK_CHANNEL,
        rationale=(
            "담당자와 1차 상향 대상에게 닿는 연락 수단이 2개 미만이면, 그 하나가 끊기는 순간 아무도 부를 수 "
            "없습니다. 공급사 지원 창구(공급사와의 공유 채널, 지원 티켓)는 공급사에 닿을 뿐 우리 쪽 "
            "담당자·상향 대상에게 닿는 수단이 아니므로 세지 않습니다."
        ),
    )

    def check(self, ctx: CheckContext) -> Iterator[Finding]:
        alt = confirmed_view(ctx)
        for svc in ctx.critical_services():
            own = analyze_owner(ctx, svc.id)
            ch = analyze_channels(ctx, svc.id, owner=own)
            if len(ch.channel_ids) >= 2:
                continue
            status, tail = _verdict(alt is not ctx and len(analyze_channels(alt, svc.id).channel_ids) >= 2)
            if ch.channel_ids:
                only = label(ctx, ch.channel_ids[0])
                why = f"연락 수단이 {only} 하나뿐이라, 이것이 끊기면 대신 쓸 채널이 없습니다"
            elif ch.person_ids:
                why = f"대응자 {labels(ctx, ch.person_ids)}에게 등록된 연락 수단이 하나도 없습니다"
            else:
                why = "연락할 활동 중인 담당자·1차 상향 대상이 없어 쓸 수 있는 연락 수단이 없습니다"
            subjects = _subjects(svc.id, ch.person_ids, ch.channel_ids)
            message = f"{service_label(ctx, svc.id)}: {why}.{tail}"
            evidence = spans_of(ctx, subjects)
            title = self.info.title
            delegated = channel_delegation(ctx, ch)
            if delegated is not None:
                status, title = FindingStatus.UNKNOWN, delegated_title(self.info.title)
                lead = CHANNEL_LEAD if delegated.element is DelegatedElement.OWNER else ""
                message = delegated_message(ctx, svc.id, delegated, lead=lead)
                evidence = merge_spans(evidence, delegated.evidence)
            yield finding(
                self.info,
                keys=subjects,
                title=title,
                message=message,
                service_id=svc.id,
                subject_ids=subjects,
                evidence=evidence,
                status=status,
                fix=FixHint(
                    kind=FixKind.ADD_FALLBACK_CHANNEL,
                    params={
                        "service_id": svc.id,
                        "person_ids": list(ch.person_ids),
                        "owner_ids": list(own.owner_ids),
                        "channel_ids": list(ch.channel_ids),
                    },
                ),
            )
        yield from generic_alert_findings(ctx, self.info, DelegatedElement.FALLBACK_CHANNEL, via_owner=True)


@register
class RollbackMissing:
    info = RuleInfo(
        rule_id="JT-PATH-040",
        title="롤백/완화 조치 없음",
        severity=Severity.HIGH,
        element=Element.ROLLBACK,
        rationale=(
            "롤백·페일오버·기능 플래그처럼 서비스를 알려진 상태로 되돌리는 조치가 없으면 장애 중에 "
            "즉흥적으로 대응하게 됩니다. 재시작·확장·임시 우회 방법만 있는 경우는 한 단계 낮은 심각도로 "
            "알립니다. 여러 서비스를 함께 들어 적은 조치나 문서 전체에 대한 조치는 그 각 서비스의 조치로 "
            "봅니다."
        ),
    )

    def check(self, ctx: CheckContext) -> Iterator[Finding]:
        alt = confirmed_view(ctx)
        for svc in ctx.critical_services():
            rb = analyze_rollback(ctx, svc.id)
            if rb.reversible:
                continue
            status, tail = _verdict(alt is not ctx and bool(analyze_rollback(alt, svc.id).reversible))
            action_ids = [a.action_id for a in rb.actions]
            if action_ids:
                severity = Severity.MEDIUM
                message = (
                    f"{service_label(ctx, svc.id)}에는 {labels(ctx, action_ids)} 같은 조치만 있고 "
                    "롤백·페일오버·기능 플래그처럼 서비스를 되돌리는 조치가 없습니다."
                )
            else:
                severity = Severity.HIGH
                message = (
                    f"{service_label(ctx, svc.id)}에 롤백·페일오버 같은 완화 조치가 적혀 있지 않습니다. 장애 "
                    "중에 무엇을 해야 할지 즉석에서 정해야 합니다."
                )
            subjects = _subjects(svc.id, rb.mitigation_edge_ids, action_ids)
            message += tail
            evidence = spans_of(ctx, subjects)
            title = self.info.title
            delegated = None if action_ids else delegation_of(ctx, svc.id, DelegatedElement.ROLLBACK)
            if delegated is not None:
                status, title = FindingStatus.UNKNOWN, "롤백/완화 조치 확인 불가"
                message = delegated_message(ctx, svc.id, delegated)
                evidence = merge_spans(evidence, delegated.evidence)
            yield finding(
                self.info,
                keys=subjects,
                title=title,
                message=message,
                service_id=svc.id,
                subject_ids=subjects,
                evidence=evidence,
                severity=severity,
                status=status,
                fix=FixHint(
                    kind=FixKind.ADD_ROLLBACK,
                    params={"service_id": svc.id, "existing_action_ids": action_ids},
                ),
            )
        yield from generic_alert_findings(ctx, self.info, DelegatedElement.ROLLBACK)


def _executor_message(ctx: CheckContext, svc: Node, action: ActionReadiness) -> str:
    head = f"{service_label(ctx, svc.id)}의 되돌리기 조치 {label(ctx, action.action_id)}"
    if action.executor_ids:
        people = ", ".join(person_label(ctx, p) for p in action.executor_ids)
        return f"{head}의 실행자 {josa(people, '이', '가')} 활동 중인 대응자로 이어지지 않습니다."
    return f"{head}에 실행자가 적혀 있지 않습니다. 장애 중에 누가 실행할지 정하느라 시간이 흐릅니다."


@register
class RollbackExecutorMissing:
    info = RuleInfo(
        rule_id="JT-PATH-041",
        title="롤백 실행자 미지정",
        severity=Severity.MEDIUM,
        element=Element.ROLLBACK,
        rationale=(
            "되돌리기 조치가 있어도 실행할 사람이 정해져 있지 않으면 장애 중에 누가 할지 정하느라 시간이 "
            "흐릅니다. 실행자가 적혀 있지 않으면 없는 것으로 봅니다."
        ),
    )

    def check(self, ctx: CheckContext) -> Iterator[Finding]:
        alt = confirmed_view(ctx)
        for svc in ctx.critical_services():
            rb = analyze_rollback(ctx, svc.id)
            reversible = rb.reversible
            if not reversible or any(a.active_executor_ids for a in reversible):
                continue
            status, tail = _verdict(
                alt is not ctx
                and any(a.active_executor_ids for a in analyze_rollback(alt, svc.id).reversible)
            )
            own = analyze_owner(ctx, svc.id)
            for action in reversible:
                subjects = _subjects(svc.id, action.action_id, action.executor_edge_ids, action.executor_ids)
                yield finding(
                    self.info,
                    keys=subjects,
                    title=self.info.title,
                    message=_executor_message(ctx, svc, action) + tail,
                    service_id=svc.id,
                    subject_ids=subjects,
                    evidence=spans_of(ctx, subjects),
                    status=status,
                    fix=FixHint(
                        kind=FixKind.ASSIGN_EXECUTOR,
                        params={
                            "service_id": svc.id,
                            "action_id": action.action_id,
                            "edge_kind": EdgeKind.EXECUTED_BY.value,
                            "current_executor_ids": list(action.executor_ids),
                            "candidates": list(own.active_ids)
                            or suggest_candidates(ctx, (*own.owner_ids, *action.executor_ids)),
                        },
                    ),
                )
