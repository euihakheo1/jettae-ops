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
    confirmed_view,
    josa,
    label,
    labels,
    person_label,
    service_label,
    spans_of,
    suggest_candidates,
)
from jettae.check.registry import finding, register
from jettae.domain import (
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


@register
class DetectionMissing:
    info = RuleInfo(
        rule_id="JT-PATH-001",
        title="감지 경로 없음",
        severity=Severity.CRITICAL,
        element=Element.DETECTION,
        rationale=(
            "장애를 알아챌 신호(알람·모니터·고객 문의 경로)가 없으면 대응이 시작되지 않습니다. 누군가 "
            "우연히 발견할 때까지 장애가 계속됩니다. 감지 경로가 판단 보류 상태여도 정상으로 보지 않습니다."
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
                yield finding(
                    self.info,
                    keys=(svc.id,),
                    title=self.info.title,
                    message=(
                        f"{josa(service_label(ctx, svc.id), '은', '는')} 장애를 알아챌 감지 "
                        "신호(알람·모니터)가 런북에 없습니다. 누군가 우연히 발견하기 전까지 대응이 시작되지 "
                        "않습니다."
                    ),
                    service_id=svc.id,
                    subject_ids=(svc.id,),
                    evidence=spans_of(ctx, (svc.id,)),
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
            "신호마다 활동 중인 대응자에게 이어지는 호출 경로가 있어야 합니다."
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
            if target_ids:
                targets = ", ".join(person_label(ctx, t) for t in target_ids)
                why = f"호출 대상 {josa(targets, '이', '가')} 활동 중인 대응자로 이어지지 않습니다"
            else:
                why = "호출 대상이 적혀 있지 않습니다"
            subjects = _subjects(svc.id, trigger_ids, notify_ids, target_ids)
            own = analyze_owner(ctx, svc.id)
            yield finding(
                self.info,
                keys=subjects,
                title=self.info.title,
                message=(
                    f"{service_label(ctx, svc.id)}의 감지 신호 {josa(labels(ctx, trigger_ids), '이', '가')} "
                    f"울려도 {why}. 알람이 아무에게도 닿지 않습니다.{tail}"
                ),
                service_id=svc.id,
                subject_ids=subjects,
                evidence=spans_of(ctx, subjects),
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
            yield finding(
                self.info,
                keys=subjects,
                title=self.info.title,
                message=f"{service_label(ctx, svc.id)}: {why}.{tail}",
                service_id=svc.id,
                subject_ids=subjects,
                evidence=spans_of(ctx, subjects),
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


def _approver_message(ctx: CheckContext, svc: Node, approval: ActionApproval) -> str:
    action = label(ctx, approval.action_id)
    need = "승인을 받도록 적혀 있지만" if approval.gate is ApprovalGate.STATED else "승인이 필요하지만"
    head = f"{service_label(ctx, svc.id)}의 조치 {josa(action, '은', '는')} {need}"
    if approval.unknown_ids:
        who = josa(labels(ctx, approval.unknown_ids), "이", "가")
        return f"{head} 승인자 {who} 명부에서 확인되지 않습니다. 실제로 승인할 수 있는지 확인할 수 없습니다."
    if approval.approver_ids:
        people = labels(ctx, approval.approver_ids)
        return (
            f"{head} 승인자 {people} 중 활동 중이고 연락 가능한 사람이 없습니다. 장애 중에 승인을 받지 못해 "
            "조치가 멈춥니다."
        )
    return (
        f"{head} 승인자가 적혀 있지 않습니다. 장애 중에 누구에게 승인을 받아야 하는지 몰라 조치가 늦어집니다."
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
                yield finding(
                    self.info,
                    keys=subjects,
                    title=self.info.title,
                    message=_approver_message(ctx, svc, approval) + tail,
                    service_id=svc.id,
                    subject_ids=subjects,
                    evidence=spans_of(ctx, subjects),
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


@register
class FallbackChannelMissing:
    info = RuleInfo(
        rule_id="JT-PATH-030",
        title="대체 채널 없음",
        severity=Severity.HIGH,
        element=Element.FALLBACK_CHANNEL,
        rationale=(
            "담당자와 1차 상향 대상에게 닿는 연락 수단이 2개 미만이면, 그 하나가 끊기는 순간 아무도 부를 수 "
            "없습니다."
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
            yield finding(
                self.info,
                keys=subjects,
                title=self.info.title,
                message=f"{service_label(ctx, svc.id)}: {why}.{tail}",
                service_id=svc.id,
                subject_ids=subjects,
                evidence=spans_of(ctx, subjects),
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


@register
class RollbackMissing:
    info = RuleInfo(
        rule_id="JT-PATH-040",
        title="롤백/완화 조치 없음",
        severity=Severity.HIGH,
        element=Element.ROLLBACK,
        rationale=(
            "롤백·페일오버·기능 플래그처럼 서비스를 알려진 상태로 되돌리는 조치가 없으면 장애 중에 "
            "즉흥적으로 대응하게 됩니다. 재시작·확장만 있는 경우는 한 단계 낮은 심각도로 알립니다."
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
            yield finding(
                self.info,
                keys=subjects,
                title=self.info.title,
                message=message + tail,
                service_id=svc.id,
                subject_ids=subjects,
                evidence=spans_of(ctx, subjects),
                severity=severity,
                status=status,
                fix=FixHint(
                    kind=FixKind.ADD_ROLLBACK,
                    params={"service_id": svc.id, "existing_action_ids": action_ids},
                ),
            )


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
