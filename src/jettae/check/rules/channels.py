"""JT-CHN-* — channel rules (채널 규칙): contact channels that fail together with the service.

Why this is its own rule family: the classic post-mortem line "장애 공지를 사내 메신저로 하려 했는데
메신저도 같은 IDC에 있었다" is a *common-mode failure* — the runbook looks complete on paper, yet
the outage takes the channel down with it. CHN-001 reads the same channel set the
FALLBACK_CHANNEL element scores (active owner responders + first escalation hop), so the finding
fires exactly when no channel outside the service's failure domain is available.

CHN-002 catches the quieter variant: a runbook says "Slack이 안 되면 카카오톡으로", but no active
person is actually registered on the fallback channel (nobody, or only people who left or are
away), so the fallback is an empty room.

A gap that would close if held items (판단 보류) were confirmed is reported as UNKNOWN (P3).
"""

from __future__ import annotations

from collections.abc import Iterator

from jettae.check.context import CheckContext
from jettae.check.elements import (
    active_responders,
    analyze_channels,
    confirmed_view,
    josa,
    label,
    labels,
    service_label,
    spans_of,
    usable,
)
from jettae.check.registry import finding, register
from jettae.domain import (
    EdgeKind,
    Element,
    Finding,
    FindingStatus,
    FixHint,
    FixKind,
    NodeKind,
    RuleInfo,
    Severity,
)

HOLD_NOTE = " 판단 보류(HOLD) 중인 항목이 맞다고 확인되면 해소되므로 확인 필요로 표시합니다."


def _status(unsure: bool) -> tuple[FindingStatus, str]:
    return (FindingStatus.UNKNOWN, HOLD_NOTE) if unsure else (FindingStatus.BROKEN, "")


def shared_domain(ctx: CheckContext, channel_id: str, service_id: str) -> list[str]:
    """Failure-domain nodes (data center, system …) a channel shares with a service."""
    service_domain = ctx.failure_domain(service_id) | {service_id}
    return sorted(ctx.channel_failure_domain(channel_id) & service_domain)


@register
class ChannelSharesFailureDomain:
    info = RuleInfo(
        rule_id="JT-CHN-001",
        title="연락 채널이 서비스와 함께 죽음",
        severity=Severity.HIGH,
        element=Element.FALLBACK_CHANNEL,
        rationale=(
            "담당자에게 닿는 연락 수단이 모두 서비스와 같은 데이터센터·시스템에 묶여 있으면(예: 같은 IDC의 "
            "사내 메신저), 서비스가 죽는 순간 연락도 함께 끊깁니다. 서비스와 장애 영역을 공유하지 않는 "
            "채널이 하나는 있어야 합니다."
        ),
    )

    def check(self, ctx: CheckContext) -> Iterator[Finding]:
        alt = confirmed_view(ctx)
        for svc in ctx.critical_services():
            ch = analyze_channels(ctx, svc.id)
            if not ch.channel_ids or ch.independent_ids:
                continue
            status, note = _status(alt is not ctx and bool(analyze_channels(alt, svc.id).independent_ids))
            domains = sorted({d for c in ch.channel_ids for d in shared_domain(ctx, c, svc.id)})
            subjects = (svc.id, *ch.person_ids, *ch.channel_ids, *domains)
            yield finding(
                self.info,
                keys=subjects,
                title=self.info.title,
                message=(
                    f"{service_label(ctx, svc.id)} 대응자에게 닿는 연락 수단 "
                    f"{josa(labels(ctx, ch.channel_ids), '이', '가')} 모두 서비스와 같은 장애 "
                    f"영역({labels(ctx, domains)})에 묶여 있습니다. 서비스가 죽으면 연락 수단도 함께 끊겨 "
                    f"아무에게도 알릴 수 없습니다.{note}"
                ),
                service_id=svc.id,
                subject_ids=subjects,
                evidence=spans_of(ctx, subjects),
                status=status,
                fix=FixHint(
                    kind=FixKind.ADD_FALLBACK_CHANNEL,
                    params={
                        "service_id": svc.id,
                        "person_ids": list(ch.person_ids),
                        "channel_ids": list(ch.channel_ids),
                        "independent_of": domains,
                    },
                ),
            )


@register
class FallbackChannelUnused:
    info = RuleInfo(
        rule_id="JT-CHN-002",
        title="대체 채널 연결 누락",
        severity=Severity.LOW,
        element=Element.FALLBACK_CHANNEL,
        rationale=(
            "'주 채널이 안 되면 이 채널로'라고 적혀 있어도 그 채널로 연락받도록 등록된 활동 중인 사람이 "
            "없으면, 장애 때 대체 채널은 빈 방이 됩니다."
        ),
    )

    def check(self, ctx: CheckContext) -> Iterator[Finding]:
        graph = ctx.graph
        alt = confirmed_view(ctx)
        for channel in graph.nodes_of(NodeKind.CHANNEL):
            fallbacks = [e for e in graph.in_edges(channel.id, EdgeKind.FALLBACK_TO) if usable(ctx, e)]
            if not fallbacks or not usable(ctx, channel):
                continue
            registered = _registered(ctx, channel.id)
            if any(active_responders(ctx, nid) for nid in registered):
                continue
            status, note = _status(
                alt is not ctx and any(active_responders(alt, nid) for nid in _registered(alt, channel.id))
            )
            sources = sorted({e.src for e in fallbacks})
            people = sorted(
                {
                    p.id
                    for src in sources
                    for nid in _registered(ctx, src)
                    for p in active_responders(ctx, nid)
                }
            )
            subjects = (channel.id, *(e.id for e in fallbacks), *sources, *registered)
            nobody = (
                f"이 채널에 등록된 {labels(ctx, registered)} 중 활동 중인 사람이 없습니다"
                if registered
                else "이 채널로 연락받도록 등록된 사람이 없습니다"
            )
            yield finding(
                self.info,
                keys=subjects,
                title=self.info.title,
                message=(
                    f"대체 채널 {josa(label(ctx, channel.id), '은', '는')} "
                    f"{josa(labels(ctx, sources), '이', '가')} 끊겼을 때 쓰도록 적혀 있지만, {nobody}. "
                    f"장애 때 대체 채널이 비어 있게 됩니다.{note}"
                ),
                subject_ids=subjects,
                evidence=spans_of(ctx, subjects),
                status=status,
                fix=FixHint(
                    kind=FixKind.ADD_FALLBACK_CHANNEL,
                    params={"channel_id": channel.id, "person_ids": people, "channel_ids": sources},
                ),
            )


def _registered(ctx: CheckContext, channel_id: str) -> list[str]:
    """People/roles/teams registered on a channel through usable REACHABLE_VIA links."""
    graph = ctx.graph
    return sorted(
        {
            e.src
            for e in graph.in_edges(channel_id, EdgeKind.REACHABLE_VIA)
            if usable(ctx, e) and usable(ctx, graph.nodes[e.src])
        }
    )
