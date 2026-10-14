"""Single points of failure (단일 장애점): ``JT-SPF-001`` (사람), ``JT-SPF-002`` (채널).

Why: a response path that works today can still hang on one person ("버스 팩터 1") or one
messenger. These rules report every size-1 minimal cut set from :mod:`jettae.check.spof` —
the people/channels whose loss *alone* leaves a critical service with nobody able to respond
or approve its mitigation. Services whose path is already broken are skipped: the PATH/PPL
rules report those, and every element would trivially be "critical" there.
"""

from __future__ import annotations

from collections.abc import Iterator

from jettae.check.context import CheckContext
from jettae.check.registry import finding, register
from jettae.check.spof import (
    RESPONDER_KEY,
    GraphIndex,
    ResponseLineage,
    minimal_cut_sets,
    response_lineage,
)
from jettae.domain import (
    Element,
    Finding,
    FixHint,
    FixKind,
    Node,
    NodeKind,
    RuleInfo,
    Severity,
    SourceSpan,
)


def _single_cuts(ctx: CheckContext, kind: NodeKind) -> Iterator[tuple[Node, ResponseLineage, Node]]:
    """(service, lineage, cut node) for every size-1 cut of the given kind on a working path."""
    index = GraphIndex(ctx.graph)
    for svc in ctx.critical_services():
        lineage = response_lineage(ctx, svc.id, index=index)
        for cut in minimal_cut_sets(lineage, unavailable=ctx.unavailable, max_size=1):
            if len(cut) != 1:  # [frozenset()] = path already broken; not a SPOF question
                continue
            node = ctx.graph.nodes[next(iter(cut))]
            if node.kind is kind:
                yield svc, lineage, node


def _incident(ctx: CheckContext, lineage: ResponseLineage, nid: str) -> tuple[list[str], list[SourceSpan]]:
    """Path edges touching ``nid`` and the source spans that put it on the path."""
    edges = sorted(
        (e for e in ctx.graph.edges.values() if e.id in lineage.path_ids and nid in (e.src, e.dst)),
        key=lambda e: e.id,
    )
    spans = [*ctx.graph.nodes[nid].spans, *(s for e in edges for s in e.spans)]
    return [e.id for e in edges], spans


def _broken_labels(ctx: CheckContext, lineage: ResponseLineage, nid: str) -> tuple[str, bool]:
    """Korean list of requirements lost with ``nid`` and whether the responder one is among them."""
    broken = lineage.broken(ctx.unavailable | {nid})
    return ", ".join(r.label for r in broken), any(r.key == RESPONDER_KEY for r in broken)


@register
class SinglePersonPoint:
    info = RuleInfo(
        rule_id="JT-SPF-001",
        title="단일 장애점(사람)",
        severity=Severity.MEDIUM,
        element=Element.OWNER,
        rationale=(
            "핵심 서비스의 대응·승인이 한 사람에게만 달려 있으면(버스 팩터 1) 그 사람의 휴가·수면·퇴사만으로 "
            "대응 경로가 끊긴다. 대응 경로의 계보(lineage)에서 크기 1인 최소 절단 집합을 찾는다."
        ),
    )

    def check(self, ctx: CheckContext) -> Iterator[Finding]:
        for svc, lineage, person in _single_cuts(ctx, NodeKind.PERSON):
            labels, responder_lost = _broken_labels(ctx, lineage, person.id)
            edge_ids, spans = _incident(ctx, lineage, person.id)
            yield finding(
                self.info,
                keys=(svc.id, person.id),
                title=self.info.title,
                message=(
                    f"'{svc.name}'의 대응 경로가 {person.name} 한 사람에게 달려 있습니다(버스 팩터 1). "
                    f"이 사람이 연락되지 않으면 {labels} 단계가 끊깁니다. 대리 담당자를 지정해 두세요."
                ),
                service_id=svc.id,
                element=Element.OWNER if responder_lost else Element.APPROVER,
                subject_ids=(svc.id, person.id, *edge_ids),
                evidence=spans,
                fix=FixHint(kind=FixKind.ADD_DEPUTY, params={"person_id": person.id, "service_id": svc.id}),
            )


@register
class SingleChannelPoint:
    info = RuleInfo(
        rule_id="JT-SPF-002",
        title="단일 장애점(채널)",
        severity=Severity.MEDIUM,
        element=Element.FALLBACK_CHANNEL,
        rationale=(
            "핵심 서비스에 응답·승인할 수 있는 모든 사람이 한 연락 수단으로만 닿으면, 그 메신저 장애 하나로 "
            "아무에게도 연락할 수 없다. 크기 1인 최소 절단 집합 중 채널을 찾는다."
        ),
    )

    def check(self, ctx: CheckContext) -> Iterator[Finding]:
        for svc, lineage, channel in _single_cuts(ctx, NodeKind.CHANNEL):
            labels, _ = _broken_labels(ctx, lineage, channel.id)
            edge_ids, spans = _incident(ctx, lineage, channel.id)
            yield finding(
                self.info,
                keys=(svc.id, channel.id),
                title=self.info.title,
                message=(
                    f"'{svc.name}'에 대응할 사람들이 모두 '{channel.name}' 하나로만 연결됩니다. "
                    f"이 채널이 멈추면 {labels} 단계가 끊깁니다. 다른 연락 수단(대체 채널)을 지정해 두세요."
                ),
                service_id=svc.id,
                subject_ids=(svc.id, channel.id, *edge_ids),
                evidence=spans,
                fix=FixHint(
                    kind=FixKind.ADD_FALLBACK_CHANNEL,
                    params={"channel_id": channel.id, "service_id": svc.id},
                ),
            )
