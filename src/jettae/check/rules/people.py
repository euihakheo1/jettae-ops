"""JT-PPL-* — people rules (사람 규칙): response duties hanging on people who cannot answer.

Why these rules look at *edges* rather than at the score: a departed person can sit on a path the
score already writes off (e.g. a second owner), yet every such edge is a page that will ring into
the void and a duty someone believes is covered. The roster (``status``) is the trusted fact;
runbook text never overrides it.

Findings are grouped per (person, service, element) so each one links to exactly one element
score and its fix replaces only the edges of that duty. Edges that do not lead to any service are
grouped under "no service".
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from jettae.check.context import CheckContext
from jettae.check.elements import analyze_owner, josa, label, service_label, spans_of, suggest_candidates
from jettae.check.registry import finding, register
from jettae.domain import (
    Edge,
    EdgeKind,
    Element,
    Finding,
    FindingStatus,
    FixHint,
    FixKind,
    Node,
    NodeKind,
    PersonStatus,
    RuleInfo,
    Severity,
    Verification,
)

#: Edges that hand a response duty to someone (담당·호출·상향·승인·실행·신고).
DUTY_EDGE_KINDS: tuple[EdgeKind, ...] = (
    EdgeKind.OWNED_BY,
    EdgeKind.NOTIFIES,
    EdgeKind.ESCALATES_TO,
    EdgeKind.APPROVED_BY,
    EdgeKind.EXECUTED_BY,
    EdgeKind.REPORTED_BY,
)

_ELEMENT_OF: dict[EdgeKind, Element | None] = {
    EdgeKind.OWNED_BY: Element.OWNER,
    EdgeKind.ESCALATES_TO: Element.OWNER,
    EdgeKind.NOTIFIES: Element.DETECTION,
    EdgeKind.APPROVED_BY: Element.APPROVER,
    EdgeKind.EXECUTED_BY: Element.ROLLBACK,
    EdgeKind.REPORTED_BY: None,
}

_DUTY_KO: dict[EdgeKind, str] = {
    EdgeKind.OWNED_BY: "담당",
    EdgeKind.NOTIFIES: "호출",
    EdgeKind.ESCALATES_TO: "상향",
    EdgeKind.APPROVED_BY: "승인",
    EdgeKind.EXECUTED_BY: "실행",
    EdgeKind.REPORTED_BY: "신고",
}

#: Predecessor edge that ties the source of a duty edge to a service.
_SERVICE_LINK: dict[EdgeKind, EdgeKind] = {
    EdgeKind.NOTIFIES: EdgeKind.DETECTED_BY,
    EdgeKind.APPROVED_BY: EdgeKind.MITIGATED_BY,
    EdgeKind.EXECUTED_BY: EdgeKind.MITIGATED_BY,
    EdgeKind.REPORTED_BY: EdgeKind.MUST_REPORT,
}


@dataclass(frozen=True)
class Duty:
    """Response duty edges of one person, grouped by the service and element they serve."""

    person_id: str
    service_id: str | None
    element: Element | None
    edges: tuple[Edge, ...]

    @property
    def edge_ids(self) -> tuple[str, ...]:
        return tuple(e.id for e in self.edges)


def response_chains(ctx: CheckContext) -> dict[str, frozenset[str]]:
    """Per service: owner nodes, owner people and their escalation closure (who may be paged)."""
    chains: dict[str, frozenset[str]] = {}
    for svc in ctx.services():
        own = analyze_owner(ctx, svc.id)
        chains[svc.id] = frozenset((*own.owner_ids, *own.people_ids, *own.escalation.node_ids))
    return chains


def _services_of(ctx: CheckContext, edge: Edge, chains: dict[str, frozenset[str]]) -> list[str]:
    graph = ctx.graph
    if edge.kind is EdgeKind.ESCALATES_TO:
        return sorted(sid for sid, chain in chains.items() if edge.src in chain)
    if edge.kind is EdgeKind.OWNED_BY:
        candidates = [edge.src]
    else:
        candidates = [n.id for n in graph.predecessors(edge.src, _SERVICE_LINK[edge.kind])]
    return sorted({c for c in candidates if graph.nodes[c].kind is NodeKind.SERVICE})


def duties(ctx: CheckContext, person: Node, chains: dict[str, frozenset[str]]) -> list[Duty]:
    groups: dict[tuple[str, str], list[Edge]] = {}
    meta: dict[tuple[str, str], tuple[str | None, Element | None]] = {}
    for edge in ctx.graph.in_edges(person.id, *DUTY_EDGE_KINDS):
        element = _ELEMENT_OF[edge.kind]
        services: list[str | None] = [*_services_of(ctx, edge, chains)] or [None]
        for sid in services:
            key = (sid or "", element.value if element else "")
            groups.setdefault(key, []).append(edge)
            meta[key] = (sid, element)
    return [
        Duty(person_id=person.id, service_id=meta[k][0], element=meta[k][1], edges=tuple(groups[k]))
        for k in sorted(groups)
    ]


def _duty_words(duty: Duty) -> str:
    kinds = sorted({e.kind for e in duty.edges}, key=DUTY_EDGE_KINDS.index)
    return "·".join(_DUTY_KO[k] for k in kinds)


def _where(ctx: CheckContext, duty: Duty) -> str:
    if duty.service_id is None:
        return "서비스에 직접 이어지지 않은 대응 경로"
    return service_label(ctx, duty.service_id)


def _status_findings(
    ctx: CheckContext, info: RuleInfo, status: PersonStatus, fix_kind: FixKind
) -> Iterator[Finding]:
    chains = response_chains(ctx)
    for person in ctx.graph.nodes_of(NodeKind.PERSON):
        if ctx.person_status(person) is not status:
            continue
        person_duties = duties(ctx, person, chains)
        if not person_duties:  # the roster imports every leaver; most carry no duty at all
            continue
        # replacement candidates depend only on the person: rank the roster once, not per duty
        candidates = [] if fix_kind is FixKind.CONFIRM_IDENTITY else suggest_candidates(ctx, (person.id,))
        for duty in person_duties:
            confirmed = person.verification is Verification.VERIFIED and any(
                e.verification is Verification.VERIFIED for e in duty.edges
            )
            finding_status = (
                FindingStatus.BROKEN
                if confirmed and status is not PersonStatus.UNKNOWN
                else FindingStatus.UNKNOWN
            )
            subjects = (person.id, *duty.edge_ids)
            params: dict[str, object] = {"person_id": person.id, "edge_ids": list(duty.edge_ids)}
            if fix_kind is FixKind.CONFIRM_IDENTITY:
                params["name"] = person.name
            else:
                params["candidates"] = list(candidates)
            yield finding(
                info,
                keys=(duty.service_id or "-", duty.element.value if duty.element else "-", *subjects),
                title=info.title,
                message=_MESSAGES[status](ctx, person, duty),
                service_id=duty.service_id,
                element=duty.element,
                subject_ids=subjects,
                evidence=spans_of(ctx, subjects),
                status=finding_status,
                fix=FixHint(kind=fix_kind, params=params),
            )


def _departed_message(ctx: CheckContext, person: Node, duty: Duty) -> str:
    return (
        f"{_where(ctx, duty)}의 {josa(_duty_words(duty), '이', '가')} 퇴사자 {label(ctx, person.id)}에게 "
        "걸려 있습니다. 장애가 나면 이 연결은 아무에게도 닿지 않습니다."
    )


def _on_leave_message(ctx: CheckContext, person: Node, duty: Duty) -> str:
    return (
        f"{_where(ctx, duty)}의 {josa(_duty_words(duty), '이', '가')} 장기 부재 중인 "
        f"{label(ctx, person.id)}에게 걸려 있습니다. 부재 기간 동안 대신 받을 사람을 지정해야 합니다."
    )


def _unknown_message(ctx: CheckContext, person: Node, duty: Duty) -> str:
    return (
        f"{josa(label(ctx, person.id), '은', '는')} 명부에서 확인되지 않는 인물인데 {_where(ctx, duty)}의 "
        f"{josa(_duty_words(duty), '을', '를')} 맡고 있습니다. 재직·연락 가능 여부를 확인할 수 없어 정상으로 "
        "보지 않습니다."
    )


_MESSAGES = {
    PersonStatus.DEPARTED: _departed_message,
    PersonStatus.ON_LEAVE: _on_leave_message,
    PersonStatus.UNKNOWN: _unknown_message,
}


@register
class DepartedResponder:
    info = RuleInfo(
        rule_id="JT-PPL-001",
        title="퇴사자에게 대응이 걸려 있음",
        severity=Severity.CRITICAL,
        element=None,
        rationale=(
            "담당·호출·상향·승인·실행·신고가 퇴사자에게 걸려 있으면 장애 때 그 연결은 아무에게도 닿지 "
            "않습니다. 런북은 사람이 바뀌어도 저절로 고쳐지지 않으므로 명부와 대조해 찾아냅니다."
        ),
    )

    def check(self, ctx: CheckContext) -> Iterator[Finding]:
        yield from _status_findings(ctx, self.info, PersonStatus.DEPARTED, FixKind.REPLACE_DEPARTED)


@register
class UnconfirmedPerson:
    info = RuleInfo(
        rule_id="JT-PPL-002",
        title="명부에서 확인되지 않는 인물",
        severity=Severity.MEDIUM,
        element=None,
        rationale=(
            "런북에 적힌 사람이 명부에 없으면 재직 여부도 연락처도 확인할 수 없습니다. 확인 불가는 정상이 "
            "아니므로 확인이 필요하다고 알립니다."
        ),
    )

    def check(self, ctx: CheckContext) -> Iterator[Finding]:
        yield from _status_findings(ctx, self.info, PersonStatus.UNKNOWN, FixKind.CONFIRM_IDENTITY)


@register
class OnLeaveResponder:
    info = RuleInfo(
        rule_id="JT-PPL-003",
        title="장기 부재자에게 대응이 걸려 있음",
        severity=Severity.HIGH,
        element=None,
        rationale=(
            "육아휴직·장기 출장처럼 오래 자리를 비운 사람에게 대응이 걸려 있으면 그 기간 동안 연결이 "
            "끊깁니다. 부재 기간의 대행자를 지정해야 합니다."
        ),
    )

    def check(self, ctx: CheckContext) -> Iterator[Finding]:
        yield from _status_findings(ctx, self.info, PersonStatus.ON_LEAVE, FixKind.ADD_DEPUTY)
