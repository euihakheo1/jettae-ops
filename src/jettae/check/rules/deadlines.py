"""Deadline rules (시한): ``JT-DDL-001`` … ``JT-DDL-003``.

Why: a response path can be complete on paper yet too slow to matter. Escalation timers that
add up past the recovery objective, a reporting step that quotes the wrong legal deadline
("KISA 48시간 이내" when the law says 24시간), or two documents disagreeing about the same
deadline all mean the team will miss a clock that someone else is enforcing.

Only *stated* numbers are used (P2): an escalation edge without ``after_minutes`` counts as 0
minutes, so the escalation total is a lower bound and a finding means "even the minimum
already exceeds the target". Durations in source quotes are parsed with a small, deterministic
Korean/English unit grammar — never inferred. Escalation timing follows the same usable
(non-HOLD) facts as the SPOF lineage, so both analyses walk the same chain.
"""

from __future__ import annotations

import heapq
import itertools
import math
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from jettae.check.context import CheckContext
from jettae.check.elements import policy_tier_basis
from jettae.check.registry import finding, register
from jettae.check.rules.regulatory import (
    applicable_rules,
    duty_scope,
    format_minutes,
    law_text,
    obligation_deadline,
    obligation_incident_type,
    obligation_recipient,
    recipient_matches,
    satisfies_rule,
    single_or_none,
)
from jettae.check.spof import GraphIndex, escalation_edges, gated_actions
from jettae.domain import (
    AttrKey,
    Edge,
    EdgeKind,
    Element,
    Finding,
    FindingStatus,
    FixHint,
    FixKind,
    IncidentType,
    Node,
    NodeKind,
    RuleInfo,
    Severity,
    SourceSpan,
    Verification,
)

# ---------------------------------------------------------------------------- JT-DDL-001


@dataclass(frozen=True)
class EscalationTiming:
    """Cheapest stated-time escalation from a service's owners to a goal (or to the chain's end)."""

    minutes: float
    nodes: tuple[str, ...]  # owner, …, end
    edges: tuple[Edge, ...]
    reached_goal: bool
    unstated_edges: int  # edges on the chain without after_minutes (counted as 0)
    approval_needed: bool  # the goal had to be able to approve a gated mitigation


def _after_minutes(edge: Edge) -> float | None:
    raw = edge.attr(AttrKey.AFTER_MINUTES)
    if isinstance(raw, bool) or not isinstance(raw, int | float) or raw < 0:
        return None
    return float(raw)


def _approver_people(index: GraphIndex, sid: str) -> frozenset[str] | None:
    """People able to approve an approval-gated mitigation of ``sid``; ``None`` if none is gated."""
    gated = gated_actions(index, sid)
    if not gated:
        return None
    return frozenset(
        p.id
        for _, action in gated
        for approver in index.out_edges(action.id, EdgeKind.APPROVED_BY)
        for p in index.resolve_people(approver.dst)
    )


def _is_goal(ctx: CheckContext, index: GraphIndex, nid: str, approvers: frozenset[str] | None) -> bool:
    """An active responder who can also approve (when approval is needed at all)."""
    return any(
        ctx.is_active(p) and (approvers is None or p.id in approvers) for p in index.resolve_people(nid)
    )


def _timing(
    prev: dict[str, tuple[str, Edge]], end: str, minutes: float, *, goal: bool, approval: bool
) -> EscalationTiming:
    nodes, edges = [end], []
    while nodes[-1] in prev:
        parent, edge = prev[nodes[-1]]
        nodes.append(parent)
        edges.append(edge)
    nodes.reverse()
    edges.reverse()
    unstated = sum(1 for e in edges if _after_minutes(e) is None)
    return EscalationTiming(minutes, tuple(nodes), tuple(edges), goal, unstated, approval)


def escalation_timing(
    ctx: CheckContext, sid: str, *, index: GraphIndex | None = None
) -> EscalationTiming | None:
    """Dijkstra over stated ``after_minutes`` from the owners to the first goal responder.

    Goal = an active responder who can approve the service's approval-gated mitigation (or any
    active responder if nothing needs approval). If no goal is reachable, the timing runs to the
    last node the chain pages (the "end"), because that is how long escalation alone takes
    before the path dead-ends. ``None`` when the service has no usable owner (a PATH rule's job).
    """
    if index is not None and index.graph is not ctx.graph:
        raise ValueError("GraphIndex가 다른 그래프로 만들어져 상향 시간을 계산할 수 없습니다")
    index = index if index is not None else GraphIndex(ctx.graph)
    owners = sorted(e.dst for e in index.out_edges(sid, EdgeKind.OWNED_BY))
    if not owners:
        return None
    approvers = _approver_people(index, sid)
    dist = dict.fromkeys(owners, 0.0)
    prev: dict[str, tuple[str, Edge]] = {}
    done: set[str] = set()
    heap = [(0.0, o) for o in owners]
    while heap:
        d, cur = heapq.heappop(heap)
        if cur in done:
            continue
        done.add(cur)
        if _is_goal(ctx, index, cur, approvers):
            return _timing(prev, cur, d, goal=True, approval=approvers is not None)
        for edge in escalation_edges(index, cur):
            nd = d + (_after_minutes(edge) or 0.0)
            if edge.dst not in dist or nd < dist[edge.dst]:
                dist[edge.dst] = nd
                prev[edge.dst] = (cur, edge)
                heapq.heappush(heap, (nd, edge.dst))
    end = max(sorted(done), key=lambda n: dist[n])
    return _timing(prev, end, dist[end], goal=False, approval=approvers is not None)


def _edge_evidence(edges: Iterable[Edge]) -> list[SourceSpan]:
    spans: list[SourceSpan] = []
    for e in edges:
        spans.extend(e.spans)
        attr = e.attrs.get(AttrKey.AFTER_MINUTES.value)
        if attr is not None:
            spans.extend(attr.spans)
    return spans


@register
class EscalationExceedsRecoveryTarget:
    info = RuleInfo(
        rule_id="JT-DDL-001",
        title="상향 누적 시간이 목표 복구 시간 초과",
        severity=Severity.MEDIUM,
        element=Element.OWNER,
        rationale=(
            "담당자에서 승인 가능한 활동 담당자(승인이 필요 없으면 활동 중인 담당자)까지 상향하는 데 "
            "적힌 대기 시간만 더해도 목표 복구 시간을 넘으면, "
            "모든 사람이 제때 응답해도 복구 목표를 지킬 수 없다(미기재 대기 시간은 0분으로 본 최솟값). "
            "목표 복구 시간은 등급과 관계없이 모든 서비스에 있으므로 모든 서비스를 본다."
        ),
    )

    def check(self, ctx: CheckContext) -> Iterator[Finding]:
        index = GraphIndex(ctx.graph)
        for svc in ctx.services():
            timing = escalation_timing(ctx, svc.id, index=index)
            slo = ctx.slo_minutes(svc.id)
            if timing is None or timing.minutes <= slo:
                continue
            yield self._finding(ctx, svc, timing, slo)

    def _finding(self, ctx: CheckContext, svc: Node, timing: EscalationTiming, slo: int) -> Finding:
        chain = " → ".join(ctx.graph.nodes[n].name for n in timing.nodes)
        end_name = ctx.graph.nodes[timing.nodes[-1]].name
        goal = "승인할 수 있는 활동 담당자" if timing.approval_needed else "활동 중인 담당자"
        reach = (
            f"{goal} '{end_name}'에게 닿기까지"
            if timing.reached_goal
            else f"상향 경로 끝까지 가는 데(끝까지 가도 {goal} 없음)"
        )
        note = (
            f" 대기 시간이 적히지 않은 단계 {timing.unstated_edges}개는 0분으로 계산했습니다."
            if timing.unstated_edges
            else ""
        )
        basis = None if isinstance(svc.attr(AttrKey.SLO_MINUTES), int) else policy_tier_basis(ctx, svc.id)
        if basis is not None:
            # the recovery target is the policy's value for the default tier: a premise, not text
            note += f" 목표 복구 시간은 문서에 없어 {basis}의 등급 기본값을 썼습니다."
        edge_ids = [e.id for e in timing.edges]
        return finding(
            self.info,
            keys=(svc.id,),
            title=self.info.title,
            message=(
                f"'{svc.name}'의 상향 경로({chain})에서 {reach} 적힌 대기 시간만 "
                f"{format_minutes(timing.minutes)}이 걸려 "
                f"목표 복구 시간 {format_minutes(slo)}을 넘습니다.{note}"
            ),
            service_id=svc.id,
            subject_ids=(svc.id, *timing.nodes, *edge_ids),
            evidence=_edge_evidence(timing.edges),
            fix=FixHint(
                kind=FixKind.FIX_DEADLINE,
                params={
                    "service_id": svc.id,
                    "edge_ids": edge_ids,
                    "total_minutes": timing.minutes,
                    "required_minutes": slo,
                },
            ),
        )


# ---------------------------------------------------------------------------- JT-DDL-002


def _deadline_evidence(node: Node) -> list[SourceSpan]:
    attr = node.attrs.get(AttrKey.DEADLINE_MINUTES.value)
    return [*(attr.spans if attr is not None else ()), *node.spans]


#: Appended when a deadline finding rests on an element under HOLD (판단 보류).
_HELD_NOTE = " 다만 이 신고 단계는 판단 보류 상태라 원문 확인이 필요합니다."


def _on_hold(ctx: CheckContext, nodes: Iterable[Node]) -> bool:
    """True if an obligation or a MUST_REPORT edge into it is on HOLD.

    Back-translation holds a MUST_REPORT edge exactly when the rendered deadline/recipient does
    not match the source, so a deadline finding built on it is "확인 불가", not a confirmed break.
    """
    return any(
        n.verification is Verification.HOLD
        or any(e.verification is Verification.HOLD for e in ctx.graph.in_edges(n.id, EdgeKind.MUST_REPORT))
        for n in nodes
    )


def _status(held: bool) -> FindingStatus:
    return FindingStatus.UNKNOWN if held else FindingStatus.BROKEN


@register
class ReportingDeadlineLaterThanLaw:
    info = RuleInfo(
        rule_id="JT-DDL-002",
        title="신고 시한이 법정 기준보다 늦음",
        severity=Severity.CRITICAL,
        element=None,
        rationale=(
            "런북이 법정 기한보다 늦은 신고 시한을 적어 두면, 런북대로 움직인 팀이 그대로 "
            "법정 기한을 넘긴다. 그 신고 단계가 판단 보류 상태면 '확인 불가'로 알린다."
        ),
        legal_basis=f"{law_text(IncidentType.BREACH)} / {law_text(IncidentType.PRIVACY_LEAK)}",
    )

    def check(self, ctx: CheckContext) -> Iterator[Finding]:
        for obligation in ctx.graph.nodes_of(NodeKind.OBLIGATION):
            stated = obligation_deadline(obligation)
            if stated is None:
                continue
            service_id, services = duty_scope(ctx, obligation.id)
            held = _on_hold(ctx, [obligation])
            for rule in applicable_rules(ctx):
                if not satisfies_rule(obligation, rule) or stated <= rule.deadline_minutes:
                    continue
                f = finding(
                    self.info,
                    keys=(rule.key, obligation.id),
                    title=self.info.title,
                    message=(
                        f"'{obligation.name}'의 신고 시한이 {format_minutes(stated)}으로 적혀 있지만, "
                        f"법정 기준은 {rule.recipient}에 {format_minutes(rule.deadline_minutes)} 이내입니다. "
                        "런북대로 하면 법정 기한을 넘깁니다." + (_HELD_NOTE if held else "")
                    ),
                    service_id=service_id,
                    subject_ids=(obligation.id, *services),
                    evidence=_deadline_evidence(obligation),
                    fix=FixHint(
                        kind=FixKind.FIX_DEADLINE,
                        params={"obligation_id": obligation.id, "required_minutes": rule.deadline_minutes},
                    ),
                    status=_status(held),
                )
                yield f.model_copy(update={"legal_basis": rule.legal_basis})


# ---------------------------------------------------------------------------- JT-DDL-003

#: "24시간", "30 분", "3일", "24h", "48 hours", "90 min", "1시간 반". Look-behinds stop dates
#: ("10월 1일"), clock times ("오전 9시 30분") and the tail of longer numbers from matching;
#: "분기"(quarter) and fractions ("3분의 1") are not durations. A trailing "반" (half) counts
#: only when a particle or a break follows it, so "1시간 반드시 보고" stays one hour.
_DURATION = re.compile(
    r"(?<![\d.])(?<!월)(?<!월\s)(?<!\d시)(?<!\d시\s)(\d+(?:\.\d+)?)\s*"
    r"(시간|분|일|hours?|hrs?|h|minutes?|mins?|days?)(?![A-Za-z기]|의\s*\d)"
    r"(\s*반(?=$|\W|이내|내|안|만|까지|동안|정도|쯤|가량|후|뒤|전|이|을|의|에|으로|가|은|는))?",
    re.IGNORECASE,
)
_UNIT_MINUTES: dict[str, int] = {"시간": 60, "분": 1, "일": 1440, "h": 60, "m": 1, "d": 1440}


def _unit_factor(unit: str) -> int:
    unit = unit.lower()
    return _UNIT_MINUTES.get(unit) or _UNIT_MINUTES[unit[0]]


def durations_minutes(text: str) -> list[float]:
    """Every duration stated in ``text``, in order, in minutes.

    Adjacent components with strictly shrinking units form one compound duration ("1시간 30분"
    = 90, "1일 12시간" = 2160, "1h30min" = 90) and "반" adds half a unit ("1시간 반" = 90):
    reading only the first component would make a correct 90-minute deadline look like it
    contradicts its own quote. Values are rounded to thousandths so "0.1시간" equals a stated
    6 minutes.
    """
    out: list[float] = []
    last_end, last_factor = -1, 0
    for match in _DURATION.finditer(text):
        factor = _unit_factor(match.group(2))
        half = 0.5 if match.group(3) and factor >= 60 else 0.0  # "1시간 반", never "30분 반"
        minutes = (float(match.group(1)) + half) * factor
        if out and factor < last_factor and not text[last_end : match.start()].strip():
            out[-1] += minutes
        else:
            out.append(minutes)
        last_end, last_factor = match.end(), factor
    return [round(v, 3) for v in out]


def parse_duration_minutes(text: str) -> float | None:
    """First (possibly compound) duration stated in ``text``, in minutes; ``None`` if there is none."""
    found = durations_minutes(text)
    return found[0] if found else None


def _stated_values(node: Node) -> set[float]:
    """The attribute's value plus the first duration of every quote that does not state it.

    A quote *agrees* with the value when the value is among the durations it states: the
    supporting sentence of a deadline often names other clocks too ("30분 내 팀장 보고 후
    24시간 이내 KISA 신고"), and that is not a contradiction. A quote whose durations all differ
    from the value does contradict it. Without a usable value, every quote contributes its
    first duration, so quotes that disagree with each other still surface.
    """
    stated = obligation_deadline(node)
    values: set[float] = set() if stated is None else {stated}
    attr = node.attrs.get(AttrKey.DEADLINE_MINUTES.value)
    for span in attr.spans if attr is not None else ():
        found = durations_minutes(span.quote)
        if found and (stated is None or not any(math.isclose(v, stated) for v in found)):
            values.add(found[0])
    return values


def _same_duty(a: Node, b: Node) -> bool:
    kind = obligation_incident_type(a)
    return (
        kind is not None
        and kind is obligation_incident_type(b)
        and recipient_matches(obligation_recipient(a), obligation_recipient(b))
    )


def _required_minutes(ctx: CheckContext, nodes: tuple[Node, ...], stated: Iterable[float]) -> float:
    """The strictest of the stated deadlines and any applicable legal one.

    Never an invented number, always lawful, and never looser than something the team already
    wrote: when the runbooks disagree between two internal targets that both beat the law
    (90분 vs 1시간), recommending the legal 24시간 would quietly relax the team's own clock.
    """
    legal = [r.deadline_minutes for r in applicable_rules(ctx) for n in nodes if satisfies_rule(n, r)]
    return float(min([*stated, *legal]))


@register
class ConflictingDeadlines:
    info = RuleInfo(
        rule_id="JT-DDL-003",
        title="같은 의무에 서로 다른 시한",
        severity=Severity.MEDIUM,
        element=None,
        rationale=(
            "같은 기관·같은 사고 유형의 신고 시한이 문서마다(또는 한 문장 안에서) 다르면, "
            "장애 중에 어느 쪽을 따를지 판단하느라 시간을 잃고 늦은 쪽을 따를 위험이 있다."
        ),
    )

    def check(self, ctx: CheckContext) -> Iterator[Finding]:
        obligations = [
            o for o in ctx.graph.nodes_of(NodeKind.OBLIGATION) if obligation_deadline(o) is not None
        ]
        for a, b in itertools.combinations(obligations, 2):
            da, db = obligation_deadline(a), obligation_deadline(b)
            if da is not None and db is not None and da != db and _same_duty(a, b):
                yield self._finding(ctx, (a, b), {da, db}, "pair")
        for node in ctx.graph.nodes_of(NodeKind.OBLIGATION):
            values = _stated_values(node)
            if len(values) > 1:
                yield self._finding(ctx, (node,), values, "spans")

    def _finding(self, ctx: CheckContext, nodes: tuple[Node, ...], values: set[float], mode: str) -> Finding:
        ids = [n.id for n in nodes]
        scopes = [duty_scope(ctx, n.id) for n in nodes]
        services = sorted({s for _, linked in scopes for s in linked})
        # one org-wide side makes the conflict the whole organisation's, not one service's
        org_wide = any(ctx.is_org_obligation(n.id) for n in nodes)
        stated = ", ".join(format_minutes(v) for v in sorted(values))
        names = ", ".join(f"'{n.name}'" for n in nodes)
        where = (
            f"{names}의 원문 근거들이 서로 다른 시한({stated})을 말합니다"
            if mode == "spans"
            else f"같은 의무인 {names}의 시한이 서로 다릅니다({stated})"
        )
        required = _required_minutes(ctx, nodes, values)
        held = _on_hold(ctx, nodes)
        return finding(
            self.info,
            keys=(mode, *ids),
            title=self.info.title,
            message=f"{where}. 하나로 통일해야 합니다(권장: {format_minutes(required)})."
            + (_HELD_NOTE if held else ""),
            service_id=None if org_wide else single_or_none(services),
            subject_ids=(*ids, *services),
            evidence=[s for n in nodes for s in _deadline_evidence(n)],
            fix=FixHint(
                kind=FixKind.FIX_DEADLINE,
                params={
                    "obligation_ids": ids,
                    "stated_minutes": sorted(values),
                    "required_minutes": required,
                },
            ),
            status=_status(held),
        )
