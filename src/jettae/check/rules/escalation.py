"""JT-ESC-* — escalation rules (상향 규칙) for critical services.

Why escalation is checked separately from "is there an owner": the owner is exactly who is
missing at 3 a.m. An escalation chain is the part of the runbook that still works when the first
person does not answer — so it must exist (ESC-001), must reach someone new (ESC-003), must not
loop forever among the same people (ESC-002) and should say how long to wait (ESC-004).

ESC-001/003/004 read the same :func:`jettae.check.elements.analyze_owner` closure that the owner
element score uses. ESC-002 builds the ESCALATES_TO subgraph reachable from the owners and runs
``networkx.simple_cycles`` on it; cycles are canonicalised (rotated to start at the smallest node
id) so the same loop always yields the same finding id.

Two guards keep ESC-002 safe on garbled or hostile input (P5, P6): the number of simple cycles
grows factorially when many people escalate to each other (8 people → 16,064 cycles), so at most
:data:`MAX_CYCLES_EXAMINED` cycles are examined and :data:`MAX_CYCLE_FINDINGS` reported per
service; and node ids are relabelled to integers before enumeration, because networkx picks start
nodes by iterating sets, whose order for *strings* changes between processes (hash randomisation).

As in the JT-PATH rules, a gap that would close if held items (판단 보류) were confirmed is reported
as UNKNOWN rather than BROKEN.
"""

from __future__ import annotations

import itertools
from collections import deque
from collections.abc import Iterable, Iterator

import networkx as nx

from jettae.check.context import CheckContext
from jettae.check.elements import (
    MAX_ESCALATION_HOPS,
    EscalationAnalysis,
    OwnerAnalysis,
    active_responders,
    analyze_owner,
    confirmed_view,
    is_terminal,
    josa,
    label,
    labels,
    service_label,
    spans_of,
    suggest_candidates,
    usable,
)
from jettae.check.registry import finding, register
from jettae.domain import (
    AttrKey,
    EdgeKind,
    Element,
    Finding,
    FindingStatus,
    FixHint,
    FixKind,
    RuleInfo,
    Severity,
)

#: Cycles examined per service (deterministic order) — bounds factorial blow-up on dense loops.
MAX_CYCLES_EXAMINED = 2000
#: Exit-less cycles reported per service; one is enough to act on, twenty keeps the report readable.
MAX_CYCLE_FINDINGS = 20

HOLD_NOTE = " 판단 보류(HOLD) 중인 상향 연결이 맞다고 확인되면 해소되므로 확인 필요로 표시합니다."


def _status(unsure: bool) -> tuple[FindingStatus, str]:
    return (FindingStatus.UNKNOWN, HOLD_NOTE) if unsure else (FindingStatus.BROKEN, "")


def _escalation_candidates(ctx: CheckContext, own: OwnerAnalysis, around: Iterable[str]) -> list[str]:
    return suggest_candidates(ctx, (*own.owner_ids, *own.people_ids, *around))


@register
class EscalationMissing:
    info = RuleInfo(
        rule_id="JT-ESC-001",
        title="상향 경로 없음",
        severity=Severity.HIGH,
        element=Element.OWNER,
        rationale=(
            "담당자가 응답하지 않을 때 다음으로 부를 사람이 정해져 있지 않으면, 담당자 한 명의 부재가 곧 "
            "대응 중단이 됩니다."
        ),
    )

    def check(self, ctx: CheckContext) -> Iterator[Finding]:
        alt = confirmed_view(ctx)
        for svc in ctx.critical_services():
            own = analyze_owner(ctx, svc.id)
            esc = own.escalation
            if not own.owner_ids or esc.edge_ids:
                continue
            subjects = (svc.id, *own.owner_ids, *esc.held_ids)
            status, note = _status(alt is not ctx and bool(analyze_owner(alt, svc.id).escalation.edge_ids))
            yield finding(
                self.info,
                keys=subjects,
                title=self.info.title,
                message=(
                    f"{service_label(ctx, svc.id)}의 담당 {labels(ctx, own.owner_ids)}에게서 이어지는 상향 "
                    f"경로가 없습니다. 담당자가 응답하지 않으면 다음으로 부를 사람이 없습니다.{note}"
                ),
                service_id=svc.id,
                subject_ids=subjects,
                evidence=spans_of(ctx, subjects),
                status=status,
                fix=FixHint(
                    kind=FixKind.ADD_ESCALATION,
                    params={
                        "service_id": svc.id,
                        "from_id": own.owner_ids[0],
                        "from_ids": list(own.owner_ids),
                        "candidates": _escalation_candidates(ctx, own, ()),
                    },
                ),
            )


def _cycle_edges(ctx: CheckContext, cycle: tuple[str, ...], esc: EscalationAnalysis) -> list[str]:
    hops = {(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))}
    edges = (ctx.graph.edges[eid] for eid in esc.edge_ids)
    return sorted(e.id for e in edges if (e.src, e.dst) in hops)


def canonical_cycle(cycle: Iterable[str]) -> tuple[str, ...]:
    """Rotate a cycle so it starts at its smallest node id (direction is preserved)."""
    nodes = list(cycle)
    if not nodes:
        return ()
    pivot = nodes.index(min(nodes))
    return tuple(nodes[pivot:] + nodes[:pivot])


def escalation_cycles(
    ctx: CheckContext, esc: EscalationAnalysis, *, limit: int = MAX_CYCLES_EXAMINED
) -> list[tuple[str, ...]]:
    """Simple cycles (canonical, sorted) of the explored escalation subgraph — at most ``limit``.

    Nodes are relabelled to integers in id order first: networkx chooses where to start each
    search by iterating sets, and integer hashes (unlike string hashes) do not change between
    processes, so the first ``limit`` cycles are the same on every run.
    """
    pairs = sorted({(e.src, e.dst) for e in (ctx.graph.edges[eid] for eid in esc.edge_ids)})
    names = sorted({n for pair in pairs for n in pair})
    index = {name: i for i, name in enumerate(names)}
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(names)))
    graph.add_edges_from((index[a], index[b]) for a, b in pairs)
    cycles = itertools.islice(nx.simple_cycles(graph), max(limit, 0))
    return sorted({canonical_cycle(names[i] for i in cycle) for cycle in cycles})


class _Exits:
    """Memoised "does this loop lead anywhere?" test for one context.

    A loop is harmless if, through usable (verified, available) escalation links, it reaches an
    active holder of a policy terminal role (CTO·대표·CISO …) or an active person outside it.
    """

    def __init__(self, ctx: CheckContext) -> None:
        self.ctx = ctx
        self._people: dict[str, frozenset[str]] = {}
        self._terminal: dict[str, bool] = {}
        #: usable escalation links, built once (out_edges scans the whole graph per call)
        self._next: dict[str, list[str]] = {}
        for e in ctx.graph.edges_of(EdgeKind.ESCALATES_TO):
            if usable(ctx, e) and usable(ctx, ctx.graph.nodes[e.dst]):
                self._next.setdefault(e.src, []).append(e.dst)

    def people(self, nid: str) -> frozenset[str]:
        if nid not in self._people:
            self._people[nid] = frozenset(p.id for p in active_responders(self.ctx, nid))
        return self._people[nid]

    def terminal(self, nid: str) -> bool:
        if nid not in self._terminal:
            node = self.ctx.graph.nodes[nid]
            self._terminal[nid] = is_terminal(self.ctx, node) and bool(self.people(nid))
        return self._terminal[nid]

    def reach(self, starts: Iterable[str]) -> set[str]:
        """Nodes reachable from ``starts`` through usable ESCALATES_TO links (starts excluded)."""
        origin = set(starts)
        seen: set[str] = set()
        queue = deque((s, 0) for s in sorted(origin))
        while queue:
            cur, hops = queue.popleft()
            if hops >= MAX_ESCALATION_HOPS:
                continue
            for dst in self._next.get(cur, ()):
                if dst not in seen and dst not in origin:
                    seen.add(dst)
                    queue.append((dst, hops + 1))
        return seen

    def has_exit(self, cycle: tuple[str, ...]) -> bool:
        members = set(cycle)
        reach = self.reach(members)
        if any(self.terminal(n) for n in sorted(members | reach)):
            return True
        inside = frozenset().union(*(self.people(n) for n in members))
        return any(self.people(n) - inside for n in sorted(reach))


def exitless_cycles(ctx: CheckContext, esc: EscalationAnalysis) -> list[tuple[str, ...]]:
    """Escalation loops that never reach a terminal role or anyone new (at most MAX_CYCLE_FINDINGS)."""
    exits = _Exits(ctx)
    return [c for c in escalation_cycles(ctx, esc) if not exits.has_exit(c)][:MAX_CYCLE_FINDINGS]


@register
class EscalationCycle:
    info = RuleInfo(
        rule_id="JT-ESC-002",
        title="상향 순환",
        severity=Severity.HIGH,
        element=Element.OWNER,
        rationale=(
            "상향이 같은 사람들 사이를 맴돌고 최종 책임자(CTO·대표·CISO 등)나 순환 밖의 다른 사람에게 "
            "이어지지 않으면, 아무도 응답하지 않을 때 호출이 끝없이 되풀이될 뿐 책임 있는 사람에게 닿지 "
            "않습니다."
        ),
    )

    def check(self, ctx: CheckContext) -> Iterator[Finding]:
        alt = confirmed_view(ctx)
        confirmed_exits = _Exits(alt)
        for svc in ctx.critical_services():
            own = analyze_owner(ctx, svc.id)
            for cycle in exitless_cycles(ctx, own.escalation):
                status, note = _status(alt is not ctx and confirmed_exits.has_exit(cycle))
                edge_ids = _cycle_edges(ctx, cycle, own.escalation)
                loop = " → ".join(label(ctx, n) for n in (*cycle, cycle[0]))
                subjects = (svc.id, *cycle, *edge_ids)
                inside = sorted({p.id for n in cycle for p in ctx.resolve_people(n)})
                yield finding(
                    self.info,
                    keys=subjects,
                    title=self.info.title,
                    message=(
                        f"{service_label(ctx, svc.id)}의 상향 경로가 {josa(loop, '으로', '로')} 되돌아오는 "
                        "순환이며, 최종 책임자나 순환 밖의 다른 사람에게 이어지지 않습니다. 아무도 응답하지 "
                        f"않으면 호출이 같은 사람들 사이를 맴돕니다.{note}"
                    ),
                    service_id=svc.id,
                    subject_ids=subjects,
                    evidence=spans_of(ctx, subjects),
                    status=status,
                    fix=FixHint(
                        kind=FixKind.BREAK_ESCALATION_CYCLE,
                        params={
                            "service_id": svc.id,
                            "cycle": list(cycle),
                            "edge_ids": edge_ids,
                            "candidates": suggest_candidates(ctx, cycle, exclude=inside),
                        },
                    ),
                )


@register
class EscalationDeadEnd:
    info = RuleInfo(
        rule_id="JT-ESC-003",
        title="상향 막다른 길",
        severity=Severity.HIGH,
        element=Element.OWNER,
        rationale=(
            "상향 경로가 적혀 있어도 그 끝이 퇴사자·부재자·빈 역할뿐이면, 담당자 외에 실제로 응답할 사람이 "
            "없습니다."
        ),
    )

    def check(self, ctx: CheckContext) -> Iterator[Finding]:
        alt = confirmed_view(ctx)
        for svc in ctx.critical_services():
            own = analyze_owner(ctx, svc.id)
            esc = own.escalation
            if not esc.edge_ids or esc.reaches_responder:
                continue
            status, note = _status(alt is not ctx and analyze_owner(alt, svc.id).escalation.reaches_responder)
            # targets come from the edges: a self-loop (A → A) reaches no *new* node at all
            targets = sorted({ctx.graph.edges[eid].dst for eid in esc.edge_ids})
            ends = list(esc.dead_end_ids) or targets
            subjects = (svc.id, *esc.edge_ids, *targets)
            yield finding(
                self.info,
                keys=subjects,
                title=self.info.title,
                message=(
                    f"{service_label(ctx, svc.id)}의 상향 경로가 {labels(ctx, targets)}까지 이어지지만, "
                    "담당자 외에 활동 중인 대응자가 하나도 없습니다(퇴사·부재·빈 역할). 담당자가 응답하지 "
                    f"않으면 대응이 멈춥니다.{note}"
                ),
                service_id=svc.id,
                subject_ids=subjects,
                evidence=spans_of(ctx, subjects),
                status=status,
                fix=FixHint(
                    kind=FixKind.ADD_ESCALATION,
                    params={
                        "service_id": svc.id,
                        "from_id": ends[0],
                        "from_ids": ends,
                        "candidates": _escalation_candidates(ctx, own, targets),
                    },
                ),
            )


@register
class EscalationWaitMissing:
    info = RuleInfo(
        rule_id="JT-ESC-004",
        title="상향 대기시간 미기재",
        severity=Severity.LOW,
        element=Element.OWNER,
        rationale=(
            "몇 분 동안 응답이 없으면 상향하는지 적혀 있지 않으면, 장애 중에 기다릴지 넘길지를 매번 "
            "즉석에서 판단하게 됩니다."
        ),
    )

    def check(self, ctx: CheckContext) -> Iterator[Finding]:
        for svc in ctx.critical_services():
            own = analyze_owner(ctx, svc.id)
            for eid in own.escalation.untimed_edge_ids:
                edge = ctx.graph.edges[eid]
                subjects = (svc.id, eid)
                yield finding(
                    self.info,
                    keys=subjects,
                    title=self.info.title,
                    message=(
                        f"{service_label(ctx, svc.id)}의 상향({label(ctx, edge.src)} → "
                        f"{label(ctx, edge.dst)})에 몇 분 동안 응답이 없으면 넘기는지(대기 시간)가 적혀 있지 "
                        "않습니다."
                    ),
                    service_id=svc.id,
                    subject_ids=subjects,
                    evidence=spans_of(ctx, subjects),
                    fix=FixHint(
                        kind=FixKind.FIX_DEADLINE,
                        params={"service_id": svc.id, "edge_id": eid, "attr": AttrKey.AFTER_MINUTES.value},
                    ),
                )
