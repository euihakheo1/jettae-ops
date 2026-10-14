"""Response-path lineage and minimal cut sets — finding single points of failure (단일 장애점).

The question: *which people or channels, if they alone (or in pairs) became unreachable,
would leave a service with nobody able to respond or approve its mitigation?*

Approach — lineage-driven fault injection (LDFI; P. Alvaro, J. Rosen, J. M. Hellerstein,
"Lineage-driven Fault Injection", SIGMOD 2015): instead of blindly injecting faults, first
record the *lineage* of a successful outcome — every way the response can succeed — as a
boolean formula over component availability, then search only the faults that can falsify
it. Here the "successful outcome" is a working response path and the lineage is:

    OK = Responder ∧ ⋀_{approval-gated mitigation a} Approver(a)

where each requirement is a disjunction of :class:`ReachOption` ``(person, channels)`` —
"this person is available AND at least one of these channels works". Status facts from the
roster (퇴사·장기 부재·명부 미확인) are baked in when the lineage is built; ``ctx.unavailable``
(the what-if set) is applied only at evaluation time, so one lineage answers every what-if
question cheaply. A minimal cut set is then a minimal hitting set of the formula, found by
enumerating candidate subsets in increasing size and pruning supersets of known cuts.

Elements on HOLD (판단 보류) are asymmetric on purpose (P3, 확인 불가 ≠ 정상): a held edge or
node never *satisfies* a requirement (it is not a way to reach anyone — the scorer and the
rehearsal simulator take the same stance), but a held fact that *adds* a requirement (a
mitigation that needs approval) is kept, because dropping it would turn "unknown" into "pass".
Whether a mitigation needs approval at all is the single rule
:meth:`CheckContext.requires_approval`, so this analysis, the score and the replay always gate
the same steps.

No LLM, no I/O, deterministic ordering everywhere (P1, P6).
"""

from __future__ import annotations

import itertools
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass

from jettae.check.context import CheckContext, requires_approval
from jettae.domain import (
    AttrKey,
    Edge,
    EdgeKind,
    Node,
    NodeKind,
    OpsGraph,
    PersonStatus,
    Verification,
)

#: Same cap as :meth:`CheckContext.escalation_closure`, so SPOF and path rules agree.
MAX_ESCALATION_HOPS = 12

RESPONDER_KEY = "responder"

_MEMBERSHIP: dict[NodeKind, EdgeKind] = {
    NodeKind.ROLE: EdgeKind.HOLDS_ROLE,
    NodeKind.TEAM: EdgeKind.MEMBER_OF,
}


@dataclass(frozen=True)
class ReachOption:
    """One way to satisfy a requirement: ``person`` answers via any of ``channel_ids``."""

    person_id: str
    channel_ids: frozenset[str]

    def usable(self, unavailable: frozenset[str]) -> bool:
        return self.person_id not in unavailable and not self.channel_ids <= unavailable


@dataclass(frozen=True)
class Requirement:
    """A disjunction of reach options that must hold for the response path to work."""

    key: str  # RESPONDER_KEY or the action node id whose approval is needed
    label: str  # Korean, for messages: "1차 대응(담당·호출·상향)" / "'롤백' 승인"
    options: tuple[ReachOption, ...]

    def satisfied(self, unavailable: frozenset[str]) -> bool:
        return any(o.usable(unavailable) for o in self.options)


@dataclass(frozen=True)
class ResponseLineage:
    """The lineage (all ways to succeed) of one service's response path."""

    service_id: str
    requirements: tuple[Requirement, ...]
    #: every node/edge id visited while building the lineage (evidence & highlighting)
    path_ids: frozenset[str]

    @property
    def candidates(self) -> tuple[str, ...]:
        """Person and channel ids whose loss can matter — the only fault-injection targets."""
        ids: set[str] = set()
        for req in self.requirements:
            for opt in req.options:
                ids.add(opt.person_id)
                ids.update(opt.channel_ids)
        return tuple(sorted(ids))

    def broken(self, unavailable: Iterable[str] = ()) -> tuple[Requirement, ...]:
        down = frozenset(unavailable)
        return tuple(r for r in self.requirements if not r.satisfied(down))

    def satisfied(self, unavailable: Iterable[str] = ()) -> bool:
        return not self.broken(unavailable)


# ---------------------------------------------------------------------------- graph index


class GraphIndex:
    """Adjacency lists over one graph, in edge-id order like ``OpsGraph.out_edges``, in two views.

    * The *usable* view (``out_edges``, ``membership_edges``, ``resolve_people``) holds only facts
      allowed to show that someone **can** respond: VERIFIED edges between VERIFIED nodes (the
      rehearsal simulator drops the same set). On a graph without HOLD it answers exactly like
      the ``CheckContext`` queries it replaces (the property tests compare against them).
    * ``all_out_edges`` holds every edge, HOLD included. It is used only for facts that *add* a
      requirement (an approval-gated mitigation), see the module docstring.

    Why an index at all: ``OpsGraph.out_edges`` cannot drop HOLD elements and filters edge kinds
    on every call; building a lineage asks dozens of such questions per service. One index per
    check run, keyed by ``(node, edge kind)`` and already restricted to usable facts, turns each
    question into a dict lookup and keeps the HOLD policy in exactly one place.
    """

    def __init__(self, graph: OpsGraph) -> None:
        self.graph = graph
        self._all_out: dict[tuple[str, EdgeKind], list[Edge]] = defaultdict(list)
        self._out: dict[tuple[str, EdgeKind], list[Edge]] = defaultdict(list)
        self._in: dict[tuple[str, EdgeKind], list[Edge]] = defaultdict(list)
        for edge in sorted(graph.edges.values(), key=lambda e: e.id):
            self._all_out[(edge.src, edge.kind)].append(edge)
            usable = edge.verification is Verification.VERIFIED
            if usable and self.is_usable(edge.src) and self.is_usable(edge.dst):
                self._out[(edge.src, edge.kind)].append(edge)
                self._in[(edge.dst, edge.kind)].append(edge)

    def is_usable(self, nid: str) -> bool:
        """The node exists and is not on HOLD."""
        node = self.graph.nodes.get(nid)
        return node is not None and node.verification is Verification.VERIFIED

    def out_edges(self, nid: str, kind: EdgeKind) -> list[Edge]:
        """Usable ``kind`` edges leaving ``nid``."""
        return self._out.get((nid, kind), [])

    def all_out_edges(self, nid: str, kind: EdgeKind) -> list[Edge]:
        """Every ``kind`` edge leaving ``nid``, including those on HOLD."""
        return self._all_out.get((nid, kind), [])

    def membership_edges(self, nid: str) -> list[Edge]:
        """Usable HOLDS_ROLE / MEMBER_OF edges that resolve a role/team ``nid`` to its people."""
        node = self.graph.nodes.get(nid)
        link = _MEMBERSHIP.get(node.kind) if node is not None else None
        return self._in.get((nid, link), []) if link is not None else []

    def resolve_people(self, nid: str) -> list[Node]:
        """:meth:`CheckContext.resolve_people` restricted to the usable view."""
        if not self.is_usable(nid):
            return []
        node = self.graph.nodes[nid]
        if node.kind is NodeKind.PERSON:
            return [node]
        return [self.graph.nodes[e.src] for e in self.membership_edges(nid)]


def gated_actions(index: GraphIndex, sid: str) -> list[tuple[Edge, Node]]:
    """(MITIGATED_BY edge, action) for every mitigation of ``sid`` that needs approval.

    Held mitigations are included: an unconfirmed approval requirement is still a requirement.
    The decision per action is :func:`jettae.check.context.requires_approval`.
    """
    out: list[tuple[Edge, Node]] = []
    for edge in index.all_out_edges(sid, EdgeKind.MITIGATED_BY):
        action = index.graph.nodes[edge.dst]
        if requires_approval(action):
            out.append((edge, action))
    return out


def escalation_edges(index: GraphIndex, nid: str) -> list[Edge]:
    """Usable ESCALATES_TO edges leaving ``nid`` or any person it resolves to (sorted by edge id).

    Runbooks state escalation both at role level ("결제 담당 미응답 시 팀장") and at person level
    ("김대리 미응답 30분 후 박과장"); a chain entered through a role must follow both.
    """
    sources = {nid, *(p.id for p in index.resolve_people(nid))}
    edges = {e.id: e for src in sources for e in index.out_edges(src, EdgeKind.ESCALATES_TO)}
    return [edges[k] for k in sorted(edges)]


# ---------------------------------------------------------------------------- lineage build


class _LineageBuilder:
    """Walks the response path once, recording visited ids and reach options."""

    def __init__(self, ctx: CheckContext, index: GraphIndex) -> None:
        self.ctx = ctx
        self.index = index
        self.graph = ctx.graph
        self.path: set[str] = set()

    # ---- channels -------------------------------------------------------------
    def channel_closure(self, seeds: Iterable[str]) -> set[str]:
        """Seed channels plus their FALLBACK_TO closure (same semantics as ``contact_channels``)."""
        out: set[str] = set()
        stack = list(seeds)
        while stack:
            cid = stack.pop()
            if cid in out:
                continue
            out.add(cid)
            self.path.add(cid)
            for edge in self.index.out_edges(cid, EdgeKind.FALLBACK_TO):
                self.path.add(edge.id)
                stack.append(edge.dst)
        return out

    def own_channels(self, nid: str) -> set[str]:
        edges = self.index.out_edges(nid, EdgeKind.REACHABLE_VIA)
        self.path.update(e.id for e in edges)
        return self.channel_closure(e.dst for e in edges)

    def notify_channels(self, notify_attr: object) -> set[str]:
        """The channel a NOTIFIES edge pages through, if it names an existing, usable channel node."""
        node = self.graph.nodes.get(notify_attr) if isinstance(notify_attr, str) else None
        if node is None or node.kind is not NodeKind.CHANNEL or not self.index.is_usable(node.id):
            return set()
        return self.channel_closure([node.id])

    # ---- people ---------------------------------------------------------------
    def options(self, via: str, extra_channels: set[str]) -> dict[str, set[str]]:
        """Active people behind ``via`` -> channels that reach them.

        Channels attached to a role/team (e.g. the on-call pager) reach whoever holds it, so they
        are shared by all of its people. Inactive people are dropped here: their status is a
        roster fact, not a what-if.
        """
        node = self.graph.nodes.get(via)
        if node is None or not self.index.is_usable(via):
            return {}
        self.path.add(via)
        shared = set(extra_channels)
        if node.kind in _MEMBERSHIP:
            shared |= self.own_channels(via)
            self.path.update(e.id for e in self.index.membership_edges(via))
        out: dict[str, set[str]] = {}
        for person in self.index.resolve_people(via):
            self.path.add(person.id)
            channels = self.own_channels(person.id) | shared
            if self.ctx.person_status(person) is PersonStatus.ACTIVE and channels:
                out[person.id] = channels
        return out

    # ---- escalation -----------------------------------------------------------
    def escalation_targets(self, starts: Iterable[str]) -> list[str]:
        """BFS over :func:`escalation_edges` (cycle-safe, hop-capped), recording visited edges."""
        seen = set(starts)
        order: list[str] = []
        queue = deque((nid, 0) for nid in sorted(seen))
        while queue:
            cur, hops = queue.popleft()
            if hops >= MAX_ESCALATION_HOPS:
                continue
            for edge in escalation_edges(self.index, cur):
                self.path.add(edge.id)
                if edge.dst not in seen:
                    seen.add(edge.dst)
                    order.append(edge.dst)
                    queue.append((edge.dst, hops + 1))
        return order

    # ---- requirements ---------------------------------------------------------
    def responder(self, sid: str) -> Requirement:
        entries: list[tuple[str, set[str]]] = []
        for edge in self.index.out_edges(sid, EdgeKind.OWNED_BY):
            self.path.add(edge.id)
            entries.append((edge.dst, set()))
        for det in self.index.out_edges(sid, EdgeKind.DETECTED_BY):
            self.path.update((det.id, det.dst))
            for notify in self.index.out_edges(det.dst, EdgeKind.NOTIFIES):
                self.path.add(notify.id)
                entries.append((notify.dst, self.notify_channels(notify.attr(AttrKey.CHANNEL))))
        merged: dict[str, set[str]] = {}
        for via, extra in entries:
            _merge(merged, self.options(via, extra))
        for via in self.escalation_targets(via for via, _ in entries):
            _merge(merged, self.options(via, set()))
        return Requirement(RESPONDER_KEY, "1차 대응(담당·호출·상향)", _freeze(merged))

    def approvals(self, sid: str) -> list[Requirement]:
        out: list[Requirement] = []
        for mit, action in gated_actions(self.index, sid):
            self.path.update((mit.id, action.id))
            merged: dict[str, set[str]] = {}
            for edge in self.index.out_edges(action.id, EdgeKind.APPROVED_BY):
                self.path.add(edge.id)
                _merge(merged, self.options(edge.dst, set()))
            out.append(Requirement(action.id, f"'{action.name}' 승인", _freeze(merged)))
        return out


def _merge(into: dict[str, set[str]], new: dict[str, set[str]]) -> None:
    for pid, channels in new.items():
        into.setdefault(pid, set()).update(channels)


def _freeze(options: dict[str, set[str]]) -> tuple[ReachOption, ...]:
    return tuple(ReachOption(pid, frozenset(options[pid])) for pid in sorted(options))


# ---------------------------------------------------------------------------- public API


def response_lineage(ctx: CheckContext, sid: str, *, index: GraphIndex | None = None) -> ResponseLineage:
    """Build the lineage of ``sid``'s response path (see module docstring for the formula).

    Responder: owners and the targets of the service's alert triggers (with the alert's own
    channel as an extra way in), then everyone in their escalation closure. Approver: for every
    mitigation that needs approval, the people its ``APPROVED_BY`` edges resolve to. Only
    verified (non-HOLD) facts provide options. Pass a shared ``index`` (built from
    ``ctx.graph``) when analysing many services.
    """
    node = ctx.graph.nodes.get(sid)
    if node is None or node.kind not in (NodeKind.SERVICE, NodeKind.SYSTEM):
        raise ValueError(f"서비스·시스템 노드가 아니라 대응 경로를 분석할 수 없습니다: {sid}")
    if index is not None and index.graph is not ctx.graph:
        raise ValueError("GraphIndex가 다른 그래프로 만들어져 대응 경로를 분석할 수 없습니다")
    builder = _LineageBuilder(ctx, index if index is not None else GraphIndex(ctx.graph))
    builder.path.add(sid)
    requirements = (builder.responder(sid), *builder.approvals(sid))
    return ResponseLineage(sid, requirements, frozenset(builder.path))


def response_path_ok(ctx: CheckContext, sid: str) -> bool:
    """True if someone active can respond to ``sid`` and every gated mitigation has a reachable approver.

    "Reachable" = an active person (roster status) not in ``ctx.unavailable`` with at least one
    contact channel not in ``ctx.unavailable``, connected through verified (non-HOLD) facts only.
    """
    return response_lineage(ctx, sid).satisfied(ctx.unavailable)


def minimal_cut_sets(
    lineage: ResponseLineage, *, unavailable: frozenset[str] = frozenset(), max_size: int = 2
) -> list[frozenset[str]]:
    """Minimal sets of candidates whose joint loss (on top of ``unavailable``) breaks ``lineage``.

    Subsets are enumerated by increasing size; any superset of an already-found cut is skipped,
    so every returned set is minimal (all its strict subsets were tested and none was a cut).
    """
    if max_size < 0:
        raise ValueError("max_size는 0 이상이어야 합니다")
    if not lineage.satisfied(unavailable):
        return [frozenset()]
    candidates = [c for c in lineage.candidates if c not in unavailable]
    cuts: list[frozenset[str]] = []
    for size in range(1, max_size + 1):
        for combo in itertools.combinations(candidates, size):
            subset = frozenset(combo)
            if any(cut <= subset for cut in cuts):
                continue
            if not lineage.satisfied(unavailable | subset):
                cuts.append(subset)
    return sorted(cuts, key=lambda s: (len(s), sorted(s)))


def response_cut_sets(ctx: CheckContext, sid: str, max_size: int = 2) -> list[frozenset[str]]:
    """Minimal PERSON/CHANNEL id sets whose joint unavailability breaks ``sid``'s response path.

    Sorted by ``(size, sorted ids)``. Returns ``[frozenset()]`` when the path is already broken
    under ``ctx.unavailable`` (the empty set is then the unique minimal cut).
    """
    return minimal_cut_sets(response_lineage(ctx, sid), unavailable=ctx.unavailable, max_size=max_size)
