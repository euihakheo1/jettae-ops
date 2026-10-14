"""Read-only query façade over an :class:`OpsGraph` + trusted organisation facts.

Every fixed rule, the scorer, the SPOF analysis, the rehearsal simulator and the patch validator
use the same :class:`CheckContext` so they always agree on questions like "who actually answers
for this service?", "does this channel die together with the service?", "does this step wait for
an approver?" or "which reporting duties bind this service?". Two of those answers are policy
decisions published here once, because a second copy elsewhere would eventually drift:

* **Approval** (:func:`requires_approval`) — an explicit ``requires_approval`` wins; otherwise
  rollback/failover need sign-off; any other step does not. Naming an approver does *not* create
  a requirement: if it did, a patch that adds a valid approver to a restart would turn an
  ungated step (full marks) into a gated one with a single approver (fewer points), so adding a
  responder could lower readiness.
* **Org-wide duties** (:meth:`CheckContext.obligations`) — an OBLIGATION node that no
  MUST_REPORT edge points at was stated by a document that names no service ("침해사고를 인지하면
  … KISA에 신고한다" in a company-wide security procedure). It binds every service, so a
  regulated service is not reported as lacking a step the organisation does have.

No LLM, no I/O. Pure functions of (graph, policy, roster, oncall, catalog, unavailable).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime

from jettae.domain import (
    ActionType,
    AttrKey,
    CatalogService,
    Edge,
    EdgeKind,
    Node,
    NodeKind,
    OnCallShift,
    OpsGraph,
    PersonStatus,
    Policy,
    RosterEntry,
)

#: Mitigations that need sign-off unless the sources explicitly waive it: rolling back or failing
#: over production is the classic step that stalls at 3 a.m. waiting for a manager, and a runbook
#: that is silent about approval has not told us it is unnecessary (P3).
APPROVAL_BY_DEFAULT: frozenset[ActionType] = frozenset({ActionType.ROLLBACK, ActionType.FAILOVER})


def requires_approval(action: Node) -> bool:
    """The single approval rule, shared by score, rules, SPOF analysis and rehearsal.

    An explicit boolean ``requires_approval`` attribute wins; otherwise rollback/failover need
    approval; every other action does not. Approver edges are deliberately ignored: they say
    *who* can approve, not *whether* approval is needed (see the module docstring for why).
    This supersedes the third clause of ARCHITECTURE §4.8 ("an approver edge implies it").
    """
    flag = action.attr(AttrKey.REQUIRES_APPROVAL)
    if isinstance(flag, bool):
        return flag
    kind = action.attr(AttrKey.ACTION_TYPE)
    return isinstance(kind, str) and kind in APPROVAL_BY_DEFAULT


@dataclass(frozen=True)
class CheckContext:
    graph: OpsGraph
    policy: Policy = field(default_factory=Policy)
    roster: tuple[RosterEntry, ...] = ()
    oncall: tuple[OnCallShift, ...] = ()
    catalog: tuple[CatalogService, ...] = ()
    now: datetime | None = None
    #: node ids treated as unavailable (what-if analysis: SPOF, rehearsal pre-checks)
    unavailable: frozenset[str] = frozenset()

    # ------------------------------------------------------------------ derivation
    def with_graph(self, graph: OpsGraph) -> CheckContext:
        return replace(self, graph=graph)

    def with_unavailable(self, ids: Iterable[str]) -> CheckContext:
        return replace(self, unavailable=self.unavailable | frozenset(ids))

    # ------------------------------------------------------------------ services
    def service_tier(self, sid: str) -> int:
        node = self.graph.nodes[sid]
        tier = node.attr(AttrKey.TIER)
        if isinstance(tier, int):
            return tier
        for svc in self.catalog:
            if svc.tier is not None and node.name in (svc.name, *svc.aliases):
                return svc.tier
        return self.policy.default_tier

    def services(self) -> list[Node]:
        return self.graph.nodes_of(NodeKind.SERVICE)

    def critical_services(self) -> list[Node]:
        return [s for s in self.services() if self.service_tier(s.id) in self.policy.critical_tiers]

    def slo_minutes(self, sid: str) -> int:
        node = self.graph.nodes[sid]
        slo = node.attr(AttrKey.SLO_MINUTES)
        if isinstance(slo, int):
            return slo
        tier = self.service_tier(sid)
        return self.policy.tier_slo_minutes.get(tier, max(self.policy.tier_slo_minutes.values()))

    # ------------------------------------------------------------------ people
    def person_status(self, person: Node) -> PersonStatus:
        raw = person.attr(AttrKey.STATUS)
        try:
            return PersonStatus(raw) if raw is not None else PersonStatus.UNKNOWN
        except ValueError:
            return PersonStatus.UNKNOWN

    def is_active(self, person: Node) -> bool:
        return (
            person.kind is NodeKind.PERSON
            and person.id not in self.unavailable
            and self.person_status(person) is PersonStatus.ACTIVE
        )

    def resolve_people(self, nid: str) -> list[Node]:
        """person -> [person]; role -> holders (HOLDS_ROLE); team -> members (MEMBER_OF)."""
        node = self.graph.nodes.get(nid)
        if node is None:
            return []
        if node.kind is NodeKind.PERSON:
            return [node]
        if node.kind is NodeKind.ROLE:
            return self.graph.predecessors(nid, EdgeKind.HOLDS_ROLE)
        if node.kind is NodeKind.TEAM:
            return self.graph.predecessors(nid, EdgeKind.MEMBER_OF)
        return []

    def responders(self, nid: str) -> list[Node]:
        """Active people that a person/role/team node resolves to."""
        return [p for p in self.resolve_people(nid) if self.is_active(p)]

    # ------------------------------------------------------------------ channels
    def contact_channels(self, nid: str) -> list[Node]:
        """Channels by which ``nid`` (person/role/team) can be reached, incl. FALLBACK_TO closure.

        For a role/team, channels attached to the role/team itself plus those of its people.
        """
        direct: list[Node] = list(self.graph.successors(nid, EdgeKind.REACHABLE_VIA))
        node = self.graph.nodes.get(nid)
        if node is not None and node.kind in (NodeKind.ROLE, NodeKind.TEAM):
            for p in self.resolve_people(nid):
                direct.extend(self.graph.successors(p.id, EdgeKind.REACHABLE_VIA))
        seen: dict[str, Node] = {}
        queue = deque(direct)
        while queue:
            ch = queue.popleft()
            if ch.id in seen:
                continue
            seen[ch.id] = ch
            queue.extend(self.graph.successors(ch.id, EdgeKind.FALLBACK_TO))
        return [seen[k] for k in sorted(seen)]

    def available_channels(self, nid: str) -> list[Node]:
        return [c for c in self.contact_channels(nid) if c.id not in self.unavailable]

    def failure_domain(self, nid: str) -> frozenset[str]:
        """Transitive DEPENDS_ON / RUNS_IN closure of any node (systems, services, locations)."""
        out: set[str] = set()
        queue = deque([nid])
        while queue:
            cur = queue.popleft()
            for e in self.graph.out_edges(cur, EdgeKind.DEPENDS_ON, EdgeKind.RUNS_IN):
                if e.dst not in out:
                    out.add(e.dst)
                    queue.append(e.dst)
        return frozenset(out)

    def channel_failure_domain(self, cid: str) -> frozenset[str]:
        return self.failure_domain(cid)

    def shares_failure_domain(self, channel_id: str, service_id: str) -> bool:
        """True if the channel would die together with the service (common-mode failure)."""
        svc_domain = self.failure_domain(service_id) | {service_id}
        return bool(self.channel_failure_domain(channel_id) & svc_domain)

    # ------------------------------------------------------------------ response path
    def owners(self, sid: str) -> list[Node]:
        return self.graph.successors(sid, EdgeKind.OWNED_BY)

    def triggers(self, sid: str) -> list[Node]:
        return self.graph.successors(sid, EdgeKind.DETECTED_BY)

    def notified(self, trigger_id: str) -> list[Edge]:
        return self.graph.out_edges(trigger_id, EdgeKind.NOTIFIES)

    def escalations(self, nid: str) -> list[Edge]:
        return self.graph.out_edges(nid, EdgeKind.ESCALATES_TO)

    def mitigations(self, sid: str) -> list[Node]:
        return self.graph.successors(sid, EdgeKind.MITIGATED_BY)

    def approvers(self, action_id: str) -> list[Node]:
        return self.graph.successors(action_id, EdgeKind.APPROVED_BY)

    def executors(self, action_id: str) -> list[Node]:
        return self.graph.successors(action_id, EdgeKind.EXECUTED_BY)

    def is_org_obligation(self, obligation_id: str) -> bool:
        """True for an OBLIGATION node that no MUST_REPORT edge (held ones included) points at.

        Such a duty comes from a document that names no service, so it binds the whole
        organisation. A held MUST_REPORT edge still ties the duty to its service: dropping it would
        silently widen an unconfirmed, service-specific step into a company-wide one.
        """
        node = self.graph.nodes.get(obligation_id)
        return (
            node is not None
            and node.kind is NodeKind.OBLIGATION
            and not self.graph.in_edges(obligation_id, EdgeKind.MUST_REPORT)
        )

    def org_obligations(self) -> list[Node]:
        """Organisation-wide reporting duties (see :meth:`is_org_obligation`), sorted by id."""
        return [o for o in self.graph.nodes_of(NodeKind.OBLIGATION) if self.is_org_obligation(o.id)]

    def obligations(self, sid: str) -> list[Node]:
        """Reporting duties binding ``sid``: its MUST_REPORT targets plus org-wide ones (by id, once)."""
        found = {o.id: o for o in self.graph.successors(sid, EdgeKind.MUST_REPORT)}
        found.update((o.id, o) for o in self.org_obligations())
        return [found[k] for k in sorted(found)]

    def obligation_services(self, obligation_id: str) -> list[Node]:
        """Services/systems an obligation binds: its MUST_REPORT sources, or every service when org-wide."""
        if self.is_org_obligation(obligation_id):
            return self.services()
        found = {n.id: n for n in self.graph.predecessors(obligation_id, EdgeKind.MUST_REPORT)}
        return [found[k] for k in sorted(found)]

    def reporters(self, obligation_id: str) -> list[Node]:
        return self.graph.successors(obligation_id, EdgeKind.REPORTED_BY)

    def requires_approval(self, action: Node) -> bool:
        """Whether ``action`` waits for an approver — the module-level :func:`requires_approval`."""
        return requires_approval(action)

    def escalation_closure(self, nid: str, *, max_hops: int = 12) -> list[Node]:
        """People/roles/teams reachable through ESCALATES_TO (BFS, cycle-safe), excluding ``nid``."""
        seen: set[str] = {nid}
        order: list[Node] = []
        queue = deque([(nid, 0)])
        while queue:
            cur, hops = queue.popleft()
            if hops >= max_hops:
                continue
            for e in self.escalations(cur):
                if e.dst not in seen:
                    seen.add(e.dst)
                    order.append(self.graph.nodes[e.dst])
                    queue.append((e.dst, hops + 1))
        return order

    # ------------------------------------------------------------------ roster
    def roster_entry(self, person: Node) -> RosterEntry | None:
        for r in self.roster:
            if person.name == r.name or person.name in r.aliases:
                return r
        return None

    def roster_names(self) -> frozenset[str]:
        return frozenset(r.name for r in self.roster)
