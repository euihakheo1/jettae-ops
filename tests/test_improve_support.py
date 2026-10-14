"""Shared builders for the ``test_improve_*`` suites (no tests here).

``Org`` assembles a small organisation the way ``extract.assemble`` would: roster people carry
ROSTER provenance, their roster media become REACHABLE_VIA links to one channel node per medium,
and runbook facts carry an exact span. ``payment_org()`` is a miniature of the dmc-shop demo
(departed owner, no escalation, ungated approver) so each drafter template can be exercised in
isolation; ``demo_inputs()`` loads the real example bundle.
"""

from __future__ import annotations

from collections.abc import Coroutine, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from jettae.check.context import CheckContext
from jettae.domain import (
    ActionType,
    Attr,
    AttrKey,
    AttrValue,
    ChannelMedium,
    Edge,
    EdgeKind,
    Extractor,
    GraphBuilder,
    IncidentType,
    InputBundle,
    Node,
    NodeKind,
    OpsGraph,
    PersonStatus,
    Policy,
    Provenance,
    RosterEntry,
    SourceKind,
    SourceSpan,
    edge_id,
    node_id,
)

T = TypeVar("T")

DOC = "doc-runbook"
ROSTER = Provenance(source=SourceKind.ROSTER, extractor=Extractor.IMPORT, ref="roster.csv")
CATALOG = Provenance(source=SourceKind.CATALOG, extractor=Extractor.IMPORT, ref="catalog.yaml")
EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "dmc-shop"


def run(coro: Coroutine[Any, Any, T]) -> T:
    """Drive a coroutine that never suspends (fake LLMs answer synchronously) without an event loop.

    Sandboxed Windows runners intermittently fail to create event-loop socket pairs, which has
    nothing to do with the code under test.
    """
    try:
        coro.send(None)
    except StopIteration as stop:
        result: T = stop.value
        return result
    coro.close()
    raise AssertionError("coroutine suspended; the fake LLM should answer synchronously")


@dataclass
class Org:
    """GraphBuilder DSL for small organisations (see the module docstring)."""

    builder: GraphBuilder = field(default_factory=GraphBuilder)
    roster: list[RosterEntry] = field(default_factory=list)
    policy: Policy = field(default_factory=Policy)
    _offset: int = 0

    def runbook(self, quote: str) -> Provenance:
        start = self._offset
        self._offset += len(quote) + 1
        span = SourceSpan(doc_id=DOC, start=start, end=start + len(quote), quote=quote, line=1)
        return Provenance(source=SourceKind.RUNBOOK, extractor=Extractor.RULE, spans=(span,))

    def node(
        self, kind: NodeKind, name: str, prov: Provenance | None = None, **attrs: AttrValue | None
    ) -> str:
        p = prov or self.runbook(name)
        nid = node_id(kind, name)
        values = {k: Attr(value=v, spans=p.spans, source=p.source) for k, v in attrs.items() if v is not None}
        self.builder.add_node(Node(id=nid, kind=kind, name=name, attrs=values, prov=(p,)))
        return nid

    def edge(
        self, kind: EdgeKind, src: str, dst: str, prov: Provenance | None = None, **attrs: AttrValue | None
    ) -> str:
        p = prov or self.runbook(f"{src} {kind.value} {dst}")
        eid = edge_id(kind, src, dst)
        values = {k: Attr(value=v, spans=p.spans, source=p.source) for k, v in attrs.items() if v is not None}
        self.builder.add_edge(Edge(id=eid, kind=kind, src=src, dst=dst, attrs=values, prov=(p,)))
        return eid

    def channel(self, medium: str) -> str:
        return self.node(NodeKind.CHANNEL, medium, ROSTER, medium=medium)

    def person(
        self,
        name: str,
        *,
        title: str | None = None,
        team: str | None = None,
        roles: Iterable[str] = (),
        channels: Iterable[str] = ("slack", "phone"),
        status: PersonStatus = PersonStatus.ACTIVE,
        in_roster: bool = True,
        link_channels: bool = True,
    ) -> str:
        """A roster person; ``link_channels=False`` registers media in the roster only."""
        media = tuple(ChannelMedium(c) for c in channels)
        if in_roster:
            self.roster.append(
                RosterEntry(
                    name=name, status=status, title=title, team=team, roles=tuple(roles), channels=media
                )
            )
        recorded = status if in_roster else PersonStatus.UNKNOWN
        prov = ROSTER if in_roster else self.runbook(name)
        pid = self.node(NodeKind.PERSON, name, prov, status=recorded.value, title=title)
        if team:
            self.edge(EdgeKind.MEMBER_OF, pid, self.node(NodeKind.TEAM, team, ROSTER), ROSTER)
        for role in roles:
            self.edge(EdgeKind.HOLDS_ROLE, pid, self.node(NodeKind.ROLE, role, ROSTER), ROSTER)
        if link_channels:
            for medium in media:
                self.edge(EdgeKind.REACHABLE_VIA, pid, self.channel(medium.value), ROSTER)
        return pid

    def service(self, name: str, tier: int = 1) -> str:
        return self.node(NodeKind.SERVICE, name, tier=tier)

    def trigger(self, sid: str, name: str, *notify: str, channel: str | None = None) -> str:
        tid = self.node(NodeKind.TRIGGER, name)
        self.edge(EdgeKind.DETECTED_BY, sid, tid)
        for target in notify:
            self.edge(EdgeKind.NOTIFIES, tid, target, channel=channel)
        return tid

    def owner(self, sid: str, who: str) -> str:
        return self.edge(EdgeKind.OWNED_BY, sid, who)

    def escalate(self, src: str, dst: str, minutes: int | None = None) -> str:
        return self.edge(EdgeKind.ESCALATES_TO, src, dst, after_minutes=minutes)

    def action(
        self,
        sid: str,
        name: str,
        *,
        kind: ActionType = ActionType.ROLLBACK,
        approvers: Iterable[str] = (),
        executors: Iterable[str] = (),
        requires_approval: bool | None = None,
    ) -> str:
        aid = self.node(NodeKind.ACTION, name, action_type=kind.value, requires_approval=requires_approval)
        self.edge(EdgeKind.MITIGATED_BY, sid, aid)
        for approver in approvers:
            self.edge(EdgeKind.APPROVED_BY, aid, approver)
        for executor in executors:
            self.edge(EdgeKind.EXECUTED_BY, aid, executor)
        return aid

    def obligation(
        self,
        name: str,
        *,
        incident: IncidentType,
        recipient: str,
        deadline: int,
        reporters: Iterable[str] = (),
        service: str | None = None,
    ) -> str:
        oid = self.node(
            NodeKind.OBLIGATION,
            name,
            incident_type=incident.value,
            recipient=recipient,
            deadline_minutes=deadline,
        )
        for reporter in reporters:
            self.edge(EdgeKind.REPORTED_BY, oid, reporter)
        if service is not None:
            self.edge(EdgeKind.MUST_REPORT, service, oid)
        return oid

    def runs_in(self, nid: str, location: str) -> str:
        lid = self.node(NodeKind.LOCATION, location, CATALOG)
        self.edge(EdgeKind.RUNS_IN, nid, lid, CATALOG)
        return lid

    def graph(self) -> OpsGraph:
        return self.builder.build()

    def ctx(self, *, roster: bool = True) -> CheckContext:
        return CheckContext(
            graph=self.graph(), policy=self.policy, roster=tuple(self.roster) if roster else ()
        )


@dataclass(frozen=True)
class PaymentOrg:
    """Ids of :func:`payment_org`."""

    org: Org
    payment: str
    alarm: str
    rollback: str
    departed: str
    oncall: str
    teammate: str
    lead: str
    cto: str
    ceo: str
    ciso: str
    privacy: str


def payment_org() -> PaymentOrg:
    """결제 (tier 1): alarm → departed 김대리 (sole owner), no escalation, rollback without approver.

    결제팀 has 이서연 (1차 당직) and 최민수 (no role); 개발팀 lead 박팀장 (role 팀장) whose role escalates
    to the CTO after 20 min; final-responsibility holders CTO 한서윤, 대표 조영석, CISO 오준석; 윤아름
    holds 개인정보 보호책임자.
    """
    org = Org()
    departed = org.person("김민준", title="대리", team="결제팀", status=PersonStatus.DEPARTED)
    oncall = org.person("이서연", title="사원", team="결제팀", roles=("1차 당직",))
    teammate = org.person("최민수", title="사원", team="결제팀", channels=("slack",))
    lead = org.person(
        "박재현", title="팀장", team="개발팀", roles=("팀장",), channels=("slack", "phone", "kakaotalk")
    )
    cto = org.person("한서윤", title="CTO", team="경영", roles=("CTO",), channels=("phone", "email"))
    ceo = org.person("조영석", title="대표", team="경영", roles=("대표",), channels=("phone",))
    ciso = org.person(
        "오준석", title="이사", team="보안", roles=("CISO", "보안 담당자"), channels=("phone", "email")
    )
    privacy = org.person(
        "윤아름", title="과장", team="경영지원", roles=("개인정보 보호책임자",), channels=("email", "phone")
    )
    org.escalate(node_id(NodeKind.ROLE, "팀장"), cto, 20)
    payment = org.service("결제", tier=1)
    alarm = org.trigger(payment, "결제 실패율 알람", departed)
    org.owner(payment, departed)
    rollback = org.action(payment, "결제 롤백")
    return PaymentOrg(
        org, payment, alarm, rollback, departed, oncall, teammate, lead, cto, ceo, ciso, privacy
    )


def demo_inputs() -> tuple[InputBundle, OpsGraph]:
    """The shipped dmc-shop bundle and its rule-extracted graph (LLM OFF)."""
    from jettae.extract.assemble import assemble
    from jettae.extract.rules import RuleExtractor
    from jettae.ingest import load_bundle_dir

    bundle = load_bundle_dir(EXAMPLE)
    extractions = [
        RuleExtractor().extract(d, roster=bundle.roster, catalog=bundle.catalog) for d in bundle.documents
    ]
    result = assemble(
        extractions,
        bundle.documents,
        roster=bundle.roster,
        catalog=bundle.catalog,
        oncall=bundle.oncall,
        policy=bundle.policy,
    )
    return bundle, result.graph


def demo_ctx(bundle: InputBundle, graph: OpsGraph) -> CheckContext:
    return CheckContext(
        graph=graph, policy=bundle.policy, roster=bundle.roster, oncall=bundle.oncall, catalog=bundle.catalog
    )


def attr_value(graph: OpsGraph, element_id: str, key: AttrKey) -> Any:
    item: Node | Edge = graph.nodes[element_id] if element_id in graph.nodes else graph.edges[element_id]
    return item.attr(key)
