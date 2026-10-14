"""Graph ↔ canonical facts: score any operations graph against the sealed ground truth.

Why a canonical fact view: the gold truth (:func:`jettae.evalkit.gold.gold_facts`) names things the
way the *spec* does, while an assembled graph names them the way the *runbook* does — a trigger is
"결제 실패율 5% 초과 시 즉시 알람", an action "결제 롤백", an obligation "KISA 침해사고 신고". Field-level
precision/recall is only meaningful when both sides use one vocabulary, so :func:`graph_facts`
maps graph nodes to exactly the keys :mod:`~jettae.evalkit.gold` uses:

* persons by roster name, teams/roles/services/systems/locations by name;
* channels by medium value (``"slack"``), because the assembler keeps one node per medium;
* triggers as ``"{service}#trigger"`` and actions as ``"{service}#{action_type}"`` for every service
  that points at them (DETECTED_BY / MITIGATED_BY) — runbooks name those freely;
* obligations by their recipient (``"KISA"``), the way the runbook states the duty.

Edge attributes are reduced to :data:`~jettae.evalkit.gold.FACT_ATTR_KEYS`; an attribute the graph
does not state stays absent (never guessed), so a missing value shows up as a field mismatch.

:func:`oracle_graph` is the inverse: it builds the graph a perfect extractor would produce from a
fact set plus the trusted roster/catalog. It is the upper bound (condition O) and the sanity check
of the rules themselves — if the fixed rules miss a fault on the oracle graph, the miss is the
judge's, not the extractor's. :func:`ungrounded_elements` audits the P2 guarantee ("원문 근거 위치
없는 요소 0건") on any graph against its source documents.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

from jettae.domain import (
    ActionType,
    Attr,
    AttrKey,
    CatalogService,
    ChannelMedium,
    Document,
    Edge,
    EdgeKind,
    Extractor,
    GraphBuilder,
    Node,
    NodeKind,
    OpsGraph,
    PersonStatus,
    Provenance,
    RosterEntry,
    SourceKind,
    SourceSpan,
    edge_id,
    node_id,
    slug,
)
from jettae.domain.graph import AttrValue
from jettae.evalkit.gold import FACT_ATTR_KEYS, Fact, FactValue, action_key, fact, trigger_key

#: ``Provenance.ref`` of every element of an oracle graph (the gold spec, not a document).
ORACLE_REF = "gold"

#: Edge kinds only the runbook can state. Roster and catalog facts (membership, roles, contact
#: media, dependencies, locations) are imported verbatim from trusted files, so scoring extraction
#: on them would reward the importer, not the extractor.
RUNBOOK_ONLY_EDGE_KINDS: frozenset[EdgeKind] = frozenset(
    {
        EdgeKind.OWNED_BY,
        EdgeKind.DETECTED_BY,
        EdgeKind.NOTIFIES,
        EdgeKind.ESCALATES_TO,
        EdgeKind.MITIGATED_BY,
        EdgeKind.APPROVED_BY,
        EdgeKind.EXECUTED_BY,
        EdgeKind.FALLBACK_TO,
        EdgeKind.MUST_REPORT,
        EdgeKind.REPORTED_BY,
    }
)

#: Attribute keys that hold minutes (ints in both the gold facts and the graph).
_MINUTE_FACT_KEYS = frozenset(
    {AttrKey.AFTER_MINUTES.value, AttrKey.DEADLINE_MINUTES.value, AttrKey.DELAY_MINUTES.value}
)


# --------------------------------------------------------------------------- graph -> facts


def canonical_keys(graph: OpsGraph) -> dict[str, tuple[str, ...]]:
    """Node id -> the canonical fact name(s) of that node (sorted, usually exactly one).

    A trigger or action that several services point at gets one key per service: the gold truth
    is per service, and a shared node really does state the fact for each of them.
    """
    keys: dict[str, tuple[str, ...]] = {}
    for node in graph.nodes.values():
        keys[node.id] = _keys_of(graph, node)
    return keys


def _keys_of(graph: OpsGraph, node: Node) -> tuple[str, ...]:
    if node.kind is NodeKind.CHANNEL:
        medium = node.attr(AttrKey.MEDIUM)
        return (medium if isinstance(medium, str) and medium else node.name,)
    if node.kind is NodeKind.OBLIGATION:
        recipient = node.attr(AttrKey.RECIPIENT)
        # stripped: " KISA" is the gold recipient "KISA", the same duty with stray whitespace
        return (recipient.strip() if isinstance(recipient, str) and recipient.strip() else node.name,)
    if node.kind is NodeKind.TRIGGER:
        parents = _parent_services(graph, node.id, EdgeKind.DETECTED_BY)
        return tuple(sorted({trigger_key(s) for s in parents})) or (node.name,)
    if node.kind is NodeKind.ACTION:
        return _action_keys(graph, node)
    return (node.name,)


def _action_keys(graph: OpsGraph, node: Node) -> tuple[str, ...]:
    """``{service}#{action_type}`` per mitigated service; an untyped action keeps its own name
    after the ``#`` so it can never be mistaken for the gold action of that service."""
    parents = _parent_services(graph, node.id, EdgeKind.MITIGATED_BY)
    if not parents:
        return (node.name,)
    action_type = _action_type(node)
    if action_type is None:
        return tuple(sorted({f"{s}#{node.name}" for s in parents}))
    return tuple(sorted({action_key(s, action_type) for s in parents}))


def _parent_services(graph: OpsGraph, nid: str, kind: EdgeKind) -> list[str]:
    return [graph.nodes[e.src].name for e in graph.edges.values() if e.kind is kind and e.dst == nid]


def _action_type(node: Node) -> ActionType | None:
    raw = node.attr(AttrKey.ACTION_TYPE)
    try:
        return ActionType(raw) if isinstance(raw, str) else None
    except ValueError:
        return None


def _fact_value(value: object) -> FactValue | None:
    """Only whole numbers (minutes) and non-empty strings are comparable fact values.

    ``bool`` is not a number here. A float counts only when it is exactly a whole number: an
    approved patch may store ``15.0`` minutes for the "15분" the gold truth states as ``15``,
    and that is the same fact, while ``15.5`` is not any gold value and is never rounded into one.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str) and value:
        return value
    return None


def _edge_attrs(graph: OpsGraph, edge: Edge, keys: Mapping[str, tuple[str, ...]]) -> dict[str, FactValue]:
    """The edge's :data:`FACT_ATTR_KEYS`, read where the graph stores them (edge or child node)."""
    out: dict[str, FactValue] = {}
    wanted = FACT_ATTR_KEYS.get(edge.kind, ())
    if not wanted:
        return out
    if edge.kind in (EdgeKind.MITIGATED_BY, EdgeKind.MUST_REPORT):
        holder: Node | Edge = graph.nodes[edge.dst]
    else:
        holder = edge
    for key in wanted:
        raw = holder.attr(key)
        if key == AttrKey.CHANNEL.value and isinstance(raw, str) and raw in keys:
            raw = keys[raw][0]  # the graph stores the channel node id; the fact names the medium
        value = _fact_value(raw)
        if value is not None and (key not in _MINUTE_FACT_KEYS or isinstance(value, int)):
            out[key] = value
    return out


def graph_facts(graph: OpsGraph) -> frozenset[Fact]:
    """Every edge of ``graph`` as canonical :class:`Fact` s (same vocabulary as ``gold_facts``).

    Elements on HOLD (판단 보류) are included: they are what the extractor produced, and the
    hallucination measurement must count an invented fact even when back-translation held it.
    """
    keys = canonical_keys(graph)
    out: set[Fact] = set()
    for edge in graph.edges.values():
        attrs = _edge_attrs(graph, edge, keys)
        for src in keys[edge.src]:
            for dst in keys[edge.dst]:
                out.add(fact(edge.kind, src, dst, **attrs))
    return frozenset(out)


# --------------------------------------------------------------------------- facts -> graph

#: Node kind of each endpoint of a gold fact. ``None`` = a catalog entity: SERVICE, or SYSTEM
#: when the catalog says so (``depends_on`` targets and their ``runs_in`` edges).
_ORACLE_ENDPOINTS: Mapping[EdgeKind, tuple[NodeKind | None, NodeKind | None]] = {
    EdgeKind.DEPENDS_ON: (None, None),
    EdgeKind.RUNS_IN: (None, NodeKind.LOCATION),
    EdgeKind.OWNED_BY: (None, NodeKind.PERSON),
    EdgeKind.DETECTED_BY: (None, NodeKind.TRIGGER),
    EdgeKind.NOTIFIES: (NodeKind.TRIGGER, NodeKind.PERSON),
    EdgeKind.ESCALATES_TO: (NodeKind.PERSON, NodeKind.PERSON),
    EdgeKind.MITIGATED_BY: (None, NodeKind.ACTION),
    EdgeKind.APPROVED_BY: (NodeKind.ACTION, NodeKind.PERSON),
    EdgeKind.EXECUTED_BY: (NodeKind.ACTION, NodeKind.PERSON),
    EdgeKind.REACHABLE_VIA: (NodeKind.PERSON, NodeKind.CHANNEL),
    EdgeKind.FALLBACK_TO: (NodeKind.CHANNEL, NodeKind.CHANNEL),
    EdgeKind.HOLDS_ROLE: (NodeKind.PERSON, NodeKind.ROLE),
    EdgeKind.MEMBER_OF: (NodeKind.PERSON, NodeKind.TEAM),
    EdgeKind.MUST_REPORT: (None, NodeKind.OBLIGATION),
    EdgeKind.REPORTED_BY: (NodeKind.OBLIGATION, NodeKind.PERSON),
}


class _OracleBuilder:
    """Builds the oracle graph; every element carries ``Provenance(POLICY, IMPORT, ref="gold")``."""

    def __init__(self, roster: Sequence[RosterEntry], catalog: Sequence[CatalogService]) -> None:
        self.prov = (Provenance(source=SourceKind.POLICY, extractor=Extractor.IMPORT, ref=ORACLE_REF),)
        self.builder = GraphBuilder()
        self.people = {e.name: e for e in roster}
        self.catalog = {e.name: e for e in catalog}

    @staticmethod
    def attr(value: AttrValue) -> Attr:
        return Attr(value=value, source=SourceKind.POLICY)

    def node(self, kind: NodeKind, name: str, attrs: Mapping[str, AttrValue] | None = None) -> str:
        nid = node_id(kind, name)
        values = {k: self.attr(v) for k, v in (attrs or {}).items()}
        self.builder.add_node(Node(id=nid, kind=kind, name=name, attrs=values, prov=self.prov))
        return nid

    def edge(self, kind: EdgeKind, src: str, dst: str, attrs: Mapping[str, AttrValue] | None = None) -> None:
        values = {k: self.attr(v) for k, v in (attrs or {}).items()}
        eid = edge_id(kind, src, dst)
        current = self.builder.edges.get(eid)
        if current is not None and {k: a.value for k, a in current.attrs.items()} != dict(attrs or {}):
            # two facts state different values for one relation: keep both, like the assembler
            eid = edge_id(kind, src, dst, "+".join(slug(f"{k}-{v}") for k, v in sorted(values.items())))
        self.builder.add_edge(Edge(id=eid, kind=kind, src=src, dst=dst, attrs=values, prov=self.prov))

    def entity(self, kind: NodeKind | None, name: str) -> str:
        """Endpoint node for ``name`` with the trusted attributes the rules read."""
        if kind is None:
            entry = self.catalog.get(name)
            is_system = entry is not None and entry.kind == "system"
            tier = {AttrKey.TIER.value: entry.tier} if entry is not None and entry.tier is not None else {}
            return self.node(NodeKind.SYSTEM if is_system else NodeKind.SERVICE, name, tier)
        if kind is NodeKind.PERSON:
            return self.node(kind, name, self._person_attrs(name))
        if kind is NodeKind.CHANNEL:
            medium = {AttrKey.MEDIUM.value: name} if name in _MEDIA else {}
            return self.node(kind, name, medium)
        return self.node(kind, name)

    def _person_attrs(self, name: str) -> dict[str, AttrValue]:
        entry = self.people.get(name)
        if entry is None:  # a name the roster cannot confirm is unknown, never active (P3)
            return {AttrKey.STATUS.value: PersonStatus.UNKNOWN.value}
        attrs: dict[str, AttrValue] = {AttrKey.STATUS.value: entry.status.value}
        if entry.title:
            attrs[AttrKey.TITLE.value] = entry.title
        if entry.aliases:
            attrs[AttrKey.ALIASES.value] = list(entry.aliases)
        return attrs


_MEDIA = frozenset(m.value for m in ChannelMedium)


def oracle_graph(
    facts: Iterable[Fact],
    *,
    roster: Sequence[RosterEntry] = (),
    catalog: Sequence[CatalogService] = (),
) -> OpsGraph:
    """The graph a perfect extractor would assemble from ``facts`` (inverse of :func:`graph_facts`).

    Node attributes come from where the product reads them: person status/title/aliases from the
    roster, service tier and service/system kind from the catalog, the action type from the
    MITIGATED_BY fact, deadline/incident type/recipient from the MUST_REPORT fact. Obligations are
    one node per (service, recipient, incident type) — a shared node would merge the deadlines of
    unrelated services — and every REPORTED_BY fact of a recipient applies to each of its nodes,
    exactly as the person-level gold truth states it.
    """
    ob = _OracleBuilder(roster, catalog)
    fact_list = sorted(set(facts))
    obligations: dict[str, list[str]] = defaultdict(list)
    for f in fact_list:
        if f.kind is EdgeKind.MUST_REPORT:
            attrs = dict(f.attrs)
            incident = str(attrs.get(AttrKey.INCIDENT_TYPE.value, "")).strip()
            name = " ".join(x for x in (f.src, f.dst, incident) if x)
            oid = ob.node(NodeKind.OBLIGATION, name, {AttrKey.RECIPIENT.value: f.dst, **attrs})
            ob.edge(EdgeKind.MUST_REPORT, ob.entity(None, f.src), oid)
            obligations[f.dst].append(oid)
    for f in fact_list:
        if f.kind is EdgeKind.MUST_REPORT:
            continue
        src_kind, dst_kind = _ORACLE_ENDPOINTS[f.kind]
        attrs = dict(f.attrs)
        if f.kind is EdgeKind.REPORTED_BY:
            targets = obligations.get(f.src) or [
                ob.node(NodeKind.OBLIGATION, f.src, {AttrKey.RECIPIENT.value: f.src})
            ]
            person = ob.entity(NodeKind.PERSON, f.dst)
            for oid in targets:
                ob.edge(EdgeKind.REPORTED_BY, oid, person)
            continue
        if f.kind is EdgeKind.MITIGATED_BY:
            action_attrs = {k: v for k, v in attrs.items() if k == AttrKey.ACTION_TYPE.value}
            aid = ob.node(NodeKind.ACTION, f.dst, action_attrs)
            ob.edge(EdgeKind.MITIGATED_BY, ob.entity(None, f.src), aid)
            continue
        src = ob.entity(src_kind, f.src)
        dst = ob.entity(dst_kind, f.dst)
        if f.kind is EdgeKind.NOTIFIES and AttrKey.CHANNEL.value in attrs:
            channel = str(attrs[AttrKey.CHANNEL.value])
            attrs[AttrKey.CHANNEL.value] = ob.entity(NodeKind.CHANNEL, channel)
        ob.edge(f.kind, src, dst, attrs)
    return ob.builder.build()


# --------------------------------------------------------------------------- grounding audit


def ungrounded_elements(graph: OpsGraph, docs: Iterable[Document]) -> int:
    """How many elements break the grounding guarantee (KPI: must be 0).

    An element is ungrounded when it has no provenance at all, when a runbook-sourced attribute
    has no span, or when any of its runbook spans does not reproduce its quote at the stated
    offsets of a known document, or lies in a quarantined (prompt-injection) range. Each node,
    edge and attribute counts once.
    """
    doc_map = {d.doc_id: d for d in docs}
    bad = 0
    elements: list[Node | Edge] = [*graph.nodes.values(), *graph.edges.values()]
    for element in elements:
        if not element.prov or not all(_span_ok(s, doc_map) for s in element.spans):
            bad += 1
        for value in element.attrs.values():
            if value.source is SourceKind.RUNBOOK and (
                not value.spans or not all(_span_ok(s, doc_map) for s in value.spans)
            ):
                bad += 1
    return bad


def _span_ok(span: SourceSpan, docs: Mapping[str, Document]) -> bool:
    doc = docs.get(span.doc_id)
    return (
        doc is not None
        and doc.text[span.start : span.end] == span.quote
        and not doc.is_quarantined(span.start, span.end)
    )
