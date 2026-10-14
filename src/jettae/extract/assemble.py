"""Graph assembly: the single funnel from extractor output to an :class:`OpsGraph`.

Both the rule parser and the LLM extractor emit the same :class:`Extraction` IR. Everything
that reaches the graph passes through here, which is what makes the "rule vs LLM" comparison
fair and the hallucination guard universal:

1. **Trusted facts first** — roster people/roles/teams/channels, on-call rotations and catalog
   services/dependencies/locations/owners are imported with ROSTER/ONCALL/CATALOG provenance.
   Because :class:`GraphBuilder` lets stronger sources win, a runbook can never overwrite a
   roster fact such as "퇴사" (P2, P3).
2. **Grounding** — every entity, relation and attribute quote is aligned back to the document
   (:mod:`jettae.extract.align`). Unalignable → dropped; found only in quarantine → dropped.
3. **Validation & coercion** — relations must satisfy ``EDGE_SIGNATURES``; attribute values
   are coerced to their documented types or dropped (absent stays absent, never guessed).
4. **Entity resolution** — surfaces become roster/catalog identities
   (:mod:`jettae.extract.resolve`); ambiguous people become HOLD nodes, unknown people become
   ``status=unknown`` nodes so fixed rules report them instead of trusting them.
5. **Provenance** — runbook elements carry ``Provenance(RUNBOOK, extractor, spans)``.

No LLM is used here (P1/P4), and the output is a pure function of the inputs (P6).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from jettae.domain import (
    EDGE_SIGNATURES,
    ActionType,
    Attr,
    AttrKey,
    CatalogService,
    ChannelMedium,
    Document,
    DocumentRef,
    Edge,
    EdgeKind,
    EntityRef,
    ExtractedAttr,
    Extraction,
    Extractor,
    GraphBuilder,
    GroundingStatus,
    IncidentType,
    LocationKind,
    Node,
    NodeKind,
    OnCallShift,
    OpsGraph,
    PersonStatus,
    Policy,
    Provenance,
    RosterEntry,
    SourceKind,
    SourceSpan,
    Verification,
    edge_id,
    node_id,
    slug,
)
from jettae.domain.graph import AttrValue
from jettae.extract.align import AlignMethod, align_quote, ground_quote
from jettae.extract.resolve import (
    CATALOG_CHANNEL_KIND,
    MEDIUM_LABEL_KO,
    EntityResolver,
    ResolutionStatus,
    clean_surface,
    compact,
    display_name,
    find_durations,
    find_media,
    is_range_bound,
    medium_of,
    names_nothing,
    parse_bool,
    parse_minutes,
    parse_tier,
    service_key,
)

#: Cap on human-readable drop examples kept in :class:`GroundingStatus` (reports stay small).
MAX_DROPPED_EXAMPLES = 20
#: On-call levels that always have a canonical role name ("1차 당직"), even without a table,
#: so "1차 당직자" in a runbook names the same ROLE node whether or not on-call data is loaded.
DEFAULT_ONCALL_LEVELS: tuple[int, ...] = (1, 2)

_SERVICE_KINDS = frozenset({NodeKind.SERVICE, NodeKind.SYSTEM})
_RESOLVER_KINDS = frozenset({NodeKind.PERSON, NodeKind.ROLE})
#: Edge kinds that attach a service-specific child (조치·감지·신고 의무) to its service.
_PARENT_EDGES: dict[EdgeKind, NodeKind] = {
    EdgeKind.MITIGATED_BY: NodeKind.ACTION,
    EdgeKind.DETECTED_BY: NodeKind.TRIGGER,
    EdgeKind.MUST_REPORT: NodeKind.OBLIGATION,
}
_MINUTE_KEYS = frozenset(
    {
        AttrKey.AFTER_MINUTES.value,
        AttrKey.DEADLINE_MINUTES.value,
        AttrKey.DELAY_MINUTES.value,
        AttrKey.DURATION_MINUTES.value,
        AttrKey.SLO_MINUTES.value,
    }
)
_ACTION_SYNONYMS = {
    "롤백": ActionType.ROLLBACK,
    "원복": ActionType.ROLLBACK,
    "되돌리기": ActionType.ROLLBACK,
    "페일오버": ActionType.FAILOVER,
    "전환": ActionType.FAILOVER,
    "절체": ActionType.FAILOVER,
    "재시작": ActionType.RESTART,
    "재기동": ActionType.RESTART,
    "기능플래그": ActionType.FEATURE_FLAG,
    "플래그": ActionType.FEATURE_FLAG,
    "증설": ActionType.SCALE,
    "스케일": ActionType.SCALE,
    "확장": ActionType.SCALE,
}
_INCIDENT_SYNONYMS = {
    "장애": IncidentType.OUTAGE,
    "서비스장애": IncidentType.OUTAGE,
    "침해": IncidentType.BREACH,
    "침해사고": IncidentType.BREACH,
    "해킹": IncidentType.BREACH,
    "유출": IncidentType.PRIVACY_LEAK,
    "개인정보유출": IncidentType.PRIVACY_LEAK,
}
_LOCATION_SYNONYMS = {
    "데이터센터": LocationKind.DATACENTER,
    "idc": LocationKind.DATACENTER,
    "리전": LocationKind.REGION,
    "가용영역": LocationKind.AZ,
    "availabilityzone": LocationKind.AZ,
}
_STATUS_SYNONYMS = {
    "재직": PersonStatus.ACTIVE,
    "퇴사": PersonStatus.DEPARTED,
    "퇴직": PersonStatus.DEPARTED,
    "휴직": PersonStatus.ON_LEAVE,
    "부재": PersonStatus.ON_LEAVE,
    "장기부재": PersonStatus.ON_LEAVE,
    "미확인": PersonStatus.UNKNOWN,
}


@dataclass(frozen=True)
class AssemblyResult:
    """The assembled graph plus the bookkeeping the report publishes.

    ``held_ids`` lists every HOLD node/edge (판단 보류); ``unresolved_people`` lists person
    names that the roster cannot confirm (명부 미확인) — both feed fixed rules and the UI.
    """

    graph: OpsGraph
    grounding: GroundingStatus
    held_ids: tuple[str, ...]
    unresolved_people: tuple[str, ...]


def assemble(
    extractions: Iterable[Extraction],
    docs: Iterable[Document],
    *,
    roster: Iterable[RosterEntry] = (),
    catalog: Iterable[CatalogService] = (),
    oncall: Iterable[OnCallShift] = (),
    policy: Policy = Policy(),  # noqa: B008 - Policy is an immutable value object
) -> AssemblyResult:
    """Ground, resolve and merge extractions with trusted facts into one :class:`OpsGraph`."""
    doc_map: dict[str, Document] = {}
    for doc in docs:
        doc_map.setdefault(doc.doc_id, doc)
    shifts = tuple(oncall)
    levels = sorted({*DEFAULT_ONCALL_LEVELS, *(s.level for s in shifts)})
    resolver = EntityResolver(
        roster, catalog, extra_roles=(*policy.terminal_roles, *(oncall_role(lv) for lv in levels))
    )
    tally = _Tally()
    mentions, statements = _ground(extractions, doc_map, tally)
    builder = _Assembler(resolver, _Namer(resolver, mentions, statements), tally)
    builder.import_roster()
    builder.import_oncall(shifts)
    builder.import_catalog()
    for mention in mentions:
        builder.add_mention(mention)
    for statement in statements:
        builder.add_statement(statement)
    for doc in doc_map.values():
        ref = DocumentRef(doc_id=doc.doc_id, title=doc.title, sha256=doc.sha256, path=doc.path)
        builder.graph.add_doc(ref)
    graph = builder.graph.build()
    held = sorted(
        [n.id for n in graph.nodes.values() if n.verification is Verification.HOLD]
        + [e.id for e in graph.edges.values() if e.verification is Verification.HOLD]
    )
    return AssemblyResult(
        graph=graph,
        grounding=tally.status(held_ambiguous=len(builder.ambiguous)),
        held_ids=tuple(held),
        unresolved_people=tuple(sorted(builder.unresolved)),
    )


def oncall_role(level: int) -> str:
    """Canonical ROLE name of an on-call rotation level (``1`` -> ``"1차 당직"``)."""
    return f"{level}차 당직"


def coerce_attr(key: str, value: AttrValue) -> AttrValue | None:
    """Coerce an extracted attribute value to the type its key documents, or ``None``.

    Rules compare ints and enum values; a string "10분" where an int is expected would make a
    deadline check silently pass or fail, so anything that cannot be read unambiguously is
    rejected rather than guessed.
    """
    coercer = _COERCERS.get(key)
    if coercer is not None:
        return coercer(value)
    if isinstance(value, str):
        return clean_surface(value) or None
    return value


def contradicts_quote(key: str, value: AttrValue, quote: str) -> bool:
    """True when ``quote`` states a different duration or channel medium than ``value``.

    Grounding proves the quoted phrase exists, not that it says *this* value: an extractor
    that emits 4320 minutes with the evidence "24시간" would otherwise plant an invented
    deadline behind a real span (P2). A quote that states no duration/medium at all cannot
    contradict the value and is accepted; one that states only a range ("5~10분") supports no
    single value.
    """
    if key in _MINUTE_KEYS and isinstance(value, int):
        durations = find_durations(quote)
        single = {d.seconds for d in durations if not is_range_bound(quote, d)}
        return bool(durations) and value * 60 not in single
    if key in (AttrKey.MEDIUM.value, AttrKey.CHANNEL.value) and isinstance(value, str):
        medium = medium_of(value)
        media = find_media(quote)
        return medium is not None and bool(media) and medium not in media
    return False


def _medium_value(value: AttrValue) -> str | None:
    medium = medium_of(value) if isinstance(value, str) else None
    return medium.value if medium is not None else None


def _text(value: AttrValue) -> str | None:
    """Free-text keys (title, condition, recipient, channel surface …) accept strings only: a
    number or list there is a malformed extraction, not a value to keep."""
    return (clean_surface(value) or None) if isinstance(value, str) else None


def _aliases(value: AttrValue) -> list[str] | None:
    items = [value] if isinstance(value, str) else value if isinstance(value, list) else []
    cleaned = [clean_surface(v) for v in items if clean_surface(v)]
    return cleaned or None


def _enum_coercer(
    members: Iterable[StrEnum], synonyms: Mapping[str, StrEnum]
) -> Callable[[AttrValue], str | None]:
    """Accept an enum value ("rollback") or a Korean synonym ("롤백"); anything else is unknown."""
    table = {**{compact(k): v.value for k, v in synonyms.items()}, **{m.value: m.value for m in members}}

    def coerce(value: AttrValue) -> str | None:
        return table.get(compact(value)) if isinstance(value, str) else None

    return coerce


_TEXT_KEYS = frozenset(
    {
        AttrKey.TITLE.value,
        AttrKey.DESCRIPTION.value,
        AttrKey.CONDITION.value,
        AttrKey.RECIPIENT.value,
        AttrKey.LEGAL_BASIS.value,
        AttrKey.CHANNEL.value,
    }
)

_COERCERS: dict[str, Callable[[AttrValue], AttrValue | None]] = {
    **dict.fromkeys(_MINUTE_KEYS, parse_minutes),
    **dict.fromkeys(_TEXT_KEYS, _text),
    AttrKey.TIER.value: parse_tier,
    AttrKey.REQUIRES_APPROVAL.value: parse_bool,
    AttrKey.MEDIUM.value: _medium_value,
    AttrKey.ALIASES.value: _aliases,
    AttrKey.ACTION_TYPE.value: _enum_coercer(ActionType, _ACTION_SYNONYMS),
    AttrKey.INCIDENT_TYPE.value: _enum_coercer(IncidentType, _INCIDENT_SYNONYMS),
    AttrKey.LOCATION_KIND.value: _enum_coercer(LocationKind, _LOCATION_SYNONYMS),
    AttrKey.STATUS.value: _enum_coercer(PersonStatus, _STATUS_SYNONYMS),
}


# --------------------------------------------------------------------------- bookkeeping


class _Drop(StrEnum):
    """Why an element was dropped; the value is the Korean label shown in examples."""

    UNGROUNDED = "원문 근거 없음"
    NO_DOCUMENT = "원문 문서 없음"
    CONTRADICTED = "근거와 다른 값"
    QUARANTINED = "격리 구간 인용"
    INVALID = "허용되지 않는 관계"
    NAMELESS = "이름 없는 개체"
    UNCOERCIBLE = "값 형식 오류"
    DUPLICATE = "중복 속성"
    UNCONFIRMED = "명부 미확인 인물의 재직 주장"


#: Which :class:`GroundingStatus` counter each drop reason feeds. A value its own quote
#: contradicts has no grounding, so it counts with the hallucination KPI.
_UNGROUNDED = frozenset({_Drop.UNGROUNDED, _Drop.NO_DOCUMENT, _Drop.CONTRADICTED})
_INVALID = frozenset({_Drop.INVALID, _Drop.NAMELESS, _Drop.UNCOERCIBLE, _Drop.DUPLICATE, _Drop.UNCONFIRMED})


@dataclass
class _Tally:
    """Counts every examined element exactly once: grounded (accepted) or dropped (why)."""

    grounded: int = 0
    fuzzy: int = 0
    dropped: Counter[_Drop] = field(default_factory=Counter)
    examples: list[str] = field(default_factory=list)

    def ground(
        self, doc: Document | None, quote: str, what: str, *, lo: int, hi: int | None
    ) -> tuple[SourceSpan, AlignMethod] | None:
        """Align ``quote``; on failure record the drop (absent vs quarantined vs no document)."""
        if doc is None:
            self.drop(_Drop.NO_DOCUMENT, what, quote)
            return None
        attempt = ground_quote(doc, quote, hint_start=lo, hint_end=hi)
        if attempt.alignment is None:
            self.drop(_Drop.QUARANTINED if attempt.quarantined else _Drop.UNGROUNDED, what, quote)
            return None
        return attempt.alignment.span, attempt.alignment.method

    def accept(self, method: AlignMethod) -> None:
        self.grounded += 1
        if method == "fuzzy":
            self.fuzzy += 1

    def drop(self, reason: _Drop, what: str, quote: str) -> None:
        self.dropped[reason] += 1
        if len(self.examples) < MAX_DROPPED_EXAMPLES:
            clipped = quote if len(quote) <= 60 else quote[:60] + "…"
            self.examples.append(f'{reason}: {what} "{clipped}"')

    def status(self, *, held_ambiguous: int) -> GroundingStatus:
        return GroundingStatus(
            grounded=self.grounded,
            fuzzy_grounded=self.fuzzy,
            dropped_ungrounded=sum(self.dropped[r] for r in _UNGROUNDED),
            dropped_quarantined=self.dropped[_Drop.QUARANTINED],
            dropped_invalid=sum(self.dropped[r] for r in _INVALID),
            held_ambiguous_entity=held_ambiguous,
            dropped_examples=tuple(self.examples),
        )


# --------------------------------------------------------------------------- grounding


@dataclass(frozen=True)
class _Mention:
    """A grounded entity mention."""

    kind: NodeKind
    name: str
    span: SourceSpan
    method: AlignMethod
    attrs: tuple[ExtractedAttr, ...]
    doc: Document
    extractor: Extractor


@dataclass(frozen=True)
class _Statement:
    """A grounded relation statement (validity is decided after resolution)."""

    kind: EdgeKind
    src: EntityRef
    dst: EntityRef
    span: SourceSpan
    method: AlignMethod
    attrs: tuple[ExtractedAttr, ...]
    doc: Document
    extractor: Extractor


def _ground(
    extractions: Iterable[Extraction], docs: dict[str, Document], tally: _Tally
) -> tuple[list[_Mention], list[_Statement]]:
    mentions: list[_Mention] = []
    statements: list[_Statement] = []
    for ex in extractions:
        doc = docs.get(ex.doc_id)
        for ent in ex.entities:
            what = f"개체({ent.kind.value} {ent.name})"
            hit = tally.ground(doc, ent.quote, what, lo=ex.chunk_start, hi=ex.chunk_end)
            if hit is not None and doc is not None:
                mentions.append(_Mention(ent.kind, ent.name, hit[0], hit[1], ent.attrs, doc, ex.extractor))
        for rel in ex.relations:
            what = f"관계({rel.kind.value} {rel.src.name}→{rel.dst.name})"
            hit = tally.ground(doc, rel.quote, what, lo=ex.chunk_start, hi=ex.chunk_end)
            if hit is not None and doc is not None:
                statements.append(
                    _Statement(rel.kind, rel.src, rel.dst, hit[0], hit[1], rel.attrs, doc, ex.extractor)
                )
    return mentions, statements


# --------------------------------------------------------------------------- naming


class _Namer:
    """Canonical names for things only the runbook knows (no roster/catalog entry).

    * Spelling variants ("결제서비스" / "결제 서비스") share one node: the most frequent
      grounded spelling wins, ties broken lexicographically (order-independent).
    * Service-specific children (롤백, 알람, 신고 의무) that several services share a name for
      are scoped by their service ("결제 서비스 롤백") — otherwise one "롤백" node would merge
      the approvers of unrelated services. Scoping happens only when that collision exists.
    """

    def __init__(
        self, resolver: EntityResolver, mentions: list[_Mention], statements: list[_Statement]
    ) -> None:
        self._resolver = resolver
        counts: dict[tuple[NodeKind, str], Counter[str]] = defaultdict(Counter)
        surfaces = [(m.kind, m.name) for m in mentions]
        surfaces += [(r.kind, r.name) for s in statements for r in (s.src, s.dst)]
        for kind, surface in surfaces:
            if kind in _RESOLVER_KINDS or (kind in _SERVICE_KINDS and resolver.resolve_service(surface)):
                continue  # people/roles and catalog services are named by the trusted sources
            counts[(kind, _key(kind, surface))][clean_surface(surface)] += 1
        self._canonical = {k: min(c.items(), key=lambda kv: (-kv[1], kv[0]))[0] for k, c in counts.items()}
        self._scope = self._scopes(statements)

    def _scopes(self, statements: list[_Statement]) -> dict[tuple[NodeKind, str, str], str]:
        parents: dict[tuple[NodeKind, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        for st in statements:
            child = _PARENT_EDGES.get(st.kind)
            if child is None or st.dst.kind is not child or st.src.kind not in _SERVICE_KINDS:
                continue
            service = self.service_name(st.src.kind, st.src.name)
            parents[(child, _key(child, st.dst.name))][st.doc.doc_id].add(service)
        scope: dict[tuple[NodeKind, str, str], str] = {}
        for (kind, key), by_doc in parents.items():
            if len(set[str]().union(*by_doc.values())) < 2:
                continue
            for doc_id, services in by_doc.items():
                if len(services) == 1:
                    scope[(kind, key, doc_id)] = next(iter(services))
        return scope

    def canonical(self, kind: NodeKind, surface: str) -> str:
        return self._canonical.get((kind, _key(kind, surface)), clean_surface(surface))

    def service_name(self, kind: NodeKind, surface: str) -> str:
        return self._resolver.resolve_service(surface) or self.canonical(kind, surface)

    def scoped(self, kind: NodeKind, surface: str, doc_id: str) -> str:
        base = self.canonical(kind, surface)
        service = self._scope.get((kind, _key(kind, surface), doc_id))
        if service is None or service_key(service) in compact(base):
            return base
        return f"{service} {base}"


def _key(kind: NodeKind, surface: str) -> str:
    return service_key(surface) if kind in _SERVICE_KINDS else compact(surface)


# --------------------------------------------------------------------------- building


@dataclass(frozen=True)
class _Target:
    """A resolved graph node for a surface, before provenance is attached."""

    node_id: str
    kind: NodeKind
    name: str
    attrs: dict[str, Attr] = field(default_factory=dict)
    hold_reason: str | None = None
    unresolved: bool = False


class _Assembler:
    """Stateful builder for one :func:`assemble` call (trusted imports, then runbook elements)."""

    def __init__(self, resolver: EntityResolver, namer: _Namer, tally: _Tally) -> None:
        self.resolver = resolver
        self.namer = namer
        self.tally = tally
        self.graph = GraphBuilder()
        self.unresolved: set[str] = set()
        self.ambiguous: set[str] = set()
        self.roster_ids = frozenset(resolver.person_node_id(e) for e in resolver.roster)

    # ------------------------------------------------------------------ trusted imports
    def import_roster(self) -> None:
        for entry in self.resolver.roster:
            prov = _import_prov(SourceKind.ROSTER, f"roster:{entry.name}")
            attrs = {AttrKey.STATUS.value: Attr(value=entry.status.value, source=SourceKind.ROSTER)}
            if entry.title:
                attrs[AttrKey.TITLE.value] = Attr(value=entry.title, source=SourceKind.ROSTER)
            if entry.aliases:
                attrs[AttrKey.ALIASES.value] = Attr(value=list(entry.aliases), source=SourceKind.ROSTER)
            pid = self.resolver.person_node_id(entry)
            self._node(_Target(pid, NodeKind.PERSON, entry.name, attrs), prov)
            roles = {self.resolver.resolve_role(r).name for r in entry.roles}
            if entry.title and self.resolver.is_role_word(entry.title):
                roles.add(self.resolver.resolve_role(entry.title).name)
            for role in sorted(roles):
                rid = self._node(_Target(node_id(NodeKind.ROLE, role), NodeKind.ROLE, role), prov)
                self._edge(EdgeKind.HOLDS_ROLE, pid, rid, {}, prov)
            if entry.team:
                team = self.resolver.resolve_team(entry.team) or clean_surface(entry.team)
                tid = self._node(_Target(node_id(NodeKind.TEAM, team), NodeKind.TEAM, team), prov)
                self._edge(EdgeKind.MEMBER_OF, pid, tid, {}, prov)
            for medium in sorted(set(entry.channels)):
                cid = self._node(_channel_target(medium, SourceKind.ROSTER, ()), prov)
                self._edge(EdgeKind.REACHABLE_VIA, pid, cid, {}, prov)

    def import_oncall(self, shifts: tuple[OnCallShift, ...]) -> None:
        levels: dict[str, set[int]] = defaultdict(set)
        for shift in shifts:
            if not names_nothing(shift.person):  # a blank or "미정" row names nobody
                levels[clean_surface(shift.person)].add(shift.level)
        for name in sorted(levels):
            prov = _import_prov(SourceKind.ONCALL, f"oncall:{name}")
            pid = self._node(self._person(name, SourceKind.ONCALL, (), allow_role=False), prov)
            for level in sorted(levels[name]):
                role = self.resolver.resolve_role(oncall_role(level)).name
                rid = self._node(_Target(node_id(NodeKind.ROLE, role), NodeKind.ROLE, role), prov)
                self._edge(EdgeKind.HOLDS_ROLE, pid, rid, {}, prov)

    def import_catalog(self) -> None:
        for svc in self.resolver.catalog:
            prov = _import_prov(SourceKind.CATALOG, f"catalog:{svc.name}")
            channel = svc.kind == CATALOG_CHANNEL_KIND
            sid = self._node(
                self._catalog_channel(svc) if channel else _catalog_target(svc, self.resolver), prov
            )
            for dep in svc.depends_on:
                did = self._node(self._dependency(dep), prov)
                if did != sid:
                    self._edge(EdgeKind.DEPENDS_ON, sid, did, {}, prov)
            if svc.location:
                loc = self.resolver.resolve_location(svc.location) or clean_surface(svc.location)
                lid = self._node(_Target(node_id(NodeKind.LOCATION, loc), NodeKind.LOCATION, loc), prov)
                self._edge(EdgeKind.RUNS_IN, sid, lid, {}, prov)
            owner = self._catalog_owner(svc.owner) if svc.owner and not channel else None
            if owner is not None:
                self._edge(EdgeKind.OWNED_BY, sid, self._node(owner, prov), {}, prov)

    @staticmethod
    def _catalog_channel(svc: CatalogService) -> _Target:
        """A catalog contact medium ("사내 메신저", location 상암 IDC) joins the same CHANNEL
        node as roster contacts and runbook mentions, so its failure domain reaches every
        response path that relies on it."""
        medium = next((m for m in map(medium_of, (svc.name, *svc.aliases)) if m is not None), None)
        if medium is not None:
            return _channel_target(medium, SourceKind.CATALOG, ())
        name = clean_surface(svc.name)
        return _Target(node_id(NodeKind.CHANNEL, name), NodeKind.CHANNEL, name)

    def _dependency(self, surface: str) -> _Target:
        """Catalog dependency target. Undeclared ones become SYSTEM nodes: dependencies are
        usually infrastructure, and a phantom SERVICE would trigger "no owner" findings for
        something the organisation never declared as a service."""
        name = self.resolver.resolve_service(surface)
        if name is not None:
            kind = self.resolver.service_kind(name)
            return _Target(node_id(kind, name), kind, name)
        name = clean_surface(surface)
        return _Target(node_id(NodeKind.SYSTEM, name), NodeKind.SYSTEM, name)

    def _catalog_owner(self, owner: str) -> _Target | None:
        """Only owners the roster can confirm (person, team or role) become OWNED_BY edges."""
        res = self.resolver.resolve_person(owner)
        if res.status is ResolutionStatus.MATCHED and res.roster_entry is not None:
            entry = res.roster_entry
            return _Target(self.resolver.person_node_id(entry), NodeKind.PERSON, entry.name)
        team = self.resolver.resolve_team(owner)
        if team is not None:
            return _Target(node_id(NodeKind.TEAM, team), NodeKind.TEAM, team)
        role = self.resolver.resolve_role(owner)
        if role.known:
            return _Target(node_id(NodeKind.ROLE, role.name), NodeKind.ROLE, role.name)
        return None

    # ------------------------------------------------------------------ runbook elements
    def add_mention(self, m: _Mention) -> None:
        what = f"개체({m.kind.value} {m.name})"
        target = self._resolve(m.kind, m.name, m.doc.doc_id, m.span)
        if target is None:
            self.tally.drop(_Drop.NAMELESS, what, m.span.quote)
            return
        self.tally.accept(m.method)
        # A channel's medium is its identity (channel:slack *is* Slack): a mention attr that
        # says otherwise contradicts the very name it is attached to.
        fixed = {k: a.value for k, a in target.attrs.items() if k == AttrKey.MEDIUM.value}
        attrs = self._attrs(
            m.attrs,
            m.doc,
            m.span,
            m.extractor,
            what,
            unconfirmed_person=_unconfirmed_person(target),
            fixed=fixed,
        )
        merged = _Target(
            target.node_id,
            target.kind,
            target.name,
            {**target.attrs, **attrs},
            target.hold_reason,
            target.unresolved,
        )
        self._node(merged, _runbook_prov(m.extractor, m.span))

    def add_statement(self, s: _Statement) -> None:
        src_span = _endpoint_span(s.doc, s.src.name, s.span)
        dst_span = _endpoint_span(s.doc, s.dst.name, s.span)
        src = self._resolve(s.src.kind, s.src.name, s.doc.doc_id, src_span)
        dst = self._resolve(s.dst.kind, s.dst.name, s.doc.doc_id, dst_span)
        what = f"관계({s.kind.value} {s.src.name}→{s.dst.name})"
        if src is None or dst is None:
            self.tally.drop(_Drop.NAMELESS, what, s.span.quote)
            return
        if not _signature_ok(s.kind, src.kind, dst.kind) or src.node_id == dst.node_id:
            self.tally.drop(_Drop.INVALID, what, s.span.quote)
            return
        self._node(src, _runbook_prov(s.extractor, src_span))
        self._node(dst, _runbook_prov(s.extractor, dst_span))
        self.tally.accept(s.method)
        attrs = self._attrs(s.attrs, s.doc, s.span, s.extractor, what)
        self._edge(s.kind, src.node_id, dst.node_id, attrs, _runbook_prov(s.extractor, s.span))

    def _attrs(
        self,
        raw: tuple[ExtractedAttr, ...],
        doc: Document,
        parent: SourceSpan,
        extractor: Extractor,
        owner: str,
        *,
        unconfirmed_person: bool = False,
        fixed: Mapping[str, AttrValue] | None = None,
    ) -> dict[str, Attr]:
        """Ground (near the parent span first), coerce and cross-check each attribute.

        Drops what is ungrounded, malformed, contradicted by its own quote or by a value the
        resolved identity ``fixed`` (a channel's medium), duplicated, or — for a person the
        roster does not confirm — a runbook claim that they are active: only the roster may
        vouch for availability (P3 확인 불가 ≠ 정상).
        """
        out: dict[str, Attr] = {}
        for attr in raw:
            what = f"{owner}의 속성({attr.key})"
            hit = self.tally.ground(doc, attr.quote, what, lo=parent.start, hi=parent.end)
            if hit is None:
                continue
            span, method = hit
            value = coerce_attr(attr.key, attr.value)
            if value is None:
                self.tally.drop(_Drop.UNCOERCIBLE, what, f"{attr.value}")
                continue
            reason = _attr_problem(
                attr.key, value, span.quote, out, unconfirmed_person=unconfirmed_person, fixed=fixed or {}
            )
            if reason is not None:
                self.tally.drop(reason, what, span.quote)
                continue
            if attr.key == AttrKey.CHANNEL and isinstance(value, str):
                channel = self._resolve(NodeKind.CHANNEL, value, doc.doc_id, span)
                if channel is None:
                    self.tally.drop(_Drop.NAMELESS, what, span.quote)
                    continue
                value = self._node(channel, _runbook_prov(extractor, span))
            self.tally.accept(method)
            out[attr.key] = Attr(value=value, spans=(span,), source=SourceKind.RUNBOOK)
        return out

    # ------------------------------------------------------------------ resolution
    def _resolve(self, kind: NodeKind, surface: str, doc_id: str, span: SourceSpan) -> _Target | None:
        """Map a runbook surface to its graph node (roster/catalog identity when one exists).

        ``None`` when the surface names nothing ("", "(미정)", "TBD", or a person surface that is
        only a parenthetical): such an endpoint would become a placeholder node
        (``person:unnamed``, ``channel:미정``) that rules would mistake for a real responder.
        """
        if names_nothing(surface):
            return None
        target = self._target(kind, surface, doc_id, span)
        return target if target.name.strip() else None

    def _target(self, kind: NodeKind, surface: str, doc_id: str, span: SourceSpan) -> _Target:
        if kind is NodeKind.PERSON:
            return self._person(surface, SourceKind.RUNBOOK, (span,), allow_role=True)
        if kind is NodeKind.ROLE:
            return self._role(surface)
        medium = medium_of(surface) if kind is NodeKind.CHANNEL else None
        if medium is not None:
            return _channel_target(medium, SourceKind.RUNBOOK, (span,))
        service = self.resolver.resolve_service(surface) if kind in _SERVICE_KINDS else None
        if service is not None:
            kind = self.resolver.service_kind(service)
        name = service or self._local_name(kind, surface, doc_id)
        return _Target(node_id(kind, name), kind, name)

    def _local_name(self, kind: NodeKind, surface: str, doc_id: str) -> str:
        """Name for a surface no roster person/catalog service matches: the roster team or
        catalog location spelling when available, else the runbook's canonical spelling."""
        if kind is NodeKind.TEAM:
            return self.resolver.resolve_team(surface) or self.namer.canonical(kind, surface)
        if kind is NodeKind.LOCATION:
            return self.resolver.resolve_location(surface) or self.namer.canonical(kind, surface)
        if kind in _PARENT_EDGES.values():
            return self.namer.scoped(kind, surface, doc_id)
        return self.namer.canonical(kind, surface)

    def _role(self, surface: str) -> _Target:
        """Role surface. A non-role word that names one roster person ("박팀장") is that person."""
        role = self.resolver.resolve_role(surface)
        if not role.known:
            res = self.resolver.resolve_person(surface)
            if res.status is ResolutionStatus.MATCHED and res.roster_entry is not None:
                entry = res.roster_entry
                return _Target(self.resolver.person_node_id(entry), NodeKind.PERSON, entry.name)
        return _Target(node_id(NodeKind.ROLE, role.name), NodeKind.ROLE, role.name)

    def _person(
        self, surface: str, source: SourceKind, spans: tuple[SourceSpan, ...], *, allow_role: bool
    ) -> _Target:
        res = self.resolver.resolve_person(surface)
        if res.status is ResolutionStatus.MATCHED and res.roster_entry is not None:
            entry = res.roster_entry
            return _Target(self.resolver.person_node_id(entry), NodeKind.PERSON, entry.name)
        if res.status is ResolutionStatus.UNKNOWN and allow_role and self.resolver.is_role_word(surface):
            # "팀장에게 보고" names a position; an unknown *person* "팀장" would be a phantom.
            return self._role(surface)
        name = display_name(surface)
        pid = self._unconfirmed_id(name)
        attrs = {AttrKey.STATUS.value: Attr(value=PersonStatus.UNKNOWN.value, spans=spans, source=source)}
        if res.status is ResolutionStatus.AMBIGUOUS:
            attrs["candidates"] = Attr(value=[e.name for e in res.candidates], source=SourceKind.ROSTER)
            described = ", ".join(_describe(e) for e in res.candidates)
            return _Target(
                pid, NodeKind.PERSON, name, attrs, hold_reason=f"동명이인/모호한 지칭: 후보 {described}"
            )
        return _Target(pid, NodeKind.PERSON, name, attrs, unresolved=True)

    def _unconfirmed_id(self, name: str) -> str:
        """Node id for a person the roster does not (uniquely) confirm.

        It must never coincide with a roster person's id — "이준호 2" slugs exactly like the
        namesake id ``person:이준호-2`` — or the unconfirmed mention would merge into that roster
        identity and inherit its status, roles and channels without any evidence (P2, P3).
        """
        pid = node_id(NodeKind.PERSON, name)
        return node_id(NodeKind.PERSON, f"{name} 미확인") if pid in self.roster_ids else pid

    # ------------------------------------------------------------------ graph writes
    def _node(self, target: _Target, prov: Provenance) -> str:
        hold = target.hold_reason is not None
        self.graph.add_node(
            Node(
                id=target.node_id,
                kind=target.kind,
                name=target.name,
                attrs=target.attrs,
                prov=(prov,),
                verification=Verification.HOLD if hold else Verification.VERIFIED,
                hold_reason=target.hold_reason,
            )
        )
        if hold:
            self.ambiguous.add(target.node_id)
        if target.unresolved:
            self.unresolved.add(target.name)
        return target.node_id

    def _edge(self, kind: EdgeKind, src: str, dst: str, attrs: dict[str, Attr], prov: Provenance) -> None:
        eid = edge_id(kind, src, dst)
        current = self.graph.edges.get(eid)
        if current is not None and _conflicts(current.attrs, attrs):
            # Two sources state different values for the same relation (e.g. 10분 vs 15분 상향):
            # keep both as distinct edges so the contradiction stays visible to the rules.
            disc = "+".join(slug(f"{k}-{attrs[k].value}") for k in sorted(attrs))
            eid = edge_id(kind, src, dst, disc)
        self.graph.add_edge(Edge(id=eid, kind=kind, src=src, dst=dst, attrs=attrs, prov=(prov,)))


def _catalog_target(svc: CatalogService, resolver: EntityResolver) -> _Target:
    kind = resolver.service_kind(svc.name)
    attrs: dict[str, Attr] = {}
    if svc.tier is not None:
        attrs[AttrKey.TIER.value] = Attr(value=svc.tier, source=SourceKind.CATALOG)
    if svc.description:
        attrs[AttrKey.DESCRIPTION.value] = Attr(value=svc.description, source=SourceKind.CATALOG)
    if svc.aliases:
        attrs[AttrKey.ALIASES.value] = Attr(value=list(svc.aliases), source=SourceKind.CATALOG)
    return _Target(node_id(kind, svc.name), kind, svc.name, attrs)


def _attr_problem(
    key: str,
    value: AttrValue,
    quote: str,
    seen: Mapping[str, Attr],
    *,
    unconfirmed_person: bool,
    fixed: Mapping[str, AttrValue],
) -> _Drop | None:
    """Why a grounded, well-typed attribute must still be dropped (``None`` = keep it)."""
    if contradicts_quote(key, value, quote) or (key in fixed and fixed[key] != value):
        return _Drop.CONTRADICTED
    if unconfirmed_person and key == AttrKey.STATUS.value and value == PersonStatus.ACTIVE.value:
        return _Drop.UNCONFIRMED
    if key in seen:
        return _Drop.DUPLICATE
    return None


def _unconfirmed_person(target: _Target) -> bool:
    """A person the roster does not (uniquely) confirm: unknown or ambiguous."""
    return target.kind is NodeKind.PERSON and (target.unresolved or target.hold_reason is not None)


def _channel_target(medium: ChannelMedium, source: SourceKind, spans: tuple[SourceSpan, ...]) -> _Target:
    """One CHANNEL node per medium (``channel:slack``) so roster contacts and runbook mentions
    of the same medium connect, and a medium outage reaches every path that uses it.

    The id stays the stable medium code; the *name* is the Korean label ("슬랙", "사내 메신저"),
    because every finding, rehearsal note and report shows node names to Korean readers
    ("'슬랙' 하나뿐", "전화로 호출"), never internal codes."""
    attrs = {AttrKey.MEDIUM.value: Attr(value=medium.value, spans=spans, source=source)}
    return _Target(node_id(NodeKind.CHANNEL, medium.value), NodeKind.CHANNEL, MEDIUM_LABEL_KO[medium], attrs)


def _import_prov(source: SourceKind, ref: str) -> Provenance:
    return Provenance(source=source, extractor=Extractor.IMPORT, ref=ref)


def _runbook_prov(extractor: Extractor, span: SourceSpan) -> Provenance:
    return Provenance(source=SourceKind.RUNBOOK, extractor=extractor, spans=(span,))


def _endpoint_span(doc: Document, surface: str, relation: SourceSpan) -> SourceSpan:
    """Evidence for a relation endpoint: its own mention inside the relation quote when it can
    be found there, otherwise the relation quote itself (which states the endpoint)."""
    hit = align_quote(doc, surface, hint_start=relation.start, hint_end=relation.end)
    if hit is not None and relation.start <= hit.span.start and hit.span.end <= relation.end:
        return hit.span
    return relation


def _signature_ok(kind: EdgeKind, src: NodeKind, dst: NodeKind) -> bool:
    signature = EDGE_SIGNATURES.get(kind)
    return signature is not None and src in signature[0] and dst in signature[1]


def _conflicts(old: dict[str, Attr], new: dict[str, Attr]) -> bool:
    return any(k in old and old[k].value != v.value for k, v in new.items())


def _describe(entry: RosterEntry) -> str:
    details = " ".join(x for x in (entry.team, entry.title) if x)
    return f"{entry.name}({details})" if details else entry.name
