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
   ``status=unknown`` nodes so fixed rules report them instead of trusting them. In documents the
   LLM read, service-like names are first settled structurally (:func:`_settle_surfaces`): a
   title that describes a state of a named component is that component, a sentence heading is
   no service, and a name given as both a service and a system is one service.
5. **Provenance** — runbook elements carry ``Provenance(RUNBOOK, extractor, spans)``.

No LLM is used here (P1/P4), and the output is a pure function of the inputs (P6).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TypeVar

from jettae.domain import (
    ANY_SERVICE_ALERT,
    EDGE_SIGNATURES,
    UNNAMED_SERVICE_ALERT,
    ActionType,
    Attr,
    AttrKey,
    CatalogService,
    ChannelMedium,
    DelegatedElement,
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
    SubjectKind,
    Verification,
    delegation_key,
    edge_id,
    node_id,
    slug,
)
from jettae.domain.graph import AttrValue
from jettae.extract.align import AlignMethod, align_quote, ground_quote
from jettae.extract.markdown import fenced_ranges, heading_texts, in_ranges
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
from jettae.extract.surfaces import deployment_base, is_sentence, state_base

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
#: How a note names a delegated element (the checks have their own labels; the layers stay apart).
_DELEGATED_KO: dict[DelegatedElement, str] = {
    DelegatedElement.DETECTION: "감지·호출 경로",
    DelegatedElement.OWNER: "담당자",
    DelegatedElement.ESCALATION: "상향 경로",
    DelegatedElement.APPROVER: "승인자",
    DelegatedElement.FALLBACK_CHANNEL: "연락 수단",
    DelegatedElement.ROLLBACK: "롤백·완화 절차",
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
    ``notes`` are Korean remarks for the report, e.g. an alert runbook that names no service, or
    an LLM-read document whose graph ends with no service at all (nothing to check is not the same
    as nothing missing, so such a run is never silently empty).
    """

    graph: OpsGraph
    grounding: GroundingStatus
    held_ids: tuple[str, ...]
    unresolved_people: tuple[str, ...]
    notes: tuple[str, ...] = ()


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
    extractions = tuple(extractions)
    #: documents the LLM extractor read: the only ones the structural readings below apply to
    llm_docs = frozenset(ex.doc_id for ex in extractions if ex.extractor is Extractor.LLM)
    tally = _Tally()
    grounded = _ground(extractions, doc_map, tally)
    settled = _settle_surfaces(
        mentions=[*grounded.mentions, *(m for s in grounded.subjects for m in _subject_mentions(s))],
        statements=[*grounded.statements, *(st for s in grounded.subjects for st in _subject_statements(s))],
        delegations=grounded.delegations,
        shared=grounded.shared,
        llm_docs=llm_docs,
        resolver=resolver,
        tally=tally,
    )
    mentions, statements = settled.mentions, settled.statements
    builder = _Assembler(resolver, _Namer(resolver, mentions, statements), tally)
    builder.import_roster()
    builder.import_oncall(shifts)
    builder.import_catalog()
    for mention in mentions:
        builder.add_mention(mention)
    for statement in statements:
        builder.add_statement(statement)
    # Document-level statements last: "every service this document covers" is known only now.
    for shared in settled.shared:
        builder.add_shared(shared)
    for subject in grounded.subjects:
        builder.add_subject(subject)
    for delegation in settled.delegations:
        builder.add_delegation(delegation)
    for doc in doc_map.values():
        ref = DocumentRef(doc_id=doc.doc_id, title=doc.title, sha256=doc.sha256, path=doc.path)
        builder.graph.add_doc(ref)
    graph = builder.graph.build()
    if llm_docs and not graph.nodes_of(NodeKind.SERVICE):
        # Never silently empty: say why an LLM-read document left nothing to check. (A rule-only run
        # keeps its sealed output; its summary already says that no service was scored.)
        for doc in doc_map.values():
            if doc.doc_id in llm_docs and doc.doc_id not in builder.explained:
                builder.notes.append(_no_service_note(doc, graph, settled.titles.get(doc.doc_id, ())))
    held = sorted(
        [n.id for n in graph.nodes.values() if n.verification is Verification.HOLD]
        + [e.id for e in graph.edges.values() if e.verification is Verification.HOLD]
    )
    return AssemblyResult(
        graph=graph,
        grounding=tally.status(held_ambiguous=len(builder.ambiguous)),
        held_ids=tuple(held),
        unresolved_people=tuple(sorted(builder.unresolved)),
        notes=tuple(dict.fromkeys(builder.notes)),
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
    CODE_ONLY = "명령·출력 안에만 나오는 이름"
    SENTENCE_TITLE = "문장으로 된 제목(서비스 이름 아님)"


#: Which :class:`GroundingStatus` counter each drop reason feeds. A value its own quote
#: contradicts has no grounding, so it counts with the hallucination KPI.
_UNGROUNDED = frozenset({_Drop.UNGROUNDED, _Drop.NO_DOCUMENT, _Drop.CONTRADICTED})
_INVALID = frozenset(
    {
        _Drop.INVALID,
        _Drop.NAMELESS,
        _Drop.UNCOERCIBLE,
        _Drop.DUPLICATE,
        _Drop.UNCONFIRMED,
        _Drop.CODE_ONLY,
        _Drop.SENTENCE_TITLE,
    }
)


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


@dataclass(frozen=True)
class _Subject:
    """A grounded document subject ("this is the runbook of alert X")."""

    kind: SubjectKind
    name: str
    services: tuple[str, ...]
    any_service: bool
    span: SourceSpan
    method: AlignMethod
    doc: Document
    extractor: Extractor


@dataclass(frozen=True)
class _Delegation:
    """A grounded hand-off of one element to an external source."""

    element: DelegatedElement
    target: str
    applies_to: str | None
    span: SourceSpan
    method: AlignMethod
    doc: Document
    extractor: Extractor


@dataclass(frozen=True)
class _Shared:
    """A grounded relation stated for everything the document covers (no service named)."""

    kind: EdgeKind
    dst: EntityRef
    span: SourceSpan
    method: AlignMethod
    doc: Document
    extractor: Extractor


@dataclass
class _Grounded:
    mentions: list[_Mention] = field(default_factory=list)
    statements: list[_Statement] = field(default_factory=list)
    subjects: list[_Subject] = field(default_factory=list)
    delegations: list[_Delegation] = field(default_factory=list)
    #: group statements, one per named member (moved into ``statements`` once their kinds are known)
    group: list[_Statement] = field(default_factory=list)
    #: document-wide group statements (no member named; linked once the document's services exist)
    shared: list[_Shared] = field(default_factory=list)


def _ground(extractions: Iterable[Extraction], docs: dict[str, Document], tally: _Tally) -> _Grounded:
    out = _Grounded()
    for ex in extractions:
        doc = docs.get(ex.doc_id)
        for ent in ex.entities:
            what = f"개체({ent.kind.value} {ent.name})"
            hit = tally.ground(doc, ent.quote, what, lo=ex.chunk_start, hi=ex.chunk_end)
            if hit is not None and doc is not None:
                out.mentions.append(
                    _Mention(ent.kind, ent.name, hit[0], hit[1], ent.attrs, doc, ex.extractor)
                )
        for rel in ex.relations:
            what = f"관계({rel.kind.value} {rel.src.name}→{rel.dst.name})"
            hit = tally.ground(doc, rel.quote, what, lo=ex.chunk_start, hi=ex.chunk_end)
            if hit is not None and doc is not None:
                out.statements.append(
                    _Statement(rel.kind, rel.src, rel.dst, hit[0], hit[1], rel.attrs, doc, ex.extractor)
                )
        for sub in ex.subjects:
            what = f"문서 대상({sub.kind.value} {sub.name})"
            hit = tally.ground(doc, sub.quote, what, lo=ex.chunk_start, hi=ex.chunk_end)
            if hit is not None and doc is not None:
                out.subjects.append(
                    _Subject(
                        sub.kind, sub.name, sub.services, sub.any_service, hit[0], hit[1], doc, ex.extractor
                    )
                )
        for dlg in ex.delegations:
            what = f"외부 위임({dlg.element.value} → {dlg.target})"
            hit = tally.ground(doc, dlg.quote, what, lo=ex.chunk_start, hi=ex.chunk_end)
            if hit is not None and doc is not None:
                out.delegations.append(
                    _Delegation(dlg.element, dlg.target, dlg.applies_to, hit[0], hit[1], doc, ex.extractor)
                )
        for shared in ex.shared:
            group = ", ".join(shared.services) or "문서 전체"
            what = f"묶음 관계({shared.kind.value} {group}→{shared.dst.name})"
            hit = tally.ground(doc, shared.quote, what, lo=ex.chunk_start, hi=ex.chunk_end)
            if hit is None or doc is None:
                continue
            # A group statement states the relation for each service it names: one statement per
            # service, through the same funnel (naming, signature, endpoint evidence) as any relation.
            out.group += [
                _Statement(
                    shared.kind,
                    EntityRef(kind=NodeKind.SERVICE, name=service),
                    shared.dst,
                    hit[0],
                    hit[1],
                    (),
                    doc,
                    ex.extractor,
                )
                for service in shared.services
            ]
            if not shared.services:
                out.shared.append(_Shared(shared.kind, shared.dst, hit[0], hit[1], doc, ex.extractor))
    out.statements += _settle_group_kinds(out)
    return out


def _settle_group_kinds(grounded: _Grounded) -> list[_Statement]:
    """Group statements name their members as written, without a kind: a member the same document
    otherwise mentions only as a system (and never as a service) is that system, not a second,
    service-kind node of the same name."""
    kinds: dict[tuple[str, str], set[NodeKind]] = defaultdict(set)
    refs = [(m.doc.doc_id, m.kind, m.name) for m in grounded.mentions]
    refs += [(s.doc.doc_id, r.kind, r.name) for s in grounded.statements for r in (s.src, s.dst)]
    for doc_id, kind, name in refs:
        if kind in _SERVICE_KINDS:
            kinds[(doc_id, service_key(name))].add(kind)

    def settle(st: _Statement) -> _Statement:
        if kinds.get((st.doc.doc_id, service_key(st.src.name))) == {NodeKind.SYSTEM}:
            return replace(st, src=EntityRef(kind=NodeKind.SYSTEM, name=st.src.name))
        return st

    return [settle(st) for st in grounded.group]


def _subject_mentions(subject: _Subject) -> list[_Mention]:
    """What a subject names: the alert (a trigger) or the service the document is the runbook of.

    Grounded on the subject's own quote, so the node exists even when the extractor stated the
    subject but not a separate entity for it.
    """
    kind = {SubjectKind.ALERT: NodeKind.TRIGGER, SubjectKind.SERVICE: NodeKind.SERVICE}.get(subject.kind)
    names = [(kind, subject.name)] if kind is not None else []
    if subject.kind is not SubjectKind.ALERT:  # an alert's services arrive with its DETECTED_BY edges
        names += [(NodeKind.SERVICE, s) for s in subject.services]
    return [
        _Mention(
            k,
            n,
            _endpoint_span(subject.doc, n, subject.span, nearby=True),
            subject.method,
            (),
            subject.doc,
            subject.extractor,
        )
        for k, n in names
    ]


def _subject_statements(subject: _Subject) -> list[_Statement]:
    """An alert runbook that names the services it watches states their detection signal."""
    if subject.kind is not SubjectKind.ALERT:
        return []
    alert = EntityRef(kind=NodeKind.TRIGGER, name=subject.name)
    return [
        _Statement(
            EdgeKind.DETECTED_BY,
            EntityRef(kind=NodeKind.SERVICE, name=service),
            alert,
            subject.span,
            subject.method,
            (),
            subject.doc,
            subject.extractor,
        )
        for service in subject.services
    ]


# --------------------------------------------------------------------------- structural names


@dataclass(frozen=True)
class _Settled:
    """Grounded statements after :func:`_settle_surfaces`."""

    mentions: list[_Mention]
    statements: list[_Statement]
    delegations: list[_Delegation]
    shared: list[_Shared]
    #: per document: sentence headings not read as services (the no-service note names them)
    titles: dict[str, tuple[str, ...]]


#: Emphasis and code marks that do not change what a heading says.
_MARKS = re.compile(r"[`*_]")


def _plain_key(text: str) -> str:
    return compact(_MARKS.sub("", text))


class _SurfaceReading:
    """The structural reading of one batch of service/system references, document by document.

    Built once from every service-like reference of the LLM-read documents, then asked for the
    reading of each reference (:meth:`read`): the same name, the named component a state phrase is
    about, the service a service/system twin merges into (same name, or the service's name plus a
    deployment noun: "<service> clusters"), or ``None`` for a sentence heading.
    """

    def __init__(
        self,
        refs: Mapping[str, list[tuple[NodeKind, str]]],
        docs: Mapping[str, Document],
        resolver: EntityResolver,
    ) -> None:
        self._resolver = resolver
        self._states: dict[tuple[str, str], tuple[str, NodeKind | None]] = {}
        self._dropped: set[tuple[str, str]] = set()
        self._twins: set[tuple[str, str]] = set()
        #: (document, system key) -> the service whose deployment the system names
        self._deployments: dict[tuple[str, str], str] = {}
        self.titles: dict[str, tuple[str, ...]] = {}
        for doc_id, items in refs.items():
            self._read_document(doc_id, items, docs[doc_id])

    def _catalog(self, surface: str) -> bool:
        return self._resolver.resolve_service(surface) is not None

    def _read_document(self, doc_id: str, items: list[tuple[NodeKind, str]], doc: Document) -> None:
        own = [(kind, clean_surface(name)) for kind, name in items if not self._catalog(name)]
        kinds: dict[str, set[NodeKind]] = defaultdict(set)
        for kind, name in own:
            kinds[service_key(name)].add(kind)
        known = set(kinds) | {service_key(name) for _kind, name in items}
        headings = [_plain_key(h) for h in (*heading_texts(doc.text), doc.title) if h.strip()]
        titles: list[str] = []
        for name in sorted({name for _kind, name in own}):
            key = service_key(name)

            def is_known(base: str, key: str = key) -> bool:
                base_key = service_key(base)
                return base_key != key and (base_key in known or self._catalog(base))

            base = state_base(name, is_known)
            if base is not None:
                base_kinds = kinds.get(service_key(base), set())
                as_kind = next((k for k in (NodeKind.SERVICE, NodeKind.SYSTEM) if k in base_kinds), None)
                self._states[(doc_id, key)] = (base, as_kind)
            elif is_sentence(name) and any(_plain_key(name) in heading for heading in headings):
                self._dropped.add((doc_id, key))
                titles.append(name)
        if titles:
            self.titles[doc_id] = tuple(titles)
        merged: dict[str, set[NodeKind]] = defaultdict(set)
        for kind, name in own:
            read = self._state(doc_id, kind, name)
            if read is not None and not self._catalog(read[1]):
                merged[service_key(read[1])].add(read[0])
        both = {NodeKind.SERVICE, NodeKind.SYSTEM}
        self._twins |= {(doc_id, k) for k, found in merged.items() if both <= found}
        services = {k for k, found in merged.items() if NodeKind.SERVICE in found}
        for kind, name in own:
            read = self._state(doc_id, kind, name)
            key = service_key(read[1]) if read is not None else ""
            if read is None or merged.get(key) != {NodeKind.SYSTEM}:
                continue
            base = deployment_base(read[1], lambda b: service_key(b) in services)
            if base is not None:
                self._deployments[(doc_id, key)] = base

    def _state(self, doc_id: str, kind: NodeKind, name: str) -> tuple[NodeKind, str] | None:
        key = service_key(name)
        if (doc_id, key) in self._dropped:
            return None
        state = self._states.get((doc_id, key))
        if state is None:
            return kind, name
        base, base_kind = state
        return base_kind or kind, base

    def read(self, doc_id: str, kind: NodeKind, name: str) -> tuple[NodeKind, str] | None:
        """How the reference ``(kind, name)`` of ``doc_id`` reaches the graph (``None`` = no node)."""
        if kind not in _SERVICE_KINDS or self._catalog(name):
            return kind, name
        read = self._state(doc_id, kind, name)
        if read is None:
            return None
        kind, name = read
        if kind is not NodeKind.SYSTEM:
            return kind, name
        key = (doc_id, service_key(name))
        if key in self._twins:
            return NodeKind.SERVICE, name
        base = self._deployments.get(key)
        return (NodeKind.SERVICE, base) if base is not None else (kind, name)


def _settle_surfaces(
    *,
    mentions: list[_Mention],
    statements: list[_Statement],
    delegations: list[_Delegation],
    shared: list[_Shared],
    llm_docs: frozenset[str],
    resolver: EntityResolver,
    tally: _Tally,
) -> _Settled:
    """Structural readings of service/system names in the documents the LLM read.

    1. **A state of a named component** — a name that is a component the document (or the catalog)
       already names plus a state phrase ("<X> is down", "<X> outage", "Outage of <X>") is that
       component (:func:`~jettae.extract.surfaces.state_base`). It takes the kind the document
       gives the component, so a title never turns a system into a service.
    2. **A sentence heading** — a heading or title that is a sentence (it conjugates a verb) and is
       no such state is not a service: it is dropped, with the statements that need it, and named
       in the no-service note. A delegation stated for it applies to the whole document.
    3. **Service/system twins** — a name the same document gives both as a service and as a
       system is one service (service wins): a mitigation or signal stated for the system twin is
       then a step of the service the findings are about. A system named as a service's
       deployment ("<service> clusters", "<service> nodes":
       :func:`~jettae.extract.surfaces.deployment_base`) is that service too. Every span stays on
       the merged node.

    Catalog names keep the catalog's kind. Documents only the rule parser read are untouched (its
    output is sealed); the readings are per document, as node kinds are a reading of each text.
    """
    refs: dict[str, list[tuple[NodeKind, str]]] = defaultdict(list)
    docs: dict[str, Document] = {}
    entries: list[tuple[Document, tuple[EntityRef, ...]]] = [
        (m.doc, (EntityRef(kind=m.kind, name=m.name),)) for m in mentions
    ]
    entries += [(st.doc, (st.src, st.dst)) for st in statements]
    entries += [(sh.doc, (sh.dst,)) for sh in shared]
    entries += [(d.doc, (_holder(d),)) for d in delegations if d.applies_to is not None]
    for doc, items in entries:
        if doc.doc_id in llm_docs:
            docs[doc.doc_id] = doc
            refs[doc.doc_id] += [(e.kind, e.name) for e in items if e.kind in _SERVICE_KINDS]
    if not refs:
        return _Settled(list(mentions), list(statements), list(delegations), list(shared), {})
    reading = _SurfaceReading(refs, docs, resolver)

    def ref(doc: Document, entity: EntityRef) -> EntityRef | None:
        if doc.doc_id not in llm_docs:
            return entity
        read = reading.read(doc.doc_id, entity.kind, entity.name)
        if read is None or read == (entity.kind, entity.name):
            return None if read is None else entity
        return EntityRef(kind=read[0], name=read[1])

    def mention(m: _Mention) -> _Mention | None:
        entity = ref(m.doc, EntityRef(kind=m.kind, name=m.name))
        return None if entity is None else replace(m, kind=entity.kind, name=entity.name)

    def statement(st: _Statement) -> _Statement | None:
        src, dst = ref(st.doc, st.src), ref(st.doc, st.dst)
        return None if src is None or dst is None else replace(st, src=src, dst=dst)

    def group(sh: _Shared) -> _Shared | None:
        dst = ref(sh.doc, sh.dst)
        return None if dst is None else replace(sh, dst=dst)

    out_mentions = _kept(mentions, mention, tally)
    out_statements = _kept(statements, statement, tally)
    out_shared = _kept(shared, group, tally)
    out_delegations = [
        d if d.applies_to is None else replace(d, applies_to=_name_of(ref(d.doc, _holder(d))))
        for d in delegations
    ]
    return _Settled(out_mentions, out_statements, out_delegations, out_shared, dict(reading.titles))


_Item = TypeVar("_Item", _Mention, _Statement, _Shared)


def _kept(items: Iterable[_Item], settle: Callable[[_Item], _Item | None], tally: _Tally) -> list[_Item]:
    """``settle`` every item; one that names a sentence heading is dropped and counted."""
    out: list[_Item] = []
    for item in items:
        settled = settle(item)
        if settled is None:
            tally.drop(_Drop.SENTENCE_TITLE, _what(item), item.span.quote)
        else:
            out.append(settled)
    return out


def _holder(d: _Delegation) -> EntityRef:
    """The service a delegation names (``applies_to`` is read as a service, like the assembler does)."""
    return EntityRef(kind=NodeKind.SERVICE, name=d.applies_to or "")


def _name_of(entity: EntityRef | None) -> str | None:
    """A delegation whose holder is no service (a sentence heading) applies to the whole document."""
    return entity.name if entity is not None else None


def _what(item: _Mention | _Statement | _Shared) -> str:
    """Drop-example label of a grounded item (same wording as the other drop reasons)."""
    if isinstance(item, _Mention):
        return f"개체({item.kind.value} {item.name})"
    if isinstance(item, _Statement):
        return f"관계({item.kind.value} {item.src.name}→{item.dst.name})"
    return f"묶음 관계({item.kind.value} 문서 전체→{item.dst.name})"


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
        self.notes: list[str] = []
        self.roster_ids = frozenset(resolver.person_node_id(e) for e in resolver.roster)
        #: per document: alerts that apply to whichever service raised them and name none
        self.generic_alerts: dict[str, list[str]] = defaultdict(list)
        #: documents whose notes already say that they leave no service to check
        self.explained: set[str] = set()
        self._fences: dict[str, tuple[tuple[int, int], ...]] = {}

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
        if self._code_only(m.extractor, m.kind, m.name, m.doc):
            self.tally.drop(_Drop.CODE_ONLY, what, m.span.quote)
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
        if m.extractor is Extractor.LLM and target.kind is NodeKind.TRIGGER:
            attrs.update(self._paging_tool(m.attrs, attrs))
        merged = _Target(
            target.node_id,
            target.kind,
            target.name,
            {**target.attrs, **attrs},
            target.hold_reason,
            target.unresolved,
        )
        self._node(merged, _runbook_prov(m.extractor, m.span))

    def _paging_tool(self, raw: tuple[ExtractedAttr, ...], attrs: Mapping[str, Attr]) -> dict[str, Attr]:
        """A signal delivered by a paging tool, with nobody named, pages whoever the tool routes to.

        "The alert is raised as a page in <tool>" states *where* the recipients are decided (the
        tool's schedule and routing), not who they are. Recorded as the trigger's detection
        delegation, so a trigger without a notify target is reported 확인 불가, not 호출 대상 없음.
        A named target, when the document gives one, still decides the verdict.
        """
        channel = attrs.get(AttrKey.CHANNEL.value)
        node = self.graph.nodes.get(str(channel.value)) if channel is not None else None
        if channel is None or node is None or node.attr(AttrKey.MEDIUM) != ChannelMedium.PAGER.value:
            return {}
        surface = next(
            (_text(a.value) for a in raw if a.key == AttrKey.CHANNEL.value and _text(a.value)), node.name
        )
        attr = Attr(value=surface or node.name, spans=channel.spans, source=SourceKind.RUNBOOK)
        return {delegation_key(DelegatedElement.DETECTION): attr}

    def add_statement(self, s: _Statement) -> None:
        nearby = s.extractor is Extractor.LLM
        src_span = _endpoint_span(s.doc, s.src.name, s.span, nearby=nearby)
        dst_span = _endpoint_span(s.doc, s.dst.name, s.span, nearby=nearby)
        src = self._resolve(s.src.kind, s.src.name, s.doc.doc_id, src_span)
        dst = self._resolve(s.dst.kind, s.dst.name, s.doc.doc_id, dst_span)
        what = f"관계({s.kind.value} {s.src.name}→{s.dst.name})"
        if src is None or dst is None:
            self.tally.drop(_Drop.NAMELESS, what, s.span.quote)
            return
        if any(self._code_only(s.extractor, r.kind, r.name, s.doc) for r in (s.src, s.dst)):
            self.tally.drop(_Drop.CODE_ONLY, what, s.span.quote)
            return
        if not _signature_ok(s.kind, src.kind, dst.kind) or src.node_id == dst.node_id:
            self.tally.drop(_Drop.INVALID, what, s.span.quote)
            return
        self._node(src, _runbook_prov(s.extractor, src_span))
        self._node(dst, _runbook_prov(s.extractor, dst_span))
        self.tally.accept(s.method)
        attrs = self._attrs(s.attrs, s.doc, s.span, s.extractor, what)
        self._edge(s.kind, src.node_id, dst.node_id, attrs, _runbook_prov(s.extractor, s.span))

    # ------------------------------------------------------------------ document-level statements
    def _doc_services(self, doc_id: str) -> list[str]:
        """SERVICE nodes this document mentions (runbook spans in it), in id order."""
        return sorted(
            n.id
            for n in self.graph.nodes.values()
            if n.kind is NodeKind.SERVICE and any(s.doc_id == doc_id for s in n.spans)
        )

    def add_shared(self, s: _Shared) -> None:
        """A relation the document states for everything it covers links every service it mentions.

        Resolution steps that name no service, a detection signal for "all components": the
        sentence states the relation for each service of the document (the same reading as a
        document-wide delegation). A document that mentions no service has nothing to link; the
        statement still counts as grounded, and the extraction log keeps it.
        """
        what = f"묶음 관계({s.kind.value} 문서 전체→{s.dst.name})"
        services = self._doc_services(s.doc.doc_id)
        if not services:
            self.tally.accept(s.method)
            return
        dst_span = _endpoint_span(s.doc, s.dst.name, s.span, nearby=True)
        dst = self._resolve(s.dst.kind, s.dst.name, s.doc.doc_id, dst_span)
        if dst is None:
            self.tally.drop(_Drop.NAMELESS, what, s.span.quote)
            return
        if not _signature_ok(s.kind, NodeKind.SERVICE, dst.kind):
            self.tally.drop(_Drop.INVALID, what, s.span.quote)
            return
        self.tally.accept(s.method)
        dst_id = self._node(dst, _runbook_prov(s.extractor, dst_span))
        for sid in services:
            if sid != dst_id:
                self._edge(s.kind, sid, dst_id, {}, _runbook_prov(s.extractor, s.span))

    def add_subject(self, s: _Subject) -> None:
        """An alert that applies to whichever service raised it detects every service of its runbook.

        The services the alert names were linked as DETECTED_BY statements already (see
        :func:`_subject_statements`); this adds the "any service" reading, which needs the full set
        of services the document mentions. A runbook of such an alert that names no service leaves
        no service to check — said in a note, because no finding is not the same as no gap. What
        the document hands off to an external source is still recorded, on the alert itself (see
        :meth:`add_delegation`), and reported as 확인 불가.

        The same holds for the runbook of an alert in a document that mentions no service at all,
        whatever the extractor's ``any_service`` flag says: its hand-offs ("owners: see the
        catalog") are statements about whatever the alert fires for, so they are kept on the alert
        (:data:`~jettae.domain.UNNAMED_SERVICE_ALERT`, worded apart) instead of being dropped with
        a note that the document has nothing to apply them to.
        """
        if s.kind is not SubjectKind.ALERT:
            return
        services = self._doc_services(s.doc.doc_id)
        if not s.any_service and services:
            return
        span = _endpoint_span(s.doc, s.name, s.span, nearby=True)
        alert = self._resolve(NodeKind.TRIGGER, s.name, s.doc.doc_id, span)
        if alert is None:
            return
        trigger_id = self._node(alert, _runbook_prov(s.extractor, span))
        if not services:
            self.generic_alerts[s.doc.doc_id].append(trigger_id)
            self.explained.add(s.doc.doc_id)
            marker = Attr(value=True, spans=(s.span,), source=SourceKind.RUNBOOK)
            key = ANY_SERVICE_ALERT if s.any_service else UNNAMED_SERVICE_ALERT
            node = self.graph.nodes[trigger_id]
            self.graph.add_node(node.model_copy(update={"attrs": {key: marker}}))
            self.notes.append(
                f"'{alert.name}' 알람 런북은 특정 서비스를 지정하지 않고 경보를 낸 서비스에 "
                "적용된다고만 적혀 있어, 이 문서에는 대응 경로(5요소)를 검사할 서비스가 없습니다"
                "(검사 대상 없음 ≠ 정상). 문서가 외부 자료로 넘긴 요소가 있으면 그 요소만 확인 불가로 따로 "
                "알립니다. 알람이 감시하는 서비스를 문서에 적으면 그 서비스의 감지 신호로 연결합니다."
                if s.any_service
                else f"'{alert.name}' 알람 런북에는 이 알람이 감시하는 서비스가 적혀 있지 않아, 이 문서에는 "
                "대응 경로(5요소)를 검사할 서비스가 없습니다(검사 대상 없음 ≠ 정상). 문서가 외부 자료로 넘긴 "
                "요소가 있으면 이 알람에 대해 그 요소만 확인 불가로 따로 알립니다. 알람이 감시하는 서비스를 "
                "문서에 적으면 그 서비스의 감지 신호로 연결합니다."
            )
            return
        for sid in services:
            self._edge(EdgeKind.DETECTED_BY, sid, trigger_id, {}, _runbook_prov(s.extractor, s.span))

    def add_delegation(self, d: _Delegation) -> None:
        """Record "element X of this service lives in source Y" on the service node(s) concerned.

        The fact is an attribute (:func:`~jettae.domain.delegation_key`) carrying the delegating
        sentence as its span; the checks read it to report the element as 확인 불가 instead of
        missing. A delegation for the whole document applies to every service it mentions — or,
        in the runbook of an alert that names no service, to that alert: the document still says
        where the element lives for whichever service raises it, and the report must say so rather
        than leave the element silently unchecked.
        """
        what = f"외부 위임({d.element.value} → {d.target})"
        target = clean_surface(d.target)
        if not target:
            self.tally.drop(_Drop.NAMELESS, what, d.span.quote)
            return
        if d.applies_to is None:
            doc_id = d.doc.doc_id
            holders = self._doc_services(doc_id) or list(dict.fromkeys(self.generic_alerts.get(doc_id, ())))
        else:
            span = _endpoint_span(d.doc, d.applies_to, d.span, nearby=True)
            resolved = self._resolve(NodeKind.SERVICE, d.applies_to, d.doc.doc_id, span)
            if resolved is None:
                self.tally.drop(_Drop.NAMELESS, what, d.span.quote)
                return
            holders = [self._node(resolved, _runbook_prov(d.extractor, span))]
        self.tally.accept(d.method)
        if not holders:
            self.notes.append(
                f"이 문서('{d.doc.title or d.doc.doc_id}')는 {_DELEGATED_KO[d.element]} 항목을 외부 "
                f"자료('{target}')로 넘겼지만, 그 위임을 적용할 서비스나 알람이 문서에 없어 검사하지 "
                "않았습니다(검사 대상 없음 ≠ 정상)."
            )
            return
        attr = Attr(value=target, spans=(d.span,), source=SourceKind.RUNBOOK)
        for nid in holders:
            node = self.graph.nodes[nid]
            # Same provenance, one more attribute: the first stated source of an element wins.
            self.graph.add_node(node.model_copy(update={"attrs": {delegation_key(d.element): attr}}))

    def _code_only(self, extractor: Extractor, kind: NodeKind, surface: str, doc: Document) -> bool:
        """An LLM-read service/system name the document writes only inside fenced code.

        Commands, configuration and captured output are text to copy, not statements about who
        responds (:mod:`jettae.extract.markdown`); a host or label that occurs nowhere else would
        become a service the response-path rules then demand five elements of. A catalog service
        is trusted whatever the runbook does with its name, and a name the document never spells
        verbatim (normalised by the extractor) cannot be judged and is kept.
        """
        if extractor is not Extractor.LLM or kind not in _SERVICE_KINDS:
            return False
        if self.resolver.resolve_service(surface) is not None:
            return False
        fences = self._fences.get(doc.doc_id)
        if fences is None:
            fences = self._fences[doc.doc_id] = fenced_ranges(doc.text)
        needle = clean_surface(surface)
        if not fences or not needle:
            return False
        hits = [m.span() for m in re.finditer(re.escape(needle), doc.text, flags=re.IGNORECASE)]
        return bool(hits) and all(in_ranges(lo, hi, fences) for lo, hi in hits)

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


def _endpoint_span(doc: Document, surface: str, relation: SourceSpan, *, nearby: bool = False) -> SourceSpan:
    """Evidence for a relation endpoint: its own mention inside the relation quote when it can
    be found there, otherwise the relation quote itself (which states the endpoint).

    With ``nearby`` (statements of the LLM extractor, which names endpoints from the surrounding
    section), an endpoint the relation quote does not name takes its nearest verbatim mention in
    the document instead: a service whose only evidence were an unrelated sentence would send a
    reader to a line that never names it. Fuzzy matches are not used for this — a merely similar
    name elsewhere is not a mention. The rule parser keeps the relation quote (its relations are
    scoped by the heading they sit under, and that evidence is part of its fixed output).
    """
    hit = align_quote(doc, surface, hint_start=relation.start, hint_end=relation.end)
    if hit is None:
        return relation
    if relation.start <= hit.span.start and hit.span.end <= relation.end:
        return hit.span
    return hit.span if nearby and hit.method != "fuzzy" else relation


def _signature_ok(kind: EdgeKind, src: NodeKind, dst: NodeKind) -> bool:
    signature = EDGE_SIGNATURES.get(kind)
    return signature is not None and src in signature[0] and dst in signature[1]


def _conflicts(old: dict[str, Attr], new: dict[str, Attr]) -> bool:
    return any(k in old and old[k].value != v.value for k, v in new.items())


#: How many names a note lists before "외 N개".
_NOTE_NAMES = 3


def _quoted(names: Iterable[str]) -> str:
    items = list(dict.fromkeys(names))
    shown = ", ".join(f"'{n}'" for n in items[:_NOTE_NAMES])
    return shown + (f" 외 {len(items) - _NOTE_NAMES}개" if len(items) > _NOTE_NAMES else "")


def _no_service_note(doc: Document, graph: OpsGraph, titles: Iterable[str]) -> str:
    """Why an LLM-read document leaves no service to check, from what its graph does hold.

    The reasons are structural (what kinds of things the document was read to name), so the note
    is the same for any organisation's document: a sentence title that is not a service, only
    systems or tools, alerts that name no service, or only roles, teams and contact means.
    """

    def named(*kinds: NodeKind) -> list[str]:
        return [n.name for n in graph.nodes_of(*kinds) if any(s.doc_id == doc.doc_id for s in n.spans)]

    reasons: list[str] = []
    sentences = list(titles)
    if sentences:
        reasons.append(
            f"서비스 이름처럼 읽힌 제목 {_quoted(sentences)}은(는) 상황을 서술한 문장이라 "
            "서비스로 보지 않았습니다"
        )
    systems, alerts = named(NodeKind.SYSTEM), named(NodeKind.TRIGGER)
    if systems:
        reasons.append(f"문서가 다루는 대상을 시스템·도구({_quoted(systems)})로만 적었습니다")
    if alerts:
        reasons.append(
            f"알람·감지 신호({_quoted(alerts)})는 있지만 그것이 감시하는 서비스가 적혀 있지 않습니다"
        )
    if not systems and not alerts:
        others = named(NodeKind.ROLE, NodeKind.TEAM, NodeKind.PERSON, NodeKind.CHANNEL)
        if others:
            reasons.append(f"문서가 역할·팀·연락 수단({_quoted(others)})만 다루는 절차·안내로 읽혔습니다")
    if not reasons:
        reasons.append("문서에서 서비스·시스템을 하나도 읽지 못했습니다")
    title = doc.title or doc.path or doc.doc_id
    return (
        f"이 문서('{title}')에는 대응 경로(5요소)를 검사할 서비스가 없어 5요소를 검사하지 않았습니다"
        f"(검사 대상 없음 ≠ 정상). 이유: {'; '.join(reasons)}. 문서가 다루는 서비스를 적거나 서비스 "
        "카탈로그를 함께 올리면 그 서비스를 검사합니다."
    )


def _describe(entry: RosterEntry) -> str:
    details = " ".join(x for x in (entry.team, entry.title) if x)
    return f"{entry.name}({details})" if details else entry.name
