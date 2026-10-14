"""Operations graph (운영 그래프) — the core asset of JETTAE Ops.

The graph answers "무엇이 어디에 연결되고, 누가 맡고 승인하며, 언제까지 알리나".
It is a *value object*: every change produces a new graph (see ``jettae.improve.patching``)
and every version is content-addressed by :meth:`OpsGraph.content_hash`.

Invariants (enforced by :meth:`OpsGraph.validate_integrity` and property tests):

* Every node and edge carries at least one :class:`Provenance`. Elements extracted from a
  runbook carry the exact :class:`SourceSpan` they came from — nothing is invented.
* An attribute that the source does not state is *absent* (key missing), never guessed.
* Edges reference existing nodes; ids are deterministic (see :func:`node_id` / :func:`edge_id`).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import unicodedata
from collections.abc import Iterable, Iterator
from enum import StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

# --------------------------------------------------------------------------- vocab


class NodeKind(StrEnum):
    SERVICE = "service"  # 사용자에게 보이는 서비스 (결제, 장바구니, 송출 …)
    SYSTEM = "system"  # 인프라 구성요소 (DB, Kafka, Redis, 클러스터 …)
    LOCATION = "location"  # 장애 도메인 (데이터센터, 리전, AZ, SaaS 벤더)
    PERSON = "person"  # 개인 (명부 기준)
    ROLE = "role"  # 직책/역할 (팀장, CTO, 1차 당직 …) — 명부·당직표로 사람에 해석
    TEAM = "team"
    CHANNEL = "channel"  # 연락 수단 (Slack, 카카오톡, 전화, SMS, PagerDuty …)
    TRIGGER = "trigger"  # 감지 신호 (알람·모니터·고객 문의)
    ACTION = "action"  # 조치 (롤백, 페일오버, 재시작, 기능 플래그 OFF …)
    OBLIGATION = "obligation"  # 신고·통지 의무 (KISA 24h, 개인정보위 72h, 고객 SLA 통지)


class EdgeKind(StrEnum):
    DEPENDS_ON = "depends_on"  # service|system|channel -> service|system
    RUNS_IN = "runs_in"  # service|system|channel -> location
    OWNED_BY = "owned_by"  # service -> person|role|team            (담당)
    DETECTED_BY = "detected_by"  # service -> trigger                    (감지)
    NOTIFIES = "notifies"  # trigger -> person|role|team (attr channel) (호출)
    ESCALATES_TO = "escalates_to"  # person|role|team -> person|role|team (상향, attr after_minutes)
    MITIGATED_BY = "mitigated_by"  # service -> action                     (롤백 등)
    APPROVED_BY = "approved_by"  # action -> person|role                  (승인)
    EXECUTED_BY = "executed_by"  # action -> person|role|team            (실행)
    REACHABLE_VIA = "reachable_via"  # person|role|team -> channel          (연락처)
    FALLBACK_TO = "fallback_to"  # channel -> channel                    (대체 채널)
    HOLDS_ROLE = "holds_role"  # person -> role
    MEMBER_OF = "member_of"  # person -> team
    MUST_REPORT = "must_report"  # service -> obligation                (신고/통지)
    REPORTED_BY = "reported_by"  # obligation -> person|role|team        (신고 담당)


#: Allowed (src kinds, dst kinds) per edge kind. Used by integrity validation and by the
#: LLM extractor's output validation so a model cannot create nonsensical relations.
EDGE_SIGNATURES: dict[EdgeKind, tuple[frozenset[NodeKind], frozenset[NodeKind]]] = {
    EdgeKind.DEPENDS_ON: (
        frozenset({NodeKind.SERVICE, NodeKind.SYSTEM, NodeKind.CHANNEL}),
        frozenset({NodeKind.SERVICE, NodeKind.SYSTEM}),
    ),
    EdgeKind.RUNS_IN: (
        frozenset({NodeKind.SERVICE, NodeKind.SYSTEM, NodeKind.CHANNEL}),
        frozenset({NodeKind.LOCATION}),
    ),
    EdgeKind.OWNED_BY: (
        frozenset({NodeKind.SERVICE, NodeKind.SYSTEM}),
        frozenset({NodeKind.PERSON, NodeKind.ROLE, NodeKind.TEAM}),
    ),
    EdgeKind.DETECTED_BY: (frozenset({NodeKind.SERVICE, NodeKind.SYSTEM}), frozenset({NodeKind.TRIGGER})),
    EdgeKind.NOTIFIES: (
        frozenset({NodeKind.TRIGGER}),
        frozenset({NodeKind.PERSON, NodeKind.ROLE, NodeKind.TEAM}),
    ),
    EdgeKind.ESCALATES_TO: (
        frozenset({NodeKind.PERSON, NodeKind.ROLE, NodeKind.TEAM}),
        frozenset({NodeKind.PERSON, NodeKind.ROLE, NodeKind.TEAM}),
    ),
    EdgeKind.MITIGATED_BY: (frozenset({NodeKind.SERVICE, NodeKind.SYSTEM}), frozenset({NodeKind.ACTION})),
    EdgeKind.APPROVED_BY: (frozenset({NodeKind.ACTION}), frozenset({NodeKind.PERSON, NodeKind.ROLE})),
    EdgeKind.EXECUTED_BY: (
        frozenset({NodeKind.ACTION}),
        frozenset({NodeKind.PERSON, NodeKind.ROLE, NodeKind.TEAM}),
    ),
    EdgeKind.REACHABLE_VIA: (
        frozenset({NodeKind.PERSON, NodeKind.ROLE, NodeKind.TEAM}),
        frozenset({NodeKind.CHANNEL}),
    ),
    EdgeKind.FALLBACK_TO: (frozenset({NodeKind.CHANNEL}), frozenset({NodeKind.CHANNEL})),
    EdgeKind.HOLDS_ROLE: (frozenset({NodeKind.PERSON}), frozenset({NodeKind.ROLE})),
    EdgeKind.MEMBER_OF: (frozenset({NodeKind.PERSON}), frozenset({NodeKind.TEAM})),
    EdgeKind.MUST_REPORT: (frozenset({NodeKind.SERVICE, NodeKind.SYSTEM}), frozenset({NodeKind.OBLIGATION})),
    EdgeKind.REPORTED_BY: (
        frozenset({NodeKind.OBLIGATION}),
        frozenset({NodeKind.PERSON, NodeKind.ROLE, NodeKind.TEAM}),
    ),
}


class AttrKey(StrEnum):
    """Well-known attribute keys. Unknown keys are allowed (extensibility) but ignored by rules."""

    # service / system
    TIER = "tier"  # int 1..3 (1 = 핵심)
    SLO_MINUTES = "slo_minutes"  # int, 목표 복구 시간
    DESCRIPTION = "description"
    # person
    STATUS = "status"  # PersonStatus value
    TITLE = "title"  # 직급/직책 문자열 (대리, 팀장 …)
    ALIASES = "aliases"  # list[str]
    # channel
    MEDIUM = "medium"  # ChannelMedium value
    # trigger
    CONDITION = "condition"  # 감지 조건 원문
    DELAY_MINUTES = "delay_minutes"  # int, 감지 지연
    # action
    ACTION_TYPE = "action_type"  # ActionType value
    REQUIRES_APPROVAL = "requires_approval"  # bool
    DURATION_MINUTES = "duration_minutes"  # int
    # obligation
    RECIPIENT = "recipient"  # 신고 기관 (KISA, 개인정보보호위원회 …)
    DEADLINE_MINUTES = "deadline_minutes"  # int
    INCIDENT_TYPE = "incident_type"  # IncidentType value
    LEGAL_BASIS = "legal_basis"
    # location
    LOCATION_KIND = "location_kind"  # datacenter|region|az|saas
    # edges
    AFTER_MINUTES = "after_minutes"  # ESCALATES_TO: 미응답 후 상향까지 분
    CHANNEL = "channel"  # NOTIFIES: channel *name* in the Extraction IR; channel node id in the graph


class PersonStatus(StrEnum):
    ACTIVE = "active"
    DEPARTED = "departed"  # 퇴사
    ON_LEAVE = "on_leave"  # 장기 부재
    UNKNOWN = "unknown"  # 명부에서 확인 불가 (확인 불가 ≠ 정상)


class ChannelMedium(StrEnum):
    SLACK = "slack"
    KAKAOTALK = "kakaotalk"
    TEAMS = "teams"
    PHONE = "phone"
    SMS = "sms"
    EMAIL = "email"
    PAGER = "pager"  # PagerDuty / Opsgenie / incident.io 등 호출 도구
    INTERNAL_MESSENGER = "internal_messenger"
    OTHER = "other"


class ActionType(StrEnum):
    ROLLBACK = "rollback"
    FAILOVER = "failover"
    RESTART = "restart"
    FEATURE_FLAG = "feature_flag"
    SCALE = "scale"
    OTHER = "other"


class LocationKind(StrEnum):
    DATACENTER = "datacenter"
    REGION = "region"
    AZ = "az"
    SAAS = "saas"


class IncidentType(StrEnum):
    OUTAGE = "outage"  # 서비스 장애
    BREACH = "breach"  # 침해사고 (정보통신망법)
    PRIVACY_LEAK = "privacy_leak"  # 개인정보 유출 (개인정보보호법)


class SourceKind(StrEnum):
    RUNBOOK = "runbook"  # 런북·문서 원문 (span 필수)
    CATALOG = "catalog"  # 서비스 카탈로그·배포 설정·OTel 서비스 그래프
    ROSTER = "roster"  # 인사 명부
    ONCALL = "oncall"  # 당직표
    POLICY = "policy"  # 팀 합의 규정 / 법정 기준
    PATCH = "patch"  # 사람이 승인한 수정안


class Extractor(StrEnum):
    RULE = "rule"  # 규칙 파서 (LLM 없음)
    LLM = "llm"  # 생성형 AI 추출 (원문 span 정렬·역번역 대조 통과분)
    IMPORT = "import"  # 구조화 입력 (카탈로그/명부/당직표)
    HUMAN = "human"  # 승인된 수정안


class Verification(StrEnum):
    VERIFIED = "verified"
    HOLD = "hold"  # 판단 보류: 역번역 대조 불일치, 동명이인, 모호한 정렬 …


# --------------------------------------------------------------------------- provenance


class SourceSpan(BaseModel):
    """A character range in a normalized document (``Document.text``)."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    quote: str
    line: int | None = Field(default=None, ge=1, description="1-based line of `start` (SARIF/GitHub)")

    @model_validator(mode="after")
    def _ordered(self) -> SourceSpan:
        if self.end < self.start:
            raise ValueError("span end < start")
        return self


class Provenance(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: SourceKind
    extractor: Extractor
    spans: tuple[SourceSpan, ...] = ()
    ref: str | None = Field(default=None, description="catalog key, roster row, patch id …")

    @model_validator(mode="after")
    def _runbook_needs_span(self) -> Provenance:
        if self.source is SourceKind.RUNBOOK and not self.spans:
            raise ValueError("runbook provenance requires at least one source span (지어내기 금지)")
        return self


JsonScalar = str | int | float | bool
AttrValue = JsonScalar | list[str]


class Attr(BaseModel):
    """A single attribute value with the span(s) that justify it."""

    model_config = ConfigDict(frozen=True)

    value: AttrValue
    spans: tuple[SourceSpan, ...] = ()
    source: SourceKind = SourceKind.RUNBOOK


# --------------------------------------------------------------------------- ids


_WS = re.compile(r"\s+")
_STRIP = re.compile(r"[^\w가-힣ㄱ-ㅎㅏ-ㅣ\-.]+")


def slug(name: str) -> str:
    """Deterministic, Korean-preserving slug. ``"결제 서비스"`` -> ``"결제-서비스"``."""
    text = unicodedata.normalize("NFC", name).strip().lower()
    text = _WS.sub("-", text)
    text = _STRIP.sub("", text)
    return text.strip("-") or "unnamed"


def node_id(kind: NodeKind, name: str) -> str:
    return f"{kind.value}:{slug(name)}"


def edge_id(kind: EdgeKind, src: str, dst: str, discriminator: str | None = None) -> str:
    base = f"{kind.value}:{src}->{dst}"
    return f"{base}#{discriminator}" if discriminator else base


# --------------------------------------------------------------------------- elements


class Node(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    kind: NodeKind
    name: str
    attrs: dict[str, Attr] = Field(default_factory=dict)
    prov: tuple[Provenance, ...]
    verification: Verification = Verification.VERIFIED
    hold_reason: str | None = None

    def attr(self, key: AttrKey | str, default: Any = None) -> Any:
        a = self.attrs.get(str(key))
        return default if a is None else a.value

    @property
    def spans(self) -> tuple[SourceSpan, ...]:
        return tuple(s for p in self.prov for s in p.spans)


class Edge(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    kind: EdgeKind
    src: str
    dst: str
    attrs: dict[str, Attr] = Field(default_factory=dict)
    prov: tuple[Provenance, ...]
    verification: Verification = Verification.VERIFIED
    hold_reason: str | None = None

    def attr(self, key: AttrKey | str, default: Any = None) -> Any:
        a = self.attrs.get(str(key))
        return default if a is None else a.value

    @property
    def spans(self) -> tuple[SourceSpan, ...]:
        return tuple(s for p in self.prov for s in p.spans)


class DocumentRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    title: str
    sha256: str
    path: str | None = None


class IntegrityError(ValueError):
    pass


class OpsGraph(BaseModel):
    """Immutable operations graph. Build with :class:`GraphBuilder`; change with patches."""

    model_config = ConfigDict(frozen=True)

    nodes: dict[str, Node] = Field(default_factory=dict)
    edges: dict[str, Edge] = Field(default_factory=dict)
    docs: dict[str, DocumentRef] = Field(default_factory=dict)

    # Lazily built adjacency index (not part of the value; excluded from hashing/serialisation).
    _out: dict[str, list[Edge]] | None = PrivateAttr(default=None)
    _in: dict[str, list[Edge]] | None = PrivateAttr(default=None)

    def _index(self) -> tuple[dict[str, list[Edge]], dict[str, list[Edge]]]:
        if self._out is None or self._in is None:
            out: dict[str, list[Edge]] = {}
            inn: dict[str, list[Edge]] = {}
            for e in sorted(self.edges.values(), key=lambda e: e.id):
                out.setdefault(e.src, []).append(e)
                inn.setdefault(e.dst, []).append(e)
            self._out, self._in = out, inn
        return self._out, self._in

    # ---- queries -------------------------------------------------------
    def node(self, nid: str) -> Node:
        return self.nodes[nid]

    def nodes_of(self, *kinds: NodeKind) -> list[Node]:
        return sorted((n for n in self.nodes.values() if n.kind in kinds), key=lambda n: n.id)

    def out_edges(self, nid: str, *kinds: EdgeKind) -> list[Edge]:
        out, _ = self._index()
        return [e for e in out.get(nid, ()) if not kinds or e.kind in kinds]

    def in_edges(self, nid: str, *kinds: EdgeKind) -> list[Edge]:
        _, inn = self._index()
        return [e for e in inn.get(nid, ()) if not kinds or e.kind in kinds]

    def successors(self, nid: str, *kinds: EdgeKind) -> list[Node]:
        return [self.nodes[e.dst] for e in self.out_edges(nid, *kinds)]

    def predecessors(self, nid: str, *kinds: EdgeKind) -> list[Node]:
        return [self.nodes[e.src] for e in self.in_edges(nid, *kinds)]

    def edges_of(self, *kinds: EdgeKind) -> list[Edge]:
        return sorted((e for e in self.edges.values() if e.kind in kinds), key=lambda e: e.id)

    def __iter__(self) -> Iterator[Node]:  # type: ignore[override]
        return iter(sorted(self.nodes.values(), key=lambda n: n.id))

    # ---- identity ------------------------------------------------------
    def canonical_json(self) -> str:
        payload = self.model_dump(mode="json")
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def content_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    # ---- interop -------------------------------------------------------
    def to_networkx(self) -> Any:
        """Return a ``networkx.MultiDiGraph`` view (imported lazily; domain stays light)."""
        import networkx as nx

        g = nx.MultiDiGraph()
        for n in self.nodes.values():
            g.add_node(n.id, kind=n.kind.value, name=n.name)
        for e in self.edges.values():
            g.add_edge(e.src, e.dst, key=e.id, kind=e.kind.value)
        return g

    # ---- integrity -----------------------------------------------------
    def validate_integrity(self) -> None:
        for nid, n in self.nodes.items():
            if nid != n.id:
                raise IntegrityError(f"node key {nid!r} != id {n.id!r}")
            if not n.prov:
                raise IntegrityError(f"node {nid} has no provenance")
        for eid, e in self.edges.items():
            if eid != e.id:
                raise IntegrityError(f"edge key {eid!r} != id {e.id!r}")
            if not e.prov:
                raise IntegrityError(f"edge {eid} has no provenance")
            if e.src not in self.nodes or e.dst not in self.nodes:
                raise IntegrityError(f"edge {eid} references a missing node")
            src_kinds, dst_kinds = EDGE_SIGNATURES[e.kind]
            if self.nodes[e.src].kind not in src_kinds or self.nodes[e.dst].kind not in dst_kinds:
                raise IntegrityError(
                    f"edge {eid}: {self.nodes[e.src].kind}->{self.nodes[e.dst].kind} not allowed for {e.kind}"
                )


def merge_provenance(a: Iterable[Provenance], b: Iterable[Provenance]) -> tuple[Provenance, ...]:
    """``a`` then ``b`` with duplicates dropped, keeping the first occurrence of each.

    Frozen models hash and compare by their field values, so equal provenance (same source,
    extractor, spans and ref — exactly what equal ``model_dump_json`` output meant) collapses in
    one pass without serializing anything. :class:`GraphBuilder` applies this same rule
    incrementally (see :class:`_ProvenanceIndex`).
    """
    return tuple(dict.fromkeys(itertools.chain(a, b)))


class _ProvenanceIndex:
    """Per-element ordered sets of the provenance a :class:`GraphBuilder` already holds.

    Re-running :func:`merge_provenance` over everything an element has accumulated on each merge
    hashes (a Python-level ``__hash__`` per model, recursing into its spans) every earlier entry
    again, which is quadratic in the mentions of one entity (a person named in every section of a
    long runbook). Instead each merged element keeps its provenance as an insertion-ordered dict
    keyed by the frozen models themselves, so a merge hashes only the *incoming* entries and
    appends the new ones to the held tuple; the result is exactly
    ``merge_provenance(held, incoming)`` (same entries, same objects, same order — P6). The tuple
    append is a C-level pointer copy, so the element stays exact after every merge at a cost that
    is negligible next to the hashing it replaces.

    An entry is trusted only while the stored element still carries the very tuple the entry was
    made for. An element replaced from outside (``builder.nodes[nid] = ...``, as tests do) has a
    different tuple and is re-indexed from its own provenance on its next merge, so such writes
    keep their usual "the stored element is the truth" meaning. Tuples are immutable, so the same
    tuple object always means the same entries.
    """

    __slots__ = ("_held",)

    def __init__(self) -> None:
        self._held: dict[str, tuple[tuple[Provenance, ...], dict[Provenance, None]]] = {}

    def union(
        self, key: str, held: tuple[Provenance, ...], incoming: Iterable[Provenance]
    ) -> tuple[Provenance, ...]:
        """``merge_provenance(held, incoming)`` for the element ``key``, hashing only ``incoming``."""
        entry = self._held.get(key)
        if entry is not None and entry[0] is held:
            seen = entry[1]
        else:  # first merge of this element, or it was replaced from outside: index it once
            seen = dict.fromkeys(held)
            if len(seen) != len(held):  # an element as first added may repeat an entry
                held = tuple(seen)
        fresh: list[Provenance] = []
        for p in incoming:
            if p not in seen:
                seen[p] = None
                fresh.append(p)
        merged = held + tuple(fresh) if fresh else held
        self._held[key] = (merged, seen)
        return merged

    def forget(self, key: str) -> None:
        self._held.pop(key, None)


class GraphBuilder:
    """Mutable helper to assemble an :class:`OpsGraph`. Merges duplicates deterministically.

    Merge semantics (used by extractors, importers and tests alike):
    * same node id -> provenance union; attrs: first writer wins *unless* the later attr comes
      from a stronger source (ROSTER/CATALOG/PATCH beat RUNBOOK for identity attributes such as
      ``status``), verification degrades to HOLD if any contributor is HOLD.
    * same edge id -> provenance union, attrs merged the same way.

    The provenance union is :func:`merge_provenance` of everything contributed, in order, but a
    merge only hashes the incoming provenance (:class:`_ProvenanceIndex`), so the dedupe work is
    linear in the number of mentions. ``nodes`` / ``edges`` and the elements returned by
    :meth:`add_node` / :meth:`add_edge` always carry the complete merged provenance.
    """

    _STRONG: ClassVar[dict[SourceKind, int]] = {
        SourceKind.PATCH: 3,
        SourceKind.ROSTER: 2,
        SourceKind.ONCALL: 2,
        SourceKind.CATALOG: 2,
        SourceKind.POLICY: 1,
        SourceKind.RUNBOOK: 0,
    }

    def __init__(self, base: OpsGraph | None = None) -> None:
        self.nodes: dict[str, Node] = dict(base.nodes) if base else {}
        self.edges: dict[str, Edge] = dict(base.edges) if base else {}
        self.docs: dict[str, DocumentRef] = dict(base.docs) if base else {}
        self._node_prov = _ProvenanceIndex()
        self._edge_prov = _ProvenanceIndex()

    def _merge_attrs(self, old: dict[str, Attr], new: dict[str, Attr]) -> dict[str, Attr]:
        out = dict(old)
        for k, v in new.items():
            cur = out.get(k)
            if cur is None or self._STRONG[v.source] > self._STRONG[cur.source]:
                out[k] = v
        return out

    def add_node(self, node: Node) -> Node:
        cur = self.nodes.get(node.id)
        if cur is None:
            self.nodes[node.id] = node
            self._node_prov.forget(node.id)
            return node
        merged = cur.model_copy(
            update={
                "attrs": self._merge_attrs(cur.attrs, node.attrs),
                "prov": self._node_prov.union(node.id, cur.prov, node.prov),
                "verification": Verification.HOLD
                if Verification.HOLD in (cur.verification, node.verification)
                else Verification.VERIFIED,
                "hold_reason": cur.hold_reason or node.hold_reason,
            }
        )
        self.nodes[node.id] = merged
        return merged

    def add_edge(self, edge: Edge) -> Edge:
        cur = self.edges.get(edge.id)
        if cur is None:
            self.edges[edge.id] = edge
            self._edge_prov.forget(edge.id)
            return edge
        merged = cur.model_copy(
            update={
                "attrs": self._merge_attrs(cur.attrs, edge.attrs),
                "prov": self._edge_prov.union(edge.id, cur.prov, edge.prov),
                "verification": Verification.HOLD
                if Verification.HOLD in (cur.verification, edge.verification)
                else Verification.VERIFIED,
                "hold_reason": cur.hold_reason or edge.hold_reason,
            }
        )
        self.edges[edge.id] = merged
        return merged

    def remove_edge(self, eid: str) -> None:
        self.edges.pop(eid, None)
        self._edge_prov.forget(eid)

    def add_doc(self, doc: DocumentRef) -> None:
        self.docs[doc.doc_id] = doc

    def build(self, *, validate: bool = True) -> OpsGraph:
        g = OpsGraph(
            nodes=dict(sorted(self.nodes.items())),
            edges=dict(sorted(self.edges.items())),
            docs=dict(sorted(self.docs.items())),
        )
        if validate:
            g.validate_integrity()
        return g
