"""Required response-path elements (필수 경로 5요소) — the single evaluator behind rules *and* score.

Why one evaluator: if the JT-PATH rules and the readiness score each walked the graph their own
way, the report could say "담당자 없음" while the owner element still earned points (or the
reverse). Every rule in ``jettae.check.rules.{path,escalation,channels}`` and the scorer in
``jettae.check.scoring`` therefore read the analyses produced here, so a finding exists exactly
when the matching element loses points for that reason.

The semantics below are published as **SCORING_V1** (rendered by
:func:`jettae.check.scoring.scoring_table_markdown` into ``docs/scoring.md``). Customers can audit
them; no LLM is involved anywhere (P1) and every function is a pure function of the
:class:`~jettae.check.context.CheckContext` (P6).

General conventions
-------------------
* *Active responder* — a person with roster status ``active`` who is not switched off by a what-if
  scenario (``ctx.unavailable``). Roles resolve to holders (HOLDS_ROLE), teams to members
  (MEMBER_OF).
* *Reachable* — the person (or the role/team they were reached through) has at least one contact
  channel (REACHABLE_VIA plus the FALLBACK_TO closure).
* Items on HOLD (판단 보류) and items in ``ctx.unavailable`` are never used as evidence that a
  requirement is met (P3: 확인 불가 ≠ 정상). An element whose status is ``UNKNOWN`` scores 0.
* Facts the sources do not state stay absent (P2): e.g. a rollback without a stated executor does
  not get the executor bonus.

Per element (score in 0..1)
---------------------------
DETECTION
    1.0 if ≥1 trigger (DETECTED_BY) has a NOTIFIES target resolving to an active responder;
    0.5 if a trigger exists but no notify target resolves; 0 if there is no trigger.
    If any DETECTED_BY edge, trigger or NOTIFIES edge of the service is on HOLD → UNKNOWN, 0.
    A NOTIFIES edge whose ``channel`` attr names an unavailable channel does not deliver.
OWNER = owner part + escalation part
    owner part 0.6 if ≥1 owner resolves to an active, reachable responder; 0.3 if active but
    unreachable; 0 otherwise (status FAIL — or UNKNOWN with score 0 when an owner person is
    unconfirmed in the roster and nobody active owns the service).
    escalation part 0.4 if the ESCALATES_TO closure from an owner node or an owner person reaches
    an active responder other than the escalating person (or the holders of the escalating role)
    through hops that all state ``after_minutes``; 0.3 if such a responder is reachable only via
    some hop without ``after_minutes``; 0 otherwise.
APPROVER
    Considers mitigation actions that need approval under the single rule
    :meth:`CheckContext.requires_approval` (shared with the rules, the SPOF analysis and the
    rehearsal simulator): an explicit ``requires_approval`` flag wins; otherwise rollback/failover
    need approval; any other action does not. Per action: 1.0 with ≥2 distinct active reachable
    approvers, 0.8 with exactly one, 0.2 if approvers are named but none is active and reachable
    (UNKNOWN with 0 instead when one of them is unconfirmed in the roster), 0 without any
    approver. Element = min over those actions; 1.0 if no action needs approval; 0 (FAIL) if the
    service has no mitigation action at all.
FALLBACK_CHANNEL
    Channels of active owner responders and of the active people at the first escalation hop:
    1.0 with ≥2 distinct known media and ≥1 channel outside the service's failure domain;
    0.5 with ≥2 channels otherwise; 0.25 with exactly one; 0 with none.
ROLLBACK
    Best mitigation action: rollback/failover/feature_flag score 0.6, +0.4 when a stated
    executor resolves to an active responder; restart/scale/other score 0.3, +0.2 with an active
    executor (0.5 max); 0 without any mitigation action.

Why naming an approver never creates a requirement: if it did, adding a valid, reachable approver
to a restart would turn an ungated step (1.0) into a gated one with a single approver (0.8), and
the monotonicity guarantee (adding a valid responder never lowers readiness) would break. A step
the runbook gates ("박팀장 승인 후 재시작") carries an explicit ``requires_approval`` and is
judged like a rollback.
"""

from __future__ import annotations

import unicodedata
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from jettae.check.context import CheckContext, requires_approval
from jettae.domain import (
    ActionType,
    AttrKey,
    ChannelMedium,
    Edge,
    EdgeKind,
    Element,
    ElementStatus,
    Node,
    NodeKind,
    OpsGraph,
    PersonStatus,
    RosterEntry,
    SourceSpan,
    Verification,
    node_id,
)

# --------------------------------------------------------------------------- published constants

DETECTION_NOTIFIED = 1.0
DETECTION_TRIGGER_ONLY = 0.5

OWNER_REACHABLE = 0.6
OWNER_ACTIVE_UNREACHABLE = 0.3
ESCALATION_TIMED = 0.4
ESCALATION_UNTIMED = 0.3

APPROVER_TWO_OR_MORE = 1.0
APPROVER_SINGLE = 0.8
APPROVER_NOT_READY = 0.2

FALLBACK_INDEPENDENT = 1.0
FALLBACK_COMMON_MODE = 0.5
FALLBACK_SINGLE = 0.25

ROLLBACK_REVERSIBLE = 0.6
ROLLBACK_EXECUTOR_BONUS = 0.4
MITIGATION_OTHER = 0.3
MITIGATION_OTHER_EXECUTOR_BONUS = 0.2

#: Mitigations that put the service back into a known state (되돌리기 조치).
REVERSIBLE_ACTION_TYPES: frozenset[ActionType] = frozenset(
    {ActionType.ROLLBACK, ActionType.FAILOVER, ActionType.FEATURE_FLAG}
)
#: Same bound as ``CheckContext.escalation_closure`` so both agree on long chains.
MAX_ESCALATION_HOPS = 12

PERSON_STATUS_KO: dict[PersonStatus, str] = {
    PersonStatus.ACTIVE: "활동 중",
    PersonStatus.DEPARTED: "퇴사",
    PersonStatus.ON_LEAVE: "장기 부재",
    PersonStatus.UNKNOWN: "명부 미확인",
}

GENERAL_SEMANTICS_KO: tuple[str, ...] = (
    "활동 중인 대응자: 명부상 재직(active)이고 가상 장애로 빠지지 않은 사람. 역할은 HOLDS_ROLE, 팀은 "
    "MEMBER_OF로 사람에 해석한다.",
    "연락 가능: 그 사람(또는 거쳐 온 역할·팀)에 연락 수단(REACHABLE_VIA, FALLBACK_TO 포함)이 1개 이상 있다.",
    "판단 보류(HOLD) 요소와 가상 장애로 꺼진 요소는 충족 근거로 쓰지 않는다. 확인 불가(UNKNOWN)는 0점이다.",
    "원문·명부에 없는 사실은 없는 그대로 둔다. 예: 실행자가 적혀 있지 않으면 실행자 가산점이 없다.",
)

SEMANTICS_KO: dict[Element, tuple[str, ...]] = {
    Element.DETECTION: (
        f"감지 신호(DETECTED_BY)의 호출(NOTIFIES) 대상이 활동 중인 대응자로 이어지면 {DETECTION_NOTIFIED}",
        f"감지 신호는 있으나 호출 대상이 없거나 활동 중인 대응자로 이어지지 않으면 {DETECTION_TRIGGER_ONLY}",
        "감지 신호가 없으면 0",
        "감지 연결·감지 신호·호출 연결 중 하나라도 판단 보류(HOLD)면 확인 불가(UNKNOWN), 0",
    ),
    Element.OWNER: (
        f"담당 부분: 활동 중이고 연락 가능한 담당자가 있으면 {OWNER_REACHABLE}, 활동 중이나 연락 수단이 "
        f"없으면 {OWNER_ACTIVE_UNREACHABLE}, 그 외 0(실패)",
        "담당자가 명부에서 확인되지 않고 활동 중인 담당자도 없으면 확인 불가(UNKNOWN), 0",
        "상향 부분: 담당 노드·담당자에서 시작한 상향(ESCALATES_TO)이 다른 활동 중 대응자에게 닿고 모든 "
        f"단계에 대기 시간(after_minutes)이 적혀 있으면 {ESCALATION_TIMED}, 대기 시간이 빠진 단계가 있으면 "
        f"{ESCALATION_UNTIMED}, 닿지 않으면 0",
    ),
    Element.APPROVER: (
        "승인이 필요한 조치만 본다(승인 필요 단일 규칙): requires_approval이 명시되면 그 값, 아니면 "
        "롤백·페일오버는 승인 필요, 그 밖의 조치는 승인 불필요. 승인자가 적혀 있다는 것만으로는 "
        "승인이 필요해지지 않는다(승인자를 추가했다고 점수가 내려가지 않도록)",
        f"조치마다 활동 중·연락 가능한 승인자가 2명 이상이면 {APPROVER_TWO_OR_MORE}, 1명이면 "
        f"{APPROVER_SINGLE}, 승인자는 적혀 있으나 아무도 활동 중·연락 가능하지 않으면 {APPROVER_NOT_READY}, "
        "승인자가 없으면 0",
        "활동 중·연락 가능한 승인자가 없고 승인자 중 명부에서 확인되지 않는 인물이 있으면 "
        "확인 불가(UNKNOWN), 0",
        "요소 점수는 조치별 점수의 최솟값. 승인이 필요한 조치가 없으면 1.0",
        "완화 조치가 하나도 없으면 0(실패): 조치가 없어 승인 경로를 평가할 수 없음",
    ),
    Element.FALLBACK_CHANNEL: (
        "활동 중인 담당자와 1차 상향 대상의 연락 수단을 모은다",
        "서로 다른 매체가 2개 이상이고 서비스와 장애 영역을 공유하지 않는 채널이 1개 이상이면 "
        f"{FALLBACK_INDEPENDENT}",
        f"채널이 2개 이상이지만 모두 같은 매체이거나 모두 서비스와 함께 죽으면 {FALLBACK_COMMON_MODE}",
        f"채널이 1개뿐이면 {FALLBACK_SINGLE}, 없으면 0",
    ),
    Element.ROLLBACK: (
        f"롤백·페일오버·기능 플래그 조치가 있으면 {ROLLBACK_REVERSIBLE}, 그 조치의 실행자가 활동 중인 "
        f"대응자로 이어지면 +{ROLLBACK_EXECUTOR_BONUS}",
        f"재시작·확장 등 다른 조치만 있으면 {MITIGATION_OTHER}, 실행자가 활동 중이면 "
        f"+{MITIGATION_OTHER_EXECUTOR_BONUS}(최대 {MITIGATION_OTHER + MITIGATION_OTHER_EXECUTOR_BONUS})",
        "완화 조치가 없으면 0. 여러 조치 중 가장 높은 점수를 쓴다",
    ),
}

# --------------------------------------------------------------------------- result types


@dataclass(frozen=True)
class ElementEval:
    """Evaluation of one required element for one service (score in 0..1)."""

    element: Element
    score: float
    status: ElementStatus
    reasons: tuple[str, ...] = ()
    subject_ids: tuple[str, ...] = ()
    evidence: tuple[SourceSpan, ...] = ()


@dataclass(frozen=True)
class TriggerRoute:
    """One verified trigger of a service and whom it pages."""

    trigger_id: str
    detect_edge_id: str
    notify_edge_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    responder_ids: tuple[str, ...]


@dataclass(frozen=True)
class DetectionAnalysis:
    service_id: str
    routes: tuple[TriggerRoute, ...]
    #: HOLD items on the detection path (DETECTED_BY edges, triggers, NOTIFIES edges)
    held_ids: tuple[str, ...]

    @property
    def notified(self) -> bool:
        return any(r.responder_ids for r in self.routes)


@dataclass(frozen=True)
class EscalationAnalysis:
    start_ids: tuple[str, ...]
    #: every usable ESCALATES_TO edge explored from the starts (incl. edges closing cycles)
    edge_ids: tuple[str, ...]
    #: nodes reached through ≥1 hop
    node_ids: tuple[str, ...]
    #: active people other than the escalating person reachable through the closure
    responder_ids: tuple[str, ...]
    #: True if a responder is reachable through hops that all state after_minutes
    timed: bool
    untimed_edge_ids: tuple[str, ...]
    #: reached nodes without any usable outgoing escalation
    dead_end_ids: tuple[str, ...]
    first_hop_ids: tuple[str, ...]
    held_ids: tuple[str, ...]

    @property
    def reaches_responder(self) -> bool:
        return bool(self.responder_ids)


@dataclass(frozen=True)
class OwnerAnalysis:
    service_id: str
    owner_edge_ids: tuple[str, ...]
    owner_ids: tuple[str, ...]
    #: all people the owner nodes resolve to (any status)
    people_ids: tuple[str, ...]
    active_ids: tuple[str, ...]
    reachable_ids: tuple[str, ...]
    unknown_ids: tuple[str, ...]
    escalation: EscalationAnalysis
    held_ids: tuple[str, ...]


class ApprovalGate(StrEnum):
    """Why an action needs sign-off under the single rule (:meth:`CheckContext.requires_approval`).

    Only the *reason* is classified here, so messages can explain it; whether approval is needed
    at all is always the context's decision.
    """

    STATED = "stated"  # the sources say so (requires_approval: true)
    DEFAULT = "default"  # rollback/failover and the sources are silent about approval


@dataclass(frozen=True)
class ActionApproval:
    action_id: str
    gate: ApprovalGate
    #: verified APPROVED_BY edges and the approvers they name (also those switched off by a what-if)
    edge_ids: tuple[str, ...]
    approver_ids: tuple[str, ...]
    #: distinct active & reachable people who can approve
    ready_ids: tuple[str, ...]
    unknown_ids: tuple[str, ...]
    score: float
    status: ElementStatus


@dataclass(frozen=True)
class ApprovalAnalysis:
    service_id: str
    mitigation_edge_ids: tuple[str, ...]
    action_ids: tuple[str, ...]
    approvals: tuple[ActionApproval, ...]
    held_ids: tuple[str, ...]


@dataclass(frozen=True)
class ChannelAnalysis:
    service_id: str
    #: active people considered (owner responders + first escalation hop)
    person_ids: tuple[str, ...]
    channel_ids: tuple[str, ...]
    media: tuple[str, ...]
    #: channels that do not share the service's failure domain
    independent_ids: tuple[str, ...]
    held_ids: tuple[str, ...]


@dataclass(frozen=True)
class ActionReadiness:
    action_id: str
    action_type: ActionType
    reversible: bool
    executor_edge_ids: tuple[str, ...]
    executor_ids: tuple[str, ...]
    active_executor_ids: tuple[str, ...]
    score: float


@dataclass(frozen=True)
class RollbackAnalysis:
    service_id: str
    mitigation_edge_ids: tuple[str, ...]
    actions: tuple[ActionReadiness, ...]
    held_ids: tuple[str, ...]

    @property
    def reversible(self) -> tuple[ActionReadiness, ...]:
        return tuple(a for a in self.actions if a.reversible)


# --------------------------------------------------------------------------- traversal


def usable(ctx: CheckContext, item: Node | Edge) -> bool:
    """Verified and not switched off by a what-if scenario — i.e. admissible as evidence."""
    return item.verification is Verification.VERIFIED and item.id not in ctx.unavailable


class _Scope:
    """Traversal helper for one analysis: admits only usable items and remembers HOLD ones.

    Remembering held items lets reasons say *why* something was not counted instead of silently
    pretending it does not exist.
    """

    def __init__(self, ctx: CheckContext) -> None:
        self.ctx = ctx
        self.held: set[str] = set()

    def admit(self, item: Node | Edge) -> bool:
        if item.id in self.ctx.unavailable:
            return False
        if item.verification is Verification.HOLD:
            self.held.add(item.id)
            return False
        return True

    def out(self, nid: str, kind: EdgeKind) -> list[Edge]:
        nodes = self.ctx.graph.nodes
        return [e for e in self.ctx.graph.out_edges(nid, kind) if self.admit(e) and self.admit(nodes[e.dst])]

    def people(self, nid: str) -> list[Node]:
        graph = self.ctx.graph
        node = graph.nodes.get(nid)
        if node is None or not self.admit(node):
            return []
        if node.kind is NodeKind.PERSON:
            return [node]
        link = {NodeKind.ROLE: EdgeKind.HOLDS_ROLE, NodeKind.TEAM: EdgeKind.MEMBER_OF}.get(node.kind)
        if link is None:
            return []
        return [
            graph.nodes[e.src]
            for e in graph.in_edges(nid, link)
            if self.admit(e) and self.admit(graph.nodes[e.src])
        ]

    def active_people(self, nid: str) -> list[Node]:
        return [p for p in self.people(nid) if self.ctx.is_active(p)]

    def channels(self, nid: str) -> list[Node]:
        """Own REACHABLE_VIA channels of ``nid`` plus their FALLBACK_TO closure."""
        nodes = self.ctx.graph.nodes
        seen: dict[str, Node] = {}
        queue = deque(e.dst for e in self.out(nid, EdgeKind.REACHABLE_VIA))
        while queue:
            cid = queue.popleft()
            if cid in seen:
                continue
            seen[cid] = nodes[cid]
            queue.extend(e.dst for e in self.out(cid, EdgeKind.FALLBACK_TO))
        return [seen[k] for k in sorted(seen)]

    def person_channels(self, person: Node, vias: Iterable[str]) -> list[Node]:
        """Channels of a person, plus those of *every* role/team they were reached through.

        All routes count: someone who owns a service both directly and through the on-call role
        is reachable on the role's pager too. Keeping only the first route would make adding a
        second, valid owner edge lower the score (monotonicity).
        """
        found = {c.id: c for c in self.channels(person.id)}
        for via in sorted(set(vias) - {person.id}):
            found.update({c.id: c for c in self.channels(via)})
        return [found[k] for k in sorted(found)]


def reachable_channels(ctx: CheckContext, person_id: str, via_id: str | None = None) -> list[Node]:
    """Usable contact channels of a person (optionally reached through a role/team)."""
    person = ctx.graph.nodes.get(person_id)
    if person is None:
        return []
    return _Scope(ctx).person_channels(person, (via_id,) if via_id else ())


def confirmed_view(ctx: CheckContext) -> CheckContext:
    """The same context with every HOLD (판단 보류) node/edge treated as confirmed.

    Why: a rule may only call a gap *broken* if it stays broken even when every held item turns
    out to be right. If confirming the held items would close the gap, the honest verdict is
    "확인 불가" (UNKNOWN, P3) — not "적혀 있지 않음". Scores never use this view; held items are
    still not evidence. Returns ``ctx`` itself when nothing is held.

    The graph is built with the constructor, never ``graph.model_copy``: a model copy carries over
    the lazily built adjacency index, so once any earlier rule had queried the graph the "confirmed"
    view would still hand out the original HOLD edges and every 확인 필요 would silently turn into
    a settled 끊김 depending on rule order.
    """
    graph = ctx.graph
    nodes = {
        k: n.model_copy(update={"verification": Verification.VERIFIED, "hold_reason": None})
        for k, n in graph.nodes.items()
        if n.verification is Verification.HOLD
    }
    edges = {
        k: e.model_copy(update={"verification": Verification.VERIFIED, "hold_reason": None})
        for k, e in graph.edges.items()
        if e.verification is Verification.HOLD
    }
    if not nodes and not edges:
        return ctx
    confirmed = OpsGraph(
        nodes={**graph.nodes, **nodes},
        edges={**graph.edges, **edges},
        docs=dict(graph.docs),
    )
    return ctx.with_graph(confirmed)


def active_responders(ctx: CheckContext, nid: str) -> list[Node]:
    """Active people a person/role/team resolves to, using verified links only."""
    return _Scope(ctx).active_people(nid)


def action_type(action: Node) -> ActionType:
    """Stated action type; an absent or unrecognised type is ``OTHER`` (absent stays absent)."""
    raw = action.attr(AttrKey.ACTION_TYPE)
    try:
        return ActionType(raw) if isinstance(raw, str) else ActionType.OTHER
    except ValueError:
        return ActionType.OTHER


def needs_approval(action: Node) -> bool:
    """Whether ``action`` waits for an approver: the single rule of :mod:`jettae.check.context`."""
    return requires_approval(action)


def _stated_approver_edges(ctx: CheckContext, action_id: str) -> list[Edge]:
    """APPROVED_BY edges the sources state (verified edge and approver), available or not.

    Availability is deliberately ignored: a what-if that switches the approver off must leave the
    *requirement* in place (and fail it), not make the step look as if it needed nobody.
    """
    nodes = ctx.graph.nodes
    return [
        e
        for e in ctx.graph.out_edges(action_id, EdgeKind.APPROVED_BY)
        if e.verification is Verification.VERIFIED and nodes[e.dst].verification is Verification.VERIFIED
    ]


def approval_gate(ctx: CheckContext, action: Node) -> ApprovalGate | None:
    """Why ``action`` needs sign-off, or ``None`` when :meth:`CheckContext.requires_approval` says no.

    The decision itself is the context's (shared with the SPOF analysis and the rehearsal
    simulator), so the score, JT-PATH-020 and the replay never disagree on which steps wait for
    an approver.
    """
    if not ctx.requires_approval(action):
        return None
    return ApprovalGate.STATED if action.attr(AttrKey.REQUIRES_APPROVAL) is True else ApprovalGate.DEFAULT


def channel_medium(channel: Node) -> ChannelMedium | None:
    raw = channel.attr(AttrKey.MEDIUM)
    try:
        return ChannelMedium(raw) if isinstance(raw, str) else None
    except ValueError:
        return None


def states_after_minutes(edge: Edge) -> bool:
    value = edge.attr(AttrKey.AFTER_MINUTES)
    return isinstance(value, int | float) and not isinstance(value, bool)


def spans_of(ctx: CheckContext, ids: Iterable[str]) -> tuple[SourceSpan, ...]:
    """Source spans of the given nodes/edges, deduplicated and in document order."""
    found: set[SourceSpan] = set()
    for item_id in ids:
        item: Node | Edge | None = ctx.graph.nodes.get(item_id) or ctx.graph.edges.get(item_id)
        if item is not None:
            found.update(item.spans)
    return tuple(sorted(found, key=lambda s: (s.doc_id, s.start, s.end, s.quote)))


# --------------------------------------------------------------------------- Korean wording


#: Final consonant of the Sino-Korean reading of a trailing digit: None = no 받침, "ㄹ" matters for
#: 으로/로. A trailing 0 is read 영·십·백·천·만 — always with a (non-ㄹ) 받침.
_DIGIT_FINAL: dict[str, str | None] = {
    "0": "ㅇ",  # 영/십/백/천/만
    "1": "ㄹ",  # 일
    "2": None,  # 이
    "3": "ㅁ",  # 삼
    "4": None,  # 사
    "5": None,  # 오
    "6": "ㄱ",  # 육
    "7": "ㄹ",  # 칠
    "8": "ㄹ",  # 팔
    "9": None,  # 구
}
_CLOSERS = " '\"’”]」』"


def _particle_anchor(word: str) -> str:
    """The character a particle must agree with.

    Closing quotes are skipped, and so is a trailing parenthetical: Korean attaches the particle
    to the word *before* the brackets ("'박과장'(장기 부재)은", not "…(장기 부재)는"), which matters
    because person labels carry their roster status in brackets.
    """
    core = unicodedata.normalize("NFC", word).rstrip(_CLOSERS)
    while core.endswith(")"):
        opening = core.rfind("(")
        if opening <= 0:
            core = core.rstrip(")" + _CLOSERS)
            break
        core = core[:opening].rstrip(_CLOSERS)
    return core[-1:]


def josa(word: str, with_final: str, without_final: str) -> str:
    """Attach the particle that agrees with the last syllable (이/가, 은/는, 을/를, 과/와, 으로/로).

    Hangul syllables and Arabic digits (read in Sino-Korean: 1 일, 2 이 …) decide the form;
    other endings (Latin letters, symbols) get the conventional "이(가)" form because the reading
    is unknown. Text is NFC-normalised first so decomposed Hangul (NFD, e.g. names pasted from
    macOS) still agrees.
    """
    last = _particle_anchor(word)
    final: str | None
    if last and "가" <= last <= "힣":
        jong = (ord(last) - 0xAC00) % 28
        final = None if jong == 0 else "ㄹ" if jong == 8 else "other"
    elif last in _DIGIT_FINAL:
        final = _DIGIT_FINAL[last]
    elif with_final == "으로":
        return f"{word}(으)로"
    else:
        return f"{word}{with_final}({without_final})"
    if with_final == "으로" and final == "ㄹ":  # ㄹ받침 + 로
        return word + without_final
    return word + (with_final if final else without_final)


def label(ctx: CheckContext, nid: str) -> str:
    """Quoted display name on one line (runbook names may carry line breaks or tabs)."""
    node = ctx.graph.nodes.get(nid)
    name = node.name if node is not None else nid
    return f"'{' '.join(name.split()) or nid}'"


def labels(ctx: CheckContext, ids: Iterable[str]) -> str:
    return ", ".join(label(ctx, i) for i in ids)


def person_label(ctx: CheckContext, pid: str) -> str:
    """``'김대리'(퇴사)`` — name plus roster status, so readers see why a person does not count."""
    node = ctx.graph.nodes.get(pid)
    if node is None or node.kind is not NodeKind.PERSON:
        return label(ctx, pid)
    return f"{label(ctx, pid)}({PERSON_STATUS_KO[ctx.person_status(node)]})"


def service_label(ctx: CheckContext, sid: str) -> str:
    """``'결제' 서비스`` — but ``'결제 서비스'`` stays as is instead of ``'결제 서비스' 서비스``."""
    quoted = label(ctx, sid)
    return quoted if quoted.rstrip("'").endswith("서비스") else f"{quoted} 서비스"


# --------------------------------------------------------------------------- analyses


def _targets(edges: Iterable[Edge]) -> tuple[str, ...]:
    """Distinct edge targets in edge order.

    Parallel edges (``#disc`` ids, e.g. one per channel) must not list the same person twice in a
    reason or a finding message.
    """
    return tuple(dict.fromkeys(e.dst for e in edges))


def _held_ids(items: Iterable[Node | Edge]) -> list[str]:
    return [i.id for i in items if i.verification is Verification.HOLD]


def _channel_down(ctx: CheckContext, notify: Edge) -> bool:
    channel = notify.attr(AttrKey.CHANNEL)
    return isinstance(channel, str) and channel in ctx.unavailable


def analyze_detection(ctx: CheckContext, sid: str) -> DetectionAnalysis:
    graph = ctx.graph
    held: list[str] = []
    routes: list[TriggerRoute] = []
    for detect in graph.out_edges(sid, EdgeKind.DETECTED_BY):
        trigger = graph.nodes[detect.dst]
        if detect.id in ctx.unavailable or trigger.id in ctx.unavailable:
            continue
        notifies = [e for e in graph.out_edges(trigger.id, EdgeKind.NOTIFIES) if e.id not in ctx.unavailable]
        held.extend(_held_ids((detect, trigger, *notifies)))
        if not (usable(ctx, detect) and usable(ctx, trigger)):
            continue
        scope = _Scope(ctx)
        edge_ids: list[str] = []
        target_ids: list[str] = []
        responders: set[str] = set()
        for notify in notifies:
            target = graph.nodes[notify.dst]
            if notify.verification is Verification.HOLD or not scope.admit(target):
                continue
            edge_ids.append(notify.id)
            target_ids.append(target.id)
            if not _channel_down(ctx, notify):
                responders.update(p.id for p in scope.active_people(target.id))
        routes.append(
            TriggerRoute(
                trigger_id=trigger.id,
                detect_edge_id=detect.id,
                notify_edge_ids=tuple(edge_ids),
                target_ids=tuple(dict.fromkeys(target_ids)),
                responder_ids=tuple(sorted(responders)),
            )
        )
    return DetectionAnalysis(service_id=sid, routes=tuple(routes), held_ids=tuple(sorted(set(held))))


def _direct_people(scope: _Scope, nid: str) -> set[str]:
    """People an escalation *from* ``nid`` must get past: the person, or the role's holders.

    A team is a pool, so escalating from a team to its own lead still reaches someone new.
    """
    node = scope.ctx.graph.nodes.get(nid)
    if node is None or node.kind is NodeKind.TEAM:
        return set()
    return {p.id for p in scope.people(nid)}


def _escalation_walk(
    scope: _Scope, start: str, *, timed_only: bool, edges: dict[str, Edge], reached: set[str]
) -> set[str]:
    """BFS over ESCALATES_TO from ``start``; returns active people met that are not the escalator."""
    excluded = _direct_people(scope, start)
    seen = {start}
    found: set[str] = set()
    queue = deque([(start, 0)])
    while queue:
        cur, hops = queue.popleft()
        if hops >= MAX_ESCALATION_HOPS:
            continue
        for e in scope.out(cur, EdgeKind.ESCALATES_TO):
            if timed_only and not states_after_minutes(e):
                continue
            edges[e.id] = e
            if e.dst in seen:
                continue
            seen.add(e.dst)
            reached.add(e.dst)
            found.update(p.id for p in scope.active_people(e.dst) if p.id not in excluded)
            queue.append((e.dst, hops + 1))
    return found


def analyze_escalation(ctx: CheckContext, start_ids: Iterable[str]) -> EscalationAnalysis:
    scope = _Scope(ctx)
    starts = tuple(sorted(set(start_ids)))
    edges: dict[str, Edge] = {}
    reached: set[str] = set()
    responders: set[str] = set()
    timed = False
    for start in starts:
        responders |= _escalation_walk(scope, start, timed_only=False, edges=edges, reached=reached)
        timed = timed or bool(_escalation_walk(scope, start, timed_only=True, edges={}, reached=set()))
    first_hop = {e.dst for s in starts for e in scope.out(s, EdgeKind.ESCALATES_TO)}
    dead_ends = [n for n in sorted(reached) if not scope.out(n, EdgeKind.ESCALATES_TO)]
    return EscalationAnalysis(
        start_ids=starts,
        edge_ids=tuple(sorted(edges)),
        node_ids=tuple(sorted(reached)),
        responder_ids=tuple(sorted(responders)),
        timed=timed,
        untimed_edge_ids=tuple(sorted(eid for eid, e in edges.items() if not states_after_minutes(e))),
        dead_end_ids=tuple(dead_ends),
        first_hop_ids=tuple(sorted(first_hop)),
        held_ids=tuple(sorted(scope.held)),
    )


def _resolve_all(
    scope: _Scope, node_ids: Iterable[str], *, active_only: bool
) -> dict[str, tuple[Node, list[str]]]:
    """person id -> (person, every person/role/team node it was resolved from)."""
    resolved: dict[str, tuple[Node, list[str]]] = {}
    for nid in node_ids:
        people = scope.active_people(nid) if active_only else scope.people(nid)
        for person in people:
            resolved.setdefault(person.id, (person, []))[1].append(nid)
    return resolved


def analyze_owner(ctx: CheckContext, sid: str) -> OwnerAnalysis:
    scope = _Scope(ctx)
    owner_edges = scope.out(sid, EdgeKind.OWNED_BY)
    owner_ids = _targets(owner_edges)
    resolved = _resolve_all(scope, owner_ids, active_only=False)
    ordered = sorted(resolved)
    active = [pid for pid in ordered if ctx.is_active(resolved[pid][0])]
    reachable = [pid for pid in active if scope.person_channels(*resolved[pid])]
    unknown = [pid for pid in ordered if ctx.person_status(resolved[pid][0]) is PersonStatus.UNKNOWN]
    escalation = analyze_escalation(ctx, (*owner_ids, *ordered))
    return OwnerAnalysis(
        service_id=sid,
        owner_edge_ids=tuple(e.id for e in owner_edges),
        owner_ids=owner_ids,
        people_ids=tuple(ordered),
        active_ids=tuple(active),
        reachable_ids=tuple(reachable),
        unknown_ids=tuple(unknown),
        escalation=escalation,
        held_ids=tuple(sorted(scope.held)),
    )


def _approval_of(ctx: CheckContext, scope: _Scope, action: Node, gate: ApprovalGate) -> ActionApproval:
    usable = scope.out(action.id, EdgeKind.APPROVED_BY)  # also remembers held links
    stated = _stated_approver_edges(ctx, action.id)
    ready: set[str] = set()
    unknown: set[str] = set()
    for edge in usable:
        for person in scope.people(edge.dst):
            if ctx.is_active(person) and scope.person_channels(person, (edge.dst,)):
                ready.add(person.id)
            elif ctx.person_status(person) is PersonStatus.UNKNOWN:
                unknown.add(person.id)
    if len(ready) >= 2:
        score, status = APPROVER_TWO_OR_MORE, ElementStatus.PASS
    elif ready:
        score, status = APPROVER_SINGLE, ElementStatus.PARTIAL
    elif unknown:
        score, status = 0.0, ElementStatus.UNKNOWN
    elif stated:
        score, status = APPROVER_NOT_READY, ElementStatus.FAIL
    else:
        score, status = 0.0, ElementStatus.FAIL
    return ActionApproval(
        action_id=action.id,
        gate=gate,
        edge_ids=tuple(e.id for e in stated),
        approver_ids=_targets(stated),
        ready_ids=tuple(sorted(ready)),
        unknown_ids=tuple(sorted(unknown)),
        score=score,
        status=status,
    )


def analyze_approval(ctx: CheckContext, sid: str) -> ApprovalAnalysis:
    scope = _Scope(ctx)
    mitigations = scope.out(sid, EdgeKind.MITIGATED_BY)
    actions = [ctx.graph.nodes[e.dst] for e in mitigations]
    approvals: list[ActionApproval] = []
    for action in actions:
        gate = approval_gate(ctx, action)
        if gate is not None:
            approvals.append(_approval_of(ctx, scope, action, gate))
    return ApprovalAnalysis(
        service_id=sid,
        mitigation_edge_ids=tuple(e.id for e in mitigations),
        action_ids=tuple(a.id for a in actions),
        approvals=tuple(approvals),
        held_ids=tuple(sorted(scope.held)),
    )


def analyze_channels(ctx: CheckContext, sid: str, *, owner: OwnerAnalysis | None = None) -> ChannelAnalysis:
    owner = owner or analyze_owner(ctx, sid)
    scope = _Scope(ctx)
    pool = _resolve_all(scope, (*owner.owner_ids, *owner.escalation.first_hop_ids), active_only=True)
    channels: dict[str, Node] = {}
    for pid in sorted(pool):
        channels.update({c.id: c for c in scope.person_channels(*pool[pid])})
    ordered = sorted(channels)
    media = {m.value for m in (channel_medium(channels[c]) for c in ordered) if m is not None}
    independent = [c for c in ordered if not ctx.shares_failure_domain(c, sid)]
    return ChannelAnalysis(
        service_id=sid,
        person_ids=tuple(sorted(pool)),
        channel_ids=tuple(ordered),
        media=tuple(sorted(media)),
        independent_ids=tuple(independent),
        held_ids=tuple(sorted(scope.held)),
    )


def analyze_rollback(ctx: CheckContext, sid: str) -> RollbackAnalysis:
    scope = _Scope(ctx)
    mitigations = scope.out(sid, EdgeKind.MITIGATED_BY)
    actions: list[ActionReadiness] = []
    for edge in mitigations:
        action = ctx.graph.nodes[edge.dst]
        kind = action_type(action)
        reversible = kind in REVERSIBLE_ACTION_TYPES
        executors = scope.out(action.id, EdgeKind.EXECUTED_BY)
        active = sorted({p.id for e in executors for p in scope.active_people(e.dst)})
        base, bonus = (
            (ROLLBACK_REVERSIBLE, ROLLBACK_EXECUTOR_BONUS)
            if reversible
            else (MITIGATION_OTHER, MITIGATION_OTHER_EXECUTOR_BONUS)
        )
        actions.append(
            ActionReadiness(
                action_id=action.id,
                action_type=kind,
                reversible=reversible,
                executor_edge_ids=tuple(e.id for e in executors),
                executor_ids=_targets(executors),
                active_executor_ids=tuple(active),
                score=round(base + (bonus if active else 0.0), 6),
            )
        )
    return RollbackAnalysis(
        service_id=sid,
        mitigation_edge_ids=tuple(e.id for e in mitigations),
        actions=tuple(actions),
        held_ids=tuple(sorted(scope.held)),
    )


# --------------------------------------------------------------------------- element scoring


def _status_for(score: float) -> ElementStatus:
    if score >= 1.0:
        return ElementStatus.PASS
    return ElementStatus.PARTIAL if score > 0 else ElementStatus.FAIL


def _held_note(held: Iterable[str]) -> tuple[str, ...]:
    ids = sorted(set(held))
    if not ids:
        return ()
    return (f"판단 보류(HOLD) 요소 {len(ids)}건은 근거로 쓰지 않음: {', '.join(ids)}",)


def _eval(
    ctx: CheckContext,
    element: Element,
    score: float,
    status: ElementStatus,
    *,
    reasons: Iterable[str],
    subjects: Iterable[str],
) -> ElementEval:
    subject_ids = tuple(dict.fromkeys(subjects))
    if status is ElementStatus.UNKNOWN:
        score = 0.0
    return ElementEval(
        element=element,
        score=round(min(max(score, 0.0), 1.0), 6),
        status=status,
        reasons=tuple(reasons),
        subject_ids=subject_ids,
        evidence=spans_of(ctx, subject_ids),
    )


def eval_detection(ctx: CheckContext, det: DetectionAnalysis) -> ElementEval:
    sid = det.service_id
    subjects = [sid]
    for r in det.routes:
        subjects += [r.detect_edge_id, r.trigger_id, *r.notify_edge_ids, *r.target_ids]
    if det.held_ids:
        reason = f"감지 경로에 판단 보류(HOLD) 요소가 있어 확인할 수 없음: {', '.join(det.held_ids)}"
        return _eval(
            ctx,
            Element.DETECTION,
            0.0,
            ElementStatus.UNKNOWN,
            reasons=[reason],
            subjects=[*subjects, *det.held_ids],
        )
    if det.notified:
        best = next(r for r in det.routes if r.responder_ids)
        reason = (
            f"감지 신호 {josa(label(ctx, best.trigger_id), '이', '가')} 활동 중인 대응자 "
            f"{labels(ctx, best.responder_ids)}에게 호출을 보냄"
        )
        return _eval(
            ctx,
            Element.DETECTION,
            DETECTION_NOTIFIED,
            ElementStatus.PASS,
            reasons=[reason],
            subjects=subjects,
        )
    if det.routes:
        triggers = labels(ctx, dict.fromkeys(r.trigger_id for r in det.routes))
        reason = (
            f"감지 신호 {josa(triggers, '은', '는')} 있으나 호출 대상이 없거나 활동 중인 대응자로 이어지지 "
            "않음"
        )
        return _eval(
            ctx,
            Element.DETECTION,
            DETECTION_TRIGGER_ONLY,
            ElementStatus.PARTIAL,
            reasons=[reason],
            subjects=subjects,
        )
    reason = "이 서비스의 장애를 알아챌 감지 신호(알람·모니터·고객 문의 경로)가 없음"
    return _eval(ctx, Element.DETECTION, 0.0, ElementStatus.FAIL, reasons=[reason], subjects=subjects)


def _owner_part_reason(ctx: CheckContext, own: OwnerAnalysis) -> str:
    if not own.owner_ids:
        return "담당자(OWNED_BY)가 지정되어 있지 않음"
    if own.reachable_ids:
        return f"활동 중인 담당자 {labels(ctx, own.reachable_ids)}에게 연락할 수 있음"
    if own.active_ids:
        return f"활동 중인 담당자 {labels(ctx, own.active_ids)}의 연락 수단이 등록되어 있지 않음"
    if own.people_ids:
        people = ", ".join(person_label(ctx, p) for p in own.people_ids)
        if len(own.people_ids) == 1:
            return f"담당자 {josa(people, '은', '는')} 활동 중이 아님"
        return f"담당자 {people} 중 활동 중인 사람이 없음"
    return (
        f"담당 {josa(labels(ctx, own.owner_ids), '이', '가')} 실제 사람으로 이어지지 않음(맡은 사람이 없음)"
    )


def _escalation_reason(ctx: CheckContext, esc: EscalationAnalysis) -> str:
    if esc.timed:
        return f"상향 경로가 {labels(ctx, esc.responder_ids)}에게 닿고 모든 단계에 대기 시간이 적혀 있음"
    if esc.reaches_responder:
        return (
            f"상향 경로가 {labels(ctx, esc.responder_ids)}에게 닿지만 대기 시간(몇 분 뒤 상향하는지)이 빠진 "
            "단계가 있음"
        )
    if esc.edge_ids:
        return "상향 경로가 있으나 담당자 외에 활동 중인 다른 대응자에게 닿지 않음(막다른 길)"
    return "담당자에게서 이어지는 상향(에스컬레이션) 경로가 없음"


def eval_owner(ctx: CheckContext, own: OwnerAnalysis) -> ElementEval:
    esc = own.escalation
    owner_part = OWNER_REACHABLE if own.reachable_ids else OWNER_ACTIVE_UNREACHABLE if own.active_ids else 0.0
    esc_part = ESCALATION_TIMED if esc.timed else ESCALATION_UNTIMED if esc.reaches_responder else 0.0
    total = round(owner_part + esc_part, 6)
    if not own.active_ids and own.unknown_ids:
        status = ElementStatus.UNKNOWN
    elif owner_part == 0.0:
        status = ElementStatus.FAIL
    else:
        status = _status_for(total)
    reasons = [_owner_part_reason(ctx, own), _escalation_reason(ctx, esc)]
    if status is ElementStatus.UNKNOWN:
        unknown = labels(ctx, own.unknown_ids)
        reasons.insert(
            0, f"담당자 {josa(unknown, '이', '가')} 명부에서 확인되지 않아 담당 여부를 확인할 수 없음"
        )
    reasons += _held_note((*own.held_ids, *esc.held_ids))
    subjects = [
        own.service_id,
        *own.owner_edge_ids,
        *own.owner_ids,
        *own.people_ids,
        *esc.edge_ids,
        *esc.responder_ids,
    ]
    return _eval(ctx, Element.OWNER, total, status, reasons=reasons, subjects=subjects)


def eval_approver(ctx: CheckContext, appr: ApprovalAnalysis) -> ElementEval:
    subjects = [appr.service_id, *appr.mitigation_edge_ids, *appr.action_ids]
    for a in appr.approvals:
        subjects += [*a.edge_ids, *a.approver_ids]
    held = _held_note(appr.held_ids)
    if not appr.action_ids:
        reasons = ["조치가 없어 승인 경로를 평가할 수 없음", *held]
        return _eval(ctx, Element.APPROVER, 0.0, ElementStatus.FAIL, reasons=reasons, subjects=subjects)
    if not appr.approvals:
        reasons = [
            "승인이 필요한 조치가 없음(롤백·페일오버도, 승인 필요로 적힌 조치도 없거나 승인 불필요로 명시)",
            *held,
        ]
        return _eval(ctx, Element.APPROVER, 1.0, ElementStatus.PASS, reasons=reasons, subjects=subjects)
    reasons = [_approval_reason(ctx, a) for a in appr.approvals]
    worst = min(appr.approvals, key=lambda a: (a.score, a.action_id))
    if any(a.status is ElementStatus.UNKNOWN for a in appr.approvals):
        status = ElementStatus.UNKNOWN
    elif any(a.status is ElementStatus.FAIL for a in appr.approvals):
        status = ElementStatus.FAIL
    else:
        status = _status_for(worst.score)
    return _eval(ctx, Element.APPROVER, worst.score, status, reasons=[*reasons, *held], subjects=subjects)


def _approval_reason(ctx: CheckContext, a: ActionApproval) -> str:
    action = label(ctx, a.action_id)
    if len(a.ready_ids) >= 2:
        return f"조치 {action}: 승인 가능한 사람 {len(a.ready_ids)}명({labels(ctx, a.ready_ids)})"
    if a.ready_ids:
        return f"조치 {action}: 승인 가능한 사람이 {labels(ctx, a.ready_ids)} 1명뿐(부재 시 대행 없음)"
    if a.status is ElementStatus.UNKNOWN:
        return f"조치 {action}: 승인자 {josa(labels(ctx, a.unknown_ids), '이', '가')} 명부에서 확인되지 않음"
    if a.approver_ids:
        return f"조치 {action}: 승인자 {labels(ctx, a.approver_ids)} 중 활동 중이고 연락 가능한 사람이 없음"
    return f"조치 {action}: 승인자가 적혀 있지 않음"


def eval_fallback(ctx: CheckContext, ch: ChannelAnalysis) -> ElementEval:
    count = len(ch.channel_ids)
    if count >= 2 and len(ch.media) >= 2 and ch.independent_ids:
        score = FALLBACK_INDEPENDENT
        reason = (
            f"연락 수단 {count}개({labels(ctx, ch.channel_ids)}) 중 서비스와 함께 죽지 않는 "
            f"{josa(labels(ctx, ch.independent_ids), '이', '가')} 있음"
        )
    elif count >= 2:
        score = FALLBACK_COMMON_MODE
        # channels without a stated medium cannot prove diversity, so do not claim "same medium"
        unstated = [c for c in ch.channel_ids if channel_medium(ctx.graph.nodes[c]) is None]
        if len(ch.media) >= 2:
            why = "모두 서비스와 같은 장애 영역에 있음"
        elif unstated:
            why = (
                f"{labels(ctx, unstated)}의 매체(전화·메신저 등)가 적혀 있지 않아 서로 다른 매체인지 "
                "확인할 수 없음"
            )
        else:
            why = "모두 같은 매체"
        reason = f"연락 수단 {count}개({labels(ctx, ch.channel_ids)})가 있으나 {why}"
    elif count == 1:
        score = FALLBACK_SINGLE
        reason = f"연락 수단이 {label(ctx, ch.channel_ids[0])} 하나뿐이라 끊기면 대체할 채널이 없음"
    elif ch.person_ids:
        score = 0.0
        reason = f"대응자 {labels(ctx, ch.person_ids)}에게 등록된 연락 수단이 없음"
    else:
        score = 0.0
        reason = "연락할 활동 중인 담당자·1차 상향 대상이 없어 채널을 확인할 수 없음"
    status = ElementStatus.FAIL if score <= FALLBACK_SINGLE else _status_for(score)
    subjects = [ch.service_id, *ch.person_ids, *ch.channel_ids]
    return _eval(
        ctx,
        Element.FALLBACK_CHANNEL,
        score,
        status,
        reasons=[reason, *_held_note(ch.held_ids)],
        subjects=subjects,
    )


def eval_rollback(ctx: CheckContext, rb: RollbackAnalysis) -> ElementEval:
    subjects = [rb.service_id, *rb.mitigation_edge_ids]
    for a in rb.actions:
        subjects += [a.action_id, *a.executor_edge_ids, *a.executor_ids]
    held = _held_note(rb.held_ids)
    if not rb.actions:
        reasons = ["롤백·페일오버 등 완화 조치가 적혀 있지 않음", *held]
        return _eval(ctx, Element.ROLLBACK, 0.0, ElementStatus.FAIL, reasons=reasons, subjects=subjects)
    best = max(rb.actions, key=lambda a: (a.score, a.reversible, a.action_id))
    action = label(ctx, best.action_id)
    if best.reversible and best.active_executor_ids:
        executors = josa(labels(ctx, best.active_executor_ids), "이", "가")
        reason = f"되돌리기 조치 {action}의 실행자 {executors} 활동 중임"
    elif best.reversible:
        reason = f"되돌리기 조치 {josa(action, '이', '가')} 있으나 활동 중인 실행자가 지정되어 있지 않음"
    else:
        reason = f"재시작·확장 같은 조치({action})만 있고 롤백·페일오버·기능 플래그 같은 되돌리기 조치가 없음"
    return _eval(
        ctx, Element.ROLLBACK, best.score, _status_for(best.score), reasons=[reason, *held], subjects=subjects
    )


def evaluate_service(ctx: CheckContext, sid: str) -> dict[Element, ElementEval]:
    """Evaluate the five required elements of one service (ordered as :class:`Element`)."""
    owner = analyze_owner(ctx, sid)
    return {
        Element.DETECTION: eval_detection(ctx, analyze_detection(ctx, sid)),
        Element.OWNER: eval_owner(ctx, owner),
        Element.APPROVER: eval_approver(ctx, analyze_approval(ctx, sid)),
        Element.FALLBACK_CHANNEL: eval_fallback(ctx, analyze_channels(ctx, sid, owner=owner)),
        Element.ROLLBACK: eval_rollback(ctx, analyze_rollback(ctx, sid)),
    }


# --------------------------------------------------------------------------- remediation hints

_LEADER_KEYWORDS: tuple[str, ...] = (
    "팀장",
    "파트장",
    "실장",
    "본부장",
    "리드",
    "lead",
    "head",
    "manager",
    "매니저",
)


def _roster_index(ctx: CheckContext) -> dict[str, RosterEntry]:
    """Name/alias → roster entry, first entry in roster order winning (as ``ctx.roster_entry``).

    Built once per ranking so suggesting candidates is O(people + roster) rather than
    O(people × roster) — the people rules ask for suggestions on every duty they report.
    """
    index: dict[str, RosterEntry] = {}
    for entry in ctx.roster:
        for key in (entry.name, *entry.aliases):
            index.setdefault(key, entry)
    return index


def _person_labels(ctx: CheckContext, person: Node, entry: RosterEntry | None) -> list[str]:
    found: list[str] = []
    title = person.attr(AttrKey.TITLE)
    if isinstance(title, str):
        found.append(title)
    found += [r.name for r in ctx.graph.successors(person.id, EdgeKind.HOLDS_ROLE)]
    if entry is not None:
        found += [*([entry.title] if entry.title else []), *entry.roles]
    return [x.casefold() for x in found]


def _team_keys(ctx: CheckContext, person: Node, entry: RosterEntry | None) -> set[str]:
    keys = {t.id for t in ctx.graph.successors(person.id, EdgeKind.MEMBER_OF)}
    if entry is not None and entry.team:
        keys.add(node_id(NodeKind.TEAM, entry.team))
    return keys


def _terminal_names(ctx: CheckContext) -> frozenset[str]:
    return frozenset(r.casefold() for r in ctx.policy.terminal_roles)


def is_terminal(ctx: CheckContext, node: Node) -> bool:
    """A node that ends an escalation chain by policy (CTO·대표·CISO …)."""
    terminal = _terminal_names(ctx)
    if node.kind is NodeKind.ROLE:
        return node.name.casefold() in terminal
    if node.kind is NodeKind.PERSON:
        return bool(terminal & set(_person_labels(ctx, node, ctx.roster_entry(node))))
    return False


def suggest_candidates(
    ctx: CheckContext, anchor_ids: Iterable[str], *, exclude: Iterable[str] = (), limit: int = 5
) -> list[str]:
    """Roster-derived replacement/escalation candidates around ``anchor_ids`` (person node ids).

    Ranking (deterministic): team leads of the anchors' teams, then holders of policy terminal
    roles, then other active teammates; reachable people first within a rank, then by id. Only
    people already in the graph are suggested, so a fix can never introduce someone outside the
    roster.
    """
    graph = ctx.graph
    roster = _roster_index(ctx)
    terminal = _terminal_names(ctx)
    anchors = sorted(set(anchor_ids))
    anchor_people = {p.id for a in anchors for p in ctx.resolve_people(a)}
    teams = {a for a in anchors if a in graph.nodes and graph.nodes[a].kind is NodeKind.TEAM}
    for pid in sorted(anchor_people):
        anchor = graph.nodes[pid]
        teams |= _team_keys(ctx, anchor, roster.get(anchor.name))
    excluded = set(exclude) | anchor_people
    ranked: list[tuple[int, int, str]] = []
    for person in graph.nodes_of(NodeKind.PERSON):
        if person.id in excluded or not usable(ctx, person) or not ctx.is_active(person):
            continue
        entry = roster.get(person.name)
        person_labels = _person_labels(ctx, person, entry)
        same_team = bool(_team_keys(ctx, person, entry) & teams)
        leader = any(k in lbl for lbl in person_labels for k in _LEADER_KEYWORDS)
        if same_team and leader:
            rank = 0
        elif terminal & set(person_labels):
            rank = 1
        elif same_team:
            rank = 2
        else:
            continue
        unreachable = 0 if reachable_channels(ctx, person.id) else 1
        ranked.append((rank, unreachable, person.id))
    return [pid for _, _, pid in sorted(ranked)[:limit]]
