"""Deterministic mock-incident replay (모의 장애 결정적 리플레이) — the rehearsal judge.

Why a replay and not a model: a rehearsal verdict is a *judgement* (P1). It must be
reproducible bit-for-bit (P6), work with the LLM switched off (P4) and never touch real
systems. So this module walks the operations graph as a tiny discrete-event simulation: a
priority queue of "page this node at minute t" hops, processed in (time, insertion) order.
It imports no LLM code (import-linter contract "Judgement path never calls an LLM").

Published semantics (v1). Changing any rule below changes results, so every rule is listed
here and every assumption the replay makes is spelled out as ``(가정)`` in event notes.

Assumptions
  * Every ``SimConfig`` timing must be a finite number ≥ 0 (optional ones may be ``None``) and
    ``max_hops`` ≥ 0; otherwise ``replay``/``replay_all`` raise ``ValueError``. A NaN timing makes
    every comparison false (any mitigation would look in time) and a negative one lets events
    precede the incident, so a broken assumption is refused instead of judged (P3).

Time and availability
  * ``t`` is minutes since the incident starts (t=0, ``INCIDENT_START``); no event precedes it.
    Perturbation windows
    are half-open ``[start_minute, end_minute)``; ``end_minute=None`` lasts forever. A window
    whose bounds are not numbers (NaN) never covers any minute.
  * A person is available at ``t`` iff their employment status is ACTIVE and no window blocks
    them. The HR roster entry (if any) wins over the graph attribute; a missing status is
    UNKNOWN and UNKNOWN is *not* available (확인 불가 ≠ 정상, P3).
  * ``PERSON_UNAVAILABLE(x)`` blocks every person ``x`` resolves to (a person resolves to itself).
  * ``CHANNEL_DOWN(c)`` takes channel ``c`` down.
  * ``SYSTEM_DOWN(x)`` / ``LOCATION_DOWN(x)`` take down every channel whose failure domain
    (transitive DEPENDS_ON/RUNS_IN closure, ``CheckContext.failure_domain``) contains ``x``.
    Services whose failure domain contains ``x`` are listed as degraded in the
    ``INCIDENT_START`` note (informational only).
  * ``CONCURRENT_INCIDENT(s2)``: the first first-responder of ``s2`` (see
    :func:`first_responders`) who is available at the window start is busy for the whole
    window. Documented simplification: exactly one person is pulled away, with no hand-over.
  * Perturbations are applied in a canonical (total) order, so the order in which a scenario
    lists them never changes its result.
  * Nodes/edges under HOLD (판단 보류) are excluded (:func:`trusted_view`): an unconfirmed fact
    must never make a rehearsal pass (P3). An obligation tied to services only through held
    MUST_REPORT edges is excluded with them — in the view it would otherwise look org-wide and
    bind every service. The ``INCIDENT_START`` note says how many elements were left out.

Detection
  * A scenario whose ``service_id`` is not a confirmed service/system node of the graph cannot
    be rehearsed: ``UNDETECTED`` and FAIL (element DETECTION) with a note saying so.
  * With DETECTED_BY triggers, each trigger fires at its ``delay_minutes`` attribute, else at
    ``SimConfig.detection_delay_minutes``; ``DETECTED`` is the earliest firing (ties by id),
    and its note marks a configured delay as an assumption.
  * Without triggers: ``undetected_discovery_minutes=None`` → ``UNDETECTED`` and FAIL
    (element DETECTION, at the service). Otherwise ``DETECTED`` at that minute, noted as
    '고객 문의로 인지(가정)'.

Paging and escalation (shared by owner, approver and reporter paging)
  * Owner paging: every trigger pages its NOTIFIES targets when it fires (time, trigger id,
    edge id order). Without NOTIFIES edges the service owners are paged at detection time
    ('호출 대상 미지정 → 담당자 직접 연락(가정)'). The people these first pages can reach are
    the service's *first responders* (:func:`first_responders`).
  * Paging node N at ``t`` resolves N to people (``CheckContext.resolve_people`` order). The
    first person who is available AND reachable on an up channel gets ``PAGE`` at ``t`` and
    ``ACK`` at ``t + ack_minutes``. Channel order: the NOTIFIES ``channel`` attribute, then the
    person's REACHABLE_VIA channels, then those of the paged role/team, each followed
    depth-first by its FALLBACK_TO chain. Availability is evaluated at page time.
  * A later alarm that pages a node again re-evaluates it (someone back from a short absence
    answers the second alarm); the same page at the same minute on the same channel is not
    repeated.
  * Otherwise ``NO_ANSWER`` / ``CHANNEL_UNAVAILABLE`` are emitted and every ESCALATES_TO edge
    of N (and of the people N resolved to) is scheduled at ``t + after_minutes``. A missing
    ``after_minutes`` uses ``default_escalation_after_minutes`` (the ``ESCALATE`` note marks it
    as an assumption); if that is ``None`` too the hop cannot be timed and is skipped — the
    replay never invents a waiting time (P2).
    Escalation never revisits a node: a hop back into the node's own escalation chain is a
    cycle ('상향 순환'); at most ``max_hops`` hops are followed. When nothing is left,
    ``DEAD_END`` is emitted at the last node paged, with the reason that ended its branch.

Mitigation
  * After the owner ACK by responder R: the first MITIGATED_BY action, preferring
    rollback/failover/feature_flag, then any other, by id. None → ``NO_MITIGATION`` → FAIL
    (element ROLLBACK).
  * Approval is required by the single rule :meth:`CheckContext.requires_approval`: an explicit
    ``requires_approval`` wins, otherwise rollback/failover need approval and any other action
    does not (an APPROVED_BY edge says who approves, not whether). If R resolves from an
    approver node, R approves directly; otherwise approvers are paged from R's ACK with the same
    paging logic.
    Reached → ``APPROVAL_REQUESTED`` at their ACK, ``APPROVED`` ``approval_minutes`` later.
    Not reached → ``APPROVAL_BLOCKED`` → FAIL (element APPROVER, at the last node tried).
  * Executor: the first available person of the EXECUTED_BY nodes, else R (noted as an
    assumption). ``ACTION_STARTED`` at approval (or ACK), ``MITIGATED`` after the action's
    ``duration_minutes`` (else ``default_action_minutes``, noted as an assumption).

Reporting (incident types BREACH and PRIVACY_LEAK only)
  * Duties are the obligations binding the service (:meth:`CheckContext.obligations`: its
    MUST_REPORT targets plus org-wide obligations no MUST_REPORT edge points at) whose
    ``incident_type`` matches, plus the policy obligation rules that apply to the service
    (``applies`` and in regulated scope: ``Policy.regulated_services`` by id or name, or every
    critical service when empty).
  * A policy rule with no matching obligation in the graph → ``REPORT_MISSED`` citing the law.
    If the runbook has obligations whose ``incident_type`` is missing or unrecognised, the note
    says their applicability cannot be confirmed instead of claiming no procedure exists.
  * Each obligation's reporters (REPORTED_BY) are paged from detection time; ``REPORT_FILED``
    at their ACK. The deadline counts from detection (법정 기준: '알게 된 때부터') and is the
    stricter of the obligation's ``deadline_minutes`` and the applicable policy rule; neither →
    the deadline is unknown and the report cannot be confirmed (P3). Late, unreachable or
    missing reporters → ``REPORT_MISSED`` (emitted at the due time, never before detection) →
    FAIL with no element.

Verdict
  * Deadline = ``Scenario.deadline_minutes`` else ``CheckContext.slo_minutes(service)``. A
    scenario deadline that is not a finite number (NaN, ±∞) states no usable limit and is
    treated like ``None`` (noted): a malformed value can never make every mitigation look in
    time (P3), and every result stays portable JSON for storage. A negative finite deadline is
    taken as given (nothing can be mitigated before the incident starts → FAIL). When neither
    the service nor the policy states an SLO (empty ``tier_slo_minutes``), the deadline is 0
    (noted): an unknown target is never reported as met.
    Mitigated after it → ``DEADLINE_MISSED`` (emitted at the deadline, or at t=0 when the
    deadline is negative) → FAIL.
  * PASS iff mitigated within the deadline and every required report was filed in time.
    On FAIL, ``failed_element``/``failed_at_id`` describe the first failure in path order
    (detection → owner → rollback → approver → deadline → reports); ``reason`` lists all.
  * ``events`` are stably sorted by ``t``. ``path_ids`` lists traversed node/edge ids in first
    visit order (unique) for UI highlighting.
  * ``graph_hash`` = ``OpsGraph.content_hash()``; ``scenario_fingerprint`` =
    ``Scenario.fingerprint()``; ``config_hash`` = :func:`config_hash` (SimConfig + policy
    fields that influence the replay + roster facts), so the triple identifies the result.

Event fields: ``PAGE``/``CHANNEL_UNAVAILABLE`` use ``actor_id`` = channel and ``target_id`` =
person; ``ACK`` uses ``actor_id`` = person and ``target_id`` = the paged node; ``ESCALATE`` uses
``actor_id`` = escalation source and ``target_id`` = destination; ``edge_id`` is the response
path edge (NOTIFIES/OWNED_BY/ESCALATES_TO/APPROVED_BY/REPORTED_BY/…) that led to the step.
"""

from __future__ import annotations

import hashlib
import heapq
import itertools
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from jettae.check.context import CheckContext
from jettae.domain import (
    DEFAULT_OBLIGATIONS,
    KOREA,
    ActionType,
    AttrKey,
    Edge,
    EdgeKind,
    Element,
    IncidentType,
    Node,
    NodeKind,
    ObligationRule,
    OpsGraph,
    Outcome,
    PersonStatus,
    Perturbation,
    PerturbationKind,
    Policy,
    RehearsalResult,
    RosterEntry,
    Scenario,
    SimConfig,
    SimEvent,
    SimEventKind,
    Verification,
)

__all__ = [
    "FirstContact",
    "applicable_obligation_rules",
    "can_respond",
    "choose_mitigation",
    "config_hash",
    "effective_status",
    "first_contacts",
    "first_responders",
    "paging_channels",
    "replay",
    "replay_all",
    "requires_approval",
    "trusted_view",
]

_DEFAULT_POLICY = Policy()
_DEFAULT_CONFIG = SimConfig()

#: Tried first because they undo or route around the fault instead of nursing it.
_PREFERRED_ACTIONS = frozenset({ActionType.ROLLBACK, ActionType.FAILOVER, ActionType.FEATURE_FLAG})
_REPORTABLE = frozenset({IncidentType.BREACH, IncidentType.PRIVACY_LEAK})
#: Node kinds that can have an incident (DETECTED_BY / MITIGATED_BY sources).
_REHEARSABLE = frozenset({NodeKind.SERVICE, NodeKind.SYSTEM})
#: Policy fields that influence a replay (SLOs, critical tiers, legal duties and their scope).
_POLICY_FIELDS = {"critical_tiers", "tier_slo_minutes", "default_tier", "obligations", "regulated_services"}
#: Roster facts that influence a replay (identity matching and employment status).
_ROSTER_FIELDS = {"name", "aliases", "status"}
#: SimConfig timings the replay adds to the clock: each must be a finite number ≥ 0.
_REQUIRED_TIMINGS = ("detection_delay_minutes", "ack_minutes", "approval_minutes", "default_action_minutes")
#: SimConfig timings where ``None`` is meaningful ("no such assumption"); a number must be valid.
_OPTIONAL_TIMINGS = ("undetected_discovery_minutes", "default_escalation_after_minutes")

_INCIDENT_LABEL = {
    IncidentType.OUTAGE: "장애",
    IncidentType.BREACH: "침해사고",
    IncidentType.PRIVACY_LEAK: "개인정보 유출",
}
_STATUS_REASON = {
    PersonStatus.ACTIVE: "재직 중",
    PersonStatus.DEPARTED: "퇴사",
    PersonStatus.ON_LEAVE: "장기 부재",
    PersonStatus.UNKNOWN: "재직 여부 확인 불가(명부 미확인)",
}

_NO_ESCALATION = "상향 경로 없음"
_NO_TIMING = "상향 대기시간 미기재로 상향 시점 불명"
_CYCLE = "상향 순환"
_REVISIT = "이미 호출에 실패한 대상으로 상향"
_OWNER_ASSUMPTION = "호출 대상 미지정 → 담당자 직접 연락(가정)"
_DISCOVERY_ASSUMPTION = "감지 신호 없음 → 고객 문의로 인지(가정)"
_DELAY_ASSUMPTION = " — 감지 지연 미기재로 기본값 가정"
_WAIT_ASSUMPTION = "(상향 대기시간 미기재로 기본값 가정)"
_NAN_DEADLINE = "시나리오 목표 시간이 숫자가 아님 → 서비스 목표 복구 시간 적용"
_INFINITE_DEADLINE = (
    "시나리오 목표 시간이 무한대(기한 없음)라 판정 기준이 될 수 없음 → 서비스 목표 복구 시간 적용"
)
_NO_SLO = "목표 복구 시간이 서비스·정책 어디에도 없음 → 시한 내 완화를 확인할 수 없어 0분 기준으로 판정"


class _Purpose(NamedTuple):
    label: str  # prefix of every paging note, so the timeline says *why* someone is paged
    nobody: str  # dead-end reason when there is no one to page at all


_OWNER = _Purpose("담당 호출", "호출 대상·담당자 미지정")
_APPROVAL = _Purpose("승인 요청", "승인자 미지정")
_REPORTING = _Purpose("신고 담당 호출", "신고 담당 미지정")


# --------------------------------------------------------------------------- Korean text


class _Particle(NamedTuple):
    after_consonant: str
    after_vowel: str
    unknown: str  # used when the final sound cannot be read (Latin letters, digits …)


_SUBJECT = _Particle("이", "가", "이(가)")
_TOWARD = _Particle("으로", "로", "(으)로")
_RIEUL = 8  # index of the ㄹ final consonant; ㄹ takes "로", not "으로"


def _attach(word: str, particle: _Particle) -> str:
    """Append the particle matching the word's final sound (김대리가, 팀장이, 슬랙으로, 전화로)."""
    last = word[-1:]
    if not "가" <= last <= "힣":
        return word + particle.unknown
    final = (ord(last) - ord("가")) % 28
    if final == 0 or (final == _RIEUL and particle is _TOWARD):
        return word + particle.after_vowel
    return word + particle.after_consonant


def _fmt(minutes: float) -> str:
    if math.isnan(minutes):
        return "?"
    if math.isinf(minutes):
        return "∞" if minutes > 0 else "-∞"
    if minutes == int(minutes):
        return str(int(minutes))
    return f"{minutes:.2f}".rstrip("0").rstrip(".")


def _duration(minutes: float) -> str:
    """Legal deadlines read naturally in hours (24시간), operational ones in minutes."""
    if math.isfinite(minutes) and minutes >= 60 and minutes % 60 == 0:
        return f"{_fmt(minutes / 60)}시간"
    return f"{_fmt(minutes)}분"


# --------------------------------------------------------------------------- attribute parsing


def _minutes(value: object) -> float | None:
    """Only an explicit, finite, non-negative number counts; anything else stays absent (P2)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _action_type(action: Node) -> ActionType | None:
    raw = action.attr(AttrKey.ACTION_TYPE)
    try:
        return ActionType(raw) if isinstance(raw, str) else None
    except ValueError:
        return None


def _incident_type(node: Node) -> IncidentType | None:
    raw = node.attr(AttrKey.INCIDENT_TYPE)
    try:
        return IncidentType(raw) if isinstance(raw, str) else None
    except ValueError:
        return None


def _stricter(a: float | None, b: float | None) -> float | None:
    present = [x for x in (a, b) if x is not None]
    return min(present) if present else None


def _bound_key(value: float | None) -> tuple[int, float]:
    """Total order for window bounds: numbers (±inf included), then NaN, then None (open end).

    Plain floats are not totally ordered (NaN compares false both ways, and ``None`` would have
    to be mapped onto ``inf``), which would let the input order leak into the result.
    """
    if value is None:
        return (2, 0.0)
    if math.isnan(value):
        return (1, 0.0)
    return (0, value)


def _perturbation_key(p: Perturbation) -> tuple[bool, str, str, tuple[int, float], tuple[int, float]]:
    # concurrent incidents last: each occupies someone still free after every absence/outage
    return (
        p.kind is PerturbationKind.CONCURRENT_INCIDENT,
        p.kind.value,
        p.target_id,
        _bound_key(p.start_minute),
        _bound_key(p.end_minute),
    )


# --------------------------------------------------------------------------- initial pages


@dataclass(frozen=True)
class _Seed:
    node_id: str
    via_edge: str
    source_id: str
    t: float
    channel_hint: tuple[str, str] | None = None  # (channel id, edge naming it)


def _hint_channel(graph: OpsGraph, notifies: Edge) -> tuple[str, str] | None:
    """The NOTIFIES ``channel`` attribute, only when it names an existing channel node."""
    value = notifies.attr(AttrKey.CHANNEL)
    node = graph.nodes.get(value) if isinstance(value, str) else None
    return (node.id, notifies.id) if node is not None and node.kind is NodeKind.CHANNEL else None


def _fire_time(trigger: Node, config: SimConfig) -> float:
    delay = _minutes(trigger.attr(AttrKey.DELAY_MINUTES))
    return config.detection_delay_minutes if delay is None else delay


def _notify_seeds(ctx: CheckContext, service_id: str, config: SimConfig) -> list[_Seed]:
    """Every trigger's NOTIFIES pages at its firing time, in paging order (t, trigger, edge)."""
    seeds: dict[str, _Seed] = {}
    for trigger in ctx.triggers(service_id):
        fire_t = _fire_time(trigger, config)
        for notifies in ctx.notified(trigger.id):
            hint = _hint_channel(ctx.graph, notifies)
            seeds.setdefault(notifies.id, _Seed(notifies.dst, notifies.id, trigger.id, fire_t, hint))
    return sorted(seeds.values(), key=lambda s: s.t)  # stable: id order within the same minute


def _owner_seeds(ctx: CheckContext, service_id: str, t: float) -> list[_Seed]:
    return [_Seed(e.dst, e.id, service_id, t) for e in ctx.graph.out_edges(service_id, EdgeKind.OWNED_BY)]


def _contact_roots(
    graph: OpsGraph, person_id: str, via_id: str, hint: tuple[str, str] | None
) -> list[tuple[str, str]]:
    """(channel id, edge offering it) the replay tries first: hint, person's, paged group's."""
    roots: list[tuple[str, str]] = [] if hint is None else [hint]
    roots += [(e.dst, e.id) for e in graph.out_edges(person_id, EdgeKind.REACHABLE_VIA)]
    if via_id != person_id:
        roots += [(e.dst, e.id) for e in graph.out_edges(via_id, EdgeKind.REACHABLE_VIA)]
    return roots


# --------------------------------------------------------------------------- shared semantics


def trusted_view(graph: OpsGraph) -> tuple[OpsGraph, int]:
    """Drop HOLD nodes/edges (and edges touching HOLD nodes); return the view + excluded count.

    The replay judges only confirmed facts (P3); scenario templates use the same view so they
    stress exactly what the replay will walk. An obligation whose every MUST_REPORT link is
    held goes too: without its links it would read as an org-wide duty
    (:meth:`CheckContext.is_org_obligation`) and an unconfirmed step of one service would
    satisfy the reporting of all of them.
    """
    held_nodes = {nid for nid, n in graph.nodes.items() if n.verification is Verification.HOLD}
    held_edges = {eid for eid, e in graph.edges.items() if e.verification is Verification.HOLD}
    for obligation in graph.nodes_of(NodeKind.OBLIGATION):
        links = graph.in_edges(obligation.id, EdgeKind.MUST_REPORT)
        if links and all(e.id in held_edges or e.src in held_nodes for e in links):
            held_nodes.add(obligation.id)
    if not held_nodes and not held_edges:
        return graph, 0
    nodes = {nid: n for nid, n in graph.nodes.items() if nid not in held_nodes}
    edges = {
        eid: e
        for eid, e in graph.edges.items()
        if eid not in held_edges and e.src not in held_nodes and e.dst not in held_nodes
    }
    return OpsGraph(nodes=nodes, edges=edges, docs=graph.docs), len(held_nodes) + len(held_edges)


def effective_status(ctx: CheckContext, person: Node) -> PersonStatus:
    """Employment status as the replay sees it: the HR roster (trusted) beats the graph."""
    entry = ctx.roster_entry(person)
    return entry.status if entry is not None else ctx.person_status(person)


def can_respond(ctx: CheckContext, person: Node) -> bool:
    """Static availability: an ACTIVE person not excluded by ``ctx.unavailable``."""
    return (
        person.kind is NodeKind.PERSON
        and person.id not in ctx.unavailable
        and effective_status(ctx, person) is PersonStatus.ACTIVE
    )


def paging_channels(graph: OpsGraph, person_id: str, via_id: str) -> tuple[str, ...]:
    """Channels the replay tries first when paging ``person_id`` through node ``via_id``.

    Fallback channels are excluded: an empty result means the person cannot be paged at all.
    """
    return tuple(dict.fromkeys(cid for cid, _ in _contact_roots(graph, person_id, via_id, None)))


class FirstContact(NamedTuple):
    """A first responder and how the replay's first page reaches them at baseline."""

    person: Node
    via_id: str  # the paged node (person/role/team) that resolved to ``person``
    channel_ids: tuple[str, ...]  # channels tried first, in the replay's order (non-empty)


def first_contacts(
    ctx: CheckContext, service_id: str, *, config: SimConfig = _DEFAULT_CONFIG
) -> list[FirstContact]:
    """The service's first pages, in the replay's paging order, one entry per person.

    Pages are the NOTIFIES targets of the service's triggers ordered by firing time (then
    trigger id, edge id), or the owners when no trigger notifies anyone. A person is listed
    only if they :func:`can_respond` and have at least one channel to be paged on — someone the
    replay could never reach would make "1차 담당 부재" or a concurrent incident a no-op.
    """
    seeds = _notify_seeds(ctx, service_id, config) or _owner_seeds(ctx, service_id, 0.0)
    contacts: dict[str, FirstContact] = {}
    for seed in seeds:
        for person in ctx.resolve_people(seed.node_id):
            if person.id in contacts or not can_respond(ctx, person):
                continue
            roots = _contact_roots(ctx.graph, person.id, seed.node_id, seed.channel_hint)
            if roots:
                channels = tuple(dict.fromkeys(cid for cid, _ in roots))
                contacts[person.id] = FirstContact(person, seed.node_id, channels)
    return list(contacts.values())


def first_responders(
    ctx: CheckContext, service_id: str, *, config: SimConfig = _DEFAULT_CONFIG
) -> list[Node]:
    """People the replay pages first for ``service_id``, in paging order (see :func:`first_contacts`).

    Templates and ``CONCURRENT_INCIDENT`` use this so a perturbation hits exactly whom the
    replay would call first.
    """
    return [contact.person for contact in first_contacts(ctx, service_id, config=config)]


def choose_mitigation(ctx: CheckContext, service_id: str) -> Edge | None:
    """The MITIGATED_BY edge the replay uses: rollback/failover/feature_flag first, then by id."""
    links = ctx.graph.out_edges(service_id, EdgeKind.MITIGATED_BY)
    if not links:
        return None
    return min(
        links,
        key=lambda e: (_action_type(ctx.graph.nodes[e.dst]) not in _PREFERRED_ACTIONS, e.dst, e.id),
    )


def requires_approval(ctx: CheckContext, action: Node) -> bool:
    """Whether the replay pages approvers before ``action``: :meth:`CheckContext.requires_approval`.

    Kept as the simulator's published entry point so templates ("승인자 부재" only when the replay
    would actually ask that approver), the score and the replay share one rule.
    """
    return ctx.requires_approval(action)


def applicable_obligation_rules(
    ctx: CheckContext, service_id: str, incident_type: IncidentType
) -> list[ObligationRule]:
    """Policy reporting rules binding ``service_id`` for ``incident_type`` (sorted by key).

    Scope follows :class:`Policy`: ``regulated_services`` (matched by id or name) when set,
    otherwise every critical service.
    """
    policy = ctx.policy
    node = ctx.graph.nodes.get(service_id)
    if policy.regulated_services:
        names = {service_id} if node is None else {service_id, node.name}
        in_scope = bool(names & set(policy.regulated_services))
    else:
        in_scope = node is not None and ctx.service_tier(service_id) in policy.critical_tiers
    if not in_scope:
        return []
    rules = list(policy.obligations_in_force(incident_type))
    return sorted(rules, key=lambda r: r.key)


#: Jurisdiction an obligation had before the field existed: Korean law for the statutory defaults,
#: none (an own duty) for everything else.
_IMPLIED_JURISDICTION = {o.key: o.jurisdiction for o in DEFAULT_OBLIGATIONS}


def _policy_payload(policy: Policy) -> dict[str, Any]:
    """The replay-relevant policy fields (``_POLICY_FIELDS``) as hashed by :func:`config_hash`.

    Jurisdictions change which duties a replay checks, so they are hashed — but only where they
    differ from what a policy implied before they existed (Korean statutory defaults, own duties
    elsewhere, a Korean organisation). Replays of such policies keep the config hash they had.
    """
    payload = policy.model_dump(mode="json", include=_POLICY_FIELDS)
    for item in payload["obligations"]:
        if item.get("jurisdiction") == _IMPLIED_JURISDICTION.get(item["key"]):
            item.pop("jurisdiction", None)
    if policy.jurisdictions != (KOREA,):
        payload["jurisdictions"] = list(policy.jurisdictions)
    return payload


def config_hash(
    config: SimConfig, policy: Policy = _DEFAULT_POLICY, roster: Iterable[RosterEntry] = ()
) -> str:
    """SHA-256 of the canonical replay assumptions.

    Besides ``SimConfig`` it covers the policy fields and roster facts that change a replay, so
    (graph_hash, scenario_fingerprint, config_hash) fully determines a :class:`RehearsalResult`.
    """
    payload = {
        "config": config.model_dump(mode="json"),
        "policy": _policy_payload(policy),
        "roster": [entry.model_dump(mode="json", include=_ROSTER_FIELDS) for entry in roster],
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _check_config(config: SimConfig) -> None:
    """Refuse replay assumptions that would make the verdict meaningless (P3).

    ``SimConfig`` itself accepts any float. A NaN ack time makes ``mitigated > deadline`` false
    for every scenario (a silent PASS) and a negative one lets an ACK precede its page. These
    are configuration errors, not runbook facts, so the replay stops instead of judging.
    """
    values = config.model_dump()
    bad = [name for name in _REQUIRED_TIMINGS if _minutes(values[name]) is None]
    bad += [name for name in _OPTIONAL_TIMINGS if values[name] is not None and _minutes(values[name]) is None]
    if config.max_hops < 0:
        bad.append("max_hops")
    if bad:
        raise ValueError(
            f"리허설 가정(SimConfig) 값이 올바르지 않습니다: {', '.join(bad)} — "
            "시간 값은 0 이상의 유한한 숫자, max_hops는 0 이상의 정수여야 합니다."
        )


# --------------------------------------------------------------------------- public API


def replay(
    graph: OpsGraph,
    scenario: Scenario,
    *,
    policy: Policy = _DEFAULT_POLICY,
    config: SimConfig = _DEFAULT_CONFIG,
    roster: Iterable[RosterEntry] = (),
) -> RehearsalResult:
    """Replay one scenario over ``graph`` deterministically (see module docstring)."""
    return _Replayer(graph, policy, config, tuple(roster)).run(scenario)


def replay_all(
    graph: OpsGraph,
    scenarios: Iterable[Scenario],
    *,
    policy: Policy = _DEFAULT_POLICY,
    config: SimConfig = _DEFAULT_CONFIG,
    roster: Iterable[RosterEntry] = (),
) -> list[RehearsalResult]:
    """Replay several scenarios (results in scenario order), sharing the graph analysis."""
    replayer = _Replayer(graph, policy, config, tuple(roster))
    return [replayer.run(s) for s in scenarios]


# --------------------------------------------------------------------------- internals


@dataclass(frozen=True)
class _Window:
    start: float
    end: float | None

    def covers(self, t: float) -> bool:
        return self.start <= t and (self.end is None or t < self.end)

    def describe(self) -> str:
        if self.end is not None:
            return f"({_fmt(self.start)}~{_fmt(self.end)}분)"
        return f"({_fmt(self.start)}분부터)" if self.start else ""


@dataclass(frozen=True)
class _Block:
    window: _Window
    reason: str


def _first_reason(blocks: Sequence[_Block], t: float) -> str | None:
    return next((b.reason for b in blocks if b.window.covers(t)), None)


@dataclass
class _Availability:
    ctx: CheckContext
    people: dict[str, list[_Block]] = field(default_factory=dict)
    channels: dict[str, list[_Block]] = field(default_factory=dict)

    def block_person(self, person_id: str, block: _Block) -> None:
        self.people.setdefault(person_id, []).append(block)

    def block_channel(self, channel_id: str, block: _Block) -> None:
        self.channels.setdefault(channel_id, []).append(block)

    def person_reason(self, person: Node, t: float) -> str | None:
        """Why ``person`` cannot respond at ``t`` (None = available)."""
        status = effective_status(self.ctx, person)
        if status is not PersonStatus.ACTIVE:
            return _STATUS_REASON[status]
        return _first_reason(self.people.get(person.id, ()), t)

    def channel_reason(self, channel_id: str, t: float) -> str | None:
        return _first_reason(self.channels.get(channel_id, ()), t)


@dataclass(frozen=True, order=True)
class _Hop:
    t: float
    seq: int
    node_id: str = field(compare=False)
    via_edge: str = field(compare=False)
    source_id: str = field(compare=False)
    parent: str | None = field(compare=False)  # paged node whose failure caused this hop
    after: float | None = field(compare=False)  # set for ESCALATES_TO hops
    channel_hint: tuple[str, str] | None = field(compare=False)
    hops: int = field(compare=False)


@dataclass(frozen=True)
class _Reached:
    person: Node
    node_id: str
    via_edge: str
    ack_t: float


@dataclass(frozen=True)
class _DeadEnd:
    last_id: str
    t: float
    reason: str


@dataclass(frozen=True)
class _Failure:
    element: Element | None
    at_id: str
    reason: str


def _is_ancestor(parent: dict[str, str | None], candidate: str, node: str) -> bool:
    current: str | None = node
    while current is not None:
        if current == candidate:
            return True
        current = parent.get(current)
    return False


class _Replayer:
    """Per-(graph, policy, config, roster) state shared by every scenario of a batch."""

    def __init__(
        self, graph: OpsGraph, policy: Policy, config: SimConfig, roster: tuple[RosterEntry, ...]
    ) -> None:
        _check_config(config)
        view, held = trusted_view(graph)
        self.graph_hash = graph.content_hash()
        self.config_hash = config_hash(config, policy, roster)
        self.ctx = CheckContext(graph=view, policy=policy, roster=roster)
        self.config = config
        self.held = held
        self.channel_domains = {c.id: self.ctx.failure_domain(c.id) for c in view.nodes_of(NodeKind.CHANNEL)}
        self.service_domains = {s.id: self.ctx.failure_domain(s.id) for s in view.nodes_of(NodeKind.SERVICE)}

    def run(self, scenario: Scenario) -> RehearsalResult:
        return _Run(self, scenario).execute()


class _Run:
    """One scenario replay: availability model, event log, traversed path and failures."""

    def __init__(self, replayer: _Replayer, scenario: Scenario) -> None:
        self.replayer = replayer
        self.ctx = replayer.ctx
        self.graph = replayer.ctx.graph
        self.config = replayer.config
        self.scenario = scenario
        self.sid = scenario.service_id
        target = self.graph.nodes.get(self.sid)
        self.rehearsable = target is not None and target.kind in _REHEARSABLE
        self.events: list[SimEvent] = []
        self.path: dict[str, None] = {}
        self.failures: list[_Failure] = []
        #: trigger id -> its DETECTED_BY edge, so every alarm that pages is highlighted end to end
        self.lead_in: dict[str, str] = {}
        self.avail = _Availability(self.ctx)
        self.assumptions = self._apply_perturbations()
        self.deadline, self.deadline_note = self._deadline()

    # ------------------------------------------------------------------ bookkeeping
    def _name(self, nid: str) -> str:
        node = self.graph.nodes.get(nid)
        return node.name if node is not None else nid

    def _visit(self, *ids: str | None) -> None:
        for nid in ids:
            if nid is not None:
                self.path.setdefault(nid, None)

    def _emit(
        self,
        t: float,
        kind: SimEventKind,
        note: str,
        *,
        actor: str | None = None,
        target: str | None = None,
        edge: str | None = None,
    ) -> None:
        self.events.append(
            SimEvent(t=t, kind=kind, actor_id=actor, target_id=target, edge_id=edge, note=note)
        )

    def _fail(self, element: Element | None, at_id: str, reason: str) -> None:
        self.failures.append(_Failure(element, at_id, reason))

    # ------------------------------------------------------------------ perturbations
    def _apply_perturbations(self) -> list[str]:
        """Build the availability model; return Korean descriptions for INCIDENT_START.

        Perturbations are applied in a canonical total order (:func:`_perturbation_key`), so the
        tuple order of a scenario never changes its result.
        """
        notes: list[str] = []
        for p in sorted(self.scenario.perturbations, key=_perturbation_key):
            window = _Window(p.start_minute, p.end_minute)
            name = self._name(p.target_id)
            if p.kind is PerturbationKind.PERSON_UNAVAILABLE:
                for person in self.ctx.resolve_people(p.target_id):
                    self.avail.block_person(person.id, _Block(window, "시나리오상 부재"))
                notes.append(f"{name} 부재{window.describe()}")
            elif p.kind is PerturbationKind.CHANNEL_DOWN:
                self.avail.block_channel(p.target_id, _Block(window, "시나리오상 다운"))
                notes.append(f"{name} 다운{window.describe()}")
            elif p.kind is PerturbationKind.CONCURRENT_INCIDENT:
                notes.append(self._occupy_responder(p.target_id, window, name))
            else:
                notes.append(self._take_down(p.target_id, window, name))
        return notes

    def _take_down(self, target: str, window: _Window, name: str) -> str:
        hit = [cid for cid, domain in self.replayer.channel_domains.items() if target in domain]
        for cid in hit:
            self.avail.block_channel(cid, _Block(window, f"{name} 장애 영향"))
        degraded = [sid for sid, domain in self.replayer.service_domains.items() if target in domain]
        note = f"{name} 장애{window.describe()}"
        if hit:
            note += f" — 영향 채널: {', '.join(self._name(c) for c in hit)}"
        if degraded:
            note += f" — 영향 서비스: {', '.join(self._name(s) for s in degraded)}"
        return note

    def _occupy_responder(self, service_id: str, window: _Window, name: str) -> str:
        busy = next(
            (
                p
                for p in first_responders(self.ctx, service_id, config=self.config)
                if self.avail.person_reason(p, window.start) is None
            ),
            None,
        )
        if busy is None:
            return f"{name} 동시 장애{window.describe()} — 투입될 1차 대응자 없음"
        self.avail.block_person(busy.id, _Block(window, f"{name} 동시 장애 대응 중"))
        return f"{name} 동시 장애{window.describe()} — {busy.name} 투입"

    # ------------------------------------------------------------------ main flow
    def execute(self) -> RehearsalResult:
        self._start()
        mitigated_at: float | None = None
        filed = 0
        detection = self._detect()
        if detection is not None:
            detected_at, seeds = detection
            owner = self._page(seeds, _OWNER, anchor=self.sid, t=detected_at)
            if isinstance(owner, _Reached):
                mitigated_at = self._mitigate(owner)
            else:
                self._fail(
                    Element.OWNER,
                    owner.last_id,
                    f"담당 호출이 {self._name(owner.last_id)}에서 끊김 — {owner.reason}",
                )
            if mitigated_at is not None and mitigated_at > self.deadline:
                note = f"목표 복구 시간 {_fmt(self.deadline)}분 초과 — 완화까지 {_fmt(mitigated_at)}분"
                # a negative deadline is missed from the start; nothing happens before t=0
                missed_at = max(self.deadline, 0.0)
                self._emit(missed_at, SimEventKind.DEADLINE_MISSED, note, target=self.sid)
                self._fail(None, self.sid, note)
            filed = self._report(detected_at)
        return self._result(mitigated_at, filed)

    def _start(self) -> None:
        label = _INCIDENT_LABEL[self.scenario.incident_type]
        parts = [f"{self._name(self.sid)} {label} 발생", *self.assumptions]
        if self.deadline_note is not None:
            parts.append(self.deadline_note)
        if self.replayer.held:
            parts.append(f"판단 보류 요소 {self.replayer.held}개는 확인 전까지 대응 경로에서 제외")
        if self.rehearsable:
            self._visit(self.sid)
        self._emit(0.0, SimEventKind.INCIDENT_START, " · ".join(parts), target=self.sid)

    def _deadline(self) -> tuple[float, str | None]:
        """Scenario deadline, else the service SLO; NaN/±∞ state no usable limit and fall back (P3).

        An infinite deadline would pass every mitigation and cannot be stored as JSON, so it is
        treated like an unstated one rather than taken literally.
        """
        requested = self.scenario.deadline_minutes
        if requested is not None and math.isfinite(requested):
            return float(requested), None
        notes = []
        if requested is not None:
            notes.append(_NAN_DEADLINE if math.isnan(requested) else _INFINITE_DEADLINE)
        slo = self._service_slo()
        if slo is None:
            notes.append(_NO_SLO)
        return (0.0 if slo is None else slo), " · ".join(notes) or None

    def _service_slo(self) -> float | None:
        """The SLO as :meth:`CheckContext.slo_minutes` defines it; None when nothing states one.

        An empty policy SLO table is a degenerate but loadable policy; the replay must not crash
        on it, and an unstated target must not become a generous one (P3).
        """
        policy = self.ctx.policy
        node = self.graph.nodes.get(self.sid)
        if node is not None and (policy.tier_slo_minutes or isinstance(node.attr(AttrKey.SLO_MINUTES), int)):
            return float(self.ctx.slo_minutes(self.sid))
        slos = policy.tier_slo_minutes
        return float(slos.get(policy.default_tier, max(slos.values()))) if slos else None

    def _detect(self) -> tuple[float, list[_Seed]] | None:
        if not self.rehearsable:
            note = (
                f"{self._name(self.sid)}: 운영 그래프에서 확인된 서비스·시스템이 아님"
                "(미등록·판단 보류·다른 종류) — 감지·대응 경로를 확인할 수 없음"
            )
            self._emit(0.0, SimEventKind.UNDETECTED, note, target=self.sid)
            self._fail(Element.DETECTION, self.sid, note)
            return None
        links = self.graph.out_edges(self.sid, EdgeKind.DETECTED_BY)
        service = self._name(self.sid)
        if not links:
            discovery = self.config.undetected_discovery_minutes
            if discovery is None:
                note = f"{service} 장애를 알릴 감지 신호(알람·모니터)가 없어 아무도 인지하지 못함"
                self._emit(0.0, SimEventKind.UNDETECTED, note, target=self.sid)
                self._fail(Element.DETECTION, self.sid, f"감지 경로 없음 — {note}")
                return None
            note = f"{_DISCOVERY_ASSUMPTION} · {_OWNER_ASSUMPTION}"
            self._emit(discovery, SimEventKind.DETECTED, note, target=self.sid)
            return discovery, _owner_seeds(self.ctx, self.sid, discovery)
        for link in links:
            self.lead_in.setdefault(link.dst, link.id)
        fires = sorted(
            ((_fire_time(self.graph.nodes[e.dst], self.config), e) for e in links),
            key=lambda f: (f[0], f[1].dst, f[1].id),
        )
        detected_at, first = fires[0]
        seeds = _notify_seeds(self.ctx, self.sid, self.config)
        trigger = self.graph.nodes[first.dst]
        assumed = "" if _minutes(trigger.attr(AttrKey.DELAY_MINUTES)) is not None else _DELAY_ASSUMPTION
        note = f"{_attach(trigger.name, _TOWARD)} 감지(장애 후 {_fmt(detected_at)}분{assumed})"
        if not seeds:
            seeds = _owner_seeds(self.ctx, self.sid, detected_at)
            note += f" · {_OWNER_ASSUMPTION}"
        self._visit(first.id, trigger.id)
        self._emit(detected_at, SimEventKind.DETECTED, note, actor=trigger.id, target=self.sid, edge=first.id)
        return detected_at, seeds

    # ------------------------------------------------------------------ paging
    def _page(
        self, seeds: Sequence[_Seed], purpose: _Purpose, *, anchor: str, t: float
    ) -> _Reached | _DeadEnd:
        """Discrete-event paging with escalation; the first ACK wins (see module docstring).

        Pages from the seed list are distinct calls (a second alarm pages again), so they are
        deduplicated only when identical (node, minute, channel). Escalation hops are
        deduplicated per node: the earliest one wins and a visited node is never re-entered.
        """
        order = itertools.count()
        heap: list[_Hop] = []
        best: dict[str, float] = {}
        parent: dict[str, str | None] = {}
        visited: set[str] = set()
        tried: set[tuple[str, float, str | None]] = set()

        for s in seeds:
            heapq.heappush(
                heap,
                _Hop(s.t, next(order), s.node_id, s.via_edge, s.source_id, None, None, s.channel_hint, 0),
            )
        dead_end = _DeadEnd(anchor, t, purpose.nobody)
        while heap:
            hop = heapq.heappop(heap)
            if hop.after is None:
                key = (hop.node_id, hop.t, None if hop.channel_hint is None else hop.channel_hint[0])
                if key in tried:
                    continue  # the very same call again changes nothing
                tried.add(key)
            elif hop.node_id in visited or hop.t > best[hop.node_id]:
                continue  # escalation never re-enters a node; a later hop was superseded
            parent.setdefault(hop.node_id, hop.parent)
            visited.add(hop.node_id)
            self._enter(hop, purpose)
            reached = self._try(hop, purpose)
            if reached is not None:
                return reached
            blocked: list[str] = []
            for edge in self._escalation_edges(hop.node_id):
                after = _minutes(edge.attr(AttrKey.AFTER_MINUTES))
                if after is None:
                    after = self.config.default_escalation_after_minutes  # validated by _check_config
                if after is None:
                    blocked.append(_NO_TIMING)
                elif edge.dst in visited:
                    blocked.append(_CYCLE if _is_ancestor(parent, edge.dst, hop.node_id) else _REVISIT)
                elif hop.hops >= self.config.max_hops:
                    blocked.append(f"최대 상향 단계({self.config.max_hops}) 초과")
                elif hop.t + after < best.get(edge.dst, math.inf):
                    best[edge.dst] = hop.t + after
                    heapq.heappush(
                        heap,
                        _Hop(
                            hop.t + after,
                            next(order),
                            edge.dst,
                            edge.id,
                            edge.src,
                            hop.node_id,
                            after,
                            None,
                            hop.hops + 1,
                        ),
                    )
            dead_end = _DeadEnd(hop.node_id, hop.t, ", ".join(dict.fromkeys(blocked)) or _NO_ESCALATION)
        note = f"{purpose.label} 중단: {self._name(dead_end.last_id)} 이후 진행 불가 — {dead_end.reason}"
        self._emit(dead_end.t, SimEventKind.DEAD_END, note, target=dead_end.last_id)
        return dead_end

    def _enter(self, hop: _Hop, purpose: _Purpose) -> None:
        if hop.after is not None:
            stated = _minutes(self.graph.edges[hop.via_edge].attr(AttrKey.AFTER_MINUTES)) is not None
            note = (
                f"{purpose.label}: {self._name(hop.source_id)} 미응답 {_fmt(hop.after)}분 → "
                f"{_attach(self._name(hop.node_id), _TOWARD)} 상향{'' if stated else _WAIT_ASSUMPTION}"
            )
            self._emit(
                hop.t, SimEventKind.ESCALATE, note, actor=hop.source_id, target=hop.node_id, edge=hop.via_edge
            )
        self._visit(self.lead_in.get(hop.source_id), hop.source_id, hop.via_edge, hop.node_id)

    def _escalation_edges(self, node_id: str) -> list[Edge]:
        sources = [node_id, *(p.id for p in self.ctx.resolve_people(node_id) if p.id != node_id)]
        edges: dict[str, Edge] = {}
        for source in sources:
            for edge in self.ctx.escalations(source):
                edges.setdefault(edge.id, edge)
        return list(edges.values())

    def _membership_edge(self, person_id: str, group_id: str) -> str | None:
        if person_id == group_id:
            return None
        links = self.graph.in_edges(group_id, EdgeKind.HOLDS_ROLE, EdgeKind.MEMBER_OF)
        return next((e.id for e in links if e.src == person_id), None)

    def _try(self, hop: _Hop, purpose: _Purpose) -> _Reached | None:
        people = self.ctx.resolve_people(hop.node_id)
        if not people:
            note = (
                f"{purpose.label}: {self._name(hop.node_id)}에 해당하는 사람이 없음(역할 보유자·팀원 미기재)"
            )
            self._emit(hop.t, SimEventKind.NO_ANSWER, note, target=hop.node_id, edge=hop.via_edge)
            return None
        for person in people:
            self._visit(self._membership_edge(person.id, hop.node_id), person.id)
            blocked = self.avail.person_reason(person, hop.t)
            if blocked is not None:
                note = f"{purpose.label}: {person.name} 응답 불가 — {blocked}"
                self._emit(hop.t, SimEventKind.NO_ANSWER, note, target=person.id, edge=hop.via_edge)
                continue
            channel = self._reach(person, hop, purpose)
            if channel is None:
                continue
            role = "" if person.id == hop.node_id else f"({self._name(hop.node_id)})"
            note = f"{purpose.label}: {person.name}{role}에게 {_attach(channel.name, _TOWARD)} 호출"
            self._emit(hop.t, SimEventKind.PAGE, note, actor=channel.id, target=person.id, edge=hop.via_edge)
            ack_t = hop.t + self.config.ack_minutes
            note = f"{purpose.label}: {_attach(person.name, _SUBJECT)} 응답"
            self._emit(ack_t, SimEventKind.ACK, note, actor=person.id, target=hop.node_id, edge=hop.via_edge)
            return _Reached(person, hop.node_id, hop.via_edge, ack_t)
        return None

    def _channel_candidates(self, person: Node, hop: _Hop) -> list[tuple[str, str]]:
        """(channel id, edge that offered it): hint, person's, group's — each + FALLBACK_TO chain."""
        roots = _contact_roots(self.graph, person.id, hop.node_id, hop.channel_hint)
        ordered: list[tuple[str, str]] = []
        seen: set[str] = set()
        stack = list(reversed(roots))
        while stack:
            channel_id, via = stack.pop()
            if channel_id in seen:
                continue
            seen.add(channel_id)
            ordered.append((channel_id, via))
            fallbacks = self.graph.out_edges(channel_id, EdgeKind.FALLBACK_TO)
            stack.extend((e.dst, e.id) for e in reversed(fallbacks))
        return ordered

    def _reach(self, person: Node, hop: _Hop, purpose: _Purpose) -> Node | None:
        candidates = self._channel_candidates(person, hop)
        for channel_id, via in candidates:
            channel = self.graph.nodes[channel_id]
            self._visit(via, channel_id)
            down = self.avail.channel_reason(channel_id, hop.t)
            if down is None:
                return channel
            note = f"{purpose.label}: {person.name}의 {channel.name} 사용 불가 — {down}"
            self._emit(
                hop.t, SimEventKind.CHANNEL_UNAVAILABLE, note, actor=channel_id, target=person.id, edge=via
            )
        why = "모든 연락 수단 사용 불가" if candidates else "연락 수단 미기재"
        note = f"{purpose.label}: {person.name}에게 닿을 수 없음 — {why}"
        self._emit(hop.t, SimEventKind.NO_ANSWER, note, target=person.id, edge=hop.via_edge)
        return None

    # ------------------------------------------------------------------ mitigation
    def _mitigate(self, responder: _Reached) -> float | None:
        link = choose_mitigation(self.ctx, self.sid)
        if link is None:
            note = f"{self._name(self.sid)}에 완화 조치(롤백·페일오버 등)가 없음"
            self._emit(
                responder.ack_t, SimEventKind.NO_MITIGATION, note, actor=responder.person.id, target=self.sid
            )
            self._fail(Element.ROLLBACK, self.sid, note)
            return None
        action = self.graph.nodes[link.dst]
        self._visit(link.id, action.id)
        start = responder.ack_t
        if requires_approval(self.ctx, action):
            approved = self._approve(action, responder)
            if approved is None:
                return None
            start = approved
        executor, exec_edge, note = self._executor(action, responder.person, start)
        self._emit(
            start,
            SimEventKind.ACTION_STARTED,
            note,
            actor=executor.id,
            target=action.id,
            edge=exec_edge or link.id,
        )
        duration = _minutes(action.attr(AttrKey.DURATION_MINUTES))
        assumed = " — 소요 시간 미기재로 기본값 가정" if duration is None else ""
        done = start + (self.config.default_action_minutes if duration is None else duration)
        note = f"{action.name} 완료 — 서비스 완화(조치 {_fmt(done - start)}분{assumed})"
        self._emit(done, SimEventKind.MITIGATED, note, actor=executor.id, target=self.sid, edge=link.id)
        return done

    def _approve(self, action: Node, responder: _Reached) -> float | None:
        links = self.graph.out_edges(action.id, EdgeKind.APPROVED_BY)
        me = responder.person
        for link in links:
            if any(p.id == me.id for p in self.ctx.resolve_people(link.dst)):
                self._visit(link.id, link.dst)
                note = f"{_attach(me.name, _SUBJECT)} 승인 권한 보유({self._name(link.dst)}) — 직접 승인"
                self._emit(
                    responder.ack_t,
                    SimEventKind.APPROVAL_REQUESTED,
                    note,
                    actor=me.id,
                    target=action.id,
                    edge=link.id,
                )
                return self._approved(action, me, link.id, responder.ack_t)
        seeds = [_Seed(link.dst, link.id, action.id, responder.ack_t) for link in links]
        outcome = self._page(seeds, _APPROVAL, anchor=action.id, t=responder.ack_t)
        if isinstance(outcome, _DeadEnd):
            note = f"{action.name} 승인 불가 — {outcome.reason}"
            self._emit(outcome.t, SimEventKind.APPROVAL_BLOCKED, note, target=action.id)
            self._fail(
                Element.APPROVER,
                outcome.last_id,
                f"{action.name} 승인이 {self._name(outcome.last_id)}에서 막힘 — {outcome.reason}",
            )
            return None
        approver = outcome.person
        note = f"{_attach(me.name, _SUBJECT)} {approver.name}에게 {action.name} 승인 요청"
        self._emit(
            outcome.ack_t,
            SimEventKind.APPROVAL_REQUESTED,
            note,
            actor=me.id,
            target=approver.id,
            edge=outcome.via_edge,
        )
        return self._approved(action, approver, outcome.via_edge, outcome.ack_t)

    def _approved(self, action: Node, approver: Node, edge: str, requested_at: float) -> float:
        approved_at = requested_at + self.config.approval_minutes
        note = f"{_attach(approver.name, _SUBJECT)} {action.name} 승인"
        self._emit(approved_at, SimEventKind.APPROVED, note, actor=approver.id, target=action.id, edge=edge)
        return approved_at

    def _executor(self, action: Node, responder: Node, t: float) -> tuple[Node, str | None, str]:
        links = self.graph.out_edges(action.id, EdgeKind.EXECUTED_BY)
        for link in links:
            for person in self.ctx.resolve_people(link.dst):
                if self.avail.person_reason(person, t) is None:
                    self._visit(link.id, link.dst, self._membership_edge(person.id, link.dst), person.id)
                    return person, link.id, f"{_attach(person.name, _SUBJECT)} {action.name} 실행 시작"
        why = "지정 실행자 응답 불가" if links else "실행자 미지정"
        return responder, None, f"{action.name} 실행 시작 — {why} → 응답자가 실행(가정)"

    # ------------------------------------------------------------------ reporting
    def _report(self, detected_at: float) -> int:
        """Simulate legal/contractual reporting; return how many reports were filed in time."""
        incident = self.scenario.incident_type
        if incident not in _REPORTABLE:
            return 0
        rules = applicable_obligation_rules(self.ctx, self.sid, incident)
        duties = self.ctx.obligations(self.sid)
        matching = [o for o in duties if _incident_type(o) is incident]
        if not matching:
            untyped = [o.name for o in duties if _incident_type(o) is None]
            for rule in rules:
                gap = (
                    f"신고 의무({', '.join(untyped)})의 사고 유형이 미기재·인식 불가라 "
                    f"{rule.recipient} 신고 절차인지 확인 불가"
                    if untyped
                    else f"{rule.recipient} 신고 절차가 런북에 없음"
                )
                note = f"{gap} (근거: {rule.legal_basis})"
                self._emit(detected_at, SimEventKind.REPORT_MISSED, note, target=self.sid)
                self._fail(None, self.sid, note)
            return 0
        law_deadline = min((float(r.deadline_minutes) for r in rules), default=None)
        law_basis = " / ".join(r.legal_basis for r in rules)
        return sum(self._file_report(o, detected_at, law_deadline, law_basis) for o in matching)

    def _file_report(
        self, obligation: Node, detected_at: float, law_deadline: float | None, law_basis: str
    ) -> bool:
        # an org-wide duty has no MUST_REPORT edge to highlight; a linked one may have parallel ones
        links = [e.id for e in self.graph.in_edges(obligation.id, EdgeKind.MUST_REPORT) if e.src == self.sid]
        self._visit(*links, obligation.id)
        own_basis = obligation.attr(AttrKey.LEGAL_BASIS)
        basis = law_basis or (own_basis if isinstance(own_basis, str) else "")
        cite = f" (근거: {basis})" if basis else ""
        deadline = _stricter(_minutes(obligation.attr(AttrKey.DEADLINE_MINUTES)), law_deadline)
        if deadline is None:
            note = f"{obligation.name}: 신고 시한 미기재 — 기한 내 신고 여부 확인 불가{cite}"
            return self._report_missed(detected_at, obligation.id, obligation.id, note)
        seeds = [
            _Seed(e.dst, e.id, obligation.id, detected_at)
            for e in self.graph.out_edges(obligation.id, EdgeKind.REPORTED_BY)
        ]
        outcome = self._page(seeds, _REPORTING, anchor=obligation.id, t=detected_at)
        if isinstance(outcome, _DeadEnd):
            note = f"{obligation.name}: 신고 담당에게 닿지 못함 — {outcome.reason}{cite}"
            return self._report_missed(outcome.t, obligation.id, outcome.last_id, note)
        reporter = outcome.person
        note = f"{obligation.name}: {_attach(reporter.name, _SUBJECT)} 신고"
        self._emit(
            outcome.ack_t,
            SimEventKind.REPORT_FILED,
            note,
            actor=reporter.id,
            target=obligation.id,
            edge=outcome.via_edge,
        )
        due = detected_at + deadline
        if outcome.ack_t > due:
            note = (
                f"{obligation.name}: 신고 시한 {_duration(deadline)} 초과"
                f"(인지 후 {_duration(outcome.ack_t - detected_at)}){cite}"
            )
            # a malformed (negative) policy deadline is due at detection, never before it
            return self._report_missed(max(due, detected_at), obligation.id, obligation.id, note)
        return True

    def _report_missed(self, t: float, obligation_id: str, at_id: str, note: str) -> bool:
        self._emit(t, SimEventKind.REPORT_MISSED, note, target=obligation_id)
        self._fail(None, at_id, note)
        return False

    # ------------------------------------------------------------------ verdict
    def _result(self, mitigated_at: float | None, filed: int) -> RehearsalResult:
        first = self.failures[0] if self.failures else None
        if first is None:
            reason = f"목표 복구 시간 {_fmt(self.deadline)}분 안에 완화({_fmt(mitigated_at or 0.0)}분)"
            if filed:
                reason += f" · 신고 {filed}건 기한 내 완료"
        else:
            reason = " / ".join(dict.fromkeys(f.reason for f in self.failures))
        scenario = self.scenario
        return RehearsalResult(
            scenario_id=scenario.id,
            scenario_fingerprint=scenario.fingerprint(),
            graph_hash=self.replayer.graph_hash,
            config_hash=self.replayer.config_hash,
            outcome=Outcome.PASS if first is None else Outcome.FAIL,
            time_to_mitigate=mitigated_at,
            deadline_minutes=self.deadline,
            failed_element=None if first is None else first.element,
            failed_at_id=None if first is None else first.at_id,
            reason=reason,
            events=tuple(sorted(self.events, key=lambda e: e.t)),
            path_ids=tuple(self.path),
        )
