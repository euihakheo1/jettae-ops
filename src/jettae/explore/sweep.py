"""Exhaustive sweep (결정적 전수 탐색) — risks the fixed rules do not ask about, found without an LLM.

The fixed rules ask closed questions; ``JT-SPF-001/002`` for instance report every person or
channel whose loss *alone* breaks a critical service. They do not ask about pairs, about the
centre that takes two channels down at once, about a colleague busy with a concurrent incident,
or about a deputy who exists on paper but is paged too late to mitigate in time. The sweep asks
exactly those questions, for every critical service, and lets the deterministic replay answer:

1. **Baseline.** The service is replayed without any perturbation. If that already fails, the
   fixed rules report the break and every combination would "break" it trivially, so the service
   is not searched (the reason is published in the summary). A held (판단 보류) service is not
   replayed at all — the replay leaves held facts out, so its "failure" would only mean "not
   confirmed" (P3) — and is listed as not searched, with a note.
2. **Search space.** Only what the service's response can depend on: the people and channels of
   its response lineage (:func:`jettae.check.spof.response_lineage` — owners, paged targets,
   escalation closure, approvers, their channels and fallbacks), the systems and locations those
   channels depend on (taking one down takes its channels down in the replay), and the other
   critical services that share a first responder (a concurrent incident pulls that person away).
   Units on the baseline replay's path come first, then by kind and id; at most
   :attr:`SweepLimits.max_candidates` units per service.
3. **Effects, not names.** Sets are compared by what the replay loses under them
   (:func:`perturbation_effect`): people and channels. A system or centre is the channels it takes
   down; a concurrent incident is *the person it pulls away* — exactly as the replay resolves it,
   the first of that service's first responders the set has not already taken. A concurrent
   incident that only pulls away someone a person unit already stands for is therefore that
   person's absence, not a new risk (in the demo every one of them is).
4. **Minimal breaking sets.** Single units, then pairs (up to :attr:`SweepLimits.max_set_size`),
   are replayed with the *service's policy deadline*. A set is skipped when what it takes down
   already contains a breaking set found earlier or equals a set already replayed, so every
   reported set is minimal: no strict subset breaks the response, and two sets with the same
   effect (a centre that hosts only one channel, a concurrent incident and the absence of the
   person it pulls away) are not reported twice.
5. **Rules first.** A breaking set whose every target — a concurrent incident counted as the person
   it pulls away — is already named by a fixed-rule finding of the same service (e.g. a size-1
   ``JT-SPF`` finding) is recorded as *covered* instead of being proposed again.

Bounds are published, never silent: the candidate cap per service, the replay budget per run and
the hypothesis caps (per service, per run) each leave a Korean note when they cut the search. The sweep only
sees dependencies the graph states; a shared network nobody wrote down is invisible to it (the AI
explorer may propose such a premise, which then stays ``needs_info``).

No LLM, no I/O, deterministic order everywhere (P1, P6).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Final

from jettae.check.context import CheckContext
from jettae.check.elements import josa
from jettae.check.spof import GraphIndex, ResponseLineage, response_lineage
from jettae.domain import (
    CoveredCut,
    Finding,
    HypothesisOrigin,
    IncidentType,
    Node,
    NodeKind,
    Outcome,
    Perturbation,
    PerturbationKind,
    RehearsalResult,
    Scenario,
    ScenarioOrigin,
    ServiceSweep,
    SimConfig,
)
from jettae.explore.drafts import (
    KIND_ORDER,
    HypothesisDraft,
    canonical_perturbations,
    join_ko,
    perturbation_label,
    set_key,
)
from jettae.rehearse.simulator import can_respond, first_responders, replay_all, trusted_view

__all__ = [
    "BASELINE_FAILED_REASON",
    "DEFAULT_LIMITS",
    "HELD_SERVICE_REASON",
    "SWEEP_SCENARIO_PREFIX",
    "SweepLimits",
    "SweepResult",
    "known_breaks",
    "perturbation_effect",
    "sweep",
]

#: Why a service whose baseline replay fails is not searched (published in the summary).
BASELINE_FAILED_REASON: Final = (
    "교란 없이도 리허설이 실패해 탐색하지 않았습니다(고정 규칙이 보고한 끊김을 먼저 고쳐야 합니다)."
)
#: Why a held (판단 보류) critical service is not searched: the replay does not use held facts.
HELD_SERVICE_REASON: Final = (
    "판단 보류(원문 대조 전) 서비스라 리허설이 쓰지 않아 탐색하지 않았습니다. 확인한 뒤 다시 진단해야 합니다."
)
#: Id prefix of the throw-away scenarios the sweep replays (never stored or shown as drills).
SWEEP_SCENARIO_PREFIX: Final = "explore:sweep:"

_KIND_OF_NODE: Final[dict[NodeKind, PerturbationKind]] = {
    NodeKind.PERSON: PerturbationKind.PERSON_UNAVAILABLE,
    NodeKind.CHANNEL: PerturbationKind.CHANNEL_DOWN,
    NodeKind.SYSTEM: PerturbationKind.SYSTEM_DOWN,
    NodeKind.LOCATION: PerturbationKind.LOCATION_DOWN,
    NodeKind.SERVICE: PerturbationKind.CONCURRENT_INCIDENT,
}


@dataclass(frozen=True)
class SweepLimits:
    """Published bounds of one sweep (docs/explore.md). Each cut leaves a note."""

    #: largest combination searched (1 = single points only, 2 = pairs)
    max_set_size: int = 2
    #: perturbation units per service (people, channels, systems, locations, concurrent services)
    max_candidates: int = 20
    #: replays per run, baselines included (bounds the CPU a diagnosis spends on exploration)
    max_replays: int = 1500
    #: hypotheses proposed per service (smallest sets first)
    max_hypotheses_per_service: int = 12
    #: hypotheses proposed per run (smallest sets first, then service order) — bounds the report
    max_hypotheses: int = 60

    def __post_init__(self) -> None:
        if not 1 <= self.max_set_size <= 3:
            raise ValueError("max_set_size는 1~3이어야 합니다.")
        bounds = (self.max_candidates, self.max_replays, self.max_hypotheses_per_service, self.max_hypotheses)
        if min(bounds) < 1:
            raise ValueError("탐색 상한은 1 이상이어야 합니다.")


DEFAULT_LIMITS: Final = SweepLimits()


@dataclass(frozen=True)
class SweepResult:
    drafts: tuple[HypothesisDraft, ...]
    services: tuple[ServiceSweep, ...]
    covered: tuple[CoveredCut, ...]
    replays: int
    notes: tuple[str, ...]
    #: every breaking set found (proposed or covered), as :func:`~jettae.explore.drafts.set_key`
    found: frozenset[tuple[str, str, tuple[tuple[str, str], ...]]]
    max_set_size: int
    #: every breaking set found (proposed or covered) as (service id, what it takes down — see
    #: :func:`perturbation_effect`), all for an outage
    found_effects: frozenset[tuple[str, frozenset[str]]] = frozenset()


@dataclass(frozen=True)
class _Unit:
    """One thing the sweep can take away, and what the replay loses with it."""

    node_id: str
    kind: PerturbationKind
    #: people/channels it takes away on its own (a concurrent incident: the responder it pulls)
    effect: frozenset[str]
    on_path: bool
    #: concurrent incident only: the other service's first responders, in the replay's paging order
    responders: tuple[str, ...] = ()

    def order(self) -> tuple[int, int, str]:
        return (0 if self.on_path else 1, KIND_ORDER[self.kind], self.node_id)

    def perturbation(self) -> Perturbation:
        return Perturbation(kind=self.kind, target_id=self.node_id)


def _resolve(units: Iterable[_Unit]) -> tuple[frozenset[str], dict[str, str | None]]:
    """What a set takes down, and whom each of its concurrent incidents pulls away (``None``: nobody).

    Mirrors the replay (:mod:`jettae.rehearse.simulator`): absences and outages first, then the
    concurrent incidents in target-id order, each occupying the first of its service's first
    responders that neither an absence of the set nor an earlier concurrent incident has taken.
    """
    atoms: set[str] = set()
    concurrent: list[_Unit] = []
    for unit in units:
        if unit.kind is PerturbationKind.CONCURRENT_INCIDENT:
            concurrent.append(unit)
        else:
            atoms |= unit.effect
    pulled: dict[str, str | None] = {}
    for unit in sorted(concurrent, key=lambda u: u.node_id):
        busy = next((pid for pid in unit.responders if pid not in atoms), None)
        pulled[unit.node_id] = busy
        if busy is not None:
            atoms.add(busy)
    return frozenset(atoms), pulled


def _replay_context(ctx: CheckContext) -> CheckContext:
    """The context the replay itself judges with: confirmed facts, the policy and the roster only.

    Whom a concurrent incident pulls away is computed on it, so the sweep names the same person the
    replay blocks.
    """
    return CheckContext(graph=trusted_view(ctx.graph)[0], policy=ctx.policy, roster=ctx.roster)


def _effect_in(
    view: CheckContext, perturbations: Iterable[Perturbation], config: SimConfig
) -> frozenset[str]:
    """:func:`perturbation_effect` on a context already reduced to what the replay judges with."""
    graph = view.graph
    units: list[_Unit] = []
    for p in canonical_perturbations(perturbations):
        if p.kind is PerturbationKind.PERSON_UNAVAILABLE:
            effect = frozenset(person.id for person in view.resolve_people(p.target_id))
        elif p.kind is PerturbationKind.CHANNEL_DOWN:
            effect = frozenset({p.target_id}) if p.target_id in graph.nodes else frozenset()
        elif p.kind is PerturbationKind.CONCURRENT_INCIDENT:
            responders = tuple(person.id for person in first_responders(view, p.target_id, config=config))
            units.append(_Unit(p.target_id, p.kind, frozenset(responders[:1]), False, responders))
            continue
        else:
            effect = frozenset(
                c.id for c in graph.nodes_of(NodeKind.CHANNEL) if p.target_id in view.failure_domain(c.id)
            )
        units.append(_Unit(p.target_id, p.kind, effect, False))
    return _resolve(units)[0]


def perturbation_effect(
    ctx: CheckContext, perturbations: Iterable[Perturbation], *, config: SimConfig | None = None
) -> frozenset[str]:
    """People and channels the replay loses under whole-incident ``perturbations``.

    A person absence is that person (a role or team: its people), a channel outage the channel, a
    system or centre outage every channel whose failure domain contains it, and a concurrent
    incident the first responder it pulls away — after the absences, in target-id order, as the
    replay resolves it. Held (판단 보류) facts are not used, as in the replay. Two sets with the
    same effect replay the same way; a set whose effect contains a breaking set's breaks too.
    """
    return _effect_in(_replay_context(ctx), perturbations, config if config is not None else SimConfig())


def known_breaks(
    ctx: CheckContext,
    swept: SweepResult,
    drafts: Iterable[HypothesisDraft],
    *,
    config: SimConfig | None = None,
) -> frozenset[str]:
    """Ids of the drafts that a breaking set the sweep already found accounts for.

    A draft without premises, for an outage of a service the sweep searched, whose perturbations
    take down (:func:`perturbation_effect`) everything a breaking set of that service takes down
    breaks *because of* that set — proposed by the sweep or already reported by a rule. It is not a
    new risk: the same set again, a known set with extra targets, and a concurrent incident that
    only pulls away a known single point of failure are all caught here. A draft naming a node the
    replay does not use (held or unknown) is left to the verification, which says what to check.
    """
    by_service: dict[str, list[frozenset[str]]] = {}
    for service_id, effect in swept.found_effects:
        by_service.setdefault(service_id, []).append(effect)
    view = _replay_context(ctx)
    sim = config if config is not None else SimConfig()
    known: set[str] = set()
    for draft in drafts:
        cuts = by_service.get(draft.service_id)
        if not cuts or draft.assumptions or draft.incident_type is not IncidentType.OUTAGE:
            continue
        if any(target not in view.graph.nodes for target in draft.target_ids):
            continue
        effect = _effect_in(view, draft.perturbations, sim)
        if any(cut <= effect for cut in cuts):
            known.add(draft.id)
    return frozenset(known)


@dataclass(frozen=True)
class _Cut:
    units: tuple[_Unit, ...]
    effect: frozenset[str]
    result: RehearsalResult


class _Budget:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0

    @property
    def left(self) -> int:
        return max(self.limit - self.used, 0)


def _scenario(service_id: str, units: Sequence[_Unit]) -> Scenario:
    key = "+".join(u.node_id for u in units) or "baseline"
    return Scenario(
        id=f"{SWEEP_SCENARIO_PREFIX}{service_id}:{key}",
        title=f"{service_id} 전수 탐색",
        service_id=service_id,
        incident_type=IncidentType.OUTAGE,
        perturbations=canonical_perturbations(u.perturbation() for u in units),
        origin=ScenarioOrigin.TEMPLATE,
    )


def _replay(ctx: CheckContext, scenarios: Sequence[Scenario], config: SimConfig) -> list[RehearsalResult]:
    return replay_all(ctx.graph, scenarios, policy=ctx.policy, config=config, roster=ctx.roster)


def _units(
    view: CheckContext,
    *,
    service: Node,
    lineage: ResponseLineage,
    baseline: RehearsalResult,
    critical: Sequence[Node],
    config: SimConfig,
) -> list[_Unit]:
    """Every unit the service's response can depend on, deduplicated by effect, in search order.

    A concurrent incident's effect on its own is the first first-responder it pulls away; when a
    person unit already stands for that person, the person is kept and the incident dropped (in
    any larger set it would only pull away someone else the person units already stand for).
    """
    graph = view.graph
    replay_view = _replay_context(view)
    path = set(baseline.path_ids)
    people = [
        nid
        for nid in lineage.candidates
        if graph.nodes[nid].kind is NodeKind.PERSON and can_respond(view, graph.nodes[nid])
    ]
    channels = [nid for nid in lineage.candidates if graph.nodes[nid].kind is NodeKind.CHANNEL]
    units = [
        _Unit(nid, _KIND_OF_NODE[graph.nodes[nid].kind], frozenset({nid}), nid in path)
        for nid in (*people, *channels)
    ]
    hosted: dict[str, set[str]] = {}
    for cid in channels:
        for dep in view.failure_domain(cid):
            if graph.nodes[dep].kind in (NodeKind.SYSTEM, NodeKind.LOCATION):
                hosted.setdefault(dep, set()).add(cid)
    for dep, hit in sorted(hosted.items()):
        units.append(_Unit(dep, _KIND_OF_NODE[graph.nodes[dep].kind], frozenset(hit), bool(hit & path)))
    mine = set(people)
    for other in critical:
        if other.id == service.id:
            continue
        responders = tuple(p.id for p in first_responders(replay_view, other.id, config=config))
        shared = set(responders) & mine
        if shared:
            units.append(
                _Unit(
                    other.id,
                    PerturbationKind.CONCURRENT_INCIDENT,
                    frozenset(responders[:1]),
                    bool(shared & path),
                    responders,
                )
            )
    seen: set[frozenset[str]] = set()
    unique: list[_Unit] = []
    # Same effect = same replay: keep the most direct unit (a channel over the centre hosting it,
    # a person over the concurrent incident that would pull exactly that person away).
    for unit in sorted(units, key=lambda u: (KIND_ORDER[u.kind], u.node_id)):
        if unit.effect in seen:
            continue
        seen.add(unit.effect)
        unique.append(unit)
    return sorted(unique, key=_Unit.order)


def _rule_targets(units: Sequence[_Unit]) -> tuple[str, ...]:
    """A set's targets as the rules name them: a concurrent incident is the person it pulls away."""
    _, pulled = _resolve(units)
    targets: list[str] = []
    for unit in units:
        if unit.kind is not PerturbationKind.CONCURRENT_INCIDENT:
            targets.append(unit.node_id)
        elif (busy := pulled.get(unit.node_id)) is not None:
            targets.append(busy)
    return tuple(dict.fromkeys(targets))


def _covering(findings: Sequence[Finding], service_id: str, target_ids: Iterable[str]) -> tuple[str, ...]:
    """Ids of the service's findings that already name every target of a breaking set."""
    targets = set(target_ids)
    if not targets:
        return ()
    return tuple(
        sorted(f.id for f in findings if f.service_id == service_id and targets <= set(f.subject_ids))
    )


def _structural(lineage: ResponseLineage, effect: frozenset[str]) -> bool:
    """Does the set cut the response lineage itself (no one / no channel left to answer or approve)?"""
    return not lineage.satisfied(effect)


_SEARCHED: Final[dict[int, str]] = {1: "하나씩", 2: "하나씩, 둘씩", 3: "하나씩, 둘씩, 셋씩"}


def _busy_notes(view: CheckContext, cut: _Cut) -> list[str]:
    """Who each concurrent incident pulls away (the first first-responder the set leaves free)."""
    _, pulled = _resolve(cut.units)
    notes: list[str] = []
    for unit in cut.units:
        busy = pulled.get(unit.node_id) if unit.kind is PerturbationKind.CONCURRENT_INCIDENT else None
        if busy is not None:
            other = view.graph.nodes[unit.node_id].name
            name = view.graph.nodes[busy].name
            notes.append(f"'{other}' 장애가 함께 나면 그쪽 1차 대응자인 {josa(name, '이', '가')} 투입됩니다.")
    return notes


def _draft(
    view: CheckContext,
    service: Node,
    cut: _Cut,
    *,
    lineage: ResponseLineage,
    searched: int,
    max_size: int,
) -> HypothesisDraft:
    graph = view.graph
    perturbations = canonical_perturbations(u.perturbation() for u in cut.units)
    labels = [perturbation_label(graph, p) for p in perturbations]
    title = f"{service.name} · {join_ko(labels)}"
    if len(cut.units) == 1:
        lead = f"{labels[0]} 하나만으로 '{service.name}' 대응이 끊깁니다."
    else:
        lead = (
            f"{', '.join(labels)} 중 하나만 생기면 대응이 이어지지만, 함께 생기면 '{service.name}' "
            "대응이 끊깁니다."
        )
    if _structural(lineage, cut.effect):
        why = "대신 응답하거나 승인할 사람·연락 수단이 남지 않습니다."
    else:
        why = (
            "구조상으로는 다른 사람·연락 수단이 남아 있지만, 리허설에서는 상향 시점·응답 순서·"
            "목표 복구 시간 때문에 시간 안에 완화하지 못합니다."
        )
    parts = [
        f"자동 전수 탐색: '{service.name}' 대응 경로에 걸린 대상 {searched}개를 "
        f"{_SEARCHED[max_size]} 조합해 리허설로 찾은 최소 조합입니다.",
        lead,
        *_busy_notes(view, cut),
        why,
    ]
    hosted = [
        f"{graph.nodes[u.node_id].name} → {', '.join(sorted(graph.nodes[c].name for c in u.effect))}"
        for u in cut.units
        if u.kind in (PerturbationKind.SYSTEM_DOWN, PerturbationKind.LOCATION_DOWN)
    ]
    if hosted:
        parts.append(f"함께 멈추는 연락 수단: {'; '.join(hosted)}.")
    return HypothesisDraft(
        origin=HypothesisOrigin.SWEEP,
        service_id=service.id,
        title=title,
        explanation=" ".join(parts),
        perturbations=perturbations,
    )


def _search(
    ctx: CheckContext,
    service_id: str,
    units: Sequence[_Unit],
    *,
    max_size: int,
    budget: _Budget,
    config: SimConfig,
) -> tuple[list[_Cut], int, bool]:
    """Minimal breaking sets by increasing size; returns (cuts, replays used, budget ran out).

    A combination is replayed only when what it takes down neither contains a breaking set found
    so far nor equals a set already replayed (same effect, same replay). After each size, a set
    whose effect strictly contains another breaking set's is dropped, so what remains is minimal.
    Within a size, sets without a concurrent incident go first, so of two sets with the same
    effect the one naming the person directly is the one replayed and reported.
    """
    cuts: list[_Cut] = []
    tested: set[frozenset[str]] = set()
    used = 0
    for size in range(1, max_size + 1):
        batch: list[tuple[tuple[_Unit, ...], frozenset[str]]] = []
        combos = sorted(
            itertools.combinations(units, size),
            key=lambda combo: sum(u.kind is PerturbationKind.CONCURRENT_INCIDENT for u in combo),
        )
        for combo in combos:
            effect = _resolve(combo)[0]
            if effect in tested or any(cut.effect <= effect for cut in cuts):
                continue
            tested.add(effect)
            batch.append((combo, effect))
        exhausted = len(batch) > budget.left
        batch = batch[: budget.left]
        results = _replay(ctx, [_scenario(service_id, combo) for combo, _ in batch], config)
        used += len(results)
        budget.used += len(results)
        found = [
            _Cut(combo, effect, result)
            for (combo, effect), result in zip(batch, results, strict=True)
            if result.outcome is Outcome.FAIL
        ]
        cuts += [cut for cut in found if not any(other.effect < cut.effect for other in found)]
        if exhausted:
            return cuts, used, True
    return cuts, used, False


@dataclass
class _Tally:
    """What the sweep has gathered so far (one per run, filled service by service)."""

    notes: list[str] = field(default_factory=list)
    stats: list[ServiceSweep] = field(default_factory=list)
    drafts: list[HypothesisDraft] = field(default_factory=list)
    covered: list[CoveredCut] = field(default_factory=list)
    found: set[tuple[str, str, tuple[tuple[str, str], ...]]] = field(default_factory=set)
    found_effects: set[tuple[str, frozenset[str]]] = field(default_factory=set)
    out_of_budget: list[str] = field(default_factory=list)


def _skip(tally: _Tally, svc: Node, baseline: Outcome | None, reason: str) -> None:
    replays = 0 if baseline is None else 1
    tally.stats.append(
        ServiceSweep(service_id=svc.id, baseline=baseline, replays=replays, skipped_reason=reason)
    )


def _sweep_service(
    ctx: CheckContext,
    view: CheckContext,
    svc: Node,
    baseline: RehearsalResult,
    findings: Sequence[Finding],
    *,
    index: GraphIndex,
    critical: Sequence[Node],
    budget: _Budget,
    limits: SweepLimits,
    config: SimConfig,
    tally: _Tally,
) -> None:
    """Search one service whose baseline passes; add its hypotheses, covered sets and stats."""
    lineage = response_lineage(view, svc.id, index=index)
    units = _units(view, service=svc, lineage=lineage, baseline=baseline, critical=critical, config=config)
    total = len(units)
    units = units[: limits.max_candidates]
    if total > len(units):
        tally.notes.append(
            f"{svc.name}: 대응 경로에 걸린 대상 {total}개 중 경로에 가까운 {len(units)}개만 "
            "탐색했습니다(상한)."
        )
    cuts, used, exhausted = _search(
        ctx, svc.id, units, max_size=limits.max_set_size, budget=budget, config=config
    )
    if exhausted:
        tally.out_of_budget.append(svc.name)
    proposed: list[HypothesisDraft] = []
    for cut in cuts:
        targets = [u.node_id for u in cut.units]
        tally.found.add(set_key(svc.id, IncidentType.OUTAGE, (u.perturbation() for u in cut.units)))
        tally.found_effects.add((svc.id, cut.effect))
        covering = _covering(findings, svc.id, _rule_targets(cut.units))
        if covering:
            tally.covered.append(
                CoveredCut(service_id=svc.id, target_ids=tuple(targets), finding_ids=covering)
            )
            continue
        proposed.append(
            _draft(view, svc, cut, lineage=lineage, searched=len(units), max_size=limits.max_set_size)
        )
    if len(proposed) > limits.max_hypotheses_per_service:
        tally.notes.append(
            f"{svc.name}: 끊기는 최소 조합 {len(proposed)}개 중 작은 조합부터 "
            f"{limits.max_hypotheses_per_service}개만 실었습니다(상한)."
        )
        proposed = proposed[: limits.max_hypotheses_per_service]
    tally.drafts.extend(proposed)
    tally.stats.append(
        ServiceSweep(
            service_id=svc.id,
            baseline=Outcome.PASS,
            candidates=len(units),
            candidates_total=total,
            replays=1 + used,
            breaking=len(cuts),
            exhausted=exhausted,
        )
    )


def _capped(drafts: list[HypothesisDraft], limit: int, notes: list[str]) -> list[HypothesisDraft]:
    """At most ``limit`` drafts: smallest sets first, then service order (noted when cut)."""
    if len(drafts) <= limit:
        return drafts
    notes.append(f"끊기는 최소 조합 {len(drafts)}개 중 작은 조합부터 {limit}개만 실었습니다(상한).")
    ranked = sorted(range(len(drafts)), key=lambda i: (len(drafts[i].perturbations), i))
    keep = set(ranked[:limit])
    return [d for i, d in enumerate(drafts) if i in keep]


def sweep(
    ctx: CheckContext,
    findings: Sequence[Finding],
    *,
    config: SimConfig | None = None,
    limits: SweepLimits = DEFAULT_LIMITS,
) -> SweepResult:
    """Search every critical service for minimal breaking sets (see the module docstring)."""
    sim = config if config is not None else SimConfig()
    budget = _Budget(limits.max_replays)
    view = ctx.with_graph(trusted_view(ctx.graph)[0])
    index = GraphIndex(view.graph)
    critical = view.critical_services()
    order = {s.id: i for i, s in enumerate(ctx.critical_services())}
    tally = _Tally()

    # A held service is left out of the replay: a baseline "failure" there would only mean "not
    # confirmed", not a break the rules report (P3).
    held = [s for s in ctx.critical_services() if s.id not in view.graph.nodes]
    for svc in held:
        _skip(tally, svc, None, HELD_SERVICE_REASON)
    if held:
        tally.notes.append(
            "판단 보류(원문 대조 전) 서비스라 탐색하지 못했습니다: " + ", ".join(s.name for s in held)
        )
    services = [s for s in ctx.critical_services() if s.id in view.graph.nodes]
    baselines = _replay(ctx, [_scenario(s.id, ()) for s in services[: budget.left]], sim)
    budget.used += len(baselines)
    if len(baselines) < len(services):
        skipped = ", ".join(s.name for s in services[len(baselines) :])
        tally.notes.append(
            f"리허설 예산({limits.max_replays}회)을 다 써 다음 서비스는 탐색하지 못했습니다: {skipped}"
        )
    for svc, baseline in zip(services, baselines, strict=False):
        if baseline.outcome is Outcome.FAIL:
            _skip(tally, svc, Outcome.FAIL, BASELINE_FAILED_REASON)
        elif budget.left == 0:
            tally.out_of_budget.append(svc.name)
            _skip(tally, svc, Outcome.PASS, "리허설 예산을 다 써 탐색하지 못했습니다.")
        else:
            _sweep_service(
                ctx,
                view,
                svc,
                baseline,
                findings,
                index=index,
                critical=critical,
                budget=budget,
                limits=limits,
                config=sim,
                tally=tally,
            )
    for svc in services[len(baselines) :]:
        _skip(tally, svc, None, "리허설 예산을 다 써 기준 리허설도 돌리지 못했습니다.")
    if tally.out_of_budget:
        tally.notes.append(
            f"리허설 예산({limits.max_replays}회)을 다 써 일부 조합을 탐색하지 못했습니다: "
            + ", ".join(tally.out_of_budget)
        )
    drafts = _capped(tally.drafts, limits.max_hypotheses, tally.notes)
    return SweepResult(
        drafts=tuple(drafts),
        # published in the critical services' own order, whatever made a service skip
        services=tuple(sorted(tally.stats, key=lambda s: order[s.service_id])),
        covered=tuple(tally.covered),
        replays=budget.used,
        notes=tuple(tally.notes),
        found=frozenset(tally.found),
        max_set_size=limits.max_set_size,
        found_effects=frozenset(tally.found_effects),
    )
