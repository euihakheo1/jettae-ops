"""Verification (결정적 검증) — the only judge of a risk hypothesis.

Whoever proposed a hypothesis (the exhaustive sweep or a generative model), its status comes from
the same deterministic replay that judges rehearsals (:mod:`jettae.rehearse.simulator`):

* The hypothesis becomes a scenario: its perturbations for the whole incident, on its service,
  with **the service's policy deadline** (the tier SLO). Nobody — least of all a model — can set
  an easier deadline.
* The same service is also replayed *without* any perturbation (the baseline).

Status rules, in this order (P3, 확인 불가 ≠ 정상):

1. Baseline fails → ``needs_info``. The service breaks without the hypothesis, so the replay cannot
   attribute anything to it; the fixed rules already report the break.
2. The service or a target is held (판단 보류) → ``needs_info``. The replay leaves held facts out,
   so a perturbation of a held node does nothing there: a pass would refute the hypothesis, and a
   failure confirm it, on a fact nobody has checked (P3). ``required_info`` names the held nodes.
   (New AI proposals cannot name held nodes; an earlier report's hypothesis can, after a re-diagnosis.)
3. The hypothesis states premises the graph does not contain (``assumptions``) → ``needs_info``,
   never ``confirmed``: the replay can only show what *would* happen if the premises held.
   ``required_info`` lists the premises to check; the replay of the representable part is kept.
4. Replay fails → ``confirmed`` (리허설로 재현). Replay mitigates in time → ``refuted``.

No LLM, no I/O, deterministic order (P1, P6).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from jettae.check.context import CheckContext
from jettae.domain import (
    HypothesisOrigin,
    HypothesisStatus,
    HypothesisVerification,
    IncidentType,
    OpsGraph,
    Outcome,
    RehearsalResult,
    RiskHypothesis,
    Scenario,
    ScenarioOrigin,
    SimConfig,
    Verification,
)
from jettae.explore.drafts import HypothesisDraft, collect_evidence
from jettae.rehearse.simulator import replay_all

__all__ = [
    "BASELINE_FAILED_INFO",
    "STATUS_ORDER",
    "VERIFY_SCENARIO_PREFIX",
    "clip_reason",
    "hypothesis_scenario",
    "verify_hypotheses",
]

#: Id prefix of the scenarios a verification replays (``explore:hyp-…``).
VERIFY_SCENARIO_PREFIX: Final = "explore:"

#: Report order: what to act on first.
STATUS_ORDER: Final[dict[HypothesisStatus, int]] = {
    HypothesisStatus.CONFIRMED: 0,
    HypothesisStatus.NEEDS_INFO: 1,
    HypothesisStatus.REFUTED: 2,
}
_ORIGIN_ORDER: Final[dict[HypothesisOrigin, int]] = {HypothesisOrigin.SWEEP: 0, HypothesisOrigin.AI: 1}

BASELINE_FAILED_INFO: Final = (
    "교란 없이도 리허설이 실패합니다. 고정 규칙이 보고한 끊김을 먼저 고친 뒤 이 가설을 다시 확인해야 합니다."
)
_REASON_LIMIT: Final = 300


def hypothesis_scenario(draft: HypothesisDraft) -> Scenario:
    """The drill a hypothesis is judged by: its perturbations, no deadline of its own."""
    return Scenario(
        id=f"{VERIFY_SCENARIO_PREFIX}{draft.id}",
        title=draft.title,
        description=draft.explanation,
        service_id=draft.service_id,
        incident_type=draft.incident_type,
        perturbations=draft.perturbations,
        deadline_minutes=None,
        origin=ScenarioOrigin.TEMPLATE if draft.origin is HypothesisOrigin.SWEEP else ScenarioOrigin.LLM,
    )


def _baseline_scenario(service_id: str, incident_type: IncidentType) -> Scenario:
    return Scenario(
        id=f"{VERIFY_SCENARIO_PREFIX}baseline:{service_id}:{incident_type.value}",
        title=f"{service_id} 기준",
        service_id=service_id,
        incident_type=incident_type,
    )


def clip_reason(text: str) -> str:
    """A replay reason cut to what a hypothesis keeps (the full timeline stays in the replay)."""
    return text if len(text) <= _REASON_LIMIT else text[: _REASON_LIMIT - 1] + "…"


def _held(graph: OpsGraph, draft: HypothesisDraft) -> tuple[str, ...]:
    """Names of the held (판단 보류) nodes a draft involves — its service and its targets."""
    ids = dict.fromkeys((draft.service_id, *draft.target_ids))
    return tuple(
        graph.nodes[nid].name
        for nid in ids
        if nid in graph.nodes and graph.nodes[nid].verification is Verification.HOLD
    )


def _status(
    draft: HypothesisDraft, result: RehearsalResult, baseline: RehearsalResult, held: Sequence[str]
) -> HypothesisStatus:
    if baseline.outcome is Outcome.FAIL or held or draft.assumptions:
        return HypothesisStatus.NEEDS_INFO
    return HypothesisStatus.CONFIRMED if result.outcome is Outcome.FAIL else HypothesisStatus.REFUTED


def _required(draft: HypothesisDraft, baseline: RehearsalResult, held: Sequence[str]) -> tuple[str, ...]:
    needed: list[str] = []
    if baseline.outcome is Outcome.FAIL:
        needed.append(BASELINE_FAILED_INFO)
    needed += [
        f"판단 보류 확인: '{name}'은(는) 원문 대조 전이라 리허설이 쓰지 않습니다. "
        "확인한 뒤 다시 검증해야 합니다."
        for name in held
    ]
    needed += [f"전제 확인: {premise}" for premise in draft.assumptions]
    return tuple(needed)


def verify_hypotheses(
    ctx: CheckContext,
    drafts: Sequence[HypothesisDraft],
    *,
    config: SimConfig | None = None,
) -> tuple[RiskHypothesis, ...]:
    """Judge every resolvable draft by deterministic replay (see the module docstring).

    Drafts whose service or targets are not in ``ctx.graph`` (with the right kind) are left out —
    a perturbation of nothing would pass for the wrong reason. Duplicates (same id) are judged
    once. The result is sorted by status (확인됨 → 확인 필요 → 기각), service, origin (sweep
    first), size and id.
    """
    sim = config if config is not None else SimConfig()
    unique: dict[str, HypothesisDraft] = {}
    for draft in drafts:
        if draft.resolvable(ctx.graph):
            unique.setdefault(draft.id, draft)
    if not unique:
        return ()
    pending = list(unique.values())
    keys = sorted({(d.service_id, d.incident_type) for d in pending}, key=lambda k: (k[0], k[1].value))
    baseline_runs = replay_all(
        ctx.graph,
        [_baseline_scenario(sid, incident) for sid, incident in keys],
        policy=ctx.policy,
        config=sim,
        roster=ctx.roster,
    )
    baselines = dict(zip(keys, baseline_runs, strict=True))
    results = replay_all(
        ctx.graph,
        [hypothesis_scenario(d) for d in pending],
        policy=ctx.policy,
        config=sim,
        roster=ctx.roster,
    )
    out: list[RiskHypothesis] = []
    for draft, result in zip(pending, results, strict=True):
        baseline = baselines[(draft.service_id, draft.incident_type)]
        held = _held(ctx.graph, draft)
        verification = HypothesisVerification(
            scenario_id=result.scenario_id,
            scenario_fingerprint=result.scenario_fingerprint,
            graph_hash=result.graph_hash,
            deadline_minutes=result.deadline_minutes,
            outcome=result.outcome,
            baseline_outcome=baseline.outcome,
            time_to_mitigate=result.time_to_mitigate,
            failed_element=result.failed_element,
            failed_at_id=result.failed_at_id,
            reason=clip_reason(result.reason),
            result_hash=result.result_hash(),
            path_ids=result.path_ids,
        )
        out.append(
            RiskHypothesis(
                id=draft.id,
                origin=draft.origin,
                service_id=draft.service_id,
                incident_type=draft.incident_type,
                title=draft.title,
                explanation=draft.explanation,
                perturbations=draft.perturbations,
                assumptions=draft.assumptions,
                evidence=collect_evidence(ctx.graph, draft.target_ids, result.path_ids),
                status=_status(draft, result, baseline, held),
                verification=verification,
                required_info=_required(draft, baseline, held),
            )
        )
    return tuple(
        sorted(
            out,
            key=lambda h: (
                STATUS_ORDER[h.status],
                h.service_id,
                _ORIGIN_ORDER[h.origin],
                len(h.perturbations),
                h.id,
            ),
        )
    )
