"""최소 실패 조건 자동 검증 — which of an AI hypothesis' own conditions its break needs.

A generative model may propose more than a break needs. On 별빛상점 v5 it proposed "추천: 강현우
부재 + 상품 카탈로그 동시 장애" and the replay confirmed it, but the catalog incident alone breaks
추천 at the same person (박재현), while 강현우's absence alone does not. This step finds that
without a person running the ablation by hand.

After verification (:func:`jettae.explore.verify.verify_hypotheses`), every ``confirmed`` AI
hypothesis with two to :data:`MINIMIZE_MAX_PERTURBATIONS` perturbations and no premises
(``assumptions``) is reduced:

1. Every non-empty proper subset of its perturbations is replayed — same service, incident type
   and policy deadline, same graph, simulator configuration, policy and roster as the
   verification (at most 2⁴ − 2 = 14 replays).
2. A subset *reproduces the break* when its replay fails at the same node (``failed_at_id``) as
   the whole hypothesis. A subset that fails at another node is a different break: it is listed as
   such (``other_failures``) and never used to drop a condition. A different step at the same node
   (담당 호출 → 승인) is disclosed (``point_differs``).
3. The minimal sets are the reproducing sets (the whole hypothesis included) no proper subset of
   which reproduces the break. A perturbation in none of them is unnecessary. When no proper
   subset reproduces the break, the proposal itself is minimal.

The preconditions hold by construction of ``confirmed``: the baseline replay passes and nothing
involved is held (판단 보류). Sweep hypotheses are minimal by construction (docs/explore.md §3) and
are left alone, as are ``refuted`` and ``needs_info`` ones. Only the new ``minimization`` field is
set: status, evidence, findings and score are unchanged.

It only removes conditions from the proposal; it does not discover causes the proposal does not
name. Replays are capped per run (:data:`MINIMIZE_MAX_REPLAYS`): a hypothesis the remaining budget
cannot cover in full is not reduced, and a Korean note says so (no silent caps).

No LLM, no I/O, deterministic order (P1, P6).
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from jettae.check.context import CheckContext
from jettae.domain import (
    FailingSubset,
    HypothesisMinimization,
    HypothesisOrigin,
    HypothesisStatus,
    Outcome,
    Perturbation,
    RehearsalResult,
    RiskHypothesis,
    Scenario,
    ScenarioOrigin,
    SimConfig,
)
from jettae.explore.drafts import canonical_perturbations
from jettae.explore.verify import VERIFY_SCENARIO_PREFIX, clip_reason
from jettae.rehearse.simulator import replay_all

__all__ = [
    "MINIMIZE_MAX_PERTURBATIONS",
    "MINIMIZE_MAX_REPLAYS",
    "Minimized",
    "minimizable",
    "minimize_hypotheses",
]

#: The most perturbations a hypothesis may have to be reduced (an AI proposal has at most 4).
MINIMIZE_MAX_PERTURBATIONS: Final = 4
#: Subset replays per run, over all hypotheses (10 AI hypotheses × 14 would be 140).
MINIMIZE_MAX_REPLAYS: Final = 100

_Indices = tuple[int, ...]


@dataclass(frozen=True)
class Minimized:
    """The hypotheses (``minimization`` set where it applied), replays spent and notes."""

    hypotheses: tuple[RiskHypothesis, ...]
    replays: int = 0
    notes: tuple[str, ...] = ()


def minimizable(hypothesis: RiskHypothesis) -> bool:
    """Is this a hypothesis the step applies to (``confirmed`` AI, 2+ perturbations, no premise)?"""
    v = hypothesis.verification
    return (
        hypothesis.origin is HypothesisOrigin.AI
        and hypothesis.status is HypothesisStatus.CONFIRMED
        and not hypothesis.assumptions
        and len(canonical_perturbations(hypothesis.perturbations)) >= 2
        and v.outcome is Outcome.FAIL
        and v.baseline_outcome is Outcome.PASS
    )


def _subsets(size: int) -> list[_Indices]:
    """Non-empty proper subsets of ``range(size)``, smallest first, then in index order."""
    return [combo for k in range(1, size) for combo in itertools.combinations(range(size), k)]


def _scenario(
    hypothesis: RiskHypothesis, perturbations: tuple[Perturbation, ...], combo: _Indices
) -> Scenario:
    """The hypothesis' drill with part of its perturbations; no deadline of its own (policy SLO)."""
    return Scenario(
        id=f"{VERIFY_SCENARIO_PREFIX}{hypothesis.id}:min:{'-'.join(str(i) for i in combo)}",
        title=hypothesis.title,
        service_id=hypothesis.service_id,
        incident_type=hypothesis.incident_type,
        perturbations=tuple(perturbations[i] for i in combo),
        deadline_minutes=None,
        origin=ScenarioOrigin.LLM,
    )


def _subset(
    hypothesis: RiskHypothesis,
    perturbations: tuple[Perturbation, ...],
    combo: _Indices,
    result: RehearsalResult,
) -> FailingSubset:
    v = hypothesis.verification
    return FailingSubset(
        perturbations=tuple(perturbations[i] for i in combo),
        failed_element=result.failed_element,
        failed_at_id=result.failed_at_id,
        reason=clip_reason(result.reason),
        result_hash=result.result_hash(),
        point_differs=(result.failed_element, result.failed_at_id) != (v.failed_element, v.failed_at_id),
    )


def _minimal(candidates: Sequence[_Indices]) -> list[_Indices]:
    """Those of ``candidates`` (smallest first) that contain no other candidate."""
    kept: list[_Indices] = []
    for combo in candidates:
        if not any(set(small) < set(combo) for small in kept):
            kept.append(combo)
    return kept


def _reduce(
    hypothesis: RiskHypothesis,
    perturbations: tuple[Perturbation, ...],
    results: dict[_Indices, RehearsalResult],
) -> HypothesisMinimization:
    v = hypothesis.verification
    whole = tuple(range(len(perturbations)))
    failing = [c for c, r in results.items() if r.outcome is Outcome.FAIL]
    same = [c for c in failing if results[c].failed_at_id == v.failed_at_id]
    minimal = _minimal([*same, whole])
    # minimal among the subsets breaking elsewhere only: one that contains a same-node set is a
    # different break all the same and stays listed
    elsewhere = _minimal([c for c in failing if c not in same])
    used = {i for combo in minimal for i in combo}
    sets = [
        _subset(hypothesis, perturbations, c, results[c])
        if c != whole
        else FailingSubset(
            perturbations=perturbations,
            failed_element=v.failed_element,
            failed_at_id=v.failed_at_id,
            reason=v.reason,
            result_hash=v.result_hash,
        )
        for c in minimal
    ]
    return HypothesisMinimization(
        replays=len(results),
        minimal_sets=tuple(sets),
        original_minimal=minimal == [whole],
        unnecessary=tuple(p for i, p in enumerate(perturbations) if i not in used),
        other_failures=tuple(_subset(hypothesis, perturbations, c, results[c]) for c in elsewhere),
    )


def minimize_hypotheses(
    ctx: CheckContext,
    hypotheses: Sequence[RiskHypothesis],
    *,
    config: SimConfig | None = None,
    max_replays: int = MINIMIZE_MAX_REPLAYS,
) -> Minimized:
    """Set ``minimization`` on every :func:`minimizable` hypothesis verified on ``ctx.graph``.

    ``config`` must be the configuration the hypotheses were verified with. Hypotheses keep their
    order; the others come back unchanged. Hypotheses are taken in that order while the replay
    budget covers all of their subsets; one it cannot cover is skipped (a later, smaller one may
    still fit) and named in a note.
    """
    sim = config if config is not None else SimConfig()
    graph_hash = ctx.graph.content_hash()
    jobs: list[tuple[int, tuple[Perturbation, ...], list[_Indices]]] = []
    too_big: list[str] = []
    over_budget: list[str] = []
    spent = 0
    for index, hypothesis in enumerate(hypotheses):
        if not minimizable(hypothesis) or hypothesis.verification.graph_hash != graph_hash:
            continue
        perturbations = canonical_perturbations(hypothesis.perturbations)
        if len(perturbations) > MINIMIZE_MAX_PERTURBATIONS:
            too_big.append(hypothesis.title)
            continue
        combos = _subsets(len(perturbations))
        if spent + len(combos) > max_replays:
            over_budget.append(hypothesis.title)
            continue
        spent += len(combos)
        jobs.append((index, perturbations, combos))
    out = list(hypotheses)
    if jobs:
        scenarios = [
            _scenario(hypotheses[index], perturbations, combo)
            for index, perturbations, combos in jobs
            for combo in combos
        ]
        results = iter(replay_all(ctx.graph, scenarios, policy=ctx.policy, config=sim, roster=ctx.roster))
        for index, perturbations, combos in jobs:
            by_combo = {combo: next(results) for combo in combos}
            reduced = _reduce(hypotheses[index], perturbations, by_combo)
            out[index] = hypotheses[index].model_copy(update={"minimization": reduced})
    notes: list[str] = []
    if too_big:
        notes.append(
            f"AI 위험 가설 {len(too_big)}건은 교란이 {MINIMIZE_MAX_PERTURBATIONS}개를 넘어 최소 실패 조건을 "
            f"자동 검증하지 않았습니다: {', '.join(too_big)}"
        )
    if over_budget:
        notes.append(
            f"AI 위험 가설 {len(over_budget)}건은 최소 실패 조건 자동 검증의 리허설 상한({max_replays}회)에 "
            f"걸려 확인하지 않았습니다(상한): {', '.join(over_budget)}"
        )
    return Minimized(hypotheses=tuple(out), replays=spent, notes=tuple(notes))
