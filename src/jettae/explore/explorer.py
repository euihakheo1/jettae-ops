"""Risk exploration end to end: sweep (+ AI proposals) → deterministic verification → summary.

Two callers share every step:

* the diagnosis pipeline's ``explore`` node runs :func:`~jettae.explore.sweep.sweep`, asks the model
  (:func:`~jettae.explore.ai.propose_hypotheses`) when one is configured — recording a note if it
  fails, so the sweep is never lost — carries earlier AI hypotheses over (:func:`carry_over`) and
  verifies everything once (:func:`finish`);
* :func:`explore_graph` is the re-judgement after an approval or a revert (no LLM, P4): the sweep
  again, plus the earlier AI hypotheses re-verified on the new graph while every node they name
  still exists.

A carried AI hypothesis that the new sweep now accounts for (same set, or a set the sweep found
plus extra targets — :func:`~jettae.explore.sweep.known_breaks`) is not carried twice: the sweep's
own entry (or the rule finding that covers it) reports that break. One naming a node that is now
held (판단 보류) is carried and ends ``needs_info`` (:mod:`jettae.explore.verify`), never refuted
or confirmed by a fact nobody has checked (P3).

Neither touches the findings or the score (P1): exploration starts from them and only adds a
separate list of hypotheses to the report.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

from jettae.check.context import CheckContext
from jettae.domain import (
    AiProposalStatus,
    ExplorationSummary,
    Finding,
    HypothesisOrigin,
    RiskHypothesis,
    SimConfig,
)
from jettae.explore.ai import AiProposals
from jettae.explore.drafts import HypothesisDraft, draft_from_hypothesis
from jettae.explore.sweep import DEFAULT_LIMITS, SweepLimits, SweepResult, known_breaks, sweep
from jettae.explore.verify import verify_hypotheses

__all__ = ["MAX_AI_HYPOTHESES", "Exploration", "carry_over", "explore_graph", "finish"]

#: AI hypotheses a report keeps (new proposals first, then carried ones), so carrying them from
#: run to run cannot grow the report without bound.
MAX_AI_HYPOTHESES: Final = 10


@dataclass(frozen=True)
class Exploration:
    hypotheses: tuple[RiskHypothesis, ...]
    summary: ExplorationSummary


def carry_over(
    ctx: CheckContext, previous: Iterable[RiskHypothesis]
) -> tuple[tuple[HypothesisDraft, ...], int]:
    """Earlier AI hypotheses as drafts for ``ctx.graph`` and how many no longer resolve there.

    Sweep hypotheses are never carried: the sweep runs again on every graph.
    """
    drafts: list[HypothesisDraft] = []
    gone = 0
    for hypothesis in previous:
        if hypothesis.origin is not HypothesisOrigin.AI:
            continue
        draft = draft_from_hypothesis(hypothesis)
        if draft.resolvable(ctx.graph):
            drafts.append(draft)
        else:
            gone += 1
    return tuple(drafts), gone


def finish(
    ctx: CheckContext,
    swept: SweepResult,
    *,
    proposals: AiProposals | None = None,
    carried: Sequence[HypothesisDraft] = (),
    carried_gone: int = 0,
    config: SimConfig | None = None,
    notes: Sequence[str] = (),
    ai_status: AiProposalStatus | None = None,
) -> Exploration:
    """Verify the sweep's drafts, the new AI drafts and the carried ones; build the summary.

    ``ai_status`` says why there are no new AI proposals when the caller never asked the model
    (re-judgement, LLM fallback, a failed call); by default it is the proposals' own status.
    """
    fresh = proposals.drafts if proposals is not None else ()
    fresh_ids = {d.id for d in fresh}
    repeated = known_breaks(ctx, swept, [d for d in carried if d.id not in fresh_ids], config=config)
    ai: dict[str, HypothesisDraft] = {}
    for draft in (*fresh, *(d for d in carried if d.id not in repeated)):
        ai.setdefault(draft.id, draft)
    ai_drafts = list(ai.values())[:MAX_AI_HYPOTHESES]
    hypotheses = verify_hypotheses(ctx, [*swept.drafts, *ai_drafts], config=config)
    extra = list(notes)
    if carried_gone:
        extra.append(
            f"이전 진단의 AI 위험 가설 {carried_gone}건은 가리키는 대상이 새 그래프에 없어 "
            "다시 확인하지 않았습니다."
        )
    if repeated:
        extra.append(
            f"이전 진단의 AI 위험 가설 {len(repeated)}건은 이번 자동 전수 탐색이 찾은 끊김 조합(고정 규칙이 "
            "이미 보고한 것 포함)으로 설명되어 따로 싣지 않았습니다."
        )
    if len(ai) > MAX_AI_HYPOTHESES:
        extra.append(f"AI 위험 가설은 최근 것부터 {MAX_AI_HYPOTHESES}건만 실었습니다(상한).")
    status = ai_status
    if status is None:
        status = proposals.status if proposals is not None else AiProposalStatus.OFF
    summary = ExplorationSummary(
        max_set_size=swept.max_set_size,
        services=swept.services,
        replays=swept.replays,
        covered=swept.covered,
        ai_status=status,
        ai_used=status is AiProposalStatus.USED,
        ai_proposed=proposals.proposed if proposals is not None else 0,
        ai_kept=len(fresh),
        ai_dropped=proposals.dropped if proposals is not None else 0,
        ai_duplicates=proposals.duplicates if proposals is not None else 0,
        ai_over_cap=proposals.over_cap if proposals is not None else 0,
        ai_carried=sum(1 for d in ai_drafts if d.id not in fresh_ids),
        notes=(*swept.notes, *extra),
    )
    return Exploration(hypotheses=hypotheses, summary=summary)


def explore_graph(
    ctx: CheckContext,
    findings: Sequence[Finding],
    *,
    config: SimConfig | None = None,
    previous: Iterable[RiskHypothesis] = (),
    limits: SweepLimits = DEFAULT_LIMITS,
) -> Exploration:
    """Deterministic exploration of one graph (no LLM): sweep + carried AI hypotheses re-verified."""
    swept = sweep(ctx, findings, config=config, limits=limits)
    carried, gone = carry_over(ctx, previous)
    return finish(
        ctx,
        swept,
        carried=carried,
        carried_gone=gone,
        config=config,
        ai_status=AiProposalStatus.REJUDGE,
    )
