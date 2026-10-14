"""Evaluation metrics (평가 지표): turn per-case results into the published numbers.

Every number answers a question a sceptical reader would ask, and is defined so that it cannot
be improved by noise:

* **Detection rate** — of the mutated cases, how many did the arm catch? A *hit* is any expected
  rule id fired for the target service, or — for the rules in :data:`ORG_LEVEL_RULE_IDS` only —
  fired at organisation level (``service=None``: the fixed rules report a duty org-wide when it
  belongs to no single service, e.g. a reporting step shared by several services). A rule that
  is always about one service (owner, approver, rollback, escalation …) never gets credit
  without the service: otherwise an arm that names no service at all (the LLM direct arm may
  answer "organisation-wide") would score a hit it did not localise. A strict variant that only
  accepts the target service is reported next to it. Active and held-out operators are never
  pooled: held-out ones were sealed for generalisation and are the honest number.
* **Twin-corrected (paired) detection** — a hit only counts when the clean control of the same
  org, style and position (its *twin*) did not raise that same credited finding. An arm that
  already reports "no approver" for a service whose approvers are written down "detects" a
  deleted approver for free; the plain rate cannot tell that apart from a real detection, the
  paired one can. The gap between the two is the share of detections that are baseline noise.
* **False-positive rate** — of the clean controls (same orgs, same styles, nothing broken), how
  many raised *any* HIGH/CRITICAL finding? Without it, "flag everything" would score 100%.
* **Field-level precision/recall/F1** — graph facts vs gold facts, exact including attributes,
  over the edge kinds only a runbook can state (:data:`~jettae.evalkit.facts.RUNBOOK_ONLY_EDGE_KINDS`;
  roster and catalog facts are imported verbatim and would only flatter the extractor).
* **Missing-field hallucination** — for deletion operators, how many deleted facts did the arm
  put back? The text no longer states them, so each one is invented.
* **Ungrounded elements** (must be 0), **latency**, **LLM calls/tokens** per case.

Rates carry Wilson 95% intervals (well-behaved at 0% and 100%, unlike the normal approximation).
A case whose run errored counts against the arm: a miss on mutated cases, an alarm on clean
controls, all gold facts missed, no paired credit when the twin errored, and whatever partial
graph it still built counts for hallucination — "확인 불가 ≠ 정상" (P3) applies to the
evaluation too.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from jettae.domain import EdgeKind
from jettae.evalkit.benchmarks import ImpactBenchmark, InjectionSelftest
from jettae.evalkit.conditions import (
    GRAPH_CONDITIONS,
    LLM_CONDITIONS,
    CaseResult,
    Condition,
    FiredKey,
    rule_severity,
)
from jettae.evalkit.corpus import EvalCase
from jettae.evalkit.facts import RUNBOOK_ONLY_EDGE_KINDS
from jettae.evalkit.gold import (
    RULE_BREACH_REPORTER,
    RULE_BREACH_STEP,
    RULE_DEADLINE_CONFLICT,
    RULE_DEADLINE_LATE,
    RULE_DEPARTED,
    RULE_ON_LEAVE,
    RULE_PRIVACY_REPORTER,
    RULE_PRIVACY_STEP,
    RULE_UNCONFIRMED,
    Fact,
)
from jettae.evalkit.harness import CostEstimate, EvalResults, Reproducibility
from jettae.evalkit.mutate import ALL_OPERATORS, Operator
from jettae.evalkit.render import POSITIONS, STYLES
from jettae.evalkit.seal import SealManifest

__all__ = [
    "DELETION_OPERATORS",
    "ORG_LEVEL_RULE_IDS",
    "Z95",
    "CleanFalsePositives",
    "ConditionSummary",
    "CorpusInfo",
    "EvalMetrics",
    "FieldScores",
    "Hallucination",
    "Latency",
    "LlmUse",
    "OperatorRate",
    "Prf",
    "Rate",
    "RuleCount",
    "SplitRate",
    "compute_metrics",
    "credited_keys",
    "is_hit",
    "is_paired_hit",
    "wilson_interval",
]

#: two-sided 95% normal quantile
Z95 = 1.959963984540054
_DIGITS = 4

#: Operators that only remove facts: what an arm still reports of them was invented.
DELETION_OPERATORS: tuple[Operator, ...] = (
    Operator.DELETE_OWNER,
    Operator.DELETE_APPROVER,
    Operator.DELETE_FALLBACK_CHANNEL,
    Operator.DELETE_ROLLBACK,
    Operator.DELETE_ESCALATION,
    Operator.DELETE_DETECTION,
    Operator.DELETE_REPORTING_STEP,
)

#: Rules whose duty can belong to the whole organisation, so the fixed rules may report them
#: without a service: people rules (an escalation hop no service's chain reaches), deadline
#: rules (a reporting step shared by several services) and the reporting-duty rules (no
#: regulated service has the step). Path, escalation and SPOF rules always name their service.
ORG_LEVEL_RULE_IDS: frozenset[str] = frozenset(
    {
        RULE_DEPARTED,
        RULE_UNCONFIRMED,
        RULE_ON_LEAVE,
        RULE_DEADLINE_LATE,
        RULE_DEADLINE_CONFLICT,
        RULE_BREACH_STEP,
        RULE_BREACH_REPORTER,
        RULE_PRIVACY_STEP,
        RULE_PRIVACY_REPORTER,
    }
)


def wilson_interval(hits: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval of ``hits / n`` (Wilson 1927).

    At 0% and 100% one bound is exactly 0 or 1 (algebraically); it is returned as such rather
    than as the floating-point residue of ``centre ± half``.
    """
    if n <= 0:
        raise ValueError("표본 수는 1 이상이어야 합니다")
    if not 0 <= hits <= n:
        raise ValueError(f"성공 수가 범위를 벗어났습니다: {hits}/{n}")
    p = hits / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    lo = 0.0 if hits == 0 else max(0.0, centre - half)
    hi = 1.0 if hits == n else min(1.0, centre + half)
    return lo, hi


def _round(value: float) -> float:
    return round(value, _DIGITS)


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True)


class Rate(_Model):
    """``hits / n`` with its Wilson 95% interval (``None`` when there is nothing to measure)."""

    hits: int
    n: int
    rate: float | None
    ci95: tuple[float, float] | None

    @classmethod
    def of(cls, hits: int, n: int) -> Rate:
        if n == 0:
            return cls(hits=0, n=0, rate=None, ci95=None)
        lo, hi = wilson_interval(hits, n)
        return cls(hits=hits, n=n, rate=_round(hits / n), ci95=(_round(lo), _round(hi)))


class SplitRate(_Model):
    active: Rate
    held_out: Rate


class OperatorRate(Rate):
    held_out: bool
    #: twin-corrected rate (``None`` when the results hold no clean controls to pair with)
    paired: Rate | None = None


class Prf(_Model):
    """Micro-averaged precision/recall/F1 from fact counts."""

    tp: int
    fp: int
    fn: int
    precision: float | None
    recall: float | None
    f1: float | None

    @classmethod
    def of(cls, tp: int, fp: int, fn: int) -> Prf:
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        # 2TP / (2TP + FP + FN): the harmonic mean wherever both are defined, and 0 — not the
        # dash of "nothing to measure" — for an arm that got every fact wrong or extracted none
        # while facts were expected. A total failure must never look unmeasured (P3).
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
        return cls(
            tp=tp,
            fp=fp,
            fn=fn,
            precision=None if precision is None else _round(precision),
            recall=None if recall is None else _round(recall),
            f1=None if f1 is None else _round(f1),
        )


class Latency(_Model):
    median_s: float
    p95_s: float
    mean_s: float
    max_s: float


class LlmUse(_Model):
    calls_per_case: float
    input_tokens_per_case: float
    output_tokens_per_case: float
    total_calls: int
    total_input_tokens: int
    total_output_tokens: int


class ConditionSummary(_Model):
    n_cases: int
    errors: int
    detection_active: Rate
    detection_held_out: Rate
    detection_strict_active: Rate
    detection_strict_held_out: Rate
    #: hits the clean twin does not explain (see the module docstring); n = cases with a twin
    detection_paired_active: Rate
    detection_paired_held_out: Rate
    clean_fp: Rate
    field: Prf | None
    hallucination: Rate | None
    ungrounded: int
    latency: Latency
    llm: LlmUse


class RuleCount(_Model):
    cases: int
    severity: str


class CleanFalsePositives(_Model):
    rate: Rate
    by_style: dict[str, Rate]
    #: rule id → number of clean controls on which it fired (any severity)
    per_rule: dict[str, RuleCount]


class FieldScores(_Model):
    all: Prf
    by_style: dict[str, Prf]
    by_kind: dict[str, Prf]


class Hallucination(_Model):
    rate: Rate
    by_operator: dict[str, Rate]
    by_style: dict[str, Rate]


class CorpusInfo(_Model):
    generator_version: str
    seed: int | None
    n_orgs: int
    n_active: int
    n_held_out: int
    n_clean: int
    held_out_manifest_root: str | None


class EvalMetrics(_Model):
    """metrics.json — consumed by the report and the pitch-deck generator.

    ``generated_at`` is passed in by the caller; nothing here reads the clock, so the same
    results always give the same file (timings aside).
    """

    generated_at: str
    n_cases: int
    manifest_root: str | None
    corpus: CorpusInfo
    #: worker processes of the offline arms (per-case latencies were measured under this load)
    workers: int
    conditions: dict[str, ConditionSummary]
    by_style: dict[str, dict[str, SplitRate]]
    by_operator: dict[str, dict[str, OperatorRate]]
    by_position: dict[str, dict[str, SplitRate]]
    clean_fp: dict[str, CleanFalsePositives]
    field_f1: dict[str, FieldScores]
    hallucination: dict[str, Hallucination]
    ungrounded_total: int
    reproducibility: Reproducibility | None
    impact_benchmark: ImpactBenchmark | None
    injection_selftest: InjectionSelftest | None
    llm_cost_estimate: CostEstimate | None
    #: LLM arms that were not run (they need an API key)
    conditions_pending: tuple[str, ...]


# --------------------------------------------------------------------------- per-case verdicts


def is_hit(case: EvalCase, result: CaseResult, *, strict: bool = False) -> bool:
    """Did ``result`` catch the fault of ``case``?

    An expected rule must fire for the target service, or (unless ``strict``) at organisation
    level when the rule is one of :data:`ORG_LEVEL_RULE_IDS`.
    """
    if case.is_clean or result.error is not None:
        return False
    return bool(credited_keys(case, result.fired, strict=strict))


def credited_keys(case: EvalCase, fired: Iterable[FiredKey], *, strict: bool = False) -> frozenset[FiredKey]:
    """The fired (service, rule) keys that earn ``case`` a hit (empty for clean controls)."""
    if case.is_clean:
        return frozenset()
    accepted = {(case.target_service, rule) for rule in case.expected_rule_ids}
    if not strict:
        accepted |= {(None, rule) for rule in case.expected_rule_ids if rule in ORG_LEVEL_RULE_IDS}
    return frozenset(accepted.intersection(fired))


def is_paired_hit(case: EvalCase, result: CaseResult, twin: CaseResult) -> bool:
    """A hit that the clean ``twin`` (same org, style and position, nothing broken) does not explain.

    At least one credited key must be absent from the twin's findings. A twin that errored
    cannot vouch for anything, so it yields no credit (P3).
    """
    if not is_hit(case, result) or twin.error is not None:
        return False
    return bool(credited_keys(case, result.fired) - twin.fired)


#: (org id, style, position) — what a mutated case shares with its clean twin
_TwinKey = tuple[str, str, str]


def _twin_key(case: EvalCase) -> _TwinKey:
    return (case.org_id, case.style, case.position)


def _twins(pairs: Sequence[tuple[EvalCase, CaseResult]]) -> dict[_TwinKey, CaseResult]:
    return {_twin_key(c): r for c, r in pairs if c.is_clean}


def _alarm(result: CaseResult) -> bool:
    return result.error is not None or bool(result.severe)


def _runbook_facts(facts: Iterable[Fact]) -> set[Fact]:
    return {f for f in facts if f.kind in RUNBOOK_ONLY_EDGE_KINDS}


# --------------------------------------------------------------------------- aggregation


def _pairs(results: EvalResults, condition: Condition) -> list[tuple[EvalCase, CaseResult]]:
    by_id = {c.case_id: c for c in results.cases}
    return [(by_id[r.case_id], r) for r in results.of(condition)]


def _detection(pairs: Sequence[tuple[EvalCase, CaseResult]], *, held_out: bool, strict: bool = False) -> Rate:
    chosen = [(c, r) for c, r in pairs if not c.is_clean and c.held_out is held_out]
    return Rate.of(sum(is_hit(c, r, strict=strict) for c, r in chosen), len(chosen))


def _paired(pairs: Sequence[tuple[EvalCase, CaseResult]], twins: Mapping[_TwinKey, CaseResult]) -> Rate:
    """Twin-corrected detection over the mutated cases in ``pairs`` that have a clean twin."""
    chosen = [(c, r, twins[_twin_key(c)]) for c, r in pairs if not c.is_clean and _twin_key(c) in twins]
    return Rate.of(sum(is_paired_hit(c, r, t) for c, r, t in chosen), len(chosen))


def _paired_split(
    pairs: Sequence[tuple[EvalCase, CaseResult]], *, held_out: bool, twins: Mapping[_TwinKey, CaseResult]
) -> Rate:
    return _paired([(c, r) for c, r in pairs if c.held_out is held_out], twins)


def _split(pairs: Sequence[tuple[EvalCase, CaseResult]]) -> SplitRate:
    return SplitRate(active=_detection(pairs, held_out=False), held_out=_detection(pairs, held_out=True))


def _by_style(pairs: Sequence[tuple[EvalCase, CaseResult]]) -> dict[str, SplitRate]:
    styles = [s for s in STYLES if any(c.style == s for c, _ in pairs)]
    return {s: _split([(c, r) for c, r in pairs if c.style == s]) for s in styles}


def _by_position(pairs: Sequence[tuple[EvalCase, CaseResult]]) -> dict[str, SplitRate]:
    positions = [p for p in POSITIONS if any(c.position == p for c, _ in pairs)]
    return {p: _split([(c, r) for c, r in pairs if c.position == p]) for p in positions}


def _by_operator(pairs: Sequence[tuple[EvalCase, CaseResult]]) -> dict[str, OperatorRate]:
    twins = _twins(pairs)
    out: dict[str, OperatorRate] = {}
    for op in ALL_OPERATORS:
        chosen = [(c, r) for c, r in pairs if c.operator is op]
        if chosen:
            rate = Rate.of(sum(is_hit(c, r) for c, r in chosen), len(chosen))
            paired = _paired(chosen, twins)
            out[op.value] = OperatorRate(
                **rate.model_dump(),
                held_out=chosen[0][0].held_out,
                paired=paired if paired.n else None,
            )
    return out


def _clean_fp(pairs: Sequence[tuple[EvalCase, CaseResult]]) -> CleanFalsePositives:
    clean = [(c, r) for c, r in pairs if c.is_clean]
    styles = [s for s in STYLES if any(c.style == s for c, _ in clean)]
    per_rule: Counter[str] = Counter()
    for _, r in clean:
        if r.error is None:
            per_rule.update({rule for _, rule in r.fired})
    return CleanFalsePositives(
        rate=Rate.of(sum(_alarm(r) for _, r in clean), len(clean)),
        by_style={
            s: Rate.of(sum(_alarm(r) for c, r in clean if c.style == s), sum(c.style == s for c, _ in clean))
            for s in styles
        },
        per_rule={
            rule: RuleCount(cases=n, severity=rule_severity(rule).value)
            for rule, n in sorted(per_rule.items())
        },
    )


def _field_scores(pairs: Sequence[tuple[EvalCase, CaseResult]]) -> FieldScores:
    total = [0, 0, 0]
    style_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    kind_counts: dict[EdgeKind, list[int]] = defaultdict(lambda: [0, 0, 0])
    for case, result in pairs:
        gold = _runbook_facts(case.gold_facts)
        pred = _runbook_facts(result.facts or ()) if result.error is None else set()
        for bucket in (total, style_counts[case.style]):
            bucket[0] += len(gold & pred)
            bucket[1] += len(pred - gold)
            bucket[2] += len(gold - pred)
        for kind in RUNBOOK_ONLY_EDGE_KINDS:
            g = {f for f in gold if f.kind is kind}
            p = {f for f in pred if f.kind is kind}
            counts = kind_counts[kind]
            counts[0] += len(g & p)
            counts[1] += len(p - g)
            counts[2] += len(g - p)
    return FieldScores(
        all=Prf.of(*total),
        by_style={s: Prf.of(*style_counts[s]) for s in STYLES if s in style_counts},
        by_kind={k.value: Prf.of(*kind_counts[k]) for k in sorted(kind_counts, key=lambda k: k.value)},
    )


def _invented(case: EvalCase, result: CaseResult) -> tuple[int, int]:
    """(deleted facts the arm still reports, deleted facts) by identity (kind, src, dst).

    A deleted key that is still true in the mutated truth (an attribute changed, or the same
    relation holds for another reason) is not counted: reporting it is not an invention.
    """
    still_true = {f.key for f in case.gold_facts}
    missing = {f.key for f in case.deleted_facts} - still_true
    reported = {f.key for f in result.facts or ()}
    return len(missing & reported), len(missing)


def _hallucination(pairs: Sequence[tuple[EvalCase, CaseResult]]) -> Hallucination:
    # Every case that produced facts counts, errored or not: an LLM arm with a rejected chunk
    # still built a graph from the other chunks, and leaving that graph out would hide exactly
    # the inventions of the runs that went wrong. A case without facts invented nothing.
    chosen = [(c, r) for c, r in pairs if c.operator in DELETION_OPERATORS and r.facts is not None]

    def rate(subset: Iterable[tuple[EvalCase, CaseResult]]) -> Rate:
        counts = [_invented(c, r) for c, r in subset]
        return Rate.of(sum(i for i, _ in counts), sum(n for _, n in counts))

    return Hallucination(
        rate=rate(chosen),
        by_operator={
            op.value: rate((c, r) for c, r in chosen if c.operator is op)
            for op in DELETION_OPERATORS
            if any(c.operator is op for c, _ in chosen)
        },
        by_style={s: rate((c, r) for c, r in chosen if c.style == s) for s in STYLES},
    )


def _latency(results: Sequence[CaseResult]) -> Latency:
    seconds = sorted(r.seconds for r in results) or [0.0]
    p95 = seconds[max(0, math.ceil(0.95 * len(seconds)) - 1)]  # nearest-rank percentile
    return Latency(
        median_s=_round(statistics.median(seconds)),
        p95_s=_round(p95),
        mean_s=_round(statistics.fmean(seconds)),
        max_s=_round(seconds[-1]),
    )


def _llm_use(results: Sequence[CaseResult]) -> LlmUse:
    n = max(1, len(results))
    calls = sum(r.llm_calls for r in results)
    tokens_in = sum(r.input_tokens for r in results)
    tokens_out = sum(r.output_tokens for r in results)
    return LlmUse(
        calls_per_case=_round(calls / n),
        input_tokens_per_case=_round(tokens_in / n),
        output_tokens_per_case=_round(tokens_out / n),
        total_calls=calls,
        total_input_tokens=tokens_in,
        total_output_tokens=tokens_out,
    )


def _summary(condition: Condition, pairs: Sequence[tuple[EvalCase, CaseResult]]) -> ConditionSummary:
    results = [r for _, r in pairs]
    graph = condition in GRAPH_CONDITIONS
    twins = _twins(pairs)
    return ConditionSummary(
        n_cases=len(pairs),
        errors=sum(r.error is not None for r in results),
        detection_active=_detection(pairs, held_out=False),
        detection_held_out=_detection(pairs, held_out=True),
        detection_strict_active=_detection(pairs, held_out=False, strict=True),
        detection_strict_held_out=_detection(pairs, held_out=True, strict=True),
        detection_paired_active=_paired_split(pairs, held_out=False, twins=twins),
        detection_paired_held_out=_paired_split(pairs, held_out=True, twins=twins),
        clean_fp=_clean_fp(pairs).rate,
        field=_field_scores(pairs).all if graph else None,
        hallucination=_hallucination(pairs).rate if graph else None,
        ungrounded=sum(r.ungrounded for r in results),
        latency=_latency(results),
        llm=_llm_use(results),
    )


def _corpus(cases: Sequence[EvalCase], manifest: SealManifest | None, seed: int | None) -> CorpusInfo:
    versions = sorted({c.generator_version for c in cases})
    return CorpusInfo(
        generator_version=manifest.generator_version if manifest is not None else ", ".join(versions),
        seed=seed,
        n_orgs=len({c.org_id for c in cases}),
        n_active=sum(not c.is_clean and not c.held_out for c in cases),
        n_held_out=sum(c.held_out for c in cases),
        n_clean=sum(c.is_clean for c in cases),
        held_out_manifest_root=manifest.held_out_merkle_root if manifest is not None else None,
    )


def compute_metrics(
    results: EvalResults,
    *,
    generated_at: str,
    manifest: SealManifest | None = None,
    seed: int | None = None,
    reproducibility: Reproducibility | None = None,
    impact: ImpactBenchmark | None = None,
    injection: InjectionSelftest | None = None,
    cost_estimate: CostEstimate | None = None,
) -> EvalMetrics:
    """All published metrics of one run (see the module docstring for the definitions)."""
    pairs: Mapping[Condition, list[tuple[EvalCase, CaseResult]]] = {
        c: _pairs(results, c) for c in results.conditions
    }
    graph_arms = [c for c in results.conditions if c in GRAPH_CONDITIONS]
    return EvalMetrics(
        generated_at=generated_at,
        n_cases=len(results.cases),
        manifest_root=manifest.merkle_root if manifest is not None else None,
        corpus=_corpus(results.cases, manifest, seed),
        workers=results.workers,
        conditions={c.value: _summary(c, pairs[c]) for c in results.conditions},
        by_style={c.value: _by_style(pairs[c]) for c in results.conditions},
        by_operator={c.value: _by_operator(pairs[c]) for c in results.conditions},
        by_position={c.value: _by_position(pairs[c]) for c in results.conditions},
        clean_fp={c.value: _clean_fp(pairs[c]) for c in results.conditions},
        field_f1={c.value: _field_scores(pairs[c]) for c in graph_arms},
        hallucination={c.value: _hallucination(pairs[c]) for c in graph_arms},
        ungrounded_total=sum(r.ungrounded for r in results.results),
        reproducibility=reproducibility,
        impact_benchmark=impact,
        injection_selftest=injection,
        llm_cost_estimate=cost_estimate,
        conditions_pending=tuple(c.value for c in LLM_CONDITIONS if c not in results.conditions),
    )
