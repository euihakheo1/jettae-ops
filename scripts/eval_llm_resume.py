"""Finish the 20% 2x2 LLM run without re-running anything that already succeeded.

Run 1 (``C:/jettae/data/eval/llm-sample20``) was::

    jettae eval run --conditions A,B,O,C,D,DR --seed 20261012 --orgs 20 --sample 0.2 \\
        --no-benchmarks --concurrency 6      # JETTAE_ANTHROPIC_MODEL=claude-sonnet-5-5, effort medium

The Anthropic credit ran out partway through: every request after that point came back as
HTTP 400 "Your credit balance is too low". Those (case, condition) pairs never reached the model,
so they say nothing about it. Every other result is a measurement and is kept as it is, including
the LLM rows that failed for content reasons (max_tokens truncation, a refused chunk). Re-running
only the failures we dislike would be cherry-picking.

What the script does:

1. Regenerates the run's 264-case sample (seed, 20 orgs, 20% hash sample) and checks it against
   run 1's results.jsonl, manifest and cost estimate. Then it rebuilds run 1's metrics.json,
   report.md and results.jsonl from what run 1 published and requires all three to match byte for
   byte, so the merge starts from exactly the published numbers.
2. Selects the credit-failed pairs only: D and DR together (they share one extraction per case
   only when run together), and C on its own.
3. Dry run (the default) prints the pairs and the cost estimate. It sends nothing and writes
   nothing. ``--yes`` re-runs the pairs with the product's own ``run_eval`` (same model, effort
   and concurrency) in small batches. Every finished batch is saved, facts included, to
   ``<out>/rerun_results.json``, so nothing that was paid for is lost. The first request that
   fails without a judged answer (credit, key, rate limit, network ...) stops the run: no further
   request is sent, and a later ``--yes`` resumes with whatever is still missing.
4. Merges. For a credit-failed pair the final result is its rerun; for every other pair it is the
   run-1 row. The script writes metrics.json, report.md (the product's renderer), results.jsonl,
   manifest.json and merge_log.json to ``<out>``.

Facts. results.jsonl has no facts, but the field scores and the missing-field hallucination rate
need them. Both are sums over cases, so the merged value is still exact:

    run-1 total - what the replaced run-1 rows contributed + what their reruns contribute

A credit failure errored before any graph existed (facts None). It contributed tp = fp = 0 and
fn = |gold facts| to the field scores (an errored case is scored with pred = {}), and nothing to
the hallucination rate (only cases that produced facts count). The rerun's share is computed by the
product's own metric code. Every other metric is a function of fired / severe / error / ungrounded
/ tokens / seconds, all of which results.jsonl holds. The offline arms A, B and O keep run 1's
sections unchanged. ``--self-test`` proves the method on condition B (offline, free), and runs the
whole resume pipeline against run 1 with a fake model.

Usage (repository root, product venv; nothing is sent without --yes):
    python -X utf8 scripts/eval_llm_resume.py                  # dry run
    python -X utf8 scripts/eval_llm_resume.py --yes            # paid: re-run, then merge
    python -X utf8 scripts/eval_llm_resume.py --merge-only     # merge the saved reruns, send nothing
    python -X utf8 scripts/eval_llm_resume.py --self-test [--work-dir DIR]
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

import jettae.evalkit as kit
from jettae.domain import EdgeKind
from jettae.evalkit import conditions as harness_conditions
from jettae.evalkit.conditions import (
    GRAPH_CONDITIONS,
    LLM_CONDITIONS,
    OFFLINE_CONDITIONS,
    CaseResult,
    Condition,
    run_sync,
)
from jettae.evalkit.corpus import EvalCase
from jettae.evalkit.gold import Fact
from jettae.evalkit.harness import (
    CostEstimate,
    EvalResults,
    Reproducibility,
    estimate_llm_cost,
    reproducibility_sample,
    run_eval,
)
from jettae.evalkit.metrics import (
    DELETION_OPERATORS,
    EvalMetrics,
    FieldScores,
    Hallucination,
    Prf,
    Rate,
    compute_metrics,
)
from jettae.evalkit.render import STYLES
from jettae.evalkit.report import render_report, results_jsonl, write_report
from jettae.evalkit.seal import SealManifest, build_manifest, write_manifest
from jettae.llm.base import (
    LLMClient,
    LLMDisabledError,
    LLMError,
    LLMOutputError,
    LLMRefusalError,
    LLMResult,
    UntrustedDocument,
)
from jettae.llm.fake import FakeLLM

_T = TypeVar("_T", bound=BaseModel)

# --------------------------------------------------------------------------- run 1, as it was run

RUN1_DIR = Path("C:/jettae/data/eval/llm-sample20")
FINAL_DIR = Path("C:/jettae/data/eval/llm-sample20-final")
RUN1_COMMAND = (
    "jettae eval run --conditions A,B,O,C,D,DR --seed 20261012 --orgs 20 --sample 0.2 "
    "--no-benchmarks --concurrency 6 (JETTAE_ANTHROPIC_MODEL=claude-sonnet-5-5, "
    "JETTAE_ANTHROPIC_EFFORT=medium)"
)
SEED, ORGS, SAMPLE = 20261012, 20, 0.2
RUN1_CONDITIONS = (
    Condition.A_KEYWORD,
    Condition.B_RULE_GRAPH,
    Condition.O_ORACLE,
    Condition.C_LLM_DIRECT,
    Condition.D_LLM_GRAPH,
    Condition.DR_HYBRID,
)
MODEL, EFFORT, CONCURRENCY = "claude-sonnet-5-5", "medium", 6
CREDIT_MARK = "Your credit balance is too low"

GRAPH_ARMS = (Condition.D_LLM_GRAPH, Condition.DR_HYBRID)
DIRECT_ARMS = (Condition.C_LLM_DIRECT,)
#: D and DR go together (one shared extraction per case), C on its own.
PHASES: tuple[tuple[str, tuple[Condition, ...]], ...] = (("D+DR", GRAPH_ARMS), ("C", DIRECT_ARMS))

STORE_NAME = "rerun_results.json"
STORE_FORMAT = "jettae-eval-llm-resume/1"
LOG_FORMAT = "jettae-eval-llm-resume-merge/1"
#: Cases per run_eval call. Each finished batch is saved, so an interruption loses one batch at most.
DEFAULT_BATCH = 24
#: Claude Sonnet 5.5 list price, USD per million tokens (Anthropic first-party API, cached 2026-10-06).
USD_PER_MTOK_IN, USD_PER_MTOK_OUT = 2.0, 10.0
#: The harness records a failed request as "<exception type>: <message>". These two types are the
#: request-level failures (no judged answer). LLMOutputError and LLMRefusalError are the model's answer.
REQUEST_FAILURE_PREFIXES = ("LLMError: ", "LLMDisabledError: ")
STOP_TAG = "[resume-stop]"
KST = timezone(timedelta(hours=9))

SELFTEST_DATE = "2026-10-18"
SELFTEST_ERRORED = 103
SELFTEST_FAIL_AFTER = 40
SIMULATED_CREDIT_ERROR = (
    "LLMError: Anthropic 요청이 거부되었습니다 (HTTP 400, task=extract): simulated by the self-test - "
    f"{CREDIT_MARK} to access the Anthropic API."
)

_Pair = tuple[str, Condition]
_Flat = dict[tuple[str, str, str], int]
_PRF_FIELDS = ("tp", "fp", "fn")
_RATE_FIELDS = ("hits", "n")
Echo = Callable[[str], object]


class ResumeError(RuntimeError):
    """A check failed. Nothing is sent or written after this point."""


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ResumeError(message)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def metrics_bytes(metrics: EvalMetrics) -> bytes:
    """metrics.json exactly as :func:`jettae.evalkit.report.write_report` writes it."""
    payload = json.dumps(metrics.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2)
    return (payload + "\n").encode("utf-8")


def _first_difference(a: Any, b: Any, path: str = "") -> str:
    """Path of the first difference between two JSON values (for error messages)."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(a.keys() | b.keys()):
            if a.get(key, "<missing>") != b.get(key, "<missing>"):
                return _first_difference(a.get(key, "<missing>"), b.get(key, "<missing>"), f"{path}/{key}")
        return path or "/"
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            if x != y:
                return _first_difference(x, y, f"{path}[{i}]")
    return f"{path}: {str(a)[:120]} != {str(b)[:120]}"


# --------------------------------------------------------------------------- run 1 on disk


@dataclass(frozen=True)
class Run1:
    """What run 1 published. Read only: this script never writes into its folder."""

    path: Path
    lines: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    metrics: dict[str, Any]
    metrics_bytes: bytes
    report_bytes: bytes | None
    manifest_bytes: bytes | None
    conditions: tuple[Condition, ...]
    case_ids: tuple[str, ...]
    results_sha256: str


def load_run1(path: Path) -> Run1:
    results_file, metrics_file = path / "results.jsonl", path / "metrics.json"
    _require(results_file.is_file() and metrics_file.is_file(), f"run-1 files missing in {path}")
    raw = results_file.read_bytes()
    lines = tuple(raw.decode("utf-8").splitlines())
    rows = tuple(json.loads(line) for line in lines)
    raw_metrics = metrics_file.read_bytes()
    report, manifest = path / "report.md", path / "manifest.json"
    return Run1(
        path=path,
        lines=lines,
        rows=rows,
        metrics=json.loads(raw_metrics.decode("utf-8")),
        metrics_bytes=raw_metrics,
        report_bytes=report.read_bytes() if report.is_file() else None,
        manifest_bytes=manifest.read_bytes() if manifest.is_file() else None,
        conditions=tuple(dict.fromkeys(Condition(r["condition"]) for r in rows)),
        case_ids=tuple(dict.fromkeys(r["case_id"] for r in rows)),
        results_sha256=_sha256(raw),
    )


def _keys(values: Iterable[Sequence[str | None]]) -> frozenset[tuple[str | None, str]]:
    return frozenset((service, str(rule)) for service, rule in values)


def row_result(row: Mapping[str, Any]) -> CaseResult:
    """A results.jsonl row as a CaseResult. The file has no facts, so ``facts`` is None and no
    fact-based metric is ever taken from these objects."""
    return CaseResult(
        case_id=row["case_id"],
        condition=Condition(row["condition"]),
        fired=_keys(row["fired"]),
        severe=_keys(row["severe"]),
        facts=None,
        graph_hash=row["graph_hash"],
        ungrounded=row["ungrounded"],
        llm_calls=row["llm_calls"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        seconds=row["seconds"],
        error=row["error"],
    )


def check_rows(run1: Run1, cases: Sequence[EvalCase]) -> None:
    """The regenerated sample is run 1's: same case ids, one row per (case, condition) in harness order."""
    ids = tuple(c.case_id for c in cases)
    _require(
        run1.case_ids == ids,
        f"the regenerated sample ({len(ids)} cases) is not run 1's ({len(run1.case_ids)} cases)",
    )
    expected = [(cid, cond) for cid in ids for cond in run1.conditions]
    actual = [(r["case_id"], Condition(r["condition"])) for r in run1.rows]
    _require(actual == expected, "run-1 rows are not exactly one per (case, condition) in harness order")


def select_credit_pairs(run1: Run1, allowed: Sequence[Condition] = LLM_CONDITIONS) -> tuple[_Pair, ...]:
    """The (case, condition) pairs whose run-1 error is the exhausted credit, in run-1 order."""
    pairs: list[_Pair] = []
    for row in run1.rows:
        if CREDIT_MARK not in (row["error"] or ""):
            continue
        pair = (row["case_id"], Condition(row["condition"]))
        _require(pair[1] in allowed, f"{pair}: a credit failure on an arm that does not call a model")
        _require(
            row["n_facts"] is None and not row["fired"] and not row["severe"],
            f"{pair}: a credit failure that still has a graph or findings, so its share cannot be removed",
        )
        pairs.append(pair)
    by_arm = {arm: {cid for cid, cond in pairs if cond is arm} for arm in GRAPH_ARMS}
    if all(arm in run1.conditions for arm in GRAPH_ARMS):
        _require(
            by_arm[Condition.D_LLM_GRAPH] == by_arm[Condition.DR_HYBRID],
            "D and DR credit failures differ, although the two arms share one extraction per case",
        )
    return tuple(pairs)


@dataclass(frozen=True)
class Context:
    """Run 1 plus the regenerated corpus it was drawn from."""

    run1: Run1
    cases: tuple[EvalCase, ...]
    manifest: SealManifest
    seed: int | None
    reproducibility: Reproducibility | None
    cost_estimate: CostEstimate | None
    credit: tuple[_Pair, ...]


def _sample() -> tuple[tuple[EvalCase, ...], SealManifest]:
    corpus = kit.build_cases(kit.generate_gold_orgs(ORGS, SEED), seed=SEED)
    return tuple(reproducibility_sample(corpus, SAMPLE)), build_manifest(corpus)


def build_context(run1_dir: Path) -> Context:
    """Load run 1, regenerate its sample and check both, including a byte-exact rebuild."""
    run1 = load_run1(run1_dir)
    _require(run1.conditions == RUN1_CONDITIONS, f"run-1 conditions are {[c.value for c in run1.conditions]}")
    cases, manifest = _sample()
    check_rows(run1, cases)
    m = run1.metrics
    _require(m["manifest_root"] == manifest.merkle_root, "the regenerated corpus has another Merkle root")
    sealed = json.loads(run1.manifest_bytes or b"{}")
    _require(
        sealed.get("merkle_root") == manifest.merkle_root
        and sealed.get("held_out_merkle_root") == manifest.held_out_merkle_root,
        "run 1's manifest.json does not seal the regenerated corpus",
    )
    _require(m["corpus"]["seed"] == SEED and m["n_cases"] == len(cases), "run-1 seed or case count differs")
    _require(m["impact_benchmark"] is None and m["injection_selftest"] is None, "run 1 had benchmarks")
    estimate = estimate_llm_cost(cases, LLM_CONDITIONS, model=MODEL, echo=None)
    _require(
        _canonical(estimate.model_dump(mode="json")) == _canonical(m["llm_cost_estimate"]),
        "the cost estimate of the regenerated sample differs from run 1's (model or requests changed)",
    )
    repro = m["reproducibility"]
    ctx = Context(
        run1=run1,
        cases=cases,
        manifest=manifest,
        seed=SEED,
        reproducibility=Reproducibility.model_validate(repro) if repro is not None else None,
        cost_estimate=estimate,
        credit=select_credit_pairs(run1),
    )
    verify_reconstruction(ctx)
    return ctx


# --------------------------------------------------------------------------- merge


def _flatten(section: Mapping[str, Any], fields: tuple[str, ...]) -> _Flat:
    """FieldScores / Hallucination JSON → {(part, group, count field): n}; ``group`` is "" for the total."""
    flat: _Flat = {}
    for part, value in section.items():
        groups = {"": value} if part in ("all", "rate") else value
        for group, counts in groups.items():
            for name in fields:
                flat[(part, group, name)] = int(counts[name])
    return flat


def _combine(base: _Flat, minus: _Flat, plus: _Flat) -> _Flat:
    out = dict(base)
    for key, value in minus.items():
        out[key] = out.get(key, 0) - value
    for key, value in plus.items():
        out[key] = out.get(key, 0) + value
    negative = sorted(k for k, v in out.items() if v < 0)
    _require(not negative, f"the subtraction went negative at {negative[:3]}")
    return out


def _groups(flat: _Flat, part: str) -> set[str]:
    return {group for p, group, _ in flat if p == part}


def _prf(flat: _Flat, part: str, group: str = "") -> Prf:
    return Prf.of(flat[(part, group, "tp")], flat[(part, group, "fp")], flat[(part, group, "fn")])


def _rate(flat: _Flat, part: str, group: str = "") -> Rate:
    return Rate.of(flat[(part, group, "hits")], flat[(part, group, "n")])


def _field_json(flat: _Flat) -> dict[str, Any]:
    """FieldScores in the product's key order (styles as in STYLES, kinds sorted by value)."""
    styles, kinds = _groups(flat, "by_style"), _groups(flat, "by_kind")
    _require(styles <= set(STYLES), f"unknown styles {styles - set(STYLES)}")
    scores = FieldScores(
        all=_prf(flat, "all"),
        by_style={s: _prf(flat, "by_style", s) for s in STYLES if s in styles},
        by_kind={k: _prf(flat, "by_kind", k) for k in sorted(kinds)},
    )
    return scores.model_dump(mode="json")


def _hallucination_json(flat: _Flat) -> dict[str, Any]:
    """Hallucination in the product's key order (deletion operators, then every style)."""
    operators = _groups(flat, "by_operator")
    known = {op.value for op in DELETION_OPERATORS}
    _require(operators <= known, f"unknown operators {operators - known}")
    result = Hallucination(
        rate=_rate(flat, "rate"),
        by_operator={
            op.value: _rate(flat, "by_operator", op.value)
            for op in DELETION_OPERATORS
            if op.value in operators
        },
        by_style={s: _rate(flat, "by_style", s) for s in STYLES},
    )
    return result.model_dump(mode="json")


def _subset_sections(
    ctx: Context, condition: Condition, results: Sequence[CaseResult]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """field_f1 and hallucination of ``results`` alone, by the product's own metric code."""
    if not results:
        return {}, {}
    by_id = {c.case_id: c for c in ctx.cases}
    subset = EvalResults(
        cases=tuple(by_id[r.case_id] for r in results), conditions=(condition,), results=tuple(results)
    )
    metrics = compute_metrics(subset, generated_at="-")
    name = condition.value
    return (
        metrics.field_f1[name].model_dump(mode="json"),
        metrics.hallucination[name].model_dump(mode="json"),
    )


def merged_fact_sections(
    ctx: Context, condition: Condition, old: Sequence[CaseResult], new: Sequence[CaseResult]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run-1 field scores and hallucination, minus the replaced rows' share, plus their reruns'."""
    name = condition.value
    run1_field, run1_hall = ctx.run1.metrics["field_f1"][name], ctx.run1.metrics["hallucination"][name]
    summary = ctx.run1.metrics["conditions"][name]
    _require(
        run1_field["all"] == summary["field"] and run1_hall["rate"] == summary["hallucination"],
        f"{name}: run-1 metrics.json is not self-consistent",
    )
    old_field, old_hall = _subset_sections(ctx, condition, old)
    new_field, new_hall = _subset_sections(ctx, condition, new)
    old_hall_flat = _flatten(old_hall, _RATE_FIELDS)
    _require(
        not any(old_hall_flat.values()) and not _groups(old_hall_flat, "by_operator"),
        f"{name}: a replaced run-1 row counted for hallucination (it would need its facts)",
    )
    field = _combine(
        _flatten(run1_field, _PRF_FIELDS), _flatten(old_field, _PRF_FIELDS), _flatten(new_field, _PRF_FIELDS)
    )
    hall = _combine(_flatten(run1_hall, _RATE_FIELDS), old_hall_flat, _flatten(new_hall, _RATE_FIELDS))
    return _field_json(field), _hallucination_json(hall)


def merge_metrics(
    ctx: Context, replacements: Mapping[_Pair, CaseResult], *, generated_at: str
) -> tuple[EvalMetrics, tuple[CaseResult, ...]]:
    """Metrics of run 1 with ``replacements`` substituted, computed exactly (see the module docstring).

    Detection, false-positive, paired, strict, per-style/operator/position, error, ungrounded and
    LLM-use numbers come from the product's ``compute_metrics`` over the merged rows: they read only
    fired / severe / error / ungrounded / tokens / seconds, all of which results.jsonl holds. Field
    scores and hallucination come from :func:`merged_fact_sections`. An arm without replacements
    keeps run 1's latency as published (results.jsonl rounds seconds to 6 digits).
    """
    rows = {(r["case_id"], Condition(r["condition"])): r for r in ctx.run1.rows}
    for pair in replacements:
        _require(pair in rows, f"{pair} is not a run-1 pair")
        _require(
            rows[pair]["error"] is not None and rows[pair]["n_facts"] is None,
            f"{pair}: only an errored run-1 row without facts can be replaced exactly",
        )
    base = [row_result(r) for r in ctx.run1.rows]
    merged = tuple(replacements.get((r.case_id, r.condition), r) for r in base)
    results = EvalResults(
        cases=ctx.cases,
        conditions=ctx.run1.conditions,
        results=merged,
        workers=int(ctx.run1.metrics["workers"]),
    )
    fresh = compute_metrics(
        results,
        generated_at=generated_at,
        manifest=ctx.manifest,
        seed=ctx.seed,
        reproducibility=ctx.reproducibility,
        cost_estimate=ctx.cost_estimate,
    ).model_dump(mode="json")
    old = {(r.case_id, r.condition): r for r in base}
    for condition in ctx.run1.conditions:
        name = condition.value
        pairs = [p for p in replacements if p[1] is condition]
        if not pairs:
            fresh["conditions"][name]["latency"] = ctx.run1.metrics["conditions"][name]["latency"]
        if condition in GRAPH_CONDITIONS:
            field, hall = merged_fact_sections(
                ctx, condition, [old[p] for p in pairs], [replacements[p] for p in pairs]
            )
            fresh["field_f1"][name], fresh["hallucination"][name] = field, hall
            fresh["conditions"][name]["field"] = field["all"]
            fresh["conditions"][name]["hallucination"] = hall["rate"]
    return EvalMetrics.model_validate(fresh), merged


def merged_jsonl(ctx: Context, replacements: Mapping[_Pair, CaseResult]) -> str:
    """results.jsonl of the merge: run-1 lines verbatim, replaced pairs in the product's format."""
    out: list[str] = []
    for line, row in zip(ctx.run1.lines, ctx.run1.rows, strict=True):
        result = replacements.get((row["case_id"], Condition(row["condition"])))
        if result is None:
            out.append(line + "\n")
        else:
            out.append(
                results_jsonl(EvalResults(cases=(), conditions=(result.condition,), results=(result,)))
            )
    return "".join(out)


def verify_reconstruction(ctx: Context) -> None:
    """A merge with nothing replaced must give back run 1's metrics.json, report.md and
    results.jsonl byte for byte. That is the proof that the merge starts from what run 1 published
    and that results.jsonl is enough for every metric that does not need facts."""
    metrics, _ = merge_metrics(ctx, {}, generated_at=ctx.run1.metrics["generated_at"])
    rebuilt = metrics_bytes(metrics)
    if rebuilt != ctx.run1.metrics_bytes:
        where = _first_difference(json.loads(rebuilt), ctx.run1.metrics)
        raise ResumeError(f"rebuilt metrics.json differs from run 1's at {where}")
    if ctx.run1.report_bytes is not None:
        _require(
            render_report(metrics).encode("utf-8") == ctx.run1.report_bytes,
            "rebuilt report.md differs from run 1's",
        )
    jsonl = merged_jsonl(ctx, {}).encode("utf-8")
    _require(_sha256(jsonl) == ctx.run1.results_sha256, "rebuilt results.jsonl differs from run 1's")


# --------------------------------------------------------------------------- rerun store


def _fact_json(fact: Fact) -> list[Any]:
    return [EdgeKind(fact.kind).value, fact.src, fact.dst, [[k, v] for k, v in fact.attrs]]


def _fact_from_json(value: Sequence[Any]) -> Fact:
    kind, src, dst, attrs = value
    return Fact(EdgeKind(kind), src, dst, tuple((str(k), v) for k, v in attrs))


def _sorted_keys(keys: Iterable[tuple[str | None, str]]) -> list[list[str | None]]:
    return [[s, r] for s, r in sorted(keys, key=lambda k: (k[0] or "", k[1]))]


def result_to_json(result: CaseResult) -> dict[str, Any]:
    """A CaseResult with everything, facts included (results.jsonl only counts them)."""
    facts = None
    if result.facts is not None:
        facts = sorted((_fact_json(f) for f in result.facts), key=_canonical)
    return {
        "case_id": result.case_id,
        "condition": result.condition.value,
        "fired": _sorted_keys(result.fired),
        "severe": _sorted_keys(result.severe),
        "facts": facts,
        "graph_hash": result.graph_hash,
        "ungrounded": result.ungrounded,
        "llm_calls": result.llm_calls,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "seconds": result.seconds,
        "error": result.error,
    }


def result_from_json(item: Mapping[str, Any]) -> CaseResult:
    facts = item["facts"]
    return dataclasses.replace(
        row_result({**item, "n_facts": None}),
        facts=None if facts is None else frozenset(_fact_from_json(f) for f in facts),
    )


def new_store(ctx: Context) -> dict[str, Any]:
    return {
        "format": STORE_FORMAT,
        "run1": {"dir": ctx.run1.path.as_posix(), "results_sha256": ctx.run1.results_sha256},
        "model": MODEL,
        "effort": EFFORT,
        "concurrency": CONCURRENCY,
        "attempts": [],
    }


def load_store(path: Path, ctx: Context) -> dict[str, Any]:
    """The saved reruns (a new, empty store when there is none yet), bound to this exact run 1."""
    if not path.is_file():
        return new_store(ctx)
    store: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    _require(store.get("format") == STORE_FORMAT, f"{path} is not a rerun store")
    _require(
        store["run1"]["results_sha256"] == ctx.run1.results_sha256,
        f"{path} belongs to another run 1 (results.jsonl hash differs)",
    )
    _require(
        (store["model"], store["effort"], store["concurrency"]) == (MODEL, EFFORT, CONCURRENCY),
        f"{path} was made with another model, effort or concurrency",
    )
    return store


def save_store(path: Path, store: Mapping[str, Any]) -> None:
    """Write atomically: a crash mid-write never leaves a truncated store behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes((json.dumps(store, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    os.replace(tmp, path)


@dataclass(frozen=True)
class Rerun:
    attempt: int
    result: CaseResult
    #: request-level failure texts the stop guard saw during that attempt
    request_failures: frozenset[str]

    @property
    def unresolved(self) -> bool:
        return is_request_failure(self.result.error, self.request_failures)


def is_request_failure(error: str | None, seen: frozenset[str] = frozenset()) -> bool:
    """True when the request got no judged answer (the pair must be sent again)."""
    return error is not None and (error.startswith(REQUEST_FAILURE_PREFIXES) or error in seen)


def latest_reruns(store: Mapping[str, Any]) -> dict[_Pair, Rerun]:
    """Latest rerun per pair. D and DR are always sent together, so both come from one extraction."""
    out: dict[_Pair, Rerun] = {}
    for index, attempt in enumerate(store["attempts"]):
        seen = frozenset(attempt.get("request_failures", ()))
        for item in attempt["results"]:
            result = result_from_json(item)
            out[(result.case_id, result.condition)] = Rerun(index, result, seen)
    return out


def pending_pairs(ctx: Context, store: Mapping[str, Any]) -> list[_Pair]:
    latest = latest_reruns(store)
    return [p for p in ctx.credit if p not in latest or latest[p].unresolved]


def pending_cases(ctx: Context, store: Mapping[str, Any], arms: Sequence[Condition]) -> list[EvalCase]:
    wanted = {cid for cid, cond in pending_pairs(ctx, store) if cond in arms}
    return [c for c in ctx.cases if c.case_id in wanted]


# --------------------------------------------------------------------------- the paid rerun


class StopOnRequestFailure:
    """LLMClient wrapper: the first request that fails without a judged answer stops the run.

    Content failures (schema-invalid output, a refusal) are the model's answer and pass through.
    Any other failure means the remaining requests would fail the same way, and some of them could
    still be billed. That covers the credit running out again, a revoked key, a rate limit the SDK's
    retries did not absorb, and a network error. Every later request is then refused here, without
    being sent, with an LLMDisabledError. That error is recorded on its case as a request failure,
    and the next ``--yes`` sends the case again.
    """

    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner
        self.provider = inner.provider
        self.model = inner.model
        self.stopped: str | None = None
        self.failures: set[str] = set()

    async def structured(
        self,
        *,
        task: str,
        instructions: str,
        schema: type[_T],
        documents: tuple[UntrustedDocument, ...] = (),
        context: str = "",
        max_tokens: int = 16000,
    ) -> LLMResult[_T]:
        if self.stopped is not None:
            raise LLMDisabledError(f"{STOP_TAG} not sent: an earlier request failed ({self.stopped[:300]})")
        try:
            return await self.inner.structured(
                task=task,
                instructions=instructions,
                schema=schema,
                documents=documents,
                context=context,
                max_tokens=max_tokens,
            )
        except (LLMOutputError, LLMRefusalError):
            raise
        except Exception as exc:
            text = f"{type(exc).__name__}: {exc}"
            self.failures.add(text)
            if self.stopped is None:
                self.stopped = text
            raise


def execute(
    ctx: Context, store_path: Path, llm: LLMClient, *, batch: int = DEFAULT_BATCH, echo: Echo = print
) -> bool:
    """Send the pending pairs through ``run_eval``, batch by batch, saving after every batch.

    Returns True when no credit-failed pair is left without a judged answer.
    """
    store = load_store(store_path, ctx)
    guard = StopOnRequestFailure(llm)
    for label, arms in PHASES:
        todo = pending_cases(ctx, store, arms)
        batches = [todo[i : i + batch] for i in range(0, len(todo), batch)]
        for number, chunk in enumerate(batches, start=1):
            if guard.stopped is not None:
                break
            began = _now()
            results = run_eval(chunk, arms, llm=guard, workers=1, concurrency=CONCURRENCY)
            store["attempts"].append(
                {
                    "phase": label,
                    "conditions": [c.value for c in arms],
                    "started_at": began,
                    "finished_at": _now(),
                    "case_ids": [c.case_id for c in chunk],
                    "stopped_by": guard.stopped,
                    "request_failures": sorted(guard.failures),
                    "results": [result_to_json(r) for r in results.results],
                }
            )
            guard.failures.clear()
            save_store(store_path, store)
            errors = Counter(r.condition.value for r in results.results if r.error is not None)
            calls = sum(r.llm_calls for r in results.results if r.condition is arms[0])
            echo(
                f"  {label} batch {number}/{len(batches)}: {len(chunk)} cases, {calls} requests, "
                f"errors {dict(errors) or 0} - saved"
            )
        if guard.stopped is not None:
            echo(f"STOPPED after a request-level failure; nothing more was sent: {guard.stopped[:300]}")
            break
    return not pending_pairs(ctx, store)


def _close_harness_loop() -> None:
    """Close this thread's harness event loop now. Its atexit hook runs at interpreter shutdown,
    when the loop can no longer start the thread that shuts down its executor."""
    runner = getattr(harness_conditions._LOCAL, "runner", None)
    if runner is not None:
        with contextlib.suppress(Exception):
            runner.close()


def live_llm() -> tuple[LLMClient, Callable[[], None]]:
    """The live model run 1 used (provider, model, effort, timeout and retries as build_llm sets
    them). The key comes from load_api_key and goes straight to the SDK constructor: it is never
    printed and never written to the environment."""
    import anthropic

    from jettae.config import Settings
    from jettae.evalkit.realdata.keys import ANTHROPIC_KEY, load_api_key
    from jettae.llm.anthropic_client import AnthropicLLM

    settings = Settings(llm_provider="anthropic", anthropic_model=MODEL, anthropic_effort=EFFORT)
    key = load_api_key(ANTHROPIC_KEY)
    _require(bool(key), "ANTHROPIC_API_KEY was found neither in the environment nor in a .env file")
    sdk = anthropic.AsyncAnthropic(api_key=key, timeout=settings.llm_timeout_seconds, max_retries=2)
    del key
    print(f"ANTHROPIC_API_KEY found (value not shown); timeout {settings.llm_timeout_seconds:g}s, 2 retries")
    client = AnthropicLLM(
        settings.anthropic_model, settings.anthropic_effort, client=sdk, timeout=settings.llm_timeout_seconds
    )

    def close() -> None:
        with contextlib.suppress(Exception):
            run_sync(sdk.close())

    return client, close


# --------------------------------------------------------------------------- plan and estimate


def _usd(tokens_in: float, tokens_out: float, prices: tuple[float, float]) -> float:
    return tokens_in / 1e6 * prices[0] + tokens_out / 1e6 * prices[1]


def _estimate(
    ctx: Context,
    label: str,
    cases: Sequence[EvalCase],
    arms: Sequence[Condition],
    prices: tuple[float, float],
) -> tuple[float, float]:
    """Print the product's estimate and a run-1-calibrated projection; return the projected USD
    (heuristic, calibrated)."""
    estimate = estimate_llm_cost(cases, arms, model=MODEL, echo=None)
    print(f"[{label}] " + estimate.render().replace("\n", f"\n[{label}] "))
    heuristic = _usd(estimate.total_input_tokens, estimate.total_output_tokens, prices)
    by_id = {c.case_id: c for c in ctx.cases}
    measured = [r for r in ctx.run1.rows if r["condition"] == arms[0].value and r["error"] is None]
    if not measured:
        return heuristic, heuristic
    basis = estimate_llm_cost([by_id[r["case_id"]] for r in measured], arms[:1], model=MODEL, echo=None)
    ratio_in = sum(r["input_tokens"] for r in measured) / max(1, basis.total_input_tokens)
    ratio_out = sum(r["output_tokens"] for r in measured) / max(1, basis.total_output_tokens)
    tokens_in = estimate.total_input_tokens * ratio_in
    tokens_out = estimate.total_output_tokens * ratio_out
    calibrated = _usd(tokens_in, tokens_out, prices)
    print(
        f"[{label}] run-1 actual/estimate on its {len(measured)} {arms[0].value} rows without error: "
        f"input x{ratio_in:.2f}, output x{ratio_out:.2f} -> about {tokens_in:,.0f} input + "
        f"{tokens_out:,.0f} output tokens"
    )
    print(f"[{label}] cost: heuristic ${heuristic:.2f}, run-1 calibrated ${calibrated:.2f}")
    return heuristic, calibrated


def show_plan(ctx: Context, store: Mapping[str, Any], prices: tuple[float, float]) -> list[_Pair]:
    run1 = ctx.run1
    print(
        f"run 1: {run1.path.as_posix()} - {len(ctx.cases)} cases x {len(run1.conditions)} conditions = "
        f"{len(run1.rows):,} rows. Sample regenerated and matched (seed {SEED}, {ORGS} orgs, "
        f"{SAMPLE:.0%}, Merkle root {ctx.manifest.merkle_root[:12]}...). Run 1's metrics.json, report.md "
        "and results.jsonl were rebuilt byte for byte from its rows."
    )
    credit = Counter(c.value for _, c in ctx.credit)
    print(f"credit-failed pairs ('{CREDIT_MARK}'): {dict(credit)} = {len(ctx.credit)} pairs")
    llm_rows = [r for r in run1.rows if Condition(r["condition"]) in LLM_CONDITIONS and r["error"]]
    kept = [r for r in llm_rows if CREDIT_MARK not in r["error"]]
    reasons = Counter(f"{r['condition']}: {r['error'][:60]}" for r in kept)
    print(f"kept as measured, never re-run (content failures): {len(kept)} rows")
    for reason, n in sorted(reasons.items()):
        print(f"  {n:3d} x {reason}")
    other = [r for r in kept if is_request_failure(r["error"])]
    print(f"other request-level failures in run 1 (kept as errors): {len(other)}")
    pending = pending_pairs(ctx, store)
    print(f"already re-run with a judged answer ({STORE_NAME}): {len(ctx.credit) - len(pending)} pairs")
    graph = pending_cases(ctx, store, GRAPH_ARMS)
    direct = pending_cases(ctx, store, DIRECT_ARMS)
    print(
        f"to send: {len(pending)} pairs = D+DR on {len(graph)} cases (one extraction each, shared by D "
        f"and DR) + C on {len(direct)} cases; model {MODEL}, effort {EFFORT}, concurrency {CONCURRENCY}"
    )
    totals = [0.0, 0.0]
    for label, cases, arms in (("D+DR", graph, GRAPH_ARMS), ("C", direct, DIRECT_ARMS)):
        if cases:
            heuristic, calibrated = _estimate(ctx, label, cases, arms, prices)
            totals[0] += heuristic
            totals[1] += calibrated
    if pending:
        print(
            f"total estimated cost: heuristic ${totals[0]:.2f}, run-1 calibrated ${totals[1]:.2f} "
            f"(at ${prices[0]:g}/${prices[1]:g} per million input/output tokens). Approximate: the harness "
            "records neither prompt-cache reads/writes nor the tokens of rejected answers (e.g. a "
            "max_tokens truncation, billed but not counted)"
        )
    return pending


# --------------------------------------------------------------------------- outputs


def _summary_row(summary: Mapping[str, Any]) -> dict[str, Any]:
    def pct(rate: Mapping[str, Any] | None) -> str:
        if rate is None or rate.get("rate") is None:
            return "-"
        return f"{100 * rate['rate']:.1f}% ({rate['hits']}/{rate['n']})"

    field = summary.get("field")
    return {
        "errors": summary["errors"],
        "detection_active": pct(summary["detection_active"]),
        "detection_held_out": pct(summary["detection_held_out"]),
        "detection_paired_active": pct(summary["detection_paired_active"]),
        "clean_fp": pct(summary["clean_fp"]),
        "field_f1": None if field is None else field["f1"],
        "hallucination": pct(summary.get("hallucination")),
        "llm_calls": summary["llm"]["total_calls"],
        "input_tokens": summary["llm"]["total_input_tokens"],
        "output_tokens": summary["llm"]["total_output_tokens"],
    }


def _merge_log(
    ctx: Context,
    *,
    store_path: Path,
    latest: Mapping[_Pair, Rerun],
    replacements: Mapping[_Pair, CaseResult],
    unresolved: Sequence[_Pair],
    metrics: EvalMetrics,
) -> dict[str, Any]:
    rows = {(r["case_id"], Condition(r["condition"])): r for r in ctx.run1.rows}
    attempts = json.loads(store_path.read_text(encoding="utf-8"))["attempts"] if store_path.is_file() else []
    replaced = []
    for pair, result in replacements.items():
        rerun = latest[pair]
        replaced.append(
            {
                "case_id": pair[0],
                "condition": pair[1].value,
                "run1_error": rows[pair]["error"],
                "rerun_error": result.error,
                "rerun_attempt": rerun.attempt,
                "rerun_started_at": attempts[rerun.attempt]["started_at"],
                "n_facts": None if result.facts is None else len(result.facts),
                "fired": len(result.fired),
                "severe": len(result.severe),
                "llm_calls": result.llm_calls,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "seconds": round(result.seconds, 6),
            }
        )
    kept = [
        {"case_id": r["case_id"], "condition": r["condition"], "error": r["error"]}
        for r in ctx.run1.rows
        if r["error"] is not None and (r["case_id"], Condition(r["condition"])) not in replacements
    ]
    merged = metrics.model_dump(mode="json")["conditions"]
    return {
        "format": LOG_FORMAT,
        "generated_at": metrics.generated_at,
        "run1": {
            "dir": ctx.run1.path.as_posix(),
            "command": RUN1_COMMAND,
            "results_sha256": ctx.run1.results_sha256,
            "metrics_sha256": _sha256(ctx.run1.metrics_bytes),
        },
        "rerun": {
            "store": store_path.as_posix(),
            "attempts": len(attempts),
            "model": MODEL,
            "effort": EFFORT,
            "concurrency": CONCURRENCY,
            "batches": [
                {"phase": a["phase"], "started_at": a["started_at"], "cases": len(a["case_ids"])}
                for a in attempts
            ],
            "note": "D and DR were sent together (one shared extraction per case), C separately, one "
            "run_eval call per batch; per-case latency is measured inside the harness's case gate as in "
            "run 1",
        },
        "policy": f"Only pairs whose run-1 error contains '{CREDIT_MARK}' were re-run (they never reached "
        "the model). Every other run-1 row is kept as published, content failures included.",
        "method": "Detection-type metrics are recomputed by compute_metrics over the merged rows "
        "(fired/severe/error suffice). Field scores and hallucination = run-1 totals - replaced run-1 "
        "rows' share (errored, facts None: fn = |gold| only, no hallucination) + reruns' share. A, B "
        "and O are unchanged from run 1.",
        "counts": {
            "replaced": len(replacements),
            "replaced_by_condition": dict(Counter(p[1].value for p in replacements)),
            "rerun_errors_by_condition": dict(
                Counter(p[1].value for p, r in replacements.items() if r.error is not None)
            ),
            "unresolved": len(unresolved),
            "kept_run1_errors_by_condition": dict(Counter(k["condition"] for k in kept)),
        },
        "summary": {
            name: {
                "run1": _summary_row(ctx.run1.metrics["conditions"][name]),
                "final": _summary_row(merged[name]),
            }
            for name in merged
        },
        "replaced": replaced,
        "unresolved": [{"case_id": cid, "condition": cond.value} for cid, cond in unresolved],
        "kept_run1_errors": kept,
    }


def write_outputs(
    ctx: Context,
    store_path: Path,
    out: Path,
    *,
    generated_at: str,
    accept_unresolved: bool = False,
    echo: Echo = print,
) -> EvalMetrics:
    """Merge the saved reruns into run 1 and write the final folder."""
    store = load_store(store_path, ctx)
    latest = latest_reruns(store)
    unresolved = [p for p in ctx.credit if p not in latest or latest[p].unresolved]
    _require(
        not unresolved or accept_unresolved,
        f"{len(unresolved)} credit-failed pairs still have no judged rerun: run --yes again "
        "(or --merge-only --accept-unresolved to count them as errors)",
    )
    replacements = {p: latest[p].result for p in ctx.credit if p in latest}
    metrics, _ = merge_metrics(ctx, replacements, generated_at=generated_at)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_bytes(metrics_bytes(metrics))
    (out / "report.md").write_bytes(render_report(metrics).encode("utf-8"))
    (out / "results.jsonl").write_bytes(merged_jsonl(ctx, replacements).encode("utf-8"))
    if ctx.run1.manifest_bytes is not None:
        (out / "manifest.json").write_bytes(ctx.run1.manifest_bytes)
    log = _merge_log(
        ctx,
        store_path=store_path,
        latest=latest,
        replacements=replacements,
        unresolved=unresolved,
        metrics=metrics,
    )
    (out / "merge_log.json").write_bytes(
        (json.dumps(log, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    )
    echo(f"wrote {out.as_posix()}: metrics.json, report.md, results.jsonl, manifest.json, merge_log.json")
    echo(f"{'condition':14s} {'errors':>9s}  {'detection (active)':>30s}  {'clean FP':>28s}  field F1")
    for name, row in log["summary"].items():
        before, after = row["run1"], row["final"]
        echo(
            f"{name:14s} {before['errors']:>4d}->{after['errors']:<4d}  "
            f"{before['detection_active']:>14s}->{after['detection_active']:<15s} "
            f"{before['clean_fp']:>13s}->{after['clean_fp']:<14s} {before['field_f1']}->{after['field_f1']}"
        )
    return metrics


# --------------------------------------------------------------------------- self-test


def _hash_pick(ids: Iterable[str], n: int) -> set[str]:
    return set(sorted(ids, key=lambda i: hashlib.sha256(f"resume-selftest:{i}".encode()).hexdigest())[:n])


def _as_credit_failure(result: CaseResult) -> CaseResult:
    """``result`` as a credit-exhausted request leaves it: no graph, no findings, no tokens."""
    return dataclasses.replace(
        result,
        fired=frozenset(),
        severe=frozenset(),
        facts=None,
        graph_hash=None,
        ungrounded=0,
        llm_calls=0,
        input_tokens=0,
        output_tokens=0,
        error=SIMULATED_CREDIT_ERROR,
    )


def _fact_parts(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "field_f1": metrics["field_f1"],
        "hallucination": metrics["hallucination"],
        "summary": {n: (s["field"], s["hallucination"]) for n, s in metrics["conditions"].items()},
    }


def _without(metrics: Mapping[str, Any], *, latency: bool = False, facts: bool = False) -> dict[str, Any]:
    out = copy.deepcopy(dict(metrics))
    for summary in out["conditions"].values():
        if latency:
            summary.pop("latency")
        if facts:
            summary.pop("field")
            summary.pop("hallucination")
    if facts:
        out.pop("field_f1")
        out.pop("hallucination")
    return out


def _simulated_run1(
    work: Path, cases: tuple[EvalCase, ...], manifest: SealManifest, first: EvalResults, errored: set[str]
) -> Context:
    """Run 1 as it would look had the credit run out on ``errored``: B's results with those pairs
    turned into credit failures, written by the product's own write_report, then read back."""
    b = Condition.B_RULE_GRAPH
    simulated = tuple(_as_credit_failure(r) if r.case_id in errored else r for r in first.results)
    run1_dir = work / "simulated-run1"
    write_report(
        EvalResults(cases, (b,), simulated, workers=first.workers),
        run1_dir,
        generated_at=SELFTEST_DATE,
        manifest=manifest,
        seed=SEED,
    )
    write_manifest(manifest, run1_dir / "manifest.json")
    run1 = load_run1(run1_dir)
    check_rows(run1, cases)
    ctx = Context(run1, cases, manifest, SEED, None, None, select_credit_pairs(run1, allowed=(b,)))
    _require(len(ctx.credit) == len(errored), f"selected {len(ctx.credit)} simulated credit failures")
    verify_reconstruction(ctx)
    return ctx


def _counts(section: Mapping[str, Any], fields: tuple[str, ...]) -> str:
    return "/".join(str(section[f]) for f in fields)


def selftest_subtraction(work: Path, *, workers: int | None, echo: Echo = print) -> None:
    """Condition B twice on the 264 cases. Errored pairs are simulated in a copy of the first run;
    the merge must rebuild the full numbers exactly from that run-1 folder and the second run."""
    b = Condition.B_RULE_GRAPH
    cases, manifest = _sample()
    first = run_eval(cases, (b,), workers=workers)
    second = run_eval(cases, (b,), workers=workers)
    for x, y in zip(first.results, second.results, strict=True):
        same = (x.case_id, x.fired, x.severe, x.facts, x.graph_hash, x.error)
        again = (y.case_id, y.fired, y.severe, y.facts, y.graph_hash, y.error)
        _require(same == again, f"B differs on {x.case_id}")
    _require(all(r.error is None for r in first.results), "condition B errored on a case")
    errored = _hash_pick((c.case_id for c in cases), SELFTEST_ERRORED)
    chosen = [c for c in cases if c.case_id in errored]
    n_clean = sum(c.is_clean for c in chosen)
    n_deletion = sum(c.operator in DELETION_OPERATORS for c in chosen)
    _require(n_clean > 0 and n_deletion > 0, "the simulated subset must hold clean and deletion cases")
    ctx = _simulated_run1(work, cases, manifest, first, errored)
    replacements: dict[_Pair, CaseResult] = {}
    for result in second.results:
        if result.case_id in errored:
            back = result_from_json(json.loads(_canonical(result_to_json(result))))
            _require(back == result, f"the store round trip changed {result.case_id}")
            replacements[(result.case_id, b)] = back
    got = merge_metrics(ctx, replacements, generated_at=SELFTEST_DATE)[0].model_dump(mode="json")
    hybrid = tuple(replacements.get((r.case_id, b), r) for r in first.results)

    def metrics_of(results: Sequence[CaseResult]) -> dict[str, Any]:
        run = EvalResults(cases, (b,), tuple(results), workers=first.workers)
        return compute_metrics(run, generated_at=SELFTEST_DATE, manifest=manifest, seed=SEED).model_dump(
            mode="json"
        )

    truth, full = metrics_of(hybrid), metrics_of(first.results)
    for name, want in (("the rerun-substituted run", truth), ("the first full run", full)):
        if _fact_parts(got) != _fact_parts(want):
            where = _first_difference(_fact_parts(got), _fact_parts(want))
            raise ResumeError(f"merged field/hallucination != {name} at {where}")
    rest_got, rest_want = _without(got, latency=True), _without(truth, latency=True)
    _require(rest_got == rest_want, f"merged metrics differ at {_first_difference(rest_got, rest_want)}")
    latency = truth["conditions"][b.value]["latency"]
    gap = max(abs(got["conditions"][b.value]["latency"][k] - v) for k, v in latency.items())
    _require(gap <= 2e-4, f"latency differs by {gap} (more than the 6-digit rounding of results.jsonl)")
    no_facts = metrics_of([dataclasses.replace(r, facts=None) for r in first.results])
    _require(_without(no_facts, facts=True) == _without(full, facts=True), "a non-fact metric needs facts")
    sim = ctx.run1.metrics
    parts = [
        ("field tp/fp/fn", ("field_f1", "all"), _PRF_FIELDS),
        ("hallucination hits/n", ("hallucination", "rate"), _RATE_FIELDS),
    ]
    _require(
        sim["field_f1"][b.value]["all"]["fn"] > truth["field_f1"][b.value]["all"]["fn"]
        and sim["hallucination"][b.value]["rate"]["n"] < truth["hallucination"][b.value]["rate"]["n"],
        "the simulated failures changed nothing: the test would be vacuous",
    )
    echo(
        f"  B on {len(cases)} cases twice: identical. {len(errored)} pairs turned into credit failures "
        f"({n_clean} clean controls, {n_deletion} deletion cases)"
    )
    for label, (section, key), fields in parts:
        echo(
            f"  {label}: simulated run 1 {_counts(sim[section][b.value][key], fields)} -> merged "
            f"{_counts(got[section][b.value][key], fields)} = "
            f"full {_counts(truth[section][b.value][key], fields)}"
        )
    echo(
        "  by_style, by_kind and by_operator equal as well; every other metric equals compute_metrics on "
        f"the full results (latency within rounding, {gap:.1e}); with facts stripped no non-fact metric moves"
    )


class _FakeModel:
    """FakeLLM handler: empty but valid answers. After ``fail_after`` answers it returns the
    credit error the live API gave in run 1."""

    def __init__(self, fail_after: int | None) -> None:
        self.fail_after = fail_after
        self.calls = 0

    def __call__(
        self, task: str, schema: type[BaseModel], documents: tuple[UntrustedDocument, ...], context: str
    ) -> BaseModel | Exception:
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            return LLMError(
                f"Anthropic 요청이 거부되었습니다 (HTTP 400, task={task}): simulated - {CREDIT_MARK}."
            )
        return schema()


def selftest_pipeline(work: Path, run1_dir: Path, echo: Echo = print) -> None:
    """The --yes path against the real run 1 (read only), with a fake model. The credit runs out
    again after a few answers, the run stops and saves, a second run resumes and completes, and the
    merge is written to ``work``."""
    ctx = build_context(run1_dir)
    out = work / "final"
    store_path = out / STORE_NAME
    store_path.unlink(missing_ok=True)
    total = len(pending_pairs(ctx, load_store(store_path, ctx)))
    _require(total == len(ctx.credit), "a fresh store must leave every credit failure pending")
    starved = _FakeModel(SELFTEST_FAIL_AFTER)
    done = execute(ctx, store_path, FakeLLM(starved, model=MODEL), echo=lambda s: echo(f"  {s}"))
    left = pending_pairs(ctx, load_store(store_path, ctx))
    _require(
        not done and 0 < len(left) < total, f"the stop did not leave a partial run ({len(left)}/{total})"
    )
    _require(starved.calls <= SELFTEST_FAIL_AFTER + CONCURRENCY, f"{starved.calls} requests after the stop")
    _require(not (out / "metrics.json").exists(), "the stopped run wrote a merge")
    try:
        write_outputs(ctx, store_path, out, generated_at=SELFTEST_DATE)
    except ResumeError:
        pass
    else:
        raise ResumeError("the merge accepted unresolved pairs")
    healthy = _FakeModel(None)
    _require(
        execute(ctx, store_path, FakeLLM(healthy, model=MODEL), echo=lambda s: echo(f"  {s}")),
        "resume failed",
    )
    metrics = write_outputs(ctx, store_path, out, generated_at=SELFTEST_DATE, echo=lambda s: echo(f"  {s}"))
    reread = EvalMetrics.model_validate_json((out / "metrics.json").read_text(encoding="utf-8"))
    _require(
        reread == metrics and (out / "report.md").read_text(encoding="utf-8") == render_report(metrics),
        "files",
    )
    log = json.loads((out / "merge_log.json").read_text(encoding="utf-8"))
    _require(len(log["replaced"]) == len(ctx.credit) and not log["unresolved"], "merge log incomplete")
    got, run1 = metrics.model_dump(mode="json"), ctx.run1.metrics
    for cond in OFFLINE_CONDITIONS:
        for section in ("conditions", "by_style", "by_operator", "by_position", "clean_fp"):
            _require(got[section][cond.value] == run1[section][cond.value], f"{cond.value} {section} changed")
    for cond in (Condition.B_RULE_GRAPH, Condition.O_ORACLE):
        for section in ("field_f1", "hallucination"):
            _require(got[section][cond.value] == run1[section][cond.value], f"{cond.value} {section} changed")
    latest = latest_reruns(load_store(store_path, ctx))
    for cond in LLM_CONDITIONS:
        kept = sum(
            1
            for r in ctx.run1.rows
            if r["condition"] == cond.value and r["error"] and CREDIT_MARK not in r["error"]
        )
        new = sum(1 for p, r in latest.items() if p[1] is cond and r.result.error is not None)
        _require(got["conditions"][cond.value]["errors"] == kept + new, f"{cond.value} error count")
    lines = (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
    replaced = {(i["case_id"], i["condition"]) for i in log["replaced"]}
    _require(len(lines) == len(ctx.run1.lines), "results.jsonl row count changed")
    for line, original in zip(lines, ctx.run1.lines, strict=True):
        row = json.loads(original)
        _require(line == original or (row["case_id"], row["condition"]) in replaced, "a kept row changed")
    echo(
        f"  fake model: stopped after {starved.calls} requests with {len(left)}/{total} pairs left, saved; "
        f"resumed with {healthy.calls} requests; merge written; A/B/O sections identical to run 1"
    )


def run_self_test(work_dir: Path | None, run1_dir: Path, workers: int | None) -> int:
    with contextlib.ExitStack() as stack:
        work = work_dir or Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="jettae-resume-")))
        work.mkdir(parents=True, exist_ok=True)
        print(f"self-test in {work.as_posix()} (no request leaves this machine)")
        print("[1/2] subtraction merge on condition B (offline)")
        selftest_subtraction(work / "b-subtraction", workers=workers)
        print("[2/2] resume pipeline on run 1 with a fake model")
        selftest_pipeline(work / "fake-pipeline", run1_dir)
    print("SELF-TEST PASSED")
    return 0


# --------------------------------------------------------------------------- command line


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0] if __doc__ else None)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print pairs and cost, send nothing (default)")
    mode.add_argument("--yes", action="store_true", help="PAID: send the credit-failed pairs, then merge")
    mode.add_argument("--merge-only", action="store_true", help="merge the saved reruns, send nothing")
    mode.add_argument("--self-test", action="store_true", help="offline validation of the merge and resume")
    p.add_argument("--run1", type=Path, default=RUN1_DIR, help="run-1 folder (read only)")
    p.add_argument("--out", type=Path, default=FINAL_DIR, help="final folder (rerun store and merge)")
    p.add_argument("--generated-at", help="date for the report (default: today, KST)")
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="cases per saved batch")
    p.add_argument(
        "--accept-unresolved", action="store_true", help="--merge-only: count unresolved as errors"
    )
    p.add_argument("--usd-per-mtok-in", type=float, default=USD_PER_MTOK_IN)
    p.add_argument("--usd-per-mtok-out", type=float, default=USD_PER_MTOK_OUT)
    p.add_argument("--work-dir", type=Path, help="--self-test: keep its files here")
    p.add_argument("--workers", type=int, help="--self-test: processes for condition B")
    return p


def _dispatch(args: argparse.Namespace) -> int:
    if args.self_test:
        return run_self_test(args.work_dir, args.run1, args.workers)
    _require(args.batch >= 1, "--batch must be at least 1")
    run1, out = args.run1.resolve(), args.out.resolve()
    _require(out != run1 and run1 not in out.parents, "--out must be outside the run-1 folder (read only)")
    ctx = build_context(args.run1)
    store_path = args.out / STORE_NAME
    store = load_store(store_path, ctx)
    pending = show_plan(ctx, store, (args.usd_per_mtok_in, args.usd_per_mtok_out))
    generated_at = args.generated_at or datetime.now(KST).date().isoformat()
    if args.merge_only:
        write_outputs(
            ctx, store_path, args.out, generated_at=generated_at, accept_unresolved=args.accept_unresolved
        )
        return 0
    if not args.yes:
        print("DRY RUN: nothing was sent and nothing was written. --yes sends the requests above (paid).")
        return 0
    if pending:
        llm, close = live_llm()
        try:
            done = execute(ctx, store_path, llm, batch=args.batch)
        finally:
            close()
        if not done:
            left = len(pending_pairs(ctx, load_store(store_path, ctx)))
            print(
                f"INCOMPLETE: {left} pairs still need a judged answer; "
                f"finished batches are saved in {store_path.as_posix()}."
            )
            print("Fix the cause (e.g. top up the credit) and run --yes again; nothing done is sent twice.")
            return 3
    write_outputs(ctx, store_path, args.out, generated_at=generated_at)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    with contextlib.suppress(AttributeError):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    try:
        return _dispatch(args)
    except ResumeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        _close_harness_loop()


if __name__ == "__main__":
    raise SystemExit(main())
