"""Evaluation harness (평가 하네스): run the conditions over the sealed corpus, reproducibly.

Why it is shaped this way:

* **Offline arms in a process pool, results in corpus order.** A, B and O are CPU-bound pure
  functions of a case, so they fan out over ``os.cpu_count() - 2`` worker processes (at most 61
  on Windows, the platform's limit for a process pool); results are collected with
  ``Executor.map`` (input order) and finally sorted by (case, condition), so the output never
  depends on scheduling (P6).
* **Paid arms only on request.** C, D and DR require an explicit ``llm`` argument — there is no
  default client, so a forgotten flag can never start a paid run — and
  :func:`estimate_llm_cost` prints the calls/token estimate from the inputs alone *before* any
  call is made. D and DR share one LLM extraction per case (see
  :func:`~jettae.evalkit.conditions.arun_llm_graphs`). ``concurrency`` bounds the *requests* in
  flight, not the cases: a long runbook fans out into several chunk calls, and a provider rate
  limit does not care which case a request belongs to.
* **Reproducibility is measured, not assumed.** :func:`check_reproducibility` re-runs an arm on a
  deterministic 5% sample in a freshly *spawned* interpreter — never the process (or a fork of
  it) that produced the first run, so it has its own string-hash seed and cold caches — and
  compares graph hashes, fired sets and facts byte for byte.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import multiprocessing
import os
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, computed_field

from jettae.evalkit.conditions import (
    DIRECT_INSTRUCTIONS,
    LLM_CONDITIONS,
    OFFLINE_CONDITIONS,
    CaseResult,
    Condition,
    DirectDefectPayload,
    arun_llm_direct,
    arun_llm_graphs,
    case_inputs,
    direct_context,
    prepare_worker,
    run_offline_condition,
    run_sync,
)
from jettae.evalkit.corpus import EvalCase
from jettae.extract.llm_extractor import (
    EXTRACTION_INSTRUCTIONS,
    LlmGraphPayload,
    blank_quarantined,
    build_extraction_context,
    chunk_document,
)
from jettae.llm.base import LLMClient, LLMResult, UntrustedDocument
from jettae.llm.schema import to_strict_json_schema

_Payload = TypeVar("_Payload", bound=BaseModel)

__all__ = [
    "CHARS_PER_TOKEN",
    "DEFAULT_LLM_CONCURRENCY",
    "DIRECT_OUTPUT_TOKENS",
    "EXTRACTION_OUTPUT_RATIO",
    "REPRODUCIBILITY_FRACTION",
    "WINDOWS_MAX_WORKERS",
    "ConditionCost",
    "CostEstimate",
    "EvalResults",
    "Reproducibility",
    "check_reproducibility",
    "default_workers",
    "estimate_llm_cost",
    "parse_conditions",
    "reproducibility_sample",
    "run_eval",
]

DEFAULT_LLM_CONCURRENCY = 4
#: Share of the corpus re-run for the reproducibility check.
REPRODUCIBILITY_FRACTION = 0.05
#: Token heuristic of the cost estimate: characters per token (Korean-heavy text and JSON).
CHARS_PER_TOKEN = 2.5
#: Expected output of one direct-judgement call: a short list of (service, defect) pairs.
DIRECT_OUTPUT_TOKENS = 300
#: Extraction output relative to the chunk's input tokens: every entity and relation repeats a
#: verbatim quote and adds kinds, names and attributes, so the JSON outgrows the text it reads.
EXTRACTION_OUTPUT_RATIO = 2.0
#: ``ProcessPoolExecutor`` refuses more workers than this on Windows (``ValueError``).
WINDOWS_MAX_WORKERS = 61


def _pool_limit() -> int | None:
    return WINDOWS_MAX_WORKERS if sys.platform == "win32" else None


def _pool_size(workers: int, jobs: int) -> int:
    """Processes actually started for ``jobs`` batches with ``workers`` requested (≥ 1)."""
    size = max(1, min(workers, jobs))
    limit = _pool_limit()
    return size if limit is None else min(size, limit)


def default_workers() -> int:
    """Worker processes for the offline arms: all cores but two (the machine stays usable),
    capped at the platform's process-pool limit so a many-core Windows host does not crash."""
    workers = max(1, (os.cpu_count() or 1) - 2)
    limit = _pool_limit()
    return workers if limit is None else min(workers, limit)


def parse_conditions(values: Iterable[Condition | str]) -> tuple[Condition, ...]:
    """Validated, de-duplicated conditions in the order given (``"A_keyword"`` or the enum)."""
    out: list[Condition] = []
    for value in values:
        try:
            condition = Condition(value)
        except ValueError:
            known = ", ".join(c.value for c in Condition)
            raise ValueError(f"알 수 없는 평가 조건입니다: {value} (가능한 값: {known})") from None
        if condition not in out:
            out.append(condition)
    if not out:
        raise ValueError("평가 조건을 하나 이상 지정해야 합니다")
    return tuple(out)


@dataclass(frozen=True)
class EvalResults:
    """Every (case, condition) result of one run, ordered by corpus position then condition."""

    cases: tuple[EvalCase, ...]
    conditions: tuple[Condition, ...]
    results: tuple[CaseResult, ...]
    #: worker processes the offline arms ran on (latencies are measured under that parallel load)
    workers: int = 1

    def of(self, condition: Condition) -> tuple[CaseResult, ...]:
        """Results of one condition, in corpus order."""
        return tuple(r for r in self.results if r.condition is condition)

    def get(self, case_id: str, condition: Condition) -> CaseResult:
        for r in self.results:
            if r.case_id == case_id and r.condition is condition:
                return r
        raise KeyError(f"{case_id} / {condition}")

    def case(self, case_id: str) -> EvalCase:
        for c in self.cases:
            if c.case_id == case_id:
                return c
        raise KeyError(case_id)


# --------------------------------------------------------------------------- running


def _offline_batch(job: tuple[tuple[EvalCase, ...], tuple[Condition, ...]]) -> list[CaseResult]:
    """Pool worker: every requested offline condition on a batch of cases (top level → picklable)."""
    cases, conditions = job
    return [run_offline_condition(case, condition) for case in cases for condition in conditions]


@dataclass
class _Progress:
    """Counts finished (case, condition) results across the offline and LLM phases."""

    callback: Callable[[int, int], None] | None
    total: int
    done: int = 0

    def advance(self, n: int) -> None:
        self.done += n
        if self.callback is not None:
            self.callback(self.done, self.total)


def _run_offline(
    cases: Sequence[EvalCase], conditions: Sequence[Condition], workers: int, progress: _Progress
) -> tuple[list[CaseResult], int]:
    """(results in corpus order, worker processes used — 1 means this process)."""
    if not conditions:
        return [], 1
    # ~4 batches per worker balances uneven case costs without drowning in pickling overhead
    size = max(1, math.ceil(len(cases) / (max(1, workers) * 4)))
    jobs = [(tuple(cases[i : i + size]), tuple(conditions)) for i in range(0, len(cases), size)]
    processes = _pool_size(workers, len(jobs))
    results: list[CaseResult] = []
    if processes <= 1:
        prepare_worker()
        for case in cases:
            batch = _offline_batch(((case,), tuple(conditions)))
            results.extend(batch)
            progress.advance(len(batch))
        return results, 1
    with ProcessPoolExecutor(max_workers=processes, initializer=prepare_worker) as pool:
        for batch in pool.map(_offline_batch, jobs):
            results.extend(batch)
            progress.advance(len(batch))
    return results, processes


class _GatedLLM:
    """An :class:`LLMClient` whose requests share one semaphore — the harness's ``concurrency``.

    Gating cases is not enough: one case may send a direct call and several chunk extractions
    (each :class:`~jettae.extract.llm_extractor.LlmExtractor` fans its chunks out on its own), so
    only a gate around every single request keeps the promised bound on a paid provider.
    """

    def __init__(self, inner: LLMClient, gate: asyncio.Semaphore) -> None:
        self.inner = inner
        self.gate = gate
        self.provider = inner.provider
        self.model = inner.model

    async def structured(
        self,
        *,
        task: str,
        instructions: str,
        schema: type[_Payload],
        documents: tuple[UntrustedDocument, ...] = (),
        context: str = "",
        max_tokens: int = 16000,
    ) -> LLMResult[_Payload]:
        async with self.gate:
            return await self.inner.structured(
                task=task,
                instructions=instructions,
                schema=schema,
                documents=documents,
                context=context,
                max_tokens=max_tokens,
            )


async def _run_llm(
    cases: Sequence[EvalCase],
    conditions: Sequence[Condition],
    llm: LLMClient,
    concurrency: int,
    progress: _Progress,
) -> list[CaseResult]:
    # Two separate gates (one shared gate would deadlock at concurrency=1): the case gate keeps
    # per-case latencies free of queueing behind the whole corpus, the request gate is the bound.
    case_gate = asyncio.Semaphore(concurrency)
    client = _GatedLLM(llm, asyncio.Semaphore(concurrency))
    graph_arms = [c for c in conditions if c is not Condition.C_LLM_DIRECT]

    async def direct(case: EvalCase) -> list[CaseResult]:
        async with case_gate:
            done = [await arun_llm_direct(case, client)]
        progress.advance(len(done))
        return done

    async def graphs(case: EvalCase) -> list[CaseResult]:
        async with case_gate:
            done = await arun_llm_graphs(case, client, graph_arms)
        progress.advance(len(done))
        return done

    tasks = []
    for case in cases:
        if Condition.C_LLM_DIRECT in conditions:
            tasks.append(direct(case))
        if graph_arms:
            tasks.append(graphs(case))
    batches = await asyncio.gather(*tasks)
    return [r for batch in batches for r in batch]


def run_eval(
    cases: Sequence[EvalCase],
    conditions: Iterable[Condition | str],
    *,
    llm: LLMClient | None = None,
    workers: int | None = None,
    concurrency: int = DEFAULT_LLM_CONCURRENCY,
    progress: Callable[[int, int], None] | None = None,
) -> EvalResults:
    """Run ``conditions`` on ``cases``; results come back in (case, condition) order.

    ``workers`` (default :func:`default_workers`) sizes the process pool of the offline arms
    (``1`` runs them in this process). ``concurrency`` bounds simultaneous LLM requests (every
    direct call and every chunk extraction counts). LLM arms raise ``ValueError`` unless ``llm``
    is given explicitly — run :func:`estimate_llm_cost` first. Must be called outside a running
    event loop, and with more than one worker from an importable main module (``python -m …`` or
    a script guarded by ``if __name__ == "__main__"``): spawned workers re-import it.
    """
    chosen = parse_conditions(conditions)
    case_list = list(cases)
    counts = Counter(c.case_id for c in case_list)
    if dupes := sorted(i for i, n in counts.items() if n > 1):
        raise ValueError(f"사례 id가 중복됩니다: {dupes[:5]}")
    llm_arms = [c for c in chosen if c in LLM_CONDITIONS]
    if llm_arms and llm is None:
        names = ", ".join(c.value for c in llm_arms)
        raise ValueError(
            f"LLM 조건({names})은 llm 인자를 명시해야 실행됩니다. 유료 호출 전에 estimate_llm_cost로 "
            "호출 수와 토큰을 먼저 확인하세요."
        )
    if concurrency < 1:
        raise ValueError("concurrency는 1 이상이어야 합니다")
    offline = [c for c in chosen if c in OFFLINE_CONDITIONS]
    tracker = _Progress(progress, total=len(case_list) * len(chosen))
    pool = default_workers() if workers is None else workers
    results, used = _run_offline(case_list, offline, pool, tracker)
    if llm_arms and llm is not None:
        results += run_sync(_run_llm(case_list, llm_arms, llm, concurrency, tracker))
    position = {c.case_id: i for i, c in enumerate(case_list)}
    rank = {c: i for i, c in enumerate(chosen)}
    results.sort(key=lambda r: (position[r.case_id], rank[r.condition]))
    return EvalResults(cases=tuple(case_list), conditions=chosen, results=tuple(results), workers=used)


# --------------------------------------------------------------------------- reproducibility


class Reproducibility(BaseModel):
    """Outcome of re-running one arm on a deterministic sample (target: 100% identical)."""

    model_config = ConfigDict(frozen=True)

    condition: Condition
    fraction: float
    sample_size: int
    identical: int
    mismatched: tuple[str, ...]
    #: the re-run happened in a freshly spawned interpreter (own hash seed, cold caches)
    fresh_process: bool = True

    @computed_field  # type: ignore[prop-decorator]
    @property
    def rate(self) -> float:
        return round(self.identical / self.sample_size, 4) if self.sample_size else 0.0


def reproducibility_sample(
    cases: Sequence[EvalCase], fraction: float = REPRODUCIBILITY_FRACTION
) -> list[EvalCase]:
    """``ceil(fraction × n)`` cases chosen by SHA-256 of their id, returned in corpus order.

    Hash order is a fixed pseudo-random draw that spreads the sample over orgs, operators and
    styles, yet anyone can recompute it from the case ids alone.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"표본 비율은 0 초과 1 이하여야 합니다: {fraction}")
    if not cases:
        return []
    size = math.ceil(len(cases) * fraction)
    chosen = set(
        sorted((c.case_id for c in cases), key=lambda i: hashlib.sha256(i.encode()).hexdigest())[:size]
    )
    return [c for c in cases if c.case_id in chosen]


#: Environment variable that fixes the string-hash seed of a new interpreter.
_HASH_SEED_VAR = "PYTHONHASHSEED"


@contextlib.contextmanager
def _random_hash_seed_for_children() -> Iterator[None]:
    """Let processes started inside the block draw their own string-hash seed.

    A spawned child inherits the environment, so a ``PYTHONHASHSEED`` pinned for the first run
    (CI often pins it) would give the re-run the very same seed and hide set-order dependence.
    ``random`` is the documented value for "draw one"; the parent's setting is restored after.
    """
    previous = os.environ.get(_HASH_SEED_VAR)
    os.environ[_HASH_SEED_VAR] = "random"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_HASH_SEED_VAR, None)
        else:
            os.environ[_HASH_SEED_VAR] = previous


def _rerun(sample: Sequence[EvalCase], condition: Condition, *, fresh_process: bool) -> list[CaseResult]:
    """Run ``condition`` on ``sample`` again, by default in a newly *spawned* interpreter.

    Spawn, not the platform default: a forked child inherits the parent's string-hash seed and
    warm caches, so it could not reveal output that depends on set/dict iteration order — the
    very nondeterminism the check exists to catch (P6). For the same reason the child never
    inherits a pinned ``PYTHONHASHSEED`` (see :func:`_random_hash_seed_for_children`).
    """
    if not fresh_process or not sample:
        return [run_offline_condition(case, condition) for case in sample]
    spawn = multiprocessing.get_context("spawn")
    with (
        _random_hash_seed_for_children(),
        ProcessPoolExecutor(max_workers=1, mp_context=spawn, initializer=prepare_worker) as pool,
    ):
        return pool.submit(_offline_batch, (tuple(sample), (condition,))).result()


def check_reproducibility(
    results: EvalResults,
    *,
    condition: Condition = Condition.B_RULE_GRAPH,
    fraction: float = REPRODUCIBILITY_FRACTION,
    fresh_process: bool = True,
) -> Reproducibility:
    """Re-run ``condition`` on the sample and compare with ``results``.

    Identical means the same graph hash (canonical JSON of the whole graph), the same fired set
    and the same facts. The re-run happens in a freshly spawned process unless ``fresh_process``
    is false (then in this one — only a self-consistency check, e.g. for quick tests or an
    interactive session, which a spawned process cannot re-import). Only offline arms qualify: a
    live model is not expected to answer twice the same.
    """
    if condition not in OFFLINE_CONDITIONS:
        raise ValueError(f"재현성은 LLM 없는 조건에서만 측정합니다: {condition}")
    if condition not in results.conditions:
        raise ValueError(f"결과에 없는 조건입니다: {condition}")
    sample = reproducibility_sample(results.cases, fraction)
    rerun = {r.case_id: r for r in _rerun(sample, condition, fresh_process=fresh_process)}
    mismatched: list[str] = []
    for case in sample:
        first = results.get(case.case_id, condition)
        again = rerun[case.case_id]
        same = (
            first.error is None
            and again.error is None
            and first.graph_hash == again.graph_hash
            and first.fired == again.fired
            and first.facts == again.facts
        )
        if not same:
            mismatched.append(case.case_id)
    return Reproducibility(
        condition=condition,
        fraction=fraction,
        sample_size=len(sample),
        identical=len(sample) - len(mismatched),
        mismatched=tuple(mismatched),
        fresh_process=fresh_process,
    )


# --------------------------------------------------------------------------- cost estimate


class ConditionCost(BaseModel):
    model_config = ConfigDict(frozen=True)

    condition: Condition
    calls: int
    input_tokens: int
    output_tokens: int


class CostEstimate(BaseModel):
    """Calls and tokens an LLM run would use, from the inputs alone (no request is made).

    ``total_*`` counts what is actually sent: D and DR share one extraction per case, so running
    both costs the extraction once.
    """

    model_config = ConfigDict(frozen=True)

    model: str
    n_cases: int
    per_condition: tuple[ConditionCost, ...]
    total_calls: int
    total_input_tokens: int
    total_output_tokens: int

    def render(self) -> str:
        """Korean summary printed before a paid run."""
        lines = [
            f"LLM 비용 추정 — 모델 {self.model}, 사례 {self.n_cases:,}건 "
            f"(토큰 = 글자 수 / {CHARS_PER_TOKEN}, 실제 과금 토큰과 다를 수 있음)",
        ]
        lines += [
            f"- {c.condition.value}: 호출 {c.calls:,}회, 입력 약 {c.input_tokens:,} 토큰, "
            f"출력 약 {c.output_tokens:,} 토큰"
            for c in self.per_condition
        ]
        lines.append(
            f"- 합계(D·DR 추출 공유 반영): 호출 {self.total_calls:,}회, "
            f"입력 약 {self.total_input_tokens:,} 토큰, 출력 약 {self.total_output_tokens:,} 토큰"
        )
        return "\n".join(lines)


@dataclass
class _Tally:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens


def _tokens(chars: int) -> int:
    return math.ceil(chars / CHARS_PER_TOKEN)


def _schema_chars(schema: type[BaseModel]) -> int:
    return len(json.dumps(to_strict_json_schema(schema), ensure_ascii=False))


def estimate_llm_cost(
    cases: Sequence[EvalCase],
    conditions: Iterable[Condition | str],
    *,
    model: str,
    echo: Callable[[str], object] | None = print,
) -> CostEstimate:
    """Estimate (and ``echo`` in Korean) the calls and tokens of the LLM arms in ``conditions``.

    The estimate replays the exact request construction offline — the same blanked text, the
    same chunking, the same trusted context, instructions and schema — and converts characters
    to tokens with :data:`CHARS_PER_TOKEN`. Output tokens are heuristics
    (:data:`DIRECT_OUTPUT_TOKENS`, :data:`EXTRACTION_OUTPUT_RATIO`). Offline arms cost nothing
    and are ignored.
    """
    chosen = [c for c in parse_conditions(conditions) if c in LLM_CONDITIONS]
    direct_fixed = len(DIRECT_INSTRUCTIONS) + _schema_chars(DirectDefectPayload)
    extract_fixed = len(EXTRACTION_INSTRUCTIONS) + _schema_chars(LlmGraphPayload)
    direct = _Tally()
    extract = _Tally()
    for case in cases:
        inputs = case_inputs(case)
        visible = blank_quarantined(inputs.document.text, inputs.document.quarantined)
        direct.add(_tokens(direct_fixed + len(visible) + len(direct_context(inputs))), DIRECT_OUTPUT_TOKENS)
        context = len(build_extraction_context(inputs.roster, inputs.catalog))
        for chunk in chunk_document(inputs.document):
            output = math.ceil(_tokens(len(chunk.text)) * EXTRACTION_OUTPUT_RATIO)
            extract.add(_tokens(extract_fixed + len(chunk.text) + context), output)
    tally = {c: direct if c is Condition.C_LLM_DIRECT else extract for c in chosen}
    sent = list({id(t): t for t in tally.values()}.values())  # D and DR share one extraction
    estimate = CostEstimate(
        model=model,
        n_cases=len(cases),
        per_condition=tuple(
            ConditionCost(
                condition=c, calls=t.calls, input_tokens=t.input_tokens, output_tokens=t.output_tokens
            )
            for c, t in tally.items()
        ),
        total_calls=sum(t.calls for t in sent),
        total_input_tokens=sum(t.input_tokens for t in sent),
        total_output_tokens=sum(t.output_tokens for t in sent),
    )
    if echo is not None:
        echo(estimate.render())
    return estimate
