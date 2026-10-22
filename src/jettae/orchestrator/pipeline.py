"""One diagnosis as a LangGraph ``StateGraph`` (진단 파이프라인).

::

    guard → extract ─┬─────────────→ assemble → verify ─┬──────────→ check → score → scenarios
                     └→ fallback ──┘                    └→ rebase ─┘
          → rehearse → explore → draft → finalize

Why a state graph rather than a plain function: every stage is a named node with its own timing,
and the two places where the run can take another road are explicit conditional edges instead of
``try``/``except`` buried in a long function —

* **fallback** — the LLM extraction raised :class:`~jettae.llm.base.LLMDisabledError` (no key, a
  cassette miss) or any other :class:`~jettae.llm.base.LLMError` (rate limit, outage). The run
  continues on the Korean rule parser alone and records a Korean note (P4: works without the LLM).
  An error that escapes the extractor means the model is unavailable, not that one answer was bad,
  so the rest of that run stays off the model too (slot-only back-translation, template and human
  drills, template drafts) instead of failing again once per edge and once per patch — the run
  stays fast and its report is the plain deterministic one, not a patchwork.
  With ``LlmMode.OFF`` no model is asked at all, so OFF never needs this road.
* **rebase** — the caller passed the current head graph. Approved patches are human decisions, so a
  re-extraction must not silently drop them: unchanged inputs reuse the head as is, changed inputs
  get the approved decisions carried over to the new extraction op by op — judged by the fact they
  are about, not by the spans it was read from, so editing a runbook line does not undo a fix —
  and every op that meets a newer fact or no longer fits is skipped with a note (P7: the graph
  changes only through approved patches; see :func:`rebase_graph`).

Extractor policy: ``LlmMode.OFF`` → rule parser only. ``LIVE``/``REPLAY`` → *hybrid*: rule and LLM
extractions both go into one :func:`~jettae.extract.assemble`, which grounds every quote the same
way (P2). That is the product default — the rules keep the recall of the patterns they know, the
model adds the free-form sentences they miss, and nothing reaches the graph without a span.

Judgement never sees the model (P1): ``check``, ``score`` and ``rehearse`` run the fixed rules and
the deterministic replay; the LLM only ever extracts, re-phrases, proposes scenarios and risk
hypotheses, and words drafts. ``explore`` (규칙 밖 위험 탐색) runs after the rehearsal: an
exhaustive sweep and optional AI proposals, every one judged by the same replay — its hypotheses
go to their own report field and never touch the findings, the score or the gate. Every LLM
call goes through one :class:`~jettae.llm.base.MeteredLLM`, so the report can publish calls and
tokens per task, and ``judgement_path_calls`` is 0 by construction (the
judgement modules cannot even import an LLM client — import-linter contract).

Determinism (P6): the same documents, structured inputs, options and LLM answers give the same
graph hash, findings, scores, scenarios, rehearsal hashes and patch ids. Only ``timings`` (wall
clock) differ between runs.

The re-judgement after an approval (no documents, no LLM) reuses the same stage functions through
:func:`evaluate_graph`, so a score after "승인" is computed exactly like a score after "진단".
"""

from __future__ import annotations

import asyncio
import re
import time
import unicodedata
from collections.abc import Awaitable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Final, Literal, Protocol, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from jettae.check.context import CheckContext
from jettae.check.registry import run_checks
from jettae.check.scoring import score
from jettae.domain import (
    AddEdge,
    AddNode,
    AiProposalStatus,
    CatalogService,
    Document,
    Edge,
    Extraction,
    Finding,
    InputBundle,
    LlmMode,
    LlmUsageSummary,
    Node,
    OnCallShift,
    OpsGraph,
    Patch,
    PatchAuthor,
    PatchOp,
    PatchStatus,
    Policy,
    ReadinessReport,
    RehearsalResult,
    RemoveEdge,
    RemoveNode,
    RiskHypothesis,
    RosterEntry,
    Scenario,
    SecurityNotice,
    SetAttr,
    SimConfig,
    Verification,
)
from jettae.explore import (
    AiProposals,
    Exploration,
    carry_over,
    explore_graph,
    finish,
    propose_hypotheses,
    sweep,
)
from jettae.extract import LlmExtractor, RuleExtractor, assemble, verify_backtranslation
from jettae.improve import LlmDrafter, PatchError, TemplateDrafter, apply_ops, validate_patch
from jettae.ingest import scan_injection
from jettae.llm.base import (
    LLMClient,
    LLMDisabledError,
    LLMError,
    LLMOutputError,
    LLMRefusalError,
    MeteredLLM,
    UsageMeter,
)
from jettae.orchestrator.state import DiagnosisOptions, DiagnosisOutcome, DiagnosisState
from jettae.rehearse import llm_scenarios, replay_all, scenarios_from_yaml, template_scenarios

__all__ = [
    "CATALOG_GAP_RULE",
    "NODE_NAMES",
    "Evaluation",
    "PipelineContext",
    "Rebase",
    "assess",
    "build_context",
    "build_diagnosis_graph",
    "catalog_gaps",
    "describe_llm_error",
    "diagnosis_graph",
    "draft_version",
    "evaluate_graph",
    "gather_scenarios",
    "held_element_ids",
    "missing_references",
    "rebase_graph",
    "rehearse_all",
    "run_diagnosis",
    "usage_summary",
    "validate_drafts",
]

#: Every node of the diagnosis graph, in execution order (``fallback``/``rebase`` are conditional).
NODE_NAMES: Final = (
    "guard",
    "extract",
    "fallback",
    "assemble",
    "verify",
    "rebase",
    "check",
    "score",
    "scenarios",
    "rehearse",
    "explore",
    "draft",
    "finalize",
)

#: The fixed rule whose findings are the report's "catalog gaps" (서비스 목록엔 있지만 담당·런북 없음).
CATALOG_GAP_RULE: Final = "JT-CAT-001"

_NO_DOCUMENTS: Final = "진단할 런북 문서가 없습니다. 런북을 먼저 올려 주세요."
_DETAIL_LIMIT: Final = 160
#: ``apply_ops`` prefixes every refusal with the op's position in the batch it was given.
_OP_POSITION: Final = re.compile(r"^\d+번째 연산\([a-z_]+\): ")
#: Characters an adapter message may carry that no store can keep or no reader can see
#: (controls, format characters, lone surrogates) — dropped from notes, which go to the audit log.
_UNPRINTABLE: Final = frozenset({"Cc", "Cf", "Cs"})
_LLM_ERROR_KO: Final[tuple[tuple[type[LLMError], str], ...]] = (
    (LLMDisabledError, "LLM을 쓸 수 없는 상태입니다"),
    (LLMRefusalError, "LLM이 응답을 거절했습니다"),
    (LLMOutputError, "LLM 응답이 정해진 형식과 맞지 않습니다"),
)


@dataclass(frozen=True)
class PipelineContext:
    """Run-scoped dependencies handed to every node through LangGraph's ``Runtime.context``.

    Kept out of the state on purpose: the state holds *data* about the run, this holds the *means*
    (the metered client, the meter) — and the compiled graph stays one stateless, shareable object.
    ``llm`` is ``None`` when the mode is OFF, so no node can call a model by accident.
    """

    mode: LlmMode
    llm: LLMClient | None
    meter: UsageMeter
    options: DiagnosisOptions


class SyncNode(Protocol):
    """A synchronous pipeline node: state in, the keys it produced out (LangGraph injects ``runtime``)."""

    def __call__(self, state: DiagnosisState, *, runtime: Runtime[PipelineContext]) -> DiagnosisState: ...


class AsyncNode(Protocol):
    """An asynchronous pipeline node (the stages that may await an LLM)."""

    def __call__(
        self, state: DiagnosisState, *, runtime: Runtime[PipelineContext]
    ) -> Awaitable[DiagnosisState]: ...


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)


def _timed(name: str, node: SyncNode) -> SyncNode:
    """Wrap a synchronous node so its wall-clock time lands in ``timings[name]``.

    Synchronous nodes run in LangGraph's worker threads under ``ainvoke``, which keeps CPU-bound
    stages (rule checks, replay) from blocking an API server's event loop.
    """

    def run(state: DiagnosisState, *, runtime: Runtime[PipelineContext]) -> DiagnosisState:
        start = time.perf_counter()
        update = node(state, runtime=runtime)
        update["timings"] = {name: _elapsed_ms(start)}
        return update

    return run


def _timed_async(name: str, node: AsyncNode) -> AsyncNode:
    """Async counterpart of :func:`_timed` for the stages that may await an LLM."""

    async def run(state: DiagnosisState, *, runtime: Runtime[PipelineContext]) -> DiagnosisState:
        start = time.perf_counter()
        update = await node(state, runtime=runtime)
        update["timings"] = {name: _elapsed_ms(start)}
        return update

    return run


# --------------------------------------------------------------------------- reusable stages


def describe_llm_error(exc: LLMError) -> str:
    """One Korean line for a note: what kind of LLM failure, plus the (clipped) adapter message.

    The adapter message is outside text (a provider's error body, a model's malformed answer),
    and notes are written to the audit log: a NUL or a lone surrogate in it would make the
    store refuse the run's audit record *after* its version and report were saved — leaving a
    version without the lineage the next re-diagnosis needs. Such characters are dropped.
    """
    label = next((text for kind, text in _LLM_ERROR_KO if isinstance(exc, kind)), "LLM 호출이 실패했습니다")
    printable = "".join(ch for ch in str(exc) if ch.isspace() or unicodedata.category(ch) not in _UNPRINTABLE)
    detail = " ".join(printable.split())
    if len(detail) > _DETAIL_LIMIT:
        detail = detail[: _DETAIL_LIMIT - 1] + "…"
    return f"{label}({detail})" if detail else label


def build_context(
    graph: OpsGraph,
    *,
    roster: Iterable[RosterEntry] = (),
    oncall: Iterable[OnCallShift] = (),
    catalog: Iterable[CatalogService] = (),
    policy: Policy | None = None,
    now: datetime | None = None,
) -> CheckContext:
    """The one :class:`CheckContext` every judge of a run shares (rules, score, replay, validator)."""
    return CheckContext(
        graph,
        policy if policy is not None else Policy(),
        tuple(roster),
        tuple(oncall),
        tuple(catalog),
        now,
    )


def assess(ctx: CheckContext) -> tuple[tuple[Finding, ...], ReadinessReport]:
    """Fixed rules, then the published score over their findings (no LLM — P1)."""
    findings = tuple(run_checks(ctx))
    return findings, score(ctx, findings)


def missing_references(graph: OpsGraph, scenario: Scenario) -> list[str]:
    """Node ids a scenario names that the graph does not have (empty = still replayable)."""
    ids = [scenario.service_id, *(p.target_id for p in scenario.perturbations)]
    return [nid for nid in dict.fromkeys(ids) if nid not in graph.nodes]


def gather_scenarios(
    ctx: CheckContext,
    *,
    carried: Iterable[Scenario] = (),
    sources: Iterable[str | bytes] = (),
) -> tuple[tuple[Scenario, ...], tuple[str, ...]]:
    """Template drills for this graph, freshly resolved human drills, then carried-over drills.

    Templates are regenerated for every graph (they aim at whoever the replay would page *now*).
    Human ``scenarios.yaml`` sources are resolved onto this graph; a file that names something the
    graph does not have raises ``ValueError`` — a drill that perturbs nothing would pass for the
    wrong reason. Carried drills (resolved on an earlier graph) are kept while every node they
    name still exists; the others are left out with a note. The first occurrence of an id wins.
    """
    picked: dict[str, Scenario] = {}
    notes: list[str] = []
    for scenario in template_scenarios(ctx):
        picked.setdefault(scenario.id, scenario)
    for source in sources:
        for scenario in scenarios_from_yaml(source, ctx):
            picked.setdefault(scenario.id, scenario)
    for scenario in carried:
        if scenario.id in picked:
            continue
        missing = missing_references(ctx.graph, scenario)
        if missing:
            notes.append(
                f"시나리오 「{scenario.title}」: 가리키는 대상({', '.join(missing)})이 새 그래프에 없어 "
                "이번 리허설에서 뺐습니다."
            )
            continue
        picked[scenario.id] = scenario
    return tuple(picked.values()), tuple(notes)


def rehearse_all(
    graph: OpsGraph, scenarios: Iterable[Scenario], *, ctx: CheckContext, config: SimConfig
) -> tuple[RehearsalResult, ...]:
    """Deterministic replay of every scenario on ``graph`` (the judge of rehearsals — no LLM)."""
    return tuple(replay_all(graph, scenarios, policy=ctx.policy, config=config, roster=ctx.roster))


def validate_drafts(
    ctx: CheckContext,
    drafts: Iterable[Patch],
    *,
    scenarios: Sequence[Scenario],
    findings: Sequence[Finding],
    results: Sequence[RehearsalResult],
    config: SimConfig,
) -> tuple[Patch, ...]:
    """Attach :func:`~jettae.improve.validate_patch` verdicts and record the resulting status.

    VALIDATED patches can be approved; patches with ops that fail become REJECTED_BY_VALIDATOR.
    Review memos (no ops — a human must decide, e.g. which detection signal to add) stay DRAFT
    with the validator's note attached: they are not wrong, just not something a click can apply.
    """
    out: list[Patch] = []
    for patch in drafts:
        verdict = validate_patch(
            ctx,
            patch,
            scenarios=scenarios,
            config=config,
            baseline_findings=findings,
            baseline_results=results,
        )
        if verdict.passed:
            status = PatchStatus.VALIDATED
        elif patch.ops:
            status = PatchStatus.REJECTED_BY_VALIDATOR
        else:
            status = PatchStatus.DRAFT
        out.append(patch.model_copy(update={"status": status, "validation": verdict}))
    return tuple(out)


def held_element_ids(graph: OpsGraph) -> tuple[str, ...]:
    """Every node and edge on HOLD (판단 보류), sorted — unknown is never shown as fine (P3)."""
    held = [n.id for n in graph.nodes.values() if n.verification is Verification.HOLD]
    held += [e.id for e in graph.edges.values() if e.verification is Verification.HOLD]
    return tuple(sorted(held))


def catalog_gaps(findings: Iterable[Finding]) -> tuple[str, ...]:
    """Services the catalog lists but no owner or runbook covers (from :data:`CATALOG_GAP_RULE`)."""
    return tuple(
        sorted({f.service_id for f in findings if f.rule_id == CATALOG_GAP_RULE and f.service_id is not None})
    )


def draft_version(graph: OpsGraph, base_graph: OpsGraph | None, base_version: int | None) -> int:
    """The version number ``graph`` will have once committed — what drafts must be based on.

    Committing a graph equal to the head appends nothing (the head stays), any other graph becomes
    the next version, and the very first graph of a workspace is version 1.
    """
    if base_graph is None or base_version is None:
        return 1
    return base_version if graph.content_hash() == base_graph.content_hash() else base_version + 1


def usage_summary(mode: LlmMode, meter: UsageMeter, llm: LLMClient | None) -> LlmUsageSummary:
    """Publishable LLM usage of one run. The judgement path is LLM-free, hence always 0 calls."""
    if llm is None:
        return LlmUsageSummary(mode=mode)
    models = sorted({record.model for record in meter.records})
    return LlmUsageSummary(
        mode=mode,
        provider=llm.provider,
        model=models[0] if len(models) == 1 else llm.model,
        calls=meter.calls,
        input_tokens=meter.input_tokens,
        output_tokens=meter.output_tokens,
        cache_read_tokens=meter.cache_read_tokens,
        by_task=meter.by_task(),
        judgement_path_calls=0,
    )


@dataclass(frozen=True)
class Rebase:
    """Result of carrying approved patches over to a new extraction (see :func:`rebase_graph`)."""

    graph: OpsGraph
    reused: bool
    reapplied: tuple[str, ...]
    notes: tuple[str, ...]


def _same_fact(current: Node | Edge, recorded: Node | Edge) -> bool:
    """Do two versions of an element state the same fact, wherever the text said it?

    Identity, endpoints (or name) and attribute *values* make the fact; provenance does not —
    editing any line of a runbook gives the document a new id and moves offsets, so every fact
    read from it comes back with different spans although nothing about it changed.
    """
    values_now = {key: attr.value for key, attr in current.attrs.items()}
    values_then = {key: attr.value for key, attr in recorded.attrs.items()}
    if values_now != values_then or current.kind != recorded.kind:
        return False
    if isinstance(current, Edge) and isinstance(recorded, Edge):
        return (current.src, current.dst) == (recorded.src, recorded.dst)
    if isinstance(current, Node) and isinstance(recorded, Node):
        return current.name == recorded.name
    return False


def _held_copy(current: Node | Edge, approved: Node | Edge) -> bool:
    """Is ``current`` a held (판단 보류) copy of a fact the approval stated as verified?

    A held element counts as unknown for every judge (P3), so the re-extraction "already saying"
    the approved fact only on HOLD does not keep the decision — the gap it closed would come back
    as unknown, without a word in the notes.
    """
    return current.verification is Verification.HOLD and approved.verification is not Verification.HOLD


def _filled_as_applied(patch: Patch) -> bool:
    """Were ``patch``'s ops recorded *as applied* (``previous``/``snapshot`` filled by ``apply_ops``)?

    Every drafted patch (template or AI) is assembled from :func:`~jettae.improve.apply_ops`'
    output, so a :class:`SetAttr` with ``previous=None`` there means "the attribute was absent
    when the patch was approved". A hand-written op may simply leave ``previous`` unfilled, and
    then ``None`` says nothing about the graph it was approved on.
    """
    return patch.author is not PatchAuthor.HUMAN


def _newer_fact(graph: OpsGraph, op: PatchOp, *, as_applied: bool) -> str | None:
    """Why carrying ``op`` over would override what the new extraction states (``None`` = it would not).

    Asked only for ops whose decision does not already hold (see :func:`_decision_holds`):

    * a :class:`SetAttr` that filled an attribute absent at approval time, while the new inputs
      now state a value of their own — the runbook (or catalog) was written after the decision,
      so it is the newer fact. ``apply_ops`` cannot see this (``previous=None`` proves nothing to
      it), hence the explicit check;
    * an :class:`AddNode` whose node the new extraction has only on HOLD — a node cannot be swapped
      for the approved copy while edges hang on it, so the doubt is reported instead of hidden.
    """
    if isinstance(op, SetAttr) and op.previous is None and as_applied:
        target: Node | Edge | None = graph.nodes.get(op.target_id) or graph.edges.get(op.target_id)
        current = target.attrs.get(op.key) if target is not None else None
        if current is not None:
            return (
                f"승인 당시 비어 있던 '{op.target_id}'의 '{op.key}' 값이 지금은 입력에 "
                f"'{current.value}'(으)로 적혀 있어, 더 새로운 사실을 덮어쓰지 않았습니다."
            )
    if isinstance(op, AddNode):
        node = graph.nodes.get(op.node.id)
        if node is not None and _same_fact(node, op.node) and _held_copy(node, op.node):
            why = node.hold_reason or "사유 미상"
            return f"노드 '{op.node.id}'가 새로 추출한 그래프에서 판단 보류 상태입니다({why})."
    return None


def _attr_value(graph: OpsGraph, target_id: str, key: str) -> tuple[bool, object]:
    """``(target exists, current value or None)`` of an attribute on a node or an edge."""
    target: Node | Edge | None = graph.nodes.get(target_id) or graph.edges.get(target_id)
    if target is None:
        return False, None
    attr = target.attrs.get(key)
    return True, None if attr is None else attr.value


def _decision_holds(graph: OpsGraph, op: PatchOp) -> bool:
    """Does the new extraction already say what an approved op decided?

    The removed element is gone (say the runbook no longer names the departed owner), the added
    one is there as the same fact (the team wrote the approved sentence into the runbook), or the
    attribute already has the approved value. Such a decision is kept without touching the graph,
    so the runbook's own wording (with its spans) wins over the patch's copy of it.
    """
    if isinstance(op, RemoveEdge):
        return op.edge_id not in graph.edges
    if isinstance(op, RemoveNode):
        return op.node_id not in graph.nodes
    if isinstance(op, AddEdge):
        edge = graph.edges.get(op.edge.id)
        return edge is not None and _same_fact(edge, op.edge) and not _held_copy(edge, op.edge)
    if isinstance(op, AddNode):
        node = graph.nodes.get(op.node.id)
        return node is not None and _same_fact(node, op.node) and not _held_copy(node, op.node)
    exists, value = _attr_value(graph, op.target_id, op.key)
    return exists and value == (None if op.value is None else op.value.value)


def _reanchored(graph: OpsGraph, op: PatchOp) -> tuple[PatchOp, ...]:
    """``op`` pointed at the element as the new extraction has it, when it is the same fact.

    Approved removals and attribute changes record the element they were approved against, and
    :func:`~jettae.improve.apply_ops` refuses them when that record differs from the graph. After
    a re-extraction the record usually differs only in provenance (see :func:`_same_fact`), and a
    strict refusal would bring back, say, a departed owner a team lead removed. A record whose
    values differ is a newer fact: the op is returned unchanged, so ``apply_ops`` refuses it.

    An approved edge the new extraction has only on HOLD (see :func:`_held_copy`) is replaced by
    the approved copy: the runbook states the very same fact and a lead confirmed it, so the
    machine's doubt about one sentence must not re-open the gap the approval closed.
    """
    if isinstance(op, RemoveEdge) and op.snapshot is not None:
        edge = graph.edges.get(op.edge_id)
        if edge is not None and _same_fact(edge, op.snapshot):
            return (op.model_copy(update={"snapshot": edge}),)
    elif isinstance(op, RemoveNode) and op.snapshot is not None:
        node = graph.nodes.get(op.node_id)
        if node is not None and _same_fact(node, op.snapshot):
            return (op.model_copy(update={"snapshot": node}),)
    elif isinstance(op, SetAttr) and op.previous is not None:
        target: Node | Edge | None = graph.nodes.get(op.target_id) or graph.edges.get(op.target_id)
        current = target.attrs.get(op.key) if target is not None else None
        if current is not None and current.value == op.previous.value:
            return (op.model_copy(update={"previous": current}),)
    elif isinstance(op, AddEdge):
        held = graph.edges.get(op.edge.id)
        if held is not None and _same_fact(held, op.edge) and _held_copy(held, op.edge):
            return (RemoveEdge(edge_id=held.id, snapshot=held), op)
    return (op,)


def _patch_error_reason(exc: PatchError) -> str:
    """The reason of a :class:`PatchError` without ``apply_ops``' "N번째 연산(op): " position prefix
    (the note names the op's position in the approved patch instead)."""
    return _OP_POSITION.sub("", str(exc), count=1)


def rebase_graph(
    extracted: OpsGraph,
    base: OpsGraph,
    *,
    base_extraction_hash: str | None = None,
    approved: Sequence[Patch] = (),
) -> Rebase:
    """Keep approved human decisions when the graph is extracted again.

    * The new extraction equals the one the head was built from → the head is reused untouched
      (it already *is* that extraction plus the approved patches).
    * Otherwise each approved patch's decisions are carried over in approval order, one op at a
      time, judged by *what* they decided rather than by *where* the runbook said it: a decision
      the new extraction already reflects is kept as is; an op whose target is the same fact with
      new provenance is re-pointed at it and applied (editing one line of a runbook must not bring
      back the departed owner a team lead removed); an op whose target now carries different
      values — a newer fact, including a value the inputs now state for an attribute the patch
      filled while it was empty — or that no longer fits the graph is skipped with a Korean note
      (:func:`_newer_fact`, or :func:`~jettae.improve.apply_ops` refusing it). An old decision
      never silently overwrites a newer fact.
    * When the result equals the head (nothing the patches touched changed) the head is reused.

    ``reapplied`` lists the patches with at least one decision kept.
    """
    if base_extraction_hash is not None and extracted.content_hash() == base_extraction_hash:
        return Rebase(graph=base, reused=True, reapplied=tuple(p.id for p in approved), notes=())
    graph = extracted
    reapplied: list[str] = []
    notes: list[str] = []
    for patch in approved:
        kept = 0
        total = len(patch.ops)
        as_applied = _filled_as_applied(patch)
        for index, op in enumerate(patch.ops, start=1):
            if _decision_holds(graph, op):
                kept += 1
                continue
            reason = _newer_fact(graph, op, as_applied=as_applied)
            if reason is None:
                try:
                    graph, _applied = apply_ops(graph, _reanchored(graph, op))
                except PatchError as exc:
                    reason = _patch_error_reason(exc)
            if reason is not None:
                notes.append(
                    f"승인된 수정안 {patch.id}의 연산 {index}/{total}({op.op})은 새로 추출한 그래프에 "
                    f"맞지 않아 건너뛰었습니다: {reason}"
                )
                continue
            kept += 1
        if kept:
            reapplied.append(patch.id)
    reused = graph.content_hash() == base.content_hash()
    if not reused:
        summary = "입력(런북·명부·카탈로그 등)이 바뀌어 그래프를 다시 만들었습니다."
        if approved:
            summary += (
                f" 승인된 수정안 {len(approved)}건 중 {len(reapplied)}건의 결정을 새 그래프에 반영했습니다."
            )
        notes.insert(0, summary)
    return Rebase(
        graph=base if reused else graph, reused=reused, reapplied=tuple(reapplied), notes=tuple(notes)
    )


@dataclass(frozen=True)
class Evaluation:
    """Judgement of one graph without extraction or LLM (see :func:`evaluate_graph`)."""

    ctx: CheckContext
    findings: tuple[Finding, ...]
    readiness: ReadinessReport
    scenarios: tuple[Scenario, ...]
    rehearsals: tuple[RehearsalResult, ...]
    patches: tuple[Patch, ...]
    held_ids: tuple[str, ...]
    catalog_gaps: tuple[str, ...]
    notes: tuple[str, ...]
    timings: dict[str, float]
    #: 규칙 밖 위험 후보 (``None`` when exploration did not run)
    exploration: Exploration | None = None


def evaluate_graph(
    graph: OpsGraph,
    *,
    roster: Iterable[RosterEntry] = (),
    oncall: Iterable[OnCallShift] = (),
    catalog: Iterable[CatalogService] = (),
    policy: Policy | None = None,
    workspace_id: str,
    version: int,
    carried: Iterable[Scenario] = (),
    config: SimConfig | None = None,
    now: datetime | None = None,
    rehearse: bool = True,
    draft: bool = True,
    explore: bool = True,
    previous_hypotheses: Iterable[RiskHypothesis] = (),
) -> Evaluation:
    """Re-judge a committed graph: check → score → scenarios → rehearse → explore → template drafts.

    Used after an approval or a revert, when the graph changed through a patch rather than through
    new documents: nothing needs extracting and no LLM is involved (P4), yet the verdicts come from
    the very same stage functions as a full diagnosis. Drafts are template drafts against
    ``version`` and arrive validated. Exploration (only with ``rehearse``) sweeps the new graph
    again and re-verifies the AI hypotheses of ``previous_hypotheses`` on it.
    """
    sim = config if config is not None else SimConfig()
    timings: dict[str, float] = {}
    start = time.perf_counter()
    ctx = build_context(graph, roster=roster, oncall=oncall, catalog=catalog, policy=policy, now=now)
    findings = tuple(run_checks(ctx))
    timings["check"] = _elapsed_ms(start)
    start = time.perf_counter()
    readiness = score(ctx, findings)
    timings["score"] = _elapsed_ms(start)
    scenarios: tuple[Scenario, ...] = ()
    results: tuple[RehearsalResult, ...] = ()
    notes: tuple[str, ...] = ()
    if rehearse:
        start = time.perf_counter()
        scenarios, notes = gather_scenarios(ctx, carried=carried)
        timings["scenarios"] = _elapsed_ms(start)
        start = time.perf_counter()
        results = rehearse_all(graph, scenarios, ctx=ctx, config=sim)
        timings["rehearse"] = _elapsed_ms(start)
    exploration: Exploration | None = None
    if rehearse and explore:
        start = time.perf_counter()
        exploration = explore_graph(ctx, findings, config=sim, previous=previous_hypotheses)
        timings["explore"] = _elapsed_ms(start)
    patches: tuple[Patch, ...] = ()
    if draft:
        start = time.perf_counter()
        drafts = TemplateDrafter().draft(ctx, findings, workspace_id=workspace_id, base_version=version)
        patches = validate_drafts(
            ctx, drafts, scenarios=scenarios, findings=findings, results=results, config=sim
        )
        timings["draft"] = _elapsed_ms(start)
    return Evaluation(
        ctx=ctx,
        findings=findings,
        readiness=readiness,
        scenarios=scenarios,
        rehearsals=results,
        patches=patches,
        held_ids=held_element_ids(graph),
        catalog_gaps=catalog_gaps(findings),
        notes=notes,
        timings=timings,
        exploration=exploration,
    )


# --------------------------------------------------------------------------- nodes


def _model(state: DiagnosisState, deps: PipelineContext) -> LLMClient | None:
    """The client the stages after extraction may use: none once the run took the fallback road.

    An error that escaped the extractor already went through the adapter's own retries (or is a
    missing key / cassette miss), so asking again for every edge and every patch would only slow
    the run down and mix model and template output in one report (see the module docstring).
    """
    return None if state.get("llm_failure") else deps.llm


def _guard(state: DiagnosisState, *, runtime: Runtime[PipelineContext]) -> DiagnosisState:
    """Quarantine prompt injections (P5) and collect the notices for the report.

    Loaders already guard what they load; running :func:`~jettae.ingest.scan_injection` again is
    idempotent, yields the notices the report publishes, and guards documents that reached the
    pipeline some other way. A document listed twice (same content id) is diagnosed once.
    """
    documents: list[Document] = []
    notices: list[SecurityNotice] = []
    seen: set[str] = set()
    for doc in state["bundle"].documents:
        if doc.doc_id in seen:
            continue
        seen.add(doc.doc_id)
        guarded, found = scan_injection(doc)
        documents.append(guarded)
        notices.extend(found)
    if not documents:
        raise ValueError(_NO_DOCUMENTS)
    return {"documents": tuple(documents), "security": tuple(notices)}


def _parse_rules(documents: Sequence[Document], bundle: InputBundle) -> tuple[Extraction, ...]:
    """The Korean rule parser over every document, in document order (no LLM)."""
    parser = RuleExtractor()
    return tuple(parser.extract(d, roster=bundle.roster, catalog=bundle.catalog) for d in documents)


async def _extract(state: DiagnosisState, *, runtime: Runtime[PipelineContext]) -> DiagnosisState:
    """Rule parser for every document; with an LLM configured, LLM extraction on top (hybrid).

    Documents are sent one after another (chunks of one document run concurrently inside the
    extractor), so the order of metered calls and the error that stops the LLM path are stable.
    An :class:`LLMError` that escapes the extractor (OFF, missing key, cassette miss, rate limit)
    drops *all* LLM extractions — a graph half from the model and half not would be harder to
    explain than a clean rule-only graph — and routes to ``fallback``.

    The rule parser is CPU-bound and this node is async (it awaits the model), so the parser runs
    in a worker thread like the synchronous nodes do: an API server's event loop keeps serving
    other requests while a large runbook set is parsed.
    """
    deps = runtime.context
    bundle = state["bundle"]
    documents = state["documents"]
    rule = await asyncio.to_thread(_parse_rules, documents, bundle)
    if deps.llm is None:
        return {"rule_extractions": rule, "llm_extractions": (), "llm_failure": None, "extractor": "rule"}
    extractor = LlmExtractor(
        deps.llm,
        max_chunk_chars=deps.options.max_chunk_chars,
        concurrency=deps.options.llm_concurrency,
    )
    found: list[Extraction] = []
    failed = 0
    try:
        for doc in documents:
            run = await extractor.aextract_chunks(doc, roster=bundle.roster, catalog=bundle.catalog)
            found.extend(run.extractions)
            failed += run.failed_chunks
    except LLMError as exc:
        return {
            "rule_extractions": rule,
            "llm_extractions": (),
            "llm_failure": describe_llm_error(exc),
            "extractor": "rule",
        }
    notes: tuple[str, ...] = ()
    if failed:
        notes = (
            f"LLM 추출에서 {failed}개 구간이 응답 오류로 빠져, 그 구간은 규칙 파서 결과만 반영했습니다.",
        )
    return {
        "rule_extractions": rule,
        "llm_extractions": tuple(found),
        "llm_failure": None,
        "extractor": "hybrid",
        "notes": notes,
    }


def _after_extract(state: DiagnosisState) -> Literal["fallback", "assemble"]:
    return "fallback" if state.get("llm_failure") else "assemble"


def _fallback(state: DiagnosisState, *, runtime: Runtime[PipelineContext]) -> DiagnosisState:
    """Continue on the rule parser alone, keep the model out of the rest of the run, say so (P4)."""
    reason = state.get("llm_failure") or "원인 미상"
    note = (
        "LLM 추출이 실패해 규칙 파서 결과만으로 그래프를 만들었습니다. 이번 실행의 나머지 단계"
        f"(역번역 대조·시나리오 제안·수정안 문장)도 LLM 없이 진행했습니다. 원인: {reason}"
    )
    return {"extractor": "rule", "llm_extractions": (), "notes": (note,)}


def _assemble(state: DiagnosisState, *, runtime: Runtime[PipelineContext]) -> DiagnosisState:
    """Ground every quote, resolve people against the roster, import trusted facts (P2)."""
    bundle = state["bundle"]
    extractions = (*state["rule_extractions"], *state.get("llm_extractions", ()))
    result = assemble(
        extractions,
        state["documents"],
        roster=bundle.roster,
        catalog=bundle.catalog,
        oncall=bundle.oncall,
        policy=bundle.policy,
    )
    return {"assembly": result, "notes": result.notes}


async def _verify(state: DiagnosisState, *, runtime: Runtime[PipelineContext]) -> DiagnosisState:
    """Back-translate response edges and hold the ones whose slots disagree with the source.

    With the LLM off the slot comparison alone decides; with an LLM the model may additionally
    flag a non-equivalent sentence (it can only *hold* an edge, never confirm a slot mismatch).
    """
    deps = runtime.context
    result = await verify_backtranslation(
        state["assembly"].graph,
        state["documents"],
        _model(state, deps),
        concurrency=deps.options.llm_concurrency,
    )
    graph = result.graph
    return {
        "verification": result,
        "graph": graph,
        "extraction_hash": graph.content_hash(),
        "graph_reused": False,
        "reapplied_patch_ids": (),
    }


def _after_verify(state: DiagnosisState) -> Literal["rebase", "check"]:
    return "rebase" if state.get("base_graph") is not None else "check"


def _rebase(state: DiagnosisState, *, runtime: Runtime[PipelineContext]) -> DiagnosisState:
    """Carry the head's approved patches over to the new extraction (see :func:`rebase_graph`)."""
    options = runtime.context.options
    # ``_after_verify`` routes here only when the run has a base graph.
    base = cast(OpsGraph, state.get("base_graph"))
    result = rebase_graph(
        state["graph"],
        base,
        base_extraction_hash=options.base_extraction_hash,
        approved=options.approved_patches,
    )
    return {
        "graph": result.graph,
        "graph_reused": result.reused,
        "reapplied_patch_ids": result.reapplied,
        "notes": result.notes,
    }


def _check(state: DiagnosisState, *, runtime: Runtime[PipelineContext]) -> DiagnosisState:
    """Fixed rules over the final graph (judgement path — no LLM)."""
    bundle = state["bundle"]
    ctx = build_context(
        state["graph"],
        roster=bundle.roster,
        oncall=bundle.oncall,
        catalog=bundle.catalog,
        policy=bundle.policy,
        now=runtime.context.options.now,
    )
    return {"ctx": ctx, "findings": tuple(run_checks(ctx))}


def _score(state: DiagnosisState, *, runtime: Runtime[PipelineContext]) -> DiagnosisState:
    """The published readiness score (SCORING_V1) over the findings."""
    return {"readiness": score(state["ctx"], state["findings"])}


async def _scenarios(state: DiagnosisState, *, runtime: Runtime[PipelineContext]) -> DiagnosisState:
    """Template + human (+ LLM-proposed) drills. LLM ideas are data the replay judges, never verdicts."""
    deps = runtime.context
    options = deps.options
    if not options.rehearse:
        return {"scenarios": ()}
    ctx = state["ctx"]
    scenarios, notes = gather_scenarios(
        ctx, carried=options.carried_scenarios, sources=options.scenario_sources
    )
    model = _model(state, deps)
    proposed: list[Scenario] = []
    if model is not None and options.llm_scenario_count > 0:
        try:
            proposed = await llm_scenarios(model, ctx, n=options.llm_scenario_count)
        except LLMError as exc:
            notes = (
                *notes,
                "LLM 시나리오를 만들지 못해 템플릿·사람 작성 시나리오만 리허설했습니다 — "
                + describe_llm_error(exc),
            )
    picked = {s.id: s for s in scenarios}
    for scenario in proposed:
        picked.setdefault(scenario.id, scenario)
    return {"scenarios": tuple(picked.values()), "notes": notes}


def _rehearse(state: DiagnosisState, *, runtime: Runtime[PipelineContext]) -> DiagnosisState:
    """Deterministic replay of every scenario (judgement path — no LLM)."""
    options = runtime.context.options
    if not options.rehearse:
        return {"rehearsals": ()}
    return {
        "rehearsals": rehearse_all(
            state["graph"], state["scenarios"], ctx=state["ctx"], config=options.sim_config
        )
    }


async def _explore(state: DiagnosisState, *, runtime: Runtime[PipelineContext]) -> DiagnosisState:
    """Risks outside the fixed rules: exhaustive sweep, optional AI proposals, replay verdicts.

    The sweep and the verification are CPU work and go to worker threads. An LLM error other
    than "disabled" costs only the AI proposals — the sweep's hypotheses are kept and a note says
    why the AI part is missing. Findings, score and gate are already final when this runs.
    """
    deps = runtime.context
    options = deps.options
    if not (options.rehearse and options.explore):
        return {"risk_hypotheses": (), "exploration": None}
    ctx = state["ctx"]
    findings = state["findings"]
    sim = options.sim_config
    swept = await asyncio.to_thread(sweep, ctx, findings, config=sim)
    model = _model(state, deps)
    proposals: AiProposals | None = None
    notes: tuple[str, ...] = ()
    status: AiProposalStatus | None = None
    if model is None:
        status = AiProposalStatus.FALLBACK if state.get("llm_failure") else AiProposalStatus.OFF
    elif options.llm_hypothesis_count <= 0:
        status = AiProposalStatus.NOT_REQUESTED
    else:
        try:
            proposals = await propose_hypotheses(
                model, ctx, findings, n=options.llm_hypothesis_count, known=swept, config=sim
            )
        except LLMError as exc:
            status = AiProposalStatus.FAILED
            notes = (
                "AI 위험 가설을 만들지 못해 자동 전수 탐색 결과만 실었습니다 — " + describe_llm_error(exc),
            )
    carried, gone = carry_over(ctx, options.carried_hypotheses)
    exploration = await asyncio.to_thread(
        lambda: finish(
            ctx,
            swept,
            proposals=proposals,
            carried=carried,
            carried_gone=gone,
            config=sim,
            notes=notes,
            ai_status=status,
        )
    )
    return {
        "risk_hypotheses": exploration.hypotheses,
        "exploration": exploration.summary,
        "notes": notes,
    }


async def _draft(state: DiagnosisState, *, runtime: Runtime[PipelineContext]) -> DiagnosisState:
    """Template drafts (LLM OFF, or after a fallback) or LLM-worded drafts over the same
    templates, then validation.

    The model may only reword a template sentence and pick among roster candidates; the ops are
    compiled by code, and every draft is re-checked by the fixed rules and the replay before it
    is offered for approval. That validation (and template drafting) is the heaviest CPU work of
    a run, so it goes to a worker thread rather than holding this async node's event loop.
    """
    deps = runtime.context
    options = deps.options
    graph = state["graph"]
    version = draft_version(graph, state.get("base_graph"), options.base_version)
    if not options.draft:
        return {"patches": (), "draft_base_version": version}
    ctx = state["ctx"]
    findings = state["findings"]
    selected: Sequence[Finding] = findings
    notes: tuple[str, ...] = ()
    if options.draft_finding_ids is not None:
        wanted = set(options.draft_finding_ids)
        selected = [f for f in findings if f.id in wanted]
        missing = sorted(wanted - {f.id for f in selected})
        if missing:
            notes = (f"요청한 끊김이 이번 진단 결과에 없어 수정안을 만들지 않았습니다: {', '.join(missing)}",)
    model = _model(state, deps)
    if model is None:
        drafts = await asyncio.to_thread(
            TemplateDrafter().draft, ctx, selected, workspace_id=options.workspace_id, base_version=version
        )
    else:
        drafts = await LlmDrafter(model).adraft(
            ctx, selected, workspace_id=options.workspace_id, base_version=version
        )
    patches = await asyncio.to_thread(
        validate_drafts,
        ctx,
        drafts,
        scenarios=state.get("scenarios", ()),
        findings=findings,
        results=state.get("rehearsals", ()),
        config=options.sim_config,
    )
    return {"patches": patches, "draft_base_version": version, "notes": notes}


def _finalize(state: DiagnosisState, *, runtime: Runtime[PipelineContext]) -> DiagnosisState:
    """Bookkeeping the report publishes: grounding, held elements, catalog gaps, LLM usage."""
    deps = runtime.context
    held_by_backtranslation = len(state["verification"].held_ids)
    grounding = state["assembly"].grounding.model_copy(
        update={"held_backtranslation": held_by_backtranslation}
    )
    return {
        "grounding": grounding,
        "held_ids": held_element_ids(state["graph"]),
        "catalog_gaps": catalog_gaps(state["findings"]),
        "usage": usage_summary(deps.mode, deps.meter, deps.llm),
    }


# --------------------------------------------------------------------------- graph


DiagnosisGraph = CompiledStateGraph[DiagnosisState, PipelineContext, DiagnosisState, DiagnosisState]


def build_diagnosis_graph() -> DiagnosisGraph:
    """Compile the diagnosis ``StateGraph`` (stateless: dependencies arrive per run as context)."""
    graph = StateGraph(DiagnosisState, context_schema=PipelineContext)
    graph.add_node("guard", _timed("guard", _guard))
    graph.add_node("extract", _timed_async("extract", _extract))
    graph.add_node("fallback", _timed("fallback", _fallback))
    graph.add_node("assemble", _timed("assemble", _assemble))
    graph.add_node("verify", _timed_async("verify", _verify))
    graph.add_node("rebase", _timed("rebase", _rebase))
    graph.add_node("check", _timed("check", _check))
    graph.add_node("score", _timed("score", _score))
    graph.add_node("scenarios", _timed_async("scenarios", _scenarios))
    graph.add_node("rehearse", _timed("rehearse", _rehearse))
    graph.add_node("explore", _timed_async("explore", _explore))
    graph.add_node("draft", _timed_async("draft", _draft))
    graph.add_node("finalize", _timed("finalize", _finalize))
    graph.add_edge(START, "guard")
    graph.add_edge("guard", "extract")
    graph.add_conditional_edges("extract", _after_extract, {"fallback": "fallback", "assemble": "assemble"})
    graph.add_edge("fallback", "assemble")
    graph.add_edge("assemble", "verify")
    graph.add_conditional_edges("verify", _after_verify, {"rebase": "rebase", "check": "check"})
    graph.add_edge("rebase", "check")
    graph.add_edge("check", "score")
    graph.add_edge("score", "scenarios")
    graph.add_edge("scenarios", "rehearse")
    graph.add_edge("rehearse", "explore")
    graph.add_edge("explore", "draft")
    graph.add_edge("draft", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(name="jettae-diagnosis")


@lru_cache(maxsize=1)
def diagnosis_graph() -> DiagnosisGraph:
    """The compiled graph, built once per process and shared by every run."""
    return build_diagnosis_graph()


async def run_diagnosis(
    bundle: InputBundle,
    *,
    base_graph: OpsGraph | None,
    llm: LLMClient,
    llm_mode: LlmMode,
    options: DiagnosisOptions,
) -> DiagnosisOutcome:
    """Run one diagnosis end to end and return everything it produced (nothing is persisted).

    ``llm`` is consulted only when ``llm_mode`` is LIVE or REPLAY, always through a fresh
    :class:`~jettae.llm.base.MeteredLLM` so the outcome's usage covers exactly this run.
    ``base_graph`` (the current head) and ``options.base_version`` go together. Raises
    ``ValueError`` (Korean message) for unusable inputs — no documents, inconsistent options, a
    ``scenarios.yaml`` naming something the graph does not have.
    """
    mode = LlmMode(llm_mode)
    if (base_graph is None) != (options.base_version is None):
        raise ValueError("base_graph와 options.base_version은 함께 주거나 함께 비워야 합니다.")
    meter = UsageMeter()
    client: LLMClient | None = None if mode is LlmMode.OFF else MeteredLLM(llm, meter)
    deps = PipelineContext(mode=mode, llm=client, meter=meter, options=options)
    start = time.perf_counter()
    initial: DiagnosisState = {"bundle": bundle, "base_graph": base_graph, "notes": (), "timings": {}}
    final = cast(DiagnosisState, await diagnosis_graph().ainvoke(initial, context=deps))
    timings: dict[str, float] = {
        name: final["timings"][name] for name in NODE_NAMES if name in final["timings"]
    }
    timings["total"] = _elapsed_ms(start)
    return DiagnosisOutcome(
        graph=final["graph"],
        findings=final["findings"],
        readiness=final["readiness"],
        scenarios=final["scenarios"],
        rehearsals=final["rehearsals"],
        patches=final["patches"],
        grounding=final["grounding"],
        security=final["security"],
        held_ids=final["held_ids"],
        catalog_gaps=final["catalog_gaps"],
        timings=timings,
        usage=final["usage"],
        extractor=final["extractor"],
        extraction_hash=final["extraction_hash"],
        graph_reused=final["graph_reused"],
        reapplied_patch_ids=final["reapplied_patch_ids"],
        draft_base_version=final["draft_base_version"],
        notes=final["notes"],
        extractions=(*final["rule_extractions"], *final.get("llm_extractions", ())),
        risk_hypotheses=final.get("risk_hypotheses", ()),
        exploration=final.get("exploration"),
    )
