"""AI 수정 초안 재검사 통과율 · 명부 외 인물 생성 — measured on the sealed set (seed 20261012).

KPI row (application): 결함 탐지(봉인셋) "AI 수정 초안 재검사 통과율·명부 외 인물 생성", targets
"90% 이상 / … / 0건". Plan text: "④ 개선: 끊긴 곳마다 AI가 고칠 문장 초안 → 과거 리허설 재실행 통과 →
팀 리드 승인분만 새 버전".

What one case goes through (product code only, frozen at ccf3b4d; nothing under src/ is changed):

1. The graph, two ways. **B** = the LLM-off product path the evalkit harness uses for condition B
   (:func:`jettae.evalkit.conditions.offline_graph`: guard → RuleExtractor → assemble → slot-only
   back-translation). **O** = the gold facts as a graph (:func:`jettae.evalkit.facts.oracle_graph`),
   the upper bound: every finding there is a real defect, none is an extraction artefact.
2. The judges of a diagnosis, exactly as the pipeline runs them with the LLM off
   (:mod:`jettae.orchestrator.pipeline`): fixed rules → template drills for this graph
   (``gather_scenarios``) → deterministic replay (``rehearse_all``).
3. Drafts: :class:`~jettae.improve.TemplateDrafter` (free, no LLM) — or, with ``--llm --yes``,
   :class:`~jettae.improve.LlmDrafter` (an LLM rewords the sentence and may pick another person
   from the *same* candidate list; the ops are compiled by code either way).
4. :func:`~jettae.orchestrator.pipeline.validate_drafts` → :func:`~jettae.improve.validate_patch`
   per draft: re-check with the fixed rules, replay of the drills that ran on the base graph,
   no-regression rule, nobody outside the roster, provenance.
5. Measured on top of the validator, independently: every person a patch adds or links to is
   looked up in the case's roster CSV parsed here with the ``csv`` module (not the product's
   loader); person names in the patch sentence are checked against the same CSV; the fixed rules
   are re-run on the patched graph to confirm the claimed resolutions.
6. Optional "approve everything" simulation (``--loop``, on by default): approve the first
   VALIDATED draft, re-judge the new version with :func:`~jettae.orchestrator.pipeline.evaluate_graph`
   (what the product does after an approval), repeat until no validated draft is left.

Usage (repository root, product venv)::

    python -X utf8 scripts/eval_patch_recheck.py                       # free: all 1,320 cases, B and O
    python -X utf8 scripts/eval_patch_recheck.py --sample 0.2          # free, deterministic 20% sample
    python -X utf8 scripts/eval_patch_recheck.py --llm --sample 0.2    # prints the paid estimate, refuses
    python -X utf8 scripts/eval_patch_recheck.py --llm --sample 0.2 --yes --max-usd 10   # PAID

The LLM mode never starts without ``--yes``; it prints the estimate first, loads
``ANTHROPIC_API_KEY`` only inside the paid path (``jettae.evalkit.realdata.keys.load_api_key``;
the key is never printed), uses ``JETTAE_ANTHROPIC_MODEL`` (default ``claude-sonnet-5-5``), and every
call goes through the budget gate of the real-data studies (``--max-usd``).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import jettae.evalkit as kit
from jettae.check.context import CheckContext
from jettae.check.registry import rule_catalog, run_checks
from jettae.domain import (
    AddEdge,
    AddNode,
    EdgeKind,
    Finding,
    NodeKind,
    OpsGraph,
    Patch,
    PatchAuthor,
    PatchStatus,
    RemoveEdge,
    SetAttr,
    SimConfig,
)
from jettae.evalkit.conditions import case_inputs, offline_graph, prepare_worker, run_sync
from jettae.evalkit.facts import oracle_graph
from jettae.evalkit.harness import default_workers, reproducibility_sample
from jettae.improve import BLOCKING_SEVERITIES, LlmDrafter, PatchError, TemplateDrafter, apply_ops
from jettae.improve.drafter import FixPlan
from jettae.llm.base import (
    LLMClient,
    LLMDisabledError,
    LLMError,
    LLMOutputError,
    LLMRefusalError,
    LLMResult,
)
from jettae.llm.fake import FakeLLM
from jettae.llm.prompting import build_system, render_user_content
from jettae.llm.schema import to_strict_json_schema
from jettae.orchestrator.pipeline import (
    build_context,
    evaluate_graph,
    gather_scenarios,
    rehearse_all,
    validate_drafts,
)

ROOT: Final = Path(__file__).resolve().parents[1]
SEED: Final = 20261012
ORGS: Final = 20
DEFAULT_OUT: Final = Path("C:/jettae/data/eval/patch-recheck")
DEFAULT_LLM_OUT: Final = Path("C:/jettae/data/eval/patch-recheck-llm")
CONDITIONS: Final = ("B", "O")
CONDITION_LABEL: Final = {"B": "B 규칙 파서 + 그래프 (제품 LLM 끔 경로)", "O": "O 정답 그래프 (상한)"}
WORKSPACE: Final = "eval-patch-recheck"
BASE_VERSION: Final = 1
SIM: Final = SimConfig()
#: Upper bound on approvals in the "approve everything" simulation (a case has at most a handful).
LOOP_MAX_APPROVALS: Final = 12
#: Default model of the paid mode (the 2×2 study's extraction model).
DEFAULT_MODEL: Final = "claude-sonnet-5-5"
#: LlmDrafter's own default output budget per call (thinking counts against it).
DRAFTER_MAX_TOKENS: Final = 2000
#: Conservative characters-per-token of the estimate (Korean prompts; realdata.pricing uses the same).
EST_CHARS_PER_TOKEN: Final = 1.0
ESTIMATE_FRACTIONS: Final = (0.1, 0.2, 1.0)

#: Roster CSV columns as the evalkit renderer writes them (render.roster_csv).
_COL_NAME, _COL_STATUS, _COL_TITLE, _COL_TEAM, _COL_ALIASES = "이름", "상태", "직급", "팀", "별칭"
_ACTIVE_LABEL: Final = "재직"
_STD_TITLES: Final = (
    "사원", "주임", "대리", "과장", "차장", "부장", "팀장", "파트장", "실장", "본부장",
    "이사", "상무", "전무", "부사장", "사장", "대표",
)  # fmt: skip
#: Words that can stand before a title without being a person's name.
_NOT_NAMES: Final = frozenset(
    {"담당", "1차", "새", "같은", "다른", "해당", "최종", "책임", "상향", "승인", "대리"}
)

PERSON_VIOLATIONS: Final = frozenset({"not_in_roster", "not_active", "ambiguous_namesake"})
_SEVERITY_KO: Final = {"critical": "치명", "high": "높음", "medium": "중간", "low": "낮음", "info": "정보"}


# --------------------------------------------------------------------------- corpus


def sealed_cases(sample: float | None) -> list[kit.EvalCase]:
    cases = kit.build_cases(kit.generate_gold_orgs(ORGS, SEED), seed=SEED)
    return reproducibility_sample(cases, sample) if sample is not None else cases


def operator_label(case: kit.EvalCase) -> str:
    return case.operator.value if case.operator is not None else "clean"


# --------------------------------------------------------------------------- independent roster


@dataclass(frozen=True)
class RosterRow:
    name: str
    status: str
    title: str
    team: str
    aliases: tuple[str, ...]


def roster_rows(case: kit.EvalCase) -> list[RosterRow]:
    """The case's roster CSV parsed with the standard library (not :func:`jettae.ingest.load_roster`)."""
    reader = csv.DictReader(io.StringIO(case.roster_csv))
    missing = {_COL_NAME, _COL_STATUS, _COL_TITLE, _COL_TEAM, _COL_ALIASES} - set(reader.fieldnames or ())
    if missing:
        raise ValueError(f"{case.case_id}: 명부 CSV 열이 없습니다: {sorted(missing)}")
    return [
        RosterRow(
            name=(r[_COL_NAME] or "").strip(),
            status=(r[_COL_STATUS] or "").strip(),
            title=(r[_COL_TITLE] or "").strip(),
            team=(r[_COL_TEAM] or "").strip(),
            aliases=tuple(a.strip() for a in (r[_COL_ALIASES] or "").split(";") if a.strip()),
        )
        for r in reader
    ]


def _teams_of(graph: OpsGraph, person_id: str) -> set[str]:
    return {t.name.replace(" ", "") for t in graph.successors(person_id, EdgeKind.MEMBER_OF)}


def person_verdict(name: str, teams: set[str], rows: Sequence[RosterRow]) -> tuple[str, str | None]:
    """(verdict, roster status label) of one person a patch links to, from the CSV alone.

    ``active`` / ``active_namesake`` are fine; ``not_in_roster``, ``not_active`` and
    ``ambiguous_namesake`` (several rows, mixed status, the node's team does not single one out)
    are what the KPI forbids.
    """
    matches = [r for r in rows if r.name == name] or [r for r in rows if name in r.aliases]
    if not matches:
        return "not_in_roster", None
    if len(matches) > 1 and teams:
        by_team = [r for r in matches if r.team.replace(" ", "") in teams]
        if len(by_team) == 1:
            matches = by_team
    statuses = {r.status for r in matches}
    if statuses == {_ACTIVE_LABEL}:
        return ("active" if len(matches) == 1 else "active_namesake"), _ACTIVE_LABEL
    if _ACTIVE_LABEL in statuses:
        return "ambiguous_namesake", "/".join(sorted(statuses))
    return "not_active", "/".join(sorted(statuses))


def linked_people(patch: Patch, base: OpsGraph, after: OpsGraph | None) -> list[str]:
    """Person node ids a patch adds, links to or edits (removals excluded) — read off the ops here."""
    added = {op.node.id: op.node for op in patch.ops if isinstance(op, AddNode)}

    def is_person(nid: str) -> bool:
        node = (after.nodes.get(nid) if after is not None else None) or base.nodes.get(nid) or added.get(nid)
        return node is not None and node.kind is NodeKind.PERSON

    ids: set[str] = set()
    for op in patch.ops:
        if isinstance(op, AddNode):
            ids.update([op.node.id] if op.node.kind is NodeKind.PERSON else [])
        elif isinstance(op, AddEdge):
            ids.update(n for n in (op.edge.src, op.edge.dst) if is_person(n))
        elif isinstance(op, SetAttr):
            ids.update([op.target_id] if is_person(op.target_id) else [])
    return sorted(ids)


def removed_people(patch: Patch, base: OpsGraph) -> set[str]:
    out: set[str] = set()
    for op in patch.ops:
        if isinstance(op, RemoveEdge):
            edge = base.edges.get(op.edge_id) or op.snapshot
            if edge is not None:
                out.update(
                    n
                    for n in (edge.src, edge.dst)
                    if (node := base.nodes.get(n)) and node.kind is NodeKind.PERSON
                )
    return out


def text_people(text: str, rows: Sequence[RosterRow], extra_words: Iterable[str]) -> dict[str, Any]:
    """People a patch sentence names, judged against the roster CSV.

    ``mentioned`` = roster full names in the text with their status label; ``invented`` = a word
    written like a person ("홍길동 대리", "홍길동 CTO") that is no roster name or alias (team and
    service names and a few common words excluded).
    """
    mentioned = {r.name: r.status for r in rows if r.name and r.name in text}
    titles = set(_STD_TITLES) | {r.title for r in rows if r.title}
    pattern = "|".join(re.escape(t) for t in sorted(titles, key=len, reverse=True))
    known = {r.name for r in rows} | {a for r in rows for a in r.aliases}
    skip = known | set(_NOT_NAMES) | {w.replace(" ", "") for w in extra_words} | {r.team for r in rows}
    invented = sorted(
        {
            word
            for word in re.findall(rf"(?<![가-힣])([가-힣]{{2,4}})\s?(?:{pattern})", text)
            if word not in skip and not word.endswith(("팀", "실", "부"))
        }
    )
    return {"mentioned": mentioned, "invented": invented}


# --------------------------------------------------------------------------- judging one case


@dataclass(frozen=True)
class Judged:
    inputs: Any
    graph: OpsGraph
    ctx: CheckContext
    findings: tuple[Finding, ...]
    scenarios: tuple[Any, ...]
    rehearsals: tuple[Any, ...]


def judge(case: kit.EvalCase, condition: str) -> Judged:
    """Graph (B or O) → fixed rules → template drills → replay: the LLM-off diagnosis stages."""
    inputs = case_inputs(case)
    if condition == "B":
        graph = offline_graph(
            (inputs.document,), roster=inputs.roster, catalog=inputs.catalog, policy=inputs.policy
        )
    elif condition == "O":
        graph = oracle_graph(case.gold_facts, roster=inputs.roster, catalog=inputs.catalog)
    else:
        raise ValueError(f"알 수 없는 조건: {condition}")
    ctx = build_context(graph, roster=inputs.roster, catalog=inputs.catalog, policy=inputs.policy)
    findings = tuple(run_checks(ctx))
    scenarios, _notes = gather_scenarios(ctx)
    rehearsals = rehearse_all(graph, scenarios, ctx=ctx, config=SIM)
    return Judged(inputs, graph, ctx, findings, scenarios, rehearsals)


def _service_name(graph: OpsGraph, sid: str | None) -> str | None:
    node = graph.nodes.get(sid) if sid is not None else None
    return node.name if node is not None else sid


def target_finding_ids(case: kit.EvalCase, graph: OpsGraph, findings: Sequence[Finding]) -> list[str]:
    """Findings that *are* the injected defect: expected rule on the target service (gold vocabulary)."""
    if case.target_service is None:
        return []
    expected = set(case.expected_rule_ids)
    return sorted(
        f.id
        for f in findings
        if f.rule_id in expected and _service_name(graph, f.service_id) == case.target_service
    )


class _RecordingRefusal(FakeLLM):
    """Records every request LlmDrafter would send and answers each with an LLMError (template kept)."""

    def __init__(self) -> None:
        super().__init__(lambda *_: LLMError("비용 추정용 기록 전용 호출 (보내지 않음)"))


def llm_requests(j: Judged) -> list[dict[str, int]]:
    """The exact requests LlmDrafter would make on this graph, rebuilt offline (no network)."""
    recorder = _RecordingRefusal()
    run_sync(
        LlmDrafter(recorder).adraft(j.ctx, j.findings, workspace_id=WORKSPACE, base_version=BASE_VERSION)
    )
    out = []
    for call in recorder.calls:
        system = build_system(call.instructions)
        user = render_user_content(call.documents, call.context)
        schema = json.dumps(to_strict_json_schema(call.schema), ensure_ascii=False)
        out.append({"input_chars": len(system) + len(user) + len(schema), "max_tokens": call.max_tokens})
    return out


def _fail_reasons(patch: Patch, new_severe: Sequence[str]) -> list[str]:
    v = patch.validation
    assert v is not None
    reasons = []
    if v.integrity_errors and not v.resolved_finding_ids and v.score_after is None:
        reasons.append("apply_error")
    elif v.integrity_errors:
        reasons.append("provenance")
    if not v.resolved_finding_ids:
        reasons.append("nothing_resolved")
    if new_severe:
        reasons.append("blocking_new_finding")
    if v.rehearsals_regressed:
        reasons.append("rehearsal_regression")
    if v.unknown_people:
        reasons.append("unknown_people")
    return reasons


def patch_record(
    patch: Patch,
    plan: FixPlan | None,
    j: Judged,
    rows: Sequence[RosterRow],
    extra_words: Sequence[str],
) -> dict[str, Any]:
    """Everything measured about one draft (validator verdict + independent checks)."""
    v = patch.validation
    assert v is not None
    before_ids = {f.id for f in j.findings}
    rec: dict[str, Any] = {
        "id": patch.id,
        "kind": plan.group.kind.value if plan is not None else None,
        "has_candidates": bool(plan.candidates) if plan is not None else False,
        "author": patch.author.value,
        "status": patch.status.value,
        "n_ops": len(patch.ops),
        "finding_ids": list(patch.finding_ids),
        "passed": v.passed,
        "resolved": list(v.resolved_finding_ids),
        "unresolved": list(v.unresolved_finding_ids),
        "new": list(v.new_finding_ids),
        "rehearsals_fixed": len(v.rehearsals_fixed),
        "rehearsals_regressed": list(v.rehearsals_regressed),
        "unknown_people": list(v.unknown_people),
        "integrity_errors": len(v.integrity_errors),
        "text": patch.runbook_text,
        "llm_usage": dict(patch.llm_usage),
    }
    if not patch.ops:
        return rec
    after_graph: OpsGraph | None
    try:
        after_graph, _ = apply_ops(j.graph, patch.ops)
    except PatchError:
        after_graph = None
    new_severity: dict[str, str] = {}
    raw_closed: set[str] = set()
    if after_graph is not None:
        after = run_checks(j.ctx.with_graph(after_graph))
        after_ids = {f.id for f in after}
        raw_closed = before_ids - after_ids
        new_severity = {f.id: f.severity.value for f in after if f.id in set(v.new_finding_ids)}
    new_severe = [fid for fid, sev in new_severity.items() if sev in {s.value for s in BLOCKING_SEVERITIES}]
    people = []
    for pid in linked_people(patch, j.graph, after_graph):
        node = (after_graph.nodes.get(pid) if after_graph is not None else None) or j.graph.nodes.get(pid)
        name = node.name if node is not None else pid
        teams = _teams_of(after_graph or j.graph, pid)
        verdict, status = person_verdict(name, teams, rows)
        people.append({"id": pid, "name": name, "verdict": verdict, "roster_status": status})
    removed_names = {j.graph.nodes[p].name for p in removed_people(patch, j.graph) if p in j.graph.nodes}
    texts = text_people(patch.runbook_text, rows, extra_words)
    rec.update(
        {
            "new_severity": new_severity,
            "new_severe": new_severe,
            "strict": v.passed and not v.unresolved_finding_ids and not v.new_finding_ids,
            "recheck_confirms_resolved": set(v.resolved_finding_ids) <= raw_closed,
            "fail_reasons": [] if v.passed else _fail_reasons(patch, new_severe),
            "adds_person_node": any(
                isinstance(op, AddNode) and op.node.kind is NodeKind.PERSON for op in patch.ops
            ),
            "people": people,
            "text_mentioned": texts["mentioned"],
            "text_invented": texts["invented"],
            "text_non_active_unlinked": sorted(
                n for n, s in texts["mentioned"].items() if s != _ACTIVE_LABEL and n not in removed_names
            ),
        }
    )
    return rec


def approve_loop(j: Judged, case: kit.EvalCase) -> dict[str, Any]:
    """Approve the first VALIDATED draft, re-judge the new version, repeat (what a lead clicking
    "승인" on every validated draft would get). Uses the product's re-judgement after approval."""
    graph = j.graph
    approvals = 0
    seen: set[str] = set()
    capped = False
    while True:
        ev = evaluate_graph(
            graph,
            roster=j.inputs.roster,
            catalog=j.inputs.catalog,
            policy=j.inputs.policy,
            workspace_id=WORKSPACE,
            version=BASE_VERSION + approvals,
            config=SIM,
        )
        validated = [p for p in ev.patches if p.status is PatchStatus.VALIDATED and p.id not in seen]
        if not validated:
            final = ev.findings
            break
        if approvals >= LOOP_MAX_APPROVALS:
            capped, final = True, ev.findings
            break
        chosen = validated[0]
        seen.add(chosen.id)
        graph, _ = apply_ops(graph, chosen.ops)
        approvals += 1
    initial_ids = {f.id for f in j.findings}
    targets = target_finding_ids(case, j.graph, j.findings)
    still = set(target_finding_ids(case, graph, final))
    return {
        "approvals": approvals,
        "capped": capped,
        "initial_findings": len(j.findings),
        "final_findings": len(final),
        "final_rules": sorted(Counter(f.rule_id for f in final).items()),
        "final_new_ids": sorted({f.id for f in final} - initial_ids),
        "targets_detected": len(targets),
        "targets_closed": len([t for t in targets if t not in still]) if targets else 0,
        "target_rule_remaining": bool(
            case.target_service is not None
            and any(
                f.rule_id in case.expected_rule_ids
                and _service_name(graph, f.service_id) == case.target_service
                for f in final
            )
        ),
    }


def case_record(
    case: kit.EvalCase, condition: str, j: Judged, plans: Sequence[FixPlan | None], patches: Sequence[Patch]
) -> dict[str, Any]:
    rows = roster_rows(case)
    extra = [s.name for s in j.inputs.catalog] + [n.name for n in j.graph.nodes_of(NodeKind.TEAM)]
    precs = [patch_record(p, plan, j, rows, extra) for p, plan in zip(patches, plans, strict=True)]
    in_patch = {fid for r in precs if r["n_ops"] for fid in r["finding_ids"]}
    in_memo = {fid for r in precs if not r["n_ops"] for fid in r["finding_ids"]}
    approvable = {fid for r in precs if r["passed"] for fid in r["finding_ids"]}
    resolved = {fid for r in precs if r["passed"] for fid in r["resolved"]}
    findings = []
    for f in j.findings:
        cover = "patch" if f.id in in_patch else "memo" if f.id in in_memo else "none"
        findings.append(
            {
                "id": f.id,
                "rule": f.rule_id,
                "severity": f.severity.value,
                "fix": f.fix.kind.value if f.fix is not None else None,
                "cover": cover,
                "approvable": f.id in approvable,
                "resolved_by_approvable": f.id in resolved,
            }
        )
    return {
        "case_id": case.case_id,
        "condition": condition,
        "org": case.org_id,
        "operator": operator_label(case),
        "style": str(case.style),
        "position": str(case.position),
        "held_out": case.held_out,
        "error": None,
        "n_scenarios": len(j.scenarios),
        "findings": findings,
        "targets": target_finding_ids(case, j.graph, j.findings),
        "patches": precs,
    }


def _plans_for(j: Judged) -> list[FixPlan]:
    return TemplateDrafter().plans(j.ctx, j.findings)


def run_template_case(case: kit.EvalCase, condition: str, loop: bool, estimate: bool) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        j = judge(case, condition)
        plans = _plans_for(j)
        drafts = TemplateDrafter().draft(j.ctx, j.findings, workspace_id=WORKSPACE, base_version=BASE_VERSION)
        if [p.finding_ids for p in drafts] != [pl.group.finding_ids for pl in plans]:
            raise RuntimeError("초안과 계획의 순서가 맞지 않습니다")
        patches = validate_drafts(
            j.ctx, drafts, scenarios=j.scenarios, findings=j.findings, results=j.rehearsals, config=SIM
        )
        rec = case_record(case, condition, j, plans, patches)
        if estimate:
            rec["llm_requests"] = llm_requests(j)
        if loop:
            rec["loop"] = approve_loop(j, case)
    except Exception as exc:  # one broken case must not abort the corpus; it is reported
        rec = {
            "case_id": case.case_id,
            "condition": condition,
            "org": case.org_id,
            "operator": operator_label(case),
            "style": str(case.style),
            "position": str(case.position),
            "held_out": case.held_out,
            "error": f"{type(exc).__name__}: {exc}",
            "findings": [],
            "targets": [],
            "patches": [],
        }
    rec["seconds"] = round(time.perf_counter() - start, 3)
    return rec


def _batch(job: tuple[tuple[kit.EvalCase, ...], tuple[str, ...], bool, bool]) -> list[dict[str, Any]]:
    cases, conditions, loop, estimate = job
    return [run_template_case(c, cond, loop, estimate) for c in cases for cond in conditions]


def run_free(
    cases: Sequence[kit.EvalCase], conditions: Sequence[str], *, workers: int, loop: bool
) -> list[dict[str, Any]]:
    size = max(1, math.ceil(len(cases) / (max(1, workers) * 4)))
    jobs = [(tuple(cases[i : i + size]), tuple(conditions), loop, True) for i in range(0, len(cases), size)]
    out: list[dict[str, Any]] = []
    done = 0
    if workers <= 1:
        prepare_worker()
        for job in jobs:
            out.extend(_batch(job))
            done += len(job[0])
            print(f"  {done}/{len(cases)} 사례", file=sys.stderr, flush=True)
        return out
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), initializer=prepare_worker) as pool:
        for batch in pool.map(_batch, jobs):
            out.extend(batch)
            done += len(batch) // max(1, len(conditions))
            print(f"  {done}/{len(cases)} 사례", file=sys.stderr, flush=True)
    return out


# --------------------------------------------------------------------------- aggregation


def rate(hits: int, n: int) -> dict[str, Any]:
    r = kit.Rate.of(hits, n)
    return {"hits": r.hits, "n": r.n, "rate": r.rate, "ci95": list(r.ci95) if r.ci95 else None}


def _patch_stats(patches: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ops = [p for p in patches if p["n_ops"]]
    people = [x for p in ops for x in p.get("people", ())]
    return {
        "drafts": len(patches),
        "with_ops": len(ops),
        "memos": len(patches) - len(ops),
        "pass_rate": rate(sum(p["passed"] for p in ops), len(ops)),
        "strict_rate": rate(sum(bool(p.get("strict")) for p in ops), len(ops)),
        "pass_over_all_drafts": rate(sum(p["passed"] for p in ops), len(patches)),
        "validator_unknown_people": sum(len(p["unknown_people"]) for p in patches),
        "patches_with_validator_unknown_people": sum(bool(p["unknown_people"]) for p in patches),
        "linked_people_checked": len(people),
        "independent_person_violations": sum(x["verdict"] in PERSON_VIOLATIONS for x in people),
        "independent_verdicts": dict(Counter(x["verdict"] for x in people)),
        "person_nodes_added": sum(bool(p.get("adds_person_node")) for p in ops),
        "text_invented_names": sum(len(p.get("text_invented", ())) for p in ops),
        "text_non_active_unlinked": sum(len(p.get("text_non_active_unlinked", ())) for p in ops),
        "rehearsal_regressions": sum(len(p["rehearsals_regressed"]) for p in ops),
        "rehearsals_fixed": sum(p["rehearsals_fixed"] for p in ops),
        "recheck_mismatch": sum(not p.get("recheck_confirms_resolved", True) for p in ops),
        "fail_reasons": dict(Counter(r for p in ops for r in p.get("fail_reasons", ()))),
        "new_low_findings_in_passed": sum(bool(p["new"]) for p in ops if p["passed"]),
    }


def _finding_stats(findings: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(findings)
    return {
        "findings": n,
        "cover_patch": rate(sum(f["cover"] == "patch" for f in findings), n),
        "cover_memo": rate(sum(f["cover"] == "memo" for f in findings), n),
        "cover_none": rate(sum(f["cover"] == "none" for f in findings), n),
        "in_approvable_patch": rate(sum(f["approvable"] for f in findings), n),
        "resolved_by_approvable": rate(sum(f["resolved_by_approvable"] for f in findings), n),
    }


def decision_units(ok: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pass rate over distinct decisions (org × operator × fix kind): the six renders of one org and
    operator (3 styles × 2 positions) usually ask the drafter the same question, so patches are
    not independent trials."""
    units: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
    for r in ok:
        for p in r["patches"]:
            if p["n_ops"]:
                units[(r["org"], r["operator"], p["kind"] or "?")].append(bool(p["passed"]))
    return {
        "units": len(units),
        "all_pass": rate(sum(all(v) for v in units.values()), len(units)),
        "mixed": sum(len(set(v)) > 1 for v in units.values()),
        "failing_units": sorted("/".join(k) for k, v in units.items() if not all(v)),
    }


BOOTSTRAP_REPS: Final = 2000


def org_bootstrap(ok: Sequence[dict[str, Any]], reps: int = BOOTSTRAP_REPS) -> list[float] | None:
    """Percentile 95% interval of the patch pass rate, resampling the 20 organisations (seeded)."""
    per_org: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in ok:
        for p in r["patches"]:
            if p["n_ops"]:
                per_org[r["org"]][0] += int(bool(p["passed"]))
                per_org[r["org"]][1] += 1
    orgs = sorted(per_org)
    if not orgs:
        return None
    rng = random.Random(SEED)
    rates = []
    for _ in range(reps):
        pick = [orgs[rng.randrange(len(orgs))] for _ in orgs]
        n = sum(per_org[o][1] for o in pick)
        rates.append(sum(per_org[o][0] for o in pick) / n if n else 0.0)
    rates.sort()
    return [round(rates[int(0.025 * reps)], 4), round(rates[int(0.975 * reps) - 1], 4)]


def summarize(records: Sequence[dict[str, Any]], condition: str) -> dict[str, Any]:
    recs = [r for r in records if r["condition"] == condition]
    ok = [r for r in recs if r["error"] is None]
    patches = [p for r in ok for p in r["patches"]]
    findings = [f for r in ok for f in r["findings"]]
    target_ids = [(r, t) for r in ok for t in r["targets"]]
    by_id = {(r["case_id"], f["id"]): f for r in ok for f in r["findings"]}
    targets = [by_id[(r["case_id"], t)] for r, t in target_ids]
    mutated = [r for r in ok if r["operator"] != "clean"]
    out: dict[str, Any] = {
        "condition": condition,
        "cases": len(recs),
        "errors": len(recs) - len(ok),
        "error_examples": [f"{r['case_id']}: {r['error']}" for r in recs if r["error"]][:5],
        "cases_with_findings": sum(bool(r["findings"]) for r in ok),
        "clean_cases_with_drafts": sum(bool(r["patches"]) for r in ok if r["operator"] == "clean"),
        "patches": _patch_stats(patches),
        "findings": _finding_stats(findings),
        "injected": {
            "mutated_cases": len(mutated),
            "cases_target_detected": sum(bool(r["targets"]) for r in mutated),
            **{f"target_{k}": v for k, v in _finding_stats(targets).items()},
        },
    }
    for key, label in (("operator", "by_operator"), ("style", "by_style")):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in ok:
            groups[r[key]].append(r)
        out[label] = {
            g: {
                "cases": len(rs),
                "patches": _patch_stats([p for r in rs for p in r["patches"]]),
                "findings": _finding_stats([f for r in rs for f in r["findings"]]),
                "targets": _finding_stats(
                    [
                        {(r["case_id"], f["id"]): f for f in r["findings"]}[(r["case_id"], t)]
                        for r in rs
                        for t in r["targets"]
                    ]
                ),
            }
            for g, rs in sorted(groups.items())
        }
    kinds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in patches:
        kinds[p["kind"] or "?"].append(p)
    out["by_fix_kind"] = {k: _patch_stats(v) for k, v in sorted(kinds.items())}
    loops = [r["loop"] for r in ok if "loop" in r]
    if loops:
        detected = [lp for lp in loops if lp["targets_detected"]]
        out["loop"] = {
            "cases": len(loops),
            "approvals": sum(lp["approvals"] for lp in loops),
            "capped": sum(lp["capped"] for lp in loops),
            "initial_findings": sum(lp["initial_findings"] for lp in loops),
            "final_findings": sum(lp["final_findings"] for lp in loops),
            "cases_all_findings_closed": rate(
                sum(lp["final_findings"] == 0 for lp in loops if lp["initial_findings"]),
                sum(bool(lp["initial_findings"]) for lp in loops),
            ),
            "target_closed_cases": rate(
                sum(not lp["target_rule_remaining"] for lp in detected), len(detected)
            ),
            "final_rules": dict(
                Counter(rule for lp in loops for rule, n in lp["final_rules"] for _ in range(n))
            ),
            "final_new_findings": sum(len(lp["final_new_ids"]) for lp in loops),
        }
    examples = [
        {"case": r["case_id"], "kind": p["kind"], "reasons": p.get("fail_reasons"), "text": p["text"]}
        for r in ok
        for p in r["patches"]
        if p["n_ops"] and not p["passed"]
    ]
    out["fail_examples"] = examples[:8]
    out["decision_units"] = decision_units(ok)
    out["pass_rate_org_bootstrap_ci95"] = org_bootstrap(ok)
    out["fail_breakdown"] = dict(
        Counter(
            f"{r['operator']} · {p['kind']} · 새 높음 이상 끊김 "
            f"{','.join(sorted({x.split(':')[0] for x in p.get('new_severe', ())})) or '없음'}"
            for r in ok
            for p in r["patches"]
            if p["n_ops"] and not p["passed"]
        )
    )
    out["strict_gap_breakdown"] = dict(
        Counter(
            f"{r['operator']} · {p['kind']} · 새 높음 미만 끊김 "
            + ",".join(
                sorted(
                    {
                        f"{x.split(':')[0]}[{_SEVERITY_KO.get(sev, sev)}]"
                        for x, sev in p["new_severity"].items()
                    }
                )
            )
            for r in ok
            for p in r["patches"]
            if p["n_ops"] and p["passed"] and p["new"]
        )
    )
    out["invented_name_examples"] = [
        {"case": r["case_id"], "names": p["text_invented"], "text": p["text"]}
        for r in ok
        for p in r["patches"]
        if p.get("text_invented")
    ][:8]
    out["non_active_text_examples"] = [
        {"case": r["case_id"], "names": p["text_non_active_unlinked"], "text": p["text"]}
        for r in ok
        for p in r["patches"]
        if p.get("text_non_active_unlinked")
    ][:8]
    out["seconds_total"] = round(sum(r.get("seconds", 0.0) for r in recs), 1)
    return out


# --------------------------------------------------------------------------- paid estimate


def price_requests(requests: Iterable[dict[str, int]], model: str, effort: str) -> dict[str, Any]:
    from jettae.evalkit.realdata.pricing import call_usd, thinking_allowance

    calls = input_tokens = output_tokens = 0
    usd = 0.0
    for req in requests:
        tin = math.ceil(req["input_chars"] / EST_CHARS_PER_TOKEN)
        # visible JSON (choice + ≤600-char sentence + ≤800-char rationale) plus thinking, capped by max_tokens
        tout = min(req["max_tokens"], 700 + thinking_allowance("anthropic", model, effort))  # type: ignore[arg-type]
        calls += 1
        input_tokens += tin
        output_tokens += tout
        usd += call_usd(model, input_tokens=tin, output_tokens=tout)
    return {
        "model": model,
        "effort": effort,
        "calls": calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "usd": round(usd, 4),
        "usd_if_every_call_repaired": round(usd * 2.2, 4),
    }


def estimate_table(
    records: Sequence[dict[str, Any]],
    cases: Sequence[kit.EvalCase],
    model: str,
    effort: str,
    conditions: Sequence[str],
) -> list[dict[str, Any]]:
    rows = []
    reqs = {(r["case_id"], r["condition"]): r.get("llm_requests", []) for r in records}
    for fraction in ESTIMATE_FRACTIONS:
        chosen = reproducibility_sample(cases, fraction)
        for cond in conditions:
            priced = price_requests(
                (q for c in chosen for q in reqs.get((c.case_id, cond), [])), model, effort
            )
            rows.append({"fraction": fraction, "cases": len(chosen), "condition": cond, **priced})
    return rows


# --------------------------------------------------------------------------- report


def _rules_ko(text: str) -> str:
    """``JT-ESC-001`` → ``JT-ESC-001(상향 경로 없음)`` wherever a rule id appears."""
    titles = {r.rule_id: r.title for r in rule_catalog()}
    return re.sub(r"JT-[A-Z]+-\d{3}", lambda m: f"{m.group(0)}({titles.get(m.group(0), '?')})", text)


def _durations(summary: dict[str, Any]) -> str:
    return ", ".join(f"{c} {summary[c]['seconds_total']:,.0f}초" for c in summary["meta"]["conditions"])


def _pct(r: dict[str, Any]) -> str:
    if not r["n"]:
        return "해당 없음 (0건)"
    lo, hi = r["ci95"]
    return f"{r['hits']:,}/{r['n']:,} = {r['rate'] * 100:.1f}% (95% 구간 {lo * 100:.1f}–{hi * 100:.1f}%)"


def _short(r: dict[str, Any]) -> str:
    return f"{r['hits']}/{r['n']} ({r['rate'] * 100:.1f}%)" if r["n"] else "–"


_KIND_KO: Final = {
    "assign_owner": "담당자 지정(퇴사자 교체 포함)",
    "add_escalation": "상향 경로 추가",
    "break_escalation_cycle": "상향 순환 끊기",
    "assign_approver": "승인자 지정",
    "add_deputy": "대리자 지정",
    "add_fallback_channel": "대체 연락 수단",
    "assign_executor": "실행자 지정",
    "fix_deadline": "신고 시한 고치기",
    "add_reporting_step": "신고 단계 추가",
    "add_detection": "감지 설계(메모)",
    "add_rollback": "롤백 절차(메모)",
    "add_failover": "페일오버(메모)",
    "adjust_oncall": "당직표(메모)",
    "add_service_runbook": "서비스 런북(메모)",
    "confirm_identity": "신원 확인(메모)",
    "review_element": "보류 요소 검토(메모)",
}
_REASON_KO: Final = {
    "nothing_resolved": "겨냥한 끊김이 하나도 해소되지 않음",
    "blocking_new_finding": "높음 이상 새 끊김 발생",
    "rehearsal_regression": "통과하던 리허설이 실패(회귀)",
    "unknown_people": "명부 외 인물",
    "apply_error": "그래프에 적용 불가",
    "provenance": "출처 표기 오류",
}


def render_llm_md(summary: dict[str, Any]) -> str:
    """Report of a paid LlmDrafter run (same tables as the free run plus the LLM-specific lines)."""
    meta = summary["meta"]
    lines = [
        "# AI 수정 초안 재검사 — LLM 초안(LlmDrafter) 실행 결과",
        "",
        f"- 모델 `{meta['model']}`, effort `{meta['effort']}`, 표본 {meta['sample'] or '전수'}, 사례 "
        f"{meta['cases']:,}건,"
        f" 조건 {','.join(meta['conditions'])}. 실제 지출 ${summary['spend']['spent_usd']:.4f}."
        + (f" **중단됨: {meta['stopped']}**" if meta.get("stopped") else ""),
        "- 표본 결과다. 전체 1,320건 값이라고 쓰지 않는다.",
        "",
    ]
    for cond in meta["conditions"]:
        llm = summary[cond]["llm"]
        lines += [
            f"## {CONDITION_LABEL[cond]} — LLM 문장",
            "",
            f"- LLM 호출 {llm['calls']:,}회, 결과: "
            + ", ".join(f"{k} {v}" for k, v in sorted(llm["call_outcomes"].items())),
            f"- LLM에 보낸 수정안 {llm['patches_sent_to_llm']:,}건 중 LLM 문장이 채택된 것 "
            f"{llm['ai_authored']:,}건."
            f" 템플릿으로 되돌아간 것: 호출 실패·잘림 {llm['fell_back_call_failed']:,}건, 문장이 연산·명부와 "
            f"맞지 않아"
            f" 제품이 거부 {llm['fell_back_rejected_by_drafter']:,}건, 호출 전 중단 "
            f"{llm['fell_back_not_sent']:,}건.",
            "  (사람 결정이 없는 수정안 — 신고 시한·연락 수단 — 은 제품 설계상 LLM에 보내지 않고 템플릿 "
            "문장을 쓴다.)",
            f"- **LLM 문장 수정안의 재검사 통과율: {_pct(llm['ai_pass_rate'])}**",
            f"- 명부 외 인물(LLM 문장 수정안): 검증기 {llm['ai_unknown_people']}건 · 명부 CSV 독립 대조"
            f" {llm['ai_independent_person_violations']}건 · 문장 속 명부에 없는 이름 "
            f"{llm['ai_text_invented_names']}건",
            f"- 같은 그래프의 템플릿 초안과 짝 비교 {llm['paired']:,}쌍: 판정 같음 "
            f"{llm['paired_same_verdict']:,},"
            f" LLM만 통과 {llm['paired_ai_pass_template_fail']}, 템플릿만 통과 "
            f"{llm['paired_ai_fail_template_pass']}.",
            "",
        ]
    body = render_md(summary, title="## 전체 표 (LLM 초안 기준)")
    return "\n".join(lines) + "\n" + body


def _target_lines(summary: dict[str, Any]) -> list[str]:
    """The application's targets (재검사 통과율 90% 이상 / 명부 외 인물 0건) against this run,
    stated plainly."""
    lines = ["## 신청서 목표 대비 (이번 실행 = 템플릿 초안, LLM 끔)", ""]
    for cond in summary["meta"]["conditions"]:
        p = summary[cond]["patches"]
        r = p["pass_rate"]
        people = (
            p["validator_unknown_people"]
            + p["independent_person_violations"]
            + p["person_nodes_added"]
            + p["text_invented_names"]
        )
        if r["n"]:
            lo, hi = r["ci95"]
            verdict = "달성" if r["rate"] >= 0.9 else "미달"
            band = "90%를 포함" if lo <= 0.9 <= hi else ("90%보다 위" if lo > 0.9 else "90%보다 아래")
            rate_txt = (
                f"{r['hits']:,}/{r['n']:,} = {r['rate'] * 100:.1f}% → 목표 90% 이상 **{verdict}** "
                f"(95% 구간 {lo * 100:.1f}–{hi * 100:.1f}%, {band})"
            )
        else:
            rate_txt = "수정안 0건 → 판정 불가"
        lines.append(
            f"- {cond}: 재검사 통과율 {rate_txt}. 명부 외 인물 네 가지 합계 {people}건 → 목표 0건 "
            f"**{'달성' if people == 0 else '미달'}**."
        )
    lines += [
        "- 이 판정은 템플릿 초안의 값이다. 신청서의 'AI 수정 초안'(LLM이 쓴 문장) 통과율은 유료 실행 전까지 "
        "값이 없다.",
        "",
    ]
    return lines


def render_md(summary: dict[str, Any], *, title: str | None = None) -> str:  # noqa: PLR0912 - one block per table
    meta = summary["meta"]
    lines = [
        title or "# AI 수정 초안 재검사 — 봉인셋 무료 실행 결과 (LLM 끔)",
        "",
        f"- 실행일: {meta['run_date']} · 제품 코드: 동결 커밋 `{meta['frozen_commit']}` (src/ 변경 없음: "
        f"{meta['src_clean']})",
        f"- 코퍼스: 봉인 평가셋 시드 {SEED}, 조직 {ORGS}곳, {meta['cases']:,}건"
        + (f" (결정적 표본 {meta['sample']:g})" if meta["sample"] else " (전수)"),
        "- 독립 대조 검사기의 양성 대조(지어낸 이름·퇴사자를 일부러 넣었을 때 잡는지): "
        + (
            "통과"
            if meta.get("checker_selftest", {}).get("passed")
            else "**실패 — 아래 명부 외 인물 독립 대조 값을 믿지 말 것**"
        ),
    ]
    if title is not None:
        lines.append("")
    else:
        lines += [
            "- 초안 작성기: `TemplateDrafter` (LLM 호출 0회, 같은 입력 → 같은 초안). 검증: 제품의 "
            "`validate_patch`",
            "  (고정 규칙 재검사 + 기준 그래프에서 돈 리허설 재실행 + 무회귀 + 명부 외 인물 금지 + 출처 "
            "표기).",
            f"- 비용: $0 (유료 API 호출 없음). 소요: 조건별 합계 {_durations(summary)}"
            f" (작업자 {meta['workers']}개 병렬).",
            "",
            "## 이번 실행이 재는 것과 재지 않는 것",
            "",
            "- **재는 것**: 제품이 LLM을 끈 상태로 만든 수정안 초안이 제품 검증기의 재검사·리허설 재실행을 "
            "통과하는 비율,",
            "  초안이 명부 밖 인물을 넣는지(검증기 집계 + 이 스크립트가 명부 CSV를 따로 읽어 대조한 집계), "
            "어떤 끊김이",
            "  연산 있는 수정안을 받고 어떤 끊김이 사람 결정이 필요한 검토 메모로 남는지.",
            "- **재지 않는 것**: LLM이 쓴 문장(`LlmDrafter`)의 통과율. 신청서 문구의 'AI 수정 초안' 중 LLM "
            "문장 부분은",
            "  유료 실행 전까지 측정값이 없다(아래 추정·계획 참고). 템플릿 초안과 LLM 초안은 같은 후보 "
            "목록·같은 연산",
            "  컴파일러를 쓰지만, LLM이 다른 후보를 고르면 연산이 달라지므로 템플릿 통과율을 LLM 통과율로 "
            "쓰지 않는다.",
            "- 리허설은 그 그래프에서 제품이 자동 생성한 템플릿 훈련만 쓴다(사람·LLM이 만든 훈련 없음).",
            "  '과거 리허설'은 수정 전 그래프에서 돌린 같은 훈련을 수정 후 그래프에서 다시 돌린 것이다.",
            "- 봉인셋은 합성 조직이다. 실제 조직 런북에서의 통과율은 따로 재야 한다.",
            "",
            *_target_lines(summary),
        ]
    if meta.get("note"):
        lines += [f"- 메모: {meta['note']}", ""]
    for cond in meta["conditions"]:
        s = summary[cond]
        p, f, inj = s["patches"], s["findings"], s["injected"]
        lines += [
            f"## {CONDITION_LABEL[cond]}",
            "",
            f"사례 {s['cases']:,}건 (오류 {s['errors']}건), 끊김이 하나 이상 나온 사례 "
            f"{s['cases_with_findings']:,}건.",
            "",
            "| 지표 | 값 |",
            "|---|---|",
            f"| 초안 수 (연산 있는 수정안 + 검토 메모) | {p['drafts']:,} (수정안 {p['with_ops']:,} · 메모 "
            f"{p['memos']:,}) |",
            f"| **재검사 통과율** (통과 수정안 / 연산 있는 수정안) | **{_pct(p['pass_rate'])}** |",
            f"| 엄격 통과 (겨냥한 끊김 전부 해소 + 새 끊김 0건) | {_pct(p['strict_rate'])} |",
            f"| 메모까지 분모에 넣은 통과율 | {_pct(p['pass_over_all_drafts'])} |",
            f"| **명부 외 인물** — 검증기 집계 | **{p['validator_unknown_people']}건** |",
            f"| 명부 외 인물 — 명부 CSV 독립 대조 (연결된 인물 {p['linked_people_checked']:,}명) | "
            f"{p['independent_person_violations']}건 |",
            f"| 수정안이 새 인물 노드를 만든 경우 | {p['person_nodes_added']}건 |",
            f"| 문장에 명부에 없는 사람 이름(직함 붙은 이름) | {p['text_invented_names']}건 |",
            f"| 문장에 이름이 나온 명부 인물 중 재직이 아닌데 연결을 지우는 대상도 아닌 사람 | "
            f"{p['text_non_active_unlinked']}건 |",
            f"| 리허설 회귀 (통과 → 실패) | {p['rehearsal_regressions']}건 |",
            f"| 수정으로 실패 → 통과가 된 리허설 | {p['rehearsals_fixed']:,}건 |",
            f"| 재검사 독립 확인 불일치 (해소라던 끊김이 다시 돌려도 남음) | {p['recheck_mismatch']}건 |",
            "",
            "끊김 기준:",
            "",
            f"- 끊김 {f['findings']:,}건 중 연산 있는 수정안을 받은 것 {_pct(f['cover_patch'])}, 검토 메모로 "
            f"간 것"
            f" {_pct(f['cover_memo'])}, 초안 없음 {_pct(f['cover_none'])}.",
            f"- 통과한 수정안이 실제로 해소한 끊김: {_pct(f['resolved_by_approvable'])}.",
            f"- 주입 결함 자체(목표 서비스의 기대 규칙 끊김): 변이 사례 {inj['mutated_cases']:,}건 중 탐지된 "
            f"사례"
            f" {inj['cases_target_detected']:,}건, 탐지된 주입 결함 끊김 {inj['target_findings']:,}건 중 "
            f"통과 수정안으로 해소"
            f" {_pct(inj['target_resolved_by_approvable'])}, 검토 메모로 간 것 "
            f"{_pct(inj['target_cover_memo'])}.",
            "",
        ]
        du = s["decision_units"]
        boot = s["pass_rate_org_bootstrap_ci95"]
        lines += [
            "통과율의 불확실성:",
            "",
            f"- 같은 결정 단위(조직 × 결함 연산자 × 수정 유형)로 묶으면 {du['units']}개 중 전부 통과 "
            f"{_pct(du['all_pass'])}."
            f" 문체·배치에 따라 판정이 갈린 단위 {du['mixed']}개.",
            "- 한 조직·연산자의 렌더 6개(문체 3 × 배치 2)는 대개 같은 결정을 다시 묻는 것이라 수정안끼리 "
            "독립이 아니다."
            + (
                f" 조직 20곳을 복원추출한 부트스트랩({BOOTSTRAP_REPS:,}회, 시드 {SEED}) 95% 구간: "
                f"{boot[0] * 100:.1f}–{boot[1] * 100:.1f}%."
                if boot
                else ""
            ),
            "",
        ]
        if p["fail_reasons"]:
            lines += [
                "통과하지 못한 수정안의 사유(한 수정안에 여러 사유 가능): "
                + ", ".join(f"{_REASON_KO.get(k, k)} {v}건" for k, v in sorted(p["fail_reasons"].items())),
                "",
            ]
        if s["fail_breakdown"]:
            lines += ["통과하지 못한 수정안 내역(연산자 · 수정 유형 · 새로 생긴 높음 이상 끊김 규칙):", ""]
            lines += [f"- {_rules_ko(k)}: {v}건" for k, v in sorted(s["fail_breakdown"].items())] + [""]
        if s["strict_gap_breakdown"]:
            lines += [
                "통과했지만 '엄격 통과'가 아닌 수정안(높음 미만의 새 끊김이 남음. 검증기는 높음 이상 새 "
                "끊김만 막는다):",
                "",
            ]
            lines += [f"- {_rules_ko(k)}: {v}건" for k, v in sorted(s["strict_gap_breakdown"].items())] + [""]
        lines += [
            "### 결함 연산자별",
            "",
            "| 연산자 | 사례 | 수정안 | 통과율 | 메모 | 명부 외(검증기/독립) | 주입 결함 해소 |",
            "|---|---|---|---|---|---|---|",
        ]
        for op, g in s["by_operator"].items():
            gp = g["patches"]
            lines.append(
                f"| {op} | {g['cases']} | {gp['with_ops']} | {_short(gp['pass_rate'])} | {gp['memos']} | "
                f"{gp['validator_unknown_people']}/{gp['independent_person_violations']} | "
                f"{_short(g['targets']['resolved_by_approvable'])} |"
            )
        lines += [
            "",
            "### 문체별",
            "",
            "| 문체 | 사례 | 수정안 | 통과율 | 메모 | 명부 외(검증기/독립) | 주입 결함 해소 |",
            "|---|---|---|---|---|---|---|",
        ]
        for st, g in s["by_style"].items():
            gp = g["patches"]
            lines.append(
                f"| {st} | {g['cases']} | {gp['with_ops']} | {_short(gp['pass_rate'])} | {gp['memos']} | "
                f"{gp['validator_unknown_people']}/{gp['independent_person_violations']} | "
                f"{_short(g['targets']['resolved_by_approvable'])} |"
            )
        lines += [
            "",
            "### 수정 유형별",
            "",
            "| 유형 | 초안 | 수정안 | 통과율 | 메모 |",
            "|---|---|---|---|---|",
        ]
        for k, g in s["by_fix_kind"].items():
            lines.append(
                f"| {_KIND_KO.get(k, k)} | {g['drafts']} | {g['with_ops']} | {_short(g['pass_rate'])} | "
                f"{g['memos']} |"
            )
        if "loop" in s:
            lp = s["loop"]
            lines += [
                "",
                "### 보조: 리드가 통과 수정안을 전부 승인한다고 가정한 모의",
                "",
                f"통과 수정안 하나 승인 → 새 버전을 제품 방식으로 다시 판정 → 다시 초안·검증을 반복. 승인 "
                f"{lp['approvals']:,}회"
                f" (상한 {LOOP_MAX_APPROVALS}회에 걸린 사례 {lp['capped']}건). 끊김 "
                f"{lp['initial_findings']:,}건 → {lp['final_findings']:,}건."
                f" 끊김이 있던 사례 중 전부 닫힌 사례 {_pct(lp['cases_all_findings_closed'])}."
                f" 주입 결함이 탐지된 사례 중 끝에 그 결함이 닫힌 사례 {_pct(lp['target_closed_cases'])}.",
                "남은 끊김(규칙별): "
                + (", ".join(f"{_rules_ko(k)} {v}" for k, v in sorted(lp["final_rules"].items())) or "없음")
                + ". 남은 것은 검토 메모 유형(감지 설계·롤백 절차·연락 수단 등록·담당자 지정)과 통과하지 "
                "못한 수정안의 끊김이다.",
                "",
            ]
        else:
            lines.append("")
    est = summary.get("llm_estimate")
    if est:
        lines += [
            "## LLM 초안(LlmDrafter) 유료 실행 추정 — 실행하지 않음",
            "",
            "아래는 LlmDrafter가 보낼 요청을 오프라인에서 그대로 다시 만들어(호출은 보내지 않음) 센 값이다.",
            f"입력 토큰 = 글자 수 / {EST_CHARS_PER_TOKEN:g}(보수적), 출력 = min(max_tokens "
            f"{DRAFTER_MAX_TOKENS:,}, 보이는 출력 700 +"
            " 사고 토큰 허용치). 가격은 `jettae.evalkit.realdata.pricing`의 공식 가격표(정가). 프롬프트 캐시 "
            "할인 미반영.",
            "",
            "| 표본 | 사례 | 조건 | 호출 | 입력 토큰 | 출력 토큰 | 추정 비용 | 모든 호출이 재요청될 때 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for row in est["rows"]:
            lines.append(
                f"| {row['fraction'] * 100:g}% | {row['cases']:,} | {row['condition']} | {row['calls']:,} | "
                f"{row['input_tokens']:,} | "
                f"{row['output_tokens']:,} | ${row['usd']:.2f} | ${row['usd_if_every_call_repaired']:.2f} |"
            )
        lines += [
            "",
            f"모델 `{est['model']}`, effort `{est['effort']}`. 실행 명령과 사전 등록은 "
            f"`docs/eval/patch-recheck-plan.md`.",
            "",
        ]
    lines += [
        "## 해석할 때 주의",
        "",
        "- 통과율의 분모는 '연산 있는 수정안'이다. 검토 메모(감지 설계·롤백 절차처럼 사람이 정해야 하는 것, "
        "명부에서 후보를",
        "  못 찾은 것)는 제품이 일부러 승인 버튼을 주지 않는 초안이라 분모에서 뺐고, 메모 비율은 따로 "
        "적었다.",
        "- 검증기의 명부 외 인물 0건은 설계상 기대값이다(초안 작성기와 검증기가 같은 명부 대조 규칙을 쓴다). "
        "그래서 이 스크립트가",
        "  명부 CSV를 표준 라이브러리로 따로 읽어 수정안이 연결하는 사람을 다시 대조했다. 이 역시 같은 "
        "명부에서 나온 값이라",
        "  '명부 자체가 틀린 경우'는 잡지 못한다.",
        "- 문장 속 이름 검사는 '이름 + 직함' 꼴만 본다. 직함 없이 이름만 지어낸 문장은 이 검사로 못 "
        "잡는다(템플릿 문장은 코드가",
        "  명부 이름으로 만들기 때문에 해당 없음; LLM 문장은 제품이 별도로 막는다).",
        "- O(정답 그래프)는 추출 오류가 없는 상한이다. B 값이 제품의 LLM 끔 경로 값이다.",
        "- 검토 메모는 제품 설계상 승인 버튼이 없는 초안이다: 감지 신호·롤백 절차는 사람이 정해야 하고, 연락 "
        "수단은 명부에",
        "  등록된 것만 쓰며(지어내지 않음), 담당자가 사라진 서비스에 같은 팀 재직자를 찾지 못하면 CTO·대표 "
        "같은 최종 책임자를",
        "  1차 담당으로 넣지 않는다(새 단일 장애점이 되므로). 그래서 해당 결함 연산자는 '수정안 0건'으로 "
        "나온다.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- paid mode


#: Outcome of a call the run refused to send after an earlier request got no answer.
NOT_SENT: Final = "not_sent"


class _Stop:
    """Shared by every case of one paid run: the first request that got no answer, if any."""

    def __init__(self) -> None:
        self.reason: str | None = None


class _CallLog:
    """Pass-through LLMClient that remembers each call's outcome (ok / error type).

    A request that gets no answer (credit, key, rate limit, network: an LLMError that is not a
    truncated, invalid or refused answer) trips the run's ``stop``. Every later call of the run is
    then refused here without being sent (LLMDisabledError, so the drafter keeps the template;
    outcome ``not_sent``) and the run reports itself as stopped, instead of quietly turning the
    rest of the sample into template fallbacks.
    """

    def __init__(self, inner: LLMClient, stop: _Stop | None = None) -> None:
        self.inner = inner
        self.provider = inner.provider
        self.model = inner.model
        self.outcomes: list[str] = []
        self.stop = stop if stop is not None else _Stop()

    async def structured(self, **kwargs: Any) -> LLMResult[Any]:
        if self.stop.reason is not None:
            self.outcomes.append(NOT_SENT)
            raise LLMDisabledError(f"보내지 않음: 앞선 요청이 답을 받지 못함 ({self.stop.reason[:200]})")
        try:
            result = await self.inner.structured(**kwargs)
        except LLMError as exc:
            self.outcomes.append(type(exc).__name__)
            if not isinstance(exc, (LLMOutputError, LLMRefusalError)) and self.stop.reason is None:
                self.stop.reason = f"{type(exc).__name__}: {exc}"
            raise
        self.outcomes.append("ok")
        return result


async def run_llm(
    cases: Sequence[kit.EvalCase],
    conditions: Sequence[str],
    client: LLMClient,
    *,
    concurrency: int,
) -> tuple[list[dict[str, Any]], str | None]:
    """LlmDrafter on every case (paired with the template drafts of the same graph).

    ``client`` is any :class:`LLMClient` (the paid path passes the budgeted Claude client; tests a
    fake). Returns (records, stop reason) — a budget stop keeps what is done.
    """
    from jettae.evalkit.realdata.providers import gather_jobs

    stop = _Stop()

    async def one(case: kit.EvalCase, cond: str) -> dict[str, Any]:
        start = time.perf_counter()
        j = await asyncio.to_thread(judge, case, cond)
        plans = await asyncio.to_thread(_plans_for, j)
        log = _CallLog(client, stop)
        drafts = await LlmDrafter(log).adraft(
            j.ctx, j.findings, workspace_id=WORKSPACE, base_version=BASE_VERSION
        )
        if [p.finding_ids for p in drafts] != [pl.group.finding_ids for pl in plans]:
            raise RuntimeError("LLM 초안과 계획의 순서가 맞지 않습니다")
        patches = await asyncio.to_thread(
            validate_drafts,
            j.ctx,
            drafts,
            scenarios=j.scenarios,
            findings=j.findings,
            results=j.rehearsals,
            config=SIM,
        )
        rec = await asyncio.to_thread(case_record, case, cond, j, plans, patches)
        template = await asyncio.to_thread(run_template_case, case, cond, False, False)
        rec["llm_call_outcomes"] = log.outcomes
        rec["template_pair"] = [
            {"id": p["id"], "passed": p["passed"], "n_ops": p["n_ops"]} for p in template["patches"]
        ]
        rec["seconds"] = round(time.perf_counter() - start, 3)
        return rec

    done: list[dict[str, Any]] = []

    async def keep(case: kit.EvalCase, cond: str) -> dict[str, Any]:
        rec = await one(case, cond)
        done.append(rec)
        return rec

    jobs = [lambda c=c, k=k: keep(c, k) for c in cases for k in conditions]
    position = {c.case_id: i for i, c in enumerate(cases)}
    try:
        await gather_jobs(jobs, concurrency=concurrency)
        stopped = None
    except Exception as exc:  # BudgetExhausted or a failure: stop, keep the cases that finished
        stopped = f"{type(exc).__name__}: {exc}"
    if stopped is None and stop.reason is not None:
        stopped = f"요청이 답을 받지 못해 그 뒤 요청은 보내지 않음: {stop.reason}"
    done.sort(key=lambda r: (position[r["case_id"]], r["condition"]))
    return done, stopped


def summarize_llm(records: Sequence[dict[str, Any]], condition: str) -> dict[str, Any]:
    base = summarize(records, condition)
    recs = [r for r in records if r["condition"] == condition and r["error"] is None]
    asked: list[tuple[dict[str, Any], str | None]] = []
    for r in recs:
        sent = [p for p in r["patches"] if p.get("has_candidates") and p["n_ops"]]
        log = list(r.get("llm_call_outcomes", ()))
        asked += [(p, log[i] if i < len(log) else None) for i, p in enumerate(sent)]
    ai = [p for p, _ in asked if p["author"] == PatchAuthor.AI.value]
    outcomes = Counter(o for r in recs for o in r.get("llm_call_outcomes", ()))
    sent_calls = sum(n for o, n in outcomes.items() if o != NOT_SENT)
    pairs = [
        (p, t)
        for r in recs
        for p, t in zip(r["patches"], r.get("template_pair", ()), strict=False)
        if p["n_ops"] and t["n_ops"]
    ]
    base["llm"] = {
        "calls": sent_calls,
        "call_outcomes": dict(outcomes),
        "patches_sent_to_llm": len(asked),
        "ai_authored": len(ai),
        "fell_back_call_failed": sum(
            p["author"] == PatchAuthor.TEMPLATE.value and o not in (None, "ok", NOT_SENT) for p, o in asked
        ),
        "fell_back_rejected_by_drafter": sum(
            p["author"] == PatchAuthor.TEMPLATE.value and o == "ok" for p, o in asked
        ),
        "fell_back_not_sent": sum(o is None or o == NOT_SENT for _, o in asked),
        "ai_pass_rate": rate(sum(p["passed"] for p in ai), len(ai)),
        "ai_unknown_people": sum(len(p["unknown_people"]) for p in ai),
        "ai_independent_person_violations": sum(
            x["verdict"] in PERSON_VIOLATIONS for p in ai for x in p.get("people", ())
        ),
        "ai_text_invented_names": sum(len(p.get("text_invented", ())) for p in ai),
        "paired_same_verdict": sum(p["passed"] == t["passed"] for p, t in pairs),
        "paired": len(pairs),
        "paired_ai_pass_template_fail": sum(p["passed"] and not t["passed"] for p, t in pairs),
        "paired_ai_fail_template_pass": sum(t["passed"] and not p["passed"] for p, t in pairs),
    }
    return base


def paid_main(args: argparse.Namespace, cases: Sequence[kit.EvalCase], conditions: Sequence[str]) -> int:
    model = os.environ.get("JETTAE_ANTHROPIC_MODEL", "").strip() or DEFAULT_MODEL
    print(
        f"LLM 초안 비용 추정: 사례 {len(cases):,}건, 조건 {','.join(conditions)} (오프라인 계산, 호출 없음)",
        file=sys.stderr,
    )
    prepare_worker()
    requests = [q for c in cases for k in conditions for q in llm_requests(judge(c, k))]
    priced = price_requests(requests, model, args.effort)
    print(
        f"- 모델 {model}, effort {args.effort}: 호출 {priced['calls']:,}회, 입력 약 "
        f"{priced['input_tokens']:,} 토큰, "
        f"출력 최대 약 {priced['output_tokens']:,} 토큰 → 약 ${priced['usd']:.2f} (모든 호출이 재요청되면 "
        f"${priced['usd_if_every_call_repaired']:.2f})",
        file=sys.stderr,
    )
    if not args.yes:
        print(
            "유료 호출입니다. 위 추정치를 확인한 뒤 --yes를 붙여 다시 실행하세요. (아무 호출도 보내지 "
            "않았습니다)",
            file=sys.stderr,
        )
        return 2
    if args.max_usd is None or args.max_usd <= 0:
        print("--max-usd(예산 상한)를 지정해야 유료 실행을 시작합니다.", file=sys.stderr)
        return 2
    from jettae.evalkit.realdata.clients import budgeted
    from jettae.evalkit.realdata.keys import ANTHROPIC_KEY, load_api_key
    from jettae.evalkit.realdata.pricing import ModelRef
    from jettae.evalkit.realdata.providers import SpendTracker, claude_llm, new_anthropic_sdk

    key = load_api_key(ANTHROPIC_KEY)
    if key is None:
        print(
            f"{ANTHROPIC_KEY}가 없습니다(환경 변수 또는 .env). 유료 실행을 시작할 수 없습니다.",
            file=sys.stderr,
        )
        return 2
    tracker = SpendTracker(max_usd=args.max_usd)
    sdk = new_anthropic_sdk(key)
    del key
    client = budgeted(
        claude_llm(model, args.effort, sdk, tracker), ModelRef("anthropic", model), args.effort, tracker
    )

    async def go() -> tuple[list[dict[str, Any]], str | None]:
        try:
            return await run_llm(cases, conditions, client, concurrency=args.concurrency)
        finally:
            await sdk.close()

    records, stopped = asyncio.run(go())
    out = args.out or DEFAULT_LLM_OUT
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "meta": _meta(args, cases, conditions, workers=1)
        | {"model": model, "effort": args.effort, "stopped": stopped},
        "spend": tracker.report().model_dump(mode="json"),
        **{c: summarize_llm(records, c) for c in conditions},
    }
    _write(out, records, summary, render_llm_md(summary))
    print(f"실제 지출 ${tracker.spent_usd:.4f}. 결과: {out}", file=sys.stderr)
    return 0 if stopped is None else 1


# --------------------------------------------------------------------------- main


def checker_selftest(case: kit.EvalCase) -> dict[str, Any]:
    """Positive controls: the independent checks must flag an invented and a departed person."""
    rows = roster_rows(case)
    departed = next((r.name for r in rows if r.status != _ACTIVE_LABEL), None)
    active = next(r for r in rows if r.status == _ACTIVE_LABEL)
    invented = "홍길동" if all(r.name != "홍길동" for r in rows) else "남궁별"
    text = f"1차 담당 {invented} 대리에게 연락한다. {active.name} {active.title or '팀장'}이 승인한다."
    checks = {
        "invented_person_flagged": person_verdict(invented, set(), rows)[0] == "not_in_roster",
        "active_person_accepted": person_verdict(active.name, set(), rows)[0]
        in ("active", "active_namesake"),
        "invented_name_in_text_flagged": invented in text_people(text, rows, ())["invented"],
        "active_name_in_text_not_flagged": active.name not in text_people(text, rows, ())["invented"],
    }
    if departed is not None:
        checks["departed_person_flagged"] = person_verdict(departed, set(), rows)[0] == "not_active"
    return {"case": case.case_id, "passed": all(checks.values()), **checks}


def _git(*argv: str) -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", *argv], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError:
        return ""


def _meta(
    args: argparse.Namespace, cases: Sequence[kit.EvalCase], conditions: Sequence[str], *, workers: int
) -> dict[str, Any]:
    import subprocess

    src_clean = (
        subprocess.run(["git", "diff", "--quiet", "ccf3b4d", "--", "src/"], cwd=ROOT, check=False).returncode
        == 0
        and subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "src/"], cwd=ROOT, check=False).returncode
        == 0
    )
    return {
        "script": "scripts/eval_patch_recheck.py",
        "run_date": time.strftime("%Y-%m-%d"),
        "head": _git("rev-parse", "--short", "HEAD"),
        "frozen_commit": "ccf3b4d",
        "src_clean": src_clean,
        "seed": SEED,
        "orgs": ORGS,
        "generator": kit.GENERATOR_VERSION,
        "cases": len(cases),
        "sample": args.sample,
        "conditions": list(conditions),
        "workers": workers,
        "drafter": "LlmDrafter" if args.llm else "TemplateDrafter",
        "loop": not args.no_loop,
        "checker_selftest": checker_selftest(
            next((c for c in cases if operator_label(c) == "departed_contact"), cases[0])
        ),
    }


def _write(out: Path, records: Sequence[dict[str, Any]], summary: dict[str, Any], md: str | None) -> None:
    with (out / "results.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(
                json.dumps(
                    {k: v for k, v in r.items() if k != "llm_requests"}, ensure_ascii=False, sort_keys=True
                )
                + "\n"
            )
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if md is not None:
        (out / "report.md").write_text(md, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sample", type=float, help="결정적 표본 비율 (jettae eval run --sample과 같은 표본)")
    ap.add_argument("--conditions", default="B,O", help="B, O 중 쉼표 구분 (기본 B,O; --llm 기본 B)")
    ap.add_argument("--workers", type=int, default=None, help="무료 실행 병렬 작업자 수 (기본: 코어 수 - 2)")
    ap.add_argument("--no-loop", action="store_true", help="'전부 승인' 모의를 건너뜀")
    ap.add_argument("--out", type=Path, help=f"결과 폴더 (기본 {DEFAULT_OUT}, --llm은 {DEFAULT_LLM_OUT})")
    ap.add_argument("--llm", action="store_true", help="LlmDrafter 유료 실행 (추정 먼저, --yes 없으면 거부)")
    ap.add_argument("--yes", action="store_true", help="유료 실행 확인")
    ap.add_argument("--max-usd", type=float, default=None, help="유료 실행 예산 상한 (USD, 필수)")
    ap.add_argument("--effort", default="medium", choices=("low", "medium", "high", "xhigh", "max"))
    ap.add_argument("--concurrency", type=int, default=4, help="유료 실행 동시 사례 수")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    if args.sample is not None and not 0 < args.sample <= 1:
        ap.error("--sample은 0 초과 1 이하")
    raw = args.conditions if not (args.llm and args.conditions == "B,O") else "B"
    conditions = [c.strip().upper() for c in raw.split(",") if c.strip()]
    if not conditions or any(c not in CONDITIONS for c in conditions):
        ap.error(f"--conditions는 {CONDITIONS} 중에서: {raw}")
    cases = sealed_cases(args.sample)
    if args.llm:
        return paid_main(args, cases, conditions)
    workers = args.workers or default_workers()
    print(
        f"봉인셋 {len(cases):,}건 × 조건 {','.join(conditions)} — TemplateDrafter(LLM 끔), 작업자 "
        f"{workers}개",
        file=sys.stderr,
    )
    started = time.perf_counter()
    records = run_free(cases, conditions, workers=workers, loop=not args.no_loop)
    position = {c.case_id: i for i, c in enumerate(cases)}
    records.sort(key=lambda r: (position[r["case_id"]], r["condition"]))
    model = os.environ.get("JETTAE_ANTHROPIC_MODEL", "").strip() or DEFAULT_MODEL
    all_cases = cases if args.sample is None else sealed_cases(None)
    summary: dict[str, Any] = {
        "meta": _meta(args, cases, conditions, workers=workers)
        | {"wall_seconds": round(time.perf_counter() - started, 1)},
        **{c: summarize(records, c) for c in conditions},
    }
    if args.sample is None:
        summary["llm_estimate"] = {
            "model": model,
            "effort": args.effort,
            "rows": estimate_table(records, all_cases, model, args.effort, conditions),
        }
    out = args.out or DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    _write(out, records, summary, render_md(summary))
    for c in conditions:
        p = summary[c]["patches"]
        print(
            f"[{c}] 수정안 {p['with_ops']:,} · 메모 {p['memos']:,} · 통과 {_pct(p['pass_rate'])} · "
            f"명부 외(검증기) {p['validator_unknown_people']} · 명부 외(독립) "
            f"{p['independent_person_violations']}",
            file=sys.stderr,
        )
    print(f"결과: {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
