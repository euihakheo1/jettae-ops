"""숨은 규칙 복원(사람 수정 반영 KPI) 측정 — LLM 끔, 무료, 제품 코드(src/) 변경 없음.

Usage (repository root, product venv):
    python -X utf8 scripts/eval_hidden_rule.py [--orgs 20] [--seed 20261012]
        [--out C:/jettae/data/eval/hidden-rule] [--no-e2e] [--generated-at 2026-10-18]

What it measures (plan and honest reading: docs/eval/hidden-rule-plan.md):

* A hidden ground-truth rule (e.g. "결제 계열은 5분 무응답 시 팀장 상향, 롤백은 팀장 승인") is never
  shown to the product. It defines two specs of the same synthetic organisation (evalkit gold
  generator): the *gap* org (the rule's clauses are not written yet) and the *truth* org (written).
* A simulated team lead who knows the rule produces the synthetic record on SOME in-scope services
  ("기록 서비스") through the product's own service layer (``JettaeService.draft_patches`` /
  ``approve_patch`` / ``reject_patch`` — the functions the API routers call): approve drafts that move
  the graph toward the rule, reject the others. Arm ``rule_edit`` additionally lets the lead write
  the corrected patch himself (author HUMAN) — the domain supports it, the API does not expose it.
* Held-out rehearsal drills (never used while drafting or validating) are then replayed on the
  product's head graph and on the truth graph; executions are compared event by event.
* Scope over/under-application is counted on the canonical fact view (``evalkit.facts.graph_facts``).
* Unapproved changes: the version history is re-derived from approved patches only, and a set of
  attempts to push unapproved / rejected / stale / unvalidated patches is made; the head must not move.

The product does not learn from the record (no retraining; the drafter ranks candidates
deterministically). Equivalence on services WITHOUT a record can only come from the drafter's fixed
defaults coinciding with the rule — the script measures that separately and checks that the drafts
for those services are identical with and without the record.

No LLM is ever called: every service gets an LLM factory that returns ``NullLLM`` and runs with
``LlmMode.OFF``. No API key is read. Standard library plus the ``jettae`` package.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from jettae.check.registry import run_checks
from jettae.config import Settings
from jettae.domain import (
    AddEdge,
    AddNode,
    Attr,
    AttrKey,
    Edge,
    EdgeKind,
    Extractor,
    IncidentType,
    InputBundle,
    LlmMode,
    Node,
    NodeKind,
    OpsGraph,
    Patch,
    PatchAuthor,
    PatchStatus,
    PatchValidation,
    PersonStatus,
    Perturbation,
    PerturbationKind,
    Provenance,
    RehearsalResult,
    RemoveEdge,
    Scenario,
    ScenarioOrigin,
    SetAttr,
    SimConfig,
    SourceKind,
    edge_id,
    node_id,
)
from jettae.domain.ports import VersionConflictError
from jettae.evalkit.conditions import run_sync
from jettae.evalkit.facts import graph_facts, oracle_graph
from jettae.evalkit.gold import (
    DEFAULT_SEED,
    LEAD_TITLE,
    ROLE_CTO,
    Fact,
    GoldEscalationStep,
    GoldOrg,
    GoldService,
    GoldTrigger,
    action_key,
    generate_gold_orgs,
    gold_facts,
    trigger_key,
)
from jettae.evalkit.render import render
from jettae.improve import PatchStateError, TemplateDrafter, apply_ops, validate_patch
from jettae.ingest import load_bytes, load_catalog, load_policy, load_roster
from jettae.llm.base import NullLLM
from jettae.orchestrator.pipeline import build_context, gather_scenarios, missing_references, rehearse_all
from jettae.orchestrator.service import JettaeService
from jettae.rehearse.scenarios import template_scenarios
from jettae.store.memory import MemoryStore

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path("C:/jettae/data/eval/hidden-rule")
FROZEN_COMMIT = "ccf3b4d"
SCRIPT_VERSION = "hidden-rule-eval/1.0.0"
SIM = SimConfig()
RENDER_STYLE = "table"  # the rule parser's best style (B: field F1 99.97% on the sealed set)

ACTOR_SETUP = "평가-준비"
ACTOR_EDITOR = "평가-편집자"
ACTOR_LEAD = "평가-팀장(시뮬레이션)"
ACTOR_REHEARSE = "평가-리허설"

PAY_WORDS = ("결제", "정산", "과금")

# --------------------------------------------------------------------------- hidden rules


@dataclass(frozen=True)
class HiddenRule:
    """A ground-truth rule the product never sees: a scope plus the clauses it writes."""

    key: str
    title: str
    statement: str
    scope_label: str
    scope: Callable[[GoldOrg, GoldService], bool]
    #: minutes before the first responder's call escalates to the service team's lead (None: rule silent)
    escalate_minutes: int | None
    #: who approves the service's rollback/failover ("lead", "cto"); None: rule silent
    approvers: tuple[str, ...] | None


RULES: tuple[HiddenRule, ...] = (
    HiddenRule(
        key="pay-5m-lead",
        title="결제 계열: 5분 상향·팀장 승인",
        statement=(
            "결제·정산·과금 계열 서비스는 1차 담당이 5분 안에 응답하지 않으면 그 팀 팀장에게 상향하고, "
            "롤백·페일오버는 팀장 한 사람이 승인한다(대리 승인자를 두지 않음)."
        ),
        scope_label="서비스 이름에 '결제'·'정산'·'과금'이 들어간 서비스",
        scope=lambda org, s: any(w in s.name for w in PAY_WORDS),
        escalate_minutes=5,
        approvers=("lead",),
    ),
    HiddenRule(
        key="reg-10m-lead",
        title="규제 대상: 10분 상향",
        statement="신고 의무가 있는(규제 대상) 서비스는 1차 담당이 10분 안에 응답하지 않으면 그 팀 팀장에게 "
        "상향한다.",
        scope_label="규제 대상(침해사고 신고 의무가 있는) 서비스",
        scope=lambda org, s: s.regulated,
        escalate_minutes=10,
        approvers=None,
    ),
    HiddenRule(
        key="t1-two-approvers",
        title="1등급: 팀장·CTO 이중 승인",
        statement="1등급(핵심) 서비스의 롤백·페일오버는 그 팀 팀장과 CTO 두 사람이 승인한다.",
        scope_label="1등급 서비스",
        scope=lambda org, s: s.tier == 1,
        escalate_minutes=None,
        approvers=("lead", "cto"),
    ),
)

ARMS: tuple[str, ...] = ("none", "approve_all", "rule_review", "rule_edit")
ARM_KO: Mapping[str, str] = {
    "none": "기록 없음(규칙 적용 전 그래프 그대로)",
    "approve_all": "규칙 모르는 팀장: 검증 통과 초안 전부 승인",
    "rule_review": "규칙 아는 팀장: 기록 서비스의 초안을 승인·거절만(현재 API 경로)",
    "rule_edit": "규칙 아는 팀장: 거절한 초안 대신 직접 고친 수정안 승인(API에 없는 경로)",
}


def lead_of(org: GoldOrg, s: GoldService) -> str:
    return next(p.name for p in org.people if p.team == s.team and p.title == LEAD_TITLE)


def cto_of(org: GoldOrg) -> str:
    return org.roles[ROLE_CTO][0]


def write_rule(org: GoldOrg, rule: HiddenRule, services: Iterable[str], *, written: bool) -> GoldOrg:
    """``written=False``: the rule's clauses are absent (gap). ``written=True``: they are stated."""
    out = org
    for name in services:
        s = out.service(name)
        update: dict[str, object] = {}
        if rule.escalate_minutes is not None:
            update["escalation"] = (
                (GoldEscalationStep(target=lead_of(org, s), after_minutes=rule.escalate_minutes),)
                if written
                else ()
            )
        if rule.approvers is not None and s.action is not None:
            names = tuple(lead_of(org, s) if a == "lead" else cto_of(org) for a in rule.approvers)
            update["action"] = s.action.model_copy(update={"approvers": names if written else ()})
        out = out.replace_service(s.model_copy(update=update))
    return out


def neutral_edit(org: GoldOrg) -> GoldOrg:
    """A runbook edit no rule talks about: the first service's alarm delay changes by a few minutes."""
    s = org.services[0]
    if s.trigger is None:
        return org
    delay = s.trigger.delay_minutes
    new = 2 if delay is None else (delay + 2 if delay < 8 else 1)
    return org.replace_service(
        s.model_copy(update={"trigger": GoldTrigger(condition=s.trigger.condition, delay_minutes=new)})
    )


# --------------------------------------------------------------------------- one (org, rule) case


@dataclass
class Case:
    org: GoldOrg
    rule: HiddenRule
    in_scope: tuple[str, ...]
    record: tuple[str, ...]
    held_out: tuple[str, ...]
    gap: GoldOrg
    truth: GoldOrg
    roster: tuple[Any, ...]
    catalog: tuple[Any, ...]
    policy: Any
    g0: OpsGraph
    gt: OpsGraph
    seed: int
    keys: dict[str, str] = field(default_factory=dict)

    @property
    def case_id(self) -> str:
        return f"{self.org.org_id}:{self.rule.key}"

    def service_class(self, name: str | None) -> str:
        if name in self.record:
            return "record"
        if name in self.held_out:
            return "held_out"
        if name in self.in_scope:
            return "in_scope_other"
        return "out_of_scope" if name is not None else "org_wide"


def service_keys(org: GoldOrg) -> dict[str, str]:
    """Canonical fact names → the service they belong to (people by their service team)."""
    keys: dict[str, str] = {}
    for s in org.services:
        names = [s.name, *s.systems, s.team, trigger_key(s.name)]
        if s.action is not None:
            names.append(action_key(s.name, s.action.type))
        names += [p.name for p in org.people if p.team == s.team]
        for n in names:
            keys.setdefault(n, s.name)
    return keys


def fact_service(keys: Mapping[str, str], f: Fact) -> str | None:
    return keys.get(f.src) or keys.get(f.dst)


def build_case(org: GoldOrg, rule: HiddenRule, seed: int) -> Case | None:
    in_scope = tuple(sorted(s.name for s in org.services if rule.scope(org, s) and s.action is not None))
    if not in_scope:
        return None
    gap = write_rule(org, rule, in_scope, written=False)
    truth = write_rule(org, rule, in_scope, written=True)
    rendered = render(gap, style=RENDER_STYLE, position="front", focus_service=None, seed=seed)
    roster = tuple(load_roster(rendered.roster_csv.encode("utf-8")))
    catalog = tuple(load_catalog(rendered.catalog_yaml.encode("utf-8")))
    policy = load_policy(rendered.policy_yaml.encode("utf-8"))
    g0 = oracle_graph(gold_facts(gap), roster=roster, catalog=catalog)
    gt = oracle_graph(gold_facts(truth), roster=roster, catalog=catalog)
    return Case(
        org=org,
        rule=rule,
        in_scope=in_scope,
        record=in_scope[0::2],
        held_out=in_scope[1::2],
        gap=gap,
        truth=truth,
        roster=roster,
        catalog=catalog,
        policy=policy,
        g0=g0,
        gt=gt,
        seed=seed,
        keys=service_keys(org),
    )


# --------------------------------------------------------------------------- product session (LLM off)


class StepClock:
    """Deterministic clock: every reading is one second after the previous one."""

    def __init__(self) -> None:
        self.t = datetime(2026, 10, 18, 0, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        self.t += timedelta(seconds=1)
        return self.t


def _no_llm(*_args: object, **_kwargs: object) -> NullLLM:
    return NullLLM()


def _settings() -> Settings:
    # No .env file is read; the LLM factory below never builds a real client anyway.
    return Settings(_env_file=None, database_url="memory://", llm_mode=LlmMode.OFF, store_raw_documents=True)


def run(coro: Any) -> Any:
    """One persistent event loop per thread, opened with retries (the evalkit harness's helper:
    some Windows hosts refuse the loop's self-pipe socket pair now and then, WinError 10014)."""
    return run_sync(coro)


class Session:
    """One workspace on a fresh in-memory store, driven through ``JettaeService`` like the API does."""

    def __init__(self, label: str, case: Case) -> None:
        self.case = case
        self.clock = StepClock()
        self.svc = JettaeService(
            MemoryStore(clock=self.clock), _settings(), clock=self.clock, llm_factory=_no_llm
        )
        self.ws = self.svc.create_workspace(label, actor=ACTOR_SETUP).id

    # ---- setup
    def start_from_graph(self, graph: OpsGraph) -> None:
        """Inputs + the gap graph as version 1 (oracle path: no extraction noise)."""
        c = self.case
        self.svc.upload_inputs(
            self.ws, documents=(), roster=c.roster, catalog=c.catalog, policy=c.policy, actor=ACTOR_SETUP
        )
        self.svc.versions_service.commit_graph(
            self.ws, graph, actor=ACTOR_SETUP, note="숨은 규칙 평가: 규칙 적기 전"
        )
        run(self.svc.draft_patches(self.ws, llm_mode=LlmMode.OFF, actor=ACTOR_EDITOR))

    def start_from_text(self, org: GoldOrg) -> None:
        """Rendered runbook + roster/catalog/policy, diagnosed by the product (rule parser, LLM off)."""
        c = self.case
        doc = runbook_document(org, c.seed)
        bundle = InputBundle(documents=(doc,), roster=c.roster, catalog=c.catalog, policy=c.policy)
        self.svc.load_bundle(self.ws, bundle, actor=ACTOR_SETUP)
        run(self.svc.diagnose(self.ws, llm_mode=LlmMode.OFF, actor=ACTOR_SETUP))

    # ---- reads
    def head(self) -> tuple[Any, OpsGraph]:
        return self.svc.graph(self.ws)

    def ctx(self, graph: OpsGraph) -> Any:
        c = self.case
        return build_context(graph, roster=c.roster, catalog=c.catalog, policy=c.policy)

    def pending(self) -> list[Patch]:
        head, _graph = self.head()
        return [
            p
            for p in self.svc.patches(self.ws)
            if p.status is PatchStatus.VALIDATED and p.base_version == head.number
        ]

    def head_state(self) -> tuple[int, str, int]:
        head, _g = self.head()
        return head.number, head.graph_hash, len(self.svc.versions(self.ws))


def runbook_document(org: GoldOrg, seed: int) -> Any:
    rendered = render(org, style=RENDER_STYLE, position="front", focus_service=None, seed=seed)
    return load_bytes("runbook.md", rendered.text.encode("utf-8"))


# --------------------------------------------------------------------------- patch helpers


def change_signature(patch: Patch) -> frozenset[str]:
    """What a patch changes, without who wrote it (provenance) — equal content ⇒ equal signature."""
    out: set[str] = set()
    for op in patch.ops:
        if isinstance(op, AddNode):
            out.add(f"add_node {op.node.id}")
        elif isinstance(op, AddEdge):
            attrs = {k: a.value for k, a in sorted(op.edge.attrs.items())}
            out.add(f"add_edge {op.edge.id} {json.dumps(attrs, ensure_ascii=False, sort_keys=True)}")
        elif isinstance(op, RemoveEdge):
            out.add(f"remove_edge {op.edge_id}")
        elif isinstance(op, SetAttr):
            value = None if op.value is None else op.value.value
            out.add(f"set_attr {op.target_id} {op.key}={json.dumps(value, ensure_ascii=False)}")
        else:
            out.add(f"remove_node {op.node_id}")
    return frozenset(out)


def fact_delta(graph: OpsGraph, ops: Sequence[Any]) -> tuple[frozenset[Fact], frozenset[Fact]]:
    after, _applied = apply_ops(graph, ops)
    before_f, after_f = graph_facts(graph), graph_facts(after)
    return after_f - before_f, before_f - after_f


def patch_service(patch: Patch, findings_by_id: Mapping[str, Any], graph: OpsGraph) -> str | None:
    sids = {findings_by_id[f].service_id for f in patch.finding_ids if f in findings_by_id}
    sids.discard(None)
    if len(sids) != 1:
        return None
    sid = next(iter(sids))
    node = graph.nodes.get(sid)
    return node.name if node is not None else None


def person_id(graph: OpsGraph, name: str) -> str | None:
    nid = node_id(NodeKind.PERSON, name)
    if nid in graph.nodes:
        return nid
    return next((n.id for n in graph.nodes.values() if n.kind is NodeKind.PERSON and n.name == name), None)


def action_node(graph: OpsGraph, service_name: str) -> str | None:
    sid = node_id(NodeKind.SERVICE, service_name)
    for e in graph.edges.values():
        if e.kind is EdgeKind.MITIGATED_BY and e.src == sid:
            return e.dst
    return None


def edge_for_fact(graph: OpsGraph, f: Fact) -> Edge | None:
    """The graph edge stating fact ``f`` (people by name, actions by their service)."""
    for e in graph.edges.values():
        if e.kind is not f.kind:
            continue
        if f.kind is EdgeKind.ESCALATES_TO:
            src, dst = graph.nodes.get(e.src), graph.nodes.get(e.dst)
            if src is not None and dst is not None and (src.name, dst.name) == (f.src, f.dst):
                return e
        elif f.kind is EdgeKind.APPROVED_BY:
            service = f.src.split("#", 1)[0]
            dst = graph.nodes.get(e.dst)
            if e.src == action_node(graph, service) and dst is not None and dst.name == f.dst:
                return e
    return None


# --------------------------------------------------------------------------- the simulated lead


@dataclass
class RecordLog:
    approvals: list[dict[str, Any]] = field(default_factory=list)
    rejections: list[dict[str, Any]] = field(default_factory=list)
    reproposed_after_rejection: int = 0
    human_patches: list[dict[str, Any]] = field(default_factory=list)
    rounds: int = 0


def acceptable(
    graph: OpsGraph, patch: Patch, truth_facts: frozenset[Fact]
) -> tuple[bool, list[str], list[str]]:
    added, removed = fact_delta(graph, patch.ops)
    ok = bool(added or removed) and added <= truth_facts and not (removed & truth_facts)
    return ok, sorted(map(fact_text, added)), sorted(map(fact_text, removed))


def fact_text(f: Fact) -> str:
    attrs = ",".join(f"{k}={v}" for k, v in f.attrs)
    return f"{f.kind.value}:{f.src}->{f.dst}" + (f"({attrs})" if attrs else "")


def run_records(session: Session, arm: str, truth_graph: OpsGraph) -> RecordLog:  # noqa: PLR0912 - one loop, the lead's rules
    """The record: a lead who knows the hidden rule decides on drafts (only the review set)."""
    case = session.case
    log = RecordLog()
    if arm == "none":
        return log
    truth_facts = graph_facts(truth_graph)
    review = (
        set(case.in_scope) | {s.name for s in case.org.services} if arm == "approve_all" else set(case.record)
    )
    rejected: set[frozenset[str]] = set()
    human_tried: set[tuple[str, int]] = set()
    for _round in range(60):
        log.rounds += 1
        head, graph = session.head()
        ctx = session.ctx(graph)
        findings = run_checks(ctx)
        by_id = {f.id: f for f in findings}
        moved = False
        for patch in session.pending():
            service = patch_service(patch, by_id, graph)
            if service not in review:
                continue
            sig = change_signature(patch)
            ok, added, removed = acceptable(graph, patch, truth_facts)
            if arm == "approve_all" or ok:
                session.svc.approve_patch(patch.id, actor=ACTOR_LEAD, role="lead")
                log.approvals.append(
                    {
                        "patch_id": patch.id,
                        "author": patch.author.value,
                        "service": service,
                        "version": head.number + 1,
                        "text": patch.runbook_text,
                        "added": added,
                        "removed": removed,
                    }
                )
                moved = True
                break
            if sig in rejected:
                log.reproposed_after_rejection += 1
            rejected.add(sig)
            session.svc.reject_patch(
                patch.id, actor=ACTOR_LEAD, role="lead", reason=f"숨은 규칙과 다름: {case.rule.statement}"
            )
            log.rejections.append(
                {
                    "patch_id": patch.id,
                    "service": service,
                    "version": head.number,
                    "text": patch.runbook_text,
                    "added": added,
                    "removed": removed,
                }
            )
        if moved:
            continue
        if arm == "rule_edit":
            for service in case.record:
                if (service, head.number) in human_tried:
                    continue
                human_tried.add((service, head.number))
                outcome = human_fix(session, service, truth_facts, findings)
                if outcome is None:
                    continue
                log.human_patches.append(outcome)
                if outcome["approved"]:
                    moved = True
                    break
        if not moved:
            break
    return log


def human_fix(  # noqa: PLR0912 - one branch per fact kind
    session: Session, service: str, truth_facts: frozenset[Fact], findings: Sequence[Any]
) -> dict[str, Any] | None:
    """The lead writes the rule's remaining change for ``service`` as a HUMAN patch and approves it.

    The domain model supports hand-written patches (PatchAuthor.HUMAN, validated like any other);
    the API has no endpoint for them, so the patch is put in the store directly and then approved
    with the product's own approve function. Returns None when nothing is left to change.
    """
    case = session.case
    head, graph = session.head()
    ctx = session.ctx(graph)
    head_facts = graph_facts(graph)
    mine = lambda fs: {f for f in fs if fact_service(case.keys, f) == service}  # noqa: E731
    to_add = sorted(mine(truth_facts) - mine(head_facts))
    to_remove = sorted(mine(head_facts) - mine(truth_facts))
    if not to_add and not to_remove:
        return None
    plan: list[tuple[str, Any]] = []
    unsupported: list[str] = []
    for f in to_remove:
        e = edge_for_fact(graph, f)
        if e is None:
            unsupported.append(fact_text(f))
        else:
            plan.append(("remove", e))
    for f in to_add:
        if f.kind is EdgeKind.ESCALATES_TO:
            src, dst = person_id(graph, f.src), person_id(graph, f.dst)
            attrs = dict(f.attrs)
        elif f.kind is EdgeKind.APPROVED_BY:
            src, dst = action_node(graph, f.src.split("#", 1)[0]), person_id(graph, f.dst)
            attrs = {}
        else:
            src = dst = None
            attrs = {}
        if src is None or dst is None:
            unsupported.append(fact_text(f))
            continue
        plan.append(("add", (f.kind, src, dst, attrs)))
    sid = node_id(NodeKind.SERVICE, service)
    claimed = tuple(sorted(f.id for f in findings if f.service_id == sid))
    digest_src = json.dumps(
        [
            session.ws,
            head.number,
            claimed,
            [p[0] + ":" + (p[1].id if p[0] == "remove" else repr(p[1])) for p in plan],
        ],
        ensure_ascii=False,
    )
    pid = "patch-human-" + hashlib.sha256(digest_src.encode("utf-8")).hexdigest()[:24]
    prov = (Provenance(source=SourceKind.PATCH, extractor=Extractor.HUMAN, ref=pid),)
    ops: list[Any] = []
    for kind, item in plan:
        if kind == "remove":
            ops.append(RemoveEdge(edge_id=item.id, snapshot=item))
        else:
            ekind, src, dst, attrs = item
            ops.append(
                AddEdge(
                    edge=Edge(
                        id=edge_id(ekind, src, dst),
                        kind=ekind,
                        src=src,
                        dst=dst,
                        attrs={k: Attr(value=v, source=SourceKind.PATCH) for k, v in attrs.items()},
                        prov=prov,
                    )
                )
            )
    record = {
        "service": service,
        "patch_id": pid,
        "version": head.number,
        "unsupported": unsupported,
        "changes": [fact_text(f) for f in to_add] + [f"삭제 {fact_text(f)}" for f in to_remove],
    }
    if not ops:
        return {
            **record,
            "approved": False,
            "passed": False,
            "notes": ["규칙 변경을 그래프 연산으로 옮길 수 없음"],
        }
    patch = Patch(
        id=pid,
        workspace_id=session.ws,
        base_version=head.number,
        finding_ids=claimed,
        ops=tuple(ops),
        runbook_text=f"{service}: {case.rule.statement}",
        rationale="숨은 규칙에 맞춰 팀장이 직접 고친 수정안(평가용 시뮬레이션, API에 없는 경로).",
        author=PatchAuthor.HUMAN,
    )
    scenarios, _notes = gather_scenarios(ctx)
    verdict = validate_patch(ctx, patch, scenarios=scenarios, config=SIM, baseline_findings=findings)
    record.update(
        passed=verdict.passed, notes=list(verdict.notes[:3]), unknown_people=list(verdict.unknown_people)
    )
    if not verdict.passed:
        session.svc.store.patches.save_patch(
            patch.model_copy(
                update={
                    "status": PatchStatus.REJECTED_BY_VALIDATOR,
                    "validation": verdict,
                    "created_at": session.clock(),
                }
            )
        )
        return {**record, "approved": False}
    session.svc.store.patches.save_patch(
        patch.model_copy(
            update={"status": PatchStatus.VALIDATED, "validation": verdict, "created_at": session.clock()}
        )
    )
    session.svc.approve_patch(pid, actor=ACTOR_LEAD, role="lead")
    return {**record, "approved": True}


# --------------------------------------------------------------------------- held-out drills


DRILL_KO: Mapping[str, str] = {
    "owner_absent_20m": "1차 담당 20분 부재",
    "owner_absent_tight": "1차 담당 부재 + 목표 30분",
    "lead_absent": "팀장 부재",
    "owner_absent_chat_down": "1차 담당 부재 + 사내 메신저 다운",
    "owner_lead_absent": "1차 담당·팀장 동시 부재",
    "lead_cto_absent": "팀장·CTO 동시 부재",
    "breach_owner_absent": "침해사고 + 1차 담당 부재",
}


def _absent(pid: str, end: float | None = None) -> Perturbation:
    return Perturbation(
        kind=PerturbationKind.PERSON_UNAVAILABLE, target_id=pid, start_minute=0.0, end_minute=end
    )


def held_out_drills(case: Case) -> list[Scenario]:
    """Hand-written-style drills (origin HUMAN) never shown to the product before measurement."""
    org = case.truth
    cto = node_id(NodeKind.PERSON, cto_of(org))
    chat = node_id(NodeKind.CHANNEL, org.chat_medium.value)
    out: list[Scenario] = []
    for s in org.services:
        if s.owner is None:
            continue
        sid = node_id(NodeKind.SERVICE, s.name)
        owner = node_id(NodeKind.PERSON, s.owner)
        lead = node_id(NodeKind.PERSON, lead_of(org, s))
        fams: list[tuple[str, tuple[Perturbation, ...], IncidentType, float | None]] = [
            ("owner_absent_20m", (_absent(owner, 20.0),), IncidentType.OUTAGE, None),
            ("owner_absent_tight", (_absent(owner),), IncidentType.OUTAGE, 30.0),
            ("lead_absent", (_absent(lead),), IncidentType.OUTAGE, None),
            (
                "owner_absent_chat_down",
                (_absent(owner), Perturbation(kind=PerturbationKind.CHANNEL_DOWN, target_id=chat)),
                IncidentType.OUTAGE,
                None,
            ),
            ("owner_lead_absent", (_absent(owner), _absent(lead)), IncidentType.OUTAGE, None),
            ("lead_cto_absent", (_absent(lead), _absent(cto)), IncidentType.OUTAGE, None),
        ]
        if s.regulated:
            fams.append(("breach_owner_absent", (_absent(owner),), IncidentType.BREACH, None))
        for key, perts, incident, deadline in fams:
            out.append(
                Scenario(
                    id=f"hold:{sid}:{key}",
                    title=f"{s.name} · {DRILL_KO[key]}",
                    description="숨은 규칙 평가용 보류 시나리오(기록·검증에 쓰지 않음).",
                    service_id=sid,
                    incident_type=incident,
                    perturbations=perts,
                    deadline_minutes=deadline,
                    origin=ScenarioOrigin.HUMAN,
                )
            )
    return out


def truth_templates(case: Case, truth_graph: OpsGraph) -> list[Scenario]:
    """The product's own template families, generated on the TRUTH graph (re-id'd, kept apart)."""
    ctx_t = build_context(truth_graph, roster=case.roster, catalog=case.catalog, policy=case.policy)
    ids = [node_id(NodeKind.SERVICE, s.name) for s in case.truth.services]
    return [t.model_copy(update={"id": f"gt-{t.id}"}) for t in template_scenarios(ctx_t, service_ids=ids)]


def signature(r: RehearsalResult) -> tuple[Any, ...]:
    return (
        r.outcome.value,
        r.time_to_mitigate,
        r.deadline_minutes,
        None if r.failed_element is None else str(r.failed_element),
        r.failed_at_id,
        tuple((e.t, e.kind.value, e.actor_id, e.target_id, e.edge_id) for e in r.events),
    )


def first_difference(a: RehearsalResult, b: RehearsalResult) -> str:
    ea = [(e.t, e.kind.value, e.actor_id, e.target_id) for e in a.events]
    eb = [(e.t, e.kind.value, e.actor_id, e.target_id) for e in b.events]
    for i, (x, y) in enumerate(zip(ea, eb, strict=False)):
        if x != y:
            return f"{i + 1}번째 사건: 제품 {x[0]:g}분 {x[1]} / 정답 {y[0]:g}분 {y[1]}"
    if len(ea) != len(eb):
        return f"사건 수 다름: 제품 {len(ea)} / 정답 {len(eb)}"
    return (
        f"결과 다름: 제품 {a.outcome.value} {a.time_to_mitigate} / "
        f"정답 {b.outcome.value} {b.time_to_mitigate}"
    )


def replay_on(case: Case, graph: OpsGraph, scenarios: Sequence[Scenario]) -> dict[str, RehearsalResult]:
    ctx = build_context(graph, roster=case.roster, catalog=case.catalog, policy=case.policy)
    return {r.scenario_id: r for r in rehearse_all(graph, scenarios, ctx=ctx, config=SIM)}


def replay_via_service(session: Session, scenarios: Sequence[Scenario]) -> dict[str, RehearsalResult]:
    """The "next rehearsal" exactly as the product runs it: stored drills, ``JettaeService.rehearse``."""
    for s in scenarios:
        session.svc.store.rehearsals.save_scenario(session.ws, s)
    results = run(
        session.svc.rehearse(session.ws, scenario_ids=[s.id for s in scenarios], actor=ACTOR_REHEARSE)
    )
    return {r.scenario_id: r for r in results}


def compare(
    case: Case,
    scenarios: Sequence[Scenario],
    head: Mapping[str, RehearsalResult],
    truth: Mapping[str, RehearsalResult],
    *,
    graph: OpsGraph,
) -> list[dict[str, Any]]:
    rows = []
    for s in scenarios:
        a, b = head[s.id], truth[s.id]
        service = graph.nodes[s.service_id].name if s.service_id in graph.nodes else None
        strict = signature(a) == signature(b)
        rows.append(
            {
                "scenario_id": s.id,
                "family": s.id.split(":")[-1] if s.id.startswith("hold:") else "tpl:" + s.id.split(":")[-1],
                "kind": "held_out_drill" if s.id.startswith("hold:") else "truth_template",
                "service": service,
                "service_class": case.service_class(service),
                "strict": strict,
                "outcome": a.outcome is b.outcome,
                "product_outcome": a.outcome.value,
                "truth_outcome": b.outcome.value,
                "product_ttm": a.time_to_mitigate,
                "truth_ttm": b.time_to_mitigate,
                "difference": None if strict else first_difference(a, b),
            }
        )
    return rows


# --------------------------------------------------------------------------- scope audit (fact level)


def signed(f0: frozenset[Fact], f1: frozenset[Fact]) -> set[tuple[str, Fact]]:
    return {("+", f) for f in f1 - f0} | {("-", f) for f in f0 - f1}


def node_attr_changes(g0: OpsGraph, g1: OpsGraph) -> list[str]:
    out: list[str] = []
    for nid in sorted(g0.nodes.keys() & g1.nodes.keys()):
        a = {k: v.value for k, v in g0.nodes[nid].attrs.items()}
        b = {k: v.value for k, v in g1.nodes[nid].attrs.items()}
        out.extend(
            f"{nid}.{k}: {a.get(k)!r} → {b.get(k)!r}"
            for k in sorted(a.keys() | b.keys())
            if a.get(k) != b.get(k)
        )
    out += [f"노드 추가 {n}" for n in sorted(g1.nodes.keys() - g0.nodes.keys())]
    out += [f"노드 삭제 {n}" for n in sorted(g0.nodes.keys() - g1.nodes.keys())]
    return out


def scope_audit(case: Case, g0: OpsGraph, gt: OpsGraph, head: OpsGraph) -> dict[str, Any]:
    f0, ft, fh = graph_facts(g0), graph_facts(gt), graph_facts(head)
    required, made = signed(f0, ft), signed(f0, fh)
    per: dict[str, dict[str, Any]] = {}
    for s in case.org.services:
        r = {x for x in required if fact_service(case.keys, x[1]) == s.name}
        m = {x for x in made if fact_service(case.keys, x[1]) == s.name}
        correct, under, over = r & m, r - m, m - r
        wrong_value = {
            (o[1].kind, o[1].src, o[1].dst)
            for o in over
            if o[0] == "+"
            and any(
                u[0] == "+" and (u[1].kind, u[1].src, u[1].dst) == (o[1].kind, o[1].src, o[1].dst)
                for u in under
            )
        }
        cls = case.service_class(s.name)
        if cls == "out_of_scope":
            status = "변경 없음" if not m else "범위 밖 변경"
        elif not r:
            status = "규칙 변경 없음"
        elif not under and not over:
            status = "정확 반영"
        elif correct:
            status = "일부 반영"
        else:
            status = "미반영" if not over else "다르게 반영"
        per[s.name] = {
            "class": cls,
            "required": len(r),
            "made": len(m),
            "correct": len(correct),
            "under": len(under),
            "over": len(over),
            "wrong_value": len(wrong_value),
            "status": status,
            "under_facts": sorted(f"{x[0]}{fact_text(x[1])}" for x in under),
            "over_facts": sorted(f"{x[0]}{fact_text(x[1])}" for x in over),
        }
    unattributed = sorted(f"{x[0]}{fact_text(x[1])}" for x in made if fact_service(case.keys, x[1]) is None)
    return {
        "services": per,
        "unattributed_changes": unattributed,
        "node_attr_changes": node_attr_changes(g0, head),
    }


# --------------------------------------------------------------------------- unapproved-change audit


def history_audit(session: Session, g0: OpsGraph) -> dict[str, Any]:
    """Re-derive the head from approved patches only; every version must be explained by one."""
    versions = session.svc.versions(session.ws)
    graph = g0
    violations: list[str] = []
    approved_ids: list[str] = []
    if not versions or versions[0].graph_hash != g0.content_hash():
        violations.append("버전 1이 시작 그래프가 아님")
    for v in versions[1:]:
        if v.revert_of is not None:
            graph = session.svc.versions_service.graph(session.ws, v.revert_of)
            continue
        if v.patch_id is None:
            violations.append(f"버전 {v.number}: 수정안 없이 생김")
            graph = session.svc.versions_service.graph(session.ws, v.number)
            continue
        p = session.svc.get_patch(v.patch_id)
        if p.status is not PatchStatus.APPROVED or not p.decided_by:
            violations.append(f"버전 {v.number}: 승인되지 않은 수정안 {p.id}({p.status.value})")
        approved_ids.append(p.id)
        graph, _ = apply_ops(graph, p.ops)
        if graph.content_hash() != v.graph_hash:
            violations.append(f"버전 {v.number}: 승인된 연산으로 다시 만든 그래프와 다름")
    head, head_graph = session.head()
    if graph.content_hash() != head.graph_hash:
        violations.append("최신 버전이 승인 이력으로 설명되지 않음")
    # facts only a rejected / pending / validator-failed patch would add must not be in the head
    approved_facts: set[Fact] = set()
    replay = g0
    for v in versions[1:]:
        if v.patch_id:
            p = session.svc.get_patch(v.patch_id)
            added, _ = fact_delta(replay, p.ops)
            approved_facts |= added
            replay, _ = apply_ops(replay, p.ops)
    head_new = graph_facts(head_graph) - graph_facts(g0)
    leaked: list[str] = []
    statuses = Counter()
    for p in session.svc.patches(session.ws):
        statuses[p.status.value] += 1
        if p.status is PatchStatus.APPROVED or not p.ops:
            continue
        base = session.svc.versions_service.graph(session.ws, p.base_version)
        try:
            added, _ = fact_delta(base, p.ops)
        except Exception:  # a patch that no longer applies adds nothing
            continue
        leaked.extend(
            f"{p.id}({p.status.value}): {fact_text(f)}" for f in (added - approved_facts) & head_new
        )
    return {
        "versions": len(versions),
        "approved_patches": approved_ids,
        "violations": violations,
        "unapproved_facts_in_head": sorted(set(leaked)),
        "patch_statuses": dict(statuses),
    }


# --------------------------------------------------------------------------- gate attempts


def _attempt(
    session: Session, name: str, expect: str, fn: Callable[[], Any], *, may_move: bool = False
) -> dict[str, Any]:
    before = session.head_state()
    error: str | None = None
    kind = "없음"
    try:
        fn()
    except (PermissionError, PatchStateError, VersionConflictError, ValueError) as exc:
        kind = type(exc).__name__
        error = str(exc).splitlines()[0][:160]
    after = session.head_state()
    moved = after != before
    return {
        "attempt": name,
        "expected": expect,
        "error": kind,
        "message": error,
        "head_moved": moved,
        "unapproved_change": moved and not may_move,
        "refused": error is not None,
    }


def gate_attempts(case: Case) -> list[dict[str, Any]]:
    """Try to get unapproved, rejected, stale or unvalidated changes into the graph."""
    s = Session(f"{case.case_id}:gate", case)
    s.start_from_graph(case.g0)
    pend = s.pending()
    out: list[dict[str, Any]] = []
    if len(pend) < 1:
        return out
    a = pend[0]
    out.append(
        _attempt(
            s,
            "편집자가 승인",
            "PermissionError",
            lambda: s.svc.approve_patch(a.id, actor=ACTOR_EDITOR, role="editor"),
        )
    )
    out.append(
        _attempt(
            s,
            "열람자가 승인",
            "PermissionError",
            lambda: s.svc.approve_patch(a.id, actor="평가-열람자", role="viewer"),
        )
    )
    out.append(
        _attempt(
            s,
            "행위자 빈칸으로 승인",
            "ValueError",
            lambda: s.svc.approve_patch(a.id, actor="  ", role="lead"),
        )
    )
    # a draft that never went through validation (same content, its own id)
    raw = a.model_copy(update={"id": f"{a.id}-unvalidated", "status": PatchStatus.DRAFT, "validation": None})
    s.svc.store.patches.save_patch(raw)
    out.append(
        _attempt(
            s,
            "검증 전 초안 승인",
            "PatchStateError",
            lambda: s.svc.approve_patch(raw.id, actor=ACTOR_LEAD, role="lead"),
        )
    )
    # a hand-written patch naming somebody outside the roster: the validator refuses it
    head, graph = s.head()
    ctx = s.ctx(graph)
    findings = run_checks(ctx)
    outsider = _outsider_patch(s, head.number, graph, findings)
    verdict = validate_patch(
        ctx, outsider, scenarios=gather_scenarios(ctx)[0], config=SIM, baseline_findings=findings
    )
    status = PatchStatus.VALIDATED if verdict.passed else PatchStatus.REJECTED_BY_VALIDATOR
    s.svc.store.patches.save_patch(outsider.model_copy(update={"status": status, "validation": verdict}))
    row = _attempt(
        s,
        "명부 외 인물 수정안 승인(검증 실패분)",
        "PatchStateError",
        lambda: s.svc.approve_patch(outsider.id, actor=ACTOR_LEAD, role="lead"),
    )
    row["validator_unknown_people"] = list(verdict.unknown_people)
    row["validator_passed"] = verdict.passed
    out.append(row)
    # a lead rejects a draft (a stored copy of a real draft, so the original can go stale below)
    b = a.model_copy(update={"id": f"{a.id}-to-reject"})
    s.svc.store.patches.save_patch(b)
    s.svc.reject_patch(b.id, actor=ACTOR_LEAD, role="lead", reason="평가: 거절")
    out.append(
        _attempt(
            s,
            "거절된 수정안 승인",
            "PatchStateError",
            lambda: s.svc.approve_patch(b.id, actor=ACTOR_LEAD, role="lead"),
        )
    )
    # a legitimate approval moves the head; drafts of the old head become stale
    stale_candidates = [p for p in s.pending() if p.id != a.id]
    if not stale_candidates:
        # one draft only: keep a copy of it pending, it goes stale when the original is approved
        twin = a.model_copy(update={"id": f"{a.id}-twin"})
        s.svc.store.patches.save_patch(twin)
        stale_candidates = [twin]
    out.append(
        _attempt(
            s,
            "검증 통과 수정안을 팀장이 승인(정상 경로)",
            "없음",
            lambda: s.svc.approve_patch(a.id, actor=ACTOR_LEAD, role="lead"),
            may_move=True,
        )
    )
    out.append(
        _attempt(
            s,
            "이미 승인된 수정안 다시 승인",
            "PatchStateError",
            lambda: s.svc.approve_patch(a.id, actor=ACTOR_LEAD, role="lead"),
        )
    )
    if stale_candidates:
        c = stale_candidates[0]
        out.append(
            _attempt(
                s,
                "옛 버전 기준 수정안(무효 처리됨) 승인",
                "VersionConflictError",
                lambda: s.svc.approve_patch(c.id, actor=ACTOR_LEAD, role="lead"),
            )
        )
        forged = s.svc.get_patch(c.id).model_copy(
            update={"id": f"{c.id}-stale-copy", "status": PatchStatus.VALIDATED}
        )
        s.svc.store.patches.save_patch(forged)
        out.append(
            _attempt(
                s,
                "옛 버전 기준 수정안(무효 표시 전 사본) 승인",
                "VersionConflictError",
                lambda: s.svc.approve_patch(forged.id, actor=ACTOR_LEAD, role="lead"),
            )
        )
    out.append(
        _attempt(
            s,
            "편집자가 되돌리기",
            "PermissionError",
            lambda: s.svc.revert(s.ws, 1, actor=ACTOR_EDITOR, role="editor"),
        )
    )
    out.append(
        _attempt(
            s,
            "팀장이 버전 1로 되돌리기(정상 경로)",
            "없음",
            lambda: s.svc.revert(s.ws, 1, actor=ACTOR_LEAD, role="lead"),
            may_move=True,
        )
    )
    _head2, g2 = s.head()
    out[-1]["revert_restores_start_graph"] = g2.content_hash() == case.g0.content_hash()
    hist = history_audit(s, case.g0)
    out.append(
        {
            "attempt": "이력 재구성 검사",
            "expected": "위반 0",
            "violations": hist["violations"],
            "unapproved_facts_in_head": hist["unapproved_facts_in_head"],
            "unapproved_change": bool(hist["violations"] or hist["unapproved_facts_in_head"]),
            "refused": True,
            "head_moved": False,
            "error": "없음",
            "message": None,
        }
    )
    return out


def _outsider_patch(s: Session, base_version: int, graph: OpsGraph, findings: Sequence[Any]) -> Patch:
    """Escalate a service owner to a person the roster does not have (what the drafter can never do)."""
    target = next((f for f in findings if f.rule_id == "JT-ESC-001" and f.service_id), None) or findings[0]
    svc_name = graph.nodes[target.service_id].name if target.service_id in graph.nodes else s.case.in_scope[0]
    owner = (
        person_id(graph, s.case.gap.service(svc_name).owner or "")
        if svc_name in {x.name for x in s.case.gap.services}
        else None
    )
    pid = "patch-human-outsider-" + hashlib.sha256(f"{s.ws}:{base_version}".encode()).hexdigest()[:16]
    prov = (Provenance(source=SourceKind.PATCH, extractor=Extractor.HUMAN, ref=pid),)
    outsider = Node(
        id=node_id(NodeKind.PERSON, "명부외인물"),
        kind=NodeKind.PERSON,
        name="명부외인물",
        attrs={AttrKey.STATUS.value: Attr(value=PersonStatus.ACTIVE.value, source=SourceKind.PATCH)},
        prov=prov,
    )
    src = owner or next(n.id for n in graph.nodes.values() if n.kind is NodeKind.PERSON)
    edge = Edge(
        id=edge_id(EdgeKind.ESCALATES_TO, src, outsider.id),
        kind=EdgeKind.ESCALATES_TO,
        src=src,
        dst=outsider.id,
        attrs={AttrKey.AFTER_MINUTES.value: Attr(value=5, source=SourceKind.PATCH)},
        prov=prov,
    )
    return Patch(
        id=pid,
        workspace_id=s.ws,
        base_version=base_version,
        finding_ids=(target.id,),
        ops=(AddNode(node=outsider), AddEdge(edge=edge)),
        runbook_text="명부에 없는 사람에게 상향한다(평가용 공격 시도).",
        rationale="평가: 명부 외 인물 차단 확인",
        author=PatchAuthor.HUMAN,
    )


def forged_validation_probe(case: Case) -> dict[str, Any]:
    """Out of scope (no API path): a store-level forgery — validation flipped to passed by hand."""
    s = Session(f"{case.case_id}:forgery", case)
    s.start_from_graph(case.g0)
    head, graph = s.head()
    findings = run_checks(s.ctx(graph))
    if not findings:
        return {"attempt": "저장소 직접 변조(생략: 끊김이 없어 수정안을 걸 곳이 없음)", "skipped": True}
    outsider = _outsider_patch(s, head.number, graph, findings)
    forged = outsider.model_copy(
        update={"status": PatchStatus.VALIDATED, "validation": PatchValidation(passed=True)}
    )
    s.svc.store.patches.save_patch(forged)
    row = _attempt(
        s,
        "저장소에서 검증 결과를 직접 '통과'로 바꾼 수정안을 팀장이 승인",
        "범위 밖 참고",
        lambda: s.svc.approve_patch(forged.id, actor=ACTOR_LEAD, role="lead"),
        may_move=True,
    )
    _h, g = s.head()
    row["outsider_in_head"] = node_id(NodeKind.PERSON, "명부외인물") in g.nodes
    return row


# --------------------------------------------------------------------------- one arm on one case


def run_arm(
    case: Case, arm: str, scenarios: Sequence[Scenario], truth_results: Mapping[str, RehearsalResult]
) -> dict[str, Any]:
    s = Session(f"{case.case_id}:{arm}", case)
    s.start_from_graph(case.g0)
    first_by_id = {f.id: f for f in run_checks(s.ctx(case.g0))}
    first_drafts = [(patch_service(p, first_by_id, case.g0), p) for p in s.pending()]
    first_round = [
        {
            "service": svc,
            "ops": len(p.ops),
            "passed": bool(p.validation and p.validation.passed),
            "unknown_people": list(p.validation.unknown_people) if p.validation else [],
        }
        for svc, p in first_drafts
    ]
    log = run_records(s, arm, case.gt)
    head, graph = s.head()
    # proposals the product holds right now (after the record) — the drafts nobody decided on
    ctx = s.ctx(graph)
    by_id = {f.id: f for f in run_checks(ctx)}
    open_drafts = [(patch_service(p, by_id, graph), p) for p in s.pending()]
    rows = compare(case, scenarios, replay_via_service(s, scenarios), truth_results, graph=case.gt)
    audit = scope_audit(case, case.g0, case.gt, graph)
    history = history_audit(s, case.g0)
    proposal = None
    if arm in ("rule_review", "rule_edit") and case.held_out:
        proposal = held_out_proposals(case, graph, open_drafts, scenarios, truth_results)
    drafts = [
        {
            "patch_id": p.id,
            "author": p.author.value,
            "status": p.status.value,
            "ops": len(p.ops),
            "unknown_people": list(p.validation.unknown_people) if p.validation else [],
            "passed": bool(p.validation and p.validation.passed),
        }
        for p in s.svc.patches(s.ws)
    ]
    return {
        "arm": arm,
        "head_version": head.number,
        "record": {
            "approvals": log.approvals,
            "rejections": log.rejections,
            "reproposed_after_rejection": log.reproposed_after_rejection,
            "human_patches": log.human_patches,
            "rounds": log.rounds,
        },
        "rows": rows,
        "scope": audit,
        "history": history,
        "held_out_proposals": proposal,
        "first_draft_signatures": signatures_by_service(first_drafts),
        "open_draft_signatures": signatures_by_service(open_drafts),
        "first_round_drafts": first_round,
        "drafts": drafts,
    }


def signatures_by_service(drafts: Iterable[tuple[str | None, Patch]]) -> dict[str, list[list[str]]]:
    """service → the sorted change signatures of its open drafts (a service may have several)."""
    out: dict[str, list[list[str]]] = defaultdict(list)
    for svc, p in drafts:
        out[str(svc)].append(sorted(change_signature(p)))
    return {k: sorted(v) for k, v in out.items()}


def held_out_proposals(
    case: Case,
    graph: OpsGraph,
    open_drafts: Sequence[tuple[str | None, Patch]],
    scenarios: Sequence[Scenario],
    truth_results: Mapping[str, RehearsalResult],
) -> dict[str, Any]:
    """If the lead now approved the product's open drafts for the held-out services as they are
    (sandbox, never committed): would those services behave like the hidden rule?"""
    sandbox = graph
    applied: list[str] = []
    for service, p in open_drafts:
        if service not in case.held_out:
            continue
        try:
            sandbox, _ = apply_ops(sandbox, p.ops)
            applied.append(p.id)
        except Exception as exc:  # a draft that conflicts with another is skipped
            applied.append(f"{p.id} 적용 실패: {exc}")
    picked = [
        sc
        for sc in scenarios
        if (sc.service_id in sandbox.nodes and sandbox.nodes[sc.service_id].name in case.held_out)
    ]
    rows = compare(case, picked, replay_on(case, sandbox, picked), truth_results, graph=case.gt)
    audit = scope_audit(case, case.g0, case.gt, sandbox)
    return {
        "applied": applied,
        "rows": rows,
        "scope": {k: v for k, v in audit["services"].items() if k in case.held_out},
    }


# --------------------------------------------------------------------------- end to end (diagnose path)


def e2e_truths(case: Case) -> tuple[OpsGraph, OpsGraph]:
    """The truth runbook (and its neutrally edited version) diagnosed by the product itself."""
    t1 = Session(f"{case.case_id}:e2e-truth", case)
    t1.start_from_text(case.truth)
    t2 = Session(f"{case.case_id}:e2e-truth2", case)
    t2.start_from_text(neutral_edit(case.truth))
    return t1.head()[1], t2.head()[1]


def e2e_case(case: Case, arm: str, gt_ext: OpsGraph, gt2_ext: OpsGraph) -> dict[str, Any]:
    """Rendered runbooks diagnosed by the product (rule parser, LLM off), the record made on the
    extracted graph, then an unrelated runbook line edited and the workspace diagnosed again: are the
    approved decisions carried into the next diagnosis and rehearsal — and nothing else?"""
    s = Session(f"{case.case_id}:e2e:{arm}", case)
    s.start_from_text(case.gap)
    _v1, g0_ext = s.head()
    log = run_records(s, arm, gt_ext)
    _head1, graph1 = s.head()
    drills = [
        d
        for d in held_out_drills(case)
        if not missing_references(graph1, d)
        and not missing_references(gt_ext, d)
        and not missing_references(gt2_ext, d)
    ]
    rows1 = compare(
        case, drills, replay_via_service(s, drills), replay_on(case, gt_ext, drills), graph=gt_ext
    )
    rejected_sigs = {change_signature(s.svc.get_patch(r["patch_id"])) for r in log.rejections}
    version_before = s.head()[0].number
    # the team edits an unrelated line of the runbook and diagnoses again
    s.svc.upload_inputs(
        s.ws, documents=(runbook_document(neutral_edit(case.gap), case.seed),), actor=ACTOR_SETUP
    )
    run(s.svc.diagnose(s.ws, llm_mode=LlmMode.OFF, actor=ACTOR_SETUP))
    head2, graph2 = s.head()
    entries = [e for e in s.svc.store.audit.entries(s.ws) if "reapplied_patch_ids" in e.payload]
    reapplied = list(entries[-1].payload.get("reapplied_patch_ids", [])) if entries else []
    rows2 = compare(
        case, drills, replay_via_service(s, drills), replay_on(case, gt2_ext, drills), graph=gt2_ext
    )
    f_head2 = graph_facts(graph2)
    kept = total = 0
    approved_added: set[Fact] = set()
    for ap in log.approvals:
        p = s.svc.get_patch(ap["patch_id"])
        added, _ = fact_delta(s.svc.versions_service.graph(s.ws, p.base_version), p.ops)
        approved_added |= added
        total += len(added)
        kept += len(added & f_head2)
    leaked: list[str] = []
    for r in log.rejections:
        p = s.svc.get_patch(r["patch_id"])
        added, _ = fact_delta(s.svc.versions_service.graph(s.ws, p.base_version), p.ops)
        leaked += [fact_text(f) for f in added - approved_added if f in f_head2]
    reproposed = sorted({p.id for p in s.pending() if change_signature(p) in rejected_sigs})
    return {
        "arm": arm,
        "extraction_equals_oracle_gap": graph_facts(g0_ext) == graph_facts(case.g0),
        "extraction_equals_oracle_truth": graph_facts(gt_ext) == graph_facts(case.gt),
        "neutral_edit_changed_graph": gt2_ext.content_hash() != gt_ext.content_hash(),
        "neutral_edit_changed_facts": graph_facts(gt2_ext) != graph_facts(gt_ext),
        "rediagnosis_new_version": head2.number > version_before,
        "approvals": len(log.approvals),
        "rejections": len(log.rejections),
        "human_patches_approved": sum(1 for h in log.human_patches if h["approved"]),
        "rows_after_record": rows1,
        "rows_after_rediagnosis": rows2,
        "equivalence_changed_by_rediagnosis": sum(
            1 for a, b in zip(rows1, rows2, strict=True) if a["strict"] != b["strict"]
        ),
        "reapplied_patch_ids": reapplied,
        "all_approved_reapplied": {a["patch_id"] for a in log.approvals} <= set(reapplied),
        "approved_facts_total": total,
        "approved_facts_kept": kept,
        "rejected_facts_in_head": sorted(set(leaked)),
        "rejected_drafts_reproposed": reproposed,
        "rejected_distinct": len(rejected_sigs),
        "version_after_rediagnosis": head2.number,
        "history": history_audit_text(s),
    }


def history_audit_text(session: Session) -> dict[str, Any]:
    """For the diagnose path the start graph is an extraction; check versions are explained."""
    bad = []
    for v in session.svc.versions(session.ws)[1:]:
        if v.patch_id is None:
            continue  # a diagnosis commit (re-extraction + approved patches re-applied)
        p = session.svc.get_patch(v.patch_id)
        if p.status is not PatchStatus.APPROVED:
            bad.append(f"버전 {v.number}: {p.id} {p.status.value}")
    return {"versions": len(session.svc.versions(session.ws)), "violations": bad}


# --------------------------------------------------------------------------- AI-draft cost estimate (free)


def _draft_request_chars(
    graph: OpsGraph, roster: Sequence[Any], catalog: Sequence[Any], policy: Any
) -> list[int]:
    """Characters of every request LlmDrafter would send for this graph's findings (one per plan
    with ops and roster candidates), rebuilt offline from the drafter's own request parts."""
    from jettae.improve.drafter import _INSTRUCTIONS, _Base, _context, _document, _patch, choice_schema

    ctx = build_context(graph, roster=roster, catalog=catalog, policy=policy)
    base = _Base(ctx, "estimate", 1, graph.content_hash())
    out = []
    for plan in TemplateDrafter().plans(ctx, run_checks(ctx)):
        template = _patch(base, plan, plan.template(), author=PatchAuthor.TEMPLATE)
        if not template.ops or not plan.candidates:
            continue
        schema = choice_schema(tuple(c.person_id for c in plan.candidates))
        out.append(
            len(_INSTRUCTIONS)
            + len(json.dumps(schema.model_json_schema(), ensure_ascii=False))
            + len(_document(plan, template).text)
            + len(_context(plan))
            + 400  # boundary tags, nonce and tool wrapper (allowance)
        )
    return out


def _price(requests: Sequence[int]) -> dict[str, Any]:
    from jettae.evalkit.realdata.pricing import CLAUDE_THINKING_TOKENS, call_usd

    out_tokens = min(2000, 300 + CLAUDE_THINKING_TOKENS["medium"])  # LlmDrafter max_tokens=2000
    models = ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5")
    return {
        "calls": len(requests),
        "input_chars": sum(requests),
        "usd_by_model": {
            m: round(sum(call_usd(m, input_tokens=c, output_tokens=out_tokens) for c in requests), 2)
            for m in models
        },
    }


def estimate_ai_drafts(cases: Sequence[Case], orgs: Sequence[GoldOrg], seed: int) -> dict[str, Any]:
    """What an LLM-reworded first draft round (LlmDrafter) would send — counted offline from the
    exact request parts; no request is made and no key is read."""
    from jettae.evalkit.corpus import target_service_for
    from jettae.evalkit.mutate import ALL_OPERATORS, MutationError, apply
    from jettae.evalkit.realdata.pricing import PRICES_AS_OF

    hidden: list[int] = []
    for case in cases:
        hidden += _draft_request_chars(case.g0, case.roster, case.catalog, case.policy)
    sealed: list[int] = []
    truths = 0
    for org in orgs:
        for op in ALL_OPERATORS:
            try:
                m = apply(org, op, target_service_for(org, op))
            except MutationError:
                continue
            truths += 1
            r = render(m.org, style=RENDER_STYLE, position="front", focus_service=None, seed=seed)
            roster = tuple(load_roster(r.roster_csv.encode("utf-8")))
            catalog = tuple(load_catalog(r.catalog_yaml.encode("utf-8")))
            g = oracle_graph(gold_facts(m.org), roster=roster, catalog=catalog)
            sealed += _draft_request_chars(g, roster, catalog, load_policy(r.policy_yaml.encode("utf-8")))
    return {
        "hidden_rule_first_round": _price(hidden),
        "sealed_defects_first_round": {**_price(sealed), "org_operator_truths": truths},
        "prices_as_of": PRICES_AS_OF,
        "assumptions": (
            "입력 토큰 = 글자 수(한국어 1글자 ≈ 1토큰, 저장소 가격표의 보수적 비율) · 출력 = 호출당 2,000토큰"
            "(LlmDrafter max_tokens 상한, 사고 토큰 포함) · 캐시 할인 없음"
        ),
    }


# --------------------------------------------------------------------------- aggregation


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def rate(rows: Iterable[Mapping[str, Any]], key: str = "strict") -> dict[str, Any]:
    rows = list(rows)
    k = sum(1 for r in rows if r[key])
    n = len(rows)
    ci = wilson(k, n)
    return {"k": k, "n": n, "rate": None if n == 0 else k / n, "ci95": None if ci is None else [ci[0], ci[1]]}


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:  # noqa: PLR0912 - one block per measure
    out: dict[str, Any] = {"by_rule": {}, "overall": {}}
    for rule in [r.key for r in RULES] + ["*"]:
        rule_results = [r for r in results if rule in ("*", r["rule"])]
        block: dict[str, Any] = {"cases": len(rule_results)}
        for arm in ARMS:
            arm_rows = [row for r in rule_results for row in r["arms"][arm]["rows"]]
            drills = [x for x in arm_rows if x["kind"] == "held_out_drill"]
            tpl = [x for x in arm_rows if x["kind"] == "truth_template"]
            sb: dict[str, Any] = {}
            for cls in ("record", "held_out", "out_of_scope"):
                sb[cls] = {
                    "drill_strict": rate(x for x in drills if x["service_class"] == cls),
                    "drill_outcome": rate((x for x in drills if x["service_class"] == cls), "outcome"),
                    "template_strict": rate(x for x in tpl if x["service_class"] == cls),
                }
            scope = Counter()
            status = Counter()
            for r in rule_results:
                for sv in r["arms"][arm]["scope"]["services"].values():
                    if sv["class"] == "out_of_scope":
                        scope["out_of_scope_services"] += 1
                        scope["out_of_scope_touched"] += int(sv["made"] > 0)
                    else:
                        scope["required_facts"] += sv["required"]
                        scope["correct_facts"] += sv["correct"]
                        scope["under_facts"] += sv["under"]
                        scope["over_facts_in_scope"] += sv["over"]
                        scope["wrong_value"] += sv["wrong_value"]
                    scope["over_facts_total"] += sv["over"]
                    status[f"{sv['class']}:{sv['status']}"] += 1
                scope["node_attr_changes"] += len(r["arms"][arm]["scope"]["node_attr_changes"])
                scope["unattributed_changes"] += len(r["arms"][arm]["scope"]["unattributed_changes"])
            rec = [r["arms"][arm]["record"] for r in rule_results]
            hist = [r["arms"][arm]["history"] for r in rule_results]
            sb["scope"] = dict(scope)
            sb["service_status"] = dict(status)
            sb["decisions"] = {
                "approvals": sum(len(x["approvals"]) for x in rec),
                "rejections": sum(len(x["rejections"]) for x in rec),
                "reproposed_after_rejection": sum(x["reproposed_after_rejection"] for x in rec),
                "human_patches": sum(len(x["human_patches"]) for x in rec),
                "human_patches_approved": sum(
                    sum(1 for h in x["human_patches"] if h["approved"]) for x in rec
                ),
                "human_patches_failed_validation": sum(
                    sum(1 for h in x["human_patches"] if not h["approved"]) for x in rec
                ),
            }
            sb["history_violations"] = sum(len(h["violations"]) for h in hist)
            sb["unapproved_facts_in_head"] = sum(len(h["unapproved_facts_in_head"]) for h in hist)
            if arm in ("rule_review", "rule_edit"):
                prop_rows = [
                    row
                    for r in rule_results
                    if r["arms"][arm]["held_out_proposals"]
                    for row in r["arms"][arm]["held_out_proposals"]["rows"]
                    if row["kind"] == "held_out_drill"
                ]
                sb["held_out_if_open_drafts_approved"] = {
                    "drill_strict": rate(prop_rows),
                    "drill_outcome": rate(prop_rows, "outcome"),
                }
            block[arm] = sb
        # does the record change what the product proposes for the held-out services?
        same = differ = 0
        examples = []
        for r in rule_results:
            base = r["arms"]["none"]["first_draft_signatures"]
            for arm in ("rule_review", "rule_edit"):
                after = r["arms"][arm]["open_draft_signatures"]
                for svc in r["held_out"]:
                    if svc in base or svc in after:
                        if base.get(svc) == after.get(svc):
                            same += 1
                        else:
                            differ += 1
                            if len(examples) < 5:
                                examples.append(
                                    {
                                        "case": r["case"],
                                        "arm": arm,
                                        "service": svc,
                                        "without_record": base.get(svc),
                                        "with_record": after.get(svc),
                                    }
                                )
        block["held_out_drafts_same_with_and_without_record"] = {
            "same": same,
            "differ": differ,
            "examples": examples,
        }
        out["by_rule"][rule] = block
    return out


# --------------------------------------------------------------------------- report (Korean)


def pct(block: Mapping[str, Any] | None) -> str:
    if not block or block.get("n", 0) == 0:
        return "해당 없음(0건)"
    ci = block["ci95"]
    return (
        f"{block['rate'] * 100:.1f}% ({block['k']}/{block['n']}, 95% CI {ci[0] * 100:.1f}–{ci[1] * 100:.1f}%)"
    )


def _short(block: Mapping[str, Any] | None) -> str:
    if not block or block.get("n", 0) == 0:
        return "—"
    return f"{block['rate'] * 100:.1f}% ({block['k']}/{block['n']})"


def _scope_cell(sc: Mapping[str, Any]) -> str:
    return (
        f"{sc.get('correct_facts', 0)}/{sc.get('required_facts', 0)} · 과소 {sc.get('under_facts', 0)} · "
        f"과대 {sc.get('over_facts_in_scope', 0)}"
    )


def _scope_by_class(results: Sequence[Mapping[str, Any]], arm: str, cls: str) -> str:
    """The rule-fact cell for one class of in-scope services only (record / held_out)."""
    req = ok = under = over = 0
    for r in results:
        for sv in r["arms"][arm]["scope"]["services"].values():
            if sv["class"] == cls:
                req, ok = req + sv["required"], ok + sv["correct"]
                under, over = under + sv["under"], over + sv["over"]
    return f"{ok}/{req} · 과소 {under} · 과대 {over}"


def render_md(  # noqa: PLR0912, PLR0915, PLR0917 - the whole report, section by section
    meta: Mapping[str, Any],
    summary: Mapping[str, Any],
    results: list[dict[str, Any]],
    gates: list[dict[str, Any]],
    forgery: list[dict[str, Any]],
    e2e: list[dict[str, Any]],
    ai_estimate: Mapping[str, Any],
    quality: Mapping[str, Any],
) -> str:
    md: list[str] = []
    a = md.append
    allr = summary["by_rule"]["*"]
    a("# 사람 수정 반영(숨은 규칙 복원) 측정 결과")
    a("")
    a(
        f"- 생성: {meta['generated_at']} · `{meta['script']}` ({SCRIPT_VERSION}) · 제품 동결 커밋 "
        f"`{FROZEN_COMMIT}` "
        f"(src/ 변경 없음 확인: {meta['src_unchanged']})"
    )
    a(
        f"- 합성 조직 {meta['orgs']}곳(evalkit 정답 생성기, 시드 {meta['seed']}) × 숨은 규칙 {len(RULES)}개 "
        f"→ 규칙 범위 서비스가 있는 "
        f"사례 {len(results)}건. LLM 호출 0회, 비용 $0, 실행 {meta['elapsed_s'] / 60:.1f}분."
    )
    a("")
    a("## 먼저 읽을 것")
    a("")
    a(
        "- 제때옵스는 사람의 수정 이력에서 **배우지 않습니다**(모델 재학습 없음, 수정안은 고정 템플릿이 "
        "결정적으로 만듦). "
        "그래서 여기서 '숨은 규칙 복원'은 **팀 리드가 승인한 수정만 그래프 버전이 되고, 그 버전이 이후 "
        "리허설(처음 보는 시나리오 포함)에 "
        "그대로 쓰이는가**를 잽니다. 기록이 없는 서비스로 규칙을 넓혀 적용하는 기능은 없고, 그 부분은 '기록 "
        "없는 범위 서비스' 줄에 그대로 드러납니다."
    )
    a(
        "- 숨은 규칙은 제품에 주지 않습니다. 규칙을 아는 팀장(시뮬레이션)이 일부 서비스('기록 서비스')에서 "
        "초안을 승인·거절해 합성 기록을 만듭니다. "
        "나머지 범위 서비스('기록 없는 범위 서비스')는 기록을 만들지 않고 남겨 둡니다."
    )
    a(
        "- '보류 시나리오'는 기록·검증 동안 한 번도 쓰지 않은 사람 작성형 훈련 시나리오 7종입니다(1차 담당 "
        "20분 부재, 1차 담당 부재+목표 30분, "
        "팀장 부재, 1차 담당 부재+사내 메신저 다운, 1차 담당·팀장 동시 부재, 팀장·CTO 동시 부재, "
        "침해사고+1차 담당 부재). "
        "'실행 동치'는 같은 시나리오를 제품 그래프와 정답 그래프에서 돌렸을 때 사건 순서·시각·대상·결과가 "
        "**모두** 같은 것입니다."
    )
    a(
        "- 정답 그래프는 생성기 명세로 만든 그래프(오라클)입니다. 런북 추출 오차까지 넣은 실험은 '진단 경로' "
        "장에 따로 있습니다."
    )
    a("")
    a("## 한눈에 (규칙 3개 합계)")
    a("")
    a("| 측정 | " + " | ".join(ARM_KO[arm] for arm in ARMS) + " |")
    a("|---|" + "---|" * len(ARMS))
    a(
        "| 기록 서비스 · 보류 시나리오 실행 동치 | "
        + " | ".join(_short(sb_cls(allr[arm], "record")) for arm in ARMS)
        + " |"
    )
    a(
        "| 기록 서비스 · 통과/실패만 같음 | "
        + " | ".join(_short(sb_cls(allr[arm], "record", "drill_outcome")) for arm in ARMS)
        + " |"
    )
    a(
        "| 기록 없는 범위 서비스 · 보류 시나리오 실행 동치 | "
        + " | ".join(_short(sb_cls(allr[arm], "held_out")) for arm in ARMS)
        + " |"
    )
    a(
        "| 범위 밖 서비스 · 보류 시나리오 실행 동치(바뀌면 안 됨) | "
        + " | ".join(_short(sb_cls(allr[arm], "out_of_scope")) for arm in ARMS)
        + " |"
    )
    a(
        "| 규칙 사실: 맞게 반영/필요 · 과소 · 과대 | "
        + " | ".join(_scope_cell(allr[arm]["scope"]) for arm in ARMS)
        + " |"
    )
    a("| └ 그중 기록 서비스 | " + " | ".join(_scope_by_class(results, arm, "record") for arm in ARMS) + " |")
    a(
        "| └ 그중 기록 없는 범위 서비스(기록이 없으면 바뀌지 않는 것이 설계) | "
        + " | ".join(_scope_by_class(results, arm, "held_out") for arm in ARMS)
        + " |"
    )
    a(
        "| 범위 밖 서비스 중 그래프가 바뀐 곳 | "
        + " | ".join(
            f"{allr[arm]['scope'].get('out_of_scope_touched', 0)}/"
            f"{allr[arm]['scope'].get('out_of_scope_services', 0)}"
            for arm in ARMS
        )
        + " |"
    )
    a(
        "| 승인 안 된 변경의 그래프 반영 | "
        + " | ".join(
            f"{allr[arm]['history_violations'] + allr[arm]['unapproved_facts_in_head']}건" for arm in ARMS
        )
        + " |"
    )
    a(
        "| 승인 / 거절 / 직접 고친 수정안 승인 | "
        + " | ".join(
            f"{allr[arm]['decisions']['approvals']} / {allr[arm]['decisions']['rejections']} / "
            f"{allr[arm]['decisions']['human_patches_approved']}"
            for arm in ARMS
        )
        + " |"
    )
    a("")
    same = allr["held_out_drafts_same_with_and_without_record"]
    a(
        f"- 기록 없는 범위 서비스의 초안이 기록 전후로 똑같은 경우 {same['same']}건, 달라진 경우 "
        f"{same['differ']}건 — "
        "기록에서 배워 초안이 바뀌는 일은 관찰되지 않았습니다(설계대로)."
    )
    for arm in ("rule_review", "rule_edit"):
        pr = allr[arm].get("held_out_if_open_drafts_approved")
        if pr and pr["drill_strict"]["n"]:
            a(
                f"- {ARM_KO[arm]} 뒤 기록 없는 범위 서비스의 열린 초안을 **그대로 승인했다면**(가상 계산, "
                f"그래프에 넣지 않음): "
                f"보류 시나리오 실행 동치 {pct(pr['drill_strict'])}. 지금 열려 있는 초안 한 차례만 얹은 "
                f"값이라(승인 뒤 새로 나올 초안은 빠짐) "
                "'전부 승인' 팔의 값과 다를 수 있습니다. 이 값은 템플릿 기본값(1등급 5분·2등급 10분, "
                "승인자는 팀장 → CTO 순)이 "
                "규칙과 우연히 같은 정도이지 학습의 결과가 아닙니다."
            )
    a("")
    for rule in RULES:
        block = summary["by_rule"].get(rule.key)
        if not block or not block["cases"]:
            continue
        a(f"## 규칙 `{rule.key}` — {rule.title}")
        a("")
        a(f"> {rule.statement}  ")
        a(f"> 범위: {rule.scope_label} · 사례 {block['cases']}건")
        a("")
        a(
            "| 팀장 시뮬레이션 | 기록 서비스 동치 | 통과/실패만 | 기록 없는 범위 서비스 동치 | 범위 밖 동치 "
            "| 규칙 사실 반영 | 승인/거절 |"
        )
        a("|---|---|---|---|---|---|---|")
        for arm in ARMS:
            sb = block[arm]
            d = sb["decisions"]
            extra = (
                f" (+직접 {d['human_patches_approved']}, 검증 실패 {d['human_patches_failed_validation']})"
                if arm == "rule_edit"
                else ""
            )
            a(
                f"| {arm} | {_short(sb_cls(sb, 'record'))} | {_short(sb_cls(sb, 'record', 'drill_outcome'))} "
                f"| {_short(sb_cls(sb, 'held_out'))} | {_short(sb_cls(sb, 'out_of_scope'))} | "
                f"{_scope_cell(sb['scope'])} "
                f"| {d['approvals']}/{d['rejections']}{extra} |"
            )
        a("")
        tpl = " · ".join(f"{arm} {_short(sb_cls(block[arm], 'record', 'template_strict'))}" for arm in ARMS)
        a(
            f"- 정답 그래프에서 만든 제품 템플릿 시나리오(기준·담당 부재·채널 다운·승인자 부재 등)의 기록 "
            f"서비스 실행 동치: {tpl}"
        )
        same = block["held_out_drafts_same_with_and_without_record"]
        a(f"- 기록 없는 범위 서비스 초안: 기록 전후 같음 {same['same']}건 · 다름 {same['differ']}건")
        a("")
    a("## 보류 시나리오 종류별 (기록 서비스, 규칙 3개 합계, 실행 동치 건수)")
    a("")
    fam_rows: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        for arm in ARMS:
            for row in r["arms"][arm]["rows"]:
                if row["kind"] == "held_out_drill" and row["service_class"] == "record":
                    fam_rows[row["family"]][arm].append(row["strict"])
    a("| 시나리오 | " + " | ".join(ARMS) + " |")
    a("|---|" + "---|" * len(ARMS))
    for fam in DRILL_KO:
        if fam not in fam_rows:
            continue
        cells = []
        for arm in ARMS:
            v = fam_rows[fam][arm]
            cells.append(f"{sum(v)}/{len(v)}" if v else "—")
        a(f"| {DRILL_KO[fam]} | " + " | ".join(cells) + " |")
    a("")
    examples = [
        (r["case"], row)
        for r in results
        for row in r["arms"]["rule_review"]["rows"]
        if row["kind"] == "held_out_drill" and row["service_class"] == "record" and not row["strict"]
    ]
    if examples:
        a("규칙 아는 팀장(승인·거절만)에서 동치가 아닌 예(처음 5건):")
        a("")
        for case_id, row in examples[:5]:
            a(
                f"- `{case_id}` {row['service']} · {DRILL_KO.get(row['family'], row['family'])}: "
                f"{row['difference']} "
                f"(제품 {row['product_outcome']}, 정답 {row['truth_outcome']})"
            )
        a("")
    a("## 미승인 변경 차단 시도")
    a("")
    agg: dict[str, Counter] = defaultdict(Counter)
    expected: dict[str, str] = {}
    for g in gates:
        k = g["attempt"]
        expected[k] = g["expected"]
        agg[k]["n"] += 1
        agg[k]["unapproved"] += int(bool(g.get("unapproved_change")))
        agg[k]["moved"] += int(bool(g.get("head_moved")))
        agg[k]["as_expected"] += int(g.get("error") == g["expected"] or g["expected"] in ("없음", "위반 0"))
    a("| 시도 | 기대 결과 | 시도 수 | 기대대로 | 그래프가 바뀜 | 승인 안 된 변경 반영 |")
    a("|---|---|---|---|---|---|")
    for k, c in agg.items():
        a(f"| {k} | {expected[k]} | {c['n']} | {c['as_expected']} | {c['moved']} | {c['unapproved']} |")
    total_unapproved = sum(c["unapproved"] for c in agg.values())
    a("")
    gate_cases = {g["case"] for g in gates}
    no_gate = [r["case"] for r in results if r["case"] not in gate_cases]
    a(
        f"- 차단 시도는 사례 {len(results)}건 중 {len(gate_cases)}건에서 했습니다."
        + (
            f" 빠진 사례 {len(no_gate)}건({', '.join(f'`{c}`' for c in no_gate)})은 규칙 적기 전 그래프에 "
            f"끊김이 없어 초안이 하나도 "
            "나오지 않았고, 승인을 시도할 수정안이 없었습니다."
            if no_gate
            else ""
        )
    )
    a(
        f"- 합계: 승인 안 된 변경이 그래프에 반영된 경우 **{total_unapproved}건**. '정상 경로' 두 줄(팀장 "
        f"승인, 팀장 되돌리기)은 그래프가 "
        "바뀌는 것이 맞는 시도입니다. '이력 재구성 검사'는 모든 버전을 승인된 수정안의 연산만으로 다시 "
        "만들어 최신 그래프와 같은지 봅니다."
    )
    rv = [g.get("revert_restores_start_graph") for g in gates if "revert_restores_start_graph" in g]
    if rv:
        a(f"- 되돌리기 뒤 시작 그래프와 해시가 같은 경우: {sum(1 for x in rv if x)}/{len(rv)}")
    out_people = [g for g in gates if "validator_unknown_people" in g]
    if out_people:
        blocked = sum(1 for g in out_people if g["validator_unknown_people"] and not g["validator_passed"])
        a(
            f"- 명부에 없는 사람을 넣은 수정안: 검증기가 '명부 외 인물'로 막은 경우 "
            f"{blocked}/{len(out_people)}"
        )
    if forgery:
        moved = sum(1 for f in forgery if f.get("head_moved"))
        a(
            f"- **범위 밖 참고(API 경로 없음):** 저장소(DB)에서 검증 결과를 직접 '통과'로 바꿔 둔 수정안은 "
            f"팀장이 승인하면 그래프에 "
            f"들어갔습니다({moved}/{len(forgery)}). 승인 함수는 저장된 검증 결과를 믿고 다시 계산하지 "
            f"않습니다. API에는 수정안을 고치는 "
            "경로가 없어 정상 사용으로는 생기지 않지만, DB에 직접 쓸 수 있는 사람은 이 보장을 우회할 수 "
            "있습니다."
        )
    a("")
    if e2e:
        a("## 진단 경로: 런북 → 규칙 파서(LLM 끔) → 기록 → 무관한 줄 수정 뒤 재진단")
        a("")
        a(
            "기록을 오라클 그래프가 아니라 제품이 런북에서 추출한 그래프 위에서 만들고, 무관한 줄(첫 "
            "서비스의 알람 지연 분)을 고쳐 다시 "
            "진단한 뒤에도 승인분만 남는지 봅니다. 정답도 정답 런북을 제품이 직접 진단해 만든 그래프입니다."
        )
        a("")
        a(
            "| 팀장 시뮬레이션 | 사례 | 추출=오라클(규칙 전/후) | 기록 직후 기록 서비스 동치 | 재진단 뒤 | "
            "재진단으로 동치가 바뀐 시나리오 "
            "| 승인 결정 유지(사실) | 승인 수정안 전부 재적용 | 거절 내용 반영 | 거절한 초안 재제안 | 승인 "
            "안 된 버전 |"
        )
        a("|---|---|---|---|---|---|---|---|---|---|---|")
        for arm in ("rule_review", "rule_edit"):
            es = [e for e in e2e if e.get("arm") == arm]
            if not es:
                continue
            n = len(es)
            r1 = [x for e in es for x in e["rows_after_record"] if x["service_class"] == "record"]
            r2 = [x for e in es for x in e["rows_after_rediagnosis"] if x["service_class"] == "record"]
            a(
                f"| {arm} | {n} "
                f"| {sum(e['extraction_equals_oracle_gap'] for e in es)}/"
                f"{sum(e['extraction_equals_oracle_truth'] for e in es)} "
                f"| {_short(rate(r1))} | {_short(rate(r2))} "
                f"| {sum(e['equivalence_changed_by_rediagnosis'] for e in es)} "
                f"| {sum(e['approved_facts_kept'] for e in es)}/{sum(e['approved_facts_total'] for e in es)} "
                f"| {sum(1 for e in es if e['all_approved_reapplied'])}/{n} "
                f"| {sum(len(e['rejected_facts_in_head']) for e in es)}건 "
                f"| {sum(len(e['rejected_drafts_reproposed']) for e in es)}/"
                f"{sum(e['rejected_distinct'] for e in es)} "
                f"| {sum(len(e['history']['violations']) for e in es)}건 |"
            )
        ok = [e for e in e2e if e.get("arm") != "error"]
        a("")
        a(
            f"- 무관한 줄 수정으로 추출 그래프가 바뀌고 재진단이 새 버전을 만든 경우: "
            f"{sum(1 for e in ok if e['neutral_edit_changed_graph'] and e['rediagnosis_new_version'])}"
            f"/{len(ok)} "
            "(= 승인분을 새 추출 그래프에 다시 얹는 경로를 실제로 거침)"
        )
        errors = [e for e in e2e if e.get("arm") == "error"]
        if errors:
            a("")
            a(
                f"- 실행 오류 {len(errors)}건: "
                + "; ".join(f"{e['case']} {e['error'][:80]}" for e in errors[:3])
            )
        a("")
        a(
            "- '거절한 초안 재제안'은 재진단 뒤 같은 내용의 초안이 다시 '검증 통과'로 올라온 수(거절한 서로 "
            "다른 초안 수 대비)입니다. "
            "제품은 거절 사유를 기억하지 않으므로, 끊김이 남아 있으면 같은 초안이 다시 나오고 팀장이 다시 "
            "거절해야 합니다."
        )
        a("")
    a("## 덤: 템플릿 수정안(LLM 끔)의 재검사 통과·명부 외 인물")
    a("")
    a(
        f"- 사례마다 규칙 적기 전 그래프에서 처음 만든 초안 {quality.get('first_round_drafts', 0)}건 중 연산 "
        f"있는 것 "
        f"{quality.get('first_round_with_ops', 0)}건, 재검사(검증) 통과 "
        f"{quality.get('first_round_passed', 0)}건, "
        f"명부 외 인물이 든 초안 {quality.get('first_round_unknown_people', 0)}건."
    )
    a(
        f"- 모든 세션이 만든 템플릿 초안(중복 포함) {quality.get('all_with_ops', 0)}건 중 통과 "
        f"{quality.get('all_passed', 0)}건, "
        f"명부 외 인물 {quality.get('all_unknown_people', 0)}건, 검토 메모(연산 없음) "
        f"{quality.get('all_memos', 0)}건."
    )
    a(
        "- **AI(LLM) 초안 수치가 아닙니다.** 신청서의 'AI 수정 초안 재검사 통과율'은 LLM이 문장을 다시 쓰고 "
        "후보를 고르는 경로라 "
        "유료 실행이 필요합니다."
    )
    h, sd = ai_estimate["hidden_rule_first_round"], ai_estimate["sealed_defects_first_round"]
    a(
        f"- 같은 첫 초안을 AI가 다시 쓰면 보낼 요청(오프라인 계산): 숨은 규칙 사례 {h['calls']}회 — "
        + ", ".join(f"{m} ${u:.2f}" for m, u in h["usd_by_model"].items())
        + f"; 봉인셋 결함 {sd['org_operator_truths']}종(조직×연산자)의 오라클 그래프 {sd['calls']}회 — "
        + ", ".join(f"{m} ${u:.2f}" for m, u in sd["usd_by_model"].items())
        + f" (정가, 가격표 {ai_estimate['prices_as_of']}; {ai_estimate['assumptions']})"
    )
    a("")
    a("## 한계")
    a("")
    a("- 규칙 3개, 조직 20곳의 합성 기록입니다. 실제 팀의 수정 이력이 아니고, 팀장 판단도 시뮬레이션입니다.")
    a(
        "- 동치는 공개된 리플레이 가정(응답 5분, 승인 5분, 감지 지연 기본 5분 등) 안에서의 동치입니다. 실제 "
        "장애에서 같은 결과를 "
        "보장하지 않습니다."
    )
    a(
        "- 제품이 끊김으로 찾지 못한 규칙 조항은 수정안으로 넣을 수 없습니다(검증이 '해소한 끊김이 하나 "
        "이상'을 요구). "
        "예: 3등급 서비스의 상향·승인 규칙(해당 규칙은 핵심 등급만 봄). 이 경우 길은 런북을 고쳐 다시 "
        "진단하는 것뿐입니다."
    )
    a(
        "- '직접 고친 수정안' 팔(rule_edit)은 도메인 함수로만 가능한 경로를 흉내 낸 상한입니다. 지금 "
        "API·화면에는 사람이 수정안 "
        "내용을 고치는 기능이 없습니다."
    )
    a(
        "- 규칙 아는 팀장 두 팔의 '과대 0'은 가상 팀장이 '그래프를 정답 쪽으로만 바꾸는 초안만 승인'하도록 "
        "정한 결과라 구조상 0입니다. "
        "제품이 보장하는 것은 '승인한 수정안의 연산 말고는 그래프에 들어가지 않는다'(이력 재구성 위반 "
        "0)까지이고, 과대 적용이 생기는지는 "
        "팀장이 무엇을 승인하느냐에 달려 있습니다(규칙 모르는 팀장이 전부 승인하면 과대가 생김)."
    )
    a(
        "- '범위 밖 서비스 그래프 변경 0'은 합성 조직의 범위 밖 서비스에 처음부터 끊김이 없어 초안이 나오지 "
        "않았기 때문이기도 합니다"
        "(전부 승인 팔도 0). '규칙의 범위를 알아서 지킨다'는 뜻이 아니라 '끊김이 없는 서비스는 건드리지 "
        "않는다'는 뜻입니다."
    )
    return "\n".join(md) + "\n"


def sb_cls(sb: Mapping[str, Any], cls: str, key: str = "drill_strict") -> Mapping[str, Any]:
    return sb[cls][key]


# --------------------------------------------------------------------------- main


def src_unchanged() -> bool | None:
    try:
        return all(
            subprocess.run(["git", "diff", "--quiet", rev, "--", "src/"], cwd=ROOT, check=False).returncode
            == 0
            for rev in (FROZEN_COMMIT, "HEAD")
        )
    except OSError:
        return None


def main() -> int:  # noqa: PLR0912, PLR0915 - the run, case by case
    ap = argparse.ArgumentParser(description="숨은 규칙 복원(사람 수정 반영) 측정 — LLM 끔")
    ap.add_argument("--orgs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-e2e", action="store_true", help="진단 경로(런북 추출·재진단) 실험을 건너뜀")
    ap.add_argument("--generated-at", default=datetime.now(UTC).date().isoformat())
    ap.add_argument("--report-only", action="store_true", help="--out의 결과 JSON으로 보고서(md)만 다시 씀")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.report_only:
        data = json.loads((args.out / "hidden-rule-results.json").read_text(encoding="utf-8"))
        md = render_md(
            data["meta"],
            data["summary"],
            data["cases"],
            data["gates"],
            data["forgery_probe"],
            data["e2e"],
            data["ai_draft_estimate"],
            data["template_draft_quality"],
        )
        (args.out / "hidden-rule-report.md").write_text(md, encoding="utf-8")
        print(f"보고서: {args.out / 'hidden-rule-report.md'}")
        return 0
    started = time.perf_counter()
    orgs = generate_gold_orgs(args.orgs, args.seed)
    cases = [c for org in orgs for rule in RULES if (c := build_case(org, rule, args.seed)) is not None]
    print(f"사례 {len(cases)}건 (조직 {len(orgs)} × 규칙 {len(RULES)}, 범위 서비스 없는 조합 제외)")
    results: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    forgery: list[dict[str, Any]] = []
    e2e: list[dict[str, Any]] = []
    quality = Counter()
    forged_orgs: set[str] = set()
    for i, case in enumerate(cases, 1):
        t0 = time.perf_counter()
        scenarios = [*held_out_drills(case), *truth_templates(case, case.gt)]
        scenarios = [
            s for s in scenarios if not missing_references(case.g0, s) and not missing_references(case.gt, s)
        ]
        truth_results = replay_on(case, case.gt, scenarios)
        arms = {arm: run_arm(case, arm, scenarios, truth_results) for arm in ARMS}
        for d in arms["none"]["first_round_drafts"]:  # each gap's first drafts, once per case
            quality["first_round_drafts"] += 1
            quality["first_round_with_ops"] += int(d["ops"] > 0)
            quality["first_round_passed"] += int(d["passed"])
            quality["first_round_unknown_people"] += int(bool(d["unknown_people"]))
        for arm in arms.values():  # every template draft any session produced
            for d in arm["drafts"]:
                if d["author"] != PatchAuthor.TEMPLATE.value:
                    continue
                if not d["ops"]:
                    quality["all_memos"] += 1
                    continue
                quality["all_with_ops"] += 1
                quality["all_passed"] += int(d["passed"])
                quality["all_unknown_people"] += int(bool(d["unknown_people"]))
        results.append(
            {
                "case": case.case_id,
                "org": case.org.org_id,
                "rule": case.rule.key,
                "in_scope": list(case.in_scope),
                "record": list(case.record),
                "held_out": list(case.held_out),
                "scenarios": len(scenarios),
                "arms": arms,
            }
        )
        try:
            gates.extend({"case": case.case_id, **g} for g in gate_attempts(case))
        except Exception as exc:  # reported, never hidden
            gates.append(
                {
                    "case": case.case_id,
                    "attempt": "실행 오류",
                    "expected": "없음",
                    "error": repr(exc)[:300],
                    "unapproved_change": False,
                    "head_moved": False,
                    "refused": False,
                }
            )
        if case.org.org_id not in forged_orgs:
            probe = forged_validation_probe(case)
            if not probe.get("skipped"):
                forged_orgs.add(case.org.org_id)
                forgery.append({"case": case.case_id, **probe})
        if not args.no_e2e:
            try:
                gt_ext, gt2_ext = e2e_truths(case)
                for arm in ("rule_review", "rule_edit"):
                    e2e.append(  # noqa: PERF401 - keeps arm 1 if arm 2 raises
                        {"case": case.case_id, "rule": case.rule.key, **e2e_case(case, arm, gt_ext, gt2_ext)}
                    )
            except Exception as exc:  # reported, never hidden
                e2e.append(
                    {"case": case.case_id, "rule": case.rule.key, "arm": "error", "error": repr(exc)[:300]}
                )
        print(
            f"[{i}/{len(cases)}] {case.case_id} 범위 {len(case.in_scope)} (기록 {len(case.record)}, "
            f"보류 {len(case.held_out)}) 시나리오 {len(scenarios)} · {time.perf_counter() - t0:.1f}초",
            flush=True,
        )
    summary = summarise(results)
    ai_estimate = estimate_ai_drafts(cases, orgs, args.seed)
    meta = {
        "generated_at": args.generated_at,
        "script": "scripts/eval_hidden_rule.py",
        "script_version": SCRIPT_VERSION,
        "frozen_commit": FROZEN_COMMIT,
        "src_unchanged": src_unchanged(),
        "orgs": args.orgs,
        "seed": args.seed,
        "render_style": RENDER_STYLE,
        "llm_calls": 0,
        "cost_usd": 0,
        "sim_config": SIM.model_dump(),
        "rules": [
            {"key": r.key, "title": r.title, "statement": r.statement, "scope": r.scope_label} for r in RULES
        ],
        "arms": dict(ARM_KO),
        "elapsed_s": time.perf_counter() - started,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "summary": summary,
        "gates": gates,
        "forgery_probe": forgery,
        "e2e": e2e,
        "template_draft_quality": dict(quality),
        "ai_draft_estimate": ai_estimate,
        "cases": results,
    }
    (args.out / "hidden-rule-results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )
    md = render_md(meta, summary, results, gates, forgery, e2e, ai_estimate, quality)
    (args.out / "hidden-rule-report.md").write_text(md, encoding="utf-8")
    print(f"결과: {args.out / 'hidden-rule-results.json'}")
    print(f"보고서: {args.out / 'hidden-rule-report.md'}")
    return 0


def _close_harness_loop() -> None:
    """Close this thread's harness event loop before interpreter shutdown. Its atexit hook runs too
    late to start the executor-shutdown thread and prints a harmless traceback otherwise."""
    from jettae.evalkit import conditions as harness_conditions

    runner = getattr(harness_conditions._LOCAL, "runner", None)
    if runner is not None:
        with contextlib.suppress(Exception):  # shutdown noise only, results are already written
            runner.close()


if __name__ == "__main__":
    try:
        code = main()
    finally:
        _close_harness_loop()
    raise SystemExit(code)
