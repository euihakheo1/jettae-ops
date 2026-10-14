"""Tests for ``jettae.improve.validator``: re-check, re-rehearse, roster-only people."""

from __future__ import annotations

import pytest

from jettae.check.context import CheckContext
from jettae.check.registry import run_checks
from jettae.domain import (
    AddEdge,
    Attr,
    Edge,
    EdgeKind,
    Extractor,
    Finding,
    Outcome,
    Patch,
    PatchAuthor,
    PatchOp,
    Perturbation,
    PerturbationKind,
    Provenance,
    RemoveEdge,
    Scenario,
    SetAttr,
    SimConfig,
    SourceKind,
    edge_id,
)
from jettae.improve import TemplateDrafter, apply_ops, validate_patch
from jettae.rehearse.scenarios import template_scenarios
from jettae.rehearse.simulator import replay_all
from test_improve_support import Org, PaymentOrg, payment_org

PATCH = Provenance(source=SourceKind.PATCH, extractor=Extractor.HUMAN, ref="patch-test")


def patch_of(ops: list[PatchOp], finding_ids: list[str]) -> Patch:
    return Patch(
        id="patch-test",
        workspace_id="ws",
        base_version=1,
        finding_ids=tuple(finding_ids),
        ops=tuple(ops),
        runbook_text="시험",
        rationale="시험",
        author=PatchAuthor.HUMAN,
    )


def link(kind: EdgeKind, src: str, dst: str) -> Edge:
    return Edge(id=edge_id(kind, src, dst), kind=kind, src=src, dst=dst, prov=(PATCH,))


def ids_of(findings: list[Finding], rule_id: str) -> list[str]:
    return [f.id for f in findings if f.rule_id == rule_id]


def template(ctx: CheckContext, rule_id: str) -> Patch:
    patches = TemplateDrafter().draft(ctx, run_checks(ctx), workspace_id="ws", base_version=1)
    return next(p for p in patches if any(f.startswith(rule_id) for f in p.finding_ids))


@pytest.fixture
def po() -> PaymentOrg:
    return payment_org()


def test_owner_replacement_passes_and_explains_the_rekeyed_gap(po: PaymentOrg) -> None:
    ctx = po.org.ctx()
    before = run_checks(ctx)
    patch = template(ctx, "JT-PPL-001")
    result = validate_patch(ctx, patch, baseline_findings=before)
    assert result.passed
    assert set(result.resolved_finding_ids) == set(patch.finding_ids)
    assert result.unresolved_finding_ids == ()
    assert result.new_finding_ids == ()  # "no escalation" was re-keyed from 김민준 to 이서연, not created
    assert result.unknown_people == ()
    assert result.integrity_errors == ()
    assert any("id만 달라진 기존 끊김" in n and "JT-ESC-001" in n for n in result.notes)
    assert any(n.startswith("결제 준비도 ") for n in result.notes)
    assert result.score_before is not None
    assert result.score_after is not None
    assert result.score_after > result.score_before
    assert validate_patch(ctx, patch) == result  # computing the baseline itself gives the same verdict


def test_approval_fix_turns_the_baseline_rehearsal_green(po: PaymentOrg) -> None:
    ctx = po.org.ctx()
    owner = template(ctx, "JT-PPL-001")
    ctx = ctx.with_graph(apply_ops(ctx.graph, owner.ops)[0])
    scenarios = template_scenarios(ctx, service_ids=[po.payment])
    approver = template(ctx, "JT-PATH-020")
    result = validate_patch(ctx, approver, scenarios=scenarios)
    assert result.passed
    assert f"tpl:{po.payment}:baseline" in result.rehearsals_fixed
    assert result.rehearsals_regressed == ()
    assert any("실패하던 리허설이 통과합니다" in n for n in result.notes)


def test_memo_is_never_approvable(po: PaymentOrg) -> None:
    ctx = po.org.ctx()
    memo = template(ctx, "JT-ESC-001")
    assert memo.ops == ()
    result = validate_patch(ctx, memo)
    assert not result.passed
    assert result.unresolved_finding_ids == memo.finding_ids
    assert result.score_before == result.score_after
    assert "검토 메모" in result.notes[0]


def test_ops_that_do_not_apply_are_integrity_errors(po: PaymentOrg) -> None:
    result = validate_patch(
        po.org.ctx(), patch_of([RemoveEdge(edge_id="owned_by:service:결제->person:없음")], [])
    )
    assert not result.passed
    assert result.integrity_errors
    assert "그래프에 없습니다" in result.integrity_errors[0]
    assert result.score_after is None


def test_people_outside_the_active_roster_fail_validation(po: PaymentOrg) -> None:
    stranger = po.org.person("홍길동", title="사원", team="결제팀", in_roster=False)
    ctx = po.org.ctx()
    findings = run_checks(ctx)
    ops: list[PatchOp] = [
        AddEdge(edge=link(EdgeKind.OWNED_BY, po.payment, stranger)),
        AddEdge(edge=link(EdgeKind.APPROVED_BY, po.rollback, po.departed)),
    ]
    result = validate_patch(ctx, patch_of(ops, ids_of(findings, "JT-PATH-010")), baseline_findings=findings)
    assert result.unknown_people == (po.departed, stranger)
    assert not result.passed
    assert any("명부에 재직 중으로 없는 인물" in n for n in result.notes)


def test_removing_departed_links_is_not_an_unknown_person(po: PaymentOrg) -> None:
    ctx = po.org.ctx()
    remove = RemoveEdge(edge_id=edge_id(EdgeKind.NOTIFIES, po.alarm, po.departed))
    result = validate_patch(ctx, patch_of([remove], []))
    assert result.unknown_people == ()


def _healthy() -> tuple[Org, dict[str, str]]:
    org = Org()
    owner = org.person("이서연", title="사원", team="결제팀", roles=("1차 당직",))
    lead = org.person("박재현", title="팀장", team="결제팀", roles=("팀장",))
    svc = org.service("결제")
    alarm = org.trigger(svc, "결제 알람", owner)
    org.owner(svc, owner)
    esc = org.escalate(owner, lead)
    rollback = org.action(svc, "결제 롤백", executors=[owner])
    return org, {"owner": owner, "lead": lead, "svc": svc, "alarm": alarm, "esc": esc, "rollback": rollback}


def test_a_new_high_finding_blocks_even_when_the_target_is_resolved() -> None:
    org, ids = _healthy()
    ctx = org.ctx()
    findings = run_checks(ctx)
    ops: list[PatchOp] = [
        AddEdge(edge=link(EdgeKind.APPROVED_BY, ids["rollback"], ids["lead"])),
        RemoveEdge(edge_id=edge_id(EdgeKind.NOTIFIES, ids["alarm"], ids["owner"])),
    ]
    result = validate_patch(ctx, patch_of(ops, ids_of(findings, "JT-PATH-020")), baseline_findings=findings)
    assert result.resolved_finding_ids == tuple(ids_of(findings, "JT-PATH-020"))
    assert any(f.startswith("JT-PATH-002") for f in result.new_finding_ids)
    assert not result.passed
    assert any("높음 이상 심각도의 새 끊김" in n for n in result.notes)


def test_a_rehearsal_that_used_to_pass_must_not_fail() -> None:
    org, ids = _healthy()
    org.edge(EdgeKind.APPROVED_BY, ids["rollback"], ids["lead"])
    ctx = org.ctx()
    findings = run_checks(ctx)
    absent = Scenario(
        id="s-owner-absent",
        title="1차 담당 부재",
        service_id=ids["svc"],
        perturbations=(Perturbation(kind=PerturbationKind.PERSON_UNAVAILABLE, target_id=ids["owner"]),),
    )
    config = SimConfig(default_escalation_after_minutes=5)
    slow = SetAttr(target_id=ids["esc"], key="after_minutes", value=Attr(value=120, source=SourceKind.PATCH))
    result = validate_patch(
        ctx,
        patch_of([slow], ids_of(findings, "JT-ESC-004")),
        scenarios=[absent, absent],
        config=config,
        baseline_findings=findings,
    )
    assert result.resolved_finding_ids == tuple(ids_of(findings, "JT-ESC-004"))
    assert result.rehearsals_regressed == ("s-owner-absent",)
    assert not result.passed
    assert any("무회귀 위반" in n for n in result.notes)


def test_baseline_results_are_reused_only_when_they_match() -> None:
    org, ids = _healthy()
    ctx = org.ctx()
    findings = run_checks(ctx)
    scenario = Scenario(id="s-base", title="기준", service_id=ids["svc"])
    approver = AddEdge(edge=link(EdgeKind.APPROVED_BY, ids["rollback"], ids["lead"]))
    patch = patch_of([approver], ids_of(findings, "JT-PATH-020"))
    plain = validate_patch(ctx, patch, scenarios=[scenario], baseline_findings=findings)
    assert plain.rehearsals_fixed == ("s-base",)
    (real,) = replay_all(ctx.graph, [scenario], policy=ctx.policy, roster=ctx.roster)
    assert real.outcome is Outcome.FAIL
    flipped = real.model_copy(update={"outcome": Outcome.PASS})
    reused = validate_patch(
        ctx, patch, scenarios=[scenario], baseline_results=[flipped], baseline_findings=findings
    )
    assert reused.rehearsals_fixed == ()  # the caller's (matching) baseline was trusted
    foreign = flipped.model_copy(update={"graph_hash": "0" * 64})
    ignored = validate_patch(
        ctx, patch, scenarios=[scenario], baseline_results=[foreign], baseline_findings=findings
    )
    assert ignored.rehearsals_fixed == ("s-base",)


def test_a_rekeyed_gap_is_not_counted_as_resolved() -> None:
    org, ids = _healthy()
    silent = org.person("유령", title="사원", team="결제팀", channels=())
    second = org.action(ids["svc"], "결제 페일오버", executors=[ids["owner"]])
    ctx = org.ctx()
    findings = run_checks(ctx)
    targets = [f for f in findings if f.rule_id == "JT-PATH-020"]
    assert len(targets) == 2
    claim = next(f for f in targets if ids["rollback"] in f.subject_ids)
    unreachable = AddEdge(edge=link(EdgeKind.APPROVED_BY, ids["rollback"], silent))
    result = validate_patch(ctx, patch_of([unreachable], [claim.id]), baseline_findings=findings)
    assert result.resolved_finding_ids == ()
    assert result.unresolved_finding_ids == (claim.id,)
    assert result.new_finding_ids == ()
    assert any(claim.id in n and "→" in n for n in result.notes)
    other = next(f for f in targets if second in f.subject_ids)
    assert not any(other.id in n for n in result.notes)  # the untouched action's gap is not conflated
    assert not result.passed


def test_claims_unknown_to_the_base_are_reported(po: PaymentOrg) -> None:
    ctx = po.org.ctx()
    owner = template(ctx, "JT-PPL-001")
    stale = owner.model_copy(update={"finding_ids": (*owner.finding_ids, "JT-PATH-999:nope")})
    result = validate_patch(ctx, stale)
    assert "JT-PATH-999:nope" in result.unresolved_finding_ids
    assert any("기준 그래프에서 찾을 수 없는 끊김" in n for n in result.notes)
    assert result.passed  # the real targets were resolved


def test_validation_is_deterministic(po: PaymentOrg) -> None:
    ctx = po.org.ctx()
    patch = template(ctx, "JT-PATH-020")
    scenarios = template_scenarios(ctx)
    assert validate_patch(ctx, patch, scenarios=scenarios) == validate_patch(ctx, patch, scenarios=scenarios)
    assert any(n.startswith("전체 준비도") for n in validate_patch(ctx, patch).notes)
