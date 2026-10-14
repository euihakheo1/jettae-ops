"""Demo acceptance: the dmc-shop 결제 path is repaired patch by patch, then reverted.

Flow (LLM OFF): ingest + rule extraction + assembly → version 1 → for each fix in the demo script
(owner replacement, escalation, approver, fallback channel) re-diagnose the head, draft, validate,
approve → the 결제 score rises monotonically to ≥ 80, the human scenario '추석 연휴 새벽 결제 장애'
(or the template baseline when the scenario loader is unavailable) goes FAIL → PASS, nobody outside
the roster appears, and reverting to version 1 restores the original content hash.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

import jettae.rehearse.scenarios as scenario_module
from jettae.check.context import CheckContext
from jettae.check.registry import run_checks
from jettae.check.scoring import score
from jettae.domain import Finding, FixKind, Outcome, Patch, PatchStatus, Scenario
from jettae.improve import TemplateDrafter, VersionService, validate_patch
from jettae.rehearse.scenarios import template_scenarios
from jettae.rehearse.simulator import replay
from jettae.store import MemoryStore
from test_improve_support import EXAMPLE, demo_ctx, demo_inputs

WS = "dmc-shop"
PAYMENT = "service:결제"
CLOCK = datetime(2026, 10, 28, 14, 0, tzinfo=UTC)

#: The demo script's order; each entry lists the fix kinds a 결제 patch of that step answers.
STEPS: tuple[tuple[str, frozenset[FixKind]], ...] = (
    ("owner replacement", frozenset({FixKind.REPLACE_DEPARTED, FixKind.ASSIGN_OWNER})),
    ("escalation", frozenset({FixKind.ADD_ESCALATION})),
    ("approver", frozenset({FixKind.ASSIGN_APPROVER})),
    ("fallback channel", frozenset({FixKind.ADD_FALLBACK_CHANNEL})),
)


def human_scenario(ctx: CheckContext) -> Scenario | None:
    """'추석 연휴 새벽 결제 장애' from scenarios.yaml, if the loader exists in this build."""
    loader = getattr(scenario_module, "scenarios_from_yaml", None)
    if loader is None:
        return None
    loaded: list[Scenario] = loader(EXAMPLE / "scenarios.yaml", ctx)
    return next((s for s in loaded if s.id.endswith("dmc-s1-chuseok-payment")), None)


def payment_score(ctx: CheckContext, findings: list[Finding]) -> float:
    return next(s.score for s in score(ctx, findings).services if s.service_id == PAYMENT)


def step_patch(patches: list[Patch], findings: list[Finding], kinds: frozenset[FixKind]) -> Patch | None:
    by_id = {f.id: f for f in findings}
    for patch in patches:
        targets = [by_id[i] for i in patch.finding_ids]
        if patch.ops and any(
            f.service_id == PAYMENT and f.fix is not None and f.fix.kind in kinds for f in targets
        ):
            return patch
    return None


def test_payment_path_is_repaired_step_by_step_and_reverted() -> None:
    bundle, graph = demo_inputs()
    store = MemoryStore(clock=lambda: CLOCK)
    store.workspaces.create_workspace("별빛상점", workspace_id=WS)
    versions = VersionService(store.graphs, store.patches, store.audit, clock=lambda: CLOCK)
    first = versions.commit_graph(WS, graph, actor="jettae", note="규칙 추출(LLM OFF)")

    base_ctx = demo_ctx(bundle, graph)
    human = human_scenario(base_ctx)
    tracked = human or next(
        s for s in template_scenarios(base_ctx, service_ids=[PAYMENT]) if s.id.endswith("baseline")
    )
    assert replay(graph, tracked, policy=bundle.policy, roster=bundle.roster).outcome is Outcome.FAIL

    scores = [payment_score(base_ctx, run_checks(base_ctx))]
    applied: list[str] = []
    fixed: set[str] = set()
    for name, kinds in STEPS:
        head, current = versions.head(WS)
        ctx = demo_ctx(bundle, current)
        findings = run_checks(ctx)
        patch = step_patch(
            TemplateDrafter().draft(ctx, findings, workspace_id=WS, base_version=head.number), findings, kinds
        )
        if patch is None:
            # an earlier step already closed this gap (the new owner brought two independent media)
            assert name == "fallback channel"
            assert not any(
                f.service_id == PAYMENT and f.fix is not None and f.fix.kind in kinds for f in findings
            )
            continue
        scenarios = [tracked, *template_scenarios(ctx, service_ids=[PAYMENT])]
        result = validate_patch(ctx, patch, scenarios=scenarios, baseline_findings=findings)
        assert result.passed, (name, result.notes)
        assert result.unknown_people == ()  # KPI: 명부 외 인물 생성 0건
        assert result.rehearsals_regressed == ()
        fixed.update(result.rehearsals_fixed)
        store.patches.save_patch(
            patch.model_copy(update={"status": PatchStatus.VALIDATED, "validation": result})
        )
        version = versions.approve(patch.id, actor="박재현", role="lead")
        assert version.number == head.number + 1
        applied.append(name)
        after_ctx = demo_ctx(bundle, versions.head(WS)[1])
        scores.append(payment_score(after_ctx, run_checks(after_ctx)))

    assert applied[:3] == ["owner replacement", "escalation", "approver"]
    assert scores == sorted(scores)
    assert len(set(scores)) == len(scores)
    assert scores[0] < 30
    assert scores[-1] >= 80
    assert tracked.id in fixed
    _, final = versions.head(WS)
    assert replay(final, tracked, policy=bundle.policy, roster=bundle.roster).outcome is Outcome.PASS

    reverted = versions.revert(WS, 1, actor="박재현", role="lead")
    assert reverted.graph_hash == first.graph_hash == graph.content_hash()
    assert reverted.revert_of == 1
    assert [e.action for e in store.audit.entries(WS)].count("patch.approved") == len(applied)


def test_human_scenario_is_the_chuseok_drill_when_the_loader_exists() -> None:
    bundle, graph = demo_inputs()
    scenario = human_scenario(demo_ctx(bundle, graph))
    if scenario is None:
        pytest.skip("rehearse.scenarios.scenarios_from_yaml is not available in this build")
    assert scenario.service_id == PAYMENT
    assert "추석" in scenario.title
