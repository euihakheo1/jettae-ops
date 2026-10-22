"""Risk exploration (규칙 밖 위험 탐색): sweep, AI proposals, verification, pipeline, reports.

What must hold:

* the sweep is LLM-free and deterministic, searches only critical services whose baseline replay
  passes, reports *minimal* breaking sets (no strict subset breaks, compared by what the replay
  loses — a concurrent incident is the person it pulls away), leaves out what a fixed rule already
  reports, and says so whenever a cap cut the search;
* AI proposals may only choose among enumerated, confirmed node ids — anything else is dropped
  whole — never see document text, and a premise the graph lacks keeps a hypothesis ``needs_info``;
  a proposal (new or carried over) that a breaking set already found accounts for is not reported
  as a new risk;
* verification uses the service's policy deadline and is the only source of a status; a held
  (판단 보류) target keeps a hypothesis ``needs_info``;
* findings, readiness, rehearsals, drafts and the judgement path's LLM calls are identical with and
  without exploration;
* every report format shows the hypotheses apart from the score.
"""

from __future__ import annotations

import asyncio
import itertools
import json
from collections.abc import Iterator
from typing import Any

import pytest
from pydantic import BaseModel

from jettae.check.context import CheckContext
from jettae.cli.output import scan_payload
from jettae.domain import (
    AiProposalStatus,
    DiagnosisReport,
    ExplorationSummary,
    HypothesisOrigin,
    HypothesisStatus,
    IncidentType,
    InputBundle,
    LlmMode,
    LlmUsageSummary,
    NodeKind,
    OpsGraph,
    Outcome,
    Patch,
    PatchStatus,
    Perturbation,
    PerturbationKind,
    RiskHypothesis,
    Scenario,
    Verification,
)
from jettae.explore import (
    EXPLORE_TASK,
    HypothesisDraft,
    LlmRiskPayload,
    SweepLimits,
    explore_graph,
    perturbation_effect,
    propose_hypotheses,
    set_key,
    sweep,
    verify_hypotheses,
)
from jettae.explore.sweep import HELD_SERVICE_REASON
from jettae.ingest import load_bundle_dir
from jettae.llm.base import LLMClient, LLMDisabledError, LLMError, NullLLM, UntrustedDocument
from jettae.llm.fake import FakeLLM, FakeOutcome
from jettae.orchestrator import DiagnosisOptions, DiagnosisOutcome, run_diagnosis
from jettae.orchestrator.pipeline import build_context
from jettae.rehearse.simulator import replay
from jettae.reporting import build_evidence, render_cli_summary, render_view_html, render_view_markdown
from jettae.reporting.evidence import AI_EXPLANATION_PREFIX
from test_orchestrator_support import (
    DEMO,
    PAYMENT,
    PAYMENT_QUOTE,
    SHARED_NETWORK,
    event_loop_runner,
    factory_of,
    failing_llm,
    fake_llm,
    new_service,
)

ORDER = "service:주문"
FRONTEND = "service:웹-프론트엔드"
CATALOG = "service:상품-카탈로그"
PARK = "person:박재현"  # 상품 카탈로그 1차 대응자, 웹 프론트엔드·장바구니·환율 롤백의 유일한 승인자
DRILLS = (DEMO / "scenarios.yaml").read_bytes()


@pytest.fixture(scope="module")
def runner() -> Iterator[asyncio.Runner]:
    with event_loop_runner() as loop:
        yield loop


@pytest.fixture(scope="module")
def bundle() -> InputBundle:
    return load_bundle_dir(DEMO)


def diagnose(
    runner: asyncio.Runner,
    bundle: InputBundle,
    *,
    llm: LLMClient | None = None,
    mode: LlmMode = LlmMode.OFF,
    **options: Any,
) -> DiagnosisOutcome:
    return runner.run(
        run_diagnosis(
            bundle,
            base_graph=None,
            llm=llm if llm is not None else NullLLM(),
            llm_mode=mode,
            options=DiagnosisOptions(**options),
        )
    )


@pytest.fixture(scope="module")
def off(runner: asyncio.Runner, bundle: InputBundle) -> DiagnosisOutcome:
    return diagnose(runner, bundle, scenario_sources=(DRILLS,))


@pytest.fixture(scope="module")
def ctx(off: DiagnosisOutcome, bundle: InputBundle) -> CheckContext:
    return build_context(
        off.graph, roster=bundle.roster, oncall=bundle.oncall, catalog=bundle.catalog, policy=bundle.policy
    )


def targets(h: RiskHypothesis | HypothesisDraft) -> set[str]:
    return {p.target_id for p in h.perturbations}


def of_service(hypotheses: tuple[RiskHypothesis, ...], service_id: str) -> list[RiskHypothesis]:
    return [h for h in hypotheses if h.service_id == service_id]


def ask(payload: dict[str, Any] | Exception) -> FakeLLM:
    """A fake model that answers every ``explore`` request with ``payload``."""

    def handler(
        task: str, schema: type[BaseModel], documents: tuple[UntrustedDocument, ...], context: str
    ) -> FakeOutcome:
        assert task == EXPLORE_TASK
        return payload

    return FakeLLM(handler)


def proposal(
    service_id: str, *pairs: tuple[str, str], assumptions: list[str] | None = None, **extra: Any
) -> dict[str, Any]:
    return {
        "service_id": service_id,
        "title": extra.get("title", "제안"),
        "explanation": extra.get("explanation", "설명"),
        "incident_type": "outage",
        "perturbations": [{"kind": kind, "target_id": target} for kind, target in pairs],
        "assumptions": assumptions or [],
    }


# --------------------------------------------------------------------------- sweep (LLM 0회)


def test_sweep_skips_services_that_already_fail_without_any_perturbation(off: DiagnosisOutcome) -> None:
    summary = off.exploration
    assert summary is not None
    by_service = {s.service_id: s for s in summary.services}
    payment = by_service[PAYMENT]  # 담당이 퇴사자 김민준: the rules already report the break
    assert payment.baseline is Outcome.FAIL
    assert payment.skipped_reason is not None
    assert "교란 없이도" in payment.skipped_reason
    assert payment.candidates == 0
    assert not of_service(off.risk_hypotheses, PAYMENT)
    assert by_service[ORDER].baseline is Outcome.PASS
    assert by_service[ORDER].candidates > 0


def test_sweep_rediscovers_the_hand_written_slack_drill_as_a_minimal_pair(off: DiagnosisOutcome) -> None:
    """S3 (슬랙 장애 + 정민혁 연차) was written by hand; the sweep finds it without being told."""
    order = of_service(off.risk_hypotheses, ORDER)
    sets = {frozenset(targets(h)) for h in order}
    assert frozenset({"person:정민혁", "channel:slack"}) in sets
    assert frozenset({"channel:phone", "channel:slack"}) in sets
    pair = next(h for h in order if targets(h) == {"person:정민혁", "channel:slack"})
    assert pair.origin is HypothesisOrigin.SWEEP
    assert pair.status is HypothesisStatus.CONFIRMED
    assert pair.verification.outcome is Outcome.FAIL
    assert pair.verification.baseline_outcome is Outcome.PASS
    assert pair.verification.failed_at_id == "person:최영훈"
    assert pair.title == "주문 · 정민혁 부재 + 슬랙 다운"
    assert pair.assumptions == ()
    assert pair.required_info == ()
    assert any("15분 미응답" in span.quote for span in pair.evidence)


def test_what_the_rules_already_report_is_covered_not_proposed_again(off: DiagnosisOutcome) -> None:
    summary = off.exploration
    assert summary is not None
    covered = {(c.service_id, frozenset(c.target_ids)): c.finding_ids for c in summary.covered}
    # 한서윤 alone approves the 주문 rollback: JT-SPF-001 says so already.
    assert any(f.startswith("JT-SPF-001:") for f in covered[(ORDER, frozenset({"person:한서윤"}))])
    # 정민혁 ↔ 최영훈 escalate to each other only: the cycle rule names both.
    assert any(
        f.startswith("JT-ESC-") for f in covered[(ORDER, frozenset({"person:정민혁", "person:최영훈"}))]
    )
    # 박재현 alone approves the 웹 프론트엔드 rollback: JT-SPF-001 again.
    assert any(f.startswith("JT-SPF-001:") for f in covered[(FRONTEND, frozenset({PARK}))])
    proposed = {(h.service_id, frozenset(targets(h))) for h in off.risk_hypotheses}
    assert not proposed & set(covered)
    # What the sweep reports on the demo is the two 주문 pairs, nothing a rule already reports.
    assert sorted(h.title for h in off.risk_hypotheses) == [
        "주문 · 전화 다운 + 슬랙 다운",
        "주문 · 정민혁 부재 + 슬랙 다운",
    ]


def test_every_sweep_hypothesis_is_minimal(off: DiagnosisOutcome, ctx: CheckContext) -> None:
    sweep_hypotheses = [h for h in off.risk_hypotheses if h.origin is HypothesisOrigin.SWEEP]
    assert sweep_hypotheses
    for h in sweep_hypotheses:
        for size in range(len(h.perturbations)):
            for subset in itertools.combinations(h.perturbations, size):
                result = replay(
                    ctx.graph,
                    Scenario(id="subset", title="subset", service_id=h.service_id, perturbations=subset),
                    policy=ctx.policy,
                    roster=ctx.roster,
                )
                assert result.outcome is Outcome.PASS, (h.title, subset)
    # minimal by effect too: no two hypotheses of a service take down the same people/channels, or
    # one a subset of the other's — nor does any contain a set the rules already report
    effects = {h.id: perturbation_effect(ctx, h.perturbations) for h in sweep_hypotheses}
    for a, b in itertools.permutations(sweep_hypotheses, 2):
        if a.service_id == b.service_id:
            assert not effects[a.id] <= effects[b.id], (a.title, b.title)
    assert off.exploration is not None
    for cut in off.exploration.covered:
        reported = perturbation_effect(ctx, [as_perturbation(ctx, t) for t in cut.target_ids])
        for h in of_service(tuple(sweep_hypotheses), cut.service_id):
            assert not reported <= effects[h.id], (h.title, cut.target_ids)


KIND_OF: dict[NodeKind, PerturbationKind] = {
    NodeKind.PERSON: PerturbationKind.PERSON_UNAVAILABLE,
    NodeKind.CHANNEL: PerturbationKind.CHANNEL_DOWN,
    NodeKind.SYSTEM: PerturbationKind.SYSTEM_DOWN,
    NodeKind.LOCATION: PerturbationKind.LOCATION_DOWN,
    NodeKind.SERVICE: PerturbationKind.CONCURRENT_INCIDENT,
}


def as_perturbation(ctx: CheckContext, target_id: str) -> Perturbation:
    return Perturbation(kind=KIND_OF[ctx.graph.nodes[target_id].kind], target_id=target_id)


CATALOG_BUSY = Perturbation(kind=PerturbationKind.CONCURRENT_INCIDENT, target_id=CATALOG)
PARK_AWAY = Perturbation(kind=PerturbationKind.PERSON_UNAVAILABLE, target_id=PARK)


def test_a_concurrent_incident_counts_as_the_person_it_pulls_away(
    off: DiagnosisOutcome, ctx: CheckContext
) -> None:
    """A 상품 카탈로그 incident pulls 박재현 away — the replay breaks exactly as when he is absent, a
    single point JT-SPF-001 already reports. It is that break again, not a risk outside the rules."""
    assert perturbation_effect(ctx, [CATALOG_BUSY]) == perturbation_effect(ctx, [PARK_AWAY]) == {PARK}
    busy, away = (
        replay(
            ctx.graph,
            Scenario(id=f"s{i}", title="s", service_id=FRONTEND, perturbations=(p,)),
            policy=ctx.policy,
            roster=ctx.roster,
        )
        for i, p in enumerate((CATALOG_BUSY, PARK_AWAY))
    )
    assert busy.outcome is away.outcome is Outcome.FAIL
    assert busy.failed_at_id == away.failed_at_id == PARK
    assert busy.path_ids == away.path_ids
    # with 박재현 already absent, the incident pulls the next 상품 카탈로그 first responder instead
    pair = perturbation_effect(ctx, [PARK_AWAY, CATALOG_BUSY])
    assert PARK in pair
    assert len(pair) == 2
    for service in (FRONTEND, "service:장바구니", "service:환율"):
        assert not [h for h in of_service(off.risk_hypotheses, service) if CATALOG_BUSY in h.perturbations]


def test_sweep_is_deterministic_and_llm_free(ctx: CheckContext, off: DiagnosisOutcome) -> None:
    first = explore_graph(ctx, off.findings)
    second = explore_graph(ctx, off.findings)
    assert [h.model_dump() for h in first.hypotheses] == [h.model_dump() for h in second.hypotheses]
    assert first.summary == second.summary
    assert [h.id for h in first.hypotheses] == [h.id for h in off.risk_hypotheses]
    assert all(h.id.startswith("hyp-") for h in first.hypotheses)


def test_caps_are_recorded_never_silent(ctx: CheckContext, off: DiagnosisOutcome) -> None:
    capped = sweep(ctx, off.findings, limits=SweepLimits(max_candidates=3))
    assert any("(상한)" in note for note in capped.notes)
    order = next(s for s in capped.services if s.service_id == ORDER)
    assert order.candidates == 3 < order.candidates_total
    budget = sweep(ctx, off.findings, limits=SweepLimits(max_replays=12))
    assert budget.replays <= 12
    assert any("리허설 예산" in note for note in budget.notes)
    tiny = sweep(ctx, off.findings, limits=SweepLimits(max_replays=3))
    # every critical service is listed, even the ones whose baseline the budget never reached
    assert len(tiny.services) == len(ctx.critical_services())
    unreplayed = [s for s in tiny.services if s.baseline is None]
    assert len(unreplayed) == len(ctx.critical_services()) - 3
    assert all(s.replays == 0 and s.skipped_reason for s in unreplayed)
    # … in the critical services' own order, the unreplayed ones last
    assert [s.service_id for s in tiny.services] == [s.id for s in ctx.critical_services()]
    assert [s.baseline is None for s in tiny.services] == sorted(s.baseline is None for s in tiny.services)
    one = sweep(ctx, off.findings, limits=SweepLimits(max_hypotheses=1))
    assert len(one.drafts) == 1
    assert any("1개만 실었습니다(상한)" in note for note in one.notes)
    singles = sweep(ctx, off.findings, limits=SweepLimits(max_set_size=1))
    assert all(len(d.perturbations) == 1 for d in singles.drafts)
    with pytest.raises(ValueError, match="max_set_size"):
        SweepLimits(max_set_size=4)


# --------------------------------------------------------------------------- verification


def draft(
    service_id: str, *perturbations: Perturbation, assumptions: tuple[str, ...] = ()
) -> HypothesisDraft:
    return HypothesisDraft(
        origin=HypothesisOrigin.AI,
        service_id=service_id,
        title="가설",
        explanation="설명",
        perturbations=perturbations,
        assumptions=assumptions,
    )


SLACK_DOWN = Perturbation(kind=PerturbationKind.CHANNEL_DOWN, target_id="channel:slack")
JEONG_AWAY = Perturbation(kind=PerturbationKind.PERSON_UNAVAILABLE, target_id="person:정민혁")


def held(ctx: CheckContext, node_id: str) -> CheckContext:
    """``ctx`` with one node put on hold (판단 보류) — as a re-diagnosis could leave it."""
    nodes = dict(ctx.graph.nodes)
    nodes[node_id] = nodes[node_id].model_copy(update={"verification": Verification.HOLD})
    return ctx.with_graph(OpsGraph(nodes=nodes, edges=ctx.graph.edges, docs=ctx.graph.docs))


def test_a_held_target_keeps_a_hypothesis_at_needs_info(ctx: CheckContext) -> None:
    """The replay leaves held facts out: a perturbation of a held node does nothing there, so the
    verdict would rest on an unchecked fact — never refuted, never confirmed (P3)."""
    jeong_held = held(ctx, "person:정민혁")
    [pair] = verify_hypotheses(jeong_held, [draft(ORDER, JEONG_AWAY, SLACK_DOWN)])
    assert pair.verification.outcome is Outcome.FAIL  # slack alone breaks once 정민혁 is out of the graph
    assert pair.status is HypothesisStatus.NEEDS_INFO
    assert any(r.startswith("판단 보류 확인: '정민혁'") for r in pair.required_info)
    catalog_held = held(ctx, CATALOG)
    [busy] = verify_hypotheses(catalog_held, [draft(FRONTEND, CATALOG_BUSY)])
    assert busy.verification.outcome is Outcome.PASS  # the incident pulls nobody in the replay
    assert busy.status is HypothesisStatus.NEEDS_INFO  # … which is not a refutation
    assert any("상품 카탈로그" in r for r in busy.required_info)


def test_a_held_critical_service_is_listed_as_not_searched(ctx: CheckContext, off: DiagnosisOutcome) -> None:
    """The replay leaves a held (판단 보류) service out: its baseline "failure" would only mean "not
    confirmed", so it is not worded as a break the rules report (P3) — and not searched."""
    result = sweep(held(ctx, ORDER), off.findings)
    order = next(s for s in result.services if s.service_id == ORDER)
    assert (order.baseline, order.skipped_reason, order.replays) == (None, HELD_SERVICE_REASON, 0)
    assert not [d for d in result.drafts if d.service_id == ORDER]
    assert "판단 보류(원문 대조 전) 서비스라 탐색하지 못했습니다: 주문" in result.notes
    assert [s.service_id for s in result.services] == [s.id for s in ctx.critical_services()]


def test_verification_statuses_follow_the_replay(ctx: CheckContext) -> None:
    hypotheses = verify_hypotheses(
        ctx,
        [
            draft(ORDER, SLACK_DOWN),  # 정민혁 still answers by phone → refuted
            draft(ORDER, JEONG_AWAY, SLACK_DOWN),  # S3 → confirmed
            draft(ORDER, SLACK_DOWN, assumptions=("슬랙과 전화가 같은 망을 쓴다",)),  # premise → needs_info
            draft(PAYMENT, SLACK_DOWN),  # 결제 fails anyway → needs_info, not confirmed
            draft(ORDER, Perturbation(kind=PerturbationKind.PERSON_UNAVAILABLE, target_id="person:홍길동")),
        ],
    )
    status = {(h.service_id, frozenset(targets(h)), h.assumptions): h.status for h in hypotheses}
    assert status == {
        (ORDER, frozenset({"channel:slack"}), ()): HypothesisStatus.REFUTED,
        (ORDER, frozenset({"person:정민혁", "channel:slack"}), ()): HypothesisStatus.CONFIRMED,
        (ORDER, frozenset({"channel:slack"}), ("슬랙과 전화가 같은 망을 쓴다",)): HypothesisStatus.NEEDS_INFO,
        (PAYMENT, frozenset({"channel:slack"}), ()): HypothesisStatus.NEEDS_INFO,
    }
    # confirmed first, then needs_info, refuted last
    assert [h.status for h in hypotheses] == sorted(
        (h.status for h in hypotheses),
        key=[HypothesisStatus.CONFIRMED, HypothesisStatus.NEEDS_INFO, HypothesisStatus.REFUTED].index,
    )
    premise = next(h for h in hypotheses if h.assumptions)
    assert premise.required_info == ("전제 확인: 슬랙과 전화가 같은 망을 쓴다",)
    assert premise.verification.outcome is Outcome.PASS  # what the replay shows is still recorded
    payment = next(h for h in hypotheses if h.service_id == PAYMENT)
    assert payment.verification.baseline_outcome is Outcome.FAIL
    assert "교란 없이도" in payment.required_info[0]
    for h in hypotheses:  # the deadline is the service's policy SLO, never the proposer's
        assert h.verification.deadline_minutes == ctx.slo_minutes(h.service_id)


# --------------------------------------------------------------------------- AI explorer


def test_ai_keeps_valid_ids_and_drops_everything_else(runner: asyncio.Runner, ctx: CheckContext) -> None:
    payload = {
        "hypotheses": [
            proposal(ORDER, ("person_unavailable", "person:최영훈"), ("channel_down", "channel:phone")),
            proposal(ORDER, ("person_unavailable", "person:홍길동")),  # unknown id
            proposal(ORDER, ("channel_down", "person:최영훈")),  # wrong kind
            proposal(ORDER, ("concurrent_incident", ORDER)),  # its own service
            proposal("service:없는-서비스", ("channel_down", "channel:slack")),
            proposal(ORDER, *[("channel_down", "channel:slack")] * 5),  # too many perturbations
            proposal(ORDER, ("person_unavailable", "person:정민혁"), ("channel_down", "channel:slack")),
        ]
    }
    model = ask(payload)
    known = sweep(ctx, ())
    result = runner.run(propose_hypotheses(model, ctx, (), n=5, known=known))
    assert result.used
    assert result.status is AiProposalStatus.USED
    assert result.proposed == 7
    # the last one repeats a breaking set the sweep already found (without a premise): a duplicate
    assert [targets(d) for d in result.drafts] == [{"person:최영훈", "channel:phone"}]
    assert (result.dropped, result.duplicates, result.over_cap) == (5, 1, 0)
    [call] = model.calls
    assert call.task == EXPLORE_TASK
    assert call.documents == ()


def test_ai_cannot_pass_a_known_break_off_as_a_new_risk(
    runner: asyncio.Runner, ctx: CheckContext, off: DiagnosisOutcome
) -> None:
    """A known set plus an extra target, or a concurrent incident that only pulls away a known single
    point, breaks *because of* what the sweep or a rule already reports — not a new risk."""
    model = ask(
        {
            "hypotheses": [
                proposal(FRONTEND, ("person_unavailable", PARK), ("channel_down", "channel:slack")),
                proposal(FRONTEND, ("concurrent_incident", CATALOG)),
                proposal(
                    ORDER,
                    ("person_unavailable", "person:정민혁"),
                    ("channel_down", "channel:slack"),
                    ("channel_down", "channel:phone"),
                ),
                proposal(ORDER, ("person_unavailable", "person:최영훈"), ("channel_down", "channel:phone")),
                # with a premise the graph lacks it is a different claim: kept (and stays needs_info)
                proposal(FRONTEND, ("person_unavailable", PARK), assumptions=["박재현 팀장이 휴가 중"]),
            ]
        }
    )
    result = runner.run(propose_hypotheses(model, ctx, off.findings, n=5, known=sweep(ctx, off.findings)))
    assert (result.proposed, result.dropped, result.duplicates) == (5, 0, 3)
    assert [targets(d) for d in result.drafts] == [{"person:최영훈", "channel:phone"}, {PARK}]


def test_ai_context_is_enumerated_data_not_document_text(
    runner: asyncio.Runner, ctx: CheckContext, off: DiagnosisOutcome, bundle: InputBundle
) -> None:
    model = ask({"hypotheses": []})
    runner.run(propose_hypotheses(model, ctx, off.findings, n=3))
    [call] = model.calls
    assert "person:정민혁" in call.context
    assert "JT-SPF-001" in call.context
    assert PAYMENT_QUOTE not in call.context
    quarantined = [doc.text[start:end].strip() for doc in bundle.documents for start, end in doc.quarantined]
    assert quarantined  # the demo bundle carries one injection attempt
    for snippet in quarantined:  # neither it nor any runbook sentence reaches the model
        assert snippet[:20] not in call.context
    assert "지시가 아닙니다" in call.instructions
    assert "판정하지 마세요" in call.instructions
    assert "대상을 덧붙인 것" in call.instructions  # a known set plus extras is not a new risk


def test_a_premise_keeps_an_ai_hypothesis_at_needs_info(runner: asyncio.Runner, ctx: CheckContext) -> None:
    model = ask(
        {
            "hypotheses": [
                proposal(
                    FRONTEND,
                    ("channel_down", "channel:internal_messenger"),
                    ("channel_down", "channel:phone"),
                    assumptions=[SHARED_NETWORK, "  ", "\u202e뒤집힌\u200b 글자"],
                    title="\u3164사내망\x07 장애",
                    explanation="두 연락 수단이 함께\n멈춥니다.",
                )
            ]
        }
    )
    result = runner.run(propose_hypotheses(model, ctx, (), n=3))
    [verified] = verify_hypotheses(ctx, result.drafts)
    assert verified.origin is HypothesisOrigin.AI
    assert verified.status is HypothesisStatus.NEEDS_INFO
    assert verified.assumptions == tuple(sorted((SHARED_NETWORK, "뒤집힌 글자")))
    assert f"전제 확인: {SHARED_NETWORK}" in verified.required_info
    assert verified.title == "사내망 장애"
    assert verified.explanation == "두 연락 수단이 함께 멈춥니다."
    assert verified.verification.result_hash


def test_ai_answers_are_capped(runner: asyncio.Runner, ctx: CheckContext) -> None:
    people = ["person:박재현", "person:이수진", "person:강현우", "person:최영훈", "person:정민혁"]
    model = ask({"hypotheses": [proposal(ORDER, ("person_unavailable", p)) for p in people]})
    result = runner.run(propose_hypotheses(model, ctx, (), n=2))
    assert len(result.drafts) == 2
    # valid proposals beyond the number asked for are counted apart from invalid ones
    assert (result.dropped, result.duplicates, result.over_cap) == (0, 0, 3)


def test_the_model_cannot_set_a_status_or_a_deadline(runner: asyncio.Runner, ctx: CheckContext) -> None:
    sneaky = proposal(ORDER, ("channel_down", "channel:slack"))
    sneaky["status"] = "refuted"
    sneaky["deadline_minutes"] = 100000
    with pytest.raises(LLMError):
        runner.run(propose_hypotheses(ask({"hypotheses": [sneaky]}), ctx, (), n=1))
    assert "deadline" not in json.dumps(LlmRiskPayload.model_json_schema())
    assert "status" not in json.dumps(LlmRiskPayload.model_json_schema())


def test_a_disabled_model_proposes_nothing(runner: asyncio.Runner, ctx: CheckContext) -> None:
    result = runner.run(propose_hypotheses(ask(LLMDisabledError("off")), ctx, (), n=3))
    assert (result.drafts, result.used, result.status) == ((), False, AiProposalStatus.OFF)
    nothing = FakeLLM(lambda *_: {"hypotheses": []})
    unasked = runner.run(propose_hypotheses(nothing, ctx, (), n=0))
    assert (unasked.used, unasked.status) == (False, AiProposalStatus.NOT_REQUESTED)
    assert nothing.calls == []


# --------------------------------------------------------------------------- pipeline


def test_score_findings_and_gate_do_not_depend_on_exploration(
    runner: asyncio.Runner, bundle: InputBundle, off: DiagnosisOutcome
) -> None:
    without = diagnose(runner, bundle, scenario_sources=(DRILLS,), explore=False)
    assert without.risk_hypotheses == ()
    assert without.exploration is None
    assert off.risk_hypotheses
    assert without.findings == off.findings
    assert without.readiness == off.readiness
    assert [r.result_hash() for r in without.rehearsals] == [r.result_hash() for r in off.rehearsals]
    assert [p.model_dump() for p in without.patches] == [p.model_dump() for p in off.patches]
    assert without.usage == off.usage
    assert off.usage.judgement_path_calls == 0
    no_drills = diagnose(runner, bundle, rehearse=False, draft=False)
    assert (no_drills.risk_hypotheses, no_drills.exploration) == ((), None)


def test_live_ai_hypotheses_are_metered_apart_and_change_no_verdict(
    runner: asyncio.Runner, bundle: InputBundle
) -> None:
    with_ai = diagnose(runner, bundle, llm=fake_llm(), mode=LlmMode.LIVE, draft=False)
    sweep_only = diagnose(
        runner, bundle, llm=fake_llm(), mode=LlmMode.LIVE, draft=False, llm_hypothesis_count=0
    )
    assert with_ai.findings == sweep_only.findings
    assert with_ai.readiness == sweep_only.readiness
    assert with_ai.usage.by_task["explore"] == 1
    assert "explore" not in sweep_only.usage.by_task
    assert with_ai.usage.judgement_path_calls == sweep_only.usage.judgement_path_calls == 0
    [ai] = [h for h in with_ai.risk_hypotheses if h.origin is HypothesisOrigin.AI]
    assert ai.status is HypothesisStatus.NEEDS_INFO
    assert ai.assumptions == (SHARED_NETWORK,)
    summary = with_ai.exploration
    assert summary is not None
    assert (summary.ai_used, summary.ai_proposed, summary.ai_kept, summary.ai_dropped) == (True, 2, 1, 1)
    assert summary.ai_status is AiProposalStatus.USED
    assert sweep_only.exploration is not None
    assert sweep_only.exploration.ai_status is AiProposalStatus.NOT_REQUESTED
    sweep_ids = [h.id for h in sweep_only.risk_hypotheses]
    assert [h.id for h in with_ai.risk_hypotheses if h.origin is HypothesisOrigin.SWEEP] == sweep_ids
    # the model's wording is shown, but marked as unverified: the replay checked the perturbations only
    section = render_view_markdown(build_evidence(report_of(with_ai), with_ai.graph)).split("## 9. ", 1)[1]
    assert f"{AI_EXPLANATION_PREFIX}사내 메신저와 사내 전화가" in section
    assert "생성형 AI 제안 2건 중 1건 검증(목록에 없는 id·판단 보류·종류 불일치 등 1건 버림)" in section


def test_an_ai_failure_keeps_the_sweep_and_says_so(runner: asyncio.Runner, bundle: InputBundle) -> None:
    outcome = diagnose(
        runner, bundle, llm=failing_llm(EXPLORE_TASK, LLMError("timeout")), mode=LlmMode.LIVE, draft=False
    )
    assert outcome.risk_hypotheses
    assert all(h.origin is HypothesisOrigin.SWEEP for h in outcome.risk_hypotheses)
    [note] = [n for n in outcome.notes if n.startswith("AI 위험 가설을 만들지 못해")]
    assert "timeout" in note
    assert outcome.exploration is not None
    assert note in outcome.exploration.notes
    assert outcome.exploration.ai_status is AiProposalStatus.FAILED
    section = render_view_markdown(build_evidence(report_of(outcome), outcome.graph)).split("## 9. ", 1)[1]
    assert "생성형 AI 제안 실패(사유는 아래 참고)" in section
    assert "LLM 끔" not in section


def test_ai_hypotheses_are_verified_again_on_a_later_graph(
    runner: asyncio.Runner, bundle: InputBundle, ctx: CheckContext, off: DiagnosisOutcome
) -> None:
    live = diagnose(runner, bundle, llm=fake_llm(), mode=LlmMode.LIVE, draft=False)
    earlier = [h for h in live.risk_hypotheses if h.origin is HypothesisOrigin.AI]
    again = explore_graph(ctx, off.findings, previous=[*earlier, *off.risk_hypotheses])
    assert [h.id for h in again.hypotheses if h.origin is HypothesisOrigin.AI] == [h.id for h in earlier]
    assert again.summary.ai_carried == 1
    gone = earlier[0].model_copy(
        update={
            "perturbations": (
                Perturbation(kind=PerturbationKind.PERSON_UNAVAILABLE, target_id="person:퇴사자"),
            )
        }
    )
    dropped = explore_graph(ctx, off.findings, previous=[gone])
    assert not [h for h in dropped.hypotheses if h.origin is HypothesisOrigin.AI]
    assert any("새 그래프에 없어" in note for note in dropped.summary.notes)
    assert dropped.summary.ai_status is AiProposalStatus.REJUDGE


def test_a_carried_ai_hypothesis_the_sweep_now_finds_is_not_listed_twice(
    ctx: CheckContext, off: DiagnosisOutcome
) -> None:
    """결제 v1→v5 path: an AI hypothesis kept from a run where the sweep could not search its service
    must not come back next to the sweep's own entry once the service can be searched."""
    [earlier] = verify_hypotheses(ctx, [draft(ORDER, JEONG_AWAY, SLACK_DOWN)])
    superset = draft(
        ORDER,
        JEONG_AWAY,
        SLACK_DOWN,
        Perturbation(kind=PerturbationKind.CHANNEL_DOWN, target_id="channel:phone"),
    )
    [bigger] = verify_hypotheses(ctx, [superset])
    again = explore_graph(ctx, off.findings, previous=[earlier, bigger])
    keys = [set_key(h.service_id, h.incident_type, h.perturbations) for h in again.hypotheses]
    assert len(keys) == len(set(keys))
    assert not [h for h in again.hypotheses if h.origin is HypothesisOrigin.AI]
    assert again.summary.ai_carried == 0
    assert any("이전 진단의 AI 위험 가설 2건은 이번 자동 전수 탐색이 찾은" in n for n in again.summary.notes)
    # a carried hypothesis naming a node now on hold stays, as needs_info (not dropped as known)
    jeong_held = held(ctx, "person:정민혁")
    later = explore_graph(jeong_held, off.findings, previous=[earlier])
    [ai] = [h for h in later.hypotheses if h.origin is HypothesisOrigin.AI]
    assert ai.status is HypothesisStatus.NEEDS_INFO


# --------------------------------------------------------------------------- reports


def next_payment_patch(report: DiagnosisReport) -> Patch | None:
    by_id = {f.id: f for f in report.findings}
    for patch in report.patches:
        services = {by_id[i].service_id for i in patch.finding_ids if i in by_id}
        if patch.status is PatchStatus.VALIDATED and PAYMENT in services:
            return patch
    return None


def report_of(outcome: DiagnosisOutcome) -> DiagnosisReport:
    from datetime import UTC, datetime

    return DiagnosisReport(
        run_id="run-000001-explore",
        workspace_id="ws-explore",
        graph_version=1,
        graph_hash=outcome.graph.content_hash(),
        created_at=datetime(2026, 10, 28, 14, 0, tzinfo=UTC),
        findings=outcome.findings,
        readiness=outcome.readiness,
        scenarios=outcome.scenarios,
        rehearsals=outcome.rehearsals,
        llm=outcome.usage,
        risk_hypotheses=outcome.risk_hypotheses,
        exploration=outcome.exploration,
    )


def test_reports_show_hypotheses_apart_from_the_score(off: DiagnosisOutcome) -> None:
    report = report_of(off)
    view = build_evidence(report, off.graph)
    markdown = render_view_markdown(view)
    assert "## 9. 규칙 밖 위험 후보 (점수에 반영하지 않음)" in markdown
    section = markdown.split("## 9. ", 1)[1].split("## 부록 A", 1)[0]
    assert "확인됨(리허설로 재현)" in section
    assert "정민혁 부재 + 슬랙 다운" in section
    assert "자동 전수 탐색" in section
    assert "준비도 점수·끊김 판정·통과 기준에는 반영하지 않습니다" in section
    html = render_view_html(view)
    assert '<section id="explore" class="break">' in html
    assert "정민혁 부재 + 슬랙 다운" in html
    confirmed = sum(1 for h in report.risk_hypotheses if h.status is HypothesisStatus.CONFIRMED)
    text = render_cli_summary(report)
    assert f"[규칙 밖 위험 후보] 확인됨(리허설로 재현) {confirmed}건" in text
    payload = scan_payload(report, target="x", tool_version="0", documents={}, notes=(), gate_failures=())
    assert payload["summary"]["risk_hypotheses"]["confirmed"] == confirmed
    assert payload["report"]["risk_hypotheses"][0]["status"] in {s.value for s in HypothesisStatus}
    assert payload["summary"]["readiness"] == off.readiness.overall


def section_9(report: DiagnosisReport, graph: OpsGraph) -> str:
    return render_view_markdown(build_evidence(report, graph)).split("## 9. ", 1)[1].split("## 부록 A", 1)[0]


def test_an_empty_list_never_reads_as_nothing_found_when_the_search_did_not_run(
    off: DiagnosisOutcome,
) -> None:
    """확인 불가 ≠ 정상 (P3): "found nothing" only where every critical service was searched in full."""
    summary = off.exploration
    assert summary is not None
    report = report_of(off).model_copy(update={"risk_hypotheses": ()})
    partly = section_9(report, off.graph)  # 5 of 9 services fail without any perturbation
    assert "탐색한 범위(핵심 서비스 9개 중 4개)에서는 고정 규칙이 보고한 것 밖에서" in partly
    assert (
        "탐색하지 못한 서비스(결제, 배송, 상품 카탈로그, 이상거래 탐지, 정산)는 확인하지 못한 것이지"
        in partly
    )
    blocked = summary.model_copy(
        update={
            "services": tuple(
                s.model_copy(update={"baseline": Outcome.FAIL, "skipped_reason": "x", "candidates": 0})
                for s in summary.services
            )
        }
    )
    nothing = section_9(report.model_copy(update={"exploration": blocked}), off.graph)
    assert "탐색할 수 있는 핵심 서비스가 없어 규칙 밖 위험을 확인하지 못했습니다" in nothing
    assert "찾지 못했습니다" not in nothing
    full = summary.model_copy(
        update={"services": tuple(s for s in summary.services if s.skipped_reason is None)}
    )
    complete = section_9(report.model_copy(update={"exploration": full}), off.graph)
    assert "\n고정 규칙이 보고한 것 밖에서 대응을 끊는 조합을 찾지 못했습니다." in complete


@pytest.mark.parametrize(
    ("status", "mode", "line"),
    [
        (AiProposalStatus.OFF, LlmMode.OFF, "생성형 AI 제안 없음(LLM 끔)"),
        (AiProposalStatus.OFF, LlmMode.REPLAY, "생성형 AI 제안 없음(녹화된 응답에 없는 요청)"),
        (AiProposalStatus.FAILED, LlmMode.LIVE, "생성형 AI 제안 실패(사유는 아래 참고)"),
        (AiProposalStatus.FALLBACK, LlmMode.LIVE, "추출 단계에서 LLM이 실패해"),
        (AiProposalStatus.REJUDGE, LlmMode.OFF, "재판정은 LLM을 쓰지 않음"),
    ],
)
def test_the_report_says_why_there_are_no_ai_proposals(
    off: DiagnosisOutcome, status: AiProposalStatus, mode: LlmMode, line: str
) -> None:
    summary = off.exploration
    assert summary is not None
    report = report_of(off).model_copy(
        update={
            "exploration": summary.model_copy(update={"ai_status": status}),
            "llm": LlmUsageSummary(mode=mode),
        }
    )
    assert line in section_9(report, off.graph)


def test_ai_counts_are_worded_apart(off: DiagnosisOutcome) -> None:
    summary = off.exploration
    assert summary is not None
    used = ExplorationSummary.model_validate(
        {
            **summary.model_dump(),
            "ai_status": "used",
            "ai_used": True,
            "ai_proposed": 6,
            "ai_kept": 2,
            "ai_dropped": 1,
            "ai_duplicates": 1,
            "ai_over_cap": 2,
        }
    )
    text = section_9(report_of(off).model_copy(update={"exploration": used}), off.graph)
    assert (
        "생성형 AI 제안 6건 중 2건 검증(목록에 없는 id·판단 보류·종류 불일치 등 1건 버림, "
        "중복이거나 이미 찾은 끊김 조합과 겹침 1건 버림, 요청 개수를 넘은 2건 제외(상한))"
    ) in text


def test_reports_from_before_exploration_still_load_and_render(off: DiagnosisOutcome) -> None:
    stored = report_of(off).model_dump(mode="json")
    del stored["risk_hypotheses"], stored["exploration"]
    old = DiagnosisReport.model_validate(stored)
    assert (old.risk_hypotheses, old.exploration) == ((), None)
    markdown = render_view_markdown(build_evidence(old, off.graph))
    assert "이번 진단에는 규칙 밖 위험 탐색 기록이 없습니다." in markdown
    assert "[규칙 밖 위험 후보]" not in render_cli_summary(old)


def test_hypotheses_survive_a_json_round_trip(off: DiagnosisOutcome) -> None:
    report = report_of(off)
    again = DiagnosisReport.model_validate_json(report.model_dump_json())
    assert again.risk_hypotheses == report.risk_hypotheses
    assert again.exploration == report.exploration
    assert all(h.incident_type is IncidentType.OUTAGE for h in again.risk_hypotheses)


# --------------------------------------------------------------------------- service


def test_the_service_keeps_hypotheses_and_rejudges_them_after_approvals(runner: asyncio.Runner) -> None:
    """A stored report carries its hypotheses; an approval re-sweeps the new graph (LLM off) and
    verifies the earlier AI hypotheses again — 결제, broken before the fixes, is searched after them."""
    service = new_service(llm_factory=factory_of(fake_llm()))
    report = runner.run(service.bootstrap_demo(llm_mode=LlmMode.LIVE))
    ws = report.workspace_id
    assert service.latest_report(ws).risk_hypotheses == report.risk_hypotheses
    [ai] = [h for h in report.risk_hypotheses if h.origin is HypothesisOrigin.AI]
    assert report.llm.by_task["explore"] == 1
    assert not of_service(report.risk_hypotheses, PAYMENT)
    while (patch := next_payment_patch(report)) is not None:
        report = service.approve_patch(patch.id, actor="박재현", role="lead").report
        assert report.llm.calls == 0
        assert report.exploration is not None
        assert report.exploration.ai_used is False
        assert report.exploration.ai_status is AiProposalStatus.REJUDGE
        assert [h.id for h in report.risk_hypotheses if h.origin is HypothesisOrigin.AI] == [ai.id]
        assert report.exploration.ai_carried == 1
    summary = report.exploration
    assert summary is not None
    payment = next(s for s in summary.services if s.service_id == PAYMENT)
    assert payment.baseline is Outcome.PASS
    assert payment.candidates > 0
    after = of_service(report.risk_hypotheses, PAYMENT)
    assert all(h.status is HypothesisStatus.CONFIRMED for h in after)
    # two people who are each replaceable, but not together; a 상품 카탈로그 incident that pulls
    # 박재현 away is his absence again, so it never shows up as a risk of its own
    assert after
    assert all(len(h.perturbations) == 2 for h in after)
    assert all(p.kind is PerturbationKind.PERSON_UNAVAILABLE for h in after for p in h.perturbations)
    markdown = service.evidence_markdown(ws)
    assert "## 9. 규칙 밖 위험 후보 (점수에 반영하지 않음)" in markdown
    assert SHARED_NETWORK in markdown
