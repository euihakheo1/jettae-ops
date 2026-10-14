"""The finals demo through :class:`JettaeService` — what the audience sees, end to end.

bootstrap_demo → every planted defect is reported by the right rule (``tests/test_integration_demo``
keeps the list) → the 결제 fixes are approved one by one by the team lead → 결제 climbs to ≥ 80 and
the 추석 drill turns PASS → a viewer cannot approve → a stale patch is refused → revert restores the
original graph hash. Then the LLM side: LIVE with a fake model is hybrid and metered (judgement path
0 calls), an LLM failure falls back to the rule parser with a note, a recorded cassette replays to
the identical graph offline, and two independent runs produce identical reports (P6).
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from jettae.domain import (
    DiagnosisReport,
    Extractor,
    FindingStatus,
    LlmMode,
    Outcome,
    Patch,
    PatchAuthor,
    PatchStatus,
    ScenarioOrigin,
)
from jettae.domain.ports import VersionConflictError
from jettae.llm.base import LLMError, NullLLM
from jettae.orchestrator import DEMO_WORKSPACE_NAME, JettaeService, record_cassette
from jettae.store import MemoryStore
from support_loops import CHILD_PRELUDE
from test_integration_demo import INTENDED, DemoRun, pipeline
from test_orchestrator_support import (
    CHUSEOK,
    LOOP_ATTEMPTS,
    PAYMENT,
    event_loop_runner,
    factory_of,
    failing_llm,
    fake_llm,
    fixed_clock,
    new_service,
    settings,
)

BASELINE = f"tpl:{PAYMENT}:baseline"
TESTS = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def runner() -> Iterator[asyncio.Runner]:
    with event_loop_runner() as loop:
        yield loop


@pytest.fixture(scope="module")
def demo(runner: asyncio.Runner) -> tuple[JettaeService, DiagnosisReport]:
    service = new_service()
    return service, runner.run(service.bootstrap_demo())


def payment_score(report: DiagnosisReport) -> float:
    return next(s.score for s in report.readiness.services if s.service_id == PAYMENT)


def outcome_of(report: DiagnosisReport, scenario_id: str) -> Outcome:
    """The drill's result on the reported graph (a report may also carry the 'before' result)."""
    [result] = [
        r for r in report.rehearsals if r.scenario_id == scenario_id and r.graph_hash == report.graph_hash
    ]
    return result.outcome


def next_payment_patch(report: DiagnosisReport) -> Patch | None:
    by_id = {f.id: f for f in report.findings}
    for patch in report.patches:
        services = {by_id[i].service_id for i in patch.finding_ids if i in by_id}
        if patch.status is PatchStatus.VALIDATED and PAYMENT in services:
            return patch
    return None


# --------------------------------------------------------------------------- diagnosis


@pytest.mark.parametrize(("label", "rule_id", "service", "words"), INTENDED, ids=[row[0] for row in INTENDED])
def test_bootstrap_reports_every_planted_defect(
    demo: tuple[JettaeService, DiagnosisReport],
    label: str,
    rule_id: str,
    service: str | None,
    words: tuple[str, ...],
) -> None:
    service_api, report = demo
    _, graph = service_api.graph(report.workspace_id)
    service_id = None if service is None else next(s.id for s in graph.nodes.values() if s.name == service)
    hits = [f for f in report.findings if f.rule_id == rule_id and f.service_id == service_id]
    assert any(f.status is FindingStatus.BROKEN and all(w in f.message for w in words) for f in hits), label


def reference_run() -> DemoRun:
    """``tests/test_integration_demo.pipeline()`` (it opens its own event loop; retried like ours)."""
    attempts = LOOP_ATTEMPTS
    while True:
        try:
            return pipeline()
        except OSError:
            attempts -= 1
            if attempts == 0:
                raise


def test_bootstrap_equals_the_reference_pipeline(demo: tuple[JettaeService, DiagnosisReport]) -> None:
    _, report = demo
    reference = reference_run()
    assert report.graph_hash == reference.graph.content_hash()
    assert [f.model_dump() for f in report.findings] == [f.model_dump() for f in reference.findings]
    assert report.readiness == reference.readiness
    assert [s.id for s in report.scenarios] == [s.id for s in reference.scenarios]
    assert [r.result_hash() for r in report.rehearsals] == [r.result_hash() for r in reference.results]
    assert report.security == reference.notices
    assert report.held_element_ids == ()
    assert report.llm.calls == 0
    assert report.llm.judgement_path_calls == 0


def test_bootstrap_persists_the_run(demo: tuple[JettaeService, DiagnosisReport]) -> None:
    service, report = demo
    ws = report.workspace_id
    assert service.get_workspace(ws).name == DEMO_WORKSPACE_NAME
    assert service.latest_report(ws) == report
    [version] = service.versions(ws)
    assert (version.number, version.graph_hash, version.patch_id) == (1, report.graph_hash, None)
    assert report.graph_version == 1
    assert report.run_id.startswith("run-000001-")
    assert {s.id for s in service.scenarios(ws)} >= {s.id for s in report.scenarios}
    assert {p.id for p in service.patches(ws)} == {p.id for p in report.patches}
    assert all(p.created_at == fixed_clock() for p in report.patches)
    human = [s for s in report.scenarios if s.origin is ScenarioOrigin.HUMAN]
    assert [s.id for s in human] == [
        "human:dmc-s1-chuseok-payment",
        "human:dmc-s2-sangam-idc-fire",
        "human:dmc-s3-slack-outage-order",
        "human:dmc-s4-breach-report",
    ]
    assert outcome_of(report, CHUSEOK) is Outcome.FAIL
    assert set(report.timings_ms) >= {"guard", "extract", "check", "rehearse", "draft", "total"}
    summary = service.input_summary(ws)
    assert (len(summary.documents), summary.security_notices, summary.pending_scenario_files) == (3, 1, 0)


def test_bootstrap_is_idempotent(runner: asyncio.Runner) -> None:
    service = new_service()
    first = runner.run(service.bootstrap_demo())
    second = runner.run(service.bootstrap_demo())
    assert second.workspace_id == first.workspace_id
    assert len(service.list_workspaces()) == 1
    assert (second.graph_version, second.graph_hash) == (1, first.graph_hash)
    assert second.run_id.startswith("run-000002-")
    assert service.latest_report(first.workspace_id) == second
    assert [s.id for s in second.scenarios] == [s.id for s in first.scenarios]  # drills carried over


# --------------------------------------------------------------------------- improvement loop


def test_the_payment_path_is_repaired_by_approvals_and_reverted(runner: asyncio.Runner) -> None:
    service = new_service()
    report = runner.run(service.bootstrap_demo())
    ws = report.workspace_id
    original = report.graph_hash
    first_patches = {p.id: p for p in report.patches}

    patch = next_payment_patch(report)
    assert patch is not None
    with pytest.raises(PermissionError):
        service.approve_patch(patch.id, actor="이수진", role="viewer")
    with pytest.raises(PermissionError):
        service.revert(ws, 1, actor="이수진", role="editor")
    assert service.get_patch(patch.id).status is PatchStatus.VALIDATED  # the refusal changed nothing

    scores = [payment_score(report)]
    approved: list[str] = []
    while (patch := next_payment_patch(report)) is not None:
        result = service.approve_patch(patch.id, actor="박재현", role="lead")
        assert result.version.patch_id == patch.id
        assert result.report.graph_version == result.version.number == len(approved) + 2
        assert result.report.llm.calls == 0  # re-judged with the LLM off
        assert service.get_patch(patch.id).status is PatchStatus.APPROVED
        approved.append(patch.id)
        report = result.report
        scores.append(payment_score(report))

    assert len(approved) >= 3
    assert scores == sorted(scores)
    assert len(set(scores)) == len(scores)
    assert scores[0] < 30
    assert scores[-1] >= 80
    assert outcome_of(report, CHUSEOK) is Outcome.PASS
    assert outcome_of(report, BASELINE) is Outcome.PASS
    # The report keeps the drill's original result next to the current one (before → after).
    before = [r for r in report.rehearsals if r.scenario_id == CHUSEOK and r.graph_hash == original]
    assert [r.outcome for r in before] == [Outcome.FAIL]
    markdown = service.evidence_markdown(ws)
    assert f"v1 실패 → v{report.graph_version} 통과" in markdown
    assert "박재현" in markdown

    # A patch drafted on version 1 and left pending was superseded by the first approval.
    stale = next(
        p for pid, p in first_patches.items() if p.status is PatchStatus.VALIDATED and pid not in approved
    )
    assert service.get_patch(stale.id).status is PatchStatus.SUPERSEDED
    with pytest.raises(VersionConflictError):
        service.approve_patch(stale.id, actor="박재현", role="lead")

    reverted = service.revert(ws, 1, actor="박재현", role="lead")
    assert reverted.version.graph_hash == original
    assert reverted.version.revert_of == 1
    assert reverted.version.number == len(approved) + 2
    assert payment_score(reverted.report) == scores[0]
    assert outcome_of(reverted.report, CHUSEOK) is Outcome.FAIL
    actions = [e.action for e in service.store.audit.entries(ws)]
    assert actions.count("patch.approved") == len(approved)
    assert actions.count("graph.reverted") == 1
    # Reverting again to the same content appends nothing and returns the current report.
    again = service.revert(ws, 1, actor="박재현", role="lead")
    assert again.version == reverted.version
    assert again.report == reverted.report


def test_rediagnosis_keeps_approved_patches(runner: asyncio.Runner) -> None:
    service = new_service()
    report = runner.run(service.bootstrap_demo())
    ws = report.workspace_id
    patch = next_payment_patch(report)
    assert patch is not None
    approved = service.approve_patch(patch.id, actor="박재현", role="lead")
    again = runner.run(service.diagnose(ws))
    assert (again.graph_version, again.graph_hash) == (2, approved.version.graph_hash)
    assert again.findings == approved.report.findings
    assert service.report_notes(ws) == ()
    assert len(service.versions(ws)) == 2


# --------------------------------------------------------------------------- the LLM side


def test_live_mode_is_hybrid_and_metered_with_a_fake_model(runner: asyncio.Runner) -> None:
    model = fake_llm()
    service = new_service(llm_factory=factory_of(model))
    report = runner.run(service.bootstrap_demo(llm_mode=LlmMode.LIVE))
    usage = report.llm
    assert usage.mode is LlmMode.LIVE
    assert usage.calls == len(model.calls) > 0
    assert usage.by_task["extract"] == 3
    assert usage.judgement_path_calls == 0
    _, graph = service.graph(report.workspace_id)
    assert any(p.extractor is Extractor.LLM for e in graph.edges.values() for p in e.prov)
    [version] = service.versions(report.workspace_id)
    assert "규칙 파서+LLM" in version.note
    assert any(s.origin is ScenarioOrigin.LLM for s in report.scenarios)
    assert any(p.author is PatchAuthor.AI and p.status is PatchStatus.VALIDATED for p in report.patches)
    assert service.report_notes(report.workspace_id) == ()


def test_llm_failure_falls_back_to_the_rule_parser_with_a_note(
    runner: asyncio.Runner, demo: tuple[JettaeService, DiagnosisReport]
) -> None:
    _, off = demo
    service = new_service(llm_factory=factory_of(failing_llm("extract", LLMError("서버 오류"))))
    report = runner.run(service.bootstrap_demo(llm_mode=LlmMode.LIVE))
    assert report.graph_hash == off.graph_hash
    assert [f.id for f in report.findings] == [f.id for f in off.findings]
    notes = service.report_notes(report.workspace_id)
    assert any(n.startswith("LLM 추출이 실패해 규칙 파서 결과만으로 그래프를 만들었습니다") for n in notes)
    assert "규칙 파서," in service.versions(report.workspace_id)[0].note


def test_a_recorded_cassette_replays_offline_to_the_same_graph(
    runner: asyncio.Runner, tmp_path: Path
) -> None:
    cassette = tmp_path / "dmc-shop.jsonl"
    live_model = fake_llm()
    recorder = new_service(llm_factory=factory_of(live_model))
    recorded = runner.run(record_cassette(recorder, cassette))
    assert recorded.llm.calls == len(live_model.calls) > 0
    assert cassette.is_file()

    venue = new_service(cassette_path=cassette)  # default factory: CassetteLLM in REPLAY mode
    replayed = runner.run(venue.bootstrap_demo(llm_mode=LlmMode.REPLAY))
    assert replayed.graph_hash == recorded.graph_hash
    assert replayed.findings == recorded.findings
    assert replayed.llm.provider == "cassette"
    assert replayed.llm.calls == recorded.llm.calls
    assert replayed.llm.by_task == recorded.llm.by_task
    assert [(p.id, p.author) for p in replayed.patches] == [(p.id, p.author) for p in recorded.patches]
    assert venue.report_notes(replayed.workspace_id) == ()


def test_a_missing_cassette_degrades_to_rules_not_to_an_error(
    runner: asyncio.Runner, tmp_path: Path, demo: tuple[JettaeService, DiagnosisReport]
) -> None:
    _, off = demo
    venue = new_service(cassette_path=tmp_path / "없음.jsonl")
    report = runner.run(venue.bootstrap_demo(llm_mode=LlmMode.REPLAY))
    assert report.graph_hash == off.graph_hash
    assert report.llm.calls == 0
    assert any("카세트에 없는 요청" in n for n in venue.report_notes(report.workspace_id))


def test_recording_without_a_model_is_an_error(runner: asyncio.Runner, tmp_path: Path) -> None:
    service = new_service(llm_factory=factory_of(NullLLM()))
    with pytest.raises(RuntimeError, match="녹화하지 못했습니다"):
        runner.run(record_cassette(service, tmp_path / "empty.jsonl"))


# --------------------------------------------------------------------------- reproducibility


def test_two_independent_runs_produce_the_same_report(runner: asyncio.Runner) -> None:
    def run_once() -> DiagnosisReport:
        service = JettaeService(MemoryStore(clock=fixed_clock), settings(), clock=fixed_clock)
        return runner.run(service.bootstrap_demo())

    first, second = run_once(), run_once()
    assert first.model_dump(exclude={"timings_ms"}) == second.model_dump(exclude={"timings_ms"})
    assert [r.result_hash() for r in first.rehearsals] == [r.result_hash() for r in second.rehearsals]


@pytest.mark.parametrize("seed", ["0", "4242"])
def test_the_report_does_not_depend_on_the_hash_seed(
    demo: tuple[JettaeService, DiagnosisReport], seed: str
) -> None:
    """Set/dict iteration order varies with PYTHONHASHSEED; nothing a report publishes may."""
    _, report = demo
    expected = hashlib.sha256(report.model_dump_json(exclude={"timings_ms"}).encode("utf-8")).hexdigest()
    code = (
        CHILD_PRELUDE
        + f"sys.path.insert(0, {str(TESTS)!r})\n"
        + "from test_orchestrator_support import demo_report_digest\n"
        + "sys.stdout.write(demo_report_digest())\n"
    )
    env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONIOENCODING": "utf-8"}
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        check=False,
        timeout=600,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == expected
