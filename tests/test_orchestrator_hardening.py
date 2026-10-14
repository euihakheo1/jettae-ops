"""Adversarial cases for the orchestrator found in review (what the happy-path suites did not pin).

* **Rebase never trades a newer fact or a confirmed decision for an older one.** An approved
  draft that filled an empty attribute must not overwrite the value the runbook states later; an
  approved fact the re-extraction holds (판단 보류) must stay confirmed, not silently fall back to
  unknown (P3); any sequence of approved demo fixes survives an edit of every runbook span.
* **The cassette records what the venue will ask.** Recording against an operator's workspace
  with approvals made the venue replay miss every draft and drill request, silently.
* **Idempotent requests stay idempotent under concurrency** (two "demo" clicks), while a head
  moved to anything else is still a conflict.
* **The error contract holds for odd input**: unknown rehearsal source, unstorable actor, an
  out-of-range ``JETTAE_LLM_CONCURRENCY``, control characters in an adapter's error message.
* **Delivery safety**: names and runbook text reach the HTML report escaped.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from jettae.config import Settings
from jettae.domain import (
    AddEdge,
    AddNode,
    Attr,
    AttrKey,
    DiagnosisReport,
    Extractor,
    InputBundle,
    LlmMode,
    Node,
    NodeKind,
    OpsGraph,
    Patch,
    PatchAuthor,
    PatchStatus,
    Provenance,
    RemoveEdge,
    ScenarioOrigin,
    SetAttr,
    SourceKind,
    SourceSpan,
    Verification,
)
from jettae.domain.ports import VersionConflictError
from jettae.improve import PatchError, apply_ops
from jettae.ingest import load_bundle_dir, load_bytes
from jettae.llm.base import LLMClient, LLMError, NullLLM
from jettae.orchestrator import (
    DiagnosisOptions,
    DiagnosisOutcome,
    JettaeService,
    rebase_graph,
    record_cassette,
)
from jettae.orchestrator.pipeline import describe_llm_error, run_diagnosis
from test_orchestrator_pipeline import approved, facts, human_patch, moved, payment_owner_patch
from test_orchestrator_support import (
    DEMO,
    PAYMENT,
    event_loop_runner,
    factory_of,
    failing_llm,
    fake_llm,
    new_service,
)

ROLLBACK = "action:결제-서비스-롤백"
RUNBOOK_SPAN = SourceSpan(doc_id="doc-edited", start=0, end=2, quote="결제")
RUNBOOK_PROV = (Provenance(source=SourceKind.RUNBOOK, extractor=Extractor.RULE, spans=(RUNBOOK_SPAN,)),)


@pytest.fixture(scope="module")
def runner() -> Iterator[asyncio.Runner]:
    with event_loop_runner() as loop:
        yield loop


@pytest.fixture(scope="module")
def bundle() -> InputBundle:
    return load_bundle_dir(DEMO)


@pytest.fixture(scope="module")
def off(runner: asyncio.Runner, bundle: InputBundle) -> DiagnosisOutcome:
    """The demo diagnosed with the LLM off (no drills: the rebase cases need only graph and drafts)."""
    return runner.run(
        run_diagnosis(
            bundle,
            base_graph=None,
            llm=NullLLM(),
            llm_mode=LlmMode.OFF,
            options=DiagnosisOptions(rehearse=False),
        )
    )


def validated(outcome: DiagnosisOutcome) -> list[Patch]:
    return [p for p in outcome.patches if p.status is PatchStatus.VALIDATED]


def with_node(graph: OpsGraph, node: Node) -> OpsGraph:
    return graph.model_copy(update={"nodes": {**graph.nodes, node.id: node}})


# --------------------------------------------------------------------------- rebase


def rollback_patch(outcome: DiagnosisOutcome) -> Patch:
    """The drafted fix that marks the 결제 rollback as needing approval (the attribute was absent)."""
    return next(
        p
        for p in validated(outcome)
        if any(isinstance(op, SetAttr) and op.target_id == ROLLBACK and op.previous is None for op in p.ops)
    )


def test_a_value_the_runbook_states_later_beats_the_draft_that_filled_the_gap(off: DiagnosisOutcome) -> None:
    """The lead approved "결제 롤백은 승인 필요" while the runbook said nothing; the team later wrote
    "결제 롤백은 승인 없이 진행". The newer runbook fact wins and the skip is reported."""
    patch = approved(rollback_patch(off))
    assert patch.author is PatchAuthor.TEMPLATE
    head, _ = apply_ops(off.graph, patch.ops)
    action = off.graph.nodes[ROLLBACK]
    assert AttrKey.REQUIRES_APPROVAL.value not in action.attrs
    waived = Attr(value=False, spans=(RUNBOOK_SPAN,), source=SourceKind.RUNBOOK)
    rewritten = with_node(
        moved(off.graph),
        action.model_copy(update={"attrs": {**action.attrs, AttrKey.REQUIRES_APPROVAL.value: waived}}),
    )
    result = rebase_graph(rewritten, head, base_extraction_hash=off.extraction_hash, approved=(patch,))
    assert result.graph.nodes[ROLLBACK].attr(AttrKey.REQUIRES_APPROVAL) is False
    assert result.reapplied == (patch.id,)  # its other decision (who approves) still holds
    added = [op.edge.id for op in patch.ops if isinstance(op, AddEdge)]
    assert added
    assert all(eid in result.graph.edges for eid in added)
    [summary, skipped] = result.notes
    assert "1건 중 1건" in summary
    assert skipped.startswith(f"승인된 수정안 {patch.id}의 연산")
    assert "(set_attr)" in skipped
    assert "더 새로운 사실을 덮어쓰지 않았습니다" in skipped
    assert "'False'" in skipped


def test_a_hand_written_op_without_a_recorded_previous_value_still_applies(off: DiagnosisOutcome) -> None:
    """``previous=None`` on a hand-written op means "not recorded", not "was empty": no newer fact."""
    note = Attr(value="사람이 적은 설명", source=SourceKind.PATCH)
    patch = human_patch("patch-hand", SetAttr(target_id=PAYMENT, key=AttrKey.DESCRIPTION.value, value=note))
    assert off.graph.nodes[PAYMENT].attr(AttrKey.DESCRIPTION) is not None
    result = rebase_graph(moved(off.graph), off.graph, base_extraction_hash="old", approved=(patch,))
    assert result.graph.nodes[PAYMENT].attr(AttrKey.DESCRIPTION) == "사람이 적은 설명"
    assert len(result.notes) == 1


def test_an_approved_fact_the_reextraction_holds_stays_confirmed(off: DiagnosisOutcome) -> None:
    """The team wrote the approved owner into the runbook, but back-translation doubts the sentence:
    the lead's confirmation keeps the fact verified instead of re-opening the gap as unknown (P3)."""
    patch = approved(payment_owner_patch(off))
    head, _ = apply_ops(off.graph, patch.ops)
    [added, *_] = [op.edge for op in patch.ops if isinstance(op, AddEdge)]
    doubted = added.model_copy(
        update={"prov": RUNBOOK_PROV, "verification": Verification.HOLD, "hold_reason": "역번역 대조 불일치"}
    )
    shifted = moved(off.graph)
    extracted = shifted.model_copy(update={"edges": {**shifted.edges, doubted.id: doubted}})
    result = rebase_graph(extracted, head, base_extraction_hash=off.extraction_hash, approved=(patch,))
    edge = result.graph.edges[added.id]
    assert edge.verification is Verification.VERIFIED
    assert edge == added
    assert facts(result.graph) == facts(head)
    assert added.id not in {
        e for e in result.graph.edges if result.graph.edges[e].verification is Verification.HOLD
    }
    assert len(result.notes) == 1  # the summary alone: nothing was skipped


def test_an_approved_node_the_reextraction_holds_is_reported(off: DiagnosisOutcome) -> None:
    channel = Node(
        id="channel:비상-전화망",
        kind=NodeKind.CHANNEL,
        name="비상 전화망",
        prov=(Provenance(source=SourceKind.PATCH, extractor=Extractor.HUMAN, ref="patch-node"),),
    )
    held = channel.model_copy(
        update={"prov": RUNBOOK_PROV, "verification": Verification.HOLD, "hold_reason": "동명"}
    )
    extracted = with_node(off.graph, held)
    result = rebase_graph(
        extracted,
        off.graph,
        base_extraction_hash="old",
        approved=(human_patch("patch-node", AddNode(node=channel)),),
    )
    assert result.reapplied == ()
    [_summary, skipped] = result.notes
    assert "판단 보류 상태입니다(동명)" in skipped
    assert result.graph.nodes[channel.id].verification is Verification.HOLD


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(data=st.data())
def test_any_sequence_of_approved_fixes_survives_an_edit_of_every_span(
    off: DiagnosisOutcome, data: st.DataObject
) -> None:
    """Approve fixes one after another (each on the head the previous one left), move every span
    as an edited runbook does, rebase: the result states exactly what the approved head stated."""
    pool = validated(off)
    order = data.draw(st.permutations(range(len(pool))), label="approval order")
    count = data.draw(st.integers(min_value=1, max_value=len(pool)), label="approvals")
    head = off.graph
    accepted: list[Patch] = []
    for patch in (pool[i] for i in order[:count]):
        try:
            head, _ = apply_ops(head, patch.ops)
        except PatchError:
            continue  # stale on this head: a real service would have superseded it
        accepted.append(approved(patch))
    result = rebase_graph(moved(off.graph), head, base_extraction_hash=off.extraction_hash, approved=accepted)
    assert facts(result.graph) == facts(head)
    assert result.reapplied == tuple(p.id for p in accepted)
    assert len(result.notes) <= 1  # at most the summary — never a skipped decision
    assert not [eid for eid, e in result.graph.edges.items() if e.verification is Verification.HOLD]


# --------------------------------------------------------------------------- cassette


def test_a_cassette_recorded_beside_approvals_still_replays_the_fresh_demo(
    runner: asyncio.Runner, tmp_path: Path
) -> None:
    model = fake_llm()
    operator = new_service(llm_factory=factory_of(model))
    report = runner.run(operator.bootstrap_demo())
    by_id = {f.id: f for f in report.findings}
    fix = next(
        p
        for p in report.patches
        if p.status is PatchStatus.VALIDATED
        and any(by_id[i].service_id == PAYMENT and by_id[i].rule_id == "JT-PPL-001" for i in p.finding_ids)
    )
    operator.approve_patch(fix.id, actor="박재현", role="lead")
    history = operator.versions(report.workspace_id)

    cassette = tmp_path / "dmc-shop.jsonl"
    recorded = runner.run(record_cassette(operator, cassette))
    assert operator.versions(report.workspace_id) == history  # the operator's workspace is untouched
    assert len(operator.list_workspaces()) == 1

    venue = new_service(cassette_path=cassette)
    replayed = runner.run(venue.bootstrap_demo(llm_mode=LlmMode.REPLAY))
    assert replayed.graph_hash == recorded.graph_hash
    assert replayed.llm.by_task == recorded.llm.by_task
    assert {"draft_fix", "scenarios"} <= set(replayed.llm.by_task)
    assert any(p.author is PatchAuthor.AI for p in replayed.patches)
    assert any(s.origin is ScenarioOrigin.LLM for s in replayed.scenarios)


# --------------------------------------------------------------------------- concurrency


def test_two_demo_clicks_at_once_both_succeed(runner: asyncio.Runner) -> None:
    service = new_service()

    async def twice() -> list[DiagnosisReport]:
        return list(await asyncio.gather(service.bootstrap_demo(), service.bootstrap_demo()))

    first, second = runner.run(twice())
    assert first.workspace_id == second.workspace_id
    assert (first.graph_version, second.graph_version) == (1, 1)
    assert first.graph_hash == second.graph_hash
    assert [v.number for v in service.versions(first.workspace_id)] == [1]
    assert len(service.list_workspaces()) == 1


@pytest.mark.parametrize("same", [True, False], ids=["same-graph", "other-graph"])
def test_a_head_committed_meanwhile_conflicts_only_when_it_differs(
    runner: asyncio.Runner, same: bool
) -> None:
    """Another diagnosis commits while this one works. The identical graph is the result this run
    would commit (no conflict); any other graph makes this run stale."""
    reference = new_service()
    expected = reference.graph(runner.run(reference.bootstrap_demo()).workspace_id)[1]
    meanwhile = expected
    if not same:
        some_edge = next(iter(expected.edges))
        meanwhile, _ = apply_ops(expected, (RemoveEdge(edge_id=some_edge),))
    service: JettaeService
    done: list[bool] = []

    def other_diagnosis(_settings: Settings, _mode: LlmMode) -> LLMClient:
        if not done:
            done.append(True)
            [workspace] = service.list_workspaces()
            service.versions_service.commit_graph(workspace.id, meanwhile, actor="다른 진단")
        return NullLLM()

    service = new_service(llm_factory=other_diagnosis)
    if same:
        report = runner.run(service.bootstrap_demo())
        assert (report.graph_version, report.graph_hash) == (1, expected.content_hash())
        assert [v.number for v in service.versions(report.workspace_id)] == [1]
        assert service.latest_report(report.workspace_id) == report
    else:
        with pytest.raises(VersionConflictError):
            runner.run(service.bootstrap_demo())


# --------------------------------------------------------------------------- error contract


def test_an_unknown_rehearsal_source_is_bad_input_not_a_crash(
    runner: asyncio.Runner, bundle: InputBundle
) -> None:
    service = new_service()
    ws = service.create_workspace("별빛상점").id
    service.load_bundle(ws, bundle)
    runner.run(service.diagnose(ws, rehearse=False, draft=False))
    with pytest.raises(ValueError, match="시나리오 종류") as raised:
        runner.run(service.rehearse(ws, generate="everything"))  # type: ignore[arg-type]
    assert not isinstance(raised.value, KeyError)


@pytest.mark.parametrize("configured", [0, -3, 65, 1000])
def test_an_out_of_range_llm_concurrency_does_not_break_diagnosis(
    runner: asyncio.Runner, bundle: InputBundle, configured: int
) -> None:
    service = new_service(llm_concurrency=configured)
    ws = service.create_workspace("별빛상점").id
    service.load_bundle(ws, bundle)
    report = runner.run(service.diagnose(ws, rehearse=False, draft=False))
    assert report.graph_version == 1
    live = new_service(llm_concurrency=configured, llm_factory=factory_of(fake_llm()))
    ws = live.create_workspace("별빛상점").id
    live.load_bundle(ws, bundle)
    assert runner.run(live.diagnose(ws, llm_mode=LlmMode.LIVE, rehearse=False, draft=False)).llm.calls > 0


@pytest.mark.parametrize("actor", ["이수진\x00", "이수진\ud800"], ids=["nul", "surrogate"])
def test_an_unstorable_actor_is_refused_before_any_write_or_model_call(
    runner: asyncio.Runner, actor: str
) -> None:
    model = fake_llm()
    service = new_service(llm_factory=factory_of(model))
    report = runner.run(service.bootstrap_demo())
    ws = report.workspace_id
    patches = {p.id for p in service.patches(ws)}
    audit = len(service.store.audit.entries(ws))
    with pytest.raises(ValueError, match="actor"):
        runner.run(service.draft_patches(ws, llm_mode=LlmMode.LIVE, actor=actor))
    with pytest.raises(ValueError, match="actor"):
        runner.run(service.diagnose(ws, llm_mode=LlmMode.LIVE, actor=actor))
    assert model.calls == []
    assert {p.id for p in service.patches(ws)} == patches  # no unaudited drafts
    assert len(service.store.audit.entries(ws)) == audit
    assert [v.number for v in service.versions(ws)] == [1]


def test_control_characters_in_an_adapter_error_never_reach_the_audit_log(
    runner: asyncio.Runner, bundle: InputBundle
) -> None:
    """A provider's error body is outside text; the run's note goes to the audit log, whose
    record the next diagnosis needs for its lineage — it must always be storable."""
    assert describe_llm_error(LLMError("bad\x00 json\ud800 \x1b[31mred\u200b")) == (
        "LLM 호출이 실패했습니다(bad json [31mred)"
    )
    service = new_service(llm_factory=factory_of(failing_llm("extract", LLMError("모델 출력:\x00\ud800 끝"))))
    ws = service.create_workspace("별빛상점").id
    service.load_bundle(ws, bundle)
    report = runner.run(service.diagnose(ws, llm_mode=LlmMode.LIVE, rehearse=False, draft=False))
    [note] = service.report_notes(ws, report.run_id)
    assert note.endswith("원인: LLM 호출이 실패했습니다(모델 출력: 끝)")


# --------------------------------------------------------------------------- delivery safety


def test_names_and_runbook_text_reach_the_html_report_escaped(
    runner: asyncio.Runner, bundle: InputBundle
) -> None:
    service = new_service()
    ws = service.create_workspace("<script>alert('ws')</script> 별빛").id
    text = (
        "# <img src=x onerror=alert(1)> 결제\n\n"
        "결제 장애가 계속되면 <script>alert(2)</script> 김대리에게 연락.\n"
        "- 담당: 김대리 <a href='javascript:alert(3)'>링크</a>\n"
    )
    hostile = load_bytes("runbooks/<b>결제</b>.md", text.encode())
    service.upload_inputs(ws, documents=(hostile,), roster=bundle.roster, catalog=bundle.catalog)
    runner.run(service.diagnose(ws))
    html = service.evidence_html(ws)
    for raw in ("<script>alert", "<img src=x", "<a href='javascript", "<b>결제</b>"):
        assert raw not in html
    assert "별빛" in html
