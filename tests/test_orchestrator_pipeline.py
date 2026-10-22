"""The diagnosis pipeline (LangGraph) and its stage functions, without the service around it.

What must hold: the graph has the documented shape (two conditional roads: LLM fallback and
rebase); LLM OFF never touches a model and gives exactly the reference pipeline of
``tests/test_integration_demo.py``; LIVE is hybrid and metered with zero judgement-path calls; an
LLM failure falls back to the rule parser with a Korean note; a head graph with approved patches is
reused when nothing changed and rebased (patches re-applied, misfits skipped with a note) when the
extraction changed; drafts arrive validated with an honest status.
"""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import Iterator
from datetime import datetime
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from jettae.check.context import CheckContext
from jettae.domain import (
    AddEdge,
    AiProposalStatus,
    Attr,
    AttrKey,
    Edge,
    EdgeKind,
    Extractor,
    Finding,
    GraphBuilder,
    InputBundle,
    LlmMode,
    LlmUsageSummary,
    Node,
    NodeKind,
    OpsGraph,
    Patch,
    PatchAuthor,
    PatchOp,
    PatchStatus,
    Provenance,
    RemoveEdge,
    Scenario,
    ScenarioOrigin,
    SetAttr,
    Severity,
    SimConfig,
    SourceKind,
    SourceSpan,
    Verification,
    edge_id,
)
from jettae.extract import RuleExtractor, assemble, verify_backtranslation
from jettae.improve import PatchError, apply_ops
from jettae.ingest import load_bundle_dir
from jettae.llm.base import (
    LLMClient,
    LLMDisabledError,
    LLMError,
    LLMOutputError,
    LLMRefusalError,
    LLMUsage,
    NullLLM,
    UsageMeter,
)
from jettae.orchestrator import (
    NODE_NAMES,
    DiagnosisOptions,
    DiagnosisOutcome,
    build_diagnosis_graph,
    evaluate_graph,
    rebase_graph,
    run_diagnosis,
)
from jettae.orchestrator.pipeline import (
    CATALOG_GAP_RULE,
    build_context,
    catalog_gaps,
    describe_llm_error,
    draft_version,
    gather_scenarios,
    held_element_ids,
    usage_summary,
    validate_drafts,
)
from jettae.rehearse import scenarios_from_yaml
from test_orchestrator_support import DEMO, PAYMENT, PAYMENT_QUOTE, event_loop_runner, failing_llm, fake_llm

DRILLS = (DEMO / "scenarios.yaml").read_bytes()
CATALOG_PROV = (Provenance(source=SourceKind.CATALOG, extractor=Extractor.IMPORT, ref="catalog.yaml"),)
LINEAR = (
    "guard",
    "extract",
    "assemble",
    "verify",
    "check",
    "score",
    "scenarios",
    "rehearse",
    "explore",
    "draft",
    "finalize",
)


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
    base_graph: OpsGraph | None = None,
    **options: Any,
) -> DiagnosisOutcome:
    return runner.run(
        run_diagnosis(
            bundle,
            base_graph=base_graph,
            llm=llm if llm is not None else NullLLM(),
            llm_mode=mode,
            options=DiagnosisOptions(**options),
        )
    )


@pytest.fixture(scope="module")
def off(runner: asyncio.Runner, bundle: InputBundle) -> DiagnosisOutcome:
    return diagnose(runner, bundle, scenario_sources=(DRILLS,))


# --------------------------------------------------------------------------- shape


def test_graph_has_the_documented_nodes_and_conditional_roads() -> None:
    drawn = build_diagnosis_graph().get_graph()
    assert set(NODE_NAMES) <= set(drawn.nodes)
    edges = {(e.source, e.target, e.conditional) for e in drawn.edges}
    for source, target in itertools.pairwise(LINEAR):
        assert (source, target) in {(s, t) for s, t, _ in edges}, (source, target)
    assert ("extract", "fallback", True) in edges
    assert ("extract", "assemble", True) in edges
    assert ("fallback", "assemble", False) in edges
    assert ("verify", "rebase", True) in edges
    assert ("verify", "check", True) in edges
    assert ("rebase", "check", False) in edges


# --------------------------------------------------------------------------- LLM OFF


def test_off_equals_the_reference_pipeline(
    off: DiagnosisOutcome, bundle: InputBundle, runner: asyncio.Runner
) -> None:
    """LLM OFF is the rule parser → assemble → slot-only back-translation, nothing else."""
    extractions = [
        RuleExtractor().extract(d, roster=bundle.roster, catalog=bundle.catalog) for d in bundle.documents
    ]
    assembled = assemble(
        extractions,
        bundle.documents,
        roster=bundle.roster,
        catalog=bundle.catalog,
        oncall=bundle.oncall,
        policy=bundle.policy,
    )
    verified = runner.run(verify_backtranslation(assembled.graph, bundle.documents, None))
    assert off.graph.content_hash() == verified.graph.content_hash() == off.extraction_hash
    assert off.extractor == "rule"
    assert off.grounding == assembled.grounding
    assert off.usage == LlmUsageSummary(mode=LlmMode.OFF)
    assert off.notes == ()
    assert off.graph_reused is False
    assert off.reapplied_patch_ids == ()
    assert off.draft_base_version == 1
    assert list(off.timings) == [*LINEAR, "total"]
    assert all(ms >= 0 for ms in off.timings.values())


def test_off_never_calls_the_model_it_was_given(runner: asyncio.Runner, bundle: InputBundle) -> None:
    spy = fake_llm()
    outcome = diagnose(runner, bundle, llm=spy, mode=LlmMode.OFF, rehearse=False, draft=False)
    assert spy.calls == []
    assert outcome.usage.calls == 0
    assert outcome.extractor == "rule"


def test_security_notices_come_from_the_guard(off: DiagnosisOutcome, bundle: InputBundle) -> None:
    [notice] = off.security
    [doc] = [d for d in bundle.documents if d.doc_id == notice.doc_id]
    assert doc.path == "runbooks/03-보안-개인정보-신고.md"
    assert doc.quarantined == ((notice.start, notice.end),)


def test_a_document_listed_twice_is_diagnosed_once(
    runner: asyncio.Runner, bundle: InputBundle, off: DiagnosisOutcome
) -> None:
    doubled = bundle.model_copy(update={"documents": (*bundle.documents, *bundle.documents)})
    outcome = diagnose(runner, doubled, rehearse=False, draft=False)
    assert outcome.graph.content_hash() == off.graph.content_hash()
    assert outcome.security == off.security


def test_bookkeeping_matches_the_graph_and_findings(off: DiagnosisOutcome) -> None:
    assert off.held_ids == held_element_ids(off.graph) == ()
    gaps = {f.service_id for f in off.findings if f.rule_id == CATALOG_GAP_RULE}
    assert set(off.catalog_gaps) == gaps
    assert {off.graph.nodes[s].name for s in off.catalog_gaps} == {
        "배송",
        "배송비 견적",
        "정산",
        "이상거래 탐지",
    }


def test_scenarios_are_templates_then_human_drills(off: DiagnosisOutcome) -> None:
    origins = [s.origin for s in off.scenarios]
    assert set(origins) == {ScenarioOrigin.TEMPLATE, ScenarioOrigin.HUMAN}
    first_human = origins.index(ScenarioOrigin.HUMAN)
    assert all(o is ScenarioOrigin.HUMAN for o in origins[first_human:])
    assert [r.scenario_id for r in off.rehearsals] == [s.id for s in off.scenarios]


def test_drafts_arrive_validated_with_an_honest_status(off: DiagnosisOutcome) -> None:
    assert off.patches
    statuses = {p.status for p in off.patches}
    assert PatchStatus.VALIDATED in statuses
    for patch in off.patches:
        assert patch.validation is not None
        assert patch.base_version == 1
        assert patch.author is PatchAuthor.TEMPLATE
        if patch.status is PatchStatus.VALIDATED:
            assert patch.ops
            assert patch.validation.passed
            assert patch.validation.unknown_people == ()
        elif patch.status is PatchStatus.DRAFT:
            assert not patch.ops  # a review memo: only a human can resolve it
        else:
            assert patch.status is PatchStatus.REJECTED_BY_VALIDATOR
            assert patch.ops
            assert not patch.validation.passed


def test_rehearse_and_draft_can_be_switched_off(runner: asyncio.Runner, bundle: InputBundle) -> None:
    outcome = diagnose(runner, bundle, rehearse=False, draft=False, scenario_sources=(DRILLS,))
    assert outcome.scenarios == ()
    assert outcome.rehearsals == ()
    assert outcome.patches == ()
    assert outcome.draft_base_version == 1
    no_rehearsal = diagnose(runner, bundle, rehearse=False)
    assert no_rehearsal.patches  # validated against the fixed rules alone
    assert all(p.validation is not None and p.validation.rehearsals_fixed == () for p in no_rehearsal.patches)


def test_drafting_can_target_chosen_findings(
    runner: asyncio.Runner, off: DiagnosisOutcome, bundle: InputBundle
) -> None:
    [owner_gap] = [f for f in off.findings if f.rule_id == "JT-PPL-001" and f.service_id == PAYMENT][:1]
    outcome = diagnose(runner, bundle, draft_finding_ids=(owner_gap.id, "JT-NOPE:0"), rehearse=False)
    assert outcome.patches
    assert all(owner_gap.id in p.finding_ids for p in outcome.patches)
    assert any("JT-NOPE:0" in note and "수정안을 만들지 않았습니다" in note for note in outcome.notes)


def test_unusable_inputs_are_refused_before_any_work(runner: asyncio.Runner, bundle: InputBundle) -> None:
    with pytest.raises(ValueError, match="런북"):
        diagnose(runner, bundle.model_copy(update={"documents": ()}))
    with pytest.raises(ValueError, match="base_version"):
        diagnose(runner, bundle, base_graph=OpsGraph())
    with pytest.raises(ValueError, match="base_graph"):
        diagnose(runner, bundle, base_version=1)


def test_a_drill_naming_something_unknown_is_an_error(runner: asyncio.Runner, bundle: InputBundle) -> None:
    broken = "- id: x\n  service: 없는-서비스\n".encode()
    with pytest.raises(ValueError, match="없는-서비스"):
        diagnose(runner, bundle, scenario_sources=(broken,), draft=False)


def test_options_refuse_inconsistent_requests() -> None:
    with pytest.raises(ValidationError, match="기준 버전"):
        DiagnosisOptions(base_extraction_hash="abc")
    with pytest.raises(ValidationError, match="시간대"):
        DiagnosisOptions(now=datetime(2026, 10, 28, 9, 0))
    pending = Patch(
        id="patch-x",
        workspace_id="w",
        base_version=1,
        finding_ids=(),
        ops=(),
        runbook_text="t",
        rationale="r",
        author=PatchAuthor.TEMPLATE,
    )
    with pytest.raises(ValidationError, match="승인되지 않은"):
        DiagnosisOptions(base_version=1, approved_patches=(pending,))
    template = Scenario(id="tpl:x", title="t", service_id=PAYMENT, origin=ScenarioOrigin.TEMPLATE)
    with pytest.raises(ValidationError, match="템플릿"):
        DiagnosisOptions(carried_scenarios=(template,))
    with pytest.raises(ValidationError):
        DiagnosisOptions(llm_concurrency=0)


# --------------------------------------------------------------------------- LLM LIVE / failures


def test_live_is_hybrid_metered_and_never_judges_with_the_model(
    runner: asyncio.Runner, bundle: InputBundle, off: DiagnosisOutcome
) -> None:
    model = fake_llm()
    outcome = diagnose(runner, bundle, llm=model, mode=LlmMode.LIVE, scenario_sources=(DRILLS,))
    assert outcome.extractor == "hybrid"
    usage = outcome.usage
    assert usage.mode is LlmMode.LIVE
    assert (usage.provider, usage.model) == ("fake", "fake-1")
    assert usage.calls == len(model.calls) > 0
    assert usage.by_task["extract"] == len(bundle.documents)
    assert set(usage.by_task) == {"extract", "backtranslate", "scenarios", "explore", "draft_fix"}
    assert usage.by_task["explore"] == 1
    assert usage.judgement_path_calls == 0
    # The model's relation is grounded and merged onto the rule parser's edge: same facts, two sources.
    edge = outcome.graph.edges[edge_id(EdgeKind.OWNED_BY, PAYMENT, "person:김민준")]
    assert {p.extractor for p in edge.prov} >= {Extractor.RULE, Extractor.LLM}
    assert any(PAYMENT_QUOTE in s.quote for p in edge.prov if p.extractor is Extractor.LLM for s in p.spans)
    assert outcome.graph.content_hash() != off.graph.content_hash()
    # Judgement is the same fixed rules over the same facts.
    assert [(f.rule_id, f.service_id) for f in outcome.findings] == [
        (f.rule_id, f.service_id) for f in off.findings
    ]
    assert [s.id for s in outcome.scenarios if s.origin is ScenarioOrigin.LLM]
    assert any(p.author is PatchAuthor.AI for p in outcome.patches)


@pytest.mark.parametrize(
    "error", [LLMDisabledError("키 없음"), LLMError("rate limited")], ids=["disabled", "error"]
)
def test_llm_failure_falls_back_to_rules_with_a_note(
    runner: asyncio.Runner, bundle: InputBundle, off: DiagnosisOutcome, error: LLMError
) -> None:
    model = failing_llm("extract", error)
    outcome = diagnose(runner, bundle, llm=model, mode=LlmMode.LIVE, rehearse=False, draft=False)
    assert outcome.extractor == "rule"
    assert outcome.graph.content_hash() == off.graph.content_hash()
    [note] = [n for n in outcome.notes if "규칙 파서 결과만으로" in n]
    assert note.startswith("LLM 추출이 실패해")
    assert "fallback" in outcome.timings
    assert outcome.usage.judgement_path_calls == 0


def test_after_a_fallback_the_rest_of_the_run_stays_off_the_model(
    runner: asyncio.Runner, bundle: InputBundle, off: DiagnosisOutcome
) -> None:
    """An unavailable model is not asked again per edge and per patch: the run equals LLM OFF."""
    model = failing_llm("extract", LLMError("overloaded"))
    outcome = diagnose(runner, bundle, llm=model, mode=LlmMode.LIVE, scenario_sources=(DRILLS,))
    assert {c.task for c in model.calls} == {"extract"}
    assert (outcome.usage.mode, outcome.usage.calls, outcome.usage.by_task) == (LlmMode.LIVE, 0, {})
    assert outcome.graph.content_hash() == off.graph.content_hash()
    assert outcome.findings == off.findings
    assert outcome.readiness == off.readiness
    assert [s.id for s in outcome.scenarios] == [s.id for s in off.scenarios]
    assert [r.result_hash() for r in outcome.rehearsals] == [r.result_hash() for r in off.rehearsals]
    assert [p.model_dump() for p in outcome.patches] == [p.model_dump() for p in off.patches]
    [note] = outcome.notes
    assert "나머지 단계" in note
    assert note.endswith("원인: LLM 호출이 실패했습니다(overloaded)")


def test_null_model_in_live_mode_takes_the_fallback_road(runner: asyncio.Runner, bundle: InputBundle) -> None:
    outcome = diagnose(runner, bundle, llm=NullLLM(), mode=LlmMode.LIVE, rehearse=False, draft=False)
    assert outcome.extractor == "rule"
    assert outcome.usage.calls == 0
    assert outcome.usage.provider == "none"
    assert any("LLM을 쓸 수 없는 상태입니다" in n for n in outcome.notes)


def test_chunk_level_failures_keep_the_hybrid_and_say_so(runner: asyncio.Runner, bundle: InputBundle) -> None:
    model = failing_llm("extract", LLMOutputError("schema mismatch"))
    outcome = diagnose(runner, bundle, llm=model, mode=LlmMode.LIVE, rehearse=False, draft=False)
    assert outcome.extractor == "hybrid"
    assert any("구간이 응답 오류로 빠져" in n for n in outcome.notes)


def test_scenario_generation_failure_is_noted_not_fatal(runner: asyncio.Runner, bundle: InputBundle) -> None:
    outcome = diagnose(
        runner, bundle, llm=failing_llm("scenarios", LLMError("timeout")), mode=LlmMode.LIVE, draft=False
    )
    assert outcome.scenarios
    assert not [s for s in outcome.scenarios if s.origin is ScenarioOrigin.LLM]
    assert any(n.startswith("LLM 시나리오를 만들지 못해") for n in outcome.notes)


def test_describe_llm_error_is_korean_and_clipped() -> None:
    assert describe_llm_error(LLMDisabledError("off")) == "LLM을 쓸 수 없는 상태입니다(off)"
    assert describe_llm_error(LLMRefusalError("")) == "LLM이 응답을 거절했습니다"
    assert describe_llm_error(LLMOutputError("x")).startswith("LLM 응답이 정해진 형식과")
    long = describe_llm_error(LLMError("a\nb " * 200))
    assert long.startswith("LLM 호출이 실패했습니다(a b")
    assert len(long) < 200
    assert "\n" not in long


def test_usage_summary_reports_recorded_models() -> None:
    meter = UsageMeter()
    meter.add(LLMUsage(provider="cassette", model="claude-opus-5-5", task="extract", input_tokens=7))
    meter.add(LLMUsage(provider="cassette", model="claude-opus-5-5", task="extract", output_tokens=3))
    summary = usage_summary(LlmMode.REPLAY, meter, NullLLM())
    assert (summary.model, summary.calls, summary.input_tokens, summary.output_tokens) == (
        "claude-opus-5-5",
        2,
        7,
        3,
    )
    assert summary.by_task == {"extract": 2}
    assert usage_summary(LlmMode.OFF, meter, None) == LlmUsageSummary(mode=LlmMode.OFF)


# --------------------------------------------------------------------------- base graph & rebase


def approved(patch: Patch) -> Patch:
    return patch.model_copy(update={"status": PatchStatus.APPROVED})


def payment_owner_patch(outcome: DiagnosisOutcome) -> Patch:
    by_id = {f.id: f for f in outcome.findings}
    return next(
        p
        for p in outcome.patches
        if p.status is PatchStatus.VALIDATED
        and any(by_id[i].rule_id == "JT-PPL-001" and by_id[i].service_id == PAYMENT for i in p.finding_ids)
    )


def test_unchanged_inputs_reuse_the_head_with_its_patches(
    runner: asyncio.Runner, bundle: InputBundle, off: DiagnosisOutcome
) -> None:
    patch = approved(payment_owner_patch(off))
    head, _ = apply_ops(off.graph, patch.ops)
    again = diagnose(
        runner,
        bundle,
        base_graph=head,
        base_version=2,
        base_extraction_hash=off.extraction_hash,
        approved_patches=(patch,),
        rehearse=False,
    )
    assert again.graph.content_hash() == head.content_hash()
    assert again.graph_reused is True
    assert again.reapplied_patch_ids == (patch.id,)
    assert again.draft_base_version == 2
    assert again.extraction_hash == off.extraction_hash
    assert "rebase" in again.timings
    assert not [f for f in again.findings if f.rule_id == "JT-PPL-001" and f.service_id == PAYMENT]
    assert all(p.base_version == 2 for p in again.patches)


def test_changed_inputs_reapply_approved_patches(
    runner: asyncio.Runner, bundle: InputBundle, off: DiagnosisOutcome
) -> None:
    patch = approved(payment_owner_patch(off))
    head, _ = apply_ops(off.graph, patch.ops)
    [cart_doc] = [d for d in bundle.documents if d.path and d.path.startswith("runbooks/02")]
    edited = cart_doc.model_copy(update={"text": cart_doc.text + "\n\n> 2026-10 점검 완료.\n"})
    changed = bundle.model_copy(
        update={"documents": tuple(edited if d.doc_id == cart_doc.doc_id else d for d in bundle.documents)}
    )
    outcome = diagnose(
        runner,
        changed,
        base_graph=head,
        base_version=2,
        base_extraction_hash=off.extraction_hash,
        approved_patches=(patch,),
        rehearse=False,
    )
    assert outcome.graph_reused is False
    assert outcome.reapplied_patch_ids == (patch.id,)
    assert outcome.extraction_hash != off.extraction_hash
    assert outcome.draft_base_version == 3
    assert outcome.notes[0].startswith("입력(런북·명부·카탈로그 등)이 바뀌어")
    assert "1건 중 1건" in outcome.notes[0]
    added = [op.edge.id for op in patch.ops if isinstance(op, AddEdge)]
    assert added
    assert all(eid in outcome.graph.edges for eid in added)
    assert not [f for f in outcome.findings if f.rule_id == "JT-PPL-001" and f.service_id == PAYMENT]


def test_an_op_that_no_longer_fits_is_skipped_with_a_note(off: DiagnosisOutcome) -> None:
    graph = off.graph
    owner = edge_id(EdgeKind.OWNED_BY, PAYMENT, "person:김민준")
    stale_snapshot = graph.edges[owner].model_copy(
        update={"attrs": {"note": Attr(value="옛 값", source=SourceKind.PATCH)}}
    )
    patch = Patch(
        id="patch-stale",
        workspace_id="w",
        base_version=1,
        finding_ids=(),
        ops=(
            RemoveEdge(edge_id=owner, snapshot=stale_snapshot),
            SetAttr(
                target_id=PAYMENT,
                key=str(AttrKey.DESCRIPTION),
                value=Attr(value="결제", source=SourceKind.PATCH),
            ),
        ),
        runbook_text="t",
        rationale="r",
        author=PatchAuthor.HUMAN,
        status=PatchStatus.APPROVED,
    )
    base = graph.model_copy(update={"docs": {}})  # any different graph: the extraction changed
    result = rebase_graph(graph, base, base_extraction_hash="something-else", approved=(patch,))
    assert result.reused is False
    assert result.reapplied == ("patch-stale",)
    [summary, skipped] = result.notes
    assert "1건 중 1건" in summary
    assert skipped.startswith("승인된 수정안 patch-stale의 연산 1/2(remove_edge)은")
    assert "건너뛰었습니다" in skipped
    assert "1번째 연산" not in skipped
    assert owner in result.graph.edges
    assert result.graph.nodes[PAYMENT].attr(AttrKey.DESCRIPTION) == "결제"


def human_patch(patch_id: str, *ops: PatchOp) -> Patch:
    return Patch(
        id=patch_id,
        workspace_id="w",
        base_version=1,
        finding_ids=(),
        ops=ops,
        runbook_text="t",
        rationale="r",
        author=PatchAuthor.HUMAN,
        status=PatchStatus.APPROVED,
    )


def test_a_patch_with_no_fitting_op_is_not_counted_as_reapplied(off: DiagnosisOutcome) -> None:
    misfit = AddEdge(
        edge=Edge(
            id=edge_id(EdgeKind.OWNED_BY, PAYMENT, "person:없는사람"),
            kind=EdgeKind.OWNED_BY,
            src=PAYMENT,
            dst="person:없는사람",
            prov=(Provenance(source=SourceKind.PATCH, extractor=Extractor.HUMAN, ref="patch-misfit"),),
        )
    )
    result = rebase_graph(off.graph, off.graph, approved=(human_patch("patch-misfit", misfit),))
    assert result.reapplied == ()
    assert result.reused is True  # nothing landed, so the graph still equals the head
    assert result.graph is off.graph
    [skipped] = result.notes
    assert "person:없는사람" in skipped


def test_a_decision_the_new_extraction_already_reflects_is_kept_quietly(off: DiagnosisOutcome) -> None:
    """The removed edge is gone (the runbook no longer says it); the added edge is there as the
    runbook's own fact (the team wrote the approved sentence in): both decisions hold as they are."""
    gone = RemoveEdge(edge_id="owned_by:service:없음->person:없음")
    owner = off.graph.edges[edge_id(EdgeKind.OWNED_BY, PAYMENT, "person:김민준")]
    as_patched = owner.model_copy(
        update={"prov": (Provenance(source=SourceKind.PATCH, extractor=Extractor.HUMAN, ref="patch-x"),)}
    )
    result = rebase_graph(
        off.graph,
        off.graph,
        approved=(human_patch("patch-held", gone, AddEdge(edge=as_patched)),),
    )
    assert result.reapplied == ("patch-held",)
    assert result.notes == ()
    assert result.graph is off.graph  # the runbook's wording (with its spans) wins over the patch's copy


def moved(graph: OpsGraph, doc_id: str = "doc-edited", shift: int = 7) -> OpsGraph:
    """``graph`` as a re-extraction of edited runbooks returns it: the same facts, every span moved."""

    def span(s: SourceSpan) -> SourceSpan:
        return s.model_copy(update={"doc_id": doc_id, "start": s.start + shift, "end": s.end + shift})

    def prov(items: tuple[Provenance, ...]) -> tuple[Provenance, ...]:
        return tuple(p.model_copy(update={"spans": tuple(span(s) for s in p.spans)}) for p in items)

    def attrs(items: dict[str, Attr]) -> dict[str, Attr]:
        return {k: a.model_copy(update={"spans": tuple(span(s) for s in a.spans)}) for k, a in items.items()}

    return OpsGraph(
        nodes={
            k: n.model_copy(update={"prov": prov(n.prov), "attrs": attrs(n.attrs)})
            for k, n in graph.nodes.items()
        },
        edges={
            k: e.model_copy(update={"prov": prov(e.prov), "attrs": attrs(e.attrs)})
            for k, e in graph.edges.items()
        },
        docs=dict(graph.docs),
    )


def facts(graph: OpsGraph) -> tuple[dict[str, object], dict[str, object]]:
    """What a graph states, without where the text said it."""
    nodes: dict[str, object] = {
        k: (n.kind, n.name, {a: v.value for a, v in n.attrs.items()}) for k, n in graph.nodes.items()
    }
    edges: dict[str, object] = {
        k: (e.kind, e.src, e.dst, {a: v.value for a, v in e.attrs.items()}) for k, e in graph.edges.items()
    }
    return nodes, edges


def test_editing_a_runbook_keeps_every_approved_decision(off: DiagnosisOutcome) -> None:
    """Every validated demo fix, approved and then rebased onto an extraction whose spans all moved
    (one runbook line edited), states exactly what the approved head stated — no skips."""
    validated = [p for p in off.patches if p.status is PatchStatus.VALIDATED]
    assert len(validated) >= 5

    def removes(patch: Patch) -> bool:
        return any(isinstance(op, RemoveEdge) and op.snapshot is not None for op in patch.ops)

    assert any(removes(p) for p in validated)
    shifted = moved(off.graph)
    assert shifted.content_hash() != off.graph.content_hash()
    for patch in validated:
        head, _ = apply_ops(off.graph, patch.ops)
        if removes(patch):  # a strict re-application would refuse it (and resurrect the removed)
            with pytest.raises(PatchError):
                apply_ops(shifted, patch.ops)
        result = rebase_graph(
            shifted, head, base_extraction_hash=off.extraction_hash, approved=(approved(patch),)
        )
        assert result.reapplied == (patch.id,), patch.id
        assert len(result.notes) == 1, (patch.id, result.notes)  # the summary alone
        assert facts(result.graph) == facts(head), patch.id


def test_an_attribute_change_follows_its_value_not_its_spans(off: DiagnosisOutcome) -> None:
    span = SourceSpan(doc_id="doc-a", start=0, end=2, quote="결제")
    before = Attr(value="옛 설명", spans=(span,), source=SourceKind.RUNBOOK)
    graph, _ = apply_ops(off.graph, (SetAttr(target_id=PAYMENT, key=str(AttrKey.DESCRIPTION), value=before),))
    change = SetAttr(
        target_id=PAYMENT,
        key=str(AttrKey.DESCRIPTION),
        value=Attr(value="새 설명", source=SourceKind.PATCH),
        previous=before,
    )
    same_value_new_span = moved(graph)
    result = rebase_graph(
        same_value_new_span, graph, base_extraction_hash="x", approved=(human_patch("p", change),)
    )
    assert result.graph.nodes[PAYMENT].attr(AttrKey.DESCRIPTION) == "새 설명"
    assert len(result.notes) == 1
    newer, _ = apply_ops(
        graph,
        (
            SetAttr(
                target_id=PAYMENT,
                key=str(AttrKey.DESCRIPTION),
                value=before.model_copy(update={"value": "더 새 설명"}),
            ),
        ),
    )
    kept = rebase_graph(newer, graph, base_extraction_hash="x", approved=(human_patch("p", change),))
    assert kept.graph.nodes[PAYMENT].attr(AttrKey.DESCRIPTION) == "더 새 설명"  # the newer fact wins
    assert kept.reapplied == ()
    assert "건너뛰었습니다" in kept.notes[-1]


def test_matching_extraction_reuses_the_head_without_reapplying(off: DiagnosisOutcome) -> None:
    head = off.graph.model_copy(update={"docs": {}})
    result = rebase_graph(off.graph, head, base_extraction_hash=off.graph.content_hash(), approved=())
    assert result.graph is head
    assert result.reused is True
    assert result.notes == ()


def test_draft_version_follows_commit_semantics(off: DiagnosisOutcome) -> None:
    other = off.graph.model_copy(update={"docs": {}})
    assert draft_version(off.graph, None, None) == 1
    assert draft_version(off.graph, off.graph, 4) == 4
    assert draft_version(off.graph, other, 4) == 5


# --------------------------------------------------------------------------- stage functions


def test_carried_drills_survive_only_while_their_nodes_exist(
    off: DiagnosisOutcome, bundle: InputBundle
) -> None:
    ctx = build_context(
        off.graph, roster=bundle.roster, oncall=bundle.oncall, catalog=bundle.catalog, policy=bundle.policy
    )
    human = scenarios_from_yaml(DRILLS, ctx)
    gone = human[0].model_copy(
        update={"id": "human:gone", "title": "사라진 담당", "service_id": "service:없는-서비스"}
    )
    picked, notes = gather_scenarios(ctx, carried=(*human, gone))
    assert [s.id for s in picked if s.origin is ScenarioOrigin.HUMAN] == [s.id for s in human]
    [note] = notes
    assert note.startswith("시나리오 「사라진 담당」: 가리키는 대상(service:없는-서비스)이")
    fresh, _ = gather_scenarios(
        ctx, carried=(human[0].model_copy(update={"title": "옛 제목"}),), sources=(DRILLS,)
    )
    assert next(s for s in fresh if s.id == human[0].id).title == human[0].title  # the fresh file wins


def test_evaluate_graph_judges_like_the_pipeline(off: DiagnosisOutcome, bundle: InputBundle) -> None:
    carried = tuple(s for s in off.scenarios if s.origin is ScenarioOrigin.HUMAN)
    evaluation = evaluate_graph(
        off.graph,
        roster=bundle.roster,
        oncall=bundle.oncall,
        catalog=bundle.catalog,
        policy=bundle.policy,
        workspace_id="local",
        version=1,
        carried=carried,
    )
    assert evaluation.findings == off.findings
    assert evaluation.readiness == off.readiness
    assert [s.id for s in evaluation.scenarios] == [s.id for s in off.scenarios]
    assert [r.result_hash() for r in evaluation.rehearsals] == [r.result_hash() for r in off.rehearsals]
    assert [p.model_dump() for p in evaluation.patches] == [p.model_dump() for p in off.patches]
    assert evaluation.catalog_gaps == off.catalog_gaps
    assert set(evaluation.timings) == {"check", "score", "scenarios", "rehearse", "explore", "draft"}
    assert evaluation.exploration is not None
    assert evaluation.exploration.hypotheses == off.risk_hypotheses
    # the same sweep and verdicts; only the reason for "no AI proposals" differs (re-judgement)
    assert off.exploration is not None
    assert evaluation.exploration.summary.ai_status is AiProposalStatus.REJUDGE
    assert evaluation.exploration.summary.model_copy(update={"ai_status": off.exploration.ai_status}) == (
        off.exploration
    )
    quiet = evaluate_graph(off.graph, workspace_id="local", version=1, rehearse=False, draft=False)
    assert (quiet.scenarios, quiet.rehearsals, quiet.patches) == ((), (), ())
    assert quiet.exploration is None


def test_validate_drafts_keeps_memos_as_drafts(off: DiagnosisOutcome, bundle: InputBundle) -> None:
    ctx = CheckContext(off.graph, bundle.policy, bundle.roster, bundle.oncall, bundle.catalog)
    memo = next(p for p in off.patches if not p.ops).model_copy(
        update={"status": PatchStatus.DRAFT, "validation": None}
    )
    [checked] = validate_drafts(
        ctx, [memo], scenarios=(), findings=off.findings, results=(), config=SimConfig()
    )
    assert checked.status is PatchStatus.DRAFT
    assert checked.validation is not None
    assert not checked.validation.passed


def test_validate_drafts_rejects_ops_that_do_not_fit(off: DiagnosisOutcome, bundle: InputBundle) -> None:
    ctx = CheckContext(off.graph, bundle.policy, bundle.roster, bundle.oncall, bundle.catalog)
    broken = next(p for p in off.patches if p.ops).model_copy(
        update={"ops": (RemoveEdge(edge_id="owned_by:service:없음->person:없음"),), "validation": None}
    )
    [checked] = validate_drafts(
        ctx, [broken], scenarios=(), findings=off.findings, results=(), config=SimConfig()
    )
    assert checked.status is PatchStatus.REJECTED_BY_VALIDATOR
    assert checked.validation is not None
    assert checked.validation.integrity_errors


def test_held_elements_and_catalog_gaps_are_sorted_and_unique() -> None:
    builder = GraphBuilder()
    prov = CATALOG_PROV
    builder.add_node(
        Node(id="service:b", kind=NodeKind.SERVICE, name="b", prov=prov, verification=Verification.HOLD)
    )
    builder.add_node(Node(id="service:a", kind=NodeKind.SERVICE, name="a", prov=prov))
    builder.add_node(Node(id="system:z", kind=NodeKind.SYSTEM, name="z", prov=prov))
    builder.add_edge(
        Edge(
            id=edge_id(EdgeKind.DEPENDS_ON, "service:a", "system:z"),
            kind=EdgeKind.DEPENDS_ON,
            src="service:a",
            dst="system:z",
            prov=prov,
            verification=Verification.HOLD,
        )
    )
    graph = builder.build()
    assert held_element_ids(graph) == ("depends_on:service:a->system:z", "service:b")
    gap = Finding(
        id="g1",
        rule_id=CATALOG_GAP_RULE,
        severity=Severity.HIGH,
        title="t",
        message="m",
        service_id="service:b",
    )
    dup = gap.model_copy(update={"id": "g2"})
    other = gap.model_copy(update={"id": "g3", "rule_id": "JT-PATH-001", "service_id": "service:a"})
    org = gap.model_copy(update={"id": "g4", "service_id": None})
    assert catalog_gaps([gap, dup, other, org]) == ("service:b",)


def _one_service(name: str) -> OpsGraph:
    builder = GraphBuilder()
    builder.add_node(Node(id=f"service:{name}", kind=NodeKind.SERVICE, name=name, prov=CATALOG_PROV))
    return builder.build()


@settings(max_examples=60, deadline=None)
@given(
    head=st.integers(min_value=1, max_value=10_000),
    names=st.lists(st.sampled_from(["a", "b", "c"]), min_size=2, max_size=2),
)
def test_draft_version_is_the_head_for_an_equal_graph_and_the_next_one_otherwise(
    head: int, names: list[str]
) -> None:
    graph, base = (_one_service(n) for n in names)
    expected = head if names[0] == names[1] else head + 1
    assert draft_version(graph, base, head) == expected
    assert draft_version(graph, None, None) == 1
