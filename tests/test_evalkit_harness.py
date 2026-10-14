"""Evaluation harness: canonical facts, conditions A–O, the runner, metrics, benchmarks, report."""

from __future__ import annotations

import asyncio
import dataclasses
import gc
import json
import math
import os
import re
import shutil
import unicodedata
import warnings
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, TypeVar

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import BaseModel

from jettae.check.context import CheckContext
from jettae.check.registry import run_checks
from jettae.domain import (
    Attr,
    AttrKey,
    Document,
    Edge,
    EdgeKind,
    Extractor,
    GraphBuilder,
    Node,
    NodeKind,
    PersonStatus,
    Provenance,
    SourceKind,
    SourceSpan,
    edge_id,
    node_id,
)
from jettae.evalkit import (
    ALL_OPERATORS,
    DEFECT_RULE_IDS,
    EXPECTED_RULE_IDS,
    HELD_OUT_OPERATORS,
    KEYWORD_SIGNALS,
    OFFLINE_CONDITIONS,
    ORG_LEVEL_RULE_IDS,
    CaseResult,
    Condition,
    DefectType,
    DirectDefect,
    DirectDefectPayload,
    EvalCase,
    EvalResults,
    GoldOrg,
    Operator,
    apply,
    arun_llm_direct,
    arun_llm_graphs,
    build_cases,
    build_manifest,
    case_inputs,
    catalog_entries,
    check_reproducibility,
    compute_metrics,
    credited_keys,
    estimate_llm_cost,
    generate_gold_orgs,
    gold_facts,
    graph_facts,
    impact_benchmark,
    injection_selftest,
    is_hit,
    is_paired_hit,
    keyword_findings,
    offline_graph,
    oracle_graph,
    parse_conditions,
    policy_for,
    read_manifest,
    render_report,
    reproducibility_sample,
    roster_entries,
    run_eval,
    run_evaluation,
    run_offline_condition,
    structure_hash,
    synthetic_catalog_graph,
    target_service_for,
    ungrounded_elements,
    verify,
    wilson_interval,
    write_report,
)
from jettae.evalkit import conditions as conditions_module
from jettae.evalkit import harness as harness_module
from jettae.evalkit.conditions import (
    CASE_RUNBOOK_NAME,
    DIRECT_INSTRUCTIONS,
    DIRECT_TASK,
    SEVERE,
    direct_context,
    fired_keys,
    rule_severity,
    run_sync,
)
from jettae.evalkit.facts import RUNBOOK_ONLY_EDGE_KINDS, canonical_keys
from jettae.evalkit.gold import RULE_APPROVER, RULE_DEADLINE_LATE, RULE_OWNER, RULE_ROLLBACK, fact
from jettae.evalkit.harness import CHARS_PER_TOKEN, WINDOWS_MAX_WORKERS
from jettae.evalkit.metrics import DELETION_OPERATORS, EvalMetrics, Prf, Rate, SplitRate
from jettae.evalkit.report import main
from jettae.extract.llm_extractor import EXTRACT_TASK, LlmGraphPayload, LlmRelation, chunk_document
from jettae.extract.rules import RuleExtractor
from jettae.llm.base import (
    LLMOutputError,
    LLMResult,
    LLMUsage,
    NullLLM,
    UntrustedDocument,
)
from jettae.llm.fake import FakeLLM

T = TypeVar("T", bound=BaseModel)
ROOT = Path(__file__).resolve().parents[1]
DMC_SHOP = ROOT / "examples" / "dmc-shop"


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def orgs() -> list[GoldOrg]:
    return generate_gold_orgs()


@pytest.fixture(scope="module")
def cases(orgs: list[GoldOrg]) -> list[EvalCase]:
    """org01's 66 cases (6 clean controls + 10 operators × 3 styles × 2 positions)."""
    return build_cases(orgs[:1])


def _case(
    cases: list[EvalCase], operator: Operator | None, style: str = "explicit", pos: str = "front"
) -> EvalCase:
    return next(c for c in cases if c.operator is operator and c.style == style and c.position == pos)


def _result(case: EvalCase, condition: Condition, **changes: Any) -> CaseResult:
    base = CaseResult(
        case_id=case.case_id,
        condition=condition,
        fired=frozenset(),
        severe=frozenset(),
        facts=None,
        graph_hash=None,
        ungrounded=0,
        llm_calls=0,
        input_tokens=0,
        output_tokens=0,
        seconds=0.01,
    )
    return dataclasses.replace(base, **changes)


def _owner_quote(case: EvalCase) -> tuple[str, str, str]:
    """(service, owner, quote) of the target's (or focus service's) OWNED_BY statement."""
    span = next(
        s for s in case.fact_spans if s.fact.kind is EdgeKind.OWNED_BY and s.fact.src == case.focus_service
    )
    return span.fact.src, span.fact.dst, span.quote


def _extract_handler(case: EvalCase) -> Callable[..., Any]:
    """Fake LLM extraction: one OWNED_BY relation quoting the case's own owner sentence."""
    service, owner, quote = _owner_quote(case)

    def handler(task: str, schema: type[BaseModel], docs: tuple[UntrustedDocument, ...], ctx: str) -> Any:
        if task == EXTRACT_TASK:
            relation = LlmRelation(
                kind=EdgeKind.OWNED_BY,
                src_kind=NodeKind.SERVICE,
                src_name=service,
                dst_kind=NodeKind.PERSON,
                dst_name=owner,
                quote=quote,
            )
            return LlmGraphPayload(relations=(relation,))
        if task == DIRECT_TASK:
            return DirectDefectPayload()
        raise AssertionError(f"unexpected task {task}")

    return handler


class _TokenLLM:
    """Wraps a FakeLLM and reports fixed token usage, like a metered live adapter would."""

    provider = "fake"
    model = "fake-tokens"

    def __init__(self, inner: FakeLLM) -> None:
        self.inner = inner

    async def structured(
        self,
        *,
        task: str,
        instructions: str,
        schema: type[T],
        documents: tuple[UntrustedDocument, ...] = (),
        context: str = "",
        max_tokens: int = 16000,
    ) -> LLMResult[T]:
        result = await self.inner.structured(
            task=task,
            instructions=instructions,
            schema=schema,
            documents=documents,
            context=context,
            max_tokens=max_tokens,
        )
        usage = LLMUsage(provider="fake", model="fake-tokens", task=task, input_tokens=100, output_tokens=7)
        return LLMResult(data=result.data, usage=usage)


# --------------------------------------------------------------------------- facts


def test_graph_facts_of_oracle_graph_equal_gold_facts_for_every_org(orgs: list[GoldOrg]) -> None:
    for org in orgs:
        truth = gold_facts(org)
        graph = oracle_graph(truth, roster=roster_entries(org), catalog=catalog_entries(org))
        assert graph_facts(graph) == truth, org.org_id


@settings(max_examples=40, deadline=None)
@given(data=st.data())
def test_oracle_round_trip_holds_for_any_subset_of_gold_facts(data: st.DataObject) -> None:
    org = generate_gold_orgs(3)[data.draw(st.integers(0, 2))]
    facts = sorted(gold_facts(org))
    subset = frozenset(data.draw(st.lists(st.sampled_from(facts), unique=True, max_size=60)))
    graph = oracle_graph(subset, roster=roster_entries(org), catalog=catalog_entries(org))
    assert graph_facts(graph) == subset


def test_oracle_graph_marks_unknown_people_unknown_and_carries_gold_provenance(orgs: list[GoldOrg]) -> None:
    org = orgs[0]
    graph = oracle_graph(
        [fact(EdgeKind.OWNED_BY, org.services[0].name, "홍길동")], roster=roster_entries(org)
    )
    person = graph.nodes[node_id(NodeKind.PERSON, "홍길동")]
    assert person.attr(AttrKey.STATUS) == PersonStatus.UNKNOWN.value
    assert all(
        p.source is SourceKind.POLICY and p.ref == "gold" for n in graph.nodes.values() for p in n.prov
    )
    assert ungrounded_elements(graph, ()) == 0


def _prov() -> tuple[Provenance, ...]:
    return (Provenance(source=SourceKind.CATALOG, extractor=Extractor.IMPORT, ref="t"),)


def _node(kind: NodeKind, name: str, **attrs: Any) -> Node:
    values = {k: Attr(value=v, source=SourceKind.CATALOG) for k, v in attrs.items()}
    return Node(id=node_id(kind, name), kind=kind, name=name, attrs=values, prov=_prov())


def _edge(kind: EdgeKind, src: str, dst: str, **attrs: Any) -> Edge:
    values = {k: Attr(value=v, source=SourceKind.CATALOG) for k, v in attrs.items()}
    return Edge(id=edge_id(kind, src, dst), kind=kind, src=src, dst=dst, attrs=values, prov=_prov())


def test_graph_facts_canonicalise_shared_triggers_channels_and_untyped_actions() -> None:
    b = GraphBuilder()
    pay, auth = b.add_node(_node(NodeKind.SERVICE, "결제")), b.add_node(_node(NodeKind.SERVICE, "인증"))
    trig = b.add_node(_node(NodeKind.TRIGGER, "5xx 알람"))
    kim = b.add_node(_node(NodeKind.PERSON, "김철수"))
    slack = b.add_node(_node(NodeKind.CHANNEL, "업무 슬랙", medium="slack"))
    act = b.add_node(_node(NodeKind.ACTION, "캐시 비우기"))
    rb = b.add_node(_node(NodeKind.ACTION, "되돌리기", action_type="rollback"))
    for svc in (pay, auth):
        b.add_edge(_edge(EdgeKind.DETECTED_BY, svc.id, trig.id))
    b.add_edge(_edge(EdgeKind.NOTIFIES, trig.id, kim.id, channel=slack.id))
    b.add_edge(_edge(EdgeKind.MITIGATED_BY, pay.id, act.id))
    b.add_edge(_edge(EdgeKind.MITIGATED_BY, pay.id, rb.id))
    b.add_edge(_edge(EdgeKind.ESCALATES_TO, kim.id, kim.id, after_minutes="15"))
    graph = b.build()
    facts = graph_facts(graph)
    assert canonical_keys(graph)[trig.id] == ("결제#trigger", "인증#trigger")
    assert fact(EdgeKind.NOTIFIES, "결제#trigger", "김철수", channel="slack") in facts
    assert fact(EdgeKind.NOTIFIES, "인증#trigger", "김철수", channel="slack") in facts
    assert fact(EdgeKind.MITIGATED_BY, "결제", "결제#캐시 비우기") in facts  # never the gold action key
    assert fact(EdgeKind.MITIGATED_BY, "결제", "결제#rollback", action_type="rollback") in facts
    # a minute value the graph states as text is not a comparable number: left out, never guessed
    assert fact(EdgeKind.ESCALATES_TO, "김철수", "김철수") in facts


def test_ungrounded_elements_flags_broken_and_quarantined_evidence(cases: list[EvalCase]) -> None:
    case = _case(cases, None)
    inputs = case_inputs(case)
    graph = offline_graph(
        (inputs.document,), roster=inputs.roster, catalog=inputs.catalog, policy=inputs.policy
    )
    assert ungrounded_elements(graph, (inputs.document,)) == 0
    assert ungrounded_elements(graph, ()) > 0  # evidence pointing at an unknown document
    edited = inputs.document.model_copy(update={"text": inputs.document.text.replace("담당", "담담")})
    assert ungrounded_elements(graph, (edited,)) > 0
    quarantined = inputs.document.model_copy(update={"quarantined": ((0, len(inputs.document.text)),)})
    assert ungrounded_elements(graph, (quarantined,)) > 0


def test_ungrounded_elements_counts_runbook_attribute_without_span() -> None:
    doc = Document(doc_id="d", title="t", text="결제 담당 김철수")
    span = SourceSpan(doc_id="d", start=0, end=2, quote="결제")
    prov = (Provenance(source=SourceKind.RUNBOOK, extractor=Extractor.RULE, spans=(span,)),)
    bare = Node(
        id="service:결제", kind=NodeKind.SERVICE, name="결제", prov=prov, attrs={"tier": Attr(value=1)}
    )
    graph = GraphBuilder()
    graph.add_node(bare)
    assert ungrounded_elements(graph.build(), (doc,)) == 1


# --------------------------------------------------------------------------- spec ↔ real rules


def test_real_rules_on_oracle_graphs_match_the_published_mapping(orgs: list[GoldOrg]) -> None:
    """The rules themselves are sound: clean orgs are silent and every operator fires exactly its
    expected rule (plus recorded collateral) at the target, nothing elsewhere."""
    for org in orgs:
        policy, catalog = policy_for(org), catalog_entries(org)
        clean = oracle_graph(gold_facts(org), roster=roster_entries(org), catalog=catalog)
        assert run_checks(CheckContext(clean, policy, roster_entries(org), catalog=catalog)) == []
        for op in ALL_OPERATORS:
            target = target_service_for(org, op)
            m = apply(org, op, target)
            roster = roster_entries(m.org)
            graph = oracle_graph(gold_facts(m.org), roster=roster, catalog=catalog)
            fired, _ = fired_keys(graph, run_checks(CheckContext(graph, policy, roster, catalog=catalog)))
            at_target = {r for s, r in fired if s in (target, None)}
            assert EXPECTED_RULE_IDS[op] <= at_target, (org.org_id, op)
            assert at_target <= EXPECTED_RULE_IDS[op] | m.collateral_rule_ids, (org.org_id, op)
            assert {s for s, _ in fired} <= {target, None}, (org.org_id, op)


# --------------------------------------------------------------------------- conditions


def test_case_inputs_map_korean_roster_labels(cases: list[EvalCase]) -> None:
    case = _case(cases, Operator.DEPARTED_CONTACT)
    inputs = case_inputs(case)
    statuses = {e.status for e in inputs.roster}
    assert PersonStatus.DEPARTED in statuses
    assert all(e.channels for e in inputs.roster)
    assert {s.name for s in inputs.catalog} >= {r.service for r in case.regions}
    assert inputs.policy.regulated_services
    assert inputs.document.text == case.text  # generated text needs no normalisation


def test_oracle_condition_is_perfect_on_one_org(cases: list[EvalCase]) -> None:
    for case in cases:
        r = run_offline_condition(case, Condition.O_ORACLE)
        assert r.error is None
        assert r.facts == frozenset(case.gold_facts)
        assert r.ungrounded == 0
        if case.is_clean:
            assert not r.fired
        else:
            assert is_hit(case, r, strict=True), case.case_id


def test_rule_graph_condition_is_grounded_and_deterministic(cases: list[EvalCase]) -> None:
    for case in (_case(cases, None, "table"), _case(cases, Operator.DELETE_ROLLBACK, "colloquial", "back")):
        first = run_offline_condition(case, Condition.B_RULE_GRAPH)
        again = run_offline_condition(case, Condition.B_RULE_GRAPH)
        assert first.error is None
        assert first.ungrounded == 0
        assert first.graph_hash is not None
        assert re.fullmatch(r"[0-9a-f]{64}", first.graph_hash)
        assert first.facts
        assert (first.graph_hash, first.fired, first.facts) == (again.graph_hash, again.fired, again.facts)
        assert first.severe <= first.fired
        assert all(rule_severity(rule) in SEVERE for _, rule in first.severe)


def test_rule_graph_finds_a_deleted_rollback(cases: list[EvalCase]) -> None:
    case = _case(cases, Operator.DELETE_ROLLBACK)
    assert (case.target_service, RULE_ROLLBACK) in run_offline_condition(case, Condition.B_RULE_GRAPH).fired


def test_keyword_condition_cannot_see_roster_or_path_faults(cases: list[EvalCase]) -> None:
    for op in (Operator.DEPARTED_CONTACT, Operator.ESCALATION_CYCLE):
        for case in (c for c in cases if c.operator is op):
            assert not is_hit(case, run_offline_condition(case, Condition.A_KEYWORD))
    rollback = _case(cases, Operator.DELETE_ROLLBACK)
    result = run_offline_condition(rollback, Condition.A_KEYWORD)
    assert (rollback.target_service, RULE_ROLLBACK) in result.fired
    assert result.facts is None
    assert result.graph_hash is None
    assert {rule for _, rule in result.fired} <= set(KEYWORD_SIGNALS)


def _single_region(case: EvalCase, service: str, text: str) -> EvalCase:
    region = case.regions[0].model_copy(update={"service": service, "start": 0, "end": len(text)})
    return case.model_copy(update={"text": text, "regions": (region,)})


def test_keyword_findings_read_deadlines_against_the_right_law(cases: list[EvalCase]) -> None:
    base = _case(cases, None)
    inputs = case_inputs(base)
    service = base.focus_service
    late = _single_region(base, service, "- 신고: 침해사고는 48시간 이내 KISA 신고(담당: 김철수 CISO)")
    assert (service, RULE_DEADLINE_LATE) in keyword_findings(late, inputs.catalog, inputs.policy)
    privacy = _single_region(
        base, service, "- 신고: 개인정보 유출은 72시간 이내 개인정보보호위원회·KISA 신고"
    )
    assert (service, RULE_DEADLINE_LATE) not in keyword_findings(privacy, inputs.catalog, inputs.policy)
    days = _single_region(base, service, "침해사고가 나면 3일 안에 KISA에 알린다")
    assert (service, RULE_DEADLINE_LATE) in keyword_findings(days, inputs.catalog, inputs.policy)


def test_keyword_findings_scope_path_rules_to_critical_services(
    cases: list[EvalCase], orgs: list[GoldOrg]
) -> None:
    base = _case(cases, None)
    inputs = case_inputs(base)
    tier3 = next(s.name for s in orgs[0].services if s.tier == 3)
    assert (
        keyword_findings(_single_region(base, tier3, "메모 없음"), inputs.catalog, inputs.policy)
        == frozenset()
    )
    critical = base.focus_service
    found = keyword_findings(
        _single_region(base, critical, "알람이 오면 롤백한다"), inputs.catalog, inputs.policy
    )
    assert (critical, RULE_OWNER) in found
    assert (critical, RULE_APPROVER) in found  # a rollback with nobody named to approve it
    named = _single_region(
        base,
        critical,
        "담당 김철수. 알람 → 슬랙, 전화. 미응답 시 상향. 팀장 승인 후 롤백. 침해사고는 24시간 이내 KISA 신고",
    )
    assert keyword_findings(named, inputs.catalog, inputs.policy) == frozenset()


@pytest.mark.parametrize(
    ("text", "named"),
    [
        ("롤백한다. 승인자: 박현우 팀장", True),
        ("롤백한다. 승인권자 김지원", True),
        ("CTO님 승인을 받고 롤백한다", True),
        ("롤백한다. 승인자: 미정", False),
        ("롤백한다. 승인자: (미정)", False),
        ("롤백한다. 승인권자 없음", False),
        ("승인자 없이 바로 롤백한다", False),
        ("롤백한다. 승인자: TBD", False),
    ],
)
def test_keyword_approver_placeholder_is_not_a_name(cases: list[EvalCase], text: str, named: bool) -> None:
    base = _case(cases, None)
    inputs = case_inputs(base)
    service = base.focus_service
    found = keyword_findings(_single_region(base, service, text), inputs.catalog, inputs.policy)
    assert ((service, RULE_APPROVER) not in found) is named


def test_run_offline_condition_rejects_llm_arms_and_records_failures(cases: list[EvalCase]) -> None:
    case = _case(cases, None)
    with pytest.raises(ValueError, match="LLM 없이"):
        run_offline_condition(case, Condition.C_LLM_DIRECT)
    broken = case.model_copy(update={"catalog_yaml": "services: [unclosed"})
    result = run_offline_condition(broken, Condition.B_RULE_GRAPH)
    assert result.error is not None
    assert not result.fired
    assert result.facts is None


def test_fired_keys_use_service_names_and_keep_org_level_findings(cases: list[EvalCase]) -> None:
    case = _case(cases, Operator.DELETE_REPORTING_STEP)
    inputs = case_inputs(case)
    graph = oracle_graph(case.gold_facts, roster=inputs.roster, catalog=inputs.catalog)
    findings = run_checks(CheckContext(graph, inputs.policy, inputs.roster, catalog=inputs.catalog))
    fired, severe = fired_keys(graph, findings)
    assert {s for s, _ in fired} <= {r.service for r in case.regions} | {None}
    assert severe <= fired


def test_run_sync_reuses_one_event_loop() -> None:
    async def loop_id() -> int:
        import asyncio

        return id(asyncio.get_running_loop())

    assert run_sync(loop_id()) == run_sync(loop_id())


# --------------------------------------------------------------------------- LLM conditions (fake)


def test_llm_direct_maps_defects_and_drops_invented_services(cases: list[EvalCase]) -> None:
    case = _case(cases, Operator.DEPARTED_CONTACT)
    target = case.target_service
    assert target is not None

    def handler(task: str, schema: type[BaseModel], docs: tuple[UntrustedDocument, ...], ctx: str) -> Any:
        assert task == DIRECT_TASK
        assert schema is DirectDefectPayload
        return DirectDefectPayload(
            defects=(
                DirectDefect(service=target, defect=DefectType.MISSING_OWNER),
                DirectDefect(service="없는 서비스", defect=DefectType.MISSING_ROLLBACK),
                DirectDefect(service="", defect=DefectType.DEPARTED_CONTACT),
            )
        )

    fake = FakeLLM(handler)
    result = run_sync(arun_llm_direct(case, _TokenLLM(fake)))
    assert result.error is None
    assert result.fired == {(target, RULE_OWNER), (None, "JT-PPL-001")}
    assert result.severe == result.fired  # both rules are CRITICAL
    assert (result.llm_calls, result.input_tokens, result.output_tokens) == (1, 100, 7)
    call = fake.calls[0]
    assert "퇴사" in call.context
    assert "신고 의무 적용 서비스" in call.context
    assert call.documents[0].text == case.text
    assert all(d.value in DIRECT_INSTRUCTIONS for d in DefectType)
    assert is_hit(case, result)  # departed contact found at organisation level


def test_defect_types_cover_exactly_the_published_operator_rules() -> None:
    assert set(DEFECT_RULE_IDS.values()) == {r for rules in EXPECTED_RULE_IDS.values() for r in rules}
    assert len(DEFECT_RULE_IDS) == len(DefectType) == 10


def test_llm_direct_records_model_failures(cases: list[EvalCase]) -> None:
    fake = FakeLLM(lambda *_: LLMOutputError("스키마 위반"))
    result = run_sync(arun_llm_direct(_case(cases, None), fake))
    assert result.error is not None
    assert result.error.startswith("LLMOutputError")
    # the model did answer (and was billed); only the tokens of a rejected answer are unknown
    assert (result.llm_calls, result.input_tokens) == (1, 0)
    assert not result.fired
    disabled = run_sync(arun_llm_direct(_case(cases, None), NullLLM()))
    assert disabled.error is not None
    assert disabled.error.startswith("LLMDisabledError")
    assert disabled.llm_calls == 0  # nothing was sent


def test_llm_graphs_share_one_extraction_and_hybrid_adds_rules(cases: list[EvalCase]) -> None:
    case = _case(cases, Operator.DELETE_APPROVER)
    fake = FakeLLM(_extract_handler(case))
    d, dr = run_sync(arun_llm_graphs(case, _TokenLLM(fake), [Condition.D_LLM_GRAPH, Condition.DR_HYBRID]))
    assert len(fake.calls) == len(chunk_document(case_inputs(case).document)) == 1
    assert (d.condition, dr.condition) == (Condition.D_LLM_GRAPH, Condition.DR_HYBRID)
    assert d.error is None
    assert dr.error is None
    assert d.llm_calls == dr.llm_calls == 1
    assert d.input_tokens == 100
    service, owner, _ = _owner_quote(case)
    assert d.facts is not None
    assert dr.facts is not None
    assert fact(EdgeKind.OWNED_BY, service, owner) in d.facts
    rule_only = run_offline_condition(case, Condition.B_RULE_GRAPH)
    assert rule_only.facts is not None
    assert dr.facts >= d.facts
    assert dr.facts >= rule_only.facts
    assert d.ungrounded == dr.ungrounded == 0
    # with almost nothing extracted, D sees the whole response path as missing
    assert (service, "JT-PATH-001") in d.fired


def test_llm_graphs_mark_rejected_chunks_and_disabled_models_as_errors(cases: list[EvalCase]) -> None:
    case = _case(cases, None)
    rejected = run_sync(
        arun_llm_graphs(case, FakeLLM(lambda *_: LLMOutputError("x")), [Condition.D_LLM_GRAPH])
    )
    assert rejected[0].error == "LLM 응답이 거부된 청크 1/1개"
    assert rejected[0].llm_calls == 1  # the rejected chunk was still a request to the model
    disabled = run_sync(arun_llm_graphs(case, NullLLM(), [Condition.D_LLM_GRAPH, Condition.DR_HYBRID]))
    assert [r.condition for r in disabled] == [Condition.D_LLM_GRAPH, Condition.DR_HYBRID]
    assert all(r.error is not None and r.error.startswith("LLMDisabledError") for r in disabled)
    assert all(r.llm_calls == 0 for r in disabled)
    with pytest.raises(ValueError, match="D, DR"):
        run_sync(arun_llm_graphs(case, NullLLM(), [Condition.C_LLM_DIRECT]))


# --------------------------------------------------------------------------- harness


def test_parse_conditions_validates_and_deduplicates() -> None:
    assert parse_conditions(["B_rule_graph", Condition.A_KEYWORD, "B_rule_graph"]) == (
        Condition.B_RULE_GRAPH,
        Condition.A_KEYWORD,
    )
    with pytest.raises(ValueError, match="알 수 없는"):
        parse_conditions(["Z_magic"])
    with pytest.raises(ValueError, match="하나 이상"):
        parse_conditions([])


def test_run_eval_refuses_llm_arms_without_an_explicit_client(cases: list[EvalCase]) -> None:
    with pytest.raises(ValueError, match="llm 인자를 명시"):
        run_eval(cases[:1], ["C_llm_direct"])
    with pytest.raises(ValueError, match="중복"):
        run_eval([cases[0], cases[0]], ["A_keyword"], workers=1)
    with pytest.raises(ValueError, match="concurrency"):
        run_eval(cases[:1], ["A_keyword"], workers=1, concurrency=0)


def test_run_eval_orders_results_and_pool_matches_inline(cases: list[EvalCase]) -> None:
    sample = cases[::11]
    seen: list[tuple[int, int]] = []
    inline = run_eval(sample, ["O_oracle", "A_keyword"], workers=1, progress=lambda d, t: seen.append((d, t)))
    assert [(r.case_id, r.condition) for r in inline.results] == [
        (c.case_id, cond) for c in sample for cond in (Condition.O_ORACLE, Condition.A_KEYWORD)
    ]
    assert seen[-1] == (len(sample) * 2, len(sample) * 2)
    pooled = run_eval(sample, ["O_oracle", "A_keyword"], workers=2)

    def view(r: CaseResult) -> tuple[object, ...]:
        return (r.case_id, r.condition, r.fired, r.severe, r.facts, r.graph_hash, r.error)

    assert [view(r) for r in pooled.results] == [view(r) for r in inline.results]
    assert (inline.workers, pooled.workers) == (1, 2)
    assert inline.of(Condition.A_KEYWORD)[0].case_id == sample[0].case_id
    assert inline.case(sample[1].case_id) == sample[1]
    with pytest.raises(KeyError):
        inline.get(sample[0].case_id, Condition.B_RULE_GRAPH)


def test_run_eval_runs_every_arm_with_a_fake_llm(cases: list[EvalCase]) -> None:
    sample = [_case(cases, None), _case(cases, Operator.DELETE_ROLLBACK)]
    by_text = {c.text: _extract_handler(c) for c in sample}

    def handler(task: str, schema: type[BaseModel], docs: tuple[UntrustedDocument, ...], ctx: str) -> Any:
        return by_text[docs[0].text](task, schema, docs, ctx)

    fake = FakeLLM(handler)
    results = run_eval(sample, list(Condition), llm=fake, workers=1, concurrency=2)
    assert [(r.case_id, r.condition) for r in results.results] == [
        (c.case_id, cond) for c in sample for cond in Condition
    ]
    assert len(fake.calls) == 2 * 2  # per case: one direct call + one shared extraction
    assert all(r.error is None for r in results.results)
    assert {r.llm_calls for r in results.results if r.condition in OFFLINE_CONDITIONS} == {0}


def test_reproducibility_sample_is_a_fixed_hash_draw(cases: list[EvalCase]) -> None:
    sample = reproducibility_sample(cases, 0.05)
    assert len(sample) == math.ceil(len(cases) * 0.05)
    assert {c.case_id for c in sample} == {c.case_id for c in reproducibility_sample(cases[::-1], 0.05)}
    assert [c.case_id for c in sample] == [c.case_id for c in cases if c in sample]  # corpus order
    assert reproducibility_sample([], 0.5) == []
    with pytest.raises(ValueError, match="표본 비율"):
        reproducibility_sample(cases, 0)


@settings(max_examples=25, deadline=None)
@given(fraction=st.floats(min_value=0.001, max_value=1.0), n=st.integers(1, 66))
def test_reproducibility_sample_size_property(cases: list[EvalCase], fraction: float, n: int) -> None:
    subset = cases[:n]
    sample = reproducibility_sample(subset, fraction)
    assert len(sample) == math.ceil(n * fraction)
    assert {c.case_id for c in sample} <= {c.case_id for c in subset}


def test_check_reproducibility_compares_hashes_and_fired_sets(cases: list[EvalCase]) -> None:
    sample = cases[:20]
    results = run_eval(sample, ["B_rule_graph"], workers=1)
    report = check_reproducibility(results, fraction=0.1)
    assert (report.sample_size, report.identical, report.mismatched, report.rate) == (2, 2, (), 1.0)
    assert report.fresh_process
    chosen = reproducibility_sample(sample, 0.1)[0].case_id
    tampered = EvalResults(
        cases=results.cases,
        conditions=results.conditions,
        results=tuple(
            dataclasses.replace(r, graph_hash="0" * 64) if r.case_id == chosen else r for r in results.results
        ),
    )
    in_process = check_reproducibility(tampered, fraction=0.1, fresh_process=False)
    assert in_process.mismatched == (chosen,)
    assert not in_process.fresh_process
    with pytest.raises(ValueError, match="LLM 없는"):
        check_reproducibility(results, condition=Condition.D_LLM_GRAPH)
    with pytest.raises(ValueError, match="결과에 없는"):
        check_reproducibility(results, condition=Condition.O_ORACLE)


def test_estimate_llm_cost_replays_request_construction_offline(cases: list[EvalCase]) -> None:
    sample = cases[:4]
    printed: list[str] = []
    estimate = estimate_llm_cost(
        sample, ["A_keyword", "C_llm_direct", "D_llm_graph", "DR_hybrid"], model="m", echo=printed.append
    )
    per = {c.condition: c for c in estimate.per_condition}
    chunks = sum(len(chunk_document(case_inputs(c).document)) for c in sample)
    assert set(per) == {Condition.C_LLM_DIRECT, Condition.D_LLM_GRAPH, Condition.DR_HYBRID}
    assert per[Condition.C_LLM_DIRECT].calls == 4
    assert per[Condition.D_LLM_GRAPH].calls == per[Condition.DR_HYBRID].calls == chunks
    assert estimate.total_calls == 4 + chunks  # D and DR share their extraction
    assert per[Condition.C_LLM_DIRECT].input_tokens > sum(len(c.text) for c in sample) / CHARS_PER_TOKEN
    assert printed
    assert "LLM 비용 추정" in printed[0]
    assert "모델 m" in printed[0]
    offline = estimate_llm_cost(sample, ["B_rule_graph"], model="m", echo=None)
    assert (offline.total_calls, offline.per_condition) == (0, ())


# --------------------------------------------------------------------------- metrics


def test_wilson_interval_known_values() -> None:
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0
    assert hi == pytest.approx(0.2775, abs=1e-4)
    lo, hi = wilson_interval(10, 10)
    assert lo == pytest.approx(0.7225, abs=1e-4)
    assert hi == 1.0
    lo, hi = wilson_interval(81, 263)
    assert (lo, hi) == pytest.approx((0.2553, 0.3662), abs=1e-4)
    with pytest.raises(ValueError, match="표본 수"):
        wilson_interval(1, 0)
    with pytest.raises(ValueError, match="범위"):
        wilson_interval(3, 2)


@given(n=st.integers(1, 10_000), data=st.data())
def test_wilson_interval_brackets_the_estimate(n: int, data: st.DataObject) -> None:
    k = data.draw(st.integers(0, n))
    lo, hi = wilson_interval(k, n)
    assert 0.0 <= lo <= k / n + 1e-12 <= hi + 2e-12 <= 1.0 + 2e-12


def test_rate_and_prf_handle_empty_denominators() -> None:
    assert Rate.of(0, 0).rate is None
    assert Rate.of(0, 0).ci95 is None
    assert Rate.of(3, 4).rate == 0.75
    assert Prf.of(0, 0, 0).f1 is None
    assert Prf.of(5, 0, 5).model_dump() == {
        "tp": 5,
        "fp": 0,
        "fn": 5,
        "precision": 1.0,
        "recall": 0.5,
        "f1": pytest.approx(0.6667),
    }


def test_compute_metrics_counts_hits_alarms_fields_and_inventions(cases: list[EvalCase]) -> None:
    clean = _case(cases, None)
    clean_err = _case(cases, None, "table", "back")
    owner = _case(cases, Operator.DELETE_OWNER)
    owner_b = _case(cases, Operator.DELETE_OWNER, "table")
    deadline = _case(cases, Operator.DEADLINE_CONTRADICTION)
    detect = _case(cases, Operator.DELETE_DETECTION)
    chosen = [clean, clean_err, owner, owner_b, deadline, detect]
    gold = {c.case_id: frozenset(c.gold_facts) for c in chosen}
    invented = owner.deleted_facts[0]
    b = Condition.B_RULE_GRAPH
    results = EvalResults(
        cases=tuple(chosen),
        conditions=(b,),
        results=(
            _result(clean, b, fired=frozenset({("x", "JT-INT-001")}), facts=gold[clean.case_id]),
            _result(clean_err, b, error="boom"),
            # an owner rule without a service localises nothing: never a hit
            _result(owner, b, fired=frozenset({(None, RULE_OWNER)}), facts=gold[owner.case_id] | {invented}),
            _result(owner_b, b, fired=frozenset({(owner_b.target_service, "JT-ESC-001")}), facts=frozenset()),
            # a deadline shared by several services is reported org-wide by the fixed rules: a hit
            _result(deadline, b, fired=frozenset({(None, RULE_DEADLINE_LATE)}), facts=gold[deadline.case_id]),
            _result(
                detect,
                b,
                fired=frozenset({(detect.target_service, "JT-PATH-001")}),
                severe=frozenset({(detect.target_service, "JT-PATH-001")}),
                facts=gold[detect.case_id],
                seconds=0.03,
            ),
        ),
    )
    m = compute_metrics(results, generated_at="2026-10-13", seed=1)
    s = m.conditions[b.value]
    assert (s.detection_active.hits, s.detection_active.n) == (1, 3)  # org-level DDL-002 counts
    assert (s.detection_strict_active.hits, s.detection_strict_active.n) == (0, 3)
    assert (s.detection_held_out.hits, s.detection_held_out.n) == (1, 1)
    assert (s.clean_fp.hits, s.clean_fp.n) == (1, 2)  # an errored control counts as an alarm
    assert m.clean_fp[b.value].per_rule["JT-INT-001"].cases == 1
    assert s.errors == 1
    hallucination = m.hallucination[b.value]
    assert hallucination.by_operator[Operator.DELETE_OWNER.value].hits == 1
    assert hallucination.rate.hits == 1
    field = m.field_f1[b.value]
    runbook_gold = sum(len([f for f in gold[c.case_id] if f.kind in RUNBOOK_ONLY_EDGE_KINDS]) for c in chosen)
    assert field.all.tp + field.all.fn == runbook_gold
    assert field.all.fp == 1  # the invented owner
    assert m.by_operator[b.value][Operator.DELETE_DETECTION.value].held_out is True
    assert m.corpus.n_clean == 2
    assert m.corpus.seed == 1
    assert m.manifest_root is None
    assert m.conditions_pending == ("C_llm_direct", "D_llm_graph", "DR_hybrid")
    assert s.latency.max_s == 0.03
    assert m.workers == 1
    assert list(m.by_position[b.value]) == ["front", "back"]


def test_is_hit_never_credits_clean_controls_or_errors(cases: list[EvalCase]) -> None:
    clean = _case(cases, None)
    owner = _case(cases, Operator.DELETE_OWNER)
    fired = frozenset({(owner.target_service, RULE_OWNER)})
    assert is_hit(owner, _result(owner, Condition.O_ORACLE, fired=fired), strict=True)
    assert not is_hit(owner, _result(owner, Condition.O_ORACLE, fired=fired, error="x"))
    assert not is_hit(clean, _result(clean, Condition.O_ORACLE, fired=fired))
    assert set(DELETION_OPERATORS) & set(HELD_OUT_OPERATORS) == set(HELD_OUT_OPERATORS)


# --------------------------------------------------------------------------- benchmarks


def test_impact_benchmark_times_a_deterministic_graph() -> None:
    a = synthetic_catalog_graph(20, 5, 2, seed=7)
    assert a.content_hash() == synthetic_catalog_graph(20, 5, 2, seed=7).content_hash()
    assert a.content_hash() != synthetic_catalog_graph(20, 5, 2, seed=8).content_hash()
    bench = impact_benchmark(n_services=20, n_systems=5, n_locations=2, runs=3, seed=7)
    assert (bench.n_services, bench.runs, bench.n_nodes) == (20, 3, 27)
    assert 0 < bench.impacted_services <= 20
    assert bench.min_ms <= bench.median_ms <= bench.max_ms
    with pytest.raises(ValueError, match="반복 횟수"):
        impact_benchmark(runs=0)
    with pytest.raises(ValueError, match="1 이상"):
        synthetic_catalog_graph(0, 1, 1)


def test_structure_hash_ignores_offsets_but_not_content() -> None:
    def graph(start: int, name: str, quote: str) -> Any:
        span = SourceSpan(doc_id=f"doc-{start}", start=start, end=start + len(quote), quote=quote)
        prov = (Provenance(source=SourceKind.RUNBOOK, extractor=Extractor.RULE, spans=(span,)),)
        b = GraphBuilder()
        b.add_node(Node(id=node_id(NodeKind.PERSON, name), kind=NodeKind.PERSON, name=name, prov=prov))
        return b.build()

    base = graph(0, "김철수", "김철수")
    assert structure_hash(base) == structure_hash(graph(40, "김철수", "김철수"))
    assert structure_hash(base) != structure_hash(graph(0, "박영희", "김철수"))
    assert structure_hash(base) == structure_hash(graph(0, "김철수", "김철수 대리"))
    assert structure_hash(base, quotes=True) != structure_hash(graph(0, "김철수", "김철수 대리"), quotes=True)


def test_injection_selftest_leaves_the_diagnosis_untouched() -> None:
    result = injection_selftest(DMC_SHOP, n=4)
    assert result.n == len(result.cases) == 4
    assert result.unauthorised_changes == 0
    assert result.quarantined == 4
    assert result.llm == "off"
    assert result.document.startswith("runbooks/")
    assert all(c.structure_unchanged and c.findings_unchanged and c.score_unchanged for c in result.cases)
    with pytest.raises(ValueError, match="묶음에 없는"):
        injection_selftest(DMC_SHOP, n=1, document="runbooks/없음.md")


# --------------------------------------------------------------------------- report


REQUIRED_KEYS = {
    "conditions",
    "by_style",
    "by_operator",
    "clean_fp",
    "field_f1",
    "hallucination",
    "ungrounded_total",
    "reproducibility",
    "n_cases",
    "manifest_root",
    "generated_at",
}


def test_write_report_publishes_metrics_report_and_results(cases: list[EvalCase], tmp_path: Path) -> None:
    sample = cases[::5]
    results = run_eval(sample, list(OFFLINE_CONDITIONS), workers=1)
    manifest = build_manifest(sample)
    metrics = write_report(
        results,
        tmp_path,
        generated_at="2026-10-13",
        manifest=manifest,
        reproducibility=check_reproducibility(results, fraction=0.2),
        impact=impact_benchmark(n_services=10, n_systems=3, n_locations=2, runs=2),
        injection=injection_selftest(DMC_SHOP, n=2),
        cost_estimate=estimate_llm_cost(sample, ["C_llm_direct"], model="m", echo=None),
    )
    data = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert set(data) >= REQUIRED_KEYS
    assert data["generated_at"] == "2026-10-13"
    assert data["n_cases"] == len(sample)
    assert data["manifest_root"] == manifest.merkle_root
    assert data["reproducibility"]["rate"] == 1.0
    assert data["ungrounded_total"] == 0
    assert set(data["conditions"]) == {c.value for c in OFFLINE_CONDITIONS}
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert report.startswith("# 제때옵스 봉인 평가 결과 (2026-10-13)")
    for heading in ("## 3. 문체별 탐지율", "## 6. 정상 대조군 오탐", "## 13. LLM 조건", "## 14. 한계"):
        assert heading in report
    assert "ANTHROPIC_API_KEY" in report
    assert report == render_report(metrics)  # rendering is a pure function of the metrics
    lines = (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(results.results)
    first = json.loads(lines[0])
    assert first["case_id"] == sample[0].case_id
    assert first["condition"] == "A_keyword"


def test_main_estimates_and_refuses_unconfirmed_paid_runs(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--estimate-only", "--orgs", "1"]) == 0
    out = capsys.readouterr().out
    assert "LLM 비용 추정" in out
    assert "사례 66건" in out
    assert main(["--orgs", "1", "--conditions", "A_keyword"]) == 2
    assert "--generated-at" in capsys.readouterr().err
    assert main(["--orgs", "1", "--conditions", "C_llm_direct", "--generated-at", "2026-10-13"]) == 2
    assert "--confirm-paid" in capsys.readouterr().err


def test_run_evaluation_end_to_end_on_one_org(tmp_path: Path) -> None:
    metrics = run_evaluation(
        tmp_path, generated_at="2026-10-13", n_orgs=1, workers=1, benchmarks=False, model="m"
    )
    manifest = read_manifest(tmp_path / "manifest.json")
    assert verify(build_cases(generate_gold_orgs(1)), manifest)
    assert metrics.manifest_root == manifest.merkle_root
    assert metrics.n_cases == 66
    assert metrics.reproducibility is not None
    assert metrics.reproducibility.rate == 1.0
    assert metrics.conditions["O_oracle"].detection_active.rate == 1.0
    assert metrics.conditions["O_oracle"].clean_fp.rate == 0.0
    assert metrics.llm_cost_estimate is not None
    assert metrics.llm_cost_estimate.model == "m"
    assert metrics.impact_benchmark is None
    assert metrics.injection_selftest is None
    assert {p.name for p in tmp_path.iterdir()} == {
        "manifest.json",
        "metrics.json",
        "report.md",
        "results.jsonl",
    }


# --------------------------------------------------------------------------- review: adversarial cases


def test_llm_graph_failures_after_extraction_are_recorded_not_raised(
    cases: list[EvalCase], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash after the (paid) extraction must become that case's error, never abort the run."""
    case = _case(cases, Operator.DELETE_APPROVER)
    both = [Condition.D_LLM_GRAPH, Condition.DR_HYBRID]

    def crash(*_: Any, **__: Any) -> Any:
        raise RuntimeError("judge exploded")

    monkeypatch.setattr(conditions_module, "judge_graph", crash)
    d, dr = run_sync(arun_llm_graphs(case, FakeLLM(_extract_handler(case)), both))
    assert d.error == dr.error == "RuntimeError: judge exploded"
    assert d.llm_calls == dr.llm_calls == 1  # the paid call is still accounted for

    monkeypatch.undo()
    monkeypatch.setattr(RuleExtractor, "extract", crash)
    d, dr = run_sync(arun_llm_graphs(case, FakeLLM(_extract_handler(case)), both))
    assert d.error is None  # D never runs the rule parser
    assert dr.error == "RuntimeError: judge exploded"
    # and the whole harness keeps every other result
    results = run_eval([case], ["DR_hybrid", "D_llm_graph"], llm=FakeLLM(_extract_handler(case)), workers=1)
    assert [(r.condition, r.error is None) for r in results.results] == [
        (Condition.DR_HYBRID, False),
        (Condition.D_LLM_GRAPH, True),
    ]


class _InFlightProbe:
    """Counts simultaneous requests; answers every schema with its empty payload."""

    provider = "fake"
    model = "probe"

    def __init__(self) -> None:
        self.in_flight = 0
        self.peak = 0
        self.calls = 0

    async def structured(
        self,
        *,
        task: str,
        instructions: str,
        schema: type[T],
        documents: tuple[UntrustedDocument, ...] = (),
        context: str = "",
        max_tokens: int = 16000,
    ) -> LLMResult[T]:
        self.calls += 1
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        await asyncio.sleep(0.01)
        self.in_flight -= 1
        usage = LLMUsage(provider="fake", model="probe", task=task, input_tokens=1, output_tokens=1)
        return LLMResult(data=schema(), usage=usage)


@pytest.mark.parametrize("concurrency", [1, 2])
def test_llm_concurrency_bounds_requests_not_cases(cases: list[EvalCase], concurrency: int) -> None:
    """A long runbook fans out into chunk calls; the bound must hold for requests, not cases."""
    base = _case(cases, Operator.DELETE_OWNER)
    long = [
        base.model_copy(update={"case_id": f"{base.case_id}-x{i}", "text": "\n\n".join([base.text] * 8)})
        for i in range(2)
    ]
    chunks = len(chunk_document(case_inputs(long[0]).document))
    assert chunks > concurrency  # the case alone would exceed the bound without a request gate
    probe = _InFlightProbe()
    results = run_eval(long, ["C_llm_direct", "D_llm_graph"], llm=probe, workers=1, concurrency=concurrency)
    assert probe.peak == concurrency
    assert probe.calls == len(long) * (1 + chunks)
    assert all(r.error is None for r in results.results)


def test_is_hit_credits_organisation_level_only_for_org_level_rules(cases: list[EvalCase]) -> None:
    mutated = [c for c in cases if c.operator is not None and c.style == "explicit" and c.position == "front"]
    for case in mutated:
        for rule in case.expected_rule_ids:
            org_level = _result(case, Condition.C_LLM_DIRECT, fired=frozenset({(None, rule)}))
            assert is_hit(case, org_level) is (rule in ORG_LEVEL_RULE_IDS), (case.operator, rule)
            assert not is_hit(case, org_level, strict=True)
            elsewhere = _result(case, Condition.C_LLM_DIRECT, fired=frozenset({("다른 서비스", rule)}))
            assert not is_hit(case, elsewhere)


@settings(max_examples=60, deadline=None)
@given(data=st.data())
def test_strict_hits_are_hits_and_org_credit_needs_an_org_level_rule(
    cases: list[EvalCase], data: st.DataObject
) -> None:
    case = data.draw(st.sampled_from([c for c in cases if c.operator is not None]))
    services = st.sampled_from([case.target_service, None, "다른 서비스"])
    rules = st.sampled_from(sorted({r for rs in EXPECTED_RULE_IDS.values() for r in rs}))
    fired = frozenset(data.draw(st.lists(st.tuples(services, rules), max_size=6)))
    result = _result(case, Condition.C_LLM_DIRECT, fired=fired)
    if is_hit(case, result, strict=True):
        assert is_hit(case, result)
    if is_hit(case, result) and not is_hit(case, result, strict=True):
        assert {r for s, r in fired if s is None} & set(case.expected_rule_ids) & ORG_LEVEL_RULE_IDS


def test_org_level_rule_list_covers_what_the_fixed_rules_report_org_wide(orgs: list[GoldOrg]) -> None:
    """Every expected rule the real rules report without a service is in ORG_LEVEL_RULE_IDS, so
    restricting organisation-level credit changes nothing for the rule-judged arms (B, O)."""
    expected = {r for rules in EXPECTED_RULE_IDS.values() for r in rules}
    sample = build_cases(orgs[:3], styles=("explicit",), positions=("front",), include_clean=False)
    for case in sample:
        oracle = run_offline_condition(case, Condition.O_ORACLE)
        assert {r for s, r in oracle.fired if s is None} & expected <= ORG_LEVEL_RULE_IDS, case.case_id
    shared = (Operator.DEADLINE_CONTRADICTION, Operator.DELETE_REPORTING_STEP)
    for case in (c for c in sample if c.operator in shared):
        rule_graph = run_offline_condition(case, Condition.B_RULE_GRAPH)
        assert {r for s, r in rule_graph.fired if s is None} & expected <= ORG_LEVEL_RULE_IDS, case.case_id


def test_windows_pool_limit_caps_default_and_explicit_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 128)
    # only the harness sees the fake platform; pytest itself keeps the real one
    monkeypatch.setattr(harness_module, "sys", SimpleNamespace(platform="win32"))
    assert harness_module.default_workers() == WINDOWS_MAX_WORKERS
    assert harness_module._pool_size(100, 500) == WINDOWS_MAX_WORKERS
    assert harness_module._pool_size(8, 3) == 3
    assert harness_module._pool_size(0, 3) == 1
    monkeypatch.setattr(harness_module, "sys", SimpleNamespace(platform="linux"))
    assert harness_module.default_workers() == 126
    assert harness_module._pool_size(100, 500) == 100
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert harness_module.default_workers() == 1


def test_reproducibility_rerun_happens_in_a_fresh_process(
    cases: list[EvalCase], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A patch in this process cannot reach the re-run: it really runs in a new interpreter."""
    results = run_eval(cases[:2], ["O_oracle"], workers=1)

    def sabotaged(case: EvalCase, condition: Condition) -> CaseResult:
        return _result(case, condition, error="ran in the parent process")

    monkeypatch.setattr(harness_module, "run_offline_condition", sabotaged)
    fresh = check_reproducibility(results, condition=Condition.O_ORACLE, fraction=1.0)
    assert (fresh.sample_size, fresh.identical, fresh.fresh_process) == (2, 2, True)
    same = check_reproducibility(results, condition=Condition.O_ORACLE, fraction=1.0, fresh_process=False)
    assert same.identical == 0  # the in-process re-run used the sabotaged runner


def test_direct_context_states_deadlines_that_are_not_whole_hours(cases: list[EvalCase]) -> None:
    inputs = case_inputs(_case(cases, None))
    odd = tuple(o.model_copy(update={"deadline_minutes": 90}) for o in inputs.policy.obligations)
    policy = inputs.policy.model_copy(update={"obligations": odd})
    context = direct_context(dataclasses.replace(inputs, policy=policy))
    assert "90분 이내" in context
    assert "1시간 이내" not in context
    assert "24시간 이내" in direct_context(inputs)


def test_graph_facts_take_whole_float_minutes_but_never_round() -> None:
    b = GraphBuilder()
    kim, lee = b.add_node(_node(NodeKind.PERSON, "김철수")), b.add_node(_node(NodeKind.PERSON, "이영희"))
    park = b.add_node(_node(NodeKind.PERSON, "박민수"))
    b.add_edge(_edge(EdgeKind.ESCALATES_TO, kim.id, lee.id, after_minutes=15.0))
    b.add_edge(_edge(EdgeKind.ESCALATES_TO, lee.id, park.id, after_minutes=15.5))
    facts = graph_facts(b.build())
    assert fact(EdgeKind.ESCALATES_TO, "김철수", "이영희", after_minutes=15) in facts
    assert fact(EdgeKind.ESCALATES_TO, "이영희", "박민수") in facts  # 15.5 is no gold value: left out


def test_injection_selftest_only_plants_lines_in_plain_text_runbooks(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    shutil.copytree(DMC_SHOP, bundle)
    html = "<html><body><h1>보조 안내</h1><p>참고용 문서입니다.</p></body></html>"
    (bundle / "runbooks" / "00-안내.html").write_text(html, encoding="utf-8")
    result = injection_selftest(bundle, n=1)  # the default skips the HTML runbook that sorts first
    assert result.document.endswith(".md")
    assert result.unauthorised_changes == 0
    with pytest.raises(ValueError, match="텍스트 런북"):
        injection_selftest(bundle, n=1, document="runbooks/00-안내.html")


def test_main_reports_usage_errors_without_a_traceback(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--conditions", "Z_magic", "--estimate-only"]) == 2
    assert "알 수 없는 평가 조건" in capsys.readouterr().err
    assert main(["--orgs", "0", "--estimate-only"]) == 2
    assert "--orgs" in capsys.readouterr().err
    assert main(["--workers", "0", "--generated-at", "2026-10-13"]) == 2
    assert "--workers" in capsys.readouterr().err


def _split(active: tuple[int, int]) -> SplitRate:
    return SplitRate(active=Rate.of(*active), held_out=Rate.of(0, 0))


def _with_b_styles(
    metrics: EvalMetrics, rates: dict[str, tuple[int, int]], recalls: dict[str, float]
) -> EvalMetrics:
    b = Condition.B_RULE_GRAPH.value
    by_style = {s: Prf.of(round(r * 1000), 0, 1000 - round(r * 1000)) for s, r in recalls.items()}
    fields = metrics.field_f1[b].model_copy(update={"by_style": by_style})
    return metrics.model_copy(
        update={
            "by_style": {**metrics.by_style, b: {s: _split(v) for s, v in rates.items()}},
            "field_f1": {**metrics.field_f1, b: fields},
        }
    )


def test_report_interpretation_only_claims_what_the_numbers_show(cases: list[EvalCase]) -> None:
    sample = [_case(cases, None), _case(cases, Operator.DELETE_ROLLBACK)]
    results = run_eval(sample, ["B_rule_graph"], workers=1)
    metrics = compute_metrics(
        results,
        generated_at="2026-10-13",
        reproducibility=check_reproducibility(results, fraction=0.5, fresh_process=False),
    )
    text = render_report(metrics)
    assert "한 프로세스에서 차례로" in text  # workers=1 is not described as a parallel run
    assert "같은 프로세스에서" in text  # an in-process re-check is not sold as cross-process
    assert "탐지율(활성 1종)" in text  # operator counts come from the data
    recalls = {"table": 0.75, "explicit": 0.82, "colloquial": 0.66}
    close = {"table": (235, 320), "explicit": (220, 320), "colloquial": (210, 320)}
    overlapping = render_report(_with_b_styles(metrics, close, recalls))
    assert "탐지율만으로는 문체 차이를 단정할 수 없습니다" in overlapping
    assert "필드 재현율은 구어체가 가장 낮습니다(구어체 66.0%, 명시형 목록 82.0%)" in overlapping
    assert "신뢰구간이 겹치지 않습니다" not in overlapping
    apart = {"table": (300, 320), "explicit": (290, 320), "colloquial": (150, 320)}
    separated = render_report(_with_b_styles(metrics, apart, recalls))
    assert "구어체에서 가장 낮고, 가장 높은 문체와 95% 신뢰구간이 겹치지 않습니다" in separated


def test_an_empty_run_still_reports_without_numbers(tmp_path: Path) -> None:
    results = run_eval([], list(OFFLINE_CONDITIONS), workers=4)
    assert results.results == ()
    assert results.workers == 1
    metrics = write_report(results, tmp_path, generated_at="2026-10-13")
    assert metrics.n_cases == 0
    assert metrics.conditions["B_rule_graph"].detection_active.rate is None
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "| B 규칙 파서 + 그래프 | — | — | — |" in report
    assert (tmp_path / "results.jsonl").read_text(encoding="utf-8") == ""


# --------------------------------------------------------------------------- review 2: data, pairing


def test_case_fields_are_parsed_as_content_never_opened_as_files(
    cases: list[EvalCase], tmp_path: Path
) -> None:
    """A case field that names a local file must not make the harness read it: in the LLM arms
    the file's content would go into the prompt of a third-party model (P5: inputs are data)."""
    base = _case(cases, None)
    roster = tmp_path / "roster.csv"
    roster.write_text(
        "이름,상태,직급,팀,역할,별칭,연락수단,퇴사일\n비밀직원,재직,대리,인사팀,,,전화,\n", encoding="utf-8"
    )
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(base.catalog_yaml, encoding="utf-8")
    policy = tmp_path / "policy.yaml"
    policy.write_text(base.policy_yaml, encoding="utf-8")
    # the path strings are parsed as (malformed) content: a header-only CSV, a bare YAML scalar
    leaky = base.model_copy(update={"roster_csv": roster.as_posix()})
    assert case_inputs(leaky).roster == ()
    with pytest.raises(ValueError, match="카탈로그 형식을 알 수 없습니다"):
        case_inputs(base.model_copy(update={"catalog_yaml": catalog.as_posix()}))
    with pytest.raises(ValueError, match="정책 파일은"):
        case_inputs(base.model_copy(update={"policy_yaml": policy.as_posix()}))
    fake = FakeLLM(lambda *_: DirectDefectPayload())
    direct = run_sync(arun_llm_direct(leaky, fake))
    assert direct.error is None
    assert "비밀직원" not in fake.calls[0].context  # the file never reached the model
    assert "(제공되지 않음)" in fake.calls[0].context
    keyword = run_offline_condition(
        base.model_copy(update={"catalog_yaml": catalog.as_posix()}), Condition.A_KEYWORD
    )
    assert keyword.error is not None
    assert keyword.error.startswith("ValueError")


def test_case_documents_carry_no_answer_key(cases: list[EvalCase]) -> None:
    """The case id names the injected operator; it must not reach the system under test."""
    for case in cases[::5]:
        doc = case_inputs(case).document
        assert doc.path == CASE_RUNBOOK_NAME
        assert case.case_id not in f"{doc.path} {doc.title} {doc.doc_id} {doc.text}"
        if case.operator is not None:
            assert case.operator.value not in f"{doc.path} {doc.title}"


def test_f1_of_a_total_failure_is_zero_not_unmeasured() -> None:
    wrong = Prf.of(0, 3, 4)
    assert (wrong.precision, wrong.recall, wrong.f1) == (0.0, 0.0, 0.0)
    nothing = Prf.of(0, 0, 4)  # nothing extracted while four facts were expected
    assert (nothing.precision, nothing.recall, nothing.f1) == (None, 0.0, 0.0)
    assert Prf.of(0, 0, 0).f1 is None


@settings(max_examples=200, deadline=None)
@given(tp=st.integers(0, 50), fp=st.integers(0, 50), fn=st.integers(0, 50))
def test_f1_is_the_harmonic_mean_wherever_both_are_defined(tp: int, fp: int, fn: int) -> None:
    prf = Prf.of(tp, fp, fn)
    if tp + fp + fn == 0:
        assert prf.f1 is None
        return
    assert prf.f1 is not None
    assert 0.0 <= prf.f1 <= 1.0
    if tp:
        p, r = tp / (tp + fp), tp / (tp + fn)
        assert prf.f1 == pytest.approx(2 * p * r / (p + r), abs=1e-4)
    else:
        assert prf.f1 == 0.0


def test_hallucination_counts_partial_graphs_of_errored_runs(cases: list[EvalCase]) -> None:
    owner = _case(cases, Operator.DELETE_OWNER)
    other = _case(cases, Operator.DELETE_OWNER, "table")
    d = Condition.D_LLM_GRAPH
    invented = owner.deleted_facts[0]
    partial = _result(owner, d, error="LLM 응답이 거부된 청크 1/2개", facts=frozenset({invented}))
    crashed = _result(other, d, error="boom")
    results = EvalResults(cases=(owner, other), conditions=(d,), results=(partial, crashed))
    m = compute_metrics(results, generated_at="2026-10-13")
    rate = m.hallucination[d.value].rate
    assert (rate.hits, rate.n) == (1, len({f.key for f in owner.deleted_facts}))
    assert m.conditions[d.value].detection_active.hits == 0  # an errored case is still a miss


def test_paired_detection_discounts_findings_the_clean_twin_also_raises(cases: list[EvalCase]) -> None:
    clean = _case(cases, None)  # the twin of every explicit/front case below
    approver = _case(cases, Operator.DELETE_APPROVER)
    rollback = _case(cases, Operator.DELETE_ROLLBACK)
    b = Condition.B_RULE_GRAPH
    noisy = (approver.target_service, RULE_APPROVER)  # raised even when nothing is broken
    rows = (
        _result(clean, b, fired=frozenset({noisy}), severe=frozenset({noisy})),
        _result(approver, b, fired=frozenset({noisy})),
        _result(rollback, b, fired=frozenset({(rollback.target_service, RULE_ROLLBACK)})),
    )
    chosen = (clean, approver, rollback)
    m = compute_metrics(EvalResults(cases=chosen, conditions=(b,), results=rows), generated_at="2026-10-13")
    s = m.conditions[b.value]
    assert (s.detection_active.hits, s.detection_active.n) == (2, 2)
    assert (s.detection_paired_active.hits, s.detection_paired_active.n) == (1, 2)
    paired = {op: r.paired for op, r in m.by_operator[b.value].items()}
    noisy_rate = paired[Operator.DELETE_APPROVER.value]
    clean_rate = paired[Operator.DELETE_ROLLBACK.value]
    assert noisy_rate is not None
    assert noisy_rate.rate == 0.0
    assert clean_rate is not None
    assert clean_rate.rate == 1.0
    text = render_report(m)
    assert "대조군 보정 탐지율(활성 · 봉인)" in text
    assert "활성 탐지 2건 중 1건(100.0% → 50.0%)은" in text
    assert "봉인 탐지" not in text  # nothing held-out was discounted
    # an errored twin vouches for nothing: no paired credit at all
    errored = (dataclasses.replace(rows[0], error="boom"), *rows[1:])
    m_err = compute_metrics(EvalResults(cases=chosen, conditions=(b,), results=errored), generated_at="x")
    assert m_err.conditions[b.value].detection_paired_active.hits == 0
    # without clean controls there is nothing to pair with: unmeasured, never 0% or 100%
    alone = compute_metrics(
        EvalResults(cases=chosen[1:], conditions=(b,), results=rows[1:]), generated_at="x"
    )
    assert alone.conditions[b.value].detection_paired_active.rate is None
    assert all(r.paired is None for r in alone.by_operator[b.value].values())
    assert "대조군 보정 탐지율에서는 빠집니다" not in render_report(alone)


@settings(max_examples=80, deadline=None)
@given(data=st.data())
def test_paired_hits_are_hits_that_the_twin_does_not_explain(
    cases: list[EvalCase], data: st.DataObject
) -> None:
    case = data.draw(st.sampled_from([c for c in cases if c.operator is not None]))
    twin_case = next(c for c in cases if c.is_clean and (c.style, c.position) == (case.style, case.position))
    services = st.sampled_from([case.target_service, None, "다른 서비스"])
    rules = st.sampled_from(sorted({r for rs in EXPECTED_RULE_IDS.values() for r in rs}))
    keys = st.frozensets(st.tuples(services, rules), max_size=6)
    result = _result(case, Condition.B_RULE_GRAPH, fired=data.draw(keys))
    twin = _result(twin_case, Condition.B_RULE_GRAPH, fired=data.draw(keys))
    if is_paired_hit(case, result, twin):
        assert is_hit(case, result)
        assert credited_keys(case, result.fired) - twin.fired
    silent = dataclasses.replace(twin, fired=frozenset())
    assert is_paired_hit(case, result, silent) is is_hit(case, result)
    echo = dataclasses.replace(twin, fired=result.fired)
    assert not is_paired_hit(case, result, echo)  # a twin raising the same keys explains every hit
    assert not is_paired_hit(case, result, dataclasses.replace(silent, error="x"))


def test_injection_selftest_reads_bundles_with_decomposed_hangul_names(tmp_path: Path) -> None:
    """Bundles zipped on macOS keep NFD file names; ``Document.path`` is NFC. The self-test must
    plant into the loaded text instead of reopening the file by its display name."""
    bundle = tmp_path / "bundle"
    shutil.copytree(DMC_SHOP, bundle)
    first = sorted((bundle / "runbooks").glob("*.md"))[0]
    data = first.read_bytes()
    first.unlink()
    (first.parent / unicodedata.normalize("NFD", first.name)).write_bytes(data)
    result = injection_selftest(bundle, n=2)
    assert (result.n, result.unauthorised_changes, result.quarantined) == (2, 0, 2)


@pytest.mark.parametrize(
    "document", ["../roster.csv", "runbooks/../../roster.csv", "runbooks/../policy.yaml", "/etc/passwd"]
)
def test_injection_selftest_only_opens_runbooks_of_the_bundle(document: str) -> None:
    with pytest.raises(ValueError, match="묶음에 없는"):
        injection_selftest(DMC_SHOP, n=1, document=document)


def test_run_sync_refuses_a_running_loop_and_closes_the_coroutine() -> None:
    """Calling the sync API from async code (a notebook, a web handler) fails clearly, and the
    coroutine handed in is closed instead of being reported as "never awaited"."""

    async def leaf() -> None:
        return None

    async def caller() -> None:
        with pytest.raises(RuntimeError, match="이벤트 루프"):
            run_sync(leaf())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_sync(caller())
        gc.collect()  # finalise the refused coroutine while warnings are recorded
    assert not [w for w in caught if "never awaited" in str(w.message)]


class _InlinePool:
    """Stands in for the spawned pool: records the environment a child would inherit."""

    seen: ClassVar[list[str | None]] = []

    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def __enter__(self) -> _InlinePool:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def submit(self, fn: Callable[..., Any], *args: Any) -> Future[Any]:
        self.seen.append(os.environ.get("PYTHONHASHSEED"))
        future: Future[Any] = Future()
        future.set_result(fn(*args))
        return future


@pytest.mark.parametrize("pinned", ["0", None])
def test_reproducibility_rerun_never_inherits_a_pinned_hash_seed(
    cases: list[EvalCase], monkeypatch: pytest.MonkeyPatch, pinned: str | None
) -> None:
    if pinned is None:
        monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    else:
        monkeypatch.setenv("PYTHONHASHSEED", pinned)
    results = run_eval(cases[:2], ["O_oracle"], workers=1)
    _InlinePool.seen = []
    monkeypatch.setattr(harness_module, "ProcessPoolExecutor", _InlinePool)
    report = check_reproducibility(results, condition=Condition.O_ORACLE, fraction=1.0)
    assert _InlinePool.seen == ["random"]  # the child draws its own seed
    assert os.environ.get("PYTHONHASHSEED") == pinned  # and the parent's setting is restored
    assert (report.sample_size, report.identical) == (2, 2)


def test_run_evaluation_refuses_zero_concurrency_instead_of_replacing_it(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="concurrency"):
        run_evaluation(
            tmp_path,
            generated_at="2026-10-13",
            n_orgs=1,
            workers=1,
            benchmarks=False,
            concurrency=0,
            model="m",
        )
