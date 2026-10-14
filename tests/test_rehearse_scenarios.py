"""Tests for rehearsal scenario generation (jettae.rehearse.scenarios)."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, TypeVar

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import BaseModel

from jettae.check.context import CheckContext
from jettae.domain import (
    Attr,
    Edge,
    EdgeKind,
    Element,
    Extractor,
    GraphBuilder,
    IncidentType,
    Node,
    NodeKind,
    OpsGraph,
    Outcome,
    PerturbationKind,
    Policy,
    Provenance,
    Scenario,
    ScenarioOrigin,
    SimEventKind,
    SourceKind,
    Verification,
    edge_id,
    node_id,
)
from jettae.extract.assemble import assemble
from jettae.extract.rules import RuleExtractor
from jettae.ingest import load_bundle_dir
from jettae.llm.base import (
    LLMClient,
    LLMOutputError,
    LLMResult,
    LLMUsage,
    NullLLM,
    UntrustedDocument,
)
from jettae.rehearse import (
    LlmScenarioPayload,
    llm_scenarios,
    replay,
    replay_all,
    scenarios_from_yaml,
    template_scenarios,
)
from jettae.rehearse.simulator import first_responders

T = TypeVar("T", bound=BaseModel)
PROV = (Provenance(source=SourceKind.CATALOG, extractor=Extractor.IMPORT, ref="test"),)

PAY = node_id(NodeKind.SERVICE, "결제")
ORDER = node_id(NodeKind.SERVICE, "주문")
BANNER = node_id(NodeKind.SERVICE, "배너")
ALARM = node_id(NodeKind.TRIGGER, "결제 알람")
KIM = node_id(NodeKind.PERSON, "김대리")
PARK = node_id(NodeKind.PERSON, "박팀장")
LEE = node_id(NodeKind.PERSON, "이주임")
LEAD = node_id(NodeKind.ROLE, "팀장")
SLACK = node_id(NodeKind.CHANNEL, "슬랙")
PHONE = node_id(NodeKind.CHANNEL, "전화")
ROLLBACK = node_id(NodeKind.ACTION, "결제 롤백")
DB = node_id(NodeKind.SYSTEM, "결제 DB")
IDC = node_id(NodeKind.LOCATION, "IDC-A")


class Graph:
    def __init__(self) -> None:
        self.b = GraphBuilder()

    def node(self, kind: NodeKind, name: str, **attrs: Any) -> str:
        nid = node_id(kind, name)
        values = {k: Attr(value=v, source=SourceKind.CATALOG) for k, v in attrs.items()}
        self.b.add_node(Node(id=nid, kind=kind, name=name, attrs=values, prov=PROV))
        return nid

    def edge(self, kind: EdgeKind, src: str, dst: str, **attrs: Any) -> str:
        eid = edge_id(kind, src, dst)
        values = {k: Attr(value=v, source=SourceKind.CATALOG) for k, v in attrs.items()}
        self.b.add_edge(Edge(id=eid, kind=kind, src=src, dst=dst, attrs=values, prov=PROV))
        return eid

    def build(self) -> OpsGraph:
        return self.b.build()


def rich(*, reverse: bool = False) -> OpsGraph:
    """결제·주문(tier 1, 김대리 공유) + 배너(tier 3). ``reverse`` inserts in the opposite order."""
    nodes: list[tuple[NodeKind, str, dict[str, Any]]] = [
        (NodeKind.SERVICE, "결제", {"tier": 1}),
        (NodeKind.SERVICE, "주문", {"tier": 1}),
        (NodeKind.SERVICE, "배너", {"tier": 3}),
        (NodeKind.TRIGGER, "결제 알람", {"delay_minutes": 3}),
        (NodeKind.PERSON, "김대리", {"status": "active"}),
        (NodeKind.PERSON, "박팀장", {"status": "active"}),
        (NodeKind.PERSON, "이주임", {"status": "active"}),
        (NodeKind.ROLE, "팀장", {}),
        (NodeKind.CHANNEL, "슬랙", {"medium": "slack"}),
        (NodeKind.CHANNEL, "전화", {"medium": "phone"}),
        (NodeKind.ACTION, "결제 롤백", {"action_type": "rollback", "duration_minutes": 10}),
        (NodeKind.SYSTEM, "결제 DB", {}),
        (NodeKind.LOCATION, "IDC-A", {}),
    ]
    edges: list[tuple[EdgeKind, str, str, dict[str, Any]]] = [
        (EdgeKind.DETECTED_BY, PAY, ALARM, {}),
        (EdgeKind.NOTIFIES, ALARM, KIM, {"channel": SLACK}),
        (EdgeKind.OWNED_BY, PAY, KIM, {}),
        (EdgeKind.OWNED_BY, ORDER, KIM, {}),
        (EdgeKind.OWNED_BY, BANNER, LEE, {}),
        (EdgeKind.REACHABLE_VIA, KIM, SLACK, {}),
        (EdgeKind.REACHABLE_VIA, KIM, PHONE, {}),
        (EdgeKind.REACHABLE_VIA, PARK, PHONE, {}),
        (EdgeKind.REACHABLE_VIA, LEE, SLACK, {}),
        (EdgeKind.HOLDS_ROLE, PARK, LEAD, {}),
        (EdgeKind.ESCALATES_TO, KIM, LEAD, {"after_minutes": 5}),
        (EdgeKind.MITIGATED_BY, PAY, ROLLBACK, {}),
        (EdgeKind.APPROVED_BY, ROLLBACK, LEAD, {}),
        (EdgeKind.DEPENDS_ON, PAY, DB, {}),
        (EdgeKind.RUNS_IN, DB, IDC, {}),
    ]
    g = Graph()
    for kind, name, attrs in reversed(nodes) if reverse else nodes:
        g.node(kind, name, **attrs)
    for ekind, src, dst, attrs in reversed(edges) if reverse else edges:
        g.edge(ekind, src, dst, **attrs)
    return g.build()


# --------------------------------------------------------------------------- templates


def test_template_ids_titles_and_targets() -> None:
    scenarios = template_scenarios(CheckContext(rich()))
    by_id = {s.id: s for s in scenarios}
    assert [s.id for s in scenarios] == [
        f"tpl:{PAY}:baseline",
        f"tpl:{PAY}:owner_absent",
        f"tpl:{PAY}:channel_down",
        f"tpl:{PAY}:approver_absent",
        f"tpl:{PAY}:location_down",
        f"tpl:{PAY}:concurrent",
        f"tpl:{PAY}:breach_report",
        f"tpl:{ORDER}:baseline",
        f"tpl:{ORDER}:owner_absent",
        f"tpl:{ORDER}:channel_down",
        f"tpl:{ORDER}:concurrent",
        f"tpl:{ORDER}:breach_report",
    ]
    assert all(s.origin is ScenarioOrigin.TEMPLATE for s in scenarios)

    def target(key: str) -> tuple[PerturbationKind, str]:
        (p,) = by_id[key].perturbations
        assert (p.start_minute, p.end_minute) == (0.0, None)
        return p.kind, p.target_id

    assert by_id[f"tpl:{PAY}:baseline"].perturbations == ()
    assert by_id[f"tpl:{PAY}:baseline"].title == "결제 · 기준"
    assert target(f"tpl:{PAY}:owner_absent") == (PerturbationKind.PERSON_UNAVAILABLE, KIM)
    assert by_id[f"tpl:{PAY}:owner_absent"].title == "결제 · 1차 담당 부재(김대리)"
    assert target(f"tpl:{PAY}:channel_down") == (PerturbationKind.CHANNEL_DOWN, SLACK)
    assert target(f"tpl:{PAY}:approver_absent") == (PerturbationKind.PERSON_UNAVAILABLE, PARK)
    assert target(f"tpl:{PAY}:location_down") == (PerturbationKind.LOCATION_DOWN, IDC)
    assert by_id[f"tpl:{PAY}:location_down"].title == "결제 · 센터 장애(IDC-A)"
    assert target(f"tpl:{PAY}:concurrent") == (PerturbationKind.CONCURRENT_INCIDENT, ORDER)
    assert target(f"tpl:{ORDER}:concurrent") == (PerturbationKind.CONCURRENT_INCIDENT, PAY)
    breach = by_id[f"tpl:{PAY}:breach_report"]
    assert breach.incident_type is IncidentType.BREACH
    assert breach.perturbations == ()
    assert "24시간 이내" in breach.description
    # tie between 슬랙 and 전화 for 주문 (no NOTIFIES hint) breaks by channel id
    assert target(f"tpl:{ORDER}:channel_down") == (PerturbationKind.CHANNEL_DOWN, SLACK)
    assert all(s.deadline_minutes is None for s in scenarios)


def test_template_ids_are_stable_across_runs_and_insertion_order() -> None:
    first = template_scenarios(CheckContext(rich()))
    again = template_scenarios(CheckContext(rich()))
    reordered = template_scenarios(CheckContext(rich(reverse=True)))
    assert first == again == reordered
    assert [s.fingerprint() for s in first] == [s.fingerprint() for s in reordered]


def test_service_ids_filter_includes_non_critical_and_ignores_unknown() -> None:
    ctx = CheckContext(rich())
    scenarios = template_scenarios(ctx, service_ids=[BANNER, "service:없음", KIM])
    assert [s.id for s in scenarios] == [
        f"tpl:{BANNER}:baseline",
        f"tpl:{BANNER}:owner_absent",
        f"tpl:{BANNER}:channel_down",
    ]
    assert template_scenarios(ctx, service_ids=[]) == []


def test_templates_skip_what_the_graph_cannot_support() -> None:
    g = Graph()
    g.node(NodeKind.SERVICE, "결제", tier=1)
    g.node(NodeKind.PERSON, "김대리", status="departed")
    g.edge(EdgeKind.OWNED_BY, PAY, KIM)
    g.node(NodeKind.ACTION, "결제 롤백", action_type="rollback")
    g.node(NodeKind.ROLE, "팀장")
    g.node(NodeKind.PERSON, "박팀장", status="departed")
    g.edge(EdgeKind.MITIGATED_BY, PAY, ROLLBACK)
    g.edge(EdgeKind.APPROVED_BY, ROLLBACK, LEAD)
    g.edge(EdgeKind.HOLDS_ROLE, PARK, LEAD)  # an approver exists, but nobody can stand in for them
    ctx = CheckContext(g.build(), policy=Policy(obligations=()))
    assert [s.id for s in template_scenarios(ctx)] == [f"tpl:{PAY}:baseline"]


def test_breach_template_follows_policy_scope() -> None:
    graph = rich()
    out_of_scope = CheckContext(graph, policy=Policy(regulated_services=(ORDER,)))
    ids = {s.id for s in template_scenarios(out_of_scope)}
    assert f"tpl:{PAY}:breach_report" not in ids
    assert f"tpl:{ORDER}:breach_report" in ids
    not_applicable = Policy(
        obligations=tuple(r.model_copy(update={"applies": False}) for r in Policy().obligations)
    )
    assert not any(
        s.id.endswith(":breach_report")
        for s in template_scenarios(CheckContext(graph, policy=not_applicable))
    )


def test_owner_absent_targets_first_paged_person_not_just_owner() -> None:
    g = Graph()
    g.node(NodeKind.SERVICE, "결제", tier=1)
    g.node(NodeKind.TRIGGER, "결제 알람")
    g.node(NodeKind.PERSON, "김대리", status="active")
    g.node(NodeKind.PERSON, "박팀장", status="active")
    g.node(NodeKind.CHANNEL, "슬랙")
    g.edge(EdgeKind.DETECTED_BY, PAY, ALARM)
    g.edge(EdgeKind.NOTIFIES, ALARM, PARK)
    g.edge(EdgeKind.OWNED_BY, PAY, KIM)
    g.edge(EdgeKind.REACHABLE_VIA, KIM, SLACK)
    g.edge(EdgeKind.REACHABLE_VIA, PARK, SLACK)
    (absent,) = (s for s in template_scenarios(CheckContext(g.build())) if s.id.endswith(":owner_absent"))
    assert absent.perturbations[0].target_id == PARK


def test_concurrent_requires_partner_whose_first_responder_is_shared() -> None:
    g = Graph()
    g.node(NodeKind.SERVICE, "결제", tier=1)
    g.node(NodeKind.SERVICE, "주문", tier=1)
    g.node(NodeKind.TEAM, "플랫폼팀")
    team = node_id(NodeKind.TEAM, "플랫폼팀")
    g.node(NodeKind.PERSON, "김대리", status="active")
    g.node(NodeKind.PERSON, "이주임", status="active")
    g.node(NodeKind.CHANNEL, "슬랙")
    g.edge(EdgeKind.REACHABLE_VIA, KIM, SLACK)
    g.edge(EdgeKind.REACHABLE_VIA, LEE, SLACK)
    g.edge(EdgeKind.MEMBER_OF, KIM, team)
    g.edge(EdgeKind.MEMBER_OF, LEE, team)
    g.edge(EdgeKind.OWNED_BY, PAY, team)  # 결제 responders: 김대리, 이주임
    g.edge(EdgeKind.OWNED_BY, ORDER, LEE)  # 주문's first responder is shared → 결제 is stressed
    ids = {s.id for s in template_scenarios(CheckContext(g.build()))}
    assert f"tpl:{PAY}:concurrent" in ids
    # 주문 is stressed only if 결제's *first* responder (김대리) were also 주문's — it is not
    assert f"tpl:{ORDER}:concurrent" not in ids


def test_templates_replay_into_meaningful_verdicts() -> None:
    graph = rich()
    scenarios = template_scenarios(CheckContext(graph), service_ids=[PAY])
    results = {r.scenario_id.rsplit(":", 1)[1]: r for r in replay_all(graph, scenarios)}
    verdicts = {key: (r.outcome, r.failed_element) for key, r in results.items()}
    assert verdicts == {
        "baseline": (Outcome.PASS, None),
        "owner_absent": (Outcome.PASS, None),  # escalation to 팀장
        "channel_down": (Outcome.PASS, None),  # 김대리 also has 전화
        "approver_absent": (Outcome.FAIL, Element.APPROVER),  # 팀장 has no deputy
        "location_down": (Outcome.PASS, None),  # no channel depends on IDC-A
        "concurrent": (Outcome.PASS, None),  # 김대리 busy with 주문 → 팀장
        "breach_report": (Outcome.FAIL, None),  # no KISA reporting step in the runbook
    }
    assert "정보통신망법" in results["breach_report"].reason


# --------------------------------------------------------------------------- LLM scenarios


class FakeLLM:
    """Inline LLMClient double: returns a fixed payload (or raises) and records calls."""

    provider = "fake"
    model = "fake-1"

    def __init__(self, payload: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        self.payload = payload
        self.error = error
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append(
            {
                "task": task,
                "instructions": instructions,
                "schema": schema,
                "documents": documents,
                "context": context,
            }
        )
        if self.error is not None:
            raise self.error
        return LLMResult(
            data=schema.model_validate(self.payload),
            usage=LLMUsage(provider=self.provider, model=self.model, task=task),
        )


def item(service_id: str = PAY, *perturbations: dict[str, Any], **kw: Any) -> dict[str, Any]:
    return {
        "title": kw.get("title", "추석 연휴 새벽 3시, 결제 담당 휴가 중, 슬랙 장애"),
        "description": kw.get("description", "연휴 새벽에 결제 오류율이 치솟는다."),
        "service_id": service_id,
        "incident_type": kw.get("incident_type", "outage"),
        "perturbations": list(perturbations),
    }


def pert(kind: str, target: str, start: float = 0.0, end: float | None = None) -> dict[str, Any]:
    return {"kind": kind, "target_id": target, "start_minute": start, "end_minute": end}


def run(llm: LLMClient, ctx: CheckContext, n: int = 3) -> list[Scenario]:
    """Drive the coroutine without an event loop.

    The fakes never suspend, so a single ``send`` completes it. This keeps the tests free of
    event-loop/socket setup (restricted in sandboxed Windows runners) and shows that
    ``llm_scenarios`` performs no I/O of its own besides the LLM call.
    """
    coro = llm_scenarios(llm, ctx, n=n)
    try:
        coro.send(None)
    except StopIteration as finished:
        result: list[Scenario] = finished.value
        return result
    coro.close()
    pytest.fail("llm_scenarios suspended although the fake LLM never awaits")


def test_llm_scenarios_keep_only_valid_ones() -> None:
    ctx = CheckContext(rich())
    payload = {
        "scenarios": [
            item(PAY, pert("person_unavailable", KIM), pert("channel_down", SLACK, 0, 120)),  # valid
            item("service:없음"),  # unknown service
            item(PAY, pert("person_unavailable", SLACK)),  # kind mismatch
            item(PAY, pert("channel_down", "channel:카카오톡")),  # unknown target
            item(PAY, pert("concurrent_incident", PAY)),  # concurrent with itself
            item(PAY, pert("system_down", DB, 30, 10)),  # end before start
            item(PAY, pert("location_down", IDC, -5)),  # negative start
            item(KIM),  # not a service
            item(ORDER, pert("concurrent_incident", PAY, 10), incident_type="breach"),  # valid
        ]
    }
    llm = FakeLLM(payload)
    assert isinstance(llm, LLMClient)
    scenarios = run(llm, ctx, n=5)
    assert len(scenarios) == 2
    first, second = scenarios
    assert first.service_id == PAY
    assert {(p.kind, p.target_id) for p in first.perturbations} == {
        (PerturbationKind.PERSON_UNAVAILABLE, KIM),
        (PerturbationKind.CHANNEL_DOWN, SLACK),
    }
    assert second.incident_type is IncidentType.BREACH
    for s in scenarios:
        assert s.origin is ScenarioOrigin.LLM
        assert re.fullmatch(r"llm:[0-9a-f]{16}", s.id)
        assert s.deadline_minutes is None
    assert first.title == "추석 연휴 새벽 3시, 결제 담당 휴가 중, 슬랙 장애"

    (call,) = llm.calls
    assert call["task"] == "scenarios"
    assert call["schema"] is LlmScenarioPayload
    assert call["documents"] == ()
    assert "추석 연휴 새벽 3시, 결제 담당 휴가 중, 슬랙 장애" in call["instructions"]
    context = call["context"]
    for line in (
        f"- 서비스(service): {PAY}, {BANNER}, {ORDER}",
        f"- 사람(person): {KIM}, {PARK}, {LEE}",
        f"- 채널(channel): {SLACK}, {PHONE}",
        f"- 시스템(system): {DB}",
        f"- 위치(location): {IDC}",
        f"- {PAY}: {KIM}",
    ):
        assert line in context


def test_llm_scenario_ids_are_content_addressed_and_order_insensitive() -> None:
    ctx = CheckContext(rich())
    a = item(PAY, pert("person_unavailable", KIM), pert("channel_down", SLACK), title="A")
    b = item(PAY, pert("channel_down", SLACK), pert("person_unavailable", KIM), title="다른 제목")
    first = run(FakeLLM({"scenarios": [a]}), ctx)
    second = run(FakeLLM({"scenarios": [b]}), ctx)
    assert first[0].id == second[0].id  # same perturbations → same id, titles ignored
    deduped = run(FakeLLM({"scenarios": [a, b]}), ctx)
    assert [s.title for s in deduped] == ["A"]
    later = run(FakeLLM({"scenarios": [item(PAY, pert("person_unavailable", KIM, 10))]}), ctx)
    assert later[0].id != first[0].id


def test_llm_scenarios_cap_and_clean_text() -> None:
    ctx = CheckContext(rich())
    noisy = item(
        PAY, pert("person_unavailable", PARK), title="  줄\n바꿈\t제목 " + "가" * 200, description="a\x00b"
    )
    blank = item(ORDER, pert("person_unavailable", KIM), title=" \n ")
    scenarios = run(FakeLLM({"scenarios": [noisy, blank, item(PAY)]}), ctx, n=2)
    assert len(scenarios) == 2
    assert scenarios[0].title.startswith("줄 바꿈 제목 가")
    assert len(scenarios[0].title) == 80
    assert scenarios[0].description == "a b"
    assert scenarios[1].title == "주문 · AI 제안 시나리오"


def test_llm_disabled_returns_empty_and_zero_n_skips_the_call() -> None:
    ctx = CheckContext(rich())
    assert run(NullLLM(), ctx) == []
    llm = FakeLLM({"scenarios": [item(PAY)]})
    assert run(llm, ctx, n=0) == []
    assert llm.calls == []


def test_other_llm_errors_propagate() -> None:
    with pytest.raises(LLMOutputError):
        run(FakeLLM(error=LLMOutputError("bad json")), CheckContext(rich()))


_TARGETS = [PAY, ORDER, BANNER, KIM, PARK, SLACK, PHONE, DB, IDC, LEAD, "x:없음"]


@settings(max_examples=60, deadline=None)
@given(
    payload=st.lists(
        st.fixed_dictionaries(
            {
                "title": st.text(max_size=100),
                "description": st.text(max_size=500),
                "service_id": st.sampled_from(_TARGETS),
                "incident_type": st.sampled_from([t.value for t in IncidentType]),
                "perturbations": st.lists(
                    st.fixed_dictionaries(
                        {
                            "kind": st.sampled_from([k.value for k in PerturbationKind]),
                            "target_id": st.sampled_from(_TARGETS),
                            "start_minute": st.floats(-10, 100, allow_nan=False),
                            "end_minute": st.one_of(st.none(), st.floats(-10, 200)),
                        }
                    ),
                    max_size=3,
                ),
            }
        ),
        max_size=6,
    ),
    n=st.integers(1, 4),
)
def test_llm_output_is_always_validated(payload: list[dict[str, Any]], n: int) -> None:
    graph = rich()
    ctx = CheckContext(graph)
    expected_kind = {
        PerturbationKind.PERSON_UNAVAILABLE: NodeKind.PERSON,
        PerturbationKind.CHANNEL_DOWN: NodeKind.CHANNEL,
        PerturbationKind.SYSTEM_DOWN: NodeKind.SYSTEM,
        PerturbationKind.LOCATION_DOWN: NodeKind.LOCATION,
        PerturbationKind.CONCURRENT_INCIDENT: NodeKind.SERVICE,
    }
    scenarios = run(FakeLLM({"scenarios": payload}), ctx, n=n)
    assert scenarios == run(FakeLLM({"scenarios": payload}), ctx, n=n)
    assert len(scenarios) <= n
    assert len({s.id for s in scenarios}) == len(scenarios)
    for s in scenarios:
        assert graph.nodes[s.service_id].kind is NodeKind.SERVICE
        assert s.title
        assert len(s.title) <= 80
        assert len(s.description) <= 400
        for p in s.perturbations:
            assert graph.nodes[p.target_id].kind is expected_kind[p.kind]
            assert p.start_minute >= 0
            assert p.end_minute is None or p.end_minute > p.start_minute


# --------------------------------------------------------------------------- reviewer: regressions


def _hold_edge(g: Graph, eid: str) -> None:
    g.b.edges[eid] = g.b.edges[eid].model_copy(update={"verification": Verification.HOLD})


def _templates_by_key(graph: OpsGraph, **kw: Any) -> dict[str, Scenario]:
    return {s.id.rsplit(":", 1)[1]: s for s in template_scenarios(CheckContext(graph, **kw))}


def _target(s: Scenario) -> str:
    (p,) = s.perturbations
    return p.target_id


def test_owner_absent_targets_whoever_is_paged_first_in_time() -> None:
    """Trigger-id order put 박팀장 (30-minute alarm) first; the replay pages 김대리 at minute 3."""
    g = Graph()
    g.node(NodeKind.SERVICE, "결제", tier=1)
    g.node(NodeKind.TRIGGER, "가 느린 알람", delay_minutes=30)
    slow = node_id(NodeKind.TRIGGER, "가 느린 알람")
    g.node(NodeKind.TRIGGER, "결제 알람", delay_minutes=3)
    assert slow < ALARM
    for name in ("김대리", "박팀장"):
        g.node(NodeKind.PERSON, name, status="active")
    g.node(NodeKind.CHANNEL, "슬랙")
    g.edge(EdgeKind.DETECTED_BY, PAY, slow)
    g.edge(EdgeKind.DETECTED_BY, PAY, ALARM)
    g.edge(EdgeKind.NOTIFIES, slow, PARK)
    g.edge(EdgeKind.NOTIFIES, ALARM, KIM)
    g.edge(EdgeKind.REACHABLE_VIA, KIM, SLACK)
    g.edge(EdgeKind.REACHABLE_VIA, PARK, SLACK)
    graph = g.build()
    absent = _templates_by_key(graph)["owner_absent"]
    assert _target(absent) == KIM
    assert absent.title == "결제 · 1차 담당 부재(김대리)"
    # the stress is real: without 김대리 the first answer moves to the 30-minute alarm
    acks = [e for e in replay(graph, absent).events if e.kind is SimEventKind.ACK]
    assert [(e.t, e.actor_id) for e in acks] == [(35.0, PARK)]


def test_templates_ignore_held_facts_like_the_replay() -> None:
    graph_builder = Graph()
    nodes: list[tuple[NodeKind, str, dict[str, Any]]] = [
        (NodeKind.SERVICE, "결제", {"tier": 1}),
        (NodeKind.TRIGGER, "결제 알람", {"delay_minutes": 3}),
        (NodeKind.PERSON, "가주임", {"status": "active"}),
        (NodeKind.PERSON, "김대리", {"status": "active"}),
        (NodeKind.CHANNEL, "슬랙", {}),
        (NodeKind.CHANNEL, "전화", {}),
    ]
    for kind, name, attrs in nodes:
        graph_builder.node(kind, name, **attrs)
    ga = node_id(NodeKind.PERSON, "가주임")
    graph_builder.edge(EdgeKind.DETECTED_BY, PAY, ALARM)
    held = graph_builder.edge(EdgeKind.NOTIFIES, ALARM, ga)  # 동명이인 등으로 판단 보류
    graph_builder.edge(EdgeKind.NOTIFIES, ALARM, KIM)
    graph_builder.edge(EdgeKind.REACHABLE_VIA, ga, PHONE)
    graph_builder.edge(EdgeKind.REACHABLE_VIA, KIM, SLACK)
    _hold_edge(graph_builder, held)
    graph = graph_builder.build()
    by_key = _templates_by_key(graph, policy=Policy(obligations=()))
    assert _target(by_key["owner_absent"]) == KIM  # 가주임 is only reachable through a held edge
    assert _target(by_key["channel_down"]) == SLACK
    acks = [e for e in replay(graph, by_key["baseline"]).events if e.kind is SimEventKind.ACK]
    assert acks[0].actor_id == KIM


def test_held_service_keeps_only_its_baseline() -> None:
    g = Graph()
    g.node(NodeKind.SERVICE, "결제", tier=1)
    g.node(NodeKind.PERSON, "김대리", status="active")
    g.node(NodeKind.CHANNEL, "슬랙")
    g.edge(EdgeKind.OWNED_BY, PAY, KIM)
    g.edge(EdgeKind.REACHABLE_VIA, KIM, SLACK)
    g.b.nodes[PAY] = g.b.nodes[PAY].model_copy(update={"verification": Verification.HOLD})
    graph = g.build()
    scenarios = template_scenarios(CheckContext(graph, policy=Policy(obligations=())))
    assert [s.id for s in scenarios] == [f"tpl:{PAY}:baseline"]
    assert replay(graph, scenarios[0]).outcome is Outcome.FAIL  # unconfirmed ≠ ready


def test_channel_down_covers_team_level_channels() -> None:
    g = Graph()
    g.node(NodeKind.SERVICE, "결제", tier=1)
    g.node(NodeKind.TRIGGER, "결제 알람", delay_minutes=3)  # no NOTIFIES: the owner team is paged
    team = g.node(NodeKind.TEAM, "결제팀")
    g.node(NodeKind.PERSON, "김대리", status="active")
    g.node(NodeKind.CHANNEL, "전화")
    g.edge(EdgeKind.DETECTED_BY, PAY, ALARM)
    g.edge(EdgeKind.MEMBER_OF, KIM, team)
    g.edge(EdgeKind.OWNED_BY, PAY, team)
    g.edge(EdgeKind.REACHABLE_VIA, team, PHONE)  # 팀 공용 당직폰
    graph = g.build()
    by_key = _templates_by_key(graph, policy=Policy(obligations=()))
    assert _target(by_key["channel_down"]) == PHONE
    assert by_key["channel_down"].title == "결제 · 주 채널 다운(전화)"
    baseline = replay(graph, by_key["baseline"])
    assert any(e.kind is SimEventKind.ACK and e.actor_id == KIM for e in baseline.events)
    result = replay(graph, by_key["channel_down"])
    assert result.failed_element is Element.OWNER  # the shared phone was the only way in
    assert result.failed_at_id == team


def test_channel_down_hits_the_channel_first_pages_actually_use() -> None:
    """Both alarms page over 전화; 슬랙 is merely listed. Old votes tied and picked 슬랙 by id."""
    g = Graph()
    g.node(NodeKind.SERVICE, "결제", tier=1)
    g.node(NodeKind.TRIGGER, "결제 알람", delay_minutes=3)
    for name in ("김대리", "박팀장"):
        g.node(NodeKind.PERSON, name, status="active")
    g.node(NodeKind.CHANNEL, "슬랙")
    g.node(NodeKind.CHANNEL, "전화")
    g.edge(EdgeKind.DETECTED_BY, PAY, ALARM)
    g.edge(EdgeKind.NOTIFIES, ALARM, KIM, channel=PHONE)
    g.edge(EdgeKind.NOTIFIES, ALARM, PARK, channel=PHONE)
    g.edge(EdgeKind.REACHABLE_VIA, KIM, SLACK)
    g.edge(EdgeKind.REACHABLE_VIA, PARK, SLACK)
    assert SLACK < PHONE
    by_key = _templates_by_key(g.build(), policy=Policy(obligations=()))
    assert _target(by_key["channel_down"]) == PHONE


def test_approver_absent_targets_the_self_approving_first_responder() -> None:
    g = Graph()
    g.node(NodeKind.SERVICE, "결제", tier=1)
    g.node(NodeKind.TRIGGER, "결제 알람", delay_minutes=3)
    for name in ("김대리", "박팀장"):
        g.node(NodeKind.PERSON, name, status="active")
    g.node(NodeKind.ROLE, "팀장")
    g.node(NodeKind.CHANNEL, "전화")
    g.node(NodeKind.ACTION, "결제 롤백", action_type="rollback", duration_minutes=10)
    g.edge(EdgeKind.DETECTED_BY, PAY, ALARM)
    g.edge(EdgeKind.NOTIFIES, ALARM, PARK)
    g.edge(EdgeKind.HOLDS_ROLE, KIM, LEAD)  # 김대리 sorts first among the 팀장 holders …
    g.edge(EdgeKind.HOLDS_ROLE, PARK, LEAD)
    g.edge(EdgeKind.REACHABLE_VIA, KIM, PHONE)
    g.edge(EdgeKind.REACHABLE_VIA, PARK, PHONE)
    g.edge(EdgeKind.MITIGATED_BY, PAY, ROLLBACK)
    g.edge(EdgeKind.APPROVED_BY, ROLLBACK, LEAD)
    graph = g.build()
    baseline = replay(graph, _templates_by_key(graph)["baseline"])
    approved = next(e for e in baseline.events if e.kind is SimEventKind.APPROVED)
    assert approved.actor_id == PARK  # … but 박팀장 answers first and approves directly
    absent = _templates_by_key(graph)["approver_absent"]
    assert _target(absent) == PARK
    assert absent.title == "결제 · 승인자 부재(박팀장)"


def test_approver_absent_skips_an_approver_nobody_can_page() -> None:
    g = Graph()
    g.node(NodeKind.SERVICE, "결제", tier=1)
    g.node(NodeKind.TRIGGER, "결제 알람", delay_minutes=3)
    g.node(NodeKind.PERSON, "가무연락", status="active")  # 팀장 holder without any channel
    nobody = node_id(NodeKind.PERSON, "가무연락")
    for name in ("김대리", "박팀장"):
        g.node(NodeKind.PERSON, name, status="active")
    g.node(NodeKind.ROLE, "팀장")
    g.node(NodeKind.CHANNEL, "전화")
    g.node(NodeKind.ACTION, "결제 롤백", action_type="rollback")
    g.edge(EdgeKind.DETECTED_BY, PAY, ALARM)
    g.edge(EdgeKind.NOTIFIES, ALARM, KIM)
    g.edge(EdgeKind.HOLDS_ROLE, nobody, LEAD)
    g.edge(EdgeKind.HOLDS_ROLE, PARK, LEAD)
    g.edge(EdgeKind.REACHABLE_VIA, KIM, PHONE)
    g.edge(EdgeKind.REACHABLE_VIA, PARK, PHONE)
    g.edge(EdgeKind.MITIGATED_BY, PAY, ROLLBACK)
    g.edge(EdgeKind.APPROVED_BY, ROLLBACK, LEAD)
    graph = g.build()
    absent = _templates_by_key(graph)["approver_absent"]
    assert _target(absent) == PARK
    assert replay(graph, absent).failed_element is Element.APPROVER


def test_empty_graph_has_no_templates() -> None:
    assert template_scenarios(CheckContext(OpsGraph())) == []


# --------------------------------------------------------------------------- reviewer: LLM text


def test_llm_text_is_nfc_without_invisible_or_unencodable_characters() -> None:
    nfd_title = unicodedata.normalize("NFD", "추석 연휴 새벽 3시") + " \u202e슬랙\u200b 장애\ud800"
    nfd_long = unicodedata.normalize("NFD", "가" * 100)  # 200 jamo code points
    payload = {
        "scenarios": [
            item(PAY, pert("person_unavailable", KIM), title=nfd_title, description="\ufeff결제 오류"),
            item(ORDER, pert("person_unavailable", KIM), title=nfd_long),
        ]
    }
    first, second = run(FakeLLM(payload), CheckContext(rich()))
    assert first.title == "추석 연휴 새벽 3시 슬랙 장애"
    assert first.description == "결제 오류"
    assert second.title == "가" * 80  # capped after composing, never mid-syllable
    for s in (first, second):
        assert unicodedata.is_normalized("NFC", s.title)
        s.model_dump_json()  # a lone surrogate used to make this raise


def test_llm_is_not_called_without_a_confirmed_service() -> None:
    llm = FakeLLM({"scenarios": [item(PAY)]})
    assert run(llm, CheckContext(OpsGraph())) == []
    g = Graph()
    g.node(NodeKind.SERVICE, "결제", tier=1)
    g.b.nodes[PAY] = g.b.nodes[PAY].model_copy(update={"verification": Verification.HOLD})
    assert run(llm, CheckContext(g.build())) == []
    assert llm.calls == []


def test_llm_sees_and_may_use_only_confirmed_ids() -> None:
    g = Graph()
    g.node(NodeKind.SERVICE, "결제", tier=1)
    g.node(NodeKind.PERSON, "김대리", status="active")
    g.node(NodeKind.PERSON, "박팀장", status="active")
    g.b.nodes[PARK] = g.b.nodes[PARK].model_copy(update={"verification": Verification.HOLD})
    ctx = CheckContext(g.build())
    llm = FakeLLM(
        {
            "scenarios": [
                item(PAY, pert("person_unavailable", PARK)),
                item(PAY, pert("person_unavailable", KIM)),
            ]
        }
    )
    (only,) = run(llm, ctx)
    assert only.perturbations[0].target_id == KIM
    assert PARK not in llm.calls[0]["context"]
    assert f"- 사람(person): {KIM}" in llm.calls[0]["context"]


def test_llm_context_lists_first_responders_in_paging_order() -> None:
    ctx = CheckContext(rich())
    llm = FakeLLM({"scenarios": []})
    assert run(llm, ctx) == []
    expected = ", ".join(p.id for p in first_responders(ctx, PAY))
    assert f"- {PAY}: {expected}" in llm.calls[0]["context"]


# --------------------------------------------------------------------------- reviewer: properties

_PEOPLE = [node_id(NodeKind.PERSON, n) for n in ("p0", "p1", "p2")]
_GROUPS = [node_id(NodeKind.ROLE, "r0"), node_id(NodeKind.TEAM, "t0")]
_CHANNELS = [node_id(NodeKind.CHANNEL, n) for n in ("c0", "c1")]
_TRIGGERS = [node_id(NodeKind.TRIGGER, n) for n in ("g0", "g1")]
_SERVICES = [node_id(NodeKind.SERVICE, n) for n in ("s0", "s1")]
_ACTION = node_id(NodeKind.ACTION, "a0")
_LOCATION = node_id(NodeKind.LOCATION, "l0")


@st.composite
def response_graphs(draw: st.DrawFn) -> OpsGraph:
    """Random response paths: triggers, notify/own targets, groups, channels, approvals, HOLD."""
    g = Graph()
    for sid in _SERVICES:
        g.node(NodeKind.SERVICE, sid.split(":", 1)[1], tier=draw(st.sampled_from([1, 2, 3])))
    for pid in _PEOPLE:
        status = draw(st.sampled_from(["active", "active", "departed", "unknown"]))
        g.node(NodeKind.PERSON, pid.split(":", 1)[1], status=status)
    g.node(NodeKind.ROLE, "r0")
    g.node(NodeKind.TEAM, "t0")
    for cid in _CHANNELS:
        g.node(NodeKind.CHANNEL, cid.split(":", 1)[1])
    for tid in _TRIGGERS:
        g.node(NodeKind.TRIGGER, tid.split(":", 1)[1], delay_minutes=draw(st.integers(0, 20)))
    g.node(NodeKind.ACTION, "a0", action_type=draw(st.sampled_from(["rollback", "restart"])))
    g.node(NodeKind.LOCATION, "l0")
    targets = _PEOPLE + _GROUPS
    candidates: list[tuple[EdgeKind, str, str]] = [
        *((EdgeKind.DETECTED_BY, s, t) for s in _SERVICES for t in _TRIGGERS),
        *((EdgeKind.NOTIFIES, t, x) for t in _TRIGGERS for x in targets),
        *((EdgeKind.OWNED_BY, s, x) for s in _SERVICES for x in targets),
        *((EdgeKind.REACHABLE_VIA, x, c) for x in targets for c in _CHANNELS),
        *((EdgeKind.HOLDS_ROLE, p, _GROUPS[0]) for p in _PEOPLE),
        *((EdgeKind.MEMBER_OF, p, _GROUPS[1]) for p in _PEOPLE),
        *((EdgeKind.MITIGATED_BY, s, _ACTION) for s in _SERVICES),
        *((EdgeKind.APPROVED_BY, _ACTION, x) for x in [*_PEOPLE, _GROUPS[0]]),
        *((EdgeKind.RUNS_IN, x, _LOCATION) for x in [*_SERVICES, *_CHANNELS]),
    ]
    for kind, src, dst in draw(st.lists(st.sampled_from(candidates), unique=True, max_size=25)):
        attrs: dict[str, Any] = {}
        if kind is EdgeKind.NOTIFIES and draw(st.booleans()):
            attrs["channel"] = draw(st.sampled_from(_CHANNELS))
        eid = g.edge(kind, src, dst, **attrs)
        if draw(st.integers(0, 7)) == 0:
            _hold_edge(g, eid)
    return g.build()


@settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(graph=response_graphs())
def test_templates_aim_at_what_the_baseline_replay_uses(graph: OpsGraph) -> None:
    ctx = CheckContext(graph)
    scenarios = template_scenarios(ctx)
    assert scenarios == template_scenarios(CheckContext(graph))
    assert len({s.id for s in scenarios}) == len(scenarios)
    seed_edges = {e.id for e in graph.edges.values() if e.kind in (EdgeKind.NOTIFIES, EdgeKind.OWNED_BY)}
    by_service: dict[str, dict[str, Scenario]] = {}
    for s in scenarios:
        assert s.title.startswith(graph.nodes[s.service_id].name + " · ")
        by_service.setdefault(s.service_id, {})[s.id.rsplit(":", 1)[1]] = s
    for sid, keyed in by_service.items():
        baseline = replay(graph, keyed["baseline"])
        owner_ack = next(
            (e for e in baseline.events if e.kind is SimEventKind.ACK and e.edge_id in seed_edges), None
        )
        if owner_ack is not None:
            # whoever answers the first page at baseline is exactly whom owner_absent removes
            assert _target(keyed["owner_absent"]) == owner_ack.actor_id, sid
            page = next(
                e
                for e in baseline.events
                if e.kind is SimEventKind.PAGE and e.target_id == owner_ack.actor_id
            )
            assert "channel_down" in keyed
            assert page.actor_id is not None
            assert graph.nodes[page.actor_id].kind is NodeKind.CHANNEL
        for s in keyed.values():
            for p in s.perturbations:
                target = graph.nodes[p.target_id]
                assert target.verification is Verification.VERIFIED


# --------------------------------------------------------------------------- reviewer 2: adversarial


def _approval_graph(**rollback: Any) -> OpsGraph:
    """결제 → 알람 → 김대리(전화); 롤백 승인자 팀장(박팀장, 전화)."""
    g = Graph()
    g.node(NodeKind.SERVICE, "결제", tier=1)
    g.node(NodeKind.TRIGGER, "결제 알람", delay_minutes=3)
    for name in ("김대리", "박팀장"):
        g.node(NodeKind.PERSON, name, status="active")
    g.node(NodeKind.ROLE, "팀장")
    g.node(NodeKind.CHANNEL, "전화")
    g.node(NodeKind.ACTION, "결제 롤백", duration_minutes=10, **rollback)
    g.edge(EdgeKind.DETECTED_BY, PAY, ALARM)
    g.edge(EdgeKind.NOTIFIES, ALARM, KIM)
    g.edge(EdgeKind.HOLDS_ROLE, PARK, LEAD)
    g.edge(EdgeKind.REACHABLE_VIA, KIM, PHONE)
    g.edge(EdgeKind.REACHABLE_VIA, PARK, PHONE)
    g.edge(EdgeKind.MITIGATED_BY, PAY, ROLLBACK)
    g.edge(EdgeKind.APPROVED_BY, ROLLBACK, LEAD)
    return g.build()


@pytest.mark.parametrize(
    ("rollback", "expected"),
    [
        ({"action_type": "rollback", "requires_approval": False}, False),  # approver named, never asked
        ({"action_type": "rollback"}, True),
        ({"action_type": "restart"}, False),  # an approver edge alone does not gate a restart
        ({"action_type": "restart", "requires_approval": True}, True),  # "박팀장 승인 후 재시작"
        ({"action_type": "restart", "requires_approval": False}, False),
    ],
)
def test_approver_absent_only_when_the_replay_would_ask_the_approver(
    rollback: dict[str, Any], expected: bool
) -> None:
    """A waived approval used to yield '승인자 부재(박팀장)' — a stress identical to the baseline."""
    graph = _approval_graph(**rollback)
    by_key = _templates_by_key(graph, policy=Policy(obligations=()))
    assert ("approver_absent" in by_key) is expected
    baseline = replay(graph, by_key["baseline"])
    asked_park = any(e.kind is SimEventKind.APPROVED and e.actor_id == PARK for e in baseline.events)
    assert asked_park is expected
    if expected:
        absent = replay(graph, by_key["approver_absent"])
        assert absent.failed_element is Element.APPROVER  # the stress really bites


@pytest.mark.parametrize(
    "title",
    [
        "\u3164\u3164",  # HANGUL FILLER: a 'blank' Korean name
        "\u115f\u1160",  # choseong/jungseong fillers
        " \uffa0 \u200b ",  # halfwidth filler + zero-width space
    ],
)
def test_invisible_hangul_titles_fall_back_to_the_default_title(title: str) -> None:
    payload = {"scenarios": [item(PAY, pert("person_unavailable", KIM), title=title, description=title)]}
    (only,) = run(FakeLLM(payload), CheckContext(rich()))
    assert only.title == "결제 · AI 제안 시나리오"
    assert only.description == ""


def test_hangul_fillers_inside_text_become_single_spaces() -> None:
    payload = {
        "scenarios": [item(PAY, pert("person_unavailable", KIM), title="\u3164결제\u3164\u3164장애\u3164")]
    }
    (only,) = run(FakeLLM(payload), CheckContext(rich()))
    assert only.title == "결제 장애"


def test_negative_zero_start_is_the_same_idea_as_zero() -> None:
    """JSON spells -0.0 and 0.0 differently; they used to get two ids for one scenario."""
    ctx = CheckContext(rich())
    negative = item(PAY, pert("person_unavailable", KIM, -0.0), title="A")
    positive = item(PAY, pert("person_unavailable", KIM, 0.0), title="B")
    scenarios = run(FakeLLM({"scenarios": [negative, positive]}), ctx)
    assert [s.title for s in scenarios] == ["A"]
    assert scenarios[0].perturbations[0].start_minute == 0.0
    assert str(scenarios[0].perturbations[0].start_minute) == "0.0"


_EXPECTED_TARGET_KIND = {
    "owner_absent": NodeKind.PERSON,
    "approver_absent": NodeKind.PERSON,
    "channel_down": NodeKind.CHANNEL,
    "location_down": NodeKind.LOCATION,
    "concurrent": NodeKind.SERVICE,
}


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(graph=response_graphs(), waive=st.sampled_from([None, True, False]))
def test_every_template_targets_what_its_key_promises(graph: OpsGraph, waive: bool | None) -> None:
    if waive is not None:
        builder = GraphBuilder(graph)
        action = builder.nodes[_ACTION]
        attrs = {**action.attrs, "requires_approval": Attr(value=waive, source=SourceKind.CATALOG)}
        builder.nodes[_ACTION] = action.model_copy(update={"attrs": attrs})
        graph = builder.build()
    ctx = CheckContext(graph)
    critical = {s.id for s in ctx.critical_services()}
    for s in template_scenarios(ctx):
        key = s.id.rsplit(":", 1)[1]
        assert s.id == f"tpl:{s.service_id}:{key}"
        result = replay(graph, s)  # every template replays without raising
        assert result.scenario_id == s.id
        if key in ("baseline", "breach_report"):
            assert s.perturbations == ()
            continue
        (p,) = s.perturbations
        assert graph.nodes[p.target_id].kind is _EXPECTED_TARGET_KIND[key]
        if key == "concurrent":
            assert p.target_id in critical
            assert p.target_id != s.service_id
        if key == "approver_absent":
            assert waive is not False  # a waived approval never yields an approver stress


# --------------------------------------------------------------------------- human scenarios (YAML)

DEMO_DIR = Path(__file__).resolve().parents[1] / "examples" / "dmc-shop"


@pytest.fixture(scope="module")
def demo_ctx() -> CheckContext:
    """The finals demo bundle through the LLM-off pipeline (rule parser → assemble)."""
    bundle = load_bundle_dir(DEMO_DIR)
    extractor = RuleExtractor()
    extractions = [
        extractor.extract(d, roster=bundle.roster, catalog=bundle.catalog) for d in bundle.documents
    ]
    graph = assemble(
        extractions,
        bundle.documents,
        roster=bundle.roster,
        catalog=bundle.catalog,
        oncall=bundle.oncall,
        policy=bundle.policy,
    ).graph
    return CheckContext(graph, bundle.policy, bundle.roster, bundle.oncall, bundle.catalog)


def human(ctx: CheckContext, text: str) -> list[Scenario]:
    return scenarios_from_yaml(text, ctx)


def one(ctx: CheckContext, kind: str, target: str, *, service: str = "결제") -> list[str]:
    """Target ids one perturbation resolves to."""
    text = f"- id: t\n  service: {service}\n  perturbations:\n    - kind: {kind}\n      target: {target}\n"
    [scenario] = human(ctx, text)
    return [p.target_id for p in scenario.perturbations]


def test_demo_file_resolves_every_target(demo_ctx: CheckContext) -> None:
    scenarios = scenarios_from_yaml(DEMO_DIR / "scenarios.yaml", demo_ctx)
    by_id = {s.id: s for s in scenarios}
    assert list(by_id) == [
        "human:dmc-s1-chuseok-payment",
        "human:dmc-s2-sangam-idc-fire",
        "human:dmc-s3-slack-outage-order",
        "human:dmc-s4-breach-report",
    ]
    assert all(s.origin is ScenarioOrigin.HUMAN for s in scenarios)
    targets = {s.id: [(p.kind.value, p.target_id) for p in s.perturbations] for s in scenarios}
    assert targets["human:dmc-s1-chuseok-payment"] == [
        ("channel_down", "channel:slack"),
        ("person_unavailable", "person:한서윤"),
    ]
    assert targets["human:dmc-s2-sangam-idc-fire"] == [
        ("location_down", "location:상암-idc"),
        ("system_down", "system:astronomy-db"),
        ("system_down", "system:kafka"),
        ("channel_down", "channel:internal_messenger"),
        ("concurrent_incident", "service:주문"),
    ]
    s4 = by_id["human:dmc-s4-breach-report"]
    assert (s4.service_id, s4.incident_type, s4.deadline_minutes) == (
        "service:결제",
        IncidentType.BREACH,
        1440.0,
    )
    s1 = by_id["human:dmc-s1-chuseok-payment"]
    assert s1.deadline_minutes is None  # the tier SLO applies
    assert s1.description.endswith("(재현용 flagd 플래그: paymentFailure)")
    assert by_id["human:dmc-s2-sangam-idc-fire"].service_id == "service:상품-카탈로그"


def test_path_bytes_and_text_sources_agree(demo_ctx: CheckContext) -> None:
    path = DEMO_DIR / "scenarios.yaml"
    from_path = scenarios_from_yaml(path, demo_ctx)
    assert scenarios_from_yaml(str(path), demo_ctx) == from_path
    assert scenarios_from_yaml(path.read_bytes(), demo_ctx) == from_path
    assert scenarios_from_yaml(path.read_text(encoding="utf-8"), demo_ctx) == from_path
    bom = b"\xef\xbb\xbf" + path.read_bytes().replace(b"\n", b"\r\n")
    assert scenarios_from_yaml(bom, demo_ctx) == from_path  # BOM + CRLF (Windows editors)


@pytest.mark.parametrize(
    ("kind", "target", "expected"),
    [
        ("person_unavailable", "person:한서윤", ["person:한서윤"]),  # node id
        ("person_unavailable", "김대리", ["person:김민준"]),  # roster alias
        ("person_unavailable", "오준석 CISO", ["person:오준석"]),  # "이름 직급"
        ("person_unavailable", "재현 팀장", ["person:박재현"]),
        ("channel_down", "슬랙", ["channel:slack"]),  # Korean medium word
        ("channel_down", "Slack", ["channel:slack"]),
        ("channel_down", "사내 메신저", ["channel:internal_messenger"]),  # catalog channel name
        ("channel_down", "사내메신저", ["channel:internal_messenger"]),
        ("channel_down", "휴대폰", ["channel:phone"]),
        ("system_down", "PostgreSQL", ["system:astronomy-db"]),  # catalog alias
        ("system_down", "카프카", ["system:kafka"]),
        ("location_down", "상암IDC", ["location:상암-idc"]),  # spacing never matters
        ("concurrent_incident", "checkout", ["service:주문"]),  # OTel name kept as alias
        ("concurrent_incident", "프론트엔드", ["service:웹-프론트엔드"]),
    ],
)
def test_targets_resolve_the_way_people_write_them(
    demo_ctx: CheckContext, kind: str, target: str, expected: list[str]
) -> None:
    assert one(demo_ctx, kind, target) == expected


def test_service_by_alias_and_scenario_defaults(demo_ctx: CheckContext) -> None:
    [s] = human(demo_ctx, "- id: 7\n  service: payment\n")
    assert (s.id, s.service_id, s.incident_type) == ("human:7", "service:결제", IncidentType.OUTAGE)
    assert s.title == "결제 · 사람 작성 시나리오"
    assert (s.description, s.perturbations, s.deadline_minutes) == ("", (), None)


def test_perturbation_window_is_kept(demo_ctx: CheckContext) -> None:
    text = (
        "- id: w\n  service: 결제\n  perturbations:\n"
        "    - {kind: channel_down, target: slack, start_minute: 10, end_minute: 40.5}\n"
    )
    [s] = human(demo_ctx, text)
    [p] = s.perturbations
    assert (p.start_minute, p.end_minute) == (10.0, 40.5)


def test_empty_file_has_no_scenarios(demo_ctx: CheckContext) -> None:
    assert human(demo_ctx, "# 아직 시나리오 없음\n") == []


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("- id: x\n  service: 없는서비스\n", "서비스 '없는서비스'를 그래프에서 찾을 수 없습니다"),
        (
            "- id: x\n  service: 결제\n  perturbations:\n    - {kind: person_unavailable, target: 홍길동}\n",
            "사람 '홍길동'을 그래프에서 찾을 수 없습니다",
        ),
        (
            "- id: x\n  service: 결제\n  perturbations:\n    - {kind: location_down, target: astronomy-db}\n",
            "'astronomy-db'은(는) 시스템 이름입니다. 여기에는 위치 이름을",  # Latin: reading unknown
        ),
        (
            "- id: x\n  service: 결제\n  perturbations:\n"
            "    - {kind: person_unavailable, target: 'action:결제-서비스-롤백'}\n",
            "'action:결제-서비스-롤백'은 조치 이름입니다. 여기에는 사람 이름을",  # a node id of any kind
        ),
        ("- id: x\n  service: role:cto\n", "'role:cto'은(는) 역할 이름입니다. 여기에는 서비스 이름을"),
        (
            "- id: x\n  service: 결제\n  perturbations:\n"
            "    - {kind: concurrent_incident, target: payment}\n",
            "동시 장애 대상은 시나리오의 서비스('결제')가 아닌",
        ),
        ("- id: x\n  service: 결제\n  perturbation: []\n", "알 수 없는 항목이 있습니다: perturbation"),
        ("- id: x\n  service: 결제\n  incident_type: 해킹\n", "incident_type 값을 알 수 없습니다: '해킹'"),
        (
            "- id: x\n  service: 결제\n  perturbations:\n    - {kind: slack_down, target: slack}\n",
            "kind 값을 알 수 없습니다: 'slack_down'",
        ),
        ("- id: x\n  service: 결제\n  deadline_minutes: -5\n", "0 이상의 숫자(분)"),
        ("- id: x\n  service: 결제\n  deadline_minutes: .nan\n", "0 이상의 숫자(분)"),
        (
            "- id: x\n  service: 결제\n  perturbations:\n"
            "    - {kind: channel_down, target: slack, start_minute: 30, end_minute: 30}\n",
            "end_minute(30)는 start_minute(30)보다 커야",
        ),
        ("- id: x\n  service: 결제\n- id: x\n  service: 주문\n", "시나리오 id가 겹칩니다: 'x'"),
        ("- service: 결제\n", "1번째 항목에 id 값이 없습니다"),
        ("- id: x\n", "1번째 항목에 service 값이 없습니다"),
        ("scenarios: []\n", "최상위는 시나리오 목록"),
        ("- 그냥 글자\n", "'키: 값' 묶음"),
        ("- id: x\n  service: 결제\n  perturbations: slack\n", "perturbations는 목록"),
        ("- id: x\n  service: [결제\n", "시나리오 파일의 YAML/JSON 형식이 올바르지 않습니다 (3행"),
        ("- id: x\n  service: 결제\n  perturbations:\n    - {kind: channel_down}\n", "target 값이 없습니다"),
    ],
)
def test_anything_that_would_test_something_else_is_rejected_in_korean(
    demo_ctx: CheckContext, text: str, fragment: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(fragment)):
        human(demo_ctx, text)


def test_missing_file_is_reported(demo_ctx: CheckContext, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="시나리오 파일을 찾을 수 없습니다"):
        scenarios_from_yaml(tmp_path / "없음.yaml", demo_ctx)


def test_ambiguous_person_must_be_written_as_a_node_id() -> None:
    g = Graph()
    g.node(NodeKind.SERVICE, "결제", tier=1)
    g.node(NodeKind.PERSON, "이준호-결제팀", status="active")
    g.node(NodeKind.PERSON, "이준호-개발팀", status="active")
    b = g.b
    for nid in list(b.nodes):  # two roster namesakes share the display name
        if nid.startswith("person:"):
            b.nodes[nid] = b.nodes[nid].model_copy(update={"name": "이준호"})
    ctx = CheckContext(g.build())
    with pytest.raises(
        ValueError, match=r"'이준호'에 해당하는 사람이 여럿입니다\(person:이준호-개발팀, person:"
    ):
        one(ctx, "person_unavailable", "이준호")
    assert one(ctx, "person_unavailable", "person:이준호-결제팀") == ["person:이준호-결제팀"]


def test_a_medium_takes_down_every_channel_of_that_medium() -> None:
    g = Graph()
    g.node(NodeKind.SERVICE, "결제", tier=1)
    g.node(NodeKind.CHANNEL, "결제팀 슬랙", medium="slack")
    g.node(NodeKind.CHANNEL, "장애 슬랙", medium="slack")
    g.node(NodeKind.CHANNEL, "전화", medium="phone")
    ctx = CheckContext(g.build())
    assert one(ctx, "channel_down", "slack") == ["channel:결제팀-슬랙", "channel:장애-슬랙"]
    assert one(ctx, "channel_down", "장애 슬랙") == ["channel:장애-슬랙"]  # a named room is just that room


def test_human_scenarios_replay_deterministically(demo_ctx: CheckContext) -> None:
    scenarios = scenarios_from_yaml(DEMO_DIR / "scenarios.yaml", demo_ctx)
    first = [r.result_hash() for r in replay_all(demo_ctx.graph, scenarios, policy=demo_ctx.policy)]
    again = [r.result_hash() for r in replay_all(demo_ctx.graph, scenarios, policy=demo_ctx.policy)]
    assert first == again
    assert [s.fingerprint() for s in scenarios] == [
        s.fingerprint() for s in scenarios_from_yaml(DEMO_DIR / "scenarios.yaml", demo_ctx)
    ]


_NAMED_KINDS = {
    PerturbationKind.PERSON_UNAVAILABLE: NodeKind.PERSON,
    PerturbationKind.SYSTEM_DOWN: NodeKind.SYSTEM,
    PerturbationKind.LOCATION_DOWN: NodeKind.LOCATION,
    PerturbationKind.CONCURRENT_INCIDENT: NodeKind.SERVICE,
}


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.data())
def test_every_node_resolves_from_its_id_or_name(demo_ctx: CheckContext, data: st.DataObject) -> None:
    """Whatever spelling of a node a person picks (id, name, spaced or not), it names that node."""
    kind = data.draw(st.sampled_from(sorted(_NAMED_KINDS, key=str)))
    candidates = [n for n in demo_ctx.graph.nodes_of(_NAMED_KINDS[kind]) if n.id != "service:결제"]
    node = data.draw(st.sampled_from(candidates))
    spelling = data.draw(
        st.sampled_from([node.id, node.name, node.name.replace(" ", ""), f"  {node.name}  "])
    )
    assert one(demo_ctx, kind.value, f"'{spelling}'") == [node.id]
