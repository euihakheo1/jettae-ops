"""The finals demo (``examples/dmc-shop``) end to end with the LLM off.

ingest → rule parser per document → assemble → back-translation (no LLM) → CheckContext →
fixed rules → score → template + human scenarios → replay. The README of the example lists the
defects planted in it on purpose; every one of them must be found by the right rule on the
right service, the well-written control (장바구니) must stay healthy, and the whole run must be
reproducible to the hash (P6). When a planted fact is missed or a false alarm fires, the fix
belongs in the rule parser or the assembler — never in the fixture or a weakened rule.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

from jettae.check.context import CheckContext
from jettae.check.elements import evaluate_service
from jettae.check.impact import blast_radius
from jettae.check.registry import rule_catalog, run_checks
from jettae.check.scoring import score
from jettae.domain import (
    Element,
    ElementStatus,
    Finding,
    FindingStatus,
    InputBundle,
    NodeKind,
    OpsGraph,
    Outcome,
    ReadinessReport,
    RehearsalResult,
    Scenario,
    ScenarioOrigin,
    SecurityNotice,
    Severity,
    SimEventKind,
    SourceKind,
)
from jettae.extract import (
    AssemblyResult,
    RuleExtractor,
    VerificationResult,
    assemble,
    verify_backtranslation,
)
from jettae.ingest import load_bundle_dir, scan_injection
from jettae.rehearse import replay_all, scenarios_from_yaml, template_scenarios
from support_loops import CHILD_PRELUDE

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "dmc-shop"
SECURITY_RUNBOOK = "runbooks/03-보안-개인정보-신고.md"


@dataclass(frozen=True)
class DemoRun:
    bundle: InputBundle
    notices: tuple[SecurityNotice, ...]
    assembled: AssemblyResult
    verified: VerificationResult
    ctx: CheckContext
    findings: tuple[Finding, ...]
    readiness: ReadinessReport
    scenarios: tuple[Scenario, ...]
    results: tuple[RehearsalResult, ...]

    @property
    def graph(self) -> OpsGraph:
        return self.verified.graph


def pipeline(bundle_dir: Path = DEMO) -> DemoRun:
    bundle = load_bundle_dir(bundle_dir)
    notices = tuple(n for doc in bundle.documents for n in scan_injection(doc)[1])
    extractor = RuleExtractor()
    extractions = [
        extractor.extract(d, roster=bundle.roster, catalog=bundle.catalog) for d in bundle.documents
    ]
    assembled = assemble(
        extractions,
        bundle.documents,
        roster=bundle.roster,
        catalog=bundle.catalog,
        oncall=bundle.oncall,
        policy=bundle.policy,
    )
    with asyncio.Runner() as runner:
        verified = runner.run(verify_backtranslation(assembled.graph, bundle.documents, None))
    ctx = CheckContext(verified.graph, bundle.policy, bundle.roster, bundle.oncall, bundle.catalog)
    findings = run_checks(ctx)
    scenarios = (*template_scenarios(ctx), *scenarios_from_yaml(bundle_dir / "scenarios.yaml", ctx))
    results = replay_all(verified.graph, scenarios, policy=bundle.policy, roster=bundle.roster)
    return DemoRun(
        bundle=bundle,
        notices=notices,
        assembled=assembled,
        verified=verified,
        ctx=ctx,
        findings=tuple(findings),
        readiness=score(ctx, findings),
        scenarios=scenarios,
        results=tuple(results),
    )


def digest(run: DemoRun) -> str:
    """Everything a viewer of the demo sees, in one hash: graph, findings, scores, rehearsals."""
    payload = {
        "graph": run.graph.content_hash(),
        "findings": [f.model_dump(mode="json") for f in run.findings],
        "readiness": run.readiness.model_dump(mode="json"),
        "scenarios": [s.model_dump(mode="json") for s in run.scenarios],
        "results": [r.result_hash() for r in run.results],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@pytest.fixture(scope="module")
def run() -> DemoRun:
    return pipeline()


def sid(run: DemoRun, name: str) -> str:
    [node] = [s for s in run.ctx.services() if s.name == name]
    return node.id


def found(run: DemoRun, rule_id: str, service: str | None) -> list[Finding]:
    service_id = None if service is None else sid(run, service)
    return [f for f in run.findings if f.rule_id == rule_id and f.service_id == service_id]


def result(run: DemoRun, scenario_id: str) -> RehearsalResult:
    [hit] = [r for r in run.results if r.scenario_id == scenario_id]
    return hit


# --------------------------------------------------------------------------- planted defects

#: README "데모 시나리오 · 1. 진단" rows → (rule, service, words the message must say).
INTENDED: tuple[tuple[str, str, str | None, tuple[str, ...]], ...] = (
    ("1 결제: 퇴사자 김대리가 유일한 연락 대상", "JT-PPL-001", "결제", ("김민준",)),
    ("1 결제: 상향 없음", "JT-ESC-001", "결제", ("상향 경로가 없습니다",)),
    ("1 결제: 롤백 승인자 없음", "JT-PATH-020", "결제", ("승인자가 적혀 있지 않습니다",)),
    ("1 결제: 연락 수단 없음", "JT-PATH-030", "결제", ("연락 수단",)),
    ("2 주문: 상향 순환", "JT-ESC-002", "주문", ("'정민혁' → '최영훈' → '정민혁'",)),
    (
        "3 상품 카탈로그: 원복 승인자 없음",
        "JT-PATH-020",
        "상품 카탈로그",
        ("롤백", "승인자가 적혀 있지 않습니다"),
    ),
    ("3 상품 카탈로그: '이상하면'뿐인 감지", "JT-PATH-001", "상품 카탈로그", ("감지",)),
    ("5 보안: 48시간 신고 시한", "JT-DDL-002", None, ("48시간", "24시간")),
    ("6 보안: 개인정보 72시간 신고 단계 없음", "JT-KR-003", None, ("72시간",)),
    ("7 카탈로그 공백: 배송", "JT-CAT-001", "배송", ()),
    ("7 카탈로그 공백: 배송비 견적", "JT-CAT-001", "배송비 견적", ()),
    ("7 카탈로그 공백: 정산", "JT-CAT-001", "정산", ()),
    ("7 카탈로그 공백: 이상거래 탐지", "JT-CAT-001", "이상거래 탐지", ()),
    ("8 당직: 강현우 3일 연속 야간", "JT-ONC-001", None, ("강현우", "3일 연속")),
    ("8 당직: 김서연 휴식 9시간", "JT-ONC-002", None, ("김서연", "9시간")),
    ("8 당직: 퇴사자 김민준 배정", "JT-ONC-004", None, ("김민준", "퇴사")),
)


@pytest.mark.parametrize(("label", "rule_id", "service", "words"), INTENDED, ids=[row[0] for row in INTENDED])
def test_every_planted_defect_is_found_by_the_right_rule(
    run: DemoRun, label: str, rule_id: str, service: str | None, words: tuple[str, ...]
) -> None:
    hits = found(run, rule_id, service)
    assert hits, f"{label}: {rule_id} 없음"
    assert any(f.status is FindingStatus.BROKEN and all(w in f.message for w in words) for f in hits), [
        f.message for f in hits
    ]


def test_catalog_service_approval_gap_is_the_rollback_not_the_flag(run: DemoRun) -> None:
    """'기능 플래그 끄고 … 원복' — the flag switch needs no sign-off, the rollback (원복) does."""
    catalog = sid(run, "상품 카탈로그")
    actions = {a.name: a for a in run.ctx.mitigations(catalog)}
    assert {name: run.ctx.requires_approval(a) for name, a in actions.items()} == {
        "상품 카탈로그 기능 플래그 OFF": False,
        "상품 카탈로그 롤백": True,
    }
    [gap] = found(run, "JT-PATH-020", "상품 카탈로그")
    assert actions["상품 카탈로그 롤백"].id in gap.subject_ids
    assert evaluate_service(run.ctx, catalog)[Element.APPROVER].score == 0.0


def test_security_procedure_kisa_step_is_org_wide_so_kr001_stays_quiet(run: DemoRun) -> None:
    """The 03 runbook names no service, yet its KISA step binds 결제·주문: the gap is the 48h
    deadline (JT-DDL-002), not a missing step (JT-KR-001)."""
    kisa = "obligation:kisa-침해사고-신고"
    assert run.ctx.is_org_obligation(kisa)
    for name in ("결제", "주문"):
        assert kisa in [o.id for o in run.ctx.obligations(sid(run, name))]
    assert not [f for f in run.findings if f.rule_id == "JT-KR-001"]
    [ddl] = found(run, "JT-DDL-002", None)
    assert ddl.subject_ids == (kisa,)
    assert ddl.severity is Severity.CRITICAL


def test_messenger_dies_with_the_frontend_but_the_phone_fallback_keeps_it_reachable(run: DemoRun) -> None:
    """Decision (README row 4): JT-CHN-001 fires only when *every* channel dies with the service.
    이수진 has '사내 메신저(1순위) → 안 되면 전화', so the frontend is not flagged."""
    ctx, front = run.ctx, sid(run, "웹 프론트엔드")
    messenger = ctx.graph.nodes["channel:internal_messenger"]
    assert messenger.name == "사내 메신저"
    assert {p.source for p in messenger.prov} >= {SourceKind.ROSTER, SourceKind.CATALOG}  # one node
    assert [
        c.id for c in ctx.graph.nodes_of(NodeKind.CHANNEL) if c.attr("medium") == "internal_messenger"
    ] == [messenger.id]
    assert ctx.shares_failure_domain(messenger.id, front)
    assert not ctx.shares_failure_domain("channel:phone", front)
    assert not found(run, "JT-CHN-001", "웹 프론트엔드")
    assert evaluate_service(ctx, front)[Element.FALLBACK_CHANNEL].score == 1.0
    idc = result(run, f"tpl:{front}:location_down")
    assert idc.outcome is Outcome.PASS
    lost = [e for e in idc.events if e.kind is SimEventKind.CHANNEL_UNAVAILABLE]
    assert [(e.actor_id, e.target_id) for e in lost] == [(messenger.id, "person:이수진")]
    assert "상암 IDC" in lost[0].note
    assert any(e.kind is SimEventKind.PAGE and e.actor_id == "channel:phone" for e in idc.events)


def test_prompt_injection_is_quarantined_and_builds_nothing(run: DemoRun) -> None:
    [doc] = [d for d in run.bundle.documents if d.path == SECURITY_RUNBOOK]
    [notice] = run.notices
    assert notice.doc_id == doc.doc_id
    assert doc.quarantined == ((notice.start, notice.end),)
    assert "100점" in doc.text[notice.start : notice.end]

    def overlaps(start: int, end: int) -> bool:
        return start < notice.end and notice.start < end

    graph = run.graph
    items = [*graph.nodes.values(), *graph.edges.values()]
    spans = [s for item in items for s in item.spans]
    spans += [s for item in items for attr in item.attrs.values() for s in attr.spans]
    assert spans  # the graph is grounded at all
    assert not [s for s in spans if s.doc_id == doc.doc_id and overlaps(s.start, s.end)]
    assert not [n for n in graph.nodes.values() if "100점" in n.name or "AI" in n.name]
    assert run.assembled.grounding.dropped_quarantined == 0  # nothing was even proposed from it


def test_injection_changes_no_verdict_and_no_score(run: DemoRun, tmp_path: Path) -> None:
    """ADR-0004: the same bundle without the hidden comment gives the same findings and scores."""
    clean = tmp_path / "dmc-shop"
    shutil.copytree(DEMO, clean)
    runbook = clean / SECURITY_RUNBOOK
    text = runbook.read_text(encoding="utf-8")
    start, end = text.index("<!--"), text.index("-->") + len("-->")
    runbook.write_text(text[:start] + " " * (end - start) + text[end:], encoding="utf-8")
    without = pipeline(clean)
    assert not without.notices
    assert [(f.rule_id, f.service_id, f.status, f.message) for f in without.findings] == [
        (f.rule_id, f.service_id, f.status, f.message) for f in run.findings
    ]
    assert [(s.service_id, s.score) for s in without.readiness.services] == [
        (s.service_id, s.score) for s in run.readiness.services
    ]
    assert without.readiness.overall == run.readiness.overall


# --------------------------------------------------------------------------- the healthy control


def test_cart_control_is_healthy(run: DemoRun) -> None:
    cart = sid(run, "장바구니")
    [readiness] = [s for s in run.readiness.services if s.service_id == cart]
    assert readiness.score >= 90
    assert all(e.status is not ElementStatus.FAIL for e in readiness.elements)
    serious = [
        f for f in run.findings if f.service_id == cart and f.severity in (Severity.HIGH, Severity.CRITICAL)
    ]
    assert serious == []
    for key in ("baseline", "owner_absent", "channel_down", "location_down", "concurrent"):
        assert result(run, f"tpl:{cart}:{key}").outcome is Outcome.PASS, key


def test_broken_services_score_far_below_the_control(run: DemoRun) -> None:
    scores = {s.service_name: s.score for s in run.readiness.services}
    assert scores["결제"] < 50
    assert scores["상품 카탈로그"] < 60
    for name in ("배송", "정산", "이상거래 탐지", "배송비 견적"):
        assert scores[name] == 0.0
    for name in ("웹 프론트엔드", "주문", "환율", "광고", "추천"):
        assert scores[name] >= 90


# --------------------------------------------------------------------------- integrity


def test_no_hold_no_invented_people_and_every_runbook_fact_grounded(run: DemoRun) -> None:
    assert run.verified.held_ids == ()
    assert run.assembled.held_ids == ()
    assert run.assembled.unresolved_people == ()
    assert run.assembled.grounding.dropped_ungrounded == 0
    roster = {r.name for r in run.bundle.roster}
    assert {p.name for p in run.graph.nodes_of(NodeKind.PERSON)} <= roster  # P2: people from the roster
    for item in [*run.graph.nodes.values(), *run.graph.edges.values()]:
        for prov in item.prov:
            assert prov.source is not SourceKind.RUNBOOK or prov.spans, item.id
    assert not [f for f in run.findings if f.rule_id.startswith("JT-INT")]


def test_sangam_idc_blast_radius(run: DemoRun) -> None:
    impact = blast_radius(run.graph, ["location:상암-idc"])
    names = {run.graph.nodes[s].name for s in impact.impacted_services}
    assert names >= {"상품 카탈로그", "주문", "정산", "이상거래 탐지", "웹 프론트엔드", "추천"}
    assert not names & {"장바구니", "결제", "환율"}
    assert impact.impacted_channels == ("channel:internal_messenger",)


# --------------------------------------------------------------------------- rehearsals


def test_human_scenarios_fail_where_the_readme_says(run: DemoRun) -> None:
    s1 = result(run, "human:dmc-s1-chuseok-payment")
    assert (s1.outcome, s1.failed_element, s1.failed_at_id) == (Outcome.FAIL, Element.OWNER, "person:김민준")
    s2 = result(run, "human:dmc-s2-sangam-idc-fire")
    assert s2.outcome is Outcome.FAIL
    s3 = result(run, "human:dmc-s3-slack-outage-order")
    assert (s3.outcome, s3.failed_element, s3.failed_at_id) == (Outcome.FAIL, Element.OWNER, "person:최영훈")
    assert "상향 순환" in s3.reason
    s4 = result(run, "human:dmc-s4-breach-report")
    assert s4.outcome is Outcome.FAIL
    assert s4.deadline_minutes == 1440
    missed = [e for e in s4.events if e.kind is SimEventKind.REPORT_MISSED]
    assert [e.target_id for e in missed] == ["obligation:kisa-침해사고-신고"]
    origins = {s.origin for s in run.scenarios}
    assert origins == {ScenarioOrigin.TEMPLATE, ScenarioOrigin.HUMAN}


# --------------------------------------------------------------------------- reproducibility


def test_a_second_run_is_identical(run: DemoRun) -> None:
    again = pipeline()
    assert again.graph.content_hash() == run.graph.content_hash()
    assert [f.model_dump(mode="json") for f in again.findings] == [
        f.model_dump(mode="json") for f in run.findings
    ]
    assert again.readiness == run.readiness
    assert [r.result_hash() for r in again.results] == [r.result_hash() for r in run.results]
    assert digest(again) == digest(run)


@pytest.mark.parametrize("seed", ["0", "4242"])
def test_digest_does_not_depend_on_hash_seed(run: DemoRun, seed: str) -> None:
    """Set/dict iteration order varies with PYTHONHASHSEED; the published results must not."""
    code = (
        CHILD_PRELUDE
        + "import runpy\n"
        + f"ns = runpy.run_path({str(Path(__file__))!r})\n"
        + "sys.stdout.write(ns['digest'](ns['pipeline']()))\n"
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
    assert out.stdout.strip() == digest(run)


# --------------------------------------------------------------------------- published docs


@pytest.fixture(scope="module")
def gen_docs() -> ModuleType:
    """``scripts/gen_docs.py`` loaded as a module (scripts are not a package)."""
    spec = importlib.util.spec_from_file_location("gen_docs", ROOT / "scripts" / "gen_docs.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["gen_docs"] = module  # dataclasses resolve their module while it executes
    spec.loader.exec_module(module)
    return module


def test_published_docs_match_the_code_and_the_demo(run: DemoRun, gen_docs: ModuleType) -> None:
    """docs/scoring.md, docs/rules.md and docs/demo-baseline.md are generated (scripts/gen_docs.py);
    a stale copy would tell customers something the judge no longer does."""
    demo = gen_docs.DemoRun(
        graph=run.graph,
        critical_tiers=tuple(sorted(run.bundle.policy.critical_tiers)),
        held_ids=tuple(sorted({*run.assembled.held_ids, *run.verified.held_ids})),
        notices=run.notices,
        findings=run.findings,
        readiness=run.readiness,
        scenarios=run.scenarios,
        results=run.results,
    )
    baseline = gen_docs.demo_baseline_doc(demo)
    print(f"\n{baseline}")  # the compact Korean summary (pytest -s)
    docs = ROOT / "docs"
    assert (docs / "demo-baseline.md").read_text(encoding="utf-8") == baseline
    assert (docs / "scoring.md").read_text(encoding="utf-8") == gen_docs.scoring_doc()
    assert (docs / "rules.md").read_text(encoding="utf-8") == gen_docs.rules_doc()
    assert gen_docs.main(["--check"]) == 0  # the script's own pipeline agrees with this one


def test_rules_doc_lists_every_rule_once(gen_docs: ModuleType) -> None:
    text = gen_docs.rules_doc()
    catalog = rule_catalog()
    for info in catalog:
        assert text.count(f"| `{info.rule_id}` |") == 1, info.rule_id
    assert f"규칙 {len(catalog)}개" in text
    assert "JT-KR-001" in text
    assert "정보통신망법 제48조의3" in text  # legal basis column
