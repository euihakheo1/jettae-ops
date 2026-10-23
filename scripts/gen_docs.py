"""Generate the docs that are rendered from code, so the published text can never drift from it.

* ``docs/scoring.md``       — the readiness formula (SCORING_V1) and every element's verdict rule,
  straight from :func:`jettae.check.scoring.scoring_table_markdown` (which embeds ``SEMANTICS_KO``).
* ``docs/rules.md``         — the fixed-rule catalog (:func:`jettae.check.registry.rule_catalog`).
* ``docs/demo-baseline.md`` — the finals demo (``examples/dmc-shop``) run with the LLM off:
  per-service scores, organisation-level findings and rehearsal outcomes.

Why generated: customers audit the judge through these pages (P1). A hand-written copy of a rule
title or a weight would sooner or later say something the code does not do.

Usage (from the project root)::

    python scripts/gen_docs.py           # write the three files
    python scripts/gen_docs.py --check   # exit 1 if any file is out of date (CI)

Everything here is deterministic (P6): no timestamps, sorted output, LLM off.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from jettae.check.context import CheckContext
from jettae.check.registry import rule_catalog, run_checks
from jettae.check.scoring import score as readiness_score
from jettae.check.scoring import scoring_table_markdown
from jettae.domain import (
    ELEMENT_LABEL_KO,
    Element,
    ElementStatus,
    Finding,
    FindingStatus,
    OpsGraph,
    Outcome,
    ReadinessReport,
    RehearsalResult,
    RuleInfo,
    Scenario,
    ScenarioOrigin,
    SecurityNotice,
    Severity,
)
from jettae.extract import RuleExtractor, assemble, verify_backtranslation
from jettae.ingest import load_bundle_dir, scan_injection
from jettae.rehearse import replay_all, scenarios_from_yaml, template_scenarios

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DEMO_DIR = ROOT / "examples" / "dmc-shop"

GENERATED_NOTE = (
    "<!-- 이 파일은 scripts/gen_docs.py가 코드에서 생성합니다. 직접 고치지 말고 다시 생성하세요. -->"
)

#: Same words as the evidence report, so the docs and the report read alike.
SEVERITY_KO: dict[Severity, str] = {
    Severity.CRITICAL: "심각",
    Severity.HIGH: "높음",
    Severity.MEDIUM: "보통",
    Severity.LOW: "낮음",
    Severity.INFO: "정보",
}
SEVERITY_RANK: dict[Severity, int] = {s: i for i, s in enumerate(SEVERITY_KO)}

#: Rule id families (ARCHITECTURE §4.4) in the order the catalog explains them.
RULE_FAMILIES: tuple[tuple[str, str], ...] = (
    ("JT-PATH", "필수 대응 경로(감지·담당·승인자·대체 채널·롤백)"),
    ("JT-PPL", "사람(퇴사·부재·명부 미확인)"),
    ("JT-ESC", "상향(에스컬레이션)"),
    ("JT-CHN", "연락 채널"),
    ("JT-DDL", "시한"),
    ("JT-KR", "국내 신고 기준"),
    ("JT-ONC", "당직표"),
    ("JT-CAT", "서비스 카탈로그 대조"),
    ("JT-SPF", "단일 장애점"),
    ("JT-INT", "판단 보류·무결성"),
)

ORIGIN_KO: dict[ScenarioOrigin, str] = {
    ScenarioOrigin.HUMAN: "사람",
    ScenarioOrigin.TEMPLATE: "템플릿",
    ScenarioOrigin.LLM: "AI 제안",
}

#: Hash prefixes shown in the docs: long enough to compare by eye, short enough to read.
HASH_CHARS = 12


def _cell(text: str) -> str:
    """One Markdown table cell: no line breaks, literal pipes escaped."""
    return " ".join(text.split()).replace("|", "\\|")


def _family(rule_id: str) -> str:
    return rule_id.rsplit("-", 1)[0]


# --------------------------------------------------------------------------- scoring.md


def scoring_doc() -> str:
    return f"{GENERATED_NOTE}\n\n{scoring_table_markdown()}"


# --------------------------------------------------------------------------- rules.md


def rules_doc(catalog: Sequence[RuleInfo] | None = None) -> str:
    """The rule catalog as one table (id, 제목, 심각도, 요소, 근거, 법적 근거), sorted by id."""
    infos = sorted(rule_catalog() if catalog is None else catalog, key=lambda r: r.rule_id)
    counts = Counter(_family(r.rule_id) for r in infos)
    known = {prefix for prefix, _ in RULE_FAMILIES}
    lines = [
        GENERATED_NOTE,
        "",
        "# 고정 규칙 목록",
        "",
        "> 끊김 판정은 아래 고정 규칙으로만 하며, 판정에 생성형 AI를 쓰지 않습니다.",
        "> LLM을 켠 진단에서는 역번역 대조의 AI 판정이 연결을 판단 보류로 돌릴 수 있고, "
        "그 연결은 JT-INT-001로 알립니다. "
        "AI가 끊김을 확정하거나 연결을 추가·수정하지는 못합니다.",
        "> 같은 그래프에는 언제나 같은 판정이 나옵니다. 점수 산식은 [scoring.md](scoring.md)에 있습니다.",
        "",
        f"규칙 {len(infos)}개. '요소'는 그 규칙이 점수의 어느 필수 경로 요소와 맞물리는지를 뜻합니다"
        "('—'는 조직 단위 점검).",
        "",
        "| 계열 | 다루는 것 | 규칙 수 |",
        "|---|---|---:|",
    ]
    lines += [
        f"| `{prefix}` | {topic} | {counts[prefix]} |" for prefix, topic in RULE_FAMILIES if counts[prefix]
    ]
    lines += [f"| `{prefix}` | 기타 | {counts[prefix]} |" for prefix in sorted(set(counts) - known)]
    lines += [
        "",
        "| ID | 제목 | 심각도 | 요소 | 근거 | 법적 근거 |",
        "|---|---|---|---|---|---|",
    ]
    for info in infos:
        element = ELEMENT_LABEL_KO[info.element] if info.element is not None else "—"
        legal = _cell(info.legal_basis) if info.legal_basis else "—"
        lines.append(
            f"| `{info.rule_id}` | {_cell(info.title)} | {SEVERITY_KO[info.severity]} | {element} | "
            f"{_cell(info.rationale)} | {legal} |"
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- the demo run


@dataclass(frozen=True)
class DemoRun:
    """Everything the LLM-off pipeline produced for one bundle."""

    graph: OpsGraph
    critical_tiers: tuple[int, ...]
    held_ids: tuple[str, ...]
    notices: tuple[SecurityNotice, ...]
    findings: tuple[Finding, ...]
    readiness: ReadinessReport
    scenarios: tuple[Scenario, ...]
    results: tuple[RehearsalResult, ...]


def run_demo(bundle_dir: Path = DEMO_DIR) -> DemoRun:
    """ingest → rule parser → assemble → back-translation (LLM off) → rules → score → rehearsal."""
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
    verified = asyncio.run(verify_backtranslation(assembled.graph, bundle.documents, None))
    graph = verified.graph
    ctx = CheckContext(graph, bundle.policy, bundle.roster, bundle.oncall, bundle.catalog)
    findings = run_checks(ctx)
    scenarios = [*template_scenarios(ctx), *scenarios_from_yaml(bundle_dir / "scenarios.yaml", ctx)]
    results = replay_all(graph, scenarios, policy=bundle.policy, roster=bundle.roster)
    return DemoRun(
        graph=graph,
        critical_tiers=tuple(sorted(bundle.policy.critical_tiers)),
        held_ids=tuple(sorted({*assembled.held_ids, *verified.held_ids})),
        notices=notices,
        findings=tuple(findings),
        readiness=readiness_score(ctx, findings),
        scenarios=tuple(scenarios),
        results=tuple(results),
    )


def results_digest(results: Iterable[RehearsalResult]) -> str:
    """One hash over every rehearsal result hash (in scenario order) — the P6 fingerprint."""
    return hashlib.sha256("\n".join(r.result_hash() for r in results).encode()).hexdigest()


def _element_cell(score: float, status: ElementStatus) -> str:
    text = f"{score:g}"
    return f"{text} (확인 불가)" if status is ElementStatus.UNKNOWN else text


def _severity_counts(findings: Iterable[Finding]) -> str:
    counts = Counter(f.severity for f in findings)
    parts = [f"{SEVERITY_KO[s]} {counts[s]}" for s in sorted(counts, key=SEVERITY_RANK.__getitem__)]
    return " · ".join(parts) if parts else "—"


def demo_baseline_doc(run: DemoRun) -> str:
    """A compact Korean summary of the demo run: scores per service, org findings, rehearsals."""
    graph, report = run.graph, run.readiness
    unknown = sum(f.status is FindingStatus.UNKNOWN for f in run.findings)
    critical_tiers = "·".join(str(t) for t in run.critical_tiers)
    lines = [
        GENERATED_NOTE,
        "",
        "# dmc-shop 데모 기준선 (LLM 끔)",
        "",
        "> 결선 예제 `examples/dmc-shop`을 규칙 파서(LLM 0회) → 조립 → 역번역 대조 → 고정 규칙 → 점수 → "
        "리허설 순서로 돌린 결과입니다.",
        "> 같은 입력이면 언제 돌려도 아래 값이 그대로 나옵니다(그래프 해시·리허설 결과 해시로 확인).",
        "",
        f"- 운영 그래프: 노드 {len(graph.nodes)}개 · 연결 {len(graph.edges)}개 · "
        f"해시 `{graph.content_hash()[:HASH_CHARS]}`",
        f"- 판단 보류: {len(run.held_ids)}건 · 격리된 문서 구간(프롬프트 인젝션 의심): {len(run.notices)}곳",
        f"- 끊김 {len(run.findings)}건: {_severity_counts(run.findings)} (그중 확인 불가 {unknown}건)",
        f"- 전체 준비도: **{report.overall:g}점** (핵심 등급 {critical_tiers} 서비스의 등급 가중 평균, "
        f"SCORING_{report.weights_version.upper()})",
        "",
        "## 서비스별 점수",
        "",
        "| 서비스 | 등급 | 점수 | " + " | ".join(ELEMENT_LABEL_KO[e] for e in Element) + " | 서비스 끊김 |",
        "|---|---:|---:|" + "---:|" * len(Element) + "---|",
    ]
    by_service: dict[str, list[Finding]] = {}
    for f in run.findings:
        if f.service_id is not None:
            by_service.setdefault(f.service_id, []).append(f)
    for svc in sorted(report.services, key=lambda s: (s.tier, -s.score, s.service_name)):
        elements = {e.element: e for e in svc.elements}
        cells = " | ".join(_element_cell(elements[e].score, elements[e].status) for e in Element)
        lines.append(
            f"| {svc.service_name} | {svc.tier} | {svc.score:g} | {cells} | "
            f"{_severity_counts(by_service.get(svc.service_id, []))} |"
        )
    org = sorted(
        (f for f in run.findings if f.service_id is None),
        key=lambda f: (SEVERITY_RANK[f.severity], f.rule_id, f.id),
    )
    lines += [
        "",
        "## 조직 단위 끊김 (한 서비스에 속하지 않음)",
        "",
        "| 규칙 | 심각도 | 내용 |",
        "|---|---|---|",
    ]
    lines += [f"| `{f.rule_id}` | {SEVERITY_KO[f.severity]} | {_cell(f.message)} |" for f in org]
    lines += _rehearsal_section(run)
    return "\n".join(lines) + "\n"


def _rehearsal_section(run: DemoRun) -> list[str]:
    graph = run.graph
    pairs = list(zip(run.scenarios, run.results, strict=True))
    lines = ["", "## 모의 장애 리허설", "", "| 시나리오 | 출처 | 결과 | 처음 끊긴 곳 |", "|---|---|---|---|"]
    for scenario, result in pairs:
        if scenario.origin is ScenarioOrigin.TEMPLATE:
            continue
        lines.append(
            f"| `{scenario.id}` {_cell(scenario.title)} | {ORIGIN_KO[scenario.origin]} | "
            f"{_outcome_ko(result)} | {_failure_point(graph, result)} |"
        )
    templates = [r for s, r in pairs if s.origin is ScenarioOrigin.TEMPLATE]
    passed = sum(r.outcome is Outcome.PASS for r in templates)
    lines += [
        "",
        f"템플릿 시나리오 {len(templates)}개(핵심 서비스별 기준·담당 부재·주 채널 다운·승인자 부재·센터 장애·"
        f"동시 장애·침해사고 신고): 통과 {passed} · 실패 {len(templates) - passed}",
        "",
        f"리허설 결과 해시 묶음(전체 {len(pairs)}개, 시나리오 순서): "
        f"`{results_digest(run.results)[:HASH_CHARS]}`",
    ]
    return lines


def _outcome_ko(result: RehearsalResult) -> str:
    return "통과" if result.outcome is Outcome.PASS else "실패"


def _failure_point(graph: OpsGraph, result: RehearsalResult) -> str:
    if result.outcome is Outcome.PASS:
        return "—"
    element = ELEMENT_LABEL_KO[result.failed_element] if result.failed_element is not None else "신고"
    node = graph.nodes.get(result.failed_at_id or "")
    return f"{element} ({_cell(node.name)})" if node is not None else element


# --------------------------------------------------------------------------- main


def targets(run: Callable[[], DemoRun] = run_demo) -> dict[Path, Callable[[], str]]:
    """Output path → renderer. The demo pipeline runs at most once, and only when needed."""
    cache: list[DemoRun] = []

    def demo() -> str:
        if not cache:
            cache.append(run())
        return demo_baseline_doc(cache[0])

    return {DOCS / "scoring.md": scoring_doc, DOCS / "rules.md": rules_doc, DOCS / "demo-baseline.md": demo}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="코드에서 생성하는 문서(점수 산식·규칙 목록·데모 기준선)를 만듭니다."
    )
    parser.add_argument("--check", action="store_true", help="파일을 쓰지 않고, 최신이 아니면 1로 끝냅니다.")
    args = parser.parse_args(argv)
    stale: list[str] = []
    for path, render in targets().items():
        text = render()
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        name = path.relative_to(ROOT).as_posix()
        if current == text:
            print(f"최신: {name}")
        elif args.check:
            stale.append(name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="\n")
            print(f"생성: {name}")
    if stale:
        print(f"최신이 아닌 문서: {', '.join(stale)}. python scripts/gen_docs.py 로 다시 생성하세요.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
