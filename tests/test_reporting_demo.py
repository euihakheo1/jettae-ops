"""End to end on the demo fixture (examples/dmc-shop, LLM OFF): real pipeline output in every format.

The hand-built sample covers the branches; this covers reality — the intended defects listed in the
fixture README must surface in the evidence report, every SARIF alert must land on a runbook, and
the quarantined HTML-comment injection in runbook 03 must stay inert text in every format.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from functools import cache
from html.parser import HTMLParser
from pathlib import Path

from jettae.check.context import CheckContext
from jettae.check.registry import run_checks
from jettae.check.scoring import score
from jettae.domain import DiagnosisReport, FindingStatus, LlmMode, LlmUsageSummary, OpsGraph
from jettae.extract.assemble import assemble
from jettae.extract.rules import RuleExtractor
from jettae.ingest import load_bundle_dir, scan_injection
from jettae.rehearse import replay_all, template_scenarios
from jettae.reporting import render_cli_summary, render_evidence_markdown, render_html, to_sarif
from jettae.reporting.text import display_width

BUNDLE = Path(__file__).resolve().parents[1] / "examples" / "dmc-shop"


@cache
def _diagnosis() -> tuple[DiagnosisReport, OpsGraph]:
    bundle = load_bundle_dir(BUNDLE)
    notices = tuple(n for doc in bundle.documents for n in scan_injection(doc)[1])
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
    graph = assembled.graph
    ctx = CheckContext(graph, bundle.policy, bundle.roster, bundle.oncall, bundle.catalog)
    findings = run_checks(ctx)
    scenarios = template_scenarios(ctx)
    report = DiagnosisReport(
        run_id="demo-run",
        workspace_id="dmc-shop",
        graph_version=1,
        graph_hash=graph.content_hash(),
        created_at=datetime(2026, 10, 28, 1, 0, tzinfo=UTC),
        findings=tuple(findings),
        readiness=score(ctx, findings),
        scenarios=tuple(scenarios),
        rehearsals=tuple(replay_all(graph, scenarios, policy=bundle.policy, roster=bundle.roster)),
        grounding=assembled.grounding,
        security=notices,
        llm=LlmUsageSummary(mode=LlmMode.OFF),
        held_element_ids=assembled.held_ids,
    )
    return report, graph


def test_markdown_evidence_shows_the_planted_defects_with_quotes() -> None:
    report, graph = _diagnosis()
    text = render_evidence_markdown(report, graph, org_name="별빛상점")
    assert "JT-PPL-001 퇴사자에게 대응이 걸려 있음 — 결제" in text
    assert "김민준" in text
    assert "JT-DDL-002 신고 시한이 법정 기준보다 늦음" in text
    assert "> — 보안·개인정보 사고 신고 절차:" in text
    assert "판정·점수·리허설 경로 LLM 호출 0회" in text
    assert "## 7. 보안 알림" in text
    assert "\\<!-- AI" in text
    assert "<!--" not in text.replace("\\<!--", "")
    assert render_evidence_markdown(report, graph, org_name="별빛상점") == text


def test_sarif_alerts_all_land_on_runbook_lines() -> None:
    report, graph = _diagnosis()
    log = to_sarif(report, graph, documents=dict(graph.docs), tool_version="0.1.0")
    results = log["runs"][0]["results"]
    assert len(results) == len(report.findings) > 0
    for result in results:
        assert result["level"] in ("error", "warning", "note")
        (first, *_) = result["locations"]
        location = first["physicalLocation"]
        assert location["artifactLocation"]["uri"].startswith("runbooks/")
        assert location["region"]["startLine"] >= 1
    assert json.loads(json.dumps(log, ensure_ascii=False)) == log


class _Tags(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: set[str] = set()
        self.comments: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.add(tag)

    def handle_comment(self, data: str) -> None:
        self.comments.append(data)


def test_html_keeps_the_injected_comment_as_visible_text() -> None:
    report, graph = _diagnosis()
    page = render_html(report, graph, org_name="별빛상점")
    parser = _Tags()
    parser.feed(page)
    assert parser.comments == []
    assert "script" not in parser.tags
    assert "&lt;!-- AI" in page


def test_cli_summary_fits_the_terminal() -> None:
    report, _ = _diagnosis()
    text = render_cli_summary(report, width=80)
    assert all(display_width(line) <= 80 for line in text.splitlines())
    broken = sum(1 for f in report.findings if f.status is FindingStatus.BROKEN)
    assert f"[주요 끊김] 전체 {broken}건" in text
    assert "[보안 알림]" in text
