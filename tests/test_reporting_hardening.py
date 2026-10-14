"""Second independent review of ``jettae.reporting``: regressions for the defects it fixed, plus a
property test that feeds *any* valid diagnosis to every renderer.

Defects pinned here: quadratic clipping and wrapping in the CLI (a long catalog service name
stalled the summary for minutes), before/after claims that contradicted the reported result,
pending forecasts counted as ISMS-P 2.11.4 before/after evidence, comparisons across different
drill definitions shown as the same drill, "(중단)" on passing replays, version labels newer than
the report, blank values rendered as two literal backticks, and duplicate SARIF rule tags.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from html.parser import HTMLParser

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from markdown_it import MarkdownIt

import jettae.reporting.text as text_module
from jettae.domain import (
    DiagnosisReport,
    DocumentRef,
    Element,
    ElementScore,
    ElementStatus,
    Finding,
    FindingStatus,
    FixHint,
    FixKind,
    GraphVersion,
    GroundingStatus,
    IncidentType,
    LlmMode,
    LlmUsageSummary,
    OpsGraph,
    Outcome,
    Patch,
    PatchAuthor,
    PatchStatus,
    PatchValidation,
    ReadinessReport,
    RehearsalResult,
    RuleInfo,
    Scenario,
    SecurityNotice,
    ServiceReadiness,
    Severity,
    SimEvent,
    SimEventKind,
    SourceSpan,
)
from jettae.reporting import (
    build_evidence,
    render_cli_summary,
    render_evidence_markdown,
    render_html,
    to_sarif,
)
from jettae.reporting.evidence import CHANGED_DRILL_NOTE, RehearsalRow, current_rehearsals
from jettae.reporting.markdown import code_span
from jettae.reporting.sarif import _rule_descriptor
from jettae.reporting.text import display_width, wrap_display
from test_reporting_sample import (
    GRAPH_HASH,
    OLD_HASH,
    P_APPROVED,
    P_DRAFT,
    REHEARSALS,
    SC_BASE,
    VERSIONS,
    sample_graph,
    sample_report,
)

_MD = MarkdownIt("commonmark").enable("table").enable("strikethrough")


# --------------------------------------------------------------------------- CLI: linear time


def _counting_display_width(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the length of every string the CLI measures (pins linear-time clipping/wrapping)."""
    measured: list[int] = []
    original = text_module.display_width

    def counting(text: str) -> int:
        measured.append(len(text))
        return original(text)

    monkeypatch.setattr(text_module, "display_width", counting)
    return measured


def test_clipping_a_huge_name_measures_it_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defect: ``_fit`` dropped one character at a time and re-measured the rest, so a catalog
    service name of 16 000 characters took ~54 s per summary row (quadratic)."""
    name = "결제" * 50_000
    measured = _counting_display_width(monkeypatch)
    fitted = text_module._fit(name, 16)
    assert sum(measured) <= len(name) + 16
    assert fitted == "결제" * 3 + "결" + "…"
    assert display_width(fitted) <= 16


def test_wrapping_a_huge_word_is_linear(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defect: every cut of an over-long word re-measured the whole remainder (quadratic)."""
    word = "가" * 100_000
    measured = _counting_display_width(monkeypatch)
    lines = wrap_display(word, 60, first="  ", rest="  ")
    assert sum(measured) <= 4 * len(word)
    assert all(display_width(line) <= 60 for line in lines)
    assert "".join(line.strip() for line in lines) == word


def test_summary_with_a_huge_service_name_still_fits() -> None:
    report = sample_report()
    long_name = "정산" * 100_000
    services = tuple(s.model_copy(update={"service_name": long_name}) for s in report.readiness.services)
    huge = report.model_copy(update={"readiness": report.readiness.model_copy(update={"services": services})})
    lines = render_cli_summary(huge, width=40).splitlines()
    assert all(display_width(line) <= 40 for line in lines)
    assert any("정산정산" in line and "…" in line for line in lines)


# --------------------------------------------------------------------------- before/after claims


def _single(result: RehearsalResult, patches: tuple[Patch, ...] = ()) -> RehearsalRow:
    report = sample_report(rehearsals=(result,), patches=patches)
    (row,) = build_evidence(report, sample_graph()).rehearsals
    return row


def test_an_approved_fix_is_not_cited_next_to_a_failing_result() -> None:
    """Defect: without an earlier replay the comparison came from patch validation alone, so a
    scenario failing on the reported graph read "실패 → 통과 (수정안 patch-1 반영)" — a fix that
    does not hold (e.g. reverted since) presented to an auditor as working."""
    failing = REHEARSALS[0].model_copy(update={"outcome": Outcome.FAIL})
    row = _single(failing, (P_APPROVED,))
    assert row.comparison is None
    assert not row.compared
    passing = _single(REHEARSALS[0], (P_APPROVED,))
    assert passing.comparison == "실패 → 통과 (수정안 patch-1 반영)"
    assert passing.compared


def test_a_pending_forecast_is_only_shown_for_a_failing_scenario() -> None:
    breach = next(r for r in REHEARSALS if r.scenario_id == "sc-breach")
    assert _single(breach, (P_DRAFT,)).comparison == "승인 시 실패 → 통과 예상 (수정안 patch-2)"
    passing = breach.model_copy(update={"outcome": Outcome.PASS})
    assert _single(passing, (P_DRAFT,)).comparison is None


def test_an_approved_regression_is_shown_only_while_the_scenario_fails() -> None:
    regressed = PatchValidation(passed=False, rehearsals_regressed=("sc-base",))
    regressing = P_APPROVED.model_copy(update={"validation": regressed})
    failing = REHEARSALS[0].model_copy(update={"outcome": Outcome.FAIL})
    row = _single(failing, (regressing,))
    assert row.comparison == "통과 → 실패 (수정안 patch-1 반영)"
    assert row.compared
    assert _single(REHEARSALS[0], (regressing,)).comparison is None


def test_pending_forecasts_are_not_counted_as_before_after_evidence() -> None:
    """Defect: ISMS-P 2.11.4 counted "승인 시 … 예상" forecasts as 수정 전후 비교."""
    breach = next(r for r in REHEARSALS if r.scenario_id == "sc-breach")
    view = build_evidence(sample_report(rehearsals=(breach,), patches=(P_DRAFT,)), sample_graph())
    (row,) = view.rehearsals
    assert row.comparison is not None
    assert not row.compared
    control = next(c for c in view.controls if c.control_id == "2.11.4")
    assert control.evidence.endswith("수정 전후 비교 0건")


def test_before_after_across_different_drill_definitions_is_flagged() -> None:
    """A template keeps its id while its target follows the graph ("1차 담당 부재(새 담당)"), so an
    earlier result may come from another definition; the comparison says so instead of implying
    the same drill was replayed twice."""
    other = REHEARSALS[1].model_copy(update={"scenario_fingerprint": "0" * 16})
    report = sample_report(rehearsals=(REHEARSALS[0], other))
    (row,) = build_evidence(report, sample_graph(), versions=VERSIONS).rehearsals
    assert row.comparison == "v1 실패 → v2 통과" + CHANGED_DRILL_NOTE
    assert row.compared
    note = CHANGED_DRILL_NOTE.strip()
    assert note in render_evidence_markdown(report, sample_graph(), versions=VERSIONS)
    assert note in render_html(report, sample_graph(), versions=VERSIONS)
    same = build_evidence(sample_report(rehearsals=REHEARSALS[:2]), sample_graph(), versions=VERSIONS)
    assert same.rehearsals[0].comparison == "v1 실패 → v2 통과"


def test_a_passing_replay_without_mitigation_time_is_not_called_broken_off() -> None:
    passing = REHEARSALS[0].model_copy(update={"time_to_mitigate": None})
    assert _single(passing).ended_at == "T+5분"
    failing = passing.model_copy(update={"outcome": Outcome.FAIL})
    assert _single(failing).ended_at == "T+5분 (중단)"


def test_before_results_are_never_labelled_newer_than_the_report() -> None:
    """Defect: a hash shared by several versions took the newest number, so with a history newer
    than the report (v3 reverted to v1's graph) the row read "v3 실패 → v2 통과"."""
    revert = VERSIONS[1].model_copy(update={"number": 3, "parent": 2, "revert_of": 2, "note": ""})
    view = build_evidence(sample_report(), sample_graph(), versions=(*VERSIONS, revert))
    base = next(r for r in view.rehearsals if r.scenario_id == SC_BASE.id)
    assert base.comparison == "v1 실패 → v2 통과"


def test_a_report_at_the_head_names_a_recreated_graph_by_its_latest_version() -> None:
    """At the head, the graph a revert recreated is called by the version right before the fix."""
    revert = VERSIONS[1].model_copy(update={"number": 3, "parent": 2, "revert_of": 2, "note": ""})
    head = VERSIONS[0].model_copy(update={"number": 4, "parent": 3, "graph_hash": "5" * 64})
    report = sample_report(
        graph_hash="5" * 64,
        graph_version=4,
        rehearsals=(REHEARSALS[1], REHEARSALS[0].model_copy(update={"graph_hash": "5" * 64})),
    )
    (row,) = build_evidence(report, sample_graph(), versions=(*VERSIONS, revert, head)).rehearsals
    assert row.comparison == "v3 실패 → v4 통과"
    ((current, before),) = current_rehearsals(report, {OLD_HASH: 3, GRAPH_HASH: 2})
    assert current.graph_hash == "5" * 64
    assert before is not None
    assert before.graph_hash == OLD_HASH


# --------------------------------------------------------------------------- blank values, SARIF tags


def test_blank_code_values_show_the_table_dash_in_both_formats() -> None:
    """Defect: a blank value rendered as an empty code span, which GFM prints as two backticks."""
    assert code_span("   ") == ""
    assert code_span(" \x1b ", table=True) == ""
    blank = DocumentRef(doc_id="doc-blank", title="빈 경로", sha256="c" * 64, path="  ")
    markdown = render_evidence_markdown(sample_report(), sample_graph(), documents=(blank,))
    assert "| 빈 경로 | — | `cccccccccccc` |" in markdown
    page = render_html(sample_report(), sample_graph(), documents=(blank,))
    assert "<tr><td>빈 경로</td><td>—</td>" in page
    assert "<code></code>" not in page


def test_sarif_rule_tags_stay_unique_for_any_family_name() -> None:
    """The SARIF schema requires unique tags; a family named like a fixed tag must not break it."""
    info = RuleInfo(
        rule_id="JT-LEGAL-001",
        title="가상 규칙",
        severity=Severity.HIGH,
        element=Element.OWNER,
        rationale="태그 중복 확인",
        legal_basis="가상 법령",
    )
    assert _rule_descriptor(info)["properties"]["tags"] == [
        "jettae",
        "runbook",
        "incident-response",
        "legal",
        "owner",
    ]
    owner_family = info.model_copy(update={"rule_id": "JT-OWNER-001", "legal_basis": None})
    tags = _rule_descriptor(owner_family)["properties"]["tags"]
    assert len(tags) == len(set(tags))


# --------------------------------------------------------------------------- any valid report

_TEXT = st.text(max_size=24)
_IDS = st.sampled_from(["service:a", "service:b", "person:x", "edge:1", "", "sc-1", "doc-1"])
_HASHES = st.sampled_from(["a" * 64, "b" * 64, "c" * 64])
_SCENARIOS = ("sc-1", "sc-2", "sc-3")


@st.composite
def _spans(draw: st.DrawFn) -> SourceSpan:
    start = draw(st.integers(0, 40))
    return SourceSpan(
        doc_id=draw(st.sampled_from(["doc-1", "doc-2", "doc-x"])),
        start=start,
        end=start + draw(st.integers(0, 40)),
        quote=draw(_TEXT),
        line=draw(st.none() | st.integers(1, 10**6)),
    )


@st.composite
def _findings(draw: st.DrawFn) -> Finding:
    rule_id = draw(st.sampled_from(["JT-PATH-001", "JT-KR-001", "JT-DDL-002", "JT-CAT-003", "X"]))
    fix = draw(st.none() | st.sampled_from(FixKind))
    return Finding(
        id=f"{rule_id}:{draw(st.integers(0, 50))}",
        rule_id=rule_id,
        severity=draw(st.sampled_from(Severity)),
        status=draw(st.sampled_from(FindingStatus)),
        title=draw(_TEXT),
        message=draw(_TEXT),
        service_id=draw(st.none() | _IDS),
        element=draw(st.none() | st.sampled_from(Element)),
        subject_ids=tuple(draw(st.lists(_IDS, max_size=8))),
        evidence=tuple(draw(st.lists(_spans(), max_size=3))),
        fix=FixHint(kind=fix) if fix is not None else None,
        legal_basis=draw(st.none() | _TEXT),
    )


@st.composite
def _services(draw: st.DrawFn) -> ServiceReadiness:
    elements = draw(st.lists(st.sampled_from(Element), max_size=5, unique=True))
    return ServiceReadiness(
        service_id=draw(_IDS),
        service_name=draw(_TEXT),
        tier=draw(st.integers(-3, 10**12)),
        weight=1.0,
        score=draw(st.floats(0, 100)),
        elements=tuple(
            ElementScore(
                element=element,
                score=draw(st.floats(0, 1)),
                status=draw(st.sampled_from(ElementStatus)),
                reasons=tuple(draw(st.lists(_TEXT, max_size=2))),
            )
            for element in elements
        ),
    )


@st.composite
def _results(draw: st.DrawFn) -> RehearsalResult:
    times = draw(st.lists(st.floats(0, 1e6), max_size=2))
    return RehearsalResult(
        scenario_id=draw(st.sampled_from((*_SCENARIOS, "sc-ghost"))),
        scenario_fingerprint=draw(st.sampled_from(["f" * 16, "e" * 16])),
        graph_hash=draw(_HASHES),
        config_hash="c" * 16,
        outcome=draw(st.sampled_from(Outcome)),
        time_to_mitigate=draw(st.none() | st.floats()),
        deadline_minutes=draw(st.floats(-1e9, 1e9)),
        failed_element=draw(st.none() | st.sampled_from(Element)),
        failed_at_id=draw(st.none() | _IDS),
        reason=draw(_TEXT),
        events=tuple(SimEvent(t=t, kind=SimEventKind.PAGE) for t in times),
    )


@st.composite
def _patches(draw: st.DrawFn) -> Patch:
    return Patch(
        id=draw(st.sampled_from(["p1", "p2", "p3"])),
        workspace_id="ws",
        base_version=1,
        finding_ids=("JT-PATH-001:1",),
        ops=(),
        runbook_text=draw(_TEXT),
        rationale="",
        author=draw(st.sampled_from(PatchAuthor)),
        status=draw(st.sampled_from(PatchStatus)),
        validation=PatchValidation(
            passed=True,
            resolved_finding_ids=("JT-PATH-001:1",),
            rehearsals_fixed=tuple(draw(st.lists(st.sampled_from(_SCENARIOS), max_size=2))),
            rehearsals_regressed=tuple(draw(st.lists(st.sampled_from(_SCENARIOS), max_size=2))),
        ),
        decided_by=draw(st.none() | _TEXT),
        decided_at=draw(st.none() | st.datetimes(timezones=st.none() | st.just(UTC))),
    )


@st.composite
def _versions(draw: st.DrawFn) -> GraphVersion:
    return GraphVersion(
        workspace_id="ws",
        number=draw(st.integers(1, 5)),
        parent=None,
        graph_hash=draw(_HASHES),
        patch_id=draw(st.none() | st.sampled_from(["p1", "p2", "p9"])),
        revert_of=draw(st.none() | st.integers(1, 5)),
        created_by=draw(_TEXT),
        created_at=datetime(2026, 10, 1, tzinfo=UTC),
        note=draw(_TEXT),
    )


@st.composite
def _reports(draw: st.DrawFn) -> DiagnosisReport:
    scenario_ids = draw(st.lists(st.sampled_from(_SCENARIOS), max_size=3, unique=True))
    notices = draw(st.lists(st.tuples(_TEXT, _TEXT), max_size=2))
    return DiagnosisReport(
        run_id=draw(_TEXT),
        workspace_id=draw(_TEXT),
        graph_version=draw(st.integers(1, 5)),
        graph_hash=draw(_HASHES),
        created_at=draw(st.datetimes(timezones=st.none() | st.just(UTC))),
        findings=tuple(draw(st.lists(_findings(), max_size=5))),
        readiness=ReadinessReport(
            overall=draw(st.floats(0, 100)),
            services=tuple(draw(st.lists(_services(), max_size=3))),
            weights_version=draw(_TEXT),
            element_weights=draw(st.dictionaries(_TEXT, st.floats(0, 1), max_size=3)),
            tier_weights=draw(st.dictionaries(st.integers(-3, 9), st.floats(0, 5), max_size=3)),
            graph_hash="h",
        ),
        scenarios=tuple(
            Scenario(
                id=sid,
                title=draw(_TEXT),
                service_id=draw(_IDS),
                incident_type=draw(st.sampled_from(IncidentType)),
            )
            for sid in scenario_ids
        ),
        rehearsals=tuple(draw(st.lists(_results(), max_size=5))),
        patches=tuple(draw(st.lists(_patches(), max_size=3))),
        grounding=GroundingStatus(
            grounded=draw(st.integers(0, 5)), dropped_examples=tuple(draw(st.lists(_TEXT, max_size=2)))
        ),
        security=tuple(
            SecurityNotice(doc_id="doc-1", start=1, end=2, excerpt=excerpt, pattern=pattern)
            for excerpt, pattern in notices
        ),
        llm=LlmUsageSummary(
            mode=draw(st.sampled_from(LlmMode)),
            by_task=draw(st.dictionaries(_TEXT, st.integers(0, 9), max_size=2)),
            judgement_path_calls=draw(st.integers(0, 2)),
        ),
        held_element_ids=tuple(draw(st.lists(_IDS, max_size=3))),
        catalog_gaps=tuple(draw(st.lists(_IDS, max_size=2))),
    )


_ALLOWED_TAGS = frozenset(
    {
        *("html", "head", "meta", "title", "style", "body", "div", "header", "main", "footer"),
        *("section", "article", "nav", "h1", "h2", "h3", "h4", "p", "small", "span", "strong"),
        *("ul", "li", "a", "dl", "dt", "dd", "blockquote", "cite", "code", "br"),
        *("table", "thead", "tbody", "tr", "th", "td"),
    }
)


class _Structure(HTMLParser):
    """Collects tags and comments and checks that every element is closed in order."""

    def __init__(self) -> None:
        super().__init__()
        self.tags: set[str] = set()
        self.comments: list[str] = []
        self.open: list[str] = []
        self.misnested: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.add(tag)
        if tag not in ("meta", "br"):
            self.open.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if self.open and self.open[-1] == tag:
            self.open.pop()
        else:
            self.misnested.append(tag)

    def handle_comment(self, data: str) -> None:
        self.comments.append(data)


def _table_widths(markdown: str) -> list[set[int]]:
    """Cell counts per row, one set per table (a well-formed table has exactly one count)."""
    widths: list[set[int]] = []
    cells = 0
    for token in _MD.parse(markdown):
        if token.type == "table_open":
            widths.append(set())
        elif token.type == "tr_open":
            cells = 0
        elif token.type in ("th_open", "td_open"):
            cells += 1
        elif token.type == "tr_close":
            widths[-1].add(cells)
    return widths


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    _reports(),
    st.lists(_versions(), max_size=4),
    st.lists(_patches(), max_size=2),
    st.none() | _TEXT,
    st.integers(text_module.MIN_WIDTH, 120),
)
def test_every_renderer_accepts_any_valid_report(
    report: DiagnosisReport,
    versions: list[GraphVersion],
    patches: list[Patch],
    org_name: str | None,
    width: int,
) -> None:
    """Whatever a valid report holds — lone surrogates, control characters, unknown ids, missing
    scenarios, NaN/inf times, huge tiers — every format renders, stays well-formed and encodes."""
    graph = OpsGraph()
    doc = DocumentRef(doc_id="doc-1", title=report.run_id, sha256="d" * 64, path=report.workspace_id)
    markdown = render_evidence_markdown(
        report, graph, versions=versions, patches=patches, documents=(doc,), org_name=org_name
    )
    page = render_html(report, graph, versions=versions, patches=patches, documents=(doc,), org_name=org_name)
    log = json.dumps(to_sarif(report, graph, documents={"doc-1": doc}, tool_version="1.0"), allow_nan=False)
    summary = render_cli_summary(report, width=width)
    for output in (markdown, page, log, summary):
        output.encode("utf-8")

    tokens = _MD.parse(markdown)
    assert not [t.type for t in tokens if t.type == "html_block"]
    inline = {child.type for t in tokens if t.type == "inline" for child in t.children or []}
    assert not {"html_inline", "link_open", "image", "s_open"} & inline, inline
    assert all(len(widths) == 1 for widths in _table_widths(markdown))

    structure = _Structure()
    structure.feed(page)
    assert structure.tags <= _ALLOWED_TAGS
    assert not structure.comments
    assert not structure.open
    assert not structure.misnested

    assert all(display_width(line) <= width for line in summary.splitlines())
