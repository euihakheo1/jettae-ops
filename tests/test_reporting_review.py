"""Regression and adversarial tests from the independent review of ``jettae.reporting``.

Each test pins a defect found in review (named in its docstring) or an edge the original suite did
not reach: GitHub-only Markdown syntax (autolinks, math), a CLI hang on an over-wide prefix, lines
wider than the terminal, an overclaiming ISMS-P mapping, revert-recreated graph hashes, path
traversal in SARIF URIs, and rendering from several threads at once.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest
from hypothesis import given
from hypothesis import strategies as st
from markdown_it import MarkdownIt

import jettae.reporting.text as text_module
from jettae.domain import DocumentRef, GraphVersion, OpsGraph, Outcome, PatchStatus
from jettae.reporting import (
    build_evidence,
    render_cli_summary,
    render_evidence_markdown,
    render_html,
    to_sarif,
)
from jettae.reporting.evidence import (
    DROPPED_EXAMPLES_TITLE,
    UNLINKED_PATCHES_TITLE,
    ControlStatus,
    current_rehearsals,
    printable,
)
from jettae.reporting.markdown import md_inline, md_text
from jettae.reporting.sarif import artifact_uri
from jettae.reporting.text import display_width, wrap_display
from test_reporting_sample import (
    F_DEPARTED,
    F_HOSTILE,
    F_ONCALL,
    GRAPH_HASH,
    OLD_HASH,
    P_APPROVED,
    REHEARSALS,
    SC_BASE,
    SPAN_SENTENCE,
    VERSIONS,
    documents,
    empty_report,
    sample_graph,
    sample_report,
)

_MD = MarkdownIt("commonmark").enable("table").enable("strikethrough")

#: What GitHub's extended autolinks and math fire on: a literal ``scheme://``, a literal ``www.``
#: (any case) or a literal ``$``. Escaped forms (``\://``, ``www\.``, ``&#36;``) do not match.
_GITHUB_TRIGGER = re.compile(r"(?<!\\):(?=//)|(?i:www)\.|\$")


def _paragraph_text(markdown: str) -> str:
    tokens = _MD.parse(markdown)
    assert [t.type for t in tokens] == ["paragraph_open", "inline", "paragraph_close"], markdown
    children = tokens[1].children or []
    assert {c.type for c in children} <= {"text", "text_special"}, markdown
    return "".join(c.content for c in children)


# --------------------------------------------------------------------------- Markdown: GitHub syntax


@pytest.mark.parametrize(
    "text",
    [
        "https://evil.example/login?next=/",
        "HTTP://EVIL.EXAMPLE",
        "ftp://files.example",
        "자세히는 www.evil.example 참고",
        "WWW.Evil.Example",
        "(www.evil.example)",
        "*www.evil.example*",
        "비용 $5 ~ $10",
        "$$\\color{green}\\Huge\\text{모든 항목 정상}$$",
        "$`\\sqrt{2}`$",
        "[링크](http://evil.example)",
    ],
)
def test_bare_urls_and_math_stay_plain_text(text: str) -> None:
    """Defect: GFM autolinks and GitHub math turned runbook text into links/styled text."""
    for escaped in (md_text(text), md_inline(text)):
        assert _GITHUB_TRIGGER.search(escaped) is None, escaped
    assert _paragraph_text(md_text(text)) == text


_HOSTILE_PIECES = st.sampled_from(
    ["://", "http", "https", "ftp", "www.", "WwW.", "$", "$$", "`", "\\", ".", ":", "/", "가", " ", "a", "("]
)


@given(st.lists(_HOSTILE_PIECES, max_size=16).map("".join))
def test_escaped_text_never_triggers_github_syntax_and_reads_back(text: str) -> None:
    escaped = md_text(text)
    assert _GITHUB_TRIGGER.search(escaped) is None, escaped
    expected = " ".join(printable(text).split())
    if expected:
        assert _paragraph_text(escaped) == expected


def test_hostile_urls_in_a_finding_do_not_become_links_in_the_report() -> None:
    hostile = F_HOSTILE.model_copy(
        update={
            "message": "점검 https://evil.example 또는 www.evil.example $\\Huge 정상$",
            "title": "www.x.y",
        }
    )
    markdown = render_evidence_markdown(
        sample_report(findings=(hostile,)), sample_graph(), documents=documents()
    )
    assert _GITHUB_TRIGGER.search(markdown) is None
    assert "점검 https\\://evil.example 또는 www\\.evil.example &#36;" in markdown


# --------------------------------------------------------------------------- wording of the history


def test_patches_without_a_version_are_not_called_unapplied() -> None:
    """Defect: with no version history passed, an *approved* patch was listed under
    "버전에 반영되지 않은 수정안" — false, an approved patch always produced a version."""
    report = sample_report()
    assert any(p.status is PatchStatus.APPROVED for p in report.patches)
    markdown = render_evidence_markdown(report, sample_graph())
    html = render_html(report, sample_graph())
    for output in (markdown, html):
        assert "반영되지 않은" not in output
        assert UNLINKED_PATCHES_TITLE in output
    section = markdown.split(f"### {UNLINKED_PATCHES_TITLE}", 1)[1].split("## 6.", 1)[0]
    assert f"`{P_APPROVED.id}` | 승인 |" in section


def test_dropped_examples_are_not_all_called_ungrounded() -> None:
    """Defect: the heading said every example lacked a source, but examples carry any drop reason."""
    grounding = sample_report().grounding.model_copy(
        update={"dropped_examples": ('격리 구간 인용: 인물 "x"',)}
    )
    report = sample_report(grounding=grounding)
    for output in (render_evidence_markdown(report, sample_graph()), render_html(report, sample_graph())):
        assert DROPPED_EXAMPLES_TITLE in output
        assert "원문 근거가 없어 버린" not in output


# --------------------------------------------------------------------------- ISMS-P mapping


def test_failing_centre_rehearsal_is_a_gap_for_preparedness_too() -> None:
    """Defect: 2.12.1 read "이상 없음" next to failing centre-down replays (demo: 5 of 9 failed)."""
    failing = tuple(
        r.model_copy(update={"outcome": Outcome.FAIL}) if r.scenario_id == "sc-dc" else r for r in REHEARSALS
    )
    view = build_evidence(sample_report(rehearsals=failing), sample_graph(), versions=VERSIONS)
    controls = {c.control_id: c for c in view.controls}
    assert controls["2.12.1"].status is ControlStatus.GAP
    assert controls["2.12.1"].evidence.endswith("센터 장애 시나리오 1건(실패 1건)")
    assert "4장 센터 장애 리허설" in controls["2.12.1"].source
    assert controls["2.12.2"].status is ControlStatus.GAP


def test_no_control_reads_ok_while_its_rehearsals_fail() -> None:
    """P3 across the whole mapping: any failing replay a control cites makes it a gap."""
    every_fail = tuple(r.model_copy(update={"outcome": Outcome.FAIL}) for r in REHEARSALS)
    view = build_evidence(sample_report(findings=(), rehearsals=every_fail), sample_graph())
    assert all(c.status is not ControlStatus.OK for c in view.controls)


def test_scoring_method_says_which_services_the_overall_averages() -> None:
    """Defect: appendix A said "핵심 서비스" while the tables label only tier 1 "핵심"; with the
    default policy tiers 1 and 2 both count, so a reader averaging "1(핵심)" rows got another number."""
    markdown = render_evidence_markdown(sample_report(), sample_graph())
    appendix = markdown.split("## 부록 A.", 1)[1].split("## 부록 B.", 1)[0]
    scope = "핵심 서비스(정책이 정한 핵심 등급, 기본값 1·2등급)"
    assert scope in appendix
    assert scope in render_html(sample_report(), sample_graph())


# --------------------------------------------------------------------------- revert keeps the number


def test_report_keeps_its_own_version_number_when_a_revert_recreates_the_graph() -> None:
    """Defect: a revert recreates an earlier graph byte for byte (same hash); the newer version
    number then overwrote the report's own number in every before/after label."""
    revert = GraphVersion(
        workspace_id="ws-1",
        number=3,
        parent=2,
        graph_hash=GRAPH_HASH,
        revert_of=2,
        created_by="박재현",
        created_at=datetime(2026, 10, 22, tzinfo=UTC),
    )
    report = sample_report(graph_version=2)
    view = build_evidence(report, sample_graph(), versions=(*VERSIONS, revert))
    base = next(r for r in view.rehearsals if r.scenario_id == SC_BASE.id)
    assert base.comparison == "v1 실패 → v2 통과"
    pairs = current_rehearsals(report, {GRAPH_HASH: 3, OLD_HASH: 1})
    current, before = next(p for p in pairs if p[0].scenario_id == SC_BASE.id)
    assert current.graph_hash == GRAPH_HASH
    assert before is not None
    assert before.graph_hash == OLD_HASH


# --------------------------------------------------------------------------- CLI: termination & width


def _bounded_take(limit: int) -> Callable[[str, int], tuple[str, str]]:
    """``_take`` that fails instead of looping forever, so a regression cannot hang the suite."""
    original = text_module._take
    calls = 0

    def take(word: str, columns: int) -> tuple[str, str]:
        nonlocal calls
        calls += 1
        assert calls <= limit, "wrap_display is not making progress"
        return original(word, columns)

    return take


def test_wrap_display_terminates_when_the_prefix_is_wider_than_the_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defect: a prefix at least as wide as ``width`` made wrap_display loop forever."""
    monkeypatch.setattr(text_module, "_take", _bounded_take(1_000))
    lines = wrap_display("결제 장애 대응", 10, first="x" * 20, rest="y" * 20)
    content = "".join(line.lstrip("xy").replace(" ", "") for line in lines)
    assert content == "결제장애대응"


def test_huge_service_tier_neither_hangs_nor_overflows_the_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defect: the tier is an unbounded catalog integer; a wide one hung ``render_cli_summary``."""
    monkeypatch.setattr(text_module, "_take", _bounded_take(10_000))
    report = sample_report()
    services = tuple(s.model_copy(update={"tier": 10**40}) for s in report.readiness.services)
    huge = report.model_copy(update={"readiness": report.readiness.model_copy(update={"services": services})})
    for width in (40, 41, 60, 100):
        lines = render_cli_summary(huge, width=width).splitlines()
        assert all(display_width(line) <= width for line in lines), width
        assert any("대체채널○" in line for line in lines)


@pytest.mark.parametrize("width", range(text_module.MIN_WIDTH, text_module.MIN_WIDTH + 6))
def test_overflow_notes_fit_the_narrowest_terminal(width: int) -> None:
    """Defect: "… 외 N건은 증적 리포트에서 확인하세요." was not wrapped (41–43 columns at width 40)."""
    many = tuple(F_ONCALL.model_copy(update={"id": f"JT-ONC-003:{i:010d}"}) for i in range(150))
    failing = tuple(
        REHEARSALS[2].model_copy(update={"scenario_id": f"sc-{i:03d}", "reason": "실패"}) for i in range(120)
    )
    output = render_cli_summary(sample_report(findings=many, rehearsals=failing), width=width, top=1)
    assert all(display_width(line) <= width for line in output.splitlines())
    assert "149건은" in output
    assert "외 115건" in output


def test_element_marks_are_never_split_from_their_label() -> None:
    """A mark is one word, so "대체 / 채널○" can no longer straddle two lines."""
    for width in range(text_module.MIN_WIDTH, 101):
        block = render_cli_summary(sample_report(), width=width).split("[주요 끊김]")[0]
        assert block.count("대체채널○") == 2, width
        assert "대체 채널" not in block


# --------------------------------------------------------------------------- SARIF


@pytest.mark.parametrize(
    ("path", "uri"),
    [
        ("../../etc/passwd.md", "etc/passwd.md"),
        ("runbooks/../../../x.md", "x.md"),
        ("..\\..\\secret\\결제.md", "secret/%EA%B2%B0%EC%A0%9C.md"),
        ("./a/./b/../c.md", "a/c.md"),
        ("..", "%EA%B2%B0%EC%A0%9C"),
        ("../", "%EA%B2%B0%EC%A0%9C"),
    ],
)
def test_relative_paths_cannot_climb_above_the_repository_root(path: str, uri: str) -> None:
    """Defect: an uploaded file name such as ``../../etc/x.md`` produced a URI outside the repo."""
    assert artifact_uri(DocumentRef(doc_id="d", title="결제", sha256="0" * 64, path=path)) == uri


@pytest.mark.parametrize(
    ("path", "uri"),
    [
        ("\\\\", "%EA%B2%B0%EC%A0%9C"),
        ("\\\\x.md", "x.md"),
        ("\\\\server", "server"),
        ("\\\\server\\", "server"),
        ("\\\\server\\share\\결제.md", "file://server/share/%EA%B2%B0%EC%A0%9C.md"),
    ],
)
def test_incomplete_unc_paths_do_not_crash_the_export(path: str, uri: str) -> None:
    """Defect: ``\\\\host`` without a share made pathlib raise ``ValueError`` inside ``to_sarif``."""
    ref = DocumentRef(doc_id="d", title="결제", sha256="0" * 64, path=path)
    assert artifact_uri(ref) == uri
    to_sarif(sample_report(findings=(F_ONCALL,)), OpsGraph(), documents={"d": ref}, tool_version="1.0")


@given(st.text())
def test_artifact_uri_never_raises_and_never_escapes_the_root(path: str) -> None:
    uri = artifact_uri(DocumentRef(doc_id="d", title="결제", sha256="0" * 64, path=path))
    assert uri
    assert uri.isascii()
    assert " " not in uri
    if not uri.startswith("file:"):
        assert not uri.startswith("/")
        assert ".." not in uri.split("/")


def test_sarif_snippets_drop_control_and_bidi_characters_but_keep_lines() -> None:
    span = SPAN_SENTENCE.model_copy(update={"quote": "첫 줄\x1b[2J\u202e\n둘째  줄\x07"})
    finding = F_DEPARTED.model_copy(update={"evidence": (span,)})
    log = to_sarif(sample_report(findings=(finding,)), sample_graph(), documents={}, tool_version="1.0")
    (location,) = log["runs"][0]["results"][0]["locations"]
    region = location["physicalLocation"]["region"]
    assert region["snippet"]["text"] == "첫 줄[2J\n둘째  줄"
    assert SPAN_SENTENCE.line is not None
    assert region["endLine"] == SPAN_SENTENCE.line + 1


def test_sarif_does_not_depend_on_the_documents_mapping_order() -> None:
    docs = documents()
    forward = {d.doc_id: d for d in docs}
    backward = {d.doc_id: d for d in reversed(docs)}
    a = to_sarif(sample_report(), sample_graph(), documents=forward, tool_version="1.0")
    b = to_sarif(sample_report(), sample_graph(), documents=backward, tool_version="1.0")
    assert json.dumps(a, ensure_ascii=False) == json.dumps(b, ensure_ascii=False)


def test_every_format_handles_an_empty_report_on_an_empty_graph() -> None:
    report, graph = empty_report(), OpsGraph()
    log = to_sarif(report, graph, documents={}, tool_version="1.0")
    run = log["runs"][0]
    assert run["results"] == []
    assert run["artifacts"] == []
    json.dumps(log)
    assert "진단에 쓴 문서 기록이 없습니다." in render_evidence_markdown(report, graph)
    assert "진단에 쓴 문서 기록이 없습니다." in render_html(report, graph)
    assert "점수를 매긴 서비스가 없습니다." in render_cli_summary(report)


@pytest.mark.parametrize("message", ["0\r\r5", "a  b", "줄\n바꿈\t탭"])
def test_html_and_markdown_carry_the_same_text_for_whitespace_runs(message: str) -> None:
    """Defect (flaky property, falsified by ``'0\\r\\r5'``): HTML kept whitespace runs that the
    Markdown report collapses, so the two "identical" evidence files held different text."""
    flat = " ".join(printable(message).split())
    report = sample_report(findings=(F_HOSTILE.model_copy(update={"message": message}),))
    assert f"<p>{flat}</p>" in render_html(report, sample_graph())
    assert f"\n{flat}\n" in render_evidence_markdown(report, sample_graph())


# --------------------------------------------------------------------------- concurrency


def test_rendering_from_many_threads_gives_identical_bytes() -> None:
    """Renderers share no mutable state, so concurrent API requests get identical output (P6)."""
    report, graph, docs = sample_report(), sample_graph(), documents()

    def render(_: int) -> tuple[str, str, str, str]:
        return (
            render_evidence_markdown(report, graph, versions=VERSIONS, documents=docs),
            render_html(report, graph, versions=VERSIONS, documents=docs),
            json.dumps(
                to_sarif(report, graph, documents={d.doc_id: d for d in docs}, tool_version="1.0"),
                ensure_ascii=False,
            ),
            render_cli_summary(report),
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        outputs = list(pool.map(render, range(16)))
    assert all(output == outputs[0] for output in outputs)
