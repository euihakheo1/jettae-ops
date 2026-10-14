"""Terminal and file output of the CLI (:mod:`jettae.cli.output`) and the CI gate.

What must hold whatever the report says: runbook text can never style, erase or reorder the
operator's terminal (P5); plain output has no colour codes and only ASCII table borders; files and
redirected streams are UTF-8; evidence is located from the working directory; the gate fails
exactly when the score is below the bar or a critical finding exists and was asked about.
"""

from __future__ import annotations

import io
import json
import os
import unicodedata
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from rich.console import Console

from jettae.check.registry import rule_catalog
from jettae.cli.main import _gate_failures
from jettae.cli.output import (
    PLAIN_WIDTH,
    ScanSummary,
    SourceLocator,
    data_table,
    display_prefix,
    ensure_utf8_streams,
    is_plain,
    judgement_line,
    located_documents,
    make_console,
    pass_rate,
    rehearsal_payload,
    rehearsal_renderables,
    render_plain,
    rules_json,
    rules_markdown,
    rules_table,
    scan_payload,
    scan_renderables,
    write_text,
)
from jettae.config import Settings
from jettae.domain import (
    DiagnosisReport,
    DocumentRef,
    Finding,
    FindingStatus,
    LlmMode,
    Outcome,
    RuleInfo,
    Severity,
    SourceSpan,
)
from jettae.ingest import load_bundle_dir
from jettae.orchestrator import JettaeService
from jettae.store import MemoryStore
from test_orchestrator_support import event_loop_runner

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "dmc-shop"
BOX_DRAWING = {chr(c) for c in range(0x2500, 0x2580)}


@pytest.fixture(scope="module")
def report() -> Iterator[DiagnosisReport]:
    service = JettaeService(MemoryStore(), Settings(_env_file=None, llm_mode=LlmMode.OFF))
    workspace = service.create_workspace("dmc-shop")
    service.load_bundle(workspace.id, load_bundle_dir(DEMO), scenarios=(DEMO / "scenarios.yaml").read_bytes())
    with event_loop_runner() as runner:
        yield runner.run(service.diagnose(workspace.id, actor="test", draft=False))


def summary_of(report: DiagnosisReport, *, top: int = 10) -> ScanSummary:
    return ScanSummary(report=report, target="examples/dmc-shop", locator=SourceLocator({}), top=top)


def colour_console(buffer: io.StringIO) -> Console:
    return Console(
        file=buffer, force_terminal=True, color_system="standard", width=100, markup=False, emoji=False
    )


# --------------------------------------------------------------------------- locations


def test_display_prefix(tmp_path: Path) -> None:
    assert display_prefix(tmp_path, tmp_path) == ""
    assert display_prefix(tmp_path / "a" / "b", tmp_path) == "a/b/"
    outside = display_prefix(tmp_path, tmp_path / "sub")
    assert outside == Path(os.path.abspath(tmp_path)).as_posix().rstrip("/") + "/"
    assert ".." not in outside


@given(st.lists(st.text(alphabet="abc가나다-_0", min_size=1, max_size=6), min_size=1, max_size=4))
def test_display_prefix_names_the_directory_from_cwd(parts: list[str]) -> None:
    base = Path(os.path.abspath("cwd-root"))
    prefix = display_prefix(base.joinpath(*parts), base)
    assert prefix == "/".join(parts) + "/"
    assert PurePosixPath(prefix.rstrip("/")).parts == tuple(parts)


def test_located_documents_only_prefix_paths() -> None:
    docs = {
        "d1": DocumentRef(doc_id="d1", title="결제", sha256="0" * 64, path="runbooks/a.md"),
        "d2": DocumentRef(doc_id="d2", title="업로드", sha256="1" * 64, path=None),
    }
    located = located_documents(docs, "ops/bundle/")
    assert located["d1"].path == "ops/bundle/runbooks/a.md"
    assert located["d2"].path is None
    assert docs["d1"].path == "runbooks/a.md"  # metadata copies; the originals are untouched
    assert located_documents(docs, "") == docs


def span(doc_id: str, start: int, line: int | None) -> SourceSpan:
    return SourceSpan(doc_id=doc_id, start=start, end=start + 3, quote="김대리", line=line)


def finding(*evidence: SourceSpan, **changes: object) -> Finding:
    base = Finding(
        id="JT-PPL-001:abc",
        rule_id="JT-PPL-001",
        severity=Severity.CRITICAL,
        title="퇴사자에게 대응이 걸려 있음",
        message="설명",
        service_id="service:결제",
        evidence=evidence,
    )
    return base.model_copy(update=changes)


def test_source_locator_where() -> None:
    locator = SourceLocator.of(
        {
            "b": DocumentRef(doc_id="b", title="B", sha256="0" * 64, path="runbooks/b.md"),
            "a": DocumentRef(doc_id="a", title="A", sha256="1" * 64, path="runbooks/a.md"),
            "t": DocumentRef(doc_id="t", title="제목\u202e뿐", sha256="2" * 64, path=None),
        },
        "bundle/",
    )
    assert locator.where(finding()) == "없음(빠진 항목이라 인용할 문장이 없음)"
    # first in reading order (path, then offset) — not in document-id order
    assert (
        locator.where(finding(span("b", 5, 2), span("a", 40, 7), span("a", 10, 3)))
        == "bundle/runbooks/a.md:3 외 2곳"
    )
    assert locator.where(finding(span("a", 0, None))) == "bundle/runbooks/a.md"
    assert locator.locate(span("t", 0, 4)) == "제목뿐:4"  # bidi override dropped (P5)
    assert locator.locate(span("unknown", 0, 1)) == "unknown:1"


# --------------------------------------------------------------------------- streams and files


class _TtyBytes(io.BytesIO):
    def isatty(self) -> bool:
        return True


def test_redirected_stream_becomes_utf8() -> None:
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp949")
    ensure_utf8_streams(stream)
    stream.write("준비도 — 55.7")
    stream.flush()
    assert raw.getvalue().decode("utf-8") == "준비도 — 55.7"


def test_terminal_keeps_its_encoding_but_never_crashes() -> None:
    raw = _TtyBytes()
    stream = io.TextIOWrapper(raw, encoding="cp949")
    ensure_utf8_streams(stream)
    assert stream.encoding == "cp949"
    stream.write("준비도 — 끝")  # the em dash has no cp949 encoding
    stream.flush()
    assert raw.getvalue().decode("cp949") == "준비도 ? 끝"


def test_utf8_and_foreign_streams_are_left_alone() -> None:
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="utf-8", errors="strict")
    ensure_utf8_streams(stream, io.StringIO())
    assert stream.errors == "strict"


def test_write_text(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b.md"
    write_text(target, "첫 줄\n둘째 줄\n")
    assert target.read_bytes() == "첫 줄\n둘째 줄\n".encode()  # \n even on Windows
    target.write_bytes(b"no newline")
    write_text(target, "next\n", append=True)
    assert target.read_bytes() == b"no newline\nnext\n"
    write_text(target, "more\n", append=True)
    assert target.read_bytes() == b"no newline\nnext\nmore\n"
    empty = tmp_path / "empty.md"
    empty.touch()
    write_text(empty, "x\n", append=True)
    assert empty.read_bytes() == b"x\n"
    with pytest.raises(IsADirectoryError, match="폴더"):
        write_text(tmp_path, "x")


def test_plain_console_width(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COLUMNS", raising=False)
    console = make_console(io.StringIO())
    assert is_plain(console)
    assert console.width == PLAIN_WIDTH
    monkeypatch.setenv("COLUMNS", "140")
    assert make_console(io.StringIO()).width == 140
    monkeypatch.setenv("COLUMNS", "20")  # too narrow for the finding text: ignored
    assert make_console(io.StringIO()).width == PLAIN_WIDTH


# --------------------------------------------------------------------------- the summary


def test_plain_summary_is_ascii_bordered_and_colourless(report: DiagnosisReport) -> None:
    text = render_plain(scan_renderables(summary_of(report), plain=True))
    assert "\x1b" not in text
    assert not BOX_DRAWING & set(text)
    assert f"{report.readiness.overall:.1f} / 100" in text
    assert "판정 경로 LLM 0회" in text
    passed, total = pass_rate(report)
    assert f"{total}건 중 {passed}건 통과" in text


def test_terminal_summary_is_coloured(report: DiagnosisReport) -> None:
    buffer = io.StringIO()
    console = colour_console(buffer)
    for renderable in scan_renderables(summary_of(report), plain=False):
        console.print(renderable)
    assert "\x1b[" in buffer.getvalue()


def test_runbook_text_cannot_drive_the_terminal(report: DiagnosisReport) -> None:
    hostile = finding(
        title="[bold red]가짜 경고[/bold red] — \x1b[2J화면 지우기",
        message="\x1b]0;창 제목\x07설명 \u202e꾸로거 [link=http://evil]여기[/link]",
    )
    buffer = io.StringIO()
    console = colour_console(buffer)
    for renderable in scan_renderables(
        summary_of(report.model_copy(update={"findings": (hostile,)})), plain=False
    ):
        console.print(renderable)
    out = buffer.getvalue()
    assert "[bold red]가짜 경고[/bold red]" in out  # shown literally, never interpreted as markup
    assert "[link=http://evil]여기[/link]" in out
    assert "\x1b[2J" not in out
    assert "\x1b]0;" not in out
    assert "\u202e" not in out


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(text=st.text(min_size=1, max_size=60))
def test_plain_output_never_carries_control_characters(report: DiagnosisReport, text: str) -> None:
    hostile = finding(title=text, message=text, evidence=())
    rendered = render_plain(
        scan_renderables(summary_of(report.model_copy(update={"findings": (hostile,)}), top=1), plain=True)
    )
    controls = {ch for ch in rendered if ch != "\n" and unicodedata.category(ch) in ("Cc", "Cf", "Cs")}
    assert not controls


def test_summary_top_and_unknown(report: DiagnosisReport) -> None:
    unknown = finding(status=FindingStatus.UNKNOWN, id="JT-PPL-001:unknown")
    many = report.model_copy(update={"findings": (*report.findings, unknown)})
    text = render_plain(scan_renderables(summary_of(many, top=2), plain=True))
    assert f"전체 {len(many.findings)}건 중 심각도순 상위 2건" in text
    assert f"외 {len(many.findings) - 2}건" in text
    assert "확인 불가 1건은 정상으로 보지 않습니다" in text
    nothing = render_plain(
        scan_renderables(summary_of(report.model_copy(update={"findings": ()})), plain=True)
    )
    assert "고정 규칙이 찾은 끊김이 없습니다." in nothing


def test_judgement_line_reports_a_violation() -> None:
    assert judgement_line(0).plain.startswith("판정 경로 LLM 0회")
    assert "위반" in judgement_line(2).plain


def test_scan_payload(report: DiagnosisReport) -> None:
    docs = {"d": DocumentRef(doc_id="d", title="결제", sha256="0" * 64, path="bundle/runbooks/a.md")}
    payload = scan_payload(
        report, target="bundle", tool_version="9.9.9", documents=docs, notes=("메모",), gate_failures=("x",)
    )
    assert payload["gate"] == {"passed": False, "failures": ["x"]}
    assert payload["documents"] == {
        "d": {"path": "bundle/runbooks/a.md", "title": "결제", "sha256": "0" * 64}
    }
    assert DiagnosisReport.model_validate(payload["report"]) == report
    assert payload["summary"]["rehearsals"] == pass_rate(report)[1]


# --------------------------------------------------------------------------- rehearsal


def test_rehearsal_views(report: DiagnosisReport) -> None:
    text = render_plain(rehearsal_renderables(report, target="bundle", notes=("드릴 하나를 뺐습니다.",)))
    assert "[참고] 드릴 하나를 뺐습니다." in text
    payload = rehearsal_payload(report, target="bundle", tool_version="1", notes=())
    outcomes = [r["result"]["outcome"] for r in payload["results"]]
    assert outcomes == sorted(outcomes)
    assert payload["passed"] == outcomes.count(Outcome.PASS.value)
    empty = report.model_copy(update={"rehearsals": (), "scenarios": ()})
    assert "재생할 시나리오가 없습니다" in render_plain(rehearsal_renderables(empty, target="b", notes=()))
    assert rehearsal_payload(empty, target="b", tool_version="1", notes=())["pass_rate"] is None


def test_nothing_replayed_is_never_shown_as_passed(report: DiagnosisReport) -> None:
    """0 of 0 is not "all passed" (P3): the rate line is yellow, not the green of a full pass."""
    empty = report.model_copy(update={"rehearsals": (), "scenarios": ()})
    buffer = io.StringIO()
    console = colour_console(buffer)
    for renderable in rehearsal_renderables(empty, target="b", notes=()):
        console.print(renderable)
    (line,) = [line for line in buffer.getvalue().splitlines() if "통과율 해당 없음" in line]
    assert "\x1b[33m" in line  # yellow
    assert "\x1b[32m" not in line  # never green


# --------------------------------------------------------------------------- rules


def test_rules_formats() -> None:
    catalog = rule_catalog()
    table = render_plain([rules_table(catalog, plain=True)])
    for rule in catalog:
        assert rule.rule_id in table
    parsed = [RuleInfo.model_validate(item) for item in json.loads(rules_json(catalog))]
    assert parsed == sorted(catalog, key=lambda r: r.rule_id)
    markdown = rules_markdown(catalog)
    assert markdown.count("\n| JT-") == len(catalog)


@given(title=st.text(min_size=1, max_size=40))
def test_markdown_rule_rows_survive_any_title(title: str) -> None:
    rule = rule_catalog()[0].model_copy(update={"title": title, "rationale": f"{title}|{title}"})
    rows = [line for line in rules_markdown([rule]).splitlines() if line.startswith("| JT-")]
    assert len(rows) == 1  # a newline in a title never splits the row
    unescaped = len(rows[0]) - len(rows[0].replace("|", "")) - rows[0].count("\\|")
    assert unescaped == 7  # six cells, never more
    assert "<" not in rows[0]


def test_data_table_never_parses_markup() -> None:
    text = render_plain([data_table(("a",), [("[red]x[/red]",)], plain=True)])
    assert "[red]x[/red]" in text


# --------------------------------------------------------------------------- gate


@given(
    overall=st.floats(min_value=0, max_value=100, allow_nan=False),
    bar=st.floats(min_value=0, max_value=100, allow_nan=False),
    critical=st.booleans(),
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
def test_gate_fails_exactly_below_the_bar(
    report: DiagnosisReport, overall: float, bar: float, critical: bool
) -> None:
    scored = report.model_copy(
        update={"readiness": report.readiness.model_copy(update={"overall": overall}), "findings": ()}
    )
    assert bool(_gate_failures(scored, fail_under=bar, fail_on_critical=critical)) == (overall < bar)


def test_gate_counts_unknown_critical_findings(report: DiagnosisReport) -> None:
    unknown = finding(status=FindingStatus.UNKNOWN)
    target = report.model_copy(update={"findings": (unknown,)})
    assert _gate_failures(target, fail_under=0, fail_on_critical=False) == []
    (failure,) = _gate_failures(target, fail_under=0, fail_on_critical=True)
    assert "1건" in failure
    assert "확인 불가 포함" in failure
