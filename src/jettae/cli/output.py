"""Output of the ``jettae`` command line (출력): Rich for people, plain text for logs and files.

Why this module exists next to :mod:`jettae.reporting`: the reporting package renders the
*documents* of a diagnosis (Markdown/HTML evidence, SARIF), identical whoever asks for them. A
terminal needs something else — the answer at a glance, coloured while a person watches and plain
when a CI log, a pipe or a file receives it — and evidence located as ``file:line`` *relative to
where the command was run*, so the path can be opened as printed. Everything here only presents:
scores, verdicts and replay outcomes are copied from the report that the fixed rules and the
deterministic replay produced (P1), and unknown stays visibly unknown (P3).

Three rules keep the output safe and stable:

* Untrusted text (runbook-derived names, finding messages, scenario titles) is always wrapped in
  :class:`rich.text.Text` after :func:`~jettae.reporting.evidence.printable`: Rich never parses it
  as markup (a ``[red]`` in a runbook stays literal) and control characters — an ANSI escape or a
  bidi override planted in a runbook — never reach the operator's terminal (P5).
* Plain output uses ASCII table borders. Box-drawing characters are East Asian *ambiguous* width
  and are drawn two columns wide by Korean-locale consoles and editors, which tears tables apart
  exactly where plain output ends up (CI logs, saved files).
* Every stream and file is UTF-8 (:func:`ensure_utf8_streams`, :func:`write_text`): Hangul and the
  em dash have no encoding on many Windows code pages, and a report must never fail to print.
"""

from __future__ import annotations

import codecs
import io
import json
import os
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, TextIO

from rich import box
from rich.box import Box
from rich.console import Console, RenderableType
from rich.table import Table
from rich.text import Text

from jettae.check.registry import sort_findings
from jettae.domain import (
    ELEMENT_LABEL_KO,
    DiagnosisReport,
    DocumentRef,
    Element,
    ElementStatus,
    Finding,
    FindingStatus,
    HypothesisStatus,
    Outcome,
    RehearsalResult,
    RuleInfo,
    Scenario,
    ScenarioOrigin,
    Severity,
    SourceSpan,
)
from jettae.reporting.evidence import (
    PATCH_STATUS_LABEL_KO,
    RULES_INTRO,
    SEVERITY_LABEL_KO,
    current_rehearsals,
    distinct_spans,
    format_datetime,
    format_minutes,
    hypothesis_counts,
    llm_usage_line,
    printable,
    service_rows,
    short_hash,
)

__all__ = [
    "PLAIN_WIDTH",
    "DataFormat",
    "RulesFormat",
    "ScanFormat",
    "ScanSummary",
    "SourceLocator",
    "data_table",
    "display_prefix",
    "ensure_utf8_streams",
    "is_plain",
    "judgement_line",
    "key_value_table",
    "located_documents",
    "make_console",
    "numbered_list",
    "pass_rate",
    "rehearsal_payload",
    "rehearsal_renderables",
    "render_plain",
    "rules_json",
    "rules_markdown",
    "rules_table",
    "scan_payload",
    "scan_renderables",
    "to_json",
    "write_text",
]

#: Width of plain output when ``COLUMNS`` does not say otherwise (CI logs are wide; 80 columns
#: squeeze the service table and the Korean finding text into slivers).
PLAIN_WIDTH: Final = 100
#: Narrower than this, ``COLUMNS`` is ignored: the service table alone needs about this much.
MIN_PLAIN_WIDTH: Final = 60
#: Characters of a finding's explanation shown in the summary (the evidence report has all).
FINDING_MESSAGE_CHARS: Final = 160
#: Failed rehearsals listed by name in the scan summary; the rest are counted.
FAILED_REHEARSALS_SHOWN: Final = 5
#: Rehearsal reasons are clipped in the terminal views (the JSON output carries the full timeline).
REASON_CHARS: Final = 160

_MISSING: Final = "-"
_ELLIPSIS: Final = "…"

SEVERITY_STYLE: Final[dict[Severity, str]] = {
    Severity.CRITICAL: "bold red",
    Severity.HIGH: "red",
    Severity.MEDIUM: "yellow",
    Severity.LOW: "cyan",
    Severity.INFO: "dim",
}
#: Short status words for table cells. The report's legend symbols (● ◐ ○) are East Asian
#: ambiguous width and misalign Korean-locale tables; words do not.
ELEMENT_STATUS_WORD: Final[dict[ElementStatus | None, tuple[str, str]]] = {
    ElementStatus.PASS: ("충족", "green"),
    ElementStatus.PARTIAL: ("부분", "yellow"),
    ElementStatus.FAIL: ("끊김", "bold red"),
    ElementStatus.UNKNOWN: ("확인 불가", "magenta"),
    None: ("기록 없음", "dim"),
}
ORIGIN_LABEL_KO: Final[dict[ScenarioOrigin, str]] = {
    ScenarioOrigin.TEMPLATE: "템플릿",
    ScenarioOrigin.HUMAN: "사람 작성",
    ScenarioOrigin.LLM: "LLM 제안",
}


class ScanFormat(StrEnum):
    """``jettae scan --format``: a terminal summary, the raw report, or one of the evidence files."""

    TEXT = "text"
    JSON = "json"
    MD = "md"
    HTML = "html"
    SARIF = "sarif"


class RulesFormat(StrEnum):
    """``jettae rules --format``."""

    TABLE = "table"
    JSON = "json"
    MD = "md"


class DataFormat(StrEnum):
    """Commands whose result is either a terminal view or JSON for another program."""

    TEXT = "text"
    JSON = "json"


# --------------------------------------------------------------------------- streams and files


def ensure_utf8_streams(*streams: TextIO) -> None:
    """Make sure ``streams`` (default: ``sys.stdout`` and ``sys.stderr``) can print any report.

    A pipe or file inherits the locale code page (cp949, cp1252 on a Windows runner), on which
    Hangul or the em dash raise ``UnicodeEncodeError`` mid-report: such streams become UTF-8, the
    encoding every reader of a saved report or CI log expects. A terminal keeps its own encoding
    (switching it would garble what the person sees) but replaces what it cannot show instead of
    crashing — that also covers devices that merely claim to be terminals, such as Windows ``NUL``.
    A real Windows console is UTF-8 already, so it is untouched.
    """
    for stream in streams or (sys.stdout, sys.stderr):
        if not isinstance(stream, io.TextIOWrapper) or codecs.lookup(stream.encoding).name == "utf-8":
            continue
        if stream.isatty():
            stream.reconfigure(errors="replace")
        else:
            stream.reconfigure(encoding="utf-8", errors="replace")


def _plain_width() -> int:
    raw = os.environ.get("COLUMNS", "").strip()
    if raw.isdigit() and int(raw) >= MIN_PLAIN_WIDTH:
        return int(raw)
    return PLAIN_WIDTH


def _console(file: TextIO, *, width: int | None, plain: bool) -> Console:
    """A console that never applies markup, emoji codes or auto-highlighting to the text it is
    given — runbook text included (P5); colours come only from explicit styles in this module.
    A plain console has no colour system at all (not even bold) and a fixed width."""
    if plain:
        return Console(
            file=file,
            width=width or _plain_width(),
            color_system=None,
            force_terminal=False,
            legacy_windows=False,
            markup=False,
            emoji=False,
            highlight=False,
        )
    return Console(file=file, width=width, markup=False, emoji=False, highlight=False)


def make_console(file: TextIO, *, width: int | None = None) -> Console:
    """A console on ``file``: Rich colours on a terminal, plain text everywhere else.

    Rich decides what a terminal is (``isatty`` plus ``FORCE_COLOR``/``NO_COLOR``/``TTY_COMPATIBLE``).
    Anything else is plain, so a CI log or a pipe receives the same bytes on every machine.
    """
    probe = _console(file, width=width, plain=False)
    return probe if probe.is_terminal else _console(file, width=width, plain=True)


def is_plain(console: Console) -> bool:
    """Plain consoles get ASCII tables (see the module docstring)."""
    return not console.is_terminal


def render_plain(renderables: Sequence[RenderableType], *, width: int = PLAIN_WIDTH) -> str:
    """``renderables`` as plain text (no colour; build them with ``plain=True``) — for files."""
    buffer = io.StringIO()
    console = _console(buffer, width=width, plain=True)
    for renderable in renderables:
        console.print(renderable)
    # Rich pads grid rows to the table width; trailing blanks only add noise to logs and diffs.
    return "".join(line.rstrip() + "\n" for line in buffer.getvalue().splitlines())


def write_text(path: Path, text: str, *, append: bool = False) -> None:
    """Write (or append) ``text`` as UTF-8 with ``\\n`` line ends on every OS; parents are created.

    Appending starts on a fresh line, so a file that does not end with a newline (a job summary
    written by an earlier step) never glues its last line to our first.
    """
    if path.is_dir():
        raise IsADirectoryError(f"출력 경로가 폴더입니다. 파일 이름까지 적어 주세요: {path.as_posix()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    if not append:
        path.write_bytes(data)
        return
    with path.open("ab") as handle:
        if handle.tell() > 0 and not _ends_with_newline(path):
            handle.write(b"\n")
        handle.write(data)


def _ends_with_newline(path: Path) -> bool:
    with path.open("rb") as handle:
        handle.seek(-1, os.SEEK_END)
        return handle.read(1) == b"\n"


def to_json(value: object) -> str:
    """Pretty, UTF-8-friendly JSON with a trailing newline (Korean stays readable, not ``\\uXXXX``)."""
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


# --------------------------------------------------------------------------- locations


def display_prefix(directory: Path, cwd: Path) -> str:
    """How the documents of ``directory`` are named from ``cwd``: ``""``, ``"sub/dir/"`` or an
    absolute POSIX path ending in ``/``.

    Documents carry bundle-relative paths (so the graph hash does not depend on where a bundle is
    checked out, P6); the prefix is added only for display and for SARIF, whose relative URIs
    GitHub resolves against the repository root — the directory the Action runs in. A directory
    outside ``cwd`` is named absolutely: ``..`` segments would be confined to the repository root
    by SARIF consumers and point at the wrong file.
    """
    target = Path(os.path.abspath(directory))
    base = Path(os.path.abspath(cwd))
    try:
        relative = target.relative_to(base)
    except ValueError:
        return target.as_posix().rstrip("/") + "/"
    text = relative.as_posix()
    return "" if text in ("", ".") else text + "/"


def located_documents(documents: Mapping[str, DocumentRef], prefix: str) -> dict[str, DocumentRef]:
    """``documents`` with ``prefix`` in front of every path (metadata copies; the graph keeps its own)."""
    if not prefix:
        return dict(documents)
    return {
        doc_id: ref.model_copy(update={"path": prefix + ref.path}) if ref.path else ref
        for doc_id, ref in documents.items()
    }


def _clean(text: str) -> str:
    """One printable line: control characters dropped, whitespace collapsed (P5)."""
    return " ".join(printable(text).split())


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + _ELLIPSIS


@dataclass(frozen=True, slots=True)
class SourceLocator:
    """Evidence spans as ``path:line`` the user can open from where the command ran."""

    paths: Mapping[str, str]

    @classmethod
    def of(cls, documents: Mapping[str, DocumentRef], prefix: str = "") -> SourceLocator:
        return cls(
            {
                doc_id: _clean(prefix + ref.path) if ref.path else _clean(ref.title) or doc_id
                for doc_id, ref in documents.items()
            }
        )

    def path(self, doc_id: str) -> str:
        return self.paths.get(doc_id, doc_id)

    def locate(self, span: SourceSpan) -> str:
        path = self.path(span.doc_id)
        return f"{path}:{span.line}" if span.line is not None else path

    def where(self, finding: Finding) -> str:
        """The first evidence location in reading order, plus how many others exist.

        A finding about something *missing* (no detection signal, no owner) has no sentence to
        cite (P2); it says so instead of pointing at an arbitrary line.
        """
        spans = sorted(distinct_spans(finding.evidence), key=lambda s: (self.path(s.doc_id), s.start))
        if not spans:
            return "없음(빠진 항목이라 인용할 문장이 없음)"
        first = self.locate(spans[0])
        return first if len(spans) == 1 else f"{first} 외 {len(spans) - 1}곳"


# --------------------------------------------------------------------------- building blocks


def _box(plain: bool) -> Box:
    return box.ASCII2 if plain else box.SIMPLE_HEAD


def data_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[str | Text]],
    *,
    plain: bool,
    title: str | None = None,
) -> Table:
    """A table whose cells are never parsed as markup (strings become plain :class:`Text`)."""
    table = Table(box=_box(plain), title=title, title_justify="left", show_lines=False, expand=False)
    for header in headers:
        table.add_column(header, overflow="fold")
    for row in rows:
        table.add_row(*(cell if isinstance(cell, Text) else Text(_clean(cell)) for cell in row))
    return table


def key_value_table(rows: Iterable[tuple[str, str | Text]]) -> Table:
    """Label/value lines without borders (the summary header)."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", no_wrap=True)
    grid.add_column(overflow="fold")
    for label, value in rows:
        grid.add_row(Text(label), value if isinstance(value, Text) else Text(_clean(value)))
    return grid


def _heading(text: str) -> Text:
    return Text(text, style="bold")


def judgement_line(calls: int) -> Text:
    """The P1 disclosure. A non-zero count is shown as a violation, never rounded away."""
    if calls == 0:
        return Text(
            "판정 경로 LLM 0회 — 끊김 판정·점수·리허설은 고정 규칙과 결정적 리플레이만 사용", style="green"
        )
    return Text(
        f"판정 경로 LLM {calls:,}회 — 설계 원칙(판정은 규칙) 위반이므로 이 결과를 증적으로 쓰지 마십시오",
        style="bold red",
    )


def _score_style(score: float) -> str:
    if score >= 80:
        return "bold green"
    if score >= 50:
        return "bold yellow"
    return "bold red"


def pass_rate(report: DiagnosisReport) -> tuple[int, int]:
    """(passed, total) over the current result of every scenario of the report."""
    pairs = current_rehearsals(report)
    return sum(1 for current, _ in pairs if current.outcome is Outcome.PASS), len(pairs)


def _rate_text(passed: int, total: int) -> str:
    rate = f"{100 * passed / total:.1f}%" if total else "해당 없음"
    return f"{total}건 중 {passed}건 통과 (통과율 {rate})"


# --------------------------------------------------------------------------- scan summary


@dataclass(frozen=True, slots=True)
class ScanSummary:
    """Everything the terminal summary of one ``jettae scan`` shows."""

    report: DiagnosisReport
    #: how the scanned path is shown (as the user typed it, POSIX separators)
    target: str
    locator: SourceLocator
    #: Korean remarks of the run (LLM fallback, dropped drills …) from the service
    notes: tuple[str, ...] = ()
    top: int = 10
    rehearsed: bool = True
    drafted: bool = True


def _severity_counts(findings: Sequence[Finding]) -> str:
    counts = Counter(f.severity for f in findings)
    return " · ".join(f"{SEVERITY_LABEL_KO[s]} {counts[s]}" for s in Severity if counts[s])


def _overview(summary: ScanSummary, broken: Sequence[Finding], unknown: int) -> Table:
    report = summary.report
    readiness = report.readiness
    score = Text(f"{readiness.overall:.1f} / 100", style=_score_style(readiness.overall))
    score.append(f"  (SCORING_{readiness.weights_version.upper()})", style="dim")
    by_severity = _severity_counts(broken)
    findings = f"{len(broken)}건" + (f" ({by_severity})" if by_severity else "") + f" · 확인 불가 {unknown}건"
    if summary.rehearsed:
        passed, total = pass_rate(report)
        rehearsal = _rate_text(passed, total) if total else "실행한 시나리오 없음"
    else:
        rehearsal = "생략(--no-rehearse)"
    created = format_datetime(report.created_at)
    graph = f"v{report.graph_version} · {short_hash(report.graph_hash)} · 진단 {created}"
    return key_value_table(
        [
            ("전체 준비도", score),
            ("그래프", graph),
            ("끊긴 경로", Text(findings, style="red" if broken else "green")),
            ("리허설", rehearsal),
            ("LLM", llm_usage_line(report.llm)),
            ("판정 방식", judgement_line(report.llm.judgement_path_calls)),
        ]
    )


def _status_cell(status: ElementStatus | None) -> Text:
    word, style = ELEMENT_STATUS_WORD[status]
    return Text(word, style=style)


def _services_table(report: DiagnosisReport, plain: bool) -> RenderableType:
    rows = service_rows(report)
    if not rows:
        return Text(
            "점수를 매긴 서비스가 없습니다(런북·카탈로그에서 서비스를 찾지 못했습니다).", style="yellow"
        )
    table = Table(box=_box(plain), title="서비스별 준비도 (핵심 등급부터, 약한 순)", title_justify="left")
    table.add_column("서비스", overflow="fold")
    table.add_column("등급", justify="right", no_wrap=True)
    table.add_column("점수", justify="right", no_wrap=True)
    for element in Element:
        table.add_column(ELEMENT_LABEL_KO[element], no_wrap=True)
    for row in rows:
        table.add_row(
            Text(_clean(row.name)),
            Text(f"{row.tier}"),
            Text(f"{row.score:.1f}", style=_score_style(row.score)),
            *(_status_cell(status) for status in row.statuses),
        )
    return table


def numbered_list(items: Sequence[Text], *, start: int = 1) -> Table:
    """Numbered entries with a hanging indent: wrapped lines stay under their entry's text, so a
    long Korean explanation never runs back under the numbers (a borderless two-column grid)."""
    grid = Table.grid(padding=(0, 1))
    grid.add_column(justify="right", no_wrap=True)
    grid.add_column(overflow="fold")
    for number, item in enumerate(items, start=start):
        grid.add_row(Text(f"{number}."), item)
    return grid


def _finding_entry(finding: Finding, service: str, where: str) -> Text:
    entry = Text(SEVERITY_LABEL_KO[finding.severity], style=SEVERITY_STYLE[finding.severity])
    if finding.status is FindingStatus.UNKNOWN:
        entry.append(" · 확인 불가", style="magenta")
    entry.append(f"  {finding.rule_id}  ", style="dim")
    entry.append(_clean(f"{finding.title} — {service}"), style="bold")
    entry.append(f"\n위치 {where}", style="cyan")
    message = _clip(_clean(finding.message), FINDING_MESSAGE_CHARS)
    if message:
        entry.append("\n" + message)
    return entry


def _findings_block(summary: ScanSummary, findings: Sequence[Finding]) -> list[RenderableType]:
    if not findings:
        return [Text("고정 규칙이 찾은 끊김이 없습니다.", style="green")]
    shown = list(findings[: summary.top])
    names = {s.service_id: s.service_name for s in summary.report.readiness.services}
    title = f"주요 끊김 — 전체 {len(findings)}건" + (
        f" 중 심각도순 상위 {len(shown)}건" if len(shown) < len(findings) else ""
    )
    blocks: list[RenderableType] = [_heading(title)]
    entries = [
        _finding_entry(
            finding,
            "조직 전체" if finding.service_id is None else names.get(finding.service_id, finding.service_id),
            summary.locator.where(finding),
        )
        for finding in shown
    ]
    if entries:
        blocks.append(numbered_list(entries))
    if len(shown) < len(findings):
        blocks.append(
            Text(f"… 외 {len(findings) - len(shown)}건은 증적 리포트(--format md 또는 html)에서 확인하세요.")
        )
    return blocks


def _failed_rehearsals(report: DiagnosisReport) -> list[RenderableType]:
    titles = {s.id: s.title for s in report.scenarios}
    failed = [current for current, _ in current_rehearsals(report) if current.outcome is Outcome.FAIL]
    if not failed:
        return []
    entries: list[Text] = []
    for result in failed[:FAILED_REHEARSALS_SHOWN]:
        entry = Text(_clean(titles.get(result.scenario_id, result.scenario_id)), style="bold")
        entry.append("\n" + (_clip(_clean(result.reason), REASON_CHARS) or "사유 기록 없음"))
        entries.append(entry)
    blocks: list[RenderableType] = [
        Text(f"리허설 실패 {len(failed)}건", style="bold red"),
        numbered_list(entries),
    ]
    if len(failed) > FAILED_REHEARSALS_SHOWN:
        blocks.append(Text(f"… 외 {len(failed) - FAILED_REHEARSALS_SHOWN}건 (전체는 jettae rehearse로)"))
    return blocks


def _closing(summary: ScanSummary) -> list[RenderableType]:
    report = summary.report
    lines: list[Text] = []
    if report.patches:
        counts = Counter(p.status for p in report.patches)
        listed = " · ".join(f"{PATCH_STATUS_LABEL_KO[s]} {n}건" for s, n in sorted(counts.items()))
        lines.append(
            Text(
                f"[수정안] {listed} — 사람이 승인한 것만 새 그래프 버전이 됩니다(수정 문장은 증적 리포트에)."
            )
        )
    elif summary.drafted and report.findings:
        lines.append(Text("[수정안] 자동으로 초안을 만들 수 있는 끊김이 없습니다."))
    if report.held_element_ids:
        held = len(set(report.held_element_ids))
        lines.append(Text(f"[판단 보류] {held}건 — 사람이 원문과 대조해 확인해야 합니다.", style="magenta"))
    if report.catalog_gaps:
        lines.append(Text(f"[카탈로그] 담당·런북이 없는 서비스 {len(report.catalog_gaps)}개", style="yellow"))
    if report.exploration is not None or report.risk_hypotheses:
        tally = hypothesis_counts(h.status for h in report.risk_hypotheses)
        lines.append(
            Text(
                f"[규칙 밖 위험 후보] {tally} — 점수에 반영하지 않음(내용은 증적 리포트 9장).",
                style="cyan",
            )
        )
    if report.security:
        lines.append(
            Text(
                f"[보안 알림] 문서 속 AI 지시문 {len(report.security)}건을 격리했습니다. "
                "판정·점수에는 영향이 없습니다.",
                style="yellow",
            )
        )
    lines += [Text("[참고] " + _clean(note)) for note in summary.notes]
    lines.append(
        Text(
            "자세한 근거(원문 인용·법적 근거·인증 기준)는 증적 리포트(--format md 또는 html)에 있습니다.",
            style="dim",
        )
    )
    return [*lines]


def scan_renderables(summary: ScanSummary, *, plain: bool) -> list[RenderableType]:
    """The terminal summary of one scan: score, per-service table, top findings with
    ``file:line``, rehearsal pass rate, LLM use and the judgement-path disclosure."""
    findings = sort_findings(summary.report.findings)
    broken = [f for f in findings if f.status is FindingStatus.BROKEN]
    unknown = len(findings) - len(broken)
    blocks: list[RenderableType] = [
        Text(f"제때옵스(JETTAE Ops) 런북 준비도 진단 — {_clean(summary.target)}", style="bold"),
        _overview(summary, broken, unknown),
        Text(""),
        _services_table(summary.report, plain),
        Text(""),
        *_findings_block(summary, findings),
    ]
    if unknown:
        blocks.append(
            Text(
                f"확인 불가 {unknown}건은 정상으로 보지 않습니다. 사람이 원문·명부와 대조해 주세요.",
                style="magenta",
            )
        )
    failed = _failed_rehearsals(summary.report)
    if failed:
        blocks += [Text(""), *failed]
    return [*blocks, Text(""), *_closing(summary)]


def scan_payload(
    report: DiagnosisReport,
    *,
    target: str,
    tool_version: str,
    documents: Mapping[str, DocumentRef],
    notes: Sequence[str],
    gate_failures: Sequence[str],
) -> dict[str, Any]:
    """``jettae scan --format json``: the report exactly as the REST API serves it, plus what a
    program needs to use it without the graph — where each document is (paths as seen from the
    working directory), the run's remarks, and the CI gate's verdict."""
    passed, total = pass_rate(report)
    return {
        "tool": {"name": "jettae", "version": tool_version},
        "target": target,
        "summary": {
            "readiness": report.readiness.overall,
            "findings": len(report.findings),
            "critical": sum(1 for f in report.findings if f.severity is Severity.CRITICAL),
            "unknown": sum(1 for f in report.findings if f.status is FindingStatus.UNKNOWN),
            "rehearsals": total,
            "rehearsals_passed": passed,
            "llm_calls": report.llm.calls,
            "judgement_path_llm_calls": report.llm.judgement_path_calls,
            # 규칙 밖 위험 후보 by status — reported, never part of the score or the gate
            "risk_hypotheses": {
                status.value: sum(1 for h in report.risk_hypotheses if h.status is status)
                for status in HypothesisStatus
            },
        },
        "gate": {"passed": not gate_failures, "failures": list(gate_failures)},
        "documents": {
            doc_id: {"path": ref.path, "title": ref.title, "sha256": ref.sha256}
            for doc_id, ref in sorted(documents.items())
        },
        "notes": list(notes),
        "report": report.model_dump(mode="json"),
    }


# --------------------------------------------------------------------------- rules


def _rule_rows(rules: Sequence[RuleInfo]) -> list[tuple[str, str, str, str, str, str]]:
    return [
        (
            rule.rule_id,
            SEVERITY_LABEL_KO[rule.severity],
            ELEMENT_LABEL_KO[rule.element] if rule.element is not None else _MISSING,
            rule.title,
            rule.rationale,
            rule.legal_basis or _MISSING,
        )
        for rule in sorted(rules, key=lambda r: r.rule_id)
    ]


_RULE_HEADERS: Final = ("규칙", "심각도", "요소", "제목", "판정 기준", "법적 근거")


def rules_table(rules: Sequence[RuleInfo], *, plain: bool) -> Table:
    """The published fixed rules — exactly what the judge does, for anyone to audit."""
    table = data_table(
        _RULE_HEADERS, _rule_rows(rules), plain=plain, title=f"제때옵스 고정 규칙 {len(rules)}개"
    )
    table.columns[0].no_wrap = True
    table.columns[1].no_wrap = True
    return table


def _md_cell(text: str) -> str:
    """A Markdown table cell: one line, pipes escaped, no raw HTML."""
    return _clean(text).replace("\\", "\\\\").replace("|", "\\|").replace("<", "&lt;").replace(">", "&gt;")


def rules_markdown(rules: Sequence[RuleInfo]) -> str:
    lines = [
        f"# 제때옵스 고정 규칙 카탈로그 ({len(rules)}개)",
        "",
        RULES_INTRO,
        "",
        "| " + " | ".join(_RULE_HEADERS) + " |",
        "|" + "---|" * len(_RULE_HEADERS),
    ]
    lines += ["| " + " | ".join(_md_cell(cell) for cell in row) + " |" for row in _rule_rows(rules)]
    return "\n".join(lines) + "\n"


def rules_json(rules: Sequence[RuleInfo]) -> str:
    return to_json([rule.model_dump(mode="json") for rule in sorted(rules, key=lambda r: r.rule_id)])


# --------------------------------------------------------------------------- rehearsal


def _rehearsal_pairs(report: DiagnosisReport) -> list[tuple[Scenario | None, RehearsalResult]]:
    """Current result per scenario, failures first, then by title (what to look at first)."""
    scenarios = {s.id: s for s in report.scenarios}
    pairs = [(scenarios.get(current.scenario_id), current) for current, _ in current_rehearsals(report)]
    return sorted(
        pairs,
        key=lambda p: (
            p[1].outcome is Outcome.PASS,
            p[0].title if p[0] else p[1].scenario_id,
            p[1].scenario_id,
        ),
    )


def _rehearsal_entry(scenario: Scenario | None, result: RehearsalResult, service: str) -> Text:
    ok = result.outcome is Outcome.PASS
    entry = Text("통과" if ok else "실패", style="green" if ok else "bold red")
    entry.append(
        "  " + (_clean(scenario.title) if scenario is not None else result.scenario_id), style="bold"
    )
    took = format_minutes(result.time_to_mitigate) if result.time_to_mitigate is not None else "완료 못 함"
    origin = ORIGIN_LABEL_KO[scenario.origin] if scenario is not None else "출처 미상"
    deadline = format_minutes(result.deadline_minutes)
    entry.append(f"\n{origin} · {service or _MISSING} · 대응 {took} / 시한 {deadline}", style="dim")
    if not ok:
        entry.append("\n끊긴 곳: " + (_clip(_clean(result.reason), REASON_CHARS) or "사유 기록 없음"))
    return entry


def rehearsal_renderables(
    report: DiagnosisReport, *, target: str, notes: Sequence[str]
) -> list[RenderableType]:
    """``jettae rehearse``: every drill's verdict from the deterministic replay (never an LLM),
    failures first."""
    passed, total = pass_rate(report)
    names = {s.service_id: s.service_name for s in report.readiness.services}
    # Nothing replayed is not "all passed" (P3): it is shown as a warning, never in green.
    rate_style = "yellow" if not total else "green" if passed == total else "red"
    header = key_value_table(
        [
            ("대상", target),
            ("그래프", f"v{report.graph_version} · {short_hash(report.graph_hash)}"),
            ("리허설", Text(_rate_text(passed, total), style=rate_style)),
            ("판정 방식", judgement_line(report.llm.judgement_path_calls)),
        ]
    )
    blocks: list[RenderableType] = [
        Text(f"제때옵스(JETTAE Ops) 모의 장애 리허설 — {_clean(target)}", style="bold"),
        header,
        Text(""),
    ]
    if not total:
        blocks.append(
            Text(
                "재생할 시나리오가 없습니다(서비스를 찾지 못했거나 시나리오 파일이 비어 있습니다).",
                style="yellow",
            )
        )
    else:
        entries = [
            _rehearsal_entry(scenario, result, _clean(names.get(scenario.service_id, scenario.service_id)))
            if scenario is not None
            else _rehearsal_entry(None, result, "")
            for scenario, result in _rehearsal_pairs(report)
        ]
        blocks += [_heading("시나리오별 결과 (실패 먼저)"), numbered_list(entries)]
    blocks += [Text("[참고] " + _clean(note)) for note in notes]
    return blocks


def rehearsal_payload(
    report: DiagnosisReport, *, target: str, tool_version: str, notes: Sequence[str]
) -> dict[str, Any]:
    """``jettae rehearse --format json``: each scenario with its full replay (timeline included)
    and the result hash that proves reproducibility (P6)."""
    passed, total = pass_rate(report)
    return {
        "tool": {"name": "jettae", "version": tool_version},
        "target": target,
        "graph_version": report.graph_version,
        "graph_hash": report.graph_hash,
        "passed": passed,
        "total": total,
        "pass_rate": round(passed / total, 4) if total else None,
        "judgement_path_llm_calls": report.llm.judgement_path_calls,
        "results": [
            {
                "scenario": scenario.model_dump(mode="json") if scenario is not None else None,
                "result": result.model_dump(mode="json"),
                "result_hash": result.result_hash(),
            }
            for scenario, result in _rehearsal_pairs(report)
        ],
        "notes": list(notes),
    }
