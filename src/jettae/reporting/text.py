"""Plain-text diagnosis summary for the terminal (``jettae scan``).

Why a separate renderer: the CLI needs the answer at a glance — the score, the worst broken paths
and how many rehearsals failed — not the full evidence report. The summary is plain text without
colour or cursor codes, so it reads the same in a CI log, a Windows console and a pager. It is a
``str``, not bytes: runbook text and the em dash (U+2014) used throughout have no cp949 encoding, so
the caller writes it as UTF-8 (or with ``errors="replace"`` on a legacy code page), never through a
strict cp949 stream.

Two details make it safe and tidy in a terminal:

* Untrusted text (runbook-derived names, finding messages, scenario titles) goes through
  :func:`~jettae.reporting.evidence.printable`, so an ANSI escape or bidi override in a runbook
  cannot recolour, erase or reorder the operator's terminal (P5).
* Lines are wrapped by *display width*, not by character count: Hangul occupies two terminal
  columns. East Asian "ambiguous" characters (``·``, ``●``, ``—``) are counted as two as well,
  which is how Korean-locale consoles draw them, so no line ever exceeds ``width`` columns there
  (in Western terminals lines simply end a little early).

Like every renderer it only reports; the verdicts and numbers come from the fixed rules and the
deterministic replay (P1), and the output is a pure function of the report (P6).
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Sequence
from typing import Final

from jettae.check.registry import sort_findings
from jettae.domain import (
    ELEMENT_LABEL_KO,
    DiagnosisReport,
    Element,
    Finding,
    FindingStatus,
    Outcome,
    Severity,
)
from jettae.reporting.evidence import (
    ELEMENT_STATUS_LABEL_KO,
    ELEMENT_STATUS_SYMBOL,
    PATCH_STATUS_LABEL_KO,
    SEVERITY_LABEL_KO,
    current_rehearsals,
    format_datetime,
    judgement_path_statement,
    llm_usage_line,
    printable,
    service_rows,
    short_hash,
)

__all__ = ["DEFAULT_TOP", "MIN_WIDTH", "display_width", "render_cli_summary", "wrap_display"]

#: Narrower than this, the label column and hanging indents leave no room for content.
MIN_WIDTH: Final = 40
#: Broken findings listed by default; the rest are counted and left to the evidence report.
DEFAULT_TOP: Final = 10
#: Failed rehearsals listed by name; the rest are counted.
FAILED_REHEARSALS_SHOWN: Final = 5
#: Lines of explanation shown under a finding or a failed rehearsal.
DETAIL_LINES: Final = 2

_LABEL_COLUMNS: Final = 12
_NAME_COLUMNS: Final = 16
#: Room a service row needs after its head for the widest single mark ("대체채널○" = 10 columns).
_MIN_MARK_COLUMNS: Final = 10
_ELLIPSIS: Final = "…"
_NOT_EVALUATED: Final = "-"


# --------------------------------------------------------------------------- display width


def _char_width(ch: str) -> int:
    if unicodedata.combining(ch):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F", "A") else 1


def display_width(text: str) -> int:
    """Terminal columns ``text`` occupies (wide and ambiguous East Asian characters count two)."""
    return sum(_char_width(ch) for ch in text)


def _take(word: str, columns: int) -> tuple[str, str]:
    """Split ``word`` after as many characters as fit in ``columns`` — at least one, so wrapping
    always makes progress even when one wide character is wider than the room left."""
    used = 0
    for index, ch in enumerate(word):
        used += _char_width(ch)
        if used > columns:
            cut = max(index, 1)
            return word[:cut], word[cut:]
    return word, ""


def wrap_display(text: str, width: int, *, first: str = "", rest: str = "") -> list[str]:
    """Greedy word wrap by display width; ``first``/``rest`` prefix the first/following lines.

    Korean separates words (어절) with spaces, so breaking at spaces keeps words whole; a single
    word longer than a line (a long id or URL) is broken inside the word. Whitespace in ``text``
    is collapsed and invisible control characters are dropped. No line exceeds ``width`` as long
    as ``width`` leaves two columns after each prefix (one wide character); with less room a
    character is still placed, so wrapping always terminates — even when a prefix alone is wider
    than ``width`` (each placed character consumes the word, never the prefix).
    """
    lines: list[str] = []
    prefix, current = first, ""
    for word in printable(text).split():
        candidate = f"{current} {word}" if current else word
        if display_width(prefix + candidate) <= width:
            current = candidate
            continue
        if current:
            lines.append(prefix + current)
            prefix, current = rest, ""
        # The remainder's width is kept as a running total: re-measuring the whole rest of a long
        # word after every cut made one 100k-character word (a pasted log line) quadratic.
        remainder, remaining = word, display_width(word)
        while remainder and display_width(prefix) + remaining > width:
            head, remainder = _take(remainder, width - display_width(prefix))
            remaining -= display_width(head)
            lines.append(prefix + head)
            prefix = rest
        current = remainder
    if current or not lines:
        lines.append(prefix + current)
    return lines


def _fit(text: str, columns: int) -> str:
    """``text`` cut to at most ``columns`` display columns, ending in an ellipsis if cut.

    One left-to-right pass: the kept part is the longest prefix that leaves room for the ellipsis.
    (Dropping one character at a time and re-measuring was quadratic, so a catalog service name of
    a few ten thousand characters stalled the summary for minutes.)
    """
    if display_width(text) <= columns:
        return text
    room = columns - display_width(_ELLIPSIS)
    used = cut = 0
    for index, ch in enumerate(text):
        used += _char_width(ch)
        if used > room:
            break
        cut = index + 1
    return text[:cut].rstrip() + _ELLIPSIS


def _clip_lines(lines: list[str], limit: int, width: int) -> list[str]:
    """Keep ``limit`` lines; when text was cut, the last kept line ends with an ellipsis."""
    if len(lines) <= limit:
        return lines
    return [*lines[: limit - 1], _fit(lines[limit - 1] + " " + _ELLIPSIS, width)]


def _pad(text: str, columns: int) -> str:
    """Left-align ``text`` in exactly ``columns`` display columns (clipped with an ellipsis)."""
    fitted = _fit(" ".join(printable(text).split()), columns)
    return fitted + " " * (columns - display_width(fitted))


def _field(label: str, value: str, width: int) -> list[str]:
    return wrap_display(value, width, first=_pad(label, _LABEL_COLUMNS), rest=" " * _LABEL_COLUMNS)


def _heading(text: str, width: int) -> list[str]:
    return ["", *wrap_display(text, width)]


# --------------------------------------------------------------------------- blocks


def _overview(report: DiagnosisReport, width: int, broken: Sequence[Finding], unknown: int) -> list[str]:
    readiness = report.readiness
    severity = Counter(f.severity for f in broken)
    by_severity = " · ".join(f"{SEVERITY_LABEL_KO[s]} {severity[s]}" for s in Severity if severity[s])
    pairs = current_rehearsals(report)
    passed = sum(1 for current, _ in pairs if current.outcome is Outcome.PASS)
    rate = f"{100 * passed / len(pairs):.1f}%" if pairs else "해당 없음"
    scoring = f"SCORING_{readiness.weights_version.upper()}"
    created = format_datetime(report.created_at)
    graph = f"v{report.graph_version} · {short_hash(report.graph_hash)} · 진단 {created}"
    lines: list[str] = []
    if report.llm.judgement_path_calls:
        lines += wrap_display("경고: " + judgement_path_statement(report.llm), width)
    lines += _field("전체 준비도", f"{readiness.overall:.1f} / 100 ({scoring})", width)
    lines += _field("그래프", graph, width)
    lines += _field(
        "끊긴 경로",
        f"{len(broken)}건" + (f" ({by_severity})" if by_severity else "") + f" · 확인 불가 {unknown}건",
        width,
    )
    lines += _field("리허설", f"{len(pairs)}건 중 {passed}건 통과 (통과율 {rate})", width)
    lines += _field("LLM", llm_usage_line(report.llm), width)
    if not report.llm.judgement_path_calls:
        lines += _field("판정 경로", judgement_path_statement(report.llm), width)
    return lines


def _services(report: DiagnosisReport, width: int) -> list[str]:
    legend = " ".join(f"{ELEMENT_STATUS_SYMBOL[s]} {label}" for s, label in ELEMENT_STATUS_LABEL_KO.items())
    lines = _heading(f"[서비스별 준비도] {legend} {_NOT_EVALUATED} 평가 기록 없음", width)
    rows = service_rows(report)
    if not rows:
        return [*lines, "  점수를 매긴 서비스가 없습니다."]
    for row in rows:
        # A mark is one word ("대체채널○"), so wrapping never separates a status from its label.
        marks = " ".join(
            ELEMENT_LABEL_KO[element].replace(" ", "")
            + (ELEMENT_STATUS_SYMBOL[status] if status is not None else _NOT_EVALUATED)
            for element, status in zip(Element, row.statuses, strict=True)
        )
        head = f"  {_pad(row.name, _NAME_COLUMNS)} {row.tier}등급 {row.score:5.1f}  "
        indent = display_width(head)
        if indent + _MIN_MARK_COLUMNS <= width:
            lines += wrap_display(marks, width, first=head, rest=" " * indent)
        else:
            # The tier is an unbounded integer from the catalog; when the row head leaves no room,
            # the head and the marks get lines of their own instead of overflowing the terminal.
            lines += wrap_display(head, width, first="  ", rest="    ")
            lines += wrap_display(marks, width, first="    ", rest="    ")
    return lines


def _findings(
    report: DiagnosisReport, width: int, broken: Sequence[Finding], unknown: int, top: int
) -> list[str]:
    shown = list(broken[:top])
    title = f"[주요 끊김] 전체 {len(broken)}건" + (
        f" 중 상위 {len(shown)}건" if len(shown) < len(broken) else ""
    )
    lines = _heading(title, width)
    if not broken:
        lines.append("  고정 규칙이 찾은 끊김이 없습니다.")
    names = {s.service_id: s.service_name for s in report.readiness.services}
    number_columns = len(str(len(shown)))
    for index, finding in enumerate(shown, start=1):
        service = (
            "조직 전체" if finding.service_id is None else names.get(finding.service_id, finding.service_id)
        )
        head = f"  {index:>{number_columns}}. "
        indent = " " * display_width(head)
        label = f"[{SEVERITY_LABEL_KO[finding.severity]}] {finding.rule_id} {finding.title} — {service}"
        lines += wrap_display(label, width, first=head, rest=indent)
        detail = wrap_display(finding.message, width, first=indent, rest=indent)
        lines += _clip_lines(detail, DETAIL_LINES, width)
    if len(shown) < len(broken):
        lines += wrap_display(
            f"… 외 {len(broken) - len(shown)}건은 증적 리포트에서 확인하세요.", width, first="  ", rest="  "
        )
    if unknown:
        lines += wrap_display(
            f"확인 불가 {unknown}건: 정상으로 보지 않으며 사람이 원문·명부와 대조해야 합니다.",
            width,
            first="  ",
            rest="  ",
        )
    return lines


def _rehearsals(report: DiagnosisReport, width: int) -> list[str]:
    titles = {s.id: s.title for s in report.scenarios}
    failed = [current for current, _ in current_rehearsals(report) if current.outcome is Outcome.FAIL]
    if not failed:
        return []
    lines = _heading(f"[리허설 실패] {len(failed)}건", width)
    for result in failed[:FAILED_REHEARSALS_SHOWN]:
        text = f"{titles.get(result.scenario_id, result.scenario_id)}: {result.reason or '사유 기록 없음'}"
        lines += _clip_lines(wrap_display(text, width, first="  - ", rest="    "), DETAIL_LINES, width)
    if len(failed) > FAILED_REHEARSALS_SHOWN:
        lines += wrap_display(f"… 외 {len(failed) - FAILED_REHEARSALS_SHOWN}건", width, first="  ", rest="  ")
    return lines


def _notes(report: DiagnosisReport, width: int) -> list[str]:
    lines: list[str] = []
    if report.patches:
        counts = Counter(p.status for p in report.patches)
        summary = " · ".join(f"{PATCH_STATUS_LABEL_KO[s]} {n}건" for s, n in sorted(counts.items()))
        lines += wrap_display(f"[수정안] {summary} (사람이 승인한 것만 새 그래프 버전이 됩니다)", width)
    if report.held_element_ids:
        lines += wrap_display(
            f"[판단 보류] {len(set(report.held_element_ids))}건: 사람이 원문과 대조해 확인해야 합니다.", width
        )
    if report.security:
        lines += wrap_display(
            f"[보안 알림] 문서 속 AI 지시문 {len(report.security)}건을 격리했습니다. "
            "판정·점수에는 영향이 없습니다.",
            width,
        )
    return ["", *lines] if lines else []


def render_cli_summary(report: DiagnosisReport, *, width: int = 100, top: int = DEFAULT_TOP) -> str:
    """Plain-text summary for the terminal: score, top broken paths, rehearsal pass rate.

    ``width`` is the terminal width in columns (at least :data:`MIN_WIDTH`); ``top`` caps how many
    broken findings are listed (most severe first). Raises ``ValueError`` for impossible values.
    """
    if width < MIN_WIDTH:
        raise ValueError(f"width는 {MIN_WIDTH} 이상이어야 합니다: {width}")
    if top < 0:
        raise ValueError(f"top은 0 이상이어야 합니다: {top}")
    findings = sort_findings(report.findings)
    broken = [f for f in findings if f.status is FindingStatus.BROKEN]
    unknown = sum(1 for f in findings if f.status is FindingStatus.UNKNOWN)
    lines = [
        *wrap_display("제때옵스(JETTAE Ops) 런북 준비도 진단 요약", width),
        "=" * width,
        *_overview(report, width, broken, unknown),
        *_services(report, width),
        *_findings(report, width, broken, unknown, top),
        *_rehearsals(report, width),
        *_notes(report, width),
        "",
        *wrap_display(
            "자세한 근거(원문 인용·법적 근거·인증 기준 매핑)는 증적 리포트(Markdown·HTML)에서 확인하세요.",
            width,
        ),
    ]
    return "\n".join(line.rstrip() for line in lines) + "\n"
