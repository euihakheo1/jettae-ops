"""Printable HTML evidence report — the same evidence as the Markdown report, laid out for A4.

Why a separate HTML renderer instead of converting the Markdown: auditors and executives print the
report or save it as PDF, which needs page breaks, colour-coded severity and a cover summary that
Markdown cannot express. Rendering straight from the shared :class:`EvidenceView` (no Markdown
library in between) keeps both formats in lock-step — the view decides *what* is shown, this module
only decides *how* — and keeps a third-party parser out of the escaping path.

Safety (P5): every string that can come from a runbook, the roster, a scenario or an LLM-drafted
patch is untrusted. Fragments are typed as :data:`Html`; the only ways to make one are :func:`esc`
(escapes text) and literal markup written in this module, so mypy rejects raw text where markup is
expected. The page is self-contained — inline CSS, no script, no external fonts or images — so it
renders identically offline and cannot phone home. Output is deterministic (P6).
"""

from __future__ import annotations

import html
from collections.abc import Iterable, Sequence
from typing import Final, NewType

from jettae.domain import (
    ELEMENT_LABEL_KO,
    DiagnosisReport,
    DocumentRef,
    Element,
    ElementStatus,
    FindingStatus,
    GraphVersion,
    HypothesisStatus,
    OpsGraph,
    Outcome,
    Patch,
    Severity,
)
from jettae.reporting.evidence import (
    CONTROLS_INTRO,
    DISCLAIMER,
    DROPPED_EXAMPLES_TITLE,
    ELEMENT_STATUS_LABEL_KO,
    ELEMENT_STATUS_SYMBOL,
    EXPLORE_INTRO,
    HISTORY_NOTE,
    NO_QUOTE_NOTE,
    OUTCOME_LABEL_KO,
    RULES_INTRO,
    SCORING_INTRO,
    SCORING_METHOD,
    SECTION_TITLES,
    SECURITY_INTRO,
    SEVERITY_LABEL_KO,
    STATUS_LEGEND,
    TIER_FALLBACK_NOTE,
    UNDECIDED_INTRO,
    UNLINKED_PATCHES_TITLE,
    ControlStatus,
    EvidenceQuote,
    EvidenceView,
    FindingView,
    HypothesisRow,
    build_evidence,
    printable,
)

__all__ = ["FONT_STACK", "Html", "esc", "render_html", "render_view_html"]

Html = NewType("Html", str)
"""A fragment of markup that is safe to concatenate: built only by :func:`esc` or this module."""

FONT_STACK: Final = 'Pretendard, "Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans KR", sans-serif'

_DASH: Final = Html("—")

_ELEMENT_STATUS_CLASS: Final[dict[ElementStatus, str]] = {
    ElementStatus.PASS: "st-pass",
    ElementStatus.PARTIAL: "st-partial",
    ElementStatus.FAIL: "st-fail",
    ElementStatus.UNKNOWN: "st-unknown",
}

_HYPOTHESIS_CLASS: Final[dict[HypothesisStatus, str]] = {
    HypothesisStatus.CONFIRMED: "out-fail",
    HypothesisStatus.NEEDS_INFO: "chip-unknown",
    HypothesisStatus.REFUTED: "out-pass",
}

_CONTROL_CLASS: Final[dict[ControlStatus, str]] = {
    ControlStatus.OK: "ctl-ok",
    ControlStatus.GAP: "ctl-gap",
    ControlStatus.MISSING: "ctl-missing",
}

_CSS: Final = """
:root {
  --ink: #1d2433; --muted: #5b6475; --line: #d5dae3; --soft: #f4f6f9; --accent: #1f4fd1;
  --crit: #b42318; --crit-bg: #fee4e2; --high: #c4320a; --high-bg: #ffead5;
  --med: #93530a; --med-bg: #fef0c7; --low: #175cd3; --low-bg: #d1e9ff;
  --info: #475467; --info-bg: #eaecf0; --pass: #067647; --pass-bg: #dcfae6;
  --unk: #5925dc; --unk-bg: #ebe9fe;
}
* { box-sizing: border-box; }
html { color-scheme: light; -webkit-text-size-adjust: 100%; }
body {
  margin: 0; background: #ffffff; color: var(--ink);
  font-family: FONT_STACK;
  font-size: 10.5pt; line-height: 1.6; word-break: keep-all; overflow-wrap: anywhere;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
.doc { max-width: 210mm; margin: 0 auto; padding: 16mm 14mm; }
h1 { font-size: 21pt; line-height: 1.3; margin: 0 0 4pt; }
h1 small { font-size: 12pt; font-weight: 600; color: var(--muted); }
h2 { font-size: 15pt; line-height: 1.35; margin: 0 0 10pt; padding-bottom: 4pt;
     border-bottom: 2px solid var(--ink); }
h3 { font-size: 12pt; margin: 16pt 0 6pt; }
h4 { font-size: 11pt; margin: 0 0 6pt; line-height: 1.45; }
p { margin: 0 0 8pt; }
section { margin-top: 22pt; }
table { width: 100%; border-collapse: collapse; margin: 6pt 0 12pt; font-size: 9pt; }
th, td { border: 1px solid var(--line); padding: 3pt 5pt; text-align: left; vertical-align: top; }
thead th { background: var(--soft); font-weight: 700; }
th.key, td.key { white-space: nowrap; width: 1%; }
td.num { text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }
td.st, th.st { text-align: center; }
td.st { font-size: 11pt; line-height: 1.3; }
.st-pass { color: var(--pass); } .st-partial { color: var(--med); }
.st-fail { color: var(--crit); } .st-unknown { color: var(--unk); font-weight: 700; }
.st-none { color: var(--muted); }
code { font-family: ui-monospace, "D2Coding", Consolas, "Courier New", monospace; font-size: 0.9em;
       background: var(--soft); padding: 0 3pt; border-radius: 3px; }
.chip { display: inline-block; padding: 0 6pt; border: 1px solid currentColor; border-radius: 999px;
        font-size: 8.5pt; font-weight: 700; line-height: 1.7; white-space: nowrap; }
.sev-critical { color: var(--crit); background: var(--crit-bg); }
.sev-high { color: var(--high); background: var(--high-bg); }
.sev-medium { color: var(--med); background: var(--med-bg); }
.sev-low { color: var(--low); background: var(--low-bg); }
.sev-info { color: var(--info); background: var(--info-bg); }
.out-pass, .ctl-ok { color: var(--pass); background: var(--pass-bg); }
.out-fail, .ctl-gap { color: var(--crit); background: var(--crit-bg); }
.ctl-missing, .chip-unknown { color: var(--unk); background: var(--unk-bg); }
.eyebrow { color: var(--muted); font-size: 9pt; font-weight: 600; margin: 0 0 6pt; }
.org { font-size: 12pt; font-weight: 700; margin: 0 0 10pt; }
.kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 6pt; margin: 10pt 0 12pt; }
.kpi { border: 1px solid var(--line); border-radius: 6px; padding: 6pt 8pt; }
.kpi-label { display: block; color: var(--muted); font-size: 8.5pt; font-weight: 600; }
.kpi-value { display: block; font-size: 17pt; font-weight: 800; line-height: 1.3;
             font-variant-numeric: tabular-nums; }
.kpi-value small { font-size: 9pt; font-weight: 600; color: var(--muted); }
.kpi-sub { display: block; color: var(--muted); font-size: 8.5pt; }
.notice { border-left: 4px solid var(--accent); background: var(--soft); padding: 6pt 10pt;
          margin: 0 0 10pt; }
.alert { border-left-color: var(--crit); background: var(--crit-bg); font-weight: 700; }
.toc { border: 1px solid var(--line); border-radius: 6px; padding: 6pt 10pt; }
.toc-title { font-weight: 700; margin: 0 0 2pt; }
.toc ul { margin: 0; padding: 0; list-style: none; columns: 2; column-gap: 18pt; }
.toc a { color: inherit; text-decoration: none; }
.finding { border: 1px solid var(--line); border-left: 4px solid var(--info); border-radius: 4px;
           padding: 7pt 10pt; margin: 0 0 10pt; }
.finding.f-critical { border-left-color: var(--crit); }
.finding.f-high { border-left-color: var(--high); }
.finding.f-medium { border-left-color: var(--med); }
.finding.f-low { border-left-color: var(--low); }
.finding.f-unknown { border-left-color: var(--unk); }
.finding h4 .chip { margin-right: 4pt; vertical-align: 1pt; }
.rule { font-family: ui-monospace, "D2Coding", Consolas, monospace; font-size: 9.5pt; }
dl.meta { display: grid; grid-template-columns: max-content 1fr; gap: 1pt 10pt; margin: 0 0 6pt;
          font-size: 9pt; }
dl.meta dt { color: var(--muted); }
dl.meta dd { margin: 0; }
blockquote.quote { margin: 4pt 0 6pt; padding: 4pt 10pt; border-left: 3px solid var(--line);
                   background: var(--soft); }
blockquote.quote p { margin: 0; }
blockquote.quote cite { display: block; color: var(--muted); font-size: 8.5pt; font-style: normal; }
.legal { font-size: 9pt; color: var(--muted); }
.empty { color: var(--muted); }
ul.plain { margin: 0 0 8pt; padding-left: 16pt; }
footer { margin-top: 20pt; padding-top: 6pt; border-top: 1px solid var(--line); color: var(--muted);
         font-size: 8.5pt; }
@page { size: A4; margin: 16mm 14mm 18mm; }
@media print {
  .doc { max-width: none; padding: 0; }
  section { margin-top: 0; }
  section.break { break-before: page; }
  h2, h3, h4 { break-after: avoid; }
  tr, .kpi, blockquote, dl.meta { break-inside: avoid; }
  thead { display: table-header-group; }
}
@media screen and (max-width: 640px) {
  .doc { padding: 16px; }
  .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .toc ul { columns: 1; }
  table { display: block; overflow-x: auto; }
}
""".replace("FONT_STACK", FONT_STACK).strip()


# --------------------------------------------------------------------------- primitives


def esc(text: str) -> Html:
    """Escape untrusted text for an HTML text node or a quoted attribute value.

    Whitespace runs (line breaks included) collapse to one space, exactly as in the Markdown
    report, so both formats carry the same text rather than merely looking alike once a browser
    collapses HTML whitespace. Multi-line quotes keep their breaks through :func:`_lines`.
    """
    return Html(html.escape(" ".join(printable(text).split()), quote=True))


def _lines(text: str) -> Html:
    """Escape multi-line text, keeping its line breaks (runbook quotes)."""
    return Html("<br>".join(esc(line) for line in text.split("\n")))


def _join(parts: Iterable[Html], sep: str = "") -> Html:
    return Html(sep.join(parts))


def _chip(label: str, css: str) -> Html:
    return Html(f'<span class="chip {esc(css)}">{esc(label)}</span>')


def _code(text: str) -> Html:
    """Inline code; blank text gives no markup, so a table cell shows its dash as in Markdown."""
    body = esc(text)
    return Html(f"<code>{body}</code>") if body else Html("")


def _p(content: Html, css: str = "") -> Html:
    attr = f' class="{esc(css)}"' if css else ""
    return Html(f"<p{attr}>{content}</p>")


def _ul(items: Iterable[str]) -> Html:
    return Html('<ul class="plain">' + "".join(f"<li>{esc(x)}</li>" for x in items) + "</ul>")


def _h3(text: str) -> Html:
    return Html(f"<h3>{esc(text)}</h3>")


def _table(headers: Sequence[str], rows: Iterable[Sequence[Html]], *, classes: Sequence[str] = ()) -> Html:
    """A table; ``classes`` (one per column, may be shorter) style both header and body cells."""

    def attr(index: int) -> str:
        css = classes[index] if index < len(classes) else ""
        return f' class="{esc(css)}"' if css else ""

    head = "".join(f"<th{attr(i)}>{esc(h)}</th>" for i, h in enumerate(headers))
    body = "\n".join(
        "<tr>" + "".join(f"<td{attr(i)}>{cell or _DASH}</td>" for i, cell in enumerate(row)) + "</tr>"
        for row in rows
    )
    return Html(f"<table>\n<thead><tr>{head}</tr></thead>\n<tbody>\n{body}\n</tbody>\n</table>")


def _section(key: str, body: Sequence[Html], *, title_suffix: str = "", page_break: bool = True) -> Html:
    """A chapter; every chapter but the first starts on a new printed page."""
    css = ' class="break"' if page_break else ""
    inner = "\n".join(body)
    title = esc(SECTION_TITLES[key] + title_suffix)
    return Html(f'<section id="{esc(key)}"{css}>\n<h2>{title}</h2>\n{inner}\n</section>')


# --------------------------------------------------------------------------- sections


def _kpi(label: str, value: Html, sub: str) -> Html:
    return Html(
        f'<div class="kpi"><span class="kpi-label">{esc(label)}</span>'
        f'<span class="kpi-value">{value}</span><span class="kpi-sub">{esc(sub)}</span></div>'
    )


def _cover(view: EvidenceView) -> Html:
    parts: list[Html] = [
        Html('<header class="cover">'),
        _p(esc("제때옵스(JETTAE Ops) · 고정 규칙과 결정적 리플레이로 자동 생성"), "eyebrow"),
        Html("<h1>런북 준비도 진단 리포트 <small>(ISMS-P 증적 초안)</small></h1>"),
    ]
    if view.org_name:
        parts.append(_p(esc(view.org_name), "org"))
    parts += [
        Html('<div class="kpis">'),
        _kpi("전체 준비도", Html(f"{view.overall:.1f} <small>/ 100</small>"), view.scoring_name),
        _kpi("끊긴 대응 경로", esc(f"{view.broken_count}건"), view.severity_summary or "없음"),
        _kpi(
            "리허설 통과율",
            esc(view.rehearsal_rate),
            f"{len(view.rehearsals)}건 중 {view.rehearsal_passed}건 통과",
        ),
        _kpi("판정 경로 LLM 호출", esc(f"{view.judgement_path_calls:,}회"), "판정·점수·리허설"),
        Html("</div>"),
    ]
    if view.judgement_path_calls:
        parts.append(_p(esc(view.judgement_path), "notice alert"))
    parts.append(_p(esc(DISCLAIMER), "notice"))
    toc = "".join(f'<li><a href="#{esc(key)}">{esc(title)}</a></li>' for key, title in SECTION_TITLES.items())
    parts.append(
        Html(f'<nav class="toc" aria-label="목차"><p class="toc-title">목차</p><ul>{toc}</ul></nav>')
    )
    parts.append(Html("</header>"))
    return _join(parts, "\n")


def _overview(view: EvidenceView) -> Html:
    rows: list[tuple[Html, Html]] = []
    if view.org_name:
        rows.append((esc("조직"), esc(view.org_name)))
    rows += [
        (esc("워크스페이스"), _code(view.workspace_id)),
        (esc("진단 실행 ID"), _code(view.run_id)),
        (
            esc("전체 준비도"),
            Html(f"<strong>{view.overall:.1f} / 100</strong> {esc(f'(산식 {view.scoring_name})')}"),
        ),
        (esc("그래프 버전"), Html(f"v{view.graph_version} · 해시 {_code(view.graph_hash)}")),
        (esc("진단 시각"), esc(view.created_at)),
        (esc("끊긴 대응 경로"), esc(view.broken_summary)),
        (esc("리허설"), esc(view.rehearsal_summary)),
        (esc("원문 근거"), esc(view.grounding)),
        (esc("LLM 사용"), esc(view.llm_usage)),
        (esc("판정 경로"), Html(f"<strong>{esc(view.judgement_path)}</strong>")),
    ]
    body: list[Html] = [_table(("항목", "내용"), rows, classes=("key",)), _h3("진단 대상 문서")]
    if view.documents:
        body.append(
            _table(
                ("문서", "경로", "SHA-256(앞 12자리)"),
                ((esc(d.title), _code(d.path) if d.path else _DASH, _code(d.sha256)) for d in view.documents),
            )
        )
    else:
        body.append(_p(esc("진단에 쓴 문서 기록이 없습니다."), "empty"))
    return _section("overview", body, page_break=False)


def _status_cell(status: ElementStatus | None) -> Html:
    if status is None:
        return Html('<span class="st-none" title="평가 기록 없음">—</span>')
    label = esc(ELEMENT_STATUS_LABEL_KO[status])
    symbol = esc(ELEMENT_STATUS_SYMBOL[status])
    return Html(
        f'<span class="{_ELEMENT_STATUS_CLASS[status]}" title="{label}" aria-label="{label}">{symbol}</span>'
    )


def _services(view: EvidenceView) -> Html:
    body: list[Html] = [_p(esc(STATUS_LEGEND))]
    if view.services:
        headers = ("서비스", "등급", "점수", *(ELEMENT_LABEL_KO[e] for e in Element))
        rows = (
            (
                esc(s.name),
                esc(f"{s.tier}({s.tier_label})"),
                esc(f"{s.score:.1f}"),
                *(_status_cell(st) for st in s.statuses),
            )
            for s in view.services
        )
        body.append(_table(headers, rows, classes=("", "", "num", *("st" for _ in Element))))
    else:
        body.append(_p(esc("점수를 매긴 서비스가 없습니다."), "empty"))
    if view.catalog_gaps:
        body.append(_p(esc("카탈로그에는 있지만 담당·런북이 없는 서비스: " + ", ".join(view.catalog_gaps))))
    return _section("services", body)


def _quote(quote: EvidenceQuote) -> Html:
    return Html(
        f'<blockquote class="quote"><p>{_lines(quote.text)}</p>'
        f"<cite>— {esc(quote.location)}</cite></blockquote>"
    )


def _finding(view: FindingView) -> Html:
    f = view.finding
    unknown = f.status is FindingStatus.UNKNOWN
    chips = _chip(view.severity, f"sev-{f.severity.value}")
    if unknown:
        chips = Html(chips + _chip(view.status, "chip-unknown"))
    css = "f-unknown" if unknown else f"f-{f.severity.value}"
    anchor = "f-" + view.number.replace(".", "-")
    heading = Html(
        f'<h4>{chips}{esc(view.number)} <span class="rule">{esc(f.rule_id)}</span> '
        f"{esc(f.title)} — {esc(view.service)}</h4>"
    )
    meta: list[tuple[str, Html]] = [
        ("판정 ID", _code(f.id)),
        ("심각도·상태", esc(f"{view.severity} · {view.status}")),
        ("서비스", esc(view.service + (f" · 요소: {view.element}" if view.element else ""))),
    ]
    if view.subjects:
        meta.append(("관련 요소", esc(view.subjects)))
    if view.fix:
        meta.append(("권장 조치", esc(view.fix)))
    meta_html = "".join(f"<dt>{esc(k)}</dt><dd>{v}</dd>" for k, v in meta)
    parts: list[Html] = [
        Html(f'<article class="finding {esc(css)}" id="{esc(anchor)}">'),
        heading,
        Html(f'<dl class="meta">{meta_html}</dl>'),
        _p(esc(f.message)),
    ]
    if view.quotes:
        parts.extend(_quote(q) for q in view.quotes)
    else:
        parts.append(_p(esc(NO_QUOTE_NOTE), "empty"))
    if f.legal_basis:
        parts.append(_p(esc(f"법적 근거: {f.legal_basis}"), "legal"))
    parts.append(Html("</article>"))
    return _join(parts, "\n")


def _severity_chip(severity: Severity) -> Html:
    return _chip(SEVERITY_LABEL_KO[severity], f"sev-{severity.value}")


def _broken(view: EvidenceView) -> Html:
    body: list[Html] = []
    if not view.broken:
        body.append(_p(esc("고정 규칙이 찾은 끊김이 없습니다."), "empty"))
    for severity, group in view.broken:
        body.append(Html(f"<h3>{_severity_chip(severity)} {esc(f'{len(group)}건')}</h3>"))
        body.extend(_finding(item) for item in group)
    if view.unknown_count:
        body.append(_p(esc(f"확인 불가로 판정한 {view.unknown_count}건은 6장에 따로 적었습니다.")))
    return _section("broken", body)


def _outcome_chip(outcome: Outcome) -> Html:
    return _chip(OUTCOME_LABEL_KO[outcome], "out-pass" if outcome is Outcome.PASS else "out-fail")


def _rehearsals(view: EvidenceView) -> Html:
    if not view.rehearsals:
        return _section("rehearsals", [_p(esc("실행한 모의 장애 리허설이 없습니다."), "empty")])
    rows = (
        (
            esc(r.title),
            esc(r.service),
            _outcome_chip(r.outcome),
            esc(r.deadline),
            esc(r.ended_at),
            esc(r.broken_at) if r.broken_at else _DASH,
            esc(r.comparison) if r.comparison else _DASH,
        )
        for r in view.rehearsals
    )
    body: list[Html] = [
        _p(esc(view.rehearsal_intro)),
        _table(
            ("시나리오", "서비스", "결과", "시한", "종료 시각", "끊긴 지점", "수정 전→후"),
            rows,
            classes=("", "", "st", "num", "num"),
        ),
    ]
    failures = [f"{r.title}: {r.reason}" for r in view.rehearsals if r.outcome is Outcome.FAIL and r.reason]
    if failures:
        body += [_h3("실패 사유"), _ul(failures)]
    return _section("rehearsals", body)


def _history(view: EvidenceView) -> Html:
    body: list[Html] = []
    if view.versions:
        version_rows = (
            (
                esc(f"v{v.number}"),
                esc(v.change),
                esc(v.sentence) if v.sentence else _DASH,
                esc(v.approver),
                esc(v.at),
                esc(v.resolved) if v.resolved else _DASH,
            )
            for v in view.versions
        )
        body.append(
            _table(
                ("버전", "변경", "수정안 문장", "승인자", "시각", "해결한 끊김"),
                version_rows,
                classes=("key",),
            )
        )
    else:
        body.append(_p(esc("그래프 버전 기록이 없습니다."), "empty"))
    if view.other_patches:
        patch_rows = (
            (
                _code(p.patch_id),
                esc(p.status),
                esc(p.author),
                esc(p.sentence),
                esc(p.targets),
                esc(p.decided) if p.decided else _DASH,
            )
            for p in view.other_patches
        )
        body += [
            _h3(UNLINKED_PATCHES_TITLE),
            _table(
                ("수정안", "상태", "작성", "수정안 문장", "대상 끊김", "결정"), patch_rows, classes=("key",)
            ),
        ]
    body.append(_p(esc(HISTORY_NOTE)))
    return _section("history", body)


def _undecided(view: EvidenceView) -> Html:
    body: list[Html] = [_p(esc(UNDECIDED_INTRO))]
    if not (view.unknown or view.held or view.unknown_elements or view.dropped_examples):
        body.append(_p(esc("판단 보류·확인 불가 항목이 없습니다."), "empty"))
        return _section("undecided", body)
    if view.unknown:
        body.append(_h3(f"확인 불가 판정 ({len(view.unknown)}건)"))
        body.extend(_finding(item) for item in view.unknown)
    if view.unknown_elements:
        element_rows = (
            (esc(u.service), esc(u.element), esc(" / ".join(u.reasons)) if u.reasons else _DASH)
            for u in view.unknown_elements
        )
        body += [_h3("확인 불가 요소 (점수 0점 처리)"), _table(("서비스", "요소", "사유"), element_rows)]
    if view.held:
        held_rows = ((esc(h.label), esc(h.kind), esc(h.reason), _code(h.ref_id)) for h in view.held)
        body += [
            _h3(f"판단 보류 요소 ({len(view.held)}건)"),
            _table(("요소", "종류", "보류 사유", "식별자"), held_rows),
        ]
    if view.dropped_examples:
        body += [_h3(DROPPED_EXAMPLES_TITLE), _ul(view.dropped_examples)]
    return _section("undecided", body)


def _security(view: EvidenceView) -> Html:
    body: list[Html] = [_p(esc(SECURITY_INTRO))]
    if view.security:
        rows = ((esc(s.document), esc(s.location), esc(s.patterns), _code(s.excerpt)) for s in view.security)
        body.append(_table(("문서", "위치", "탐지 유형", "격리된 지시문 발췌"), rows, classes=("", "key")))
    else:
        body.append(_p(esc("격리한 지시문이 없습니다."), "empty"))
    return _section("security", body)


def _controls(view: EvidenceView) -> Html:
    rows = (
        (
            esc(c.control_id),
            esc(c.title),
            esc(c.source),
            esc(c.evidence),
            _chip(c.status.value, _CONTROL_CLASS[c.status]),
        )
        for c in view.controls
    )
    body: list[Html] = [
        _p(esc(CONTROLS_INTRO)),
        _table(("기준", "항목", "증적 위치", "이번 진단 내용", "판단"), rows, classes=("key",)),
    ]
    return _section("controls", body)


def _hypothesis(row: HypothesisRow) -> Html:
    css = "f-high" if row.status is HypothesisStatus.CONFIRMED else "f-unknown"
    anchor = "h-" + row.number.replace(".", "-")
    heading = Html(
        f"<h4>{_chip(row.status_label, _HYPOTHESIS_CLASS[row.status])}{esc(row.number)} {esc(row.title)}</h4>"
    )
    meta: list[tuple[str, Html]] = [
        ("가설 ID", _code(row.hypothesis_id)),
        ("상태·출처", esc(f"{row.status_label} · {row.origin_label}")),
        ("서비스", esc(row.service)),
        ("함께 일어나는 일", esc(row.perturbations)),
        ("리허설", esc(row.replay)),
    ]
    if row.reason:
        meta.append(("리허설 기록", esc(row.reason)))
    meta += [("그래프에 없는 전제", esc(a)) for a in row.assumptions]
    meta += [("확인할 것", esc(r)) for r in row.required_info]
    meta_html = "".join(f"<dt>{esc(k)}</dt><dd>{v}</dd>" for k, v in meta)
    parts: list[Html] = [
        Html(f'<article class="finding {esc(css)}" id="{esc(anchor)}">'),
        heading,
        Html(f'<dl class="meta">{meta_html}</dl>'),
    ]
    if row.explanation:
        parts.append(_p(esc(row.explanation)))
    parts.extend(_quote(q) for q in row.quotes)
    parts.append(Html("</article>"))
    return _join(parts, "\n")


def _explore(view: EvidenceView) -> Html:
    body: list[Html] = [_p(esc(EXPLORE_INTRO))]
    if not view.explored:
        body.append(_p(esc("이번 진단에는 규칙 밖 위험 탐색 기록이 없습니다."), "empty"))
        return _section("explore", body)
    facts = [f"결과: {view.hypothesis_summary}"]
    if view.exploration:
        facts.append(f"탐색 범위: {view.exploration}")
    facts += [f"참고: {note}" for note in view.exploration_notes]
    body.append(_ul(facts))
    if not view.hypotheses:
        body.append(_p(esc(view.exploration_empty), "empty"))
        return _section("explore", body)
    rows = (
        (
            esc(h.number),
            _chip(h.status_label, _HYPOTHESIS_CLASS[h.status]),
            esc(h.origin_label),
            esc(h.service),
            esc(h.perturbations),
            esc(h.replay),
        )
        for h in view.hypotheses
    )
    body.append(
        _table(
            ("#", "상태", "출처", "서비스", "함께 일어나는 일", "리허설 결과"),
            rows,
            classes=("key", "st"),
        )
    )
    body.extend(_hypothesis(row) for row in view.hypotheses if row.status is not HypothesisStatus.REFUTED)
    return _section("explore", body)


def _appendix_scoring(view: EvidenceView) -> Html:
    body: list[Html] = [
        _p(esc(SCORING_INTRO)),
        _ul(SCORING_METHOD),
        _h3("요소 가중치"),
        _table(
            ("요소", "키", "가중치"),
            ((esc(w.label), _code(w.key), esc(f"{w.weight:.2f}")) for w in view.element_weights),
            classes=("", "", "num"),
        ),
        _h3("서비스 등급 가중치"),
        _table(
            ("등급", "가중치"),
            ((esc(w.label), esc(f"{w.weight:.1f}")) for w in view.tier_weights),
            classes=("", "num"),
        ),
        _p(esc(TIER_FALLBACK_NOTE)),
    ]
    return _section("scoring", body, title_suffix=f" ({view.scoring_name})")


def _appendix_rules(view: EvidenceView) -> Html:
    rows = (
        (
            _code(r.rule_id),
            _severity_chip(r.severity),
            esc(r.element) if r.element else _DASH,
            esc(r.title),
            esc(r.rationale + (f" (법적 근거: {r.legal_basis})" if r.legal_basis else "")),
            esc(str(r.count)),
        )
        for r in view.rules
    )
    body: list[Html] = [
        _p(esc(RULES_INTRO)),
        _table(
            ("규칙 ID", "심각도", "요소", "제목", "판정 근거", "이번 진단"),
            rows,
            classes=("key", "st", "", "", "", "num"),
        ),
    ]
    return _section("rules", body)


def render_view_html(view: EvidenceView) -> str:
    """Render an already-built :class:`EvidenceView` as one self-contained HTML document."""
    title = view.title + (f" — {view.org_name}" if view.org_name else "")
    sections = (
        _overview,
        _services,
        _broken,
        _rehearsals,
        _history,
        _undecided,
        _security,
        _controls,
        _explore,
        _appendix_scoring,
        _appendix_rules,
    )
    footer = esc(
        f"제때옵스(JETTAE Ops) · 실행 {view.run_id} · 그래프 v{view.graph_version} ({view.graph_hash}) · "
        f"{view.created_at}"
    )
    parts = [
        "<!doctype html>",
        '<html lang="ko">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<meta name="color-scheme" content="light">',
        '<meta name="generator" content="JETTAE Ops">',
        f"<title>{esc(title)}</title>",
        f"<style>\n{_CSS}\n</style>",
        "</head>",
        "<body>",
        '<div class="doc">',
        _cover(view),
        "<main>",
        *(section(view) for section in sections),
        "</main>",
        f"<footer>{footer}</footer>",
        "</div>",
        "</body>",
        "</html>",
    ]
    return "\n".join(parts) + "\n"


def render_html(
    report: DiagnosisReport,
    graph: OpsGraph,
    *,
    versions: Sequence[GraphVersion] = (),
    patches: Sequence[Patch] = (),
    documents: Sequence[DocumentRef] = (),
    org_name: str | None = None,
) -> str:
    """Render the Korean evidence report (ISMS-P 증적 초안) as printable, self-contained HTML."""
    view = build_evidence(
        report, graph, versions=versions, patches=patches, documents=documents, org_name=org_name
    )
    return render_view_html(view)
