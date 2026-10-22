"""Markdown evidence report — '런북 준비도 진단 리포트 (ISMS-P 증적 초안)'.

Why Markdown: it is what auditors, wikis and pull requests accept as-is, it diffs cleanly between
runs, and it can be committed next to the runbooks it describes. The output is GitHub Flavored
Markdown (tables, no raw HTML) and is a pure function of its inputs (P6): rendering the same report
twice yields identical bytes, so the file itself can be hashed and kept as evidence.

Runbook quotes, names and LLM-drafted sentences are *untrusted data* (P5). Every such string is
escaped so it can neither inject Markdown structure (headings, tables, links) nor raw HTML into the
report: a runbook containing ``<script>`` or ``| 100점 |`` renders as that literal text.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Final

from jettae.domain import (
    ELEMENT_LABEL_KO,
    DiagnosisReport,
    DocumentRef,
    Element,
    GraphVersion,
    HypothesisStatus,
    OpsGraph,
    Outcome,
    Patch,
)
from jettae.reporting.evidence import (
    CONTROLS_INTRO,
    DISCLAIMER,
    DROPPED_EXAMPLES_TITLE,
    ELEMENT_STATUS_SYMBOL,
    EXPLORE_INTRO,
    HISTORY_NOTE,
    MINIMIZE_LABEL,
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
    EvidenceQuote,
    EvidenceView,
    FindingView,
    HypothesisRow,
    build_evidence,
    printable,
)

__all__ = ["code_span", "md_inline", "md_text", "render_evidence_markdown", "render_view_markdown"]

#: Characters with inline meaning in CommonMark/GFM. ``|`` is included so text can sit in a table
#: cell; ``&`` so entity references (``&lt;``) stay literal; ``#`` so a heading's closing sequence
#: cannot be faked. Backslash-escaping ASCII punctuation is always allowed by CommonMark.
#:
#: GitHub also turns *plain text* into markup, which escaping the CommonMark specials alone does not
#: stop: extended autolinks make ``https://…`` and ``www.…`` clickable, and ``$…$`` is typeset as
#: math (which can render large, coloured text). A runbook sentence must not become a link or styled
#: text in an evidence report, so the ``:`` of ``://`` and the ``.`` of ``www.`` are escaped (the
#: autolink scanners only fire on the literal characters) and ``$`` becomes the entity ``&#36;``
#: (GitHub does not honour ``\$`` outside math, but an entity can never open a math span).
_INLINE_SPECIAL: Final = re.compile(r"[\\`*_\[\]<>|~&#$]|:(?=//)|(?<=(?i:www))\.")
_DOLLAR_ENTITY: Final = "&#36;"
#: Block markers that only matter at the start of a line: list bullets, setext underlines and
#: ordered-list numbers (``1.`` / ``1)``).
_BLOCK_START: Final = re.compile(r"^(?:[-+=]|\d{1,9}[.)])")
_BACKTICK_RUN: Final = re.compile(r"`+")

_DASH: Final = "—"


def _escape_special(match: re.Match[str]) -> str:
    char = match.group(0)
    return _DOLLAR_ENTITY if char == "$" else "\\" + char


def _escape_inline(line: str) -> str:
    return _INLINE_SPECIAL.sub(_escape_special, line)


def _escape_line(line: str) -> str:
    escaped = _escape_inline(line)
    match = _BLOCK_START.match(escaped)
    if match is None:
        return escaped
    token = match.group(0)
    if token[0] in "-+=":
        return "\\" + escaped
    return token[:-1] + "\\" + escaped[len(token) - 1 :]


def md_text(text: str) -> str:
    """Escape untrusted text for one line of GFM (whitespace, including newlines, is collapsed).

    Safe anywhere, including at the start of a paragraph, list item or quote line, where ``1.``
    or ``-`` would otherwise open a list.
    """
    return _escape_line(" ".join(printable(text).split()))


def md_inline(text: str) -> str:
    """Like :func:`md_text` for places where block markers are inert (headings, table cells), so
    ``3.1`` stays readable in the Markdown source instead of becoming ``3\\.1``."""
    return _escape_inline(" ".join(printable(text).split()))


def code_span(text: str, *, table: bool = False) -> str:
    """A code span that survives any content: the fence is longer than any backtick run inside.

    Inside a table cell GFM still splits on ``|`` within code spans, so pipes are escaped there.
    Blank text gives an empty string (a table shows its dash): an empty span would be printed as
    two literal backticks.
    """
    flat = " ".join(printable(text).split())
    if not flat:
        return ""
    longest = max((len(run) for run in _BACKTICK_RUN.findall(flat)), default=0)
    fence = "`" * (longest + 1)
    pad = " " if flat.startswith("`") or flat.endswith("`") else ""
    body = flat.replace("|", "\\|") if table else flat
    return f"{fence}{pad}{body}{pad}{fence}"


def _heading(level: int, key: str, suffix: str = "") -> list[str]:
    return [f"{'#' * level} {md_inline(SECTION_TITLES[key] + suffix)}", ""]


def _quote_block(quote: EvidenceQuote) -> list[str]:
    lines = []
    for raw in quote.text.split("\n"):
        flat = " ".join(printable(raw).split())
        lines.append(f"> {_escape_line(flat)}" if flat else ">")
    lines += [">", f"> {_DASH} {md_text(quote.location)}"]
    return lines


def _table(headers: Sequence[str], rows: Iterable[Sequence[str]], align: str) -> list[str]:
    """``align`` has one char per column: ``l`` left, ``r`` right, ``c`` centre."""
    delimiter = {"l": ":---", "r": "---:", "c": ":---:"}
    out = [
        "| " + " | ".join(md_text(h) for h in headers) + " |",
        "| " + " | ".join(delimiter[a] for a in align) + " |",
    ]
    out += ["| " + " | ".join(cell or _DASH for cell in row) + " |" for row in rows]
    return out


def _finding(view: FindingView) -> list[str]:
    f = view.finding
    heading = f"#### {md_inline(f'{view.number} {f.rule_id} {f.title} {_DASH} {view.service}')}"
    service = view.service + (f" · 요소: {view.element}" if view.element else "")
    meta = [
        f"- 판정 ID: {code_span(f.id)}",
        f"- 심각도: {md_text(view.severity)} · 상태: {md_text(view.status)}",
        f"- 서비스: {md_text(service)}",
    ]
    if view.subjects:
        meta.append(f"- 관련 요소: {md_text(view.subjects)}")
    if view.fix:
        meta.append(f"- 권장 조치: {md_text(view.fix)}")
    out = [heading, "", *meta, "", md_text(f.message), ""]
    if view.quotes:
        out += ["근거 원문:", ""]
        for quote in view.quotes:
            out += [*_quote_block(quote), ""]
    else:
        out += [md_text(NO_QUOTE_NOTE), ""]
    if f.legal_basis:
        out += [f"법적 근거: {md_text(f.legal_basis)}", ""]
    return out


def _overview(view: EvidenceView) -> list[str]:
    rows: list[tuple[str, str]] = []
    if view.org_name:
        rows.append(("조직", md_text(view.org_name)))
    rows += [
        ("워크스페이스", code_span(view.workspace_id, table=True)),
        ("진단 실행 ID", code_span(view.run_id, table=True)),
        ("전체 준비도", f"**{view.overall:.1f} / 100** {md_text(f'(산식 {view.scoring_name})')}"),
        ("그래프 버전", f"v{view.graph_version} · 해시 {code_span(view.graph_hash, table=True)}"),
        ("진단 시각", md_text(view.created_at)),
        ("끊긴 대응 경로", md_text(view.broken_summary)),
        ("리허설", md_text(view.rehearsal_summary)),
        ("원문 근거", md_text(view.grounding)),
        ("LLM 사용", md_text(view.llm_usage)),
        ("판정 경로", f"**{md_text(view.judgement_path)}**"),
    ]
    out = [*_heading(2, "overview"), *_table(("항목", "내용"), rows, "ll"), ""]
    out += ["### 진단 대상 문서", ""]
    if view.documents:
        out += _table(
            ("문서", "경로", "SHA-256(앞 12자리)"),
            (
                (
                    md_text(d.title),
                    code_span(d.path, table=True) if d.path else _DASH,
                    code_span(d.sha256, table=True),
                )
                for d in view.documents
            ),
            "lll",
        )
    else:
        out.append("진단에 쓴 문서 기록이 없습니다.")
    return [*out, ""]


def _services(view: EvidenceView) -> list[str]:
    out = [*_heading(2, "services"), md_text(STATUS_LEGEND), ""]
    if view.services:
        headers = ("서비스", "등급", "점수", *(ELEMENT_LABEL_KO[e] for e in Element))
        rows = (
            (
                md_text(s.name),
                md_text(f"{s.tier}({s.tier_label})"),
                f"{s.score:.1f}",
                *(ELEMENT_STATUS_SYMBOL[st] if st is not None else _DASH for st in s.statuses),
            )
            for s in view.services
        )
        out += _table(headers, rows, "lcr" + "c" * len(Element))
    else:
        out.append("점수를 매긴 서비스가 없습니다.")
    if view.catalog_gaps:
        out += ["", md_text("카탈로그에는 있지만 담당·런북이 없는 서비스: " + ", ".join(view.catalog_gaps))]
    return [*out, ""]


def _broken(view: EvidenceView) -> list[str]:
    out = _heading(2, "broken")
    if not view.broken:
        out += ["고정 규칙이 찾은 끊김이 없습니다.", ""]
    for severity, group in view.broken:
        out += [f"### {SEVERITY_LABEL_KO[severity]} ({len(group)}건)", ""]
        for item in group:
            out += _finding(item)
    if view.unknown_count:
        out += [f"확인 불가로 판정한 {view.unknown_count}건은 6장에 따로 적었습니다.", ""]
    return out


def _outcome(outcome: Outcome) -> str:
    label = OUTCOME_LABEL_KO[outcome]
    return label if outcome is Outcome.PASS else f"**{label}**"


def _rehearsals(view: EvidenceView) -> list[str]:
    out = _heading(2, "rehearsals")
    if not view.rehearsals:
        return [*out, "실행한 모의 장애 리허설이 없습니다.", ""]
    out += [md_text(view.rehearsal_intro), ""]
    rows = (
        (
            md_text(r.title),
            md_text(r.service),
            _outcome(r.outcome),
            md_text(r.deadline),
            md_text(r.ended_at),
            md_text(r.broken_at) if r.broken_at else _DASH,
            md_text(r.comparison) if r.comparison else _DASH,
        )
        for r in view.rehearsals
    )
    headers = ("시나리오", "서비스", "결과", "시한", "종료 시각", "끊긴 지점", "수정 전→후")
    out += _table(headers, rows, "llcrrll")
    failures = [r for r in view.rehearsals if r.outcome is Outcome.FAIL and r.reason]
    if failures:
        out += ["", "실패 사유:", ""]
        out += [f"- {md_text(r.title)}: {md_text(r.reason)}" for r in failures]
    return [*out, ""]


def _history(view: EvidenceView) -> list[str]:
    out = _heading(2, "history")
    if view.versions:
        version_rows = (
            (
                f"v{v.number}",
                md_text(v.change),
                md_text(v.sentence) if v.sentence else _DASH,
                md_text(v.approver),
                md_text(v.at),
                md_text(v.resolved) if v.resolved else _DASH,
            )
            for v in view.versions
        )
        headers = ("버전", "변경", "수정안 문장", "승인자", "시각", "해결한 끊김")
        out += _table(headers, version_rows, "llllll")
    else:
        out.append("그래프 버전 기록이 없습니다.")
    if view.other_patches:
        out += ["", f"### {md_inline(UNLINKED_PATCHES_TITLE)}", ""]
        patch_rows = (
            (
                code_span(p.patch_id, table=True),
                md_text(p.status),
                md_text(p.author),
                md_text(p.sentence),
                md_text(p.targets),
                md_text(p.decided) if p.decided else _DASH,
            )
            for p in view.other_patches
        )
        out += _table(("수정안", "상태", "작성", "수정안 문장", "대상 끊김", "결정"), patch_rows, "llllll")
    out += ["", md_text(HISTORY_NOTE)]
    return [*out, ""]


def _undecided(view: EvidenceView) -> list[str]:
    out = [*_heading(2, "undecided"), md_text(UNDECIDED_INTRO), ""]
    if not (view.unknown or view.held or view.unknown_elements or view.dropped_examples):
        return [*out, "판단 보류·확인 불가 항목이 없습니다.", ""]
    if view.unknown:
        out += [f"### 확인 불가 판정 ({len(view.unknown)}건)", ""]
        for item in view.unknown:
            out += _finding(item)
    if view.unknown_elements:
        out += ["### 확인 불가 요소 (점수 0점 처리)", ""]
        element_rows = (
            (md_text(u.service), md_text(u.element), md_text(" / ".join(u.reasons)) if u.reasons else _DASH)
            for u in view.unknown_elements
        )
        out += [*_table(("서비스", "요소", "사유"), element_rows, "lll"), ""]
    if view.held:
        out += [f"### 판단 보류 요소 ({len(view.held)}건)", ""]
        held_rows = (
            (md_text(h.label), md_text(h.kind), md_text(h.reason), code_span(h.ref_id, table=True))
            for h in view.held
        )
        out += [*_table(("요소", "종류", "보류 사유", "식별자"), held_rows, "llll"), ""]
    if view.dropped_examples:
        out += [f"### {md_inline(DROPPED_EXAMPLES_TITLE)}", ""]
        out += [f"- {md_text(x)}" for x in view.dropped_examples]
        out.append("")
    return out


def _security(view: EvidenceView) -> list[str]:
    out = [*_heading(2, "security"), md_text(SECURITY_INTRO), ""]
    if not view.security:
        return [*out, "격리한 지시문이 없습니다.", ""]
    rows = (
        (md_text(s.document), md_text(s.location), md_text(s.patterns), md_text(s.excerpt))
        for s in view.security
    )
    return [*out, *_table(("문서", "위치", "탐지 유형", "격리된 지시문 발췌"), rows, "llll"), ""]


def _controls(view: EvidenceView) -> list[str]:
    out = [*_heading(2, "controls"), md_text(CONTROLS_INTRO), ""]
    rows = (
        (md_text(c.control_id), md_text(c.title), md_text(c.source), md_text(c.evidence), md_text(c.status))
        for c in view.controls
    )
    return [*out, *_table(("기준", "항목", "증적 위치", "이번 진단 내용", "판단"), rows, "lllll"), ""]


def _hypothesis_status(row: HypothesisRow) -> str:
    return (
        f"**{md_inline(row.status_label)}**"
        if row.status is HypothesisStatus.CONFIRMED
        else md_inline(row.status_label)
    )


def _hypothesis(row: HypothesisRow) -> list[str]:
    heading = f"### {md_inline(f'{row.number} {row.title} {_DASH} {row.status_label}')}"
    meta = [
        f"- 가설 ID: {code_span(row.hypothesis_id)}",
        f"- 상태: {md_text(row.status_label)} · 출처: {md_text(row.origin_label)}",
        f"- 서비스: {md_text(row.service)}",
        f"- 함께 일어나는 일: {md_text(row.perturbations)}",
        f"- 리허설: {md_text(row.replay)}",
    ]
    if row.reason:
        meta.append(f"- 리허설 기록: {md_text(row.reason)}")
    meta += [f"- 그래프에 없는 전제: {md_text(a)}" for a in row.assumptions]
    meta += [f"- 확인할 것: {md_text(r)}" for r in row.required_info]
    if row.minimal is not None:
        meta.append(f"- {md_text(MINIMIZE_LABEL)}: {md_text(row.minimal)}")
        meta += [f"  - {md_text(note)}" for note in row.minimal_notes]
    out = [heading, "", *meta, ""]
    if row.explanation:
        out += [md_text(row.explanation), ""]
    if row.quotes:
        out += ["근거 원문:", ""]
        for quote in row.quotes:
            out += [*_quote_block(quote), ""]
    return out


def _explore(view: EvidenceView) -> list[str]:
    out = [*_heading(2, "explore"), md_text(EXPLORE_INTRO), ""]
    if not view.explored:
        return [*out, "이번 진단에는 규칙 밖 위험 탐색 기록이 없습니다.", ""]
    out.append(f"- 결과: {md_text(view.hypothesis_summary)}")
    if view.exploration:
        out.append(f"- 탐색 범위: {md_text(view.exploration)}")
    out += [f"- 참고: {md_text(note)}" for note in view.exploration_notes]
    out.append("")
    if not view.hypotheses:
        return [*out, md_text(view.exploration_empty), ""]
    rows = (
        (
            md_inline(h.number),
            _hypothesis_status(h),
            md_text(h.origin_label),
            md_text(h.service),
            md_text(h.perturbations),
            md_text(h.replay),
        )
        for h in view.hypotheses
    )
    out += [*_table(("#", "상태", "출처", "서비스", "함께 일어나는 일", "리허설 결과"), rows, "llllll"), ""]
    for row in view.hypotheses:
        if row.status is not HypothesisStatus.REFUTED:
            out += _hypothesis(row)
    return out


def _appendix_scoring(view: EvidenceView) -> list[str]:
    out = [
        *_heading(2, "scoring", f" ({view.scoring_name})"),
        md_text(SCORING_INTRO),
        "",
        *(f"- {md_text(line)}" for line in SCORING_METHOD),
        "",
        "### 요소 가중치",
        "",
    ]
    out += _table(
        ("요소", "키", "가중치"),
        ((md_text(w.label), code_span(w.key, table=True), f"{w.weight:.2f}") for w in view.element_weights),
        "llr",
    )
    out += ["", "### 서비스 등급 가중치", ""]
    out += _table(
        ("등급", "가중치"), ((md_text(w.label), f"{w.weight:.1f}") for w in view.tier_weights), "lr"
    )
    out += ["", md_text(TIER_FALLBACK_NOTE)]
    return [*out, ""]


def _appendix_rules(view: EvidenceView) -> list[str]:
    out = [*_heading(2, "rules"), md_text(RULES_INTRO), ""]
    rows = (
        (
            md_text(r.rule_id),
            md_text(r.severity_label),
            md_text(r.element) if r.element else _DASH,
            md_text(r.title),
            md_text(r.rationale + (f" (법적 근거: {r.legal_basis})" if r.legal_basis else "")),
            str(r.count),
        )
        for r in view.rules
    )
    headers = ("규칙 ID", "심각도", "요소", "제목", "판정 근거", "이번 진단")
    return [*out, *_table(headers, rows, "lcllll"), ""]


def render_view_markdown(view: EvidenceView) -> str:
    """Render an already-built :class:`EvidenceView` (lets callers reuse one view for MD and HTML)."""
    lines = [f"# {md_inline(view.title)}", ""]
    if view.org_name:
        lines += [f"**{md_text(view.org_name)}**", ""]
    lines += [f"> {md_text(DISCLAIMER)}", ""]
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
    for section in sections:
        lines += section(view)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + "\n"


def render_evidence_markdown(
    report: DiagnosisReport,
    graph: OpsGraph,
    *,
    versions: Sequence[GraphVersion] = (),
    patches: Sequence[Patch] = (),
    documents: Sequence[DocumentRef] = (),
    org_name: str | None = None,
) -> str:
    """Render the Korean evidence report (ISMS-P 증적 초안) as deterministic GFM."""
    view = build_evidence(
        report, graph, versions=versions, patches=patches, documents=documents, org_name=org_name
    )
    return render_view_markdown(view)
