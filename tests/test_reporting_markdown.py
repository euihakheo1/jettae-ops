"""Markdown evidence report: structure, Korean content, GFM validity, escaping, determinism.

Validity is checked by parsing the output with markdown-it-py (CommonMark + GFM tables): every
table keeps its column count, no raw HTML token survives, and escaped text reads back verbatim.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from markdown_it import MarkdownIt
from markdown_it.token import Token

from jettae.check.registry import rule_catalog
from jettae.domain import LlmMode, LlmUsageSummary
from jettae.reporting.evidence import EVIDENCE_TITLE, SECURITY_EXCERPT_CHARS, build_evidence, printable
from jettae.reporting.markdown import (
    code_span,
    md_inline,
    md_text,
    render_evidence_markdown,
    render_view_markdown,
)
from test_reporting_sample import (
    F_DEPARTED,
    FINDINGS,
    GRAPH_HASH,
    PIPE_NAME,
    SCRIPT,
    VERSIONS,
    documents,
    empty_report,
    sample_graph,
    sample_report,
)

_MD = MarkdownIt("commonmark").enable("table").enable("strikethrough")

SECTIONS = [
    "1. 개요",
    "2. 서비스별 준비도",
    "3. 끊긴 대응 경로",
    "4. 리허설 결과",
    "5. 개정 이력",
    "6. 판단 보류·확인 불가 항목",
    "7. 보안 알림",
    "8. 인증 기준 매핑",
    "9. 규칙 밖 위험 후보 (점수에 반영하지 않음)",
    "부록 A. 점수 산식 (SCORING_V1)",
    "부록 B. 규칙 카탈로그",
]


def _render(**changes: object) -> str:
    return render_evidence_markdown(
        sample_report(**changes),
        sample_graph(),
        versions=VERSIONS,
        documents=documents(),
        org_name="별빛상점",
    )


def _text(token: Token) -> str:
    """Plain text of an inline token (escapes resolved, code spans included, soft breaks as LF)."""
    parts = []
    for child in token.children or ():
        if child.type in ("text", "text_special", "code_inline"):
            parts.append(child.content)
        elif child.type == "softbreak":
            parts.append("\n")
    return "".join(parts)


def _headings(tokens: list[Token], level: int) -> list[str]:
    tag = f"h{level}"
    return [_text(tokens[i + 1]) for i, t in enumerate(tokens) if t.type == "heading_open" and t.tag == tag]


def _tables(tokens: list[Token]) -> Iterator[list[list[str]]]:
    """Each table as rows of cell texts (header row first)."""
    rows: list[list[str]] = []
    row: list[str] = []
    for i, token in enumerate(tokens):
        if token.type == "table_open":
            rows = []
        elif token.type == "tr_open":
            row = []
        elif token.type in ("th_open", "td_open"):
            row.append(_text(tokens[i + 1]))
        elif token.type == "tr_close":
            rows.append(row)
        elif token.type == "table_close":
            yield rows


def _table_after(tokens: list[Token], heading: str) -> list[list[str]]:
    """The first table that follows the heading with this text."""
    start = next(i for i, t in enumerate(tokens) if t.type == "inline" and _text(t) == heading)
    return next(_tables(tokens[start:]))


# --------------------------------------------------------------------------- structure


def test_sections_appear_in_order_with_the_report_title() -> None:
    tokens = _MD.parse(_render())
    assert _headings(tokens, 1) == [EVIDENCE_TITLE]
    assert _headings(tokens, 2) == SECTIONS


def test_output_is_valid_gfm_without_raw_html_and_with_consistent_tables() -> None:
    tokens = _MD.parse(_render())
    assert not [t for t in tokens if t.type == "html_block"]
    inline_html = [
        c for t in tokens if t.type == "inline" for c in t.children or () if c.type == "html_inline"
    ]
    assert not inline_html
    tables = list(_tables(tokens))
    assert len(tables) >= 10
    for table in tables:
        assert all(len(row) == len(table[0]) for row in table), table[0]


def test_overview_content_in_korean() -> None:
    text = _render()
    assert "**별빛상점**" in text
    assert f"| 그래프 버전 | v2 · 해시 `{GRAPH_HASH[:12]}` |" in text
    assert "| 진단 시각 | 2026-10-20 10:02:03 KST |" in text
    assert "| 전체 준비도 | **37.2 / 100** (산식 SCORING\\_V1) |" in text
    assert "녹화 재생(결정적) · anthropic / claude-opus-5-5 · 호출 12회 · 토큰 입력 34,567" in text
    assert "**판정·점수·리허설 경로 LLM 호출 0회 (고정 규칙과 결정적 리플레이만 사용)**" in text
    assert "| 끊긴 대응 경로 | 4건 (심각 2 · 높음 1 · 낮음 1) · 확인 불가 2건 |" in text


def test_judgement_path_violation_is_printed_not_hidden() -> None:
    llm = LlmUsageSummary(mode=LlmMode.LIVE, calls=3, judgement_path_calls=2)
    assert "판정·점수·리허설 경로 LLM 호출 2회 — 설계 원칙(판정은 규칙) 위반" in _render(llm=llm)


def test_service_table_shows_symbols_and_escapes_pipes() -> None:
    tokens = _MD.parse(_render())
    table = _table_after(tokens, SECTIONS[1])
    assert table[0] == ["서비스", "등급", "점수", "감지", "담당", "승인자", "대체 채널", "롤백"]
    assert table[1] == ["결제", "1(핵심)", "22.0", "◐", "○", "?", "○", "●"]
    assert table[2] == [PIPE_NAME, "2(중요)", "60.0", "●", "●", "●", "○", "—"]
    assert "| 주문\\|정산 |" in _render()


def test_findings_show_rule_quote_location_and_legal_basis() -> None:
    text = _render()
    assert "#### 3.1 JT-DDL-002 신고 시한이 법정 기준보다 늦음 — 조직 전체" in text
    assert "#### 3.2 JT-PPL-001 퇴사자에게 대응이 걸려 있음 — 결제" in text
    assert "> 결제 장애가 계속되면 김대리에게 연락\n>\n> — 결제 런북:3" in text
    assert "- 관련 요소: 김민준, 결제 → 김민준 (담당)" in text
    assert "- 권장 조치: 퇴사자를 재직자로 교체" in text
    assert "법적 근거: 정보통신망법 제48조의3 제1항: 24시간 이내 신고" in text
    assert "> — 보안 \\| 신고 \\<절차\\>:13" in text
    assert "근거 원문: 인용할 문장이 없습니다." in text
    assert "### 심각 (2건)" in text
    assert "### 낮음 (1건)" in text
    assert "확인 불가로 판정한 2건은 6장에 따로 적었습니다." in text


def test_multiline_quote_stays_inside_one_blockquote_as_text() -> None:
    tokens = _MD.parse(_render())
    depth = 0
    for token in tokens:
        if token.type == "blockquote_open":
            depth += 1
        elif token.type == "blockquote_close":
            depth -= 1
        elif depth:
            assert token.type in ("paragraph_open", "paragraph_close", "inline"), token.type
    quoted = [_text(t) for t in tokens if t.type == "inline"]
    assert "첫 줄 | 파이프\n# 둘째 줄 <b>굵게</b>" in quoted


def test_hostile_text_is_rendered_literally() -> None:
    tokens = _MD.parse(_render())
    texts = [_text(t) for t in tokens if t.type == "inline"]
    message = next(t for t in texts if t.startswith("<script>"))
    assert message == f"{SCRIPT} | 100점 | [31m빨강[0m evil [링크](http://evil.example) {{0}}"
    assert not [c for t in tokens if t.type == "inline" for c in t.children or () if c.type == "link_open"]
    assert "\x1b" not in _render()
    assert "\u202e" not in _render()


def test_rehearsal_table_and_failure_reasons() -> None:
    tokens = _MD.parse(_render())
    table = _table_after(tokens, SECTIONS[3])
    assert table[0] == ["시나리오", "서비스", "결과", "시한", "종료 시각", "끊긴 지점", "수정 전→후"]
    assert table[1] == ["결제 · 기준", "결제", "통과", "1시간", "T+30분", "—", "v1 실패 → v2 통과"]
    assert table[2][2] == "실패"
    assert table[2][-1] == "승인 시 실패 → 통과 예상 (수정안 patch-2)"
    text = _render()
    assert "모의 장애 5건을 운영 그래프 위에서 결정적으로 재생했습니다. 통과 2건 (통과율 40.0%)" in text
    assert "- 결제 · 침해사고 신고: 신고 시한 초과 \\<script\\>" in text


def test_history_lists_versions_and_pending_patches() -> None:
    tokens = _MD.parse(_render())
    versions = _table_after(tokens, SECTIONS[4])
    assert versions[0] == ["버전", "변경", "수정안 문장", "승인자", "시각", "해결한 끊김"]
    assert versions[2][:2] == ["v2", "수정안 patch-1 반영"]
    assert versions[2][2] == f"결제 장애 시 이수진 대리에게 연락한다. | 보조 {SCRIPT}"
    assert versions[2][3:] == ["박재현", "2026-10-21 09:30:00 KST", "JT-PPL-001 (1건)"]
    pending = _table_after(tokens, "버전 기록과 연결되지 않은 수정안")
    assert [row[0] for row in pending[1:]] == ["patch-2", "patch-3"]


def test_undecided_section_lists_unknown_findings_held_elements_and_dropped_examples() -> None:
    text = _render()
    section = text.split("## 6. 판단 보류·확인 불가 항목", 1)[1].split("## 7.", 1)[0]
    assert "### 확인 불가 판정 (2건)" in section
    assert "#### 6.1 JT-INT-001 판단 보류 요소 — 결제" in section
    assert "- 심각도: 보통 · 상태: 확인 불가" in section
    assert "### 확인 불가 요소 (점수 0점 처리)" in section
    assert "### 판단 보류 요소 (2건)" in section
    assert "- 김과장에게 문자 \\| 발송" in section


def test_security_excerpt_is_masked_to_sixty_characters() -> None:
    tokens = _MD.parse(_render())
    table = _table_after(tokens, SECTIONS[6])
    assert table[0] == ["문서", "위치", "탐지 유형", "격리된 지시문 발췌"]
    excerpt = table[1][3]
    assert len(excerpt) == SECURITY_EXCERPT_CHARS
    assert excerpt.startswith("<!-- AI 검토 도구에게")
    assert table[1][0] == "보안 | 신고 <절차>"


def test_certification_mapping_and_appendices() -> None:
    tokens = _MD.parse(_render())
    controls = _table_after(tokens, SECTIONS[7])
    assert [row[0] for row in controls[1:]] == ["2.11.4", "2.11.5", "2.12.1", "2.12.2"]
    ok = "이상 없음(고정 규칙 기준)"
    assert [row[-1] for row in controls[1:]] == ["보완 필요", "보완 필요", ok, ok]
    weights = _table_after(tokens, "요소 가중치")
    assert weights[1:] == [
        ["감지", "detection", "0.20"],
        ["담당", "owner", "0.25"],
        ["승인자", "approver", "0.20"],
        ["대체 채널", "fallback_channel", "0.15"],
        ["롤백", "rollback", "0.20"],
    ]
    rules = _table_after(tokens, SECTIONS[10])
    assert [row[0] for row in rules[1:]] == sorted(i.rule_id for i in rule_catalog())
    assert {row[0]: row[-1] for row in rules[1:]}["JT-PPL-001"] == "1"


def test_empty_report_renders_placeholders() -> None:
    text = render_evidence_markdown(empty_report(), sample_graph().model_copy(update={"docs": {}}))
    tokens = _MD.parse(text)
    assert _headings(tokens, 2) == SECTIONS
    for phrase in (
        "진단에 쓴 문서 기록이 없습니다.",
        "점수를 매긴 서비스가 없습니다.",
        "고정 규칙이 찾은 끊김이 없습니다.",
        "실행한 모의 장애 리허설이 없습니다.",
        "그래프 버전 기록이 없습니다.",
        "판단 보류·확인 불가 항목이 없습니다.",
        "격리한 지시문이 없습니다.",
        "2026-10-20 09:00:00 (시간대 미기재)",
    ):
        assert phrase in text
    assert text.endswith("\n")
    assert not text.endswith("\n\n")


# --------------------------------------------------------------------------- determinism


def test_rendering_is_deterministic_and_reuses_the_view() -> None:
    first = _render()
    assert first == _render()
    view = build_evidence(
        sample_report(), sample_graph(), versions=VERSIONS, documents=documents(), org_name="별빛상점"
    )
    assert render_view_markdown(view) == first


@settings(max_examples=20)
@given(st.permutations(FINDINGS))
def test_finding_order_does_not_change_the_bytes(findings: list[object]) -> None:
    assert _render(findings=tuple(findings)) == _render()


# --------------------------------------------------------------------------- escaping primitives


def _paragraph_text(markdown: str) -> str:
    tokens = _MD.parse(markdown)
    assert [t.type for t in tokens] == ["paragraph_open", "inline", "paragraph_close"], markdown
    children = tokens[1].children or []
    assert {c.type for c in children} <= {"text", "text_special"}, markdown
    return _text(tokens[1])


@pytest.mark.parametrize(
    "text",
    [
        "- 목록",
        "1. 순서",
        "12) 순서",
        "# 제목",
        "> 인용",
        "***",
        "===",
        "+ 더하기",
        "| a | b |",
        "<b>굵게</b>",
        "&amp;",
    ],
)
def test_md_text_neutralises_block_and_inline_syntax(text: str) -> None:
    assert _paragraph_text(md_text(text)) == text


@given(st.text())
def test_md_text_round_trips_through_a_commonmark_parser(text: str) -> None:
    expected = " ".join(printable(text).split())
    if expected:
        assert _paragraph_text(md_text(text)) == expected


@given(st.text())
def test_md_inline_round_trips_inside_a_heading(text: str) -> None:
    expected = " ".join(printable(text).split())
    if expected:
        tokens = _MD.parse(f"## {md_inline(text)}")
        assert tokens[0].type == "heading_open"
        assert _text(tokens[1]) == expected


@given(st.text(min_size=1))
def test_code_span_survives_any_backticks(text: str) -> None:
    expected = " ".join(printable(text).split())
    if expected:
        tokens = _MD.parse(f"앞 {code_span(text)} 뒤")
        codes = [c.content for c in tokens[1].children or () if c.type == "code_inline"]
        assert codes == [expected]


@given(st.text(min_size=1))
def test_table_cells_keep_their_column_whatever_the_service_name(name: str) -> None:
    report = sample_report()
    services = (
        report.readiness.services[0].model_copy(update={"service_name": name}),
        *report.readiness.services[1:],
    )
    readiness = report.readiness.model_copy(update={"services": services})
    text = render_evidence_markdown(report.model_copy(update={"readiness": readiness}), sample_graph())
    table = _table_after(_MD.parse(text), SECTIONS[1])
    assert all(len(row) == 8 for row in table)
    expected = " ".join(printable(name).split()) or "—"
    assert expected in {table[1][0], table[2][0]}


def test_code_span_in_tables_escapes_pipes() -> None:
    assert code_span("a|b", table=True) == "`a\\|b`"
    assert code_span("`x`") == "`` `x` ``"
    assert F_DEPARTED.id in _render()
