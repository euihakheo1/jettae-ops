"""Printable HTML evidence report: self-contained, A4-ready, escaped, deterministic.

The strongest check here is structural invariance: whatever text a runbook, roster or LLM puts into
a report, the HTML element tree stays exactly the same as with harmless text — so no user text can
ever become markup.
"""

from __future__ import annotations

from html.parser import HTMLParser

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from jettae.domain import DiagnosisReport, LlmMode, LlmUsageSummary
from jettae.reporting.evidence import SECTION_TITLES, build_evidence, printable
from jettae.reporting.html import FONT_STACK, esc, render_html, render_view_html
from test_reporting_sample import (
    F_DEPARTED,
    F_HOSTILE,
    FINDINGS,
    SCRIPT,
    VERSIONS,
    documents,
    empty_report,
    sample_graph,
    sample_report,
)

ALLOWED_TAGS = {
    "html", "head", "meta", "title", "style", "body", "div", "header", "p", "h1", "h2", "h3", "h4",
    "small", "nav", "ul", "li", "a", "main", "section", "table", "thead", "tbody", "tr", "th", "td",
    "code", "span", "strong", "article", "dl", "dt", "dd", "blockquote", "cite", "br", "footer",
}  # fmt: skip
VOID_TAGS = {"meta", "br"}
ALLOWED_ATTRS = {"lang", "charset", "name", "content", "class", "id", "href", "title", "aria-label"}


class _Page(HTMLParser):
    """Collects the element tree (tags + attributes) and the decoded text; checks nesting."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.elements: list[tuple[str, tuple[tuple[str, str | None], ...]]] = []
        self.texts: list[str] = []
        self.stack: list[str] = []
        self.doctype = ""

    def handle_decl(self, decl: str) -> None:
        self.doctype = decl

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.elements.append((tag, tuple(attrs)))
        if tag not in VOID_TAGS:
            self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        assert self.stack, f"</{tag}> without an open element"
        assert self.stack[-1] == tag, f"</{tag}> closes <{self.stack[-1]}>"
        self.stack.pop()

    def handle_data(self, data: str) -> None:
        self.texts.append(data)

    @property
    def text(self) -> str:
        return "".join(self.texts)


def _parse(page: str) -> _Page:
    parser = _Page()
    parser.feed(page)
    parser.close()
    assert parser.stack == []
    return parser


def _render(report: DiagnosisReport | None = None, org_name: str | None = "별빛상점") -> str:
    return render_html(
        report or sample_report(), sample_graph(), versions=VERSIONS, documents=documents(), org_name=org_name
    )


def _shape(page: str) -> list[tuple[str, tuple[tuple[str, str | None], ...]]]:
    return _parse(page).elements


# --------------------------------------------------------------------------- document shell


def test_page_is_a_self_contained_korean_html_document() -> None:
    page = _render()
    parsed = _parse(page)
    assert parsed.doctype == "doctype html"
    assert page.startswith('<!doctype html>\n<html lang="ko">')
    assert page.endswith("</html>\n")
    assert ("meta", (("charset", "utf-8"),)) in parsed.elements
    assert "<title>런북 준비도 진단 리포트 (ISMS-P 증적 초안) — 별빛상점</title>" in page
    for tag, attrs in parsed.elements:
        assert tag in ALLOWED_TAGS, tag
        for name, value in attrs:
            assert name in ALLOWED_ATTRS, (tag, name)
            if name == "href":
                assert value is not None
                assert value.startswith("#")
    assert "<script" not in page.lower()
    assert "src=" not in page
    assert "http://" not in page.replace("http://evil.example", "")
    assert "https://" not in page


def test_print_styles_font_stack_and_page_breaks() -> None:
    page = _render()
    assert f"font-family: {FONT_STACK};" in page
    assert FONT_STACK == 'Pretendard, "Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans KR", sans-serif'
    assert "@page { size: A4;" in page
    assert "section.break { break-before: page; }" in page
    assert "print-color-adjust: exact" in page
    for severity in ("critical", "high", "medium", "low", "info"):
        assert f".sev-{severity} {{" in page


def test_sections_and_table_of_contents_follow_the_report_order() -> None:
    parsed = _parse(_render())
    sections = [dict(attrs) for tag, attrs in parsed.elements if tag == "section"]
    assert [s["id"] for s in sections] == list(SECTION_TITLES)
    assert "class" not in sections[0]
    assert all(s["class"] == "break" for s in sections[1:])
    links = [dict(attrs)["href"] for tag, attrs in parsed.elements if tag == "a"]
    assert links == [f"#{key}" for key in SECTION_TITLES]
    text = parsed.text
    positions = [text.index(title) for title in SECTION_TITLES.values()]
    assert positions == sorted(positions)


# --------------------------------------------------------------------------- content


def test_cover_and_overview_show_the_key_numbers() -> None:
    page = _render()
    text = _parse(page).text
    assert "37.2 / 100" in text
    assert "SCORING_V1" in text
    assert "리허설 통과율40.0%5건 중 2건 통과" in text.replace("\n", "")
    assert "판정 경로 LLM 호출0회" in text
    assert "판정·점수·리허설 경로 LLM 호출 0회 (고정 규칙과 결정적 리플레이만 사용)" in text
    assert "2026-10-20 10:02:03 KST" in text
    assert 'class="notice alert"' not in page


def test_judgement_path_violation_gets_an_alert_box() -> None:
    llm = LlmUsageSummary(mode=LlmMode.LIVE, judgement_path_calls=1)
    page = _render(sample_report(llm=llm))
    assert 'class="notice alert"' in page
    assert "설계 원칙(판정은 규칙) 위반" in page


def test_findings_have_severity_chips_quotes_and_legal_basis() -> None:
    page = _render()
    assert '<article class="finding f-critical" id="f-3-2">' in page
    assert '<span class="chip sev-critical">심각</span>3.2 <span class="rule">JT-PPL-001</span>' in page
    assert '<span class="chip chip-unknown">확인 불가</span>' in page
    assert '<article class="finding f-unknown" id="f-6-1">' in page
    quote = (
        '<blockquote class="quote"><p>결제 장애가 계속되면 김대리에게 연락</p>'
        "<cite>— 결제 런북:3</cite></blockquote>"
    )
    assert quote in page
    assert "<p>첫 줄 | 파이프<br># 둘째 줄 &lt;b&gt;굵게&lt;/b&gt;</p>" in page
    assert "법적 근거: 정보통신망법 제48조의3 제1항: 24시간 이내 신고" in page
    assert F_DEPARTED.id in page


def test_service_table_uses_status_symbols_with_labels() -> None:
    page = _render()
    assert '<span class="st-unknown" title="확인 불가(0점)" aria-label="확인 불가(0점)">?</span>' in page
    assert '<span class="st-none" title="평가 기록 없음">—</span>' in page
    assert "<td>주문|정산</td>" in page


def test_rehearsal_history_security_and_controls() -> None:
    page = _render()
    assert '<span class="chip out-fail">실패</span>' in page
    assert '<span class="chip out-pass">통과</span>' in page
    assert "v1 실패 → v2 통과" in page
    assert "<td>박재현</td>" in page
    assert "<code>patch-2</code>" in page
    assert '<span class="chip ctl-gap">보완 필요</span>' in page
    assert '<span class="chip ctl-ok">이상 없음(고정 규칙 기준)</span>' in page
    excerpt = next(t for t in _parse(page).texts if t.startswith("<!-- AI 검토 도구에게"))
    assert len(excerpt) == 60
    assert excerpt.endswith("…")


def test_empty_report_renders_placeholders() -> None:
    graph = sample_graph().model_copy(update={"docs": {}})
    page = render_html(empty_report(), graph)
    text = _parse(page).text
    for phrase in (
        "진단에 쓴 문서 기록이 없습니다.",
        "점수를 매긴 서비스가 없습니다.",
        "고정 규칙이 찾은 끊김이 없습니다.",
        "실행한 모의 장애 리허설이 없습니다.",
        "그래프 버전 기록이 없습니다.",
        "격리한 지시문이 없습니다.",
        "해당 없음",
    ):
        assert phrase in text
    assert '<span class="chip ctl-missing">증적 없음</span>' in page
    assert "<title>런북 준비도 진단 리포트 (ISMS-P 증적 초안)</title>" in page


# --------------------------------------------------------------------------- escaping


def test_hostile_text_is_escaped_and_shown_literally() -> None:
    page = _render(org_name=SCRIPT)
    assert "&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;" in page
    text = _parse(page).text
    assert SCRIPT in text
    assert "\x1b" not in page
    assert "\u202e" not in page
    assert "[링크](http://evil.example)" in text


def test_escape_helper() -> None:
    assert esc('<a href="x">&</a>') == "&lt;a href=&quot;x&quot;&gt;&amp;&lt;/a&gt;"
    assert esc("탭\t\x1b[0m") == "탭 [0m"


@settings(max_examples=40, deadline=None)
@given(st.text(), st.text(), st.text())
@example(message="", name=" \t", quote="\n")
@example(message="<p>x</p>", name="</td><td>", quote="]]> <!-- -->")
def test_user_text_never_changes_the_element_tree(message: str, name: str, quote: str) -> None:
    neutral_finding = F_HOSTILE.model_copy(update={"message": "x", "title": "x"})
    hostile_finding = F_HOSTILE.model_copy(update={"message": message, "title": name})
    span = F_DEPARTED.evidence[1]
    neutral_departed = F_DEPARTED.model_copy(update={"evidence": (span.model_copy(update={"quote": "x"}),)})
    hostile_departed = F_DEPARTED.model_copy(update={"evidence": (span.model_copy(update={"quote": quote}),)})

    def report(finding: object, departed: object, service: str) -> DiagnosisReport:
        base = sample_report(findings=(finding, departed))
        services = (base.readiness.services[0].model_copy(update={"service_name": service}),)
        return base.model_copy(update={"readiness": base.readiness.model_copy(update={"services": services})})

    neutral = _render(report(neutral_finding, neutral_departed, "x"), org_name="x")
    org = name if printable(name).strip() else "x"  # a blank organisation name is omitted by design
    hostile = _render(report(hostile_finding, hostile_departed, name), org_name=org)
    neutral_shape = [(t, a) for t, a in _shape(neutral) if t != "br"]
    hostile_shape = [(t, a) for t, a in _shape(hostile) if t != "br"]
    assert hostile_shape == neutral_shape
    text = _parse(hostile).text
    for value in (message, name):
        flat = " ".join(printable(value).split())
        assert flat in text


# --------------------------------------------------------------------------- determinism


def test_rendering_is_deterministic_and_reuses_the_view() -> None:
    first = _render()
    assert first == _render()
    view = build_evidence(
        sample_report(), sample_graph(), versions=VERSIONS, documents=documents(), org_name="별빛상점"
    )
    assert render_view_html(view) == first


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_finding_order_does_not_change_the_bytes(seed: int) -> None:
    rotated = FINDINGS[seed:] + FINDINGS[:seed]
    assert _render(sample_report(findings=tuple(reversed(rotated)))) == _render()
