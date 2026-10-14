"""Terminal summary: content, display-width wrapping, terminal safety, determinism."""

from __future__ import annotations

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from jettae.domain import Finding, FindingStatus, LlmMode, LlmUsageSummary, Outcome, Severity
from jettae.reporting.evidence import printable
from jettae.reporting.text import (
    DEFAULT_TOP,
    MIN_WIDTH,
    display_width,
    render_cli_summary,
    wrap_display,
)
from test_reporting_sample import (
    ANSI,
    BIDI,
    F_DEPARTED,
    F_HOSTILE,
    FINDINGS,
    GRAPH_HASH,
    REHEARSALS,
    empty_report,
    sample_report,
)


def _lines(text: str) -> list[str]:
    assert text.endswith("\n")
    return text[:-1].split("\n")


# --------------------------------------------------------------------------- width primitives


@pytest.mark.parametrize(
    ("text", "width"),
    [("abc", 3), ("결제", 4), ("결제 OK", 7), ("·●—", 6), ("e\u0301", 1), ("", 0), ("ｱ", 1), ("ＡＢ", 4)],
)
def test_display_width(text: str, width: int) -> None:
    assert display_width(text) == width


def test_wrap_display_breaks_at_spaces_with_hanging_indent() -> None:
    lines = wrap_display("결제 장애가 계속되면 김대리에게 연락 안 되면 롤백", 20, first="- ", rest="  ")
    assert lines == ["- 결제 장애가", "  계속되면", "  김대리에게 연락 안", "  되면 롤백"]
    assert all(display_width(line) <= 20 for line in lines)


def test_wrap_display_splits_words_longer_than_a_line() -> None:
    lines = wrap_display("가" * 25, 20, first="> ", rest="> ")
    assert lines == ["> " + "가" * 9, "> " + "가" * 9, "> " + "가" * 7]
    assert wrap_display("", 20, first="* ") == ["* "]


@given(
    st.text(),
    st.integers(min_value=4, max_value=120),
    st.sampled_from(["", "- ", "    ", "가 "]),
)
def test_wrap_display_never_overflows_and_keeps_every_character(text: str, width: int, prefix: str) -> None:
    assume(width >= display_width(prefix) + 2)  # room for at least one wide character after the prefix
    lines = wrap_display(text, width, first=prefix, rest=prefix)
    assert all(display_width(line) <= width for line in lines)
    content = "".join(line[len(prefix) :].replace(" ", "") for line in lines)
    assert content == "".join(printable(text).split())


# --------------------------------------------------------------------------- summary content


def test_summary_lists_score_findings_and_rehearsal_rate() -> None:
    lines = _lines(render_cli_summary(sample_report()))
    text = "\n".join(lines)
    assert lines[0] == "제때옵스(JETTAE Ops) 런북 준비도 진단 요약"
    assert lines[1] == "=" * 100
    assert "전체 준비도 37.2 / 100 (SCORING_V1)" in text
    assert f"그래프      v2 · {GRAPH_HASH[:12]} · 진단 2026-10-20 10:02:03 KST" in text
    assert "끊긴 경로   4건 (심각 2 · 높음 1 · 낮음 1) · 확인 불가 2건" in text
    assert "리허설      5건 중 2건 통과 (통과율 40.0%)" in text
    assert "판정 경로   판정·점수·리허설 경로 LLM 호출 0회" in text
    assert "[주요 끊김] 전체 4건" in text
    assert "  1. [심각] JT-DDL-002 신고 시한이 법정 기준보다 늦음 — 조직 전체" in text
    assert "  2. [심각] JT-PPL-001 퇴사자에게 대응이 걸려 있음 — 결제" in text
    assert "     '결제' 서비스의 호출이 퇴사자 '김민준'에게 걸려 있습니다." in text
    assert "확인 불가 2건: 정상으로 보지 않으며" in text
    assert "[리허설 실패] 3건" in text
    assert "  - 결제 · 침해사고 신고: 신고 시한 초과" in text
    assert "  - sc-ghost: 사유 기록 없음" in text
    assert "[수정안] 초안" not in text
    assert "검증 통과(승인 대기) 1건" in text
    assert "승인 1건" in text
    assert "[판단 보류] 2건" in text
    assert "[보안 알림] 문서 속 AI 지시문 1건을 격리했습니다." in text
    assert lines[-1].startswith("자세한 근거")


def test_service_rows_show_tier_score_and_element_marks() -> None:
    text = render_cli_summary(sample_report())
    assert "  결제             1등급  22.0  감지◐ 담당○ 승인자? 대체채널○ 롤백●" in text
    assert "  주문|정산        2등급  60.0  감지● 담당● 승인자● 대체채널○ 롤백-" in text


def test_top_limits_the_listed_findings() -> None:
    text = render_cli_summary(sample_report(), top=1)
    assert "[주요 끊김] 전체 4건 중 상위 1건" in text
    assert "  … 외 3건은 증적 리포트에서 확인하세요." in text
    assert "JT-PPL-001" not in text
    assert "[주요 끊김] 전체 4건 중 상위" not in render_cli_summary(sample_report(), top=DEFAULT_TOP)
    assert (
        "   1." not in render_cli_summary(sample_report(), top=0).split("[주요 끊김]")[1].split("[리허설")[0]
    )


def test_long_messages_are_clipped_to_two_lines_with_an_ellipsis() -> None:
    long = F_DEPARTED.model_copy(update={"message": "아주 긴 설명입니다 " * 40})
    lines = _lines(render_cli_summary(sample_report(findings=(long,)), width=60))
    start = next(i for i, line in enumerate(lines) if "JT-PPL-001" in line)
    assert lines[start + 2].endswith("…")
    assert lines[start + 3] == ""
    assert all(display_width(line) <= 60 for line in lines)


def test_many_failed_rehearsals_are_counted() -> None:
    failing = tuple(
        REHEARSALS[2].model_copy(update={"scenario_id": f"sc-{i:02d}", "reason": f"실패 {i}"})
        for i in range(8)
    )
    text = render_cli_summary(sample_report(rehearsals=failing))
    assert "[리허설 실패] 8건" in text
    assert "  … 외 3건" in text
    assert "sc-04: 실패 4" in text
    assert "sc-05" not in text


def test_judgement_path_violation_is_the_first_thing_shown() -> None:
    llm = LlmUsageSummary(mode=LlmMode.LIVE, calls=2, judgement_path_calls=2)
    lines = _lines(render_cli_summary(sample_report(llm=llm)))
    assert lines[2].startswith("경고: 판정·점수·리허설 경로 LLM 호출 2회")
    assert not any(line.startswith("판정 경로") for line in lines)


def test_empty_report_summary() -> None:
    text = render_cli_summary(empty_report())
    assert "전체 준비도 0.0 / 100 (SCORING_V1)" in text
    assert "2026-10-20 09:00:00 (시간대 미기재)" in text
    assert "점수를 매긴 서비스가 없습니다." in text
    assert "고정 규칙이 찾은 끊김이 없습니다." in text
    assert "0건 중 0건 통과 (통과율 해당 없음)" in text
    assert "[리허설 실패]" not in text
    assert "[보안 알림]" not in text


def test_terminal_control_sequences_never_reach_the_output() -> None:
    hostile = F_HOSTILE.model_copy(update={"title": f"{ANSI}제목{BIDI}", "severity": Severity.CRITICAL})
    text = render_cli_summary(sample_report(findings=(hostile,)))
    assert "\x1b" not in text
    assert "\u202e" not in text
    assert all(ch == "\n" or ch.isprintable() for ch in text)


@pytest.mark.parametrize("width", [0, 10, MIN_WIDTH - 1])
def test_too_narrow_width_is_rejected(width: int) -> None:
    with pytest.raises(ValueError, match="width"):
        render_cli_summary(sample_report(), width=width)


def test_negative_top_is_rejected() -> None:
    with pytest.raises(ValueError, match="top"):
        render_cli_summary(sample_report(), top=-1)


@settings(max_examples=40, deadline=None)
@given(st.integers(min_value=MIN_WIDTH, max_value=160), st.text(), st.text())
def test_no_line_is_wider_than_the_terminal(width: int, title: str, message: str) -> None:
    finding = F_HOSTILE.model_copy(update={"title": title, "message": message})
    lines = _lines(render_cli_summary(sample_report(findings=(finding, *FINDINGS)), width=width))
    assert all(display_width(line) <= width for line in lines)
    assert all(line == line.rstrip() for line in lines)


@settings(max_examples=20)
@given(st.permutations(FINDINGS), st.permutations(REHEARSALS))
def test_summary_is_deterministic(findings: list[Finding], rehearsals: list[object]) -> None:
    shuffled = sample_report(findings=tuple(findings), rehearsals=tuple(rehearsals))
    assert render_cli_summary(shuffled) == render_cli_summary(sample_report())


def test_only_broken_findings_are_listed_as_broken() -> None:
    text = render_cli_summary(sample_report())
    listed = text.split("[주요 끊김]")[1].split("[리허설 실패]")[0]
    unknown = [f for f in FINDINGS if f.status is FindingStatus.UNKNOWN]
    assert unknown
    assert all(f.rule_id not in listed for f in unknown)
    assert all(r.outcome in (Outcome.PASS, Outcome.FAIL) for r in REHEARSALS)
