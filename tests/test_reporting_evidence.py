"""Evidence view model: the single place that groups, sorts, labels and maps the report."""

from __future__ import annotations

import math
import unicodedata
from datetime import UTC, datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jettae.check.registry import rule_catalog
from jettae.check.scoring import SCORING_V1
from jettae.domain import (
    DocumentRef,
    Element,
    ElementStatus,
    LlmMode,
    LlmUsageSummary,
    Outcome,
    PatchStatus,
    ReadinessReport,
    Severity,
    SourceSpan,
)
from jettae.reporting.evidence import (
    SECURITY_EXCERPT_CHARS,
    ControlStatus,
    EvidenceView,
    build_evidence,
    current_rehearsals,
    distinct_spans,
    format_datetime,
    format_minutes,
    judgement_path_statement,
    llm_usage_line,
    mask_excerpt,
    printable,
    service_rows,
    short_hash,
)
from test_reporting_sample import (
    E_ESC,
    F_DEPARTED,
    F_HELD,
    F_HOSTILE,
    F_INFO,
    F_LEGAL,
    F_ONCALL,
    FINDINGS,
    GRAPH_HASH,
    OLD_HASH,
    P_APPROVED,
    P_DRAFT,
    PAY_DOC,
    REHEARSALS,
    SC_BASE,
    SCRIPT,
    SPAN_LEGAL,
    SPAN_MULTILINE,
    SPAN_NO_LINE,
    SPAN_SENTENCE,
    SPAN_WORD,
    SVC_PAY,
    VERSIONS,
    documents,
    empty_report,
    sample_graph,
    sample_report,
)


def _view(**changes: object) -> EvidenceView:
    return build_evidence(
        sample_report(**changes),
        sample_graph(),
        versions=VERSIONS,
        documents=documents(),
        org_name="  별빛\n상점  ",
    )


# --------------------------------------------------------------------------- formatting helpers


@pytest.mark.parametrize(
    ("minutes", "expected"),
    [
        (0, "0분"),
        (5, "5분"),
        (60, "1시간"),
        (90, "1시간 30분"),
        (1440, "24시간"),
        (2.5, "2.5분"),
        (-30, "-30분"),
        (None, "미정"),
        (math.inf, "미정"),
        (math.nan, "미정"),
    ],
)
def test_format_minutes(minutes: float | None, expected: str) -> None:
    assert format_minutes(minutes) == expected


def test_format_datetime_converts_to_kst_and_labels_naive_values() -> None:
    assert format_datetime(datetime(2026, 10, 20, 15, 30, tzinfo=UTC)) == "2026-10-21 00:30:00 KST"
    assert format_datetime(datetime(2026, 10, 20, 9, 0)) == "2026-10-20 09:00:00 (시간대 미기재)"


def test_short_hash_is_twelve_hex_digits() -> None:
    assert short_hash("0123456789abcdef") == "0123456789ab"


def test_printable_drops_terminal_and_bidi_controls_but_keeps_korean() -> None:
    assert printable("결제\x1b[31m 장애\u202e\u200b") == "결제[31m 장애"
    assert printable("줄\n바꿈\t탭\u2028끝") == "줄 바꿈 탭 끝"
    assert printable("깨진\ud800바이트") == "깨진\ufffd바이트"
    plain = "결제 장애 — 김대리에게 연락 (5분)"
    assert printable(plain) is plain


@given(st.text())
def test_printable_output_has_no_control_or_format_characters(text: str) -> None:
    cleaned = printable(text)
    assert not any(unicodedata.category(ch) in ("Cc", "Cf", "Cs") for ch in cleaned)
    assert printable(cleaned) == cleaned
    cleaned.encode("utf-8")


def test_mask_excerpt_clips_to_the_published_limit() -> None:
    long = "가" * 100
    masked = mask_excerpt(long)
    assert len(masked) == SECURITY_EXCERPT_CHARS
    assert masked.endswith("…")
    assert mask_excerpt("짧은\n지시") == "짧은 지시"


@given(st.text(), st.integers(min_value=1, max_value=200))
def test_mask_excerpt_never_exceeds_limit(text: str, limit: int) -> None:
    assert len(mask_excerpt(text, limit)) <= limit


def test_distinct_spans_drops_duplicates_and_contained_spans_in_document_order() -> None:
    spans = [SPAN_MULTILINE, SPAN_WORD, SPAN_SENTENCE, SPAN_WORD, SPAN_LEGAL, SPAN_NO_LINE]
    assert distinct_spans(spans) == [SPAN_SENTENCE, SPAN_MULTILINE, SPAN_LEGAL, SPAN_NO_LINE]
    same_range_other_doc = SPAN_WORD.model_copy(update={"doc_id": "doc-other"})
    assert same_range_other_doc in distinct_spans([SPAN_SENTENCE, same_range_other_doc])


@given(st.permutations([SPAN_MULTILINE, SPAN_WORD, SPAN_SENTENCE, SPAN_LEGAL, SPAN_NO_LINE]))
def test_distinct_spans_ignores_input_order(spans: list[SourceSpan]) -> None:
    assert distinct_spans(spans) == [SPAN_SENTENCE, SPAN_MULTILINE, SPAN_LEGAL, SPAN_NO_LINE]


def test_llm_usage_line_discloses_mode_model_calls_and_tokens() -> None:
    line = llm_usage_line(sample_report().llm)
    assert line == (
        "녹화 재생(결정적) · anthropic / claude-opus-5-5 · 호출 12회 · "
        "토큰 입력 34,567 · 출력 890 · 캐시 읽기 1,200 · 작업별 draft 2회, extract 10회"
    )
    assert llm_usage_line(LlmUsageSummary(mode=LlmMode.OFF)).startswith("끔(규칙 파서만 사용) · 호출 0회")


def test_judgement_path_statement_flags_any_llm_call_as_a_violation() -> None:
    assert judgement_path_statement(LlmUsageSummary(mode=LlmMode.OFF)) == (
        "판정·점수·리허설 경로 LLM 호출 0회 (고정 규칙과 결정적 리플레이만 사용)"
    )
    violation = judgement_path_statement(LlmUsageSummary(mode=LlmMode.LIVE, judgement_path_calls=3))
    assert "3회" in violation
    assert "위반" in violation


# --------------------------------------------------------------------------- overview and services


def test_overview_fields() -> None:
    view = _view()
    assert view.org_name == "별빛 상점"
    assert view.graph_hash == GRAPH_HASH[:12]
    assert view.created_at == "2026-10-20 10:02:03 KST"
    assert view.scoring_name == "SCORING_V1"
    assert view.broken_by_severity == ((Severity.CRITICAL, 2), (Severity.HIGH, 1), (Severity.LOW, 1))
    assert view.broken_summary == "4건 (심각 2 · 높음 1 · 낮음 1) · 확인 불가 2건"
    assert view.grounding.startswith("원문 근거율 93.0% — 원문 위치에 정렬된 요소 40건")
    assert [d.title for d in view.documents] == ["결제 런북", "보안 | 신고 <절차>"]
    assert view.catalog_gaps == ("service:배송",)


@pytest.mark.parametrize("org_name", [None, "", "   \n\t", "\x1b"])
def test_blank_org_name_is_absent(org_name: str | None) -> None:
    assert build_evidence(sample_report(), sample_graph(), org_name=org_name).org_name is None


def test_services_are_ordered_by_tier_then_weakest_and_missing_elements_stay_missing() -> None:
    rows = service_rows(sample_report())
    assert [(r.name, r.tier_label) for r in rows] == [("결제", "핵심"), ("주문|정산", "중요")]
    assert rows[0].statuses == (
        ElementStatus.PARTIAL,
        ElementStatus.FAIL,
        ElementStatus.UNKNOWN,
        ElementStatus.FAIL,
        ElementStatus.PASS,
    )
    assert rows[1].statuses[-1] is None


def test_unknown_elements_are_listed_for_human_review() -> None:
    (row,) = _view().unknown_elements
    assert (row.service, row.element) == ("결제", "승인자")
    assert row.reasons == ("승인자가 판단 보류 상태", "명부 | 미확인")


# --------------------------------------------------------------------------- findings


def test_broken_findings_are_grouped_by_severity_and_numbered_in_order() -> None:
    view = _view()
    numbered = [(sev, [(v.number, v.finding.id) for v in group]) for sev, group in view.broken]
    assert numbered == [
        (Severity.CRITICAL, [("3.1", F_LEGAL.id), ("3.2", F_DEPARTED.id)]),
        (Severity.HIGH, [("3.3", F_HOSTILE.id)]),
        (Severity.LOW, [("3.4", F_ONCALL.id)]),
    ]
    assert [(v.number, v.finding.id) for v in view.unknown] == [("6.1", F_HELD.id), ("6.2", F_INFO.id)]
    assert all(v.status == "확인 불가" for v in view.unknown)


def test_finding_view_resolves_names_quotes_and_fix() -> None:
    view = _view()
    departed = view.broken[0][1][1]
    assert departed.service == "결제"
    assert departed.severity == "심각"
    assert departed.status == "끊김"
    assert departed.fix == "퇴사자를 재직자로 교체"
    assert departed.subjects == "김민준, 결제 → 김민준 (담당)"
    assert [q.location for q in departed.quotes] == ["결제 런북:3", "결제 런북:5"]
    assert departed.quotes[1].text == SPAN_MULTILINE.quote

    legal = view.broken[0][1][0]
    assert legal.service == "조직 전체"
    assert [q.location for q in legal.quotes] == [
        "보안 | 신고 <절차>:13",
        "보안 | 신고 <절차> (문자 200–210)",
    ]

    hostile = view.broken[1][1][0]
    assert hostile.subjects is None  # the only subject is the finding's own service
    assert hostile.quotes == ()


def test_subjects_are_capped_with_a_remainder_count() -> None:
    many = F_DEPARTED.model_copy(update={"subject_ids": tuple(f"person:p{i}" for i in range(9))})
    view = build_evidence(sample_report(findings=(many,)), sample_graph())
    assert (
        view.broken[0][1][0].subjects
        == "person:p0, person:p1, person:p2, person:p3, person:p4, person:p5 외 3개"
    )


def test_long_quotes_are_clipped_but_located_exactly() -> None:
    span = SourceSpan(doc_id=PAY_DOC, start=0, end=500, quote="가" * 500, line=7)
    finding = F_DEPARTED.model_copy(update={"evidence": (span,)})
    quote = build_evidence(sample_report(findings=(finding,)), sample_graph()).broken[0][1][0].quotes[0]
    assert len(quote.text) == 300
    assert quote.text.endswith("…")
    assert quote.location == "결제 런북:7"


def test_explicit_documents_override_graph_titles() -> None:
    renamed = DocumentRef(doc_id=PAY_DOC, title="새 결제 런북", sha256="c" * 64, path=None)
    view = build_evidence(sample_report(), sample_graph(), documents=[renamed])
    assert view.broken[0][1][1].quotes[0].location == "새 결제 런북:3"
    assert ("새 결제 런북", None) in [(d.title, d.path) for d in view.documents]


@settings(max_examples=30)
@given(st.permutations(FINDINGS), st.permutations(REHEARSALS))
def test_view_does_not_depend_on_input_order(findings: list[object], rehearsals: list[object]) -> None:
    shuffled = sample_report(findings=tuple(findings), rehearsals=tuple(rehearsals))
    reference = build_evidence(sample_report(), sample_graph(), versions=VERSIONS)
    assert build_evidence(shuffled, sample_graph(), versions=tuple(reversed(VERSIONS))) == reference


# --------------------------------------------------------------------------- rehearsals


def test_rehearsal_rows_pick_the_current_graph_and_compare_before_after() -> None:
    rows = {r.scenario_id: r for r in _view().rehearsals}
    assert list(rows) == ["sc-base", "sc-breach", "sc-dc", "sc-ghost", "sc-late"]

    base = rows["sc-base"]
    assert base.outcome is Outcome.PASS
    assert base.comparison == "v1 실패 → v2 통과"
    assert base.ended_at == "T+30분"
    assert base.broken_at is None

    breach = rows["sc-breach"]
    assert breach.incident == "침해사고"
    assert breach.broken_at == "KISA 침해사고 신고"
    assert breach.comparison == "승인 시 실패 → 통과 예상 (수정안 patch-2)"
    assert breach.ended_at == "T+8분 (중단)"

    assert rows["sc-dc"].location_down
    assert not rows["sc-base"].location_down
    assert rows["sc-late"].ended_at == "T+1시간 35분 (시한 초과)"
    assert rows["sc-late"].broken_at == "롤백"

    ghost = rows["sc-ghost"]
    assert (ghost.title, ghost.service, ghost.broken_at) == ("sc-ghost", "확인 불가", "기록 없음")
    assert _view().rehearsal_summary == "5건 중 2건 통과 (통과율 40.0%)"


def test_current_rehearsals_falls_back_to_newest_known_version() -> None:
    newer = REHEARSALS[0].model_copy(update={"graph_hash": "1" * 64})
    report = sample_report(graph_hash="9" * 64, rehearsals=(REHEARSALS[1], newer))
    ((current, before),) = current_rehearsals(report, {OLD_HASH: 1, "1" * 64: 2})
    assert current.graph_hash == "1" * 64
    assert before is not None
    assert before.graph_hash == OLD_HASH


def test_current_rehearsals_prefers_known_versions_for_the_before_result() -> None:
    unknown_old = REHEARSALS[1].model_copy(update={"graph_hash": "7" * 64})
    report = sample_report(rehearsals=(unknown_old, REHEARSALS[0], REHEARSALS[1]))
    ((current, before),) = current_rehearsals(report, {OLD_HASH: 1})
    assert current.graph_hash == GRAPH_HASH
    assert before is not None
    assert before.graph_hash == OLD_HASH
    ((_, only_unknown),) = current_rehearsals(sample_report(rehearsals=(unknown_old, REHEARSALS[0])))
    assert only_unknown is not None
    assert only_unknown.graph_hash == "7" * 64


def test_comparison_without_version_history_uses_approved_patch_validation() -> None:
    report = sample_report(rehearsals=(REHEARSALS[0],))
    (row,) = build_evidence(report, sample_graph()).rehearsals
    assert row.comparison == "실패 → 통과 (수정안 patch-1 반영)"


def test_unknown_previous_graph_is_labelled_by_hash() -> None:
    (row,) = build_evidence(sample_report(rehearsals=REHEARSALS[:2]), sample_graph()).rehearsals
    assert row.comparison == f"이전 그래프({OLD_HASH[:12]}) 실패 → v2 통과"


# --------------------------------------------------------------------------- history and patches


def test_version_rows_show_sentence_approver_time_and_resolved_findings() -> None:
    v1, v2 = _view().versions
    assert (v1.number, v1.change, v1.approver, v1.sentence) == (
        1,
        "최초 추출 — dmc-shop 최초 추출",
        "system",
        None,
    )
    assert v1.at == "2026-10-20 00:00:00 KST"
    assert v2.change == "수정안 patch-1 반영"
    assert v2.sentence == P_APPROVED.runbook_text
    assert (v2.approver, v2.at, v2.resolved) == ("박재현", "2026-10-21 09:30:00 KST", "JT-PPL-001 (1건)")


def test_revert_versions_are_labelled() -> None:
    revert = VERSIONS[1].model_copy(update={"number": 3, "parent": 2, "revert_of": 2, "note": ""})
    view = build_evidence(sample_report(), sample_graph(), versions=(*VERSIONS, revert))
    assert view.versions[-1].change == "v2 되돌리기"


def test_patches_not_in_any_version_are_listed_with_their_decision() -> None:
    rows = {p.patch_id: p for p in _view().other_patches}
    assert list(rows) == ["patch-2", "patch-3"]
    assert rows["patch-2"].status == "검증 통과(승인 대기)"
    assert rows["patch-2"].author == "생성형 AI 초안"
    assert rows["patch-2"].targets == "JT-DDL-002, JT-INT-001 (2건)"
    assert rows["patch-2"].decided is None
    assert rows["patch-3"].decided == "한서윤 · 시각 기록 없음"
    assert rows["patch-3"].targets == "—"


def test_explicit_patches_override_the_report_copy() -> None:
    approved = P_DRAFT.model_copy(update={"status": PatchStatus.APPROVED, "decided_by": "한서윤"})
    # the breach drill passes on the reported graph, as it must for the approved fix to be cited
    rehearsals = tuple(
        r.model_copy(update={"outcome": Outcome.PASS}) if r.scenario_id == "sc-breach" else r
        for r in REHEARSALS
    )
    report = sample_report(rehearsals=rehearsals)
    view = build_evidence(report, sample_graph(), versions=VERSIONS, patches=[approved])
    assert {p.patch_id: p.status for p in view.other_patches}["patch-2"] == "승인"
    breach = next(r for r in view.rehearsals if r.scenario_id == "sc-breach")
    assert breach.comparison == "실패 → 통과 (수정안 patch-2 반영)"
    assert breach.compared


# --------------------------------------------------------------------------- held, security, controls


def test_held_rows_are_unique_named_and_never_invented() -> None:
    held = _view().held
    assert [h.ref_id for h in held] == [E_ESC, "node:missing"]
    assert held[0].kind == "관계(상향)"
    assert held[0].label == f"김민준 → {SCRIPT} (상향)"
    assert held[0].reason == "역번역 대조 불일치 | 숫자 슬롯"
    assert (held[1].kind, held[1].reason) == ("그래프에 없음", "사유 기록 없음")


def test_security_rows_label_patterns_and_mask_the_excerpt() -> None:
    (row,) = _view().security
    assert row.document == "보안 | 신고 <절차>"
    assert row.location == "문자 70–144"
    assert row.patterns == "AI를 향한 지시 · 점수 조작 지시 · 숨김 텍스트(HTML 주석 등) · brand-new-detector"
    assert len(row.excerpt) == SECURITY_EXCERPT_CHARS
    assert row.excerpt.endswith("…")
    assert "기록하라" not in row.excerpt


def test_control_mapping_for_the_sample() -> None:
    controls = {c.control_id: c for c in _view().controls}
    assert list(controls) == ["2.11.4", "2.11.5", "2.12.1", "2.12.2"]
    assert controls["2.11.4"].status is ControlStatus.GAP
    assert "모의 장애 5건 결정적 재생 · 통과 2건 · 실패 3건" in controls["2.11.4"].evidence
    # sc-base was replayed before and after patch-1; sc-breach only has a pending patch's forecast
    assert "수정 전후 비교 1건" in controls["2.11.4"].evidence
    assert controls["2.11.5"].status is ControlStatus.GAP
    assert "(JT-DDL-002)" in controls["2.11.5"].evidence
    assert "신고 리허설 1건 중 통과 0건" in controls["2.11.5"].evidence
    assert controls["2.12.1"].status is ControlStatus.OK
    assert controls["2.12.1"].evidence == "JT-CAT-003 0건 · JT-CHN-001 0건 · 센터 장애 시나리오 1건(실패 0건)"
    assert controls["2.12.2"].status is ControlStatus.OK


def test_controls_without_evidence_are_missing_never_satisfied() -> None:
    view = build_evidence(empty_report(), sample_graph())
    assert {c.control_id: c.status for c in view.controls} == {
        "2.11.4": ControlStatus.MISSING,
        "2.11.5": ControlStatus.MISSING,
        "2.12.1": ControlStatus.MISSING,
        "2.12.2": ControlStatus.MISSING,
    }


def test_disaster_findings_and_passing_reporting_rehearsals_change_the_mapping() -> None:
    chn = F_HOSTILE.model_copy(update={"rule_id": "JT-CHN-001", "id": "JT-CHN-001:x"})
    passing = tuple(
        r.model_copy(update={"outcome": Outcome.PASS}) for r in REHEARSALS if r.graph_hash == GRAPH_HASH
    )
    controls = {
        c.control_id: c.status
        for c in build_evidence(sample_report(findings=(chn,), rehearsals=passing), sample_graph()).controls
    }
    assert controls == {
        "2.11.4": ControlStatus.OK,
        "2.11.5": ControlStatus.OK,
        "2.12.1": ControlStatus.GAP,
        "2.12.2": ControlStatus.OK,
    }


# --------------------------------------------------------------------------- appendices


def test_rule_rows_cover_the_whole_catalog_with_counts() -> None:
    rules = _view().rules
    assert [r.rule_id for r in rules] == sorted(info.rule_id for info in rule_catalog())
    counts = {r.rule_id: r.count for r in rules}
    assert counts["JT-PPL-001"] == 1
    assert counts["JT-ESC-001"] == 0


def test_weights_come_from_the_report_and_fall_back_to_scoring_v1() -> None:
    view = _view()
    assert [(w.key, w.weight) for w in view.element_weights] == [
        (e.value, SCORING_V1["element_weights"][e.value]) for e in Element
    ]
    assert [w.label for w in view.tier_weights] == ["1등급(핵심)", "2등급(중요)", "3등급(일반)"]
    assert build_evidence(empty_report(), sample_graph()).element_weights == view.element_weights

    custom = ReadinessReport(
        overall=0.0,
        services=(),
        weights_version="v2",
        element_weights={"zeta": 0.5, Element.OWNER.value: 0.5},
        tier_weights={4: 0.5},
        graph_hash=GRAPH_HASH,
    )
    custom_view = build_evidence(sample_report(readiness=custom), sample_graph())
    assert [(w.label, w.key) for w in custom_view.element_weights] == [("담당", "owner"), ("zeta", "zeta")]
    assert [w.label for w in custom_view.tier_weights] == ["4등급(기타)"]
    assert custom_view.scoring_name == "SCORING_V2"


def test_build_is_deterministic() -> None:
    assert _view() == _view()
    assert build_evidence(sample_report(), sample_graph()).services[0].service_id == SVC_PAY


def test_rehearsal_rate_without_rehearsals_is_not_applicable() -> None:
    view = build_evidence(empty_report(), sample_graph())
    assert view.rehearsal_rate == "해당 없음"
    assert view.broken_summary == "0건 · 확인 불가 0건"
    assert view.severity_summary == ""
    assert SC_BASE.id not in {r.scenario_id for r in view.rehearsals}
