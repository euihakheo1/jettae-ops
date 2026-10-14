"""Rule-parser patterns for the wording real runbooks use beyond "key: value" lines.

Each test pairs a pattern with its own hand-written Korean examples (none taken from the evaluation
corpus) and, where a pattern could over-read, with the near miss it must leave alone: the parser
may only emit what a pattern can point at (P2), and absent facts stay absent (P3).
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from jettae.domain import (
    AttrKey,
    CatalogService,
    Document,
    EdgeKind,
    Extraction,
    NodeKind,
    RosterEntry,
)
from jettae.extract.korean import Lexicon, find_mentions
from jettae.extract.rules import RuleExtractor

# --------------------------------------------------------------------------- helpers


def _extract(
    text: str, *, roster: Sequence[RosterEntry] = (), catalog: Sequence[CatalogService] = ()
) -> Extraction:
    doc = Document(doc_id=Document.make_id("runbook", text), title="runbook", text=text)
    ex = RuleExtractor().extract(doc, roster=roster, catalog=catalog)
    for item in [*ex.entities, *ex.relations]:  # P2: every quote is an exact substring
        assert item.quote in text, item
        for attr in item.attrs:
            assert attr.quote in text, attr
    return ex


def _triples(ex: Extraction, *kinds: EdgeKind) -> set[tuple[EdgeKind, str, str]]:
    return {(r.kind, r.src.name, r.dst.name) for r in ex.relations if not kinds or r.kind in kinds}


def _pairs(ex: Extraction, kind: EdgeKind) -> set[tuple[str, str]]:
    return {(s, d) for _, s, d in _triples(ex, kind)}


def _channel_of(ex: Extraction, src: str, dst: str) -> object:
    found = [
        r for r in ex.relations if r.kind is EdgeKind.NOTIFIES and r.src.name == src and r.dst.name == dst
    ]
    assert found, f"missing notifies {src} -> {dst}; have {sorted(_triples(ex, EdgeKind.NOTIFIES))}"
    return next((a.value for a in found[0].attrs if a.key == AttrKey.CHANNEL.value), None)


def _detected(ex: Extraction) -> set[str]:
    return {s for s, _ in _pairs(ex, EdgeKind.DETECTED_BY)}


def _deadlines(ex: Extraction) -> set[object]:
    return {
        a.value
        for e in ex.entities
        if e.kind is NodeKind.OBLIGATION
        for a in e.attrs
        if a.key == AttrKey.DEADLINE_MINUTES.value
    }


ROSTER = (
    RosterEntry(name="윤하람", title="대리", aliases=("하람", "Hailey")),
    RosterEntry(name="서도윤", title="팀장", aliases=("도윤",)),
    RosterEntry(name="국민재", title="CISO", aliases=("Leo",)),
)

# --------------------------------------------------------------------------- mentions


def test_a_roster_name_keeps_its_honorific_in_the_span_not_in_the_name() -> None:
    lexicon = Lexicon(roster=ROSTER)
    found = find_mentions("Hailey님한테 연락", lexicon)
    assert [(m.kind, m.name, m.end) for m in found] == [(NodeKind.PERSON, "Hailey", len("Hailey님"))]
    ex = _extract("## 정산 서비스\nHailey님한테 문자로 연락해 주세요", roster=ROSTER)
    assert ("정산", "Hailey") in _pairs(ex, EdgeKind.OWNED_BY)
    assert ("Hailey", "문자") in _pairs(ex, EdgeKind.REACHABLE_VIA)


@pytest.mark.parametrize(
    "text", ["정산 담당은 남대리예요", "정산 담당은 남대리입니다", "정산 담당은 남대리였어요"]
)
def test_a_name_may_end_in_a_copula(text: str) -> None:
    ex = _extract(f"## 정산 서비스\n{text}")
    assert _pairs(ex, EdgeKind.OWNED_BY) == {("정산", "남대리")}


# --------------------------------------------------------------------------- contacts and triggers


def test_a_call_with_its_channel_in_parentheses() -> None:
    ex = _extract("## 예약 서비스\n- 감지: 예약 API 오류율 3% 초과 시 알람 → 표지민 과장 호출(카카오톡)")
    assert _channel_of(ex, "예약 API 오류율 3% 초과", "표지민 과장") == "카카오톡"
    assert ("표지민 과장", "카카오톡") in _pairs(ex, EdgeKind.REACHABLE_VIA)


def test_a_passive_call_names_who_receives_it() -> None:
    ex = _extract(
        "## 예약 서비스\n예약 대기열이 2천 건을 넘으면 알람이 울립니다. "
        "그러면 하람 대리가 문자로 호출을 받습니다."
    )
    assert "예약" in _detected(ex)
    notified = _pairs(ex, EdgeKind.NOTIFIES)
    assert [d for _, d in notified] == ["하람 대리"]
    assert _channel_of(ex, next(iter(notified))[0], "하람 대리") == "문자"
    # receiving a page is the alarm's route, not an assignment: no owner is guessed from it
    assert not _pairs(ex, EdgeKind.OWNED_BY)


def test_the_paged_person_is_notified_before_the_first_contact() -> None:
    ex = _extract(
        "## 예약 서비스\n장애가 나면 서도윤 팀장에게 먼저 공유한다.\n"
        "예약 실패가 5분 이어지면 알람이 울린다. 하람 대리가 슬랙으로 호출을 받는다."
    )
    alarm = next(d for _, d in _pairs(ex, EdgeKind.DETECTED_BY) if "예약 실패" in d)
    assert _channel_of(ex, alarm, "하람 대리") == "슬랙"
    assert (alarm, "서도윤 팀장") not in _pairs(ex, EdgeKind.NOTIFIES)


@pytest.mark.parametrize(
    "line",
    [
        "정산 배치가 40분 넘게 밀리면 알람이 울립니다.",
        "알람은 정산 지연 상태가 감지되면 바로 떠요.",
        "모니터링에서 정산 실패가 3회 이어지면 알람이 와요.",
        "정산 파일 누락 2건 이상 10분 지속 시 즉시 알람",
    ],
)
def test_an_alarm_under_any_stated_condition_is_a_detection(line: str) -> None:
    ex = _extract(f"## 정산 서비스\n{line}")
    assert _detected(ex) == {"정산"}


@pytest.mark.parametrize(
    "line",
    [
        "배송 지연 시 고객에게 알림 문자를 보낸다.",  # a customer notice, not a monitoring alarm
        "알람 설정 방법은 위키를 참고한다.",
        "정산이 늦어지면 다음 날 다시 확인한다.",
    ],
)
def test_notices_and_alarm_chatter_are_no_detection(line: str) -> None:
    assert not _detected(_extract(f"## 정산 서비스\n{line}"))


def test_a_detection_cell_names_the_call_and_its_channel() -> None:
    text = (
        "## 예약 서비스\n\n| 항목 | 내용 |\n|---|---|\n| 담당 | 표지민 과장 |\n"
        "| 감지 | 좌석 동기화 실패 5분 지속 시 알람 → 표지민 과장 호출(문자) |\n"
    )
    ex = _extract(text)
    trigger = next(s for s, d in _pairs(ex, EdgeKind.NOTIFIES) if d == "표지민 과장")
    assert "→" not in trigger  # the condition ends where the call begins
    assert _channel_of(ex, trigger, "표지민 과장") == "문자"
    assert _pairs(ex, EdgeKind.OWNED_BY) == {("예약", "표지민 과장")}


def test_adverbs_between_the_recipient_and_the_channel() -> None:
    ex = _extract("## 예약 서비스\n표지민 과장한테는 평소엔 팀즈로 연락하시면 돼요")
    assert ("표지민 과장", "팀즈") in _pairs(ex, EdgeKind.REACHABLE_VIA)


# --------------------------------------------------------------------------- service-labelled lines


LABELLED = """# 운영 런북

## 서비스별 기준

- 예약: 예약 실패율 2% 초과 5분 지속 시 알람 → 표지민 과장 호출(슬랙)
- 정산 서비스: 정산 파일 누락 1건 이상 시 즉시 알람 → 남대리 호출(전화)

## 신고

- 예약: 침해사고는 24시간 이내 KISA 신고(담당: 국민재 CISO)
"""


def test_a_line_led_by_a_service_name_states_that_services_facts() -> None:
    catalog = (CatalogService(name="예약"), CatalogService(name="정산"))
    ex = _extract(LABELLED, roster=ROSTER, catalog=catalog)
    assert _detected(ex) == {"예약", "정산"}
    assert {d for _, d in _pairs(ex, EdgeKind.NOTIFIES)} == {"표지민 과장", "남대리"}
    assert ("예약", "KISA 침해사고 신고") in _pairs(ex, EdgeKind.MUST_REPORT)
    assert ("KISA 침해사고 신고", "국민재 CISO") in _pairs(ex, EdgeKind.REPORTED_BY)
    # whom an alarm calls is not who owns the service, and the reporter owns nothing either
    assert not _pairs(ex, EdgeKind.OWNED_BY)


def test_a_label_that_is_not_only_a_service_opens_no_scope() -> None:
    ex = _extract(
        "- 예약 담당 변경: 다음 달부터 표지민 과장에게 연락", catalog=(CatalogService(name="예약"),)
    )
    assert not _pairs(ex, EdgeKind.OWNED_BY)


# --------------------------------------------------------------------------- owners and reporters


@pytest.mark.parametrize(
    "line",
    [
        "침해사고는 24시간 이내 KISA 신고(담당: 국민재 CISO)",
        "KISA 신고는 침해사고 기준 24시간 안에 해야 하는데, 국민재 CISO 담당이에요.",
        "침해사고가 의심되면 24시간 안에 KISA에 신고해야 하고, 신고는 Leo님이 맡아요.",
    ],
)
def test_the_person_in_charge_of_a_report_reports_and_owns_nothing(line: str) -> None:
    ex = _extract(f"## 예약 서비스\n담당: 표지민 과장\n{line}", roster=ROSTER)
    assert {d for _, d in _pairs(ex, EdgeKind.REPORTED_BY)} <= {"국민재 CISO", "Leo"}
    assert _pairs(ex, EdgeKind.REPORTED_BY)
    assert _pairs(ex, EdgeKind.OWNED_BY) == {("예약", "표지민 과장")}


def test_a_parenthesised_owner_after_the_service_still_owns_it() -> None:
    ex = _extract("## 예약 서비스\n예약 서비스 (담당: 표지민 과장)\n- 예약 DB 작업(담당: 남대리)")
    assert _pairs(ex, EdgeKind.OWNED_BY) == {("예약", "표지민 과장")}


def test_a_labelled_alarm_field_does_not_make_the_called_person_owner() -> None:
    fields = _extract(
        "## 예약 서비스\n- 감지: 예약 실패율 알람 → 표지민 과장 호출\n- 상향: 10분 미응답 시 서도윤 팀장"
    )
    assert not _pairs(fields, EdgeKind.OWNED_BY)
    assert ("표지민 과장", "서도윤 팀장") in _pairs(fields, EdgeKind.ESCALATES_TO)
    prose = _extract("## 예약 서비스\n예약 장애가 계속되면 표지민 과장에게 연락")
    assert _pairs(prose, EdgeKind.OWNED_BY) == {("예약", "표지민 과장")}  # free prose: unchanged


# --------------------------------------------------------------------------- escalation


@pytest.mark.parametrize(
    "second",
    [
        "거기서도 20분 넘게 조용하면 서도윤 팀장님한테 바로 연락하세요.",
        "20분 넘게 무응답이면 서도윤 팀장님께 전화로 연락하세요.",
        "표지민 과장님이 20분 안에 못 받으시면 그다음은 서도윤 팀장님이에요.",
    ],
)
def test_spoken_no_answer_conditions_escalate(second: str) -> None:
    ex = _extract(f"## 예약 서비스\n예약 쪽은 표지민 과장님이 맡고 있어요. {second}")
    assert ("표지민 과장", "서도윤 팀장") in _pairs(ex, EdgeKind.ESCALATES_TO)
    esc = next(r for r in ex.relations if r.kind is EdgeKind.ESCALATES_TO)
    assert next(a.value for a in esc.attrs if a.key == AttrKey.AFTER_MINUTES.value) == 20


# --------------------------------------------------------------------------- channels


@pytest.mark.parametrize(
    ("line", "pair"),
    [
        ("- 연락: 팀즈(1순위), 문자(2순위)", ("팀즈", "문자")),
        ("연락 수단은 슬랙(주), 전화(백업)", ("슬랙", "전화")),
        ("표지민 과장한테 카톡 먼저 보내시고, 답이 안 오면 전화로 다시 걸어 주세요", ("카카오톡", "전화")),
        ("표지민 과장한테는 평소엔 슬랙으로 연락하시고, 안 되면 문자로 하시면 돼요", ("슬랙", "문자")),
    ],
)
def test_ranked_and_sequential_channels_are_fallbacks(line: str, pair: tuple[str, str]) -> None:
    ex = _extract(f"## 예약 서비스\n{line}")
    assert pair in _pairs(ex, EdgeKind.FALLBACK_TO)


@pytest.mark.parametrize(
    "line",
    [
        "슬랙(공지용), 전화(긴급)",  # notes that rank nothing
        "슬랙으로 공유하고 해결이 안 되면 이메일로 경영진에 보고",  # unresolved is not unreachable
        "표지민 과장에게 슬랙으로 연락하고, 10분 응답이 없으면 전화로 서도윤 팀장에게 상향",  # escalation
    ],
)
def test_channel_lists_that_are_no_fallback(line: str) -> None:
    assert not _pairs(_extract(f"## 예약 서비스\n{line}"), EdgeKind.FALLBACK_TO)


def test_a_bare_contact_column_with_channels_feeds_the_fallback_column() -> None:
    text = (
        "| 서비스 | 담당 | 연락 | 대체 채널 |\n|---|---|---|---|\n"
        "| 예약 | 표지민 과장 | 팀즈 | 문자 |\n| 정산 | 남대리 | 슬랙 | - |\n"
    )
    ex = _extract(text)
    assert _pairs(ex, EdgeKind.FALLBACK_TO) == {("팀즈", "문자")}
    assert {("표지민 과장", "팀즈"), ("남대리", "슬랙")} <= _pairs(ex, EdgeKind.REACHABLE_VIA)
    assert _pairs(ex, EdgeKind.OWNED_BY) == {("예약", "표지민 과장"), ("정산", "남대리")}


# --------------------------------------------------------------------------- approval and execution


@pytest.mark.parametrize(
    ("line", "approvers"),
    [
        (
            "페일오버 판단은 서도윤 팀장님이나 국민재 CISO님이 하시고 실제 작업은 남대리가 해요.",
            {"서도윤 팀장", "국민재 CISO"},
        ),
        ("롤백 여부는 서도윤 팀장이 결정하고, 실행은 남대리가 맡는다.", {"서도윤 팀장"}),
    ],
)
def test_who_decides_on_an_action_approves_it(line: str, approvers: set[str]) -> None:
    ex = _extract(f"## 예약 서비스\n{line}", roster=ROSTER)
    assert {d for _, d in _pairs(ex, EdgeKind.APPROVED_BY)} == approvers
    assert {d for _, d in _pairs(ex, EdgeKind.EXECUTED_BY)} == {"남대리"}


def test_a_spoken_run_verb_names_the_executor() -> None:
    ex = _extract("## 예약 서비스\n롤백이 필요하면 서도윤 팀장님께 승인받은 다음에 남대리가 돌리면 됩니다.")
    assert _pairs(ex, EdgeKind.APPROVED_BY) == {("예약 롤백", "서도윤 팀장")}
    assert _pairs(ex, EdgeKind.EXECUTED_BY) == {("예약 롤백", "남대리")}


@pytest.mark.parametrize("key", ["페일오버", "재시작", "기능 플래그 OFF"])
def test_a_field_row_keyed_by_any_mitigation_names_that_action(key: str) -> None:
    text = (
        "## 예약 서비스\n\n| 항목 | 내용 |\n|---|---|\n| 담당 | 표지민 과장 |\n"
        f"| {key} | 서도윤 팀장 승인 후 남대리가 실행 |\n"
    )
    ex = _extract(text)
    (_, action), *rest = sorted(_pairs(ex, EdgeKind.MITIGATED_BY))
    assert not rest
    assert _pairs(ex, EdgeKind.APPROVED_BY) == {(action, "서도윤 팀장")}
    assert _pairs(ex, EdgeKind.EXECUTED_BY) == {(action, "남대리")}


def test_judgement_words_without_an_action_approve_nothing() -> None:
    ex = _extract("## 예약 서비스\n장애 등급 판단은 서도윤 팀장이 한다.")
    assert not _pairs(ex, EdgeKind.APPROVED_BY)


# --------------------------------------------------------------------------- obligations


def test_one_duty_stated_with_two_deadlines_keeps_both() -> None:
    text = (
        "## 예약 서비스\n침해사고는 24시간 이내 KISA 신고\n\n"
        "## 정산 서비스\n침해사고는 48시간 이내 KISA 신고\n"
    )
    ex = _extract(text)
    assert _deadlines(ex) == {24 * 60, 48 * 60}
    assert {s for s, _ in _pairs(ex, EdgeKind.MUST_REPORT)} == {"예약", "정산"}
