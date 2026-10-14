"""mask_pii: length-preserving masking that never touches times, dates, versions or HTTP codes."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jettae.ingest import PiiHit, PiiKind, mask_pii


@pytest.mark.parametrize(
    ("text", "expected", "kind"),
    [
        ("010-1234-5678", "010-****-5678", PiiKind.MOBILE),
        ("01012345678", "010****5678", PiiKind.MOBILE),
        ("011-123-4567", "011-***-4567", PiiKind.MOBILE),
        ("010.1234.5678", "010.****.5678", PiiKind.MOBILE),
        ("010 1234 5678", "010 **** 5678", PiiKind.MOBILE),
        ("+82-10-1234-5678", "+82-10-****-5678", PiiKind.MOBILE),
        ("02-123-4567", "02-***-4567", PiiKind.LANDLINE),
        ("031-1234-5678", "031-****-5678", PiiKind.LANDLINE),
        ("(02) 123-4567", "(02) ***-4567", PiiKind.LANDLINE),
        ("070-1234-5678", "070-****-5678", PiiKind.LANDLINE),
        ("kimjihoon@dmc.co.kr", "k********@dmc.co.kr", PiiKind.EMAIL),
        ("oncall+pay@example.com", "o*********@example.com", PiiKind.EMAIL),
        ("192.168.0.15", "192.***.*.**", PiiKind.IPV4),
        ("10.0.0.1", "10.*.*.*", PiiKind.IPV4),
        ("2001:db8::1", "2001:***::*", PiiKind.IPV6),
        ("fe80::1ff:fe23:4567:890a", "fe80::***:****:****:****", PiiKind.IPV6),
        ("2001:0db8:85a3:0000:0000:8a2e:0370:7334", "2001:****:****:****:****:****:****:****", PiiKind.IPV6),
        ("900101-1234567", "******-*******", PiiKind.RRN),
        ("1234-5678-9012-3456", "****-****-****-3456", PiiKind.CARD),
        ("1234 5678 9012 3456", "**** **** **** 3456", PiiKind.CARD),
        ("4111111111111111", "************1111", PiiKind.CARD),
    ],
)
def test_masks_each_kind(text: str, expected: str, kind: PiiKind) -> None:
    masked, hits = mask_pii(text)
    assert masked == expected
    assert hits == [PiiHit(kind, 0, len(text))]


def test_bank_account_masked_only_when_labelled() -> None:
    labelled = "환불 계좌: 국민 123456-01-234567"
    masked, hits = mask_pii(labelled)
    assert masked == "환불 계좌: 국민 ******-**-**4567"
    assert [h.kind for h in hits] == [PiiKind.ACCOUNT]
    assert mask_pii("배포 번호 123456-01-234567") == ("배포 번호 123456-01-234567", [])


def test_short_labelled_digit_run_is_not_an_account() -> None:
    assert mask_pii("계좌 이체 한도 50000원")[1] == []


@pytest.mark.parametrize(
    "text",
    [
        "10분 미응답 시 팀장에게 상향",
        "KISA에 24시간 이내 신고",
        "점검일 2026-10-28 09:00 ~ 2026-10-28 18:30",
        "2026.10.28 배포",
        "v1.2.3 으로 롤백",
        "버전 v1.2.3.4 배포",
        "5xx/503 비율 5% 초과",
        "HTTP 503 Service Unavailable",
        "1.2.3.4.5",
        "10:30:00 에 재시작",
        "std::vector 와 add::bead 는 코드",
        "대표번호 1588-1234",
        "주문번호 1234567890123",
        "주문번호 1234567890123456",  # 16 digits but fails the Luhn check
        "RTO 4h, RPO 15m, p99 300ms",
        "티켓 INC-20261028-0042",
    ],
)
def test_does_not_mask_times_dates_versions_codes(text: str) -> None:
    assert mask_pii(text) == (text, [])


def test_korean_particles_glued_to_tokens_still_masked() -> None:
    masked, hits = mask_pii("서버 10.20.30.40에서 kim@dmc.co.kr로 보고, 김대리(010-1234-5678)에게 전화")
    assert masked == "서버 10.**.**.**에서 k**@dmc.co.kr로 보고, 김대리(010-****-5678)에게 전화"
    assert [h.kind for h in hits] == [PiiKind.IPV4, PiiKind.EMAIL, PiiKind.MOBILE]


def test_hits_sorted_and_offsets_index_original_text() -> None:
    text = "a 010-1234-5678\nb kim@x.io\nc 10.1.2.3"
    masked, hits = mask_pii(text)
    assert [h.start for h in hits] == sorted(h.start for h in hits)
    for hit in hits:
        assert text[hit.start : hit.end] != masked[hit.start : hit.end]
    assert masked.split("\n")[0] == "a 010-****-5678"


def test_email_wins_over_digits_inside_it() -> None:
    masked, hits = mask_pii("ops01012345678@dmc.co.kr")
    assert [h.kind for h in hits] == [PiiKind.EMAIL]
    assert masked.endswith("@dmc.co.kr")


def test_masking_never_spans_lines() -> None:
    text = "(02)\n123-4567"
    masked, _ = mask_pii(text)
    assert masked.index("\n") == text.index("\n")


_PII_ALPHABET = st.sampled_from(
    [
        "0", "1", "2", "5", "8", "9", "-", ".", " ", "@", ":", "(", ")", "+",
        "a", "k", "x", "Z", "결", "제", "\n", "010", "02-", ".co.kr", "계좌", "분", "v",
    ]
)  # fmt: skip


@settings(max_examples=400, deadline=None)
@given(st.lists(_PII_ALPHABET, max_size=80).map("".join))
def test_property_length_and_structure_preserved(text: str) -> None:
    masked, hits = mask_pii(text)
    assert len(masked) == len(text)
    covered: set[int] = set()
    previous_end = 0
    for hit in hits:
        assert 0 <= hit.start < hit.end <= len(text)
        assert hit.start >= previous_end  # sorted and non-overlapping
        previous_end = hit.end
        covered.update(range(hit.start, hit.end))
    for i, (orig, new) in enumerate(zip(text, masked, strict=True)):
        if i not in covered:
            assert orig == new
        else:
            assert new in (orig, "*")
            assert orig != "\n"


@settings(max_examples=200, deadline=None)
@given(
    st.integers(min_value=0, max_value=9999),
    st.integers(min_value=0, max_value=9999),
    st.sampled_from(["-", "", " ", "."]),
)
def test_property_mobile_numbers_keep_prefix_and_last_four(mid: int, last: int, sep: str) -> None:
    number = f"010{sep}{mid:04d}{sep}{last:04d}"
    masked, hits = mask_pii(f"담당 {number} 연락")
    assert masked == f"담당 010{sep}****{sep}{last:04d} 연락"
    assert [h.kind for h in hits] == [PiiKind.MOBILE]


@settings(max_examples=200, deadline=None)
@given(
    st.integers(min_value=0, max_value=999),
    st.sampled_from(["분", "시간", "초", "일", "ms", "%"]),
    st.dates(),
)
def test_property_durations_and_dates_are_never_masked(n: int, unit: str, day: object) -> None:
    text = f"{n}{unit} 이내, {day} 기준"
    assert mask_pii(text) == (text, [])


@settings(max_examples=200, deadline=None)
@given(
    st.from_regex(r"[a-z][a-z0-9._]{0,10}[a-z0-9]", fullmatch=True),
    st.sampled_from(["dmc.co.kr", "example.com", "ops.io"]),
)
def test_property_emails_keep_first_char_and_domain(local: str, domain: str) -> None:
    masked, hits = mask_pii(f"메일: {local}@{domain}")
    assert masked == f"메일: {local[0]}{'*' * (len(local) - 1)}@{domain}"
    assert [h.kind for h in hits] == [PiiKind.EMAIL]


# --------------------------------------------------------------------------- review: edge cases


@pytest.mark.parametrize(
    "text",
    [
        "버전 4.12.0.3 배포",
        "롤백 대상 버전은 2.14.0.1",
        "release 1.10.2.0",
        "Version: 10.0.12.4",
        "빌드 3.21.0.15 확인",
    ],
)
def test_labelled_four_part_versions_are_not_masked_as_ip(text: str) -> None:
    """Masking "2.14.0.1" would hide the rollback target the runbook names."""
    assert mask_pii(text) == (text, [])


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("server 10.0.0.12", "server 10.*.*.**"),
        ("Server: 10.0.0.12", "Server: 10.*.*.**"),
        ("IPv4 10.0.0.12", "IPv4 10.*.*.**"),
        ("dev 10.0.0.12", "dev 10.*.*.**"),
    ],
)
def test_words_ending_in_version_like_letters_do_not_unmask_ips(text: str, expected: str) -> None:
    assert mask_pii(text)[0] == expected


def test_empty_and_pii_free_text() -> None:
    assert mask_pii("") == ("", [])
    assert mask_pii("결제 장애 시 담당자에게 연락") == ("결제 장애 시 담당자에게 연락", [])


def test_hangul_glued_on_both_sides_of_tokens() -> None:
    text = "김대리01012345678에게, 메일은kim@dmc.co.kr로"
    masked, hits = mask_pii(text)
    assert masked == "김대리010****5678에게, 메일은k**@dmc.co.kr로"
    assert [h.kind for h in hits] == [PiiKind.MOBILE, PiiKind.EMAIL]


def test_masking_is_deterministic() -> None:
    text = "담당 010-1234-5678, 10.0.0.1, kim@dmc.co.kr, 카드 1234-5678-9012-3456"
    assert mask_pii(text) == mask_pii(text)
