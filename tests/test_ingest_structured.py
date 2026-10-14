"""Structured sources: roster, on-call, policy, catalog (trusted facts → strict parsing)."""

from __future__ import annotations

import random
import unicodedata
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jettae.domain import (
    DEFAULT_OBLIGATIONS,
    CatalogService,
    ChannelMedium,
    FatiguePolicy,
    IncidentType,
    PersonStatus,
    Policy,
)
from jettae.ingest import (
    KST,
    load_catalog,
    load_oncall,
    load_policy,
    load_roster,
    merge_catalogs,
    parse_channel,
    parse_status,
)

ZWSP = "\N{ZERO WIDTH SPACE}"

# --------------------------------------------------------------------------- roster

ROSTER_CSV = """이름,상태,직급,직책,팀,역할,별칭,연락수단,퇴사일,연락처
김민수,재직,대리,,결제팀,1차 당직; 결제 담당,김대리/Minsu,슬랙|전화,,010-1234-5678
박지훈,퇴사,과장,팀장,결제팀,,박과장,카톡,2026.09.30,010-0000-0000
이서연,휴직,,CTO,경영,,,"메일, 문자, PagerDuty",,
최하늘,,사원,,인프라,,,사내메신저;팀즈;비둘기,,
"""


def test_roster_csv_korean_headers() -> None:
    roster = load_roster(ROSTER_CSV)
    assert [r.name for r in roster] == ["김민수", "박지훈", "이서연", "최하늘"]
    kim, park, lee, choi = roster
    assert kim.status is PersonStatus.ACTIVE
    assert kim.title == "대리"
    assert kim.team == "결제팀"
    assert kim.roles == ("1차 당직", "결제 담당")
    assert kim.aliases == ("김대리", "Minsu")
    assert kim.channels == (ChannelMedium.SLACK, ChannelMedium.PHONE)
    assert kim.left_on is None
    assert park.status is PersonStatus.DEPARTED
    assert park.title == "과장"
    assert park.roles == ("팀장",)  # 직책 joins roles when 직급 is the title
    assert park.left_on == "2026-09-30"
    assert lee.status is PersonStatus.ON_LEAVE
    assert lee.title == "CTO"  # only 직책 given
    assert lee.roles == ("CTO",)
    assert lee.channels == (ChannelMedium.EMAIL, ChannelMedium.SMS, ChannelMedium.PAGER)
    assert choi.status is PersonStatus.UNKNOWN  # empty status is unknown, never active
    assert choi.channels == (ChannelMedium.INTERNAL_MESSENGER, ChannelMedium.TEAMS, ChannelMedium.OTHER)


def test_unrecognised_columns_such_as_phone_numbers_are_not_stored() -> None:
    dumped = str([r.model_dump() for r in load_roster(ROSTER_CSV)])
    assert "1234-5678" not in dumped


def test_roster_english_headers_tsv_and_bytes() -> None:
    tsv = (
        "Name\tStatus\tTitle\tTeam\tRoles\tAliases\tChannels\tLeft_On\n"
        "Alex Kim\tactive\tSRE\tInfra\toncall\tAK\tslack\t\n"
    )
    (entry,) = load_roster(tsv.encode("utf-8"))
    assert entry.name == "Alex Kim"
    assert entry.status is PersonStatus.ACTIVE
    assert entry.roles == ("oncall",)
    assert entry.channels == (ChannelMedium.SLACK,)


def test_roster_semicolon_delimited_cp949_bytes() -> None:
    data = "이름;상태;연락수단\n홍길동;근무;휴대폰\n".encode("cp949")
    (entry,) = load_roster(data)
    assert entry.name == "홍길동"
    assert entry.channels == (ChannelMedium.PHONE,)


def test_roster_yaml_list_and_mapping() -> None:
    as_list = "- name: 김민수\n  status: active\n  roles: [팀장, 보안 담당]\n  channels: [slack, sms]\n"
    as_map = "roster:\n" + "".join("  " + line + "\n" for line in as_list.splitlines())
    for source in (as_list, as_map):
        (entry,) = load_roster(source)
        assert entry.roles == ("팀장", "보안 담당")
        assert entry.channels == (ChannelMedium.SLACK, ChannelMedium.SMS)


def test_roster_yaml_date_object_for_left_on() -> None:
    (entry,) = load_roster("- 이름: 박지훈\n  상태: 퇴사\n  퇴사일: 2026-09-30\n")
    assert entry.left_on == "2026-09-30"


def test_roster_from_path_and_path_string(tmp_path: Path) -> None:
    path = tmp_path / "roster.csv"
    path.write_text("name,status\n김민수,재직\n", encoding="utf-8")
    assert load_roster(path) == load_roster(str(path)) == load_roster("name,status\n김민수,재직\n")
    with pytest.raises(FileNotFoundError, match="명부 파일을 찾을 수 없습니다"):
        load_roster(tmp_path / "missing.csv")
    with pytest.raises(FileNotFoundError):
        load_roster(str(tmp_path / "missing.yaml"))


def test_roster_names_are_normalized() -> None:
    (entry,) = load_roster(f"이름,상태\n 김{ZWSP}민수 ,재직\n")
    assert entry.name == "김민수"


def test_roster_without_status_column_is_unknown() -> None:
    (entry,) = load_roster("이름,팀\n김민수,결제팀\n")
    assert entry.status is PersonStatus.UNKNOWN


def test_roster_blank_rows_skipped_and_empty_roster() -> None:
    assert len(load_roster("이름,상태\n\n김민수,재직\n,\n")) == 1
    assert load_roster("") == ()
    assert load_roster("이름,상태\n") == ()


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("팀,상태\n결제팀,재직\n", "이름 열"),
        ("이름,팀\n,결제팀\n", "이름이 비어"),
        ("이름,퇴사일\n김민수,어제\n", "날짜 형식"),
        ("이름,퇴사일\n김민수,2026-02-30\n", "날짜 형식"),
        ("roster: 3\n", "항목 목록"),
        ("staff:\n  - name: 김민수\n", "목록 키"),
    ],
)
def test_roster_errors(source: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_roster(source)


@pytest.mark.parametrize(
    ("value", "status"),
    [
        ("재직", PersonStatus.ACTIVE),
        ("근무 중", PersonStatus.ACTIVE),
        ("Active", PersonStatus.ACTIVE),
        ("퇴사", PersonStatus.DEPARTED),
        ("departed", PersonStatus.DEPARTED),
        ("휴직", PersonStatus.ON_LEAVE),
        ("휴가", PersonStatus.ON_LEAVE),
        ("장기부재", PersonStatus.ON_LEAVE),
        ("on_leave", PersonStatus.ON_LEAVE),
        ("퇴사 예정", PersonStatus.UNKNOWN),
        ("", PersonStatus.UNKNOWN),
        (None, PersonStatus.UNKNOWN),
    ],
)
def test_parse_status(value: str | None, status: PersonStatus) -> None:
    assert parse_status(value) is status


@pytest.mark.parametrize(
    ("value", "medium"),
    [
        ("슬랙", ChannelMedium.SLACK),
        ("Slack", ChannelMedium.SLACK),
        ("카톡", ChannelMedium.KAKAOTALK),
        ("카카오톡", ChannelMedium.KAKAOTALK),
        ("전화", ChannelMedium.PHONE),
        ("휴대폰", ChannelMedium.PHONE),
        ("phone", ChannelMedium.PHONE),
        ("문자", ChannelMedium.SMS),
        ("SMS", ChannelMedium.SMS),
        ("메일", ChannelMedium.EMAIL),
        ("이메일", ChannelMedium.EMAIL),
        ("e-mail", ChannelMedium.EMAIL),
        ("PagerDuty", ChannelMedium.PAGER),
        ("Opsgenie", ChannelMedium.PAGER),
        ("호출기", ChannelMedium.PAGER),
        ("팀즈", ChannelMedium.TEAMS),
        ("MS Teams", ChannelMedium.TEAMS),
        ("사내 메신저", ChannelMedium.INTERNAL_MESSENGER),
        ("비둘기", ChannelMedium.OTHER),
    ],
)
def test_parse_channel(value: str, medium: ChannelMedium) -> None:
    assert parse_channel(value) is medium


@settings(max_examples=100, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.text(alphabet="김이박", min_size=2, max_size=2),  # small alphabet → namesakes
            st.sampled_from(["재직", "퇴사", "휴직", ""]),
        ),
        min_size=1,
        max_size=8,
    ),
    st.integers(min_value=0, max_value=2**32),
)
def test_property_roster_is_row_order_independent(rows: list[tuple[str, str]], seed: int) -> None:
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)

    def to_csv(items: list[tuple[str, str]]) -> str:
        return "이름,상태\n" + "".join(f"{name},{status}\n" for name, status in items)

    assert load_roster(to_csv(rows)) == load_roster(to_csv(shuffled))


# --------------------------------------------------------------------------- on-call

ONCALL_CSV = """이름,시작,종료,레벨
김민수,2026-10-13 09:00,2026-10-13 21:00,1
박지훈,2026-10-13T00:00:00Z,2026-10-13T12:00:00Z,2차
최하늘,2026.10.12 21:00,2026/10/13 09:00,
"""


def test_oncall_csv_formats_and_timezones() -> None:
    shifts = load_oncall(ONCALL_CSV)
    # 김민수 and 박지훈 both start at 09:00 KST; the primary (level 1) sorts first
    assert [s.person for s in shifts] == ["최하늘", "김민수", "박지훈"]
    choi, kim, park = shifts
    assert kim.start == datetime(2026, 10, 13, 9, 0, tzinfo=KST)
    assert kim.level == 1
    assert park.start == datetime(2026, 10, 13, 9, 0, tzinfo=KST)  # 00:00Z → 09:00 KST
    assert park.start.utcoffset() == timedelta(hours=9)
    assert park.level == 2
    assert choi.start == datetime(2026, 10, 12, 21, 0, tzinfo=KST)
    assert choi.level == 1  # empty level → primary
    assert all(s.start.tzinfo is not None and s.end.tzinfo is not None for s in shifts)


def test_oncall_sort_key_is_start_level_person() -> None:
    window = "2026-10-13 09:00,2026-10-13 10:00"
    csv = f"person,start,end,level\nB,{window},2\nA,{window},2\nC,{window},1\n"
    assert [s.person for s in load_oncall(csv)] == ["C", "A", "B"]


def test_oncall_custom_default_timezone() -> None:
    (shift,) = load_oncall("person,start,end\nA,2026-10-13 09:00,2026-10-13 10:00\n", default_tz=UTC)
    assert shift.start == datetime(2026, 10, 13, 9, 0, tzinfo=UTC)
    assert shift.start.utcoffset() == timedelta(0)


def test_oncall_yaml_with_native_datetimes() -> None:
    source = "- person: 김민수\n  start: 2026-10-13 09:00:00\n  end: 2026-10-14\n  level: primary\n"
    (shift,) = load_oncall(source)
    assert shift.start == datetime(2026, 10, 13, 9, tzinfo=KST)
    assert shift.end == datetime(2026, 10, 14, tzinfo=KST)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("김민수,2026-10-13 10:00,2026-10-13 09:00,1", "종료 시각"),
        ("김민수,내일 아침,2026-10-13 09:00,1", "시작 시각 형식"),
        ("김민수,,2026-10-13 09:00,1", "비어 있습니다"),
        ("김민수,2026-10-13 09:00,2026-10-13 10:00,세번째", "레벨"),
        ("김민수,2026-10-13 09:00,2026-10-13 10:00,0", "레벨"),
        (",2026-10-13 09:00,2026-10-13 10:00,1", "이름이 비어"),
    ],
)
def test_oncall_errors(row: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_oncall("이름,시작,종료,레벨\n" + row + "\n")


# --------------------------------------------------------------------------- policy


def test_empty_policy_is_default() -> None:
    assert load_policy("") == Policy()
    assert load_policy("# 주석만\n") == Policy()


def test_policy_fields_nested_fatigue_and_wrapper() -> None:
    source = """policy:
  critical_tiers: [1]
  default_tier: 3
  terminal_roles: [CTO, 대표]
  regulated_services: [결제]
  fatigue:
    max_consecutive_night_shifts: 3
"""
    policy = load_policy(source)
    assert policy.critical_tiers == (1,)
    assert policy.default_tier == 3
    assert policy.terminal_roles == ("CTO", "대표")
    assert policy.regulated_services == ("결제",)
    assert policy.fatigue == FatiguePolicy(max_consecutive_night_shifts=3)
    assert policy.obligations == DEFAULT_OBLIGATIONS


def test_policy_tier_slo_merges_with_defaults() -> None:
    policy = load_policy("tier_slo_minutes:\n  1: 30\n")
    assert policy.tier_slo_minutes == {1: 30, 2: 240, 3: 1440}


def test_policy_obligations_merge_by_key() -> None:
    source = """obligations:
  - key: kr-breach-24h
    applies: false
  - key: customer-sla-4h
    incident_type: outage
    recipient: 주요 고객사
    deadline_minutes: 240
    legal_basis: 고객 계약 SLA 제7조
"""
    policy = load_policy(source)
    by_key = {o.key: o for o in policy.obligations}
    assert list(by_key) == ["kr-breach-24h", "kr-privacy-72h", "customer-sla-4h"]
    assert by_key["kr-breach-24h"].applies is False
    assert by_key["kr-breach-24h"].deadline_minutes == 24 * 60  # untouched fields kept
    assert by_key["kr-privacy-72h"] == DEFAULT_OBLIGATIONS[1]
    assert by_key["customer-sla-4h"].incident_type is IncidentType.OUTAGE


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("critical_tier: [1]\n", "알 수 없는 항목.*critical_tier"),
        ("fatigue:\n  max_nights: 2\n", "fatigue에 알 수 없는 항목"),
        ("obligations:\n  - recipient: KISA\n", "key가 필요"),
        ("obligations: 3\n", "항목 목록"),
        ("default_tier: 높음\n", "형식이 올바르지"),
        ("- 1\n- 2\n", "키-값 형식"),
        ("obligations:\n  - key: new\n    recipient: x\n", "형식이 올바르지"),
    ],
)
def test_policy_errors(source: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_policy(source)


def test_policy_from_bytes_and_path(tmp_path: Path) -> None:
    path = tmp_path / "policy.yaml"
    path.write_text("default_tier: 1\n", encoding="utf-8")
    assert load_policy(path).default_tier == 1
    assert load_policy(b"default_tier: 1\n").default_tier == 1


# --------------------------------------------------------------------------- catalog

NATIVE = """services:
  - name: 결제
    kind: service
    tier: 1
    aliases: [payment, 결제 서비스]
    depends_on: [postgres, kafka]
    location: 판교 IDC
    owner: 결제팀
    description: 카드 결제 승인
  - name: postgres
    kind: system
    tier: tier2
  - name: kafka
    kind: system
"""


def test_native_catalog() -> None:
    catalog = load_catalog(NATIVE)
    assert [s.name for s in catalog] == ["kafka", "postgres", "결제"]
    pay = catalog[2]
    assert pay == CatalogService(
        name="결제",
        kind="service",
        tier=1,
        aliases=("payment", "결제 서비스"),
        depends_on=("kafka", "postgres"),
        location="판교 IDC",
        owner="결제팀",
        description="카드 결제 승인",
    )
    assert catalog[1].tier == 2
    assert catalog[0].tier is None


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("services:\n  - name: a\n  - name: a\n", "두 번"),
        ("services:\n  - name: a\n    kind: database\n", "kind는"),
        ("services:\n  - name: a\n    tier: high\n", "tier 값"),
        ("services:\n  - name: a\n    tier: 0\n", "1 이상"),
        ("services:\n  - name: a\n    dependson: [b]\n    owners: x\n", "알 수 없는 항목이 있습니다: owners"),
        ("services:\n  - kind: service\n", "name이 없습니다"),
        ("services: [1, 2]\n", "항목 목록"),
        ("hello: world\n", "형식을 알 수 없습니다"),
        ("", "비어 있습니다"),
    ],
)
def test_catalog_errors(source: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_catalog(source)


BACKSTAGE = """apiVersion: backstage.io/v1alpha1
kind: Component
metadata:
  name: payment
  title: 결제 서비스
  description: 결제 API
  annotations:
    jettae.io/tier: "1"
    jettae.io/location: aws-ap-northeast-2
spec:
  type: service
  lifecycle: production
  owner: group:default/payments-team
  system: commerce
  dependsOn:
    - resource:default/payment-db
    - component:ledger
---
apiVersion: backstage.io/v1alpha1
kind: Resource
metadata:
  name: payment-db
spec:
  type: database
  owner: dba
---
apiVersion: backstage.io/v1alpha1
kind: System
metadata:
  name: commerce
spec:
  owner: user:jdoe
---
apiVersion: backstage.io/v1alpha1
kind: Group
metadata:
  name: payments-team
---
apiVersion: backstage.io/v1alpha1
kind: API
metadata:
  name: payment-api
"""


def test_backstage_catalog() -> None:
    catalog = {s.name: s for s in load_catalog(BACKSTAGE)}
    assert sorted(catalog) == ["commerce", "payment", "payment-db"]
    pay = catalog["payment"]
    assert pay.kind == "service"
    assert pay.tier == 1
    assert pay.aliases == ("결제 서비스",)
    assert pay.depends_on == ("ledger", "payment-db")
    assert pay.owner == "payments-team"
    assert pay.location == "aws-ap-northeast-2"
    assert pay.description == "결제 API"
    assert catalog["payment-db"].kind == "system"
    assert catalog["payment-db"].tier is None  # lifecycle is not a tier
    assert catalog["commerce"].owner == "jdoe"


COMPOSE_LIST = """services:
  web:
    image: nginx
    depends_on: [api]
    labels:
      jettae.io/tier: "2"
  api:
    build: .
    depends_on:
      - db
      - cache
    labels:
      - jettae.io/tier=1
      - jettae.io/owner=결제팀
  db:
    image: postgres:16
  cache:
"""

COMPOSE_DICT = """version: "3.9"
services:
  api:
    depends_on:
      db:
        condition: service_healthy
  db:
    image: postgres
"""


def test_compose_catalog() -> None:
    catalog = {s.name: s for s in load_catalog(COMPOSE_LIST)}
    assert sorted(catalog) == ["api", "cache", "db", "web"]
    assert catalog["web"].depends_on == ("api",)
    assert catalog["web"].tier == 2
    assert catalog["api"].depends_on == ("cache", "db")
    assert catalog["api"].tier == 1
    assert catalog["api"].owner == "결제팀"
    assert catalog["cache"] == CatalogService(name="cache")
    compose = {s.name: s for s in load_catalog(COMPOSE_DICT)}
    assert compose["api"].depends_on == ("db",)


def test_jaeger_dependencies_json() -> None:
    links = (
        '[{"parent": "frontend", "child": "payment", "callCount": 120},'
        ' {"parent": "payment", "child": "payment-db", "callCount": 80},'
        ' {"parent": "payment", "child": "payment", "callCount": 3},'
        ' {"parent": "frontend", "child": "legacy", "callCount": 0}]'
    )
    catalog = {s.name: s for s in load_catalog(links)}
    assert sorted(catalog) == ["frontend", "legacy", "payment", "payment-db"]
    assert catalog["frontend"].depends_on == ("payment",)  # zero-call link adds no dependency
    assert catalog["payment"].depends_on == ("payment-db",)  # self-call dropped
    wrapped = load_catalog('{"data": ' + links + "}")
    assert wrapped == tuple(sorted(catalog.values(), key=lambda s: s.name))


def test_catalog_from_path(tmp_path: Path) -> None:
    path = tmp_path / "dependencies.json"
    path.write_text('[{"parent": "a", "child": "b", "callCount": 1}]', encoding="utf-8")
    assert [s.name for s in load_catalog(path)] == ["a", "b"]


def test_merge_catalogs_precedence_and_union() -> None:
    first = (CatalogService(name="pay", tier=1, aliases=("결제",), depends_on=("db",), owner="결제팀"),)
    second = (
        CatalogService(
            name="pay", kind="system", tier=3, aliases=("payment",), depends_on=("ledger",), location="IDC"
        ),
        CatalogService(name="db", kind="system"),
    )
    merged = {s.name: s for s in merge_catalogs(first, second)}
    assert list(merged) == ["db", "pay"]
    pay = merged["pay"]
    assert pay.kind == "service"
    assert pay.tier == 1
    assert pay.owner == "결제팀"
    assert pay.location == "IDC"  # filled from the later source
    assert pay.aliases == ("payment", "결제")
    assert pay.depends_on == ("db", "ledger")
    assert merge_catalogs() == ()


@settings(max_examples=100, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.sampled_from("abcdefg"), st.sampled_from("abcdefg"), st.integers(min_value=0, max_value=3)
        ),
        min_size=1,
        max_size=15,
    )
)
def test_property_jaeger_catalog_sorted_and_without_self_loops(links: list[tuple[str, str, int]]) -> None:
    payload = (
        "[" + ",".join(f'{{"parent": "{p}", "child": "{c}", "callCount": {n}}}' for p, c, n in links) + "]"
    )
    catalog = load_catalog(payload)
    names = [s.name for s in catalog]
    assert names == sorted(set(names))
    assert set(names) == {p for p, _, _ in links} | {c for _, c, _ in links}
    for svc in catalog:
        assert svc.name not in svc.depends_on
        assert list(svc.depends_on) == sorted(svc.depends_on)
        expected = {c for p, c, n in links if p == svc.name and c != svc.name and n > 0}
        assert set(svc.depends_on) == expected


# --------------------------------------------------------------------------- review: strictness


@pytest.mark.parametrize(
    ("loader", "source", "what"),
    [
        (load_roster, "- name: [unclosed\n", "명부"),
        (load_oncall, "- person: a\n  start: [\n", "당직표"),
        (load_policy, "critical_tiers: [1\n", "정책"),
        (load_catalog, "services: [\n", "카탈로그"),
    ],
)
def test_yaml_syntax_errors_are_korean_value_errors(loader: object, source: str, what: str) -> None:
    assert callable(loader)
    with pytest.raises(
        ValueError, match=rf"{what} 파일의 YAML/JSON 형식이 올바르지 않습니다 \(\d+행 \d+열 근처\)"
    ):
        loader(source)


@pytest.mark.parametrize(
    ("loader", "source"),
    [
        (load_roster, "- name: a\n---\n- name: b\n"),
        (load_oncall, "- person: a\n  start: 2026-10-13 09:00\n  end: 2026-10-13 10:00\n---\n- person: b\n"),
        (load_policy, "critical_tiers: [1]\n---\ndefault_tier: 3\n"),
    ],
)
def test_second_yaml_document_is_rejected_not_ignored(loader: object, source: str) -> None:
    assert callable(loader)
    with pytest.raises(ValueError, match="문서가 하나만"):
        loader(source)


def test_csv_row_with_unquoted_delimiter_is_rejected() -> None:
    """Unquoted "슬랙,전화" would silently drop 전화 and fake an unreachable person."""
    with pytest.raises(ValueError, match=r"명부 2행: 값이 머리글 열 수\(2개\)보다 많습니다"):
        load_roster("이름,연락수단\n김민수,슬랙,전화\n")


def test_csv_trailing_empty_cells_are_fine() -> None:
    (entry,) = load_roster("이름,상태\n김민수,재직,,\n")
    assert entry.status is PersonStatus.ACTIVE


def test_csv_duplicate_header_first_column_wins_like_yaml() -> None:
    (entry,) = load_roster("이름,직급,title\n김민수,대리,Engineer\n")
    assert entry.title == "대리"


def test_roster_left_on_exported_with_time() -> None:
    (entry,) = load_roster("이름,상태,퇴사일\n김민수,퇴사,2026-09-30 00:00:00\n")
    assert entry.left_on == "2026-09-30"


def test_policy_obligation_typo_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"obligations 'kr-breach-24h'에 알 수 없는 항목.*deadline_minute"):
        load_policy("obligations:\n  - key: kr-breach-24h\n    deadline_minute: 60\n")


def test_policy_strings_are_normalized() -> None:
    decomposed = unicodedata.normalize("NFD", "대표")
    policy = load_policy(f"terminal_roles: [' CTO ', '{decomposed}']\nregulated_services: ['결{ZWSP}제']\n")
    assert policy.terminal_roles == ("CTO", "대표")
    assert policy.regulated_services == ("결제",)


def test_native_catalog_accepts_channel_kind_case_insensitively() -> None:
    source = (
        "services:\n  - name: 사내 메신저\n    kind: channel\n    location: 상암 IDC\n"
        "  - name: db\n    kind: System\n"
    )
    catalog = {s.name: s for s in load_catalog(source)}
    assert catalog["사내 메신저"].kind == "channel"
    assert catalog["사내 메신저"].location == "상암 IDC"
    assert catalog["db"].kind == "system"


def test_merge_keeps_the_first_kind_including_channel() -> None:
    merged = merge_catalogs(
        (CatalogService(name="m", kind="channel"),), (CatalogService(name="m", kind="service", tier=1),)
    )
    assert merged == (CatalogService(name="m", kind="channel", tier=1),)


@pytest.mark.parametrize(
    ("tier", "expected"),
    # "2.0" is a YAML float with an integral value
    [("1", 1), ("tier2", 2), ("Tier 3", 3), ("T1", 1), ("P1", 1), ("1등급", 1), ("등급 2", 2), ("2.0", 2)],
)
def test_tier_spellings(tier: str, expected: int) -> None:
    assert load_catalog(f"services:\n  - name: a\n    tier: {tier}\n")[0].tier == expected


@pytest.mark.parametrize("tier", ["'-1'", "1.5", "'1-2'", "high", "'tier'"])
def test_ambiguous_tiers_are_rejected_not_guessed(tier: str) -> None:
    with pytest.raises(ValueError, match="tier"):
        load_catalog(f"services:\n  - name: a\n    tier: {tier}\n")


def test_backstage_single_depends_on_string_is_one_reference() -> None:
    source = (
        "apiVersion: backstage.io/v1alpha1\nkind: Component\nmetadata:\n  name: pay\n"
        "spec:\n  dependsOn: component:ledger\n---\n"
        "apiVersion: backstage.io/v1alpha1\nkind: Component\nmetadata: broken\n"
    )
    assert load_catalog(source) == (CatalogService(name="pay", depends_on=("ledger",)),)


def test_jaeger_tab_indented_json_and_count_spellings() -> None:
    links = (
        '[\n\t{"parent": "a", "child": "b", "callCount": "12"},\n'
        '\t{"parent": "a", "child": "c", "callCount": 0.0},\n'
        '\t{"parent": "b", "child": "c"}\n]'
    )
    catalog = {s.name: s.depends_on for s in load_catalog(links)}
    assert catalog == {"a": ("b",), "b": ("c",), "c": ()}


@pytest.mark.parametrize(
    ("link", "message"),
    [
        ('{"parent": "a", "child": null}', "이름이 비어"),
        ('{"parent": " ", "child": "b"}', "이름이 비어"),
        ('{"parent": "a", "child": "b", "callCount": "many"}', "callCount"),
    ],
)
def test_jaeger_malformed_links_are_rejected(link: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_catalog(f"[{link}]")


@settings(max_examples=100, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.sampled_from(["김민수", "박지훈", "이서연"]), st.sampled_from(["슬랙", "전화", "카톡", ""])
        ),
        min_size=1,
        max_size=6,
    )
)
def test_property_roster_csv_and_yaml_agree(rows: list[tuple[str, str]]) -> None:
    csv_text = "이름,연락수단\n" + "".join(f"{name},{channel}\n" for name, channel in rows)
    yaml_text = "".join(f"- 이름: {name}\n  연락수단: '{channel}'\n" for name, channel in rows)
    assert load_roster(csv_text) == load_roster(yaml_text)
