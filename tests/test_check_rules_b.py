"""Rule families B: deadlines, Korean reporting duties, on-call, catalog, integrity, SPOF rules."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jettae.check.context import CheckContext
from jettae.check.impact import blast_radius
from jettae.check.registry import Rule, run_checks, sort_findings
from jettae.check.rules.catalog import (
    CatalogServiceWithoutOwnerOrRunbook,
    CriticalServicesConcentrated,
    RunbookServiceNotInCatalog,
)
from jettae.check.rules.deadlines import (
    ConflictingDeadlines,
    EscalationExceedsRecoveryTarget,
    ReportingDeadlineLaterThanLaw,
    durations_minutes,
    escalation_timing,
    parse_duration_minutes,
)
from jettae.check.rules.integrity import HeldElement, UngroundedElement
from jettae.check.rules.oncall import (
    ConsecutiveNightShifts,
    InsufficientRest,
    OnCallPersonUnavailable,
    Shift,
    TooManyShiftsPerWeek,
    night_dates,
    shifts_by_person,
    table_zone,
)
from jettae.check.rules.regulatory import (
    BreachReporterMissing,
    BreachReportStepMissing,
    PrivacyReporterMissing,
    PrivacyReportStepMissing,
    format_minutes,
    law_text,
    recipient_matches,
    reporting_services,
)
from jettae.check.rules.spof import SingleChannelPoint, SinglePersonPoint
from jettae.domain import (
    DEFAULT_OBLIGATIONS,
    ActionType,
    Attr,
    CatalogService,
    Edge,
    EdgeKind,
    Element,
    Extractor,
    FatiguePolicy,
    Finding,
    FindingStatus,
    FixKind,
    GraphBuilder,
    IncidentType,
    Node,
    NodeKind,
    ObligationRule,
    OnCallShift,
    OpsGraph,
    PersonStatus,
    Policy,
    Provenance,
    RosterEntry,
    Severity,
    SourceKind,
    SourceSpan,
    Verification,
    edge_id,
    node_id,
)

# ---------------------------------------------------------------------------- fixtures


def span(quote: str) -> SourceSpan:
    return SourceSpan(doc_id="doc-test", start=0, end=len(quote), quote=quote, line=1)


RUNBOOK = Provenance(source=SourceKind.RUNBOOK, extractor=Extractor.RULE, spans=(span("런북 근거"),))
ROSTER = Provenance(source=SourceKind.ROSTER, extractor=Extractor.IMPORT, ref="roster.csv")
CATALOG = Provenance(source=SourceKind.CATALOG, extractor=Extractor.IMPORT, ref="catalog.yaml")
BREACH_LAW, PRIVACY_LAW = DEFAULT_OBLIGATIONS


def _attrs(prov: Provenance, attrs: dict[str, Any]) -> dict[str, Attr]:
    return {
        k: v if isinstance(v, Attr) else Attr(value=v, spans=prov.spans, source=prov.source)
        for k, v in attrs.items()
    }


class G:
    """Fluent builder for hand-made operations graphs."""

    def __init__(self) -> None:
        self.b = GraphBuilder()

    def node(
        self,
        kind: NodeKind,
        name: str,
        *,
        prov: tuple[Provenance, ...] = (RUNBOOK,),
        hold: str | None = None,
        **attrs: Any,
    ) -> str:
        nid = node_id(kind, name)
        self.b.add_node(
            Node(
                id=nid,
                kind=kind,
                name=name,
                attrs=_attrs(prov[0], attrs),
                prov=prov,
                verification=Verification.HOLD if hold is not None else Verification.VERIFIED,
                hold_reason=hold or None,
            )
        )
        return nid

    def edge(
        self,
        kind: EdgeKind,
        src: str,
        dst: str,
        *,
        prov: tuple[Provenance, ...] = (RUNBOOK,),
        hold: str | None = None,
        **attrs: Any,
    ) -> str:
        eid = edge_id(kind, src, dst)
        self.b.add_edge(
            Edge(
                id=eid,
                kind=kind,
                src=src,
                dst=dst,
                attrs=_attrs(prov[0], attrs),
                prov=prov,
                verification=Verification.HOLD if hold is not None else Verification.VERIFIED,
                hold_reason=hold or None,
            )
        )
        return eid

    def service(self, name: str, tier: int = 1, **attrs: Any) -> str:
        return self.node(NodeKind.SERVICE, name, tier=tier, **attrs)

    def person(self, name: str, *channels: str, status: PersonStatus = PersonStatus.ACTIVE) -> str:
        pid = self.node(NodeKind.PERSON, name, prov=(ROSTER,), status=status.value)
        for ch in channels:
            self.edge(EdgeKind.REACHABLE_VIA, pid, ch)
        return pid

    def obligation(self, name: str, *services: str, reporter: str | None = None, **attrs: Any) -> str:
        oid = self.node(NodeKind.OBLIGATION, name, **attrs)
        for svc in services:
            self.edge(EdgeKind.MUST_REPORT, svc, oid)
        if reporter is not None:
            self.edge(EdgeKind.REPORTED_BY, oid, reporter)
        return oid

    def build(self) -> OpsGraph:
        return self.b.build()


def run(rule: Rule, ctx: CheckContext) -> list[Finding]:
    return sort_findings(rule.check(ctx))


def ctx_of(g: G, **kw: Any) -> CheckContext:
    return CheckContext(graph=g.build(), **kw)


def breach(**overrides: Any) -> dict[str, Any]:
    attrs: dict[str, Any] = {"incident_type": "breach", "recipient": "KISA", "deadline_minutes": 1440}
    attrs.update(overrides)
    return attrs


def privacy(**overrides: Any) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "incident_type": "privacy_leak",
        "recipient": "개인정보위",
        "deadline_minutes": 4320,
    }
    attrs.update(overrides)
    return attrs


# ---------------------------------------------------------------------------- JT-DDL-001


def _escalation_graph(second_after: int, *, approver: str = "cto") -> tuple[G, dict[str, str]]:
    g = G()
    slack = g.node(NodeKind.CHANNEL, "Slack")
    ids = {
        "svc": g.service("결제"),
        "kim": g.person("김대리", slack),
        "park": g.person("박팀장", slack),
        "cto": g.person("최CTO", slack),
        "rollback": g.node(NodeKind.ACTION, "롤백", action_type=ActionType.ROLLBACK.value),
    }
    g.edge(EdgeKind.OWNED_BY, ids["svc"], ids["kim"])
    g.edge(EdgeKind.ESCALATES_TO, ids["kim"], ids["park"], after_minutes=30)
    g.edge(EdgeKind.ESCALATES_TO, ids["park"], ids["cto"], after_minutes=second_after)
    g.edge(EdgeKind.MITIGATED_BY, ids["svc"], ids["rollback"])
    g.edge(EdgeKind.APPROVED_BY, ids["rollback"], ids[approver])
    return g, ids


def test_ddl001_escalation_to_approver_exceeds_slo() -> None:
    g, ids = _escalation_graph(60)
    [f] = run(EscalationExceedsRecoveryTarget(), ctx_of(g))
    assert f.rule_id == "JT-DDL-001"
    assert f.severity is Severity.MEDIUM
    assert f.service_id == ids["svc"]
    assert "90분" in f.message
    assert "1시간" in f.message
    assert "최CTO" in f.message
    assert f.fix is not None
    assert f.fix.kind is FixKind.FIX_DEADLINE
    assert f.fix.params["total_minutes"] == 90
    assert f.fix.params["required_minutes"] == 60
    assert edge_id(EdgeKind.ESCALATES_TO, ids["park"], ids["cto"]) in f.subject_ids
    assert f.evidence  # spans of the escalation edges


def test_ddl001_within_slo_or_nearer_approver_is_fine() -> None:
    g, _ = _escalation_graph(20)
    assert run(EscalationExceedsRecoveryTarget(), ctx_of(g)) == []
    g, _ = _escalation_graph(600, approver="park")
    assert run(EscalationExceedsRecoveryTarget(), ctx_of(g)) == []


def test_ddl001_explicit_slo_and_non_critical_tier() -> None:
    g, ids = _escalation_graph(60)
    g.service("결제", slo_minutes=120)  # merge: first writer keeps tier, adds slo
    assert run(EscalationExceedsRecoveryTarget(), ctx_of(g)) == []
    g, _ = _escalation_graph(60)
    g.b.nodes[ids["svc"]] = g.b.nodes[ids["svc"]].model_copy(update={"attrs": _attrs(RUNBOOK, {"tier": 3})})
    assert run(EscalationExceedsRecoveryTarget(), ctx_of(g)) == []


def test_ddl001_chain_end_without_approver_and_unstated_minutes() -> None:
    g = G()
    slack = g.node(NodeKind.CHANNEL, "Slack")
    svc = g.service("결제")
    gone = g.person("이퇴사", slack, status=PersonStatus.DEPARTED)
    mid = g.person("박휴직", slack, status=PersonStatus.ON_LEAVE)
    last = g.person("정부재", slack, status=PersonStatus.DEPARTED)
    g.edge(EdgeKind.OWNED_BY, svc, gone)
    g.edge(EdgeKind.ESCALATES_TO, gone, mid, after_minutes=50)
    g.edge(EdgeKind.ESCALATES_TO, mid, last, after_minutes=40)
    ghost = g.person("유령", status=PersonStatus.UNKNOWN)
    g.edge(EdgeKind.ESCALATES_TO, last, ghost)  # no after_minutes stated -> counted as 0
    timing = escalation_timing(ctx_of(g), svc)
    assert timing is not None
    assert not timing.reached_goal
    assert timing.minutes == 90
    assert timing.unstated_edges == 1
    assert timing.nodes == (gone, mid, last, ghost)
    [f] = run(EscalationExceedsRecoveryTarget(), ctx_of(g))
    assert "끝까지" in f.message
    assert "0분으로 계산" in f.message


def test_ddl001_no_owner_is_left_to_path_rules() -> None:
    g = G()
    svc = g.service("결제")
    assert escalation_timing(ctx_of(g), svc) is None
    assert run(EscalationExceedsRecoveryTarget(), ctx_of(g)) == []


# ---------------------------------------------------------------------------- JT-DDL-002


def test_ddl002_kisa_48h_contradicts_24h_law() -> None:
    g = G()
    svc = g.service("회원")
    deadline = Attr(value=2880, spans=(span("KISA에 48시간 이내 신고"),))
    oid = g.obligation("KISA 침해사고 신고", svc, **breach(deadline_minutes=deadline))
    [f] = run(ReportingDeadlineLaterThanLaw(), ctx_of(g))
    assert f.severity is Severity.CRITICAL
    assert f.legal_basis == BREACH_LAW.legal_basis
    assert "48시간" in f.message
    assert "24시간" in f.message
    assert f.service_id == svc
    assert f.fix is not None
    assert f.fix.params == {"obligation_id": oid, "required_minutes": 1440}
    assert span("KISA에 48시간 이내 신고") in f.evidence


def test_ddl002_uses_the_policy_rules_legal_basis() -> None:
    g = G()
    g.obligation("보고", g.service("회원"), **breach(deadline_minutes=2880))
    custom = ObligationRule(
        key="in-house",
        incident_type=IncidentType.BREACH,
        recipient="KISA",
        deadline_minutes=1440,
        legal_basis="사내 보안 규정 제7조",
    )
    [f] = run(ReportingDeadlineLaterThanLaw(), ctx_of(g, policy=Policy(obligations=(custom,))))
    assert f.legal_basis == "사내 보안 규정 제7조"


@pytest.mark.parametrize(
    "attrs",
    [
        breach(deadline_minutes=1440),
        breach(deadline_minutes=720),
        breach(deadline_minutes=2880, recipient="고객"),
        {"recipient": "KISA", "deadline_minutes": 2880},  # incident type unstated -> not judged
        privacy(deadline_minutes=4320),
    ],
)
def test_ddl002_negative(attrs: dict[str, Any]) -> None:
    g = G()
    g.obligation("신고", g.service("회원"), **attrs)
    assert run(ReportingDeadlineLaterThanLaw(), ctx_of(g)) == []


def test_ddl002_rule_not_applicable() -> None:
    g = G()
    g.obligation("신고", g.service("회원"), **breach(deadline_minutes=2880))
    policy = Policy(obligations=(BREACH_LAW.model_copy(update={"applies": False}), PRIVACY_LAW))
    assert run(ReportingDeadlineLaterThanLaw(), ctx_of(g, policy=policy)) == []


# ---------------------------------------------------------------------------- JT-DDL-003


def test_ddl003_two_documents_disagree_on_the_same_duty() -> None:
    g = G()
    svc = g.service("회원")
    a = g.obligation("KISA 신고(운영 런북)", svc, **breach())
    b = g.obligation(
        "KISA 신고(보안 런북)", svc, **breach(recipient="한국인터넷진흥원", deadline_minutes=2880)
    )
    [f] = run(ConflictingDeadlines(), ctx_of(g))
    assert f.severity is Severity.MEDIUM
    assert set(f.subject_ids) >= {a, b}
    assert "24시간" in f.message
    assert "48시간" in f.message
    assert f.fix is not None
    assert f.fix.params["required_minutes"] == 1440  # the legal one, not an invented number


def test_ddl003_conflicting_quotes_inside_one_attribute() -> None:
    g = G()
    deadline = Attr(value=1440, spans=(span("24시간 이내 신고"), span("48시간 내 보고")))
    oid = g.obligation("KISA 신고", g.service("회원"), **breach(deadline_minutes=deadline))
    [f] = run(ConflictingDeadlines(), ctx_of(g))
    assert f.subject_ids[0] == oid
    assert "원문 근거들이" in f.message


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (breach(), breach(recipient="한국인터넷진흥원")),  # same deadline
        (breach(), breach(recipient="고객", deadline_minutes=2880)),  # different recipient
        (breach(), privacy(recipient="KISA", deadline_minutes=2880)),  # different incident type
    ],
)
def test_ddl003_negative(first: dict[str, Any], second: dict[str, Any]) -> None:
    g = G()
    svc = g.service("회원")
    g.obligation("의무1", svc, **first)
    g.obligation("의무2", svc, **second)
    assert run(ConflictingDeadlines(), ctx_of(g)) == []


@pytest.mark.parametrize(
    ("text", "minutes"),
    [
        ("KISA에 24시간 이내 신고", 1440),
        ("30 분 후 상향", 30),
        ("3일 이내 통지", 4320),
        ("within 24h", 1440),
        ("48 hours", 2880),
        ("90 min", 90),
        ("1.5시간", 90),
        ("2026년 10월 1일까지", None),
        ("10월 15일", None),
        ("1분기 보고", None),
        ("지체 없이", None),
    ],
)
def test_parse_duration_minutes(text: str, minutes: float | None) -> None:
    assert parse_duration_minutes(text) == minutes


# ---------------------------------------------------------------------------- JT-KR-*


def _two_critical(g: G) -> tuple[str, str]:
    return g.service("결제"), g.service("회원")


def test_kr001_and_kr003_once_per_organisation_when_nobody_has_the_step() -> None:
    g = G()
    pay, member = _two_critical(g)
    g.service("사내위키", tier=3)  # not critical -> out of scope
    ctx = ctx_of(g)
    [kr1] = run(BreachReportStepMissing(), ctx)
    assert kr1.service_id is None
    assert kr1.subject_ids == (pay, member)
    assert kr1.legal_basis == BREACH_LAW.legal_basis
    assert kr1.fix is not None
    assert kr1.fix.kind is FixKind.ADD_REPORTING_STEP
    assert kr1.fix.params == {
        "incident_type": "breach",
        "recipient": BREACH_LAW.recipient,
        "deadline_minutes": 1440,
        "service_ids": [pay, member],
    }
    [kr3] = run(PrivacyReportStepMissing(), ctx)
    assert kr3.legal_basis == PRIVACY_LAW.legal_basis
    assert "72시간" in kr3.message


def test_kr001_per_missing_service_when_some_have_the_step() -> None:
    g = G()
    pay, member = _two_critical(g)
    g.obligation("침해 신고", pay, **breach())
    [f] = run(BreachReportStepMissing(), ctx_of(g))
    assert f.service_id == member
    assert f.subject_ids == (member,)


def test_kr001_satisfied_and_scope_by_regulated_services() -> None:
    g = G()
    pay, member = _two_critical(g)
    g.obligation("침해 신고", pay, **breach(recipient="과기정통부·KISA"))
    g.obligation("침해 신고2", member, **breach())
    assert run(BreachReportStepMissing(), ctx_of(g)) == []

    g = G()
    pay, _ = _two_critical(g)
    [f] = run(BreachReportStepMissing(), ctx_of(g, policy=Policy(regulated_services=("결제",))))
    assert f.subject_ids == (pay,)


def test_kr001_wrong_recipient_or_not_applicable() -> None:
    g = G()
    pay = g.service("결제")
    g.obligation("고객 공지", pay, **breach(recipient="고객"))
    assert len(run(BreachReportStepMissing(), ctx_of(g))) == 1
    policy = Policy(obligations=(BREACH_LAW.model_copy(update={"applies": False}),))
    assert run(BreachReportStepMissing(), ctx_of(g, policy=policy)) == []


def _reporter_case(status: PersonStatus | None) -> CheckContext:
    g = G()
    pay = g.service("결제")
    reporter = None if status is None else g.person("보안담당", status=status)
    g.obligation("KISA 신고", pay, reporter=reporter, **breach())
    g.obligation("개인정보 신고", pay, reporter=reporter, **privacy())
    return ctx_of(g)


def test_kr002_departed_reporter_is_broken() -> None:
    [f] = run(BreachReporterMissing(), _reporter_case(PersonStatus.DEPARTED))
    assert f.status is FindingStatus.BROKEN
    assert f.severity is Severity.HIGH
    assert "퇴사" in f.message
    assert f.fix is not None
    assert f.fix.kind is FixKind.REPLACE_DEPARTED
    assert f.legal_basis == BREACH_LAW.legal_basis


def test_kr002_unknown_reporter_is_unknown_not_pass() -> None:
    [f] = run(BreachReporterMissing(), _reporter_case(PersonStatus.UNKNOWN))
    assert f.status is FindingStatus.UNKNOWN
    assert f.fix is not None
    assert f.fix.kind is FixKind.CONFIRM_IDENTITY


def test_kr002_missing_reporter_and_active_reporter() -> None:
    [f] = run(BreachReporterMissing(), _reporter_case(None))
    assert f.fix is not None
    assert f.fix.kind is FixKind.ASSIGN_OWNER
    assert run(BreachReporterMissing(), _reporter_case(PersonStatus.ACTIVE)) == []


def test_kr002_temporarily_unreachable_reporter_suggests_a_deputy() -> None:
    ctx = _reporter_case(PersonStatus.ACTIVE)
    [f] = run(BreachReporterMissing(), ctx.with_unavailable([node_id(NodeKind.PERSON, "보안담당")]))
    assert f.fix is not None
    assert f.fix.kind is FixKind.ADD_DEPUTY


def test_kr004_privacy_reporter() -> None:
    [f] = run(PrivacyReporterMissing(), _reporter_case(PersonStatus.ON_LEAVE))
    assert f.rule_id == "JT-KR-004"
    assert "장기 부재" in f.message
    assert f.legal_basis == PRIVACY_LAW.legal_basis
    assert run(PrivacyReporterMissing(), _reporter_case(PersonStatus.ACTIVE)) == []


def test_kr_rules_cite_the_default_law_text() -> None:
    assert BreachReportStepMissing.info.legal_basis == BREACH_LAW.legal_basis
    assert BreachReporterMissing.info.legal_basis == BREACH_LAW.legal_basis
    assert PrivacyReportStepMissing.info.legal_basis == PRIVACY_LAW.legal_basis
    assert PrivacyReporterMissing.info.legal_basis == PRIVACY_LAW.legal_basis


@pytest.mark.parametrize(
    ("stated", "required", "expected"),
    [
        ("KISA", BREACH_LAW.recipient, True),
        ("한국인터넷진흥원", BREACH_LAW.recipient, True),
        ("개인정보위", PRIVACY_LAW.recipient, True),
        ("개인정보 보호위원회", PRIVACY_LAW.recipient, True),
        ("개인정보보호위원회", "과학기술정보통신부", False),
        ("고객", "KISA", False),
        (None, "KISA", True),
        ("금융감독원", "금융감독원", True),
    ],
)
def test_recipient_matches(stated: str | None, required: str, expected: bool) -> None:
    assert recipient_matches(stated, required) is expected


@pytest.mark.parametrize(
    ("minutes", "text"), [(1440, "24시간"), (2880, "48시간"), (90, "90분"), (30, "30분")]
)
def test_format_minutes(minutes: float, text: str) -> None:
    assert format_minutes(minutes) == text


# ---------------------------------------------------------------------------- JT-ONC-*


ROSTER_ENTRIES = (
    RosterEntry(name="김대리", aliases=("Kim",)),
    RosterEntry(name="이퇴사", status=PersonStatus.DEPARTED),
    RosterEntry(name="박휴직", status=PersonStatus.ON_LEAVE),
    RosterEntry(name="최미정", status=PersonStatus.UNKNOWN),
    RosterEntry(name="정예정", left_on="2026-10-05"),
)


def at(day: int, hour: int, month: int = 10) -> datetime:
    return datetime(2026, month, day, hour)


def shift(person: str, start: datetime, hours: float) -> OnCallShift:
    return OnCallShift(person=person, start=start, end=start + timedelta(hours=hours))


def oncall_ctx(
    *shifts: OnCallShift, roster: tuple[RosterEntry, ...] = ROSTER_ENTRIES, **kw: Any
) -> CheckContext:
    g = G()
    g.person("김대리")
    return CheckContext(graph=g.build(), roster=roster, oncall=shifts, **kw)


def test_onc001_three_consecutive_nights_across_midnight() -> None:
    shifts = [shift("김대리", at(d, 22), 8) for d in (1, 2, 3)]
    [f] = run(ConsecutiveNightShifts(), oncall_ctx(*shifts))
    assert f.severity is Severity.MEDIUM
    assert "2026-10-01부터 2026-10-03까지 3일 연속" in f.message
    assert f.subject_ids == (node_id(NodeKind.PERSON, "김대리"),)
    assert f.fix is not None
    assert f.fix.kind is FixKind.ADJUST_ONCALL
    assert len(f.fix.params["shift_starts"]) == 3


def test_onc001_negative_two_nights_or_day_shifts() -> None:
    assert run(ConsecutiveNightShifts(), oncall_ctx(*(shift("김대리", at(d, 22), 8) for d in (1, 2)))) == []
    assert (
        run(ConsecutiveNightShifts(), oncall_ctx(*(shift("김대리", at(d, 9), 9) for d in (1, 2, 3, 4)))) == []
    )
    # a gap night breaks the run
    nights = [shift("김대리", at(d, 22), 8) for d in (1, 2, 4, 5)]
    assert run(ConsecutiveNightShifts(), oncall_ctx(*nights)) == []


def test_onc001_alias_rows_merge_into_one_person() -> None:
    shifts = [shift("김대리", at(1, 22), 8), shift("Kim", at(2, 22), 8), shift("김대리", at(3, 22), 8)]
    assert len(run(ConsecutiveNightShifts(), oncall_ctx(*shifts))) == 1


def test_night_dates_wrapping_non_wrapping_and_timezone_aware() -> None:
    policy = FatiguePolicy()
    evening = Shift(at(1, 20), at(1, 23), "x")
    assert night_dates(evening, policy) == [at(1, 0).date()]
    early = Shift(at(2, 4), at(2, 7), "x")
    assert night_dates(early, policy) == [at(1, 0).date()]
    day = Shift(at(2, 9), at(2, 18), "x")
    assert night_dates(day, policy) == []
    midnight_only = FatiguePolicy(night_start_hour=0, night_end_hour=6)
    assert night_dates(Shift(at(1, 22), at(2, 2), "x"), midnight_only) == [at(2, 0).date()]
    assert night_dates(day, FatiguePolicy(night_start_hour=3, night_end_hour=3)) == []
    kst = timezone(timedelta(hours=9))
    aware = [
        OnCallShift(
            person="김대리", start=at(d, 22).replace(tzinfo=kst), end=at(d + 1, 6).replace(tzinfo=kst)
        )
        for d in (1, 2, 3)
    ]
    assert len(run(ConsecutiveNightShifts(), oncall_ctx(*aware))) == 1


def test_onc002_short_rest_back_to_back_and_overlap() -> None:
    short = oncall_ctx(shift("김대리", at(1, 18), 12), shift("김대리", at(2, 14), 8))
    [f] = run(InsufficientRest(), short)
    assert "8시간 만인" in f.message
    assert "최소 휴식 11시간" in f.message
    back_to_back = oncall_ctx(shift("김대리", at(1, 9), 9), shift("김대리", at(1, 18), 15))
    [g] = run(InsufficientRest(), back_to_back)
    assert "0시간 만인" in g.message
    overlap = oncall_ctx(shift("김대리", at(1, 9), 24), shift("김대리", at(1, 18), 6))
    assert len(run(InsufficientRest(), overlap)) == 1


def test_onc002_enough_rest_and_invalid_rows() -> None:
    ok = oncall_ctx(shift("김대리", at(1, 18), 12), shift("김대리", at(2, 18), 8))
    assert run(InsufficientRest(), ok) == []
    reversed_row = OnCallShift(person="김대리", start=at(2, 10), end=at(2, 8))
    assert run(InsufficientRest(), oncall_ctx(shift("김대리", at(1, 18), 12), reversed_row)) == []


def test_onc003_weekly_load_by_iso_week() -> None:
    four = [shift("김대리", at(d, 9), 9) for d in (5, 6, 7, 8)]  # Mon-Thu, ISO week 41
    [f] = run(TooManyShiftsPerWeek(), oncall_ctx(*four))
    assert f.severity is Severity.LOW
    assert "41주차" in f.message
    assert run(TooManyShiftsPerWeek(), oncall_ctx(*four[:3])) == []
    split = [shift("김대리", at(d, 9), 9) for d in (3, 4, 5, 6)]  # Sat-Sun (W40) + Mon-Tue (W41)
    assert run(TooManyShiftsPerWeek(), oncall_ctx(*split)) == []


@pytest.mark.parametrize(
    ("person", "status", "fragment"),
    [
        ("이퇴사", FindingStatus.BROKEN, "퇴사"),
        ("박휴직", FindingStatus.BROKEN, "장기 부재"),
        ("최미정", FindingStatus.UNKNOWN, "확인 불가"),
        ("유령", FindingStatus.UNKNOWN, "명부에 없는"),
    ],
)
def test_onc004_departed_absent_or_unknown(person: str, status: FindingStatus, fragment: str) -> None:
    [f] = run(OnCallPersonUnavailable(), oncall_ctx(shift(person, at(1, 22), 8)))
    assert f.severity is Severity.HIGH
    assert f.status is status
    assert fragment in f.message


def test_onc004_scheduled_departure() -> None:
    [f] = run(
        OnCallPersonUnavailable(), oncall_ctx(shift("정예정", at(4, 9), 9), shift("정예정", at(6, 9), 9))
    )
    assert "2026-10-05 퇴사 예정" in f.message
    assert f.fix is not None
    assert f.fix.params["shift_starts"] == [at(6, 9).isoformat()]
    assert run(OnCallPersonUnavailable(), oncall_ctx(shift("정예정", at(4, 9), 9))) == []


def test_onc004_active_alias_past_shift_and_missing_roster() -> None:
    assert run(OnCallPersonUnavailable(), oncall_ctx(shift("Kim", at(1, 22), 8))) == []
    past = oncall_ctx(shift("이퇴사", at(1, 22), 8), now=at(5, 0))
    assert run(OnCallPersonUnavailable(), past) == []
    [f] = run(OnCallPersonUnavailable(), oncall_ctx(shift("김대리", at(1, 22), 8), roster=()))
    assert f.status is FindingStatus.UNKNOWN
    assert "명부가 없어" in f.message
    assert run(OnCallPersonUnavailable(), oncall_ctx()) == []


# ---------------------------------------------------------------------------- JT-CAT-*


def test_cat001_catalog_service_without_owner_or_runbook() -> None:
    g = G()
    sid = g.node(NodeKind.SERVICE, "정산", prov=(CATALOG,), tier=1)
    [f] = run(CatalogServiceWithoutOwnerOrRunbook(), ctx_of(g))
    assert f.severity is Severity.HIGH
    assert f.service_id == sid
    assert f.fix is not None
    assert f.fix.kind is FixKind.ADD_SERVICE_RUNBOOK


def test_cat001_negative_owner_runbook_or_runbook_edge() -> None:
    g = G()
    sid = g.node(NodeKind.SERVICE, "정산", prov=(CATALOG,))
    g.edge(EdgeKind.OWNED_BY, sid, g.node(NodeKind.TEAM, "재무팀", prov=(CATALOG,)), prov=(CATALOG,))
    assert run(CatalogServiceWithoutOwnerOrRunbook(), ctx_of(g)) == []

    g = G()
    g.node(NodeKind.SERVICE, "정산", prov=(CATALOG, RUNBOOK))
    assert run(CatalogServiceWithoutOwnerOrRunbook(), ctx_of(g)) == []

    g = G()
    sid = g.node(NodeKind.SERVICE, "정산", prov=(CATALOG,))
    g.edge(EdgeKind.DEPENDS_ON, g.service("결제"), sid)  # runbook mentions it in a relation
    assert run(CatalogServiceWithoutOwnerOrRunbook(), ctx_of(g)) == []


def test_cat002_runbook_only_service_when_catalog_given() -> None:
    g = G()
    coupon = g.service("쿠폰")
    g.service("결제")
    g.node(NodeKind.SERVICE, "정산", prov=(CATALOG,))
    catalog = (CatalogService(name="정산"), CatalogService(name="payment", aliases=("결제",)))
    [f] = run(RunbookServiceNotInCatalog(), ctx_of(g, catalog=catalog))
    assert f.service_id == coupon
    assert f.severity is Severity.INFO
    assert f.status is FindingStatus.UNKNOWN
    assert run(RunbookServiceNotInCatalog(), ctx_of(g)) == []  # no catalog -> nothing to compare


def _concentrated(*, failover: bool = False, second_location: bool = False) -> G:
    g = G()
    idc_a = g.node(NodeKind.LOCATION, "IDC-A", prov=(CATALOG,))
    pay, order = g.service("결제"), g.service("주문")
    db = g.node(NodeKind.SYSTEM, "주문DB", prov=(CATALOG,))
    g.edge(EdgeKind.RUNS_IN, pay, idc_a, prov=(CATALOG,))
    g.edge(EdgeKind.DEPENDS_ON, order, db)
    target = g.node(NodeKind.LOCATION, "IDC-B", prov=(CATALOG,)) if second_location else idc_a
    g.edge(EdgeKind.RUNS_IN, db, target, prov=(CATALOG,))
    g.service("사내위키", tier=3)
    if failover:
        g.edge(EdgeKind.MITIGATED_BY, db, g.node(NodeKind.ACTION, "DR 전환", action_type="failover"))
    return g


def test_cat003_all_critical_services_in_one_location() -> None:
    [f] = run(CriticalServicesConcentrated(), ctx_of(_concentrated()))
    idc_a = node_id(NodeKind.LOCATION, "IDC-A")
    assert f.severity is Severity.MEDIUM
    assert f.subject_ids[0] == idc_a
    assert node_id(NodeKind.SERVICE, "주문") in f.subject_ids
    assert edge_id(EdgeKind.RUNS_IN, node_id(NodeKind.SYSTEM, "주문DB"), idc_a) in f.subject_ids
    assert "2개" in f.message
    assert f.fix is not None
    assert f.fix.params["action_type"] == "failover"


def test_cat003_negative_failover_or_split_locations() -> None:
    assert run(CriticalServicesConcentrated(), ctx_of(_concentrated(failover=True))) == []
    assert run(CriticalServicesConcentrated(), ctx_of(_concentrated(second_location=True))) == []
    g = G()
    g.service("사내위키", tier=3)
    assert run(CriticalServicesConcentrated(), ctx_of(g)) == []


# ---------------------------------------------------------------------------- JT-INT-*


def test_int001_held_person_asks_to_confirm_identity() -> None:
    g = G()
    pid = g.node(NodeKind.PERSON, "김대리", hold="동명이인: 명부에 김대리 2명")
    [f] = run(HeldElement(), ctx_of(g))
    assert f.status is FindingStatus.UNKNOWN
    assert f.severity is Severity.MEDIUM
    assert "동명이인" in f.message
    assert f.fix is not None
    assert f.fix.kind is FixKind.CONFIRM_IDENTITY
    assert f.fix.params["person_ids"] == [pid]


def test_int001_held_edge_maps_to_path_element() -> None:
    g = G()
    svc = g.service("결제")
    eid = g.edge(EdgeKind.DETECTED_BY, svc, g.node(NodeKind.TRIGGER, "오류율 알람"), hold="")
    [f] = run(HeldElement(), ctx_of(g))
    assert f.subject_ids == (eid,)
    assert f.service_id == svc
    assert f.element is Element.DETECTION
    assert f.fix is not None  # spec: CONFIRM_IDENTITY *or a generic review hint*, never nothing
    assert f.fix.kind is FixKind.REVIEW_ELEMENT
    assert f.fix.params == {"element_id": eid}
    assert "사유 미기재" in f.message
    assert "원문" in f.message


def test_int001_nothing_held() -> None:
    g = G()
    g.service("결제")
    assert run(HeldElement(), ctx_of(g)) == []


def test_int002_runbook_provenance_without_span_is_an_invariant_alarm() -> None:
    g = G()
    forged = Provenance.model_construct(
        source=SourceKind.RUNBOOK, extractor=Extractor.LLM, spans=(), ref=None
    )
    nid = node_id(NodeKind.SERVICE, "유령 서비스")
    # Only a validation bypass can produce this; model_construct is exactly such a bypass.
    g.b.add_node(
        Node.model_construct(id=nid, kind=NodeKind.SERVICE, name="유령 서비스", attrs={}, prov=(forged,))
    )
    [f] = run(UngroundedElement(), ctx_of(g))
    assert f.severity is Severity.CRITICAL
    assert f.subject_ids == (nid,)


def test_int002_runbook_attr_without_span() -> None:
    g = G()
    a, b = g.person("김대리"), g.person("박과장")
    eid = g.edge(EdgeKind.ESCALATES_TO, a, b, after_minutes=Attr(value=30))
    [f] = run(UngroundedElement(), ctx_of(g))
    assert f.subject_ids == (eid,)
    assert "after_minutes" in f.message


def test_int002_clean_graph() -> None:
    g, _ = _escalation_graph(60)
    assert run(UngroundedElement(), ctx_of(g)) == []


# ---------------------------------------------------------------------------- JT-SPF-*


def test_spf_single_person_and_single_channel() -> None:
    g = G()
    slack = g.node(NodeKind.CHANNEL, "Slack")
    svc = g.service("결제")
    kim = g.person("김대리", slack)
    g.edge(EdgeKind.OWNED_BY, svc, kim)
    ctx = ctx_of(g)
    [person] = run(SinglePersonPoint(), ctx)
    assert person.element is Element.OWNER
    assert person.fix is not None
    assert person.fix.kind is FixKind.ADD_DEPUTY
    assert person.fix.params["person_id"] == kim
    assert "버스 팩터 1" in person.message
    assert edge_id(EdgeKind.OWNED_BY, svc, kim) in person.subject_ids
    [channel] = run(SingleChannelPoint(), ctx)
    assert channel.element is Element.FALLBACK_CHANNEL
    assert channel.fix is not None
    assert channel.fix.kind is FixKind.ADD_FALLBACK_CHANNEL
    assert channel.fix.params["channel_id"] == slack


def test_spf_approver_only_single_point_reports_approver_element() -> None:
    g = G()
    slack, phone = g.node(NodeKind.CHANNEL, "Slack"), g.node(NodeKind.CHANNEL, "전화")
    svc = g.service("결제")
    for name in ("김대리", "박과장"):
        g.edge(EdgeKind.OWNED_BY, svc, g.person(name, slack, phone))
    lead = g.person("최팀장", slack, phone)
    rollback = g.node(NodeKind.ACTION, "롤백", action_type="rollback")
    g.edge(EdgeKind.MITIGATED_BY, svc, rollback)
    g.edge(EdgeKind.APPROVED_BY, rollback, lead)
    [f] = run(SinglePersonPoint(), ctx_of(g))
    assert f.subject_ids[1] == lead
    assert f.element is Element.APPROVER
    assert "'롤백' 승인" in f.message
    assert run(SingleChannelPoint(), ctx_of(g)) == []


def test_spf_negative_redundant_broken_or_non_critical() -> None:
    g = G()
    slack, phone = g.node(NodeKind.CHANNEL, "Slack"), g.node(NodeKind.CHANNEL, "전화")
    svc = g.service("결제")
    g.edge(EdgeKind.OWNED_BY, svc, g.person("김대리", slack))
    g.edge(EdgeKind.OWNED_BY, svc, g.person("박과장", phone))
    assert run(SinglePersonPoint(), ctx_of(g)) == []
    assert run(SingleChannelPoint(), ctx_of(g)) == []

    g = G()
    svc = g.service("결제")
    g.edge(EdgeKind.OWNED_BY, svc, g.person("이퇴사", status=PersonStatus.DEPARTED))
    assert run(SinglePersonPoint(), ctx_of(g)) == []  # broken path: PATH rules report it

    g = G()
    slack = g.node(NodeKind.CHANNEL, "Slack")
    wiki = g.service("사내위키", tier=3)
    g.edge(EdgeKind.OWNED_BY, wiki, g.person("김대리", slack))
    assert run(SinglePersonPoint(), ctx_of(g)) == []


# ---------------------------------------------------------------------------- determinism

ALL_RULES: tuple[Rule, ...] = (
    EscalationExceedsRecoveryTarget(),
    ReportingDeadlineLaterThanLaw(),
    ConflictingDeadlines(),
    BreachReportStepMissing(),
    BreachReporterMissing(),
    PrivacyReportStepMissing(),
    PrivacyReporterMissing(),
    ConsecutiveNightShifts(),
    InsufficientRest(),
    TooManyShiftsPerWeek(),
    OnCallPersonUnavailable(),
    CatalogServiceWithoutOwnerOrRunbook(),
    RunbookServiceNotInCatalog(),
    CriticalServicesConcentrated(),
    HeldElement(),
    UngroundedElement(),
    SinglePersonPoint(),
    SingleChannelPoint(),
)


def _everything() -> CheckContext:
    g, ids = _escalation_graph(60)
    g.obligation("KISA 신고", ids["svc"], reporter=ids["kim"], **breach(deadline_minutes=2880))
    g.obligation("KISA 신고2", ids["svc"], **breach(recipient="한국인터넷진흥원"))
    g.node(NodeKind.SERVICE, "정산", prov=(CATALOG,), tier=1)
    idc = g.node(NodeKind.LOCATION, "IDC-A", prov=(CATALOG,))
    g.edge(EdgeKind.RUNS_IN, ids["svc"], idc, prov=(CATALOG,))
    g.node(NodeKind.PERSON, "동명이인", hold="동명이인")
    shifts = (
        *(shift("김대리", at(d, 22), 8) for d in (1, 2, 3)),
        shift("이퇴사", at(5, 9), 9),
        shift("김대리", at(4, 12), 8),
    )
    return CheckContext(
        graph=g.build(),
        roster=ROSTER_ENTRIES,
        oncall=shifts,
        catalog=(CatalogService(name="정산"),),
    )


def test_all_rules_are_deterministic_and_ids_unique() -> None:
    first = sort_findings(f for r in ALL_RULES for f in r.check(_everything()))
    second = sort_findings(f for r in reversed(ALL_RULES) for f in r.check(_everything()))
    assert [f.model_dump_json() for f in first] == [f.model_dump_json() for f in second]
    rule_ids = {f.rule_id for f in first}
    assert {"JT-DDL-001", "JT-DDL-002", "JT-DDL-003", "JT-KR-003", "JT-ONC-001", "JT-ONC-004"} <= rule_ids
    assert {"JT-CAT-001", "JT-CAT-002", "JT-INT-001", "JT-SPF-002"} <= rule_ids
    assert len({f.id for f in first}) == len(first)
    assert all(f.id.startswith(f.rule_id + ":") for f in first)


def test_rule_infos_are_complete_and_korean() -> None:
    ids = [r.info.rule_id for r in ALL_RULES]
    assert len(set(ids)) == len(ids)
    for rule in ALL_RULES:
        assert rule.info.title
        assert rule.info.rationale
        assert any("가" <= ch <= "힣" for ch in rule.info.title + rule.info.rationale)


# ============================================================================ reviewer: adversarial


def hold(g: G, element_id: str, reason: str = "역번역 대조 불일치") -> str:
    """Put an existing node or edge on HOLD (판단 보류)."""
    table: dict[str, Any] = g.b.nodes if element_id in g.b.nodes else g.b.edges
    table[element_id] = table[element_id].model_copy(
        update={"verification": Verification.HOLD, "hold_reason": reason}
    )
    return element_id


def _hangul(text: str) -> bool:
    return any("가" <= ch <= "힣" for ch in text)


# ---------------------------------------------------------------------------- durations


@pytest.mark.parametrize(
    ("text", "minutes"),
    [
        ("1시간 30분 이내 신고", 90),
        ("1 시간 30 분", 90),
        ("1일 12시간", 2160),
        ("1h30min", 90),
        ("0.1시간", 6),  # float noise rounded away
        ("３０분 이내", 30),  # full-width digits
        ("30분 1시간", 30),  # growing units are two separate durations
        ("24시간, 48시간", 1440),
        ("이십사 시간 이내", None),  # Korean numerals are not guessed
        ("", None),
    ],
)
def test_parse_duration_compound_and_korean_edge_cases(text: str, minutes: float | None) -> None:
    assert parse_duration_minutes(text) == minutes


def test_durations_minutes_lists_every_clock_in_a_sentence() -> None:
    assert durations_minutes("30분 내 팀장 보고 후 KISA에 24시간 이내 신고") == [30, 1440]
    assert durations_minutes("1시간 30분 또는 2시간") == [90, 120]
    assert durations_minutes("2026년 10월 1일 시행") == []


# ---------------------------------------------------------------------------- JT-DDL-* (reviewer)


@pytest.mark.parametrize(
    "quote",
    [
        "KISA에 1시간 30분 이내 신고",  # compound duration equals the value
        "30분 내 팀장 보고 후 KISA에 1시간 30분 이내 신고",  # other clocks in the same sentence
    ],
)
def test_ddl003_quote_that_states_the_value_is_not_a_conflict(quote: str) -> None:
    g = G()
    deadline = Attr(value=90, spans=(span(quote),))
    g.obligation("KISA 신고", g.service("회원"), **breach(deadline_minutes=deadline))
    assert run(ConflictingDeadlines(), ctx_of(g)) == []


def test_ddl003_single_quote_contradicting_its_value() -> None:
    g = G()
    deadline = Attr(value=1440, spans=(span("KISA에 48시간 이내 신고"),))
    g.obligation("KISA 신고", g.service("회원"), **breach(deadline_minutes=deadline))
    [f] = run(ConflictingDeadlines(), ctx_of(g))
    assert f.status is FindingStatus.BROKEN
    assert f.fix is not None
    assert f.fix.params["stated_minutes"] == [1440, 2880]
    assert f.fix.params["required_minutes"] == 1440


def test_ddl002_and_ddl003_on_held_steps_are_unknown_not_broken() -> None:
    g = G()
    svc = g.service("회원")
    late = g.obligation("KISA 신고(운영)", **breach(deadline_minutes=2880))
    hold(g, g.edge(EdgeKind.MUST_REPORT, svc, late), "역번역 대조 불일치: 48시간 vs 24시간")
    g.obligation("KISA 신고(보안)", svc, **breach())
    ctx = ctx_of(g)
    [f] = run(ReportingDeadlineLaterThanLaw(), ctx)
    assert f.status is FindingStatus.UNKNOWN
    assert f.severity is Severity.CRITICAL
    assert "판단 보류" in f.message
    [pair] = run(ConflictingDeadlines(), ctx)
    assert pair.status is FindingStatus.UNKNOWN


@pytest.mark.parametrize(
    "recipient", ["보호나라", "KrCERT/CC", "ＫＩＳＡ", "한국 인터넷 진흥원", "인터넷침해대응센터(118)"]
)
def test_ddl002_kisa_named_by_its_reporting_front_door(recipient: str) -> None:
    g = G()
    g.obligation("침해 신고", g.service("회원"), **breach(recipient=recipient, deadline_minutes=2880))
    [f] = run(ReportingDeadlineLaterThanLaw(), ctx_of(g))
    assert f.legal_basis == BREACH_LAW.legal_basis
    assert f.status is FindingStatus.BROKEN


def test_ddl001_applies_to_any_tier_with_a_tight_target() -> None:
    g, ids = _escalation_graph(60)
    g.b.nodes[ids["svc"]] = g.b.nodes[ids["svc"]].model_copy(
        update={"attrs": _attrs(RUNBOOK, {"tier": 3, "slo_minutes": 30})}
    )
    [f] = run(EscalationExceedsRecoveryTarget(), ctx_of(g))
    assert f.service_id == ids["svc"]
    assert "목표 복구 시간 30분" in f.message


def test_ddl001_ignores_held_escalation_edges() -> None:
    g, ids = _escalation_graph(60)
    assert len(run(EscalationExceedsRecoveryTarget(), ctx_of(g))) == 1
    hold(g, edge_id(EdgeKind.ESCALATES_TO, ids["park"], ids["cto"]))
    timing = escalation_timing(ctx_of(g), ids["svc"])
    assert timing is not None
    assert not timing.reached_goal
    assert timing.minutes == 30
    assert run(EscalationExceedsRecoveryTarget(), ctx_of(g)) == []


# ---------------------------------------------------------------------------- JT-KR-* (reviewer)


def test_kr001_declared_regulated_service_missing_from_graph_is_unknown_not_silent() -> None:
    g = G()
    g.service("결제")
    ctx = ctx_of(g, policy=Policy(regulated_services=("회원DB",)))
    for rule in (BreachReportStepMissing(), PrivacyReportStepMissing()):
        [f] = run(rule, ctx)
        assert f.status is FindingStatus.UNKNOWN
        assert f.service_id is None
        assert "'회원DB'" in f.message
        assert f.fix is not None
        assert f.fix.params["service_names"] == ["회원DB"]


def test_kr001_regulated_names_match_spacing_case_and_systems() -> None:
    g = G()
    pay = g.service("결제 서비스")
    member_db = g.node(NodeKind.SYSTEM, "회원DB")
    policy = Policy(regulated_services=("결제서비스", "system:회원db", "  "))
    [f] = run(BreachReportStepMissing(), ctx_of(g, policy=policy))
    assert f.status is FindingStatus.BROKEN
    assert f.subject_ids == (pay, member_db)


def test_kr001_step_without_stated_incident_type_is_unknown_not_missing() -> None:
    g = G()
    pay = g.service("결제")
    g.obligation("KISA 신고", pay, recipient="KISA", deadline_minutes=1440)
    ctx = ctx_of(g)
    for rule in (BreachReportStepMissing(), PrivacyReportStepMissing()):  # KISA takes both reports
        [f] = run(rule, ctx)
        assert f.status is FindingStatus.UNKNOWN
        assert "확인할 수 없습니다" in f.message


def test_kr001_held_step_is_unknown() -> None:
    g = G()
    pay = g.service("결제")
    oid = g.obligation("KISA 침해 신고", **breach())
    hold(g, g.edge(EdgeKind.MUST_REPORT, pay, oid))
    [f] = run(BreachReportStepMissing(), ctx_of(g))
    assert f.status is FindingStatus.UNKNOWN


def test_kr001_absent_and_unclear_steps_are_reported_per_service() -> None:
    g = G()
    pay, member = _two_critical(g)
    g.obligation("KISA 신고", pay, recipient="KISA")
    findings = run(BreachReportStepMissing(), ctx_of(g))
    assert {f.service_id: f.status for f in findings} == {
        pay: FindingStatus.UNKNOWN,
        member: FindingStatus.BROKEN,
    }


def test_kr002_role_without_holders_names_the_role() -> None:
    g = G()
    pay = g.service("결제")
    role = g.node(NodeKind.ROLE, "보안팀장")
    g.obligation("KISA 신고", pay, reporter=role, **breach())
    [f] = run(BreachReporterMissing(), ctx_of(g))
    assert f.status is FindingStatus.BROKEN
    assert "보안팀장" in f.message
    assert "지정되어 있지 않습니다" not in f.message
    assert f.fix is not None
    assert f.fix.kind is FixKind.ASSIGN_OWNER
    assert f.fix.params["reporter_ids"] == [role]


def test_kr002_held_reporter_link_is_unknown_not_fine() -> None:
    g = G()
    pay = g.service("결제")
    reporter = g.person("보안담당")
    oid = g.obligation("KISA 신고", pay, **breach())
    hold(g, g.edge(EdgeKind.REPORTED_BY, oid, reporter))
    [f] = run(BreachReporterMissing(), ctx_of(g))
    assert f.status is FindingStatus.UNKNOWN
    assert "판단 보류" in f.message
    assert f.fix is not None
    assert f.fix.kind is FixKind.CONFIRM_IDENTITY
    assert f.fix.params["held_ids"] == [reporter]


@pytest.mark.parametrize(
    ("stated", "required", "expected"),
    [
        ("보호나라", BREACH_LAW.recipient, True),
        ("ＫＩＳＡ", BREACH_LAW.recipient, True),
        ("과학기술정보통신부장관", BREACH_LAW.recipient, True),
        ("개인정보 포털", PRIVACY_LAW.recipient, True),
        ("경찰청", BREACH_LAW.recipient, False),
        ("   ", BREACH_LAW.recipient, True),  # blank = unstated
    ],
)
def test_recipient_matches_korean_variants(stated: str, required: str, expected: bool) -> None:
    assert recipient_matches(stated, required) is expected


@pytest.mark.parametrize(
    ("minutes", "text"),
    [(0, "0분"), (59, "59분"), (60, "1시간"), (1500, "25시간"), (7.5, "7.5분"), (1e7, "10000000분")],
)
def test_format_minutes_edge_cases(minutes: float, text: str) -> None:
    assert format_minutes(minutes) == text


# ---------------------------------------------------------------------------- JT-ONC-* (reviewer)


def test_onc003_iso_week_spanning_new_year() -> None:
    shifts = [shift("김대리", datetime(y, m, d, 9), 9) for y, m, d in ((2026, 12, 29), (2026, 12, 30))]
    shifts += [shift("김대리", datetime(y, m, d, 9), 9) for y, m, d in ((2026, 12, 31), (2027, 1, 1))]
    [f] = run(TooManyShiftsPerWeek(), oncall_ctx(*shifts))
    assert "2026년 53주차" in f.message


def test_onc001_run_across_a_month_boundary() -> None:
    nights = [shift("김대리", datetime(2026, m, d, 22), 8) for m, d in ((10, 30), (10, 31), (11, 1))]
    [f] = run(ConsecutiveNightShifts(), oncall_ctx(*nights))
    assert "2026-10-30부터 2026-11-01까지 3일 연속" in f.message


def test_oncall_rules_are_silent_on_empty_inputs() -> None:
    for ctx in (oncall_ctx(), oncall_ctx(roster=())):
        for rule in (
            ConsecutiveNightShifts(),
            InsufficientRest(),
            TooManyShiftsPerWeek(),
            OnCallPersonUnavailable(),
        ):
            assert run(rule, ctx) == []


def test_onc004_timezone_aware_now_against_naive_shifts() -> None:
    kst = timezone(timedelta(hours=9))
    past = oncall_ctx(shift("이퇴사", at(1, 22), 8), now=at(5, 0).replace(tzinfo=kst))
    assert run(OnCallPersonUnavailable(), past) == []
    upcoming = oncall_ctx(shift("이퇴사", at(6, 22), 8), now=at(5, 0).replace(tzinfo=kst))
    assert len(run(OnCallPersonUnavailable(), upcoming)) == 1


# ---------------------------------------------------------------------------- JT-CAT-* (reviewer)


def test_cat003_failover_on_an_unrelated_service_does_not_hide_concentration() -> None:
    g = _concentrated()
    wiki = node_id(NodeKind.SERVICE, "사내위키")
    g.edge(EdgeKind.RUNS_IN, wiki, node_id(NodeKind.LOCATION, "IDC-A"), prov=(CATALOG,))
    g.edge(EdgeKind.MITIGATED_BY, wiki, g.node(NodeKind.ACTION, "위키 DR 전환", action_type="failover"))
    assert len(run(CriticalServicesConcentrated(), ctx_of(g))) == 1


def test_cat003_held_failover_does_not_protect() -> None:
    g = _concentrated()
    db = node_id(NodeKind.SYSTEM, "주문DB")
    hold(g, g.edge(EdgeKind.MITIGATED_BY, db, g.node(NodeKind.ACTION, "DR 전환", action_type="failover")))
    assert len(run(CriticalServicesConcentrated(), ctx_of(g))) == 1


def test_cat003_single_critical_service_message_reads_naturally() -> None:
    g = G()
    idc = g.node(NodeKind.LOCATION, "IDC-A", prov=(CATALOG,))
    g.edge(EdgeKind.RUNS_IN, g.service("결제"), idc, prov=(CATALOG,))
    [f] = run(CriticalServicesConcentrated(), ctx_of(g))
    assert "모두" not in f.message


def test_cat002_name_variants_match_the_catalog() -> None:
    g = G()
    g.service("결제 서비스")
    g.service("Order-API")
    catalog = (CatalogService(name="결제서비스"), CatalogService(name="order api"))
    assert run(RunbookServiceNotInCatalog(), ctx_of(g, catalog=catalog)) == []


# ---------------------------------------------------------------------------- JT-INT-001 (reviewer)


def test_int001_held_relation_between_verified_people_asks_to_check_the_source() -> None:
    g = G()
    a, b = g.person("김대리"), g.person("박과장")
    g.edge(EdgeKind.ESCALATES_TO, a, b, hold="역번역 대조 불일치: 10분 vs 15분", after_minutes=10)
    [f] = run(HeldElement(), ctx_of(g))
    assert f.fix is not None
    assert f.fix.kind is FixKind.REVIEW_ELEMENT  # a relation mismatch is not an identity question
    assert "원문" in f.message
    assert f.element is Element.OWNER


def test_int001_held_relation_to_an_ambiguous_person_confirms_only_that_person() -> None:
    g = G()
    svc = g.service("결제")
    kim = g.node(NodeKind.PERSON, "김대리", hold="동명이인/모호한 지칭")
    eid = g.edge(EdgeKind.OWNED_BY, svc, kim, hold="역번역 대조 불일치")
    findings = run(HeldElement(), ctx_of(g))
    assert len(findings) == 2
    [edge_finding] = [f for f in findings if f.subject_ids == (eid,)]
    assert edge_finding.fix is not None
    assert edge_finding.fix.kind is FixKind.CONFIRM_IDENTITY
    assert edge_finding.fix.params["person_ids"] == [kim]


# ---------------------------------------------------------------------------- invariants (reviewer)

_PEOPLE_NAMES = ("김대리", "박과장", "이주임", "최CTO", "정팀장")
_SERVICE_NAMES = ("결제", "회원", "주문 서비스", "Search-API")


@st.composite
def ops_contexts(draw: st.DrawFn) -> CheckContext:
    """Small random operations graphs with HOLD, roster, on-call and catalog noise."""
    g = G()
    channels = [
        g.node(NodeKind.CHANNEL, n)
        for n in draw(
            st.lists(st.sampled_from(["Slack", "카카오톡", "전화"]), min_size=1, max_size=3, unique=True)
        )
    ]
    names = draw(st.lists(st.sampled_from(_PEOPLE_NAMES), min_size=1, max_size=4, unique=True))
    people = [
        g.person(
            n,
            *draw(st.lists(st.sampled_from(channels), max_size=2, unique=True)),
            status=draw(st.sampled_from(list(PersonStatus))),
        )
        for n in names
    ]
    loc = g.node(NodeKind.LOCATION, "IDC-A", prov=(CATALOG,))
    for name in draw(st.lists(st.sampled_from(_SERVICE_NAMES), min_size=1, max_size=3, unique=True)):
        svc = g.service(name, tier=draw(st.integers(1, 3)))
        for p in draw(st.lists(st.sampled_from(people), max_size=2, unique=True)):
            g.edge(EdgeKind.OWNED_BY, svc, p)
        if draw(st.booleans()):
            g.edge(EdgeKind.RUNS_IN, svc, loc, prov=(CATALOG,))
        if draw(st.booleans()):
            action = g.node(
                NodeKind.ACTION, f"{name} 조치", action_type=draw(st.sampled_from(list(ActionType))).value
            )
            g.edge(EdgeKind.MITIGATED_BY, svc, action)
            for p in draw(st.lists(st.sampled_from(people), max_size=1)):
                g.edge(EdgeKind.APPROVED_BY, action, p)
        if draw(st.booleans()):
            g.obligation(
                f"{name} 신고",
                svc,
                reporter=draw(st.sampled_from([None, *people])),
                incident_type=draw(st.sampled_from(["breach", "privacy_leak", "outage"])),
                recipient=draw(st.sampled_from(["KISA", "개인정보위", "보호나라", "고객"])),
                deadline_minutes=draw(st.sampled_from([720, 1440, 2880, 4320])),
            )
    for a, b in draw(st.lists(st.tuples(st.sampled_from(people), st.sampled_from(people)), max_size=4)):
        if a != b:
            g.edge(EdgeKind.ESCALATES_TO, a, b, after_minutes=draw(st.integers(0, 120)))
    for element in draw(st.lists(st.sampled_from(sorted([*g.b.nodes, *g.b.edges])), max_size=3, unique=True)):
        hold(g, element)
    shifts = tuple(
        shift(draw(st.sampled_from([*_PEOPLE_NAMES, "유령"])), at(day, hour), hours)
        for day, hour, hours in draw(
            st.lists(st.tuples(st.integers(1, 20), st.integers(0, 23), st.integers(1, 24)), max_size=6)
        )
    )
    roster = tuple(
        RosterEntry(name=n, status=draw(st.sampled_from(list(PersonStatus))))
        for n in draw(st.lists(st.sampled_from(_PEOPLE_NAMES), max_size=5, unique=True))
    )
    catalog = tuple(
        CatalogService(name=n) for n in draw(st.lists(st.sampled_from(_SERVICE_NAMES), unique=True))
    )
    policy = Policy(
        regulated_services=tuple(draw(st.lists(st.sampled_from([*_SERVICE_NAMES, "회원DB"]), max_size=2)))
    )
    return CheckContext(graph=g.build(), roster=roster, oncall=shifts, catalog=catalog, policy=policy)


@settings(max_examples=60, deadline=None)
@given(ops_contexts())
def test_findings_are_deterministic_well_formed_and_grounded_in_the_graph(ctx: CheckContext) -> None:
    raw = [f for rule in ALL_RULES for f in rule.check(ctx)]
    again = [f for rule in ALL_RULES for f in rule.check(ctx)]
    assert [f.model_dump_json() for f in raw] == [f.model_dump_json() for f in again]
    assert len({f.id for f in raw}) == len(raw)  # no rule emits the same finding twice
    known = set(ctx.graph.nodes) | set(ctx.graph.edges)
    services = {n.id for n in ctx.graph.nodes_of(NodeKind.SERVICE, NodeKind.SYSTEM)}
    for f in raw:
        assert f.id.startswith(f.rule_id + ":")
        assert set(f.subject_ids) <= known
        assert f.service_id is None or f.service_id in services
        assert _hangul(f.title)
        assert _hangul(f.message)
        assert "e+" not in f.message  # never scientific notation in a Korean sentence
        if f.rule_id.startswith("JT-KR-") or f.rule_id == "JT-DDL-002":
            assert f.legal_basis


# ============================================================================ second review: adversarial

KST = timezone(timedelta(hours=9))


# ---------------------------------------------------------------------------- durations (Korean idioms)


@pytest.mark.parametrize(
    ("text", "minutes"),
    [
        ("1시간 반 이내 신고", [90]),
        ("1시간반이내", [90]),
        ("2일 반", [3600]),
        ("2시간 반 동안", [150]),
        ("１시간 반", [90]),  # full-width digit
        ("1시간 반드시 보고", [60]),  # "반드시"(must) is not a half
        ("1시간 반기 보고", [60]),  # "반기"(half-year) is not a half
        ("30분 반", [30]),  # nobody writes half a minute; "반" is ignored there
        ("익일 오전 9시 30분까지 신고", []),  # a clock time, not a 30-minute duration
        ("9시30분", []),
        ("10월 1일 9시 30분", []),
        ("18시간 30분", [1110]),  # "…시간 30분" is still a compound duration
        ("인원의 3분의 2 이상", []),  # a fraction, not 3 minutes
        ("1시간 반에서 2시간", [90, 120]),
    ],
)
def test_durations_korean_half_clock_times_and_fractions(text: str, minutes: list[float]) -> None:
    assert durations_minutes(text) == minutes


@settings(max_examples=150, deadline=None)
@given(st.integers(0, 200), st.integers(1, 59), st.integers(0, 23))
def test_duration_grammar_properties(hours: int, mins: int, clock: int) -> None:
    assert durations_minutes(f"{hours}시간 {mins}분 이내") == [hours * 60 + mins]
    assert durations_minutes(f"{hours}시간 반 이내") == [hours * 60 + 30]
    assert durations_minutes(f"{hours}일 반") == [hours * 1440 + 720]
    assert durations_minutes(f"오전 {clock}시 {mins}분까지") == []


# ---------------------------------------------------------------------------- JT-DDL-* (second review)


@pytest.mark.parametrize(
    ("value", "quote"),
    [
        (90, "KISA에 1시간 반 이내 신고"),  # was read as 1시간 -> false "1시간 vs 90분" conflict
        (1440, "익일 오전 9시 30분까지 KISA 신고"),  # clock time was read as a 30-minute deadline
    ],
)
def test_ddl003_korean_idioms_are_not_false_conflicts(value: int, quote: str) -> None:
    g = G()
    deadline = Attr(value=value, spans=(span(quote),))
    g.obligation("KISA 신고", g.service("회원"), **breach(deadline_minutes=deadline))
    assert run(ConflictingDeadlines(), ctx_of(g)) == []


def test_ddl003_recommendation_never_relaxes_a_stricter_internal_target() -> None:
    g = G()
    svc = g.service("회원")
    g.obligation("KISA 신고(운영)", svc, **breach(deadline_minutes=60))
    g.obligation("KISA 신고(보안)", svc, **breach(deadline_minutes=90))
    [f] = run(ConflictingDeadlines(), ctx_of(g))
    assert f.fix is not None
    assert f.fix.params["required_minutes"] == 60
    assert "권장: 1시간" in f.message
    assert "24시간" not in f.message  # the legal maximum would loosen the team's own clock

    g = G()
    svc = g.service("회원")
    g.obligation("KISA 신고(운영)", svc, **breach(deadline_minutes=2880))
    g.obligation("KISA 신고(보안)", svc, **breach(deadline_minutes=4320))
    [f] = run(ConflictingDeadlines(), ctx_of(g))
    assert f.fix is not None
    assert f.fix.params["required_minutes"] == 1440  # both too late: the law wins


def test_parallel_must_report_edges_keep_the_findings_service() -> None:
    """The assembler keeps contradicting statements as ``#disc`` parallel edges."""
    g = G()
    svc = g.service("회원")
    oid = g.obligation("KISA 신고", svc, **breach(deadline_minutes=2880))
    g.b.add_edge(
        Edge(
            id=edge_id(EdgeKind.MUST_REPORT, svc, oid, "deadline-2880"),
            kind=EdgeKind.MUST_REPORT,
            src=svc,
            dst=oid,
            prov=(RUNBOOK,),
        )
    )
    ctx = ctx_of(g)
    assert reporting_services(ctx, oid) == [svc]
    [late] = run(ReportingDeadlineLaterThanLaw(), ctx)
    assert late.service_id == svc
    [no_reporter] = run(BreachReporterMissing(), ctx)
    assert no_reporter.service_id == svc


def test_law_text_rejects_incident_types_without_a_statutory_default() -> None:
    assert law_text(IncidentType.BREACH) == BREACH_LAW.legal_basis
    assert law_text(IncidentType.PRIVACY_LEAK) == PRIVACY_LAW.legal_basis
    with pytest.raises(ValueError, match="outage"):
        law_text(IncidentType.OUTAGE)


def test_ddl001_wording_follows_whether_approval_is_needed() -> None:
    g = G()
    slack = g.node(NodeKind.CHANNEL, "Slack")
    svc = g.service("결제")
    gone = g.person("이퇴사", slack, status=PersonStatus.DEPARTED)
    g.edge(EdgeKind.OWNED_BY, svc, gone)
    g.edge(EdgeKind.ESCALATES_TO, gone, g.person("박팀장", slack), after_minutes=90)
    timing = escalation_timing(ctx_of(g), svc)
    assert timing is not None
    assert timing.reached_goal
    assert not timing.approval_needed
    [f] = run(EscalationExceedsRecoveryTarget(), ctx_of(g))
    assert "활동 중인 담당자 '박팀장'에게" in f.message
    assert "승인" not in f.message  # nothing here needs sign-off

    g, _ = _escalation_graph(60)
    [gated] = run(EscalationExceedsRecoveryTarget(), ctx_of(g))
    assert "승인할 수 있는 활동 담당자 '최CTO'에게" in gated.message


# ---------------------------------------------------------------------------- JT-CAT-003 (second review)


def _two_centres(*, second: str | None = "verified", db_only_in_a: bool = False) -> tuple[G, dict[str, str]]:
    g = G()
    ids = {
        "a": g.node(NodeKind.LOCATION, "IDC-A", prov=(CATALOG,)),
        "b": g.node(NodeKind.LOCATION, "IDC-B", prov=(CATALOG,)),
        "pay": g.service("결제"),
    }
    g.edge(EdgeKind.RUNS_IN, ids["pay"], ids["a"], prov=(CATALOG,))
    if second is not None:
        g.edge(
            EdgeKind.RUNS_IN, ids["pay"], ids["b"], hold="역번역 대조 불일치" if second == "held" else None
        )
    if db_only_in_a:
        ids["db"] = g.node(NodeKind.SYSTEM, "결제DB", prov=(CATALOG,))
        g.edge(EdgeKind.DEPENDS_ON, ids["pay"], ids["db"], prov=(CATALOG,))
        g.edge(EdgeKind.RUNS_IN, ids["db"], ids["a"], prov=(CATALOG,))
    return g, ids


def test_cat003_active_active_service_is_not_concentrated_in_either_centre() -> None:
    g, ids = _two_centres()
    ctx = ctx_of(g)
    assert run(CriticalServicesConcentrated(), ctx) == []
    # blast_radius still lists the exposure on purpose; only the concentration verdict differs
    assert blast_radius(ctx.graph, [ids["a"]]).impacted_services == (ids["pay"],)


def test_cat003_redundant_placement_does_not_hide_a_single_centre_dependency() -> None:
    g, ids = _two_centres(db_only_in_a=True)
    [f] = run(CriticalServicesConcentrated(), ctx_of(g))
    assert f.subject_ids[0] == ids["a"]
    # the explanation follows the real cause (through the DB), not the redundant direct placement
    assert edge_id(EdgeKind.DEPENDS_ON, ids["pay"], ids["db"]) in f.subject_ids
    assert edge_id(EdgeKind.RUNS_IN, ids["db"], ids["a"]) in f.subject_ids
    assert edge_id(EdgeKind.RUNS_IN, ids["pay"], ids["a"]) not in f.subject_ids
    assert f.fix is not None
    assert f.fix.kind is FixKind.ADD_FAILOVER
    assert f.fix.params["location_id"] == ids["a"]


def test_cat003_held_second_placement_is_not_an_escape_route() -> None:
    g, ids = _two_centres(second="held")
    assert [f.subject_ids[0] for f in run(CriticalServicesConcentrated(), ctx_of(g))] == [ids["a"]]


def _naive_down(graph: OpsGraph, loc: str) -> set[str]:
    """Independent fixpoint: all dependencies are needed, verified placements are alternatives."""
    down = {loc}
    changed = True
    while changed:
        changed = False
        for nid in sorted(graph.nodes):
            if nid in down:
                continue
            out = [e for e in graph.edges.values() if e.src == nid]
            deps = [e.dst for e in out if e.kind is EdgeKind.DEPENDS_ON]
            places = [e for e in out if e.kind is EdgeKind.RUNS_IN]
            verified = [e.dst for e in places if e.verification is Verification.VERIFIED]
            if any(d in down for d in deps) or (
                any(e.dst in down for e in places) and all(v in down for v in verified)
            ):
                down.add(nid)
                changed = True
    return down


@st.composite
def placement_contexts(draw: st.DrawFn) -> CheckContext:
    g = G()
    locations = [
        g.node(NodeKind.LOCATION, f"loc{i}", prov=(CATALOG,)) for i in range(draw(st.integers(1, 3)))
    ]
    nodes = [g.service(f"svc{i}", tier=draw(st.sampled_from([1, 3]))) for i in range(draw(st.integers(1, 4)))]
    nodes += [g.node(NodeKind.SYSTEM, f"sys{i}", prov=(CATALOG,)) for i in range(draw(st.integers(0, 3)))]
    for src in nodes:
        for dst in draw(st.lists(st.sampled_from(nodes), max_size=2, unique=True)):
            if dst != src:
                g.edge(EdgeKind.DEPENDS_ON, src, dst, prov=(CATALOG,))
        for loc in draw(st.lists(st.sampled_from(locations), max_size=2, unique=True)):
            g.edge(
                EdgeKind.RUNS_IN, src, loc, prov=(CATALOG,), hold=draw(st.sampled_from([None, None, "보류"]))
            )
    return ctx_of(g)


@settings(max_examples=150, deadline=None)
@given(placement_contexts())
def test_cat003_matches_a_naive_redundancy_aware_fixpoint(ctx: CheckContext) -> None:
    critical = {s.id for s in ctx.critical_services()}
    findings = run(CriticalServicesConcentrated(), ctx)
    fired = {f.subject_ids[0]: f for f in findings}
    assert len(fired) == len(findings)
    for loc in ctx.graph.nodes_of(NodeKind.LOCATION):
        down = _naive_down(ctx.graph, loc.id)
        expected = bool(critical) and critical <= down
        assert (loc.id in fired) is expected
        if not expected:
            continue
        # a true concentration is always also exposure, and the evidence is a causal chain
        assert critical <= set(blast_radius(ctx.graph, [loc.id]).impacted_services)
        for eid in (s for s in fired[loc.id].subject_ids if s in ctx.graph.edges):
            edge = ctx.graph.edges[eid]
            assert edge.kind in (EdgeKind.DEPENDS_ON, EdgeKind.RUNS_IN)
            assert edge.src in down
            assert edge.dst in down


# ---------------------------------------------------------------------------- JT-ONC-* (second review)


def _kst(day: int, hour: int) -> datetime:
    return datetime(2026, 10, day, hour, tzinfo=KST)


def _utc(day: int, hour: int) -> datetime:
    return datetime(2026, 10, day, hour, tzinfo=UTC)


def test_onc002_rest_gap_is_real_elapsed_time_across_utc_offsets() -> None:
    night = OnCallShift(person="김대리", start=_kst(1, 21), end=_kst(2, 9))
    # 13:00Z is 22:00 KST: 13 hours after the night shift ended, although the wall clocks read 4h
    late = OnCallShift(person="김대리", start=_utc(2, 13), end=_utc(2, 21))
    assert run(InsufficientRest(), oncall_ctx(night, late)) == []
    # 01:00Z is 10:00 KST: one hour of rest, although the raw wall clocks read 09:00 -> 01:00
    early = OnCallShift(person="김대리", start=_utc(2, 1), end=_utc(2, 8))
    [f] = run(InsufficientRest(), oncall_ctx(night, early))
    assert "2026-10-02 09:00에 당직을 마치고 1시간 만인 2026-10-02 10:00에" in f.message


def test_onc001_utc_rows_in_a_kst_table_are_night_shifts() -> None:
    first = OnCallShift(person="김대리", start=_kst(1, 22), end=_kst(2, 6))
    utc_nights = [OnCallShift(person="김대리", start=_utc(d, 13), end=_utc(d, 21)) for d in (2, 3)]
    [f] = run(ConsecutiveNightShifts(), oncall_ctx(first, *utc_nights))
    assert "2026-10-01부터 2026-10-03까지 3일 연속" in f.message


def test_table_zone_is_the_earliest_aware_rows_zone() -> None:
    naive = shift("김대리", at(1, 0), 8)
    later_kst = OnCallShift(person="김대리", start=_kst(3, 9), end=_kst(3, 18))
    earlier_utc = OnCallShift(person="김대리", start=_utc(2, 9), end=_utc(2, 18))
    assert table_zone(oncall_ctx(naive, later_kst, earlier_utc)) == UTC
    assert table_zone(oncall_ctx(naive)) is None
    assert table_zone(oncall_ctx()) is None
    # same instant written twice: the smaller offset wins whatever the row order
    same_kst = OnCallShift(person="김대리", start=_kst(2, 18), end=_kst(2, 20))
    for rows in ((earlier_utc, same_kst), (same_kst, earlier_utc)):
        assert table_zone(oncall_ctx(*rows)) == UTC


_ZONES = (None, KST, UTC, timezone(timedelta(hours=-5)))


@st.composite
def oncall_tables(draw: st.DrawFn) -> list[OnCallShift]:
    rows = draw(
        st.lists(
            st.tuples(
                st.sampled_from(["김대리", "Kim", "이퇴사", "정예정", "유령"]),
                st.integers(1, 14),
                st.integers(0, 23),
                st.integers(1, 30),
                st.sampled_from(_ZONES),
            ),
            max_size=8,
        )
    )
    return [
        OnCallShift(
            person=person,
            start=datetime(2026, 10, day, hour, tzinfo=zone),
            end=datetime(2026, 10, day, hour, tzinfo=zone) + timedelta(hours=hours),
        )
        for person, day, hour, hours, zone in rows
    ]


@settings(max_examples=100, deadline=None)
@given(oncall_tables(), st.data())
def test_oncall_findings_do_not_depend_on_row_order(rows: list[OnCallShift], data: st.DataObject) -> None:
    shuffled = data.draw(st.permutations(rows))
    rules = (ConsecutiveNightShifts(), InsufficientRest(), TooManyShiftsPerWeek(), OnCallPersonUnavailable())

    def dump(table: list[OnCallShift]) -> list[str]:
        return [f.model_dump_json() for rule in rules for f in run(rule, oncall_ctx(*table))]

    assert dump(rows) == dump(list(shuffled))
    for shifts in shifts_by_person(oncall_ctx(*rows)).values():
        assert shifts == sorted(set(shifts))
        assert all(s.start.tzinfo is None and s.end > s.start for s in shifts)


# ---------------------------------------------------------------------------- whole family


def test_every_rule_is_silent_on_an_empty_organisation() -> None:
    ctx = CheckContext(graph=OpsGraph())
    for rule in ALL_RULES:
        assert run(rule, ctx) == []


def test_registry_runs_this_family_exactly_like_direct_calls() -> None:
    ctx = _everything()
    via_registry = run_checks(ctx, [r.info.rule_id for r in ALL_RULES])
    direct = sort_findings(f for rule in ALL_RULES for f in rule.check(ctx))
    assert [f.model_dump_json() for f in via_registry] == [f.model_dump_json() for f in direct]


def test_org_level_message_stays_readable_for_200_services() -> None:
    g = G()
    ids = [g.service(f"서비스{i:03d}") for i in range(200)]
    [f] = run(BreachReportStepMissing(), ctx_of(g))
    assert f.service_id is None
    assert f.subject_ids == tuple(sorted(ids))  # the full list still travels with the finding
    assert "'서비스000', '서비스001', '서비스002', '서비스003', '서비스004' 외 195개" in f.message
    assert "서비스199" not in f.message


# ---------------------------------------------------------------------------- org-wide duties


def test_obligation_without_must_report_is_org_wide_and_binds_every_service() -> None:
    g = G()
    pay, member = _two_critical(g)
    own = g.obligation("결제 고객 공지", pay, recipient="고객", incident_type="outage")
    org = g.obligation("KISA 침해사고 신고", **breach())
    ctx = ctx_of(g)
    assert ctx.is_org_obligation(org)
    assert not ctx.is_org_obligation(own)
    assert not ctx.is_org_obligation(pay)  # not an obligation at all
    assert [o.id for o in ctx.org_obligations()] == [org]
    assert [o.id for o in ctx.obligations(pay)] == sorted([own, org])
    assert [o.id for o in ctx.obligations(member)] == [org]
    assert [s.id for s in ctx.obligation_services(org)] == sorted([pay, member])
    assert [s.id for s in ctx.obligation_services(own)] == [pay]
    assert reporting_services(ctx, org) == sorted([pay, member])


def test_held_must_report_link_keeps_the_duty_service_specific() -> None:
    """An unconfirmed link must not silently widen one service's step into a company-wide one."""
    g = G()
    pay, member = _two_critical(g)
    oid = g.obligation("KISA 침해 신고", **breach())
    hold(g, g.edge(EdgeKind.MUST_REPORT, pay, oid))
    ctx = ctx_of(g)
    assert not ctx.is_org_obligation(oid)
    assert [o.id for o in ctx.obligations(member)] == []
    assert [o.id for o in ctx.obligations(pay)] == [oid]


def test_org_wide_breach_step_satisfies_kr001_but_not_the_privacy_duty() -> None:
    """The demo case: the security procedure has a KISA step that names no service."""
    g = G()
    pay, member = _two_critical(g)
    g.obligation("KISA 침해사고 신고", reporter=g.person("오준석"), **breach(deadline_minutes=2880))
    ctx = ctx_of(g)
    assert run(BreachReportStepMissing(), ctx) == []
    [kr3] = run(PrivacyReportStepMissing(), ctx)
    assert kr3.service_id is None
    assert kr3.subject_ids == (pay, member)
    assert run(BreachReporterMissing(), ctx) == []


def test_ddl002_on_an_org_wide_step_belongs_to_no_single_service() -> None:
    g = G()
    _two_critical(g)
    oid = g.obligation("KISA 침해사고 신고", **breach(deadline_minutes=2880))
    [f] = run(ReportingDeadlineLaterThanLaw(), ctx_of(g))
    assert f.service_id is None
    assert f.subject_ids == (oid,)  # highlighting every service would bury the node to fix
    assert f.status is FindingStatus.BROKEN
    assert "48시간" in f.message


def test_ddl003_conflict_with_an_org_wide_step_is_organisation_level() -> None:
    g = G()
    pay, _ = _two_critical(g)
    own = g.obligation("결제 KISA 신고", pay, **breach())
    org = g.obligation("전사 KISA 신고", **breach(recipient="한국인터넷진흥원", deadline_minutes=2880))
    [f] = run(ConflictingDeadlines(), ctx_of(g))
    assert f.service_id is None
    assert set(f.subject_ids) >= {own, org, pay}


def test_kr002_departed_reporter_of_an_org_wide_step() -> None:
    g = G()
    pay, member = _two_critical(g)
    gone = g.person("오준석", status=PersonStatus.DEPARTED)
    oid = g.obligation("KISA 침해사고 신고", reporter=gone, **breach())
    [f] = run(BreachReporterMissing(), ctx_of(g))
    assert f.status is FindingStatus.BROKEN
    assert f.service_id is None
    assert set(f.subject_ids) == {oid, gone, pay, member}


@settings(max_examples=60, deadline=None)
@given(st.lists(st.booleans(), min_size=1, max_size=5), st.booleans())
def test_org_wide_step_closes_every_kr001_gap(linked: list[bool], org_wide: bool) -> None:
    """Without an org-wide step, KR-001 names exactly the services lacking their own step; with
    one, nobody lacks it. Adding a company-wide duty can never add a finding (monotonicity)."""
    g = G()
    services = [g.service(f"서비스{i}") for i in range(len(linked))]
    for i, (sid, has_step) in enumerate(zip(services, linked, strict=True)):
        if has_step:
            g.obligation(f"신고{i}", sid, **breach())
    if org_wide:
        g.obligation("전사 침해사고 신고", **breach())
    findings = run(BreachReportStepMissing(), ctx_of(g))
    if org_wide:
        assert findings == []
        return
    missing = sorted(sid for sid, has_step in zip(services, linked, strict=True) if not has_step)
    flagged = sorted(s for f in findings for s in f.subject_ids)
    assert flagged == missing
