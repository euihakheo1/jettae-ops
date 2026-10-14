"""On-call schedule rules (당직): ``JT-ONC-001`` … ``JT-ONC-004``.

Why: an exhausted or departed on-call engineer is a broken response path that no runbook text
reveals. These rules compare the on-call table (``ctx.oncall``) with the *team-agreed* limits in
``Policy.fatigue`` and with the roster (``ctx.roster``).

What they deliberately do NOT do: judge whether on-call waiting time is legally working time.
The Supreme Court's November 2024 reasoning on standby time (대법원 2024.11 대기시간 판단
기준 — degree of restriction, required response time, frequency of calls) is cited only as
context for why teams set such limits; the thresholds here are the team's own agreement.

Time handling: night hours are a local-time notion, so shifts are compared on the wall clock of
the table's own zone — the zone of its earliest timezone-aware shift (ingest writes the whole
table in one zone, naive input → Asia/Seoul). Aware datetimes in any other offset are first
converted to that zone, so a row written in UTC ("13:00Z" = 22:00 KST) is still a night shift
and rest gaps between rows with different offsets are real elapsed time; naive datetimes are
taken as already on that wall clock. Shifts whose end is not after their start are malformed
input and ignored by the fatigue rules.
``ctx.now`` (when given) limits the roster check to shifts that are not over yet; the fatigue
rules look at the whole table because a past pattern is still a policy violation.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime, timedelta, tzinfo

from jettae.check.context import CheckContext
from jettae.check.registry import finding, register
from jettae.check.rules.regulatory import node_aliases
from jettae.domain import (
    FatiguePolicy,
    Finding,
    FindingStatus,
    FixHint,
    FixKind,
    NodeKind,
    PersonStatus,
    RosterEntry,
    RuleInfo,
    Severity,
    node_id,
)

_DAY = timedelta(days=1)
_TIME_FMT = "%Y-%m-%d %H:%M"


@dataclass(frozen=True, order=True)
class Shift:
    """A normalised shift: wall-clock interval, owned by a canonical roster name."""

    start: datetime
    end: datetime
    person: str


def table_zone(ctx: CheckContext) -> tzinfo | None:
    """The zone the on-call table is written in: that of its earliest timezone-aware shift.

    Ties between equal instants are broken by UTC offset so the choice never depends on row
    order. ``None`` when every shift is naive (then wall clocks are compared as written).
    """
    aware = [s.start for s in ctx.oncall if s.start.utcoffset() is not None]
    if not aware:
        return None
    return min(aware, key=lambda dt: (dt, dt.utcoffset())).tzinfo


def _wall(dt: datetime, zone: tzinfo | None) -> datetime:
    """``dt`` on the table's wall clock (naive), converting aware values into ``zone`` first."""
    if zone is not None and dt.utcoffset() is not None:
        dt = dt.astimezone(zone)
    return dt.replace(tzinfo=None)


def roster_lookup(roster: tuple[RosterEntry, ...], name: str) -> RosterEntry | None:
    """Exact name first, then alias — the same order the assembler resolves people in."""
    exact = next((r for r in roster if r.name == name), None)
    if exact is not None:
        return exact
    return next((r for r in roster if name in r.aliases), None)


def shifts_by_person(ctx: CheckContext) -> dict[str, list[Shift]]:
    """Valid shifts grouped by canonical roster name (alias rows merge), sorted and de-duplicated."""
    zone = table_zone(ctx)
    grouped: dict[str, set[Shift]] = defaultdict(set)
    for raw in ctx.oncall:
        entry = roster_lookup(ctx.roster, raw.person)
        name = entry.name if entry is not None else raw.person
        start, end = _wall(raw.start, zone), _wall(raw.end, zone)
        if end > start:
            grouped[name].add(Shift(start, end, name))
    return {name: sorted(grouped[name]) for name in sorted(grouped)}


def _person_subjects(ctx: CheckContext, name: str) -> list[str]:
    """Graph person node for a roster name, if the graph has one."""
    nid = node_id(NodeKind.PERSON, name)
    if nid in ctx.graph.nodes:
        return [nid]
    return [p.id for p in ctx.graph.nodes_of(NodeKind.PERSON) if p.name == name or name in node_aliases(p)]


def _fix(name: str, shifts: list[Shift]) -> FixHint:
    return FixHint(
        kind=FixKind.ADJUST_ONCALL,
        params={"person": name, "shift_starts": [s.start.isoformat() for s in shifts]},
    )


# ---------------------------------------------------------------------------- JT-ONC-001


def night_dates(shift: Shift, fatigue: FatiguePolicy) -> list[date]:
    """Calendar nights (dated by the evening they start) that ``shift`` overlaps.

    A night ``N`` is ``[N + night_start_hour, N(+1 if wrapping) + night_end_hour)``; with the
    default 22→6 it spans midnight. Equal start/end hours mean "no night window".
    """
    start_h, end_h = fatigue.night_start_hour, fatigue.night_end_hour
    if start_h == end_h:
        return []
    out: list[date] = []
    day = shift.start.date() - _DAY
    while day <= shift.end.date():
        midnight = datetime.combine(day, datetime.min.time())
        w_start = midnight + timedelta(hours=start_h)
        w_end = midnight + timedelta(days=1 if start_h > end_h else 0, hours=end_h)
        if w_start < shift.end and shift.start < w_end:
            out.append(day)
        day += _DAY
    return out


def _runs(days: list[date]) -> list[list[date]]:
    """Maximal runs of consecutive dates in a sorted, de-duplicated list."""
    runs: list[list[date]] = []
    for d in days:
        if runs and runs[-1][-1] + _DAY == d:
            runs[-1].append(d)
        else:
            runs.append([d])
    return runs


@register
class ConsecutiveNightShifts:
    info = RuleInfo(
        rule_id="JT-ONC-001",
        title="연속 야간 당직 초과",
        severity=Severity.MEDIUM,
        element=None,
        rationale=(
            "팀이 합의한 연속 야간 당직 한도를 넘기면 응답이 늦어지고 판단 실수가 늘어난다. "
            "법적 근로시간 여부는 판단하지 않고 팀 합의 기준만 확인한다."
        ),
    )

    def check(self, ctx: CheckContext) -> Iterator[Finding]:
        fatigue = ctx.policy.fatigue
        for name, shifts in shifts_by_person(ctx).items():
            nights = sorted({d for s in shifts for d in night_dates(s, fatigue)})
            for run in _runs(nights):
                if len(run) <= fatigue.max_consecutive_night_shifts:
                    continue
                involved = [s for s in shifts if set(night_dates(s, fatigue)) & set(run)]
                yield finding(
                    self.info,
                    keys=(name, run[0].isoformat()),
                    title=self.info.title,
                    message=(
                        f"{name}님이 {run[0].isoformat()}부터 {run[-1].isoformat()}까지 "
                        f"{len(run)}일 연속 야간 당직을 맡습니다"
                        f"(팀 합의 기준: 최대 {fatigue.max_consecutive_night_shifts}일 연속)."
                    ),
                    subject_ids=_person_subjects(ctx, name),
                    fix=_fix(name, involved),
                )


# ---------------------------------------------------------------------------- JT-ONC-002


@register
class InsufficientRest:
    info = RuleInfo(
        rule_id="JT-ONC-002",
        title="휴식 시간 미달",
        severity=Severity.MEDIUM,
        element=None,
        rationale=(
            "당직이 끝나고 다음 당직까지 팀 합의 최소 휴식 시간이 보장되지 않으면 피로가 누적된다. "
            "이어 붙은(간격 0) 당직이나 겹친 당직도 휴식이 없는 것으로 본다."
        ),
    )

    def check(self, ctx: CheckContext) -> Iterator[Finding]:
        min_rest = ctx.policy.fatigue.min_rest_hours
        for name, shifts in shifts_by_person(ctx).items():
            busy_until = shifts[0].end
            for prev, nxt in itertools.pairwise(shifts):
                busy_until = max(busy_until, prev.end)
                gap_hours = max(0.0, (nxt.start - busy_until).total_seconds() / 3600)
                if gap_hours < min_rest:
                    yield finding(
                        self.info,
                        keys=(name, busy_until.isoformat(), nxt.start.isoformat()),
                        title=self.info.title,
                        message=(
                            f"{name}님은 {busy_until.strftime(_TIME_FMT)}에 당직을 마치고 "
                            f"{round(gap_hours, 1):g}시간 만인 {nxt.start.strftime(_TIME_FMT)}에 다시 당직을 "
                            f"시작합니다(팀 합의 최소 휴식 {min_rest:g}시간)."
                        ),
                        subject_ids=_person_subjects(ctx, name),
                        fix=_fix(name, [prev, nxt]),
                    )


# ---------------------------------------------------------------------------- JT-ONC-003


@register
class TooManyShiftsPerWeek:
    info = RuleInfo(
        rule_id="JT-ONC-003",
        title="주간 당직 과다",
        severity=Severity.LOW,
        element=None,
        rationale="한 사람이 한 주(ISO 주)에 팀 합의 횟수보다 많은 당직을 맡으면 특정인에게 부담이 쏠린다.",
    )

    def check(self, ctx: CheckContext) -> Iterator[Finding]:
        limit = ctx.policy.fatigue.max_shifts_per_week
        for name, shifts in shifts_by_person(ctx).items():
            weeks: dict[tuple[int, int], list[Shift]] = defaultdict(list)
            for s in shifts:
                iso = s.start.isocalendar()
                weeks[(iso.year, iso.week)].append(s)
            for (year, week), in_week in sorted(weeks.items()):
                if len(in_week) <= limit:
                    continue
                yield finding(
                    self.info,
                    keys=(name, f"{year}-W{week:02d}"),
                    title=self.info.title,
                    message=(
                        f"{name}님이 {year}년 {week}주차(ISO)에 당직을 {len(in_week)}회 맡습니다"
                        f"(팀 합의 기준: 주 {limit}회 이하)."
                    ),
                    subject_ids=_person_subjects(ctx, name),
                    fix=_fix(name, in_week),
                )


# ---------------------------------------------------------------------------- JT-ONC-004

_STATUS_KO: dict[PersonStatus, str] = {
    PersonStatus.DEPARTED: "퇴사",
    PersonStatus.ON_LEAVE: "장기 부재",
    PersonStatus.UNKNOWN: "확인 불가",
}


def _left_on(entry: RosterEntry) -> date | None:
    try:
        return date.fromisoformat(entry.left_on) if entry.left_on else None
    except ValueError:
        return None


def _upcoming(ctx: CheckContext) -> dict[str, tuple[RosterEntry | None, list[Shift]]]:
    """Shifts not over at ``ctx.now`` (all if ``now`` is None), grouped by canonical name."""
    zone = table_zone(ctx)
    now = _wall(ctx.now, zone) if ctx.now is not None else None
    grouped: dict[str, tuple[RosterEntry | None, set[Shift]]] = {}
    for raw in ctx.oncall:
        start, end = _wall(raw.start, zone), _wall(raw.end, zone)
        if now is not None and end <= now:
            continue
        entry = roster_lookup(ctx.roster, raw.person)
        name = entry.name if entry is not None else raw.person
        grouped.setdefault(name, (entry, set()))[1].add(Shift(start, end, name))
    return {name: (entry, sorted(shifts)) for name, (entry, shifts) in sorted(grouped.items())}


def _shift_problem(
    entry: RosterEntry | None, shifts: list[Shift]
) -> tuple[FindingStatus, str, list[Shift]] | None:
    """Status, Korean reason and affected shifts if the roster says this person cannot take them."""
    if entry is None:
        return FindingStatus.UNKNOWN, "명부에 없는 사람이라 재직·연락 여부를 확인할 수 없습니다", shifts
    if entry.status is not PersonStatus.ACTIVE:
        # 확인 불가 ≠ 정상: an UNKNOWN roster status is reported, but as UNKNOWN, not BROKEN.
        status = FindingStatus.UNKNOWN if entry.status is PersonStatus.UNKNOWN else FindingStatus.BROKEN
        return status, f"명부상 '{_STATUS_KO[entry.status]}' 상태입니다", shifts
    left = _left_on(entry)
    after = [s for s in shifts if left is not None and s.start.date() >= left]
    if left is not None and after:
        return (
            FindingStatus.BROKEN,
            f"명부상 {left.isoformat()} 퇴사 예정인데 그 이후 당직이 잡혀 있습니다",
            after,
        )
    return None


@register
class OnCallPersonUnavailable:
    info = RuleInfo(
        rule_id="JT-ONC-004",
        title="당직표에 퇴사·부재자",
        severity=Severity.HIGH,
        element=None,
        rationale=(
            "당직표에 퇴사자·장기 부재자·명부에 없는 사람이 있으면 그 시간대 호출은 아무도 받지 않는다. "
            "명부로 확인할 수 없는 경우는 정상이 아니라 '확인 불가'로 보고한다."
        ),
    )

    def check(self, ctx: CheckContext) -> Iterator[Finding]:
        upcoming = _upcoming(ctx)
        if upcoming and not ctx.roster:
            yield self._no_roster(ctx, upcoming)
            return
        for name, (entry, shifts) in upcoming.items():
            problem = _shift_problem(entry, shifts)
            if problem is None:
                continue
            status, reason, affected = problem
            yield finding(
                self.info,
                keys=(name,),
                title=self.info.title,
                message=(
                    f"당직표의 {name}님은 {reason}. {affected[0].start.strftime(_TIME_FMT)} 시작 당직 등 "
                    f"{len(affected)}건의 호출이 응답 없이 끝날 수 있습니다."
                ),
                subject_ids=_person_subjects(ctx, name),
                fix=_fix(name, affected),
                status=status,
            )

    def _no_roster(
        self, ctx: CheckContext, upcoming: dict[str, tuple[RosterEntry | None, list[Shift]]]
    ) -> Finding:
        """Without a roster nobody can be vouched for: one UNKNOWN finding instead of one per person."""
        names = list(upcoming)
        return finding(
            self.info,
            keys=("no-roster",),
            title=self.info.title,
            message=(
                f"명부가 없어 당직자 {len(names)}명({', '.join(names)})의 재직·부재 여부를 "
                "확인할 수 없습니다. 명부를 함께 올려 주세요."
            ),
            subject_ids=[nid for n in names for nid in _person_subjects(ctx, n)],
            fix=FixHint(kind=FixKind.ADJUST_ONCALL, params={"people": names}),
            status=FindingStatus.UNKNOWN,
        )
