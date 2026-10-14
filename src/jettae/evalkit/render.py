"""Render a gold spec to a Korean input bundle (runbook + roster + catalog + policy).

Why three styles: an extractor that only works on tidy tables is not ready for the runbooks
people actually write. ``table`` and ``explicit`` are the structured end, ``colloquial`` is the
onboarding-memo end (spoken Korean, nicknames, given-name-only references resolvable through
roster aliases). All three state exactly the same facts — the spec — and nothing else (P2).

Every fact the text states is recorded as a :class:`FactSpan` with exact character offsets, so
the harness can check grounding (원문 근거) without re-parsing, and every service owns one or
more :class:`ServiceRegion` so a mutation can be shown to touch only its target service.

Determinism (P6): phrasing choices come from a ``random.Random`` seeded per *slot*
(seed, org, service, slot), and per person where one sentence names several people. Removing
one fact therefore never shifts the wording of another — not even of a person named in the
same sentence — which keeps the clean and mutated renders identical outside the mutated
statement.
"""

from __future__ import annotations

import csv
import io
import random
import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from jettae.domain import (
    DEFAULT_OBLIGATIONS,
    ActionType,
    CatalogService,
    ChannelMedium,
    EdgeKind,
    IncidentType,
    PersonStatus,
    Policy,
    RosterEntry,
)
from jettae.evalkit.gold import (
    EXEC_TITLES,
    Fact,
    GoldAction,
    GoldObligation,
    GoldOrg,
    GoldService,
    action_key,
    fact,
    trigger_key,
)

Style = Literal["table", "explicit", "colloquial"]
Position = Literal["front", "back"]
Layout = Literal["sections", "org_table"]
STYLES: tuple[Style, ...] = ("table", "explicit", "colloquial")
POSITIONS: tuple[Position, ...] = ("front", "back")

MEDIUM_LABEL_KO: Mapping[ChannelMedium, str] = {
    ChannelMedium.SLACK: "슬랙",
    ChannelMedium.KAKAOTALK: "카카오톡",
    ChannelMedium.TEAMS: "팀즈",
    ChannelMedium.PHONE: "전화",
    ChannelMedium.SMS: "문자",
    ChannelMedium.EMAIL: "이메일",
    ChannelMedium.PAGER: "PagerDuty",
    ChannelMedium.INTERNAL_MESSENGER: "사내 메신저",
    ChannelMedium.OTHER: "기타 연락 수단",
}
STATUS_LABEL_KO: Mapping[PersonStatus, str] = {
    PersonStatus.ACTIVE: "재직",
    PersonStatus.DEPARTED: "퇴사",
    PersonStatus.ON_LEAVE: "휴직",
    PersonStatus.UNKNOWN: "확인불가",
}
ACTION_LABEL_KO: Mapping[ActionType, str] = {
    ActionType.ROLLBACK: "롤백",
    ActionType.FAILOVER: "페일오버",
    ActionType.RESTART: "재시작",
    ActionType.FEATURE_FLAG: "기능 플래그 끄기",
    ActionType.SCALE: "증설",
    ActionType.OTHER: "조치",
}
INCIDENT_LABEL_KO: Mapping[IncidentType, str] = {
    IncidentType.BREACH: "침해사고",
    IncidentType.PRIVACY_LEAK: "개인정보 유출",
    IncidentType.OUTAGE: "서비스 장애",
}
ROSTER_HEADER: tuple[str, ...] = ("이름", "상태", "직급", "팀", "역할", "별칭", "연락수단", "퇴사일")
#: Separator for multi-valued roster cells (역할·별칭·연락수단).
ROSTER_LIST_SEP = ";"

_HONORIFIC_TITLES: frozenset[str] = frozenset({"과장", "차장", "팀장", "CTO", "CISO", "대표"})
#: titles idiomatically used with a bare surname ("김대리", "박팀장님"); "이사원" would misread
_SURNAME_ADDRESS_TITLES: frozenset[str] = frozenset({"주임", "대리", "과장", "차장", "팀장"})


# --------------------------------------------------------------------------- result models


class FactSpan(BaseModel):
    """A fact and the exact text range that states it (``text[start:end] == quote``)."""

    model_config = ConfigDict(frozen=True)

    fact: Fact
    start: int
    end: int
    quote: str


class ServiceRegion(BaseModel):
    """A text range that belongs to one service (a section, a table row or a bullet)."""

    model_config = ConfigDict(frozen=True)

    service: str
    start: int
    end: int


class RenderedCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    style: Style
    position: Position
    focus_service: str | None
    layout: Layout
    text: str
    roster_csv: str
    catalog_yaml: str
    policy_yaml: str
    fact_spans: tuple[FactSpan, ...]
    regions: tuple[ServiceRegion, ...]

    def stated_facts(self) -> frozenset[Fact]:
        return frozenset(s.fact for s in self.fact_spans)

    def region_text(self, service: str) -> str:
        """All text owned by ``service``, regions joined in document order."""
        return "".join(self.text[r.start : r.end] for r in self.regions if r.service == service)

    def text_outside(self, service: str) -> str:
        """The document with ``service``'s regions cut out (for "nothing else changed" checks)."""
        out, pos = [], 0
        for r in sorted(self.regions, key=lambda r: r.start):
            if r.service == service:
                out.append(self.text[pos : r.start])
                pos = r.end
        out.append(self.text[pos:])
        return "".join(out)


# --------------------------------------------------------------------------- Korean helpers


def _final_sound(word: str) -> str:
    """Final consonant of ``word`` as read aloud: ``""`` (vowel), ``"ㄹ"`` or ``"C"`` (other)."""
    stripped = word.rstrip(" )]")
    if not stripped:
        return ""
    ch = stripped[-1]
    if "가" <= ch <= "힣":
        jong = (ord(ch) - ord("가")) % 28
        return "" if jong == 0 else ("ㄹ" if jong == 8 else "C")
    if ch.isdigit():
        return {"0": "C", "1": "ㄹ", "3": "C", "6": "C", "7": "ㄹ", "8": "ㄹ"}.get(ch, "")
    if ch.isupper():  # acronyms are read letter by letter: 엘, 엠, 엔, 알 end in a consonant
        return {"L": "ㄹ", "M": "C", "N": "C", "R": "ㄹ"}.get(ch, "")
    if ch.isalpha():
        return "" if ch in "aeiouy" else ("ㄹ" if ch == "l" else "C")
    return ""


_PARTICLES: Mapping[str, tuple[str, str]] = {
    "이/가": ("이", "가"),
    "은/는": ("은", "는"),
    "을/를": ("을", "를"),
    "과/와": ("과", "와"),
    "이나/나": ("이나", "나"),
    "이에요/예요": ("이에요", "예요"),
    "이고/고": ("이고", "고"),
}


def josa(word: str, particle: str) -> str:
    """``word`` + the particle allomorph that fits its final sound (``"결제"``+``"은/는"``→``"결제는"``)."""
    final = _final_sound(word)
    if particle == "으로/로":
        return word + ("으로" if final == "C" else "로")
    with_final, without_final = _PARTICLES[particle]
    return word + (with_final if final else without_final)


_FIELD = re.compile(r"\{(\w+)(?::([^}]+))?\}")


def _fill(template: str, values: Mapping[str, str]) -> str:
    """Expand ``{name}`` / ``{name:이/가}`` placeholders (particle chosen by final sound)."""

    def sub(m: re.Match[str]) -> str:
        word = values[m.group(1)]
        return josa(word, m.group(2)) if m.group(2) else word

    return _FIELD.sub(sub, template)


def medium_label(medium: ChannelMedium) -> str:
    return MEDIUM_LABEL_KO[medium]


def person_label(org: GoldOrg, name: str) -> str:
    """Formal reference used by the structured styles: ``"김민수 대리"``."""
    return f"{name} {org.person(name).title}"


def person_refs(org: GoldOrg, name: str) -> tuple[str, ...]:
    """Every colloquial way of naming ``name``; each resolves via roster name, alias or 성+직급.

    The given-name form (``"민수 대리"``) is offered only when the roster lists the given name as
    an alias (aliases are unique and unambiguous, see :class:`GoldOrg`), so a hand-built spec
    without aliases never gets an unresolvable reference. The 성+직급 form (``"김대리"``) is
    offered only for titles people actually address that way and only when no one else shares
    surname and title.
    """
    p = org.person(name)
    hon = "님" if p.title in _HONORIFIC_TITLES else ""
    refs = [f"{p.name} {p.title}{hon}"]
    if p.title not in EXEC_TITLES and p.given_name in p.aliases:
        refs.append(f"{p.given_name} {p.title}{hon}")
    refs += [f"{a}님" for a in p.aliases if a != p.given_name]
    same = Counter((q.surname, q.title) for q in org.people)
    if p.title in _SURNAME_ADDRESS_TITLES and same[(p.surname, p.title)] == 1:
        refs.append(f"{p.surname}{p.title}{hon}")
    return tuple(refs)


def _heading(service: GoldService) -> str:
    return f"{service.name} (Tier {service.tier})"


# --------------------------------------------------------------------------- text builder


@dataclass(frozen=True)
class _Piece:
    text: str
    facts: tuple[Fact, ...] = ()


@dataclass(frozen=True)
class _Clause:
    """Pieces with their own facts, plus facts justified by the clause as a whole."""

    pieces: tuple[_Piece, ...]
    facts: tuple[Fact, ...] = ()


def _sentence(text: str, *facts: Fact) -> _Clause:
    return _Clause((_Piece(text),), facts)


class _Writer:
    """Appends text while recording fact spans and service regions by character offset."""

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._pos = 0
        self._spans: set[tuple[Fact, int, int]] = set()
        self._regions: list[tuple[str, int, int]] = []

    def write(self, text: str, facts: Iterable[Fact] = ()) -> None:
        start = self._pos
        self._parts.append(text)
        self._pos += len(text)
        self._spans.update((f, start, self._pos) for f in facts)

    def clause(self, clause: _Clause, *, lead: str = "") -> None:
        """Write ``lead`` + the clause; clause-level facts span the lead too (e.g. "담당: …")."""
        start = self._pos
        self.write(lead)
        for piece in clause.pieces:
            self.write(piece.text, piece.facts)
        self._spans.update((f, start, self._pos) for f in clause.facts)

    def cell(self, clause: _Clause | None) -> None:
        self.write(" | ")
        if clause is None:
            self.write("-")
        else:
            self.clause(clause)

    @contextmanager
    def region(self, service: str) -> Iterator[None]:
        start = self._pos
        yield
        self._regions.append((service, start, self._pos))

    def finish(self) -> tuple[str, tuple[FactSpan, ...], tuple[ServiceRegion, ...]]:
        text = "".join(self._parts)
        spans = tuple(
            FactSpan(fact=f, start=a, end=b, quote=text[a:b])
            for f, a, b in sorted(self._spans, key=lambda s: (s[1], s[2], s[0]))
        )
        regions = tuple(
            ServiceRegion(service=s, start=a, end=b) for s, a, b in sorted(self._regions, key=lambda r: r[1])
        )
        return text, spans, regions


# --------------------------------------------------------------------------- structured clauses


def _systems_clause(s: GoldService) -> _Clause:
    pieces: list[_Piece] = []
    for i, sys in enumerate(s.systems):
        if i:
            pieces.append(_Piece(", "))
        pieces.append(_Piece(sys, (fact(EdgeKind.DEPENDS_ON, s.name, sys),)))
    pieces += [
        _Piece(" ("),
        _Piece(s.location, (fact(EdgeKind.RUNS_IN, s.name, s.location),)),
        _Piece(")"),
    ]
    return _Clause(tuple(pieces), tuple(fact(EdgeKind.RUNS_IN, sys, s.location) for sys in s.systems))


def _owner_clause(org: GoldOrg, s: GoldService, owner: str) -> _Clause:
    return _Clause((_Piece(person_label(org, owner)),), (fact(EdgeKind.OWNED_BY, s.name, owner),))


def _primary_piece(s: GoldService, suffix: str = "") -> _Piece:
    p = s.fallback.primary_channel
    return _Piece(medium_label(p) + suffix, (fact(EdgeKind.REACHABLE_VIA, s.first_responder, p.value),))


def _fallback_piece(s: GoldService, fb: ChannelMedium, suffix: str = "") -> _Piece:
    p = s.fallback.primary_channel
    facts = (
        fact(EdgeKind.REACHABLE_VIA, s.first_responder, fb.value),
        fact(EdgeKind.FALLBACK_TO, p.value, fb.value),
    )
    return _Piece(medium_label(fb) + suffix, facts)


def _contact_clause(s: GoldService) -> _Clause:
    fb = s.fallback.fallback_channel
    if fb is None:
        return _Clause((_primary_piece(s),))
    return _Clause((_primary_piece(s, "(1순위)"), _Piece(", "), _fallback_piece(s, fb, "(대체)")))


def _detection_clause(org: GoldOrg, s: GoldService) -> _Clause | None:
    if s.trigger is None:
        return None
    tk = trigger_key(s.name)
    t = s.trigger
    when = f"{t.delay_minutes}분 지속 시 알람" if t.delay_minutes else "시 즉시 알람"
    notify = fact(EdgeKind.NOTIFIES, tk, s.first_responder, channel=s.notify_channel.value)
    return _Clause(
        (
            _Piece(f"{t.condition} {when}", (fact(EdgeKind.DETECTED_BY, s.name, tk),)),
            _Piece(" → "),
            _Piece(
                f"{person_label(org, s.first_responder)} 호출({medium_label(s.notify_channel)})", (notify,)
            ),
        )
    )


def _escalation_clause(org: GoldOrg, s: GoldService) -> _Clause | None:
    pieces: list[_Piece] = []
    for i, (src, dst, after) in enumerate(s.escalation_edges()):
        if i:
            pieces.append(_Piece(", 이후 "))
        hop = fact(EdgeKind.ESCALATES_TO, src, dst, after_minutes=after)
        pieces.append(_Piece(f"{after}분 미응답 시 {person_label(org, dst)}", (hop,)))
    return _Clause(tuple(pieces)) if pieces else None


def _mitigation_fact(s: GoldService, a: GoldAction) -> Fact:
    return fact(EdgeKind.MITIGATED_BY, s.name, action_key(s.name, a.type), action_type=a.type.value)


def _approval_facts(s: GoldService, a: GoldAction) -> tuple[Fact, ...]:
    ak = action_key(s.name, a.type)
    return tuple(fact(EdgeKind.APPROVED_BY, ak, n) for n in a.approvers)


def _executor_fact(s: GoldService, a: GoldAction, executor: str) -> Fact:
    return fact(EdgeKind.EXECUTED_BY, action_key(s.name, a.type), executor)


def _action_clause(org: GoldOrg, s: GoldService, a: GoldAction) -> _Clause:
    pieces: list[_Piece] = []
    if a.approvers:
        names = " 또는 ".join(person_label(org, n) for n in a.approvers)
        pieces += [_Piece(f"{names} 승인", _approval_facts(s, a)), _Piece(" 후 ")]
    elif a.requires_approval:
        pieces.append(_Piece("승인 후 "))
    if a.executor is not None:
        executor = josa(person_label(org, a.executor), "이/가")
        pieces.append(_Piece(f"{executor} 실행", (_executor_fact(s, a, a.executor),)))
    else:
        pieces.append(_Piece("실행"))
    pieces.append(_Piece(f" (약 {a.duration_minutes}분 소요)"))
    return _Clause(tuple(pieces), (_mitigation_fact(s, a),))


def _must_report(s: GoldService, ob: GoldObligation) -> Fact:
    return fact(
        EdgeKind.MUST_REPORT,
        s.name,
        ob.recipient,
        deadline_minutes=ob.deadline_hours * 60,
        incident_type=ob.incident_type.value,
    )


def _obligation_clause(org: GoldOrg, s: GoldService, ob: GoldObligation) -> _Clause:
    incident = josa(INCIDENT_LABEL_KO[ob.incident_type], "은/는")
    duty = f"{incident} {ob.deadline_hours}시간 이내 {ob.recipient} 신고"
    pieces = [_Piece(duty, (_must_report(s, ob),))]
    if ob.reporter is not None:
        reporter = fact(EdgeKind.REPORTED_BY, ob.recipient, ob.reporter)
        pieces.append(_Piece(f"(담당: {person_label(org, ob.reporter)})", (reporter,)))
    return _Clause(tuple(pieces))


def _line_items(org: GoldOrg, s: GoldService) -> list[tuple[str, _Clause]]:
    """(label, clause) lines of one service for the explicit and per-service table styles."""
    items: list[tuple[str, _Clause | None]] = [
        ("구성", _systems_clause(s)),
        ("담당", _owner_clause(org, s, s.owner) if s.owner else None),
        ("연락", _contact_clause(s)),
        ("감지", _detection_clause(org, s)),
        ("상향", _escalation_clause(org, s)),
    ]
    if s.action is not None:
        items.append((ACTION_LABEL_KO[s.action.type], _action_clause(org, s, s.action)))
    items += [("신고", _obligation_clause(org, s, ob)) for ob in s.obligations]
    return [(label, clause) for label, clause in items if clause is not None]


# --------------------------------------------------------------------------- structured layouts


def _intro(org: GoldOrg) -> str:
    return (
        f"{org.company}의 {org.theme} 서비스 장애 대응 절차를 정리한 문서입니다. "
        "사람의 재직 여부와 연락처는 인사 명부를 기준으로 확인합니다."
    )


def _render_explicit(w: _Writer, org: GoldOrg, order: Sequence[GoldService]) -> None:
    w.write(f"# {org.company} 장애 대응 런북\n\n{_intro(org)}\n\n")
    for s in order:
        with w.region(s.name):
            w.write(f"## {_heading(s)}\n\n")
            for label, clause in _line_items(org, s):
                w.write("- ")
                w.clause(clause, lead=f"{label}: ")
                w.write("\n")
        w.write("\n")


def _render_service_tables(w: _Writer, org: GoldOrg, order: Sequence[GoldService]) -> None:
    w.write(f"# {org.company} 장애 대응 런북\n\n{_intro(org)}\n\n")
    for s in order:
        with w.region(s.name):
            w.write(f"## {_heading(s)}\n\n| 항목 | 내용 |\n|---|---|\n")
            for label, clause in _line_items(org, s):
                w.write("| ")
                w.clause(clause, lead=f"{label} | ")
                w.write(" |\n")
        w.write("\n")


_ORG_TABLE_HEADER = (
    "| 서비스 | 담당 | 연락 | 대체 채널 | 상향 | 승인 | 롤백 |\n|---|---|---|---|---|---|---|\n"
)


def _approver_cell(org: GoldOrg, s: GoldService) -> _Clause | None:
    a = s.action
    if a is None or not a.approvers:
        return None
    names = ", ".join(person_label(org, n) for n in a.approvers)
    return _Clause((_Piece(names, _approval_facts(s, a)),))


def _rollback_cell(org: GoldOrg, s: GoldService) -> _Clause | None:
    a = s.action
    if a is None:
        return None
    act = ACTION_LABEL_KO[a.type]
    if a.executor is None:
        work = _Piece(f"{act} 실행")
    else:
        executor = josa(person_label(org, a.executor), "이/가")
        work = _Piece(f"{executor} {act} 실행", (_executor_fact(s, a, a.executor),))
    return _Clause((work, _Piece(f" (약 {a.duration_minutes}분)")), (_mitigation_fact(s, a),))


def _org_table_row(w: _Writer, org: GoldOrg, s: GoldService) -> None:
    fb = s.fallback.fallback_channel
    w.write(f"| {_heading(s)}")
    w.cell(_owner_clause(org, s, s.owner) if s.owner else None)
    w.cell(_Clause((_primary_piece(s),)))
    w.cell(_Clause((_fallback_piece(s, fb),)) if fb is not None else None)
    w.cell(_escalation_clause(org, s))
    w.cell(_approver_cell(org, s))
    w.cell(_rollback_cell(org, s))
    w.write(" |\n")


def _bullet_section(w: _Writer, title: str, bullets: Sequence[tuple[GoldService, _Clause]]) -> None:
    if not bullets:
        return
    w.write(f"## {title}\n\n")
    for s, clause in bullets:
        with w.region(s.name):
            w.write("- ")
            w.clause(clause, lead=f"{s.name}: ")
            w.write("\n")
    w.write("\n")


def _render_org_table(w: _Writer, org: GoldOrg, order: Sequence[GoldService]) -> None:
    w.write(f"# {org.company} 장애 대응 런북\n\n{_intro(org)}\n\n## 서비스별 대응 요약\n\n")
    w.write(_ORG_TABLE_HEADER)
    for s in order:
        with w.region(s.name):
            _org_table_row(w, org, s)
    w.write("\n")
    _bullet_section(w, "구성", [(s, _systems_clause(s)) for s in order])
    detections = [(s, c) for s in order if (c := _detection_clause(org, s)) is not None]
    _bullet_section(w, "감지 기준", detections)
    reports = [(s, _obligation_clause(org, s, ob)) for s in order for ob in s.obligations]
    _bullet_section(w, "신고 의무", reports)


# --------------------------------------------------------------------------- colloquial style

#: openers depend on the section's place in the document ("…부터" only fits the first one); the
#: order is fixed per (org, position, focus), so clean and mutated renders still share them
_OPENERS_FIRST: tuple[str, ...] = (
    "{svc} 쪽 얘기부터 할게요.",
    "먼저 {svc:이에요/예요}.",
    "{svc}부터 볼게요.",
)
_OPENERS_NEXT: tuple[str, ...] = (
    "다음은 {svc:이에요/예요}.",
    "{svc:은/는} 이렇게 챙기시면 돼요.",
    "이어서 {svc} 쪽이에요.",
)
_SYSTEMS: tuple[str, ...] = (
    "{svc:은/는} {systems} 위에서 돌아가고, 위치는 {loc:이에요/예요}.",
    "참고로 {svc:은/는} {loc}에 있는 {systems} 구성이에요.",
    "{svc} 구성은 {systems} 쪽이고 전부 {loc}에 올라가 있어요.",
)
_OWNER: tuple[str, ...] = (
    "{svc:은/는} {owner:이/가} 담당이에요.",
    "{svc} 담당은 {owner:이에요/예요}.",
    "{svc} 쪽은 {owner:이/가} 맡고 있어요.",
)
_DETECTION: tuple[str, ...] = (
    "모니터링에서 {cond} 상태가 {when} 알람이 울리고, {fr}한테 {ch:으로/로} 호출이 가요.",
    "{cond} 상황이 {when} 알람이 떠요. 그러면 {fr:이/가} {ch:으로/로} 호출을 받아요.",
    "알람은 {cond} 상태가 {when} 울리고, 호출은 {fr}한테 {ch:으로/로} 가요.",
)
_CONTACT_WITH_FALLBACK: tuple[str, ...] = (
    "{fr}한테는 평소엔 {p:으로/로} 연락하시고, 안 되면 {fb:으로/로} 하시면 돼요.",
    "{fr} 연락은 {p:이/가} 1순위고, {p:이/가} 안 되면 {fb:으로/로} 돌리세요.",
    "급할 땐 {fr}한테 {p} 먼저 주시고, 답이 안 오면 {fb:으로/로} 다시 연락해 주세요.",
)
_CONTACT_SINGLE: tuple[str, ...] = (
    "{fr}한테는 {p:으로/로}만 연락이 돼요.",
    "{fr} 연락 수단은 {p} 하나예요.",
    "{fr}한테 연락하실 땐 {p:으로/로} 하시면 돼요.",
)
#: first hop; the last variant also names the target's out-of-band medium (a roster fact)
_ESCALATION_FIRST: tuple[str, ...] = (
    "{after}분 지나도 답이 없으면 {target}께 올려주세요.",
    "{src:이/가} {after}분 안에 응답이 없으면 {target}한테 에스컬레이션하시고요.",
    "{after}분 넘게 무응답이면 {target}께 {ch:으로/로} 바로 연락하세요.",
)
_ESCALATION_NEXT: tuple[str, ...] = (
    "{src}도 {after}분 동안 답이 없으면 {target}께 올리시면 돼요.",
    "{src:이/가} {after}분 안에 못 받으시면 그다음은 {target:이에요/예요}.",
    "거기서도 {after}분 넘게 조용하면 {target}한테 바로 연락하세요.",
)
_REPORT_WITH_REPORTER: tuple[str, ...] = (
    "{inc:이/가} 의심되면 {hours}시간 안에 {rcpt}에 신고해야 하고, 신고는 {rep:이/가} 맡아요.",
    "{inc:은/는} {rep:이/가} {hours}시간 이내로 {rcpt} 신고까지 챙기셔야 해요.",
    "{rcpt} 신고는 {inc} 기준 {hours}시간 안에 해야 하는데, {rep} 담당이에요.",
)
_REPORT_NO_REPORTER: tuple[str, ...] = (
    "{inc:이/가} 의심되면 {hours}시간 안에 {rcpt}에 신고해야 해요.",
    "{inc:은/는} {hours}시간 이내로 {rcpt} 신고까지 챙기셔야 해요.",
    "{rcpt} 신고는 {inc} 기준 {hours}시간 안에 해야 해요.",
)

#: approval phrase -> (approver given, approval required but nobody named, no approval needed)
_ApprovalForms = tuple[Callable[[str], str], str, str]
#: executor phrase -> (executor given, executor not stated)
_ExecutorForms = tuple[Callable[[str], str], str]
_ACTION_VARIANTS: tuple[tuple[str, _ApprovalForms, _ExecutorForms, str], ...] = (
    (
        "{act:은/는} ",
        (lambda apr: f"{apr} 승인을 받고 ", "승인을 받고 ", ""),
        (lambda ex: f"{josa(ex, '이/가')} 진행하는데", "진행하는데"),
        ", 보통 {d}분 정도 걸려요.",
    ),
    (
        "{act:이/가} 필요하면 ",
        (lambda apr: f"{apr}께 승인받은 다음에 ", "승인부터 받은 다음에 ", ""),
        (lambda ex: f"{josa(ex, '이/가')} 돌리면 되고요", "돌리면 되고요"),
        ", {d}분쯤 걸립니다.",
    ),
    (
        "{act} ",
        (lambda apr: f"판단은 {josa(apr, '이/가')} 하시고 ", "승인 절차를 거친 뒤 ", ""),
        (lambda ex: f"실제 작업은 {josa(ex, '이/가')} 해요.", "작업을 진행해요."),
        " {d}분 안팎이에요.",
    ),
)

#: number of phrasing variants per colloquial slot (published so tests can assert ≥ 3)
COLLOQUIAL_VARIANTS: Mapping[str, int] = {
    "opener_first": len(_OPENERS_FIRST),
    "opener_next": len(_OPENERS_NEXT),
    "systems": len(_SYSTEMS),
    "owner": len(_OWNER),
    "detection": len(_DETECTION),
    "contact_with_fallback": len(_CONTACT_WITH_FALLBACK),
    "contact_single": len(_CONTACT_SINGLE),
    "escalation_first": len(_ESCALATION_FIRST),
    "escalation_next": len(_ESCALATION_NEXT),
    "action": len(_ACTION_VARIANTS),
    "report_with_reporter": len(_REPORT_WITH_REPORTER),
    "report_no_reporter": len(_REPORT_NO_REPORTER),
}


class _Colloquial:
    """Builds the spoken-style sentences of one service; one seeded RNG per slot."""

    def __init__(self, org: GoldOrg, service: GoldService, seed: int, *, first: bool) -> None:
        self.org = org
        self.s = service
        self.seed = seed
        self.first = first

    def rng(self, slot: str) -> random.Random:
        return random.Random(f"{self.seed}|{self.org.org_id}|{self.s.name}|{slot}")

    def ref(self, rng: random.Random, name: str) -> str:
        return rng.choice(person_refs(self.org, name))

    def paragraphs(self) -> list[list[_Clause]]:
        first = [self.opener(), self.systems(), *self.owner(), *self.detection(), self.contact()]
        second = [*self.escalation(), *self.action(), *self.reports()]
        return [first, second] if second else [first]

    def opener(self) -> _Clause:
        templates = _OPENERS_FIRST if self.first else _OPENERS_NEXT
        return _sentence(_fill(self.rng("opener").choice(templates), {"svc": self.s.name}))

    def systems(self) -> _Clause:
        s = self.s
        text = _fill(
            self.rng("systems").choice(_SYSTEMS),
            {"svc": s.name, "systems": ", ".join(s.systems), "loc": s.location},
        )
        clause = _systems_clause(s)
        return _sentence(text, *(f for p in clause.pieces for f in p.facts), *clause.facts)

    def owner(self) -> list[_Clause]:
        if self.s.owner is None:
            return []
        rng = self.rng("owner")
        template = rng.choice(_OWNER)
        text = _fill(template, {"svc": self.s.name, "owner": self.ref(rng, self.s.owner)})
        return [_sentence(text, fact(EdgeKind.OWNED_BY, self.s.name, self.s.owner))]

    def detection(self) -> list[_Clause]:
        s, t = self.s, self.s.trigger
        if t is None:
            return []
        rng = self.rng("detection")
        template = rng.choice(_DETECTION)
        values = {
            "cond": t.condition,
            "when": f"{t.delay_minutes}분 넘게 이어지면" if t.delay_minutes else "감지되면 바로",
            "fr": self.ref(rng, s.first_responder),
            "ch": medium_label(s.notify_channel),
        }
        tk = trigger_key(s.name)
        facts = (
            fact(EdgeKind.DETECTED_BY, s.name, tk),
            fact(EdgeKind.NOTIFIES, tk, s.first_responder, channel=s.notify_channel.value),
        )
        return [_sentence(_fill(template, values), *facts)]

    def contact(self) -> _Clause:
        s = self.s
        rng = self.rng("contact")
        fb = s.fallback.fallback_channel
        templates = _CONTACT_SINGLE if fb is None else _CONTACT_WITH_FALLBACK
        template = rng.choice(templates)
        values = {"fr": self.ref(rng, s.first_responder), "p": medium_label(s.fallback.primary_channel)}
        clause = _contact_clause(s)
        facts = [f for p in clause.pieces for f in p.facts]
        if fb is not None:
            values["fb"] = medium_label(fb)
        return _sentence(_fill(template, values), *facts)

    def escalation(self) -> list[_Clause]:
        out: list[_Clause] = []
        for i, (src, dst, after) in enumerate(self.s.escalation_edges()):
            rng = self.rng(f"escalation-{i}")
            templates = _ESCALATION_FIRST if i == 0 else _ESCALATION_NEXT
            index = rng.randrange(len(templates))
            values = {"src": self.ref(rng, src), "target": self.ref(rng, dst), "after": str(after)}
            facts = [fact(EdgeKind.ESCALATES_TO, src, dst, after_minutes=after)]
            if i == 0 and index == len(templates) - 1:
                medium = self._out_of_band(dst)
                values["ch"] = medium_label(medium)
                facts.append(fact(EdgeKind.REACHABLE_VIA, dst, medium.value))
            out.append(_sentence(_fill(templates[index], values), *facts))
        return out

    def _out_of_band(self, name: str) -> ChannelMedium:
        channels = self.org.person(name).channels
        return next((m for m in channels if m is not self.org.chat_medium), channels[0])

    def action(self) -> list[_Clause]:
        s, a = self.s, self.s.action
        if a is None:
            return []
        head, approval_forms, executor_forms, tail = self.rng("action").choice(_ACTION_VARIANTS)
        with_approver, approval_only, no_approval = approval_forms
        if a.approvers:
            # one RNG per person: deleting the approvers must not reword the executor (or each other)
            refs = [self.ref(self.rng(f"action-approver|{n}"), n) for n in a.approvers]
            joined = "".join(josa(r, "이나/나") + " " for r in refs[:-1]) + refs[-1]
            approval = with_approver(joined)
        else:
            approval = approval_only if a.requires_approval else no_approval
        with_executor, no_executor = executor_forms
        executor = a.executor
        work = with_executor(self.ref(self.rng("action-executor"), executor)) if executor else no_executor
        act = ACTION_LABEL_KO[a.type]
        text = _fill(head, {"act": act}) + approval + work + _fill(tail, {"d": str(a.duration_minutes)})
        facts = [_mitigation_fact(s, a), *_approval_facts(s, a)]
        if a.executor is not None:
            facts.append(_executor_fact(s, a, a.executor))
        return [_sentence(text, *facts)]

    def reports(self) -> list[_Clause]:
        out: list[_Clause] = []
        for ob in self.s.obligations:
            rng = self.rng(f"report-{ob.incident_type.value}")
            templates = _REPORT_NO_REPORTER if ob.reporter is None else _REPORT_WITH_REPORTER
            template = rng.choice(templates)
            values = {
                "inc": INCIDENT_LABEL_KO[ob.incident_type],
                "hours": str(ob.deadline_hours),
                "rcpt": ob.recipient,
            }
            facts = [_must_report(self.s, ob)]
            if ob.reporter is not None:
                values["rep"] = self.ref(rng, ob.reporter)
                facts.append(fact(EdgeKind.REPORTED_BY, ob.recipient, ob.reporter))
            out.append(_sentence(_fill(template, values), *facts))
        return out


_COLLOQUIAL_INTRO = (
    "온보딩 때 말로 설명하던 내용을 그대로 옮겨 적은 메모예요. "
    "서비스별로 누구를 찾으면 되는지 위주로 정리했어요."
)


def _render_colloquial(w: _Writer, org: GoldOrg, order: Sequence[GoldService], seed: int) -> None:
    w.write(f"# {org.company} 장애 대응 메모\n\n{_COLLOQUIAL_INTRO}\n\n")
    for index, s in enumerate(order):
        with w.region(s.name):
            w.write(f"## {_heading(s)}\n\n")
            for p_index, paragraph in enumerate(_Colloquial(org, s, seed, first=index == 0).paragraphs()):
                if p_index:
                    w.write("\n\n")
                for c_index, sentence in enumerate(paragraph):
                    if c_index:
                        w.write(" ")
                    w.clause(sentence)
            w.write("\n")
        w.write("\n")


# --------------------------------------------------------------------------- structured inputs


def roster_entries(org: GoldOrg) -> tuple[RosterEntry, ...]:
    """The roster (명부) as trusted structured facts, sorted by name."""
    entries = (
        RosterEntry(
            name=p.name,
            status=p.status,
            title=p.title,
            team=p.team,
            roles=org.roles_of(p.name),
            aliases=p.aliases,
            channels=p.channels,
            left_on=p.left_on,
        )
        for p in org.people
    )
    return tuple(sorted(entries, key=lambda e: e.name))


def roster_csv(org: GoldOrg) -> str:
    """Korean HR-export style CSV (headers :data:`ROSTER_HEADER`, ``;`` inside multi-value cells)."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(ROSTER_HEADER)
    for e in roster_entries(org):
        writer.writerow(
            [
                e.name,
                STATUS_LABEL_KO[e.status],
                e.title or "",
                e.team or "",
                ROSTER_LIST_SEP.join(e.roles),
                ROSTER_LIST_SEP.join(e.aliases),
                ROSTER_LIST_SEP.join(MEDIUM_LABEL_KO[m] for m in e.channels),
                e.left_on or "",
            ]
        )
    return buf.getvalue()


def catalog_entries(org: GoldOrg) -> tuple[CatalogService, ...]:
    """Services and their systems as a catalog would list them (no owners: ownership is the
    runbook's claim, so DELETE_OWNER must not be undone by the catalog)."""
    entries = [
        CatalogService(name=s.name, kind="service", tier=s.tier, depends_on=s.systems, location=s.location)
        for s in org.services
    ]
    entries += [
        CatalogService(name=sys, kind="system", location=s.location)
        for s in org.services
        for sys in s.systems
    ]
    return tuple(sorted(entries, key=lambda e: (e.kind, e.name)))


def catalog_yaml(org: GoldOrg) -> str:
    """Native catalog format (``services:`` list of :class:`CatalogService` fields)."""
    return _dump_catalog(catalog_entries(org))


def policy_for(org: GoldOrg) -> Policy:
    """Policy with the org's declared scope: regulated services and which duties apply."""
    return _policy(org.handles_personal_info, _regulated(org))


def policy_yaml(org: GoldOrg) -> str:
    """Native policy format (the :class:`Policy` fields)."""
    return _dump_policy(org.handles_personal_info, _regulated(org))


def _regulated(org: GoldOrg) -> tuple[str, ...]:
    return tuple(sorted(s.name for s in org.services if s.regulated))


def _policy(handles_personal_info: bool, regulated: tuple[str, ...]) -> Policy:
    obligations = tuple(
        rule.model_copy(
            update={"applies": rule.incident_type is not IncidentType.PRIVACY_LEAK or handles_personal_info}
        )
        for rule in DEFAULT_OBLIGATIONS
    )
    return Policy(obligations=obligations, regulated_services=regulated)


# PyYAML's pure-Python emitter dominates corpus build time, and every mutation of an org shares
# its catalog and policy. These dumps are pure functions of their (hashable) content, so they
# are memoised; the libyaml emitter is deliberately not used because its output could differ
# between machines and change the sealed hashes.


@lru_cache(maxsize=256)
def _dump_catalog(entries: tuple[CatalogService, ...]) -> str:
    services = [e.model_dump(mode="json", exclude_none=True) for e in entries]
    return yaml.safe_dump({"services": services}, allow_unicode=True, sort_keys=True)


@lru_cache(maxsize=256)
def _dump_policy(handles_personal_info: bool, regulated: tuple[str, ...]) -> str:
    payload = _policy(handles_personal_info, regulated).model_dump(mode="json")
    return yaml.safe_dump(payload, allow_unicode=True, sort_keys=True)


# --------------------------------------------------------------------------- entry point


def _ordered(org: GoldOrg, position: Position, focus_service: str | None) -> list[GoldService]:
    if focus_service is None:
        return list(org.services)
    try:
        focus = org.service(focus_service)
    except KeyError:
        raise ValueError(f"{org.org_id}에 없는 서비스입니다: {focus_service}") from None
    others = [s for s in org.services if s.name != focus_service]
    return [focus, *others] if position == "front" else [*others, focus]


def table_layout(org: GoldOrg, seed: int) -> Layout:
    """Per-service tables or one org-wide table — fixed per (seed, org) so clean and mutated match."""
    layouts: tuple[Layout, ...] = ("sections", "org_table")
    return random.Random(f"{seed}|{org.org_id}|layout").choice(layouts)


def render(
    org: GoldOrg,
    *,
    style: Style,
    position: Position,
    focus_service: str | None,
    seed: int,
) -> RenderedCase:
    """Render ``org`` as a runbook in ``style``; ``focus_service`` goes first/last per ``position``."""
    if position not in POSITIONS:
        raise ValueError(f"알 수 없는 배치입니다: {position} (front 또는 back)")
    order = _ordered(org, position, focus_service)
    w = _Writer()
    layout: Layout = "sections"
    if style == "explicit":
        _render_explicit(w, org, order)
    elif style == "colloquial":
        _render_colloquial(w, org, order, seed)
    elif style == "table":
        layout = table_layout(org, seed)
        if layout == "org_table":
            _render_org_table(w, org, order)
        else:
            _render_service_tables(w, org, order)
    else:
        raise ValueError(f"알 수 없는 문체입니다: {style}")
    text, spans, regions = w.finish()
    return RenderedCase(
        style=style,
        position=position,
        focus_service=focus_service,
        layout=layout,
        text=text,
        roster_csv=roster_csv(org),
        catalog_yaml=catalog_yaml(org),
        policy_yaml=policy_yaml(org),
        fact_spans=spans,
        regions=regions,
    )
