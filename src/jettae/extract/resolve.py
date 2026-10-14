"""Entity and value resolution: document surfaces -> canonical identities and typed values.

Runbooks name things the way people talk ("김대리", "박 팀장님", "결제서비스", "카톡"). The
graph must instead name *one* roster person, *one* catalog service, *one* channel — otherwise
two mentions of the same person become two nodes and a fixed rule sees a broken path that is
not there (or misses one that is).

Trust model: the roster and catalog are trusted facts; the surface is untrusted text. So the
resolver only ever *selects* among roster/catalog entries and never invents an identity. When
the surface fits several people (동명이인, "김대리" with two 김 대리s) the answer is
``AMBIGUOUS`` and the assembler holds the element (P3: 확인 불가 ≠ 정상) instead of guessing.

Value helpers (durations, booleans, tiers, channel media) live here too because the
assembler (attribute coercion) and the back-translation check (slot comparison) must agree on
exactly the same normalisation.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

from rapidfuzz import fuzz

from jettae.domain import CatalogService, ChannelMedium, NodeKind, RosterEntry, node_id

# --------------------------------------------------------------------------- text keys

_HONORIFICS = ("님", "씨")
_PAREN = re.compile(r"\([^)]*\)|\[[^\]]*\]")
_SERVICE_NOISE = ("서비스", "시스템")

#: Rank titles (직급). They identify a person together with a surname but are *not* roles.
RANK_TITLES: frozenset[str] = frozenset(
    {
        "사원", "주임", "대리", "과장", "차장", "부장", "이사", "상무", "전무", "선임", "책임",
        "수석", "연구원", "매니저", "프로", "주무관",
    }
)  # fmt: skip

#: Position words that denote a responsibility (역할) and become ROLE nodes.
ROLE_WORDS: frozenset[str] = frozenset(
    {
        "팀장", "파트장", "실장", "본부장", "센터장", "그룹장", "셀장", "부문장", "리드", "테크리드",
        "대표", "대표이사", "사장", "부사장", "CEO", "CTO", "CIO", "CISO", "CPO", "CFO", "COO",
        "개인정보 보호책임자", "정보보호 최고책임자",
    }
)  # fmt: skip

#: Two-syllable Korean surnames; everything else is treated as a one-syllable surname.
_TWO_CHAR_SURNAMES: frozenset[str] = frozenset(
    {"남궁", "황보", "제갈", "선우", "독고", "사공", "서문", "동방", "을지", "어금", "망절"}
)


def clean_surface(surface: str) -> str:
    """NFC + collapse internal whitespace. The display form used for nodes named by a surface."""
    return " ".join(unicodedata.normalize("NFC", surface).split())


def compact(text: str) -> str:
    """Comparison key: NFC, no whitespace, casefolded (Latin only is affected)."""
    return "".join(unicodedata.normalize("NFC", text).split()).casefold()


def strip_honorific(text: str) -> str:
    """Drop one trailing honorific (님/씨) — "박팀장님" and "박팀장" name the same person."""
    for h in _HONORIFICS:
        if text.endswith(h) and len(text) > len(h):
            return text[: -len(h)]
    return text


def display_name(surface: str) -> str:
    """Display form for a person/role surface: cleaned, honorific and parentheticals removed."""
    text = clean_surface(_PAREN.sub("", surface))
    stripped = strip_honorific(text).rstrip()
    return stripped or text


#: Words that fill a slot without naming anyone or anything ("담당: 미정", "백업 채널: TBD").
_PLACEHOLDERS: frozenset[str] = frozenset(
    {
        "미정", "미지정", "미배정", "미확인", "공석", "공란", "없음", "해당없음", "추후", "추후지정",
        "추후결정", "지정필요", "확인필요", "tbd", "tba", "n/a", "none",
    }
)  # fmt: skip
_BRACKETS = "()[]{}<>"


def names_nothing(surface: str) -> bool:
    """True for a blank, punctuation-only or placeholder surface ("", "-", "(미정)", "TBD").

    Such a slot means the runbook leaves the role unfilled. Turning it into a node would give
    fixed rules a phantom owner/channel ("미정") where they must see a missing one (P3).
    A parenthetical is a note about the slot, not its filler: "미정 (추후 지정)" names nobody,
    while "(미정)" alone is judged by what the brackets hold.
    """
    outside = compact(_PAREN.sub("", surface)).strip(_BRACKETS)
    bare = outside or compact(surface).strip(_BRACKETS)
    return bare in _PLACEHOLDERS or not any(ch.isalnum() for ch in bare)


_NON_WORD = re.compile(r"[\W_]+")


def name_key(text: str) -> str:
    """Comparison key for names of people, titles, roles, teams and places: compact, no punctuation.

    "김민수.", "Kim-Minsu" and "김민수" must meet in one roster identity: node ids drop
    punctuation (:func:`~jettae.domain.graph.slug`), so a key that kept it would let an
    unresolved surface silently share a roster person's node id.
    """
    return _NON_WORD.sub("", compact(text))


def split_korean_name(name: str) -> tuple[str, str] | None:
    """``"김민수" -> ("김", "민수")``; ``None`` for non-Hangul or one-syllable names."""
    key = name_key(name)
    if len(key) < 2 or not all("가" <= ch <= "힣" for ch in key):
        return None
    if len(key) >= 3 and key[:2] in _TWO_CHAR_SURNAMES:
        return key[:2], key[2:]
    return key[0], key[1:]


def service_key(surface: str) -> str:
    """Key for service/system names: compact, without the generic words 서비스/시스템."""
    key = compact(surface)
    for noise in _SERVICE_NOISE:
        key = key.replace(noise, "")
    return key or compact(surface)


# --------------------------------------------------------------------------- channels

#: Surface keywords per medium (compact, casefolded). Longest keyword wins on overlap.
_MEDIUM_KEYWORDS: dict[str, ChannelMedium] = {
    "슬랙": ChannelMedium.SLACK,
    "slack": ChannelMedium.SLACK,
    "카카오톡": ChannelMedium.KAKAOTALK,
    "카톡": ChannelMedium.KAKAOTALK,
    "kakaotalk": ChannelMedium.KAKAOTALK,
    "kakao": ChannelMedium.KAKAOTALK,
    "팀즈": ChannelMedium.TEAMS,
    "msteams": ChannelMedium.TEAMS,
    "teams": ChannelMedium.TEAMS,
    "전화": ChannelMedium.PHONE,
    "휴대폰": ChannelMedium.PHONE,
    "핸드폰": ChannelMedium.PHONE,
    "휴대전화": ChannelMedium.PHONE,
    "유선": ChannelMedium.PHONE,
    "통화": ChannelMedium.PHONE,
    "phone": ChannelMedium.PHONE,
    "문자": ChannelMedium.SMS,
    "sms": ChannelMedium.SMS,
    "이메일": ChannelMedium.EMAIL,
    "메일": ChannelMedium.EMAIL,
    "email": ChannelMedium.EMAIL,
    "e-mail": ChannelMedium.EMAIL,
    "페이저듀티": ChannelMedium.PAGER,
    "pagerduty": ChannelMedium.PAGER,
    "옵스지니": ChannelMedium.PAGER,
    "opsgenie": ChannelMedium.PAGER,
    "incident.io": ChannelMedium.PAGER,
    "호출기": ChannelMedium.PAGER,
    "페이저": ChannelMedium.PAGER,
    "pager": ChannelMedium.PAGER,
    "사내메신저": ChannelMedium.INTERNAL_MESSENGER,
    "메신저": ChannelMedium.INTERNAL_MESSENGER,
}
_KEYWORDS_LONGEST_FIRST: tuple[str, ...] = tuple(sorted(_MEDIUM_KEYWORDS, key=lambda k: (-len(k), k)))

#: Korean label per medium, used by the back-translation renderer and in messages.
MEDIUM_LABEL_KO: dict[ChannelMedium, str] = {
    ChannelMedium.SLACK: "슬랙",
    ChannelMedium.KAKAOTALK: "카카오톡",
    ChannelMedium.TEAMS: "팀즈",
    ChannelMedium.PHONE: "전화",
    ChannelMedium.SMS: "문자",
    ChannelMedium.EMAIL: "이메일",
    ChannelMedium.PAGER: "호출 도구",
    ChannelMedium.INTERNAL_MESSENGER: "사내 메신저",
    ChannelMedium.OTHER: "기타 채널",
}


def find_media(text: str) -> tuple[ChannelMedium, ...]:
    """All channel media mentioned in ``text`` (sorted, unique).

    Longest keywords claim their characters first, so "이메일" is read once (as email) and
    "pagerduty" is not additionally read as a generic "pager" mention.
    """
    key = compact(text)
    claimed = [False] * len(key)
    found: set[ChannelMedium] = set()
    for kw in _KEYWORDS_LONGEST_FIRST:
        pos = key.find(kw)
        while pos != -1:
            if not any(claimed[pos : pos + len(kw)]):
                claimed[pos : pos + len(kw)] = [True] * len(kw)
                found.add(_MEDIUM_KEYWORDS[kw])
            pos = key.find(kw, pos + 1)
    return tuple(sorted(found))


def medium_of(value: str) -> ChannelMedium | None:
    """The single medium a channel surface/value names, or ``None`` if unknown or mixed."""
    key = compact(value)
    for medium in ChannelMedium:
        if key == medium.value:
            return medium
    media = find_media(value)
    return media[0] if len(media) == 1 else None


# --------------------------------------------------------------------------- values


@dataclass(frozen=True)
class Duration:
    """A duration mention: char range in the scanned text and its length in seconds."""

    start: int
    end: int
    seconds: int


_UNIT_SECONDS: dict[str, int] = {
    "일": 86400,
    "시간": 3600,
    "분": 60,
    "초": 1,
    "day": 86400,
    "days": 86400,
    "d": 86400,
    "hour": 3600,
    "hours": 3600,
    "hr": 3600,
    "hrs": 3600,
    "h": 3600,
    "minute": 60,
    "minutes": 60,
    "min": 60,
    "mins": 60,
    "m": 60,
    "second": 1,
    "seconds": 1,
    "sec": 1,
    "secs": 1,
    "s": 1,
}
# Korean units carry look-aheads for words that merely start with the unit syllable
# (분기 quarter, 일차 day-N, 초과 exceeding). English units must end at a word boundary.
# Thousands separators are part of the number: "1,440분" is a day, not 440 minutes.
_DURATION_RE = re.compile(
    r"(?<![\d.])(?P<num>\d{1,3}(?:,\d{3})+|\d+)(?P<frac>\.\d+)?\s*"
    r"(?P<unit>시간(?:\s*반(?![가-힣]))?|분(?![기위])|초(?!과)|일(?![차자])"
    r"|(?i:days?|hours?|hrs?|minutes?|mins?|seconds?|secs?|[dhms])(?![A-Za-z]))"
)
_MONTH_BEFORE = re.compile(r"\d+\s*월\s*$")
_JOIN_GAP = re.compile(r"\s*")
#: Tildes may be spaced; a dash only joins a range when unspaced ("10분 - 2차 당직" is a list).
_RANGE_SEP = r"(?:\s*[~∼〜]\s*|[\-–—])"
#: A duration is a range bound when a number sits on the other side of a dash/tilde
#: ("5~10분", "10-15 min", "5분~10분").
_RANGE_BEFORE = re.compile(rf"\d(?:\s*(?:시간|분|초|일|[A-Za-z]+))?{_RANGE_SEP}$")
_RANGE_AFTER = re.compile(rf"^{_RANGE_SEP}\d")
_THOUSANDS = re.compile(r"\d{1,3}(?:,\d{3})+")


def find_durations(text: str) -> tuple[Duration, ...]:
    """Every duration in ``text``; "1시간 30분" is one 90-minute duration, "3월 1일" none."""
    raw: list[tuple[int, int, int, int]] = []  # start, end, seconds, unit size
    for m in _DURATION_RE.finditer(text):
        unit = m.group("unit")
        if unit == "일" and _MONTH_BEFORE.search(text[: m.start()]):
            continue  # a calendar date, not a duration
        half = unit.endswith("반")
        base = "시간" if half else unit.casefold() if unit.isascii() else unit
        size = _UNIT_SECONDS[base]
        number = Fraction(m.group("num").replace(",", "") + (m.group("frac") or ""))
        value = number * size + (1800 if half else 0)
        if value.denominator != 1:
            continue
        raw.append((m.start(), m.end(), int(value), size))
    merged: list[tuple[int, int, int, int]] = []
    for item in raw:
        if merged:
            p_start, p_end, p_sec, p_size = merged[-1]
            if item[3] < p_size and _JOIN_GAP.fullmatch(text[p_end : item[0]]):
                merged[-1] = (p_start, item[1], p_sec + item[2], item[3])
                continue
        merged.append(item)
    return tuple(Duration(s, e, sec) for s, e, sec, _ in merged)


def parse_minutes(value: object) -> int | None:
    """Coerce an extracted value to whole minutes (``10``, ``"10"``, ``"10분"``, ``"1시간"``).

    ``None`` when the value is not exactly one non-negative whole-minute duration — a guessed
    number would silently change escalation/deadline judgements. A range ("5~10분") is not one
    value either: picking a bound would state a precision the runbook does not.
    """
    if not isinstance(value, str):
        return _whole_number(value)
    text = value.strip()
    if _THOUSANDS.fullmatch(text):
        text = text.replace(",", "")
    if text.isdecimal():
        return _whole_number(text)
    durations = find_durations(text)
    if len(durations) != 1 or durations[0].seconds % 60 or is_range_bound(text, durations[0]):
        return None
    return durations[0].seconds // 60


def is_range_bound(text: str, duration: Duration) -> bool:
    """Whether ``duration`` (found in ``text``) is one end of a range such as "5~10분"."""
    return bool(_RANGE_BEFORE.search(text[: duration.start]) or _RANGE_AFTER.search(text[duration.end :]))


def _whole_number(value: object) -> int | None:
    """Non-negative integer from an int, an integral float or a digit string; bools excluded.

    ``isdecimal`` rather than ``isdigit``: superscripts such as "²" are digits that ``int()``
    rejects, and extractor output must never crash the assembler.
    """
    if isinstance(value, str):
        return int(value) if value.isdecimal() else None
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    return int(value) if value >= 0 else None


_TRUE_WORDS = frozenset({"true", "yes", "y", "예", "네", "필요", "필수", "요구", "있음", "required"})
_FALSE_WORDS = frozenset({"false", "no", "n", "아니오", "아니요", "불필요", "없음", "not required"})


def parse_bool(value: object) -> bool | None:
    """Coerce ``requires_approval``-style flags; unknown words stay unknown (``None``)."""
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        return None
    key = " ".join(value.casefold().split())
    if key in _TRUE_WORDS:
        return True
    if key in _FALSE_WORDS:
        return False
    return None


_TIER_RE = re.compile(r"(?:tier|t|티어)?[-_ ]?(\d+)(?:등급|티어)?")


def parse_tier(value: object) -> int | None:
    """Coerce a tier (``1``, ``"1"``, ``"tier-1"``, ``"T1"``, ``"1등급"``)."""
    if not isinstance(value, str):
        return _whole_number(value)
    m = _TIER_RE.fullmatch(compact(value))
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------- resolution


class ResolutionStatus(StrEnum):
    MATCHED = "matched"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PersonResolution:
    """Result of resolving a person surface against the roster.

    ``method`` records which rule decided (name/alias/surname_title/given_name) so reports can
    explain *why* "김대리" became 김민수.
    """

    status: ResolutionStatus
    roster_entry: RosterEntry | None = None
    candidates: tuple[RosterEntry, ...] = ()
    method: str | None = None


@dataclass(frozen=True)
class RoleResolution:
    """Canonical role name plus the roster people who hold it (``roles`` or ``title``)."""

    name: str
    holders: tuple[RosterEntry, ...]
    known: bool


@dataclass(frozen=True)
class _PersonKeys:
    """:func:`name_key` forms of one roster row."""

    entry: RosterEntry
    name: str
    aliases: frozenset[str]
    titles: frozenset[str]  # title + roles
    team: str | None
    split: tuple[str, str] | None

    def qualifiers(self, core: str) -> tuple[str, ...] | None:
        """What ``core`` adds around the full name, or ``None`` if it is not this name.

        "김민수대리" -> ("대리",); "결제팀김민수대리" -> ("결제팀", "대리"). Only the person's own
        team is recognised as a prefix: Korean writes "결제팀 김민수 대리" as often as the reverse.
        """
        found: list[str] = []
        if (
            self.team is not None
            and core.startswith(self.team)
            and core[len(self.team) :].startswith(self.name)
        ):
            found.append(self.team)
            core = core[len(self.team) :]
        if not core.startswith(self.name):
            return None
        if core != self.name:
            found.append(core[len(self.name) :])
        return tuple(found)

    def narrowed_by(self, qualifier: str) -> bool:
        """The qualifier is this person's own title, role or team."""
        return qualifier in self.titles or qualifier == self.team


_FUZZY_SERVICE_MIN = 92.0
#: ``CatalogService.kind`` of a contact medium entry (e.g. an on-premise messenger with a location).
CATALOG_CHANNEL_KIND = "channel"


class EntityResolver:
    """Resolve surfaces against the trusted roster and catalog (pure, deterministic).

    ``extra_roles`` adds organisation role names that are not in the roster (policy terminal
    roles, on-call rotation roles) so "CISO" or "1차 당직자" resolve to canonical ROLE names.
    """

    def __init__(
        self,
        roster: Iterable[RosterEntry] = (),
        catalog: Iterable[CatalogService] = (),
        *,
        extra_roles: Iterable[str] = (),
    ) -> None:
        # A row listed twice is one person, not a pair of namesakes needing discriminated ids.
        self.roster: tuple[RosterEntry, ...] = tuple(dict.fromkeys(roster))
        self.catalog: tuple[CatalogService, ...] = tuple(catalog)
        # Channel entries describe a contact medium's failure domain, not something a runbook
        # names as a service, so service resolution never selects them.
        self._services = tuple(s for s in self.catalog if s.kind != CATALOG_CHANNEL_KIND)
        self._people = tuple(
            _PersonKeys(
                entry=e,
                name=name_key(e.name),
                aliases=frozenset(name_key(a) for a in e.aliases),
                titles=frozenset(name_key(t) for t in (e.title, *e.roles) if t),
                team=name_key(e.team) if e.team else None,
                split=split_korean_name(e.name),
            )
            for e in self.roster
        )
        self._title_words = frozenset(name_key(t) for t in RANK_TITLES | ROLE_WORDS).union(
            *(p.titles for p in self._people)
        )
        self._roles = self._role_index(extra_roles)
        self._teams = _first_by_key(e.team for e in self.roster if e.team)
        self._locations = _first_by_key(s.location for s in self.catalog if s.location)
        self._person_ids = self._assign_person_ids()

    # ------------------------------------------------------------------ indexes
    def _role_index(self, extra_roles: Iterable[str]) -> dict[str, str]:
        """role key -> display. Roster spelling wins over policy and built-in vocabulary."""
        index: dict[str, str] = {}
        roster_roles = sorted({clean_surface(r) for e in self.roster for r in e.roles})
        for name in [*roster_roles, *sorted(map(clean_surface, extra_roles)), *sorted(ROLE_WORDS)]:
            index.setdefault(name_key(name), name)
        return index

    def _assign_person_ids(self) -> dict[RosterEntry, str]:
        """Node ids for roster people; namesakes (동명이인) get a team/ordinal discriminator.

        Without this two different 김민수 would collapse into one node and inherit each
        other's channels and roles.
        """
        by_name: dict[str, list[RosterEntry]] = {}
        for e in self.roster:
            # the same key the resolver compares by, so names that slug alike are namesakes
            by_name.setdefault(name_key(e.name), []).append(e)
        ids: dict[RosterEntry, str] = {}
        for group in by_name.values():
            if len(group) == 1:
                ids.setdefault(group[0], node_id(NodeKind.PERSON, group[0].name))
                continue
            teams = [e.team for e in group]
            distinct = all(teams) and len(set(teams)) == len(teams)
            for i, e in enumerate(group, start=1):
                tag = e.team if distinct and e.team else str(i)
                ids.setdefault(e, node_id(NodeKind.PERSON, f"{e.name} {tag}"))
        return ids

    # ------------------------------------------------------------------ people
    def person_node_id(self, entry: RosterEntry) -> str:
        """Graph id of a roster person (stable across runs, namesake-safe)."""
        return self._person_ids[entry]

    def resolve_person(self, surface: str) -> PersonResolution:
        """exact name -> alias -> 성+직급 ("김대리") -> 이름+님/씨/직급 ("민수 대리"); unique or bust."""
        full = name_key(_PAREN.sub("", surface))
        core = strip_honorific(full)
        honorific = core != full
        if not core:
            return PersonResolution(ResolutionStatus.UNKNOWN)
        stages = (
            # the full surface first: a real name may itself end in 님/씨 ("이하님")
            ("name", self._by_name(full) or self._by_name(core)),
            ("alias", [p for p in self._people if full in p.aliases or core in p.aliases]),
            ("surname_title", [p for p in self._people if _is_surname_title(core, p.split, p.titles)]),
            ("given_name", [p for p in self._people if _is_given(core, p.split, p.titles, honorific)]),
        )
        for method, matches in stages:
            if not matches:
                continue
            entries = tuple(sorted({p.entry for p in matches}, key=_entry_order))
            if len(entries) == 1:
                return PersonResolution(ResolutionStatus.MATCHED, entries[0], entries, method)
            return PersonResolution(ResolutionStatus.AMBIGUOUS, None, entries, method)
        return PersonResolution(ResolutionStatus.UNKNOWN)

    def _by_name(self, core: str) -> list[_PersonKeys]:
        """Full name, optionally qualified by a title ("김민수 대리") or by the person's own team
        ("김민수 결제팀", "결제팀 김민수 대리"); qualifiers that fit only one namesake narrow it."""
        exact = [p for p in self._people if core == p.name]
        if exact:
            return exact
        qualified: list[tuple[_PersonKeys, tuple[str, ...]]] = []
        for p in self._people:
            found = p.qualifiers(core)
            if found and all(q in self._title_words or q == p.team for q in found):
                qualified.append((p, found))
        narrowed = [p for p, found in qualified if all(p.narrowed_by(q) for q in found)]
        return narrowed if len(narrowed) == 1 else [p for p, _ in qualified]

    # ------------------------------------------------------------------ roles / teams
    def resolve_role(self, surface: str) -> RoleResolution:
        """Canonical role name ("cto" -> "CTO", "1차 당직자" -> "1차 당직") and its holders."""
        key = strip_honorific(name_key(surface))
        name = self._roles.get(key)
        if name is None and key.endswith("자"):
            name = self._roles.get(key[:-1])
        known = name is not None
        name = name or display_name(surface)
        role_key = name_key(name)
        holders = tuple(p.entry for p in self._people if role_key in p.titles)
        return RoleResolution(name=name, holders=tuple(sorted(holders, key=_entry_order)), known=known)

    def is_role_word(self, surface: str) -> bool:
        """True when the surface names a position (팀장, CTO, 1차 당직자) rather than a person."""
        return self.resolve_role(surface).known

    def resolve_team(self, surface: str) -> str | None:
        """Roster team spelling for a team surface ("결제 팀" -> "결제팀")."""
        return self._teams.get(name_key(surface))

    # ------------------------------------------------------------------ catalog
    def resolve_service(self, surface: str) -> str | None:
        """Canonical catalog name via exact/alias -> normalised key -> rapidfuzz ratio >= 92.

        Returns ``None`` when nothing (or more than one service) fits — the caller keeps the
        runbook's own name rather than attaching facts to the wrong service.
        """
        text = clean_surface(surface)
        exact = {s.name for s in self._services if text == s.name or text in s.aliases}
        if exact:
            return min(exact) if len(exact) == 1 else None
        key = service_key(text)
        keyed = {s.name for s in self._services for n in (s.name, *s.aliases) if service_key(n) == key}
        if keyed:
            return next(iter(keyed)) if len(keyed) == 1 else None
        scored: dict[str, float] = {}
        for s in self._services:
            for n in (s.name, *s.aliases):
                score = fuzz.ratio(key, service_key(n))
                if score >= _FUZZY_SERVICE_MIN:
                    scored[s.name] = max(scored.get(s.name, 0.0), score)
        if not scored:
            return None
        best = max(scored.values())
        winners = [name for name, score in scored.items() if score == best]
        return winners[0] if len(winners) == 1 else None

    def catalog_entry(self, name: str) -> CatalogService | None:
        return next((s for s in self._services if s.name == name), None)

    def service_kind(self, name: str) -> NodeKind:
        """Catalog-declared kind; services the catalog does not know stay SERVICE."""
        entry = self.catalog_entry(name)
        return NodeKind.SYSTEM if entry is not None and entry.kind == "system" else NodeKind.SERVICE

    def resolve_location(self, surface: str) -> str | None:
        return self._locations.get(name_key(surface))


def _is_surname_title(core: str, split: tuple[str, str] | None, titles: frozenset[str]) -> bool:
    """ "김대리" / "박팀장": surname immediately followed by the person's title or role."""
    return split is not None and any(core == split[0] + t for t in titles)


def _is_given(core: str, split: tuple[str, str] | None, titles: frozenset[str], honorific: bool) -> bool:
    """ "민수 대리" / "민수님": given name with a title, or with an honorific alone."""
    if split is None:
        return False
    given = split[1]
    return (honorific and core == given) or any(core == given + t for t in titles)


def _entry_order(e: RosterEntry) -> tuple[str, str, str]:
    return (e.name, e.team or "", e.title or "")


def _first_by_key(names: Iterable[str]) -> dict[str, str]:
    """name key -> the lexicographically first spelling (deterministic canonical form)."""
    out: dict[str, str] = {}
    for name in sorted(clean_surface(n) for n in names):
        out.setdefault(name_key(name), name)
    return out
