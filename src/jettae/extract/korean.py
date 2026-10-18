"""Korean lexicons and recognisers for the deterministic rule parser (LLM-OFF path).

Why hand-written lexicons instead of a statistical tagger? The rule parser is both the
fallback that must work with the LLM switched off (P4) and the "rule" arm of the 2×2
ablation. Both roles need behaviour a reviewer can audit line by line and that never changes
between runs (P6): every mention below comes from a closed word list or a narrow pattern, so
one can always say *why* a span was read as a person, channel or action.

The recognisers are deliberately conservative. A surface that could be a person or a common
noun ("결제 팀장" — the payment team's lead, not a person called 결제) is left to the role
lexicon or not recognised at all; an unrecognised mention stays absent and the fixed rules
later report the gap as unknown instead of trusting a guess (P2, P3).

All offsets returned here index the scanned string plus the caller-supplied ``offset`` so the
rule parser can map them straight back to ``Document.text``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from jettae.domain import ActionType, CatalogService, ChannelMedium, IncidentType, NodeKind, RosterEntry
from jettae.extract.resolve import CATALOG_CHANNEL_KIND

# --------------------------------------------------------------------------- vocabulary

#: Rank titles (직급) that turn a surname or name into a person ("김대리", "김민수 대리").
TITLES: tuple[str, ...] = (
    "본부장", "센터장", "그룹장", "파트장", "연구원", "매니저",
    "팀장", "실장", "부장", "차장", "과장", "대리", "주임", "사원", "선임", "책임", "수석",
    "이사", "상무", "전무",
)  # fmt: skip

#: Common one-syllable surnames. Rare surnames that collide with frequent prefixes
#: (부팀장 = deputy lead, 전사원 = all staff) are left out on purpose.
SURNAMES: str = (
    "김이박최정강조윤장임한오서신권황안송류유전홍고문양손배백허남심노하곽성차주우구민나진지엄"
    "채원천방공현함변염여추도소석선설마길연위표명기반왕금옥육인맹제모탁국봉경"
)
COMPOUND_SURNAMES: tuple[str, ...] = ("남궁", "황보", "제갈", "선우", "독고", "사공", "서문")

#: Particles that may follow a name; used to decide where a person mention ends.
JOSA: tuple[str, ...] = (
    "께서", "에게서", "에게", "한테", "께", "이랑", "랑", "하고", "부터", "까지", "으로", "로",
    "이나", "이", "가", "은", "는", "을", "를", "의", "에", "와", "과", "도", "만", "나", "야",
)  # fmt: skip

#: Words that look like a two/three-syllable name in "X 팀장" / "X님" but are nouns.
NON_NAME_WORDS: frozenset[str] = frozenset(
    {
        "결제", "주문", "정산", "배송", "검색", "회원", "인증", "로그인", "상품", "재고", "물류", "광고",
        "추천", "쿠폰", "포인트", "알림", "메일", "메시지", "채팅", "송출", "방송", "편성", "영상", "음원",
        "고객", "고객사", "회사", "개발", "기획", "디자인", "영업", "마케팅", "재무", "회계", "인사", "총무",
        "법무", "감사", "경영", "전략", "사업", "기술", "품질", "보안", "운영", "인프라", "플랫폼", "데이터",
        "서버", "백엔드", "프론트", "모바일", "클라우드", "네트워크", "시스템", "서비스", "정보", "정보보호",
        "개인정보", "지원", "상담", "콜센터", "센터", "본부", "부서", "그룹", "파트", "담당", "해당",
        "당직", "현장", "소속", "직속", "상위", "차상위", "최종", "관련", "전체", "유관", "기존", "신규",
        "신입", "선배", "후배", "사장", "선생", "기사", "여러분", "부모", "주인", "손님", "관리", "관리자",
        "사용자", "운영자", "담당자", "책임자", "대표", "팀원", "직원", "임원", "외부", "내부", "협력사",
        "장애", "대응", "연락", "비상", "긴급", "공통", "일반", "전사", "모든", "각자", "우리", "저희",
        "다른", "같은", "이번", "다음", "현재", "주간", "야간", "주말", "평일", "새벽", "오전", "오후",
        "팀", "각", "직급", "직책", "결재", "승인", "보고", "상향", "실무", "최근", "이후", "이전",
        "당시", "직접", "바로", "먼저", "우선", "일단", "다시", "반드시", "즉시", "별도", "추가", "또는",
        "그리고", "그래도", "만약", "혹시", "보통", "평소", "항상", "가끔", "주로", "각각", "모두",
    }
)  # fmt: skip

#: Generic words that are never a service name even when followed by 서비스/장애/쪽.
NON_SERVICE_WORDS: frozenset[str] = frozenset(
    {
        "공통", "전체", "전사", "일반", "개요", "부록", "참고", "기타", "장애", "서비스", "시스템", "대응",
        "연락", "연락망", "비상", "긴급", "보안", "침해", "개인정보", "신고", "보고", "상향", "에스컬레이션",
        "당직", "운영", "모든", "각", "주요", "핵심", "해당", "관련", "만약", "혹시", "만일", "계속",
        "지속", "대형", "중대", "심각", "경미", "단순", "일시", "일시적", "전면", "부분", "내부", "외부",
        "신규", "기존", "동일", "다른", "이번", "이전", "다음", "현재", "실제", "예상", "예시", "가상",
        "모의", "훈련", "테스트", "사전", "사후", "최종", "1차", "2차", "3차", "우리", "저희", "본", "이",
        "그", "저", "어떤", "무슨", "큰", "작은", "갑자기", "또", "또는", "및", "그리고", "그래도",
        "다시", "전", "후", "당일", "야간", "주간", "새벽", "주말", "평일", "일단", "우선", "먼저",
        "즉시", "바로", "여러", "각종", "기본", "공용", "고객", "사용자", "외부의",
        "문제", "이슈", "에러", "오류", "알람", "경보", "감지", "조치", "복구", "롤백", "절차", "체계",
        "담당", "승인", "결재", "런북", "매뉴얼", "가이드", "개선", "점검", "확인", "발생", "처리",
    }
)  # fmt: skip

#: Words that cannot start a team name ("각 팀", "담당 팀").
NON_TEAM_PREFIXES: frozenset[str] = frozenset(
    {
        "각", "해당", "우리", "저희", "전", "타", "상위", "관련", "소속", "모든", "유관", "다른", "같은",
        "본", "이", "그", "저", "한", "두", "세", "담당", "대응", "각각", "여러", "전체", "외부", "내부",
        "협력", "상대", "자기", "어느", "새", "신규", "기존", "1차", "2차", "3차",
    }
)  # fmt: skip

#: Team bases that are usually qualified by the noun before them ("결제 개발팀").
GENERIC_TEAM_BASES: frozenset[str] = frozenset(
    {"서비스", "시스템", "플랫폼", "개발", "운영", "인프라", "엔지니어링", "기술", "지원"}
)

#: Tokens that look like surname+title but are ordinary words ("이상 무" = no anomaly).
_FALSE_SURNAME_TITLE: frozenset[str] = frozenset({"이상무", "정사원", "전사원", "이사장", "정책임"})

# (pattern, canonical role name). Longest alternatives first inside each pattern.
_ROLE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"개인정보\s*보호\s*책임자|개인정보보호책임자", "개인정보 보호책임자"),
    (r"정보보호\s*최고\s*책임자", "정보보호 최고책임자"),
    (r"보안\s*담당자|정보보호\s*담당자", "보안 담당자"),
    (r"1차\s*당직(?:자)?|1차\s*온콜|[Pp]rimary\s*[Oo]n-?[Cc]all", "1차 당직"),
    (r"2차\s*당직(?:자)?|2차\s*온콜|[Ss]econdary\s*[Oo]n-?[Cc]all", "2차 당직"),
    (r"당직\s*책임자", "당직 책임자"),
    (r"당직자|당직\s*담당자", "당직자"),
    (r"온콜\s*담당자|온콜|(?<![A-Za-z])[Oo]n-?[Cc]all(?![A-Za-z])", "온콜"),
    (r"인시던트\s*커맨더|[Ii]ncident\s*[Cc]ommander", "인시던트 커맨더"),
    (r"부문장", "부문장"),
    (r"본부장", "본부장"),
    (r"센터장", "센터장"),
    (r"그룹장", "그룹장"),
    (r"파트장", "파트장"),
    (r"테크\s*리드", "테크리드"),
    (r"팀장", "팀장"),
    (r"실장", "실장"),
    (r"대표\s*이사|대표(?!\s*(?:번호|전화|메일|이메일|계정|채널|번))(?![적])", "대표"),
    (r"부사장", "부사장"),
)
_LATIN_ROLES: tuple[str, ...] = ("CISO", "CPO", "CTO", "CIO", "CEO", "COO", "CFO")

# (pattern, medium, canonical name or None to keep the surface). Order = priority on ties.
_CHANNEL_PATTERNS: tuple[tuple[str, ChannelMedium, str | None], ...] = (
    (r"(?<![\w#&/])#[A-Za-z가-힣][\w가-힣\-]*", ChannelMedium.SLACK, None),
    (r"슬랙|(?<![A-Za-z])[Ss]lack(?![A-Za-z])", ChannelMedium.SLACK, "슬랙"),
    (
        r"카카오\s*톡|카톡|알림톡|(?<![A-Za-z])[Kk]akao\s*[Tt]alk(?![A-Za-z])",
        ChannelMedium.KAKAOTALK,
        "카카오톡",
    ),
    (r"사내\s*메신[저져]", ChannelMedium.INTERNAL_MESSENGER, "사내 메신저"),
    (r"팀즈|(?:MS|Microsoft)\s*Teams|(?<![A-Za-z])Teams(?![A-Za-z])", ChannelMedium.TEAMS, "팀즈"),
    (r"(?<![A-Za-z])[Pp]ager[Dd]uty(?![A-Za-z])|페이저\s*듀티", ChannelMedium.PAGER, "PagerDuty"),
    (r"(?<![A-Za-z])[Oo]ps[Gg]enie(?![A-Za-z])|옵스지니", ChannelMedium.PAGER, "Opsgenie"),
    (r"(?<![A-Za-z])incident\.io(?![A-Za-z])", ChannelMedium.PAGER, "incident.io"),
    (r"호출기|페이저|삐삐", ChannelMedium.PAGER, "호출기"),
    (r"이메일|메일(?!링)|(?<![A-Za-z])[Ee]-?[Mm]ail(?![A-Za-z])", ChannelMedium.EMAIL, "이메일"),
    # bounded and anchored at a token start: unbounded "[\w.]+@" is quadratic on long words
    (r"(?<![\w.*+\-])[\w.*+\-]{1,64}@[\w\-]{1,63}(?:\.[\w\-]{1,63}){1,4}", ChannelMedium.EMAIL, "이메일"),
    (r"문자\s*메시지|문자(?!열)|(?<![A-Za-z])SMS(?![A-Za-z])", ChannelMedium.SMS, "문자"),
    (r"유선\s*전화|휴대\s*전화|휴대폰|핸드폰|전화|유선|통화", ChannelMedium.PHONE, "전화"),
    (r"(?<![가-힣A-Za-z])콜(?=$|[^가-힣]|로|을|를|이|은|는|도|주|해|하|드)", ChannelMedium.PHONE, "전화"),
    (r"(?<![\d*])0\d{1,2}-[\d*xX]{3,4}-[\d*xX]{4}(?![\d*])", ChannelMedium.PHONE, "전화"),
)

# (pattern, action type). Canonical names come from ACTION_LABEL.
_ACTION_PATTERNS: tuple[tuple[str, ActionType], ...] = (
    (
        r"롤\s*백|원복|되돌리기|되돌림|되돌려|되돌리|이전\s*버전(?:으로)?\s*(?:재)?배포"
        r"|(?<![A-Za-z])[Rr]oll\s*-?\s*[Bb]ack(?![A-Za-z])",
        ActionType.ROLLBACK,
    ),
    (
        r"페일\s*오버|절체|DR\s*(?:센터\s*)?전환|재해\s*복구\s*전환|(?<![A-Za-z])[Ff]ail\s*-?\s*[Oo]ver(?![A-Za-z])",
        ActionType.FAILOVER,
    ),
    (r"재시작|재기동|리스타트|재부팅|(?<![A-Za-z])[Rr]e(?:start|boot)(?![A-Za-z])", ActionType.RESTART),
    (
        r"기능\s*플래그\s*(?:를\s*)?(?:끄기|끄고|꺼|끄|OFF|[Oo]ff|비활성화?)|플래그\s*(?:를\s*)?(?:OFF|[Oo]ff|끄기|꺼|끄|비활성화?)"
        r"|(?<![A-Za-z])[Ff]eature\s*[Ff]lag(?![A-Za-z])|피처\s*플래그|킬\s*스위치|(?<![A-Za-z])[Kk]ill\s*[Ss]witch(?![A-Za-z])",
        ActionType.FEATURE_FLAG,
    ),
    (r"스케일\s*아웃|(?<![A-Za-z])[Ss]cale\s*-?\s*[Oo]ut(?![A-Za-z])|증설", ActionType.SCALE),
)

#: Display label per action type. Rule-parser action names are "<서비스> <label>".
ACTION_LABEL: dict[ActionType, str] = {
    ActionType.ROLLBACK: "롤백",
    ActionType.FAILOVER: "페일오버",
    ActionType.RESTART: "재시작",
    ActionType.FEATURE_FLAG: "기능 플래그 OFF",
    ActionType.SCALE: "스케일 아웃",
    ActionType.OTHER: "조치",
}

#: Reporting authorities (pattern, display label) for Korean breach/privacy obligations.
RECIPIENT_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"(?<![A-Za-z])KISA(?![A-Za-z])|한국인터넷진흥원|인터넷진흥원", "KISA"),
    (r"과학기술정보통신부|과기정통부|과기부", "과기정통부"),
    (r"개인정보\s*보호\s*위원회|개인정보위|(?<![A-Za-z])PIPC(?![A-Za-z])", "개인정보보호위원회"),
)
_BREACH_RE = re.compile(
    r"침해|해킹|침입|랜섬웨어|디도스|(?<![A-Za-z])DD[oO]S(?![A-Za-z])|악성\s*코드|서비스\s*거부"
)
_PRIVACY_RE = re.compile(r"개인\s*정보|정보\s*주체")

# --------------------------------------------------------------------------- durations

_UNIT_MIN = r"(?:분|minutes?|mins?|m)(?![A-Za-z])"
_UNIT_HOUR = r"(?:시간|hours?|hrs?|h)(?![A-Za-z])"

#: Non-capturing duration pattern for embedding in larger patterns ("10분", "1시간 30분", "2h").
DURATION_PATTERN: str = (
    rf"(?<![\d.])(?:\d+(?:\.\d+)?\s*{_UNIT_HOUR}(?:\s*\d+\s*{_UNIT_MIN})?|\d+\s*{_UNIT_MIN})"
)
_DURATION_RE = re.compile(
    rf"(?<![\d.])(?:(?P<h>\d+(?:\.\d+)?)\s*{_UNIT_HOUR}(?:\s*(?P<hm>\d+)\s*{_UNIT_MIN})?"
    rf"|(?P<m>\d+)\s*{_UNIT_MIN})"
)
_DELAY_RE = re.compile(rf"(?P<d>{DURATION_PATTERN})\s*(?:이상\s*)?(?:지속|계속|동안|간)")


@dataclass(frozen=True, slots=True)
class Duration:
    """A duration mention: absolute char range and its value in whole minutes."""

    start: int
    end: int
    minutes: int


def _minutes_of(match: re.Match[str]) -> int | None:
    hours, hour_minutes, minutes = match.group("h"), match.group("hm"), match.group("m")
    if minutes is not None:
        return int(minutes)
    total = float(hours) * 60 + (int(hour_minutes) if hour_minutes else 0)
    # "1.25시간" is 75 minutes; "0.01h" is not a whole minute -> absent rather than rounded.
    return int(total) if total.is_integer() else None


def parse_minutes(text: str) -> int | None:
    """Minutes stated by the first duration in ``text`` ('10분', '1시간 30분', '30 min', '2h')."""
    for match in _DURATION_RE.finditer(text):
        value = _minutes_of(match)
        if value is not None:
            return value
    return None


def find_durations(text: str, offset: int = 0) -> list[Duration]:
    """Every whole-minute duration in ``text`` (document order)."""
    found: list[Duration] = []
    for match in _DURATION_RE.finditer(text):
        value = _minutes_of(match)
        if value is not None:
            found.append(Duration(offset + match.start(), offset + match.end(), value))
    return found


def find_delay(text: str, offset: int = 0) -> Duration | None:
    """A detection delay such as "5분 이상 지속" / "10분 동안" inside a trigger condition."""
    match = _DELAY_RE.search(text)
    if match is None:
        return None
    value = parse_minutes(match.group("d"))
    if value is None:
        return None
    return Duration(offset + match.start("d"), offset + match.end("d"), value)


# --------------------------------------------------------------------------- incidents


def incident_type_of(text: str, recipients: Iterable[str]) -> IncidentType | None:
    """Incident type implied by an obligation sentence (or its heading context).

    개인정보 → privacy leak; 침해/해킹 with KISA/과기정통부 → breach; an obligation addressed only
    to 개인정보보호위원회 is a privacy obligation by law. Otherwise the type stays absent.
    """
    labels = set(recipients)
    if _PRIVACY_RE.search(text):
        return IncidentType.PRIVACY_LEAK
    if _BREACH_RE.search(text) and labels & {"KISA", "과기정통부"}:
        return IncidentType.BREACH
    if "개인정보보호위원회" in labels:
        return IncidentType.PRIVACY_LEAK
    return None


def incident_word_span(text: str) -> tuple[int, int] | None:
    """Span of the first word in ``text`` that decides :func:`incident_type_of`."""
    match = _PRIVACY_RE.search(text) or _BREACH_RE.search(text)
    return None if match is None else match.span()


# --------------------------------------------------------------------------- mentions


@dataclass(frozen=True, slots=True)
class Mention:
    """A recognised mention. ``name`` is the surface used for the extracted entity."""

    kind: NodeKind
    name: str
    start: int
    end: int
    medium: ChannelMedium | None = None
    action: ActionType | None = None
    title: str | None = None


_TITLE_ALT = "|".join(sorted(TITLES, key=lambda t: (-len(t), t)))
_JOSA_ALT = "|".join(JOSA)
#: Copula endings written straight after a name ("담당은 김대리예요", "백미영 대리입니다"): a
#: name ending in a vowel takes 예요/였, one ending in a consonant 이에요 (covered by 이).
_COPULA_ALT = "예요|에요|였|입니|이세요|세요"
_END = rf"(?=(?:{_JOSA_ALT}|{_COPULA_ALT})|[^가-힣]|$)"
_SURNAME_ALT = "|".join(COMPOUND_SURNAMES) + f"|[{SURNAMES}]"
_BOUNDARY = r"(?<![가-힣A-Za-z0-9])"

_SURNAME_TITLE_RE = re.compile(
    rf"{_BOUNDARY}(?P<core>(?:{_SURNAME_ALT})(?P<title>{_TITLE_ALT}))(?P<hon>님)?{_END}"
)
#: "김 대리", "박 팀장님" — the standard spelling writes a title apart from the surname, so
#: formal runbooks use it more than "김대리". It names the same person, hence the compact name.
_SPACED_SURNAME_TITLE_RE = re.compile(
    rf"{_BOUNDARY}(?P<sur>{_SURNAME_ALT}) (?P<title>{_TITLE_ALT})(?P<hon>님)?{_END}"
)
#: One-syllable words that are also determiners before a title ("이 과장" = this manager,
#: "전 팀장" = the former lead, "한 팀장" = one lead, "여 팀장" = a female lead): reading them
#: as surnames would be a guess, so the title is left to the role lexicon.
_DETERMINER_SURNAMES: frozenset[str] = frozenset("이전현주정신구원고차한양모제남여")
_NAME_TITLE_RE = re.compile(
    rf"{_BOUNDARY}(?P<name>[가-힣]{{2,4}})\s(?P<title>{_TITLE_ALT})(?P<hon>님)?{_END}"
)
#: "한서윤 CTO님께", "오준석 CISO가": a full name with an executive title is that person, not the
#: generic role. Only full names (3–4 syllables or a roster name) — "본사 CTO" is no one's name.
_NAME_EXEC_RE = re.compile(
    rf"{_BOUNDARY}(?P<name>[가-힣]{{2,4}})\s(?P<title>{'|'.join(_LATIN_ROLES)})(?![A-Za-z])(?P<hon>님)?"
)
_NAME_HON_RE = re.compile(rf"{_BOUNDARY}(?P<name>[가-힣]{{2,4}})(?P<hon>님|씨){_END}")
_TEAM_RE = re.compile(rf"{_BOUNDARY}(?P<base>[A-Za-z0-9가-힣]{{1,12}})\s?팀(?![장원즈들])")
_TEAM_QUALIFIER_RE = re.compile(r"(?P<qual>[A-Za-z0-9가-힣]{1,10})\s$")
_ROLE_RES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(rf"(?:{pattern})(?:님)?"), name) for pattern, name in _ROLE_PATTERNS
) + tuple((re.compile(rf"(?<![A-Za-z]){role}(?![A-Za-z])(?:님)?"), role) for role in _LATIN_ROLES)
_CHANNEL_RES: tuple[tuple[re.Pattern[str], ChannelMedium, str | None], ...] = tuple(
    (re.compile(pattern), medium, name) for pattern, medium, name in _CHANNEL_PATTERNS
)
_ACTION_RES: tuple[tuple[re.Pattern[str], ActionType], ...] = tuple(
    (re.compile(pattern), action) for pattern, action in _ACTION_PATTERNS
)
_ROLE_NAMES: frozenset[str] = frozenset(name for _, name in _ROLE_PATTERNS) | frozenset(_LATIN_ROLES)

#: Priority on equal-length overlaps: trusted lists first, then specific vocabularies.
_PRIO_ROSTER, _PRIO_CATALOG, _PRIO_CHANNEL, _PRIO_ACTION, _PRIO_PERSON, _PRIO_ROLE, _PRIO_TEAM = range(7)


class Lexicon:
    """Trusted vocabularies from the roster and catalog plus the patterns built from them.

    Roster names/aliases are matched exactly (they are facts about who exists); catalog
    names/aliases become service/system mentions, except catalog *contact media* (kind
    "channel", e.g. a self-hosted "사내 메신저" with its own failure domain), which are channels.
    Built once per extraction.
    """

    def __init__(self, roster: Iterable[RosterEntry] = (), catalog: Iterable[CatalogService] = ()) -> None:
        people = {name.strip() for entry in roster for name in (entry.name, *entry.aliases) if name.strip()}
        self.person_names: frozenset[str] = frozenset(people)
        services: dict[str, NodeKind] = {}
        channels: dict[str, tuple[str, ChannelMedium]] = {}
        for svc in catalog:
            surfaces = [name.strip() for name in (svc.name, *svc.aliases) if name.strip()]
            if svc.kind == CATALOG_CHANNEL_KIND:
                reading = _catalog_channel(surfaces)
                for surface in surfaces:
                    channels.setdefault(surface, reading)
                continue
            kind = NodeKind.SYSTEM if svc.kind == "system" else NodeKind.SERVICE
            for surface in surfaces:
                services.setdefault(surface, kind)
        self.service_kinds: dict[str, NodeKind] = dict(sorted(services.items()))
        #: catalog contact-medium surface -> (channel name, medium)
        self.catalog_channels: dict[str, tuple[str, ChannelMedium]] = dict(sorted(channels.items()))
        self._channel_re = _alternation(
            frozenset(self.catalog_channels), tail=rf"(?=(?:{_JOSA_ALT})|[^가-힣]|$)"
        )
        # A roster name or alias may carry an honorific ("Erin님이", "민수씨한테"): it belongs to
        # the mention's span (so "님" never separates a name from its particle in a pattern), not
        # to its name, which stays the roster surface.
        self._person_re = _alternation(
            self.person_names,
            tail=rf"(?P<hon>님|씨)?(?=님|씨|\s|{_TITLE_ALT}|{_JOSA_ALT}|{_COPULA_ALT}|[^가-힣]|$)",
        )
        self._service_re = _alternation(
            frozenset(self.service_kinds),
            tail=rf"(?=서비스|시스템|쪽|측|장애|문제|이슈|에러|오류|(?:{_JOSA_ALT})|[^가-힣]|$)",
        )

    def person_matches(self, text: str) -> Iterable[re.Match[str]]:
        return () if self._person_re is None else self._person_re.finditer(text)

    def service_matches(self, text: str) -> Iterable[re.Match[str]]:
        return () if self._service_re is None else self._service_re.finditer(text)

    def channel_matches(self, text: str) -> Iterable[re.Match[str]]:
        return () if self._channel_re is None else self._channel_re.finditer(text)

    def is_service(self, name: str) -> bool:
        return name in self.service_kinds


def _alternation(names: frozenset[str], *, tail: str) -> re.Pattern[str] | None:
    if not names:
        return None
    alts = "|".join(re.escape(n) for n in sorted(names, key=lambda n: (-len(n), n)))
    return re.compile(rf"(?<![가-힣A-Za-z0-9])(?:{alts}){tail}")


def _catalog_channel(surfaces: Iterable[str]) -> tuple[str, ChannelMedium]:
    """Name and medium of a catalog contact medium: the vocabulary's reading of its first
    surface that is a channel word (so "사내메신저" joins "사내 메신저"), else its own name."""
    names = list(surfaces)
    for surface in names:
        reading = channel_reading(surface)
        if reading is not None and reading.medium is not None:
            return reading.name, reading.medium
    return names[0], ChannelMedium.OTHER


#: The vocabulary alone (no roster, no catalog): what a word means before trusted lists weigh in.
_PLAIN_LEXICON = Lexicon()


def find_mentions(text: str, lexicon: Lexicon, *, offset: int = 0) -> list[Mention]:
    """All non-overlapping mentions in ``text`` (document order).

    Overlaps are resolved longest-first, then leftmost, then by source priority, so
    "김민수 대리님" is one person rather than a person plus the role "대리", and "결제팀" is a
    team rather than the service "결제".
    """
    cands: list[tuple[int, Mention]] = []
    cands += [(_PRIO_ROSTER, _person(m, offset, _roster_surface(m))) for m in lexicon.person_matches(text)]
    for m in lexicon.service_matches(text):
        kind = lexicon.service_kinds[m.group(0)]
        cands.append((_PRIO_CATALOG, Mention(kind, m.group(0), offset + m.start(), offset + m.end())))
    for m in lexicon.channel_matches(text):
        name, medium = lexicon.catalog_channels[m.group(0)]
        cands.append(
            (
                _PRIO_CATALOG,
                Mention(NodeKind.CHANNEL, name, offset + m.start(), offset + m.end(), medium=medium),
            )
        )
    for rx, medium, canonical in _CHANNEL_RES:
        for m in rx.finditer(text):
            name = canonical or m.group(0)
            cands.append(
                (
                    _PRIO_CHANNEL,
                    Mention(NodeKind.CHANNEL, name, offset + m.start(), offset + m.end(), medium=medium),
                )
            )
    for rx, action in _ACTION_RES:
        for m in rx.finditer(text):
            label = ACTION_LABEL[action]
            cands.append(
                (
                    _PRIO_ACTION,
                    Mention(NodeKind.ACTION, label, offset + m.start(), offset + m.end(), action=action),
                )
            )
    cands += [(_PRIO_PERSON, p) for p in _people(text, lexicon, offset)]
    for rx, role in _ROLE_RES:
        cands += [
            (_PRIO_ROLE, Mention(NodeKind.ROLE, role, offset + m.start(), offset + m.end()))
            for m in rx.finditer(text)
        ]
    cands += [(_PRIO_TEAM, t) for t in _teams(text, offset)]
    return _resolve_overlaps(cands)


def channel_reading(surface: str) -> Mention | None:
    """The channel ``surface`` names when read without the catalog, if it is exactly one
    channel word. A catalog may name a service "이메일"; in "(휴대폰, 이메일)" it is a channel."""
    found = find_mentions(surface, _PLAIN_LEXICON)
    if (
        len(found) == 1
        and found[0].kind is NodeKind.CHANNEL
        and (found[0].start, found[0].end) == (0, len(surface))
    ):
        return found[0]
    return None


def names_channel(text: str) -> bool:
    """Whether ``text`` contains a channel word when read without the catalog ("이메일 서비스")."""
    return any(m.kind is NodeKind.CHANNEL for m in find_mentions(text, _PLAIN_LEXICON))


def _resolve_overlaps(cands: list[tuple[int, Mention]]) -> list[Mention]:
    if not cands:
        return []
    base = min(m.start for _, m in cands)
    taken = bytearray(max(m.end for _, m in cands) - base)  # occupancy: linear, not pairwise
    chosen: list[Mention] = []
    for _, m in sorted(cands, key=lambda c: (-(c[1].end - c[1].start), c[1].start, c[0], c[1].name)):
        lo, hi = m.start - base, m.end - base
        if hi <= lo or any(taken[lo:hi]):
            continue
        chosen.append(m)
        taken[lo:hi] = b"\x01" * (hi - lo)
    return sorted(chosen, key=lambda m: m.start)


def _roster_surface(match: re.Match[str]) -> str:
    """The roster name or alias a person match names, without the honorific its span includes."""
    hon = match.group("hon")
    return match.group(0)[: -len(hon)] if hon else match.group(0)


def _person(match: re.Match[str], offset: int, name: str, title: str | None = None) -> Mention:
    return Mention(NodeKind.PERSON, name, offset + match.start(), offset + match.end(), title=title)


def _people(text: str, lexicon: Lexicon, offset: int) -> list[Mention]:
    found = [
        _person(m, offset, m.group("core"), m.group("title"))
        for m in _SURNAME_TITLE_RE.finditer(text)
        if m.group("core") not in _FALSE_SURNAME_TITLE
    ]
    for m in _SPACED_SURNAME_TITLE_RE.finditer(text):
        surname, title = m.group("sur"), m.group("title")
        if surname not in _DETERMINER_SURNAMES:
            found.append(_person(m, offset, surname + title, title))
    for m in _NAME_TITLE_RE.finditer(text):
        name, title = m.group("name"), m.group("title")
        if name not in COMPOUND_SURNAMES and _plausible_name(name, lexicon):
            found.append(_person(m, offset, f"{name} {title}", title))
    for m in _NAME_EXEC_RE.finditer(text):
        name, title = m.group("name"), m.group("title")
        if name in lexicon.person_names or (len(name) >= 3 and _plausible_name(name, lexicon)):
            found.append(_person(m, offset, f"{name} {title}", title))
    for m in _NAME_HON_RE.finditer(text):
        name = m.group("name")
        if _plausible_name(name, lexicon) and not name.endswith(TITLES):
            # Keep the honorific: a bare given name is only a person *because* of 님/씨.
            found.append(_person(m, offset, name + m.group("hon")))
    return found


_NOT_A_NAME: frozenset[str] = NON_NAME_WORDS | NON_SERVICE_WORDS
#: "조치는 팀장" is a noun plus a particle, not a person: names practically never end in these.
_NON_NAME_ENDINGS: tuple[str, ...] = (
    "는", "를", "을", "에", "의", "로", "와", "과", "께", "서",
    "팀", "부", "실", "센터", "본부", "그룹", "파트",
)  # fmt: skip


def _plausible_name(name: str, lexicon: Lexicon) -> bool:
    """Whether a 2–4 syllable token before a title/honorific can be a personal name."""
    if name in lexicon.person_names:
        return True
    if name in _ROLE_NAMES or name in TITLES or lexicon.is_service(name) or name.endswith(_NON_NAME_ENDINGS):
        return False
    # a known noun, alone or followed by any particle ("담당은"), is never a name
    if name in _NOT_A_NAME or name[:-1] in _NOT_A_NAME:
        return False
    if len(name) == 3:
        return name[0] in SURNAMES
    return len(name) != 4 or name[:2] in COMPOUND_SURNAMES


def _teams(text: str, offset: int) -> list[Mention]:
    found: list[Mention] = []
    for m in _TEAM_RE.finditer(text):
        base = m.group("base")
        if base in NON_TEAM_PREFIXES or base.isdigit():
            continue
        start = m.start()
        name = base + "팀"
        if base in GENERIC_TEAM_BASES:
            qual = _TEAM_QUALIFIER_RE.search(text, 0, start)
            if (
                qual is not None
                and qual.group("qual") not in NON_SERVICE_WORDS
                and _boundary_before(text, qual.start())
            ):
                start = qual.start()
                name = f"{qual.group('qual')} {name}"
        found.append(Mention(NodeKind.TEAM, name, offset + start, offset + m.end()))
    return found


def _boundary_before(text: str, pos: int) -> bool:
    return pos == 0 or not re.match(r"[가-힣A-Za-z0-9]", text[pos - 1])


# --------------------------------------------------------------------------- services

_NOUN = r"[A-Za-z0-9가-힣][A-Za-z0-9가-힣_\-]{0,19}"
#: A single noun-like token (service names, team bases). Exported for the rule parser.
NOUN_PATTERN: str = _NOUN
_LEAD_NOISE_RE = re.compile(r"[^\w\[]*(?:(?:\d+(?:\.\d+)*[.)]?|[IVXivx]+\.)\s+)?")
#: Up to three words before 서비스/시스템 ("웹 프론트엔드 서비스"); each must be a service noun.
#: A Latin "Service"/"System" suffix is a word of its own: it needs whitespace before it (or a
#: Hangul noun glued to it, "결제Service") and a word boundary after it, so an identifier such as
#: "KubeServiceApiserverApdex" is not read as a service called "Kube".
_SERVICE_SUFFIX_RE = re.compile(
    rf"(?P<noun>{_NOUN}?(?: {_NOUN}?){{0,2}})"
    r"(?:\s*(?:서비스|시스템)|(?:(?<=[가-힣])|\s+)(?:[Ss]ervice|[Ss]ystem)(?![A-Za-z0-9]))(?!팀)"
)
_INCIDENT_SUFFIX_RE = re.compile(rf"(?P<noun>{_NOUN}?)\s+(?:장애|문제)")
#: "[결제 서비스]" header. A bracket followed by "(" or "[" is markdown link text ("[title](url)",
#: "[title][ref]"): it names another page, not the service a section is about.
_BRACKET_INNER_RE = re.compile(r"\[(?P<inner>[^\]\n]{1,40})\](?![(\[])")
_JOSA_END_RE = re.compile(r"(?:은|는|을|를|에|의|에서|으로|가)$")


def is_service_noun(noun: str, lexicon: Lexicon) -> bool:
    """Whether ``noun`` can name a service: not generic, not a person/channel/action/team word."""
    if lexicon.is_service(noun):
        return True
    if not noun or len(noun) > 20 or noun in NON_SERVICE_WORDS or noun.endswith("팀"):
        return False
    if not re.fullmatch(_NOUN, noun) or noun.isdigit():
        return False
    if _JOSA_END_RE.search(noun) and len(noun) > 2:
        return False
    return not any(m.kind not in (NodeKind.SERVICE, NodeKind.SYSTEM) for m in find_mentions(noun, lexicon))


def service_in_label(
    text: str, lexicon: Lexicon, *, offset: int = 0, incident_suffix: bool = True
) -> Mention | None:
    """The service a heading / bracket header / table label names, if it names one.

    Order: a catalog name or alias anywhere in the label; a bracketed label ("[결제 서비스]");
    a leading noun followed by 서비스/시스템 ("결제 서비스 장애 대응"); a leading noun followed
    by 장애 ("결제 장애 대응"). Generic labels ("공통 연락망", "보안 사고 대응") name nothing.
    ``incident_suffix=False`` disables the last form — a document *title* such as
    "DMC샵 장애 대응 런북" names the organisation, not one service.
    """
    for mention in find_mentions(text, lexicon):
        if mention.kind in (NodeKind.SERVICE, NodeKind.SYSTEM):
            return Mention(mention.kind, mention.name, offset + mention.start, offset + mention.end)
    bracket = _BRACKET_INNER_RE.search(text)
    if bracket is not None:
        inner = service_in_label(
            bracket.group("inner"),
            lexicon,
            offset=offset + bracket.start("inner"),
            incident_suffix=incident_suffix,
        )
        if inner is not None:
            return inner
    lead = _LEAD_NOISE_RE.match(text)
    pos = lead.end() if lead is not None else 0
    patterns = (_SERVICE_SUFFIX_RE, _INCIDENT_SUFFIX_RE) if incident_suffix else (_SERVICE_SUFFIX_RE,)
    for rx in patterns:
        m = rx.match(text, pos)
        if m is not None and _service_phrase(m.group("noun"), lexicon):
            return Mention(
                NodeKind.SERVICE, m.group("noun"), offset + m.start("noun"), offset + m.end("noun")
            )
    return None


def _service_phrase(phrase: str, lexicon: Lexicon) -> bool:
    """A one-to-three word service name: every word a service noun and, read as a whole, no
    person/channel/action ("사내 메신저 시스템" names a channel, not a service)."""
    words = phrase.split(" ")
    if len(words) == 1:
        return is_service_noun(phrase, lexicon)
    return all(is_service_noun(w, lexicon) for w in words) and not any(
        m.kind not in (NodeKind.SERVICE, NodeKind.SYSTEM) for m in find_mentions(phrase, lexicon)
    )


def service_from_cell(text: str, lexicon: Lexicon, *, offset: int = 0) -> Mention | None:
    """A table cell under a 서비스 column: catalog match, else the cell minus 서비스/시스템.

    The column header already states that the cell is a service, so any short, non-generic
    label is accepted (still never a person/channel/action word).
    """
    labelled = service_in_label(text, lexicon, offset=offset)
    if labelled is not None:
        return labelled
    m = re.fullmatch(r"(?P<name>[A-Za-z0-9가-힣][A-Za-z0-9가-힣_\-./ ]{0,29}?)\s*(?:서비스|시스템)?", text)
    if m is None:
        return None
    name = " ".join(m.group("name").split())
    if name in NON_SERVICE_WORDS or any(
        mm.kind not in (NodeKind.SERVICE, NodeKind.SYSTEM) for mm in find_mentions(name, lexicon)
    ):
        return None
    return Mention(NodeKind.SERVICE, name, offset + m.start("name"), offset + m.end("name"))


def recipients_in(text: str, offset: int = 0) -> list[tuple[str, int, int]]:
    """Reporting authorities mentioned in ``text`` as (label, start, end), document order."""
    found: list[tuple[str, int, int]] = []
    for pattern, label in RECIPIENT_PATTERNS:
        found += [(label, offset + m.start(), offset + m.end()) for m in re.finditer(pattern, text)]
    return sorted(found, key=lambda r: r[1])


__all__ = [
    "ACTION_LABEL",
    "COMPOUND_SURNAMES",
    "DURATION_PATTERN",
    "JOSA",
    "NON_NAME_WORDS",
    "NON_SERVICE_WORDS",
    "NOUN_PATTERN",
    "SURNAMES",
    "TITLES",
    "Duration",
    "Lexicon",
    "Mention",
    "channel_reading",
    "find_delay",
    "find_durations",
    "find_mentions",
    "incident_type_of",
    "incident_word_span",
    "is_service_noun",
    "names_channel",
    "parse_minutes",
    "recipients_in",
    "service_from_cell",
    "service_in_label",
]
