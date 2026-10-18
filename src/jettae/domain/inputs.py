"""Inputs of a diagnosis: documents (runbooks) and optional structured sources.

Structured sources are *trusted facts* about the organisation (who is employed, what services
exist, who is on call). Runbook text is *untrusted data*: it is parsed, never obeyed.
"""

from __future__ import annotations

import bisect
import hashlib
from datetime import datetime
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, computed_field, field_validator

from jettae.domain.graph import ChannelMedium, IncidentType, PersonStatus


class Document(BaseModel):
    """A normalized text document. All :class:`SourceSpan` offsets index into ``text``."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    title: str
    text: str
    media_type: str = "text/markdown"
    path: str | None = None
    #: char ranges excluded from extraction (prompt-injection quarantine). PII is masked in place,
    #: length-preserving, and is NOT quarantined.
    quarantined: tuple[tuple[int, int], ...] = ()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    _line_starts: list[int] | None = PrivateAttr(default=None)

    def line_of(self, offset: int) -> int:
        """1-based line number of a character offset (line table cached per document)."""
        if self._line_starts is None:
            self._line_starts = [0] + [i + 1 for i, ch in enumerate(self.text) if ch == "\n"]
        return bisect.bisect_right(self._line_starts, offset)

    def is_quarantined(self, start: int, end: int) -> bool:
        return any(start < q_end and end > q_start for q_start, q_end in self.quarantined)

    @staticmethod
    def make_id(title: str, text: str) -> str:
        h = hashlib.sha256(f"{title}\x00{text}".encode()).hexdigest()[:12]
        return f"doc-{h}"


class RosterEntry(BaseModel):
    """One person in the HR roster / org directory (명부)."""

    model_config = ConfigDict(frozen=True)

    name: str
    status: PersonStatus = PersonStatus.ACTIVE
    title: str | None = None  # 직급·직책 (대리, 과장, 팀장, CTO …)
    team: str | None = None
    roles: tuple[str, ...] = ()  # 역할 (팀장, 보안 담당, CISO, 1차 당직 …)
    aliases: tuple[str, ...] = ()  # 별칭 (영문명, 닉네임, '김대리' 등)
    channels: tuple[ChannelMedium, ...] = ()  # 등록된 연락 수단
    left_on: str | None = None  # 퇴사일 (YYYY-MM-DD)


class OnCallShift(BaseModel):
    """One on-call shift. ``start``/``end`` are timezone-aware (ingest assigns Asia/Seoul to naive input)."""

    model_config = ConfigDict(frozen=True)

    person: str  # roster name
    start: datetime
    end: datetime
    level: int = 1  # 1 = primary, 2 = secondary


class FatiguePolicy(BaseModel):
    """Team-agreed on-call limits (팀 합의 기준). JETTAE flags violations; it does NOT judge
    whether waiting time is legally working time (대법원 2024.11 판단 기준 참고만)."""

    model_config = ConfigDict(frozen=True)

    max_consecutive_night_shifts: int = 2
    min_rest_hours: float = 11.0
    night_start_hour: int = 22
    night_end_hour: int = 6
    max_shifts_per_week: int = 3


#: Jurisdiction of the statutory defaults (:data:`DEFAULT_OBLIGATIONS`) and of a default policy.
KOREA: Final = "KR"


def _jurisdiction_code(raw: str) -> str:
    """``" kr "`` -> ``"KR"``: jurisdictions compare as trimmed upper-case codes (ISO 3166-1 style)."""
    return raw.strip().upper()


class ObligationRule(BaseModel):
    """A reporting/notification obligation the organisation must be able to meet."""

    model_config = ConfigDict(frozen=True)

    key: str  # e.g. "kr-breach-24h"
    incident_type: IncidentType
    recipient: str
    deadline_minutes: int
    legal_basis: str
    applies: bool = True  # org declares whether it is in scope (e.g. 정보통신서비스 제공자)
    #: Legal system a statutory duty comes from ("KR" = 대한민국 법령). ``None`` = the
    #: organisation's own duty (e.g. a customer SLA notice), in force wherever it operates.
    jurisdiction: str | None = None

    @field_validator("jurisdiction", mode="before")
    @classmethod
    def _code(cls, value: object) -> object:
        if isinstance(value, str):
            return _jurisdiction_code(value) or None
        return value


DEFAULT_OBLIGATIONS: tuple[ObligationRule, ...] = (
    ObligationRule(
        key="kr-breach-24h",
        incident_type=IncidentType.BREACH,
        recipient="과학기술정보통신부·한국인터넷진흥원(KISA)",
        deadline_minutes=24 * 60,
        legal_basis=(
            "정보통신망법 제48조의3 제1항(침해사고의 신고 등, 법률 제21500호·2026.10.1 시행): "
            "침해사고를 알게 된 때부터 24시간 이내 과학기술정보통신부장관 또는 KISA에 신고, "
            "추가 확인 사항은 확인한 때부터 24시간 이내 보완 신고(시행령 제58조의8)"
        ),
        jurisdiction=KOREA,
    ),
    ObligationRule(
        key="kr-privacy-72h",
        incident_type=IncidentType.PRIVACY_LEAK,
        recipient="개인정보보호위원회·한국인터넷진흥원(KISA)",
        deadline_minutes=72 * 60,
        legal_basis=(
            "개인정보 보호법 제34조 제1항·제4항, 시행령 제39조(통지)·제40조(신고): "
            "유출등을 알게 된 때부터 72시간 이내 정보주체 통지, 1천명 이상·민감/고유식별정보·"
            "외부 불법접근 시 개인정보보호위원회 또는 KISA 신고"
        ),
        jurisdiction=KOREA,
    ),
)


class Policy(BaseModel):
    """Organisation policy + scoring knobs. Defaults are published in docs/scoring.md."""

    model_config = ConfigDict(frozen=True)

    critical_tiers: tuple[int, ...] = (1, 2)
    tier_slo_minutes: dict[int, int] = Field(default_factory=lambda: {1: 60, 2: 240, 3: 1440})
    default_tier: int = 2
    fatigue: FatiguePolicy = FatiguePolicy()
    obligations: tuple[ObligationRule, ...] = DEFAULT_OBLIGATIONS
    #: roles that terminate an escalation chain (최종 책임자)
    terminal_roles: tuple[str, ...] = ("CTO", "대표", "CEO", "CISO")
    #: services in scope of breach / privacy obligations (empty = all critical services)
    regulated_services: tuple[str, ...] = ()
    #: Jurisdictions whose statutory duties bind the organisation ("KR", "US" …). A duty of
    #: another jurisdiction (e.g. the Korean 24h/72h reports for a company operating only in the
    #: US) is not checked; the organisation's own duties (``jurisdiction=None``) always are.
    #: Accepts a single code in a policy file (``jurisdictions: US``).
    jurisdictions: tuple[str, ...] = (KOREA,)

    @field_validator("jurisdictions", mode="before")
    @classmethod
    def _codes(cls, value: object) -> object:
        items = [value] if isinstance(value, str) else value
        if isinstance(items, list | tuple):
            codes = (_jurisdiction_code(v) if isinstance(v, str) else v for v in items)
            return tuple(dict.fromkeys(c for c in codes if c != ""))
        return value

    def in_force(self, rule: ObligationRule) -> bool:
        """Whether ``rule`` binds this organisation: declared in scope and of one of its
        jurisdictions (or an own duty without one). The single rule every check and replay uses."""
        return rule.applies and (rule.jurisdiction is None or rule.jurisdiction in self.jurisdictions)

    def obligations_in_force(self, incident_type: IncidentType | None = None) -> tuple[ObligationRule, ...]:
        """:attr:`obligations` that :meth:`in_force` keeps, optionally of one incident type."""
        return tuple(
            r
            for r in self.obligations
            if self.in_force(r) and (incident_type is None or r.incident_type is incident_type)
        )


class CatalogService(BaseModel):
    """A service/system from a catalog (Backstage, k8s manifests, OTel service graph …)."""

    model_config = ConfigDict(frozen=True)

    name: str
    kind: str = "service"  # "service" | "system"
    tier: int | None = None
    aliases: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    location: str | None = None  # datacenter / region / SaaS vendor
    owner: str | None = None  # declared owner (team or person)
    description: str | None = None


class InputBundle(BaseModel):
    """Everything a diagnosis run consumes."""

    model_config = ConfigDict(frozen=True)

    documents: tuple[Document, ...]
    roster: tuple[RosterEntry, ...] = ()
    oncall: tuple[OnCallShift, ...] = ()
    catalog: tuple[CatalogService, ...] = ()
    policy: Policy = Policy()


class LlmMode(StrEnum):
    OFF = "off"  # 규칙 파서만 (LLM 호출 0회)
    LIVE = "live"  # 실제 LLM 호출
    REPLAY = "replay"  # 녹화된 LLM 응답 재생 (데모·회귀 테스트, 결정적)
