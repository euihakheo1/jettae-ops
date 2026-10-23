"""Evidence view model shared by the Markdown and HTML evidence reports (ISMS-P 증적 초안).

Why a separate view model instead of letting each renderer read the domain objects: the Markdown
and HTML reports are two *presentations of the same evidence*. If each renderer grouped findings,
picked the "current" rehearsal result or mapped certification controls on its own, the two files
handed to an auditor could disagree. :func:`build_evidence` makes every such decision exactly once;
the renderers only escape and lay out the resulting strings.

Everything here is a pure function of its inputs (P6): no clock, no randomness, no locale, and every
collection is sorted by a total key, so the same report renders byte-for-byte identically. Nothing
here judges anything (P1) — verdicts, scores and rehearsal outcomes are copied from the report that
the fixed rules and the deterministic replay produced. Unknown stays unknown (P3): a missing value is
shown as missing, never as a pass.
"""

from __future__ import annotations

import math
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Final

from jettae.check.registry import rule_catalog, sort_findings
from jettae.check.scoring import SCORING_V1
from jettae.domain import (
    ELEMENT_LABEL_KO,
    AiProposalStatus,
    DiagnosisReport,
    DocumentRef,
    EdgeKind,
    Element,
    ElementStatus,
    ExplorationSummary,
    FailingSubset,
    Finding,
    FindingStatus,
    FixKind,
    GraphVersion,
    GroundingStatus,
    HypothesisMinimization,
    HypothesisOrigin,
    HypothesisStatus,
    IncidentType,
    LlmMode,
    LlmUsageSummary,
    NodeKind,
    OpsGraph,
    Outcome,
    Patch,
    PatchAuthor,
    PatchStatus,
    Perturbation,
    PerturbationKind,
    ReadinessReport,
    RehearsalResult,
    RiskHypothesis,
    Scenario,
    SecurityNotice,
    Severity,
    SourceSpan,
)

__all__ = [
    "CHANGED_DRILL_NOTE",
    "CONTROLS_INTRO",
    "DISCLAIMER",
    "DROPPED_EXAMPLES_TITLE",
    "ELEMENT_STATUS_LABEL_KO",
    "ELEMENT_STATUS_SYMBOL",
    "EVIDENCE_TITLE",
    "EXPLORE_INTRO",
    "EXPLORE_NOTHING_FOUND",
    "FINDING_STATUS_LABEL_KO",
    "HISTORY_NOTE",
    "HYPOTHESIS_ORIGIN_LABEL_KO",
    "HYPOTHESIS_STATUS_LABEL_KO",
    "INJECTION_PATTERN_LABEL_KO",
    "KST",
    "MINIMIZE_LABEL",
    "MINIMIZE_SCOPE",
    "NO_QUOTE_NOTE",
    "OUTCOME_LABEL_KO",
    "RULES_INTRO",
    "SCORING_INTRO",
    "SCORING_METHOD",
    "SECTION_TITLES",
    "SECURITY_EXCERPT_CHARS",
    "SECURITY_INTRO",
    "SEVERITY_LABEL_KO",
    "STATUS_LEGEND",
    "TIER_FALLBACK_NOTE",
    "UNDECIDED_INTRO",
    "UNLINKED_PATCHES_TITLE",
    "ControlRow",
    "ControlStatus",
    "DocumentRow",
    "EvidenceQuote",
    "EvidenceView",
    "FindingView",
    "HeldRow",
    "HypothesisRow",
    "PatchRow",
    "RehearsalRow",
    "RuleRow",
    "SecurityRow",
    "ServiceRow",
    "UnknownElementRow",
    "VersionRow",
    "WeightRow",
    "build_evidence",
    "current_rehearsals",
    "distinct_spans",
    "format_datetime",
    "format_minutes",
    "hypothesis_counts",
    "judgement_path_statement",
    "llm_usage_line",
    "mask_excerpt",
    "printable",
    "service_rows",
    "short_hash",
]

EVIDENCE_TITLE: Final = "런북 준비도 진단 리포트 (ISMS-P 증적 초안)"

#: Korea has observed no daylight saving time since 1988, so a fixed +09:00 offset is exact and
#: keeps rendering independent of whether the host ships the IANA tz database (Windows does not).
KST: Final = timezone(timedelta(hours=9), "KST")

#: How much of a quarantined instruction the report may show. Enough for a reviewer to recognise
#: the attempt, too little to carry a working payload into another tool that reads the report.
SECURITY_EXCERPT_CHARS: Final = 60

#: Long evidence quotes are clipped for display; the location (document:line) stays exact.
QUOTE_DISPLAY_CHARS: Final = 300

#: Related elements listed under a finding; the rest is summarised as "외 N개".
SUBJECT_DISPLAY_LIMIT: Final = 6

HASH_PREFIX_CHARS: Final = 12

SEVERITY_LABEL_KO: Final[dict[Severity, str]] = {
    Severity.CRITICAL: "심각",
    Severity.HIGH: "높음",
    Severity.MEDIUM: "보통",
    Severity.LOW: "낮음",
    Severity.INFO: "정보",
}

FINDING_STATUS_LABEL_KO: Final[dict[FindingStatus, str]] = {
    FindingStatus.BROKEN: "끊김",
    FindingStatus.UNKNOWN: "확인 불가",
}

ELEMENT_STATUS_SYMBOL: Final[dict[ElementStatus, str]] = {
    ElementStatus.PASS: "●",
    ElementStatus.PARTIAL: "◐",
    ElementStatus.FAIL: "○",
    ElementStatus.UNKNOWN: "?",
}

ELEMENT_STATUS_LABEL_KO: Final[dict[ElementStatus, str]] = {
    ElementStatus.PASS: "충족",
    ElementStatus.PARTIAL: "부분 충족",
    ElementStatus.FAIL: "끊김",
    ElementStatus.UNKNOWN: "확인 불가(0점)",
}

OUTCOME_LABEL_KO: Final[dict[Outcome, str]] = {Outcome.PASS: "통과", Outcome.FAIL: "실패"}

LLM_MODE_LABEL_KO: Final[dict[LlmMode, str]] = {
    LlmMode.OFF: "끔(규칙 파서만 사용)",
    LlmMode.LIVE: "실시간 호출",
    LlmMode.REPLAY: "녹화 재생(결정적)",
}

TIER_LABEL_KO: Final[dict[int, str]] = {1: "핵심", 2: "중요", 3: "일반"}

INCIDENT_TYPE_LABEL_KO: Final[dict[IncidentType, str]] = {
    IncidentType.OUTAGE: "서비스 장애",
    IncidentType.BREACH: "침해사고",
    IncidentType.PRIVACY_LEAK: "개인정보 유출",
}

PATCH_STATUS_LABEL_KO: Final[dict[PatchStatus, str]] = {
    PatchStatus.DRAFT: "초안(승인 대기)",
    PatchStatus.VALIDATED: "검증 통과(승인 대기)",
    PatchStatus.REJECTED_BY_VALIDATOR: "검증 실패",
    PatchStatus.APPROVED: "승인",
    PatchStatus.REJECTED: "거절",
    PatchStatus.SUPERSEDED: "무효(기준 버전 변경)",
}

#: Risk hypotheses (규칙 밖 위험 후보): the replay's verdict, worded for the reader.
HYPOTHESIS_STATUS_LABEL_KO: Final[dict[HypothesisStatus, str]] = {
    HypothesisStatus.CONFIRMED: "확인됨(리허설로 재현)",
    HypothesisStatus.NEEDS_INFO: "확인 필요",
    HypothesisStatus.REFUTED: "기각",
}

HYPOTHESIS_ORIGIN_LABEL_KO: Final[dict[HypothesisOrigin, str]] = {
    HypothesisOrigin.SWEEP: "자동 전수 탐색",
    HypothesisOrigin.AI: "생성형 AI 제안",
}

PERTURBATION_SUFFIX_KO: Final[dict[PerturbationKind, str]] = {
    PerturbationKind.PERSON_UNAVAILABLE: "부재",
    PerturbationKind.CHANNEL_DOWN: "다운",
    PerturbationKind.SYSTEM_DOWN: "장애",
    PerturbationKind.LOCATION_DOWN: "센터 장애",
    PerturbationKind.CONCURRENT_INCIDENT: "동시 장애",
}

PATCH_AUTHOR_LABEL_KO: Final[dict[PatchAuthor, str]] = {
    PatchAuthor.AI: "생성형 AI 초안",
    PatchAuthor.TEMPLATE: "템플릿 초안",
    PatchAuthor.HUMAN: "사람 작성",
}

FIX_KIND_LABEL_KO: Final[dict[FixKind, str]] = {
    FixKind.ASSIGN_OWNER: "활동 중인 담당자 지정",
    FixKind.REPLACE_DEPARTED: "퇴사자를 재직자로 교체",
    FixKind.ADD_ESCALATION: "상향 경로 추가",
    FixKind.BREAK_ESCALATION_CYCLE: "상향 순환 끊기(최종 책임자 연결)",
    FixKind.ASSIGN_APPROVER: "승인자 지정",
    FixKind.ADD_DEPUTY: "부재 시 대행자 지정",
    FixKind.ADD_FALLBACK_CHANNEL: "대체 연락 수단 추가",
    FixKind.ADD_ROLLBACK: "롤백·완화 조치 추가",
    FixKind.ADD_DETECTION: "감지 신호·호출 대상 추가",
    FixKind.FIX_DEADLINE: "시한을 기준에 맞게 수정",
    FixKind.ADD_REPORTING_STEP: "신고·통지 단계 추가",
    FixKind.ADJUST_ONCALL: "당직표 조정",
    FixKind.ADD_SERVICE_RUNBOOK: "서비스 런북·담당 추가",
    FixKind.CONFIRM_IDENTITY: "인물 확인(명부 대조)",
    FixKind.ASSIGN_EXECUTOR: "실행자 지정",
    FixKind.ADD_FAILOVER: "페일오버 마련",
    FixKind.REVIEW_ELEMENT: "사람이 원문과 대조해 확인",
}

#: Labels for ``SecurityNotice.pattern`` ids emitted by ``jettae.ingest.guard``. Ids the guard adds
#: later are shown verbatim rather than hidden, so a new detector never makes a notice disappear.
INJECTION_PATTERN_LABEL_KO: Final[dict[str, str]] = {
    "ignore-instructions": "이전 지시 무시 요구",
    "system-prompt": "시스템 프롬프트 언급·유출 요구",
    "role-reassignment": "역할 바꾸기 요구",
    "as-an-ai": "AI 화법 흉내",
    "addressed-to-ai": "AI를 향한 지시",
    "score-manipulation": "점수 조작 지시",
    "verdict-override": "판정 덮어쓰기 지시",
    "finding-suppression": "끊김 숨기기 지시",
    "graph-tamper": "그래프 변경 지시",
    "fake-role-tag": "가짜 역할 태그",
    "tool-call-payload": "도구 호출 위장",
    "hidden-content": "숨김 텍스트(HTML 주석 등)",
    "new-instructions": "새 지시 주입",
    "jailbreak": "탈옥 시도",
    "encoded-payload": "인코딩된 숨은 내용",
    "link-title-instruction": "링크 제목 속 지시",
    "role-tag-block": "가짜 역할 블록",
}

NODE_KIND_LABEL_KO: Final[dict[NodeKind, str]] = {
    NodeKind.SERVICE: "서비스",
    NodeKind.SYSTEM: "시스템",
    NodeKind.LOCATION: "위치",
    NodeKind.PERSON: "인물",
    NodeKind.ROLE: "역할",
    NodeKind.TEAM: "팀",
    NodeKind.CHANNEL: "연락 수단",
    NodeKind.TRIGGER: "감지 신호",
    NodeKind.ACTION: "조치",
    NodeKind.OBLIGATION: "신고 의무",
}

EDGE_KIND_LABEL_KO: Final[dict[EdgeKind, str]] = {
    EdgeKind.DEPENDS_ON: "의존",
    EdgeKind.RUNS_IN: "위치",
    EdgeKind.OWNED_BY: "담당",
    EdgeKind.DETECTED_BY: "감지",
    EdgeKind.NOTIFIES: "호출",
    EdgeKind.ESCALATES_TO: "상향",
    EdgeKind.MITIGATED_BY: "완화 조치",
    EdgeKind.APPROVED_BY: "승인",
    EdgeKind.EXECUTED_BY: "실행",
    EdgeKind.REACHABLE_VIA: "연락처",
    EdgeKind.FALLBACK_TO: "대체 채널",
    EdgeKind.HOLDS_ROLE: "역할 보유",
    EdgeKind.MEMBER_OF: "소속",
    EdgeKind.MUST_REPORT: "신고 의무",
    EdgeKind.REPORTED_BY: "신고 담당",
}

#: Rule-id families relevant to legal notification (ISMS-P 2.11.5) and to disaster recovery
#: (ISMS-P 2.12.1). Legal families are prefixes, so rules added to a family are mapped automatically.
LEGAL_RULE_PREFIXES: Final = ("JT-KR-", "JT-DDL-")
DISASTER_RULE_IDS: Final = ("JT-CAT-003", "JT-CHN-001")

# --------------------------------------------------------------------------- shared wording
# Every output format prints these sentences verbatim, so an auditor comparing the Markdown and the
# printed HTML never finds two phrasings of the same statement.

DISCLAIMER: Final = (
    "제때옵스(JETTAE Ops)가 고정 규칙과 결정적 리플레이로 자동 생성한 증적 초안입니다. "
    "판정·점수 계산과 리허설 통과/실패 판정에는 생성형 AI를 쓰지 않으며"
    "(LLM을 켠 진단에서는 AI가 리허설 시나리오를 제안하고, 역번역 대조에서 연결을 판단 보류로 돌릴 수 있음), "
    "인증 심사에 제출하기 전에 담당자가 내용을 확인해야 합니다."
)
NO_QUOTE_NOTE: Final = (
    "근거 원문: 인용할 문장이 없습니다. 런북에 해당 내용이 없다는 것"
    "(또는 명부·카탈로그 대조 결과) 자체가 판정 근거입니다."
)
UNDECIDED_INTRO: Final = (
    "확인할 수 없는 항목은 정상으로 치지 않습니다. 점수에서는 0점으로 계산했고, "
    "사람이 원문·명부와 대조해야 합니다."
)
SECURITY_INTRO: Final = (
    "문서 속 AI를 향한 지시문은 격리해 추출·판정에 쓰지 않았습니다. "
    "점수와 판정에는 영향이 없습니다. "
    f"발췌는 앞 {SECURITY_EXCERPT_CHARS}자까지만 싣습니다."
)
CONTROLS_INTRO: Final = (
    "ISMS-P 인증 기준별로 이 리포트에서 증적으로 쓸 수 있는 부분입니다. "
    "인증 심사의 판단을 대신하지 않으며, 제출 전에 담당자가 검토해야 합니다."
)
HISTORY_NOTE: Final = "수정안은 사람이 승인한 것만 새 그래프 버전이 되며, 되돌리기도 새 버전으로 남습니다."
#: Patches that no listed version points to. Not "반영되지 않은": when the caller passes no version
#: history, an approved patch lands here although it did produce a version.
UNLINKED_PATCHES_TITLE: Final = "버전 기록과 연결되지 않은 수정안"
#: The examples carry every drop reason (근거 없음, 격리 구간 인용, 형식 오류 …), not only "no source".
DROPPED_EXAMPLES_TITLE: Final = "추출 단계에서 버린 요소 예시"
SCORING_INTRO: Final = (
    "점수는 고정 규칙으로만 계산하며 계산에 생성형 AI를 쓰지 않습니다. "
    "LLM을 켠 진단에서 역번역 대조의 AI 판정으로 판단 보류된 연결은 충족 근거로 쓰지 않습니다. "
    "같은 그래프에는 언제나 같은 점수가 나옵니다."
)
SCORING_METHOD: Final = (
    "서비스 점수 = 100 × Σ(요소 가중치 × 요소 점수), 소수 첫째 자리 반올림(5 이상 올림)",
    # "핵심 서비스" is the policy's critical tiers (default 1·2), not the tier-1 label "핵심" in the
    # tables: without saying so, a reader who averages only the "1(핵심)" rows cannot reproduce it.
    "전체 점수 = 핵심 서비스(정책이 정한 핵심 등급, 기본값 1·2등급)의 서비스 점수를 등급 가중치로 "
    "가중 평균해 소수 첫째 자리 반올림 (핵심 서비스가 없으면 전체 서비스, 서비스가 없으면 0)",
    "요소 점수는 0~1이며, 확인 불가는 0점입니다",
)
TIER_FALLBACK_NOTE: Final = "표에 없는 등급은 가장 낮은 등급 가중치를 씁니다."
RULES_INTRO: Final = "판정에 쓰는 고정 규칙 전체입니다. 고객이 판정 기준을 직접 감사할 수 있도록 공개합니다."
EXPLORE_INTRO: Final = (
    "고정 규칙이 묻지 않는 위험(두 사람이 함께 빠질 때, 연락 수단 두 개가 함께 멈출 때, 다른 서비스 장애와 "
    "겹칠 때 등)을 자동 전수 탐색과 생성형 AI가 가설로 제안하고, 결정적 리허설이 서비스 목표 복구 시간으로 "
    "확인한 결과입니다. 준비도 점수·끊김 판정·통과 기준에는 반영하지 않습니다. 그래프에 없는 전제가 붙은 "
    "가설은 확인되기 전까지 '확인 필요'로 남습니다."
)
#: 9장, nothing to list and every critical service fully searched.
EXPLORE_NOTHING_FOUND: Final = "고정 규칙이 보고한 것 밖에서 대응을 끊는 조합을 찾지 못했습니다."
#: How a model's explanation is labelled: only the listed perturbations were replayed.
AI_EXPLANATION_PREFIX: Final = (
    "AI 설명(검증하지 않음 — 리허설이 확인한 것은 '함께 일어나는 일'의 교란뿐입니다): "
)
#: 최소 실패 조건 자동 검증 of a confirmed AI hypothesis — what it is called, and what it is not.
MINIMIZE_LABEL: Final = "최소 실패 조건(자동 검증)"
MINIMIZE_SCOPE: Final = (
    "AI가 제안한 조건의 모든 부분 조합을 같은 서비스·같은 목표 시간으로 다시 리허설해, 필요 없는 조건만 "
    "덜어 낸 결과입니다. 제안에 없는 다른 원인까지 찾지는 않습니다."
)
#: Why a run has no AI proposals, by status (USED and OFF are worded from the counts and the mode).
_AI_STATUS_LINE: Final[dict[AiProposalStatus, str]] = {
    AiProposalStatus.FAILED: "생성형 AI 제안 실패(사유는 아래 참고) — 자동 전수 탐색 결과만 실음",
    AiProposalStatus.FALLBACK: "생성형 AI 제안 없음(추출 단계에서 LLM이 실패해 이 실행은 LLM을 더 쓰지 않음)",
    AiProposalStatus.NOT_REQUESTED: "생성형 AI 제안을 요청하지 않음(제안 개수 0 또는 제안받을 서비스 없음)",
    AiProposalStatus.REJUDGE: "생성형 AI 제안 없음(승인·되돌리기 뒤 재판정은 LLM을 쓰지 않음)",
}
_AI_OFF_REASON: Final[dict[LlmMode, str]] = {
    LlmMode.OFF: "LLM 끔",
    LlmMode.REPLAY: "녹화된 응답에 없는 요청",
}
#: Chapter titles by key, in report order (the HTML table of contents and both formats' headings).
SECTION_TITLES: Final[dict[str, str]] = {
    "overview": "1. 개요",
    "services": "2. 서비스별 준비도",
    "broken": "3. 끊긴 대응 경로",
    "rehearsals": "4. 리허설 결과",
    "history": "5. 개정 이력",
    "undecided": "6. 판단 보류·확인 불가 항목",
    "security": "7. 보안 알림",
    "controls": "8. 인증 기준 매핑",
    "explore": "9. 규칙 밖 위험 후보 (점수에 반영하지 않음)",
    "scoring": "부록 A. 점수 산식",
    "rules": "부록 B. 규칙 카탈로그",
}
STATUS_LEGEND: Final = (
    "범례: "
    + " · ".join(f"{ELEMENT_STATUS_SYMBOL[s]} {label}" for s, label in ELEMENT_STATUS_LABEL_KO.items())
    + " · — 평가 기록 없음"
)


# --------------------------------------------------------------------------- formatting helpers


def printable(text: str) -> str:
    """Drop control and invisible format characters; whitespace becomes a plain space.

    Report text comes from runbooks, rosters and LLM-drafted sentences (P5). An ANSI escape would
    recolour or erase a terminal, and a bidi override (U+202E) can make a quoted sentence read
    differently from what it says. Lone surrogates (from undecodable bytes) become U+FFFD so the
    report can always be encoded as UTF-8. Ordinary text takes the fast path unchanged.
    """
    if text.isprintable():
        return text
    out: list[str] = []
    for ch in text:
        if ch.isspace():
            out.append(" ")
            continue
        category = unicodedata.category(ch)
        if category in ("Cc", "Cf"):
            continue
        out.append("�" if category == "Cs" else ch)
    return "".join(out)


def short_hash(value: str) -> str:
    """The first 12 hex digits — enough to match a version by eye, short enough for a table."""
    return value[:HASH_PREFIX_CHARS]


def format_datetime(value: datetime) -> str:
    """Render in Korean Standard Time; a naive timestamp is printed as-is and marked as such.

    Converting a naive value would silently assume a zone (P3), so it is labelled instead.
    """
    if value.tzinfo is None or value.utcoffset() is None:
        return f"{value:%Y-%m-%d %H:%M:%S} (시간대 미기재)"
    return f"{value.astimezone(KST):%Y-%m-%d %H:%M:%S} KST"


def format_minutes(minutes: float | None) -> str:
    """``90`` → ``1시간 30분``, ``1440`` → ``24시간``; non-finite or missing → ``미정``."""
    if minutes is None or not math.isfinite(minutes):
        return "미정"
    sign = "-" if minutes < 0 else ""
    value = abs(minutes)
    if value != int(value):
        return f"{sign}{value:.1f}분"
    hours, rest = divmod(int(value), 60)
    if hours == 0:
        return f"{sign}{rest}분"
    return f"{sign}{hours}시간" + (f" {rest}분" if rest else "")


def mask_excerpt(text: str, limit: int = SECURITY_EXCERPT_CHARS) -> str:
    """Collapse whitespace, drop invisible characters and clip to ``limit`` characters (ellipsis
    included), so a quarantined instruction is recognisable but never shown whole."""
    flat = " ".join(printable(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _one_line(text: str | None) -> str | None:
    """Caller-supplied labels (organisation name) as one clean line; blank means absent."""
    if text is None:
        return None
    return " ".join(printable(text).split()) or None


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _percent(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "해당 없음"


def rule_id_of(finding_id: str) -> str:
    """``JT-PPL-001:3fa9…`` → ``JT-PPL-001`` (see :meth:`Finding.make_id`)."""
    return finding_id.rsplit(":", 1)[0] if ":" in finding_id else finding_id


def llm_usage_line(usage: LlmUsageSummary) -> str:
    """One-line LLM usage disclosure (KPI: 1건당 LLM 호출 수·토큰 공개)."""
    parts = [LLM_MODE_LABEL_KO[usage.mode]]
    model = " / ".join(x for x in (usage.provider, usage.model) if x)
    if model:
        parts.append(model)
    parts.append(f"호출 {usage.calls:,}회")
    parts.append(
        f"토큰 입력 {usage.input_tokens:,} · 출력 {usage.output_tokens:,} · "
        f"캐시 읽기 {usage.cache_read_tokens:,}"
    )
    if usage.by_task:
        parts.append("작업별 " + ", ".join(f"{task} {n:,}회" for task, n in sorted(usage.by_task.items())))
    return " · ".join(parts)


def judgement_path_statement(usage: LlmUsageSummary) -> str:
    """The P1 disclosure. A non-zero count is printed as a violation, never rounded away."""
    calls = usage.judgement_path_calls
    if calls == 0:
        return "판정·점수·리허설 경로 LLM 호출 0회 (고정 규칙과 결정적 리플레이만 사용)"
    return (
        f"판정·점수·리허설 경로 LLM 호출 {calls:,}회 — 설계 원칙(판정은 규칙) 위반이므로 "
        "이 리포트를 증적으로 쓰지 마십시오"
    )


def distinct_spans(spans: Iterable[SourceSpan]) -> list[SourceSpan]:
    """Evidence spans in document order, dropping spans contained in another kept span.

    Rules often cite both a sentence and a word inside it ('김대리에게 연락' and '김대리'); showing
    both repeats the same evidence without adding information. Shared by every output format so
    the Markdown, HTML and SARIF reports cite exactly the same locations.
    """
    ordered = sorted(set(spans), key=lambda s: (s.doc_id, s.start, -s.end, s.quote, s.line or 0))
    kept: list[SourceSpan] = []
    for span in ordered:
        if any(k.doc_id == span.doc_id and k.start <= span.start and span.end <= k.end for k in kept):
            continue
        kept.append(span)
    return kept


# --------------------------------------------------------------------------- name resolution


@dataclass(frozen=True, slots=True)
class _Names:
    """Turns graph ids into the names a reader knows; unknown ids are shown verbatim."""

    graph: OpsGraph
    service_names: Mapping[str, str]
    docs: Mapping[str, DocumentRef]

    def node(self, node_id: str) -> str:
        node = self.graph.nodes.get(node_id)
        if node is not None:
            return node.name
        return self.service_names.get(node_id, node_id)

    def ref(self, ref_id: str) -> str:
        """Name of a node, or ``출발 → 도착 (관계)`` for an edge."""
        edge = self.graph.edges.get(ref_id)
        if edge is not None:
            return f"{self.node(edge.src)} → {self.node(edge.dst)} ({EDGE_KIND_LABEL_KO[edge.kind]})"
        return self.node(ref_id)

    def service(self, service_id: str | None) -> str:
        return "조직 전체" if service_id is None else self.node(service_id)

    def document(self, doc_id: str) -> str:
        ref = self.docs.get(doc_id)
        return ref.title if ref is not None else doc_id


# --------------------------------------------------------------------------- view rows


@dataclass(frozen=True, slots=True)
class EvidenceQuote:
    """One verbatim runbook quote with its location (근거 인용)."""

    doc_id: str
    document: str
    line: int | None
    start: int
    end: int
    text: str

    @property
    def location(self) -> str:
        if self.line is not None:
            return f"{self.document}:{self.line}"
        return f"{self.document} (문자 {self.start}–{self.end})"


@dataclass(frozen=True, slots=True)
class FindingView:
    finding: Finding
    number: str  # section-qualified reference such as "3.4"
    service: str
    severity: str
    status: str
    element: str | None
    fix: str | None
    subjects: str | None  # related graph elements by name, "외 N개" beyond the display limit
    quotes: tuple[EvidenceQuote, ...]


@dataclass(frozen=True, slots=True)
class ServiceRow:
    service_id: str
    name: str
    tier: int
    tier_label: str
    score: float
    statuses: tuple[ElementStatus | None, ...]  # in ``Element`` order; None = not evaluated


@dataclass(frozen=True, slots=True)
class UnknownElementRow:
    service: str
    element: str
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RehearsalRow:
    scenario_id: str
    title: str
    service: str
    incident: str
    incident_type: IncidentType
    location_down: bool
    outcome: Outcome
    deadline: str
    ended_at: str
    broken_at: str | None
    reason: str
    comparison: str | None
    #: The comparison rests on a replay of a *made* change: the scenario's result on an earlier
    #: graph, or an approved patch's validation. A pending patch's forecast does not count.
    compared: bool = False


@dataclass(frozen=True, slots=True)
class HypothesisRow:
    """One risk hypothesis as the evidence report shows it (규칙 밖 위험 후보)."""

    number: str  # "9.1"
    hypothesis_id: str
    status: HypothesisStatus
    status_label: str
    origin_label: str
    service: str
    title: str
    explanation: str
    perturbations: str
    #: one line: "실패 — 담당 · 최영훈" / "통과 — T+30분 완화(시한 1시간)"
    replay: str
    #: the replay's own reason (failures and passes alike)
    reason: str
    assumptions: tuple[str, ...]
    required_info: tuple[str, ...]
    quotes: tuple[EvidenceQuote, ...]
    #: 최소 실패 조건 자동 검증 (confirmed AI hypotheses only): the result in one line —
    #: "상품 카탈로그 동시 장애 — 불필요한 조건: 강현우 부재 (부분 조합 2개 리허설)" — or ``None``
    minimal: str | None = None
    #: where a minimal set breaks elsewhere than the whole hypothesis, other breaks, and the scope
    minimal_notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class VersionRow:
    number: int
    graph_hash: str
    change: str
    sentence: str | None
    approver: str
    at: str
    resolved: str | None


@dataclass(frozen=True, slots=True)
class PatchRow:
    patch_id: str
    status: str
    author: str
    sentence: str
    targets: str
    decided: str | None


@dataclass(frozen=True, slots=True)
class HeldRow:
    ref_id: str
    kind: str
    label: str
    reason: str


@dataclass(frozen=True, slots=True)
class SecurityRow:
    document: str
    location: str
    patterns: str
    excerpt: str


class ControlStatus(StrEnum):
    OK = "이상 없음(고정 규칙 기준)"
    GAP = "보완 필요"
    MISSING = "증적 없음"


@dataclass(frozen=True, slots=True)
class ControlRow:
    control_id: str
    title: str
    source: str
    evidence: str
    status: ControlStatus


@dataclass(frozen=True, slots=True)
class RuleRow:
    rule_id: str
    severity: Severity
    severity_label: str
    element: str | None
    title: str
    rationale: str
    legal_basis: str | None
    count: int


@dataclass(frozen=True, slots=True)
class WeightRow:
    label: str
    key: str
    weight: float


@dataclass(frozen=True, slots=True)
class DocumentRow:
    title: str
    path: str | None
    sha256: str


@dataclass(frozen=True, slots=True)
class EvidenceView:
    """Everything the evidence report shows, already resolved, sorted and labelled."""

    title: str
    org_name: str | None
    workspace_id: str
    run_id: str
    overall: float
    weights_version: str
    graph_version: int
    graph_hash: str
    created_at: str
    broken_by_severity: tuple[tuple[Severity, int], ...]
    unknown_count: int
    grounding: str
    llm_usage: str
    judgement_path: str
    judgement_path_calls: int
    documents: tuple[DocumentRow, ...]
    services: tuple[ServiceRow, ...]
    catalog_gaps: tuple[str, ...]
    broken: tuple[tuple[Severity, tuple[FindingView, ...]], ...]
    unknown: tuple[FindingView, ...]
    rehearsals: tuple[RehearsalRow, ...]
    rehearsal_passed: int
    versions: tuple[VersionRow, ...]
    other_patches: tuple[PatchRow, ...]
    held: tuple[HeldRow, ...]
    unknown_elements: tuple[UnknownElementRow, ...]
    dropped_examples: tuple[str, ...]
    security: tuple[SecurityRow, ...]
    controls: tuple[ControlRow, ...]
    element_weights: tuple[WeightRow, ...]
    tier_weights: tuple[WeightRow, ...]
    rules: tuple[RuleRow, ...]
    #: 규칙 밖 위험 후보 (verified hypotheses, not scored); ``exploration`` is ``None`` for reports
    #: made before exploration existed or with it turned off
    hypotheses: tuple[HypothesisRow, ...] = ()
    exploration: str | None = None
    exploration_notes: tuple[str, ...] = ()
    #: what 9장 says when no hypothesis is listed — "nothing found" only when every critical
    #: service was searched in full; otherwise what could not be searched (P3)
    exploration_empty: str = EXPLORE_NOTHING_FOUND

    @property
    def hypothesis_summary(self) -> str:
        """``확인됨(리허설로 재현) 2건 · 확인 필요 1건 · 기각 0건``."""
        return hypothesis_counts(h.status for h in self.hypotheses)

    @property
    def explored(self) -> bool:
        """Did this report run exploration at all (or carry hypotheses)?"""
        return self.exploration is not None or bool(self.hypotheses)

    @property
    def broken_count(self) -> int:
        return sum(count for _, count in self.broken_by_severity)

    @property
    def rehearsal_rate(self) -> str:
        return _percent(self.rehearsal_passed, len(self.rehearsals))

    @property
    def scoring_name(self) -> str:
        """``v1`` → ``SCORING_V1``: the name docs/scoring.md publishes the method under."""
        return f"SCORING_{self.weights_version.upper()}"

    @property
    def severity_summary(self) -> str:
        """``심각 2 · 높음 1`` (empty when nothing is broken)."""
        return " · ".join(f"{SEVERITY_LABEL_KO[s]} {n}" for s, n in self.broken_by_severity)

    @property
    def broken_summary(self) -> str:
        severity = self.severity_summary
        return (
            f"{self.broken_count}건"
            + (f" ({severity})" if severity else "")
            + f" · 확인 불가 {self.unknown_count}건"
        )

    @property
    def rehearsal_summary(self) -> str:
        return f"{len(self.rehearsals)}건 중 {self.rehearsal_passed}건 통과 (통과율 {self.rehearsal_rate})"

    @property
    def rehearsal_intro(self) -> str:
        return (
            f"모의 장애 {len(self.rehearsals)}건을 운영 그래프 위에서 결정적으로 재생했습니다. "
            f"통과 {self.rehearsal_passed}건 (통과율 {self.rehearsal_rate}). "
            "실제 시스템은 건드리지 않습니다."
        )


# --------------------------------------------------------------------------- builders


def _quotes(spans: Iterable[SourceSpan], names: _Names) -> tuple[EvidenceQuote, ...]:
    return tuple(
        EvidenceQuote(
            doc_id=s.doc_id,
            document=names.document(s.doc_id),
            line=s.line,
            start=s.start,
            end=s.end,
            text=_clip(s.quote, QUOTE_DISPLAY_CHARS),
        )
        for s in distinct_spans(spans)
    )


def _subjects(finding: Finding, names: _Names) -> str | None:
    """Related graph elements by name; the finding's own service is already in its heading."""
    labels = list(dict.fromkeys(names.ref(sid) for sid in finding.subject_ids if sid != finding.service_id))
    if not labels:
        return None
    shown = ", ".join(labels[:SUBJECT_DISPLAY_LIMIT])
    rest = len(labels) - SUBJECT_DISPLAY_LIMIT
    return shown + (f" 외 {rest}개" if rest > 0 else "")


def _finding_view(finding: Finding, number: str, names: _Names) -> FindingView:
    return FindingView(
        finding=finding,
        number=number,
        service=names.service(finding.service_id),
        severity=SEVERITY_LABEL_KO[finding.severity],
        status=FINDING_STATUS_LABEL_KO[finding.status],
        element=ELEMENT_LABEL_KO[finding.element] if finding.element is not None else None,
        fix=FIX_KIND_LABEL_KO[finding.fix.kind] if finding.fix is not None else None,
        subjects=_subjects(finding, names),
        quotes=_quotes(finding.evidence, names),
    )


def _grounding_line(g: GroundingStatus) -> str:
    dropped = g.dropped_ungrounded + g.dropped_quarantined + g.dropped_invalid
    examined = g.grounded + dropped
    if examined == 0:
        return "추출 기록 없음"
    held = g.held_backtranslation + g.held_ambiguous_entity
    return (
        f"원문 근거율 {_percent(g.grounded, examined)} — 원문 위치에 정렬된 요소 {g.grounded:,}건"
        f"(유사 일치 {g.fuzzy_grounded:,}건 포함) · 폐기 {dropped:,}건(근거 없음 {g.dropped_ungrounded:,} · "
        f"격리 구간 {g.dropped_quarantined:,} · 형식 오류 {g.dropped_invalid:,}) · 판단 보류 {held:,}건"
    )


def service_rows(report: DiagnosisReport) -> tuple[ServiceRow, ...]:
    """Services most critical tier first, weakest first inside a tier: what to fix first."""
    rows: list[ServiceRow] = []
    for svc in report.readiness.services:
        by_element = {e.element: e.status for e in svc.elements}
        rows.append(
            ServiceRow(
                service_id=svc.service_id,
                name=svc.service_name,
                tier=svc.tier,
                tier_label=TIER_LABEL_KO.get(svc.tier, "기타"),
                score=svc.score,
                statuses=tuple(by_element.get(element) for element in Element),
            )
        )
    return tuple(sorted(rows, key=lambda r: (r.tier, r.score, r.name, r.service_id)))


def _unknown_elements(report: DiagnosisReport) -> tuple[UnknownElementRow, ...]:
    rows = [
        UnknownElementRow(service=svc.service_name, element=ELEMENT_LABEL_KO[e.element], reasons=e.reasons)
        for svc in report.readiness.services
        for e in svc.elements
        if e.status is ElementStatus.UNKNOWN
    ]
    return tuple(sorted(rows, key=lambda r: (r.service, r.element, r.reasons)))


def current_rehearsals(
    report: DiagnosisReport, version_of: Mapping[str, int] | None = None
) -> list[tuple[RehearsalResult, RehearsalResult | None]]:
    """Per scenario (sorted by id): the result for the reported graph and the earliest other one.

    A report may carry results of the same scenario on several graph versions (before and after an
    approved fix). The *current* result is the one on the reported graph hash; failing that, the one
    on the newest known version, then the last listed. The *before* result is the one on the oldest
    known version among the others (unknown versions only as a last resort), or ``None``.
    """
    # The reported graph keeps its own number: a revert recreates an earlier graph byte for byte,
    # so the same hash can belong to several versions and the history must not relabel the report.
    versions = {**(version_of or {}), report.graph_hash: report.graph_version}
    grouped: dict[str, list[tuple[int, RehearsalResult]]] = {}
    for position, result in enumerate(report.rehearsals):
        grouped.setdefault(result.scenario_id, []).append((position, result))
    pairs: list[tuple[RehearsalResult, RehearsalResult | None]] = []
    for scenario_id in sorted(grouped):
        entries = grouped[scenario_id]
        current = max(
            entries,
            key=lambda e: (e[1].graph_hash == report.graph_hash, versions.get(e[1].graph_hash, -1), e[0]),
        )[1]
        older = [e for e in entries if e[1].graph_hash != current.graph_hash]
        before = (
            min(
                older,
                key=lambda e: (
                    (0, versions[e[1].graph_hash], e[0]) if e[1].graph_hash in versions else (1, 0, e[0])
                ),
            )[1]
            if older
            else None
        )
        pairs.append((current, before))
    return pairs


def _version_numbers(versions: Iterable[GraphVersion], report: DiagnosisReport) -> dict[str, int]:
    """Graph hash → the version number it is called by in this report.

    A revert recreates an earlier graph byte for byte, so one hash can carry several numbers. The
    label is the newest of them not newer than the reported version — the graph as it stood just
    before the report — and only a graph first seen after the report (a history passed alongside a
    stale report) takes its first later number. A before result is therefore never labelled newer
    than the report it is compared with ("v3 실패 → v2 통과"). The reported graph always keeps its
    own number, even without any history.
    """
    numbers: dict[str, int] = {}
    for version in sorted(versions, key=lambda v: (v.number, v.graph_hash)):
        if version.number <= report.graph_version or version.graph_hash not in numbers:
            numbers[version.graph_hash] = version.number
    numbers[report.graph_hash] = report.graph_version
    return numbers


def _version_label(graph_hash: str, versions: Mapping[str, int]) -> str:
    number = versions.get(graph_hash)
    return f"v{number}" if number is not None else f"이전 그래프({short_hash(graph_hash)})"


def _patch_comparison(current: RehearsalResult, patches: Sequence[Patch]) -> tuple[str | None, bool]:
    """Before/after from patch validation when the report has no earlier replay of the scenario.

    Returns the text and whether it reports a made change (an approved patch). A validation replays
    the patch on its base graph when it is drafted, not the reported graph — the patch may since
    have been reverted — so a transition is only cited when it ends in the outcome the reported
    graph shows: "실패 → 통과 (반영)" next to a failing result would tell an auditor the fix holds
    when it does not. A pending patch's forecast likewise only makes sense for a scenario that
    still fails.
    """
    scenario_id, outcome = current.scenario_id, current.outcome
    approved_fixed: list[str] = []
    approved_regressed: list[str] = []
    pending_fixed: list[str] = []
    for patch in patches:
        validation = patch.validation
        if validation is None:
            continue
        if patch.status is PatchStatus.APPROVED:
            if outcome is Outcome.PASS and scenario_id in validation.rehearsals_fixed:
                approved_fixed.append(patch.id)
            if outcome is Outcome.FAIL and scenario_id in validation.rehearsals_regressed:
                approved_regressed.append(patch.id)
        elif (
            outcome is Outcome.FAIL
            and patch.status in (PatchStatus.DRAFT, PatchStatus.VALIDATED)
            and scenario_id in validation.rehearsals_fixed
        ):
            pending_fixed.append(patch.id)
    parts: list[str] = []
    if approved_fixed:
        parts.append(f"실패 → 통과 (수정안 {', '.join(approved_fixed)} 반영)")
    if approved_regressed:
        parts.append(f"통과 → 실패 (수정안 {', '.join(approved_regressed)} 반영)")
    if pending_fixed:
        parts.append(f"승인 시 실패 → 통과 예상 (수정안 {', '.join(pending_fixed)})")
    return "; ".join(parts) or None, bool(approved_fixed or approved_regressed)


#: Appended when the earlier result replayed another definition of the scenario: template drills
#: keep their id while their target follows the graph (e.g. "1차 담당 부재" names the new first
#: responder after a fix), so the two outcomes are not the same drill run twice.
CHANGED_DRILL_NOTE: Final = " (시나리오 조건 달라짐)"


def _ended_at(result: RehearsalResult) -> str:
    """When the replay ended: mitigation time, flagged when it came after the deadline."""
    ttm = result.time_to_mitigate
    if ttm is not None:
        late = math.isfinite(ttm) and ttm > result.deadline_minutes
        return f"T+{format_minutes(ttm)}" + (" (시한 초과)" if late else "")
    if result.events:
        # "중단" means the response broke off; a passing replay without a mitigation time did not.
        stopped = " (중단)" if result.outcome is Outcome.FAIL else ""
        return f"T+{format_minutes(max(e.t for e in result.events))}{stopped}"
    return "기록 없음"


def _rehearsal_rows(
    report: DiagnosisReport, names: _Names, versions: Mapping[str, int], patches: Sequence[Patch]
) -> tuple[RehearsalRow, ...]:
    scenarios: dict[str, Scenario] = {s.id: s for s in report.scenarios}
    rows: list[RehearsalRow] = []
    for current, before in current_rehearsals(report, versions):
        scenario = scenarios.get(current.scenario_id)
        incident = scenario.incident_type if scenario is not None else IncidentType.OUTAGE
        broken_at: str | None = None
        if current.outcome is Outcome.FAIL:
            pieces = [ELEMENT_LABEL_KO[current.failed_element]] if current.failed_element is not None else []
            if current.failed_at_id is not None:
                pieces.append(names.ref(current.failed_at_id))
            broken_at = " · ".join(pieces) or "기록 없음"
        comparison: str | None
        if before is not None:
            same_drill = before.scenario_fingerprint == current.scenario_fingerprint
            comparison = (
                f"{_version_label(before.graph_hash, versions)} {OUTCOME_LABEL_KO[before.outcome]} → "
                f"{_version_label(current.graph_hash, versions)} {OUTCOME_LABEL_KO[current.outcome]}"
                + ("" if same_drill else CHANGED_DRILL_NOTE)
            )
            compared = True
        else:
            comparison, compared = _patch_comparison(current, patches)
        rows.append(
            RehearsalRow(
                scenario_id=current.scenario_id,
                title=scenario.title if scenario is not None else current.scenario_id,
                service=names.service(scenario.service_id) if scenario is not None else "확인 불가",
                incident=INCIDENT_TYPE_LABEL_KO[incident],
                incident_type=incident,
                location_down=scenario is not None
                and any(p.kind is PerturbationKind.LOCATION_DOWN for p in scenario.perturbations),
                outcome=current.outcome,
                deadline=format_minutes(current.deadline_minutes),
                ended_at=_ended_at(current),
                broken_at=broken_at,
                reason=current.reason,
                comparison=comparison,
                compared=compared,
            )
        )
    return tuple(rows)


def hypothesis_counts(statuses: Iterable[HypothesisStatus]) -> str:
    """``확인됨(리허설로 재현) 2건 · 확인 필요 1건 · 기각 0건`` (every status, zeros included)."""
    counts = Counter(statuses)
    return " · ".join(f"{label} {counts[s]}건" for s, label in HYPOTHESIS_STATUS_LABEL_KO.items())


def _perturbation_label(perturbation: Perturbation, names: _Names) -> str:
    return f"{names.node(perturbation.target_id)} {PERTURBATION_SUFFIX_KO[perturbation.kind]}"


def _failure_point(element: Element | None, at_id: str | None, names: _Names) -> str:
    """'담당 · 최영훈' — the step and the node where a replay broke."""
    pieces = [ELEMENT_LABEL_KO[element]] if element is not None else []
    if at_id is not None:
        pieces.append(names.ref(at_id))
    return " · ".join(pieces) or "끊긴 지점 기록 없음"


def _replay_line(hypothesis: RiskHypothesis, names: _Names) -> str:
    v = hypothesis.verification
    if v.baseline_outcome is Outcome.FAIL:
        return "기준 리허설부터 실패 — 가설만의 영향 확인 불가"
    if v.outcome is Outcome.PASS:
        ended = f"T+{format_minutes(v.time_to_mitigate)} 완화" if v.time_to_mitigate is not None else "완화"
        return f"통과 — {ended}(시한 {format_minutes(v.deadline_minutes)})"
    where = _failure_point(v.failed_element, v.failed_at_id, names)
    return f"실패 — {where}(시한 {format_minutes(v.deadline_minutes)})"


def _conditions(perturbations: Sequence[Perturbation], names: _Names) -> str:
    return " + ".join(_perturbation_label(p, names) for p in perturbations)


def _minimal_line(m: HypothesisMinimization, names: _Names) -> str:
    """'상품 카탈로그 동시 장애 — 불필요한 조건: 강현우 부재 (부분 조합 2개 리허설)'."""
    tried = f"(부분 조합 {m.replays}개 리허설)"
    if m.original_minimal:
        return f"제안한 조합 그대로 — 어느 조건을 빼도 같은 지점에서 끊기지 않음 {tried}"
    line = " / ".join(_conditions(s.perturbations, names) for s in m.minimal_sets)
    if len(m.minimal_sets) > 1:
        line += " (어느 하나만으로도 같은 지점에서 끊김)"
    if m.unnecessary:
        line += f" — 불필요한 조건: {_conditions(m.unnecessary, names)}"
    return f"{line} {tried}"


def _subset_point(subset: FailingSubset, names: _Names) -> str:
    where = _failure_point(subset.failed_element, subset.failed_at_id, names)
    return f"{where}에서 끊김" + (f"({subset.reason})" if subset.reason else "")


def _minimal_notes(hypothesis: RiskHypothesis, m: HypothesisMinimization, names: _Names) -> tuple[str, ...]:
    """Disclosures next to the result: a minimal set breaking at another step of the same node, a
    subset breaking at another node (a different break), and what the check does not do."""
    v = hypothesis.verification
    whole = _failure_point(v.failed_element, v.failed_at_id, names)
    notes = [
        f"끊긴 단계가 제안한 조합과 다름: {_conditions(s.perturbations, names)}만 있으면 "
        f"{_subset_point(s, names)}. 제안한 조합 전체는 {whole}에서 끊김."
        for s in m.minimal_sets
        if s.point_differs
    ]
    notes += [
        "다른 지점에서 끊기는 부분 조합(같은 위험의 재현으로 보지 않음): "
        f"{_conditions(s.perturbations, names)} — {_subset_point(s, names)}"
        for s in m.other_failures
    ]
    notes.append(MINIMIZE_SCOPE)
    return tuple(notes)


def _hypothesis_rows(report: DiagnosisReport, names: _Names) -> tuple[HypothesisRow, ...]:
    """Hypotheses in the report's order (확인됨 → 확인 필요 → 기각, as verification sorted them)."""
    return tuple(
        HypothesisRow(
            number=f"9.{i}",
            hypothesis_id=h.id,
            status=h.status,
            status_label=HYPOTHESIS_STATUS_LABEL_KO[h.status],
            origin_label=HYPOTHESIS_ORIGIN_LABEL_KO[h.origin],
            service=names.service(h.service_id),
            title=h.title,
            explanation=(
                f"{AI_EXPLANATION_PREFIX}{h.explanation}"
                if h.origin is HypothesisOrigin.AI and h.explanation
                else h.explanation
            ),
            perturbations=_conditions(h.perturbations, names),
            replay=_replay_line(h, names),
            reason=h.verification.reason,
            assumptions=h.assumptions,
            required_info=h.required_info,
            quotes=_quotes(h.evidence, names),
            minimal=_minimal_line(h.minimization, names) if h.minimization is not None else None,
            minimal_notes=_minimal_notes(h, h.minimization, names) if h.minimization is not None else (),
        )
        for i, h in enumerate(report.risk_hypotheses, start=1)
    )


def _ai_line(summary: ExplorationSummary, mode: LlmMode) -> str:
    """What the generative model contributed — or why it did not, in the case's own words."""
    status = AiProposalStatus.USED if summary.ai_used else summary.ai_status
    if status is AiProposalStatus.USED:
        left: list[str] = []
        if summary.ai_dropped:
            left.append(f"목록에 없는 id·판단 보류·종류 불일치 등 {summary.ai_dropped}건 버림")
        if summary.ai_duplicates:
            left.append(f"중복이거나 이미 찾은 끊김 조합과 겹침 {summary.ai_duplicates}건 버림")
        if summary.ai_over_cap:
            left.append(f"요청 개수를 넘은 {summary.ai_over_cap}건 제외(상한)")
        return f"생성형 AI 제안 {summary.ai_proposed}건 중 {summary.ai_kept}건 검증" + (
            f"({', '.join(left)})" if left else ""
        )
    if status is AiProposalStatus.OFF:
        return f"생성형 AI 제안 없음({_AI_OFF_REASON.get(mode, 'LLM을 쓸 수 없음 — 자격 증명 없음 등')})"
    return _AI_STATUS_LINE[status]


def _exploration_empty(summary: ExplorationSummary | None, names: _Names) -> str:
    """9장 with no hypothesis: "nothing found" only where the search ran in full (P3)."""
    if summary is None:
        return EXPLORE_NOTHING_FOUND
    if not summary.services:
        return "핵심 서비스가 없어 탐색할 대상이 없었습니다."
    searched = [s for s in summary.services if s.skipped_reason is None]
    if not searched:
        return (
            "탐색할 수 있는 핵심 서비스가 없어 규칙 밖 위험을 확인하지 못했습니다"
            "(모든 핵심 서비스가 교란 없이도 리허설에 실패했거나 리허설 예산을 다 씀). "
            "위험이 없다는 뜻이 아닙니다."
        )
    skipped = [names.service(s.service_id) for s in summary.services if s.skipped_reason is not None]
    partial = [
        names.service(s.service_id) for s in searched if s.exhausted or s.candidates < s.candidates_total
    ]
    if not skipped and not partial:
        return EXPLORE_NOTHING_FOUND
    unchecked: list[str] = []
    if skipped:
        unchecked.append(f"탐색하지 못한 서비스({', '.join(skipped)})")
    if partial:
        unchecked.append(f"상한·예산으로 일부만 탐색한 서비스({', '.join(partial)})의 나머지 조합")
    topic = "와 ".join(unchecked) + ("은" if partial else "는")
    return (
        f"탐색한 범위(핵심 서비스 {len(summary.services)}개 중 {len(searched)}개)에서는 "
        f"{EXPLORE_NOTHING_FOUND} {topic} 확인하지 못한 것이지 위험이 없다는 뜻이 아닙니다."
    )


def _exploration_line(summary: ExplorationSummary, names: _Names, mode: LlmMode) -> str:
    """How far the search went — what was searched, skipped, already covered, asked of the AI."""
    searched = [s for s in summary.services if s.skipped_reason is None]
    broken = sum(1 for s in summary.services if s.baseline is Outcome.FAIL)
    parts = [
        f"자동 전수 탐색({summary.method}): 핵심 서비스 {len(summary.services)}개 중 {len(searched)}개를 "
        f"최대 {summary.max_set_size}개 조합까지 리허설 {summary.replays:,}회로 탐색"
        + (f"(교란 없이도 실패하는 {broken}개는 고정 규칙의 끊김이 우선이라 제외)" if broken else "")
    ]
    if summary.covered:
        parts.append(f"고정 규칙이 이미 보고한 조합 {len(summary.covered)}개는 다시 싣지 않음")
    parts.append(_ai_line(summary, mode))
    if summary.ai_carried:
        parts.append(f"이전 진단의 AI 가설 {summary.ai_carried}건 다시 확인")
    return " · ".join(parts)


def _resolved_summary(finding_ids: Sequence[str]) -> str | None:
    if not finding_ids:
        return None
    rules = sorted({rule_id_of(fid) for fid in finding_ids})
    return f"{', '.join(rules)} ({len(finding_ids)}건)"


def _version_rows(versions: Sequence[GraphVersion], patches: Mapping[str, Patch]) -> tuple[VersionRow, ...]:
    rows: list[VersionRow] = []
    for version in sorted(versions, key=lambda v: (v.number, v.graph_hash)):
        patch = patches.get(version.patch_id) if version.patch_id is not None else None
        if version.revert_of is not None:
            change = f"v{version.revert_of} 되돌리기"
        elif version.patch_id is not None:
            change = f"수정안 {version.patch_id} 반영"
        else:
            change = "최초 추출"
        sentence = patch.runbook_text if patch is not None else None
        approver = patch.decided_by if patch is not None and patch.decided_by else version.created_by
        resolved = (
            _resolved_summary(patch.validation.resolved_finding_ids)
            if patch is not None and patch.validation is not None
            else None
        )
        rows.append(
            VersionRow(
                number=version.number,
                graph_hash=short_hash(version.graph_hash),
                change=change + (f" — {version.note}" if version.note else ""),
                sentence=sentence,
                approver=approver,
                at=format_datetime(version.created_at),
                resolved=resolved,
            )
        )
    return tuple(rows)


def _patch_rows(patches: Iterable[Patch], used: set[str]) -> tuple[PatchRow, ...]:
    rows: list[PatchRow] = []
    for patch in sorted(patches, key=lambda p: p.id):
        if patch.id in used:
            continue
        decided = None
        if patch.decided_by or patch.decided_at:
            who = patch.decided_by or "기록 없음"
            when = format_datetime(patch.decided_at) if patch.decided_at is not None else "시각 기록 없음"
            decided = f"{who} · {when}"
        rows.append(
            PatchRow(
                patch_id=patch.id,
                status=PATCH_STATUS_LABEL_KO[patch.status],
                author=PATCH_AUTHOR_LABEL_KO[patch.author],
                sentence=patch.runbook_text,
                targets=_resolved_summary(patch.finding_ids) or "—",
                decided=decided,
            )
        )
    return tuple(rows)


def _held_rows(report: DiagnosisReport, graph: OpsGraph, names: _Names) -> tuple[HeldRow, ...]:
    rows: list[HeldRow] = []
    for ref_id in sorted(set(report.held_element_ids)):
        node = graph.nodes.get(ref_id)
        edge = graph.edges.get(ref_id)
        if node is not None:
            kind, reason = NODE_KIND_LABEL_KO[node.kind], node.hold_reason
        elif edge is not None:
            kind, reason = f"관계({EDGE_KIND_LABEL_KO[edge.kind]})", edge.hold_reason
        else:
            kind, reason = "그래프에 없음", None
        label = names.ref(ref_id)
        rows.append(HeldRow(ref_id=ref_id, kind=kind, label=label, reason=reason or "사유 기록 없음"))
    return tuple(rows)


def _security_rows(notices: Iterable[SecurityNotice], names: _Names) -> tuple[SecurityRow, ...]:
    rows = [
        SecurityRow(
            document=names.document(n.doc_id),
            location=f"문자 {n.start}–{n.end}",
            patterns=" · ".join(
                INJECTION_PATTERN_LABEL_KO.get(p, p) for p in (x.strip() for x in n.pattern.split(",")) if p
            ),
            excerpt=mask_excerpt(n.excerpt),
        )
        for n in sorted(notices, key=lambda n: (n.doc_id, n.start, n.end, n.pattern, n.excerpt))
    ]
    return tuple(rows)


def _tally(rows: Sequence[RehearsalRow]) -> tuple[int, int]:
    passed = sum(1 for r in rows if r.outcome is Outcome.PASS)
    return passed, len(rows) - passed


def _control_rows(findings: Sequence[Finding], rehearsals: Sequence[RehearsalRow]) -> tuple[ControlRow, ...]:
    """ISMS-P mapping. A control with no evidence is reported as missing, never as satisfied (P3)."""
    controls: list[ControlRow] = []

    passed, failed = _tally(rehearsals)
    compared = sum(1 for r in rehearsals if r.compared)
    if rehearsals:
        evidence = (
            f"모의 장애 {len(rehearsals)}건 결정적 재생 · 통과 {passed}건 · 실패 {failed}건 "
            f"(통과율 {_percent(passed, len(rehearsals))}) · 수정 전후 비교 {compared}건"
        )
        status = ControlStatus.GAP if failed else ControlStatus.OK
    else:
        evidence, status = "리허설 기록이 없습니다. 모의 장애 리허설을 실행하세요.", ControlStatus.MISSING
    controls.append(ControlRow("2.11.4", "사고 대응 훈련 및 개선", "4장 리허설 결과", evidence, status))

    legal = [f for f in findings if f.rule_id.startswith(LEGAL_RULE_PREFIXES)]
    legal_broken = sum(1 for f in legal if f.status is FindingStatus.BROKEN)
    legal_rules = ", ".join(sorted({f.rule_id for f in legal}))
    reporting = [r for r in rehearsals if r.incident_type is not IncidentType.OUTAGE]
    r_passed, r_failed = _tally(reporting)
    evidence = (
        f"신고·시한 규칙(JT-KR·JT-DDL) 끊김 {legal_broken}건 · 확인 불가 {len(legal) - legal_broken}건"
        + (f" ({legal_rules})" if legal_rules else "")
        + f" · 신고 리허설 {len(reporting)}건 중 통과 {r_passed}건"
    )
    # Rules finding nothing shows the steps are written down; only a passing reporting rehearsal
    # shows they work in time, so without one the control has no evidence yet (P3).
    if legal or r_failed:
        status = ControlStatus.GAP
    else:
        status = ControlStatus.OK if reporting else ControlStatus.MISSING
    controls.append(
        ControlRow(
            "2.11.5",
            "사고 대응 및 복구(법적 통지·신고)",
            "3장 JT-KR·JT-DDL, 4장 신고 리허설",
            evidence,
            status,
        )
    )

    disaster = [f for f in findings if f.rule_id in DISASTER_RULE_IDS]
    center = [r for r in rehearsals if r.location_down]
    c_passed, c_failed = _tally(center)
    counts = Counter(f.rule_id for f in disaster)
    evidence = (
        " · ".join(f"{rule_id} {counts.get(rule_id, 0)}건" for rule_id in DISASTER_RULE_IDS)
        + f" · 센터 장애 시나리오 {len(center)}건(실패 {c_failed}건)"
    )
    # A centre-down replay that breaks is direct evidence that the safeguards do not hold, so the
    # preparedness control cannot read "이상 없음" next to it even when no structural rule fired.
    if disaster or c_failed:
        status = ControlStatus.GAP
    else:
        status = ControlStatus.OK if center else ControlStatus.MISSING
    controls.append(
        ControlRow(
            "2.12.1",
            "재해·재난 대비 안전조치",
            "3장 한 위치 집중·채널 동반 장애, 4장 센터 장애 리허설",
            evidence,
            status,
        )
    )

    if center:
        evidence = f"센터 장애 시나리오 {len(center)}건 재생 · 통과 {c_passed}건 · 실패 {c_failed}건"
        status = ControlStatus.GAP if c_failed else ControlStatus.OK
    else:
        evidence, status = "센터(위치) 장애 시나리오 리허설 기록이 없습니다.", ControlStatus.MISSING
    controls.append(ControlRow("2.12.2", "재해 복구 시험 및 개선", "4장 센터 장애 리허설", evidence, status))
    return tuple(controls)


def _rule_rows(findings: Sequence[Finding]) -> tuple[RuleRow, ...]:
    counts = Counter(f.rule_id for f in findings)
    return tuple(
        RuleRow(
            rule_id=info.rule_id,
            severity=info.severity,
            severity_label=SEVERITY_LABEL_KO[info.severity],
            element=ELEMENT_LABEL_KO[info.element] if info.element is not None else None,
            title=info.title,
            rationale=info.rationale,
            legal_basis=info.legal_basis,
            count=counts.get(info.rule_id, 0),
        )
        for info in sorted(rule_catalog(), key=lambda i: i.rule_id)
    )


def _weight_rows(readiness: ReadinessReport) -> tuple[tuple[WeightRow, ...], tuple[WeightRow, ...]]:
    """The weights the report was actually scored with, so a reader can recompute every number.

    A report that carries no weights (hand-built, or from before weights were stored) falls back to
    the published SCORING_V1 table rather than showing an empty, unverifiable method.
    """
    element_weights = readiness.element_weights or SCORING_V1["element_weights"]
    tier_weights = readiness.tier_weights or SCORING_V1["tier_weights"]
    labels = {e.value: ELEMENT_LABEL_KO[e] for e in Element}
    order = {e.value: i for i, e in enumerate(Element)}
    elements = tuple(
        WeightRow(labels.get(key, key), key, weight)
        for key, weight in sorted(
            element_weights.items(), key=lambda kv: (order.get(kv[0], len(order)), kv[0])
        )
    )
    tiers = tuple(
        WeightRow(f"{tier}등급({TIER_LABEL_KO.get(tier, '기타')})", str(tier), weight)
        for tier, weight in sorted(tier_weights.items())
    )
    return elements, tiers


def build_evidence(
    report: DiagnosisReport,
    graph: OpsGraph,
    *,
    versions: Sequence[GraphVersion] = (),
    patches: Sequence[Patch] = (),
    documents: Sequence[DocumentRef] = (),
    org_name: str | None = None,
) -> EvidenceView:
    """Resolve, group, sort and label everything the evidence report shows.

    ``patches`` are merged with ``report.patches`` (an explicitly passed patch wins, because it is
    the newer state — e.g. approved after the report was produced). ``documents`` are merged over
    ``graph.docs`` the same way.
    """
    docs = {**graph.docs, **{d.doc_id: d for d in documents}}
    names = _Names(graph, {s.service_id: s.service_name for s in report.readiness.services}, docs)
    patch_map = {**{p.id: p for p in report.patches}, **{p.id: p for p in patches}}
    all_patches = [patch_map[k] for k in sorted(patch_map)]
    version_of = _version_numbers(versions, report)

    findings = sort_findings(report.findings)
    broken = [f for f in findings if f.status is FindingStatus.BROKEN]
    unknown = [f for f in findings if f.status is FindingStatus.UNKNOWN]

    numbered: list[tuple[Severity, tuple[FindingView, ...]]] = []
    counter = 0
    for severity in Severity:
        group: list[FindingView] = []
        for f in broken:
            if f.severity is severity:
                counter += 1
                group.append(_finding_view(f, f"3.{counter}", names))
        if group:
            numbered.append((severity, tuple(group)))

    rehearsal_rows = _rehearsal_rows(report, names, version_of, all_patches)
    used_patches = {v.patch_id for v in versions if v.patch_id is not None}
    element_weights, tier_weights = _weight_rows(report.readiness)

    return EvidenceView(
        title=EVIDENCE_TITLE,
        org_name=_one_line(org_name),
        workspace_id=report.workspace_id,
        run_id=report.run_id,
        overall=report.readiness.overall,
        weights_version=report.readiness.weights_version,
        graph_version=report.graph_version,
        graph_hash=short_hash(report.graph_hash),
        created_at=format_datetime(report.created_at),
        broken_by_severity=tuple(
            (s, n) for s in Severity if (n := sum(1 for f in broken if f.severity is s))
        ),
        unknown_count=len(unknown),
        grounding=_grounding_line(report.grounding),
        llm_usage=llm_usage_line(report.llm),
        judgement_path=judgement_path_statement(report.llm),
        judgement_path_calls=report.llm.judgement_path_calls,
        documents=tuple(
            DocumentRow(title=d.title, path=d.path, sha256=short_hash(d.sha256))
            for d in sorted(docs.values(), key=lambda d: (d.title, d.doc_id))
        ),
        services=service_rows(report),
        catalog_gaps=tuple(sorted({names.node(gap) for gap in report.catalog_gaps})),
        broken=tuple(numbered),
        unknown=tuple(_finding_view(f, f"6.{i}", names) for i, f in enumerate(unknown, start=1)),
        rehearsals=rehearsal_rows,
        rehearsal_passed=sum(1 for r in rehearsal_rows if r.outcome is Outcome.PASS),
        versions=_version_rows(versions, patch_map),
        other_patches=_patch_rows(all_patches, used_patches),
        held=_held_rows(report, graph, names),
        unknown_elements=_unknown_elements(report),
        dropped_examples=report.grounding.dropped_examples,
        security=_security_rows(report.security, names),
        controls=_control_rows(findings, rehearsal_rows),
        element_weights=element_weights,
        tier_weights=tier_weights,
        rules=_rule_rows(findings),
        hypotheses=_hypothesis_rows(report, names),
        exploration=(
            _exploration_line(report.exploration, names, report.llm.mode)
            if report.exploration is not None
            else None
        ),
        exploration_notes=report.exploration.notes if report.exploration is not None else (),
        exploration_empty=_exploration_empty(report.exploration, names),
    )
