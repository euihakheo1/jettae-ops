"""Risk exploration (규칙 밖 위험 탐색) contracts.

The fixed rules (P1) answer a closed list of questions. Some ways a response can fail are not on
that list: two people who are each replaceable but not *together*, a deputy who is reachable but
too late, a centre whose loss takes two channels down at once, a dependency that no input states.
Exploration proposes such risks as **hypotheses** — from a deterministic exhaustive sweep and,
optionally, from a generative model — and the deterministic replay alone decides each one's
status. Hypotheses never change the readiness score or the rule findings: they are reported next
to them, labelled as not scored.

Status (P3, 확인 불가 ≠ 정상):

* ``confirmed`` — the replay breaks under the hypothesis' perturbations (리허설로 재현).
* ``refuted``   — the replay still mitigates in time under them (리허설상 대응 가능). Only as good
  as the graph: a dependency no input states is invisible to it.
* ``needs_info`` — the hypothesis rests on premises the graph does not contain (``assumptions``),
  names a held (판단 보류) node the replay does not use, or the service fails even without any
  perturbation, so the replay cannot attribute the break. Never shown as confirmed;
  ``required_info`` says what a person must check.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict

from jettae.domain.findings import Element
from jettae.domain.graph import IncidentType, SourceSpan
from jettae.domain.rehearsal import Outcome, Perturbation

#: Version of the published exploration method (docs/explore.md). Changing the sweep's search
#: space or the status rules changes results, so the report names the method it used.
EXPLORE_METHOD: Final = "EXPLORE_V1"


class HypothesisOrigin(StrEnum):
    SWEEP = "sweep"  # 결정적 전수 탐색 (LLM 0회)
    AI = "ai"  # 생성형 AI 제안 (열거된 노드 id 중 선택만)


class HypothesisStatus(StrEnum):
    CONFIRMED = "confirmed"  # 확인됨 (리허설로 재현)
    REFUTED = "refuted"  # 기각 (리허설상 대응 가능)
    NEEDS_INFO = "needs_info"  # 확인 필요 (그래프에 없는 전제 · 판단 보류 대상 · 기존 끊김)


class AiProposalStatus(StrEnum):
    """Why a run has (or has no) AI risk proposals — the report words each case differently."""

    USED = "used"  # 모델이 답했고 제안을 검증함
    OFF = "off"  # LLM 끔 · 모델 없음 · 쓸 수 없음 (LLMDisabledError: 키 없음, 녹화에 없는 요청)
    FAILED = "failed"  # 호출이 실패함 (사유는 notes) — 자동 전수 탐색 결과만 실림
    FALLBACK = "fallback"  # 추출 단계 LLM 실패 뒤 이 실행은 LLM을 더 쓰지 않음
    NOT_REQUESTED = "not_requested"  # 제안 개수 0, 또는 제안받을 서비스가 없음
    REJUDGE = "rejudge"  # 승인·되돌리기 뒤 재판정 (LLM 0회, P4)


class HypothesisVerification(BaseModel):
    """What the deterministic replay showed for a hypothesis (the representable part of it)."""

    model_config = ConfigDict(frozen=True)

    scenario_id: str
    scenario_fingerprint: str
    graph_hash: str
    #: the service's policy deadline (tier SLO); a hypothesis can never set its own
    deadline_minutes: float
    outcome: Outcome
    #: the same service replayed without any perturbation (same incident type and deadline)
    baseline_outcome: Outcome
    time_to_mitigate: float | None = None
    failed_element: Element | None = None
    failed_at_id: str | None = None
    reason: str = ""
    result_hash: str
    path_ids: tuple[str, ...] = ()


class FailingSubset(BaseModel):
    """Part of an AI hypothesis' perturbations under which the replay still fails.

    Replayed like the hypothesis itself: same service, incident type and policy deadline.
    """

    model_config = ConfigDict(frozen=True)

    perturbations: tuple[Perturbation, ...]
    failed_element: Element | None = None
    failed_at_id: str | None = None
    reason: str = ""
    result_hash: str
    #: the replay breaks at another step (``failed_element``) or node (``failed_at_id``) than the
    #: whole hypothesis' replay did — disclosed in the report, never hidden
    point_differs: bool = False


class HypothesisMinimization(BaseModel):
    """최소 실패 조건 자동 검증 — which of an AI hypothesis' own perturbations its break needs.

    Every non-empty proper subset of the perturbations of a ``confirmed`` AI hypothesis without
    premises is replayed. A subset *reproduces the break* when its replay fails at the same node
    (``failed_at_id``) as the whole hypothesis'; one that fails elsewhere is a different break —
    listed in ``other_failures``, never used to drop a condition.

    This only removes conditions from the model's own proposal: it does not look for causes the
    proposal does not name (another person's absence may break the same service too). It never
    changes the hypothesis' status, the findings or the score.
    """

    model_config = ConfigDict(frozen=True)

    #: proper subsets replayed (2ⁿ − 2 for n perturbations)
    replays: int
    #: the smallest perturbation sets that reproduce the break (no proper subset of one does),
    #: smallest first; the whole hypothesis when no proper subset does
    minimal_sets: tuple[FailingSubset, ...]
    #: no proper subset reproduces the break: the proposal is already minimal
    original_minimal: bool
    #: perturbations of the hypothesis in no minimal set — the break does not need them
    unnecessary: tuple[Perturbation, ...] = ()
    #: the smallest proper subsets whose replay fails at another node (a different break)
    other_failures: tuple[FailingSubset, ...] = ()


class RiskHypothesis(BaseModel):
    """One proposed risk outside the fixed rules, with the replay's verdict on it.

    ``id`` is ``hyp-{fingerprint}``: a hash of origin, service, incident type, the canonically
    sorted perturbations and assumptions (titles and wording excluded), so the same idea keeps
    its id across runs (P6).
    """

    model_config = ConfigDict(frozen=True)

    id: str
    origin: HypothesisOrigin
    service_id: str
    incident_type: IncidentType = IncidentType.OUTAGE
    title: str
    explanation: str
    perturbations: tuple[Perturbation, ...]
    #: premises the graph does not contain ("슬랙과 인터넷 전화가 같은 사내망을 쓴다")
    assumptions: tuple[str, ...] = ()
    evidence: tuple[SourceSpan, ...] = ()
    status: HypothesisStatus
    verification: HypothesisVerification
    #: what the team must confirm before this hypothesis can be settled (Korean)
    required_info: tuple[str, ...] = ()
    #: 최소 실패 조건 자동 검증 — set only for ``confirmed`` AI hypotheses of two or more
    #: perturbations and no premises (``None`` otherwise, and in reports from before it existed)
    minimization: HypothesisMinimization | None = None

    @staticmethod
    def fingerprint_of(
        origin: HypothesisOrigin,
        service_id: str,
        incident_type: IncidentType,
        perturbations: tuple[Perturbation, ...],
        assumptions: tuple[str, ...],
    ) -> str:
        payload = {
            "origin": origin.value,
            "service_id": service_id,
            "incident_type": incident_type.value,
            "perturbations": sorted(
                (p.model_dump(mode="json") for p in perturbations),
                key=lambda p: json.dumps(p, sort_keys=True),
            ),
            "assumptions": sorted(assumptions),
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class CoveredCut(BaseModel):
    """A breaking combination the sweep found that a fixed-rule finding already reports."""

    model_config = ConfigDict(frozen=True)

    service_id: str
    target_ids: tuple[str, ...]
    finding_ids: tuple[str, ...]


class ServiceSweep(BaseModel):
    """How the exhaustive sweep went for one critical service (published with the report)."""

    model_config = ConfigDict(frozen=True)

    service_id: str
    #: replay of the service without any perturbation; FAIL means nothing was searched, ``None``
    #: that it was never replayed (the service is held — 판단 보류 — or the replay budget ran out
    #: before even this replay; ``skipped_reason`` says which)
    baseline: Outcome | None
    #: perturbation units (people, channels, systems, locations, concurrent services) searched
    candidates: int = 0
    #: units on the response path before the cap (``candidates`` < this = truncated, noted)
    candidates_total: int = 0
    replays: int = 0
    #: minimal breaking combinations found (reported ones + those the rules already cover)
    breaking: int = 0
    #: the replay budget ran out while this service was being searched (combinations left untried)
    exhausted: bool = False
    skipped_reason: str | None = None


class ExplorationSummary(BaseModel):
    """Bookkeeping of one exploration: search bounds, what was skipped or capped, AI usage.

    Every cap that cut the search is a Korean line in ``notes`` (no silent caps).
    """

    model_config = ConfigDict(frozen=True)

    method: str = EXPLORE_METHOD
    max_set_size: int = 2
    services: tuple[ServiceSweep, ...] = ()
    replays: int = 0
    covered: tuple[CoveredCut, ...] = ()
    #: whether (and why not) the model proposed anything; ``ai_used`` ⇔ ``ai_status`` is USED
    ai_status: AiProposalStatus = AiProposalStatus.OFF
    ai_used: bool = False
    #: AI proposals received, and kept after validation
    ai_proposed: int = 0
    ai_kept: int = 0
    #: dropped as invalid: unknown or held id, wrong kind, own service, too many perturbations …
    ai_dropped: int = 0
    #: dropped because a breaking set already found (by the sweep or a rule) accounts for them —
    #: the same set, a set with extra targets, or a concurrent incident pulling a known single point
    ai_duplicates: int = 0
    #: valid proposals beyond the number asked for
    ai_over_cap: int = 0
    #: AI hypotheses of an earlier report re-verified on this graph (after an approval)
    ai_carried: int = 0
    notes: tuple[str, ...] = ()
