"""Readiness score (준비도 점수, 0~100) — SCORING_V1.

Why the score is a fixed, published weighted average and not a model judgement: customers must
be able to recompute it by hand from the five element scores and the weights below, and two
runs on the same graph must give the same number (P1, P6). Element scores come from
:func:`jettae.check.elements.evaluate_service`, the very evaluator the JT-PATH rules use, so the
score and the findings never disagree.

* service score = round(100 × Σ element_weight × element_score, 1)
* overall = tier-weighted mean of the (rounded) critical service scores; all services if none is
  critical; 0 if there are no services. Rounded service scores are averaged so the published
  per-service numbers reproduce the overall exactly.
* tiers missing from the table get the lowest published tier weight.

Arithmetic is exact decimal with half-up rounding (반올림, 5 이상 올림) under a fixed decimal
context. Binary floats would turn 0.25×0.3 + 0.15×0.25 = 0.1125 into 11.249999… and Python's
``round`` rounds halves to even, so the published 11.25 would print as 11.2 while anyone
recomputing by hand gets 11.3. A fixed context also keeps the result independent of whatever
global ``decimal`` context another library set (P6).
"""

from __future__ import annotations

from collections.abc import Iterable
from decimal import ROUND_HALF_UP, Context, Decimal
from typing import Final, TypedDict

from jettae.check.context import CheckContext
from jettae.check.elements import GENERAL_SEMANTICS_KO, SEMANTICS_KO, evaluate_service
from jettae.domain import (
    ELEMENT_LABEL_KO,
    Element,
    ElementScore,
    Finding,
    ReadinessReport,
    ServiceReadiness,
)


class ScoringSpec(TypedDict):
    version: str
    element_weights: dict[str, float]
    tier_weights: dict[int, float]


SCORING_V1: Final[ScoringSpec] = {
    "version": "v1",
    "element_weights": {
        Element.DETECTION.value: 0.20,
        Element.OWNER.value: 0.25,
        Element.APPROVER.value: 0.20,
        Element.FALLBACK_CHANNEL.value: 0.15,
        Element.ROLLBACK.value: 0.20,
    },
    "tier_weights": {1: 3.0, 2: 2.0, 3: 1.0},
}

_TIER_LABEL_KO: Final[dict[int, str]] = {1: "핵심", 2: "중요", 3: "일반"}


def element_weight(element: Element) -> float:
    return SCORING_V1["element_weights"][element.value]


def tier_weight(tier: int) -> float:
    weights = SCORING_V1["tier_weights"]
    return weights.get(tier, min(weights.values()))


#: Fixed arithmetic context: same digits and rounding on every machine, whatever the global context.
_DECIMAL: Final = Context(prec=28, rounding=ROUND_HALF_UP)
_ONE_DECIMAL: Final = Decimal("0.1")
_HUNDRED: Final = Decimal(100)


def _exact(value: float) -> Decimal:
    """The decimal a reader sees (``0.15`` → ``Decimal('0.15')``), not the binary approximation."""
    return Decimal(repr(value))


def _dot(pairs: Iterable[tuple[float, float]]) -> Decimal:
    """Σ a·b computed exactly in the fixed context (never the process-global one)."""
    total = Decimal(0)
    for a, b in pairs:
        total = _DECIMAL.add(total, _DECIMAL.multiply(_exact(a), _exact(b)))
    return total


def _publish(value: Decimal) -> float:
    """Clamp to ``[0, 100]`` and round half-up to one decimal place."""
    clamped = min(max(value, Decimal(0)), _HUNDRED)
    return float(clamped.quantize(_ONE_DECIMAL, rounding=ROUND_HALF_UP, context=_DECIMAL))


def _finding_index(findings: Iterable[Finding]) -> dict[tuple[str, Element], list[str]]:
    index: dict[tuple[str, Element], list[str]] = {}
    for f in findings:
        if f.service_id is not None and f.element is not None:
            index.setdefault((f.service_id, f.element), []).append(f.id)
    return index


def service_readiness(ctx: CheckContext, sid: str, findings: Iterable[Finding] = ()) -> ServiceReadiness:
    """Score one service; ``finding_ids`` link each element to the findings that explain it."""
    index = _finding_index(findings)
    evals = evaluate_service(ctx, sid)
    elements = tuple(
        ElementScore(
            element=element,
            score=ev.score,
            status=ev.status,
            reasons=ev.reasons,
            finding_ids=tuple(sorted(set(index.get((sid, element), ())))),
        )
        for element, ev in evals.items()
    )
    raw = _DECIMAL.multiply(_HUNDRED, _dot((element_weight(e.element), e.score) for e in elements))
    tier = ctx.service_tier(sid)
    return ServiceReadiness(
        service_id=sid,
        service_name=ctx.graph.nodes[sid].name,
        tier=tier,
        weight=tier_weight(tier),
        score=_publish(raw),
        elements=elements,
    )


def overall_score(services: Iterable[ServiceReadiness], critical_ids: Iterable[str]) -> float:
    """Tier-weighted mean over critical services (all services if none is critical)."""
    pool = list(services)
    critical = set(critical_ids)
    considered = [s for s in pool if s.service_id in critical] or pool
    total_weight = _dot((s.weight, 1.0) for s in considered)
    if not considered or total_weight <= 0:
        return 0.0
    return _publish(_DECIMAL.divide(_dot((s.weight, s.score) for s in considered), total_weight))


def score(ctx: CheckContext, findings: Iterable[Finding]) -> ReadinessReport:
    """Compute the readiness report for every service in the graph (sorted by service id)."""
    finding_list = list(findings)
    services = tuple(service_readiness(ctx, s.id, finding_list) for s in ctx.services())
    critical = [s.id for s in ctx.critical_services()]
    return ReadinessReport(
        overall=overall_score(services, critical),
        services=services,
        weights_version=SCORING_V1["version"],
        element_weights=dict(SCORING_V1["element_weights"]),
        tier_weights=dict(SCORING_V1["tier_weights"]),
        graph_hash=ctx.graph.content_hash(),
    )


def scoring_table_markdown() -> str:
    """Render the published scoring method (docs/scoring.md) from the code's own constants."""
    version = SCORING_V1["version"]
    lines = [
        f"# 준비도 점수 산식 (SCORING_{version.upper()})",
        "",
        "> 점수는 고정 규칙으로만 계산합니다. 생성형 AI는 판정·점수에 관여하지 않습니다.",
        "> 같은 그래프에는 언제나 같은 점수가 나옵니다.",
        "",
        "## 요소 가중치",
        "",
        "| 요소 | 키 | 가중치 |",
        "|---|---|---:|",
    ]
    lines += [f"| {ELEMENT_LABEL_KO[e]} | `{e.value}` | {element_weight(e):.2f} |" for e in Element]
    lines += ["", "## 서비스 등급(tier) 가중치", "", "| 등급 | 구분 | 가중치 |", "|---:|---|---:|"]
    lines += [
        f"| {tier} | {_TIER_LABEL_KO.get(tier, '-')} | {weight:.1f} |"
        for tier, weight in sorted(SCORING_V1["tier_weights"].items())
    ]
    lines += [
        "",
        "표에 없는 등급은 가장 낮은 등급 가중치를 씁니다.",
        "",
        "## 계산 방법",
        "",
        "- 서비스 점수 = 100 × Σ(요소 가중치 × 요소 점수)를 소수 첫째 자리까지 반올림(5 이상 올림)",
        "- 전체 점수 = 핵심 서비스(정책의 critical_tiers 등급)의 서비스 점수를 등급 가중치로 가중 평균해 "
        "소수 첫째 자리까지 반올림",
        "  (핵심 서비스가 없으면 전체 서비스, 서비스가 없으면 0)",
        "- 계산은 10진수로 정확하게 합니다(이진 부동소수점 오차 없음). 손으로 다시 계산해도 같은 값이 "
        "나옵니다",
        "- 요소 점수는 0~1이며, 확인 불가(UNKNOWN)는 0점으로 계산합니다",
        "",
        "## 공통 원칙",
        "",
    ]
    lines += [f"- {line}" for line in GENERAL_SEMANTICS_KO]
    lines += ["", "## 요소별 판정 기준", ""]
    for element in Element:
        lines += [f"### {ELEMENT_LABEL_KO[element]} (`{element.value}`)", ""]
        lines += [f"- {line}" for line in SEMANTICS_KO[element]]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
