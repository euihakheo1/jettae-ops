"""The published fixed-rule catalog (고정 규칙 카탈로그).

Public on purpose, also in token mode: the rules *are* the judge (P1), and customers, auditors and
prospective users must be able to read exactly what it checks without an account.
"""

from __future__ import annotations

from typing import Final

from fastapi import APIRouter

from jettae.api.deps import ServiceDep
from jettae.domain import RuleInfo

__all__ = ["TAG", "TAG_DESCRIPTION", "router"]

TAG: Final = "규칙"
TAG_DESCRIPTION: Final = "공개 고정 규칙 카탈로그 — 판정이 하는 일 그대로(인증 없이 열람)"

router = APIRouter(tags=[TAG])


@router.get("/rules", operation_id="listRules", summary="고정 규칙 카탈로그")
def list_rules(service: ServiceDep) -> list[RuleInfo]:
    """끊김 판정에 쓰는 고정 규칙 전부(규칙 id·심각도·5요소·근거·법적 기준)를 돌려줍니다."""
    return service.rule_catalog()
