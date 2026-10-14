"""Mock-incident drills (모의 장애 리허설).

A replay is a deterministic discrete-event simulation on the head graph (P1, P6): the same graph,
drill and settings always give the same result hash. An LLM may only *propose* drills (``llm`` /
``all`` with LIVE or REPLAY), and every proposal is checked against the graph before it runs.
"""

from __future__ import annotations

from typing import Annotated, Final

from fastapi import APIRouter, Body

from jettae.api.deps import ServiceDep
from jettae.api.errors import error_responses
from jettae.api.schemas import RehearsalRequest
from jettae.api.security import EditorDep, IdentityDep
from jettae.domain import RehearsalResult, Scenario

__all__ = ["TAG", "TAG_DESCRIPTION", "router"]

TAG: Final = "리허설"
TAG_DESCRIPTION: Final = "모의 장애 시나리오와 결정적 리플레이(같은 그래프·시나리오면 언제나 같은 결과)"

router = APIRouter(prefix="/workspaces", tags=[TAG])


@router.get(
    "/{ws}/scenarios",
    operation_id="listScenarios",
    summary="저장된 시나리오",
    responses=error_responses(401, 404, 422),
)
def list_scenarios(ws: str, service: ServiceDep, identity: IdentityDep) -> list[Scenario]:
    """워크스페이스에 저장된 시나리오(템플릿·사람 작성·LLM 제안)를 id순으로 돌려줍니다.

    무료 검진 워크스페이스는 템플릿 시나리오를 따로 저장하지 않으므로(진단 결과 `scenarios`에 들어 있음)
    사람 작성·LLM 제안 시나리오만 나옵니다.
    """
    return service.scenarios(ws)


@router.post(
    "/{ws}/rehearsals",
    operation_id="runRehearsals",
    summary="리허설 실행",
    responses=error_responses(401, 403, 404, 422, 502),
)
async def run_rehearsals(
    ws: str,
    service: ServiceDep,
    identity: EditorDep,
    body: Annotated[RehearsalRequest | None, Body()] = None,
) -> list[RehearsalResult]:
    """최신 그래프에서 시나리오를 재생하고 결과를 저장합니다(편집자 이상).

    `scenario_ids`를 주면 그 시나리오만, 아니면 `generate`가 고른 출처의 시나리오를 재생합니다.
    지금 그래프에 없는 대상을 가리키는 시나리오는 422입니다(없는 것을 끊으면 이유 없이 통과하므로).
    """
    request = body if body is not None else RehearsalRequest()
    generate, llm_mode = request.resolved()
    return await service.rehearse(
        ws, scenario_ids=request.scenario_ids, generate=generate, llm_mode=llm_mode, actor=identity.user
    )
