"""The 별빛상점 demo (가상 회사, 상암 DMC): one call creates — or reuses — the example workspace,
loads ``examples/dmc-shop`` with its hand-written drills, and diagnoses it.

The demo is the public site's main call to action, and the server already diagnoses it at start-up
(``jettae demo``). So for viewers and anonymous callers the call only *opens* it: when the demo
workspace has a report, that report is returned as it is — no new diagnosis, nothing taken from
the free-scan budget, nothing written. A diagnosis runs only for editors and above, or when there
is no demo report yet; it is then metered like any free scan for viewers and anonymous callers
(scan slot, daily budget, per-client share).

The LLM mode is the server's (``JETTAE_LLM_MODE``) unless an editor or above asks for another, so a
venue laptop started with ``JETTAE_LLM_MODE=replay`` makes the web app's demo button replay the
recorded model answers, and no visitor can switch the shared demo to a live model.
"""

from __future__ import annotations

from typing import Annotated, Final

from fastapi import APIRouter, Body
from starlette.concurrency import run_in_threadpool

from jettae.api.deps import ServiceDep
from jettae.api.errors import ApiProblem, ErrorCode, error_responses
from jettae.api.ratelimit import BUSY_RESPONSE, ClientDep, ScanSlotsDep
from jettae.api.schemas import DemoRequest, DemoResult
from jettae.api.security import IdentityDep, QuotaDep, Role, free_scan, llm_mode_for
from jettae.orchestrator import DEMO_WORKSPACE_NAME

__all__ = ["TAG", "TAG_DESCRIPTION", "router"]

TAG: Final = "데모"
TAG_DESCRIPTION: Final = "별빛상점(상암 DMC) 예제 워크스페이스 — 만들고(또는 다시 쓰고) 바로 진단"

router = APIRouter(prefix="/demo", tags=[TAG])


@router.post(
    "/dmc-shop",
    operation_id="bootstrapDemo",
    summary="별빛상점 예제 진단",
    responses={
        **error_responses(401, 409, 422, 429),
        503: {
            **BUSY_RESPONSE,
            "description": (
                "예제 묶음이 없음(demo_unavailable) 또는 다른 진단이 진행 중(server_busy, Retry-After)"
            ),
        },
    },
)
async def bootstrap_demo(
    *,
    service: ServiceDep,
    identity: IdentityDep,
    quota: QuotaDep,
    slots: ScanSlotsDep,
    client: ClientDep,
    body: Annotated[DemoRequest | None, Body()] = None,
) -> DemoResult:
    """예제 워크스페이스를 만들거나 다시 쓰고, 예제 입력을 올려 진단한 결과를 돌려줍니다.

    보기 전용·익명 요청은 이미 진단된 예제가 있으면 그 결과를 그대로 돌려줍니다(새로 진단하지 않고
    무료 진단 한도도 쓰지 않음). 다시 진단하는 것은 편집자 이상이거나 예제 결과가 아직 없을 때뿐이며,
    LLM 모드는 편집자 이상만 고를 수 있습니다(그 밖에는 서버 기본값).
    다시 부르면 같은 워크스페이스를 쓰며, 입력이 그대로면 승인한 수정안이 반영된 그래프를 유지합니다.
    """
    if not identity.allows(Role.EDITOR):
        existing = await run_in_threadpool(service.demo_report)
        if existing is not None:
            return DemoResult(
                workspace_id=existing.workspace_id, workspace_name=DEMO_WORKSPACE_NAME, report=existing
            )
    requested = body.llm_mode if body is not None else None
    mode = llm_mode_for(identity, requested, service.settings.llm_mode)
    with free_scan(identity, quota, service.clock(), client=client, slots=slots):
        try:
            report = await service.bootstrap_demo(llm_mode=mode, actor=identity.user)
        except FileNotFoundError as exc:
            # the loader's message lists server paths; the client only needs to know what to fix
            raise ApiProblem(
                503,
                "예제 묶음(examples/dmc-shop)을 찾을 수 없어 데모를 시작할 수 없습니다. "
                "서버 관리자에게 JETTAE_DEMO_BUNDLE 설정을 요청하세요.",
                code=ErrorCode.DEMO_UNAVAILABLE,
            ) from exc
    workspace = await run_in_threadpool(service.get_workspace, report.workspace_id)
    return DemoResult(workspace_id=workspace.id, workspace_name=workspace.name, report=report)
