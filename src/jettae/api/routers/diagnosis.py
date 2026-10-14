"""Running a diagnosis (진단 실행).

The judgement (fixed rules, score, deterministic replay) never uses an LLM (P1); ``llm_mode`` only
decides whether extraction and wording may ask one, and every LLM step falls back to the rule
parser / templates when it is off or fails (P4). Only editors and above may choose the mode; other
callers always get the server's (``JETTAE_LLM_MODE``). A run requested by a viewer or an anonymous
caller is a *free scan*: it needs a free scan slot (503 when every slot is busy) and counts against
the daily budget and the caller's per-client share of it (429 when spent).

The shared demo workspace (별빛상점) is not re-diagnosed for viewers and anonymous callers: like
``POST /demo/dmc-shop``, they get its stored report back — no new run, nothing metered, nothing
written. Only when the demo has no report yet does their request run (metered as above).
"""

from __future__ import annotations

from typing import Annotated, Final

from fastapi import APIRouter, Body
from starlette.concurrency import run_in_threadpool

from jettae.api.deps import ServiceDep
from jettae.api.errors import error_responses
from jettae.api.ratelimit import BUSY_RESPONSE, ClientDep, ScanSlotsDep
from jettae.api.schemas import DiagnoseRequest
from jettae.api.security import IdentityDep, QuotaDep, Role, free_scan, llm_mode_for
from jettae.domain import DiagnosisReport

__all__ = ["TAG", "TAG_DESCRIPTION", "router"]

TAG: Final = "진단"
TAG_DESCRIPTION: Final = (
    "진단 실행 — 고정 규칙 판정·준비도 점수·리허설·수정안 초안(판정·점수·리허설에 LLM 0회)"
)

router = APIRouter(prefix="/workspaces", tags=[TAG])


@router.post(
    "/{ws}/diagnose",
    operation_id="diagnoseWorkspace",
    summary="진단 실행",
    responses={**error_responses(401, 404, 409, 422, 429, 502), 503: BUSY_RESPONSE},
)
async def diagnose(
    ws: str,
    *,
    service: ServiceDep,
    identity: IdentityDep,
    quota: QuotaDep,
    slots: ScanSlotsDep,
    client: ClientDep,
    body: Annotated[DiagnoseRequest | None, Body()] = None,
) -> DiagnosisReport:
    """현재 입력으로 진단하고 그래프 버전·보고서를 저장합니다.

    보기 전용·익명 요청은 하루(한국 시간) 무료 진단 한도(`JETTAE_DAILY_FREE_SCAN_LIMIT`, 접속 주소당
    `JETTAE_FREE_SCANS_PER_CLIENT_PER_DAY`)를 쓰고, 서버 기본 LLM 모드로만 진단하며, 동시에 돌 수 있는
    진단 수(`JETTAE_MAX_CONCURRENT_SCANS`)가 차 있으면 503입니다.
    예제(별빛상점) 워크스페이스는 보기 전용·익명 요청이면 다시 진단하지 않고 저장된 결과를 그대로 돌려줍니다
    (무료 진단 한도도 쓰지 않음). 진단하는 동안 다른 변경이 승인되면 409입니다.
    """
    if not identity.allows(Role.EDITOR):
        stored = await run_in_threadpool(service.stored_demo_report, ws)
        if stored is not None:
            return stored
    request = body if body is not None else DiagnoseRequest()
    mode = llm_mode_for(identity, request.llm_mode, service.settings.llm_mode)
    with free_scan(identity, quota, service.clock(), client=client, slots=slots):
        return await service.diagnose(ws, llm_mode=mode, actor=identity.user)
