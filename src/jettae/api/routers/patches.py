"""Fixes (수정안): draft → validate → human decision → new graph version.

Drafts are written by templates (an LLM may only reword the sentence and pick among roster
candidates; the graph ops are always compiled by code), then re-checked and re-rehearsed before a
person sees them. Only ``lead``/``admin`` may approve or reject; an approval appends a version and
re-judges it with the fixed rules, without any LLM (P1, P4).
"""

from __future__ import annotations

from typing import Annotated, Final

from fastapi import APIRouter, Body

from jettae.api.deps import ServiceDep
from jettae.api.errors import error_responses
from jettae.api.schemas import ApprovalResult, DraftRequest, RejectRequest
from jettae.api.security import DeciderDep, EditorDep, IdentityDep
from jettae.domain import Patch

__all__ = ["TAG", "TAG_DESCRIPTION", "router"]

TAG: Final = "수정안"
TAG_DESCRIPTION: Final = "수정안 초안·검증 결과·승인·거절(승인·거절은 팀 리드·관리자)"

router = APIRouter(tags=[TAG])


@router.get(
    "/workspaces/{ws}/patches",
    operation_id="listPatches",
    summary="수정안 목록",
    responses=error_responses(401, 404, 422),
)
def list_patches(ws: str, service: ServiceDep, identity: IdentityDep) -> list[Patch]:
    """워크스페이스의 모든 수정안(초안·검증됨·승인·거절·무효)을 만든 순서대로 돌려줍니다."""
    return service.patches(ws)


@router.post(
    "/workspaces/{ws}/patches/draft",
    operation_id="draftPatches",
    summary="수정안 초안 만들기",
    responses=error_responses(401, 403, 404, 409, 422, 502),
)
async def draft_patches(
    ws: str,
    service: ServiceDep,
    identity: EditorDep,
    body: Annotated[DraftRequest | None, Body()] = None,
) -> list[Patch]:
    """최신 그래프의 끊김(전부 또는 `finding_ids`)에 대한 수정안을 만들고 검증해 돌려줍니다(편집자 이상)."""
    request = body if body is not None else DraftRequest()
    return await service.draft_patches(
        ws, finding_ids=request.finding_ids, llm_mode=request.llm_mode, actor=identity.user
    )


@router.get(
    "/patches/{patch_id}",
    operation_id="getPatch",
    summary="수정안 조회",
    responses=error_responses(401, 404, 422),
)
def get_patch(patch_id: str, service: ServiceDep, identity: IdentityDep) -> Patch:
    return service.get_patch(patch_id)


@router.post(
    "/patches/{patch_id}/approve",
    operation_id="approvePatch",
    summary="수정안 승인",
    responses=error_responses(401, 403, 404, 409, 422),
)
def approve_patch(patch_id: str, service: ServiceDep, identity: DeciderDep) -> ApprovalResult:
    """검증을 통과한 수정안을 승인해 새 그래프 버전을 만들고 다시 판정합니다(팀 리드·관리자).

    기준 버전이 바뀌어 무효가 된 수정안은 409입니다.
    """
    result = service.approve_patch(patch_id, actor=identity.user, role=identity.role.value)
    return ApprovalResult(patch=service.get_patch(patch_id), version=result.version, report=result.report)


@router.post(
    "/patches/{patch_id}/reject",
    operation_id="rejectPatch",
    summary="수정안 거절",
    responses=error_responses(401, 403, 404, 409, 422),
)
def reject_patch(patch_id: str, body: RejectRequest, service: ServiceDep, identity: DeciderDep) -> Patch:
    """수정안을 거절합니다(팀 리드·관리자). 사유는 감사 로그에 남습니다."""
    return service.reject_patch(patch_id, actor=identity.user, role=identity.role.value, reason=body.reason)
