"""The operations graph and its version history (운영 그래프·버전).

Versions are append-only (P7): a revert does not rewrite history, it appends a version whose graph
equals the chosen one and re-judges it with the fixed rules — which is why it needs the same
``lead``/``admin`` role as an approval.
"""

from __future__ import annotations

from typing import Final

from fastapi import APIRouter, Path, Query

from jettae.api.deps import ServiceDep, latest_report_or_none
from jettae.api.errors import error_responses
from jettae.api.schemas import ApprovalResult, GraphDiffView, GraphView, graph_diff_view, graph_view
from jettae.api.security import DeciderDep, IdentityDep
from jettae.domain import GraphVersion
from jettae.orchestrator import JettaeService

__all__ = ["TAG", "TAG_DESCRIPTION", "router"]

TAG: Final = "그래프"
TAG_DESCRIPTION: Final = "운영 그래프와 버전 이력·비교·되돌리기(되돌리기도 새 버전, 팀 리드·관리자)"

router = APIRouter(prefix="/workspaces", tags=[TAG])


@router.get(
    "/{ws}/graph",
    operation_id="getGraph",
    summary="운영 그래프",
    responses=error_responses(401, 404, 422),
)
def get_graph(
    ws: str,
    service: ServiceDep,
    identity: IdentityDep,
    version: int | None = Query(default=None, ge=1, description="그래프 버전(비우면 최신)"),
) -> GraphView:
    """그래프를 그리기 좋은 형태로 돌려줍니다(노드·엣지 id순, 근거 span, 요소별 끊김 id).

    요소별 끊김은 최신 진단이 **이 버전**을 판정했을 때만 채웁니다.
    """
    head, graph = service.graph(ws, version)
    return graph_view(head, graph, latest_report_or_none(service, ws))


@router.get(
    "/{ws}/versions",
    operation_id="listVersions",
    summary="그래프 버전 이력",
    responses=error_responses(401, 404, 422),
)
def list_versions(ws: str, service: ServiceDep, identity: IdentityDep) -> list[GraphVersion]:
    """모든 그래프 버전을 번호순으로 돌려줍니다(승인된 수정안·되돌리기 포함)."""
    return service.versions(ws)


def _diff(service: JettaeService, ws: str, a: int, b: int) -> GraphDiffView:
    ids = service.diff(ws, a, b)
    _, before = service.graph(ws, a)
    _, after = service.graph(ws, b)
    return graph_diff_view(a, b, ids, before, after)


_A = Query(ge=1, description="비교 기준 버전")
_B = Query(ge=1, description="비교 대상 버전")


@router.get(
    "/{ws}/diff",
    operation_id="diffVersions",
    summary="두 버전 비교",
    responses=error_responses(401, 404, 422),
)
def diff_versions(
    ws: str, service: ServiceDep, identity: IdentityDep, a: int = _A, b: int = _B
) -> GraphDiffView:
    """버전 a → b 사이에 추가·삭제·변경된 노드와 엣지(변경은 속성별 전후 값 포함)."""
    return _diff(service, ws, a, b)


@router.get("/{ws}/versions/diff", include_in_schema=False)
def diff_versions_legacy(
    ws: str, service: ServiceDep, identity: IdentityDep, a: int = _A, b: int = _B
) -> GraphDiffView:
    """The path the current web client calls (same answer as ``/diff``; not in the schema, so a
    regenerated client uses the documented one)."""
    return _diff(service, ws, a, b)


@router.post(
    "/{ws}/versions/{n}/revert",
    operation_id="revertVersion",
    summary="이 버전으로 되돌리기",
    responses=error_responses(401, 403, 404, 409, 422),
)
def revert_version(
    ws: str,
    service: ServiceDep,
    identity: DeciderDep,
    n: int = Path(ge=1, description="되돌아갈 그래프 버전"),
) -> ApprovalResult:
    """버전 n과 같은 그래프를 **새 버전**으로 추가하고 고정 규칙으로 다시 판정합니다(팀 리드·관리자)."""
    result = service.revert(ws, n, actor=identity.user, role=identity.role.value)
    return ApprovalResult(patch=None, version=result.version, report=result.report)
