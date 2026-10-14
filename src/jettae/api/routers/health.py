"""Liveness and readiness probes (served at the root, outside ``/api/v1``).

``/healthz`` answers as long as the process serves HTTP — it touches nothing, so a slow database
never gets a healthy process restarted. ``/readyz`` asks the store a cheap point query, so a load
balancer only routes traffic to an instance that can actually read and write. Failure details stay
in the server log: a connection error can carry the database URL, credentials included.
"""

from __future__ import annotations

from typing import Final

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from jettae import __version__
from jettae.api.schemas import HealthStatus, ReadinessStatus
from jettae.domain.ports import NotFoundError
from jettae.orchestrator import JettaeService

__all__ = ["READINESS_PROBE_ID", "TAG", "TAG_DESCRIPTION", "router"]

TAG: Final = "상태"
TAG_DESCRIPTION: Final = "서버 상태 확인 — 살아 있음(/healthz)·준비됨(/readyz, 저장소 연결)"

#: Workspace id looked up by the readiness probe; "not found" is the expected, healthy answer.
READINESS_PROBE_ID: Final = "readyz-probe"

router = APIRouter(tags=[TAG])
_log = structlog.get_logger("jettae.api.health")


@router.get("/healthz", operation_id="healthz", summary="살아 있음")
async def healthz() -> HealthStatus:
    return HealthStatus(version=__version__)


@router.get(
    "/readyz",
    operation_id="readyz",
    summary="준비됨(저장소 연결)",
    response_model=ReadinessStatus,
    responses={503: {"model": ReadinessStatus, "description": "저장소에 연결할 수 없음"}},
)
def readyz(request: Request) -> ReadinessStatus | JSONResponse:
    service = getattr(request.app.state, "service", None)
    if not isinstance(service, JettaeService):
        return _unavailable("서버가 아직 시작 중입니다.")
    try:
        service.get_workspace(READINESS_PROBE_ID)
    except NotFoundError:
        pass
    except Exception as exc:
        _log.error("readyz.store_unreachable", error=type(exc).__name__)
        return _unavailable("저장소(데이터베이스)에 연결할 수 없습니다.")
    return ReadinessStatus(status="ready")


def _unavailable(detail: str) -> JSONResponse:
    body = ReadinessStatus(status="unavailable", detail=detail)
    return JSONResponse(body.model_dump(), status_code=503)
