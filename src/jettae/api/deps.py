"""FastAPI dependencies that hand routers the application's shared objects.

Everything a request needs lives on ``app.state`` — set by :func:`jettae.api.app.create_app` and its
lifespan — so routers never import a global and tests can run several apps side by side (one per
settings / store) in the same process.

The accessors are ``async`` on purpose: they only read attributes, and a plain ``def`` dependency
would cost a thread-pool hop on every request.
"""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, Request

from jettae.api.errors import ApiProblem, ErrorCode
from jettae.config import Settings
from jettae.domain import DiagnosisReport
from jettae.domain.ports import NotFoundError
from jettae.orchestrator import JettaeService

__all__ = [
    "ServiceDep",
    "SettingsDep",
    "get_service",
    "get_settings",
    "latest_report_or_none",
]


async def get_service(request: Request) -> JettaeService:
    """The application service; 503 while the lifespan has not built it (or after shutdown)."""
    service = getattr(request.app.state, "service", None)
    if not isinstance(service, JettaeService):
        raise ApiProblem(
            503,
            "서버가 아직 준비 중입니다. 잠시 후 다시 시도하세요.",
            code=ErrorCode.SERVICE_UNAVAILABLE,
        )
    return service


async def get_settings(request: Request) -> Settings:
    """The settings the app was created with (auth mode, upload limit, free-scan limit …)."""
    return cast(Settings, request.app.state.settings)


ServiceDep = Annotated[JettaeService, Depends(get_service)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


def latest_report_or_none(service: JettaeService, workspace_id: str) -> DiagnosisReport | None:
    """The latest report, or None when the workspace has not been diagnosed yet.

    For views that are useful without a report (workspace list, graph): "not diagnosed" is a
    state to show there, not an error. An unknown workspace still raises ``NotFoundError``.
    """
    service.get_workspace(workspace_id)
    try:
        return service.latest_report(workspace_id)
    except NotFoundError:
        return None
