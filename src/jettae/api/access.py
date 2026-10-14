"""Private workspaces on the public site (공개 모드의 워크스페이스 접근 키).

In ``JETTAE_AUTH_MODE=public`` an anonymous visitor may open only

* the demo workspace (별빛상점), and
* a workspace whose access key they present in ``X-Jettae-Workspace-Key`` — the key their free
  scan returned (:meth:`~jettae.orchestrator.JettaeService.grant_workspace_key`).

Workspace ids are identifiers, not secrets (deterministic, and they appear in URLs and logs), so
the key is what keeps one visitor's scan away from the next visitor. Any other workspace answers
exactly like an unknown id (404 ``not_found``): a visitor cannot tell a private workspace from a
missing one. The same holds for a patch addressed by its own id (``/patches/{patch_id}``): it is
visible only if its workspace is. Token holders, and the ``dev`` and ``token`` modes, are not
affected at all.

:func:`workspace_access` is added to *every* router by :func:`jettae.api.app.create_app` and
reads the ``{ws}`` / ``{patch_id}`` path parameters itself, so a route added later under
``/workspaces/{ws}`` is guarded without anyone having to remember it; other routes pass straight
through. It establishes the caller only when it has to — a router-wide ``IdentityDep`` would make
the open routes (rules, sign-ups, events) demand a token in ``token`` mode.
"""

from __future__ import annotations

from typing import Final, cast

from fastapi import Request
from starlette.concurrency import run_in_threadpool

from jettae.api.deps import get_service
from jettae.api.security import WORKSPACE_KEY_HEADER, current_identity
from jettae.config import Settings

__all__ = ["PATCH_PATH_PARAM", "WORKSPACE_PATH_PARAM", "workspace_access"]

#: Name of the path parameter that addresses a workspace in every router.
WORKSPACE_PATH_PARAM: Final = "ws"
#: Name of the path parameter that addresses a patch (which belongs to one workspace).
PATCH_PATH_PARAM: Final = "patch_id"


async def workspace_access(request: Request) -> None:
    """Router-wide dependency: 404 for an anonymous public-mode caller who may not open the
    workspace addressed by ``{ws}`` (or owning ``{patch_id}``)."""
    workspace_id = request.path_params.get(WORKSPACE_PATH_PARAM)
    patch_id = request.path_params.get(PATCH_PATH_PARAM)
    if not isinstance(workspace_id, str) and not isinstance(patch_id, str):
        return
    if cast(Settings, request.app.state.settings).auth_mode != "public":
        return
    identity = await current_identity(request)
    if not identity.anonymous:
        return
    service = await get_service(request)
    key = request.headers.get(WORKSPACE_KEY_HEADER)
    if isinstance(workspace_id, str):
        await run_in_threadpool(service.workspace_for_visitor, workspace_id, key)
    if isinstance(patch_id, str):
        await run_in_threadpool(service.patch_for_visitor, patch_id, key)
