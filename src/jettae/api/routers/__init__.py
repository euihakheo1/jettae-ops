"""REST routers of the API (ARCHITECTURE §6).

Every router talks to :class:`~jettae.orchestrator.JettaeService` only; none of them reaches the
store, the pipeline or an LLM directly. Paths are relative — :func:`jettae.api.app.create_app`
mounts :data:`API_ROUTERS` under ``/api/v1`` and :data:`ROOT_ROUTERS` (the probes ``/healthz``,
``/readyz``) at the root.

Each module names its own OpenAPI tag (``TAG``, Korean) and describes it (``TAG_DESCRIPTION``);
:data:`OPENAPI_TAGS` lists them in documentation order. Operation ids are camelCase and fixed by
hand, because the TypeScript client is generated from them: renaming a Python function must not
rename a client method.
"""

from types import ModuleType
from typing import Any, Final

from fastapi import APIRouter

from jettae.api.routers import (
    admin,
    demo,
    diagnosis,
    free_scans,
    graph,
    growth,
    health,
    patches,
    rehearsals,
    reports,
    rules,
    session,
    workspaces,
)

_API_MODULES: Final[tuple[ModuleType, ...]] = (
    workspaces,
    diagnosis,
    reports,
    graph,
    rehearsals,
    patches,
    rules,
    demo,
    free_scans,
    growth,
    session,
    admin,
)

#: Routers served under ``/api/v1``.
API_ROUTERS: Final[tuple[APIRouter, ...]] = tuple(module.router for module in _API_MODULES)
#: Routers served at the root (load balancers and orchestrators probe fixed paths).
ROOT_ROUTERS: Final[tuple[APIRouter, ...]] = (health.router,)
#: Tag descriptions for the OpenAPI document (order = order in the docs).
OPENAPI_TAGS: Final[list[dict[str, Any]]] = [
    {"name": module.TAG, "description": module.TAG_DESCRIPTION} for module in (*_API_MODULES, health)
]

__all__ = ["API_ROUTERS", "OPENAPI_TAGS", "ROOT_ROUTERS"]
