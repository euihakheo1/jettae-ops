"""REST API of JETTAE Ops (FastAPI) — a thin delivery layer over
:class:`~jettae.orchestrator.JettaeService`.

``create_app()`` returns the ASGI application (``uvicorn "jettae.api:create_app" --factory``);
the routers map HTTP to service calls and domain errors to status codes, nothing more. No judgement
happens here (P1): every verdict, score and replay comes from the service.
"""

from jettae.api.app import API_PREFIX, DOCS_URL, OPENAPI_URL, create_app

__all__ = ["API_PREFIX", "DOCS_URL", "OPENAPI_URL", "create_app"]
