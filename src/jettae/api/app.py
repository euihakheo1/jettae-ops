"""The FastAPI application: ``create_app()`` wires settings, the service, routers and middleware.

* REST API under ``/api/v1`` (ARCHITECTURE §6), probes at ``/healthz`` and ``/readyz``, the OpenAPI
  document at ``/api/v1/openapi.json`` and interactive docs at ``/api/docs``.
* The built web app (``JETTAE_WEB_DIST``, default ``web/dist``) is served at ``/`` when it exists,
  with an SPA fallback: an unknown path outside ``/api`` returns ``index.html`` so deep links such
  as ``/app/w/<ws>/patches`` survive a reload. A missing ``/assets/*`` file stays a 404 — answering
  a stale script request with HTML would only turn a clean "not found" into a confusing parse
  error in the browser.
* Every router gets :func:`jettae.api.access.workspace_access`, which keeps anonymous visitors of
  the public site (``JETTAE_AUTH_MODE=public``) out of other people's workspaces (404).

The service is either injected (tests, the CLI that already built one) or built by the lifespan
from ``settings.database_url``; only a store the app opened itself is closed on shutdown.

Retention: in ``public`` mode only (``JETTAE_AUTH_MODE=public``, the public beta) the lifespan runs
:meth:`~jettae.orchestrator.JettaeService.purge_expired` once at start-up and then every
:data:`PURGE_INTERVAL_SECONDS` in a background task, cancelled on shutdown. The retention periods
(``JETTAE_FREE_SCAN_RETENTION_DAYS``, ``JETTAE_SIGNUP_RETENTION_DAYS``) are what the public site's
privacy notice promises; a ``dev`` or ``token`` server — a developer's database, a team's own
install — never deletes anything on its own (calling ``purge_expired`` directly still works). A
failed purge is logged (class names only — a database error's message would quote
sign-up data) and retried at the next tick; it never stops the app.

Abuse limits for the public site live on ``app.state`` too (one set per app, i.e. per process):
the daily free-scan budget with its per-client share, the scan slots, the per-client rate
limiters, the daily cap on stored events and the client-address resolver
(``JETTAE_CLIENT_IP_SOURCE``) of :mod:`jettae.api.ratelimit`. A malformed setting stops the app at
creation (``ValueError`` with a Korean message), like a malformed token table.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import Final

import structlog
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response
from starlette.routing import Match, Mount
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

from jettae import __version__
from jettae.api.access import workspace_access
from jettae.api.errors import install_error_handlers
from jettae.api.middleware import (
    REQUEST_ID_HEADER,
    BodySizeLimitMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
    UnhandledErrorMiddleware,
    configure_logging,
)
from jettae.api.ratelimit import (
    FREE_SCAN_ATTEMPTS_PER_MINUTE,
    ClientIpResolver,
    DailyCap,
    RateLimiter,
    ScanSlots,
)
from jettae.api.routers import API_ROUTERS, OPENAPI_TAGS, ROOT_ROUTERS
from jettae.api.security import (
    ROLE_HEADER,
    USER_HEADER,
    WORKSPACE_KEY_HEADER,
    Authenticator,
    FreeScanQuota,
)
from jettae.config import Settings, get_settings
from jettae.orchestrator import JettaeService
from jettae.store import database_error_summary, open_store

__all__ = [
    "API_PREFIX",
    "DOCS_URL",
    "OPENAPI_URL",
    "PURGE_INTERVAL_SECONDS",
    "SpaStaticFiles",
    "create_app",
    "purge_once",
]

_log = structlog.get_logger("jettae.api.app")

API_PREFIX: Final = "/api/v1"
OPENAPI_URL: Final = f"{API_PREFIX}/openapi.json"
DOCS_URL: Final = "/api/docs"
#: JSON bodies are small (ids, a reason, a sign-up); anything bigger is a mistake or an attack.
MAX_JSON_BODY: Final = 1024 * 1024
#: multipart framing (boundaries, part headers) on top of the files themselves
MULTIPART_ENVELOPE: Final = 256 * 1024
#: routes whose bodies are file uploads (the upload budget applies instead of MAX_JSON_BODY)
_UPLOAD_PATH: Final = re.compile(rf"{re.escape(API_PREFIX)}/(?:workspaces/[^/]+/inputs|free-scans)")
#: How often the retention purge runs after the one at start-up (6 hours).
PURGE_INTERVAL_SECONDS: Final = 6 * 60 * 60

_DESCRIPTION: Final = """
제때옵스(JETTAE Ops) — 장애 대응 런북을 **운영 그래프**로 바꾸고, **고정 규칙**으로 끊긴 대응 경로를 찾고,
**결정적 리플레이**로 모의 장애를 돌리고, **사람이 승인한 수정안**만 새 그래프 버전으로 만듭니다.

* 판정·점수·리허설에는 LLM을 쓰지 않습니다(LLM 호출 0회). LLM은 추출·문장 다듬기에만 쓰이며 꺼도 동작합니다.
* 오류 응답은 언제나 `{"detail": "한국어 설명", "code": "기계용 코드", "request_id": "…"}` 형태입니다.
* 인증: 개발 모드는 `X-Jettae-User`·`X-Jettae-Role`(viewer·editor·lead·admin) 헤더,
  토큰 모드는 `Authorization: Bearer <토큰>`입니다. 공개 모드는 토큰이 없으면 익명 보기 전용(viewer)이며,
  익명 방문자는 예제 워크스페이스와 무료 검진(`POST /api/v1/free-scans`)이 돌려준 접근 키를
  `X-Jettae-Workspace-Key` 헤더로 보낸 워크스페이스만 열 수 있습니다(그 밖에는 404).
* 쓰기는 편집자(editor) 이상, 승인·거절·되돌리기는 팀 리드(lead)·관리자(admin)만 합니다.
  사전 신청 명단(개인정보)은 관리자(admin)만 봅니다.
"""

#: paths the SPA must never answer for (they belong to the API, even when no route matches)
_BACKEND_PREFIXES: Final = ("/api/", "/healthz/", "/readyz/")
_BACKEND_PATHS: Final = frozenset({"/api", "/healthz", "/readyz"})


def _is_backend(path: str) -> bool:
    return path in _BACKEND_PATHS or path.startswith(_BACKEND_PREFIXES)


def _route_path(scope: Scope) -> str:
    """The request path below the app's ``root_path`` (set when served under a proxy prefix)."""
    path = str(scope.get("path", "/"))
    root = str(scope.get("root_path", "")).rstrip("/")
    if root and (path == root or path.startswith(f"{root}/")):
        return path[len(root) :] or "/"
    return path


class _SpaMount(Mount):
    """The web app's mount at ``/``, which never claims a backend path.

    A plain ``Mount("/")`` matches *every* path fully, and Starlette's router prefers a full match
    to the partial one an API route gives for a wrong method — so ``DELETE /api/v1/workspaces``
    would reach the SPA and come back 404 instead of 405 with an ``Allow`` header. Declining
    backend paths here leaves them to the router: 405 for a known path with the wrong method, the
    Korean JSON 404 for an unknown one.
    """

    def matches(self, scope: Scope) -> tuple[Match, Scope]:
        if scope["type"] == "http" and _is_backend(_route_path(scope)):
            return Match.NONE, {}
        return super().matches(scope)


class SpaStaticFiles(StaticFiles):
    """Static files of the built web app with the single-page-app fallback described above."""

    def __init__(self, *, directory: Path) -> None:
        # follow_symlink: Starlette compares realpaths, and on Windows a realpath of 260+ characters
        # comes back \\?\-prefixed, so every asset under a long checkout path (or a subst drive onto
        # one) would 404. With it the check is a normalised-path prefix test, which still refuses
        # ``..`` traversal (covered by tests).
        super().__init__(directory=directory, html=True, follow_symlink=True)
        self.index = directory / "index.html"

    def _index(self) -> Response:
        # always revalidated: a new build must reach users at their next navigation
        return FileResponse(self.index, media_type="text/html", headers={"Cache-Control": "no-cache"})

    async def get_response(self, path: str, scope: Scope) -> Response:
        url_path = _route_path(scope)
        # refused here too (not only by the mount), so the class stays safe when mounted directly
        if _is_backend(url_path):
            raise StarletteHTTPException(status_code=404)
        is_asset = url_path.startswith("/assets/")
        try:
            response = await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code != 404 or is_asset:
                raise
            return self._index()
        if is_asset and response.status_code == 200:
            # Vite puts a content hash in every asset name, so an asset never changes
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        elif url_path in ("/", "/index.html"):
            response.headers["Cache-Control"] = "no-cache"
        return response


async def purge_once(service: JettaeService) -> None:
    """One retention purge (:meth:`JettaeService.purge_expired`) that never raises.

    A failure is logged with class names only: a database error's message quotes the statement's
    values, and a sign-up purge touches e-mail addresses.
    """
    try:
        outcome = await run_in_threadpool(service.purge_expired)
    except Exception as exc:  # the app must outlive a failed purge
        _log.warning(
            "retention.purge_failed", error=type(exc).__name__, **(database_error_summary(exc) or {})
        )
        return
    if outcome.workspaces or outcome.signups:
        _log.info("retention.purged", free_scan_workspaces=outcome.workspaces, signups=outcome.signups)


async def _purge_periodically(service: JettaeService, interval: float) -> None:
    while True:
        await asyncio.sleep(interval)
        await purge_once(service)


@asynccontextmanager
async def _retention(service: JettaeService) -> AsyncIterator[None]:
    """Purge now, then every :data:`PURGE_INTERVAL_SECONDS` until the block ends."""
    await purge_once(service)
    task = asyncio.create_task(_purge_periodically(service, PURGE_INTERVAL_SECONDS))
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def _retention_for(settings: Settings, service: JettaeService) -> AbstractAsyncContextManager[None]:
    """The retention purge in ``public`` mode, nothing in the other modes (module docstring)."""
    if settings.auth_mode == "public":
        return _retention(service)
    return contextlib.nullcontext()


def _lifespan(
    settings: Settings, injected: JettaeService | None
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(json_logs=settings.log_json)
        if injected is not None:
            async with _retention_for(settings, injected):
                yield
            return
        store = await run_in_threadpool(open_store, settings.database_url)
        service = JettaeService(store, settings)
        app.state.service = service
        try:
            async with _retention_for(settings, service):
                yield
        finally:
            app.state.service = None
            await run_in_threadpool(store.close)

    return lifespan


def _install_middleware(app: FastAPI, settings: Settings) -> None:
    """Outermost last (Starlette wraps in reverse order of registration)."""
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(
        BodySizeLimitMiddleware,
        max_body=MAX_JSON_BODY,
        max_upload_mb=settings.max_upload_mb,
        upload_envelope=MULTIPART_ENVELOPE,
        upload_path=_UPLOAD_PATH,
    )
    app.add_middleware(UnhandledErrorMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            USER_HEADER,
            ROLE_HEADER,
            WORKSPACE_KEY_HEADER,
            REQUEST_ID_HEADER,
        ],
        expose_headers=[REQUEST_ID_HEADER, "Content-Disposition", "Retry-After"],
        max_age=600,
    )
    app.add_middleware(SecurityHeadersMiddleware, docs_path=DOCS_URL)
    app.add_middleware(RequestContextMiddleware)


def create_app(settings: Settings | None = None, *, service: JettaeService | None = None) -> FastAPI:
    """Build the API application.

    ``settings`` defaults to the injected service's settings, else to the environment
    (``JETTAE_*``). The token table and the client-address source are validated here, so a
    malformed ``JETTAE_API_TOKENS`` or ``JETTAE_CLIENT_IP_SOURCE`` stops the server at start-up
    (``ValueError``) instead of failing requests later.
    """
    resolved = (
        settings if settings is not None else service.settings if service is not None else get_settings()
    )
    authenticator = Authenticator(resolved.auth_mode, resolved.api_tokens)
    client_ip = ClientIpResolver.from_setting(resolved.client_ip_source)
    app = FastAPI(
        title="제때옵스 API",
        version=__version__,
        description=_DESCRIPTION.strip(),
        openapi_tags=OPENAPI_TAGS,
        openapi_url=OPENAPI_URL,
        docs_url=DOCS_URL,
        redoc_url=None,
        lifespan=_lifespan(resolved, service),
    )
    app.state.settings = resolved
    app.state.authenticator = authenticator
    app.state.free_scans = FreeScanQuota(
        resolved.daily_free_scan_limit, per_client=resolved.free_scans_per_client_per_day
    )
    app.state.scan_slots = ScanSlots(resolved.max_concurrent_scans)
    app.state.event_limiter = RateLimiter(resolved.events_per_client_per_minute, 60, subject="이벤트 기록")
    app.state.signup_limiter = RateLimiter(resolved.signups_per_client_per_hour, 60 * 60, subject="사전 신청")
    app.state.free_scan_attempts = RateLimiter(FREE_SCAN_ATTEMPTS_PER_MINUTE, 60, subject="무료 검진")
    app.state.event_cap = DailyCap(resolved.events_per_day, subject="events")
    app.state.client_ip = client_ip
    app.state.service = service
    install_error_handlers(app)
    guard = [Depends(workspace_access)]
    for router in API_ROUTERS:
        app.include_router(router, prefix=API_PREFIX, dependencies=guard)
    for router in ROOT_ROUTERS:
        app.include_router(router, dependencies=guard)
    _install_middleware(app, resolved)
    web_dist = resolved.web_dist
    if web_dist is not None and (web_dist / "index.html").is_file():
        app.router.routes.append(_SpaMount("/", app=SpaStaticFiles(directory=web_dist), name="web"))
    return app
