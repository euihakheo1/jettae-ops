"""ASGI middleware of the API: request ids, access log, security headers, body limits, crash shield.

All of them are plain ASGI callables rather than ``BaseHTTPMiddleware``: they only look at the
``http.response.start`` message or wrap ``receive``, so responses (report downloads, static files)
stream through untouched and no extra task is spawned per request.

Order (outermost first, see :func:`jettae.api.app.create_app`)::

    RequestContext → SecurityHeaders → CORS → UnhandledError → BodySizeLimit → GZip → routes

so every response — including CORS preflights, 413s decided before routing and the 500 that
replaces an unexpected crash — carries the request id and the security headers, and the 500 still
gets CORS headers a browser needs to read it.

The access log is structured (structlog) and deliberately thin: method, path, route template,
status and duration. No query strings, headers, bodies, user ids or client addresses are logged,
so the log never becomes a second store of personal data. Even the path is not logged as sent
(:func:`access_log_path`): for a matched route it is the route template with its path parameters
filled in, except parameters that are personal data (:data:`SENSITIVE_PATH_PARAMS` — the address
of ``DELETE /api/v1/signups/{email}``), which are logged as ``<redacted>``; for any other path
(an unknown API path, a page of the web app) every segment that looks like an e-mail address
becomes ``<email>``. Paths are percent-encoded ASCII whatever the console encoding. The log
replaces uvicorn's own access log, which records the client address (the visitor's real IP behind
a proxy) and the query string; ``jettae serve``/``demo`` start uvicorn with that log switched off.
"""

from __future__ import annotations

import re
import sys
import time
import uuid
from collections.abc import Mapping
from typing import Final
from urllib.parse import quote, unquote

import structlog
from starlette.datastructures import Headers, MutableHeaders
from starlette.routing import Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from structlog.types import Processor

from jettae.api.errors import ApiProblem, ErrorCode, failure_log_fields, problem_body
from jettae.store import database_error_summary

__all__ = [
    "API_CSP",
    "REQUEST_ID_HEADER",
    "SENSITIVE_PATH_PARAMS",
    "WEB_CSP",
    "BodySizeLimitMiddleware",
    "RequestContextMiddleware",
    "SecurityHeadersMiddleware",
    "UnhandledErrorMiddleware",
    "access_log_path",
    "configure_logging",
]

REQUEST_ID_HEADER: Final = "X-Request-ID"
#: an inbound id (from a proxy) is kept only if it is short and harmless in logs and headers
_INBOUND_ID: Final = re.compile(r"[A-Za-z0-9._:-]{1,64}")

#: JSON/text API responses never need to load anything.
API_CSP: Final = "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
#: The web app: own scripts only (no eval), inline styles (React Flow positions nodes with them),
#: data/blob images and the ELK layout worker.
WEB_CSP: Final = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; font-src 'self' data:; connect-src 'self'; worker-src 'self' blob:; "
    "object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
)
#: Path parameters whose values are personal data and never reach the access log.
SENSITIVE_PATH_PARAMS: Final = frozenset({"email"})
_REDACTED: Final = "<redacted>"
_EMAIL_MARK: Final = "<email>"
#: ``{name}`` or ``{name:converter}`` in a route template
_TEMPLATE_PARAM: Final = re.compile(r"\{([^{}:]+)(?::[^{}]*)?\}")
#: what a logged path may keep as is (everything else is percent-encoded, so logs stay ASCII)
_PATH_SAFE: Final = "/-._~!$&'()*+,;=:"
#: percent-decoding rounds tried when looking for an '@' (a double-encoded address too)
_DECODE_ROUNDS: Final = 3
_STATIC_HEADERS: Final = (
    ("X-Content-Type-Options", "nosniff"),
    ("X-Frame-Options", "DENY"),
    ("Referrer-Policy", "no-referrer"),
    ("Cross-Origin-Opener-Policy", "same-origin"),
    ("Permissions-Policy", "camera=(), microphone=(), geolocation=(), payment=()"),
)

_access_log = structlog.get_logger("jettae.api.access")


def configure_logging(*, json_logs: bool) -> None:
    """Configure structlog once (JSON lines when ``JETTAE_LOG_JSON``; plain text otherwise).

    A host that configured structlog already (the CLI, a test) keeps its configuration.
    """
    if structlog.is_configured():
        return
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    if json_logs:
        processors += [structlog.processors.format_exc_info, structlog.processors.JSONRenderer()]
    else:
        # plain tracebacks: the rich formatter's default shows local variables, which can hold
        # uploaded text or tokens — exactly what the log must never contain
        processors.append(
            structlog.dev.ConsoleRenderer(colors=False, exception_formatter=structlog.dev.plain_traceback)
        )
    structlog.configure(processors=processors, logger_factory=structlog.PrintLoggerFactory(sys.stderr))


def _is_api_path(path: str) -> bool:
    return path == "/api" or path.startswith("/api/") or path in ("/healthz", "/readyz")


def _logged_path(scope: Scope) -> str:
    """The path as sent (percent-encoded ASCII), so logs stay ASCII whatever the console encoding."""
    raw = scope.get("raw_path")
    if isinstance(raw, bytes):
        return raw.decode("ascii", errors="replace")
    return quote(str(scope.get("path", "")))


def _looks_like_email(segment: str) -> bool:
    text = segment
    for _ in range(_DECODE_ROUNDS):
        if "@" in text:
            return True
        decoded = unquote(text)
        if decoded == text:
            return False
        text = decoded
    return "@" in text


def _without_emails(path: str) -> str:
    """``path`` with every segment that holds an '@' (also percent-encoded) replaced by ``<email>``."""
    return "/".join(_EMAIL_MARK if _looks_like_email(part) else part for part in path.split("/"))


def _param_text(name: str, params: Mapping[str, object]) -> str:
    if name in SENSITIVE_PATH_PARAMS:
        return _REDACTED
    if name not in params:
        return f"{{{name}}}"
    value = str(params[name])
    return _EMAIL_MARK if _looks_like_email(value) else quote(value, safe=_PATH_SAFE)


def access_log_path(scope: Scope) -> str:
    """The path as the access log shows it (see the module docstring): the matched route's
    template with its non-sensitive path parameters filled in, else the path as sent with
    e-mail-like segments replaced.

    FastAPI reports an included route's template without the router prefix (``/signups/{email}``
    under ``/api/v1``), so the prefix is taken from the path as sent: the segments in front of the
    ones the template accounts for (still with e-mail-like segments replaced, for safety).
    """
    sent = _logged_path(scope)
    route = scope.get("route")
    template = route.path if isinstance(route, Route) else None
    if not template:
        return _without_emails(sent)
    params = scope.get("path_params")
    values: Mapping[str, object] = params if isinstance(params, Mapping) else {}
    filled = _TEMPLATE_PARAM.sub(lambda match: _param_text(match.group(1), values), template)
    segments = sent.split("/")
    prefix = "/".join(segments[: max(len(segments) - template.count("/"), 0)])
    return _without_emails(prefix) + filled


class RequestContextMiddleware:
    """Request id (``X-Request-ID``) for every response, bound to the log context, plus the access log."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        inbound = Headers(scope=scope).get(REQUEST_ID_HEADER, "")
        request_id = inbound if _INBOUND_ID.fullmatch(inbound) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        status = 500
        started = time.perf_counter()

        async def send_with_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        with structlog.contextvars.bound_contextvars(request_id=request_id):
            try:
                await self.app(scope, receive, send_with_id)
            finally:
                route = scope.get("route")
                log = _access_log.error if status >= 500 else _access_log.info
                log(
                    "http.request",
                    method=scope.get("method"),
                    path=access_log_path(scope),
                    route=route.path if isinstance(route, Route) else None,
                    status=status,
                    duration_ms=round((time.perf_counter() - started) * 1000, 1),
                )


class SecurityHeadersMiddleware:
    """Baseline browser hardening on every response; a header a route set itself is kept.

    The interactive docs (``/api/docs``) load Swagger UI from a CDN, so they get no CSP; HSTS is
    only sent over HTTPS (on plain HTTP it would be ignored anyway, and wrong behind a proxy).
    """

    def __init__(self, app: ASGIApp, *, docs_path: str | None) -> None:
        self.app = app
        self.docs_path = docs_path

    def _csp(self, path: str) -> str | None:
        if self.docs_path is not None and path.startswith(self.docs_path):
            return None
        return API_CSP if _is_api_path(path) else WEB_CSP

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = str(scope.get("path", ""))
        csp = self._csp(path)
        https = scope.get("scheme") == "https"
        api = _is_api_path(path)

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in _STATIC_HEADERS:
                    headers.setdefault(name, value)
                if csp is not None:
                    headers.setdefault("Content-Security-Policy", csp)
                if api:
                    # reports and graphs describe an organisation's incident response: never cache
                    headers.setdefault("Cache-Control", "no-store")
                if https:
                    headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
            await send(message)

        await self.app(scope, receive, send_with_headers)


class UnhandledErrorMiddleware:
    """Turn an unexpected exception into a Korean 500 body with the request id (and log it).

    Domain errors are mapped by the exception handlers long before this point; what reaches it
    is a bug — or a database failure (unreachable, full disk). The traceback goes to the log, never
    to the client; for a database error only class names are logged, since its messages quote the
    values being written (:func:`~jettae.api.errors.failure_log_fields`). If the response had
    already started there is nothing left to replace, so the error propagates to the server — a
    database error as a bare ``RuntimeError``, because the server prints the whole chain.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        response_started = False

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive, tracking_send)
        except Exception as exc:
            _access_log.error("http.unhandled_error", **failure_log_fields(exc))
            if response_started:
                database = database_error_summary(exc)
                if database is not None:
                    names = ", ".join(database.values())
                    raise RuntimeError(f"database error after the response started ({names})") from None
                raise
            request_id = scope.get("state", {}).get("request_id")
            body = problem_body(
                "서버 내부 오류가 발생했습니다. 잠시 후 다시 시도하고, 계속되면 요청 id와 함께 알려 주세요.",
                ErrorCode.INTERNAL,
                request_id if isinstance(request_id, str) else None,
            )
            await _send_json(send, 500, body)


async def _send_json(send: Send, status: int, body: bytes) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class BodySizeLimitMiddleware:
    """Refuse request bodies over the limit with 413, before they are buffered or spooled to disk.

    Upload routes (``upload_path``) may carry ``max_upload_mb`` of files plus ``upload_envelope``
    bytes of multipart framing; every other body gets ``max_body`` bytes. A declared
    ``Content-Length`` over the limit is refused without reading anything; a chunked body is
    counted as it streams in, and the read that crosses the limit raises :class:`ApiProblem` 413 —
    FastAPI re-raises HTTP exceptions met while reading the body, so the client gets the same
    Korean 413 either way. (The upload endpoint itself enforces the exact file budget.)
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_body: int,
        max_upload_mb: int,
        upload_envelope: int,
        upload_path: re.Pattern[str],
    ) -> None:
        self.app = app
        self.max_body = max_body
        self.max_upload_mb = max_upload_mb
        self.max_upload = max_upload_mb * 1024 * 1024 + upload_envelope
        self.upload_path = upload_path

    def _detail(self, upload: bool) -> str:
        if upload:
            return (
                f"올린 파일이 너무 큽니다(한 번에 모두 합쳐 최대 {self.max_upload_mb}MB). "
                "파일을 나눠 올리거나 관리자에게 JETTAE_MAX_UPLOAD_MB 조정을 요청하세요."
            )
        megabytes = f"{self.max_body / (1024 * 1024):.1f}".rstrip("0").rstrip(".")
        return f"요청 본문이 너무 큽니다(최대 {megabytes}MB)."

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") in ("GET", "HEAD", "OPTIONS"):
            await self.app(scope, receive, send)
            return
        upload = self.upload_path.fullmatch(str(scope.get("path", ""))) is not None
        limit = self.max_upload if upload else self.max_body
        declared = Headers(scope=scope).get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > limit:
            request_id = scope.get("state", {}).get("request_id")
            body = problem_body(
                self._detail(upload),
                ErrorCode.PAYLOAD_TOO_LARGE,
                request_id if isinstance(request_id, str) else None,
            )
            await _send_json(send, 413, body)
            return
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise ApiProblem(413, self._detail(upload), code=ErrorCode.PAYLOAD_TOO_LARGE)
            return message

        await self.app(scope, limited_receive, send)
