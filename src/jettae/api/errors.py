"""HTTP error contract of the JETTAE Ops API (오류 응답 규약).

Every error body has the same shape — ``{"detail": "<한국어 설명>", "code": "<기계용 코드>",
"request_id": "…"}`` — so the web app, the CLI and scripts can show ``detail`` to a person and
branch on ``code`` without parsing prose. Validation errors add ``errors`` (one entry per field).

Why the mapping lives here and not in the service: :class:`~jettae.orchestrator.JettaeService`
raises plain domain errors (``NotFoundError``, ``PermissionError``, ``VersionConflictError``,
``ValueError``) so the API and the CLI behave identically; turning them into status codes is a
delivery concern, done once, mechanically:

=================================  ======  ====================
exception                          status  code
=================================  ======  ====================
``NotFoundError`` (domain port)    404     ``not_found``
``PermissionError``                403     ``forbidden``
``PermissionError`` from the OS    500     ``internal_error``
``VersionConflictError``           409     ``version_conflict``
``AlreadyExistsError``             409     ``already_exists``
``PatchStateError``                409     ``patch_state``
pydantic / request validation      422     ``validation_error``
``ValueError`` (anything else)     422     ``invalid_input``
``LLMError``                       502     ``llm_failed``
``StoreIntegrityError``            500     ``integrity_error``
=================================  ======  ====================

The ``detail`` is always Korean: domain errors already carry Korean messages; anything else (an
English library message, a framework default such as "Not Found") is replaced by a Korean
sentence, because the person reading it is an operator, not a Python developer.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from enum import StrEnum
from typing import Any, Final, cast

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from jettae.domain.ports import NotFoundError, VersionConflictError
from jettae.improve import PatchStateError
from jettae.llm.base import LLMError
from jettae.store import AlreadyExistsError, StoreIntegrityError, database_error_summary

__all__ = [
    "ApiError",
    "ApiProblem",
    "ErrorCode",
    "FieldError",
    "ValidationApiError",
    "error_responses",
    "failure_log_fields",
    "install_error_handlers",
    "korean_or",
    "problem_body",
    "translate_validation_errors",
]

_log = structlog.get_logger("jettae.api.errors")

_HANGUL: Final = re.compile(r"[가-힣]")
#: ``loc`` prefixes FastAPI adds to say *where* a value came from; the field name is what follows.
_LOCATIONS: Final = frozenset({"body", "query", "path", "header", "cookie"})
#: ``field`` of a problem that concerns the request body as a whole
_WHOLE_BODY: Final = "요청 본문"


class ErrorCode(StrEnum):
    """Machine-readable error codes (stable: clients branch on them)."""

    BAD_REQUEST = "bad_request"
    UNAUTHORIZED = "unauthorized"
    INVALID_IDENTITY = "invalid_identity"
    FORBIDDEN = "forbidden"
    NOT_FOUND = "not_found"
    REPORT_NOT_FOUND = "report_not_found"
    METHOD_NOT_ALLOWED = "method_not_allowed"
    VERSION_CONFLICT = "version_conflict"
    ALREADY_EXISTS = "already_exists"
    PATCH_STATE = "patch_state"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    VALIDATION = "validation_error"
    INVALID_INPUT = "invalid_input"
    CONSENT_REQUIRED = "consent_required"
    FREE_SCAN_LIMIT = "free_scan_limit"
    RATE_LIMITED = "rate_limited"
    SERVER_BUSY = "server_busy"
    INTERNAL = "internal_error"
    INTEGRITY = "integrity_error"
    LLM_FAILED = "llm_failed"
    SERVICE_UNAVAILABLE = "service_unavailable"
    DEMO_UNAVAILABLE = "demo_unavailable"


# --------------------------------------------------------------------------- body models


class ApiError(BaseModel):
    """Body of every error response."""

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "detail": "워크스페이스 ws-0123456789abcdef을(를) 찾을 수 없습니다.",
                    "code": "not_found",
                    "request_id": "6f1c2b0e9a8d4c7b8e5f4a3b2c1d0e9f",
                }
            ]
        },
    )

    detail: str
    code: str
    #: the request's id (also in the ``X-Request-ID`` header) — quote it when reporting a problem
    request_id: str | None = None


class FieldError(BaseModel):
    """One invalid input value."""

    model_config = ConfigDict(frozen=True)

    #: dotted path of the value inside the request (``email``, ``props.team``) or ``요청 본문``
    field: str
    message: str
    #: pydantic error type (``missing``, ``string_too_long`` …), for programmatic handling
    type: str


class ValidationApiError(ApiError):
    """Body of a 422 response: the first problem in ``detail``, all of them in ``errors``."""

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "detail": "reason: 1자 이상이어야 합니다.",
                    "code": "validation_error",
                    "request_id": "6f1c2b0e9a8d4c7b8e5f4a3b2c1d0e9f",
                    "errors": [
                        {"field": "reason", "message": "1자 이상이어야 합니다.", "type": "string_too_short"}
                    ],
                }
            ]
        },
    )

    errors: tuple[FieldError, ...] = ()


class ApiProblem(StarletteHTTPException):
    """An HTTP error raised by the API layer itself (401/403/413/429 …) with a stable ``code``.

    It subclasses Starlette's ``HTTPException`` on purpose: FastAPI re-raises that type unchanged
    when it surfaces while the request body is being read, which is how the body-size limit can
    answer 413 from inside ``receive`` instead of FastAPI's generic 400.
    """

    def __init__(
        self,
        status_code: int,
        detail: str,
        *,
        code: ErrorCode,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=detail, headers=dict(headers) if headers else None)
        self.code = code


# --------------------------------------------------------------------------- messages


def korean_or(message: object, fallback: str) -> str:
    """``message`` when it is Korean text, else ``fallback`` (never show English internals)."""
    text = message if isinstance(message, str) else ""
    return text if _HANGUL.search(text) else fallback


def _exception_message(exc: BaseException, fallback: str) -> str:
    """The first argument of a domain error (``KeyError`` would quote ``str(exc)``)."""
    first = exc.args[0] if exc.args else None
    return korean_or(first, fallback)


def _expected(ctx: Mapping[str, Any]) -> str:
    """pydantic's ``"'a', 'b' or 'c'"`` as ``'a', 'b', 'c'``."""
    return str(ctx.get("expected", "")).replace(" or ", ", ")


def _reason(error_type: str, ctx: Mapping[str, Any]) -> str:
    """Korean reason for one pydantic error type (the field name is prefixed by the caller)."""
    simple: dict[str, str] = {
        "missing": "필수 값이 빠졌습니다.",
        "string_type": "문자열이어야 합니다.",
        "int_type": "정수여야 합니다.",
        "int_parsing": "정수여야 합니다.",
        "int_from_float": "정수여야 합니다.",
        "float_type": "숫자여야 합니다.",
        "float_parsing": "숫자여야 합니다.",
        "finite_number": "유한한 숫자여야 합니다.",
        "bool_type": "true 또는 false여야 합니다.",
        "bool_parsing": "true 또는 false여야 합니다.",
        "list_type": "목록이어야 합니다.",
        "tuple_type": "목록이어야 합니다.",
        "dict_type": "객체여야 합니다.",
        "model_type": "객체여야 합니다.",
        "model_attributes_type": "객체여야 합니다.",
        "string_pattern_mismatch": "형식이 올바르지 않습니다.",
        "json_invalid": "올바른 JSON이 아닙니다.",
        "json_type": "JSON 값이어야 합니다.",
        "extra_forbidden": "알 수 없는 항목입니다.",
        "string_unicode": "올바른 유니코드 문자열이어야 합니다.",
    }
    if error_type in simple:
        return simple[error_type]
    bounded: dict[str, str] = {
        "string_too_short": "{min_length}자 이상이어야 합니다.",
        "string_too_long": "{max_length}자 이하여야 합니다.",
        "too_short": "항목이 {min_length}개 이상이어야 합니다.",
        "too_long": "항목은 {max_length}개 이하여야 합니다.",
        "greater_than_equal": "{ge} 이상이어야 합니다.",
        "greater_than": "{gt}보다 커야 합니다.",
        "less_than_equal": "{le} 이하여야 합니다.",
        "less_than": "{lt}보다 작아야 합니다.",
    }
    if error_type in bounded:
        try:
            return bounded[error_type].format(**ctx)
        except (KeyError, IndexError):
            return "값의 범위가 올바르지 않습니다."
    if error_type in ("literal_error", "enum"):
        return f"허용되지 않은 값입니다(허용: {_expected(ctx)})."
    if error_type in ("value_error", "assertion_error"):
        return korean_or(str(ctx.get("error", "")), "값이 올바르지 않습니다.")
    return "값이 올바르지 않습니다."


def _field_name(loc: Sequence[object]) -> str:
    parts = list(loc)
    if parts and parts[0] in _LOCATIONS:
        parts = parts[1:]
    return ".".join(str(part) for part in parts) or _WHOLE_BODY


def translate_validation_errors(errors: Sequence[Mapping[str, Any]]) -> tuple[FieldError, ...]:
    """pydantic/FastAPI error dicts → Korean :class:`FieldError` entries (order kept)."""
    out: list[FieldError] = []
    for error in errors:
        error_type = str(error.get("type", "value_error"))
        ctx = error.get("ctx") or {}
        # a JSON syntax error is located by its character offset (``("body", 17)``), which would
        # read as a field named "17": it concerns the body as a whole
        field = _WHOLE_BODY if error_type == "json_invalid" else _field_name(error.get("loc", ()))
        out.append(FieldError(field=field, message=_reason(error_type, ctx), type=error_type))
    return tuple(out)


# --------------------------------------------------------------------------- rendering


def _request_id(request: Request) -> str | None:
    value = getattr(request.state, "request_id", None)
    return value if isinstance(value, str) else None


def problem_body(detail: str, code: ErrorCode, request_id: str | None) -> bytes:
    """Serialized :class:`ApiError` for responses sent below the routing layer (middleware)."""
    payload = ApiError(detail=detail, code=code.value, request_id=request_id).model_dump()
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _respond(
    request: Request,
    status_code: int,
    detail: str,
    code: ErrorCode,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    body = ApiError(detail=detail, code=code.value, request_id=_request_id(request))
    return JSONResponse(
        body.model_dump(), status_code=status_code, headers=dict(headers) if headers else None
    )


def _validation_response(request: Request, errors: Sequence[Mapping[str, Any]]) -> JSONResponse:
    fields = translate_validation_errors(errors)
    first = fields[0] if fields else None
    detail = f"{first.field}: {first.message}" if first else "입력값이 올바르지 않습니다."
    body = ValidationApiError(
        detail=detail, code=ErrorCode.VALIDATION.value, request_id=_request_id(request), errors=fields
    )
    return JSONResponse(body.model_dump(), status_code=422)


#: Korean text for framework-raised HTTP errors whose own detail is English ("Not Found").
_STATUS_TEXT: Final[Mapping[int, tuple[str, ErrorCode]]] = {
    400: ("요청 형식이 올바르지 않습니다.", ErrorCode.BAD_REQUEST),
    401: ("인증이 필요합니다.", ErrorCode.UNAUTHORIZED),
    403: ("이 작업을 할 권한이 없습니다.", ErrorCode.FORBIDDEN),
    404: ("요청한 경로를 찾을 수 없습니다.", ErrorCode.NOT_FOUND),
    405: ("이 경로에서 허용되지 않은 메서드입니다.", ErrorCode.METHOD_NOT_ALLOWED),
    413: ("요청이 너무 큽니다.", ErrorCode.PAYLOAD_TOO_LARGE),
}

Handler = Callable[[Request, Exception], Awaitable[JSONResponse]]


def _http_handler() -> Handler:
    async def handle(request: Request, exc: Exception) -> JSONResponse:
        http_exc = cast(StarletteHTTPException, exc)  # registered for this type only
        default_text, default_code = _STATUS_TEXT.get(
            http_exc.status_code, ("요청을 처리할 수 없습니다.", ErrorCode.BAD_REQUEST)
        )
        code = http_exc.code if isinstance(http_exc, ApiProblem) else default_code
        detail = korean_or(http_exc.detail, default_text)
        return _respond(request, http_exc.status_code, detail, code, http_exc.headers)

    return handle


def failure_log_fields(exc: BaseException) -> dict[str, object]:
    """What a log line may say about a server-side failure: the class name and the traceback.

    A database error (anywhere in the exception's chain) gets class names only: its messages quote
    the statement's values — a sign-up's e-mail, organisation and free-text message — and a
    traceback prints every message of the chain (see :func:`jettae.store.database_error_summary`).
    """
    fields: dict[str, object] = {"error": type(exc).__name__}
    database = database_error_summary(exc)
    if database is None:
        fields["exc_info"] = exc
    else:
        fields.update(database)
    return fields


def _domain_handler(status_code: int, code: ErrorCode, fallback: str) -> Handler:
    async def handle(request: Request, exc: Exception) -> JSONResponse:
        if status_code >= 500:
            _log.error("api.domain_error", status=status_code, **failure_log_fields(exc))
        return _respond(request, status_code, _exception_message(exc, fallback), code)

    return handle


def _permission_handler() -> Handler:
    """``PermissionError`` is the service's "role too low" (403) — and also the OS error for a file
    the *server process* may not read or write (``errno`` set). The latter is a server fault: a 403
    would send the caller after a higher role instead of an administrator after the file."""
    forbidden = _domain_handler(403, ErrorCode.FORBIDDEN, "이 작업을 할 권한이 없습니다.")
    server_side = _domain_handler(
        500,
        ErrorCode.INTERNAL,
        "서버가 필요한 파일에 접근할 권한이 없습니다. 요청 id와 함께 서버 관리자에게 알려 주세요.",
    )

    async def handle(request: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, OSError) and exc.errno is not None:
            return await server_side(request, exc)
        return await forbidden(request, exc)

    return handle


def _validation_handler() -> Handler:
    """For FastAPI's request validation and for a pydantic model built inside a handler."""

    async def handle(request: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, RequestValidationError):
            return _validation_response(request, list(exc.errors()))
        return _validation_response(request, cast(ValidationError, exc).errors(include_url=False))

    return handle


def install_error_handlers(app: FastAPI) -> None:
    """Register the mapping in the module docstring on ``app``.

    Starlette looks handlers up along the exception's MRO, so the specific classes win over the
    ``ValueError`` catch-all (pydantic's ``ValidationError`` and ``PatchStateError`` are both
    ``ValueError`` subclasses).
    """
    app.add_exception_handler(StarletteHTTPException, _http_handler())
    app.add_exception_handler(RequestValidationError, _validation_handler())
    app.add_exception_handler(ValidationError, _validation_handler())
    app.add_exception_handler(
        NotFoundError, _domain_handler(404, ErrorCode.NOT_FOUND, "요청한 대상을 찾을 수 없습니다.")
    )
    app.add_exception_handler(PermissionError, _permission_handler())
    app.add_exception_handler(
        VersionConflictError,
        _domain_handler(
            409,
            ErrorCode.VERSION_CONFLICT,
            "그래프 버전이 바뀌었습니다. 최신 상태를 다시 불러온 뒤 시도하세요.",
        ),
    )
    app.add_exception_handler(
        AlreadyExistsError, _domain_handler(409, ErrorCode.ALREADY_EXISTS, "이미 있는 항목입니다.")
    )
    app.add_exception_handler(
        PatchStateError,
        _domain_handler(409, ErrorCode.PATCH_STATE, "지금 상태의 수정안에는 이 작업을 할 수 없습니다."),
    )
    app.add_exception_handler(
        LLMError,
        _domain_handler(
            502,
            ErrorCode.LLM_FAILED,
            "생성형 AI 호출에 실패했습니다. LLM 모드를 끄고(off) 다시 시도하거나 잠시 뒤 다시 시도하세요.",
        ),
    )
    app.add_exception_handler(
        StoreIntegrityError,
        _domain_handler(500, ErrorCode.INTEGRITY, "저장된 데이터가 기록과 일치하지 않습니다(손상 의심)."),
    )
    app.add_exception_handler(
        ValueError, _domain_handler(422, ErrorCode.INVALID_INPUT, "입력값이 올바르지 않습니다.")
    )


# --------------------------------------------------------------------------- OpenAPI


_RESPONSE_TEXT: Final[Mapping[int, str]] = {
    400: "요청 형식 오류",
    401: "인증 실패(토큰 모드에서 토큰이 없거나, 토큰·공개 모드에서 토큰이 올바르지 않음)",
    403: "권한 없음(역할 부족)",
    404: "대상을 찾을 수 없음",
    409: "충돌(그래프 버전이 바뀌었거나 이미 결정된 수정안)",
    413: "요청이 너무 큼(JETTAE_MAX_UPLOAD_MB)",
    422: "입력값 오류(한국어 설명 + 항목별 오류)",
    429: "오늘의 무료 진단 한도 초과(Retry-After 헤더 참고)",
    502: "생성형 AI 호출 실패",
    503: "서비스 준비 안 됨",
}


def error_responses(*statuses: int) -> dict[int | str, dict[str, Any]]:
    """OpenAPI ``responses`` entries for the given error statuses (422 uses the field list body).

    Declaring 422 explicitly also stops FastAPI from documenting its default
    ``HTTPValidationError`` body, which this API never sends.
    """
    out: dict[int | str, dict[str, Any]] = {}
    for status in statuses:
        model: type[ApiError] = ValidationApiError if status == 422 else ApiError
        out[status] = {"model": model, "description": _RESPONSE_TEXT.get(status, "오류")}
    return out
