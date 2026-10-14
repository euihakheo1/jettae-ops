"""Who is calling and what they may do (인증·역할·무료 진단 한도).

Three authentication modes (``JETTAE_AUTH_MODE``):

* ``dev`` — the identity comes from ``X-Jettae-User`` (default ``demo``) and ``X-Jettae-Role``
  (default ``viewer``). Anyone can claim any role, so this mode is for local use and demos only;
  a request without ``X-Jettae-User`` counts as *anonymous*.
* ``token`` — ``Authorization: Bearer <token>`` must match an entry of ``JETTAE_API_TOKENS``
  (``{"<token>": {"user": "...", "role": "..."}}``); anything else is 401. The ``X-Jettae-*``
  headers are ignored, otherwise a token holder could raise their own role.
* ``public`` — the public beta website. A request *without* ``Authorization`` is an anonymous
  ``viewer`` (user ``anonymous``); a valid bearer token (same table as ``token``) gives that
  token's identity; an ``Authorization`` header that is present but unknown or malformed is 401,
  so a mistyped admin token never silently degrades to a visitor. The ``X-Jettae-*`` headers are
  ignored as in ``token`` mode. Anonymous visitors see only the demo workspace and the workspaces
  their own free scans created, opened with the access key the scan returned
  (:mod:`jettae.api.access`); every other workspace answers 404 to them. Like ``token``, the mode
  refuses to start without a token: an administrator needs one to read the demand metrics.

Roles are ordered ``viewer < editor < lead < admin``. Reading is open to every identity, writes
(workspaces, uploads, drills, drafts) need ``editor``, and the decisions that change the graph
(approve / reject / revert) need ``lead`` or ``admin`` — the same set as
:data:`jettae.improve.DECIDER_ROLES`, which the service checks again (defence in depth). Personal
data (the sign-up list) is for ``admin`` only.

Diagnosing is the one write a viewer may do: it is the "free scan" of the product. Because every
run costs CPU (and LLM calls in LIVE mode), runs requested by viewers and anonymous callers share
one daily budget (``JETTAE_DAILY_FREE_SCAN_LIMIT``, counted per Korean calendar day), of which one
client address may use at most ``JETTAE_FREE_SCANS_PER_CLIENT_PER_DAY``; at most
``JETTAE_MAX_CONCURRENT_SCANS`` of their runs go at once (:class:`jettae.api.ratelimit.ScanSlots`),
and they always run with the server's LLM mode (:func:`llm_mode_for`). Editors and above are not
metered. The budget is process-local: several API workers each keep their own count.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import math
import threading
import unicodedata
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import StrEnum
from typing import Annotated, Final, Literal, cast

from fastapi import Depends, Request
from starlette.exceptions import HTTPException as StarletteHTTPException

from jettae.api.errors import ApiProblem, ErrorCode
from jettae.api.ratelimit import ScanSlots
from jettae.domain import LlmMode
from jettae.domain.ports import NotFoundError
from jettae.ingest import KST

__all__ = [
    "ANONYMOUS_USER",
    "DEFAULT_USER",
    "ROLE_HEADER",
    "ROLE_LABEL_KO",
    "USER_HEADER",
    "WORKSPACE_KEY_HEADER",
    "AdminDep",
    "AuthMode",
    "Authenticator",
    "DeciderDep",
    "EditorDep",
    "FreeScanQuota",
    "Identity",
    "IdentityDep",
    "QuotaDep",
    "Role",
    "caused_by_request",
    "current_identity",
    "free_scan",
    "get_free_scans",
    "llm_mode_for",
    "parse_role",
    "parse_user",
    "require_role",
]

AuthMode = Literal["dev", "token", "public"]

USER_HEADER: Final = "X-Jettae-User"
ROLE_HEADER: Final = "X-Jettae-Role"
AUTHORIZATION_HEADER: Final = "Authorization"
#: the key that opens a private workspace for anonymous callers in ``public`` mode
WORKSPACE_KEY_HEADER: Final = "X-Jettae-Workspace-Key"
#: identity of a dev-mode request without ``X-Jettae-User``
DEFAULT_USER: Final = "demo"
#: identity of a public-mode request without ``Authorization``
ANONYMOUS_USER: Final = "anonymous"
#: user ids end up in the audit log and in reports, so they are short one-line labels
MAX_USER_LENGTH: Final = 64
#: shorter tokens are guessable; refusing them at start-up beats discovering it in an incident
MIN_TOKEN_LENGTH: Final = 16


class Role(StrEnum):
    VIEWER = "viewer"
    EDITOR = "editor"
    LEAD = "lead"
    ADMIN = "admin"


_RANK: Final[Mapping[Role, int]] = {Role.VIEWER: 0, Role.EDITOR: 1, Role.LEAD: 2, Role.ADMIN: 3}

ROLE_LABEL_KO: Final[Mapping[Role, str]] = {
    Role.VIEWER: "보기 전용(viewer)",
    Role.EDITOR: "편집자(editor)",
    Role.LEAD: "팀 리드(lead)",
    Role.ADMIN: "관리자(admin)",
}


@dataclass(frozen=True, slots=True)
class Identity:
    """The caller: a user id for the audit log, a role, and whether they named themselves."""

    user: str
    role: Role
    anonymous: bool = False

    def allows(self, minimum: Role) -> bool:
        return _RANK[self.role] >= _RANK[minimum]


#: a public-mode visitor (no ``Authorization`` header)
ANONYMOUS: Final = Identity(user=ANONYMOUS_USER, role=Role.VIEWER, anonymous=True)


# --------------------------------------------------------------------------- parsing


def parse_role(value: str) -> Role:
    """``viewer|editor|lead|admin`` (case-insensitive) or ``ValueError`` with a Korean message."""
    try:
        return Role(value.strip().casefold())
    except ValueError:
        allowed = "|".join(role.value for role in Role)
        raise ValueError(f"알 수 없는 역할입니다: '{value.strip()[:20]}'. 허용 값: {allowed}") from None


def _recover_utf8(value: str) -> str:
    """HTTP header bytes arrive decoded as Latin-1; a client that sent UTF-8 (a Korean user id from
    curl) gets its text back, while genuine Latin-1 text is left as it was."""
    try:
        return value.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


def parse_user(value: str) -> str:
    """A user id fit for the audit log: NFC, trimmed, 1–64 visible characters, one line."""
    user = unicodedata.normalize("NFC", _recover_utf8(value)).strip()
    if not user:
        raise ValueError("사용자 id가 비어 있습니다.")
    if len(user) > MAX_USER_LENGTH:
        raise ValueError(f"사용자 id는 {MAX_USER_LENGTH}자 이하여야 합니다.")
    # Zl/Zp (U+2028/U+2029) end a line in many viewers: the id would split an audit-log line
    if any(unicodedata.category(char) in ("Cc", "Cf", "Cs", "Zl", "Zp") for char in user):
        raise ValueError("사용자 id에는 제어 문자, 줄 구분 문자, 보이지 않는 문자를 쓸 수 없습니다.")
    return user


def _token_digest(token: str) -> bytes:
    """Tokens are compared by digest so the comparison time does not depend on their lengths."""
    return hashlib.sha256(token.encode("utf-8")).digest()


# --------------------------------------------------------------------------- authenticator


class Authenticator:
    """Turns request headers into an :class:`Identity` (see the module docstring for the modes).

    The token table is validated when the app is created: a typo in ``JETTAE_API_TOKENS`` must stop
    the server from starting, not lock everyone out (or let a misspelt role through) at run time.
    """

    def __init__(self, mode: AuthMode, tokens: Mapping[str, Mapping[str, str]]) -> None:
        self.mode: AuthMode = mode
        if mode == "token" and not tokens:
            # a forgotten JETTAE_API_TOKENS would otherwise start a server that answers 401 to all
            raise ValueError(
                "토큰 인증 모드(JETTAE_AUTH_MODE=token)에는 JETTAE_API_TOKENS에 토큰이 하나 이상 "
                '있어야 합니다(16자 이상, 32자 이상 권장). 예: {"<토큰>": {"user": "kim", "role": "lead"}}'
            )
        if mode == "public" and not tokens:
            # a public site without any token would collect sign-ups nobody can ever read
            raise ValueError(
                "공개 모드(JETTAE_AUTH_MODE=public)에도 JETTAE_API_TOKENS에 관리자 토큰이 하나 이상 "
                "있어야 합니다(수요 지표·사전 신청 명단 조회용, 토큰은 16자 이상·32자 이상 권장). "
                '예: {"<토큰>": {"user": "heo", "role": "admin"}}'
            )
        entries: list[tuple[bytes, Identity]] = []
        for index, (token, entry) in enumerate(tokens.items(), start=1):
            where = f"JETTAE_API_TOKENS의 {index}번째 토큰"
            if len(token) < MIN_TOKEN_LENGTH or token != token.strip() or any(c.isspace() for c in token):
                raise ValueError(f"{where}은(는) 공백 없이 {MIN_TOKEN_LENGTH}자 이상이어야 합니다.")
            try:
                user = parse_user(str(entry.get("user", "")))
                role = parse_role(str(entry.get("role", Role.VIEWER.value)))
            except ValueError as exc:
                raise ValueError(f"{where}의 설정이 올바르지 않습니다: {exc}") from None
            entries.append((_token_digest(token), Identity(user=user, role=role)))
        self._tokens: tuple[tuple[bytes, Identity], ...] = tuple(entries)

    def identify(self, headers: Mapping[str, str]) -> Identity:
        """The caller's identity; :class:`ApiProblem` 401 (token/public) or 422 (malformed dev headers)."""
        if self.mode == "token":
            return self._from_token(headers.get(AUTHORIZATION_HEADER))
        if self.mode == "public":
            authorization = headers.get(AUTHORIZATION_HEADER)
            # only a *missing* header means "visitor": a wrong token must not pass as anonymous
            return ANONYMOUS if authorization is None else self._from_token(authorization)
        return self._from_dev_headers(headers.get(USER_HEADER), headers.get(ROLE_HEADER))

    def _from_token(self, authorization: str | None) -> Identity:
        scheme, _, credentials = (authorization or "").strip().partition(" ")
        presented = credentials.strip()
        if scheme.casefold() == "bearer" and presented:
            digest = _token_digest(presented)
            match: Identity | None = None
            for known, identity in self._tokens:  # no early exit: same work for every token
                if hmac.compare_digest(known, digest) and match is None:
                    match = identity
            if match is not None:
                return match
        raise ApiProblem(
            401,
            "인증 토큰이 없거나 올바르지 않습니다. 'Authorization: Bearer <토큰>' 헤더를 보내 주세요.",
            code=ErrorCode.UNAUTHORIZED,
            headers={"WWW-Authenticate": 'Bearer realm="jettae"'},
        )

    @staticmethod
    def _from_dev_headers(user: str | None, role: str | None) -> Identity:
        try:
            parsed_role = parse_role(role) if role is not None and role.strip() else Role.VIEWER
            if user is None or not user.strip():
                return Identity(user=DEFAULT_USER, role=parsed_role, anonymous=True)
            return Identity(user=parse_user(user), role=parsed_role)
        except ValueError as exc:
            raise ApiProblem(
                422,
                f"{USER_HEADER}/{ROLE_HEADER} 헤더가 올바르지 않습니다: {exc}",
                code=ErrorCode.INVALID_IDENTITY,
            ) from None


# --------------------------------------------------------------------------- dependencies


async def current_identity(request: Request) -> Identity:
    """FastAPI dependency: the caller's :class:`Identity` (401/422 when it cannot be established)."""
    authenticator = cast(Authenticator, request.app.state.authenticator)
    return authenticator.identify(request.headers)


IdentityDep = Annotated[Identity, Depends(current_identity)]


def _forbidden_message(minimum: Role, actual: Role) -> str:
    current = ROLE_LABEL_KO[actual]
    if minimum is Role.EDITOR:
        return f"이 작업은 편집자(editor) 이상만 할 수 있습니다. 지금 역할은 {current}입니다."
    if minimum is Role.ADMIN:
        return (
            "사전 신청 명단 같은 개인정보 조회는 관리자(admin)만 할 수 있습니다. "
            f"지금 역할은 {current}입니다."
        )
    return (
        "승인·거절·되돌리기와 수요 지표 조회는 팀 리드(lead)·관리자(admin)만 할 수 있습니다. "
        f"지금 역할은 {current}입니다."
    )


def require_role(minimum: Role) -> Callable[[Identity], Awaitable[Identity]]:
    """Dependency factory: the identity, or 403 with a Korean explanation of the missing role."""

    async def dependency(identity: IdentityDep) -> Identity:
        if not identity.allows(minimum):
            raise ApiProblem(403, _forbidden_message(minimum, identity.role), code=ErrorCode.FORBIDDEN)
        return identity

    return dependency


EditorDep = Annotated[Identity, Depends(require_role(Role.EDITOR))]
DeciderDep = Annotated[Identity, Depends(require_role(Role.LEAD))]
AdminDep = Annotated[Identity, Depends(require_role(Role.ADMIN))]


# --------------------------------------------------------------------------- LLM mode


def llm_mode_for(identity: Identity, requested: LlmMode | None, configured: LlmMode) -> LlmMode:
    """The LLM mode a run of ``identity`` may use: what it asked for (else ``configured``) for
    editors and above; always ``configured`` (``JETTAE_LLM_MODE``) for viewers and anonymous
    callers, whatever they sent. The server's mode is a ceiling the public cannot lift: LIVE costs
    money once a provider key is set, and the demo's report is shared by every visitor."""
    if identity.allows(Role.EDITOR) and requested is not None:
        return LlmMode(requested)
    return configured


# --------------------------------------------------------------------------- free-scan budget


def _kst_day(now: datetime) -> date:
    return now.astimezone(KST).date()


def _seconds_until_next_day(now: datetime) -> int:
    tomorrow = datetime.combine(_kst_day(now) + timedelta(days=1), time(0), tzinfo=KST)
    return max(1, math.ceil((tomorrow - now).total_seconds()))


def caused_by_request(exc: BaseException) -> bool:
    """Whether a failed scan is charged anyway: the request was at fault, or it was abandoned.

    * a 4xx refusal (413 too large, 422 unreadable file, 404 unknown workspace) and ``ValueError``
      — the parsers had to read the input first (a YAML, HTML, DOCX or PDF bomb ends here too);
    * ``asyncio.CancelledError`` — the client went away; the worker threads keep running anyway.

    Everything else is a server fault (5xx, a failed LLM call, a store error, a version conflict
    with a concurrent approval) and the scan is given back.
    """
    if isinstance(exc, StarletteHTTPException):
        return 400 <= exc.status_code < 500
    return isinstance(exc, ValueError | NotFoundError | asyncio.CancelledError)


class FreeScanQuota:
    """Daily budget of free scans (Korean calendar day), shared by viewers and anonymous callers,
    with a share per client address (``per_client``; None = no per-client share).

    The budget counts work, not only successes: a scan that fails because of its input, or whose
    client gave up, stays counted (:func:`caused_by_request`) — otherwise every parse bomb would be
    free. Only a server fault gives the scan back. Memory is bounded by the budget itself: a client
    is tracked only while it holds at least one scan of the current day.
    """

    def __init__(self, limit: int, *, per_client: int | None = None) -> None:
        self.limit = limit
        self.per_client = per_client
        self._lock = threading.Lock()
        self._day: date | None = None
        self._used = 0
        self._by_client: dict[str, int] = {}

    def used(self, now: datetime) -> int:
        """Scans counted for ``now``'s day."""
        with self._lock:
            return self._used if self._day == _kst_day(now) else 0

    def used_by(self, client: str, now: datetime) -> int:
        """Scans ``client`` used on ``now``'s day."""
        with self._lock:
            return self._by_client.get(client, 0) if self._day == _kst_day(now) else 0

    def _refused(self, now: datetime) -> ApiProblem:
        if self.limit <= 0:
            detail = (
                "무료 진단이 꺼져 있습니다(JETTAE_DAILY_FREE_SCAN_LIMIT=0). "
                "편집자(editor) 이상 권한으로 요청하세요."
            )
        else:
            detail = (
                f"오늘(한국 시간) 무료 진단 {self.limit}회를 모두 썼습니다. 내일 다시 시도하거나, "
                "편집자(editor) 이상 권한으로 요청하거나, 사전 신청을 남겨 주세요."
            )
        return self._problem(detail, now)

    def _refused_client(self, now: datetime) -> ApiProblem:
        return self._problem(
            f"이 접속 주소에서 오늘(한국 시간) 쓸 수 있는 무료 진단 {self.per_client}회를 모두 썼습니다. "
            "내일 다시 시도하거나 사전 신청을 남겨 주세요.",
            now,
        )

    @staticmethod
    def _problem(detail: str, now: datetime) -> ApiProblem:
        return ApiProblem(
            429,
            detail,
            code=ErrorCode.FREE_SCAN_LIMIT,
            headers={"Retry-After": str(_seconds_until_next_day(now))},
        )

    @contextmanager
    def reserve(self, now: datetime, client: str | None = None) -> Iterator[None]:
        """Take one scan from ``now``'s budget (and from ``client``'s share) for the duration of the
        block; 429 when either is spent. A server fault inside the block gives the scan back."""
        day = _kst_day(now)
        with self._lock:
            if self._day != day:
                self._day, self._used, self._by_client = day, 0, {}
            if self._used >= self.limit:
                raise self._refused(now)
            if client is not None and self.per_client is not None:
                if self._by_client.get(client, 0) >= self.per_client:
                    raise self._refused_client(now)
                self._by_client[client] = self._by_client.get(client, 0) + 1
            self._used += 1
        try:
            yield
        except BaseException as exc:
            if not caused_by_request(exc):
                self._give_back(day, client)
            raise

    def _give_back(self, day: date, client: str | None) -> None:
        with self._lock:
            if self._day != day:
                return
            self._used = max(0, self._used - 1)
            if client is not None and client in self._by_client:
                left = self._by_client[client] - 1
                if left > 0:
                    self._by_client[client] = left
                else:
                    del self._by_client[client]


async def get_free_scans(request: Request) -> FreeScanQuota:
    return cast(FreeScanQuota, request.app.state.free_scans)


QuotaDep = Annotated[FreeScanQuota, Depends(get_free_scans)]


@contextmanager
def free_scan(
    identity: Identity,
    quota: FreeScanQuota,
    now: datetime,
    *,
    client: str | None = None,
    slots: ScanSlots | None = None,
) -> Iterator[None]:
    """Meter a diagnosis for viewers and anonymous callers; editors and above run unmetered.

    A metered run first takes a scan slot (503 at once when every slot is busy; nothing charged),
    then one scan from the daily budget and from ``client``'s share of it (429 when spent).
    """
    if identity.allows(Role.EDITOR):
        yield
        return
    with ExitStack() as stack:
        if slots is not None:
            stack.enter_context(slots.hold())
        stack.enter_context(quota.reserve(now, client))
        yield
