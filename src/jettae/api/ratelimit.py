"""Load shedding for the public site (요청 빈도 제한·동시 진단 수 제한).

The public beta runs on one small instance (0.5 CPU, 512 MB) with a 1 GB database, and its write
endpoints are anonymous. Two in-process guards keep a single client from taking the site down, or
from filling the database and faking the demand numbers:

* :class:`RateLimiter` — at most ``limit`` requests per ``window`` seconds per client address (a
  fixed window that starts at the client's first counted request). Used for ``POST /events`` (per
  minute), ``POST /signups`` (per hour) and the attempts of viewers and anonymous callers at
  ``POST /free-scans`` (per minute; the scans themselves are counted per Korean day by
  :class:`jettae.api.security.FreeScanQuota`). Memory is bounded: expired windows are dropped on
  every call and, past ``max_keys`` clients, the oldest windows are evicted (an evicted client
  merely starts a fresh window).
* :class:`ScanSlots` — how many diagnoses for viewers and anonymous callers may run at once
  (``JETTAE_MAX_CONCURRENT_SCANS``). A request that finds every slot taken is answered 503 with
  ``Retry-After`` straight away instead of waiting: queued scans would each hold their upload in
  memory, and a queue only moves the overload from the CPU to the memory of a 512 MB instance.
  uvicorn does not cancel a request's task when its client disconnects, so a slot stays taken
  until the work it guards has really finished.

* :class:`DailyCap` — a site-wide ceiling per Korean calendar day. ``POST /events`` uses it
  (``JETTAE_EVENTS_PER_DAY``): past the cap an event is still answered 202 but not stored, so even
  a client that rotates addresses cannot fill the events table.

**Which address is "the client"** (:class:`ClientIpResolver`, ``JETTAE_CLIENT_IP_SOURCE``). Behind
a proxy the transport peer is the proxy, and what the client claims about itself cannot be trusted:
a proxy that *appends* to ``X-Forwarded-For`` keeps whatever the client wrote at the left, and
uvicorn with ``FORWARDED_ALLOW_IPS="*"`` reports the leftmost entry. So the source is configurable:

* ``peer`` (default) — ``request.client.host`` as uvicorn resolved it;
* ``header:<Header-Name>`` — exactly one address in that header (a CDN's ``CF-Connecting-IP``,
  say); missing, repeated or unparsable → the rightmost ``X-Forwarded-For`` entry (written by the
  nearest proxy, so a client cannot forge it), and only without one → the peer;
* ``xff:<n>`` — the ``n``-th ``X-Forwarded-For`` entry counted from the right (1 = what the nearest
  proxy appended); fewer entries → the peer, an unparsable entry → :data:`INVALID_CLIENT`.

The operator checks which header carries the real address with the admin-only
``GET /api/v1/admin/request-info`` once deployed. Whatever the source, the key is never the raw
header text: it is parsed with :func:`ipaddress.ip_address` (an IPv4-mapped IPv6 address counts as
its IPv4 address), an IPv6 address is keyed by its /64 (one subscriber usually holds a whole /64),
anything unparsable shares the key :data:`INVALID_CLIENT` and a missing peer :data:`UNKNOWN_CLIENT`.
Keys are therefore at most ~45 characters, which keeps the limiters' memory bounded.

Like the daily free-scan budget, all guards live in the process: several workers would each keep
their own counts.
"""

from __future__ import annotations

import ipaddress
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from typing import Annotated, Any, Final, Protocol, cast

import structlog
from fastapi import Depends, Request

from jettae.api.errors import ApiError, ApiProblem, ErrorCode
from jettae.config import ClientIpSource, parse_client_ip_source
from jettae.ingest import KST

__all__ = [
    "BUSY_RESPONSE",
    "BUSY_RETRY_AFTER",
    "DEFAULT_MAX_KEYS",
    "FREE_SCAN_ATTEMPTS_PER_MINUTE",
    "INVALID_CLIENT",
    "IPV6_PREFIX",
    "RATE_LIMITED_RESPONSE",
    "UNKNOWN_CLIENT",
    "X_FORWARDED_FOR",
    "ClientDep",
    "ClientIpResolver",
    "DailyCap",
    "EventCapDep",
    "IpAddress",
    "RateLimiter",
    "ScanSlots",
    "ScanSlotsDep",
    "address_key",
    "client_address",
    "forwarded_entries",
    "get_client_ip_resolver",
    "get_event_cap",
    "get_scan_slots",
    "limit_events",
    "limit_free_scan_attempts",
    "limit_signups",
    "parse_ip",
]

_log = structlog.get_logger("jettae.api.ratelimit")

#: Key of a request whose transport did not report a client address (ASGI ``client`` is None).
UNKNOWN_CLIENT: Final = "unknown"
#: Key shared by every client whose address could not be parsed as an IP address.
INVALID_CLIENT: Final = "invalid"
#: IPv6 clients are counted per network of this prefix length.
IPV6_PREFIX: Final = 64
#: The header proxies append the address they received a request from to.
X_FORWARDED_FOR: Final = "X-Forwarded-For"
#: Longest text :func:`parse_ip` looks at (a bracketed, scoped IPv6 address with a port fits).
_MAX_ADDRESS_CHARS: Final = 64
#: Clients tracked per limiter before the oldest windows are evicted (~100 bytes each).
DEFAULT_MAX_KEYS: Final = 10_000
#: Seconds a caller is asked to wait when every scan slot is taken.
BUSY_RETRY_AFTER: Final = 30
#: Free-scan *attempts* (allowed or not) one client address may make per minute — a scripted loop
#: of failing or refused uploads is cut off before it reaches the parsers again and again.
FREE_SCAN_ATTEMPTS_PER_MINUTE: Final = 10

#: OpenAPI entries for the two refusals (the shared 429/503 texts describe other causes).
RATE_LIMITED_RESPONSE: Final[dict[str, Any]] = {
    "model": ApiError,
    "description": "같은 곳에서 요청이 너무 잦음(Retry-After 헤더 참고)",
}
BUSY_RESPONSE: Final[dict[str, Any]] = {
    "model": ApiError,
    "description": "다른 진단이 진행 중이라 지금은 받을 수 없음(Retry-After 헤더 참고)",
}


def _wait_text(seconds: int) -> str:
    """``42초`` below a minute, else whole minutes rounded up (``59분``)."""
    return f"{seconds}초" if seconds < 60 else f"{math.ceil(seconds / 60)}분"


# --------------------------------------------------------------------------- rate limiter


@dataclass(slots=True)
class _Window:
    start: float
    count: int


class RateLimiter:
    """At most ``limit`` requests per ``window_seconds`` per key (see the module docstring).

    ``subject`` names the guarded action in the Korean refusal ("사전 신청", "이벤트 기록" …).
    ``clock`` returns seconds on a monotonic scale (injected by tests).
    """

    def __init__(
        self,
        limit: int,
        window_seconds: float,
        *,
        subject: str,
        max_keys: int = DEFAULT_MAX_KEYS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if limit < 1 or window_seconds <= 0 or max_keys < 1:
            raise ValueError("요청 빈도 제한의 한도·기간·추적 수는 1 이상이어야 합니다.")
        self.limit = limit
        self.window = float(window_seconds)
        self.max_keys = max_keys
        self.subject = subject
        self._clock = clock
        self._lock = threading.Lock()
        #: ordered by window start: a key is (re)inserted only when its window begins
        self._windows: OrderedDict[str, _Window] = OrderedDict()

    def __len__(self) -> int:
        """Keys currently tracked (expired windows may linger until the next call)."""
        return len(self._windows)

    def _expire(self, now: float) -> None:
        while self._windows:
            key, window = next(iter(self._windows.items()))
            if now - window.start < self.window:
                return
            del self._windows[key]

    def retry_after(self, key: str) -> int | None:
        """Count one request from ``key``: None when it is allowed, otherwise the whole seconds
        until the key's window ends (at least 1). Refused requests are not counted."""
        now = self._clock()
        with self._lock:
            self._expire(now)
            window = self._windows.get(key)
            if window is None:
                while len(self._windows) >= self.max_keys:
                    self._windows.popitem(last=False)
                self._windows[key] = _Window(start=now, count=1)
                return None
            if window.count >= self.limit:
                return max(1, math.ceil(window.start + self.window - now))
            window.count += 1
            return None

    def check(self, key: str) -> None:
        """:meth:`retry_after`, raising 429 ``rate_limited`` with ``Retry-After`` when refused."""
        wait = self.retry_after(key)
        if wait is None:
            return
        raise ApiProblem(
            429,
            f"같은 곳에서 {self.subject} 요청이 너무 잦습니다. {_wait_text(wait)} 뒤에 다시 시도하세요.",
            code=ErrorCode.RATE_LIMITED,
            headers={"Retry-After": str(wait)},
        )


# --------------------------------------------------------------------------- scan slots


class ScanSlots:
    """A non-blocking counting semaphore for the diagnoses of viewers and anonymous callers.

    Not an ``asyncio.Semaphore``: nothing ever waits here, and a plain counter under a lock does
    not bind itself to one event loop (tests run several apps, each on its own loop).
    """

    def __init__(self, capacity: int, *, retry_after: int = BUSY_RETRY_AFTER) -> None:
        if capacity < 1:
            raise ValueError("동시 진단 수(JETTAE_MAX_CONCURRENT_SCANS)는 1 이상이어야 합니다.")
        self.capacity = capacity
        self.retry_after = retry_after
        self._lock = threading.Lock()
        self._in_use = 0

    @property
    def in_use(self) -> int:
        with self._lock:
            return self._in_use

    @contextmanager
    def hold(self) -> Iterator[None]:
        """One slot for the duration of the block; 503 ``server_busy`` at once when none is free."""
        with self._lock:
            if self._in_use >= self.capacity:
                raise ApiProblem(
                    503,
                    "지금은 다른 진단이 진행 중이라 새 진단을 시작할 수 없습니다. "
                    f"{_wait_text(self.retry_after)}쯤 뒤에 다시 시도하세요.",
                    code=ErrorCode.SERVER_BUSY,
                    headers={"Retry-After": str(self.retry_after)},
                )
            self._in_use += 1
        try:
            yield
        finally:
            with self._lock:
                self._in_use -= 1


# --------------------------------------------------------------------------- daily cap


class DailyCap:
    """At most ``limit`` uses per Korean calendar day, site-wide (``limit`` 0 allows none).

    ``subject`` names what is capped in the one warning logged per day once the cap is reached
    (the log says *that* it happened, never who sent what).
    """

    def __init__(self, limit: int, *, subject: str) -> None:
        if limit < 0:
            raise ValueError("하루 상한은 0 이상이어야 합니다.")
        self.limit = limit
        self.subject = subject
        self._lock = threading.Lock()
        self._day: date | None = None
        self._used = 0
        self._warned = False

    def used(self, now: datetime) -> int:
        """Uses counted for ``now``'s day."""
        with self._lock:
            return self._used if self._day == now.astimezone(KST).date() else 0

    def take(self, now: datetime) -> bool:
        """Count one use on ``now``'s day: True while the cap allows it, False once it is spent."""
        day = now.astimezone(KST).date()
        with self._lock:
            if self._day != day:
                self._day, self._used, self._warned = day, 0, False
            if self._used < self.limit:
                self._used += 1
                return True
            warn = not self._warned
            self._warned = True
        if warn:
            _log.warning("daily_cap.reached", subject=self.subject, limit=self.limit, day=day.isoformat())
        return False


# --------------------------------------------------------------------------- client address


IpAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


def parse_ip(text: str) -> IpAddress | None:
    """The IP address in ``text`` (surrounding spaces, a ``:port`` after IPv4 and ``[v6]:port``
    allowed), with an IPv4-mapped IPv6 address unwrapped to IPv4; None when it is not one."""
    candidate = text.strip()
    if not candidate or len(candidate) > _MAX_ADDRESS_CHARS:
        return None
    if candidate.startswith("[") and "]" in candidate:
        candidate = candidate[1 : candidate.index("]")]
    elif candidate.count(":") == 1:
        candidate = candidate.partition(":")[0]
    try:
        address = ipaddress.ip_address(candidate)
    except ValueError:
        return None
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped
    return address


def address_key(address: IpAddress) -> str:
    """The limiter key of an address: the IPv4 address itself, or its IPv6 /:data:`IPV6_PREFIX`."""
    if isinstance(address, ipaddress.IPv4Address):
        return str(address)
    return str(ipaddress.IPv6Network((address, IPV6_PREFIX), strict=False))


def forwarded_entries(values: Sequence[str]) -> list[str]:
    """The entries of (possibly repeated) ``X-Forwarded-For`` headers, left to right, blanks dropped."""
    return [entry.strip() for value in values for entry in value.split(",") if entry.strip()]


class _Headers(Protocol):
    def getlist(self, key: str) -> list[str]: ...


class ClientIpResolver:
    """Turns a request's peer and headers into the per-client key (see the module docstring)."""

    def __init__(self, source: ClientIpSource) -> None:
        self.source = source

    @classmethod
    def from_setting(cls, value: str) -> ClientIpResolver:
        """From ``JETTAE_CLIENT_IP_SOURCE`` (``ValueError`` with a Korean message when malformed)."""
        return cls(parse_client_ip_source(value))

    @staticmethod
    def peer_key(peer: str | None) -> str:
        """The key of the transport's client host (:data:`UNKNOWN_CLIENT` when there is none)."""
        if not peer:
            return UNKNOWN_CLIENT
        address = parse_ip(peer)
        return address_key(address) if address is not None else INVALID_CLIENT

    def resolve(self, peer: str | None, headers: _Headers) -> str:
        """The key of a request from ``peer`` (the transport's client host) with ``headers``."""
        source = self.source
        if source.kind == "header" and source.header is not None:
            values = headers.getlist(source.header)
            address = parse_ip(values[0]) if len(values) == 1 else None
            if address is not None:
                return address_key(address)
            # never the peer first: behind FORWARDED_ALLOW_IPS="*" it is the client-written leftmost entry
            entries = forwarded_entries(headers.getlist(X_FORWARDED_FOR))
            if entries:
                nearest = parse_ip(entries[-1])
                return address_key(nearest) if nearest is not None else INVALID_CLIENT
            return self.peer_key(peer)
        if source.kind == "xff" and source.hops is not None:
            entries = forwarded_entries(headers.getlist(X_FORWARDED_FOR))
            if len(entries) < source.hops:
                return self.peer_key(peer)
            address = parse_ip(entries[-source.hops])
            return address_key(address) if address is not None else INVALID_CLIENT
        return self.peer_key(peer)


#: The resolver of an app that was not given one (``JETTAE_CLIENT_IP_SOURCE=peer``).
_PEER: Final = ClientIpResolver(ClientIpSource(kind="peer"))


# --------------------------------------------------------------------------- dependencies


async def get_client_ip_resolver(request: Request) -> ClientIpResolver:
    resolver = getattr(request.app.state, "client_ip", None)
    return resolver if isinstance(resolver, ClientIpResolver) else _PEER


async def client_address(request: Request) -> str:
    """The key the limits are counted under (the app's :class:`ClientIpResolver`)."""
    resolver = await get_client_ip_resolver(request)
    client = request.client
    return resolver.resolve(client.host if client is not None else None, request.headers)


ClientDep = Annotated[str, Depends(client_address)]


async def get_event_cap(request: Request) -> DailyCap:
    return cast(DailyCap, request.app.state.event_cap)


EventCapDep = Annotated[DailyCap, Depends(get_event_cap)]


async def get_scan_slots(request: Request) -> ScanSlots:
    return cast(ScanSlots, request.app.state.scan_slots)


ScanSlotsDep = Annotated[ScanSlots, Depends(get_scan_slots)]


async def limit_events(request: Request) -> None:
    """Route dependency of ``POST /events`` (``JETTAE_EVENTS_PER_CLIENT_PER_MINUTE``)."""
    cast(RateLimiter, request.app.state.event_limiter).check(await client_address(request))


async def limit_signups(request: Request) -> None:
    """Route dependency of ``POST /signups`` (``JETTAE_SIGNUPS_PER_CLIENT_PER_HOUR``)."""
    cast(RateLimiter, request.app.state.signup_limiter).check(await client_address(request))


def limit_free_scan_attempts(request: Request, client: str) -> None:
    """Count a free-scan attempt of a viewer or anonymous caller (:data:`FREE_SCAN_ATTEMPTS_PER_MINUTE`)."""
    cast(RateLimiter, request.app.state.free_scan_attempts).check(client)
