"""Operator tools (관리): what the server sees of a request, for checking the proxy set-up.

The per-client limits (free scans, events, sign-ups) are only as good as the address they count
under, and which proxy header carries the visitor's real address is a property of the host
(Render, a CDN in front of it), not of the code. ``GET /admin/request-info`` shows an administrator
exactly what arrived — the peer, ``X-Forwarded-For``, ``CF-Connecting-IP``, ``True-Client-IP``,
``X-Real-IP`` — next to the setting (``JETTAE_CLIENT_IP_SOURCE``) and the key it produced, so the
setting can be verified from a browser or curl right after a deploy. Nothing of it is stored or
logged, and like every API response it is sent with ``Cache-Control: no-store``.
"""

from __future__ import annotations

from typing import Final

from fastapi import APIRouter, Request

from jettae.api.errors import error_responses
from jettae.api.ratelimit import X_FORWARDED_FOR, client_address, get_client_ip_resolver
from jettae.api.schemas import RequestInfo
from jettae.api.security import AdminDep

__all__ = ["MAX_SHOWN_HEADER", "TAG", "TAG_DESCRIPTION", "router"]

TAG: Final = "관리"
TAG_DESCRIPTION: Final = "운영 점검(관리자) — 프록시 뒤에서 방문자 주소가 어느 헤더로 오는지 확인"

router = APIRouter(prefix="/admin", tags=[TAG])

#: Characters of one header shown back (a proxy chain is far shorter; a forged header may not be).
MAX_SHOWN_HEADER: Final = 512


def _header(request: Request, name: str) -> str | None:
    values = request.headers.getlist(name)
    return ", ".join(values)[:MAX_SHOWN_HEADER] if values else None


@router.get(
    "/request-info",
    operation_id="getRequestInfo",
    summary="요청 주소 정보(관리자)",
    responses=error_responses(401, 403, 422),
)
async def get_request_info(request: Request, identity: AdminDep) -> RequestInfo:
    """이 요청이 어떤 주소·프록시 헤더로 도착했는지와 그로부터 고른 접속 주소 키(관리자 전용).

    배포한 뒤 curl로 열어 방문자 실제 IP가 어느 헤더(`CF-Connecting-IP`, `X-Forwarded-For`의 오른쪽에서
    몇 번째 …)에 오는지 확인하고 `JETTAE_CLIENT_IP_SOURCE`를 맞추세요. 저장하거나 로그에 남기지 않습니다.
    """
    resolver = await get_client_ip_resolver(request)
    client = request.client
    return RequestInfo(
        peer=client.host[:MAX_SHOWN_HEADER] if client is not None else None,
        x_forwarded_for=_header(request, X_FORWARDED_FOR),
        cf_connecting_ip=_header(request, "CF-Connecting-IP"),
        true_client_ip=_header(request, "True-Client-IP"),
        x_real_ip=_header(request, "X-Real-IP"),
        client_ip_source=str(resolver.source),
        resolved_client=await client_address(request),
    )
