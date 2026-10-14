"""The caller's session as the server sees it (현재 사용자·역할).

The web app's role switcher only means something in dev mode, where the role comes from request
headers. In token and public mode the token (or its absence) decides, so the app asks here and
shows the real role instead of offering buttons the server would refuse.
"""

from __future__ import annotations

from typing import Final, cast

from fastapi import APIRouter, Request

from jettae.api.schemas import SessionInfo
from jettae.api.security import Authenticator, IdentityDep

__all__ = ["TAG", "TAG_DESCRIPTION", "router"]

TAG: Final = "세션"
TAG_DESCRIPTION: Final = "현재 사용자·역할·인증 방식(웹 화면이 실제 권한대로 버튼을 보여 주도록)"

router = APIRouter(tags=[TAG])


@router.get("/me", operation_id="getSession", summary="현재 사용자·역할")
def get_session(request: Request, identity: IdentityDep) -> SessionInfo:
    """토큰(또는 개발 모드의 헤더)으로 정해진 사용자와 역할을 돌려줍니다.

    공개 모드의 방문자는 보기 전용(viewer)이고, 역할 전환은 개발 모드에서만 됩니다.
    """
    mode = cast(Authenticator, request.app.state.authenticator).mode
    return SessionInfo(
        user=identity.user,
        role=identity.role.value,
        anonymous=identity.anonymous,
        auth_mode=mode,
        can_switch_role=mode == "dev",
    )
