"""GET /api/v1/me — the role the server actually grants, per authentication mode."""

from __future__ import annotations

from typing import Any

import pytest

from test_api_support import API, new_app, open_client

ADMIN_TOKEN = "admin-token-0123456789abcdef0123"
TOKENS = {ADMIN_TOKEN: {"user": "heo", "role": "admin"}}


def _me(headers: dict[str, str] | None = None, **settings: Any) -> tuple[int, dict[str, Any]]:
    with open_client(new_app(**settings)) as client:
        response = client.get(f"{API}/me", headers=headers or {})
    return response.status_code, response.json()


def test_dev_mode_takes_the_role_from_the_headers_and_allows_switching() -> None:
    status, body = _me({"X-Jettae-User": "park.jaehyun", "X-Jettae-Role": "lead"}, auth_mode="dev")
    assert status == 200
    assert body == {
        "user": "park.jaehyun",
        "role": "lead",
        "anonymous": False,
        "auth_mode": "dev",
        "can_switch_role": True,
    }


def test_public_mode_visitor_is_an_anonymous_viewer_whatever_the_headers_claim() -> None:
    status, body = _me(
        {"X-Jettae-User": "intruder", "X-Jettae-Role": "admin"}, auth_mode="public", api_tokens=TOKENS
    )
    assert status == 200
    assert body["role"] == "viewer"
    assert body["anonymous"] is True
    assert body["auth_mode"] == "public"
    assert body["can_switch_role"] is False


def test_public_mode_token_holder_gets_the_token_role() -> None:
    status, body = _me({"Authorization": f"Bearer {ADMIN_TOKEN}"}, auth_mode="public", api_tokens=TOKENS)
    assert status == 200
    assert (body["user"], body["role"], body["anonymous"], body["can_switch_role"]) == (
        "heo",
        "admin",
        False,
        False,
    )


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer not-a-known-token-value"}])
def test_token_mode_without_a_valid_token_is_401(headers: dict[str, str]) -> None:
    status, body = _me(headers, auth_mode="token", api_tokens=TOKENS)
    assert status == 401
    assert body["code"] == "unauthorized"
