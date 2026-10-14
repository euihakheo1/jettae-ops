"""Authentication, roles, the free-scan budget and the HTTP hardening every response gets.

Security properties are checked at both levels: the pure pieces (:class:`Authenticator`,
:class:`FreeScanQuota`, role ranks) directly — including hypothesis properties — and their effect
over HTTP (status codes, headers, Korean details), because a correct helper wired to the wrong
route protects nothing.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st
from starlette.datastructures import Headers
from structlog.testing import capture_logs

from jettae.api import create_app
from jettae.api.errors import ApiProblem
from jettae.api.middleware import API_CSP
from jettae.api.security import (
    MIN_TOKEN_LENGTH,
    Authenticator,
    AuthMode,
    FreeScanQuota,
    Identity,
    Role,
    parse_role,
    parse_user,
)
from jettae.improve import DECIDER_ROLES
from test_api_support import API, EDITOR, VIEWER, new_app, open_client
from test_api_support import settings as api_settings

LEAD_TOKEN = "lead-token-0123456789abcdef"
VIEWER_TOKEN = "viewer-token-0123456789abcdef"
TOKENS = {LEAD_TOKEN: {"user": "kim.lead", "role": "lead"}, VIEWER_TOKEN: {"user": "lee.viewer"}}


def _headers(**values: str) -> Headers:
    return Headers(headers=values)


# --------------------------------------------------------------------------- roles


def test_role_ranks_match_the_service_decider_roles() -> None:
    deciders = {role.value for role in Role if Identity("u", role).allows(Role.LEAD)}
    assert deciders == set(DECIDER_ROLES)
    editors = {role.value for role in Role if Identity("u", role).allows(Role.EDITOR)}
    assert editors == {"editor", "lead", "admin"}


@given(st.sampled_from(list(Role)), st.sampled_from(list(Role)), st.sampled_from(list(Role)))
def test_role_order_is_total_and_transitive(a: Role, b: Role, c: Role) -> None:
    ia, ib = Identity("u", a), Identity("u", b)
    assert ia.allows(b) or ib.allows(a)
    if ia.allows(b) and ib.allows(c):
        assert ia.allows(c)


@pytest.mark.parametrize("value", ["lead", "LEAD", " Lead ", "admin"])
def test_parse_role_is_case_insensitive(value: str) -> None:
    assert parse_role(value).value == value.strip().lower()


def test_parse_role_refuses_unknown_roles_in_korean() -> None:
    with pytest.raises(ValueError, match="알 수 없는 역할"):
        parse_role("root")


# --------------------------------------------------------------------------- dev identities


def test_dev_mode_defaults_to_an_anonymous_viewer() -> None:
    identity = Authenticator("dev", {}).identify(_headers())
    assert identity == Identity(user="demo", role=Role.VIEWER, anonymous=True)


def test_dev_mode_reads_user_and_role_headers() -> None:
    identity = Authenticator("dev", {}).identify(
        _headers(**{"x-jettae-user": " park ", "x-jettae-role": "Lead"})
    )
    assert identity == Identity(user="park", role=Role.LEAD)


def test_utf8_user_ids_survive_latin1_header_decoding() -> None:
    raw = Headers(raw=[(b"x-jettae-user", "박재현".encode())])
    assert Authenticator("dev", {}).identify(raw).user == "박재현"


@pytest.mark.parametrize("separator", ["\u2028", "\u2029"])
def test_line_separators_in_user_ids_are_refused_even_via_utf8_headers(separator: str) -> None:
    """U+2028/U+2029 end a line in many log viewers; sent as UTF-8 bytes they survive the
    Latin-1 header decoding, so the parser itself must refuse them."""
    raw = Headers(raw=[(b"x-jettae-user", f"kim{separator}승인자: admin".encode())])
    with pytest.raises(ApiProblem) as caught:
        Authenticator("dev", {}).identify(raw)
    assert caught.value.status_code == 422
    assert "줄 구분 문자" in str(caught.value.detail)


@given(st.text(min_size=1, max_size=40).filter(lambda s: s.strip() != ""))
def test_parse_user_is_idempotent_or_refuses(value: str) -> None:
    try:
        user: str | None = parse_user(value)
        refusal = ""
    except ValueError as exc:
        user, refusal = None, str(exc)
    if user is None:
        assert any("가" <= ch <= "힣" for ch in refusal)
    else:
        assert parse_user(user) == user
        assert user == user.strip()
        assert 0 < len(user) <= 64


@pytest.mark.parametrize(
    "headers",
    [
        {"X-Jettae-Role": "root"},
        {"X-Jettae-User": "a" * 65},
        {"X-Jettae-User": "bad\tuser"},
    ],
)
def test_malformed_dev_headers_are_422(headers: dict[str, str]) -> None:
    with open_client(new_app()) as client:
        response = client.get(f"{API}/workspaces", headers=headers)
    assert response.status_code == 422
    assert response.json()["code"] == "invalid_identity"
    assert "헤더가 올바르지 않습니다" in response.json()["detail"]


# --------------------------------------------------------------------------- token mode


@pytest.fixture(scope="module")
def token_client() -> Iterator[TestClient]:
    with open_client(new_app(auth_mode="token", api_tokens=TOKENS)) as client:
        yield client


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer nope-nope-nope-nope"}, {"Authorization": f"Basic {LEAD_TOKEN}"}, EDITOR],
)
def test_token_mode_refuses_missing_or_wrong_tokens(
    token_client: TestClient, headers: dict[str, str]
) -> None:
    response = token_client.get(f"{API}/workspaces", headers=headers)
    assert response.status_code == 401
    assert response.json()["code"] == "unauthorized"
    assert response.headers["www-authenticate"].startswith("Bearer")
    assert "Authorization: Bearer" in response.json()["detail"]


def test_token_mode_maps_tokens_to_roles_and_ignores_dev_headers(token_client: TestClient) -> None:
    assert token_client.get(f"{API}/workspaces", headers=_bearer(VIEWER_TOKEN)).status_code == 200
    escalated = {**_bearer(VIEWER_TOKEN), "X-Jettae-Role": "admin", "X-Jettae-User": "someone"}
    denied = token_client.post(f"{API}/workspaces", json={"name": "새 팀"}, headers=escalated)
    assert denied.status_code == 403
    assert "보기 전용(viewer)" in denied.json()["detail"]
    created = token_client.post(f"{API}/workspaces", json={"name": "새 팀"}, headers=_bearer(LEAD_TOKEN))
    assert created.status_code == 201


def test_public_endpoints_need_no_token(token_client: TestClient) -> None:
    assert token_client.get("/healthz").status_code == 200
    assert token_client.get("/readyz").status_code == 200
    assert token_client.get(f"{API}/rules").status_code == 200
    assert token_client.post(f"{API}/events", json={"name": "landing_view"}).status_code == 202
    signup = {"email": "kim@example.com", "consent_privacy": True}
    assert token_client.post(f"{API}/signups", json=signup).status_code == 201


@pytest.mark.parametrize(
    ("tokens", "fragment"),
    [
        ({"short": {"user": "kim", "role": "lead"}}, f"{MIN_TOKEN_LENGTH}자 이상"),
        ({"x" * 20 + " y": {"user": "kim"}}, "공백 없이"),
        ({"x" * 20: {"user": "kim", "role": "root"}}, "알 수 없는 역할"),
        ({"x" * 20: {"role": "lead"}}, "사용자 id가 비어"),
    ],
)
def test_bad_token_tables_stop_the_app_at_creation(tokens: dict[str, dict[str, str]], fragment: str) -> None:
    with pytest.raises(ValueError, match=fragment):
        create_app(api_settings(auth_mode="token", api_tokens=tokens))


def test_token_mode_without_any_token_stops_the_app_at_creation() -> None:
    """A forgotten JETTAE_API_TOKENS must fail the start, not serve 401 to everyone."""
    with pytest.raises(ValueError, match="하나 이상"):
        create_app(api_settings(auth_mode="token", api_tokens={}))
    assert Authenticator("dev", {}).mode == "dev"  # dev mode needs no table


def test_token_errors_never_echo_the_token() -> None:
    secret = "s" * 10
    with pytest.raises(ValueError, match="토큰") as caught:
        Authenticator("token", {secret: {"user": "kim"}})
    assert secret not in str(caught.value)


# --------------------------------------------------------------------------- free-scan budget

KST_MIDNIGHT_UTC = datetime(2026, 10, 28, 15, 0, tzinfo=UTC)  # 2026-10-29 00:00 KST


def test_quota_counts_per_korean_day_and_rolls_over_at_kst_midnight() -> None:
    quota = FreeScanQuota(2)
    before = KST_MIDNIGHT_UTC - timedelta(seconds=30)
    for _ in range(2):
        with quota.reserve(before):
            pass
    with pytest.raises(ApiProblem) as refused, quota.reserve(before):
        pass
    assert refused.value.status_code == 429
    assert refused.value.headers == {"Retry-After": "30"}
    assert quota.used(before) == 2
    assert quota.used(KST_MIDNIGHT_UTC) == 0
    with quota.reserve(KST_MIDNIGHT_UTC):
        pass
    assert quota.used(KST_MIDNIGHT_UTC) == 1


def test_quota_returns_the_scan_when_the_work_fails() -> None:
    quota = FreeScanQuota(1)
    now = KST_MIDNIGHT_UTC
    with pytest.raises(RuntimeError), quota.reserve(now):
        raise RuntimeError("boom")
    assert quota.used(now) == 0
    with quota.reserve(now):
        pass
    assert quota.used(now) == 1


@settings(max_examples=60)
@given(st.integers(min_value=0, max_value=4), st.lists(st.booleans(), max_size=12))
def test_quota_never_exceeds_its_limit(limit: int, outcomes: list[bool]) -> None:
    quota = FreeScanQuota(limit)
    now = KST_MIDNIGHT_UTC
    done = 0
    for succeeds in outcomes:
        try:
            with quota.reserve(now):
                if not succeeds:
                    raise KeyError("fails")
            done += 1
        except ApiProblem:
            assert quota.used(now) == limit
        except KeyError:
            pass
        assert quota.used(now) == min(done, limit)
        assert quota.used(now) <= limit


# --------------------------------------------------------------------------- HTTP hardening


def test_every_api_response_carries_the_security_headers() -> None:
    with open_client(new_app()) as client:
        for response in (client.get(f"{API}/rules"), client.get(f"{API}/workspaces/nope")):
            assert response.headers["x-content-type-options"] == "nosniff"
            assert response.headers["x-frame-options"] == "DENY"
            assert response.headers["referrer-policy"] == "no-referrer"
            assert response.headers["content-security-policy"] == API_CSP
            assert response.headers["cache-control"] == "no-store"
            assert "strict-transport-security" not in response.headers
        docs = client.get("/api/docs")
        assert docs.status_code == 200
        assert "content-security-policy" not in docs.headers
    with open_client(new_app(), base_url="https://testserver") as secure:
        assert "max-age=" in secure.get("/healthz").headers["strict-transport-security"]


def test_request_ids_are_generated_echoed_and_put_in_error_bodies() -> None:
    with open_client(new_app()) as client:
        generated = client.get("/healthz").headers["x-request-id"]
        assert len(generated) == 32
        assert (
            client.get("/healthz", headers={"X-Request-ID": "lb-42.a:b"}).headers["x-request-id"]
            == "lb-42.a:b"
        )
        hostile = client.get("/healthz", headers={"X-Request-ID": "x" * 65})
        assert hostile.headers["x-request-id"] != "x" * 65
        error = client.get(f"{API}/workspaces/nope")
        assert error.json()["request_id"] == error.headers["x-request-id"]


def test_cors_allows_the_configured_origin_only() -> None:
    preflight = {
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "X-Jettae-Role, Content-Type",
    }
    with open_client(new_app(cors_origins=["http://localhost:5173"])) as client:
        allowed = client.options(f"{API}/workspaces", headers=preflight)
        assert allowed.status_code == 200
        assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
        assert "x-jettae-role" in allowed.headers["access-control-allow-headers"].lower()
        assert allowed.headers["x-request-id"]
        refused = client.options(f"{API}/workspaces", headers={**preflight, "Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in refused.headers
        simple = client.get(f"{API}/rules", headers={"Origin": "http://localhost:5173"})
        assert "x-request-id" in simple.headers["access-control-expose-headers"].lower()


def test_access_log_has_no_query_strings_headers_or_user_ids() -> None:
    with open_client(new_app()) as client, capture_logs() as logs:
        client.get(
            f"{API}/workspaces",
            params={"email": "kim@example.com"},
            headers={"X-Jettae-User": "secret.person", "X-Jettae-Role": "viewer"},
        )
    entries = [entry for entry in logs if entry["event"] == "http.request"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["method"] == "GET"
    assert entry["path"] == f"{API}/workspaces"
    assert entry["route"] == "/workspaces"
    assert entry["status"] == 200
    assert isinstance(entry["duration_ms"], float)
    flattened = repr(entry)
    assert "kim@example.com" not in flattened
    assert "secret.person" not in flattened


def test_viewers_cannot_write() -> None:
    with open_client(new_app()) as client:
        created = client.post(f"{API}/workspaces", json={"name": "보기 전용"}, headers=VIEWER)
        assert created.status_code == 403
        assert created.json()["detail"].startswith("이 작업은 편집자(editor) 이상만")
        assert client.post(f"{API}/workspaces", json={"name": "편집자"}, headers=EDITOR).status_code == 201


@pytest.mark.parametrize("mode", ["token", "public"])
def test_a_missing_token_table_names_the_length_the_validator_wants(mode: AuthMode) -> None:
    """The example must not ask for more (or less) than the check: 16 characters, 32 recommended."""
    with pytest.raises(ValueError, match="JETTAE_API_TOKENS") as refused:
        Authenticator(mode, {})
    message = str(refused.value)
    assert "16자 이상" in message
    assert "32자 이상 권장" in message
    assert "<32자 이상 토큰>" not in message
