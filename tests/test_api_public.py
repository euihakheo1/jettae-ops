"""The public beta website (``JETTAE_AUTH_MODE=public``): anonymous visitors, private workspaces
opened by an access key, the anonymous free scan, and the admin-only sign-up export.

The access rule is checked from the outside, over HTTP, for *every* route that addresses a
workspace — enumerated from the app's routing table, so a route added later cannot slip past the
guard unnoticed.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import re
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass
from datetime import timedelta

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from sqlalchemy import select
from starlette.datastructures import Headers

from jettae.api import API_PREFIX, OPENAPI_URL, create_app
from jettae.api.errors import ApiProblem
from jettae.api.routers import API_ROUTERS, ROOT_ROUTERS
from jettae.api.routers.growth import SIGNUP_CSV_COLUMNS
from jettae.api.security import ANONYMOUS_USER, Authenticator, Identity, Role
from jettae.config import Settings
from jettae.ingest import load_bytes, load_roster
from jettae.orchestrator import (
    DEMO_WORKSPACE_NAME,
    EVENT_RUNBOOK_UPLOAD,
    FREE_SCAN_WORKSPACE_NAME,
    JettaeService,
)
from jettae.orchestrator.service import ADMIN_AUDIT_SCOPE, AUDIT_SIGNUPS_READ
from jettae.store import Signup, SqlStore
from jettae.store.models import EventRow
from test_api_support import (
    API,
    CLOCK,
    EDITOR,
    LEAD,
    SMALL_ROSTER,
    SMALL_RUNBOOK,
    new_app,
    new_service,
    open_client,
)
from test_api_support import settings as api_settings

ADMIN_TOKEN = "admin-token-0123456789abcdef0123"
LEAD_TOKEN = "lead-token-0123456789abcdef01234"
VIEWER_TOKEN = "viewer-token-0123456789abcdef012"
TOKENS = {
    ADMIN_TOKEN: {"user": "heo", "role": "admin"},
    LEAD_TOKEN: {"user": "kim.lead", "role": "lead"},
    VIEWER_TOKEN: {"user": "lee.viewer", "role": "viewer"},
}
KEY_HEADER = "X-Jettae-Workspace-Key"


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


ADMIN = _bearer(ADMIN_TOKEN)


def _scan_files(name: str = "결제.md") -> list[tuple[str, tuple[str, bytes, str]]]:
    return [
        ("files", (name, SMALL_RUNBOOK.encode("utf-8"), "text/markdown")),
        ("roster", ("roster.csv", SMALL_ROSTER.encode("utf-8"), "text/csv")),
    ]


def _not_found(workspace_id: str) -> str:
    return f"워크스페이스 {workspace_id}을(를) 찾을 수 없습니다."


def _diagnosed_workspace(service: JettaeService, name: str) -> str:
    """A workspace created behind the API (the demo name is reserved there), diagnosed once."""
    workspace = service.create_workspace(name)
    service.upload_inputs(
        workspace.id,
        documents=[load_bytes("결제.md", SMALL_RUNBOOK.encode("utf-8"))],
        roster=load_roster(SMALL_ROSTER.encode("utf-8")),
    )
    asyncio.run(service.diagnose(workspace.id, actor="test"))
    return workspace.id


@dataclass
class Site:
    client: TestClient
    service: JettaeService
    demo: str
    private: str


@pytest.fixture(scope="module")
def site() -> Iterator[Site]:
    service = new_service()
    demo = _diagnosed_workspace(service, DEMO_WORKSPACE_NAME)
    private = _diagnosed_workspace(service, "결제 플랫폼팀")
    app = new_app(service=service, auth_mode="public", api_tokens=TOKENS)
    with open_client(app) as client:
        yield Site(client=client, service=service, demo=demo, private=private)


# --------------------------------------------------------------------------- authenticator


def _identify(**headers: str) -> Identity:
    return Authenticator("public", TOKENS).identify(Headers(headers=headers))


def test_public_mode_without_authorization_is_an_anonymous_viewer() -> None:
    assert _identify() == Identity(user=ANONYMOUS_USER, role=Role.VIEWER, anonymous=True)


def test_public_mode_maps_valid_tokens_like_token_mode() -> None:
    assert _identify(authorization=f"Bearer {ADMIN_TOKEN}") == Identity(user="heo", role=Role.ADMIN)
    assert _identify(authorization=f"bearer  {LEAD_TOKEN} ") == Identity(user="kim.lead", role=Role.LEAD)


@pytest.mark.parametrize(
    "authorization",
    ["", "Bearer", "Bearer ", "Bearer wrong-token-0123456789abcdef", f"Basic {ADMIN_TOKEN}", ADMIN_TOKEN],
)
def test_public_mode_refuses_present_but_invalid_authorization(authorization: str) -> None:
    """A mistyped admin token must not quietly turn the admin into a visitor."""
    with pytest.raises(ApiProblem) as caught:
        _identify(authorization=authorization)
    assert caught.value.status_code == 401
    assert caught.value.headers == {"WWW-Authenticate": 'Bearer realm="jettae"'}


def test_public_mode_ignores_the_dev_identity_headers() -> None:
    claimed = {"x-jettae-user": "mallory", "x-jettae-role": "admin"}
    assert _identify(**claimed) == Identity(user=ANONYMOUS_USER, role=Role.VIEWER, anonymous=True)
    assert _identify(**claimed, authorization=f"Bearer {VIEWER_TOKEN}") == Identity(
        user="lee.viewer", role=Role.VIEWER
    )


def test_public_mode_without_any_token_stops_the_app_at_creation() -> None:
    """Sign-ups nobody can read are worse than a server that refuses to start."""
    with pytest.raises(ValueError, match="관리자 토큰"):
        create_app(api_settings(auth_mode="public", api_tokens={}))
    with pytest.raises(ValueError, match="16자 이상"):
        create_app(api_settings(auth_mode="public", api_tokens={"short": {"user": "heo", "role": "admin"}}))


def test_settings_accept_exactly_three_auth_modes() -> None:
    assert Settings(_env_file=None, auth_mode="public").auth_mode == "public"
    with pytest.raises(ValueError, match="auth_mode"):
        Settings(_env_file=None, auth_mode="open")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- over HTTP


def test_http_public_mode_identities(site: Site) -> None:
    client = site.client
    assert client.get(f"{API}/workspaces").status_code == 200
    wrong = client.get(f"{API}/workspaces", headers=_bearer("wrong-token-0123456789abcdef"))
    assert wrong.status_code == 401
    assert wrong.json()["code"] == "unauthorized"
    # the dev headers do not lift a visitor to editor
    created = client.post(f"{API}/workspaces", json={"name": "새 팀"}, headers=EDITOR)
    assert created.status_code == 403
    assert "보기 전용(viewer)" in created.json()["detail"]


def test_anonymous_workspace_list_holds_only_the_demo(site: Site) -> None:
    anonymous = site.client.get(f"{API}/workspaces")
    assert anonymous.status_code == 200
    assert [(w["id"], w["is_demo"]) for w in anonymous.json()] == [(site.demo, True)]
    as_admin = site.client.get(f"{API}/workspaces", headers=ADMIN)
    assert {site.demo, site.private} <= {w["id"] for w in as_admin.json()}


def test_demo_workspace_is_open_to_visitors(site: Site) -> None:
    for path in ("", "/report", "/graph", "/versions", "/patches", "/inputs", "/report.html"):
        response = site.client.get(f"{API}/workspaces/{site.demo}{path}")
        assert response.status_code == 200, (path, response.text)


def test_private_workspace_answers_visitors_like_an_unknown_one(site: Site) -> None:
    unknown = site.client.get(f"{API}/workspaces/ws-0000000000000000")
    for headers in ({}, {KEY_HEADER: "not-the-key"}, {KEY_HEADER: ""}):
        hidden = site.client.get(f"{API}/workspaces/{site.private}", headers=headers)
        assert hidden.status_code == unknown.status_code == 404
        assert hidden.json()["code"] == unknown.json()["code"] == "not_found"
        assert hidden.json()["detail"] == _not_found(site.private)
    # token holders are not affected
    assert (
        site.client.get(f"{API}/workspaces/{site.private}", headers=_bearer(VIEWER_TOKEN)).status_code == 200
    )


def _routes_with(param: str) -> list[tuple[str, str]]:
    """``(method, full path)`` of every mounted route with ``{param}`` in its path, hidden ones
    included — taken from the routers ``create_app`` mounts (FastAPI versions differ in how they
    expose included routes on the app itself)."""
    mounted = [(API_PREFIX, router) for router in API_ROUTERS] + [("", router) for router in ROOT_ROUTERS]
    found: list[tuple[str, str]] = []
    for prefix, router in mounted:
        for route in router.routes:
            assert isinstance(route, APIRoute)
            if f"{{{param}}}" in route.path:
                found += [(method, prefix + route.path) for method in sorted(route.methods)]
    return found


def test_route_enumeration_covers_the_published_api(site: Site) -> None:
    document = site.client.get(OPENAPI_URL).json()
    published = {
        (method.upper(), path) for path, ops in document["paths"].items() for method in ops if "{ws}" in path
    }
    assert published
    assert published <= set(_routes_with("ws"))


def _fill(path: str, workspace_id: str) -> str:
    return re.sub(r"\{(\w+)\}", lambda match: workspace_id if match[1] == "ws" else "1", path)


def test_every_workspace_route_is_guarded(site: Site) -> None:
    """Enumerated from the routing table: a new ``{ws}`` route is covered the day it is added."""
    routes = _routes_with("ws")
    assert len(routes) >= 17
    assert {
        ("POST", f"{API}/workspaces/{{ws}}/inputs"),
        ("POST", f"{API}/workspaces/{{ws}}/diagnose"),
        ("GET", f"{API}/workspaces/{{ws}}/report.html"),
        ("GET", f"{API}/workspaces/{{ws}}/versions/diff"),  # hidden from the schema, guarded all the same
        ("POST", f"{API}/workspaces/{{ws}}/versions/{{n}}/revert"),
        ("POST", f"{API}/workspaces/{{ws}}/patches/draft"),
    } <= set(routes)
    for method, path in routes:
        url = _fill(path, site.private)
        for headers in ({}, {KEY_HEADER: "guess"}):
            response = site.client.request(method, url, headers=headers)
            assert response.status_code == 404, (method, path, response.status_code, response.text)
            assert response.json() == {
                "detail": _not_found(site.private),
                "code": "not_found",
                "request_id": response.headers["x-request-id"],
            }, (method, path)


def test_guard_is_a_no_op_outside_public_mode() -> None:
    service = new_service()
    private = _diagnosed_workspace(service, "결제 플랫폼팀")
    with open_client(new_app(service=service)) as dev:
        routes = _routes_with("ws")
        assert dev.get(f"{API}/workspaces/{private}").status_code == 200
        assert dev.get(f"{API}/workspaces/{private}/report").status_code == 200
        assert [w["id"] for w in dev.get(f"{API}/workspaces").json()] == [private]
    tokens = {ADMIN_TOKEN: {"user": "heo", "role": "admin"}}
    with open_client(new_app(service=service, auth_mode="token", api_tokens=tokens)) as token:
        for method, path in routes:
            assert token.request(method, _fill(path, private)).status_code == 401, (method, path)
        assert token.get(f"{API}/workspaces/{private}/graph", headers=ADMIN).status_code == 200


def test_every_patch_route_follows_the_patch_workspace(site: Site) -> None:
    private_patch = site.service.patches(site.private)[0].id
    routes = _routes_with("patch_id")
    assert len(routes) >= 3
    for method, path in routes:
        response = site.client.request(method, path.replace("{patch_id}", private_patch))
        assert response.status_code == 404, (method, path, response.text)
        assert response.json()["detail"] == f"수정안 {private_patch}을(를) 찾을 수 없습니다."


def test_patches_by_id_follow_their_workspace(site: Site) -> None:
    private_patch = site.service.patches(site.private)[0].id
    demo_patch = site.service.patches(site.demo)[0].id
    unknown = site.client.get(f"{API}/patches/patch-000000000000000000000000")
    assert unknown.status_code == 404
    hidden = site.client.get(f"{API}/patches/{private_patch}")
    assert hidden.status_code == 404
    assert hidden.json()["detail"] == f"수정안 {private_patch}을(를) 찾을 수 없습니다."
    assert hidden.json()["code"] == unknown.json()["code"] == "not_found"
    assert site.client.get(f"{API}/patches/{demo_patch}").status_code == 200
    assert site.client.get(f"{API}/patches/{private_patch}", headers=ADMIN).status_code == 200
    # deciding needs lead: a visitor is refused (and a private patch is not even revealed)
    for action, body in (("approve", None), ("reject", {"reason": "아니요"})):
        on_demo = site.client.post(f"{API}/patches/{demo_patch}/{action}", json=body)
        assert on_demo.status_code == 403, on_demo.text
        assert on_demo.json()["code"] == "forbidden"
        on_private = site.client.post(f"{API}/patches/{private_patch}/{action}", json=body)
        assert on_private.status_code == 404
        viewer = site.client.post(
            f"{API}/patches/{private_patch}/{action}", json=body, headers=_bearer(VIEWER_TOKEN)
        )
        assert viewer.status_code == 403


def test_open_routes_stay_open_in_public_mode(site: Site) -> None:
    client = site.client
    assert client.get("/healthz").status_code == 200
    assert client.get(f"{API}/rules").status_code == 200
    assert client.post(f"{API}/events", json={"name": "landing_first_view"}).status_code == 202
    assert client.get(f"{API}/metrics/demand").status_code == 403
    assert client.get(f"{API}/metrics/demand", headers=_bearer(LEAD_TOKEN)).status_code == 200


def test_cors_lets_the_web_app_send_the_workspace_key() -> None:
    preflight = {
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": KEY_HEADER,
    }
    app = new_app(auth_mode="public", api_tokens=TOKENS, cors_origins=["http://localhost:5173"])
    with open_client(app) as client:
        allowed = client.options(f"{API}/workspaces/ws-1/report", headers=preflight)
    assert allowed.status_code == 200
    assert KEY_HEADER.lower() in allowed.headers["access-control-allow-headers"].lower()


# --------------------------------------------------------------------------- free scans


@pytest.fixture
def public() -> Iterator[Site]:
    service = new_service()
    demo = _diagnosed_workspace(service, DEMO_WORKSPACE_NAME)
    with open_client(new_app(service=service, auth_mode="public", api_tokens=TOKENS)) as client:
        yield Site(client=client, service=service, demo=demo, private="")


def test_free_scan_returns_a_key_that_opens_the_workspace(public: Site) -> None:
    client, service = public.client, public.service
    response = client.post(f"{API}/free-scans", files=_scan_files())
    assert response.status_code == 201, response.text
    body = response.json()
    assert set(body) == {"workspace_id", "access_key", "readiness_overall", "findings"}
    ws, key = body["workspace_id"], body["access_key"]
    report = service.latest_report(ws)
    assert body["readiness_overall"] == report.readiness.overall
    assert body["findings"] == len(report.findings) > 0
    assert service.get_workspace(ws).name == FREE_SCAN_WORKSPACE_NAME
    assert len(key) >= 43
    assert response.headers["cache-control"] == "no-store"

    # only the digest is stored
    digest = service.store.access.access_key_digest(ws)
    assert digest == hashlib.sha256(key.encode()).hexdigest()
    assert key not in repr(service.store.audit.entries(ws))

    with_key = {KEY_HEADER: key}
    for path in ("", "/report", "/graph", "/inputs", "/patches", "/versions", "/report.md", "/report.html"):
        opened = client.get(f"{API}/workspaces/{ws}{path}", headers=with_key)
        assert opened.status_code == 200, (path, opened.text)
        hidden = client.get(f"{API}/workspaces/{ws}{path}")
        assert hidden.status_code == 404, path
        assert hidden.json()["detail"] == _not_found(ws)
    assert client.get(f"{API}/workspaces/{ws}", headers={KEY_HEADER: key[:-1]}).status_code == 404
    patch_id = service.patches(ws)[0].id
    assert client.get(f"{API}/patches/{patch_id}", headers=with_key).status_code == 200
    assert client.get(f"{API}/patches/{patch_id}").status_code == 404
    # a key opens its own workspace only
    other = client.post(f"{API}/free-scans", files=_scan_files("다른.md")).json()
    assert client.get(f"{API}/workspaces/{other['workspace_id']}", headers=with_key).status_code == 404
    # visitors still see only the demo in the list
    assert [w["id"] for w in client.get(f"{API}/workspaces").json()] == [public.demo]
    # the key does not lift the role: uploads still need an editor
    upload = client.post(f"{API}/workspaces/{ws}/inputs", files=_scan_files(), headers=with_key)
    assert upload.status_code == 403


def test_free_scan_counts_runbook_uploads_server_side(public: Site) -> None:
    client, service = public.client, public.service
    files = [*_scan_files(), ("files", ("추가.md", b"# DB runbook\n\n- owner: kim\n", "text/markdown"))]
    assert client.post(f"{API}/free-scans", files=files).status_code == 201
    assert client.post(f"{API}/free-scans", files=_scan_files()).status_code == 201
    assert service.demand_metrics()["events"] == {EVENT_RUNBOOK_UPLOAD: 2}
    refused = client.post(f"{API}/events", json={"name": EVENT_RUNBOOK_UPLOAD, "props": {"files": 9}})
    assert refused.status_code == 422
    assert service.demand_metrics()["events"] == {EVENT_RUNBOOK_UPLOAD: 2}


def test_free_scan_event_records_the_number_of_runbook_files() -> None:
    with closing(SqlStore("sqlite+pysqlite:///:memory:", clock=lambda: CLOCK)) as store:
        service = new_service(store=store)
        documents = [
            load_bytes("a.md", SMALL_RUNBOOK.encode("utf-8")),
            load_bytes("b.md", b"# DB runbook\n\n- owner: kim\n"),
        ]
        outcome = asyncio.run(service.free_scan(documents=documents, actor="anonymous"))
        with store.engine.connect() as conn:
            rows = conn.execute(select(EventRow.name, EventRow.props)).all()
    assert [(name, props) for name, props in rows] == [(EVENT_RUNBOOK_UPLOAD, {"files": 2})]
    assert outcome.access_key
    assert outcome.workspace_id.startswith("ws-")


def test_free_scans_in_the_same_instant_get_distinct_workspaces_and_keys(public: Site) -> None:
    first = public.client.post(f"{API}/free-scans", files=_scan_files()).json()
    second = public.client.post(f"{API}/free-scans", files=_scan_files()).json()
    assert first["workspace_id"] != second["workspace_id"]
    assert first["access_key"] != second["access_key"]
    assert first["findings"] == second["findings"]  # same runbook, same (deterministic) verdict


def test_free_scan_keeps_no_runbook_text(public: Site) -> None:
    service = public.service
    ws = public.client.post(f"{API}/free-scans", files=_scan_files()).json()["workspace_id"]
    stored = service.store.documents.list_documents(ws)
    assert stored
    assert not any(meta.raw_stored for meta in stored)
    assert all(service.store.documents.get_document(ws, meta.doc_id) is None for meta in stored)
    # nor in the process: a stream of anonymous scans must not accumulate text in memory
    assert service.input_summary(ws).diagnosable is False


@pytest.mark.parametrize(
    ("files", "status", "fragment", "charged"),
    [
        # the server had to read the file: charged (a parse bomb must not be free)
        ([("files", ("런북.exe", b"MZ\x00\x00", "application/octet-stream"))], 422, "", 1),
        # refused by request validation before the route ran: nothing to charge
        ([("roster", ("roster.csv", SMALL_ROSTER.encode("utf-8"), "text/csv"))], 422, "files", 0),
        ([("files", ("", b"# x", "text/markdown"))], 422, "", 0),
    ],
)
def test_free_scan_input_errors_leave_nothing_behind(
    files: list[tuple[str, tuple[str, bytes, str]]], status: int, fragment: str, charged: int
) -> None:
    service = new_service()
    with open_client(new_app(service=service, auth_mode="public", api_tokens=TOKENS)) as client:
        response = client.post(f"{API}/free-scans", files=files)
        quota = client.app.state.free_scans  # type: ignore[attr-defined]
    assert response.status_code == status, response.text
    assert fragment in response.text
    assert re.search(r"[가-힣]", response.json()["detail"])
    assert quota.used(CLOCK) == charged
    assert service.list_workspaces() == []
    assert service.demand_metrics()["events"] == {}


def test_free_scan_size_budget_is_the_upload_budget() -> None:
    """The route takes files: the 1 MB JSON limit must not apply, ``JETTAE_MAX_UPLOAD_MB`` does."""
    big = b"x" * (int(1.5 * 1024 * 1024))
    with open_client(new_app(auth_mode="public", api_tokens=TOKENS, max_upload_mb=2)) as client:
        # past the body limit: refused by the loader (unsupported format), not as too large
        loaded = client.post(
            f"{API}/free-scans", files=[("files", ("big.bin", big, "application/octet-stream"))]
        )
        assert loaded.status_code == 422, loaded.text
        too_big = client.post(f"{API}/free-scans", files=[("files", ("big.md", big + big, "text/markdown"))])
        assert too_big.status_code == 413
        assert too_big.json()["code"] == "payload_too_large"


def test_free_scan_is_metered_for_visitors() -> None:
    app = new_app(auth_mode="public", api_tokens=TOKENS, daily_free_scan_limit=1)
    with open_client(app) as client:
        assert client.post(f"{API}/free-scans", files=_scan_files()).status_code == 201
        refused = client.post(f"{API}/free-scans", files=_scan_files())
        assert refused.status_code == 429
        assert refused.json()["code"] == "free_scan_limit"
        assert 0 < int(refused.headers["retry-after"]) <= 24 * 3600
        # editors and above are not metered (here: an admin token)
        assert client.post(f"{API}/free-scans", files=_scan_files(), headers=ADMIN).status_code == 201


def test_free_scan_in_token_and_dev_modes() -> None:
    with open_client(new_app(auth_mode="token", api_tokens=TOKENS)) as token:
        assert token.post(f"{API}/free-scans", files=_scan_files()).status_code == 401
        viewer = token.post(f"{API}/free-scans", files=_scan_files(), headers=_bearer(VIEWER_TOKEN))
        assert viewer.status_code == 201
        ws = viewer.json()["workspace_id"]
        # outside public mode the key is not needed (token holders read every workspace)
        assert token.get(f"{API}/workspaces/{ws}/report", headers=_bearer(VIEWER_TOKEN)).status_code == 200
    with open_client(new_app()) as dev:
        assert dev.post(f"{API}/free-scans", files=_scan_files()).status_code == 201


# --------------------------------------------------------------------------- sign-up export


def _signup(service: JettaeService, email: str, *, minutes: int, **fields: object) -> None:
    service.signup(
        Signup(
            email=email,
            consent_privacy=True,
            created_at=CLOCK + timedelta(minutes=minutes),
            **fields,  # type: ignore[arg-type]
        )
    )


@pytest.fixture
def signups() -> Iterator[Site]:
    service = new_service()
    _signup(service, "late@example.com", minutes=5, organization="늦은 회사", source="landing")
    _signup(
        service,
        "kim@example.com",
        minutes=0,
        organization='=HYPERLINK("https://evil.example","열기")',
        team_size="10-49",
        message="추석 대비, 결제 런북\n점검하고 싶습니다",
        consent_marketing=True,
        source="landing",
    )
    with open_client(new_app(service=service, auth_mode="public", api_tokens=TOKENS)) as client:
        yield Site(client=client, service=service, demo="", private="")


def test_signup_list_is_for_admins_only(signups: Site) -> None:
    client = signups.client
    anonymous = client.get(f"{API}/signups")
    assert anonymous.status_code == 403
    lead = client.get(f"{API}/signups", headers=_bearer(LEAD_TOKEN))
    assert lead.status_code == 403
    assert lead.json()["code"] == "forbidden"
    assert "관리자(admin)만" in lead.json()["detail"]
    assert "팀 리드(lead)" in lead.json()["detail"]
    assert client.get(f"{API}/signups?format=csv", headers=_bearer(LEAD_TOKEN)).status_code == 403
    assert client.get(f"{API}/signups", headers=_bearer("wrong-token-0123456789abcdef")).status_code == 401
    # refusals are not reads: nothing in the audit log
    assert signups.service.store.audit.entries(ADMIN_AUDIT_SCOPE) == []


def test_signup_list_as_json(signups: Site) -> None:
    response = signups.client.get(f"{API}/signups", headers=ADMIN)
    assert response.status_code == 200
    body = response.json()
    assert [s["email"] for s in body] == ["kim@example.com", "late@example.com"]
    assert body[0]["consent_marketing"] is True
    assert body[0]["team_size"] == "10-49"
    assert body[1]["organization"] == "늦은 회사"
    [entry] = signups.service.store.audit.entries(ADMIN_AUDIT_SCOPE)
    assert (entry.actor, entry.action, entry.payload) == (
        "heo",
        AUDIT_SIGNUPS_READ,
        {"count": 2, "format": "json"},
    )


def test_signup_list_as_csv_for_excel(signups: Site) -> None:
    response = signups.client.get(f"{API}/signups", params={"format": "csv"}, headers=ADMIN)
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/csv; charset=utf-8"
    assert response.headers["content-disposition"] == 'attachment; filename="signups.csv"'
    assert response.content.startswith(b"\xef\xbb\xbf")  # BOM: Excel reads the Korean as UTF-8
    rows = list(csv.reader(io.StringIO(response.content.decode("utf-8-sig"), newline="")))
    assert (
        tuple(rows[0])
        == SIGNUP_CSV_COLUMNS
        == (
            "created_at",
            "email",
            "organization",
            "team_size",
            "consent_marketing",
            "source",
            "message",
        )
    )
    assert rows[1] == [
        "2026-10-28T14:00:00+00:00",
        "kim@example.com",
        '\'=HYPERLINK("https://evil.example","열기")',  # CSV injection neutralised
        "10-49",
        "true",
        "landing",
        "추석 대비, 결제 런북\n점검하고 싶습니다",
    ]
    assert rows[2] == [
        "2026-10-28T14:05:00+00:00",
        "late@example.com",
        "늦은 회사",
        "",
        "false",
        "landing",
        "",
    ]
    assert len(rows) == 3
    entries = signups.service.store.audit.entries(ADMIN_AUDIT_SCOPE)
    assert entries[-1].payload == {"count": 2, "format": "csv"}


def test_signup_list_refuses_unknown_formats(signups: Site) -> None:
    response = signups.client.get(f"{API}/signups", params={"format": "xlsx"}, headers=ADMIN)
    assert response.status_code == 422
    assert response.json()["errors"][0]["field"] == "format"


def test_empty_signup_csv_still_has_the_header() -> None:
    with open_client(new_app(auth_mode="public", api_tokens=TOKENS)) as client:
        response = client.get(f"{API}/signups?format=csv", headers=ADMIN)
    assert response.content == ("﻿" + ",".join(SIGNUP_CSV_COLUMNS) + "\r\n").encode("utf-8")


def test_dev_mode_admin_header_reads_the_list() -> None:
    with open_client(new_app()) as client:
        assert client.get(f"{API}/signups", headers=LEAD).status_code == 403
        admin = {"X-Jettae-User": "demo.admin", "X-Jettae-Role": "admin"}
        assert client.get(f"{API}/signups", headers=admin).json() == []
