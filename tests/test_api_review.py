"""Adversarial checks of the API added by the independent review: races between decisions,
hostile runbook text in the served report, byte-for-byte reproducibility across processes (P6),
personal data in the access log, and role claims that a token mode must ignore.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi.testclient import TestClient
from structlog.testing import capture_logs

from test_api_support import (
    API,
    EDITOR,
    LEAD,
    SMALL_ROSTER,
    create_workspace,
    new_app,
    new_service,
    open_client,
)

LEAD_TOKEN = "review-lead-token-0123456789"
VIEWER_TOKEN = "review-viewer-token-0123456789"
TOKENS = {LEAD_TOKEN: {"user": "kim.lead", "role": "lead"}, VIEWER_TOKEN: {"user": "lee.viewer"}}

#: runbook text that tries to become markup in the HTML evidence report
HOSTILE_RUNBOOK = (
    "# <script>alert('runbook')</script> 결제 런북\n\n## 결제 서비스\n\n"
    "- 감지: 결제 실패율 알람 → 김서연 사원 호출 <img src=x onerror=alert(1)>\n"
    "- 담당: 김서연 사원 (슬랙, 전화) <iframe src=https://evil.example></iframe>\n"
)


def _demo(client: TestClient, headers: dict[str, str] | None = None) -> str:
    response = client.post(f"{API}/demo/dmc-shop", json={"llm_mode": "off"}, headers=headers)
    assert response.status_code == 200, response.text
    workspace_id: str = response.json()["workspace_id"]
    return workspace_id


def _validated(client: TestClient, ws: str, headers: dict[str, str] | None = None) -> list[dict[str, Any]]:
    patches = client.get(f"{API}/workspaces/{ws}/patches", headers=headers).json()
    return [patch for patch in patches if patch["status"] == "validated"]


def test_concurrent_approvals_append_exactly_one_version() -> None:
    """Drafts share one base version: whichever approval commits first wins, every other one is a
    409 conflict — never a second version built on a stale base."""
    with open_client(new_app()) as client:
        ws = _demo(client)
        candidates = _validated(client, ws)[:4]
        assert len(candidates) >= 2

        def approve(patch: dict[str, Any]) -> tuple[int, str | None]:
            response = client.post(f"{API}/patches/{patch['id']}/approve", headers=LEAD)
            return response.status_code, response.json().get("code")

        with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
            outcomes = sorted(pool.map(approve, candidates), key=lambda outcome: outcome[0])
        assert outcomes == [(200, None)] + [(409, "version_conflict")] * (len(candidates) - 1)
        assert [v["number"] for v in client.get(f"{API}/workspaces/{ws}/versions").json()] == [1, 2]


def test_runbook_markup_never_reaches_the_served_html_report() -> None:
    with open_client(new_app()) as client:
        ws = create_workspace(client)
        files = [
            ("runbooks", ("<svg onload=alert(1)>.md", HOSTILE_RUNBOOK.encode("utf-8"), "text/markdown")),
            ("roster", ("roster.csv", SMALL_ROSTER.encode("utf-8"), "text/csv")),
        ]
        assert client.post(f"{API}/workspaces/{ws}/inputs", files=files, headers=EDITOR).status_code == 200
        assert client.post(f"{API}/workspaces/{ws}/diagnose", headers=EDITOR).status_code == 200
        report = client.get(f"{API}/workspaces/{ws}/report.html")
    assert report.status_code == 200
    for markup in ("<script>alert", "<img src=x", "<iframe", "<svg onload"):
        assert markup not in report.text
    csp = report.headers["content-security-policy"]
    assert "sandbox" in csp
    assert "default-src 'none'" in csp
    assert report.headers["x-content-type-options"] == "nosniff"


def _demo_views() -> tuple[bytes, bytes, bytes, dict[str, Any]]:
    with open_client(new_app(service=new_service())) as client:
        ws = _demo(client)
        report = client.get(f"{API}/workspaces/{ws}/report").json()
        del report["timings_ms"]  # wall-clock stage durations: metadata, not a result
        return (
            client.get(f"{API}/workspaces/{ws}/graph").content,
            client.get(f"{API}/workspaces/{ws}/report.md").content,
            client.get(f"{API}/workspaces/{ws}/report.sarif").content,
            report,
        )


def test_same_inputs_give_byte_identical_responses_in_separate_apps() -> None:
    """P6 over HTTP: two independent apps (own store, same clock) answer with the same bytes —
    the graph view, both evidence exports — and the same report apart from its timings."""
    assert _demo_views() == _demo_views()


def test_signup_and_event_bodies_never_reach_the_access_log() -> None:
    signup = {"email": "kim.seoyeon@example.com", "consent_privacy": True, "message": "010-1234-5678"}
    with open_client(new_app()) as client, capture_logs() as logs:
        assert client.post(f"{API}/signups", json=signup).status_code == 201
        refused = client.post(f"{API}/signups", json={**signup, "consent_privacy": False})
        assert refused.status_code == 422
        event = {"name": "signup_click", "path": "/kim.seoyeon@example.com", "props": {"marketing": True}}
        assert client.post(f"{API}/events", json=event).status_code == 422  # not a screen path
        assert client.post(f"{API}/events", json={**event, "path": "/"}).status_code == 202
    flattened = repr(logs)
    assert "kim.seoyeon" not in flattened
    assert "1234-5678" not in flattened
    assert sum(entry["event"] == "http.request" for entry in logs) == 4


def test_a_viewer_token_cannot_claim_a_decider_role_with_dev_headers() -> None:
    spoofed = {
        "Authorization": f"Bearer {VIEWER_TOKEN}",
        "X-Jettae-Role": "lead",
        "X-Jettae-User": "kim.lead",
    }
    with open_client(new_app(auth_mode="token", api_tokens=TOKENS)) as client:
        lead = {"Authorization": f"Bearer {LEAD_TOKEN}"}
        ws = _demo(client, lead)
        patch = _validated(client, ws, lead)[0]
        denied = client.post(f"{API}/patches/{patch['id']}/approve", headers=spoofed)
        assert denied.status_code == 403
        assert "보기 전용(viewer)" in denied.json()["detail"]
        assert client.post(f"{API}/workspaces/{ws}/versions/1/revert", headers=spoofed).status_code == 403
        assert client.get(f"{API}/metrics/demand", headers=spoofed).status_code == 403
        approved = client.post(f"{API}/patches/{patch['id']}/approve", headers=lead)
        assert approved.status_code == 200
        assert approved.json()["patch"]["decided_by"] == "kim.lead"
