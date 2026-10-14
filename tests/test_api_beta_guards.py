"""Guards of the public beta over HTTP: what visitors may and may not make the server do.

* the server's LLM mode is a ceiling visitors cannot lift (demo, diagnosis);
* the demo button — and a visitor's diagnose of the demo workspace — opens the stored demo
  instead of re-diagnosing it (and costs no free scan);
* free scans: per-client daily share, failures caused by the input stay charged, server faults are
  given back, text and row caps, nothing structured kept, nothing left behind on failure;
* sign-ups cannot be overwritten (and say nothing about whether an address exists); an admin can
  delete one; the free-scan workspace name is reserved;
* the retention purge runs at start-up in public mode only, survives its own failures and stops
  with the app;
* audit entries name the HTTP caller.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import pytest
from fastapi import FastAPI
from structlog.testing import capture_logs

import jettae.api.app as app_module
from jettae.api.ratelimit import INVALID_CLIENT
from jettae.api.security import Identity, Role, llm_mode_for
from jettae.config import Settings
from jettae.domain import LlmMode
from jettae.ingest import load_bytes
from jettae.llm.base import LLMClient
from jettae.orchestrator import DEMO_WORKSPACE_NAME, FREE_SCAN_WORKSPACE_NAME, JettaeService
from jettae.orchestrator.service import ADMIN_AUDIT_SCOPE, AUDIT_SIGNUP_DELETED, email_fingerprint
from jettae.store import MemoryStore, StoredInputs
from test_api_support import (
    API,
    CLOCK,
    EDITOR,
    SMALL_ROSTER,
    SMALL_RUNBOOK,
    VIEWER,
    create_workspace,
    fixed_clock,
    new_app,
    new_service,
    open_client,
)
from test_api_support import settings as api_settings
from test_orchestrator_support import fake_llm

ADMIN_TOKEN = "admin-token-0123456789abcdef0123"
LEAD_TOKEN = "lead-token-0123456789abcdef01234"
TOKENS = {ADMIN_TOKEN: {"user": "heo", "role": "admin"}, LEAD_TOKEN: {"user": "kim.lead", "role": "lead"}}
ADMIN = {"Authorization": f"Bearer {ADMIN_TOKEN}"}
LEAD = {"Authorization": f"Bearer {LEAD_TOKEN}"}
OTHER_CLIENT = ("198.51.100.7", 40000)


def _scan(name: str = "결제.md", text: str = SMALL_RUNBOOK) -> list[tuple[str, tuple[str, bytes, str]]]:
    return [
        ("files", (name, text.encode("utf-8"), "text/markdown")),
        ("roster", ("roster.csv", SMALL_ROSTER.encode("utf-8"), "text/csv")),
    ]


def _public(service: JettaeService | None = None, **overrides: Any) -> FastAPI:
    return new_app(
        service=service if service is not None else new_service(),
        auth_mode="public",
        api_tokens=TOKENS,
        **overrides,
    )


@dataclass
class ModeRecorder:
    """An LLM factory that records the mode of every client it builds."""

    modes: list[LlmMode] = field(default_factory=list)

    def __call__(self, _settings: Settings, mode: LlmMode) -> LLMClient:
        self.modes.append(LlmMode(mode))
        return fake_llm()


# --------------------------------------------------------------------------- LLM mode ceiling


@pytest.mark.parametrize(
    ("role", "requested", "expected"),
    [
        (Role.VIEWER, LlmMode.LIVE, LlmMode.OFF),
        (Role.VIEWER, None, LlmMode.OFF),
        (Role.EDITOR, LlmMode.REPLAY, LlmMode.REPLAY),
        (Role.ADMIN, None, LlmMode.OFF),
    ],
)
def test_llm_mode_is_a_ceiling_below_editor(role: Role, requested: LlmMode | None, expected: LlmMode) -> None:
    assert llm_mode_for(Identity(user="u", role=role), requested, LlmMode.OFF) is expected


def test_visitors_cannot_switch_the_llm_on() -> None:
    recorder = ModeRecorder()
    service = JettaeService(
        MemoryStore(clock=fixed_clock), api_settings(), clock=fixed_clock, llm_factory=recorder
    )
    with open_client(_public(service)) as client:
        boot = client.post(f"{API}/demo/dmc-shop", json={"llm_mode": "live"})
        assert boot.status_code == 200, boot.text
        assert recorder.modes == [LlmMode.OFF]
        assert boot.json()["report"]["llm"]["mode"] == "off"
        ws = boot.json()["workspace_id"]
        # a token holder with an editor role or above may choose
        chosen = client.post(f"{API}/workspaces/{ws}/diagnose", json={"llm_mode": "replay"}, headers=ADMIN)
        assert chosen.status_code == 200, chosen.text
        assert recorder.modes[-1] is LlmMode.REPLAY
    with open_client(new_app(service=service)) as dev:
        team = create_workspace(dev)
        runbook = [("runbooks", ("결제.md", SMALL_RUNBOOK.encode("utf-8"), "text/markdown"))]
        assert dev.post(f"{API}/workspaces/{team}/inputs", files=runbook, headers=EDITOR).status_code == 200
        diagnosed = dev.post(f"{API}/workspaces/{team}/diagnose", json={"llm_mode": "live"}, headers=VIEWER)
        assert diagnosed.status_code == 200, diagnosed.text
        assert recorder.modes[-1:] == [LlmMode.OFF]
        assert diagnosed.json()["llm"]["mode"] == "off"


# --------------------------------------------------------------------------- demo


def test_visitors_open_the_stored_demo_without_a_new_diagnosis() -> None:
    service = new_service()
    stored = asyncio.run(service.bootstrap_demo())  # what `jettae demo` does at container start
    ws = stored.workspace_id
    entries = len(service.store.audit.entries(ws))
    # even with free scans switched off, the demo still opens
    app = _public(service, daily_free_scan_limit=0)
    with open_client(app) as client:
        for body in (None, {"llm_mode": "live"}):
            opened = client.post(f"{API}/demo/dmc-shop", json=body)
            assert opened.status_code == 200, opened.text
            payload = opened.json()
            assert payload["workspace_id"] == ws
            assert payload["workspace_name"] == DEMO_WORKSPACE_NAME
            assert payload["report"]["run_id"] == stored.run_id
        assert app.state.free_scans.used(CLOCK) == 0
        assert len(service.store.audit.entries(ws)) == entries  # nothing was written
        # editors and above still re-run it (unmetered)
        rerun = client.post(f"{API}/demo/dmc-shop", headers=ADMIN)
        assert rerun.status_code == 200, rerun.text
        assert rerun.json()["report"]["run_id"] != stored.run_id
        assert rerun.json()["workspace_id"] == ws


def test_visitors_diagnosing_the_demo_get_the_stored_report() -> None:
    """``POST /workspaces/{demo}/diagnose`` below editor: same as the demo button — no re-run."""
    recorder = ModeRecorder()
    service = JettaeService(
        MemoryStore(clock=fixed_clock), api_settings(), clock=fixed_clock, llm_factory=recorder
    )
    stored = asyncio.run(service.bootstrap_demo())
    ws = stored.workspace_id
    built = len(recorder.modes)
    entries = len(service.store.audit.entries(ws))
    app = _public(service, daily_free_scan_limit=0)  # nothing to spend
    with open_client(app) as client:
        for body in (None, {"llm_mode": "live"}):
            opened = client.post(f"{API}/workspaces/{ws}/diagnose", json=body)
            assert opened.status_code == 200, opened.text
            assert opened.json()["run_id"] == stored.run_id
            assert opened.json()["graph_hash"] == stored.graph_hash
        assert app.state.free_scans.used(CLOCK) == 0
        assert len(recorder.modes) == built  # no LLM client was even built
        assert len(service.store.audit.entries(ws)) == entries  # nothing written
        # editors and above still re-run it
        rerun = client.post(f"{API}/workspaces/{ws}/diagnose", headers=ADMIN)
        assert rerun.status_code == 200, rerun.text
        assert rerun.json()["run_id"] != stored.run_id
    with open_client(new_app(service=service)) as dev:
        latest = service.latest_report(ws)
        viewer = dev.post(f"{API}/workspaces/{ws}/diagnose", headers=VIEWER)
        assert viewer.json()["run_id"] == latest.run_id
        editor = dev.post(f"{API}/workspaces/{ws}/diagnose", headers=EDITOR)
        assert editor.json()["run_id"] != latest.run_id


def test_a_visitor_diagnose_of_an_undiagnosed_demo_runs_metered() -> None:
    service = new_service()
    ws = service._load_demo(DEMO_WORKSPACE_NAME)  # loaded, never diagnosed (a failed start-up run)
    app = _public(service, daily_free_scan_limit=5)
    with open_client(app) as client:
        first = client.post(f"{API}/workspaces/{ws}/diagnose")
        assert first.status_code == 200, first.text
        assert app.state.free_scans.used(CLOCK) == 1
        again = client.post(f"{API}/workspaces/{ws}/diagnose")
        assert again.json()["run_id"] == first.json()["run_id"]
        assert app.state.free_scans.used(CLOCK) == 1


def test_a_missing_demo_report_is_diagnosed_once_and_metered() -> None:
    app = _public(daily_free_scan_limit=5)
    with open_client(app) as client:
        first = client.post(f"{API}/demo/dmc-shop")
        assert first.status_code == 200, first.text
        assert app.state.free_scans.used(CLOCK) == 1
        again = client.post(f"{API}/demo/dmc-shop")
        assert again.json()["report"]["run_id"] == first.json()["report"]["run_id"]
        assert app.state.free_scans.used(CLOCK) == 1


# --------------------------------------------------------------------------- free-scan budget


def test_one_client_gets_its_share_of_the_daily_budget() -> None:
    app = _public(daily_free_scan_limit=10, free_scans_per_client_per_day=1)
    with open_client(app) as client, open_client(app, client=OTHER_CLIENT) as other:
        assert client.post(f"{API}/free-scans", files=_scan()).status_code == 201
        refused = client.post(f"{API}/free-scans", files=_scan())
        assert refused.status_code == 429
        assert refused.json()["code"] == "free_scan_limit"
        assert "접속 주소" in refused.json()["detail"]
        assert 0 < int(refused.headers["retry-after"]) <= 24 * 3600
        assert other.post(f"{API}/free-scans", files=_scan()).status_code == 201
        quota = app.state.free_scans
        assert quota.used(CLOCK) == 2
        assert quota.used_by(INVALID_CLIENT, CLOCK) == 1  # TestClient's peer "testclient" is no IP
        # the share also covers viewer diagnoses (the demo workspace here)
        demo = client.post(f"{API}/demo/dmc-shop")
        assert demo.status_code == 429
        assert "접속 주소" in demo.json()["detail"]


def test_failures_caused_by_the_input_stay_charged() -> None:
    service = new_service()
    app = _public(service, daily_free_scan_limit=2, max_document_chars=40)
    with open_client(app) as client:
        too_long = client.post(f"{API}/free-scans", files=_scan(text="# 런북\n\n" + "가" * 100))
        assert too_long.status_code == 422, too_long.text
        assert "문서 텍스트가 너무 깁니다" in too_long.json()["detail"]
        unreadable = client.post(f"{API}/free-scans", files=[("files", ("x.exe", b"MZ", "text/plain"))])
        assert unreadable.status_code == 422
        assert app.state.free_scans.used(CLOCK) == 2
        spent = client.post(f"{API}/free-scans", files=_scan(text="# 짧은 런북\n"))
        assert spent.status_code == 429
    assert service.list_workspaces() == []
    assert service.demand_metrics()["events"] == {}


def test_server_faults_give_the_scan_back(monkeypatch: pytest.MonkeyPatch) -> None:
    service = new_service()

    async def broken(**_kwargs: object) -> object:
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(service, "free_scan", broken)
    app = _public(service, daily_free_scan_limit=1)
    with open_client(app, raise_server_exceptions=False) as client:
        failed = client.post(f"{API}/free-scans", files=_scan())
        assert failed.status_code == 500
        assert app.state.free_scans.used(CLOCK) == 0
        assert app.state.free_scans.used_by("testclient", CLOCK) == 0
        assert app.state.scan_slots.in_use == 0


def test_the_total_text_and_the_rows_of_a_free_scan_are_capped() -> None:
    app = _public(max_document_chars=60)
    with open_client(app) as client:
        two = [
            ("files", ("a.md", ("# 가\n\n" + "나" * 30).encode("utf-8"), "text/markdown")),
            ("files", ("b.md", ("# 다\n\n" + "라" * 30).encode("utf-8"), "text/markdown")),
        ]
        total = client.post(f"{API}/free-scans", files=two)
        assert total.status_code == 422, total.text
        assert "모두 합쳐 너무 깁니다" in total.json()["detail"]
    header, row = SMALL_ROSTER.splitlines()[0], SMALL_ROSTER.splitlines()[1]
    rows = "\n".join([header, *(row.replace("김서연", f"사람{i:04d}") for i in range(1001))]) + "\n"
    with open_client(_public()) as client:
        big = client.post(
            f"{API}/free-scans",
            files=[
                ("files", ("a.md", SMALL_RUNBOOK.encode("utf-8"), "text/markdown")),
                ("roster", ("roster.csv", rows.encode("utf-8"), "text/csv")),
            ],
        )
        assert big.status_code == 422, big.text
        assert "1,000행까지" in big.json()["detail"]


def test_a_free_scan_keeps_its_result_but_not_its_structured_inputs() -> None:
    service = new_service()
    with open_client(_public(service)) as client:
        body = client.post(f"{API}/free-scans", files=_scan()).json()
        ws, key = body["workspace_id"], body["access_key"]
        assert service.store.inputs.load_inputs(ws) == StoredInputs()
        with_key = {"X-Jettae-Workspace-Key": key}
        inputs = client.get(f"{API}/workspaces/{ws}/inputs", headers=with_key).json()
        assert (inputs["roster"], inputs["oncall"], inputs["catalog"]) == (0, 0, 0)
        for path in ("/report", "/graph", "/report.md", "/patches"):
            assert client.get(f"{API}/workspaces/{ws}{path}", headers=with_key).status_code == 200, path
        actors = {e.actor for e in service.store.audit.entries(ws)}
        assert actors == {"anonymous"}


# --------------------------------------------------------------------------- workspaces


def test_the_free_scan_workspace_name_is_reserved() -> None:
    service = new_service()
    with open_client(new_app(service=service)) as client:
        for name in (FREE_SCAN_WORKSPACE_NAME, f" {FREE_SCAN_WORKSPACE_NAME}\u200b"):
            refused = client.post(f"{API}/workspaces", json={"name": name}, headers=EDITOR)
            assert refused.status_code == 422
            assert "무료 검진용으로 예약된 이름" in refused.json()["detail"]
    assert service.list_workspaces() == []


def test_audit_entries_name_the_http_caller() -> None:
    service = new_service()
    with open_client(new_app(service=service)) as client:
        ws = create_workspace(client)
        upload = client.post(
            f"{API}/workspaces/{ws}/inputs",
            files=[("runbooks", ("결제.md", SMALL_RUNBOOK.encode("utf-8"), "text/markdown"))],
            headers=EDITOR,
        )
        assert upload.status_code == 200, upload.text
        assert client.post(f"{API}/workspaces/{ws}/diagnose", headers=EDITOR).status_code == 200
        drills = client.post(
            f"{API}/workspaces/{ws}/rehearsals", json={"generate": "template"}, headers=EDITOR
        )
        assert drills.status_code == 200, drills.text
    actors = {(e.action, e.actor) for e in service.store.audit.entries(ws)}
    for action in ("workspace.created", "inputs.uploaded", "diagnosis.completed", "rehearsal.run"):
        assert (action, "demo.editor") in actors, action
    assert all(actor == "demo.editor" for _action, actor in actors)


def test_a_visitor_demo_run_is_audited_as_anonymous() -> None:
    service = new_service()
    with open_client(_public(service)) as client:
        ws = client.post(f"{API}/demo/dmc-shop").json()["workspace_id"]
    assert {e.actor for e in service.store.audit.entries(ws)} == {"anonymous"}


# --------------------------------------------------------------------------- sign-ups


SIGNUP = {
    "email": "Kim@Example.com",
    "organization": "별빛상점",
    "team_size": "11~50명",
    "consent_privacy": True,
    "consent_marketing": False,
}


def test_a_signup_cannot_be_overwritten_and_the_answer_reveals_nothing() -> None:
    service = new_service()
    with open_client(_public(service)) as client:
        first = client.post(f"{API}/signups", json=SIGNUP)
        stored = service.store.signups.get_signup("kim@example.com")
        forged = client.post(
            f"{API}/signups", json={**SIGNUP, "consent_marketing": True, "organization": "가짜"}
        )
    assert first.status_code == forged.status_code == 201
    assert first.json() == forged.json() == {"accepted": True}
    assert service.store.signups.get_signup("kim@example.com") == stored
    assert stored is not None
    assert stored.consent_marketing is False


def test_admins_delete_a_signup_and_only_a_fingerprint_is_audited() -> None:
    service = new_service()
    with open_client(_public(service)) as client:
        assert client.post(f"{API}/signups", json=SIGNUP).status_code == 201
        for headers in ({}, LEAD):
            refused = client.delete(f"{API}/signups/kim@example.com", headers=headers)
            assert refused.status_code == 403
        assert service.demand_metrics()["signups"] == 1
        deleted = client.delete(f"{API}/signups/KIM@example.com", headers=ADMIN)
        assert deleted.status_code == 200, deleted.text
        assert deleted.json() == {"deleted": True}
        again = client.delete(f"{API}/signups/kim@example.com", headers=ADMIN)
        assert again.json() == {"deleted": False}
        malformed = client.delete(f"{API}/signups/not-an-address", headers=ADMIN)
        assert malformed.json() == {"deleted": False}
    assert service.demand_metrics()["signups"] == 0
    entries = [e for e in service.store.audit.entries(ADMIN_AUDIT_SCOPE) if e.action == AUDIT_SIGNUP_DELETED]
    assert [(e.actor, e.payload["deleted"]) for e in entries] == [
        ("heo", True),
        ("heo", False),
        ("heo", False),
    ]
    assert entries[0].payload["email_sha256"] == email_fingerprint("kim@example.com")
    assert not re.search(r"kim@", repr(service.store.audit.entries(ADMIN_AUDIT_SCOPE)), re.IGNORECASE)


def test_deleting_a_signup_needs_a_token_in_token_mode() -> None:
    with open_client(new_app(auth_mode="token", api_tokens=TOKENS)) as client:
        assert client.delete(f"{API}/signups/kim@example.com").status_code == 401
        assert client.delete(f"{API}/signups/kim@example.com", headers=ADMIN).json() == {"deleted": False}


# --------------------------------------------------------------------------- retention in the app


def _expired_free_scan(service: JettaeService) -> str:
    """A free-scan workspace created long before the service's clock (the store's own clock)."""
    old = service.store.workspaces.create_workspace(FREE_SCAN_WORKSPACE_NAME)
    return old.id


def test_the_app_purges_expired_free_scans_at_start_up() -> None:
    store = MemoryStore(clock=lambda: CLOCK - timedelta(days=30))
    service = JettaeService(store, api_settings(), clock=fixed_clock)
    old = _expired_free_scan(service)
    kept = service.create_workspace("결제 플랫폼팀").id
    with open_client(_public(service)):
        assert [w.id for w in service.list_workspaces()] == [kept]
    assert old not in {w.id for w in service.list_workspaces()}


@pytest.mark.parametrize("mode", ["dev", "token"])
def test_only_the_public_site_purges_on_its_own(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's database or a team's own install keeps everything (the retention periods are
    the public site's privacy promise) — no purge at start-up, none periodically."""
    store = MemoryStore(clock=lambda: CLOCK - timedelta(days=30))
    service = JettaeService(store, api_settings(), clock=fixed_clock)
    old = _expired_free_scan(service)
    calls: list[int] = []
    real = service.purge_expired

    def counting(*args: Any) -> Any:
        calls.append(1)
        return real(*args)

    monkeypatch.setattr(service, "purge_expired", counting)
    monkeypatch.setattr(app_module, "PURGE_INTERVAL_SECONDS", 0.01)
    tokens = TOKENS if mode == "token" else {}
    with open_client(new_app(service=service, auth_mode=mode, api_tokens=tokens)) as client:
        for _ in range(20):
            client.get("/healthz")
        time.sleep(0.05)
    assert calls == []
    assert old in {w.id for w in service.list_workspaces()}


def test_a_failing_purge_never_stops_the_app(monkeypatch: pytest.MonkeyPatch) -> None:
    service = new_service()

    def broken(*_args: object) -> object:
        raise RuntimeError("could not reach kim@example.com's row")

    monkeypatch.setattr(service, "purge_expired", broken)
    with capture_logs() as logs, open_client(_public(service)) as client:
        assert client.get("/healthz").status_code == 200
    failed = [entry for entry in logs if entry["event"] == "retention.purge_failed"]
    assert failed == [{"event": "retention.purge_failed", "error": "RuntimeError", "log_level": "warning"}]
    assert "kim@" not in repr(logs)


def test_the_periodic_purge_runs_and_stops_with_the_app(monkeypatch: pytest.MonkeyPatch) -> None:
    service = new_service()
    calls: list[int] = []
    real = service.purge_expired

    def counting(*args: Any) -> Any:
        calls.append(1)
        return real(*args)

    monkeypatch.setattr(service, "purge_expired", counting)
    monkeypatch.setattr(app_module, "PURGE_INTERVAL_SECONDS", 0.01)
    with open_client(_public(service)) as client:
        for _ in range(500):
            if len(calls) >= 3:
                break
            client.get("/healthz")
        assert len(calls) >= 3  # once at start-up, then on every tick
    stopped = len(calls)
    time.sleep(0.05)
    assert len(calls) == stopped  # cancelled with the app


def test_purged_workspaces_disappear_from_a_running_app() -> None:
    store = MemoryStore(clock=fixed_clock)
    service = JettaeService(store, api_settings(), clock=lambda: CLOCK + timedelta(days=8))
    with open_client(new_app(service=service)) as client:
        documents = [load_bytes("결제.md", SMALL_RUNBOOK.encode("utf-8"))]
        ws = asyncio.run(service.free_scan(documents=documents, actor="anonymous")).workspace_id
        assert client.get(f"{API}/workspaces/{ws}").status_code == 200
        assert service.purge_expired().workspaces == 1
        assert client.get(f"{API}/workspaces/{ws}").status_code == 404
