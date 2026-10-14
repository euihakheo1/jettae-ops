"""Sign-ups (consent required, never overwritten) and demand events (allow-list of names, each
with a few enumerated or bounded props and a screen path from a fixed list, at most
``JETTAE_EVENTS_PER_DAY`` stored per Korean day) over HTTP."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError
from structlog.testing import capture_logs

from jettae.api.ratelimit import DailyCap
from jettae.api.schemas import EVENT_NAMES, EVENT_PROPS, EVENT_SCREEN_PATHS, TEAM_SIZES, EventRequest
from jettae.orchestrator import EVENT_RUNBOOK_UPLOAD, JettaeService
from test_api_support import (
    API,
    CLOCK,
    EDITOR,
    LEAD,
    ROOT,
    SMALL_RUNBOOK,
    VIEWER,
    new_app,
    new_service,
    open_client,
)


@dataclass
class Growth:
    client: TestClient
    service: JettaeService


@pytest.fixture
def growth() -> Iterator[Growth]:
    service = new_service()
    with open_client(new_app(service=service)) as client:
        yield Growth(client=client, service=service)


# --------------------------------------------------------------------------- sign-ups


def test_signup_without_consent_is_refused_and_not_stored(growth: Growth) -> None:
    response = growth.client.post(
        f"{API}/signups", json={"email": "kim@example.com", "consent_privacy": False}
    )
    assert response.status_code == 422
    assert response.json()["code"] == "consent_required"
    assert "동의" in response.json()["detail"]
    assert growth.service.demand_metrics()["signups"] == 0
    missing = growth.client.post(f"{API}/signups", json={"email": "kim@example.com"})
    assert missing.status_code == 422
    assert missing.json()["errors"][0] == {
        "field": "consent_privacy",
        "message": "필수 값이 빠졌습니다.",
        "type": "missing",
    }


def test_signup_is_stored_with_the_web_field_names_and_never_overwritten(growth: Growth) -> None:
    body = {
        "email": "Kim@Example.com",
        "org_name": "별빛상점",
        "team_size": "10-49",
        "message": "",
        "consent_privacy": True,
        "consent_marketing": True,
        "source": "landing",
    }
    first = growth.client.post(f"{API}/signups", json=body)
    assert first.status_code == 201
    assert first.json() == {"accepted": True}
    stored = growth.service.store.signups.get_signup("kim@example.com")
    assert stored is not None
    assert stored.organization == "별빛상점"
    assert stored.message is None
    assert stored.consent_marketing is True
    again = growth.client.post(
        f"{API}/signups", json={**body, "consent_marketing": False, "organization": "새 이름"}
    )
    # answered exactly like a new sign-up (no enumeration), but the first record stays as it was
    assert again.status_code == 201
    assert again.json() == first.json()
    assert growth.service.demand_metrics()["signups"] == 1
    assert growth.service.store.signups.get_signup("kim@example.com") == stored


@pytest.mark.parametrize(
    ("body", "field", "fragment"),
    [
        ({"email": "not-an-email", "consent_privacy": True}, "email", "이메일 주소 형식"),
        ({"email": "a@b.co", "consent_privacy": True, "message": "x" * 2001}, "message", "2000자 이하"),
        ({"email": "a@b.co", "consent_privacy": "maybe"}, "consent_privacy", "true 또는 false"),
    ],
)
def test_signup_validation_errors_are_korean(
    growth: Growth, body: dict[str, object], field: str, fragment: str
) -> None:
    response = growth.client.post(f"{API}/signups", json=body)
    assert response.status_code == 422
    payload = response.json()
    assert payload["code"] == "validation_error"
    assert payload["errors"][0]["field"] == field
    assert fragment in payload["detail"]
    assert payload["detail"].startswith(f"{field}: ")


# --------------------------------------------------------------------------- events


#: one valid props value per allowed key, as the web app sends them
VALID_PROPS: dict[str, dict[str, object]] = {
    "landing_view": {"referrer": "external"},
    "landing_first_view": {"referrer": "direct"},
    "upload_click": {"files": 3},
    "signup_click": {"team_size": "11~50명", "marketing": True},
    "demo_start": {"from": "workspace_home"},
    "report_download": {"kind": "sarif"},
}


def test_every_event_has_its_props_rules() -> None:
    assert set(EVENT_PROPS) == set(EVENT_NAMES) == set(VALID_PROPS)
    for name, props in VALID_PROPS.items():
        assert set(props) == set(EVENT_PROPS[name]), name


def test_team_sizes_match_the_signup_form() -> None:
    form = (ROOT / "web" / "src" / "pages" / "landing" / "SignupForm.tsx").read_text(encoding="utf-8")
    match = re.search(r"const TEAM_SIZES = \[(.*?)\];", form)
    assert match is not None
    assert tuple(re.findall(r"'([^']*)'", match.group(1))) == TEAM_SIZES


def test_allowed_events_are_counted(growth: Growth) -> None:
    for name in EVENT_NAMES:
        response = growth.client.post(
            f"{API}/events", json={"name": name, "path": "/", "props": VALID_PROPS[name]}
        )
        assert response.status_code == 202, response.text
        bare = growth.client.post(f"{API}/events", json={"name": name})  # every prop is optional
        assert bare.status_code == 202, bare.text
    assert growth.service.demand_metrics()["events"] == dict.fromkeys(sorted(EVENT_NAMES), 2)


def test_unknown_event_names_are_refused_with_the_allow_list(growth: Growth) -> None:
    response = growth.client.post(f"{API}/events", json={"name": "page_scroll"})
    assert response.status_code == 422
    assert "landing_view" in response.json()["detail"]
    assert growth.service.demand_metrics()["events"] == {}


def test_landing_first_view_is_allowed_and_runbook_upload_is_server_only(growth: Growth) -> None:
    """``runbook_upload`` counts free scans; a client that could send it could inflate the count."""
    assert "landing_first_view" in EVENT_NAMES
    assert EVENT_RUNBOOK_UPLOAD not in EVENT_NAMES
    first = growth.client.post(f"{API}/events", json={"name": "landing_first_view", "path": "/"})
    assert first.status_code == 202
    forged = growth.client.post(f"{API}/events", json={"name": EVENT_RUNBOOK_UPLOAD, "props": {"files": 3}})
    assert forged.status_code == 422
    assert "landing_first_view" in forged.json()["detail"]
    assert growth.service.demand_metrics()["events"] == {"landing_first_view": 1}


def test_paths_are_screen_routes_without_query_or_fragment(growth: Growth) -> None:
    body = {
        "name": "signup_click",
        "path": "/app/report/?email=kim@example.com#top",
        "props": {"marketing": False},
    }
    stored = EventRequest.model_validate(body).stored_props()
    assert stored == {"marketing": False, "path": "/app/report"}
    assert growth.client.post(f"{API}/events", json=body).status_code == 202
    for screen in EVENT_SCREEN_PATHS:
        assert EventRequest.model_validate({"name": "landing_view", "path": screen}).path == screen


@pytest.mark.parametrize(
    "path",
    [
        "/app/w/ws-1",  # a deep link names a workspace
        "/9001011234567/홍길동/4111111111111111",  # free text: names and numbers
        "/u/010-1234-5678",
        "/privacyx",
        "/app//report",
    ],
)
def test_paths_outside_the_screen_list_are_refused(growth: Growth, path: str) -> None:
    response = growth.client.post(f"{API}/events", json={"name": "landing_view", "path": path})
    assert response.status_code == 422, response.text
    assert response.json()["errors"][0]["field"] == "path"
    assert "/app/report" in response.json()["detail"]
    assert growth.service.demand_metrics()["events"] == {}


def test_the_web_app_sends_only_listed_screen_paths() -> None:
    """Every ``path`` literal the web app passes to ``api.track`` is on the server's list."""
    sent: set[str] = set()
    for source in (ROOT / "web" / "src").rglob("*.ts*"):
        if ".test." in source.name:
            continue
        text = source.read_text(encoding="utf-8")
        sent.update(re.findall(r"api\.track\(\{[^}]*?path: '([^']*)'", text))
    assert sent, "no api.track call found"
    assert sent <= set(EVENT_SCREEN_PATHS)


@pytest.mark.parametrize(
    ("name", "props", "fragment"),
    [
        # keys another event allows, or nobody allows
        ("landing_view", {"from": "landing"}, "쓸 수 없는 항목"),
        ("demo_start", {"note": "연락처 010-1234-5678"}, "쓸 수 없는 항목"),
        ("upload_click", {"01012345678": 1}, "쓸 수 없는 항목"),
        ("report_download", {"kind": "pdf"}, "md, html, sarif 중 하나"),
        ("landing_first_view", {"referrer": "https://evil.example/"}, "direct, external 중 하나"),
        ("upload_click", {"files": 101}, "0~100 사이의 정수"),
        ("upload_click", {"files": -1}, "0~100 사이의 정수"),
        ("upload_click", {"files": True}, "0~100 사이의 정수"),
        ("upload_click", {"files": "3"}, "0~100 사이의 정수"),
        ("signup_click", {"team_size": "9001101234567"}, "중 하나여야"),
        ("signup_click", {"marketing": 1}, "true 또는 false"),
        ("demo_start", {"from": "landing ", "extra": 1}, "쓸 수 없는 항목"),
    ],
)
def test_props_outside_the_allow_list_are_refused(
    growth: Growth, name: str, props: dict[str, object], fragment: str
) -> None:
    response = growth.client.post(f"{API}/events", json={"name": name, "props": props})
    assert response.status_code == 422, response.text
    payload = response.json()
    assert payload["code"] == "validation_error"
    assert payload["errors"][0]["field"] == "props"
    assert fragment in payload["detail"]
    assert growth.service.demand_metrics()["events"] == {}


@pytest.mark.parametrize("props", [{"nested": {"a": 1}}, {"files": 1.5}, {"files": None}])
def test_odd_prop_values_are_refused(props: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        EventRequest.model_validate({"name": "upload_click", "props": props})


@pytest.mark.parametrize("value", [1.5, 3.0, None, [3], {"n": 3}])
def test_a_wrong_prop_type_gets_the_rule_message(growth: Growth, value: object) -> None:
    """Not one line per member of the str/bool/int union ('props.files.str: 문자열이어야 합니다')."""
    response = growth.client.post(f"{API}/events", json={"name": "upload_click", "props": {"files": value}})
    assert response.status_code == 422
    payload = response.json()
    assert payload["errors"] == [
        {
            "field": "props",
            "message": "'upload_click' 이벤트의 props.files는 0~100 사이의 정수여야 합니다.",
            "type": "value_error",
        }
    ]
    assert ".str" not in payload["detail"]


@given(st.sampled_from(sorted(VALID_PROPS)), st.data())
def test_valid_props_are_kept_as_sent(name: str, data: st.DataObject) -> None:
    keys = data.draw(st.sets(st.sampled_from(sorted(VALID_PROPS[name]))))
    props = {key: VALID_PROPS[name][key] for key in keys}
    assert EventRequest.model_validate({"name": name, "props": props}).stored_props() == props


def test_an_explicit_null_path_is_kept_absent() -> None:
    assert EventRequest.model_validate({"name": "landing_view", "path": None}).stored_props() == {}


def test_relative_paths_are_refused() -> None:
    with pytest.raises(ValidationError, match="'/'로 시작"):
        EventRequest.model_validate({"name": "landing_view", "path": "https://evil.example/"})


# --------------------------------------------------------------------------- daily cap


def test_events_past_the_daily_cap_are_accepted_but_not_stored() -> None:
    service = new_service()
    app = new_app(service=service, events_per_day=3)
    with capture_logs() as logs, open_client(app) as client:
        responses = [client.post(f"{API}/events", json={"name": "landing_view"}) for _ in range(5)]
    assert [r.status_code for r in responses] == [202] * 5
    assert {r.text for r in responses} == {'{"accepted":true}'}
    assert service.demand_metrics()["events"] == {"landing_view": 3}
    warnings = [entry for entry in logs if entry["event"] == "daily_cap.reached"]
    assert len(warnings) == 1  # once per day, not once per dropped event
    assert warnings[0]["limit"] == 3
    assert app.state.event_cap.used(CLOCK) == 3


def test_invalid_events_do_not_use_the_daily_cap() -> None:
    app = new_app(events_per_day=1)
    with open_client(app) as client:
        assert client.post(f"{API}/events", json={"name": "page_scroll"}).status_code == 422
        assert client.post(f"{API}/events", json={"name": "landing_view"}).status_code == 202
    assert app.state.event_cap.used(CLOCK) == 1


def test_free_scans_are_counted_whatever_the_event_cap() -> None:
    """The cap guards what clients send; the server's own ``runbook_upload`` is not capped."""
    service = new_service()
    files = [("files", ("결제.md", SMALL_RUNBOOK.encode("utf-8"), "text/markdown"))]
    with open_client(new_app(service=service, events_per_day=0)) as client:
        assert client.post(f"{API}/events", json={"name": "landing_view"}).status_code == 202
        assert client.post(f"{API}/free-scans", files=files).status_code == 201
    assert service.demand_metrics()["events"] == {EVENT_RUNBOOK_UPLOAD: 1}


def test_the_daily_cap_restarts_at_midnight_korean_time() -> None:
    cap = DailyCap(2, subject="events")
    before_midnight = datetime.fromisoformat("2026-10-28T14:59:59+00:00")  # 23:59:59 KST
    assert [cap.take(before_midnight) for _ in range(3)] == [True, True, False]
    assert cap.take(before_midnight + timedelta(seconds=1)) is True  # 00:00 KST: a new day
    assert cap.used(before_midnight + timedelta(seconds=1)) == 1
    assert cap.used(before_midnight) == 0
    with pytest.raises(ValueError, match="0 이상"):
        DailyCap(-1, subject="events")


# --------------------------------------------------------------------------- metrics


def test_demand_metrics_are_for_leads_and_admins(growth: Growth) -> None:
    growth.client.post(f"{API}/events", json={"name": "demo_start", "props": {"from": "landing"}})
    for headers in ({}, VIEWER, EDITOR):
        assert growth.client.get(f"{API}/metrics/demand", headers=headers).status_code == 403
    metrics = growth.client.get(f"{API}/metrics/demand", headers=LEAD)
    assert metrics.status_code == 200
    assert metrics.json() == {"signups": 0, "events": {"demo_start": 1}}
