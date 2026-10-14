"""Load shedding for the public site: per-client rate limits and the scan slots.

The limiter is tested with an injected clock (no sleeping); the HTTP tests check that the routes
use it — 429 ``rate_limited`` / 503 ``server_busy`` with ``Retry-After`` and a Korean message.
"""

from __future__ import annotations

import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jettae.api.errors import ApiProblem
from jettae.api.ratelimit import (
    FREE_SCAN_ATTEMPTS_PER_MINUTE,
    UNKNOWN_CLIENT,
    RateLimiter,
    ScanSlots,
)
from test_api_support import (
    API,
    EDITOR,
    SMALL_RUNBOOK,
    VIEWER,
    create_workspace,
    new_app,
    new_service,
    open_client,
)

TEST_CLIENT = "testclient"  # what starlette's TestClient reports as the client host


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _limiter(limit: int, window: float, *, max_keys: int = 100) -> tuple[RateLimiter, FakeClock]:
    clock = FakeClock()
    return RateLimiter(limit, window, subject="테스트", max_keys=max_keys, clock=clock), clock


# --------------------------------------------------------------------------- RateLimiter


def test_limiter_allows_the_limit_then_refuses_until_the_window_ends() -> None:
    limiter, clock = _limiter(3, 60)
    assert [limiter.retry_after("a") for _ in range(3)] == [None, None, None]
    clock.now += 15
    assert limiter.retry_after("a") == 45
    clock.now += 44.5
    assert limiter.retry_after("a") == 1  # rounded up, never 0
    clock.now += 0.5
    assert limiter.retry_after("a") is None  # a fresh window
    assert limiter.retry_after("b") is None  # keys are independent


def test_refused_requests_do_not_extend_the_window() -> None:
    limiter, clock = _limiter(1, 10)
    assert limiter.retry_after("a") is None
    for _ in range(5):
        clock.now += 1
        assert limiter.retry_after("a") is not None
    clock.now += 5
    assert limiter.retry_after("a") is None


def test_check_raises_a_korean_429_with_retry_after() -> None:
    limiter, clock = _limiter(1, 3600)
    limiter.check("a")
    clock.now += 60
    with pytest.raises(ApiProblem) as refused:
        limiter.check("a")
    problem = refused.value
    assert problem.status_code == 429
    assert problem.code == "rate_limited"
    assert problem.headers == {"Retry-After": "3540"}
    assert problem.detail == "같은 곳에서 테스트 요청이 너무 잦습니다. 59분 뒤에 다시 시도하세요."


def test_expired_windows_are_dropped_and_memory_stays_bounded() -> None:
    limiter, clock = _limiter(1, 10, max_keys=5)
    for index in range(5):
        assert limiter.retry_after(f"k{index}") is None
    assert len(limiter) == 5
    # a sixth client evicts the oldest window instead of growing the table
    assert limiter.retry_after("k5") is None
    assert len(limiter) == 5
    assert limiter.retry_after("k0") is None  # evicted: starts afresh
    clock.now += 10
    assert limiter.retry_after("late") is None
    assert len(limiter) == 1  # every expired window went at once


@settings(max_examples=50)
@given(
    st.integers(min_value=1, max_value=5),
    st.lists(st.tuples(st.sampled_from("abc"), st.floats(min_value=0, max_value=30)), max_size=40),
)
def test_no_key_ever_gets_more_than_the_limit_per_window(
    limit: int, requests: list[tuple[str, float]]
) -> None:
    window = 10.0
    limiter, clock = _limiter(limit, window)
    allowed: dict[str, list[float]] = {}
    for key, step in requests:
        clock.now += step
        if limiter.retry_after(key) is None:
            allowed.setdefault(key, []).append(clock.now)
    for times in allowed.values():
        for start in times:
            assert sum(1 for t in times if start <= t < start + window) <= limit


@pytest.mark.parametrize(("limit", "window", "keys"), [(0, 60, 10), (1, 0, 10), (1, 60, 0)])
def test_limiter_refuses_nonsense_settings(limit: int, window: float, keys: int) -> None:
    with pytest.raises(ValueError, match="1 이상"):
        RateLimiter(limit, window, subject="x", max_keys=keys)


# --------------------------------------------------------------------------- ScanSlots


def test_scan_slots_refuse_at_once_when_full_and_free_on_exit() -> None:
    slots = ScanSlots(2)
    with slots.hold(), slots.hold():
        assert slots.in_use == 2
        with pytest.raises(ApiProblem) as busy, slots.hold():
            pass
        assert busy.value.status_code == 503
        assert busy.value.code == "server_busy"
        assert busy.value.headers == {"Retry-After": "30"}
        assert "다른 진단이 진행 중" in str(busy.value.detail)
    assert slots.in_use == 0
    with pytest.raises(RuntimeError), slots.hold():
        raise RuntimeError("boom")
    assert slots.in_use == 0


def test_scan_slots_need_at_least_one() -> None:
    with pytest.raises(ValueError, match="JETTAE_MAX_CONCURRENT_SCANS"):
        ScanSlots(0)


# --------------------------------------------------------------------------- HTTP


def _assert_rate_limited(response_json: dict[str, object], headers: dict[str, str]) -> None:
    assert response_json["code"] == "rate_limited"
    assert re.search(r"[가-힣]", str(response_json["detail"]))
    assert int(headers["retry-after"]) >= 1


def test_events_are_limited_per_client_per_minute() -> None:
    service = new_service()
    with open_client(new_app(service=service, events_per_client_per_minute=3)) as client:
        for _ in range(3):
            assert client.post(f"{API}/events", json={"name": "landing_view"}).status_code == 202
        refused = client.post(f"{API}/events", json={"name": "landing_view"})
        assert refused.status_code == 429
        _assert_rate_limited(refused.json(), dict(refused.headers))
        assert int(refused.headers["retry-after"]) <= 60
        assert "이벤트 기록" in refused.json()["detail"]
        # the limit is checked before anything is stored, invalid bodies included
        assert client.post(f"{API}/events", json={"name": "nope"}).status_code == 429
    assert service.demand_metrics()["events"] == {"landing_view": 3}


def test_signups_are_limited_per_client_per_hour() -> None:
    service = new_service()
    with open_client(new_app(service=service, signups_per_client_per_hour=2)) as client:
        for index in range(2):
            body = {"email": f"p{index}@example.com", "consent_privacy": True}
            assert client.post(f"{API}/signups", json=body).status_code == 201
        refused = client.post(f"{API}/signups", json={"email": "x@example.com", "consent_privacy": True})
        assert refused.status_code == 429
        _assert_rate_limited(refused.json(), dict(refused.headers))
        assert 60 < int(refused.headers["retry-after"]) <= 3600
        assert "사전 신청" in refused.json()["detail"]
    assert service.demand_metrics()["signups"] == 2


def test_limits_are_per_client_address() -> None:
    app = new_app(events_per_client_per_minute=1)
    with open_client(app) as first, open_client(app, client=("203.0.113.9", 50000)) as second:
        assert first.post(f"{API}/events", json={"name": "landing_view"}).status_code == 202
        assert first.post(f"{API}/events", json={"name": "landing_view"}).status_code == 429
        assert second.post(f"{API}/events", json={"name": "landing_view"}).status_code == 202
    assert len(app.state.event_limiter) == 2


def test_free_scan_attempts_are_limited_for_visitors_only() -> None:
    files = [("files", ("런북.exe", b"MZ", "application/octet-stream"))]
    with open_client(new_app()) as client:
        for _ in range(FREE_SCAN_ATTEMPTS_PER_MINUTE):
            assert client.post(f"{API}/free-scans", files=files).status_code == 422
        refused = client.post(f"{API}/free-scans", files=files)
        assert refused.status_code == 429
        _assert_rate_limited(refused.json(), dict(refused.headers))
        # editors are not limited (here: the same address with an editor role)
        assert client.post(f"{API}/free-scans", files=files, headers=EDITOR).status_code == 422


def test_busy_scan_slots_answer_503_to_visitors_and_charge_nothing() -> None:
    app = new_app(max_concurrent_scans=1)
    scan = [("files", ("결제.md", SMALL_RUNBOOK.encode("utf-8"), "text/markdown"))]
    with open_client(app) as client:
        ws = create_workspace(client)
        slots: ScanSlots = app.state.scan_slots
        assert slots.capacity == 1
        with slots.hold():
            for response in (
                client.post(f"{API}/free-scans", files=scan),
                client.post(f"{API}/workspaces/{ws}/diagnose", headers=VIEWER),
                client.post(f"{API}/demo/dmc-shop"),  # no demo report yet: it would have to run
            ):
                assert response.status_code == 503, response.text
                assert response.json()["code"] == "server_busy"
                assert response.headers["retry-after"] == "30"
            # editors and above are not throttled
            assert client.post(f"{API}/free-scans", files=scan, headers=EDITOR).status_code == 201
        assert app.state.free_scans.used(new_service().clock()) == 0
        assert slots.in_use == 0
        assert client.post(f"{API}/free-scans", files=scan).status_code == 201


def test_requests_without_a_client_address_share_one_key() -> None:
    app = new_app(events_per_client_per_minute=1)
    with open_client(app, client=None) as client:
        assert client.post(f"{API}/events", json={"name": "landing_view"}).status_code == 202
        assert client.post(f"{API}/events", json={"name": "landing_view"}).status_code == 429
    assert app.state.event_limiter.retry_after(UNKNOWN_CLIENT) is not None
    assert app.state.event_limiter.retry_after(TEST_CLIENT) is None
