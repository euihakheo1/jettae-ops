"""Which address the per-client limits count under (``JETTAE_CLIENT_IP_SOURCE``).

* the setting's three forms, refused at start-up when malformed (Korean message, CLI and app);
* keys are parsed addresses (IPv6 by /64, ``invalid``/``unknown`` otherwise), never header text;
* ``header:<name>`` and ``xff:<n>`` resist a forged leftmost ``X-Forwarded-For``; a missing
  header falls back to the rightmost forwarded entry (``xff:<n>`` with too few entries: the peer);
* the admin-only ``GET /admin/request-info`` shows what arrived and the key it gave (no-store).
"""

from __future__ import annotations

import ipaddress
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError
from starlette.datastructures import Headers

from jettae.api import create_app
from jettae.api.ratelimit import (
    INVALID_CLIENT,
    UNKNOWN_CLIENT,
    ClientIpResolver,
    address_key,
    forwarded_entries,
    parse_ip,
)
from jettae.config import MAX_XFF_HOPS, ClientIpSource, parse_client_ip_source
from test_api_support import ADMIN, API, EDITOR, SMALL_RUNBOOK, VIEWER, new_app, open_client, settings

ADMIN_TOKEN = "admin-token-0123456789abcdef0123"
TOKENS = {ADMIN_TOKEN: {"user": "heo", "role": "admin"}}
BEARER = {"Authorization": f"Bearer {ADMIN_TOKEN}"}
VISITOR = ("203.0.113.9", 50000)
PROXY = ("10.214.3.7", 40000)


def _headers(*pairs: tuple[str, str]) -> Headers:
    return Headers(raw=[(name.lower().encode("latin-1"), value.encode("latin-1")) for name, value in pairs])


def _resolve(source: str, peer: str | None, *pairs: tuple[str, str]) -> str:
    return ClientIpResolver.from_setting(source).resolve(peer, _headers(*pairs))


# --------------------------------------------------------------------------- the setting


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("peer", ClientIpSource(kind="peer")),
        (" PEER ", ClientIpSource(kind="peer")),
        ("header:CF-Connecting-IP", ClientIpSource(kind="header", header="CF-Connecting-IP")),
        ("Header: True-Client-IP", ClientIpSource(kind="header", header="True-Client-IP")),
        ("xff:1", ClientIpSource(kind="xff", hops=1)),
        ("XFF:2", ClientIpSource(kind="xff", hops=2)),
        (f"xff:{MAX_XFF_HOPS}", ClientIpSource(kind="xff", hops=MAX_XFF_HOPS)),
    ],
)
def test_the_three_forms_parse(value: str, expected: ClientIpSource) -> None:
    assert parse_client_ip_source(value) == expected
    assert parse_client_ip_source(str(expected)) == expected  # str() is the normalised setting


@pytest.mark.parametrize(
    "value",
    [
        "",
        "peer:1",
        "header:",
        "header:Bad Header",
        "header:" + "x" * 65,
        "header:한글",
        "xff:0",
        "xff:-1",
        f"xff:{MAX_XFF_HOPS + 1}",
        "xff:１",  # a full-width digit
        "xff:two",
        "forwarded",
        "cf",
    ],
)
def test_malformed_settings_are_refused_in_korean(value: str) -> None:
    with pytest.raises(ValueError, match="JETTAE_CLIENT_IP_SOURCE") as refused:
        parse_client_ip_source(value)
    assert "peer, header:<헤더 이름>" in str(refused.value)
    with pytest.raises(ValidationError, match="JETTAE_CLIENT_IP_SOURCE"):
        settings(client_ip_source=value)


def test_the_setting_is_stored_normalised_and_checked_again_by_the_app() -> None:
    assert settings(client_ip_source=" XFF:2 ").client_ip_source == "xff:2"
    broken = settings().model_copy(update={"client_ip_source": "xff:0"})  # bypasses validation
    with pytest.raises(ValueError, match="JETTAE_CLIENT_IP_SOURCE"):
        create_app(broken)


# --------------------------------------------------------------------------- keys


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("203.0.113.9", "203.0.113.9"),
        (" 203.0.113.9 ", "203.0.113.9"),
        ("203.0.113.9:443", "203.0.113.9"),
        ("::ffff:203.0.113.9", "203.0.113.9"),
        ("2001:db8:1:2:aaaa:bbbb:cccc:dddd", "2001:db8:1:2::/64"),
        ("[2001:db8:1:2::9]:443", "2001:db8:1:2::/64"),
        ("::1", "::/64"),
    ],
)
def test_addresses_become_bounded_keys(text: str, key: str) -> None:
    address = parse_ip(text)
    assert address is not None
    assert address_key(address) == key


@pytest.mark.parametrize(
    "text", ["", "testclient", "unknown", "1.2.3", "1.2.3.4, 5.6.7.8", "x" * 8000, "1.2.3.4:a:b"]
)
def test_anything_else_is_not_an_address(text: str) -> None:
    assert parse_ip(text) is None


def test_one_ipv6_subscriber_is_one_client() -> None:
    first = _resolve("peer", "2001:db8:1:2::1")
    assert first == _resolve("peer", "2001:db8:1:2:ffff::1")
    assert first != _resolve("peer", "2001:db8:1:3::1")


@given(st.text(max_size=200))
def test_keys_are_never_raw_header_text(header: str) -> None:
    value = header.encode("latin-1", errors="replace").decode("latin-1").replace("\r", "").replace("\n", "")
    for source in ("header:CF-Connecting-IP", "xff:1", "xff:2"):
        key = _resolve(source, "198.51.100.7", ("CF-Connecting-IP", value), ("X-Forwarded-For", value))
        assert key in (INVALID_CLIENT, "198.51.100.7") or ipaddress.ip_network(key, strict=False)
        assert len(key) <= 45


def test_peer_keys() -> None:
    assert _resolve("peer", None) == UNKNOWN_CLIENT
    assert _resolve("peer", "") == UNKNOWN_CLIENT
    assert _resolve("peer", "testclient") == INVALID_CLIENT
    assert _resolve("peer", "203.0.113.9", ("X-Forwarded-For", "198.51.100.1")) == "203.0.113.9"


# --------------------------------------------------------------------------- header:<name>


def test_a_header_source_reads_one_address_and_falls_back_to_the_peer() -> None:
    source = "header:CF-Connecting-IP"
    assert _resolve(source, "10.0.0.1", ("cf-connecting-ip", "203.0.113.9")) == "203.0.113.9"
    assert _resolve(source, "10.0.0.1") == "10.0.0.1"  # missing
    assert _resolve(source, "10.0.0.1", ("CF-Connecting-IP", "not-an-ip")) == "10.0.0.1"
    assert _resolve(source, "10.0.0.1", ("CF-Connecting-IP", "203.0.113.9, 198.51.100.1")) == "10.0.0.1"
    repeated = (("CF-Connecting-IP", "203.0.113.9"), ("CF-Connecting-IP", "198.51.100.1"))
    assert _resolve(source, "10.0.0.1", *repeated) == "10.0.0.1"
    assert _resolve(source, None) == UNKNOWN_CLIENT


def test_a_missing_header_falls_back_to_the_rightmost_forwarded_entry_not_the_forgeable_left() -> None:
    source = "header:CF-Connecting-IP"
    # uvicorn behind FORWARDED_ALLOW_IPS="*" reports the client-written leftmost entry as the peer
    forged = ("X-Forwarded-For", "6.6.6.6, 203.0.113.9")
    assert _resolve(source, "6.6.6.6", forged) == "203.0.113.9"
    assert _resolve(source, "7.7.7.7", ("X-Forwarded-For", "7.7.7.7, 203.0.113.9")) == "203.0.113.9"
    assert _resolve(source, "6.6.6.6", ("X-Forwarded-For", "6.6.6.6, garbage")) == INVALID_CLIENT


# --------------------------------------------------------------------------- xff:<n>


def test_xff_counts_from_the_right_so_a_forged_left_part_does_not_matter() -> None:
    honest = ("X-Forwarded-For", "203.0.113.9")
    forged = ("X-Forwarded-For", "10.9.9.9, 1.1.1.1, 203.0.113.9")
    assert _resolve("xff:1", "10.0.0.1", honest) == "203.0.113.9"
    assert _resolve("xff:1", "10.0.0.1", forged) == "203.0.113.9"
    # two proxies: the visitor is the second entry from the right
    assert (
        _resolve("xff:2", "10.0.0.1", ("X-Forwarded-For", "6.6.6.6, 203.0.113.9, 10.214.3.7"))
        == "203.0.113.9"
    )


def test_xff_falls_back_to_the_peer_when_too_short_and_keys_garbage_as_invalid() -> None:
    assert _resolve("xff:2", "10.0.0.1", ("X-Forwarded-For", "203.0.113.9")) == "10.0.0.1"
    assert _resolve("xff:1", "10.0.0.1") == "10.0.0.1"
    assert _resolve("xff:1", "10.0.0.1", ("X-Forwarded-For", " , ")) == "10.0.0.1"
    assert _resolve("xff:2", "10.0.0.1", ("X-Forwarded-For", "garbage, 10.214.3.7")) == INVALID_CLIENT


def test_repeated_xff_headers_are_one_list() -> None:
    lines = (("X-Forwarded-For", "6.6.6.6"), ("X-Forwarded-For", "203.0.113.9, 10.214.3.7"))
    assert forwarded_entries(_headers(*lines).getlist("X-Forwarded-For")) == [
        "6.6.6.6",
        "203.0.113.9",
        "10.214.3.7",
    ]
    assert _resolve("xff:2", "10.0.0.1", *lines) == "203.0.113.9"


# --------------------------------------------------------------------------- over HTTP


def test_a_forged_leftmost_xff_no_longer_escapes_the_event_limit() -> None:
    """The verifier's bypass: a fresh leftmost entry per request. With xff:1 they all count as one."""
    app = new_app(events_per_client_per_minute=2, client_ip_source="xff:1")
    with open_client(app, client=PROXY) as client:
        statuses = [
            client.post(
                f"{API}/events",
                json={"name": "landing_view"},
                headers={"X-Forwarded-For": f"10.{index}.0.1, 203.0.113.9"},
            ).status_code
            for index in range(4)
        ]
    assert statuses == [202, 202, 429, 429]
    assert len(app.state.event_limiter) == 1


def test_free_scan_shares_follow_the_configured_source() -> None:
    app = new_app(free_scans_per_client_per_day=1, client_ip_source="header:CF-Connecting-IP")
    scan = [("files", ("결제.md", SMALL_RUNBOOK.encode("utf-8"), "text/markdown"))]
    with open_client(app, client=PROXY) as client:
        first = client.post(f"{API}/free-scans", files=scan, headers={"CF-Connecting-IP": "203.0.113.9"})
        again = client.post(f"{API}/free-scans", files=scan, headers={"CF-Connecting-IP": "203.0.113.9"})
        other = client.post(f"{API}/free-scans", files=scan, headers={"CF-Connecting-IP": "198.51.100.1"})
    assert (first.status_code, again.status_code, other.status_code) == (201, 429, 201)
    assert again.json()["code"] == "free_scan_limit"


def _public(**overrides: Any) -> Any:
    return new_app(auth_mode="public", api_tokens=TOKENS, **overrides)


def test_request_info_shows_the_headers_and_the_resolved_key_to_admins() -> None:
    app = _public(client_ip_source="xff:2")
    headers = {
        **BEARER,
        "X-Forwarded-For": "6.6.6.6, 203.0.113.9, 10.214.3.7",
        "CF-Connecting-IP": "203.0.113.9",
        "X-Real-IP": "10.214.3.7",
    }
    with open_client(app, client=PROXY) as client:
        response = client.get(f"{API}/admin/request-info", headers=headers)
    assert response.status_code == 200, response.text
    assert response.json() == {
        "peer": "10.214.3.7",
        "x_forwarded_for": "6.6.6.6, 203.0.113.9, 10.214.3.7",
        "cf_connecting_ip": "203.0.113.9",
        "true_client_ip": None,
        "x_real_ip": "10.214.3.7",
        "client_ip_source": "xff:2",
        "resolved_client": "203.0.113.9",
    }
    assert response.headers["cache-control"] == "no-store"


def test_request_info_cuts_long_headers_and_is_for_admins_only() -> None:
    with open_client(_public(), client=VISITOR) as client:
        long = client.get(f"{API}/admin/request-info", headers={**BEARER, "True-Client-IP": "9" * 5000})
        assert long.status_code == 200
        assert len(long.json()["true_client_ip"]) == 512
        assert long.json()["resolved_client"] == "203.0.113.9"
        assert long.json()["client_ip_source"] == "peer"
        anonymous = client.get(f"{API}/admin/request-info")
        assert anonymous.status_code == 403
        assert anonymous.json()["code"] == "forbidden"
        assert (
            client.get(f"{API}/admin/request-info", headers={"Authorization": "Bearer nope"}).status_code
            == 401
        )
    with open_client(new_app()) as dev:
        for headers in ({}, VIEWER, EDITOR):
            assert dev.get(f"{API}/admin/request-info", headers=headers).status_code == 403
        assert dev.get(f"{API}/admin/request-info", headers=ADMIN).status_code == 200
