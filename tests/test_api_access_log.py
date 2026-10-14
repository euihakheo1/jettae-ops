"""The access log never records personal data from the path.

A matched route is logged as its template with the path parameters filled in, except the ones
that are personal data (``{email}`` of ``DELETE /signups/{email}``); any other path keeps its text
but loses every segment that looks like an e-mail address (also percent-encoded).
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from structlog.testing import capture_logs

from jettae.api.middleware import SENSITIVE_PATH_PARAMS, access_log_path
from test_api_support import ADMIN, API, new_app, open_client


def _access_lines(logs: Sequence[MutableMapping[str, Any]]) -> list[MutableMapping[str, Any]]:
    return [entry for entry in logs if entry["event"] == "http.request"]


def test_deleting_a_signup_logs_no_address() -> None:
    with capture_logs() as logs, open_client(new_app()) as client:
        for address in ("kim@corp.example", "Kim%40Corp.example", "%EA%B9%80@corp.example"):
            response = client.delete(f"{API}/signups/{address}", headers=ADMIN)
            assert response.status_code == 200, response.text
    lines = _access_lines(logs)
    assert len(lines) == 3
    for line in lines:
        assert line["method"] == "DELETE"
        assert line["route"] == "/signups/{email}"
        assert line["path"] == f"{API}/signups/<redacted>"
        assert line["status"] == 200
    text = repr(logs).casefold()
    assert "corp.example" not in text
    assert "kim" not in text
    assert "email" in SENSITIVE_PATH_PARAMS


def test_matched_routes_keep_their_other_path_parameters() -> None:
    with capture_logs() as logs, open_client(new_app()) as client:
        client.get(f"{API}/workspaces/ws-1/report")
        client.get(f"{API}/patches/p-1")
        client.delete(f"{API}/rules")  # 405: the route matched partially
    lines = _access_lines(logs)
    assert [(line["route"], line["path"], line["status"]) for line in lines] == [
        ("/workspaces/{ws}/report", f"{API}/workspaces/ws-1/report", 404),
        ("/patches/{patch_id}", f"{API}/patches/p-1", 404),
        ("/rules", f"{API}/rules", 405),
    ]


def test_an_address_in_another_parameter_is_masked_too() -> None:
    with capture_logs() as logs, open_client(new_app()) as client:
        client.get(f"{API}/workspaces/kim@corp.example/report")
    (line,) = _access_lines(logs)
    assert line["path"] == f"{API}/workspaces/<email>/report"
    assert "corp.example" not in repr(logs)


@pytest.fixture
def web_dist(tmp_path: Path) -> Path:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>제때옵스</title>", encoding="utf-8")
    return dist


def test_unmatched_paths_keep_their_text_without_addresses(web_dist: Path) -> None:
    with capture_logs() as logs, open_client(new_app(web_dist=web_dist)) as client:
        client.get(f"{API}/nope/kim@corp.example/x")
        client.get("/app/w/lee%40corp.example?email=park@corp.example")
        client.get("/app/w/ws-1/patches")
        client.get("/app/a%252540b/y")
    lines = _access_lines(logs)
    assert [(line["route"], line["path"]) for line in lines] == [
        (None, f"{API}/nope/<email>/x"),
        (None, "/app/w/<email>"),
        (None, "/app/w/ws-1/patches"),
        (None, "/app/<email>/y"),
    ]
    assert "corp.example" not in repr(logs)


def test_access_log_path_without_a_route_uses_the_raw_path() -> None:
    assert access_log_path({"raw_path": b"/a/b%40c/d", "path": "/a/b@c/d"}) == "/a/<email>/d"
    assert access_log_path({"path": "/api/v1/workspaces/별빛"}) == "/api/v1/workspaces/%EB%B3%84%EB%B9%9B"
