"""Uploads over HTTP (multipart → ingest loaders with *bytes*), size limits, and the free-scan budget.

What matters here is that the HTTP path changes nothing about the facts: uploading the demo bundle
file by file must give the same findings and score as loading the bundle directory, every runbook
format must arrive as a guarded document, and nothing a client sends can name a server file.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from jettae.ingest import load_bundle_dir
from jettae.orchestrator import DEMO_WORKSPACE_NAME, JettaeService
from test_api_support import (
    API,
    DEMO,
    EDITOR,
    LEAD,
    SMALL_ROSTER,
    SMALL_RUNBOOK,
    VIEWER,
    create_workspace,
    demo_files,
    docx_bytes,
    fixed_clock,
    html_bytes,
    new_app,
    new_service,
    open_client,
    pdf_bytes,
)

_MIB = 1024 * 1024


def _small_inputs() -> list[tuple[str, tuple[str, bytes, str]]]:
    return [
        ("runbooks", ("결제.md", SMALL_RUNBOOK.encode("utf-8"), "text/markdown")),
        ("roster", ("roster.csv", SMALL_ROSTER.encode("utf-8"), "text/csv")),
    ]


# --------------------------------------------------------------------------- uploads


def test_uploaded_bundle_diagnoses_like_the_bundle_directory() -> None:
    with open_client(new_app()) as client:
        # the demo endpoint loads examples/dmc-shop with the directory loader
        demo = client.post(f"{API}/demo/dmc-shop", json={"llm_mode": "off"})
        assert demo.status_code == 200, demo.text
        expected = demo.json()["report"]
        ws = create_workspace(client)
        uploaded = client.post(f"{API}/workspaces/{ws}/inputs", files=demo_files(), headers=EDITOR)
        assert uploaded.status_code == 200, uploaded.text
        summary = uploaded.json()
        assert len(summary["documents"]) == 3
        assert summary["roster"] == len(load_bundle_dir(DEMO).roster)
        assert summary["pending_scenario_files"] == 1
        assert summary["diagnosable"] is True
        report = client.post(f"{API}/workspaces/{ws}/diagnose", headers=EDITOR)
        assert report.status_code == 200, report.text
        actual = report.json()
    assert sorted(f["id"] for f in actual["findings"]) == sorted(f["id"] for f in expected["findings"])
    assert actual["readiness"] == expected["readiness"] | {"graph_hash": actual["graph_hash"]}
    assert sorted(s["id"] for s in actual["scenarios"]) == sorted(s["id"] for s in expected["scenarios"])


def test_pdf_docx_and_html_runbooks_are_loaded_from_bytes() -> None:
    files = [
        (
            "runbooks",
            ("payment.pdf", pdf_bytes(["Payment runbook", "Call the on-call lead"]), "application/pdf"),
        ),
        (
            "runbooks",
            (
                "주문.docx",
                docx_bytes("주문 서비스 런북", ["담당: 김서연 사원 (슬랙, 전화)"]),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ),
        ),
        ("runbooks", ("환율.html", html_bytes("환율 서비스 런북", ["담당: 김서연 사원"]), "text/html")),
        ("roster", ("roster.csv", SMALL_ROSTER.encode("utf-8"), "text/csv")),
    ]
    with open_client(new_app()) as client:
        ws = create_workspace(client)
        response = client.post(f"{API}/workspaces/{ws}/inputs", files=files, headers=EDITOR)
        assert response.status_code == 200, response.text
        documents = response.json()["documents"]
        assert sorted(d["title"] for d in documents) == ["payment", "주문 서비스 런북", "환율 서비스 런북"]
        assert sorted(d["path"] for d in documents) == ["payment.pdf", "주문.docx", "환율.html"]
        diagnosed = client.post(f"{API}/workspaces/{ws}/diagnose", json={"llm_mode": "off"}, headers=EDITOR)
        assert diagnosed.status_code == 200, diagnosed.text
        graph = client.get(f"{API}/workspaces/{ws}/graph").json()
        assert {d["title"] for d in graph["docs"]} == {d["title"] for d in documents}


def test_new_runbooks_replace_the_set_and_structured_inputs_are_kept() -> None:
    with open_client(new_app()) as client:
        ws = create_workspace(client)
        url = f"{API}/workspaces/{ws}/inputs"
        assert client.post(url, files=_small_inputs(), headers=EDITOR).status_code == 200
        second = [("runbooks", ("다른.md", "# 다른 런북\n\n- 담당: 김서연 사원\n".encode(), "text/markdown"))]
        summary = client.post(url, files=second, headers=EDITOR).json()
        assert [d["title"] for d in summary["documents"]] == ["다른 런북"]
        assert summary["roster"] == 1
        assert client.get(url).json() == summary


def test_duplicate_runbooks_are_kept_once_and_paths_lose_their_directories() -> None:
    body = SMALL_RUNBOOK.encode("utf-8")
    files = [
        ("runbooks", ("../../etc/결제.md", body, "text/markdown")),
        ("runbooks", ("C:\\secret\\결제.md", body, "text/markdown")),
    ]
    with open_client(new_app()) as client:
        ws = create_workspace(client)
        summary = client.post(f"{API}/workspaces/{ws}/inputs", files=files, headers=EDITOR).json()
    assert [d["path"] for d in summary["documents"]] == ["결제.md"]


def _multipart(filename: bytes, body: bytes) -> tuple[bytes, dict[str, str]]:
    """A one-part multipart body with a raw (UTF-8) file name, as a browser or curl sends it."""
    payload = (
        b'--B0\r\nContent-Disposition: form-data; name="runbooks"; filename="'
        + filename
        + b'"\r\nContent-Type: text/markdown\r\n\r\n'
        + body
        + b"\r\n--B0--\r\n"
    )
    return payload, {**EDITOR, "Content-Type": "multipart/form-data; boundary=B0"}


def test_file_names_lose_control_bidi_and_line_separator_characters() -> None:
    """The name is stored as the document path and shown in reports, the web app and terminals:
    an ANSI escape, a bell, a bidi override or a line separator must not survive into it."""
    raw = "결제\u2028# 승인 완료\x1b[31m\x07\u202egpj.md"
    payload, headers = _multipart(raw.encode("utf-8"), SMALL_RUNBOOK.encode("utf-8"))
    with open_client(new_app()) as client:
        ws = create_workspace(client)
        response = client.post(f"{API}/workspaces/{ws}/inputs", content=payload, headers=headers)
    assert response.status_code == 200, response.text
    (document,) = response.json()["documents"]
    assert document["path"] == "결제_# 승인 완료_[31m__gpj.md"


def test_overlong_file_names_are_refused() -> None:
    files = [("runbooks", ("가" * 253 + ".md", SMALL_RUNBOOK.encode("utf-8"), "text/markdown"))]
    with open_client(new_app()) as client:
        ws = create_workspace(client)
        response = client.post(f"{API}/workspaces/{ws}/inputs", files=files, headers=EDITOR)
        assert response.status_code == 422
        assert response.json()["detail"] == "runbooks: 파일 이름이 너무 깁니다(최대 255자)."
        files[0] = ("runbooks", ("가" * 252 + ".md", files[0][1][1], "text/markdown"))
        assert client.post(f"{API}/workspaces/{ws}/inputs", files=files, headers=EDITOR).status_code == 200


def test_the_demo_workspace_name_is_reserved() -> None:
    """The public demo endpoint reloads the example inputs into the workspace with this name, so a
    team workspace created under it would be overwritten by any anonymous visitor."""
    service: JettaeService = new_service()
    with open_client(new_app(service=service)) as client:
        for name in (DEMO_WORKSPACE_NAME, f"  {DEMO_WORKSPACE_NAME}\u200b"):
            refused = client.post(f"{API}/workspaces", json={"name": name}, headers=EDITOR)
            assert refused.status_code == 422
            assert refused.json()["code"] == "invalid_input"
            assert "예약된 이름" in refused.json()["detail"]
        assert service.list_workspaces() == []
        assert client.post(f"{API}/demo/dmc-shop", json={"llm_mode": "off"}).status_code == 200
    assert [w.name for w in service.list_workspaces()] == [DEMO_WORKSPACE_NAME]


@pytest.mark.parametrize(
    ("files", "fragment"),
    [
        ([("runbooks", ("tool.exe", b"MZ", "application/octet-stream"))], "지원하지 않는 문서 형식"),
        ([("runbooks", ("빈.md", b"   \n", "text/markdown"))], "텍스트를 추출하지 못했습니다"),
        (
            [("roster", ("roster.csv", "이름,상태,퇴사일\n김서연,퇴사,2026-13-45\n".encode(), "text/csv"))],
            "명부 2행",
        ),
        ([("runbooks", ("깨짐.pdf", b"%PDF-1.4 not really", "application/pdf"))], "깨짐.pdf"),
    ],
)
def test_unreadable_inputs_are_422_in_korean(
    files: list[tuple[str, tuple[str, bytes, str]]], fragment: str
) -> None:
    with open_client(new_app()) as client:
        ws = create_workspace(client)
        response = client.post(f"{API}/workspaces/{ws}/inputs", files=files, headers=EDITOR)
    assert response.status_code == 422, response.text
    assert response.json()["code"] == "invalid_input"
    assert fragment in response.json()["detail"]


def test_upload_needs_files_an_editor_and_a_workspace() -> None:
    with open_client(new_app()) as client:
        ws = create_workspace(client)
        url = f"{API}/workspaces/{ws}/inputs"
        nothing = client.post(url, headers=EDITOR, files={})
        assert nothing.status_code == 422
        assert "올린 파일이 없습니다" in nothing.json()["detail"]
        assert client.post(url, files=_small_inputs(), headers=VIEWER).status_code == 403
        missing = client.post(f"{API}/workspaces/ws-missing/inputs", files=_small_inputs(), headers=EDITOR)
        assert missing.status_code == 404


# --------------------------------------------------------------------------- size limits


@pytest.fixture
def tiny_limit_client() -> Iterator[tuple[TestClient, str]]:
    with open_client(new_app(max_upload_mb=1)) as client:
        yield client, create_workspace(client)


def test_declared_oversized_upload_is_refused_before_reading(
    tiny_limit_client: tuple[TestClient, str],
) -> None:
    client, ws = tiny_limit_client
    big = [("runbooks", ("big.md", b"#" * (2 * _MIB), "text/markdown"))]
    response = client.post(f"{API}/workspaces/{ws}/inputs", files=big, headers=EDITOR)
    assert response.status_code == 413
    assert response.json()["code"] == "payload_too_large"
    assert "최대 1MB" in response.json()["detail"]
    assert response.headers["x-request-id"]


def test_files_over_the_budget_are_refused_by_name(tiny_limit_client: tuple[TestClient, str]) -> None:
    client, ws = tiny_limit_client
    # just over 1 MiB of files: inside the multipart envelope allowance, so the endpoint decides
    parts = [
        ("runbooks", ("a.md", b"x" * (_MIB // 2), "text/markdown")),
        ("runbooks", ("b.md", b"y" * (_MIB // 2 + 10), "text/markdown")),
    ]
    response = client.post(f"{API}/workspaces/{ws}/inputs", files=parts, headers=EDITOR)
    assert response.status_code == 413
    assert "b.md" in response.json()["detail"]


def test_streamed_bodies_are_counted_without_content_length() -> None:
    def chunks() -> Iterator[bytes]:
        yield b'{"email": "a@example.com", "message": "'
        yield b"x" * (2 * _MIB)
        yield b'", "consent_privacy": true}'

    with open_client(new_app()) as client:
        response = client.post(
            f"{API}/signups", content=chunks(), headers={"Content-Type": "application/json"}
        )
        assert "content-length" not in {k.lower() for k in response.request.headers}
        assert response.status_code == 413
        assert response.json()["code"] == "payload_too_large"
        declared = client.post(f"{API}/events", json={"name": "landing_view", "path": "/" + "a" * (2 * _MIB)})
        assert declared.status_code == 413


# --------------------------------------------------------------------------- free-scan budget


def _prepared(client: TestClient) -> str:
    ws = create_workspace(client)
    response = client.post(f"{API}/workspaces/{ws}/inputs", files=_small_inputs(), headers=EDITOR)
    assert response.status_code == 200
    return ws


def test_viewer_and_anonymous_diagnoses_share_the_daily_budget() -> None:
    with open_client(new_app(daily_free_scan_limit=1)) as client:
        ws = _prepared(client)
        url = f"{API}/workspaces/{ws}/diagnose"
        assert client.post(url, headers=VIEWER).status_code == 200
        refused = client.post(url)
        assert refused.status_code == 429
        assert refused.json()["code"] == "free_scan_limit"
        assert "1회" in refused.json()["detail"]
        assert 0 < int(refused.headers["retry-after"]) <= 24 * 3600
        assert client.post(url, headers=VIEWER).status_code == 429
        # editors and above are not metered
        assert client.post(url, headers=EDITOR).status_code == 200
        assert client.post(url, headers=LEAD).status_code == 200


def test_scans_that_fail_on_their_request_still_use_the_budget() -> None:
    """Only a server fault gives a scan back: otherwise every input that makes the parsers work
    hard and then fails (a parse bomb ends in 422) would be free."""
    app = new_app(daily_free_scan_limit=3)
    with open_client(app) as client:
        ws = _prepared(client)
        assert client.post(f"{API}/workspaces/ws-missing/diagnose", headers=VIEWER).status_code == 404
        empty = create_workspace(client, "빈 워크스페이스")
        no_runbooks = client.post(f"{API}/workspaces/{empty}/diagnose", headers=VIEWER)
        assert no_runbooks.status_code == 422
        assert "런북" in no_runbooks.json()["detail"]
        assert app.state.free_scans.used(fixed_clock()) == 2
        assert client.post(f"{API}/workspaces/{ws}/diagnose", headers=VIEWER).status_code == 200
        assert client.post(f"{API}/workspaces/{ws}/diagnose", headers=VIEWER).status_code == 429


def test_demo_runs_are_free_scans_too() -> None:
    service: JettaeService = new_service()
    with open_client(new_app(service=service, daily_free_scan_limit=0)) as client:
        refused = client.post(f"{API}/demo/dmc-shop")
        assert refused.status_code == 429
        assert "꺼져 있습니다" in refused.json()["detail"]
    assert service.list_workspaces() == []


def test_file_names_without_a_name_part_are_refused() -> None:
    with open_client(new_app()) as client:
        ws = create_workspace(client)
        for name in ("/", "folder/"):
            parts = [("runbooks", (name, b"# x\n", "text/markdown"))]
            response = client.post(f"{API}/workspaces/{ws}/inputs", files=parts, headers=EDITOR)
            assert response.status_code == 422
            assert response.json()["detail"] == "runbooks: 파일 이름이 없습니다."


def test_missing_demo_bundle_is_503_and_does_not_use_the_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("JETTAE_DEMO_BUNDLE", str(tmp_path / "nowhere"))
    app = new_app(daily_free_scan_limit=1)
    with open_client(app) as client:
        response = client.post(f"{API}/demo/dmc-shop")
        assert response.status_code == 503
        assert response.json()["code"] == "demo_unavailable"
        assert str(tmp_path) not in response.text
    assert app.state.free_scans.used(fixed_clock()) == 0
