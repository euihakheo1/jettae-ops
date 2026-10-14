"""The finals demo over HTTP: bootstrap → graph → drafts → approval (403 for a viewer) → revert →
evidence downloads — exactly the calls the web app makes, against an in-memory store.

The judgement must stay LLM-free end to end (P1): every report on the way reports zero LLM calls
on the judgement path, and every view the API builds agrees with the service it wraps.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote

import pytest
from fastapi.testclient import TestClient

from jettae.api.schemas import RehearsalRequest, graph_view
from jettae.domain import LlmMode
from jettae.orchestrator import DEMO_WORKSPACE_NAME, JettaeService
from test_api_support import ADMIN, API, EDITOR, LEAD, VIEWER, new_app, new_service, open_client


@dataclass
class Demo:
    client: TestClient
    service: JettaeService
    ws: str
    report: dict[str, Any]


@pytest.fixture(scope="module")
def demo() -> Iterator[Demo]:
    """A diagnosed demo workspace for read-only checks (nothing here approves or reverts)."""
    service = new_service()
    with open_client(new_app(service=service)) as client:
        response = client.post(f"{API}/demo/dmc-shop", json={"llm_mode": "off"})
        assert response.status_code == 200, response.text
        body = response.json()
        yield Demo(client=client, service=service, ws=body["workspace_id"], report=body["report"])


def _filename_star(disposition: str) -> str:
    match = re.search(r"filename\*=UTF-8''(\S+)", disposition)
    assert match is not None, disposition
    return unquote(match.group(1))


def _validated_patch(client: TestClient, ws: str) -> dict[str, Any]:
    patches = client.get(f"{API}/workspaces/{ws}/patches").json()
    return next(p for p in patches if p["status"] == "validated")


# --------------------------------------------------------------------------- the full flow


def test_demo_flow_over_http() -> None:
    service = new_service()
    with open_client(new_app(service=service)) as client:
        # 1. demo bootstrap — anonymous, LLM off, judgement without any LLM call
        boot = client.post(f"{API}/demo/dmc-shop", json={"llm_mode": "off"})
        assert boot.status_code == 200, boot.text
        body = boot.json()
        ws = body["workspace_id"]
        first = body["report"]
        assert body["workspace_name"] == DEMO_WORKSPACE_NAME
        assert first["workspace_id"] == ws
        assert first["graph_version"] == 1
        assert first["llm"]["calls"] == 0
        assert first["llm"]["judgement_path_calls"] == 0
        assert first["findings"], "the demo bundle has intended defects"

        # 2. the graph view is the graph the report judged
        graph = client.get(f"{API}/workspaces/{ws}/graph").json()
        assert graph["version"] == 1
        assert graph["graph_hash"] == first["graph_hash"]
        assert graph["findings_run_id"] == first["run_id"]

        # 3. drafts exist and were validated before anyone sees them
        patch = _validated_patch(client, ws)
        assert patch["validation"]["passed"] is True

        # 4. viewers and editors cannot approve (Korean 403), the service never ran
        for headers in (VIEWER, EDITOR):
            denied = client.post(f"{API}/patches/{patch['id']}/approve", headers=headers)
            assert denied.status_code == 403
            assert denied.json()["code"] == "forbidden"
            assert "팀 리드" in denied.json()["detail"]
        assert [v["number"] for v in client.get(f"{API}/workspaces/{ws}/versions").json()] == [1]

        # 5. a lead approves → version 2, re-judged by the fixed rules only
        approved = client.post(f"{API}/patches/{patch['id']}/approve", headers=LEAD)
        assert approved.status_code == 200, approved.text
        result = approved.json()
        assert result["patch"]["status"] == "approved"
        assert result["patch"]["decided_by"] == LEAD["X-Jettae-User"]
        assert result["version"]["number"] == 2
        assert result["version"]["patch_id"] == patch["id"]
        assert result["report"]["graph_version"] == 2
        assert result["report"]["llm"]["calls"] == 0

        # 6. deciding twice is a state conflict, not a crash
        again = client.post(f"{API}/patches/{patch['id']}/approve", headers=LEAD)
        assert again.status_code == 409
        assert again.json()["code"] == "patch_state"

        # 7. history and comparison
        versions = client.get(f"{API}/workspaces/{ws}/versions").json()
        assert [v["number"] for v in versions] == [1, 2]
        diff = client.get(f"{API}/workspaces/{ws}/diff", params={"a": 1, "b": 2})
        assert diff.status_code == 200
        ids = service.diff(ws, 1, 2)
        view = diff.json()
        assert [e["id"] for e in view["added_edges"]] == list(ids.added_edges)
        assert [e["id"] for e in view["removed_edges"]] == list(ids.removed_edges)
        assert [n["id"] for n in view["changed_nodes"]] == list(ids.changed_nodes)
        assert not ids.is_empty
        legacy = client.get(f"{API}/workspaces/{ws}/versions/diff", params={"a": 1, "b": 2})
        assert legacy.json() == view

        # 8. the report endpoint refuses to pass v2's result off as v1's
        stale = client.get(f"{API}/workspaces/{ws}/report", params={"version": 1})
        assert stale.status_code == 404
        assert stale.json()["code"] == "report_not_found"
        assert client.get(f"{API}/workspaces/{ws}/report", params={"version": 2}).status_code == 200

        # 9. revert: viewer 403, lead appends version 3 equal to version 1
        assert client.post(f"{API}/workspaces/{ws}/versions/1/revert", headers=VIEWER).status_code == 403
        reverted = client.post(f"{API}/workspaces/{ws}/versions/1/revert", headers=ADMIN)
        assert reverted.status_code == 200, reverted.text
        revert = reverted.json()
        assert revert["patch"] is None
        assert revert["version"]["number"] == 3
        assert revert["version"]["revert_of"] == 1
        assert revert["version"]["graph_hash"] == first["graph_hash"]
        assert revert["report"]["graph_version"] == 3

        # 10. evidence downloads with the right types and a Korean file name
        expected = {
            "md": "text/markdown; charset=utf-8",
            "html": "text/html; charset=utf-8",
            "sarif": "application/sarif+json",
        }
        for kind, media_type in expected.items():
            download = client.get(f"{API}/workspaces/{ws}/report.{kind}")
            assert download.status_code == 200, download.text
            assert download.headers["content-type"] == media_type
            disposition = download.headers["content-disposition"]
            assert disposition.startswith("inline;" if kind == "html" else "attachment;")
            assert f'filename="jettae-report-{ws}-v3.{kind}"' in disposition
            assert _filename_star(disposition) == f"제때옵스-증적리포트-{DEMO_WORKSPACE_NAME}-v3.{kind}"
        markdown = client.get(f"{API}/workspaces/{ws}/report.md").content.decode("utf-8")
        assert markdown == service.evidence_markdown(ws)
        html = client.get(f"{API}/workspaces/{ws}/report.html")
        assert "sandbox" in html.headers["content-security-policy"]
        assert html.headers["x-frame-options"] == "SAMEORIGIN"
        sarif = json.loads(client.get(f"{API}/workspaces/{ws}/report.sarif").content)
        assert sarif["version"] == "2.1.0"
        assert sarif == json.loads(json.dumps(service.sarif(ws)))

        # 11. the workspace list shows the latest headline
        listed = client.get(f"{API}/workspaces").json()
        assert [w["id"] for w in listed] == [ws]
        assert listed[0]["latest_version"] == 3
        assert listed[0]["is_demo"] is True
        assert listed[0]["overall"] == revert["report"]["readiness"]["overall"]


# --------------------------------------------------------------------------- read-only views


def test_graph_view_matches_the_service(demo: Demo) -> None:
    response = demo.client.get(f"{API}/workspaces/{demo.ws}/graph", params={"version": 1})
    assert response.status_code == 200
    head, graph = demo.service.graph(demo.ws, 1)
    expected = graph_view(head, graph, demo.service.latest_report(demo.ws))
    assert response.json() == expected.model_dump(mode="json")
    view = response.json()
    assert [n["id"] for n in view["nodes"]] == sorted(graph.nodes)
    assert [e["id"] for e in view["edges"]] == sorted(graph.edges)
    subjects = {sid for f in demo.report["findings"] for sid in f["subject_ids"]}
    assert set(view["findings_by_element"]) == subjects
    assert {d["doc_id"] for d in view["docs"]} == set(graph.docs)


def test_graph_of_unknown_version_or_workspace_is_404(demo: Demo) -> None:
    missing = demo.client.get(f"{API}/workspaces/{demo.ws}/graph", params={"version": 99})
    assert missing.status_code == 404
    assert missing.json()["code"] == "not_found"
    unknown = demo.client.get(f"{API}/workspaces/ws-nope/graph")
    assert unknown.status_code == 404
    assert "ws-nope" in unknown.json()["detail"]
    assert demo.client.get(f"{API}/workspaces/{demo.ws}/graph", params={"version": 0}).status_code == 422


def test_report_and_workspace_reads(demo: Demo) -> None:
    report = demo.client.get(f"{API}/workspaces/{demo.ws}/report")
    assert report.status_code == 200
    assert report.json() == demo.service.latest_report(demo.ws).model_dump(mode="json")
    summary = demo.client.get(f"{API}/workspaces/{demo.ws}").json()
    assert summary["id"] == demo.ws
    assert summary["finding_count"] == len(demo.report["findings"])
    inputs = demo.client.get(f"{API}/workspaces/{demo.ws}/inputs").json()
    assert inputs["workspace_id"] == demo.ws
    assert len(inputs["documents"]) == 3
    assert inputs["roster"] > 0
    assert inputs["diagnosable"] is True
    rules = demo.client.get(f"{API}/rules").json()
    assert [r["rule_id"] for r in rules] == [r.rule_id for r in demo.service.rule_catalog()]
    scenarios = demo.client.get(f"{API}/workspaces/{demo.ws}/scenarios").json()
    assert any(s["origin"] == "human" for s in scenarios)


def test_rehearsals_need_an_editor_and_replay_deterministically(demo: Demo) -> None:
    url = f"{API}/workspaces/{demo.ws}/rehearsals"
    scenario_id = demo.report["scenarios"][0]["id"]
    assert demo.client.post(url, json={"scenario_ids": [scenario_id]}, headers=VIEWER).status_code == 403
    first = demo.client.post(url, json={"scenario_ids": [scenario_id]}, headers=EDITOR)
    second = demo.client.post(url, json={"scenario_ids": [scenario_id]}, headers=EDITOR)
    assert first.status_code == 200, first.text
    assert first.json() == second.json()
    assert [r["scenario_id"] for r in first.json()] == [scenario_id]
    assert first.json()[0]["graph_hash"] == demo.report["graph_hash"]
    everything = demo.client.post(url, json={"generate": False}, headers=EDITOR)
    assert everything.status_code == 200
    assert {r["scenario_id"] for r in everything.json()} >= {scenario_id}
    unknown = demo.client.post(url, json={"scenario_ids": ["nope"]}, headers=EDITOR)
    assert unknown.status_code == 404
    assert unknown.json()["code"] == "not_found"
    empty = demo.client.post(url, json={"scenario_ids": []}, headers=EDITOR)
    assert empty.status_code == 422
    assert empty.json()["errors"][0]["field"] == "scenario_ids"


def test_drafting_needs_an_editor_and_known_findings(demo: Demo) -> None:
    url = f"{API}/workspaces/{demo.ws}/patches/draft"
    finding_id = demo.report["findings"][0]["id"]
    assert demo.client.post(url, json={"finding_ids": [finding_id]}, headers=VIEWER).status_code == 403
    drafted = demo.client.post(url, json={"finding_ids": [finding_id], "llm_mode": "off"}, headers=EDITOR)
    assert drafted.status_code == 200, drafted.text
    assert drafted.json()
    assert all(finding_id in p["finding_ids"] for p in drafted.json())
    assert all(p["workspace_id"] == demo.ws for p in drafted.json())
    unknown = demo.client.post(url, json={"finding_ids": ["JT-NOPE:0"]}, headers=EDITOR)
    assert unknown.status_code == 404
    patch_id = drafted.json()[0]["id"]
    assert demo.client.get(f"{API}/patches/{patch_id}").json()["id"] == patch_id
    assert demo.client.get(f"{API}/patches/patch-nope").status_code == 404


def test_rejection_needs_a_lead_and_a_reason(demo: Demo) -> None:
    patch = _validated_patch(demo.client, demo.ws)
    url = f"{API}/patches/{patch['id']}/reject"
    assert demo.client.post(url, json={"reason": "아직 이릅니다"}, headers=EDITOR).status_code == 403
    blank = demo.client.post(url, json={"reason": "   "}, headers=LEAD)
    assert blank.status_code == 422
    assert blank.json()["detail"] == "reason: 거절 사유를 적어 주세요."
    rejected = demo.client.post(url, json={"reason": "김서연 사원은 다음 달 휴직 예정입니다."}, headers=LEAD)
    assert rejected.status_code == 200, rejected.text
    assert rejected.json()["status"] == "rejected"
    assert rejected.json()["decided_by"] == LEAD["X-Jettae-User"]
    twice = demo.client.post(url, json={"reason": "다시"}, headers=LEAD)
    assert twice.status_code == 409


@pytest.mark.parametrize(
    ("generate", "llm_mode", "expected"),
    [
        (True, "replay", ("all", LlmMode.REPLAY)),
        (False, "live", ("all", LlmMode.OFF)),
        ("human", None, ("human", None)),
        ("all", None, ("all", None)),
    ],
)
def test_rehearsal_request_generation(
    generate: object, llm_mode: str | None, expected: tuple[str, object]
) -> None:
    request = RehearsalRequest.model_validate({"generate": generate, "llm_mode": llm_mode})
    assert request.resolved() == expected
