"""What a free scan keeps (무료 검진 보관): masked free text, and no second copy of the report.

* the free text of an uploaded catalog (description, owner, location) and roster (title, team,
  aliases) is PII-masked before assembly, so phone numbers and RRNs never reach the stored graph;
  the roster's names, statuses, roles and contact media stay as uploaded (the checks need them);
* template drills and replay results of a free scan live only in its report — their own tables
  are not filled a second time — while every page the web app shows for a free scan still works;
* other workspaces store exactly as before.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi.testclient import TestClient

from jettae.api.security import WORKSPACE_KEY_HEADER as WORKSPACE_KEY
from jettae.domain import CatalogService, ChannelMedium, PersonStatus, RosterEntry, ScenarioOrigin
from jettae.ingest import load_bundle_dir, mask_pii
from jettae.orchestrator import JettaeService
from jettae.orchestrator.service import mask_catalog_free_text, mask_roster_free_text
from test_api_support import (
    API,
    DEMO,
    SMALL_ROSTER,
    SMALL_RUNBOOK,
    new_app,
    new_service,
    open_client,
)

PHONE = "010-9876-5432"
RRN = "900101-1234567"
EMAIL = "kim.seoyeon@corp.example"

CATALOG = f"""
services:
  - name: 결제 서비스
    kind: service
    tier: 1
    aliases: [payment]
    location: 상암 IDC (당직실 {PHONE})
    owner: {EMAIL}
    description: 결제 승인 API — 장애 시 김서연 {PHONE}, 주민번호 {RRN} 확인 후 처리
"""

ROSTER = (
    "이름,상태,직급,팀,역할,별칭,연락수단,퇴사일\n"
    f"김서연,재직,사원 {PHONE},결제팀 {EMAIL},1차 당직,서연;{RRN},슬랙;전화,\n"
)


def _free_scan(client: TestClient, *, catalog: str = CATALOG, roster: str = ROSTER) -> dict[str, Any]:
    files = [
        ("files", ("결제.md", SMALL_RUNBOOK.encode("utf-8"), "text/markdown")),
        ("roster", ("roster.csv", roster.encode("utf-8"), "text/csv")),
        ("catalog", ("catalog.yaml", catalog.encode("utf-8"), "application/yaml")),
    ]
    response = client.post(f"{API}/free-scans", files=files)
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


# --------------------------------------------------------------------------- masking helpers


def test_catalog_free_text_is_masked_and_identifiers_are_kept() -> None:
    service = CatalogService(
        name="결제 서비스",
        aliases=("payment", PHONE),
        depends_on=("DB",),
        location=f"IDC {PHONE}",
        owner=EMAIL,
        description=f"연락 {PHONE} / {RRN}",
        tier=1,
    )
    (masked,) = mask_catalog_free_text([service])
    assert masked.description == mask_pii(service.description or "")[0]
    assert PHONE not in (masked.description or "")
    assert RRN not in (masked.description or "")
    assert PHONE not in (masked.location or "")
    assert masked.owner != EMAIL
    assert (masked.name, masked.aliases, masked.depends_on, masked.tier, masked.kind) == (
        service.name,
        service.aliases,
        service.depends_on,
        service.tier,
        service.kind,
    )


def test_roster_labels_are_masked_and_what_the_checks_need_is_kept() -> None:
    entry = RosterEntry(
        name="김서연",
        status=PersonStatus.ACTIVE,
        title=f"사원 {PHONE}",
        team=f"결제팀 {EMAIL}",
        roles=("1차 당직",),
        aliases=("서연", RRN),
        channels=(ChannelMedium.SLACK, ChannelMedium.PHONE),
        left_on="2026-09-30",
    )
    (masked,) = mask_roster_free_text([entry])
    assert PHONE not in (masked.title or "")
    assert EMAIL not in (masked.team or "")
    assert masked.aliases[0] == "서연"
    assert RRN not in masked.aliases[1]
    assert (masked.name, masked.status, masked.roles, masked.channels, masked.left_on) == (
        entry.name,
        entry.status,
        entry.roles,
        entry.channels,
        entry.left_on,
    )


def test_inputs_without_personal_data_come_back_as_the_same_objects() -> None:
    """P6: normal inputs — the demo's — are not touched at all."""
    bundle = load_bundle_dir(DEMO)
    assert bundle.catalog
    assert bundle.roster
    for original, masked in zip(bundle.catalog, mask_catalog_free_text(bundle.catalog), strict=True):
        assert masked is original
    for person, kept in zip(bundle.roster, mask_roster_free_text(bundle.roster), strict=True):
        assert kept is person


# --------------------------------------------------------------------------- over HTTP


def test_a_free_scan_graph_holds_no_phone_number_or_rrn_from_the_catalog() -> None:
    """The verifier's probe: a catalog description with a phone number and an RRN."""
    with open_client(new_app()) as client:
        scan = _free_scan(client)
        ws = scan["workspace_id"]
        graph = client.get(f"{API}/workspaces/{ws}/graph", headers={WORKSPACE_KEY: scan["access_key"]})
        assert graph.status_code == 200, graph.text
        text = graph.text
        for secret in (PHONE, RRN, EMAIL):
            assert secret not in text, secret
        nodes = {node["name"]: node for node in graph.json()["nodes"]}
        description = nodes["결제 서비스"]["values"]["description"]
        assert (
            description == mask_pii(f"결제 승인 API — 장애 시 김서연 {PHONE}, 주민번호 {RRN} 확인 후 처리")[0]
        )
        assert "김서연" in nodes  # the roster name is kept: the checks match on it
        report = client.get(f"{API}/workspaces/{ws}/report")
        for secret in (PHONE, RRN, EMAIL):
            assert secret not in report.text, secret


# --------------------------------------------------------------------------- lean storage


def _demo_scan(service: JettaeService) -> str:
    bundle = load_bundle_dir(DEMO)
    outcome = asyncio.run(
        service.free_scan(
            documents=bundle.documents,
            roster=bundle.roster,
            oncall=bundle.oncall,
            catalog=bundle.catalog,
            policy=bundle.policy,
            actor="anonymous",
        )
    )
    return outcome.workspace_id


def test_a_free_scan_stores_template_drills_and_replays_only_in_its_report() -> None:
    service = new_service()
    ws = _demo_scan(service)
    report = service.latest_report(ws)
    assert report.rehearsals
    assert report.scenarios
    assert report.patches
    assert service.store.rehearsals.list_results(ws) == []
    stored = service.store.rehearsals.list_scenarios(ws)
    assert all(s.origin is not ScenarioOrigin.TEMPLATE for s in stored)
    assert {p.id for p in service.patches(ws)} == {p.id for p in report.patches}


def test_every_page_of_a_free_scan_still_answers() -> None:
    """What the web app reads for a free scan: report, graph, versions, diff, patches, downloads."""
    service = new_service()
    with open_client(new_app(service=service)) as client:
        files = [
            ("files", ("결제.md", SMALL_RUNBOOK.encode("utf-8"), "text/markdown")),
            ("roster", ("roster.csv", SMALL_ROSTER.encode("utf-8"), "text/csv")),
        ]
        scan = client.post(f"{API}/free-scans", files=files)
        assert scan.status_code == 201, scan.text
        ws = scan.json()["workspace_id"]
        headers = {WORKSPACE_KEY: scan.json()["access_key"]}
        report = client.get(f"{API}/workspaces/{ws}/report", headers=headers).json()
        for path in (
            "",
            "/report",
            "/graph",
            "/versions",
            "/patches",
            "/report.md",
            "/report.html",
            "/report.sarif",
        ):
            response = client.get(f"{API}/workspaces/{ws}{path}", headers=headers)
            assert response.status_code == 200, (path, response.text)
        patches = client.get(f"{API}/workspaces/{ws}/patches", headers=headers).json()
        assert sorted(p["id"] for p in patches) == sorted(p["id"] for p in report["patches"])
        assert report["rehearsals"]
        assert service.store.rehearsals.list_results(ws) == []


def test_other_workspaces_store_their_drills_and_replays_as_before() -> None:
    service = new_service()
    report = asyncio.run(service.bootstrap_demo())
    ws = report.workspace_id
    assert {r.result_hash() for r in service.store.rehearsals.list_results(ws, report.graph_hash)} == {
        r.result_hash() for r in report.rehearsals if r.graph_hash == report.graph_hash
    }
    origins = {s.origin for s in service.store.rehearsals.list_scenarios(ws)}
    assert ScenarioOrigin.TEMPLATE in origins
    assert ScenarioOrigin.HUMAN in origins


def test_a_free_scan_report_is_the_same_as_a_full_store_would_give() -> None:
    """Storing less changes no output: the free scan's report judges exactly what the same inputs
    give in a workspace that stores everything (same graph, score, findings, drills, replays)."""
    lean = new_service()
    full = new_service()
    lean_report = lean.latest_report(_demo_scan(lean))
    team = full.create_workspace("결제 플랫폼팀").id
    full.load_bundle(team, load_bundle_dir(DEMO))
    full_report = asyncio.run(full.diagnose(team))
    assert lean_report.graph_hash == full_report.graph_hash
    assert lean_report.readiness == full_report.readiness
    assert [f.rule_id for f in lean_report.findings] == [f.rule_id for f in full_report.findings]
    assert [s.id for s in lean_report.scenarios] == [s.id for s in full_report.scenarios]
    assert [r.result_hash() for r in lean_report.rehearsals] == [
        r.result_hash() for r in full_report.rehearsals
    ]
    assert len(lean_report.patches) == len(full_report.patches)
