"""Contract suite for :mod:`jettae.store` — one set of tests, run against every backend.

Each test receives a ``make_store`` factory parametrised over ``MemoryStore`` and ``SqlStore``
(in-memory SQLite, ``StaticPool``). Passing the same suite is what makes the backends
interchangeable for the orchestrator. Backend-specific behaviour (pooling, persistence across
reopen, tamper detection in the database) is tested at the end of the file.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import traceback
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import closing
from datetime import UTC, datetime, timedelta, timezone
from enum import StrEnum
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError
from sqlalchemy import create_engine, func, make_url, select, text
from sqlalchemy.exc import IntegrityError as SqlIntegrityError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from jettae.domain import (
    AddEdge,
    Attr,
    AttrKey,
    CatalogService,
    ChannelMedium,
    DiagnosisReport,
    Document,
    DocumentRef,
    Edge,
    EdgeKind,
    Element,
    ElementScore,
    ElementStatus,
    Extractor,
    Finding,
    FindingStatus,
    FixHint,
    FixKind,
    GraphBuilder,
    GraphVersion,
    InputBundle,
    LlmMode,
    LlmUsageSummary,
    Node,
    NodeKind,
    OnCallShift,
    OpsGraph,
    Outcome,
    Patch,
    PatchAuthor,
    PatchStatus,
    PatchValidation,
    PersonStatus,
    Perturbation,
    PerturbationKind,
    Policy,
    Provenance,
    ReadinessReport,
    RehearsalResult,
    RosterEntry,
    Scenario,
    ServiceReadiness,
    SetAttr,
    Severity,
    SimEvent,
    SimEventKind,
    SourceKind,
    SourceSpan,
    edge_id,
    node_id,
    ports,
)
from jettae.domain.ports import (
    AuditLog,
    GraphRepository,
    PatchRepository,
    RehearsalRepository,
    ReportRepository,
)
from jettae.store import (
    MEMORY_URL,
    AlreadyExistsError,
    Clock,
    MemoryStore,
    NotFoundError,
    Signup,
    SignupRepository,
    SqlStore,
    Store,
    StoredInputs,
    StoreIntegrityError,
    VersionConflictError,
    WorkspaceAccessRepository,
    WorkspaceRepository,
    open_store,
    rehearsal_id,
)
from jettae.store.base import decode_graph, normalize_email
from jettae.store.models import Base
from jettae.store.sql import (
    WORKSPACE_TABLES,
    SqlGraphRepository,
    build_engine,
    database_error_summary,
    is_memory_sqlite,
    normalize_database_url,
)

T0 = datetime(2026, 10, 13, 9, 0, tzinfo=UTC)
KST = timezone(timedelta(hours=9))
MEMORY_SQLITE = "sqlite+pysqlite:///:memory:"
HYPOTHESIS = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)


class TickingClock:
    """Deterministic clock: every call returns the next second after ``start``."""

    def __init__(self, start: datetime = T0) -> None:
        self._next = start
        self._lock = threading.Lock()

    def __call__(self) -> datetime:
        with self._lock:
            now = self._next
            self._next = now + timedelta(seconds=1)
            return now


def _fixed(instant: datetime) -> Clock:
    return lambda: instant


StoreFactory = Callable[..., Store]

BACKENDS: dict[str, Callable[[Clock], Store]] = {
    "memory": lambda clock: MemoryStore(clock=clock),
    "sql": lambda clock: SqlStore(MEMORY_SQLITE, clock=clock),
}


@pytest.fixture(params=sorted(BACKENDS))
def make_store(request: pytest.FixtureRequest) -> StoreFactory:
    """Factory of fresh, empty stores of one backend (default: ticking clock from T0)."""
    backend: str = request.param

    def factory(clock: Clock | None = None) -> Store:
        return BACKENDS[backend](clock if clock is not None else TickingClock())

    return factory


@pytest.fixture
def store(make_store: StoreFactory) -> Iterator[Store]:
    with closing(make_store()) as opened:
        yield opened


# --------------------------------------------------------------------------- domain fixtures

RUNBOOK_TEXT = "결제 서비스 담당은 김민수 대리. 장애 시 슬랙 #pay-oncall 채널로 호출한다."
RUNBOOK = Document(doc_id="doc-pay", title="결제 장애 대응", text=RUNBOOK_TEXT, quarantined=((0, 2),))
SERVICE_ID = node_id(NodeKind.SERVICE, "결제 서비스")
PERSON_ID = node_id(NodeKind.PERSON, "김민수")
CHANNEL_ID = node_id(NodeKind.CHANNEL, "#pay-oncall")


def _span(quote: str) -> SourceSpan:
    start = RUNBOOK_TEXT.index(quote)
    return SourceSpan(doc_id=RUNBOOK.doc_id, start=start, end=start + len(quote), quote=quote, line=1)


def _runbook_prov(quote: str) -> tuple[Provenance, ...]:
    return (Provenance(source=SourceKind.RUNBOOK, extractor=Extractor.RULE, spans=(_span(quote),)),)


def graph_v1() -> OpsGraph:
    builder = GraphBuilder()
    builder.add_doc(DocumentRef(doc_id=RUNBOOK.doc_id, title=RUNBOOK.title, sha256=RUNBOOK.sha256))
    builder.add_node(
        Node(
            id=SERVICE_ID,
            kind=NodeKind.SERVICE,
            name="결제 서비스",
            attrs={str(AttrKey.TIER): Attr(value=1, source=SourceKind.CATALOG)},
            prov=_runbook_prov("결제 서비스"),
        )
    )
    builder.add_node(
        Node(
            id=PERSON_ID,
            kind=NodeKind.PERSON,
            name="김민수",
            attrs={
                str(AttrKey.STATUS): Attr(value=PersonStatus.ACTIVE.value, source=SourceKind.ROSTER),
                str(AttrKey.ALIASES): Attr(value=["김대리"], source=SourceKind.ROSTER),
            },
            prov=(Provenance(source=SourceKind.ROSTER, extractor=Extractor.IMPORT, ref="roster:3"),),
        )
    )
    builder.add_edge(
        Edge(
            id=edge_id(EdgeKind.OWNED_BY, SERVICE_ID, PERSON_ID),
            kind=EdgeKind.OWNED_BY,
            src=SERVICE_ID,
            dst=PERSON_ID,
            prov=_runbook_prov("담당은 김민수 대리"),
        )
    )
    return builder.build()


def graph_v2() -> OpsGraph:
    builder = GraphBuilder(graph_v1())
    builder.add_node(
        Node(
            id=CHANNEL_ID,
            kind=NodeKind.CHANNEL,
            name="#pay-oncall",
            attrs={str(AttrKey.MEDIUM): Attr(value=ChannelMedium.SLACK.value, spans=(_span("슬랙"),))},
            prov=_runbook_prov("#pay-oncall"),
        )
    )
    builder.add_edge(
        Edge(
            id=edge_id(EdgeKind.REACHABLE_VIA, PERSON_ID, CHANNEL_ID),
            kind=EdgeKind.REACHABLE_VIA,
            src=PERSON_ID,
            dst=CHANNEL_ID,
            attrs={"weight": Attr(value=0.5, source=SourceKind.PATCH)},
            prov=(Provenance(source=SourceKind.PATCH, extractor=Extractor.HUMAN, ref="patch-1"),),
        )
    )
    return builder.build()


def version_of(
    workspace_id: str,
    number: int,
    graph: OpsGraph,
    *,
    created_at: datetime = T0,
    patch_id: str | None = None,
) -> GraphVersion:
    return GraphVersion(
        workspace_id=workspace_id,
        number=number,
        parent=number - 1 if number > 1 else None,
        graph_hash=graph.content_hash(),
        patch_id=patch_id,
        rehearsal_ids=("rh-a", "rh-b") if patch_id else (),
        created_by="lead@example.com",
        created_at=created_at,
        note="초기 추출" if number == 1 else "수정안 승인",
    )


def make_patch(
    patch_id: str,
    workspace_id: str = "ws-a",
    *,
    created_at: datetime | None = T0,
    status: PatchStatus = PatchStatus.DRAFT,
) -> Patch:
    edge = Edge(
        id=edge_id(EdgeKind.REACHABLE_VIA, PERSON_ID, CHANNEL_ID),
        kind=EdgeKind.REACHABLE_VIA,
        src=PERSON_ID,
        dst=CHANNEL_ID,
        prov=(Provenance(source=SourceKind.PATCH, extractor=Extractor.HUMAN, ref=patch_id),),
    )
    return Patch(
        id=patch_id,
        workspace_id=workspace_id,
        base_version=1,
        finding_ids=("JT-CHN-001:0123456789",),
        ops=(
            AddEdge(edge=edge),
            SetAttr(target_id=SERVICE_ID, key="tier", value=Attr(value=1, source=SourceKind.PATCH)),
        ),
        runbook_text="결제 서비스 담당 김민수 대리는 슬랙이 끊기면 전화로 호출한다.",
        rationale="주 채널이 서비스와 같은 장애 도메인에 있음",
        author=PatchAuthor.TEMPLATE,
        status=status,
        validation=PatchValidation(passed=True, resolved_finding_ids=("JT-CHN-001:0123456789",)),
        created_at=created_at,
        llm_usage={"calls": 0},
    )


def make_scenario(scenario_id: str = "sc-owner-absent") -> Scenario:
    return Scenario(
        id=scenario_id,
        title="담당자 부재",
        description="주 담당자가 휴가 중",
        service_id=SERVICE_ID,
        perturbations=(Perturbation(kind=PerturbationKind.PERSON_UNAVAILABLE, target_id=PERSON_ID),),
        deadline_minutes=60.0,
    )


def make_result(scenario: Scenario, graph_hash: str, *, config_hash: str = "cfg-1") -> RehearsalResult:
    return RehearsalResult(
        scenario_id=scenario.id,
        scenario_fingerprint=scenario.fingerprint(),
        graph_hash=graph_hash,
        config_hash=config_hash,
        outcome=Outcome.FAIL,
        time_to_mitigate=None,
        deadline_minutes=60.0,
        failed_element=Element.OWNER,
        failed_at_id=PERSON_ID,
        reason="담당자에게 연락할 수 없음",
        events=(
            SimEvent(t=0.0, kind=SimEventKind.INCIDENT_START),
            SimEvent(t=5.0, kind=SimEventKind.PAGE, actor_id=PERSON_ID, target_id=CHANNEL_ID, note="호출"),
            SimEvent(t=10.0, kind=SimEventKind.NO_ANSWER, actor_id=PERSON_ID),
        ),
        path_ids=(SERVICE_ID, PERSON_ID),
    )


def make_report(run_id: str, workspace_id: str = "ws-a", *, created_at: datetime = T0) -> DiagnosisReport:
    graph = graph_v1()
    finding = Finding(
        id=Finding.make_id("JT-PPL-002", SERVICE_ID),
        rule_id="JT-PPL-002",
        severity=Severity.HIGH,
        status=FindingStatus.UNKNOWN,
        title="부담당 없음",
        message="결제 서비스의 담당자가 한 명뿐입니다.",
        service_id=SERVICE_ID,
        element=Element.OWNER,
        subject_ids=(SERVICE_ID, PERSON_ID),
        evidence=(_span("담당은 김민수 대리"),),
        fix=FixHint(kind=FixKind.ADD_DEPUTY, params={"service_id": SERVICE_ID, "candidates": ["박지영"]}),
    )
    readiness = ReadinessReport(
        overall=55.5,
        services=(
            ServiceReadiness(
                service_id=SERVICE_ID,
                service_name="결제 서비스",
                tier=1,
                weight=3.0,
                score=55.5,
                elements=(
                    ElementScore(
                        element=Element.OWNER,
                        score=0.5,
                        status=ElementStatus.PARTIAL,
                        reasons=("부담당 없음",),
                        finding_ids=(finding.id,),
                    ),
                ),
            ),
        ),
        weights_version="SCORING_V1",
        element_weights={"owner": 0.25, "detection": 0.25},
        tier_weights={1: 3.0, 2: 2.0},
        graph_hash=graph.content_hash(),
    )
    scenario = make_scenario()
    return DiagnosisReport(
        run_id=run_id,
        workspace_id=workspace_id,
        graph_version=1,
        graph_hash=graph.content_hash(),
        created_at=created_at,
        findings=(finding,),
        readiness=readiness,
        scenarios=(scenario,),
        rehearsals=(make_result(scenario, graph.content_hash()),),
        patches=(make_patch("patch-1", workspace_id),),
        llm=LlmUsageSummary(mode=LlmMode.OFF),
        held_element_ids=(PERSON_ID,),
        timings_ms={"check": 1.25},
    )


def make_signup(email: str = "Kim@Example.com", *, marketing: bool = True) -> Signup:
    return Signup(
        email=email,
        organization="디엠씨샵",
        team_size="11-50",
        message="온콜 표가 자꾸 어긋나요",
        consent_privacy=True,
        consent_marketing=marketing,
        created_at=T0,
        source="landing",
    )


def _dump(models: object) -> object:
    """JSON-mode dump of a model or list of models, for exact cross-backend comparison."""
    if isinstance(models, list):
        return [_dump(m) for m in models]
    if models is None:
        return None
    dump = getattr(models, "model_dump", None)
    assert dump is not None
    return dump(mode="json")


# --------------------------------------------------------------------------- protocol shape


def test_store_satisfies_every_protocol(store: Store) -> None:
    assert isinstance(store, Store)
    assert isinstance(store.graphs, GraphRepository)
    assert isinstance(store.patches, PatchRepository)
    assert isinstance(store.rehearsals, RehearsalRepository)
    assert isinstance(store.reports, ReportRepository)
    assert isinstance(store.audit, AuditLog)
    assert isinstance(store.signups, SignupRepository)
    assert isinstance(store.access, WorkspaceAccessRepository)
    assert isinstance(store.workspaces, WorkspaceRepository)


# --------------------------------------------------------------------------- workspaces


def test_create_workspace_derives_deterministic_id(make_store: StoreFactory) -> None:
    with closing(make_store()) as first, closing(make_store()) as second:
        a = first.workspaces.create_workspace("  결제팀  ")
        b = second.workspaces.create_workspace("결제팀")
    assert a == b
    assert a.id.startswith("ws-")
    assert a.name == "결제팀"
    assert a.created_at == T0


def test_create_workspace_with_explicit_id(store: Store) -> None:
    ws = store.workspaces.create_workspace("데모", workspace_id="dmc-shop")
    assert ws.id == "dmc-shop"
    assert store.workspaces.get_workspace("dmc-shop") == ws
    with pytest.raises(AlreadyExistsError, match="이미 사용 중"):
        store.workspaces.create_workspace("다른 이름", workspace_id="dmc-shop")
    assert store.workspaces.list_workspaces() == [ws]


def test_same_name_same_instant_gets_distinct_ids(make_store: StoreFactory) -> None:
    with closing(make_store(_fixed(T0))) as store:
        created = [store.workspaces.create_workspace("같은 이름") for _ in range(4)]
        assert len({w.id for w in created}) == 4
        assert store.workspaces.list_workspaces() == sorted(created, key=lambda w: w.id)


@pytest.mark.parametrize("name", ["", "   ", "x" * 201, "a\x00b"])
def test_invalid_workspace_name_rejected(store: Store, name: str) -> None:
    with pytest.raises(ValueError, match="워크스페이스 이름"):
        store.workspaces.create_workspace(name)


@pytest.mark.parametrize("workspace_id", ["", "has space", "../etc", "-leading", "x" * 65, "한글"])
def test_invalid_workspace_id_rejected(store: Store, workspace_id: str) -> None:
    with pytest.raises(ValueError, match="워크스페이스 id"):
        store.workspaces.create_workspace("이름", workspace_id=workspace_id)


def test_list_workspaces_in_creation_order(store: Store) -> None:
    names = ["다", "가", "나"]
    created = [store.workspaces.create_workspace(n) for n in names]
    assert store.workspaces.list_workspaces() == created
    assert store.workspaces.get_workspace("ws-missing") is None


def test_workspace_clock_must_be_timezone_aware(make_store: StoreFactory) -> None:
    with (
        closing(make_store(lambda: datetime(2026, 1, 1))) as store,
        pytest.raises(ValueError, match="시간대"),
    ):
        store.workspaces.create_workspace("팀")


def test_workspace_created_at_is_normalised_to_utc(make_store: StoreFactory) -> None:
    kst_noon = datetime(2026, 10, 13, 12, 0, tzinfo=KST)
    with closing(make_store(_fixed(kst_noon))) as store:
        ws = store.workspaces.create_workspace("팀")
        stored = store.workspaces.get_workspace(ws.id)
    assert stored is not None
    assert stored.created_at == kst_noon
    assert stored.created_at.utcoffset() == timedelta(0)


# --------------------------------------------------------------------------- documents


def test_documents_without_raw_text_keep_metadata_only(store: Store) -> None:
    store.documents.save_documents("ws-a", [RUNBOOK], store_raw=False)
    [meta] = store.documents.list_documents("ws-a")
    assert meta.doc_id == RUNBOOK.doc_id
    assert meta.title == RUNBOOK.title
    assert meta.sha256 == RUNBOOK.sha256
    assert meta.media_type == RUNBOOK.media_type
    assert meta.quarantined == RUNBOOK.quarantined
    assert meta.raw_stored is False
    assert meta.to_ref() == DocumentRef(doc_id=RUNBOOK.doc_id, title=RUNBOOK.title, sha256=RUNBOOK.sha256)
    assert store.documents.get_document("ws-a", RUNBOOK.doc_id) is None


def test_documents_with_raw_text_roundtrip(store: Store) -> None:
    doc = RUNBOOK.model_copy(update={"path": "runbooks/pay.md", "media_type": "text/markdown"})
    store.documents.save_documents("ws-a", [doc], store_raw=True)
    loaded = store.documents.get_document("ws-a", doc.doc_id)
    assert loaded == doc
    assert loaded is not None
    assert loaded.sha256 == doc.sha256
    assert store.documents.list_documents("ws-a")[0].raw_stored is True


def test_resaving_without_raw_text_erases_stored_text(store: Store) -> None:
    store.documents.save_documents("ws-a", [RUNBOOK], store_raw=True)
    store.documents.save_documents("ws-a", [RUNBOOK], store_raw=False)
    assert store.documents.get_document("ws-a", RUNBOOK.doc_id) is None
    assert [d.raw_stored for d in store.documents.list_documents("ws-a")] == [False]


def test_documents_are_scoped_and_sorted(store: Store) -> None:
    b = Document(doc_id="doc-b", title="나 문서", text="둘")
    a2 = Document(doc_id="doc-a2", title="가 문서", text="하나 둘")
    a1 = Document(doc_id="doc-a1", title="가 문서", text="하나")
    store.documents.save_documents("ws-a", [b, a2, a1], store_raw=False)
    store.documents.save_documents("ws-b", [RUNBOOK], store_raw=True)
    assert [d.doc_id for d in store.documents.list_documents("ws-a")] == ["doc-a1", "doc-a2", "doc-b"]
    assert [d.doc_id for d in store.documents.list_documents("ws-b")] == [RUNBOOK.doc_id]
    assert store.documents.get_document("ws-a", RUNBOOK.doc_id) is None
    assert store.documents.list_documents("ws-none") == []


def test_document_batch_validation(store: Store) -> None:
    with pytest.raises(ValueError, match="두 번"):
        store.documents.save_documents("ws-a", [RUNBOOK, RUNBOOK], store_raw=True)
    with pytest.raises(ValueError, match="NUL"):
        store.documents.save_documents(
            "ws-a", [Document(doc_id="d", title="t", text="a\x00b")], store_raw=False
        )
    assert store.documents.list_documents("ws-a") == []


# --------------------------------------------------------------------------- inputs


def _inputs() -> StoredInputs:
    return StoredInputs(
        roster=(
            RosterEntry(
                name="김민수",
                title="대리",
                team="결제팀",
                roles=("1차 당직",),
                aliases=("김대리",),
                channels=(ChannelMedium.SLACK, ChannelMedium.PHONE),
            ),
            RosterEntry(name="이영희", status=PersonStatus.DEPARTED, left_on="2026-08-31"),
        ),
        oncall=(
            OnCallShift(person="김민수", start=T0, end=T0 + timedelta(hours=12)),
            OnCallShift(person="박지영", start=T0, end=T0 + timedelta(hours=12), level=2),
        ),
        catalog=(CatalogService(name="결제 서비스", tier=1, depends_on=("pg-main",), location="IDC-가산"),),
        policy=Policy(tier_slo_minutes={1: 30, 2: 120}, regulated_services=("결제 서비스",)),
    )


def test_inputs_default_to_empty(store: Store) -> None:
    assert store.inputs.load_inputs("ws-a") == StoredInputs()


def test_inputs_roundtrip_and_overwrite(store: Store) -> None:
    inputs = _inputs()
    store.inputs.save_inputs("ws-a", inputs)
    assert store.inputs.load_inputs("ws-a") == inputs
    smaller = StoredInputs(roster=inputs.roster[:1])
    store.inputs.save_inputs("ws-a", smaller)
    assert store.inputs.load_inputs("ws-a") == smaller
    assert store.inputs.load_inputs("ws-b") == StoredInputs()


def test_inputs_bundle_conversion() -> None:
    inputs = _inputs()
    bundle = inputs.to_bundle([RUNBOOK])
    assert bundle == InputBundle(
        documents=(RUNBOOK,),
        roster=inputs.roster,
        oncall=inputs.oncall,
        catalog=inputs.catalog,
        policy=inputs.policy,
    )
    assert StoredInputs.from_bundle(bundle) == inputs


# --------------------------------------------------------------------------- graph versions


def test_empty_graph_history(store: Store) -> None:
    assert store.graphs.head("ws-a") is None
    assert store.graphs.list_versions("ws-a") == []
    with pytest.raises(NotFoundError, match="그래프 버전 1"):
        store.graphs.get_version("ws-a", 1)
    with pytest.raises(KeyError):
        store.graphs.load_graph("0" * 64)


def test_missing_records_raise_the_domain_port_error(store: Store) -> None:
    """``improve``/``orchestrator`` only know the domain ports, so that is what they catch."""
    lookups: list[Callable[[], object]] = [
        lambda: store.graphs.get_version("ws-a", 1),
        lambda: store.graphs.load_graph("0" * 64),
        lambda: store.patches.get_patch("patch-404"),
    ]
    for lookup in lookups:
        with pytest.raises(ports.NotFoundError) as caught:
            lookup()
        message = str(caught.value)
        assert message == caught.value.args[0]
        assert not message.startswith("'")  # readable Korean, not KeyError's repr


def test_graph_versions_append_head_list_load(store: Store) -> None:
    g1, g2 = graph_v1(), graph_v2()
    v1 = version_of("ws-a", 1, g1)
    v2 = version_of("ws-a", 2, g2, created_at=T0 + timedelta(minutes=5), patch_id="patch-1")
    store.graphs.append_version(v1, g1, expected_head=None)
    store.graphs.append_version(v2, g2, expected_head=1)

    assert store.graphs.head("ws-a") == v2
    assert store.graphs.list_versions("ws-a") == [v1, v2]
    assert store.graphs.get_version("ws-a", 1) == v1
    loaded = store.graphs.load_graph(v2.graph_hash)
    assert loaded == g2
    assert loaded.content_hash() == g2.content_hash() == v2.graph_hash
    assert store.graphs.head("ws-b") is None


def test_append_detects_version_conflict(store: Store) -> None:
    g1, g2 = graph_v1(), graph_v2()
    store.graphs.append_version(version_of("ws-a", 1, g1), g1, expected_head=None)
    with pytest.raises(VersionConflictError, match="버전 충돌"):
        store.graphs.append_version(version_of("ws-a", 1, g2), g2, expected_head=None)
    with pytest.raises(VersionConflictError):
        store.graphs.append_version(version_of("ws-a", 4, g2), g2, expected_head=3)
    assert [v.number for v in store.graphs.list_versions("ws-a")] == [1]


def test_append_rejects_inconsistent_versions(store: Store) -> None:
    g1, g2 = graph_v1(), graph_v2()
    with pytest.raises(ValueError, match="graph_hash"):
        store.graphs.append_version(version_of("ws-a", 1, g1), g2, expected_head=None)
    with pytest.raises(ValueError, match="버전 번호"):
        store.graphs.append_version(version_of("ws-a", 2, g1), g1, expected_head=None)
    naive = version_of("ws-a", 1, g1, created_at=datetime(2026, 1, 1))
    with pytest.raises(ValueError, match="시간대"):
        store.graphs.append_version(naive, g1, expected_head=None)
    assert store.graphs.list_versions("ws-a") == []


def test_graph_blobs_are_shared_and_versions_scoped(store: Store) -> None:
    g1 = graph_v1()
    store.graphs.append_version(version_of("ws-a", 1, g1), g1, expected_head=None)
    store.graphs.append_version(version_of("ws-b", 1, g1), g1, expected_head=None)
    assert store.graphs.load_graph(g1.content_hash()) == g1
    assert [v.workspace_id for v in store.graphs.list_versions("ws-b")] == ["ws-b"]


def test_version_timestamp_normalised_to_utc(store: Store) -> None:
    g1 = graph_v1()
    kst = datetime(2026, 10, 13, 18, 0, tzinfo=KST)
    store.graphs.append_version(version_of("ws-a", 1, g1, created_at=kst), g1, expected_head=None)
    head = store.graphs.head("ws-a")
    assert head is not None
    assert head.created_at == kst
    assert head.created_at.utcoffset() == timedelta(0)


def test_concurrent_appends_have_exactly_one_winner(store: Store) -> None:
    graph = graph_v1()
    outcomes = _race_appends(store, graph, writers=8)
    assert outcomes == Counter({"ok": 1, "conflict": 7})
    assert len(store.graphs.list_versions("ws-race")) == 1


def _race_appends(store: Store, graph: OpsGraph, *, writers: int) -> Counter[str]:
    barrier = threading.Barrier(writers)
    outcomes: Counter[str] = Counter()
    lock = threading.Lock()

    def append(i: int) -> None:
        version = version_of("ws-race", 1, graph, created_at=T0 + timedelta(seconds=i))
        barrier.wait()
        try:
            store.graphs.append_version(version, graph, expected_head=None)
            outcome = "ok"
        except VersionConflictError:
            outcome = "conflict"
        with lock:
            outcomes[outcome] += 1

    threads = [threading.Thread(target=append, args=(i,)) for i in range(writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    return outcomes


@HYPOTHESIS
@given(heads=st.lists(st.one_of(st.none(), st.integers(min_value=0, max_value=6)), max_size=10))
def test_append_accepts_exactly_the_current_head(make_store: StoreFactory, heads: list[int | None]) -> None:
    """Model check: an append succeeds iff ``expected_head`` equals the real head."""
    graph = graph_v1()
    with closing(make_store()) as store:
        current: int | None = None
        for expected in heads:
            number = 1 if expected is None else expected + 1
            version = version_of("ws-a", number, graph)
            if expected == current:
                store.graphs.append_version(version, graph, expected_head=expected)
                current = number
            else:
                with pytest.raises(VersionConflictError):
                    store.graphs.append_version(version, graph, expected_head=expected)
        head = store.graphs.head("ws-a")
        assert (head.number if head else None) == current
        assert [v.number for v in store.graphs.list_versions("ws-a")] == list(range(1, (current or 0) + 1))


_names = st.text(alphabet="결제서비스김민수abcXYZ-_ 0123456789", min_size=1, max_size=12)
_attr_values = st.one_of(
    st.text(max_size=20),
    st.integers(min_value=-(2**53), max_value=2**53),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
    st.lists(st.text(max_size=8), max_size=3),
)


@st.composite
def ops_graphs(draw: st.DrawFn) -> OpsGraph:
    builder = GraphBuilder()
    services = [node_id(NodeKind.SERVICE, n) for n in draw(st.lists(_names, min_size=1, max_size=4))]
    people = [node_id(NodeKind.PERSON, n) for n in draw(st.lists(_names, min_size=1, max_size=4))]
    for nid in [*services, *people]:
        kind, _, name = nid.partition(":")
        attrs = draw(
            st.dictionaries(st.sampled_from(["tier", "title", "x", "note"]), _attr_values, max_size=3)
        )
        builder.add_node(
            Node(
                id=nid,
                kind=NodeKind(kind),
                name=name,
                attrs={k: Attr(value=v, source=SourceKind.CATALOG) for k, v in attrs.items()},
                prov=(Provenance(source=SourceKind.CATALOG, extractor=Extractor.IMPORT, ref=name),),
            )
        )
    for src in services:
        for dst in draw(st.lists(st.sampled_from(people), unique=True, max_size=len(people))):
            builder.add_edge(
                Edge(
                    id=edge_id(EdgeKind.OWNED_BY, src, dst),
                    kind=EdgeKind.OWNED_BY,
                    src=src,
                    dst=dst,
                    prov=(Provenance(source=SourceKind.CATALOG, extractor=Extractor.IMPORT),),
                )
            )
    return builder.build()


@HYPOTHESIS
@given(graph=ops_graphs())
def test_graph_roundtrip_preserves_content_hash(make_store: StoreFactory, graph: OpsGraph) -> None:
    with closing(make_store()) as store:
        store.graphs.append_version(version_of("ws-a", 1, graph), graph, expected_head=None)
        loaded = store.graphs.load_graph(graph.content_hash())
    assert loaded.content_hash() == graph.content_hash()
    assert loaded == graph


def test_decode_graph_detects_tampering() -> None:
    graph = graph_v1()
    tampered = graph.canonical_json().replace("김민수", "홍길동")
    assert decode_graph(graph.content_hash(), graph.canonical_json()) == graph
    with pytest.raises(StoreIntegrityError, match="손상"):
        decode_graph(graph.content_hash(), tampered)


# --------------------------------------------------------------------------- patches


def test_patch_roundtrip_and_missing(store: Store) -> None:
    patch = make_patch("patch-1")
    store.patches.save_patch(patch)
    assert store.patches.get_patch("patch-1") == patch
    with pytest.raises(NotFoundError, match="patch-404"):
        store.patches.get_patch("patch-404")


def test_patch_save_is_upsert(store: Store) -> None:
    draft = make_patch("patch-1")
    approved = draft.model_copy(
        update={"status": PatchStatus.APPROVED, "decided_by": "lead", "decided_at": T0 + timedelta(hours=1)}
    )
    store.patches.save_patch(draft)
    store.patches.save_patch(approved)
    assert store.patches.get_patch("patch-1") == approved
    assert store.patches.list_patches("ws-a") == [approved]


def test_list_patches_order_and_scope(store: Store) -> None:
    later = make_patch("p-a", created_at=T0 + timedelta(minutes=1))
    tie_b = make_patch("p-c", created_at=T0)
    tie_a = make_patch("p-b", created_at=T0)
    undated = make_patch("p-z", created_at=None)
    other = make_patch("p-0", workspace_id="ws-b")
    for patch in (later, tie_b, other, undated, tie_a):
        store.patches.save_patch(patch)
    assert [p.id for p in store.patches.list_patches("ws-a")] == ["p-z", "p-b", "p-c", "p-a"]
    assert [p.id for p in store.patches.list_patches("ws-b")] == ["p-0"]
    assert store.patches.list_patches("ws-none") == []


def test_patch_naive_timestamp_rejected(store: Store) -> None:
    with pytest.raises(ValueError, match="시간대"):
        store.patches.save_patch(make_patch("p", created_at=datetime(2026, 1, 1)))


# --------------------------------------------------------------------------- rehearsals


def test_scenarios_upsert_sorted_and_scoped(store: Store) -> None:
    second, first = make_scenario("sc-b"), make_scenario("sc-a")
    store.rehearsals.save_scenario("ws-a", second)
    store.rehearsals.save_scenario("ws-a", first)
    renamed = first.model_copy(update={"title": "담당자 부재(수정)"})
    store.rehearsals.save_scenario("ws-a", renamed)
    store.rehearsals.save_scenario("ws-b", second)
    assert store.rehearsals.list_scenarios("ws-a") == [renamed, second]
    assert store.rehearsals.list_scenarios("ws-b") == [second]


def test_results_filtered_by_graph_hash(store: Store) -> None:
    h1, h2 = graph_v1().content_hash(), graph_v2().content_hash()
    sc_a, sc_b = make_scenario("sc-a"), make_scenario("sc-b")
    r_b1, r_a1, r_a2 = make_result(sc_b, h1), make_result(sc_a, h1), make_result(sc_a, h2)
    for result in (r_b1, r_a2, r_a1):
        store.rehearsals.save_result("ws-a", result)
    store.rehearsals.save_result("ws-b", r_a1)

    assert store.rehearsals.list_results("ws-a", h1) == [r_a1, r_b1]
    assert store.rehearsals.list_results("ws-a", h2) == [r_a2]
    assert store.rehearsals.list_results("ws-a") == sorted(
        [r_a1, r_a2, r_b1], key=lambda r: (r.scenario_id, r.graph_hash)
    )
    assert store.rehearsals.list_results("ws-b") == [r_a1]
    assert store.rehearsals.list_results("ws-a", "f" * 64) == []


def test_result_ids_are_deterministic_and_idempotent(store: Store) -> None:
    result = make_result(make_scenario(), graph_v1().content_hash())
    first = store.rehearsals.save_result("ws-a", result)
    second = store.rehearsals.save_result("ws-a", result)
    assert first == second == rehearsal_id("ws-a", result)
    assert first.startswith("rh-")
    assert store.rehearsals.list_results("ws-a") == [result]
    assert store.rehearsals.save_result("ws-b", result) != first
    changed = make_result(make_scenario(), graph_v1().content_hash(), config_hash="cfg-2")
    assert store.rehearsals.save_result("ws-a", changed) != first


# --------------------------------------------------------------------------- reports


def test_latest_report(store: Store) -> None:
    assert store.reports.latest_report("ws-a") is None
    newest = make_report("run-1", created_at=T0 + timedelta(hours=1))
    oldest = make_report("run-9", created_at=T0)
    store.reports.save_report(newest)
    store.reports.save_report(oldest)
    store.reports.save_report(make_report("run-x", workspace_id="ws-b", created_at=T0 + timedelta(days=1)))
    assert store.reports.latest_report("ws-a") == newest


def test_latest_report_tie_breaks_on_run_id(store: Store) -> None:
    kst_same_instant = (T0 + timedelta(hours=1)).astimezone(KST)
    store.reports.save_report(make_report("run-b", created_at=T0 + timedelta(hours=1)))
    store.reports.save_report(make_report("run-a", created_at=kst_same_instant))
    latest = store.reports.latest_report("ws-a")
    assert latest is not None
    assert latest.run_id == "run-b"


def test_report_requires_aware_timestamp(store: Store) -> None:
    with pytest.raises(ValueError, match="시간대"):
        store.reports.save_report(make_report("run-1", created_at=datetime(2026, 1, 1)))


# --------------------------------------------------------------------------- audit log


class _Kind(StrEnum):
    APPROVE = "approve"


def test_audit_log_is_ordered_scoped_and_json_normalised(store: Store) -> None:
    store.audit.record(
        "ws-a", "lead", "patch.approve", {"patch": "p-1", "ids": ("a", "b"), "kind": _Kind.APPROVE}
    )
    store.audit.record("ws-b", "admin", "workspace.create", {})
    store.audit.record("ws-a", "lead", "version.revert", {"at": T0, "to": 1})

    entries = store.audit.entries("ws-a")
    assert [e.action for e in entries] == ["patch.approve", "version.revert"]
    assert entries[0].seq < entries[1].seq
    assert entries[0].payload == {"ids": ["a", "b"], "kind": "approve", "patch": "p-1"}
    assert entries[1].payload == {"at": "2026-10-13T09:00:00Z", "to": 1}
    assert [e.recorded_at for e in entries] == [T0, T0 + timedelta(seconds=2)]
    assert [e.workspace_id for e in store.audit.entries("ws-b")] == ["ws-b"]
    assert store.audit.entries("ws-none") == []


@pytest.mark.parametrize(
    ("actor", "action", "payload", "message"),
    [
        ("lead", "x", {"bad": object()}, "JSON"),
        ("lead", "x", {"nan": math.nan}, "JSON"),
        ("  ", "x", {}, "행위자"),
        ("lead", "", {}, "행위자"),
    ],
)
def test_audit_log_rejects_invalid_records(
    store: Store, actor: str, action: str, payload: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        store.audit.record("ws-a", actor, action, payload)
    assert store.audit.entries("ws-a") == []


_json_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**53), max_value=2**53),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=10),
)
_json_values = st.recursive(
    _json_scalars,
    lambda inner: st.lists(inner, max_size=3) | st.dictionaries(st.text(max_size=5), inner, max_size=3),
    max_leaves=10,
)


@HYPOTHESIS
@given(payloads=st.lists(st.dictionaries(st.text(max_size=6), _json_values, max_size=4), max_size=4))
def test_audit_payloads_roundtrip(make_store: StoreFactory, payloads: list[dict[str, object]]) -> None:
    with closing(make_store()) as store:
        for i, payload in enumerate(payloads):
            store.audit.record("ws-a", "tester", f"action-{i}", payload)
        entries = store.audit.entries("ws-a")
    assert [e.payload for e in entries] == payloads
    assert [e.action for e in entries] == [f"action-{i}" for i in range(len(payloads))]


# --------------------------------------------------------------------------- sign-ups & events


def test_signup_requires_privacy_consent() -> None:
    with pytest.raises(ValidationError, match="동의"):
        Signup(email="a@b.co", consent_privacy=False, created_at=T0)


@pytest.mark.parametrize("email", ["", "no-at-sign", "a@b", "a b@c.de", "a@b.c\x00"])
def test_signup_rejects_invalid_email(email: str) -> None:
    with pytest.raises(ValidationError, match="이메일"):
        Signup(email=email, consent_privacy=True, created_at=T0)


def test_signup_normalisation() -> None:
    signup = Signup.model_validate(
        {
            "email": "  Kim@Example.COM ",
            "team_size": 12,
            "consent_privacy": True,
            "created_at": T0.astimezone(KST),
        }
    )
    assert signup.email == "kim@example.com"
    assert signup.team_size == "12"
    assert signup.created_at.utcoffset() == timedelta(0)
    with pytest.raises(ValidationError, match="시간대"):
        Signup(email="a@b.co", consent_privacy=True, created_at=datetime(2026, 1, 1))


def test_signups_upsert_by_email(store: Store) -> None:
    assert store.signups.count_signups() == 0
    first = make_signup("Kim@Example.com", marketing=True)
    store.signups.add_signup(first)
    assert store.signups.get_signup("KIM@example.com") == first
    withdrawn = make_signup("kim@example.com", marketing=False)
    store.signups.add_signup(withdrawn)
    store.signups.add_signup(make_signup("lee@example.com"))
    assert store.signups.count_signups() == 2
    assert store.signups.get_signup("kim@example.com") == withdrawn
    assert store.signups.get_signup("nobody@example.com") is None


def test_list_signups_by_creation_time_then_email(store: Store) -> None:
    assert store.signups.list_signups() == []
    later = make_signup("an@example.com").model_copy(update={"created_at": T0 + timedelta(hours=1)})
    store.signups.add_signup(later)
    store.signups.add_signup(make_signup("lee@example.com"))
    store.signups.add_signup(make_signup("Kim@Example.com", marketing=True))
    # a re-submission replaces the person's record (one entry, latest answers)
    store.signups.add_signup(make_signup("kim@example.com", marketing=False))
    listed = store.signups.list_signups()
    assert [s.email for s in listed] == ["kim@example.com", "lee@example.com", "an@example.com"]
    assert listed[0].consent_marketing is False
    assert len(listed) == store.signups.count_signups()


# --------------------------------------------------------------------------- workspace access keys

_DIGEST_A = "a" * 64
_DIGEST_B = "0123456789abcdef" * 4


def test_access_key_digests_roundtrip_replace_and_scope(store: Store) -> None:
    assert store.access.access_key_digest("ws-a") is None
    store.access.set_access_key_digest("ws-a", _DIGEST_A)
    store.access.set_access_key_digest("ws-b", _DIGEST_B)
    assert store.access.access_key_digest("ws-a") == _DIGEST_A
    assert store.access.access_key_digest("ws-b") == _DIGEST_B
    store.access.set_access_key_digest("ws-a", _DIGEST_B)  # a new key replaces the old one
    assert store.access.access_key_digest("ws-a") == _DIGEST_B
    assert store.access.access_key_digest("ws-c") is None


@pytest.mark.parametrize(
    ("workspace_id", "digest", "message"),
    [
        ("ws-a", "plaintext-key-that-must-never-be-stored", "다이제스트"),
        ("ws-a", "A" * 64, "다이제스트"),
        ("ws-a", "a" * 63, "다이제스트"),
        ("ws-a", "a" * 65, "다이제스트"),
        ("ws\x00a", _DIGEST_A, "NUL"),
    ],
)
def test_access_keys_store_only_sha256_digests(
    store: Store, workspace_id: str, digest: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        store.access.set_access_key_digest(workspace_id, digest)
    assert store.access.access_key_digest(workspace_id) is None


def test_event_counts(store: Store) -> None:
    assert store.events.event_counts() == {}
    for name in ["page_view", "signup.submit", "page_view", "demo:start", "page_view"]:
        store.events.record_event(name, {"path": "/", "ref": None}, T0)
    assert store.events.event_counts() == {"demo:start": 1, "page_view": 3, "signup.submit": 1}
    assert list(store.events.event_counts()) == sorted(store.events.event_counts())


@pytest.mark.parametrize(
    ("name", "props", "ts", "message"),
    [
        ("", {}, T0, "이벤트 이름"),
        ("has space", {}, T0, "이벤트 이름"),
        ("x" * 65, {}, T0, "이벤트 이름"),
        ("ok", {"bad": object()}, T0, "JSON"),
        ("ok", {}, datetime(2026, 1, 1), "시간대"),
    ],
)
def test_invalid_events_rejected(
    store: Store, name: str, props: dict[str, object], ts: datetime, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        store.events.record_event(name, props, ts)
    assert store.events.event_counts() == {}


@HYPOTHESIS
@given(names=st.lists(st.sampled_from(["view", "signup", "demo.start", "cta:click"]), max_size=20))
def test_event_counts_match_recorded_events(make_store: StoreFactory, names: list[str]) -> None:
    with closing(make_store()) as store:
        for i, name in enumerate(names):
            store.events.record_event(name, {"i": i}, T0 + timedelta(seconds=i))
        counts = store.events.event_counts()
    assert counts == dict(sorted(Counter(names).items()))


# --------------------------------------------------------------------------- determinism


_PATCHES = [
    make_patch("p-1", created_at=T0),
    make_patch("p-2", created_at=T0),
    make_patch("p-3", created_at=None),
    make_patch("p-4", created_at=T0 - timedelta(days=1)),
    make_patch("p-5", created_at=(T0 + timedelta(hours=2)).astimezone(KST)),
]
_SCENARIOS = [make_scenario(f"sc-{c}") for c in "dbeac"]
_DOCS = [Document(doc_id=f"doc-{i}", title=t, text=t * 2) for i, t in enumerate(["다", "가", "나", "가"])]


@HYPOTHESIS
@given(
    patch_order=st.permutations(range(len(_PATCHES))),
    scenario_order=st.permutations(range(len(_SCENARIOS))),
    doc_order=st.permutations(range(len(_DOCS))),
)
def test_listing_order_is_independent_of_insertion_order(
    make_store: StoreFactory, patch_order: list[int], scenario_order: list[int], doc_order: list[int]
) -> None:
    def fill(store: Store, patches: list[int], scenarios: list[int], docs: list[int]) -> object:
        for i in patches:
            store.patches.save_patch(_PATCHES[i])
        for i in scenarios:
            store.rehearsals.save_scenario("ws-a", _SCENARIOS[i])
        store.documents.save_documents("ws-a", [_DOCS[i] for i in docs], store_raw=False)
        return (
            _dump(store.patches.list_patches("ws-a")),
            _dump(store.rehearsals.list_scenarios("ws-a")),
            _dump(store.documents.list_documents("ws-a")),
        )

    natural = list(range(10))
    with closing(make_store()) as canonical, closing(make_store()) as shuffled:
        expected = fill(
            canonical, natural[: len(_PATCHES)], natural[: len(_SCENARIOS)], natural[: len(_DOCS)]
        )
        assert fill(shuffled, patch_order, scenario_order, doc_order) == expected


def _exercise(store: Store) -> list[object]:
    """A representative session; returns every observable output for cross-backend comparison."""
    g1, g2 = graph_v1(), graph_v2()
    ws = store.workspaces.create_workspace("결제팀")
    store.workspaces.create_workspace("데모", workspace_id="demo")
    store.documents.save_documents(ws.id, [RUNBOOK, _DOCS[0]], store_raw=True)
    store.inputs.save_inputs(ws.id, _inputs())
    store.graphs.append_version(version_of(ws.id, 1, g1), g1, expected_head=None)
    store.graphs.append_version(version_of(ws.id, 2, g2, patch_id="p-1"), g2, expected_head=1)
    for patch in _PATCHES:
        store.patches.save_patch(patch.model_copy(update={"workspace_id": ws.id}))
    for scenario in _SCENARIOS:
        store.rehearsals.save_scenario(ws.id, scenario)
    rids = [store.rehearsals.save_result(ws.id, make_result(sc, g1.content_hash())) for sc in _SCENARIOS]
    store.reports.save_report(make_report("run-1", ws.id))
    store.audit.record(ws.id, "lead", "patch.approve", {"patch": "p-1"})
    store.signups.add_signup(make_signup())
    store.signups.add_signup(make_signup("lee@example.com", marketing=False))
    store.events.record_event("view", {"path": "/"}, T0)
    store.access.set_access_key_digest(ws.id, _DIGEST_A)
    return [
        _dump(store.workspaces.list_workspaces()),
        _dump(store.documents.list_documents(ws.id)),
        _dump(store.documents.get_document(ws.id, RUNBOOK.doc_id)),
        _dump(store.inputs.load_inputs(ws.id)),
        _dump(store.graphs.list_versions(ws.id)),
        store.graphs.load_graph(g2.content_hash()).canonical_json(),
        _dump(store.patches.list_patches(ws.id)),
        _dump(store.rehearsals.list_scenarios(ws.id)),
        rids,
        _dump(store.rehearsals.list_results(ws.id, g1.content_hash())),
        _dump(store.reports.latest_report(ws.id)),
        _dump(store.audit.entries(ws.id)),
        _dump(store.signups.get_signup("kim@example.com")),
        store.signups.count_signups(),
        _dump(store.signups.list_signups()),
        store.events.event_counts(),
        store.access.access_key_digest(ws.id),
        store.access.access_key_digest("demo"),
    ]


def test_backends_are_observably_identical() -> None:
    with (
        closing(MemoryStore(clock=TickingClock())) as memory,
        closing(SqlStore(MEMORY_SQLITE, clock=TickingClock())) as sql,
    ):
        assert _exercise(memory) == _exercise(sql)


# --------------------------------------------------------------------------- backend specifics


def test_open_store_selects_backend(tmp_path: Path) -> None:
    with closing(open_store(MEMORY_URL)) as memory:
        assert isinstance(memory, MemoryStore)
    url = f"sqlite:///{(tmp_path / 'nested' / 'dir' / 'jettae.db').as_posix()}"
    with closing(open_store(url, clock=TickingClock())) as sql:
        assert isinstance(sql, SqlStore)
        sql.workspaces.create_workspace("팀", workspace_id="team")
    assert (tmp_path / "nested" / "dir" / "jettae.db").is_file()


def test_sql_file_store_persists_across_reopen(tmp_path: Path) -> None:
    url = f"sqlite:///{(tmp_path / 'jettae.db').as_posix()}"
    graph = graph_v1()
    with closing(SqlStore(url, clock=TickingClock())) as first:
        first.workspaces.create_workspace("팀", workspace_id="team")
        first.graphs.append_version(version_of("team", 1, graph), graph, expected_head=None)
    with closing(SqlStore(url, clock=TickingClock())) as reopened:  # create_all is idempotent
        assert [w.id for w in reopened.workspaces.list_workspaces()] == ["team"]
        assert reopened.graphs.load_graph(graph.content_hash()) == graph


def test_sql_memory_url_uses_static_pool() -> None:
    assert is_memory_sqlite(MEMORY_SQLITE)
    assert is_memory_sqlite("sqlite://")
    assert not is_memory_sqlite("sqlite:///./data/jettae.db")
    assert not is_memory_sqlite("postgresql+psycopg://u@h/db")
    with closing(SqlStore(MEMORY_SQLITE)) as store:
        assert isinstance(store.engine.pool, StaticPool)


def test_sql_store_leaves_caller_engine_open() -> None:
    engine = create_engine(MEMORY_SQLITE, poolclass=StaticPool, connect_args={"check_same_thread": False})
    try:
        SqlStore(engine, clock=TickingClock()).workspaces.create_workspace("팀", workspace_id="team")
        store = SqlStore(engine, clock=TickingClock())
        store.close()
        assert [w.id for w in SqlStore(engine, create_schema=False).workspaces.list_workspaces()] == ["team"]
    finally:
        engine.dispose()


def test_sql_concurrent_appends_on_file_database(tmp_path: Path) -> None:
    url = f"sqlite:///{(tmp_path / 'race.db').as_posix()}"
    with closing(SqlStore(url, clock=TickingClock())) as store:
        outcomes = _race_appends(store, graph_v1(), writers=6)
        assert outcomes == Counter({"ok": 1, "conflict": 5})


def test_sql_primary_key_guards_a_lost_race(monkeypatch: pytest.MonkeyPatch) -> None:
    """A writer whose head check read stale data must still lose: the (workspace, number)
    primary key rejects the insert and the store reports a conflict, not a database error."""
    graph = graph_v1()
    with closing(SqlStore(MEMORY_SQLITE, clock=TickingClock())) as store:
        store.graphs.append_version(version_of("ws-a", 1, graph), graph, expected_head=None)
        stale_reads = iter([None])
        real_head = SqlGraphRepository._head_number

        def head_number(session: Session, workspace_id: str) -> int | None:
            return next(stale_reads, real_head(session, workspace_id))

        monkeypatch.setattr(SqlGraphRepository, "_head_number", staticmethod(head_number))
        late = version_of("ws-a", 1, graph, created_at=T0 + timedelta(minutes=1))
        with pytest.raises(VersionConflictError, match="현재 최신 버전은 1"):
            store.graphs.append_version(late, graph, expected_head=None)
        assert store.graphs.list_versions("ws-a") == [version_of("ws-a", 1, graph)]


def test_sql_detects_tampered_graph_blob() -> None:
    graph = graph_v1()
    with closing(SqlStore(MEMORY_SQLITE, clock=TickingClock())) as store:
        store.graphs.append_version(version_of("ws-a", 1, graph), graph, expected_head=None)
        forged = json.loads(graph.canonical_json())
        forged["nodes"][PERSON_ID]["name"] = "홍길동"
        with store.engine.begin() as conn:
            conn.execute(
                text("UPDATE graph_blobs SET canonical_json = :body WHERE graph_hash = :h"),
                {"body": json.dumps(forged, ensure_ascii=False, sort_keys=True), "h": graph.content_hash()},
            )
        with pytest.raises(StoreIntegrityError):
            store.graphs.load_graph(graph.content_hash())


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "postgres://u:p%40ss@db.example:5432/jettae",
            "postgresql+psycopg://u:p%40ss@db.example:5432/jettae",
        ),
        (
            "postgresql://u:pw@db.example/jettae?sslmode=require",
            "postgresql+psycopg://u:pw@db.example/jettae?sslmode=require",
        ),
        ("postgresql+psycopg://u:pw@db.example/jettae", "postgresql+psycopg://u:pw@db.example/jettae"),
        ("postgresql+psycopg2://u:pw@db.example/jettae", "postgresql+psycopg2://u:pw@db.example/jettae"),
        ("postgresql+asyncpg://u:pw@db.example/jettae", "postgresql+asyncpg://u:pw@db.example/jettae"),
        ("sqlite:///./data/jettae.db", "sqlite:///./data/jettae.db"),
        ("sqlite+pysqlite:///:memory:", "sqlite+pysqlite:///:memory:"),
    ],
)
def test_driverless_postgres_urls_are_pointed_at_psycopg(url: str, expected: str) -> None:
    """Render and Heroku inject ``postgres://``/``postgresql://`` URLs; SQLAlchemy 2 does not know
    the former and would pick psycopg2 (not installed) for the latter."""
    normalized = normalize_database_url(url)
    assert normalized == make_url(expected)
    assert normalized.password == make_url(url).password


def test_build_engine_uses_the_normalised_postgres_driver() -> None:
    pytest.importorskip("psycopg")
    engine = build_engine("postgres://jettae:secret@db.invalid:5432/jettae")
    try:
        assert engine.url.drivername == "postgresql+psycopg"
        assert engine.dialect.driver == "psycopg"
    finally:
        engine.dispose()


def test_build_engine_creates_sqlite_parent_directory(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "x.db"
    engine = build_engine(f"sqlite:///{target.as_posix()}")
    try:
        assert target.parent.is_dir()
    finally:
        engine.dispose()


# --------------------------------------------------------------------------- adversarial edge cases
# Added in review: each test pins a defect that both backends had (or could have had) and that
# the suite above did not catch.


def test_patch_id_owned_by_another_workspace_is_not_rehomed(store: Store) -> None:
    """Deterministic patch ids can collide between workspaces that diagnose identical inputs;
    a silent upsert would move ws-a's patch into ws-b and ws-a would lose it."""
    mine = make_patch("p-shared", "ws-a")
    store.patches.save_patch(mine)
    with pytest.raises(AlreadyExistsError, match="다른 워크스페이스"):
        store.patches.save_patch(make_patch("p-shared", "ws-b"))
    assert store.patches.list_patches("ws-a") == [mine]
    assert store.patches.list_patches("ws-b") == []
    assert store.patches.get_patch("p-shared") == mine
    # The owner itself may still update its patch.
    approved = mine.model_copy(update={"status": PatchStatus.APPROVED})
    store.patches.save_patch(approved)
    assert store.patches.get_patch("p-shared") == approved


def test_report_run_id_owned_by_another_workspace_is_not_rehomed(store: Store) -> None:
    mine = make_report("run-shared", "ws-a")
    store.reports.save_report(mine)
    with pytest.raises(AlreadyExistsError, match="다른 워크스페이스"):
        store.reports.save_report(make_report("run-shared", "ws-b", created_at=T0 + timedelta(hours=1)))
    assert store.reports.latest_report("ws-a") == mine
    assert store.reports.latest_report("ws-b") is None


def test_ownership_error_does_not_reveal_the_owning_workspace(store: Store) -> None:
    store.patches.save_patch(make_patch("p-x", "secret-tenant"))
    with pytest.raises(AlreadyExistsError) as caught:
        store.patches.save_patch(make_patch("p-x", "ws-b"))
    assert "secret-tenant" not in str(caught.value)


def _race_cross_workspace_saves(store: Store, writers: int) -> Counter[str]:
    barrier = threading.Barrier(writers)
    outcomes: Counter[str] = Counter()
    lock = threading.Lock()

    def save(i: int) -> None:
        patch = make_patch("p-race", f"ws-{i}")
        barrier.wait()
        try:
            store.patches.save_patch(patch)
            outcome = "ok"
        except AlreadyExistsError:
            outcome = "taken"
        with lock:
            outcomes[outcome] += 1

    threads = [threading.Thread(target=save, args=(i,)) for i in range(writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    return outcomes


def test_concurrent_cross_workspace_patch_saves_have_one_owner(store: Store) -> None:
    assert _race_cross_workspace_saves(store, writers=6) == Counter({"ok": 1, "taken": 5})
    owners = [ws for ws in (f"ws-{i}" for i in range(6)) if store.patches.list_patches(ws)]
    assert len(owners) == 1
    assert store.patches.get_patch("p-race").workspace_id == owners[0]


def test_sql_cross_workspace_race_on_file_database(tmp_path: Path) -> None:
    url = f"sqlite:///{(tmp_path / 'owner-race.db').as_posix()}"
    with closing(SqlStore(url, clock=TickingClock())) as store:
        assert _race_cross_workspace_saves(store, writers=5) == Counter({"ok": 1, "taken": 4})


def _with_scores(patch: Patch, score: float) -> Patch:
    assert patch.validation is not None
    validation = patch.validation.model_copy(update={"score_before": score})
    return patch.model_copy(update={"validation": validation})


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_floats_are_refused_by_every_backend(store: Store, bad: float) -> None:
    """Postgres JSON rejects NaN/Infinity; SQLite and memory used to keep them, so the same
    record was storable on one backend and not on another."""
    scenario = make_scenario().model_copy(update={"deadline_minutes": bad})
    result = make_result(make_scenario(), "h" * 64).model_copy(update={"deadline_minutes": bad})
    report = make_report("run-nan")
    bad_report = report.model_copy(update={"readiness": report.readiness.model_copy(update={"overall": bad})})
    fatigue = Policy().fatigue.model_copy(update={"min_rest_hours": bad})
    bad_inputs = StoredInputs(policy=Policy(fatigue=fatigue))
    with pytest.raises(ValueError, match="JSON"):
        store.patches.save_patch(_with_scores(make_patch("p-nan"), bad))
    with pytest.raises(ValueError, match="JSON"):
        store.rehearsals.save_scenario("ws-a", scenario)
    with pytest.raises(ValueError, match="JSON"):
        store.rehearsals.save_result("ws-a", result)
    with pytest.raises(ValueError, match="JSON"):
        store.reports.save_report(bad_report)
    with pytest.raises(ValueError, match="JSON"):
        store.inputs.save_inputs("ws-a", bad_inputs)
    assert store.patches.list_patches("ws-a") == []
    assert store.rehearsals.list_scenarios("ws-a") == []
    assert store.rehearsals.list_results("ws-a") == []
    assert store.reports.latest_report("ws-a") is None
    assert store.inputs.load_inputs("ws-a") == StoredInputs()


def test_patch_decided_at_must_be_timezone_aware(store: Store) -> None:
    naive = make_patch("p-d").model_copy(update={"decided_at": datetime(2026, 1, 1)})
    with pytest.raises(ValueError, match="결정 시각"):
        store.patches.save_patch(naive)
    assert store.patches.list_patches("ws-a") == []


@pytest.mark.parametrize("field", ["created_by", "note", "patch_id"])
def test_version_text_fields_refuse_nul(store: Store, field: str) -> None:
    graph = graph_v1()
    version = version_of("ws-a", 1, graph).model_copy(update={field: "a\x00b"})
    with pytest.raises(ValueError, match="NUL"):
        store.graphs.append_version(version, graph, expected_head=None)
    assert store.graphs.list_versions("ws-a") == []


def test_version_errors_speak_korean_not_python(store: Store) -> None:
    graph = graph_v1()
    with pytest.raises(VersionConflictError) as empty_history:
        store.graphs.append_version(version_of("ws-a", 4, graph), graph, expected_head=3)
    assert "아직 저장된 버전이 없습니다" in str(empty_history.value)
    store.graphs.append_version(version_of("ws-a", 1, graph), graph, expected_head=None)
    with pytest.raises(VersionConflictError) as stale:
        store.graphs.append_version(version_of("ws-a", 1, graph), graph, expected_head=None)
    assert "기준 버전(없음)" in str(stale.value)
    assert "현재 최신 버전은 1입니다" in str(stale.value)
    with pytest.raises(ValueError, match="기대 최신 버전: 없음") as wrong_number:
        store.graphs.append_version(version_of("ws-z", 3, graph), graph, expected_head=None)
    for message in (empty_history, stale, wrong_number):
        assert "None" not in str(message.value)


@pytest.mark.parametrize("garbage", ["", "{not json", '{"nodes": 1}', "null"])
def test_unparseable_graph_blob_is_an_integrity_error(garbage: str) -> None:
    with pytest.raises(StoreIntegrityError, match="손상"):
        decode_graph(graph_v1().content_hash(), garbage)


def test_sql_detects_garbage_graph_blob() -> None:
    graph = graph_v1()
    with closing(SqlStore(MEMORY_SQLITE, clock=TickingClock())) as store:
        store.graphs.append_version(version_of("ws-a", 1, graph), graph, expected_head=None)
        with store.engine.begin() as conn:
            conn.execute(text("UPDATE graph_blobs SET canonical_json = '{broken'"))
        with pytest.raises(StoreIntegrityError):
            store.graphs.load_graph(graph.content_hash())


def test_one_shot_iterable_of_documents_is_saved(store: Store) -> None:
    """Validation used to exhaust a generator, so nothing was saved and no error was raised."""
    docs = (d for d in [RUNBOOK, _DOCS[0]])
    store.documents.save_documents("ws-a", docs, store_raw=True)  # type: ignore[arg-type]
    assert {d.doc_id for d in store.documents.list_documents("ws-a")} == {RUNBOOK.doc_id, _DOCS[0].doc_id}


def test_document_ids_and_media_types_refuse_nul(store: Store) -> None:
    for doc in (
        Document(doc_id="d\x00", title="t", text="x"),
        Document(doc_id="d", title="t", text="x", media_type="text/\x00"),
    ):
        with pytest.raises(ValueError, match="NUL"):
            store.documents.save_documents("ws-a", [doc], store_raw=False)
    assert store.documents.list_documents("ws-a") == []


def test_empty_batches_and_empty_payloads(store: Store) -> None:
    store.documents.save_documents("ws-a", [], store_raw=True)
    assert store.documents.list_documents("ws-a") == []
    store.inputs.save_inputs("ws-a", StoredInputs())
    assert store.inputs.load_inputs("ws-a") == StoredInputs()
    store.audit.record("ws-a", "lead", "noop", {})
    assert [e.payload for e in store.audit.entries("ws-a")] == [{}]
    store.events.record_event("view", {}, T0)
    assert store.events.event_counts() == {"view": 1}
    assert store.rehearsals.list_scenarios("ws-a") == []
    assert store.rehearsals.list_results("ws-a") == []


def test_empty_text_is_stored_text_not_absent_text(store: Store) -> None:
    """P3: an empty stored body is a fact; "not stored" is absence. They must not collapse."""
    empty = Document(doc_id="doc-empty", title="빈 문서", text="")
    store.documents.save_documents("ws-a", [empty], store_raw=True)
    assert store.documents.get_document("ws-a", "doc-empty") == empty
    assert [d.raw_stored for d in store.documents.list_documents("ws-a")] == [True]


_KOREAN_TEXTS = [
    "결제 서비스 담당은 김민수 대리.",
    unicodedata.normalize("NFD", "결제 서비스 담당은 김민수 대리."),  # macOS-style decomposed jamo
    "줄1\r\n줄2\n\n탭\t끝",
    "제로폭\u200b문자\ufeff와 결합\u0301 문자",
    "이모지 \U0001f525\U0001f6a8 장애 \u00b7 \u300c인용\u300d \uff03전각",
    "ㄱㄴㄷ ㅏㅑ 한자 障害 対応",
]


@pytest.mark.parametrize("body", _KOREAN_TEXTS)
def test_korean_document_text_roundtrips_byte_for_byte(store: Store, body: str) -> None:
    """Span offsets index into the stored text, so the store must never normalise it."""
    doc = Document(doc_id="doc-k", title=body[:10], text=body, quarantined=((0, 2),))
    store.documents.save_documents("ws-a", [doc], store_raw=True)
    loaded = store.documents.get_document("ws-a", "doc-k")
    assert loaded is not None
    assert loaded.text.encode("utf-8") == body.encode("utf-8")
    assert loaded.sha256 == doc.sha256
    assert loaded == doc


_TEXT = st.characters(codec="utf-8", exclude_characters="\x00")


@HYPOTHESIS
@given(body=st.text(alphabet=_TEXT))
def test_any_document_text_roundtrips(make_store: StoreFactory, body: str) -> None:
    doc = Document(doc_id="doc-h", title="임의 문서", text=body)
    with closing(make_store()) as store:
        store.documents.save_documents("ws-a", [doc], store_raw=True)
        loaded = store.documents.get_document("ws-a", "doc-h")
        [meta] = store.documents.list_documents("ws-a")
    assert loaded == doc
    assert meta.sha256 == doc.sha256


def test_nfc_and_nfd_workspace_names_are_one_name(make_store: StoreFactory) -> None:
    nfc = "결제팀"
    nfd = unicodedata.normalize("NFD", nfc)
    assert nfc != nfd
    with closing(make_store()) as first, closing(make_store()) as second:
        a = first.workspaces.create_workspace(nfc)
        b = second.workspaces.create_workspace(nfd)
    assert a == b
    assert a.name == nfc


def test_nfd_name_is_measured_after_composition(store: Store) -> None:
    """200 syllables are 200 characters, even when they arrive as 400 decomposed jamo."""
    name = "가" * 200
    ws = store.workspaces.create_workspace(unicodedata.normalize("NFD", name))
    assert ws.name == name


@pytest.mark.parametrize(
    "name", ["\u3164", "\u3164\u3164", "\u200b", "\ufeff ", "\u115f\u1160", "\u3000", "\n\t", "\uffa0"]
)
def test_invisible_workspace_names_rejected(store: Store, name: str) -> None:
    with pytest.raises(ValueError, match="비어 있습니다"):
        store.workspaces.create_workspace(name)
    assert store.workspaces.list_workspaces() == []


_NAME_CHARS = st.characters(codec="utf-8").filter(lambda char: unicodedata.category(char) != "Cc")


def _without_padding(name: str) -> str:
    """Reference model of name trimming: whitespace, format characters and Hangul fillers."""

    def padding(char: str) -> bool:
        return char.isspace() or unicodedata.category(char) == "Cf" or char in "\u115f\u1160\u3164\uffa0"

    chars = list(unicodedata.normalize("NFC", name))
    while chars and padding(chars[0]):
        chars.pop(0)
    while chars and padding(chars[-1]):
        chars.pop()
    return "".join(chars)


@HYPOTHESIS
@given(
    name=st.text(alphabet=_NAME_CHARS, min_size=1, max_size=40).filter(lambda n: any(c.isalnum() for c in n))
)
def test_workspace_names_roundtrip_and_ids_agree_across_backends(name: str) -> None:
    with (
        closing(MemoryStore(clock=_fixed(T0))) as memory,
        closing(SqlStore(MEMORY_SQLITE, clock=_fixed(T0))) as sql,
    ):
        a = memory.workspaces.create_workspace(name)
        b = sql.workspaces.create_workspace(name)
        assert a == b
        assert sql.workspaces.get_workspace(b.id) == b
    assert a.name == _without_padding(name)


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        (
            "\uff4b\uff49\uff4d\uff20\uff45\uff58\uff41\uff4d\uff50\uff4c\uff45\uff0e\uff43\uff4f\uff4d",
            "kim@example.com",
        ),
        (unicodedata.normalize("NFD", "김@예시.한국"), "김@예시.한국"),
        ("  KIM@Example.COM\u3000", "kim@example.com"),
    ],
    ids=["full-width-ime", "nfd-idn", "ideographic-space"],
)
def test_email_variants_share_one_key(store: Store, raw: str, canonical: str) -> None:
    store.signups.add_signup(make_signup(canonical))
    store.signups.add_signup(make_signup(raw, marketing=False))
    assert store.signups.count_signups() == 1
    found = store.signups.get_signup(raw)
    assert found is not None
    assert found.email == canonical
    assert found.consent_marketing is False


def test_email_length_is_checked_after_case_folding() -> None:
    """``\u00df`` folds to ``ss``: 245 characters in, 305 out, past the 254-wide column."""
    email = "\u00df" * 60 + "@" + "e" * 180 + ".com"
    assert len(email) <= 254
    with pytest.raises(ValidationError, match="254자"):
        Signup(email=email, consent_privacy=True, created_at=T0)


def test_blank_optional_signup_answers_are_absent(store: Store) -> None:
    signup = Signup(
        email="a@b.co",
        organization="   ",
        team_size="",
        message="\u3000",
        consent_privacy=True,
        created_at=T0,
        source="",
    )
    assert (signup.organization, signup.team_size, signup.message, signup.source) == (None, None, None, None)
    store.signups.add_signup(signup)
    assert store.signups.get_signup("a@b.co") == signup


_TZS = st.sampled_from([UTC, KST, timezone(timedelta(hours=-5)), timezone(timedelta(hours=5, minutes=30))])


@HYPOTHESIS
@given(
    saves=st.lists(
        st.tuples(
            st.sampled_from(["p-1", "p-2", "p-3"]),
            st.sampled_from(["ws-a", "ws-b"]),
            st.one_of(st.none(), st.integers(min_value=-3, max_value=3)),
            _TZS,
        ),
        max_size=12,
    )
)
def test_patch_saves_match_a_reference_model(
    make_store: StoreFactory, saves: list[tuple[str, str, int | None, timezone]]
) -> None:
    """Model check of upsert, ownership and ordering, with timestamps in mixed zones."""
    model: dict[str, Patch] = {}
    with closing(make_store()) as store:
        for patch_id, workspace_id, minutes, tz in saves:
            created = None if minutes is None else (T0 + timedelta(minutes=minutes)).astimezone(tz)
            patch = make_patch(patch_id, workspace_id, created_at=created)
            owner = model.get(patch_id)
            if owner is not None and owner.workspace_id != workspace_id:
                with pytest.raises(AlreadyExistsError):
                    store.patches.save_patch(patch)
                continue
            store.patches.save_patch(patch)
            model[patch_id] = patch
        for workspace_id in ("ws-a", "ws-b"):
            expected = sorted(
                (p for p in model.values() if p.workspace_id == workspace_id),
                key=lambda p: (p.created_at is not None, p.created_at or T0, p.id),
            )
            assert _dump(store.patches.list_patches(workspace_id)) == _dump(expected)


@HYPOTHESIS
@given(
    reports=st.lists(
        st.tuples(st.sampled_from(["run-a", "run-b", "run-c"]), st.integers(min_value=0, max_value=3), _TZS),
        min_size=1,
        max_size=8,
    )
)
def test_latest_report_matches_a_reference_model(
    make_store: StoreFactory, reports: list[tuple[str, int, timezone]]
) -> None:
    model: dict[str, DiagnosisReport] = {}
    with closing(make_store()) as store:
        for run_id, hours, tz in reports:
            report = make_report(run_id, created_at=(T0 + timedelta(hours=hours)).astimezone(tz))
            store.reports.save_report(report)
            model[run_id] = report
        latest = store.reports.latest_report("ws-a")
    expected = max(model.values(), key=lambda r: (r.created_at, r.run_id))
    assert latest is not None
    assert latest == expected
    assert latest.run_id == expected.run_id


def test_same_backend_sessions_are_reproducible(make_store: StoreFactory) -> None:
    with closing(make_store()) as first, closing(make_store()) as second:
        assert _exercise(first) == _exercise(second)


# --------------------------------------------------------------------------- second review
# Most tests below failed against the implementation as first reviewed and pin the defects
# fixed in the second review pass; the others guard the behaviour those fixes must keep
# (valid reverts, inner characters of names, Korean IDN addresses, empty lookups).


def test_returned_records_do_not_alias_stored_state(store: Store) -> None:
    """Frozen models still hold mutable dicts/lists. The memory backend used to hand out the
    very objects it stored, so a caller could rewrite the append-only audit log or a saved
    report by mutating what it got back."""
    store.audit.record("ws-a", "lead", "patch.approve", {"nested": {"ids": ["p-1"]}})
    store.reports.save_report(make_report("run-1"))
    entry = store.audit.entries("ws-a")[0]
    entry.payload["nested"]["ids"].append("forged")
    entry.payload["extra"] = True
    report = store.reports.latest_report("ws-a")
    assert report is not None
    assert report.findings[0].fix is not None
    report.findings[0].fix.params["candidates"].append("명부 외 인물")
    report.timings_ms["check"] = 0.0

    assert store.audit.entries("ws-a")[0].payload == {"nested": {"ids": ["p-1"]}}
    assert store.reports.latest_report("ws-a") == make_report("run-1")


def test_caller_mutation_after_save_does_not_leak_in(store: Store) -> None:
    payload: dict[str, object] = {"ids": ["p-1"]}
    store.audit.record("ws-a", "lead", "x", payload)
    ids = payload["ids"]
    assert isinstance(ids, list)
    ids.append("late")
    assert store.audit.entries("ws-a")[0].payload == {"ids": ["p-1"]}


_SURROGATE = "\ud800"


def test_unencodable_text_is_refused_by_every_backend(store: Store) -> None:
    """A lone surrogate can live in a Python ``str`` but in no UTF-8 database. The memory
    backend used to store it while SQL crashed with a raw ``UnicodeEncodeError``."""
    graph = graph_v1()
    attempts: list[Callable[[], object]] = [
        lambda: store.workspaces.create_workspace(f"결제{_SURROGATE}팀", workspace_id="team"),
        lambda: store.workspaces.create_workspace(f"결제{_SURROGATE}팀"),
        lambda: store.audit.record("ws-a", "lead", "x", {"note": f"메모{_SURROGATE}"}),
        lambda: store.audit.record("ws-a", f"lead{_SURROGATE}", "x", {}),
        lambda: store.patches.save_patch(
            make_patch("p-s").model_copy(update={"runbook_text": f"문장{_SURROGATE}"})
        ),
        lambda: store.graphs.append_version(
            version_of("ws-a", 1, graph).model_copy(update={"note": f"메모{_SURROGATE}"}),
            graph,
            expected_head=None,
        ),
        lambda: store.rehearsals.save_scenario(
            "ws-a", make_scenario().model_copy(update={"title": f"시나리오{_SURROGATE}"})
        ),
        lambda: store.events.record_event("view", {"q": _SURROGATE}, T0),
    ]
    for attempt in attempts:
        with pytest.raises(ValueError, match="UTF-8"):
            attempt()
    assert store.workspaces.list_workspaces() == []
    assert store.audit.entries("ws-a") == []
    assert store.patches.list_patches("ws-a") == []
    assert store.graphs.list_versions("ws-a") == []
    assert store.rehearsals.list_scenarios("ws-a") == []
    assert store.events.event_counts() == {}


def test_unencodable_graph_is_refused_with_a_korean_message(store: Store) -> None:
    builder = GraphBuilder()
    builder.add_node(
        Node(
            id="service:pay",
            kind=NodeKind.SERVICE,
            name=f"결제{_SURROGATE}",
            prov=(Provenance(source=SourceKind.CATALOG, extractor=Extractor.IMPORT),),
        )
    )
    graph = builder.build()
    version = GraphVersion(
        workspace_id="ws-a", number=1, parent=None, graph_hash="0" * 64, created_by="lead", created_at=T0
    )
    with pytest.raises(ValueError, match="UTF-8"):
        store.graphs.append_version(version, graph, expected_head=None)
    assert store.graphs.list_versions("ws-a") == []


def test_document_batch_is_all_or_nothing(store: Store) -> None:
    """The memory backend used to keep the documents before the failing one."""
    good = Document(doc_id="doc-good", title="정상", text="결제 서비스")
    bad = Document(doc_id="doc-bad", title="깨짐", text=f"결제{_SURROGATE}")
    with pytest.raises(ValueError, match="UTF-8"):
        store.documents.save_documents("ws-a", [good, bad], store_raw=True)
    assert store.documents.list_documents("ws-a") == []


def _nul_key_writes(store: Store) -> dict[str, Callable[[], object]]:
    graph = graph_v1()
    scenario = make_scenario()
    return {
        "patch-id": lambda: store.patches.save_patch(make_patch("p\x00")),
        "patch-workspace": lambda: store.patches.save_patch(make_patch("p-1", "ws\x00")),
        "report-run-id": lambda: store.reports.save_report(make_report("run\x00")),
        "report-workspace": lambda: store.reports.save_report(make_report("run-1", "ws\x00")),
        "scenario-id": lambda: store.rehearsals.save_scenario("ws-a", make_scenario("sc\x00")),
        "scenario-workspace": lambda: store.rehearsals.save_scenario("ws\x00", scenario),
        "result-workspace": lambda: store.rehearsals.save_result("ws\x00", make_result(scenario, "h")),
        "result-graph-hash": lambda: store.rehearsals.save_result("ws-a", make_result(scenario, "h\x00")),
        "documents-workspace": lambda: store.documents.save_documents("ws\x00", [RUNBOOK], store_raw=False),
        "inputs-workspace": lambda: store.inputs.save_inputs("ws\x00", _inputs()),
        "audit-workspace": lambda: store.audit.record("ws\x00", "lead", "x", {}),
        "version-workspace": lambda: store.graphs.append_version(
            version_of("ws\x00", 1, graph), graph, expected_head=None
        ),
    }


@pytest.mark.parametrize("write", sorted(_nul_key_writes(MemoryStore())))
def test_keys_with_nul_are_refused_by_every_backend(store: Store, write: str) -> None:
    """Postgres text cannot hold NUL. Keys used to pass on memory/SQLite and fail only in
    production; now every backend refuses the same records."""
    with pytest.raises(ValueError, match="NUL"):
        _nul_key_writes(store)[write]()
    assert store.patches.list_patches("ws\x00") == []
    assert store.rehearsals.list_scenarios("ws-a") == []
    assert store.rehearsals.list_results("ws-a") == []
    assert store.reports.latest_report("ws-a") is None
    assert store.documents.list_documents("ws\x00") == []
    assert store.inputs.load_inputs("ws\x00") == StoredInputs()
    assert store.audit.entries("ws\x00") == []


@pytest.mark.parametrize(
    "variant",
    [
        "kim@example.com\u200b",  # zero-width space pasted from a messenger
        "\ufeffkim@example.com",  # byte-order mark from a CSV export
        "kim\u00ad@example.com",  # soft hyphen
        "kim@exam\u200dple.com",  # zero-width joiner
        "\u202ekim@example.com",  # right-to-left override (spoofing)
    ],
)
def test_invisible_format_characters_do_not_split_one_person(store: Store, variant: str) -> None:
    store.signups.add_signup(make_signup("kim@example.com"))
    store.signups.add_signup(make_signup(variant, marketing=False))
    assert store.signups.count_signups() == 1
    found = store.signups.get_signup("kim@example.com")
    assert found is not None
    assert found.consent_marketing is False


@pytest.mark.parametrize("email", ["a\x01b@c.de", "a\u3164b@c.de", "김\u115f@회사.한국", "\u200b@c.de"])
def test_email_with_invisible_characters_rejected(email: str) -> None:
    with pytest.raises(ValidationError, match="이메일"):
        Signup(email=email, consent_privacy=True, created_at=T0)


def test_email_with_unencodable_characters_rejected() -> None:
    # pydantic refuses the surrogate itself (before our validator); the point is it is refused.
    with pytest.raises(ValidationError):
        Signup(email=f"a{_SURROGATE}@c.de", consent_privacy=True, created_at=T0)
    with pytest.raises(ValueError, match="이메일"):
        normalize_email(f"a{_SURROGATE}@c.de")


def test_korean_idn_email_is_kept(store: Store) -> None:
    store.signups.add_signup(make_signup("김민수@회사.한국"))
    found = store.signups.get_signup(unicodedata.normalize("NFD", "김민수@회사.한국"))
    assert found is not None
    assert found.email == "김민수@회사.한국"


@pytest.mark.parametrize("email", ["", "   ", "not-an-email", "a@b", "a\x00@b.co", f"a{_SURROGATE}@b.co"])
def test_lookup_of_malformed_email_is_absent_not_an_error(store: Store, email: str) -> None:
    """Nothing malformed can ever have been stored, so the lookup's answer is "absent"."""
    assert store.signups.get_signup(email) is None


@pytest.mark.parametrize("blank", ["\u200b", "\u3164", "\ufeff\u200b", "\u115f\u1160"])
def test_invisible_optional_signup_answers_are_absent(blank: str) -> None:
    signup = Signup(
        email="a@b.co",
        organization=blank,
        team_size=blank,
        message=blank,
        source=blank,
        consent_privacy=True,
        created_at=T0,
    )
    assert (signup.organization, signup.team_size, signup.message, signup.source) == (None, None, None, None)


def test_undated_patches_sort_before_every_dated_patch(store: Store) -> None:
    """The documented order is "patches without a timestamp first". Undated patches used to
    sort as 1970-01-01, i.e. after anything older and interleaved with the epoch itself."""
    store.patches.save_patch(make_patch("p-1960", created_at=datetime(1960, 1, 1, tzinfo=UTC)))
    store.patches.save_patch(make_patch("a-epoch", created_at=datetime(1970, 1, 1, tzinfo=UTC)))
    store.patches.save_patch(make_patch("z-undated", created_at=None))
    assert [p.id for p in store.patches.list_patches("ws-a")] == ["z-undated", "p-1960", "a-epoch"]


@pytest.mark.parametrize(
    ("number", "update", "message"),
    [
        (1, {"parent": 7}, "부모 버전"),
        (2, {"parent": None}, "부모 버전"),
        (2, {"parent": 3}, "부모 버전"),
        (2, {"revert_of": 0}, "되돌릴 버전"),
        (2, {"revert_of": 2}, "되돌릴 버전"),
        (2, {"revert_of": 2**40}, "되돌릴 버전"),
        (1, {"revert_of": 1}, "되돌릴 버전"),
    ],
)
def test_version_lineage_must_reference_existing_history(
    store: Store, number: int, update: dict[str, object], message: str
) -> None:
    """P2: a version's lineage is a fact about history, so it may not point at versions that do
    not exist. Postgres would also reject ``revert_of=2**40`` (int4) where the others kept it."""
    g1, g2 = graph_v1(), graph_v2()
    expected_head = None
    if number == 2:
        store.graphs.append_version(version_of("ws-a", 1, g1), g1, expected_head=None)
        expected_head = 1
    bad = version_of("ws-a", number, g2).model_copy(update=update)
    with pytest.raises(ValueError, match=message):
        store.graphs.append_version(bad, g2, expected_head=expected_head)
    assert len(store.graphs.list_versions("ws-a")) == number - 1


def test_revert_of_an_existing_version_is_accepted(store: Store) -> None:
    g1, g2 = graph_v1(), graph_v2()
    store.graphs.append_version(version_of("ws-a", 1, g1), g1, expected_head=None)
    store.graphs.append_version(version_of("ws-a", 2, g2, patch_id="p-1"), g2, expected_head=1)
    revert = version_of("ws-a", 3, g1, patch_id="p-revert").model_copy(update={"revert_of": 2})
    store.graphs.append_version(revert, g1, expected_head=2)
    assert store.graphs.head("ws-a") == revert
    assert store.graphs.load_graph(revert.graph_hash) == g1


@pytest.mark.parametrize("name", ["결제\n팀", "결제\t팀", "\x1b[31m결제팀", "결제팀\x7f", "결제\u0085팀"])
def test_workspace_names_refuse_control_characters(store: Store, name: str) -> None:
    """Names are one-line labels shown in reports, SARIF and terminals; a newline or an ANSI
    escape in a name would let the name restructure that output."""
    with pytest.raises(ValueError, match="제어 문자"):
        store.workspaces.create_workspace(name)
    assert store.workspaces.list_workspaces() == []


@pytest.mark.parametrize(
    "padded", ["\u200b결제팀", "결제팀\u3164", "\ufeff 결제팀 \u200b", "\u3000결제팀\u3000"]
)
def test_invisible_padding_does_not_make_a_new_name(make_store: StoreFactory, padded: str) -> None:
    with closing(make_store(_fixed(T0))) as first, closing(make_store(_fixed(T0))) as second:
        plain = first.workspaces.create_workspace("결제팀")
        padded_ws = second.workspaces.create_workspace(padded)
    assert padded_ws == plain


def test_inner_hangul_filler_is_kept(store: Store) -> None:
    """Only the edges are trimmed: an inner character is part of what the user typed."""
    assert store.workspaces.create_workspace("결제\u3164팀").name == "결제\u3164팀"


def test_audit_append_order_and_timestamps_agree_under_concurrency(store: Store) -> None:
    writers = 8
    barrier = threading.Barrier(writers)

    def record(i: int) -> None:
        barrier.wait()
        store.audit.record("ws-a", f"actor-{i}", "x", {"i": i})

    threads = [threading.Thread(target=record, args=(i,)) for i in range(writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    entries = store.audit.entries("ws-a")
    assert len(entries) == writers
    assert [e.recorded_at for e in entries] == sorted(e.recorded_at for e in entries)
    assert len({e.recorded_at for e in entries}) == writers


def test_sql_detects_tampered_document_text() -> None:
    """Evidence spans index into the stored text; a changed text must not be served as the
    document the spans were extracted from (P2)."""
    with closing(SqlStore(MEMORY_SQLITE, clock=TickingClock())) as store:
        store.documents.save_documents("ws-a", [RUNBOOK], store_raw=True)
        with store.engine.begin() as conn:
            conn.execute(
                text("UPDATE documents SET text = :t"), {"t": RUNBOOK_TEXT.replace("김민수", "홍길동")}
            )
        with pytest.raises(StoreIntegrityError, match="손상"):
            store.documents.get_document("ws-a", RUNBOOK.doc_id)


_EMAIL_LOCAL = st.text(alphabet="abcxyz019._+김민수", min_size=1, max_size=10)
_EMAIL_DOMAIN = st.sampled_from(["example.com", "회사.한국", "EXAMPLE.ORG", "ｅｘａｍｐｌｅ．ｃｏｍ"])
_FORMAT_CHARS = st.lists(
    st.sampled_from(["\u200b", "\u200c", "\u200d", "\ufeff", "\u00ad", "\u2060"]), max_size=3
)


@HYPOTHESIS
@given(local=_EMAIL_LOCAL, domain=_EMAIL_DOMAIN, noise=_FORMAT_CHARS, cut=st.integers(min_value=0))
def test_email_key_is_idempotent_and_ignores_format_characters(
    local: str, domain: str, noise: list[str], cut: int
) -> None:
    email = f"{local}@{domain}"
    key = Signup(email=email, consent_privacy=True, created_at=T0).email
    assert Signup(email=key, consent_privacy=True, created_at=T0).email == key
    position = cut % (len(email) + 1)
    noisy = email[:position] + "".join(noise) + email[position:]
    assert Signup(email=noisy, consent_privacy=True, created_at=T0).email == key


@HYPOTHESIS
@given(name=st.text(alphabet=_TEXT, min_size=1, max_size=30))
def test_workspace_name_validation_is_idempotent(name: str) -> None:
    """Whatever a name normalises to must itself be accepted unchanged (stable ids, P6)."""
    with closing(MemoryStore(clock=_fixed(T0))) as first:
        try:
            ws = first.workspaces.create_workspace(name)
        except ValueError:
            return
    with closing(MemoryStore(clock=_fixed(T0))) as second:
        again = second.workspaces.create_workspace(ws.name)
    assert again == ws


def test_empty_and_unknown_lookups_are_absent(store: Store) -> None:
    assert store.workspaces.get_workspace("") is None
    assert store.documents.get_document("", "") is None
    assert store.documents.list_documents("") == []
    assert store.inputs.load_inputs("") == StoredInputs()
    assert store.rehearsals.list_results("ws-a", "") == []
    assert store.reports.latest_report("") is None
    lookups: list[Callable[[], object]] = [
        lambda: store.graphs.get_version("ws-a", 0),
        lambda: store.graphs.get_version("ws-a", -1),
        lambda: store.graphs.load_graph(""),
        lambda: store.patches.get_patch(""),
    ]
    for lookup in lookups:
        with pytest.raises(ports.NotFoundError):
            lookup()


# --------------------------------------------------------------------------- security review
# Sign-ups an anonymous caller cannot rewrite, deletion and retention primitives, and database
# errors that never quote the values being written (they end up in logs).


def test_add_signup_if_absent_keeps_the_first_record(store: Store) -> None:
    first = make_signup("Kim@Example.com", marketing=False)
    assert store.signups.add_signup_if_absent(first) is True
    forged = Signup(
        email="KIM@example.com",
        organization="사칭",
        message="덮어쓰기",
        consent_privacy=True,
        consent_marketing=True,
        created_at=T0 + timedelta(days=30),
        source="elsewhere",
    )
    assert store.signups.add_signup_if_absent(forged) is False
    # consent, consent date and every answer stay exactly as first given
    assert store.signups.get_signup("kim@example.com") == first
    assert store.signups.count_signups() == 1
    assert store.signups.add_signup_if_absent(make_signup("lee@example.com")) is True
    assert [s.email for s in store.signups.list_signups()] == ["kim@example.com", "lee@example.com"]
    # the internal upsert still replaces (administrator corrections)
    store.signups.add_signup(forged)
    assert store.signups.get_signup("kim@example.com") == forged


def _race_signups(store: Store, *, writers: int) -> Counter[bool]:
    barrier = threading.Barrier(writers)
    outcomes: Counter[bool] = Counter()
    lock = threading.Lock()

    def submit(i: int) -> None:
        signup = make_signup("race@example.com", marketing=i % 2 == 0).model_copy(update={"message": f"#{i}"})
        barrier.wait()
        inserted = store.signups.add_signup_if_absent(signup)
        with lock:
            outcomes[inserted] += 1

    threads = [threading.Thread(target=submit, args=(i,)) for i in range(writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    return outcomes


def test_concurrent_add_signup_if_absent_has_one_winner(store: Store) -> None:
    assert _race_signups(store, writers=6) == Counter({True: 1, False: 5})
    assert store.signups.count_signups() == 1


def test_sql_concurrent_signups_on_file_database(tmp_path: Path) -> None:
    """On a file database the losers really reach the INSERT; the primary key turns them away."""
    url = f"sqlite:///{(tmp_path / 'signups.db').as_posix()}"
    with closing(SqlStore(url, clock=TickingClock())) as store:
        assert _race_signups(store, writers=6) == Counter({True: 1, False: 5})
        assert store.signups.count_signups() == 1


def test_delete_signup_withdraws_one_person(store: Store) -> None:
    store.signups.add_signup(make_signup("Kim@Example.com"))
    store.signups.add_signup(make_signup("lee@example.com"))
    full_width = "ＫＩＭ@example.com"  # "KIM" typed with a Korean IME
    assert store.signups.delete_signup(full_width) is True  # normalised like on insert
    assert store.signups.get_signup("kim@example.com") is None
    assert store.signups.delete_signup("kim@example.com") is False
    assert store.signups.delete_signup("not-an-email") is False
    assert [s.email for s in store.signups.list_signups()] == ["lee@example.com"]
    assert store.signups.count_signups() == 1
    # a person who withdrew may sign up again from scratch
    assert store.signups.add_signup_if_absent(make_signup("kim@example.com", marketing=False)) is True


def test_delete_signups_before_a_cutoff(store: Store) -> None:
    cutoff = T0 - timedelta(days=365)
    for email, days in (("old@example.com", 400), ("edge@example.com", 365), ("young@example.com", 1)):
        created_at = T0 - timedelta(days=days)
        store.signups.add_signup(make_signup(email).model_copy(update={"created_at": created_at}))
    with pytest.raises(ValueError, match="시간대"):
        store.signups.delete_signups_before(cutoff.replace(tzinfo=None))
    assert store.signups.count_signups() == 3
    assert store.signups.delete_signups_before(cutoff.astimezone(KST)) == 1  # strictly before
    assert [s.email for s in store.signups.list_signups()] == ["edge@example.com", "young@example.com"]
    assert store.signups.delete_signups_before(cutoff) == 0


def _populate(store: Store, workspace_id: str, *, graphs: tuple[OpsGraph, ...], tag: str) -> None:
    """Records of ``workspace_id`` in every repository that is scoped by workspace."""
    store.documents.save_documents(workspace_id, [RUNBOOK], store_raw=True)
    store.inputs.save_inputs(workspace_id, _inputs())
    for number, graph in enumerate(graphs, start=1):
        head = number - 1 if number > 1 else None
        store.graphs.append_version(version_of(workspace_id, number, graph), graph, expected_head=head)
    store.patches.save_patch(make_patch(f"p-{tag}", workspace_id))
    scenario = make_scenario(f"sc-{tag}")
    store.rehearsals.save_scenario(workspace_id, scenario)
    store.rehearsals.save_result(workspace_id, make_result(scenario, graphs[0].content_hash()))
    store.reports.save_report(make_report(f"run-{tag}", workspace_id))
    store.audit.record(workspace_id, "lead", "patch.approve", {"patch": f"p-{tag}"})
    store.access.set_access_key_digest(workspace_id, _DIGEST_A)


def _snapshot(store: Store, workspace_id: str) -> list[object]:
    return [
        _dump(store.workspaces.get_workspace(workspace_id)),
        _dump(store.documents.list_documents(workspace_id)),
        _dump(store.documents.get_document(workspace_id, RUNBOOK.doc_id)),
        _dump(store.inputs.load_inputs(workspace_id)),
        _dump(store.graphs.list_versions(workspace_id)),
        _dump(store.patches.list_patches(workspace_id)),
        _dump(store.rehearsals.list_scenarios(workspace_id)),
        _dump(store.rehearsals.list_results(workspace_id)),
        _dump(store.reports.latest_report(workspace_id)),
        _dump(store.audit.entries(workspace_id)),
        store.access.access_key_digest(workspace_id),
    ]


def _assert_absent(store: Store, workspace_id: str, *, tag: str) -> None:
    assert store.workspaces.get_workspace(workspace_id) is None
    assert workspace_id not in [w.id for w in store.workspaces.list_workspaces()]
    assert store.documents.list_documents(workspace_id) == []
    assert store.documents.get_document(workspace_id, RUNBOOK.doc_id) is None
    assert store.inputs.load_inputs(workspace_id) == StoredInputs()
    assert store.graphs.head(workspace_id) is None
    assert store.graphs.list_versions(workspace_id) == []
    assert store.patches.list_patches(workspace_id) == []
    assert store.rehearsals.list_scenarios(workspace_id) == []
    assert store.rehearsals.list_results(workspace_id) == []
    assert store.reports.latest_report(workspace_id) is None
    assert store.audit.entries(workspace_id) == []
    assert store.access.access_key_digest(workspace_id) is None
    lookups: list[Callable[[], object]] = [
        lambda: store.graphs.get_version(workspace_id, 1),
        lambda: store.patches.get_patch(f"p-{tag}"),
    ]
    for lookup in lookups:
        with pytest.raises(ports.NotFoundError):
            lookup()


def test_delete_workspace_removes_everything_it_owns(store: Store) -> None:
    g1, g2 = graph_v1(), graph_v2()
    doomed = store.workspaces.create_workspace("무료 검진", workspace_id="ws-a")
    kept = store.workspaces.create_workspace("결제팀", workspace_id="ws-b")
    _populate(store, doomed.id, graphs=(g1, g2), tag="a")
    _populate(store, kept.id, graphs=(g1,), tag="b")
    store.signups.add_signup(make_signup())
    store.events.record_event("view", {"path": "/"}, T0)
    kept_before = _snapshot(store, kept.id)

    assert store.workspaces.delete_workspace(doomed.id) is True

    _assert_absent(store, doomed.id, tag="a")
    with pytest.raises(ports.NotFoundError):
        store.graphs.load_graph(g2.content_hash())  # only the deleted workspace used it
    assert store.graphs.load_graph(g1.content_hash()) == g1  # still a version of ws-b
    assert _snapshot(store, kept.id) == kept_before
    assert store.signups.count_signups() == 1  # sign-ups and events are not workspace data
    assert store.events.event_counts() == {"view": 1}
    assert store.workspaces.delete_workspace(doomed.id) is False  # nothing left: a no-op
    assert store.workspaces.delete_workspace("ws-never") is False


def test_delete_workspace_also_removes_records_without_a_workspace(store: Store) -> None:
    """One-shot CLI scans write records under an id that has no workspace record."""
    graph = graph_v1()
    _populate(store, "scan-1", graphs=(graph,), tag="s")
    assert store.workspaces.delete_workspace("scan-1") is True
    _assert_absent(store, "scan-1", tag="s")
    with pytest.raises(ports.NotFoundError):
        store.graphs.load_graph(graph.content_hash())


def test_a_deleted_workspace_id_starts_from_scratch(store: Store) -> None:
    graph = graph_v1()
    store.workspaces.create_workspace("무료 검진", workspace_id="ws-a")
    _populate(store, "ws-a", graphs=(graph,), tag="a")
    assert store.workspaces.delete_workspace("ws-a") is True
    again = store.workspaces.create_workspace("다시", workspace_id="ws-a")
    assert store.workspaces.get_workspace("ws-a") == again
    store.graphs.append_version(version_of("ws-a", 1, graph), graph, expected_head=None)
    assert store.graphs.load_graph(graph.content_hash()) == graph
    # global record ids the deleted workspace owned are free for any workspace again
    store.patches.save_patch(make_patch("p-a", "ws-c"))
    store.reports.save_report(make_report("run-a", "ws-c"))
    assert [p.id for p in store.patches.list_patches("ws-c")] == ["p-a"]


@pytest.mark.parametrize("workspace_id", ["ws\x00a", f"ws{_SURROGATE}"])
def test_delete_workspace_refuses_unstorable_ids(store: Store, workspace_id: str) -> None:
    with pytest.raises(ValueError, match="워크스페이스 id"):
        store.workspaces.delete_workspace(workspace_id)


def _exercise_deletions(store: Store) -> list[object]:
    _exercise(store)
    (team,) = [w for w in store.workspaces.list_workspaces() if w.id != "demo"]
    store.audit.record("demo", "lead", "demo.open", {})
    deleted = store.workspaces.delete_workspace(team.id)
    store.audit.record("demo", "lead", "demo.reopen", {})  # seq continues the same way everywhere
    return [
        deleted,
        _dump(store.workspaces.list_workspaces()),
        _dump(store.audit.entries("demo")),
        store.signups.delete_signup("kim@example.com"),
        store.signups.add_signup_if_absent(make_signup("lee@example.com")),
        store.signups.delete_signups_before(T0 + timedelta(seconds=1)),
        _dump(store.signups.list_signups()),
    ]


def test_backends_agree_after_deletions() -> None:
    with (
        closing(MemoryStore(clock=_fixed(T0))) as memory,
        closing(SqlStore(MEMORY_SQLITE, clock=_fixed(T0))) as sql,
    ):
        assert _exercise_deletions(memory) == _exercise_deletions(sql)


def _scoped_row_counts(store: SqlStore, workspace_id: str) -> dict[str, int]:
    with store.engine.connect() as conn:
        counts = {
            table.name: conn.scalar(
                select(func.count()).select_from(table).where(table.c.workspace_id == workspace_id)
            )
            or 0
            for table in WORKSPACE_TABLES
        }
        for name in ("workspaces", "graph_blobs"):
            counts[name] = conn.scalar(select(func.count()).select_from(Base.metadata.tables[name])) or 0
    return counts


def test_every_table_with_a_workspace_id_is_emptied_on_delete() -> None:
    """The SQL backend deletes from every table with a ``workspace_id`` column, derived from the
    schema; pinning the list here makes a new table reach the in-memory backend and this suite too."""
    assert sorted(table.name for table in WORKSPACE_TABLES) == [
        "audit_log",
        "documents",
        "graph_versions",
        "inputs",
        "patches",
        "rehearsal_results",
        "reports",
        "scenarios",
        "workspace_access_keys",
    ]
    unscoped = {table.name for table in Base.metadata.sorted_tables} - {t.name for t in WORKSPACE_TABLES}
    assert unscoped == {"workspaces", "graph_blobs", "signups", "events"}
    with closing(SqlStore(MEMORY_SQLITE, clock=TickingClock())) as store:
        store.workspaces.create_workspace("무료 검진", workspace_id="ws-a")
        _populate(store, "ws-a", graphs=(graph_v1(),), tag="a")
        assert all(_scoped_row_counts(store, "ws-a").values())  # the fixture fills every table
        assert store.workspaces.delete_workspace("ws-a") is True
        assert set(_scoped_row_counts(store, "ws-a").values()) == {0}


def test_sql_append_stores_a_blob_collected_after_it_was_written(monkeypatch: pytest.MonkeyPatch) -> None:
    """A workspace deletion may collect a shared blob between append_version's blob write and its
    version insert. The version transaction then stores the blob again rather than leave a version
    that points at nothing (SQLite enforces no foreign keys)."""
    graph = graph_v1()
    with closing(SqlStore(MEMORY_SQLITE, clock=TickingClock())) as store:
        monkeypatch.setattr(SqlGraphRepository, "_put_blob", lambda self, graph_hash, blob: None)
        store.graphs.append_version(version_of("ws-a", 1, graph), graph, expected_head=None)
        assert store.graphs.load_graph(graph.content_hash()) == graph


def test_sql_errors_never_quote_the_values_being_written() -> None:
    """A failing INSERT's error — and its whole traceback — must not carry the row (personal data)."""
    with closing(SqlStore(MEMORY_SQLITE, clock=TickingClock())) as store:
        with store.engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TRIGGER refuse_signups BEFORE INSERT ON signups "
                    "BEGIN SELECT RAISE(ABORT, 'disk full'); END"
                )
            )
        signup = make_signup("Victim@Corp.example").model_copy(update={"message": "연락처 010-1234-5678"})
        for write in (store.signups.add_signup, store.signups.add_signup_if_absent):
            # add_signup_if_absent must not mistake this integrity failure for "already stored"
            with pytest.raises(SQLAlchemyError) as raised:
                write(signup)
            rendered = "".join(traceback.format_exception(raised.value))
            assert "victim@corp.example" not in rendered
            assert "010-1234-5678" not in rendered
            assert "parameters hidden" in rendered
            assert database_error_summary(raised.value) == {
                "db_error": "IntegrityError",
                "db_driver_error": "IntegrityError",
            }
        assert store.signups.count_signups() == 0


def test_database_error_summary_finds_database_errors_anywhere_in_the_chain() -> None:
    names = {"db_error": "IntegrityError", "db_driver_error": "IntegrityError"}
    assert database_error_summary(ValueError("victim@corp.example")) is None
    driver = sqlite3.IntegrityError("UNIQUE constraint failed: victim@corp.example")
    wrapped = SqlIntegrityError("INSERT INTO signups ...", {"email": "victim@corp.example"}, driver)
    outer = RuntimeError("저장 실패")
    outer.__cause__ = wrapped
    wrapped.__cause__ = driver
    assert database_error_summary(outer) == names
    assert database_error_summary(sqlite3.OperationalError("disk I/O error")) == {
        "db_driver_error": "OperationalError"
    }
    assert database_error_summary(ExceptionGroup("tasks", [ValueError("a"), wrapped])) == names
    looped = ValueError("a")
    looped.__context__ = KeyError("b")
    looped.__context__.__context__ = looped  # a cycle must not hang the walk
    assert database_error_summary(looped) is None
