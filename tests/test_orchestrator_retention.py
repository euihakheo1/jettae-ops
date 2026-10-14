"""Retention, withdrawal and who-did-what for the public beta (service level).

* :meth:`JettaeService.purge_expired` deletes free-scan workspaces and sign-ups past their retention,
  never anything else, with a fixed clock (both store backends).
* :meth:`JettaeService.delete_signup` audits a fingerprint of the address, never the address.
* A free scan keeps only its result: no structured inputs after the diagnosis, nothing at all when it
  fails.
* Audit entries name the real caller (workspace creation, uploads, drills, the demo).
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime, timedelta

import pytest

from jettae.domain import LlmMode
from jettae.ingest import load_bytes, load_roster
from jettae.orchestrator import DEMO_WORKSPACE_NAME, FREE_SCAN_WORKSPACE_NAME, JettaeService, PurgeOutcome
from jettae.orchestrator.demo import DEMO_ACTOR
from jettae.orchestrator.service import (
    ADMIN_AUDIT_SCOPE,
    AUDIT_RETENTION_PURGED,
    AUDIT_SIGNUP_DELETED,
    email_fingerprint,
)
from jettae.store import MemoryStore, Signup, SqlStore, Store, StoredInputs
from test_orchestrator_support import event_loop_runner, settings

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
RUNBOOK = "# 결제 런북\n\n## 결제 서비스\n\n- 담당: 김서연 사원 (슬랙, 전화)\n"
ROSTER = "이름,상태,직급,팀,역할,별칭,연락수단,퇴사일\n김서연,재직,사원,결제팀,1차 당직,,슬랙;전화,\n"


class Clock:
    """A settable clock shared by the store (record times) and the service."""

    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture(scope="module")
def runner() -> Iterator[asyncio.Runner]:
    with event_loop_runner() as loop:
        yield loop


@contextmanager
def _store(kind: str, clock: Clock) -> Iterator[Store]:
    if kind == "memory":
        yield MemoryStore(clock=clock)
        return
    with closing(SqlStore("sqlite+pysqlite:///:memory:", clock=clock)) as store:
        yield store


def _service(store: Store, clock: Clock, **overrides: object) -> JettaeService:
    return JettaeService(store, settings(**overrides), clock=clock)


def _free_scan(runner: asyncio.Runner, service: JettaeService) -> str:
    documents = [load_bytes("결제.md", RUNBOOK.encode("utf-8"))]
    outcome = runner.run(
        service.free_scan(documents=documents, roster=load_roster(ROSTER.encode("utf-8")), actor="anonymous")
    )
    return outcome.workspace_id


def _signup(service: JettaeService, email: str, created_at: datetime) -> None:
    assert service.signup(Signup(email=email, consent_privacy=True, created_at=created_at))


# --------------------------------------------------------------------------- purge


@pytest.mark.parametrize("kind", ["memory", "sql"])
def test_purge_deletes_only_expired_free_scans_and_signups(runner: asyncio.Runner, kind: str) -> None:
    clock = Clock(T0)
    with _store(kind, clock) as store:
        service = _service(store, clock)
        old_scan = _free_scan(runner, service)
        team = service.create_workspace("결제 플랫폼팀").id
        demo = service.create_workspace(DEMO_WORKSPACE_NAME).id
        _signup(service, "old@example.com", T0 - timedelta(days=366))
        _signup(service, "new@example.com", T0 - timedelta(days=10))
        clock.now = T0 + timedelta(days=6)
        recent_scan = _free_scan(runner, service)
        clock.now = T0 + timedelta(days=8)

        outcome = service.purge_expired()

        assert outcome == PurgeOutcome(workspaces=1, signups=1)
        assert {w.id for w in service.list_workspaces()} == {team, demo, recent_scan}
        assert store.reports.latest_report(old_scan) is None
        assert store.access.access_key_digest(old_scan) is None
        assert store.audit.entries(old_scan) == []
        assert service.latest_report(recent_scan).workspace_id == recent_scan
        assert [s.email for s in service.store.signups.list_signups()] == ["new@example.com"]
        [entry] = store.audit.entries(ADMIN_AUDIT_SCOPE)
        assert (entry.actor, entry.action) == ("system", AUDIT_RETENTION_PURGED)
        assert entry.payload == {
            "free_scan_workspaces": 1,
            "signups": 1,
            "free_scan_retention_days": 7,
            "signup_retention_days": 365,
        }
        # nothing left to delete: no new audit entry
        assert service.purge_expired() == PurgeOutcome(workspaces=0, signups=0)
        assert len(store.audit.entries(ADMIN_AUDIT_SCOPE)) == 1


def test_purge_takes_an_explicit_moment_and_refuses_a_naive_one(runner: asyncio.Runner) -> None:
    clock = Clock(T0)
    service = _service(MemoryStore(clock=clock), clock)
    scan = _free_scan(runner, service)
    assert service.purge_expired(T0 + timedelta(days=7)) == PurgeOutcome(workspaces=0, signups=0)
    with pytest.raises(ValueError, match="삭제 기준 시각"):
        service.purge_expired(datetime(2026, 12, 1))
    assert service.purge_expired(T0 + timedelta(days=7, seconds=1)).workspaces == 1
    assert service.list_workspaces() == []
    assert scan not in service._texts  # process-local state is forgotten too


def test_zero_retention_keeps_everything(runner: asyncio.Runner) -> None:
    clock = Clock(T0)
    service = _service(MemoryStore(clock=clock), clock, free_scan_retention_days=0, signup_retention_days=0)
    _free_scan(runner, service)
    _signup(service, "old@example.com", T0 - timedelta(days=4000))
    assert service.purge_expired(T0 + timedelta(days=4000)) == PurgeOutcome(workspaces=0, signups=0)
    assert len(service.list_workspaces()) == 1
    assert service.demand_metrics()["signups"] == 1


def test_a_team_workspace_is_never_purged_whatever_its_age() -> None:
    clock = Clock(T0)
    service = _service(MemoryStore(clock=clock), clock)
    team = service.create_workspace("무료 검진팀").id  # close to, but not, the free-scan name
    demo = service.create_workspace(DEMO_WORKSPACE_NAME).id
    assert service.purge_expired(T0 + timedelta(days=10_000)).workspaces == 0
    assert {w.id for w in service.list_workspaces()} == {team, demo}


# --------------------------------------------------------------------------- free scans keep only results


def test_a_free_scan_keeps_no_structured_inputs(runner: asyncio.Runner) -> None:
    clock = Clock(T0)
    service = _service(MemoryStore(clock=clock), clock)
    ws = _free_scan(runner, service)
    assert service.store.inputs.load_inputs(ws) == StoredInputs()
    summary = service.input_summary(ws)
    assert (summary.roster, summary.oncall, summary.catalog) == (0, 0, 0)
    # the result is still there
    report = service.latest_report(ws)
    assert report.findings
    assert service.evidence_markdown(ws)
    assert service.graph(ws)[1].nodes


def test_a_free_scan_that_fails_leaves_nothing(
    runner: asyncio.Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = Clock(T0)
    service = _service(MemoryStore(clock=clock), clock)

    async def broken(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("pipeline crashed")

    monkeypatch.setattr(service, "diagnose", broken)
    with pytest.raises(RuntimeError, match="pipeline crashed"):
        _free_scan(runner, service)
    assert service.list_workspaces() == []
    assert service._texts == {}
    assert service.demand_metrics()["events"] == {}


# --------------------------------------------------------------------------- withdrawal


def test_deleting_a_signup_audits_a_fingerprint_not_the_address() -> None:
    clock = Clock(T0)
    service = _service(MemoryStore(clock=clock), clock)
    _signup(service, "Kim@Example.com", T0)
    assert service.delete_signup("ＫＩＭ@example.com", actor="heo") is True  # normalised like on insert
    assert service.delete_signup("kim@example.com", actor="heo") is False
    assert service.demand_metrics()["signups"] == 0
    entries = service.store.audit.entries(ADMIN_AUDIT_SCOPE)
    assert [(e.actor, e.action, e.payload["deleted"]) for e in entries] == [
        ("heo", AUDIT_SIGNUP_DELETED, True),
        ("heo", AUDIT_SIGNUP_DELETED, False),
    ]
    expected = hashlib.sha256(b"kim@example.com").hexdigest()[:12]
    assert {e.payload["email_sha256"] for e in entries} == {expected}
    assert "kim@" not in repr(entries).casefold()
    with pytest.raises(ValueError, match="actor"):
        service.delete_signup("kim@example.com", actor=" ")


def test_email_fingerprints_are_short_and_normalised() -> None:
    assert email_fingerprint(" Kim@Example.com ") == email_fingerprint("kim@example.com")
    assert len(email_fingerprint("kim@example.com")) == 12
    assert email_fingerprint("not an address") != email_fingerprint("kim@example.com")


def test_a_second_signup_for_an_address_changes_nothing() -> None:
    clock = Clock(T0)
    service = _service(MemoryStore(clock=clock), clock)
    first = Signup(email="kim@example.com", consent_privacy=True, consent_marketing=False, created_at=T0)
    assert service.signup(first) is True
    later = first.model_copy(update={"consent_marketing": True, "created_at": T0 + timedelta(days=1)})
    assert service.signup(later) is False
    assert service.store.signups.get_signup("kim@example.com") == first


# --------------------------------------------------------------------------- audit actors


def test_audit_entries_name_the_caller(runner: asyncio.Runner) -> None:
    clock = Clock(T0)
    service = _service(MemoryStore(clock=clock), clock)
    workspace = service.create_workspace("결제 플랫폼팀", actor="kim.editor")
    service.upload_inputs(
        workspace.id, documents=[load_bytes("결제.md", RUNBOOK.encode("utf-8"))], actor="kim.editor"
    )
    runner.run(service.diagnose(workspace.id, actor="kim.editor"))
    runner.run(service.rehearse(workspace.id, generate="template", actor="lee.editor"))
    actors = {(e.action, e.actor) for e in service.store.audit.entries(workspace.id)}
    assert ("workspace.created", "kim.editor") in actors
    assert ("inputs.uploaded", "kim.editor") in actors
    assert ("rehearsal.run", "lee.editor") in actors
    assert all(actor != "system" for _action, actor in actors)
    with pytest.raises(ValueError, match="actor"):
        service.create_workspace("빈 요청자", actor="")
    assert [w.name for w in service.list_workspaces()] == ["결제 플랫폼팀"]


def test_free_scans_and_demo_runs_are_audited_as_their_caller(runner: asyncio.Runner) -> None:
    clock = Clock(T0)
    service = _service(MemoryStore(clock=clock), clock)
    scan = _free_scan(runner, service)
    assert {e.actor for e in service.store.audit.entries(scan)} == {"anonymous"}

    visitor = runner.run(service.bootstrap_demo(llm_mode=LlmMode.OFF, actor="anonymous"))
    assert {e.actor for e in service.store.audit.entries(visitor.workspace_id)} == {"anonymous"}
    # the CLI and the start-up set-up keep the demo actor
    startup = _service(MemoryStore(clock=clock), clock)
    report = runner.run(startup.bootstrap_demo())
    assert {e.actor for e in startup.store.audit.entries(report.workspace_id)} == {DEMO_ACTOR}


def test_demo_report_is_none_until_the_demo_was_diagnosed(runner: asyncio.Runner) -> None:
    clock = Clock(T0)
    service = _service(MemoryStore(clock=clock), clock)
    assert service.demo_report() is None
    service.create_workspace(DEMO_WORKSPACE_NAME)
    assert service.demo_report() is None
    report = runner.run(service.bootstrap_demo())
    assert service.demo_report() == report
    assert FREE_SCAN_WORKSPACE_NAME != DEMO_WORKSPACE_NAME
