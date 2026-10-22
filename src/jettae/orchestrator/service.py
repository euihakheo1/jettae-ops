"""``JettaeService`` — the single application entry point for the API and the CLI (애플리케이션 서비스).

Why one façade: the REST API, the CLI and the GitHub Action must behave identically, so every use
case — upload, diagnose, rehearse, draft, approve, revert, export — lives here once, on top of the
ports (:class:`~jettae.store.Store`, :class:`~jettae.llm.base.LLMClient`) and never in a delivery
layer. Errors are domain errors the delivery layers map mechanically:
:class:`~jettae.domain.ports.NotFoundError` → 404, :class:`PermissionError` → 403,
:class:`~jettae.domain.ports.VersionConflictError` → 409, :class:`ValueError` → 422.

Workspace state that has no column of its own is kept as events in the append-only audit log (P7),
so it survives restarts and is the same for every process sharing the store:

* ``inputs.uploaded`` names the workspace's *current* runbook set (an upload of documents replaces
  the set, so a re-uploaded runbook does not leave its old version in the diagnosis);
* ``diagnosis.completed`` records each report with the extraction its graph was built from, the
  approved patches re-applied on top and the drill set in effect (hand-written and LLM drills).
  Together with the version history this gives every version a *lineage*
  (:func:`resolve_lineage`), which is how a re-diagnosis keeps human decisions when the documents
  change (see :func:`~jettae.orchestrator.pipeline.rebase_graph`).

Raw runbook text follows ``settings.store_raw_documents`` (default False — 원문 비저장). The pipeline
still needs the text, so with storage off the texts of the current upload are kept in this
process only (keyed by workspace and content id) and must be uploaded again after a restart; the
store keeps only metadata, and the graph keeps only the evidence quotes. Hand-written drill files
(``scenarios.yaml``) wait in the same process-local way until the next diagnosis resolves them onto
the graph; resolved drills are then persisted and carried from report to report.

Workspaces created by anonymous visitors of the public site (free scans) are private: a workspace
id is an identifier, not a secret, so each gets an access key — returned once, stored only as its
SHA-256 digest (:meth:`JettaeService.grant_workspace_key`). The demo workspace is open to everyone
(:meth:`JettaeService.workspace_for_visitor`). Free scans keep only their result: the free text
of their structured inputs is PII-masked before assembly (:func:`mask_catalog_free_text`,
:func:`mask_roster_free_text`), the inputs themselves are dropped once the diagnosis is done, the
copies of drills and replay results the report already carries are not stored a second time, and
:meth:`JettaeService.purge_expired` deletes the whole workspace after
``JETTAE_FREE_SCAN_RETENTION_DAYS`` (sign-ups after ``JETTAE_SIGNUP_RETENTION_DAYS``).

Every judgement is the fixed rules and the deterministic replay (P1); the LLM is used only by the
pipeline's extraction/wording stages and only when the requested mode is LIVE or REPLAY (P4).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import secrets
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict
from typing_extensions import TypedDict

from jettae import __version__
from jettae.check.context import CheckContext
from jettae.check.registry import rule_catalog as fixed_rule_catalog
from jettae.config import Settings
from jettae.domain import (
    CatalogService,
    DiagnosisReport,
    Document,
    DocumentRef,
    ExplorationSummary,
    Finding,
    GraphVersion,
    GroundingStatus,
    HypothesisOrigin,
    InputBundle,
    LlmMode,
    LlmUsageSummary,
    OnCallShift,
    OpsGraph,
    Patch,
    PatchStatus,
    Policy,
    ReadinessReport,
    RehearsalResult,
    RiskHypothesis,
    RosterEntry,
    RuleInfo,
    Scenario,
    ScenarioOrigin,
    SecurityNotice,
    SimConfig,
)
from jettae.domain.ports import NotFoundError as PortNotFoundError
from jettae.domain.ports import VersionConflictError
from jettae.improve import DECIDER_ROLES, GraphDiff, LlmDrafter, TemplateDrafter, VersionService, utc_now
from jettae.ingest import decode_text, load_bundle_dir, mask_pii, scan_injection
from jettae.llm.base import LLMClient, MeteredLLM, UsageMeter
from jettae.llm.factory import build_llm
from jettae.orchestrator.demo import DEMO_ACTOR, DEMO_SCENARIOS_FILE, DEMO_WORKSPACE_NAME, demo_bundle_path
from jettae.orchestrator.pipeline import (
    assess,
    build_context,
    evaluate_graph,
    gather_scenarios,
    missing_references,
    rehearse_all,
    run_diagnosis,
    validate_drafts,
)
from jettae.orchestrator.state import (
    MAX_LLM_CONCURRENCY,
    MIN_LLM_CONCURRENCY,
    DiagnosisOptions,
    DiagnosisOutcome,
)
from jettae.rehearse import llm_scenarios, template_scenarios
from jettae.reporting import EvidenceView, build_evidence, render_view_html, render_view_markdown, to_sarif
from jettae.store import AuditEntry, NotFoundError, Signup, Store, StoredDocument, StoredInputs, Workspace
from jettae.store.base import deletion_cutoff, signup_key, validate_workspace_name

__all__ = [
    "ADMIN_AUDIT_SCOPE",
    "AUDIT_DIAGNOSIS",
    "AUDIT_INPUTS",
    "AUDIT_KEY_GRANTED",
    "AUDIT_RETENTION_PURGED",
    "AUDIT_SIGNUPS_READ",
    "AUDIT_SIGNUP_DELETED",
    "EVENT_RUNBOOK_UPLOAD",
    "FREE_SCAN_WORKSPACE_NAME",
    "ApprovalResult",
    "Clock",
    "DemandMetrics",
    "FreeScanOutcome",
    "InputSummary",
    "JettaeService",
    "Lineage",
    "LlmFactory",
    "PurgeOutcome",
    "RunKind",
    "ScenarioGeneration",
    "diagnosis_lineages",
    "drill_ids",
    "email_fingerprint",
    "mask_catalog_free_text",
    "mask_roster_free_text",
    "merge_rehearsal_history",
    "resolve_lineage",
    "run_id_for",
]

#: Returns the current instant, timezone-aware. Stamps metadata (report time, patch time) only.
Clock = Callable[[], datetime]
#: Builds the LLM client for a mode (``jettae.llm.factory.build_llm`` fits; tests inject fakes).
LlmFactory = Callable[[Settings, LlmMode], LLMClient]
#: How a report came about: a full diagnosis, or a re-judgement after an approval / a revert.
RunKind = Literal["diagnosis", "approval", "revert"]
#: Which scenarios :meth:`JettaeService.rehearse` replays when no ids are given.
ScenarioGeneration = Literal["template", "llm", "human", "all"]

#: Audit action naming the workspace's current runbook set.
AUDIT_INPUTS: Final = "inputs.uploaded"
#: Audit action recorded for every saved report (lineage, notes, LLM mode).
AUDIT_DIAGNOSIS: Final = "diagnosis.completed"
#: Audit action recorded when a workspace gets a (new) access key — never the key or its digest.
AUDIT_KEY_GRANTED: Final = "workspace.key_granted"
#: Audit action recorded whenever an administrator reads the sign-up list (personal data).
AUDIT_SIGNUPS_READ: Final = "signups.read"
#: Audit action recorded when an administrator deletes a sign-up (withdrawal of consent); the
#: address itself is not recorded, only :func:`email_fingerprint`.
AUDIT_SIGNUP_DELETED: Final = "signup.deleted"
#: Audit action recorded when the retention purge deleted something (counts only).
AUDIT_RETENTION_PURGED: Final = "retention.purged"
#: Audit scope of reads that concern no workspace. No workspace id can start with ``_``
#: (:func:`jettae.store.base.validate_workspace_id`), so the scope never mixes with a workspace's log.
ADMIN_AUDIT_SCOPE: Final = "_admin"
#: Name of every workspace a free scan creates (ids still differ: name + clock + attempt).
FREE_SCAN_WORKSPACE_NAME: Final = "무료 검진"
#: Demand event the server records for each free scan; clients cannot send it (not on the allow-list).
EVENT_RUNBOOK_UPLOAD: Final = "runbook_upload"

_DECIDED: Final = frozenset({PatchStatus.APPROVED, PatchStatus.REJECTED, PatchStatus.SUPERSEDED})
_GENERATED_ORIGINS: Final[Mapping[str, frozenset[ScenarioOrigin]]] = {
    "template": frozenset({ScenarioOrigin.TEMPLATE}),
    "human": frozenset({ScenarioOrigin.HUMAN}),
    "llm": frozenset({ScenarioOrigin.LLM}),
    "all": frozenset(ScenarioOrigin),
}
_DEFAULT_SIM: Final = SimConfig()


# --------------------------------------------------------------------------- records


class InputSummary(BaseModel):
    """What a workspace will be diagnosed with, as the upload left it."""

    model_config = ConfigDict(frozen=True)

    workspace_id: str
    documents: tuple[DocumentRef, ...]
    roster: int
    oncall: int
    catalog: int
    #: quarantined prompt-injection ranges across the current runbooks (격리된 지시문 수)
    security_notices: int
    #: whether raw runbook text is persisted (``settings.store_raw_documents``)
    raw_stored: bool
    #: whether every current runbook text is available right now (False after a restart with
    #: raw storage off — the runbooks must be uploaded again before the next diagnosis)
    diagnosable: bool
    #: drill files waiting to be resolved by the next diagnosis
    pending_scenario_files: int


class ApprovalResult(BaseModel):
    """The version an approval/revert created and the report re-judged on it."""

    model_config = ConfigDict(frozen=True)

    version: GraphVersion
    report: DiagnosisReport


class FreeScanOutcome(BaseModel):
    """What a free scan produced: its private workspace, the key that opens it, and the report."""

    model_config = ConfigDict(frozen=True)

    workspace_id: str
    #: plaintext access key — returned exactly once; the store keeps only its SHA-256 digest
    access_key: str
    report: DiagnosisReport


class DemandMetrics(TypedDict):
    """Pre-launch demand (수요 측정): distinct sign-ups and event counts per name."""

    signups: int
    events: dict[str, int]


class PurgeOutcome(BaseModel):
    """What one retention purge deleted (:meth:`JettaeService.purge_expired`)."""

    model_config = ConfigDict(frozen=True)

    #: free-scan workspaces deleted with everything they owned
    workspaces: int
    #: sign-ups deleted
    signups: int


@dataclass(frozen=True)
class Lineage:
    """How a graph version came to be: the extraction it started from + approved patches on top."""

    extraction_hash: str | None
    patch_ids: tuple[str, ...]


_EMPTY_LINEAGE: Final = Lineage(extraction_hash=None, patch_ids=())


@dataclass(frozen=True)
class _DiagnosisPlan:
    """What one diagnosis starts from, read from the store before the (long) pipeline runs.

    ``pending`` is the drill file the run resolves; it is cleared after the commit only if no newer
    upload replaced it in the meantime.
    """

    bundle: InputBundle
    base_graph: OpsGraph | None
    options: DiagnosisOptions
    pending: tuple[bytes, ...]


# --------------------------------------------------------------------------- pure helpers


def _strings(value: object) -> tuple[str, ...]:
    return tuple(item for item in value if isinstance(item, str)) if isinstance(value, list) else ()


def diagnosis_lineages(entries: Iterable[AuditEntry]) -> dict[int, Lineage]:
    """Lineage recorded by full diagnoses, per version number (the latest record of a version wins).

    Only ``kind == "diagnosis"`` records carry an extraction; approvals and reverts derive theirs
    from the version history.
    """
    out: dict[int, Lineage] = {}
    for entry in entries:
        payload = entry.payload
        if entry.action != AUDIT_DIAGNOSIS or payload.get("kind") != "diagnosis":
            continue
        version = payload.get("version")
        extraction = payload.get("extraction_hash")
        if isinstance(version, int) and not isinstance(version, bool):
            out[version] = Lineage(
                extraction_hash=extraction if isinstance(extraction, str) else None,
                patch_ids=_strings(payload.get("reapplied_patch_ids")),
            )
    return out


def resolve_lineage(versions: Iterable[GraphVersion], recorded: Mapping[int, Lineage]) -> dict[int, Lineage]:
    """Lineage of every version, oldest first (parents and revert targets always come earlier).

    * a version a full diagnosis confirmed → what that diagnosis recorded (latest knowledge);
    * a patch version → its parent's lineage plus that patch;
    * a revert → the lineage of the version it restores (same graph, same history);
    * anything else (a graph committed without a record) → unknown extraction, no patches.
    """
    out: dict[int, Lineage] = {}
    for version in sorted(versions, key=lambda v: v.number):
        if version.number in recorded:
            out[version.number] = recorded[version.number]
        elif version.patch_id is not None:
            parent = out.get(version.parent, _EMPTY_LINEAGE) if version.parent is not None else _EMPTY_LINEAGE
            out[version.number] = Lineage(parent.extraction_hash, (*parent.patch_ids, version.patch_id))
        elif version.revert_of is not None:
            out[version.number] = out.get(version.revert_of, _EMPTY_LINEAGE)
        else:
            out[version.number] = _EMPTY_LINEAGE
    return out


def merge_rehearsal_history(
    current: Sequence[RehearsalResult],
    previous: Iterable[RehearsalResult],
    version_of: Mapping[str, int],
) -> tuple[RehearsalResult, ...]:
    """Current results plus, per scenario, its earliest earlier result (the "before" of a fix).

    The evidence report pairs each scenario's result on the reported graph with its oldest other
    result to show "실패 → 통과". Carrying exactly one earlier result per scenario keeps that
    comparison across any number of approvals without letting reports grow with the history.
    Results on graphs the history does not know rank after known ones.
    """
    hashes = {r.graph_hash for r in current}
    scenario_ids = {r.scenario_id for r in current}
    unknown = max(version_of.values(), default=0) + 1
    earliest: dict[str, RehearsalResult] = {}
    for result in previous:
        if result.scenario_id not in scenario_ids or result.graph_hash in hashes:
            continue
        best = earliest.get(result.scenario_id)
        if best is None or version_of.get(result.graph_hash, unknown) < version_of.get(
            best.graph_hash, unknown
        ):
            earliest[result.scenario_id] = result
    return (*current, *(earliest[sid] for sid in sorted(earliest)))


def run_id_for(workspace_id: str, sequence: int, created_at: datetime, graph_hash: str) -> str:
    """Report id ``run-{sequence:06d}-{digest}``: unique per workspace, deterministic, sortable.

    The zero-padded sequence makes ids sort in run order, which is the store's tie-breaker when
    two reports share a timestamp (a fixed demo clock); the digest keeps ids distinct across
    workspaces, because report ids are global keys.
    """
    payload = f"{workspace_id}\x00{sequence}\x00{created_at.isoformat()}\x00{graph_hash}"
    return f"run-{sequence:06d}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]}"


def drill_ids(carried: Iterable[Scenario], scenarios: Iterable[Scenario]) -> tuple[str, ...]:
    """The workspace's drill set after a run: what it carried plus every non-template drill it ran.

    Carried drills stay in the set even when this graph could not replay them (a node they name is
    gone) — a later graph may have it again, and a drill a team wrote must not vanish silently.
    Templates are not part of the set: they are regenerated for every graph.
    """
    ids = [s.id for s in carried]
    ids += [s.id for s in scenarios if s.origin is not ScenarioOrigin.TEMPLATE]
    return tuple(dict.fromkeys(ids))


def _ai_hypotheses(report: DiagnosisReport | None) -> tuple[RiskHypothesis, ...]:
    """The AI risk hypotheses of a report, to verify again on the next graph.

    Sweep hypotheses are not carried: every run sweeps its own graph. AI ones would otherwise be
    lost by a re-judgement (no LLM) or an LLM-off re-diagnosis, although the replay can judge them
    on any graph that still has the nodes they name.
    """
    if report is None:
        return ()
    return tuple(h for h in report.risk_hypotheses if h.origin is HypothesisOrigin.AI)


def _bounded_concurrency(configured: int) -> int:
    """``settings.llm_concurrency`` within what :class:`DiagnosisOptions` accepts (1–64).

    ``JETTAE_LLM_CONCURRENCY`` is an operator's tuning knob for the model path; an out-of-range
    value must not make every diagnosis fail — least of all an LLM-OFF one, which never uses it
    (P4). Too high is capped, zero or less means one call at a time.
    """
    return min(max(configured, MIN_LLM_CONCURRENCY), MAX_LLM_CONCURRENCY)


def _committed_meanwhile(current: GraphVersion | None, outcome: DiagnosisOutcome) -> bool:
    """Is the moved head a diagnosis commit of exactly ``outcome``'s graph, at the version its
    drafts target? (See :meth:`JettaeService._commit_diagnosis`.)"""
    return (
        current is not None
        and current.patch_id is None
        and current.revert_of is None
        and current.number == outcome.draft_base_version
        and current.graph_hash == outcome.graph.content_hash()
    )


def _first_seen(versions: Iterable[GraphVersion]) -> dict[str, int]:
    """Graph hash → the first version number that had it (a revert re-creates an older graph)."""
    out: dict[str, int] = {}
    for version in sorted(versions, key=lambda v: v.number):
        out.setdefault(version.graph_hash, version.number)
    return out


@dataclass(frozen=True)
class _DraftBasis:
    """What on-demand drafting judges against: the head's findings, the chosen ones, its drills."""

    findings: tuple[Finding, ...]
    selected: tuple[Finding, ...]
    scenarios: tuple[Scenario, ...]
    results: tuple[RehearsalResult, ...]


def _draft_basis(
    ctx: CheckContext, drills: Iterable[Scenario], finding_ids: Sequence[str] | None
) -> _DraftBasis:
    """Judge the head (rules, score, replay of its drills) and pick the findings to draft for.

    An id that is not a finding of the head raises :class:`NotFoundError`: a fix for a gap the
    current graph does not have would be validated against nothing.
    """
    findings, _readiness = assess(ctx)
    selected = findings
    if finding_ids is not None:
        by_id = {f.id: f for f in findings}
        missing = [fid for fid in finding_ids if fid not in by_id]
        if missing:
            raise NotFoundError(f"현재 그래프에 없는 끊김입니다: {', '.join(missing)}")
        selected = tuple(by_id[fid] for fid in dict.fromkeys(finding_ids))
    scenarios, _notes = gather_scenarios(ctx, carried=drills)
    results = rehearse_all(ctx.graph, scenarios, ctx=ctx, config=_DEFAULT_SIM)
    return _DraftBasis(findings=findings, selected=selected, scenarios=scenarios, results=results)


def _workspace_not_found(workspace_id: str) -> NotFoundError:
    """The one "unknown workspace" error (404) — also what a visitor without the key gets."""
    return NotFoundError(f"워크스페이스 {workspace_id}을(를) 찾을 수 없습니다.")


def _key_digest(key: str) -> str:
    """SHA-256 hex digest of an access key (text that cannot be encoded never matches anyway)."""
    return hashlib.sha256(key.encode("utf-8", errors="replace")).hexdigest()


def email_fingerprint(email: str) -> str:
    """Short SHA-256 fingerprint (12 hex digits) of an e-mail address, normalised like the sign-up
    key, for audit entries that must name a person without storing the address: whoever holds the
    address can confirm the entry concerns it, the log alone does not reveal it."""
    key = signup_key(email) or email.strip().casefold()
    return hashlib.sha256(key.encode("utf-8", errors="replace")).hexdigest()[:12]


def _masked(text: str | None) -> str | None:
    return mask_pii(text)[0] if text else text


def mask_catalog_free_text(catalog: Iterable[CatalogService]) -> tuple[CatalogService, ...]:
    """Catalog services with their free text PII-masked like a runbook (:func:`jettae.ingest.mask_pii`):
    ``description``, ``owner`` and ``location`` (a region or datacenter label). The identifiers the
    diagnosis matches on — ``name``, ``aliases``, ``depends_on`` — and ``kind``/``tier`` are kept.
    A service without personal data comes back as the same object."""
    out: list[CatalogService] = []
    for svc in catalog:
        update = {
            field: masked
            for field, value in (
                ("description", svc.description),
                ("owner", svc.owner),
                ("location", svc.location),
            )
            if (masked := _masked(value)) != value
        }
        out.append(svc.model_copy(update=update) if update else svc)
    return tuple(out)


def mask_roster_free_text(roster: Iterable[RosterEntry]) -> tuple[RosterEntry, ...]:
    """Roster entries with their free-text labels PII-masked: ``title`` (직급), ``team`` and
    ``aliases``. What the diagnosis needs is kept as uploaded: ``name``, ``status``, ``roles``,
    ``channels`` (only the kind of medium — the loader keeps no number or address) and ``left_on``
    (a date). An entry without personal data comes back as the same object."""
    out: list[RosterEntry] = []
    for entry in roster:
        update: dict[str, object] = {
            field: masked
            for field, value in (("title", entry.title), ("team", entry.team))
            if (masked := _masked(value)) != value
        }
        aliases = tuple(mask_pii(alias)[0] for alias in entry.aliases)
        if aliases != entry.aliases:
            update["aliases"] = aliases
        out.append(entry.model_copy(update=update) if update else entry)
    return tuple(out)


# --------------------------------------------------------------------------- the service


class JettaeService:
    """Application service: every use case of JETTAE Ops (see the module docstring)."""

    def __init__(
        self,
        store: Store,
        settings: Settings,
        *,
        clock: Clock = utc_now,
        llm_factory: LlmFactory = build_llm,
    ) -> None:
        self.store = store
        self.settings = settings
        self.clock = clock
        self.llm_factory = llm_factory
        self.versions_service = VersionService(store.graphs, store.patches, store.audit, clock)
        self._texts: dict[str, dict[str, Document]] = {}
        self._pending_scenarios: dict[str, tuple[bytes, ...]] = {}
        self._locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()
        self._demo_guard = threading.Lock()

    def derive(
        self, *, settings: Settings | None = None, llm_factory: LlmFactory | None = None
    ) -> JettaeService:
        """A sibling service on the same store and clock with other settings or another LLM factory
        (e.g. one request served by another provider, or with every LIVE answer also appended to a
        cassette via :func:`~jettae.orchestrator.demo.recording_factory`).

        Both live in one process and serve the same workspaces, so they also share the
        process-local state: runbook texts kept when raw storage is off, drill files waiting for a
        diagnosis, and the per-workspace locks (two services racing on one workspace would
        otherwise commit against each other's head)."""
        sibling = JettaeService(
            self.store,
            settings if settings is not None else self.settings,
            clock=self.clock,
            llm_factory=llm_factory if llm_factory is not None else self.llm_factory,
        )
        sibling._texts = self._texts
        sibling._pending_scenarios = self._pending_scenarios
        sibling._locks = self._locks
        sibling._locks_guard = self._locks_guard
        sibling._demo_guard = self._demo_guard
        return sibling

    # ------------------------------------------------------------------ workspaces
    def create_workspace(self, name: str, *, actor: str = "system") -> Workspace:
        """A new workspace; ``actor`` (the caller, ``system`` for internal use) goes to the audit log."""
        self._require_actor(actor)
        workspace = self.store.workspaces.create_workspace(name)
        self.store.audit.record(workspace.id, actor, "workspace.created", {"name": workspace.name})
        return workspace

    def list_workspaces(self) -> list[Workspace]:
        return self.store.workspaces.list_workspaces()

    def get_workspace(self, workspace_id: str) -> Workspace:
        workspace = self.store.workspaces.get_workspace(workspace_id)
        if workspace is None:
            raise _workspace_not_found(workspace_id)
        return workspace

    # ------------------------------------------------------------------ private workspaces
    def grant_workspace_key(self, workspace_id: str, *, actor: str = "system") -> str:
        """A fresh access key for the workspace, replacing any previous one.

        Only the key's SHA-256 digest is stored, so the plaintext returned here cannot be recovered
        later — not by an operator with database access either. 32 random bytes (URL-safe base64)
        make guessing hopeless, which is why a plain digest (no slow KDF) suffices.
        """
        self._require_actor(actor)
        self.get_workspace(workspace_id)
        key = secrets.token_urlsafe(32)
        self.store.access.set_access_key_digest(workspace_id, _key_digest(key))
        self.store.audit.record(workspace_id, actor, AUDIT_KEY_GRANTED, {})
        return key

    def workspace_key_matches(self, workspace_id: str, presented: str | None) -> bool:
        """Whether ``presented`` is the workspace's access key (constant-time comparison of digests).

        False when nothing was presented or the workspace has no key."""
        if not presented:
            return False
        stored = self.store.access.access_key_digest(workspace_id)
        if stored is None:
            return False
        return hmac.compare_digest(stored, _key_digest(presented))

    def workspace_for_visitor(self, workspace_id: str, access_key: str | None) -> Workspace:
        """The workspace as an anonymous visitor of the public site may see it.

        The demo workspace is open to everyone; any other workspace only with its access key.
        Otherwise the error is exactly the one for an unknown id, so a visitor without the key
        cannot even learn that the workspace exists.
        """
        workspace = self.get_workspace(workspace_id)
        if workspace.name == DEMO_WORKSPACE_NAME or self.workspace_key_matches(workspace_id, access_key):
            return workspace
        raise _workspace_not_found(workspace_id)

    # ------------------------------------------------------------------ inputs
    def upload_inputs(
        self,
        workspace_id: str,
        *,
        documents: Sequence[Document],
        roster: Sequence[RosterEntry] | None = None,
        oncall: Sequence[OnCallShift] | None = None,
        catalog: Sequence[CatalogService] | None = None,
        policy: Policy | None = None,
        scenarios: bytes | None = None,
        actor: str = "system",
    ) -> InputSummary:
        """Store a workspace's inputs for the next diagnosis.

        Non-empty ``documents`` *replace* the current runbook set. ``roster``/``oncall``/``catalog``/
        ``policy`` replace their previous value when given and keep it when ``None`` (``()`` means
        "provided, and empty"). ``scenarios`` is the raw bytes of a ``scenarios.yaml`` drill file,
        resolved onto the graph by the next diagnosis (never a path: an uploaded string must not be
        able to name a server file). ``actor`` (who uploaded) goes to the audit log.
        """
        self._require_actor(actor)
        self.get_workspace(workspace_id)
        guarded = tuple(scan_injection(doc)[0] for doc in documents)
        if scenarios is not None:
            # Checked at run time too: a str here would be read as YAML text *or a file path* by
            # the scenario loader, and a delivery layer passing a form field must not reach a file.
            if not isinstance(scenarios, bytes):
                raise ValueError("시나리오 파일은 올린 파일의 내용(바이트)으로만 받을 수 있습니다.")
            decode_text(scenarios)  # refuse undecodable bytes now rather than at the next diagnosis
        with self._lock(workspace_id):
            previous = self.store.inputs.load_inputs(workspace_id)
            merged = StoredInputs(
                roster=tuple(roster) if roster is not None else previous.roster,
                oncall=tuple(oncall) if oncall is not None else previous.oncall,
                catalog=tuple(catalog) if catalog is not None else previous.catalog,
                policy=policy if policy is not None else previous.policy,
            )
            if guarded:
                self.store.documents.save_documents(
                    workspace_id, guarded, store_raw=self.settings.store_raw_documents
                )
                if self.settings.store_raw_documents:
                    self._texts.pop(workspace_id, None)
                else:
                    self._texts[workspace_id] = {doc.doc_id: doc for doc in guarded}
            self.store.inputs.save_inputs(workspace_id, merged)
            if scenarios is not None:
                self._pending_scenarios[workspace_id] = (scenarios,)
            self.store.audit.record(
                workspace_id,
                actor,
                AUDIT_INPUTS,
                {
                    "doc_ids": [doc.doc_id for doc in guarded] if guarded else None,
                    "roster": len(merged.roster),
                    "oncall": len(merged.oncall),
                    "catalog": len(merged.catalog),
                    "scenario_file": scenarios is not None,
                },
            )
        return self.input_summary(workspace_id)

    def load_bundle(
        self,
        workspace_id: str,
        bundle: InputBundle,
        *,
        scenarios: bytes | None = None,
        actor: str = "system",
    ) -> InputSummary:
        """Replace every input of the workspace with ``bundle`` (an absent source becomes empty).

        A bundle is everything one diagnosis consumes, so one without runbooks is refused
        (``ValueError``) rather than silently keeping the previous runbook set next to new
        structured inputs.
        """
        if not bundle.documents:
            raise ValueError("입력 묶음에 런북 문서가 없습니다. runbooks/ 폴더에 런북을 넣어 주세요.")
        return self.upload_inputs(
            workspace_id,
            documents=bundle.documents,
            roster=bundle.roster,
            oncall=bundle.oncall,
            catalog=bundle.catalog,
            policy=bundle.policy,
            scenarios=scenarios,
            actor=actor,
        )

    def input_summary(self, workspace_id: str) -> InputSummary:
        self.get_workspace(workspace_id)
        active = self._active_documents(workspace_id)
        inputs = self.store.inputs.load_inputs(workspace_id)
        return InputSummary(
            workspace_id=workspace_id,
            documents=tuple(meta.to_ref() for meta in active),
            roster=len(inputs.roster),
            oncall=len(inputs.oncall),
            catalog=len(inputs.catalog),
            security_notices=sum(len(meta.quarantined) for meta in active),
            raw_stored=self.settings.store_raw_documents,
            diagnosable=bool(active) and all(self._text_available(workspace_id, m) for m in active),
            pending_scenario_files=len(self._pending_scenarios.get(workspace_id, ())),
        )

    # ------------------------------------------------------------------ diagnosis
    async def diagnose(
        self,
        workspace_id: str,
        *,
        llm_mode: LlmMode | None = None,
        actor: str = "system",
        rehearse: bool = True,
        draft: bool = True,
    ) -> DiagnosisReport:
        """Diagnose the current inputs, commit the graph, persist and return the report.

        The head (with its approved patches) is the base: unchanged inputs keep it as is, changed
        inputs re-apply the approved patches on the new extraction. Raises
        :class:`VersionConflictError` when another change was approved while this run was working.
        Store work and the locked commit run in worker threads, so an API server's event loop is
        never held by them (nor by a lock an approval in another thread is holding).
        """
        self._require_actor(actor)
        mode = self._mode(llm_mode)
        plan = await asyncio.to_thread(self._plan_diagnosis, workspace_id, rehearse=rehearse, draft=draft)
        outcome = await run_diagnosis(
            plan.bundle,
            base_graph=plan.base_graph,
            llm=self.llm_factory(self.settings, mode),
            llm_mode=mode,
            options=plan.options,
        )
        return await asyncio.to_thread(self._commit_diagnosis, plan, outcome, mode=mode, actor=actor)

    async def free_scan(
        self,
        *,
        documents: Sequence[Document],
        roster: Sequence[RosterEntry] | None = None,
        oncall: Sequence[OnCallShift] | None = None,
        catalog: Sequence[CatalogService] | None = None,
        policy: Policy | None = None,
        actor: str,
        llm_mode: LlmMode | None = None,
    ) -> FreeScanOutcome:
        """The public site's anonymous free scan (무료 검진): a new private workspace named
        :data:`FREE_SCAN_WORKSPACE_NAME`, the inputs uploaded into it, a normal diagnosis (the
        configured LLM mode unless ``llm_mode`` is given), a fresh access key, and the server-side
        demand event :data:`EVENT_RUNBOOK_UPLOAD` (``{"files": <runbook files>}``; a file uploaded
        twice counts once).

        A free scan is one-shot and keeps only its result (report, graph, evidence quotes, drafts):

        * with raw storage off (the default) its runbook texts are dropped from this process as soon
          as the diagnosis is done, so a stream of anonymous scans does not accumulate uploaded text;
        * the free text of the catalog and the roster is PII-masked before anything is assembled
          (:func:`mask_catalog_free_text`, :func:`mask_roster_free_text`), so phone numbers, RRNs or
          addresses in a service description never reach the stored graph; the roster's names,
          statuses, roles and contact media are what the checks match on and stay as uploaded;
        * its structured inputs (roster, on-call, catalog, policy) are not kept after the diagnosis
          that needed them;
        * template drills and replay results are stored only inside the report, not a second time
          in their own tables (see :meth:`_publish`);
        * a scan that fails leaves nothing behind: its workspace is deleted again (best effort —
          anything left over is removed by :meth:`purge_expired` with the other expired scans).
        """
        if not documents:
            raise ValueError("무료 검진에는 런북 파일이 하나 이상 필요합니다.")
        self._require_actor(actor)
        workspace = await asyncio.to_thread(
            lambda: self.create_workspace(FREE_SCAN_WORKSPACE_NAME, actor=actor)
        )
        masked_roster = mask_roster_free_text(roster) if roster is not None else None
        masked_catalog = mask_catalog_free_text(catalog) if catalog is not None else None
        try:
            await asyncio.to_thread(
                lambda: self.upload_inputs(
                    workspace.id,
                    documents=documents,
                    roster=masked_roster,
                    oncall=oncall,
                    catalog=masked_catalog,
                    policy=policy,
                    actor=actor,
                )
            )
            report = await self.diagnose(workspace.id, llm_mode=llm_mode, actor=actor)
            await asyncio.to_thread(self.store.inputs.save_inputs, workspace.id, StoredInputs())
            key = await asyncio.to_thread(lambda: self.grant_workspace_key(workspace.id, actor=actor))
        except Exception:
            await asyncio.to_thread(self._discard_quietly, workspace.id)
            raise
        finally:
            if not self.settings.store_raw_documents:
                self._texts.pop(workspace.id, None)
        files = len({doc.doc_id for doc in documents})
        await asyncio.to_thread(self.record_event, EVENT_RUNBOOK_UPLOAD, {"files": files})
        return FreeScanOutcome(workspace_id=workspace.id, access_key=key, report=report)

    def latest_report(self, workspace_id: str) -> DiagnosisReport:
        self.get_workspace(workspace_id)
        report = self.store.reports.latest_report(workspace_id)
        if report is None:
            raise NotFoundError(f"워크스페이스 {workspace_id}에는 아직 진단 결과가 없습니다.")
        return report

    def report_notes(self, workspace_id: str, run_id: str | None = None) -> tuple[str, ...]:
        """Korean remarks of a run (LLM fallback, skipped patch ops, dropped drills); latest by default."""
        target = run_id if run_id is not None else self.latest_report(workspace_id).run_id
        for entry in reversed(self.store.audit.entries(workspace_id)):
            if entry.action == AUDIT_DIAGNOSIS and entry.payload.get("run_id") == target:
                return _strings(entry.payload.get("notes"))
        raise NotFoundError(f"진단 실행 {target}을(를) 찾을 수 없습니다.")

    # ------------------------------------------------------------------ graph & versions
    def graph(self, workspace_id: str, version: int | None = None) -> tuple[GraphVersion, OpsGraph]:
        self.get_workspace(workspace_id)
        if version is None:
            found = self.store.graphs.head(workspace_id)
            if found is None:
                raise NotFoundError(
                    f"워크스페이스 {workspace_id}에는 아직 그래프가 없습니다. 먼저 진단하세요."
                )
        else:
            found = self.store.graphs.get_version(workspace_id, version)
        return found, self.store.graphs.load_graph(found.graph_hash)

    def versions(self, workspace_id: str) -> list[GraphVersion]:
        self.get_workspace(workspace_id)
        return self.versions_service.history(workspace_id)

    def diff(self, workspace_id: str, a: int, b: int) -> GraphDiff:
        self.get_workspace(workspace_id)
        return self.versions_service.diff(workspace_id, a, b)

    # ------------------------------------------------------------------ rehearsal
    async def rehearse(
        self,
        workspace_id: str,
        *,
        scenario_ids: Sequence[str] | None = None,
        generate: ScenarioGeneration = "all",
        llm_mode: LlmMode | None = None,
        actor: str = "system",
    ) -> list[RehearsalResult]:
        """Replay drills on the head graph and persist scenarios and results (audited as ``actor``).

        With ``scenario_ids`` exactly those drills run (current templates or stored drills; an
        unknown id raises :class:`NotFoundError`, and a stored drill naming something the head no
        longer has raises ``ValueError`` — perturbing a missing node would pass for the wrong
        reason, P3). Otherwise ``generate`` picks the sources:
        templates for this graph, the workspace's hand-written drills, its LLM drills plus fresh
        ones when the mode is LIVE/REPLAY, or all of them. Ad-hoc LLM drills are stored but do not
        join the drill set the reports carry. An LLM failure other than "LLM off" propagates — the
        caller asked for model-written drills. The replay and the locked writes run in worker threads.
        """
        if generate not in _GENERATED_ORIGINS:
            raise ValueError(
                f"리허설할 시나리오 종류를 알 수 없습니다: {generate!r} "
                f"(가능한 값: {', '.join(_GENERATED_ORIGINS)})"
            )
        self._require_actor(actor)
        head, graph = self.graph(workspace_id)
        ctx = self._context(workspace_id, graph)
        mode = self._mode(llm_mode)
        meter = UsageMeter()
        if scenario_ids is not None:
            known = {s.id: s for s in self.store.rehearsals.list_scenarios(workspace_id)}
            known.update((s.id, s) for s in template_scenarios(ctx))
            missing = [sid for sid in scenario_ids if sid not in known]
            if missing:
                raise NotFoundError(f"시나리오를 찾을 수 없습니다: {', '.join(missing)}")
            selected = [known[sid] for sid in dict.fromkeys(scenario_ids)]
            stale = [
                f"「{s.title}」({', '.join(gone)})"
                for s in selected
                if (gone := missing_references(graph, s))
            ]
            if stale:
                raise ValueError(
                    "현재 그래프에 없는 대상을 가리키는 시나리오는 리허설할 수 없습니다"
                    "(없는 것을 끊으면 이유 없이 통과합니다): " + "; ".join(stale)
                )
        else:
            drills = self._active_drills(workspace_id)
            selected = await self._generated(ctx, drills, generate, mode, meter)
        results = await asyncio.to_thread(rehearse_all, graph, selected, ctx=ctx, config=_DEFAULT_SIM)
        audit: dict[str, object] = {
            "version": head.number,
            "scenario_ids": [s.id for s in selected],
            "llm_mode": mode.value,
            "llm_calls": meter.calls,
        }
        await asyncio.to_thread(self._save_rehearsal, workspace_id, selected, results, audit, actor)
        return list(results)

    def scenarios(self, workspace_id: str) -> list[Scenario]:
        self.get_workspace(workspace_id)
        return self.store.rehearsals.list_scenarios(workspace_id)

    # ------------------------------------------------------------------ patches
    def patches(self, workspace_id: str) -> list[Patch]:
        self.get_workspace(workspace_id)
        return self.store.patches.list_patches(workspace_id)

    def get_patch(self, patch_id: str) -> Patch:
        return self.store.patches.get_patch(patch_id)

    def patch_for_visitor(self, patch_id: str, access_key: str | None) -> Patch:
        """A patch as an anonymous visitor may see it: only if they may open its workspace
        (:meth:`workspace_for_visitor`); otherwise the error for an unknown patch id."""
        patch = self.get_patch(patch_id)
        try:
            self.workspace_for_visitor(patch.workspace_id, access_key)
        except PortNotFoundError:
            raise NotFoundError(f"수정안 {patch_id}을(를) 찾을 수 없습니다.") from None
        return patch

    async def draft_patches(
        self,
        workspace_id: str,
        *,
        finding_ids: Sequence[str] | None = None,
        llm_mode: LlmMode | None = None,
        actor: str,
    ) -> list[Patch]:
        """Draft fixes for the head's findings (all, or ``finding_ids``) and return them validated.

        Templates write the ops; with LIVE/REPLAY an LLM may only reword the sentence and pick among
        roster candidates. Every draft is re-checked and re-rehearsed before it is returned. The
        judging (checks, replays, validation) and the locked writes run in worker threads.
        """
        self._require_actor(actor)
        head, graph = self.graph(workspace_id)
        ctx = self._context(workspace_id, graph)
        basis = await asyncio.to_thread(_draft_basis, ctx, self._active_drills(workspace_id), finding_ids)
        mode = self._mode(llm_mode)
        meter = UsageMeter()
        if mode is LlmMode.OFF:
            drafts = await asyncio.to_thread(
                TemplateDrafter().draft,
                ctx,
                basis.selected,
                workspace_id=workspace_id,
                base_version=head.number,
            )
        else:
            client = MeteredLLM(self.llm_factory(self.settings, mode), meter)
            drafts = await LlmDrafter(client).adraft(
                ctx, basis.selected, workspace_id=workspace_id, base_version=head.number
            )
        validated = await asyncio.to_thread(
            validate_drafts,
            ctx,
            drafts,
            scenarios=basis.scenarios,
            findings=basis.findings,
            results=basis.results,
            config=_DEFAULT_SIM,
        )
        audit: dict[str, object] = {"version": head.number, "llm_mode": mode.value, "llm_calls": meter.calls}
        return await asyncio.to_thread(self._save_drafts, workspace_id, head.number, validated, actor, audit)

    def approve_patch(self, patch_id: str, *, actor: str, role: str) -> ApprovalResult:
        """Approve a validated patch (lead/admin), then re-judge the new head without any LLM.

        The new version is checked, scored and rehearsed again, a new report is saved, and the
        remaining findings get fresh validated template drafts. A patch made stale by an earlier
        decision raises :class:`VersionConflictError`.
        """
        self._authorize(role)
        self._require_actor(actor)
        patch = self.store.patches.get_patch(patch_id)
        workspace_id = patch.workspace_id
        with self._lock(workspace_id):
            patch = self.store.patches.get_patch(patch_id)
            if patch.status is PatchStatus.SUPERSEDED:
                raise VersionConflictError(
                    f"수정안 {patch.id}은(는) 기준 버전({patch.base_version})이 바뀌어 무효가 되었습니다. "
                    "최신 진단의 수정안을 승인하세요."
                )
            version = self.versions_service.approve(patch_id, actor=actor, role=role)
            report = self._rejudge(workspace_id, version, actor=actor, kind="approval")
        return ApprovalResult(version=version, report=report)

    def reject_patch(self, patch_id: str, *, actor: str, role: str, reason: str) -> Patch:
        """A lead/admin turns a pending patch down; the reason goes to the audit log."""
        self._authorize(role)
        self._require_actor(actor)
        patch = self.store.patches.get_patch(patch_id)
        with self._lock(patch.workspace_id):
            return self.versions_service.reject(patch_id, actor=actor, role=role, reason=reason)

    def revert(self, workspace_id: str, to_version: int, *, actor: str, role: str) -> ApprovalResult:
        """Append a version equal to ``to_version`` (history is never rewritten) and re-judge it."""
        self._authorize(role)
        self._require_actor(actor)
        self.get_workspace(workspace_id)
        with self._lock(workspace_id):
            before = self.store.graphs.head(workspace_id)
            version = self.versions_service.revert(workspace_id, to_version, actor=actor, role=role)
            latest = self.store.reports.latest_report(workspace_id)
            unchanged = before is not None and version.number == before.number
            if unchanged and latest is not None and latest.graph_version == version.number:
                return ApprovalResult(version=version, report=latest)
            report = self._rejudge(workspace_id, version, actor=actor, kind="revert")
        return ApprovalResult(version=version, report=report)

    # ------------------------------------------------------------------ exports
    def evidence_markdown(self, workspace_id: str) -> str:
        return render_view_markdown(self._evidence(workspace_id))

    def evidence_html(self, workspace_id: str) -> str:
        return render_view_html(self._evidence(workspace_id))

    def sarif(self, workspace_id: str) -> dict[str, Any]:
        report = self.latest_report(workspace_id)
        graph = self.store.graphs.load_graph(report.graph_hash)
        return to_sarif(report, graph, documents=dict(graph.docs), tool_version=__version__)

    # ------------------------------------------------------------------ demand & catalog
    def signup(self, signup: Signup) -> bool:
        """Store a pre-launch sign-up unless the address already has one (True when stored).

        The model itself refuses a sign-up without privacy consent. The first submission is kept
        as it is: the public form has no e-mail verification, so a later one for the same address
        — possibly from someone else — must not rewrite that person's consents, answers or date.
        """
        return self.store.signups.add_signup_if_absent(signup)

    def delete_signup(self, email: str, *, actor: str) -> bool:
        """Delete one person's sign-up (withdrawal of consent) — for an administrator.

        Audited in :data:`ADMIN_AUDIT_SCOPE` with :func:`email_fingerprint` instead of the address,
        and whether a record existed. True when one was deleted.
        """
        self._require_actor(actor)
        deleted = self.store.signups.delete_signup(email)
        self.store.audit.record(
            ADMIN_AUDIT_SCOPE,
            actor,
            AUDIT_SIGNUP_DELETED,
            {"email_sha256": email_fingerprint(email), "deleted": deleted},
        )
        return deleted

    def list_signups(self, *, actor: str, export_format: str = "json") -> list[Signup]:
        """Every sign-up, by creation time then e-mail — for an administrator.

        Reading personal data is itself recorded in the audit log (scope :data:`ADMIN_AUDIT_SCOPE`):
        who read how many records, in which format. The records themselves are not copied there.
        """
        signups = self.store.signups.list_signups()
        self.store.audit.record(
            ADMIN_AUDIT_SCOPE, actor, AUDIT_SIGNUPS_READ, {"count": len(signups), "format": export_format}
        )
        return signups

    def record_event(self, name: str, props: Mapping[str, object]) -> None:
        self.store.events.record_event(name, props, self._now())

    def demand_metrics(self) -> DemandMetrics:
        return DemandMetrics(
            signups=self.store.signups.count_signups(), events=self.store.events.event_counts()
        )

    def rule_catalog(self) -> list[RuleInfo]:
        """The published fixed rules — exactly what the judge does, for customers to audit."""
        return fixed_rule_catalog()

    # ------------------------------------------------------------------ retention
    def purge_expired(self, now: datetime | None = None) -> PurgeOutcome:
        """Delete what the retention settings no longer allow keeping, as of ``now`` (default: the
        service clock; timezone-aware, ``ValueError`` otherwise).

        * free-scan workspaces (named :data:`FREE_SCAN_WORKSPACE_NAME`) created more than
          ``JETTAE_FREE_SCAN_RETENTION_DAYS`` ago, with everything they own (report, graph, drafts,
          access key, audit entries) — no other workspace is ever touched;
        * sign-ups created more than ``JETTAE_SIGNUP_RETENTION_DAYS`` ago.

        A setting of 0 keeps that kind forever. When anything was deleted, the counts (never ids or
        addresses) are audited in :data:`ADMIN_AUDIT_SCOPE`. Safe to run at any time and from
        several processes: deleting what is already gone is a no-op. The API server calls it on
        its own only in ``public`` mode (:mod:`jettae.api.app`); other installs keep everything.
        """
        moment = deletion_cutoff(now if now is not None else self._now())
        scan_days = self.settings.free_scan_retention_days
        signup_days = self.settings.signup_retention_days
        workspaces = 0
        if scan_days > 0:
            cutoff = moment - timedelta(days=scan_days)
            for workspace in self.store.workspaces.list_workspaces():
                expired = workspace.name == FREE_SCAN_WORKSPACE_NAME and workspace.created_at < cutoff
                if expired and self._discard_workspace(workspace.id):
                    workspaces += 1
        signups = 0
        if signup_days > 0:
            signups = self.store.signups.delete_signups_before(moment - timedelta(days=signup_days))
        if workspaces or signups:
            self.store.audit.record(
                ADMIN_AUDIT_SCOPE,
                "system",
                AUDIT_RETENTION_PURGED,
                {
                    "free_scan_workspaces": workspaces,
                    "signups": signups,
                    "free_scan_retention_days": scan_days,
                    "signup_retention_days": signup_days,
                },
            )
        return PurgeOutcome(workspaces=workspaces, signups=signups)

    # ------------------------------------------------------------------ demo
    async def bootstrap_demo(
        self,
        *,
        llm_mode: LlmMode = LlmMode.OFF,
        workspace_name: str = DEMO_WORKSPACE_NAME,
        actor: str = DEMO_ACTOR,
    ) -> DiagnosisReport:
        """Create (or reuse) the 별빛상점 demo workspace, load ``examples/dmc-shop`` incl. its
        hand-written drills, and diagnose it. Running it again reuses the workspace; unchanged
        inputs keep the head graph (and any approved patches). Loading (file I/O, parsing) runs
        in a worker thread. ``actor`` is who asked (the HTTP caller; :data:`DEMO_ACTOR` for the CLI
        and start-up set-up)."""
        self._require_actor(actor)
        workspace_id = await asyncio.to_thread(self._load_demo, workspace_name, actor)
        return await self.diagnose(workspace_id, llm_mode=llm_mode, actor=actor)

    def stored_demo_report(self, workspace_id: str) -> DiagnosisReport | None:
        """The latest report of ``workspace_id`` if it is the demo workspace; None for any other
        workspace (an unknown id too — the caller's normal path reports that) or a demo that was
        never diagnosed.

        What viewers and anonymous callers get instead of a new diagnosis of the shared demo."""
        workspace = self.store.workspaces.get_workspace(workspace_id)
        if workspace is None or workspace.name != DEMO_WORKSPACE_NAME:
            return None
        return self.store.reports.latest_report(workspace_id)

    def demo_report(self, workspace_name: str = DEMO_WORKSPACE_NAME) -> DiagnosisReport | None:
        """The latest report of the demo workspace; None when it does not exist or was never
        diagnosed (the workspace :meth:`bootstrap_demo` would reuse)."""
        name = validate_workspace_name(workspace_name)
        for workspace in self.store.workspaces.list_workspaces():
            if workspace.name == name:
                return self.store.reports.latest_report(workspace.id)
        return None

    # ------------------------------------------------------------------ internals
    def _load_demo(self, workspace_name: str, actor: str = DEMO_ACTOR) -> str:
        """Find or create the demo workspace (once per name, even under concurrent calls) and load
        the example bundle with its drill file into it; return the workspace id."""
        path = demo_bundle_path()
        bundle = load_bundle_dir(path)
        drills = path / DEMO_SCENARIOS_FILE
        name = validate_workspace_name(workspace_name)
        with self._demo_guard:
            existing = [w for w in self.store.workspaces.list_workspaces() if w.name == name]
            workspace = existing[0] if existing else self.create_workspace(name, actor=actor)
        self.load_bundle(
            workspace.id, bundle, scenarios=drills.read_bytes() if drills.is_file() else None, actor=actor
        )
        return workspace.id

    def _discard_workspace(self, workspace_id: str) -> bool:
        """Delete a workspace from the store and forget its process-local state (texts, drill
        files, lock); True when the store had anything under the id."""
        with self._lock(workspace_id):
            deleted = self.store.workspaces.delete_workspace(workspace_id)
            self._texts.pop(workspace_id, None)
            self._pending_scenarios.pop(workspace_id, None)
        with self._locks_guard:
            self._locks.pop(workspace_id, None)
        return deleted

    def _discard_quietly(self, workspace_id: str) -> None:
        """:meth:`_discard_workspace` for a failed free scan: the scan's own error is what the
        caller must see, so a failure here is swallowed (the retention purge retries later)."""
        with contextlib.suppress(Exception):
            self._discard_workspace(workspace_id)

    def _now(self) -> datetime:
        return self.clock()

    def _mode(self, llm_mode: LlmMode | None) -> LlmMode:
        return LlmMode(llm_mode) if llm_mode is not None else self.settings.llm_mode

    @staticmethod
    def _require_actor(actor: str) -> None:
        """Checked before anything is written (and before any model is asked): the audit log
        refuses a blank actor and text no store can keep (NUL, a lone surrogate), and drafts or a
        version saved before that refusal would be a change nobody can be held to (P7)."""
        if not actor.strip():
            raise ValueError("누가 요청했는지(actor)를 적어 주세요. 모든 변경은 감사 로그에 남습니다.")
        if "\x00" in actor or any("\ud800" <= ch <= "\udfff" for ch in actor):
            raise ValueError("요청자(actor)에 저장할 수 없는 문자(NUL·짝 없는 서로게이트)가 있습니다.")

    @staticmethod
    def _authorize(role: str) -> None:
        if role.strip().casefold() not in DECIDER_ROLES:
            raise PermissionError(
                "승인 권한이 없습니다(팀 리드·관리자만 승인·거절·되돌리기를 할 수 있습니다)."
            )

    def _lock(self, workspace_id: str) -> threading.RLock:
        with self._locks_guard:
            return self._locks.setdefault(workspace_id, threading.RLock())

    def _active_documents(self, workspace_id: str) -> list[StoredDocument]:
        """Metadata of the current runbook set (the latest upload that carried documents)."""
        doc_ids: tuple[str, ...] = ()
        for entry in reversed(self.store.audit.entries(workspace_id)):
            if entry.action == AUDIT_INPUTS and isinstance(entry.payload.get("doc_ids"), list):
                doc_ids = _strings(entry.payload["doc_ids"])
                break
        by_id = {meta.doc_id: meta for meta in self.store.documents.list_documents(workspace_id)}
        return [by_id[doc_id] for doc_id in doc_ids if doc_id in by_id]

    def _active_drills(self, workspace_id: str) -> tuple[Scenario, ...]:
        """Human/LLM drills in effect: the set the latest report recorded, as stored."""
        ids: tuple[str, ...] = ()
        for entry in reversed(self.store.audit.entries(workspace_id)):
            if entry.action == AUDIT_DIAGNOSIS:
                ids = _strings(entry.payload.get("drill_ids"))
                break
        stored = {s.id: s for s in self.store.rehearsals.list_scenarios(workspace_id)}
        return tuple(stored[sid] for sid in ids if sid in stored)

    def _text_available(self, workspace_id: str, meta: StoredDocument) -> bool:
        return meta.raw_stored or meta.doc_id in self._texts.get(workspace_id, {})

    def _text_of(self, workspace_id: str, meta: StoredDocument) -> Document | None:
        """The full document from this process's cache, else from the store when it kept the text."""
        cached = self._texts.get(workspace_id, {}).get(meta.doc_id)
        if cached is not None:
            return cached
        return self.store.documents.get_document(workspace_id, meta.doc_id) if meta.raw_stored else None

    def _bundle(self, workspace_id: str) -> InputBundle:
        active = self._active_documents(workspace_id)
        if not active:
            raise ValueError("진단할 런북이 없습니다. 먼저 런북 문서를 올려 주세요.")
        documents: list[Document] = []
        for meta in active:
            doc = self._text_of(workspace_id, meta)
            if doc is None:
                raise ValueError(
                    "원문을 저장하지 않는 설정(JETTAE_STORE_RAW_DOCUMENTS=false)이라 서버가 다시 "
                    "시작된 뒤나 일회성 무료 검진 뒤에는 런북을 다시 올려야 진단할 수 있습니다"
                    f"(문서: {meta.title})."
                )
            documents.append(doc)
        return self.store.inputs.load_inputs(workspace_id).to_bundle(documents)

    def _context(self, workspace_id: str, graph: OpsGraph) -> CheckContext:
        inputs = self.store.inputs.load_inputs(workspace_id)
        return build_context(
            graph, roster=inputs.roster, oncall=inputs.oncall, catalog=inputs.catalog, policy=inputs.policy
        )

    def _lineage(self, workspace_id: str, head: GraphVersion | None) -> Lineage:
        if head is None:
            return _EMPTY_LINEAGE
        recorded = diagnosis_lineages(self.store.audit.entries(workspace_id))
        resolved = resolve_lineage(self.store.graphs.list_versions(workspace_id), recorded)
        return resolved.get(head.number, _EMPTY_LINEAGE)

    @staticmethod
    def _commit_note(mode: LlmMode, extractor: str) -> str:
        how = "규칙 파서" if extractor == "rule" else "규칙 파서+LLM"
        return f"진단 추출({how}, LLM 모드 {mode.value})"

    async def _generated(
        self,
        ctx: CheckContext,
        carried: Iterable[Scenario],
        generate: ScenarioGeneration,
        mode: LlmMode,
        meter: UsageMeter,
    ) -> list[Scenario]:
        """The drills ``generate`` asks for; fresh LLM ideas only in LIVE/REPLAY, metered into ``meter``."""
        valid, _notes = gather_scenarios(ctx, carried=carried)
        wanted = _GENERATED_ORIGINS[generate]
        picked = {s.id: s for s in valid if s.origin in wanted}
        if generate in ("llm", "all") and mode is not LlmMode.OFF:
            client = MeteredLLM(self.llm_factory(self.settings, mode), meter)
            for scenario in await llm_scenarios(client, ctx):
                picked.setdefault(scenario.id, scenario)
        return list(picked.values())

    def _save_rehearsal(
        self,
        workspace_id: str,
        scenarios: Sequence[Scenario],
        results: Sequence[RehearsalResult],
        audit: dict[str, object],
        actor: str,
    ) -> None:
        """Persist an ad-hoc rehearsal and audit it (worker thread; takes the workspace lock)."""
        with self._lock(workspace_id):
            for scenario in scenarios:
                self.store.rehearsals.save_scenario(workspace_id, scenario)
            for result in results:
                self.store.rehearsals.save_result(workspace_id, result)
            self.store.audit.record(workspace_id, actor, "rehearsal.run", audit)

    def _save_drafts(
        self,
        workspace_id: str,
        version: int,
        patches: Sequence[Patch],
        actor: str,
        audit: dict[str, object],
    ) -> list[Patch]:
        """Persist on-demand drafts made against ``version`` — refused if the head moved meanwhile."""
        with self._lock(workspace_id):
            current = self.store.graphs.head(workspace_id)
            if current is None or current.number != version:
                raise VersionConflictError(
                    "수정안을 만드는 동안 그래프 버전이 바뀌었습니다. 다시 요청하세요."
                )
            stored = self._store_patches(patches, self._now())
            self.store.audit.record(
                workspace_id, actor, "patch.drafted", {**audit, "patch_ids": [p.id for p in stored]}
            )
        return list(stored)

    def _store_patches(self, patches: Iterable[Patch], now: datetime) -> tuple[Patch, ...]:
        """Persist drafts, never overwriting a human decision on a patch with the same content id."""
        out: list[Patch] = []
        for patch in patches:
            try:
                existing: Patch | None = self.store.patches.get_patch(patch.id)
            except PortNotFoundError:
                existing = None
            if existing is not None and existing.status in _DECIDED:
                out.append(existing)
                continue
            created = existing.created_at if existing is not None and existing.created_at else now
            stored = patch.model_copy(update={"created_at": created})
            self.store.patches.save_patch(stored)
            out.append(stored)
        return tuple(out)

    def _plan_diagnosis(self, workspace_id: str, *, rehearse: bool, draft: bool) -> _DiagnosisPlan:
        """Everything a diagnosis starts from: inputs, head graph, lineage, drills (store reads only)."""
        self.get_workspace(workspace_id)
        bundle = self._bundle(workspace_id)
        head = self.store.graphs.head(workspace_id)
        base_graph = self.store.graphs.load_graph(head.graph_hash) if head is not None else None
        lineage = self._lineage(workspace_id, head)
        previous = self.store.reports.latest_report(workspace_id)
        # A new drill file replaces the hand-written drills; it is resolved only when drills run.
        pending = self._pending_scenarios.get(workspace_id, ()) if rehearse else ()
        carried = tuple(
            s for s in self._active_drills(workspace_id) if not (pending and s.origin is ScenarioOrigin.HUMAN)
        )
        options = DiagnosisOptions(
            workspace_id=workspace_id,
            base_version=head.number if head is not None else None,
            base_extraction_hash=lineage.extraction_hash,
            approved_patches=tuple(self.store.patches.get_patch(pid) for pid in lineage.patch_ids),
            carried_scenarios=carried,
            scenario_sources=pending,
            rehearse=rehearse,
            draft=draft,
            carried_hypotheses=_ai_hypotheses(previous),
            llm_concurrency=_bounded_concurrency(self.settings.llm_concurrency),
        )
        return _DiagnosisPlan(bundle=bundle, base_graph=base_graph, options=options, pending=pending)

    def _commit_diagnosis(
        self, plan: _DiagnosisPlan, outcome: DiagnosisOutcome, *, mode: LlmMode, actor: str
    ) -> DiagnosisReport:
        """Commit the run's graph and publish its report — unless the head moved meanwhile.

        One kind of move is not a conflict: another *diagnosis* committed exactly this run's
        graph as the very version this run's drafts were made against (two "demo" clicks, two
        first diagnoses racing). Nothing this run computed refers to anything else — same graph,
        same version number inside every patch id — so it is published like a run that found the
        head unchanged, and the idempotent request stays idempotent. A head moved by an approval
        or a revert is always a conflict: this run's lineage does not know that decision.
        """
        options = plan.options
        workspace_id = options.workspace_id
        with self._lock(workspace_id):
            current = self.store.graphs.head(workspace_id)
            moved = (current.number if current is not None else None) != options.base_version
            if moved and not _committed_meanwhile(current, outcome):
                raise VersionConflictError(
                    "진단하는 동안 다른 변경이 승인되어 그래프 버전이 바뀌었습니다. 다시 진단하세요."
                )
            version = self.versions_service.commit_graph(
                workspace_id, outcome.graph, actor=actor, note=self._commit_note(mode, outcome.extractor)
            )
            # Drafts based on an older head can never be approved; say so even when the head stayed.
            self.versions_service.supersede_pending(workspace_id, actor=actor)
            report = self._publish(
                workspace_id,
                version,
                kind="diagnosis",
                actor=actor,
                findings=outcome.findings,
                readiness=outcome.readiness,
                scenarios=outcome.scenarios,
                rehearsals=outcome.rehearsals,
                patches=outcome.patches,
                grounding=outcome.grounding,
                security=outcome.security,
                usage=outcome.usage,
                held_ids=outcome.held_ids,
                catalog_gaps=outcome.catalog_gaps,
                timings=outcome.timings,
                notes=outcome.notes,
                extraction_hash=outcome.extraction_hash,
                reapplied=outcome.reapplied_patch_ids,
                drills=drill_ids(options.carried_scenarios, outcome.scenarios),
                risk_hypotheses=outcome.risk_hypotheses,
                exploration=outcome.exploration,
            )
            pending = plan.pending
            if pending and self._pending_scenarios.get(workspace_id) == pending:
                self._pending_scenarios.pop(workspace_id, None)
        return report

    def _rejudge(
        self, workspace_id: str, version: GraphVersion, *, actor: str, kind: RunKind
    ) -> DiagnosisReport:
        """Re-judge a version created by a patch or a revert (no documents, no LLM — P4)."""
        graph = self.store.graphs.load_graph(version.graph_hash)
        inputs = self.store.inputs.load_inputs(workspace_id)
        previous = self.store.reports.latest_report(workspace_id)
        carried = self._active_drills(workspace_id)
        evaluation = evaluate_graph(
            graph,
            roster=inputs.roster,
            oncall=inputs.oncall,
            catalog=inputs.catalog,
            policy=inputs.policy,
            workspace_id=workspace_id,
            version=version.number,
            carried=carried,
            config=_DEFAULT_SIM,
            previous_hypotheses=_ai_hypotheses(previous),
        )
        exploration = evaluation.exploration
        return self._publish(
            workspace_id,
            version,
            kind=kind,
            actor=actor,
            findings=evaluation.findings,
            readiness=evaluation.readiness,
            scenarios=evaluation.scenarios,
            rehearsals=evaluation.rehearsals,
            patches=evaluation.patches,
            grounding=previous.grounding if previous is not None else GroundingStatus(),
            security=previous.security if previous is not None else (),
            usage=LlmUsageSummary(mode=LlmMode.OFF),
            held_ids=evaluation.held_ids,
            catalog_gaps=evaluation.catalog_gaps,
            timings=evaluation.timings,
            notes=evaluation.notes,
            extraction_hash=None,
            reapplied=(),
            drills=drill_ids(carried, evaluation.scenarios),
            risk_hypotheses=exploration.hypotheses if exploration is not None else (),
            exploration=exploration.summary if exploration is not None else None,
        )

    def _publish(
        self,
        workspace_id: str,
        version: GraphVersion,
        *,
        kind: RunKind,
        actor: str,
        findings: tuple[Finding, ...],
        readiness: ReadinessReport,
        scenarios: tuple[Scenario, ...],
        rehearsals: tuple[RehearsalResult, ...],
        patches: tuple[Patch, ...],
        grounding: GroundingStatus,
        security: tuple[SecurityNotice, ...],
        usage: LlmUsageSummary,
        held_ids: tuple[str, ...],
        catalog_gaps: tuple[str, ...],
        timings: dict[str, float],
        notes: tuple[str, ...],
        extraction_hash: str | None,
        reapplied: tuple[str, ...],
        drills: tuple[str, ...],
        risk_hypotheses: tuple[RiskHypothesis, ...] = (),
        exploration: ExplorationSummary | None = None,
    ) -> DiagnosisReport:
        """Persist scenarios, results, patches and the report of one run; audit it (caller holds the lock).

        A free-scan workspace stores leaner: its template drills and its replay results live only
        in the report. Nothing reads them back from their own tables there — templates are derived
        from the graph again whenever drills run, replay results are never listed, and the web app's
        free-scan pages read the report — while drills a person or a model wrote still go to the
        scenario table (the drill set refers to them), and patches stay in theirs (listing,
        decisions and the evidence report read them there).
        """
        now = self._now()
        previous = self.store.reports.latest_report(workspace_id)
        lean = self.get_workspace(workspace_id).name == FREE_SCAN_WORKSPACE_NAME
        for scenario in scenarios:
            if not (lean and scenario.origin is ScenarioOrigin.TEMPLATE):
                self.store.rehearsals.save_scenario(workspace_id, scenario)
        if not lean:
            for result in rehearsals:
                self.store.rehearsals.save_result(workspace_id, result)
        stored = self._store_patches(patches, now)
        history = merge_rehearsal_history(
            rehearsals,
            previous.rehearsals if previous is not None else (),
            _first_seen(self.store.graphs.list_versions(workspace_id)),
        )
        sequence = 1 + sum(1 for e in self.store.audit.entries(workspace_id) if e.action == AUDIT_DIAGNOSIS)
        report = DiagnosisReport(
            run_id=run_id_for(workspace_id, sequence, now, version.graph_hash),
            workspace_id=workspace_id,
            graph_version=version.number,
            graph_hash=version.graph_hash,
            created_at=now,
            findings=findings,
            readiness=readiness,
            scenarios=scenarios,
            rehearsals=history,
            patches=stored,
            grounding=grounding,
            security=security,
            llm=usage,
            held_element_ids=held_ids,
            catalog_gaps=catalog_gaps,
            timings_ms=timings,
            risk_hypotheses=risk_hypotheses,
            exploration=exploration,
        )
        self.store.reports.save_report(report)
        self.store.audit.record(
            workspace_id,
            actor,
            AUDIT_DIAGNOSIS,
            {
                "run_id": report.run_id,
                "kind": kind,
                "version": version.number,
                "graph_hash": version.graph_hash,
                "extraction_hash": extraction_hash,
                "reapplied_patch_ids": list(reapplied),
                "drill_ids": list(drills),
                "llm_mode": usage.mode.value,
                "llm_calls": usage.calls,
                "findings": len(findings),
                "overall": readiness.overall,
                "notes": list(notes),
            },
        )
        return report

    def _evidence(self, workspace_id: str) -> EvidenceView:
        workspace = self.get_workspace(workspace_id)
        report = self.latest_report(workspace_id)
        graph = self.store.graphs.load_graph(report.graph_hash)
        listed = {p.id for p in report.patches}
        patches = [
            p
            for p in self.store.patches.list_patches(workspace_id)
            if p.id in listed or p.status in (PatchStatus.APPROVED, PatchStatus.REJECTED)
        ]
        return build_evidence(
            report,
            graph,
            versions=self.store.graphs.list_versions(workspace_id),
            patches=patches,
            org_name=workspace.name,
        )
