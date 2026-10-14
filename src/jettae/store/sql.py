"""SQLAlchemy 2.0 implementation of :class:`~jettae.store.base.Store` (SQLite and Postgres).

Concurrency model — and why it is portable:

* Every repository call is one short transaction.
* ``append_version`` checks the head and inserts the next version in one transaction, but the
  real guard is the composite primary key ``(workspace_id, number)``: when two writers race
  past the head check, the database rejects the second insert and we report
  :class:`~jettae.domain.ports.VersionConflictError`. This behaves the same on SQLite (whose
  driver begins transactions lazily) and on Postgres READ COMMITTED, with no
  ``SELECT … FOR UPDATE`` or other dialect-specific SQL.
* Graph blobs are immutable and content-addressed, so each is written in its own transaction
  just before the version row. An unreferenced blob is harmless (and is reused when the caller
  retries with the same graph); a concurrent insert of the same blob is identical by definition.
* Engines that share a single DBAPI connection (``StaticPool``, used for in-memory SQLite) are
  serialised with a lock, because one sqlite3 connection cannot interleave transactions.

``create_schema=True`` uses ``metadata.create_all`` for development and tests; deployments run
the Alembic migrations instead (:func:`upgrade_schema`, or ``alembic upgrade head``).
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Final, TypeVar

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, Table, create_engine, delete, func, select, update
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import InstrumentedAttribute, Session, sessionmaker
from sqlalchemy.pool import StaticPool

from jettae.domain import DiagnosisReport, Document, GraphVersion, OpsGraph, Patch, RehearsalResult, Scenario
from jettae.store.base import (
    AuditEntry,
    Clock,
    NotFoundError,
    Signup,
    StoredDocument,
    StoredInputs,
    Workspace,
    as_utc,
    check_documents,
    check_key_digest,
    check_owner,
    check_patch,
    check_report,
    check_result,
    check_scenario,
    check_workspace_key,
    create_workspace_with,
    decode_graph,
    deletion_cutoff,
    document_order,
    json_payload,
    patch_order,
    prepare_append,
    rehearsal_id,
    report_order,
    restore_document,
    result_order,
    scenario_order,
    signup_key,
    signup_order,
    to_json_object,
    utc_now,
    validate_audit,
    validate_event_name,
    version_conflict,
    workspace_order,
)
from jettae.store.models import (
    AuditRow,
    Base,
    DocumentRow,
    EventRow,
    GraphBlobRow,
    GraphVersionRow,
    InputsRow,
    PatchRow,
    RehearsalResultRow,
    ReportRow,
    ScenarioRow,
    SignupRow,
    WorkspaceAccessKeyRow,
    WorkspaceRow,
)

#: Directory holding ``env.py`` and ``versions/`` (shipped inside the package).
MIGRATIONS_DIR = Path(__file__).parent / "migrations"

#: Rows keyed by a global id but owned by one workspace.
_Owned = TypeVar("_Owned", PatchRow, ReportRow)

#: Every table whose rows belong to one workspace (a ``workspace_id`` column). Derived from the
#: schema rather than listed, so a table added later is deleted with its workspace automatically.
WORKSPACE_TABLES: Final[tuple[Table, ...]] = tuple(
    table for table in Base.metadata.sorted_tables if "workspace_id" in table.c
)


# --------------------------------------------------------------------------- engine & migrations


def is_memory_sqlite(url: str | URL) -> bool:
    parsed = make_url(url)
    return parsed.get_backend_name() == "sqlite" and (
        parsed.database in (None, "", ":memory:") or parsed.query.get("mode") == "memory"
    )


#: Driver-less PostgreSQL schemes as hosting providers hand them out (Render, Heroku: ``postgres://``).
_BARE_POSTGRES = frozenset({"postgres", "postgresql"})
#: The PostgreSQL driver the ``postgres`` extra installs (psycopg 3).
POSTGRES_DRIVER = "postgresql+psycopg"


def normalize_database_url(url: str | URL) -> URL:
    """``url`` with a driver-less PostgreSQL scheme pointed at psycopg 3.

    Hosting providers inject ``postgres://…`` or ``postgresql://…``: SQLAlchemy 2 no longer knows
    the former, and resolves the latter to psycopg2, which the ``postgres`` extra does not install.
    An explicit driver (``postgresql+psycopg2``, ``postgresql+asyncpg``) and every other backend,
    SQLite included, are left exactly as given.
    """
    parsed = make_url(url)
    if parsed.drivername in _BARE_POSTGRES:
        return parsed.set(drivername=POSTGRES_DRIVER)
    return parsed


def build_engine(url: str | URL) -> Engine:
    """Engine with the pooling each backend needs.

    In-memory SQLite lives inside one connection, so it gets ``StaticPool`` (the default
    per-thread pool would hand every thread a different, empty database). File SQLite gets
    its parent directory created so the default ``./data/jettae.db`` works out of the box.
    A driver-less PostgreSQL URL is normalised first (:func:`normalize_database_url`).

    Every engine hides statement parameters from its error messages: they are the stored values
    themselves — a sign-up's e-mail and free-text message among them — and an error message ends
    up in tracebacks and logs (see also :func:`database_error_summary`).
    """
    parsed = normalize_database_url(url)
    if parsed.get_backend_name() != "sqlite":
        return create_engine(parsed, pool_pre_ping=True, hide_parameters=True)
    if is_memory_sqlite(parsed):
        return create_engine(
            parsed,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
            hide_parameters=True,
        )
    if parsed.database:
        Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(parsed, connect_args={"check_same_thread": False}, hide_parameters=True)


#: Root packages of the DBAPI drivers a store can run on; their exceptions quote SQL values too.
_DRIVER_PACKAGES: Final = frozenset({"sqlite3", "_sqlite3", "psycopg", "psycopg2", "psycopg_pool", "asyncpg"})


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    """``exc``, then its causes, contexts and group members (each once, cycles tolerated)."""
    seen: set[int] = set()
    pending: list[BaseException] = [exc]
    while pending:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        pending.extend(link for link in (current.__cause__, current.__context__) if link is not None)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)


def database_error_summary(exc: BaseException) -> dict[str, str] | None:
    """Log-safe description of a database failure in ``exc`` or its chain: class names only.

    None when no database error is involved. Otherwise ``{"db_error": …}`` (the SQLAlchemy class)
    and/or ``{"db_driver_error": …}`` (the DBAPI class, e.g. ``UniqueViolation``). The *messages*
    must never be logged: ``hide_parameters`` (see :func:`build_engine`) removes SQLAlchemy's
    ``[parameters: …]`` part, but the driver's own text still quotes values (PostgreSQL's
    ``DETAIL: Key (email)=(…) already exists``), and a traceback prints every message of the
    chain. So a caller logging such an exception logs these names, never ``str(exc)`` or
    ``exc_info``.
    """
    for link in _exception_chain(exc):
        if isinstance(link, SQLAlchemyError):
            names = {"db_error": type(link).__name__}
            orig = getattr(link, "orig", None)
            if isinstance(orig, BaseException):
                names["db_driver_error"] = type(orig).__name__
            return names
        if type(link).__module__.partition(".")[0] in _DRIVER_PACKAGES:
            return {"db_driver_error": type(link).__name__}
    return None


def alembic_config(url: str | None = None) -> Config:
    """Programmatic Alembic config pointing at the packaged migrations.

    ``%`` is doubled because Alembic stores options in a ``ConfigParser`` that would otherwise
    treat percent-encoded URL characters (``%40`` in passwords) as interpolation.
    """
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR).replace("%", "%%"))
    if url is not None:
        cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return cfg


def upgrade_schema(url: str, revision: str = "head") -> None:
    """Run Alembic migrations up to ``revision`` (what deployments use instead of create_all)."""
    command.upgrade(alembic_config(url), revision)


class _Database:
    """Session factory plus the lock needed by engines that share one connection."""

    def __init__(self, engine: Engine) -> None:
        self._sessions = sessionmaker(engine, expire_on_commit=False)
        self._lock: AbstractContextManager[Any] = (
            threading.RLock() if isinstance(engine.pool, StaticPool) else nullcontext()
        )

    @contextmanager
    def transaction(self) -> Iterator[Session]:
        with self._lock, self._sessions.begin() as session:
            yield session

    def upsert(self, rows: Sequence[Base]) -> None:
        """Insert-or-replace by primary key, atomically for all ``rows``.

        ``merge`` is SELECT-then-INSERT/UPDATE; if a concurrent writer inserts the same key in
        between, the retry sees that row and becomes an UPDATE, so one retry always suffices.
        """
        try:
            self._merge(rows)
        except IntegrityError:
            self._merge(rows)

    def _merge(self, rows: Sequence[Base]) -> None:
        with self.transaction() as session:
            for row in rows:
                session.merge(row)

    def save_owned(
        self, model: type[_Owned], key: InstrumentedAttribute[str], values: dict[str, Any], what: str
    ) -> None:
        """Upsert a globally keyed row without ever moving it to another workspace.

        A read-then-``merge`` cannot enforce ownership: under READ COMMITTED (and SQLite's
        deferred transactions) a concurrent insert can land between the ownership read and
        merge's own read, turning the merge into an UPDATE of the other workspace's row. The
        ownership condition therefore lives in the UPDATE's WHERE clause, which the database
        evaluates atomically; a missing row is INSERTed, and a lost insert race surfaces as an
        IntegrityError, after which the winning row's owner is judged.
        """
        record_id: str = values[key.key]
        workspace_id: str = values["workspace_id"]
        try:
            self._update_or_insert(model, key, values)
        except IntegrityError:
            with self.transaction() as session:
                owner = session.scalar(select(model.workspace_id).where(key == record_id))
            check_owner(what, record_id, owner, workspace_id)
            # The winner was this workspace: the row now exists, so this becomes an UPDATE.
            self._update_or_insert(model, key, values)

    def _update_or_insert(
        self, model: type[_Owned], key: InstrumentedAttribute[str], values: dict[str, Any]
    ) -> None:
        owned = (key == values[key.key]) & (model.workspace_id == values["workspace_id"])
        with self.transaction() as session:
            # Core execution on the session's connection: same transaction, and a typed rowcount.
            updated = session.connection().execute(update(model).where(owned).values(values))
            if updated.rowcount == 0:
                session.add(model(**values))


# --------------------------------------------------------------------------- repositories


class SqlWorkspaceRepository:
    def __init__(self, db: _Database, clock: Clock) -> None:
        self._db = db
        self._clock = clock

    def create_workspace(self, name: str, *, workspace_id: str | None = None) -> Workspace:
        return create_workspace_with(self._insert, self._clock, name, workspace_id)

    def _insert(self, workspace: Workspace) -> bool:
        try:
            with self._db.transaction() as session:
                if session.get(WorkspaceRow, workspace.id) is not None:
                    return False
                session.add(
                    WorkspaceRow(id=workspace.id, name=workspace.name, created_at=workspace.created_at)
                )
        except IntegrityError:
            return False
        return True

    def get_workspace(self, workspace_id: str) -> Workspace | None:
        with self._db.transaction() as session:
            row = session.get(WorkspaceRow, workspace_id)
            return None if row is None else _workspace(row)

    def list_workspaces(self) -> list[Workspace]:
        with self._db.transaction() as session:
            rows = session.scalars(select(WorkspaceRow)).all()
            return sorted((_workspace(row) for row in rows), key=workspace_order)

    def delete_workspace(self, workspace_id: str) -> bool:
        """One transaction removes the workspace's rows from every table; then each graph blob
        its versions used is collected unless another workspace's version still references it.

        Collection runs in transactions of its own, after the rows are gone: a blob that a
        concurrent append references again in the meantime is protected by the foreign key on
        PostgreSQL (that DELETE then fails and is ignored — an unreferenced blob is harmless) and
        is stored again by :meth:`SqlGraphRepository.append_version` when it vanished between that
        append's two transactions. The deletion itself is never undone by a failed collection.
        """
        check_workspace_key(workspace_id)
        hashes = select(GraphVersionRow.graph_hash).where(GraphVersionRow.workspace_id == workspace_id)
        with self._db.transaction() as session:
            connection = session.connection()
            used = set(session.scalars(hashes))
            removed = connection.execute(delete(WorkspaceRow).where(WorkspaceRow.id == workspace_id)).rowcount
            for table in WORKSPACE_TABLES:
                owned = table.c.workspace_id == workspace_id
                removed += connection.execute(delete(table).where(owned)).rowcount
        for graph_hash in sorted(used):
            self._collect_blob(graph_hash)
        return removed > 0

    def _collect_blob(self, graph_hash: str) -> None:
        referenced = select(GraphVersionRow.number).where(GraphVersionRow.graph_hash == graph_hash).exists()
        unused = (GraphBlobRow.graph_hash == graph_hash) & ~referenced
        try:
            with self._db.transaction() as session:
                session.connection().execute(delete(GraphBlobRow).where(unused))
        except IntegrityError:
            return


def _workspace(row: WorkspaceRow) -> Workspace:
    return Workspace(id=row.id, name=row.name, created_at=row.created_at)


class SqlDocumentRepository:
    def __init__(self, db: _Database) -> None:
        self._db = db

    def save_documents(self, workspace_id: str, docs: Sequence[Document], *, store_raw: bool) -> None:
        batch = check_documents(workspace_id, docs)
        self._db.upsert(
            [
                DocumentRow(
                    workspace_id=workspace_id,
                    doc_id=doc.doc_id,
                    title=doc.title,
                    sha256=doc.sha256,
                    media_type=doc.media_type,
                    path=doc.path,
                    quarantined=[list(span) for span in doc.quarantined],
                    text=doc.text if store_raw else None,
                )
                for doc in batch
            ]
        )

    def list_documents(self, workspace_id: str) -> list[StoredDocument]:
        # Select metadata columns only: the (possibly large) text is never loaded for listings.
        query = select(
            DocumentRow.doc_id,
            DocumentRow.title,
            DocumentRow.sha256,
            DocumentRow.media_type,
            DocumentRow.path,
            DocumentRow.quarantined,
            DocumentRow.text.is_not(None),
        ).where(DocumentRow.workspace_id == workspace_id)
        with self._db.transaction() as session:
            rows = session.execute(query).all()
        docs = [
            StoredDocument(
                workspace_id=workspace_id,
                doc_id=doc_id,
                title=title,
                sha256=sha256,
                media_type=media_type,
                path=path,
                quarantined=tuple((start, end) for start, end in quarantined),
                raw_stored=bool(raw_stored),
            )
            for doc_id, title, sha256, media_type, path, quarantined, raw_stored in rows
        ]
        return sorted(docs, key=document_order)

    def get_document(self, workspace_id: str, doc_id: str) -> Document | None:
        with self._db.transaction() as session:
            row = session.get(DocumentRow, (workspace_id, doc_id))
            if row is None or row.text is None:
                return None
            meta = StoredDocument(
                workspace_id=row.workspace_id,
                doc_id=row.doc_id,
                title=row.title,
                sha256=row.sha256,
                media_type=row.media_type,
                path=row.path,
                quarantined=tuple((start, end) for start, end in row.quarantined),
                raw_stored=True,
            )
            return restore_document(meta, row.text)


class SqlInputRepository:
    def __init__(self, db: _Database) -> None:
        self._db = db

    def save_inputs(self, workspace_id: str, inputs: StoredInputs) -> None:
        check_workspace_key(workspace_id)
        payload = json_payload(inputs, "입력 자료")
        self._db.upsert(
            [
                InputsRow(
                    workspace_id=workspace_id,
                    roster=payload["roster"],
                    oncall=payload["oncall"],
                    catalog=payload["catalog"],
                    policy=payload["policy"],
                )
            ]
        )

    def load_inputs(self, workspace_id: str) -> StoredInputs:
        with self._db.transaction() as session:
            row = session.get(InputsRow, workspace_id)
            if row is None:
                return StoredInputs()
            payload = {
                "roster": row.roster,
                "oncall": row.oncall,
                "catalog": row.catalog,
                "policy": row.policy,
            }
        return StoredInputs.model_validate(payload)


class SqlGraphRepository:
    def __init__(self, db: _Database) -> None:
        self._db = db

    def head(self, workspace_id: str) -> GraphVersion | None:
        query = (
            select(GraphVersionRow)
            .where(GraphVersionRow.workspace_id == workspace_id)
            .order_by(GraphVersionRow.number.desc())
            .limit(1)
        )
        with self._db.transaction() as session:
            row = session.scalars(query).first()
            return None if row is None else _version(row)

    def get_version(self, workspace_id: str, number: int) -> GraphVersion:
        with self._db.transaction() as session:
            row = session.get(GraphVersionRow, (workspace_id, number))
            if row is not None:
                return _version(row)
        raise NotFoundError(f"워크스페이스 {workspace_id}에 그래프 버전 {number}이(가) 없습니다.")

    def list_versions(self, workspace_id: str) -> list[GraphVersion]:
        query = (
            select(GraphVersionRow)
            .where(GraphVersionRow.workspace_id == workspace_id)
            .order_by(GraphVersionRow.number)
        )
        with self._db.transaction() as session:
            return [_version(row) for row in session.scalars(query)]

    def load_graph(self, graph_hash: str) -> OpsGraph:
        with self._db.transaction() as session:
            row = session.get(GraphBlobRow, graph_hash)
            blob = None if row is None else row.canonical_json
        if blob is None:
            raise NotFoundError(f"그래프 {graph_hash[:12]}…을(를) 찾을 수 없습니다.")
        return decode_graph(graph_hash, blob)

    def append_version(self, version: GraphVersion, graph: OpsGraph, *, expected_head: int | None) -> None:
        normalized, blob = prepare_append(version, graph, expected_head)
        self._put_blob(normalized.graph_hash, blob)
        workspace_id = normalized.workspace_id
        try:
            with self._db.transaction() as session:
                current = self._head_number(session, workspace_id)
                if current != expected_head:
                    raise version_conflict(workspace_id, expected_head, current)
                if session.get(GraphBlobRow, normalized.graph_hash) is None:
                    # A concurrent workspace deletion collected the blob after _put_blob stored
                    # it (it was unreferenced then): store it again together with its version.
                    session.add(GraphBlobRow(graph_hash=normalized.graph_hash, canonical_json=blob))
                    session.flush()
                session.add(_version_row(normalized))
        except IntegrityError as exc:
            # Lost the race to a concurrent append of the same version number.
            with self._db.transaction() as session:
                current = self._head_number(session, workspace_id)
            raise version_conflict(workspace_id, expected_head, current) from exc

    def _put_blob(self, graph_hash: str, blob: str) -> None:
        try:
            with self._db.transaction() as session:
                if session.get(GraphBlobRow, graph_hash) is None:
                    session.add(GraphBlobRow(graph_hash=graph_hash, canonical_json=blob))
        except IntegrityError:
            # A concurrent writer stored the same content-addressed blob first: identical bytes.
            return

    @staticmethod
    def _head_number(session: Session, workspace_id: str) -> int | None:
        return session.scalar(
            select(func.max(GraphVersionRow.number)).where(GraphVersionRow.workspace_id == workspace_id)
        )


def _version(row: GraphVersionRow) -> GraphVersion:
    return GraphVersion(
        workspace_id=row.workspace_id,
        number=row.number,
        parent=row.parent,
        graph_hash=row.graph_hash,
        patch_id=row.patch_id,
        revert_of=row.revert_of,
        rehearsal_ids=tuple(row.rehearsal_ids),
        created_by=row.created_by,
        created_at=row.created_at,
        note=row.note,
    )


def _version_row(version: GraphVersion) -> GraphVersionRow:
    return GraphVersionRow(
        workspace_id=version.workspace_id,
        number=version.number,
        parent=version.parent,
        graph_hash=version.graph_hash,
        patch_id=version.patch_id,
        revert_of=version.revert_of,
        rehearsal_ids=list(version.rehearsal_ids),
        created_by=version.created_by,
        created_at=version.created_at,
        note=version.note,
    )


class SqlPatchRepository:
    def __init__(self, db: _Database) -> None:
        self._db = db

    def save_patch(self, patch: Patch) -> None:
        check_patch(patch)
        values: dict[str, Any] = {
            "id": patch.id,
            "workspace_id": patch.workspace_id,
            "status": patch.status.value,
            "payload": json_payload(patch, "수정안"),
        }
        self._db.save_owned(PatchRow, PatchRow.id, values, "수정안")

    def get_patch(self, patch_id: str) -> Patch:
        with self._db.transaction() as session:
            row = session.get(PatchRow, patch_id)
            payload = None if row is None else row.payload
        if payload is None:
            raise NotFoundError(f"수정안 {patch_id}을(를) 찾을 수 없습니다.")
        return Patch.model_validate(payload)

    def list_patches(self, workspace_id: str) -> list[Patch]:
        query = select(PatchRow.payload).where(PatchRow.workspace_id == workspace_id)
        with self._db.transaction() as session:
            payloads = session.scalars(query).all()
        return sorted((Patch.model_validate(p) for p in payloads), key=patch_order)


class SqlRehearsalRepository:
    def __init__(self, db: _Database) -> None:
        self._db = db

    def save_scenario(self, workspace_id: str, scenario: Scenario) -> None:
        check_scenario(workspace_id, scenario)
        self._db.upsert(
            [
                ScenarioRow(
                    workspace_id=workspace_id,
                    scenario_id=scenario.id,
                    payload=json_payload(scenario, "시나리오"),
                )
            ]
        )

    def list_scenarios(self, workspace_id: str) -> list[Scenario]:
        query = select(ScenarioRow.payload).where(ScenarioRow.workspace_id == workspace_id)
        with self._db.transaction() as session:
            payloads = session.scalars(query).all()
        return sorted((Scenario.model_validate(p) for p in payloads), key=scenario_order)

    def save_result(self, workspace_id: str, result: RehearsalResult) -> str:
        check_result(workspace_id, result)
        payload = json_payload(result, "리허설 결과")
        rid = rehearsal_id(workspace_id, result)
        self._db.upsert(
            [
                RehearsalResultRow(
                    id=rid, workspace_id=workspace_id, graph_hash=result.graph_hash, payload=payload
                )
            ]
        )
        return rid

    def list_results(self, workspace_id: str, graph_hash: str | None = None) -> list[RehearsalResult]:
        query = select(RehearsalResultRow.id, RehearsalResultRow.payload).where(
            RehearsalResultRow.workspace_id == workspace_id
        )
        if graph_hash is not None:
            query = query.where(RehearsalResultRow.graph_hash == graph_hash)
        with self._db.transaction() as session:
            rows = session.execute(query).all()
        decoded = [(rid, RehearsalResult.model_validate(payload)) for rid, payload in rows]
        return [result for _, result in sorted(decoded, key=result_order)]


class SqlReportRepository:
    def __init__(self, db: _Database) -> None:
        self._db = db

    def save_report(self, report: DiagnosisReport) -> None:
        values: dict[str, Any] = {
            "run_id": report.run_id,
            "workspace_id": report.workspace_id,
            "created_at": check_report(report),
            "payload": json_payload(report, "리포트"),
        }
        self._db.save_owned(ReportRow, ReportRow.run_id, values, "리포트")

    def latest_report(self, workspace_id: str) -> DiagnosisReport | None:
        # Pick the winner in Python (shared tie-break on run_id) so every backend agrees.
        keys = select(ReportRow.created_at, ReportRow.run_id).where(ReportRow.workspace_id == workspace_id)
        with self._db.transaction() as session:
            candidates = [report_order(created_at, run_id) for created_at, run_id in session.execute(keys)]
            if not candidates:
                return None
            _, run_id = max(candidates)
            row = session.get_one(ReportRow, run_id)
            payload = row.payload
        return DiagnosisReport.model_validate(payload)


class SqlAuditRepository:
    def __init__(self, db: _Database, clock: Clock) -> None:
        self._db = db
        self._clock = clock

    def record(self, workspace_id: str, actor: str, action: str, payload: dict[str, object]) -> None:
        validate_audit(workspace_id, actor, action)
        body = to_json_object(payload, "감사 로그 내용")
        with self._db.transaction() as session:
            # Read the clock inside the transaction: on engines serialised by the store's lock
            # (in-memory SQLite) the id order and the timestamps then always agree.
            session.add(
                AuditRow(
                    workspace_id=workspace_id,
                    actor=actor,
                    action=action,
                    payload=body,
                    recorded_at=as_utc(self._clock(), "clock()의 반환값"),
                )
            )

    def entries(self, workspace_id: str) -> list[AuditEntry]:
        query = select(AuditRow).where(AuditRow.workspace_id == workspace_id).order_by(AuditRow.id)
        with self._db.transaction() as session:
            return [
                AuditEntry(
                    seq=row.id,
                    workspace_id=row.workspace_id,
                    actor=row.actor,
                    action=row.action,
                    payload=row.payload,
                    recorded_at=row.recorded_at,
                )
                for row in session.scalars(query)
            ]


class SqlSignupRepository:
    def __init__(self, db: _Database) -> None:
        self._db = db

    def add_signup(self, signup: Signup) -> None:
        self._db.upsert([SignupRow(**signup.model_dump())])

    def add_signup_if_absent(self, signup: Signup) -> bool:
        """A plain INSERT behind an existence check; the primary key settles a race.

        Losing the race to a concurrent insert of the same address surfaces as an IntegrityError,
        which means "already stored" only if the row is there now — any other integrity failure
        is re-raised rather than reported as a duplicate.
        """
        try:
            with self._db.transaction() as session:
                if session.get(SignupRow, signup.email) is not None:
                    return False
                session.add(SignupRow(**signup.model_dump()))
        except IntegrityError:
            with self._db.transaction() as session:
                if session.get(SignupRow, signup.email) is None:
                    raise
            return False
        return True

    def delete_signup(self, email: str) -> bool:
        key = signup_key(email)
        if key is None:
            return False
        with self._db.transaction() as session:
            deleted = session.connection().execute(delete(SignupRow).where(SignupRow.email == key))
            return deleted.rowcount > 0

    def delete_signups_before(self, cutoff: datetime) -> int:
        expired = SignupRow.created_at < deletion_cutoff(cutoff)
        with self._db.transaction() as session:
            return session.connection().execute(delete(SignupRow).where(expired)).rowcount

    def get_signup(self, email: str) -> Signup | None:
        key = signup_key(email)
        if key is None:
            return None
        with self._db.transaction() as session:
            row = session.get(SignupRow, key)
            return None if row is None else _signup(row)

    def count_signups(self) -> int:
        with self._db.transaction() as session:
            return session.scalar(select(func.count()).select_from(SignupRow)) or 0

    def list_signups(self) -> list[Signup]:
        # Sorted in Python (shared key): ORDER BY over e-mail text would follow the DB's collation.
        with self._db.transaction() as session:
            signups = [_signup(row) for row in session.scalars(select(SignupRow))]
        return sorted(signups, key=signup_order)


def _signup(row: SignupRow) -> Signup:
    return Signup(
        email=row.email,
        organization=row.organization,
        team_size=row.team_size,
        message=row.message,
        consent_privacy=row.consent_privacy,
        consent_marketing=row.consent_marketing,
        created_at=row.created_at,
        source=row.source,
    )


class SqlEventRepository:
    def __init__(self, db: _Database) -> None:
        self._db = db

    def record_event(self, name: str, props: Mapping[str, object], ts: datetime) -> None:
        row = EventRow(
            name=validate_event_name(name),
            props=to_json_object(props, "이벤트 속성"),
            occurred_at=as_utc(ts, "이벤트 시각"),
        )
        with self._db.transaction() as session:
            session.add(row)

    def event_counts(self) -> dict[str, int]:
        query = select(EventRow.name, func.count()).group_by(EventRow.name)
        with self._db.transaction() as session:
            counts = {name: int(count) for name, count in session.execute(query)}
        return dict(sorted(counts.items()))


class SqlWorkspaceAccessRepository:
    def __init__(self, db: _Database, clock: Clock) -> None:
        self._db = db
        self._clock = clock

    def set_access_key_digest(self, workspace_id: str, digest: str) -> None:
        row = WorkspaceAccessKeyRow(
            workspace_id=workspace_id,
            key_digest=check_key_digest(workspace_id, digest),
            created_at=as_utc(self._clock(), "clock()의 반환값"),
        )
        self._db.upsert([row])

    def access_key_digest(self, workspace_id: str) -> str | None:
        with self._db.transaction() as session:
            row = session.get(WorkspaceAccessKeyRow, workspace_id)
            return None if row is None else row.key_digest


# --------------------------------------------------------------------------- aggregate


class SqlStore:
    """Relational store implementing every repository protocol.

    ``url`` is a SQLAlchemy URL (``sqlite:///./data/jettae.db``,
    ``postgresql+psycopg://…``) or a ready :class:`~sqlalchemy.Engine` whose lifecycle then
    stays with the caller (``close`` disposes only engines the store created itself).
    """

    def __init__(self, url: str | Engine, *, create_schema: bool = True, clock: Clock = utc_now) -> None:
        self._owns_engine = not isinstance(url, Engine)
        self.engine = url if isinstance(url, Engine) else build_engine(url)
        if create_schema:
            Base.metadata.create_all(self.engine)
        db = _Database(self.engine)
        self.workspaces = SqlWorkspaceRepository(db, clock)
        self.documents = SqlDocumentRepository(db)
        self.inputs = SqlInputRepository(db)
        self.graphs = SqlGraphRepository(db)
        self.patches = SqlPatchRepository(db)
        self.rehearsals = SqlRehearsalRepository(db)
        self.reports = SqlReportRepository(db)
        self.audit = SqlAuditRepository(db, clock)
        self.signups = SqlSignupRepository(db)
        self.events = SqlEventRepository(db)
        self.access = SqlWorkspaceAccessRepository(db, clock)

    def close(self) -> None:
        if self._owns_engine:
            self.engine.dispose()


__all__ = [
    "MIGRATIONS_DIR",
    "POSTGRES_DRIVER",
    "WORKSPACE_TABLES",
    "SqlStore",
    "alembic_config",
    "build_engine",
    "database_error_summary",
    "is_memory_sqlite",
    "normalize_database_url",
    "upgrade_schema",
]
