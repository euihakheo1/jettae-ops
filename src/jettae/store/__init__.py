"""Persistence for JETTAE Ops: one :class:`Store` contract, two interchangeable backends.

* :class:`MemoryStore` — tests, demos, one-shot CLI scans (``memory://``).
* :class:`SqlStore` — SQLite for development, Postgres for production (SQLAlchemy 2.0 +
  Alembic migrations under ``jettae/store/migrations``).

Both pass the same contract suite (``tests/test_store_contract.py``), so the orchestrator can
be developed against the fast backend and deployed against the durable one.
"""

from jettae.store.base import (
    AlreadyExistsError,
    AuditEntry,
    AuditRepository,
    Clock,
    DocumentRepository,
    EventRepository,
    InputRepository,
    NotFoundError,
    Signup,
    SignupRepository,
    Store,
    StoredDocument,
    StoredInputs,
    StoreIntegrityError,
    VersionConflictError,
    Workspace,
    WorkspaceAccessRepository,
    WorkspaceRepository,
    rehearsal_id,
    utc_now,
)
from jettae.store.memory import MemoryStore
from jettae.store.sql import SqlStore, alembic_config, database_error_summary, upgrade_schema

#: URL that :func:`open_store` maps to the in-memory backend.
MEMORY_URL = "memory://"


def open_store(url: str, *, clock: Clock = utc_now, create_schema: bool = True) -> Store:
    """Open the store named by ``url`` (``memory://`` or any SQLAlchemy URL).

    This is how settings (``JETTAE_DATABASE_URL``) become a store, so the API, CLI and tests
    never branch on the backend themselves.
    """
    if url == MEMORY_URL:
        return MemoryStore(clock=clock)
    return SqlStore(url, create_schema=create_schema, clock=clock)


__all__ = [
    "MEMORY_URL",
    "AlreadyExistsError",
    "AuditEntry",
    "AuditRepository",
    "Clock",
    "DocumentRepository",
    "EventRepository",
    "InputRepository",
    "MemoryStore",
    "NotFoundError",
    "Signup",
    "SignupRepository",
    "SqlStore",
    "Store",
    "StoreIntegrityError",
    "StoredDocument",
    "StoredInputs",
    "VersionConflictError",
    "Workspace",
    "WorkspaceAccessRepository",
    "WorkspaceRepository",
    "alembic_config",
    "database_error_summary",
    "open_store",
    "rehearsal_id",
    "upgrade_schema",
    "utc_now",
]
