"""SQLAlchemy 2.0 typed ORM schema of the SQL store (SQLite for dev, Postgres for production).

Design choices, and why:

* **Portable types only** (``String``/``Text``/``Integer``/``Boolean``/``JSON``/aware
  ``DateTime``) — no dialect-specific DDL, so ``create_all`` and the Alembic migration produce
  the same schema on SQLite and Postgres.
* **Domain records as JSON payloads** (patches, scenarios, rehearsal results, reports): the
  pydantic models are the contract and evolve with the domain; the columns next to the payload
  exist only to filter (workspace, status, graph hash) and to order by time.
* **Graph blobs as canonical JSON text**, keyed by their SHA-256: the exact bytes that were
  hashed are what is stored, so the content address can be re-verified on every load.
* **Timestamps are UTC**. :class:`UtcDateTime` refuses naive values on the way in and always
  returns aware UTC values, also on SQLite which has no time-zone support of its own.
* **No foreign keys to ``workspaces``**: the domain ports are keyed by plain workspace-id
  strings and are also used for ephemeral CLI scans, so workspace ids act as partition keys.
  Existence is checked by the service layer. For the same reason deletion does not cascade in
  the database: ``delete_workspace`` empties every table that has a ``workspace_id`` column
  (:data:`jettae.store.sql.WORKSPACE_TABLES`), so a new table needs only that column to be
  covered.
* A **naming convention** gives every constraint/index a stable name, which keeps Alembic
  migrations reproducible across databases.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from alembic.autogenerate.api import AutogenContext
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Dialect,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator

NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

#: SHA-256 hex digests are always 64 characters.
HASH_LENGTH = 64


class UtcDateTime(TypeDecorator[datetime]):
    """Timezone-aware ``DateTime`` that always stores and returns UTC."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("naive datetime cannot be stored; pass a timezone-aware value")
        utc = value.astimezone(UTC)
        # SQLite stores the wall-clock text only; strip the (UTC) zone so nothing is lost.
        return utc.replace(tzinfo=None) if dialect.name == "sqlite" else utc

    def process_result_value(self, value: Any | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if not isinstance(value, datetime):
            raise TypeError(f"expected datetime from the database, got {type(value).__name__}")
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def render_item(type_: str, obj: object, autogen_context: AutogenContext) -> str | Literal[False]:
    """Alembic autogenerate hook: render :class:`UtcDateTime` as a plain SQLAlchemy type.

    Without it ``alembic revision --autogenerate`` writes ``jettae.store.models.UtcDateTime()``
    into new revisions, coupling migrations to today's models, which the migration template
    promises never to do. UTC normalisation is a Python-side concern; the database column is
    just ``DateTime(timezone=True)``.
    """
    if type_ == "type" and isinstance(obj, UtcDateTime):
        prefix = autogen_context.opts.get("sqlalchemy_module_prefix") or "sa."
        return f"{prefix}DateTime(timezone=True)"
    return False


def _autoincrement_id() -> BigInteger:
    """64-bit autoincrement id that still maps to SQLite's ``INTEGER PRIMARY KEY`` (rowid)."""
    return BigInteger().with_variant(Integer(), "sqlite")


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class WorkspaceRow(Base):
    __tablename__ = "workspaces"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime())


class DocumentRow(Base):
    __tablename__ = "documents"

    workspace_id: Mapped[str] = mapped_column(String, primary_key=True)
    doc_id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(String(HASH_LENGTH))
    media_type: Mapped[str] = mapped_column(String)
    path: Mapped[str | None] = mapped_column(Text)
    quarantined: Mapped[list[list[int]]] = mapped_column(JSON)
    #: NULL unless raw storage was enabled for the save (privacy default: not stored)
    text: Mapped[str | None] = mapped_column(Text)


class InputsRow(Base):
    __tablename__ = "inputs"

    workspace_id: Mapped[str] = mapped_column(String, primary_key=True)
    roster: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    oncall: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    catalog: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    policy: Mapped[dict[str, Any]] = mapped_column(JSON)


class GraphBlobRow(Base):
    __tablename__ = "graph_blobs"

    graph_hash: Mapped[str] = mapped_column(String(HASH_LENGTH), primary_key=True)
    canonical_json: Mapped[str] = mapped_column(Text)


class GraphVersionRow(Base):
    """Append-only history; the composite primary key is the optimistic-concurrency guard."""

    __tablename__ = "graph_versions"

    workspace_id: Mapped[str] = mapped_column(String, primary_key=True)
    number: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    parent: Mapped[int | None] = mapped_column(Integer)
    graph_hash: Mapped[str] = mapped_column(
        String(HASH_LENGTH), ForeignKey("graph_blobs.graph_hash"), index=True
    )
    patch_id: Mapped[str | None] = mapped_column(String)
    revert_of: Mapped[int | None] = mapped_column(Integer)
    rehearsal_ids: Mapped[list[str]] = mapped_column(JSON)
    created_by: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime())
    note: Mapped[str] = mapped_column(Text)


class PatchRow(Base):
    __tablename__ = "patches"
    __table_args__ = (Index("ix_patches_workspace_id_status", "workspace_id", "status"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class ScenarioRow(Base):
    __tablename__ = "scenarios"

    workspace_id: Mapped[str] = mapped_column(String, primary_key=True)
    scenario_id: Mapped[str] = mapped_column(String, primary_key=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class RehearsalResultRow(Base):
    __tablename__ = "rehearsal_results"
    __table_args__ = (Index("ix_rehearsal_results_workspace_id_graph_hash", "workspace_id", "graph_hash"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String)
    graph_hash: Mapped[str] = mapped_column(String)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class ReportRow(Base):
    __tablename__ = "reports"
    __table_args__ = (Index("ix_reports_workspace_id_created_at", "workspace_id", "created_at"),)

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime())
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class AuditRow(Base):
    """Append-only: the store never updates rows, and deletes them only together with their whole
    workspace (privacy deletion and retention); ``id`` is the append order."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(_autoincrement_id(), primary_key=True, autoincrement=True)
    workspace_id: Mapped[str] = mapped_column(String, index=True)
    actor: Mapped[str] = mapped_column(Text)
    action: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    recorded_at: Mapped[datetime] = mapped_column(UtcDateTime())


class SignupRow(Base):
    __tablename__ = "signups"

    email: Mapped[str] = mapped_column(String(254), primary_key=True)
    organization: Mapped[str | None] = mapped_column(Text)
    team_size: Mapped[str | None] = mapped_column(Text)
    message: Mapped[str | None] = mapped_column(Text)
    consent_privacy: Mapped[bool] = mapped_column(Boolean)
    consent_marketing: Mapped[bool] = mapped_column(Boolean)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime())
    source: Mapped[str | None] = mapped_column(Text)


class EventRow(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(_autoincrement_id(), primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), index=True)
    props: Mapped[dict[str, Any]] = mapped_column(JSON)
    occurred_at: Mapped[datetime] = mapped_column(UtcDateTime())


class WorkspaceAccessKeyRow(Base):
    """SHA-256 digest of a private workspace's access key; the plaintext key is never stored."""

    __tablename__ = "workspace_access_keys"

    workspace_id: Mapped[str] = mapped_column(String, primary_key=True)
    key_digest: Mapped[str] = mapped_column(String(HASH_LENGTH))
    #: when the current key was granted
    created_at: Mapped[datetime] = mapped_column(UtcDateTime())


__all__ = [
    "HASH_LENGTH",
    "NAMING_CONVENTION",
    "AuditRow",
    "Base",
    "DocumentRow",
    "EventRow",
    "GraphBlobRow",
    "GraphVersionRow",
    "InputsRow",
    "PatchRow",
    "RehearsalResultRow",
    "ReportRow",
    "ScenarioRow",
    "SignupRow",
    "UtcDateTime",
    "WorkspaceAccessKeyRow",
    "WorkspaceRow",
    "render_item",
]
