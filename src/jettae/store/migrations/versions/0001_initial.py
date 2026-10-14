"""Initial schema of the JETTAE Ops SQL store.

Revision ID: 0001
Revises:
Create Date: 2026-10-13 00:00:00+00:00

Migrations are self-contained: they use plain SQLAlchemy types and never import
``jettae.store.models``, so a revision keeps meaning the same thing after the models evolve.
``tests/test_store_migrations.py`` asserts that this revision yields exactly the schema of
``jettae.store.models`` (Alembic autogenerate finds no difference).
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _utc() -> sa.DateTime:
    return sa.DateTime(timezone=True)


def _autoincrement_id() -> sa.BigInteger:
    # BIGINT/BIGSERIAL on Postgres, INTEGER PRIMARY KEY (rowid) on SQLite
    return sa.BigInteger().with_variant(sa.Integer(), "sqlite")


def upgrade() -> None:
    op.create_table(
        "workspaces",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("created_at", _utc(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_workspaces"),
    )
    op.create_table(
        "documents",
        sa.Column("workspace_id", sa.String(), nullable=False),
        sa.Column("doc_id", sa.String(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("media_type", sa.String(), nullable=False),
        sa.Column("path", sa.Text(), nullable=True),
        sa.Column("quarantined", sa.JSON(), nullable=False),
        sa.Column("text", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("workspace_id", "doc_id", name="pk_documents"),
    )
    op.create_table(
        "inputs",
        sa.Column("workspace_id", sa.String(), nullable=False),
        sa.Column("roster", sa.JSON(), nullable=False),
        sa.Column("oncall", sa.JSON(), nullable=False),
        sa.Column("catalog", sa.JSON(), nullable=False),
        sa.Column("policy", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("workspace_id", name="pk_inputs"),
    )
    op.create_table(
        "graph_blobs",
        sa.Column("graph_hash", sa.String(length=64), nullable=False),
        sa.Column("canonical_json", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("graph_hash", name="pk_graph_blobs"),
    )
    op.create_table(
        "graph_versions",
        sa.Column("workspace_id", sa.String(), nullable=False),
        sa.Column("number", sa.Integer(), autoincrement=False, nullable=False),
        sa.Column("parent", sa.Integer(), nullable=True),
        sa.Column("graph_hash", sa.String(length=64), nullable=False),
        sa.Column("patch_id", sa.String(), nullable=True),
        sa.Column("revert_of", sa.Integer(), nullable=True),
        sa.Column("rehearsal_ids", sa.JSON(), nullable=False),
        sa.Column("created_by", sa.Text(), nullable=False),
        sa.Column("created_at", _utc(), nullable=False),
        sa.Column("note", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["graph_hash"],
            ["graph_blobs.graph_hash"],
            name="fk_graph_versions_graph_hash_graph_blobs",
        ),
        sa.PrimaryKeyConstraint("workspace_id", "number", name="pk_graph_versions"),
    )
    op.create_index("ix_graph_versions_graph_hash", "graph_versions", ["graph_hash"])
    op.create_table(
        "patches",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("workspace_id", sa.String(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_patches"),
    )
    op.create_index("ix_patches_workspace_id_status", "patches", ["workspace_id", "status"])
    op.create_table(
        "scenarios",
        sa.Column("workspace_id", sa.String(), nullable=False),
        sa.Column("scenario_id", sa.String(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("workspace_id", "scenario_id", name="pk_scenarios"),
    )
    op.create_table(
        "rehearsal_results",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("workspace_id", sa.String(), nullable=False),
        sa.Column("graph_hash", sa.String(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_rehearsal_results"),
    )
    op.create_index(
        "ix_rehearsal_results_workspace_id_graph_hash", "rehearsal_results", ["workspace_id", "graph_hash"]
    )
    op.create_table(
        "reports",
        sa.Column("run_id", sa.String(), nullable=False),
        sa.Column("workspace_id", sa.String(), nullable=False),
        sa.Column("created_at", _utc(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("run_id", name="pk_reports"),
    )
    op.create_index("ix_reports_workspace_id_created_at", "reports", ["workspace_id", "created_at"])
    op.create_table(
        "audit_log",
        sa.Column("id", _autoincrement_id(), autoincrement=True, nullable=False),
        sa.Column("workspace_id", sa.String(), nullable=False),
        sa.Column("actor", sa.Text(), nullable=False),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("recorded_at", _utc(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_audit_log"),
    )
    op.create_index("ix_audit_log_workspace_id", "audit_log", ["workspace_id"])
    op.create_table(
        "signups",
        sa.Column("email", sa.String(length=254), nullable=False),
        sa.Column("organization", sa.Text(), nullable=True),
        sa.Column("team_size", sa.Text(), nullable=True),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("consent_privacy", sa.Boolean(), nullable=False),
        sa.Column("consent_marketing", sa.Boolean(), nullable=False),
        sa.Column("created_at", _utc(), nullable=False),
        sa.Column("source", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("email", name="pk_signups"),
    )
    op.create_table(
        "events",
        sa.Column("id", _autoincrement_id(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("props", sa.JSON(), nullable=False),
        sa.Column("occurred_at", _utc(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_events"),
    )
    op.create_index("ix_events_name", "events", ["name"])


def downgrade() -> None:
    op.drop_index("ix_events_name", table_name="events")
    op.drop_table("events")
    op.drop_table("signups")
    op.drop_index("ix_audit_log_workspace_id", table_name="audit_log")
    op.drop_table("audit_log")
    op.drop_index("ix_reports_workspace_id_created_at", table_name="reports")
    op.drop_table("reports")
    op.drop_index("ix_rehearsal_results_workspace_id_graph_hash", table_name="rehearsal_results")
    op.drop_table("rehearsal_results")
    op.drop_table("scenarios")
    op.drop_index("ix_patches_workspace_id_status", table_name="patches")
    op.drop_table("patches")
    op.drop_index("ix_graph_versions_graph_hash", table_name="graph_versions")
    op.drop_table("graph_versions")
    op.drop_table("graph_blobs")
    op.drop_table("inputs")
    op.drop_table("documents")
    op.drop_table("workspaces")
