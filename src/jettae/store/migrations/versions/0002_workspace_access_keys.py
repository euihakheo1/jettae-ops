"""Access keys of private workspaces (public beta: anonymous free scans).

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-14 00:00:00+00:00

Only the SHA-256 digest of a key is stored. Like every other per-workspace table, the workspace id
is a partition key without a foreign key to ``workspaces`` (see ``jettae.store.models``).
Self-contained like ``0001``: plain SQLAlchemy types, no import of ``jettae.store.models``.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "workspace_access_keys",
        sa.Column("workspace_id", sa.String(), nullable=False),
        sa.Column("key_digest", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("workspace_id", name="pk_workspace_access_keys"),
    )


def downgrade() -> None:
    op.drop_table("workspace_access_keys")
