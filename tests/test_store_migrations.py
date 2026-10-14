"""Alembic migrations must build exactly the schema of ``jettae.store.models``.

``create_all`` (dev/tests) and ``alembic upgrade head`` (deployments) are two roads to the same
schema; these tests fail as soon as a model change is not accompanied by a migration.

The migrations directory is located from this checkout rather than from the installed
package: on Windows the package may live under a path too long for ``MAX_PATH`` while the
checkout is reached through a short (``subst``) drive. The packaged default is exercised by
``test_upgrade_schema_with_packaged_migrations`` where the platform allows it.
"""

from __future__ import annotations

import io
import os
import shutil
from argparse import Namespace
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect, text

from jettae.domain import Attr, Extractor, GraphBuilder, GraphVersion, Node, NodeKind, Provenance, SourceKind
from jettae.store import SqlStore, alembic_config, upgrade_schema
from jettae.store.models import Base
from jettae.store.sql import MIGRATIONS_DIR

REPO_ROOT = Path(__file__).parent.parent
CHECKOUT_MIGRATIONS = REPO_ROOT / "src" / "jettae" / "store" / "migrations"
MODEL_TABLES = set(Base.metadata.tables)
T0 = datetime(2026, 10, 13, 9, 0, tzinfo=UTC)


def _config(url: str | None) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(CHECKOUT_MIGRATIONS))
    if url is not None:
        cfg.set_main_option("sqlalchemy.url", url)
    return cfg


def _sqlite_url(path: Path) -> str:
    return f"sqlite:///{path.as_posix()}"


def _tables(url: str) -> set[str]:
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            return set(inspect(conn).get_table_names())
    finally:
        engine.dispose()


def test_single_linear_head() -> None:
    script = ScriptDirectory.from_config(_config(None))
    assert script.get_heads() == ["0002"]
    assert [rev.revision for rev in script.walk_revisions()] == ["0002", "0001"]


def test_upgrade_head_matches_models(tmp_path: Path) -> None:
    url = _sqlite_url(tmp_path / "migrated.db")
    command.upgrade(_config(url), "head")

    assert _tables(url) == MODEL_TABLES | {"alembic_version"}
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            context = MigrationContext.configure(conn, opts={"compare_type": True})
            assert compare_metadata(context, Base.metadata) == []
            assert context.get_current_revision() == "0002"
    finally:
        engine.dispose()


def test_downgrade_to_base_removes_every_table(tmp_path: Path) -> None:
    url = _sqlite_url(tmp_path / "roundtrip.db")
    cfg = _config(url)
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")
    assert _tables(url) == {"alembic_version"}
    command.upgrade(cfg, "head")
    assert _tables(url) == MODEL_TABLES | {"alembic_version"}


def test_migrated_database_serves_the_store(tmp_path: Path) -> None:
    url = _sqlite_url(tmp_path / "served.db")
    command.upgrade(_config(url), "head")
    builder = GraphBuilder()
    builder.add_node(
        Node(
            id="service:pay",
            kind=NodeKind.SERVICE,
            name="pay",
            attrs={"tier": Attr(value=1, source=SourceKind.CATALOG)},
            prov=(Provenance(source=SourceKind.CATALOG, extractor=Extractor.IMPORT),),
        )
    )
    built = builder.build()
    version = GraphVersion(
        workspace_id="team",
        number=1,
        parent=None,
        graph_hash=built.content_hash(),
        created_by="lead",
        created_at=T0,
    )
    with closing(SqlStore(url, create_schema=False, clock=lambda: T0)) as store:
        store.workspaces.create_workspace("팀", workspace_id="team")
        store.graphs.append_version(version, built, expected_head=None)
        store.audit.record("team", "lead", "version.init", {"number": 1})
        assert store.graphs.head("team") == version
        assert [e.seq for e in store.audit.entries("team")] == [1]


def test_access_keys_migration_upgrades_an_existing_0001_database(tmp_path: Path) -> None:
    """A database deployed at 0001 keeps its data and gains ``workspace_access_keys`` (and loses
    only that table again on downgrade)."""
    url = _sqlite_url(tmp_path / "existing.db")
    cfg = _config(url)
    command.upgrade(cfg, "0001")
    assert "workspace_access_keys" not in _tables(url)
    with closing(SqlStore(url, create_schema=False, clock=lambda: T0)) as store:
        store.workspaces.create_workspace("팀", workspace_id="team")

    command.upgrade(cfg, "head")
    assert _tables(url) == MODEL_TABLES | {"alembic_version"}
    digest = "a" * 64
    with closing(SqlStore(url, create_schema=False, clock=lambda: T0)) as store:
        assert store.workspaces.get_workspace("team") is not None
        store.access.set_access_key_digest("team", digest)
        assert store.access.access_key_digest("team") == digest

    command.downgrade(cfg, "0001")
    assert _tables(url) == (MODEL_TABLES - {"workspace_access_keys"}) | {"alembic_version"}


def test_create_all_after_upgrade_is_a_no_op(tmp_path: Path) -> None:
    """The container runs ``jettae db upgrade`` and then opens the store with ``create_all``: on a
    migrated database that must neither fail nor change the schema or the recorded revision."""
    url = _sqlite_url(tmp_path / "both.db")
    command.upgrade(_config(url), "head")
    with closing(SqlStore(url, create_schema=True, clock=lambda: T0)) as store:
        store.workspaces.create_workspace("팀", workspace_id="team")
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            context = MigrationContext.configure(conn, opts={"compare_type": True})
            assert context.get_current_revision() == "0002"
            assert compare_metadata(context, Base.metadata) == []
    finally:
        engine.dispose()


def test_alembic_ini_is_ascii() -> None:
    # configparser reads with the locale encoding (cp949 on Korean Windows): keep it ASCII.
    assert (REPO_ROOT / "alembic.ini").read_bytes().isascii()


def test_root_alembic_ini_with_x_url(tmp_path: Path) -> None:
    url = _sqlite_url(tmp_path / "cli.db")
    cfg = Config(str(REPO_ROOT / "alembic.ini"), cmd_opts=Namespace(x=[f"url={url}"]))
    cfg.attributes["configure_logger"] = False
    command.upgrade(cfg, "head")
    assert _tables(url) == MODEL_TABLES | {"alembic_version"}


def test_env_falls_back_to_jettae_database_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    url = _sqlite_url(tmp_path / "from-env" / "jettae.db")
    monkeypatch.setenv("JETTAE_DATABASE_URL", url)
    command.upgrade(_config(None), "head")
    assert _tables(url) == MODEL_TABLES | {"alembic_version"}


def test_offline_sql_for_postgres_is_portable() -> None:
    buffer = io.StringIO()
    cfg = _config("postgresql://jettae@db.invalid/jettae")
    cfg.output_buffer = buffer
    command.upgrade(cfg, "head", sql=True)
    ddl = buffer.getvalue()
    for table in sorted(MODEL_TABLES):
        assert f"CREATE TABLE {table} (" in ddl
    assert "TIMESTAMP WITH TIME ZONE" in ddl
    assert "BIGSERIAL" in ddl
    assert "JSON" in ddl
    assert "AUTOINCREMENT" not in ddl


def test_alembic_config_points_at_packaged_migrations() -> None:
    url = "postgresql+psycopg://jettae:p%40ss@db.invalid/jettae"
    cfg = alembic_config(url)
    assert cfg.get_main_option("sqlalchemy.url") == url
    assert cfg.get_main_option("script_location") == str(MIGRATIONS_DIR)
    assert (MIGRATIONS_DIR / "env.py").name == "env.py"
    assert alembic_config().get_main_option("sqlalchemy.url") is None


def test_autogenerate_never_imports_application_models(tmp_path: Path) -> None:
    """A new revision must render ``UtcDateTime`` as a plain ``sa`` type.

    Without the ``render_item`` hook in ``env.py``, autogenerate wrote
    ``jettae.store.models.UtcDateTime()`` (and an ``import jettae`` line) into the revision,
    tying a migration to whatever the models look like when it is later replayed.
    """
    scripts = tmp_path / "migrations"
    shutil.copytree(CHECKOUT_MIGRATIONS, scripts, ignore=shutil.ignore_patterns("__pycache__"))
    url = _sqlite_url(tmp_path / "autogen.db")
    cfg = Config()
    cfg.set_main_option("script_location", str(scripts))
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")
    engine = create_engine(url)
    try:
        with engine.begin() as conn:
            conn.execute(text("DROP INDEX ix_events_name"))
            conn.execute(text("DROP TABLE events"))
    finally:
        engine.dispose()

    command.revision(cfg, message="re-add events", autogenerate=True, rev_id="0003")

    [generated] = (scripts / "versions").glob("0003_*.py")
    code = generated.read_text(encoding="utf-8")
    assert "create_table('events'" in code
    assert "sa.Column('occurred_at', sa.DateTime(timezone=True), nullable=False)" in code
    # The template's docstring names the models module; no executable line may use it.
    assert "import jettae" not in code
    assert "jettae.store.models." not in code
    command.upgrade(cfg, "head")  # the generated revision is valid, self-contained Python
    assert _tables(url) == MODEL_TABLES | {"alembic_version"}


@pytest.mark.skipif(
    os.name == "nt" and len(str(MIGRATIONS_DIR / "versions" / "0001_initial.py")) >= 260,
    reason="installed package path exceeds Windows MAX_PATH; covered via the checkout path above",
)
def test_upgrade_schema_with_packaged_migrations(tmp_path: Path) -> None:
    url = _sqlite_url(tmp_path / "packaged.db")
    upgrade_schema(url)
    assert _tables(url) == MODEL_TABLES | {"alembic_version"}
