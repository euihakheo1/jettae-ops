"""Alembic environment for the JETTAE Ops SQL store.

Database URL resolution (first hit wins), so the same scripts serve developers, CI and
deployments without editing ``alembic.ini``:

1. ``alembic -x url=…`` on the command line;
2. ``sqlalchemy.url`` in the config (set programmatically by
   :func:`jettae.store.sql.alembic_config`, empty in the shipped ``alembic.ini``);
3. ``JETTAE_DATABASE_URL`` (environment or ``.env``, via :class:`jettae.config.Settings`),
   whose default is the development SQLite file.

A caller may also hand over an open connection in ``config.attributes["connection"]`` to run
migrations inside its own transaction. SQLite migrations use batch mode because SQLite cannot
``ALTER`` most column properties in place.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import Connection

from jettae.config import Settings
from jettae.store.models import Base, render_item
from jettae.store.sql import build_engine

config = context.config
target_metadata = Base.metadata

if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    # Keep loggers configured by the host application alive; embedding callers (tests, the
    # API) can opt out entirely with ``config.attributes["configure_logger"] = False``.
    fileConfig(config.config_file_name, disable_existing_loggers=False)


def database_url() -> str:
    cli_url = context.get_x_argument(as_dictionary=True).get("url")
    if cli_url:
        return cli_url
    configured = config.get_main_option("sqlalchemy.url")
    if configured:
        return configured
    return Settings().database_url


def run_migrations_offline() -> None:
    """Emit SQL to the output buffer instead of executing it (``alembic upgrade --sql``)."""
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def _run_with(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=connection.dialect.name == "sqlite",
        compare_type=True,
        render_item=render_item,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    shared = config.attributes.get("connection")
    if isinstance(shared, Connection):
        _run_with(shared)
        return
    engine = build_engine(database_url())
    try:
        with engine.connect() as connection:
            _run_with(connection)
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
