"""Command line interface (``jettae``): scan, rehearse, rules, serve, demo, eval, db.

* :mod:`~jettae.cli.main` — the Typer app (console script ``jettae = jettae.cli.main:app``); every
  use case goes through :class:`~jettae.orchestrator.JettaeService`.
* :mod:`~jettae.cli.output` — Rich output for people, plain text and UTF-8 files for machines.
* ``python -m jettae.cli`` runs the same app where the console script is not on ``PATH``.
"""

from jettae.cli.main import app

__all__ = ["app"]
