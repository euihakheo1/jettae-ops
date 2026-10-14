"""``python -m jettae.cli`` — the ``jettae`` command where the console script is not on ``PATH``
(an unactivated virtual environment, a CI image that installed the package elsewhere)."""

from jettae.cli.main import app

# Guarded: the evaluation harness spawns worker processes, which re-import the main module.
if __name__ == "__main__":
    app(prog_name="jettae")
