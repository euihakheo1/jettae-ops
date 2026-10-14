"""``python -m jettae.evalkit`` — run the sealed evaluation (see :mod:`jettae.evalkit.report`).

A package ``__main__`` rather than ``python -m jettae.evalkit.report``: the package exports the
report API, and running an already-imported submodule as ``__main__`` would load it twice. The
guard also keeps process-pool workers (spawned, they re-import the main module) from re-running it.
"""

from jettae.evalkit.report import main

if __name__ == "__main__":
    raise SystemExit(main())
