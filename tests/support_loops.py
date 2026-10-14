"""Resilient asyncio event-loop policy for the test session and the interpreters it spawns.

Every Windows asyncio event loop opens a loopback socket pair for its self-pipe. Some sandboxed
Windows hosts (e.g. desktop-app sandboxes, CI runners with restrictive socket policies) refuse
that call now and then with ``WinError 10014``, which surfaces as random fixture *setup* errors
late in a long run, or as a child interpreter exiting with status 1. Product code already retries
loop creation where it opens loops itself (``jettae.cli.main``, ``jettae.cli.loops``,
``jettae.evalkit.conditions``); this policy gives the same resilience to loops opened by pytest
fixtures, ``asyncio.run`` and Starlette's TestClient, without changing what any test computes.

``conftest.py`` installs it for the session. Tests that start a fresh interpreter (hash-seed
checks) put this folder on ``sys.path`` and call :func:`install` first — see :data:`CHILD_PRELUDE`.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

#: A busy host (a parallel build, an office app rendering) has been seen refusing the socket pair
#: for a couple of seconds, so the policy waits up to ~12 s in total before letting the error out.
ATTEMPTS = 30
BACKOFF_SECONDS = 0.05
BACKOFF_CAP_SECONDS = 0.5

TESTS_DIR = Path(__file__).resolve().parent

#: Code to prepend to a ``python -c`` script so the child gets the same policy as the session.
CHILD_PRELUDE = (
    f"import sys\nsys.path.insert(0, {str(TESTS_DIR)!r})\nimport support_loops\nsupport_loops.install()\n"
)


def install() -> None:
    """Install the retrying loop policy (Windows only; other platforms keep asyncio's default)."""
    if sys.platform != "win32":
        return

    class _RetryingProactorPolicy(asyncio.WindowsProactorEventLoopPolicy):
        def new_event_loop(self) -> asyncio.AbstractEventLoop:
            for attempt in range(1, ATTEMPTS):
                try:
                    return super().new_event_loop()
                except OSError:
                    time.sleep(min(BACKOFF_SECONDS * attempt, BACKOFF_CAP_SECONDS))
            return super().new_event_loop()

    asyncio.set_event_loop_policy(_RetryingProactorPolicy())
