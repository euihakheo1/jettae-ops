"""Event-loop factory for the API server.

Every Windows asyncio loop opens a loopback socket pair for its self-pipe. Some sandboxed or
policy-restricted Windows hosts refuse that call now and then (``WinError 10014``), which makes
uvicorn die before it ever binds the port. The server's loop creation is retried with a short
back-off; nothing the server does afterwards is retried. Other platforms keep uvicorn's default.
"""

from __future__ import annotations

import asyncio
import sys
import time

ATTEMPTS = 10
BACKOFF_SECONDS = 0.1

#: uvicorn ``loop=`` value: an import string for Windows, uvicorn's own default elsewhere.
UVICORN_LOOP = "jettae.cli.loops:resilient_event_loop" if sys.platform == "win32" else "auto"


def resilient_event_loop() -> asyncio.AbstractEventLoop:
    """Return a new event loop, retrying creation when the OS refuses the self-pipe socket pair."""
    for attempt in range(1, ATTEMPTS):
        try:
            return asyncio.new_event_loop()
        except OSError:
            time.sleep(BACKOFF_SECONDS * attempt)
    return asyncio.new_event_loop()
