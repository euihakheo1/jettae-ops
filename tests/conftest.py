"""Test-session configuration.

* Event loops: a retrying loop policy for hosts that sometimes refuse the self-pipe socket pair
  (``WinError 10014``) — see ``support_loops``.
* Hypothesis: no per-example deadline. Wall-clock deadlines turn a busy host (a parallel build,
  a virus scan, the first example paying for imports) into "flaky" failures without saying
  anything about correctness; examples, shrinking and the example database are unchanged.
  Select another registered profile with ``HYPOTHESIS_PROFILE``.
"""

from __future__ import annotations

import os

from hypothesis import settings

import support_loops

support_loops.install()

settings.register_profile("jettae", deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "jettae"))
