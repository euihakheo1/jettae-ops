"""Risk exploration (규칙 밖 위험 탐색): proposals beyond the fixed rules, judged by the replay.

The 35 fixed rules (:mod:`jettae.check`) answer a closed list of questions. This package looks for
risks outside that list and keeps the same division of labour as the rest of the product —
*proposal* and *judgement* are separate (P1):

* :mod:`~jettae.explore.sweep` — a deterministic exhaustive sweep (LLM 0회) over single and
  paired losses of what each critical service's response depends on, reporting minimal breaking
  sets (compared by what the replay loses under them) the rules do not already report;
* :mod:`~jettae.explore.ai` — optional generative proposals that may only choose among enumerated
  node ids and must state premises the graph lacks as ``assumptions``;
* :mod:`~jettae.explore.verify` — the deterministic replay with the service's policy deadline
  decides ``confirmed`` / ``refuted`` / ``needs_info`` for every proposal;
* :mod:`~jettae.explore.minimize` — 최소 실패 조건 자동 검증: the same replay drops the conditions a
  confirmed AI hypothesis' break does not need (it never looks for causes the proposal omits).

Hypotheses never change the findings, the readiness score or the gate; reports show them in a
separate section labelled "점수에 반영하지 않음". ``sweep``, ``verify``, ``drafts`` and ``minimize``
cannot import an LLM client (import-linter contract), and nothing in ``check``/``rehearse``/
``improve`` can import this package.
"""

from jettae.explore.ai import (
    EXPLORE_TASK,
    AiProposals,
    LlmRiskHypothesis,
    LlmRiskPayload,
    LlmRiskTarget,
    explore_context,
    propose_hypotheses,
)
from jettae.explore.drafts import HypothesisDraft, clean_text, draft_from_hypothesis, set_key
from jettae.explore.explorer import MAX_AI_HYPOTHESES, Exploration, carry_over, explore_graph, finish
from jettae.explore.minimize import (
    MINIMIZE_MAX_REPLAYS,
    Minimized,
    minimizable,
    minimize_hypotheses,
)
from jettae.explore.sweep import (
    DEFAULT_LIMITS,
    SweepLimits,
    SweepResult,
    known_breaks,
    perturbation_effect,
    sweep,
)
from jettae.explore.verify import STATUS_ORDER, hypothesis_scenario, verify_hypotheses

__all__ = [
    "DEFAULT_LIMITS",
    "EXPLORE_TASK",
    "MAX_AI_HYPOTHESES",
    "MINIMIZE_MAX_REPLAYS",
    "STATUS_ORDER",
    "AiProposals",
    "Exploration",
    "HypothesisDraft",
    "LlmRiskHypothesis",
    "LlmRiskPayload",
    "LlmRiskTarget",
    "Minimized",
    "SweepLimits",
    "SweepResult",
    "carry_over",
    "clean_text",
    "draft_from_hypothesis",
    "explore_context",
    "explore_graph",
    "finish",
    "hypothesis_scenario",
    "known_breaks",
    "minimizable",
    "minimize_hypotheses",
    "perturbation_effect",
    "propose_hypotheses",
    "set_key",
    "sweep",
    "verify_hypotheses",
]
