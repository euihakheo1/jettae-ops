"""Mock-incident rehearsal (모의 장애 리허설): scenarios in, deterministic verdicts out.

``simulator`` is the judge (no LLM import, enforced by import-linter); ``scenarios`` writes
the questions — fixed templates, hand-written drills (``scenarios.yaml``) resolved onto the
graph, or LLM ideas validated against the graph.
"""

from jettae.rehearse.scenarios import (
    LlmPerturbation,
    LlmScenario,
    LlmScenarioPayload,
    llm_scenarios,
    scenarios_from_yaml,
    template_scenarios,
)
from jettae.rehearse.simulator import config_hash, replay, replay_all

__all__ = [
    "LlmPerturbation",
    "LlmScenario",
    "LlmScenarioPayload",
    "config_hash",
    "llm_scenarios",
    "replay",
    "replay_all",
    "scenarios_from_yaml",
    "template_scenarios",
]
