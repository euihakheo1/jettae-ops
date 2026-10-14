"""Improvement loop (개선): draft → validate → human approval → new graph version.

* :mod:`~jettae.improve.patching` — apply / invert / diff patch ops (judgement path, no LLM).
* :mod:`~jettae.improve.drafter` — fixes from fixed templates; an LLM may only reword them and pick
  among roster candidates (the ops are always compiled by code).
* :mod:`~jettae.improve.validator` — re-check, re-rehearse, no regression, nobody outside the roster
  (judgement path, no LLM).
* :mod:`~jettae.improve.versioning` — approve / reject / revert as append-only, audited versions.
"""

from jettae.improve.drafter import (
    ESCALATION_PROPOSAL_MINUTES,
    MEMO_KINDS,
    NO_CANDIDATE,
    Candidate,
    Draft,
    FixGroup,
    FixPlan,
    LlmDrafter,
    TemplateDrafter,
    choice_schema,
    group_findings,
)
from jettae.improve.patching import GraphDiff, PatchError, apply_ops, graph_diff, invert_ops
from jettae.improve.validator import BLOCKING_SEVERITIES, validate_patch
from jettae.improve.versioning import (
    DECIDER_ROLES,
    SYSTEM_ACTOR,
    Clock,
    PatchStateError,
    VersionService,
    utc_now,
)

__all__ = [
    "BLOCKING_SEVERITIES",
    "DECIDER_ROLES",
    "ESCALATION_PROPOSAL_MINUTES",
    "MEMO_KINDS",
    "NO_CANDIDATE",
    "SYSTEM_ACTOR",
    "Candidate",
    "Clock",
    "Draft",
    "FixGroup",
    "FixPlan",
    "GraphDiff",
    "LlmDrafter",
    "PatchError",
    "PatchStateError",
    "TemplateDrafter",
    "VersionService",
    "apply_ops",
    "choice_schema",
    "graph_diff",
    "group_findings",
    "invert_ops",
    "utc_now",
    "validate_patch",
]
