"""Orchestration: one diagnosis as a LangGraph pipeline, and the application service around it.

* :mod:`~jettae.orchestrator.pipeline` — ``guard → extract → assemble → verify → check → score →
  scenarios → rehearse → draft → finalize`` with conditional fallback (LLM failure → rule parser)
  and rebase (keep approved patches across a re-extraction).
* :mod:`~jettae.orchestrator.service` — :class:`JettaeService`, the only entry point for the API
  and the CLI.
* :mod:`~jettae.orchestrator.demo` — the finals demo bundle and cassette recording.
"""

from jettae.orchestrator.demo import DEMO_WORKSPACE_NAME, demo_bundle_path, record_cassette
from jettae.orchestrator.pipeline import (
    NODE_NAMES,
    Evaluation,
    PipelineContext,
    Rebase,
    build_diagnosis_graph,
    diagnosis_graph,
    evaluate_graph,
    rebase_graph,
    run_diagnosis,
)
from jettae.orchestrator.service import (
    EVENT_RUNBOOK_UPLOAD,
    FREE_SCAN_WORKSPACE_NAME,
    ApprovalResult,
    DemandMetrics,
    FreeScanOutcome,
    InputSummary,
    JettaeService,
    Lineage,
    PurgeOutcome,
    ScenarioGeneration,
)
from jettae.orchestrator.state import DiagnosisOptions, DiagnosisOutcome, DiagnosisState, ExtractorPolicy

__all__ = [
    "DEMO_WORKSPACE_NAME",
    "EVENT_RUNBOOK_UPLOAD",
    "FREE_SCAN_WORKSPACE_NAME",
    "NODE_NAMES",
    "ApprovalResult",
    "DemandMetrics",
    "DiagnosisOptions",
    "DiagnosisOutcome",
    "DiagnosisState",
    "Evaluation",
    "ExtractorPolicy",
    "FreeScanOutcome",
    "InputSummary",
    "JettaeService",
    "Lineage",
    "PipelineContext",
    "PurgeOutcome",
    "Rebase",
    "ScenarioGeneration",
    "build_diagnosis_graph",
    "demo_bundle_path",
    "diagnosis_graph",
    "evaluate_graph",
    "rebase_graph",
    "record_cassette",
    "run_diagnosis",
]
