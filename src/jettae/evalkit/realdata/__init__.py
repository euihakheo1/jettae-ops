"""Real-data validation harness (실데이터 실증): the product on inputs nobody built for it.

Two studies, both judged by two independent LLM coders of different model families:

1. :mod:`~jettae.evalkit.realdata.runbooks` — public runbooks (GitLab.com operations runbooks):
   are the product's findings correct about the documents? (precision with Wilson intervals,
   Cohen's κ / Krippendorff's α, a blind human-review sample).
2. :mod:`~jettae.evalkit.realdata.retro` — postmortem retrospective: of the response-process gaps
   an incident report states, how many would the product have flagged on the pre-incident set-up?

Paid calls are guarded end to end (:mod:`~jettae.evalkit.realdata.entry`): a dry run prices every
request first (free ``count_tokens``), a real run needs a dollar cap and a confirmation, refuses
to start above the cap, tracks actual usage and stops cleanly at it, and resumes where it stopped.
The cap is cumulative per output folder (a spend ledger, :mod:`~jettae.evalkit.realdata.resume`),
and a study can grow: pointed at a superset of its inputs it processes — and pays for — only the
new items, then recomputes every metric over all of them.
The CLI binds this package by name (``jettae eval real-runbooks`` / ``jettae eval retro``).
"""

from jettae.evalkit.realdata.clients import StudyClients, build_study_clients
from jettae.evalkit.realdata.coding import RUBRIC_VERSION, derive_label
from jettae.evalkit.realdata.common import (
    FAMILY_CODES,
    Agreement,
    is_korean_text,
    measure_agreement,
    precision,
    rule_family,
)
from jettae.evalkit.realdata.diagnose import (
    BACKTRANSLATION_SKIPPED,
    DiagnosisRecord,
    FindingRecord,
    PipelineFallbackError,
    diagnose_bundle,
)
from jettae.evalkit.realdata.dryrun import (
    AnthropicTokenCounter,
    CharTokenCounter,
    CostTable,
    PlannedCall,
    estimate_costs,
)
from jettae.evalkit.realdata.entry import CommandResult, PaidRunRefusedError, run_real_runbooks, run_retro
from jettae.evalkit.realdata.keys import load_api_key, parse_dotenv
from jettae.evalkit.realdata.pricing import PRICES, ModelRef, call_usd
from jettae.evalkit.realdata.providers import (
    BudgetedLLM,
    BudgetExhausted,
    OpenAIResponsesLLM,
    RecordingAnthropic,
    SkipNonKoreanBacktranslation,
    SpendTracker,
)
from jettae.evalkit.realdata.resume import InputChangeError, SpendLedger
from jettae.evalkit.realdata.retro import RetroConfig, load_reports, plan_retro_calls, run_retro_study
from jettae.evalkit.realdata.runbooks import (
    RunbookStudyConfig,
    load_runbooks,
    load_study_sources,
    plan_runbook_calls,
    review_sample,
    run_runbook_study,
)
from jettae.evalkit.realdata.split import (
    SPLIT_NAMES,
    SplitAssignment,
    SplitError,
    assign_split,
    load_split,
)

__all__ = [
    "BACKTRANSLATION_SKIPPED",
    "FAMILY_CODES",
    "PRICES",
    "RUBRIC_VERSION",
    "SPLIT_NAMES",
    "Agreement",
    "AnthropicTokenCounter",
    "BudgetExhausted",
    "BudgetedLLM",
    "CharTokenCounter",
    "CommandResult",
    "CostTable",
    "DiagnosisRecord",
    "FindingRecord",
    "InputChangeError",
    "ModelRef",
    "OpenAIResponsesLLM",
    "PaidRunRefusedError",
    "PipelineFallbackError",
    "PlannedCall",
    "RecordingAnthropic",
    "RetroConfig",
    "RunbookStudyConfig",
    "SkipNonKoreanBacktranslation",
    "SpendLedger",
    "SpendTracker",
    "SplitAssignment",
    "SplitError",
    "StudyClients",
    "assign_split",
    "build_study_clients",
    "call_usd",
    "derive_label",
    "diagnose_bundle",
    "estimate_costs",
    "is_korean_text",
    "load_api_key",
    "load_reports",
    "load_runbooks",
    "load_split",
    "load_study_sources",
    "measure_agreement",
    "parse_dotenv",
    "plan_retro_calls",
    "plan_runbook_calls",
    "precision",
    "review_sample",
    "rule_family",
    "run_real_runbooks",
    "run_retro",
    "run_retro_study",
    "run_runbook_study",
]
