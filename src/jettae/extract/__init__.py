"""Extraction: runbook text → grounded operations graph (ARCHITECTURE §4.3).

Two extractors write the same IR (:mod:`jettae.domain.extraction`) — the Korean rule parser
(:class:`RuleExtractor`, no LLM, the P4 fallback) and :class:`LlmExtractor` — and everything after
them is deterministic: :func:`assemble` grounds every quote in the source (:func:`align_quote`),
resolves people against the roster (:class:`EntityResolver`) and builds the graph;
:func:`verify_backtranslation` renders each element back into Korean (:func:`render_sentence`)
and puts it on HOLD when its slots disagree with the source (:func:`slot_check`).

Only the stable entry points are re-exported here; helpers stay in their modules.
"""

from jettae.extract.align import align_quote
from jettae.extract.assemble import AssemblyResult, assemble
from jettae.extract.backtranslate import (
    VerificationResult,
    render_sentence,
    slot_check,
    verify_backtranslation,
)
from jettae.extract.llm_extractor import ExtractionRun, LlmExtractor
from jettae.extract.resolve import EntityResolver
from jettae.extract.rules import RuleExtractor

__all__ = [
    "AssemblyResult",
    "EntityResolver",
    "ExtractionRun",
    "LlmExtractor",
    "RuleExtractor",
    "VerificationResult",
    "align_quote",
    "assemble",
    "render_sentence",
    "slot_check",
    "verify_backtranslation",
]
