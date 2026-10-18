"""Run the product pipeline on real inputs and keep what the studies need from its outcome.

The product path is used unchanged — :func:`~jettae.orchestrator.pipeline.run_diagnosis` in LLM
mode: guard → rule parser + LLM extractor (hybrid) → assemble → back-translation check → fixed
rules → score. Drills and drafts are switched off (they neither change findings nor belong to
what is measured). The only intervention is :class:`SkipNonKoreanBacktranslation` for sources
that are not Korean (see :mod:`jettae.evalkit.realdata.providers`), and it is recorded.

A run whose LLM extraction failed (the pipeline then silently falls back to the rule parser) is
reported as :class:`PipelineFallbackError` instead of a result: a study item measured without its LLM
would not be the item the study claims to measure. A *chunk-local* failure (one chunk's answer
refused, truncated or invalid — the product keeps the rule parser's result for that chunk and
carries on) is the product's own behaviour and stays a result, but it is counted
(``extract_failed_chunks``) so the reports can say how many items were measured that way.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Final, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from jettae.domain import (
    Document,
    EdgeKind,
    EntityRef,
    ExtractedAttr,
    Extraction,
    Extractor,
    Finding,
    GroundingStatus,
    InputBundle,
    LlmMode,
    OpsGraph,
    SourceSpan,
    Verification,
)
from jettae.domain.graph import AttrValue
from jettae.evalkit.realdata.common import is_korean_text, rule_family
from jettae.evalkit.realdata.dryrun import PlannedCall
from jettae.evalkit.realdata.providers import SkipNonKoreanBacktranslation
from jettae.extract.align import ground_quote
from jettae.extract.backtranslate import _INSTRUCTIONS as BACKTRANSLATE_INSTRUCTIONS
from jettae.extract.backtranslate import RENDERED_DOC_ID, EquivalenceVerdict
from jettae.extract.llm_extractor import EXTRACT_TASK, blank_quarantined
from jettae.llm.base import LLMClient, LLMOutputError, LLMRefusalError, LLMResult, UntrustedDocument
from jettae.orchestrator.pipeline import run_diagnosis
from jettae.orchestrator.state import DiagnosisOptions

__all__ = [
    "BACKTRANSLATE_MAX_TOKENS",
    "BACKTRANSLATION_LLM",
    "BACKTRANSLATION_SKIPPED",
    "ChunkLog",
    "DiagnosisRecord",
    "EvidenceRecord",
    "FindingRecord",
    "GraphLog",
    "GroundingLog",
    "HoldRecord",
    "PipelineFallbackError",
    "QuoteLog",
    "chunk_log",
    "diagnose_bundle",
    "extraction_log",
    "graph_log",
    "grounding_log",
    "plan_backtranslations",
    "raw_line",
]

BACKTRANSLATION_SKIPPED: Final = "backtranslation: skipped (non-Korean source)"
BACKTRANSLATION_LLM: Final = "backtranslation: llm + slot check (Korean source)"
_MIXED: Final = "backtranslation: llm for Korean sources, skipped for non-Korean sources"


_T = TypeVar("_T", bound=BaseModel)


class PipelineFallbackError(RuntimeError):
    """The pipeline ran without its LLM extraction (an LLM error made it fall back)."""


class _ChunkFailureCounter:
    """Pass-through client counting extraction answers the product will drop chunk-locally
    (refused or invalid — the extractor catches exactly these two and goes on without the chunk)."""

    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner
        self.provider = inner.provider
        self.model = inner.model
        self.failed = 0

    async def structured(
        self,
        *,
        task: str,
        instructions: str,
        schema: type[_T],
        documents: tuple[UntrustedDocument, ...] = (),
        context: str = "",
        max_tokens: int = 16000,
    ) -> LLMResult[_T]:
        try:
            return await self.inner.structured(
                task=task,
                instructions=instructions,
                schema=schema,
                documents=documents,
                context=context,
                max_tokens=max_tokens,
            )
        except (LLMOutputError, LLMRefusalError):
            if task == EXTRACT_TASK:
                self.failed += 1
            raise


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True)


class EvidenceRecord(_Model):
    quote: str
    #: 1-based line in the normalised document text
    line: int | None
    #: 1-based line in the original file (what a source link points at)
    source_line: int | None
    url: str | None = None


class FindingRecord(_Model):
    key: str
    finding_id: str
    rule_id: str
    family: str
    severity: str
    status: str
    title: str
    message: str
    service_id: str | None
    service_name: str | None
    element: str | None
    doc_path: str | None
    evidence: tuple[EvidenceRecord, ...]


class HoldRecord(_Model):
    element_id: str
    kind: str
    reason: str | None
    backtranslation: bool


class QuoteLog(_Model):
    """One extractor quote and where grounding put it (``line`` is ``None`` when it was not found)."""

    quote: str
    #: 1-based line in the normalised document text
    line: int | None
    #: 1-based line in the original file
    source_line: int | None
    #: exact | normalized | fuzzy when grounded; quarantined | absent when dropped as ungrounded
    grounding: str


class AttrLog(_Model):
    key: str
    value: str
    quote: QuoteLog


class EntityLog(_Model):
    kind: str
    name: str
    quote: QuoteLog
    attrs: tuple[AttrLog, ...] = ()


class RelationLog(_Model):
    kind: str
    #: ``"<kind>:<name>"`` as the extractor wrote it
    src: str
    dst: str
    quote: QuoteLog
    attrs: tuple[AttrLog, ...] = ()


class SubjectLog(_Model):
    kind: str
    name: str
    services: tuple[str, ...]
    any_service: bool
    quote: QuoteLog


class DelegationLog(_Model):
    element: str
    target: str
    applies_to: str | None
    quote: QuoteLog


class SharedLog(_Model):
    kind: str
    dst: str
    #: empty = everything the document covers
    services: tuple[str, ...]
    quote: QuoteLog


class ChunkLog(_Model):
    """What one extractor proposed for one chunk, before the assembler grounded and resolved it."""

    extractor: str
    start: int
    end: int | None
    entities: tuple[EntityLog, ...] = ()
    relations: tuple[RelationLog, ...] = ()
    subjects: tuple[SubjectLog, ...] = ()
    delegations: tuple[DelegationLog, ...] = ()
    shared: tuple[SharedLog, ...] = ()


class GroundingLog(_Model):
    """The assembler's bookkeeping: accepted elements and what it dropped, with examples."""

    grounded: int
    fuzzy_grounded: int
    dropped_ungrounded: int
    dropped_quarantined: int
    dropped_invalid: int
    held_backtranslation: int
    held_ambiguous_entity: int
    dropped_examples: tuple[str, ...]


class GraphNodeLog(_Model):
    id: str
    kind: str
    name: str
    lines: tuple[int, ...]
    attrs: dict[str, str] = Field(default_factory=dict)


class GraphEdgeLog(_Model):
    id: str
    kind: str
    src: str
    dst: str
    lines: tuple[int, ...]


class GraphLog(_Model):
    nodes: tuple[GraphNodeLog, ...]
    edges: tuple[GraphEdgeLog, ...]


class DiagnosisRecord(_Model):
    graph_hash: str
    extractor: str
    nodes: int
    edges: int
    findings: tuple[FindingRecord, ...]
    holds: tuple[HoldRecord, ...]
    llm_calls: int
    llm_by_task: dict[str, int]
    llm_input_tokens: int
    llm_output_tokens: int
    backtranslation: str
    backtranslation_skipped_calls: int
    security_notices: int
    notes: tuple[str, ...]
    #: chunks whose LLM extraction answer was refused/invalid (diagnosed by the rule parser alone)
    extract_failed_chunks: int = 0
    #: the extraction IR per chunk with grounded lines (empty rule-parser chunks left out), the
    #: assembler's grounding bookkeeping and the graph it built — so a wrong finding can be traced
    #: to the proposal, to grounding, or to the rules (public documents only)
    extraction: tuple[ChunkLog, ...] = ()
    grounding: GroundingLog | None = None
    graph: GraphLog | None = None


#: max_tokens of one back-translation judge request (``jettae.extract.backtranslate``)
BACKTRANSLATE_MAX_TOKENS: Final = 1024
_PLACEHOLDER_RENDERED: Final = "1차 당직이 10분 안에 응답하지 않으면 팀장에게 상향한다."


def plan_backtranslations(item: str, doc: Document, count: int) -> list[PlannedCall]:
    """Placeholder back-translation judge requests for a Korean document (dry run).

    How many response edges the extraction will produce is unknown beforehand, so ``count``
    requests of the product's exact shape (quote + rendered sentence as untrusted documents, the
    edge kind as trusted context) stand in, each quoting one visible line of the document.
    """
    visible = blank_quarantined(doc.text, doc.quarantined)
    lines = [line.strip() for line in visible.split("\n") if line.strip()] or [" "]
    calls: list[PlannedCall] = []
    for index in range(count):
        quote = lines[index % len(lines)][:160]
        calls.append(
            PlannedCall(
                item=item,
                stage="backtranslate",
                role="main",
                instructions=BACKTRANSLATE_INSTRUCTIONS,
                schema=EquivalenceVerdict,
                documents=(
                    UntrustedDocument(doc_id=doc.doc_id, title=doc.title, text=quote),
                    UntrustedDocument(
                        doc_id=RENDERED_DOC_ID, title="역서술 문장", text=_PLACEHOLDER_RENDERED
                    ),
                ),
                context=f"관계 종류: {EdgeKind.ESCALATES_TO.value}",
                max_tokens=BACKTRANSLATE_MAX_TOKENS,
                korean=True,
                exact=False,
            )
        )
    return calls


def raw_line(raw: str | None, quote: str, normalised_line: int | None) -> int | None:
    """The quote's line in the original file: the occurrence nearest the normalised line.

    Ingest normalises text (invisible characters, PII masking), so offsets may shift; lines
    almost never do, but a source link must point at the file as published.
    """
    if raw is None or not quote:
        return normalised_line
    lines: list[int] = []
    start = raw.find(quote)
    while start != -1:
        lines.append(raw.count("\n", 0, start) + 1)
        start = raw.find(quote, start + 1)
    if not lines:
        return normalised_line
    if normalised_line is None:
        return lines[0]
    return min(lines, key=lambda line: (abs(line - normalised_line), line))


#: ``(doc, line) -> URL`` of a source line (``None`` when the study has no public source)
UrlFor = Callable[[Document, int | None], str | None]


# --------------------------------------------------------------------------- extraction log


def _value_text(value: AttrValue) -> str:
    return ", ".join(map(str, value)) if isinstance(value, list) else str(value)


@dataclass(frozen=True)
class _Locator:
    """Grounds quotes of one chunk the way the assembler does (same aligner, chunk as the hint)."""

    doc: Document | None
    raw: str | None
    start: int
    end: int | None

    def quote(self, quote: str) -> QuoteLog:
        if self.doc is None:
            return QuoteLog(quote=quote, line=None, source_line=None, grounding="absent")
        attempt = ground_quote(self.doc, quote, hint_start=self.start, hint_end=self.end)
        if attempt.alignment is None:
            grounding = "quarantined" if attempt.quarantined else "absent"
            return QuoteLog(quote=quote, line=None, source_line=None, grounding=grounding)
        span = attempt.alignment.span
        return QuoteLog(
            quote=quote,
            line=span.line,
            source_line=raw_line(self.raw, span.quote, span.line),
            grounding=attempt.alignment.method,
        )

    def attrs(self, attrs: tuple[ExtractedAttr, ...]) -> tuple[AttrLog, ...]:
        return tuple(AttrLog(key=a.key, value=_value_text(a.value), quote=self.quote(a.quote)) for a in attrs)


def _ref(ref: EntityRef) -> str:
    return f"{ref.kind.value}:{ref.name}"


def chunk_log(extraction: Extraction, docs: Mapping[str, Document], raws: Mapping[str, str]) -> ChunkLog:
    """The IR of one extraction with every quote's grounded line (or why it has none)."""
    at = _Locator(
        docs.get(extraction.doc_id), raws.get(extraction.doc_id), extraction.chunk_start, extraction.chunk_end
    )
    return ChunkLog(
        extractor=extraction.extractor.value,
        start=extraction.chunk_start,
        end=extraction.chunk_end,
        entities=tuple(
            EntityLog(kind=e.kind.value, name=e.name, quote=at.quote(e.quote), attrs=at.attrs(e.attrs))
            for e in extraction.entities
        ),
        relations=tuple(
            RelationLog(
                kind=r.kind.value,
                src=_ref(r.src),
                dst=_ref(r.dst),
                quote=at.quote(r.quote),
                attrs=at.attrs(r.attrs),
            )
            for r in extraction.relations
        ),
        subjects=tuple(
            SubjectLog(
                kind=s.kind.value,
                name=s.name,
                services=s.services,
                any_service=s.any_service,
                quote=at.quote(s.quote),
            )
            for s in extraction.subjects
        ),
        delegations=tuple(
            DelegationLog(
                element=d.element.value, target=d.target, applies_to=d.applies_to, quote=at.quote(d.quote)
            )
            for d in extraction.delegations
        ),
        shared=tuple(
            SharedLog(kind=s.kind.value, dst=_ref(s.dst), services=s.services, quote=at.quote(s.quote))
            for s in extraction.shared
        ),
    )


def _states_anything(extraction: Extraction) -> bool:
    return any(
        (
            extraction.entities,
            extraction.relations,
            extraction.subjects,
            extraction.delegations,
            extraction.shared,
        )
    )


def extraction_log(
    extractions: Iterable[Extraction], docs: Mapping[str, Document], raws: Mapping[str, str]
) -> tuple[ChunkLog, ...]:
    """Every LLM chunk (an empty answer is itself a finding about the extraction) and every rule
    parser extraction that proposed something."""
    return tuple(
        chunk_log(ex, docs, raws)
        for ex in extractions
        if ex.extractor is Extractor.LLM or _states_anything(ex)
    )


def _lines(spans: Iterable[SourceSpan]) -> tuple[int, ...]:
    return tuple(sorted({s.line for s in spans if s.line is not None}))


def graph_log(graph: OpsGraph) -> GraphLog:
    """Nodes and edges of the assembled graph with the lines they were grounded on."""
    return GraphLog(
        nodes=tuple(
            GraphNodeLog(
                id=n.id,
                kind=n.kind.value,
                name=n.name,
                lines=_lines(n.spans),
                attrs={k: _value_text(a.value) for k, a in sorted(n.attrs.items())},
            )
            for n in sorted(graph.nodes.values(), key=lambda n: n.id)
        ),
        edges=tuple(
            GraphEdgeLog(id=e.id, kind=e.kind.value, src=e.src, dst=e.dst, lines=_lines(e.spans))
            for e in sorted(graph.edges.values(), key=lambda e: e.id)
        ),
    )


def grounding_log(status: GroundingStatus) -> GroundingLog:
    return GroundingLog.model_validate(status.model_dump())


def _evidence(
    span: SourceSpan, docs: Mapping[str, Document], raws: Mapping[str, str], url_for: UrlFor | None
) -> EvidenceRecord:
    doc = docs.get(span.doc_id)
    line = span.line if span.line is not None else (doc.line_of(span.start) if doc is not None else None)
    source = raw_line(raws.get(span.doc_id), span.quote, line)
    url = url_for(doc, source) if url_for is not None and doc is not None else None
    return EvidenceRecord(quote=span.quote, line=line, source_line=source, url=url)


def _finding_record(
    finding: Finding,
    *,
    names: Mapping[str, str],
    docs: Mapping[str, Document],
    raws: Mapping[str, str],
    url_for: UrlFor | None,
    key_prefix: str,
) -> FindingRecord:
    evidence = tuple(_evidence(span, docs, raws, url_for) for span in finding.evidence)
    doc_paths = sorted({docs[s.doc_id].path or s.doc_id for s in finding.evidence if s.doc_id in docs})
    return FindingRecord(
        key=f"{key_prefix}::{finding.id}",
        finding_id=finding.id,
        rule_id=finding.rule_id,
        family=rule_family(finding.rule_id),
        severity=finding.severity.value,
        status=finding.status.value,
        title=finding.title,
        message=finding.message,
        service_id=finding.service_id,
        service_name=names.get(finding.service_id) if finding.service_id else None,
        element=finding.element.value if finding.element is not None else None,
        doc_path=doc_paths[0] if len(doc_paths) == 1 else None,
        evidence=evidence,
    )


async def diagnose_bundle(
    bundle: InputBundle,
    llm: LLMClient,
    *,
    concurrency: int,
    key_prefix: str,
    raw_texts: Mapping[str, str] | None = None,
    url_for: UrlFor | None = None,
) -> DiagnosisRecord:
    """The product pipeline (LLM on) on ``bundle``; raises :class:`PipelineFallbackError` if it fell back.

    ``raw_texts`` maps doc ids to the original file text (for source lines); ``key_prefix``
    namespaces finding keys (the document path or the incident id).
    """
    non_korean = frozenset(d.doc_id for d in bundle.documents if not is_korean_text(d.text))
    counted = _ChunkFailureCounter(llm)
    gated = SkipNonKoreanBacktranslation(counted, non_korean)
    options = DiagnosisOptions(rehearse=False, draft=False, llm_scenario_count=0, llm_concurrency=concurrency)
    outcome = await run_diagnosis(bundle, base_graph=None, llm=gated, llm_mode=LlmMode.LIVE, options=options)
    if outcome.extractor != "hybrid":
        reason = " ".join(outcome.notes) or "LLM 추출 결과가 없습니다"
        raise PipelineFallbackError(f"LLM 추출이 실패해 제품이 규칙 파서로 대체했습니다: {reason}")
    graph = outcome.graph
    docs = {d.doc_id: d for d in bundle.documents}
    names = {node_id: node.name for node_id, node in graph.nodes.items()}
    raws = dict(raw_texts or {})
    findings = tuple(
        _finding_record(f, names=names, docs=docs, raws=raws, url_for=url_for, key_prefix=key_prefix)
        for f in outcome.findings
    )
    holds = [
        HoldRecord(
            element_id=n.id,
            kind="node",
            reason=n.hold_reason,
            backtranslation=bool(n.hold_reason and "역번역" in n.hold_reason),
        )
        for n in graph.nodes.values()
        if n.verification is Verification.HOLD
    ]
    holds += [
        HoldRecord(
            element_id=e.id,
            kind="edge",
            reason=e.hold_reason,
            backtranslation=bool(e.hold_reason and "역번역" in e.hold_reason),
        )
        for e in graph.edges.values()
        if e.verification is Verification.HOLD
    ]
    if not non_korean:
        mode = BACKTRANSLATION_LLM
    elif len(non_korean) == len(docs):
        mode = BACKTRANSLATION_SKIPPED
    else:
        mode = _MIXED
    usage = outcome.usage
    return DiagnosisRecord(
        graph_hash=graph.content_hash(),
        extractor=outcome.extractor,
        nodes=len(graph.nodes),
        edges=len(graph.edges),
        findings=findings,
        holds=tuple(sorted(holds, key=lambda h: h.element_id)),
        llm_calls=usage.calls,
        llm_by_task=dict(usage.by_task),
        llm_input_tokens=usage.input_tokens,
        llm_output_tokens=usage.output_tokens,
        backtranslation=mode,
        backtranslation_skipped_calls=gated.skipped,
        security_notices=len(outcome.security),
        notes=outcome.notes,
        extract_failed_chunks=counted.failed,
        extraction=extraction_log(outcome.extractions, docs, raws),
        grounding=grounding_log(outcome.grounding),
        graph=graph_log(graph),
    )
