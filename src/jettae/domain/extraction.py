"""Extraction IR — the common output of every extractor (rule parser or LLM).

Both extractors emit *mentions with verbatim quotes*. The assembler
(``jettae.extract.assemble``) aligns every quote back to the source document, resolves
entities against the roster/catalog, drops anything it cannot ground, and builds the
:class:`~jettae.domain.graph.OpsGraph`. This single funnel is what makes the 2×2 ablation
(rule vs LLM extraction × no-graph vs graph checking) a fair comparison.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from jettae.domain.graph import AttrValue, DelegatedElement, EdgeKind, Extractor, NodeKind


class ExtractedAttr(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(description="AttrKey value, e.g. 'after_minutes', 'tier', 'medium'")
    value: AttrValue
    quote: str = Field(description="verbatim substring of the document that states this value")


class EntityRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: NodeKind
    name: str = Field(description="surface form as written in the document, e.g. '김대리', '결제 서비스'")


class ExtractedEntity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: NodeKind
    name: str
    quote: str = Field(description="verbatim substring where the entity is mentioned")
    attrs: tuple[ExtractedAttr, ...] = ()


class ExtractedRelation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: EdgeKind
    src: EntityRef
    dst: EntityRef
    quote: str = Field(description="verbatim substring that states the relation")
    attrs: tuple[ExtractedAttr, ...] = ()


class SubjectKind(StrEnum):
    """What a document says it is the runbook *for*."""

    ALERT = "alert"  # the runbook of one alert ("This alert fires when …")
    SERVICE = "service"  # the runbook of one service / component
    PROCEDURE = "procedure"  # a general procedure (incident workflow, on-call handbook …)
    OTHER = "other"


class ExtractedSubject(BaseModel):
    """The document's own statement of what it covers.

    An alert runbook rarely writes "service X is detected by alert Y": the document *being* the
    runbook of alert Y is the statement. The assembler links such an alert as the detection signal
    of the services it covers — the ones named here, or, with ``any_service``, every service the
    same document mentions (the alert fires for whichever service breaches it).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: SubjectKind
    name: str = Field(description="alert / service / procedure name as written")
    services: tuple[str, ...] = Field(default=(), description="services/systems it covers, as written")
    any_service: bool = Field(default=False, description="applies to whichever service raised it")
    quote: str = Field(description="verbatim substring stating what the document is for")


class ExtractedDelegation(BaseModel):
    """A response-path element the document hands off to an external source instead of stating it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    element: DelegatedElement
    target: str = Field(description="the external source as written, e.g. 'team directory'")
    #: service/system the delegation is about, as written; ``None`` = everything the document covers
    applies_to: str | None = None
    quote: str = Field(description="verbatim substring that states the delegation")


class ExtractedSharedRelation(BaseModel):
    """One relation the document states for several services at once, or for everything it covers.

    Runbooks often state a response step once for a group ("services A, B and C can be moved to
    another zone") or for the whole document (the resolution steps of an alert runbook, a step
    "for all components"). Such a sentence states the relation for *each* service it covers; the
    assembler links every one of them (the services named here, or — with ``services`` empty —
    every service the same document mentions) to ``dst``, grounded on the stating sentence.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: EdgeKind
    dst: EntityRef
    #: services/systems as written; empty = every service the document covers
    services: tuple[str, ...] = ()
    quote: str = Field(description="verbatim substring that states the relation for the group")


class Extraction(BaseModel):
    """Extractor output for one document (or one chunk of it)."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    extractor: Extractor
    entities: tuple[ExtractedEntity, ...] = ()
    relations: tuple[ExtractedRelation, ...] = ()
    #: document-level statements (only the LLM extractor reads them; the rule parser leaves them empty)
    subjects: tuple[ExtractedSubject, ...] = ()
    delegations: tuple[ExtractedDelegation, ...] = ()
    shared: tuple[ExtractedSharedRelation, ...] = ()
    #: offsets of the chunk inside the document, used to disambiguate repeated quotes
    chunk_start: int = 0
    chunk_end: int | None = None


class GroundingStatus(BaseModel):
    """Bookkeeping for the hallucination guard (KPI: 원문 근거 위치 없는 필드 0건)."""

    model_config = ConfigDict(frozen=True)

    grounded: int = 0
    fuzzy_grounded: int = 0
    dropped_ungrounded: int = 0
    dropped_quarantined: int = 0
    dropped_invalid: int = 0
    held_backtranslation: int = 0
    held_ambiguous_entity: int = 0
    dropped_examples: tuple[str, ...] = ()
