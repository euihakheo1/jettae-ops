"""Extraction IR — the common output of every extractor (rule parser or LLM).

Both extractors emit *mentions with verbatim quotes*. The assembler
(``jettae.extract.assemble``) aligns every quote back to the source document, resolves
entities against the roster/catalog, drops anything it cannot ground, and builds the
:class:`~jettae.domain.graph.OpsGraph`. This single funnel is what makes the 2×2 ablation
(rule vs LLM extraction × no-graph vs graph checking) a fair comparison.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from jettae.domain.graph import AttrValue, EdgeKind, Extractor, NodeKind


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


class Extraction(BaseModel):
    """Extractor output for one document (or one chunk of it)."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    extractor: Extractor
    entities: tuple[ExtractedEntity, ...] = ()
    relations: tuple[ExtractedRelation, ...] = ()
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
