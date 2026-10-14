"""Ingest: turn runbook files and structured sources into trusted, guarded inputs.

Pipeline per document (ARCHITECTURE §4.1)::

    decode → normalize_text → mask_pii (length-preserving) → Document → scan_injection

No LLM is used anywhere in this package; everything is deterministic.
"""

from jettae.ingest.bundle import load_bundle_dir
from jettae.ingest.guard import COMMENT_MARKER, HIDDEN_MARKER, scan_injection
from jettae.ingest.loaders import DEFAULT_MAX_BYTES, MEDIA_TYPES, decode_text, load_bytes, load_document
from jettae.ingest.normalize import normalize_text
from jettae.ingest.pii import PiiHit, PiiKind, mask_pii
from jettae.ingest.structured import (
    KST,
    StructuredSource,
    load_catalog,
    load_oncall,
    load_policy,
    load_roster,
    load_yaml_documents,
    merge_catalogs,
    parse_channel,
    parse_status,
)

__all__ = [
    "COMMENT_MARKER",
    "DEFAULT_MAX_BYTES",
    "HIDDEN_MARKER",
    "KST",
    "MEDIA_TYPES",
    "PiiHit",
    "PiiKind",
    "StructuredSource",
    "decode_text",
    "load_bundle_dir",
    "load_bytes",
    "load_catalog",
    "load_document",
    "load_oncall",
    "load_policy",
    "load_roster",
    "load_yaml_documents",
    "mask_pii",
    "merge_catalogs",
    "normalize_text",
    "parse_channel",
    "parse_status",
    "scan_injection",
]
