"""Grounding: map an extractor's verbatim quote back to exact character offsets.

Why this exists (P2 지어내기 금지): extractors — especially an LLM — *claim* that a phrase
appears in the runbook. The graph only believes that claim once the phrase is found in
``Document.text`` and pinned to a :class:`~jettae.domain.graph.SourceSpan`. Whatever cannot be
pinned is dropped by the assembler, so every runbook-derived element carries evidence a human
can click on.

Search order (strongest evidence first, see docs/ARCHITECTURE.md §4.3):

1. exact match, preferring occurrences inside the hint window (the extractor's chunk) and then
   the occurrence nearest to it — the same phrase often repeats across sections;
2. whitespace-insensitive match — models (and Korean 띄어쓰기) routinely add or drop spaces
   (invisible format characters such as zero-width spaces are ignored the same way);
3. ``rapidfuzz.fuzz.partial_ratio_alignment`` with score >= 90, only for quotes of 8+ chars
   because short strings reach high similarity by accident;
4. otherwise no alignment.

Quarantine (P5): occurrences inside a quarantined range (prompt-injection text) never count as
evidence. When the exact occurrences are all quarantined, the whitespace-insensitive stage may
still find the same characters in clean text — that is genuine evidence, and refusing it would
let injected copies of legitimate sentences knock those sentences out of the graph. The fuzzy
stage, however, is never tried once quarantined text matched: a phrase lifted from injected text
must not be laundered onto merely *similar* legitimate text. Such quotes are reported as
quarantined.
"""

from __future__ import annotations

import functools
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from rapidfuzz import fuzz

from jettae.domain import Document, SourceSpan

AlignMethod = Literal["exact", "normalized", "fuzzy"]

#: Minimum rapidfuzz score for a fuzzy alignment (architecture contract).
FUZZY_MIN_SCORE = 90.0
#: Quotes shorter than this are never fuzzy-aligned: "김대리" vs "김과장" is already 67 %.
FUZZY_MIN_CHARS = 8
#: How far (in chars) fuzzy span boundaries may move while refining the alignment.
_BOUNDARY_SLACK = 3

_EXACT_SCORE = 100.0


@dataclass(frozen=True)
class Alignment:
    """Where a quote was found, how, and how confidently (100 for exact/normalized)."""

    span: SourceSpan
    method: AlignMethod
    score: float


@dataclass(frozen=True)
class GroundingAttempt:
    """Outcome of grounding one quote.

    ``quarantined`` distinguishes "found, but only inside quarantined text" from "not found at
    all" so the assembler can report ``dropped_quarantined`` separately from
    ``dropped_ungrounded`` (security KPI vs hallucination KPI).
    """

    alignment: Alignment | None
    quarantined: bool = False


@dataclass(frozen=True)
class _Candidate:
    start: int
    end: int
    score: float


def align_quote(
    doc: Document, quote: str, *, hint_start: int = 0, hint_end: int | None = None
) -> Alignment | None:
    """Locate ``quote`` in ``doc`` (see module docstring for the search order).

    Returns ``None`` when the quote cannot be grounded or is grounded only in quarantine.
    """
    return ground_quote(doc, quote, hint_start=hint_start, hint_end=hint_end).alignment


def ground_quote(
    doc: Document, quote: str, *, hint_start: int = 0, hint_end: int | None = None
) -> GroundingAttempt:
    """Like :func:`align_quote` but also tells *why* grounding failed (quarantine vs absent)."""
    needle = unicodedata.normalize("NFC", quote).strip()
    if all(_invisible(ch) for ch in needle):  # nothing a reader could see is quoted
        return GroundingAttempt(None)
    text = doc.text
    lo = min(max(hint_start, 0), len(text))
    hi = len(text) if hint_end is None else min(max(hint_end, lo), len(text))

    quarantined = False
    for method, find in _STAGES:
        if quarantined and method == "fuzzy":
            break  # see module docstring: no similarity search after a quarantined match
        candidates = find(text, needle, lo, hi)
        for cand in candidates:
            if not doc.is_quarantined(cand.start, cand.end):
                return GroundingAttempt(Alignment(_span(doc, cand.start, cand.end), method, cand.score))
        quarantined = quarantined or bool(candidates)
    return GroundingAttempt(None, quarantined=quarantined)


# --------------------------------------------------------------------------- stages


def _distance(start: int, end: int, lo: int, hi: int) -> int:
    """0 when ``[start, end)`` lies inside the hint window, otherwise how far it sticks out."""
    return max(0, lo - start) + max(0, end - hi)


def _rank(cands: list[_Candidate], lo: int, hi: int) -> list[_Candidate]:
    return sorted(cands, key=lambda c: (_distance(c.start, c.end, lo, hi), c.start))


def _exact_candidates(text: str, needle: str, lo: int, hi: int) -> list[_Candidate]:
    found: list[_Candidate] = []
    pos = text.find(needle)
    while pos != -1:
        found.append(_Candidate(pos, pos + len(needle), _EXACT_SCORE))
        pos = text.find(needle, pos + 1)
    return _rank(found, lo, hi)


def _invisible(ch: str) -> bool:
    """Whitespace or an invisible format character (zero-width space/joiner, soft hyphen, BOM).

    Ingest strips format characters from documents, but a model may still emit them inside a
    quote (or a document may be built without ingest); neither should block grounding.
    """
    return ch.isspace() or unicodedata.category(ch) == "Cf"


@functools.lru_cache(maxsize=64)
def _compact_index(text: str) -> tuple[str, tuple[int, ...]]:
    """Text with invisible characters removed + the original offset of every kept character.

    Cached because the assembler grounds hundreds of quotes against the same document.
    """
    kept = [(i, ch) for i, ch in enumerate(text) if not _invisible(ch)]
    return "".join(ch for _, ch in kept), tuple(i for i, _ in kept)


def _normalized_candidates(text: str, needle: str, lo: int, hi: int) -> list[_Candidate]:
    compact_needle = "".join(ch for ch in needle if not _invisible(ch))
    if not compact_needle:
        return []
    compact_text, index = _compact_index(text)
    found: list[_Candidate] = []
    pos = compact_text.find(compact_needle)
    while pos != -1:
        start = index[pos]
        end = index[pos + len(compact_needle) - 1] + 1
        found.append(_Candidate(start, end, _EXACT_SCORE))
        pos = compact_text.find(compact_needle, pos + 1)
    return _rank(found, lo, hi)


def _fuzzy_candidates(text: str, needle: str, lo: int, hi: int) -> list[_Candidate]:
    if len(needle) < FUZZY_MIN_CHARS:
        return []
    windows = [(lo, hi)] if (lo, hi) == (0, len(text)) else [(lo, hi), (0, len(text))]
    found: list[_Candidate] = []
    for w_lo, w_hi in windows:
        window = text[w_lo:w_hi]
        if len(needle) > len(window):
            continue
        hit = fuzz.partial_ratio_alignment(needle, window, score_cutoff=FUZZY_MIN_SCORE)
        if hit is None:
            continue
        cand = _refine(text, needle, w_lo + hit.dest_start, w_lo + hit.dest_end)
        if cand.score >= FUZZY_MIN_SCORE and cand not in found:
            found.append(cand)
    return found


def _refine(text: str, needle: str, start: int, end: int) -> _Candidate:
    """Nudge fuzzy boundaries so the span covers the whole phrase and not stray whitespace.

    ``partial_ratio_alignment`` returns a window as long as the needle; when the needle lost
    a character the window cuts the last syllable off. We pick the best-scoring boundaries
    within a few characters, preferring the smallest move on ties (deterministic).
    """
    best: tuple[float, int, int, int] | None = None
    for ds in range(-_BOUNDARY_SLACK, _BOUNDARY_SLACK + 1):
        for de in range(-_BOUNDARY_SLACK, _BOUNDARY_SLACK + 1):
            s, e = start + ds, end + de
            if s < 0 or e > len(text) or e <= s or text[s].isspace() or text[e - 1].isspace():
                continue
            score = fuzz.ratio(needle, text[s:e])
            key = (-score, abs(ds) + abs(de), s, e)
            if best is None or key < best:
                best = key
    if best is None:
        return _Candidate(start, end, fuzz.ratio(needle, text[start:end]))
    return _Candidate(best[2], best[3], -best[0])


def _span(doc: Document, start: int, end: int) -> SourceSpan:
    quote = doc.text[start:end]
    return SourceSpan(doc_id=doc.doc_id, start=start, end=end, quote=quote, line=doc.line_of(start))


_Finder = Callable[[str, str, int, int], list[_Candidate]]

#: Evaluated in order; a later (costlier, weaker) stage runs only when earlier ones found nothing.
_STAGES: tuple[tuple[AlignMethod, _Finder], ...] = (
    ("exact", _exact_candidates),
    ("normalized", _normalized_candidates),
    ("fuzzy", _fuzzy_candidates),
)
