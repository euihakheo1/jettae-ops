"""Text normalization — the first step after decoding any runbook.

Every downstream offset (PII hits, quarantine ranges, extractor spans, SARIF line numbers)
indexes into the *normalized* text, so normalization must be deterministic and must keep the
line structure of the source: one source line in, one normalized line out. That is what lets
a finding point at "runbook.md:42" and have the reader land on the right line.

Invisible characters are removed *before* the injection guard runs, because they are the
classic way to smuggle instructions past pattern matching ("ig<U+200B>nore previous …") or to
hide a payload entirely (Unicode TAG characters, bidi overrides).
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["normalize_text"]

#: Line separators other than LF. Each one becomes exactly one ``\n`` so the line count of the
#: normalized text equals the line count a reader sees in their editor.
_LINE_BREAKS = re.compile("\r\n|[\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029]")

#: Default-ignorable code points that are *not* category Cf/Cc and so escape
#: :func:`_is_invisible`: variation selectors (a known covert channel, "emoji smuggling"), the
#: combining grapheme joiner, Hangul conjoining fillers, Khmer inherent vowels, Mongolian free
#: variation selectors and the whole TAG/variation-selector supplement plane block. They render
#: as nothing, so "ig<U+034F>nore previous instructions" would otherwise slip past the guard.
_IGNORABLES = re.compile(
    "[\u034f\u115f\u1160\u17b4\u17b5\u180b-\u180d\u180f\ufe00-\ufe0f\U000e0000-\U000e0fff]"
)

#: Characters that are not category Zs but render as an empty *space* (Hangul filler "\u3164",
#: its half-width form, the blank Braille cell). They are mapped to a space rather than
#: removed: "ignore<U+3164>previous" reads as two words, and must match as two words.
_BLANK_LOOKALIKES = frozenset("\u3164\uffa0\u2800")

_TAB_AS_SPACES = " " * 4


def _is_invisible(ch: str) -> bool:
    """True for characters that occupy no visible width and carry no runbook meaning.

    Category ``Cf`` covers zero-width space/joiners (U+200B..U+200D), BOM (U+FEFF), word joiner
    (U+2060), soft hyphen, bidi controls and TAG characters; ``Cc`` covers C0/C1 controls.
    ``\\n`` is the only control that survives because it *is* the line structure.
    """
    if ch == "\n":
        return False
    return unicodedata.category(ch) in ("Cf", "Cc")


def _normalize_char(ch: str) -> str:
    if ch == "\t":
        return _TAB_AS_SPACES
    if _is_invisible(ch):
        return ""
    # NBSP, ideographic space (U+3000) etc. look like spaces to a reader, so they must match
    # like spaces too — otherwise "결제<NBSP>서비스" would never align with "결제 서비스".
    if ch != " " and (ch in _BLANK_LOOKALIKES or unicodedata.category(ch) == "Zs"):
        return " "
    return ch


def normalize_text(text: str) -> str:
    """Return the canonical form of ``text`` used for every offset in JETTAE.

    Steps: Unicode NFC → all line breaks to ``\\n`` → tabs to four spaces → strip invisible
    format/control characters and other default-ignorable code points → other space
    separators and blank look-alikes to ``" "`` → strip trailing spaces of each line. Blank
    lines are preserved so line numbers stay stable. The function is idempotent:
    ``normalize_text(normalize_text(t)) == normalize_text(t)``.
    """
    unified = _LINE_BREAKS.sub("\n", unicodedata.normalize("NFC", text))
    stripped = _IGNORABLES.sub("", unified)
    cleaned = "".join(_normalize_char(ch) for ch in stripped)
    # Removing a combining-sensitive invisible char can create a new composable pair, so NFC is
    # applied once more to keep the function idempotent.
    recomposed = unicodedata.normalize("NFC", cleaned)
    return "\n".join(line.rstrip(" ") for line in recomposed.split("\n"))
