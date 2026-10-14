"""normalize_text: canonical text whose offsets/line numbers every other module relies on."""

from __future__ import annotations

import re
import unicodedata

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jettae.ingest import normalize_text

ZWSP = "\N{ZERO WIDTH SPACE}"
ZWNJ = "\N{ZERO WIDTH NON-JOINER}"
ZWJ = "\N{ZERO WIDTH JOINER}"
BOM = "\N{ZERO WIDTH NO-BREAK SPACE}"
WORD_JOINER = "\N{WORD JOINER}"
RLO = "\N{RIGHT-TO-LEFT OVERRIDE}"
SOFT_HYPHEN = "\N{SOFT HYPHEN}"
TAG_A = chr(0xE0041)  # Unicode TAG LATIN CAPITAL LETTER A ("ASCII smuggling")
VS16 = chr(0xFE0F)
NBSP = "\N{NO-BREAK SPACE}"
IDEOGRAPHIC_SPACE = "\N{IDEOGRAPHIC SPACE}"
LINE_SEP = "\N{LINE SEPARATOR}"
PARA_SEP = "\N{PARAGRAPH SEPARATOR}"

#: every sequence the normalizer treats as one line break (LF included)
_BREAKS = re.compile("\r\n|[\n\r\x0b\x0c\x1c\x1d\x1e\x85" + LINE_SEP + PARA_SEP + "]")


def test_crlf_and_cr_become_lf_keeping_line_count() -> None:
    assert normalize_text("a\r\nb\rc\nd") == "a\nb\nc\nd"


def test_unicode_line_and_paragraph_separators_become_lf() -> None:
    assert normalize_text(f"a{LINE_SEP}b{PARA_SEP}c\x0cd") == "a\nb\nc\nd"


def test_tabs_become_four_spaces() -> None:
    assert normalize_text("\t- 결제\tAPI") == "    - 결제    API"


@pytest.mark.parametrize(
    "invisible", [ZWSP, ZWNJ, ZWJ, BOM, WORD_JOINER, RLO, SOFT_HYPHEN, TAG_A, VS16, "\x00", "\x1b"]
)
def test_invisible_characters_are_stripped(invisible: str) -> None:
    assert normalize_text(f"ig{invisible}nore 결{invisible}제") == "ignore 결제"


def test_leading_bom_is_removed() -> None:
    assert normalize_text(f"{BOM}# 런북") == "# 런북"


def test_nfc_composes_hangul_jamo_and_accents() -> None:
    decomposed = "\N{HANGUL CHOSEONG KIYEOK}\N{HANGUL JUNGSEONG A}" + "e\N{COMBINING ACUTE ACCENT}"
    assert normalize_text(decomposed) == "가é"


def test_space_separators_become_ascii_space() -> None:
    assert normalize_text(f"결제{NBSP}서비스{IDEOGRAPHIC_SPACE}장애") == "결제 서비스 장애"


def test_trailing_spaces_stripped_leading_kept_blank_lines_preserved() -> None:
    assert normalize_text("  a  \n\n\t\nb ") == "  a\n\n\nb"


def test_empty_text() -> None:
    assert normalize_text("") == ""


_ALPHABET = st.sampled_from(
    [
        "a", "Z", "1", " ", "\t", "\n", "\r", "\r\n", "결", "제", "-", "|",
        ZWSP, BOM, RLO, TAG_A, VS16, NBSP, LINE_SEP, "\x0b", "\x85",
        "\N{HANGUL CHOSEONG KIYEOK}", "\N{HANGUL JUNGSEONG A}", "\N{COMBINING ACUTE ACCENT}",
    ]
)  # fmt: skip
_TEXTS = st.lists(_ALPHABET, max_size=60).map("".join) | st.text(max_size=60)


@settings(max_examples=300, deadline=None)
@given(_TEXTS)
def test_property_idempotent(text: str) -> None:
    once = normalize_text(text)
    assert normalize_text(once) == once


@settings(max_examples=300, deadline=None)
@given(_TEXTS)
def test_property_line_structure_preserved(text: str) -> None:
    assert normalize_text(text).count("\n") == len(_BREAKS.findall(text))


@settings(max_examples=300, deadline=None)
@given(_TEXTS)
def test_property_output_has_no_invisible_or_tab_and_no_trailing_space(text: str) -> None:
    out = normalize_text(text)
    assert "\t" not in out
    assert "\r" not in out
    assert all(unicodedata.category(ch) not in ("Cf", "Cc") for ch in out if ch != "\n")
    assert all(not line.endswith(" ") for line in out.split("\n"))
    assert unicodedata.is_normalized("NFC", out)


# --------------------------------------------------------------------------- review: ignorables

#: Default-ignorable code points outside categories Cf/Cc (combining grapheme joiner, Hangul
#: conjoining fillers, Khmer inherent vowels, Mongolian free variation selectors, a TAG-block
#: variation selector supplement character).
_NON_CF_IGNORABLES = ["͏", "ᅟ", "ᅠ", "឴", "឵", "᠋", "᠏", chr(0xE0100)]


@pytest.mark.parametrize("invisible", _NON_CF_IGNORABLES)
def test_default_ignorables_outside_cf_are_stripped(invisible: str) -> None:
    assert normalize_text(f"ig{invisible}nore 결{invisible}제") == "ignore 결제"


@pytest.mark.parametrize("blank", ["ㅤ", "ﾠ", "⠀"])
def test_blank_lookalikes_become_spaces(blank: str) -> None:
    """Hangul filler "ㅤ" renders as an empty space: it must separate words like a space."""
    assert normalize_text(f"ignore{blank}previous{blank}") == "ignore previous"


def test_only_invisible_line_becomes_blank_line_keeping_numbering() -> None:
    assert normalize_text(f"a\n{ZWSP}͏{TAG_A}\nb") == "a\n\nb"


@settings(max_examples=300, deadline=None)
@given(st.text(alphabet=st.sampled_from([*_NON_CF_IGNORABLES, "ㅤ", "가", "a", " ", "\n", "\t"])))
def test_property_no_ignorable_survives_and_idempotent(text: str) -> None:
    out = normalize_text(text)
    assert not any(ch in out for ch in [*_NON_CF_IGNORABLES, "ㅤ"])
    assert normalize_text(out) == out
    assert out.count("\n") == text.count("\n")
