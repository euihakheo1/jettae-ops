"""Markdown structure the extractors share: fenced code blocks and headings.

A fenced code block in a runbook holds commands, configuration or captured output — text to copy
or to recognise, not a sentence about who responds. The LLM chunker uses the fences so that a
``#`` comment inside a shell block is not taken for a heading; the assembler uses them so that a
name the document only ever writes inside such a block (a host in a command, a label in sample
output) does not become a service the response-path rules then hold to account.

Fences follow CommonMark: up to three spaces of indent, three or more backticks or tildes, closed
by a line of at least as many of the same character and nothing else. An unclosed fence runs to
the end of the document.

Headings (ATX ``#`` lines outside fences) are what a document calls itself and its sections; the
assembler reads them to tell a title that *describes a state* ("<component> is down") from a name.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Final

__all__ = ["closes_fence", "fence_opener", "fenced_ranges", "heading_texts", "in_ranges"]

#: Opening code fence; '#' lines inside fences are shell comments, not headings.
_FENCE_OPEN: Final = re.compile(r" {0,3}(?P<fence>`{3,}|~{3,})(?P<info>.*)")
#: Closing code fence: only fence characters (at least as many as the opener) and whitespace.
_FENCE_CLOSE: Final = re.compile(r" {0,3}(?P<fence>`{3,}|~{3,})[ \t]*")
#: ATX heading: up to three spaces, 1-6 '#', a space, the text, an optional closing '#' run.
_ATX_HEADING: Final = re.compile(r" {0,3}#{1,6}[ \t]+(?P<text>.*?)(?:[ \t]+#+)?[ \t]*")


def fence_opener(line: str) -> str | None:
    """The fence string a line opens (```` ``` ````/``~~~``…), or ``None``.

    A backtick "fence" whose info string contains a backtick is inline code (```` ```x``` ````),
    not a fence (CommonMark); treating it as one would hide every later heading.
    """
    match = _FENCE_OPEN.fullmatch(line)
    if match is None or (match["fence"][0] == "`" and "`" in match["info"]):
        return None
    return match["fence"]


def closes_fence(line: str, fence: str) -> bool:
    """Whether ``line`` closes ``fence``: same character, at least as long, nothing after it."""
    match = _FENCE_CLOSE.fullmatch(line)
    return match is not None and match["fence"][0] == fence[0] and len(match["fence"]) >= len(fence)


def _lines(text: str) -> Iterator[tuple[int, str]]:
    offset = 0
    for line in text.split("\n"):
        yield offset, line
        offset += len(line) + 1


def fenced_ranges(text: str) -> tuple[tuple[int, int], ...]:
    """``[start, end)`` offsets of every fenced code block, fence lines included, in order."""
    ranges: list[tuple[int, int]] = []
    fence: str | None = None
    opened = 0
    for offset, line in _lines(text):
        if fence is None:
            fence = fence_opener(line)
            opened = offset
        elif closes_fence(line, fence):
            ranges.append((opened, offset + len(line)))
            fence = None
    if fence is not None:
        ranges.append((opened, len(text)))
    return tuple(ranges)


def in_ranges(start: int, end: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    """Whether ``[start, end)`` lies entirely inside one of ``ranges``."""
    return any(lo <= start and end <= hi for lo, hi in ranges)


def heading_texts(text: str) -> tuple[str, ...]:
    """Text of every ATX heading outside fenced code blocks, in document order (``#`` marks and a
    closing ``#`` run removed). A ``# comment`` inside a shell block is not a heading."""
    headings: list[str] = []
    fence: str | None = None
    for _offset, line in _lines(text):
        if fence is not None:
            if closes_fence(line, fence):
                fence = None
            continue
        fence = fence_opener(line)
        if fence is not None:
            continue
        match = _ATX_HEADING.fullmatch(line.rstrip("\r"))
        if match is not None and match["text"].strip():
            headings.append(match["text"].strip())
    return tuple(headings)
