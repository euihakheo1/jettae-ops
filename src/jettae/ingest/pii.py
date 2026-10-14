"""Length-preserving PII masking.

Runbooks routinely contain phone numbers, e-mail addresses and server IPs. JETTAE never needs
the actual digits to judge readiness (it needs to know *that* 김대리 is reachable by phone, not
the number), so personal data is masked right after normalization — before anything is
stored, shown, or sent to an LLM.

Masking is **length-preserving**: every masked character is replaced by exactly one ``*`` and
separators are kept. Offsets, line numbers and therefore every :class:`SourceSpan` computed on
the masked text are identical to those on the unmasked text, which is what allows the original
file to be discarded (``JETTAE_STORE_RAW_DOCUMENTS=false``, the default) without breaking
evidence links.

Detection is deliberately conservative where false positives would hurt the runbook's meaning:
durations ("10분", "24시간"), dates ("2026-10-28"), versions ("v1.2.3") and HTTP codes
("5xx/503") never match, and long digit runs are treated as bank accounts only when the line
labels them as such (``계좌``).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum

__all__ = ["PiiHit", "PiiKind", "mask_pii"]

MASK_CHAR = "*"


class PiiKind(StrEnum):
    MOBILE = "mobile"  # 휴대폰 번호
    LANDLINE = "landline"  # 유선·대표 번호
    EMAIL = "email"
    IPV4 = "ipv4"
    IPV6 = "ipv6"
    RRN = "rrn"  # 주민등록번호
    CARD = "card"  # 카드 번호
    ACCOUNT = "account"  # 계좌 번호 (라벨이 있을 때만)


@dataclass(frozen=True, slots=True)
class PiiHit:
    """A masked range ``[start, end)`` of the (equally long) original and masked text."""

    kind: PiiKind
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _Candidate:
    kind: PiiKind
    start: int
    end: int
    masked: str


_Detector = Callable[[str], Iterator[_Candidate]]

# --------------------------------------------------------------------------- helpers


def _mask_digits(token: str, *, keep_head: int, keep_tail: int) -> str:
    """Mask every digit except the first ``keep_head`` and last ``keep_tail`` digits."""
    digit_positions = [i for i, ch in enumerate(token) if ch.isdigit()]
    keep = set(digit_positions[:keep_head]) | set(digit_positions[len(digit_positions) - keep_tail :])
    return "".join(MASK_CHAR if ch.isdigit() and i not in keep else ch for i, ch in enumerate(token))


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


# --------------------------------------------------------------------------- detectors


def _ascii(pattern: str) -> re.Pattern[str]:
    """Compile with ``re.ASCII`` so word/digit boundaries ignore Hangul: "10.0.0.1에서" and
    "kim@dmc.co.kr로" must still match even though a Korean particle is glued to the token."""
    return re.compile(pattern, re.ASCII)


_EMAIL = _ascii(
    r"(?<![\w.%+-])([A-Za-z0-9][A-Za-z0-9._%+-]*)@([A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)*\.[A-Za-z]{2,})(?![\w-])"
)


def _emails(text: str) -> Iterator[_Candidate]:
    """Keep the first character of the local part and the full domain ("k*****@dmc.co.kr").

    The domain identifies the organisation, not the person, and is useful evidence that the
    contact is internal.
    """
    for m in _EMAIL.finditer(text):
        local = m.group(1)
        masked = local[0] + MASK_CHAR * (len(local) - 1) + "@" + m.group(2)
        yield _Candidate(PiiKind.EMAIL, m.start(), m.end(), masked)


# 주민등록번호: YYMMDD-GNNNNNN with a plausible month/day and gender digit 1-8.
_RRN = _ascii(r"(?<![\d-])\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])-[1-8]\d{6}(?![\d-])")


def _rrns(text: str) -> Iterator[_Candidate]:
    for m in _RRN.finditer(text):
        yield _Candidate(PiiKind.RRN, m.start(), m.end(), _mask_digits(m.group(), keep_head=0, keep_tail=0))


_CARD_GROUPED = _ascii(r"(?<![\d-])\d{4}([- ])\d{4}\1\d{4}\1\d{4}(?![\d-])")
_CARD_PLAIN = _ascii(r"(?<![\d-])\d{16}(?![\d-])")


def _cards(text: str) -> Iterator[_Candidate]:
    """4-4-4-4 groups are masked as-is (the format is distinctive); a bare 16-digit run must
    also pass the Luhn check so order ids and timestamps are left alone."""
    for m in _CARD_GROUPED.finditer(text):
        yield _Candidate(PiiKind.CARD, m.start(), m.end(), _mask_digits(m.group(), keep_head=0, keep_tail=4))
    for m in _CARD_PLAIN.finditer(text):
        if _luhn_ok(m.group()):
            masked = _mask_digits(m.group(), keep_head=0, keep_tail=4)
            yield _Candidate(PiiKind.CARD, m.start(), m.end(), masked)


# A bank account is only recognised next to a 계좌 label: unlabelled long digit runs are far
# more likely to be ticket numbers, order ids or build numbers.
_ACCOUNT = _ascii(r"계좌[^\n\d]{0,20}?(?<![\d-])(\d[\d -]{8,22}\d)(?![\d-])")
_MIN_ACCOUNT_DIGITS = 10


def _accounts(text: str) -> Iterator[_Candidate]:
    for m in _ACCOUNT.finditer(text):
        token = m.group(1)
        if sum(ch.isdigit() for ch in token) < _MIN_ACCOUNT_DIGITS:
            continue
        yield _Candidate(PiiKind.ACCOUNT, m.start(1), m.end(1), _mask_digits(token, keep_head=0, keep_tail=4))


_MOBILE = _ascii(r"(?<![\d+-])01[016789]([-. ]?)\d{3,4}\1\d{4}(?![\d-])")
#: ``+82-10-…`` plus the common (technically wrong) ``+82-010-…`` / ``+82 (0)10 …`` spellings;
#: group 1 is the prefix up to the carrier code, which stays readable.
_MOBILE_INTL = _ascii(r"(?<![\d+])(\+82[- ]?(?:\(0\) ?|0)?1[016789])[-. ]?\d{3,4}[-. ]?\d{4}(?![\d-])")


def _mobiles(text: str) -> Iterator[_Candidate]:
    """Keep the carrier prefix and the last four digits: "010-****-5678"."""
    for m in _MOBILE.finditer(text):
        yield _Candidate(
            PiiKind.MOBILE, m.start(), m.end(), _mask_digits(m.group(), keep_head=3, keep_tail=4)
        )
    for m in _MOBILE_INTL.finditer(text):
        # "+82-10-1234-5678" → "+82-10-****-5678": country + carrier code and the last four.
        prefix_digits = sum(ch.isdigit() for ch in m.group(1))
        masked = _mask_digits(m.group(), keep_head=prefix_digits, keep_tail=4)
        yield _Candidate(PiiKind.MOBILE, m.start(), m.end(), masked)


# Landlines need explicit separators: an unseparated 0-prefixed digit run is too ambiguous.
# Area codes: 02 (Seoul), 0[3-6][1-5] (regions), 070 (VoIP), 050x (safe numbers).
_LANDLINE = _ascii(
    r"(?<![\d+-])(?:\(0(?:2|[3-6][1-5]|70|50\d)\) ?|0(?:2|[3-6][1-5]|70|50\d)[-.])\d{3,4}[-.]\d{4}(?![\d-])"
)


_AREA_CODE = _ascii(r"\(?(\d+)")


def _landlines(text: str) -> Iterator[_Candidate]:
    for m in _LANDLINE.finditer(text):
        token = m.group()
        area = _AREA_CODE.match(token)
        keep_head = len(area.group(1)) if area else 0
        yield _Candidate(
            PiiKind.LANDLINE, m.start(), m.end(), _mask_digits(token, keep_head=keep_head, keep_tail=4)
        )


_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
# Not preceded by a letter/digit/dot ("v1.2.3.4" is a version) and not followed by ".<digit>"
# (that would be a five-part version string), but a sentence-ending period is fine.
_IPV4 = _ascii(rf"(?<![\w.]){_OCTET}(?:\.{_OCTET}){{3}}(?!\.?\d)(?![\w])")
#: A dotted quad right after a version label is a four-part version ("롤백 대상 버전 2.14.0.1",
#: "release 1.10.2.0"), and masking it would hide the rollback target the runbook names.
_VERSION_LABEL = re.compile(
    r"(?:(?<![A-Za-z])(?:version|ver|release|rel|build|revision|rev|v)\.?|버전|릴리스|릴리즈|빌드|배포판)"
    r"(?:은|는|이|가|을|를)?\s*[:=]?\s*$",
    re.IGNORECASE,
)
_VERSION_LABEL_WINDOW = 16


def _ipv4s(text: str) -> Iterator[_Candidate]:
    """Keep the first octet (network class is useful context, the host is not).

    Dotted quads made only of single digits ("3.1.2.1 롤백 절차") are section numbers far more
    often than hosts, and the few real ones (8.8.8.8, 1.1.1.1) are public resolvers, not
    personal data — so they are left readable, like version numbers.
    """
    for m in _IPV4.finditer(text):
        token = m.group()
        if all(len(octet) == 1 for octet in token.split(".")):
            continue
        if _VERSION_LABEL.search(text, max(0, m.start() - _VERSION_LABEL_WINDOW), m.start()):
            continue
        first_dot = token.index(".")
        masked = token[:first_dot] + "".join(MASK_CHAR if ch.isdigit() else ch for ch in token[first_dot:])
        yield _Candidate(PiiKind.IPV4, m.start(), m.end(), masked)


_H16 = r"[0-9A-Fa-f]{1,4}"
_IPV6 = _ascii(
    rf"(?<![\w:])(?:(?:{_H16}:){{7}}{_H16}|(?:{_H16}:){{1,6}}(?::{_H16}){{1,6}}"
    rf"|(?:{_H16}:){{1,7}}:|::(?:{_H16}:){{0,6}}{_H16})(?![\w:])"
)


def _ipv6s(text: str) -> Iterator[_Candidate]:
    """Simple IPv6 forms (full or ``::``-compressed). A digit is required so identifiers like
    ``add::bead`` in code snippets are not mistaken for addresses; times ("10:30:00") never
    have the eight groups or the ``::`` that the pattern requires."""
    for m in _IPV6.finditer(text):
        token = m.group()
        if not any(ch.isdigit() for ch in token):
            continue
        head, sep, tail = token.partition(":")
        masked_tail = "".join(MASK_CHAR if ch != ":" else ch for ch in tail)
        yield _Candidate(PiiKind.IPV6, m.start(), m.end(), head + sep + masked_tail)


#: Earlier detectors win overlaps: an e-mail local part may contain digits, an RRN or card
#: must not be re-read as a phone number, and an IPv6 address must not be read as IPv4.
_DETECTORS: tuple[_Detector, ...] = (
    _emails,
    _rrns,
    _cards,
    _accounts,
    _mobiles,
    _landlines,
    _ipv6s,
    _ipv4s,
)


def mask_pii(text: str) -> tuple[str, list[PiiHit]]:
    """Mask personal data in ``text`` without changing its length.

    Returns the masked text and the masked ranges sorted by position. ``len(masked) ==
    len(text)`` and ``\\n`` positions are unchanged, so offsets and line numbers computed on the
    masked text are valid for the original.
    """
    taken: list[_Candidate] = []
    # One byte per character marks what is already masked, so the overlap check costs the
    # length of the token, not the number of earlier hits: a pasted log excerpt with tens of
    # thousands of IPs must stay linear instead of taking minutes.
    occupied = bytearray(len(text))
    for detector in _DETECTORS:  # tuple order is the overlap priority
        for cand in detector(text):
            if occupied.find(1, cand.start, cand.end) != -1:
                continue
            occupied[cand.start : cand.end] = b"\x01" * (cand.end - cand.start)
            taken.append(cand)
    taken.sort(key=lambda c: c.start)
    chars = list(text)
    for cand in taken:
        chars[cand.start : cand.end] = list(cand.masked)
    hits = [PiiHit(c.kind, c.start, c.end) for c in taken]
    return "".join(chars), hits
