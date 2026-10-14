"""Structured inputs: roster (명부), on-call schedule (당직표), policy, service catalog.

These are the *trusted facts* that let the fixed rules tell "the runbook names 김대리" apart
from "김대리 is employed, reachable and on call". Because rules treat them as ground truth,
the loaders are strict about anything that would silently change a verdict:

* unknown or empty status → ``PersonStatus.UNKNOWN`` (확인 불가 ≠ 정상, P3), never ACTIVE;
* malformed dates, unknown policy keys and duplicate catalog entries raise ``ValueError``
  with a Korean message pointing at the offending row instead of being guessed around;
* every string is normalized (NFC, invisible characters removed) so "김대리" in the roster
  matches "김대리" in a runbook even if one of them was pasted from a messenger.

Headers are accepted in Korean or English and matched case/space/underscore-insensitively,
because real rosters are exported from HR tools, spreadsheets and wikis with ad-hoc headers.
Columns that are not recognised (e.g. raw phone numbers) are ignored and never stored.
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, timedelta, timezone, tzinfo
from pathlib import Path, PurePath
from typing import Any, Literal

import yaml
from pydantic import ValidationError

from jettae.domain import (
    DEFAULT_OBLIGATIONS,
    CatalogService,
    ChannelMedium,
    FatiguePolicy,
    ObligationRule,
    OnCallShift,
    PersonStatus,
    Policy,
    RosterEntry,
)
from jettae.ingest.loaders import decode_text
from jettae.ingest.normalize import normalize_text

__all__ = [
    "KST",
    "MAX_YAML_DEPTH",
    "MAX_YAML_NODES",
    "StructuredSource",
    "load_catalog",
    "load_oncall",
    "load_policy",
    "load_roster",
    "load_yaml_documents",
    "merge_catalogs",
    "parse_channel",
    "parse_status",
]

#: A path, raw bytes, or a string that is either a path (single line ending in a structured
#: file extension) or the file content itself.
StructuredSource = str | bytes | Path

#: Korea has no DST, so a fixed +09:00 offset is exact and needs no tz database (absent on
#: many Windows installs).
KST = timezone(timedelta(hours=9), "KST")

_STRUCTURED_SUFFIXES = frozenset({".csv", ".tsv", ".yaml", ".yml", ".json"})
_TOKEN_SPLIT = re.compile(r"[;,/|]")

# --------------------------------------------------------------------------- source reading


def _read_source(src: StructuredSource, what: str) -> tuple[str, str | None]:
    """Return ``(text, suffix)``; ``suffix`` is ``None`` when ``src`` was content, not a path."""
    if isinstance(src, bytes):
        return decode_text(src), None
    if isinstance(src, str) and ("\n" in src or PurePath(src).suffix.lower() not in _STRUCTURED_SUFFIXES):
        return src, None
    path = Path(src)
    if not path.is_file():
        raise FileNotFoundError(f"{what} 파일을 찾을 수 없습니다: {path.as_posix()}")
    return decode_text(path.read_bytes()), path.suffix.lower()


def _canonical(key: object) -> str:
    """Header/key form used for matching: NFC, lower-case, no spaces, ``_`` or ``-``."""
    return re.sub(r"[\s_\-]+", "", normalize_text(str(key)).lower())


def _clean(value: object) -> str:
    return normalize_text(str(value)).strip()


_BOM = "\ufeff"


#: Bounds for one structured YAML/JSON upload. The example policy is ~40 nodes and the example
#: catalog ~400 (depth 5); a 1,000-person roster in YAML is ~13,000. Nodes are counted *expanded*:
#: an alias (``*name``) costs the full size of the subtree it points to, because every reader
#: below walks the parsed data as a tree.
MAX_YAML_NODES = 20_000
MAX_YAML_DEPTH = 50


class _YamlLimitError(yaml.MarkedYAMLError):
    """A structured file over :data:`MAX_YAML_NODES` / :data:`MAX_YAML_DEPTH` (Korean ``reason``)."""

    def __init__(self, reason: str, mark: yaml.Mark | None) -> None:
        super().__init__(problem=reason, problem_mark=mark)
        self.reason = reason


class _GuardedSafeLoader(yaml.SafeLoader):
    """``yaml.SafeLoader`` that bounds the *expanded* size and depth of what it builds.

    An alias shares one composed node, but policy normalization and the roster/catalog readers
    walk the constructed data as a tree, so ten nested aliases of ten items each are 10^n nodes
    of work for a few hundred bytes of input ("billion laughs"). Each alias is therefore charged
    the full size and height of the subtree it points to, and an alias into its own still-open
    anchor (a cycle every tree walk would follow forever) is rejected. Plain anchors such as
    docker-compose ``x-`` blocks merged with ``<<: *defaults`` keep working.
    """

    def __init__(self, stream: str) -> None:
        super().__init__(stream)
        self._expanded = 0  # expanded nodes so far in this stream (all documents)
        self._depth = 0  # nesting of the node being composed
        self._measured: dict[int, tuple[int, int]] = {}  # id(node) -> (expanded size, height)
        self._heights: list[int] = []  # tallest child so far, per node being composed

    def compose_document(self) -> yaml.Node | None:
        self._measured.clear()  # anchors never cross documents; ids of freed nodes get reused
        return super().compose_document()

    def compose_node(self, parent: yaml.Node | None, index: int) -> yaml.Node | None:
        event: yaml.Event = self.peek_event()  # type: ignore[no-untyped-call]
        mark = event.start_mark if isinstance(event.start_mark, yaml.Mark) else None
        if isinstance(event, yaml.AliasEvent):
            target = self.anchors.get(event.anchor)
            if target is not None:  # an undefined alias is left to the composer's own error
                measured = self._measured.get(id(target))
                if measured is None:
                    raise _YamlLimitError("YAML 별칭(&/*)이 자기 자신을 가리킵니다", mark)
                size, height = measured
                self._charge(size, self._depth + height, mark)
                self._note_height(height)
            return super().compose_node(parent, index)
        before = self._expanded
        self._charge(1, self._depth + 1, mark)
        self._depth += 1
        self._heights.append(0)
        try:
            node = super().compose_node(parent, index)
        finally:
            self._depth -= 1
            height = self._heights.pop() + 1
        self._measured[id(node)] = (self._expanded - before, height)
        self._note_height(height)
        return node

    def _note_height(self, height: int) -> None:
        if self._heights:
            self._heights[-1] = max(self._heights[-1], height)

    def _charge(self, nodes: int, depth: int, mark: yaml.Mark | None) -> None:
        self._expanded += nodes
        if self._expanded > MAX_YAML_NODES:
            raise _YamlLimitError(
                f"YAML 항목이 너무 많습니다(별칭 &/*로 펼친 항목 포함 최대 {MAX_YAML_NODES:,}개)", mark
            )
        if depth > MAX_YAML_DEPTH:
            raise _YamlLimitError(f"YAML 중첩이 너무 깊습니다(최대 {MAX_YAML_DEPTH}단계)", mark)


def _near(exc: yaml.YAMLError) -> str:
    mark = getattr(exc, "problem_mark", None)
    return f" ({mark.line + 1}행 {mark.column + 1}열 근처)" if mark is not None else ""


def load_yaml_documents(text: str, what: str) -> list[Any]:
    """Parse a (multi-document) YAML stream into its non-empty documents, safely.

    Same result as ``yaml.safe_load_all`` for ordinary files, but the stream is capped at
    :data:`MAX_YAML_NODES` nodes with aliases expanded and :data:`MAX_YAML_DEPTH` levels, so an
    alias bomb or a deeply nested file is rejected while it is being parsed instead of
    exhausting memory later. Raises ``ValueError`` with a Korean message naming ``what``
    ("정책", "카탈로그" …) for limit violations and syntax errors alike.
    """
    try:
        return [doc for doc in yaml.load_all(text, Loader=_GuardedSafeLoader) if doc is not None]
    except _YamlLimitError as exc:
        raise ValueError(f"{what} 파일의 {exc.reason}{_near(exc)}.") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"{what} 파일의 YAML/JSON 형식이 올바르지 않습니다{_near(exc)}.") from exc


def _json_depth(value: Any) -> int:
    """Nesting depth of parsed JSON, without recursion."""
    deepest = 0
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        deepest = max(deepest, depth)
        if isinstance(item, dict):
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
    return deepest


def _documents(text: str, what: str) -> list[Any]:
    """Parse JSON or (multi-document) YAML into its non-empty documents.

    JSON is tried first: it is what OTel/Jaeger exports are, and PyYAML rejects perfectly valid
    tab-indented JSON. Syntax errors become a Korean ``ValueError`` pointing at the line, like
    every other malformed input, instead of leaking a parser exception to the API. Both paths
    share the :data:`MAX_YAML_DEPTH` bound, which keeps the recursive readers below clear of
    Python's recursion limit.
    """
    body = text.lstrip(_BOM)
    if body.lstrip().startswith(("[", "{")):
        try:
            data = json.loads(body)
        except RecursionError as exc:
            raise ValueError(f"{what} 파일의 JSON 중첩이 너무 깊습니다(최대 {MAX_YAML_DEPTH}단계).") from exc
        except json.JSONDecodeError:
            pass  # flow-style YAML such as "{services: [...]}" is not JSON
        else:
            if _json_depth(data) > MAX_YAML_DEPTH:
                raise ValueError(f"{what} 파일의 JSON 중첩이 너무 깊습니다(최대 {MAX_YAML_DEPTH}단계).")
            return [data]
    return load_yaml_documents(body, what)


def _single_document(text: str, what: str) -> Any:
    """The one document of a roster/on-call/policy file (``None`` when empty). A second
    ``---`` document would otherwise be silently ignored or crash the parser."""
    docs = _documents(text, what)
    if len(docs) > 1:
        raise ValueError(f"{what} 파일에는 YAML 문서가 하나만 있어야 합니다 ('---' 구분 {len(docs)}개).")
    return docs[0] if docs else None


def _looks_like_yaml(text: str) -> bool:
    for line in text.splitlines():
        stripped = line.strip().lstrip(_BOM)
        if not stripped or stripped.startswith("#"):
            continue
        return stripped.startswith(("-", "[", "{")) or re.match(r"^[^,\t;]+:(\s|$)", stripped) is not None
    return False


def _is_yaml(text: str, suffix: str | None) -> bool:
    if suffix in (".csv", ".tsv"):
        return False
    if suffix in (".yaml", ".yml", ".json"):
        return True
    return _looks_like_yaml(text)


_Row = tuple[str, dict[str, Any]]  # (row label for messages, canonical field -> raw value)


def _map_keys(raw: Mapping[Any, Any], aliases: Mapping[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in raw.items():
        field = aliases.get(_canonical(key))
        if field is not None and field not in out:
            out[field] = value
    return out


def _csv_rows(text: str, aliases: Mapping[str, str], what: str) -> list[_Row]:
    """Header-mapped CSV/TSV rows. When two headers map to the same field the first one wins,
    exactly as for YAML keys.

    A row with more non-empty cells than the header has columns is rejected: it almost always
    means an unquoted delimiter inside a value ("슬랙, 전화" in a comma-separated file), and
    silently dropping the overflow would lose a contact channel and produce a false finding.
    """
    body = text.lstrip(_BOM)
    first = next((ln for ln in body.splitlines() if ln.strip()), "")
    delimiter = max((",", "\t", ";"), key=first.count)
    reader = csv.reader(io.StringIO(body), delimiter=delimiter)
    header: list[str] | None = None
    rows: list[_Row] = []
    for cells in reader:
        if not any(cell.strip() for cell in cells):
            continue
        if header is None:
            header = [aliases.get(_canonical(cell), "") for cell in cells]
            continue
        if any(cell.strip() for cell in cells[len(header) :]):
            raise ValueError(
                f"{what} {reader.line_num}행: 값이 머리글 열 수({len(header)}개)보다 많습니다. "
                f"구분 문자('{delimiter}')가 들어간 값은 큰따옴표로 감싸 주세요."
            )
        record: dict[str, Any] = {}
        for field, cell in zip(header, cells, strict=False):
            if field and field not in record:
                record[field] = cell
        rows.append((f"{reader.line_num}행", record))
    return rows


def _yaml_rows(text: str, aliases: Mapping[str, str], list_keys: Iterable[str], what: str) -> list[_Row]:
    data = _single_document(text, what)
    if isinstance(data, Mapping):
        keys = tuple(list_keys)
        wanted = {_canonical(k) for k in keys}
        found = [v for k, v in data.items() if _canonical(k) in wanted]
        if not found:
            raise ValueError(f"{what} YAML에서 목록 키({', '.join(keys)})를 찾을 수 없습니다.")
        data = found[0]
    if data is None:
        return []
    if not isinstance(data, list) or not all(isinstance(item, Mapping) for item in data):
        raise ValueError(f"{what} YAML은 항목 목록이어야 합니다 (예: '- name: 김대리').")
    return [(f"{i}번째 항목", _map_keys(item, aliases)) for i, item in enumerate(data, start=1)]


def _rows(
    src: StructuredSource, aliases: Mapping[str, str], list_keys: Iterable[str], what: str
) -> list[_Row]:
    text, suffix = _read_source(src, what)
    if _is_yaml(text, suffix):
        return _yaml_rows(text, aliases, list_keys, what)
    return _csv_rows(text, aliases, what)


def _alias_table(spec: Mapping[str, Sequence[str]]) -> dict[str, str]:
    return {_canonical(alias): field for field, names in spec.items() for alias in names}


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime | date):
        return value.isoformat()
    cleaned = _clean(value)
    return cleaned or None


def _tokens(value: Any) -> list[str]:
    """Split ``"a; b, c"`` or a YAML list into unique, order-preserving tokens."""
    if value is None:
        return []
    parts = value if isinstance(value, list | tuple) else _TOKEN_SPLIT.split(str(value))
    out: list[str] = []
    for part in parts:
        token = _clean(part)
        if token and token not in out:
            out.append(token)
    return out


# --------------------------------------------------------------------------- roster

_ROSTER_ALIASES = _alias_table(
    {
        "name": ("이름", "성명", "name", "full_name", "person"),
        "status": ("상태", "재직상태", "재직여부", "status"),
        "rank": ("직급", "rank", "title"),
        "position": ("직책", "position", "job_title"),
        "team": ("팀", "소속", "부서", "소속팀", "team", "department", "dept"),
        "roles": ("역할", "roles", "role"),
        "aliases": ("별칭", "별명", "aliases", "alias", "nickname"),
        "channels": ("연락수단", "연락채널", "연락방법", "channels", "channel"),
        "left_on": ("퇴사일", "퇴직일", "left_on", "left_at", "left"),
    }
)

_STATUS_WORDS: dict[str, PersonStatus] = {
    **dict.fromkeys(("재직", "재직중", "근무", "근무중", "active", "employed"), PersonStatus.ACTIVE),
    **dict.fromkeys(("퇴사", "퇴직", "departed", "left", "resigned", "terminated"), PersonStatus.DEPARTED),
    **dict.fromkeys(
        ("휴직", "휴직중", "휴가", "휴가중", "장기부재", "육아휴직", "병가", "onleave", "leave"),
        PersonStatus.ON_LEAVE,
    ),
}

_CHANNEL_WORDS: dict[str, ChannelMedium] = {
    **dict.fromkeys(("슬랙", "slack"), ChannelMedium.SLACK),
    **dict.fromkeys(("카톡", "카카오톡", "카카오", "kakaotalk", "kakao"), ChannelMedium.KAKAOTALK),
    **dict.fromkeys(("전화", "휴대폰", "핸드폰", "휴대전화", "phone", "mobile", "call"), ChannelMedium.PHONE),
    **dict.fromkeys(("문자", "문자메시지", "sms"), ChannelMedium.SMS),
    **dict.fromkeys(("메일", "이메일", "email", "mail"), ChannelMedium.EMAIL),
    **dict.fromkeys(
        ("pagerduty", "opsgenie", "incident.io", "호출기", "pager", "페이저", "페이저듀티"),
        ChannelMedium.PAGER,
    ),
    **dict.fromkeys(("팀즈", "teams", "msteams", "microsoftteams"), ChannelMedium.TEAMS),
    **dict.fromkeys(("사내메신저", "internalmessenger"), ChannelMedium.INTERNAL_MESSENGER),
}


def parse_status(value: str | None) -> PersonStatus:
    """Map a roster status cell to :class:`PersonStatus`. Anything unrecognised — including an
    empty cell or "퇴사 예정" — is ``UNKNOWN``: a guess of ACTIVE would hide a broken path."""
    if not value:
        return PersonStatus.UNKNOWN
    return _STATUS_WORDS.get(_canonical(value), PersonStatus.UNKNOWN)


def parse_channel(value: str) -> ChannelMedium:
    """Map a channel name (Korean or English) to :class:`ChannelMedium`; unknown → ``OTHER``."""
    return _CHANNEL_WORDS.get(_canonical(value), ChannelMedium.OTHER)


#: A calendar date; a trailing time ("2026-09-30 00:00:00", how spreadsheets export date
#: cells) is accepted and dropped because 퇴사일 is a day, not an instant.
_DATE = re.compile(r"^(\d{4})[-./](\d{1,2})[-./](\d{1,2})\.?(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?$")


def _parse_date(value: Any, label: str, what: str) -> str | None:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = _text(value)
    if text is None:
        return None
    match = _DATE.match(text)
    try:
        if match is None:
            raise ValueError(text)
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3))).isoformat()
    except ValueError as exc:
        raise ValueError(f"{what} {label}: 날짜 형식을 알 수 없습니다: '{text}' (예: 2026-09-30)") from exc


def _roster_entry(label: str, record: Mapping[str, Any]) -> RosterEntry | None:
    name = _text(record.get("name"))
    if name is None:
        if any(_text(v) for v in record.values()):
            raise ValueError(f"명부 {label}: 이름이 비어 있습니다.")
        return None
    rank = _text(record.get("rank"))
    position = _text(record.get("position"))
    roles = _tokens(record.get("roles"))
    if position is not None and position not in roles:
        roles.append(position)  # 직책(팀장, CTO …) is how runbooks address people
    return RosterEntry(
        name=name,
        status=parse_status(_text(record.get("status"))),
        title=rank or position,
        team=_text(record.get("team")),
        roles=tuple(roles),
        aliases=tuple(alias for alias in _tokens(record.get("aliases")) if alias != name),
        channels=tuple(dict.fromkeys(parse_channel(t) for t in _tokens(record.get("channels")))),
        left_on=_parse_date(record.get("left_on"), label, "명부"),
    )


def load_roster(src: StructuredSource) -> tuple[RosterEntry, ...]:
    """Load a roster from CSV/TSV or YAML (a list, or a mapping with ``roster:``/``people:``).

    Recognised headers: 이름|name, 상태|status, 직급|title, 직책|position, 팀|team, 역할|roles,
    별칭|aliases, 연락수단|channels, 퇴사일|left_on. When both 직급 and 직책 are given, 직급 is the
    title and 직책 is added to roles. Entries are totally ordered (name, then the full entry, so
    namesakes too) so the same roster in a different row order produces the same graph.
    """
    rows = _rows(src, _ROSTER_ALIASES, ("roster", "people", "members", "명부", "직원"), "명부")
    if rows and not any("name" in record for _, record in rows):
        raise ValueError("명부에 이름 열(이름 또는 name)이 없습니다.")
    entries = [entry for label, record in rows if (entry := _roster_entry(label, record)) is not None]
    return tuple(sorted(entries, key=lambda e: (e.name, e.model_dump_json())))


# --------------------------------------------------------------------------- on-call

_ONCALL_ALIASES = _alias_table(
    {
        "person": ("이름", "담당자", "당직자", "person", "name"),
        "start": ("시작", "시작시각", "시작일시", "start", "from", "starts_at"),
        "end": ("종료", "종료시각", "종료일시", "end", "to", "until", "ends_at"),
        "level": ("레벨", "단계", "차수", "구분", "level"),
    }
)

_LEVEL_WORDS: dict[str, int] = {
    **dict.fromkeys(("1차", "주", "정", "primary", "main"), 1),
    **dict.fromkeys(("2차", "부", "secondary", "backup"), 2),
    "3차": 3,
}

_DATETIME = re.compile(r"^(\d{4})[-./](\d{1,2})[-./](\d{1,2})(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}))?)?$")


def _parse_datetime(value: Any, label: str, field: str, default_tz: tzinfo) -> datetime:
    """Parse ISO-8601 or ``YYYY-MM-DD HH:MM`` (``.``/``/`` separators allowed).

    Naive times are read as ``default_tz`` and every result is converted to it, so the fatigue
    rules' night-hour arithmetic sees local wall-clock hours regardless of how the file was
    written.
    """
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime(value.year, value.month, value.day)
    else:
        text = _text(value)
        if text is None:
            raise ValueError(f"당직표 {label}: {field} 시각이 비어 있습니다.")
        parsed = _parse_datetime_text(text, label, field)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=default_tz)
    return parsed.astimezone(default_tz)


def _parse_datetime_text(text: str, label: str, field: str) -> datetime:
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    match = _DATETIME.match(text)
    try:
        if match is None:
            raise ValueError(text)
        year, month, day, hour, minute, second = (int(g) if g else 0 for g in match.groups())
        return datetime(year, month, day, hour, minute, second)
    except ValueError as exc:
        raise ValueError(
            f"당직표 {label}: {field} 시각 형식을 알 수 없습니다: '{text}' (예: 2026-10-13 09:00)"
        ) from exc


def _parse_level(value: Any, label: str) -> int:
    text = _text(value)
    if text is None:
        return 1
    level = _LEVEL_WORDS.get(_canonical(text))
    if level is None and re.fullmatch(r"\d+", text):
        level = int(text)
    if level is None or level < 1:
        raise ValueError(f"당직표 {label}: 레벨을 알 수 없습니다: '{text}' (1, 2, 1차, primary …)")
    return level


def load_oncall(src: StructuredSource, *, default_tz: tzinfo = KST) -> tuple[OnCallShift, ...]:
    """Load an on-call schedule (CSV, or YAML list) with 이름|person, 시작|start, 종료|end,
    레벨|level. All datetimes are timezone-aware in ``default_tz`` (naive input is taken to be
    local time). Shifts are sorted by (start, level, person, end)."""
    shifts: list[OnCallShift] = []
    for label, record in _rows(src, _ONCALL_ALIASES, ("oncall", "shifts", "당직", "당직표"), "당직표"):
        person = _text(record.get("person"))
        if person is None:
            if any(_text(v) for v in record.values()):
                raise ValueError(f"당직표 {label}: 이름이 비어 있습니다.")
            continue
        start = _parse_datetime(record.get("start"), label, "시작", default_tz)
        end = _parse_datetime(record.get("end"), label, "종료", default_tz)
        if end <= start:
            raise ValueError(f"당직표 {label}: 종료 시각이 시작 시각보다 늦어야 합니다.")
        shifts.append(
            OnCallShift(person=person, start=start, end=end, level=_parse_level(record.get("level"), label))
        )
    return tuple(sorted(shifts, key=lambda s: (s.start, s.level, s.person, s.end)))


# --------------------------------------------------------------------------- policy


def _check_keys(data: Mapping[str, Any], allowed: Iterable[str], where: str) -> None:
    known = set(allowed)
    unknown = sorted(str(k) for k in data if k not in known)
    if unknown:
        raise ValueError(f"정책 파일의 {where}에 알 수 없는 항목이 있습니다: {', '.join(unknown)}")


def _normalized(value: Any) -> Any:
    """Policy values with every string normalized like roster and catalog strings, so a
    ``terminal_roles: [대표]`` typed on another OS still matches "대표" in the roster."""
    if isinstance(value, str):
        return _clean(value)
    if isinstance(value, Mapping):
        return {_normalized(k): _normalized(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalized(v) for v in value]
    return value


def _merge_obligations(items: Any) -> list[dict[str, Any]]:
    """User obligations override defaults *by key*; defaults not mentioned are kept.

    Statutory deadlines (KISA 24h, 개인정보위 72h) must not disappear because a policy file
    lists only a custom SLA notice. To opt out of one, set ``applies: false`` for its key.
    Unknown fields raise: a typo such as ``deadline_minute: 60`` on a default obligation would
    otherwise be ignored and the statutory deadline silently kept.
    """
    if not isinstance(items, list) or not all(isinstance(i, Mapping) for i in items):
        raise ValueError("정책 파일의 obligations는 항목 목록이어야 합니다.")
    merged: dict[str, dict[str, Any]] = {o.key: o.model_dump() for o in DEFAULT_OBLIGATIONS}
    for item in items:
        key = item.get("key")
        if not key:
            raise ValueError("정책 파일의 obligations 항목에는 key가 필요합니다.")
        _check_keys(item, ObligationRule.model_fields, f"obligations '{key}'")
        merged[str(key)] = {**merged.get(str(key), {}), **item, "key": str(key)}
    return list(merged.values())


def load_policy(src: StructuredSource) -> Policy:
    """Load ``policy.yaml`` using the field names of :class:`Policy` (optionally under a
    top-level ``policy:`` key). Missing fields keep their published defaults; ``fatigue`` and
    ``tier_slo_minutes`` are merged field-by-field and ``obligations`` by key. Unknown keys
    raise — a typo must not silently fall back to a default verdict."""
    text, _suffix = _read_source(src, "정책")
    data = _normalized(_single_document(text, "정책")) or {}
    if isinstance(data, Mapping) and set(data) == {"policy"} and isinstance(data["policy"], Mapping):
        data = data["policy"]
    if not isinstance(data, Mapping):
        raise ValueError("정책 파일은 'critical_tiers: [1, 2]' 같은 키-값 형식이어야 합니다.")
    values = dict(data)
    _check_keys(values, Policy.model_fields, "최상위")
    if isinstance(values.get("fatigue"), Mapping):
        _check_keys(values["fatigue"], FatiguePolicy.model_fields, "fatigue")
    if isinstance(values.get("tier_slo_minutes"), Mapping):
        values["tier_slo_minutes"] = {**Policy().tier_slo_minutes, **values["tier_slo_minutes"]}
    if "obligations" in values:
        values["obligations"] = _merge_obligations(values["obligations"])
    try:
        return Policy.model_validate(values)
    except ValidationError as exc:
        raise ValueError(f"정책 파일 형식이 올바르지 않습니다: {exc}") from exc


# --------------------------------------------------------------------------- catalog

#: ``CatalogService.kind`` values. "channel" marks a contact medium with its own failure domain
#: (e.g. an on-premise messenger hosted in one IDC); the assembler turns it into a CHANNEL node.
_CatalogKind = Literal["service", "system", "channel"]
_CATALOG_KINDS: tuple[_CatalogKind, ...] = ("service", "system", "channel")
_TIER_KEY = "jettae.io/tier"
_LOCATION_KEY = "jettae.io/location"
_OWNER_KEY = "jettae.io/owner"
_BACKSTAGE_KINDS: dict[str, _CatalogKind] = {
    "component": "service",
    "system": "service",
    "resource": "system",
}
_ENTITY_REF = re.compile(r"^(?:[A-Za-z]+:)?(?:[^/]+/)?(?P<name>[^/]+)$")

_NATIVE_ALIASES = _alias_table(
    {
        "name": ("name", "이름", "서비스"),
        "kind": ("kind", "종류"),
        "tier": ("tier", "등급"),
        "aliases": ("aliases", "alias", "별칭"),
        "depends_on": ("depends_on", "dependencies", "의존"),
        "location": ("location", "위치", "region"),
        "owner": ("owner", "담당", "소유"),
        "description": ("description", "설명"),
    }
)


#: "1", "tier2", "Tier 1", "T1", "P1", "1등급", "등급 2". Anything else — "-1", "1.5", "high",
#: "1-2" — is rejected instead of searching for the first digit run, which turned "-1" into the
#: most critical tier 1.
_TIER_TEXT = re.compile(r"^(?:(?:tier|t|p|등급)[\s_-]*)?(\d+)\s*(?:등급|티어)?$", re.IGNORECASE)


def _parse_tier(value: Any, where: str) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, int):
        tier = value
    else:
        match = _TIER_TEXT.match(_clean(value))
        if match is None:
            raise ValueError(f"카탈로그 {where}: tier 값을 알 수 없습니다: '{value}' (1, 2, 3)")
        tier = int(match.group(1))
    if tier < 1:
        raise ValueError(f"카탈로그 {where}: tier는 1 이상이어야 합니다: {tier}")
    return tier


def _entity_name(ref: Any) -> str | None:
    """``component:default/payment`` → ``payment`` (Backstage entity reference)."""
    text = _text(ref)
    if text is None:
        return None
    match = _ENTITY_REF.match(text)
    return match.group("name") if match else text


def _service(
    name: str,
    *,
    kind: str = "service",
    tier: int | None = None,
    aliases: Iterable[str] = (),
    depends_on: Iterable[str] = (),
    location: str | None = None,
    owner: str | None = None,
    description: str | None = None,
) -> CatalogService:
    return CatalogService(
        name=name,
        kind=kind,
        tier=tier,
        aliases=tuple(sorted({a for a in aliases if a and a != name})),
        depends_on=tuple(sorted({d for d in depends_on if d and d != name})),
        location=location,
        owner=owner,
        description=description,
    )


def _unique(services: list[CatalogService]) -> list[CatalogService]:
    seen: set[str] = set()
    for svc in services:
        if svc.name in seen:
            raise ValueError(f"카탈로그에 같은 이름의 서비스가 두 번 있습니다: {svc.name}")
        seen.add(svc.name)
    return services


def _from_native(items: Any) -> list[CatalogService]:
    if not isinstance(items, list) or not all(isinstance(i, Mapping) for i in items):
        raise ValueError("카탈로그의 services는 항목 목록이어야 합니다.")
    services: list[CatalogService] = []
    for index, raw in enumerate(items, start=1):
        unknown = sorted(str(k) for k in raw if _canonical(k) not in _NATIVE_ALIASES)
        if unknown:
            raise ValueError(
                f"카탈로그 {index}번째 서비스에 알 수 없는 항목이 있습니다: {', '.join(unknown)}"
            )
        record = _map_keys(raw, _NATIVE_ALIASES)
        name = _text(record.get("name"))
        if name is None:
            raise ValueError(f"카탈로그 {index}번째 서비스에 name이 없습니다.")
        kind = (_text(record.get("kind")) or "service").lower()
        if kind not in _CATALOG_KINDS:
            raise ValueError(
                f"카탈로그 서비스 '{name}': kind는 {', '.join(_CATALOG_KINDS)} 중 하나여야 합니다: '{kind}'"
            )
        services.append(
            _service(
                name,
                kind=kind,
                tier=_parse_tier(record.get("tier"), f"서비스 '{name}'"),
                aliases=_tokens(record.get("aliases")),
                depends_on=_tokens(record.get("depends_on")),
                location=_text(record.get("location")),
                owner=_text(record.get("owner")),
                description=_text(record.get("description")),
            )
        )
    return _unique(services)


def _is_backstage(doc: Any) -> bool:
    return isinstance(doc, Mapping) and str(doc.get("apiVersion", "")).startswith("backstage.io/")


def _mapping(value: Any) -> Mapping[Any, Any]:
    """A YAML sub-object, or an empty one when the entity leaves it out or malforms it."""
    return value if isinstance(value, Mapping) else {}


def _ref_list(value: Any) -> list[Any]:
    """``dependsOn`` as a list: a lone ``dependsOn: component:ledger`` must not be iterated
    character by character into dependencies on "c", "o", "m" …"""
    if value is None:
        return []
    return list(value) if isinstance(value, list | tuple) else [value]


def _from_backstage(docs: list[Any]) -> list[CatalogService]:
    services: list[CatalogService] = []
    for doc in docs:
        if not _is_backstage(doc):
            continue
        kind = _BACKSTAGE_KINDS.get(str(doc.get("kind", "")).lower())
        metadata = _mapping(doc.get("metadata"))
        spec = _mapping(doc.get("spec"))
        name = _text(metadata.get("name"))
        if kind is None or name is None:
            continue
        annotations = _mapping(metadata.get("annotations"))
        title = _text(metadata.get("title"))
        depends = [n for ref in _ref_list(spec.get("dependsOn")) if (n := _entity_name(ref))]
        services.append(
            _service(
                name,
                kind=kind,
                tier=_parse_tier(annotations.get(_TIER_KEY), f"'{name}'"),
                aliases=[title] if title else [],
                depends_on=depends,
                location=_text(annotations.get(_LOCATION_KEY)),
                owner=_entity_name(spec.get("owner")),
                description=_text(metadata.get("description")),
            )
        )
    return _unique(services)


def _compose_labels(labels: Any) -> dict[str, str]:
    if isinstance(labels, Mapping):
        return {str(k): str(v) for k, v in labels.items()}
    if isinstance(labels, list):
        pairs = (str(item).split("=", 1) for item in labels)
        return {p[0]: p[1] for p in pairs if len(p) == 2}
    return {}


def _from_compose(services_map: Mapping[Any, Any]) -> list[CatalogService]:
    services: list[CatalogService] = []
    for raw_name, raw_spec in services_map.items():
        name = _clean(raw_name)
        spec = raw_spec if isinstance(raw_spec, Mapping) else {}
        depends = spec.get("depends_on") or []
        dep_names = [_clean(d) for d in (depends.keys() if isinstance(depends, Mapping) else depends)]
        labels = _compose_labels(spec.get("labels"))
        services.append(
            _service(
                name,
                tier=_parse_tier(labels.get(_TIER_KEY), f"'{name}'"),
                depends_on=dep_names,
                location=_text(labels.get(_LOCATION_KEY)),
                owner=_text(labels.get(_OWNER_KEY)),
            )
        )
    return services


def _jaeger_links(doc: Any) -> list[Mapping[str, Any]] | None:
    links = doc.get("data") if isinstance(doc, Mapping) else doc
    if (
        isinstance(links, list)
        and links
        and all(isinstance(link, Mapping) and "parent" in link and "child" in link for link in links)
    ):
        return links
    return None


def _observed_calls(count: Any, index: int) -> bool:
    """Whether a dependency link carries traffic. A missing count means the exporter does not
    record counts (the link itself is the observation); numeric strings and floats are
    accepted because some exporters quote or scale them."""
    if count is None:
        return True
    try:
        return float(count) > 0
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"카탈로그 의존성 {index}번째 항목: callCount 값을 알 수 없습니다: '{count}'"
        ) from exc


def _from_jaeger(links: list[Mapping[str, Any]]) -> list[CatalogService]:
    """Every endpoint of an observed call is a real service; ``parent`` depends on ``child``.
    Links with ``callCount: 0`` record the services but not a dependency. A link with an empty
    endpoint is rejected rather than turned into a service named "" or "None"."""
    deps: dict[str, set[str]] = {}
    for index, link in enumerate(links, start=1):
        parent, child = _text(link["parent"]), _text(link["child"])
        if parent is None or child is None:
            raise ValueError(f"카탈로그 의존성 {index}번째 항목: parent/child 서비스 이름이 비어 있습니다.")
        deps.setdefault(parent, set())
        deps.setdefault(child, set())
        if _observed_calls(link.get("callCount"), index):
            deps[parent].add(child)
    return [_service(name, depends_on=children) for name, children in deps.items()]


def load_catalog(src: StructuredSource) -> tuple[CatalogService, ...]:
    """Load a service catalog, auto-detecting the format:

    (a) native YAML ``{services: [{name, kind, tier, aliases, depends_on, location, owner,
    description}]}`` with kind ``service``/``system``/``channel``; (b) Backstage
    ``catalog-info.yaml`` (multi-document; Component/System → service, Resource → system; tier
    from the ``jettae.io/tier`` annotation); (c)
    ``docker-compose.yml`` (``depends_on`` list or mapping; optional ``jettae.io/*`` labels);
    (d) Jaeger/OTel service-graph JSON ``[{parent, child, callCount}]``. Sorted by name.
    """
    text, _suffix = _read_source(src, "카탈로그")
    docs = _documents(text, "카탈로그")
    if not docs:
        raise ValueError("카탈로그가 비어 있습니다.")
    links = _jaeger_links(docs[0]) if len(docs) == 1 else None
    if links is not None:
        services = _from_jaeger(links)
    elif any(_is_backstage(doc) for doc in docs):
        services = _from_backstage(docs)
    elif len(docs) == 1 and isinstance(docs[0], Mapping) and isinstance(docs[0].get("services"), list):
        services = _from_native(docs[0]["services"])
    elif len(docs) == 1 and isinstance(docs[0], Mapping) and isinstance(docs[0].get("services"), Mapping):
        services = _from_compose(docs[0]["services"])
    else:
        raise ValueError(
            "카탈로그 형식을 알 수 없습니다. 지원 형식: services 목록 YAML, Backstage catalog-info.yaml, "
            "docker-compose.yml, Jaeger 의존성 JSON([{parent, child, callCount}])"
        )
    return tuple(sorted(services, key=lambda s: s.name))


def merge_catalogs(*catalogs: Iterable[CatalogService]) -> tuple[CatalogService, ...]:
    """Merge catalogs from several sources by service name.

    Earlier catalogs take precedence for scalar fields (kind, tier, location, owner,
    description); later ones only fill fields that are still unknown. Aliases and dependencies
    are unioned — e.g. Backstage ownership plus OTel-observed dependencies.
    """
    merged: dict[str, CatalogService] = {}
    for catalog in catalogs:
        for svc in catalog:
            cur = merged.get(svc.name)
            if cur is None:
                merged[svc.name] = svc
                continue
            merged[svc.name] = _service(
                svc.name,
                kind=cur.kind,
                tier=cur.tier if cur.tier is not None else svc.tier,
                aliases=(*cur.aliases, *svc.aliases),
                depends_on=(*cur.depends_on, *svc.depends_on),
                location=cur.location or svc.location,
                owner=cur.owner or svc.owner,
                description=cur.description or svc.description,
            )
    return tuple(merged[name] for name in sorted(merged))
