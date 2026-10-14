"""Persistence ports beyond the domain ones, plus the :class:`Store` aggregate.

``jettae.domain.ports`` holds only what the pure core (improve / orchestrator) needs to talk
about graph versions, patches, rehearsals, reports and the audit log. Delivery concerns —
workspaces, uploaded inputs, workspace access keys, landing-page sign-ups and demand events —
live here so the domain stays free of them.

Both backends (:mod:`jettae.store.memory` and :mod:`jettae.store.sql`) share every rule that
decides *what* gets stored and *in which order* it comes back: validation, timestamp
normalisation, deterministic ids and sort keys are all defined once in this module. That is
what lets one contract suite run unchanged against both, and what keeps list ordering
byte-identical between SQLite and Postgres (whose text collation is locale-dependent, so we
never rely on ``ORDER BY`` over strings).
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from pydantic_core import to_jsonable_python

from jettae.domain import (
    CatalogService,
    DiagnosisReport,
    Document,
    DocumentRef,
    GraphVersion,
    InputBundle,
    OnCallShift,
    OpsGraph,
    Patch,
    Policy,
    RehearsalResult,
    RosterEntry,
    Scenario,
    ports,
)
from jettae.domain.ports import (
    AuditLog,
    GraphRepository,
    PatchRepository,
    RehearsalRepository,
    ReportRepository,
    VersionConflictError,
)

#: A clock returns the current instant as a timezone-aware datetime. It is injected so
#: timestamps are reproducible in tests and demos; it only stamps metadata, never judgement.
Clock = Callable[[], datetime]

JsonObject = dict[str, Any]

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_WORKSPACE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_EVENT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}")
#: an access key is stored only as its SHA-256 digest (lowercase hex)
_KEY_DIGEST = re.compile(r"[0-9a-f]{64}")
_EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
_MAX_EMAIL = 254
_MAX_WORKSPACE_NAME = 200
#: Hangul filler code points render as nothing, yet ``str.isspace`` is False for them; Korean
#: users paste U+3164 precisely to get a "blank" name, so they count as invisible.
_HANGUL_FILLERS = frozenset("\u115f\u1160\u3164\uffa0")
#: Unicode categories that draw no glyph: controls, format characters (zero-width space,
#: joiners, BOM), separators and lone surrogates.
_INVISIBLE_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Zs", "Zl", "Zp"})


def utc_now() -> datetime:
    """Default :data:`Clock` (wall clock). Inject a fixed clock wherever reproducibility matters."""
    return datetime.now(UTC)


# --------------------------------------------------------------------------- errors


class NotFoundError(ports.NotFoundError):
    """A record addressed by id does not exist.

    Subclasses the domain port's error so ``improve``/``orchestrator`` can catch
    ``jettae.domain.ports.NotFoundError`` without knowing which backend raised it (and,
    through it, :class:`KeyError` for mapping-style callers). It only adds a readable
    ``str()``: ``KeyError`` would wrap the Korean message in quotes.
    """

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


class AlreadyExistsError(ValueError):
    """An explicitly chosen id is already taken."""


class StoreIntegrityError(RuntimeError):
    """Stored content no longer matches its content address (corruption or tampering)."""


# --------------------------------------------------------------------------- records


class Workspace(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    created_at: datetime


class StoredDocument(BaseModel):
    """Metadata of an uploaded document. Kept even when the raw text is not stored.

    ``raw_stored`` is False under the default privacy setting (``JETTAE_STORE_RAW_DOCUMENTS=false``,
    ``Settings.store_raw_documents``): the evidence quotes inside the graph remain, the full text
    does not.
    """

    model_config = ConfigDict(frozen=True)

    workspace_id: str
    doc_id: str
    title: str
    sha256: str
    media_type: str
    path: str | None = None
    quarantined: tuple[tuple[int, int], ...] = ()
    raw_stored: bool

    def to_ref(self) -> DocumentRef:
        return DocumentRef(doc_id=self.doc_id, title=self.title, sha256=self.sha256, path=self.path)


class StoredInputs(BaseModel):
    """Trusted structured inputs of a workspace (everything in an :class:`InputBundle` but text).

    Empty tuples mean "not provided", exactly as in :class:`InputBundle`; the store never fills
    in facts that were not uploaded.
    """

    model_config = ConfigDict(frozen=True)

    roster: tuple[RosterEntry, ...] = ()
    oncall: tuple[OnCallShift, ...] = ()
    catalog: tuple[CatalogService, ...] = ()
    policy: Policy = Field(default_factory=Policy)

    @classmethod
    def from_bundle(cls, bundle: InputBundle) -> StoredInputs:
        return cls(roster=bundle.roster, oncall=bundle.oncall, catalog=bundle.catalog, policy=bundle.policy)

    def to_bundle(self, documents: Sequence[Document]) -> InputBundle:
        return InputBundle(
            documents=tuple(documents),
            roster=self.roster,
            oncall=self.oncall,
            catalog=self.catalog,
            policy=self.policy,
        )


class Signup(BaseModel):
    """A pre-launch sign-up (사전 신청).

    Personal data is only ever accepted with privacy consent, so the model itself refuses to
    exist without it — no backend can persist a sign-up that lacks consent. E-mail addresses
    are case-folded so one person never has two records: :meth:`SignupRepository.add_signup`
    replaces the record (internal and admin use), while
    :meth:`SignupRepository.add_signup_if_absent` keeps the first one — anonymous submissions
    must not let whoever knows an address rewrite that person's consent or its date.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True, coerce_numbers_to_str=True)

    email: str = Field(max_length=254)
    organization: str | None = Field(default=None, max_length=200)
    team_size: str | None = Field(default=None, max_length=50)
    message: str | None = Field(default=None, max_length=2000)
    consent_privacy: bool
    consent_marketing: bool = False
    created_at: datetime
    source: str | None = Field(default=None, max_length=64)

    @field_validator("email")
    @classmethod
    def _normalise_email(cls, value: str) -> str:
        return normalize_email(value)

    @field_validator("organization", "team_size", "message", "source")
    @classmethod
    def _optional_text(cls, value: str | None) -> str | None:
        """A blank answer is "not provided": store None, never an empty string that would
        later read as a stated (empty) fact. Blank includes answers made only of invisible
        characters (zero-width space, Hangul filler), which look empty to every reader."""
        if value is None or not any(_is_visible(char) for char in value):
            return None
        check_text(value, "신청 정보")
        return value

    @field_validator("created_at")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return as_utc(value, "신청 시각")

    @model_validator(mode="after")
    def _requires_consent(self) -> Signup:
        if not self.consent_privacy:
            raise ValueError("개인정보 수집·이용에 동의해야 사전 신청을 저장할 수 있습니다.")
        return self


class AuditEntry(BaseModel):
    """One append-only audit record. ``seq`` is the global append order."""

    model_config = ConfigDict(frozen=True)

    seq: int
    workspace_id: str
    actor: str
    action: str
    payload: JsonObject
    recorded_at: datetime


# --------------------------------------------------------------------------- protocols


@runtime_checkable
class WorkspaceRepository(Protocol):
    def create_workspace(self, name: str, *, workspace_id: str | None = None) -> Workspace:
        """Create a workspace. Without ``workspace_id`` a deterministic id is derived."""
        ...

    def get_workspace(self, workspace_id: str) -> Workspace | None: ...

    def list_workspaces(self) -> list[Workspace]: ...

    def delete_workspace(self, workspace_id: str) -> bool:
        """Delete the workspace and everything it owns, in every repository.

        Removed: the workspace record, its documents, inputs, graph versions, patches, scenarios,
        rehearsal results, reports, access key and audit entries — plus every graph blob that no
        other workspace's version still references (blobs hold names and evidence quotes too).
        Sign-ups and demand events are not workspace data and stay. Records keyed by the id
        without a workspace record (one-shot CLI scans) are removed as well.

        Returns True when anything was deleted, False when nothing was stored under the id (so a
        repeated call is a harmless no-op). Afterwards every read for the id answers "absent"
        (None, empty, or :class:`NotFoundError`), and the id may be used again from scratch.
        """
        ...


@runtime_checkable
class DocumentRepository(Protocol):
    def save_documents(self, workspace_id: str, docs: Sequence[Document], *, store_raw: bool) -> None:
        """Upsert documents by ``doc_id``. Text is kept only when ``store_raw`` is True; saving
        again with ``store_raw=False`` erases previously stored text."""
        ...

    def list_documents(self, workspace_id: str) -> list[StoredDocument]: ...

    def get_document(self, workspace_id: str, doc_id: str) -> Document | None:
        """The full document, or None when it is unknown *or its text was not stored*.

        ``Document.text`` is mandatory, and fabricating an empty text would silently change
        ``sha256`` — so absent text stays absent. Use :meth:`list_documents` for metadata."""
        ...


@runtime_checkable
class InputRepository(Protocol):
    def save_inputs(self, workspace_id: str, inputs: StoredInputs) -> None: ...

    def load_inputs(self, workspace_id: str) -> StoredInputs:
        """Last saved inputs, or empty :class:`StoredInputs` (= nothing provided)."""
        ...


@runtime_checkable
class SignupRepository(Protocol):
    def add_signup(self, signup: Signup) -> None:
        """Upsert by (normalised) e-mail; the latest submission wins.

        For internal and admin use only: an anonymous caller must go through
        :meth:`add_signup_if_absent`, or anyone who knows an address could rewrite that person's
        record — marketing consent and consent date included.
        """
        ...

    def add_signup_if_absent(self, signup: Signup) -> bool:
        """Insert ``signup`` only when its (normalised) e-mail is not stored yet.

        True when it was inserted; False when the person already has a record, which is then left
        exactly as it was (``created_at``, both consents and every answer). Atomic: of two
        concurrent submissions for one address exactly one is inserted.
        """
        ...

    def delete_signup(self, email: str) -> bool:
        """Delete the person's record (withdrawal of consent). True when one existed.

        ``email`` is normalised like on insert; a malformed address matches nobody (False)."""
        ...

    def delete_signups_before(self, cutoff: datetime) -> int:
        """Delete every sign-up whose ``created_at`` is strictly before ``cutoff`` (retention).

        ``cutoff`` must be timezone-aware (``ValueError`` otherwise). Returns how many were deleted.
        """
        ...

    def get_signup(self, email: str) -> Signup | None: ...

    def count_signups(self) -> int:
        """Number of distinct people who signed up."""
        ...

    def list_signups(self) -> list[Signup]:
        """Every sign-up (the latest submission per person), by creation time, then e-mail."""
        ...


@runtime_checkable
class EventRepository(Protocol):
    def record_event(self, name: str, props: Mapping[str, object], ts: datetime) -> None: ...

    def event_counts(self) -> dict[str, int]:
        """Occurrences per event name, keys sorted."""
        ...


@runtime_checkable
class WorkspaceAccessRepository(Protocol):
    """Access keys of private workspaces (공개 베타의 무료 검진 결과).

    Workspace ids are identifiers, not secrets (see :func:`derive_workspace_id`), so a workspace
    that anonymous visitors created is opened by a separate key. Only the key's SHA-256 digest is
    ever stored: the plaintext exists in the response that granted it and in the visitor's hands.
    """

    def set_access_key_digest(self, workspace_id: str, digest: str) -> None:
        """Store the digest of the workspace's key, replacing a previous one (64 lowercase hex)."""
        ...

    def access_key_digest(self, workspace_id: str) -> str | None:
        """The stored digest, or None when the workspace has no access key."""
        ...


@runtime_checkable
class AuditRepository(AuditLog, Protocol):
    """:class:`AuditLog` plus read access (audit views, tests)."""

    def entries(self, workspace_id: str) -> list[AuditEntry]:
        """Entries of one workspace in append order."""
        ...


@runtime_checkable
class Store(Protocol):
    """Everything persistence offers, as one object handed to the orchestrator."""

    @property
    def graphs(self) -> GraphRepository: ...

    @property
    def patches(self) -> PatchRepository: ...

    @property
    def rehearsals(self) -> RehearsalRepository: ...

    @property
    def reports(self) -> ReportRepository: ...

    @property
    def audit(self) -> AuditRepository: ...

    @property
    def workspaces(self) -> WorkspaceRepository: ...

    @property
    def documents(self) -> DocumentRepository: ...

    @property
    def inputs(self) -> InputRepository: ...

    @property
    def signups(self) -> SignupRepository: ...

    @property
    def events(self) -> EventRepository: ...

    @property
    def access(self) -> WorkspaceAccessRepository: ...

    def close(self) -> None: ...


# --------------------------------------------------------------------------- shared rules


def as_utc(value: datetime, what: str) -> datetime:
    """Normalise an aware datetime to UTC; reject naive ones.

    A naive timestamp would be interpreted in the server's local zone, so the same input
    could sort differently on two machines (P6). Refusing it is the only safe choice.
    """
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{what}에는 시간대 정보가 있어야 합니다(예: UTC).")
    return value.astimezone(UTC)


def check_text(value: str, what: str) -> None:
    """Refuse text that a supported backend cannot store, so all backends agree.

    Postgres text columns cannot hold NUL, and no UTF-8 database can hold a lone surrogate —
    which a Python ``str`` can (file names decoded with ``surrogateescape``, broken pastes).
    Without this check the in-memory store and SQLite kept such records while Postgres (or the
    SQLite driver, with a raw ``UnicodeEncodeError``) refused them.
    """
    if "\x00" in value:
        raise ValueError(f"{what}에 NUL 문자(\\x00)를 저장할 수 없습니다.")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{what}에 UTF-8로 저장할 수 없는 문자(짝 없는 서로게이트)가 있습니다.") from exc


def check_workspace_key(workspace_id: str) -> None:
    """Workspace ids partition every table, so they obey the same storability rule as text."""
    check_text(workspace_id, "워크스페이스 id")


def normalize_email(email: str) -> str:
    """Canonical e-mail key: format characters dropped, NFKC, case-folded, at most 254 characters.

    NFKC folds full-width input from Korean IMEs (``ｋｉｍ＠ｅｘａｍｐｌｅ．ｃｏｍ``) and NFD/NFC
    variants of IDN domains into one key, so the same person never gets two records. Format
    characters (zero-width space/joiner, BOM, soft hyphen, bidi overrides) are dropped first:
    they ride along invisibly with addresses pasted from messengers or CSV exports and would
    otherwise split one person into several records. Any other invisible character (controls,
    Hangul fillers, lone surrogates) makes the address invalid instead of being guessed away.
    The length is checked *after* folding because case folding can lengthen a string (``ß`` ->
    ``ss``) past the column width, which Postgres would reject while SQLite would not.
    """
    unformatted = "".join(char for char in email if unicodedata.category(char) != "Cf")
    clean = unicodedata.normalize("NFKC", unformatted).casefold().strip()
    if _EMAIL.fullmatch(clean) is None or not all(_is_visible(char) for char in clean):
        raise ValueError("이메일 주소 형식이 올바르지 않습니다.")
    if len(clean) > _MAX_EMAIL:
        raise ValueError(f"이메일 주소는 {_MAX_EMAIL}자 이하여야 합니다.")
    return clean


def deletion_cutoff(cutoff: datetime) -> datetime:
    """UTC form of a retention cutoff ("delete what is older than this"); naive values are refused,
    since the same cutoff would otherwise delete different records on differently zoned hosts."""
    return as_utc(cutoff, "삭제 기준 시각")


def signup_key(email: str) -> str | None:
    """Lookup key for :meth:`SignupRepository.get_signup`; None when the address is malformed.

    A malformed address can never have been stored, so the honest answer to "who signed up
    with it" is "nobody" — not an error for what is only a query.
    """
    try:
        return normalize_email(email)
    except ValueError:
        return None


def _is_visible(char: str) -> bool:
    return char not in _HANGUL_FILLERS and unicodedata.category(char) not in _INVISIBLE_CATEGORIES


def _is_padding(char: str) -> bool:
    """Characters trimmed from name edges: whitespace plus what renders as nothing."""
    return char.isspace() or char in _HANGUL_FILLERS or unicodedata.category(char) == "Cf"


def validate_workspace_name(name: str) -> str:
    """Trimmed, NFC-normalised, single-line name with at least one visible character.

    NFC matters for Korean: the same name typed on macOS (NFD jamo) and on Windows (NFC
    syllables) must be one name — and derive one id — rather than two lookalikes. For the same
    reason invisible padding (zero-width space, BOM, Hangul filler) is trimmed like whitespace.
    Control characters are refused rather than trimmed: names are one-line labels in reports,
    SARIF and terminals, where a newline or an ANSI escape would restructure the output.
    """
    normalized = unicodedata.normalize("NFC", name)
    check_text(normalized, "워크스페이스 이름")
    start, end = 0, len(normalized)
    while start < end and _is_padding(normalized[start]):
        start += 1
    while end > start and _is_padding(normalized[end - 1]):
        end -= 1
    clean = normalized[start:end]
    if not clean:
        raise ValueError("워크스페이스 이름이 비어 있습니다.")
    if any(unicodedata.category(char) == "Cc" for char in clean):
        raise ValueError("워크스페이스 이름에는 줄바꿈·탭 같은 제어 문자를 쓸 수 없습니다.")
    if len(clean) > _MAX_WORKSPACE_NAME:
        raise ValueError(f"워크스페이스 이름은 {_MAX_WORKSPACE_NAME}자 이하여야 합니다.")
    return clean


def validate_workspace_id(workspace_id: str) -> str:
    """Workspace ids appear in REST paths, so only URL-safe characters are accepted."""
    if _WORKSPACE_ID.fullmatch(workspace_id) is None:
        raise ValueError(
            "워크스페이스 id는 영문·숫자로 시작하고 영문·숫자·'_'·'.'·'-'만 쓸 수 있으며 "
            "64자 이하여야 합니다."
        )
    return workspace_id


def derive_workspace_id(name: str, created_at: datetime, attempt: int) -> str:
    """Deterministic workspace id (P6): same name + clock + attempt -> same id.

    Ids are identifiers, not secrets; authorisation must never rely on them being unguessable.
    ``attempt`` disambiguates workspaces created with the same name at the same instant.
    """
    digest = hashlib.sha256(f"{name}\x00{created_at.isoformat()}\x00{attempt}".encode()).hexdigest()
    return f"ws-{digest[:16]}"


def create_workspace_with(
    insert: Callable[[Workspace], bool], clock: Clock, name: str, workspace_id: str | None
) -> Workspace:
    """Shared create flow. ``insert`` returns False when the id is already taken."""
    clean = validate_workspace_name(name)
    created_at = as_utc(clock(), "clock()의 반환값")
    if workspace_id is not None:
        workspace = Workspace(id=validate_workspace_id(workspace_id), name=clean, created_at=created_at)
        if not insert(workspace):
            raise AlreadyExistsError(f"워크스페이스 id '{workspace_id}'는 이미 사용 중입니다.")
        return workspace
    attempt = 0
    while True:
        workspace = Workspace(
            id=derive_workspace_id(clean, created_at, attempt), name=clean, created_at=created_at
        )
        if insert(workspace):
            return workspace
        attempt += 1


def check_documents(workspace_id: str, docs: Iterable[Document]) -> tuple[Document, ...]:
    """Validate one save request as a whole and return it materialised.

    Every document is checked before any is stored, so a batch is saved entirely or not at
    all on every backend. Returning a tuple means callers iterate a snapshot: a one-shot
    iterable would otherwise be exhausted by validation and silently save nothing.
    """
    check_workspace_key(workspace_id)
    batch = tuple(docs)
    seen: set[str] = set()
    for doc in batch:
        if doc.doc_id in seen:
            raise ValueError(f"같은 문서 id({doc.doc_id})가 한 번의 저장 요청에 두 번 들어 있습니다.")
        seen.add(doc.doc_id)
        for value, what in (
            (doc.doc_id, "문서 id"),
            (doc.text, "문서 본문"),
            (doc.title, "문서 제목"),
            (doc.media_type, "문서 형식"),
            (doc.path or "", "문서 경로"),
        ):
            check_text(value, what)
    return batch


def stored_document(workspace_id: str, doc: Document, *, raw_stored: bool) -> StoredDocument:
    return StoredDocument(
        workspace_id=workspace_id,
        doc_id=doc.doc_id,
        title=doc.title,
        sha256=doc.sha256,
        media_type=doc.media_type,
        path=doc.path,
        quarantined=doc.quarantined,
        raw_stored=raw_stored,
    )


def restore_document(meta: StoredDocument, text: str) -> Document:
    """Rebuild a stored document, verifying its text still has the recorded SHA-256.

    Evidence spans are offsets into exactly this text, so serving altered text would silently
    point every quote at different words (P2). Like a graph blob whose content no longer
    matches its address, it is reported as corruption instead.
    """
    doc = Document(
        doc_id=meta.doc_id,
        title=meta.title,
        text=text,
        media_type=meta.media_type,
        path=meta.path,
        quarantined=meta.quarantined,
    )
    if doc.sha256 != meta.sha256:
        raise StoreIntegrityError(
            f"저장된 문서 {meta.doc_id}의 본문이 기록된 해시와 일치하지 않습니다(손상 의심)."
        )
    return doc


def prepare_append(
    version: GraphVersion, graph: OpsGraph, expected_head: int | None
) -> tuple[GraphVersion, str]:
    """Validate an append request; return the UTC-normalised version and the graph blob.

    Content addressing is only trustworthy if the address is checked on the way in, so a
    version whose ``graph_hash`` does not match ``graph`` is refused, as is a version number
    that does not directly follow the expected head (history has no gaps). The lineage is a
    fact about that history too (P2): ``parent`` must be the version directly before, and
    ``revert_of`` must name a version that already exists.
    """
    try:
        graph_hash = graph.content_hash()
    except UnicodeEncodeError as exc:
        raise ValueError("그래프에 UTF-8로 저장할 수 없는 문자(짝 없는 서로게이트)가 있습니다.") from exc
    if version.graph_hash != graph_hash:
        raise ValueError(
            f"버전의 graph_hash({version.graph_hash[:12]}…)가 "
            f"그래프 내용 해시({graph_hash[:12]}…)와 다릅니다."
        )
    expected_number = 1 if expected_head is None else expected_head + 1
    if version.number != expected_number:
        raise ValueError(
            f"새 버전 번호는 {expected_number}이어야 합니다"
            f"(기대 최신 버전: {_version_label(expected_head)}, 요청: {version.number})."
        )
    _check_lineage(version)
    for value, what in (
        (version.workspace_id, "워크스페이스 id"),
        (version.created_by, "버전 작성자"),
        (version.note, "버전 메모"),
        (version.patch_id or "", "수정안 id"),
        *((rid, "리허설 id") for rid in version.rehearsal_ids),
    ):
        check_text(value, what)
    normalized = version.model_copy(update={"created_at": as_utc(version.created_at, "버전 생성 시각")})
    return normalized, graph.canonical_json()


def _check_lineage(version: GraphVersion) -> None:
    expected_parent = version.number - 1 if version.number > 1 else None
    if version.parent != expected_parent:
        required = (
            "부모 버전이 없어야 합니다"
            if expected_parent is None
            else f"부모 버전이 {expected_parent}이어야 합니다"
        )
        raise ValueError(f"버전 {version.number}에는 {required}(요청: {_version_label(version.parent)}).")
    if version.revert_of is not None and not 1 <= version.revert_of < version.number:
        raise ValueError(
            f"되돌릴 버전({version.revert_of})은 버전 {version.number}보다 앞선, "
            "이미 저장된 버전이어야 합니다."
        )


def _version_label(number: int | None) -> str:
    return "없음" if number is None else str(number)


def version_conflict(
    workspace_id: str, expected_head: int | None, current_head: int | None
) -> VersionConflictError:
    """Conflict message a person can act on (no Python ``None`` leaks into the Korean text)."""
    current = (
        "아직 저장된 버전이 없습니다" if current_head is None else f"현재 최신 버전은 {current_head}입니다"
    )
    return VersionConflictError(
        f"워크스페이스 {workspace_id}의 그래프 버전 충돌: 이 변경의 기준 버전"
        f"({_version_label(expected_head)})이 최신이 아닙니다. {current}. "
        "최신 버전을 기준으로 다시 시도하세요."
    )


def decode_graph(graph_hash: str, canonical_json: str) -> OpsGraph:
    """Parse a stored blob and verify it still hashes to its address.

    A blob that no longer parses is corruption just like one that parses to other content, so
    both surface as :class:`StoreIntegrityError` — callers need one failure mode, not two.
    """
    try:
        graph = OpsGraph.model_validate_json(canonical_json)
    except ValidationError as exc:
        raise StoreIntegrityError(
            f"저장된 그래프 {graph_hash[:12]}…을(를) 읽을 수 없습니다(손상 의심)."
        ) from exc
    if graph.content_hash() != graph_hash:
        raise StoreIntegrityError(
            f"저장된 그래프 {graph_hash[:12]}…의 내용이 해시와 일치하지 않습니다(손상 의심)."
        )
    return graph


def check_patch(patch: Patch) -> None:
    """Keys must be storable everywhere, and every timestamp aware — otherwise the ordering
    would depend on the host's time zone."""
    check_text(patch.id, "수정안 id")
    check_workspace_key(patch.workspace_id)
    if patch.created_at is not None:
        as_utc(patch.created_at, "수정안 생성 시각")
    if patch.decided_at is not None:
        as_utc(patch.decided_at, "수정안 결정 시각")


def check_report(report: DiagnosisReport) -> datetime:
    """Validate a report's keys; return its UTC creation time (the ordering key)."""
    check_text(report.run_id, "리포트 실행 id")
    check_workspace_key(report.workspace_id)
    return as_utc(report.created_at, "리포트 생성 시각")


def check_scenario(workspace_id: str, scenario: Scenario) -> None:
    check_workspace_key(workspace_id)
    check_text(scenario.id, "시나리오 id")


def check_result(workspace_id: str, result: RehearsalResult) -> None:
    """``graph_hash`` is a filter column of its own, so it must be storable like a key."""
    check_workspace_key(workspace_id)
    check_text(result.graph_hash, "리허설 결과의 graph_hash")


def check_owner(what: str, record_id: str, owner: str | None, workspace_id: str) -> None:
    """Refuse to re-home a record that another workspace already owns.

    Patch and report ids are global keys (``get_patch(patch_id)``), and deterministic ids
    (P6) make collisions between workspaces that diagnose identical inputs realistic. A silent
    upsert would hand one workspace's record to another — the other workspace's data would
    vanish — so the save fails instead. The message does not reveal the owner.
    """
    if owner is not None and owner != workspace_id:
        raise AlreadyExistsError(f"{what} id '{record_id}'는 다른 워크스페이스에서 이미 사용 중입니다.")


def json_payload(model: BaseModel, what: str) -> JsonObject:
    """JSON-mode dump of a domain record, refused if a value is not portable JSON.

    ``model_dump(mode="json")`` keeps NaN/Infinity floats; SQLite and the in-memory store would
    keep them as-is while Postgres rejects the row. Normalising through :func:`to_json_object`
    makes all backends accept and refuse exactly the same records.
    """
    return to_json_object(model.model_dump(mode="json"), what)


def model_json_text(model: BaseModel, what: str) -> str:
    """:func:`json_text` of a domain record (what the in-memory backend keeps)."""
    return json_text(model.model_dump(mode="json"), what)


def rehearsal_id(workspace_id: str, result: RehearsalResult) -> str:
    """Deterministic rehearsal id: identical replays (same graph, scenario, config) are the same
    fact, so saving one twice is idempotent instead of creating a duplicate."""
    digest = hashlib.sha256(f"{workspace_id}\x00{result.result_hash()}".encode()).hexdigest()
    return f"rh-{digest[:24]}"


def json_text(payload: Mapping[str, object], what: str) -> str:
    """Plain-JSON text of a free-form payload (sorted keys, no NaN, UTF-8 encodable).

    Both backends store exactly this, so a payload that Postgres would reject (NaN,
    non-serialisable objects) is rejected by the in-memory store too. ``ensure_ascii=False``
    would carry a lone surrogate into the text unescaped, so the text itself is checked too.
    Text (not a dict) is also what the in-memory backend keeps: every read decodes fresh
    objects, so no caller can mutate a stored record through what it got back.
    """
    try:
        text = json.dumps(
            to_jsonable_python(dict(payload)), ensure_ascii=False, sort_keys=True, allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what}에 JSON으로 저장할 수 없는 값이 있습니다: {exc}") from exc
    check_text(text, what)
    return text


def to_json_object(payload: Mapping[str, object], what: str) -> JsonObject:
    """:func:`json_text` decoded again: the plain-JSON dict the SQL backend hands to its columns."""
    decoded: JsonObject = json.loads(json_text(payload, what))
    return decoded


def validate_event_name(name: str) -> str:
    if _EVENT_NAME.fullmatch(name) is None:
        raise ValueError(
            "이벤트 이름은 영문·숫자로 시작하고 영문·숫자·'_'·'.'·':'·'-'만 쓸 수 있으며 "
            "64자 이하여야 합니다."
        )
    return name


def check_key_digest(workspace_id: str, digest: str) -> str:
    """A workspace key and the digest of an access key (SHA-256, 64 lowercase hex characters).

    Refusing anything else keeps a plaintext key from ever being stored by mistake."""
    check_workspace_key(workspace_id)
    if _KEY_DIGEST.fullmatch(digest) is None:
        raise ValueError("접근 키는 SHA-256 다이제스트(16진수 소문자 64자)로만 저장할 수 있습니다.")
    return digest


def validate_audit(workspace_id: str, actor: str, action: str) -> None:
    check_workspace_key(workspace_id)
    if not actor.strip() or not action.strip():
        raise ValueError("감사 로그에는 행위자와 행위가 모두 있어야 합니다.")
    check_text(actor, "감사 로그 행위자")
    check_text(action, "감사 로그 행위")


# --------------------------------------------------------------------------- ordering


def workspace_order(workspace: Workspace) -> tuple[datetime, str]:
    return (workspace.created_at, workspace.id)


def signup_order(signup: Signup) -> tuple[datetime, str]:
    return (signup.created_at, signup.email)


def document_order(doc: StoredDocument) -> tuple[str, str]:
    return (doc.title, doc.doc_id)


def patch_order(patch: Patch) -> tuple[bool, datetime, str]:
    """Patches without a timestamp first, then chronological; id breaks ties (batch drafts).

    The leading flag keeps undated patches ahead of *every* dated one; substituting a sentinel
    date alone would sort them after anything older than the sentinel.
    """
    return (patch.created_at is not None, patch.created_at or _EPOCH, patch.id)


def scenario_order(scenario: Scenario) -> str:
    return scenario.id


def result_order(item: tuple[str, RehearsalResult]) -> tuple[str, str, str, str]:
    rid, result = item
    return (result.scenario_id, result.graph_hash, result.config_hash, rid)


def report_order(created_at: datetime, run_id: str) -> tuple[datetime, str]:
    return (created_at, run_id)


__all__ = [
    "AlreadyExistsError",
    "AuditEntry",
    "AuditRepository",
    "Clock",
    "DocumentRepository",
    "EventRepository",
    "InputRepository",
    "JsonObject",
    "NotFoundError",
    "Signup",
    "SignupRepository",
    "Store",
    "StoreIntegrityError",
    "StoredDocument",
    "StoredInputs",
    "VersionConflictError",
    "Workspace",
    "WorkspaceAccessRepository",
    "WorkspaceRepository",
    "as_utc",
    "check_documents",
    "check_key_digest",
    "check_owner",
    "check_patch",
    "check_report",
    "check_result",
    "check_scenario",
    "check_text",
    "check_workspace_key",
    "create_workspace_with",
    "decode_graph",
    "deletion_cutoff",
    "derive_workspace_id",
    "document_order",
    "json_payload",
    "json_text",
    "model_json_text",
    "normalize_email",
    "patch_order",
    "prepare_append",
    "rehearsal_id",
    "report_order",
    "restore_document",
    "result_order",
    "scenario_order",
    "signup_key",
    "signup_order",
    "stored_document",
    "to_json_object",
    "utc_now",
    "validate_audit",
    "validate_event_name",
    "validate_workspace_id",
    "validate_workspace_name",
    "version_conflict",
    "workspace_order",
]
