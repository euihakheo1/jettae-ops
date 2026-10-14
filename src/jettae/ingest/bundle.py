"""Load a whole input bundle (runbooks + structured sources) from one directory.

Layout (every structured file is optional)::

    <bundle>/runbooks/**/*.{md,markdown,txt,pdf,docx,html,htm}   (or runbook files at top level)
    <bundle>/roster.csv | roster.yaml | roster.yml
    <bundle>/oncall.csv | oncall.yaml | oncall.yml
    <bundle>/policy.yaml | policy.yml
    <bundle>/catalog.yaml | catalog.yml | catalog-info.yaml | catalog-info.yml
             | docker-compose.yml | docker-compose.yaml | compose.yaml | compose.yml
             | dependencies.json

Reproducibility (P6): documents are loaded in sorted relative-path order and carry their
*bundle-relative* POSIX path, so the same bundle checked out anywhere yields byte-identical
documents and therefore the same graph hash. Paths are compared in NFC form because macOS
stores Korean file names decomposed (NFD): sorting the raw names would order "결제.md" against
other names differently than on Linux or Windows.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

from jettae.domain import CatalogService, Document, InputBundle, OnCallShift, Policy, RosterEntry
from jettae.ingest.loaders import DEFAULT_MAX_BYTES, MEDIA_TYPES, load_bytes, read_limited
from jettae.ingest.structured import load_catalog, load_oncall, load_policy, load_roster, merge_catalogs

__all__ = ["load_bundle_dir"]

_RUNBOOK_DIR = "runbooks"
_ROSTER_FILES = ("roster.csv", "roster.yaml", "roster.yml")
_ONCALL_FILES = ("oncall.csv", "oncall.yaml", "oncall.yml")
_POLICY_FILES = ("policy.yaml", "policy.yml")
#: Precedence order for merging: hand-written catalog first, then Backstage, then deployment
#: descriptors, then observed traffic (which only adds dependencies and missing services).
_CATALOG_FILES = (
    "catalog.yaml",
    "catalog.yml",
    "catalog-info.yaml",
    "catalog-info.yml",
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yaml",
    "compose.yml",
    "dependencies.json",
)


def _is_runbook(path: Path) -> bool:
    name = path.name
    return (
        path.is_file()
        and path.suffix.lower() in MEDIA_TYPES
        and not name.startswith((".", "~$"))  # hidden files, Office lock files
    )


def _runbook_paths(root: Path) -> list[Path]:
    """Files under ``runbooks/`` (recursively) if that directory exists, else top-level runbook
    files except ``README*`` — a bundle's README describes the bundle, it is not a runbook."""
    runbook_dir = root / _RUNBOOK_DIR
    if runbook_dir.is_dir():
        candidates = [
            p
            for p in runbook_dir.rglob("*")
            if not any(part.startswith(".") for part in p.relative_to(runbook_dir).parts[:-1])
        ]
    else:
        candidates = [p for p in root.iterdir() if not p.name.upper().startswith("README")]
    return sorted((p for p in candidates if _is_runbook(p)), key=lambda p: _relative_name(root, p))


def _relative_name(root: Path, path: Path) -> str:
    """Bundle-relative POSIX path in NFC form: the document path and the sort key."""
    return unicodedata.normalize("NFC", path.relative_to(root).as_posix())


def _single(root: Path, names: tuple[str, ...], what: str) -> Path | None:
    found = [root / n for n in names if (root / n).is_file()]
    if len(found) > 1:
        listed = ", ".join(p.name for p in found)
        raise ValueError(f"{what} 파일이 여러 개 있습니다: {listed}. 하나만 남겨 주세요.")
    return found[0] if found else None


def _load_documents(root: Path, max_bytes: int) -> tuple[Document, ...]:
    paths = _runbook_paths(root)
    if not paths:
        supported = ", ".join(sorted(MEDIA_TYPES))
        raise ValueError(
            f"런북 문서를 찾을 수 없습니다: {root.as_posix()} "
            f"(runbooks/ 폴더에 {supported} 파일을 넣어 주세요)"
        )
    documents: list[Document] = []
    for path in paths:
        name = _relative_name(root, path)
        documents.append(load_bytes(name, read_limited(path, name, max_bytes), max_bytes=max_bytes))
    return tuple(documents)


def _load_catalog(root: Path) -> tuple[CatalogService, ...]:
    sources = [root / n for n in _CATALOG_FILES if (root / n).is_file()]
    return merge_catalogs(*(load_catalog(p) for p in sources))


def load_bundle_dir(path: str | Path, *, max_bytes: int = DEFAULT_MAX_BYTES) -> InputBundle:
    """Load every input of a diagnosis from ``path`` (see module docstring for the layout).

    Raises ``FileNotFoundError`` if ``path`` is not a directory and ``ValueError`` (Korean
    message) if no runbook is found or a structured source is ambiguous or malformed.
    """
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"입력 폴더를 찾을 수 없습니다: {root.as_posix()}")
    roster_path = _single(root, _ROSTER_FILES, "명부")
    oncall_path = _single(root, _ONCALL_FILES, "당직표")
    policy_path = _single(root, _POLICY_FILES, "정책")
    roster: tuple[RosterEntry, ...] = load_roster(roster_path) if roster_path else ()
    oncall: tuple[OnCallShift, ...] = load_oncall(oncall_path) if oncall_path else ()
    policy = load_policy(policy_path) if policy_path else Policy()
    return InputBundle(
        documents=_load_documents(root, max_bytes),
        roster=roster,
        oncall=oncall,
        catalog=_load_catalog(root),
        policy=policy,
    )
