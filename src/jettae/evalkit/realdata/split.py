"""Dev/test split of a real-data study: fix the product on dev, report the numbers on test only.

Why: every fix in this study is made after reading results — the pilot's coder labels, the
investigation's traces. Numbers measured on documents that shaped the fixes would flatter them.
So the documents are assigned once, by a published rule, and the assignment is committed
(``docs/eval/gitlab-runbooks-split.json``) before any fix is measured:

1. documents already looked at (the pilot, the investigation) are forced into dev;
2. dev is filled up to its size by ``random.Random(seed).sample`` over the remaining paths in
   sorted order;
3. everything else is test. The final report uses ``--split test`` only.

:func:`assign_split` is the rule; a committed assignment must equal what the rule gives for its
own universe (``dev + test``), seed and forced list — :func:`check_assignment` verifies that, so
an edited file cannot pass as the rule's output. The assignment's digest goes into the study
settings (``run.json``): results of two different assignments never share an output folder.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "DEFAULT_SPLIT_FILE",
    "SPLIT_NAMES",
    "SplitAssignment",
    "SplitError",
    "SplitName",
    "assign_split",
    "check_assignment",
    "load_split",
    "select_split",
    "split_rule_text",
]

SplitName = Literal["dev", "test", "all"]
SPLIT_NAMES: Final[tuple[SplitName, ...]] = ("dev", "test", "all")
#: The committed assignment of the GitLab study (repository checkout; pass ``--split-file`` otherwise).
DEFAULT_SPLIT_FILE: Final = (
    Path(__file__).resolve().parents[4] / "docs" / "eval" / "gitlab-runbooks-split.json"
)

_T = TypeVar("_T")


class SplitError(ValueError):
    """The split cannot be used as given (Korean message)."""


def split_rule_text(*, dev_size: int, seed: int, forced: int) -> str:
    """The rule in words, as the assignment file and the report state it."""
    return (
        f"이미 결과를 본 문서 {forced}건(파일럿·디버깅 조사)은 dev에 강제 배정하고, "
        f"나머지 문서를 경로 순으로 정렬해 random.Random({seed}).sample로 dev가 {dev_size}건이 될 때까지 "
        "채운다. 남은 문서는 모두 test다. 제품 수정은 dev만 보고 하고, 최종 보고 수치는 test에서만 낸다."
    )


class SplitAssignment(BaseModel):
    """A committed dev/test assignment (paths are repository-relative, as the study names items)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    study: str = "real-runbooks"
    #: where the documents come from (repository @ pinned commit)
    source: str
    seed: int
    dev_size: int = Field(ge=0)
    #: documents forced into dev and why (already seen in the pilot or the investigation)
    forced: dict[str, str]
    rule: str
    dev: tuple[str, ...]
    test: tuple[str, ...]

    def members(self, split: SplitName) -> tuple[str, ...]:
        if split == "dev":
            return self.dev
        if split == "test":
            return self.test
        return tuple(sorted((*self.dev, *self.test)))

    def digest(self) -> str:
        """SHA-256 of the canonical assignment (what results are tied to)."""
        canonical = json.dumps(self.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def assign_split(
    paths: Sequence[str], *, forced: Mapping[str, str], dev_size: int, seed: int, source: str
) -> SplitAssignment:
    """The published rule (module doc): forced documents first, then a seeded sample, rest test."""
    universe = sorted(set(paths))
    missing = sorted(set(forced) - set(universe))
    if missing:
        raise SplitError(f"강제로 dev에 넣을 문서가 문서 목록에 없습니다: {', '.join(missing)}")
    if len(forced) > dev_size:
        raise SplitError(f"강제 배정 문서 {len(forced)}건이 dev 크기 {dev_size}건보다 많습니다.")
    if dev_size > len(universe):
        raise SplitError(f"dev 크기 {dev_size}건이 전체 문서 {len(universe)}건보다 큽니다.")
    rest = [p for p in universe if p not in forced]
    fill = random.Random(seed).sample(rest, dev_size - len(forced))
    dev = sorted({*forced, *fill})
    chosen = set(dev)
    return SplitAssignment(
        source=source,
        seed=seed,
        dev_size=dev_size,
        forced=dict(sorted(forced.items())),
        rule=split_rule_text(dev_size=dev_size, seed=seed, forced=len(forced)),
        dev=tuple(dev),
        test=tuple(p for p in universe if p not in chosen),
    )


def check_assignment(assignment: SplitAssignment) -> None:
    """Refuse an assignment that is not exactly the rule's output for its own inputs."""
    overlap = sorted(set(assignment.dev) & set(assignment.test))
    if overlap:
        raise SplitError(f"dev와 test에 함께 든 문서가 있습니다: {', '.join(overlap)}")
    expected = assign_split(
        (*assignment.dev, *assignment.test),
        forced=assignment.forced,
        dev_size=assignment.dev_size,
        seed=assignment.seed,
        source=assignment.source,
    )
    if expected.model_copy(update={"study": assignment.study}) != assignment:
        raise SplitError(
            "분할 파일이 공개한 규칙(강제 배정 + 시드 표본)의 결과와 다릅니다 — "
            "손으로 고친 분할은 쓰지 않습니다."
        )


def load_split(path: Path) -> SplitAssignment:
    """A committed assignment, checked against the rule."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SplitError(
            f"분할 파일을 읽을 수 없습니다: {path.as_posix()} — --split-file로 지정하세요({exc.strerror})."
        ) from exc
    try:
        assignment = SplitAssignment.model_validate_json(text)
    except ValueError as exc:
        raise SplitError(f"분할 파일 형식이 올바르지 않습니다: {path.as_posix()}") from exc
    check_assignment(assignment)
    return assignment


def select_split(
    items: Sequence[_T], *, paths: Sequence[str], assignment: SplitAssignment | None, split: SplitName
) -> list[_T]:
    """The items of ``split`` (input order kept). ``paths[i]`` is the study path of ``items[i]``.

    With a split other than ``all`` every input document must be assigned: a document the
    assignment does not know could be either side, so it is refused rather than guessed.
    """
    if len(paths) != len(items):
        raise ValueError("items와 paths의 길이가 다릅니다.")
    if split == "all":
        return list(items)
    if assignment is None:
        raise SplitError(f"--split {split}에는 분할 파일이 필요합니다.")
    known = {*assignment.dev, *assignment.test}
    unknown = [p for p in paths if p not in known]
    if unknown:
        shown = ", ".join(unknown[:5]) + (f" 외 {len(unknown) - 5}건" if len(unknown) > 5 else "")
        raise SplitError(f"분할 파일에 배정되지 않은 문서가 입력에 있습니다: {shown}")
    wanted = set(assignment.members(split))
    return [item for item, path in zip(items, paths, strict=True) if path in wanted]
