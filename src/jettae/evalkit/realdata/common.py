"""Shared plumbing of the real-data studies: files, language, rule families, agreement statistics.

Why these live together: both studies (public runbooks, postmortem retrospective) write the same
kinds of files (append-only ``results.jsonl`` for resuming, atomically replaced summaries), judge
the same product findings by rule *family*, and report the same agreement numbers. Keeping the
definitions in one place means a number in one study's report means exactly what it means in the
other's.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final, Literal, TypeVar

from pydantic import BaseModel, ConfigDict

from jettae.check.registry import rule_catalog
from jettae.domain import Element
from jettae.evalkit.kappa import cohen_kappa, krippendorff_alpha_nominal
from jettae.evalkit.metrics import Rate

__all__ = [
    "FAMILY_CODES",
    "FAMILY_LABEL_KO",
    "KOREAN_MIN_SHARE",
    "LABELS",
    "PIPELINE_REVISION",
    "Agreement",
    "Label",
    "LabelCounts",
    "append_jsonl",
    "family_catalog_text",
    "is_korean_text",
    "label_counts",
    "measure_agreement",
    "precision",
    "read_jsonl",
    "rule_family",
    "write_json",
    "write_jsonl",
    "write_text",
]

_M = TypeVar("_M", bound=BaseModel)

#: Revision of the product behaviour the studies' results depend on beyond the models, the
#: extraction prompt and the rubric — how the assembler reads names, which notes a run leaves and
#: how findings word their premises. It is part of every study's settings, so a folder made under
#: an earlier revision cannot be resumed into (its results would mix two products). v2: a title
#: stating a component's state is that component, a sentence heading is no service, a
#: service/system twin is one service; an alert runbook in a document without services keeps its
#: hand-offs; a document left without services always says why; policy-supplied premises are named
#: as such; study 2 diagnoses each incident under its own jurisdiction.
PIPELINE_REVISION: Final = "pipeline-v2"

#: A coder's verdict on one finding.
Label = Literal["correct", "incorrect", "unclear"]
LABELS: Final[tuple[Label, ...]] = ("correct", "incorrect", "unclear")

#: Rule families = the middle part of a rule id (``JT-PATH-010`` → ``PATH``). Static on purpose:
#: the gap-coding schema needs a fixed enum; a test pins it to the live rule registry.
FAMILY_CODES: Final[tuple[str, ...]] = (
    "CAT",
    "CHN",
    "DDL",
    "ESC",
    "INT",
    "KR",
    "ONC",
    "PATH",
    "PPL",
    "SPF",
)
FAMILY_LABEL_KO: Final[dict[str, str]] = {
    "CAT": "카탈로그 대조",
    "CHN": "연락 채널",
    "DDL": "시한·목표 시간",
    "ESC": "상향(에스컬레이션)",
    "INT": "판단 보류·근거 무결성",
    "KR": "한국 법정 신고",
    "ONC": "당직표",
    "PATH": "필수 대응 경로 5요소",
    "PPL": "사람(명부 대조)",
    "SPF": "단일 실패 지점",
}

#: A text counts as Korean when Hangul makes up at least this share of its letters. Korean
#: runbooks carry many Latin service names and commands, so the bar is low; an English runbook
#: has (almost) no Hangul at all.
KOREAN_MIN_SHARE: Final = 0.2
_HANGUL: Final = re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏]")
_LATIN: Final = re.compile(r"[A-Za-z]")


def is_korean_text(text: str) -> bool:
    """Is ``text`` Korean (by Hangul share of letters)? Text without letters is not."""
    hangul = len(_HANGUL.findall(text))
    letters = hangul + len(_LATIN.findall(text))
    return letters > 0 and hangul / letters >= KOREAN_MIN_SHARE


def rule_family(rule_id: str) -> str:
    """``JT-PATH-010`` → ``PATH`` (an id without that shape is its own family)."""
    parts = rule_id.split("-")
    return parts[1] if len(parts) >= 3 else rule_id


def family_catalog_text() -> str:
    """Trusted description of every rule family for coders: the published rules, grouped.

    Built from the live registry (titles, elements, rationale) so coders judge against exactly
    the criteria the product applies, in the product's own words.
    """
    lines: list[str] = []
    rules = sorted(rule_catalog(), key=lambda r: r.rule_id)
    for family in FAMILY_CODES:
        members = [r for r in rules if rule_family(r.rule_id) == family]
        lines.append(f"## {family} — {FAMILY_LABEL_KO[family]}")
        for rule in members:
            element = f" [요소: {rule.element.value}]" if rule.element is not None else ""
            rationale = " ".join(rule.rationale.split())
            lines.append(f"- {rule.rule_id} {rule.title}{element}: {rationale}")
        lines.append("")
    elements = ", ".join(e.value for e in Element)
    lines.append(f"필수 대응 경로 요소(element) 값: {elements}")
    return "\n".join(lines).strip()


# --------------------------------------------------------------------------- statistics


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True)


class LabelCounts(_Model):
    correct: int = 0
    incorrect: int = 0
    unclear: int = 0
    #: findings this coder gave no usable label (refusal, invalid output, omitted id)
    missing: int = 0


def label_counts(labels: Iterable[Label | None]) -> LabelCounts:
    counts = Counter(label if label is not None else "missing" for label in labels)
    return LabelCounts(
        correct=counts["correct"],
        incorrect=counts["incorrect"],
        unclear=counts["unclear"],
        missing=counts["missing"],
    )


class Agreement(_Model):
    """Agreement of two coders over the units both labelled (``None`` when not computable)."""

    n: int
    observed: float | None
    cohen_kappa: float | None
    krippendorff_alpha: float | None


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def measure_agreement(a: Sequence[str | None], b: Sequence[str | None]) -> Agreement:
    """Cohen's κ and Krippendorff's α (nominal) over the units where both coders gave a label."""
    if len(a) != len(b):
        raise ValueError(f"두 코더의 판정 수가 다릅니다: {len(a)} ≠ {len(b)}")
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    if not pairs:
        return Agreement(n=0, observed=None, cohen_kappa=None, krippendorff_alpha=None)
    left = [x for x, _ in pairs]
    right = [y for _, y in pairs]
    observed = sum(1 for x, y in pairs if x == y) / len(pairs)
    try:
        alpha: float | None = krippendorff_alpha_nominal([left, right])
    except ValueError:
        alpha = None
    return Agreement(
        n=len(pairs),
        observed=_rounded(observed),
        cohen_kappa=_rounded(cohen_kappa(left, right)),
        krippendorff_alpha=_rounded(alpha),
    )


def precision(labels: Iterable[Label | None]) -> Rate:
    """correct / (correct + incorrect) with its Wilson 95% interval; unclear/missing are left out."""
    counts = label_counts(labels)
    return Rate.of(counts.correct, counts.correct + counts.incorrect)


# --------------------------------------------------------------------------- files


def read_jsonl(path: Path, model: type[_M]) -> list[_M]:
    """Every record of a JSONL file (missing file = none). A torn last line — the process was
    killed mid-write — is ignored, so a resumed run simply redoes that item."""
    if not path.is_file():
        return []
    records: list[_M] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(model.model_validate_json(line))
        except ValueError:
            if index == len(lines) - 1:
                break
            raise
    return records


def append_jsonl(path: Path, record: BaseModel) -> None:
    """Append one record and flush it to disk (the resume point of a long paid run)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(record.model_dump_json() + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Replace ``path`` atomically (a crash never leaves half a report behind)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_json(path: Path, payload: object) -> None:
    write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n")


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> None:
    write_text(path, "".join(record.model_dump_json() + "\n" for record in records))
