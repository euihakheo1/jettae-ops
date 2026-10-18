"""Study 1 (실증 ①): the product on real public runbooks, every finding judged by two coders.

Input: a folder of runbooks (the GitLab.com operations runbooks at a pinned commit, English,
MIT). For each runbook the product pipeline runs in LLM mode with no roster or catalog — person
checks then report '확인 불가' as designed — under a policy whose jurisdiction is the source
company's (``--jurisdiction``, US for GitLab: Korean statutory reports are not applied to it), and
two independent coders of different model families answer two questions per finding against the
runbook text: is its premise valid (does the document cover that service/element?) and is its
gap claim true? The label is derived: ``correct`` = both yes, ``incorrect`` = either no,
``unclear`` otherwise (:func:`~jettae.evalkit.realdata.coding.derive_label`). ``--split dev|test``
restricts the run to one side of the committed dev/test assignment
(:mod:`~jettae.evalkit.realdata.split`); final numbers come from ``test`` only. Outputs (``--out``):

* ``results.jsonl`` — one line per finished runbook: graph hash, findings (rule, severity,
  element, evidence quote + line + source link), holds, LLM usage, both coders' labels. Append-
  only while running (the resume point), rewritten in path order at the end.
* ``metrics.json`` — Cohen's κ / Krippendorff's α, precision = correct / (correct + incorrect)
  with Wilson 95% intervals per coder and for agreed labels, overall, without the Korean-law
  family, and per rule family.
* ``review.json`` / ``review.csv`` — a seeded sample (~100) stratified by rule family for blind
  human review: no coder label is in it (they stay in ``results.jsonl``).
* ``report.md`` — Korean summary with the study's limitations stated next to the numbers.
* ``errors.jsonl`` — items that could not be measured (retried by the next run), ``run.json`` —
  the settings the results belong to (a resume with other settings is refused).
* ``spend.json`` / ``manifest.json`` — the cumulative spend ledger and the input folders used
  (:mod:`~jettae.evalkit.realdata.resume`). ``--input`` may later point at a larger folder (a
  superset of the finished runbooks): only the new runbooks are processed and paid for, and every
  metric is recomputed over all of them. A ``review.csv`` / ``review.json`` that already carries
  human labels is never overwritten — a changed sample goes to ``review.next.*``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from jettae.check.registry import rule_catalog
from jettae.domain import Document, InputBundle, Policy
from jettae.evalkit.metrics import Rate
from jettae.evalkit.realdata.clients import StudyClients
from jettae.evalkit.realdata.coding import (
    ANSWERS,
    CODER_MAX_TOKENS,
    FINDING_CODER_INSTRUCTIONS,
    RUBRIC_VERSION,
    Answer,
    CoderVerdict,
    FindingForCoder,
    FindingLabelBatch,
    code_findings,
    finding_coder_request,
)
from jettae.evalkit.realdata.common import (
    FAMILY_LABEL_KO,
    PIPELINE_REVISION,
    Agreement,
    Label,
    LabelCounts,
    append_jsonl,
    is_korean_text,
    label_counts,
    measure_agreement,
    precision,
    read_jsonl,
    write_json,
    write_jsonl,
    write_text,
)
from jettae.evalkit.realdata.diagnose import (
    BACKTRANSLATION_SKIPPED,
    DiagnosisRecord,
    EvidenceRecord,
    FindingRecord,
    PipelineFallbackError,
    diagnose_bundle,
    plan_backtranslations,
)
from jettae.evalkit.realdata.dryrun import PlannedCall, Role
from jettae.evalkit.realdata.pricing import ASSUMED_BACKTRANSLATIONS_PER_DOC
from jettae.evalkit.realdata.providers import BudgetExhausted, SpendReport, gather_jobs
from jettae.evalkit.realdata.resume import (
    SUPERSEDED_FILE,
    Finished,
    ResumeState,
    SpendLedger,
    StopKind,
    prepare_resume,
    read_manifest,
    start_run,
    stop_kind_of,
)
from jettae.evalkit.realdata.split import (
    DEFAULT_SPLIT_FILE,
    SplitAssignment,
    SplitName,
    load_split,
    select_split,
)
from jettae.extract.llm_extractor import (
    EXTRACTION_INSTRUCTIONS,
    LlmGraphPayload,
    blank_quarantined,
    build_extraction_context,
    chunk_document,
)
from jettae.ingest import MEDIA_TYPES, load_bytes, scan_injection
from jettae.llm.anthropic_client import Effort
from jettae.llm.base import LLMError, UntrustedDocument
from jettae.llm.schema import to_strict_json_schema
from jettae.orchestrator.state import DiagnosisOptions

__all__ = [
    "DEFAULT_COMMIT",
    "DEFAULT_JURISDICTIONS",
    "DEFAULT_REPO_URL",
    "DEFAULT_SEED",
    "EXTRACT_MAX_TOKENS",
    "RECORD_VERSION",
    "FindingLabels",
    "ItemError",
    "ReviewRow",
    "RunbookMetrics",
    "RunbookResult",
    "RunbookSource",
    "RunbookStudyConfig",
    "StudyOutcome",
    "compute_metrics",
    "extraction_prompt_digest",
    "load_runbooks",
    "load_study_sources",
    "plan_runbook_calls",
    "render_report",
    "resume_state",
    "review_sample",
    "run_runbook_study",
    "source_commit",
    "source_url",
    "study_design",
]

DEFAULT_REPO_URL: Final = "https://gitlab.com/gitlab-com/runbooks"
DEFAULT_COMMIT: Final = "844a2b882687492cf117cc2fc553f7fbf9a02197"
DEFAULT_SEED: Final = 20261017
#: Jurisdiction of the default source (GitLab Inc., a US company): Korean statutory reports
#: (JT-KR-*) do not bind it, so the study's policy does not apply them.
DEFAULT_JURISDICTIONS: Final[tuple[str, ...]] = ("US",)
#: ``LLMClient.structured``'s default, which the product's extractor uses for every chunk.
EXTRACT_MAX_TOKENS: Final = 16_000
STUDY: Final = "real-runbooks"
#: What one ``results.jsonl`` line records. v2 adds the extraction IR per chunk (quotes with their
#: grounded lines), the assembler's grounding bookkeeping and a summary of the assembled graph.
RECORD_VERSION: Final = "record-v2"
#: Korean-law rules: the headline precision is also given without them (jurisdiction differs).
KR_FAMILY: Final = "KR"
#: Stop after this many runbooks in a row could not be measured (an outage, not a bad document).
MAX_CONSECUTIVE_FAILURES: Final = 3
RETRY_NOTE: Final = "다음 실행에서 다시 시도"
OUTAGE_NOTE: Final = "공급자 장애 가능성"
REVIEW_INSTRUCTIONS: Final = (
    "사람 검토(블라인드): 각 지적을 원문 링크(source_url)에서 확인하고 "
    "human_label에 correct | incorrect | unclear, human_reason에 한 줄 이유를 적으세요. "
    "코더 판정은 이 파일에 없습니다 — 검토를 마치기 전에는 results.jsonl을 열지 마세요."
)
_EXCLUDED_TOP_LEVEL: Final = ("source", "readme", "license")
_COMMIT: Final = re.compile(r"\b[0-9a-f]{40}\b")
#: Placeholder findings of the dry run (rules a roster-less English runbook typically trips).
_PLACEHOLDER_RULES: Final = (
    "JT-PATH-001",
    "JT-PATH-010",
    "JT-PATH-020",
    "JT-PATH-030",
    "JT-PATH-040",
    "JT-KR-001",
    "JT-KR-003",
    "JT-PPL-002",
    "JT-PATH-011",
    "JT-ESC-001",
    "JT-SPF-001",
    "JT-CHN-002",
)


_CODER_ROLES: Final[tuple[Role, ...]] = ("coder_a", "coder_b")


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True)


# --------------------------------------------------------------------------- inputs


@dataclass(frozen=True)
class RunbookSource:
    """One runbook: its path in the source repository, the file as published, the loaded document."""

    path: str
    raw: str
    document: Document


def _is_excluded(relative: Path) -> bool:
    if any(part.startswith(".") for part in relative.parts):
        return True
    return len(relative.parts) == 1 and relative.stem.casefold() in _EXCLUDED_TOP_LEVEL


def load_runbooks(input_dir: Path, *, limit: int | None = None) -> list[RunbookSource]:
    """Every runbook file under ``input_dir`` in path order (top-level SOURCE/README/LICENSE are
    the folder's own notes, not runbooks). Loaded exactly like an upload (normalise, mask PII,
    guard) and named by repository-relative path so the graph hash is location-independent."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"입력 폴더를 찾을 수 없습니다: {input_dir.as_posix()}")
    paths = sorted(
        (p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in MEDIA_TYPES),
        key=lambda p: p.relative_to(input_dir).as_posix(),
    )
    sources: list[RunbookSource] = []
    for path in paths:
        relative = path.relative_to(input_dir)
        if _is_excluded(relative):
            continue
        data = path.read_bytes()
        name = relative.as_posix()
        sources.append(
            RunbookSource(
                path=name, raw=data.decode("utf-8", errors="replace"), document=load_bytes(name, data)
            )
        )
        if limit is not None and len(sources) >= limit:
            break
    if not sources:
        raise ValueError(f"런북 파일이 없습니다: {input_dir.as_posix()}")
    return sources


def load_study_sources(config: RunbookStudyConfig) -> list[RunbookSource]:
    """The runbooks of this run: every runbook of the input, then the chosen split, then ``limit``.

    The split is applied before the limit, so ``--split test --limit 3`` takes the first three
    *test* runbooks, never a dev one.
    """
    sources = load_runbooks(config.input_dir)
    chosen = select_split(
        sources, paths=[s.path for s in sources], assignment=config.assignment(), split=config.split
    )
    if not chosen:
        raise ValueError(
            f"분할 '{config.split}'에 해당하는 런북이 입력에 없습니다: {config.input_dir.as_posix()}"
        )
    return chosen[: config.limit] if config.limit is not None else chosen


def source_commit(input_dir: Path, default: str = DEFAULT_COMMIT) -> str:
    """The pinned commit named in ``SOURCE.md`` (the folder's provenance note), else ``default``."""
    note = input_dir / "SOURCE.md"
    try:
        found = _COMMIT.search(note.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return default
    return found.group(0) if found else default


def source_url(repo_url: str, commit: str, path: str, line: int | None = None) -> str:
    anchor = f"#L{line}" if line is not None else ""
    return f"{repo_url.rstrip('/')}/-/blob/{commit}/{path}{anchor}"


# --------------------------------------------------------------------------- configuration


class RunbookStudyConfig(_Model):
    input_dir: Path
    out_dir: Path
    model: str = "claude-opus-5-5"
    effort: Effort = "medium"
    coder_a: str = "anthropic:claude-opus-5-5"
    coder_b: str = "openai:gpt-6.1-sol"
    #: cumulative cap over every run in ``out_dir`` (the spend ledger), USD
    max_usd: float | None = None
    #: cap of one invocation, USD (optional)
    max_usd_run: float | None = None
    #: go on although finished runbooks left the input or changed (see the resume module)
    allow_input_change: bool = False
    limit: int | None = Field(default=None, ge=1)
    concurrency: int = Field(default=4, ge=1, le=32)
    coder_batch: int = Field(default=8, ge=1, le=50)
    review_size: int = Field(default=100, ge=1)
    seed: int = DEFAULT_SEED
    assumed_findings: int = Field(default=12, ge=0)
    repo_url: str = DEFAULT_REPO_URL
    commit: str | None = None
    generated_at: str = ""
    #: jurisdictions of the policy the product runs under (Korean statutory rules apply only with KR)
    jurisdictions: tuple[str, ...] = DEFAULT_JURISDICTIONS
    #: which side of the committed dev/test assignment to run (``all`` = every runbook)
    split: SplitName = "all"
    #: the committed assignment (default: ``docs/eval/gitlab-runbooks-split.json`` of the checkout)
    split_file: Path | None = None

    @field_validator("jurisdictions", mode="before")
    @classmethod
    def _codes(cls, value: object) -> object:
        """Same normalisation as :class:`~jettae.domain.Policy` (one code or several, upper-case)."""
        return Policy.model_validate({"jurisdictions": value}).jurisdictions

    def policy(self) -> Policy:
        """The policy every runbook is diagnosed under (defaults except the jurisdictions)."""
        return Policy(jurisdictions=self.jurisdictions)

    def assignment(self) -> SplitAssignment | None:
        """The dev/test assignment in use: loaded (and checked against its rule) for ``dev``/``test``
        or when a split file is given; ``None`` for ``all`` without one."""
        if self.split == "all" and self.split_file is None:
            return None
        return load_split(self.split_file or DEFAULT_SPLIT_FILE)

    def settings(self) -> dict[str, Any]:
        """What results depend on; a resume under different settings would mix two studies.

        Besides the models and the rubric this pins the product's extraction prompt (a digest of
        the instructions and the trusted context it sends), the product's own behaviour beyond the
        prompt (:data:`~jettae.evalkit.realdata.common.PIPELINE_REVISION`) and the record format:
        results made by an earlier extraction prompt or product revision, coded under an earlier
        rubric or recorded without the extraction log cannot be resumed into the same folder.
        """
        assignment = self.assignment()
        return {
            "study": STUDY,
            "model": self.model,
            "effort": self.effort,
            "coder_a": self.coder_a,
            "coder_b": self.coder_b,
            "coder_batch": self.coder_batch,
            "rubric": RUBRIC_VERSION,
            "extraction_prompt": extraction_prompt_digest(),
            "pipeline": PIPELINE_REVISION,
            "record": RECORD_VERSION,
            "jurisdictions": list(self.jurisdictions),
            "split": self.split,
            "split_digest": assignment.digest() if assignment is not None else None,
        }


def extraction_prompt_digest() -> str:
    """Short digest of what the product's extractor tells the model.

    Instructions, empty trusted context and the strict output schema: the schema's descriptions
    (field and enum docstrings) reach the model too, so a change to them is a change of prompt.
    """
    schema = json.dumps(to_strict_json_schema(LlmGraphPayload), ensure_ascii=False, sort_keys=True)
    text = f"{EXTRACTION_INSTRUCTIONS}\n{build_extraction_context((), ())}\n{schema}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def study_design(config: RunbookStudyConfig) -> dict[str, Any]:
    """What the manifest and the report record about how the study was set up."""
    assignment = config.assignment()
    split: dict[str, Any] = {"name": config.split}
    if assignment is not None:
        split |= {
            "file": (config.split_file or DEFAULT_SPLIT_FILE).as_posix(),
            "digest": assignment.digest(),
            "rule": assignment.rule,
            "seed": assignment.seed,
            "forced": dict(assignment.forced),
            "dev": list(assignment.dev),
            "test": list(assignment.test),
        }
    return {
        "jurisdictions": list(config.jurisdictions),
        "korean_law_rules": "KR" in config.jurisdictions,
        "rubric": RUBRIC_VERSION,
        "extraction_prompt": extraction_prompt_digest(),
        "pipeline": PIPELINE_REVISION,
        "record": RECORD_VERSION,
        "split": split,
    }


# --------------------------------------------------------------------------- records


class FindingLabels(_Model):
    key: str
    coder_a: CoderVerdict
    coder_b: CoderVerdict


class RunbookResult(_Model):
    doc_path: str
    doc_sha256: str
    title: str
    korean: bool
    source_url: str
    diagnosis: DiagnosisRecord
    labels: tuple[FindingLabels, ...]
    #: actual spend of this item (pipeline + both coders), USD
    usd: float


class ItemError(_Model):
    doc_path: str
    error: str
    message: str


@dataclass(frozen=True)
class StudyOutcome:
    results: list[RunbookResult]
    metrics: RunbookMetrics
    stop_reason: str | None
    failed: int
    stop_kind: StopKind | None = None
    #: set when a human-labelled review file was kept and the new sample written beside it
    review_note: str | None = None


# --------------------------------------------------------------------------- dry-run plan


def _placeholder_findings(doc: Document, count: int) -> list[FindingRecord]:
    """Findings of typical size standing in for those the pipeline has not produced yet."""
    catalog = {r.rule_id: r for r in rule_catalog()}
    visible = blank_quarantined(doc.text, doc.quarantined)
    first = next((line.strip() for line in visible.splitlines() if line.strip()), "")[:160]
    out: list[FindingRecord] = []
    for index in range(count):
        rule = catalog[_PLACEHOLDER_RULES[index % len(_PLACEHOLDER_RULES)]]
        out.append(
            FindingRecord(
                key=f"placeholder-{index}",
                finding_id=f"{rule.rule_id}:placeholder{index}",
                rule_id=rule.rule_id,
                family=rule.rule_id.split("-")[1],
                severity=rule.severity.value,
                status="broken",
                title=rule.title,
                message=" ".join(rule.rationale.split()),
                service_id="service:placeholder",
                service_name="placeholder service",
                element=rule.element.value if rule.element is not None else None,
                doc_path=doc.path,
                evidence=(EvidenceRecord(quote=first, line=1, source_line=1),) if first else (),
            )
        )
    return out


def _for_coder(findings: Sequence[FindingRecord]) -> list[FindingForCoder]:
    return [
        FindingForCoder(
            ref=f"F{index}",
            rule_id=f.rule_id,
            status=f.status,
            severity=f.severity,
            title=f.title,
            message=f.message,
            service=f.service_name,
            element=f.element,
            evidence=tuple((e.line, e.quote) for e in f.evidence),
        )
        for index, f in enumerate(findings, start=1)
    ]


def _batches(items: Sequence[Any], size: int) -> list[Sequence[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _visible_title(title: str, visible: str) -> str:
    """Same rule as the extractor: forward the title only when it is visible text anyway."""
    stripped = title.strip()
    return stripped if stripped and stripped in visible else ""


def plan_runbook_calls(
    sources: Sequence[RunbookSource], *, coder_batch: int, assumed_findings: int
) -> list[PlannedCall]:
    """Every request a real run would send, built by the functions the run uses.

    Extraction requests are exact (same guard, blanking, chunking, title rule, context,
    instructions, schema and ``max_tokens`` as the product's extractor). Back-translation is not
    asked for non-Korean sources; for Korean ones the number of response edges is unknown before
    extraction, so a documented number of placeholder requests stands in. Coder requests contain
    the real runbook and ``assumed_findings`` placeholder findings in batches of ``coder_batch``.
    """
    calls: list[PlannedCall] = []
    context = build_extraction_context((), ())
    max_chunk = DiagnosisOptions().max_chunk_chars
    for source in sources:
        doc, _notices = scan_injection(source.document)
        korean = is_korean_text(doc.text)
        visible = blank_quarantined(doc.text, doc.quarantined)
        title = _visible_title(doc.title, visible)
        calls.extend(
            PlannedCall(
                item=source.path,
                stage="extract",
                role="main",
                instructions=EXTRACTION_INSTRUCTIONS,
                schema=LlmGraphPayload,
                documents=(
                    UntrustedDocument(doc_id=doc.doc_id, title=title, text=chunk.text, offset=chunk.start),
                ),
                context=context,
                max_tokens=EXTRACT_MAX_TOKENS,
                doc_chars=len(chunk.text),
                korean=korean,
            )
            for chunk in chunk_document(doc, max_chunk_chars=max_chunk)
        )
        if korean:
            calls += plan_backtranslations(source.path, doc, ASSUMED_BACKTRANSLATIONS_PER_DOC)
        for batch in _batches(_for_coder(_placeholder_findings(doc, assumed_findings)), coder_batch):
            documents, coder_context = finding_coder_request(doc, batch)
            calls.extend(
                PlannedCall(
                    item=source.path,
                    stage="code_findings",
                    role=role,
                    instructions=FINDING_CODER_INSTRUCTIONS,
                    schema=FindingLabelBatch,
                    documents=documents,
                    context=coder_context,
                    max_tokens=CODER_MAX_TOKENS,
                    items=len(batch),
                    korean=korean,
                    exact=False,
                )
                for role in _CODER_ROLES
            )
    return calls


# --------------------------------------------------------------------------- running


async def _label_all(
    doc: Document, findings: Sequence[FindingRecord], clients: StudyClients, *, batch: int, concurrency: int
) -> tuple[FindingLabels, ...]:
    """Both coders on every finding (batched, concurrent, results in finding order). A failure
    stops new coder requests of this runbook (:func:`gather_jobs`), so none outlives the item."""
    groups = _batches(list(findings), batch)
    jobs = [
        partial(code_findings, coder, doc, _for_coder(group))
        for coder in (clients.coder_a, clients.coder_b)
        for group in groups
    ]
    results = await gather_jobs(jobs, concurrency=concurrency)
    a_verdicts = [v for group in results[: len(groups)] for v in group]
    b_verdicts = [v for group in results[len(groups) :] for v in group]
    return tuple(
        FindingLabels(key=f.key, coder_a=a, coder_b=b)
        for f, a, b in zip(findings, a_verdicts, b_verdicts, strict=True)
    )


async def _study_item(
    source: RunbookSource, config: RunbookStudyConfig, clients: StudyClients, commit: str
) -> RunbookResult:
    before = clients.tracker.spent_usd
    doc = source.document

    def url_for(_doc: Document, line: int | None) -> str:
        return source_url(config.repo_url, commit, source.path, line)

    diagnosis = await diagnose_bundle(
        InputBundle(documents=(doc,), policy=config.policy()),
        clients.main,
        concurrency=config.concurrency,
        key_prefix=source.path,
        raw_texts={doc.doc_id: source.raw},
        url_for=url_for,
    )
    labels = await _label_all(
        doc, diagnosis.findings, clients, batch=config.coder_batch, concurrency=config.concurrency
    )
    return RunbookResult(
        doc_path=source.path,
        doc_sha256=doc.sha256,
        title=doc.title,
        korean=is_korean_text(doc.text),
        source_url=source_url(config.repo_url, commit, source.path),
        diagnosis=diagnosis,
        labels=labels,
        usd=round(clients.tracker.spent_usd - before, 6),
    )


def _finished(out: Path) -> dict[str, Finished]:
    """Finished runbooks in ``results.jsonl`` (a later line of the same path wins)."""
    return {
        r.doc_path: Finished(r.doc_sha256, r.usd) for r in read_jsonl(out / "results.jsonl", RunbookResult)
    }


def resume_state(config: RunbookStudyConfig, items: Sequence[RunbookSource]) -> ResumeState:
    """Settings, inputs (superset rule) and spend ledger of ``config.out_dir`` — writes nothing."""
    return prepare_resume(
        config.out_dir,
        study=STUDY,
        settings=config.settings(),
        input_dir=config.input_dir,
        current=[(s.path, s.document.sha256) for s in items],
        finished=_finished(config.out_dir),
        allow_input_change=config.allow_input_change,
        item_word="런북",
    )


def _settle_results(path: Path, current: dict[str, str]) -> list[RunbookResult]:
    """Rewrite ``results.jsonl`` with the current runbooks (path order, one line each); results of
    runbooks no longer in the input or with other content move to the superseded file."""
    records = read_jsonl(path, RunbookResult)
    latest = {r.doc_path: r for r in records}
    keep = sorted(
        (r for r in latest.values() if current.get(r.doc_path) == r.doc_sha256), key=lambda r: r.doc_path
    )
    kept = {id(r) for r in keep}
    for record in records:
        if id(record) not in kept:
            append_jsonl(path.parent / SUPERSEDED_FILE, record)
    write_jsonl(path, keep)
    return keep


def _commit_note(out: Path, commit: str) -> str | None:
    """A warning when earlier inputs of this folder were pinned to another commit."""
    others = sorted(
        {use.commit for use in read_manifest(out, STUDY).inputs if use.commit and use.commit != commit}
    )
    if not others:
        return None
    return (
        f"주의: 이전 입력의 고정 커밋({', '.join(c[:12] for c in others)})과 이번 커밋({commit[:12]})이 "
        "다릅니다 — 이전 결과의 원문 링크는 이전 커밋을 가리킵니다(내용은 같은 해시로 확인됨)."
    )


def _start(
    state: ResumeState,
    config: RunbookStudyConfig,
    items: Sequence[RunbookSource],
    clients: StudyClients,
    *,
    commit: str,
    echo: Callable[[str], object],
) -> SpendLedger:
    """Warn about a changed source commit, then start the run (settings, manifest, ledger)."""
    if note := _commit_note(config.out_dir, commit):
        echo(note)
    return start_run(
        state,
        config.out_dir,
        settings=config.settings(),
        input_dir=config.input_dir,
        items=[(s.path, s.document.sha256, s.path) for s in items],
        tracker=clients.tracker,
        commit=commit,
        design=study_design(config),
    )


async def run_runbook_study(
    config: RunbookStudyConfig,
    clients: StudyClients,
    *,
    echo: Callable[[str], object] = print,
    sources: Sequence[RunbookSource] | None = None,
) -> StudyOutcome:
    """The paid study (see module doc). Resumes and grows from ``results.jsonl`` (superset
    inputs), books every call in the folder's spend ledger, stops cleanly at either cap."""
    out = config.out_dir
    out.mkdir(parents=True, exist_ok=True)
    items = list(sources) if sources is not None else load_study_sources(config)
    commit = config.commit or source_commit(config.input_dir)
    state = resume_state(config, items)
    ledger = _start(state, config, items, clients, commit=commit, echo=echo)
    results_path = out / "results.jsonl"
    done = set(state.inputs.done)
    stop_reason: str | None = None
    stop_kind: StopKind | None = None
    failed = 0
    in_a_row = 0
    try:
        for index, source in enumerate(items, start=1):
            if source.path in done:
                continue
            ledger.begin_item(source.path)
            try:
                result = await _study_item(source, config, clients, commit)
            except BudgetExhausted as exc:
                ledger.end_item(source.path, "stopped")
                stop_reason, stop_kind = str(exc), stop_kind_of(exc.scope)
                echo(f"[{index}/{len(items)}] {source.path}: {stop_reason}")
                break
            except (PipelineFallbackError, LLMError) as exc:
                ledger.end_item(source.path, "failed")
                failed += 1
                in_a_row += 1
                append_jsonl(
                    out / "errors.jsonl",
                    ItemError(doc_path=source.path, error=type(exc).__name__, message=str(exc)[:500]),
                )
                echo(f"[{index}/{len(items)}] {source.path}: 측정 실패({type(exc).__name__}) — {RETRY_NOTE}")
                if in_a_row >= MAX_CONSECUTIVE_FAILURES:
                    stop_reason = (
                        f"연속 {in_a_row}건 실패로 멈춥니다({OUTAGE_NOTE}). 마지막 오류: {type(exc).__name__}"
                    )
                    stop_kind = "failures"
                    break
                continue
            in_a_row = 0
            append_jsonl(results_path, result)
            ledger.end_item(source.path, "done")
            echo(
                f"[{index}/{len(items)}] {source.path}: 지적 {len(result.diagnosis.findings)}건, "
                f"${result.usd:.4f} (이번 실행 ${clients.tracker.spent_usd:.4f}, "
                f"이 폴더 누적 ${ledger.spent_usd:.4f})"
            )
    except BaseException as exc:  # interrupted (Ctrl+C, a crash): the ledger still closes the run
        ledger.finish_run(stop_reason=stop_reason or f"예외로 중단: {type(exc).__name__}")
        raise
    ledger.finish_run(stop_reason=stop_reason)
    unique = _settle_results(results_path, {s.path: s.document.sha256 for s in items})
    metrics = compute_metrics(
        unique,
        config=config,
        total=len(items),
        failed=failed,
        stop_reason=stop_reason,
        spend=clients.tracker.report(),
        commit=commit,
        ledger_usd=ledger.spent_usd,
        stop_kind=stop_kind,
    )
    review_note = write_outputs(unique, metrics, config)
    if review_note:
        echo(review_note)
    return StudyOutcome(
        results=unique,
        metrics=metrics,
        stop_reason=stop_reason,
        failed=failed,
        stop_kind=stop_kind,
        review_note=review_note,
    )


# --------------------------------------------------------------------------- metrics


class PrecisionBlock(_Model):
    coder_a: Rate
    coder_b: Rate
    #: both coders gave the same label: hits = both correct, n = both correct + both incorrect
    agreed: Rate


class AnswerCounts(_Model):
    yes: int = 0
    no: int = 0
    unclear: int = 0
    #: findings this coder gave no usable answer (refusal, invalid output, omitted id, old rubric)
    missing: int = 0


def answer_counts(answers: Iterable[Answer | None]) -> AnswerCounts:
    counts = Counter(a if a in ANSWERS else "missing" for a in answers)
    return AnswerCounts(
        yes=counts["yes"], no=counts["no"], unclear=counts["unclear"], missing=counts["missing"]
    )


class QuestionsBlock(_Model):
    """The two rubric questions one by one: how each coder answered and how much they agree.

    ``premise_error`` is the share of findings whose premise a coder judged invalid (no / (yes +
    no); agreed = both no / (both no + both yes)). ``precision_valid_premise`` is the precision
    over findings whose premise that coder judged valid: claim yes / (claim yes + claim no)
    (agreed: over findings both judged valid, both claim yes / (both yes + both no)).
    """

    premise_a: AnswerCounts
    premise_b: AnswerCounts
    claim_a: AnswerCounts
    claim_b: AnswerCounts
    agreement_premise: Agreement
    agreement_claim: Agreement
    premise_error: PrecisionBlock
    precision_valid_premise: PrecisionBlock


class FamilyBlock(_Model):
    family: str
    label_ko: str
    findings: int
    labels_a: LabelCounts
    labels_b: LabelCounts
    precision: PrecisionBlock
    agreement: Agreement
    agreement_premise: Agreement | None = None
    agreement_claim: Agreement | None = None


class ReviewInfo(_Model):
    size: int
    seed: int
    by_family: dict[str, int]


class RunbookMetrics(_Model):
    study: Literal["real-runbooks"] = "real-runbooks"
    generated_at: str
    source: str
    model: str
    effort: str
    coder_a: str
    coder_b: str
    runbooks_total: int
    runbooks_done: int
    runbooks_failed: int
    partial: bool
    stop_reason: str | None
    findings: int
    by_status: dict[str, int]
    by_severity: dict[str, int]
    labels_a: LabelCounts
    labels_b: LabelCounts
    agreement: Agreement
    precision: PrecisionBlock
    precision_ex_kr: PrecisionBlock
    by_status_precision: dict[str, PrecisionBlock]
    families: tuple[FamilyBlock, ...]
    holds: int
    holds_backtranslation: int
    backtranslation_skipped_documents: int
    backtranslation_note: str
    llm_calls_by_task: dict[str, int]
    #: every dollar spent in this output folder, all runs (the spend ledger; what --max-usd caps),
    #: failed and stopped runbooks included
    spent_usd_total: float
    spend_this_run: SpendReport | None
    review: ReviewInfo
    #: runbooks with at least one extraction chunk the model refused or answered invalidly (the
    #: product then diagnosed that chunk with the rule parser alone), and those chunks
    runbooks_partial_extraction: int = 0
    extract_failed_chunks: int = 0
    #: extraction log totals over the results (record-v2): items the LLM proposed, and elements
    #: the assembler dropped (no grounding, quarantined text, invalid or code-only)
    extraction_items: int = 0
    grounding_dropped: int = 0
    #: what the runbooks in these results cost (their last, finished attempt)
    spent_usd_results: float = 0.0
    stop_kind: str | None = None
    #: how findings were coded (two questions, derived label) and the study design
    rubric: str = RUBRIC_VERSION
    questions: QuestionsBlock | None = None
    jurisdictions: tuple[str, ...] = ()
    split: str = "all"
    split_digest: str | None = None
    split_rule: str | None = None


@dataclass(frozen=True)
class _Pair:
    finding: FindingRecord
    a: Label | None
    b: Label | None
    premise_a: Answer | None = None
    premise_b: Answer | None = None
    claim_a: Answer | None = None
    claim_b: Answer | None = None


def _pairs(results: Iterable[RunbookResult]) -> list[_Pair]:
    pairs: list[_Pair] = []
    for result in results:
        labels = {item.key: item for item in result.labels}
        for finding in result.diagnosis.findings:
            item = labels.get(finding.key)
            if item is None:
                pairs.append(_Pair(finding, None, None))
                continue
            a, b = item.coder_a, item.coder_b
            pairs.append(
                _Pair(finding, a.label, b.label, a.premise_valid, b.premise_valid, a.claim_true, b.claim_true)
            )
    return pairs


def _share(hits: int, misses: int) -> Rate:
    return Rate.of(hits, hits + misses)


def _premise_error(pairs: Sequence[_Pair]) -> PrecisionBlock:
    def coder(answers: list[Answer | None]) -> Rate:
        return _share(answers.count("no"), answers.count("yes"))

    both_no = sum(1 for p in pairs if p.premise_a == p.premise_b == "no")
    both_yes = sum(1 for p in pairs if p.premise_a == p.premise_b == "yes")
    return PrecisionBlock(
        coder_a=coder([p.premise_a for p in pairs]),
        coder_b=coder([p.premise_b for p in pairs]),
        agreed=_share(both_no, both_yes),
    )


def _precision_valid_premise(pairs: Sequence[_Pair]) -> PrecisionBlock:
    def coder(answers: list[tuple[Answer | None, Answer | None]]) -> Rate:
        claims = [claim for premise, claim in answers if premise == "yes"]
        return _share(claims.count("yes"), claims.count("no"))

    valid = [p for p in pairs if p.premise_a == p.premise_b == "yes"]
    return PrecisionBlock(
        coder_a=coder([(p.premise_a, p.claim_a) for p in pairs]),
        coder_b=coder([(p.premise_b, p.claim_b) for p in pairs]),
        agreed=_share(
            sum(1 for p in valid if p.claim_a == p.claim_b == "yes"),
            sum(1 for p in valid if p.claim_a == p.claim_b == "no"),
        ),
    )


def _questions(pairs: Sequence[_Pair]) -> QuestionsBlock:
    return QuestionsBlock(
        premise_a=answer_counts(p.premise_a for p in pairs),
        premise_b=answer_counts(p.premise_b for p in pairs),
        claim_a=answer_counts(p.claim_a for p in pairs),
        claim_b=answer_counts(p.claim_b for p in pairs),
        agreement_premise=measure_agreement([p.premise_a for p in pairs], [p.premise_b for p in pairs]),
        agreement_claim=measure_agreement([p.claim_a for p in pairs], [p.claim_b for p in pairs]),
        premise_error=_premise_error(pairs),
        precision_valid_premise=_precision_valid_premise(pairs),
    )


def _agreed(pairs: Sequence[_Pair]) -> Rate:
    both_correct = sum(1 for p in pairs if p.a == p.b == "correct")
    both_incorrect = sum(1 for p in pairs if p.a == p.b == "incorrect")
    return Rate.of(both_correct, both_correct + both_incorrect)


def _llm_items(diagnosis: DiagnosisRecord) -> int:
    """IR items the LLM proposed for one runbook (entities, relations, document-level statements)."""
    return sum(
        len(c.entities) + len(c.relations) + len(c.subjects) + len(c.delegations) + len(c.shared)
        for c in diagnosis.extraction
        if c.extractor == "llm"
    )


def _dropped(diagnosis: DiagnosisRecord) -> int:
    g = diagnosis.grounding
    return 0 if g is None else g.dropped_ungrounded + g.dropped_quarantined + g.dropped_invalid


def _precision_block(pairs: Sequence[_Pair]) -> PrecisionBlock:
    return PrecisionBlock(
        coder_a=precision(p.a for p in pairs), coder_b=precision(p.b for p in pairs), agreed=_agreed(pairs)
    )


def compute_metrics(
    results: Sequence[RunbookResult],
    *,
    config: RunbookStudyConfig,
    total: int,
    failed: int,
    stop_reason: str | None,
    spend: SpendReport | None,
    commit: str,
    ledger_usd: float | None = None,
    stop_kind: StopKind | None = None,
) -> RunbookMetrics:
    pairs = _pairs(results)
    results_usd = round(sum(r.usd for r in results), 6)
    by_family: dict[str, list[_Pair]] = defaultdict(list)
    for pair in pairs:
        by_family[pair.finding.family].append(pair)
    families = tuple(
        FamilyBlock(
            family=family,
            label_ko=FAMILY_LABEL_KO.get(family, family),
            findings=len(group),
            labels_a=label_counts(p.a for p in group),
            labels_b=label_counts(p.b for p in group),
            precision=_precision_block(group),
            agreement=measure_agreement([p.a for p in group], [p.b for p in group]),
            agreement_premise=measure_agreement([p.premise_a for p in group], [p.premise_b for p in group]),
            agreement_claim=measure_agreement([p.claim_a for p in group], [p.claim_b for p in group]),
        )
        for family, group in sorted(by_family.items())
    )
    statuses = sorted({p.finding.status for p in pairs})
    tasks: Counter[str] = Counter()
    for result in results:
        tasks.update(result.diagnosis.llm_by_task)
    sample = review_sample(results, size=config.review_size, seed=config.seed)
    assignment = config.assignment()
    return RunbookMetrics(
        generated_at=config.generated_at,
        source=f"{config.repo_url}@{commit}",
        model=config.model,
        effort=config.effort,
        coder_a=config.coder_a,
        coder_b=config.coder_b,
        runbooks_total=total,
        runbooks_done=len(results),
        runbooks_failed=failed,
        partial=len(results) < total,
        stop_reason=stop_reason,
        findings=len(pairs),
        by_status=dict(sorted(Counter(p.finding.status for p in pairs).items())),
        by_severity=dict(sorted(Counter(p.finding.severity for p in pairs).items())),
        labels_a=label_counts(p.a for p in pairs),
        labels_b=label_counts(p.b for p in pairs),
        agreement=measure_agreement([p.a for p in pairs], [p.b for p in pairs]),
        precision=_precision_block(pairs),
        precision_ex_kr=_precision_block([p for p in pairs if p.finding.family != KR_FAMILY]),
        by_status_precision={
            s: _precision_block([p for p in pairs if p.finding.status == s]) for s in statuses
        },
        families=families,
        holds=sum(len(r.diagnosis.holds) for r in results),
        holds_backtranslation=sum(1 for r in results for h in r.diagnosis.holds if h.backtranslation),
        backtranslation_skipped_documents=sum(
            1 for r in results if r.diagnosis.backtranslation == BACKTRANSLATION_SKIPPED
        ),
        backtranslation_note=BACKTRANSLATION_SKIPPED,
        llm_calls_by_task=dict(sorted(tasks.items())),
        runbooks_partial_extraction=sum(1 for r in results if r.diagnosis.extract_failed_chunks),
        extract_failed_chunks=sum(r.diagnosis.extract_failed_chunks for r in results),
        extraction_items=sum(_llm_items(r.diagnosis) for r in results),
        grounding_dropped=sum(_dropped(r.diagnosis) for r in results),
        spent_usd_total=round(ledger_usd, 6) if ledger_usd is not None else results_usd,
        spent_usd_results=results_usd,
        stop_kind=stop_kind,
        spend_this_run=spend,
        review=ReviewInfo(
            size=len(sample),
            seed=config.seed,
            by_family=dict(sorted(Counter(row.family for row in sample).items())),
        ),
        rubric=RUBRIC_VERSION,
        questions=_questions(pairs),
        jurisdictions=config.jurisdictions,
        split=config.split,
        split_digest=assignment.digest() if assignment is not None else None,
        split_rule=assignment.rule if assignment is not None else None,
    )


# --------------------------------------------------------------------------- review sample


class ReviewRow(_Model):
    review_id: str
    finding_key: str
    doc_path: str
    source_url: str
    line: int | None
    rule_id: str
    family: str
    severity: str
    status: str
    title: str
    message: str
    quote: str
    human_label: str = ""
    human_reason: str = ""


REVIEW_COLUMNS: Final = tuple(ReviewRow.model_fields)


def _allocate(sizes: dict[str, int], size: int) -> dict[str, int]:
    """Proportional allocation (largest remainder), at least one per family when it fits."""
    total = sum(sizes.values())
    if total <= size:
        return dict(sizes)
    families = sorted(sizes)
    base = {f: min(sizes[f], 1) for f in families} if size >= len(families) else dict.fromkeys(families, 0)
    remaining = size - sum(base.values())
    capacity = {f: sizes[f] - base[f] for f in families}
    capacity_total = sum(capacity.values())
    quotas = {f: remaining * capacity[f] / capacity_total if capacity_total else 0.0 for f in families}
    alloc = {f: base[f] + min(capacity[f], int(quotas[f])) for f in families}
    order = sorted(families, key=lambda f: (-(quotas[f] - int(quotas[f])), f))
    while sum(alloc.values()) < size:
        progressed = False
        for family in order:
            if sum(alloc.values()) >= size:
                break
            if alloc[family] < sizes[family]:
                alloc[family] += 1
                progressed = True
        if not progressed:
            break
    return alloc


def review_sample(results: Sequence[RunbookResult], *, size: int, seed: int) -> list[ReviewRow]:
    """A seeded sample stratified by rule family, presented in a seeded shuffled order.

    Deterministic: the same results, size and seed give the same rows. Coder labels are left out
    on purpose (blind review); join on ``finding_key`` with ``results.jsonl`` afterwards.
    """
    findings = sorted(
        ((r, f) for r in results for f in r.diagnosis.findings), key=lambda rf: (rf[1].family, rf[1].key)
    )
    by_family: dict[str, list[tuple[RunbookResult, FindingRecord]]] = defaultdict(list)
    for result, finding in findings:
        by_family[finding.family].append((result, finding))
    alloc = _allocate({f: len(v) for f, v in by_family.items()}, size)
    rng = random.Random(seed)
    chosen: list[tuple[RunbookResult, FindingRecord]] = []
    for family in sorted(by_family):
        chosen += rng.sample(by_family[family], alloc[family])
    rng.shuffle(chosen)
    rows: list[ReviewRow] = []
    for index, (result, finding) in enumerate(chosen, start=1):
        evidence = finding.evidence[0] if finding.evidence else None
        rows.append(
            ReviewRow(
                review_id=f"R{index:03d}",
                finding_key=finding.key,
                doc_path=result.doc_path,
                source_url=(evidence.url if evidence is not None and evidence.url else result.source_url),
                line=evidence.source_line if evidence is not None else None,
                rule_id=finding.rule_id,
                family=finding.family,
                severity=finding.severity,
                status=finding.status,
                title=finding.title,
                message=finding.message,
                quote=evidence.quote if evidence is not None else "",
            )
        )
    return rows


def review_csv(rows: Sequence[ReviewRow]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(REVIEW_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: ("" if v is None else v) for k, v in row.model_dump().items()})
    return buffer.getvalue()


def _human_labelled(out: Path) -> bool:
    """Does an existing review file already carry a human label or reason?"""
    try:
        with (out / "review.csv").open(encoding="utf-8-sig", newline="") as handle:
            if any(
                (row.get("human_label") or "").strip() or (row.get("human_reason") or "").strip()
                for row in csv.DictReader(handle)
            ):
                return True
    except (OSError, UnicodeDecodeError, csv.Error):
        pass
    try:
        rows = json.loads((out / "review.json").read_text(encoding="utf-8")).get("rows", [])
    except (OSError, UnicodeDecodeError, ValueError, AttributeError):
        return False
    return any(
        isinstance(row, dict)
        and (str(row.get("human_label") or "").strip() or str(row.get("human_reason") or "").strip())
        for row in rows
    )


def write_outputs(
    results: Sequence[RunbookResult], metrics: RunbookMetrics, config: RunbookStudyConfig
) -> str | None:
    """Write metrics, review sample and report. A review file that already carries human labels is
    kept; the sample of these results then goes to ``review.next.*`` (returns a note saying so)."""
    out = config.out_dir
    rows = review_sample(results, size=config.review_size, seed=config.seed)
    write_json(out / "metrics.json", metrics.model_dump(mode="json"))
    keep = _human_labelled(out)
    stem = "review.next" if keep else "review"
    write_json(
        out / f"{stem}.json",
        {
            "instructions": REVIEW_INSTRUCTIONS,
            "seed": config.seed,
            "rows": [row.model_dump(mode="json") for row in rows],
        },
    )
    # UTF-8 with BOM so spreadsheet programs open the Korean columns correctly
    write_text(out / f"{stem}.csv", review_csv(rows), encoding="utf-8-sig")
    write_text(out / "report.md", render_report(metrics))
    if not keep:
        return None
    return (
        "사람 판정이 적힌 review.csv/review.json은 덮어쓰지 않았습니다. "
        "지금 결과로 다시 뽑은 표본은 review.next.csv/review.next.json에 있습니다(같은 시드)."
    )


# --------------------------------------------------------------------------- report


def _rate(rate: Rate) -> str:
    if rate.rate is None or rate.ci95 is None:
        return "— (0건)"
    lo, hi = rate.ci95
    return f"{100 * rate.rate:.1f}% ({rate.hits}/{rate.n}, 95% CI {100 * lo:.1f}–{100 * hi:.1f}%)"


def _num(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


_BACKTRANSLATION_EXPLAINED: Final = (
    "제품의 역번역 LLM 대조는 한국어 역서술 문장과 원문을 비교하므로 영어 원문에서는 묻지 않았고, "
    "결정적 슬롯 대조(시간·채널·인물·부정어)만 적용했다(제품의 LLM 끔 동작과 같음)."
)
_RUNBOOK_LIMITS: Final = (
    "- **영어 문서.** 제품의 규칙 파서는 한국어용이라 영어 런북에서는 거의 기여하지 않고, "
    "그래프는 사실상 LLM 추출이 만든다. 역번역 LLM 대조도 생략했다(위 참고).",
    "- **명부·카탈로그 없음, 정책은 기본값(관할만 원 회사 기준).** 사람 관련 검사는 설계대로 "
    "'확인 불가'가 되고, 등급을 모르는 서비스는 기본 등급(핵심)으로 취급되어 5요소를 모두 요구받는다.",
    "- **외부 위임.** 문서가 담당·상향·호출 같은 요소를 서비스 카탈로그나 호출 도구 설정으로 보낸다고 "
    "적었으면 제품은 그 요소를 '없음'이 아니라 '확인 불가'로 낸다(그 자료가 입력에 없으므로).",
    "- **특정 서비스가 없는 알람 런북.** 경보를 낸 서비스 어디에나 적용된다고만 적혔거나 서비스가 하나도 "
    "적히지 않은 알람 런북은 검사할 서비스가 없어 5요소를 검사하지 않고 메모로 남긴다"
    "(검사 대상 없음 ≠ 정상). 그런 문서가 외부 자료로 넘긴 요소만 서비스 없는 '확인 불가' 지적으로 낸다. "
    "서비스가 하나도 남지 않은 문서는 언제나 그 이유를 메모로 남긴다.",
    "- **정책이 정한 전제.** 등급이 적히지 않은 서비스를 핵심 등급으로 보는 것, 롤백·페일오버에 승인이 "
    "필요하다는 것, 관할은 정책의 기본값이다. 그런 전제에 기댄 지적은 메시지에 '정책(…) 기준'이라고 밝히고, "
    "코더는 그 전제를 주어진 것으로 보고 문서 내용만 판정했다(루브릭 v4).",
    "- **성숙한 SRE 조직.** GitLab은 호출·상향을 incident.io·PagerDuty 같은 별도 시스템 설정으로 "
    "운영한다. 런북 문서에 적히지 않은 것을 '끊김'으로 보는 것은 문서 수준 판정이며, "
    "실제 조직의 대응 능력 판정이 아니다.",
    "- **정밀도는 재현율이 아니다.** 코더는 제품이 낸 지적만 판정했다. "
    "제품이 놓친 끊김(미탐)은 이 실증으로 측정되지 않는다.",
    "- **코더도 LLM이다.** 두 코더는 서로 다른 계열이지만 사람이 아니다. "
    "코더 A는 주 모델과 같은 회사 모델일 수 있다(자기 선호 가능성). "
    "사람 검토 표본으로 코더 판정을 보정해야 한다.",
)


def _precision_row(label: str, block: PrecisionBlock) -> str:
    return f"| {label} | {_rate(block.coder_a)} | {_rate(block.coder_b)} | {_rate(block.agreed)} |"


def _labels_line(name: str, counts: LabelCounts) -> str:
    return (
        f"- {name} 판정: correct {counts.correct}, incorrect {counts.incorrect}, "
        f"unclear {counts.unclear}, 판정 없음 {counts.missing}"
    )


def _jurisdiction_line(m: RunbookMetrics) -> str:
    codes = ", ".join(m.jurisdictions) or "—"
    if "KR" in m.jurisdictions:
        return f"- 관할(정책): {codes} — 한국 법정 신고 규칙(JT-KR-*)을 적용했다"
    return (
        f"- 관할(정책): {codes} — 원 회사의 관할로 진단해 한국 법정 신고 규칙(JT-KR-*, 침해사고 24시간·"
        "개인정보 72시간)은 적용하지 않았다"
    )


def _split_line(m: RunbookMetrics) -> str:
    if m.split_digest is None:
        return f"- 평가 분할: {m.split} (분할 파일 없음) — 최종 보고 수치는 test 분할로만 낸다"
    head = f"- 평가 분할: **{m.split}** (분할 해시 {m.split_digest[:12]}) — 규칙: {m.split_rule}"
    if m.split == "test":
        return head
    return head + " — 이 결과는 최종 보고 수치가 아니다(최종 수치는 test 분할로만 낸다)"


def _answers(counts: AnswerCounts) -> str:
    return f"yes {counts.yes} · no {counts.no} · unclear {counts.unclear} · 없음 {counts.missing}"


def _question_lines(q: QuestionsBlock | None) -> list[str]:
    if q is None:
        return []
    return [
        "## 판정 질문별 (전제·주장)",
        "",
        "코더는 지적마다 두 질문에 따로 답했다: ① 전제 — 지적이 다루는 서비스·요소를 문서가 정말 다루는가, "
        "② 주장 — 문서에 비추어 그 공백이 정말 있는가. 판정은 둘 다 yes면 correct, 하나라도 no면 incorrect, "
        "그 밖은 unclear로 만든다.",
        "",
        "| 질문 | 코더 A | 코더 B | 일치 n | κ | α |",
        "|---|---|---|---:|---|---|",
        f"| ① 전제 | {_answers(q.premise_a)} | {_answers(q.premise_b)} | {q.agreement_premise.n} | "
        f"{_num(q.agreement_premise.cohen_kappa)} | {_num(q.agreement_premise.krippendorff_alpha)} |",
        f"| ② 주장 | {_answers(q.claim_a)} | {_answers(q.claim_b)} | {q.agreement_claim.n} | "
        f"{_num(q.agreement_claim.cohen_kappa)} | {_num(q.agreement_claim.krippendorff_alpha)} |",
        "",
        "| 지표 | 코더 A | 코더 B | 두 코더 일치 판정 |",
        "|---|---|---|---|",
        _precision_row("전제 오류율 = no / (yes + no)", q.premise_error),
        _precision_row("전제가 맞는 지적의 정밀도 = 주장 yes / (yes + no)", q.precision_valid_premise),
        "",
    ]


def render_report(m: RunbookMetrics) -> str:
    """Korean ``report.md``; every sentence is generated from the numbers in ``metrics.json``."""
    agreement = m.agreement
    this_run = m.spend_this_run.spent_usd if m.spend_this_run is not None else 0.0
    lines = [
        "# 실증 ① — 공개 실무 런북(GitLab) 진단 결과의 코더 판정",
        "",
        f"- 생성일: {m.generated_at or '—'} · 원자료: {m.source} (MIT)",
        f"- 주 모델: {m.model} (effort {m.effort}) · 코더 A: {m.coder_a} · 코더 B: {m.coder_b}",
        _jurisdiction_line(m),
        _split_line(m),
        f"- 판정 방식: {m.rubric} (전제·주장 두 질문, 판정은 두 답에서 유도)",
        f"- 런북: {m.runbooks_done}/{m.runbooks_total}건 완료"
        + (f", 측정 실패 {m.runbooks_failed}건" if m.runbooks_failed else "")
        + (" — **부분 결과**" if m.partial else ""),
    ]
    if m.stop_reason:
        lines.append(f"- 중단 사유: {m.stop_reason}")
    lines += [
        f"- 지적: {m.findings}건 (상태별 {m.by_status}, 심각도별 {m.by_severity})",
        "- 실제 지출(이 결과 폴더 누적 — 모든 실행, 실패·중단 항목 포함, spend.json): "
        f"${m.spent_usd_total:.4f} · 결과에 든 런북 {m.runbooks_done}건의 비용 "
        f"${m.spent_usd_results:.4f} · 이번 실행 ${this_run:.4f}",
        "",
        "## 코더 일치도 (유도된 판정)",
        "",
        f"- 두 코더가 모두 판정한 지적 {agreement.n}건: 단순 일치율 {_num(agreement.observed)}, "
        f"Cohen's κ {_num(agreement.cohen_kappa)}, "
        f"Krippendorff's α {_num(agreement.krippendorff_alpha)}",
        _labels_line("코더 A", m.labels_a),
        _labels_line("코더 B", m.labels_b),
        "",
        "## 정밀도 = correct / (correct + incorrect)",
        "",
        "| 범위 | 코더 A | 코더 B | 두 코더 일치 판정 |",
        "|---|---|---|---|",
        _precision_row("전체", m.precision),
        _precision_row("한국 법정 신고(KR) 제외", m.precision_ex_kr),
        *(_precision_row(f"상태 {status}", block) for status, block in m.by_status_precision.items()),
        "",
        *_question_lines(m.questions),
        "## 규칙 계열별",
        "",
        "| 계열 | 지적 | 코더 A 정밀도 | 코더 B 정밀도 | 일치 판정 정밀도 | κ | κ(전제) | κ(주장) |",
        "|---|---:|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {f.family} ({f.label_ko}) | {f.findings} | {_rate(f.precision.coder_a)} | "
        f"{_rate(f.precision.coder_b)} | {_rate(f.precision.agreed)} | {_num(f.agreement.cohen_kappa)} | "
        f"{_num(f.agreement_premise.cohen_kappa if f.agreement_premise else None)} | "
        f"{_num(f.agreement_claim.cohen_kappa if f.agreement_claim else None)} |"
        for f in m.families
    ]
    lines += [
        "",
        "## 판단 보류·역번역",
        "",
        f"- 판단 보류(HOLD) 요소 {m.holds}개, 그중 역번역 대조 불일치 {m.holds_backtranslation}개",
        f"- 한국어가 아닌 원문 {m.backtranslation_skipped_documents}건: `{m.backtranslation_note}` — "
        + _BACKTRANSLATION_EXPLAINED,
        f"- LLM 호출(작업별): {m.llm_calls_by_task}",
        f"- LLM 추출 일부 구간 실패(거절·잘림·형식 오류): 런북 {m.runbooks_partial_extraction}건, "
        f"구간 {m.extract_failed_chunks}개 — 제품 동작대로 그 구간은 규칙 파서 결과만으로 진단했다"
        "(영어 런북에서는 그 구간의 요소가 '없음'으로 지적될 수 있다).",
        f"- 추출 기록({RECORD_VERSION}): LLM이 제안한 항목 {m.extraction_items}개, 조립 단계에서 버린 요소 "
        f"{m.grounding_dropped}개(근거 없음·격리 구간·허용되지 않는 관계·명령/출력 안의 이름). 런북마다 "
        "results.jsonl의 diagnosis.extraction(구간별 추출 IR과 인용의 줄), diagnosis.grounding(탈락 사유 "
        "예시), diagnosis.graph(조립된 그래프 요약)에 남겼다.",
        "",
        "## 사람 검토 표본",
        "",
        f"- review.csv / review.json: {m.review.size}건, 시드 {m.review.seed}, "
        f"규칙 계열별 층화 {m.review.by_family}. 코더 판정은 숨김(블라인드) — "
        "검토 후 finding_key로 results.jsonl과 대조한다.",
        "",
        "## 한계 (수치와 함께 읽을 것)",
        "",
        *_RUNBOOK_LIMITS,
        f"- **작은 표본.** 런북 {m.runbooks_done}건 — 신뢰구간(Wilson 95%)을 함께 읽어야 한다.",
    ]
    return "\n".join(lines) + "\n"
