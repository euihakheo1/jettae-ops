"""실문서 결함 주입(축소판): GitLab 런북 한국어판에서 대응 요소 진술을 지우고 제품이 알아채는지 잰다.

Context. The competition application promised a sealed defect-injection set whose base runbooks are
adapted public runbooks, with pre-existing defects kept out of the denominator ("기존 결함은 분모에서
분리"). The sealed set that was actually built uses generator organisations. This is the reduced
real-text version: the deletion sites an AI agent annotated in the Korean translation of the GitLab.com
runbooks (study ①'s test split) and another AI agent checked (``annotations_checked.json``). The
pre-registration is ``docs/eval/real-inject-plan.md``.

What happens (product code only, frozen at ccf3b4d; nothing under src/ is changed):

1. BASELINE = the translated runbook as published in the translation folder (sha256 checked against
   the annotation). MUTATED = the same file with one site's lines blanked: newline characters kept,
   so line count and line numbers are unchanged; the blanked text is checked against the annotation's
   ``deleted_text`` and ``deleted_text_sha256``. Both are loaded like an upload
   (:func:`jettae.ingest.load_bytes`) under the repository-relative path, as study ① loads runbooks.
2. Condition **B** (free): the evalkit condition-B path,
   :func:`jettae.evalkit.conditions.offline_graph` (guard → RuleExtractor → assemble → slot-only
   back-translation) → fixed rules (:func:`jettae.check.registry.run_checks`), no roster / catalog,
   study ①'s policy (defaults, jurisdiction US). Cross-check: the product pipeline with the LLM off
   (:func:`jettae.orchestrator.pipeline.run_diagnosis`, ``LlmMode.OFF``, drills and drafts off) must
   give the same (service, rule) set; a difference is reported, B stays the primary result.
3. Condition **DR** (paid, only with ``--llm --yes --max-usd``): study ①'s path,
   :func:`jettae.evalkit.realdata.diagnose.diagnose_bundle` (product pipeline in LLM mode: rule parser
   + LLM extraction, the back-translation judge for Korean text), ``claude-sonnet-5-5`` effort medium,
   behind the real-data budget gate. No coders: scoring is automatic. Without ``--yes`` the script
   prints the estimate and stops before any call. The key is loaded only inside the paid path
   (:func:`jettae.evalkit.realdata.keys.load_api_key`) and never printed.
4. Scoring (see the plan): a site is DETECTED when the expected rule fires for the site's service in
   the mutated run and did not fire for it in the baseline run. Sites whose expected finding already
   fires in the baseline, or that the annotation marks as a pre-existing defect, are PRE-EXISTING and
   left out of the denominator. Everything else is a miss (an unrecognised service and a run error
   are misses with their own reason). Side effects: new / vanished (service, rule) findings of the
   mutated run that are not the site's target; baseline findings per document.

Usage (repository root, product venv)::

    python -X utf8 scripts/eval_real_inject.py                          # free: B
    python -X utf8 scripts/eval_real_inject.py --llm                    # prints the DR estimate, refuses
    python -X utf8 scripts/eval_real_inject.py --llm --yes --max-usd 2  # PAID: DR

Outputs: ``<out>/results.json`` and ``<out>/report.md`` (default ``C:/jettae/data/eval/real-inject``);
the paid arm also writes ``<out>/llm/results.jsonl``, ``<out>/llm/errors.jsonl`` (extractions the product
fell back from; retried on the next run) and ``<out>/llm/spend.json`` (the cap is cumulative over that
folder). Results of an earlier paid run in that folder are read back into the report.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from jettae.check.context import CheckContext
from jettae.check.registry import rule_catalog, run_checks
from jettae.domain import Document, Finding, InputBundle, LlmMode, NodeKind, OpsGraph, Policy
from jettae.evalkit.conditions import offline_graph, prepare_worker, run_sync
from jettae.evalkit.metrics import Rate
from jettae.evalkit.realdata.common import is_korean_text, write_json, write_text
from jettae.evalkit.realdata.diagnose import DiagnosisRecord, PipelineFallbackError, diagnose_bundle
from jettae.evalkit.realdata.dryrun import claude_fallback_counter
from jettae.evalkit.realdata.pricing import (
    ASSUMED_BACKTRANSLATIONS_PER_DOC,
    CLAUDE_THINKING_TOKENS,
    PRICES_AS_OF,
    ModelRef,
    call_usd,
    price_of,
)
from jettae.evalkit.realdata.runbooks import DEFAULT_JURISDICTIONS, RunbookSource, plan_runbook_calls
from jettae.ingest import load_bytes
from jettae.llm.base import LLMError, NullLLM
from jettae.orchestrator.pipeline import run_diagnosis
from jettae.orchestrator.state import DiagnosisOptions

ROOT: Final = Path(__file__).resolve().parents[1]
FROZEN_COMMIT: Final = "ccf3b4d"
DEFAULT_ANNOTATIONS: Final = Path("C:/jettae/data/eval/real-inject/annotations_checked.json")
DEFAULT_OUT: Final = Path("C:/jettae/data/eval/real-inject")
CHECKED_SCHEMA: Final = "jettae-real-inject-annotations-checked/v1"
PLAN_DOC: Final = "docs/eval/real-inject-plan.md"
#: The earlier Korean study-① run (LLM on): its per-document Claude spend is a measured reference.
PRIOR_KO_RUN: Final = Path("C:/jettae/data/real/results/gitlab-test-ko")
#: Output folders other (paid) runs are writing to — never written here.
PROTECTED_DIRS: Final = ("llm-sample20", "patch-recheck", "paraphrase")
#: Study ①'s extraction model and effort (``gitlab-test-ko/run.json``).
DEFAULT_MODEL: Final = "claude-sonnet-5-5"
DEFAULT_EFFORT: Final = "medium"
#: Condition labels.
B: Final = "B"
DR: Final = "DR"
CONDITION_LABEL: Final = {
    B: "B 규칙 파서 + 그래프 (제품 LLM 끔 경로, 무료)",
    DR: "DR 규칙 파서 + LLM 추출 (실증 ①과 같은 제품 LLM 켬 경로, 유료)",
}
OPERATORS: Final = (
    "delete_owner",
    "delete_escalation",
    "delete_approver",
    "delete_rollback",
    "delete_fallback_channel",
)
OPERATOR_KO: Final = {
    "delete_owner": "담당 삭제",
    "delete_escalation": "상향 삭제",
    "delete_approver": "승인자 삭제",
    "delete_rollback": "완화 조치 삭제",
    "delete_fallback_channel": "대체 채널 삭제",
}
SEVERITY_KO: Final = {"critical": "치명", "high": "높음", "medium": "보통", "low": "낮음", "info": "정보"}
OUTCOME_KO: Final = {
    "detected": "탐지",
    "missed": "놓침",
    "preexisting": "기존 결함(분모 제외)",
    "error": "실행 오류(놓침으로 셈)",
}
MISS_REASON_KO: Final = {
    "rule_not_fired": "서비스는 잡았지만 기대 규칙이 뜨지 않음",
    "service_not_recognised": "서비스 미인식(원본·변형 모두)",
    "service_lost_in_mutant": "변형에서 서비스가 사라짐",
    "run_error": "실행 오류",
}
SERVICE_KINDS: Final = frozenset({NodeKind.SERVICE.value, NodeKind.SYSTEM.value})
_QUOTES: Final = "`'\"“”‘’"
_HEAD: Final = re.compile(r"[\s(（]")


class InputError(RuntimeError):
    """Annotation and source disagree: nothing is measured or written."""


# =========================================================================== small helpers


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256_hex(data: bytes | str) -> str:
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(raw).hexdigest()


def norm_name(name: str) -> str:
    """Service-name comparison key: case-insensitive, surrounding blanks / backticks / quotes ignored."""
    return name.strip().strip(_QUOTES).strip().casefold()


def head_name(label: str) -> str:
    """First word of an annotation's service label ("GitLab.com 애플리케이션 코드(…)" → "GitLab.com")."""
    return _HEAD.split(label.strip(), maxsplit=1)[0]


def _pct(rate: Mapping[str, Any]) -> str:
    if not rate.get("n"):
        return "측정 불가(분모 0)"
    lo, hi = rate["ci95"]
    return (
        f"{rate['hits']}/{rate['n']} = {rate['rate'] * 100:.0f}% (Wilson 95% {lo * 100:.0f}–{hi * 100:.0f}%)"
    )


def _clip(text: str, limit: int = 160) -> str:
    one = " ".join(text.split())
    return one if len(one) <= limit else one[: limit - 1] + "…"


def _md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _git(*args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def product_provenance() -> dict[str, Any]:
    src_clean = _git("diff", "--quiet", FROZEN_COMMIT, "--", "src/")
    return {
        "frozen_commit": FROZEN_COMMIT,
        "checkout_head": _git("rev-parse", "--short", "HEAD"),
        "src_identical_to_frozen_commit": src_clean is not None,
    }


def guard_out_dir(out: Path) -> None:
    parts = [p.casefold() for p in out.resolve().parts]
    for name in PROTECTED_DIRS:
        if any(p.startswith(name) for p in parts):
            raise SystemExit(f"다른 유료 실행이 쓰는 폴더라 쓰지 않습니다: {out.as_posix()}")


# =========================================================================== annotations → sites


@dataclass(frozen=True)
class Site:
    site_id: str
    doc_path: str
    doc_sha256: str
    operator: str
    lines: tuple[int, ...]
    expected_rule: str
    service_label: str
    #: normalised service names the site is about (see :func:`site_aliases`)
    aliases: tuple[str, ...]
    annotated_preexisting: bool
    deleted_text: tuple[str, ...]
    deleted_text_sha256: str
    verdict_reason: str


def site_aliases(raw: Mapping[str, Any]) -> tuple[str, ...]:
    """The site's service names, fixed before the run (plan): the first word of the annotation's
    ``service`` label plus the product's names in ``prior_ko_run_hint.services_in_prior_findings``."""
    names = [head_name(str(raw["service"]))]
    hint = raw.get("prior_ko_run_hint") or {}
    names += [str(n) for n in hint.get("services_in_prior_findings") or ()]
    out: list[str] = []
    for name in names:
        key = norm_name(name)
        if key and key not in out:
            out.append(key)
    return tuple(out)


def load_annotations(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if data.get("checked_schema") != CHECKED_SCHEMA:
        raise InputError(
            f"점검된 주석 파일이 아닙니다(checked_schema={data.get('checked_schema')!r}): {path}"
        )
    return data


def kept_sites(annotations: Mapping[str, Any]) -> list[Site]:
    sites: list[Site] = []
    for raw in annotations["sites"]:
        if raw.get("verdict") != "keep":
            continue
        sites.append(
            Site(
                site_id=raw["site_id"],
                doc_path=raw["doc_path"],
                doc_sha256=raw["doc_sha256"],
                operator=raw["operator"],
                lines=tuple(int(n) for n in raw["lines"]),
                expected_rule=raw["expected_rule"],
                service_label=raw["service"],
                aliases=site_aliases(raw),
                annotated_preexisting=bool(raw.get("ideal_base_preexisting"))
                or raw.get("denominator") == "excluded_preexisting",
                deleted_text=tuple(raw["deleted_text"]),
                deleted_text_sha256=raw["deleted_text_sha256"],
                verdict_reason=raw.get("verdict_reason_ko", ""),
            )
        )
    if not sites:
        raise InputError("verdict=keep인 지점이 없습니다.")
    return sites


def dropped_sites(annotations: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "site_id": s["site_id"],
            "doc_path": s["doc_path"],
            "operator": s["operator"],
            "expected_rule": s["expected_rule"],
            "reason_ko": s.get("verdict_reason_ko", ""),
        }
        for s in annotations["sites"]
        if s.get("verdict") != "keep"
    ]


# =========================================================================== variants


@dataclass(frozen=True)
class Variant:
    """One document the product is run on: a baseline (``site_id`` None) or one site's mutation."""

    variant_id: str
    doc_path: str
    site_id: str | None
    raw: str
    document: Document

    @property
    def sha256(self) -> str:
        return sha256_hex(self.raw)


def baseline_id(doc_path: str) -> str:
    return f"BASE::{doc_path}"


def blank_lines(raw: str, site: Site) -> str:
    """``raw`` with the site's lines emptied (newlines kept); checked against the annotation."""
    lines = raw.split("\n")
    removed: list[str] = []
    for number in site.lines:
        if not 1 <= number <= len(lines):
            raise InputError(f"{site.site_id}: 줄 {number}이 문서 범위(1~{len(lines)}) 밖입니다.")
        line = lines[number - 1]
        cr = line.endswith("\r")
        removed.append(line[:-1] if cr else line)
        lines[number - 1] = "\r" if cr else ""
    if tuple(removed) != site.deleted_text:
        raise InputError(f"{site.site_id}: 지울 줄이 주석의 deleted_text와 다릅니다.")
    joined = "\n".join(removed)
    if site.deleted_text_sha256 not in (sha256_hex(joined), sha256_hex(joined + "\n")):
        raise InputError(f"{site.site_id}: deleted_text_sha256이 맞지 않습니다.")
    mutated = "\n".join(lines)
    if mutated.count("\n") != raw.count("\n"):
        raise InputError(f"{site.site_id}: 변형 뒤 줄 수가 달라졌습니다.")
    return mutated


def build_variants(sites: Sequence[Site], korean_root: Path) -> list[Variant]:
    """Every baseline document (once, path order), each followed by its sites' mutations."""
    by_doc: dict[str, list[Site]] = defaultdict(list)
    for site in sites:
        by_doc[site.doc_path].append(site)
    variants: list[Variant] = []
    for doc_path in sorted(by_doc):
        data = (korean_root / doc_path).read_bytes()
        expected = {s.doc_sha256 for s in by_doc[doc_path]}
        if expected != {sha256_hex(data)}:
            raise InputError(f"{doc_path}: 원본 파일 sha256이 주석과 다릅니다(번역판이 바뀌었을 수 있음).")
        raw = data.decode("utf-8")
        variants.append(Variant(baseline_id(doc_path), doc_path, None, raw, load_bytes(doc_path, data)))
        for site in sorted(by_doc[doc_path], key=lambda s: s.site_id):
            mutated = blank_lines(raw, site)
            document = load_bytes(doc_path, mutated.encode("utf-8"))
            variants.append(Variant(site.site_id, doc_path, site.site_id, mutated, document))
    return variants


# =========================================================================== diagnoses


@dataclass(frozen=True)
class FindingRow:
    rule_id: str
    severity: str
    status: str
    service_id: str | None
    service_name: str | None
    element: str | None
    title: str
    message: str
    finding_id: str
    lines: tuple[int, ...]

    def key(self) -> tuple[str, str]:
        """(service, rule) — service by normalised name, else id, else '' (organisation level)."""
        if self.service_name:
            return norm_name(self.service_name), self.rule_id
        return (self.service_id or ""), self.rule_id

    def as_json(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "status": self.status,
            "service_id": self.service_id,
            "service_name": self.service_name,
            "element": self.element,
            "title": self.title,
            "message": self.message,
            "finding_id": self.finding_id,
            "lines": list(self.lines),
        }


@dataclass(frozen=True)
class Diag:
    condition: str
    variant_id: str
    extractor: str
    graph_hash: str | None
    nodes: int
    edges: int
    #: (node id, kind, name, grounded lines) of every node of the graph
    node_list: tuple[tuple[str, str, str, tuple[int, ...]], ...]
    findings: tuple[FindingRow, ...]
    llm_calls: int = 0
    error: str | None = None

    @property
    def services(self) -> tuple[tuple[str, str, str], ...]:
        """(node id, kind, name) of the service and system nodes."""
        return tuple((i, k, n) for i, k, n, _lines in self.node_list if k in SERVICE_KINDS)

    def keys(self) -> set[tuple[str, str]]:
        return {f.key() for f in self.findings}

    def as_json(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "variant_id": self.variant_id,
            "extractor": self.extractor,
            "graph_hash": self.graph_hash,
            "nodes": self.nodes,
            "edges": self.edges,
            "graph_nodes": [
                {"id": i, "kind": k, "name": n, "lines": list(ls)} for i, k, n, ls in self.node_list
            ],
            "findings": [f.as_json() for f in self.findings],
            "llm_calls": self.llm_calls,
            "error": self.error,
        }


def _graph_nodes(graph: OpsGraph) -> tuple[tuple[str, str, str, tuple[int, ...]], ...]:
    return tuple(
        sorted(
            (n.id, n.kind.value, n.name, tuple(sorted({s.line for s in n.spans if s.line is not None})))
            for n in graph.nodes.values()
        )
    )


def diag_from_graph(
    condition: str, variant_id: str, graph: OpsGraph, findings: Iterable[Finding], extractor: str
) -> Diag:
    rows: list[FindingRow] = []
    for f in findings:
        node = graph.nodes.get(f.service_id) if f.service_id is not None else None
        rows.append(
            FindingRow(
                rule_id=f.rule_id,
                severity=f.severity.value,
                status=f.status.value,
                service_id=f.service_id,
                service_name=node.name if node is not None else None,
                element=f.element.value if f.element is not None else None,
                title=f.title,
                message=f.message,
                finding_id=f.id,
                lines=tuple(sorted({s.line for s in f.evidence if s.line is not None})),
            )
        )
    return Diag(
        condition=condition,
        variant_id=variant_id,
        extractor=extractor,
        graph_hash=graph.content_hash(),
        nodes=len(graph.nodes),
        edges=len(graph.edges),
        node_list=_graph_nodes(graph),
        findings=tuple(sorted(rows, key=lambda r: (r.rule_id, r.service_name or "", r.finding_id))),
    )


def diag_from_record(variant_id: str, record: DiagnosisRecord) -> Diag:
    graph = record.graph
    node_list = (
        tuple(sorted((n.id, n.kind, n.name, tuple(n.lines)) for n in graph.nodes))
        if graph is not None
        else ()
    )
    rows = [
        FindingRow(
            rule_id=f.rule_id,
            severity=f.severity,
            status=f.status,
            service_id=f.service_id,
            service_name=f.service_name,
            element=f.element,
            title=f.title,
            message=f.message,
            finding_id=f.finding_id,
            lines=tuple(sorted({e.line for e in f.evidence if e.line is not None})),
        )
        for f in record.findings
    ]
    return Diag(
        condition=DR,
        variant_id=variant_id,
        extractor=record.extractor,
        graph_hash=record.graph_hash,
        nodes=record.nodes,
        edges=record.edges,
        node_list=node_list,
        findings=tuple(sorted(rows, key=lambda r: (r.rule_id, r.service_name or "", r.finding_id))),
        llm_calls=record.llm_calls,
    )


def study_policy() -> Policy:
    """Study ①'s policy: defaults except the jurisdiction (``RunbookStudyConfig.policy()``)."""
    return Policy(jurisdictions=DEFAULT_JURISDICTIONS)


def run_b(variant: Variant, policy: Policy) -> Diag:
    """Condition B exactly as the evalkit harness runs it (offline_graph → run_checks), no roster."""
    try:
        graph = offline_graph((variant.document,), policy=policy)
        findings = run_checks(CheckContext(graph=graph, policy=policy))
    except Exception as exc:  # recorded on the result, scored as a miss
        return Diag(B, variant.variant_id, "rule", None, 0, 0, (), (), error=f"{type(exc).__name__}: {exc}")
    return diag_from_graph(B, variant.variant_id, graph, findings, "rule")


def run_product_off(variant: Variant, policy: Policy) -> Diag:
    """The product pipeline itself with the LLM off (drills and drafts off) — the B cross-check."""
    options = DiagnosisOptions(rehearse=False, draft=False, llm_scenario_count=0)
    bundle = InputBundle(documents=(variant.document,), policy=policy)
    outcome = run_sync(
        run_diagnosis(bundle, base_graph=None, llm=NullLLM(), llm_mode=LlmMode.OFF, options=options)
    )
    return diag_from_graph(
        "product_off", variant.variant_id, outcome.graph, outcome.findings, outcome.extractor
    )


def cross_check(b: Diag, product: Diag) -> dict[str, Any]:
    only_b = sorted(b.keys() - product.keys())
    only_p = sorted(product.keys() - b.keys())
    return {
        "variant_id": b.variant_id,
        "same_findings": not only_b and not only_p,
        "same_graph_hash": b.graph_hash == product.graph_hash,
        "only_in_b": [list(k) for k in only_b],
        "only_in_product_off": [list(k) for k in only_p],
    }


# =========================================================================== scoring


def _on_site(f: FindingRow, site: Site, ids: set[str]) -> bool:
    if f.rule_id != site.expected_rule:
        return False
    if f.service_name is not None and norm_name(f.service_name) in site.aliases:
        return True
    return f.service_id is not None and f.service_id in ids


def _recognised(diag: Diag, site: Site) -> bool:
    names = {norm_name(n) for _i, _k, n in diag.services}
    names |= {norm_name(f.service_name) for f in diag.findings if f.service_name}
    return bool(names & set(site.aliases))


def _signature(rows: Sequence[FindingRow]) -> list[tuple[str, str, str, str]]:
    return sorted((r.severity, r.status, r.finding_id, r.message) for r in rows)


def _brief(rows: Sequence[FindingRow]) -> list[dict[str, Any]]:
    return [
        {
            "service": r.service_name or r.service_id,
            "severity": r.severity,
            "status": r.status,
            "finding_id": r.finding_id,
            "message": r.message,
        }
        for r in rows
    ]


def _first(diag: Diag, key: tuple[str, str]) -> FindingRow | None:
    return next((f for f in diag.findings if f.key() == key), None)


def score_site(site: Site, base: Diag, mut: Diag) -> dict[str, Any]:
    """One site under one condition (plan: 채점)."""
    aliases = set(site.aliases)
    ids = {i for d in (base, mut) for i, _k, n in d.services if norm_name(n) in aliases}
    base_hits = [f for f in base.findings if _on_site(f, site, ids)]
    mut_hits = [f for f in mut.findings if _on_site(f, site, ids)]
    in_base, in_mut = _recognised(base, site), _recognised(mut, site)
    errored = base.error is not None or mut.error is not None
    miss_reason: str | None = None
    # Pre-existing is a fact of the baseline run alone (plan (가)/(나)): a failed *mutated* run must
    # not pull a site whose baseline already fires back into the denominator as an error-miss.
    # (An errored baseline has no findings, so ``base_hits`` is empty then.)
    if site.annotated_preexisting or base_hits:
        outcome = "preexisting"
    elif errored:
        outcome, miss_reason = "error", "run_error"
    elif mut_hits:
        outcome = "detected"
    else:
        outcome = "missed"
        if not in_base and not in_mut:
            miss_reason = "service_not_recognised"
        elif in_base and not in_mut:
            miss_reason = "service_lost_in_mutant"
        else:
            miss_reason = "rule_not_fired"
    target = {f.key() for f in (*base_hits, *mut_hits)}
    base_keys, mut_keys = base.keys(), mut.keys()
    new = sorted(mut_keys - base_keys - target)
    gone = sorted(base_keys - mut_keys - target)
    return {
        "site_id": site.site_id,
        "doc_path": site.doc_path,
        "operator": site.operator,
        "expected_rule": site.expected_rule,
        "service_label": site.service_label,
        "service_names": list(site.aliases),
        "outcome": outcome,
        "miss_reason": miss_reason,
        "preexisting_by_run": bool(base_hits),
        "preexisting_by_annotation": site.annotated_preexisting,
        "service_recognised": {"baseline": in_base, "mutated": in_mut},
        "expected_rule_in_baseline": _brief(base_hits),
        "expected_rule_in_mutated": _brief(mut_hits),
        "expected_finding_changed": _signature(base_hits) != _signature(mut_hits),
        "side_effects": {
            "new": [_side(mut, k) for k in new],
            "vanished": [_side(base, k) for k in gone],
        },
        "errors": {"baseline": base.error, "mutated": mut.error},
    }


def _side(diag: Diag, key: tuple[str, str]) -> dict[str, Any]:
    row = _first(diag, key)
    return {
        "service": key[0] or None,
        "rule_id": key[1],
        "severity": row.severity if row else None,
        "message": row.message if row else None,
    }


def _rate(hits: int, n: int) -> dict[str, Any]:
    return Rate.of(hits, n).model_dump(mode="json")


def summarise(scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counted = [s for s in scores if s["outcome"] != "preexisting"]
    detected = sum(s["outcome"] == "detected" for s in counted)
    recognised = [s for s in counted if s["miss_reason"] != "service_not_recognised"]
    by_operator: dict[str, Any] = {}
    for op in OPERATORS:
        ops = [s for s in scores if s["operator"] == op]
        if not ops:
            continue
        rows = [s for s in ops if s["outcome"] != "preexisting"]
        by_operator[op] = {
            "sites": len(ops),
            "preexisting": len(ops) - len(rows),
            "detection": _rate(sum(s["outcome"] == "detected" for s in rows), len(rows)),
        }
    pre = [s for s in scores if s["outcome"] == "preexisting"]
    return {
        "sites_kept": len(scores),
        "preexisting": {
            "total": len(pre),
            "by_baseline_run": sum(s["preexisting_by_run"] for s in pre),
            "by_annotation": sum(s["preexisting_by_annotation"] for s in pre),
            "by_both": sum(s["preexisting_by_run"] and s["preexisting_by_annotation"] for s in pre),
            "ids": [s["site_id"] for s in pre],
        },
        "denominator": len(counted),
        "detection": _rate(detected, len(counted)),
        "detection_without_unrecognised_service": _rate(
            sum(s["outcome"] == "detected" for s in recognised), len(recognised)
        ),
        "outcomes": dict(Counter(s["outcome"] for s in scores)),
        "miss_reasons": dict(Counter(s["miss_reason"] for s in scores if s["miss_reason"])),
        "by_operator": by_operator,
        "side_effects": {
            "new_findings": sum(len(s["side_effects"]["new"]) for s in scores),
            "vanished_findings": sum(len(s["side_effects"]["vanished"]) for s in scores),
        },
    }


def _node_text(kind: str, name: str, lines: Sequence[int]) -> str:
    where = f"({', '.join(map(str, lines))}행)" if lines else ""
    return f"{kind}:{name}{where}"


def evaluate_condition(sites: Sequence[Site], diags: Mapping[str, Diag]) -> dict[str, Any] | None:
    """Scores of every site whose baseline and mutation both have a diagnosis under this condition."""
    scores = [
        score_site(s, diags[baseline_id(s.doc_path)], diags[s.site_id])
        for s in sites
        if baseline_id(s.doc_path) in diags and s.site_id in diags
    ]
    if not scores:
        return None
    baselines = {}
    for vid, d in sorted(diags.items()):
        if vid.startswith("BASE::"):
            baselines[d.variant_id.removeprefix("BASE::")] = {
                "findings": len(d.findings),
                "by_rule": dict(sorted(Counter(f.rule_id for f in d.findings).items())),
                "services": [n for _i, _k, n in d.services],
                "graph_nodes": [_node_text(k, n, ls) for _i, k, n, ls in d.node_list],
                "error": d.error,
            }
    return {
        "summary": summarise(scores),
        "sites": scores,
        "baseline_findings_per_document": baselines,
        "sites_scored": len(scores),
        "sites_total": len(sites),
    }


# =========================================================================== DR: estimate and paid run


def _source(variant: Variant) -> RunbookSource:
    return RunbookSource(path=variant.doc_path, raw=variant.raw, document=variant.document)


def dr_estimate(variants: Sequence[Variant], model: str, effort: str) -> dict[str, Any]:
    """Free estimate of the DR arm: the requests study ① would send (its own planner, coders left
    out), input = characters / 1 (Claude fallback, errs high), output = the documented assumptions."""
    ref = ModelRef("anthropic", model)
    counter = claude_fallback_counter()
    rows: list[dict[str, Any]] = []
    for variant in variants:
        calls = plan_runbook_calls([_source(variant)], coder_batch=8, assumed_findings=0)
        usd = 0.0
        stages: Counter[str] = Counter()
        for call in calls:
            tokens_in = math.ceil(call.input_chars() / counter.chars_per_token)
            tokens_out = call.output_tokens(ref, effort)  # type: ignore[arg-type]
            usd += call_usd(model, input_tokens=tokens_in, output_tokens=tokens_out)
            stages[call.stage] += 1
        rows.append(
            {
                "variant_id": variant.variant_id,
                "korean": is_korean_text(variant.document.text),
                "calls": dict(stages),
                "usd": round(usd, 4),
            }
        )
    thinking = CLAUDE_THINKING_TOKENS[effort]  # type: ignore[index]
    return {
        "model": model,
        "effort": effort,
        "prices_as_of": PRICES_AS_OF,
        "price_per_mtok": [price_of(model).input_per_mtok, price_of(model).output_per_mtok],
        "diagnoses": len(rows),
        "per_diagnosis": rows,
        "usd_estimate": round(sum(r["usd"] for r in rows), 4),
        "assumptions_ko": [
            f"입력 토큰 = 요청 전체 글자 수 / {counter.chars_per_token:g} "
            f"({counter.method}, 한국어 기준으로 크게 잡음)",
            f"출력 = 추출은 청크 토큰 × 2 + 사고 토큰 {thinking:,}(effort {effort}), "
            "역번역 대조는 150 + 사고 토큰",
            f"한국어 문서마다 역번역 대조 {ASSUMED_BACKTRANSLATIONS_PER_DOC}회를 가정 "
            "(실제 횟수는 추출 뒤에야 앎)",
            "프롬프트 캐시 할인 없음, 코더 판정 없음(이 측정은 자동 채점)",
        ],
        "measured_reference": prior_run_reference(variants, model),
    }


def prior_run_reference(variants: Sequence[Variant], model: str) -> dict[str, Any] | None:
    """Measured Claude spend per document of the earlier Korean study-① run (extraction,
    back-translation *and* coder A, which used the same model — so an upper reference)."""
    path = PRIOR_KO_RUN / "spend.json"
    try:
        items = json.loads(path.read_text(encoding="utf-8"))["items"]
    except (OSError, ValueError, KeyError):
        return None
    per_doc: dict[str, float] = {}
    for doc_path in sorted({v.doc_path for v in variants}):
        lines = (items.get(doc_path) or {}).get("by_model") or []
        per_doc[doc_path] = round(sum(x["usd"] for x in lines if x.get("model") == model), 6)
    runs = Counter(v.doc_path for v in variants)
    return {
        "source": path.as_posix(),
        "claude_usd_per_document": per_doc,
        "usd_if_same_per_diagnosis": round(sum(per_doc[d] * runs[d] for d in per_doc), 4),
        "note_ko": "이전 실행의 Claude 지출(추출·역번역·코더 A 포함, 캐시 반영) × 진단 수. "
        "코더가 빠지므로 상한 참고값.",
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_dr(out: Path, variants: Sequence[Variant], model: str, effort: str) -> dict[str, Diag]:
    """DR diagnoses of earlier paid runs that match the current variants (same text, model, effort).

    A variant with a record uses it; one that only failed (``errors.jsonl``, the product fell back
    from its LLM extraction) is an error diagnosis, scored as a miss, and is sent again next run.
    """
    current = {v.variant_id: v.sha256 for v in variants}

    def matching(name: str) -> Iterable[dict[str, Any]]:
        for row in _read_jsonl(out / "llm" / name):
            same_run = row.get("model") == model and row.get("effort") == effort
            if same_run and current.get(str(row.get("variant_id"))) == row.get("sha256"):
                yield row

    found: dict[str, Diag] = {}
    for row in matching("errors.jsonl"):
        vid = row["variant_id"]
        found[vid] = Diag(DR, vid, "error", None, 0, 0, (), (), error=row["error"])
    for row in matching("results.jsonl"):
        vid = row["variant_id"]
        found[vid] = diag_from_record(vid, DiagnosisRecord.model_validate(row["record"]))
    return found


class SpendLedger:
    """``<out>/llm/spend.json``: every paid run of this folder; the cap covers their sum."""

    def __init__(self, path: Path, run: dict[str, Any]) -> None:
        self.path = path
        self.data: dict[str, Any] = {"runs": []}
        if path.is_file():
            self.data = json.loads(path.read_text(encoding="utf-8"))
        self.prior_usd = sum(float(r.get("spent_usd", 0.0)) for r in self.data["runs"])
        self.run = run
        self.data["runs"].append(run)

    def save(self, tracker: Any) -> None:
        """Calls in flight are booked at their reservation (they may be billed): if the process
        dies mid-call, the next run's cumulative cap must not under-count (as the product ledger)."""
        report = tracker.report()
        in_flight = tracker.reserved_usd
        self.run["spent_usd"] = round(report.spent_usd + in_flight, 6)
        self.run["in_flight_usd"] = round(in_flight, 6)
        self.run["by_model"] = [line.model_dump(mode="json") for line in report.by_model]
        self.data["spent_usd_total"] = round(self.prior_usd + self.run["spent_usd"], 6)
        write_json(self.path, self.data)


async def run_dr(
    variants: Sequence[Variant], out: Path, *, model: str, effort: str, max_usd: float, concurrency: int
) -> dict[str, Any]:
    """PAID: study ①'s diagnose_bundle on every variant not yet in ``<out>/llm/results.jsonl``.

    The budget cap (:class:`BudgetExhausted`) and an :class:`LLMError` that escapes the pipeline stop
    the run without recording the variant, so the next run sends it again. Note that the product
    itself turns a failed extraction request (credit, key, network, rate limit) into its rule-parser
    fallback, so such a failure normally arrives as :class:`PipelineFallbackError`: it is recorded
    as that variant's error in ``errors.jsonl`` (scored as a miss, reported as a run error) and the
    variant is sent again on the next run.
    """
    from jettae.evalkit.realdata.clients import budgeted
    from jettae.evalkit.realdata.keys import ANTHROPIC_KEY, load_api_key
    from jettae.evalkit.realdata.providers import BudgetExhausted, SpendTracker, claude_llm, new_anthropic_sdk

    ledger = SpendLedger(out / "llm" / "spend.json", {"started": _now(), "model": model, "max_usd": max_usd})
    tracker = SpendTracker(max_usd=max_usd, prior_usd=ledger.prior_usd)
    tracker.on_change = lambda: ledger.save(tracker)
    key = load_api_key(ANTHROPIC_KEY)
    if key is None:
        raise SystemExit(f"{ANTHROPIC_KEY}가 없습니다(환경 변수 또는 .env). DR을 시작할 수 없습니다.")
    sdk = new_anthropic_sdk(key)
    del key
    ref = ModelRef("anthropic", model)
    llm = budgeted(claude_llm(model, effort, sdk, tracker), ref, effort, tracker)  # type: ignore[arg-type]
    done = {vid for vid, diag in load_dr(out, variants, model, effort).items() if diag.error is None}
    stopped: str | None = None
    try:
        for variant in (v for v in variants if v.variant_id not in done):
            row: dict[str, Any] = {"variant_id": variant.variant_id, "sha256": variant.sha256, "model": model}
            row |= {"effort": effort, "started": _now()}
            try:
                record = await diagnose_bundle(
                    InputBundle(documents=(variant.document,), policy=study_policy()),
                    llm,
                    concurrency=concurrency,
                    key_prefix=variant.variant_id,
                    raw_texts={variant.document.doc_id: variant.raw},
                )
            except PipelineFallbackError as exc:
                _append_jsonl(out / "llm" / "errors.jsonl", row | {"error": f"{type(exc).__name__}: {exc}"})
                print(f"  DR {variant.variant_id}: 측정 실패(LLM 추출 대체), 다음 실행에서 다시 시도")
                continue
            except (BudgetExhausted, LLMError) as exc:
                stopped = f"{type(exc).__name__}: {str(exc)[:300]}"
                break
            _append_jsonl(out / "llm" / "results.jsonl", row | {"record": record.model_dump(mode="json")})
            print(f"  DR {variant.variant_id}: 완료 (이번 실행 ${tracker.spent_usd:.4f})")
    finally:
        ledger.run |= {"finished": _now(), "stopped": stopped}
        ledger.save(tracker)
        await sdk.close()
    return {"stopped": stopped, "spent_usd": tracker.report().spent_usd, "prior_usd": ledger.prior_usd}


# =========================================================================== report


def _yes(flag: bool) -> str:
    return "예" if flag else "아니오"


def _severity(code: str) -> str:
    return SEVERITY_KO.get(code, code)


def _hits_text(hits: Sequence[Mapping[str, Any]]) -> str:
    if not hits:
        return "안 뜸"
    return ", ".join(f"{_md(str(h['service']))}({_severity(h['severity'])})" for h in hits)


def _site_rows(evaluation: Mapping[str, Any]) -> list[str]:
    rows = [
        "| 지점 | 연산자 | 기대 규칙 | 원본에서 기대 규칙 | 변형에서 기대 규칙 | 판정 | 비고 |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in evaluation["sites"]:
        notes: list[str] = []
        if s["preexisting_by_annotation"]:
            notes.append("주석이 원래 있던 결함으로 표시")
        if s["preexisting_by_run"]:
            notes.append("원본 실행에서 이미 뜸")
        if s["miss_reason"]:
            notes.append(MISS_REASON_KO[s["miss_reason"]])
        rec = s["service_recognised"]
        notes.append(f"서비스 인식: 원본 {_yes(rec['baseline'])}, 변형 {_yes(rec['mutated'])}")
        rows.append(
            f"| {s['site_id']} | {OPERATOR_KO.get(s['operator'], s['operator'])} | {s['expected_rule']} "
            f"| {_hits_text(s['expected_rule_in_baseline'])} | {_hits_text(s['expected_rule_in_mutated'])} "
            f"| **{OUTCOME_KO[s['outcome']]}** | {'; '.join(notes)} |"
        )
    return rows


def _operator_rows(summary: Mapping[str, Any]) -> list[str]:
    rows = ["| 연산자 | 남긴 지점 | 기존 결함(제외) | 탐지율 |", "|---|---:|---:|---|"]
    rows += [
        f"| {OPERATOR_KO.get(op, op)} | {b['sites']} | {b['preexisting']} | {_pct(b['detection'])} |"
        for op, b in summary["by_operator"].items()
    ]
    return rows


def _keys_text(items: Sequence[Mapping[str, Any]]) -> str:
    if not items:
        return ""
    return " (" + ", ".join(f"{_md(str(i['service'] or '조직'))}·{i['rule_id']}" for i in items) + ")"


def _side_rows(evaluation: Mapping[str, Any]) -> list[str]:
    rows = ["| 지점 | 새로 생긴 지적 | 사라진 지적 |", "|---|---|---|"]
    for s in evaluation["sites"]:
        new, gone = s["side_effects"]["new"], s["side_effects"]["vanished"]
        rows.append(f"| {s['site_id']} | {len(new)}건{_keys_text(new)} | {len(gone)}건{_keys_text(gone)} |")
    return rows


def _baseline_rows(evaluation: Mapping[str, Any]) -> list[str]:
    rows = [
        "| 문서 | 원본 지적 수 | 규칙별 | 서비스·시스템 노드 | 그래프의 모든 노드 |",
        "|---|---:|---|---|---|",
    ]
    for doc, block in evaluation["baseline_findings_per_document"].items():
        by_rule = ", ".join(f"{r} {n}" for r, n in block["by_rule"].items()) or "없음"
        services = ", ".join(_md(n) for n in block["services"]) or "없음"
        nodes = ", ".join(_md(n) for n in block["graph_nodes"]) or "없음"
        rows.append(f"| `{doc}` | {block['findings']} | {by_rule} | {services} | {nodes} |")
    return rows


def _change_lines(evaluation: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for s in evaluation["sites"]:
        if s["outcome"] != "preexisting":
            continue
        verdict = "바뀜" if s["expected_finding_changed"] else "그대로"
        lines.append(f"- {s['site_id']} ({s['expected_rule']}): 원본 → 변형, 기대 규칙 지적 **{verdict}**.")
        for label, key in (("원본", "expected_rule_in_baseline"), ("변형", "expected_rule_in_mutated")):
            lines += [
                f"  - {label}: {_md(str(h['service']))} · {_severity(h['severity'])} · "
                f"{_md(_clip(h['message']))}"
                for h in s[key]
            ] or [f"  - {label}: 지점 서비스에 기대 규칙 지적 없음"]
    return lines or ["- 기존 결함으로 분리된 지점이 없다."]


def _reading_lines(cond: str, evaluation: Mapping[str, Any]) -> list[str]:
    """How to read the numbers — derived from the data, not written for one outcome."""
    s = evaluation["summary"]
    baselines = evaluation["baseline_findings_per_document"]
    blind = [d for d, b in baselines.items() if not b["services"]]
    lines: list[str] = []
    if blind:
        lines.append(
            f"- 제품이 원본 문서 {len(blind)}/{len(baselines)}건에서 서비스를 하나도 잡지 못했다. "
            "서비스가 없으면 담당·상향·승인·완화 규칙이 검사할 대상이 없어 지적도 나오지 않는다."
        )
        if cond == B:
            lines.append(
                "- 규칙 파서는 '[결제 서비스]' 같은 머리말, '서비스:' 같은 칸, 표, 카탈로그에 있는 이름처럼 "
                "정해진 모양으로 적힌 서비스만 잡는다(`src/jettae/extract/rules.py` 설명). 이 문서들은 "
                "서비스 이름을 문장 속에만 적었고, 이번 설정에는 카탈로그도 없다."
            )
    unrecognised = s["miss_reasons"].get("service_not_recognised", 0)
    if s["denominator"] and unrecognised == s["denominator"]:
        lines.append(
            "- 그래서 분모의 놓침은 모두 '서비스 미인식'이다. 지운 것을 놓쳤다기보다 이 문서들을 "
            "처음부터 읽지 못했다는 뜻이다. 사전 등록대로 놓침으로 셌고, 서비스 미인식을 뺀 민감도 "
            "값은 분모가 0이라 낼 수 없다."
        )
    elif unrecognised:
        lines.append(f"- 놓침 가운데 {unrecognised}개는 '서비스 미인식'이다.")
    return lines


def _condition_section(cond: str, evaluation: Mapping[str, Any] | None) -> list[str]:
    title = f"## {CONDITION_LABEL[cond]}"
    if evaluation is None:
        return [title, "", "- 결과 없음(돌리지 않음).", ""]
    s = evaluation["summary"]
    pre = s["preexisting"]
    side = s["side_effects"]
    reading = _reading_lines(cond, evaluation)
    return [
        title,
        "",
        f"- 채점한 지점: {evaluation['sites_scored']}/{evaluation['sites_total']}",
        f"- 기존 결함으로 분모에서 뺀 지점: {pre['total']}개({', '.join(pre['ids']) or '없음'}) — "
        f"원본 실행에서 이미 뜸 {pre['by_baseline_run']}, 주석 표시 {pre['by_annotation']}, "
        f"둘 다 {pre['by_both']}",
        f"- **탐지율: {_pct(s['detection'])}**",
        f"- 민감도(서비스 미인식 지점을 뺀 값): {_pct(s['detection_without_unrecognised_service'])}",
        f"- 부작용(지점 합계): 지점 목표가 아닌 새 지적 {side['new_findings']}건, "
        f"사라진 지적 {side['vanished_findings']}건",
        "",
        *(["### 읽는 법", "", *reading, ""] if reading else []),
        "### 지점별",
        "",
        *_site_rows(evaluation),
        "",
        "### 연산자별",
        "",
        *_operator_rows(s),
        "",
        "### 부작용(변형에서 지점과 관계없이 바뀐 지적)",
        "",
        *_side_rows(evaluation),
        "",
        "### 원본 문서의 지적",
        "",
        *_baseline_rows(evaluation),
        "",
        "### 보조 관찰: 기존 결함 지점에서 기대 규칙 지적이 바뀌었나 (탐지로 세지 않음)",
        "",
        *_change_lines(evaluation),
        "",
    ]


def _headline(cond: str, evaluation: Mapping[str, Any] | None) -> str:
    if evaluation is None:
        return f"- {cond}: 돌리지 않음."
    s = evaluation["summary"]
    unrecognised = s["miss_reasons"].get("service_not_recognised", 0)
    errors = s["miss_reasons"].get("run_error", 0)
    tail = f" 놓침 중 {unrecognised}개는 서비스를 아예 잡지 못한 경우다." if unrecognised else ""
    if errors:
        tail += f" 놓침 중 {errors}개는 실행 오류다(다음 실행에서 다시 시도)."
    return (
        f"- {cond}: 남긴 지점 {s['sites_kept']}개 중 기존 결함 {s['preexisting']['total']}개를 빼면 "
        f"분모 {s['denominator']}개, 탐지 {_pct(s['detection'])}.{tail}"
    )


_ABOUT: Final = (
    "## 무엇을 쟀나",
    "",
    "공개 런북(GitLab 운영 런북의 한국어 번역판)에서 특정 서비스의 담당·상향·승인·완화 조치 진술을",
    "줄 단위로 지웠을 때, 제품이 그 서비스에 기대 규칙으로 **새** 지적을 내는지 쟀다. 원본에서 이미",
    "그 지적이 뜨는 지점(기존 결함)은 분모에서 뺐다. 신청서가 말한 '공개 런북을 고친 봉인 결함 주입",
    "세트'를 줄인 실문서판이며, 실제 봉인 평가셋은 생성기로 만든 가상 조직을 썼다. 이 결과는 봉인셋",
    "결과를 대신하지 않는다.",
    "",
    "- 원본: 번역판 파일 그대로(sha256 확인). 변형: 지점의 줄을 빈 줄로(줄 수·줄 번호 그대로).",
    "  지점마다 변형 1개.",
    "- 명부·카탈로그 없음, 정책 기본값 + 관할 US(실증 ①과 같다). 그래서 사람·팀이 '활동 중'으로",
    "  확인되지 않아 담당·승인 규칙이 원본에서도 뜰 수 있다(주석이 미리 적은 주의점).",
)


def _sample_size(results: Mapping[str, Any], topic: str = "") -> str:
    """'지점{topic} n개(문서 m건)' from the sites actually run (not written for one input)."""
    sites = results["sites"]
    return f"지점{topic} {len(sites)}개(문서 {len({s['doc_path'] for s in sites})}건)"


def _limits(results: Mapping[str, Any]) -> list[str]:
    return [
        "## 한계 (그대로 적음)",
        "",
        "- 지점 선정과 '남은 진술 없음' 판단은 AI(Claude)가 했고, 다른 AI(Claude) 에이전트가 점검했다.",
        "  같은 종류의 모델이라 같은 방향으로 틀렸을 수 있다. 사람 검토 전이다.",
        "- 한국어판은 Claude가 만든 번역이다. 지점 줄은 영어 원문과 대조했지만 번역 전체를 사람이",
        "  검수하지 않았다.",
        f"- 출처는 GitLab 런북 하나뿐이고 {_sample_size(results, '은')}다. 분모는 아주 작고 신뢰구간은",
        "  매우 넓다. 일반화할 수 없다.",
        "- 명부 없이 돌렸다(실증 ①과 같은 설정). 명부가 있는 실제 사용에서는 담당·승인 규칙의 원본",
        "  지적이 달라질 수 있다.",
        "- 줄 단위 삭제라 문장 일부가 잘리거나 빈 코드 블록이 남는다.",
        "- 채점 규칙은 지점을 본 뒤에 썼다(실행 전이지만 봉인은 아님). 결과와 상관없이 이 보고서를 그대로",
        "  둔다.",
    ]


def _estimate_lines(results: Mapping[str, Any]) -> list[str]:
    est = results["dr_estimate"]
    ref = est.get("measured_reference")
    base_n = sum(v["site_id"] is None for v in results["variants"])
    mut_n = len(results["variants"]) - base_n
    lines = [
        "## LLM 조건(DR) 비용 추정 (호출 없음)",
        "",
        f"- 모델 {est['model']}, effort {est['effort']}, 가격표 {est['prices_as_of']} "
        f"(100만 토큰당 입력 ${est['price_per_mtok'][0]:g} / 출력 ${est['price_per_mtok'][1]:g}).",
        f"- 진단 {est['diagnoses']}회(원본 {base_n} + 변형 {mut_n}), **추정 ${est['usd_estimate']:.2f}** "
        f"(보수적: 캐시 할인 없음, 한국어 문서마다 역번역 대조 {ASSUMED_BACKTRANSLATIONS_PER_DOC}회 가정).",
    ]
    not_korean = sorted({v["doc_path"] for v in results["variants"] if not v["korean"]})
    if not_korean:
        lines.append(
            "- 제품의 한국어 판정(`is_korean_text`)에서 한국어가 아닌 문서로 나온 것: "
            + ", ".join(f"`{d}`" for d in not_korean)
            + " (글자 가운데 한글 비율이 기준보다 낮음: 영어 명령·출력 줄이 많은 문서). DR에서 이 문서는 "
            "LLM 역번역 대조를 건너뛰고(실증 ①과 같은 처리), 추정에도 역번역 호출이 없다."
        )
    if ref:
        lines.append(
            "- 실측 참고: 이전 한국어 실행의 문서당 Claude 지출 × 진단 수 = "
            f"${ref['usd_if_same_per_diagnosis']:.2f} (코더 A 포함이라 상한 참고값)."
        )
    if results["conditions"].get(DR) is None:
        lines.append(
            "- 실행하려면: `python -X utf8 scripts/eval_real_inject.py --llm --yes --max-usd <상한>`. "
            "이번에는 돌리지 않았다."
        )
    else:
        lines.append("- DR 결과는 위에 있다.")
    return [*lines, ""]


def render_report(results: Mapping[str, Any]) -> str:
    cross = results["b_cross_check"]
    same = sum(c["same_findings"] for c in cross)
    product = results["product"]
    inputs = results["inputs"]
    src_same = "예" if product["src_identical_to_frozen_commit"] else "확인 못 함"
    lines = [
        "# 실문서 결함 주입(축소판) 결과",
        "",
        f"생성: {results['generated_at']} · 계획(사전 등록): `{PLAN_DOC}` · "
        "스크립트: `scripts/eval_real_inject.py`",
        f"제품: 동결 커밋 `{product['frozen_commit']}`(src/ 동일: {src_same}) · "
        f"주석: `{inputs['annotations']}` (sha256 {inputs['annotations_sha256'][:12]}…)",
        "",
        "## 한 줄 요약",
        "",
        _headline(B, results["conditions"].get(B)),
        _headline(DR, results["conditions"].get(DR)),
        f"- 작은 사례 보고다. {_sample_size(results)}, 출처 하나(GitLab 런북), AI가 고르고 AI가 점검한 지점,",
        "  AI 번역문이다.",
        "",
        *_ABOUT,
        "- 지점의 서비스 이름(실행 전에 정함, 대소문자 무시): "
        + "; ".join(f"{s['site_id']} = {', '.join(s['service_names'])}" for s in results["sites"]),
        "",
    ]
    for cond in (B, DR):
        lines += _condition_section(cond, results["conditions"].get(cond))
    lines += [
        "## 제품 파이프라인(LLM 끔)과 조건 B가 같은가",
        "",
        "- 같은 문서를 `run_diagnosis`(LLM 끔, 리허설·초안 끔)로도 돌렸다. "
        f"(서비스, 규칙) 지적 집합이 같은 문서: {same}/{len(cross)}, "
        f"그래프 해시까지 같은 문서: {sum(c['same_graph_hash'] for c in cross)}/{len(cross)}.",
        *[
            f"- 다름: {c['variant_id']} — B에만 {c['only_in_b']}, 제품에만 {c['only_in_product_off']}"
            for c in cross
            if not c["same_findings"]
        ],
        "",
        "## 뺀 지점 (점검 단계에서 뺌, 돌리지 않음)",
        "",
        *[
            f"- {d['site_id']} (`{d['doc_path']}`, {OPERATOR_KO.get(d['operator'], d['operator'])}, "
            f"{d['expected_rule']}): {_md(d['reason_ko'])}"
            for d in results["dropped_sites"]
        ],
        "",
        *_estimate_lines(results),
        *_limits(results),
        "",
    ]
    return "\n".join(lines)


# =========================================================================== main


def _variant_json(variant: Variant) -> dict[str, Any]:
    return {
        "variant_id": variant.variant_id,
        "doc_path": variant.doc_path,
        "site_id": variant.site_id,
        "file_sha256": variant.sha256,
        "document_sha256": variant.document.sha256,
        "lines": variant.raw.count("\n") + 1,
        "korean": is_korean_text(variant.document.text),
    }


def _site_json(site: Site) -> dict[str, Any]:
    return {
        "site_id": site.site_id,
        "doc_path": site.doc_path,
        "operator": site.operator,
        "lines": list(site.lines),
        "expected_rule": site.expected_rule,
        "service_label": site.service_label,
        "service_names": list(site.aliases),
        "annotated_preexisting": site.annotated_preexisting,
        "verdict_reason_ko": site.verdict_reason,
    }


def _refuse(estimate: Mapping[str, Any]) -> None:
    print(
        f"DR(LLM) 추정: 진단 {estimate['diagnoses']}회, 모델 {estimate['model']}, "
        f"약 ${estimate['usd_estimate']:.2f}(보수적). 유료 호출입니다. "
        "--yes와 --max-usd를 붙여 다시 실행하세요. (호출하지 않았습니다)"
    )


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS, help="점검된 주석 파일")
    ap.add_argument("--korean-root", type=Path, default=None, help="번역판 폴더(기본: 주석의 값)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"결과 폴더 (기본 {DEFAULT_OUT})")
    ap.add_argument("--llm", action="store_true", help="DR(Claude) 조건, 유료. --yes 없이는 추정만")
    ap.add_argument("--yes", action="store_true", help="유료 호출 승인")
    ap.add_argument("--max-usd", type=float, default=None, help="DR 누적 지출 상한(<out>/llm 기준)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"DR 추출 모델 (기본 {DEFAULT_MODEL})")
    ap.add_argument("--effort", default=DEFAULT_EFFORT, help=f"DR effort (기본 {DEFAULT_EFFORT})")
    ap.add_argument("--concurrency", type=int, default=4, help="DR 동시 요청 수")
    return ap


def _maybe_paid(
    args: argparse.Namespace, variants: Sequence[Variant], estimate: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, int]:
    """(paid-run summary or None, exit code). Never calls a model without ``--yes`` and a cap."""
    if not args.llm:
        return None, 0
    if not args.yes or args.max_usd is None:
        _refuse(estimate)
        return None, 2
    if not math.isfinite(args.max_usd) or args.max_usd <= 0:
        # nan / inf would switch the budget gate off (every comparison with them is False / never
        # reached); 0 or less can only stop at the first call, after the key is loaded.
        print(f"--max-usd는 0보다 큰 유한한 값이어야 합니다: {args.max_usd!r} (호출하지 않았습니다)")
        return None, 2
    print(f"DR 시작: 추정 ${estimate['usd_estimate']:.2f}, 상한 ${args.max_usd:.2f}")
    paid = run_sync(
        run_dr(
            variants,
            args.out,
            model=args.model,
            effort=args.effort,
            max_usd=args.max_usd,
            concurrency=args.concurrency,
        )
    )
    return paid, 0 if paid["stopped"] is None else 3


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    guard_out_dir(args.out)
    started = time.perf_counter()
    try:
        annotations = load_annotations(args.annotations)
        korean_root = args.korean_root or Path(annotations["source"]["korean_root"])
        sites = kept_sites(annotations)
        variants = build_variants(sites, korean_root)
    except InputError as exc:
        print(f"입력 오류, 아무것도 쓰지 않았습니다: {exc}", file=sys.stderr)
        return 1
    n_base = sum(v.site_id is None for v in variants)
    print(f"지점 {len(sites)}개, 진단할 문서 {len(variants)}개(원본 {n_base})")

    prepare_worker()
    policy = study_policy()
    b_diags = {v.variant_id: run_b(v, policy) for v in variants}
    cross = [cross_check(b_diags[v.variant_id], run_product_off(v, policy)) for v in variants]
    estimate = dr_estimate(variants, args.model, args.effort)
    paid, exit_code = _maybe_paid(args, variants, estimate)
    dr_diags = load_dr(args.out, variants, args.model, args.effort)

    results: dict[str, Any] = {
        "schema": "jettae-real-inject-results/v1",
        "generated_at": _now(),
        "plan": PLAN_DOC,
        "product": product_provenance(),
        "inputs": {
            "annotations": args.annotations.as_posix(),
            "annotations_sha256": sha256_hex(args.annotations.read_bytes()),
            "korean_root": korean_root.as_posix(),
            "upstream": annotations["source"].get("upstream"),
        },
        "settings": {
            "policy": {"jurisdictions": list(policy.jurisdictions), "other": "기본값"},
            "roster": "없음",
            "catalog": "없음",
            "conditions": dict(CONDITION_LABEL),
            "dr_model": args.model,
            "dr_effort": args.effort,
        },
        "sites": [_site_json(s) for s in sites],
        "dropped_sites": dropped_sites(annotations),
        "variants": [_variant_json(v) for v in variants],
        "conditions": {B: evaluate_condition(sites, b_diags), DR: evaluate_condition(sites, dr_diags)},
        "b_cross_check": cross,
        "diagnoses": {
            B: [b_diags[v.variant_id].as_json() for v in variants],
            DR: [dr_diags[v.variant_id].as_json() for v in variants if v.variant_id in dr_diags],
        },
        "dr_estimate": estimate,
        "dr_run": paid,
        "rule_titles": {r.rule_id: r.title for r in rule_catalog()},
        "seconds": round(time.perf_counter() - started, 2),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "results.json", results)
    write_text(args.out / "report.md", render_report(results))
    for cond in (B, DR):
        print(_headline(cond, results["conditions"][cond]))
    print(f"DR 추정 ${estimate['usd_estimate']:.2f} (호출 {'함' if paid else '안 함'})")
    print(f"→ {(args.out / 'results.json').as_posix()}, {(args.out / 'report.md').as_posix()}")
    return exit_code


def _close_harness_loop() -> None:
    """Close this thread's harness event loop before interpreter shutdown (its atexit hook runs too
    late to start the executor-shutdown thread and prints a harmless traceback otherwise)."""
    from jettae.evalkit import conditions as harness_conditions

    runner = getattr(harness_conditions._LOCAL, "runner", None)
    if runner is not None:
        with contextlib.suppress(Exception):  # shutdown noise only, results are already written
            runner.close()


if __name__ == "__main__":
    try:
        code = main()
    finally:
        _close_harness_loop()
    sys.exit(code)
