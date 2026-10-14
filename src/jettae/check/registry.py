"""Fixed-rule registry. Rules are plain classes discovered from ``jettae.check.rules.*``.

Adding a rule = drop a module into ``jettae/check/rules/`` with a ``@register``-decorated
class. Rule metadata (:class:`RuleInfo`) is published via ``GET /api/v1/rules`` and
``jettae rules`` so customers can audit exactly what the judge does.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Iterable
from typing import Protocol, TypeVar

from jettae.check.context import CheckContext
from jettae.domain import (
    Element,
    Finding,
    FindingStatus,
    FixHint,
    RuleInfo,
    Severity,
    SourceSpan,
)


class Rule(Protocol):
    info: RuleInfo

    def check(self, ctx: CheckContext) -> Iterable[Finding]: ...


_REGISTRY: dict[str, Rule] = {}
_LOADED = False

R = TypeVar("R")

SEVERITY_ORDER: dict[Severity, int] = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
    Severity.INFO: 4,
}


def register(cls: type[R]) -> type[R]:
    instance = cls()
    info: RuleInfo = instance.info  # type: ignore[attr-defined]
    if info.rule_id in _REGISTRY and type(_REGISTRY[info.rule_id]) is not cls:
        raise ValueError(f"duplicate rule id {info.rule_id}")
    _REGISTRY[info.rule_id] = instance  # type: ignore[assignment]
    return cls


def _load_rules() -> None:
    global _LOADED  # noqa: PLW0603 - one-time plugin discovery
    if _LOADED:
        return
    import jettae.check.rules as pkg

    for mod in sorted(pkgutil.iter_modules(pkg.__path__), key=lambda m: m.name):
        if not mod.name.startswith("_"):
            importlib.import_module(f"{pkg.__name__}.{mod.name}")
    _LOADED = True


def rules() -> list[Rule]:
    _load_rules()
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def rule_catalog() -> list[RuleInfo]:
    return [r.info for r in rules()]


def get_rule(rule_id: str) -> Rule:
    _load_rules()
    return _REGISTRY[rule_id]


def sort_findings(findings: Iterable[Finding]) -> list[Finding]:
    unique: dict[str, Finding] = {}
    for f in findings:
        unique.setdefault(f.id, f)
    return sorted(unique.values(), key=lambda f: (SEVERITY_ORDER[f.severity], f.rule_id, f.id))


def run_checks(ctx: CheckContext, rule_ids: Iterable[str] | None = None) -> list[Finding]:
    """Run fixed rules deterministically. Never calls an LLM (enforced by import-linter)."""
    selected = rules() if rule_ids is None else [get_rule(r) for r in sorted(set(rule_ids))]
    out: list[Finding] = []
    for rule in selected:
        out.extend(rule.check(ctx))
    return sort_findings(out)


def finding(
    info: RuleInfo,
    *,
    keys: Iterable[str],
    title: str,
    message: str,
    service_id: str | None = None,
    element: Element | None = None,
    subject_ids: Iterable[str] = (),
    evidence: Iterable[SourceSpan] = (),
    fix: FixHint | None = None,
    status: FindingStatus = FindingStatus.BROKEN,
    severity: Severity | None = None,
) -> Finding:
    """Helper that builds a :class:`Finding` with a deterministic id."""
    key_tuple = tuple(keys)
    return Finding(
        id=Finding.make_id(info.rule_id, *key_tuple),
        rule_id=info.rule_id,
        severity=severity or info.severity,
        status=status,
        title=title,
        message=message,
        service_id=service_id,
        element=element if element is not None else info.element,
        subject_ids=tuple(dict.fromkeys(subject_ids)),
        evidence=tuple(dict.fromkeys(evidence)),
        fix=fix,
        legal_basis=info.legal_basis,
    )
