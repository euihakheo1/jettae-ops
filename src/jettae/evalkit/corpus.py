"""The sealed evaluation corpus: every (org × operator × style × position) case, plus clean controls.

Why clean controls are part of the corpus: a checker that flags everything finds every injected
fault. Clean renders of the same orgs in the same styles measure the false-positive side, so
"detection rate" can never be bought with noise.

Case ids are deterministic (``{org}-{op}-{style}-{pos}``, ``clean`` for controls) and the case
JSON is canonical, so :mod:`jettae.evalkit.seal` can hash and seal the set.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from jettae.evalkit.gold import DEFAULT_SEED, Fact, GoldOrg, gold_facts
from jettae.evalkit.mutate import ALL_OPERATORS, HELD_OUT_OPERATORS, Mutation, Operator, apply
from jettae.evalkit.render import (
    POSITIONS,
    STYLES,
    FactSpan,
    Layout,
    Position,
    ServiceRegion,
    Style,
    render,
)

#: Bump when generation, rendering or mutation semantics change (part of the seal).
GENERATOR_VERSION = "jettae-evalkit-gen/1.1.0"
CLEAN_LABEL = "clean"


class EvalCase(BaseModel):
    """One evaluation input bundle with its ground truth.

    ``expected_rule_ids`` must fire for ``target_service`` (a hit = any of them fires there);
    ``collateral_rule_ids`` may also fire as a direct consequence and are not false positives.
    Clean controls have no operator, no target and no expected findings. Unknown keys are
    rejected so a JSONL file cannot carry fields the seal does not commit to.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    #: which generator produced the case (part of the hashed JSON, hence of the seal)
    generator_version: str
    org_id: str
    operator: Operator | None
    style: Style
    position: Position
    layout: Layout
    held_out: bool
    focus_service: str
    target_service: str | None
    expected_rule_ids: tuple[str, ...]
    collateral_rule_ids: tuple[str, ...]
    text: str
    roster_csv: str
    catalog_yaml: str
    policy_yaml: str
    gold_facts: tuple[Fact, ...]
    deleted_facts: tuple[Fact, ...]
    added_facts: tuple[Fact, ...]
    fact_spans: tuple[FactSpan, ...]
    regions: tuple[ServiceRegion, ...]

    @property
    def is_clean(self) -> bool:
        return self.operator is None

    def canonical_json(self) -> str:
        """Key-sorted, compact, UTF-8 JSON — the byte string the seal hashes."""
        return json.dumps(
            self.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )


def case_id(org_id: str, operator: Operator | None, style: Style, position: Position) -> str:
    label = operator.value if operator is not None else CLEAN_LABEL
    return f"{org_id}-{label}-{style}-{position}"


def target_service_for(org: GoldOrg, operator: Operator) -> str:
    """Deterministic target: rotate operators over the org's regulated tier-1 services."""
    candidates = _tier1_regulated(org)
    return candidates[ALL_OPERATORS.index(operator) % len(candidates)]


def clean_focus_service(org: GoldOrg) -> str:
    """Service placed first/last in clean controls (mirrors where targets sit in mutated cases)."""
    return _tier1_regulated(org)[0]


def _tier1_regulated(org: GoldOrg) -> list[str]:
    candidates = [s.name for s in org.services if s.tier == 1 and s.regulated]
    if not candidates:
        raise ValueError(f"{org.org_id}: 결함을 주입할 규제 대상 1등급 서비스가 없습니다")
    return candidates


def build_cases(
    orgs: Iterable[GoldOrg],
    *,
    operators: Sequence[Operator] = ALL_OPERATORS,
    styles: Sequence[Style] = STYLES,
    positions: Sequence[Position] = POSITIONS,
    include_clean: bool = True,
    seed: int = DEFAULT_SEED,
) -> list[EvalCase]:
    """All cases in a deterministic order: per org, clean controls first, then operators.

    With the defaults and 20 orgs: 20×10×3×2 = 1,200 mutated cases (960 active + 240 held-out)
    and 20×3×2 = 120 clean controls. Repeated orgs, operators, styles or positions are rejected:
    they would yield two cases with one id, which the harness and the seal cannot tell apart.
    """
    orgs = list(orgs)
    for what, values in (
        ("조직", [o.org_id for o in orgs]),
        ("결함 연산자", [str(op) for op in operators]),
        ("문체", [str(s) for s in styles]),
        ("배치", [str(p) for p in positions]),
    ):
        if dupes := sorted({v for v in values if values.count(v) > 1}):
            raise ValueError(f"{what} 목록에 중복이 있어 사례 id가 겹칩니다: {dupes}")
    truths: list[_Truth] = []
    for org in orgs:
        if include_clean:
            truths.append(_Truth.clean(org))
        truths += [_Truth.mutated(apply(org, op, target_service_for(org, op))) for op in operators]
    return [_case(t, style, pos, seed) for t in truths for style in styles for pos in positions]


@dataclass(frozen=True)
class _Truth:
    """Ground truth shared by the style × position renders of one (org, operator)."""

    org: GoldOrg
    operator: Operator | None
    focus: str
    target: str | None
    expected: tuple[str, ...]
    collateral: tuple[str, ...]
    gold: tuple[Fact, ...]
    deleted: tuple[Fact, ...]
    added: tuple[Fact, ...]

    @classmethod
    def clean(cls, org: GoldOrg) -> _Truth:
        gold = tuple(sorted(gold_facts(org)))
        return cls(org, None, clean_focus_service(org), None, (), (), gold, (), ())

    @classmethod
    def mutated(cls, m: Mutation) -> _Truth:
        return cls(
            org=m.org,
            operator=m.operator,
            focus=m.target_service,
            target=m.target_service,
            expected=tuple(sorted(m.expected_rule_ids)),
            collateral=tuple(sorted(m.collateral_rule_ids)),
            gold=tuple(sorted(gold_facts(m.org))),
            deleted=tuple(sorted(m.deleted_facts)),
            added=tuple(sorted(m.added_facts)),
        )


def _case(truth: _Truth, style: Style, position: Position, seed: int) -> EvalCase:
    org = truth.org
    rendered = render(org, style=style, position=position, focus_service=truth.focus, seed=seed)
    return EvalCase(
        case_id=case_id(org.org_id, truth.operator, style, position),
        generator_version=GENERATOR_VERSION,
        org_id=org.org_id,
        operator=truth.operator,
        style=style,
        position=position,
        layout=rendered.layout,
        held_out=truth.operator in HELD_OUT_OPERATORS,
        focus_service=truth.focus,
        target_service=truth.target,
        expected_rule_ids=truth.expected,
        collateral_rule_ids=truth.collateral,
        text=rendered.text,
        roster_csv=rendered.roster_csv,
        catalog_yaml=rendered.catalog_yaml,
        policy_yaml=rendered.policy_yaml,
        gold_facts=truth.gold,
        deleted_facts=truth.deleted,
        added_facts=truth.added,
        fact_spans=rendered.fact_spans,
        regions=rendered.regions,
    )


def split_held_out(cases: Iterable[EvalCase]) -> tuple[list[EvalCase], list[EvalCase]]:
    """(open cases usable for development, sealed held-out cases that must not be used to tune)."""
    open_cases: list[EvalCase] = []
    sealed: list[EvalCase] = []
    for case in cases:
        (sealed if case.held_out else open_cases).append(case)
    return open_cases, sealed


def write_cases_jsonl(cases: Iterable[EvalCase], path: str | Path) -> None:
    """One canonical JSON case per line (UTF-8, ``\\n`` line endings on every platform)."""
    Path(path).write_bytes("".join(c.canonical_json() + "\n" for c in cases).encode("utf-8"))


def read_cases_jsonl(path: str | Path) -> list[EvalCase]:
    """Inverse of :func:`write_cases_jsonl`.

    Splits on ``\\n`` only: canonical JSON escapes control characters but leaves U+2028/U+2029
    and U+0085 raw, so ``str.splitlines`` would cut a case whose text contains them in half.
    """
    lines = Path(path).read_bytes().decode("utf-8").split("\n")
    return [EvalCase.model_validate_json(line) for line in lines if line.strip()]
