"""Patch validation (수정안 검증): re-check, re-rehearse, and refuse anyone outside the roster.

Why a patch is validated before a human ever sees an "approve" button (ADR-0003): a fix that
closes one gap can open another (a new owner without an escalation, an approver who is the only
one left), and a fix can quietly break a rehearsal that used to pass. So :func:`validate_patch`
applies the ops to a copy of the graph and asks the very judges that produced the findings:

* **Fixed rules** (:func:`jettae.check.registry.run_checks`) before and after. A claimed finding is
  *resolved* when it is gone afterwards; a finding that appears is *new*. Finding ids embed their
  subjects (the people involved and the channels they are reached by), so replacing a departed
  owner re-keys an untouched gap such as "no escalation" from ``…김민준`` to ``…김서연``, and giving a
  person without any contact medium a first one re-keys "fewer than two channels" from ``()`` to
  ``(전화)``. Such a pair — one finding disappears and another with the same rule, service and
  element, no higher severity and the same subjects apart from people and channels appears — is a
  *continuation*: it counts neither as new nor as resolved, and the notes say so. Without this,
  every owner replacement would look like a regression, and a fix that merely renames a gap
  would look like a fix. A gap that got *worse* (higher severity) is new, never a continuation.
* **Rehearsals** (:func:`jettae.rehearse.simulator.replay_all`, roster from the context) before
  and after: FAIL → PASS is *fixed*, PASS → FAIL is a *regression* and fails validation (the
  no-regression rule in the spirit of STRATUS: an automated change must never make a known-good
  path worse).
* **People** — every person a patch adds or links to must be a *confirmed* ACTIVE roster person
  (KPI "명부 외 인물 생성 0건"): exactly one roster row stands for the node (:func:`roster_entry`,
  namesake-safe), that row is ACTIVE, the node's own status as the sources stated it is ACTIVE (a
  patch cannot vouch for itself by rewriting it) and the node is not on HOLD (동명이인·모호한 지칭).
  Removing links to people who left is of course allowed.
* **Integrity** — ops that do not apply (:class:`~jettae.improve.patching.PatchError`) fail, and so
  does an element the patch creates without saying it comes from *this* patch: added nodes and
  edges must carry ``Provenance(source=PATCH, extractor=HUMAN, ref=<patch id>)`` and every value a
  patch writes must be ``Attr(source=PATCH)``. Otherwise an approved patch could plant a "fact"
  that claims to come from a runbook or the roster (P2) and the history could not say which
  approval introduced it (P7).

``passed`` iff the patch has ops, applies cleanly, attributes what it creates to itself, regresses
no rehearsal, names nobody outside the roster, resolves at least one finding it claims and
introduces no new HIGH/CRITICAL finding;
new findings of lower severity are reported but do not block. The function is pure — the caller
records VALIDATED / REJECTED_BY_VALIDATOR. No LLM is imported here (P1, enforced by import-linter)
and the result depends only on the inputs (P6).
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from jettae.check.context import CheckContext
from jettae.check.registry import SEVERITY_ORDER, run_checks
from jettae.check.scoring import score
from jettae.domain import (
    AddEdge,
    AddNode,
    Attr,
    AttrKey,
    EdgeKind,
    Extractor,
    Finding,
    Node,
    NodeKind,
    OpsGraph,
    Outcome,
    Patch,
    PatchOp,
    PatchValidation,
    PersonStatus,
    Provenance,
    ReadinessReport,
    RehearsalResult,
    RosterEntry,
    Scenario,
    SetAttr,
    Severity,
    SimConfig,
    SourceKind,
    Verification,
)
from jettae.improve.patching import PatchError, apply_ops
from jettae.rehearse.simulator import config_hash, replay_all

__all__ = ["BLOCKING_SEVERITIES", "roster_entry", "validate_patch"]

#: New findings of these severities block a patch.
BLOCKING_SEVERITIES: frozenset[Severity] = frozenset({Severity.CRITICAL, Severity.HIGH})

#: Subject kinds that say *who* is involved and *how they are reached*; a patch that changes people
#: re-keys findings on them, and so does one that gives a person another contact medium.
_WHO_KINDS = frozenset({NodeKind.PERSON, NodeKind.ROLE, NodeKind.TEAM, NodeKind.CHANNEL})

_DEFAULT_CONFIG = SimConfig()
_MEMO_NOTE = "그래프를 바꾸는 연산이 없는 검토 메모라 승인 대상이 아닙니다. 사람이 정한 뒤 다시 진단하세요."


@dataclass(frozen=True)
class _Matching:
    """How the findings before and after a patch relate."""

    resolved: tuple[str, ...]
    unresolved: tuple[str, ...]
    new: tuple[Finding, ...]
    continued: tuple[tuple[Finding, Finding], ...]  # (before, after)
    side_resolved: tuple[str, ...]


def _is_who_ref(item_id: str, graphs: Sequence[OpsGraph]) -> bool:
    for graph in graphs:
        node = graph.nodes.get(item_id)
        if node is not None:
            return node.kind in _WHO_KINDS
        edge = graph.edges.get(item_id)
        if edge is not None:
            ends = [graph.nodes[n] for n in (edge.src, edge.dst) if n in graph.nodes]
            return any(n.kind in _WHO_KINDS for n in ends)
    return False


def _core(finding: Finding, graphs: Sequence[OpsGraph]) -> frozenset[str]:
    """Subjects other than the service itself and other than who is involved and how they are reached."""
    return frozenset(s for s in finding.subject_ids if s != finding.service_id and not _is_who_ref(s, graphs))


def _continues(before: Finding, after: Finding, graphs: Sequence[OpsGraph]) -> bool:
    if (before.rule_id, before.service_id, before.element) != (
        after.rule_id,
        after.service_id,
        after.element,
    ):
        return False
    if SEVERITY_ORDER[after.severity] < SEVERITY_ORDER[before.severity]:
        return False  # the gap got worse: that is a new finding, not the old one renamed
    a, b = _core(before, graphs), _core(after, graphs)
    return (not a and not b) or bool(a & b)


def _match(
    claimed: Iterable[str], before: Sequence[Finding], after: Sequence[Finding], graphs: Sequence[OpsGraph]
) -> _Matching:
    """Pair every finding that disappeared with at most one look-alike that appeared (greedy, stable)."""
    before_ids = {f.id for f in before}
    after_ids = {f.id for f in after}
    gone = sorted((f for f in before if f.id not in after_ids), key=lambda f: f.id)
    appeared = sorted(
        (f for f in after if f.id not in before_ids),
        key=lambda f: (SEVERITY_ORDER[f.severity], f.rule_id, f.id),
    )
    pairs: list[tuple[Finding, Finding]] = []
    paired: set[str] = set()
    new: list[Finding] = []
    for f in appeared:
        partner = next((g for g in gone if g.id not in paired and _continues(g, f, graphs)), None)
        if partner is None:
            new.append(f)
        else:
            paired.add(partner.id)
            pairs.append((partner, f))
    claims = sorted(set(claimed))
    resolved = tuple(c for c in claims if c in before_ids and c not in after_ids and c not in paired)
    unresolved = tuple(c for c in claims if c not in resolved)
    side = tuple(f.id for f in gone if f.id not in paired and f.id not in claims)
    return _Matching(resolved, unresolved, tuple(new), tuple(pairs), side)


def _additive_people(ops: Sequence[PatchOp], before: OpsGraph, after: OpsGraph) -> list[Node]:
    """Person nodes a patch adds, links to or edits — removals do not count.

    A person the patch adds and removes again is still looked at (as the patch added it): the
    lookup falls back from the result graph to the patch's own AddNode ops and the base graph.
    """
    added = {op.node.id: op.node for op in ops if isinstance(op, AddNode)}
    nodes = {**before.nodes, **added, **after.nodes}

    def person(nid: str) -> Node | None:
        node = nodes.get(nid)
        return node if node is not None and node.kind is NodeKind.PERSON else None

    found: dict[str, Node] = {}
    for op in ops:
        if isinstance(op, AddNode):
            ids: tuple[str, ...] = (op.node.id,)
        elif isinstance(op, AddEdge):
            ids = (op.edge.src, op.edge.dst)
        elif isinstance(op, SetAttr):
            ids = (op.target_id,)
        else:
            continue
        found.update((n.id, n) for n in map(person, ids) if n is not None)
    return [found[k] for k in sorted(found)]


def _key(text: str) -> str:
    """Comparison key for roster facts: NFKC, no whitespace, case-folded ("결제 팀" == "결제팀")."""
    return "".join(unicodedata.normalize("NFKC", text).split()).casefold()


def _agrees(graph: OpsGraph, person: Node, row: RosterEntry) -> bool:
    """Do the facts the roster import wrote on ``person`` (status, title, team) fit ``row``?

    Facts the node does not carry are no evidence either way.
    """
    status = person.attr(AttrKey.STATUS)
    if isinstance(status, str) and status != row.status.value:
        return False
    title = person.attr(AttrKey.TITLE)
    if isinstance(title, str) and row.title and _key(title) != _key(row.title):
        return False
    teams = {_key(t.name) for t in graph.successors(person.id, EdgeKind.MEMBER_OF)}
    return not (teams and row.team and _key(row.team) not in teams)


def roster_entry(ctx: CheckContext, person: Node) -> RosterEntry | None:
    """The one roster row a person node stands for — safe for namesakes (동명이인).

    :meth:`CheckContext.roster_entry` returns the *first* row whose name or alias matches, so for
    two 김민수 (결제팀, 물류팀) both nodes would read the first row: its status, team, roles and
    registered media. A patch could then hand a 결제팀 duty to "a teammate" from 물류팀 or register a
    medium the person never gave. When several rows match by name, the row is the one whose facts
    agree with what the roster import wrote on the node (status, title, MEMBER_OF team); when that
    does not single out one row, nobody can say who the node is and ``None`` is returned (P3).
    Identical rows (a roster listing one person twice) count once. The drafter uses the same rule,
    so whatever it proposes passes this check.
    """
    rows = list(dict.fromkeys(r for r in ctx.roster if person.name == r.name or person.name in r.aliases))
    if len(rows) <= 1:
        return rows[0] if rows else None
    fitting = [r for r in rows if _agrees(ctx.graph, person, r)]
    return fitting[0] if len(fitting) == 1 else None


def _unknown_people(ctx: CheckContext, ops: Sequence[PatchOp], after: OpsGraph) -> tuple[str, ...]:
    """People the patch adds, links to or edits who are not confirmed ACTIVE roster people.

    Identity is judged on the node *as the sources stated it* (the base graph, or the patch's own
    AddNode for a node the base lacks): a patch that rewrites a departed person's status to
    "active" must not vouch for itself.
    """
    unknown: list[str] = []
    for node in _additive_people(ops, ctx.graph, after):
        stated = ctx.graph.nodes.get(node.id, node)
        entry = roster_entry(ctx, stated)
        confirmed = (
            entry is not None
            and entry.status is PersonStatus.ACTIVE
            and ctx.person_status(stated) is PersonStatus.ACTIVE
            and Verification.HOLD not in (stated.verification, node.verification)
        )
        if not confirmed:
            unknown.append(node.id)
    return tuple(unknown)


def _foreign_prov(prov: Iterable[Provenance], patch_id: str) -> bool:
    return any(
        p.source is not SourceKind.PATCH or p.extractor is not Extractor.HUMAN or p.ref != patch_id
        for p in prov
    )


def _foreign_attrs(attrs: Mapping[str, Attr]) -> list[str]:
    return sorted(k for k, a in attrs.items() if a.source is not SourceKind.PATCH)


def _provenance_errors(patch: Patch) -> tuple[str, ...]:
    """Elements and values the patch writes without attributing them to itself (P2, P7)."""
    errors: list[str] = []
    for index, op in enumerate(patch.ops, start=1):
        where = f"{index}번째 연산({op.op})"
        if isinstance(op, AddNode | AddEdge):
            element = op.node if isinstance(op, AddNode) else op.edge
            if _foreign_prov(element.prov, patch.id):
                errors.append(
                    f"{where}: '{element.id}'의 출처가 이 수정안(PATCH·HUMAN·{patch.id})으로 "
                    "적혀 있지 않습니다."
                )
            keys = _foreign_attrs(element.attrs)
            if keys:
                errors.append(f"{where}: '{element.id}'의 속성 {', '.join(keys)}의 출처가 수정안이 아닙니다.")
        elif isinstance(op, SetAttr) and op.value is not None and op.value.source is not SourceKind.PATCH:
            errors.append(f"{where}: '{op.target_id}'의 '{op.key}' 값의 출처가 수정안이 아닙니다.")
    return tuple(errors)


def _rehearse(
    ctx: CheckContext,
    graph: OpsGraph,
    scenarios: Sequence[Scenario],
    config: SimConfig,
    reuse: Mapping[str, RehearsalResult],
) -> dict[str, RehearsalResult]:
    """Results per scenario id; a reused result must match the graph, the scenario and the config."""
    graph_hash = graph.content_hash()
    key = config_hash(config, ctx.policy, ctx.roster)
    out: dict[str, RehearsalResult] = {}
    missing: list[Scenario] = []
    for s in scenarios:
        known = reuse.get(s.id)
        if (
            known is not None
            and known.graph_hash == graph_hash
            and known.scenario_fingerprint == s.fingerprint()
            and known.config_hash == key
        ):
            out[s.id] = known
        else:
            missing.append(s)
    if missing:
        results = replay_all(graph, missing, policy=ctx.policy, config=config, roster=ctx.roster)
        out.update((s.id, r) for s, r in zip(missing, results, strict=True))
    return out


def _compare(
    scenarios: Sequence[Scenario], before: Mapping[str, RehearsalResult], after: Mapping[str, RehearsalResult]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(FAIL → PASS, PASS → FAIL) scenario ids, in scenario order."""

    def flipped(old: Outcome, new: Outcome) -> tuple[str, ...]:
        return tuple(s.id for s in scenarios if before[s.id].outcome is old and after[s.id].outcome is new)

    return flipped(Outcome.FAIL, Outcome.PASS), flipped(Outcome.PASS, Outcome.FAIL)


def _score_notes(
    before: ReadinessReport, after: ReadinessReport, findings: Iterable[Finding], claimed: Iterable[str]
) -> list[str]:
    """Readiness of the services the patch targets, then overall, before → after."""
    wanted = set(claimed)
    service_ids = {f.service_id for f in findings if f.id in wanted and f.service_id is not None}
    old = {s.service_id: s for s in before.services if s.service_id in service_ids}
    new = {s.service_id: s for s in after.services if s.service_id in service_ids}
    notes = [
        f"{old[sid].service_name} 준비도 {old[sid].score:.1f} → {new[sid].score:.1f}"
        for sid in sorted(old.keys() & new.keys())
    ]
    return [*notes, f"전체 준비도 {before.overall:.1f} → {after.overall:.1f}"]


def _ids(items: Iterable[str]) -> str:
    return ", ".join(items)


def _verdict_notes(
    matching: _Matching,
    *,
    stale: Sequence[str],
    fixed: Sequence[str],
    regressed: Sequence[str],
    unknown: Sequence[str],
) -> list[str]:
    """Korean explanation of the verdict: blockers first, then what else changed."""
    blocking = [f.id for f in matching.new if f.severity in BLOCKING_SEVERITIES]
    minor = [f.id for f in matching.new if f.severity not in BLOCKING_SEVERITIES]
    pairs = [f"{a.id} → {b.id}" for a, b in matching.continued]
    candidates = [
        (not matching.resolved, "수정안이 겨냥한 끊김 중 해소된 것이 없습니다."),
        (bool(stale), f"기준 그래프에서 찾을 수 없는 끊김을 겨냥합니다(기준 버전 확인 필요): {_ids(stale)}"),
        (bool(blocking), f"높음 이상 심각도의 새 끊김이 생깁니다: {_ids(blocking)}"),
        (bool(regressed), f"전에 통과하던 리허설이 실패합니다(무회귀 위반): {_ids(regressed)}"),
        (
            bool(unknown),
            "명부에 재직 중으로 없는 인물이 수정안에 들어 있습니다(동명이인·판단 보류 포함): "
            + _ids(unknown),
        ),
        (bool(minor), f"새로 생기는 낮은 심각도 끊김(통과에는 영향 없음): {_ids(minor)}"),
        (bool(pairs), f"담당자가 바뀌어 id만 달라진 기존 끊김(새 끊김으로 보지 않음): {_ids(pairs)}"),
        (bool(matching.side_resolved), f"함께 해소된 끊김: {_ids(matching.side_resolved)}"),
        (bool(fixed), f"실패하던 리허설이 통과합니다: {_ids(fixed)}"),
    ]
    return [text for show, text in candidates if show]


def validate_patch(
    ctx: CheckContext,
    patch: Patch,
    *,
    scenarios: Iterable[Scenario] = (),
    config: SimConfig = _DEFAULT_CONFIG,
    baseline_findings: Sequence[Finding] | None = None,
    baseline_results: Iterable[RehearsalResult] | None = None,
) -> PatchValidation:
    """Validate ``patch`` against ``ctx`` (its base graph); see the module docstring for the rules.

    ``baseline_findings`` (the findings of ``ctx``) and ``baseline_results`` (rehearsals on
    ``ctx.graph``) may be passed to avoid recomputing them; a baseline result is reused only if
    its graph hash, scenario fingerprint and config hash match, otherwise it is replayed.
    Duplicate scenario ids are replayed once (first occurrence wins).
    """
    before = list(baseline_findings) if baseline_findings is not None else run_checks(ctx)
    report_before = score(ctx, before)
    claimed = tuple(sorted(set(patch.finding_ids)))
    if not patch.ops:
        return PatchValidation(
            passed=False,
            unresolved_finding_ids=claimed,
            score_before=report_before.overall,
            score_after=report_before.overall,
            notes=(_MEMO_NOTE,),
        )
    try:
        after_graph, _applied = apply_ops(ctx.graph, patch.ops)
    except PatchError as exc:
        return PatchValidation(
            passed=False,
            unresolved_finding_ids=claimed,
            integrity_errors=(str(exc),),
            score_before=report_before.overall,
            notes=("수정안을 기준 그래프에 적용할 수 없어 검증하지 못했습니다.",),
        )
    after_ctx = ctx.with_graph(after_graph)
    after = run_checks(after_ctx)
    report_after = score(after_ctx, after)
    matching = _match(claimed, before, after, (ctx.graph, after_graph))

    unique: dict[str, Scenario] = {}
    for scenario in scenarios:
        unique.setdefault(scenario.id, scenario)
    replays = list(unique.values())
    reuse = {r.scenario_id: r for r in baseline_results or ()}
    fixed, regressed = _compare(
        replays,
        _rehearse(ctx, ctx.graph, replays, config, reuse),
        _rehearse(after_ctx, after_graph, replays, config, {}),
    )
    unknown = _unknown_people(ctx, patch.ops, after_graph)
    foreign = _provenance_errors(patch)
    blocking = any(f.severity in BLOCKING_SEVERITIES for f in matching.new)
    known = {f.id for f in before}
    notes = _verdict_notes(
        matching,
        stale=[c for c in claimed if c not in known],
        fixed=fixed,
        regressed=regressed,
        unknown=unknown,
    )
    if foreign:
        notes.insert(0, "수정안이 만드는 요소·값 중 출처가 이 수정안으로 적혀 있지 않은 것이 있습니다.")
    return PatchValidation(
        passed=bool(matching.resolved) and not blocking and not regressed and not unknown and not foreign,
        resolved_finding_ids=matching.resolved,
        unresolved_finding_ids=matching.unresolved,
        new_finding_ids=tuple(f.id for f in matching.new),
        rehearsals_fixed=fixed,
        rehearsals_regressed=regressed,
        unknown_people=unknown,
        integrity_errors=foreign,
        score_before=report_before.overall,
        score_after=report_after.overall,
        notes=(*notes, *_score_notes(report_before, report_after, before, claimed)),
    )
