"""Korean incident-reporting duties (국내 신고 기준): ``JT-KR-001`` … ``JT-KR-004``.

Why these rules exist: a runbook can restore a service perfectly and still leave the company
in breach of the law, because the *report* (침해사고 24시간, 개인정보 유출 72시간) was nobody's
step. The obligations themselves come from ``Policy.obligations`` (defaults: ``DEFAULT_OBLIGATIONS``)
so the legal text cited is always the one the organisation configured, and only rules with
``applies=True`` are checked — whether a company is e.g. an 정보통신서비스 제공자 is a fact it
declares, not something we guess. The statutory defaults are Korean law (``jurisdiction="KR"``):
they bind only an organisation whose ``Policy.jurisdictions`` include KR (the default), so a
company operating under another legal system is not told it misses a Korean report
(:meth:`~jettae.domain.Policy.in_force`).

"확인 불가 ≠ 정상" (P3) shapes the statuses: a reporting step that exists but does not say which
incident it covers, or that is on HOLD (판단 보류), is reported as UNKNOWN rather than as
missing (BROKEN) or as fine; so is a regulated service the policy names but the graph lacks.

Which steps bind a service is :meth:`CheckContext.obligations`: the service's own MUST_REPORT
steps plus organisation-wide ones (a security procedure that says "침해사고를 인지하면 … KISA에
신고한다" without naming a service binds every service). Reporting "no KISA step" for 결제 while
the company procedure has one would send the team to write a duplicate instead of fixing the
real gap (e.g. its 48-hour deadline, which ``JT-DDL-002`` reports).

This module also hosts the small obligation-matching helpers that ``rules/deadlines.py`` reuses,
so both rule families agree on what "a breach report to KISA" looks like in the graph.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterator, Sequence

from jettae.check.context import CheckContext
from jettae.check.registry import finding, register
from jettae.check.spof import GraphIndex
from jettae.domain import (
    DEFAULT_OBLIGATIONS,
    AttrKey,
    EdgeKind,
    Finding,
    FindingStatus,
    FixHint,
    FixKind,
    IncidentType,
    Node,
    NodeKind,
    ObligationRule,
    PersonStatus,
    RuleInfo,
    Severity,
)

# ---------------------------------------------------------------------------- shared helpers

#: Well-known recipient agencies and the ways runbooks name them. The first entry of each group
#: is its canonical key. Matching is substring-based on an NFKC, whitespace-free, case-folded
#: text. KISA is often named by its reporting front doors (보호나라·KrCERT·인터넷침해대응센터)
#: and the PIPC by its reporting portal (개인정보 포털), so those count as the same body.
RECIPIENT_ALIASES: tuple[tuple[str, ...], ...] = (
    ("과학기술정보통신부", "과기정통부", "과기부", "msit"),
    ("한국인터넷진흥원", "인터넷진흥원", "kisa", "보호나라", "krcert", "인터넷침해대응센터"),
    ("개인정보보호위원회", "개인정보위", "pipc", "개인정보포털", "개인정보보호포털"),
)

_SPACES = re.compile(r"\s+")
_NAME_NOISE = re.compile(r"[\s\-_]+")


def _norm(text: str) -> str:
    """NFKC (full-width "ＫＩＳＡ" -> "KISA", composed Hangul), no whitespace, case-folded."""
    return _SPACES.sub("", unicodedata.normalize("NFKC", text)).casefold()


def name_key(text: str) -> str:
    """Comparison key for service names: NFKC, case-folded, without spaces, hyphens, underscores.

    "결제 서비스", "결제-서비스" and "결제서비스" name the same service in policies, catalogs and
    runbooks; comparing raw strings would silently drop one of them from a check.
    """
    return _NAME_NOISE.sub("", unicodedata.normalize("NFKC", text)).casefold()


def recipient_keys(text: str) -> frozenset[str]:
    """Canonical agency keys mentioned in ``text`` ("과기정통부·KISA" -> both agencies)."""
    norm = _norm(text)
    return frozenset(group[0] for group in RECIPIENT_ALIASES if any(alias in norm for alias in group))


def recipient_matches(stated: str | None, required: str | None) -> bool:
    """Do two recipient texts name the same body?

    An unstated recipient matches (the obligation's incident type already says what it is
    about; a missing recipient is a wording gap, not a different duty). Known agencies match on
    canonical keys so "KISA" equals "한국인터넷진흥원"; other names fall back to containment.
    """
    if not stated or not stated.strip() or not required or not required.strip():
        return True
    a, b = recipient_keys(stated), recipient_keys(required)
    if a and b:
        return bool(a & b)
    na, nb = _norm(stated), _norm(required)
    return na in nb or nb in na


def obligation_incident_type(node: Node) -> IncidentType | None:
    raw = node.attr(AttrKey.INCIDENT_TYPE)
    try:
        return IncidentType(raw) if isinstance(raw, str) else None
    except ValueError:
        return None


def obligation_deadline(node: Node) -> float | None:
    """Stated deadline in minutes, or ``None`` when the runbook does not state a usable one."""
    raw = node.attr(AttrKey.DEADLINE_MINUTES)
    if isinstance(raw, bool) or not isinstance(raw, int | float) or not math.isfinite(raw) or raw < 0:
        return None
    return float(raw)


def obligation_recipient(node: Node) -> str | None:
    raw = node.attr(AttrKey.RECIPIENT)
    return raw if isinstance(raw, str) and raw.strip() else None


def satisfies_rule(node: Node, rule: ObligationRule) -> bool:
    """Is this OBLIGATION node the runbook's version of the policy duty ``rule``?"""
    return obligation_incident_type(node) is rule.incident_type and recipient_matches(
        obligation_recipient(node), rule.recipient
    )


def might_satisfy_rule(node: Node, rule: ObligationRule) -> bool:
    """A step to the same recipient whose incident type the runbook leaves unstated."""
    return obligation_incident_type(node) is None and recipient_matches(
        obligation_recipient(node), rule.recipient
    )


def applicable_rules(ctx: CheckContext, incident_type: IncidentType | None = None) -> list[ObligationRule]:
    """Duties that bind the organisation (:meth:`Policy.in_force`: declared in scope *and* of one of
    its jurisdictions — the Korean 24h/72h reports bind only a policy with ``jurisdictions`` KR)."""
    return list(ctx.policy.obligations_in_force(incident_type))


def reporting_services(ctx: CheckContext, obligation_id: str) -> list[str]:
    """Services/systems bound by this obligation (sorted, each once).

    Its MUST_REPORT sources, or every service for an organisation-wide duty
    (:meth:`CheckContext.obligation_services`). The assembler keeps contradicting statements of
    one relation as parallel edges (``must_report:a->b#…``), so one service can reach the same
    obligation twice; listing it twice would make :func:`single_or_none` drop the finding's service.
    """
    return [n.id for n in ctx.obligation_services(obligation_id)]


def single_or_none(ids: list[str]) -> str | None:
    """A finding belongs to a service only when exactly one service is concerned."""
    return ids[0] if len(ids) == 1 else None


def duty_scope(ctx: CheckContext, obligation_id: str) -> tuple[str | None, list[str]]:
    """``(service_id, service ids to highlight)`` for a finding about one obligation.

    An organisation-wide duty belongs to no single service, and highlighting every service of
    the company would bury the one node that needs fixing (the obligation), so it gets neither.
    """
    if ctx.is_org_obligation(obligation_id):
        return None, []
    ids = reporting_services(ctx, obligation_id)
    return single_or_none(ids), ids


def format_minutes(minutes: float) -> str:
    """Korean duration as laws and runbooks state it: whole hours as "N시간", otherwise "N분".

    Never scientific notation ("1e+06분"): the text is read by people under time pressure.
    """
    if minutes >= 60 and minutes % 60 == 0:
        return f"{int(minutes // 60)}시간"
    if float(minutes).is_integer():
        return f"{int(minutes)}분"
    return f"{minutes:.1f}".rstrip("0").rstrip(".") + "분"


def law_text(incident_type: IncidentType) -> str:
    """The default legal text for a reporting duty (cited in ``RuleInfo.legal_basis``).

    Raises ``ValueError`` for an incident type without a statutory default (e.g. ``outage``)
    instead of leaking ``StopIteration``, which Python turns into ``RuntimeError`` inside
    generators and which callers would never think to catch.
    """
    for rule in DEFAULT_OBLIGATIONS:
        if rule.incident_type is incident_type:
            return rule.legal_basis
    raise ValueError(f"법정 신고 기준이 정의되지 않은 사고 유형입니다: {incident_type.value}")


#: How many names a Korean message spells out before summarising the rest ("… 외 N개").
MAX_LISTED_NAMES = 5


def listed_names(names: Sequence[str]) -> str:
    """ "'결제', '회원' 외 198개": an organisation-level finding over 200 services must stay readable.

    The full list always travels in ``subject_ids``/fix params; the message only needs enough
    names to recognise the problem.
    """
    shown = ", ".join(f"'{n}'" for n in names[:MAX_LISTED_NAMES])
    rest = len(names) - MAX_LISTED_NAMES
    return f"{shown} 외 {rest}개" if rest > 0 else shown


# ---------------------------------------------------------------------------- scope


def node_aliases(node: Node) -> list[str]:
    """The node's ``aliases`` attribute, tolerating a malformed (non-list) value as "no aliases"."""
    raw = node.attr(AttrKey.ALIASES)
    return [a for a in raw if isinstance(a, str)] if isinstance(raw, list) else []


def service_keys(node: Node) -> set[str]:
    """:func:`name_key` of the node's id (without the kind prefix), name and aliases."""
    local_id = node.id.split(":", 1)[-1]
    return {name_key(x) for x in (local_id, node.name, *node_aliases(node))}


def _declared_key(name: str) -> str:
    name = name.strip()
    for kind in (NodeKind.SERVICE, NodeKind.SYSTEM):
        name = name.removeprefix(f"{kind.value}:")
    return name_key(name)


def _declared(ctx: CheckContext) -> list[str]:
    return sorted({d.strip() for d in ctx.policy.regulated_services if d.strip()})


def _declarable(ctx: CheckContext) -> list[Node]:
    """Services and systems: a policy may name a system (e.g. 회원DB) as the regulated asset."""
    return ctx.graph.nodes_of(NodeKind.SERVICE, NodeKind.SYSTEM)


def regulated_scope(ctx: CheckContext) -> list[Node]:
    """``policy.regulated_services`` (services/systems by id, name or alias), else critical services."""
    declared = _declared(ctx)
    if not declared:
        return ctx.critical_services()
    wanted = {_declared_key(d) for d in declared}
    return [s for s in _declarable(ctx) if service_keys(s) & wanted]


def unmatched_regulated(ctx: CheckContext) -> list[str]:
    """Regulated names from the policy that match no service or system in the graph (sorted)."""
    known = {key for s in _declarable(ctx) for key in service_keys(s)}
    return [d for d in _declared(ctx) if _declared_key(d) not in known]


# ---------------------------------------------------------------------------- rule bodies

_STATUS_KO: dict[PersonStatus, str] = {
    PersonStatus.DEPARTED: "퇴사",
    PersonStatus.ON_LEAVE: "장기 부재",
    PersonStatus.UNKNOWN: "명부 미확인",
}


def _duty(rule: ObligationRule) -> str:
    return f"{rule.recipient}에 {format_minutes(rule.deadline_minutes)} 이내"


def _usable_duty(ctx: CheckContext, index: GraphIndex, sid: str, obligation: Node) -> bool:
    """The duty binds ``sid`` through confirmed facts only: a verified obligation that is
    org-wide or linked by a verified MUST_REPORT edge (held facts never prove a step exists)."""
    if not index.is_usable(obligation.id):
        return False
    if ctx.is_org_obligation(obligation.id):
        return True
    return any(e.dst == obligation.id for e in index.out_edges(sid, EdgeKind.MUST_REPORT))


def _step_status(
    ctx: CheckContext, index: GraphIndex, sid: str, rule: ObligationRule
) -> FindingStatus | None:
    """``None`` if ``sid`` has the step; UNKNOWN if only an unclear or held step might be it."""
    every = ctx.obligations(sid)
    usable = [o for o in every if _usable_duty(ctx, index, sid, o)]
    if any(satisfies_rule(o, rule) for o in usable):
        return None
    held = [o for o in every if o not in usable]
    unclear = any(might_satisfy_rule(o, rule) for o in every) or any(satisfies_rule(o, rule) for o in held)
    return FindingStatus.UNKNOWN if unclear else FindingStatus.BROKEN


def _step_params(incident_type: IncidentType, rule: ObligationRule, ids: list[str]) -> dict[str, object]:
    return {
        "incident_type": incident_type.value,
        "recipient": rule.recipient,
        "deadline_minutes": rule.deadline_minutes,
        "service_ids": ids,
    }


def _missing_step_findings(
    ctx: CheckContext, info: RuleInfo, incident_type: IncidentType, label: str
) -> Iterator[Finding]:
    """KR-001/003: one org-level finding when no service has the step, else one per service."""
    scope = regulated_scope(ctx)
    index = GraphIndex(ctx.graph)
    unmatched = unmatched_regulated(ctx)
    for rule in applicable_rules(ctx, incident_type):
        if unmatched:
            yield _unmatched_finding(info, incident_type, label, rule, unmatched)
        states = {s.id: st for s in scope if (st := _step_status(ctx, index, s.id, rule)) is not None}
        if not states:
            continue
        # Nobody has the step (and for the same reason) -> one organisation-level finding.
        whole_org = len(states) == len(scope) and len(set(states.values())) == 1
        groups = [scope] if whole_org else [[s] for s in scope if s.id in states]
        for group in groups:
            ids = [s.id for s in group]
            names = listed_names([s.name for s in group])
            status = states[ids[0]]
            message = (
                f"{names} 런북에 {label} 발생 시 {_duty(rule)} 신고하는 단계가 없습니다. "
                "복구와 별개로 신고 기한을 놓치면 법 위반이 됩니다."
                if status is FindingStatus.BROKEN
                else f"{names} 런북의 신고 단계는 어떤 사고에 대한 것인지 적혀 있지 않거나 판단 보류 상태라, "
                f"{label} 발생 시 {_duty(rule)} 신고하는 단계가 있는지 확인할 수 없습니다. "
                "신고 대상 사고 유형을 명시해 주세요."
            )
            yield _with_basis(
                finding(
                    info,
                    keys=(rule.key, *ids),
                    title=info.title,
                    message=message,
                    service_id=None if whole_org else ids[0],
                    subject_ids=ids,
                    fix=FixHint(
                        kind=FixKind.ADD_REPORTING_STEP, params=_step_params(incident_type, rule, ids)
                    ),
                    status=status,
                ),
                rule,
            )


def _unmatched_finding(
    info: RuleInfo, incident_type: IncidentType, label: str, rule: ObligationRule, names: list[str]
) -> Finding:
    """Regulated services the graph lacks: nothing proves they can report — UNKNOWN, not silence."""
    shown = listed_names(names)
    params = {**_step_params(incident_type, rule, []), "service_names": names}
    return _with_basis(
        finding(
            info,
            keys=(rule.key, "unmatched", *names),
            title=info.title,
            message=(
                f"정책의 신고 대상 서비스 중 그래프에서 찾을 수 없는 것이 있습니다: {shown}. "
                f"그래서 {label} 발생 시 {_duty(rule)} 신고하는 단계가 있는지 확인할 수 없습니다. "
                "서비스 이름을 확인하거나 해당 런북을 올려 주세요."
            ),
            fix=FixHint(kind=FixKind.ADD_REPORTING_STEP, params=params),
            status=FindingStatus.UNKNOWN,
        ),
        rule,
    )


def _names(index: GraphIndex, ids: list[str]) -> str:
    return ", ".join(index.graph.nodes[i].name for i in ids)


def _reporter_problem(
    ctx: CheckContext, index: GraphIndex, obligation: Node
) -> tuple[FindingStatus, FixHint, str] | None:
    """Why nobody can file this report, or ``None`` if an active reporter exists.

    Only verified (non-HOLD) reporter links can show that someone will file the report; a held
    link makes the answer UNKNOWN instead of "fine".
    """
    oid = obligation.id
    usable = index.out_edges(oid, EdgeKind.REPORTED_BY)
    people = {p.id: p for e in usable for p in index.resolve_people(e.dst)}
    if any(ctx.is_active(p) for p in people.values()):
        return None
    usable_ids = {e.id for e in usable}
    held = sorted({e.dst for e in index.all_out_edges(oid, EdgeKind.REPORTED_BY) if e.id not in usable_ids})
    status = {pid: ctx.person_status(people[pid]) for pid in sorted(people)}
    described = ", ".join(f"{people[p].name}({_STATUS_KO.get(st, '연락 불가')})" for p, st in status.items())
    unknown = [pid for pid, st in status.items() if st is PersonStatus.UNKNOWN and pid not in ctx.unavailable]
    if unknown or held:
        # 확인 불가 ≠ 정상: the reporter may exist, but we cannot vouch for it.
        reasons = [f"신고 담당자의 재직 여부를 확인할 수 없습니다: {described}"] if unknown else []
        if held:
            reasons.append(f"신고 담당 지정이 판단 보류 상태라 확인할 수 없습니다: {_names(index, held)}")
        params = {"obligation_id": oid, "person_ids": unknown, "held_ids": held}
        return (
            FindingStatus.UNKNOWN,
            FixHint(kind=FixKind.CONFIRM_IDENTITY, params=params),
            "; ".join(reasons),
        )
    if not usable:
        fix = FixHint(kind=FixKind.ASSIGN_OWNER, params={"obligation_id": oid})
        return FindingStatus.BROKEN, fix, "신고 담당자가 지정되어 있지 않습니다"
    if not people:
        targets = [e.dst for e in usable]
        fix = FixHint(kind=FixKind.ASSIGN_OWNER, params={"obligation_id": oid, "reporter_ids": targets})
        return FindingStatus.BROKEN, fix, f"신고 담당({_names(index, targets)})을 맡은 사람이 명부에 없습니다"
    gone = [pid for pid, st in status.items() if st in (PersonStatus.DEPARTED, PersonStatus.ON_LEAVE)]
    fix = (
        FixHint(kind=FixKind.REPLACE_DEPARTED, params={"obligation_id": oid, "person_ids": gone})
        if gone
        else FixHint(kind=FixKind.ADD_DEPUTY, params={"obligation_id": oid})
    )
    return FindingStatus.BROKEN, fix, f"연락 가능한 활동 중 신고 담당자가 없습니다: {described}"


def _reporter_findings(ctx: CheckContext, info: RuleInfo, incident_type: IncidentType) -> Iterator[Finding]:
    """KR-002/004: an obligation exists but nobody active can file it."""
    scope_ids = {s.id for s in regulated_scope(ctx)}
    index = GraphIndex(ctx.graph)
    for rule in applicable_rules(ctx, incident_type):
        for obligation in ctx.graph.nodes_of(NodeKind.OBLIGATION):
            services = [s for s in reporting_services(ctx, obligation.id) if s in scope_ids]
            if not services or not satisfies_rule(obligation, rule):
                continue
            problem = _reporter_problem(ctx, index, obligation)
            if problem is None:
                continue
            status, fix, reason = problem
            reporter_ids = [r.id for r in ctx.reporters(obligation.id)]
            yield _with_basis(
                finding(
                    info,
                    keys=(rule.key, obligation.id),
                    title=info.title,
                    message=(
                        f"'{obligation.name}'({_duty(rule)}) 단계는 있으나 {reason}. "
                        "담당자가 없으면 신고 기한이 그대로 지나갑니다."
                    ),
                    service_id=single_or_none(services),
                    subject_ids=(obligation.id, *reporter_ids, *services),
                    evidence=obligation.spans,
                    fix=fix,
                    status=status,
                ),
                rule,
            )


def _with_basis(f: Finding, rule: ObligationRule) -> Finding:
    """Cite the legal text the organisation configured for this duty, not just the default."""
    return f.model_copy(update={"legal_basis": rule.legal_basis})


# ---------------------------------------------------------------------------- rules


@register
class BreachReportStepMissing:
    info = RuleInfo(
        rule_id="JT-KR-001",
        title="침해사고 24시간 신고 단계 없음",
        severity=Severity.HIGH,
        element=None,
        rationale=(
            "침해사고는 알게 된 때부터 24시간 이내 신고해야 하므로, 대상 서비스 런북에 "
            "침해사고 신고 단계(신고 대상 기관·시한)가 있어야 한다. 서비스를 지정하지 않은 전사 공통 "
            "신고 절차는 모든 서비스의 단계로 본다. 사고 유형이 적혀 있지 않거나 "
            "판단 보류인 신고 단계는 '확인 불가'로 알린다."
        ),
        legal_basis=law_text(IncidentType.BREACH),
    )

    def check(self, ctx: CheckContext) -> Iterator[Finding]:
        return _missing_step_findings(ctx, self.info, IncidentType.BREACH, "침해사고")


@register
class BreachReporterMissing:
    info = RuleInfo(
        rule_id="JT-KR-002",
        title="침해사고 신고 담당 없음/비활성",
        severity=Severity.HIGH,
        element=None,
        rationale="신고 단계가 있어도 실제로 신고할 활동 중인 담당자가 없으면 24시간 기한을 지킬 수 없다.",
        legal_basis=law_text(IncidentType.BREACH),
    )

    def check(self, ctx: CheckContext) -> Iterator[Finding]:
        return _reporter_findings(ctx, self.info, IncidentType.BREACH)


@register
class PrivacyReportStepMissing:
    info = RuleInfo(
        rule_id="JT-KR-003",
        title="개인정보 유출 72시간 신고 단계 없음",
        severity=Severity.HIGH,
        element=None,
        rationale=(
            "개인정보 유출 등은 알게 된 때부터 72시간 이내 통지·신고해야 하므로, 대상 서비스 런북에 "
            "유출 신고 단계가 있어야 한다. 서비스를 지정하지 않은 전사 공통 신고 절차는 모든 서비스의 "
            "단계로 본다. 사고 유형이 적혀 있지 않거나 판단 보류인 신고 단계는 '확인 불가'로 알린다."
        ),
        legal_basis=law_text(IncidentType.PRIVACY_LEAK),
    )

    def check(self, ctx: CheckContext) -> Iterator[Finding]:
        return _missing_step_findings(ctx, self.info, IncidentType.PRIVACY_LEAK, "개인정보 유출")


@register
class PrivacyReporterMissing:
    info = RuleInfo(
        rule_id="JT-KR-004",
        title="개인정보 유출 신고 담당 없음/비활성",
        severity=Severity.HIGH,
        element=None,
        rationale="유출 신고 단계가 있어도 활동 중인 신고 담당자가 없으면 72시간 기한을 지킬 수 없다.",
        legal_basis=law_text(IncidentType.PRIVACY_LEAK),
    )

    def check(self, ctx: CheckContext) -> Iterator[Finding]:
        return _reporter_findings(ctx, self.info, IncidentType.PRIVACY_LEAK)
