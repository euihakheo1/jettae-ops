"""Gold organisation specs — ground truth *by construction* for the sealed evaluation set.

사람 평가자 없이 정답이 정해지는 방식: nobody (no human, no LLM) is asked whether a runbook is
broken. We generate a fully READY organisation spec, render it to Korean runbook text
(:mod:`jettae.evalkit.render`) and break it with a known operator (:mod:`jettae.evalkit.mutate`).
Because the spec is the source of every sentence, the expected findings and the exact set of
graph facts are known exactly.

Design decisions that keep the ground truth exact (each one closes a leak we would otherwise
have to "judge" later):

* Every service has its own team, owner and lead. Escalation edges and contact channels are
  *person-level* facts in the operations graph, so a lead shared by two services would make a
  single-service mutation leak into its sibling.
* Every rollback has two approvers (lead + CTO) and every responder two contact media, so a clean
  org has no single point of failure (JT-SPF-001/002) — "clean" must mean clean under *all* rules.
* Critical services are spread over two locations (no JT-CAT-003 concentration) and at least two
  services are regulated, so deleting one reporting step yields a per-service JT-KR-001.
* A team shares one out-of-band medium (phone/SMS/pager). Removing a fallback channel is then a
  pure deletion for both the owner and the first escalation hop, which JT-PATH-030 pools.

:func:`spec_findings` is a spec-level mirror of the published, service-scoped rule semantics. It
is used to prove that a mutation breaks what it claims (and to list legitimate collateral
findings) — it is NOT the product's judge, which runs on graphs in ``jettae.check``.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from jettae.domain import (
    ActionType,
    ChannelMedium,
    EdgeKind,
    IncidentType,
    PersonStatus,
    Policy,
)

DEFAULT_SEED = 20261012

# --------------------------------------------------------------------------- published rule ids

RULE_DETECTION = "JT-PATH-001"
RULE_NOTIFY_NOBODY = "JT-PATH-002"
RULE_OWNER = "JT-PATH-010"
RULE_APPROVER = "JT-PATH-020"
RULE_FALLBACK = "JT-PATH-030"
RULE_ROLLBACK = "JT-PATH-040"
RULE_EXECUTOR = "JT-PATH-041"
RULE_DEPARTED = "JT-PPL-001"
RULE_UNCONFIRMED = "JT-PPL-002"
RULE_ON_LEAVE = "JT-PPL-003"
RULE_NO_ESCALATION = "JT-ESC-001"
RULE_ESCALATION_CYCLE = "JT-ESC-002"
RULE_ESCALATION_DEAD_END = "JT-ESC-003"
RULE_DEADLINE_LATE = "JT-DDL-002"
RULE_DEADLINE_CONFLICT = "JT-DDL-003"
RULE_BREACH_STEP = "JT-KR-001"
RULE_BREACH_REPORTER = "JT-KR-002"
RULE_PRIVACY_STEP = "JT-KR-003"
RULE_PRIVACY_REPORTER = "JT-KR-004"
RULE_SINGLE_PERSON = "JT-SPF-001"

# --------------------------------------------------------------------------- vocabulary

RECIPIENT_KISA = "KISA"
RECIPIENT_PIPC = "개인정보보호위원회"
#: Canonical recipient per incident type (the name the runbook text uses).
LEGAL_RECIPIENT: Mapping[IncidentType, str] = {
    IncidentType.BREACH: RECIPIENT_KISA,
    IncidentType.PRIVACY_LEAK: RECIPIENT_PIPC,
}
#: Legal deadlines (정보통신망법 24h, 개인정보 보호법 72h) — kept in sync with DEFAULT_OBLIGATIONS.
LEGAL_DEADLINE_HOURS: Mapping[IncidentType, int] = {
    IncidentType.BREACH: 24,
    IncidentType.PRIVACY_LEAK: 72,
}
TERMINAL_ROLES: tuple[str, ...] = Policy().terminal_roles
CRITICAL_TIERS: tuple[int, ...] = Policy().critical_tiers

ROLE_CEO = "대표"
ROLE_CTO = "CTO"
ROLE_CISO = "CISO"
ROLE_LEAD = "팀장"
ROLE_SECURITY = "보안 담당"
ROLE_CPO = "개인정보 보호책임자"
EXEC_TEAM = "경영진"
SECURITY_TEAM = "정보보호팀"
LEAD_TITLE = "팀장"
STAFF_TITLES: tuple[str, ...] = ("사원", "주임", "대리", "과장", "차장")
EXEC_TITLES: frozenset[str] = frozenset({"대표", "CTO", "CISO"})

# --------------------------------------------------------------------------- facts

FactValue = str | int


class Fact(NamedTuple):
    """One edge of the expected operations graph, in canonical names.

    Canonical names: service / system / location / team / role names, roster person names,
    channel medium values (``"slack"``), obligation recipients (``"KISA"``), and the keys of
    :func:`trigger_key` / :func:`action_key` for trigger and action nodes (runbooks name those
    freely, so the harness canonicalises them by the service they hang off). ``attrs`` holds
    only the keys listed in :data:`FACT_ATTR_KEYS`, sorted by key.
    """

    kind: EdgeKind
    src: str
    dst: str
    attrs: tuple[tuple[str, FactValue], ...] = ()

    @property
    def key(self) -> tuple[EdgeKind, str, str]:
        """Identity without attributes — for comparisons that ignore stated numbers."""
        return (self.kind, self.src, self.dst)


#: Attribute keys carried by facts (graph attr vocabulary: AttrKey values, minutes as ints).
#: MITIGATED_BY reads ``action_type`` from the action node, MUST_REPORT reads
#: ``deadline_minutes``/``incident_type`` from the obligation node.
FACT_ATTR_KEYS: Mapping[EdgeKind, tuple[str, ...]] = {
    EdgeKind.ESCALATES_TO: ("after_minutes",),
    EdgeKind.NOTIFIES: ("channel",),
    EdgeKind.MITIGATED_BY: ("action_type",),
    EdgeKind.MUST_REPORT: ("deadline_minutes", "incident_type"),
}


def fact(kind: EdgeKind, src: str, dst: str, **attrs: FactValue) -> Fact:
    """Build a :class:`Fact` with attributes in canonical (sorted) order."""
    return Fact(kind, src, dst, tuple(sorted(attrs.items())))


def trigger_key(service: str) -> str:
    return f"{service}#trigger"


def action_key(service: str, action_type: ActionType) -> str:
    return f"{service}#{action_type.value}"


# --------------------------------------------------------------------------- spec models


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GoldPerson(_Frozen):
    name: str = Field(min_length=2)
    title: str
    team: str
    status: PersonStatus = PersonStatus.ACTIVE
    #: contact media, primary first
    channels: tuple[ChannelMedium, ...] = Field(min_length=1)
    #: given name and optional English nickname — what colloquial text calls the person
    aliases: tuple[str, ...] = ()
    left_on: str | None = None

    @property
    def surname(self) -> str:
        return self.name[0]

    @property
    def given_name(self) -> str:
        return self.name[1:]

    @property
    def is_active(self) -> bool:
        return self.status is PersonStatus.ACTIVE


class GoldEscalationStep(_Frozen):
    target: str
    after_minutes: int = Field(ge=1)


class GoldTrigger(_Frozen):
    condition: str
    delay_minutes: int | None = Field(default=None, ge=1)


class GoldAction(_Frozen):
    type: ActionType
    approvers: tuple[str, ...] = ()
    executor: str | None = None
    duration_minutes: int = Field(ge=1)
    requires_approval: bool = True


class GoldFallback(_Frozen):
    primary_channel: ChannelMedium
    fallback_channel: ChannelMedium | None = None

    @model_validator(mode="after")
    def _distinct(self) -> GoldFallback:
        if self.fallback_channel == self.primary_channel:
            raise ValueError("대체 채널이 주 채널과 같습니다")
        return self


class GoldObligation(_Frozen):
    incident_type: IncidentType
    recipient: str
    deadline_hours: int = Field(ge=1)
    reporter: str | None = None


class GoldService(_Frozen):
    name: str
    tier: int = Field(ge=1, le=3)
    team: str
    systems: tuple[str, ...] = Field(min_length=1)
    location: str
    owner: str | None
    #: the person the alert pages and the root of the escalation chain (== owner when clean)
    first_responder: str
    notify_channel: ChannelMedium
    escalation: tuple[GoldEscalationStep, ...] = ()
    trigger: GoldTrigger | None = None
    action: GoldAction | None = None
    fallback: GoldFallback
    obligations: tuple[GoldObligation, ...] = ()
    regulated: bool = False

    def escalation_edges(self) -> tuple[tuple[str, str, int], ...]:
        """(source, target, after_minutes) per hop; hop *k* starts where hop *k-1* ended."""
        out: list[tuple[str, str, int]] = []
        src = self.first_responder
        for step in self.escalation:
            out.append((src, step.target, step.after_minutes))
            src = step.target
        return tuple(out)

    def referenced_people(self) -> tuple[str, ...]:
        """Everyone the service text names, in order of first mention (deduplicated)."""
        names: list[str | None] = [self.owner, self.first_responder]
        names += [s.target for s in self.escalation]
        if self.action is not None:
            names += [*self.action.approvers, self.action.executor]
        names += [o.reporter for o in self.obligations]
        return tuple(dict.fromkeys(n for n in names if n is not None))

    def obligation(self, incident_type: IncidentType) -> GoldObligation | None:
        return next((o for o in self.obligations if o.incident_type is incident_type), None)


class GoldOrg(_Frozen):
    org_id: str
    theme: str
    company: str
    handles_personal_info: bool
    chat_medium: ChannelMedium
    locations: tuple[str, ...] = Field(min_length=1)
    people: tuple[GoldPerson, ...]
    #: role -> holder names (sorted)
    roles: dict[str, tuple[str, ...]]
    services: tuple[GoldService, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> GoldOrg:
        names = [p.name for p in self.people]
        _require_unique(names, "인물 이름")
        aliases = [a for p in self.people for a in p.aliases]
        _require_unique(aliases, "별칭")
        if clash := sorted(set(aliases) & set(names)):
            raise ValueError(f"별칭이 다른 사람 이름과 겹칩니다: {clash}")
        if ambiguous := _ambiguous_aliases(self):
            raise ValueError(
                f"별칭이 다른 이름이나 문서 용어 안에 들어 있어 한 사람으로 풀리지 않습니다: {ambiguous}"
            )
        known = set(names)
        for role, holders in self.roles.items():
            if missing := sorted(set(holders) - known):
                raise ValueError(f"역할 {role}의 보유자가 명부에 없습니다: {missing}")
        _require_unique([s.name for s in self.services], "서비스 이름")
        _require_unique([sys for s in self.services for sys in s.systems], "시스템 이름")
        for s in self.services:
            if missing := sorted(set(s.referenced_people()) - known):
                raise ValueError(f"{s.name}: 명부에 없는 인물 {missing}")
            if s.location not in self.locations:
                raise ValueError(f"{s.name}: 알 수 없는 위치 {s.location}")
        return self

    # ---- lookups ----------------------------------------------------------
    def person(self, name: str) -> GoldPerson:
        for p in self.people:
            if p.name == name:
                return p
        raise KeyError(name)

    def service(self, name: str) -> GoldService:
        for s in self.services:
            if s.name == name:
                return s
        raise KeyError(name)

    def roles_of(self, name: str) -> tuple[str, ...]:
        return tuple(sorted(role for role, holders in self.roles.items() if name in holders))

    def is_terminal(self, name: str) -> bool:
        """Holds a terminal role or title (CTO·대표·CISO …) — ends an escalation chain."""
        terminal = {t.casefold() for t in TERMINAL_ROLES}
        labels = {self.person(name).title, *self.roles_of(name)}
        return bool(terminal & {x.casefold() for x in labels})

    def replace_service(self, service: GoldService) -> GoldOrg:
        """New org with ``service`` swapped in by name (re-validated)."""
        services = tuple(service if s.name == service.name else s for s in self.services)
        return self._revalidated(services=services)

    def replace_people(self, *people: GoldPerson) -> GoldOrg:
        """New org with the given people swapped in by name (re-validated)."""
        by_name = {p.name: p for p in people}
        return self._revalidated(people=tuple(by_name.get(p.name, p) for p in self.people))

    def _revalidated(self, **update: object) -> GoldOrg:
        data = self.model_dump()
        data.update(update)
        return GoldOrg.model_validate(data)


def _require_unique(values: list[str], what: str) -> None:
    if dupes := sorted(v for v, count in Counter(values).items() if count > 1):
        raise ValueError(f"{what} 중복: {dupes}")


def _aliases_inside(texts: Iterable[str], aliases: Iterable[str]) -> list[str]:
    """The ``aliases`` that occur inside any of ``texts`` (case-insensitive, for Latin nicknames)."""
    haystack = "\n".join(texts).casefold()
    return [a for a in aliases if a.casefold() in haystack]


def _ambiguous_aliases(org: GoldOrg) -> list[str]:
    """Aliases that also occur inside another person's name or alias, or in the org's vocabulary.

    Colloquial text names people by alias ("지원 사원", "Kate님"), so an alias must point at one
    person wherever it appears: "정우" inside "정우진", "Ian" inside "Brian", or "지원" inside the
    service "지원 접수" would let a correct alias resolver bind text to the wrong person — noise
    the ground truth cannot account for.
    """
    terms = [
        *org.roles,
        *(p.team for p in org.people),
        *(t for s in org.services for t in (s.name, s.team, s.location, *s.systems)),
        *(s.trigger.condition for s in org.services if s.trigger is not None),
        org.company,
        org.theme,
    ]
    out: set[str] = set()
    for p in org.people:
        others = [t for q in org.people if q.name != p.name for t in (q.name, *q.aliases)]
        out.update(_aliases_inside([*others, *terms], p.aliases))
    return sorted(out)


# --------------------------------------------------------------------------- facts of a spec


def roster_facts(org: GoldOrg) -> frozenset[Fact]:
    """Facts the roster states: team membership, roles and registered contact media."""
    out: set[Fact] = set()
    for p in org.people:
        out.add(fact(EdgeKind.MEMBER_OF, p.name, p.team))
        out.update(fact(EdgeKind.REACHABLE_VIA, p.name, m.value) for m in p.channels)
    for role, holders in org.roles.items():
        out.update(fact(EdgeKind.HOLDS_ROLE, h, role) for h in holders)
    return frozenset(out)


def catalog_facts(service: GoldService) -> tuple[Fact, ...]:
    """Facts the catalog (and the runbook's 구성 line) states for one service."""
    out = [fact(EdgeKind.RUNS_IN, service.name, service.location)]
    for sys in service.systems:
        out.append(fact(EdgeKind.DEPENDS_ON, service.name, sys))
        out.append(fact(EdgeKind.RUNS_IN, sys, service.location))
    return tuple(out)


def response_facts(service: GoldService) -> tuple[Fact, ...]:
    """Facts only the runbook states: the response path of one service, in reading order."""
    name, fr, fb = service.name, service.first_responder, service.fallback
    out: list[Fact] = []
    if service.owner is not None:
        out.append(fact(EdgeKind.OWNED_BY, name, service.owner))
    out.append(fact(EdgeKind.REACHABLE_VIA, fr, fb.primary_channel.value))
    if fb.fallback_channel is not None:
        out.append(fact(EdgeKind.REACHABLE_VIA, fr, fb.fallback_channel.value))
        out.append(fact(EdgeKind.FALLBACK_TO, fb.primary_channel.value, fb.fallback_channel.value))
    if service.trigger is not None:
        tk = trigger_key(name)
        out.append(fact(EdgeKind.DETECTED_BY, name, tk))
        out.append(fact(EdgeKind.NOTIFIES, tk, fr, channel=service.notify_channel.value))
    for src, dst, after in service.escalation_edges():
        out.append(fact(EdgeKind.ESCALATES_TO, src, dst, after_minutes=after))
    if service.action is not None:
        ak = action_key(name, service.action.type)
        out.append(fact(EdgeKind.MITIGATED_BY, name, ak, action_type=service.action.type.value))
        out += [fact(EdgeKind.APPROVED_BY, ak, a) for a in service.action.approvers]
        if service.action.executor is not None:
            out.append(fact(EdgeKind.EXECUTED_BY, ak, service.action.executor))
    for ob in service.obligations:
        out.append(
            fact(
                EdgeKind.MUST_REPORT,
                name,
                ob.recipient,
                deadline_minutes=ob.deadline_hours * 60,
                incident_type=ob.incident_type.value,
            )
        )
        if ob.reporter is not None:
            out.append(fact(EdgeKind.REPORTED_BY, ob.recipient, ob.reporter))
    return tuple(out)


def runbook_facts(org: GoldOrg) -> frozenset[Fact]:
    """Every fact the rendered runbook states (each one carries a source span in the render)."""
    return frozenset(f for s in org.services for f in (*catalog_facts(s), *response_facts(s)))


def gold_facts(org: GoldOrg) -> frozenset[Fact]:
    """The complete expected fact set of the input bundle (runbook ∪ roster ∪ catalog)."""
    return runbook_facts(org) | roster_facts(org)


# --------------------------------------------------------------------------- spec-level oracle


class SpecFinding(NamedTuple):
    """A rule the published semantics fire for ``service`` (``None`` = organisation level)."""

    service: str | None
    rule_id: str


def spec_findings(org: GoldOrg) -> frozenset[SpecFinding]:
    """Rules that fire on this spec, mirroring the published rule semantics at spec level.

    Covers the service-scoped rules our operators can affect (PATH, PPL-001..003, ESC-001..003,
    DDL-002, KR-001..004, SPF-001) plus two organisation-level findings: the JT-DDL-003
    deadline conflict, and a PPL rule for an escalation hop that no service's owner chain
    reaches (the published rule cannot tie such a duty to a service).
    A clean org yields the empty set — that is the readiness invariant the generator guarantees.
    """
    out = {SpecFinding(s.name, rule) for s in org.services for rule in _service_rules(org, s)}
    chained = {p for s in org.services for p in _chain(s)}
    orphans = [org.person(dst).status for src, dst in _escalation_hops(org) if src not in chained]
    out.update(SpecFinding(None, rule) for status, rule in _STATUS_RULES.items() if status in orphans)
    if _deadline_conflict(org):
        out.add(SpecFinding(None, RULE_DEADLINE_CONFLICT))
    return frozenset(out)


#: Status-specific people rules: a departed, on-leave or unconfirmed duty holder is a different
#: finding each (unknown is reported as unknown, never folded into "departed" — P3).
_STATUS_RULES: Mapping[PersonStatus, str] = {
    PersonStatus.DEPARTED: RULE_DEPARTED,
    PersonStatus.UNKNOWN: RULE_UNCONFIRMED,
    PersonStatus.ON_LEAVE: RULE_ON_LEAVE,
}


def _service_rules(org: GoldOrg, s: GoldService) -> Iterator[str]:
    statuses = {org.person(p).status for p in _duty_people(org, s)}
    yield from (rule for status, rule in _STATUS_RULES.items() if status in statuses)
    for ob in s.obligations:
        if ob.deadline_hours > LEGAL_DEADLINE_HOURS[ob.incident_type]:
            yield RULE_DEADLINE_LATE
    if s.regulated:
        yield from _reporting_rules(org, s)
    if s.tier in CRITICAL_TIERS:
        yield from _path_rules(org, s)
        yield from _escalation_rules(org, s)
        if _single_person_point(org, s):
            yield RULE_SINGLE_PERSON


def _duty_people(org: GoldOrg, s: GoldService) -> tuple[str, ...]:
    """People that hold a response duty edge of ``s`` (담당·호출·상향 대상·승인·실행·신고).

    An escalation hop is a duty of every service whose owner chain contains the hop's source —
    the same attribution the published people rules use.
    """
    names: list[str | None] = [s.owner, s.first_responder if s.trigger else None]
    chain = _chain(s)
    names += [dst for src, dst in _escalation_hops(org) if src in chain]
    if s.action is not None:
        names += [*s.action.approvers, s.action.executor]
    names += [o.reporter for o in s.obligations]
    return tuple(dict.fromkeys(n for n in names if n is not None))


def _path_rules(org: GoldOrg, s: GoldService) -> Iterator[str]:
    if s.trigger is None:
        yield RULE_DETECTION
    elif not org.person(s.first_responder).is_active:
        yield RULE_NOTIFY_NOBODY
    if s.owner is None or not org.person(s.owner).is_active:
        yield RULE_OWNER
    if len(_fallback_pool(org, s)) < 2:
        yield RULE_FALLBACK
    if s.action is None:
        yield RULE_ROLLBACK
        return
    if s.action.requires_approval and not _active(org, s.action.approvers):
        yield RULE_APPROVER
    if s.action.executor is None or not org.person(s.action.executor).is_active:
        yield RULE_EXECUTOR


def _escalation_rules(org: GoldOrg, s: GoldService) -> Iterator[str]:
    if s.owner is None:
        return
    chain = _owner_chain(s)
    if not chain:
        yield RULE_NO_ESCALATION
        return
    if not _active(org, [dst for _, dst, _ in chain if dst != s.owner]):
        yield RULE_ESCALATION_DEAD_END
    if _cycle_without_exit(org, s.owner, chain):
        yield RULE_ESCALATION_CYCLE


def _owner_chain(s: GoldService) -> tuple[tuple[str, str, int], ...]:
    """Escalation edges reachable from the owner (the chain is rooted at the first responder)."""
    return s.escalation_edges() if s.owner == s.first_responder else ()


def _chain(s: GoldService) -> frozenset[str]:
    """The owner and everyone the owner's escalation chain reaches (who may be paged for ``s``)."""
    if s.owner is None:
        return frozenset()
    return frozenset({s.owner, *(dst for _, dst, _ in _owner_chain(s))})


def _escalation_hops(org: GoldOrg) -> list[tuple[str, str]]:
    """Distinct (source, target) escalation hops of the whole org, sorted."""
    return sorted({(src, dst) for s in org.services for src, dst, _ in s.escalation_edges()})


def _cycle_without_exit(org: GoldOrg, root: str, chain: tuple[tuple[str, str, int], ...]) -> bool:
    walk = [root]
    for _, dst, _ in chain:
        if dst in walk:
            members = walk[walk.index(dst) :]
            return not any(org.is_terminal(m) and org.person(m).is_active for m in members)
        walk.append(dst)
    return False


def _fallback_pool(org: GoldOrg, s: GoldService) -> set[ChannelMedium]:
    """Media reaching the active owner and first escalation hop, plus org-wide FALLBACK_TO closure."""
    if s.owner is None:
        return set()
    chain = _owner_chain(s)
    people = [s.owner, *([chain[0][1]] if chain else [])]
    media = {m for p in _active(org, people) for m in org.person(p).channels}
    pairs = {
        (x.fallback.primary_channel, x.fallback.fallback_channel)
        for x in org.services
        if x.fallback.fallback_channel is not None
    }
    grown = True
    while grown:
        extra = {fb for primary, fb in pairs if primary in media and fb is not None} - media
        media |= extra
        grown = bool(extra)
    return media


def _single_person_point(org: GoldOrg, s: GoldService) -> bool:
    """A working response path that hangs on exactly one person (responder or approver pool)."""
    roots = [n for n in (s.owner, s.first_responder if s.trigger else None) if n is not None]
    closure = [step.target for step in s.escalation] if s.first_responder in roots else []
    pools = [set(_active(org, [*roots, *closure]))]
    if s.action is not None and s.action.requires_approval:
        pools.append(set(_active(org, s.action.approvers)))
    return all(pools) and any(len(p) == 1 for p in pools)


def _reporting_rules(org: GoldOrg, s: GoldService) -> Iterator[str]:
    required = [(IncidentType.BREACH, RULE_BREACH_STEP, RULE_BREACH_REPORTER)]
    if org.handles_personal_info:
        required.append((IncidentType.PRIVACY_LEAK, RULE_PRIVACY_STEP, RULE_PRIVACY_REPORTER))
    for incident, step_rule, reporter_rule in required:
        ob = s.obligation(incident)
        if ob is None or ob.recipient != LEGAL_RECIPIENT[incident]:
            yield step_rule
        elif ob.reporter is None or not org.person(ob.reporter).is_active:
            yield reporter_rule


def _deadline_conflict(org: GoldOrg) -> bool:
    for incident in IncidentType:
        stated = {
            o.deadline_hours for s in org.services for o in s.obligations if o.incident_type is incident
        }
        if len(stated) > 1:
            return True
    return False


def _active(org: GoldOrg, names: list[str] | tuple[str, ...]) -> list[str]:
    return [n for n in dict.fromkeys(names) if org.person(n).is_active]


# --------------------------------------------------------------------------- themes


@dataclass(frozen=True)
class _ServiceTemplate:
    name: str
    team: str
    systems: tuple[str, str]
    condition: str
    core: bool = False
    action: ActionType = ActionType.ROLLBACK


@dataclass(frozen=True)
class _Theme:
    key: str
    label: str
    companies: tuple[str, ...]
    personal_info: bool
    services: tuple[_ServiceTemplate, ...]


def _t(
    name: str,
    team: str,
    systems: tuple[str, str],
    condition: str,
    *,
    core: bool = False,
    failover: bool = False,
) -> _ServiceTemplate:
    action = ActionType.FAILOVER if failover else ActionType.ROLLBACK
    return _ServiceTemplate(name, team, systems, condition, core, action)


# fmt: off
THEMES: tuple[_Theme, ...] = (
    _Theme("shop", "쇼핑몰", ("마포마켓", "망원상회몰", "합정스토어"), True, (
        _t("결제", "결제팀", ("pay-api", "pay-db"), "결제 실패율 5% 초과", core=True),
        _t("주문", "주문팀", ("order-api", "order-db"), "주문 생성 API 5xx 비율 2% 초과", core=True),
        _t("회원 인증", "회원팀", ("auth-api", "session-redis"), "로그인 실패율 10% 초과", core=True),
        _t("상품 검색", "검색팀", ("search-api", "search-es"), "검색 응답 p95 2초 초과"),
        _t("배송 추적", "물류연동팀", ("tracking-api", "carrier-gw"), "택배사 연동 오류율 5% 초과"),
        _t("쿠폰", "프로모션팀", ("coupon-api", "coupon-db"), "쿠폰 적용 실패 분당 50건 초과"),
    )),
    _Theme("ott", "OTT 스트리밍", ("상암플릭스", "DMC스트림"), True, (
        _t("영상 재생", "플레이어팀", ("playback-api", "cdn-origin"), "재생 시작 실패율 3% 초과", core=True),
        _t("구독 결제", "빌링팀", ("billing-api", "billing-db"), "정기결제 실패율 5% 초과", core=True),
        _t("로그인", "계정팀", ("account-api", "account-db"), "로그인 오류율 5% 초과", core=True),
        _t("추천", "추천팀", ("reco-api", "feature-store"), "추천 API 타임아웃 비율 10% 초과"),
        _t("콘텐츠 업로드", "미디어팀", ("ingest-worker", "transcode-farm"), "트랜스코딩 대기열 500건 초과"),
        _t("자막", "로컬라이제이션팀", ("subtitle-api", "subtitle-store"), "자막 로딩 실패율 5% 초과"),
    )),
    _Theme("broadcast", "방송 송출", ("DMC송출센터", "상암브로드캐스트"), False, (
        _t("주조정실 송출", "송출운영팀", ("playout-server", "master-switcher"), "송출 신호 끊김(블랙 화면)",
           core=True, failover=True),
        _t("광고 송출", "광고운영팀", ("ad-insert-server", "ad-scheduler"), "광고 삽입 누락 1건 이상",
           core=True),
        _t("실시간 스트리밍", "스트리밍팀", ("live-encoder", "live-origin"), "스트림 비트레이트 50% 이하",
           core=True, failover=True),
        _t("편성 관리", "편성팀", ("schedule-api", "schedule-db"), "편성표 동기화 실패"),
        _t("자막 송출", "자막팀", ("caption-encoder", "caption-api"), "자막 지연 10초 초과"),
        _t("다시보기", "VOD팀", ("vod-packager", "vod-cdn"), "VOD 패키징 실패율 5% 초과"),
    )),
    _Theme("fintech", "핀테크 결제", ("공덕페이", "마포머니"), True, (
        _t("간편결제", "결제코어팀", ("payment-core", "ledger-db"), "결제 승인 실패율 2% 초과", core=True),
        _t("송금", "송금팀", ("transfer-api", "bank-gw"), "송금 지연 분당 30건 초과", core=True),
        _t("본인인증", "인증팀", ("kyc-api", "kyc-vendor-gw"), "본인인증 실패율 10% 초과", core=True),
        _t("정산", "정산팀", ("settlement-batch", "settlement-db"), "정산 배치 30분 이상 지연"),
        _t("이상거래 탐지", "FDS팀", ("fds-engine", "fds-rules-db"), "FDS 판정 지연 p95 1초 초과"),
        _t("가맹점 관리", "가맹점팀", ("merchant-api", "merchant-db"), "가맹점 API 5xx 비율 3% 초과"),
    )),
    _Theme("game", "게임", ("망원게임즈", "DMC플레이"), True, (
        _t("게임 서버", "서버팀", ("game-server", "match-maker"), "동시 접속자 30% 급감", core=True),
        _t("인앱 결제", "인앱결제팀", ("iap-api", "receipt-verifier"), "영수증 검증 실패율 5% 초과",
           core=True),
        _t("계정 로그인", "계정팀", ("login-api", "account-db"), "로그인 실패율 10% 초과", core=True),
        _t("랭킹", "라이브팀", ("ranking-api", "ranking-redis"), "랭킹 갱신 5분 이상 지연"),
        _t("채팅", "커뮤니티팀", ("chat-gateway", "chat-redis"), "채팅 메시지 유실률 1% 초과"),
        _t("패치 배포", "빌드팀", ("patch-cdn", "patch-manifest"), "패치 다운로드 실패율 5% 초과"),
    )),
    _Theme("edtech", "에듀테크", ("연남클래스", "합정에듀"), True, (
        _t("강의 재생", "플레이어팀", ("lecture-player", "lecture-cdn"), "강의 재생 실패율 3% 초과",
           core=True),
        _t("수강 결제", "결제팀", ("enroll-pay-api", "enroll-db"), "수강 결제 실패율 5% 초과", core=True),
        _t("실시간 수업", "라이브팀", ("live-class-sfu", "signaling-api"), "실시간 수업 접속 실패율 5% 초과",
           core=True),
        _t("과제 제출", "학습팀", ("assignment-api", "file-store"), "과제 업로드 오류율 5% 초과"),
        _t("학습 리포트", "데이터팀", ("report-batch", "analytics-db"), "리포트 배치 1시간 이상 지연"),
        _t("알림 발송", "메시징팀", ("push-api", "push-queue"), "푸시 발송 대기열 1만 건 초과"),
    )),
    _Theme("logistics", "물류", ("상암로지스", "마포풀필먼트"), True, (
        _t("배차", "배차팀", ("dispatch-api", "dispatch-db"), "배차 실패율 5% 초과", core=True),
        _t("운송장 발행", "운송장팀", ("waybill-api", "label-printer-gw"), "운송장 발행 오류율 3% 초과",
           core=True),
        _t("창고 관리", "WMS팀", ("wms-api", "wms-db"), "입출고 처리 15분 이상 지연", core=True),
        _t("배송 조회", "조회팀", ("tracking-query-api", "tracking-cache"), "배송 조회 응답 p95 3초 초과"),
        _t("운임 정산", "정산팀", ("freight-batch", "freight-db"), "운임 정산 배치 실패"),
        _t("기사 앱", "모바일팀", ("driver-app-api", "driver-push"), "기사 앱 API 5xx 비율 2% 초과"),
    )),
    _Theme("webtoon", "웹툰 플랫폼", ("망원툰", "DMC코믹스"), True, (
        _t("작품 뷰어", "뷰어팀", ("viewer-api", "image-cdn"), "뷰어 이미지 로딩 실패율 3% 초과", core=True),
        _t("코인 결제", "결제팀", ("coin-api", "coin-ledger"), "코인 충전 실패율 5% 초과", core=True),
        _t("작가 정산", "정산팀", ("creator-settle", "settle-db"), "작가 정산 배치 1시간 이상 지연",
           core=True),
        _t("연재 업로드", "연재운영팀", ("upload-api", "upload-queue"), "연재 업로드 처리 실패 3건 이상"),
        _t("댓글", "커뮤니티팀", ("comment-api", "comment-db"), "댓글 API 5xx 비율 5% 초과"),
        _t("작품 추천", "추천팀", ("reco-api", "reco-feature"), "추천 응답 타임아웃 10% 초과"),
    )),
    _Theme("adtech", "광고 플랫폼", ("합정애즈", "상암애드"), True, (
        _t("광고 입찰", "입찰팀", ("bidder", "bid-cache"), "입찰 응답 100ms 초과 비율 5% 초과", core=True),
        _t("광고 과금", "과금팀", ("ad-billing-api", "billing-ledger"), "과금 이벤트 유실률 0.5% 초과",
           core=True),
        _t("타기팅", "데이터팀", ("segment-api", "segment-store"), "세그먼트 조회 실패율 5% 초과", core=True),
        _t("리포팅", "리포팅팀", ("report-api", "olap-db"), "리포트 집계 30분 이상 지연"),
        _t("광고주 콘솔", "콘솔팀", ("console-web", "console-api"), "콘솔 API 5xx 비율 3% 초과"),
        _t("소재 심사", "심사팀", ("review-queue", "review-api"), "심사 대기열 1천 건 초과"),
    )),
    _Theme("health-saas", "헬스케어 SaaS", ("공덕헬스", "마포메디"), True, (
        _t("진료 예약", "예약팀", ("booking-api", "booking-db"), "예약 생성 실패율 3% 초과", core=True),
        _t("전자 처방", "처방팀", ("rx-api", "rx-db"), "처방 전송 실패 1건 이상", core=True),
        _t("비대면 진료", "원격진료팀", ("tele-sfu", "tele-signal"), "화상 연결 실패율 5% 초과", core=True),
        _t("병원 연동", "연동팀", ("emr-gw", "hl7-broker"), "EMR 연동 오류율 2% 초과"),
        _t("보험 청구", "청구팀", ("claim-api", "claim-db"), "청구 배치 실패"),
        _t("알림톡", "메시징팀", ("notify-api", "notify-queue"), "알림톡 발송 실패율 5% 초과"),
    )),
    _Theme("music", "음원 스트리밍", ("망원뮤직", "합정사운드"), True, (
        _t("음원 재생", "재생팀", ("stream-api", "audio-cdn"), "재생 실패율 2% 초과", core=True),
        _t("이용권 결제", "빌링팀", ("subscription-api", "subscription-db"), "이용권 결제 실패율 5% 초과",
           core=True),
        _t("로그인", "계정팀", ("auth-api", "auth-db"), "로그인 실패율 10% 초과", core=True),
        _t("차트", "차트팀", ("chart-batch", "chart-cache"), "차트 갱신 10분 이상 지연"),
        _t("저작권 정산", "정산팀", ("royalty-batch", "royalty-db"), "저작권 정산 배치 실패"),
        _t("가사", "메타데이터팀", ("lyrics-api", "lyrics-db"), "가사 조회 오류율 5% 초과"),
    )),
    _Theme("live-commerce", "라이브 커머스", ("상암라이브", "연남라이브샵"), True, (
        _t("라이브 방송", "방송팀", ("live-ingest", "live-cdn"), "라이브 송출 30초 이상 끊김",
           core=True, failover=True),
        _t("주문 결제", "결제팀", ("checkout-api", "order-db"), "결제 실패율 5% 초과", core=True),
        _t("상품 재고", "재고팀", ("inventory-api", "inventory-db"), "재고 차감 오류 분당 10건 초과",
           core=True),
        _t("실시간 채팅", "채팅팀", ("chat-ws", "chat-redis"), "채팅 연결 실패율 5% 초과"),
        _t("셀러 센터", "셀러팀", ("seller-api", "seller-db"), "셀러 센터 5xx 비율 3% 초과"),
        _t("쿠폰", "마케팅팀", ("coupon-api", "coupon-cache"), "쿠폰 발급 실패율 5% 초과"),
    )),
    _Theme("recruit", "채용 플랫폼", ("공덕잡스", "마포커리어"), True, (
        _t("이력서", "이력서팀", ("resume-api", "resume-store"), "이력서 저장 실패율 2% 초과", core=True),
        _t("공고 검색", "검색팀", ("job-search", "job-index"), "검색 응답 p95 2초 초과", core=True),
        _t("기업 결제", "결제팀", ("corp-billing", "billing-db"), "기업 결제 실패율 5% 초과", core=True),
        _t("지원 접수", "지원팀", ("apply-api", "apply-db"), "지원 접수 오류율 3% 초과"),
        _t("메일 알림", "메시징팀", ("notify-api", "mail-relay"), "메일 발송 실패율 5% 초과"),
        _t("면접 일정", "일정팀", ("interview-api", "calendar-sync"), "일정 동기화 실패 10건 이상"),
    )),
    _Theme("realestate", "부동산 중개", ("망원집터", "상암하우스"), True, (
        _t("매물 검색", "검색팀", ("listing-search", "listing-index"), "매물 검색 오류율 3% 초과", core=True),
        _t("전자 계약", "계약팀", ("contract-api", "esign-gw"), "전자계약 서명 실패 3건 이상", core=True),
        _t("회원", "회원팀", ("member-api", "member-db"), "회원 API 5xx 비율 2% 초과", core=True),
        _t("시세", "데이터팀", ("price-batch", "price-db"), "시세 배치 1시간 이상 지연"),
        _t("매물 등록", "등록팀", ("listing-api", "image-store"), "매물 등록 실패율 5% 초과"),
        _t("상담 채팅", "상담팀", ("consult-chat", "consult-db"), "상담 연결 실패율 5% 초과"),
    )),
    _Theme("travel", "여행 예약", ("합정트래블", "DMC투어"), True, (
        _t("항공 예약", "항공팀", ("air-booking", "gds-gw"), "GDS 응답 실패율 5% 초과", core=True),
        _t("숙소 예약", "숙소팀", ("stay-booking", "stay-db"), "숙소 예약 실패율 3% 초과", core=True),
        _t("결제", "결제팀", ("pay-api", "pay-ledger"), "결제 실패율 5% 초과", core=True),
        _t("일정 관리", "일정팀", ("itinerary-api", "itinerary-db"), "일정 저장 오류율 5% 초과"),
        _t("여행 후기", "커뮤니티팀", ("review-api", "review-db"), "후기 API 5xx 비율 5% 초과"),
        _t("환불", "CS팀", ("refund-api", "refund-queue"), "환불 대기열 200건 초과"),
    )),
    _Theme("delivery", "배달", ("마포배달", "공덕딜리버리"), True, (
        _t("주문 접수", "주문팀", ("order-api", "order-db"), "주문 접수 실패율 2% 초과", core=True),
        _t("라이더 배차", "배차팀", ("dispatch-engine", "rider-location"), "5분 이상 미배차 주문 100건 이상",
           core=True),
        _t("결제", "결제팀", ("pay-api", "pay-db"), "결제 실패율 5% 초과", core=True),
        _t("가게 관리", "사장님팀", ("store-api", "store-db"), "가게 API 5xx 비율 3% 초과"),
        _t("리뷰", "리뷰팀", ("review-api", "review-db"), "리뷰 등록 오류율 5% 초과"),
        _t("푸시 알림", "메시징팀", ("push-api", "push-queue"), "푸시 대기열 5만 건 초과"),
    )),
    _Theme("secondhand", "중고거래", ("망원중고", "연남마켓"), True, (
        _t("안전결제", "결제팀", ("escrow-api", "escrow-ledger"), "안전결제 실패율 3% 초과", core=True),
        _t("채팅 거래", "채팅팀", ("chat-api", "chat-store"), "채팅 전송 실패율 2% 초과", core=True),
        _t("게시글", "게시글팀", ("post-api", "post-db"), "게시글 등록 실패율 5% 초과", core=True),
        _t("동네 인증", "위치팀", ("geo-verify", "geo-db"), "위치 인증 실패율 10% 초과"),
        _t("신고 처리", "운영정책팀", ("report-queue", "moderation-api"), "신고 대기열 500건 초과"),
        _t("검색", "검색팀", ("search-api", "search-index"), "검색 응답 p95 2초 초과"),
    )),
    _Theme("civil-service", "공공 민원 위탁", ("마포민원센터", "상암행정정보"), True, (
        _t("민원 접수", "민원팀", ("petition-api", "petition-db"), "민원 접수 실패율 2% 초과", core=True),
        _t("본인 확인", "인증팀", ("gpki-gw", "identity-api"), "본인 확인 실패율 10% 초과", core=True),
        _t("증명서 발급", "발급팀", ("doc-issuer", "doc-store"), "증명서 발급 실패 5건 이상", core=True),
        _t("수수료 결제", "결제팀", ("fee-pay", "fee-db"), "수수료 결제 실패율 5% 초과"),
        _t("알림 문자", "메시징팀", ("sms-gw", "sms-queue"), "문자 발송 실패율 5% 초과"),
        _t("처리 현황 조회", "조회팀", ("status-api", "status-cache"), "조회 응답 p95 3초 초과"),
    )),
    _Theme("crowdfunding", "크라우드펀딩", ("합정펀딩", "상암크라우드"), True, (
        _t("펀딩 결제", "결제팀", ("funding-pay", "funding-ledger"), "결제 예약 실패율 3% 초과", core=True),
        _t("프로젝트", "프로젝트팀", ("project-api", "project-db"), "프로젝트 페이지 5xx 비율 2% 초과",
           core=True),
        _t("투자자 인증", "인증팀", ("investor-kyc", "kyc-db"), "투자자 인증 실패율 10% 초과", core=True),
        _t("정산", "정산팀", ("payout-batch", "payout-db"), "정산 배치 실패"),
        _t("리워드 배송", "리워드팀", ("reward-api", "reward-db"), "배송 정보 동기화 실패 10건 이상"),
        _t("알림", "메시징팀", ("notice-api", "notice-queue"), "알림 발송 10분 이상 지연"),
    )),
    _Theme("cloud-msp", "클라우드 MSP", ("DMC클라우드", "마포MSP"), False, (
        _t("고객 관제", "관제팀", ("noc-console", "alert-router"), "관제 알람 수신 5분 이상 중단",
           core=True, failover=True),
        _t("백업", "백업팀", ("backup-scheduler", "backup-vault"), "백업 작업 실패 1건 이상", core=True),
        _t("고객 포털", "포털팀", ("portal-web", "portal-api"), "포털 5xx 비율 3% 초과", core=True),
        _t("자동 프로비저닝", "자동화팀", ("provision-api", "terraform-runner"), "프로비저닝 실패율 10% 초과",
           ),
        _t("비용 정산", "빌링팀", ("cost-batch", "cost-db"), "비용 집계 배치 2시간 이상 지연"),
        _t("장애 티켓", "서비스데스크팀", ("ticket-api", "ticket-db"), "티켓 생성 오류 10건 이상"),
    )),
)
# fmt: on

# --------------------------------------------------------------------------- name pools

#: (surname, weight) — common Korean surnames. Single syllable so given name = name[1:].
SURNAMES: tuple[tuple[str, int], ...] = (
    ("김", 21), ("이", 15), ("박", 8), ("최", 5), ("정", 5), ("강", 2), ("조", 2), ("윤", 2), ("장", 2),
    ("임", 2), ("한", 1), ("오", 1), ("서", 1), ("신", 1), ("권", 1), ("황", 1), ("안", 1), ("송", 1),
    ("류", 1), ("전", 1), ("고", 1), ("문", 1), ("양", 1), ("손", 1), ("배", 1), ("백", 1), ("허", 1),
    ("남", 1), ("심", 1), ("노", 1),
)  # fmt: skip
GIVEN_NAMES: tuple[str, ...] = (
    "민수", "지훈", "서준", "유진", "지민", "서연", "하준", "도윤", "지우", "수빈", "예준", "시우",
    "하은", "서윤", "지호", "은우", "민준", "서현", "지유", "채원", "현우", "준서", "다은", "수아",
    "건우", "윤서", "우진", "예린", "태윤", "소윤", "지원", "승민", "혜진", "동현", "은지", "성민",
    "재현", "미영", "정우", "수진", "영호", "경민", "상우", "나연", "보람", "다현", "준혁", "하린",
    "유나", "시윤", "태현", "가은", "주원", "민서", "재윤", "예은", "도현", "수현", "지환", "세영",
    "한결", "나은", "선우", "아린", "규민", "소연", "진호", "은서", "형준", "지아", "재민", "다인",
)  # fmt: skip
#: English nicknames common in Korean startups (chosen so that Korean particles read naturally).
NICKNAMES: tuple[str, ...] = (
    "Mason", "Ella", "Leo", "Chloe", "Ryan", "Jay", "Sophie", "Daniel", "Grace", "Kevin", "Lucy",
    "Ethan", "Mia", "Sam", "Tony", "Kate", "Jake", "Ben", "Eric", "Liam", "Luna", "Nina", "Paul",
    "Steve", "Amy", "Harry", "Ian", "Zoe", "Olivia", "Dana", "Henry", "Eden", "Ruby", "Owen", "Ivy",
    "Erin", "Alan", "Dylan", "Sean", "Brian", "Jenny", "Cindy", "Andy", "Emma", "Julia", "Mike",
)  # fmt: skip
LOCATIONS: tuple[str, ...] = (
    "상암 DMC IDC", "마포 IDC", "KT 목동 IDC", "가산 IDC", "AWS 서울 리전", "NCP 서울 리전", "GCP 서울 리전",
)  # fmt: skip
CHAT_MEDIA: tuple[tuple[ChannelMedium, int], ...] = (
    (ChannelMedium.SLACK, 5),
    (ChannelMedium.KAKAOTALK, 2),
    (ChannelMedium.TEAMS, 2),
    (ChannelMedium.INTERNAL_MESSENGER, 1),
)
OUT_OF_BAND_MEDIA: tuple[tuple[ChannelMedium, int], ...] = (
    (ChannelMedium.PHONE, 4),
    (ChannelMedium.SMS, 1),
    (ChannelMedium.PAGER, 2),
)
_STAFF_WEIGHTS: tuple[int, ...] = (2, 2, 4, 3, 1)
_NICKNAME_PROBABILITY = 0.35


def _weighted(rng: random.Random, options: tuple[tuple[ChannelMedium, int], ...]) -> ChannelMedium:
    return rng.choices([o for o, _ in options], weights=[w for _, w in options])[0]


def _theme_vocabulary(theme: _Theme) -> tuple[str, ...]:
    """Every term an org of ``theme`` may print besides people (a superset of the final org's)."""
    templates = [t for s in theme.services for t in (s.name, s.team, s.condition, *s.systems)]
    roles = (ROLE_CEO, ROLE_CTO, ROLE_CISO, ROLE_LEAD, ROLE_SECURITY, ROLE_CPO, EXEC_TEAM, SECURITY_TEAM)
    return (*templates, *roles, *LOCATIONS, *theme.companies, theme.label)


class _PeoplePool:
    """Draws people whose aliases resolve to exactly one person (see :func:`_ambiguous_aliases`).

    Given names and nicknames are unique per org, and a candidate is skipped while its alias
    would occur inside someone else's name or alias, or inside the theme's vocabulary (or one
    of theirs inside its name). Skipping scans the shuffled pool from the end without drawing
    random numbers, so orgs that never meet a clash are generated exactly as before.
    """

    def __init__(self, rng: random.Random, chat: ChannelMedium, vocabulary: Iterable[str]) -> None:
        self._rng = rng
        self._chat = chat
        self._vocabulary = tuple(vocabulary)
        self._given = rng.sample(GIVEN_NAMES, len(GIVEN_NAMES))
        self._nicks = rng.sample(NICKNAMES, len(NICKNAMES))
        self.people: list[GoldPerson] = []

    def _clashes(self, name: str, aliases: tuple[str, ...]) -> bool:
        others = [t for q in self.people for t in (q.name, *q.aliases)]
        theirs = [a for q in self.people for a in q.aliases]
        return bool(
            _aliases_inside([*others, *self._vocabulary], aliases)
            or _aliases_inside([name, *aliases], theirs)
        )

    @staticmethod
    def _take(pool: list[str], fits: Callable[[str], bool]) -> str:
        for index in range(len(pool) - 1, -1, -1):
            if fits(pool[index]):
                return pool.pop(index)
        raise ValueError("겹치지 않는 이름 후보가 남아 있지 않습니다")

    def draw(
        self,
        title: str,
        team: str,
        oob: ChannelMedium,
        *,
        status: PersonStatus = PersonStatus.ACTIVE,
        left_on: str | None = None,
    ) -> GoldPerson:
        rng = self._rng
        surname = rng.choices([s for s, _ in SURNAMES], weights=[w for _, w in SURNAMES])[0]
        given = self._take(self._given, lambda g: not self._clashes(surname + g, (g,)))
        aliases: tuple[str, ...] = (given,)
        if rng.random() < _NICKNAME_PROBABILITY:
            nick = self._take(self._nicks, lambda n: not self._clashes(surname + given, (given, n)))
            aliases = (given, nick)
        channels = [self._chat, oob]
        third = rng.choice((None, ChannelMedium.EMAIL, _weighted(rng, OUT_OF_BAND_MEDIA)))
        if third is not None and third not in channels:
            channels.append(third)
        person = GoldPerson(
            name=surname + given,
            title=title,
            team=team,
            status=status,
            channels=tuple(channels),
            aliases=aliases,
            left_on=left_on,
        )
        self.people.append(person)
        return person

    def staff_title(self) -> str:
        return self._rng.choices(STAFF_TITLES, weights=_STAFF_WEIGHTS)[0]


# --------------------------------------------------------------------------- generator


def generate_gold_orgs(n: int = 20, seed: int = DEFAULT_SEED) -> list[GoldOrg]:
    """``n`` fully READY gold organisations; org *i* uses theme *i mod 20* and its own sub-seed.

    Per-org sub-seeds make org *i* independent of ``n`` (the first 5 of 20 equal a run of 5),
    and string seeds hash via SHA-512 inside :mod:`random`, so results do not depend on
    ``PYTHONHASHSEED``.
    """
    if n < 1:
        raise ValueError("생성할 조직 수는 1 이상이어야 합니다")
    return [_OrgBuilder(index=i, seed=seed).build() for i in range(n)]


@dataclass(frozen=True)
class _Execs:
    ceo: GoldPerson
    cto: GoldPerson
    ciso: GoldPerson
    security: GoldPerson
    cpo: GoldPerson | None


class _OrgBuilder:
    def __init__(self, *, index: int, seed: int) -> None:
        self.index = index
        self.theme = THEMES[index % len(THEMES)]
        self.rng = random.Random(f"jettae-evalkit:{seed}:{index}")

    def build(self) -> GoldOrg:
        rng, theme = self.rng, self.theme
        company = rng.choice(theme.companies)
        chat = _weighted(rng, CHAT_MEDIA)
        locations = tuple(rng.sample(LOCATIONS, 2))
        pool = _PeoplePool(rng, chat, _theme_vocabulary(theme))
        execs = self._executives(pool)
        roles: dict[str, list[str]] = {
            ROLE_CEO: [execs.ceo.name],
            ROLE_CTO: [execs.cto.name],
            ROLE_CISO: [execs.ciso.name],
            ROLE_SECURITY: [execs.security.name],
            ROLE_LEAD: [],
        }
        if execs.cpo is not None:
            roles[ROLE_CPO] = [execs.cpo.name]
        services: list[GoldService] = []
        critical_seen = 0
        for template, tier, regulated in self._plan_services():
            if tier in CRITICAL_TIERS:
                location = locations[critical_seen % 2]
                critical_seen += 1
            else:
                location = rng.choice(locations)
            service, lead = self._service(
                pool, execs=execs, template=template, tier=tier, regulated=regulated, location=location
            )
            roles[ROLE_LEAD].append(lead.name)
            services.append(service)
        self._departed_distractor(pool, services)
        return GoldOrg(
            org_id=f"org{self.index + 1:02d}-{theme.key}",
            theme=theme.label,
            company=company,
            handles_personal_info=theme.personal_info,
            chat_medium=chat,
            locations=locations,
            people=tuple(pool.people),
            roles={k: tuple(sorted(v)) for k, v in sorted(roles.items())},
            services=tuple(services),
        )

    def _executives(self, pool: _PeoplePool) -> _Execs:
        rng = self.rng
        ceo = pool.draw("대표", EXEC_TEAM, _weighted(rng, OUT_OF_BAND_MEDIA))
        cto = pool.draw("CTO", EXEC_TEAM, _weighted(rng, OUT_OF_BAND_MEDIA))
        sec_oob = _weighted(rng, OUT_OF_BAND_MEDIA)
        ciso = pool.draw("CISO", SECURITY_TEAM, sec_oob)
        security = pool.draw(rng.choice(("대리", "과장")), SECURITY_TEAM, sec_oob)
        cpo = rng.choice((ciso, security)) if self.theme.personal_info else None
        return _Execs(ceo, cto, ciso, security, cpo)

    def _plan_services(self) -> list[tuple[_ServiceTemplate, int, bool]]:
        """Pick 3–6 services: 1–2 tier-1 (core), ≥2 critical, ≥2 regulated (see module notes)."""
        rng, templates = self.rng, self.theme.services
        n_services = rng.randint(3, 6)
        tier1 = rng.sample([t for t in templates if t.core], rng.randint(1, 2))
        rest = [t for t in templates if t not in tier1]
        others = rng.sample(rest, n_services - len(tier1))
        plan = [(t, 1, True) for t in tier1]
        for k, t in enumerate(others):
            tier = 2 if k == 0 else rng.choice((2, 2, 3))
            forced = k == 0 and len(tier1) == 1
            plan.append((t, tier, forced or (tier == 2 and rng.random() < 0.4)))
        return plan

    def _service(
        self,
        pool: _PeoplePool,
        *,
        execs: _Execs,
        template: _ServiceTemplate,
        tier: int,
        regulated: bool,
        location: str,
    ) -> tuple[GoldService, GoldPerson]:
        rng = self.rng
        team_oob = _weighted(rng, OUT_OF_BAND_MEDIA)
        lead = pool.draw(LEAD_TITLE, template.team, team_oob)
        members = [pool.draw(pool.staff_title(), template.team, team_oob) for _ in range(rng.randint(1, 3))]
        owner = members[0]
        executor = rng.choice(members)
        obligations: tuple[GoldObligation, ...] = ()
        if regulated:
            breach = GoldObligation(
                incident_type=IncidentType.BREACH,
                recipient=RECIPIENT_KISA,
                deadline_hours=LEGAL_DEADLINE_HOURS[IncidentType.BREACH],
                reporter=execs.ciso.name,
            )
            obligations = (breach,)
            if execs.cpo is not None:
                privacy = GoldObligation(
                    incident_type=IncidentType.PRIVACY_LEAK,
                    recipient=RECIPIENT_PIPC,
                    deadline_hours=LEGAL_DEADLINE_HOURS[IncidentType.PRIVACY_LEAK],
                    reporter=execs.cpo.name,
                )
                obligations = (breach, privacy)
        service = GoldService(
            name=template.name,
            tier=tier,
            team=template.team,
            systems=template.systems,
            location=location,
            owner=owner.name,
            first_responder=owner.name,
            notify_channel=owner.channels[0],
            escalation=(
                GoldEscalationStep(target=lead.name, after_minutes=rng.choice((5, 10, 15))),
                GoldEscalationStep(target=execs.cto.name, after_minutes=rng.choice((15, 20, 30))),
            ),
            trigger=GoldTrigger(condition=template.condition, delay_minutes=rng.choice((None, 1, 3, 5))),
            action=GoldAction(
                type=template.action,
                approvers=(lead.name, execs.cto.name),
                executor=executor.name,
                duration_minutes=rng.choice((5, 10, 15, 20, 30)),
            ),
            fallback=GoldFallback(primary_channel=owner.channels[0], fallback_channel=team_oob),
            obligations=obligations,
            regulated=regulated,
        )
        return service, lead

    def _departed_distractor(self, pool: _PeoplePool, services: list[GoldService]) -> None:
        """A former member nobody references: DEPARTED_CONTACT must not be "any departed person"."""
        rng = self.rng
        left_on = f"2026-{rng.randint(3, 8):02d}-{rng.randint(1, 28):02d}"
        team = rng.choice(services).team
        pool.draw(
            pool.staff_title(),
            team,
            _weighted(rng, OUT_OF_BAND_MEDIA),
            status=PersonStatus.DEPARTED,
            left_on=left_on,
        )
