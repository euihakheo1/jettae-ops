"""Fault-injection operators: break one READY service in one known way.

Why operators instead of hand-written broken runbooks: the expected finding is then a property
of the operator, not an opinion. Each operator maps to exactly one published rule id
(:data:`EXPECTED_RULE_IDS`), and :func:`apply` *proves* the mapping on every call with the
spec-level oracle (:func:`jettae.evalkit.gold.spec_findings`):

* the expected rule must newly fire for the target service — so the operator really broke it;
* no other service may newly break — so the ground truth of the rest of the document is intact;
* anything else that newly fires for the target is reported as ``collateral_rule_ids``
  (e.g. a departed owner also leaves the service without an active owner, JT-PATH-010). Those
  are legitimate consequences, not false positives.

``deleted_facts``/``added_facts`` are exact set differences of :func:`gold_facts`, so the
harness can score extraction on mutated documents with the same machinery as on clean ones.

Two operators are *held out* (sealed for generalisation): they must never be used to tune
extractors, rules or prompts, so the evaluation can show the system was not fitted to its test.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from jettae.domain import IncidentType, PersonStatus
from jettae.evalkit.gold import (
    LEGAL_DEADLINE_HOURS,
    RULE_APPROVER,
    RULE_BREACH_STEP,
    RULE_DEADLINE_LATE,
    RULE_DEPARTED,
    RULE_DETECTION,
    RULE_ESCALATION_CYCLE,
    RULE_FALLBACK,
    RULE_NO_ESCALATION,
    RULE_OWNER,
    RULE_ROLLBACK,
    Fact,
    GoldEscalationStep,
    GoldFallback,
    GoldOrg,
    GoldService,
    gold_facts,
    spec_findings,
)


class Operator(StrEnum):
    DELETE_OWNER = "delete_owner"
    DEPARTED_CONTACT = "departed_contact"
    DEADLINE_CONTRADICTION = "deadline_contradiction"
    ESCALATION_CYCLE = "escalation_cycle"
    DELETE_APPROVER = "delete_approver"
    DELETE_FALLBACK_CHANNEL = "delete_fallback_channel"
    DELETE_ROLLBACK = "delete_rollback"
    DELETE_ESCALATION = "delete_escalation"
    DELETE_DETECTION = "delete_detection"
    DELETE_REPORTING_STEP = "delete_reporting_step"


ACTIVE_OPERATORS: tuple[Operator, ...] = (
    Operator.DELETE_OWNER,
    Operator.DEPARTED_CONTACT,
    Operator.DEADLINE_CONTRADICTION,
    Operator.ESCALATION_CYCLE,
    Operator.DELETE_APPROVER,
    Operator.DELETE_FALLBACK_CHANNEL,
    Operator.DELETE_ROLLBACK,
    Operator.DELETE_ESCALATION,
)
#: Sealed for generalisation — NOT to be used for tuning.
HELD_OUT_OPERATORS: tuple[Operator, ...] = (Operator.DELETE_DETECTION, Operator.DELETE_REPORTING_STEP)
ALL_OPERATORS: tuple[Operator, ...] = ACTIVE_OPERATORS + HELD_OUT_OPERATORS

#: Published mapping operator -> rule ids that must fire for the target service.
EXPECTED_RULE_IDS: Mapping[Operator, frozenset[str]] = {
    Operator.DELETE_OWNER: frozenset({RULE_OWNER}),
    Operator.DEPARTED_CONTACT: frozenset({RULE_DEPARTED}),
    Operator.DEADLINE_CONTRADICTION: frozenset({RULE_DEADLINE_LATE}),
    Operator.ESCALATION_CYCLE: frozenset({RULE_ESCALATION_CYCLE}),
    Operator.DELETE_APPROVER: frozenset({RULE_APPROVER}),
    Operator.DELETE_FALLBACK_CHANNEL: frozenset({RULE_FALLBACK}),
    Operator.DELETE_ROLLBACK: frozenset({RULE_ROLLBACK}),
    Operator.DELETE_ESCALATION: frozenset({RULE_NO_ESCALATION}),
    Operator.DELETE_DETECTION: frozenset({RULE_DETECTION}),
    Operator.DELETE_REPORTING_STEP: frozenset({RULE_BREACH_STEP}),
}

#: Fixed (not wall-clock) departure date written to the roster by DEPARTED_CONTACT.
DEPARTURE_DATE = "2026-09-30"
#: The wrong breach deadline DEADLINE_CONTRADICTION writes (law: 24h).
CONTRADICTED_BREACH_HOURS = 48


class MutationError(ValueError):
    """The operator cannot be applied to this service in isolation."""


@dataclass(frozen=True)
class Mutation:
    org: GoldOrg
    operator: Operator
    target_service: str
    expected_rule_ids: frozenset[str]
    #: rules that also newly fire for the target as a direct consequence (not false positives)
    collateral_rule_ids: frozenset[str]
    deleted_facts: frozenset[Fact]
    added_facts: frozenset[Fact]


def apply(org: GoldOrg, op: Operator, service_name: str) -> Mutation:
    """Apply ``op`` to ``service_name`` and verify that it breaks exactly that service.

    Raises :class:`MutationError` when the service does not exist, the element to break is
    already missing, or the change would also break another service.
    """
    try:
        service = org.service(service_name)
    except KeyError:
        raise MutationError(f"{org.org_id}에 없는 서비스입니다: {service_name}") from None
    mutated = _MUTATORS[op](org, service)
    expected = EXPECTED_RULE_IDS[op]
    new = spec_findings(mutated) - spec_findings(org)
    stray = sorted(f"{f.service}:{f.rule_id}" for f in new if f.service not in (service_name, None))
    if stray:
        raise MutationError(f"{op.value}가 {service_name} 밖의 서비스까지 깨뜨립니다: {stray}")
    fired = frozenset(f.rule_id for f in new)
    if not expected <= fired:
        raise MutationError(
            f"{op.value}를 적용해도 {sorted(expected)} 규칙이 새로 걸리지 않습니다: {service_name}"
        )
    before, after = gold_facts(org), gold_facts(mutated)
    return Mutation(
        org=mutated,
        operator=op,
        target_service=service_name,
        expected_rule_ids=expected,
        collateral_rule_ids=fired - expected,
        deleted_facts=before - after,
        added_facts=after - before,
    )


# --------------------------------------------------------------------------- operators


def _with(org: GoldOrg, s: GoldService, **update: object) -> GoldOrg:
    return org.replace_service(s.model_copy(update=update))


def _delete_owner(org: GoldOrg, s: GoldService) -> GoldOrg:
    if s.owner is None:
        raise MutationError(f"{s.name}: 담당자가 이미 없습니다")
    return _with(org, s, owner=None)


def _departed_contact(org: GoldOrg, s: GoldService) -> GoldOrg:
    """Owner leaves the company: only the roster changes, the runbook text stays as it was."""
    if s.owner is None or not org.person(s.owner).is_active:
        raise MutationError(f"{s.name}: 퇴사 처리할 활동 중인 담당자가 없습니다")
    owner = org.person(s.owner)
    return org.replace_people(
        owner.model_copy(update={"status": PersonStatus.DEPARTED, "left_on": DEPARTURE_DATE})
    )


def _deadline_contradiction(org: GoldOrg, s: GoldService) -> GoldOrg:
    ob = s.obligation(IncidentType.BREACH)
    if ob is None or ob.deadline_hours != LEGAL_DEADLINE_HOURS[IncidentType.BREACH]:
        raise MutationError(f"{s.name}: 법정 시한대로 적힌 침해사고 신고 단계가 없습니다")
    late = tuple(
        o.model_copy(update={"deadline_hours": CONTRADICTED_BREACH_HOURS}) if o is ob else o
        for o in s.obligations
    )
    return _with(org, s, obligations=late)


def _escalation_cycle(org: GoldOrg, s: GoldService) -> GoldOrg:
    """The first hop escalates back to the first responder instead of onwards (A → B → A)."""
    if not s.escalation or s.escalation[0].target == s.first_responder:
        raise MutationError(f"{s.name}: 1차 대응자에게서 다른 사람으로 가는 상향 경로가 없습니다")
    first = s.escalation[0]
    back_after = (s.escalation[1] if len(s.escalation) > 1 else first).after_minutes
    back = GoldEscalationStep(target=s.first_responder, after_minutes=back_after)
    return _with(org, s, escalation=(first, back))


def _delete_approver(org: GoldOrg, s: GoldService) -> GoldOrg:
    """Approval stays required ("승인 후 …") but nobody is named to give it."""
    if s.action is None or not s.action.approvers:
        raise MutationError(f"{s.name}: 지울 승인자가 없습니다")
    return _with(org, s, action=s.action.model_copy(update={"approvers": ()}))


def _delete_fallback_channel(org: GoldOrg, s: GoldService) -> GoldOrg:
    """Owner and first escalation hop keep only the shared out-of-band medium.

    JT-PATH-030 pools the owner's and the first hop's channels, so both must drop to one
    medium. The kept medium is the old fallback (never a FALLBACK_TO source in our specs), so
    no org-wide fallback edge can silently re-add a second channel.
    """
    keep = s.fallback.fallback_channel
    first_hop = s.escalation[0].target if s.escalation else None
    names = [n for n in dict.fromkeys((s.owner, s.first_responder, first_hop)) if n is not None]
    people = [org.person(n) for n in names]
    if keep is None or not all(keep in p.channels for p in people):
        raise MutationError(f"{s.name}: 대응자 모두가 쓰는 대체 채널이 없습니다")
    org = org.replace_people(*(p.model_copy(update={"channels": (keep,)}) for p in people))
    return _with(org, org.service(s.name), fallback=GoldFallback(primary_channel=keep), notify_channel=keep)


def _delete_rollback(org: GoldOrg, s: GoldService) -> GoldOrg:
    if s.action is None:
        raise MutationError(f"{s.name}: 완화 조치가 이미 없습니다")
    return _with(org, s, action=None)


def _delete_escalation(org: GoldOrg, s: GoldService) -> GoldOrg:
    if not s.escalation:
        raise MutationError(f"{s.name}: 상향 경로가 이미 없습니다")
    return _with(org, s, escalation=())


def _delete_detection(org: GoldOrg, s: GoldService) -> GoldOrg:
    if s.trigger is None:
        raise MutationError(f"{s.name}: 감지 경로가 이미 없습니다")
    return _with(org, s, trigger=None)


def _delete_reporting_step(org: GoldOrg, s: GoldService) -> GoldOrg:
    """Remove the 침해사고 24시간 KISA step (the privacy step, if any, stays)."""
    ob = s.obligation(IncidentType.BREACH)
    if ob is None:
        raise MutationError(f"{s.name}: 침해사고 신고 단계가 이미 없습니다")
    return _with(org, s, obligations=tuple(o for o in s.obligations if o is not ob))


_MUTATORS: Mapping[Operator, Callable[[GoldOrg, GoldService], GoldOrg]] = {
    Operator.DELETE_OWNER: _delete_owner,
    Operator.DEPARTED_CONTACT: _departed_contact,
    Operator.DEADLINE_CONTRADICTION: _deadline_contradiction,
    Operator.ESCALATION_CYCLE: _escalation_cycle,
    Operator.DELETE_APPROVER: _delete_approver,
    Operator.DELETE_FALLBACK_CHANNEL: _delete_fallback_channel,
    Operator.DELETE_ROLLBACK: _delete_rollback,
    Operator.DELETE_ESCALATION: _delete_escalation,
    Operator.DELETE_DETECTION: _delete_detection,
    Operator.DELETE_REPORTING_STEP: _delete_reporting_step,
}
