"""Tests for ``jettae.improve.drafter``: template patches, candidate policies, LLM rewording."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import BaseModel

from jettae.check.context import CheckContext
from jettae.check.registry import finding, run_checks
from jettae.domain import (
    ActionType,
    AddEdge,
    AddNode,
    AttrKey,
    EdgeKind,
    Element,
    Extractor,
    Finding,
    FixHint,
    FixKind,
    IncidentType,
    NodeKind,
    Patch,
    PatchAuthor,
    PersonStatus,
    RemoveEdge,
    RuleInfo,
    SetAttr,
    Severity,
    SourceKind,
    edge_id,
    node_id,
)
from jettae.improve import (
    ESCALATION_PROPOSAL_MINUTES,
    MEMO_KINDS,
    NO_CANDIDATE,
    LlmDrafter,
    TemplateDrafter,
    apply_ops,
    choice_schema,
    group_findings,
    validate_patch,
)
from jettae.llm.base import LLMDisabledError, LLMOutputError, UntrustedDocument
from jettae.llm.fake import FakeLLM
from jettae.llm.schema import to_strict_json_schema
from test_improve_support import Org, PaymentOrg, attr_value, demo_ctx, demo_inputs, payment_org, run

RULE = RuleInfo(
    rule_id="JT-TEST-001", title="시험용 끊김", severity=Severity.HIGH, element=None, rationale="시험"
)


def draft(ctx: CheckContext, findings: Iterable[Finding] | None = None, **kw: Any) -> list[Patch]:
    found = list(findings) if findings is not None else run_checks(ctx)
    return TemplateDrafter().draft(
        ctx, found, workspace_id=kw.get("workspace_id", "ws"), base_version=kw.get("base_version", 1)
    )


def patch_for(patches: Iterable[Patch], rule_prefix: str) -> Patch:
    return next(p for p in patches if any(f.startswith(rule_prefix) for f in p.finding_ids))


def fake_finding(
    kind: FixKind, *, service_id: str | None = None, element: Element | None = None, **params: Any
) -> Finding:
    return finding(
        RULE,
        keys=(kind.value, service_id or "-", *map(str, params.values())),
        title="시험용 끊김",
        message="시험용 끊김 설명.",
        service_id=service_id,
        element=element,
        fix=FixHint(kind=kind, params=params),
    )


def added_edges(patch: Patch) -> dict[str, Any]:
    return {op.edge.id: op.edge for op in patch.ops if isinstance(op, AddEdge)}


def removed_edges(patch: Patch) -> set[str]:
    return {op.edge_id for op in patch.ops if isinstance(op, RemoveEdge)}


@pytest.fixture
def po() -> PaymentOrg:
    return payment_org()


# --------------------------------------------------------------------------- owner replacement


def test_departed_owner_is_replaced_by_primary_oncall_teammate(po: PaymentOrg) -> None:
    patches = draft(po.org.ctx())
    owner = patch_for(patches, "JT-PPL-001")
    assert owner.runbook_text == "결제 장애 알람이 오면 1차 담당 이서연 사원에게 슬랙으로 연락한다."
    assert owner.author is PatchAuthor.TEMPLATE
    # REPLACE_DEPARTED, ASSIGN_OWNER and the silent alarm (ADD_DETECTION) are one decision
    assert {f.split(":")[0] for f in owner.finding_ids} == {"JT-PPL-001", "JT-PATH-010", "JT-PATH-002"}
    assert removed_edges(owner) == {
        edge_id(EdgeKind.OWNED_BY, po.payment, po.departed),
        edge_id(EdgeKind.NOTIFIES, po.alarm, po.departed),
    }
    added = added_edges(owner)
    notify = added[edge_id(EdgeKind.NOTIFIES, po.alarm, po.oncall)]
    assert notify.attr(AttrKey.CHANNEL) == node_id(NodeKind.CHANNEL, "slack")
    assert edge_id(EdgeKind.OWNED_BY, po.payment, po.oncall) in added
    assert "퇴사자 김민준 대리" in owner.rationale
    assert "1차 당직" in owner.rationale
    assert "다른 후보: 최민수 사원" in owner.rationale


def test_patch_created_elements_carry_patch_provenance_with_the_patch_id(po: PaymentOrg) -> None:
    owner = patch_for(draft(po.org.ctx()), "JT-PPL-001")
    for op in owner.ops:
        if isinstance(op, AddEdge):
            (prov,) = op.edge.prov
            assert (prov.source, prov.extractor, prov.ref) == (SourceKind.PATCH, Extractor.HUMAN, owner.id)
            assert all(a.source is SourceKind.PATCH for a in op.edge.attrs.values())
        if isinstance(op, RemoveEdge):
            assert op.snapshot is not None  # stored as applied: invertible without the base graph


def test_replacement_without_teammates_is_a_memo(po: PaymentOrg) -> None:
    org = Org()
    gone = org.person("김민준", title="대리", team="결제팀", status=PersonStatus.DEPARTED)
    org.person("한서윤", title="CTO", team="경영", roles=("CTO",))
    svc = org.service("결제")
    org.trigger(svc, "알람", gone)
    org.owner(svc, gone)
    owner = patch_for(draft(org.ctx()), "JT-PPL-001")
    assert owner.ops == ()
    assert NO_CANDIDATE in owner.runbook_text
    assert "검토 메모" in owner.rationale


def test_no_roster_means_no_people_in_any_patch(po: PaymentOrg) -> None:
    ctx = po.org.ctx(roster=False)
    for patch in draft(ctx):
        people = {
            n
            for op in patch.ops
            if isinstance(op, AddEdge)
            for n in (op.edge.src, op.edge.dst)
            if ctx.graph.nodes[n].kind is NodeKind.PERSON
        }
        assert not people, patch


def test_departed_approver_and_escalation_target_go_to_a_lead() -> None:
    org = Org()
    owner = org.person("이서연", title="사원", team="결제팀", roles=("1차 당직",))
    gone = org.person("김팀장", title="팀장", team="결제팀", status=PersonStatus.DEPARTED)
    lead = org.person("박재현", title="팀장", team="개발팀", roles=("팀장",))
    org.person("한서윤", title="CTO", roles=("CTO",))
    org.escalate(node_id(NodeKind.ROLE, "팀장"), node_id(NodeKind.PERSON, "한서윤"), 20)
    svc = org.service("결제")
    org.trigger(svc, "알람", owner)
    org.owner(svc, owner)
    org.escalate(owner, gone, 15)
    org.action(svc, "결제 롤백", approvers=[gone])
    patch = patch_for(draft(org.ctx()), "JT-PPL-001")
    added = added_edges(patch)
    assert edge_id(EdgeKind.ESCALATES_TO, owner, lead) in added
    assert (
        added[edge_id(EdgeKind.ESCALATES_TO, owner, lead)].attr(AttrKey.AFTER_MINUTES) == 15
    )  # stated, kept
    assert edge_id(EdgeKind.APPROVED_BY, node_id(NodeKind.ACTION, "결제 롤백"), lead) in added
    assert "15분 안에 응답하지 않으면 박재현 팀장에게 상향한다" in patch.runbook_text


# --------------------------------------------------------------------------- escalation & cycle


@pytest.mark.parametrize(("tier", "minutes"), sorted(ESCALATION_PROPOSAL_MINUTES.items()))
def test_escalation_goes_to_the_lead_with_a_tier_proposal(tier: int, minutes: int) -> None:
    org = Org()
    owner = org.person("이서연", title="사원", team="결제팀", roles=("1차 당직",))
    lead = org.person("박재현", title="팀장", team="결제팀", roles=("팀장",), channels=("slack", "phone"))
    svc = org.service("결제", tier=tier)
    org.trigger(svc, "알람", owner)
    org.owner(svc, owner)
    org.policy = org.policy.model_copy(update={"critical_tiers": (1, 2, 3)})
    patch = patch_for(draft(org.ctx()), "JT-ESC-001")
    (edge,) = added_edges(patch).values()
    assert (edge.src, edge.dst, edge.attr(AttrKey.AFTER_MINUTES)) == (owner, lead, minutes)
    assert (
        patch.runbook_text
        == f"결제 장애 때 이서연 사원이 {minutes}분 안에 응답하지 않으면 박재현 팀장에게 전화로 상향한다."
    )
    assert f"대기 시간 {minutes}분은" in patch.rationale
    assert "제안값" in patch.rationale


def test_escalation_from_a_departed_owner_waits_for_a_new_owner(po: PaymentOrg) -> None:
    patch = patch_for(draft(po.org.ctx()), "JT-ESC-001")
    assert patch.ops == ()
    assert "담당자를 먼저 지정" in patch.runbook_text


def test_escalation_cycle_is_broken_toward_the_cto() -> None:
    org = Org()
    a = org.person("정민혁", title="과장", team="주문팀")
    b = org.person("최영훈", title="차장", team="주문팀", channels=("slack",))
    cto = org.person("한서윤", title="CTO", team="경영", roles=("CTO",), channels=("phone", "email"))
    org.person("조영석", title="대표", team="경영", roles=("대표",), channels=("phone",))
    svc = org.service("주문")
    org.trigger(svc, "주문 알람", a)
    org.owner(svc, a)
    org.escalate(a, b, 15)
    closing = org.escalate(b, a, 20)
    org.action(svc, "주문 롤백", approvers=[cto], executors=[a])
    patch = patch_for(draft(org.ctx()), "JT-ESC-002")
    assert removed_edges(patch) == {closing}
    (edge,) = added_edges(patch).values()
    assert (edge.src, edge.dst, edge.attr(AttrKey.AFTER_MINUTES)) == (b, cto, 20)
    assert patch.runbook_text == (
        "주문 장애 때 최영훈 차장이 20분 안에 응답하지 않으면 정민혁 과장에게 다시 연락하는 대신 "
        "한서윤 CTO에게 전화로 상향한다."
    )
    assert "제안값" not in patch.rationale  # the 20 minutes are the runbook's own


# --------------------------------------------------------------------------- approver, executor, deputy


def test_approver_is_a_lead_and_the_step_is_marked_gated(po: PaymentOrg) -> None:
    patch = patch_for(draft(po.org.ctx()), "JT-PATH-020")
    assert patch.runbook_text == "결제 롤백은 박재현 팀장 승인 후 실행한다."
    assert edge_id(EdgeKind.APPROVED_BY, po.rollback, po.lead) in added_edges(patch)
    (gate,) = [op for op in patch.ops if isinstance(op, SetAttr)]
    assert (gate.target_id, gate.key, gate.value and gate.value.value) == (
        po.rollback,
        "requires_approval",
        True,
    )
    assert "다른 팀의 리드이며 자신의 상향 경로가 한서윤 CTO까지 이어짐" in patch.rationale


def test_named_owner_is_never_proposed_as_approver_and_executes_instead() -> None:
    org = Org()
    lead = org.person("박재현", title="팀장", team="개발팀", roles=("팀장",))
    org.person("강현우", title="사원", team="개발팀")
    cto = org.person("한서윤", title="CTO", roles=("CTO",))
    svc = org.service("상품 카탈로그", tier=2)
    org.trigger(svc, "카탈로그 알람", lead)
    org.owner(svc, lead)
    org.owner(svc, org.node(NodeKind.TEAM, "개발팀"))
    org.escalate(lead, cto, 20)
    org.action(svc, "기능 플래그 OFF", kind=ActionType.FEATURE_FLAG, requires_approval=True)
    patches = draft(org.ctx())
    approver = patch_for(patches, "JT-PATH-020")
    assert approver.runbook_text == "기능 플래그 OFF는 한서윤 CTO 승인 후 실행한다."
    executor = patch_for(patches, "JT-PATH-041")
    assert executor.runbook_text == "기능 플래그 OFF는 박재현 팀장이 실행한다."
    assert "서비스 담당자" in executor.rationale


def test_lone_approver_gets_a_final_responsibility_deputy() -> None:
    org = Org()
    owner = org.person("이수진", title="대리", team="개발팀")
    lead = org.person("박재현", title="팀장", team="개발팀", roles=("팀장",))
    cto = org.person("한서윤", title="CTO", roles=("CTO",))
    svc = org.service("장바구니")
    org.trigger(svc, "장바구니 알람", owner)
    org.owner(svc, owner)
    org.escalate(owner, lead, 10)
    rollback = org.action(svc, "장바구니 롤백", approvers=[lead], executors=[owner])
    finding_ = fake_finding(
        FixKind.ADD_DEPUTY, service_id=svc, element=Element.APPROVER, person_id=lead, service=svc
    )
    (patch,) = draft(org.ctx(), [finding_])
    assert patch.runbook_text == "장바구니 롤백은 박재현 팀장이 부재일 때 한서윤 CTO가 대신 승인한다."
    assert set(added_edges(patch)) == {edge_id(EdgeKind.APPROVED_BY, rollback, cto)}


def test_lone_responder_gets_an_escalation_and_leave_gets_cover() -> None:
    org = Org()
    owner = org.person("이수진", title="대리", team="개발팀")
    away = org.person("배은지", title="대리", team="개발팀", status=PersonStatus.ON_LEAVE)
    mate = org.person("강현우", title="사원", team="개발팀", roles=("1차 당직",))
    lead = org.person("박재현", title="팀장", team="개발팀", roles=("팀장",))
    svc = org.service("추천", tier=2)
    alarm = org.trigger(svc, "추천 알람", away)
    org.owner(svc, owner)
    spof = fake_finding(FixKind.ADD_DEPUTY, service_id=svc, element=Element.OWNER, person_id=owner)
    leave = fake_finding(
        FixKind.ADD_DEPUTY, service_id=svc, person_id=away, edge_ids=[edge_id(EdgeKind.NOTIFIES, alarm, away)]
    )
    (escalation,) = draft(org.ctx(), [spof])
    assert (
        added_edges(escalation)[edge_id(EdgeKind.ESCALATES_TO, owner, lead)].attr(AttrKey.AFTER_MINUTES) == 10
    )
    assert "10분 안에 응답하지 않으면 박재현 팀장에게 전화로 상향한다" in escalation.runbook_text
    (cover,) = draft(org.ctx(), [leave])
    assert edge_id(EdgeKind.NOTIFIES, alarm, mate) in added_edges(cover)
    # JT-PPL-003 reports every duty still on someone on leave: the link moves, it is not doubled
    assert removed_edges(cover) == {edge_id(EdgeKind.NOTIFIES, alarm, away)}
    assert cover.runbook_text == "배은지 대리가 장기 부재인 동안 추천의 호출은 강현우 사원이 대신 맡는다."
    assert "복귀하면 원래 담당으로 되돌리는 수정을 사람이 해야 합니다" in cover.rationale


# --------------------------------------------------------------------------- channels


def test_fallback_channel_uses_only_roster_registered_media() -> None:
    org = Org()
    owner = org.person(
        "최영훈", title="차장", team="주문팀", channels=("slack", "phone"), link_channels=False
    )
    org.edge(EdgeKind.REACHABLE_VIA, owner, org.channel("slack"))
    other = org.person("홍길동", title="사원", team="주문팀", channels=("slack",))
    svc = org.service("주문")
    finding_ = fake_finding(FixKind.ADD_FALLBACK_CHANNEL, service_id=svc, person_ids=[owner, other])
    (patch,) = draft(org.ctx(), [finding_])
    new_channel = node_id(NodeKind.CHANNEL, "phone")
    assert [op.node.id for op in patch.ops if isinstance(op, AddNode)] == [new_channel]
    assert set(added_edges(patch)) == {edge_id(EdgeKind.REACHABLE_VIA, owner, new_channel)}
    assert patch.runbook_text == "최영훈 차장에게 슬랙이 안 되면 전화로 연락한다."
    assert "홍길동 사원에게는 명부에 등록된 다른 연락 수단이 없어" in patch.rationale


def test_fallback_channel_memo_when_roster_has_nothing_more() -> None:
    org = Org()
    owner = org.person("최영훈", title="차장", team="주문팀", channels=("slack",))
    svc = org.service("주문")
    (patch,) = draft(
        org.ctx(), [fake_finding(FixKind.ADD_FALLBACK_CHANNEL, service_id=svc, person_ids=[owner])]
    )
    assert patch.ops == ()
    assert "명부에 연락 수단(예: 전화)을 추가 등록" in patch.runbook_text


def test_empty_fallback_channel_gets_its_registered_people() -> None:
    org = Org()
    owner = org.person(
        "이수진", title="대리", team="개발팀", channels=("internal_messenger", "phone"), link_channels=False
    )
    messenger, phone = org.channel("internal_messenger"), org.channel("phone")
    org.edge(EdgeKind.REACHABLE_VIA, owner, messenger)
    org.edge(EdgeKind.FALLBACK_TO, messenger, phone)
    finding_ = fake_finding(
        FixKind.ADD_FALLBACK_CHANNEL, channel_id=phone, person_ids=[owner], channel_ids=[messenger]
    )
    (patch,) = draft(org.ctx(), [finding_])
    assert set(added_edges(patch)) == {edge_id(EdgeKind.REACHABLE_VIA, owner, phone)}
    assert patch.runbook_text == "대체 채널 전화로도 이수진 대리에게 연락한다."


# --------------------------------------------------------------------------- deadlines & reporting


def test_late_legal_deadline_is_set_to_the_legal_value_with_its_basis() -> None:
    org = Org()
    ciso = org.person("오준석", title="이사", roles=("CISO",))
    oid = org.obligation(
        "KISA 침해사고 신고", incident=IncidentType.BREACH, recipient="KISA", deadline=2880, reporters=[ciso]
    )
    ctx = org.ctx()
    (patch,) = draft(ctx, [f for f in run_checks(ctx) if f.rule_id == "JT-DDL-002"])
    (op,) = patch.ops
    assert isinstance(op, SetAttr)
    assert (op.target_id, op.key, op.value and op.value.value) == (oid, "deadline_minutes", 1440)
    assert patch.runbook_text == "침해사고를 알게 된 때부터 24시간 이내에 KISA에 신고한다."
    assert "48시간에서 24시간으로" in patch.rationale
    assert "정보통신망법 제48조의3" in patch.rationale


def test_unstated_and_slow_escalation_waits_get_proposals() -> None:
    org = Org()
    a = org.person("가나다", title="사원", team="팀")
    b = org.person("라마바", title="팀장", team="팀", roles=("팀장",))
    svc = org.service("결제")
    untimed = org.escalate(a, b)
    slow = org.escalate(b, a, 50)
    unstated = fake_finding(FixKind.FIX_DEADLINE, service_id=svc, edge_id=untimed, attr="after_minutes")
    too_slow = fake_finding(
        FixKind.FIX_DEADLINE, service_id=svc, edge_ids=[untimed, slow], required_minutes=60
    )
    (patch,) = draft(org.ctx(), [unstated])
    assert attr_value(apply_ops(org.graph(), patch.ops)[0], untimed, AttrKey.AFTER_MINUTES) == 5
    assert "제안값" in patch.rationale
    (shorten,) = draft(org.ctx(), [too_slow])
    assert attr_value(apply_ops(org.graph(), shorten.ops)[0], slow, AttrKey.AFTER_MINUTES) == 5
    hopeless = fake_finding(FixKind.FIX_DEADLINE, service_id=svc, edge_ids=[slow], required_minutes=1)
    (memo,) = draft(org.ctx(), [hopeless])
    assert memo.ops == ()
    assert "경로를 정해야" in memo.rationale


def test_org_wide_reporting_step_has_no_must_report_and_service_step_has_one(po: PaymentOrg) -> None:
    ctx = po.org.ctx()
    patch = patch_for(draft(ctx), "JT-KR-003")
    nodes = [op.node for op in patch.ops if isinstance(op, AddNode)]
    assert {n.attr(AttrKey.INCIDENT_TYPE) for n in nodes} == {"breach", "privacy_leak"}
    assert not any(e.kind is EdgeKind.MUST_REPORT for e in added_edges(patch).values())
    reporters = {e.src: e.dst for e in added_edges(patch).values() if e.kind is EdgeKind.REPORTED_BY}
    privacy = next(n.id for n in nodes if n.attr(AttrKey.INCIDENT_TYPE) == "privacy_leak")
    assert reporters[privacy] == po.privacy
    assert "개인정보 유출을 알게 된 때부터 72시간 이내에" in patch.runbook_text
    scoped = fake_finding(
        FixKind.ADD_REPORTING_STEP,
        service_id=po.payment,
        incident_type="breach",
        recipient="KISA",
        deadline_minutes=1440,
        service_ids=[po.payment],
    )
    (one,) = draft(ctx, [scoped])
    assert any(e.kind is EdgeKind.MUST_REPORT and e.src == po.payment for e in added_edges(one).values())
    assert one.runbook_text.startswith(
        "결제 서비스에서 침해사고를 알게 된 때부터 24시간 이내에 KISA에 신고한다"
    )


def test_unclear_reporting_step_is_left_to_a_human(po: PaymentOrg) -> None:
    unclear = fake_finding(
        FixKind.ADD_REPORTING_STEP,
        incident_type="breach",
        recipient="KISA",
        deadline_minutes=1440,
        service_names=["없는 서비스"],
    )
    (patch,) = draft(po.org.ctx(), [unclear])
    assert patch.ops == ()
    assert "사람이 먼저 확인할 신고 단계" in patch.rationale


# --------------------------------------------------------------------------- memos & grouping


@pytest.mark.parametrize("kind", sorted(MEMO_KINDS))
def test_memo_kinds_never_carry_ops(po: PaymentOrg, kind: FixKind) -> None:
    (patch,) = draft(po.org.ctx(), [fake_finding(kind, service_id=po.payment, trigger_ids=[po.alarm])])
    assert patch.ops == ()
    assert patch.runbook_text
    assert "검토 메모" in patch.rationale


def test_grouping_merges_one_decision_and_orders_org_wide_last(po: PaymentOrg) -> None:
    groups = group_findings(run_checks(po.org.ctx()))
    keys = [(g.service_id, g.kind) for g in groups]
    assert keys == sorted(keys, key=lambda k: (k[0] is None, k[0] or "", list(FixKind).index(k[1])))
    owner = next(g for g in groups if g.kind is FixKind.ASSIGN_OWNER)
    assert {f.fix.kind for f in owner.findings if f.fix} == {
        FixKind.ASSIGN_OWNER,
        FixKind.REPLACE_DEPARTED,
        FixKind.ADD_DETECTION,
    }
    assert not any(g.kind is FixKind.ADD_DETECTION for g in groups)


def test_drafting_is_deterministic_and_ids_depend_on_the_base(po: PaymentOrg) -> None:
    ctx = po.org.ctx()
    first, second = draft(ctx), draft(ctx)
    assert first == second
    assert len({p.id for p in first}) == len(first)
    moved = draft(ctx, base_version=2)
    assert {p.id for p in moved}.isdisjoint({p.id for p in first})


# --------------------------------------------------------------------------- no invented people


def _roster_active(ctx: CheckContext, pid: str) -> bool:
    entry = ctx.roster_entry(ctx.graph.nodes[pid])
    return entry is not None and entry.status is PersonStatus.ACTIVE


PERSON_FACTS = st.tuples(
    st.sampled_from(list(PersonStatus)),
    st.booleans(),
    st.sampled_from(["결제팀", "개발팀"]),
    st.sampled_from([(), ("1차 당직",), ("팀장",), ("CTO",), ("CISO",)]),
)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    facts=st.lists(PERSON_FACTS, min_size=5, max_size=5),
    escalations=st.lists(
        st.tuples(st.integers(0, 4), st.integers(0, 4), st.sampled_from([None, 10])), max_size=3
    ),
    approvers=st.lists(st.integers(0, 4), max_size=2),
)
def test_property_drafts_link_only_active_roster_people(
    facts: list[tuple[PersonStatus, bool, str, tuple[str, ...]]],
    escalations: list[tuple[int, int, int | None]],
    approvers: list[int],
) -> None:
    """Whatever the roster says, no drafted op links someone who is not ACTIVE in it (P2)."""
    org = Org()
    people = [
        org.person(f"사람{i}", title="사원", team=team, roles=roles, status=status, in_roster=listed)
        for i, (status, listed, team, roles) in enumerate(facts)
    ]
    svc = org.service("결제")
    org.trigger(svc, "알람", people[0])
    org.owner(svc, people[0])
    for src, dst, minutes in escalations:
        if src != dst:
            org.escalate(people[src], people[dst], minutes)
    org.action(svc, "롤백", approvers=[people[i] for i in approvers], executors=[people[3]])
    ctx = org.ctx()
    findings = run_checks(ctx)
    for patch in draft(ctx, findings):
        after, _ = apply_ops(ctx.graph, patch.ops)
        for op in patch.ops:
            if isinstance(op, AddEdge):
                for n in (op.edge.src, op.edge.dst):
                    if after.nodes[n].kind is NodeKind.PERSON:
                        assert _roster_active(ctx, n), (patch.runbook_text, n)
        if patch.ops:
            assert validate_patch(ctx, patch, baseline_findings=findings).unknown_people == ()


def test_demo_drafts_name_only_active_roster_people() -> None:
    bundle, graph = demo_inputs()
    ctx = demo_ctx(bundle, graph)
    patches = draft(ctx)
    assert patches
    for patch in patches:
        for op in patch.ops:
            if (
                isinstance(op, AddEdge)
                and graph.nodes.get(op.edge.dst)
                and graph.nodes[op.edge.dst].kind is NodeKind.PERSON
            ):
                assert _roster_active(ctx, op.edge.dst)
    texts = {p.runbook_text for p in patches}
    assert "결제 장애 알람이 오면 1차 담당 김서연 사원에게 슬랙으로 연락한다." in texts
    assert "결제 서비스 롤백은 박재현 팀장 승인 후 실행한다." in texts
    assert (
        "주문 장애 때 최영훈 차장이 20분 안에 응답하지 않으면 정민혁 과장에게 다시 연락하는 대신 "
        "한서윤 CTO에게 전화로 상향한다."
    ) in texts
    assert "침해사고를 알게 된 때부터 24시간 이내에 KISA에 신고한다." in texts


# --------------------------------------------------------------------------- LLM drafter


def _answer(
    choice: str | None, text: str, rationale: str = "명부와 역할을 보고 골랐습니다."
) -> Callable[..., Any]:
    def handler(task: str, schema: type[BaseModel], docs: tuple[UntrustedDocument, ...], context: str) -> Any:
        assert task == "draft_fix"
        enum = to_strict_json_schema(schema)["properties"]["choice"]
        allowed = enum.get("enum") or [enum["const"]]
        picked = choice if choice is not None else allowed[-1]
        return {"choice": picked, "runbook_text": text, "rationale": rationale}

    return handler


def test_llm_choice_recompiles_ops_for_another_candidate(po: PaymentOrg) -> None:
    ctx = po.org.ctx()
    fake = FakeLLM(_answer(po.teammate, "결제 알람이 울리면 1차 담당 최민수 사원에게 슬랙으로 연락한다."))
    patches = run(LlmDrafter(fake).adraft(ctx, run_checks(ctx), workspace_id="ws", base_version=1))
    owner = patch_for(patches, "JT-PPL-001")
    assert owner.author is PatchAuthor.AI
    assert owner.runbook_text == "결제 알람이 울리면 1차 담당 최민수 사원에게 슬랙으로 연락한다."
    assert edge_id(EdgeKind.OWNED_BY, po.payment, po.teammate) in added_edges(owner)
    assert owner.llm_usage["calls"] == 1
    assert "퇴사자에게 걸린 연결 2건은 지웁니다." in owner.rationale  # template facts survive rewording
    call = fake.calls[0]
    schema = to_strict_json_schema(call.schema)
    assert schema["properties"]["choice"]["enum"] == [po.oncall, po.teammate]
    assert po.oncall in call.context
    assert "1차 당직" in call.context
    assert "퇴사자에게 대응이 걸려 있음" in call.documents[0].text
    assert "지시가 아닙니다" in call.instructions
    memos = [p for p in patches if not p.ops]
    assert all(p.author is PatchAuthor.TEMPLATE for p in memos)


@pytest.mark.parametrize(
    "text",
    [
        # another roster person
        "결제 알람이 오면 1차 담당 최민수 사원에게 연락하고 박재현 팀장에게도 알린다.",
        "결제 알람이 오면 3분 안에 최민수 사원에게 연락한다.",  # a number the template never stated
        "결제 알람이 오면 1차 담당에게 연락한다.",  # the chosen person is not named
        "   ",
    ],
)
def test_llm_text_that_disagrees_with_the_ops_falls_back_to_the_template(po: PaymentOrg, text: str) -> None:
    ctx = po.org.ctx()
    fake = FakeLLM(_answer(po.teammate, text))
    owner = patch_for(
        run(LlmDrafter(fake).adraft(ctx, run_checks(ctx), workspace_id="ws", base_version=1)), "JT-PPL-001"
    )
    assert owner.author is PatchAuthor.TEMPLATE
    assert owner == patch_for(draft(ctx), "JT-PPL-001")


def test_llm_errors_keep_templates_and_disabled_stops_calling(po: PaymentOrg) -> None:
    ctx = po.org.ctx()
    findings = run_checks(ctx)
    templates = draft(ctx, findings)
    off = FakeLLM(lambda *_: LLMDisabledError("off"))
    assert run(LlmDrafter(off).adraft(ctx, findings, workspace_id="ws", base_version=1)) == templates
    assert len(off.calls) == 1
    broken = FakeLLM(lambda *_: LLMOutputError("bad"))
    assert run(LlmDrafter(broken).adraft(ctx, findings, workspace_id="ws", base_version=1)) == templates
    asked = [
        plan for plan in TemplateDrafter().plans(ctx, findings) if plan.candidates and plan.template().ops
    ]
    assert len(broken.calls) == len(asked) > 1  # one failure does not stop the others


def test_choice_schema_is_an_exact_cached_enum() -> None:
    schema = choice_schema(("person:가", "person:나"))
    assert schema is choice_schema(("person:가", "person:나"))
    strict = to_strict_json_schema(schema)
    assert strict["properties"]["choice"]["enum"] == ["person:가", "person:나"]
    assert set(strict["required"]) == {"choice", "runbook_text", "rationale"}
    with pytest.raises(ValueError, match="후보가 없으면"):
        choice_schema(())


# --------------------------------------------------------------------------- reporters & less common duties


def _breach_org(*, reporter_status: PersonStatus = PersonStatus.DEPARTED) -> tuple[Org, dict[str, str]]:
    org = Org()
    owner = org.person("정민혁", title="과장", team="주문팀")
    gone = org.person("김보안", title="대리", team="보안", roles=("보안 담당자",), status=reporter_status)
    ciso = org.person("오준석", title="이사", team="보안", roles=("CISO",))
    cto = org.person("한서윤", title="CTO", team="경영", roles=("CTO",), channels=("phone",))
    svc = org.service("주문")
    org.trigger(svc, "주문 알람", owner)
    org.owner(svc, owner)
    oid = org.obligation(
        "KISA 침해사고 신고",
        incident=IncidentType.BREACH,
        recipient="KISA",
        deadline=1440,
        reporters=[gone],
        service=svc,
    )
    return org, {"owner": owner, "gone": gone, "ciso": ciso, "cto": cto, "svc": svc, "oid": oid}


def test_departed_reporter_of_an_obligation_is_replaced_by_the_responsible_role() -> None:
    org, ids = _breach_org()
    ctx = org.ctx()
    real = [f for f in run_checks(ctx) if f.rule_id == "JT-KR-002"]
    assert real
    assert real[0].fix is not None
    assert real[0].fix.kind is FixKind.REPLACE_DEPARTED
    (patch,) = draft(ctx, real)
    assert removed_edges(patch) == {edge_id(EdgeKind.REPORTED_BY, ids["oid"], ids["gone"])}
    assert set(added_edges(patch)) == {edge_id(EdgeKind.REPORTED_BY, ids["oid"], ids["ciso"])}
    assert patch.runbook_text == "KISA 침해사고 신고 담당은 오준석 이사로 한다."
    assert "퇴사자에게 걸린 신고 연결 1건은 지웁니다." in patch.rationale


def test_obligation_without_reporter_and_deputy_for_a_lone_reporter() -> None:
    org, ids = _breach_org(reporter_status=PersonStatus.ACTIVE)
    ctx = org.ctx()
    (assign,) = draft(ctx, [fake_finding(FixKind.ASSIGN_OWNER, obligation_id=ids["oid"])])
    assert set(added_edges(assign)) == {edge_id(EdgeKind.REPORTED_BY, ids["oid"], ids["ciso"])}
    (cover,) = draft(ctx, [fake_finding(FixKind.ADD_DEPUTY, obligation_id=ids["oid"])])
    assert cover.runbook_text == "KISA 침해사고 신고는 김보안 대리가 부재일 때 오준석 이사가 대신한다."
    (memo,) = draft(ctx, [fake_finding(FixKind.ASSIGN_OWNER, obligation_id="obligation:없음")])
    assert memo.ops == ()


def test_reporting_duty_of_a_departed_person_moves_with_the_service() -> None:
    org, ids = _breach_org()
    duty = fake_finding(
        FixKind.REPLACE_DEPARTED,
        service_id=ids["svc"],
        person_id=ids["gone"],
        edge_ids=[edge_id(EdgeKind.REPORTED_BY, ids["oid"], ids["gone"])],
    )
    (patch,) = draft(org.ctx(), [duty])
    assert set(added_edges(patch)) == {edge_id(EdgeKind.REPORTED_BY, ids["oid"], ids["ciso"])}
    assert patch.runbook_text == "KISA 침해사고 신고 담당은 오준석 이사로 한다."


def test_duties_outside_any_service_and_executor_duties_are_handed_over() -> None:
    org = Org()
    gone = org.person("김민준", title="대리", team="결제팀", status=PersonStatus.DEPARTED)
    mate = org.person("이서연", title="사원", team="결제팀", roles=("1차 당직",))
    stray = org.node(NodeKind.TRIGGER, "야간 배치 알람")
    org.edge(EdgeKind.NOTIFIES, stray, gone)
    svc = org.service("결제")
    action = org.action(svc, "결제 롤백", executors=[gone])
    loose = fake_finding(
        FixKind.REPLACE_DEPARTED, person_id=gone, edge_ids=[edge_id(EdgeKind.NOTIFIES, stray, gone)]
    )
    (patch,) = draft(org.ctx(), [loose])
    assert patch.runbook_text == "김민준 대리가 맡던 호출은 이서연 사원이 넘겨받는다."
    executor = fake_finding(
        FixKind.REPLACE_DEPARTED,
        service_id=svc,
        person_id=gone,
        edge_ids=[edge_id(EdgeKind.EXECUTED_BY, action, gone)],
    )
    (handover,) = draft(org.ctx(), [executor])
    assert handover.runbook_text == (
        "결제 장애가 나면 1차 담당 이서연 사원이 대응한다. 결제 롤백은 이서연 사원이 실행한다."
    )
    assert edge_id(EdgeKind.EXECUTED_BY, action, mate) in added_edges(handover)


def test_cycle_without_stated_wait_gets_a_proposal() -> None:
    org = Org()
    a = org.person("정민혁", title="과장", team="주문팀")
    b = org.person("최영훈", title="차장", team="주문팀")
    org.person("한서윤", title="CTO", roles=("CTO",))
    svc = org.service("주문", tier=2)
    org.owner(svc, a)
    org.escalate(a, b, 15)
    back = org.escalate(b, a)
    edges = [edge_id(EdgeKind.ESCALATES_TO, a, b), back]
    cycle = fake_finding(FixKind.BREAK_ESCALATION_CYCLE, service_id=svc, cycle=[a, b], edge_ids=edges)
    (patch,) = draft(org.ctx(), [cycle])
    assert removed_edges(patch) == {back}
    assert "10분 안에 응답하지 않으면 정민혁 과장에게 다시 연락하는 대신" in patch.runbook_text
    assert "제안값" in patch.rationale


def test_cycle_starting_at_an_unconfirmed_person_stays_with_a_human() -> None:
    org = Org()
    x = org.person("홍길동", title="사원", team="주문팀", in_roster=False)
    y = org.person("최영훈", title="차장", team="주문팀")
    org.person("한서윤", title="CTO", roles=("CTO",))
    svc = org.service("주문")
    org.owner(svc, y)
    org.escalate(y, x, 5)
    closing = org.escalate(x, y, 5)
    loop = fake_finding(FixKind.BREAK_ESCALATION_CYCLE, service_id=svc, cycle=[x, y], edge_ids=[closing])
    (memo,) = draft(org.ctx(), [loop])
    assert memo.ops == ()
    assert "명부에서 확인되지 않아" in memo.rationale


def test_lone_channel_people_get_a_second_medium() -> None:
    org = Org()
    owner = org.person(
        "최영훈", title="차장", team="주문팀", channels=("slack", "phone"), link_channels=False
    )
    slack = org.channel("slack")
    org.edge(EdgeKind.REACHABLE_VIA, owner, slack)
    bare = org.person("정민혁", title="과장", team="주문팀", channels=("phone",), link_channels=False)
    svc = org.service("주문")
    org.trigger(svc, "주문 알람", owner)
    org.owner(svc, owner)
    (patch,) = draft(
        org.ctx(), [fake_finding(FixKind.ADD_FALLBACK_CHANNEL, service_id=svc, channel_id=slack)]
    )
    assert patch.runbook_text == "최영훈 차장에게 슬랙이 안 되면 전화로 연락한다."
    (first,) = draft(
        org.ctx(), [fake_finding(FixKind.ADD_FALLBACK_CHANNEL, service_id=svc, person_ids=[bare])]
    )
    assert first.runbook_text == "정민혁 과장에게는 전화로 연락한다."


def test_a_medium_whose_channel_dies_with_the_service_is_not_offered() -> None:
    org = Org()
    owner = org.person(
        "이수진", title="대리", team="개발팀", channels=("phone", "internal_messenger"), link_channels=False
    )
    org.edge(EdgeKind.REACHABLE_VIA, owner, org.channel("phone"))
    messenger = org.channel("internal_messenger")
    svc = org.service("웹 프론트엔드")
    org.edge(EdgeKind.RUNS_IN, messenger, org.runs_in(svc, "상암 IDC"))
    same_room = fake_finding(FixKind.ADD_FALLBACK_CHANNEL, service_id=svc, person_ids=[owner])
    (memo,) = draft(org.ctx(), [same_room])
    assert memo.ops == ()
    nobody = fake_finding(FixKind.ADD_FALLBACK_CHANNEL, channel_id=messenger, person_ids=["person:없음"])
    (empty,) = draft(org.ctx(), [nobody])
    assert empty.ops == ()


def test_deadline_texts_cover_unnamed_duties_and_fractions() -> None:
    org = Org()
    oid = org.node(NodeKind.OBLIGATION, "고객 통지")
    (patch,) = draft(
        org.ctx(), [fake_finding(FixKind.FIX_DEADLINE, obligation_id=oid, required_minutes=90.5)]
    )
    assert patch.runbook_text == "'고객 통지' 사유를 알게 된 때부터 90.5분 이내에 신고한다."
    assert "미기재에서 90.5분으로" in patch.rationale


def test_reporting_steps_follow_each_incident_type_and_skip_existing_ones() -> None:
    org = Org()
    org.person("최개인", title="과장", roles=("CPO",))
    svc = org.service("결제")
    breach = fake_finding(
        FixKind.ADD_REPORTING_STEP,
        incident_type="breach",
        recipient="KISA",
        deadline_minutes=1440,
        service_ids=[svc],
    )
    privacy = fake_finding(
        FixKind.ADD_REPORTING_STEP,
        incident_type="privacy_leak",
        recipient="개인정보보호위원회",
        deadline_minutes=4320,
        service_ids=[svc],
    )
    (patch,) = draft(org.ctx(), [breach, privacy])
    names = [op.node.name for op in patch.ops if isinstance(op, AddNode)]
    assert names == ["개인정보보호위원회 개인정보 유출 신고"]
    assert "KISA 침해사고 신고 담당은 명부에서 후보를 찾지 못해" in patch.rationale
    org.obligation("KISA 침해사고 신고", incident=IncidentType.BREACH, recipient="KISA", deadline=1440)
    (again,) = draft(org.ctx(), [breach])
    assert again.ops == ()


def test_llm_choice_outside_the_candidates_keeps_the_template() -> None:
    plan_org = payment_org()
    ctx = plan_org.org.ctx()
    findings = run_checks(ctx)
    rogue = FakeLLM(lambda *_: {"choice": plan_org.cto, "runbook_text": "x", "rationale": "y"})
    with pytest.raises(LLMOutputError):  # the enum alone already refuses it at the adapter
        run(rogue.structured(task="draft_fix", instructions="", schema=choice_schema((plan_org.oncall,))))
    assert run(LlmDrafter(rogue).adraft(ctx, findings, workspace_id="ws", base_version=1)) == draft(
        ctx, findings
    )


def test_a_short_alias_inside_an_allowed_name_is_not_another_person() -> None:
    plan_org = payment_org()
    plan_org.org.roster.append(plan_org.org.roster[0].model_copy(update={"name": "서연", "aliases": ()}))
    ctx = plan_org.org.ctx()
    fake = FakeLLM(
        _answer(plan_org.oncall, "결제 알람이 오면 1차 담당 이서연 사원에게 슬랙으로 바로 연락한다.")
    )
    patches = run(LlmDrafter(fake).adraft(ctx, run_checks(ctx), workspace_id="ws", base_version=1))
    assert patch_for(patches, "JT-PPL-001").author is PatchAuthor.AI
