"""Review tests: who a patch may name, and how the validator judges the people and channels it touches.

Each test pins a defect found in review (and fixed):

* namesakes (동명이인) read the *first* roster row with their name, so a 결제팀 김민수 borrowed the
  물류팀 김민수's role ("1차 당직") and registered media, the sentence named an ambiguous "김민수 사원",
  and the validator vouched for a departed namesake because an active one came first;
* a node on HOLD (모호한 지칭) or whose status a patch rewrote counted as a roster person;
* owners, approvers and escalation targets nobody can contact were proposed (JT-PATH-011/020 stay
  open, so such patches could only be rejected), and a CTO teammate became a first responder;
* leave cover kept the absent person's links beside the deputy's, which JT-PPL-003 never clears;
* giving a person their first contact medium re-keyed "fewer than two channels" and was rejected
  as a new HIGH finding, and a re-keyed gap that got *worse* would have passed as a mere rename;
* a draft whose links already exist announced the change ("…승인 후 실행한다") with no ops at all.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import BaseModel

from jettae.check.context import CheckContext
from jettae.check.elements import reachable_channels
from jettae.check.registry import finding, run_checks
from jettae.domain import (
    AddEdge,
    Attr,
    ChannelMedium,
    Edge,
    EdgeKind,
    Extractor,
    Finding,
    FixHint,
    FixKind,
    Node,
    NodeKind,
    Patch,
    PatchAuthor,
    PatchOp,
    PersonStatus,
    Provenance,
    RemoveEdge,
    RosterEntry,
    RuleInfo,
    SetAttr,
    Severity,
    SourceKind,
    Verification,
    edge_id,
    node_id,
)
from jettae.improve import NO_CANDIDATE, LlmDrafter, TemplateDrafter, apply_ops, validate_patch
from jettae.improve.validator import _match, roster_entry
from jettae.llm.base import UntrustedDocument
from jettae.llm.fake import FakeLLM
from test_improve_support import ROSTER, Org, payment_org, run

WS = "ws-identity"
PATCH_ID = "patch-identity"
PATCH = Provenance(source=SourceKind.PATCH, extractor=Extractor.HUMAN, ref=PATCH_ID)


def drafted(ctx: CheckContext, findings: list[Finding] | None = None) -> list[Patch]:
    found = run_checks(ctx) if findings is None else findings
    return TemplateDrafter().draft(ctx, found, workspace_id=WS, base_version=1)


def claiming(patches: list[Patch], rule_prefix: str) -> Patch:
    return next(p for p in patches if any(f.startswith(rule_prefix) for f in p.finding_ids))


def added(patch: Patch) -> dict[str, Edge]:
    return {op.edge.id: op.edge for op in patch.ops if isinstance(op, AddEdge)}


def hand_patch(ops: list[PatchOp], finding_ids: list[str]) -> Patch:
    return Patch(
        id=PATCH_ID,
        workspace_id=WS,
        base_version=1,
        finding_ids=tuple(finding_ids),
        ops=tuple(ops),
        runbook_text="시험",
        rationale="시험",
        author=PatchAuthor.HUMAN,
    )


def namesake(
    org: Org,
    name: str,
    *,
    team: str,
    title: str,
    roles: tuple[str, ...] = (),
    channels: tuple[str, ...] = ("slack",),
    status: PersonStatus = PersonStatus.ACTIVE,
) -> str:
    """A roster person whose name another row shares; the node id carries the team, as assemble does."""
    media = tuple(ChannelMedium(c) for c in channels)
    org.roster.append(
        RosterEntry(name=name, status=status, title=title, team=team, roles=roles, channels=media)
    )
    pid = node_id(NodeKind.PERSON, f"{name} {team}")
    attrs = {
        "status": Attr(value=status.value, source=SourceKind.ROSTER),
        "title": Attr(value=title, source=SourceKind.ROSTER),
    }
    org.builder.add_node(Node(id=pid, kind=NodeKind.PERSON, name=name, attrs=attrs, prov=(ROSTER,)))
    org.edge(EdgeKind.MEMBER_OF, pid, org.node(NodeKind.TEAM, team, ROSTER), ROSTER)
    for role in roles:
        org.edge(EdgeKind.HOLDS_ROLE, pid, org.node(NodeKind.ROLE, role, ROSTER), ROSTER)
    for medium in media:
        org.edge(EdgeKind.REACHABLE_VIA, pid, org.channel(medium.value), ROSTER)
    return pid


def two_minsu(org: Org) -> tuple[str, str]:
    """물류팀 김민수 (1차 당직, phone·kakaotalk) is listed first; 결제팀 김민수 has only slack."""
    logistics = namesake(
        org, "김민수", team="물류팀", title="사원", roles=("1차 당직",), channels=("phone", "kakaotalk")
    )
    payment = namesake(org, "김민수", team="결제팀", title="사원", channels=("slack",))
    return logistics, payment


# --------------------------------------------------------------------------- namesakes (drafter)


def test_a_namesake_is_proposed_with_their_own_roster_facts_and_named_with_their_team() -> None:
    org = Org()
    logistics, payment = two_minsu(org)
    gone = org.person(
        "김민준", title="대리", team="결제팀", status=PersonStatus.DEPARTED, channels=("slack",)
    )
    svc = org.service("결제")
    alarm = org.trigger(svc, "결제 알람", gone)
    org.owner(svc, gone)
    patch = claiming(drafted(org.ctx()), "JT-PPL-001")
    assert edge_id(EdgeKind.NOTIFIES, alarm, payment) in added(patch)
    assert patch.runbook_text == "결제 장애 알람이 오면 1차 담당 결제팀 김민수 사원에게 슬랙으로 연락한다."
    assert "1차 당직" not in patch.rationale  # that role belongs to the 물류팀 김민수
    assert logistics not in {n for e in added(patch).values() for n in (e.src, e.dst)}


def test_a_namesake_is_never_given_the_other_rows_contact_media() -> None:
    org = Org()
    _logistics, payment = two_minsu(org)
    svc = org.service("정산")
    org.trigger(svc, "정산 알람", payment)
    org.owner(svc, payment)
    patch = claiming(drafted(org.ctx()), "JT-PATH-030")
    assert patch.ops == ()  # the 결제팀 김민수 registered slack only; phone is the other 김민수's
    assert "명부에 연락 수단(예: 전화)을 추가 등록" in patch.runbook_text


def test_namesakes_no_fact_can_tell_apart_are_never_proposed() -> None:
    org = Org()
    first = namesake(org, "김민수", team="결제팀", title="사원", roles=("1차 당직",))
    second = node_id(NodeKind.PERSON, "김민수 2")
    org.roster.append(
        RosterEntry(name="김민수", title="사원", team="결제팀", channels=(ChannelMedium.SLACK,))
    )
    org.builder.add_node(
        org.builder.nodes[first].model_copy(update={"id": second}),
    )
    org.edge(EdgeKind.REACHABLE_VIA, second, org.channel("slack"), ROSTER)
    ctx = org.ctx()
    assert roster_entry(ctx, ctx.graph.nodes[first]) is None  # same team, title and status: who is who?
    gone = org.person("김민준", title="대리", team="결제팀", status=PersonStatus.DEPARTED)
    svc = org.service("결제")
    org.trigger(svc, "결제 알람", gone)
    org.owner(svc, gone)
    assert claiming(drafted(org.ctx()), "JT-PPL-001").ops == ()


def test_an_identical_roster_row_listed_twice_is_one_person() -> None:
    org = Org()
    oncall = org.person("이서연", title="사원", team="결제팀", roles=("1차 당직",))
    org.roster.append(org.roster[-1])
    ctx = org.ctx()
    assert roster_entry(ctx, ctx.graph.nodes[oncall]) == org.roster[0]


def _choose(person: str, text: str) -> Callable[..., Any]:
    def handler(task: str, schema: type[BaseModel], docs: tuple[UntrustedDocument, ...], context: str) -> Any:
        return {"choice": person, "runbook_text": text, "rationale": "명부의 팀과 역할을 보고 골랐습니다."}

    return handler


@pytest.mark.parametrize(
    ("text", "author"),
    [
        ("결제 장애 알람이 울리면 1차 담당인 결제팀 김민수 사원에게 슬랙으로 바로 연락한다.", PatchAuthor.AI),
        (
            "결제 장애 알람이 울리면 1차 담당인 물류팀 김민수 사원에게 슬랙으로 바로 연락한다.",
            PatchAuthor.TEMPLATE,
        ),
        ("결제 장애 알람이 울리면 1차 담당인 김민수 사원에게 슬랙으로 바로 연락한다.", PatchAuthor.TEMPLATE),
    ],
)
def test_a_rewording_must_name_the_chosen_namesake_by_team(text: str, author: PatchAuthor) -> None:
    org = Org()
    _logistics, payment = two_minsu(org)
    gone = org.person(
        "김민준", title="대리", team="결제팀", status=PersonStatus.DEPARTED, channels=("slack",)
    )
    svc = org.service("결제")
    org.trigger(svc, "결제 알람", gone)
    org.owner(svc, gone)
    ctx = org.ctx()
    fake = FakeLLM(_choose(payment, text))
    patches = run(LlmDrafter(fake).adraft(ctx, run_checks(ctx), workspace_id=WS, base_version=1))
    assert claiming(patches, "JT-PPL-001").author is author


# --------------------------------------------------------------------------- namesakes & identity (validator)


def _rollback_org() -> tuple[Org, str, str]:
    org = Org()
    svc = org.service("결제")
    return org, svc, org.action(svc, "결제 롤백")


def _executor(rollback: str, person: str) -> AddEdge:
    edge = Edge(
        id=edge_id(EdgeKind.EXECUTED_BY, rollback, person),
        kind=EdgeKind.EXECUTED_BY,
        src=rollback,
        dst=person,
        prov=(PATCH,),
    )
    return AddEdge(edge=edge)


def test_a_departed_namesake_is_unknown_even_when_an_active_one_is_listed_first() -> None:
    org, _svc, rollback = _rollback_org()
    active = namesake(org, "김민수", team="결제팀", title="사원")
    departed = namesake(org, "김민수", team="물류팀", title="사원", status=PersonStatus.DEPARTED)
    ctx = org.ctx()
    assert validate_patch(ctx, hand_patch([_executor(rollback, active)], [])).unknown_people == ()
    result = validate_patch(ctx, hand_patch([_executor(rollback, departed)], []))
    assert result.unknown_people == (departed,)
    assert any("동명이인" in note for note in result.notes)


def test_an_ambiguous_mention_on_hold_is_not_a_roster_person() -> None:
    org, _svc, rollback = _rollback_org()
    namesake(org, "김민수", team="결제팀", title="사원")
    namesake(org, "김민수", team="물류팀", title="사원")
    held = node_id(NodeKind.PERSON, "김민수")
    org.builder.add_node(
        Node(
            id=held,
            kind=NodeKind.PERSON,
            name="김민수",
            attrs={
                "status": Attr(value="unknown", source=SourceKind.RUNBOOK, spans=org.runbook("김민수").spans)
            },
            prov=(org.runbook("김민수 사원"),),
            verification=Verification.HOLD,
            hold_reason="동명이인/모호한 지칭",
        )
    )
    result = validate_patch(org.ctx(), hand_patch([_executor(rollback, held)], []))
    assert result.unknown_people == (held,)


def test_a_patch_cannot_vouch_for_a_person_by_rewriting_their_status() -> None:
    org, _svc, rollback = _rollback_org()
    gone = org.person("김민준", title="대리", team="결제팀", status=PersonStatus.DEPARTED)
    org.roster[-1] = org.roster[-1].model_copy(update={"status": PersonStatus.ACTIVE})  # roster rehired …
    revive = SetAttr(target_id=gone, key="status", value=Attr(value="active", source=SourceKind.PATCH))
    result = validate_patch(org.ctx(), hand_patch([revive, _executor(rollback, gone)], []))
    assert result.unknown_people == (gone,)  # … but the graph still says departed: re-diagnose first


# --------------------------------------------------------------------------- who can be reached


def test_an_approver_nobody_can_reach_is_never_proposed() -> None:
    org = Org()
    owner = org.person("이서연", title="사원", team="결제팀", roles=("1차 당직",))
    mute = org.person("정우성", title="팀장", team="결제팀", roles=("팀장",), channels=())
    cto = org.person("한서윤", title="CTO", team="경영", roles=("CTO",), channels=("phone",))
    svc = org.service("결제")
    org.trigger(svc, "결제 알람", owner)
    org.owner(svc, owner)
    org.escalate(owner, cto, 10)
    rollback = org.action(svc, "결제 롤백")
    ctx = org.ctx()
    findings = run_checks(ctx)
    patch = claiming(drafted(ctx, findings), "JT-PATH-020")
    assert set(added(patch)) == {edge_id(EdgeKind.APPROVED_BY, rollback, cto)}
    assert mute not in patch.rationale
    assert validate_patch(ctx, patch, baseline_findings=findings).passed


def test_an_owner_nobody_can_reach_is_never_proposed() -> None:
    org = Org()
    gone = org.person("김민준", title="대리", team="결제팀", status=PersonStatus.DEPARTED)
    org.person("이서연", title="사원", team="결제팀", roles=("1차 당직",), channels=())
    reachable = org.person("최민수", title="사원", team="결제팀", channels=("slack", "phone"))
    svc = org.service("결제")
    alarm = org.trigger(svc, "결제 알람", gone)
    org.owner(svc, gone)
    ctx = org.ctx()
    findings = run_checks(ctx)
    patch = claiming(drafted(ctx, findings), "JT-PPL-001")
    assert edge_id(EdgeKind.NOTIFIES, alarm, reachable) in added(patch)  # not the silent on-call
    result = validate_patch(ctx, patch, baseline_findings=findings)
    assert result.passed, result.notes


@pytest.mark.parametrize("with_teammate", [False, True])
def test_a_final_responsibility_holder_is_never_the_first_responder(with_teammate: bool) -> None:
    org = Org()
    gone = org.person("김민준", title="대리", team="결제팀", status=PersonStatus.DEPARTED)
    org.person("한서윤", title="CTO", team="결제팀", channels=("phone",))  # a title is enough
    mate = org.person("최민수", title="사원", team="결제팀") if with_teammate else None
    svc = org.service("결제")
    org.trigger(svc, "결제 알람", gone)
    org.owner(svc, gone)
    patch = claiming(drafted(org.ctx()), "JT-PPL-001")
    if mate is None:
        assert patch.ops == ()
        assert NO_CANDIDATE in patch.runbook_text
    else:
        assert edge_id(EdgeKind.OWNED_BY, svc, mate) in added(patch)
        assert "CTO" not in patch.runbook_text


# --------------------------------------------------------------------------- leave cover


def test_leave_cover_hands_the_duty_over_and_passes_validation() -> None:
    org = Org()
    away = org.person("배은지", title="대리", team="개발팀", status=PersonStatus.ON_LEAVE)
    mate = org.person("강현우", title="사원", team="개발팀", roles=("1차 당직",))
    lead = org.person("박재현", title="팀장", team="개발팀", roles=("팀장",))
    svc = org.service("추천", tier=2)
    alarm = org.trigger(svc, "추천 알람", away, mate)
    org.owner(svc, mate)
    org.escalate(mate, lead, 10)
    ctx = org.ctx()
    findings = run_checks(ctx)
    leave = [f.id for f in findings if f.rule_id == "JT-PPL-003"]
    assert leave
    patch = claiming(drafted(ctx, findings), "JT-PPL-003")
    assert {op.edge_id for op in patch.ops if isinstance(op, RemoveEdge)} == {
        edge_id(EdgeKind.NOTIFIES, alarm, away)
    }
    result = validate_patch(ctx, patch, baseline_findings=findings)
    assert result.passed, result.notes
    assert set(leave) <= set(result.resolved_finding_ids)


def test_a_deputy_whose_escalation_leads_back_to_the_absent_person_is_skipped() -> None:
    org = Org()
    away = org.person("배은지", title="대리", team="개발팀", status=PersonStatus.ON_LEAVE)
    looping = org.person("강현우", title="사원", team="개발팀", roles=("1차 당직",))
    other = org.person("이수진", title="대리", team="개발팀")
    svc = org.service("추천", tier=2)
    org.trigger(svc, "추천 알람", away)
    org.owner(svc, away)
    org.escalate(looping, away, 10)
    patch = claiming(drafted(org.ctx()), "JT-PPL-003")
    targets = {e.dst for e in added(patch).values()}
    assert looping not in targets
    assert other in targets


# --------------------------------------------------------------------------- continuations


def test_a_first_contact_medium_is_a_fix_not_a_new_gap() -> None:
    org = Org()
    owner = org.person("최영훈", title="차장", team="주문팀", channels=("phone",), link_channels=False)
    svc = org.service("주문")
    org.trigger(svc, "주문 알람", owner)
    org.owner(svc, owner)
    ctx = org.ctx()
    findings = run_checks(ctx)
    patch = claiming(drafted(ctx, findings), "JT-PATH-011")
    result = validate_patch(ctx, patch, baseline_findings=findings)
    assert result.passed, result.notes
    assert any(i.startswith("JT-PATH-011") for i in result.resolved_finding_ids)
    assert not any(i.startswith("JT-PATH-030") for i in result.new_finding_ids)  # still one medium: same gap


RULE = RuleInfo(rule_id="JT-TEST-009", title="시험", severity=Severity.MEDIUM, element=None, rationale="시험")


def _gap(person: str, severity: Severity) -> Finding:
    return finding(
        RULE,
        keys=(person,),
        title="시험",
        message="시험",
        service_id="service:결제",
        subject_ids=("service:결제", person),
        severity=severity,
        fix=FixHint(kind=FixKind.ASSIGN_OWNER, params={}),
    )


def test_a_rekeyed_gap_that_got_worse_is_new() -> None:
    org = Org()
    a = org.person("가나다", title="사원", team="팀")
    b = org.person("라마바", title="사원", team="팀")
    graph = org.graph()
    before = _gap(a, Severity.MEDIUM)
    renamed = _match([], [before], [_gap(b, Severity.MEDIUM)], (graph,))
    assert (renamed.new, len(renamed.continued)) == ((), 1)
    worse = _gap(b, Severity.HIGH)
    assert _match([], [before], [worse], (graph,)).new == (worse,)


# --------------------------------------------------------------------------- honesty of the sentence


@pytest.mark.parametrize("rule", ["JT-PATH-020", "JT-PATH-041"])
def test_a_change_already_in_the_graph_is_a_memo_not_an_announced_fix(rule: str) -> None:
    ctx = payment_org().org.ctx()
    findings = run_checks(ctx)
    fix = claiming(drafted(ctx, findings), rule)
    later = ctx.with_graph(apply_ops(ctx.graph, fix.ops)[0])
    (stale,) = drafted(later, [f for f in findings if f.rule_id == rule])  # findings of the older run
    assert stale.ops == ()
    assert "이미 그래프에 있어 바꿀 것이 없습니다" in stale.runbook_text
    assert "검토 메모" in stale.rationale
    assert not validate_patch(later, stale).passed


# --------------------------------------------------------------------------- property


PERSON = st.tuples(
    st.sampled_from([PersonStatus.ACTIVE, PersonStatus.ACTIVE, PersonStatus.DEPARTED, PersonStatus.ON_LEAVE]),
    st.sampled_from(["결제팀", "개발팀"]),
    st.sampled_from([(), ("1차 당직",), ("팀장",), ("CTO",)]),
    st.sampled_from([(), ("slack",), ("phone", "slack")]),
    st.booleans(),  # a namesake of the person before
)
_CONTACT_DUTIES = frozenset(
    {EdgeKind.OWNED_BY, EdgeKind.NOTIFIES, EdgeKind.APPROVED_BY, EdgeKind.ESCALATES_TO}
)


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(people=st.lists(PERSON, min_size=3, max_size=6), approver=st.integers(0, 5), loop=st.booleans())
def test_property_contact_duties_go_only_to_reachable_confirmed_people(
    people: list[tuple[PersonStatus, str, tuple[str, ...], tuple[str, ...], bool]], approver: int, loop: bool
) -> None:
    """Whatever the roster looks like (namesakes, people without media), a drafted owner, page,
    approver or escalation target is a confirmed roster person someone can actually reach."""
    org = Org()
    ids: list[str] = []
    for i, (status, team, roles, channels, same_name) in enumerate(people):
        name = "김민수" if same_name and i else f"사람{i}"
        if name == "김민수":
            ids.append(namesake(org, name, team=f"{team}{i}", title="사원", roles=roles, channels=channels))
        else:
            ids.append(
                org.person(name, title="사원", team=team, roles=roles, status=status, channels=channels)
            )
    svc = org.service("결제")
    org.trigger(svc, "결제 알람", ids[0])
    org.owner(svc, ids[0])
    org.action(svc, "결제 롤백", approvers=[ids[approver % len(ids)]])
    if loop:
        org.escalate(ids[1], ids[0], 10)
    ctx = org.ctx()
    findings = run_checks(ctx)
    for patch in drafted(ctx, findings):
        result = validate_patch(ctx, patch, baseline_findings=findings)
        assert (result.unknown_people, result.integrity_errors) == ((), ())
        for edge in added(patch).values():
            node = ctx.graph.nodes[edge.dst]
            if edge.kind in _CONTACT_DUTIES and node.kind is NodeKind.PERSON:
                assert reachable_channels(ctx, edge.dst), (patch.runbook_text, edge.id)
                assert roster_entry(ctx, node) is not None
