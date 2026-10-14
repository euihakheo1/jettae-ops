"""Adversarial tests for ``jettae.improve`` added in review.

Each block pins a defect found in review (and fixed): concurrent approvals of one patch, blank
actors leaving unaudited versions, model sentences that contradict the compiled ops or carry
invented contact details, AI and template patches sharing one id, runbook text in the trusted LLM
context, patch elements claiming somebody else's provenance, a validator crash on a person added
and removed by the same patch, services after the first unable to join an approved reporting step,
detection memos suggesting a CTO/CISO as first responder, and a Korean particle error.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import BaseModel

from jettae.check.context import CheckContext
from jettae.check.registry import finding, run_checks
from jettae.domain import (
    AddEdge,
    AddNode,
    Attr,
    Edge,
    EdgeKind,
    Extractor,
    Finding,
    FixHint,
    FixKind,
    GraphVersion,
    IncidentType,
    Node,
    NodeKind,
    OpsGraph,
    Patch,
    PatchAuthor,
    PatchOp,
    PatchStatus,
    PersonStatus,
    Policy,
    Provenance,
    RemoveNode,
    RuleInfo,
    SetAttr,
    Severity,
    SourceKind,
    SourceSpan,
    edge_id,
    node_id,
)
from jettae.domain.ports import VersionConflictError
from jettae.improve import (
    LlmDrafter,
    PatchStateError,
    TemplateDrafter,
    VersionService,
    apply_ops,
    validate_patch,
)
from jettae.llm.base import UntrustedDocument
from jettae.llm.fake import FakeLLM
from jettae.store import MemoryStore
from test_improve_support import Org, PaymentOrg, demo_ctx, demo_inputs, payment_org, run

WS = "ws-review"
NOW = datetime(2026, 10, 13, 9, 0, tzinfo=UTC)
PATCH_ID = "patch-review"
PATCH = Provenance(source=SourceKind.PATCH, extractor=Extractor.HUMAN, ref=PATCH_ID)


@pytest.fixture
def po() -> PaymentOrg:
    return payment_org()


def owner_patch(patches: list[Patch]) -> Patch:
    return next(p for p in patches if any(f.startswith("JT-PPL-001") for f in p.finding_ids))


def drafted(ctx: CheckContext, findings: list[Finding] | None = None) -> list[Patch]:
    found = run_checks(ctx) if findings is None else findings
    return TemplateDrafter().draft(ctx, found, workspace_id=WS, base_version=1)


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


# --------------------------------------------------------------------------- versioning


def _validated_store(po: PaymentOrg) -> tuple[MemoryStore, VersionService, Patch]:
    store = MemoryStore(clock=lambda: NOW)
    store.workspaces.create_workspace("리뷰", workspace_id=WS)
    service = VersionService(store.graphs, store.patches, store.audit, clock=lambda: NOW)
    service.commit_graph(WS, po.org.graph(), actor="jettae")
    ctx = po.org.ctx()
    patch = owner_patch(drafted(ctx))
    validated = patch.model_copy(
        update={"status": PatchStatus.VALIDATED, "validation": validate_patch(ctx, patch)}
    )
    store.patches.save_patch(validated)
    return store, service, validated


class _StalePatches:
    """A patch repository whose reads are frozen at an earlier moment (another approver's view)."""

    def __init__(self, store: MemoryStore, seen: Patch) -> None:
        self.inner = store.patches
        self.seen = seen

    def save_patch(self, patch: Patch) -> None:
        self.inner.save_patch(patch)

    def get_patch(self, patch_id: str) -> Patch:
        return self.seen if patch_id == self.seen.id else self.inner.get_patch(patch_id)

    def list_patches(self, workspace_id: str) -> list[Patch]:
        return self.inner.list_patches(workspace_id)


class _StaleHead:
    """A graph repository that still shows the head another approver saw before the race."""

    def __init__(self, store: MemoryStore, head: GraphVersion) -> None:
        self.inner = store.graphs
        self.seen = head

    def head(self, workspace_id: str) -> GraphVersion | None:
        return self.seen

    def get_version(self, workspace_id: str, number: int) -> GraphVersion:
        return self.inner.get_version(workspace_id, number)

    def list_versions(self, workspace_id: str) -> list[GraphVersion]:
        return self.inner.list_versions(workspace_id)

    def load_graph(self, graph_hash: str) -> OpsGraph:
        return self.inner.load_graph(graph_hash)

    def append_version(self, version: GraphVersion, graph: OpsGraph, *, expected_head: int | None) -> None:
        self.inner.append_version(version, graph, expected_head=expected_head)


def test_a_second_approval_read_before_the_first_keeps_the_patch_approved(po: PaymentOrg) -> None:
    store, first, patch = _validated_store(po)
    second = VersionService(store.graphs, _StalePatches(store, patch), store.audit, clock=lambda: NOW)
    first.approve(patch.id, actor="박재현", role="lead")
    with pytest.raises(PatchStateError, match="이미 승인되어 버전 2가 된 수정안"):
        second.approve(patch.id, actor="한서윤", role="admin")
    saved = store.patches.get_patch(patch.id)
    assert (saved.status, saved.decided_by) == (PatchStatus.APPROVED, "박재현")
    assert [v.patch_id for v in first.history(WS)] == [None, patch.id]
    assert "patch.superseded" not in [e.action for e in store.audit.entries(WS)]


def test_losing_the_append_race_to_the_same_patch_keeps_it_approved(po: PaymentOrg) -> None:
    store, first, patch = _validated_store(po)
    seen_head = store.graphs.head(WS)
    assert seen_head is not None
    second = VersionService(
        _StaleHead(store, seen_head), _StalePatches(store, patch), store.audit, clock=lambda: NOW
    )
    first.approve(patch.id, actor="박재현", role="lead")
    with pytest.raises(PatchStateError):
        second.approve(patch.id, actor="한서윤", role="lead")
    assert store.patches.get_patch(patch.id).status is PatchStatus.APPROVED
    assert len(first.history(WS)) == 2


def test_parallel_approvals_of_one_patch_append_exactly_one_version(po: PaymentOrg) -> None:
    store, service, patch = _validated_store(po)
    barrier = threading.Barrier(6)
    outcomes: list[str] = []
    lock = threading.Lock()

    def approve(name: str) -> None:
        barrier.wait()
        try:
            service.approve(patch.id, actor=name, role="lead")
            result = "ok"
        except (PatchStateError, VersionConflictError) as exc:
            result = type(exc).__name__
        with lock:
            outcomes.append(result)

    threads = [threading.Thread(target=approve, args=(f"리드{i}",)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert outcomes.count("ok") == 1
    assert set(outcomes) <= {"ok", "PatchStateError"}
    assert [v.patch_id for v in service.history(WS)] == [None, patch.id]
    assert store.patches.get_patch(patch.id).status is PatchStatus.APPROVED


def test_supersede_never_overwrites_a_decision_taken_meanwhile(po: PaymentOrg) -> None:
    store, _service, patch = _validated_store(po)
    rejected = patch.model_copy(update={"status": PatchStatus.REJECTED, "decided_by": "박재현"})

    class Listed(_StalePatches):
        """``list_patches`` still shows the patch pending; it was rejected right after."""

        def get_patch(self, patch_id: str) -> Patch:
            return self.inner.get_patch(patch_id)

        def list_patches(self, workspace_id: str) -> list[Patch]:
            return [patch]

    store.patches.save_patch(rejected)
    racing = VersionService(store.graphs, Listed(store, patch), store.audit, clock=lambda: NOW)
    racing.commit_graph(WS, apply_ops(po.org.graph(), patch.ops)[0], actor="jettae")
    assert store.patches.get_patch(patch.id).status is PatchStatus.REJECTED
    assert "patch.superseded" not in [e.action for e in store.audit.entries(WS)]


@pytest.mark.parametrize("actor", ["", "   ", "\t\n", "박재\x00현", "박재\ud800현"])
def test_a_blank_or_unstorable_actor_changes_nothing(po: PaymentOrg, actor: str) -> None:
    store, service, patch = _validated_store(po)
    audit_before = store.audit.entries(WS)
    with pytest.raises(ValueError, match="행위자"):
        service.approve(patch.id, actor=actor, role="lead")
    with pytest.raises(ValueError, match="행위자"):
        service.reject(patch.id, actor=actor, role="lead", reason="다시 검토")
    with pytest.raises(ValueError, match="행위자"):
        service.commit_graph(WS, apply_ops(po.org.graph(), patch.ops)[0], actor=actor)
    with pytest.raises(ValueError, match="행위자"):
        service.revert(WS, 1, actor=actor, role="lead")
    assert [v.number for v in service.history(WS)] == [1]
    assert store.patches.get_patch(patch.id).status is PatchStatus.VALIDATED
    assert store.audit.entries(WS) == audit_before


def test_an_unstorable_rejection_reason_leaves_the_patch_pending(po: PaymentOrg) -> None:
    store, service, patch = _validated_store(po)
    audit_before = store.audit.entries(WS)
    with pytest.raises(ValueError, match="거절 사유"):
        service.reject(patch.id, actor="박재현", role="lead", reason="깨진 글자 \ud800")
    assert store.patches.get_patch(patch.id).status is PatchStatus.VALIDATED
    assert store.audit.entries(WS) == audit_before


def test_actor_is_recorded_without_surrounding_whitespace(po: PaymentOrg) -> None:
    store, service, patch = _validated_store(po)
    version = service.approve(patch.id, actor="  박재현 ", role="lead")
    assert version.created_by == "박재현"
    assert store.patches.get_patch(patch.id).decided_by == "박재현"
    assert store.audit.entries(WS)[-1].actor == "박재현"


# --------------------------------------------------------------------------- validator


def test_a_person_added_and_removed_by_one_patch_is_still_checked(po: PaymentOrg) -> None:
    stranger = Node(
        id=node_id(NodeKind.PERSON, "홍길동"),
        kind=NodeKind.PERSON,
        name="홍길동",
        attrs={"status": Attr(value="active", source=SourceKind.PATCH)},
        prov=(PATCH,),
    )
    ops: list[PatchOp] = [AddNode(node=stranger), RemoveNode(node_id=stranger.id)]
    result = validate_patch(po.org.ctx(), hand_patch(ops, []))
    assert result.unknown_people == (stranger.id,)
    assert not result.passed


def _span_prov() -> Provenance:
    span = SourceSpan(doc_id="doc-runbook", start=0, end=4, quote="지어낸 문장", line=1)
    return Provenance(source=SourceKind.RUNBOOK, extractor=Extractor.RULE, spans=(span,))


@pytest.mark.parametrize(
    ("prov", "attr_source", "problem"),
    [
        (_span_prov(), SourceKind.PATCH, "출처가 이 수정안"),  # claims a runbook span it never had
        (Provenance(source=SourceKind.PATCH, extractor=Extractor.HUMAN, ref="patch-other"), None, "출처가"),
        (Provenance(source=SourceKind.ROSTER, extractor=Extractor.IMPORT), None, "출처가"),
        (PATCH, SourceKind.ROSTER, "속성 note"),
    ],
)
def test_patch_elements_must_be_attributed_to_the_patch(
    po: PaymentOrg, prov: Provenance, attr_source: SourceKind | None, problem: str
) -> None:
    ctx = po.org.ctx()
    findings = run_checks(ctx)
    attrs = {} if attr_source is None else {"note": Attr(value="x", source=attr_source)}
    approver = Edge(
        id=edge_id(EdgeKind.APPROVED_BY, po.rollback, po.lead),
        kind=EdgeKind.APPROVED_BY,
        src=po.rollback,
        dst=po.lead,
        attrs=attrs,
        prov=(prov,),
    )
    claimed = [f.id for f in findings if f.rule_id == "JT-PATH-020"]
    gate = SetAttr(
        target_id=po.rollback, key="requires_approval", value=Attr(value=True, source=SourceKind.PATCH)
    )
    result = validate_patch(
        ctx, hand_patch([AddEdge(edge=approver), gate], claimed), baseline_findings=findings
    )
    assert result.resolved_finding_ids == tuple(claimed)  # the fix itself works …
    assert not result.passed  # … but the patch passes it off as somebody else's fact
    assert any(problem in e for e in result.integrity_errors)


def test_a_set_value_must_come_from_the_patch(po: PaymentOrg) -> None:
    ctx = po.org.ctx()
    findings = run_checks(ctx)
    claimed = [f.id for f in findings if f.rule_id == "JT-PATH-020"]
    approver = Edge(
        id=edge_id(EdgeKind.APPROVED_BY, po.rollback, po.lead),
        kind=EdgeKind.APPROVED_BY,
        src=po.rollback,
        dst=po.lead,
        prov=(PATCH,),
    )
    gate = SetAttr(target_id=po.rollback, key="requires_approval", value=Attr(value=True))  # RUNBOOK default
    result = validate_patch(
        ctx, hand_patch([AddEdge(edge=approver), gate], claimed), baseline_findings=findings
    )
    assert not result.passed
    assert result.integrity_errors == (
        "2번째 연산(set_attr): 'action:결제-롤백'의 'requires_approval' 값의 출처가 수정안이 아닙니다.",
    )
    fixed = SetAttr(
        target_id=po.rollback, key="requires_approval", value=Attr(value=True, source=SourceKind.PATCH)
    )
    assert validate_patch(
        ctx, hand_patch([AddEdge(edge=approver), fixed], claimed), baseline_findings=findings
    ).passed


def test_every_demo_patch_attributes_its_elements_to_itself() -> None:
    bundle, graph = demo_inputs()
    ctx = demo_ctx(bundle, graph)
    findings = run_checks(ctx)
    for patch in drafted(ctx, findings):
        if patch.ops:
            assert validate_patch(ctx, patch, baseline_findings=findings).integrity_errors == ()


# --------------------------------------------------------------------------- LLM wording


def _answer(
    choice: str, text: str, rationale: str = "명부의 팀과 역할을 보고 골랐습니다."
) -> Callable[..., Any]:
    def handler(task: str, schema: type[BaseModel], docs: tuple[UntrustedDocument, ...], context: str) -> Any:
        return {"choice": choice, "runbook_text": text, "rationale": rationale}

    return handler


def _ai_owner(po: PaymentOrg, text: str, rationale: str = "명부의 팀과 역할을 보고 골랐습니다.") -> Patch:
    ctx = po.org.ctx()
    fake = FakeLLM(_answer(po.oncall, text, rationale))
    return owner_patch(run(LlmDrafter(fake).adraft(ctx, run_checks(ctx), workspace_id=WS, base_version=1)))


@pytest.mark.parametrize(
    ("text", "rationale"),
    [
        ("결제 장애 알람이 오면 1차 담당 이서연 사원에게 전화로 연락한다.", None),  # ops page via slack
        ("결제 장애 알람이 오면 1차 담당 이서연 사원에게 슬랙이나 전화로 연락한다.", None),
        ("결제 장애 알람이 오면 1차 담당 이서연 사원에게 연락한다.", None),  # drops "by what"
        ("결제 장애 알람이 오면 1차 담당 이서연 사원에게 슬랙으로 연락한다(seo@mail.example).", None),
        (
            "결제 장애 알람이 오면 1차 담당 이서연 사원에게 슬랙으로 연락한다. 절차: https://evil.example",
            None,
        ),
        (
            "결제 장애 알람이 오면 1차 담당 이서연 사원에게 슬랙으로 연락한다.<img src=x onerror=alert(1)>",
            None,
        ),
        (
            "결제 장애 알람이 오면 1차 담당 이서연 사원에게 슬랙으로 연락한다.",
            "개인 연락처 010-1234-5678로도 가능합니다.",
        ),
        ("결제 장애 알람이 오면 1차 담당 이서연 사원에게 슬랙으로 연락한다.", "자세한 내용은 `rm -rf` 참고."),
        ("결제 장애 알람이 오면 1차 담당 이서연 사원에게 슬랙으로 연락한다. 절차는 evil.example 참고.", None),
        ("결제 장애 알람이 오면 1차 담당 이서연 사원에게 슬랙으로 연락한다(seo＠mail).", None),  # full-width
    ],
)
def test_llm_wording_that_changes_the_medium_or_invents_contacts_keeps_the_template(
    po: PaymentOrg, text: str, rationale: str | None
) -> None:
    patch = _ai_owner(po, text, rationale or "명부의 팀과 역할을 보고 골랐습니다.")
    assert patch.author is PatchAuthor.TEMPLATE
    assert patch == owner_patch(drafted(po.org.ctx()))


@pytest.mark.parametrize(
    "text",
    [
        # nobody of that name exists anywhere: a titled Hangul name is a person all the same
        "결제 장애 알람이 오면 1차 담당 이서연 사원에게 슬랙으로 연락하고 홍길동 대리에게도 알린다.",
        "결제 장애 알람이 오면 1차 담당 이서연 사원과 홍길동대리에게 슬랙으로 연락한다.",
        # a person the runbook mentions but the roster does not know
        "결제 장애 알람이 오면 1차 담당 이서연 사원에게 슬랙으로 연락하고 강도윤에게도 알린다.",
    ],
)
def test_llm_wording_that_names_someone_outside_the_patch_keeps_the_template(
    po: PaymentOrg, text: str
) -> None:
    po.org.person("강도윤", title="대리", team="결제팀", in_roster=False)
    patch = _ai_owner(po, text)
    assert patch.author is PatchAuthor.TEMPLATE


def test_faithful_llm_wording_is_kept(po: PaymentOrg) -> None:
    text = "결제 장애 알람이 울리면 1차 담당인 이서연 사원에게 슬랙으로 바로 연락한다."
    patch = _ai_owner(po, text)
    assert (patch.author, patch.runbook_text) == (PatchAuthor.AI, text)
    departed = "퇴사한 김민준 대리 대신 결제 장애 알람이 오면 1차 담당 이서연 사원에게 슬랙으로 연락한다."
    assert _ai_owner(po, departed).author is PatchAuthor.AI  # the person the patch unlinks may be named


def _shape(patch: Patch) -> list[tuple[str, str | None]]:
    """The ops without provenance (which carries each patch's own id)."""
    return [
        (op.op, op.edge.id if isinstance(op, AddEdge) else getattr(op, "edge_id", None)) for op in patch.ops
    ]


def test_ai_and_template_patches_never_share_an_id(po: PaymentOrg) -> None:
    template = owner_patch(drafted(po.org.ctx()))
    same_words = _ai_owner(po, template.runbook_text)
    reworded = _ai_owner(po, "결제 장애 알람이 울리면 1차 담당인 이서연 사원에게 슬랙으로 바로 연락한다.")
    assert (same_words.author, reworded.author) == (PatchAuthor.AI, PatchAuthor.AI)
    assert _shape(same_words) == _shape(template) == _shape(reworded)  # same choice → same change
    assert len({template.id, same_words.id, reworded.id}) == 3
    assert reworded.id == _ai_owner(po, reworded.runbook_text).id  # still deterministic
    for patch in (same_words, reworded):  # elements are stamped with the patch's own id
        refs = {p.ref for op in patch.ops if isinstance(op, AddEdge) for p in op.edge.prov}
        assert refs == {patch.id}


def test_runbook_names_reach_the_model_only_inside_the_untrusted_document(po: PaymentOrg) -> None:
    ctx = po.org.ctx()
    fake = FakeLLM(lambda *_: {"choice": "x", "runbook_text": "x", "rationale": "x"})
    run(LlmDrafter(fake).adraft(ctx, run_checks(ctx), workspace_id=WS, base_version=1))
    assert fake.calls
    runbook_names = [ctx.graph.nodes[n].name for n in (po.rollback, po.alarm)] + [
        "결제 서비스",
        "결제 롤백 승인자",
    ]
    for call in fake.calls:
        assert not any(name in call.context for name in runbook_names), call.context
        assert call.documents[0].text.startswith("[결정할 것]")
    assert any("결제 롤백 승인자" in call.documents[0].text for call in fake.calls)


_NOISE = st.text(alphabet="<>@:/`.-가나다슬랙전화메일문자카톡메신저0123456789 ", max_size=24)


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(prefix=_NOISE, suffix=_NOISE)
def test_property_accepted_wording_never_changes_media_or_adds_contacts(prefix: str, suffix: str) -> None:
    """Whatever the model wraps around a faithful sentence, an accepted one keeps the template's facts."""
    plan = payment_org()
    ctx = plan.org.ctx()
    base = "결제 장애 알람이 오면 1차 담당 이서연 사원에게 슬랙으로 연락한다."
    findings = [f for f in run_checks(ctx) if f.rule_id == "JT-PPL-001"]
    fake = FakeLLM(_answer(plan.oncall, f"{prefix}{base}{suffix}"))
    (patch,) = run(LlmDrafter(fake).adraft(ctx, findings, workspace_id=WS, base_version=1))
    if patch.author is PatchAuthor.AI:
        assert not re.search(r"[<>@`]|://", patch.runbook_text)
        assert not any(word in patch.runbook_text for word in ("전화", "메일", "문자", "카톡", "메신저"))
        assert set(re.findall(r"\d+", patch.runbook_text)) <= {"1"}
        assert "슬랙" in patch.runbook_text


# --------------------------------------------------------------------------- drafter


def test_nothing_to_fix_means_no_patches_and_no_calls(po: PaymentOrg) -> None:
    ctx = po.org.ctx()
    without_fix = [f.model_copy(update={"fix": None}) for f in run_checks(ctx)]
    assert drafted(ctx, []) == drafted(ctx, without_fix) == []
    fake = FakeLLM(lambda *_: {"choice": "x", "runbook_text": "x", "rationale": "x"})
    assert run(LlmDrafter(fake).adraft(ctx, without_fix, workspace_id=WS, base_version=1)) == []
    assert fake.calls == []


def _regulated_org() -> tuple[Org, dict[str, str], str]:
    org = Org()
    org.policy = Policy(regulated_services=("결제", "주문", "회원"))
    privacy = org.person("윤아름", title="과장", team="경영지원", roles=("개인정보 보호책임자",))
    services: dict[str, str] = {}
    for name in ("결제", "주문", "회원"):
        owner = org.person(f"{name}담당", title="사원", team=f"{name}팀", roles=("1차 당직",))
        sid = org.service(name)
        org.trigger(sid, f"{name} 알람", owner)
        org.owner(sid, owner)
        services[name] = sid
    org.obligation(
        "결제 개인정보 유출 신고",
        incident=IncidentType.PRIVACY_LEAK,
        recipient="개인정보보호위원회",
        deadline=4320,
        reporters=[privacy],
        service=services["결제"],
    )
    return org, services, privacy


def test_services_after_the_first_join_the_approved_reporting_step() -> None:
    org, services, privacy = _regulated_org()
    graph = org.graph()
    rounds = []
    for _ in range(3):
        ctx = org.ctx().with_graph(graph)
        findings = run_checks(ctx)
        steps = [p for p in drafted(ctx, findings) if any(i.startswith("JT-KR-003") for i in p.finding_ids)]
        if not steps:
            break
        rounds.append(steps)
        assert all(p.ops for p in steps), [p.runbook_text for p in steps]
        first = steps[0]
        result = validate_patch(ctx, first, baseline_findings=findings)
        assert result.passed, result.notes
        graph, _ = apply_ops(graph, first.ops)
    assert [len(r) for r in rounds] == [2, 1]
    (shared,) = rounds[1]
    (op,) = shared.ops  # joins the existing step: no second node, no second reporter
    assert isinstance(op, AddEdge)
    assert op.edge.kind is EdgeKind.MUST_REPORT
    assert op.edge.src in services.values()
    assert "이미 있는" in shared.rationale
    assert "윤아름 과장" in shared.runbook_text
    assert "72시간" in shared.runbook_text
    obligations = [n for n in graph.nodes.values() if n.kind is NodeKind.OBLIGATION]
    assert len(obligations) == 2  # the runbook's own 결제 step plus one shared patch step
    assert privacy in {e.dst for e in graph.edges.values() if e.kind is EdgeKind.REPORTED_BY}


def test_a_late_existing_step_is_not_joined_automatically() -> None:
    org, _services, _privacy = _regulated_org()
    graph = org.graph()
    ctx = org.ctx().with_graph(graph)
    findings = run_checks(ctx)
    first = next(p for p in drafted(ctx, findings) if any(i.startswith("JT-KR-003") for i in p.finding_ids))
    graph, _ = apply_ops(graph, first.ops)
    step = next(op.node for op in first.ops if isinstance(op, AddNode))
    late = SetAttr(
        target_id=step.id, key="deadline_minutes", value=Attr(value=10080, source=SourceKind.PATCH)
    )
    graph, _ = apply_ops(graph, [late])
    ctx = org.ctx().with_graph(graph)
    (memo,) = [p for p in drafted(ctx) if any(i.startswith("JT-KR-003") for i in p.finding_ids)]
    assert memo.ops == ()
    assert "사람이 먼저 확인할 신고 단계" in memo.rationale


def _fix_kind(f: Finding) -> FixKind | None:
    return None if f.fix is None else f.fix.kind


def test_detection_memos_never_suggest_a_final_responsibility_holder_as_first_responder() -> None:
    bundle, graph = demo_inputs()
    ctx = demo_ctx(bundle, graph)
    findings = run_checks(ctx)
    by_id = {f.id: f for f in findings}
    memos = [
        p
        for p in drafted(ctx, findings)
        if not p.ops and any(_fix_kind(by_id[i]) is FixKind.ADD_DETECTION for i in p.finding_ids)
    ]
    assert memos
    terminal = {"한서윤", "조영석", "오준석"}  # CTO, 대표, CISO of the demo roster
    for memo in memos:
        services = {by_id[i].service_id for i in memo.finding_ids} - {None}
        owners = {ctx.graph.nodes[o.id].name for s in services if s for o in ctx.owners(s)}
        named = [name for name in terminal - owners if name in memo.runbook_text]
        assert not named, memo.runbook_text


RULE = RuleInfo(
    rule_id="JT-TEST-002", title="시험용 끊김", severity=Severity.HIGH, element=None, rationale="시험"
)


def test_slow_chain_rationale_uses_the_right_particle() -> None:
    org = Org()
    a = org.person("가나다", title="사원", team="팀")
    b = org.person("라마바", title="팀장", team="팀", roles=("팀장",))
    svc = org.service("결제")
    first = org.escalate(a, b, 30)
    second = org.escalate(b, a, 50)
    slow = finding(
        RULE,
        keys=("slow",),
        title="시험용 끊김",
        message="상향이 느립니다.",
        service_id=svc,
        fix=FixHint(kind=FixKind.FIX_DEADLINE, params={"edge_ids": [first, second], "required_minutes": 60}),
    )
    (patch,) = drafted(org.ctx(), [slow])
    assert "상향 대기 시간 합계 80분이 목표 복구 시간을 넘습니다." in patch.rationale


def _lone_owner_with_departed_reporter() -> Org:
    """주문: one owner without escalation (JT-ESC-001 and JT-SPF-001); its KISA reporter left."""
    org = Org()
    owner = org.person("정민혁", title="과장", team="주문팀")
    gone = org.person(
        "김보안", title="대리", team="보안", roles=("보안 담당자",), status=PersonStatus.DEPARTED
    )
    org.person("오준석", title="이사", team="보안", roles=("CISO",))
    org.person("한서윤", title="CTO", team="경영", roles=("CTO",), channels=("phone",))
    svc = org.service("주문")
    org.trigger(svc, "주문 알람", owner)
    org.owner(svc, owner)
    org.obligation(
        "KISA 침해사고 신고",
        incident=IncidentType.BREACH,
        recipient="KISA",
        deadline=1440,
        reporters=[gone],
        service=svc,
    )
    return org


def test_the_same_change_asked_by_two_rules_is_one_patch() -> None:
    org = _lone_owner_with_departed_reporter()
    ctx = org.ctx()
    findings = run_checks(ctx)
    rules = {f.id: f.rule_id for f in findings}
    patches = [p for p in drafted(ctx, findings) if p.ops]
    shapes = [frozenset(_shape(p)) for p in patches]
    assert len(set(shapes)) == len(shapes)  # no two proposals make the same change
    reporter = next(p for p in patches if any(rules[i] == "JT-KR-002" for i in p.finding_ids))
    assert sorted(rules[i] for i in reporter.finding_ids) == ["JT-KR-002", "JT-PPL-001"]
    assert reporter.runbook_text == "KISA 침해사고 신고 담당은 오준석 이사로 한다."  # not "신고 신고 담당"
    escalation = next(p for p in patches if any(rules[i] == "JT-ESC-001" for i in p.finding_ids))
    assert sorted(rules[i] for i in escalation.finding_ids) == ["JT-ESC-001", "JT-SPF-001"]
    for patch in (reporter, escalation):
        result = validate_patch(ctx, patch, baseline_findings=findings)
        assert result.passed, result.notes
        assert result.resolved_finding_ids == patch.finding_ids  # one approval closes both findings
    asked = [
        plan for plan in TemplateDrafter().plans(ctx, findings) if plan.candidates and plan.template().ops
    ]
    fake = FakeLLM(lambda *_: {"choice": "x", "runbook_text": "x", "rationale": "x"})
    run(LlmDrafter(fake).adraft(ctx, findings, workspace_id=WS, base_version=1))
    assert len(fake.calls) == len(asked)  # the LLM is asked once per merged decision
