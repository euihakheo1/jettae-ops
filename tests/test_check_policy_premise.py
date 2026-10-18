"""Findings whose premise the policy supplies say so (rubric v4 judges such premises as given).

A service whose tier no source states is critical only by the policy's default tier; a rollback
or failover of a critical service needs an approver only by the policy's default gate; a recovery
target the document does not state is the default tier's value. A finding built on those premises
names the policy in its message ("정책(기본: 미분류 서비스는 핵심 등급) 기준"), so a reader judges the
document only on what it says under them (is an approver named?). A tier the catalog or the
runbook states, or an approval the runbook states, is no policy premise and keeps its wording.
All fixtures are written for these tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from jettae.check.context import CheckContext
from jettae.check.elements import policy_tier_basis
from jettae.check.registry import run_checks
from jettae.check.rules.deadlines import EscalationExceedsRecoveryTarget
from jettae.check.rules.path import POLICY_APPROVAL_TAIL
from jettae.domain import (
    ActionType,
    Attr,
    CatalogService,
    Edge,
    EdgeKind,
    EntityRef,
    ExtractedAttr,
    ExtractedEntity,
    ExtractedRelation,
    Extraction,
    Extractor,
    Finding,
    GraphBuilder,
    Node,
    NodeKind,
    PersonStatus,
    Policy,
    Provenance,
    SourceKind,
    SourceSpan,
    edge_id,
    node_id,
)
from jettae.extract import assemble
from jettae.ingest import load_bytes

BASIS = "정책(기본: 미분류 서비스는 핵심 등급) 기준"

RUNBOOK = """\
# Ledger API

The Ledger API records every payment.
To mitigate, roll back the last ledger deploy.
"""
STEP = "roll back the last ledger deploy"


def _approver_finding(
    *, catalog: Sequence[CatalogService] = (), requires_approval: str | None = None
) -> Finding:
    doc = load_bytes("ledger/api.md", RUNBOOK.encode("utf-8"))
    attrs = [ExtractedAttr(key="action_type", value="rollback", quote="roll back")]
    if requires_approval is not None:
        attrs.append(ExtractedAttr(key="requires_approval", value=requires_approval, quote="To mitigate"))
    extraction = Extraction(
        doc_id=doc.doc_id,
        extractor=Extractor.LLM,
        entities=(
            ExtractedEntity(
                kind=NodeKind.SERVICE, name="Ledger API", quote="The Ledger API records every payment."
            ),
            ExtractedEntity(kind=NodeKind.ACTION, name=STEP, quote=STEP, attrs=tuple(attrs)),
        ),
        relations=(
            ExtractedRelation(
                kind=EdgeKind.MITIGATED_BY,
                src=EntityRef(kind=NodeKind.SERVICE, name="Ledger API"),
                dst=EntityRef(kind=NodeKind.ACTION, name=STEP),
                quote="To mitigate, roll back the last ledger deploy.",
            ),
        ),
    )
    graph = assemble([extraction], [doc], catalog=catalog).graph
    findings = run_checks(CheckContext(graph=graph, catalog=tuple(catalog)))
    return next(f for f in findings if f.rule_id == "JT-PATH-020")


def test_an_approval_premise_from_the_default_tier_names_the_policy() -> None:
    finding = _approver_finding()
    assert f"{BASIS}으로 승인이 필요하지만 승인자가 적혀 있지 않습니다" in finding.message
    assert finding.message.endswith(POLICY_APPROVAL_TAIL)
    assert "문서가 아니라 정책에서 왔습니다" in finding.message


def test_a_stated_tier_keeps_the_plain_wording() -> None:
    finding = _approver_finding(catalog=[CatalogService(name="Ledger API", tier=1)])
    assert "승인이 필요하지만 승인자가 적혀 있지 않습니다" in finding.message
    assert "정책(" not in finding.message


def test_a_stated_approval_is_no_policy_premise() -> None:
    finding = _approver_finding(requires_approval="true")
    assert "승인을 받도록 적혀 있지만" in finding.message
    assert "정책(" not in finding.message


def test_policy_basis_follows_the_policy_default_tier() -> None:
    builder = GraphBuilder()
    builder.add_node(Node(id="service:a", kind=NodeKind.SERVICE, name="a", prov=(_RUNBOOK,)))
    graph = builder.build()
    assert policy_tier_basis(CheckContext(graph=graph), "service:a") == BASIS
    lenient = Policy(default_tier=3)
    assert policy_tier_basis(CheckContext(graph=graph, policy=lenient), "service:a") == (
        "정책(기본: 미분류 서비스는 3등급) 기준"
    )
    stated = [CatalogService(name="a", tier=2)]
    assert policy_tier_basis(CheckContext(graph=graph, catalog=tuple(stated)), "service:a") is None


# --------------------------------------------------------------------------- recovery target


_SPAN = SourceSpan(doc_id="doc-test", start=0, end=4, quote="근거", line=1)
_RUNBOOK = Provenance(source=SourceKind.RUNBOOK, extractor=Extractor.RULE, spans=(_SPAN,))


def _attrs(values: dict[str, Any]) -> dict[str, Attr]:
    return {k: Attr(value=v, spans=(_SPAN,), source=SourceKind.RUNBOOK) for k, v in values.items()}


def _escalation_ctx(**service_attrs: Any) -> CheckContext:
    builder = GraphBuilder()

    def node(kind: NodeKind, name: str, **attrs: Any) -> str:
        nid = node_id(kind, name)
        builder.add_node(Node(id=nid, kind=kind, name=name, attrs=_attrs(attrs), prov=(_RUNBOOK,)))
        return nid

    def edge(kind: EdgeKind, src: str, dst: str, **attrs: Any) -> None:
        eid = edge_id(kind, src, dst)
        builder.add_edge(Edge(id=eid, kind=kind, src=src, dst=dst, attrs=_attrs(attrs), prov=(_RUNBOOK,)))

    svc = node(NodeKind.SERVICE, "정산", **service_attrs)
    people = [node(NodeKind.PERSON, n, status=PersonStatus.ACTIVE.value) for n in ("담당A", "팀장B", "책임C")]
    rollback = node(NodeKind.ACTION, "정산 롤백", action_type=ActionType.ROLLBACK.value)
    edge(EdgeKind.OWNED_BY, svc, people[0])
    edge(EdgeKind.ESCALATES_TO, people[0], people[1], after_minutes=60)
    edge(EdgeKind.ESCALATES_TO, people[1], people[2], after_minutes=300)
    edge(EdgeKind.MITIGATED_BY, svc, rollback)
    edge(EdgeKind.APPROVED_BY, rollback, people[2])
    return CheckContext(graph=builder.build())


def test_a_recovery_target_from_the_default_tier_names_the_policy() -> None:
    [finding] = EscalationExceedsRecoveryTarget().check(_escalation_ctx())
    assert "목표 복구 시간 4시간" in finding.message
    assert f"목표 복구 시간은 문서에 없어 {BASIS}의 등급 기본값을 썼습니다." in finding.message


def test_a_stated_recovery_target_or_tier_is_no_policy_premise() -> None:
    [stated_slo] = EscalationExceedsRecoveryTarget().check(_escalation_ctx(slo_minutes=120))
    assert "정책(" not in stated_slo.message
    [stated_tier] = EscalationExceedsRecoveryTarget().check(_escalation_ctx(tier=2))
    assert "정책(" not in stated_tier.message
