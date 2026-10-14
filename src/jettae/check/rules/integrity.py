"""Integrity rules (판단 보류/무결성): ``JT-INT-001``, ``JT-INT-002``.

Why: the other rules can only be as trustworthy as the graph they read.

* ``JT-INT-001`` surfaces every element the pipeline put on HOLD (역번역 대조 불일치, 동명이인,
  모호한 정렬 …). A held element is *not* treated as fine — "확인 불가 ≠ 정상" (P3) — so each one
  becomes an UNKNOWN finding that asks a human to confirm it.
* ``JT-INT-002`` is an invariant alarm for P2 (지어내기 금지): anything claiming to come from a
  runbook must carry the span it came from. ``Provenance`` already refuses this at construction,
  so a hit means validation was bypassed (``model_construct``, unvalidated deserialisation, an
  ``Attr`` left on its RUNBOOK default without spans) — and the value must not be judged on.
"""

from __future__ import annotations

from collections.abc import Iterator

from jettae.check.context import CheckContext
from jettae.check.registry import finding, register
from jettae.domain import (
    Edge,
    EdgeKind,
    Element,
    Finding,
    FindingStatus,
    FixHint,
    FixKind,
    Node,
    NodeKind,
    RuleInfo,
    Severity,
    SourceKind,
    Verification,
)

EDGE_LABEL_KO: dict[EdgeKind, str] = {
    EdgeKind.DEPENDS_ON: "의존",
    EdgeKind.RUNS_IN: "실행 위치",
    EdgeKind.OWNED_BY: "담당",
    EdgeKind.DETECTED_BY: "감지",
    EdgeKind.NOTIFIES: "호출",
    EdgeKind.ESCALATES_TO: "상향",
    EdgeKind.MITIGATED_BY: "조치",
    EdgeKind.APPROVED_BY: "승인",
    EdgeKind.EXECUTED_BY: "실행",
    EdgeKind.REACHABLE_VIA: "연락 수단",
    EdgeKind.FALLBACK_TO: "대체 채널",
    EdgeKind.HOLDS_ROLE: "역할",
    EdgeKind.MEMBER_OF: "소속",
    EdgeKind.MUST_REPORT: "신고 의무",
    EdgeKind.REPORTED_BY: "신고 담당",
}

#: Which response-path element a held edge/node belongs to, so scoring can treat it as UNKNOWN.
_EDGE_ELEMENT: dict[EdgeKind, Element] = {
    EdgeKind.OWNED_BY: Element.OWNER,
    EdgeKind.ESCALATES_TO: Element.OWNER,
    EdgeKind.DETECTED_BY: Element.DETECTION,
    EdgeKind.NOTIFIES: Element.DETECTION,
    EdgeKind.MITIGATED_BY: Element.ROLLBACK,
    EdgeKind.EXECUTED_BY: Element.ROLLBACK,
    EdgeKind.APPROVED_BY: Element.APPROVER,
    EdgeKind.REACHABLE_VIA: Element.FALLBACK_CHANNEL,
    EdgeKind.FALLBACK_TO: Element.FALLBACK_CHANNEL,
}
_NODE_ELEMENT: dict[NodeKind, Element] = {
    NodeKind.TRIGGER: Element.DETECTION,
    NodeKind.ACTION: Element.ROLLBACK,
    NodeKind.CHANNEL: Element.FALLBACK_CHANNEL,
}
_PEOPLE = frozenset({NodeKind.PERSON, NodeKind.ROLE, NodeKind.TEAM})


def element_label(ctx: CheckContext, element: Node | Edge) -> str:
    """Human-readable Korean label for a node ("김대리") or an edge ("결제 → 김대리 (담당)")."""
    if isinstance(element, Node):
        return element.name
    src, dst = ctx.graph.nodes.get(element.src), ctx.graph.nodes.get(element.dst)
    src_name = src.name if src is not None else element.src
    dst_name = dst.name if dst is not None else element.dst
    return f"{src_name} → {dst_name} ({EDGE_LABEL_KO[element.kind]})"


def _hold_context(ctx: CheckContext, element: Node | Edge) -> tuple[str | None, Element | None, list[str]]:
    """(service id, path element, people whose *identity* is unresolved) for a held element.

    Identity is in question only when a person/role/team node itself is on HOLD (동명이인,
    모호한 지칭). A held edge between verified people is a back-translation mismatch about the
    relation (e.g. "10분" vs "15분"), so asking to confirm who the people are would mislead.
    """
    if isinstance(element, Node):
        sid = element.id if element.kind is NodeKind.SERVICE else None
        people = [element.id] if element.kind in _PEOPLE else []
        return sid, _NODE_ELEMENT.get(element.kind), people
    ends = [ctx.graph.nodes[n] for n in (element.src, element.dst) if n in ctx.graph.nodes]
    src = ctx.graph.nodes.get(element.src)
    sid = element.src if src is not None and src.kind is NodeKind.SERVICE else None
    people = [n.id for n in ends if n.kind in _PEOPLE and n.verification is Verification.HOLD]
    return sid, _EDGE_ELEMENT.get(element.kind), people


def _held(ctx: CheckContext) -> Iterator[Node | Edge]:
    yield from (n for n in ctx.graph if n.verification is Verification.HOLD)
    yield from (e for e in ctx.graph.edges_of(*EdgeKind) if e.verification is Verification.HOLD)


@register
class HeldElement:
    info = RuleInfo(
        rule_id="JT-INT-001",
        title="판단 보류 요소",
        severity=Severity.MEDIUM,
        element=None,
        rationale=(
            "역번역 대조가 어긋나거나 동명이인처럼 확정할 수 없는 요소는 정상으로 치지 않고 "
            "'확인 불가'로 보고해 사람이 확인하게 한다."
        ),
    )

    def check(self, ctx: CheckContext) -> Iterator[Finding]:
        for element in _held(ctx):
            sid, path_element, people = _hold_context(ctx, element)
            reason = element.hold_reason or "사유 미기재"
            if people:
                hint = "명부에서 맞는 사람을 확인해 주세요."
                fix = FixHint(
                    kind=FixKind.CONFIRM_IDENTITY, params={"element_id": element.id, "person_ids": people}
                )
            else:
                # Not an identity question: a person must re-read the source and confirm or
                # correct the element itself, so the drafter gets a generic review intent.
                hint = "원문 해당 부분을 확인해 주세요."
                fix = FixHint(kind=FixKind.REVIEW_ELEMENT, params={"element_id": element.id})
            yield finding(
                self.info,
                keys=(element.id,),
                title=self.info.title,
                message=(
                    f"'{element_label(ctx, element)}'은(는) 판단 보류 상태입니다(사유: {reason}). "
                    f"확인 전까지 이 요소는 정상으로 보지 않습니다. {hint}"
                ),
                service_id=sid,
                element=path_element,
                subject_ids=(element.id,),
                evidence=element.spans,
                fix=fix,
                status=FindingStatus.UNKNOWN,
            )


def _ungrounded(element: Node | Edge) -> list[str]:
    """What in ``element`` claims a runbook origin without a span ("" = the element itself)."""
    out = [""] if any(p.source is SourceKind.RUNBOOK and not p.spans for p in element.prov) else []
    out.extend(
        key
        for key, attr in sorted(element.attrs.items())
        if attr.source is SourceKind.RUNBOOK and not attr.spans
    )
    return out


@register
class UngroundedElement:
    info = RuleInfo(
        rule_id="JT-INT-002",
        title="원문 근거 없는 요소",
        severity=Severity.CRITICAL,
        element=None,
        rationale=(
            "런북에서 왔다는 요소·속성에 원문 위치가 없으면 지어낸 값일 수 있다. "
            "정상 경로에서는 생길 수 없으므로 생기면 무결성 경보로 다룬다(지어내기 금지)."
        ),
    )

    def check(self, ctx: CheckContext) -> Iterator[Finding]:
        elements: list[Node | Edge] = [*ctx.graph, *ctx.graph.edges_of(*EdgeKind)]
        for element in elements:
            label = element_label(ctx, element)
            for key in _ungrounded(element):
                what = f"'{label}'의 속성 '{key}'은(는)" if key else f"'{label}'은(는)"
                yield finding(
                    self.info,
                    keys=(element.id, key),
                    title=self.info.title,
                    message=(
                        f"{what} 런북에서 왔다고 되어 있지만 원문 위치(근거)가 없습니다. "
                        "지어낸 값일 수 있어 판정에 쓰면 안 됩니다."
                    ),
                    subject_ids=(element.id,),
                )
