"""Catalog cross-checks (카탈로그 대조): ``JT-CAT-001`` … ``JT-CAT-003``.

Why: runbooks only describe the services someone bothered to write about. The service catalog
(Backstage, deployment manifests, OTel service graph) says what actually runs. Comparing the two
finds the "목록 공백" — services that exist but that no runbook or owner covers, which is
exactly what a whole-site outage such as a data-centre fire exposes all at once — and the
concentration risk of every critical service living in one location with no way to move.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator, Mapping

from jettae.check.context import CheckContext
from jettae.check.impact import PROPAGATING_EDGES
from jettae.check.registry import finding, register
from jettae.check.rules.regulatory import listed_names, name_key, service_keys
from jettae.domain import (
    ActionType,
    AttrKey,
    Edge,
    EdgeKind,
    Element,
    Finding,
    FindingStatus,
    FixHint,
    FixKind,
    Node,
    NodeKind,
    OpsGraph,
    RuleInfo,
    Severity,
    SourceKind,
    Verification,
)


def _sources(node: Node) -> set[SourceKind]:
    return {p.source for p in node.prov}


def _runbook_touched(ctx: CheckContext) -> set[str]:
    """Node ids that any runbook-derived node or edge mentions."""
    touched = {n.id for n in ctx.graph.nodes.values() if SourceKind.RUNBOOK in _sources(n)}
    for edge in ctx.graph.edges.values():
        if any(p.source is SourceKind.RUNBOOK for p in edge.prov):
            touched.update((edge.src, edge.dst))
    return touched


@register
class CatalogServiceWithoutOwnerOrRunbook:
    info = RuleInfo(
        rule_id="JT-CAT-001",
        title="담당·런북 없는 서비스",
        severity=Severity.HIGH,
        element=Element.OWNER,
        rationale=(
            "서비스 목록에는 있지만 담당자도 대응 런북도 없는 서비스는 장애 시 누가 무엇을 할지 "
            "아무 데도 적혀 있지 않다. 센터 화재처럼 여러 시스템이 한꺼번에 멈추는 상황에서 "
            "가장 늦게 복구되는 '목록 공백'이다."
        ),
    )

    def check(self, ctx: CheckContext) -> Iterator[Finding]:
        touched = _runbook_touched(ctx)
        for svc in ctx.services():
            if SourceKind.CATALOG not in _sources(svc) or svc.id in touched or ctx.owners(svc.id):
                continue
            yield finding(
                self.info,
                keys=(svc.id,),
                title=self.info.title,
                message=(
                    f"'{svc.name}'은(는) 서비스 목록(카탈로그)에는 있지만 담당자도 장애 대응 런북도 "
                    "없습니다. 장애가 나면 누가 무엇을 할지 아무 문서에도 없습니다."
                ),
                service_id=svc.id,
                subject_ids=(svc.id,),
                fix=FixHint(kind=FixKind.ADD_SERVICE_RUNBOOK, params={"service_id": svc.id}),
            )


def _catalog_keys(ctx: CheckContext) -> set[str]:
    """:func:`name_key` of every catalog name and alias ("결제 서비스" == "결제-서비스")."""
    return {name_key(n) for svc in ctx.catalog for n in (svc.name, *svc.aliases)}


@register
class RunbookServiceNotInCatalog:
    info = RuleInfo(
        rule_id="JT-CAT-002",
        title="카탈로그에 없는 런북 서비스",
        severity=Severity.INFO,
        element=None,
        rationale=(
            "카탈로그가 있는데 런북에만 나오는 서비스는 이름이 다르거나, 폐기됐거나, 목록에서 빠진 것이다. "
            "어느 쪽인지 확인할 수 없으므로 '확인 불가'로 알린다. 띄어쓰기·대소문자·하이픈 차이는 "
            "같은 이름으로 본다."
        ),
    )

    def check(self, ctx: CheckContext) -> Iterator[Finding]:
        if not ctx.catalog:
            return
        catalog_keys = _catalog_keys(ctx)
        for svc in ctx.services():
            known = bool(service_keys(svc) & catalog_keys)
            if known or SourceKind.CATALOG in _sources(svc) or SourceKind.RUNBOOK not in _sources(svc):
                continue
            yield finding(
                self.info,
                keys=(svc.id,),
                title=self.info.title,
                message=(
                    f"'{svc.name}'은(는) 런북에는 있지만 서비스 카탈로그에서 찾을 수 없습니다. "
                    "이름이 다른지, 폐기된 서비스인지 확인해 주세요."
                ),
                service_id=svc.id,
                subject_ids=(svc.id,),
                evidence=svc.spans,
                status=FindingStatus.UNKNOWN,
            )


def _failover_protected(ctx: CheckContext) -> set[str]:
    """Services/systems with a verified failover mitigation.

    A failover on HOLD (판단 보류) is not counted: an unconfirmed escape route must not hide a
    concentration risk (P3).
    """
    return {
        e.src
        for e in ctx.graph.edges_of(EdgeKind.MITIGATED_BY)
        if e.verification is Verification.VERIFIED
        and ctx.graph.nodes[e.dst].verification is Verification.VERIFIED
        and ctx.graph.nodes[e.dst].attr(AttrKey.ACTION_TYPE) == ActionType.FAILOVER.value
    }


def _dependency_closure(graph: OpsGraph, starts: Iterable[str]) -> set[str]:
    """``starts`` plus everything they transitively DEPENDS_ON / RUNS_IN (same edges as impact)."""
    targets: dict[str, list[str]] = {}
    for edge in graph.edges.values():
        if edge.kind in PROPAGATING_EDGES:
            targets.setdefault(edge.src, []).append(edge.dst)
    seen = set(starts)
    stack = sorted(seen)
    while stack:
        for dst in targets.get(stack.pop(), ()):
            if dst not in seen:
                seen.add(dst)
                stack.append(dst)
    return seen


def _falls_through(graph: OpsGraph, nid: str, down: Mapping[str, Edge | None]) -> Edge | None:
    """The edge that takes ``nid`` down given the ``down`` nodes, or ``None`` if it survives.

    Dependencies are all needed (one failed DEPENDS_ON target is enough), but placements are
    alternatives: a node that verifiably RUNS_IN two locations survives losing one of them.
    A placement on HOLD is never counted as an escape route (P3), only as exposure.
    """
    edges = graph.out_edges(nid, EdgeKind.DEPENDS_ON, EdgeKind.RUNS_IN)
    dependency = next((e for e in edges if e.kind is EdgeKind.DEPENDS_ON and e.dst in down), None)
    if dependency is not None:
        return dependency
    placements = [e for e in edges if e.kind is EdgeKind.RUNS_IN]
    hit = next((e for e in placements if e.dst in down), None)
    escapes = [e for e in placements if e.verification is Verification.VERIFIED and e.dst not in down]
    return hit if hit is not None and not escapes else None


def _location_failure(graph: OpsGraph, loc_id: str) -> dict[str, Edge | None]:
    """Every node down when ``loc_id`` burns, mapped to the edge that took it down.

    Unlike :func:`~jettae.check.impact.blast_radius` (which over-reports on purpose to list
    *exposure*), this asks whether the location is a true single point of failure, so a
    service running active-active in two centres is not reported as concentrated in both.
    Each newly failed node re-examines its dependents, so the fixpoint is reached in
    O(V + E) re-checks; the traversal order is sorted, so the recorded causes are stable.
    """
    down: dict[str, Edge | None] = {loc_id: None}
    queue = deque([loc_id])
    while queue:
        cur = queue.popleft()
        for edge in graph.in_edges(cur, EdgeKind.DEPENDS_ON, EdgeKind.RUNS_IN):
            if edge.src in down:
                continue
            cause = _falls_through(graph, edge.src, down)
            if cause is not None:
                down[edge.src] = cause
                queue.append(edge.src)
    return down


def _cause_chain(down: Mapping[str, Edge | None], nid: str) -> list[str]:
    """Edge ids from ``nid`` back to the failed location (the "why" shown in the finding)."""
    chain: list[str] = []
    cause = down[nid]
    while cause is not None:
        chain.append(cause.id)
        cause = down[cause.dst]
    return chain


@register
class CriticalServicesConcentrated:
    info = RuleInfo(
        rule_id="JT-CAT-003",
        title="핵심 서비스가 한 위치에 집중",
        severity=Severity.MEDIUM,
        element=None,
        rationale=(
            "모든 핵심 서비스가 (직접 또는 의존 시스템을 통해) 한 위치에서 돌고, 핵심 서비스나 그 서비스와 "
            "위치 사이의 의존 시스템에 페일오버 조치가 없으면, 그 위치 하나의 화재·정전으로 핵심 서비스가 "
            "한꺼번에 멈춘다. 핵심 서비스와 무관한 시스템의 페일오버는 보호로 치지 않고, 확인된 두 위치에서 "
            "함께 도는(이중화) 서비스는 한 위치에 집중된 것으로 보지 않는다."
        ),
    )

    def check(self, ctx: CheckContext) -> Iterator[Finding]:
        critical = ctx.critical_services()
        if not critical:
            return
        critical_ids = {s.id for s in critical}
        protected = _failover_protected(ctx) & _dependency_closure(ctx.graph, critical_ids)
        for loc in ctx.graph.nodes_of(NodeKind.LOCATION):
            down = _location_failure(ctx.graph, loc.id)
            # A failover only helps if it sits on a critical service or between one and ``loc``.
            if not critical_ids <= down.keys() or protected & down.keys():
                continue
            yield self._finding(ctx, loc, critical, down)

    def _finding(
        self, ctx: CheckContext, loc: Node, critical: list[Node], down: Mapping[str, Edge | None]
    ) -> Finding:
        edge_ids = sorted({eid for s in critical for eid in _cause_chain(down, s.id)})
        spans = [span for eid in edge_ids for span in ctx.graph.edges[eid].spans]
        every = "모두 " if len(critical) > 1 else ""
        names = listed_names([s.name for s in critical])
        return finding(
            self.info,
            keys=(loc.id,),
            title=self.info.title,
            message=(
                f"핵심 서비스 {len(critical)}개({names})가 {every}'{loc.name}'에 의존합니다. "
                "이 위치에 화재·정전 같은 장애가 나면 한꺼번에 멈추는데, "
                "다른 위치로 넘기는 페일오버 조치가 없습니다."
            ),
            subject_ids=(loc.id, *(s.id for s in critical), *edge_ids),
            evidence=spans,
            fix=FixHint(
                kind=FixKind.ADD_FAILOVER,
                params={
                    "action_type": ActionType.FAILOVER.value,
                    "location_id": loc.id,
                    "service_ids": [s.id for s in critical],
                },
            ),
        )
