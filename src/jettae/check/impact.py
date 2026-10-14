"""Blast radius (영향 범위): what goes down when a node fails.

Used for the KPI "센터 1곳 장애 가정 시 영향 서비스 목록 산출 시간"; ``JT-CAT-003`` shares its
:data:`PROPAGATING_EDGES`. The question it answers is deliberately narrow and structural:

    "If these nodes fail, which services, systems and channels transitively *depend on* or
    *run in* them?"

That is *exposure*: a service running in two locations is listed when either fails (the
concentration rule asks the stricter "does it actually go down?" question itself).

Why one reverse breadth-first search over a purpose-built index: a single pass that keeps only
``DEPENDS_ON``/``RUNS_IN`` edges, keyed by destination and pre-sorted by ``(src, edge id)``,
makes the traversal O(V + E) with no per-step filtering or sorting, so 200+ services resolve in
milliseconds and ties between equally short explanations are broken the same way every run.

Why edges on HOLD (판단 보류) still propagate: an unverified dependency may be real, and
"확인 불가 ≠ 정상" (P3) means we prefer to over-report impact rather than silently drop a service
from the list.

Determinism (P6): failed ids are de-duplicated and sorted, dependents are visited in sorted
``(src, edge id)`` order, so the shortest path chosen for every impacted node is reproducible.
No LLM, no I/O.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from jettae.domain import EdgeKind, NodeKind, OpsGraph

#: Edge kinds through which failure propagates (src fails when dst fails).
PROPAGATING_EDGES: frozenset[EdgeKind] = frozenset({EdgeKind.DEPENDS_ON, EdgeKind.RUNS_IN})


class ImpactResult(BaseModel):
    """Outcome of :func:`blast_radius`.

    ``paths[x]`` is the shortest dependency chain ``(x, …, failed)`` explaining *why* ``x`` is
    impacted, and ``via_edges[x]`` holds the edge ids along that chain (for highlighting).
    Failed nodes themselves are listed only in ``failed``; ``impacted_*`` hold the *collateral*
    damage so a report can say "센터 A 장애 → 서비스 12개 중단".
    """

    model_config = ConfigDict(frozen=True)

    failed: tuple[str, ...]
    impacted_services: tuple[str, ...]
    impacted_systems: tuple[str, ...]
    impacted_channels: tuple[str, ...]
    paths: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    via_edges: dict[str, tuple[str, ...]] = Field(default_factory=dict)

    def is_down(self, nid: str) -> bool:
        """True if ``nid`` failed directly or was taken down by the failure."""
        return nid in self.paths or nid in self.failed


def _dependents_index(graph: OpsGraph) -> dict[str, list[tuple[str, str]]]:
    """dst -> sorted [(src, edge id)] for every failure-propagating edge."""
    index: dict[str, list[tuple[str, str]]] = {}
    for edge in graph.edges.values():
        if edge.kind in PROPAGATING_EDGES:
            index.setdefault(edge.dst, []).append((edge.src, edge.id))
    for dependents in index.values():
        dependents.sort()
    return index


def blast_radius(graph: OpsGraph, failed_ids: Iterable[str]) -> ImpactResult:
    """Everything that transitively ``DEPENDS_ON`` or ``RUNS_IN`` a failed node.

    A failed LOCATION therefore takes down every service/system/channel running in it, and
    everything depending on those. Dependency cycles are safe (each node is visited once).

    Raises ``ValueError`` for ids that are not in the graph: an impact list for a typo'd
    data-centre name would be empty and look like "no impact", which is exactly the kind of
    silent pass P3 forbids. A bare string is one id, not an iterable of its characters.
    """
    failed = (failed_ids,) if isinstance(failed_ids, str) else tuple(sorted(set(failed_ids)))
    missing = [nid for nid in failed if nid not in graph.nodes]
    if missing:
        raise ValueError(f"그래프에 없는 노드라 영향 범위를 계산할 수 없습니다: {', '.join(missing)}")

    index = _dependents_index(graph)
    paths: dict[str, tuple[str, ...]] = {nid: (nid,) for nid in failed}
    via: dict[str, tuple[str, ...]] = dict.fromkeys(failed, ())
    queue = deque(failed)
    while queue:
        cur = queue.popleft()
        for src, eid in index.get(cur, ()):
            if src not in paths:
                paths[src] = (src, *paths[cur])
                via[src] = (eid, *via[cur])
                queue.append(src)

    failed_set = frozenset(failed)
    impacted = sorted(nid for nid in paths if nid not in failed_set)

    def of_kind(kind: NodeKind) -> tuple[str, ...]:
        return tuple(nid for nid in impacted if graph.nodes[nid].kind is kind)

    return ImpactResult(
        failed=failed,
        impacted_services=of_kind(NodeKind.SERVICE),
        impacted_systems=of_kind(NodeKind.SYSTEM),
        impacted_channels=of_kind(NodeKind.CHANNEL),
        paths={nid: paths[nid] for nid in impacted},
        via_edges={nid: via[nid] for nid in impacted},
    )
