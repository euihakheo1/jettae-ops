"""Applying, inverting and diffing patch operations (수정안 적용·역연산·비교).

Why the graph changes only through these functions: ADR-0003 makes every graph version an
immutable, content-addressed value and every change an approved patch. For that history to be
trustworthy the change itself must be mechanical and checkable:

* **Strict application.** Each op is checked against the graph it lands on — endpoints must
  exist, :data:`~jettae.domain.EDGE_SIGNATURES` must allow the relation, edge ids must follow the
  deterministic ``{kind}:{src}->{dst}`` scheme, and a node may only be removed once no edge
  touches it. The result must pass :meth:`OpsGraph.validate_integrity`. A patch that does not fit
  the graph fails loudly with :class:`PatchError` (Korean message) instead of being half-applied.
* **Invertibility without the base graph.** :func:`apply_ops` returns the ops *as applied*:
  removals carry a snapshot of what they removed and attribute changes carry the previous value.
  :func:`invert_ops` turns those into the exact inverse, so ``apply(invert(apply(g, ops)))``
  restores ``g`` byte for byte (same :meth:`OpsGraph.content_hash`). Ops that change nothing
  (adding an element that is already there unchanged, setting an attribute to its current value)
  are left out of the returned tuple — keeping them would make the inverse delete something the
  patch never added.
* **Staleness detection.** A completed op re-applied to a graph that has moved on (its snapshot
  or previous value no longer matches) is refused, so an old patch can never silently overwrite a
  newer fact.

This module is on the judgement path (P1): it must never import an LLM client (import-linter
contract "Judgement path never calls an LLM"). It is a pure function of its inputs (P6).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pydantic import BaseModel, ConfigDict

from jettae.domain import (
    EDGE_SIGNATURES,
    AddEdge,
    AddNode,
    Attr,
    Edge,
    IntegrityError,
    Node,
    OpsGraph,
    PatchOp,
    RemoveEdge,
    RemoveNode,
    SetAttr,
    edge_id,
)

__all__ = ["GraphDiff", "PatchError", "apply_ops", "graph_diff", "invert_ops"]


class PatchError(ValueError):
    """A patch operation does not fit the graph it is applied to (message in Korean)."""


class GraphDiff(BaseModel):
    """Element ids that differ between two graphs, each tuple sorted (stable for UI and audit)."""

    model_config = ConfigDict(frozen=True)

    added_nodes: tuple[str, ...] = ()
    removed_nodes: tuple[str, ...] = ()
    changed_nodes: tuple[str, ...] = ()
    added_edges: tuple[str, ...] = ()
    removed_edges: tuple[str, ...] = ()
    changed_edges: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not (
            self.added_nodes
            or self.removed_nodes
            or self.changed_nodes
            or self.added_edges
            or self.removed_edges
            or self.changed_edges
        )


# --------------------------------------------------------------------------- application


class _Workspace:
    """Mutable copy of a graph's element maps while a patch is being applied."""

    def __init__(self, graph: OpsGraph) -> None:
        self.nodes: dict[str, Node] = dict(graph.nodes)
        self.edges: dict[str, Edge] = dict(graph.edges)

    def touching(self, node_id: str) -> list[str]:
        return sorted(e.id for e in self.edges.values() if node_id in (e.src, e.dst))


def _where(index: int, op: PatchOp) -> str:
    return f"{index}번째 연산({op.op})"


def _add_node(ws: _Workspace, op: AddNode, where: str) -> PatchOp | None:
    node = op.node
    if not node.id.startswith(f"{node.kind.value}:"):
        raise PatchError(f"{where}: 노드 id '{node.id}'가 종류({node.kind.value})와 맞지 않습니다.")
    current = ws.nodes.get(node.id)
    if current is not None:
        if current == node:
            return None
        raise PatchError(f"{where}: 같은 id의 다른 노드 '{node.id}'가 이미 있습니다.")
    ws.nodes[node.id] = node
    return op


def _add_edge(ws: _Workspace, op: AddEdge, where: str) -> PatchOp | None:
    edge = op.edge
    base = edge_id(edge.kind, edge.src, edge.dst)
    if edge.id != base and not edge.id.startswith(f"{base}#"):
        raise PatchError(f"{where}: 연결 id '{edge.id}'가 '{base}' 형식이 아닙니다.")
    missing = [n for n in (edge.src, edge.dst) if n not in ws.nodes]
    if missing:
        raise PatchError(f"{where}: 연결 '{edge.id}'의 끝 노드가 없습니다: {', '.join(missing)}")
    src_kinds, dst_kinds = EDGE_SIGNATURES[edge.kind]
    src_kind, dst_kind = ws.nodes[edge.src].kind, ws.nodes[edge.dst].kind
    if src_kind not in src_kinds or dst_kind not in dst_kinds:
        raise PatchError(
            f"{where}: {edge.kind.value} 연결은 {src_kind.value} → {dst_kind.value} 사이에 둘 수 없습니다."
        )
    current = ws.edges.get(edge.id)
    if current is not None:
        if current == edge:
            return None
        raise PatchError(f"{where}: 같은 id의 다른 연결 '{edge.id}'가 이미 있습니다.")
    ws.edges[edge.id] = edge
    return op


def _remove_edge(ws: _Workspace, op: RemoveEdge, where: str) -> PatchOp:
    current = ws.edges.get(op.edge_id)
    if current is None:
        raise PatchError(f"{where}: 지울 연결 '{op.edge_id}'가 그래프에 없습니다.")
    if op.snapshot is not None and op.snapshot != current:
        raise PatchError(f"{where}: 수정안을 만든 뒤 연결 '{op.edge_id}'의 내용이 바뀌었습니다.")
    del ws.edges[op.edge_id]
    return op.model_copy(update={"snapshot": current})


def _remove_node(ws: _Workspace, op: RemoveNode, where: str) -> PatchOp:
    current = ws.nodes.get(op.node_id)
    if current is None:
        raise PatchError(f"{where}: 지울 노드 '{op.node_id}'가 그래프에 없습니다.")
    if op.snapshot is not None and op.snapshot != current:
        raise PatchError(f"{where}: 수정안을 만든 뒤 노드 '{op.node_id}'의 내용이 바뀌었습니다.")
    linked = ws.touching(op.node_id)
    if linked:
        raise PatchError(
            f"{where}: 노드 '{op.node_id}'에 연결 {len(linked)}건이 남아 있어 지울 수 없습니다: "
            f"{', '.join(linked)}"
        )
    del ws.nodes[op.node_id]
    return op.model_copy(update={"snapshot": current})


def _updated_attrs(attrs: dict[str, Attr], key: str, value: Attr | None) -> dict[str, Attr] | None:
    """New attribute map, or ``None`` when the op would not change anything."""
    if value is None:
        if key not in attrs:
            return None
        return {k: v for k, v in attrs.items() if k != key}
    if attrs.get(key) == value:
        return None
    return {**attrs, key: value}


def _set_attr(ws: _Workspace, op: SetAttr, where: str) -> PatchOp | None:
    node = ws.nodes.get(op.target_id)
    edge = ws.edges.get(op.target_id)
    if node is not None and edge is not None:
        raise PatchError(f"{where}: '{op.target_id}'가 노드와 연결 모두의 id라 대상을 정할 수 없습니다.")
    target: Node | Edge | None = node if node is not None else edge
    if target is None:
        raise PatchError(f"{where}: 속성을 바꿀 대상 '{op.target_id}'가 그래프에 없습니다.")
    current = target.attrs.get(op.key)
    # ``previous`` is None both for "not filled yet" and for "was absent", so only a filled
    # value can prove the op is stale.
    if op.previous is not None and op.previous != current:
        raise PatchError(f"{where}: 수정안을 만든 뒤 '{op.target_id}'의 '{op.key}' 값이 바뀌었습니다.")
    attrs = _updated_attrs(target.attrs, op.key, op.value)
    if attrs is None:
        return None
    if node is not None:
        ws.nodes[node.id] = node.model_copy(update={"attrs": attrs})
    elif edge is not None:
        ws.edges[edge.id] = edge.model_copy(update={"attrs": attrs})
    return op.model_copy(update={"previous": current})


def _apply_one(ws: _Workspace, op: PatchOp, where: str) -> PatchOp | None:
    if isinstance(op, AddNode):
        return _add_node(ws, op, where)
    if isinstance(op, AddEdge):
        return _add_edge(ws, op, where)
    if isinstance(op, RemoveEdge):
        return _remove_edge(ws, op, where)
    if isinstance(op, RemoveNode):
        return _remove_node(ws, op, where)
    return _set_attr(ws, op, where)


def apply_ops(graph: OpsGraph, ops: Iterable[PatchOp]) -> tuple[OpsGraph, tuple[PatchOp, ...]]:
    """Apply ``ops`` in order; return the new graph and the ops as applied.

    The returned ops carry ``snapshot``/``previous`` so they can be inverted without the base
    graph, and leave out ops that changed nothing. Raises :class:`PatchError` if an op does not
    fit or the result fails :meth:`OpsGraph.validate_integrity`; ``graph`` itself is never
    modified (it is immutable).
    """
    ws = _Workspace(graph)
    applied: list[PatchOp] = []
    for index, op in enumerate(ops, start=1):
        done = _apply_one(ws, op, _where(index, op))
        if done is not None:
            applied.append(done)
    result = OpsGraph(
        nodes=dict(sorted(ws.nodes.items())),
        edges=dict(sorted(ws.edges.items())),
        docs=dict(graph.docs),
    )
    try:
        result.validate_integrity()
    except IntegrityError as exc:
        raise PatchError(f"수정 결과 그래프가 무결성 검사를 통과하지 못했습니다: {exc}") from exc
    return result, tuple(applied)


# --------------------------------------------------------------------------- inversion


def _inverse(op: PatchOp, where: str) -> PatchOp:
    if isinstance(op, AddNode):
        return RemoveNode(node_id=op.node.id, snapshot=op.node)
    if isinstance(op, AddEdge):
        return RemoveEdge(edge_id=op.edge.id, snapshot=op.edge)
    if isinstance(op, RemoveEdge):
        if op.snapshot is None:
            raise PatchError(
                f"{where}: 지운 연결의 스냅숏이 없어 되돌릴 수 없습니다(apply_ops가 돌려준 연산 필요)."
            )
        return AddEdge(edge=op.snapshot)
    if isinstance(op, RemoveNode):
        if op.snapshot is None:
            raise PatchError(
                f"{where}: 지운 노드의 스냅숏이 없어 되돌릴 수 없습니다(apply_ops가 돌려준 연산 필요)."
            )
        return AddNode(node=op.snapshot)
    return SetAttr(target_id=op.target_id, key=op.key, value=op.previous, previous=op.value)


def invert_ops(ops: Sequence[PatchOp]) -> tuple[PatchOp, ...]:
    """The inverse of *applied* ops (as returned by :func:`apply_ops`), in reverse order.

    Removals need their snapshot (:class:`PatchError` otherwise). A :class:`SetAttr` restores
    its ``previous`` value, where ``None`` means the attribute was absent — which is why only
    the ops returned by :func:`apply_ops` may be inverted.
    """
    return tuple(_inverse(op, _where(index, op)) for index, op in reversed(list(enumerate(ops, start=1))))


# --------------------------------------------------------------------------- diff


def graph_diff(a: OpsGraph, b: OpsGraph) -> GraphDiff:
    """What changed from ``a`` to ``b`` (element ids; documents are not compared)."""
    common_nodes = a.nodes.keys() & b.nodes.keys()
    common_edges = a.edges.keys() & b.edges.keys()
    return GraphDiff(
        added_nodes=tuple(sorted(b.nodes.keys() - a.nodes.keys())),
        removed_nodes=tuple(sorted(a.nodes.keys() - b.nodes.keys())),
        changed_nodes=tuple(sorted(k for k in common_nodes if a.nodes[k] != b.nodes[k])),
        added_edges=tuple(sorted(b.edges.keys() - a.edges.keys())),
        removed_edges=tuple(sorted(a.edges.keys() - b.edges.keys())),
        changed_edges=tuple(sorted(k for k in common_edges if a.edges[k] != b.edges[k])),
    )
