/**
 * 수정안 op 적용·역연산·버전 비교 (백엔드 jettae.improve.patching 의 브라우저 쪽 거울).
 * 목업/오프라인 데모에서만 실제로 그래프를 바꾸는 데 쓰고, API 모드에서는 화면 표시용 비교에만 쓴다.
 */
import type { Attr, Edge, GraphDiff, Node, OpsGraph, PatchOp } from '../api/types';

function sortRecord<T>(rec: Record<string, T>): Record<string, T> {
  return Object.fromEntries(Object.entries(rec).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)));
}

/** op를 순서대로 적용한 새 그래프와, 역연산이 가능하도록 previous/snapshot을 채운 op 목록 */
export function applyOps(graph: OpsGraph, ops: readonly PatchOp[]): { graph: OpsGraph; applied: PatchOp[] } {
  const nodes: Record<string, Node> = { ...graph.nodes };
  const edges: Record<string, Edge> = { ...graph.edges };
  const applied: PatchOp[] = [];
  for (const op of ops) {
    switch (op.op) {
      case 'add_node':
        nodes[op.node.id] = op.node;
        applied.push(op);
        break;
      case 'add_edge':
        edges[op.edge.id] = op.edge;
        applied.push(op);
        break;
      case 'remove_edge': {
        const snap = edges[op.edge_id] ?? op.snapshot;
        delete edges[op.edge_id];
        applied.push({ ...op, snapshot: snap ?? null });
        break;
      }
      case 'remove_node': {
        const snap = nodes[op.node_id] ?? op.snapshot;
        delete nodes[op.node_id];
        applied.push({ ...op, snapshot: snap ?? null });
        break;
      }
      case 'set_attr': {
        const isNode = op.target_id in nodes;
        const target = isNode ? nodes[op.target_id] : edges[op.target_id];
        if (!target) {
          applied.push(op);
          break;
        }
        const previous: Attr | null = target.attrs[op.key] ?? null;
        const attrs = { ...target.attrs };
        if (op.value === null) delete attrs[op.key];
        else attrs[op.key] = op.value;
        if (isNode) nodes[op.target_id] = { ...(target as Node), attrs: sortRecord(attrs) };
        else edges[op.target_id] = { ...(target as Edge), attrs: sortRecord(attrs) };
        applied.push({ ...op, previous });
        break;
      }
    }
  }
  return { graph: { nodes: sortRecord(nodes), edges: sortRecord(edges), docs: graph.docs }, applied };
}

/** 역연산 (되돌리기 = 새 버전) */
export function invertOps(ops: readonly PatchOp[]): PatchOp[] {
  const inv: PatchOp[] = [];
  for (const op of [...ops].reverse()) {
    switch (op.op) {
      case 'add_node':
        inv.push({ op: 'remove_node', node_id: op.node.id, snapshot: op.node });
        break;
      case 'add_edge':
        inv.push({ op: 'remove_edge', edge_id: op.edge.id, snapshot: op.edge });
        break;
      case 'remove_edge':
        if (op.snapshot) inv.push({ op: 'add_edge', edge: op.snapshot });
        break;
      case 'remove_node':
        if (op.snapshot) inv.push({ op: 'add_node', node: op.snapshot });
        break;
      case 'set_attr':
        inv.push({ op: 'set_attr', target_id: op.target_id, key: op.key, value: op.previous, previous: op.value });
        break;
    }
  }
  return inv;
}

function attrsEqual(a: Attr | undefined, b: Attr | undefined): boolean {
  return JSON.stringify(a?.value ?? null) === JSON.stringify(b?.value ?? null);
}

function attrChanges(before: Record<string, Attr>, after: Record<string, Attr>): GraphDiff['changed_nodes'][number]['attrs'] {
  const keys = [...new Set([...Object.keys(before), ...Object.keys(after)])].sort();
  return keys
    .filter((k) => !attrsEqual(before[k], after[k]))
    .map((k) => ({ key: k, before: before[k] ?? null, after: after[k] ?? null }));
}

/** 두 그래프 버전 비교 (노드·엣지 추가/삭제/속성 변경) */
export function graphDiff(a: OpsGraph, b: OpsGraph, aNum: number, bNum: number): GraphDiff {
  const diff: GraphDiff = {
    a: aNum,
    b: bNum,
    added_nodes: [],
    removed_nodes: [],
    changed_nodes: [],
    added_edges: [],
    removed_edges: [],
    changed_edges: [],
  };
  const nodeIds = [...new Set([...Object.keys(a.nodes), ...Object.keys(b.nodes)])].sort();
  for (const id of nodeIds) {
    const x = a.nodes[id];
    const y = b.nodes[id];
    if (!x && y) diff.added_nodes.push(y);
    else if (x && !y) diff.removed_nodes.push(x);
    else if (x && y) {
      const attrs = attrChanges(x.attrs, y.attrs);
      if (attrs.length || x.name !== y.name || x.verification !== y.verification)
        diff.changed_nodes.push({ id, before: x, after: y, attrs });
    }
  }
  const edgeIds = [...new Set([...Object.keys(a.edges), ...Object.keys(b.edges)])].sort();
  for (const id of edgeIds) {
    const x = a.edges[id];
    const y = b.edges[id];
    if (!x && y) diff.added_edges.push(y);
    else if (x && !y) diff.removed_edges.push(x);
    else if (x && y) {
      const attrs = attrChanges(x.attrs, y.attrs);
      if (attrs.length || x.verification !== y.verification) diff.changed_edges.push({ id, before: x, after: y, attrs });
    }
  }
  return diff;
}

/** 키를 정렬한 JSON (그래프 해시용) */
export function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  const entries = Object.entries(value as Record<string, unknown>)
    .filter(([, v]) => v !== undefined)
    .sort(([x], [y]) => (x < y ? -1 : x > y ? 1 : 0));
  return `{${entries.map(([k, v]) => `${JSON.stringify(k)}:${canonicalJson(v)}`).join(',')}}`;
}

export async function sha256Hex(text: string): Promise<string> {
  const data = new TextEncoder().encode(text);
  const subtle = globalThis.crypto?.subtle;
  if (subtle) {
    const buf = await subtle.digest('SHA-256', data);
    return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('');
  }
  // 보안 컨텍스트가 아닌 곳(http 원격 접속)용 대체 해시 — 목업 표시용
  let h1 = 0x811c9dc5;
  let h2 = 0x01000193;
  for (const b of data) {
    h1 = Math.imul(h1 ^ b, 0x01000193) >>> 0;
    h2 = Math.imul(h2 + b, 0x85ebca6b) >>> 0;
  }
  return (h1.toString(16).padStart(8, '0') + h2.toString(16).padStart(8, '0')).repeat(4);
}
