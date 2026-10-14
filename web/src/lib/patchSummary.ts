import type { Attr, Edge, EdgeKind, Node, Patch, PatchOp } from '../api/types';
import { attrLabel, displayName, formatAttr, nameFromId } from './format';
import { EDGE_KIND_LABEL, NODE_KIND_LABEL } from './labels';

/**
 * 수정안 op → 사람이 읽는 한국어 요약.
 *   add_edge escalates_to(김서연→박재현, 5분)  →  "+ 상향: 김서연 사원 → 박재현 팀장 (5분, 전화)"
 * 결정적·순수 함수 (같은 입력 → 같은 문장).
 */

/** '대기 중'으로 보는 상태 — 수정안 화면의 탭 숫자와 왼쪽 메뉴 배지가 같은 기준을 쓴다 */
// 'superseded' is closed, not pending: its base version is no longer the head, so it can never be approved.
export const PENDING_STATUSES: readonly Patch['status'][] = ['draft', 'validated', 'rejected_by_validator'];

export function isPendingPatch(p: Pick<Patch, 'status'>): boolean {
  return PENDING_STATUSES.includes(p.status);
}

export type SummarySign = '+' | '−' | '~';

export interface OpSummary {
  sign: SummarySign;
  /** "상향", "담당" … */
  label: string;
  /** "김서연 사원 → 박재현 팀장 (5분, 전화)" */
  detail: string;
  /** 전체 문장 "+ 상향: …" */
  text: string;
}

export type NodeLookup = (id: string) => Node | undefined;

export function lookupFrom(nodes: Record<string, Node> | Node[] | Map<string, Node>): NodeLookup {
  if (nodes instanceof Map) return (id) => nodes.get(id);
  if (Array.isArray(nodes)) {
    const m = new Map(nodes.map((n) => [n.id, n]));
    return (id) => m.get(id);
  }
  return (id) => nodes[id];
}

/** 같은 수정안이 추가하는 노드도 이름 해석에 쓴다. */
function withPatchNodes(ops: readonly PatchOp[], base: NodeLookup): NodeLookup {
  const added = new Map<string, Node>();
  for (const op of ops) if (op.op === 'add_node') added.set(op.node.id, op.node);
  return (id) => base(id) ?? added.get(id);
}

function name(id: string, lookup: NodeLookup): string {
  const n = lookup(id);
  return n ? displayName(n, id, { withStatus: true }) : nameFromId(id);
}

/** 사람이 보는 엣지 라벨 (op 요약용; 그래프 라벨보다 조금 길다) */
const OP_EDGE_LABEL: Partial<Record<EdgeKind, string>> = {
  fallback_to: '대체 채널',
  must_report: '신고 의무',
  reported_by: '신고 담당',
  mitigated_by: '조치',
};

function edgeLabel(kind: EdgeKind): string {
  return OP_EDGE_LABEL[kind] ?? EDGE_KIND_LABEL[kind];
}

function edgeQualifiers(edge: Pick<Edge, 'attrs'>, lookup: NodeLookup): string[] {
  const q: string[] = [];
  const after = edge.attrs.after_minutes?.value;
  if (typeof after === 'number') q.push(`${after}분`);
  const ch = edge.attrs.channel?.value;
  if (typeof ch === 'string') q.push(ch.startsWith('channel:') ? name(ch, lookup) : ch);
  return q;
}

function describeEdge(edge: Pick<Edge, 'kind' | 'src' | 'dst' | 'attrs'>, lookup: NodeLookup): string {
  const q = edgeQualifiers(edge, lookup);
  const base = `${name(edge.src, lookup)} → ${name(edge.dst, lookup)}`;
  return q.length ? `${base} (${q.join(', ')})` : base;
}

/** "edge-kind:src->dst#disc" 에서 kind/src/dst 복원 (스냅숏이 없을 때) */
export function parseEdgeId(id: string): { kind: EdgeKind; src: string; dst: string } | null {
  const hash = id.indexOf('#');
  const core = hash >= 0 ? id.slice(0, hash) : id;
  const colon = core.indexOf(':');
  const arrow = core.indexOf('->');
  if (colon < 0 || arrow < 0) return null;
  return {
    kind: core.slice(0, colon) as EdgeKind,
    src: core.slice(colon + 1, arrow),
    dst: core.slice(arrow + 2),
  };
}

function attrText(key: string, a: Attr | null | undefined): string {
  if (!a) return '미기재';
  return formatAttr(key, a.value);
}

const make = (sign: SummarySign, label: string, detail: string): OpSummary => ({
  sign,
  label,
  detail,
  text: `${sign} ${label}: ${detail}`,
});

export function summarizeOp(op: PatchOp, lookup: NodeLookup, edgeLookup?: (id: string) => Edge | undefined): OpSummary {
  switch (op.op) {
    case 'add_edge':
      return make('+', edgeLabel(op.edge.kind), describeEdge(op.edge, lookup));
    case 'remove_edge': {
      const e = op.snapshot ?? edgeLookup?.(op.edge_id);
      if (e) return make('−', edgeLabel(e.kind), describeEdge(e, lookup));
      const parsed = parseEdgeId(op.edge_id);
      if (parsed) return make('−', edgeLabel(parsed.kind), describeEdge({ ...parsed, attrs: {} }, lookup));
      return make('−', '연결', op.edge_id);
    }
    case 'add_node':
      return make('+', NODE_KIND_LABEL[op.node.kind], displayName(op.node));
    case 'remove_node': {
      const n = op.snapshot ?? lookup(op.node_id);
      return make('−', n ? NODE_KIND_LABEL[n.kind] : '노드', n ? displayName(n) : nameFromId(op.node_id));
    }
    case 'set_attr': {
      const label = attrLabel(op.key);
      const target = name(op.target_id, lookup);
      const current = op.previous ?? lookup(op.target_id)?.attrs[op.key] ?? null;
      if (op.value === null) return make('−', label, `${target} (${attrText(op.key, current)})`);
      return make('~', label, `${target} ${attrText(op.key, current)} → ${attrText(op.key, op.value)}`);
    }
  }
}

/**
 * 수정안 전체 요약. 같은 연결을 지우고 다시 잇는 쌍(퇴사자 → 대체자)은 그대로 둔다 —
 * 사람이 승인할 때 무엇이 사라지는지도 보여야 하기 때문.
 */
export function summarizePatch(
  patch: Pick<Patch, 'ops'>,
  lookup: NodeLookup,
  edgeLookup?: (id: string) => Edge | undefined,
): OpSummary[] {
  const look = withPatchNodes(patch.ops, lookup);
  return patch.ops.map((op) => summarizeOp(op, look, edgeLookup));
}

/** 메모 수정안(op 없음) — 사람 확인 필요 */
export function isMemoPatch(patch: Pick<Patch, 'ops'>): boolean {
  return patch.ops.length === 0;
}

/** 수정안에 등장하는 사람 id (명부 외 인물 검사 표시용) */
export function peopleInPatch(patch: Pick<Patch, 'ops'>): string[] {
  const ids = new Set<string>();
  const visit = (id: string): void => {
    if (id.startsWith('person:')) ids.add(id);
  };
  for (const op of patch.ops) {
    if (op.op === 'add_edge') {
      visit(op.edge.src);
      visit(op.edge.dst);
    } else if (op.op === 'add_node') visit(op.node.id);
  }
  return [...ids].sort();
}
