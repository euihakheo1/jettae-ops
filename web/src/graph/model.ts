/**
 * 운영 그래프 뷰 모델 (순수 함수 · 단위 테스트 대상).
 *
 * GraphView(+끊김 목록, 준비도) → React Flow 노드/엣지.
 *  - 서비스 집중(focus) 부분 그래프 / 전체 보기
 *  - 끊김(finding)에서 '빠진 필수 요소'를 유령(ghost) 노드로 만든다 (예: JT-PATH-020 → '승인자 없음')
 *  - 퇴사자·휴직자·명부 미확인·판단 보류(HOLD) 표시 플래그
 *  - 끊김 subject 엣지는 빨강(치명·높음, 애니메이션) / 주황(보통)
 *  - 모든 출력은 id로 정렬 → 같은 입력이면 같은 순서(결정적 레이아웃의 전제)
 */
import type { Edge as FlowEdge, Node as FlowNode } from '@xyflow/react';
import type {
  ActionType,
  ChannelMedium,
  EdgeKind,
  Finding,
  GraphView,
  GraphViewEdge,
  GraphViewNode,
  NodeKind,
  OpsGraph,
  PersonStatus,
  ReadinessReport,
  Severity,
  SourceKind,
} from '../api/types';
import { attrNumber, attrString, formatMinutes, personStatus } from '../lib/format';
import type { EdgeRoute } from './layout';
import { MEDIUM_LABEL, SEVERITY_RANK } from '../lib/labels';

// --------------------------------------------------------------------------- types

export type FocusMode = { mode: 'service'; serviceId: string } | { mode: 'all' };

export type BadgeTone = 'danger' | 'warn' | 'muted' | 'info' | 'ok';

export interface NodeBadge {
  text: string;
  tone: BadgeTone;
}

export type OpsNodeKind = NodeKind | 'ghost';

/** React Flow 노드 data (type alias: Record<string, unknown> 호환) */
export type OpsNodeData = {
  graphId: string;
  kind: OpsNodeKind;
  /** 유령 노드가 흉내 내는 종류 (승인자 없음 → person) */
  ghostOf?: NodeKind;
  ghostRule?: string;
  label: string;
  sublabel?: string;
  badges: NodeBadge[];
  personStatus?: PersonStatus;
  departed: boolean;
  onLeave: boolean;
  unknownPerson: boolean;
  hold: boolean;
  holdReason?: string;
  severity?: Severity;
  findingIds: string[];
  tier?: number;
  score?: number;
  medium?: ChannelMedium;
  actionType?: ActionType;
  deadlineMinutes?: number;
  recipient?: string;
  isFocus: boolean;
  /** 리허설 경로 강조 */
  highlighted?: boolean;
  /** 강조 방식: 리허설 경로(브랜드색) / 선택한 끊김(빨강 유지) */
  highlightStyle?: HighlightStyle;
  current?: boolean;
  failed?: boolean;
  dimmed?: boolean;
  selected?: boolean;
};

export type OpsEdgeKind = EdgeKind | 'ghost';

export type HighlightStyle = 'path' | 'finding';

export type EdgeTone = 'broken' | 'warn' | 'normal' | 'faded' | 'ghost';

export type OpsEdgeData = {
  kind: OpsEdgeKind;
  label: string;
  tone: EdgeTone;
  severity?: Severity;
  findingIds: string[];
  /** 레이아웃·표시를 위해 원래 방향(src→dst)을 뒤집었는지 (detected_by) */
  reversed: boolean;
  hold: boolean;
  highlighted?: boolean;
  highlightStyle?: HighlightStyle;
  current?: boolean;
  dimmed?: boolean;
  /** 라벨 위치(0~1, 곡선 위 비율). 한 점에서 퍼지는 연결은 도착 쪽, 모이는 연결은 출발 쪽으로 옮겨 겹침을 줄인다 */
  labelT?: number;
  /** ELK가 그린 직교 경로 (없으면 베지어 곡선) */
  route?: EdgeRoute;
};

export type OpsFlowNode = FlowNode<OpsNodeData, 'ops'>;
export type OpsFlowEdge = FlowEdge<OpsEdgeData, 'ops'>;

export interface GraphModel {
  nodes: OpsFlowNode[];
  edges: OpsFlowEdge[];
  focusId: string | null;
  /** ELK 레이어 제약: 왼쪽 끝(감지)·오른쪽 끝(채널) */
  layer: Record<string, 'FIRST' | 'LAST'>;
}

export interface ModelInput {
  view: GraphView;
  findings: readonly Finding[];
  readiness?: ReadinessReport | null;
}

export interface ModelOptions {
  focus: FocusMode;
  /** 빠진 요소(유령 노드) 표시 (기본 true) */
  ghosts?: boolean;
  /** 전체 보기에서 구조 엣지(소속·역할·위치) 숨김 (기본 false) */
  hideStructural?: boolean;
}

// --------------------------------------------------------------------------- sizes

export const NODE_SIZE: Record<OpsNodeKind, { width: number; height: number }> = {
  service: { width: 244, height: 100 },
  person: { width: 224, height: 78 },
  role: { width: 184, height: 58 },
  team: { width: 184, height: 58 },
  channel: { width: 168, height: 52 },
  trigger: { width: 256, height: 78 },
  action: { width: 228, height: 68 },
  obligation: { width: 240, height: 74 },
  system: { width: 196, height: 58 },
  location: { width: 196, height: 58 },
  ghost: { width: 252, height: 70 },
};

// --------------------------------------------------------------------------- index

export interface GraphIndex {
  nodes: Map<string, GraphViewNode>;
  edges: Map<string, GraphViewEdge>;
  out: Map<string, GraphViewEdge[]>;
  in: Map<string, GraphViewEdge[]>;
}

export function buildIndex(view: Pick<GraphView, 'nodes' | 'edges'>): GraphIndex {
  const nodes = new Map<string, GraphViewNode>();
  const edges = new Map<string, GraphViewEdge>();
  const out = new Map<string, GraphViewEdge[]>();
  const inn = new Map<string, GraphViewEdge[]>();
  for (const n of [...view.nodes].sort(byId)) nodes.set(n.id, n);
  for (const e of [...view.edges].sort(byId)) {
    if (!nodes.has(e.src) || !nodes.has(e.dst)) continue;
    edges.set(e.id, e);
    pushMap(out, e.src, e);
    pushMap(inn, e.dst, e);
  }
  return { nodes, edges, out, in: inn };
}

function pushMap<K, V>(m: Map<K, V[]>, k: K, v: V): void {
  const arr = m.get(k);
  if (arr) arr.push(v);
  else m.set(k, [v]);
}

function byId<T extends { id: string }>(a: T, b: T): number {
  return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
}

function outOf(ix: GraphIndex, id: string, ...kinds: EdgeKind[]): GraphViewEdge[] {
  return (ix.out.get(id) ?? []).filter((e) => kinds.length === 0 || kinds.includes(e.kind));
}

function inOf(ix: GraphIndex, id: string, ...kinds: EdgeKind[]): GraphViewEdge[] {
  return (ix.in.get(id) ?? []).filter((e) => kinds.length === 0 || kinds.includes(e.kind));
}

// --------------------------------------------------------------------------- OpsGraph → GraphView

/** 백엔드가 OpsGraph 원본을 줄 때(또는 목업) GraphView DTO로 바꾼다. */
export function graphViewFromOpsGraph(
  graph: OpsGraph,
  findings: readonly Finding[],
  version: number,
  graphHash: string,
): GraphView {
  const spansOf = (prov: { spans: GraphViewNode['spans'] }[]): GraphViewNode['spans'] => {
    const seen = new Set<string>();
    const out: GraphViewNode['spans'] = [];
    for (const p of prov)
      for (const s of p.spans) {
        const key = `${s.doc_id}:${s.start}:${s.end}`;
        if (seen.has(key)) continue;
        seen.add(key);
        out.push(s);
      }
    return out;
  };
  const sourcesOf = (prov: { source: SourceKind }[]): SourceKind[] => [...new Set(prov.map((p) => p.source))].sort();
  const nodes: GraphViewNode[] = Object.values(graph.nodes)
    .sort(byId)
    .map((n) => ({
      id: n.id,
      kind: n.kind,
      name: n.name,
      attrs: n.attrs,
      verification: n.verification,
      hold_reason: n.hold_reason,
      spans: spansOf(n.prov),
      sources: sourcesOf(n.prov),
    }));
  const edges: GraphViewEdge[] = Object.values(graph.edges)
    .sort(byId)
    .map((e) => ({
      id: e.id,
      kind: e.kind,
      src: e.src,
      dst: e.dst,
      attrs: e.attrs,
      verification: e.verification,
      hold_reason: e.hold_reason,
      spans: spansOf(e.prov),
      sources: sourcesOf(e.prov),
    }));
  return {
    version,
    graph_hash: graphHash,
    nodes,
    edges,
    findings_by_element: findingsByElement(findings),
    docs: Object.values(graph.docs).sort((a, b) => (a.doc_id < b.doc_id ? -1 : 1)),
  };
}

export function findingsByElement(findings: readonly Finding[]): Record<string, string[]> {
  const map: Record<string, string[]> = {};
  for (const f of findings) {
    for (const id of f.subject_ids) {
      (map[id] ??= []).push(f.id);
    }
  }
  for (const k of Object.keys(map)) map[k] = [...new Set(map[k])].sort();
  return map;
}

// --------------------------------------------------------------------------- focus

/** 기본 집중 서비스: 가장 중요한 등급(숫자가 가장 작은 tier) 안에서 점수가 가장 낮은 서비스. 동점이면 id 순. */
export function defaultFocusService(readiness: ReadinessReport | null | undefined, view?: Pick<GraphView, 'nodes'>): string | null {
  const services = readiness?.services ?? [];
  if (services.length > 0) {
    const topTier = Math.min(...services.map((s) => s.tier));
    const pool = services.filter((s) => s.tier === topTier);
    const sorted = [...pool].sort((a, b) => a.score - b.score || (a.service_id < b.service_id ? -1 : 1));
    return sorted[0]?.service_id ?? null;
  }
  const first = view?.nodes.filter((n) => n.kind === 'service').sort(byId)[0];
  return first?.id ?? null;
}

const RESPONDER_KINDS: ReadonlySet<NodeKind> = new Set<NodeKind>(['person', 'role', 'team']);

/**
 * 서비스 집중 부분 그래프:
 * 서비스 + 감지 신호(+호출 대상) + 담당 + 역할→사람 해석 + 상향 사슬 + 조치(+승인자·실행자) + 신고 의무(+신고 담당)
 * + 등장한 사람·역할·팀의 연락 채널 + 대체 채널.
 */
export function focusSubgraph(ix: GraphIndex, serviceId: string): { nodeIds: Set<string>; edgeIds: Set<string> } {
  const nodeIds = new Set<string>();
  const edgeIds = new Set<string>();
  if (!ix.nodes.has(serviceId)) return { nodeIds, edgeIds };
  nodeIds.add(serviceId);
  const responders: string[] = [];
  const take = (e: GraphViewEdge, other: string, responder = false): void => {
    edgeIds.add(e.id);
    nodeIds.add(other);
    if (responder && RESPONDER_KINDS.has(ix.nodes.get(other)!.kind)) responders.push(other);
  };

  for (const e of outOf(ix, serviceId, 'detected_by')) {
    take(e, e.dst);
    for (const n of outOf(ix, e.dst, 'notifies')) take(n, n.dst, true);
  }
  for (const e of outOf(ix, serviceId, 'owned_by')) take(e, e.dst, true);
  for (const e of outOf(ix, serviceId, 'mitigated_by')) {
    take(e, e.dst);
    for (const a of outOf(ix, e.dst, 'approved_by', 'executed_by')) take(a, a.dst, true);
  }
  for (const e of outOf(ix, serviceId, 'must_report')) {
    take(e, e.dst);
    for (const r of outOf(ix, e.dst, 'reported_by')) take(r, r.dst, true);
  }

  // 상향 사슬 + 역할 해석 (BFS, 결정적 순서)
  const queue = [...responders];
  const seen = new Set<string>();
  let guard = 0;
  while (queue.length > 0 && guard < 200) {
    guard += 1;
    const id = queue.shift()!;
    if (seen.has(id)) continue;
    seen.add(id);
    const node = ix.nodes.get(id);
    if (!node) continue;
    for (const e of outOf(ix, id, 'escalates_to')) {
      take(e, e.dst);
      queue.push(e.dst);
    }
    if (node.kind === 'role') {
      // 역할을 맡은 사람 (명부·당직표로 해석)
      for (const e of inOf(ix, id, 'holds_role')) {
        take(e, e.src);
        queue.push(e.src);
      }
    } else if (node.kind === 'person') {
      // 상향이 이어지는 역할만 따라간다 (예: 박재현 → 팀장 → 한서윤 CTO)
      for (const e of outOf(ix, id, 'holds_role')) {
        if (outOf(ix, e.dst, 'escalates_to').length > 0) {
          take(e, e.dst);
          queue.push(e.dst);
        }
      }
    }
  }

  // 연락 채널 + 대체 채널
  for (const id of [...nodeIds].sort()) {
    const node = ix.nodes.get(id);
    if (!node || !RESPONDER_KINDS.has(node.kind)) continue;
    for (const e of outOf(ix, id, 'reachable_via')) take(e, e.dst);
  }
  for (const id of [...nodeIds].sort()) {
    if (ix.nodes.get(id)?.kind !== 'channel') continue;
    for (const e of outOf(ix, id, 'fallback_to')) take(e, e.dst);
  }
  return { nodeIds, edgeIds };
}

// --------------------------------------------------------------------------- ghosts

interface GhostSpec {
  label: string;
  sublabel?: string;
  ghostOf: NodeKind;
  edgeLabel: string;
  anchor: 'service' | 'action' | 'from' | 'owner' | 'services' | 'obligation';
  /** 유령 → 앵커 방향(감지 신호처럼 왼쪽에 놓일 때) */
  incoming?: boolean;
}

export const GHOST_RULES: Record<string, GhostSpec> = {
  'JT-PATH-001': { label: '감지 신호 없음', sublabel: '알람·모니터 없음', ghostOf: 'trigger', edgeLabel: '감지', anchor: 'service', incoming: true },
  'JT-PATH-010': { label: '담당자 없음', sublabel: '활동 중인 담당 필요', ghostOf: 'person', edgeLabel: '담당', anchor: 'service' },
  'JT-PATH-011': { label: '연락 수단 없음', ghostOf: 'channel', edgeLabel: '연락', anchor: 'owner' },
  'JT-ESC-001': { label: '상향 경로 없음', sublabel: '다음 사람 미지정', ghostOf: 'person', edgeLabel: '상향', anchor: 'from' },
  'JT-ESC-003': { label: '상향 끝 응답자 없음', ghostOf: 'person', edgeLabel: '상향', anchor: 'from' },
  'JT-PATH-020': { label: '승인자 없음', sublabel: '승인 경로 미기재', ghostOf: 'person', edgeLabel: '승인', anchor: 'action' },
  'JT-PATH-041': { label: '실행자 없음', sublabel: '실행 담당 미지정', ghostOf: 'person', edgeLabel: '실행', anchor: 'action' },
  'JT-PATH-030': { label: '대체 채널 없음', sublabel: '연락 수단 2개 미만', ghostOf: 'channel', edgeLabel: '대체', anchor: 'owner' },
  'JT-PATH-040': { label: '롤백 조치 없음', sublabel: '되돌리기 조치 미기재', ghostOf: 'action', edgeLabel: '조치', anchor: 'service' },
  'JT-KR-001': { label: '침해사고 신고 단계 없음', sublabel: '24시간 · KISA', ghostOf: 'obligation', edgeLabel: '신고', anchor: 'services' },
  'JT-KR-003': { label: '개인정보 유출 신고 없음', sublabel: '72시간 · 개인정보위', ghostOf: 'obligation', edgeLabel: '신고', anchor: 'services' },
  'JT-KR-002': { label: '신고 담당 없음', ghostOf: 'person', edgeLabel: '신고 담당', anchor: 'obligation' },
  'JT-KR-004': { label: '신고 담당 없음', ghostOf: 'person', edgeLabel: '신고 담당', anchor: 'obligation' },
};

function paramStr(f: Finding, key: string): string | undefined {
  const v = f.fix?.params[key];
  return typeof v === 'string' ? v : undefined;
}

function paramStrs(f: Finding, key: string): string[] {
  const v = f.fix?.params[key];
  return Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string') : [];
}

function ghostAnchors(f: Finding, spec: GhostSpec, visible: Set<string>): string[] {
  const inView = (ids: (string | undefined | null)[]): string[] =>
    [...new Set(ids.filter((x): x is string => typeof x === 'string' && visible.has(x)))].sort();
  switch (spec.anchor) {
    case 'service':
      return inView([f.service_id]);
    case 'action':
      return inView([paramStr(f, 'action_id') ?? f.subject_ids.find((s) => s.startsWith('action:'))]);
    case 'from': {
      const from = paramStrs(f, 'from_ids');
      const single = paramStr(f, 'from_id');
      const ids = from.length ? from : single ? [single] : f.subject_ids.filter((s) => s.startsWith('person:') || s.startsWith('role:'));
      const anchors = inView(ids);
      return anchors.length ? anchors : inView([f.service_id]);
    }
    case 'owner': {
      const owners = inView(paramStrs(f, 'owner_ids'));
      return owners.length ? owners.slice(0, 1) : inView([f.service_id]);
    }
    case 'services':
      return inView(f.subject_ids.filter((s) => s.startsWith('service:')));
    case 'obligation':
      return inView([paramStr(f, 'obligation_id') ?? f.subject_ids.find((s) => s.startsWith('obligation:'))]);
  }
}

interface Ghost {
  id: string;
  spec: GhostSpec;
  rule: string;
  findingIds: string[];
  severity: Severity;
  anchors: string[];
}

/** 끊김 → 유령 노드 (보이는 노드에 붙일 수 있는 것만). 같은 id는 합친다. */
export function deriveGhosts(findings: readonly Finding[], visible: Set<string>): Ghost[] {
  const ghosts = new Map<string, Ghost>();
  const sorted = [...findings].sort((a, b) => (a.id < b.id ? -1 : 1));
  for (const f of sorted) {
    const spec = GHOST_RULES[f.rule_id];
    if (!spec) continue;
    const anchors = ghostAnchors(f, spec, visible);
    if (anchors.length === 0) continue;
    const groups = spec.anchor === 'services' ? [anchors] : anchors.map((a) => [a]);
    for (const group of groups) {
      const id = spec.anchor === 'services' ? `ghost:${f.rule_id}` : `ghost:${f.rule_id}:${group[0]}`;
      const cur = ghosts.get(id);
      if (cur) {
        cur.findingIds = [...new Set([...cur.findingIds, f.id])].sort();
        cur.anchors = [...new Set([...cur.anchors, ...group])].sort();
        if (SEVERITY_RANK[f.severity] < SEVERITY_RANK[cur.severity]) cur.severity = f.severity;
      } else {
        ghosts.set(id, { id, spec, rule: f.rule_id, findingIds: [f.id], severity: f.severity, anchors: [...group] });
      }
    }
  }
  return [...ghosts.values()].sort(byId);
}

// --------------------------------------------------------------------------- labels

const STRUCTURAL: ReadonlySet<EdgeKind> = new Set<EdgeKind>(['member_of', 'holds_role', 'runs_in', 'depends_on']);

export function edgeLabel(e: Pick<GraphViewEdge, 'kind' | 'attrs'>, nodes?: Map<string, GraphViewNode>): string {
  switch (e.kind) {
    case 'owned_by':
      return '담당';
    case 'notifies': {
      const ch = e.attrs.channel?.value;
      const chName = typeof ch === 'string' ? channelName(ch, nodes) : null;
      return chName ? `호출 · ${chName}` : '호출';
    }
    case 'escalates_to': {
      const after = e.attrs.after_minutes?.value;
      return typeof after === 'number' ? `상향 ${formatMinutes(after)}` : '상향';
    }
    case 'approved_by':
      return '승인';
    case 'executed_by':
      return '실행';
    case 'reachable_via':
      return '연락';
    case 'fallback_to':
      return '대체';
    case 'depends_on':
      return '의존';
    case 'must_report':
      return '신고';
    case 'reported_by':
      return '신고 담당';
    case 'mitigated_by':
      return '조치';
    case 'detected_by':
      return '감지';
    case 'holds_role':
      return '역할';
    case 'member_of':
      return '소속';
    case 'runs_in':
      return '위치';
  }
}

function channelName(value: string, nodes?: Map<string, GraphViewNode>): string {
  const node = nodes?.get(value);
  const medium = node ? attrString(node, 'medium') : value.replace(/^channel:/, '');
  if (medium && medium in MEDIUM_LABEL) return MEDIUM_LABEL[medium as ChannelMedium];
  return node?.name ?? value.replace(/^channel:/, '');
}

// --------------------------------------------------------------------------- severity helpers

function maxSeverity(ids: readonly string[], byIdMap: Map<string, Finding>): Severity | undefined {
  let best: Severity | undefined;
  for (const id of ids) {
    const f = byIdMap.get(id);
    if (!f) continue;
    if (best === undefined || SEVERITY_RANK[f.severity] < SEVERITY_RANK[best]) best = f.severity;
  }
  return best;
}

function toneFor(sev: Severity | undefined): EdgeTone {
  if (sev === 'critical' || sev === 'high') return 'broken';
  if (sev === 'medium') return 'warn';
  return 'normal';
}

/** element id → 관련 finding id (DTO의 findings_by_element ∪ subject_ids) */
function findingIndex(view: GraphView, findings: readonly Finding[]): Map<string, string[]> {
  const m = new Map<string, Set<string>>();
  const add = (k: string, v: string): void => {
    const s = m.get(k);
    if (s) s.add(v);
    else m.set(k, new Set([v]));
  };
  for (const [k, ids] of Object.entries(view.findings_by_element ?? {})) for (const id of ids) add(k, id);
  for (const f of findings) for (const s of f.subject_ids) add(s, f.id);
  const out = new Map<string, string[]>();
  for (const [k, s] of m) out.set(k, [...s].sort());
  return out;
}

// --------------------------------------------------------------------------- node data

function teamOf(ix: GraphIndex, personId: string): string | undefined {
  const e = outOf(ix, personId, 'member_of')[0];
  return e ? ix.nodes.get(e.dst)?.name : undefined;
}

function nodeData(
  n: GraphViewNode,
  ix: GraphIndex,
  fIndex: Map<string, string[]>,
  fById: Map<string, Finding>,
  readiness: ReadinessReport | null | undefined,
  focusId: string | null,
): OpsNodeData {
  const findingIds = fIndex.get(n.id) ?? [];
  const severity = maxSeverity(findingIds, fById);
  const badges: NodeBadge[] = [];
  const hold = n.verification === 'hold';
  const data: OpsNodeData = {
    graphId: n.id,
    kind: n.kind,
    label: n.name,
    badges,
    departed: false,
    onLeave: false,
    unknownPerson: false,
    hold,
    findingIds,
    isFocus: n.id === focusId,
    ...(severity ? { severity } : {}),
    ...(hold && n.hold_reason ? { holdReason: n.hold_reason } : {}),
  };
  switch (n.kind) {
    case 'person': {
      const st = personStatus(n) ?? 'unknown';
      data.personStatus = st;
      data.departed = st === 'departed';
      data.onLeave = st === 'on_leave';
      data.unknownPerson = st === 'unknown';
      const title = attrString(n, 'title');
      const team = teamOf(ix, n.id);
      const sub = [title, team].filter(Boolean).join(' · ');
      if (sub) data.sublabel = sub;
      if (st === 'departed') badges.push({ text: '퇴사', tone: 'danger' });
      else if (st === 'on_leave') badges.push({ text: '휴직', tone: 'warn' });
      else if (st === 'unknown') badges.push({ text: '명부 미확인', tone: 'muted' });
      break;
    }
    case 'service': {
      const tier = attrNumber(n, 'tier');
      if (tier !== undefined) data.tier = tier;
      const sr = readiness?.services.find((s) => s.service_id === n.id);
      if (sr) {
        data.score = sr.score;
        data.tier = sr.tier;
      }
      break;
    }
    case 'channel': {
      const medium = attrString(n, 'medium');
      if (medium && medium in MEDIUM_LABEL) {
        data.medium = medium as ChannelMedium;
        data.label = MEDIUM_LABEL[medium as ChannelMedium];
      } else data.medium = 'other';
      break;
    }
    case 'trigger': {
      const cond = attrString(n, 'condition');
      if (cond) data.label = cond;
      const delay = attrNumber(n, 'delay_minutes');
      data.sublabel = delay !== undefined ? `감지 지연 ${formatMinutes(delay)}` : '감지 지연 미기재';
      break;
    }
    case 'action': {
      const t = attrString(n, 'action_type');
      if (t) data.actionType = t as ActionType;
      const req = n.attrs.requires_approval?.value;
      if (req === true) data.sublabel = '승인 필요';
      break;
    }
    case 'obligation': {
      const deadline = attrNumber(n, 'deadline_minutes');
      const recipient = attrString(n, 'recipient');
      if (deadline !== undefined) data.deadlineMinutes = deadline;
      if (recipient) data.recipient = recipient;
      data.sublabel = [deadline !== undefined ? formatMinutes(deadline) : null, recipient].filter(Boolean).join(' · ');
      break;
    }
    default:
      break;
  }
  if (hold) badges.push({ text: '판단 보류', tone: 'warn' });
  return data;
}

// --------------------------------------------------------------------------- build

export function buildGraphModel(input: ModelInput, options: ModelOptions): GraphModel {
  const { view, findings, readiness } = input;
  const ix = buildIndex(view);
  const fById = new Map(findings.map((f) => [f.id, f]));
  const fIndex = findingIndex(view, findings);
  const ghostsOn = options.ghosts ?? true;

  let nodeIds: Set<string>;
  let edgeIds: Set<string>;
  let focusId: string | null = null;
  if (options.focus.mode === 'service' && ix.nodes.has(options.focus.serviceId)) {
    focusId = options.focus.serviceId;
    ({ nodeIds, edgeIds } = focusSubgraph(ix, focusId));
  } else {
    const visibleEdges = [...ix.edges.values()].filter((e) => !(options.hideStructural && STRUCTURAL.has(e.kind)));
    edgeIds = new Set(visibleEdges.map((e) => e.id));
    if (options.hideStructural) {
      // 구조 연결을 숨기면 떠 있는 노드(팀·위치·시스템 등)도 뺀다. 서비스는 항상 남긴다.
      const touched = new Set(visibleEdges.flatMap((e) => [e.src, e.dst]));
      nodeIds = new Set([...ix.nodes.values()].filter((n) => n.kind === 'service' || touched.has(n.id)).map((n) => n.id));
    } else nodeIds = new Set(ix.nodes.keys());
  }

  const layer: Record<string, 'FIRST' | 'LAST'> = {};
  const nodes: OpsFlowNode[] = [];
  for (const id of [...nodeIds].sort()) {
    const n = ix.nodes.get(id)!;
    const data = nodeData(n, ix, fIndex, fById, readiness, focusId);
    if (n.kind === 'trigger') layer[id] = 'FIRST';
    if (n.kind === 'channel' && focusId) layer[id] = 'LAST';
    nodes.push({
      id,
      type: 'ops',
      position: { x: 0, y: 0 },
      data,
      width: NODE_SIZE[n.kind].width,
      height: NODE_SIZE[n.kind].height,
    });
  }

  const edges: OpsFlowEdge[] = [];
  for (const id of [...edgeIds].sort()) {
    const e = ix.edges.get(id)!;
    const fids = fIndex.get(e.id) ?? [];
    const severity = maxSeverity(fids, fById);
    const reversed = e.kind === 'detected_by';
    let tone = toneFor(severity);
    if (tone === 'normal' && STRUCTURAL.has(e.kind) && !focusId) tone = 'faded';
    const hold = e.verification === 'hold';
    edges.push({
      id,
      type: 'ops',
      source: reversed ? e.dst : e.src,
      target: reversed ? e.src : e.dst,
      animated: tone === 'broken',
      data: {
        kind: e.kind,
        label: edgeLabel(e, ix.nodes),
        tone,
        findingIds: fids,
        reversed,
        hold,
        ...(severity ? { severity } : {}),
      },
    });
  }

  if (ghostsOn) {
    const relevant = focusId
      ? findings.filter((f) => f.service_id === focusId || f.subject_ids.includes(focusId!))
      : findings;
    for (const g of deriveGhosts(relevant, nodeIds)) {
      if (g.spec.incoming) layer[g.id] = 'FIRST';
      else if (g.spec.ghostOf === 'channel' && focusId) layer[g.id] = 'LAST';
      nodes.push({
        id: g.id,
        type: 'ops',
        position: { x: 0, y: 0 },
        width: NODE_SIZE.ghost.width,
        height: NODE_SIZE.ghost.height,
        data: {
          graphId: g.id,
          kind: 'ghost',
          ghostOf: g.spec.ghostOf,
          ghostRule: g.rule,
          label: g.spec.label,
          ...(g.spec.sublabel ? { sublabel: g.spec.sublabel } : {}),
          badges: [{ text: '빠짐', tone: 'danger' }],
          departed: false,
          onLeave: false,
          unknownPerson: false,
          hold: false,
          severity: g.severity,
          findingIds: g.findingIds,
          isFocus: false,
        },
      });
      for (const a of g.anchors) {
        const incoming = g.spec.incoming === true;
        edges.push({
          id: `ghost-edge:${g.id}:${a}`,
          type: 'ops',
          source: incoming ? g.id : a,
          target: incoming ? a : g.id,
          animated: true,
          data: {
            kind: 'ghost',
            label: g.spec.edgeLabel,
            tone: 'ghost',
            severity: g.severity,
            findingIds: g.findingIds,
            reversed: incoming,
            hold: false,
          },
        });
      }
    }
  }

  nodes.sort(byId);
  edges.sort(byId);
  return { nodes, edges, focusId, layer };
}

// --------------------------------------------------------------------------- highlight

export interface HighlightOptions {
  /** 강조할 node/edge id (리허설 path_ids 또는 finding subject_ids) */
  ids: readonly string[];
  /** 현재 단계 id */
  currentId?: string | null;
  failedId?: string | null;
  /** 강조 밖 요소를 흐리게 */
  dimOthers?: boolean;
  /** 'path'(기본): 리허설 경로를 브랜드색으로 / 'finding': 끊김 색(빨강)을 유지한 채 굵고 밝게 */
  style?: HighlightStyle;
  /** 흐리게 하지 않을 요소 (예: 집중 중인 서비스) */
  keepIds?: readonly string[];
}

/** 리허설 경로·선택 강조 (원본을 바꾸지 않고 새 배열을 돌려준다). */
export function applyHighlight(model: GraphModel, opts: HighlightOptions): GraphModel {
  const ids = new Set(opts.ids);
  const style: HighlightStyle = opts.style ?? 'path';
  const keep = new Set(opts.keepIds ?? []);
  const nodes = model.nodes.map((n) => {
    const hit = ids.has(n.id);
    return {
      ...n,
      data: {
        ...n.data,
        highlighted: hit,
        highlightStyle: style,
        current: opts.currentId === n.id,
        failed: opts.failedId === n.id,
        dimmed: opts.dimOthers === true && !hit && opts.failedId !== n.id && !keep.has(n.id),
      },
    };
  });
  const edges = model.edges.map((e) => {
    const hit = ids.has(e.id);
    return {
      ...e,
      animated: hit ? true : e.animated,
      data: {
        ...(e.data as OpsEdgeData),
        highlighted: hit,
        highlightStyle: style,
        current: opts.currentId === e.id,
        dimmed: opts.dimOthers === true && !hit,
      },
    };
  });
  return { ...model, nodes, edges };
}

/** 리허설 경로만으로 만든 작은 그래프 (path_ids + 실패 지점) */
export function pathSubgraphModel(input: ModelInput, pathIds: readonly string[], failedId?: string | null): GraphModel {
  const ix = buildIndex(input.view);
  const keepNodes = new Set<string>();
  const keepEdges = new Set<string>();
  for (const id of pathIds) {
    if (ix.nodes.has(id)) keepNodes.add(id);
    else if (ix.edges.has(id)) {
      const e = ix.edges.get(id)!;
      keepEdges.add(id);
      keepNodes.add(e.src);
      keepNodes.add(e.dst);
    }
  }
  if (failedId && ix.nodes.has(failedId)) keepNodes.add(failedId);
  const view: GraphView = {
    ...input.view,
    nodes: input.view.nodes.filter((n) => keepNodes.has(n.id)),
    edges: input.view.edges.filter((e) => keepEdges.has(e.id)),
  };
  const model = buildGraphModel({ ...input, view }, { focus: { mode: 'all' }, ghosts: false });
  // 경로 그래프에서는 구조 엣지도 흐리게 하지 않고, 채널은 오른쪽 끝에 둔다
  const layer: GraphModel['layer'] = { ...model.layer };
  for (const n of model.nodes) if (n.data.kind === 'channel') layer[n.id] = 'LAST';
  const edges = model.edges.map((e): OpsFlowEdge => {
    const data = e.data as OpsEdgeData;
    return data.tone === 'faded' ? { ...e, data: { ...data, tone: 'normal' } } : e;
  });
  return { ...model, layer, edges };
}
