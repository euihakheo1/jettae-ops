/**
 * ELK layered 레이아웃 (왼쪽 → 오른쪽): 감지 → 서비스 → 담당 → 상향 → 승인/조치 → 채널.
 * 연결선도 ELK가 곡선(스플라인)으로 그린다 → 여러 층을 건너는 선이 중간 노드를 가로지르지 않는다.
 * 브라우저에서는 ELK 공식 웹 워커(elk-worker.min.js)로 계산해 메인 스레드를 막지 않는다.
 * 입력을 id 순으로 정렬하고 randomSeed를 고정해 같은 그래프는 언제나 같은 배치가 나온다.
 */
import type { ELK, ElkExtendedEdge, ElkNode } from 'elkjs/lib/elk-api';
import type { GraphModel } from './model';

export interface Point {
  x: number;
  y: number;
}

export interface EdgeRoute {
  /** 3차 베지어 조각들: 시작점, 그다음 (제어점1, 제어점2, 끝점) 묶음이 이어진다 → 길이 = 1 + 3k */
  points: Point[];
}

export interface LayoutResult {
  positions: Record<string, { x: number; y: number }>;
  routes: Record<string, EdgeRoute>;
  width: number;
  height: number;
}

export interface LayoutOptionsInput {
  /** 작은 경로 그래프용 촘촘한 간격 */
  compact?: boolean;
  /** 전체 보기: 위아래는 촘촘하게, 좌우는 넓게 (세로로 길어지는 큰 그래프를 화면 비율에 맞춘다) */
  overview?: boolean;
}

export function elkLayoutOptions(opts: LayoutOptionsInput = {}): Record<string, string> {
  return {
    'elk.algorithm': 'layered',
    'elk.direction': 'RIGHT',
    'elk.randomSeed': '1',
    'elk.edgeRouting': 'SPLINES',
    'elk.layered.edgeRouting.splines.mode': 'CONSERVATIVE',
    'elk.layered.nodePlacement.strategy': 'NETWORK_SIMPLEX',
    'elk.layered.crossingMinimization.strategy': 'LAYER_SWEEP',
    'elk.layered.considerModelOrder.strategy': 'NODES_AND_EDGES',
    'elk.layered.cycleBreaking.strategy': 'DEPTH_FIRST',
    // 층 사이에 연결 라벨(담당·호출·상향 5분…)이 들어갈 자리를 둔다
    'elk.layered.spacing.nodeNodeBetweenLayers': opts.compact ? '112' : opts.overview ? '190' : '104',
    'elk.spacing.nodeNode': opts.compact ? '30' : opts.overview ? '16' : '34',
    'elk.spacing.edgeNode': '24',
    'elk.spacing.edgeEdge': '14',
    'elk.separateConnectedComponents': 'true',
    'elk.spacing.componentComponent': '72',
    'elk.padding': '[top=24,left=24,bottom=24,right=24]',
  };
}

/** GraphModel → ELK 입력 (순수·결정적) */
export function toElkGraph(model: GraphModel, opts: LayoutOptionsInput = {}): ElkNode {
  const nodeIds = new Set(model.nodes.map((n) => n.id));
  const children: ElkNode[] = [...model.nodes]
    .sort((a, b) => (a.id < b.id ? -1 : 1))
    .map((n) => {
      const constraint = model.layer[n.id];
      return {
        id: n.id,
        width: n.width ?? 200,
        height: n.height ?? 60,
        ...(constraint ? { layoutOptions: { 'elk.layered.layering.layerConstraint': constraint } } : {}),
      };
    });
  const edges: ElkExtendedEdge[] = [...model.edges]
    .filter((e) => nodeIds.has(e.source) && nodeIds.has(e.target) && e.source !== e.target)
    .sort((a, b) => (a.id < b.id ? -1 : 1))
    .map((e) => ({ id: e.id, sources: [e.source], targets: [e.target] }));
  return { id: 'root', layoutOptions: elkLayoutOptions(opts), children, edges };
}

export async function layoutWithElk(elk: ELK, model: GraphModel, opts: LayoutOptionsInput = {}): Promise<LayoutResult> {
  const result = await elk.layout(toElkGraph(model, opts));
  const positions: LayoutResult['positions'] = {};
  for (const c of result.children ?? []) positions[c.id] = { x: Math.round(c.x ?? 0), y: Math.round(c.y ?? 0) };
  const routes: LayoutResult['routes'] = {};
  for (const e of result.edges ?? []) {
    const sec = e.sections?.[0];
    if (!sec) continue;
    const points = [sec.startPoint, ...(sec.bendPoints ?? []), sec.endPoint].map((p) => ({ x: Math.round(p.x * 10) / 10, y: Math.round(p.y * 10) / 10 }));
    // 베지어 조각 모양(1 + 3k)이 아니면 쓰지 않는다 → React Flow 곡선으로 그린다
    if (points.length >= 4 && (points.length - 1) % 3 === 0) routes[e.id] = { points };
  }
  return { positions, routes, width: result.width ?? 0, height: result.height ?? 0 };
}

/** 모델 식별 키 (레이아웃 캐시용): 노드 id·크기 + 엣지 id·양끝 */
export function layoutKey(model: GraphModel, opts: LayoutOptionsInput = {}): string {
  const n = model.nodes.map((x) => `${x.id}|${x.width}x${x.height}|${model.layer[x.id] ?? ''}`).join(';');
  const e = model.edges.map((x) => `${x.id}=${x.source}>${x.target}`).join(';');
  return `${opts.compact ? 'c' : opts.overview ? 'o' : 'n'}#${n}#${e}`;
}

let elkPromise: Promise<ELK> | null = null;

/** 브라우저용 ELK 인스턴스 — 워커 우선, 실패 시 메인 스레드 번들 */
export function getElk(): Promise<ELK> {
  if (elkPromise) return elkPromise;
  elkPromise = (async () => {
    if (typeof Worker !== 'undefined') {
      try {
        const [{ default: ElkApi }, { default: workerUrl }] = await Promise.all([
          import('elkjs/lib/elk-api'),
          import('elkjs/lib/elk-worker.min.js?url'),
        ]);
        return new ElkApi({ workerUrl });
      } catch {
        // 아래 번들로 폴백
      }
    }
    const { default: ElkBundled } = await import('elkjs/lib/elk.bundled.js');
    return new ElkBundled();
  })();
  return elkPromise;
}

const cache = new Map<string, LayoutResult>();

export async function layoutModel(model: GraphModel, opts: LayoutOptionsInput = {}): Promise<LayoutResult> {
  const key = layoutKey(model, opts);
  const hit = cache.get(key);
  if (hit) return hit;
  const elk = await getElk();
  const res = await layoutWithElk(elk, model, opts);
  if (cache.size > 40) cache.clear();
  cache.set(key, res);
  return res;
}
