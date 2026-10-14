/**
 * 제때옵스 REST 클라이언트 (ARCHITECTURE §6, /api/v1).
 *  - 개발용 인증 헤더 X-Jettae-User / X-Jettae-Role 은 상단 역할 전환기 값으로 보낸다.
 *  - VITE_USE_MOCKS=1 이면 목업(src/api/mock.ts)을, API에 닿지 않으면 자동으로 오프라인 데모 데이터를 쓴다.
 *  - 배포 빌드(공개 베타)에서는 쓰기를 목업으로 보내지 않고, 예제가 아닌 워크스페이스에 예제 데이터를 보여 주지 않는다
 *    ({@link strictOffline}).
 */
import { graphViewFromOpsGraph } from '../graph/model';
import { ROLE_PERSON, ROLE_USER, roleStore } from '../app/prefs';
import { accessKeyFor, expireFreeScan, listFreeScans, saveFreeScan, workspaceIdFromPath } from '../lib/freeScans';
import { ApiError, UnreachableError } from './errors';
import { getSourceMode, mocksForced, setSourceMode } from './source';
import type {
  DiagnoseRequest,
  DiagnosisReport,
  DraftRequest,
  EventRequest,
  FreeScanResult,
  GraphDiff,
  GraphVersion,
  GraphView,
  InputFiles,
  OpsGraph,
  Patch,
  RehearsalRequest,
  RehearsalResult,
  RuleInfo,
  SessionInfo,
  SignupRequest,
  SourceDocument,
  WorkspaceSummary,
} from './types';

export const API_BASE = '/api/v1';

/** 오프라인 데모 데이터(mock.ts)의 예제 워크스페이스 id — mock.DEMO_WORKSPACE_ID와 같다 */
export const OFFLINE_DEMO_ID = 'dmc-shop';

/**
 * 배포 빌드(공개 베타)에서 오프라인 대체를 좁힐지.
 *  - 쓰기(사전 신청·무료 검진·예제 열기·이벤트 등)는 목업으로 보내지 않는다 — 실패하면 오류를 보여 다시 시도하게 한다
 *  - 예제가 아닌 워크스페이스(방문자의 무료 검진 등)에는 예제 데이터를 보여 주지 않는다
 *  - 오프라인으로 바꾸기 전에 /healthz를 한 번 더 확인하고, 오프라인 중에도 서버가 답하면 API로 돌아온다
 * 개발 서버(npm run dev)와 목업 모드(VITE_USE_MOCKS=1)는 예전처럼 동작한다.
 */
export function strictOffline(): boolean {
  return import.meta.env.PROD && !mocksForced();
}

/** 예제 워크스페이스 id — 오프라인 데모 데이터의 id와, 서버 목록·예제 열기에서 본 서버의 예제 id */
const demoIds = new Set<string>([OFFLINE_DEMO_ID]);

function rememberDemos(list: readonly WorkspaceSummary[]): void {
  for (const w of list) if (w.is_demo) demoIds.add(w.id);
}

export function isDemoWorkspace(ws: string): boolean {
  return demoIds.has(ws);
}

/** 서버가 마지막으로 알려 준 세션(GET /me) — 공개 체험판 방문자의 403을 안내 문구로 바꿀 때 쓴다 */
let lastSession: SessionInfo | null = null;

/** 공개 베타(토큰 없는 방문자)가 팀 계정이 필요한 작업을 했을 때의 안내 */
export const PUBLIC_TRIAL_MESSAGE =
  '공개 체험판에서는 할 수 없는 작업입니다(팀 계정 필요). 런북 무료 검진과 별빛상점 예제 둘러보기는 그대로 쓸 수 있습니다.';

/**
 * 서버에서 사라진 무료 검진(클라이언트가 붙이는 오류 코드). 보관 기간(JETTAE_FREE_SCAN_RETENTION_DAYS)이 지났을 수도,
 * 무료 DB를 다시 만들어 지워졌을 수도 있어 이유를 하나로 단정하지 않는다.
 */
export const FREE_SCAN_EXPIRED_CODE = 'free_scan_expired';
export const FREE_SCAN_EXPIRED_NOTICE = '보관 기간이 지났거나 서버에서 삭제된 무료 검진이라 목록에서 뺐습니다.';
export const FREE_SCAN_EXPIRED_MESSAGE = `${FREE_SCAN_EXPIRED_NOTICE} 런북을 다시 올려 새로 검진하세요.`;

/** 서버에 잠깐 닿지 않을 때(배포 빌드) — 목업으로 대신하지 않고 다시 시도하게 한다 */
export const RETRY_LATER_MESSAGE = '서버에 잠시 연결할 수 없습니다. 잠시 후 다시 시도하세요.';

type MockModule = typeof import('./mock');
let mockModule: Promise<MockModule> | null = null;
export function loadMock(): Promise<MockModule> {
  mockModule ??= import('./mock');
  return mockModule;
}

function authHeaders(): Record<string, string> {
  const role = roleStore.get();
  return { 'X-Jettae-User': ROLE_USER[role], 'X-Jettae-Role': role };
}

/** 공개 베타의 무료 검진 워크스페이스는 이 브라우저에 저장된 접근 키가 있어야 열린다 */
export const WORKSPACE_KEY_HEADER = 'X-Jettae-Workspace-Key';

export function workspaceKeyHeaders(path: string): Record<string, string> {
  const ws = workspaceIdFromPath(path);
  const key = ws ? accessKeyFor(ws) : null;
  return key ? { [WORKSPACE_KEY_HEADER]: key } : {};
}

interface RequestOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE';
  json?: unknown;
  form?: FormData;
  query?: Record<string, string | number | undefined | null>;
  signal?: AbortSignal;
  accept?: 'json' | 'text' | 'blob';
}

function buildUrl(path: string, query?: RequestOptions['query']): string {
  const url = path.startsWith('/') ? path : `${API_BASE}/${path}`;
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(query ?? {})) if (v !== undefined && v !== null && v !== '') qs.set(k, String(v));
  const s = qs.toString();
  return s ? `${url}?${s}` : url;
}

interface ErrorBody {
  detail?: unknown;
  message?: unknown;
  code?: unknown;
  error?: { message?: unknown };
}

async function errorFrom(res: Response): Promise<ApiError | UnreachableError> {
  const type = res.headers.get('content-type') ?? '';
  const text = await res.text().catch(() => '');
  let body: ErrorBody | null = null;
  if (type.includes('json')) {
    try {
      const parsed: unknown = JSON.parse(text);
      if (parsed && typeof parsed === 'object') body = parsed as ErrorBody;
    } catch {
      // 본문이 JSON이 아님
    }
  }
  // 개발 프록시·호스팅 프록시가 대상(FastAPI)에 닿지 못하면 JSON이 아닌(빈) 500/502/503/504를 준다.
  // JSON 본문이 있는 502/503/504(예제 묶음 없음·서비스 준비 중 등)는 서버가 직접 한 답이라 API 오류로 다룬다.
  if (([502, 503, 504].includes(res.status) && !body) || (res.status === 500 && !type.includes('json') && text.trim() === '')) {
    return new UnreachableError();
  }
  let message = `서버 오류 (${res.status})`;
  if (body) {
    if (body.error && typeof body.error.message === 'string') message = body.error.message;
    else if (typeof body.detail === 'string') message = body.detail;
    else if (Array.isArray(body.detail))
      message = body.detail
        .map((d: unknown) => (typeof d === 'object' && d && 'msg' in d ? String((d as { msg: unknown }).msg) : String(d)))
        .join(', ');
    else if (typeof body.message === 'string') message = body.message;
  }
  // 공개 베타(토큰 없는 방문자): 초안·리허설·승인·되돌리기·팀 워크스페이스 만들기는 팀 계정 토큰이 있어야 한다
  if (res.status === 401 || (res.status === 403 && lastSession?.anonymous)) message = PUBLIC_TRIAL_MESSAGE;
  if (res.status === 403 && message.startsWith('서버 오류')) message = '권한이 없습니다. 상단에서 역할을 확인하세요.';
  if (res.status === 404 && message.startsWith('서버 오류')) message = '요청한 항목을 찾을 수 없습니다.';
  return new ApiError(res.status, message, text.slice(0, 500), typeof body?.code === 'string' ? body.code : undefined);
}

const enc = encodeURIComponent;

/** 같은 워크스페이스를 여러 화면이 동시에 물어도 한 번만 확인한다 */
const expiryChecks = new Map<string, Promise<boolean>>();

/**
 * 이 브라우저에 키가 있는 무료 검진이 서버에서 사라졌는지(보관 기간이 지났거나 서버에서 삭제) 확인한다.
 * 하위 항목의 404(원문 없음 등)와 구분하려고 워크스페이스 자체(GET /workspaces/{ws})를 같은 키로 다시 묻고,
 * 그것도 404면 목록에서 빼고(화면이 알림을 띄운다) true를 돌려준다.
 */
function freeScanGone(ws: string, key: string): Promise<boolean> {
  let check = expiryChecks.get(ws);
  if (!check) {
    check = (async () => {
      try {
        const res = await fetch(buildUrl(`workspaces/${enc(ws)}`), {
          headers: { ...authHeaders(), Accept: 'application/json', [WORKSPACE_KEY_HEADER]: key },
        });
        return res.status === 404 && expireFreeScan(ws);
      } catch {
        return false;
      }
    })();
    expiryChecks.set(ws, check);
    void check.finally(() => expiryChecks.delete(ws));
  }
  return check;
}

export async function request<T>(path: string, opts: RequestOptions = {}): Promise<T> {
  const headers: Record<string, string> = { ...authHeaders(), ...workspaceKeyHeaders(path) };
  let body: BodyInit | undefined;
  if (opts.json !== undefined) {
    headers['Content-Type'] = 'application/json';
    body = JSON.stringify(opts.json);
  } else if (opts.form) body = opts.form;
  if ((opts.accept ?? 'json') === 'json') headers.Accept = 'application/json';
  let res: Response;
  try {
    res = await fetch(buildUrl(path, opts.query), {
      method: opts.method ?? (body ? 'POST' : 'GET'),
      headers,
      ...(body !== undefined ? { body } : {}),
      ...(opts.signal ? { signal: opts.signal } : {}),
    });
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') throw e;
    throw new UnreachableError();
  }
  if (!res.ok) {
    const err = await errorFrom(res);
    const key = headers[WORKSPACE_KEY_HEADER];
    const ws = workspaceIdFromPath(path);
    if (res.status === 404 && key && ws && (await freeScanGone(ws, key))) {
      throw new ApiError(404, FREE_SCAN_EXPIRED_MESSAGE, err instanceof ApiError ? err.detail : undefined, FREE_SCAN_EXPIRED_CODE);
    }
    throw err;
  }
  const accept = opts.accept ?? 'json';
  if (accept === 'text') return (await res.text()) as T;
  if (accept === 'blob') return (await res.blob()) as T;
  if (res.status === 204) return undefined as T;
  const type = res.headers.get('content-type') ?? '';
  if (!type.includes('json')) {
    // SPA 폴백(index.html)이 돌아왔다면 API가 없는 것
    throw new UnreachableError('API 응답이 JSON이 아닙니다 (API 서버가 꺼져 있을 수 있습니다)');
  }
  return (await res.json()) as T;
}

/** GET /healthz 가 2xx면 true — 닿지 않거나(네트워크·프록시 오류) 다른 서버가 응답하면 false */
async function probeHealth(): Promise<boolean> {
  try {
    const res = await fetch('/healthz', { headers: { Accept: 'application/json' } });
    return res.ok;
  } catch {
    return false;
  }
}

/**
 * 처음 한 번 GET /healthz 로 제때옵스 API가 살아 있는지 확인한다.
 * 닿지 않거나 다른 서버가 응답하면 오프라인 데모 데이터로 전환한다
 * (배포 빌드는 한 번의 실패로 세션 내내 오프라인이 되지 않게 한 번 더 확인한다).
 */
let apiCheck: Promise<void> | null = null;
export function resetApiCheck(): void {
  apiCheck = null;
}
function ensureApi(): Promise<void> {
  apiCheck ??= (async () => {
    if (await probeHealth()) return;
    if (strictOffline() && (await probeHealth())) return;
    setSourceMode('offline');
  })();
  return apiCheck;
}

/** 무엇을 하는 호출인지 — 배포 빌드에서 목업으로 대신해도 되는지 정한다 */
interface FallbackScope {
  /** 서버에 남는 쓰기(사전 신청·무료 검진·예제 열기 등): 배포 빌드에서는 목업으로 보내지 않는다 */
  write?: boolean;
  /** 서버만 답할 수 있는 읽기(GET /me의 역할 등): 배포 빌드에서는 목업으로 꾸며 내지 않는다 */
  serverOnly?: boolean;
  /** 워크스페이스 화면의 호출(승인·거절·되돌리기 포함): 배포 빌드에서는 예제 워크스페이스일 때만 목업을 쓴다 */
  ws?: string;
}

/** API 우선, 닿지 않으면 오프라인 데모 데이터(배포 빌드에서는 {@link strictOffline}의 범위 안에서만) */
async function withFallback<T>(
  live: () => Promise<T>,
  viaMock: (m: MockModule) => Promise<T>,
  scope: FallbackScope = {},
): Promise<T> {
  const strict = strictOffline();
  const mockable = !strict || (!scope.write && !scope.serverOnly && (scope.ws === undefined || isDemoWorkspace(scope.ws)));
  if (getSourceMode() === 'api') await ensureApi();
  if (getSourceMode() !== 'api' && mockable) return viaMock(await loadMock());
  try {
    const result = await live();
    // 배포 빌드: 오프라인으로 바뀐 뒤에도 서버가 답했다면 다시 API로 돌아온다
    if (strict && getSourceMode() === 'offline') {
      resetApiCheck();
      setSourceMode('api');
    }
    return result;
  } catch (e) {
    if (!(e instanceof UnreachableError)) throw e;
    // 배포 빌드: 오프라인으로 바꾸기 전에 /healthz를 다시 확인한다 — 서버가 살아 있으면 일시적인 오류
    if (strict && getSourceMode() === 'api' && (await probeHealth())) throw new UnreachableError(RETRY_LATER_MESSAGE);
    setSourceMode('offline');
    if (mockable) return viaMock(await loadMock());
    throw new UnreachableError(RETRY_LATER_MESSAGE);
  }
}

const OFFLINE_WRITE = new ApiError(
  501,
  '오프라인 데모 모드에서는 새 진단을 실행할 수 없습니다. API 서버(jettae serve)를 켠 뒤 다시 시도하거나 별빛상점 예제를 불러오세요.',
);

function isGraphView(x: unknown): x is GraphView {
  return typeof x === 'object' && x !== null && Array.isArray((x as GraphView).nodes);
}

/** GraphView DTO 또는 {version, graph: OpsGraph, layout_hints} 둘 다 받는다 */
export function normalizeGraphResponse(res: unknown): GraphView {
  if (isGraphView(res)) return { ...res, findings_by_element: res.findings_by_element ?? {} };
  const r = res as { version?: number; graph?: OpsGraph; graph_hash?: string };
  if (r && r.graph && typeof r.graph === 'object') return graphViewFromOpsGraph(r.graph, [], r.version ?? 1, r.graph_hash ?? '');
  throw new ApiError(500, '그래프 응답 형식을 알 수 없습니다.');
}

export interface ApproveResult {
  patch: Patch | null;
  version: GraphVersion | null;
}

function normalizeApprove(res: unknown): ApproveResult {
  if (res && typeof res === 'object') {
    const o = res as Record<string, unknown>;
    if ('patch' in o || 'version' in o)
      return { patch: (o.patch as Patch | undefined) ?? null, version: (o.version as GraphVersion | undefined) ?? null };
    if ('number' in o && 'graph_hash' in o) return { patch: null, version: o as unknown as GraphVersion };
    if ('ops' in o) return { patch: o as unknown as Patch, version: null };
  }
  return { patch: null, version: null };
}

/** POST /free-scans 의 런북 파일 필드 이름 */
const FREE_SCAN_RUNBOOK_FIELD = 'files';

/**
 * 서버 목록 + 이 브라우저의 무료 검진. 공개 베타의 방문자에게 서버는 예제만 돌려주므로,
 * 방문자가 올린 검진은 브라우저에 저장된 목록으로 보여 준다(서버 목록에 이미 있으면 그쪽을 쓴다).
 */
export function withLocalFreeScans(server: WorkspaceSummary[]): WorkspaceSummary[] {
  const known = new Set(server.map((w) => w.id));
  const local = listFreeScans()
    .filter((s) => !known.has(s.workspace_id))
    .map<WorkspaceSummary>((s) => ({
      id: s.workspace_id,
      name: '무료 검진',
      created_at: s.created_at,
      // 무료 검진은 진단 한 번으로 끝나고(다시 진단·승인 없음) 언제나 v1이다
      latest_version: 1,
      overall: s.readiness_overall,
      finding_count: s.findings,
      is_free_scan: true,
    }));
  return [...local, ...server];
}

/**
 * 서버 목록에서 진단 결과(리포트)가 있는 예제 워크스페이스 id를 찾는다.
 * 목록을 못 받거나, 예제가 있어도 진단 전(latest_version 없음 — 시작 때 예제 진단이 실패한 경우)이면 null:
 * 그때는 POST /demo/dmc-shop 으로 넘어가 서버가 진단하게 한다.
 */
async function findDemoWorkspace(): Promise<string | null> {
  try {
    const res = await request<WorkspaceSummary[] | { items: WorkspaceSummary[] }>('workspaces');
    const list = Array.isArray(res) ? res : res.items;
    rememberDemos(list);
    return list.find((w) => w.is_demo && w.latest_version !== undefined && w.latest_version !== null)?.id ?? null;
  } catch (e) {
    if (e instanceof ApiError) return null;
    throw e;
  }
}

export const api = {
  async health(): Promise<boolean> {
    if (getSourceMode() !== 'api') return false;
    await ensureApi();
    return getSourceMode() === 'api';
  },

  /**
   * 서버가 보는 현재 사용자.
   *  - 개발 서버·목업: 오프라인이면 화면의 역할 전환기를 그대로 쓰는 로컬 세션
   *  - 배포 빌드: 세션을 꾸며 내지 않는다 — 서버가 답하기 전(대기·실패)에는 화면이 보기 전용으로 본다(app/permissions.ts)
   */
  session(): Promise<SessionInfo> {
    const local = (): Promise<SessionInfo> =>
      Promise.resolve({ user: ROLE_USER[roleStore.get()], role: roleStore.get(), anonymous: false, auth_mode: 'dev', can_switch_role: true });
    return withFallback(
      async () => {
        const session = await request<SessionInfo>('me');
        lastSession = session;
        return session;
      },
      local,
      { serverOnly: true },
    );
  },

  listWorkspaces(): Promise<WorkspaceSummary[]> {
    return withFallback(
      async () => {
        const res = await request<WorkspaceSummary[] | { items: WorkspaceSummary[] }>('workspaces');
        const list = Array.isArray(res) ? res : res.items;
        rememberDemos(list);
        return withLocalFreeScans(list);
      },
      // 배포 빌드의 오프라인 목록에도 이 브라우저의 무료 검진을 남긴다(서버가 돌아오면 열린다)
      async (m) => {
        const list = await m.listWorkspaces();
        return strictOffline() ? withLocalFreeScans(list) : list;
      },
    );
  },

  createWorkspace(name: string): Promise<{ id: string }> {
    return withFallback(
      () => request<{ id: string }>('workspaces', { json: { name } }),
      () => Promise.reject(OFFLINE_WRITE),
      { write: true },
    );
  },

  uploadInputs(ws: string, files: InputFiles): Promise<unknown> {
    const form = new FormData();
    for (const f of files.runbooks) form.append('runbooks', f, f.name);
    if (files.roster) form.append('roster', files.roster, files.roster.name);
    if (files.oncall) form.append('oncall', files.oncall, files.oncall.name);
    if (files.catalog) form.append('catalog', files.catalog, files.catalog.name);
    if (files.policy) form.append('policy', files.policy, files.policy.name);
    return withFallback(
      () => request<unknown>(`workspaces/${enc(ws)}/inputs`, { form, method: 'POST' }),
      () => Promise.reject(OFFLINE_WRITE),
      { write: true, ws },
    );
  },

  /**
   * 무료 검진: 업로드·진단을 한 번에 하고, 결과를 열 접근 키를 이 브라우저에 저장한다.
   * 공개 베타에서 팀 계정이 없는 방문자가 쓰는 길이다(서버가 'runbook_upload'를 직접 센다).
   */
  async freeScan(files: InputFiles): Promise<FreeScanResult> {
    const form = new FormData();
    for (const f of files.runbooks) form.append(FREE_SCAN_RUNBOOK_FIELD, f, f.name);
    if (files.roster) form.append('roster', files.roster, files.roster.name);
    if (files.oncall) form.append('oncall', files.oncall, files.oncall.name);
    if (files.catalog) form.append('catalog', files.catalog, files.catalog.name);
    if (files.policy) form.append('policy', files.policy, files.policy.name);
    const res = await withFallback(
      () => request<FreeScanResult>('free-scans', { form, method: 'POST' }),
      () => Promise.reject(OFFLINE_WRITE),
      { write: true },
    );
    saveFreeScan({ ...res, created_at: new Date().toISOString() });
    return res;
  },

  diagnose(ws: string, req: DiagnoseRequest): Promise<DiagnosisReport> {
    return withFallback(
      () => request<DiagnosisReport>(`workspaces/${enc(ws)}/diagnose`, { json: req }),
      () => Promise.reject(OFFLINE_WRITE),
      { write: true, ws },
    );
  },

  /**
   * 별빛상점 예제 열기. 서버에 예제가 이미 있으면(누구에게나 목록에 보인다) 그 워크스페이스로 가고,
   * 없을 때만 POST /demo/dmc-shop 으로 만든다 — 예제 진단은 무료 검진 한도를 쓰고 CPU도 많이 들기 때문.
   * 만들기가 한도 초과(429)·일시 불가(503)면 그사이 생긴 예제가 있는지 한 번 더 본다.
   */
  createDemo(): Promise<{ id: string }> {
    return withFallback(
      async () => {
        const existing = await findDemoWorkspace();
        if (existing) return { id: existing };
        try {
          const res = await request<{ id?: string; workspace_id?: string }>('demo/dmc-shop', { method: 'POST' });
          const id = res.id ?? res.workspace_id ?? OFFLINE_DEMO_ID;
          demoIds.add(id);
          return { id };
        } catch (e) {
          if (e instanceof ApiError && (e.status === 429 || e.status === 503)) {
            const fallback = await findDemoWorkspace();
            if (fallback) return { id: fallback };
          }
          throw e;
        }
      },
      (m) => m.createDemo(),
      { write: true },
    );
  },

  getReport(ws: string, version?: number | null): Promise<DiagnosisReport> {
    return withFallback(
      () => request<DiagnosisReport>(`workspaces/${enc(ws)}/report`, { query: { version } }),
      (m) => m.getReport(ws, version),
      { ws },
    );
  },

  getGraph(ws: string, version?: number | null): Promise<GraphView> {
    return withFallback(
      async () => normalizeGraphResponse(await request<unknown>(`workspaces/${enc(ws)}/graph`, { query: { version } })),
      (m) => m.getGraph(ws, version),
      { ws },
    );
  },

  getVersions(ws: string): Promise<GraphVersion[]> {
    return withFallback(
      async () => {
        const res = await request<GraphVersion[] | { items: GraphVersion[] }>(`workspaces/${enc(ws)}/versions`);
        return Array.isArray(res) ? res : res.items;
      },
      (m) => m.getVersions(ws),
      { ws },
    );
  },

  getDiff(ws: string, a: number, b: number): Promise<GraphDiff> {
    return withFallback(
      () => request<GraphDiff>(`workspaces/${enc(ws)}/versions/diff`, { query: { a, b } }),
      (m) => m.getDiff(ws, a, b),
      { ws },
    );
  },

  revertVersion(ws: string, n: number): Promise<GraphVersion> {
    const role = roleStore.get();
    return withFallback(
      () => request<GraphVersion>(`workspaces/${enc(ws)}/versions/${n}/revert`, { method: 'POST', json: {} }),
      (m) => m.revertVersion(ws, n, ROLE_PERSON[role], role),
      { ws },
    );
  },

  /** §6에는 목록 엔드포인트가 없어 GET /patches 를 먼저 시도하고, 없으면 진단 보고서의 patches 를 쓴다 */
  getPatches(ws: string): Promise<Patch[]> {
    return withFallback(
      async () => {
        try {
          const res = await request<Patch[] | { items: Patch[] }>(`workspaces/${enc(ws)}/patches`);
          return Array.isArray(res) ? res : res.items;
        } catch (e) {
          if (e instanceof ApiError && (e.status === 404 || e.status === 405) && e.code !== FREE_SCAN_EXPIRED_CODE) {
            return (await request<DiagnosisReport>(`workspaces/${enc(ws)}/report`)).patches;
          }
          throw e;
        }
      },
      (m) => m.getPatches(ws),
      { ws },
    );
  },

  draftPatches(ws: string, req: DraftRequest): Promise<Patch[]> {
    return withFallback(
      () => request<Patch[]>(`workspaces/${enc(ws)}/patches/draft`, { json: req }),
      (m) => m.draftPatches(ws, req),
      { ws },
    );
  },

  /** 수정안 승인 — `ws`는 수정안이 속한 워크스페이스(배포 빌드에서는 예제일 때만 오프라인 목업으로 대신한다) */
  approvePatch(ws: string, id: string): Promise<ApproveResult> {
    const role = roleStore.get();
    return withFallback(
      async () => normalizeApprove(await request<unknown>(`patches/${enc(id)}/approve`, { method: 'POST', json: {} })),
      (m) => m.approvePatch(id, ROLE_PERSON[role], role),
      { ws },
    );
  },

  /** 수정안 거절 — `ws`는 {@link api.approvePatch}와 같다 */
  rejectPatch(ws: string, id: string, reason: string): Promise<Patch | null> {
    const role = roleStore.get();
    return withFallback(
      async () => {
        const res = await request<unknown>(`patches/${enc(id)}/reject`, { method: 'POST', json: { reason } });
        return res && typeof res === 'object' && 'ops' in res ? (res as Patch) : null;
      },
      (m) => m.rejectPatch(id, reason, ROLE_PERSON[role], role),
      { ws },
    );
  },

  runRehearsals(ws: string, req: RehearsalRequest): Promise<RehearsalResult[]> {
    return withFallback(
      () => request<RehearsalResult[]>(`workspaces/${enc(ws)}/rehearsals`, { json: req }),
      (m) => m.runRehearsals(ws, req),
      { ws },
    );
  },

  getRules(): Promise<RuleInfo[]> {
    return withFallback(
      () => request<RuleInfo[]>('rules'),
      (m) => m.getRules(),
    );
  },

  /** 원문 문서(근거 문맥 표시용). 원문 비저장 설정이면 없을 수 있다 → 빈 목록 */
  getDocuments(ws: string): Promise<SourceDocument[]> {
    return withFallback(
      async () => {
        try {
          return await request<SourceDocument[]>(`workspaces/${enc(ws)}/documents`);
        } catch (e) {
          if (e instanceof ApiError && (e.status === 404 || e.status === 405 || e.status === 403) && e.code !== FREE_SCAN_EXPIRED_CODE) return [];
          throw e;
        }
      },
      (m) => m.getDocuments(ws),
      { ws },
    );
  },

  signup(req: SignupRequest): Promise<unknown> {
    return withFallback(
      () => request<unknown>('signups', { json: req }),
      (m) => m.signup(req),
      { write: true },
    );
  },

  /**
   * 수요 측정 이벤트 — 실패해도 예외를 던지지 않고, 서버가 2xx로 받았는지만 돌려준다.
   * 경로는 서버가 받는 화면 경로(types.ts의 `EVENT_PATHS`) 중 하나라 워크스페이스 id가 없고, 접근 키도 붙이지 않는다.
   * 배포 빌드에서는 오프라인이어도 목업으로 보내지 않고 서버로 보낸다(닿지 않으면 false).
   */
  async track(req: EventRequest): Promise<boolean> {
    try {
      if (!strictOffline()) {
        if (getSourceMode() === 'api') await ensureApi();
        if (getSourceMode() !== 'api') {
          await (await loadMock()).trackEvent(req);
          return true;
        }
      }
      const res = await fetch(buildUrl('events'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders() },
        body: JSON.stringify(req),
        keepalive: true,
      });
      return res.ok;
    } catch {
      return false;
    }
  },

  /** 증적 리포트 내려받기 (md | html | sarif) */
  async reportFile(ws: string, kind: 'md' | 'html' | 'sarif'): Promise<{ filename: string; blob: Blob }> {
    const type = kind === 'md' ? 'text/markdown' : kind === 'html' ? 'text/html' : 'application/sarif+json';
    const filename = `jettae-report-${ws}.${kind === 'sarif' ? 'sarif' : kind}`;
    const text = await withFallback(
      () => request<string>(`workspaces/${enc(ws)}/report.${kind}`, { accept: 'text' }),
      (m) => (kind === 'md' ? m.reportMarkdownText(ws) : kind === 'html' ? m.reportHtmlText(ws) : m.reportSarifText(ws)),
      { ws },
    );
    return { filename, blob: new Blob([text], { type: `${type};charset=utf-8` }) };
  },

  reportHtml(ws: string): Promise<string> {
    return withFallback(
      () => request<string>(`workspaces/${enc(ws)}/report.html`, { accept: 'text' }),
      (m) => m.reportHtmlText(ws),
      { ws },
    );
  },
};

export function saveBlob(filename: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}
