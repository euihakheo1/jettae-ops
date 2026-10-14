import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { SignupRequest } from './types';

/** 오프라인 대체로 불렸는지 보려고 목업 모듈을 가짜로 바꾼다 */
const mockApi = {
  signup: vi.fn(async () => ({ ok: true as const })),
  trackEvent: vi.fn(async () => undefined),
  createDemo: vi.fn(async () => ({ id: 'dmc-shop' })),
  listWorkspaces: vi.fn(async () => [{ id: 'dmc-shop', name: '별빛상점 (DMC 예제)', is_demo: true }]),
  getReport: vi.fn(async () => ({ from: 'mock' })),
  approvePatch: vi.fn(async () => ({ patch: null, version: null })),
  rejectPatch: vi.fn(async () => null),
};

interface Call {
  method: string;
  url: string;
  headers: Record<string, string>;
}

type Route = (call: Call) => Response | Promise<Response>;

const json = (body: unknown, status = 200): Response =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
const empty = (status: number): Response => new Response(null, { status });

/** `METHOD path` → 응답. 맞는 길이 없으면 네트워크 오류(서버에 닿지 않음)처럼 던진다 */
function serve(routes: Record<string, Route | Response>): Call[] {
  const calls: Call[] = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = init?.method ?? 'GET';
      const call: Call = { method, url, headers: (init?.headers ?? {}) as Record<string, string> };
      calls.push(call);
      const route = routes[`${method} ${url.split('?')[0]}`];
      if (!route) throw new TypeError('Failed to fetch');
      return route instanceof Response ? route.clone() : route(call);
    }),
  );
  return calls;
}

async function load() {
  vi.resetModules();
  vi.doMock('./mock', () => mockApi);
  const client = await import('./client');
  const source = await import('./source');
  const errors = await import('./errors');
  const freeScans = await import('../lib/freeScans');
  const demand = await import('../lib/demand');
  return { ...client, ...source, ...errors, freeScans, demand };
}

const HEALTHY = { 'GET /healthz': json({ status: 'ok' }) };
const posts = (calls: Call[], url: string) => calls.filter((c) => c.method === 'POST' && c.url === url);

beforeEach(() => {
  localStorage.clear();
  vi.clearAllMocks();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
  vi.doUnmock('./mock');
});

describe('예제 열기 (createDemo)', () => {
  it('서버에 예제가 있으면 만들지 않고 그 워크스페이스로 간다', async () => {
    const { api } = await load();
    const calls = serve({ ...HEALTHY, 'GET /api/v1/workspaces': json([{ id: 'ws-demo', name: '별빛상점', is_demo: true, latest_version: 1 }]) });
    await expect(api.createDemo()).resolves.toEqual({ id: 'ws-demo' });
    expect(posts(calls, '/api/v1/demo/dmc-shop')).toHaveLength(0);
  });

  it('예제가 없을 때만 POST /demo/dmc-shop', async () => {
    const { api } = await load();
    const calls = serve({
      ...HEALTHY,
      'GET /api/v1/workspaces': json([{ id: 'ws-team', name: '결제팀' }]),
      'POST /api/v1/demo/dmc-shop': json({ workspace_id: 'ws-new-demo', workspace_name: '별빛상점' }),
    });
    await expect(api.createDemo()).resolves.toEqual({ id: 'ws-new-demo' });
    expect(posts(calls, '/api/v1/demo/dmc-shop')).toHaveLength(1);
  });

  it('예제 워크스페이스가 있어도 진단 전(리포트 없음)이면 POST /demo/dmc-shop 으로 진단하게 한다', async () => {
    const { api, isDemoWorkspace } = await load();
    const calls = serve({
      ...HEALTHY,
      'GET /api/v1/workspaces': json([{ id: 'ws-empty-demo', name: '별빛상점', is_demo: true, latest_version: null }]),
      'POST /api/v1/demo/dmc-shop': json({ workspace_id: 'ws-empty-demo', workspace_name: '별빛상점' }),
    });
    await expect(api.createDemo()).resolves.toEqual({ id: 'ws-empty-demo' });
    expect(posts(calls, '/api/v1/demo/dmc-shop')).toHaveLength(1);
    // 진단 전이어도 예제 id로는 기억한다(오프라인 대체 범위)
    expect(isDemoWorkspace('ws-empty-demo')).toBe(true);
  });

  it('한도 초과(429)면 그사이 생긴 예제로 간다', async () => {
    const { api } = await load();
    let listed = 0;
    serve({
      ...HEALTHY,
      'GET /api/v1/workspaces': () => json(listed++ === 0 ? [] : [{ id: 'ws-demo', name: '별빛상점', is_demo: true, latest_version: 1 }]),
      'POST /api/v1/demo/dmc-shop': json({ detail: '오늘의 무료 검진 한도를 다 썼습니다.', code: 'free_scan_limit' }, 429),
    });
    await expect(api.createDemo()).resolves.toEqual({ id: 'ws-demo' });
  });

  it('예제도 없고 503이면 서버의 안내를 그대로 보여 준다(오프라인 예제로 바꾸지 않음)', async () => {
    const { api, getSourceMode, ApiError } = await load();
    serve({
      ...HEALTHY,
      'GET /api/v1/workspaces': json([]),
      'POST /api/v1/demo/dmc-shop': json({ detail: '예제 묶음을 찾을 수 없어 데모를 시작할 수 없습니다.', code: 'demo_unavailable' }, 503),
    });
    const err = await api.createDemo().catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err).toMatchObject({ status: 503, code: 'demo_unavailable', message: '예제 묶음을 찾을 수 없어 데모를 시작할 수 없습니다.' });
    expect(getSourceMode()).toBe('api');
    expect(mockApi.createDemo).not.toHaveBeenCalled();
  });
});

describe('수요 이벤트 (track)', () => {
  it('서버가 2xx로 받았는지만 돌려주고 예외를 던지지 않는다', async () => {
    const { api } = await load();
    let status = 204;
    serve({ ...HEALTHY, 'POST /api/v1/events': () => empty(status) });
    await expect(api.track({ name: 'landing_view', path: '/' })).resolves.toBe(true);
    status = 500;
    await expect(api.track({ name: 'landing_view', path: '/' })).resolves.toBe(false);
    serve(HEALTHY); // /events 없음 → 네트워크 오류
    await expect(api.track({ name: 'landing_view', path: '/' })).resolves.toBe(false);
  });

  it('첫 방문 표시는 서버가 이벤트를 받은 뒤에만 남긴다', async () => {
    const { demand, freeScans } = await load();
    let status = 500;
    const calls = serve({ ...HEALTHY, 'POST /api/v1/events': () => empty(status) });
    await expect(demand.trackFirstVisit('direct')).resolves.toBe(false);
    expect(freeScans.isFirstVisit()).toBe(true);
    expect(localStorage.getItem('jettae.visited')).toBeNull();
    status = 204;
    await expect(demand.trackFirstVisit('direct')).resolves.toBe(true);
    expect(freeScans.isFirstVisit()).toBe(false);
    expect(localStorage.getItem('jettae.visited')).toBe('1');
    await expect(demand.trackFirstVisit('direct')).resolves.toBe(false);
    const sent = posts(calls, '/api/v1/events');
    expect(sent).toHaveLength(2);
  });

  it('upload_click은 페이지 세션마다 한 번, 런북을 처음 골랐을 때만', async () => {
    const { demand } = await load();
    const bodies: unknown[] = [];
    serve({
      ...HEALTHY,
      'POST /api/v1/events': () => empty(204),
    });
    const fetchMock = vi.mocked(fetch);
    demand.trackUploadClick(0);
    demand.trackUploadClick(2);
    demand.trackUploadClick(3);
    await vi.waitFor(() => expect(fetchMock.mock.calls.some(([u]) => String(u) === '/api/v1/events')).toBe(true));
    for (const [u, init] of fetchMock.mock.calls) if (String(u) === '/api/v1/events') bodies.push(JSON.parse(String(init?.body)));
    expect(bodies).toEqual([{ name: 'upload_click', path: '/app', props: { files: 2 } }]);
  });

  it('upload_click의 파일 수는 서버 허용 범위(0~100)로 자른다', async () => {
    const { demand } = await load();
    serve({ ...HEALTHY, 'POST /api/v1/events': () => empty(204) });
    const fetchMock = vi.mocked(fetch);
    demand.trackUploadClick(250);
    await vi.waitFor(() => expect(fetchMock.mock.calls.some(([u]) => String(u) === '/api/v1/events')).toBe(true));
    const sent = fetchMock.mock.calls.filter(([u]) => String(u) === '/api/v1/events').map(([, init]) => JSON.parse(String(init?.body)) as unknown);
    expect(sent).toEqual([{ name: 'upload_click', path: '/app', props: { files: demand.MAX_UPLOAD_CLICK_FILES } }]);
    expect(demand.MAX_UPLOAD_CLICK_FILES).toBe(100);
  });

  it('이벤트 경로는 서버가 받는 화면 경로 네 가지뿐이고, 이벤트에는 워크스페이스 키를 붙이지 않는다', async () => {
    const { api, freeScans } = await load();
    const { EVENT_PATHS } = await import('./types');
    expect([...EVENT_PATHS].sort()).toEqual(['/', '/app', '/app/report', '/privacy']);
    freeScans.saveFreeScan({ workspace_id: 'ws-mine', access_key: 'k', created_at: '', readiness_overall: 1, findings: 0 });
    const calls = serve({ ...HEALTHY, 'POST /api/v1/events': () => empty(204) });
    for (const path of EVENT_PATHS) await expect(api.track({ name: 'landing_view', path })).resolves.toBe(true);
    const sent = posts(calls, '/api/v1/events');
    expect(sent).toHaveLength(EVENT_PATHS.length);
    for (const c of sent) expect(c.headers['X-Jettae-Workspace-Key']).toBeUndefined();
  });

  it('report_download 경로에는 워크스페이스 키가 붙지 않는다', async () => {
    const { workspaceKeyHeaders, freeScans } = await load();
    freeScans.saveFreeScan({ workspace_id: 'ws-mine', access_key: 'k', created_at: '', readiness_overall: 1, findings: 0 });
    expect(workspaceKeyHeaders('/app/report')).toEqual({});
  });
});

describe('서버 오류 분류', () => {
  it('JSON 본문이 있는 503은 API 오류 — 오프라인으로 바꾸지 않는다', async () => {
    const { api, getSourceMode, ApiError } = await load();
    serve({ ...HEALTHY, 'GET /api/v1/workspaces/ws-1/report': json({ detail: '서비스를 준비 중입니다.', code: 'service_unavailable' }, 503) });
    const err = await api.getReport('ws-1').catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err).toMatchObject({ status: 503, message: '서비스를 준비 중입니다.' });
    expect(getSourceMode()).toBe('api');
    expect(mockApi.getReport).not.toHaveBeenCalled();
  });

  it('개발 빌드: 빈 502(프록시가 서버에 닿지 못함)는 예전처럼 오프라인 예제로 바꾼다', async () => {
    const { api, getSourceMode } = await load();
    serve({ ...HEALTHY, 'GET /api/v1/workspaces/ws-1/report': empty(502) });
    await expect(api.getReport('ws-1')).resolves.toEqual({ from: 'mock' });
    expect(getSourceMode()).toBe('offline');
  });

  it('개발 빌드: 오프라인이면 사전 신청은 예전처럼 목업으로 간다', async () => {
    const { api, getSourceMode } = await load();
    serve({});
    await api.signup({ email: 'a@b.co', org_name: 'x', team_size: '1~10명', message: '', consent_privacy: true, consent_marketing: false, source: 'landing' });
    expect(getSourceMode()).toBe('offline');
    expect(mockApi.signup).toHaveBeenCalledOnce();
  });

  it('개발 빌드: 오프라인이면 세션은 예전처럼 역할 전환기의 역할(기본 팀 리드)을 쓴다', async () => {
    const { api, getSourceMode } = await load();
    serve({});
    await expect(api.session()).resolves.toMatchObject({ role: 'lead', anonymous: false, auth_mode: 'dev', can_switch_role: true });
    expect(getSourceMode()).toBe('offline');
  });

  it('공개 체험판 방문자의 403은 체험판 안내로 바꾼다', async () => {
    const { api, PUBLIC_TRIAL_MESSAGE } = await load();
    serve({
      ...HEALTHY,
      'GET /api/v1/me': json({ user: 'visitor', role: 'viewer', anonymous: true, auth_mode: 'public', can_switch_role: false }),
      'POST /api/v1/workspaces/ws-demo/rehearsals': json({ detail: '이 작업은 editor 이상만 할 수 있습니다.', code: 'forbidden' }, 403),
    });
    await api.session();
    await expect(api.runRehearsals('ws-demo', {})).rejects.toThrow(PUBLIC_TRIAL_MESSAGE);
  });

  it('팀 계정의 403은 서버 문구를 그대로 쓴다', async () => {
    const { api } = await load();
    serve({
      ...HEALTHY,
      'GET /api/v1/me': json({ user: 'kim', role: 'viewer', anonymous: false, auth_mode: 'token', can_switch_role: false }),
      'POST /api/v1/workspaces/ws-1/rehearsals': json({ detail: '이 작업은 editor 이상만 할 수 있습니다.', code: 'forbidden' }, 403),
    });
    await api.session();
    await expect(api.runRehearsals('ws-1', {})).rejects.toThrow('이 작업은 editor 이상만 할 수 있습니다.');
  });
});

describe('배포 빌드(PROD): 쓰기는 목업으로 보내지 않는다', () => {
  beforeEach(() => vi.stubEnv('PROD', true));

  const SIGNUP: SignupRequest = { email: 'a@b.co', org_name: 'x', team_size: '1~10명', message: '', consent_privacy: true, consent_marketing: false, source: 'landing' };

  it('서버에 닿지 않으면 사전 신청은 오류(다시 시도) — 목업에 저장했다고 하지 않는다', async () => {
    const { api, strictOffline, UnreachableError, RETRY_LATER_MESSAGE } = await load();
    expect(strictOffline()).toBe(true);
    serve({});
    const err = await api.signup(SIGNUP).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(UnreachableError);
    expect((err as Error).message).toBe(RETRY_LATER_MESSAGE);
    expect(mockApi.signup).not.toHaveBeenCalled();
  });

  it('무료 검진·예제 열기도 목업으로 가지 않는다', async () => {
    const { api } = await load();
    serve({});
    await expect(api.freeScan({ runbooks: [new File(['# 런북'], 'a.md')] })).rejects.toThrow();
    await expect(api.createDemo()).rejects.toThrow();
    expect(mockApi.createDemo).not.toHaveBeenCalled();
  });

  it('오프라인이어도 이벤트는 서버로 보낸다(닿지 않으면 false)', async () => {
    const { api, getSourceMode } = await load();
    const calls = serve({});
    await expect(api.track({ name: 'signup_click', path: '/' })).resolves.toBe(false);
    expect(posts(calls, '/api/v1/events')).toHaveLength(1);
    expect(mockApi.trackEvent).not.toHaveBeenCalled();
    expect(getSourceMode()).toBe('api');
  });

  it('/healthz가 한 번 실패해도 다시 확인해 살아 있으면 API를 쓴다', async () => {
    const { api, getSourceMode } = await load();
    let probes = 0;
    serve({
      'GET /healthz': () => (probes++ === 0 ? empty(502) : json({ status: 'ok' })),
      'GET /api/v1/workspaces': json([]),
      'POST /api/v1/signups': json({ ok: true }, 201),
    });
    await api.listWorkspaces();
    expect(probes).toBe(2);
    expect(getSourceMode()).toBe('api');
    await expect(api.signup(SIGNUP)).resolves.toEqual({ ok: true });
  });

  it('일시적인 빈 502는 /healthz를 다시 확인하고, 살아 있으면 오프라인으로 바꾸지 않는다', async () => {
    const { api, getSourceMode, UnreachableError } = await load();
    serve({ ...HEALTHY, 'GET /api/v1/workspaces/ws-1/report': empty(502) });
    await expect(api.getReport('ws-1')).rejects.toBeInstanceOf(UnreachableError);
    expect(getSourceMode()).toBe('api');
    expect(mockApi.getReport).not.toHaveBeenCalled();
  });

  it('오프라인이면 예제 워크스페이스만 예제 데이터로 보여 준다', async () => {
    const { api, getSourceMode, UnreachableError } = await load();
    serve({});
    await expect(api.getReport('ws-visitor-scan')).rejects.toBeInstanceOf(UnreachableError);
    expect(getSourceMode()).toBe('offline');
    expect(mockApi.getReport).not.toHaveBeenCalled();
    await expect(api.getReport('dmc-shop')).resolves.toEqual({ from: 'mock' });
  });

  it('서버 목록에서 본 예제 id는 오프라인에서도 예제로 본다', async () => {
    const { api, isDemoWorkspace } = await load();
    serve({ ...HEALTHY, 'GET /api/v1/workspaces': json([{ id: 'ws-server-demo', name: '별빛상점', is_demo: true }]) });
    await api.listWorkspaces();
    expect(isDemoWorkspace('ws-server-demo')).toBe(true);
    expect(isDemoWorkspace('ws-visitor-scan')).toBe(false);
  });

  it('서버에 닿지 않으면 세션을 꾸며 내지 않는다(개발용 세션·역할 전환기 없음)', async () => {
    const { api, UnreachableError, RETRY_LATER_MESSAGE } = await load();
    serve({});
    const err = await api.session().catch((e: unknown) => e);
    expect(err).toBeInstanceOf(UnreachableError);
    expect((err as Error).message).toBe(RETRY_LATER_MESSAGE);
  });

  it('개발 모드 서버(결선 시연)에 붙으면 /me의 역할 전환 가능 세션을 그대로 쓴다', async () => {
    const { api } = await load();
    const dev = { user: 'park.jaehyun', role: 'lead', anonymous: false, auth_mode: 'dev', can_switch_role: true };
    serve({ ...HEALTHY, 'GET /api/v1/me': json(dev) });
    await expect(api.session()).resolves.toEqual(dev);
  });

  it('승인·거절은 예제가 아닌 워크스페이스라면 오프라인에서 목업으로 대신하지 않는다', async () => {
    const { api, UnreachableError } = await load();
    serve({});
    await expect(api.approvePatch('ws-team', 'patch-dmc-01')).rejects.toBeInstanceOf(UnreachableError);
    await expect(api.rejectPatch('ws-team', 'patch-dmc-01', '이유')).rejects.toBeInstanceOf(UnreachableError);
    expect(mockApi.approvePatch).not.toHaveBeenCalled();
    expect(mockApi.rejectPatch).not.toHaveBeenCalled();
    // 예제 워크스페이스는 다른 화면 호출처럼 오프라인 예제로 흉내 낸다
    await api.approvePatch('dmc-shop', 'patch-dmc-01');
    await api.rejectPatch('dmc-shop', 'patch-dmc-02', '이유');
    expect(mockApi.approvePatch).toHaveBeenCalledOnce();
    expect(mockApi.rejectPatch).toHaveBeenCalledOnce();
  });

  it('오프라인 중 서버가 다시 답하면 API로 돌아온다', async () => {
    const { api, getSourceMode, setSourceMode } = await load();
    setSourceMode('offline');
    serve({ ...HEALTHY, 'POST /api/v1/signups': json({ ok: true }, 201) });
    await api.signup(SIGNUP);
    expect(getSourceMode()).toBe('api');
    expect(mockApi.signup).not.toHaveBeenCalled();
  });
});

describe('무료 검진 보관 기간 만료', () => {
  const stored = { workspace_id: 'ws-old', access_key: 'old-key', created_at: '2026-10-01T09:00:00+09:00', readiness_overall: 40, findings: 3 };

  it('키를 보낸 워크스페이스가 404면 목록에서 빼고 한 번만 알린다', async () => {
    const { api, freeScans, FREE_SCAN_EXPIRED_CODE } = await load();
    freeScans.saveFreeScan(stored);
    const seen: string[] = [];
    freeScans.onFreeScanExpired((ws) => seen.push(ws));
    const gone = json({ detail: '워크스페이스 ws-old을(를) 찾을 수 없습니다.', code: 'not_found' }, 404);
    const calls = serve({
      ...HEALTHY,
      'GET /api/v1/workspaces/ws-old/report': gone,
      'GET /api/v1/workspaces/ws-old/graph': gone,
      'GET /api/v1/workspaces/ws-old': gone,
    });
    const [a, b] = await Promise.all([api.getReport('ws-old').catch((e: unknown) => e), api.getGraph('ws-old').catch((e: unknown) => e)]);
    for (const err of [a, b]) expect(err).toMatchObject({ status: 404, code: FREE_SCAN_EXPIRED_CODE });
    expect((a as Error).message).toContain('보관 기간이 지났거나 서버에서 삭제된 무료 검진이라 목록에서 뺐습니다.');
    expect(seen).toEqual(['ws-old']);
    expect(freeScans.accessKeyFor('ws-old')).toBeNull();
    // 확인 요청에도 같은 키를 보낸다
    const probe = calls.find((c) => c.url === '/api/v1/workspaces/ws-old');
    expect(probe?.headers['X-Jettae-Workspace-Key']).toBe('old-key');
  });

  it('하위 항목만 404이고 워크스페이스가 살아 있으면 그대로 둔다', async () => {
    const { api, freeScans } = await load();
    freeScans.saveFreeScan(stored);
    serve({
      ...HEALTHY,
      'GET /api/v1/workspaces/ws-old/documents': json({ detail: '원문을 저장하지 않았습니다.', code: 'not_found' }, 404),
      'GET /api/v1/workspaces/ws-old': json({ id: 'ws-old', name: '무료 검진' }),
    });
    await expect(api.getDocuments('ws-old')).resolves.toEqual([]);
    expect(freeScans.accessKeyFor('ws-old')).toBe('old-key');
  });
});
