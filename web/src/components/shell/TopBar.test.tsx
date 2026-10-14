import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { SessionInfo } from '../../api/types';

const json = (body: unknown, status = 200): Response =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });

const DEV_SESSION: SessionInfo = { user: 'park.jaehyun', role: 'lead', anonymous: false, auth_mode: 'dev', can_switch_role: true };
const VISITOR: SessionInfo = { user: 'visitor', role: 'viewer', anonymous: true, auth_mode: 'public', can_switch_role: false };

/** GET /me 의 답을 테스트가 정한 때에 돌려준다 */
function serveWithPendingMe(): (session: SessionInfo) => void {
  let answer: (res: Response) => void = () => undefined;
  const me = new Promise<Response>((resolve) => {
    answer = resolve;
  });
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input).split('?')[0];
      if (url === '/healthz') return json({ status: 'ok' });
      if (url === '/api/v1/workspaces') return json([]);
      if (url === '/api/v1/me') return me;
      throw new TypeError('Failed to fetch');
    }),
  );
  return (session) => answer(json(session));
}

async function renderTopBar(): Promise<void> {
  vi.resetModules();
  const { TopBar } = await import('./TopBar');
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={qc}>
      <MemoryRouter>
        <TopBar ws={null} />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

const ROLE_SWITCH = { name: '역할 전환 (데모용 인증)' };

beforeEach(() => localStorage.clear());

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe('TopBar 역할 표시', () => {
  it('배포 빌드 + 개발 모드 서버(결선 시연): /me를 기다리는 동안은 보기 전용, 답이 오면 역할 전환기와 팀 리드가 나온다', async () => {
    vi.stubEnv('PROD', true);
    const answerMe = serveWithPendingMe();
    await renderTopBar();
    expect(await screen.findByText('역할 확인 중 · 보기 전용')).toBeInTheDocument();
    expect(screen.queryByRole('radiogroup', ROLE_SWITCH)).toBeNull();

    answerMe(DEV_SESSION);
    expect(await screen.findByRole('radiogroup', ROLE_SWITCH)).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: '팀 리드' })).toHaveAttribute('aria-checked', 'true');
    expect(screen.getByText('박재현 팀장 · 승인 가능')).toBeInTheDocument();
    expect(screen.queryByText('역할 확인 중 · 보기 전용')).toBeNull();
  });

  it('배포 빌드 + 공개 체험판: 역할 전환기는 끝까지 보이지 않고 서버 역할을 보여 준다', async () => {
    vi.stubEnv('PROD', true);
    const answerMe = serveWithPendingMe();
    await renderTopBar();
    expect(await screen.findByText('역할 확인 중 · 보기 전용')).toBeInTheDocument();
    answerMe(VISITOR);
    expect(await screen.findByText('공개 체험판 · 보기 전용')).toBeInTheDocument();
    expect(screen.queryByRole('radiogroup', ROLE_SWITCH)).toBeNull();
  });

  it('개발 빌드: /me를 기다리는 동안에도 예전처럼 역할 전환기(기본 팀 리드)를 보여 준다', async () => {
    serveWithPendingMe();
    await renderTopBar();
    expect(await screen.findByRole('radiogroup', ROLE_SWITCH)).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: '팀 리드' })).toHaveAttribute('aria-checked', 'true');
    expect(screen.queryByText('역할 확인 중 · 보기 전용')).toBeNull();
  });
});
