import { useQueryClient } from '@tanstack/react-query';
import { useEffect } from 'react';
import { Outlet, useParams } from 'react-router';
import { FREE_SCAN_EXPIRED_NOTICE } from '../../api/client';
import { qk, usePatches, useReport } from '../../api/hooks';
import { getElk } from '../../graph/layout';
import { onFreeScanExpired } from '../../lib/freeScans';
import { isPendingPatch } from '../../lib/patchSummary';
import { useToast } from '../ui/Toast';
import { OfflineBanner } from './OfflineBanner';
import { SideNav } from './SideNav';
import { TopBar } from './TopBar';
import { currentRehearsals } from '../../lib/rehearsals';

const LAST_WS = 'jettae.lastWorkspace';

export function lastWorkspace(): string | null {
  try {
    return window.localStorage.getItem(LAST_WS);
  } catch {
    return null;
  }
}

function forgetLastWorkspace(): void {
  try {
    window.localStorage.removeItem(LAST_WS);
  } catch {
    // 무시
  }
}

/**
 * 서버에서 사라진 무료 검진을 목록에서 뺐을 때 알리고 목록을 다시 읽는다.
 * 보관 기간이 지났을 수도, 서버(무료 DB)에서 지워졌을 수도 있어 이유를 하나로 단정하지 않는다.
 */
function useFreeScanExpiryNotice(): void {
  const toast = useToast();
  const qc = useQueryClient();
  const show = toast.show;
  useEffect(
    () =>
      onFreeScanExpired((expired) => {
        if (lastWorkspace() === expired) forgetLastWorkspace();
        show({ tone: 'info', title: FREE_SCAN_EXPIRED_NOTICE, description: '런북을 다시 올리면 새로 검진할 수 있습니다.' });
        void qc.invalidateQueries({ queryKey: qk.workspaces });
      }),
    [show, qc],
  );
}

/** 앱 레이아웃: 왼쪽 내비 + 상단 바 + 본문 */
export function AppShell() {
  const { ws } = useParams();
  const current = ws ?? lastWorkspace();
  useFreeScanExpiryNotice();
  // 그래프 배치 엔진(웹 워커)을 미리 띄워 첫 그래프가 빨리 뜨게 한다
  useEffect(() => {
    void getElk();
  }, []);
  useEffect(() => {
    if (!ws) return;
    try {
      window.localStorage.setItem(LAST_WS, ws);
    } catch {
      // 무시
    }
  }, [ws]);
  const patches = usePatches(current ?? undefined);
  const report = useReport(current ?? undefined);
  const pending = patches.data?.filter(isPendingPatch).length;
  const failing = report.data ? currentRehearsals(report.data).filter((r) => r.outcome === 'fail').length : undefined;
  return (
    <div className="flex h-full min-h-0 bg-bg">
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-50 focus:rounded-lg focus:bg-brand focus:px-3 focus:py-2 focus:text-brand-fg">
        본문으로 건너뛰기
      </a>
      <SideNav ws={current} badges={{ patches: pending, rehearsals: failing }} />
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar ws={ws ?? null} />
        <OfflineBanner />
        <main id="main" className="min-h-0 flex-1 overflow-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
