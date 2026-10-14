import { useQueryClient } from '@tanstack/react-query';
import { useEffect } from 'react';
import { Outlet, useSearchParams } from 'react-router';
import { loadMock } from '../api/client';
import { useSourceMode } from '../api/source';
import { ROLES, roleStore, themeStore, THEMES, type Theme } from './prefs';
import type { Role } from '../api/types';

/**
 * 스크린숏·시연용 URL 옵션 (어느 화면에서나):
 *   ?theme=projector|dark   ?role=viewer|lead|editor|admin   ?mock=after (목업/오프라인일 때 승인 이후 상태)
 */
export function RootLayout() {
  const [params] = useSearchParams();
  const mode = useSourceMode();
  const qc = useQueryClient();
  const theme = params.get('theme');
  const role = params.get('role');
  const preset = params.get('mock');

  useEffect(() => {
    if (theme && (THEMES as readonly string[]).includes(theme)) themeStore.set(theme as Theme);
  }, [theme]);

  useEffect(() => {
    if (role && (ROLES as readonly string[]).includes(role)) roleStore.set(role as Role);
  }, [role]);

  useEffect(() => {
    if (!preset || mode === 'api') return;
    let cancelled = false;
    void (async () => {
      const m = await loadMock();
      await m.applyPreset(preset);
      if (!cancelled) await qc.invalidateQueries();
    })();
    return () => {
      cancelled = true;
    };
  }, [preset, mode, qc]);

  return <Outlet />;
}
