import { createBrowserRouter } from 'react-router';
import { AppShell } from '../components/shell/AppShell';
import { NotFound, RouteError } from '../pages/NotFound';
import { RootLayout } from './RootLayout';
import { RouteLoading } from './RouteLoading';

/** 모든 화면은 URL(경로 + 쿼리)로 바로 열 수 있다 — 스크린숏·시연 링크용 */
export const router = createBrowserRouter([
  {
    Component: RootLayout,
    HydrateFallback: RouteLoading,
    errorElement: <RouteError />,
    children: [
      { path: '/', lazy: () => import('../pages/Landing').then((m) => ({ Component: m.Landing })) },
      { path: '/privacy', lazy: () => import('../pages/Privacy').then((m) => ({ Component: m.Privacy })) },
      {
        path: '/app',
        Component: AppShell,
        errorElement: <RouteError />,
        children: [
          { index: true, lazy: () => import('../pages/WorkspaceHome').then((m) => ({ Component: m.WorkspaceHome })) },
          {
            path: 'w/:ws',
            children: [
              { index: true, lazy: () => import('../pages/Dashboard').then((m) => ({ Component: m.Dashboard })) },
              { path: 'rehearsals', lazy: () => import('../pages/Rehearsals').then((m) => ({ Component: m.Rehearsals })) },
              { path: 'patches', lazy: () => import('../pages/Patches').then((m) => ({ Component: m.Patches })) },
              { path: 'versions', lazy: () => import('../pages/Versions').then((m) => ({ Component: m.Versions })) },
              { path: 'report', lazy: () => import('../pages/Report').then((m) => ({ Component: m.Report })) },
              { path: 'trust', lazy: () => import('../pages/Trust').then((m) => ({ Component: m.Trust })) },
            ],
          },
        ],
      },
      { path: '*', Component: NotFound },
    ],
  },
]);
