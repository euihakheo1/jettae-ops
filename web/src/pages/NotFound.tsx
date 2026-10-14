import { Compass, TriangleAlert } from 'lucide-react';
import { Link, isRouteErrorResponse, useRouteError } from 'react-router';
import { BrandWordmark } from '../components/shell/Brand';

export function NotFound() {
  return (
    <div className="grid min-h-full place-items-center p-8">
      <div className="card flex max-w-md flex-col items-center gap-4 p-10 text-center">
        <BrandWordmark />
        <Compass className="size-10 text-fg-subtle" aria-hidden />
        <h1 className="text-2xl font-bold">페이지를 찾을 수 없습니다</h1>
        <p className="text-fg-muted">주소가 바뀌었거나 없는 화면입니다.</p>
        <div className="flex gap-2">
          <Link to="/" className="rounded-xl border border-line bg-surface-2 px-4 py-2 font-semibold hover:bg-surface-3">
            소개 페이지
          </Link>
          <Link to="/app" className="rounded-xl bg-brand px-4 py-2 font-semibold text-brand-fg">
            워크스페이스로
          </Link>
        </div>
      </div>
    </div>
  );
}

export function RouteError() {
  const err = useRouteError();
  const message = isRouteErrorResponse(err) ? `${err.status} ${err.statusText}` : err instanceof Error ? err.message : '알 수 없는 오류';
  return (
    <div role="alert" className="grid min-h-full place-items-center p-8">
      <div className="card flex max-w-lg flex-col items-center gap-4 p-10 text-center">
        <TriangleAlert className="size-10 text-danger" aria-hidden />
        <h1 className="text-2xl font-bold">화면을 그리는 중 문제가 생겼습니다</h1>
        <p className="font-mono text-sm break-all text-fg-muted">{message}</p>
        <div className="flex gap-2">
          <button type="button" onClick={() => window.location.reload()} className="rounded-xl border border-line bg-surface-2 px-4 py-2 font-semibold">
            새로고침
          </button>
          <Link to="/app" className="rounded-xl bg-brand px-4 py-2 font-semibold text-brand-fg">
            워크스페이스로
          </Link>
        </div>
      </div>
    </div>
  );
}
