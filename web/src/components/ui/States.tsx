import { clsx } from 'clsx';
import { CircleAlert, Inbox, RefreshCw, type LucideIcon } from 'lucide-react';
import type { ReactNode } from 'react';
import { errorMessage } from '../../api/errors';
import { Button } from './Button';

export function Skeleton({ className }: { className?: string }) {
  return <div aria-hidden className={clsx('animate-pulse rounded-lg bg-surface-3/70', className)} />;
}

export function LoadingBlock({ label = '불러오는 중…', className }: { label?: string; className?: string }) {
  return (
    <div role="status" aria-live="polite" className={clsx('space-y-3 p-4', className)}>
      <span className="sr-only">{label}</span>
      <Skeleton className="h-6 w-1/3" />
      <Skeleton className="h-4 w-2/3" />
      <Skeleton className="h-4 w-1/2" />
      <Skeleton className="h-24 w-full" />
    </div>
  );
}

export function EmptyState({
  icon: Icon = Inbox,
  title,
  children,
  action,
  className,
}: {
  icon?: LucideIcon;
  title: string;
  children?: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div className={clsx('flex flex-col items-center justify-center gap-3 px-6 py-12 text-center', className)}>
      <div className="grid size-12 place-items-center rounded-2xl border border-line bg-surface-2 text-fg-subtle">
        <Icon className="size-6" aria-hidden />
      </div>
      <h3 className="text-lg font-semibold text-fg">{title}</h3>
      {children ? <div className="max-w-md text-[15px] text-fg-muted">{children}</div> : null}
      {action}
    </div>
  );
}

export function ErrorState({
  error,
  title = '데이터를 불러오지 못했습니다',
  onRetry,
  className,
}: {
  error: unknown;
  title?: string;
  onRetry?: () => void;
  className?: string;
}) {
  return (
    <div role="alert" className={clsx('flex flex-col items-center justify-center gap-3 px-6 py-12 text-center', className)}>
      <div className="grid size-12 place-items-center rounded-2xl border border-danger/40 bg-danger/10 text-danger">
        <CircleAlert className="size-6" aria-hidden />
      </div>
      <h3 className="text-lg font-semibold">{title}</h3>
      <p className="max-w-md text-[15px] text-fg-muted">{errorMessage(error)}</p>
      {onRetry ? (
        <Button icon={<RefreshCw className="size-4" aria-hidden />} onClick={onRetry}>
          다시 시도
        </Button>
      ) : null}
    </div>
  );
}
