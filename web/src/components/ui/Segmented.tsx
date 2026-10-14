import { clsx } from 'clsx';
import type { LucideIcon } from 'lucide-react';

export interface SegmentOption<T extends string> {
  value: T;
  label: string;
  icon?: LucideIcon;
  count?: number;
}

/** 탭/토글 겸용 세그먼트 컨트롤 (role=tablist 또는 radiogroup) */
export function Segmented<T extends string>({
  options,
  value,
  onChange,
  label,
  size = 'md',
  as = 'tabs',
  className,
}: {
  options: SegmentOption<T>[];
  value: T;
  onChange: (v: T) => void;
  label: string;
  size?: 'sm' | 'md';
  as?: 'tabs' | 'radio';
  className?: string;
}) {
  return (
    <div
      role={as === 'tabs' ? 'tablist' : 'radiogroup'}
      aria-label={label}
      className={clsx('inline-flex items-center gap-1 rounded-xl border border-line bg-surface-2 p-1', className)}
    >
      {options.map((o) => {
        const active = o.value === value;
        const Icon = o.icon;
        return (
          <button
            key={o.value}
            type="button"
            role={as === 'tabs' ? 'tab' : 'radio'}
            aria-selected={as === 'tabs' ? active : undefined}
            aria-checked={as === 'radio' ? active : undefined}
            onClick={() => onChange(o.value)}
            className={clsx(
              'inline-flex items-center gap-1.5 rounded-lg font-medium whitespace-nowrap transition-colors',
              size === 'sm' ? 'h-7 px-2.5 text-[13px]' : 'h-8 px-3 text-sm',
              active ? 'bg-surface text-fg shadow-sm ring-1 ring-line-strong' : 'text-fg-muted hover:text-fg',
            )}
          >
            {Icon ? <Icon className="size-4" aria-hidden /> : null}
            {o.label}
            {o.count !== undefined ? (
              <span className={clsx('tnum rounded-md px-1.5 text-xs', active ? 'bg-surface-3' : 'bg-surface-3/60')}>{o.count}</span>
            ) : null}
          </button>
        );
      })}
    </div>
  );
}
