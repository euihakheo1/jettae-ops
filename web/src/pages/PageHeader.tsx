import type { LucideIcon } from 'lucide-react';
import type { ReactNode } from 'react';

export function PageHeader({ icon: Icon, title, description, actions }: { icon: LucideIcon; title: string; description?: ReactNode; actions?: ReactNode }) {
  return (
    <header className="flex flex-wrap items-start gap-4">
      <div className="grid size-12 shrink-0 place-items-center rounded-2xl border border-line bg-surface-2 text-brand">
        <Icon className="size-6" aria-hidden />
      </div>
      <div className="min-w-0 flex-1">
        <h1 className="text-[28px] leading-tight font-extrabold tracking-tight">{title}</h1>
        {description ? <p className="mt-1 max-w-[60rem] text-[16px] leading-relaxed text-balance text-fg-muted">{description}</p> : null}
      </div>
      {actions ? <div className="flex shrink-0 items-center gap-2">{actions}</div> : null}
    </header>
  );
}
