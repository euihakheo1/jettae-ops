import { clsx } from 'clsx';
import { CircleCheck, CircleX, Info, X } from 'lucide-react';
import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from 'react';

type ToastTone = 'success' | 'error' | 'info';

interface ToastItem {
  id: number;
  tone: ToastTone;
  title: string;
  description?: ReactNode;
}

interface ToastApi {
  show: (t: Omit<ToastItem, 'id'>) => void;
}

const ToastContext = createContext<ToastApi | null>(null);

export function useToast(): ToastApi {
  const ctx = useContext(ToastContext);
  if (!ctx) return { show: () => undefined };
  return ctx;
}

const ICON = { success: CircleCheck, error: CircleX, info: Info } as const;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([]);
  const seq = useRef(0);
  const dismiss = useCallback((id: number) => setItems((xs) => xs.filter((x) => x.id !== id)), []);
  const show = useCallback(
    (t: Omit<ToastItem, 'id'>) => {
      seq.current += 1;
      const id = seq.current;
      setItems((xs) => [...xs.slice(-3), { ...t, id }]);
      window.setTimeout(() => dismiss(id), t.tone === 'error' ? 8000 : 6000);
    },
    [dismiss],
  );
  const api = useMemo(() => ({ show }), [show]);
  return (
    <ToastContext.Provider value={api}>
      {children}
      <div aria-live="polite" className="pointer-events-none fixed right-5 bottom-5 z-[100] flex w-[min(420px,92vw)] flex-col gap-3">
        {items.map((t) => {
          const Icon = ICON[t.tone];
          return (
            <div
              key={t.id}
              role={t.tone === 'error' ? 'alert' : 'status'}
              className={clsx(
                'pointer-events-auto flex animate-slide-in items-start gap-3 rounded-2xl border bg-surface-2 p-4 shadow-2xl',
                t.tone === 'success' && 'border-ok/50',
                t.tone === 'error' && 'border-danger/50',
                t.tone === 'info' && 'border-line-strong',
              )}
            >
              <Icon
                className={clsx(
                  'mt-0.5 size-6 shrink-0',
                  t.tone === 'success' && 'text-ok',
                  t.tone === 'error' && 'text-danger',
                  t.tone === 'info' && 'text-info',
                )}
                aria-hidden
              />
              <div className="min-w-0 flex-1">
                <p className="text-base font-bold">{t.title}</p>
                {t.description ? <div className="mt-0.5 text-sm text-fg-muted">{t.description}</div> : null}
              </div>
              <button
                type="button"
                aria-label="알림 닫기"
                onClick={() => dismiss(t.id)}
                className="grid size-7 place-items-center rounded-md text-fg-subtle hover:bg-surface-3 hover:text-fg"
              >
                <X className="size-4" aria-hidden />
              </button>
            </div>
          );
        })}
      </div>
    </ToastContext.Provider>
  );
}
