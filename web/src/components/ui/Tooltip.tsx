import { clsx } from 'clsx';
import { useCallback, useId, useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import { createPortal } from 'react-dom';

type Side = 'top' | 'bottom' | 'right' | 'left';

/**
 * 가벼운 툴팁 — 마우스 오버와 키보드 포커스 모두에서 보인다.
 * 화면 표시는 document.body 포털(스크롤 영역에 잘리지 않음), 보조기기용 텍스트는 제자리(role=tooltip)에 둔다.
 */
export function Tooltip({
  content,
  children,
  side = 'top',
  className,
  focusable = false,
  wide = false,
}: {
  content: ReactNode;
  children: ReactNode;
  side?: Side;
  className?: string;
  /** 자식이 버튼이 아닐 때 키보드로 열 수 있도록 tabIndex=0 */
  focusable?: boolean;
  wide?: boolean;
}) {
  const id = useId();
  const ref = useRef<HTMLSpanElement>(null);
  const tip = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ left: number; top: number; side: Side } | null>(null);

  const show = useCallback(() => setOpen(true), []);
  const hide = useCallback(() => {
    setOpen(false);
    setPos(null);
  }, []);

  useLayoutEffect(() => {
    if (!open || !ref.current || !tip.current) return;
    const r = ref.current.getBoundingClientRect();
    const t = tip.current.getBoundingClientRect();
    const gap = 8;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    let s: Side = side;
    if (s === 'top' && r.top - t.height - gap < 4) s = 'bottom';
    else if (s === 'bottom' && r.bottom + t.height + gap > vh - 4) s = 'top';
    let left = 0;
    let top = 0;
    if (s === 'top' || s === 'bottom') {
      left = r.left + r.width / 2 - t.width / 2;
      top = s === 'top' ? r.top - t.height - gap : r.bottom + gap;
    } else {
      top = r.top + r.height / 2 - t.height / 2;
      left = s === 'right' ? r.right + gap : r.left - t.width - gap;
    }
    left = Math.max(8, Math.min(left, vw - t.width - 8));
    top = Math.max(8, Math.min(top, vh - t.height - 8));
    setPos({ left, top, side: s });
  }, [open, side, content]);

  return (
    <span
      ref={ref}
      className={clsx('tt-wrap relative inline-flex', className)}
      aria-describedby={id}
      onMouseEnter={show}
      onMouseLeave={hide}
      onFocus={show}
      onBlur={hide}
      onKeyDown={(e) => {
        if (e.key === 'Escape') hide();
      }}
      {...(focusable ? { tabIndex: 0 } : {})}
    >
      {children}
      <span role="tooltip" id={id} className="sr-only">
        {content}
      </span>
      {open && typeof document !== 'undefined'
        ? createPortal(
            <div
              ref={tip}
              aria-hidden
              style={{ left: pos?.left ?? -9999, top: pos?.top ?? -9999 }}
              className={clsx(
                'pointer-events-none fixed z-[200] rounded-lg border border-line-strong bg-surface-3 px-3 py-2 text-left text-[13px] leading-snug font-normal text-fg shadow-xl transition-opacity duration-100',
                pos ? 'opacity-100' : 'opacity-0',
                wide ? 'w-80' : 'w-max max-w-72',
              )}
            >
              {content}
            </div>,
            document.body,
          )
        : null}
    </span>
  );
}

/**
 * 꺼 둔 버튼에 이유를 붙인다(이유가 없으면 그대로 둔다).
 * 비활성 버튼은 마우스·포커스 이벤트를 받지 않으므로 감싼 요소가 툴팁을 열고 키보드 포커스를 받는다.
 */
export function DisabledReason({ reason, children }: { reason: string | null | undefined; children: ReactNode }) {
  if (!reason) return <>{children}</>;
  return (
    <Tooltip content={reason} focusable>
      {children}
    </Tooltip>
  );
}
