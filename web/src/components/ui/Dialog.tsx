import { X } from 'lucide-react';
import { useEffect, useId, useRef, type ReactNode } from 'react';

/** 네이티브 <dialog> 기반 모달 — 포커스 가두기·Esc 닫기는 브라우저가 처리한다. */
export function Dialog({
  open,
  onClose,
  title,
  description,
  children,
  footer,
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  description?: ReactNode;
  children?: ReactNode;
  footer?: ReactNode;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const titleId = useId();
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) {
      if (typeof d.showModal === 'function') d.showModal();
      else d.setAttribute('open', '');
    } else if (!open && d.open) {
      if (typeof d.close === 'function') d.close();
      else d.removeAttribute('open');
    }
  }, [open]);
  return (
    <dialog
      ref={ref}
      aria-labelledby={titleId}
      onClose={onClose}
      onCancel={(e) => {
        e.preventDefault();
        onClose();
      }}
      className="m-auto w-[min(560px,92vw)] rounded-2xl border border-line-strong bg-surface p-0 text-fg shadow-2xl backdrop:bg-black/60 backdrop:backdrop-blur-sm"
    >
      {open ? (
        <div className="flex flex-col gap-4 p-6">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h2 id={titleId} className="text-xl font-bold">
                {title}
              </h2>
              {description ? <div className="mt-1 text-[15px] text-fg-muted">{description}</div> : null}
            </div>
            <button
              type="button"
              onClick={onClose}
              aria-label="닫기"
              className="grid size-9 place-items-center rounded-lg text-fg-subtle hover:bg-surface-2 hover:text-fg"
            >
              <X className="size-5" aria-hidden />
            </button>
          </div>
          {children}
          {footer ? <div className="flex justify-end gap-2 pt-2">{footer}</div> : null}
        </div>
      ) : null}
    </dialog>
  );
}
