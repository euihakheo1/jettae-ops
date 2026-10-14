import { clsx } from 'clsx';
import { ChevronDown, Ghost, Map as MapIcon, UserX } from 'lucide-react';
import { useState } from 'react';
import { KIND_ICON } from '../components/domain/icons';

const NODE_ITEMS = [
  { icon: KIND_ICON.trigger, label: '감지 신호', cls: 'text-warn' },
  { icon: KIND_ICON.service, label: '서비스 (등급·점수)', cls: 'text-brand' },
  { icon: KIND_ICON.person, label: '사람', cls: 'text-fg-muted' },
  { icon: KIND_ICON.role, label: '역할', cls: 'text-fg-muted' },
  { icon: KIND_ICON.action, label: '조치 (롤백 등)', cls: 'text-ok' },
  { icon: KIND_ICON.channel, label: '연락 채널', cls: 'text-info' },
  { icon: KIND_ICON.obligation, label: '신고 의무', cls: 'text-brand' },
] as const;

/** 그래프 범례 — 접을 수 있다 */
export function Legend({ className, defaultOpen = true }: { className?: string; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className={clsx('rounded-xl border border-line bg-surface/95 text-[13px] shadow-lg backdrop-blur', open ? 'w-60' : 'w-auto', className)}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center gap-2 px-3 py-1.5 font-semibold text-fg-muted hover:text-fg"
      >
        <MapIcon className="size-4 text-fg-subtle" aria-hidden />
        범례
        <ChevronDown className={clsx('ml-auto size-4 transition-transform', !open && '-rotate-90')} aria-hidden />
      </button>
      {open ? (
        <div className="space-y-3 border-t border-line px-3 py-3">
          <ul className="grid grid-cols-1 gap-1.5">
            {NODE_ITEMS.map(({ icon: Icon, label, cls }) => (
              <li key={label} className="flex items-center gap-2 text-fg-muted">
                <Icon className={clsx('size-4', cls)} aria-hidden />
                {label}
              </li>
            ))}
          </ul>
          <ul className="space-y-1.5 border-t border-line pt-3">
            <li className="flex items-center gap-2">
              <span className="grid size-5 place-items-center rounded-full bg-danger text-white">
                <UserX className="size-3" aria-hidden />
              </span>
              <span className="text-danger line-through decoration-2">퇴사자</span>
              <span className="text-fg-subtle">· 빨강 + 취소선</span>
            </li>
            <li className="flex items-center gap-2">
              <span className="grid size-5 place-items-center rounded-full border-2 border-dashed border-ghost text-ghost">
                <Ghost className="size-3" aria-hidden />
              </span>
              <span className="text-ghost">빠진 필수 요소</span>
              <span className="text-fg-subtle">· 점선</span>
            </li>
            <li className="flex items-center gap-2">
              <span className="h-4 w-5 rounded border-2 border-dashed border-warn" aria-hidden />
              <span className="text-warn">판단 보류</span>
              <span className="text-fg-subtle">· 확인 필요</span>
            </li>
          </ul>
          <ul className="space-y-1.5 border-t border-line pt-3">
            <EdgeRow color="var(--danger)" dashed label="끊긴 연결 (치명·높음)" />
            <EdgeRow color="var(--warn)" label="주의 (보통)" />
            <EdgeRow color="var(--edge)" label="정상 연결" />
            <EdgeRow color="var(--brand)" label="리허설 경로" thick />
          </ul>
        </div>
      ) : null}
    </div>
  );
}

function EdgeRow({ color, label, dashed = false, thick = false }: { color: string; label: string; dashed?: boolean; thick?: boolean }) {
  return (
    <li className="flex items-center gap-2 text-fg-muted">
      <svg width="28" height="8" aria-hidden>
        <line x1="0" y1="4" x2="28" y2="4" stroke={color} strokeWidth={thick ? 3.5 : 2.5} strokeDasharray={dashed ? '6 4' : undefined} />
      </svg>
      {label}
    </li>
  );
}
