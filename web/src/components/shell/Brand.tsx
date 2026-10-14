import { clsx } from 'clsx';
import { useId } from 'react';

/** 제때옵스 로고 마크: 이어진 경로(그래프) + 마지막 한 칸이 끊긴 점 */
export function BrandMark({ size = 32, className }: { size?: number; className?: string }) {
  const gid = useId();
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" className={clsx('shrink-0', className)} aria-hidden>
      <defs>
        <linearGradient id={gid} x1="0" y1="0" x2="32" y2="32" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#8e95ff" />
          <stop offset="1" stopColor="#4f46e5" />
        </linearGradient>
      </defs>
      <rect width="32" height="32" rx="9" fill={`url(#${gid})`} />
      <path d="M8.5 21.5 L15.5 11 L21 17" stroke="#fff" strokeWidth="2.6" fill="none" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M21 17 L24.5 20.5" stroke="#ffb4b4" strokeWidth="2.6" strokeLinecap="round" strokeDasharray="1.5 3" />
      <circle cx="8.5" cy="21.5" r="3" fill="#fff" />
      <circle cx="15.5" cy="11" r="3" fill="#fff" />
      <circle cx="21" cy="17" r="2.6" fill="#fff" />
      <circle cx="25" cy="21" r="2.4" fill="none" stroke="#ffb4b4" strokeWidth="1.8" strokeDasharray="2 2" />
    </svg>
  );
}

export function BrandWordmark({ compact = false, className }: { compact?: boolean; className?: string }) {
  return (
    <span className={clsx('flex items-center gap-2.5', className)}>
      <BrandMark />
      {compact ? null : (
        <span className="flex flex-col leading-none">
          <span className="text-[17px] font-extrabold tracking-tight">제때옵스</span>
          <span className="mt-0.5 font-mono text-[10px] font-semibold tracking-[0.18em] text-fg-subtle">JETTAE OPS</span>
        </span>
      )}
    </span>
  );
}
