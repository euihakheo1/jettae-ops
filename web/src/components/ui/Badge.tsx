import { clsx } from 'clsx';
import {
  CircleAlert,
  CircleCheck,
  CircleDashed,
  CircleQuestionMark,
  CircleX,
  Info,
  OctagonAlert,
  TriangleAlert,
  type LucideIcon,
} from 'lucide-react';
import type { ReactNode } from 'react';
import type { ElementStatus, Severity } from '../../api/types';
import { ELEMENT_STATUS_LABEL, SEVERITY_LABEL } from '../../lib/labels';

export type Tone = 'danger' | 'danger-solid' | 'warn' | 'ok' | 'info' | 'brand' | 'muted' | 'neutral';

const TONE: Record<Tone, string> = {
  danger: 'bg-danger/12 text-danger border-danger/35',
  // 치명: 꽉 찬 빨강 + 흰 글자 (밝은 프로젝터 화면에서도 대비 유지)
  'danger-solid': 'bg-danger text-white border-danger',
  warn: 'bg-warn/12 text-warn border-warn/35',
  ok: 'bg-ok/12 text-ok border-ok/35',
  info: 'bg-info/12 text-info border-info/35',
  brand: 'bg-brand/14 text-brand border-brand/35',
  muted: 'bg-surface-2 text-fg-subtle border-line',
  neutral: 'bg-surface-2 text-fg-muted border-line',
};

export function Badge({
  tone = 'neutral',
  icon: Icon,
  children,
  className,
  size = 'md',
  title,
}: {
  tone?: Tone;
  icon?: LucideIcon | undefined;
  children: ReactNode;
  className?: string;
  size?: 'sm' | 'md' | 'lg';
  title?: string;
}) {
  return (
    <span
      title={title}
      className={clsx(
        'inline-flex shrink-0 items-center gap-1 rounded-full border font-semibold whitespace-nowrap',
        size === 'sm' && 'h-5 px-2 text-[11px]',
        size === 'md' && 'h-6 px-2.5 text-xs',
        size === 'lg' && 'h-8 px-3 text-sm',
        TONE[tone],
        className,
      )}
    >
      {Icon ? <Icon className={size === 'lg' ? 'size-4' : 'size-3.5'} aria-hidden strokeWidth={2.4} /> : null}
      {children}
    </span>
  );
}

export const SEVERITY_TONE: Record<Severity, Tone> = {
  critical: 'danger',
  high: 'danger',
  medium: 'warn',
  low: 'info',
  info: 'muted',
};

export const SEVERITY_ICON: Record<Severity, LucideIcon> = {
  critical: OctagonAlert,
  high: TriangleAlert,
  medium: CircleAlert,
  low: Info,
  info: Info,
};

/** 심각도 칩: 아이콘 + 한국어 텍스트 (색만으로 전달하지 않음) */
export function SeverityChip({ severity, size = 'md', className }: { severity: Severity; size?: 'sm' | 'md' | 'lg'; className?: string }) {
  return (
    <Badge tone={severity === 'critical' ? 'danger-solid' : SEVERITY_TONE[severity]} icon={SEVERITY_ICON[severity]} size={size} {...(className ? { className } : {})}>
      {SEVERITY_LABEL[severity]}
    </Badge>
  );
}

export const ELEMENT_STATUS_TONE: Record<ElementStatus, Tone> = {
  pass: 'ok',
  partial: 'warn',
  fail: 'danger',
  unknown: 'muted',
};

export const ELEMENT_STATUS_ICON: Record<ElementStatus, LucideIcon> = {
  pass: CircleCheck,
  partial: CircleDashed,
  fail: CircleX,
  unknown: CircleQuestionMark,
};

export function ElementStatusBadge({ status, size = 'sm' }: { status: ElementStatus; size?: 'sm' | 'md' }) {
  return (
    <Badge tone={ELEMENT_STATUS_TONE[status]} icon={ELEMENT_STATUS_ICON[status]} size={size}>
      {ELEMENT_STATUS_LABEL[status]}
    </Badge>
  );
}
