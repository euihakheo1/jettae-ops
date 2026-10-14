import { clsx } from 'clsx';
import { CircleAlert, CircleCheck, TriangleAlert } from 'lucide-react';
import { formatScore, SCORE_BAND_LABEL, scoreBand, type ScoreBand } from '../../lib/format';

export const BAND_TEXT: Record<ScoreBand, string> = {
  red: 'text-danger',
  amber: 'text-warn',
  green: 'text-ok',
};

export const BAND_BG: Record<ScoreBand, string> = {
  red: 'bg-danger',
  amber: 'bg-warn',
  green: 'bg-ok',
};

const BAND_ICON = { red: CircleAlert, amber: TriangleAlert, green: CircleCheck } as const;

/** 점수 알약: 숫자 + 띠 아이콘 (색만으로 전달하지 않음) */
export function ScorePill({ score, className, size = 'md' }: { score: number; className?: string; size?: 'sm' | 'md' }) {
  const band = scoreBand(score);
  const Icon = BAND_ICON[band];
  return (
    <span
      className={clsx(
        'tnum inline-flex items-center gap-1 rounded-full border font-bold',
        size === 'sm' ? 'h-5 px-1.5 text-[11px]' : 'h-6 px-2 text-[13px]',
        band === 'red' && 'border-danger/40 bg-danger/12 text-danger',
        band === 'amber' && 'border-warn/40 bg-warn/12 text-warn',
        band === 'green' && 'border-ok/40 bg-ok/12 text-ok',
        className,
      )}
      title={`준비도 ${formatScore(score)}점 · ${SCORE_BAND_LABEL[band]}`}
    >
      <Icon className={size === 'sm' ? 'size-3' : 'size-3.5'} aria-hidden strokeWidth={2.6} />
      {formatScore(score)}
    </span>
  );
}

/** 가로 점수 막대 */
export function ScoreBar({ score, className, label }: { score: number; className?: string; label?: string }) {
  const band = scoreBand(score);
  return (
    <div
      role="meter"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={score}
      aria-label={label ?? `준비도 ${formatScore(score)}점`}
      className={clsx('h-1.5 w-full overflow-hidden rounded-full bg-surface-3', className)}
    >
      <div className={clsx('h-full rounded-full transition-[width] duration-700', BAND_BG[band])} style={{ width: `${Math.max(2, score)}%` }} />
    </div>
  );
}

/** 원형 게이지 (KPI) */
export function ScoreRing({ score, size = 112, stroke = 10 }: { score: number; size?: number; stroke?: number }) {
  const band = scoreBand(score);
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const offset = c * (1 - Math.min(100, Math.max(0, score)) / 100);
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden className="shrink-0 -rotate-90">
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--surface-3)" strokeWidth={stroke} />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        stroke={band === 'red' ? 'var(--danger)' : band === 'amber' ? 'var(--warn)' : 'var(--ok)'}
        strokeWidth={stroke}
        strokeLinecap="round"
        strokeDasharray={c}
        strokeDashoffset={offset}
        style={{ transition: 'stroke-dashoffset 900ms cubic-bezier(.2,.8,.2,1)' }}
      />
    </svg>
  );
}
