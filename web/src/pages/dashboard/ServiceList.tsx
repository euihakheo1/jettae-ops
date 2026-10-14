import { clsx } from 'clsx';
import { Network } from 'lucide-react';
import type { Finding, ReadinessReport, ServiceReadiness } from '../../api/types';
import { ElementDots } from '../../components/domain/ElementDots';
import { BAND_TEXT, ScoreBar } from '../../components/domain/Score';
import { formatScore, scoreBand } from '../../lib/format';

const TIER_LABEL: Record<number, string> = { 1: '1등급 · 핵심', 2: '2등급', 3: '3등급' };

/** 왼쪽 서비스 목록: 등급별, 점수 낮은 순. 누르면 그래프가 그 서비스에 집중한다. */
export function ServiceList({
  readiness,
  findings,
  focusId,
  onFocus,
  onShowAll,
  allActive,
}: {
  readiness: ReadinessReport;
  findings: readonly Finding[];
  focusId: string | null;
  onFocus: (serviceId: string) => void;
  onShowAll: () => void;
  allActive: boolean;
}) {
  const tiers = [...new Set(readiness.services.map((s) => s.tier))].sort((a, b) => a - b);
  const count = (sid: string) => findings.filter((f) => f.service_id === sid).length;
  return (
    <nav aria-label="서비스 목록" className="flex h-full min-h-0 flex-col">
      <div className="flex items-center justify-between px-4 pt-4 pb-2">
        <h2 className="text-[15px] font-bold">서비스 {readiness.services.length}</h2>
        <button
          type="button"
          onClick={onShowAll}
          aria-pressed={allActive}
          className={clsx(
            'inline-flex items-center gap-1.5 rounded-lg px-2 py-1 text-[13px] font-semibold',
            allActive ? 'bg-brand/15 text-brand' : 'text-fg-muted hover:bg-surface-2 hover:text-fg',
          )}
        >
          <Network className="size-4" aria-hidden />
          전체 보기
        </button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-4">
        {tiers.map((tier) => {
          const services = readiness.services
            .filter((s) => s.tier === tier)
            .sort((a, b) => a.score - b.score || (a.service_id < b.service_id ? -1 : 1));
          return (
            <section key={tier} className="mt-2">
              <h3 className="px-2 pb-1 text-[12px] font-semibold tracking-wide text-fg-subtle">{TIER_LABEL[tier] ?? `${tier}등급`}</h3>
              <ul className="space-y-1">
                {services.map((s) => (
                  <li key={s.service_id}>
                    <ServiceRow s={s} active={!allActive && s.service_id === focusId} findings={count(s.service_id)} onClick={() => onFocus(s.service_id)} />
                  </li>
                ))}
              </ul>
            </section>
          );
        })}
      </div>
    </nav>
  );
}

function ServiceRow({ s, active, findings, onClick }: { s: ServiceReadiness; active: boolean; findings: number; onClick: () => void }) {
  const band = scoreBand(s.score);
  return (
    <div
      className={clsx(
        'group relative rounded-xl border px-3 py-2.5 transition-colors',
        active ? 'border-brand/60 bg-brand/10' : 'border-transparent hover:border-line hover:bg-surface-2',
      )}
    >
      <button type="button" onClick={onClick} aria-current={active ? 'true' : undefined} className="absolute inset-0 rounded-xl" aria-label={`${s.service_name} 그래프 보기 — 준비도 ${formatScore(s.score)}점, 끊김 ${findings}건`} />
      <div className="pointer-events-none flex items-center gap-2">
        <span className="min-w-0 flex-1 truncate text-[15px] font-semibold">{s.service_name}</span>
        <span className="rounded border border-line px-1 font-mono text-[10.5px] font-bold text-fg-subtle">T{s.tier}</span>
        <span className={clsx('tnum w-11 text-right text-[16px] font-extrabold', BAND_TEXT[band])}>{formatScore(s.score)}</span>
      </div>
      <ScoreBar score={s.score} className="pointer-events-none mt-1.5" />
      <div className="mt-2 flex items-center justify-between">
        <div className="relative z-10">
          <ElementDots elements={s.elements} />
        </div>
        {findings ? <span className="pointer-events-none text-[12px] font-semibold text-danger">끊김 {findings}</span> : <span className="pointer-events-none text-[12px] text-fg-subtle">끊김 없음</span>}
      </div>
    </div>
  );
}
