import { clsx } from 'clsx';
import type { ReactNode } from 'react';
import { ArrowDownRight, ArrowUpRight, BotOff, Bot, FlaskConical, CirclePause, Unplug } from 'lucide-react';
import type { DiagnosisReport, Severity } from '../../api/types';
import { BAND_TEXT, ScoreRing } from '../../components/domain/Score';
import { SeverityChip } from '../../components/ui/Badge';
import { formatDelta, formatPercent, formatScore, SCORE_BAND_LABEL, scoreBand } from '../../lib/format';
import { LLM_MODE_LABEL, SEVERITY_ORDER } from '../../lib/labels';
import { currentRehearsals } from '../../lib/rehearsals';

function Kpi({ label, children, className }: { label: string; children: ReactNode; className?: string }) {
  return (
    <section aria-label={label} className={clsx('card flex min-w-0 flex-col justify-between gap-2 px-5 py-4', className)}>
      <h2 className="text-[13px] font-semibold tracking-wide text-fg-subtle">{label}</h2>
      {children}
    </section>
  );
}

export function KpiStrip({ report, previous }: { report: DiagnosisReport; previous?: DiagnosisReport | undefined }) {
  const score = report.readiness.overall;
  const band = scoreBand(score);
  const delta = previous ? score - previous.readiness.overall : null;
  const counts = SEVERITY_ORDER.map((s) => [s, report.findings.filter((f) => f.severity === s).length] as [Severity, number]).filter(
    ([, n]) => n > 0,
  );
  const unknown = report.findings.filter((f) => f.status === 'unknown').length;
  const current = currentRehearsals(report);
  const passed = current.filter((r) => r.outcome === 'pass').length;
  const total = current.length;
  const held = report.held_element_ids.length;
  const llmOff = report.llm.mode === 'off';
  return (
    <div className="grid grid-cols-2 gap-3 xl:grid-cols-[1.35fr_1.4fr_1fr_0.8fr_1fr]">
      <Kpi label="준비도 점수" className="col-span-2 xl:col-span-1">
        <div className="flex items-center gap-4">
          <div className="relative grid place-items-center">
            <ScoreRing score={score} size={84} stroke={9} />
            <span className={clsx('absolute text-[13px] font-bold', BAND_TEXT[band])}>{SCORE_BAND_LABEL[band]}</span>
          </div>
          <div>
            <p className={clsx('tnum text-5xl leading-none font-extrabold tracking-tight whitespace-nowrap', BAND_TEXT[band])}>
              {formatScore(score)}
              <span className="ml-1 text-lg font-semibold text-fg-subtle">/100</span>
            </p>
            {delta !== null && previous ? (
              <p
                className={clsx(
                  'tnum mt-2 inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[13px] font-bold',
                  delta > 0 ? 'bg-ok/14 text-ok' : delta < 0 ? 'bg-danger/14 text-danger' : 'bg-surface-3 text-fg-muted',
                )}
              >
                {delta > 0 ? <ArrowUpRight className="size-4" aria-hidden /> : delta < 0 ? <ArrowDownRight className="size-4" aria-hidden /> : null}
                {previous.graph_version === 1 ? '최초 진단' : `v${previous.graph_version}`} 대비 {delta === 0 ? '변화 없음' : formatDelta(delta)}
              </p>
            ) : (
              <p className="mt-2 text-[13px] text-fg-subtle">핵심 서비스 등급 가중 평균</p>
            )}
          </div>
        </div>
      </Kpi>
      <Kpi label="끊긴 대응 경로">
        <div className="flex items-end gap-3">
          <p className="tnum flex items-center gap-2 text-4xl leading-none font-extrabold">
            <Unplug className="size-7 text-danger" aria-hidden />
            {report.findings.length}
            <span className="text-base font-semibold text-fg-subtle">건</span>
          </p>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {counts.map(([sev, n]) => (
            <span key={sev} className="inline-flex items-center gap-1">
              <SeverityChip severity={sev} size="md" />
              <span className="tnum text-[14px] font-bold">{n}</span>
            </span>
          ))}
          {unknown ? <span className="text-[13px] text-fg-muted">· 확인 불가 {unknown}</span> : null}
        </div>
      </Kpi>
      <Kpi label="리허설 작동률">
        <p className="tnum flex items-center gap-2 text-4xl leading-none font-extrabold">
          <FlaskConical className="size-7 text-info" aria-hidden />
          {formatPercent(passed, total)}
        </p>
        <div>
          <div className="flex h-2 overflow-hidden rounded-full bg-danger/40" role="img" aria-label={`통과 ${passed}건, 실패 ${total - passed}건`}>
            <div className="h-full bg-ok" style={{ width: total ? `${(passed / total) * 100}%` : '0%' }} />
          </div>
          <p className="tnum mt-1.5 text-[13px] text-fg-muted">
            통과 {passed} · 실패 {total - passed} / {total}개 시나리오
          </p>
        </div>
      </Kpi>
      <Kpi label="판단 보류">
        <p className={clsx('tnum flex items-center gap-2 text-4xl leading-none font-extrabold', held ? 'text-warn' : '')}>
          <CirclePauseIcon held={held} />
          {held}
        </p>
        <p className="text-[13px] text-fg-muted">{held ? '사람 확인이 필요한 요소' : '역번역 대조·동명이인 모두 확정'}</p>
      </Kpi>
      <Kpi label="AI 사용">
        <p className={clsx('flex items-center gap-2 text-[22px] leading-tight font-extrabold', llmOff ? 'text-ok' : 'text-brand')}>
          {llmOff ? <BotOff className="size-6" aria-hidden /> : <Bot className="size-6" aria-hidden />}
          {LLM_MODE_LABEL[report.llm.mode]}
        </p>
        <p className="text-[13px] leading-snug text-fg-muted">
          <strong className="text-fg">판정은 규칙</strong> · 판정·점수 경로 LLM 호출 {report.llm.judgement_path_calls}회
        </p>
      </Kpi>
    </div>
  );
}

function CirclePauseIcon({ held }: { held: number }) {
  return <CirclePause className={clsx('size-7', held ? 'text-warn' : 'text-fg-subtle')} aria-hidden />;
}
