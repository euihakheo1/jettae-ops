import { clsx } from 'clsx';
import {
  AlarmClock,
  Bell,
  CircleCheck,
  CircleSlash,
  CircleX,
  Flag,
  Hand,
  Layers,
  Megaphone,
  MapPin,
  PhoneCall,
  PhoneOff,
  Play,
  Server,
  ShieldCheck,
  Siren,
  Stamp,
  TrendingUp,
  Unplug,
  UserCheck,
  UserX,
  Wrench,
  type LucideIcon,
} from 'lucide-react';
import type { GraphView, Perturbation, PerturbationKind, RehearsalResult, Scenario, SimEvent, SimEventKind } from '../../api/types';
import { Badge } from '../../components/ui/Badge';
import { Button } from '../../components/ui/Button';
import { DisabledReason } from '../../components/ui/Tooltip';
import { displayName, formatMinutes, formatT, nameFromId } from '../../lib/format';
import { ELEMENT_LABEL, INCIDENT_TYPE_LABEL, ORIGIN_LABEL, PERTURBATION_LABEL, SIM_EVENT_BAD, SIM_EVENT_GOOD, SIM_EVENT_LABEL } from '../../lib/labels';

export const PERTURBATION_ICON: Record<PerturbationKind, LucideIcon> = {
  person_unavailable: UserX,
  channel_down: Unplug,
  system_down: Server,
  location_down: MapPin,
  concurrent_incident: Layers,
};

export const EVENT_ICON: Record<SimEventKind, LucideIcon> = {
  incident_start: Siren,
  detected: Bell,
  undetected: CircleSlash,
  page: PhoneCall,
  channel_unavailable: Unplug,
  no_answer: PhoneOff,
  ack: UserCheck,
  escalate: TrendingUp,
  dead_end: CircleX,
  approval_requested: Hand,
  approved: Stamp,
  approval_blocked: CircleX,
  action_started: Wrench,
  mitigated: ShieldCheck,
  no_mitigation: CircleX,
  report_filed: Megaphone,
  report_missed: Megaphone,
  deadline_missed: AlarmClock,
};

export function nodeLabel(view: GraphView | undefined, id: string): string {
  const n = view?.nodes.find((x) => x.id === id);
  return n ? displayName(n, id) : nameFromId(id);
}

export function PerturbationChip({ p, view }: { p: Perturbation; view: GraphView | undefined }) {
  const Icon = PERTURBATION_ICON[p.kind];
  return (
    <span className="inline-flex items-center gap-1 rounded-lg border border-warn/40 bg-warn/10 px-2 py-0.5 text-[12.5px] font-semibold text-fg">
      <Icon className="size-3.5 text-warn" aria-hidden />
      {PERTURBATION_LABEL[p.kind]}
      <span className="font-normal text-fg-muted">· {nodeLabel(view, p.target_id)}</span>
    </span>
  );
}

export function OutcomeBadge({ result, size = 'md' }: { result: RehearsalResult | undefined; size?: 'sm' | 'md' | 'lg' }) {
  if (!result) return <Badge tone="muted" size={size}>미실행</Badge>;
  return result.outcome === 'pass' ? (
    <Badge tone="ok" icon={CircleCheck} size={size}>
      통과
    </Badge>
  ) : (
    <Badge tone="danger" icon={CircleX} size={size}>
      실패
    </Badge>
  );
}

export function ScenarioCard({
  scenario,
  result,
  view,
  serviceName,
  active,
  running,
  runBlocked = null,
  onSelect,
  onRun,
}: {
  scenario: Scenario;
  result: RehearsalResult | undefined;
  view: GraphView | undefined;
  serviceName: string;
  active: boolean;
  running: boolean;
  /** 실행 버튼을 꺼 둔 이유(역할 부족 등) — null이면 실행 가능 */
  runBlocked?: string | null;
  onSelect: () => void;
  onRun: () => void;
}) {
  return (
    <div
      className={clsx(
        'relative rounded-2xl border p-4 transition-colors',
        active ? 'border-brand/70 bg-brand/8 ring-1 ring-brand/40' : 'border-line bg-surface hover:border-line-strong',
      )}
    >
      <button type="button" onClick={onSelect} aria-current={active ? 'true' : undefined} className="absolute inset-0 rounded-2xl" aria-label={`${scenario.title} 결과 보기`} />
      <div className="pointer-events-none flex items-start gap-2">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-1.5">
            <Badge tone={scenario.origin === 'human' ? 'brand' : scenario.origin === 'llm' ? 'info' : 'neutral'} size="sm">
              {ORIGIN_LABEL[scenario.origin]}
            </Badge>
            {scenario.incident_type !== 'outage' ? (
              <Badge tone="warn" icon={Flag} size="sm">
                {INCIDENT_TYPE_LABEL[scenario.incident_type]}
              </Badge>
            ) : null}
            <span className="text-[12.5px] text-fg-subtle">{serviceName}</span>
          </div>
          <h3 className="mt-1.5 text-[16px] leading-snug font-bold">{scenario.title}</h3>
        </div>
        <OutcomeBadge result={result} />
      </div>
      {scenario.perturbations.length ? (
        <div className="pointer-events-none mt-2.5 flex flex-wrap gap-1.5">
          {scenario.perturbations.map((p, i) => (
            <PerturbationChip key={i} p={p} view={view} />
          ))}
        </div>
      ) : (
        <p className="pointer-events-none mt-2 text-[13px] text-fg-subtle">교란 없음 — 런북대로 이어지는지 확인</p>
      )}
      <div className="relative mt-3 flex items-center justify-between">
        <span className="pointer-events-none text-[12.5px] text-fg-muted">
          {result ? (result.outcome === 'pass' ? `완화 ${formatMinutes(result.time_to_mitigate)} / 시한 ${formatMinutes(result.deadline_minutes)}` : `막힘: ${result.failed_element ? ELEMENT_LABEL[result.failed_element] : '시한'}`) : ''}
        </span>
        <DisabledReason reason={runBlocked}>
          <Button
            size="sm"
            variant={active ? 'primary' : 'secondary'}
            icon={<Play className="size-3.5" aria-hidden />}
            loading={running}
            disabled={!!runBlocked}
            onClick={onRun}
            aria-label={`${scenario.title} 리허설 실행`}
          >
            실행
          </Button>
        </DisabledReason>
      </div>
    </div>
  );
}

/** 완화 시간 vs 시한 막대 */
export function DeadlineBar({ result }: { result: RehearsalResult }) {
  const max = Math.max(result.deadline_minutes, result.time_to_mitigate ?? 0) * 1.12;
  const ttm = result.time_to_mitigate;
  const deadlinePct = (result.deadline_minutes / max) * 100;
  const pass = result.outcome === 'pass';
  return (
    <div>
      <div className="relative h-9 overflow-hidden rounded-xl border border-line bg-surface-3/60">
        {ttm !== null ? (
          <div
            className={clsx('flex h-full items-center rounded-l-xl px-3 text-[14px] font-bold transition-[width] duration-700', pass ? 'bg-ok/80 text-[#04150d] projector:text-white' : 'bg-danger/80 text-white')}
            style={{ width: `${Math.max(8, (ttm / max) * 100)}%` }}
          >
            {formatMinutes(ttm)}
          </div>
        ) : (
          <div className="flex h-full w-full items-center bg-[repeating-linear-gradient(135deg,color-mix(in_oklab,var(--danger)_35%,transparent)_0_10px,transparent_10px_20px)] px-3 text-[14px] font-bold text-danger">
            완화하지 못함 — 대응 경로가 끊김
          </div>
        )}
        <div className="absolute top-0 bottom-0 w-0.5 bg-fg" style={{ left: `${deadlinePct}%` }} aria-hidden />
      </div>
      <div className="relative mt-1 h-5 text-[12.5px] text-fg-muted">
        <span className="absolute -translate-x-1/2 font-semibold whitespace-nowrap" style={{ left: `${deadlinePct}%` }}>
          시한 {formatMinutes(result.deadline_minutes)}
        </span>
        <span className="absolute left-0">0분</span>
      </div>
    </div>
  );
}

export function Timeline({ events, revealed, onJump, view }: { events: readonly SimEvent[]; revealed: number; onJump: (i: number) => void; view: GraphView | undefined }) {
  return (
    <ol className="relative space-y-1" aria-label="리허설 타임라인">
      {events.map((e, i) => {
        const Icon = EVENT_ICON[e.kind];
        const bad = SIM_EVENT_BAD.has(e.kind);
        const good = SIM_EVENT_GOOD.has(e.kind);
        const shown = i < revealed;
        const current = i === revealed - 1;
        return (
          <li key={i} className={clsx('relative transition-opacity duration-300', shown ? 'opacity-100' : 'opacity-30')}>
            {i < events.length - 1 ? <span aria-hidden className="absolute top-9 bottom-[-6px] left-[70px] w-px bg-line" /> : null}
            <button
              type="button"
              onClick={() => onJump(i + 1)}
              aria-current={current ? 'step' : undefined}
              className={clsx('flex w-full items-start gap-3 rounded-xl px-2 py-1.5 text-left', current ? 'bg-brand/10 ring-1 ring-brand/50' : 'hover:bg-surface-2')}
            >
              <span className="tnum w-12 shrink-0 pt-1.5 text-right font-mono text-[13px] font-semibold text-fg-muted">{formatT(e.t)}</span>
              <span
                className={clsx(
                  'relative z-10 grid size-8 shrink-0 place-items-center rounded-full border',
                  bad ? 'border-danger/60 bg-danger/15 text-danger' : good ? 'border-ok/50 bg-ok/12 text-ok' : 'border-line-strong bg-surface-2 text-fg-muted',
                )}
              >
                <Icon className="size-4" aria-hidden />
              </span>
              <span className="min-w-0 pt-0.5">
                <span className={clsx('text-[13px] font-bold', bad ? 'text-danger' : good ? 'text-ok' : 'text-fg-muted')}>{SIM_EVENT_LABEL[e.kind]}</span>
                {e.target_id && !e.target_id.startsWith('service:') ? <span className="ml-1.5 text-[12.5px] text-fg-subtle">{nodeLabel(view, e.target_id)}</span> : null}
                <span className="block text-[14.5px] leading-snug">{e.note}</span>
              </span>
            </button>
          </li>
        );
      })}
    </ol>
  );
}
