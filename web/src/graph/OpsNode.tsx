import { Handle, Position, useStore, type NodeProps } from '@xyflow/react';
import { clsx } from 'clsx';
import { CircleQuestionMark, UserX } from 'lucide-react';
import { memo, type ReactNode } from 'react';
import { ACTION_ICON, KIND_ICON, MEDIUM_ICON } from '../components/domain/icons';
import { ScorePill } from '../components/domain/Score';
import { formatScore, initials, scoreBand } from '../lib/format';
import { NODE_KIND_LABEL } from '../lib/labels';
import type { NodeBadge, OpsFlowNode, OpsNodeData, OpsNodeKind } from './model';

/** 이 배율보다 작게 보이면(전체 보기) 글자를 크게 키운 '지도' 모양으로 그린다 */
export const FAR_ZOOM = 0.58;

/** 화면 배율이 FAR_ZOOM 아래인지 — 경계를 넘을 때만 다시 그린다 */
export function useFarZoom(): boolean {
  return useStore((s) => s.transform[2] < FAR_ZOOM);
}

const BADGE_TONE: Record<NodeBadge['tone'], string> = {
  danger: 'bg-danger text-white',
  warn: 'bg-warn text-[#1b1200] projector:text-white',
  muted: 'bg-surface-3 text-fg-muted',
  info: 'bg-info text-[#03131d]',
  ok: 'bg-ok text-[#04150d]',
};

function Badges({ badges }: { badges: NodeBadge[] }) {
  if (!badges.length) return null;
  return (
    <div className="absolute -top-3 right-2 flex gap-1">
      {badges.map((b) => (
        <span key={b.text} className={clsx('rounded-full px-2 py-px text-[12px] font-bold shadow', BADGE_TONE[b.tone])}>
          {b.text}
        </span>
      ))}
    </div>
  );
}

/** 몸통과 같은 둥글기 — 강조 고리(box-shadow)가 네모로 삐져나오지 않게 */
function radiusOf(data: OpsNodeData): string {
  const k: OpsNodeKind = data.kind === 'ghost' ? (data.ghostOf === 'channel' ? 'channel' : 'ghost') : data.kind;
  if (k === 'channel') return 'rounded-full';
  if (k === 'role' || k === 'team' || k === 'system' || k === 'location') return 'rounded-xl';
  return 'rounded-2xl';
}

const FINDING_GLOW =
  'shadow-[0_0_0_7px_color-mix(in_oklab,var(--danger)_20%,transparent),0_0_32px_4px_color-mix(in_oklab,var(--danger)_40%,transparent)]';

/** 공통 껍데기: 심각도 테두리 · 강조 · 흐림 · 선택 */
function Shell({ data, children, far }: { data: OpsNodeData; children: ReactNode; far: boolean }) {
  const sev = data.severity;
  const brokenRing = sev === 'critical' || sev === 'high';
  const findingHl = data.highlighted === true && data.highlightStyle === 'finding';
  const pathHl = data.highlighted === true && !findingHl;
  const plain = !data.highlighted && !data.failed && !data.selected;
  return (
    <div
      className={clsx(
        'relative h-full w-full transition-[opacity,box-shadow,transform,filter] duration-300',
        radiusOf(data),
        data.dimmed && 'opacity-[0.36] saturate-[0.55]',
        pathHl && 'ring-[3px] ring-brand ring-offset-2 ring-offset-bg',
        findingHl && clsx('ring-[3px] ring-danger ring-offset-2 ring-offset-bg', FINDING_GLOW),
        data.current && 'scale-[1.04]',
        data.failed && 'animate-pulse-ring ring-[3px] ring-danger ring-offset-2 ring-offset-bg',
        data.selected && !data.highlighted && 'ring-[3px] ring-info ring-offset-2 ring-offset-bg',
        plain && brokenRing && data.kind !== 'ghost' && (far ? 'ring-[6px] ring-danger/80' : 'ring-2 ring-danger/75'),
        plain && sev === 'medium' && data.kind !== 'ghost' && (far ? 'ring-[5px] ring-warn/70' : 'ring-2 ring-warn/60'),
        data.hold && 'outline-2 outline-offset-2 outline-warn outline-dashed',
      )}
    >
      <Handle type="target" position={Position.Left} isConnectable={false} />
      {children}
      <Handle type="source" position={Position.Right} isConnectable={false} />
      {far ? null : <Badges badges={data.badges} />}
    </div>
  );
}

function ServiceBody({ data }: { data: OpsNodeData }) {
  const Icon = KIND_ICON.service;
  return (
    <div
      className={clsx(
        'flex h-full flex-col justify-between rounded-2xl border bg-surface-2 px-4 py-3 shadow-lg',
        data.isFocus ? 'border-brand/70 bg-[color-mix(in_oklab,var(--brand)_10%,var(--surface-2))]' : 'border-line-strong',
      )}
    >
      <div className="flex items-center gap-2">
        <span className="grid size-8 place-items-center rounded-lg bg-brand/15 text-brand">
          <Icon className="size-[18px]" aria-hidden />
        </span>
        <span className="min-w-0 flex-1 truncate text-[19px] font-bold">{data.label}</span>
        {data.tier !== undefined ? (
          <span className="rounded-md border border-line-strong bg-surface px-1.5 py-px font-mono text-[12px] font-bold text-fg-muted">T{data.tier}</span>
        ) : null}
      </div>
      <div className="flex items-center justify-between gap-2">
        <span className="text-[13px] text-fg-subtle">{NODE_KIND_LABEL.service}</span>
        {data.score !== undefined ? <ScorePill score={data.score} /> : <span className="text-[13px] text-fg-subtle">점수 없음</span>}
      </div>
    </div>
  );
}

function avatarTone(data: OpsNodeData): string {
  return data.departed
    ? 'bg-danger text-white'
    : data.onLeave
      ? 'bg-warn text-[#1b1200]'
      : data.unknownPerson
        ? 'border-2 border-dashed border-fg-subtle bg-surface-3 text-fg-muted'
        : 'bg-[color-mix(in_oklab,var(--brand)_25%,var(--surface-3))] text-fg';
}

function personFrame(data: OpsNodeData): string {
  return data.departed ? 'border-danger/70 bg-danger/10' : data.onLeave ? 'border-warn/60 bg-warn/8' : 'border-line bg-surface-2';
}

function PersonBody({ data }: { data: OpsNodeData }) {
  return (
    <div className={clsx('flex h-full items-center gap-3 rounded-2xl border px-3 shadow-md', personFrame(data))}>
      <span className={clsx('relative grid size-12 shrink-0 place-items-center rounded-full text-[16px] font-bold', avatarTone(data))}>
        {data.departed ? <UserX className="size-6" aria-hidden /> : data.unknownPerson ? <CircleQuestionMark className="size-6" aria-hidden /> : initials(data.label)}
      </span>
      <div className="min-w-0">
        <p className={clsx('truncate text-[17px] font-bold', data.departed && 'text-danger line-through decoration-2')}>{data.label}</p>
        {data.sublabel ? <p className="truncate text-[13px] text-fg-muted">{data.sublabel}</p> : null}
      </div>
    </div>
  );
}

function ChipBody({ data, icon: Icon, hex = false }: { data: OpsNodeData; icon: typeof KIND_ICON.role; hex?: boolean }) {
  return (
    <div
      className={clsx(
        'flex h-full items-center gap-2 border border-line bg-surface-2 px-4 shadow-sm',
        hex ? 'rounded-[14px] [clip-path:polygon(8%_0,92%_0,100%_50%,92%_100%,8%_100%,0_50%)] px-6' : 'rounded-xl',
      )}
    >
      <Icon className="size-[18px] shrink-0 text-fg-muted" aria-hidden />
      <span className="truncate text-[15px] font-semibold">{data.label}</span>
    </div>
  );
}

function ChannelBody({ data }: { data: OpsNodeData }) {
  const Icon = MEDIUM_ICON[data.medium ?? 'other'];
  return (
    <div className="flex h-full items-center gap-2 rounded-full border border-line-strong bg-surface-2 px-3 shadow-sm">
      <span className="grid size-8 place-items-center rounded-full bg-info/15 text-info">
        <Icon className="size-[18px]" aria-hidden />
      </span>
      <span className="truncate text-[15px] font-semibold">{data.label}</span>
    </div>
  );
}

function TriggerBody({ data }: { data: OpsNodeData }) {
  const Icon = KIND_ICON.trigger;
  return (
    <div className="flex h-full items-center gap-3 rounded-2xl border border-line bg-surface-2 px-3 shadow-sm">
      <span className="grid size-10 shrink-0 place-items-center rounded-xl bg-warn/15 text-warn">
        <Icon className="size-5" aria-hidden />
      </span>
      <div className="min-w-0">
        <p className="line-clamp-2 text-[14.5px] leading-snug font-semibold">{data.label}</p>
        {data.sublabel ? <p className="truncate text-[12.5px] text-fg-subtle">{data.sublabel}</p> : null}
      </div>
    </div>
  );
}

function ActionBody({ data }: { data: OpsNodeData }) {
  const Icon = ACTION_ICON[data.actionType ?? 'other'];
  return (
    <div className="flex h-full items-center gap-3 rounded-2xl border border-line bg-surface-2 px-3 shadow-sm">
      <span className="grid size-10 shrink-0 place-items-center rounded-xl bg-ok/15 text-ok">
        <Icon className="size-5" aria-hidden />
      </span>
      <div className="min-w-0">
        <p className="truncate text-[15px] font-semibold">{data.label}</p>
        <p className="truncate text-[12.5px] text-fg-subtle">{data.sublabel ?? NODE_KIND_LABEL.action}</p>
      </div>
    </div>
  );
}

function ObligationBody({ data }: { data: OpsNodeData }) {
  const Icon = KIND_ICON.obligation;
  const bad = data.severity === 'critical' || data.severity === 'high';
  return (
    <div className="flex h-full items-center gap-3 rounded-2xl border border-line bg-surface-2 px-3 shadow-sm">
      <span className="grid size-10 shrink-0 place-items-center rounded-xl bg-brand/15 text-brand">
        <Icon className="size-5" aria-hidden />
      </span>
      <div className="min-w-0">
        <p className="truncate text-[15px] font-semibold">{data.label}</p>
        {data.sublabel ? (
          <p className={clsx('mt-0.5 inline-block max-w-full truncate rounded px-1.5 text-[12.5px] font-bold', bad ? 'bg-danger/15 text-danger' : 'bg-surface-3 text-fg-muted')}>
            {data.sublabel}
          </p>
        ) : null}
      </div>
    </div>
  );
}

function GhostBody({ data }: { data: OpsNodeData }) {
  const Icon = KIND_ICON[data.ghostOf ?? 'ghost'];
  return (
    <div
      className={clsx(
        'flex h-full items-center gap-3 border-2 border-dashed border-ghost bg-danger/8 px-3',
        data.ghostOf === 'channel' ? 'rounded-full' : 'rounded-2xl',
      )}
    >
      <span className="grid size-9 shrink-0 place-items-center rounded-full border-2 border-dashed border-ghost text-ghost">
        <Icon className="size-[18px]" aria-hidden />
      </span>
      <div className="min-w-0">
        <p className="truncate text-[15px] font-bold text-ghost">{data.label}</p>
        {data.sublabel ? <p className="truncate text-[12.5px] text-fg-muted">{data.sublabel}</p> : null}
      </div>
    </div>
  );
}

/** 멀리서 볼 때(전체 보기): 아이콘 + 큰 이름 한 줄. 색·테두리 규칙은 가까이 볼 때와 같다. */
function FarBody({ data, height }: { data: OpsNodeData; height: number }) {
  const size = Math.round(Math.min(36, Math.max(22, height * 0.46)));
  const icon = Math.round(size * 1.05);
  let Icon = KIND_ICON[data.kind];
  let frame = 'border-2 border-line-strong bg-surface-2';
  let iconCls = 'text-fg-muted';
  let labelCls = '';
  let shape = 'rounded-2xl';
  switch (data.kind) {
    case 'service':
      frame = data.isFocus ? 'border-[3px] border-brand bg-[color-mix(in_oklab,var(--brand)_14%,var(--surface-2))]' : 'border-[3px] border-line-strong bg-surface-2';
      iconCls = 'text-brand';
      break;
    case 'person':
      frame = clsx('border-2', personFrame(data));
      if (data.departed) {
        Icon = UserX;
        iconCls = 'text-danger';
        labelCls = 'text-danger line-through decoration-[3px]';
      } else if (data.onLeave) iconCls = 'text-warn';
      break;
    case 'trigger':
      iconCls = 'text-warn';
      break;
    case 'action':
      Icon = ACTION_ICON[data.actionType ?? 'other'];
      iconCls = 'text-ok';
      break;
    case 'obligation':
      iconCls = 'text-brand';
      break;
    case 'channel':
      Icon = MEDIUM_ICON[data.medium ?? 'other'];
      iconCls = 'text-info';
      shape = 'rounded-full';
      break;
    case 'ghost':
      Icon = KIND_ICON[data.ghostOf ?? 'ghost'];
      frame = 'border-[3px] border-dashed border-ghost bg-danger/10';
      iconCls = 'text-ghost';
      labelCls = 'text-ghost';
      if (data.ghostOf === 'channel') shape = 'rounded-full';
      break;
    case 'role':
    case 'team':
    case 'system':
    case 'location':
      shape = 'rounded-xl';
      break;
  }
  const band = data.score !== undefined ? scoreBand(data.score) : null;
  // 긴 이름은 두 줄로 (노드 높이가 넉넉할 때만) — 멀리서도 '결제 서…'처럼 잘리지 않게
  const twoLines = height >= 64 && [...data.label].length > 6;
  if (data.kind === 'service') {
    // 서비스는 두 줄: 이름 / 점수 (멀리서도 이름이 잘리지 않게)
    return (
      <div className={clsx('flex h-full flex-col justify-center gap-1 px-4 shadow-md', shape, frame)}>
        <span className="flex min-w-0 items-center gap-2">
          <Icon className={clsx('shrink-0', iconCls)} style={{ width: 30, height: 30 }} aria-hidden strokeWidth={2.4} />
          <span className="min-w-0 flex-1 truncate text-[34px] leading-tight font-extrabold">{data.label}</span>
        </span>
        {data.score !== undefined ? (
          <span className={clsx('tnum text-[30px] leading-none font-black', band === 'red' ? 'text-danger' : band === 'amber' ? 'text-warn' : 'text-ok')}>
            {formatScore(data.score)}
            {data.tier !== undefined ? <span className="ml-2 text-[22px] font-bold text-fg-subtle">T{data.tier}</span> : null}
          </span>
        ) : null}
      </div>
    );
  }
  return (
    <div className={clsx('flex h-full items-center gap-2.5 px-4 shadow-md', shape, frame)}>
      <Icon className={clsx('shrink-0', iconCls)} style={{ width: icon, height: icon }} aria-hidden strokeWidth={2.4} />
      {twoLines ? (
        <span className={clsx('line-clamp-2 min-w-0 flex-1 leading-[1.12] font-extrabold', labelCls)} style={{ fontSize: Math.round(size * 0.78) }}>
          {data.label}
        </span>
      ) : (
        <span className={clsx('min-w-0 flex-1 truncate leading-tight font-extrabold', labelCls)} style={{ fontSize: size }}>
          {data.label}
        </span>
      )}
    </div>
  );
}

function OpsNodeImpl({ data, height }: NodeProps<OpsFlowNode>) {
  const far = useFarZoom();
  let body: ReactNode;
  if (far) body = <FarBody data={data} height={height ?? 60} />;
  else
    switch (data.kind) {
      case 'service':
        body = <ServiceBody data={data} />;
        break;
      case 'person':
        body = <PersonBody data={data} />;
        break;
      case 'role':
        body = <ChipBody data={data} icon={KIND_ICON.role} hex />;
        break;
      case 'team':
        body = <ChipBody data={data} icon={KIND_ICON.team} />;
        break;
      case 'system':
        body = <ChipBody data={data} icon={KIND_ICON.system} />;
        break;
      case 'location':
        body = <ChipBody data={data} icon={KIND_ICON.location} />;
        break;
      case 'channel':
        body = <ChannelBody data={data} />;
        break;
      case 'trigger':
        body = <TriggerBody data={data} />;
        break;
      case 'action':
        body = <ActionBody data={data} />;
        break;
      case 'obligation':
        body = <ObligationBody data={data} />;
        break;
      case 'ghost':
        body = <GhostBody data={data} />;
        break;
    }
  return (
    <Shell data={data} far={far}>
      {body}
    </Shell>
  );
}

export const OpsNode = memo(OpsNodeImpl);
