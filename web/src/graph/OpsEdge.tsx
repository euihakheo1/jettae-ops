import { BaseEdge, EdgeLabelRenderer, getBezierPath, type EdgeProps } from '@xyflow/react';
import { clsx } from 'clsx';
import { memo } from 'react';
import type { Point } from './layout';
import type { EdgeTone, OpsEdgeData, OpsFlowEdge } from './model';
import { useFarZoom } from './OpsNode';

const STROKE: Record<EdgeTone, { color: string; width: number; dash?: string }> = {
  broken: { color: 'var(--danger)', width: 2.75 },
  warn: { color: 'var(--warn)', width: 2 },
  normal: { color: 'var(--edge)', width: 1.6 },
  faded: { color: 'var(--edge-faded)', width: 1 },
  ghost: { color: 'var(--ghost)', width: 2.25, dash: '7 6' },
};

export const MARKER_ID: Record<EdgeTone | 'highlight', string> = {
  broken: 'jt-arrow-broken',
  warn: 'jt-arrow-warn',
  normal: 'jt-arrow-normal',
  faded: 'jt-arrow-faded',
  ghost: 'jt-arrow-ghost',
  highlight: 'jt-arrow-highlight',
};

const LABEL_TONE: Record<EdgeTone, string> = {
  broken: 'border-danger/60 bg-[color-mix(in_oklab,var(--danger)_18%,var(--surface))] text-danger',
  warn: 'border-warn/60 bg-[color-mix(in_oklab,var(--warn)_14%,var(--surface))] text-warn',
  normal: 'border-line-strong bg-surface text-fg-muted',
  faded: 'border-line bg-surface text-fg-subtle',
  ghost: 'border-ghost/70 border-dashed bg-[color-mix(in_oklab,var(--danger)_12%,var(--surface))] text-ghost',
};

/**
 * 곡선 위 t 지점. React Flow 베지어(오른쪽→왼쪽 핸들)의 제어점은 (sx + d/2, sy), (tx − d/2, ty) 이다.
 * 뒤로 가는 연결(tx < sx)은 모양이 달라 가운데 점을 그대로 쓴다.
 */
function labelPoint(sx: number, sy: number, tx: number, ty: number, t: number, midX: number, midY: number): [number, number] {
  if (t === 0.5 || tx < sx) return [midX, midY];
  const d = (tx - sx) / 2;
  const u = 1 - t;
  const a = u * u * u;
  const b = 3 * u * u * t;
  const c = 3 * u * t * t;
  const e = t * t * t;
  return [a * sx + b * (sx + d) + c * (tx - d) + e * tx, a * sy + b * sy + c * ty + e * ty];
}

/** ELK 스플라인(시작점 + 3차 베지어 조각들) → SVG path */
export function splinePath(points: readonly Point[]): string {
  const f = (n: number) => Math.round(n * 10) / 10;
  const [p0, ...rest] = points;
  if (!p0) return '';
  let d = `M${f(p0.x)},${f(p0.y)}`;
  for (let i = 0; i + 2 < rest.length; i += 3) {
    const [a, b, c] = [rest[i]!, rest[i + 1]!, rest[i + 2]!];
    d += ` C${f(a.x)},${f(a.y)} ${f(b.x)},${f(b.y)} ${f(c.x)},${f(c.y)}`;
  }
  return d;
}

/** 스플라인 위 u(0~1) 지점 — 조각 수로 나눠 해당 조각의 t에서 계산 */
export function splinePoint(points: readonly Point[], u: number): Point {
  const k = Math.floor((points.length - 1) / 3);
  if (k < 1) return points[0] ?? { x: 0, y: 0 };
  const pos = Math.min(Math.max(u, 0), 1) * k;
  const seg = Math.min(Math.floor(pos), k - 1);
  const t = pos - seg;
  const p0 = points[seg * 3]!;
  const p1 = points[seg * 3 + 1]!;
  const p2 = points[seg * 3 + 2]!;
  const p3 = points[seg * 3 + 3]!;
  const v = 1 - t;
  const a = v * v * v;
  const b = 3 * v * v * t;
  const c = 3 * v * t * t;
  const e = t * t * t;
  return { x: a * p0.x + b * p1.x + c * p2.x + e * p3.x, y: a * p0.y + b * p1.y + c * p2.y + e * p3.y };
}

function OpsEdgeImpl({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, data, animated }: EdgeProps<OpsFlowEdge>) {
  const d = data as OpsEdgeData;
  let path: string;
  let labelX: number;
  let labelY: number;
  const route = d.route;
  if (route && route.points.length >= 4) {
    // ELK가 그린 곡선 — 여러 층을 건너는 연결이 중간 노드를 가로지르지 않는다
    path = splinePath(route.points);
    const first = route.points[0]!;
    const last = route.points[route.points.length - 1]!;
    const t = d.labelT ?? (Math.abs(last.y - first.y) > 40 ? (last.y > first.y ? 0.62 : 0.38) : 0.5);
    ({ x: labelX, y: labelY } = splinePoint(route.points, t));
  } else {
    const [bezier, midX, midY] = getBezierPath({ sourceX, sourceY, sourcePosition, targetX, targetY, targetPosition, curvature: 0.3 });
    path = bezier;
    // 퍼짐·모임이 같으면(X자로 엇갈리는 연결) 오르는 연결은 출발 쪽, 내리는 연결은 도착 쪽에 라벨을 둔다
    const t = d.labelT ?? (Math.abs(targetY - sourceY) > 40 ? (targetY > sourceY ? 0.62 : 0.38) : 0.5);
    [labelX, labelY] = labelPoint(sourceX, sourceY, targetX, targetY, t, midX, midY);
  }
  const far = useFarZoom();
  const tone = d.tone;
  const s = STROKE[tone];
  // 리허설 경로는 브랜드색, 선택한 끊김은 원래 색(빨강)을 유지한 채 굵게
  const findingHl = d.highlighted === true && d.highlightStyle === 'finding';
  const hl = d.highlighted === true && !findingHl;
  const color = hl ? 'var(--brand)' : s.color;
  const width = (hl ? 3.5 : findingHl ? s.width + 1.5 : s.width) * (far ? 2.2 : 1);
  const flowing = hl || findingHl || animated || tone === 'broken' || tone === 'ghost';
  // 멀리서 보면(전체 보기) 글자 라벨은 읽을 수 없어 숨긴다
  const showLabel = !far && (tone !== 'faded' || hl || findingHl);
  const marker = hl ? MARKER_ID.highlight : MARKER_ID[tone];
  return (
    <>
      <BaseEdge
        id={id}
        path={path}
        markerEnd={`url(#${marker})`}
        className={clsx(flowing && 'edge-flow')}
        style={{
          stroke: color,
          strokeWidth: width,
          opacity: d.dimmed ? 0.18 : 1,
          transition: 'opacity 300ms, stroke 300ms',
          ...(s.dash && !flowing ? { strokeDasharray: s.dash } : {}),
        }}
        interactionWidth={18}
      />
      {showLabel ? (
        <EdgeLabelRenderer>
          <div
            className={clsx(
              'nodrag nopan pointer-events-none absolute rounded-full border px-2 py-0.5 text-[12.5px] leading-[18px] font-bold whitespace-nowrap shadow-sm transition-opacity',
              findingHl ? 'z-10 border-danger bg-danger text-white' : hl ? 'border-brand bg-brand text-brand-fg' : LABEL_TONE[tone],
              d.dimmed && 'opacity-25',
              d.hold && 'outline-1 outline-warn outline-dashed',
            )}
            style={{ transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)` }}
          >
            {d.label}
            {d.hold ? ' · 보류' : ''}
          </div>
        </EdgeLabelRenderer>
      ) : null}
    </>
  );
}

export const OpsEdge = memo(OpsEdgeImpl);

/** 화살표 마커 정의 (React Flow SVG 밖에 한 번 그린다) */
export function EdgeMarkers() {
  const entries: [string, string][] = [
    [MARKER_ID.broken, 'var(--danger)'],
    [MARKER_ID.warn, 'var(--warn)'],
    [MARKER_ID.normal, 'var(--edge)'],
    [MARKER_ID.faded, 'var(--edge-faded)'],
    [MARKER_ID.ghost, 'var(--ghost)'],
    [MARKER_ID.highlight, 'var(--brand)'],
  ];
  return (
    <svg width="0" height="0" className="absolute" aria-hidden focusable="false">
      <defs>
        {entries.map(([id, color]) => (
          <marker key={id} id={id} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill={color} />
          </marker>
        ))}
      </defs>
    </svg>
  );
}
