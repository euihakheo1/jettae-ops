import type { AttrValue, Node, PersonStatus } from '../api/types';
import { ATTR_LABEL, INCIDENT_TYPE_LABEL, MEDIUM_LABEL, PERSON_STATUS_LABEL, ACTION_TYPE_LABEL } from './labels';

/** 점수 색 띠: 50 미만 빨강, 80 미만 주황, 80 이상 초록 */
export type ScoreBand = 'red' | 'amber' | 'green';

export function scoreBand(score: number): ScoreBand {
  if (score < 50) return 'red';
  if (score < 80) return 'amber';
  return 'green';
}

export const SCORE_BAND_LABEL: Record<ScoreBand, string> = {
  red: '위험',
  amber: '주의',
  green: '양호',
};

/** 소수 첫째 자리 반올림(5 이상 올림) — 백엔드 SCORING_V1과 같은 규칙. */
export function roundHalfUp1(value: number): number {
  const sign = value < 0 ? -1 : 1;
  const scaled = Math.abs(value) * 10;
  // 부동소수 오차(예: 11.25*10 = 112.49999…) 보정
  const rounded = Math.floor(scaled + 0.5 + 1e-9);
  return (sign * rounded) / 10;
}

/** 55.7 → "55.7", 22 → "22.0" */
export function formatScore(score: number | null | undefined): string {
  if (score === null || score === undefined || Number.isNaN(score)) return '—';
  return roundHalfUp1(score).toFixed(1);
}

/** +16.3 → "+16.3", -2 → "−2.0", 0 → "±0.0" */
export function formatDelta(delta: number): string {
  const r = roundHalfUp1(delta);
  if (r === 0) return '±0.0';
  return `${r > 0 ? '+' : '−'}${Math.abs(r).toFixed(1)}`;
}

/** 분 → 사람이 읽는 길이. 5 → "5분", 90 → "1시간 30분", 1440 → "24시간", 4320 → "72시간" */
export function formatMinutes(minutes: number | null | undefined): string {
  if (minutes === null || minutes === undefined || Number.isNaN(minutes)) return '—';
  const m = Math.round(minutes);
  if (m < 60) return `${m}분`;
  const h = Math.floor(m / 60);
  const rest = m % 60;
  return rest === 0 ? `${h}시간` : `${h}시간 ${rest}분`;
}

/** 리허설 타임라인 시각 "t+5분" */
export function formatT(t: number): string {
  return `t+${Number.isInteger(t) ? t : t.toFixed(1)}분`;
}

/** 18/37 → "49%" (분모 0이면 "—") */
export function formatPercent(part: number, total: number): string {
  if (total <= 0) return '—';
  return `${Math.round((part / total) * 100)}%`;
}

/** 해시 앞 6자리 */
export function hashPrefix(hash: string | null | undefined, n = 6): string {
  return hash ? hash.slice(0, n) : '——————';
}

/** 버전 칩 "v1 · 317399" */
export function versionChip(version: number, hash: string | null | undefined): string {
  return `v${version} · ${hashPrefix(hash)}`;
}

const KST = 'Asia/Seoul';

/** ISO 시각 → "2026.10.12 09:04" (한국 시간) */
export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const parts = new Intl.DateTimeFormat('ko-KR', {
    timeZone: KST,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  }).formatToParts(d);
  const get = (type: Intl.DateTimeFormatPartTypes): string => parts.find((p) => p.type === type)?.value ?? '';
  return `${get('year')}.${get('month')}.${get('day')} ${get('hour')}:${get('minute')}`;
}

/** 숫자 천 단위 구분 */
export function formatNumber(n: number): string {
  return new Intl.NumberFormat('ko-KR').format(n);
}

/** id에서 이름 추정: "person:김서연" → "김서연", "service:상품-카탈로그" → "상품 카탈로그" */
export function nameFromId(id: string): string {
  const raw = id.includes(':') ? id.slice(id.indexOf(':') + 1) : id;
  return raw.replace(/-/g, ' ');
}

export function attrValue(node: Pick<Node, 'attrs'>, key: string): AttrValue | undefined {
  return node.attrs[key]?.value;
}

export function attrString(node: Pick<Node, 'attrs'>, key: string): string | undefined {
  const v = attrValue(node, key);
  return typeof v === 'string' ? v : undefined;
}

export function attrNumber(node: Pick<Node, 'attrs'>, key: string): number | undefined {
  const v = attrValue(node, key);
  return typeof v === 'number' ? v : undefined;
}

export function personStatus(node: Pick<Node, 'attrs' | 'kind'>): PersonStatus | undefined {
  if (node.kind !== 'person') return undefined;
  const v = attrString(node, 'status');
  if (v === 'active' || v === 'departed' || v === 'on_leave' || v === 'unknown') return v;
  return 'unknown';
}

/** 사람 표시명: "박재현 팀장", 퇴사자는 "(퇴사)" 꼬리표 옵션 */
export function displayName(
  node: Pick<Node, 'name' | 'kind' | 'attrs'> | undefined,
  fallbackId?: string,
  opts: { withStatus?: boolean } = {},
): string {
  if (!node) return fallbackId ? nameFromId(fallbackId) : '—';
  if (node.kind === 'person') {
    const title = attrString(node, 'title');
    const base = title ? `${node.name} ${title}` : node.name;
    const st = personStatus(node);
    if (opts.withStatus && st && st !== 'active') return `${base} (${PERSON_STATUS_LABEL[st]})`;
    return base;
  }
  if (node.kind === 'channel') {
    const medium = attrString(node, 'medium');
    if (medium && medium in MEDIUM_LABEL) return MEDIUM_LABEL[medium as keyof typeof MEDIUM_LABEL];
  }
  return node.name;
}

/** 이니셜(아바타): 한국어 이름은 성을 뺀 두 글자, 영문은 머리글자 */
export function initials(name: string): string {
  const trimmed = name.trim();
  if (!trimmed) return '?';
  if (/^[가-힣]+$/.test(trimmed)) return trimmed.length >= 3 ? trimmed.slice(1, 3) : trimmed;
  const words = trimmed.split(/\s+/);
  return words
    .slice(0, 2)
    .map((w) => w[0]?.toUpperCase() ?? '')
    .join('');
}

/** 속성 값을 한국어로 표시 */
export function formatAttr(key: string, value: AttrValue): string {
  if (Array.isArray(value)) return value.join(', ');
  if (typeof value === 'boolean') return value ? '예' : '아니요';
  if (typeof value === 'number') {
    if (key.endsWith('_minutes')) return formatMinutes(value);
    if (key === 'tier') return `${value}등급`;
    return String(value);
  }
  if (key === 'status' && value in PERSON_STATUS_LABEL) return PERSON_STATUS_LABEL[value as keyof typeof PERSON_STATUS_LABEL];
  if (key === 'medium' && value in MEDIUM_LABEL) return MEDIUM_LABEL[value as keyof typeof MEDIUM_LABEL];
  if (key === 'action_type' && value in ACTION_TYPE_LABEL) return ACTION_TYPE_LABEL[value as keyof typeof ACTION_TYPE_LABEL];
  if (key === 'incident_type' && value in INCIDENT_TYPE_LABEL) return INCIDENT_TYPE_LABEL[value as keyof typeof INCIDENT_TYPE_LABEL];
  return value;
}

export function attrLabel(key: string): string {
  return ATTR_LABEL[key] ?? key;
}

/** 서비스 이름/slug/id → service id ("결제" → "service:결제") */
export function toServiceId(value: string): string {
  return value.startsWith('service:') ? value : `service:${value.trim().toLowerCase().replace(/\s+/g, '-')}`;
}
