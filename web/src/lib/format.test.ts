import { describe, expect, it } from 'vitest';
import {
  displayName,
  formatAttr,
  formatDateTime,
  formatDelta,
  formatMinutes,
  formatPercent,
  formatScore,
  formatT,
  hashPrefix,
  initials,
  nameFromId,
  roundHalfUp1,
  scoreBand,
  toServiceId,
  versionChip,
} from './format';
import type { Node } from '../api/types';

const person = (name: string, title?: string, status = 'active'): Node => ({
  id: `person:${name}`,
  kind: 'person',
  name,
  attrs: {
    status: { value: status, spans: [], source: 'roster' },
    ...(title ? { title: { value: title, spans: [], source: 'roster' as const } } : {}),
  },
  prov: [],
  verification: 'verified',
  hold_reason: null,
});

describe('점수', () => {
  it('색 띠: 50 미만 빨강, 80 미만 주황, 80 이상 초록', () => {
    expect(scoreBand(0)).toBe('red');
    expect(scoreBand(49.9)).toBe('red');
    expect(scoreBand(50)).toBe('amber');
    expect(scoreBand(79.9)).toBe('amber');
    expect(scoreBand(80)).toBe('green');
    expect(scoreBand(100)).toBe('green');
  });

  it('반올림은 5 이상 올림 (백엔드 SCORING_V1과 같음)', () => {
    expect(roundHalfUp1(11.25)).toBe(11.3);
    expect(roundHalfUp1(55.727)).toBe(55.7);
    expect(roundHalfUp1(0.05)).toBe(0.1);
    expect(roundHalfUp1(-1.25)).toBe(-1.3);
  });

  it('formatScore / formatDelta', () => {
    expect(formatScore(22)).toBe('22.0');
    expect(formatScore(55.7)).toBe('55.7');
    expect(formatScore(null)).toBe('—');
    expect(formatDelta(10.08)).toBe('+10.1');
    expect(formatDelta(-2)).toBe('−2.0');
    expect(formatDelta(0.01)).toBe('±0.0');
  });
});

describe('시간', () => {
  it('분을 사람이 읽는 길이로', () => {
    expect(formatMinutes(5)).toBe('5분');
    expect(formatMinutes(60)).toBe('1시간');
    expect(formatMinutes(90)).toBe('1시간 30분');
    expect(formatMinutes(1440)).toBe('24시간');
    expect(formatMinutes(2880)).toBe('48시간');
    expect(formatMinutes(4320)).toBe('72시간');
    expect(formatMinutes(null)).toBe('—');
  });

  it('리허설 시각 t+분', () => {
    expect(formatT(0)).toBe('t+0분');
    expect(formatT(28)).toBe('t+28분');
    expect(formatT(2.5)).toBe('t+2.5분');
  });

  it('한국 시간으로 표시', () => {
    expect(formatDateTime('2026-10-12T00:04:12Z')).toBe('2026.10.12 09:04');
    expect(formatDateTime('2026-10-12T09:00:00+09:00')).toBe('2026.10.12 09:00');
    expect(formatDateTime(null)).toBe('—');
  });
});

describe('기타', () => {
  it('백분율·해시·버전 칩', () => {
    expect(formatPercent(18, 37)).toBe('49%');
    expect(formatPercent(0, 0)).toBe('—');
    expect(hashPrefix('317399fa370714940c1a')).toBe('317399');
    expect(versionChip(1, '317399fa37')).toBe('v1 · 317399');
  });

  it('이름: id에서 추정, 직급 붙이기, 퇴사 꼬리표', () => {
    expect(nameFromId('service:상품-카탈로그')).toBe('상품 카탈로그');
    expect(displayName(person('박재현', '팀장'))).toBe('박재현 팀장');
    expect(displayName(person('김민준', '대리', 'departed'), undefined, { withStatus: true })).toBe('김민준 대리 (퇴사)');
    expect(displayName(undefined, 'person:김서연')).toBe('김서연');
  });

  it('아바타 이니셜', () => {
    expect(initials('김서연')).toBe('서연');
    expect(initials('한서윤')).toBe('서윤');
    expect(initials('Jane Doe')).toBe('JD');
    expect(initials('')).toBe('?');
  });

  it('속성 값 한국어 표시', () => {
    expect(formatAttr('deadline_minutes', 2880)).toBe('48시간');
    expect(formatAttr('tier', 1)).toBe('1등급');
    expect(formatAttr('status', 'departed')).toBe('퇴사');
    expect(formatAttr('medium', 'kakaotalk')).toBe('카카오톡');
    expect(formatAttr('requires_approval', true)).toBe('예');
    expect(formatAttr('aliases', ['김대리', 'kim'])).toBe('김대리, kim');
  });

  it('서비스 이름 → id', () => {
    expect(toServiceId('결제')).toBe('service:결제');
    expect(toServiceId('상품 카탈로그')).toBe('service:상품-카탈로그');
    expect(toServiceId('service:주문')).toBe('service:주문');
  });
});
