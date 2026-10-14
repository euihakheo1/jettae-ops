import { describe, expect, it } from 'vitest';
import type { SessionInfo } from '../api/types';
import { permissionsFor, PUBLIC_TRIAL_REASON, ROLE_UNKNOWN_REASON } from './permissions';

const session = (s: Partial<SessionInfo>): SessionInfo => ({
  user: 'visitor',
  role: 'viewer',
  anonymous: true,
  auth_mode: 'public',
  can_switch_role: false,
  ...s,
});

describe('permissionsFor', () => {
  it('공개 체험판 방문자: 화면의 역할과 상관없이 서버 역할(보기 전용)로 막고 체험판 안내를 쓴다', () => {
    const p = permissionsFor('lead', session({}));
    expect(p).toMatchObject({ role: 'viewer', publicTrial: true, canSwitchRole: false });
    expect(p.editBlocked).toBe(PUBLIC_TRIAL_REASON);
    expect(p.decideBlocked).toBe(PUBLIC_TRIAL_REASON);
  });

  it('토큰 모드의 편집자: 초안·리허설은 되고 승인은 안 된다', () => {
    const p = permissionsFor('viewer', session({ user: 'kim', role: 'editor', anonymous: false, auth_mode: 'token' }));
    expect(p).toMatchObject({ role: 'editor', publicTrial: false, canSwitchRole: false, editBlocked: null });
    expect(p.decideBlocked).toBe('팀 리드만 할 수 있습니다');
  });

  it('개발 모드(역할 전환 가능)·세션 모름: 상단 역할 전환기의 역할을 쓴다', () => {
    const dev = session({ user: 'park.jaehyun', role: 'lead', anonymous: false, auth_mode: 'dev', can_switch_role: true });
    expect(permissionsFor('viewer', dev)).toMatchObject({ role: 'viewer', canSwitchRole: true, editBlocked: '편집자 이상만 실행할 수 있습니다' });
    expect(permissionsFor('lead', undefined)).toMatchObject({ role: 'lead', canSwitchRole: true, editBlocked: null, decideBlocked: null });
  });

  it('배포 빌드: /me 대기·실패 중에는 화면의 역할(기본 팀 리드)과 상관없이 보기 전용, 역할 전환기도 숨긴다', () => {
    const p = permissionsFor('lead', undefined, true);
    expect(p).toMatchObject({ role: 'viewer', publicTrial: false, roleUnknown: true, canSwitchRole: false });
    expect(p.editBlocked).toBe(ROLE_UNKNOWN_REASON);
    expect(p.decideBlocked).toBe(ROLE_UNKNOWN_REASON);
  });

  it('배포 빌드 + 개발 모드 서버(결선 시연): /me가 답하면 역할 전환기와 팀 리드 역할이 예전처럼 나온다', () => {
    const dev = session({ user: 'park.jaehyun', role: 'lead', anonymous: false, auth_mode: 'dev', can_switch_role: true });
    expect(permissionsFor('lead', dev, true)).toEqual({
      role: 'lead',
      publicTrial: false,
      roleUnknown: false,
      canSwitchRole: true,
      editBlocked: null,
      decideBlocked: null,
    });
  });

  it('배포 빌드 + 공개 체험판: /me가 답하면 서버 역할(보기 전용)과 체험판 안내', () => {
    expect(permissionsFor('lead', session({}), true)).toMatchObject({ role: 'viewer', publicTrial: true, roleUnknown: false, editBlocked: PUBLIC_TRIAL_REASON });
  });
});
