/**
 * 화면의 버튼을 켜고 끄는 기준 — 서버가 실제로 허락하는 역할에 맞춘다.
 *  - 역할 전환이 안 되는 서버(공개 체험판·토큰 모드): GET /me 의 역할
 *  - 개발 모드·오프라인·목업: 상단 역할 전환기의 역할
 *  - 배포 빌드에서 GET /me 를 기다리는 중이거나 실패했을 때: 보기 전용(역할 전환기도 숨긴다)
 *    — 공개 체험판 방문자에게 잠깐이라도 팀 리드 버튼이 보이지 않게. 개발 서버·목업은 예전처럼 전환기의 역할('lead' 기본)을 쓴다.
 * 초안·리허설·다시 진단은 편집자 이상, 승인·거절·되돌리기는 팀 리드 이상이다(서버 security.py와 같은 기준).
 */
import { strictOffline } from '../api/client';
import { useSession } from '../api/hooks';
import type { Role, SessionInfo } from '../api/types';
import { canApprove, useRole } from './prefs';

/** 공개 체험판(토큰 없는 방문자)에서 꺼 둔 버튼의 이유 */
export const PUBLIC_TRIAL_REASON = '공개 체험판에서는 할 수 없습니다(팀 계정 필요)';

/** 배포 빌드에서 서버가 역할을 알려 주기 전(GET /me 대기·실패)에 꺼 둔 버튼의 이유 */
export const ROLE_UNKNOWN_REASON = '서버에서 역할을 확인한 뒤에 할 수 있습니다';

export interface Permissions {
  role: Role;
  /** 토큰 없이 둘러보는 공개 체험판 방문자 */
  publicTrial: boolean;
  /** 배포 빌드에서 서버가 아직 역할을 알려 주지 않았다(대기·실패) — 그동안은 보기 전용 */
  roleUnknown: boolean;
  /** 상단 역할 전환기를 쓸 수 있는지(개발 모드·오프라인·목업) */
  canSwitchRole: boolean;
  /** 초안 만들기·리허설 실행·다시 진단을 못 하는 이유 — 할 수 있으면 null */
  editBlocked: string | null;
  /** 승인·거절·되돌리기를 못 하는 이유(공개 체험판·역할 확인 전일 때만; 역할 부족은 버튼마다 문구가 다르다) — 할 수 있으면 null */
  decideBlocked: string | null;
}

export function canEdit(role: Role): boolean {
  return role !== 'viewer';
}

/**
 * @param localRole 상단 역할 전환기의 역할
 * @param session GET /me 의 답(대기 중이거나 실패했으면 undefined)
 * @param requireSession 서버 답 없이는 역할을 정하지 않는다(배포 빌드, {@link strictOffline})
 */
export function permissionsFor(localRole: Role, session: SessionInfo | undefined, requireSession = false): Permissions {
  if (!session && requireSession) {
    return {
      role: 'viewer',
      publicTrial: false,
      roleUnknown: true,
      canSwitchRole: false,
      editBlocked: ROLE_UNKNOWN_REASON,
      decideBlocked: ROLE_UNKNOWN_REASON,
    };
  }
  const serverDecides = !!session && !session.can_switch_role;
  const role = serverDecides ? session.role : localRole;
  const publicTrial = serverDecides && session.anonymous;
  return {
    role,
    publicTrial,
    roleUnknown: false,
    canSwitchRole: !serverDecides,
    editBlocked: canEdit(role) ? null : publicTrial ? PUBLIC_TRIAL_REASON : '편집자 이상만 실행할 수 있습니다',
    decideBlocked: canApprove(role) ? null : publicTrial ? PUBLIC_TRIAL_REASON : '팀 리드만 할 수 있습니다',
  };
}

export function usePermissions(): Permissions {
  const local = useRole();
  const session = useSession().data;
  return permissionsFor(local, session, strictOffline());
}
