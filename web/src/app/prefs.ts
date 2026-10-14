/** 화면 설정(역할·테마) — localStorage에 저장, 실패해도 기본값으로 동작. */
import { useSyncExternalStore } from 'react';
import type { Role } from '../api/types';

function createStore<T extends string>(key: string, allowed: readonly T[], fallback: T) {
  const read = (): T => {
    try {
      const v = window.localStorage.getItem(key);
      return v && (allowed as readonly string[]).includes(v) ? (v as T) : fallback;
    } catch {
      return fallback;
    }
  };
  let value: T = typeof window === 'undefined' ? fallback : read();
  const listeners = new Set<() => void>();
  return {
    get: (): T => value,
    set: (next: T): void => {
      if (!(allowed as readonly string[]).includes(next) || next === value) return;
      value = next;
      try {
        window.localStorage.setItem(key, next);
      } catch {
        // 저장 불가(사생활 보호 모드 등) — 세션 동안만 유지
      }
      for (const l of listeners) l();
    },
    subscribe: (cb: () => void): (() => void) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
  };
}

export const ROLES: readonly Role[] = ['viewer', 'editor', 'lead', 'admin'];
export const roleStore = createStore<Role>('jettae.role', ROLES, 'lead');

export type Theme = 'dark' | 'projector';
export const THEMES: readonly Theme[] = ['dark', 'projector'];
export const themeStore = createStore<Theme>('jettae.theme', THEMES, 'dark');

export function useRole(): Role {
  return useSyncExternalStore(roleStore.subscribe, roleStore.get, roleStore.get);
}

export function useTheme(): Theme {
  return useSyncExternalStore(themeStore.subscribe, themeStore.get, themeStore.get);
}

export function canApprove(role: Role): boolean {
  return role === 'lead' || role === 'admin';
}

/** 개발용 인증 헤더 X-Jettae-User (ASCII만 허용되는 HTTP 헤더라 영문 id 사용) */
export const ROLE_USER: Record<Role, string> = {
  viewer: 'demo.viewer',
  editor: 'demo.editor',
  lead: 'park.jaehyun',
  admin: 'demo.admin',
};

/** 화면 표시용 사용자 이름 */
export const ROLE_PERSON: Record<Role, string> = {
  viewer: '데모 뷰어',
  editor: '데모 편집자',
  lead: '박재현 팀장',
  admin: '관리자',
};

export function applyTheme(theme: Theme): void {
  const root = document.documentElement;
  root.dataset.theme = theme;
  root.style.colorScheme = theme === 'projector' ? 'light' : 'dark';
}
