/**
 * 데이터 출처 모드.
 *  - 'api'     : FastAPI(/api/v1) 사용
 *  - 'mock'    : VITE_USE_MOCKS=1 — src/mocks/*.json (별빛상점 예제)
 *  - 'offline' : API에 닿지 않아 예제 데이터로 자동 전환 (화면에 '오프라인 데모 데이터' 배너)
 */
import { useSyncExternalStore } from 'react';

export type SourceMode = 'api' | 'mock' | 'offline';

const initial: SourceMode = import.meta.env.VITE_USE_MOCKS === '1' ? 'mock' : 'api';
let mode: SourceMode = initial;
const listeners = new Set<() => void>();

export function getSourceMode(): SourceMode {
  return mode;
}

export function setSourceMode(next: SourceMode): void {
  if (next === mode) return;
  mode = next;
  for (const l of listeners) l();
}

/** 환경 변수로 목업 모드가 강제됐는지 */
export function mocksForced(): boolean {
  return initial === 'mock';
}

export function usesMockData(): boolean {
  return mode !== 'api';
}

function subscribe(cb: () => void): () => void {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

export function useSourceMode(): SourceMode {
  return useSyncExternalStore(subscribe, getSourceMode, getSourceMode);
}
