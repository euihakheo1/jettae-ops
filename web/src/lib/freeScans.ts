/**
 * 무료 검진 결과 — 이 브라우저에만 남기는 목록.
 *
 * 공개 베타(JETTAE_AUTH_MODE=public)에서 방문자가 올린 런북의 워크스페이스는 접근 키가 있어야 열린다.
 * 키는 서버에 해시로만 남고 평문은 이 목록에만 있으므로, 브라우저 저장소를 지우면 결과를 다시 열 수 없다.
 * 저장소가 막힌 환경(차단된 사이트 데이터)에서도 화면이 깨지지 않게 모든 접근을 try/catch로 감싸고,
 * 메모리 사본을 함께 두어 이 탭이 열려 있는 동안은 방금 받은 키로 결과를 열 수 있게 한다.
 */

export interface FreeScan {
  workspace_id: string;
  access_key: string;
  created_at: string;
  readiness_overall: number;
  findings: number;
}

const KEY = 'jettae.freeScans';
const VISITED_KEY = 'jettae.visited';
/** 오래된 것부터 버린다 — 키가 무한히 쌓이지 않게(항목 하나가 200바이트 안팎이라 넉넉히 둔다) */
export const MAX_SCANS = 50;

/** 저장소 사본(삽입 순서 = 오래된 것부터). 저장소를 읽거나 쓸 수 없을 때 이 탭의 기준이 된다 */
const memory = new Map<string, FreeScan>();
/** 마지막 쓰기가 저장소에 남지 못했으면 true — 그동안은 메모리 사본이 더 최신이다 */
let unsaved = false;
/** 저장소를 쓸 수 없을 때도 한 탭에서 첫 방문을 두 번 세지 않게 */
let visitedInMemory = false;

function remember(scans: readonly FreeScan[]): void {
  memory.clear();
  for (const s of scans) memory.set(s.workspace_id, s);
}

function parse(raw: string | null): FreeScan[] {
  if (!raw) return [];
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (s): s is FreeScan =>
        typeof s === 'object' &&
        s !== null &&
        typeof (s as FreeScan).workspace_id === 'string' &&
        typeof (s as FreeScan).access_key === 'string',
    );
  } catch {
    return [];
  }
}

function read(): FreeScan[] {
  if (unsaved) return [...memory.values()];
  let raw: string | null;
  try {
    raw = localStorage.getItem(KEY);
  } catch {
    return [...memory.values()];
  }
  const scans = parse(raw);
  remember(scans);
  return scans;
}

function write(scans: FreeScan[]): void {
  const kept = scans.slice(-MAX_SCANS);
  remember(kept);
  try {
    localStorage.setItem(KEY, JSON.stringify(kept));
    unsaved = false;
  } catch {
    // 저장소를 쓸 수 없으면 메모리 사본으로 이 탭이 열려 있는 동안만 결과를 볼 수 있다
    unsaved = true;
  }
}

/** 최근 것이 앞에 오도록 */
export function listFreeScans(): FreeScan[] {
  return [...read()].reverse();
}

export function saveFreeScan(scan: FreeScan): void {
  write([...read().filter((s) => s.workspace_id !== scan.workspace_id), scan]);
}

export function removeFreeScan(workspaceId: string): void {
  write(read().filter((s) => s.workspace_id !== workspaceId));
}

export function accessKeyFor(workspaceId: string): string | null {
  return read().find((s) => s.workspace_id === workspaceId)?.access_key ?? null;
}

type ExpiredListener = (workspaceId: string) => void;
const expiredListeners = new Set<ExpiredListener>();

/** 서버에서 사라진(보관 기간이 지났거나 서버에서 삭제된) 무료 검진을 목록에서 빼면 불린다 — 화면이 알림을 띄운다 */
export function onFreeScanExpired(listener: ExpiredListener): () => void {
  expiredListeners.add(listener);
  return () => {
    expiredListeners.delete(listener);
  };
}

/** 목록에 있던 무료 검진이면 빼고 알린 뒤 true, 없던 것이면 false */
export function expireFreeScan(workspaceId: string): boolean {
  if (!read().some((s) => s.workspace_id === workspaceId)) return false;
  removeFreeScan(workspaceId);
  for (const l of expiredListeners) l(workspaceId);
  return true;
}

/** `workspaces/{ws}/…` 또는 `/api/v1/workspaces/{ws}/…` 경로에서 워크스페이스 id를 꺼낸다 */
export function workspaceIdFromPath(path: string): string | null {
  const m = /(?:^|\/)workspaces\/([^/?#]+)/.exec(path);
  if (!m?.[1]) return null;
  try {
    return decodeURIComponent(m[1]);
  } catch {
    return null;
  }
}

/**
 * 이 브라우저의 첫 방문을 아직 세지 않았으면 true (고유 방문 수 측정용, 식별자는 저장하지 않음).
 * 표시는 남기지 않는다 — 이벤트 전송이 성공한 뒤 {@link markVisited}로 남긴다.
 * 저장소를 읽을 수 없으면 첫 방문으로 본다(한 탭에서는 한 번만).
 */
export function isFirstVisit(): boolean {
  if (visitedInMemory) return false;
  try {
    return !localStorage.getItem(VISITED_KEY);
  } catch {
    return true;
  }
}

/** 첫 방문을 셌다는 표시('방문함')만 남긴다 */
export function markVisited(): void {
  visitedInMemory = true;
  try {
    localStorage.setItem(VISITED_KEY, '1');
  } catch {
    // 저장소를 쓸 수 없으면 이 탭에서만 기억한다
  }
}

/** 테스트용: 메모리 사본과 표시를 처음 상태로 */
export function resetFreeScanMemory(): void {
  memory.clear();
  unsaved = false;
  visitedInMemory = false;
}
