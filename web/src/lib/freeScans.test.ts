import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { WORKSPACE_KEY_HEADER, withLocalFreeScans, workspaceKeyHeaders } from '../api/client';
import {
  accessKeyFor,
  expireFreeScan,
  isFirstVisit,
  listFreeScans,
  markVisited,
  MAX_SCANS,
  onFreeScanExpired,
  removeFreeScan,
  resetFreeScanMemory,
  saveFreeScan,
  workspaceIdFromPath,
} from './freeScans';

const scan = (id: string, key = `key-${id}`) => ({
  workspace_id: id,
  access_key: key,
  created_at: '2026-10-15T09:00:00+09:00',
  readiness_overall: 41.5,
  findings: 7,
});

/** 사이트 데이터를 막은 브라우저처럼 localStorage 접근이 모두 예외를 던지게 한다 */
function blockStorage() {
  const boom = () => {
    throw new DOMException('The operation is insecure.', 'SecurityError');
  };
  vi.spyOn(Storage.prototype, 'getItem').mockImplementation(boom);
  vi.spyOn(Storage.prototype, 'setItem').mockImplementation(boom);
}

describe('freeScans', () => {
  beforeEach(() => {
    localStorage.clear();
    resetFreeScanMemory();
  });
  afterEach(() => vi.restoreAllMocks());

  it('saves, lists newest first, replaces the same workspace and removes', () => {
    saveFreeScan(scan('ws-a'));
    saveFreeScan(scan('ws-b'));
    saveFreeScan(scan('ws-a', 'key-new'));
    expect(listFreeScans().map((s) => s.workspace_id)).toEqual(['ws-a', 'ws-b']);
    expect(accessKeyFor('ws-a')).toBe('key-new');
    removeFreeScan('ws-a');
    expect(accessKeyFor('ws-a')).toBeNull();
  });

  it('ignores corrupt storage instead of breaking the page', () => {
    localStorage.setItem('jettae.freeScans', '{not json');
    expect(listFreeScans()).toEqual([]);
    localStorage.setItem('jettae.freeScans', JSON.stringify([{ workspace_id: 1 }, scan('ws-ok')]));
    expect(listFreeScans().map((s) => s.workspace_id)).toEqual(['ws-ok']);
  });

  it('finds the workspace id in relative and absolute API paths', () => {
    expect(workspaceIdFromPath('workspaces/ws-1/report')).toBe('ws-1');
    expect(workspaceIdFromPath('/api/v1/workspaces/ws%2D2/graph?version=3')).toBe('ws-2');
    expect(workspaceIdFromPath('workspaces')).toBeNull();
    expect(workspaceIdFromPath('patches/p-1/approve')).toBeNull();
  });

  it('counts the first visit until it is marked (after a delivered event)', () => {
    expect(isFirstVisit()).toBe(true);
    expect(isFirstVisit()).toBe(true);
    markVisited();
    expect(isFirstVisit()).toBe(false);
    expect(localStorage.getItem('jettae.visited')).toBe('1');
  });

  it('treats unreadable storage as a first visit, once per tab', () => {
    blockStorage();
    expect(isFirstVisit()).toBe(true);
    markVisited();
    expect(isFirstVisit()).toBe(false);
  });

  it('keeps keys in memory for this tab when storage throws', () => {
    blockStorage();
    saveFreeScan(scan('ws-a'));
    saveFreeScan(scan('ws-b'));
    expect(listFreeScans().map((s) => s.workspace_id)).toEqual(['ws-b', 'ws-a']);
    expect(accessKeyFor('ws-a')).toBe('key-ws-a');
    removeFreeScan('ws-a');
    expect(accessKeyFor('ws-a')).toBeNull();
    expect(accessKeyFor('ws-b')).toBe('key-ws-b');
  });

  it('prefers the in-memory copy while writes fail but reads work (quota exceeded)', () => {
    saveFreeScan(scan('ws-old'));
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new DOMException('quota', 'QuotaExceededError');
    });
    saveFreeScan(scan('ws-new'));
    expect(accessKeyFor('ws-new')).toBe('key-ws-new');
    expect(listFreeScans().map((s) => s.workspace_id)).toEqual(['ws-new', 'ws-old']);
    vi.restoreAllMocks();
    // 저장소가 돌아오면 다음 쓰기에서 메모리 사본이 그대로 저장된다
    saveFreeScan(scan('ws-third'));
    resetFreeScanMemory();
    expect(listFreeScans().map((s) => s.workspace_id)).toEqual(['ws-third', 'ws-new', 'ws-old']);
  });

  it('survives a storage block that starts mid-session (keys read earlier stay usable)', () => {
    saveFreeScan(scan('ws-a'));
    expect(accessKeyFor('ws-a')).toBe('key-ws-a');
    blockStorage();
    expect(accessKeyFor('ws-a')).toBe('key-ws-a');
  });

  it(`keeps the newest ${MAX_SCANS} scans`, () => {
    expect(MAX_SCANS).toBe(50);
    for (let i = 0; i < MAX_SCANS + 2; i += 1) saveFreeScan(scan(`ws-${i}`));
    const ids = listFreeScans().map((s) => s.workspace_id);
    expect(ids).toHaveLength(MAX_SCANS);
    expect(ids[0]).toBe(`ws-${MAX_SCANS + 1}`);
    expect(ids).not.toContain('ws-0');
    expect(ids).not.toContain('ws-1');
  });

  it('expires a stored scan once and tells listeners', () => {
    const seen: string[] = [];
    const off = onFreeScanExpired((ws) => seen.push(ws));
    saveFreeScan(scan('ws-gone'));
    expect(expireFreeScan('ws-gone')).toBe(true);
    expect(expireFreeScan('ws-gone')).toBe(false);
    expect(expireFreeScan('ws-never')).toBe(false);
    expect(seen).toEqual(['ws-gone']);
    expect(accessKeyFor('ws-gone')).toBeNull();
    off();
  });
});

describe('client: workspace access key', () => {
  beforeEach(() => {
    localStorage.clear();
    resetFreeScanMemory();
  });
  afterEach(() => vi.restoreAllMocks());

  it('sends the stored key only for that workspace', () => {
    saveFreeScan(scan('ws-mine', 'secret-key'));
    expect(workspaceKeyHeaders('workspaces/ws-mine/report')).toEqual({ [WORKSPACE_KEY_HEADER]: 'secret-key' });
    expect(workspaceKeyHeaders('workspaces/ws-other/report')).toEqual({});
    expect(workspaceKeyHeaders('rules')).toEqual({});
  });

  it('sends the key from memory when storage is blocked', () => {
    blockStorage();
    saveFreeScan(scan('ws-mine', 'secret-key'));
    expect(workspaceKeyHeaders('/api/v1/workspaces/ws-mine/graph?version=1')).toEqual({ [WORKSPACE_KEY_HEADER]: 'secret-key' });
    expect(workspaceKeyHeaders('workspaces/ws-other/report')).toEqual({});
    expect(workspaceKeyHeaders('/app/report')).toEqual({});
  });

  it('adds this browser’s scans to the server list without duplicates', () => {
    saveFreeScan(scan('ws-mine'));
    saveFreeScan(scan('ws-shared'));
    const merged = withLocalFreeScans([{ id: 'ws-shared', name: '팀 진단' }, { id: 'demo', name: '별빛상점', is_demo: true }]);
    expect(merged.map((w) => w.id)).toEqual(['ws-mine', 'ws-shared', 'demo']);
    expect(merged[0]).toMatchObject({ name: '무료 검진', is_free_scan: true, overall: 41.5, finding_count: 7 });
    expect(merged[1]?.is_free_scan).toBeUndefined();
  });
});
