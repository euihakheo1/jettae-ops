import { describe, expect, it } from 'vitest';
import type { RehearsalResult } from '../api/types';
import { isPendingPatch } from './patchSummary';
import { currentRehearsals } from './rehearsals';

const result = (scenario_id: string, graph_hash: string, outcome: 'pass' | 'fail') =>
  ({ scenario_id, graph_hash, outcome }) as unknown as RehearsalResult;

describe('currentRehearsals', () => {
  it('keeps only results on the report graph, even when an older result comes later', () => {
    const report = {
      graph_hash: 'v5',
      rehearsals: [result('s1', 'v5', 'pass'), result('s2', 'v5', 'fail'), result('s1', 'v1', 'fail')],
    };
    const current = currentRehearsals(report);
    expect(current.map((r) => `${r.scenario_id}:${r.outcome}`)).toEqual(['s1:pass', 's2:fail']);
  });

  it('is empty when the report has no results on its own graph', () => {
    expect(currentRehearsals({ graph_hash: 'v2', rehearsals: [result('s1', 'v1', 'fail')] })).toEqual([]);
  });
});

describe('pending patches', () => {
  it('does not count superseded drafts (their base version can never be approved)', () => {
    expect(isPendingPatch({ status: 'superseded' })).toBe(false);
    expect(isPendingPatch({ status: 'validated' })).toBe(true);
  });
});
