import { describe, expect, it } from 'vitest';
import type { Patch } from '../api/types';
import { graph, patches } from '../test/fixtures';
import { applyOps, canonicalJson, graphDiff, invertOps, sha256Hex } from './graphOps';

const p1 = patches.find((p) => p.id === 'patch-dmc-01') as Patch;
const p6 = patches.find((p) => p.id === 'patch-dmc-06') as Patch;

describe('applyOps / invertOps / graphDiff', () => {
  it('수정안 적용: 퇴사자 연결이 빠지고 새 담당·상향이 생긴다', () => {
    const { graph: g2 } = applyOps(graph, p1.ops);
    expect(g2.edges['owned_by:service:결제->person:김민준']).toBeUndefined();
    expect(g2.edges['owned_by:service:결제->person:김서연']).toBeDefined();
    expect(g2.edges['escalates_to:person:김서연->person:박재현']?.attrs.after_minutes?.value).toBe(5);
    // 원본은 그대로 (불변)
    expect(graph.edges['owned_by:service:결제->person:김민준']).toBeDefined();
  });

  it('역연산을 적용하면 원래 그래프와 같은 내용(같은 해시)', async () => {
    const { graph: g2, applied } = applyOps(graph, [...p1.ops, ...p6.ops]);
    const { graph: back } = applyOps(g2, invertOps(applied));
    expect(await sha256Hex(canonicalJson(back))).toBe(await sha256Hex(canonicalJson(graph)));
  });

  it('set_attr은 이전 값을 채워 되돌릴 수 있게 한다', () => {
    const { applied } = applyOps(graph, p6.ops);
    const set = applied.find((o) => o.op === 'set_attr');
    expect(set && set.op === 'set_attr' && set.previous?.value).toBe(2880);
  });

  it('버전 비교: 추가·삭제·속성 변경', () => {
    const { graph: g2 } = applyOps(graph, [...p1.ops, ...p6.ops]);
    const d = graphDiff(graph, g2, 1, 2);
    expect(d.removed_edges.map((e) => e.id)).toContain('owned_by:service:결제->person:김민준');
    expect(d.added_edges.map((e) => e.id)).toContain('escalates_to:person:김서연->person:박재현');
    const changed = d.changed_nodes.find((c) => c.id === 'obligation:kisa-침해사고-신고');
    expect(changed?.attrs.map((a) => [a.key, a.before?.value, a.after?.value])).toEqual([['deadline_minutes', 2880, 1440]]);
    expect(graphDiff(graph, graph, 1, 1).added_edges).toHaveLength(0);
  });

  it('canonicalJson은 키 순서와 무관', () => {
    expect(canonicalJson({ b: 1, a: [{ d: 2, c: 3 }] })).toBe(canonicalJson({ a: [{ c: 3, d: 2 }], b: 1 }));
  });
});
