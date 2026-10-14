import { describe, expect, it } from 'vitest';
import type { Patch, PatchOp } from '../api/types';
import { graph, patches } from '../test/fixtures';
import { isMemoPatch, isPendingPatch, lookupFrom, parseEdgeId, peopleInPatch, summarizeOp, summarizePatch } from './patchSummary';

const lookup = lookupFrom(graph.nodes);
const byId = (id: string): Patch => patches.find((p) => p.id === id)!;

describe('summarizePatch — 수정안 op를 사람이 읽는 한국어로', () => {
  it('결제 담당 교체·상향 추가', () => {
    const lines = summarizePatch(byId('patch-dmc-01'), lookup).map((s) => s.text);
    expect(lines).toEqual([
      '− 담당: 결제 → 김민준 대리 (퇴사)',
      '+ 담당: 결제 → 김서연 사원',
      '− 호출: 결제 서비스 실패율 알람(5% 초과 3분 지속) → 김민준 대리 (퇴사)',
      '+ 호출: 결제 서비스 실패율 알람(5% 초과 3분 지속) → 김서연 사원',
      '− 호출: 결제 서비스 장애 지속 → 김민준 대리 (퇴사)',
      '+ 호출: 결제 서비스 장애 지속 → 김서연 사원',
      '+ 상향: 김서연 사원 → 박재현 팀장 (5분, 전화)',
    ]);
  });

  it('승인·실행 연결', () => {
    expect(summarizePatch(byId('patch-dmc-02'), lookup).map((s) => s.text)).toEqual([
      '+ 승인: 결제 서비스 롤백 → 박재현 팀장',
      '+ 실행: 결제 서비스 롤백 → 김서연 사원',
    ]);
  });

  it('속성 변경은 이전 값 → 새 값 (신고 기한 48시간 → 24시간)', () => {
    const [first] = summarizePatch(byId('patch-dmc-06'), lookup);
    expect(first!.sign).toBe('~');
    expect(first!.text).toBe('~ 신고 기한: KISA 침해사고 신고 48시간 → 24시간');
  });

  it('같은 수정안이 추가하는 노드 이름도 해석한다', () => {
    const lines = summarizePatch(byId('patch-dmc-07'), lookup).map((s) => s.text);
    expect(lines[0]).toBe('+ 신고 의무: 개인정보 유출 신고');
    expect(lines).toContain('+ 신고 담당: 개인정보 유출 신고 → 윤아름 과장');
    expect(lines).toContain('+ 신고 의무: 결제 → 개인정보 유출 신고');
  });

  it('스냅숏 없는 삭제는 id에서 복원', () => {
    const op: PatchOp = { op: 'remove_edge', edge_id: 'escalates_to:person:최영훈->person:정민혁', snapshot: null };
    expect(summarizeOp(op, lookup).text).toBe('− 상향: 최영훈 차장 → 정민혁 과장');
  });

  it('속성 삭제', () => {
    const op: PatchOp = { op: 'set_attr', target_id: 'obligation:kisa-침해사고-신고', key: 'deadline_minutes', value: null, previous: null };
    expect(summarizeOp(op, lookup).text).toBe('− 신고 기한: KISA 침해사고 신고 (48시간)');
  });

  it('결정적: 같은 입력이면 같은 문장', () => {
    const a = summarizePatch(byId('patch-dmc-05'), lookup);
    const b = summarizePatch(byId('patch-dmc-05'), lookup);
    expect(a).toEqual(b);
  });
});

describe('보조 함수', () => {
  it('edge id 파싱', () => {
    expect(parseEdgeId('approved_by:action:a->person:b')).toEqual({ kind: 'approved_by', src: 'action:a', dst: 'person:b' });
    expect(parseEdgeId('notifies:trigger:x->person:y#disc')).toEqual({ kind: 'notifies', src: 'trigger:x', dst: 'person:y' });
    expect(parseEdgeId('garbage')).toBeNull();
  });

  it('대기 중 판별 — 수정안 탭 숫자와 왼쪽 메뉴 배지가 같은 기준', () => {
    expect(isPendingPatch({ status: 'validated' })).toBe(true);
    expect(isPendingPatch({ status: 'rejected_by_validator' })).toBe(true);
    expect(isPendingPatch({ status: 'approved' })).toBe(false);
    expect(isPendingPatch({ status: 'rejected' })).toBe(false);
  });

  it('메모 수정안(그래프 변경 없음) 판별', () => {
    expect(isMemoPatch(byId('patch-dmc-08'))).toBe(true);
    expect(isMemoPatch(byId('patch-dmc-01'))).toBe(false);
  });

  it('수정안의 사람은 모두 명부(그래프)에 있다 — 명부 외 인물 0명', () => {
    for (const p of patches) for (const id of peopleInPatch(p)) expect(graph.nodes[id], `${p.id}: ${id}`).toBeDefined();
  });
});
