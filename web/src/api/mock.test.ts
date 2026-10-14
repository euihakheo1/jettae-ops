import { beforeEach, describe, expect, it } from 'vitest';
import * as mock from './mock';

const WS = mock.DEMO_WORKSPACE_ID;

describe('목업 API (오프라인 데모)', () => {
  beforeEach(() => mock.resetMock());

  it('v1 진단 보고서: 실제 백엔드 출력 그대로', async () => {
    const r = await mock.getReport(WS);
    expect(r.graph_version).toBe(1);
    expect(r.readiness.overall).toBe(55.7);
    expect(r.findings).toHaveLength(39);
    expect(r.llm.judgement_path_calls).toBe(0);
    expect(r.security).toHaveLength(1);
  });

  it('뷰어는 승인할 수 없다', async () => {
    await expect(mock.approvePatch('patch-dmc-01', '데모 뷰어', 'viewer', { skipDelay: true })).rejects.toThrow('팀 리드만');
  });

  it('승인 → 새 버전, 끊김 해소, 점수 재계산(공개 가중치)', async () => {
    const res = await mock.approvePatch('patch-dmc-01', '박재현 팀장', 'lead', { skipDelay: true });
    expect(res.version?.number).toBe(2);
    expect(res.version?.graph_hash).toMatch(/^[0-9a-f]{64}$/);
    const r = await mock.getReport(WS);
    expect(r.graph_version).toBe(2);
    expect(r.findings.some((f) => f.id === 'JT-PPL-001:9dfa62f887')).toBe(false);
    expect(r.readiness.services.find((s) => s.service_id === 'service:결제')?.score).toBe(72);
    // 2번 수정안의 검증은 현재 버전 기준으로 다시 계산된다
    const p2 = (await mock.getPatches(WS)).find((p) => p.id === 'patch-dmc-02')!;
    expect(p2.validation?.score_before).toBe(72);
    expect(p2.validation?.score_after).toBe(96);
    expect(p2.validation?.rehearsals_fixed).toContain('dmc-s1-chuseok-payment');
  });

  it('결제 두 문장 승인 → 추석 새벽 리허설 FAIL → PASS', async () => {
    const before = (await mock.getReport(WS)).rehearsals.find((x) => x.scenario_id === 'dmc-s1-chuseok-payment');
    expect(before?.outcome).toBe('fail');
    await mock.approvePatch('patch-dmc-01', '박재현 팀장', 'lead', { skipDelay: true });
    await mock.approvePatch('patch-dmc-02', '박재현 팀장', 'lead', { skipDelay: true });
    const [after] = await mock.runRehearsals(WS, { scenario_ids: ['dmc-s1-chuseok-payment'] });
    expect(after?.outcome).toBe('pass');
    expect(after?.time_to_mitigate).toBe(28);
    expect(after?.graph_hash).toBe((await mock.getVersions(WS)).at(-1)?.graph_hash);
  });

  it('메모 수정안과 검증 탈락 수정안은 승인할 수 없다', async () => {
    await expect(mock.approvePatch('patch-dmc-08', 'x', 'lead', { skipDelay: true })).rejects.toThrow('메모');
    await expect(mock.approvePatch('patch-dmc-03', 'x', 'lead', { skipDelay: true })).rejects.toThrow('검증');
  });

  it('되돌리기는 새 버전으로 남고 이전 내용으로 돌아간다', async () => {
    await mock.approvePatch('patch-dmc-01', '박재현 팀장', 'lead', { skipDelay: true });
    const v3 = await mock.revertVersion(WS, 2, '박재현 팀장', 'lead');
    expect(v3.number).toBe(3);
    expect(v3.revert_of).toBe(2);
    const versions = await mock.getVersions(WS);
    expect(versions.map((v) => v.number)).toEqual([1, 2, 3]);
    expect(v3.graph_hash).not.toBe(versions[1]!.graph_hash);
    const r = await mock.getReport(WS);
    expect(r.readiness.overall).toBe(55.7);
    const diff = await mock.getDiff(WS, 1, 3);
    expect(diff.added_edges).toHaveLength(0);
    expect(diff.removed_edges).toHaveLength(0);
  });

  it('?mock=after 프리셋: 결제 두 문장 + 침해사고 문장', async () => {
    await mock.applyPreset('after');
    const r = await mock.getReport(WS);
    expect(r.graph_version).toBe(4);
    expect(r.readiness.overall).toBe(65.8);
    expect(r.rehearsals.find((x) => x.scenario_id === 'dmc-s4-breach-report')?.outcome).toBe('pass');
  });

  it('거절은 사유를 남긴다', async () => {
    const p = await mock.rejectPatch('patch-dmc-04', '명부 보완 후 다시', '박재현 팀장', 'lead');
    expect(p.status).toBe('rejected');
    expect(p.validation?.notes.at(-1)).toBe('거절 사유: 명부 보완 후 다시');
  });

  it('사전 신청은 개인정보 동의가 필수', async () => {
    await expect(
      mock.signup({ email: 'a@b.example', org_name: 'x', team_size: '1~10명', message: '', consent_privacy: false as unknown as true, consent_marketing: false, source: 't' }),
    ).rejects.toThrow('동의');
  });
});
