/**
 * ISMS-P 인증 기준 매핑 — 백엔드 reporting/evidence.py 의 _control_rows 와 같은 판정 규칙.
 * 증적이 없으면 '증적 없음'(만족으로 치지 않음, 확인 불가 ≠ 정상).
 */
import type { Finding, RehearsalResult, Scenario } from '../api/types';
import { formatPercent } from './format';

export type ControlStatus = 'ok' | 'gap' | 'missing';

export const CONTROL_STATUS_LABEL: Record<ControlStatus, string> = {
  ok: '이상 없음(고정 규칙 기준)',
  gap: '보완 필요',
  missing: '증적 없음',
};

export interface ControlRow {
  id: string;
  title: string;
  section: string;
  evidence: string;
  status: ControlStatus;
}

export const LEGAL_RULE_PREFIXES = ['JT-KR-', 'JT-DDL-'] as const;
export const DISASTER_RULE_IDS = ['JT-CAT-003', 'JT-CHN-001'] as const;

interface Row {
  result: RehearsalResult;
  scenario: Scenario | undefined;
}

function tally(rows: readonly Row[]): [number, number] {
  const passed = rows.filter((r) => r.result.outcome === 'pass').length;
  return [passed, rows.length - passed];
}

export function controlRows(
  findings: readonly Finding[],
  rehearsals: readonly RehearsalResult[],
  scenarios: readonly Scenario[],
): ControlRow[] {
  const byId = new Map(scenarios.map((s) => [s.id, s]));
  const rows: Row[] = rehearsals.map((r) => ({ result: r, scenario: byId.get(r.scenario_id) }));
  const out: ControlRow[] = [];

  const [passed, failed] = tally(rows);
  out.push({
    id: '2.11.4',
    title: '사고 대응 훈련 및 개선',
    section: '리허설 결과',
    evidence: rows.length
      ? `모의 장애 ${rows.length}건 결정적 재생 · 통과 ${passed}건 · 실패 ${failed}건 (통과율 ${formatPercent(passed, rows.length)})`
      : '리허설 기록이 없습니다. 모의 장애 리허설을 실행하세요.',
    status: rows.length ? (failed ? 'gap' : 'ok') : 'missing',
  });

  const legal = findings.filter((f) => LEGAL_RULE_PREFIXES.some((p) => f.rule_id.startsWith(p)));
  const legalBroken = legal.filter((f) => f.status === 'broken').length;
  const legalRules = [...new Set(legal.map((f) => f.rule_id))].sort().join(', ');
  const reporting = rows.filter((r) => r.scenario && r.scenario.incident_type !== 'outage');
  const [rPassed, rFailed] = tally(reporting);
  out.push({
    id: '2.11.5',
    title: '사고 대응 및 복구(법적 통지·신고)',
    section: '신고·시한 규칙(JT-KR·JT-DDL), 신고 리허설',
    evidence:
      `신고·시한 규칙 끊김 ${legalBroken}건 · 확인 불가 ${legal.length - legalBroken}건` +
      (legalRules ? ` (${legalRules})` : '') +
      ` · 신고 리허설 ${reporting.length}건 중 통과 ${rPassed}건`,
    status: legal.length || rFailed ? 'gap' : reporting.length ? 'ok' : 'missing',
  });

  const disaster = findings.filter((f) => (DISASTER_RULE_IDS as readonly string[]).includes(f.rule_id));
  const center = rows.filter((r) => r.scenario?.perturbations.some((p) => p.kind === 'location_down'));
  const [cPassed, cFailed] = tally(center);
  out.push({
    id: '2.12.1',
    title: '재해·재난 대비 안전조치',
    section: '한 위치 집중·채널 동반 장애',
    evidence:
      DISASTER_RULE_IDS.map((id) => `${id} ${disaster.filter((f) => f.rule_id === id).length}건`).join(' · ') +
      ` · 센터 장애 시나리오 ${center.length}건`,
    status: disaster.length ? 'gap' : center.length ? 'ok' : 'missing',
  });

  out.push({
    id: '2.12.2',
    title: '재해 복구 시험 및 개선',
    section: '센터 장애 리허설',
    evidence: center.length
      ? `센터 장애 시나리오 ${center.length}건 재생 · 통과 ${cPassed}건 · 실패 ${cFailed}건`
      : '센터(위치) 장애 시나리오 리허설 기록이 없습니다.',
    status: center.length ? (cFailed ? 'gap' : 'ok') : 'missing',
  });
  return out;
}
