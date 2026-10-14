import type { DiagnosisReport, RehearsalResult } from '../api/types';

/**
 * Rehearsal results on the report's own graph version.
 *
 * After approvals a report also carries, per scenario, its earliest earlier result (the "before"
 * of a fix, for the evidence report's 실패 → 통과 comparison). Anything that shows or counts the
 * *current* state — KPI, sidebar badge, scenario cards — must use only these.
 */
export function currentRehearsals(report: Pick<DiagnosisReport, 'graph_hash' | 'rehearsals'>): RehearsalResult[] {
  return report.rehearsals.filter((r) => r.graph_hash === report.graph_hash);
}
