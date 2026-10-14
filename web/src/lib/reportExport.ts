/**
 * 목업/오프라인 모드용 증적 리포트 내보내기 (Markdown · HTML · SARIF).
 * API 모드에서는 백엔드(reporting/markdown.py · html · sarif)가 만든 파일을 그대로 내려받는다.
 * 이 파일은 같은 DiagnosisReport를 간단한 형식으로 옮길 뿐, 아무것도 판정하지 않는다.
 */
import type { DiagnosisReport, Finding, GraphVersion, RuleInfo, Severity, SourceDocument } from '../api/types';
import { formatDateTime, formatPercent, formatScore, hashPrefix } from './format';
import { CONTROL_STATUS_LABEL, controlRows } from './isms';
import { ELEMENT_LABEL, ELEMENT_ORDER, ELEMENT_STATUS_LABEL, SEVERITY_LABEL, SEVERITY_ORDER } from './labels';
import { currentRehearsals } from './rehearsals';

export const EVIDENCE_TITLE = '런북 준비도 진단 리포트 (ISMS-P 증적 초안)';
const MOCK_NOTE = '> 이 파일은 오프라인 데모 데이터로 브라우저에서 만든 미리보기입니다. 실제 증적은 제때옵스 서버의 리포트를 사용하세요.';

function serviceName(report: DiagnosisReport, id: string | null): string {
  if (!id) return '전사';
  return report.readiness.services.find((s) => s.service_id === id)?.service_name ?? id.replace(/^service:/, '');
}

function docTitle(docs: readonly SourceDocument[], id: string): string {
  return docs.find((d) => d.doc_id === id)?.title ?? id;
}

function mdEscape(text: string): string {
  return text.replace(/\|/g, '\\|').replace(/\n/g, ' ');
}

export function reportMarkdown(
  report: DiagnosisReport,
  versions: readonly GraphVersion[],
  docs: readonly SourceDocument[],
): string {
  const out: string[] = [];
  const current = currentRehearsals(report);
  const passed = current.filter((r) => r.outcome === 'pass').length;
  out.push(`# ${EVIDENCE_TITLE}`, '', MOCK_NOTE, '');
  out.push('## 1. 개요', '');
  out.push(`- 그래프 버전: v${report.graph_version} (\`${hashPrefix(report.graph_hash, 12)}\`)`);
  out.push(`- 진단 시각: ${formatDateTime(report.created_at)} (KST)`);
  out.push(`- 준비도 점수: **${formatScore(report.readiness.overall)}** / 100 (가중치 ${report.readiness.weights_version})`);
  out.push(`- 끊김: ${report.findings.length}건 · 리허설 작동률 ${formatPercent(passed, current.length)}`);
  out.push(`- LLM 사용: ${report.llm.mode.toUpperCase()} · 판정·점수·리허설 경로 LLM 호출 ${report.llm.judgement_path_calls}회`, '');
  out.push('## 2. 서비스별 준비도', '');
  out.push(`| 서비스 | 등급 | 점수 | ${ELEMENT_ORDER.map((e) => ELEMENT_LABEL[e]).join(' | ')} |`);
  out.push(`|---|---:|---:|${ELEMENT_ORDER.map(() => '---').join('|')}|`);
  for (const s of [...report.readiness.services].sort((a, b) => a.tier - b.tier || a.score - b.score)) {
    const cells = ELEMENT_ORDER.map((e) => {
      const el = s.elements.find((x) => x.element === e);
      return el ? ELEMENT_STATUS_LABEL[el.status] : '—';
    });
    out.push(`| ${s.service_name} | ${s.tier} | ${formatScore(s.score)} | ${cells.join(' | ')} |`);
  }
  out.push('', '## 3. 끊긴 대응 경로', '');
  for (const sev of SEVERITY_ORDER) {
    const group = report.findings.filter((f) => f.severity === sev);
    if (!group.length) continue;
    out.push(`### ${SEVERITY_LABEL[sev]} (${group.length}건)`, '');
    for (const f of group) {
      out.push(`#### ${f.rule_id} ${f.title} — ${serviceName(report, f.service_id)}`, '', f.message, '');
      for (const s of dedupeEvidence(f).slice(0, 2)) out.push(`> ${docTitle(docs, s.doc_id)} : ${s.line ?? '?'}줄 — “${s.quote.replace(/\n/g, ' ')}”`);
      if (f.legal_basis) out.push('', `법적 근거: ${f.legal_basis}`);
      out.push('');
    }
  }
  out.push('## 4. 모의 장애 리허설', '');
  out.push('| 시나리오 | 결과 | 완화 시간 | 시한 | 끊긴 지점 |', '|---|---|---:|---:|---|');
  for (const r of current) {
    const sc = report.scenarios.find((s) => s.id === r.scenario_id);
    out.push(
      `| ${mdEscape(sc?.title ?? r.scenario_id)} | ${r.outcome === 'pass' ? '통과' : '실패'} | ${r.time_to_mitigate ?? '—'}분 | ${r.deadline_minutes}분 | ${mdEscape(r.failed_at_id ?? '')} |`,
    );
  }
  out.push('', '## 5. 그래프 버전 이력', '');
  for (const v of versions)
    out.push(`- v${v.number} \`${hashPrefix(v.graph_hash)}\` · ${formatDateTime(v.created_at)} · ${v.created_by}${v.revert_of ? ` · v${v.revert_of} 되돌리기` : ''} — ${v.note}`);
  out.push('', '## 6. ISMS-P 매핑', '', '| 기준 | 항목 | 증적 | 상태 |', '|---|---|---|---|');
  for (const c of controlRows(report.findings, current, report.scenarios))
    out.push(`| ${c.id} | ${c.title} | ${mdEscape(c.evidence)} | ${CONTROL_STATUS_LABEL[c.status]} |`);
  out.push('', '---', '인증 심사의 판단을 대신하지 않습니다. 판정·점수는 고정 규칙(LLM 0회)으로 계산했습니다.', '');
  return out.join('\n');
}

function dedupeEvidence(f: Finding): Finding['evidence'] {
  const seen = new Set<string>();
  return f.evidence.filter((s) => {
    const k = `${s.doc_id}:${s.line}`;
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
}

function esc(text: string): string {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

/** 아주 단순한 Markdown → HTML (리포트 미리보기·HTML 내려받기용) */
export function markdownToHtml(md: string): string {
  const lines = md.split('\n');
  const html: string[] = [];
  let inTable = false;
  let inList = false;
  const inline = (t: string): string =>
    esc(t)
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/`([^`]+)`/g, '<code>$1</code>');
  const close = (): void => {
    if (inTable) html.push('</tbody></table>');
    if (inList) html.push('</ul>');
    inTable = false;
    inList = false;
  };
  for (const line of lines) {
    if (/^\|.*\|$/.test(line)) {
      if (/^\|[-:|\s]+\|$/.test(line)) continue;
      const cells = line.slice(1, -1).split(/(?<!\\)\|/).map((c) => inline(c.trim().replace(/\\\|/g, '|')));
      if (!inTable) {
        close();
        html.push('<table><thead><tr>' + cells.map((c) => `<th>${c}</th>`).join('') + '</tr></thead><tbody>');
        inTable = true;
      } else html.push('<tr>' + cells.map((c) => `<td>${c}</td>`).join('') + '</tr>');
      continue;
    }
    if (line.startsWith('- ')) {
      if (!inList) {
        close();
        html.push('<ul>');
        inList = true;
      }
      html.push(`<li>${inline(line.slice(2))}</li>`);
      continue;
    }
    close();
    const h = /^(#{1,4}) (.*)$/.exec(line);
    if (h) html.push(`<h${h[1]!.length}>${inline(h[2]!)}</h${h[1]!.length}>`);
    else if (line.startsWith('> ')) html.push(`<blockquote>${inline(line.slice(2))}</blockquote>`);
    else if (line === '---') html.push('<hr/>');
    else if (line.trim()) html.push(`<p>${inline(line)}</p>`);
  }
  close();
  return html.join('\n');
}

export function reportHtml(md: string): string {
  return `<!doctype html><html lang="ko"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/><title>${EVIDENCE_TITLE}</title><style>
body{font-family:'Pretendard Variable',Pretendard,system-ui,sans-serif;max-width:960px;margin:40px auto;padding:0 24px;color:#111;line-height:1.7;word-break:keep-all}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:14px}th,td{border:1px solid #ccd;padding:6px 8px;text-align:left}th{background:#f3f4f8}
blockquote{margin:8px 0;padding:8px 12px;border-left:4px solid #6470f0;background:#f5f6ff}code{background:#f0f0f4;padding:1px 4px;border-radius:4px}
h1{font-size:28px}h2{margin-top:36px;border-bottom:2px solid #e3e5ee;padding-bottom:4px}
</style></head><body>${markdownToHtml(md)}</body></html>`;
}

const SARIF_LEVEL: Record<Severity, 'error' | 'warning' | 'note'> = {
  critical: 'error',
  high: 'error',
  medium: 'warning',
  low: 'note',
  info: 'note',
};

export function reportSarif(report: DiagnosisReport, rules: readonly RuleInfo[], docs: readonly SourceDocument[]): string {
  const docPath = (id: string): string => docs.find((d) => d.doc_id === id)?.path ?? id;
  const usedRules = [...new Set(report.findings.map((f) => f.rule_id))].sort();
  const sarif = {
    $schema: 'https://json.schemastore.org/sarif-2.1.0.json',
    version: '2.1.0',
    runs: [
      {
        tool: {
          driver: {
            name: 'jettae',
            informationUri: 'https://jettae.example/rules',
            rules: usedRules.map((id) => {
              const r = rules.find((x) => x.rule_id === id);
              return {
                id,
                name: id,
                shortDescription: { text: r?.title ?? id },
                fullDescription: { text: r?.rationale ?? '' },
                defaultConfiguration: { level: SARIF_LEVEL[r?.severity ?? 'medium'] },
              };
            }),
          },
        },
        results: report.findings.map((f) => ({
          ruleId: f.rule_id,
          level: SARIF_LEVEL[f.severity],
          message: { text: `${f.title} — ${f.message}` },
          locations: f.evidence.slice(0, 1).map((s) => ({
            physicalLocation: {
              artifactLocation: { uri: docPath(s.doc_id) },
              region: { startLine: s.line ?? 1, snippet: { text: s.quote } },
            },
          })),
          partialFingerprints: { jettaeFindingId: f.id },
        })),
      },
    ],
  };
  return JSON.stringify(sarif, null, 2);
}
