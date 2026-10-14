/**
 * 목업 API — VITE_USE_MOCKS=1 이거나 API에 닿지 않을 때(오프라인 데모) 쓴다.
 *
 * 읽기 엔드포인트는 src/mocks/*.json(백엔드가 별빛상점 예제로 만든 실제 출력)을 그대로 돌려준다.
 * 쓰기(승인·거절·되돌리기·리허설 실행)는 브라우저 메모리에서만 흉내 낸다:
 *  - 승인: 수정안 op를 그래프에 적용해 새 버전(내용 해시)을 만든다
 *  - 끊김: 승인된 수정안이 해결하는 finding을 뺀다
 *  - 점수: 수정안의 요소 효과(demo-overlay.json)를 공개 가중치(SCORING_V1)로 다시 계산한다
 *  - 리허설: 미리 만든 '수정 후' 결과(demo-overlay.json)를 조건(필요 수정안)에 맞춰 고른다
 * 새로고침하면 처음 상태로 돌아간다. 실제 판정은 백엔드 규칙·리플레이가 한다.
 */
import documentsJson from '../mocks/documents.json';
import findingsJson from '../mocks/findings.json';
import graphJson from '../mocks/graph.json';
import overlayJson from '../mocks/demo-overlay.json';
import patchesJson from '../mocks/patches.json';
import readinessJson from '../mocks/readiness.json';
import rehearsalsJson from '../mocks/rehearsals.json';
import rulesJson from '../mocks/rules.json';
import scenariosJson from '../mocks/scenarios.json';
import versionsJson from '../mocks/versions.json';
import { graphViewFromOpsGraph } from '../graph/model';
import { roundHalfUp1 } from '../lib/format';
import { applyOps, canonicalJson, graphDiff, invertOps, sha256Hex } from '../lib/graphOps';
import { reportHtml, reportMarkdown, reportSarif } from '../lib/reportExport';
import { ApiError } from './errors';
import type {
  DiagnosisReport,
  DraftRequest,
  ElementScore,
  EventRequest,
  Finding,
  GraphDiff,
  GraphVersion,
  GraphView,
  GroundingStatus,
  OpsGraph,
  Patch,
  PatchOp,
  ReadinessReport,
  RehearsalRequest,
  RehearsalResult,
  Role,
  RuleInfo,
  Scenario,
  SecurityNotice,
  SignupRequest,
  SourceDocument,
  WorkspaceSummary,
} from './types';

// --------------------------------------------------------------------------- fixtures

interface Overlay {
  note: string;
  patch_effects: Record<string, { service_id: string; elements: ElementScore[] }>;
  human_scenarios: Scenario[];
  human_rehearsals: RehearsalResult[];
  after_rehearsals: { scenario_id: string; requires_patches: string[]; result: RehearsalResult }[];
}

const BASE_GRAPH = graphJson as unknown as OpsGraph;
const BASE_FINDINGS = findingsJson as unknown as Finding[];
const BASE_READINESS = readinessJson as unknown as ReadinessReport;
const TEMPLATE_SCENARIOS = scenariosJson as unknown as Scenario[];
const TEMPLATE_REHEARSALS = rehearsalsJson as unknown as RehearsalResult[];
const RULES = rulesJson as unknown as RuleInfo[];
const DOCS = documentsJson as unknown as SourceDocument[];
const PATCHES = patchesJson as unknown as Patch[];
const VERSIONS = versionsJson as unknown as GraphVersion[];
const OVERLAY = overlayJson as unknown as Overlay;

export const DEMO_WORKSPACE_ID = 'dmc-shop';
const SCENARIOS: Scenario[] = [...OVERLAY.human_scenarios, ...TEMPLATE_SCENARIOS];
const BASE_REHEARSALS: RehearsalResult[] = [...OVERLAY.human_rehearsals, ...TEMPLATE_REHEARSALS];
/** 정책 기본값: 핵심 등급 1·2 (docs/scoring.md) */
const CRITICAL_TIERS = [1, 2];

// --------------------------------------------------------------------------- state

interface VersionState {
  meta: GraphVersion;
  graph: OpsGraph;
  /** 이 버전에 반영돼 있는 수정안 id (적용 순서) */
  effective: string[];
  /** 이 버전을 만든 op (되돌리기용) */
  applied: PatchOp[];
}

interface State {
  versions: VersionState[];
  patches: Patch[];
  signups: SignupRequest[];
  events: EventRequest[];
}

const clone = <T>(v: T): T => structuredClone(v);

function freshState(): State {
  const first = VERSIONS[0]!;
  return {
    versions: [{ meta: clone(first), graph: BASE_GRAPH, effective: [], applied: [] }],
    patches: clone(PATCHES),
    signups: [],
    events: [],
  };
}

let state: State = freshState();

export function resetMock(): void {
  state = freshState();
}

const wait = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));
const READ_MS = 90;
const WRITE_MS = 420;

function latest(): VersionState {
  return state.versions[state.versions.length - 1]!;
}

function versionState(n?: number | null): VersionState {
  if (n === undefined || n === null) return latest();
  const v = state.versions.find((x) => x.meta.number === n);
  if (!v) throw new ApiError(404, `그래프 버전 v${n}을(를) 찾을 수 없습니다.`);
  return v;
}

function nowIso(): string {
  return new Date().toISOString();
}

// --------------------------------------------------------------------------- derived

function resolvedBy(effective: readonly string[]): Set<string> {
  const ids = new Set<string>();
  for (const pid of effective) for (const f of state.patches.find((p) => p.id === pid)?.finding_ids ?? []) ids.add(f);
  return ids;
}

function findingsFor(effective: readonly string[]): Finding[] {
  const resolved = resolvedBy(effective);
  return BASE_FINDINGS.filter((f) => !resolved.has(f.id));
}

function serviceScore(elements: readonly ElementScore[]): number {
  const w = BASE_READINESS.element_weights;
  const sum = elements.reduce((acc, e) => acc + (w[e.element] ?? 0) * e.score, 0);
  return roundHalfUp1(Math.min(100, Math.max(0, sum * 100)));
}

function readinessFor(effective: readonly string[], graphHash: string): ReadinessReport {
  const resolved = resolvedBy(effective);
  const services = BASE_READINESS.services.map((s) => {
    let elements = s.elements.map((e) => ({ ...e, finding_ids: e.finding_ids.filter((id) => !resolved.has(id)) }));
    for (const pid of effective) {
      const eff = OVERLAY.patch_effects[pid];
      if (!eff || eff.service_id !== s.service_id) continue;
      elements = elements.map((e) => eff.elements.find((x) => x.element === e.element) ?? e);
    }
    return { ...s, elements, score: serviceScore(elements) };
  });
  const tierW = BASE_READINESS.tier_weights;
  const critical = services.filter((s) => CRITICAL_TIERS.includes(s.tier));
  const pool = critical.length ? critical : services;
  const wsum = pool.reduce((a, s) => a + (tierW[String(s.tier)] ?? 1), 0);
  const overall = wsum ? roundHalfUp1(pool.reduce((a, s) => a + (tierW[String(s.tier)] ?? 1) * s.score, 0) / wsum) : 0;
  return { ...BASE_READINESS, services, overall, graph_hash: graphHash };
}

function resultFor(scenarioId: string, effective: readonly string[], graphHash: string): RehearsalResult | undefined {
  const eff = new Set(effective);
  const candidates = OVERLAY.after_rehearsals
    .filter((a) => a.scenario_id === scenarioId && a.requires_patches.every((p) => eff.has(p)))
    .sort((a, b) => b.requires_patches.length - a.requires_patches.length);
  const base = candidates[0]?.result ?? BASE_REHEARSALS.find((r) => r.scenario_id === scenarioId);
  return base ? { ...clone(base), graph_hash: graphHash } : undefined;
}

function rehearsalsFor(effective: readonly string[], graphHash: string): RehearsalResult[] {
  return SCENARIOS.map((s) => resultFor(s.id, effective, graphHash)).filter((r): r is RehearsalResult => !!r);
}

/** 아직 반영되지 않은 수정안의 검증 결과를 '현재 버전 기준'으로 다시 계산 */
function liveValidation(p: Patch, cur: VersionState): Patch {
  if (!p.validation || p.status === 'approved' || p.status === 'rejected' || p.status === 'rejected_by_validator') return p;
  const eff = OVERLAY.patch_effects[p.id];
  const nextEffective = [...cur.effective, p.id];
  const curFindings = new Set(findingsFor(cur.effective).map((f) => f.id));
  let scoreBefore: number | null = null;
  let scoreAfter: number | null = null;
  if (eff) {
    const before = readinessFor(cur.effective, cur.meta.graph_hash).services.find((s) => s.service_id === eff.service_id);
    const after = readinessFor(nextEffective, cur.meta.graph_hash).services.find((s) => s.service_id === eff.service_id);
    scoreBefore = before?.score ?? null;
    scoreAfter = after?.score ?? null;
  }
  const fixed: string[] = [];
  const regressed: string[] = [];
  for (const s of SCENARIOS) {
    const a = resultFor(s.id, cur.effective, '')?.outcome;
    const b = resultFor(s.id, nextEffective, '')?.outcome;
    if (a === 'fail' && b === 'pass') fixed.push(s.id);
    if (a === 'pass' && b === 'fail') regressed.push(s.id);
  }
  return {
    ...p,
    base_version: cur.meta.number,
    validation: {
      ...p.validation,
      resolved_finding_ids: p.finding_ids.filter((id) => curFindings.has(id)),
      rehearsals_fixed: fixed,
      rehearsals_regressed: regressed,
      score_before: scoreBefore,
      score_after: scoreAfter,
    },
  };
}

function groundingFor(graph: OpsGraph): GroundingStatus {
  const spans = new Set<string>();
  const visit = (prov: { spans: { doc_id: string; start: number; end: number }[] }[]): void => {
    for (const p of prov) for (const s of p.spans) spans.add(`${s.doc_id}:${s.start}:${s.end}`);
  };
  for (const n of Object.values(graph.nodes)) visit(n.prov);
  for (const e of Object.values(graph.edges)) visit(e.prov);
  const quarantined = DOCS.reduce((a, d) => a + d.quarantined.length, 0);
  return {
    grounded: spans.size,
    fuzzy_grounded: 0,
    dropped_ungrounded: 0,
    dropped_quarantined: quarantined,
    dropped_invalid: 0,
    held_backtranslation: 0,
    held_ambiguous_entity: 0,
    dropped_examples: [],
  };
}

function securityNotices(): SecurityNotice[] {
  const out: SecurityNotice[] = [];
  for (const d of DOCS)
    for (const [start, end] of d.quarantined)
      out.push({ doc_id: d.doc_id, start, end, excerpt: d.text.slice(start, end), pattern: 'addressed-to-ai' });
  return out;
}

function heldIds(graph: OpsGraph): string[] {
  return [
    ...Object.values(graph.nodes).filter((n) => n.verification === 'hold').map((n) => n.id),
    ...Object.values(graph.edges).filter((e) => e.verification === 'hold').map((e) => e.id),
  ].sort();
}

function buildReport(v: VersionState): DiagnosisReport {
  const findings = findingsFor(v.effective);
  return {
    run_id: `mock-run-v${v.meta.number}`,
    workspace_id: DEMO_WORKSPACE_ID,
    graph_version: v.meta.number,
    graph_hash: v.meta.graph_hash,
    created_at: v.meta.created_at,
    findings,
    readiness: readinessFor(v.effective, v.meta.graph_hash),
    scenarios: SCENARIOS,
    rehearsals: rehearsalsFor(v.effective, v.meta.graph_hash),
    patches: state.patches.map((p) => liveValidation(p, v)),
    grounding: groundingFor(BASE_GRAPH),
    security: securityNotices(),
    llm: {
      mode: 'off',
      provider: null,
      model: null,
      calls: 0,
      input_tokens: 0,
      output_tokens: 0,
      cache_read_tokens: 0,
      by_task: {},
      judgement_path_calls: 0,
    },
    held_element_ids: heldIds(v.graph),
    catalog_gaps: [...new Set(findings.filter((f) => f.rule_id === 'JT-CAT-001').map((f) => f.service_id ?? ''))]
      .filter(Boolean)
      .sort(),
    timings_ms: {},
  };
}

// --------------------------------------------------------------------------- public API (client.ts가 호출)

export interface ApproveResult {
  patch: Patch;
  version: GraphVersion | null;
}

export async function listWorkspaces(): Promise<WorkspaceSummary[]> {
  await wait(READ_MS);
  const v = latest();
  return [
    {
      id: DEMO_WORKSPACE_ID,
      name: '별빛상점 (DMC 예제)',
      created_at: VERSIONS[0]?.created_at ?? null,
      latest_version: v.meta.number,
      graph_hash: v.meta.graph_hash,
      overall: readinessFor(v.effective, v.meta.graph_hash).overall,
      finding_count: findingsFor(v.effective).length,
      is_demo: true,
    },
  ];
}

export async function createDemo(): Promise<{ id: string }> {
  await wait(WRITE_MS);
  return { id: DEMO_WORKSPACE_ID };
}

export async function getReport(_ws: string, version?: number | null): Promise<DiagnosisReport> {
  await wait(READ_MS);
  return buildReport(versionState(version));
}

export async function getGraph(_ws: string, version?: number | null): Promise<GraphView> {
  await wait(READ_MS);
  const v = versionState(version);
  return graphViewFromOpsGraph(v.graph, findingsFor(v.effective), v.meta.number, v.meta.graph_hash);
}

export async function getVersions(_ws: string): Promise<GraphVersion[]> {
  await wait(READ_MS);
  return state.versions.map((v) => clone(v.meta));
}

export async function getDiff(_ws: string, a: number, b: number): Promise<GraphDiff> {
  await wait(READ_MS);
  return graphDiff(versionState(a).graph, versionState(b).graph, a, b);
}

export async function getPatches(_ws: string): Promise<Patch[]> {
  await wait(READ_MS);
  const cur = latest();
  return state.patches.map((p) => liveValidation(p, cur));
}

export async function draftPatches(_ws: string, req: DraftRequest): Promise<Patch[]> {
  await wait(WRITE_MS);
  const cur = latest();
  const wanted = new Set(req.finding_ids ?? []);
  return state.patches
    .filter((p) => p.status !== 'approved' && (wanted.size === 0 || p.finding_ids.some((f) => wanted.has(f))))
    .map((p) => liveValidation(p, cur));
}

export async function getRules(): Promise<RuleInfo[]> {
  await wait(READ_MS);
  return RULES;
}

export async function getDocuments(_ws: string): Promise<SourceDocument[]> {
  await wait(READ_MS);
  return DOCS;
}

export async function runRehearsals(_ws: string, req: RehearsalRequest): Promise<RehearsalResult[]> {
  await wait(WRITE_MS + 300);
  const v = latest();
  const ids = req.scenario_ids && req.scenario_ids.length ? new Set(req.scenario_ids) : null;
  return rehearsalsFor(v.effective, v.meta.graph_hash).filter((r) => !ids || ids.has(r.scenario_id));
}

function requireLead(role: Role): void {
  if (role !== 'lead' && role !== 'admin') throw new ApiError(403, '팀 리드만 승인할 수 있습니다.');
}

async function commitVersion(ops: readonly PatchOp[], meta: Omit<GraphVersion, 'graph_hash' | 'number' | 'parent' | 'workspace_id'>, effective: string[]): Promise<GraphVersion> {
  const prev = latest();
  const { graph, applied } = applyOps(prev.graph, ops);
  const hash = await sha256Hex(canonicalJson(graph));
  const version: GraphVersion = {
    workspace_id: DEMO_WORKSPACE_ID,
    number: prev.meta.number + 1,
    parent: prev.meta.number,
    graph_hash: hash,
    ...meta,
  };
  state.versions.push({ meta: version, graph, effective, applied });
  return clone(version);
}

export async function approvePatch(id: string, user: string, role: Role, opts: { skipDelay?: boolean } = {}): Promise<ApproveResult> {
  if (!opts.skipDelay) await wait(WRITE_MS);
  requireLead(role);
  const idx = state.patches.findIndex((p) => p.id === id);
  const patch = state.patches[idx];
  if (!patch) throw new ApiError(404, '수정안을 찾을 수 없습니다.');
  if (patch.status === 'approved') throw new ApiError(409, '이미 승인된 수정안입니다.');
  if (patch.status === 'rejected_by_validator') throw new ApiError(409, '검증에서 탈락한 수정안은 승인할 수 없습니다.');
  if (patch.ops.length === 0) throw new ApiError(409, '그래프 변경이 없는 메모 수정안입니다. 사람이 내용을 채운 뒤 다시 검증하세요.');
  const cur = latest();
  const validated = liveValidation(patch, cur);
  const createdAt = nowIso();
  const version = await commitVersion(
    patch.ops,
    {
      patch_id: patch.id,
      revert_of: null,
      rehearsal_ids: validated.validation?.rehearsals_fixed ?? [],
      created_by: user,
      created_at: createdAt,
      note: patch.runbook_text,
    },
    [...cur.effective, patch.id],
  );
  const approved: Patch = { ...validated, status: 'approved', decided_by: user, decided_at: createdAt };
  state.patches[idx] = approved;
  return { patch: clone(approved), version };
}

export async function rejectPatch(id: string, reason: string, user: string, role: Role): Promise<Patch> {
  await wait(WRITE_MS);
  requireLead(role);
  const idx = state.patches.findIndex((p) => p.id === id);
  const patch = state.patches[idx];
  if (!patch) throw new ApiError(404, '수정안을 찾을 수 없습니다.');
  const note = `거절 사유: ${reason.trim() || '(사유 없음)'}`;
  const rejected: Patch = {
    ...patch,
    status: 'rejected',
    decided_by: user,
    decided_at: nowIso(),
    validation: patch.validation ? { ...patch.validation, notes: [...patch.validation.notes, note] } : patch.validation,
  };
  state.patches[idx] = rejected;
  return clone(rejected);
}

export async function revertVersion(_ws: string, n: number, user: string, role: Role): Promise<GraphVersion> {
  await wait(WRITE_MS);
  requireLead(role);
  const target = versionState(n);
  if (target.meta.number === 1) throw new ApiError(409, '최초 진단 버전은 되돌릴 수 없습니다.');
  const cur = latest();
  let effective: string[];
  if (target.meta.revert_of) {
    const original = versionState(target.meta.revert_of);
    effective = original.meta.patch_id ? [...cur.effective, original.meta.patch_id] : [...cur.effective];
  } else {
    effective = cur.effective.filter((p) => p !== target.meta.patch_id);
  }
  const patch = state.patches.find((p) => p.id === target.meta.patch_id);
  return commitVersion(
    invertOps(target.applied),
    {
      patch_id: null,
      revert_of: n,
      rehearsal_ids: [],
      created_by: user,
      created_at: nowIso(),
      note: `v${n} 되돌리기${patch ? ` — ${patch.runbook_text}` : ''}`,
    },
    effective,
  );
}

export async function signup(req: SignupRequest): Promise<{ ok: true }> {
  await wait(WRITE_MS);
  if (!req.consent_privacy) throw new ApiError(422, '개인정보 수집·이용 동의가 필요합니다.');
  state.signups.push(req);
  return { ok: true };
}

export async function trackEvent(req: EventRequest): Promise<void> {
  state.events.push(req);
}

export async function reportMarkdownText(ws: string): Promise<string> {
  const report = await getReport(ws);
  return reportMarkdown(report, state.versions.map((v) => v.meta), DOCS);
}

export async function reportHtmlText(ws: string): Promise<string> {
  return reportHtml(await reportMarkdownText(ws));
}

export async function reportSarifText(ws: string): Promise<string> {
  const report = await getReport(ws);
  return reportSarif(report, RULES, DOCS);
}

/** 스크린숏·리허설용 상태 미리 만들기 (?mock=after): 결제 두 문장 + 침해사고 문장 승인 */
export async function applyPreset(name: string): Promise<void> {
  if (name !== 'after' || state.versions.length > 1) return;
  for (const id of ['patch-dmc-01', 'patch-dmc-02', 'patch-dmc-06'])
    await approvePatch(id, '박재현 팀장', 'lead', { skipDelay: true });
}

export const MOCK_NOTE = OVERLAY.note;
