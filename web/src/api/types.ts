/**
 * 제때옵스 도메인 JSON 타입 — `src/jettae/domain/*.py`(pydantic, mode="json")와 1:1로 맞춘다.
 * 백엔드 계약이 바뀌면 이 파일부터 고친다. 화면 전용 타입은 아래쪽 "View DTO" 구역에 둔다.
 */

// --------------------------------------------------------------------------- vocab (graph.py)

export type NodeKind =
  | 'service'
  | 'system'
  | 'location'
  | 'person'
  | 'role'
  | 'team'
  | 'channel'
  | 'trigger'
  | 'action'
  | 'obligation';

export type EdgeKind =
  | 'depends_on'
  | 'runs_in'
  | 'owned_by'
  | 'detected_by'
  | 'notifies'
  | 'escalates_to'
  | 'mitigated_by'
  | 'approved_by'
  | 'executed_by'
  | 'reachable_via'
  | 'fallback_to'
  | 'holds_role'
  | 'member_of'
  | 'must_report'
  | 'reported_by';

export type PersonStatus = 'active' | 'departed' | 'on_leave' | 'unknown';

export type ChannelMedium =
  | 'slack'
  | 'kakaotalk'
  | 'teams'
  | 'phone'
  | 'sms'
  | 'email'
  | 'pager'
  | 'internal_messenger'
  | 'other';

export type ActionType = 'rollback' | 'failover' | 'restart' | 'feature_flag' | 'scale' | 'other';

export type LocationKind = 'datacenter' | 'region' | 'az' | 'saas';

export type IncidentType = 'outage' | 'breach' | 'privacy_leak';

export type SourceKind = 'runbook' | 'catalog' | 'roster' | 'oncall' | 'policy' | 'patch';

export type Extractor = 'rule' | 'llm' | 'import' | 'human';

export type Verification = 'verified' | 'hold';

/** 잘 알려진 속성 키 (AttrKey). 알 수 없는 키도 허용된다. */
export type AttrKey =
  | 'tier'
  | 'slo_minutes'
  | 'description'
  | 'status'
  | 'title'
  | 'aliases'
  | 'medium'
  | 'condition'
  | 'delay_minutes'
  | 'action_type'
  | 'requires_approval'
  | 'duration_minutes'
  | 'recipient'
  | 'deadline_minutes'
  | 'incident_type'
  | 'legal_basis'
  | 'location_kind'
  | 'after_minutes'
  | 'channel';

// --------------------------------------------------------------------------- provenance

export interface SourceSpan {
  doc_id: string;
  start: number;
  end: number;
  quote: string;
  line: number | null;
}

export interface Provenance {
  source: SourceKind;
  extractor: Extractor;
  spans: SourceSpan[];
  ref: string | null;
}

export type JsonScalar = string | number | boolean;
export type AttrValue = JsonScalar | string[];

export interface Attr {
  value: AttrValue;
  spans: SourceSpan[];
  source: SourceKind;
}

// --------------------------------------------------------------------------- elements

export interface Node {
  id: string;
  kind: NodeKind;
  name: string;
  attrs: Record<string, Attr>;
  prov: Provenance[];
  verification: Verification;
  hold_reason: string | null;
}

export interface Edge {
  id: string;
  kind: EdgeKind;
  src: string;
  dst: string;
  attrs: Record<string, Attr>;
  prov: Provenance[];
  verification: Verification;
  hold_reason: string | null;
}

export interface DocumentRef {
  doc_id: string;
  title: string;
  sha256: string;
  path: string | null;
}

/** 불변 운영 그래프 (OpsGraph.model_dump(mode="json")). */
export interface OpsGraph {
  nodes: Record<string, Node>;
  edges: Record<string, Edge>;
  docs: Record<string, DocumentRef>;
}

// --------------------------------------------------------------------------- findings.py

export type Severity = 'critical' | 'high' | 'medium' | 'low' | 'info';
export type FindingStatus = 'broken' | 'unknown';
export type Element = 'detection' | 'owner' | 'approver' | 'fallback_channel' | 'rollback';

export type FixKind =
  | 'assign_owner'
  | 'replace_departed'
  | 'add_escalation'
  | 'break_escalation_cycle'
  | 'assign_approver'
  | 'add_deputy'
  | 'add_fallback_channel'
  | 'add_rollback'
  | 'add_detection'
  | 'fix_deadline'
  | 'add_reporting_step'
  | 'adjust_oncall'
  | 'add_service_runbook'
  | 'confirm_identity'
  | 'assign_executor'
  | 'add_failover'
  | 'review_element';

export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };

export interface FixHint {
  kind: FixKind;
  params: Record<string, JsonValue>;
}

export interface Finding {
  id: string;
  rule_id: string;
  severity: Severity;
  status: FindingStatus;
  title: string;
  message: string;
  service_id: string | null;
  element: Element | null;
  subject_ids: string[];
  evidence: SourceSpan[];
  fix: FixHint | null;
  legal_basis: string | null;
}

export interface RuleInfo {
  rule_id: string;
  title: string;
  severity: Severity;
  element: Element | null;
  rationale: string;
  legal_basis: string | null;
}

export type ElementStatus = 'pass' | 'partial' | 'fail' | 'unknown';

export interface ElementScore {
  element: Element;
  score: number;
  status: ElementStatus;
  reasons: string[];
  finding_ids: string[];
}

export interface ServiceReadiness {
  service_id: string;
  service_name: string;
  tier: number;
  weight: number;
  score: number;
  elements: ElementScore[];
}

export interface ReadinessReport {
  overall: number;
  services: ServiceReadiness[];
  weights_version: string;
  element_weights: Record<string, number>;
  /** JSON 직렬화 시 int 키는 문자열이 된다 ("1", "2", "3"). */
  tier_weights: Record<string, number>;
  graph_hash: string;
}

// --------------------------------------------------------------------------- rehearsal.py

export type PerturbationKind =
  | 'person_unavailable'
  | 'channel_down'
  | 'system_down'
  | 'location_down'
  | 'concurrent_incident';

export interface Perturbation {
  kind: PerturbationKind;
  target_id: string;
  start_minute: number;
  end_minute: number | null;
}

export type ScenarioOrigin = 'template' | 'llm' | 'human';

export interface Scenario {
  id: string;
  title: string;
  description: string;
  service_id: string;
  incident_type: IncidentType;
  perturbations: Perturbation[];
  deadline_minutes: number | null;
  origin: ScenarioOrigin;
}

export interface SimConfig {
  detection_delay_minutes: number;
  undetected_discovery_minutes: number | null;
  ack_minutes: number;
  default_escalation_after_minutes: number | null;
  approval_minutes: number;
  default_action_minutes: number;
  max_hops: number;
}

export type SimEventKind =
  | 'incident_start'
  | 'detected'
  | 'undetected'
  | 'page'
  | 'channel_unavailable'
  | 'no_answer'
  | 'ack'
  | 'escalate'
  | 'dead_end'
  | 'approval_requested'
  | 'approved'
  | 'approval_blocked'
  | 'action_started'
  | 'mitigated'
  | 'no_mitigation'
  | 'report_filed'
  | 'report_missed'
  | 'deadline_missed';

export interface SimEvent {
  t: number;
  kind: SimEventKind;
  actor_id: string | null;
  target_id: string | null;
  edge_id: string | null;
  note: string;
}

export type Outcome = 'pass' | 'fail';

export interface RehearsalResult {
  scenario_id: string;
  scenario_fingerprint: string;
  graph_hash: string;
  config_hash: string;
  outcome: Outcome;
  time_to_mitigate: number | null;
  deadline_minutes: number;
  failed_element: Element | null;
  failed_at_id: string | null;
  reason: string;
  events: SimEvent[];
  path_ids: string[];
}

// --------------------------------------------------------------------------- patch.py

export interface AddNodeOp {
  op: 'add_node';
  node: Node;
}

export interface AddEdgeOp {
  op: 'add_edge';
  edge: Edge;
}

export interface RemoveEdgeOp {
  op: 'remove_edge';
  edge_id: string;
  snapshot: Edge | null;
}

export interface RemoveNodeOp {
  op: 'remove_node';
  node_id: string;
  snapshot: Node | null;
}

export interface SetAttrOp {
  op: 'set_attr';
  target_id: string;
  key: string;
  value: Attr | null;
  previous: Attr | null;
}

export type PatchOp = AddNodeOp | AddEdgeOp | RemoveEdgeOp | RemoveNodeOp | SetAttrOp;

export type PatchAuthor = 'ai' | 'template' | 'human';

export type PatchStatus =
  | 'draft'
  | 'validated'
  | 'rejected_by_validator'
  | 'approved'
  | 'rejected'
  | 'superseded';

export interface PatchValidation {
  passed: boolean;
  resolved_finding_ids: string[];
  unresolved_finding_ids: string[];
  new_finding_ids: string[];
  rehearsals_fixed: string[];
  rehearsals_regressed: string[];
  unknown_people: string[];
  integrity_errors: string[];
  score_before: number | null;
  score_after: number | null;
  notes: string[];
}

export interface Patch {
  id: string;
  workspace_id: string;
  base_version: number;
  finding_ids: string[];
  ops: PatchOp[];
  runbook_text: string;
  rationale: string;
  author: PatchAuthor;
  status: PatchStatus;
  validation: PatchValidation | null;
  decided_by: string | null;
  decided_at: string | null;
  created_at: string | null;
  llm_usage: Record<string, number>;
}

export interface GraphVersion {
  workspace_id: string;
  number: number;
  parent: number | null;
  graph_hash: string;
  patch_id: string | null;
  revert_of: number | null;
  rehearsal_ids: string[];
  created_by: string;
  created_at: string;
  note: string;
}

// --------------------------------------------------------------------------- extraction.py / report.py

export interface GroundingStatus {
  grounded: number;
  fuzzy_grounded: number;
  dropped_ungrounded: number;
  dropped_quarantined: number;
  dropped_invalid: number;
  held_backtranslation: number;
  held_ambiguous_entity: number;
  dropped_examples: string[];
}

export type LlmMode = 'off' | 'live' | 'replay';

export interface LlmUsageSummary {
  mode: LlmMode;
  provider: string | null;
  model: string | null;
  calls: number;
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  by_task: Record<string, number>;
  judgement_path_calls: number;
}

export interface SecurityNotice {
  doc_id: string;
  start: number;
  end: number;
  excerpt: string;
  pattern: string;
}

export interface DiagnosisReport {
  run_id: string;
  workspace_id: string;
  graph_version: number;
  graph_hash: string;
  created_at: string;
  findings: Finding[];
  readiness: ReadinessReport;
  scenarios: Scenario[];
  rehearsals: RehearsalResult[];
  patches: Patch[];
  grounding: GroundingStatus;
  security: SecurityNotice[];
  llm: LlmUsageSummary;
  held_element_ids: string[];
  catalog_gaps: string[];
  timings_ms: Record<string, number>;
  /** 규칙 밖 위험 후보 — 점수·끊김 판정·통과 기준에 반영하지 않음 (탐색 전 보고서는 빈 목록) */
  risk_hypotheses: RiskHypothesis[];
  /** 탐색 범위·제외·상한·AI 사용 (탐색을 돌리지 않았으면 null) */
  exploration: ExplorationSummary | null;
}

// --------------------------------------------------------------------------- explore.py (규칙 밖 위험 탐색)

export type HypothesisOrigin = 'sweep' | 'ai';

/** confirmed = 확인됨(리허설로 재현), needs_info = 확인 필요, refuted = 기각 */
export type HypothesisStatus = 'confirmed' | 'refuted' | 'needs_info';

export type AiProposalStatus = 'used' | 'off' | 'failed' | 'fallback' | 'not_requested' | 'rejudge';

export interface HypothesisVerification {
  scenario_id: string;
  scenario_fingerprint: string;
  graph_hash: string;
  /** 서비스 정책 목표 복구 시간 (가설이 정할 수 없음) */
  deadline_minutes: number;
  outcome: Outcome;
  baseline_outcome: Outcome;
  time_to_mitigate: number | null;
  failed_element: Element | null;
  failed_at_id: string | null;
  reason: string;
  result_hash: string;
  path_ids: string[];
}

/** AI 가설 교란 중 일부만으로 다시 리허설한 결과 (같은 서비스·같은 목표 시간) */
export interface FailingSubset {
  perturbations: Perturbation[];
  failed_element: Element | null;
  failed_at_id: string | null;
  reason: string;
  result_hash: string;
  /** 제안한 조합 전체와 끊긴 단계·지점이 다름 (숨기지 않고 표시) */
  point_differs: boolean;
}

/**
 * 최소 실패 조건 자동 검증 — 확인됨 AI 가설(전제 없음, 교란 2개 이상)에서 필요 없는 조건만 덜어 낸 결과.
 * 제안에 없는 다른 원인은 찾지 않으며, 상태·점수에 반영하지 않음.
 */
export interface HypothesisMinimization {
  replays: number;
  /** 같은 지점에서 끊기는 가장 작은 조합들 (부분 조합이 없으면 제안한 조합 그대로) */
  minimal_sets: FailingSubset[];
  original_minimal: boolean;
  /** 어느 최소 조합에도 들지 않는 조건 */
  unnecessary: Perturbation[];
  /** 다른 지점에서 끊기는 부분 조합 — 같은 위험의 재현으로 보지 않음 */
  other_failures: FailingSubset[];
}

export interface RiskHypothesis {
  id: string;
  origin: HypothesisOrigin;
  service_id: string;
  incident_type: IncidentType;
  title: string;
  explanation: string;
  perturbations: Perturbation[];
  /** 그래프에 없는 전제 — 하나라도 있으면 확인됨이 될 수 없음 */
  assumptions: string[];
  evidence: SourceSpan[];
  status: HypothesisStatus;
  verification: HypothesisVerification;
  required_info: string[];
  /** 최소 실패 조건 자동 검증 (해당 없거나 이전 보고서면 null) */
  minimization: HypothesisMinimization | null;
}

export interface CoveredCut {
  service_id: string;
  target_ids: string[];
  finding_ids: string[];
}

export interface ServiceSweep {
  service_id: string;
  baseline: Outcome | null;
  candidates: number;
  candidates_total: number;
  replays: number;
  breaking: number;
  exhausted: boolean;
  skipped_reason: string | null;
}

export interface ExplorationSummary {
  method: string;
  max_set_size: number;
  services: ServiceSweep[];
  replays: number;
  covered: CoveredCut[];
  ai_status: AiProposalStatus;
  ai_used: boolean;
  ai_proposed: number;
  ai_kept: number;
  ai_dropped: number;
  ai_duplicates: number;
  ai_over_cap: number;
  ai_carried: number;
  notes: string[];
}

// --------------------------------------------------------------------------- inputs.py (문서)

/** 정규화된 문서 (Document). `text`는 PII가 길이 보존 마스킹된 본문. */
export interface SourceDocument {
  doc_id: string;
  title: string;
  text: string;
  media_type?: string;
  path: string | null;
  quarantined: [number, number][];
}

// --------------------------------------------------------------------------- View DTO (API 응답 형태)

/** GraphView 노드: Node에서 prov 대신 펼친 spans를 준다. */
export interface GraphViewNode {
  id: string;
  kind: NodeKind;
  name: string;
  attrs: Record<string, Attr>;
  verification: Verification;
  hold_reason: string | null;
  spans: SourceSpan[];
  /** 출처 종류 (화면 표시용, 선택) */
  sources?: SourceKind[];
}

export interface GraphViewEdge {
  id: string;
  kind: EdgeKind;
  src: string;
  dst: string;
  attrs: Record<string, Attr>;
  verification: Verification;
  hold_reason: string | null;
  spans: SourceSpan[];
  sources?: SourceKind[];
}

/** GET /api/v1/workspaces/{ws}/graph 응답 (GraphView DTO). */
export interface GraphView {
  version: number;
  graph_hash: string;
  nodes: GraphViewNode[];
  edges: GraphViewEdge[];
  findings_by_element: Record<string, string[]>;
  docs?: DocumentRef[];
}

/** 워크스페이스 목록 항목 (GET /api/v1/workspaces — §6에 없는 보조 엔드포인트, 필드는 모두 선택). */
export interface WorkspaceSummary {
  id: string;
  name: string;
  created_at?: string | null;
  latest_version?: number | null;
  graph_hash?: string | null;
  overall?: number | null;
  finding_count?: number | null;
  is_demo?: boolean;
  /** 이 브라우저에 접근 키가 저장된 무료 검진(클라이언트 전용 표시) */
  is_free_scan?: boolean;
}

/** 그래프 버전 비교 (GET /versions/diff?a=&b=). */
export interface AttrChange {
  key: string;
  before: Attr | null;
  after: Attr | null;
}

export interface ElementChange<T> {
  id: string;
  before: T;
  after: T;
  attrs: AttrChange[];
}

export interface GraphDiff {
  a: number;
  b: number;
  added_nodes: Node[];
  removed_nodes: Node[];
  changed_nodes: ElementChange<Node>[];
  added_edges: Edge[];
  removed_edges: Edge[];
  changed_edges: ElementChange<Edge>[];
}

export type Role = 'viewer' | 'editor' | 'lead' | 'admin';

/** GET /me — 서버가 실제로 인정하는 사용자·역할. 역할 전환기는 개발 모드(can_switch_role)에서만 의미가 있다. */
export interface SessionInfo {
  user: string;
  role: Role;
  anonymous: boolean;
  auth_mode: 'dev' | 'token' | 'public';
  can_switch_role: boolean;
}

export interface SignupRequest {
  email: string;
  org_name: string;
  team_size: string;
  message: string;
  consent_privacy: true;
  consent_marketing: boolean;
  source: string;
}

/**
 * 수요 측정 이벤트가 보낼 수 있는 화면 경로 — 서버가 받는 목록과 같다.
 * 워크스페이스 id·검색어 같은 자유 텍스트가 기록에 남지 않도록 실제 주소 대신 이 중 하나를 고른다.
 */
export const EVENT_PATHS = ['/', '/app', '/app/report', '/privacy'] as const;
export type EventPath = (typeof EVENT_PATHS)[number];

export interface EventRequest {
  name: 'landing_view' | 'landing_first_view' | 'upload_click' | 'signup_click' | 'demo_start' | 'report_download' | (string & {});
  path: EventPath;
  props?: Record<string, string | number | boolean>;
}

export interface DiagnoseRequest {
  llm_mode: LlmMode;
}

export interface RehearsalRequest {
  scenario_ids?: string[];
  generate?: boolean;
}

export interface DraftRequest {
  finding_ids?: string[];
  llm_mode: LlmMode;
}

/** 무료 검진 (POST /free-scans) — 접근 키는 이 응답에만 평문으로 온다. */
export interface FreeScanResult {
  workspace_id: string;
  access_key: string;
  readiness_overall: number;
  findings: number;
}

export interface InputFiles {
  runbooks: File[];
  roster?: File | null;
  oncall?: File | null;
  catalog?: File | null;
  policy?: File | null;
}
