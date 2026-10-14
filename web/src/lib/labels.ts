import type {
  ActionType,
  ChannelMedium,
  EdgeKind,
  Element,
  ElementStatus,
  FindingStatus,
  IncidentType,
  LlmMode,
  NodeKind,
  PatchAuthor,
  PatchStatus,
  PerturbationKind,
  PersonStatus,
  Role,
  ScenarioOrigin,
  Severity,
  SimEventKind,
  SourceKind,
} from '../api/types';

/** 화면에 쓰는 한국어 라벨 모음. 색만으로 상태를 전달하지 않도록 항상 텍스트와 함께 쓴다. */

export const SEVERITY_LABEL: Record<Severity, string> = {
  critical: '치명',
  high: '높음',
  medium: '보통',
  low: '낮음',
  info: '정보',
};

export const SEVERITY_ORDER: Severity[] = ['critical', 'high', 'medium', 'low', 'info'];

export const SEVERITY_RANK: Record<Severity, number> = {
  critical: 0,
  high: 1,
  medium: 2,
  low: 3,
  info: 4,
};

export const FINDING_STATUS_LABEL: Record<FindingStatus, string> = {
  broken: '끊김',
  unknown: '확인 불가',
};

export const ELEMENT_LABEL: Record<Element, string> = {
  detection: '감지',
  owner: '담당',
  approver: '승인자',
  fallback_channel: '대체 채널',
  rollback: '롤백',
};

export const ELEMENT_ORDER: Element[] = ['detection', 'owner', 'approver', 'fallback_channel', 'rollback'];

export const ELEMENT_STATUS_LABEL: Record<ElementStatus, string> = {
  pass: '통과',
  partial: '부분',
  fail: '끊김',
  unknown: '확인 불가',
};

export const NODE_KIND_LABEL: Record<NodeKind, string> = {
  service: '서비스',
  system: '시스템',
  location: '위치',
  person: '사람',
  role: '역할',
  team: '팀',
  channel: '연락 채널',
  trigger: '감지 신호',
  action: '조치',
  obligation: '신고 의무',
};

export const EDGE_KIND_LABEL: Record<EdgeKind, string> = {
  depends_on: '의존',
  runs_in: '위치',
  owned_by: '담당',
  detected_by: '감지',
  notifies: '호출',
  escalates_to: '상향',
  mitigated_by: '조치',
  approved_by: '승인',
  executed_by: '실행',
  reachable_via: '연락',
  fallback_to: '대체',
  holds_role: '역할',
  member_of: '소속',
  must_report: '신고',
  reported_by: '신고 담당',
};

export const PERSON_STATUS_LABEL: Record<PersonStatus, string> = {
  active: '재직',
  departed: '퇴사',
  on_leave: '휴직',
  unknown: '명부 미확인',
};

export const MEDIUM_LABEL: Record<ChannelMedium, string> = {
  slack: '슬랙',
  kakaotalk: '카카오톡',
  teams: 'Teams',
  phone: '전화',
  sms: '문자',
  email: '이메일',
  pager: '호출 도구',
  internal_messenger: '사내 메신저',
  other: '기타',
};

export const ACTION_TYPE_LABEL: Record<ActionType, string> = {
  rollback: '롤백',
  failover: '페일오버',
  restart: '재시작',
  feature_flag: '기능 플래그',
  scale: '확장',
  other: '기타 조치',
};

export const INCIDENT_TYPE_LABEL: Record<IncidentType, string> = {
  outage: '서비스 장애',
  breach: '침해사고',
  privacy_leak: '개인정보 유출',
};

export const SOURCE_LABEL: Record<SourceKind, string> = {
  runbook: '런북',
  catalog: '카탈로그',
  roster: '명부',
  oncall: '당직표',
  policy: '정책',
  patch: '승인 수정안',
};

export const PERTURBATION_LABEL: Record<PerturbationKind, string> = {
  person_unavailable: '담당자 부재',
  channel_down: '채널 다운',
  system_down: '시스템 장애',
  location_down: '센터 장애',
  concurrent_incident: '동시 장애',
};

export const ORIGIN_LABEL: Record<ScenarioOrigin, string> = {
  template: '템플릿',
  llm: 'AI',
  human: '사람',
};

export const AUTHOR_LABEL: Record<PatchAuthor, string> = {
  ai: 'AI 초안',
  template: '템플릿',
  human: '사람 작성',
};

export const PATCH_STATUS_LABEL: Record<PatchStatus, string> = {
  draft: '초안',
  validated: '검증 통과',
  rejected_by_validator: '검증 탈락',
  approved: '승인됨',
  rejected: '거절됨',
  superseded: '기준 버전 변경',
};

export const LLM_MODE_LABEL: Record<LlmMode, string> = {
  off: 'LLM OFF',
  live: 'LLM LIVE',
  replay: 'LLM REPLAY',
};

export const ROLE_LABEL: Record<Role, string> = {
  viewer: '뷰어',
  editor: '편집자',
  lead: '팀 리드',
  admin: '관리자',
};

export const SIM_EVENT_LABEL: Record<SimEventKind, string> = {
  incident_start: '장애 발생',
  detected: '감지',
  undetected: '감지 실패',
  page: '호출',
  channel_unavailable: '채널 사용 불가',
  no_answer: '응답 없음',
  ack: '응답',
  escalate: '상향',
  dead_end: '막다른 길',
  approval_requested: '승인 요청',
  approved: '승인',
  approval_blocked: '승인 막힘',
  action_started: '조치 시작',
  mitigated: '완화',
  no_mitigation: '조치 불가',
  report_filed: '신고',
  report_missed: '신고 누락',
  deadline_missed: '시한 초과',
};

/** 실패를 뜻하는 리허설 이벤트 */
export const SIM_EVENT_BAD: ReadonlySet<SimEventKind> = new Set<SimEventKind>([
  'undetected',
  'channel_unavailable',
  'no_answer',
  'dead_end',
  'approval_blocked',
  'no_mitigation',
  'report_missed',
  'deadline_missed',
]);

export const SIM_EVENT_GOOD: ReadonlySet<SimEventKind> = new Set<SimEventKind>([
  'ack',
  'approved',
  'mitigated',
  'report_filed',
  'detected',
]);

/** 속성 키 → 한국어 */
export const ATTR_LABEL: Record<string, string> = {
  tier: '등급',
  slo_minutes: '목표 복구 시간',
  description: '설명',
  status: '상태',
  title: '직급·직책',
  aliases: '별칭',
  medium: '연락 수단',
  condition: '감지 조건',
  delay_minutes: '감지 지연',
  action_type: '조치 종류',
  requires_approval: '승인 필요',
  duration_minutes: '소요 시간',
  recipient: '신고 기관',
  deadline_minutes: '신고 기한',
  incident_type: '사고 유형',
  legal_basis: '법적 근거',
  location_kind: '위치 종류',
  after_minutes: '상향 대기',
  channel: '연락 채널',
};

export const INJECTION_PATTERN_LABEL: Record<string, string> = {
  'ignore-instructions': '이전 지시 무시 요구',
  'system-prompt': '시스템 프롬프트 언급·유출 요구',
  'role-reassignment': '역할 바꾸기 요구',
  'as-an-ai': 'AI 화법 흉내',
  'addressed-to-ai': 'AI를 향한 지시',
  'score-manipulation': '점수 조작 지시',
  'verdict-override': '판정 덮어쓰기 지시',
  'finding-suppression': '끊김 숨기기 지시',
  'graph-tamper': '그래프 변경 지시',
  'fake-role-tag': '가짜 역할 태그',
  'tool-call-payload': '도구 호출 위장',
  'hidden-content': '숨김 텍스트(HTML 주석 등)',
};

export const RULE_FAMILY_LABEL: Record<string, string> = {
  PATH: '필수 경로',
  PPL: '사람',
  ESC: '상향',
  CHN: '채널',
  DDL: '시한',
  KR: '국내 신고 기준',
  ONC: '당직',
  CAT: '카탈로그 대조',
  INT: '판단 보류·무결성',
  SPF: '단일 장애점',
};

export function ruleFamily(ruleId: string): string {
  const fam = ruleId.split('-')[1] ?? '';
  return RULE_FAMILY_LABEL[fam] ?? fam;
}
