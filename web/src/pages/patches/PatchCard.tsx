import { clsx } from 'clsx';
import {
  ArrowRight,
  BadgeCheck,
  Check,
  CircleAlert,
  CircleCheck,
  CircleX,
  FileCode,
  GitCommitVertical,
  Hand,
  Info,
  Sparkles,
  UserCheck,
  X,
  type LucideIcon,
} from 'lucide-react';
import type { ReactNode } from 'react';
import type { Finding, Patch, Scenario } from '../../api/types';
import { Badge, type Tone } from '../../components/ui/Badge';
import { Button } from '../../components/ui/Button';
import { DisabledReason } from '../../components/ui/Tooltip';
import { formatDateTime, formatScore } from '../../lib/format';
import { AUTHOR_LABEL, PATCH_STATUS_LABEL } from '../../lib/labels';
import { isMemoPatch, type OpSummary } from '../../lib/patchSummary';

const STATUS_TONE: Record<Patch['status'], Tone> = {
  draft: 'neutral',
  validated: 'ok',
  rejected_by_validator: 'danger',
  approved: 'brand',
  rejected: 'muted',
  superseded: 'warn',
};

const STATUS_ICON: Record<Patch['status'], LucideIcon> = {
  draft: Info,
  validated: BadgeCheck,
  rejected_by_validator: CircleX,
  approved: CircleCheck,
  rejected: X,
  superseded: CircleAlert,
};

export interface PatchCardProps {
  patch: Patch;
  summary: OpSummary[];
  serviceName: string | null;
  findings: readonly Finding[];
  scenarios: readonly Scenario[];
  canApprove: boolean;
  /** 승인·거절이 꺼진 이유(공개 체험판 등). 없으면 역할 기준 문구('팀 리드만 …')를 쓴다 */
  lockedReason?: string | null;
  highlighted: boolean;
  approving: boolean;
  approvedVersion?: number | null;
  onApprove: () => void;
  onReject: () => void;
}

export function PatchCard({
  patch,
  summary,
  serviceName,
  findings,
  scenarios,
  canApprove,
  lockedReason = null,
  highlighted,
  approving,
  approvedVersion,
  onApprove,
  onReject,
}: PatchCardProps) {
  const memo = isMemoPatch(patch);
  const v = patch.validation;
  const decided = patch.status === 'approved' || patch.status === 'rejected';
  const blocked = patch.status === 'rejected_by_validator' || patch.status === 'superseded';
  const resolved = (v?.resolved_finding_ids ?? []).map((id) => findings.find((f) => f.id === id)).filter((f): f is Finding => !!f);
  // 같은 제목의 끊김은 한 번만 쓰고 개수를 붙인다 ("퇴사자에게 대응이 걸려 있음 ×2")
  const resolvedTitles = [...resolved.reduce((m, f) => m.set(f.title, (m.get(f.title) ?? 0) + 1), new Map<string, number>())].map(([t, n]) =>
    n > 1 ? `${t} ×${n}` : t,
  );
  const scenarioTitle = (id: string) => scenarios.find((s) => s.id === id)?.title ?? id;
  const regressions = (v?.rehearsals_regressed.length ?? 0) + (v?.new_finding_ids.length ?? 0);
  const AuthorIcon = patch.author === 'ai' ? Sparkles : patch.author === 'template' ? FileCode : UserCheck;

  return (
    <article
      id={`patch-${patch.id}`}
      aria-label={`수정안: ${patch.runbook_text}`}
      className={clsx(
        'card scroll-mt-24 overflow-hidden transition-shadow',
        highlighted && 'ring-[3px] ring-brand ring-offset-2 ring-offset-bg',
        decided && 'opacity-90',
      )}
    >
      <header className="flex flex-wrap items-center gap-2 border-b border-line px-5 py-3">
        {memo ? (
          <Badge tone="warn" icon={Hand} size="lg">
            사람 확인 필요
          </Badge>
        ) : (
          <Badge tone={STATUS_TONE[patch.status]} icon={STATUS_ICON[patch.status]} size="lg">
            {PATCH_STATUS_LABEL[patch.status]}
          </Badge>
        )}
        <Badge tone={patch.author === 'ai' ? 'brand' : 'neutral'} icon={AuthorIcon} size="lg">
          {AUTHOR_LABEL[patch.author]}
        </Badge>
        {serviceName ? <span className="text-[14px] font-semibold text-fg-muted">{serviceName}</span> : null}
        <span className="ml-auto text-[13px] text-fg-subtle">
          기준 v{patch.base_version}
          {patch.created_at ? ` · ${formatDateTime(patch.created_at)}` : ''}
          {patch.author === 'ai' && patch.llm_usage.calls ? ` · LLM ${patch.llm_usage.calls}회(문장만)` : ''}
        </span>
      </header>

      <div className="grid gap-5 p-5 xl:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
        <div className="min-w-0">
          <blockquote className="border-l-4 border-brand pl-4">
            <p className="text-[24px] leading-snug font-extrabold tracking-tight 2xl:text-[26px]">“{patch.runbook_text}”</p>
            <footer className="mt-1.5 text-[13px] text-fg-subtle">
              {memo ? '사람이 빈칸을 채워야 하는 초안 · 담당 지정·명부·당직표는 조직이 정하고, 제때옵스는 확정하지 않습니다' : '런북에 추가할 문장 · 사람이 읽고 승인하는 대상'}
            </footer>
          </blockquote>
          <p className="mt-4 text-[15.5px] leading-relaxed text-fg-muted">{patch.rationale}</p>

          <h4 className="mt-5 mb-2 text-[13px] font-bold tracking-wide text-fg-subtle">그래프 변경</h4>
          {memo ? (
            <div className="flex gap-2 rounded-xl border border-dashed border-warn/50 bg-warn/8 px-4 py-3 text-[14.5px]">
              <Hand className="mt-0.5 size-4 shrink-0 text-warn" aria-hidden />
              <span>
                그래프를 바꾸지 않는 <strong>메모 수정안</strong>입니다. 담당자 지정·명부·당직표처럼 조직이 정할 일이라 제때옵스가 확정하지 않습니다. 사람이 빈칸을 채운 뒤 다시 검증합니다.
              </span>
            </div>
          ) : (
            <ul className="space-y-1.5 font-medium">
              {summary.map((s, i) => (
                <li
                  key={i}
                  className={clsx(
                    'flex items-start gap-2 rounded-lg px-3 py-1.5 text-[15px]',
                    s.sign === '+' && 'bg-ok/8',
                    s.sign === '−' && 'bg-danger/8',
                    s.sign === '~' && 'bg-warn/8',
                  )}
                >
                  <span
                    aria-label={s.sign === '+' ? '추가' : s.sign === '−' ? '삭제' : '변경'}
                    className={clsx(
                      'tnum w-4 shrink-0 text-center font-mono font-bold',
                      s.sign === '+' && 'text-ok',
                      s.sign === '−' && 'text-danger',
                      s.sign === '~' && 'text-warn',
                    )}
                  >
                    {s.sign}
                  </span>
                  <span>
                    <span className="font-bold">{s.label}</span>
                    <span className="text-fg-subtle">: </span>
                    <span className={clsx(s.sign === '−' && 'line-through decoration-danger/60')}>{s.detail}</span>
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="min-w-0">
          <h4 className="mb-2 text-[13px] font-bold tracking-wide text-fg-subtle">검증 결과 (고정 규칙 재검사 · 과거 리허설 재실행)</h4>
          {v ? (
            <ul className="space-y-1.5 rounded-xl border border-line bg-surface-2 p-3">
              <CheckRow ok={resolved.length > 0 || v.resolved_finding_ids.length > 0} label={`해결되는 끊김 ${v.resolved_finding_ids.length}건`}>
                {resolvedTitles.length ? resolvedTitles.join(' · ') : null}
              </CheckRow>
              <CheckRow ok={v.rehearsals_fixed.length > 0} neutral={v.rehearsals_fixed.length === 0} label={`리허설 FAIL → PASS ${v.rehearsals_fixed.length}건`}>
                {v.rehearsals_fixed.length ? v.rehearsals_fixed.map(scenarioTitle).join(' · ') : '이 수정안만으로 통과로 바뀌는 리허설은 없습니다'}
              </CheckRow>
              <CheckRow ok={regressions === 0} label={`회귀 ${regressions}건`}>
                {v.rehearsals_regressed.length ? `통과 → 실패: ${v.rehearsals_regressed.map(scenarioTitle).join(' · ')}` : '과거에 통과하던 리허설과 판정이 그대로 유지됩니다'}
              </CheckRow>
              <CheckRow ok={v.unknown_people.length === 0} label={`명부 외 인물 ${v.unknown_people.length}건`}>
                {v.unknown_people.length ? v.unknown_people.join(', ') : '수정안의 사람은 모두 명부의 재직자입니다'}
              </CheckRow>
              {v.integrity_errors.length ? <CheckRow ok={false} label={`무결성 오류 ${v.integrity_errors.length}건`}>{v.integrity_errors.join(' · ')}</CheckRow> : null}
              <li className="mt-1 flex items-center gap-2 border-t border-line pt-2.5">
                {/* PatchValidation.score_before/after are the overall readiness (check.scoring.score) */}
                <span className="text-[14px] text-fg-muted">전체 준비도</span>
                {v.score_before !== null && v.score_after !== null && v.score_before === v.score_after ? (
                  <span className="tnum ml-auto inline-flex items-baseline gap-2 text-[22px] font-extrabold">
                    {formatScore(v.score_after)}
                    <span className="text-[14px] font-semibold text-fg-subtle">변화 없음</span>
                  </span>
                ) : v.score_before !== null && v.score_after !== null ? (
                  <span className="tnum ml-auto inline-flex items-center gap-2 text-[22px] font-extrabold">
                    <span className="text-fg-muted">{formatScore(v.score_before)}</span>
                    <ArrowRight className="size-5 text-fg-subtle" aria-hidden />
                    <span className={v.score_after > v.score_before ? 'text-ok' : 'text-fg'}>{formatScore(v.score_after)}</span>
                  </span>
                ) : (
                  <span className="ml-auto text-[14px] text-fg-subtle">점수 영향 없음 · 5요소 밖 항목</span>
                )}
              </li>
            </ul>
          ) : (
            <p className="rounded-xl border border-line bg-surface-2 p-3 text-[14px] text-fg-muted">검증 전입니다. 그래프 변경이 없는 메모는 검증 대상이 아닙니다.</p>
          )}
          {v?.notes.length ? (
            <ul className="mt-3 space-y-1.5">
              {v.notes.map((n) => (
                <li key={n} className="flex gap-2 rounded-lg bg-info/8 px-3 py-2 text-[14px] leading-relaxed">
                  <Info className="mt-0.5 size-4 shrink-0 text-info" aria-hidden />
                  {n}
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      </div>

      <footer className="flex flex-wrap items-center gap-3 border-t border-line bg-surface-2/50 px-5 py-3">
        {patch.status === 'approved' ? (
          <p className="inline-flex items-center gap-2 text-[15px] font-semibold text-brand">
            <GitCommitVertical className="size-5" aria-hidden />
            {approvedVersion ? `v${approvedVersion}로 반영` : '그래프에 반영'} · {patch.decided_by ?? '팀 리드'} 승인 · {formatDateTime(patch.decided_at)}
          </p>
        ) : patch.status === 'rejected' ? (
          <p className="text-[15px] text-fg-muted">
            {patch.decided_by ?? '팀 리드'}가 거절함 · {formatDateTime(patch.decided_at)}
          </p>
        ) : (
          <>
            <ApproveButton
              canApprove={canApprove}
              lockedReason={lockedReason}
              disabledReason={memo ? '그래프 변경이 없는 메모는 승인 대상이 아닙니다' : blocked ? '검증을 통과하지 못한 수정안은 승인할 수 없습니다' : null}
              loading={approving}
              onClick={onApprove}
            />
            <RejectButton canApprove={canApprove} lockedReason={lockedReason} onClick={onReject} />
            <span className="text-[13px] text-fg-subtle">
              {memo
                ? '담당 조직이 빈칸을 채워 다시 올리면 검증 후 승인할 수 있습니다.'
                : blocked
                  ? '검증을 통과하지 못해 승인할 수 없습니다. 거절하거나 초안을 다시 검증하세요.'
                  : '승인하면 새 그래프 버전이 생기고, 되돌리기도 새 버전으로 남습니다.'}
            </span>
          </>
        )}
      </footer>
    </article>
  );
}

function CheckRow({ ok, neutral = false, label, children }: { ok: boolean; neutral?: boolean; label: string; children?: ReactNode }) {
  const Icon = neutral ? Info : ok ? Check : X;
  return (
    <li className="flex gap-2.5">
      <span
        className={clsx(
          'mt-0.5 grid size-5 shrink-0 place-items-center rounded-full',
          neutral ? 'bg-surface-3 text-fg-subtle' : ok ? 'bg-ok text-[#04150d] projector:text-white' : 'bg-danger text-white',
        )}
      >
        <Icon className="size-3.5" strokeWidth={3} aria-label={neutral ? '해당 없음' : ok ? '통과' : '실패'} />
      </span>
      <div className="min-w-0">
        <p className="text-[15px] font-bold">{label}</p>
        {children ? <p className="text-[13px] leading-snug text-fg-muted">{children}</p> : null}
      </div>
    </li>
  );
}

function ApproveButton({
  canApprove,
  lockedReason,
  disabledReason,
  loading,
  onClick,
}: {
  canApprove: boolean;
  lockedReason: string | null;
  disabledReason: string | null;
  loading: boolean;
  onClick: () => void;
}) {
  const reason = !canApprove ? (lockedReason ?? '팀 리드만 승인할 수 있습니다') : disabledReason;
  return (
    <DisabledReason reason={reason}>
      <Button variant="success" size="lg" icon={<Check className="size-5" aria-hidden />} disabled={!!reason} loading={loading} onClick={onClick}>
        승인
      </Button>
    </DisabledReason>
  );
}

function RejectButton({ canApprove, lockedReason, onClick }: { canApprove: boolean; lockedReason: string | null; onClick: () => void }) {
  return (
    <DisabledReason reason={canApprove ? null : (lockedReason ?? '팀 리드만 거절할 수 있습니다')}>
      <Button variant="danger" size="lg" icon={<X className="size-5" aria-hidden />} disabled={!canApprove} onClick={onClick}>
        거절
      </Button>
    </DisabledReason>
  );
}
