import { CircleCheck, GitPullRequestArrow, ListChecks, RefreshCw } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { useParams, useSearchParams } from 'react-router';
import { errorMessage } from '../api/errors';
import { useApprovePatch, useDraftPatches, useGraph, usePatches, useRejectPatch, useReport, useVersions } from '../api/hooks';
import type { Finding, GraphView, Node, Patch } from '../api/types';
import { usePermissions } from '../app/permissions';
import { Button } from '../components/ui/Button';
import { Dialog } from '../components/ui/Dialog';
import { Segmented } from '../components/ui/Segmented';
import { EmptyState, ErrorState, LoadingBlock } from '../components/ui/States';
import { useToast } from '../components/ui/Toast';
import { DisabledReason } from '../components/ui/Tooltip';
import { formatScore, toServiceId } from '../lib/format';
import { isPendingPatch, summarizePatch } from '../lib/patchSummary';
import { PageHeader } from './PageHeader';
import { PatchCard } from './patches/PatchCard';

type Filter = 'pending' | 'done' | 'all';

/** 수정안이 다루는 서비스: 끊김 → op의 서비스 노드 → 조치를 가진 서비스 순으로 찾는다 */
export function patchServiceId(patch: Patch, findings: readonly Finding[], view: GraphView | undefined): string | null {
  for (const fid of patch.finding_ids) {
    const f = findings.find((x) => x.id === fid);
    if (f?.service_id) return f.service_id;
  }
  const ids: string[] = [];
  for (const op of patch.ops) {
    if (op.op === 'add_edge') ids.push(op.edge.src, op.edge.dst);
    else if (op.op === 'remove_edge' && op.snapshot) ids.push(op.snapshot.src, op.snapshot.dst);
    else if (op.op === 'set_attr') ids.push(op.target_id);
  }
  const svc = ids.find((id) => id.startsWith('service:'));
  if (svc) return svc;
  const action = ids.find((id) => id.startsWith('action:'));
  if (action && view) return view.edges.find((e) => e.kind === 'mitigated_by' && e.dst === action)?.src ?? null;
  return null;
}

export function Patches() {
  const { ws = '' } = useParams();
  const [params, setParams] = useSearchParams();
  const patches = usePatches(ws);
  const report = useReport(ws);
  const base = useReport(ws, 1);
  const graph = useGraph(ws);
  const versions = useVersions(ws);
  const perm = usePermissions();
  const approve = useApprovePatch(ws);
  const reject = useRejectPatch(ws);
  const draft = useDraftPatches(ws);
  const toast = useToast();
  const [rejecting, setRejecting] = useState<Patch | null>(null);
  const [reason, setReason] = useState('');
  const [approvingId, setApprovingId] = useState<string | null>(null);

  const filter: Filter = (['pending', 'done', 'all'] as Filter[]).find((f) => f === params.get('show')) ?? 'pending';
  const focusPatch = params.get('patch');
  const focusFinding = params.get('finding');
  const serviceFilter = params.get('service') ? toServiceId(params.get('service')!) : null;

  const allFindings = useMemo(() => {
    const m = new Map<string, Finding>();
    for (const f of [...(base.data?.findings ?? []), ...(report.data?.findings ?? [])]) m.set(f.id, f);
    return [...m.values()];
  }, [base.data, report.data]);

  const lookup = useMemo(() => {
    const nodes = new Map<string, Node>();
    for (const n of graph.data?.nodes ?? []) nodes.set(n.id, { ...n, prov: [] });
    return (id: string) => nodes.get(id);
  }, [graph.data]);

  useEffect(() => {
    const target = focusPatch ?? (focusFinding ? patches.data?.find((p) => p.finding_ids.includes(focusFinding))?.id : null);
    if (!target || !patches.data) return;
    const el = document.getElementById(`patch-${target}`);
    el?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, [focusPatch, focusFinding, patches.data]);

  if (patches.isError) return <ErrorState error={patches.error} onRetry={() => void patches.refetch()} />;
  if (!patches.data || !report.data) return <LoadingBlock className="p-8" />;

  const list = patches.data;
  const services = report.data.readiness.services;
  const svcName = (sid: string | null) => (sid ? (services.find((s) => s.service_id === sid)?.service_name ?? sid.replace(/^service:/, '')) : null);
  const visible = list.filter((p) => {
    if (filter === 'pending' && !isPendingPatch(p)) return false;
    if (filter === 'done' && isPendingPatch(p)) return false;
    if (serviceFilter && patchServiceId(p, allFindings, graph.data) !== serviceFilter) return false;
    return true;
  });

  // 서비스별 묶음 (서비스 없는 항목은 '전사 공통')
  const groups = new Map<string, Patch[]>();
  for (const p of visible) {
    const key = patchServiceId(p, allFindings, graph.data) ?? '__org';
    groups.set(key, [...(groups.get(key) ?? []), p]);
  }
  // 서비스 안에서는 바로 승인할 수 있는(검증 통과) 수정안을 점수 개선 폭이 큰 순서로 먼저 보여 주고,
  // 사람이 정해야 하는 검토 메모·검증 탈락·처리 완료 항목은 뒤로 보낸다.
  const rank = (p: Patch) => (p.status === 'validated' && p.ops.length ? 0 : isPendingPatch(p) ? 1 : 2);
  const gain = (p: Patch) => (p.validation?.score_after ?? 0) - (p.validation?.score_before ?? 0);
  for (const items of groups.values()) items.sort((a, b) => rank(a) - rank(b) || gain(b) - gain(a) || a.id.localeCompare(b.id));
  const order = [...groups.keys()].sort((a, b) => {
    if (a === '__org') return 1;
    if (b === '__org') return -1;
    const sa = services.find((s) => s.service_id === a);
    const sb = services.find((s) => s.service_id === b);
    return (sa?.tier ?? 9) - (sb?.tier ?? 9) || (sa?.score ?? 0) - (sb?.score ?? 0);
  });

  const pendingCount = list.filter(isPendingPatch).length;
  const doneCount = list.length - pendingCount;
  const canApprove = perm.decideBlocked === null;

  const onApprove = async (p: Patch) => {
    setApprovingId(p.id);
    try {
      const res = await approve.mutateAsync(p.id);
      const vnum = res.version?.number ?? (report.data ? report.data.graph_version + 1 : null);
      const before = p.validation?.score_before;
      const after = p.validation?.score_after;
      const scorePart = before !== null && before !== undefined && after !== null && after !== undefined ? ` · 전체 준비도 ${formatScore(before)} → ${formatScore(after)}` : '';
      toast.show({
        tone: 'success',
        title: `${vnum ? `v${vnum} 생성` : '새 버전 생성'}${scorePart}`,
        description: `“${p.runbook_text}” 승인 — 그래프와 리허설 결과를 새로 고쳤습니다.`,
      });
    } catch (e) {
      toast.show({ tone: 'error', title: '승인하지 못했습니다', description: errorMessage(e) });
    } finally {
      setApprovingId(null);
    }
  };

  const onConfirmReject = async () => {
    if (!rejecting) return;
    try {
      await reject.mutateAsync({ id: rejecting.id, reason });
      toast.show({ tone: 'info', title: '수정안을 거절했습니다', description: reason ? `사유: ${reason}` : undefined });
      setRejecting(null);
      setReason('');
    } catch (e) {
      toast.show({ tone: 'error', title: '거절하지 못했습니다', description: errorMessage(e) });
    }
  };

  const setFilter = (f: Filter) =>
    setParams(
      (prev) => {
        const n = new URLSearchParams(prev);
        if (f === 'pending') n.delete('show');
        else n.set('show', f);
        return n;
      },
      { replace: true },
    );

  return (
    <div className="mx-auto max-w-[1440px] p-4 2xl:p-6">
      <PageHeader
        icon={GitPullRequestArrow}
        title="수정안"
        description="AI·템플릿이 초안을 쓰고, 고정 규칙 재검사와 과거 리허설 재실행을 통과한 수정안만 팀 리드가 승인합니다. 수정안의 사람은 모두 명부에서 고릅니다."
        actions={
          <DisabledReason reason={perm.editBlocked}>
            <Button
              variant="secondary"
              icon={<RefreshCw className="size-4" aria-hidden />}
              loading={draft.isPending}
              disabled={!!perm.editBlocked}
              onClick={() =>
                draft.mutate(
                  { llm_mode: 'off' },
                  {
                    onSuccess: (r) => toast.show({ tone: 'info', title: `초안 ${r.length}건을 다시 검증했습니다`, description: '규칙 템플릿(LLM OFF)' }),
                    onError: (e) => toast.show({ tone: 'error', title: '초안을 만들지 못했습니다', description: errorMessage(e) }),
                  },
                )
              }
            >
              초안 다시 검증
            </Button>
          </DisabledReason>
        }
      />
      <div className="mt-5 flex flex-wrap items-center gap-3">
        <Segmented<Filter>
          label="수정안 상태"
          value={filter}
          onChange={setFilter}
          options={[
            { value: 'pending', label: '대기 중', icon: ListChecks, count: pendingCount },
            { value: 'done', label: '처리 완료', icon: CircleCheck, count: doneCount },
            { value: 'all', label: '전체', count: list.length },
          ]}
        />
        {serviceFilter ? (
          <button
            type="button"
            onClick={() =>
              setParams((prev) => {
                const n = new URLSearchParams(prev);
                n.delete('service');
                return n;
              })
            }
            className="rounded-full border border-line bg-surface-2 px-3 py-1 text-[13px] font-semibold"
          >
            서비스: {svcName(serviceFilter)} ✕
          </button>
        ) : null}
        {!canApprove ? (
          <p className="text-[14px] text-warn" role="note">
            {perm.publicTrial
              ? '공개 체험판(보기 전용)입니다 — 초안 검증·승인·거절은 팀 계정이 있어야 할 수 있습니다.'
              : perm.roleUnknown
                ? '서버에서 역할을 확인하기 전이라 보기 전용입니다 — 확인되면 권한에 맞게 버튼이 켜집니다.'
                : perm.canSwitchRole
                  ? '지금은 뷰어 역할입니다 — 승인·거절은 팀 리드만 할 수 있습니다 (상단 역할 전환).'
                  : '이 계정은 승인 권한이 없습니다 — 승인·거절은 팀 리드만 할 수 있습니다.'}
          </p>
        ) : null}
      </div>

      {visible.length === 0 ? (
        <EmptyState icon={CircleCheck} title={filter === 'pending' ? '대기 중인 수정안이 없습니다' : '수정안이 없습니다'} className="card mt-6">
          {filter === 'pending' ? '모든 수정안을 처리했습니다. 리허설로 결과를 다시 확인해 보세요.' : null}
        </EmptyState>
      ) : (
        <div className="mt-6 space-y-10">
          {order.map((key) => (
            <section key={key} aria-label={key === '__org' ? '전사 공통' : (svcName(key) ?? key)}>
              <h2 className="mb-3 flex items-baseline gap-2 text-xl font-extrabold">
                {key === '__org' ? '전사 공통 (신고·당직)' : svcName(key)}
                <span className="text-[14px] font-semibold text-fg-subtle">{groups.get(key)!.length}건</span>
              </h2>
              <div className="space-y-4">
                {groups.get(key)!.map((p) => (
                  <PatchCard
                    key={p.id}
                    patch={p}
                    summary={summarizePatch(p, lookup)}
                    serviceName={svcName(patchServiceId(p, allFindings, graph.data))}
                    findings={allFindings}
                    scenarios={report.data!.scenarios}
                    canApprove={canApprove}
                    lockedReason={perm.publicTrial || perm.roleUnknown ? perm.decideBlocked : null}
                    highlighted={p.id === focusPatch || (!!focusFinding && p.finding_ids.includes(focusFinding))}
                    approving={approvingId === p.id}
                    approvedVersion={versions.data?.find((v) => v.patch_id === p.id)?.number ?? null}
                    onApprove={() => void onApprove(p)}
                    onReject={() => {
                      setReason('');
                      setRejecting(p);
                    }}
                  />
                ))}
              </div>
            </section>
          ))}
        </div>
      )}

      <Dialog
        open={!!rejecting}
        onClose={() => setRejecting(null)}
        title="수정안 거절"
        description={rejecting ? `“${rejecting.runbook_text}”` : undefined}
        footer={
          <>
            <Button variant="ghost" onClick={() => setRejecting(null)}>
              취소
            </Button>
            <Button variant="danger" loading={reject.isPending} onClick={() => void onConfirmReject()} disabled={reason.trim().length < 2}>
              거절하기
            </Button>
          </>
        }
      >
        <label className="block">
          <span className="text-[14px] font-semibold">거절 사유 (감사 로그에 남습니다)</span>
          <textarea
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            rows={4}
            placeholder="예: 결제팀과 담당자 협의 중 — 다음 주 회의 후 다시 검토"
            className="mt-2 w-full rounded-xl border border-line bg-surface-2 p-3 text-[15px] placeholder:text-fg-subtle"
            autoFocus
          />
        </label>
      </Dialog>
    </div>
  );
}
