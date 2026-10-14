import { clsx } from 'clsx';
import { GitBranch, GitCommitVertical, GitCompare, Undo2 } from 'lucide-react';
import { useMemo, useState } from 'react';
import { Link, useParams, useSearchParams } from 'react-router';
import { errorMessage } from '../api/errors';
import { useDiff, useGraph, usePatches, useRevertVersion, useVersions } from '../api/hooks';
import type { GraphDiff, GraphVersion, Node, PatchOp } from '../api/types';
import { usePermissions } from '../app/permissions';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { Dialog } from '../components/ui/Dialog';
import { EmptyState, ErrorState, LoadingBlock } from '../components/ui/States';
import { useToast } from '../components/ui/Toast';
import { Tooltip } from '../components/ui/Tooltip';
import { formatDateTime, hashPrefix } from '../lib/format';
import { lookupFrom, summarizeOp, type NodeLookup, type OpSummary } from '../lib/patchSummary';
import { PageHeader } from './PageHeader';

/** 버전 비교 결과 → 사람이 읽는 줄 (수정안 요약과 같은 문장) */
export function diffLines(diff: GraphDiff, lookup: NodeLookup): { added: OpSummary[]; removed: OpSummary[]; changed: OpSummary[] } {
  const look: NodeLookup = (id) => lookup(id) ?? diff.added_nodes.find((n) => n.id === id) ?? diff.removed_nodes.find((n) => n.id === id);
  const added: OpSummary[] = [
    ...diff.added_nodes.map((n) => summarizeOp({ op: 'add_node', node: n }, look)),
    ...diff.added_edges.map((e) => summarizeOp({ op: 'add_edge', edge: e }, look)),
  ];
  const removed: OpSummary[] = [
    ...diff.removed_nodes.map((n) => summarizeOp({ op: 'remove_node', node_id: n.id, snapshot: n }, look)),
    ...diff.removed_edges.map((e) => summarizeOp({ op: 'remove_edge', edge_id: e.id, snapshot: e }, look)),
  ];
  const changed: OpSummary[] = [...diff.changed_nodes, ...diff.changed_edges].flatMap((c) =>
    c.attrs.map((a) => summarizeOp({ op: 'set_attr', target_id: c.id, key: a.key, value: a.after, previous: a.before } satisfies PatchOp, look)),
  );
  return { added, removed, changed };
}

export function Versions() {
  const { ws = '' } = useParams();
  const [params, setParams] = useSearchParams();
  const versions = useVersions(ws);
  const patches = usePatches(ws);
  const graph = useGraph(ws);
  const perm = usePermissions();
  const revert = useRevertVersion(ws);
  const toast = useToast();
  const [confirm, setConfirm] = useState<GraphVersion | null>(null);

  const list = useMemo(() => [...(versions.data ?? [])].sort((x, y) => y.number - x.number), [versions.data]);
  const latest = list[0];
  const b = Number(params.get('b') ?? latest?.number ?? 0) || null;
  const bVersion = list.find((v) => v.number === b);
  const a = Number(params.get('a') ?? bVersion?.parent ?? 0) || null;
  const diff = useDiff(ws, a, b);
  const lookup = useMemo(() => lookupFrom((graph.data?.nodes ?? []).map((n) => ({ ...n, prov: [] }) as Node)), [graph.data]);

  if (versions.isError) return <ErrorState error={versions.error} onRetry={() => void versions.refetch()} />;
  if (!versions.data) return <LoadingBlock className="p-8" />;

  const patchText = (id: string | null) => (id ? patches.data?.find((p) => p.id === id)?.runbook_text : undefined);
  const setPair = (na: number | null, nb: number | null) =>
    setParams(
      (prev) => {
        const n = new URLSearchParams(prev);
        if (na) n.set('a', String(na));
        else n.delete('a');
        if (nb) n.set('b', String(nb));
        else n.delete('b');
        return n;
      },
      { replace: true },
    );
  const lead = perm.decideBlocked === null;

  const doRevert = async () => {
    if (!confirm) return;
    try {
      const v = await revert.mutateAsync(confirm.number);
      toast.show({ tone: 'success', title: `v${v.number} 생성 · v${confirm.number} 되돌리기`, description: '이력은 지워지지 않고 새 버전으로 남았습니다.' });
      setPair(v.parent, v.number);
      setConfirm(null);
    } catch (e) {
      toast.show({ tone: 'error', title: '되돌리지 못했습니다', description: errorMessage(e) });
    }
  };

  const lines = diff.data ? diffLines(diff.data, lookup) : null;

  return (
    <div className="mx-auto max-w-[1440px] p-4 2xl:p-6">
      <PageHeader
        icon={GitBranch}
        title="그래프 버전"
        description="그래프는 승인된 수정안으로만 바뀌고, 버전마다 내용 해시가 붙습니다. 되돌리기도 지우는 것이 아니라 새 버전으로 남습니다(감사 가능)."
      />
      <div className="mt-6 grid gap-5 xl:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
        <section aria-label="버전 이력" className="card p-5">
          <h2 className="mb-4 text-[15px] font-bold">이력 {list.length}개</h2>
          <ol className="relative">
            {list.map((v, i) => {
              const text = patchText(v.patch_id);
              const selected = v.number === b;
              return (
                <li key={v.number} className="relative pb-5 pl-12 last:pb-0">
                  {i < list.length - 1 ? <span aria-hidden className="absolute top-10 bottom-0 left-[19px] w-0.5 bg-line" /> : null}
                  <span
                    className={clsx(
                      'absolute top-1 left-0 grid size-10 place-items-center rounded-full border-2 font-mono text-[13px] font-bold',
                      v.revert_of ? 'border-warn bg-warn/15 text-warn' : i === 0 ? 'border-brand bg-brand text-brand-fg' : 'border-line-strong bg-surface-2 text-fg-muted',
                    )}
                  >
                    v{v.number}
                  </span>
                  <div className={clsx('rounded-2xl border p-4', selected ? 'border-brand/60 bg-brand/6' : 'border-line bg-surface-2')}>
                    <div className="flex flex-wrap items-center gap-2">
                      <code className="inline-flex items-center gap-1 rounded-md bg-surface-3 px-2 py-0.5 text-[12.5px] font-semibold">
                        <GitCommitVertical className="size-3.5" aria-hidden />
                        {hashPrefix(v.graph_hash, 8)}
                      </code>
                      {i === 0 ? <Badge tone="brand" size="sm">현재</Badge> : null}
                      {v.revert_of ? <Badge tone="warn" icon={Undo2} size="sm">v{v.revert_of} 되돌리기</Badge> : null}
                      {v.patch_id === null && !v.revert_of ? <Badge tone="neutral" size="sm">최초 진단</Badge> : null}
                      <span className="ml-auto text-[13px] text-fg-subtle">{formatDateTime(v.created_at)}</span>
                    </div>
                    <p className="mt-2 text-[16px] leading-snug font-semibold">{text ? `“${text}”` : v.note}</p>
                    <p className="mt-1 text-[13.5px] text-fg-muted">
                      {v.revert_of ? '되돌린 사람' : v.patch_id ? '승인' : '생성'}: {v.created_by}
                      {v.parent ? ` · 이전 v${v.parent}` : ''}
                      {v.rehearsal_ids.length ? ` · 검증 리허설 ${v.rehearsal_ids.length}건` : ''}
                    </p>
                    <div className="mt-3 flex flex-wrap gap-2">
                      {v.parent ? (
                        <Button size="sm" variant={selected ? 'primary' : 'secondary'} icon={<GitCompare className="size-4" aria-hidden />} onClick={() => setPair(v.parent, v.number)}>
                          v{v.parent} ↔ v{v.number} 비교
                        </Button>
                      ) : null}
                      {v.number !== 1 ? (
                        lead ? (
                          <Button size="sm" variant="ghost" icon={<Undo2 className="size-4" aria-hidden />} onClick={() => setConfirm(v)}>
                            되돌리기
                          </Button>
                        ) : (
                          <Tooltip content={perm.publicTrial || perm.roleUnknown ? perm.decideBlocked : '팀 리드만 되돌릴 수 있습니다'} focusable>
                            <Button size="sm" variant="ghost" disabled icon={<Undo2 className="size-4" aria-hidden />}>
                              되돌리기
                            </Button>
                          </Tooltip>
                        )
                      ) : null}
                    </div>
                  </div>
                </li>
              );
            })}
          </ol>
        </section>

        <section aria-label="버전 비교" className="card flex flex-col p-5">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-[15px] font-bold">변경 내용</h2>
            <div className="ml-auto flex items-center gap-2 text-[14px]">
              <VersionSelect label="기준" value={a} versions={list} onChange={(n) => setPair(n, b)} />
              <span className="text-fg-subtle">→</span>
              <VersionSelect label="비교" value={b} versions={list} onChange={(n) => setPair(a, n)} />
            </div>
          </div>
          {list.length < 2 ? (
            <EmptyState icon={GitBranch} title="아직 승인된 수정안이 없습니다" className="flex-1">
              수정안을 승인하면 새 버전이 여기에 쌓이고, 무엇이 바뀌었는지 비교할 수 있습니다.{' '}
              <Link to={`/app/w/${encodeURIComponent(ws)}/patches`} className="font-semibold text-brand hover:underline">
                수정안 보러 가기
              </Link>
            </EmptyState>
          ) : !a || !b || a === b ? (
            <EmptyState title="비교할 두 버전을 고르세요" />
          ) : diff.isError ? (
            <ErrorState error={diff.error} />
          ) : !lines ? (
            <LoadingBlock />
          ) : (
            <div className="mt-4 space-y-5">
              <DiffGroup title="추가" tone="ok" items={lines.added} />
              <DiffGroup title="삭제" tone="danger" items={lines.removed} />
              <DiffGroup title="속성 변경" tone="warn" items={lines.changed} />
              {!lines.added.length && !lines.removed.length && !lines.changed.length ? <EmptyState title="두 버전의 그래프가 같습니다" /> : null}
            </div>
          )}
        </section>
      </div>

      <Dialog
        open={!!confirm}
        onClose={() => setConfirm(null)}
        title={confirm ? `v${confirm.number} 되돌리기` : '되돌리기'}
        description="이 버전에서 바뀐 연결을 역연산으로 되돌린 새 버전이 생깁니다. 기존 이력은 지워지지 않습니다."
        footer={
          <>
            <Button variant="ghost" onClick={() => setConfirm(null)}>
              취소
            </Button>
            <Button variant="danger" icon={<Undo2 className="size-4" aria-hidden />} loading={revert.isPending} onClick={() => void doRevert()}>
              새 버전으로 되돌리기
            </Button>
          </>
        }
      >
        {confirm ? <p className="rounded-xl border border-line bg-surface-2 p-3 text-[15px]">{patchText(confirm.patch_id) ?? confirm.note}</p> : null}
      </Dialog>
    </div>
  );
}

function VersionSelect({ label, value, versions, onChange }: { label: string; value: number | null; versions: GraphVersion[]; onChange: (n: number) => void }) {
  return (
    <label className="flex items-center gap-1.5">
      <span className="text-fg-subtle">{label}</span>
      <select value={value ?? ''} onChange={(e) => onChange(Number(e.target.value))} className="h-9 rounded-lg border border-line bg-surface-2 px-2 font-mono font-semibold">
        {versions.map((v) => (
          <option key={v.number} value={v.number}>
            v{v.number}
          </option>
        ))}
      </select>
    </label>
  );
}

function DiffGroup({ title, tone, items }: { title: string; tone: 'ok' | 'danger' | 'warn'; items: OpSummary[] }) {
  if (!items.length) return null;
  return (
    <div>
      <h3 className={clsx('mb-2 text-[14px] font-bold', tone === 'ok' && 'text-ok', tone === 'danger' && 'text-danger', tone === 'warn' && 'text-warn')}>
        {title} {items.length}
      </h3>
      <ul className="space-y-1.5">
        {items.map((s, i) => (
          <li
            key={i}
            className={clsx('flex gap-2 rounded-lg px-3 py-2 text-[15px]', tone === 'ok' && 'bg-ok/8', tone === 'danger' && 'bg-danger/8', tone === 'warn' && 'bg-warn/8')}
          >
            <span className={clsx('w-4 shrink-0 text-center font-mono font-bold', tone === 'ok' && 'text-ok', tone === 'danger' && 'text-danger', tone === 'warn' && 'text-warn')}>{s.sign}</span>
            <span>
              <strong>{s.label}</strong>
              <span className="text-fg-subtle">: </span>
              <span className={clsx(s.sign === '−' && 'line-through decoration-danger/60')}>{s.detail}</span>
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
