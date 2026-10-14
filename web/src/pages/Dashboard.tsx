import { clsx } from 'clsx';
import { Crosshair, Ghost, Network, Rows3, Table2, Workflow } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useParams, useSearchParams } from 'react-router';
import { useDocuments, useGraph, usePatches, useReport } from '../api/hooks';
import type { Severity } from '../api/types';
import { ErrorState, Skeleton } from '../components/ui/States';
import { GraphCanvas } from '../graph/GraphCanvas';
import { Legend } from '../graph/Legend';
import { applyHighlight, buildGraphModel, defaultFocusService, type FocusMode, type OpsEdgeData, type OpsNodeData } from '../graph/model';
import { toServiceId } from '../lib/format';
import { SEVERITY_ORDER } from '../lib/labels';
import { DetailDrawer, type Selection } from './dashboard/DetailDrawer';
import { FindingsTable, type FindingFilters, type SortKey } from './dashboard/FindingsTable';
import { KpiStrip } from './dashboard/KpiStrip';
import { ServiceList } from './dashboard/ServiceList';

type Patch = Record<string, string | null>;

function useQueryState() {
  const [params, setParams] = useSearchParams();
  const update = useCallback(
    (patch: Patch) => {
      setParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          for (const [k, v] of Object.entries(patch)) {
            if (v === null || v === '') next.delete(k);
            else next.set(k, v);
          }
          return next;
        },
        { replace: true },
      );
    },
    [setParams],
  );
  return [params, update] as const;
}

export function Dashboard() {
  const { ws = '' } = useParams();
  const [params, update] = useQueryState();
  const report = useReport(ws);
  const graph = useGraph(ws);
  const docs = useDocuments(ws);
  const patches = usePatches(ws);
  const version = report.data?.graph_version ?? 1;
  // 점수 변화는 최초 진단(v1)과 비교한다 — 승인을 여러 번 해도 '처음보다 얼마나 나아졌나'가 보이게
  const previous = useReport(version > 1 ? ws : undefined, 1);

  const tab = params.get('tab') === 'findings' ? 'findings' : 'graph';
  const ghosts = params.get('ghosts') !== '0';
  const hideStructural = params.get('struct') !== '1';
  const focusParam = params.get('focus');
  const selection: Selection = params.get('finding')
    ? { type: 'finding', id: params.get('finding')! }
    : params.get('node')
      ? { type: 'node', id: params.get('node')! }
      : params.get('edge')
        ? { type: 'edge', id: params.get('edge')! }
        : null;

  const filters: FindingFilters = {
    sev: (params.get('sev') ?? '').split(',').filter((s): s is Severity => (SEVERITY_ORDER as string[]).includes(s)),
    service: params.get('svc') ?? '',
    element: params.get('el') ?? '',
    sort: (['severity', 'rule', 'service', 'element'] as SortKey[]).find((k) => k === params.get('sort')) ?? 'severity',
    desc: params.get('desc') === '1',
  };

  const data = report.data;
  const view = graph.data;
  const serviceIds = useMemo(() => new Set(data?.readiness.services.map((s) => s.service_id)), [data]);
  const focus: FocusMode = useMemo(() => {
    if (focusParam === 'all') return { mode: 'all' };
    const resolved = focusParam ? toServiceId(focusParam) : null;
    const sid = resolved && serviceIds.has(resolved) ? resolved : defaultFocusService(data?.readiness, view);
    return sid ? { mode: 'service', serviceId: sid } : { mode: 'all' };
  }, [focusParam, serviceIds, data, view]);

  const baseModel = useMemo(() => {
    if (!data || !view) return null;
    return buildGraphModel({ view, findings: data.findings, readiness: data.readiness }, { focus, ghosts, hideStructural });
  }, [data, view, focus, ghosts, hideStructural]);

  const highlight = useMemo(() => {
    if (!baseModel || !selection || !data) return null;
    let ids: string[] = [];
    if (selection.type === 'finding') {
      const f = data.findings.find((x) => x.id === selection.id);
      if (f) ids = [...f.subject_ids, ...baseModel.nodes.filter((n) => n.data.kind === 'ghost' && n.data.findingIds.includes(f.id)).map((n) => n.id)];
      ids = [...ids, ...baseModel.edges.filter((e) => ids.includes(e.target) && e.data?.kind === 'ghost').map((e) => e.id)];
    } else if (selection.type === 'node') ids = [selection.id];
    else {
      const e = view?.edges.find((x) => x.id === selection.id);
      ids = e ? [e.id, e.src, e.dst] : [selection.id];
    }
    // 강조한 연결의 양 끝도 함께 밝힌다 (예: 알람 → 퇴사자 호출이면 알람까지)
    const idSet = new Set(ids);
    for (const e of baseModel.edges) if (idSet.has(e.id)) for (const end of [e.source, e.target]) idSet.add(end);
    ids = [...idSet];
    const nodeIds = new Set(baseModel.nodes.map((n) => n.id));
    let fitTo = ids.filter((id) => nodeIds.has(id));
    // 요소가 하나뿐이면 바로 이웃까지 보이게 맞춘다
    if (fitTo.length === 1) {
      const only = fitTo[0]!;
      const near = baseModel.edges.filter((e) => e.source === only || e.target === only).map((e) => (e.source === only ? e.target : e.source));
      fitTo = [...new Set([only, ...near])];
    }
    return { ids, fitTo };
  }, [baseModel, selection, data, view]);

  const model = useMemo(() => {
    if (!baseModel || !highlight || !selection) return baseModel;
    const isFinding = selection.type === 'finding';
    return applyHighlight(baseModel, {
      ids: highlight.ids,
      dimOthers: isFinding && highlight.ids.length > 0,
      style: isFinding ? 'finding' : 'path',
      // 집중 중인 서비스는 흐리지 않는다 (화면의 기준점)
      keepIds: baseModel.focusId ? [baseModel.focusId] : [],
    });
  }, [baseModel, highlight, selection]);

  const select = useCallback(
    (s: Selection) => {
      const clear = { finding: null, node: null, edge: null };
      if (!s) return update(clear);
      if (s.type === 'finding' && data && baseModel) {
        const f = data.findings.find((x) => x.id === s.id);
        const patch: Patch = { ...clear, finding: s.id };
        if (f?.service_id && focus.mode === 'service' && f.service_id !== focus.serviceId) patch.focus = f.service_id.replace(/^service:/, '');
        else if (f && !f.service_id && !f.subject_ids.some((id) => baseModel.nodes.some((n) => n.id === id))) patch.focus = 'all';
        return update(patch);
      }
      update({ ...clear, [s.type]: s.id });
    },
    [update, data, baseModel, focus],
  );

  if (report.isError || graph.isError) {
    return (
      <ErrorState
        error={report.error ?? graph.error}
        onRetry={() => {
          void report.refetch();
          void graph.refetch();
        }}
      />
    );
  }

  if (!data || !view || !model) return <DashboardSkeleton />;

  const focusName = focus.mode === 'service' ? data.readiness.services.find((s) => s.service_id === focus.serviceId)?.service_name : null;

  const onNodeSelect = (id: string, d: OpsNodeData) => {
    if (d.kind === 'ghost' && d.findingIds[0]) select({ type: 'finding', id: d.findingIds[0] });
    else select({ type: 'node', id });
  };
  const onEdgeSelect = (id: string, d: OpsEdgeData) => {
    if (d.findingIds[0]) select({ type: 'finding', id: d.findingIds[0] });
    else if (!id.startsWith('ghost-edge:')) select({ type: 'edge', id });
  };

  return (
    <div className="flex h-full min-h-[760px] flex-col gap-3 p-3 2xl:gap-4 2xl:p-4">
      <KpiStrip report={data} previous={previous.data} />
      <div className="grid min-h-0 flex-1 grid-cols-[296px_minmax(0,1fr)] grid-rows-[minmax(0,1fr)] gap-3 2xl:grid-cols-[336px_minmax(0,1fr)] 2xl:gap-4">
        <div className="card min-h-0 overflow-hidden">
          <ServiceList
            readiness={data.readiness}
            findings={data.findings}
            focusId={focus.mode === 'service' ? focus.serviceId : null}
            allActive={focus.mode === 'all'}
            onFocus={(sid) => update({ focus: sid.replace(/^service:/, ''), finding: null, node: null, edge: null, tab: null })}
            onShowAll={() => update({ focus: 'all', finding: null, node: null, edge: null, tab: null })}
          />
        </div>
        <section aria-label="운영 그래프" className="card relative flex min-h-0 flex-col overflow-hidden">
          <div className="flex flex-wrap items-center gap-2 border-b border-line px-3 py-2">
            <div role="tablist" aria-label="보기 전환" className="inline-flex rounded-xl border border-line bg-surface-2 p-1">
              <TabButton active={tab === 'graph'} onClick={() => update({ tab: null })} icon={Workflow} label="운영 그래프" />
              <TabButton active={tab === 'findings'} onClick={() => update({ tab: 'findings' })} icon={Table2} label={`끊김 목록 ${data.findings.length}`} />
            </div>
            {tab === 'graph' ? (
              <>
                <div className="mx-1 h-6 w-px bg-line" aria-hidden />
                <div role="radiogroup" aria-label="그래프 범위" className="inline-flex rounded-xl border border-line bg-surface-2 p-1">
                  <ScopeButton
                    active={focus.mode === 'service'}
                    onClick={() => focus.mode === 'all' && update({ focus: null })}
                    icon={Crosshair}
                    label={focus.mode === 'service' ? `집중: ${focusName ?? ''}` : '서비스 집중'}
                  />
                  <ScopeButton active={focus.mode === 'all'} onClick={() => update({ focus: 'all' })} icon={Network} label="전체 보기" />
                </div>
                <ToggleChip on={ghosts} onClick={() => update({ ghosts: ghosts ? '0' : null })} icon={Ghost} label="빠진 요소" />
                {focus.mode === 'all' ? (
                  <ToggleChip on={!hideStructural} onClick={() => update({ struct: hideStructural ? '1' : null })} icon={Rows3} label="소속·위치·의존 연결" />
                ) : null}
                <span className="ml-auto hidden text-[13px] text-fg-subtle xl:inline">
                  노드 {model.nodes.length} · 연결 {model.edges.length} · v{data.graph_version}
                </span>
              </>
            ) : null}
          </div>
          <div className="relative min-h-0 flex-1">
            {tab === 'graph' ? (
              <GraphCanvas
                model={model}
                ariaLabel={focus.mode === 'service' ? `${focusName} 서비스 운영 그래프` : '전체 운영 그래프'}
                selectedId={selection?.type === 'node' ? selection.id : null}
                onNodeSelect={onNodeSelect}
                onEdgeSelect={onEdgeSelect}
                onPaneClick={() => select(null)}
                minimap={focus.mode === 'all'}
                overview={focus.mode === 'all'}
                reserveRight={selection ? 456 : 0}
                fitTo={selection?.type === 'finding' && highlight && highlight.fitTo.length > 0 ? highlight.fitTo : null}
                fitKey={`${focus.mode}:${focus.mode === 'service' ? focus.serviceId : ''}:${ghosts}:${hideStructural}`}
              >
                <Legend defaultOpen={false} />
              </GraphCanvas>
            ) : (
              <FindingsTable
                findings={data.findings}
                readiness={data.readiness}
                docs={docs.data ?? []}
                filters={filters}
                selectedId={selection?.type === 'finding' ? selection.id : null}
                onSelect={(id) => update({ finding: id, node: null, edge: null })}
                onFilters={(f) =>
                  update({
                    sev: f.sev.length ? f.sev.join(',') : null,
                    svc: f.service || null,
                    el: f.element || null,
                    sort: f.sort === 'severity' ? null : f.sort,
                    desc: f.desc ? '1' : null,
                  })
                }
              />
            )}
            <DetailDrawer
              selection={selection}
              onClose={() => select(null)}
              report={data}
              view={view}
              docs={docs.data ?? []}
              patches={patches.data ?? data.patches}
              ws={ws}
              onSelect={select}
              onFocusService={(sid) => update({ focus: sid.replace(/^service:/, ''), tab: null })}
            />
          </div>
        </section>
      </div>
    </div>
  );
}

function TabButton({ active, onClick, icon: Icon, label }: { active: boolean; onClick: () => void; icon: typeof Workflow; label: string }) {
  return (
    <button
      type="button"
      role="tab"
      aria-selected={active}
      onClick={onClick}
      className={clsx(
        'inline-flex h-8 items-center gap-1.5 rounded-lg px-3 text-[14px] font-semibold',
        active ? 'bg-surface text-fg ring-1 ring-line-strong' : 'text-fg-muted hover:text-fg',
      )}
    >
      <Icon className="size-4" aria-hidden />
      {label}
    </button>
  );
}

function ScopeButton({ active, onClick, icon: Icon, label }: { active: boolean; onClick: () => void; icon: typeof Workflow; label: string }) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={active}
      onClick={onClick}
      className={clsx(
        'inline-flex h-8 max-w-[220px] items-center gap-1.5 truncate rounded-lg px-3 text-[14px] font-semibold',
        active ? 'bg-brand/15 text-fg ring-1 ring-brand/50' : 'text-fg-muted hover:text-fg',
      )}
    >
      <Icon className="size-4 shrink-0" aria-hidden />
      <span className="truncate">{label}</span>
    </button>
  );
}

function ToggleChip({ on, onClick, icon: Icon, label }: { on: boolean; onClick: () => void; icon: typeof Workflow; label: string }) {
  return (
    <button
      type="button"
      aria-pressed={on}
      onClick={onClick}
      className={clsx(
        'inline-flex h-9 items-center gap-1.5 rounded-xl border px-3 text-[13px] font-semibold',
        on ? 'border-line-strong bg-surface-2 text-fg' : 'border-line text-fg-subtle line-through decoration-1',
      )}
    >
      <Icon className="size-4" aria-hidden />
      {label}
    </button>
  );
}

function DashboardSkeleton() {
  return (
    <div className="flex min-h-full flex-col gap-3 p-3 2xl:p-4" role="status" aria-live="polite">
      <span className="sr-only">진단 결과를 불러오는 중…</span>
      <div className="grid grid-cols-5 gap-3">
        {Array.from({ length: 5 }, (_, i) => (
          <Skeleton key={i} className="h-[128px] rounded-2xl" />
        ))}
      </div>
      <div className="grid min-h-[600px] flex-1 grid-cols-[296px_1fr] gap-3">
        <Skeleton className="rounded-2xl" />
        <Skeleton className="rounded-2xl" />
      </div>
    </div>
  );
}
