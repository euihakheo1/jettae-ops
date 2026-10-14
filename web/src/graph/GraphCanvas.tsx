import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  Panel,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  type EdgeTypes,
  type FitViewOptions,
  type NodeTypes,
} from '@xyflow/react';
import { clsx } from 'clsx';
import { LoaderCircle } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, type KeyboardEvent, type ReactNode } from 'react';
import { useTheme } from '../app/prefs';
import { NODE_KIND_LABEL } from '../lib/labels';
import type { GraphModel, OpsEdgeData, OpsFlowEdge, OpsFlowNode, OpsNodeData } from './model';
import { EdgeMarkers, OpsEdge } from './OpsEdge';
import { OpsNode } from './OpsNode';
import { useLayout } from './useLayout';

const nodeTypes: NodeTypes = { ops: OpsNode };
const edgeTypes: EdgeTypes = { ops: OpsEdge };

export interface GraphCanvasProps {
  model: GraphModel;
  ariaLabel: string;
  selectedId?: string | null;
  onNodeSelect?: (id: string, data: OpsNodeData) => void;
  onEdgeSelect?: (id: string, data: OpsEdgeData) => void;
  onPaneClick?: () => void;
  compact?: boolean;
  /** 전체 보기용 배치 (좌우로 넓게) */
  overview?: boolean;
  minimap?: boolean;
  controls?: boolean;
  /** 이 값이 바뀌면 화면 맞춤 */
  fitKey?: string;
  /** 줌 인 후 중앙에 둘 노드 */
  centerOn?: string | null;
  /** 오른쪽을 가리는 패널 너비(px) — 맞춤 시 비워 둔다 */
  reserveRight?: number;
  /** 이 노드들만 화면에 맞춘다 (선택한 끊김의 관련 요소) */
  fitTo?: readonly string[] | null;
  className?: string;
  children?: ReactNode;
}

function nodeAria(n: OpsFlowNode): string {
  const d = n.data;
  const kind = d.kind === 'ghost' ? '빠진 요소' : NODE_KIND_LABEL[d.kind];
  const status = d.badges.map((b) => b.text).join(', ');
  return `${kind}: ${d.label}${d.sublabel ? ` (${d.sublabel})` : ''}${status ? ` — ${status}` : ''}${d.findingIds.length ? ` — 끊김 ${d.findingIds.length}건` : ''}`;
}

function CanvasInner({
  model,
  ariaLabel,
  selectedId,
  onNodeSelect,
  onEdgeSelect,
  onPaneClick,
  compact = false,
  overview = false,
  minimap = true,
  controls = true,
  fitKey,
  centerOn,
  reserveRight = 0,
  fitTo,
  className,
  children,
}: GraphCanvasProps) {
  const theme = useTheme();
  const layout = useLayout(model, { compact, overview });
  const rf = useReactFlow();

  const nodes = useMemo(() => {
    if (!layout.nodes) return [];
    return layout.nodes.map((n) => ({
      ...n,
      ariaLabel: nodeAria(n),
      selectable: true,
      draggable: false,
      data: n.data.selected === (n.id === selectedId) ? n.data : { ...n.data, selected: n.id === selectedId },
    }));
  }, [layout.nodes, selectedId]);

  // 라벨 겹침 줄이기: 한 노드에서 여러 갈래로 퍼지면 라벨을 도착 쪽에, 여러 갈래가 한 노드로 모이면 출발 쪽에 둔다
  const routes = layout.routes;
  const edges = useMemo((): OpsFlowEdge[] => {
    const out = new Map<string, number>();
    const inn = new Map<string, number>();
    for (const e of model.edges) {
      out.set(e.source, (out.get(e.source) ?? 0) + 1);
      inn.set(e.target, (inn.get(e.target) ?? 0) + 1);
    }
    return model.edges.map((e) => {
      if (!e.data) return e;
      const route = routes?.[e.id];
      const fanOut = out.get(e.source) ?? 1;
      const fanIn = inn.get(e.target) ?? 1;
      const labelT = fanOut > fanIn ? 0.66 : fanIn > fanOut ? 0.34 : undefined;
      if (!route && labelT === undefined) return e;
      return { ...e, data: { ...e.data, ...(route ? { route } : {}), ...(labelT !== undefined ? { labelT } : {}) } };
    });
  }, [model.edges, routes]);

  const fitToKey = fitTo?.join('|') ?? '';
  /** 맞춤 옵션: 오른쪽 패널(상세 서랍)이 가리는 폭은 비워 두고, 선택한 끊김이 있으면 그 요소들만 */
  const fitOptions = useMemo((): FitViewOptions => {
    const ids = fitTo?.filter((id) => layout.nodes?.some((n) => n.id === id)) ?? [];
    const base = compact ? 28 : 44;
    // 아래쪽 확대·축소 버튼과 위쪽 범례 자리를 비운다
    const top = compact ? base : base + 20;
    return {
      padding: { top: `${top}px`, bottom: `${base}px`, left: `${base}px`, right: `${base + reserveRight}px` },
      maxZoom: compact ? 1.05 : ids.length ? 1.05 : 1.1,
      minZoom: 0.15,
      ...(ids.length ? { nodes: ids.map((id) => ({ id })) } : {}),
    };
    // fitToKey가 fitTo 내용을 대표한다
  }, [compact, reserveRight, fitToKey, layout.nodes]);

  const fit = useCallback(
    (duration: number) => {
      void rf.fitView({ ...fitOptions, duration });
    },
    [rf, fitOptions],
  );

  // 배치가 끝나면(또는 집중·선택·서랍 폭이 바뀌면) 화면 맞춤.
  // fitView는 노드 측정이 끝날 때까지 React Flow 안에서 대기했다가 적용된다.
  useEffect(() => {
    if (!layout.nodes) return;
    const a = window.setTimeout(() => fit(380), 20);
    // 글꼴 로딩 등으로 크기가 한 번 더 바뀌는 경우를 위해 한 번 더 맞춘다
    const b = window.setTimeout(() => fit(260), 650);
    return () => {
      window.clearTimeout(a);
      window.clearTimeout(b);
    };
  }, [layout.key, fitKey, layout.nodes === null, reserveRight, fitToKey]);

  useEffect(() => {
    if (!centerOn || !layout.nodes) return;
    const n = layout.nodes.find((x) => x.id === centerOn);
    if (!n) return;
    void rf.setCenter(n.position.x + (n.width ?? 0) / 2, n.position.y + (n.height ?? 0) / 2, { zoom: 1.05, duration: 500 });
  }, [centerOn, layout.nodes, rf]);

  const wrapRef = useRef<HTMLDivElement>(null);
  const fitRef = useRef(fit);
  useEffect(() => {
    fitRef.current = fit;
  });
  useEffect(() => {
    const el = wrapRef.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    let last = { w: el.clientWidth, h: el.clientHeight };
    let t = 0;
    const ro = new ResizeObserver(() => {
      const w = el.clientWidth;
      const h = el.clientHeight;
      if (Math.abs(w - last.w) < 24 && Math.abs(h - last.h) < 24) return;
      last = { w, h };
      window.clearTimeout(t);
      t = window.setTimeout(() => fitRef.current(300), 120);
    });
    ro.observe(el);
    return () => {
      ro.disconnect();
      window.clearTimeout(t);
    };
  }, []);

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    const el = (e.target as HTMLElement).closest('.react-flow__node');
    const id = el?.getAttribute('data-id');
    if (!id) return;
    const n = nodes.find((x) => x.id === id);
    if (n && onNodeSelect) {
      e.preventDefault();
      onNodeSelect(n.id, n.data);
    }
  };

  return (
    <div ref={wrapRef} className={clsx('relative h-full w-full', className)} onKeyDown={onKeyDown} role="region" aria-label={ariaLabel}>
      <EdgeMarkers />
      <ReactFlow<OpsFlowNode, OpsFlowEdge>
        nodes={nodes}
        edges={layout.nodes ? edges : []}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        colorMode={theme === 'projector' ? 'light' : 'dark'}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable
        edgesFocusable={false}
        proOptions={{ hideAttribution: true }}
        minZoom={0.15}
        maxZoom={2.2}
        onNodeClick={(_, n) => onNodeSelect?.(n.id, n.data)}
        onEdgeClick={(_, e) => e.data && onEdgeSelect?.(e.id, e.data)}
        onPaneClick={() => onPaneClick?.()}
        fitView
        fitViewOptions={fitOptions}
      >
        <Background variant={BackgroundVariant.Dots} gap={22} size={1.4} color="var(--grid-dot)" />
        {controls ? <Controls showInteractive={false} position="bottom-left" aria-label="확대·축소" /> : null}
        {minimap ? (
          <MiniMap
            position="bottom-right"
            pannable
            zoomable
            style={{ width: 168, height: 112 }}
            ariaLabel="미니맵"
            nodeBorderRadius={8}
            nodeColor={(n) => {
              const d = n.data as OpsNodeData;
              if (d.kind === 'ghost' || d.departed) return 'var(--danger)';
              if (d.severity === 'critical' || d.severity === 'high') return 'color-mix(in oklab, var(--danger) 70%, var(--surface-3))';
              if (d.kind === 'service') return 'var(--brand)';
              return 'var(--line-strong)';
            }}
          />
        ) : null}
        {children ? <Panel position="top-left">{children}</Panel> : null}
      </ReactFlow>
      {layout.nodes === null ? (
        <div className="absolute inset-0 grid place-items-center bg-bg/40" role="status" aria-live="polite">
          <span className="inline-flex items-center gap-2 rounded-full border border-line bg-surface px-4 py-2 text-sm text-fg-muted shadow">
            <LoaderCircle className="size-4 animate-spin" aria-hidden />
            {layout.error ? '그래프 배치에 실패했습니다' : '그래프 배치 중…'}
          </span>
        </div>
      ) : null}
    </div>
  );
}

export function GraphCanvas(props: GraphCanvasProps) {
  return (
    <ReactFlowProvider>
      <CanvasInner {...props} />
    </ReactFlowProvider>
  );
}
