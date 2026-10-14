import { useEffect, useMemo, useState } from 'react';
import { layoutKey, layoutModel, type EdgeRoute, type LayoutOptionsInput, type LayoutResult } from './layout';
import type { GraphModel, OpsFlowNode } from './model';

export interface LayoutState {
  nodes: OpsFlowNode[] | null;
  /** ELK가 그린 연결 경로 (엣지 id → 점들·라벨 위치) */
  routes: Record<string, EdgeRoute> | null;
  loading: boolean;
  error: unknown;
  key: string;
}

/** 모델이 바뀌면 ELK로 비동기 배치. 같은 구조면 캐시를 쓴다(강조만 바뀌는 경우 즉시 반영). */
export function useLayout(model: GraphModel, opts: LayoutOptionsInput = {}): LayoutState {
  const key = useMemo(() => layoutKey(model, opts), [model, opts.compact, opts.overview]);
  const [result, setResult] = useState<{ key: string; layout: LayoutResult } | null>(null);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    layoutModel(model, opts)
      .then((layout) => {
        if (!cancelled) setResult({ key, layout });
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e);
      });
    return () => {
      cancelled = true;
    };
    // key가 구조를 대표한다
  }, [key]);

  const nodes = useMemo(() => {
    if (!result || result.key !== key) return null;
    const pos = result.layout.positions;
    return model.nodes.map((n) => ({ ...n, position: pos[n.id] ?? { x: 0, y: 0 } }));
  }, [model, result, key]);

  const routes = result && result.key === key ? result.layout.routes : null;
  return { nodes, routes, loading: !result || result.key !== key, error, key };
}
