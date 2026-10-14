/** 테스트 공용 픽스처 — 실제 백엔드 출력(src/mocks)을 그대로 쓴다. */
import findingsJson from '../mocks/findings.json';
import graphJson from '../mocks/graph.json';
import patchesJson from '../mocks/patches.json';
import readinessJson from '../mocks/readiness.json';
import type { Finding, OpsGraph, Patch, ReadinessReport } from '../api/types';
import { graphViewFromOpsGraph } from '../graph/model';

export const graph = graphJson as unknown as OpsGraph;
export const findings = findingsJson as unknown as Finding[];
export const readiness = readinessJson as unknown as ReadinessReport;
export const patches = patchesJson as unknown as Patch[];
export const view = graphViewFromOpsGraph(graph, findings, 1, readiness.graph_hash);
