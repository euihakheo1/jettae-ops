/** React Query 훅 — 화면은 이 훅만 쓴다. */
import { useMutation, useQuery, useQueryClient, type QueryClient } from '@tanstack/react-query';
import { api, strictOffline } from './client';
import { ApiError } from './errors';
import { useSourceMode } from './source';
import type { DiagnoseRequest, DraftRequest, InputFiles, RehearsalRequest, RehearsalResult } from './types';

export const qk = {
  workspaces: ['workspaces'] as const,
  report: (ws: string, version?: number | null) => ['report', ws, version ?? 'latest'] as const,
  graph: (ws: string, version?: number | null) => ['graph', ws, version ?? 'latest'] as const,
  versions: (ws: string) => ['versions', ws] as const,
  diff: (ws: string, a: number, b: number) => ['diff', ws, a, b] as const,
  patches: (ws: string) => ['patches', ws] as const,
  rules: ['rules'] as const,
  documents: (ws: string) => ['documents', ws] as const,
  reportHtml: (ws: string) => ['report-html', ws] as const,
};

/** 출처 모드가 바뀌면(오프라인 전환 등) 캐시 키가 달라지도록 */
function useModeKey(): string {
  return useSourceMode() === 'api' ? 'api' : 'demo';
}

/**
 * 서버가 인정하는 역할(공개 체험판이면 보기 전용) — 한 번 받아 두고 다시 묻지 않는다.
 * 배포 빌드는 답을 받기 전까지 보기 전용으로 보여 주므로(app/permissions.ts) 실패하면 두 번 더 묻는다.
 */
export function useSession() {
  const mode = useModeKey();
  return useQuery({ queryKey: ['session', mode], queryFn: () => api.session(), staleTime: Infinity, retry: strictOffline() ? 2 : false });
}

export function useWorkspaces() {
  const mode = useModeKey();
  return useQuery({ queryKey: [...qk.workspaces, mode], queryFn: () => api.listWorkspaces() });
}

export function useReport(ws: string | undefined, version?: number | null) {
  const mode = useModeKey();
  return useQuery({
    queryKey: [...qk.report(ws ?? '', version), mode],
    queryFn: () => api.getReport(ws!, version),
    enabled: !!ws,
  });
}

export function useGraph(ws: string | undefined, version?: number | null) {
  const mode = useModeKey();
  return useQuery({
    queryKey: [...qk.graph(ws ?? '', version), mode],
    queryFn: () => api.getGraph(ws!, version),
    enabled: !!ws,
  });
}

export function useVersions(ws: string | undefined) {
  const mode = useModeKey();
  return useQuery({ queryKey: [...qk.versions(ws ?? ''), mode], queryFn: () => api.getVersions(ws!), enabled: !!ws });
}

export function useDiff(ws: string | undefined, a: number | null, b: number | null) {
  const mode = useModeKey();
  return useQuery({
    queryKey: [...qk.diff(ws ?? '', a ?? 0, b ?? 0), mode],
    queryFn: () => api.getDiff(ws!, a!, b!),
    enabled: !!ws && a !== null && b !== null && a !== b,
  });
}

export function usePatches(ws: string | undefined) {
  const mode = useModeKey();
  return useQuery({ queryKey: [...qk.patches(ws ?? ''), mode], queryFn: () => api.getPatches(ws!), enabled: !!ws });
}

export function useRules() {
  const mode = useModeKey();
  return useQuery({ queryKey: [...qk.rules, mode], queryFn: () => api.getRules(), staleTime: Infinity });
}

export function useDocuments(ws: string | undefined) {
  const mode = useModeKey();
  return useQuery({
    queryKey: [...qk.documents(ws ?? ''), mode],
    queryFn: () => api.getDocuments(ws!),
    enabled: !!ws,
    staleTime: Infinity,
  });
}

export function useReportHtml(ws: string | undefined) {
  const mode = useModeKey();
  return useQuery({ queryKey: [...qk.reportHtml(ws ?? ''), mode], queryFn: () => api.reportHtml(ws!), enabled: !!ws });
}

/** 그래프가 바뀌는 쓰기 뒤에 다시 읽을 것들 */
export function invalidateWorkspace(qc: QueryClient, ws: string): Promise<void> {
  return Promise.all(
    [['report', ws], ['graph', ws], ['versions', ws], ['patches', ws], ['diff', ws], ['report-html', ws], ['workspaces']].map(
      (key) => qc.invalidateQueries({ queryKey: key }),
    ),
  ).then(() => undefined);
}

export function useApprovePatch(ws: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.approvePatch(ws, id),
    onSuccess: () => invalidateWorkspace(qc, ws),
  });
}

export function useRejectPatch(ws: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, reason }: { id: string; reason: string }) => api.rejectPatch(ws, id, reason),
    onSuccess: () => invalidateWorkspace(qc, ws),
  });
}

export function useDraftPatches(ws: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (req: DraftRequest) => api.draftPatches(ws, req),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['patches', ws] }),
  });
}

export function useRevertVersion(ws: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (n: number) => api.revertVersion(ws, n),
    onSuccess: () => invalidateWorkspace(qc, ws),
  });
}

export function useRunRehearsals(ws: string, onResults?: (r: RehearsalResult[]) => void) {
  return useMutation({
    mutationFn: (req: RehearsalRequest) => api.runRehearsals(ws, req),
    ...(onResults ? { onSuccess: onResults } : {}),
  });
}

export function useCreateDemo() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.createDemo(),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.workspaces }),
  });
}

export interface NewDiagnosisInput {
  name: string;
  files: InputFiles;
  llm: DiagnoseRequest['llm_mode'];
}

/**
 * 새 진단: 워크스페이스 생성 → 입력 업로드 → 진단 실행.
 * 팀 계정이 없는 방문자(공개 베타에서 401/403)는 무료 검진 한 번으로 같은 결과를 받는다.
 */
export function useNewDiagnosis() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ name, files, llm }: NewDiagnosisInput) => {
      let id: string;
      try {
        ({ id } = await api.createWorkspace(name));
      } catch (e) {
        if (e instanceof ApiError && (e.status === 401 || e.status === 403)) {
          return (await api.freeScan(files)).workspace_id;
        }
        throw e;
      }
      await api.uploadInputs(id, files);
      await api.diagnose(id, { llm_mode: llm });
      return id;
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.workspaces }),
  });
}
