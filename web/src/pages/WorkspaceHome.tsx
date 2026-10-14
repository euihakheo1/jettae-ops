import { clsx } from 'clsx';
import {
  ArrowRight,
  Bot,
  BotOff,
  FileText,
  FileUp,
  FlaskConical,
  FolderOpen,
  GitBranch,
  GitPullRequestArrow,
  LayoutDashboard,
  PlayCircle,
  Rocket,
  ScrollText,
  Telescope,
  Upload,
  X,
  type LucideIcon,
} from 'lucide-react';
import { useRef, useState, type DragEvent } from 'react';
import { Link, useNavigate } from 'react-router';
import { api } from '../api/client';
import { errorMessage } from '../api/errors';
import { useCreateDemo, useNewDiagnosis, useWorkspaces } from '../api/hooks';
import { useSourceMode } from '../api/source';
import type { LlmMode, WorkspaceSummary } from '../api/types';
import { ScorePill } from '../components/domain/Score';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { EmptyState, ErrorState, Skeleton } from '../components/ui/States';
import { useToast } from '../components/ui/Toast';
import { trackUploadClick } from '../lib/demand';
import { formatDateTime, versionChip } from '../lib/format';

const RUNBOOK_EXT = ['.md', '.txt', '.pdf', '.docx', '.html', '.htm'];

const LLM_OPTIONS: { value: LlmMode; title: string; help: string; icon: typeof Bot }[] = [
  {
    value: 'off',
    title: 'OFF · 규칙 파서만',
    help: 'LLM 호출 0회. 한국어 규칙 파서로만 읽습니다. 같은 문서면 언제나 같은 결과가 나옵니다 (권장).',
    icon: BotOff,
  },
  {
    value: 'live',
    title: 'LIVE · 생성형 AI 추출',
    help: 'AI가 문장에서 담당·상향·승인 후보를 뽑습니다. 원문 위치에 맞지 않는 값은 버리고, 판정·점수는 여전히 규칙으로 계산합니다. 문서 내용(개인정보 마스킹 후)이 외부 LLM API로 전송됩니다.',
    icon: Bot,
  },
  {
    value: 'replay',
    title: 'REPLAY · 녹화 재생',
    help: '미리 녹화한 AI 응답을 재생합니다. 네트워크 없이 결정적이라 시연·회귀 테스트용입니다. 녹화가 없거나 실패하면 규칙 파서로 폴백하고 보고서에 기록합니다.',
    icon: PlayCircle,
  },
];

type Slot = 'roster' | 'oncall' | 'catalog' | 'policy';
const SLOTS: { key: Slot; label: string; accept: string; help: string }[] = [
  { key: 'roster', label: '인사 명부', accept: '.csv', help: 'roster.csv — 재직·퇴사·휴직, 직급, 연락 수단' },
  { key: 'oncall', label: '당직표', accept: '.csv', help: 'oncall.csv — 이름, 시작, 종료, 레벨' },
  { key: 'catalog', label: '서비스 카탈로그', accept: '.yaml,.yml,.json', help: 'catalog.yaml — 서비스·의존·위치' },
  { key: 'policy', label: '정책', accept: '.yaml,.yml', help: 'policy.yaml — 핵심 등급·목표 복구 시간·당직 기준' },
];

export function WorkspaceHome() {
  const workspaces = useWorkspaces();
  const createDemo = useCreateDemo();
  const diagnose = useNewDiagnosis();
  const navigate = useNavigate();
  const toast = useToast();
  const mode = useSourceMode();
  const [name, setName] = useState('');
  const [runbooks, setRunbooks] = useState<File[]>([]);
  const [structured, setStructured] = useState<Partial<Record<Slot, File>>>({});
  const [llm, setLlm] = useState<LlmMode>('off');
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const addFiles = (files: FileList | File[]) => {
    const list = [...files];
    const ok = list.filter((f) => RUNBOOK_EXT.some((e) => f.name.toLowerCase().endsWith(e)));
    const rejected = list.length - ok.length;
    // 업로드 의향은 페이지 세션마다 한 번, 받을 수 있는 런북 파일을 처음 골랐을 때만 센다(수는 100에서 자른다)
    trackUploadClick(ok.length);
    setRunbooks((cur) => [...cur, ...ok.filter((f) => !cur.some((c) => c.name === f.name && c.size === f.size))]);
    if (rejected) toast.show({ tone: 'error', title: `${rejected}개 파일을 건너뛰었습니다`, description: `런북은 ${RUNBOOK_EXT.join(' ')} 형식만 받습니다.` });
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
  };

  const startDemo = () => {
    void api.track({ name: 'demo_start', path: '/app', props: { from: 'workspace_home' } });
    createDemo.mutate(undefined, {
      onSuccess: ({ id }) => void navigate(`/app/w/${encodeURIComponent(id)}`),
      onError: (e) => toast.show({ tone: 'error', title: '예제를 불러오지 못했습니다', description: errorMessage(e) }),
    });
  };

  const submit = () => {
    diagnose.mutate(
      { name: name.trim() || '새 진단', files: { runbooks, ...structured }, llm },
      {
        onSuccess: (id) => {
          toast.show({ tone: 'success', title: '진단을 마쳤습니다', description: '끊긴 대응 경로를 확인하세요.' });
          void navigate(`/app/w/${encodeURIComponent(id)}`);
        },
        onError: (e) => toast.show({ tone: 'error', title: '진단을 실행하지 못했습니다', description: errorMessage(e) }),
      },
    );
  };

  return (
    <div className="mx-auto max-w-[1280px] p-4 2xl:p-8">
      <div className="flex flex-wrap items-end gap-4">
        <div className="min-w-0 flex-1">
          <h1 className="text-[30px] font-extrabold tracking-tight">워크스페이스</h1>
          <p className="mt-1 text-[16px] text-fg-muted">런북을 올려 진단하거나, 별빛상점 예제로 바로 둘러보세요.</p>
        </div>
      </div>

      <section aria-label="별빛상점 예제" className="card relative mt-6 overflow-hidden p-6 2xl:p-8">
        <div aria-hidden className="absolute -top-24 -right-10 size-72 rounded-full bg-brand/20 blur-3xl" />
        <div className="relative flex flex-wrap items-center gap-6">
          <div className="grid size-16 place-items-center rounded-2xl bg-brand/15 text-brand">
            <Telescope className="size-8" aria-hidden />
          </div>
          <div className="min-w-0 flex-1">
            <p className="text-[14px] font-semibold text-brand">결선 시연용 예제 · 가상 회사</p>
            <h2 className="mt-0.5 text-[24px] font-extrabold">별빛상점(DMC) 예제 불러오기</h2>
            <p className="mt-1 max-w-2xl text-[15.5px] leading-relaxed text-fg-muted">
              상암 DMC의 천체 관측 용품 쇼핑몰. 문체가 제각각인 런북 3건, 명부 12명, 당직표 2주, 서비스 13개 — 퇴사자에게 걸린 결제 장애 대응부터 법정 기한보다 늦은 침해사고 신고까지 곳곳이 끊겨 있습니다.
            </p>
          </div>
          <Button variant="primary" size="lg" icon={<Rocket className="size-5" aria-hidden />} loading={createDemo.isPending} onClick={startDemo}>
            예제로 시작
          </Button>
        </div>
      </section>

      <div className="mt-6 grid gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)]">
        <section aria-label="워크스페이스 목록">
          <h2 className="mb-3 text-[18px] font-bold">내 워크스페이스</h2>
          {workspaces.isError ? (
            <ErrorState error={workspaces.error} onRetry={() => void workspaces.refetch()} className="card" />
          ) : !workspaces.data ? (
            <div className="space-y-3">
              <Skeleton className="h-28 rounded-2xl" />
              <Skeleton className="h-28 rounded-2xl" />
            </div>
          ) : workspaces.data.length === 0 ? (
            <EmptyState icon={FolderOpen} title="아직 워크스페이스가 없습니다" className="card">
              오른쪽에서 런북을 올리거나 별빛상점 예제를 불러오세요.
            </EmptyState>
          ) : (
            <ul className="space-y-3">
              {workspaces.data.map((w) => (
                <li key={w.id}>
                  <Link to={`/app/w/${encodeURIComponent(w.id)}`} className="card group flex items-center gap-4 p-5 hover:border-line-strong">
                    <div className="grid size-12 shrink-0 place-items-center rounded-xl bg-surface-3 text-fg-muted">
                      <FolderOpen className="size-6" aria-hidden />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <h3 className="truncate text-[18px] font-bold">{w.name}</h3>
                        {w.is_demo ? <Badge tone="brand" size="sm">예제</Badge> : null}
                        {w.is_free_scan ? (
                          <Badge tone="neutral" size="sm" title="결과를 여는 키가 이 브라우저에만 저장되어 있습니다">
                            내 무료 검진
                          </Badge>
                        ) : null}
                      </div>
                      <p className="mt-0.5 text-[13.5px] text-fg-muted">
                        {w.latest_version ? versionChip(w.latest_version, w.graph_hash) : '진단 전'}
                        {w.finding_count !== undefined && w.finding_count !== null ? ` · 끊김 ${w.finding_count}건` : ''}
                        {w.created_at ? ` · ${formatDateTime(w.created_at)}` : ''}
                      </p>
                    </div>
                    {w.overall !== undefined && w.overall !== null ? <ScorePill score={w.overall} /> : null}
                    <ArrowRight className="size-5 text-fg-subtle transition-transform group-hover:translate-x-1" aria-hidden />
                  </Link>
                </li>
              ))}
            </ul>
          )}
          {workspaces.data?.length ? <DemoFlow ws={workspaces.data.find((w) => w.is_demo) ?? workspaces.data[0]!} /> : null}
        </section>

        <section aria-label="새 진단" className="card p-6">
          <h2 className="flex items-center gap-2 text-[18px] font-bold">
            <Upload className="size-5 text-brand" aria-hidden />새 진단
          </h2>
          {mode !== 'api' ? (
            <p className="mt-2 rounded-xl border border-warn/40 bg-warn/10 px-3 py-2 text-[14px]">
              {mode === 'offline' ? '오프라인 데모 모드' : '목업 모드'}에서는 업로드 진단을 실행할 수 없습니다. API 서버(<code>jettae serve</code>)를 켜거나 위의 예제를 사용하세요.
            </p>
          ) : null}
          <label className="mt-4 block">
            <span className="text-[14px] font-semibold">워크스페이스 이름</span>
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="예: 우리 회사 결제팀" className="mt-1.5 h-11 w-full rounded-xl border border-line bg-surface-2 px-3 text-[15px] placeholder:text-fg-subtle" />
          </label>

          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            className={clsx('mt-4 rounded-2xl border-2 border-dashed p-6 text-center transition-colors', dragging ? 'border-brand bg-brand/8' : 'border-line-strong bg-surface-2/50')}
          >
            <FileUp className="mx-auto size-8 text-fg-subtle" aria-hidden />
            <p className="mt-2 text-[16px] font-semibold">런북 파일을 끌어다 놓으세요</p>
            <p className="text-[13.5px] text-fg-muted">.md .txt .pdf .docx .html · 여러 개 가능 · 문서 속 지시문은 따르지 않고 격리합니다</p>
            <Button className="mt-3" size="sm" onClick={() => fileInput.current?.click()}>
              파일 고르기
            </Button>
            <input
              ref={fileInput}
              type="file"
              multiple
              accept={RUNBOOK_EXT.join(',')}
              className="sr-only"
              aria-label="런북 파일 선택"
              onChange={(e) => {
                if (e.target.files) addFiles(e.target.files);
                e.target.value = '';
              }}
            />
          </div>
          {runbooks.length ? (
            <ul className="mt-3 space-y-1.5">
              {runbooks.map((f) => (
                <li key={`${f.name}:${f.size}`} className="flex items-center gap-2 rounded-lg bg-surface-2 px-3 py-1.5 text-[14px]">
                  <FileText className="size-4 text-fg-subtle" aria-hidden />
                  <span className="min-w-0 flex-1 truncate">{f.name}</span>
                  <span className="text-[12px] text-fg-subtle">{Math.ceil(f.size / 1024)}KB</span>
                  <button type="button" aria-label={`${f.name} 빼기`} onClick={() => setRunbooks((cur) => cur.filter((c) => c !== f))} className="rounded p-0.5 text-fg-subtle hover:text-fg">
                    <X className="size-4" aria-hidden />
                  </button>
                </li>
              ))}
            </ul>
          ) : null}

          <fieldset className="mt-5">
            <legend className="text-[14px] font-semibold">함께 대조할 자료 (선택)</legend>
            <div className="mt-2 grid gap-2 sm:grid-cols-2">
              {SLOTS.map((s) => (
                <label key={s.key} className="block cursor-pointer rounded-xl border border-line bg-surface-2 p-3 hover:border-line-strong">
                  <span className="text-[14px] font-semibold">{s.label}</span>
                  <span className="block truncate text-[12.5px] text-fg-muted">{structured[s.key]?.name ?? s.help}</span>
                  <input
                    type="file"
                    accept={s.accept}
                    className="sr-only"
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      setStructured((cur) => ({ ...cur, [s.key]: f }));
                    }}
                  />
                </label>
              ))}
            </div>
          </fieldset>

          <fieldset className="mt-5">
            <legend className="text-[14px] font-semibold">LLM 모드</legend>
            <div role="radiogroup" className="mt-2 space-y-2">
              {LLM_OPTIONS.map((o) => {
                const Icon = o.icon;
                const on = llm === o.value;
                return (
                  <label key={o.value} className={clsx('flex cursor-pointer gap-3 rounded-xl border p-3', on ? 'border-brand/60 bg-brand/8' : 'border-line bg-surface-2 hover:border-line-strong')}>
                    <input type="radio" name="llm" value={o.value} checked={on} onChange={() => setLlm(o.value)} className="mt-1 accent-[var(--brand)]" />
                    <span className="min-w-0">
                      <span className="flex items-center gap-1.5 text-[15px] font-bold">
                        <Icon className="size-4" aria-hidden />
                        {o.title}
                      </span>
                      <span className="mt-0.5 block text-[13.5px] leading-relaxed text-fg-muted">{o.help}</span>
                    </span>
                  </label>
                );
              })}
            </div>
            <p className="mt-2 text-[13px] text-fg-subtle">어느 모드든 끊김 판정·준비도 점수·리허설은 고정 규칙으로 계산합니다(LLM 0회).</p>
          </fieldset>

          <Button variant="primary" size="lg" className="mt-5 w-full" icon={<PlayCircle className="size-5" aria-hidden />} disabled={runbooks.length === 0} loading={diagnose.isPending} onClick={submit}>
            진단 시작
          </Button>
          <p className="mt-3 text-[13px] leading-relaxed text-fg-subtle">
            팀 계정 없이 쓰는 공개 체험판에서는 <b className="font-semibold text-fg-muted">무료 검진</b>으로 실행됩니다: LLM을 쓰지 않고(외부 전송 없음), 원문은 저장하지 않으며, 결과는 이
            브라우저에서만 다시 열 수 있고 7일 뒤 자동 삭제됩니다. 런북 본문과 카탈로그 설명의 연락처·주민번호 등은 자동으로 가리고, 인사 명부는 이름·역할·재직 상태와
            연락 수단의 종류만 진단에 씁니다(번호·주소는 저장하지 않음).{' '}
            <Link to="/privacy" className="font-semibold text-fg-muted underline underline-offset-2 hover:text-fg">
              개인정보 처리방침
            </Link>
          </p>
        </section>
      </div>
    </div>
  );
}

const FLOW: { icon: LucideIcon; title: string; body: string; path: string }[] = [
  { icon: LayoutDashboard, title: '진단', body: '끊긴 대응 경로를 빨간색으로 · 준비도 0~100', path: '' },
  { icon: FlaskConical, title: '리허설', body: '추석 새벽 결제 장애를 그래프 위에서 재생', path: '/rehearsals?only=fail' },
  { icon: GitPullRequestArrow, title: '수정안 승인', body: 'AI 초안 → 규칙 재검사 → 팀 리드 승인', path: '/patches' },
  { icon: GitBranch, title: '그래프 버전', body: '승인마다 새 버전 · 되돌리기도 새 버전', path: '/versions' },
  { icon: ScrollText, title: '증적 리포트', body: 'ISMS-P 2.11.4 · 2.11.5 · 2.12.2 초안', path: '/report' },
];

/** 시연 순서 바로가기 — 진단 → 리허설 → 승인 → 버전 → 증적 */
function DemoFlow({ ws }: { ws: WorkspaceSummary }) {
  const base = `/app/w/${encodeURIComponent(ws.id)}`;
  return (
    <section aria-label="시연 순서" className="card mt-6 p-5">
      <h2 className="text-[16px] font-bold">
        {ws.name} <span className="font-semibold text-fg-subtle">· 한 바퀴 둘러보기</span>
      </h2>
      <ol className="mt-3 space-y-1.5">
        {FLOW.map(({ icon: Icon, title, body, path }, i) => (
          <li key={title}>
            <Link to={`${base}${path}`} className="group flex items-center gap-3 rounded-xl px-2 py-2 hover:bg-surface-2">
              <span className="tnum grid size-7 shrink-0 place-items-center rounded-full bg-brand/14 text-[13px] font-extrabold text-brand">{i + 1}</span>
              <Icon className="size-5 shrink-0 text-fg-muted" aria-hidden />
              <span className="min-w-0 flex-1">
                <span className="text-[15.5px] font-bold">{title}</span>
                <span className="ml-2 text-[14px] text-fg-muted">{body}</span>
              </span>
              <ArrowRight className="size-4 shrink-0 text-fg-subtle transition-transform group-hover:translate-x-1" aria-hidden />
            </Link>
          </li>
        ))}
      </ol>
    </section>
  );
}
