import { clsx } from 'clsx';
import { BotOff, Bot, ChevronDown, Database, GitCommitVertical, Moon, Projector, ShieldCheck, Eye } from 'lucide-react';
import { useEffect } from 'react';
import { Link, useLocation, useNavigate } from 'react-router';
import { useReport, useSession, useWorkspaces } from '../../api/hooks';
import { useSourceMode } from '../../api/source';
import type { Role, SessionInfo } from '../../api/types';
import { usePermissions } from '../../app/permissions';
import { canApprove, ROLE_PERSON, roleStore, themeStore, useRole, useTheme } from '../../app/prefs';
import { versionChip } from '../../lib/format';
import { LLM_MODE_LABEL } from '../../lib/labels';
import { Badge } from '../ui/Badge';
import { Tooltip } from '../ui/Tooltip';

function WorkspaceSwitcher({ ws }: { ws: string | null }) {
  const { data } = useWorkspaces();
  const navigate = useNavigate();
  const location = useLocation();
  const list = data ?? [];
  const current = list.find((w) => w.id === ws);
  if (!ws) return <span className="text-[15px] font-semibold text-fg-muted">워크스페이스를 고르세요</span>;
  return (
    <label className="relative flex items-center">
      <span className="sr-only">워크스페이스 전환</span>
      <select
        value={ws}
        onChange={(e) => {
          const rest = location.pathname.split('/').slice(4).join('/');
          void navigate(`/app/w/${encodeURIComponent(e.target.value)}${rest ? `/${rest}` : ''}`);
        }}
        className="h-10 max-w-[260px] cursor-pointer appearance-none truncate rounded-xl border border-line bg-surface-2 pr-9 pl-3 text-[15px] font-semibold text-fg hover:border-line-strong"
      >
        {!current ? <option value={ws}>{ws}</option> : null}
        {list.map((w) => (
          <option key={w.id} value={w.id}>
            {w.name}
          </option>
        ))}
      </select>
      <ChevronDown className="pointer-events-none absolute right-3 size-4 text-fg-subtle" aria-hidden />
    </label>
  );
}

function RoleSwitch() {
  const role = useRole();
  const opts: { value: Role; label: string; icon: typeof Eye }[] = [
    { value: 'viewer', label: '뷰어', icon: Eye },
    { value: 'lead', label: '팀 리드', icon: ShieldCheck },
  ];
  return (
    <div className="flex items-center gap-2">
      <div role="radiogroup" aria-label="역할 전환 (데모용 인증)" className="inline-flex items-center gap-1 rounded-xl border border-line bg-surface-2 p-1">
        {opts.map((o) => {
          const active = role === o.value || (o.value === 'lead' && role === 'admin') || (o.value === 'viewer' && role === 'editor');
          const Icon = o.icon;
          return (
            <button
              key={o.value}
              type="button"
              role="radio"
              aria-checked={active}
              onClick={() => roleStore.set(o.value)}
              className={clsx(
                'inline-flex h-8 items-center gap-1.5 rounded-lg px-2.5 text-[13px] font-semibold',
                active ? 'bg-surface text-fg ring-1 ring-line-strong' : 'text-fg-muted hover:text-fg',
              )}
            >
              <Icon className="size-4" aria-hidden />
              {o.label}
            </button>
          );
        })}
      </div>
      <span className="hidden text-[13px] text-fg-subtle xl:inline" title="X-Jettae-User">
        {ROLE_PERSON[role]}
        {canApprove(role) ? ' · 승인 가능' : ' · 읽기 전용'}
      </span>
    </div>
  );
}

const ROLE_LABEL: Record<Role, string> = { viewer: '보기 전용', editor: '편집자', lead: '팀 리드', admin: '관리자' };

/** 토큰이 역할을 정하는 서버(공개 체험판·토큰 모드): 전환기 대신 실제 역할만 보여 준다 */
function SessionBadge({ session }: { session: SessionInfo }) {
  const text = session.anonymous ? '공개 체험판 · 보기 전용' : `${session.user} · ${ROLE_LABEL[session.role]}`;
  return (
    <Badge tone={session.anonymous ? 'info' : 'brand'} icon={session.anonymous ? Eye : ShieldCheck} size="lg" title="서버가 인정한 역할">
      {text}
    </Badge>
  );
}

/** 배포 빌드에서 서버가 역할을 알려 주기 전(GET /me 대기·실패): 전환기 대신 보기 전용 표시 */
function RoleUnknownBadge({ failed }: { failed: boolean }) {
  return (
    <Badge
      tone="neutral"
      icon={Eye}
      size="lg"
      title={failed ? '서버에서 역할을 확인하지 못해 보기 전용으로 보여 줍니다. 잠시 후 새로고침하세요.' : '서버에서 역할을 확인하는 중입니다'}
    >
      {failed ? '보기 전용 · 역할 확인 실패' : '역할 확인 중 · 보기 전용'}
    </Badge>
  );
}

/** 역할 전환이 안 되는 서버라면 화면의 역할(버튼 표시 기준)을 서버 역할에 맞춘다 */
function useServerRole(session: SessionInfo | undefined): void {
  useEffect(() => {
    if (session && !session.can_switch_role && roleStore.get() !== session.role) roleStore.set(session.role);
  }, [session]);
}

export function TopBar({ ws }: { ws: string | null }) {
  const report = useReport(ws ?? undefined);
  const mode = useSourceMode();
  const theme = useTheme();
  const llm = report.data?.llm.mode;
  const sessionQuery = useSession();
  const session = sessionQuery.data;
  const perm = usePermissions();
  useServerRole(session);
  return (
    <header className="flex h-16 shrink-0 items-center gap-3 border-b border-line bg-surface/80 px-4 backdrop-blur 2xl:px-6">
      <WorkspaceSwitcher ws={ws} />
      {ws && report.data ? (
        <Link
          to={`/app/w/${encodeURIComponent(ws)}/versions`}
          className="tnum inline-flex h-8 items-center gap-1.5 rounded-lg border border-line bg-surface-2 px-2.5 font-mono text-[13px] font-semibold text-fg-muted hover:text-fg"
          title="그래프 버전 · 내용 해시"
        >
          <GitCommitVertical className="size-4" aria-hidden />
          {versionChip(report.data.graph_version, report.data.graph_hash)}
        </Link>
      ) : null}
      {llm ? (
        <Tooltip
          focusable
          side="bottom"
          wide
          content="끊김 판정·준비도 점수·리허설 재생은 고정 규칙과 결정적 리플레이로 계산합니다. 판정 경로의 LLM 호출은 항상 0회입니다."
        >
          <Badge tone={llm === 'off' ? 'ok' : 'brand'} icon={llm === 'off' ? BotOff : Bot} size="lg">
            {LLM_MODE_LABEL[llm]} — 판정은 규칙
          </Badge>
        </Tooltip>
      ) : null}
      {mode !== 'api' ? (
        <Badge tone={mode === 'offline' ? 'warn' : 'info'} icon={Database} size="lg" title="데이터 출처">
          {mode === 'offline' ? '오프라인 데모 데이터' : '목업 데이터'}
        </Badge>
      ) : null}
      <div className="ml-auto flex items-center gap-3">
        {perm.roleUnknown ? (
          <RoleUnknownBadge failed={sessionQuery.isError} />
        ) : session && !session.can_switch_role ? (
          <SessionBadge session={session} />
        ) : (
          <RoleSwitch />
        )}
        <button
          type="button"
          onClick={() => themeStore.set(theme === 'dark' ? 'projector' : 'dark')}
          aria-pressed={theme === 'projector'}
          className="inline-flex h-10 items-center gap-2 rounded-xl border border-line bg-surface-2 px-3 text-[14px] font-semibold text-fg-muted hover:text-fg"
          title="프로젝터 모드: 밝은 고대비 화면과 큰 글씨"
        >
          {theme === 'dark' ? <Projector className="size-4" aria-hidden /> : <Moon className="size-4" aria-hidden />}
          <span className="hidden lg:inline">{theme === 'dark' ? '프로젝터 모드' : '어두운 화면'}</span>
        </button>
      </div>
    </header>
  );
}
