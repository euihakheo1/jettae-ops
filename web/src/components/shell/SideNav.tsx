import { clsx } from 'clsx';
import { FlaskConical, GitBranch, GitPullRequestArrow, Home, LayoutDashboard, ScrollText, ShieldCheck, type LucideIcon } from 'lucide-react';
import { NavLink } from 'react-router';
import { BrandWordmark } from './Brand';

interface Item {
  to: string;
  label: string;
  /** 좁은 아이콘 레일(2xl 미만)에서 쓰는 짧은 이름 — 말줄임표 없이 */
  short: string;
  icon: LucideIcon;
  end?: boolean;
}

export function SideNav({ ws, badges }: { ws: string | null; badges?: Partial<Record<'patches' | 'rehearsals', number>> }) {
  const base = ws ? `/app/w/${encodeURIComponent(ws)}` : null;
  const items: (Item & { badge?: number | undefined })[] = base
    ? [
        { to: base, label: '진단 대시보드', short: '진단', icon: LayoutDashboard, end: true },
        { to: `${base}/rehearsals`, label: '리허설', short: '리허설', icon: FlaskConical, badge: badges?.rehearsals },
        { to: `${base}/patches`, label: '수정안', short: '수정안', icon: GitPullRequestArrow, badge: badges?.patches },
        { to: `${base}/versions`, label: '그래프 버전', short: '버전', icon: GitBranch },
        { to: `${base}/report`, label: '증적 리포트', short: '리포트', icon: ScrollText },
        { to: `${base}/trust`, label: '보안·AI 사용', short: '보안·AI', icon: ShieldCheck },
      ]
    : [];
  return (
    <nav aria-label="주 메뉴" className="flex h-full w-[68px] shrink-0 flex-col border-r border-line bg-surface 2xl:w-[216px]">
      <NavLink to="/" className="flex h-16 items-center justify-center border-b border-line px-3 2xl:justify-start 2xl:px-5" aria-label="제때옵스 소개 페이지">
        <BrandWordmark compact className="2xl:hidden" />
        <BrandWordmark className="hidden 2xl:flex" />
      </NavLink>
      <ul className="flex flex-1 flex-col gap-1 p-2 2xl:p-3">
        <li>
          <NavItem to="/app" label="워크스페이스" short="홈" icon={Home} end />
        </li>
        {items.length ? <li aria-hidden className="mx-2 my-2 border-t border-line" /> : null}
        {items.map((it) => (
          <li key={it.to}>
            <NavItem {...it} />
          </li>
        ))}
      </ul>
      <div className="hidden border-t border-line p-4 text-[12px] leading-relaxed text-fg-subtle 2xl:block">
        판정·점수는 고정 규칙
        <br />
        AI는 제안만 합니다
      </div>
    </nav>
  );
}

function NavItem({ to, label, short, icon: Icon, end, badge }: Item & { badge?: number | undefined }) {
  return (
    <NavLink
      to={to}
      end={end ?? false}
      title={label}
      className={({ isActive }) =>
        clsx(
          'group relative flex flex-col items-center gap-1 rounded-xl px-1 py-2 text-[11px] font-medium transition-colors 2xl:flex-row 2xl:gap-3 2xl:px-3 2xl:py-2.5 2xl:text-[15px]',
          isActive ? 'bg-brand/14 text-fg' : 'text-fg-muted hover:bg-surface-2 hover:text-fg',
        )
      }
    >
      {({ isActive }) => (
        <>
          {isActive ? <span aria-hidden className="absolute top-2 bottom-2 left-0 w-1 rounded-r-full bg-brand 2xl:-left-3" /> : null}
          <Icon className={clsx('size-5 shrink-0', isActive ? 'text-brand' : '')} aria-hidden />
          <span className="max-w-full truncate text-center 2xl:hidden">{short}</span>
          <span className="hidden truncate text-left 2xl:inline">{label}</span>
          {badge ? (
            <span className="tnum absolute top-1 right-1 min-w-5 rounded-full bg-danger px-1 text-center text-[11px] font-bold text-white 2xl:static 2xl:ml-auto">
              {badge}
            </span>
          ) : null}
        </>
      )}
    </NavLink>
  );
}
