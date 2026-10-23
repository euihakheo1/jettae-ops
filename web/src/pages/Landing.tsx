import { clsx } from 'clsx';
import {
  ArrowRight,
  BotOff,
  Check,
  FileSearch,
  FlaskConical,
  GitPullRequestArrow,
  Lock,
  Moon,
  Network,
  Projector,
  Quote,
  Rocket,
  ShieldCheck,
  Sparkles,
  Stamp,
  Unplug,
  UserX,
  X,
  type LucideIcon,
} from 'lucide-react';
import { useEffect, useRef, type ReactNode } from 'react';
import { Link, useNavigate } from 'react-router';
import { api } from '../api/client';
import { errorMessage } from '../api/errors';
import { useCreateDemo } from '../api/hooks';
import { themeStore, useTheme } from '../app/prefs';
import { trackFirstVisit } from '../lib/demand';
import { BrandWordmark } from '../components/shell/Brand';
import { Button } from '../components/ui/Button';
import { useToast } from '../components/ui/Toast';
import { HeroGraph } from './landing/HeroGraph';
import { SignupForm } from './landing/SignupForm';

const STEPS: { icon: LucideIcon; title: string; body: string; tag: string }[] = [
  { icon: Network, title: '변환', tag: '런북 → 운영 그래프', body: '위키·PDF·표·슬랙 공지체 문서를 읽어 “누가, 몇 분 뒤, 무엇으로”를 연결한 그래프로 만듭니다. 모든 연결에는 원문 위치가 붙습니다.' },
  { icon: FileSearch, title: '진단', tag: '고정 규칙 35개', body: '퇴사자에게 걸린 호출, 돌기만 하는 상향, 승인자 없는 롤백, 법정 기한보다 늦은 신고 시한을 찾아 0~100 준비도 점수로 보여 줍니다.' },
  { icon: FlaskConical, title: '리허설', tag: '결정적 모의 장애', body: '추석 새벽·메신저 장애·센터 화재를 그래프 위에서 재생합니다. 실제 시스템은 건드리지 않고, 같은 입력은 언제나 같은 결과입니다.' },
  { icon: GitPullRequestArrow, title: '개선', tag: '사람이 승인', body: 'AI가 명부 안의 사람으로 수정 문장을 제안하고, 검증기를 통과한 수정안만 팀 리드가 승인합니다. 승인마다 새 그래프 버전이 남습니다.' },
];

const AI_DOES = ['문장에서 담당·상향·승인 후보를 뽑기 (원문 위치에 다시 맞춰 검증)', '그래프 문장이 원문과 같은 뜻인지 한 번 더 보기 (다르면 판단 보류로만 돌림)', '런북에 넣을 수정 문장 초안 쓰기', '리허설 시나리오 · 규칙 밖 위험 가설 제안 (결정적 리허설이 확인)'];
const AI_DOESNT = ['끊김 판정 · 준비도 점수 · 리허설 통과/실패 판정', '그래프 변경 — 사람이 승인한 수정안만 반영', '명부에 없는 사람 만들기 · 문서 속 지시문 따르기'];

const TRUST: { icon: LucideIcon; title: string; body: string }[] = [
  { icon: BotOff, title: '판정·점수 LLM 0회', body: '판정·점수·리허설 경로는 코드 계약으로 LLM을 쓸 수 없습니다. 규칙과 가중치는 모두 공개합니다.' },
  { icon: Quote, title: '원문 근거 위치', body: '런북에서 뽑은 요소는 모두 “문서명 : 줄”까지 원문에 맞춰 보고, 명부·당직표·카탈로그에서 온 요소는 그 출처 항목을 남깁니다. 맞지 않으면 버리고, 애매하면 판단 보류로 사람에게 넘깁니다.' },
  { icon: Stamp, title: '사람 승인만 반영', body: 'AI 초안은 제안일 뿐입니다. 팀 리드가 승인한 수정안만 새 버전이 되고, 되돌리기도 새 버전으로 남습니다.' },
  { icon: Lock, title: '원문 비저장', body: '런북 속 연락처·주민번호 등은 길이 보존 마스킹 후 처리하고, 원문은 기본적으로 저장하지 않습니다. 설치형도 지원합니다.' },
];

const PAINS: { icon: LucideIcon; quote: string; body: string }[] = [
  { icon: UserX, quote: '“김대리에게 연락”', body: '김대리는 8월 31일에 퇴사했습니다. 다음 사람도, 연락 수단도 적혀 있지 않습니다.' },
  { icon: Unplug, quote: '“20분 미응답 시 정민혁 과장에게 다시 연락”', body: '상향이 두 사람 사이를 돌기만 하고 최종 책임자에게 닿지 않습니다.' },
  { icon: ShieldCheck, quote: '“48시간 이내 KISA에 신고”', body: '정보통신망법은 24시간입니다. 런북대로 움직이면 법정 기한을 넘깁니다.' },
];

export function Landing() {
  const navigate = useNavigate();
  const createDemo = useCreateDemo();
  const toast = useToast();
  const theme = useTheme();
  const viewed = useRef(false);

  useEffect(() => {
    if (viewed.current) return;
    viewed.current = true;
    const referrer = document.referrer ? 'external' : 'direct';
    void api.track({ name: 'landing_view', path: '/', props: { referrer } });
    // 고유 방문 수: 브라우저마다 한 번만 — 서버가 받은 뒤에만 '방문함' 표시를 남긴다(식별자는 저장하지 않음)
    void trackFirstVisit(referrer);
  }, []);

  const openDemo = () => {
    void api.track({ name: 'demo_start', path: '/', props: { from: 'landing' } });
    createDemo.mutate(undefined, {
      onSuccess: ({ id }) => void navigate(`/app/w/${encodeURIComponent(id)}`),
      onError: (e) => toast.show({ tone: 'error', title: '예제를 열지 못했습니다', description: errorMessage(e) }),
    });
  };

  return (
    <div className="min-h-full overflow-x-hidden bg-bg">
      <header className="sticky top-0 z-40 border-b border-line/70 bg-bg/80 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-[1240px] items-center gap-6 px-5">
          <Link to="/" aria-label="제때옵스 홈">
            <BrandWordmark />
          </Link>
          <nav aria-label="소개 메뉴" className="hidden items-center gap-6 text-[15px] font-medium text-fg-muted md:flex">
            <a href="#how" className="hover:text-fg">
              작동 방식
            </a>
            <a href="#trust" className="hover:text-fg">
              신뢰
            </a>
            <a href="#pricing" className="hover:text-fg">
              가격
            </a>
            <a href="#signup" className="hover:text-fg">
              사전 신청
            </a>
          </nav>
          <div className="ml-auto flex items-center gap-2">
            <button
              type="button"
              onClick={() => themeStore.set(theme === 'dark' ? 'projector' : 'dark')}
              aria-label={theme === 'dark' ? '프로젝터 모드(밝은 화면)로 전환' : '어두운 화면으로 전환'}
              className="grid size-10 place-items-center rounded-xl border border-line text-fg-muted hover:text-fg"
            >
              {theme === 'dark' ? <Projector className="size-4" aria-hidden /> : <Moon className="size-4" aria-hidden />}
            </button>
            <Link to="/app" className="hidden h-10 items-center rounded-xl bg-brand px-4 text-[15px] font-semibold text-brand-fg hover:brightness-110 sm:inline-flex">
              무료 런북 건강검진
            </Link>
          </div>
        </div>
      </header>

      <main>
        <section className="relative">
          <div aria-hidden className="pointer-events-none absolute inset-x-0 -top-40 h-[560px] bg-[radial-gradient(60%_60%_at_50%_0%,color-mix(in_oklab,var(--brand)_22%,transparent),transparent)]" />
          <div className="relative mx-auto grid max-w-[1240px] items-center gap-14 px-5 pt-16 pb-24 lg:grid-cols-[1.05fr_1fr] lg:pt-24">
            <div>
              <p className="inline-flex items-center gap-2 rounded-full border border-brand/40 bg-brand/10 px-3 py-1 text-[14px] font-semibold text-brand">
                <Sparkles className="size-4" aria-hidden />
                SRE·보안 담당자를 위한 런북 건강검진
              </p>
              <h1 className="mt-5 text-[42px] leading-[1.18] font-black tracking-tight sm:text-[54px] xl:text-[60px]">
                장애 대응 매뉴얼,
                <br />
                <span className="bg-gradient-to-r from-brand to-[#c4b5fd] bg-clip-text text-transparent projector:from-brand projector:to-[#7c3aed]">장애가 나기 전에</span>
                <br />
                작동하는지 확인하세요
              </h1>
              <p className="mt-6 max-w-xl text-[19px] leading-relaxed text-fg-muted">런북을 운영 그래프로 바꾸고, 끊긴 대응 경로를 장애 전에 찾아 고칩니다.</p>
              <div className="mt-8 flex flex-wrap gap-3">
                <Link
                  to="/app"
                  className="inline-flex h-14 items-center gap-2 rounded-2xl bg-brand px-7 text-[17px] font-bold text-brand-fg shadow-[0_10px_40px_-10px_var(--brand)] hover:brightness-110"
                >
                  무료 런북 건강검진
                  <ArrowRight className="size-5" aria-hidden />
                </Link>
                <Button size="lg" className="h-14 rounded-2xl px-6 text-[17px]" icon={<Rocket className="size-5" aria-hidden />} loading={createDemo.isPending} onClick={openDemo}>
                  별빛상점 예제로 바로 보기
                </Button>
              </div>
              <ul className="mt-8 flex flex-wrap gap-x-5 gap-y-2 text-[14.5px] text-fg-muted">
                {['판정·점수 LLM 0회', '원문 근거 위치', '사람 승인만 반영', '원문 비저장'].map((t) => (
                  <li key={t} className="inline-flex items-center gap-1.5">
                    <Check className="size-4 text-ok" aria-hidden strokeWidth={3} />
                    {t}
                  </li>
                ))}
              </ul>
            </div>
            <HeroGraph />
          </div>
        </section>

        <section aria-labelledby="pain-title" className="border-y border-line bg-surface/60">
          <div className="mx-auto max-w-[1240px] px-5 py-20">
            <h2 id="pain-title" className="max-w-3xl text-[32px] leading-tight font-extrabold tracking-tight">
              문서에는 적혀 있는데, 장애가 나면 아무도 받지 않는 전화
            </h2>
            <p className="mt-3 max-w-2xl text-[17px] text-fg-muted">가상의 쇼핑몰 ‘별빛상점’ 런북에서 실제로 찾은 끊김입니다. 사람 눈으로는 지나치기 쉽습니다.</p>
            <div className="mt-10 grid gap-4 md:grid-cols-3">
              {PAINS.map(({ icon: Icon, quote, body }) => (
                <article key={quote} className="card p-6">
                  <Icon className="size-7 text-danger" aria-hidden />
                  <p className="mt-4 text-[19px] font-extrabold">{quote}</p>
                  <p className="mt-2 text-[16px] leading-relaxed text-fg-muted">{body}</p>
                </article>
              ))}
            </div>
          </div>
        </section>

        <section id="how" aria-labelledby="how-title" className="mx-auto max-w-[1240px] scroll-mt-20 px-5 py-24">
          <p className="text-[15px] font-bold text-brand">작동 방식</p>
          <h2 id="how-title" className="mt-2 text-[34px] leading-tight font-extrabold tracking-tight">
            변환 → 진단 → 리허설 → 개선, 한 바퀴에 끝나지 않는 고리
          </h2>
          <ol className="mt-10 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            {STEPS.map(({ icon: Icon, title, body, tag }, i) => (
              <li key={title} className="card relative p-6">
                <span className="absolute top-5 right-5 font-mono text-[13px] font-bold text-fg-subtle">0{i + 1}</span>
                <span className="grid size-12 place-items-center rounded-2xl bg-brand/14 text-brand">
                  <Icon className="size-6" aria-hidden />
                </span>
                <h3 className="mt-4 text-[22px] font-extrabold">{title}</h3>
                <p className="text-[14px] font-semibold text-brand">{tag}</p>
                <p className="mt-2 text-[15.5px] leading-relaxed text-fg-muted">{body}</p>
              </li>
            ))}
          </ol>
          <div className="mt-6 grid gap-4 md:grid-cols-2">
            <div className="card p-6">
              <h3 className="flex items-center gap-2 text-[19px] font-extrabold">
                <Sparkles className="size-5 text-brand" aria-hidden />
                AI가 하는 일
              </h3>
              <ul className="mt-4 space-y-2.5">
                {AI_DOES.map((t) => (
                  <li key={t} className="flex gap-2.5 text-[16px]">
                    <Check className="mt-1 size-4 shrink-0 text-ok" strokeWidth={3} aria-hidden />
                    {t}
                  </li>
                ))}
              </ul>
            </div>
            <div className="card border-danger/30 p-6">
              <h3 className="flex items-center gap-2 text-[19px] font-extrabold">
                <BotOff className="size-5 text-danger" aria-hidden />
                AI가 하지 않는 일
              </h3>
              <ul className="mt-4 space-y-2.5">
                {AI_DOESNT.map((t) => (
                  <li key={t} className="flex gap-2.5 text-[16px]">
                    <X className="mt-1 size-4 shrink-0 text-danger" strokeWidth={3} aria-hidden />
                    {t}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </section>

        <section id="trust" aria-labelledby="trust-title" className="scroll-mt-20 border-y border-line bg-surface/60">
          <div className="mx-auto max-w-[1240px] px-5 py-24">
            <p className="text-[15px] font-bold text-brand">신뢰</p>
            <h2 id="trust-title" className="mt-2 text-[34px] leading-tight font-extrabold tracking-tight">
              AI는 제안하고, 판정은 규칙이, 결정은 사람이
            </h2>
            <div className="mt-10 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
              {TRUST.map(({ icon: Icon, title, body }) => (
                <article key={title} className="card p-6">
                  <Icon className="size-7 text-ok" aria-hidden />
                  <h3 className="mt-4 text-[19px] font-extrabold">{title}</h3>
                  <p className="mt-2 text-[15.5px] leading-relaxed text-fg-muted">{body}</p>
                </article>
              ))}
            </div>
          </div>
        </section>

        <section id="pricing" aria-labelledby="pricing-title" className="mx-auto max-w-[1240px] scroll-mt-20 px-5 py-24">
          <p className="text-[15px] font-bold text-brand">가격</p>
          <h2 id="pricing-title" className="mt-2 text-[34px] leading-tight font-extrabold tracking-tight">
            검진은 무료로, 고치는 일은 함께
          </h2>
          <div className="mt-10 grid gap-4 lg:grid-cols-3">
            <PriceCard
              name="무료 검진"
              price="0원"
              items={['런북 업로드 → 끊긴 경로·준비도 점수', '고정 규칙 35개 · LLM OFF 기본', '별빛상점 예제로 바로 체험']}
              cta={
                <Link to="/app" className="inline-flex h-12 w-full items-center justify-center rounded-xl border border-line-strong font-bold hover:bg-surface-2">
                  지금 검진하기
                </Link>
              }
            />
            <PriceCard
              highlight
              name="상세 진단"
              price="300만 원부터"
              items={['런북·명부·당직표·카탈로그 전수 대조', '모의 장애 리허설 + 수정안 워크숍', 'ISMS-P 2.11.4 · 2.11.5 · 2.12.2 증적 리포트']}
              cta={
                <a href="#signup" className="inline-flex h-12 w-full items-center justify-center rounded-xl bg-brand font-bold text-brand-fg hover:brightness-110">
                  사전 신청 (결제 없음)
                </a>
              }
            />
            <PriceCard
              name="팀 구독"
              price="출시 예정"
              items={['그래프 버전 관리 · 정기 리허설', 'GitHub Action으로 런북 PR마다 SARIF 검사', '설치형(사내 LLM·원문 비저장) 지원']}
              cta={
                <a href="#signup" className="inline-flex h-12 w-full items-center justify-center rounded-xl border border-line-strong font-bold hover:bg-surface-2">
                  소식 받기
                </a>
              }
            />
          </div>
        </section>

        <section id="signup" aria-label="상세 진단 사전 신청" className="scroll-mt-20 border-t border-line bg-surface/60">
          <div className="mx-auto grid max-w-[1240px] gap-10 px-5 py-24 lg:grid-cols-[0.9fr_1.1fr]">
            <div>
              <p className="text-[15px] font-bold text-brand">사전 신청</p>
              <h2 className="mt-2 text-[34px] leading-tight font-extrabold tracking-tight">다음 장애 전에, 우리 팀 런북부터</h2>
              <p className="mt-4 text-[17px] leading-relaxed text-fg-muted">
                상세 진단 일정과 범위를 안내드립니다. 신청만으로는 아무것도 결제되지 않으며, 입력하신 정보는 안내 목적으로만 씁니다.
              </p>
              <ul className="mt-6 space-y-2 text-[16px]">
                {['킥오프 1회 · 자료 수집 체크리스트 제공', '진단 결과 브리핑 + 수정안 승인 워크숍', '증적 리포트(Markdown·HTML·SARIF) 전달'].map((t) => (
                  <li key={t} className="flex gap-2">
                    <Check className="mt-1 size-4 shrink-0 text-ok" strokeWidth={3} aria-hidden />
                    {t}
                  </li>
                ))}
              </ul>
            </div>
            <SignupForm />
          </div>
        </section>
      </main>

      <footer className="border-t border-line">
        <div className="mx-auto flex max-w-[1240px] flex-wrap items-center gap-4 px-5 py-8 text-[14px] text-fg-subtle">
          <BrandWordmark />
          <p>© 2026 제때옵스 · 마포구 청년 창업 경진대회 출품작 · 예제의 회사·인물은 모두 가상입니다.</p>
          <div className="ml-auto flex items-center gap-4">
            <Link to="/privacy" className="font-semibold text-fg-muted hover:text-fg">
              개인정보 처리방침
            </Link>
            <Link to="/app" className="font-semibold text-fg-muted hover:text-fg">
              앱 열기 →
            </Link>
          </div>
        </div>
      </footer>
    </div>
  );
}

function PriceCard({ name, price, items, cta, highlight = false }: { name: string; price: string; items: string[]; cta: ReactNode; highlight?: boolean }) {
  return (
    <article className={clsx('card flex flex-col p-7', highlight && 'border-brand/60 ring-1 ring-brand/40')}>
      {highlight ? <span className="mb-3 self-start rounded-full bg-brand px-3 py-1 text-[12.5px] font-bold text-brand-fg">결선 시연 기준</span> : null}
      <h3 className="text-[19px] font-bold text-fg-muted">{name}</h3>
      <p className="mt-1 text-[32px] font-black tracking-tight">{price}</p>
      <ul className="mt-5 mb-7 flex-1 space-y-2.5">
        {items.map((t) => (
          <li key={t} className="flex gap-2 text-[15.5px]">
            <Check className="mt-1 size-4 shrink-0 text-ok" strokeWidth={3} aria-hidden />
            {t}
          </li>
        ))}
      </ul>
      {cta}
    </article>
  );
}
