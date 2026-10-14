import { ArrowLeft, BarChart3, Bot, FileSearch, Globe, Mail, Moon, Projector, Send, UserCheck, type LucideIcon } from 'lucide-react';
import { useEffect, type ReactNode } from 'react';
import { Link } from 'react-router';
import { themeStore, useTheme } from '../app/prefs';
import { BrandWordmark } from '../components/shell/Brand';

/** 처리방침 시행일 — 내용을 바꾸면 날짜도 바꾼다 */
export const PRIVACY_EFFECTIVE_DATE = '2026-10-15';
const OPERATOR = '제때옵스(예비창업자 허의학)';
const CONTACT_EMAIL = 'euihakeg@naver.com';

/** 항목 한 줄: 이름 → 내용 */
function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="grid gap-1 border-t border-line py-3 first:border-t-0 first:pt-0 last:pb-0 sm:grid-cols-[140px_minmax(0,1fr)] sm:gap-4">
      <dt className="text-[14px] font-bold text-fg-subtle">{label}</dt>
      <dd className="text-[16px] leading-relaxed">{children}</dd>
    </div>
  );
}

function Section({ id, icon: Icon, title, children }: { id: string; icon: LucideIcon; title: string; children: ReactNode }) {
  return (
    <section aria-labelledby={id} className="card p-6 sm:p-7">
      <h2 id={id} className="flex items-center gap-2.5 text-[21px] font-extrabold">
        <span className="grid size-10 shrink-0 place-items-center rounded-xl bg-brand/14 text-brand">
          <Icon className="size-5" aria-hidden />
        </span>
        {title}
      </h2>
      <div className="mt-5">{children}</div>
    </section>
  );
}

function SubHeading({ children }: { children: ReactNode }) {
  return <h3 className="mt-6 mb-3 text-[16px] font-extrabold text-fg-muted first:mt-0">{children}</h3>;
}

/** 개인정보 처리방침 (/privacy) — 공개 체험판이 받는 정보·목적·보유 기간·국외 이전·권리 */
export function Privacy() {
  const theme = useTheme();

  useEffect(() => {
    const previous = document.title;
    document.title = '개인정보 처리방침 · 제때옵스';
    return () => {
      document.title = previous;
    };
  }, []);

  return (
    <div className="min-h-full overflow-x-hidden bg-bg">
      <header className="sticky top-0 z-40 border-b border-line/70 bg-bg/80 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-[1240px] items-center gap-6 px-5">
          <Link to="/" aria-label="제때옵스 홈">
            <BrandWordmark />
          </Link>
          <div className="ml-auto flex items-center gap-2">
            <button
              type="button"
              onClick={() => themeStore.set(theme === 'dark' ? 'projector' : 'dark')}
              aria-label={theme === 'dark' ? '프로젝터 모드(밝은 화면)로 전환' : '어두운 화면으로 전환'}
              className="grid size-10 place-items-center rounded-xl border border-line text-fg-muted hover:text-fg"
            >
              {theme === 'dark' ? <Projector className="size-4" aria-hidden /> : <Moon className="size-4" aria-hidden />}
            </button>
            <Link to="/" className="inline-flex h-10 items-center gap-1.5 rounded-xl border border-line px-4 text-[15px] font-semibold text-fg-muted hover:text-fg">
              <ArrowLeft className="size-4" aria-hidden />
              소개 페이지
            </Link>
          </div>
        </div>
      </header>

      <main className="relative">
        <div aria-hidden className="pointer-events-none absolute inset-x-0 -top-40 h-[420px] bg-[radial-gradient(60%_60%_at_50%_0%,color-mix(in_oklab,var(--brand)_16%,transparent),transparent)]" />
        <div className="relative mx-auto max-w-[880px] px-5 pt-14 pb-24">
          <p className="text-[15px] font-bold text-brand">공개 체험판</p>
          <h1 className="mt-2 text-[34px] leading-tight font-extrabold tracking-tight sm:text-[40px]">개인정보 처리방침</h1>
          <p className="mt-4 text-[17px] leading-relaxed text-fg-muted">
            {OPERATOR}가 제때옵스 공개 체험판에서 어떤 정보를 무엇 때문에 받고, 언제 지우는지 알려 드립니다.
          </p>
          <p className="mt-2 text-[14px] text-fg-subtle">시행일 {PRIVACY_EFFECTIVE_DATE}</p>

          <div className="mt-10 space-y-5">
            <Section id="privacy-operator" icon={Mail} title="운영자와 문의">
              <dl>
                <Row label="운영자">{OPERATOR}</Row>
                <Row label="문의">
                  <a href={`mailto:${CONTACT_EMAIL}`} className="font-semibold text-brand hover:underline">
                    {CONTACT_EMAIL}
                  </a>
                </Row>
              </dl>
            </Section>

            <Section id="privacy-items" icon={FileSearch} title="수집 항목과 목적, 보유 기간">
              <SubHeading>(1) 상세 진단 사전 신청</SubHeading>
              <dl>
                <Row label="수집 항목">이메일 · 조직명 · 팀 규모 · 남기신 말씀</Row>
                <Row label="목적">사전 신청 확인과 일정 안내</Row>
                <Row label="보유 기간">신청일로부터 1년 또는 동의 철회 시 지체 없이 파기</Row>
              </dl>
              <SubHeading>마케팅 정보 수신 동의 (선택)</SubHeading>
              <dl>
                <Row label="수집 항목">이메일</Row>
                <Row label="목적">새 기능·사례 소식 안내</Row>
                <Row label="보유 기간">동의 철회 시 또는 신청일로부터 1년 중 먼저 오는 때까지</Row>
              </dl>
              <SubHeading>(2) 무료 런북 검진</SubHeading>
              <dl>
                <Row label="수집 항목">
                  올린 파일에서 만든 운영 그래프·진단 결과
                  <span className="block text-[14.5px] text-fg-muted">
                    원문 파일은 저장하지 않습니다. 런북 본문과 카탈로그 설명의 연락처·주민번호 등은 자동으로 가립니다.
                  </span>
                  <span className="block text-[14.5px] text-fg-muted">
                    인사 명부의 이름·역할·재직 상태·퇴사일과 연락 수단의 종류(번호·주소는 저장하지 않음), 당직표의 이름·시간은 진단에 그대로 쓰이고, 직급·팀·별칭과 카탈로그의 담당·위치는 가린 뒤 저장되며, 모두 결과와 함께 7일 뒤 삭제됩니다.
                  </span>
                </Row>
                <Row label="목적">검진 결과 제공</Row>
                <Row label="보유 기간">7일 뒤 자동 삭제</Row>
                <Row label="열람">
                  결과는 접근 키가 있는 브라우저에서만 열립니다.
                  <span className="block text-[14.5px] text-fg-muted">운영자는 장애 대응·악용 방지 목적에 한해 열람할 수 있습니다.</span>
                </Row>
              </dl>
              <SubHeading>(3) 이용 통계</SubHeading>
              <dl>
                <Row label="수집 항목">
                  <span className="inline-flex items-center gap-1.5">
                    <BarChart3 className="size-4 text-fg-subtle" aria-hidden />
                    방문·클릭 횟수
                  </span>
                  <span className="block text-[14.5px] text-fg-muted">
                    브라우저 단위로 세며, 개인을 식별하는 정보(이름·이메일·IP 주소 등)는 담지 않습니다.
                  </span>
                </Row>
                <Row label="목적">수요 측정(서비스를 얼마나 쓰는지 파악)</Row>
                <Row label="보유 기간">서비스 운영 기간 동안(개인을 식별할 수 없는 횟수 정보)</Row>
              </dl>
              <SubHeading>(4) 서버 접속 기록</SubHeading>
              <dl>
                <Row label="수집 항목">
                  요청 경로·응답 상태·처리 시간
                  <span className="block text-[14.5px] text-fg-muted">IP 주소와 이메일은 접속 기록에 남기지 않습니다.</span>
                </Row>
                <Row label="목적">장애 대응과 악용 방지</Row>
                <Row label="보유 기간">호스팅(Render)의 로그 보관 기간</Row>
              </dl>
            </Section>

            <Section id="privacy-transfer" icon={Globe} title="처리 위탁과 국외 이전">
              <dl>
                <Row label="받는 곳">Render Services, Inc.(미국)의 클라우드</Row>
                <Row label="서버 위치">싱가포르</Row>
                <Row label="이전 항목">위 (1)·(2)의 정보</Row>
                <Row label="이전 시기·방법">서비스 이용 시 네트워크로 전송</Row>
                <Row label="보유 기간">위 보유 기간과 같음</Row>
              </dl>
              <p className="mt-4 rounded-xl border border-line bg-surface-2 px-4 py-3 text-[15px] leading-relaxed text-fg-muted">
                국외 이전을 원하지 않으면 사전 신청·검진을 이용하지 않을 수 있습니다.
              </p>
            </Section>

            <Section id="privacy-rights" icon={UserCheck} title="정보주체의 권리">
              <p className="text-[16px] leading-relaxed">
                열람·정정·삭제·처리정지·동의 철회는{' '}
                <a href={`mailto:${CONTACT_EMAIL}`} className="font-semibold text-brand hover:underline">
                  {CONTACT_EMAIL}
                </a>
                로 요청해 주세요. 본인 확인 후 10일 이내에 처리합니다.
              </p>
            </Section>

            <Section id="privacy-ai" icon={Bot} title="생성형 AI">
              <p className="text-[16px] leading-relaxed">
                공개 체험판에서는 생성형 AI(LLM)를 쓰지 않으며, 업로드한 내용이 외부 AI로 전송되지 않습니다.
              </p>
            </Section>
          </div>

          <div className="mt-10 flex flex-wrap gap-3">
            {/* 해시 위치로 스크롤되도록 문서 이동(라우터 링크는 해시로 스크롤하지 않는다) */}
            <a href="/#signup" className="inline-flex h-12 items-center gap-2 rounded-xl bg-brand px-5 font-bold text-brand-fg hover:brightness-110">
              <Send className="size-4" aria-hidden />
              사전 신청으로 돌아가기
            </a>
            <Link to="/app" className="inline-flex h-12 items-center rounded-xl border border-line-strong px-5 font-bold hover:bg-surface-2">
              무료 런북 건강검진
            </Link>
          </div>
        </div>
      </main>

      <footer className="border-t border-line">
        <div className="mx-auto flex max-w-[1240px] flex-wrap items-center gap-4 px-5 py-8 text-[14px] text-fg-subtle">
          <BrandWordmark />
          <p>
            © 2026 제때옵스 · 개인정보 처리방침 시행일 {PRIVACY_EFFECTIVE_DATE} · 문의 {CONTACT_EMAIL}
          </p>
        </div>
      </footer>
    </div>
  );
}
