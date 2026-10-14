import { CircleCheck, Send } from 'lucide-react';
import { useId, useState, type FormEvent } from 'react';
import { Link } from 'react-router';
import { api } from '../../api/client';
import { errorMessage } from '../../api/errors';
import { Button } from '../../components/ui/Button';

const TEAM_SIZES = ['1~10명', '11~50명', '51~200명', '201~1,000명', '1,000명 이상'];
const EMAIL = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;

interface Errors {
  email?: string;
  org?: string;
  consent?: string;
  submit?: string;
}

/** 상세 진단 사전 신청 (결제 없음) — POST /api/v1/signups */
export function SignupForm() {
  const id = useId();
  const [email, setEmail] = useState('');
  const [org, setOrg] = useState('');
  const [size, setSize] = useState(TEAM_SIZES[1]!);
  const [message, setMessage] = useState('');
  const [consent, setConsent] = useState(false);
  const [marketing, setMarketing] = useState(false);
  const [errors, setErrors] = useState<Errors>({});
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState<string | null>(null);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    void api.track({ name: 'signup_click', path: '/', props: { team_size: size, marketing } });
    const next: Errors = {};
    if (!EMAIL.test(email.trim())) next.email = '올바른 이메일 주소를 입력하세요.';
    if (!org.trim()) next.org = '조직명을 입력하세요.';
    if (!consent) next.consent = '개인정보 수집·이용에 동의해야 신청할 수 있습니다.';
    setErrors(next);
    if (Object.keys(next).length) return;
    setBusy(true);
    try {
      await api.signup({
        email: email.trim(),
        org_name: org.trim(),
        team_size: size,
        message: message.trim(),
        consent_privacy: true,
        consent_marketing: marketing,
        source: 'landing',
      });
      setDone(email.trim());
    } catch (err) {
      setErrors({ submit: errorMessage(err) });
    } finally {
      setBusy(false);
    }
  };

  if (done) {
    return (
      <div role="status" className="card flex flex-col items-center gap-3 p-10 text-center">
        <span className="grid size-14 place-items-center rounded-full bg-ok/15 text-ok">
          <CircleCheck className="size-8" aria-hidden />
        </span>
        <h3 className="text-[24px] font-extrabold">사전 신청이 접수되었습니다</h3>
        <p className="max-w-md text-[16px] text-fg-muted">
          <strong className="text-fg">{done}</strong> 로 상세 진단 일정을 안내드리겠습니다. 결제는 진행되지 않았습니다.
        </p>
      </div>
    );
  }

  const field = 'mt-1.5 h-12 w-full rounded-xl border bg-surface-2 px-3.5 text-[16px] placeholder:text-fg-subtle';
  return (
    <form onSubmit={(e) => void submit(e)} noValidate className="card space-y-4 p-6 2xl:p-8" aria-labelledby={`${id}-title`}>
      <h3 id={`${id}-title`} className="text-[22px] font-extrabold">
        상세 진단 사전 신청 <span className="text-[15px] font-semibold text-ok">(결제 없음)</span>
      </h3>
      <div className="grid gap-4 sm:grid-cols-2">
        <label className="block">
          <span className="text-[14px] font-semibold">업무용 이메일 *</span>
          <input
            type="email"
            autoComplete="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            aria-invalid={!!errors.email}
            aria-describedby={errors.email ? `${id}-email` : undefined}
            placeholder="sre@company.example"
            className={`${field} ${errors.email ? 'border-danger' : 'border-line'}`}
          />
          {errors.email ? (
            <span id={`${id}-email`} className="mt-1 block text-[13px] text-danger">
              {errors.email}
            </span>
          ) : null}
        </label>
        <label className="block">
          <span className="text-[14px] font-semibold">조직명 *</span>
          <input
            value={org}
            onChange={(e) => setOrg(e.target.value)}
            autoComplete="organization"
            aria-invalid={!!errors.org}
            aria-describedby={errors.org ? `${id}-org` : undefined}
            placeholder="예: 별빛상점"
            className={`${field} ${errors.org ? 'border-danger' : 'border-line'}`}
          />
          {errors.org ? (
            <span id={`${id}-org`} className="mt-1 block text-[13px] text-danger">
              {errors.org}
            </span>
          ) : null}
        </label>
      </div>
      <label className="block">
        <span className="text-[14px] font-semibold">팀 규모</span>
        <select value={size} onChange={(e) => setSize(e.target.value)} className={`${field} border-line`}>
          {TEAM_SIZES.map((s) => (
            <option key={s}>{s}</option>
          ))}
        </select>
      </label>
      <label className="block">
        <span className="text-[14px] font-semibold">남기실 말씀 (선택)</span>
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          rows={3}
          placeholder="런북 형식(위키·노션·PDF), 당직 도구, ISMS-P 일정 등"
          className="mt-1.5 w-full rounded-xl border border-line bg-surface-2 p-3.5 text-[16px] placeholder:text-fg-subtle"
        />
      </label>
      <div className="rounded-xl border border-line bg-surface-2 p-4">
        <label className="flex items-start gap-3">
          <input
            type="checkbox"
            checked={consent}
            onChange={(e) => setConsent(e.target.checked)}
            aria-invalid={!!errors.consent}
            aria-describedby={`${id}-privacy`}
            className="mt-1 size-5 accent-[var(--brand)]"
          />
          <span className="text-[15px] font-semibold">
            개인정보 수집·이용에 동의합니다 <span className="text-danger">(필수)</span>
          </span>
        </label>
        <p id={`${id}-privacy`} className="mt-2 pl-8 text-[13px] leading-relaxed text-fg-muted">
          수집 항목: 이메일, 조직명, 팀 규모, 남기신 말씀 · 목적: 상세 진단 사전 신청 확인과 일정 안내 · 보유 기간: 신청일로부터 1년(동의를 철회하면 지체 없이
          파기) · 국외 이전: Render Services, Inc.(미국)의 클라우드에 맡겨 싱가포르 서버에 보관 · 동의를 거부할 수 있으나, 거부하면 사전 신청을 할 수 없습니다.{' '}
          <Link to="/privacy" target="_blank" rel="noopener noreferrer" className="font-semibold text-fg underline underline-offset-2 hover:text-brand">
            개인정보 처리방침
          </Link>
        </p>
        {errors.consent ? <p className="mt-1 pl-8 text-[13px] text-danger">{errors.consent}</p> : null}
        <label className="mt-3 flex items-start gap-3 border-t border-line pt-3">
          <input
            type="checkbox"
            checked={marketing}
            onChange={(e) => setMarketing(e.target.checked)}
            aria-describedby={`${id}-marketing`}
            className="mt-1 size-5 accent-[var(--brand)]"
          />
          <span className="text-[15px]">
            새 기능·사례 소식 받기 <span className="text-fg-subtle">(선택 · 마케팅 정보 수신 동의)</span>
          </span>
        </label>
        <p id={`${id}-marketing`} className="mt-2 pl-8 text-[13px] leading-relaxed text-fg-muted">
          수집 항목: 이메일 · 목적: 새 기능·사례 소식 안내 · 보유 기간: 동의 철회 시 또는 신청일로부터 1년 중 먼저 오는 때까지 · 동의하지 않아도 사전 신청할 수
          있습니다.
        </p>
      </div>
      {errors.submit ? (
        <p role="alert" className="rounded-xl border border-danger/50 bg-danger/10 px-4 py-3 text-[14px] text-danger">
          {errors.submit}
        </p>
      ) : null}
      <Button type="submit" variant="primary" size="lg" className="w-full" loading={busy} icon={<Send className="size-5" aria-hidden />}>
        사전 신청하기
      </Button>
    </form>
  );
}
