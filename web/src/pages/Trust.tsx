import { clsx } from 'clsx';
import { BotOff, Bot, CircleCheck, Lock, Scale, Search, ShieldAlert, ShieldCheck, Sparkles } from 'lucide-react';
import { useMemo, useState } from 'react';
import { useParams } from 'react-router';
import { useDocuments, usePatches, useReport, useRules } from '../api/hooks';
import type { Severity, SourceDocument } from '../api/types';
import { Badge, SeverityChip } from '../components/ui/Badge';
import { ErrorState, LoadingBlock } from '../components/ui/States';
import { formatNumber } from '../lib/format';
import { ELEMENT_LABEL, INJECTION_PATTERN_LABEL, LLM_MODE_LABEL, ruleFamily, SEVERITY_LABEL, SEVERITY_ORDER } from '../lib/labels';
import { PageHeader } from './PageHeader';

function lineOf(doc: SourceDocument | undefined, offset: number): number | null {
  if (!doc) return null;
  let line = 1;
  for (let i = 0; i < Math.min(offset, doc.text.length); i += 1) if (doc.text[i] === '\n') line += 1;
  return line;
}

const TASK_LABEL: Record<string, string> = {
  extract: '추출',
  verify: '역번역 대조',
  scenarios: '시나리오 생성',
  draft: '수정안 문장',
};

export function Trust() {
  const { ws = '' } = useParams();
  const report = useReport(ws);
  const rules = useRules();
  const docs = useDocuments(ws);
  const patches = usePatches(ws);
  const [q, setQ] = useState('');
  const [sev, setSev] = useState<Severity | ''>('');

  const ruleCounts = useMemo(() => {
    const m = new Map<string, number>();
    for (const f of report.data?.findings ?? []) m.set(f.rule_id, (m.get(f.rule_id) ?? 0) + 1);
    return m;
  }, [report.data]);

  if (report.isError) return <ErrorState error={report.error} onRetry={() => void report.refetch()} />;
  if (!report.data) return <LoadingBlock className="p-8" />;
  const data = report.data;
  const g = data.grounding;
  const llm = data.llm;
  const aiPatches = (patches.data ?? data.patches).filter((p) => p.author === 'ai');
  const draftCalls = aiPatches.reduce((a, p) => a + (p.llm_usage.calls ?? 0), 0);
  const draftTokens = aiPatches.reduce((a, p) => a + (p.llm_usage.input_tokens ?? 0) + (p.llm_usage.output_tokens ?? 0), 0);
  const filteredRules = (rules.data ?? []).filter(
    (r) => (!sev || r.severity === sev) && (!q || `${r.rule_id} ${r.title} ${r.rationale}`.toLowerCase().includes(q.toLowerCase())),
  );

  return (
    <div className="mx-auto max-w-[1440px] p-4 2xl:p-6">
      <PageHeader
        icon={ShieldCheck}
        title="보안 · AI 사용"
        description="문서는 데이터이지 지시가 아닙니다. 판정·점수·리허설은 고정 규칙과 결정적 리플레이로만 계산하고, 그래프의 모든 요소는 원문 위치나 명부·카탈로그 같은 구조화 출처를 가집니다."
      />

      <div className="mt-6 grid gap-4 xl:grid-cols-[1.1fr_1fr]">
        <section className="card relative overflow-hidden p-6" aria-label="판정 경로 LLM 호출">
          <div className="absolute -top-16 -right-16 size-56 rounded-full bg-ok/10 blur-2xl" aria-hidden />
          <p className="flex items-center gap-2 text-[14px] font-bold text-ok">
            <BotOff className="size-5" aria-hidden />
            판정 경로 AI 차단
          </p>
          <p className="mt-3 text-[28px] leading-tight font-extrabold 2xl:text-[32px]">
            판정·점수·리허설 경로
            <br />
            LLM 호출 <span className="tnum text-ok">{llm.judgement_path_calls}회</span>
          </p>
          <p className="mt-3 max-w-xl text-[15px] leading-relaxed text-fg-muted">
            끊김 판정(고정 규칙 {rules.data?.length ?? '—'}개), 준비도 점수(공개 가중치 {data.readiness.weights_version}), 리허설 재생(결정적 리플레이)은 코드 계약으로 LLM 모듈을 가져올 수 없습니다. AI는 추출 후보와 수정안 문장만 제안합니다.
          </p>
        </section>

        <section className="card p-6" aria-label="LLM 사용량">
          <div className="flex items-center gap-2">
            <h2 className="text-[16px] font-bold">LLM 사용량</h2>
            <Badge tone={llm.mode === 'off' ? 'ok' : 'brand'} icon={llm.mode === 'off' ? BotOff : Bot} size="lg" className="ml-auto">
              {LLM_MODE_LABEL[llm.mode]}
            </Badge>
          </div>
          <dl className="mt-4 grid grid-cols-2 gap-3 text-[14px] sm:grid-cols-4">
            <Stat label="공급자·모델" value={llm.provider ? `${llm.provider} · ${llm.model ?? ''}` : '사용 안 함'} small />
            <Stat label="진단 호출" value={`${llm.calls}회`} />
            <Stat label="입력 토큰" value={formatNumber(llm.input_tokens)} />
            <Stat label="출력 토큰" value={formatNumber(llm.output_tokens)} />
          </dl>
          <table className="mt-4 w-full text-left text-[14px]">
            <thead className="text-[12.5px] text-fg-subtle">
              <tr>
                <th className="border-b border-line py-1.5 font-semibold">작업</th>
                <th className="border-b border-line py-1.5 text-right font-semibold">호출</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(llm.by_task).map(([task, n]) => (
                <tr key={task}>
                  <td className="border-b border-line py-1.5">{TASK_LABEL[task] ?? task}</td>
                  <td className="tnum border-b border-line py-1.5 text-right">{n}</td>
                </tr>
              ))}
              <tr>
                <td className="border-b border-line py-1.5">
                  <span className="inline-flex items-center gap-1.5">
                    <Sparkles className="size-4 text-brand" aria-hidden />
                    수정안 문장 초안 (AI {aiPatches.length}건, 그래프 op는 코드가 생성)
                  </span>
                </td>
                <td className="tnum border-b border-line py-1.5 text-right">
                  {draftCalls}회 · {formatNumber(draftTokens)} 토큰
                </td>
              </tr>
              <tr className="font-bold text-ok">
                <td className="py-1.5">판정·점수·리허설</td>
                <td className="tnum py-1.5 text-right">{llm.judgement_path_calls}회</td>
              </tr>
            </tbody>
          </table>
        </section>
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-[1fr_1fr]">
        <section className="card p-6" aria-label="격리된 문서 속 지시문">
          <h2 className="flex items-center gap-2 text-[16px] font-bold">
            <ShieldAlert className="size-5 text-warn" aria-hidden />
            격리된 문서 속 지시문 {data.security.length}건
          </h2>
          <p className="mt-1 text-[14px] text-fg-muted">AI 검토 도구를 향한 지시처럼 보이는 구간은 추출에서 빼고 기록만 남깁니다. 판정·점수에는 영향이 없습니다.</p>
          {data.security.length === 0 ? (
            <p className="mt-4 inline-flex items-center gap-2 text-ok">
              <CircleCheck className="size-5" aria-hidden /> 발견된 지시문이 없습니다
            </p>
          ) : (
            <ul className="mt-4 space-y-3">
              {data.security.map((s) => {
                const doc = docs.data?.find((d) => d.doc_id === s.doc_id);
                const line = lineOf(doc, s.start);
                return (
                  <li key={`${s.doc_id}:${s.start}`} className="rounded-xl border border-warn/40 bg-warn/6 p-4">
                    <div className="flex flex-wrap items-center gap-2 text-[13.5px]">
                      <Badge tone="warn" icon={Lock}>
                        격리됨 · 따르지 않음
                      </Badge>
                      <span className="font-semibold">{doc?.title ?? s.doc_id}</span>
                      <span className="tnum font-mono text-fg-subtle">: {line ?? '?'}줄</span>
                      <span className="ml-auto text-fg-muted">{INJECTION_PATTERN_LABEL[s.pattern] ?? s.pattern}</span>
                    </div>
                    <pre className="mt-2 overflow-x-auto rounded-lg border border-line bg-surface-2 p-3 font-mono text-[13px] whitespace-pre-wrap text-fg-muted">{s.excerpt}</pre>
                  </li>
                );
              })}
            </ul>
          )}
          <ol className="mt-5 grid gap-2 border-t border-line pt-4 text-[14px] text-fg-muted sm:grid-cols-3">
            {[
              ['문서는 데이터', '런북 속 문장은 읽기만 하고 실행하지 않습니다'],
              ['지시문은 격리', 'AI를 향한 구간은 추출 입력에서 뺍니다'],
              ['기록은 남김', '격리 위치를 리포트·보안 화면에 보여 줍니다'],
            ].map(([t, b], i) => (
              <li key={t} className="rounded-xl bg-surface-2 p-3">
                <p className="flex items-center gap-1.5 font-bold text-fg">
                  <span className="tnum grid size-5 place-items-center rounded-full bg-warn/20 text-[12px] text-warn">{i + 1}</span>
                  {t}
                </p>
                <p className="mt-1 leading-snug">{b}</p>
              </li>
            ))}
          </ol>
        </section>

        <section className="card p-6" aria-label="원문 근거 정렬">
          <h2 className="text-[16px] font-bold">원문 근거 정렬 (지어내기 방지)</h2>
          <p className="mt-1 text-[14px] text-fg-muted">추출한 모든 값은 원문 위치에 다시 맞춰 봅니다. 맞지 않으면 버리고, 애매하면 판단 보류로 사람에게 넘깁니다.</p>
          <div className="mt-4 grid grid-cols-2 gap-3">
            <Tile label="정확 정렬" value={g.grounded} tone="ok" hint="원문과 글자 그대로 일치" />
            <Tile label="퍼지 정렬" value={g.fuzzy_grounded} tone="info" hint="공백·표기 차이만 허용 (≥90)" />
            <Tile label="폐기" value={g.dropped_ungrounded + g.dropped_quarantined + g.dropped_invalid} tone="danger" hint={`근거 없음 ${g.dropped_ungrounded} · 격리 구간 ${g.dropped_quarantined} · 형식 오류 ${g.dropped_invalid}`} />
            <Tile label="판단 보류" value={g.held_backtranslation + g.held_ambiguous_entity} tone="warn" hint={`역번역 불일치 ${g.held_backtranslation} · 동명이인 ${g.held_ambiguous_entity}`} />
          </div>
          {g.dropped_examples.length ? (
            <ul className="mt-3 space-y-1 text-[13px] text-fg-muted">
              {g.dropped_examples.map((x) => (
                <li key={x}>· {x}</li>
              ))}
            </ul>
          ) : null}
        </section>
      </div>

      <section className="card mt-4 overflow-hidden" aria-label="고정 규칙 카탈로그">
        <div className="flex flex-wrap items-center gap-3 border-b border-line px-5 py-4">
          <Scale className="size-5 text-brand" aria-hidden />
          <h2 className="text-[16px] font-bold">고정 규칙 카탈로그 {rules.data?.length ?? ''}개</h2>
          <span className="text-[13px] text-fg-subtle">고객이 판정 기준을 직접 감사할 수 있도록 공개합니다</span>
          <div className="ml-auto flex items-center gap-2">
            <label className="relative">
              <span className="sr-only">규칙 검색</span>
              <Search className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-fg-subtle" aria-hidden />
              <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="규칙 검색" className="h-9 w-56 rounded-lg border border-line bg-surface-2 pr-3 pl-8 text-[14px] placeholder:text-fg-subtle" />
            </label>
            <select aria-label="심각도" value={sev} onChange={(e) => setSev(e.target.value as Severity | '')} className="h-9 rounded-lg border border-line bg-surface-2 px-2 text-[14px]">
              <option value="">모든 심각도</option>
              {SEVERITY_ORDER.map((s) => (
                <option key={s} value={s}>
                  {SEVERITY_LABEL[s]}
                </option>
              ))}
            </select>
          </div>
        </div>
        {rules.isError ? (
          <ErrorState error={rules.error} />
        ) : !rules.data ? (
          <LoadingBlock />
        ) : (
          <div className="max-h-[640px] overflow-auto">
            <table className="w-full border-separate border-spacing-0 text-left text-[14px]">
              <thead className="sticky top-0 bg-surface text-[12.5px] text-fg-subtle">
                <tr>
                  <th className="border-b border-line px-4 py-2 font-semibold">규칙</th>
                  <th className="border-b border-line px-3 py-2 font-semibold">심각도</th>
                  <th className="border-b border-line px-3 py-2 font-semibold">계열 · 요소</th>
                  <th className="border-b border-line px-3 py-2 font-semibold">내용</th>
                  <th className="border-b border-line px-3 py-2 text-right font-semibold whitespace-nowrap">이번 진단</th>
                </tr>
              </thead>
              <tbody>
                {filteredRules.map((r) => {
                  const n = ruleCounts.get(r.rule_id) ?? 0;
                  return (
                    <tr key={r.rule_id} className="align-top hover:bg-surface-2">
                      <td className="border-b border-line px-4 py-3 font-mono text-[13px] font-semibold whitespace-nowrap">{r.rule_id}</td>
                      <td className="border-b border-line px-3 py-3">
                        <SeverityChip severity={r.severity} size="sm" />
                      </td>
                      <td className="border-b border-line px-3 py-3 whitespace-nowrap text-fg-muted">
                        {ruleFamily(r.rule_id)}
                        {r.element ? ` · ${ELEMENT_LABEL[r.element]}` : ''}
                      </td>
                      <td className="border-b border-line px-3 py-3">
                        <p className="font-semibold">{r.title}</p>
                        <p className="mt-0.5 text-[13px] leading-relaxed text-fg-muted">{r.rationale}</p>
                        {r.legal_basis ? (
                          <p className="mt-1 inline-flex items-start gap-1 text-[12.5px] text-brand">
                            <Scale className="mt-0.5 size-3.5 shrink-0" aria-hidden />
                            {r.legal_basis}
                          </p>
                        ) : null}
                      </td>
                      <td className={clsx('tnum border-b border-line px-3 py-3 text-right font-bold', n ? 'text-danger' : 'text-fg-subtle')}>{n ? `${n}건` : '—'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

function Stat({ label, value, small = false }: { label: string; value: string; small?: boolean }) {
  return (
    <div className="rounded-xl border border-line bg-surface-2 p-3">
      <dt className="text-[12px] text-fg-subtle">{label}</dt>
      <dd className={clsx('tnum mt-0.5 font-extrabold', small ? 'text-[14px]' : 'text-[20px]')}>{value}</dd>
    </div>
  );
}

function Tile({ label, value, tone, hint }: { label: string; value: number; tone: 'ok' | 'info' | 'danger' | 'warn'; hint: string }) {
  return (
    <div className="rounded-xl border border-line bg-surface-2 p-4">
      <p className="text-[13px] font-semibold text-fg-subtle">{label}</p>
      <p className={clsx('tnum mt-1 text-[30px] leading-none font-extrabold', tone === 'ok' && 'text-ok', tone === 'info' && 'text-info', tone === 'danger' && 'text-danger', tone === 'warn' && 'text-warn')}>
        {value}
      </p>
      <p className="mt-1.5 text-[12.5px] text-fg-muted">{hint}</p>
    </div>
  );
}
