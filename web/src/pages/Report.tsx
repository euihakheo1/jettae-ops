import { clsx } from 'clsx';
import { CircleAlert, CircleCheck, CircleDashed, Download, FileCode, FileText, Globe, ScrollText } from 'lucide-react';
import { Fragment, useMemo, useState, type ReactNode } from 'react';
import { useParams } from 'react-router';
import { api, saveBlob } from '../api/client';
import { errorMessage } from '../api/errors';
import { useReport, useReportHtml } from '../api/hooks';
import { useSourceMode } from '../api/source';
import { Button } from '../components/ui/Button';
import { ErrorState, LoadingBlock, Skeleton } from '../components/ui/States';
import { useToast } from '../components/ui/Toast';
import { CONTROL_STATUS_LABEL, controlRows, type ControlStatus } from '../lib/isms';
import { formatDateTime, versionChip } from '../lib/format';
import { PageHeader } from './PageHeader';
import { currentRehearsals } from '../lib/rehearsals';

const STATUS_STYLE: Record<ControlStatus, { cls: string; icon: typeof CircleCheck }> = {
  ok: { cls: 'border-ok/50 bg-ok/8 text-ok', icon: CircleCheck },
  gap: { cls: 'border-warn/50 bg-warn/8 text-warn', icon: CircleAlert },
  missing: { cls: 'border-line-strong bg-surface-2 text-fg-subtle', icon: CircleDashed },
};

const SHOWN_CONTROLS = ['2.11.4', '2.11.5', '2.12.2'];

/** 규칙 ID(JT-KR-001)는 하이픈에서 줄이 바뀌지 않게 묶는다 */
function NoBreakIds({ text }: { text: string }): ReactNode {
  const parts = text.split(/(JT-[A-Z]+(?:-\d{3})?)/g);
  return parts.map((p, i) =>
    /^JT-[A-Z]+/.test(p) ? (
      <span key={i} className="whitespace-nowrap">
        {p}
      </span>
    ) : (
      <Fragment key={i}>{p}</Fragment>
    ),
  );
}

/**
 * 미리보기 iframe은 앱의 글꼴을 물려받지 않는다 — 앱에 이미 번들된 Pretendard @font-face 규칙을 그대로 넣어
 * 리포트도 같은 글꼴로 보이게 한다 (오프라인 행사장에서도 CDN 없이).
 */
function withAppFonts(html: string): string {
  let css = '';
  try {
    for (const sheet of Array.from(document.styleSheets)) {
      let rules: CSSRuleList;
      try {
        rules = sheet.cssRules;
      } catch {
        continue;
      }
      for (const r of Array.from(rules)) if (r instanceof CSSFontFaceRule && r.style.getPropertyValue('font-family').includes('Pretendard')) css += r.cssText;
    }
  } catch {
    return html;
  }
  if (!css) return html;
  const tag = `<style>${css}</style>`;
  return /<head[^>]*>/i.test(html) ? html.replace(/<head[^>]*>/i, (m) => m + tag) : tag + html;
}

export function Report() {
  const { ws = '' } = useParams();
  const report = useReport(ws);
  const html = useReportHtml(ws);
  const mode = useSourceMode();
  const toast = useToast();
  const [busy, setBusy] = useState<string | null>(null);
  const srcDoc = useMemo(() => (html.data ? withAppFonts(html.data) : null), [html.data]);

  if (report.isError) return <ErrorState error={report.error} onRetry={() => void report.refetch()} />;
  if (!report.data) return <LoadingBlock className="p-8" />;
  const data = report.data;
  const controls = controlRows(data.findings, currentRehearsals(data), data.scenarios).filter((c) => SHOWN_CONTROLS.includes(c.id));

  const download = async (kind: 'md' | 'html' | 'sarif') => {
    setBusy(kind);
    try {
      const { filename, blob } = await api.reportFile(ws, kind);
      saveBlob(filename, blob);
      // 수요 측정 — 경로에 워크스페이스 id를 넣지 않는다(접근 키 헤더가 붙지 않고, id가 기록에 남지 않게)
      void api.track({ name: 'report_download', path: '/app/report', props: { kind } });
    } catch (e) {
      toast.show({ tone: 'error', title: '리포트를 내려받지 못했습니다', description: errorMessage(e) });
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="mx-auto max-w-[1440px] p-4 2xl:p-6">
      <PageHeader
        icon={ScrollText}
        title="증적 리포트"
        description="진단·리허설·승인 이력을 한 문서로 묶은 ISMS-P 증적 초안입니다. 인증 심사의 판단을 대신하지 않으며, 증적이 없는 항목은 만족으로 표시하지 않습니다."
        actions={
          <>
            <Button icon={<FileText className="size-4" aria-hidden />} loading={busy === 'md'} onClick={() => void download('md')}>
              Markdown
            </Button>
            <Button icon={<Globe className="size-4" aria-hidden />} loading={busy === 'html'} onClick={() => void download('html')}>
              HTML
            </Button>
            <Button icon={<FileCode className="size-4" aria-hidden />} loading={busy === 'sarif'} onClick={() => void download('sarif')}>
              SARIF
            </Button>
          </>
        }
      />

      <section aria-label="ISMS-P 매핑" className="mt-6 grid gap-4 lg:grid-cols-3">
        {controls.map((c) => {
          const st = STATUS_STYLE[c.status];
          const Icon = st.icon;
          return (
            <article key={c.id} className="card flex flex-col gap-3 p-5">
              <div className="flex items-center gap-2">
                <span className="rounded-lg bg-brand/14 px-2.5 py-1 font-mono text-[15px] font-extrabold text-brand">ISMS-P {c.id}</span>
                <span className={clsx('ml-auto inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-[13px] font-bold', st.cls)}>
                  <Icon className="size-4" aria-hidden />
                  {CONTROL_STATUS_LABEL[c.status]}
                </span>
              </div>
              <h2 className="text-[19px] font-extrabold">{c.title}</h2>
              <p className="text-[15px] leading-relaxed text-fg-muted">
                <NoBreakIds text={c.evidence} />
              </p>
              <p className="mt-auto text-[13px] text-fg-subtle">
                리포트 위치: <NoBreakIds text={c.section} />
              </p>
            </article>
          );
        })}
      </section>

      <section aria-label="리포트 미리보기" className="card mt-6 overflow-hidden">
        <div className="flex flex-wrap items-center gap-3 border-b border-line px-5 py-3">
          <Download className="size-4 text-fg-subtle" aria-hidden />
          <h2 className="text-[15px] font-bold">미리보기</h2>
          <span className="font-mono text-[13px] text-fg-subtle">
            {versionChip(data.graph_version, data.graph_hash)} · {formatDateTime(data.created_at)}
          </span>
          {mode !== 'api' ? <span className="ml-auto text-[13px] text-warn">데모 데이터로 브라우저에서 만든 미리보기</span> : null}
        </div>
        {html.isError ? (
          <ErrorState error={html.error} title="리포트 HTML을 불러오지 못했습니다" onRetry={() => void html.refetch()} />
        ) : srcDoc ? (
          <iframe
            title="증적 리포트 미리보기"
            srcDoc={srcDoc}
            sandbox="allow-same-origin"
            className="h-[880px] w-full border-0 bg-white"
          />
        ) : (
          <div className="space-y-3 p-6">
            <Skeleton className="h-8 w-1/2" />
            <Skeleton className="h-4 w-3/4" />
            <Skeleton className="h-96 w-full" />
          </div>
        )}
      </section>
    </div>
  );
}
