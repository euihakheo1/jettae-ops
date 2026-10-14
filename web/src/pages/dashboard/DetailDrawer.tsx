import { clsx } from 'clsx';
import { ArrowRight, ChevronDown, CircleCheck, FileQuestion, GitPullRequestArrow, Info, Scale, X } from 'lucide-react';
import { useState, type ReactNode } from 'react';
import { Link } from 'react-router';
import type { DiagnosisReport, Finding, GraphView, GraphViewEdge, GraphViewNode, Patch, SourceDocument, SourceSpan } from '../../api/types';
import { EvidenceQuote } from '../../components/domain/EvidenceQuote';
import { KIND_ICON } from '../../components/domain/icons';
import { Badge, SeverityChip } from '../../components/ui/Badge';
import { attrLabel, displayName, formatAttr, nameFromId } from '../../lib/format';
import { EDGE_KIND_LABEL, ELEMENT_LABEL, FINDING_STATUS_LABEL, NODE_KIND_LABEL, PERSON_STATUS_LABEL, ruleFamily, SOURCE_LABEL } from '../../lib/labels';
import { groupSpans } from '../../lib/quote';
import { edgeLabel } from '../../graph/model';

export type Selection = { type: 'finding' | 'node' | 'edge'; id: string } | null;

interface Ctx {
  report: DiagnosisReport;
  view: GraphView;
  docs: readonly SourceDocument[];
  patches: readonly Patch[];
  ws: string;
  onSelect: (s: Selection) => void;
  onFocusService: (sid: string) => void;
}

export function DetailDrawer({ selection, onClose, ...ctx }: Ctx & { selection: Selection; onClose: () => void }) {
  if (!selection) return null;
  let body: ReactNode = null;
  if (selection.type === 'finding') {
    const f = ctx.report.findings.find((x) => x.id === selection.id);
    body = f ? <FindingDetail f={f} {...ctx} /> : <Missing what="끊김" />;
  } else if (selection.type === 'node') {
    const n = ctx.view.nodes.find((x) => x.id === selection.id);
    body = n ? <NodeDetail n={n} {...ctx} /> : <Missing what="요소" />;
  } else {
    const e = ctx.view.edges.find((x) => x.id === selection.id);
    body = e ? <EdgeDetail e={e} {...ctx} /> : <Missing what="연결" />;
  }
  return (
    <aside
      aria-label="상세 정보"
      className="absolute top-3 right-3 bottom-3 z-20 flex w-[min(440px,calc(100%-24px))] animate-slide-in flex-col overflow-hidden rounded-2xl border border-line-strong bg-surface shadow-2xl"
    >
      <div className="flex items-center justify-between border-b border-line px-4 py-2.5">
        <span className="text-[13px] font-semibold text-fg-subtle">
          {selection.type === 'finding' ? '끊김 상세' : selection.type === 'node' ? '요소 상세' : '연결 상세'}
        </span>
        <button type="button" onClick={onClose} aria-label="상세 닫기" className="grid size-8 place-items-center rounded-lg text-fg-subtle hover:bg-surface-2 hover:text-fg">
          <X className="size-5" aria-hidden />
        </button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">{body}</div>
    </aside>
  );
}

function Missing({ what }: { what: string }) {
  return <p className="text-fg-muted">선택한 {what}을(를) 이 버전에서 찾을 수 없습니다. 수정안 승인으로 해결됐을 수 있습니다.</p>;
}

function Section({ title, children, icon: Icon }: { title: string; children: ReactNode; icon?: typeof Info }) {
  return (
    <section className="mt-5">
      <h4 className="mb-2 flex items-center gap-1.5 text-[13px] font-bold tracking-wide text-fg-subtle">
        {Icon ? <Icon className="size-4" aria-hidden /> : null}
        {title}
      </h4>
      {children}
    </section>
  );
}

function Evidence({ spans, docs, view, empty }: { spans: readonly SourceSpan[]; docs: readonly SourceDocument[]; view: GraphView; empty?: ReactNode }) {
  const [all, setAll] = useState(false);
  const list = groupSpans(spans);
  if (!list.length) return <>{empty ?? <p className="text-[14px] text-fg-muted">원문 근거 위치가 없습니다.</p>}</>;
  const shown = all ? list : list.slice(0, 3);
  return (
    <div className="space-y-2.5">
      {shown.map((s) => (
        <EvidenceQuote
          key={`${s.doc_id}:${s.spans[0]!.start}`}
          spans={s.spans}
          doc={docs.find((d) => d.doc_id === s.doc_id)}
          docRef={view.docs?.find((d) => d.doc_id === s.doc_id)}
        />
      ))}
      {list.length > 3 ? (
        <button type="button" onClick={() => setAll((v) => !v)} className="inline-flex items-center gap-1 text-[14px] font-semibold text-brand hover:underline">
          <ChevronDown className={clsx('size-4 transition-transform', all && 'rotate-180')} aria-hidden />
          {all ? '접기' : `근거 ${list.length - 3}건 더 보기`}
        </button>
      ) : null}
    </div>
  );
}

function nodeName(view: GraphView, id: string): string {
  const n = view.nodes.find((x) => x.id === id);
  return n ? displayName(n, id) : nameFromId(id);
}

function FindingDetail({ f, report, view, docs, patches, ws, onSelect, onFocusService }: Ctx & { f: Finding }) {
  const service = f.service_id ? report.readiness.services.find((s) => s.service_id === f.service_id) : undefined;
  const patch = patches.find((p) => p.finding_ids.includes(f.id) && p.status !== 'rejected_by_validator');
  const subjects = f.subject_ids.filter((id) => id !== f.service_id).slice(0, 8);
  return (
    <div>
      <div className="flex flex-wrap items-center gap-2">
        <SeverityChip severity={f.severity} size="lg" />
        <Badge tone={f.status === 'unknown' ? 'muted' : 'danger'} size="lg">
          {FINDING_STATUS_LABEL[f.status]}
        </Badge>
        <code className="rounded-md border border-line bg-surface-2 px-2 py-0.5 text-[13px] font-semibold">{f.rule_id}</code>
        <span className="text-[13px] text-fg-subtle">{ruleFamily(f.rule_id)}</span>
      </div>
      <h3 className="mt-3 text-[22px] leading-snug font-extrabold">{f.title}</h3>
      <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[14px] text-fg-muted">
        {service ? (
          <button type="button" onClick={() => onFocusService(service.service_id)} className="font-semibold text-fg hover:text-brand hover:underline">
            {service.service_name} · {service.tier}등급
          </button>
        ) : (
          <span>전사 공통</span>
        )}
        {f.element ? <span>필수 요소: {ELEMENT_LABEL[f.element]}</span> : null}
      </div>
      <p className="mt-4 text-[16px] leading-relaxed">{f.message}</p>

      <Section title="근거 원문" icon={FileQuestion}>
        <Evidence
          spans={f.evidence}
          docs={docs}
          view={view}
          empty={
            <div className="flex gap-2 rounded-xl border border-dashed border-line-strong bg-surface-2 px-4 py-3 text-[14px] text-fg-muted">
              <Info className="mt-0.5 size-4 shrink-0" aria-hidden />
              <span>
                런북에 해당 내용이 <strong className="text-fg">아예 없습니다</strong>. 없는 것 자체가 끊김이며, 카탈로그·명부·당직표 대조로 찾았습니다.
              </span>
            </div>
          }
        />
      </Section>

      {f.legal_basis ? (
        <Section title="법적 근거" icon={Scale}>
          <p className="rounded-xl border border-brand/30 bg-brand/8 px-4 py-3 text-[14px] leading-relaxed">{f.legal_basis}</p>
        </Section>
      ) : null}

      {subjects.length ? (
        <Section title="관련 요소">
          <ul className="flex flex-wrap gap-1.5">
            {subjects.map((id) => {
              const isEdge = id.includes('->');
              return (
                <li key={id}>
                  <button
                    type="button"
                    onClick={() => onSelect({ type: isEdge ? 'edge' : 'node', id })}
                    className="rounded-lg border border-line bg-surface-2 px-2 py-1 text-[13px] hover:border-line-strong hover:bg-surface-3"
                  >
                    {isEdge ? edgeText(view, id) : nodeName(view, id)}
                  </button>
                </li>
              );
            })}
          </ul>
        </Section>
      ) : null}

      <Section title="수정안" icon={GitPullRequestArrow}>
        {patch ? (
          <Link
            to={`/app/w/${encodeURIComponent(ws)}/patches?patch=${encodeURIComponent(patch.id)}`}
            className="group block rounded-xl border border-brand/40 bg-brand/8 px-4 py-3 hover:bg-brand/14"
          >
            <p className="text-[15px] leading-relaxed font-semibold">“{patch.runbook_text}”</p>
            <p className="mt-1.5 inline-flex items-center gap-1 text-[14px] font-bold text-brand">
              수정안 보기 <ArrowRight className="size-4 transition-transform group-hover:translate-x-0.5" aria-hidden />
            </p>
          </Link>
        ) : (
          <p className="text-[14px] text-fg-muted">아직 수정안이 없습니다. 담당자 지정처럼 조직이 정할 일은 사람이 확인합니다.</p>
        )}
      </Section>
    </div>
  );
}

function edgeText(view: GraphView, id: string): string {
  const e = view.edges.find((x) => x.id === id);
  if (!e) return id;
  return `${nodeName(view, e.src)} → ${nodeName(view, e.dst)} (${EDGE_KIND_LABEL[e.kind]})`;
}

function RelatedFindings({ ids, report, onSelect }: { ids: readonly string[]; report: DiagnosisReport; onSelect: Ctx['onSelect'] }) {
  const list = report.findings.filter((f) => ids.includes(f.id));
  if (!list.length)
    return (
      <p className="inline-flex items-center gap-1.5 text-[14px] text-ok">
        <CircleCheck className="size-4" aria-hidden /> 이 요소에 걸린 끊김이 없습니다
      </p>
    );
  return (
    <ul className="space-y-1.5">
      {list.map((f) => (
        <li key={f.id}>
          <button
            type="button"
            onClick={() => onSelect({ type: 'finding', id: f.id })}
            className="flex w-full items-center gap-2 rounded-xl border border-line bg-surface-2 px-3 py-2 text-left hover:border-line-strong"
          >
            <SeverityChip severity={f.severity} size="sm" />
            <span className="min-w-0 flex-1 truncate text-[14px] font-semibold">{f.title}</span>
            <code className="text-[11px] text-fg-subtle">{f.rule_id}</code>
          </button>
        </li>
      ))}
    </ul>
  );
}

function AttrTable({ attrs }: { attrs: GraphViewNode['attrs'] }) {
  const entries = Object.entries(attrs).filter(([k]) => k !== 'aliases' || true);
  if (!entries.length) return <p className="text-[14px] text-fg-muted">적힌 속성이 없습니다.</p>;
  return (
    <dl className="divide-y divide-line overflow-hidden rounded-xl border border-line">
      {entries.map(([k, a]) => (
        <div key={k} className="grid grid-cols-[110px_1fr] gap-3 bg-surface-2 px-3 py-2 text-[14px]">
          <dt className="text-fg-subtle">{attrLabel(k)}</dt>
          <dd className="min-w-0">
            {formatAttr(k, a.value)}
            <span className="ml-1.5 text-[11px] text-fg-subtle">({SOURCE_LABEL[a.source]})</span>
          </dd>
        </div>
      ))}
    </dl>
  );
}

function NodeDetail({ n, report, view, docs, onSelect }: Ctx & { n: GraphViewNode }) {
  const Icon = KIND_ICON[n.kind];
  const related = view.findings_by_element[n.id] ?? report.findings.filter((f) => f.subject_ids.includes(n.id)).map((f) => f.id);
  const status = n.kind === 'person' ? (n.attrs.status?.value as keyof typeof PERSON_STATUS_LABEL | undefined) : undefined;
  const channels = view.edges.filter((e) => e.src === n.id && e.kind === 'reachable_via').map((e) => nodeName(view, e.dst));
  return (
    <div>
      <div className="flex items-center gap-3">
        <span className="grid size-11 place-items-center rounded-xl bg-surface-3 text-fg-muted">
          <Icon className="size-6" aria-hidden />
        </span>
        <div className="min-w-0">
          <p className="text-[13px] text-fg-subtle">{NODE_KIND_LABEL[n.kind]}</p>
          <h3 className={clsx('truncate text-[22px] font-extrabold', status === 'departed' && 'text-danger line-through')}>{displayName(n)}</h3>
        </div>
      </div>
      <div className="mt-3 flex flex-wrap gap-1.5">
        {status && status !== 'active' ? (
          <Badge tone={status === 'departed' ? 'danger' : status === 'on_leave' ? 'warn' : 'muted'} size="lg">
            {PERSON_STATUS_LABEL[status]}
          </Badge>
        ) : null}
        {n.verification === 'hold' ? (
          <Badge tone="warn" size="lg">
            판단 보류{n.hold_reason ? ` · ${n.hold_reason}` : ''}
          </Badge>
        ) : null}
        {(n.sources ?? []).map((s) => (
          <Badge key={s} tone="neutral">
            출처: {SOURCE_LABEL[s]}
          </Badge>
        ))}
      </div>
      <Section title="관련 끊김">
        <RelatedFindings ids={related} report={report} onSelect={onSelect} />
      </Section>
      {channels.length ? (
        <Section title="연락 수단">
          <p className="text-[15px]">{channels.join(' · ')}</p>
        </Section>
      ) : null}
      <Section title="속성">
        <AttrTable attrs={n.attrs} />
      </Section>
      <Section title="근거 원문" icon={FileQuestion}>
        <Evidence spans={n.spans} docs={docs} view={view} empty={<p className="text-[14px] text-fg-muted">런북 원문이 아니라 명부·카탈로그 같은 구조화 출처에서 왔습니다.</p>} />
      </Section>
    </div>
  );
}

function EdgeDetail({ e, report, view, docs, onSelect }: Ctx & { e: GraphViewEdge }) {
  const nodes = new Map(view.nodes.map((n) => [n.id, n]));
  const related = view.findings_by_element[e.id] ?? report.findings.filter((f) => f.subject_ids.includes(e.id)).map((f) => f.id);
  return (
    <div>
      <p className="text-[13px] text-fg-subtle">연결 · {EDGE_KIND_LABEL[e.kind]}</p>
      <h3 className="mt-1 text-[20px] leading-snug font-extrabold">
        <button type="button" className="hover:text-brand" onClick={() => onSelect({ type: 'node', id: e.src })}>
          {nodeName(view, e.src)}
        </button>
        <span className="mx-2 text-fg-subtle">→</span>
        <button type="button" className="hover:text-brand" onClick={() => onSelect({ type: 'node', id: e.dst })}>
          {nodeName(view, e.dst)}
        </button>
      </h3>
      <p className="mt-1 text-[14px] text-fg-muted">그래프 라벨: {edgeLabel(e, nodes)}</p>
      <Section title="관련 끊김">
        <RelatedFindings ids={related} report={report} onSelect={onSelect} />
      </Section>
      <Section title="속성">
        <AttrTable attrs={e.attrs} />
      </Section>
      <Section title="근거 원문" icon={FileQuestion}>
        <Evidence spans={e.spans} docs={docs} view={view} />
      </Section>
    </div>
  );
}
