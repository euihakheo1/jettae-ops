import { clsx } from 'clsx';
import { ArrowDown, ArrowUp, ArrowUpDown, Funnel } from 'lucide-react';
import { useMemo } from 'react';
import type { Element, Finding, ReadinessReport, Severity, SourceDocument } from '../../api/types';
import { SeverityChip } from '../../components/ui/Badge';
import { EmptyState } from '../../components/ui/States';
import { ELEMENT_LABEL, ELEMENT_ORDER, FINDING_STATUS_LABEL, SEVERITY_LABEL, SEVERITY_ORDER, SEVERITY_RANK } from '../../lib/labels';

export type SortKey = 'severity' | 'rule' | 'service' | 'element';

export interface FindingFilters {
  sev: Severity[];
  service: string;
  element: string;
  sort: SortKey;
  desc: boolean;
}

export function FindingsTable({
  findings,
  readiness,
  docs,
  filters,
  onFilters,
  selectedId,
  onSelect,
}: {
  findings: readonly Finding[];
  readiness: ReadinessReport;
  docs: readonly SourceDocument[];
  filters: FindingFilters;
  onFilters: (f: FindingFilters) => void;
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const svcName = (id: string | null) => (id ? (readiness.services.find((s) => s.service_id === id)?.service_name ?? id.replace(/^service:/, '')) : '전사');
  const rows = useMemo(() => {
    const filtered = findings.filter(
      (f) =>
        (filters.sev.length === 0 || filters.sev.includes(f.severity)) &&
        (!filters.service || (filters.service === '__none' ? !f.service_id : f.service_id === filters.service)) &&
        (!filters.element || (filters.element === '__none' ? !f.element : f.element === filters.element)),
    );
    const key = (f: Finding): string | number => {
      switch (filters.sort) {
        case 'severity':
          return SEVERITY_RANK[f.severity];
        case 'rule':
          return f.rule_id;
        case 'service':
          return svcName(f.service_id);
        case 'element':
          return f.element ? ELEMENT_ORDER.indexOf(f.element) : 99;
      }
    };
    const sorted = [...filtered].sort((a, b) => {
      const x = key(a);
      const y = key(b);
      const c = x < y ? -1 : x > y ? 1 : SEVERITY_RANK[a.severity] - SEVERITY_RANK[b.severity] || (a.id < b.id ? -1 : 1);
      return filters.desc ? -c : c;
    });
    return sorted;
  }, [findings, filters]);

  const services = [...new Set(findings.map((f) => f.service_id).filter((x): x is string => !!x))].sort((a, b) => svcName(a).localeCompare(svcName(b), 'ko'));
  const toggleSev = (s: Severity) => onFilters({ ...filters, sev: filters.sev.includes(s) ? filters.sev.filter((x) => x !== s) : [...filters.sev, s] });
  const sortBy = (k: SortKey) => onFilters({ ...filters, sort: k, desc: filters.sort === k ? !filters.desc : false });
  const docTitle = (id: string) => docs.find((d) => d.doc_id === id)?.title ?? '문서';

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex flex-wrap items-center gap-2 border-b border-line px-4 py-3">
        <Funnel className="size-4 text-fg-subtle" aria-hidden />
        <div role="group" aria-label="심각도 필터" className="flex flex-wrap gap-1.5">
          {SEVERITY_ORDER.filter((s) => findings.some((f) => f.severity === s)).map((s) => {
            const on = filters.sev.includes(s);
            return (
              <button
                key={s}
                type="button"
                aria-pressed={on}
                onClick={() => toggleSev(s)}
                className={clsx('rounded-full border px-2.5 py-1 text-[13px] font-semibold', on ? 'border-brand bg-brand/15 text-fg' : 'border-line text-fg-muted hover:text-fg')}
              >
                {SEVERITY_LABEL[s]} {findings.filter((f) => f.severity === s).length}
              </button>
            );
          })}
        </div>
        <select
          aria-label="서비스 필터"
          value={filters.service}
          onChange={(e) => onFilters({ ...filters, service: e.target.value })}
          className="h-8 rounded-lg border border-line bg-surface-2 px-2 text-[13px]"
        >
          <option value="">모든 서비스</option>
          {services.map((s) => (
            <option key={s} value={s}>
              {svcName(s)}
            </option>
          ))}
          <option value="__none">전사 공통</option>
        </select>
        <select
          aria-label="필수 요소 필터"
          value={filters.element}
          onChange={(e) => onFilters({ ...filters, element: e.target.value })}
          className="h-8 rounded-lg border border-line bg-surface-2 px-2 text-[13px]"
        >
          <option value="">모든 요소</option>
          {ELEMENT_ORDER.map((el) => (
            <option key={el} value={el}>
              {ELEMENT_LABEL[el]}
            </option>
          ))}
          <option value="__none">요소 밖(신고·당직 등)</option>
        </select>
        <span className="tnum ml-auto text-[13px] text-fg-muted">
          {rows.length} / {findings.length}건
        </span>
      </div>
      {rows.length === 0 ? (
        <EmptyState title="조건에 맞는 끊김이 없습니다">필터를 줄여 보세요.</EmptyState>
      ) : (
        <div className="min-h-0 flex-1 overflow-auto">
          <table className="w-full border-separate border-spacing-0 text-left text-[14px]">
            <thead className="sticky top-0 z-10 bg-surface">
              <tr className="text-[12.5px] text-fg-subtle">
                <Th label="심각도" k="severity" filters={filters} onSort={sortBy} />
                <Th label="규칙" k="rule" filters={filters} onSort={sortBy} />
                <th className="border-b border-line px-3 py-2 font-semibold">끊김</th>
                <Th label="서비스" k="service" filters={filters} onSort={sortBy} />
                <Th label="요소" k="element" filters={filters} onSort={sortBy} />
                <th className="border-b border-line px-3 py-2 font-semibold">근거</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((f) => {
                const ev = f.evidence[0];
                const active = f.id === selectedId;
                return (
                  <tr
                    key={f.id}
                    tabIndex={0}
                    aria-selected={active}
                    onClick={() => onSelect(f.id)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        onSelect(f.id);
                      }
                    }}
                    className={clsx('cursor-pointer align-top', active ? 'bg-brand/10' : 'hover:bg-surface-2')}
                  >
                    <td className="border-b border-line px-3 py-2.5">
                      <SeverityChip severity={f.severity} size="sm" />
                    </td>
                    <td className="border-b border-line px-3 py-2.5 font-mono text-[12.5px] whitespace-nowrap text-fg-muted">{f.rule_id}</td>
                    <td className="border-b border-line px-3 py-2.5">
                      <p className="font-semibold">{f.title}</p>
                      {f.status === 'unknown' ? <p className="text-[12px] text-fg-subtle">{FINDING_STATUS_LABEL.unknown}</p> : null}
                    </td>
                    <td className="border-b border-line px-3 py-2.5 whitespace-nowrap">{svcName(f.service_id)}</td>
                    <td className="border-b border-line px-3 py-2.5 whitespace-nowrap text-fg-muted">{f.element ? ELEMENT_LABEL[f.element as Element] : '—'}</td>
                    <td className="border-b border-line px-3 py-2.5 text-[12.5px] text-fg-muted">
                      {ev ? `${docTitle(ev.doc_id)} : ${ev.line ?? '?'}줄` : <span className="text-fg-subtle">원문에 없음</span>}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function Th({ label, k, filters, onSort }: { label: string; k: SortKey; filters: FindingFilters; onSort: (k: SortKey) => void }) {
  const active = filters.sort === k;
  const Icon = !active ? ArrowUpDown : filters.desc ? ArrowDown : ArrowUp;
  return (
    <th className="border-b border-line px-3 py-2 font-semibold" aria-sort={active ? (filters.desc ? 'descending' : 'ascending') : 'none'}>
      <button type="button" onClick={() => onSort(k)} className={clsx('inline-flex items-center gap-1 hover:text-fg', active && 'text-fg')}>
        {label}
        <Icon className="size-3.5" aria-hidden />
      </button>
    </th>
  );
}
