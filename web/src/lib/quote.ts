import type { SourceDocument, SourceSpan } from '../api/types';

export interface QuoteSegment {
  text: string;
  hit: boolean;
}

export interface QuoteLine {
  /** 1부터 시작하는 줄 번호 */
  n: number;
  segments: QuoteSegment[];
}

export interface QuoteContext {
  lines: QuoteLine[];
  startLine: number;
  endLine: number;
  /** 원문 위치가 모두 정확히 맞았는지 (아니면 본문 검색으로 찾은 것) */
  exact: boolean;
}

/** 근거 위치를 문서 본문에서 확정한다: 오프셋이 맞으면 그대로, 아니면 인용문 검색. */
function locate(text: string, span: SourceSpan): { start: number; end: number; exact: boolean } | null {
  if (text.slice(span.start, span.end) === span.quote) return { start: span.start, end: span.end, exact: true };
  const idx = text.indexOf(span.quote);
  return idx < 0 ? null : { start: idx, end: idx + span.quote.length, exact: false };
}

function lineStartsOf(text: string): number[] {
  const starts = [0];
  for (let i = 0; i < text.length; i += 1) if (text[i] === '\n') starts.push(i + 1);
  return starts;
}

function lineIndex(starts: readonly number[], offset: number): number {
  let lo = 0;
  let hi = starts.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (starts[mid]! <= offset) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

/**
 * 근거 원문(하나 또는 여럿)을 문서 문맥(앞뒤 radius줄) 안에서 강조할 수 있게 자른다.
 * 같은 문단의 여러 인용을 한 블록에 함께 강조한다. 찾지 못하면 null.
 */
export function quoteContext(doc: SourceDocument | undefined, spans: SourceSpan | readonly SourceSpan[], radius = 2): QuoteContext | null {
  if (!doc) return null;
  const list = Array.isArray(spans) ? (spans as readonly SourceSpan[]) : [spans as SourceSpan];
  const text = doc.text;
  const ranges = list.map((s) => locate(text, s)).filter((r): r is { start: number; end: number; exact: boolean } => r !== null);
  if (!ranges.length) return null;
  const starts = lineStartsOf(text);
  const first = Math.min(...ranges.map((r) => lineIndex(starts, r.start)));
  const last = Math.max(...ranges.map((r) => lineIndex(starts, Math.max(r.start, r.end - 1))));
  const from = Math.max(0, first - radius);
  const to = Math.min(starts.length - 1, last + radius);
  const lines: QuoteLine[] = [];
  for (let li = from; li <= to; li += 1) {
    const ls = starts[li]!;
    const le = li + 1 < starts.length ? starts[li + 1]! - 1 : text.length;
    const lineText = text.slice(ls, le);
    // 이 줄에서 강조할 글자 표시 (겹치는 인용은 합친다)
    const mask = new Array<boolean>(lineText.length).fill(false);
    for (const r of ranges) {
      const a = Math.max(r.start, ls) - ls;
      const b = Math.min(r.end, le) - ls;
      for (let i = a; i < b; i += 1) mask[i] = true;
    }
    const segments: QuoteSegment[] = [];
    let cur = '';
    let curHit = mask[0] ?? false;
    for (let i = 0; i < lineText.length; i += 1) {
      if (mask[i] !== curHit) {
        if (cur) segments.push({ text: cur, hit: curHit });
        cur = '';
        curHit = mask[i]!;
      }
      cur += lineText[i];
    }
    if (cur) segments.push({ text: cur, hit: curHit });
    lines.push({ n: li + 1, segments: segments.length ? segments : [{ text: '', hit: false }] });
  }
  return { lines, startLine: first + 1, endLine: last + 1, exact: ranges.every((r) => r.exact) };
}

export interface SpanGroup {
  doc_id: string;
  /** 대표 줄 (가장 앞 인용의 줄) */
  line: number | null;
  spans: SourceSpan[];
}

/**
 * 같은 문서의 같은 줄(또는 겹치는 범위)에 있는 근거를 한 묶음으로 모으고,
 * 정보가 많은 묶음(인용 글자 수 합이 큰 것)부터 보여 준다. 제목 한 단어 같은 약한 근거는 뒤로 간다.
 */
export function groupSpans(spans: readonly SourceSpan[]): SpanGroup[] {
  const sorted = [...spans].sort((a, b) => (a.doc_id === b.doc_id ? a.start - b.start || b.end - a.end : a.doc_id < b.doc_id ? -1 : 1));
  const groups: { doc_id: string; start: number; end: number; lines: Set<number>; spans: SourceSpan[] }[] = [];
  const seen = new Set<string>();
  for (const s of sorted) {
    const key = `${s.doc_id}:${s.start}:${s.end}`;
    if (seen.has(key)) continue;
    seen.add(key);
    const line = s.line ?? -1;
    const g = groups.find((x) => x.doc_id === s.doc_id && ((s.start < x.end && s.end > x.start) || (line >= 0 && x.lines.has(line))));
    if (g) {
      g.spans.push(s);
      g.start = Math.min(g.start, s.start);
      g.end = Math.max(g.end, s.end);
      if (line >= 0) g.lines.add(line);
    } else groups.push({ doc_id: s.doc_id, start: s.start, end: s.end, lines: new Set(line >= 0 ? [line] : []), spans: [s] });
  }
  const weight = (g: (typeof groups)[number]): number => {
    // 겹치는 인용은 한 번만 센다
    const covered = new Set<number>();
    for (const s of g.spans) for (let i = s.start; i < s.end; i += 1) covered.add(i);
    return covered.size;
  };
  return groups
    .map((g) => ({ g, w: weight(g) }))
    .sort((a, b) => b.w - a.w || (a.g.doc_id < b.g.doc_id ? -1 : a.g.doc_id > b.g.doc_id ? 1 : a.g.start - b.g.start))
    .map(({ g }) => ({ doc_id: g.doc_id, line: g.spans.reduce<number | null>((m, s) => (s.line !== null && (m === null || s.line < m) ? s.line : m), null), spans: g.spans }));
}

/** 같은 문서·같은 줄의 근거는 하나로, 긴 인용을 우선 (리포트 등 한 줄 요약용) */
export function dedupeSpans(spans: readonly SourceSpan[]): SourceSpan[] {
  return groupSpans(spans).map((g) => g.spans.reduce((best, s) => (s.quote.length > best.quote.length ? s : best)));
}
