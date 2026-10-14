import { FileText, Quote } from 'lucide-react';
import type { DocumentRef, SourceDocument, SourceSpan } from '../../api/types';
import { quoteContext } from '../../lib/quote';

/**
 * 근거 원문 블록: '문서명 : 줄' + 문서 문맥 안에서 인용을 강조.
 * 원문을 저장하지 않는 설정이면 인용만 보여 준다.
 */
export function EvidenceQuote({
  spans,
  doc,
  docRef,
  radius = 1,
}: {
  /** 같은 문서의 한 묶음 (보통 같은 줄) */
  spans: readonly SourceSpan[];
  doc?: SourceDocument | undefined;
  docRef?: DocumentRef | undefined;
  radius?: number;
}) {
  const span = spans[0]!;
  const ctx = quoteContext(doc, spans, radius);
  const title = doc?.title ?? docRef?.title ?? span.doc_id;
  const line = ctx?.startLine ?? span.line;
  return (
    <figure className="overflow-hidden rounded-xl border border-line bg-surface-2">
      <figcaption className="flex items-center gap-2 border-b border-line px-3 py-2 text-[13px] text-fg-muted">
        <FileText className="size-4 shrink-0" aria-hidden />
        <span className="truncate font-medium text-fg">{title}</span>
        <span className="text-fg-subtle">:</span>
        <span className="tnum shrink-0 font-mono">{line ? `${line}줄` : '줄 정보 없음'}</span>
      </figcaption>
      {ctx ? (
        <pre className="overflow-x-auto px-0 py-2 font-sans text-[14px] leading-relaxed whitespace-pre-wrap">
          {ctx.lines.map((l) => {
            const inHit = l.n >= ctx.startLine && l.n <= ctx.endLine;
            return (
              <div key={l.n} className={inHit ? 'bg-warn/6' : 'text-fg-muted'}>
                <span className="tnum inline-block w-10 pr-3 text-right font-mono text-[12px] text-fg-subtle select-none">{l.n}</span>
                {l.segments.map((s, i) =>
                  s.hit ? (
                    <mark key={i} className="quote-hit">
                      {s.text}
                    </mark>
                  ) : (
                    <span key={i}>{s.text}</span>
                  ),
                )}
              </div>
            );
          })}
        </pre>
      ) : (
        <blockquote className="flex gap-2 px-4 py-3 text-[15px]">
          <Quote className="mt-1 size-4 shrink-0 text-fg-subtle" aria-hidden />
          <span className="space-x-1">
            {spans.map((s) => (
              <mark key={`${s.start}:${s.end}`} className="quote-hit whitespace-pre-wrap">
                {s.quote}
              </mark>
            ))}
          </span>
        </blockquote>
      )}
    </figure>
  );
}
