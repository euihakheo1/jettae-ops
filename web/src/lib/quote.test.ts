import { describe, expect, it } from 'vitest';
import documentsJson from '../mocks/documents.json';
import type { SourceDocument, SourceSpan } from '../api/types';
import { findings } from '../test/fixtures';
import { groupSpans, quoteContext } from './quote';

const docs = documentsJson as unknown as SourceDocument[];
const pay = docs.find((d) => d.doc_id === 'doc-c6998bc94c89')!;

describe('quoteContext — 근거를 문서 문맥 안에서 강조', () => {
  const span: SourceSpan = { doc_id: pay.doc_id, start: 241, end: 249, quote: '안 되면 롤백.', line: 13 };

  it('정확한 오프셋이면 그 줄을 강조하고 앞뒤 줄을 붙인다', () => {
    const ctx = quoteContext(pay, span, 1)!;
    expect(ctx.exact).toBe(true);
    expect(ctx.startLine).toBe(13);
    expect(ctx.lines.map((l) => l.n)).toEqual([12, 13, 14]);
    const hit = ctx.lines[1]!.segments.filter((s) => s.hit).map((s) => s.text);
    expect(hit).toEqual(['안 되면 롤백.']);
  });

  it('같은 줄의 여러 인용을 함께 강조한다', () => {
    const a: SourceSpan = { ...span, start: pay.text.indexOf('김대리에게 연락'), end: pay.text.indexOf('김대리에게 연락') + 8, quote: '김대리에게 연락' };
    const ctx = quoteContext(pay, [a, span], 0)!;
    expect(ctx.lines).toHaveLength(1);
    expect(ctx.lines[0]!.segments.filter((s) => s.hit).map((s) => s.text)).toEqual(['김대리에게 연락', '안 되면 롤백.']);
  });

  it('오프셋이 어긋나면 인용문을 검색해서 찾는다', () => {
    const ctx = quoteContext(pay, { ...span, start: 0, end: 8 }, 0)!;
    expect(ctx.exact).toBe(false);
    expect(ctx.startLine).toBe(13);
  });

  it('문서가 없거나 인용을 못 찾으면 null', () => {
    expect(quoteContext(undefined, span)).toBeNull();
    expect(quoteContext(pay, { ...span, quote: '없는 문장' })).toBeNull();
  });
});

describe('groupSpans — 정보가 많은 근거부터', () => {
  it('JT-PATH-020(결제 승인자 없음): 13줄 묶음이 맨 앞, 제목 한 단어는 뒤로', () => {
    const f = findings.find((x) => x.id === 'JT-PATH-020:08e9c34f06')!;
    const groups = groupSpans(f.evidence);
    expect(groups[0]!.line).toBe(13);
    expect(groups[0]!.spans.length).toBeGreaterThan(1);
    expect(groups.at(-1)!.line).toBe(1);
  });

  it('완전히 같은 근거는 한 번만', () => {
    const s: SourceSpan = { doc_id: 'd', start: 0, end: 3, quote: 'abc', line: 1 };
    expect(groupSpans([s, { ...s }])[0]!.spans).toHaveLength(1);
  });
});
