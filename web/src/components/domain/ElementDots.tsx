import { clsx } from 'clsx';
import type { ElementScore, ElementStatus } from '../../api/types';
import { ELEMENT_LABEL, ELEMENT_ORDER, ELEMENT_STATUS_LABEL } from '../../lib/labels';
import { ELEMENT_STATUS_ICON } from '../ui/Badge';
import { Tooltip } from '../ui/Tooltip';

const DOT: Record<ElementStatus, string> = {
  pass: 'text-ok',
  partial: 'text-warn',
  fail: 'text-danger',
  unknown: 'text-fg-subtle',
};

/** 5요소(감지·담당·승인자·대체 채널·롤백) 상태 점 — 아이콘 모양으로도 구분, 툴팁에 사유 */
export function ElementDots({ elements, showLabels = false }: { elements: readonly ElementScore[]; showLabels?: boolean }) {
  return (
    <ul className="flex items-center gap-1.5" aria-label="필수 경로 5요소">
      {ELEMENT_ORDER.map((el) => {
        const e = elements.find((x) => x.element === el);
        const status: ElementStatus = e?.status ?? 'unknown';
        const Icon = ELEMENT_STATUS_ICON[status];
        return (
          <li key={el}>
            <Tooltip
              focusable
              wide
              content={
                <span className="block">
                  <span className="font-bold">
                    {ELEMENT_LABEL[el]} · {ELEMENT_STATUS_LABEL[status]}
                  </span>
                  {e?.reasons.length ? (
                    <span className="mt-1 block text-fg-muted">{e.reasons.join(' · ')}</span>
                  ) : null}
                </span>
              }
            >
              <span
                className={clsx('inline-flex items-center gap-0.5 rounded', DOT[status])}
                aria-label={`${ELEMENT_LABEL[el]}: ${ELEMENT_STATUS_LABEL[status]}`}
              >
                <Icon className="size-4" strokeWidth={2.5} aria-hidden />
                {showLabels ? <span className="text-xs font-medium">{ELEMENT_LABEL[el]}</span> : null}
              </span>
            </Tooltip>
          </li>
        );
      })}
    </ul>
  );
}
