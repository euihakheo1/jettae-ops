import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { findings, graph, patches } from '../../test/fixtures';
import { lookupFrom, summarizePatch } from '../../lib/patchSummary';
import { PatchCard } from './PatchCard';

const lookup = lookupFrom(graph.nodes);

function renderCard(id: string, canApprove: boolean, lockedReason?: string) {
  const patch = patches.find((p) => p.id === id)!;
  const onApprove = vi.fn();
  render(
    <PatchCard
      patch={patch}
      summary={summarizePatch(patch, lookup)}
      serviceName="결제"
      findings={findings}
      scenarios={[]}
      canApprove={canApprove}
      lockedReason={lockedReason ?? null}
      highlighted={false}
      approving={false}
      onApprove={onApprove}
      onReject={vi.fn()}
    />,
  );
  return { onApprove };
}

describe('PatchCard', () => {
  it('뷰어에게는 승인 버튼이 꺼져 있고 이유를 알려 준다', () => {
    renderCard('patch-dmc-01', false);
    expect(screen.getByRole('button', { name: '승인' })).toBeDisabled();
    expect(screen.getByText('팀 리드만 승인할 수 있습니다')).toHaveAttribute('role', 'tooltip');
    expect(screen.getByRole('button', { name: '거절' })).toBeDisabled();
  });

  it('공개 체험판에서는 승인·거절 버튼에 체험판 안내를 붙인다', () => {
    renderCard('patch-dmc-01', false, '공개 체험판에서는 할 수 없습니다(팀 계정 필요)');
    expect(screen.getByRole('button', { name: '승인' })).toBeDisabled();
    expect(screen.getByRole('button', { name: '거절' })).toBeDisabled();
    expect(screen.getAllByText('공개 체험판에서는 할 수 없습니다(팀 계정 필요)')).toHaveLength(2);
    expect(screen.queryByText('팀 리드만 승인할 수 있습니다')).toBeNull();
  });

  it('팀 리드는 승인할 수 있고, 문장·요약·검증이 보인다', () => {
    const { onApprove } = renderCard('patch-dmc-01', true);
    const btn = screen.getByRole('button', { name: '승인' });
    expect(btn).toBeEnabled();
    btn.click();
    expect(onApprove).toHaveBeenCalledOnce();
    expect(screen.getByText(/결제 장애 알람 시 1차 담당은 김서연 사원/)).toBeInTheDocument();
    expect(screen.getByText('김서연 사원 → 박재현 팀장 (5분, 전화)')).toBeInTheDocument();
    expect(screen.getByText('명부 외 인물 0건')).toBeInTheDocument();
    expect(screen.getByText('AI 초안')).toBeInTheDocument();
  });

  it('그래프 변경이 없는 메모는 "사람 확인 필요"', () => {
    renderCard('patch-dmc-08', true);
    expect(screen.getByText('사람 확인 필요')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '승인' })).toBeDisabled();
  });
});
