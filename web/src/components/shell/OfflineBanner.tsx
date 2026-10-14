import { useQueryClient } from '@tanstack/react-query';
import { RefreshCw, WifiOff } from 'lucide-react';
import { resetApiCheck } from '../../api/client';
import { setSourceMode, useSourceMode } from '../../api/source';
import { Button } from '../ui/Button';

/** API에 닿지 않을 때만 보이는 배너 (VITE_USE_MOCKS=1 목업 모드에서는 상단 칩만 보임) */
export function OfflineBanner() {
  const mode = useSourceMode();
  const qc = useQueryClient();
  if (mode !== 'offline') return null;
  return (
    <div role="status" className="flex items-center gap-3 border-b border-warn/40 bg-warn/12 px-4 py-2 text-[14px] text-fg 2xl:px-6">
      <WifiOff className="size-5 shrink-0 text-warn" aria-hidden />
      <p className="min-w-0 flex-1">
        <strong className="font-bold text-warn">오프라인 데모 데이터</strong>
        <span className="text-fg-muted">
          {' '}
          — 제때옵스 API 서버(/healthz)에 연결할 수 없어 별빛상점 예제(실제 진단 결과)를 보여 줍니다. 승인·되돌리기·리허설은 이 브라우저 안에서만 흉내 내며 새로고침하면 처음으로 돌아갑니다.
        </span>
      </p>
      <Button
        size="sm"
        variant="secondary"
        icon={<RefreshCw className="size-4" aria-hidden />}
        onClick={() => {
          resetApiCheck();
          setSourceMode('api');
          void qc.invalidateQueries();
        }}
      >
        다시 연결
      </Button>
    </div>
  );
}
