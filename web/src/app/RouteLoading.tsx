import { LoaderCircle } from 'lucide-react';
import { BrandMark } from '../components/shell/Brand';

/** 첫 화면 코드 조각을 불러오는 동안 */
export function RouteLoading() {
  return (
    <div className="grid h-full place-items-center bg-bg" role="status" aria-live="polite">
      <div className="flex flex-col items-center gap-4">
        <BrandMark size={48} />
        <span className="inline-flex items-center gap-2 text-[15px] text-fg-muted">
          <LoaderCircle className="size-4 animate-spin" aria-hidden />
          제때옵스를 여는 중…
        </span>
      </div>
    </div>
  );
}
