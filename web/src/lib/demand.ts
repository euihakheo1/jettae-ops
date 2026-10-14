/**
 * 수요 측정 이벤트 중 '한 번만' 세는 것들 (docs/deploy.md의 베타 지표).
 *  - landing_first_view: 브라우저마다 한 번 — 서버가 받은(2xx) 뒤에만 '방문함' 표시를 남겨, 전송에 실패한 방문도 다음에 센다
 *  - upload_click: 페이지 세션(새로고침 전까지)마다 한 번 — 처음 런북 파일을 골랐을 때
 * 식별자는 저장하지 않는다.
 */
import { api } from '../api/client';
import { isFirstVisit, markVisited } from './freeScans';

/** upload_click의 files 상한 — 서버의 이벤트 허용 범위(0~100)와 같다 */
export const MAX_UPLOAD_CLICK_FILES = 100;

let firstViewSending = false;
let uploadClickSent = false;

/** 이 브라우저의 첫 방문이면 landing_first_view를 보낸다. 서버가 받았으면 true */
export async function trackFirstVisit(referrer: string): Promise<boolean> {
  if (firstViewSending || !isFirstVisit()) return false;
  firstViewSending = true;
  try {
    const ok = await api.track({ name: 'landing_first_view', path: '/', props: { referrer } });
    if (ok) markVisited();
    return ok;
  } finally {
    firstViewSending = false;
  }
}

/**
 * 이 페이지 세션에서 처음 런북 파일을 골랐을 때 한 번만 upload_click({files: 고른 런북 수})을 보낸다.
 * 한 번에 100개보다 많이 골라도 100으로 보낸다(넘으면 서버가 422로 거절해 이 세션의 업로드 의향을 놓친다).
 */
export function trackUploadClick(files: number): void {
  if (uploadClickSent || files <= 0) return;
  uploadClickSent = true;
  void api.track({ name: 'upload_click', path: '/app', props: { files: Math.min(files, MAX_UPLOAD_CLICK_FILES) } });
}

/** 테스트용: 한 번만 세는 표시를 처음 상태로 */
export function resetDemandTracking(): void {
  firstViewSending = false;
  uploadClickSent = false;
}
