/** API 오류 — 화면에는 한국어 메시지를 보여 준다. */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: string | undefined;
  /** 서버 오류 본문의 기계용 코드(`not_found`, `free_scan_limit` …) 또는 클라이언트가 붙인 코드 */
  readonly code: string | undefined;

  constructor(status: number, message: string, detail?: string, code?: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
    this.code = code;
  }
}

/** 네트워크 단절·프록시 대상 없음 → 오프라인 데모 데이터로 전환할 오류 */
export class UnreachableError extends Error {
  constructor(message = 'API 서버에 연결할 수 없습니다') {
    super(message);
    this.name = 'UnreachableError';
  }
}

export function errorMessage(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 401 || err.status === 403) return err.message || '권한이 없습니다. 역할을 확인하세요.';
    if (err.status === 404) return err.message || '요청한 항목을 찾을 수 없습니다.';
    if (err.status === 409) return err.message || '다른 변경과 충돌했습니다. 새로고침 후 다시 시도하세요.';
    return err.message || `서버 오류 (${err.status})`;
  }
  if (err instanceof UnreachableError) return err.message;
  if (err instanceof Error) return err.message;
  return '알 수 없는 오류가 발생했습니다.';
}
