# 제때옵스(JETTAE Ops) 웹

런북(장애 대응 매뉴얼)을 **운영 그래프**로 바꾸고, 끊긴 대응 경로를 장애 전에 찾아 고치는 제때옵스의 웹 앱입니다.
공개 소개 페이지(`/`)와 진단 앱(`/app`)이 한 번들에 들어 있고, 빌드 결과(`dist/`)는 FastAPI가 정적 파일로 서빙합니다.

- **스택**: Vite 8 · React 19 · TypeScript 7(strict) · Tailwind CSS 4 · React Flow 12(`@xyflow/react`) · elkjs 0.12(웹 워커) · TanStack Query 5 · React Router 7 · lucide-react · Pretendard(자체 호스팅)
- **오프라인 대응**: 글꼴·아이콘·그래프 배치 엔진까지 모두 번들에 들어 있어 런타임에 외부(CDN) 요청이 없습니다. 결선장처럼 인터넷이 불안정해도 동작합니다.

## 실행

Node 24 / npm 11 기준. 처음 한 번 `npm ci`(또는 `npm install`)를 실행합니다.

### 1) 목업 모드 — API 없이 별빛상점 예제로

```bash
npm run dev:mock            # = vite --mode mock (.env.mock 의 VITE_USE_MOCKS=1)
# 또는 Git Bash / macOS / Linux
VITE_USE_MOCKS=1 npm run dev
# Windows PowerShell
$env:VITE_USE_MOCKS=1; npm run dev
```

http://localhost:5173 → `별빛상점 예제로 바로 보기` 또는 http://localhost:5173/app/w/dmc-shop

### 2) API와 함께

```bash
# 저장소 루트(J:\jettae-ops)에서
jettae serve                # FastAPI — http://127.0.0.1:8765 (/api/v1, /healthz)
# web/ 에서
npm run dev                 # /api, /healthz 는 127.0.0.1:8765 으로 프록시
```

앱은 처음에 `GET /healthz`로 API를 확인합니다. **닿지 않거나 2xx가 아니면** 자동으로 별빛상점 예제 데이터로 전환하고
상단에 `오프라인 데모 데이터` 배너를 띄웁니다(배너의 `다시 연결`로 재시도). 이때 승인·되돌리기·리허설은 브라우저 메모리에서만 흉내 내며 새로고침하면 처음으로 돌아갑니다.
**이 자동 오프라인 데모는 개발 서버(`npm run dev`)와 목업 모드에서만 그대로입니다.** 배포 빌드는 아래처럼 좁게 동작합니다.

#### 배포 빌드(`npm run build` · `npm run preview` · Render)의 차이

공개 베타에서 방문자의 쓰기가 브라우저 메모리에만 남고 "됐다"고 보이는 일을 막으려고, `VITE_USE_MOCKS` 없이 만든 배포 빌드
(`import.meta.env.PROD`, `src/api/client.ts`의 `strictOffline()`)는 오프라인 대체를 좁힙니다.

| | 개발 서버 · 목업 모드 | 배포 빌드 |
|---|---|---|
| 쓰기(사전 신청 · 무료 검진 · `예제로 시작`/`별빛상점 예제로 바로 보기` · 새 진단 · 이벤트) | API에 닿지 않으면 목업으로 흉내 | **항상 서버로**. 닿지 않으면 `서버에 잠시 연결할 수 없습니다. 잠시 후 다시 시도하세요.` |
| 워크스페이스 화면(읽기 · 초안 · 리허설 · 승인 · 거절 · 되돌리기) | 어느 워크스페이스든 예제 데이터로 | **예제 워크스페이스만**(`dmc-shop`과 서버 목록에서 본 예제 id). 그 밖의 워크스페이스는 오류 |
| 역할(`GET /me`) | 오프라인이면 상단 역할 전환기의 역할(기본 팀 리드) | 세션을 꾸며 내지 않음. **`/me`가 답하기 전(대기·실패)에는 보기 전용**이고 역할 전환기도 숨김 |
| 오프라인 전환 | `/healthz` 한 번 실패(목업 모드는 처음부터 목업) | `/healthz`를 한 번 더 확인. 오프라인 중에도 서버가 답하면 API로 돌아옴 |

- 그래서 백엔드 없이 `npm run preview`만 띄우면 `예제로 시작` 버튼은 다시 시도 안내를 띄웁니다. 예제는 `/app/w/dmc-shop`으로 직접 열면 보이지만(읽기 전용),
  승인·리허설까지 오프라인으로 시연하려면 **목업 모드**를 쓰세요: `npm run dev:mock`, 또는 목업 번들이 필요하면
  `npx vite build --mode mock`(`.env.mock`의 `VITE_USE_MOCKS=1`) 뒤 `npm run preview`. 이렇게 만든 번들은 배포 빌드여도 예전처럼 동작합니다
  (`dist/`를 덮어쓰니, FastAPI로 서빙하거나 배포하기 전에는 `npm run build`로 다시 만드세요).
- 결선 시연처럼 배포 빌드를 **개발 모드 서버**(`jettae serve`/`jettae demo`, 역할 전환 가능)에 붙이면 `/me`가 `can_switch_role: true`를
  돌려주므로, 답을 받는 즉시 역할 전환기와 팀 리드 역할이 예전처럼 나타납니다. 공개 모드 서버면 `공개 체험판 · 보기 전용` 배지가 나옵니다.
- 수요 이벤트(`POST /events`)의 `path`는 서버가 받는 화면 경로 `/` · `/app` · `/app/report` · `/privacy` 중 하나만 씁니다(`EVENT_PATHS`, `src/api/types.ts`).

### 3) 빌드·테스트

```bash
npm run build      # tsc -b && vite build → dist/
npm run preview    # dist 미리보기 (프록시 동일)
npm test           # vitest run (그래프 뷰 모델·포매터·수정안 요약·목업 API·컴포넌트)
npm run typecheck  # tsc -b --noEmit
```

### 4) 화면 스크린숏 (발표 자료·시각 QA)

```bash
npm run screenshots                       # 목업 빌드(.shots-dist) → vite preview(빈 포트) → 캡처 → 서버 종료
npm run screenshots -- --qa               # 스크롤 화면 아래쪽도 screenshots/qa/ 에 추가로
npm run screenshots -- --only dashboard   # 이름 일부가 맞는 장면만
npm run screenshots -- --live --base http://127.0.0.1:8765   # 실제 API(jettae serve)에 붙어서 (목업 환경 변수 없음)
```

- 브라우저는 내려받지 않습니다. 설치된 Google Chrome(`channel: 'chrome'`) → 없으면 Edge(`msedge`)를 씁니다.
  처음 설치할 때는 `PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 npm install`.
- 1920×1080, 배율 1, 어두운 화면·프로젝터 모드 두 벌을 `web/screenshots/*.png`로 저장합니다.
- 장면: 소개 · 워크스페이스 · 대시보드(결제 집중 + 끊김 선택, 그래프만, 전체 보기) · 리허설 FAIL · 수정안 · 그래프 버전 · 리포트 · 보안·AI 사용 · 승인 뒤 대시보드 · 승인 뒤 리허설 PASS.
  목업에서는 승인 뒤 장면을 `?mock=after`로 만들고, `--live`에서는 서버의 현재 상태를 그대로 찍습니다.
- 한 화면만: `npm run screenshots -- --url "/app/w/dmc-shop?focus=all" --name overview` (Git Bash에서는 `MSYS_NO_PATHCONV=1`을 앞에 붙이세요).

FastAPI에서 `dist/`를 서빙할 때는 `/assets/*`는 정적 파일로, 그 밖의 비-API 경로(`/app/...`)는 `index.html`로 돌려주는 SPA 폴백이 필요합니다.

## 화면과 URL (스크린숏·시연용 딥링크)

| 경로 | 화면 |
|---|---|
| `/` | 소개 페이지 — 히어로, 변환→진단→리허설→개선, AI가 하는/하지 않는 일, 신뢰, 가격, **상세 진단 사전 신청(결제 없음)** |
| `/privacy` | 개인정보 처리방침 — 수집 항목·목적·보유 기간, 국외 이전(Render·싱가포르), 이용 통계·서버 접속 기록, 정보주체의 권리 |
| `/app` | 워크스페이스 목록 · 새 진단(런북 끌어다 놓기 + 명부/당직표/카탈로그/정책, LLM OFF/LIVE/REPLAY) · 별빛상점 예제 불러오기 |
| `/app/w/:ws` | **진단 대시보드** — 준비도 점수·끊김·리허설 작동률·판단 보류 KPI, 서비스 목록(5요소 점), 운영 그래프, 상세 서랍, 끊김 목록 |
| `/app/w/:ws/rehearsals` | 모의 장애 리허설 — 시나리오 카드, PASS/FAIL, 완화 시간 vs 시한, 타임라인, 경로 재생, 수정 전후 비교 |
| `/app/w/:ws/patches` | 수정안 — 서비스별 카드, 런북 문장, 그래프 변경 요약, 검증 결과, 승인(팀 리드)/거절 |
| `/app/w/:ws/versions` | 그래프 버전 — 이력, 버전 비교(추가·삭제·속성 변경), 되돌리기(새 버전) |
| `/app/w/:ws/report` | 증적 리포트 — 미리보기, Markdown·HTML·SARIF 내려받기, ISMS-P 2.11.4·2.11.5·2.12.2 |
| `/app/w/:ws/trust` | 보안·AI 사용 — 격리된 문서 속 지시문, 원문 근거 정렬, LLM 사용량, 고정 규칙 카탈로그 |

쿼리 파라미터:

| 파라미터 | 어디서 | 뜻 |
|---|---|---|
| `theme=projector` / `theme=dark` | 모든 화면 | 프로젝터 모드(밝은 고대비·큰 글씨) / 어두운 화면 — 저장됨 |
| `role=viewer` / `role=lead` (`editor`, `admin`) | 모든 화면 | 역할 전환(`X-Jettae-Role`) — 저장됨 |
| `mock=after` | 목업/오프라인 | 결제 두 문장 + 침해사고 문장을 승인한 뒤 상태(v4)로 시작 |
| `focus=결제` / `focus=all` | 대시보드 | 집중 서비스(이름 또는 `service:` id) / 전체 보기 |
| `finding=<finding id>` · `node=<id>` · `edge=<id>` | 대시보드 | 상세 서랍 열기 + 그래프 강조 |
| `tab=findings` · `sev=critical,high` · `svc=service:결제` · `el=owner` · `sort=rule` · `desc=1` | 대시보드 | 끊김 목록 탭과 필터·정렬 |
| `ghosts=0` · `struct=1` | 대시보드 | 빠진 요소(유령 노드) 숨김 · 전체 보기에서 소속·위치·의존 연결 표시 |
| `scenario=<id>` · `play=1` · `step=n` · `only=fail` · `svc=결제` | 리허설 | 시나리오 선택 · 자동 재생 · n단계까지 공개 · 실패만 |
| `patch=<id>` · `finding=<id>` · `service=결제` · `show=done\|all` | 수정안 | 해당 카드로 스크롤·강조 · 필터 |
| `a=1&b=2` | 그래프 버전 | 비교할 두 버전 |

예: `/app/w/dmc-shop?finding=JT-PATH-020:08e9c34f06&theme=projector`,
`/app/w/dmc-shop/rehearsals?scenario=dmc-s1-chuseok-payment&mock=after&play=1`

## 구조

```
src/
  api/        types.ts(도메인 JSON 타입) · client.ts(§6 REST + 오프라인 폴백) · mock.ts(목업 API) · hooks.ts(React Query) · source.ts · errors.ts
  graph/      model.ts(순수 뷰 모델: 집중 부분 그래프·유령 노드·상태 표시) · layout.ts(ELK 레이어드, 워커) · OpsNode/OpsEdge/Legend/GraphCanvas
  lib/        format.ts · labels.ts(한국어 라벨) · patchSummary.ts(수정안 op → 한국어) · graphOps.ts(적용·역연산·비교) · quote.ts(근거 문맥) · isms.ts · reportExport.ts
  pages/      Landing · Privacy · WorkspaceHome · Dashboard · Rehearsals · Patches · Versions · Report · Trust
  components/ ui(Button·Badge·Tooltip·Dialog·Toast·Segmented·States) · domain(Score·ElementDots·EvidenceQuote·icons) · shell(AppShell·SideNav·TopBar·OfflineBanner)
  mocks/      백엔드 실제 출력(graph·findings·readiness·scenarios·rehearsals·rules·documents) + 목업(patches·versions·demo-overlay)
```

### 그래프 뷰 모델 규칙 (`src/graph/model.ts`)

- **집중 보기**: 서비스 + 감지 신호(+호출 대상) + 담당 + 역할→사람 + 상향 사슬 + 조치(+승인자·실행자) + 신고 의무 + 연락·대체 채널. 기본 집중 서비스는 **가장 중요한 등급(1등급) 중 점수가 가장 낮은 서비스**(별빛상점: 결제).
- **유령 노드**(빨간 점선): 끊김에서 빠진 필수 요소를 만든다 — `JT-PATH-001` 감지 신호 없음, `JT-PATH-010` 담당자 없음, `JT-ESC-001` 상향 경로 없음, `JT-PATH-020` 승인자 없음, `JT-PATH-041` 실행자 없음, `JT-PATH-030` 대체 채널 없음, `JT-PATH-040` 롤백 조치 없음, `JT-KR-001/003` 신고 단계 없음.
- 퇴사자는 빨강 + 취소선 + `퇴사` 배지, 휴직은 주황 `휴직`, 명부 미확인은 회색 점선, 판단 보류(HOLD)는 주황 점선 `판단 보류`.
- 끊김 subject 엣지: 치명·높음은 빨강 흐름 애니메이션, 보통은 주황. 감지 엣지는 감지 신호 → 서비스 방향으로 그린다.
- 입력을 id 순으로 정렬하고 ELK `randomSeed`를 고정해 **같은 그래프는 언제나 같은 배치**가 나온다(테스트로 확인).
- 연결선은 ELK가 계산한 스플라인을 그대로 그린다 → 여러 층을 건너는 연결이 중간 노드 밑으로 지나가지 않는다. 라벨은 퍼지는 연결은 도착 쪽, 모이는 연결은 출발 쪽에 둔다.
- 끊김을 고르면 관련 요소(연결의 양 끝 포함)만 화면에 맞추고 빨간색을 유지한 채 굵게·빛나게 강조한다. 나머지는 흐리게(집중 중인 서비스는 그대로).
- 배율이 0.58 아래(전체 보기 등)면 노드를 '지도' 모양(아이콘 + 큰 이름, 서비스는 점수)으로 바꾸고 연결 라벨은 숨긴다.

## API 계약 메모 (백엔드 팀과 맞출 것)

ARCHITECTURE §6 를 따르되, 아래는 웹이 가정한 부분입니다. 응답이 다르면 `src/api/client.ts`만 고치면 됩니다.

- `GET /api/v1/workspaces` (목록) — §6에 없어 추가 가정. 배열 또는 `{items}`. 필드는 `WorkspaceSummary`(id, name 외 선택).
- `GET /workspaces/{ws}/graph` — GraphView DTO `{version, graph_hash, nodes:[{…, spans}], edges, findings_by_element}` 와 `{version, graph: OpsGraph}` 둘 다 받습니다.
- `GET /workspaces/{ws}/report?version=n` — 이전 버전 점수(점수 변화 표시)와 v1 결과(리허설 수정 전후 비교)에 `version` 쿼리를 씁니다.
- `GET /workspaces/{ws}/patches` — 없으면(404/405) 진단 보고서의 `patches`를 씁니다.
- `GET /workspaces/{ws}/documents` — 근거 원문 문맥 표시용 `Document[]`. 원문 비저장이면 404 → 인용문만 표시.
- `GET /workspaces/{ws}/report.html`, `/report.sarif` — 리포트 미리보기·내려받기 (§6에는 `report.md`만 있음).
- `POST /patches/{id}/approve` 응답은 `Patch`, `GraphVersion`, `{patch, version}` 모두 처리. `PatchValidation.score_before/after`는 **수정 대상 서비스 점수**로 표시합니다.
- `POST /patches/{id}/reject` 본문 `{reason}`. `POST /signups` 본문 `{email, org_name, team_size, message, consent_privacy: true, consent_marketing, source}`. `POST /events` 본문 `{name, path, props}` (실패해도 무시).
- 오류 본문은 `{detail}`(FastAPI 기본)과 `{error:{message}}` 둘 다 읽습니다.
- 개발용 인증 헤더: `X-Jettae-User`(ASCII: `park.jaehyun`, `demo.viewer` …), `X-Jettae-Role`(viewer|editor|lead|admin).

## 목업 데이터

- `src/mocks/graph.json · findings.json · readiness.json · scenarios.json · rehearsals.json · rules.json · documents.json` — 백엔드가 별빛상점 예제로 만든 **실제 출력**입니다.
- `patches.json · versions.json · demo-overlay.json` — `examples/dmc-shop/patches/expected-fixes.md`와 `scenarios.yaml`을 바탕으로 손으로 만든 **목업**입니다(수정안 9건, 사람 작성 시나리오 3건, 승인 이후 리허설 결과). 승인 시 점수는 공개 가중치(SCORING_V1)로 다시 계산하고, 그래프는 op를 실제로 적용해 새 내용 해시를 만듭니다. 실제 판정은 백엔드 규칙·리플레이가 합니다.

## 알려진 한계

- 업로드 진단(새 워크스페이스·입력 업로드·진단 실행)은 API 모드에서만 동작합니다.
- 목업의 승인·되돌리기·리허설 결과는 미리 계산한 '수정 후' 표에서 고릅니다(브라우저에 규칙 엔진·시뮬레이터가 없음).
- 전체 보기(노드 90여 개)는 개요용입니다. 발표 때는 서비스 집중 보기 + 끊김 선택(관련 요소로 자동 확대)을 권장합니다.
