# 제때옵스(JETTAE Ops) 아키텍처

> "기억 말고 그래프로, 끊긴 곳은 장애 전에."
> 런북(장애 대응 매뉴얼)을 **운영 그래프**로 바꾸고, **고정 규칙**으로 끊긴 대응 경로를 찾고,
> **결정적 리플레이**로 모의 장애를 돌리고, **사람이 승인한 수정안**만 새 그래프 버전이 된다.

## 0. 설계 원칙 (모든 모듈 공통 — 위반 시 리뷰에서 반려)

| # | 원칙 | 구현으로 보장하는 방법 |
|---|---|---|
| P1 | **AI는 제안, 판정은 규칙** — 끊김 판정·점수·리허설 판정·그래프 변경에 LLM이 관여하지 않는다 | `jettae.check`, `jettae.rehearse.simulator`, `jettae.improve.{validator,patching}`, `jettae.explore.{sweep,verify,drafts}` 은 `jettae.llm`/`anthropic`/`openai` import 금지, 판정·점수 모듈은 `jettae.explore` import 금지 (import-linter 계약) |
| P2 | **지어내기 금지** — 그래프의 모든 요소는 원문 span 또는 구조화 출처(명부·카탈로그·승인 패치)를 가진다 | `Provenance` 검증기(RUNBOOK 출처는 span 필수), `assemble`의 정렬 실패 요소 폐기 |
| P3 | **확인 불가 ≠ 정상** — 모르면 통과가 아니라 `UNKNOWN`/`HOLD` | `FindingStatus.UNKNOWN`, `ElementStatus.UNKNOWN`은 점수 0 |
| P4 | **LLM을 꺼도 동작** — `LlmMode.OFF`에서 규칙 파서 → 동일 파이프라인 | 모든 LLM 단계는 `LLMDisabledError` 시 결정적 폴백 |
| P5 | **문서는 데이터, 지시가 아니다** — 문서 속 지시문은 격리, 출력은 스키마 강제 | `ingest.guard` 격리 + 구조화 출력 + LLM은 열거된 후보 중 선택만(그래프 op는 코드가 컴파일) |
| P6 | **결정성·재현성** — 같은 입력 → 같은 그래프 해시·같은 리허설 결과 해시 | 정렬된 id, canonical JSON, 난수 없음(시드만), `RehearsalResult.result_hash()` |
| P7 | **변경은 이벤트** — 그래프는 불변 값, 승인 패치만 새 버전, 되돌리기도 새 버전 | `GraphVersion` append-only, content-addressed |

## 1. 컨텍스트 & 컨테이너

```
 runbooks(md/pdf/docx/html) ─┐        ┌──────────────────── jettae (Python 3.11+) ────────────────────┐
 roster.csv / oncall.csv  ───┼─ingest─▶  extract ─▶ assemble ─▶ verify ─▶ check ─▶ score ─▶ rehearse   │
 catalog.yaml (Backstage/    │         │ (rule|LLM)  (grounding,  (역번역  (고정    (가중   (결정적     │
   compose/OTel deps)        │         │             entity res.) 대조)    규칙)    평균)   리플레이)   │
 policy.yaml ────────────────┘         │                                   └─▶ improve(초안→검증→승인) │
                                       │  orchestrator (LangGraph StateGraph) · store(SQLAlchemy)       │
                                       └──────────┬──────────────────────────────┬────────────────────┘
                                         FastAPI REST + 정적 웹(React/React Flow)   CLI(Typer) / GitHub Action(SARIF)
```

## 2. 패키지 레이아웃 & 의존 규칙

```
src/jettae/
  domain/        순수 모델 (graph, inputs, extraction, findings, rehearsal, patch, report, ports)
  ingest/        loaders, normalize, pii, guard(프롬프트 인젝션 격리), structured(명부/당직/정책/카탈로그), bundle
  llm/           base(포트), anthropic_client, openai_compat, replay(카세트), fake, factory, schema(strict JSON schema)
  extract/       rules(규칙 파서), llm_extractor, assemble(정렬·개체해석·그래프 조립), backtranslate(역번역 대조)
  check/         context, registry, rules/*, scoring, impact(영향 범위), spof(최소 절단 집합)
  rehearse/      scenarios(템플릿+LLM), simulator(결정적 이산사건 리플레이)
  explore/       규칙 밖 위험 탐색: sweep(결정적 전수 탐색), ai(가설 제안), verify(리허설 검증), explorer
  improve/       patching(적용·역연산·diff), drafter(템플릿+LLM), validator, versioning
  store/         memory(테스트/데모), sql(SQLAlchemy 2.0; SQLite/Postgres), migrations(alembic)
  reporting/     markdown(증적 리포트), sarif, html
  orchestrator/  pipeline(LangGraph), service(애플리케이션 서비스 파사드)
  api/           FastAPI app + routers
  cli/           Typer CLI (`jettae`)
  evalkit/       봉인 결함 주입 평가셋·2×2 하네스·주입 공격셋·κ
  config.py      pydantic-settings (env prefix JETTAE_)
```

레이어(위가 아래만 import): `api|cli|evalkit` → `orchestrator` → `improve|reporting|explore` → `rehearse` → `extract|check|store` → `ingest|llm` → `domain`.
같은 레이어의 형제끼리는 import 금지(`|`). `config.py`는 어디서나 import 가능하지만 `domain`은 import하지 않는다.

## 3. 도메인 모델 요약 (정의: `src/jettae/domain/*.py` — 계약이므로 변경 시 전 모듈 영향)

- **Node** `id = "{kind}:{slug(name)}"`(동명이인은 명부 출처 ref로 구분된 접미사 허용), kind ∈ service/system/location/person/role/team/channel/trigger/action/obligation
- **Edge** `id = "{kind}:{src}->{dst}"`(+`#disc`), kind/허용 시그니처는 `EDGE_SIGNATURES`
- **Attr** `{value, spans, source}` — 잘 알려진 키는 `AttrKey` (tier, status, title, medium, after_minutes, deadline_minutes …)
- **Provenance** `{source: runbook|catalog|roster|oncall|policy|patch, extractor: rule|llm|import|human, spans, ref}`
- **OpsGraph** 불변, `content_hash()` = canonical JSON의 SHA-256, `GraphBuilder`로 병합 조립
- **Extraction IR** 추출기 공통 출력(엔티티·관계 + 원문 그대로의 quote)
- **Finding** `{rule_id, severity, status(broken|unknown), service_id, element, subject_ids, evidence, fix: FixHint}`
- **ReadinessReport** 서비스별 5요소 점수 → 0~100
- **Scenario / RehearsalResult / SimConfig** 리허설 계약
- **Patch / PatchOp / PatchValidation / GraphVersion** 개선·버전 계약
- **DiagnosisReport** 한 번의 진단 결과(그래프 버전, 끊김, 점수, 리허설, 수정안, 근거율, LLM 사용량, 규칙 밖 위험 후보)
- **RiskHypothesis / ExplorationSummary** 규칙 밖 위험 가설(출처 sweep|ai, 교란, 전제, 상태 confirmed|refuted|needs_info, 재생 결과)과 탐색 범위

## 4. 모듈별 공개 API (구현 계약)

### 4.1 ingest
```python
load_document(path: str | Path, *, title: str | None = None) -> Document
load_bytes(filename: str, data: bytes, *, title: str | None = None) -> Document   # .md .txt .pdf .docx .html
normalize_text(text: str) -> str            # NFC, CRLF→LF, 제로폭·비가시 서식 문자 제거, 탭→공백 (guard보다 먼저 실행)
mask_pii(text: str) -> tuple[str, list[PiiHit]]   # 길이 보존 마스킹(오프셋·줄번호 유지): 휴대폰·이메일·IP·주민번호·카드
scan_injection(doc: Document) -> tuple[Document, list[SecurityNotice]]   # 격리 구간을 Document.quarantined에 기록
load_roster(src) / load_oncall(src) / load_policy(src) / load_catalog(src)    # CSV/YAML/Backstage/compose/Jaeger deps
load_bundle_dir(path) -> InputBundle        # runbooks/*, roster.csv, oncall.csv, policy.yaml, catalog.yaml
```
문서 생성 순서: 원본 → `normalize_text` → `mask_pii`(길이 보존) → `Document` → `scan_injection`.

### 4.2 llm
```python
class LLMClient(Protocol): provider; model; async structured(*, task, instructions, schema, documents, context, max_tokens) -> LLMResult[T]
AnthropicLLM(model="claude-opus-5-5", effort="medium")   # beta messages + fallbacks="default" + output_config.format(json_schema)
OpenAICompatLLM(base_url, model, api_key)                  # vLLM/사내 설치형·다른 계열 모델(평가용 재서술)
CassetteLLM(inner | None, path, mode="replay"|"record"|"auto")  # 데모·회귀 테스트용 결정적 재생
FakeLLM(handler)                                           # 테스트
NullLLM()                                                  # OFF → LLMDisabledError
build_llm(settings, mode, meter) -> LLMClient              # MeteredLLM로 감싸 사용량 집계
to_strict_json_schema(model: type[BaseModel]) -> dict      # $ref 인라인, additionalProperties=false, 모든 키 required(nullable)
```
비신뢰 문서는 `<untrusted_document nonce=… id=… title=…>` 경계로 감싸고(nonce = 문서 내용 SHA-256 앞 16자리: 문서가 자기 해시를 담을 수 없으므로 위조 불가, 카세트·프롬프트 캐시는 안정),
지시는 system에만 둔다. 문서 속 지시문을 따르지 말 것을 system에 명시한다. 거절(`stop_reason=="refusal"`)은 `LLMRefusalError`.

### 4.3 extract
```python
RuleExtractor().extract(doc, *, roster=(), catalog=()) -> Extraction                 # LLM 0회, 한국어 패턴
LlmExtractor(llm, *, max_chunk_chars=6000, concurrency=4).aextract(doc, *, roster=(), catalog=()) -> list[Extraction]  # 청크별
LlmExtractor(...).aextract_chunks(doc, ...) -> ExtractionRun(extractions, failed_chunks, calls)
assemble(extractions, docs, *, roster=(), catalog=(), oncall=(), policy=Policy()) -> AssemblyResult
    # AssemblyResult(graph: OpsGraph, grounding: GroundingStatus, held_ids: tuple[str,...], unresolved_people: tuple[str,...])
render_sentence(graph, element_id) -> str           # 결정적 한국어 역서술
slot_check(rendered, quote) -> SlotCheck             # 숫자·단위·인물·채널·부정어 슬롯 일치 (결정적)
async verify_backtranslation(graph, docs, llm: LLMClient | None) -> VerificationResult(graph, held_ids, checks)
```
정렬 순서: 청크 내 정확 일치 → 문서 전체 정확 일치 → 공백 정규화 일치 → rapidfuzz `partial_ratio_alignment` ≥ 90 → 실패 시 폐기.
격리 구간과 겹치는 span은 폐기(`dropped_quarantined`). 인물 해석: 정확 이름 → 별칭 → "성+직급"(김대리) 유일 매칭 → 실패 시
`PersonStatus.UNKNOWN`(명부 미확인), 다중 매칭 시 HOLD. 명부의 모든 사람/역할/팀/연락수단과 카탈로그 서비스·의존·위치를 그래프로 가져온다
(출처 ROSTER/CATALOG).

### 4.4 check (LLM 금지)
```python
CheckContext(graph, policy, roster=(), oncall=(), catalog=(), now=None)
  .critical_services() .resolve_people(nid) .is_active(person) .contact_channels(nid)
  .channel_failure_domain(channel_id) .owners(sid) .escalations(nid) .mitigations(sid) .approvers(action_id) .triggers(sid)
@register class SomeRule: info: RuleInfo; def check(self, ctx) -> Iterable[Finding]
run_checks(ctx, rule_ids: Iterable[str] | None = None) -> list[Finding]     # 결정적 정렬
rule_catalog() -> list[RuleInfo]
score(ctx, findings) -> ReadinessReport                                     # SCORING_V1 (docs/scoring.md 공개)
blast_radius(graph, failed_ids) -> ImpactResult                            # 센터 1곳 장애 시 영향 서비스
response_cut_sets(ctx, service_id, max_size=2) -> list[frozenset[str]]     # 대응 경로 최소 절단 집합(SPOF)
```
규칙 ID 체계: `JT-PATH-*`(필수 경로), `JT-PPL-*`(사람), `JT-ESC-*`(상향), `JT-CHN-*`(채널), `JT-DDL-*`(시한),
`JT-KR-*`(국내 신고 기준), `JT-ONC-*`(당직), `JT-CAT-*`(카탈로그 대조), `JT-INT-*`(판단 보류/무결성).

### 4.5 rehearse
```python
template_scenarios(ctx, *, service_ids=None) -> list[Scenario]   # 기준·담당 부재·주 채널 다운·승인자 부재·센터 다운·동시 장애
async llm_scenarios(llm, ctx, *, n=3) -> list[Scenario]          # 스키마·존재 id 검증 실패분 폐기
replay(graph, scenario, *, policy=Policy(), config=SimConfig(), roster=()) -> RehearsalResult   # LLM 금지, 결정적
replay_all(graph, scenarios, *, policy, config, roster=()) -> list[RehearsalResult]
```

### 4.5.1 explore (규칙 밖 위험 탐색 — [explore.md](explore.md))
```python
sweep(ctx, findings, *, config=None, limits=SweepLimits()) -> SweepResult   # LLM 0회: 핵심 서비스별 최소 끊김 조합(1·2개)
perturbation_effect(ctx, perturbations, *, config=None) -> frozenset[str]     # 리허설에서 빠지는 사람·채널(동시 장애 = 끌려가는 사람)
known_breaks(ctx, swept, drafts, *, config=None) -> frozenset[str]           # 이미 찾은 끊김 조합으로 설명되는 가설 id
async propose_hypotheses(llm, ctx, findings, *, n=3, known=None, config=None) -> AiProposals  # 열거된 id 중 선택만, 전제는 assumptions
verify_hypotheses(ctx, drafts, *, config=None) -> tuple[RiskHypothesis, ...]   # 서비스 정책 시한으로 재생 → 상태
explore_graph(ctx, findings, *, config=None, previous=()) -> Exploration       # 재판정용(LLM 없음)
```
가설은 끊김·점수·통과 기준에 반영하지 않는다(보고서의 별도 필드 `risk_hypotheses`). 전제가 있거나, 판단 보류 대상을 가리키거나,
교란 없이도 실패하는 서비스의 가설은 `needs_info`, 리허설이 끊기면 `confirmed`, 시한 안에 완화하면 `refuted`.

### 4.6 improve
```python
apply_ops(graph, ops) -> tuple[OpsGraph, tuple[PatchOp, ...]]   # previous/snapshot 채워 역연산 가능
invert_ops(ops) -> tuple[PatchOp, ...]
graph_diff(a, b) -> GraphDiff
TemplateDrafter().draft(ctx, findings, *, workspace_id, base_version) -> list[Patch]          # LLM 0회
LlmDrafter(llm).adraft(ctx, findings, *, workspace_id, base_version) -> list[Patch]          # 문장만 AI, op는 코드가 컴파일
validate_patch(ctx, patch, *, scenarios, config=SimConfig()) -> PatchValidation              # 재검사+과거 리허설 무회귀+명부 외 인물 0
VersionService(graphs, patches, audit, clock).init / approve / reject / revert / history
```
LLM 초안 안전장치: LLM에게는 끊김 설명과 **명부에서 뽑은 후보 목록(enum)**만 준다. LLM은 후보 중 선택 + 한국어 문장만 쓰고,
그래프 op는 `FixHint`+선택값으로 코드가 생성한다(LLM이 임의 노드를 만들 수 없음 → "명부 외 인물 생성 0건").

### 4.7 store / orchestrator / api / cli
- `store.memory.MemoryStore` (모든 Repository 프로토콜 구현), `store.sql.SqlStore(url)` (동일 프로토콜, 낙관적 동시성)
- `orchestrator.pipeline.build_diagnosis_graph()` — LangGraph `StateGraph`
  노드: `guard → extract → assemble → verify → check → score → scenarios → rehearse → explore → draft → finalize`
  LLM 실패 시 `extract`는 규칙 파서로 폴백하고 보고서에 기록.
- `orchestrator.service.JettaeService` — API·CLI가 쓰는 유일한 진입점
- API: `/api/v1/...` (§6), CLI: `jettae scan|rehearse|serve|demo|eval|rules`

## 4.8 계약 메모
- `AttrKey.CHANNEL`: 추출 IR에서는 채널 **이름**, 조립된 그래프에서는 채널 **노드 id**.
- `OnCallShift.start/end`는 timezone-aware(입력이 naive면 Asia/Seoul).
- 저장소 not-found는 `domain.ports.NotFoundError`(KeyError 하위).
- 승인 필요 여부의 단일 규칙(`jettae.check.context.requires_approval`): `requires_approval` 명시값 → 없으면 rollback/failover는 필요 → 그 외는 불필요. 승인자 엣지만으로는 '승인 필요'로 보지 않는다(승인자를 추가했는데 점수가 내려가는 일을 막기 위해). 점수·SPOF·리허설이 모두 이 함수 하나를 쓴다.
- 조직 전체 신고 의무: 들어오는 `MUST_REPORT`가 없는 OBLIGATION은 조직 전체 의무다. `CheckContext.obligations(sid)`는 서비스 의무와 조직 의무를 합쳐 돌려준다.
- 사람이 쓴 리허설 시나리오: `rehearse.scenarios_from_yaml(src, ctx)`(사람 이름·채널 매체·시스템/위치 이름 → 노드 id, 모르는 대상은 한국어 `ValueError`).
- API 서버의 이벤트 루프: Windows에서는 `jettae.cli.loops:resilient_event_loop`(소켓 쌍 생성 거부 시 재시도)를 uvicorn `loop=`로 넘긴다.

## 5. 보안·프라이버시
- 길이 보존 PII 마스킹(원문 비저장 옵션 `JETTAE_STORE_RAW_DOCUMENTS=false` 기본) · 연락처는 마스킹된 형태로만 저장
- 구조화 입력의 마스킹 범위(무료 검진): 카탈로그의 `description`·`owner`·`location`과 명부의 `title`·`team`·`aliases`는
  그래프에 넣기 전에 마스킹한다(`orchestrator.service.mask_catalog_free_text`·`mask_roster_free_text`). 명부의 이름·
  재직 상태·역할·연락 매체 종류·퇴사일과 카탈로그의 서비스 이름·별칭·의존 관계·종류·등급은 진단이 맞춰 보는 값이라
  가리지 않는다. 입력 파일은 진단 뒤 버리고, 그래프 속 값만 결과와 함께 보관 기간 뒤 삭제한다.
- 프롬프트 인젝션: (1) 격리(guard) (2) 경계 태그+nonce (3) 구조화 출력 (4) 열거형 선택만 허용 (5) 판정 경로 LLM 0회 —
  CaMeL(제어 흐름과 데이터 흐름 분리) 원칙. 자기 안전 시험 30건은 `evalkit.injection`이 배포마다 실행.
- 승인 RBAC: `lead`/`admin`만 승인·되돌리기. 모든 변경은 감사 로그.
- 인증 모드 3가지(`JETTAE_AUTH_MODE`): `dev`(헤더 역할, 로컬 전용) · `token`(Bearer 필수) · `public`(공개 베타 — 토큰이
  없으면 익명 viewer, 틀린 토큰은 401). 공개 모드의 익명 방문자는 예제와 접근 키(`X-Jettae-Workspace-Key`)를 보낸
  워크스페이스만 연다(`api.access.workspace_access`가 모든 라우터에서 `{ws}`·`{patch_id}`를 검사, 그 밖은 404).
  접근 키는 `JettaeService.grant_workspace_key`가 만들고 `store.access`(`workspace_access_keys`)에는 SHA-256 해시만 둔다.
- 사전 신청 명단은 `admin`만 조회·삭제(`GET /api/v1/signups`, `DELETE /api/v1/signups/{email}`, 감사 로그 범위 `_admin`).
  같은 이메일의 재신청은 기존 기록을 덮어쓰지 않는다(`SignupRepository.add_signup_if_absent`).

### 5.1 공개 베타(Render) — 위협 모델과 통제
배포 단위는 Render 웹 서비스 한 대(0.5 CPU · 512 MB, Docker, uvicorn 워커 1개)와 1 GB 무료 PostgreSQL(싱가포르)이다.
익명 쓰기 API(무료 검진·사전 신청·이벤트)가 열려 있으므로, 한 사람이 서버·DB를 독점하거나 개인정보가 필요 이상
남는 것을 막는 데 초점을 둔다. 상세 표는 [security.md](security.md), 운영 절차는 [deploy.md](deploy.md).

| 층 | 통제 | 위치 |
|---|---|---|
| 접근 | `public` 인증 모드, 접근 키(해시만 저장)가 없는 워크스페이스는 404, 토큰 표 검증 실패 시 기동 거부 | `api.security` · `api.access` |
| 부하 | 방문자 진단 동시 수(`JETTAE_MAX_CONCURRENT_SCANS`, 넘치면 즉시 503), 무료 검진 하루 한도(전체 `JETTAE_DAILY_FREE_SCAN_LIMIT`·접속 주소별), 이벤트·신청 빈도 제한, 사이트 전체 이벤트 하루 상한(`JETTAE_EVENTS_PER_DAY`) — 모두 프로세스 메모리. 방문자는 예제를 다시 진단하지 않고 저장된 보고서를 받음 | `api.ratelimit`(`RateLimiter`·`DailyCap`·`ScanSlots`) · `api.security.FreeScanQuota` · `api.routers.diagnosis` |
| 방문자 주소 | 접속 주소당 한도의 열쇠를 `JETTAE_CLIENT_IP_SOURCE`(`peer` · `header:<이름>` · `xff:<n>` 오른쪽에서 n번째)로 고름. 공개 사이트는 `header:CF-Connecting-IP`로 시작하고 배포 직후 `GET /api/v1/admin/request-info`(`admin`)로 실제 헤더를 확인. 헤더가 없으면 `peer`, IPv6는 `/64` 대역으로 셈 | `config.ClientIpSource` · `api.ratelimit.ClientIpResolver` · `api.routers.admin` · `render.yaml` |
| 입력 크기 | 업로드 MB(`JETTAE_MAX_UPLOAD_MB`), 변환 텍스트 글자 수(`JETTAE_MAX_DOCUMENT_CHARS`, 공개 사이트 50,000 — 진단 비용이 글자 수보다 빠르게 늘어남), 무료 검진 구조화 입력 각 1,000행(`MAX_FREE_SCAN_ROWS`), 파서 폭탄 가드(YAML 별칭 펼침 항목·깊이, DOCX 압축 해제, PDF 쪽·글자·스트림·연산, HTML 깊이·태그) | `api.middleware` · `api.routers.free_scans` · `ingest.loaders` · `ingest.structured` |
| 보관 | 무료 검진은 결과만 남기고 `JETTAE_FREE_SCAN_RETENTION_DAYS`(7) 뒤 워크스페이스째 삭제, 사전 신청은 `JETTAE_SIGNUP_RETENTION_DAYS`(365) 뒤 삭제 — 공개 모드(`JETTAE_AUTH_MODE=public`)에서만, 앱 수명 주기가 시작할 때와 `PURGE_INTERVAL_SECONDS`(6시간)마다 실행(개발·설치형 DB는 지우지 않음) | `JettaeService.purge_expired` · `WorkspaceRepository.delete_workspace` · `api.app` |
| 로그 | uvicorn 접속 로그 끔(IP·쿼리), 요청 로그는 메서드·경로·라우트·상태·시간만(경로 속 이메일은 가림), SQLAlchemy `hide_parameters=True`, DB 오류는 클래스 이름만 | `cli.main._serve` · `api.middleware` · `api.errors` · `store.sql` |
| 플랫폼 | DB 외부 접속 차단(`ipAllowList: []`), 비밀 값은 Render 환경 변수(`sync: false`), 의존성 `constraints.txt`(`==`) 고정·웹 `npm ci`, 이미지에 LLM SDK 없음, root 아닌 사용자 | `render.yaml` · `Dockerfile` |
| 고지 | `/privacy` 처리방침(항목·목적·보유 기간·Render 싱가포르 국외 이전·권리 행사), 공개 사이트는 LLM off | `web/src/pages/Privacy.tsx` |

접속 주소는 횟수 계산에만 쓰고 로그·DB에 남기지 않는다. Render는 `X-Forwarded-Proto`를 믿게 하려고
`FORWARDED_ALLOW_IPS="*"`를 켜는데, 그러면 uvicorn의 `request.client.host`(`peer`)가 `X-Forwarded-For`의 맨 왼쪽 값,
즉 방문자가 써 보낼 수 있는 값이 된다. 그래서 공개 사이트는 `peer` 대신 앞단 Cloudflare가 넣는 것으로 보이는
`CF-Connecting-IP`를 읽는다. 이 헤더는 Render가 문서로 보장하지 않으므로, 운영자가 배포 직후 `request-info`로 자기
공인 IP가 실린 칸을 확인하고 꾸민 헤더로 다시 불러 `resolved_client`가 바뀌지 않는지 본 뒤, 필요하면 `xff:<n>`으로
바꾼다([deploy.md](deploy.md) 4-2). 그래도 주소별 한도는 단순 반복을 막는 장치이고, 주소와 무관한 전체 한도·이벤트
하루 상한·동시 수·크기 한도·보관 기간이 마지막 방어선이다.

## 6. REST API (요약)
```
POST /api/v1/workspaces                                   {name} -> {id}
POST /api/v1/workspaces/{ws}/inputs        (multipart)    runbooks[], roster, oncall, catalog, policy
POST /api/v1/workspaces/{ws}/diagnose                      {llm_mode} -> DiagnosisReport
GET  /api/v1/workspaces/{ws}/graph?version=n              -> {version, graph, layout_hints}
GET  /api/v1/workspaces/{ws}/report                       -> DiagnosisReport (latest)
GET  /api/v1/workspaces/{ws}/report.md                    -> 증적 리포트(Markdown)
POST /api/v1/workspaces/{ws}/rehearsals                   {scenario_ids?|generate} -> RehearsalResult[]
POST /api/v1/workspaces/{ws}/patches/draft                {finding_ids?, llm_mode} -> Patch[]
POST /api/v1/patches/{id}/approve | /reject               (X-Jettae-Role: lead)
POST /api/v1/workspaces/{ws}/versions/{n}/revert
GET  /api/v1/workspaces/{ws}/versions | /diff?a=&b=
GET  /api/v1/rules                                        고정 규칙 카탈로그
POST /api/v1/demo/dmc-shop                                예제 워크스페이스 생성
POST /api/v1/free-scans                    (multipart)    files[], roster?, oncall?, catalog?, policy?
                                                          -> {workspace_id, access_key, readiness_overall, findings}
POST /api/v1/signups · POST /api/v1/events                 사전 신청·수요 측정(동의 필수)
GET  /api/v1/signups?format=json|csv                      사전 신청 명단(admin, CSV는 UTF-8 BOM)
DELETE /api/v1/signups/{email}                            사전 신청 삭제(admin, 동의 철회·삭제 요청 처리)
GET  /api/v1/metrics/demand                               수요 지표(lead 이상)
GET  /api/v1/me                                           현재 사용자·역할
GET  /api/v1/admin/request-info                           서버가 받은 접속 주소·프록시 헤더와 고른 방문자 주소(admin)
GET  /healthz · /readyz
```

## 7. 품질 게이트
`ruff check` · `ruff format --check` · `mypy --strict`(src) · `lint-imports` · `pytest`(hypothesis 속성 테스트 포함) ·
`jettae eval run --conditions A,B,O --smoke` · 웹 `npm run build` · GitHub Actions CI.
