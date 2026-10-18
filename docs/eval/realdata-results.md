# 실데이터 실증 결과 — GitLab 공개 런북 · 공개 사후 보고서 회고

작성 2026-10-18 · 제품: 동결 버전(main 커밋 `ccf3b4d`, [동결 기록](realdata-freeze.md)) · 수치 출처: 결과 파일
([`realdata/`](realdata/)의 `*-metrics.json`, 그리고 공개하지 않는 `results.jsonl`)

> 이 문서의 수치는 모두 결과 파일에서 옮겼다. dev 분할 수치(r1–r3)는 제품을 고치면서 본 **튜닝 수치이고 결과가 아니다.**
> 실증 ②의 범위 안 같은 계열·요소 일치는 진단한 24건 기준 82.5%(33/40)로, 다른 사고의 지적과 맞춘 우연 기준선 82.2%와
> 같다. 하네스 보고서의 m/k 80.5%를 포함해 이 자동 일치는 **사고별 사전 탐지의 근거가 아니다.** 어떤 수치도 "제품이 그
> 사고를 막았을 것"이라는 뜻이 아니다.
> 실증 ①의 언어 대조(같은 20건의 한국어 충실 번역판, 두 코더 일치 71.8%)는 언어를 바꿔도 정밀도가 같은 수준이라는 대조
> 결과이고, 보고 수치 68.6%를 바꾸지 않는다. 문서별로 나오는 지적은 두 실행에서 꽤 달라 안정성 한계로 적었다.

## 한눈에 보기

| | 실증 ① GitLab 공개 런북 (test 분할) | 실증 ② 공개 사후 보고서 회고 |
|---|---|---|
| 물은 것 | 제품이 실제 운영 런북에서 낸 지적이 맞는가 (정밀도) | 보고서가 말한 대응 공백 중 제품 규칙 범위에 드는 것은 얼마인가, 사고 전 묶음에서 제품이 그 공백을 지적하는가 |
| 입력 | GitLab.com 운영 런북 20건 (MIT, 고정 커밋) | 공개 사후 보고서 27건 (한국 5 · 해외 22) |
| 핵심 수치 | 두 코더 일치 정밀도 **68.6%** (35/51, 95% CI 55.0–79.7%) | 공백 N=378 중 제품 규칙 범위 안 k=41 — **10.8%** |
| 함께 볼 수치 | 코더 A 64.4% · 코더 B 67.8% · 지적 59건 · 20건 중 9건은 지적 0건 | m=33 (하네스 m/k 80.5%) — 진단한 24건 기준 82.5%(33/40)가 우연 기준선 82.2%와 같아 근거가 아님 |
| 언어 대조 | 같은 20건의 한국어 번역판(언어만 바꿈): 두 코더 일치 71.8% (28/39) — 원문과 같은 수준. 같은 지적은 33건뿐(원문 59 · 번역판 45) | — |
| 코더 일치도 | 판정 κ 0.698 · α 0.700 (상당한 일치) | 공백→계열 코딩 κ 0.185 · α 0.142 (약함) |
| 비용 | $0.7322 (언어 대조 $0.7031 별도) | $11.7547 |
| 사람 검토 | 2026-10-22, 개발자 1인 블라인드 (지적 59건 전부) | 같은 날 (범위 안 단위 41개 엄격 일치 등) |

실데이터 API 지출 합계 약 $15.5 = 실증 ① test $0.7322 + 실증 ① 언어 대조 $0.7031 + dev r1–r3 $1.937 + 파일럿 $0.159 +
디버깅 조사 약 $0.22 + 실증 ② $11.7547.

**실증 ②에서 정직하게 말할 수 있는 것은 세 가지다.** ① 공개 사후 보고서가 진술한 대응 공백 가운데 제품 규칙 계열에 드는
것은 약 11%(대부분 PATH, 두 코더가 모두 동의한 값 — 코더별로는 13~40%로 갈린다)뿐이다 — 제품 범위의 실제 한계다. ② 재구성부터 진단·코딩까지 전체 파이프라인이 한국·해외 실제
보고서에서 끝까지 돈다. ③ 같은 서비스·사람을 가리키는지까지 보는 엄격 일치는 10/22 블라인드 사람 검토에서 잰다.

---

## 실증 ① GitLab 공개 런북

### 설계

| 항목 | 내용 |
|---|---|
| 원자료 | [gitlab.com/gitlab-com/runbooks](https://gitlab.com/gitlab-com/runbooks) (GitLab.com 운영팀 런북), 고정 커밋 `844a2b882687492cf117cc2fc553f7fbf9a02197`, MIT 라이선스(Copyright GitLab, Inc.). 장애 대응 절차(호출·상향·승인·복구·대체 경로)가 담겼을 문서 30건을 골랐다(incidents, incident-io, cloudflare, disaster-recovery, gamedays, frontend, alerts, patroni 등). 원문은 고치지 않았다. |
| 분할 | dev 10건 / test 20건 — [`gitlab-runbooks-split.json`](gitlab-runbooks-split.json), 분할 digest `441efb8bb3f3…`(배정 내용을 정규화한 JSON의 sha256, 전체 값은 [동결 기록](realdata-freeze.md)) |
| 분할 규칙 | 파일럿·디버깅 조사에서 이미 결과를 본 6건은 dev에 강제 배정하고, 나머지를 경로순으로 정렬해 `random.Random(20261017).sample`로 dev가 10건이 될 때까지 채운다. 나머지는 모두 test다. |
| 동결 | 제품 수정은 dev만 보고 했다. test와 실증 ② 실행 전에 제품을 동결했다([realdata-freeze.md](realdata-freeze.md)). 동결 코드는 main의 커밋 `ccf3b4d`(2026-10-18)로 공개했다. 순서를 밝혀 둔다: 동결 기록과 작업 트리 해시(`bebb1484…`)는 test 유료 실행(10/18 09:22–09:27) 전에 로컬에서 만들었고, 공개 커밋 `ccf3b4d`는 실증 ② 실행(09:28–10:04)이 끝난 뒤인 10:11에 올렸다. 그래서 "실행 전 동결"을 외부 타임스탬프로 증명할 수는 없다. 근거는 실행 뒤 다시 잰 작업 트리 해시가 같다는 것과, 결과 폴더 `run.json`의 값(test: 프롬프트 digest `e83619532826b179` · `pipeline-v2` · `premise-claim-v4`, 회고: `pipeline-v2`)이 동결 값과 같다는 것이다. |
| 제품 | 추출 모델 `claude-sonnet-5-5` (effort medium), 추출 프롬프트 digest `e83619532826b179`, `pipeline-v2` |
| 코더 | A `anthropic:claude-sonnet-5-5`, B `openai:gpt-6.1-sol` — 서로 다른 두 모델 계열. 요청 하나에 지적 8건씩 판정 |
| 판정 루브릭 | `premise-claim-v4`. 지적마다 두 질문에 따로 답한다: ① 전제 — 지적이 다루는 서비스·요소를 문서가 정말 다루는가, ② 주장 — 문서에 비추어 그 공백이 정말 있는가. 둘 다 yes면 맞음(correct), 하나라도 no면 틀림(incorrect). |
| 관할(정책) | US — 한국 법정 신고 규칙(JT-KR-*)은 적용하지 않았다. |
| 입력 조건 | 런북만 넣었다(명부·카탈로그 없음, 정책은 기본값). 영어 원문이라 제품의 역번역 LLM 대조는 생략하고 결정적 슬롯 대조만 적용했다. |

정밀도 = 맞음 / (맞음 + 틀림), 괄호는 Wilson 95% 신뢰구간. '두 코더 일치' = 두 코더가 같은 판정을 낸 지적만 센 값이다.

### 결과 (test 20건만)

- 런북 20/20건 완료. 지적 **59건** — 상태별 끊김(broken) 55 · 확인 불가(unknown) 4, 심각도별 critical 25 · high 27 · medium 7.
- 20건 중 **9건은 지적 0건**이다(아래 '지적 0건 런북').
- LLM 추출 호출 26회, 추출 일부 실패 0건. LLM이 제안한 항목 132개 중 조립 단계에서 근거 없음 등으로 버린 요소 1개.
  판단 보류(HOLD) 요소 2개 — 둘 다 결정적 슬롯 대조(인물·역할)의 불일치다(제품은 이를 '역번역 대조 불일치'로 표기한다).
  이 2개가 INT 지적 2건(`JT-INT-001`)이 되었고, 두 코더 모두 틀림으로 판정했다.

| 범위 | 코더 A | 코더 B | 두 코더 일치 |
|---|---|---|---|
| 전체 (59건) | 64.4% (38/59, 51.7–75.4%) | 67.8% (40/59, 55.1–78.3%) | **68.6% (35/51, 55.0–79.7%)** |
| 상태 끊김 (55건) | 65.5% (36/55, 52.2–76.6%) | 69.1% (38/55, 56.0–79.7%) | 70.2% (33/47, 56.0–81.3%) |
| 상태 확인 불가 (4건) | 50.0% (2/4, 15.0–85.0%) | 50.0% (2/4, 15.0–85.0%) | 50.0% (2/4, 15.0–85.0%) |

두 코더 일치 51건 = 둘 다 맞음 35 + 둘 다 틀림 16. 판정이 갈린 8건은 일치 정밀도에서 빠진다.

**코더 일치도** (지적 59건, 두 코더 모두 판정):

| 대상 | 단순 일치율 | Cohen's κ | Krippendorff's α |
|---|---:|---:|---:|
| 판정(맞음/틀림) | 86.4% | 0.698 | 0.700 |
| ① 전제 | 93.2% | 0.633 | 0.632 |
| ② 주장 | 86.4% | 0.626 | 0.629 |

**전제·주장으로 나눠 보기:**

| 지표 | 코더 A | 코더 B | 두 코더 일치 |
|---|---|---|---|
| 전제 오류율 = 전제 no / (yes + no) | 13.6% (8/59, 7.0–24.5%) | 6.8% (4/59, 2.7–16.2%) | 7.3% (4/55, 2.9–17.3%) |
| 전제가 맞는 지적의 정밀도 = 주장 yes / (yes + no) | 74.5% (38/51, 61.1–84.5%) | 72.7% (40/55, 59.8–82.7%) | 77.8% (35/45, 63.7–87.5%) |

전제 오류(문서가 다루지 않는 서비스·요소를 지적)는 두 코더 일치 기준 7.3%로 작다. 틀린 지적은 주로 ② "그 공백이 정말
있는가"에서 나왔다 — 문서에 적혀 있는 것을 제품이 못 읽은 경우다.

**규칙 계열별:**

| 계열 | 지적 | 코더 A | 코더 B | 두 코더 일치 | κ (판정) |
|---|---:|---|---|---|---:|
| PATH (필수 대응 경로 5요소) | 57 | 66.7% (38/57, 53.7–77.5%) | 70.2% (40/57, 57.3–80.5%) | 71.4% (35/49, 57.6–82.2%) | 0.676 |
| INT (판단 보류·근거 무결성) | 2 | 0.0% (0/2) | 0.0% (0/2) | 0.0% (0/2, 0.0–65.8%) | 1.000 |

**확인 불가 지적.** 문서가 담당·연락 수단 같은 요소를 서비스 카탈로그나 호출 도구 설정으로 넘긴다고 적었으면 제품은 그 요소를
'없음'이 아니라 '확인 불가'로 낸다. test에서 4건이었고, 두 코더 모두 그중 2건을 맞음으로 판정했다(50.0%, 표본이 매우 작다).

**지적 0건 런북 (9건).** 모두 제품이 문서에서 검사할 서비스를 찾지 못해 5요소 검사를 하지 않은 경우다. 제품은 이를 정상으로
치지 않고 이유를 메모로 남긴다(검사 대상 없음 ≠ 정상).

| 런북 | 제품이 남긴 이유(요약) |
|---|---|
| `docs/frontend/gitlab-com-is-down.md` | 서비스·시스템을 하나도 읽지 못함 |
| `docs/gamedays/scenarios/zonal-outage-recovery.md` | 서비스·시스템을 하나도 읽지 못함 |
| `docs/uncategorized/namespace-restore.md` | 서비스·시스템을 하나도 읽지 못함 |
| `docs/cloudflare/troubleshooting.md` | 다루는 대상을 시스템·도구로만 적음, 알람이 감시하는 서비스가 없음 |
| `docs/incident-io/incident-management.md` | 다루는 대상을 시스템·도구로만 적음, 알람이 감시하는 서비스가 없음 |
| `docs/patroni/primary-unresponsive-assess-before-failover.md` | 다루는 대상을 시스템·도구로만 적음 |
| `docs/disaster-recovery/gameday.md` | 역할·팀·연락 수단만 다루는 절차·안내로 읽음 |
| `docs/frontend/high-error-rate.md` | 알람은 있지만 그 알람이 감시하는 서비스가 적혀 있지 않음 |
| `docs/frontend/ssh-maxstartups-breach.md` | 알람 런북에 감시 대상 서비스가 적혀 있지 않음 |

이 9건에서 제품이 놓친 끊김이 있었는지는 이 실증으로 측정되지 않는다(정밀도는 재현율이 아니다).

<details>
<summary>런북별 지적과 코더 판정 (20건)</summary>

| 런북 (GitLab, 고정 커밋) | 지적 | 둘 다 맞음 | 둘 다 틀림 | 판정 갈림 | USD |
|---|---:|---:|---:|---:|---:|
| [`docs/cloudflare/troubleshooting.md`](https://gitlab.com/gitlab-com/runbooks/-/blob/844a2b882687492cf117cc2fc553f7fbf9a02197/docs/cloudflare/troubleshooting.md) | 0 | 0 | 0 | 0 | 0.0462 |
| [`docs/cloudflare/vendor.md`](https://gitlab.com/gitlab-com/runbooks/-/blob/844a2b882687492cf117cc2fc553f7fbf9a02197/docs/cloudflare/vendor.md) | 4 | 3 | 0 | 1 | 0.0514 |
| [`docs/disaster-recovery/README.md`](https://gitlab.com/gitlab-com/runbooks/-/blob/844a2b882687492cf117cc2fc553f7fbf9a02197/docs/disaster-recovery/README.md) | 4 | 3 | 1 | 0 | 0.0365 |
| [`docs/disaster-recovery/gameday.md`](https://gitlab.com/gitlab-com/runbooks/-/blob/844a2b882687492cf117cc2fc553f7fbf9a02197/docs/disaster-recovery/gameday.md) | 0 | 0 | 0 | 0 | 0.0238 |
| [`docs/frontend/asset-proxy-down.md`](https://gitlab.com/gitlab-com/runbooks/-/blob/844a2b882687492cf117cc2fc553f7fbf9a02197/docs/frontend/asset-proxy-down.md) | 4 | 3 | 1 | 0 | 0.0420 |
| [`docs/frontend/gitlab-com-is-down.md`](https://gitlab.com/gitlab-com/runbooks/-/blob/844a2b882687492cf117cc2fc553f7fbf9a02197/docs/frontend/gitlab-com-is-down.md) | 0 | 0 | 0 | 0 | 0.0085 |
| [`docs/frontend/high-error-rate.md`](https://gitlab.com/gitlab-com/runbooks/-/blob/844a2b882687492cf117cc2fc553f7fbf9a02197/docs/frontend/high-error-rate.md) | 0 | 0 | 0 | 0 | 0.0132 |
| [`docs/frontend/ssh-maxstartups-breach.md`](https://gitlab.com/gitlab-com/runbooks/-/blob/844a2b882687492cf117cc2fc553f7fbf9a02197/docs/frontend/ssh-maxstartups-breach.md) | 0 | 0 | 0 | 0 | 0.0211 |
| [`docs/gamedays/scenarios/zonal-outage-recovery.md`](https://gitlab.com/gitlab-com/runbooks/-/blob/844a2b882687492cf117cc2fc553f7fbf9a02197/docs/gamedays/scenarios/zonal-outage-recovery.md) | 0 | 0 | 0 | 0 | 0.0091 |
| [`docs/incident-io/incident-management.md`](https://gitlab.com/gitlab-com/runbooks/-/blob/844a2b882687492cf117cc2fc553f7fbf9a02197/docs/incident-io/incident-management.md) | 0 | 0 | 0 | 0 | 0.0157 |
| [`docs/incident-io/oncall.md`](https://gitlab.com/gitlab-com/runbooks/-/blob/844a2b882687492cf117cc2fc553f7fbf9a02197/docs/incident-io/oncall.md) | 1 | 0 | 1 | 0 | 0.0539 |
| [`docs/incidents/when-gitlab-com-is-down.md`](https://gitlab.com/gitlab-com/runbooks/-/blob/844a2b882687492cf117cc2fc553f7fbf9a02197/docs/incidents/when-gitlab-com-is-down.md) | 8 | 6 | 1 | 1 | 0.0527 |
| [`docs/patroni/primary-unresponsive-assess-before-failover.md`](https://gitlab.com/gitlab-com/runbooks/-/blob/844a2b882687492cf117cc2fc553f7fbf9a02197/docs/patroni/primary-unresponsive-assess-before-failover.md) | 0 | 0 | 0 | 0 | 0.0558 |
| [`docs/release-management/deploy_pressure.md`](https://gitlab.com/gitlab-com/runbooks/-/blob/844a2b882687492cf117cc2fc553f7fbf9a02197/docs/release-management/deploy_pressure.md) | 4 | 4 | 0 | 0 | 0.0354 |
| [`docs/security-patching/linux-os/notifications.md`](https://gitlab.com/gitlab-com/runbooks/-/blob/844a2b882687492cf117cc2fc553f7fbf9a02197/docs/security-patching/linux-os/notifications.md) | 8 | 4 | 4 | 0 | 0.0475 |
| [`docs/uncategorized/block-high-load-project.md`](https://gitlab.com/gitlab-com/runbooks/-/blob/844a2b882687492cf117cc2fc553f7fbf9a02197/docs/uncategorized/block-high-load-project.md) | 4 | 3 | 1 | 0 | 0.0352 |
| [`docs/uncategorized/externalvendors/GCP-rackspace-support.md`](https://gitlab.com/gitlab-com/runbooks/-/blob/844a2b882687492cf117cc2fc553f7fbf9a02197/docs/uncategorized/externalvendors/GCP-rackspace-support.md) | 13 | 5 | 2 | 6 | 0.0779 |
| [`docs/uncategorized/namespace-restore.md`](https://gitlab.com/gitlab-com/runbooks/-/blob/844a2b882687492cf117cc2fc553f7fbf9a02197/docs/uncategorized/namespace-restore.md) | 0 | 0 | 0 | 0 | 0.0273 |
| [`docs/uncategorized/upgrade-and-rollback.md`](https://gitlab.com/gitlab-com/runbooks/-/blob/844a2b882687492cf117cc2fc553f7fbf9a02197/docs/uncategorized/upgrade-and-rollback.md) | 5 | 1 | 4 | 0 | 0.0424 |
| [`docs/uncategorized/workers-high-load.md`](https://gitlab.com/gitlab-com/runbooks/-/blob/844a2b882687492cf117cc2fc553f7fbf9a02197/docs/uncategorized/workers-high-load.md) | 4 | 3 | 1 | 0 | 0.0367 |
| **합계** | **59** | **35** | **16** | **8** | **0.7322** |

</details>

생성된 보고서와 지표 원본: [`realdata/gitlab-test-report.md`](realdata/gitlab-test-report.md),
[`realdata/gitlab-test-metrics.json`](realdata/gitlab-test-metrics.json).

### dev 이력 — 튜닝 수치이며 결과가 아니다

아래는 dev 분할 10건으로 제품을 고치며 본 수치다. 이 문서들을 보고 제품을 고쳤으므로 dev 수치는 좋게 나오는 쪽으로 치우쳐
있고, **결과로 보고하지 않는다.** 보고 수치는 위의 test 68.6%다. r3(78.3%)보다 test가 낮은 것은 이 치우침으로 예상되는
일이다. 루브릭 판이 회차마다 달라 회차 간 비교도 참고용이다(자세한 표는 [동결 기록](realdata-freeze.md)).

| 회차 | 루브릭 | 지적 | 두 코더 일치 정밀도 (튜닝용) | κ (판정) | 지출 |
|---|---|---:|---|---:|---:|
| r1 | v1 | 65 | 68.9% (31/45) | 0.376 | $0.5389 |
| r2 | v3 | 66 | 74.6% (44/59) | 0.748 | $0.6406 |
| r3 | v4 (동결 버전) | 79 | 78.3% (54/69) | 0.670 | $0.7575 |

동결 시점에 남은 것으로 기록한 한계(고치지 않음):

- 문서 전체에 적용되는 감지 신호(공급자 장애 공지, 클러스터 헬스체크 실패)를 서비스별 감지로 잇지 못한다.
- 로드밸런서처럼 서비스가 거치는 시스템에 적힌 트래픽 드레인을 그 뒤 서비스의 완화로 읽지 않고, 재구축형 완화도 세지 않는다.
- 절차 문서가 의존한다고만 적은 도구(Slack, Incident.io)를 서비스로 읽을 때가 있다.

### 비용

- test: **$0.7322** (Claude $0.5640 · 38회, OpenAI $0.1682 · 12회). 5회로 나눠 실행했고 누적 상한 `--max-usd 2.50`은 올리지 않았다.
  같은 조합의 드라이런 추정은 $6.60이었다(캐시 할인을 빼고 OpenAI 토큰을 Claude 토큰 수로 대신한 보수적 추정).
- dev r1–r3: $1.937 · 파일럿: $0.159 · 디버깅 조사: 약 $0.22.

---

## 실증 ① 언어 대조 — 같은 20건의 한국어 번역판

### 왜 했나

- 신청서는 '공개 런북 번안 평가셋'을 약속했다. 실제로는 번안 대신 원문과 줄을 맞춘 **충실 번역**을 만들었다. 회사·도구를
  한국식으로 바꾸는 번안은 내용까지 바꿔 문서를 더 합성에 가깝게 만든다. 충실 번역이면 같은 문서에서 언어만 바뀐다.
- 실증 ①은 영어 문서라 제품의 한국어 경로(한국어 규칙 파서, 역번역 LLM 대조)를 거의 지나지 않았다. 번역판은 이 경로도
  함께 지난다.

### 방법

| 항목 | 내용 |
|---|---|
| 입력 | 실증 ① test 분할 20건의 한국어 번역판 — [`realdata/gitlab-runbooks-ko/`](realdata/gitlab-runbooks-ko/) (방법과 원문·번역 해시는 그 폴더의 [`SOURCE.md`](realdata/gitlab-runbooks-ko/SOURCE.md)). MIT 원문(고정 커밋 `844a2b88…`)의 2차 저작물이며 출처는 GitLab, Inc.다. |
| 번역 | 2026-10-18, Claude Code 세션의 Claude 에이전트가 번역했다 — 제품 추출 모델(Claude Sonnet 5.5)·코더 A와 같은 모델 계열이다. 원문과 줄 수·줄 번호를 맞췄다. 코드 블록·인라인 코드·URL·명령어·서비스/도구/사람/채널 이름·숫자는 원문 그대로 두었다. 내용 추가·삭제·오류 수정은 하지 않았고, 회사·도구를 바꾸는 번안도 하지 않았다. |
| 검사 | 구조 검사기(줄 수, 코드 블록, 줄 앞 마크다운 구조, 빈 줄, URL, 링크 대상, 인라인 코드, 줄마다 숫자) 실패 0건. 번역하지 않은 별도 검토 에이전트가 줄마다 대조해 7건을 찾았고 모두 고쳤다. 남은 경고 두 가지: 제목을 번역해 문서 안 앵커 링크 12개가 번역판에서는 맞지 않고, 원문의 닫히지 않은 코드 블록 안 2줄은 규칙대로 영어로 두었다. |
| 제품 | 동결 버전 그대로(`src`는 커밋 `ccf3b4d` 이후 바뀌지 않음). 설정은 실증 ① test와 같다: 추출 `claude-sonnet-5-5`(effort medium), 코더 A `anthropic:claude-sonnet-5-5`, 코더 B `openai:gpt-6.1-sol`, 루브릭 `premise-claim-v4`, 관할 US, `--split test`. |
| 실행 | 2026-10-18, 4건씩 5회로 나눠 실행, 누적 상한 `--max-usd 3.00`(드라이런 추정 $9.46). 실제 지출 **$0.7031** (Claude $0.5641 · 36회, OpenAI $0.1391 · 10회). |

줄 번호가 원문과 같아서, 번역판 지적의 근거 링크(GitLab 원문의 같은 줄)로 바로 대조할 수 있다.

### 결과 — 원문과 나란히

| 지표 | 원문 (영어, 실증 ①) | 한국어 번역판 |
|---|---|---|
| 런북 | 20/20건 | 20/20건 |
| 지적 | 59건 (끊김 55 · 확인 불가 4) | 45건 (끊김 40 · 확인 불가 5) |
| 지적 0건 런북 | 9건 | 11건 |
| **두 코더 일치 정밀도** | **68.6% (35/51, 55.0–79.7%)** | **71.8% (28/39, 56.2–83.5%)** |
| 코더 A | 64.4% (38/59, 51.7–75.4%) | 73.3% (33/45, 59.0–84.0%) |
| 코더 B | 67.8% (40/59, 55.1–78.3%) | 64.4% (29/45, 49.8–76.8%) |
| 판정 일치도 (단순 일치율 · κ · α) | 86.4% · 0.698 · 0.700 | 86.7% · 0.692 · 0.692 |
| 전제가 맞는 지적의 정밀도 (두 코더 일치) | 77.8% (35/45, 63.7–87.5%) | 75.7% (28/37, 59.9–86.6%) |
| 상태 끊김 (두 코더 일치) | 70.2% (33/47, 56.0–81.3%) | 79.4% (27/34, 63.2–89.6%) |
| 상태 확인 불가 (두 코더 일치) | 50.0% (2/4, 15.0–85.0%) | 20.0% (1/5, 3.6–62.5%) |
| INT 지적 (두 코더 일치) | 2건, 0/2 | 4건, 0/4 |
| 판단 보류(HOLD) 요소 | 2개 | 4개 |
| LLM 호출 (제품) | 추출 26 | 추출 24 · 역번역 2 |
| 지출 | $0.7322 | $0.7031 |

- 번역판의 판단 보류 4개는 모두 '역번역 대조 불일치'다. 2개는 결정적 슬롯 대조(인물·역할, 부정어)가, 2개는 LLM 역번역
  대조가 낸 것이다. 4개 모두 INT 지적(`JT-INT-001`)이 되었고 두 코더 모두 틀림으로 판정했다.
- 제품은 번역판 20건 중 2건(`docs/frontend/asset-proxy-down.md`, `docs/uncategorized/namespace-restore.md` — 코드 비중이
  큰 문서)을 한국어가 아닌 문서로 분류해, 이 2건에서는 역번역 LLM 대조를 건너뛰었다.

### 문서별 지적 — 같은 지적은 33건

같은 지적 = 같은 규칙, 같은 서비스(대소문자 무시), 같은 요소. 원문 59건 중 33건이 번역판에도 나왔고, 원문에만 나온 지적이
26건, 번역판에만 나온 지적이 12건이다.

| 런북 | 지적 (원문 → 번역판) | 같은 지적 |
|---|---:|---|
| `docs/cloudflare/troubleshooting.md` | 0 → 4 | 0 |
| `docs/cloudflare/vendor.md` | 4 → 4 | 4 (전부) |
| `docs/disaster-recovery/README.md` | 4 → 0 | 0 |
| `docs/frontend/asset-proxy-down.md` | 4 → 4 | 4 (전부) |
| `docs/incident-io/oncall.md` | 1 → 3 | 1 (번역판에 `JT-INT-001` 2건 추가) |
| `docs/incidents/when-gitlab-com-is-down.md` | 8 → 8 | 8 (전부) |
| `docs/release-management/deploy_pressure.md` | 4 → 4 | 4 (전부) |
| `docs/security-patching/linux-os/notifications.md` | 8 → 4 | 4 |
| `docs/uncategorized/block-high-load-project.md` | 4 → 4 | 3 (롤백 지적이 원문 `JT-PATH-040`, 번역판 `JT-PATH-041`) |
| `docs/uncategorized/externalvendors/GCP-rackspace-support.md` | 13 → 0 | 0 |
| `docs/uncategorized/upgrade-and-rollback.md` | 5 → 10 | 5 |
| `docs/uncategorized/workers-high-load.md` | 4 → 0 | 0 |
| 나머지 8건 | 0 → 0 | — |
| **합계** | **59 → 45** | **33** |

지적 0건 런북은 원문 9건, 번역판 11건이다.

### 해석

- **정밀도는 같은 수준이다.** 두 코더 일치 정밀도는 원문 68.6%, 번역판 71.8%이고, 차이는 두 신뢰구간 안에 넉넉히 들어간다.
  언어를 한국어로 바꿔도 정밀도는 낮아지지 않았다. 차이가 이만큼 작으므로 번역판이 더 낫다고도 말하지 않는다.
- **문서별로 어떤 지적이 나오는지는 꽤 달라졌다.** 같은 지적은 33건뿐이고, 한쪽 실행에서만 지적이 나온 런북이 4건이다
  (GCP-rackspace-support 13 → 0 등).
- **이 실행으로는 언어 효과와 실행 간 변동을 가르지 못한다.** 영어 원문 실행을 반복하지 않았으므로, 이 차이 중 얼마가
  언어 때문이고 얼마가 LLM 추출이 실행마다 달라지는 탓인지 알 수 없다. 안정성 한계로 밝힌다(아래 '한계').
- 제품이 코드 비중이 큰 번역판 2건을 한국어가 아닌 문서로 분류해 그 2건에서는 역번역 LLM 대조가 돌지 않았다.
- 이 대조는 실증 ① 결과를 대신하지 않는다. 보고 수치는 원문 test 68.6%다.

생성된 보고서와 지표 원본: [`realdata/gitlab-test-ko-report.md`](realdata/gitlab-test-ko-report.md),
[`realdata/gitlab-test-ko-metrics.json`](realdata/gitlab-test-ko-metrics.json). 보고서의 '한계' 절 첫 줄('영어 문서')은
도구가 늘 쓰는 고정 문구라 이 실행에는 맞지 않는다(입력은 한국어 번역판이다). '사람 검토 표본' 절도 도구가 자동으로 쓴 안내라, 번역판 지적은 10/22 사람 검토 대상이 아니다(아래 '한계').

---

## 실증 ② 공개 사후 보고서 회고

### 설계

**보고서 27건.** 한국 5건 — 카카오 판교 데이터센터 화재(과기정통부 보고서), 카카오톡 연속 장애(과기정통부), KT 전국 인터넷
장애(과기정통부), 국가정보자원관리원 대전 화재(감사원), 지방행정 전산망(새올) 장애(감사원) — 와 해외 22건. 사고별 관할은
보고서 front matter의 `language`로 정했다: KR 5건, ZZ(국가 미상, 한국 아님) 22건. 한국 법정 신고 규칙(JT-KR-*)은 KR
사고에만 적용된다.

**사고마다 다섯 단계** (모두 동결 제품과 같은 커밋의 `jettae eval retro`):

1. **재구성.** `claude-sonnet-5-5`가 보고서에서 사고 *전*의 묶음(런북·명부·카탈로그·정책 — 제품이 원래 받는 파일)을 복원한다.
   사고 자체, 결과, 사후 개선은 쓰지 말라고 지시한다.
2. **결과 누설 검사.** 같은 모델이 복원한 문장 중 사고·결과·사후 개선·사후 지식을 담은 문장을 표시하고, 표시된 문장은
   결정적으로 지운다.
3. **진단.** 동결 제품이 남은 묶음을 진단한다(지적 목록).
4. **공백 코딩.** 코더 A·B가 각자 보고서가 진술한 '대응 과정' 공백을 나열하고, 공백마다 사고 전에 그것을 지적했을 제품 규칙
   계열 하나 또는 `out_of_scope`(어느 규칙으로도 사고 전 문서에서 알 수 없음)를 붙인다. 인용 구간이 겹치는 공백끼리 단위로
   묶고, 상대만 나열한 단위는 인용만 보고(상대의 라벨은 보지 않고) 교차 코딩한다. 그래서 두 코더가 모든 단위를 코딩한다.
5. **일치 판정.** 결정적 규칙(아래)으로 단위마다 일치 여부를 정한다.

**N · k · m.**

- N = 두 코더가 모두 공백으로 코딩한 단위(어느 쪽도 `not_a_gap`이 아님)
- k = N 중 두 코더가 모두 제품 규칙 계열로 코딩한 단위(어느 쪽도 `out_of_scope`가 아님)
- m = k 중 재구성 묶음의 제품 지적과 일치한 단위

**일치 규칙 (느슨함).** 단위의 두 코딩(코더 A의 계열·요소, 코더 B의 계열·요소) *각각에 대해* 다음을 만족하는 제품 지적이
하나 이상 있으면 일치다: 지적의 계열이 코딩한 계열과 같고, 요소가 같거나 어느 한쪽 요소가 null이다. 지적이 같은 서비스나
같은 사람을 가리키는지는 **보지 않는다.** 코드 그대로(`src/jettae/evalkit/realdata/retro.py`):

```python
def _finding_matches(finding: FindingRecord, family: str, element: str | None) -> bool:
    return finding.family == family and (
        element is None or finding.element is None or finding.element == element
    )
```

`match_units`는 두 코딩 모두에 맞는 지적이 있을 때만 일치로 센다. 끊김(broken) 지적만으로 센 m_broken도 함께 계산한다.

### 결과

- 사고 27/27건 완료. 공백 단위 438개(두 코더가 모두 코딩) 중 **N = 378**.
- **k = 41** — N의 **10.8%**.
- **m = 33** — m/k 80.5% (33/41, 95% CI 66.0–89.8%). 끊김 지적만: 33. 하네스가 낸 값이며, 진단하지 않은 roblox의
  범위 안 단위 1개를 불일치로 센다. **진단한 24건만 보면 82.5%(33/40)이고, 아래 우연 기준선 82.2%와 같아 사고별 사전
  탐지의 근거가 아니다.**
- 한국: N=122, k=15, m=13 · 해외: N=256, k=26, m=20.

| 계열 | k | m |
|---|---:|---:|
| PATH (필수 대응 경로 5요소) | 31 | 29 |
| CHN (연락 채널) | 3 | 0 |
| CAT (카탈로그 대조) | 1 | 1 |
| (코더 간 계열 불일치) | 6 | 3 |

- 제품 지적 합계 509건 (PATH 395 · CAT 59 · ESC 22 · PPL 17 · KR 10 · INT 6), 사고당 중앙값 17건.
- 진단하지 않은 3건(복원 묶음에 런북 없음) — `cloudflare-2025-11-18`, `roblox-2021-10-28`, `square-2023-09-07`: 재구성
  묶음에 사고 전 런북 문장이 남지 않아(보고서가 사고 전 절차를 말하지 않음) 진단을 돌리지 않았다. 그래서 이 3건은 어떤
  공백과도 일치할 수 없다. 범위 안 단위는 roblox의 1개뿐이다.
- CHN 단위 3개는 m=0이다. 재구성 묶음 어디에서도 제품이 CHN 지적을 내지 않았다.

생성된 보고서와 지표 원본: [`realdata/retro-report.md`](realdata/retro-report.md),
[`realdata/retro-metrics.json`](realdata/retro-metrics.json), 사고별 요약 [`realdata/retro-incidents.csv`](realdata/retro-incidents.csv).

### 우연 기준선 — 자동 일치는 사고별 탐지의 근거가 아니다

질문: 그 사고의 묶음에서 나온 지적이, *다른* 사고의 묶음에서 나온 지적보다 그 사고의 공백과 더 잘 맞는가?

방법: 제품이 실제로 진단한 사고 24건의 범위 안 단위 40개 각각을, 같은 일치 규칙으로 진단한 다른 사고 23건 각각의 제품
지적과 대조해 일치 비율을 구하고 단위에 걸쳐 평균했다. 지적이 그 사고와 아무 관계가 없어도 나오는 m/k 수준이다.
진단하지 않은 3건(복원 묶음에 런북 없음)은 지적이 없어 어떤 공백과도 일치할 수 없으므로, 관측 쪽과 비교 대상 양쪽에서
뺐다. 재현: [`scripts/realdata_retro_baseline.py`](../../scripts/realdata_retro_baseline.py) (표준 라이브러리만,
`python -I`로 실행), 출력 [`realdata/retro-baseline.txt`](realdata/retro-baseline.txt) 2절.

| 계열 (진단한 24건) | k | 관측 m/k | 우연 기준선 |
|---|---:|---:|---:|
| 전체 | 40 | 82.5% (33/40) | **82.2%** |
| PATH | 30 | 96.7% (29/30) | 95.8% |
| (코더 간 계열 불일치) | 6 | 50% (3/6) | 54% |
| CHN | 3 | 0% (0/3) | 0% |
| CAT | 1 | 100% (1/1) | 91% |

- 우연일 때 기대되는 m은 32.9이고 관측 m은 33이다. 우연과 차이가 없다.
- **결론: 자동 계열·요소 일치는 사고별 사전 탐지에 대해 아무것도 말하지 않는다.** "제품이 그 사고의 공백을 사고 전에
  짚었다"는 근거가 아니며 헤드라인으로 쓰지 않는다. 이 수치가 보여 주는 것은 PATH 계열이 대부분의 재구성 묶음에서
  지적된다는 사실뿐이다. 같은 서비스·사람까지 맞는지(엄격 일치)는 10/22 사람 블라인드 검토(검토자는 개발자 1인, 밝혀
  보고)에서 잰다.
- 기준선이 높은 이유: 제품은 대부분의 묶음에서 PATH 5요소(detection·owner·approver·fallback_channel·rollback) 대부분을
  지적한다. 진단한 24건 중 23건에서 5요소 중 4개 이상을 지적했다(PATH 지적이 없는 1건은 nats). 느슨한 규칙에서는 같은 계열·요소의 지적이 묶음 어딘가에 있기만 하면 일치가 된다.
- 참고 — 하네스 값으로 견준 비교(치우침, 쓰지 않음): 하네스 보고서의 m/k 80.5%(33/41)는 진단하지 않은 roblox의 범위 안
  단위 1개를 불일치로 센 값이다. 이 값을 진단하지 않은 3건까지 비교 대상에 넣은 기준선 73.1%(기대 m 30.0)와 견주면 차이가
  있어 보이지만, 진단하지 않은 사고는 일치할 수 없어 기준선이 아래로 치우친 비교다. 그래서 이 비교로 80.5%가 우연보다
  높다고 말하지 않는다.

### 범위: 보고서가 말한 대응 공백의 약 11%만 제품 규칙 계열 안

- k/N = 41/378 = 10.8%. 범위 안 단위 41개 중 31개가 PATH다.
- 나머지 337개(약 89%)는 적어도 한 코더가 `out_of_scope`(사고 전 문서에서 어느 제품 규칙으로도 알 수 없는 공백)로 본 것이다. 공개 사후
  보고서가 말하는 대응 공백 대부분은 지금 제품이 보는 범위 밖에 있다 — 제품 범위의 실제 한계다.
- 이 비율은 코더에 따라 크게 다르다. N 378개 중 코더 A는 152개(40.2%), 코더 B는 50개(13.2%)를 범위 안으로 코딩했고,
  둘 다 범위 안으로 본 것이 41개다. 약 11%는 두 코더가 모두 동의한 것만 센 보수적인 값이며, 사람 검토로 확인한다.

### 코더 일치도와 비대칭

- 공백→계열 코딩 일치도(단위 438개): Cohen's κ **0.185**, Krippendorff's α **0.142**, 단순 일치율 57.5% — **약하다.**
  N·k·m은 두 LLM 코더의 합의에 기대므로 이 약한 일치도와 함께 읽어야 한다.
- 나열한 공백 수가 크게 다르다: 코더 A 169개(보고서당 최대 11개, 인용이 보고서에 없어 5개 제외), 코더 B 370개(보고서당
  최대 53개, 제외 0).
- 단위 438개 = 둘 다 나열 94 + A만 나열 69 + B만 나열 275. 상대만 나열한 단위는 교차 코딩으로 채웠다. N 378개 중 236개는
  B만 나열하고 A가 교차 코딩에서 공백이라고 동의한 단위다.
- 범위 판정도 비대칭이다: 단위 438개 중 코더 B는 367개, 코더 A는 237개를 `out_of_scope`로 코딩했다. k는 사실상 코더 B의
  범위 판정에 묶여 있다.

### 결과 누설 검사

복원한 런북 문장 793개 중 77개(9.7%)를 지웠다: 사고 진행 서술(incident_event) 39, 사후 지식(hindsight) 18, 결과(outcome) 13,
사후 개선(post_incident_change) 7. 누설 검사는 문장을 지울 뿐, "무엇을 쓰지 않았는가"에 스민 사후 지식은 지우지 못한다(한계 참고).

### 사고별 결과 (27건)

보고서 본문은 각 발행처의 저작물이라 이 저장소에는 출처 주소와 수치만 둔다. 제목은 짧게 줄인 설명이다(원 제목은
[`realdata/retro-incidents.csv`](realdata/retro-incidents.csv)).

| 사고 id | 관할 | 제목(요약) | 출처 | N | k | m | 제품 지적 | USD |
|---|---|---|---|---:|---:|---:|---:|---:|
| `kakao-2022-pangyo-fire-msit` | KR | 카카오 판교 데이터센터 화재 (과기정통부 보고서) | [korea.kr](https://www.korea.kr/news/policyNewsView.do?newsId=156540855) | 13 | 4 | 2 | 16 | 0.3056 |
| `kakao-2024-05-kakaotalk-consecutive-outages-msit` | KR | 카카오톡 연속 장애 (과기정통부, 2024.5) | [eiec.kdi.re.kr](https://eiec.kdi.re.kr/policy/materialView.do?num=252148) | 6 | 0 | 0 | 7 | 0.1268 |
| `kt-2021-10-25-nationwide-internet-outage` | KR | KT 전국 인터넷 장애 (과기정통부, 2021.10) | [korea.kr](https://www.korea.kr/briefing/policyBriefingView.do?newsId=156477990) | 3 | 0 | 0 | 22 | 0.3114 |
| `nirs-2025-daejeon-fire-bai` | KR | 국가정보자원관리원 대전 화재 (감사원) | [bai.go.kr](https://www.bai.go.kr/bai/result/branch/detail?srno=3361) | 46 | 1 | 1 | 17 | 2.1700 |
| `saeol-2023-11-17-admin-network-outage-bai` | KR | 지방행정 전산망(새올) 장애 (감사원) | [bai.go.kr](https://www.bai.go.kr/bai/result/branch/detail?srno=3220) | 54 | 10 | 10 | 19 | 2.9345 |
| `atlassian-2022-04-site-deletion` | ZZ | Atlassian 사이트 삭제 장애 (2022.4) | [atlassian.com](https://www.atlassian.com/blog/how-we-build/post-incident-review-april-2022-outage) | 25 | 4 | 4 | 34 | 0.3397 |
| `aws-2017-02-28-s3-us-east-1` | ZZ | AWS S3 us-east-1 장애 (2017.2) | [aws.amazon.com](https://aws.amazon.com/message/41926/) | 2 | 0 | 0 | 31 | 0.0935 |
| `aws-2021-12-07-us-east-1` | ZZ | AWS us-east-1 장애 (2021.12) | [aws.amazon.com](https://aws.amazon.com/message/12721/) | 6 | 1 | 1 | 37 | 0.1749 |
| `azure-2023-08-30-australia-east-cooling` | ZZ | Azure 호주 동부 전력·냉각 장애 (2023.8) | [azure.status.microsoft](https://azure.status.microsoft/en-us/status/history/?trackingId=VVTQ-J98) | 6 | 0 | 0 | 48 | 0.1931 |
| `azure-2025-10-29-front-door` | ZZ | Azure Front Door 장애 (2025.10) | [azure.status.microsoft](https://azure.status.microsoft/en-us/status/history/?trackingId=YKYN-BWZ) | 3 | 1 | 1 | 49 | 0.1649 |
| `cloudflare-2019-07-02-waf-regex` | ZZ | Cloudflare WAF 정규식 장애 (2019.7) | [blog.cloudflare.com](https://blog.cloudflare.com/details-of-the-cloudflare-outage-on-july-2-2019/) | 7 | 0 | 0 | 33 | 0.2485 |
| `cloudflare-2023-11-02-pdx-control-plane` | ZZ | Cloudflare 제어 평면·분석 장애 (2023.11) | [blog.cloudflare.com](https://blog.cloudflare.com/post-mortem-on-cloudflare-control-plane-and-analytics-outage/) | 13 | 3 | 3 | 21 | 0.1644 |
| `cloudflare-2025-11-18-bot-feature-file` | ZZ | Cloudflare 봇 기능 파일 장애 (2025.11) | [blog.cloudflare.com](https://blog.cloudflare.com/18-november-2025-outage/) | 1 | 0 | 0 | 0 | 0.0839 |
| `datadog-2023-03-08-systemd` | ZZ | Datadog 다중 리전 연결 장애 (2023.3) | [datadoghq.com](https://www.datadoghq.com/blog/2023-03-08-multiregion-infrastructure-connectivity-issue/) | 2 | 0 | 0 | 10 | 0.1321 |
| `gcp-2025-06-12-service-control` | ZZ | Google Cloud Service Control 장애 (2025.6) | [status.cloud.google.com](https://status.cloud.google.com/incidents/ow5i3PPK96RduMcb1SsW) | 2 | 1 | 1 | 11 | 0.1656 |
| `github-2018-10-21-mysql-failover` | ZZ | GitHub MySQL 페일오버 장애 (2018.10) | [github.blog](https://github.blog/news-insights/company-news/oct21-post-incident-analysis/) | 5 | 0 | 0 | 16 | 0.1502 |
| `gitlab-2017-01-31-db-deletion` | ZZ | GitLab 데이터베이스 삭제 장애 (2017.1) | [about.gitlab.com](https://about.gitlab.com/blog/postmortem-of-database-outage-of-january-31/) | 7 | 3 | 3 | 10 | 0.1964 |
| `heroku-2025-06-10-platform-outage` | ZZ | Heroku 플랫폼 장애 (2025.6) | [heroku.com](https://www.heroku.com/blog/summary-of-june-10-outage/) | 7 | 2 | 1 | 30 | 0.1473 |
| `meta-2021-10-04-backbone-bgp` | ZZ | Meta 백본·BGP 장애 (2021.10) | [engineering.fb.com](https://engineering.fb.com/2021/10/05/networking-traffic/outage-details/) | 4 | 0 | 0 | 10 | 0.0756 |
| `mizuho-2021-02-28-minori-atm` | ZZ | 미즈호은행 ATM 장애 (2021.2) | [mizuhogroup.com](https://www.mizuhogroup.com/global-news/2021-06-20210615-3release-eng) | 34 | 1 | 1 | 9 | 0.2764 |
| `nats-2023-08-28-fprsa-r` | ZZ | 영국 NATS 비행계획 시스템 장애 (2023.8) | [caa.co.uk](https://www.caa.co.uk/data-and-publications/publications/documents/content/cap2993/) | 50 | 1 | 0 | 1 | 0.8350 |
| `pagerduty-2025-08-28-kafka` | ZZ | PagerDuty Kafka 장애 (2025.8) | [pagerduty.com](https://www.pagerduty.com/eng/august-28-kafka-outages-what-happened-and-how-were-improving/) | 16 | 0 | 0 | 30 | 0.1951 |
| `roblox-2021-10-28-consul-73h` | ZZ | Roblox Consul 73시간 장애 (2021.10) | [about.roblox.com](https://about.roblox.com/newsroom/2022/01/roblox-return-to-service-10-28-10-31-2021/) | 4 | 1 | 0 | 0 | 0.1544 |
| `rogers-2022-07-08-core-outage` | ZZ | 캐나다 Rogers 코어망 장애 (2022.7) | [publications.gc.ca](https://publications.gc.ca/site/eng/9.933245/publication.html) | 27 | 5 | 2 | 11 | 0.7583 |
| `slack-2021-01-04-tgw` | ZZ | Slack 장애 (2021.1) | [slack.engineering](https://slack.engineering/slacks-outage-on-january-4th-2021/) | 3 | 1 | 1 | 20 | 0.1162 |
| `square-2023-09-07-dns` | ZZ | Square DNS 장애 (2023.9) | [developer.squareup.com](https://developer.squareup.com/blog/incident-summary-2023-09-07/) | 2 | 0 | 0 | 0 | 0.0642 |
| `tsb-2018-04-migration` | ZZ | TSB 은행 시스템 이전 장애 (2018.4) | [fca.org.uk](https://www.fca.org.uk/publication/final-notices/tsb-bank-plc-2022.pdf) | 30 | 2 | 2 | 17 | 1.1769 |
| **합계 27건** | KR 5 · ZZ 22 | | | **378** | **41** | **33** | **509** | **11.7547** |

### 받지 못했거나 뺀 자료

자동 수집에서 403·무응답은 재시도·우회하지 않고 기록만 했다. 아래 4건은 입력에 넣지 않았다.

| 사고 | 출처 | 상태 |
|---|---|---|
| Knight Capital (2012-08-01) | [SEC Release 34-70694](https://www.sec.gov/litigation/admin/2013/34-70694.pdf) | 차단 — HTTP 403 |
| AT&T Mobility 전국 무선 장애 (2024-02-22) | [FCC PSHSB 보고서](https://docs.fcc.gov/public/attachments/DOC-404150A1.pdf) | 차단 — HTTP 403 |
| Optus Triple Zero 장애 (2025-09-18) | [Optus 독립 검토 보고서](https://www.optus.com.au/content/dam/optus/cloud/documents/about-us/media-centre/speeches-and-reports/2025/Independent_Report_Triple_Zero_Outage_at_Optus_18_September_2025.pdf) | 차단 — 응답 없음(시간 초과) |
| Heathrow North Hyde 변전소 화재 (2025-03-21) | [Heathrow 보도자료](https://mediacentre.heathrow.com/pressrelease/detail/22887) | 보류 — 보도자료 본문만 받음(보고서 PDF는 양식 제출로만 제공되어 받지 않음), 본문이 짧아 입력에서 뺌 |

### 비용

- **$11.7547** (Claude $8.6304 · 145회, OpenAI $3.1243 · 53회). 같은 조합의 드라이런 추정은 $42.90이었다(보수적 추정).
- 28회로 나눠 실행했다(아래 '재현 방법'). 마지막 1회는 새 항목 없이 27건 전체로 지표만 다시 계산했다($0).

---

## 봉인 합성 평가셋과의 관계

봉인 평가셋(1,320건, [report.md](report.md))은 LLM을 끈 규칙 파서 경로(조건 B)를 잰다: 활성 결함 탐지 96.5%, 보류 결함(개발에
쓰지 않은 2종) 100%, 정상 대조군 오탐 0%, 문체별 표 100% · 명시형 100% · 구어체 89.4%, 누락 필드 환각 1.9%, 프롬프트 인젝션 30건 중 무단
변경 0건, 재현성 100%. 봉인셋의 LLM 조건(C·D·DR)은 실행하지 않았다(Opus 기준 약 $80 추정). 대신 LLM 파이프라인은 이
문서의 실제 문서 두 실증으로 쟀다. 봉인셋 LLM 조건 실행은 선택 사항이며 일정에 없다.

## 사람 검토 계획 (2026-10-22)

- 약 2시간, 블라인드: 검토자는 코더 판정을 보지 않는다. 검토용 비공개 페이지는 따로 만든다.
- 검토자는 개발자 본인 1명이다(코더 판정만 가린다. 제품을 만든 사람이라는 이해 상충은 남는다). 결과는 '개발자 1인 블라인드
  검토'로 밝혀 보고한다.
- 잴 것:
  1. **실증 ① 지적 59건 전부의 사람 판정** — 사람 기준 정밀도와, 코더 A·B 판정이 사람 판정과 얼마나 맞는지(표본은
     `review.csv`/`review.json`, 시드 20261017, 규칙 계열별 층화 PATH 57 · INT 2. 검토 뒤 `finding_key`로 결과와 대조).
  2. **실증 ② 사람 계열 코딩** — 범위 안 41개와 범위 밖 337개 중 무작위 20개(시드 20261022), 모두 61개 단위를 사람이
     계열로 코딩해 그 단위들에서 코더 판정이 맞는지 본다(코더 결과로 고른 표본이라 438개 전체의 κ 0.185를 대신하지는 않는다).
  3. **실증 ② 엄격 일치** — 범위 안 단위 41개마다 일치한 지적이 정말 같은 서비스·같은 사람을 가리키는지.
  4. **실증 ② 범위 확인** — 위의 범위 밖 표본 20개로 범위 밖 판정이 맞는지.
- 결과가 나오면 이 문서의 수치를 갱신한다. 사람 결과가 나오기 전까지 위 수치는 LLM 코더 판정에 기댄 값이다.

## 한계

**실증 ①**

- **영어 문서.** 제품의 규칙 파서는 한국어용이라 영어 런북에서는 거의 기여하지 않고, 그래프는 사실상 LLM 추출이 만든다.
  같은 20건의 한국어 번역판으로 정밀도를 다시 쟀다(위 '언어 대조').
- **런북만 넣음.** 명부·카탈로그가 없어 사람 관련 검사는 설계대로 '확인 불가'가 되고, 등급을 모르는 서비스는 핵심 등급으로
  보아 5요소를 모두 요구받는다(정책 기본값). 그런 지적은 메시지에 정책 기준임을 밝혔고, 코더는 그 전제를 주어진 것으로 보고 판정했다.
- **성숙한 SRE 조직.** GitLab은 호출·상향을 별도 시스템 설정으로 운영한다. 런북에 적히지 않은 것을 끊김으로 보는 것은 문서
  수준 판정이며 실제 조직의 대응 능력 판정이 아니다.
- **정밀도는 재현율이 아니다.** 코더는 제품이 낸 지적만 판정했다. 놓친 끊김(특히 지적 0건 런북 9건)은 측정되지 않는다.
- **작은 표본.** 런북 20건, 지적 59건 — 신뢰구간 폭(55.0–79.7%)이 크다.
- **코더도 LLM이다.** 코더 A는 제품 추출 모델과 같은 모델이라 자기 선호 가능성이 있다. 사람 검토(개발자 1인)로
  확인해 코더 값과 나란히 보고한다.
- 동결 시점에 기록한 남은 실패 유형 3가지(위 dev 이력)는 고치지 않은 채 test를 돌렸다.

**실증 ① 언어 대조**

- **실행 간 안정성.** 원문과 번역판에서 같은 지적은 33건뿐이다(원문 59 · 번역판 45). 영어 실행을 반복하지 않아 이 차이가
  언어 때문인지 LLM 추출의 실행 간 변동 때문인지 가르지 못한다. 실증 ①의 지적 목록도 한 번 실행한 결과로 읽어야 한다.
- **번역도 LLM이 했다.** 번역은 제품 추출 모델·코더 A와 같은 Claude 계열이 했다. 그 번역 문체가 추출이나 판정에 유리했을
  가능성을 배제하지 못한다.
- **충실 번역은 한국 조직 문서가 아니다.** 내용·조직·도구는 GitLab 그대로이고 언어만 바뀌었다. 한국 기업이 쓴 런북에서의
  정밀도는 이 대조로 알 수 없다.
- 번역판 지적 45건은 사람 검토 계획에 들어 있지 않다(코더 판정만 있음).

**실증 ②**

- **사후 확증 편향.** 사고 전 묶음은 사고를 다 아는 모델이 보고서에서 복원했다. 복원된 묶음은 실제 사고 전 문서보다 바로 그
  공백을 드러내기 쉽고, 누설 검사로도 "무엇을 쓰지 않았는가"에 스민 사후 지식은 지울 수 없다.
- **재구성은 원본이 아니다.** 실제 사고 전 런북이 아니라 보고서 진술을 옮긴 것이다.
- **느슨한 일치 규칙과 우연 기준선.** 같은 계열·요소면 일치로 보고 서비스·사람은 보지 않는다. 그래서 진단한 24건 기준 관측
  82.5%는 다른 사고의 지적으로도 나오는 우연 기준선 82.2%와 같고, 사고별 탐지의 근거가 아니다(하네스 m/k 80.5%도 마찬가지).
- **약한 코딩 일치도와 코더 비대칭.** κ 0.185. 범위 비율은 코더에 따라 13.2%–40.2%로 다르다.
- **보고서 표본.** 공개 사후 보고서는 공개를 택한 조직·사고만 담는다. 한국 보고서는 5건이며 감사·정부 보고서다.
  자동 수집이 막힌 3건과 본문이 짧은 1건은 빠졌다.
- **사고를 막았을 것이라는 주장은 하지 않는다.** m은 같은 계열·요소의 지적이 재구성 묶음 어딘가에 있었던 공백의 수일
  뿐이다(느슨한 일치). 우연 기준선과 같은 수준이라 "사고 전에 지적할 수 있었다"는 근거도 아니다.

## 재현 방법

준비: 저장소를 커밋 `ccf3b4d`로 받아 `pip install -e ".[anthropic,openai]"`, 키는 환경 변수 `ANTHROPIC_API_KEY`·`OPENAI_API_KEY`.
아래 유료 실행 명령은 실제 실행 스크립트에서 옮긴 것이다(Git Bash, 저장소 루트). 출력 필터와 완료 확인 같은 반복 부분은
줄였고, `--max-usd`에는 실제로 쓴 상한 값을 적었다. 드라이런은 같은 인자에 `--dry-run`을 붙인 것이다. 원자료 폴더
`C:/jettae/data/real/...`는 이 저장소에 없다 — GitLab 런북은 위 고정 커밋에서 받을 수 있고, 사고 보고서는 저작권 때문에
출처 주소만 공개한다.

공통 규칙:

- **드라이런 먼저.** `--dry-run`은 남은 항목의 비용만 추정한다(유료 호출 0회). `ANTHROPIC_API_KEY`가 있으면 무료
  `count_tokens`로 입력 토큰을 세고, 없으면 글자 수로 어림한다.
- 유료 실행은 `--max-usd`와 `--yes`가 모두 있어야 한다. `--max-usd`는 그 `--out` 폴더의 **누적** 상한이다(이전 실행 지출
  `spend.json` 포함). 남은 항목의 추정 비용이 남은 예산을 넘으면 시작하지 않고, 실제 지출이 상한에 닿으면 멈춘다.
- 같은 `--out`으로 다시 실행하면 끝난 항목은 건너뛰고 남은 항목만 처리한다. 입력 폴더를 끝난 항목을 모두 담은 더 큰 폴더(상위
  집합)로 바꾸면 새 항목만 처리하고 지표를 전체로 다시 계산한다.

### 실증 ① test

```bash
jettae eval real-runbooks --input C:/jettae/data/real/gitlab-runbooks --out C:/jettae/data/real/results/gitlab-test \
  --split test --model claude-sonnet-5-5 --coder-a anthropic:claude-sonnet-5-5 --coder-b openai:gpt-6.1-sol --dry-run
```

```bash
for L in 4 8 12 16 20; do
  jettae eval real-runbooks --input C:/jettae/data/real/gitlab-runbooks --out C:/jettae/data/real/results/gitlab-test \
    --split test --model claude-sonnet-5-5 --coder-a anthropic:claude-sonnet-5-5 --coder-b openai:gpt-6.1-sol \
    --max-usd 2.50 --limit $L --yes
done
```

`--limit`은 test 20건을 경로순으로 앞에서부터 자르므로 한 번에 4건씩 늘어난다. 누적 상한 $2.50은 5회 내내 같았다.
dev 회차는 같은 명령에 `--split dev`, 결과 폴더 `gitlab-dev`·`gitlab-dev-r2`·`gitlab-dev-r3`으로 돌렸다(누적 상한은
회차마다 $1.50 · $1.00 · $1.00).

### 실증 ① 언어 대조 (한국어 번역판)

입력 폴더(번역판)와 결과 폴더(`gitlab-test-ko`), 누적 상한(`--max-usd 3.00`, test는 2.50)만 다르고 나머지 인자는 test와 같다. 번역판은 이 저장소의
[`docs/eval/realdata/gitlab-runbooks-ko/`](realdata/gitlab-runbooks-ko/)에 있다(실행할 때 쓴 폴더는 같은 내용의
`C:/jettae/data/real/gitlab-runbooks-ko`).

```bash
jettae eval real-runbooks --input C:/jettae/data/real/gitlab-runbooks-ko --out C:/jettae/data/real/results/gitlab-test-ko \
  --split test --model claude-sonnet-5-5 --coder-a anthropic:claude-sonnet-5-5 --coder-b openai:gpt-6.1-sol --dry-run
```

```bash
for L in 4 8 12 16 20; do
  jettae eval real-runbooks --input C:/jettae/data/real/gitlab-runbooks-ko --out C:/jettae/data/real/results/gitlab-test-ko \
    --split test --model claude-sonnet-5-5 --coder-a anthropic:claude-sonnet-5-5 --coder-b openai:gpt-6.1-sol \
    --max-usd 3.00 --limit $L --yes
done
```

누적 상한 $3.00은 5회 내내 같았다. 원문과 번역판의 문서별 지적 대조(같은 규칙·서비스·요소)는 두 결과 폴더의
`results.jsonl`에서 계산했다.

### 실증 ② 회고 — 나눠 실행과 스테이징 폴더

남은 잔액 안에서 한 번에 보고서 하나씩, 큰 보고서는 그 보고서만의 추정치로 상한을 걸어 돌렸다. 도구는 끝난 보고서를 모두 담은
입력 폴더(상위 집합)에서만 이어서 실행하고 `--limit`은 경로순으로 자르므로, 순서를 직접 정하려고 **스테이징 폴더**
(`incident-reports/stage`)에 원본의 바이트 그대로의 복사본을 하나씩 넣으며 키웠다.

```bash
R=C:/jettae/data/real/incident-reports
ARGS="--out C:/jettae/data/real/results/retro --model claude-sonnet-5-5 \
  --coder-a anthropic:claude-sonnet-5-5 --coder-b openai:gpt-6.1-sol"

# 0) 드라이런: 27건 전체 추정 $42.90
jettae eval retro --reports $R/reports $ARGS --dry-run

# 1) B안 폴더(9건)의 경로순 앞 6건을 하나씩 — 누적 상한 $4.00 (실행 1–6)
for L in 1 2 3 4 5 6; do jettae eval retro --reports $R/reports-b $ARGS --max-usd 4.00 --limit $L --yes; done

# 2) 스테이징: 끝난 6건을 복사한 뒤 중·소형 보고서를 작은 것부터 하나씩 추가 — 누적 상한 $4.50 (실행 7–19, 13건)
mkdir -p $R/stage
for f in atlassian-2022-04-site-deletion cloudflare-2023-11-02-pdx-control-plane gitlab-2017-01-31-db-deletion \
         kakao-2022-pangyo-fire-msit kakao-2024-05-kakaotalk-consecutive-outages-msit \
         kt-2021-10-25-nationwide-internet-outage; do cp $R/reports/$f.md $R/stage/; done
for f in aws-2017-02-28-s3-us-east-1 meta-2021-10-04-backbone-bgp square-2023-09-07-dns slack-2021-01-04-tgw \
         datadog-2023-03-08-systemd heroku-2025-06-10-platform-outage azure-2025-10-29-front-door \
         github-2018-10-21-mysql-failover cloudflare-2025-11-18-bot-feature-file gcp-2025-06-12-service-control \
         pagerduty-2025-08-28-kafka azure-2023-08-30-australia-east-cooling aws-2021-12-07-us-east-1; do
  cp $R/reports/$f.md $R/stage/
  jettae eval retro --reports $R/stage $ARGS --max-usd 4.50 --yes
done
# (cloudflare-2019-07-02-waf-regex는 이 단계에서 유료 호출 전에 Windows 소켓 오류로 실패해 스테이징에서 뺐다가 4단계에서 다시 넣었다)

# 3) 큰 보고서 5건을 하나씩 — 상한 = 그때까지 누적 지출 + 그 보고서의 드라이런 추정 + $0.25 (실행 20–24)
cp $R/reports/rogers-2022-07-08-core-outage.md $R/stage/;             jettae eval retro --reports $R/stage $ARGS --max-usd 6.53 --yes
cp $R/reports/nats-2023-08-28-fprsa-r.md $R/stage/;                   jettae eval retro --reports $R/stage $ARGS --max-usd 7.73 --yes
cp $R/reports/tsb-2018-04-migration.md $R/stage/;                     jettae eval retro --reports $R/stage $ARGS --max-usd 9.89 --yes
cp $R/reports/nirs-2025-daejeon-fire-bai.md $R/stage/;                jettae eval retro --reports $R/stage $ARGS --max-usd 12.41 --yes
cp $R/reports/saeol-2023-11-17-admin-network-outage-bai.md $R/stage/; jettae eval retro --reports $R/stage $ARGS --max-usd 16.99 --yes

# 4) 남은 3건 — 같은 상한 규칙 (실행 25–27)
cp $R/reports/cloudflare-2019-07-02-waf-regex.md $R/stage/; jettae eval retro --reports $R/stage $ARGS --max-usd 12.33 --yes
cp $R/reports/roblox-2021-10-28-consul-73h.md $R/stage/;    jettae eval retro --reports $R/stage $ARGS --max-usd 12.77 --yes
cp $R/reports/mizuho-2021-02-28-minori-atm.md $R/stage/;    jettae eval retro --reports $R/stage $ARGS --max-usd 13.03 --yes

# 5) 실제 27건 폴더로 마지막 실행 — 새 항목 0건, $0, 27건 전체로 지표·보고서 재계산 (실행 28)
jettae eval retro --reports $R/reports $ARGS --max-usd 12.25 --yes
```

3단계 직후 한 번 실제 27건 폴더로 실행을 시도했지만, 남은 3건의 추정 비용($2.91)이 남은 예산($0.50)을 넘어 도구가 시작을
거부했다(유료 호출 없음, 지출 기록에 남지 않음). 실행별 상한·지출은 결과 폴더의 `spend.json`에 남아 있다.

### 우연 기준선

```bash
python -I scripts/realdata_retro_baseline.py C:/jettae/data/real/results/retro/results.jsonl \
  --out docs/eval/realdata/retro-baseline.txt
```

스크립트는 저장된 단위 판정을 같은 규칙으로 다시 계산해 맞는지 확인하고(다르면 종료 코드 1), N·k·m, 우연 기준선(진단한
사고만 쓴 주 비교, 그리고 진단하지 않은 사고까지 넣어 아래로 치우친 참고 비교), PATH 5요소 지적 범위를 낸다. `results.jsonl`은 보고서 인용과 재구성 묶음 글을 담고 있어 공개하지 않는다.

## 공개한 파일 (`docs/eval/realdata/`)

| 파일 | 내용 |
|---|---|
| [`gitlab-test-metrics.json`](realdata/gitlab-test-metrics.json) | 실증 ① test 지표 (도구가 쓴 원본 그대로) |
| [`gitlab-test-report.md`](realdata/gitlab-test-report.md) | 실증 ① test 보고서 (도구가 쓴 원본 그대로) |
| [`gitlab-test-ko-metrics.json`](realdata/gitlab-test-ko-metrics.json) | 실증 ① 언어 대조(한국어 번역판) 지표 (원본 그대로) |
| [`gitlab-test-ko-report.md`](realdata/gitlab-test-ko-report.md) | 실증 ① 언어 대조 보고서 (원본 그대로 — '한계' 첫 줄 '영어 문서'는 도구의 고정 문구라 이 실행에 맞지 않음) |
| [`gitlab-runbooks-ko/`](realdata/gitlab-runbooks-ko/) | test 20건의 한국어 충실 번역판(`docs/**`), 출처·방법·해시(`SOURCE.md`), 원문 MIT 라이선스(`LICENSE`) — 출처 GitLab, Inc. |
| [`retro-metrics.json`](realdata/retro-metrics.json) | 실증 ② 지표 (원본 그대로) |
| [`retro-report.md`](realdata/retro-report.md) | 실증 ② 보고서 (도구 원본에 맨 위 주의 문구 한 줄만 더함 — m/k는 사고별 사전 탐지의 근거가 아님) |
| [`retro-incidents.csv`](realdata/retro-incidents.csv) | 사고별 요약: 관할, 원 제목, 출처 주소, N·k·m, 단위·공백 수, 지적 수, 누설 검사, 비용 (보고서 본문·묶음 글 없음) |
| [`retro-baseline.txt`](realdata/retro-baseline.txt) | 우연 기준선 스크립트 출력 |
| [`README.md`](realdata/README.md) | 이 폴더 안내 (원본 보고서만 읽을 때의 주의) |

공개하지 않는 것: `results.jsonl`(보고서 인용·재구성 묶음 글), `bundles/`(재구성 묶음), 사람 검토 표본(검토 전 블라인드 유지),
언어 대조 결과 폴더의 `results.jsonl`·`review.*`.
