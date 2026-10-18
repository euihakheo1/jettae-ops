# 실증 연구 제품 동결 기록 (realdata freeze)

동결 일자: 2026-10-18 · 기준 커밋: `b77f8bbc64d203a7288f970c169666d561144710` (아래 파일은 커밋되지 않은 작업 트리 상태) · 패키지 `jettae` 0.1.0

**규칙: 이 시점 이후 제품(추출·조립·검사·점수·프롬프트·루브릭)은 바꾸지 않는다.** 실증 ①의 test 분할 수치와
실증 ②(사후 보고서 회고) 수치는 아래 버전·해시와 같은 제품으로만 만들고 보고한다. 제품을 바꿔야 하면 이 기록을
대체하는 새 동결 기록(새 버전·해시)을 쓰고, 그 전에 만든 test/retro 결과는 새 제품의 수치로 보고하지 않는다.
설정 고정값(아래 '버전')이 다른 결과 폴더는 이어서 실행되지 않는다.

## 버전

| 항목 | 값 |
|---|---|
| 추출 프롬프트 digest (`extraction_prompt_digest()`) | `e83619532826b179` |
| 코딩 루브릭 (`RUBRIC_VERSION`) | `premise-claim-v4` |
| 결과 기록 형식 (`RECORD_VERSION`) | `record-v2` |
| 제품 동작 개정 (`PIPELINE_REVISION`) | `pipeline-v2` |
| 실증 ② 관할 규칙 (`JURISDICTION_RULE`) | `per-incident-front-matter-v1` (국가 미상 = `ZZ`) |
| 주 모델 / effort | `claude-sonnet-5-5` / `medium` |
| 코더 A / 코더 B / 코더 묶음 | `anthropic:claude-sonnet-5-5` / `openai:gpt-6.1-sol` / 지적 8건 |
| 실증 ① 관할(정책) | `US` (한국 법정 신고 규칙 JT-KR-* 미적용) |
| 실증 ① 분할 파일 digest | `441efb8bb3f3d1927f5eea6ba4a68710a7e412501d862e232a06e9528db040c1` (dev 10건 / test 20건) |

## 파일 해시 (sha256, 커밋되지 않은 작업 트리 파일 그대로의 바이트)

| 파일 | git 상태 | sha256 |
|---|---|---|
| `src/jettae/extract/llm_extractor.py` | 수정됨(미커밋) | `4060e59ff14030f88df7b298c499dde9c16d18d8429ba76bec77346a6c8de8d9` |
| `src/jettae/extract/assemble.py` | 수정됨(미커밋) | `8c6d9b61d8f4888ef8bbabe35d8c371fbcbd6dcbb6db0608ea6342784c02fa57` |
| `src/jettae/check/rules/path.py` | 수정됨(미커밋) | `0ba27ef1fba40d235feba1014fff4df516bbbba17fc4437ce5fffcc0beb031ab` |
| `src/jettae/evalkit/realdata/coding.py` | 추적 안 됨 | `1bcfcf9d58fe8e3b434490e2e81a09ca3dbdbca61fef001c5155799813f2438c` |

전체 소스 트리(`src/jettae/**/*.py` 139개, `__pycache__` 제외): 경로순으로 `<sha256>  <경로>` 줄을 만들어(줄마다
`\n`) 그 전체의 sha256 = `bebb1484e57a88956ac8866922c2e6ba47a011f448be8ebc3e03372b0e058127`.

다시 확인하기 (Git Bash, 저장소 루트):

```bash
sha256sum src/jettae/extract/llm_extractor.py src/jettae/extract/assemble.py \
  src/jettae/check/rules/path.py src/jettae/evalkit/realdata/coding.py
```

## 동결 시점 검증

- 데모(LLM 끔, `examples/dmc-shop`): 그래프 해시 `883b5d40d60c…`, 종합 점수 55.7 — 바뀌지 않음.
- `tests/test_integration_demo.py`, `tests/test_orchestrator_demo.py`, `tests/test_domain_graph_builder.py`,
  `tests/test_evalkit*.py`: 478개 통과. `tests/` 전체: 4,001개 중 4,000개 통과, 1개 건너뜀(`psycopg` 미설치).
- ruff·ruff format·mypy strict·import-linter(계약 3개 유지) 통과.

## dev 이력 (실증 ①, dev 분할 10건, 같은 모델 조합)

정밀도 = 코더가 판정한 지적 중 '맞음'의 비율(불명확 제외), 괄호는 Wilson 95% 신뢰구간. '두 코더 일치' = 두 코더가
같은 판정(맞음/틀림)을 낸 지적만 센 것. 루브릭 판이 회차마다 달라 회차 간 비교는 참고용이다.

| 회차 | 결과 폴더 | 루브릭 | 지적(끊김/확인 불가) | 코더 A 정밀도 | 코더 B 정밀도 | 두 코더 일치 정밀도 | κ / α (라벨) | κ / α (전제) | κ / α (주장) | 지출(상한) |
|---|---|---|---|---|---|---|---|---|---|---|
| r1 | `gitlab-dev` | v1 | 65 (57/8) | 71.4% (45/63, 59.3–81.1) | 55.4% (36/65, 43.3–66.8) | 68.9% (31/45, 54.3–80.5) | 0.376 / 0.365 | 0.000 / −0.024 | 0.464 / 0.445 | $0.5389 ($1.50) |
| r2 | `gitlab-dev-r2` | v3 | 66 (48/18) | 75.8% (47/62, 63.8–84.8) | 72.7% (48/66, 61.0–82.0) | 74.6% (44/59, 62.2–83.9) | 0.748 / 0.749 | 1.000 / 1.000 | 0.702 / 0.703 | $0.6406 ($1.00) |
| r3 | `gitlab-dev-r3` | v4 | 79 (57/22) | 79.7% (63/79, 69.6–87.1) | 69.6% (55/79, 58.8–78.7) | 78.3% (54/69, 67.2–86.4) | 0.670 / 0.667 | 0.000 / −0.047 | 0.670 / 0.667 | $0.7575 ($1.00) |

dev 누적 지출: $1.9370. r3이 이 동결 버전으로 만든 마지막 dev 실행이다(같은 프롬프트 digest, `pipeline-v2`,
`premise-claim-v4`).

r3에서 남은 dev 실패 유형(동결 후 고치지 않음 — 보고서의 한계로 적는다):

- 문서 전체에 적용되는 감지 신호(공급자 장애 공지, 클러스터 헬스체크 실패)가 서비스별 감지로 이어지지 않아 감지 없음
  지적 8건을 두 코더가 모두 틀렸다고 판정(재해 복구 문서).
- 서비스가 거치는 시스템에 적힌 완화 조치(트래픽 드레인)와 재구축형 완화가 서비스의 완화로 읽히지 않아 완화 없음 지적
  7건을 두 코더가 모두 틀렸다고 판정(같은 문서).
- 절차 문서가 의존한다고만 적은 도구를 서비스로 읽은 지적 8건에서 두 코더의 판정이 갈림(대응 절차 안내 문서).

## 동결 버전의 비용 추정 (드라이런, 무료 count_tokens만 사용)

| 연구 | 대상 | 결과 폴더 | 추정 합계(선택 조합) |
|---|---|---|---|
| 실증 ① test 분할 | 런북 20건, 추출 청크 26개 | `gitlab-test` | $6.60 |
| 실증 ② 사후 보고서 회고 | 보고서 27건 (관할 KR 5건, ZZ 22건 — 모두 front matter `language` 기준) | `retro` | $42.90 |

유료 test/retro 실행은 이 기록을 쓴 시점에 하지 않았다.
