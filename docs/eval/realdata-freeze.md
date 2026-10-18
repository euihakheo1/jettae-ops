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

## 공개 저장소에서 확인하기 (줄바꿈 무관)

동결한 코드는 main의 커밋 `ccf3b4d`(`ccf3b4da8844b5f735827dfd0278f154b09a5d12`, 2026-10-18 10:11)로 공개했다. 이 커밋은
유료 test(09:22–09:27)·회고(09:28–10:04) 실행이 끝난 뒤에 올렸으므로, 실행 전 동결을 외부 타임스탬프로 증명하지는 못한다
(근거는 아래 끝 문단의 작업 트리 해시 재확인과 결과 폴더 `run.json`의 설정값). 위 표의 파일
해시는 Windows 작업 트리(`core.autocrlf=true`)에서 잰 바이트 그대로의 값이다. 이 작업 트리에서는 `src/jettae/**/*.py`
139개 중 48개가 CRLF 줄바꿈으로 풀려 있고 `coding.py`도 그중 하나라서, 다른 컴퓨터에서 받은 사본으로는 같은 값이 나오지
않을 수 있다. 그래서 다시 확인할 수 있는 기준은 **LF로 정규화한 내용 = git blob 내용**의 해시다.

| 파일 | sha256 (LF 정규화 = `ccf3b4d`의 git blob) |
|---|---|
| `src/jettae/extract/llm_extractor.py` | `4060e59ff14030f88df7b298c499dde9c16d18d8429ba76bec77346a6c8de8d9` |
| `src/jettae/extract/assemble.py` | `8c6d9b61d8f4888ef8bbabe35d8c371fbcbd6dcbb6db0608ea6342784c02fa57` |
| `src/jettae/check/rules/path.py` | `0ba27ef1fba40d235feba1014fff4df516bbbba17fc4437ce5fffcc0beb031ab` |
| `src/jettae/evalkit/realdata/coding.py` | `0d1a78004d48feb8aeefe7c737ddcd0a61b36f90a5b067acd9249198ffe0d4d2` |

앞의 세 파일은 작업 트리에서도 LF라 위 표와 값이 같고, `coding.py`만 다르다(작업 트리 CRLF `1bcfcf9d…`, LF `0d1a7800…`).

전체 소스 트리(LF): `ccf3b4d`의 `src/jettae/**/*.py` 139개(git이 추적하는 파일, 위 작업 트리 값과 같은 집합)를 경로순으로
놓고 `<sha256>  <경로>` 줄(줄마다 `\n`)을 만들어 그 전체의 sha256 =
`e0d6a3feadd594a91060dd8f0c0fb3d12460fb6b2bd833488def892920e79c83`.
파일별 목록은 [`realdata-freeze-lf.sha256`](realdata-freeze-lf.sha256)이다(LF 줄바꿈 그대로일 때 이 목록 파일의 sha256이
곧 위 값이다). 저장소의 `.gitattributes`가 이 파일을 `eol=lf`로 지정해 Windows(`core.autocrlf=true`)에서 받아도 CRLF로
바뀌지 않는다. 그래도 작업 트리 사본이 의심되면 `git cat-file blob HEAD:docs/eval/realdata-freeze-lf.sha256`으로 커밋된
내용을 직접 읽어 비교한다.

다시 확인하기 (저장소를 받은 뒤 Git Bash·Linux, 저장소 루트 — 작업 트리의 줄바꿈과 상관없이 git blob을 읽는다):

```bash
git ls-tree -r --name-only ccf3b4d -- src/jettae | grep '\.py$' | LC_ALL=C sort | while read -r f; do printf '%s  %s\n' "$(git cat-file blob "ccf3b4d:$f" | sha256sum | cut -d' ' -f1)" "$f"; done | sha256sum
```

출력의 해시가 `e0d6a3feadd594a91060dd8f0c0fb3d12460fb6b2bd833488def892920e79c83`이면 같은 코드다(뒤에 Linux는 `-`,
Git Bash는 `*-`를 붙인다). 파일별로 보려면 마지막
`| sha256sum`을 `| diff - <(git cat-file blob HEAD:docs/eval/realdata-freeze-lf.sha256)`으로 바꾼다(작업 트리의 줄바꿈과
상관없이 커밋된 목록과 비교한다. macOS는 `sha256sum` 대신 `shasum -a 256`).
파일 목록을 `git ls-files 'src/jettae/**/*.py'`로 만들면 git 기본 pathspec이 `src/jettae/__init__.py` 같은 최상위 파일 2개를
빠뜨려 137개가 되므로 `git ls-tree`(또는 `':(glob)src/jettae/**/*.py'`)를 쓴다.

유료 test·retro 실행을 마친 뒤 같은 Windows 작업 트리에서 위 '파일 해시'의 전체 소스 트리 값을 다시 계산했고, 여전히
`bebb1484e57a88956ac8866922c2e6ba47a011f448be8ebc3e03372b0e058127`로 같았다. 결과 수치는
[realdata-results.md](realdata-results.md)에 있다.

## 봉인 평가셋 재측정 — 동결 코드로 (2026-10-18)

봉인셋 수치(규칙 파서 B: 활성 결함 탐지 96.5%, 보류 결함 100%, 정상 대조군 오탐 0%)는 10/14 코드(`d4acd3c`)로 낸 값이다. 동결(`ccf3b4d`)에서 판정·조립 코드가 바뀌었으므로, 같은 봉인셋(시드 20261012, 조직 20곳)을 동결 코드로 LLM 없이 다시 쟀다.

```bash
jettae eval run --conditions A,B,O --seed 20261012 --orgs 20 --out <임시 폴더> --no-benchmarks
python -I scripts/sealed_set_recheck.py --metrics <임시 폴더>/metrics.json
```

- 결과: 조건 A·B·O의 모든 지표(탐지율, 문체·연산자·위치별, 오탐, 필드 F1, 누락 필드 환각, 근거 없는 요소)가 공개 값과 같다. 다른 것은 지연 시간뿐이다. 재측정 지표 사본: `sealed-recheck-2026-10-18-metrics.json`.
- 봉인 해시가 달라 보이는 이유: 동결에서 정책 모델에 관할 필드가 생겨, 생성기가 쓰는 정책 YAML에 기본값 두 줄(`jurisdictions: [KR]`, 의무마다 `jurisdiction: KR`)이 더 적힌다. 그래서 동결 코드로 다시 만든 사례의 머클 루트는 `8c15d9db…`로, 공개 봉인 `23f7b84c…`와 다르다. 그 두 줄을 빼면 1,320건 모두 공개 봉인의 사례 해시와 같다(스크립트가 확인). 동결 전 코드(`8316fea`)로 만들면 `23f7b84c…`가 그대로 나온다.
- 한계: 봉인 해시는 결과와 같은 커밋(`d4acd3c`, 10/14)으로 처음 공개됐고, 약속(10.12 전)보다 2일 늦었다. 공개 뒤로 바뀌지 않았다는 것과, 공개된 생성기·시드로 누구나 같은 사례를 다시 만들 수 있다는 것은 증명된다. 결과보다 먼저 고정했다는 사전 등록 증거는 아니다.
