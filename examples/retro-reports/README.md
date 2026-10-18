# 실증 ② 입력 형식 — 장애 사후 보고서 폴더

`jettae eval retro --reports <이 폴더> --out <결과 폴더>`가 읽는 형식입니다.

- 보고서 하나 = 파일 하나: `<사고-id>.md` 또는 `<사고-id>.txt` (파일 이름이 사고 id, 이름순으로 처리).
- 맨 위 `README*` 파일은 보고서가 아니라 이 폴더의 설명으로 보고 건너뜁니다.
- 본문은 공개된 보고서 원문 그대로 둡니다(언어 무관). 맨 앞에 YAML front matter를 둘 수 있습니다.

```markdown
---
title: Database failover took 3 hours          # 보고서 제목
source_url: https://example.com/postmortem      # 원문이 공개된 곳
organisation: Example Corp                      # 보고서를 낸 조직
published: 2025-01-31                           # 공개일
license: CC BY 4.0                              # 원문 라이선스(재배포 가능 여부 확인용)
country: US                                     # 선택: 조직의 관할(ISO 3166-1 두 글자, jurisdiction:도 같음)
language: en                                    # 선택: 보고서 언어
---
<보고서 원문>
```

- **관할은 사고마다 정합니다.** `jurisdiction:`/`country:`가 있으면 그 코드, 없으면 한국어 보고서
  (`language: ko`, 언어 표기가 없으면 본문이 한국어)는 `KR`, 그 밖은 `ZZ`(미상·한국 아님)로 진단합니다.
  한국 법정 신고 규칙(JT-KR-*)은 `KR` 사고에만 적용되고, 사고별 관할과 그 근거는 `results.jsonl`에 남습니다.

`fictional-parcel-db-outage.md`는 형식과 테스트를 위한 **가상의** 보고서입니다(실존 조직·사건 아님).
실제 보고서를 모으면 이 폴더 대신 새 폴더를 만들고 `--reports`로 지정하세요.

## 나눠서 실행하기 — 일부(예: B안 9건)를 먼저, 나머지(예: 18건)는 나중에

같은 `--out` 폴더로 다시 실행하면 이미 끝난 사고는 `results.jsonl`에 있어 **다시 처리·과금하지 않고**,
남은 사고만 처리합니다. `--reports`를 끝난 사고를 모두(같은 내용으로) 담은 더 큰 폴더(상위 집합)로
바꿔도 됩니다 — 새 사고만 처리하고 N·k·m·κ는 전체(예: 27건)로 다시 계산합니다.
두 번째 실행에도 **첫 실행과 같은 `--model`·`--effort`·`--coder-a`·`--coder-b`·`--bundle-language`**를
그대로 붙이세요(빠뜨리면 기본값과 달라 `run.json` 검사에서 시작을 거부합니다).

```powershell
$m = "--model", "claude-sonnet-5-5", "--coder-a", "anthropic:claude-sonnet-5-5", "--coder-b", "openai:gpt-6.1-sol"
# 1) B안 9건 먼저 (먼저 --dry-run으로 비용 확인)
jettae eval retro --reports C:/jettae/data/real/incident-reports/reports-b --out C:/jettae/data/real/results/retro @m --dry-run
jettae eval retro --reports C:/jettae/data/real/incident-reports/reports-b --out C:/jettae/data/real/results/retro @m --max-usd 30 --yes
# 2) 나중에 전체 폴더(27건)로: 새 18건만 추정·처리·과금
jettae eval retro --reports C:/jettae/data/real/incident-reports/reports --out C:/jettae/data/real/results/retro @m --dry-run
jettae eval retro --reports C:/jettae/data/real/incident-reports/reports --out C:/jettae/data/real/results/retro @m --max-usd 60 --yes
```

- **`--max-usd`는 이 `--out` 폴더의 누적 상한**입니다(이전 실행의 지출 포함). 위 2)의 `--max-usd 60`은
  "이 폴더에서 모두 합쳐 $60까지"라는 뜻이고, 시작 전에 *남은 사고의 추정 비용*을
  *남은 예산 = --max-usd − 누적 지출*과 비교합니다(지출·추정·남은 예산을 먼저 출력).
- `--max-usd-run`(선택)은 이번 실행만의 상한입니다. 닿으면 멈추고, 같은 명령으로 다시 실행하면 이어서 합니다.
- 실제 지출은 결과 폴더의 `spend.json`에 모델별(입력·출력·캐시 토큰, USD)·사고별·실행별로 호출마다 기록됩니다.
  예산에 닿아 멈춘 사고는 다음 실행에서 처음부터 다시 처리하며, 그때 이미 쓴 비용도 누적 지출에 들어갑니다.
- 사용한 입력 폴더와 사고별 내용 해시는 `manifest.json`에 남습니다. 이미 끝난 사고가 새 입력에서 빠졌거나
  내용이 바뀌었으면 시작하지 않습니다(결과를 조용히 잃지 않게). 의도한 변경이면 `--allow-input-change`:
  바뀐 사고는 다시 처리(과금)하고, 이전 결과는 `results.superseded.jsonl`로 옮겨 지표에서 뺍니다.
- 모델·effort·코더·묶음 언어가 다르거나, 제품 동작(`pipeline`)·관할 규칙(`jurisdiction`)이 바뀐 뒤에는 같은 `--out`으로 이어서 실행할 수 없습니다(`run.json`). 새 `--out` 폴더로 실행하세요.
