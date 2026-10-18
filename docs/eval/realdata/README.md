# 실데이터 실증 공개 파일

해석과 한계는 [../realdata-results.md](../realdata-results.md)에 있다. 이 폴더의 `*-metrics.json`·`*-report.md`는 평가 도구가
쓴 원본을 그대로 옮긴 것이다. `retro-report.md`에만 맨 위에 주의 문구 한 줄을 덧붙였다.

- `gitlab-test-*` — 실증 ① GitLab 공개 런북 test 20건. 보고서가 가리키는 `results.jsonl`·`review.csv`는 이 폴더에 없다
  (사람 검토 표본은 2026-10-22 검토 전까지 블라인드로 둔다).
- `retro-*` — 실증 ② 공개 사후 보고서 27건. `retro-report.md`의 m/k 80.5%는 느슨한 계열·요소 일치다. 제품이 진단한 24건만
  보면 82.5%(33/40)이고, 다른 사고의 지적과 대조한 우연 기준선 82.2%(`retro-baseline.txt` 2절)와 같아 **사고별 사전 탐지의
  근거가 아니다.** 진단하지 않은 3건(복원 묶음에 런북 없음)까지 넣은 기준선 73.1%(2-1절)는 아래로 치우친 값이라 80.5%와
  견주지 않는다.
- `retro-incidents.csv` — 사고별 수치·원 제목·출처 주소만 담았다. 보고서 본문, 인용, 재구성 묶음 글은 넣지 않았다.
- `retro-baseline.txt` — `python -I scripts/realdata_retro_baseline.py <retro results.jsonl>`의 출력.
