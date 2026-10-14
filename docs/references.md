# 참고 문헌과 설계 근거

제때옵스의 각 구성요소가 어떤 연구·오픈소스에서 출발했고 어디서 달라지는지 정리했다.
모든 항목은 2026-10-13에 1차 출처(arXiv·학회 프로시딩·DOI·공식 저장소)로 다시 확인했다.
원자료는 `docs/research/`에 있다.

## 1. 문제 정의 — 런북(TSG)은 체계적으로 결함이 있다

| 문헌 | 핵심 | 제때옵스에서 |
|---|---|---|
| Shetty et al., **AutoTSG**, ESEC/FSE 2022 (arXiv:2205.13457) | Microsoft TSG 4,000여 건·사고 17,000건 분석. 흩어짐·불완전·노후화 등 TSG 결함 분류 | 고정 규칙 체계가 이런 결함 분류를 실행 가능한 형태로 옮긴 것이다. 단, 우리는 기술 명령이 아니라 **사람·승인·채널·신고 경로**를 검사한다 |
| Las-Casas et al., **LLexus**, ACM SIGOPS OSR 2024 | LLM은 오프라인 계획 단계(사람과 함께 TSG→플로차트)에만 쓰고, 실행은 LLM 없이 | "AI는 장애 전에 사람과 함께, 판정·재생은 결정적으로"라는 원칙의 가장 가까운 선행 연구다 |
| An et al., **Nissist**, ECAI 2024 (arXiv:2402.17531) | 비정형 TSG가 당직 피로와 신입 어려움의 원인. 장애 중 TSG 검색 코파일럿 | 문제의식은 같고 시점이 다르다. 제때옵스는 장애 **전** 감사 도구다 |
| Mao et al., **StepFly**, FSE 2026 (arXiv:2510.10074) / Xiao et al., **TSGen**, FSE 2026 Industry | TSG 자동 실행과 TSG 자동 생성 | 연동·비교 대상. 우리는 생성된 런북이 *사람 기준으로 작동하는지*를 점수화한다 |

## 2. "AI 진단은 아직 불안정" — 판정을 규칙에 두는 이유

| 문헌 | 핵심 |
|---|---|
| Jha et al., **ITBench**, ICML 2025 (arXiv:2502.05352) | 실제 IT 자동화 과제에서 SRE 에이전트 성공률이 낮다 |
| Chen et al., **AIOpsLab**, MLSys 2025 (arXiv:2501.06706) · Xu et al., **OpenRCA**, ICLR 2025 | 장애 진단·근본원인 분석 벤치마크. 최고 수준 LLM도 큰 폭으로 실패한다 |
| Chen et al., **RCACopilot**, EuroSys 2024 (arXiv:2305.15778) | 결정적 핸들러가 먼저, LLM은 마지막이라는 신뢰성 패턴 |
| Ahmed et al., ICSE 2023 (arXiv:2301.03797) | LLM 기반 장애 대응 연구의 출발점 |
| Chen et al., **STRATUS**, NeurIPS 2025 (arXiv:2506.02009) | 에이전트 조치에 대한 무회귀(Transactional No-Regression)·되돌리기 |

→ 제때옵스: 끊김 판정·점수·리허설 판정·그래프 변경에 **LLM 호출 0회**(import-linter로 구조적으로 강제).
수정안은 과거 리허설을 모두 다시 돌려 **무회귀**를 확인한 뒤 사람 승인분만 새 버전이 된다(STRATUS의 무회귀 원칙을 사람 승인 흐름으로 옮김).

## 3. 근거 있는 추출 — 지어내기 금지

| 문헌·도구 | 핵심 | 제때옵스 구현 |
|---|---|---|
| Google **LangExtract** (github.com/google/langextract, 2025) | 추출마다 원문 문자 위치를 붙이는 source grounding, 정확·퍼지 정렬 | `extract/align.py`: 정확 → 공백 정규화 → 퍼지(≥90) 정렬, 실패 시 폐기 |
| Zhang & Soh, **EDC: Extract-Define-Canonicalize**, EMNLP 2024 | LLM 지식그래프 구축 후 스키마 정규화 | 고정 스키마(`EDGE_SIGNATURES`)에 맞지 않는 관계는 폐기, 명부·카탈로그로 개체 정규화 |
| Edge et al., **GraphRAG**, arXiv:2404.16130 | 텍스트 → 그래프 → 질의 | 그래프를 검색이 아니라 **검사·리플레이 대상**으로 쓴다 |

## 4. 역번역 대조 — 그래프가 원문과 같은 말을 하는가

| 문헌 | 핵심 | 제때옵스 구현 |
|---|---|---|
| Cosler et al., **nl2spec**, CAV 2023 (arXiv:2303.04864) | 자연어 → 형식 명세, 사람과 상호 수정 | 상향·승인·시한을 형식 요소로 변환 |
| Amrollahi, Lopez, Barrett, **Faithful Autoformalization via Roundtrip Verification and Repair**, arXiv:2604.25031 (2026) | 형식화 결과를 다시 자연어로 되돌려 대조 | `extract/backtranslate.py`: 그래프 요소 → 한국어 문장 → 원문과 슬롯(숫자·인물·채널·부정어) 대조, 불일치 시 **판단 보류** |
| Dhuliawala et al., **Chain-of-Verification**, Findings of ACL 2024 · Manakul et al., **SelfCheckGPT**, EMNLP 2023 | 생성 결과의 자기 검증 | LLM 등가성 판정은 보조 신호로만 쓰고, LLM OFF에서도 슬롯 대조는 결정적으로 동작 |

## 5. 프롬프트 인젝션 — 문서는 데이터, 지시가 아니다

| 문헌 | 핵심 | 제때옵스 구현 |
|---|---|---|
| Debenedetti et al., **CaMeL: Defeating Prompt Injections by Design**, arXiv:2503.18813 (2025) | 제어 흐름과 데이터 흐름을 분리하면 데이터 속 지시가 행동을 바꿀 수 없다 | 판정·점수·그래프 변경은 코드가 결정, LLM은 열거된 후보 중 선택과 문장 작성만 |
| Beurer-Kellner et al., **Design Patterns for Securing LLM Agents against Prompt Injections**, arXiv:2506.08837 (2025) | Plan-then-execute, 이중 LLM 등 설계 패턴 | 수정안 op는 `FixHint`+선택값으로 코드가 컴파일 |
| Hines et al., **Spotlighting**, arXiv:2403.14720 (2024) | 비신뢰 입력을 경계·표시로 구분 | `<untrusted_document nonce=…>` 경계 + 문서별 nonce |
| Chen et al., **StruQ**, USENIX Security 2025 | 구조화 질의로 지시/데이터 분리 | 스키마 강제 구조화 출력 |
| Debenedetti et al., **AgentDojo**, NeurIPS 2024 D&B | 인젝션 공격·방어 평가 환경 | `evalkit.injection`: 자기 안전 시험 30건을 배포마다 실행 |

## 6. 리허설·단일 장애점 — 어디가 끊기면 대응이 멈추는가

| 문헌 | 핵심 | 제때옵스 구현 |
|---|---|---|
| Alvaro, Rosen, Hellerstein, **Lineage-driven Fault Injection**, ACM SIGMOD 2015 (doi:10.1145/2723372.2723711) | 성공 경로의 계보에서 거꾸로 추론해 결과를 깨는 장애 조합을 찾는다 | `check/spof.py`: 대응 경로의 **최소 절단 집합**(사람·채널) 계산 → 버스 팩터 1 탐지 |
| Vesely et al., **Fault Tree Handbook** (NUREG-0492), U.S. NRC 1981 | 최소 절단 집합 개념 | 같은 개념을 조직 대응 경로에 적용 |
| Basiri et al., **Chaos Engineering**, IEEE Software 2016 | 실제 시스템에 장애 주입 | 제때옵스는 **실제 시스템을 건드리지 않고** 그래프 위에서 사람·채널 장애를 재생한다(카오스 엔지니어링과 보완 관계) |

## 7. 평가 방법론 — 사람 평가자 없이 정답이 정해지는 방식

| 문헌 | 핵심 | 제때옵스 구현 |
|---|---|---|
| Jia & Harman, **An Analysis and Survey of the Development of Mutation Testing**, IEEE TSE 2011 | 의도적 결함(돌연변이) 주입으로 검사기 성능 측정 | 금본(gold) 조직 명세 → 런북 3문체 렌더링 → 결함 연산자 10종(활성 8 + 보류 2) 주입, 정답은 구성상 확정 |
| Hayes & Krippendorff, **Answering the Call for a Standard Reliability Measure for Coding Data**, CMM 2007 | Krippendorff's α | `evalkit/kappa.py`: 공개 사후보고서 회고 코딩의 LLM 코더 간 일치도(κ, α) |

## 8. 오픈소스·상용 선행 사례 (2026-10 기준)

- **lore** (github.com/get-h3/lore, 2026-09) · **OpsWeave** (github.com/SauravMukherjee44/OpsWeave, 2026-08) — 오픈소스 런북 컴파일러. 부분 선행 사례로 인정한다.
- **FireHydrant** Readiness Checklists · **Cortex** Scorecards · **Backstage Tech Insights** — 서비스·소유자 *존재* 점검(사람이 직접 등록). 제때옵스는 자유 서술 런북에서 경로를 추출하고 *작동 여부*를 리플레이한다.
- **PagerDuty SRE Agent / Service Standards**, **incident.io**, **Rootly**, **ServiceNow Knowledge Graph** — 장애 **중** 도구. 연동 대상이다.
- **Atlassian Opsgenie** 판매 종료 2025-06-04, 지원 종료 **2027-04-05** — 당직 도구 이전 시점이 런북 재점검 시점이다.
- **비브라늄랩스 Vibe AI** — 국내 AI SRE(장애 중 원인 분석). 연동 대상이다.

## 9. 국내 법·제도 (조문 확인: 2026-10-13, `docs/research/kr_regulation.md`)

- **정보통신망법 제48조의3 제1항** (법률 제21500호, 2026.10.1 시행): 침해사고를 알게 된 때부터 **24시간 이내** 과기정통부장관 또는 KISA 신고. 보완 신고는 시행령 제58조의8.
- **정보통신망법 제48조의9** (2026.3.31 신설, 2026.10.1 시행): 침해사고 관리·대응 **매뉴얼** 표준안 보급, 대상 사업자의 작성·제출, 정부의 실태 점검과 시정 명령. 시행령 제60조의15는 필수 항목으로 정기 점검·현행화를 정한다.
- **개인정보 보호법 제34조**, 시행령 제39조(통지)·제40조(신고): 유출등을 알게 된 때부터 **72시간 이내**. 신고는 1천명 이상·민감/고유식별정보·외부 불법접근일 때.
- **ISMS/ISMS-P 인증기준 2.11.4** 사고 대응 훈련(연 1회 이상), **2.12.2** 재해 복구 시험 및 개선 — 리허설 결과가 그대로 심사 증적 초안이 된다.
- **대법원 2024.11.14. 선고 2021다220062 등** — 당직·대기시간의 근로시간 여부는 사안별 실질 판단. 제때옵스는 근로시간 여부를 **판단하지 않고**, 팀이 합의한 피로 기준 위반만 표시한다.
- **인공지능 기본법** (2026.1.22 시행) 제34조 — 고영향 AI 사업자의 사람 관리·감독 의무. 제때옵스는 고영향 AI에 해당하지 않을 가능성이 높으며, '사람 승인 후에만 반영'은 이 원칙에 *부합하는 설계*로 설명한다.
