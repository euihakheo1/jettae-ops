# ADR-0001: AI는 제안하고, 판정은 고정 규칙이 한다

- 상태: 채택 (2026-10-13)
- 관련: ADR-0004, `pyproject.toml [tool.importlinter]`

## 맥락
장애 진단·근본원인 분석 벤치마크(ITBench ICML 2025, AIOpsLab MLSys 2025, OpenRCA ICLR 2025)에서 최상위 LLM 에이전트도
성공률이 높지 않다. 런북 준비도 진단의 산출물은 ISMS 심사 증적, 고객사 SLA 심사 자료로 쓰인다. 이런 용도에는 **재현 가능하고
설명 가능한 판정**이 필요하다. 같은 런북에 실행할 때마다 다른 점수가 나오면 증적으로 쓸 수 없다.

## 결정
- 끊김 판정(`jettae.check`), 준비도 점수(`check.scoring`), 리허설 판정(`rehearse.simulator`),
  수정안 검증·적용(`improve.validator`, `improve.patching`)은 **LLM을 import할 수 없다.** import-linter 계약
  "Judgement path never calls an LLM"으로 CI에서 강제한다.
- LLM은 (1) 자유 서술 런북 → 그래프 초안(원문 위치 필수), (2) 모의 장애 시나리오 아이디어, (3) 수정 문장 초안,
  (4) 역번역 등가성의 보조 신호에만 쓴다.
- 모든 LLM 단계는 `LLMDisabledError` 발생 시 결정적 폴백이 있다(규칙 파서, 템플릿 시나리오, 템플릿 수정안).

## 결과
- (+) `LlmMode.OFF`에서도 전체 파이프라인이 돈다. 데모장 네트워크가 끊겨도 시연이 가능하다.
- (+) 같은 그래프 + 같은 규칙 버전 → 같은 점수. 리허설 결과 해시로 재현성을 증명한다.
- (−) 규칙이 다루지 않는 결함 유형은 탐지하지 못한다. 이 범위는 평가셋의 '보류 연산자'와 회고 평가로 공개한다.
