# ADR-0005: 육각형 아키텍처와 import 계약

- 상태: 채택 (2026-10-13)

## 결정
- `jettae.domain`은 순수 모델이다. I/O, LLM, 프레임워크 import를 금지한다.
- 포트(`domain.ports`, `llm.base`)와 어댑터(`store.memory`/`store.sql`, `llm.anthropic_client`/`openai_compat`/`replay`)를
  분리한다. 기본은 SQLite이고 Postgres로 바꿀 수 있다. LLM은 Claude가 기본이며, 설치형 고객에게는 vLLM(OpenAI 호환)을 쓴다.
- 레이어는 `api|cli|evalkit → orchestrator → extract|check|rehearse|improve|store|reporting → ingest|llm → domain` 순이다.
  같은 레이어 형제끼리는 import하지 않는다. import-linter가 CI에서 강제한다.
- 규칙은 플러그인이다. `check/rules/*.py`에 `@register` 클래스를 추가하면 자동으로 발견되고 `GET /api/v1/rules`로 공개된다.
- 오케스트레이션은 LangGraph `StateGraph`로 한다. 단계마다 노드를 두고, LLM 실패 시 조건부 엣지로 폴백한다.
  애플리케이션 진입점은 `JettaeService` 하나로, API·CLI·GitHub Action이 같은 경로를 쓴다.

## 결과
- 설치형(고객 서버, 원문 외부 전송 없음)은 LLM 어댑터와 DB URL만 바꾸면 된다.
- 새 국내 기준(예: 업종별 신고 의무)은 규칙 모듈 하나를 추가하는 것으로 확장된다.
