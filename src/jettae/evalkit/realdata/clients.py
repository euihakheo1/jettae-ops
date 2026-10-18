"""The three model roles of a study (main pipeline model, coder A, coder B), built for a paid run.

Every client is wrapped in :class:`~jettae.evalkit.realdata.providers.BudgetedLLM` with the same
price table and output assumptions as the dry run, and all of them share one
:class:`~jettae.evalkit.realdata.providers.SpendTracker`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Final

from jettae.evalkit.realdata.coding import CODE_FINDINGS_TASK, FINDINGS_DOC_ID, UNITS_DOC_ID
from jettae.evalkit.realdata.common import is_korean_text
from jettae.evalkit.realdata.keys import ANTHROPIC_KEY, OPENAI_KEY, load_api_key
from jettae.evalkit.realdata.pricing import CallKind, ModelRef
from jettae.evalkit.realdata.providers import (
    BudgetedLLM,
    OpenAIResponsesLLM,
    SpendTracker,
    claude_llm,
    expected_call_usd,
    new_anthropic_sdk,
    new_openai_sdk,
)
from jettae.llm.anthropic_client import Effort
from jettae.llm.base import LLMClient, UntrustedDocument

__all__ = ["StudyClients", "budgeted", "build_study_clients", "call_kind"]

_TASK_KINDS: Final[dict[str, CallKind]] = {
    "extract": "extract",
    "backtranslate": "backtranslate",
    "code_findings": "code_findings",
    "reconstruct": "reconstruct",
    "leakage": "leakage",
    "code_gaps": "code_gaps",
    "cross_code": "cross_code",
}
_LISTS: Final = frozenset({FINDINGS_DOC_ID, UNITS_DOC_ID})


def call_kind(task: str) -> CallKind:
    """The cost model of a task; an unknown task is priced like an extraction (the largest)."""
    return _TASK_KINDS.get(task, "extract")


@dataclass(frozen=True)
class StudyClients:
    main: LLMClient
    coder_a: LLMClient
    coder_b: LLMClient
    tracker: SpendTracker
    main_ref: ModelRef
    coder_a_ref: ModelRef
    coder_b_ref: ModelRef
    #: SDK clients built here (closed by :meth:`aclose`); injected ones belong to the caller
    owned: tuple[Any, ...] = field(default=())

    async def aclose(self) -> None:
        for sdk in self.owned:
            await sdk.close()


def _estimator(
    ref: ModelRef, effort: Effort
) -> Callable[[str, tuple[UntrustedDocument, ...], int, int], float]:
    def estimate(
        task: str, documents: tuple[UntrustedDocument, ...], extra_chars: int, max_tokens: int
    ) -> float:
        sources = [d for d in documents if d.doc_id not in _LISTS]
        lists = [d for d in documents if d.doc_id in _LISTS]
        doc_chars = len(sources[0].text) if sources else 0
        items = sum(d.text.count("\n[") + 1 for d in lists) if task == CODE_FINDINGS_TASK else 0
        return expected_call_usd(
            ref,
            effort,
            call_kind(task),
            input_chars=extra_chars + sum(len(d.text) for d in documents),
            doc_chars=doc_chars,
            items=items,
            korean=any(is_korean_text(d.text) for d in sources),
            max_tokens=max_tokens,
        )

    return estimate


def budgeted(inner: LLMClient, ref: ModelRef, effort: Effort, tracker: SpendTracker) -> BudgetedLLM:
    return BudgetedLLM(inner, tracker, _estimator(ref, effort))


def build_study_clients(
    main: ModelRef,
    coder_a: ModelRef,
    coder_b: ModelRef,
    *,
    effort: Effort,
    tracker: SpendTracker,
    anthropic_sdk: Any | None = None,
    openai_sdk: Any | None = None,
) -> StudyClients:
    """Live clients for a paid run. Keys come from the environment or ``.env`` (never printed).

    Raises ``ValueError`` (Korean) when a role names an unsupported provider or a key is missing.
    """
    if main.provider != "anthropic":
        raise ValueError(f"주 모델은 Claude(anthropic)여야 합니다: {main}")
    if coder_a.provider != "anthropic" or coder_b.provider != "openai":
        raise ValueError(
            "코더는 서로 다른 계열이어야 합니다: 코더 A는 anthropic, 코더 B는 openai "
            f"(지정값: {coder_a}, {coder_b})"
        )
    owned: list[Any] = []
    if anthropic_sdk is None:
        key = load_api_key(ANTHROPIC_KEY)
        if key is None:
            raise ValueError(
                f"{ANTHROPIC_KEY}가 없습니다(환경 변수 또는 .env). 유료 실행을 시작할 수 없습니다."
            )
        anthropic_sdk = new_anthropic_sdk(key)
        owned.append(anthropic_sdk)
    if openai_sdk is None:
        key = load_api_key(OPENAI_KEY)
        if key is None:
            raise ValueError(f"{OPENAI_KEY}가 없습니다(환경 변수 또는 .env). 유료 실행을 시작할 수 없습니다.")
        openai_sdk = new_openai_sdk(key)
        owned.append(openai_sdk)
    main_llm = claude_llm(main.model, effort, anthropic_sdk, tracker)
    coder_a_llm = claude_llm(coder_a.model, effort, anthropic_sdk, tracker)
    coder_b_llm = OpenAIResponsesLLM(coder_b.model, client=openai_sdk, tracker=tracker)
    return StudyClients(
        main=budgeted(main_llm, main, effort, tracker),
        coder_a=budgeted(coder_a_llm, coder_a, effort, tracker),
        coder_b=budgeted(coder_b_llm, coder_b, effort, tracker),
        tracker=tracker,
        main_ref=main,
        coder_a_ref=coder_a,
        coder_b_ref=coder_b,
        owned=tuple(owned),
    )
