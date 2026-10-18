"""Model prices and the documented per-call output assumptions behind every cost estimate.

Prices are the official per-1M-token list prices as of :data:`PRICES_AS_OF`. Output tokens cannot
be counted before a call, so each call kind has a documented *visible* output estimate plus a
*thinking* allowance: thinking (Claude) and reasoning (OpenAI) tokens bill as output, and Claude
Opus 5.5 cannot switch thinking off (its default effort is medium). The same functions price the
dry run, the refuse-to-start guard and the per-call budget reservation, so the three never
disagree about what a call is expected to cost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

from jettae.llm.anthropic_client import Effort

__all__ = [
    "ASSUMED_BACKTRANSLATIONS_PER_DOC",
    "ASSUMED_BUNDLE_TO_REPORT_RATIO",
    "ASSUMED_CROSS_UNITS",
    "ASSUMED_FINDINGS_PER_RUNBOOK",
    "ASSUMED_GAPS_PER_REPORT",
    "CLAUDE_CACHE_READ_MULTIPLIER",
    "CLAUDE_CACHE_WRITE_MULTIPLIER",
    "CLAUDE_THINKING_TOKENS",
    "CODER_B_CHOICES",
    "MAIN_MODEL_CHOICES",
    "OPENAI_CHARS_PER_TOKEN",
    "OPENAI_REASONING_TOKENS",
    "PRICES",
    "PRICES_AS_OF",
    "ModelPrice",
    "ModelRef",
    "Provider",
    "assumptions_text",
    "call_usd",
    "price_of",
    "supports_server_fallbacks",
    "thinking_allowance",
    "visible_output_tokens",
]

Provider = Literal["anthropic", "openai"]

PRICES_AS_OF: Final = "2026-10-14"


@dataclass(frozen=True)
class ModelPrice:
    """USD per 1M tokens. With ``long_prompt_above``, a request whose prompt (input + cache read +
    cache write tokens) is larger is billed entirely at the ``long_*`` rates."""

    input_per_mtok: float
    output_per_mtok: float
    long_prompt_above: int | None = None
    long_input_per_mtok: float | None = None
    long_output_per_mtok: float | None = None

    def rates(self, prompt_tokens: int) -> tuple[float, float]:
        """(input, output) USD per 1M tokens for one request with ``prompt_tokens``."""
        if self.long_prompt_above is not None and prompt_tokens > self.long_prompt_above:
            return (
                self.long_input_per_mtok if self.long_input_per_mtok is not None else self.input_per_mtok,
                self.long_output_per_mtok if self.long_output_per_mtok is not None else self.output_per_mtok,
            )
        return self.input_per_mtok, self.output_per_mtok


#: Models offered as choices (official pages, :data:`PRICES_AS_OF`). Haiku 5.5 costs 0.10/0.50
#: for prompts up to 100K tokens and 0.50/2.50 above that — study 2 sends whole reports, and
#: the long audit reports (≈ 160K–420K tokens per request) are above it.
PRICES: Final[dict[str, ModelPrice]] = {
    "claude-opus-5-5": ModelPrice(4.00, 20.00),
    "claude-sonnet-5-5": ModelPrice(2.00, 10.00),
    "claude-haiku-5-5": ModelPrice(
        0.10, 0.50, long_prompt_above=100_000, long_input_per_mtok=0.50, long_output_per_mtok=2.50
    ),
    "gpt-6-luna": ModelPrice(0.10, 0.50),
    "gpt-6.1-sol": ModelPrice(2.00, 10.00),
}
#: Only for pricing a response that a server-side refusal fallback served with another model.
_FALLBACK_SERVED_PRICES: Final[dict[str, ModelPrice]] = {
    "claude-opus-5": ModelPrice(5.00, 25.00),
    "claude-opus-4-8": ModelPrice(5.00, 25.00),
    "claude-opus-4-7": ModelPrice(5.00, 25.00),
    "claude-sonnet-5": ModelPrice(2.00, 10.00),
    "claude-sonnet-4-6": ModelPrice(3.00, 15.00),
    "claude-haiku-4-5": ModelPrice(1.00, 5.00),
}
#: An unrecognised served model is priced like the most expensive tier (never under-count spend).
_UNKNOWN_PRICE: Final = ModelPrice(10.00, 50.00)

MAIN_MODEL_CHOICES: Final = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5")
CODER_B_CHOICES: Final = ("gpt-6-luna", "gpt-6.1-sol")

#: Prompt-cache accounting for Claude responses: reads at ≤ 0.1× (Opus 5.5 lists 0.05×; 0.1× is
#: the conservative bound), 5-minute cache writes at 1.25×. OpenAI cached input is billed at the
#: full input price here (conservative).
CLAUDE_CACHE_READ_MULTIPLIER: Final = 0.1
CLAUDE_CACHE_WRITE_MULTIPLIER: Final = 1.25

#: Thinking tokens assumed per Claude call at each effort (adaptive thinking; billed as output).
CLAUDE_THINKING_TOKENS: Final[dict[Effort, int]] = {
    "low": 1_000,
    "medium": 3_000,
    "high": 6_000,
    "xhigh": 10_000,
    "max": 14_000,
}
#: Reasoning tokens assumed per OpenAI call (billed as output). Assumption, not measured.
OPENAI_REASONING_TOKENS: Final[dict[str, int]] = {"gpt-6-luna": 1_000, "gpt-6.1-sol": 3_000}
_DEFAULT_OPENAI_REASONING: Final = 3_000
#: Plain characters-per-token heuristic (only when a caller asks for it explicitly; the dry run
#: prices OpenAI input with the Claude count of the same request, see ``ClaudeProxyCounter``).
OPENAI_CHARS_PER_TOKEN: Final = 4.0

# Visible (non-thinking) output per call kind. ------------------------------------------------
#: Extraction JSON relative to the chunk's tokens (the evalkit harness's documented ratio: every
#: entity and relation repeats a verbatim quote).
EXTRACT_OUTPUT_RATIO: Final = 2.0
#: Characters per token used for that chunk-size estimate. Measured with ``count_tokens``
#: (claude-opus-5-5, 2026-10-17): the 39 study-1 chunks of English GitLab markdown 2.60, the
#: Korean demo runbooks 1.07 — rounded down so the estimate errs high.
_CHARS_PER_TOKEN_EN: Final = 2.5
_CHARS_PER_TOKEN_KO: Final = 1.0
BACKTRANSLATE_VISIBLE: Final = 150
CODER_VISIBLE_PER_FINDING: Final = 80
CODER_VISIBLE_BASE: Final = 40
RECONSTRUCT_VISIBLE_BASE: Final = 1_500
RECONSTRUCT_VISIBLE_RATIO: Final = 0.8
LEAKAGE_VISIBLE: Final = 400
GAP_VISIBLE_PER_GAP: Final = 120
CROSS_VISIBLE_PER_UNIT: Final = 60

# Quantities a dry run cannot know before the paid steps produce them. ----------------------
ASSUMED_FINDINGS_PER_RUNBOOK: Final = 12
ASSUMED_GAPS_PER_REPORT: Final = 6
ASSUMED_CROSS_UNITS: Final = 4
#: Back-translation judge calls per Korean document (one per runbook-derived response edge).
ASSUMED_BACKTRANSLATIONS_PER_DOC: Final = 10
#: Reconstructed runbook text relative to the report's length.
ASSUMED_BUNDLE_TO_REPORT_RATIO: Final = 0.6

CallKind = Literal[
    "extract", "backtranslate", "code_findings", "reconstruct", "leakage", "code_gaps", "cross_code"
]


@dataclass(frozen=True)
class ModelRef:
    """``anthropic:claude-opus-5-5`` / ``openai:gpt-6-luna`` (a bare id implies the provider)."""

    provider: Provider
    model: str

    @classmethod
    def parse(cls, text: str) -> ModelRef:
        raw = text.strip()
        provider, sep, model = raw.partition(":")
        if not sep:
            provider, model = ("anthropic" if raw.startswith("claude-") else "openai"), raw
        provider = provider.strip().lower()
        model = model.strip()
        if provider not in ("anthropic", "openai") or not model:
            raise ValueError(f"모델 지정은 'anthropic:<모델>' 또는 'openai:<모델>' 형식이어야 합니다: {text}")
        return cls("anthropic" if provider == "anthropic" else "openai", model)

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"


def price_of(model: str) -> ModelPrice:
    """List price of ``model`` (a fallback-served or unknown model is priced conservatively)."""
    return PRICES.get(model) or _FALLBACK_SERVED_PRICES.get(model) or _UNKNOWN_PRICE


def supports_server_fallbacks(model: str) -> bool:
    """Server-side refusal fallbacks exist for Opus/Sonnet 5.x and Fable; Haiku 5.5 has none."""
    return model.startswith(("claude-opus-5", "claude-sonnet-5-5", "claude-fable-5"))


def call_usd(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """USD of **one** request (a long-prompt tier, like Haiku 5.5's above 100K, applies per request
    — sum per-call results; never price a sum of several calls' tokens in one go)."""
    input_rate, output_rate = price_of(model).rates(input_tokens + cache_read_tokens + cache_write_tokens)
    weighted_input = (
        input_tokens
        + cache_read_tokens * CLAUDE_CACHE_READ_MULTIPLIER
        + cache_write_tokens * CLAUDE_CACHE_WRITE_MULTIPLIER
    )
    return (weighted_input * input_rate + output_tokens * output_rate) / 1_000_000


def thinking_allowance(provider: Provider, model: str, effort: Effort) -> int:
    if provider == "anthropic":
        return CLAUDE_THINKING_TOKENS[effort]
    return OPENAI_REASONING_TOKENS.get(model, _DEFAULT_OPENAI_REASONING)


def _text_tokens(chars: int, *, korean: bool) -> int:
    return math.ceil(chars / (_CHARS_PER_TOKEN_KO if korean else _CHARS_PER_TOKEN_EN))


def visible_output_tokens(kind: CallKind, *, chars: int = 0, items: int = 0, korean: bool = False) -> int:
    """Expected visible output of one call. ``chars`` = the document part it reads, ``items`` =
    findings / gaps / units it must label."""
    text_tokens = _text_tokens(chars, korean=korean)
    estimates: dict[CallKind, int] = {
        "extract": math.ceil(text_tokens * EXTRACT_OUTPUT_RATIO),
        "backtranslate": BACKTRANSLATE_VISIBLE,
        "code_findings": CODER_VISIBLE_BASE + CODER_VISIBLE_PER_FINDING * max(1, items),
        "reconstruct": RECONSTRUCT_VISIBLE_BASE + math.ceil(text_tokens * RECONSTRUCT_VISIBLE_RATIO),
        "leakage": LEAKAGE_VISIBLE,
        "code_gaps": CODER_VISIBLE_BASE + GAP_VISIBLE_PER_GAP * max(1, items or ASSUMED_GAPS_PER_REPORT),
        "cross_code": CODER_VISIBLE_BASE + CROSS_VISIBLE_PER_UNIT * max(1, items or ASSUMED_CROSS_UNITS),
    }
    return estimates[kind]


def assumptions_text(effort: Effort) -> list[str]:
    """The assumptions every estimate rests on, in Korean, for the dry-run table and reports."""
    claude = ", ".join(f"{level} {tokens:,}" for level, tokens in CLAUDE_THINKING_TOKENS.items())
    openai = ", ".join(f"{model} {tokens:,}" for model, tokens in OPENAI_REASONING_TOKENS.items())
    return [
        f"가격: 공식 가격표({PRICES_AS_OF}) 100만 토큰당 입력/출력 USD — "
        + ", ".join(f"{m} {p.input_per_mtok:g}/{p.output_per_mtok:g}" for m, p in PRICES.items())
        + "".join(
            f" ({m}: 프롬프트 {p.long_prompt_above // 1000}K 토큰 초과 호출은 "
            f"{p.long_input_per_mtok:g}/{p.long_output_per_mtok:g}, 호출마다 적용)"
            for m, p in PRICES.items()
            if p.long_prompt_above is not None
        ),
        "입력 토큰: Claude 호출은 messages.count_tokens(무료)로 실제 요청 그대로 셈, "
        "OpenAI 호출은 같은 요청의 Claude 토큰 수로 대신함(Claude 토크나이저가 더 많이 셈 — 보수적; "
        f"글자 수/{OPENAI_CHARS_PER_TOKEN:g}는 한국어 지시문·코드가 섞인 요청을 크게 덜 셈). "
        "캐시 할인은 반영하지 않음(보수적)",
        "출력 토큰 = 보이는 출력 추정 + 사고/추론 토큰(출력으로 과금). "
        f"Claude 사고 토큰(호출당, effort별): {claude} — 이번 effort={effort}. "
        f"OpenAI 추론 토큰(호출당, 가정): {openai}",
        f"보이는 출력 추정: 추출 = 청크 토큰 × {EXTRACT_OUTPUT_RATIO:g}(영어 글자/{_CHARS_PER_TOKEN_EN:g}, "
        f"한국어 글자/{_CHARS_PER_TOKEN_KO:g}), "
        f"지적 판정 = {CODER_VISIBLE_BASE} + 지적당 {CODER_VISIBLE_PER_FINDING}, "
        f"재구성 = {RECONSTRUCT_VISIBLE_BASE:,} + 보고서 토큰 × {RECONSTRUCT_VISIBLE_RATIO:g}, "
        f"누설 검사 = {LEAKAGE_VISIBLE}, 공백 코딩 = {CODER_VISIBLE_BASE} + 공백당 {GAP_VISIBLE_PER_GAP}, "
        f"교차 코딩 = {CODER_VISIBLE_BASE} + 단위당 {CROSS_VISIBLE_PER_UNIT}",
        "출력 추정은 호출의 max_tokens를 넘지 않게 자름",
    ]
