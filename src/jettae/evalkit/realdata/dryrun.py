"""Dry run: build every request a study would send, count its input tokens, price the choices.

* Requests are built by the same functions the real run calls (extraction chunks, trusted
  context, coder prompts …). Where a request depends on the output of an earlier paid step — the
  findings a coder judges, the bundle a reconstruction writes — a documented placeholder of the
  same shape stands in, and the call is marked ``exact=False``.
* Claude input tokens come from Anthropic's free ``messages.count_tokens`` on the exact request
  (system prompt with cache marker, user turn, ``output_config`` with the strict schema), counted
  per model because tokenizers differ. OpenAI input tokens are the Claude count of the same
  request (:class:`ClaudeProxyCounter`, conservative; no OpenAI endpoint is called).
* Output tokens are the documented estimates of :mod:`~jettae.evalkit.realdata.pricing`
  (visible output + thinking/reasoning allowance), capped at each call's ``max_tokens``.

No paid endpoint is reachable from this module: the only network call is ``count_tokens``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from jettae.evalkit.realdata.pricing import (
    CODER_B_CHOICES,
    MAIN_MODEL_CHOICES,
    OPENAI_CHARS_PER_TOKEN,
    CallKind,
    ModelRef,
    assumptions_text,
    call_usd,
    thinking_allowance,
    visible_output_tokens,
)
from jettae.llm.anthropic_client import AnthropicLLM, Effort
from jettae.llm.base import UntrustedDocument
from jettae.llm.prompting import build_system, render_user_content
from jettae.llm.schema import to_strict_json_schema

__all__ = [
    "AnthropicTokenCounter",
    "CharTokenCounter",
    "ClaudeProxyCounter",
    "ComboCost",
    "CostTable",
    "PlannedCall",
    "Role",
    "StageCost",
    "TokenCounter",
    "estimate_costs",
]

Role = Literal["main", "coder_a", "coder_b"]
_ROLE_KO: Final[dict[Role, str]] = {"main": "주 모델", "coder_a": "코더 A", "coder_b": "코더 B"}
_STAGE_KO: Final[dict[str, str]] = {
    "extract": "추출",
    "backtranslate": "역번역 대조",
    "code_findings": "지적 판정",
    "reconstruct": "사고 전 묶음 재구성",
    "leakage": "결과 누설 검사",
    "code_gaps": "공백 코딩",
    "cross_code": "교차 코딩",
}
#: Claude token heuristic when no Anthropic key is available for ``count_tokens`` (measured ≈ 1.7
#: characters per token on the real study-1 requests, ≈ 1.07 on Korean runbook text; 1.0 errs high).
_CLAUDE_FALLBACK_CHARS_PER_TOKEN: Final = 1.0


@dataclass(frozen=True)
class PlannedCall:
    """One request a study would send, built exactly as the real run builds it."""

    item: str
    stage: CallKind
    role: Role
    instructions: str
    schema: type[BaseModel]
    documents: tuple[UntrustedDocument, ...]
    context: str
    max_tokens: int
    #: characters of the part whose size the visible output scales with (chunk, report …)
    doc_chars: int = 0
    #: findings / gaps / units the call labels
    items: int = 0
    korean: bool = False
    #: ``False`` when built around a documented placeholder (an earlier paid step's output)
    exact: bool = True

    def system_text(self) -> str:
        return build_system(self.instructions)

    def user_text(self) -> str:
        return render_user_content(self.documents, self.context)

    def output_tokens(self, ref: ModelRef, effort: Effort) -> int:
        visible = visible_output_tokens(
            self.stage, chars=self.doc_chars, items=self.items, korean=self.korean
        )
        return min(self.max_tokens, visible + thinking_allowance(ref.provider, ref.model, effort))

    def input_chars(self) -> int:
        schema = json.dumps(to_strict_json_schema(self.schema), ensure_ascii=False)
        return len(self.system_text()) + len(self.user_text()) + len(schema)


class TokenCounter(Protocol):
    #: how the counts were obtained, for the table header
    method: str

    async def count(self, ref: ModelRef, effort: Effort, call: PlannedCall) -> int: ...


class CharTokenCounter:
    """characters / ``chars_per_token`` of everything the request carries (system, user, schema)."""

    def __init__(self, chars_per_token: float = OPENAI_CHARS_PER_TOKEN, method: str | None = None) -> None:
        self.chars_per_token = chars_per_token
        self.method = method or f"글자 수/{chars_per_token:g}"

    async def count(self, ref: ModelRef, effort: Effort, call: PlannedCall) -> int:
        return math.ceil(call.input_chars() / self.chars_per_token)


class ClaudeProxyCounter:
    """OpenAI input tokens as Claude counts the identical request (no free OpenAI counter is used).

    Characters/4 badly undercounts these prompts — Korean instructions, numbered runbook lines,
    tables and code: ``count_tokens`` measured ≈ 1.7 characters per token on the study-1 coder
    requests. Claude's tokenizer is the less compact of the two, so its count is a conservative
    stand-in, and as coder A's requests are the same text it costs no extra ``count_tokens``
    request (the counter's cache answers).
    """

    def __init__(self, inner: TokenCounter, proxy: ModelRef) -> None:
        self.inner = inner
        self.proxy = proxy
        self.method = f"같은 요청의 {proxy.model} 토큰 수로 대신함({inner.method}, 보수적)"

    async def count(self, ref: ModelRef, effort: Effort, call: PlannedCall) -> int:
        return await self.inner.count(self.proxy, effort, call)


class _CountTokensAPI(Protocol):
    async def count_tokens(self, **kwargs: Any) -> Any: ...


class _AnthropicCountingClient(Protocol):
    @property
    def messages(self) -> _CountTokensAPI: ...

    @property
    def beta(self) -> Any: ...


class AnthropicTokenCounter:
    """Anthropic ``messages.count_tokens`` (free, rate-limited) on the request the adapter would send."""

    method = "Anthropic messages.count_tokens(무료)"

    def __init__(self, sdk: _AnthropicCountingClient, *, concurrency: int = 4) -> None:
        self._sdk = sdk
        self._gate = asyncio.Semaphore(max(1, concurrency))
        self._cache: dict[str, int] = {}
        self.requests = 0

    def count_request(self, ref: ModelRef, effort: Effort, call: PlannedCall) -> dict[str, Any]:
        """``count_tokens`` keyword arguments: system, messages and output_config of the real request."""
        adapter = AnthropicLLM(ref.model, effort, client=self._sdk)
        kwargs = adapter.request_kwargs(
            call.system_text(),
            [{"role": "user", "content": call.user_text()}],
            to_strict_json_schema(call.schema),
            call.max_tokens,
        )
        output_config = kwargs.get("output_config") or kwargs.get("extra_body", {}).get("output_config")
        request: dict[str, Any] = {
            "model": ref.model,
            "system": kwargs["system"],
            "messages": kwargs["messages"],
        }
        if output_config is not None:
            request["output_config"] = output_config
        return request

    async def count(self, ref: ModelRef, effort: Effort, call: PlannedCall) -> int:
        request = self.count_request(ref, effort, call)
        key = hashlib.sha256(json.dumps(request, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        async with self._gate:
            self.requests += 1
            response = await self._sdk.messages.count_tokens(**request)
        tokens = int(response.input_tokens)
        self._cache[key] = tokens
        return tokens


# --------------------------------------------------------------------------- tables


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True)


class StageCost(_Model):
    stage: str
    role: Role
    model: str
    calls: int
    exact_calls: int
    input_tokens: int
    output_tokens: int
    usd: float


class ComboCost(_Model):
    main: str
    coder_a: str
    coder_b: str
    usd: float
    chosen: bool


class CostTable(_Model):
    study: str
    items: int
    effort: str
    main_counter: str
    coder_a_counter: str
    coder_b_counter: str
    notes: tuple[str, ...]
    rows: tuple[StageCost, ...]
    combos: tuple[ComboCost, ...]

    @property
    def chosen(self) -> ComboCost:
        return next(c for c in self.combos if c.chosen)

    def render(self) -> str:
        """Korean plain-text table (stdout of ``--dry-run`` and ``dry_run.md``)."""
        lines = [
            f"{self.study} 비용 추정 — 드라이런(유료 호출 0회) · 대상 {self.items}건 · effort={self.effort}",
            f"입력 토큰 계산: 주 모델 = {self.main_counter}, 코더 A = {self.coder_a_counter}, "
            f"코더 B = {self.coder_b_counter}",
            *(f"- {note}" for note in self.notes),
            "",
            "| 단계 | 역할 | 모델 | 호출 | (정확 프롬프트) | 입력 토큰 | 출력 토큰(추정) | 비용(USD) |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
        lines += [
            f"| {_STAGE_KO.get(r.stage, r.stage)} | {_ROLE_KO[r.role]} | {r.model} | {r.calls:,} | "
            f"{r.exact_calls:,} | {r.input_tokens:,} | {r.output_tokens:,} | ${r.usd:,.2f} |"
            for r in self.rows
        ]
        coder_a = self.combos[0].coder_a if self.combos else "-"
        mains = list(dict.fromkeys(c.main for c in self.combos))
        coder_bs = list(dict.fromkeys(c.coder_b for c in self.combos))
        lines += [
            "",
            f"조합별 합계 (코더 A = {coder_a}):",
            "| 주 모델 \\ 코더 B | " + " | ".join(coder_bs) + " |",
            "|---|" + "---:|" * len(coder_bs),
        ]
        by_pair = {(c.main, c.coder_b): c for c in self.combos}
        for main in mains:
            cells = []
            for coder_b in coder_bs:
                combo = by_pair[(main, coder_b)]
                cells.append(f"${combo.usd:,.2f}" + (" ← 선택" if combo.chosen else ""))
            lines.append(f"| {main} | " + " | ".join(cells) + " |")
        chosen = self.chosen
        lines += [
            "",
            f"선택한 조합 합계: ${chosen.usd:,.2f} "
            f"(주 모델 {chosen.main}, 코더 A {chosen.coder_a}, 코더 B {chosen.coder_b})",
        ]
        return "\n".join(lines)


def _candidates(chosen: str, choices: Sequence[str]) -> list[str]:
    return list(dict.fromkeys([*choices, chosen]))


async def _stage_rows(
    calls: Sequence[PlannedCall], ref: ModelRef, effort: Effort, role: Role, counter: TokenCounter
) -> list[StageCost]:
    mine = [c for c in calls if c.role == role]
    counts = await asyncio.gather(*(counter.count(ref, effort, c) for c in mine))
    rows: list[StageCost] = []
    for stage in dict.fromkeys(c.stage for c in mine):
        pairs = [(c, n) for c, n in zip(mine, counts, strict=True) if c.stage == stage]
        outputs = [c.output_tokens(ref, effort) for c, _ in pairs]
        # priced call by call: a long-prompt tier (Haiku 5.5 above 100K) applies per request
        usd = sum(
            call_usd(ref.model, input_tokens=n, output_tokens=out)
            for (_, n), out in zip(pairs, outputs, strict=True)
        )
        rows.append(
            StageCost(
                stage=stage,
                role=role,
                model=ref.model,
                calls=len(pairs),
                exact_calls=sum(1 for c, _ in pairs if c.exact),
                input_tokens=sum(n for _, n in pairs),
                output_tokens=sum(outputs),
                usd=round(usd, 4),
            )
        )
    return rows


async def estimate_costs(
    calls: Sequence[PlannedCall],
    *,
    study: str,
    items: int,
    main: ModelRef,
    coder_a: ModelRef,
    coder_b: ModelRef,
    effort: Effort,
    claude_counter: TokenCounter,
    openai_counter: TokenCounter | None = None,
    compare: bool = True,
    notes: Iterable[str] = (),
) -> CostTable:
    """Price ``calls`` for the chosen models and (with ``compare``) the alternative choices.

    Main-model alternatives are the Claude choices (opus/sonnet/haiku), coder-B alternatives the
    OpenAI choices (luna/sol); coder A stays as chosen. ``compare=False`` prices only the chosen
    combination (what the real-run guard needs — fewer ``count_tokens`` requests). OpenAI input
    tokens default to :class:`ClaudeProxyCounter` over coder A's (else the main) Claude model.
    """
    proxy = coder_a if coder_a.provider == "anthropic" else main
    heuristic = openai_counter or ClaudeProxyCounter(claude_counter, proxy)

    def counter_for(ref: ModelRef) -> TokenCounter:
        return claude_counter if ref.provider == "anthropic" else heuristic

    mains = (
        [ModelRef("anthropic", m) for m in _candidates(main.model, MAIN_MODEL_CHOICES)] if compare else [main]
    )
    if main.provider != "anthropic":
        mains = [main]
    coder_bs = (
        [ModelRef(coder_b.provider, m) for m in _candidates(coder_b.model, CODER_B_CHOICES)]
        if compare and coder_b.provider == "openai"
        else [coder_b]
    )
    rows: list[StageCost] = []
    main_usd: dict[str, float] = {}
    for ref in mains:
        found = await _stage_rows(calls, ref, effort, "main", counter_for(ref))
        rows += found
        main_usd[ref.model] = sum(r.usd for r in found)
    coder_a_rows = await _stage_rows(calls, coder_a, effort, "coder_a", counter_for(coder_a))
    rows += coder_a_rows
    coder_b_usd: dict[str, float] = {}
    for ref in coder_bs:
        found = await _stage_rows(calls, ref, effort, "coder_b", counter_for(ref))
        rows += found
        coder_b_usd[ref.model] = sum(r.usd for r in found)
    coder_a_usd = sum(r.usd for r in coder_a_rows)
    combos = tuple(
        ComboCost(
            main=m,
            coder_a=coder_a.model,
            coder_b=b,
            usd=round(main_usd[m] + coder_a_usd + coder_b_usd[b], 4),
            chosen=(m == main.model and b == coder_b.model),
        )
        for m in main_usd
        for b in coder_b_usd
    )
    return CostTable(
        study=study,
        items=items,
        effort=effort,
        main_counter=counter_for(main).method,
        coder_a_counter=counter_for(coder_a).method,
        coder_b_counter=counter_for(coder_b).method,
        notes=(*notes, *assumptions_text(effort)),
        rows=tuple(rows),
        combos=combos,
    )


def claude_fallback_counter() -> CharTokenCounter:
    """Used only when no Anthropic key is available (the table header says so)."""
    return CharTokenCounter(
        _CLAUDE_FALLBACK_CHARS_PER_TOKEN,
        method=f"글자 수/{_CLAUDE_FALLBACK_CHARS_PER_TOKEN:g} (Anthropic 키가 없어 count_tokens 대신 추정)",
    )
