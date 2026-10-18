"""What ``jettae eval real-runbooks`` and ``jettae eval retro`` do, behind the CLI's option parsing.

The paid-run guard lives here so every caller gets it, not only the CLI:

1. Find what is left to do in the output folder (:mod:`~jettae.evalkit.realdata.resume`): items
   already in ``results.jsonl`` with unchanged content are done; a different input folder is
   accepted when it is a superset of them (the settings must match, as before). Build every
   request of the *unfinished* items only and price it (:mod:`~jettae.evalkit.realdata.dryrun`).
   ``--dry-run`` stops here — the only network call made is Anthropic's free ``count_tokens``.
2. A real run needs ``max_usd`` and an explicit confirmation. ``max_usd`` is **cumulative** for
   the output folder: the run refuses to start when the estimate for the unfinished items
   exceeds what is left of it (``max_usd`` − spend already in the folder's ``spend.json``).
   ``max_usd_run`` optionally caps this invocation alone.
3. During the run actual usage is tracked, written to the ledger after every call, and the run
   stops cleanly at either cap; the next run resumes where it stopped.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jettae.evalkit.realdata.clients import build_study_clients
from jettae.evalkit.realdata.common import write_json, write_text
from jettae.evalkit.realdata.dryrun import (
    AnthropicTokenCounter,
    CostTable,
    PlannedCall,
    TokenCounter,
    claude_fallback_counter,
    estimate_costs,
)
from jettae.evalkit.realdata.keys import ANTHROPIC_KEY, load_api_key
from jettae.evalkit.realdata.pricing import ModelRef
from jettae.evalkit.realdata.providers import SpendTracker, new_anthropic_sdk
from jettae.evalkit.realdata.resume import ResumeState, resume_lines
from jettae.evalkit.realdata.retro import (
    RetroConfig,
    RetroOutcome,
    load_reports,
    plan_retro_calls,
    run_retro_study,
)
from jettae.evalkit.realdata.retro import resume_state as retro_resume_state
from jettae.evalkit.realdata.runbooks import (
    RunbookStudyConfig,
    StudyOutcome,
    load_study_sources,
    plan_runbook_calls,
    run_runbook_study,
)
from jettae.evalkit.realdata.runbooks import resume_state as runbook_resume_state
from jettae.llm.anthropic_client import Effort

__all__ = ["CommandResult", "PaidRunRefusedError", "run_real_runbooks", "run_retro"]

Echo = Callable[[str], object]


class PaidRunRefusedError(ValueError):
    """The guard refused to start a paid run (Korean message; the estimate was already shown)."""


@dataclass(frozen=True)
class CommandResult:
    table: CostTable
    dry_run: bool
    outcome: StudyOutcome | RetroOutcome | None = None
    #: spent in the output folder before this command (the spend ledger)
    spent_before_usd: float = 0.0


def _refs(model: str, coder_a: str, coder_b: str) -> tuple[ModelRef, ModelRef, ModelRef]:
    main = ModelRef.parse(model)
    if main.provider != "anthropic":
        raise ValueError(f"--model은 Claude 모델이어야 합니다: {model}")
    a, b = ModelRef.parse(coder_a), ModelRef.parse(coder_b)
    if a.provider != "anthropic" or b.provider != "openai":
        raise ValueError(
            f"--coder-a는 anthropic:<모델>, --coder-b는 openai:<모델>이어야 합니다(서로 다른 계열): {a}, {b}"
        )
    return main, a, b


async def _close(sdk: Any) -> None:
    close = getattr(sdk, "close", None)
    if close is not None:
        await close()


@dataclass(frozen=True)
class _Priced:
    table: CostTable
    #: estimate of the first unfinished item alone (what ``max_usd_run`` must at least allow)
    next_item_usd: float | None


async def _price(
    calls: list[PlannedCall],
    *,
    study: str,
    items: int,
    refs: tuple[ModelRef, ModelRef, ModelRef],
    effort: Effort,
    compare: bool,
    notes: list[str],
    counter: TokenCounter | None,
    echo: Echo,
    next_item: str | None = None,
) -> _Priced:
    main, a, b = refs
    sdk: Any | None = None
    claude_counter = counter
    if claude_counter is None:
        key = load_api_key(ANTHROPIC_KEY)
        if key is None:
            echo(
                f"주의: {ANTHROPIC_KEY}가 없어 Claude 입력 토큰을 글자 수로 추정합니다(count_tokens 미사용)."
            )
            claude_counter = claude_fallback_counter()
        else:
            sdk = new_anthropic_sdk(key)
            claude_counter = AnthropicTokenCounter(sdk)
    try:
        table = await estimate_costs(
            calls,
            study=study,
            items=items,
            main=main,
            coder_a=a,
            coder_b=b,
            effort=effort,
            claude_counter=claude_counter,
            compare=compare,
            notes=notes,
        )
        next_usd: float | None = None
        if next_item is not None:
            # the same requests again: answered from the counter's cache, no new count_tokens call
            first = await estimate_costs(
                [c for c in calls if c.item == next_item],
                study=study,
                items=1,
                main=main,
                coder_a=a,
                coder_b=b,
                effort=effort,
                claude_counter=claude_counter,
                compare=False,
            )
            next_usd = first.chosen.usd
        return _Priced(table=table, next_item_usd=next_usd)
    finally:
        if sdk is not None:
            await _close(sdk)


def _check_caps(max_usd: float | None, max_usd_run: float | None) -> float:
    """The cumulative cap, once both caps are valid."""
    if max_usd is None:
        raise PaidRunRefusedError(
            "유료 실행에는 --max-usd(이 --out 폴더의 누적 달러 상한)가 필요합니다. "
            "먼저 --dry-run으로 비용을 확인하세요."
        )
    if max_usd <= 0:
        raise PaidRunRefusedError(f"--max-usd는 0보다 커야 합니다: {max_usd:g}")
    if max_usd_run is not None and max_usd_run <= 0:
        raise PaidRunRefusedError(f"--max-usd-run은 0보다 커야 합니다: {max_usd_run:g}")
    return max_usd


def _guard(
    priced: _Priced,
    *,
    max_usd: float | None,
    max_usd_run: float | None,
    spent_usd: float,
    confirmed: bool,
) -> None:
    """Refuse a paid start unless the unfinished items' estimate fits the cumulative budget left."""
    cap = _check_caps(max_usd, max_usd_run)
    estimate = priced.table.chosen.usd
    remaining = cap - spent_usd
    if estimate > 0 and remaining <= 0:
        raise PaidRunRefusedError(
            f"이 --out 폴더의 누적 지출 ${spent_usd:,.4f}가 이미 누적 상한 --max-usd ${cap:,.2f}에 닿아 "
            "시작하지 않습니다. 남은 항목을 이어서 하려면 --max-usd(누적 상한)를 누적 지출보다 크게 올리세요."
        )
    if estimate > remaining:
        raise PaidRunRefusedError(
            f"남은 항목의 추정 비용 ${estimate:,.2f}가 남은 예산 ${remaining:,.2f}"
            f"(= 누적 상한 --max-usd ${cap:,.2f} − 이 --out 폴더의 누적 지출 ${spent_usd:,.4f})를 "
            "넘어 시작하지 않습니다. 상한을 올리거나 모델·--limit을 바꾸세요."
        )
    if max_usd_run is not None and priced.next_item_usd is not None and priced.next_item_usd > max_usd_run:
        raise PaidRunRefusedError(
            f"이번 실행 상한 --max-usd-run ${max_usd_run:,.2f}이 다음 항목 하나의 추정 비용 "
            f"${priced.next_item_usd:,.2f}보다 작아, 한 건도 끝내지 못하고 멈출 수 있어 시작하지 않습니다"
            "(멈춘 항목은 다음에 처음부터 다시 처리합니다). --max-usd-run을 올리세요."
        )
    if not confirmed:
        raise PaidRunRefusedError(
            f"유료 호출입니다(남은 항목 추정 ${estimate:,.2f}). "
            "위 추정치를 확인한 뒤 --yes를 붙여 다시 실행하세요."
        )


def _save_table(table: CostTable, out_dir: Path) -> None:
    write_json(out_dir / "dry_run.json", table.model_dump(mode="json"))
    write_text(out_dir / "dry_run.md", table.render() + "\n")


def _resume_note(state: ResumeState, item_word: str) -> list[str]:
    done = len(state.inputs.done)
    if not done:
        return []
    return [f"이 --out 폴더에서 이미 끝난 {item_word} {done}건은 제외(results.jsonl) — 남은 항목만 추정"]


def _announce(
    echo: Echo,
    state: ResumeState,
    priced: _Priced,
    *,
    item_word: str,
    max_usd: float | None,
    max_usd_run: float | None,
) -> None:
    for line in resume_lines(
        state,
        item_word=item_word,
        estimate_usd=priced.table.chosen.usd,
        max_usd=max_usd,
        max_usd_run=max_usd_run,
    ):
        echo(line)


def _todo(items: Sequence[Any], key: Callable[[Any], str], state: ResumeState) -> list[Any]:
    todo = set(state.inputs.todo)
    return [item for item in items if key(item) in todo]


async def run_real_runbooks(
    config: RunbookStudyConfig,
    *,
    dry_run: bool,
    confirmed: bool = False,
    echo: Echo = print,
    counter: TokenCounter | None = None,
    anthropic_sdk: Any | None = None,
    openai_sdk: Any | None = None,
) -> CommandResult:
    """Study 1 command. ``counter``/``*_sdk`` are injection points for tests (fakes, no network)."""
    refs = _refs(config.model, config.coder_a, config.coder_b)
    sources = load_study_sources(config)
    state = runbook_resume_state(config, sources)
    todo = _todo(sources, lambda s: s.path, state)
    calls = plan_runbook_calls(todo, coder_batch=config.coder_batch, assumed_findings=config.assumed_findings)
    chunks = sum(1 for c in calls if c.stage == "extract")
    notes = [
        *_resume_note(state, "런북"),
        f"런북 {len(todo)}건, 추출 청크 {chunks}개(정확한 요청). 역번역 LLM 대조: 한국어가 아닌 원문은 생략",
        f"코더 판정 요청은 실제 런북 + 가정한 지적 {config.assumed_findings}건/런북(자리표시)으로 만듦, "
        f"요청당 지적 {config.coder_batch}건 — 실제 지적 수에 비례해 달라짐",
        f"평가 분할: {config.split} · 관할(정책): {', '.join(config.jurisdictions)}",
    ]
    priced = await _price(
        calls,
        study="실증 ① 공개 런북",
        items=len(todo),
        refs=refs,
        effort=config.effort,
        compare=dry_run,
        notes=notes,
        counter=counter,
        echo=echo,
        next_item=todo[0].path if todo and config.max_usd_run is not None and not dry_run else None,
    )
    config.out_dir.mkdir(parents=True, exist_ok=True)
    spent = state.ledger.prior_usd
    if dry_run:
        _save_table(priced.table, config.out_dir)
        _announce(
            echo, state, priced, item_word="런북", max_usd=config.max_usd, max_usd_run=config.max_usd_run
        )
        return CommandResult(table=priced.table, dry_run=True, spent_before_usd=spent)
    echo(priced.table.render())
    _announce(echo, state, priced, item_word="런북", max_usd=config.max_usd, max_usd_run=config.max_usd_run)
    _guard(
        priced, max_usd=config.max_usd, max_usd_run=config.max_usd_run, spent_usd=spent, confirmed=confirmed
    )
    tracker = SpendTracker(max_usd=config.max_usd, max_usd_run=config.max_usd_run, prior_usd=spent)
    main, a, b = refs
    clients = build_study_clients(
        main, a, b, effort=config.effort, tracker=tracker, anthropic_sdk=anthropic_sdk, openai_sdk=openai_sdk
    )
    try:
        outcome = await run_runbook_study(config, clients, echo=echo, sources=sources)
    finally:
        await clients.aclose()
    return CommandResult(table=priced.table, dry_run=False, outcome=outcome, spent_before_usd=spent)


async def run_retro(
    config: RetroConfig,
    *,
    dry_run: bool,
    confirmed: bool = False,
    echo: Echo = print,
    counter: TokenCounter | None = None,
    anthropic_sdk: Any | None = None,
    openai_sdk: Any | None = None,
) -> CommandResult:
    """Study 2 command (same guard as study 1)."""
    refs = _refs(config.model, config.coder_a, config.coder_b)
    reports = load_reports(config.reports_dir, limit=config.limit)
    state = retro_resume_state(config, reports)
    todo = _todo(reports, lambda r: r.incident_id, state)
    calls = plan_retro_calls(todo, bundle_language=config.bundle_language)
    notes = [
        *_resume_note(state, "사고 보고서"),
        f"사고 보고서 {len(todo)}건. 재구성·공백 코딩 요청은 정확한 요청, 누설 검사·묶음 추출·교차 코딩은 "
        "보고서 길이의 일정 비율로 만든 자리표시 요청(앞 단계의 출력이 입력이므로)",
    ]
    priced = await _price(
        calls,
        study="실증 ② 사후 보고서 회고",
        items=len(todo),
        refs=refs,
        effort=config.effort,
        compare=dry_run,
        notes=notes,
        counter=counter,
        echo=echo,
        next_item=todo[0].incident_id if todo and config.max_usd_run is not None and not dry_run else None,
    )
    config.out_dir.mkdir(parents=True, exist_ok=True)
    spent = state.ledger.prior_usd
    word = "사고 보고서"
    if dry_run:
        _save_table(priced.table, config.out_dir)
        _announce(echo, state, priced, item_word=word, max_usd=config.max_usd, max_usd_run=config.max_usd_run)
        return CommandResult(table=priced.table, dry_run=True, spent_before_usd=spent)
    echo(priced.table.render())
    _announce(echo, state, priced, item_word=word, max_usd=config.max_usd, max_usd_run=config.max_usd_run)
    _guard(
        priced, max_usd=config.max_usd, max_usd_run=config.max_usd_run, spent_usd=spent, confirmed=confirmed
    )
    tracker = SpendTracker(max_usd=config.max_usd, max_usd_run=config.max_usd_run, prior_usd=spent)
    main, a, b = refs
    clients = build_study_clients(
        main, a, b, effort=config.effort, tracker=tracker, anthropic_sdk=anthropic_sdk, openai_sdk=openai_sdk
    )
    try:
        outcome = await run_retro_study(config, clients, echo=echo, reports=reports)
    finally:
        await clients.aclose()
    return CommandResult(table=priced.table, dry_run=False, outcome=outcome, spent_before_usd=spent)
