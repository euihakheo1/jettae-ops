"""Real-data harness building blocks: key loading, prices, families, language, agreement."""

from __future__ import annotations

import asyncio
import math
import typing
from pathlib import Path

import pytest
from pydantic import BaseModel

from jettae.check.registry import rule_catalog
from jettae.evalkit.realdata.coding import FamilyCode, UnitFamily
from jettae.evalkit.realdata.common import (
    FAMILY_CODES,
    FAMILY_LABEL_KO,
    family_catalog_text,
    is_korean_text,
    label_counts,
    measure_agreement,
    precision,
    read_jsonl,
    rule_family,
)
from jettae.evalkit.realdata.diagnose import raw_line
from jettae.evalkit.realdata.dryrun import PlannedCall, estimate_costs
from jettae.evalkit.realdata.keys import load_api_key, parse_dotenv
from jettae.evalkit.realdata.pricing import (
    CLAUDE_THINKING_TOKENS,
    PRICES,
    ModelRef,
    assumptions_text,
    call_usd,
    price_of,
    supports_server_fallbacks,
    thinking_allowance,
    visible_output_tokens,
)
from jettae.evalkit.realdata.providers import SpendTracker
from jettae.evalkit.realdata.runbooks import FindingLabels
from jettae.llm.anthropic_client import Effort

# --------------------------------------------------------------------------- keys


def test_parse_dotenv_handles_quotes_comments_and_export() -> None:
    text = (
        "﻿# keys\n"
        "ANTHROPIC_API_KEY=sk-ant-test\n"
        "export OPENAI_API_KEY='sk-openai test'\n"
        'QUOTED="a # not a comment"\n'
        "INLINE=value # comment\n"
        "EMPTY=\n"
        "not a line\n"
        "=novalue\n"
        "ANTHROPIC_API_KEY=second\n"
    )
    values = parse_dotenv(text)
    assert values["ANTHROPIC_API_KEY"] == "second"  # last assignment wins, like a shell
    assert values["OPENAI_API_KEY"] == "sk-openai test"
    assert values["QUOTED"] == "a # not a comment"
    assert values["INLINE"] == "value"
    assert values["EMPTY"] == ""
    assert "not a line" not in values
    assert "" not in values


def test_load_api_key_prefers_environment_then_dotenv(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("ANTHROPIC_API_KEY=from-file\nOPENAI_API_KEY=\n", encoding="utf-8")
    assert load_api_key(
        "ANTHROPIC_API_KEY", environ={"ANTHROPIC_API_KEY": "from-env"}, dotenv_paths=[dotenv]
    ) == ("from-env")
    # an empty environment value falls through to the file
    assert (
        load_api_key("ANTHROPIC_API_KEY", environ={"ANTHROPIC_API_KEY": " "}, dotenv_paths=[dotenv])
        == "from-file"
    )
    assert load_api_key("OPENAI_API_KEY", environ={}, dotenv_paths=[dotenv]) is None
    assert load_api_key("ANTHROPIC_API_KEY", environ={}, dotenv_paths=[tmp_path / "missing.env"]) is None


# --------------------------------------------------------------------------- prices


def test_model_ref_parsing() -> None:
    assert ModelRef.parse("anthropic:claude-opus-5-5") == ModelRef("anthropic", "claude-opus-5-5")
    assert ModelRef.parse("claude-haiku-5-5") == ModelRef("anthropic", "claude-haiku-5-5")
    assert ModelRef.parse(" openai:gpt-6-luna ") == ModelRef("openai", "gpt-6-luna")
    assert ModelRef.parse("gpt-6.1-sol") == ModelRef("openai", "gpt-6.1-sol")
    assert str(ModelRef("openai", "gpt-6-luna")) == "openai:gpt-6-luna"
    for bad in ("google:gemini", "anthropic:", ":x"):
        with pytest.raises(ValueError, match="형식"):
            ModelRef.parse(bad)


def test_official_prices_and_cost_math() -> None:
    assert PRICES["claude-opus-5-5"].input_per_mtok == 4.0
    assert PRICES["claude-opus-5-5"].output_per_mtok == 20.0
    assert PRICES["claude-sonnet-5-5"].output_per_mtok == 10.0
    assert PRICES["claude-haiku-5-5"].input_per_mtok == 0.10
    assert PRICES["gpt-6-luna"].output_per_mtok == 0.50
    assert PRICES["gpt-6.1-sol"].input_per_mtok == 2.00
    assert math.isclose(call_usd("claude-opus-5-5", input_tokens=1_000_000, output_tokens=1_000_000), 24.0)
    # cache reads at 0.1×, cache writes at 1.25× the input price
    assert math.isclose(
        call_usd("claude-opus-5-5", input_tokens=0, output_tokens=0, cache_read_tokens=1_000_000), 0.4
    )
    assert math.isclose(
        call_usd("claude-opus-5-5", input_tokens=0, output_tokens=0, cache_write_tokens=1_000_000), 5.0
    )
    # a model that served a fallback, or an unknown one, is never priced below the list
    assert price_of("claude-opus-4-8").input_per_mtok == 5.0
    assert price_of("some-future-model").output_per_mtok == 50.0


def test_haiku_long_prompt_tier_is_priced_per_request() -> None:
    """Haiku 5.5: 0.10/0.50 up to a 100K-token prompt, 0.50/2.50 for the whole request above it."""
    at_limit = call_usd("claude-haiku-5-5", input_tokens=100_000, output_tokens=1_000)
    assert math.isclose(at_limit, (100_000 * 0.10 + 1_000 * 0.50) / 1e6)
    above = call_usd("claude-haiku-5-5", input_tokens=100_001, output_tokens=1_000)
    assert math.isclose(above, (100_001 * 0.50 + 1_000 * 2.50) / 1e6)
    # cached prompt tokens count toward the prompt size
    cached = call_usd("claude-haiku-5-5", input_tokens=10, output_tokens=0, cache_read_tokens=100_000)
    assert math.isclose(cached, (10 + 100_000 * 0.1) * 0.50 / 1e6)
    # flat-priced models are unaffected
    assert math.isclose(call_usd("claude-sonnet-5-5", input_tokens=400_000, output_tokens=0), 0.8)
    # actual spend is metered at the tier the response was billed at (the cap never under-counts)
    tracker = SpendTracker()
    tracker.record("claude-haiku-5-5", input_tokens=200_000, output_tokens=0)
    assert tracker.spent_usd == pytest.approx(0.10)
    assert any("100K 토큰 초과 호출은 0.5/2.5" in line for line in assumptions_text("medium"))


class _OneAnswer(BaseModel):
    answer: str


class _FixedCounter:
    method = "fixed"

    def __init__(self, counts: dict[str, int]) -> None:
        self.counts = counts

    async def count(self, ref: ModelRef, effort: Effort, call: PlannedCall) -> int:
        return self.counts[call.item]


def test_dry_run_prices_each_call_on_its_own_tier() -> None:
    """A 50K and a 150K prompt: only the second is long — summing first would price both long."""
    calls = [
        PlannedCall(
            item=item,
            stage="leakage",
            role="main",
            instructions="x",
            schema=_OneAnswer,
            documents=(),
            context="",
            max_tokens=8_000,
        )
        for item in ("short", "long")
    ]
    haiku = ModelRef("anthropic", "claude-haiku-5-5")
    table = asyncio.run(
        estimate_costs(
            calls,
            study="t",
            items=2,
            main=haiku,
            coder_a=haiku,
            coder_b=ModelRef("openai", "gpt-6-luna"),
            effort="medium",
            claude_counter=_FixedCounter({"short": 50_000, "long": 150_000}),
            compare=False,
        )
    )
    [row] = [r for r in table.rows if r.role == "main"]
    out = calls[0].output_tokens(haiku, "medium")
    expected = call_usd(haiku.model, input_tokens=50_000, output_tokens=out) + call_usd(
        haiku.model, input_tokens=150_000, output_tokens=out
    )
    assert (row.input_tokens, row.output_tokens) == (200_000, 2 * out)
    assert row.usd == round(expected, 4)
    assert row.usd < call_usd(haiku.model, input_tokens=200_000, output_tokens=2 * out)


def test_fallbacks_and_thinking_allowances() -> None:
    assert supports_server_fallbacks("claude-opus-5-5")
    assert supports_server_fallbacks("claude-sonnet-5-5")
    assert not supports_server_fallbacks("claude-haiku-5-5")
    assert thinking_allowance("anthropic", "claude-opus-5-5", "medium") == CLAUDE_THINKING_TOKENS["medium"]
    assert thinking_allowance("anthropic", "claude-opus-5-5", "high") > thinking_allowance(
        "anthropic", "claude-opus-5-5", "low"
    )
    assert thinking_allowance("openai", "gpt-6.1-sol", "medium") > thinking_allowance(
        "openai", "gpt-6-luna", "medium"
    )


def test_visible_output_estimates() -> None:
    english = visible_output_tokens("extract", chars=4000)
    korean = visible_output_tokens("extract", chars=4000, korean=True)
    assert english == 3200  # 4000 chars / 2.5 per token (measured 2.6) × 2.0
    assert korean > english
    assert visible_output_tokens("code_findings", items=8) == 40 + 80 * 8
    assert visible_output_tokens("code_gaps") > visible_output_tokens("code_gaps", items=1)


# --------------------------------------------------------------------------- families & language


def test_family_codes_match_the_rule_registry() -> None:
    assert set(FAMILY_CODES) == {rule_family(r.rule_id) for r in rule_catalog()}
    assert set(FAMILY_LABEL_KO) == set(FAMILY_CODES)
    assert set(typing.get_args(FamilyCode)) == {*FAMILY_CODES, "out_of_scope"}
    assert set(typing.get_args(UnitFamily)) == {*FAMILY_CODES, "out_of_scope", "not_a_gap"}
    text = family_catalog_text()
    for rule in rule_catalog():
        assert rule.rule_id in text


def test_rule_family() -> None:
    assert rule_family("JT-PATH-010") == "PATH"
    assert rule_family("JT-KR-001") == "KR"
    assert rule_family("custom") == "custom"


def test_korean_detection() -> None:
    assert is_korean_text("결제 서비스 장애 시 1차 당직 김민수 대리가 슬랙 #pay-alert 채널로 알린다.")
    assert is_korean_text(
        "결제 API(payments-api)는 k8s prod 클러스터에서 운영한다. kubectl rollout undo 실행"
    )
    assert not is_korean_text("When the error rate alarm fires, page the on-call engineer in #db-oncall.")
    assert not is_korean_text("12345 ... ---")


def test_raw_line_prefers_the_occurrence_nearest_the_normalised_line() -> None:
    raw = "a\nquote here\nb\nc\nquote here\n"
    assert raw_line(raw, "quote here", 5) == 5
    assert raw_line(raw, "quote here", 1) == 2
    assert raw_line(raw, "missing", 3) == 3
    assert raw_line(None, "quote here", 4) == 4


# --------------------------------------------------------------------------- agreement


def test_agreement_and_precision() -> None:
    a = ["correct", "correct", "incorrect", "unclear", None]
    b = ["correct", "incorrect", "incorrect", "unclear", "correct"]
    agreement = measure_agreement(a, b)
    assert agreement.n == 4  # the unit coder A did not label is left out
    assert agreement.observed == 0.75
    assert agreement.cohen_kappa is not None
    assert -1.0 <= agreement.cohen_kappa <= 1.0
    assert agreement.krippendorff_alpha is not None
    rate = precision(["correct", "correct", "incorrect", "unclear", None])
    assert (rate.hits, rate.n) == (2, 3)
    assert rate.ci95 is not None
    assert rate.ci95[0] < 2 / 3 < rate.ci95[1]
    counts = label_counts(["correct", None, "unclear"])
    assert (counts.correct, counts.unclear, counts.missing) == (1, 1, 1)
    assert measure_agreement([None], ["correct"]).cohen_kappa is None
    with pytest.raises(ValueError, match="판정 수"):
        measure_agreement(["correct"], [])


def test_read_jsonl_ignores_a_torn_last_line(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    good = FindingLabels.model_validate(
        {"key": "k", "coder_a": {"label": "correct"}, "coder_b": {"label": None, "error": "omitted"}}
    )
    path.write_text(good.model_dump_json() + "\n" + '{"key": "torn', encoding="utf-8")
    assert read_jsonl(path, FindingLabels) == [good]
    assert read_jsonl(tmp_path / "none.jsonl", FindingLabels) == []
