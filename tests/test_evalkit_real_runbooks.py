"""Study 1 (public runbooks) end to end with fake models: pipeline, coders, metrics, resume, budget."""

from __future__ import annotations

import asyncio
import csv
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from jettae.domain import InputBundle, LlmMode
from jettae.evalkit.realdata.clients import budgeted
from jettae.evalkit.realdata.coding import FINDINGS_DOC_ID, CoderVerdict
from jettae.evalkit.realdata.diagnose import (
    BACKTRANSLATION_LLM,
    BACKTRANSLATION_SKIPPED,
    DiagnosisRecord,
    PipelineFallbackError,
    diagnose_bundle,
)
from jettae.evalkit.realdata.dryrun import AnthropicTokenCounter, CharTokenCounter, estimate_costs
from jettae.evalkit.realdata.entry import PaidRunRefusedError, run_real_runbooks
from jettae.evalkit.realdata.pricing import ModelRef
from jettae.evalkit.realdata.providers import BudgetExhausted, SpendTracker
from jettae.evalkit.realdata.runbooks import (
    REVIEW_COLUMNS,
    EvidenceRecord,
    FindingLabels,
    FindingRecord,
    RunbookResult,
    RunbookStudyConfig,
    load_runbooks,
    plan_runbook_calls,
    review_sample,
    run_runbook_study,
    source_url,
)
from jettae.ingest import load_bundle_dir
from jettae.llm.base import LLMError, LLMRefusalError
from jettae.llm.fake import FakeLLM
from jettae.orchestrator import demo_bundle_path
from jettae.orchestrator.pipeline import run_diagnosis
from jettae.orchestrator.state import DiagnosisOptions
from support_realdata import (
    FakeAnthropicSDK,
    FakeOpenAISDK,
    fake_clients,
    fake_handler,
    write_runbook_folder,
)

COMMIT = "a" * 40


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    return write_runbook_folder(tmp_path / "input")


def _config(folder: Path, out: Path, **overrides: Any) -> RunbookStudyConfig:
    return RunbookStudyConfig(input_dir=folder, out_dir=out, generated_at="2026-10-17", **overrides)


# --------------------------------------------------------------------------- inputs


def test_load_runbooks_skips_folder_notes_and_keeps_path_order(folder: Path) -> None:
    sources = load_runbooks(folder)
    assert [s.path for s in sources] == ["docs/payments/api.md", "docs/search/service.md"]
    assert sources[0].document.path == "docs/payments/api.md"
    assert load_runbooks(folder, limit=1)[0].path == "docs/payments/api.md"
    with pytest.raises(FileNotFoundError):
        load_runbooks(folder / "missing")


def test_source_url_points_at_the_pinned_commit() -> None:
    url = source_url("https://gitlab.com/gitlab-com/runbooks", "844a2b8", "docs/x.md", 12)
    assert url == "https://gitlab.com/gitlab-com/runbooks/-/blob/844a2b8/docs/x.md#L12"
    assert source_url("https://gitlab.com/gitlab-com/runbooks/", "c", "d.md") == (
        "https://gitlab.com/gitlab-com/runbooks/-/blob/c/d.md"
    )


# --------------------------------------------------------------------------- pipeline


def test_english_runbook_runs_hybrid_and_skips_llm_backtranslation(folder: Path) -> None:
    source = load_runbooks(folder)[0]
    fake = FakeLLM(fake_handler("main"))
    record = asyncio.run(
        diagnose_bundle(
            InputBundle(documents=(source.document,)),
            fake,
            concurrency=2,
            key_prefix=source.path,
            raw_texts={source.document.doc_id: source.raw},
            url_for=lambda doc, line: source_url("https://g", COMMIT, source.path, line),
        )
    )
    assert record.extractor == "hybrid"
    assert record.backtranslation == BACKTRANSLATION_SKIPPED
    assert record.backtranslation_skipped_calls >= 1  # the owned_by edge would have been judged
    assert {c.task for c in fake.calls} == {"extract"}  # no back-translation request reached the model
    assert record.findings
    owner = next(f for f in record.findings if f.rule_id == "JT-PATH-010")
    assert owner.family == "PATH"
    assert owner.key.startswith("docs/payments/api.md::JT-PATH-010:")
    quotes = {e.quote: e for e in owner.evidence}
    evidence = quotes["The payments API is owned by the payments team."]
    assert evidence.line == evidence.source_line == 3
    assert evidence.url == f"https://g/-/blob/{COMMIT}/docs/payments/api.md#L3"


def test_korean_documents_are_diagnosed_exactly_as_the_product_does() -> None:
    """The back-translation gate is transparent for Korean sources (no behaviour change)."""
    bundle = load_bundle_dir(demo_bundle_path())
    gated_fake = FakeLLM(fake_handler("main"))
    record = asyncio.run(diagnose_bundle(bundle, gated_fake, concurrency=2, key_prefix="demo"))
    plain_fake = FakeLLM(fake_handler("main"))
    options = DiagnosisOptions(rehearse=False, draft=False, llm_scenario_count=0, llm_concurrency=2)
    plain = asyncio.run(
        run_diagnosis(bundle, base_graph=None, llm=plain_fake, llm_mode=LlmMode.LIVE, options=options)
    )
    assert record.backtranslation == BACKTRANSLATION_LLM
    assert record.backtranslation_skipped_calls == 0
    assert record.graph_hash == plain.graph.content_hash()
    assert [f.finding_id for f in record.findings] == [f.id for f in plain.findings]
    tasks = [c.task for c in gated_fake.calls]
    assert tasks.count("backtranslate") == [c.task for c in plain_fake.calls].count("backtranslate") > 0


def test_an_llm_failure_is_not_recorded_as_a_measurement(folder: Path) -> None:
    source = load_runbooks(folder)[0]

    def outage(task: str, *_: Any) -> Any:
        return LLMError("rate limited")

    with pytest.raises(PipelineFallbackError, match="규칙 파서"):
        asyncio.run(
            diagnose_bundle(
                InputBundle(documents=(source.document,)),
                FakeLLM(outage),
                concurrency=1,
                key_prefix=source.path,
            )
        )


# --------------------------------------------------------------------------- the study


def test_study_writes_results_metrics_review_and_report(folder: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    clients, _main, _coder_a, _coder_b = fake_clients()
    outcome = asyncio.run(
        run_runbook_study(_config(folder, out, review_size=5), clients, echo=lambda _: None)
    )
    assert outcome.stop_reason is None
    assert [r.doc_path for r in outcome.results] == ["docs/payments/api.md", "docs/search/service.md"]
    lines = (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = RunbookResult.model_validate_json(lines[0])
    assert first.source_url == f"https://gitlab.com/gitlab-com/runbooks/-/blob/{COMMIT}/docs/payments/api.md"
    assert {item.key for item in first.labels} == {f.key for f in first.diagnosis.findings}
    assert all(item.coder_a.label is not None and item.coder_b.label is not None for item in first.labels)

    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["runbooks_done"] == metrics["runbooks_total"] == 2
    assert metrics["findings"] == sum(len(r.diagnosis.findings) for r in outcome.results)
    assert metrics["agreement"]["n"] == metrics["findings"]
    assert metrics["agreement"]["cohen_kappa"] is not None
    assert (
        metrics["precision"]["coder_a"]["n"]
        == metrics["labels_a"]["correct"] + metrics["labels_a"]["incorrect"]
    )
    families = {f["family"] for f in metrics["families"]}
    assert "PATH" in families
    assert "KR" not in families  # GitLab is diagnosed under its own (US) jurisdiction
    assert metrics["jurisdictions"] == ["US"]
    assert metrics["precision_ex_kr"] == metrics["precision"]
    assert metrics["rubric"] == "premise-claim-v4"
    assert metrics["questions"]["agreement_premise"]["n"] == metrics["findings"]
    assert metrics["split"] == "all"
    assert metrics["backtranslation_skipped_documents"] == 2
    assert metrics["source"] == f"https://gitlab.com/gitlab-com/runbooks@{COMMIT}"

    with (out / "review.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    assert tuple(rows[0]) == REVIEW_COLUMNS
    assert all(row["human_label"] == "" for row in rows)
    review = json.loads((out / "review.json").read_text(encoding="utf-8"))
    flat = json.dumps(review, ensure_ascii=False)
    assert "a 판정" not in flat
    assert "b 판정" not in flat
    assert "코더 판정은 이 파일에 없습니다" in review["instructions"]

    report = (out / "report.md").read_text(encoding="utf-8")
    for phrase in (
        "Cohen's κ",
        "Krippendorff's α",
        "정밀도는 재현율이 아니다",
        "영어 문서",
        "명부·카탈로그 없음",
        "판정 질문별 (전제·주장)",
        "관할(정책): US",
        "성숙한 SRE",
    ):
        assert phrase in report
    assert json.loads((out / "run.json").read_text(encoding="utf-8"))["model"] == "claude-opus-5-5"


def test_the_jurisdiction_is_the_policy_the_product_runs_under(folder: Path, tmp_path: Path) -> None:
    """Korean statutory rules apply to the study only when its jurisdiction says KR."""

    def families(**overrides: Any) -> set[str]:
        out = tmp_path / "-".join(overrides.get("jurisdictions", ("default",)))
        clients, *_ = fake_clients()
        outcome = asyncio.run(
            run_runbook_study(_config(folder, out, **overrides), clients, echo=lambda _: None)
        )
        return {f.family for r in outcome.results for f in r.diagnosis.findings}

    assert "KR" not in families()  # default: the source company's jurisdiction (US)
    assert "KR" in families(jurisdictions=("KR",))
    report = (tmp_path / "KR" / "report.md").read_text(encoding="utf-8")
    assert "관할(정책): KR — 한국 법정 신고 규칙(JT-KR-*)을 적용했다" in report


def test_coders_are_independent(folder: Path, tmp_path: Path) -> None:
    clients, _main, coder_a, coder_b = fake_clients()
    asyncio.run(run_runbook_study(_config(folder, tmp_path / "out"), clients, echo=lambda _: None))
    assert coder_a.calls
    assert coder_b.calls
    for call in coder_b.calls:
        text = "\n".join(d.text for d in call.documents) + call.context
        assert "a 판정" not in text  # coder B never sees coder A's labels or reasons
        assert [d.doc_id for d in call.documents][-1] == FINDINGS_DOC_ID
        assert call.documents[0].text.startswith("L1: ")  # the runbook, line-numbered
    for call in coder_a.calls:
        assert "b 판정" not in "\n".join(d.text for d in call.documents)


def test_resume_skips_finished_runbooks_and_refuses_other_settings(folder: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    clients, main, _a, _b = fake_clients()
    asyncio.run(run_runbook_study(_config(folder, out), clients, echo=lambda _: None))
    calls = len(main.calls)
    again, main_again, _a2, _b2 = fake_clients()
    outcome = asyncio.run(run_runbook_study(_config(folder, out), again, echo=lambda _: None))
    assert main_again.calls == []  # nothing redone
    assert len(outcome.results) == 2
    assert calls > 0
    with pytest.raises(ValueError, match="다른 설정"):
        asyncio.run(
            run_runbook_study(_config(folder, out, model="claude-sonnet-5-5"), again, echo=lambda _: None)
        )


def test_budget_stop_keeps_what_is_done_and_resumes(folder: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    cheap = {
        "model": "claude-haiku-5-5",
        "coder_a": "anthropic:claude-haiku-5-5",
        "coder_b": "openai:gpt-6-luna",
    }

    def run(max_usd: float, sdk: FakeAnthropicSDK) -> Any:
        return asyncio.run(
            run_real_runbooks(
                _config(folder, out, max_usd=max_usd, **cheap),
                dry_run=False,
                confirmed=True,
                echo=lambda _: None,
                counter=CharTokenCounter(1000.0),  # small estimate: the guard lets the run start
                anthropic_sdk=sdk,
                openai_sdk=FakeOpenAISDK(),
            )
        ).outcome

    # every Claude response costs 100k × $0.10/M + 10k × $0.50/M = $0.015
    first_sdk = FakeAnthropicSDK(input_tokens=100_000, output_tokens=10_000)
    stopped = run(0.05, first_sdk)
    assert stopped.stop_reason is not None
    assert "예산 상한" in stopped.stop_reason
    assert stopped.metrics.partial
    assert [r.doc_path for r in stopped.results] == ["docs/payments/api.md"]
    assert stopped.metrics.spend_this_run.spent_usd <= 0.05 + 0.015  # at most one call past the cap
    assert json.loads((out / "metrics.json").read_text(encoding="utf-8"))["partial"] is True
    # resume with a larger cap: finishes the rest without redoing the recorded runbook
    second_sdk = FakeAnthropicSDK(input_tokens=100_000, output_tokens=10_000)
    final = run(100.0, second_sdk)
    assert final.stop_reason is None
    assert [r.doc_path for r in final.results] == ["docs/payments/api.md", "docs/search/service.md"]
    assert not final.metrics.partial
    assert all("payments API" not in c["messages"][0]["content"] for c in second_sdk.created)
    assert final.metrics.spent_usd_results == pytest.approx(sum(r.usd for r in final.results))
    # the folder's total keeps every dollar: the runbook the first run stopped inside counts too
    assert final.metrics.spend_this_run is not None
    assert final.metrics.spent_usd_total == pytest.approx(
        stopped.metrics.spend_this_run.spent_usd + final.metrics.spend_this_run.spent_usd
    )
    assert final.metrics.spent_usd_total > final.metrics.spent_usd_results


def test_full_stack_with_fake_sdks(folder: Path, tmp_path: Path) -> None:
    """Real adapters (Anthropic structured outputs, OpenAI Responses) over fake SDKs, guard included."""
    sdk_a, sdk_o = FakeAnthropicSDK(), FakeOpenAISDK()
    result = asyncio.run(
        run_real_runbooks(
            _config(folder, tmp_path / "out", max_usd=50.0),
            dry_run=False,
            confirmed=True,
            echo=lambda _: None,
            counter=CharTokenCounter(),
            anthropic_sdk=sdk_a,
            openai_sdk=sdk_o,
        )
    )
    assert result.outcome is not None
    assert result.outcome.stop_reason is None
    models = {c["model"] for c in sdk_a.created}
    assert models == {"claude-opus-5-5"}
    assert all(c["model"] == "gpt-6.1-sol" and c["store"] is False for c in sdk_o.created)
    spend = result.outcome.metrics.spend_this_run
    assert spend is not None
    assert {line.model for line in spend.by_model} == {"claude-opus-5-5", "gpt-6.1-sol"}
    assert result.outcome.metrics.spent_usd_total == pytest.approx(spend.spent_usd, rel=1e-6)
    labels = [item for r in result.outcome.results for item in r.labels]
    assert labels
    assert all(item.coder_b.label is not None for item in labels)


# --------------------------------------------------------------------------- guard


def test_paid_run_guard(folder: Path, tmp_path: Path) -> None:
    def run(**kwargs: Any) -> Any:
        return asyncio.run(
            run_real_runbooks(
                _config(folder, tmp_path / "out", **kwargs.pop("config", {})),
                dry_run=False,
                echo=lambda _: None,
                counter=CharTokenCounter(),
                anthropic_sdk=_Exploding(),
                openai_sdk=_Exploding(),
                **kwargs,
            )
        )

    with pytest.raises(PaidRunRefusedError, match="--max-usd"):
        run(confirmed=True)
    with pytest.raises(PaidRunRefusedError, match="넘어 시작하지 않습니다"):
        run(confirmed=True, config={"max_usd": 0.0001})
    with pytest.raises(PaidRunRefusedError, match="--yes"):
        run(confirmed=False, config={"max_usd": 100.0})


class _Exploding:
    """An SDK that must never be called by a refused run."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"paid client used: {name}")


# --------------------------------------------------------------------------- dry run


def test_planned_extraction_requests_are_exactly_what_the_pipeline_sends(folder: Path) -> None:
    sources = load_runbooks(folder)
    planned = [
        c for c in plan_runbook_calls(sources, coder_batch=8, assumed_findings=4) if c.stage == "extract"
    ]
    sent: list[Any] = []
    for source in sources:
        fake = FakeLLM(fake_handler("main"))
        asyncio.run(
            diagnose_bundle(InputBundle(documents=(source.document,)), fake, concurrency=1, key_prefix="x")
        )
        sent += [c for c in fake.calls if c.task == "extract"]
    assert len(planned) == len(sent) == 2
    for plan, call in zip(planned, sent, strict=True):
        assert plan.documents == call.documents
        assert plan.context == call.context
        assert plan.instructions == call.instructions
        assert plan.schema is call.schema
        assert plan.max_tokens == call.max_tokens


def test_plan_contains_both_coders_and_placeholders(folder: Path) -> None:
    calls = plan_runbook_calls(load_runbooks(folder), coder_batch=5, assumed_findings=12)
    coder_calls = [c for c in calls if c.stage == "code_findings"]
    assert {c.role for c in coder_calls} == {"coder_a", "coder_b"}
    assert len(coder_calls) == 2 * 2 * 3  # 2 runbooks × 2 coders × ceil(12 / 5)
    assert not any(c.exact for c in coder_calls)
    assert all(c.exact for c in calls if c.stage == "extract")
    assert not any(c.stage == "backtranslate" for c in calls)  # English sources: not asked


def test_anthropic_counter_counts_the_real_request_once(folder: Path) -> None:
    sdk = FakeAnthropicSDK()
    counter = AnthropicTokenCounter(sdk)
    call = next(c for c in plan_runbook_calls(load_runbooks(folder), coder_batch=8, assumed_findings=1))
    ref = ModelRef("anthropic", "claude-opus-5-5")
    first = asyncio.run(counter.count(ref, "medium", call))
    second = asyncio.run(counter.count(ref, "medium", call))
    assert first == second > 0
    assert len(sdk.counted) == 1  # cached
    request = sdk.counted[0]
    assert request["model"] == "claude-opus-5-5"
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert request["messages"][0]["role"] == "user"
    assert "fallbacks" not in request
    assert "betas" not in request


def test_cost_table_compares_model_choices(folder: Path) -> None:
    calls = plan_runbook_calls(load_runbooks(folder), coder_batch=8, assumed_findings=12)
    table = asyncio.run(
        estimate_costs(
            calls,
            study="test",
            items=2,
            main=ModelRef("anthropic", "claude-opus-5-5"),
            coder_a=ModelRef("anthropic", "claude-opus-5-5"),
            coder_b=ModelRef("openai", "gpt-6-luna"),
            effort="medium",
            claude_counter=CharTokenCounter(3.0),
        )
    )
    assert len(table.combos) == 6
    assert table.chosen.main == "claude-opus-5-5"
    assert table.chosen.coder_b == "gpt-6-luna"
    by_main = {c.main: c.usd for c in table.combos if c.coder_b == "gpt-6-luna"}
    assert by_main["claude-opus-5-5"] > by_main["claude-sonnet-5-5"] > by_main["claude-haiku-5-5"]
    rendered = table.render()
    assert "← 선택" in rendered
    assert "조합별 합계" in rendered
    assert "count_tokens" in rendered


def test_dry_run_command_writes_the_table_and_sends_nothing(folder: Path, tmp_path: Path) -> None:
    out = tmp_path / "dry"
    result = asyncio.run(
        run_real_runbooks(
            _config(folder, out),
            dry_run=True,
            echo=lambda _: None,
            counter=CharTokenCounter(3.0),
            anthropic_sdk=_Exploding(),
            openai_sdk=_Exploding(),
        )
    )
    assert result.dry_run
    assert result.outcome is None
    assert (out / "dry_run.md").read_text(encoding="utf-8").startswith("실증 ① 공개 런북 비용 추정")
    assert json.loads((out / "dry_run.json").read_text(encoding="utf-8"))["items"] == 2
    assert not (out / "results.jsonl").exists()


# --------------------------------------------------------------------------- review sample


def _result(path: str, families: dict[str, int]) -> RunbookResult:
    findings = [
        FindingRecord(
            key=f"{path}::JT-{family}-001:{i}",
            finding_id=f"JT-{family}-001:{i}",
            rule_id=f"JT-{family}-001",
            family=family,
            severity="high",
            status="broken",
            title="t",
            message="m",
            service_id=None,
            service_name=None,
            element=None,
            doc_path=path,
            evidence=(EvidenceRecord(quote="q", line=1, source_line=1, url=f"https://x/{path}#L1"),),
        )
        for family, count in families.items()
        for i in range(count)
    ]
    diagnosis = DiagnosisRecord(
        graph_hash="h",
        extractor="hybrid",
        nodes=0,
        edges=0,
        findings=tuple(findings),
        holds=(),
        llm_calls=0,
        llm_by_task={},
        llm_input_tokens=0,
        llm_output_tokens=0,
        backtranslation=BACKTRANSLATION_SKIPPED,
        backtranslation_skipped_calls=0,
        security_notices=0,
        notes=(),
    )
    verdict = CoderVerdict(label="correct")
    return RunbookResult(
        doc_path=path,
        doc_sha256="s",
        title="t",
        korean=False,
        source_url=f"https://x/{path}",
        diagnosis=diagnosis,
        labels=tuple(FindingLabels(key=f.key, coder_a=verdict, coder_b=verdict) for f in findings),
        usd=0.0,
    )


def test_review_sample_is_stratified_seeded_and_blind() -> None:
    results = [_result("a.md", {"PATH": 120, "KR": 40}), _result("b.md", {"PPL": 30, "ESC": 2, "SPF": 1})]
    rows = review_sample(results, size=100, seed=7)
    assert len(rows) == 100
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.family] = counts.get(row.family, 0) + 1
    assert set(counts) == {"PATH", "KR", "PPL", "ESC", "SPF"}  # every family is represented
    assert counts["PATH"] > counts["KR"] > counts["PPL"] > counts["SPF"]  # roughly proportional
    assert rows == review_sample(results, size=100, seed=7)  # deterministic
    assert rows != review_sample(results, size=100, seed=8)
    assert len({r.finding_key for r in rows}) == 100
    assert [r.review_id for r in rows[:2]] == ["R001", "R002"]
    assert "label" not in json.dumps([r.model_dump() for r in rows]).replace("human_label", "")
    small = review_sample([_result("c.md", {"PATH": 3})], size=100, seed=1)
    assert len(small) == 3  # fewer findings than the sample size: all of them


# --------------------------------------------------------------------------- review fixes


def test_a_budget_stop_inside_the_extraction_is_not_swallowed(folder: Path) -> None:
    """The product turns an LLMError into its rule-parser fallback; the budget stop must escape."""
    source = load_runbooks(folder)[0]
    tracker = SpendTracker(max_usd=0.001)
    llm = budgeted(FakeLLM(fake_handler("main")), ModelRef("anthropic", "claude-opus-5-5"), "medium", tracker)
    with pytest.raises(BudgetExhausted):
        asyncio.run(
            diagnose_bundle(
                InputBundle(documents=(source.document,)), llm, concurrency=2, key_prefix=source.path
            )
        )


def test_chunk_local_extraction_failures_are_counted(folder: Path, tmp_path: Path) -> None:
    handler = fake_handler("main")

    def refuse_payments(task: str, schema: Any, documents: Any, context: str) -> Any:
        if task == "extract" and any("payments API" in d.text for d in documents):
            return LLMRefusalError("refused")
        return handler(task, schema, documents, context)

    source = load_runbooks(folder)[0]
    record = asyncio.run(
        diagnose_bundle(
            InputBundle(documents=(source.document,)),
            FakeLLM(refuse_payments),
            concurrency=1,
            key_prefix=source.path,
        )
    )
    assert record.extractor == "hybrid"  # the product carries on without the chunk ...
    assert record.extract_failed_chunks == 1  # ... and the study says so
    clients = dataclasses.replace(fake_clients()[0], main=FakeLLM(refuse_payments))
    outcome = asyncio.run(run_runbook_study(_config(folder, tmp_path / "out"), clients, echo=lambda _: None))
    assert outcome.metrics.runbooks_partial_extraction == 1
    assert outcome.metrics.extract_failed_chunks == 1
    assert "LLM 추출 일부 구간 실패" in (tmp_path / "out" / "report.md").read_text(encoding="utf-8")


def test_openai_input_is_priced_with_the_claude_count_of_the_same_request(folder: Path) -> None:
    sdk = FakeAnthropicSDK()
    counter = AnthropicTokenCounter(sdk)
    calls = plan_runbook_calls(load_runbooks(folder), coder_batch=8, assumed_findings=12)
    table = asyncio.run(
        estimate_costs(
            calls,
            study="test",
            items=2,
            main=ModelRef("anthropic", "claude-opus-5-5"),
            coder_a=ModelRef("anthropic", "claude-opus-5-5"),
            coder_b=ModelRef("openai", "gpt-6.1-sol"),
            effort="medium",
            claude_counter=counter,
            compare=False,
        )
    )
    rows = {r.role: r for r in table.rows if r.stage == "code_findings"}
    assert rows["coder_b"].input_tokens == rows["coder_a"].input_tokens
    assert all(r["model"] == "claude-opus-5-5" for r in sdk.counted)  # never an OpenAI model id
    # coder B's requests are coder A's text: answered from the cache, no extra count_tokens call
    assert counter.requests == len(sdk.counted) == sum(1 for c in calls if c.role != "coder_b")
    assert "claude-opus-5-5 토큰 수로 대신함" in table.coder_b_counter
