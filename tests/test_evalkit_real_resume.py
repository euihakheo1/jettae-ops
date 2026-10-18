"""Resuming and growing a study in one output folder: the cumulative spend ledger, the resume guard,
and extension to a superset of the inputs — all offline, with fake models.

Exact budget tests use :func:`support_realdata.metered_clients`: every call reserves a fixed amount
before it is sent and costs a smaller fixed amount after, through the real budget gate.
"""

from __future__ import annotations

import asyncio
import csv
import json
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from jettae.evalkit.realdata.dryrun import CharTokenCounter
from jettae.evalkit.realdata.entry import PaidRunRefusedError, run_real_runbooks, run_retro
from jettae.evalkit.realdata.providers import BudgetExhausted, SpendLine, SpendTracker
from jettae.evalkit.realdata.resume import (
    INTERRUPTED_LINE,
    LEDGER_FILE,
    LEGACY_LINE,
    MANIFEST_FILE,
    SUPERSEDED_FILE,
    InputChangeError,
    ItemSpend,
    LedgerData,
    RunSpend,
    SpendLedger,
)
from jettae.evalkit.realdata.retro import IncidentResult, RetroConfig, run_retro_study
from jettae.evalkit.realdata.runbooks import RunbookResult, RunbookStudyConfig, run_runbook_study
from support_realdata import (
    FakeAnthropicSDK,
    FakeOpenAISDK,
    fake_clients,
    metered_clients,
    write_reports,
    write_runbook_folder,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_REPORT = ROOT / "examples" / "retro-reports" / "fictional-parcel-db-outage.md"
PAYMENTS = "docs/payments/api.md"
SEARCH = "docs/search/service.md"
#: what every metered call costs, and what it reserves before it is sent
COST, RESERVE = 0.004, 0.01


def quiet(_: str) -> None:
    return None


@pytest.fixture
def runner() -> Iterator[asyncio.Runner]:
    """One event loop for every run of a test (fewer loops to create on Windows)."""
    with asyncio.Runner() as loop:
        yield loop


def _ledger(out: Path) -> dict[str, Any]:
    return json.loads((out / LEDGER_FILE).read_text(encoding="utf-8"))


def _rb_config(folder: Path, out: Path, **overrides: Any) -> RunbookStudyConfig:
    return RunbookStudyConfig(
        input_dir=folder, out_dir=out, generated_at="2026-10-17", concurrency=1, **overrides
    )


def _retro_config(reports: Path, out: Path, **overrides: Any) -> RetroConfig:
    return RetroConfig(reports_dir=reports, out_dir=out, generated_at="2026-10-17", **overrides)


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    return write_runbook_folder(tmp_path / "input")


@pytest.fixture
def subset(tmp_path: Path) -> Path:
    """The same runbook folder without the search runbook (a strict subset)."""
    root = write_runbook_folder(tmp_path / "subset")
    shutil.rmtree(root / "docs" / "search")
    return root


@pytest.fixture
def calls_per_runbook(runner: asyncio.Runner, folder: Path, tmp_path: Path) -> dict[str, int]:
    """Metered calls each runbook takes (measured once in a scratch folder)."""
    clients, *_ = metered_clients(usd=COST, reserve=RESERVE)
    runner.run(run_runbook_study(_rb_config(folder, tmp_path / "probe"), clients, echo=quiet))
    items = _ledger(tmp_path / "probe")["items"]
    return {path: items[path]["calls"] for path in (PAYMENTS, SEARCH)}


def _run_rb(runner: asyncio.Runner, config: RunbookStudyConfig, **clients_kw: Any) -> Any:
    clients, main, _a, _b = metered_clients(usd=COST, reserve=RESERVE, **clients_kw)
    outcome = runner.run(run_runbook_study(config, clients, echo=quiet))
    return outcome, main


def _docs(fake: Any, task: str) -> set[str]:
    return {d.doc_id for c in fake.calls if c.task == task for d in c.documents}


def _extracted(fake: Any) -> set[str]:
    """Which test runbooks the main model was asked to extract."""
    texts = [" ".join(d.text for d in c.documents) for c in fake.calls if c.task == "extract"]
    marks = {PAYMENTS: "payments API", SEARCH: "search service"}
    return {path for path, mark in marks.items() if any(mark in text for text in texts)}


# --------------------------------------------------------------------------- tracker


def test_tracker_caps_are_cumulative_and_per_run() -> None:
    tracker = SpendTracker(max_usd=1.0, prior_usd=0.7)
    tracker.reserve(0.2, what="extract")  # 0.7 earlier + 0.2 ≤ 1.0
    with pytest.raises(BudgetExhausted, match="누적 예산 상한") as total:
        tracker.reserve(0.2, what="extract")  # 0.7 + 0.2 in flight + 0.2 > 1.0
    assert total.value.scope == "total"
    assert "이전 실행 $0.7000" in str(total.value)

    per_run = SpendTracker(max_usd=10.0, max_usd_run=0.5, prior_usd=9.0)
    per_run.record("claude-opus-5-5", input_tokens=100_000, output_tokens=0)  # $0.40
    with pytest.raises(BudgetExhausted, match="--max-usd-run") as run:
        per_run.reserve(0.2, what="extract")  # this run 0.4 + 0.2 > 0.5 (cumulative 9.6 ≤ 10)
    assert run.value.scope == "run"
    report = per_run.report()
    assert (report.prior_usd, report.max_usd, report.max_usd_run) == (9.0, 10.0, 0.5)

    changes: list[int] = []
    hooked = SpendTracker(on_change=lambda: changes.append(1))
    hooked.reserve(0.1, what="x")
    hooked.release(0.1)
    hooked.record("claude-haiku-5-5", input_tokens=10, output_tokens=10)
    hooked.record_unmetered(0.1)
    assert len(changes) == 4  # every change reaches the ledger


# --------------------------------------------------------------------------- ledger


def test_ledger_records_per_model_per_item_and_accumulates_over_runs(
    runner: asyncio.Runner, folder: Path, tmp_path: Path, calls_per_runbook: dict[str, int]
) -> None:
    out = tmp_path / "out"
    first, _main = _run_rb(runner, _rb_config(folder, out, limit=1))
    assert [r.doc_path for r in first.results] == [PAYMENTS]
    ledger = _ledger(out)
    n1, n2 = calls_per_runbook[PAYMENTS], calls_per_runbook[SEARCH]
    assert ledger["study"] == "real-runbooks"
    assert ledger["spent_usd"] == pytest.approx(n1 * COST)
    [line] = ledger["by_model"]
    assert line["model"] == "claude-haiku-5-5"
    assert line["calls"] == n1
    assert line["output_tokens"] == n1 * 8000
    assert {"input_tokens", "cache_read_tokens", "cache_write_tokens", "usd"} <= set(line)
    item = ledger["items"][PAYMENTS]
    assert (item["status"], item["calls"], item["attempts"]) == ("done", n1, 1)
    assert item["usd"] == pytest.approx(first.results[0].usd)
    assert item["by_model"][0]["model"] == "claude-haiku-5-5"
    [run] = ledger["runs"]
    assert run["finished_at"] is not None
    assert (run["prior_usd"], run["items_done"], run["in_flight_usd"]) == (0.0, 1, 0.0)

    # the next run loads the ledger: the folder total is both runs, each item keeps its own cost
    second, main = _run_rb(runner, _rb_config(folder, out))
    assert _extracted(main) == {SEARCH}  # only the unfinished runbook was processed
    ledger = _ledger(out)
    assert ledger["spent_usd"] == pytest.approx((n1 + n2) * COST)
    assert ledger["items"][SEARCH]["calls"] == n2
    assert ledger["items"][PAYMENTS]["attempts"] == 1
    assert [r["prior_usd"] for r in ledger["runs"]] == [0.0, pytest.approx(n1 * COST)]
    assert second.metrics.spent_usd_total == pytest.approx((n1 + n2) * COST)
    assert second.metrics.spend_this_run.spent_usd == pytest.approx(n2 * COST)
    assert second.metrics.spend_this_run.prior_usd == pytest.approx(n1 * COST)


def test_the_ledger_is_written_after_every_call(
    runner: asyncio.Runner, folder: Path, tmp_path: Path, calls_per_runbook: dict[str, int]
) -> None:
    out = tmp_path / "out"
    clients, *fakes = metered_clients(usd=COST, reserve=RESERVE)
    seen: list[tuple[float, float, str]] = []

    def spy(handler: Any) -> Any:
        def handle(*args: Any) -> Any:
            ledger = _ledger(out)  # read from disk while this call is in flight
            item = ledger["items"][PAYMENTS]
            seen.append((ledger["spent_usd"], ledger["runs"][-1]["in_flight_usd"], item["status"]))
            return handler(*args)

        return handle

    for fake in fakes:
        fake.handler = spy(fake.handler)
    runner.run(run_runbook_study(_rb_config(folder, out, limit=1), clients, echo=quiet))
    # call k (sequential): k-1 calls already booked on disk, this call's reservation in flight
    expected = [(pytest.approx(k * COST), pytest.approx(RESERVE), "running") for k in range(len(seen))]
    assert seen == expected
    assert len(seen) == calls_per_runbook[PAYMENTS]
    assert _ledger(out)["items"][PAYMENTS]["status"] == "done"


def test_a_killed_run_is_booked_at_its_reservations_on_the_next_load(tmp_path: Path) -> None:
    line = SpendLine(
        model="claude-opus-5-5",
        calls=2,
        input_tokens=1000,
        cache_write_tokens=0,
        cache_read_tokens=0,
        output_tokens=500,
        usd=0.5,
    )
    killed = LedgerData(
        study="retro",
        spent_usd=0.5,
        by_model=(line,),
        items={"incident-01": ItemSpend(usd=0.5, calls=2, by_model=(line,), attempts=1, status="running")},
        runs=(
            RunSpend(
                run=1,
                started_at="2026-10-17T00:00:00+00:00",
                input_dir="C:/reports-b",
                max_usd=5.0,
                max_usd_run=None,
                prior_usd=0.0,
                spent_usd=0.5,
                in_flight_usd=0.3,
            ),
        ),
    )
    (tmp_path / LEDGER_FILE).write_text(killed.model_dump_json(), encoding="utf-8")
    ledger = SpendLedger.load(tmp_path, study="retro")
    assert ledger.prior_usd == pytest.approx(0.8)  # the calls in flight count (never under-count)
    data = ledger.data()
    assert data.runs[0].interrupted
    assert data.items["incident-01"].status == "interrupted"
    assert {x.model for x in data.by_model} == {"claude-opus-5-5", INTERRUPTED_LINE}
    assert SpendLedger.load(tmp_path, study="retro").prior_usd == pytest.approx(0.8)  # idempotent
    tracker = SpendTracker(max_usd=5.0)
    ledger.begin_run(tracker, input_dir=tmp_path)
    assert tracker.prior_usd == pytest.approx(0.8)
    ledger.finish_run(stop_reason=None)
    again = SpendLedger.load(tmp_path, study="retro")
    assert again.prior_usd == pytest.approx(0.8)  # booked once, persisted
    assert len(again.data().runs) == 2
    with pytest.raises(ValueError, match="다른 실증"):
        SpendLedger.load(tmp_path, study="real-runbooks")


def test_a_folder_without_a_ledger_imports_its_results(
    runner: asyncio.Runner, folder: Path, tmp_path: Path, calls_per_runbook: dict[str, int]
) -> None:
    out = tmp_path / "out"
    _run_rb(runner, _rb_config(folder, out, limit=1))
    (out / LEDGER_FILE).unlink()  # a folder written before the ledger existed
    lines: list[str] = []
    result = runner.run(
        run_real_runbooks(
            _rb_config(folder, out),
            dry_run=True,
            echo=lines.append,
            counter=CharTokenCounter(3.0),
        )
    )
    assert result.spent_before_usd == pytest.approx(calls_per_runbook[PAYMENTS] * COST)
    assert any("spend.json이 없던 이전 실행" in line for line in lines)
    _run_rb(runner, _rb_config(folder, out))
    ledger = _ledger(out)
    assert LEGACY_LINE in {x["model"] for x in ledger["by_model"]}
    total = sum(calls_per_runbook.values()) * COST
    assert ledger["spent_usd"] == pytest.approx(total)


# --------------------------------------------------------------------------- cumulative cap


def test_budget_stop_then_resume_never_passes_the_cumulative_cap(
    runner: asyncio.Runner, folder: Path, tmp_path: Path, calls_per_runbook: dict[str, int]
) -> None:
    out = tmp_path / "out"
    n1, n2 = calls_per_runbook[PAYMENTS], calls_per_runbook[SEARCH]
    # room for the first runbook and one call of the second; the call after it cannot reserve
    cap = (n1 + 1) * COST + RESERVE - COST / 2

    stopped, _main = _run_rb(runner, _rb_config(folder, out, max_usd=cap), max_usd=cap)
    assert stopped.stop_kind == "budget_total"
    assert "누적 예산 상한" in stopped.stop_reason
    assert [r.doc_path for r in stopped.results] == [PAYMENTS]
    ledger = _ledger(out)
    assert ledger["spent_usd"] == pytest.approx((n1 + 1) * COST)
    assert ledger["spent_usd"] <= cap
    assert ledger["items"][SEARCH]["status"] == "stopped"
    assert ledger["items"][SEARCH]["calls"] == 1

    # the same command again: the cap is cumulative, so this run cannot add a cent
    again, main = _run_rb(runner, _rb_config(folder, out, max_usd=cap), max_usd=cap)
    assert again.stop_kind == "budget_total"
    assert main.calls == []  # refused before anything was sent
    assert _ledger(out)["spent_usd"] == pytest.approx((n1 + 1) * COST)
    assert again.metrics.spent_usd_total <= cap
    assert again.metrics.spend_this_run.spent_usd == 0.0

    # a larger cap finishes the rest; the finished runbook is not redone
    final, main = _run_rb(runner, _rb_config(folder, out, max_usd=1.0), max_usd=1.0)
    assert final.stop_reason is None
    assert _extracted(main) == {SEARCH}
    assert [r.doc_path for r in final.results] == [PAYMENTS, SEARCH]
    ledger = _ledger(out)
    # every dollar is in the folder total: the call wasted on the stopped attempt included
    assert ledger["spent_usd"] == pytest.approx((n1 + 1 + n2) * COST)
    assert ledger["items"][SEARCH]["attempts"] == 3
    assert ledger["items"][SEARCH]["status"] == "done"
    assert [r["spent_usd"] for r in ledger["runs"]] == [
        pytest.approx((n1 + 1) * COST),
        0.0,
        pytest.approx(n2 * COST),
    ]
    assert final.metrics.spent_usd_total == pytest.approx(ledger["spent_usd"])
    assert final.metrics.spent_usd_results == pytest.approx(sum(r.usd for r in final.results))


def test_the_per_run_cap_stops_one_invocation_only(
    runner: asyncio.Runner, folder: Path, tmp_path: Path, calls_per_runbook: dict[str, int]
) -> None:
    out = tmp_path / "out"
    n1, n2 = calls_per_runbook[PAYMENTS], calls_per_runbook[SEARCH]
    total_cap = (n1 + n2) * COST + RESERVE  # exactly enough for both runbooks
    first_run_cap = n1 * COST + RESERVE - COST / 2  # the first runbook, not one call more

    first, _main = _run_rb(
        runner,
        _rb_config(folder, out, max_usd=total_cap, max_usd_run=first_run_cap),
        max_usd=total_cap,
        max_usd_run=first_run_cap,
    )
    assert first.stop_kind == "budget_run"
    assert "이번 실행 예산 상한 --max-usd-run" in first.stop_reason
    assert [r.doc_path for r in first.results] == [PAYMENTS]
    # a fresh allowance for the next invocation, within the cumulative cap
    second, main = _run_rb(
        runner,
        _rb_config(folder, out, max_usd=total_cap, max_usd_run=n2 * COST + RESERVE),
        max_usd=total_cap,
        max_usd_run=n2 * COST + RESERVE,
    )
    assert second.stop_reason is None
    assert _extracted(main) == {SEARCH}
    ledger = _ledger(out)
    assert ledger["spent_usd"] == pytest.approx((n1 + n2) * COST)
    assert ledger["spent_usd"] <= total_cap
    assert [r["max_usd_run"] for r in ledger["runs"]] == [first_run_cap, n2 * COST + RESERVE]
    # the cumulative cap wins over a roomy per-run cap
    other = tmp_path / "other"
    capped, _main = _run_rb(
        runner,
        _rb_config(folder, other, max_usd=first_run_cap, max_usd_run=1.0),
        max_usd=first_run_cap,
        max_usd_run=1.0,
    )
    assert capped.stop_kind == "budget_total"


# --------------------------------------------------------------------------- resume guard (entry)


def _retro_paid(
    runner: asyncio.Runner,
    config: RetroConfig,
    *,
    echo: Callable[[str], object] = quiet,
    counter: Any = None,
    confirmed: bool = True,
) -> Any:
    return runner.run(
        run_retro(
            config,
            dry_run=False,
            confirmed=confirmed,
            echo=echo,
            counter=counter or CharTokenCounter(1000.0),  # small estimate: the guard lets it start
            anthropic_sdk=FakeAnthropicSDK(),
            openai_sdk=FakeOpenAISDK(),
        )
    )


def test_retro_resume_guard_prices_only_unfinished_items_against_the_remaining_budget(
    runner: asyncio.Runner, tmp_path: Path
) -> None:
    subset = write_reports(tmp_path / "reports-b", 2, source=FIXTURE_REPORT)
    full = write_reports(tmp_path / "reports", 4, source=FIXTURE_REPORT)
    out = tmp_path / "retro"
    first = _retro_paid(runner, _retro_config(subset, out, max_usd=50.0))
    assert first.outcome.stop_reason is None
    spent = _ledger(out)["spent_usd"]
    assert spent > 0
    assert first.outcome.metrics.spent_usd_total == pytest.approx(spent)

    # dry run over the full folder: only the two new reports are priced
    lines: list[str] = []
    dry = runner.run(
        run_retro(_retro_config(full, out), dry_run=True, echo=lines.append, counter=CharTokenCounter(3.0))
    )
    assert dry.table.items == 2
    assert dry.spent_before_usd == pytest.approx(spent)
    one_shot = runner.run(
        run_retro(
            _retro_config(full, tmp_path / "fresh"), dry_run=True, echo=quiet, counter=CharTokenCounter(3.0)
        )
    )
    assert one_shot.table.items == 4
    assert dry.table.chosen.usd == pytest.approx(one_shot.table.chosen.usd / 2, rel=0.01)
    text = "\n".join(lines)
    assert "이미 끝난 사고 보고서 2건" in text
    assert "남은 2건 추정" in text
    assert f"${spent:,.4f}" in text
    assert "입력 폴더가 이전 실행과 다릅니다" in text

    # the cap is compared with what is left of it, not with the cap itself
    estimate = runner.run(
        run_retro(_retro_config(full, out), dry_run=True, echo=quiet, counter=CharTokenCounter(3.0))
    ).table.chosen.usd
    with pytest.raises(PaidRunRefusedError, match="남은 예산") as refused:
        _retro_paid(
            runner, _retro_config(full, out, max_usd=spent + estimate / 2), counter=CharTokenCounter(3.0)
        )
    assert f"누적 지출 ${spent:,.4f}" in str(refused.value)
    with pytest.raises(PaidRunRefusedError, match="이미 누적 상한"):
        _retro_paid(runner, _retro_config(full, out, max_usd=spent), counter=CharTokenCounter(3.0))
    # but a cap that only covers the new items' estimate on top of the spend is enough
    lines.clear()
    grown = _retro_paid(
        runner,
        _retro_config(full, out, max_usd=spent + estimate * 1.01),
        echo=lines.append,
        counter=CharTokenCounter(3.0),
    )
    assert grown.outcome.stop_reason is None
    text = "\n".join(lines)
    assert "남은 예산: $" in text
    assert "누적 상한 --max-usd" in text
    ledger = _ledger(out)
    assert ledger["spent_usd"] <= spent + estimate * 1.01
    assert ledger["spent_usd"] == pytest.approx(spent + grown.outcome.metrics.spend_this_run.spent_usd)
    assert sorted(ledger["items"]) == [f"incident-{i:02d}" for i in range(1, 5)]
    assert grown.outcome.metrics.incidents_done == 4


def test_a_refused_start_writes_no_settings(folder: Path, tmp_path: Path, runner: asyncio.Runner) -> None:
    out = tmp_path / "out"

    def start(model: str, confirmed: bool) -> Any:
        return runner.run(
            run_real_runbooks(
                _rb_config(folder, out, max_usd=100.0, model=model),
                dry_run=False,
                confirmed=confirmed,
                echo=quiet,
                counter=CharTokenCounter(),
                anthropic_sdk=FakeAnthropicSDK(),
                openai_sdk=FakeOpenAISDK(),
            )
        )

    with pytest.raises(PaidRunRefusedError, match="--yes"):
        start("claude-sonnet-5-5", confirmed=False)
    assert not (out / "run.json").exists()  # the folder is not pinned to a model nobody ran
    assert start("claude-opus-5-5", confirmed=True).outcome.stop_reason is None
    with pytest.raises(ValueError, match="다른 설정"):  # model changes on resume are still refused
        start("claude-sonnet-5-5", confirmed=True)


def test_the_per_run_cap_must_allow_the_next_item(
    folder: Path, tmp_path: Path, runner: asyncio.Runner
) -> None:
    with pytest.raises(PaidRunRefusedError, match="--max-usd-run"):
        runner.run(
            run_real_runbooks(
                _rb_config(folder, tmp_path / "out", max_usd=100.0, max_usd_run=0.0001),
                dry_run=False,
                confirmed=True,
                echo=quiet,
                counter=CharTokenCounter(3.0),
                anthropic_sdk=FakeAnthropicSDK(),
                openai_sdk=FakeOpenAISDK(),
            )
        )


# --------------------------------------------------------------------------- extension (superset)


_AGGREGATES = ("n", "k", "m", "m_broken", "units", "gaps_a", "gaps_b", "findings", "incidents_done")


#: The user's plan: 9 reports in ``reports-b`` now, the folder of all 31 later (any 9 of them).
SUBSET_IDS = ("incident-02", "incident-05", "incident-06", "incident-11", "incident-17")
SUBSET_IDS += ("incident-20", "incident-23", "incident-29", "incident-31")


def test_retro_grows_from_9_to_31_reports_paying_only_for_the_22_new_ones(
    runner: asyncio.Runner, tmp_path: Path
) -> None:
    full = write_reports(tmp_path / "reports", 31, source=FIXTURE_REPORT)
    subset = tmp_path / "reports-b"
    subset.mkdir()
    for incident in SUBSET_IDS:
        shutil.copy(full / f"{incident}.md", subset / f"{incident}.md")  # the same files
    out = tmp_path / "retro"
    clients, main, *_ = fake_clients()
    first = runner.run(run_retro_study(_retro_config(subset, out), clients, echo=quiet))
    assert first.metrics.incidents_done == first.metrics.incidents_total == 9

    clients, main, *_ = fake_clients()
    progress: list[str] = []
    grown = runner.run(run_retro_study(_retro_config(full, out), clients, echo=progress.append))
    new_ids = {f"incident-{i:02d}" for i in range(1, 32)} - set(SUBSET_IDS)
    assert len(new_ids) == 22
    assert _docs(main, "reconstruct") == {f"report-{i}" for i in new_ids}  # only the 22 new ones
    assert len([c for c in main.calls if c.task == "reconstruct"]) == 22  # each once
    assert len(progress) == 22
    assert [r.incident_id for r in grown.results] == [f"incident-{i:02d}" for i in range(1, 32)]

    clients, *_ = fake_clients()
    once = runner.run(run_retro_study(_retro_config(full, tmp_path / "once"), clients, echo=quiet))
    for name in _AGGREGATES:
        assert getattr(grown.metrics, name) == getattr(once.metrics, name), name
    assert grown.metrics.agreement == once.metrics.agreement  # κ, α over all 31 incidents
    assert grown.metrics.flagged_share == once.metrics.flagged_share
    assert grown.metrics.by_family == once.metrics.by_family
    assert (grown.metrics.incidents_total, grown.metrics.incidents_done) == (31, 31)
    assert grown.metrics.n == 31 * first.metrics.n // 9
    assert not grown.metrics.partial
    saved = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert (saved["n"], saved["k"], saved["m"]) == (once.metrics.n, once.metrics.k, once.metrics.m)
    assert saved["agreement"]["cohen_kappa"] == once.metrics.agreement.cohen_kappa
    assert "N = " + str(once.metrics.n) in (out / "report.md").read_text(encoding="utf-8")

    manifest = json.loads((out / MANIFEST_FILE).read_text(encoding="utf-8"))
    dirs = [use["dir"] for use in manifest["inputs"]]
    assert dirs == [subset.resolve().as_posix(), full.resolve().as_posix()]  # both recorded
    assert [use["items"] for use in manifest["inputs"]] == [9, 31]
    assert manifest["items"]["incident-02"]["input_dir"] == subset.resolve().as_posix()
    assert manifest["items"]["incident-01"]["input_dir"] == full.resolve().as_posix()
    lines = (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 31  # nothing duplicated, nothing dropped
    assert not (out / SUPERSEDED_FILE).exists()


def test_retro_refuses_inputs_that_lose_or_change_finished_incidents(
    runner: asyncio.Runner, tmp_path: Path
) -> None:
    subset = write_reports(tmp_path / "reports-b", 2, source=FIXTURE_REPORT)
    full = write_reports(tmp_path / "reports", 4, source=FIXTURE_REPORT)
    out = tmp_path / "retro"
    clients, *_ = fake_clients()
    runner.run(run_retro_study(_retro_config(full, out), clients, echo=quiet))

    # a subset would silently drop paid results: refused, before any request (dry run too)
    with pytest.raises(InputChangeError, match="이번 입력에 없음 2건"):
        runner.run(run_retro_study(_retro_config(subset, out), fake_clients()[0], echo=quiet))
    with pytest.raises(InputChangeError, match="--allow-input-change"):
        runner.run(
            run_retro(_retro_config(subset, out), dry_run=True, echo=quiet, counter=CharTokenCounter())
        )
    with pytest.raises(InputChangeError, match="--limit"):
        runner.run(run_retro_study(_retro_config(full, out, limit=3), fake_clients()[0], echo=quiet))
    assert len((out / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 4

    # a finished incident whose report changed: refused too
    changed = full / "incident-01.md"
    changed.write_text(changed.read_text(encoding="utf-8") + "\nEdited.\n", encoding="utf-8")
    with pytest.raises(InputChangeError, match="내용이 바뀜 1건"):
        runner.run(run_retro_study(_retro_config(full, out), fake_clients()[0], echo=quiet))

    # on purpose: the changed incident is redone, the old result is kept aside, not deleted
    clients, main, *_ = fake_clients()
    redone = runner.run(
        run_retro_study(_retro_config(full, out, allow_input_change=True), clients, echo=quiet)
    )
    assert _docs(main, "reconstruct") == {"report-incident-01"}
    assert len(redone.results) == 4
    superseded = [
        IncidentResult.model_validate_json(line)
        for line in (out / SUPERSEDED_FILE).read_text(encoding="utf-8").splitlines()
    ]
    assert [r.incident_id for r in superseded] == ["incident-01"]
    current = next(r for r in redone.results if r.incident_id == "incident-01")
    assert current.report_sha256 != superseded[0].report_sha256

    # dropping incidents on purpose: they leave the metrics, their results move aside
    shutil.copy(changed, subset / "incident-01.md")
    clients, main, *_ = fake_clients()
    dropped = runner.run(
        run_retro_study(_retro_config(subset, out, allow_input_change=True), clients, echo=quiet)
    )
    assert main.calls == []
    assert [r.incident_id for r in dropped.results] == ["incident-01", "incident-02"]
    assert dropped.metrics.incidents_total == 2
    moved = (out / SUPERSEDED_FILE).read_text(encoding="utf-8").splitlines()
    assert sorted(IncidentResult.model_validate_json(x).incident_id for x in moved) == [
        "incident-01",
        "incident-03",
        "incident-04",
    ]


def test_runbooks_grow_to_a_superset_input_and_keep_a_labelled_review(
    runner: asyncio.Runner, subset: Path, folder: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    clients, *_ = fake_clients()
    first = runner.run(run_runbook_study(_rb_config(subset, out, review_size=5), clients, echo=quiet))
    assert [r.doc_path for r in first.results] == [PAYMENTS]
    # a reviewer has started on the sample
    with (out / "review.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["human_label"] = "correct"
    with (out / "review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    labelled = (out / "review.csv").read_bytes()

    clients, main, *_ = fake_clients()
    grown = runner.run(run_runbook_study(_rb_config(folder, out, review_size=5), clients, echo=quiet))
    assert _extracted(main) == {SEARCH}  # the finished runbook was not redone
    assert [r.doc_path for r in grown.results] == [PAYMENTS, SEARCH]
    assert grown.review_note is not None
    assert (out / "review.csv").read_bytes() == labelled  # the reviewer's work is untouched
    assert (out / "review.next.csv").is_file()

    clients, *_ = fake_clients()
    once = runner.run(
        run_runbook_study(_rb_config(folder, tmp_path / "once", review_size=5), clients, echo=quiet)
    )
    assert grown.metrics.findings == once.metrics.findings
    assert grown.metrics.agreement == once.metrics.agreement
    assert grown.metrics.precision == once.metrics.precision
    assert grown.metrics.runbooks_done == grown.metrics.runbooks_total == 2
    manifest = json.loads((out / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert [use["dir"] for use in manifest["inputs"]] == [
        subset.resolve().as_posix(),
        folder.resolve().as_posix(),
    ]
    assert all(use["commit"] == "a" * 40 for use in manifest["inputs"])

    # the subset again would drop the search runbook's paid result: refused
    with pytest.raises(InputChangeError, match=SEARCH):
        runner.run(run_runbook_study(_rb_config(subset, out), fake_clients()[0], echo=quiet))
    lines = (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert [RunbookResult.model_validate_json(x).doc_path for x in lines] == [PAYMENTS, SEARCH]


def test_runbook_dry_run_prices_only_unfinished_runbooks(
    runner: asyncio.Runner, subset: Path, folder: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    clients, *_ = fake_clients()
    runner.run(run_runbook_study(_rb_config(subset, out), clients, echo=quiet))
    lines: list[str] = []
    dry = runner.run(
        run_real_runbooks(
            _rb_config(folder, out), dry_run=True, echo=lines.append, counter=CharTokenCounter(3.0)
        )
    )
    assert dry.table.items == 1
    assert any("이미 끝난 런북 1건은 제외" in note for note in dry.table.notes)
    assert any("남은 1건 추정" in line for line in lines)
    assert len(_ledger(out)["runs"]) == 1  # a dry run books no run


def _run_metered_retro(runner: asyncio.Runner, config: RetroConfig, cap: float) -> Any:
    clients, main, *_ = metered_clients(max_usd=cap, usd=COST, reserve=RESERVE)
    return runner.run(run_retro_study(config, clients, echo=quiet)), main


def test_retro_extension_under_a_cumulative_cap(runner: asyncio.Runner, tmp_path: Path) -> None:
    """9-then-31 in miniature: 2 incidents, then the folder of 4, under one cumulative cap."""
    subset = write_reports(tmp_path / "reports-b", 2, source=FIXTURE_REPORT)
    full = write_reports(tmp_path / "reports", 4, source=FIXTURE_REPORT)
    probe, _main = _run_metered_retro(runner, _retro_config(full, tmp_path / "probe"), 10.0)
    per_incident = _ledger(tmp_path / "probe")["items"]["incident-01"]["usd"]
    out = tmp_path / "retro"
    cap = 3.5 * per_incident + RESERVE  # the 2 now, then room for 1.5 of the 2 new ones
    first, _main = _run_metered_retro(runner, _retro_config(subset, out), cap)
    assert first.stop_reason is None
    grown, main = _run_metered_retro(runner, _retro_config(full, out), cap)
    assert grown.stop_kind == "budget_total"
    assert "incident-01" not in {d.removeprefix("report-") for d in _docs(main, "reconstruct")}
    assert _ledger(out)["spent_usd"] <= cap
    assert [r.incident_id for r in grown.results] == ["incident-01", "incident-02", "incident-03"]
    again, main = _run_metered_retro(runner, _retro_config(full, out), cap)
    assert again.stop_kind == "budget_total"
    assert len(again.results) == 3
    assert _ledger(out)["spent_usd"] <= cap  # the same cap again adds nothing past it
    final, main = _run_metered_retro(runner, _retro_config(full, out), 10.0)
    assert final.stop_reason is None
    assert _docs(main, "reconstruct") == {"report-incident-04"}
    assert final.metrics.incidents_done == 4
    assert final.metrics.n == probe.metrics.n
    assert final.metrics.agreement == probe.metrics.agreement
