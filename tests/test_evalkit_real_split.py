"""Dev/test split: a fixed, published assignment; fixes look at dev, final numbers come from test."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from jettae.cli.main import app
from jettae.evalkit.realdata.runbooks import (
    RunbookStudyConfig,
    load_study_sources,
    run_runbook_study,
)
from jettae.evalkit.realdata.split import (
    DEFAULT_SPLIT_FILE,
    SplitAssignment,
    SplitError,
    assign_split,
    check_assignment,
    load_split,
    select_split,
)
from support_realdata import fake_clients, write_runbook_folder

ROOT = Path(__file__).resolve().parents[1]
COMMITTED = ROOT / "docs" / "eval" / "gitlab-runbooks-split.json"
#: the pilot run's runbooks and the investigation's — both looked at before the split existed
PILOT = {
    "docs/alerts/ApdexSLOViolation.md",
    "docs/alerts/ErrorSLOViolation.md",
    "docs/alerts/TrafficAbsent.md",
}
INVESTIGATION = {
    "docs/alerts/ApdexSLOViolation.md",
    "docs/alerts/ErrorSLOViolation.md",
    "docs/incidents/README.md",
    "docs/cloudflare/oncall.md",
    "docs/uncategorized/rails-is-down.md",
}
PATHS = [f"docs/{name}.md" for name in "abcdefghijklmnopqrst"]


def test_the_rule_forces_seen_documents_into_dev_and_fills_it_by_seed() -> None:
    forced = {"docs/c.md": "seen", "docs/q.md": "seen"}
    split = assign_split(PATHS, forced=forced, dev_size=5, seed=7, source="s")
    assert len(split.dev) == 5
    assert len(split.test) == 15
    assert set(forced) <= set(split.dev)
    assert not set(split.dev) & set(split.test)
    assert sorted((*split.dev, *split.test)) == sorted(PATHS)
    assert split == assign_split(list(reversed(PATHS)), forced=forced, dev_size=5, seed=7, source="s")
    assert split.dev != assign_split(PATHS, forced=forced, dev_size=5, seed=8, source="s").dev
    check_assignment(split)


def test_an_assignment_that_is_not_the_rules_output_is_refused() -> None:
    split = assign_split(PATHS, forced={"docs/c.md": "seen"}, dev_size=4, seed=7, source="s")
    moved = split.test[0]
    edited = split.model_copy(
        update={
            "dev": tuple(sorted((*split.dev[1:], moved))),
            "test": tuple(sorted((*split.test[1:], split.dev[0]))),
        }
    )
    with pytest.raises(SplitError, match="규칙"):
        check_assignment(edited)
    with pytest.raises(SplitError, match="문서 목록에 없습니다"):
        assign_split(PATHS, forced={"docs/zz.md": "seen"}, dev_size=4, seed=7, source="s")
    with pytest.raises(SplitError, match="dev 크기"):
        assign_split(PATHS, forced=dict.fromkeys(PATHS[:3], "seen"), dev_size=2, seed=7, source="s")


def test_the_committed_gitlab_assignment() -> None:
    assert DEFAULT_SPLIT_FILE == COMMITTED
    split = load_split(COMMITTED)  # checked against the rule on load
    assert (len(split.dev), len(split.test)) == (10, 20)
    assert set(split.forced) == PILOT | INVESTIGATION
    assert set(split.dev) >= PILOT | INVESTIGATION
    assert split.dev_size == 10
    assert split.source.endswith("@844a2b882687492cf117cc2fc553f7fbf9a02197")
    assert "test에서만" in split.rule


def test_select_split_keeps_input_order_and_refuses_unassigned_documents() -> None:
    split = assign_split(PATHS, forced={}, dev_size=5, seed=1, source="s")
    items = list(reversed(PATHS))
    dev = select_split(items, paths=items, assignment=split, split="dev")
    assert set(dev) == set(split.dev)
    assert dev == [p for p in items if p in split.dev]
    assert select_split(items, paths=items, assignment=None, split="all") == items
    with pytest.raises(SplitError, match="배정되지 않은"):
        select_split([*items, "docs/new.md"], paths=[*items, "docs/new.md"], assignment=split, split="test")
    with pytest.raises(SplitError, match="분할 파일이 필요"):
        select_split(items, paths=items, assignment=None, split="dev")


def _folder_split(folder: Path, tmp_path: Path) -> Path:
    """A split file for the two-runbook test folder: payments in dev, search in test."""
    split = assign_split(
        ["docs/payments/api.md", "docs/search/service.md"],
        forced={"docs/payments/api.md": "seen"},
        dev_size=1,
        seed=3,
        source="https://example.com/runbooks@" + "a" * 40,
    )
    path = tmp_path / "split.json"
    path.write_text(json.dumps(split.model_dump(mode="json"), ensure_ascii=False), encoding="utf-8")
    return path


def _config(folder: Path, out: Path, split_file: Path, **overrides: Any) -> RunbookStudyConfig:
    return RunbookStudyConfig(
        input_dir=folder, out_dir=out, generated_at="2026-10-17", split_file=split_file, **overrides
    )


def test_a_split_run_measures_only_its_side_and_records_the_design(tmp_path: Path) -> None:
    folder = write_runbook_folder(tmp_path / "input")
    split_file = _folder_split(folder, tmp_path)
    out = tmp_path / "test-out"
    config = _config(folder, out, split_file, split="test")
    assert [s.path for s in load_study_sources(config)] == ["docs/search/service.md"]
    clients, *_ = fake_clients()
    outcome = asyncio.run(run_runbook_study(config, clients, echo=lambda _: None))
    assert [r.doc_path for r in outcome.results] == ["docs/search/service.md"]
    digest = SplitAssignment.model_validate_json(split_file.read_text(encoding="utf-8")).digest()
    settings = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert settings["split"] == "test"
    assert settings["split_digest"] == digest
    assert settings["jurisdictions"] == ["US"]
    assert settings["rubric"] == "premise-claim-v4"
    assert settings["pipeline"] == "pipeline-v2"
    assert settings["record"] == "record-v2"
    assert len(settings["extraction_prompt"]) == 16
    design = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["design"]
    assert design["split"]["name"] == "test"
    assert design["split"]["test"] == ["docs/search/service.md"]
    assert design["jurisdictions"] == ["US"]
    assert design["korean_law_rules"] is False
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert (metrics["split"], metrics["split_digest"]) == ("test", digest)
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "평가 분할: **test**" in report
    assert "최종 보고 수치가 아니다" not in report
    # the same folder cannot be continued with the other side of the split
    with pytest.raises(ValueError, match="다른 설정"):
        asyncio.run(
            run_runbook_study(_config(folder, out, split_file, split="dev"), clients, echo=lambda _: None)
        )


def test_a_dev_report_says_it_is_not_the_final_number(tmp_path: Path) -> None:
    folder = write_runbook_folder(tmp_path / "input")
    split_file = _folder_split(folder, tmp_path)
    out = tmp_path / "dev-out"
    clients, *_ = fake_clients()
    asyncio.run(
        run_runbook_study(_config(folder, out, split_file, split="dev"), clients, echo=lambda _: None)
    )
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "평가 분할: **dev**" in report
    assert "최종 보고 수치가 아니다" in report


def test_limit_applies_within_the_split(tmp_path: Path) -> None:
    folder = write_runbook_folder(tmp_path / "input")
    split_file = _folder_split(folder, tmp_path)
    config = _config(folder, tmp_path / "out", split_file, split="test", limit=1)
    assert [s.path for s in load_study_sources(config)] == ["docs/search/service.md"]


def test_cli_dry_run_with_a_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("jettae.evalkit.realdata.entry.load_api_key", lambda *_, **__: None)
    folder = write_runbook_folder(tmp_path / "input")
    split_file = _folder_split(folder, tmp_path)
    out = tmp_path / "dry"
    args = ["eval", "real-runbooks", "--input", str(folder), "--out", str(out), "--dry-run"]
    result = CliRunner().invoke(app, [*args, "--split", "dev", "--split-file", str(split_file)])
    assert result.exit_code == 0, result.stderr
    assert json.loads((out / "dry_run.json").read_text(encoding="utf-8"))["items"] == 1
    bad = CliRunner().invoke(app, [*args, "--split", "holdout"])
    assert bad.exit_code != 0
    assert "--split" in bad.stderr
    bad = CliRunner().invoke(app, [*args, "--jurisdiction", "U.S."])
    assert bad.exit_code != 0
    assert "--jurisdiction" in bad.stderr
