"""``jettae eval real-runbooks`` / ``jettae eval retro``: dry run and the paid-run guard, offline.

No test here can reach a model: the Anthropic key lookup is patched to "no key" (so the dry run
counts with the character heuristic instead of ``count_tokens``) and the client factories raise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from jettae.cli.main import app
from support_realdata import metered_clients, write_reports, write_runbook_folder

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "examples" / "retro-reports"


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    def no_key(name: str, **_: Any) -> None:
        return None

    def forbidden(*_: Any, **__: Any) -> Any:
        raise AssertionError("a paid client was built")

    monkeypatch.setattr("jettae.evalkit.realdata.entry.load_api_key", no_key)
    monkeypatch.setattr("jettae.evalkit.realdata.entry.new_anthropic_sdk", forbidden)
    monkeypatch.setattr("jettae.evalkit.realdata.entry.build_study_clients", forbidden)


def run(*args: str) -> tuple[int, str, str]:
    result = CliRunner().invoke(app, list(args))
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        raise result.exception
    return result.exit_code, result.stdout, result.stderr


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    return write_runbook_folder(tmp_path / "input")


def test_real_runbooks_dry_run_prints_the_cost_table(folder: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    code, stdout, stderr = run(
        "eval", "real-runbooks", "--input", str(folder), "--out", str(out), "--dry-run"
    )
    assert code == 0, stderr
    assert "실증 ① 공개 런북 비용 추정" in stdout
    for model in ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "gpt-6-luna", "gpt-6.1-sol"):
        assert model in stdout
    assert "← 선택" in stdout
    assert "유료 호출 0회" in stderr
    assert "count_tokens 미사용" in stderr  # no key: the heuristic is announced
    assert (out / "dry_run.md").is_file()
    assert (out / "dry_run.json").is_file()
    assert not (out / "results.jsonl").exists()


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ((), "--max-usd"),
        (("--max-usd", "0.0001"), "넘어 시작하지 않습니다"),
        (("--max-usd", "1000"), "--yes"),
        (("--max-usd", "-1"), "0보다 큰"),
        (("--max-usd", "5", "--effort", "turbo"), "--effort"),
        (("--max-usd", "5", "--coder-b", "anthropic:claude-sonnet-5-5"), "계열"),
        (("--max-usd", "5", "--model", "gpt-6-luna"), "Claude"),
        (("--concurrency", "0"), "--concurrency"),
    ],
)
def test_real_runbooks_paid_run_guard(
    folder: Path, tmp_path: Path, extra: tuple[str, ...], message: str
) -> None:
    code, _stdout, stderr = run(
        "eval", "real-runbooks", "--input", str(folder), "--out", str(tmp_path / "o"), *extra
    )
    assert code == 2
    assert message in stderr


def test_real_runbooks_missing_input(tmp_path: Path) -> None:
    code, _out, stderr = run(
        "eval", "real-runbooks", "--input", str(tmp_path / "nope"), "--out", str(tmp_path / "o")
    )
    assert code == 2
    assert "입력 폴더를 찾을 수 없습니다" in stderr


def test_retro_dry_run_on_the_fixture(tmp_path: Path) -> None:
    out = tmp_path / "retro"
    code, stdout, stderr = run("eval", "retro", "--reports", str(REPORTS), "--out", str(out), "--dry-run")
    assert code == 0, stderr
    assert "실증 ② 사후 보고서 회고 비용 추정" in stdout
    assert "사고 전 묶음 재구성" in stdout
    assert "공백 코딩" in stdout
    assert (out / "dry_run.md").is_file()


def test_retro_guard_and_options(tmp_path: Path) -> None:
    code, _out, stderr = run("eval", "retro", "--reports", str(REPORTS), "--out", str(tmp_path / "r"))
    assert code == 2
    assert "--max-usd" in stderr
    code, _out, stderr = run(
        "eval", "retro", "--reports", str(REPORTS), "--out", str(tmp_path / "r"), "--bundle-language", "fr"
    )
    assert code == 2
    assert "--bundle-language" in stderr


def test_help_lists_the_new_commands() -> None:
    code, stdout, _err = run("eval", "--help")
    assert code == 0
    assert "real-runbooks" in stdout
    assert "retro" in stdout


# --------------------------------------------------------------------------- cumulative cap & growth


CHEAP = (
    "--model",
    "claude-haiku-5-5",
    "--coder-a",
    "anthropic:claude-haiku-5-5",
    "--coder-b",
    "openai:gpt-6-luna",
    "--assume-findings",
    "0",
    "--concurrency",
    "1",
    "--yes",
)


@pytest.fixture
def metered(monkeypatch: pytest.MonkeyPatch) -> None:
    """Paid runs go to metered fakes: each call reserves $0.01 and costs $0.004 (no network)."""

    def build(*_: Any, tracker: Any, **__: Any) -> Any:
        return metered_clients(tracker=tracker, usd=0.004, reserve=0.01)[0]

    monkeypatch.setattr("jettae.evalkit.realdata.entry.build_study_clients", build)


def test_help_explains_the_cumulative_cap_and_the_new_options() -> None:
    for command in ("real-runbooks", "retro"):
        code, stdout, _err = run("eval", command, "--help")
        assert code == 0
        flat = " ".join(stdout.split())
        for option in ("--max-usd-run", "--allow-input-change"):
            assert option in flat
        assert "누적" in flat


def test_per_run_cap_is_validated(folder: Path, tmp_path: Path) -> None:
    code, _out, stderr = run(
        "eval",
        "real-runbooks",
        "--input",
        str(folder),
        "--out",
        str(tmp_path / "o"),
        "--max-usd",
        "5",
        "--max-usd-run",
        "0",
    )
    assert code == 2
    assert "--max-usd-run" in stderr


@pytest.mark.usefixtures("metered")
def test_budget_stop_messages_tell_cumulative_from_per_run(folder: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    base = ("eval", "real-runbooks", "--input", str(folder), "--out", str(out), *CHEAP)
    code, _stdout, stderr = run(*base, "--max-usd", "0.02")
    assert code == 1
    flat = " ".join(stderr.split())
    assert "누적 예산 상한 --max-usd $0.02" in flat
    assert "누적 상한 --max-usd(이 --out 폴더의 모든 실행 합계)에 닿았습니다" in flat
    assert "--max-usd를 누적 지출 $0.01보다 크게 올려" in flat
    assert "이 --out 폴더 누적 지출 $0.0120 / 누적 상한 --max-usd $0.02" in flat

    # the same command again: what was spent counts, the run cannot go on
    code, _stdout, stderr = run(*base, "--max-usd", "0.02")
    assert code == 1
    flat = " ".join(stderr.split())
    assert "지출(이 --out 폴더 누적, spend.json): $0.0120" in flat
    assert "이번 실행 지출 $0.0000 · 이 --out 폴더 누적 지출 $0.0120" in flat

    # a raised cumulative cap finishes the rest, the finished runbook is skipped
    code, _stdout, stderr = run(*base, "--max-usd", "1")
    assert code == 0, stderr
    flat = " ".join(stderr.split())
    assert "이미 끝난 런북 1건은 results.jsonl에 있어 다시 하지 않습니다" in flat
    assert "이 --out 폴더 누적 지출 $0.0320" in flat

    # a per-run cap says so, and that re-running continues
    other = ("eval", "real-runbooks", "--input", str(folder), "--out", str(tmp_path / "other"), *CHEAP)
    code, _stdout, stderr = run(*other, "--max-usd", "1", "--max-usd-run", "0.015")
    assert code == 1
    flat = " ".join(stderr.split())
    assert "이번 실행 상한 --max-usd-run에 닿았습니다" in flat
    assert "새로 --max-usd-run만큼(누적 상한 --max-usd 안에서) 이어서 합니다" in flat


@pytest.mark.usefixtures("metered")
def test_retro_cli_grows_to_a_superset_and_refuses_a_subset(tmp_path: Path) -> None:
    source = REPORTS / "fictional-parcel-db-outage.md"
    subset = write_reports(tmp_path / "reports-b", 1, source=source)
    full = write_reports(tmp_path / "reports", 2, source=source)
    out = tmp_path / "retro"
    cheap = ("--model", "claude-haiku-5-5", "--coder-a", "anthropic:claude-haiku-5-5")
    cheap += ("--coder-b", "openai:gpt-6-luna", "--yes", "--max-usd", "5")
    code, _stdout, stderr = run("eval", "retro", "--reports", str(subset), "--out", str(out), *cheap)
    assert code == 0, stderr
    code, _stdout, stderr = run("eval", "retro", "--reports", str(full), "--out", str(out), *cheap)
    assert code == 0, stderr
    flat = " ".join(stderr.split())
    assert "이미 끝난 사고 보고서 1건" in flat
    assert "입력 폴더가 이전 실행과 다릅니다" in flat
    assert "[2/2] incident-02" in flat
    assert "[1/2]" not in flat  # the finished incident was skipped, not redone

    code, _stdout, stderr = run(
        "eval", "retro", "--reports", str(subset), "--out", str(out), *cheap, "--dry-run"
    )
    assert code == 2
    flat = " ".join(stderr.split())
    assert "이번 입력에 없음 1건(incident-02)" in flat
    assert "--allow-input-change" in flat
    code, _stdout, stderr = run(
        "eval",
        "retro",
        "--reports",
        str(subset),
        "--out",
        str(out),
        *cheap,
        "--dry-run",
        "--allow-input-change",
    )
    assert code == 0, stderr
    # another model in the same folder is still refused (settings are pinned)
    code, _stdout, stderr = run("eval", "retro", "--reports", str(full), "--out", str(out), "--dry-run")
    assert code == 2
    assert "다른 설정" in stderr
