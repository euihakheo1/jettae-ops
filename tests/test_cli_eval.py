"""``jettae eval build|seal|run|report|injection`` on tiny deterministic corpora.

The commands wrap :mod:`jettae.evalkit`; what the CLI adds is the file pipeline (build → seal →
run against the seal), the paid-run guard (estimate first, ``--yes`` required) and exit codes a CI
job can rely on. Everything here runs offline: no test reaches a model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from jettae.cli import main
from jettae.cli.main import CliError, app
from jettae.evalkit import Condition, EvalMetrics, read_cases_jsonl, render_report

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    for name in ("JETTAE_LLM_MODE", "JETTAE_LLM_PROVIDER"):
        monkeypatch.delenv(name, raising=False)


def run(*args: str) -> tuple[int, str, str]:
    result = CliRunner().invoke(app, list(args))
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        raise result.exception
    return result.exit_code, result.stdout, result.stderr


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One organisation's sealed corpus (66 cases), built and sealed through the CLI."""
    cases = tmp_path_factory.mktemp("eval") / "cases.jsonl"
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(ROOT)
        assert run("eval", "build", "--orgs", "1", "--out", str(cases))[0] == 0
        assert run("eval", "seal", "--cases", str(cases))[0] == 0
    return cases


def test_build_writes_the_corpus(corpus: Path) -> None:
    cases = read_cases_jsonl(corpus)
    assert len(cases) == 66  # 1 org × (10 operators + clean) × 3 styles × 2 positions
    assert sum(c.held_out for c in cases) == 12
    code, out, _ = run("eval", "build", "--orgs", "1", "--out", str(corpus.with_name("again.jsonl")))
    assert code == 0
    assert "평가 사례 66건" in out
    assert corpus.with_name("again.jsonl").read_bytes() == corpus.read_bytes()  # deterministic (P6)


def test_build_refuses_impossible_sizes(tmp_path: Path) -> None:
    code, _, err = run("eval", "build", "--orgs", "0", "--out", str(tmp_path / "x.jsonl"))
    assert code == 2
    assert "--orgs 값은 1 이상" in err


def test_seal_and_check(corpus: Path, tmp_path: Path) -> None:
    manifest = json.loads(corpus.with_name("manifest.json").read_bytes())
    assert manifest["n_cases"] == 66
    code, _, err = run("eval", "seal", "--cases", str(corpus), "--check")
    assert code == 0
    assert "봉인 일치" in err
    tampered = tmp_path / "cases.jsonl"
    lines = corpus.read_bytes().decode("utf-8").splitlines()
    lines[3] = lines[3].replace("결제", "결재", 1)
    tampered.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    code, _, err = run(
        "eval", "seal", "--cases", str(tampered), "--out", str(corpus.with_name("manifest.json")), "--check"
    )
    assert code == 1
    assert "봉인 불일치" in err


def test_seal_without_cases_is_a_korean_error(tmp_path: Path) -> None:
    code, _, err = run("eval", "seal", "--cases", str(tmp_path / "none.jsonl"))
    assert code == 2
    assert "평가 사례 파일이 없습니다" in err
    assert "jettae eval build" in err


def test_run_against_the_seal_and_rerender_the_report(corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "results"
    code, stdout, err = run(
        "eval", "run", "--cases", str(corpus), "--conditions", "B,O", "--sample", "0.1",
        "--workers", "1", "--no-benchmarks", "--out", str(out), "--generated-at", "2026-10-13",
    )  # fmt: skip
    assert code == 0, err
    assert "B_rule_graph" in stdout
    assert "확인됨" in stdout  # the corpus was verified against its seal
    metrics = EvalMetrics.model_validate_json(out.joinpath("metrics.json").read_bytes())
    assert set(metrics.conditions) == {"B_rule_graph", "O_oracle"}
    assert metrics.generated_at == "2026-10-13"
    assert metrics.reproducibility is not None
    assert metrics.reproducibility.mismatched == ()
    assert metrics.manifest_root == json.loads(corpus.with_name("manifest.json").read_bytes())["merkle_root"]
    assert not (out / "manifest.json").exists()  # the seal belongs to `eval seal`, never overwritten
    assert out.joinpath("results.jsonl").read_bytes()
    # `eval report` renders the published metrics.json (the single source of the numbers)
    target = tmp_path / "again" / "report.md"
    code, _, err = run("eval", "report", "--metrics", str(out / "metrics.json"), "--out", str(target))
    assert code == 0, err
    assert target.read_bytes().decode("utf-8") == render_report(metrics)
    assert "보고서를 썼습니다" in err


def test_run_refuses_a_corpus_that_does_not_match_its_seal(corpus: Path, tmp_path: Path) -> None:
    tampered = tmp_path / "cases.jsonl"
    tampered.write_bytes(corpus.read_bytes().replace("결제".encode(), "결재".encode(), 1))
    tmp_path.joinpath("manifest.json").write_bytes(corpus.with_name("manifest.json").read_bytes())
    out = tmp_path / "results"
    code, _, err = run("eval", "run", "--cases", str(tampered), "--conditions", "A", "--out", str(out))
    assert code == 1
    assert "봉인 불일치" in err
    assert not out.exists()


def test_run_builds_in_memory_and_writes_its_manifest(tmp_path: Path) -> None:
    out = tmp_path / "smoke"
    code, stdout, err = run(
        "eval", "run", "--smoke", "--conditions", "A", "--sample", "0.05", "--workers", "1",
        "--out", str(out),
    )  # fmt: skip
    assert code == 0, err
    assert "즉석 생성(조직 1곳" in stdout
    manifest = json.loads(out.joinpath("manifest.json").read_bytes())
    metrics = EvalMetrics.model_validate_json(out.joinpath("metrics.json").read_bytes())
    assert metrics.manifest_root == manifest["merkle_root"]
    assert metrics.reproducibility is None  # measured on B only
    assert metrics.impact_benchmark is None  # --smoke skips the benchmarks


def test_paid_conditions_need_yes_after_the_estimate(corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "paid"
    code, _, err = run("eval", "run", "--cases", str(corpus), "--conditions", "B,D", "--out", str(out))
    assert code == 2
    assert "LLM 비용 추정" in err
    assert "--yes" in err
    assert not out.exists()


def test_paid_conditions_need_a_key(corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    code, _, err = run(
        "eval", "run", "--cases", str(corpus), "--conditions", "C", "--yes", "--out", str(tmp_path / "x")
    )
    assert code == 2
    assert "ANTHROPIC_API_KEY" in err


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("--smoke", "--cases", "x.jsonl"), "--smoke는"),
        (("--sample", "0"), "--sample 값은"),
        (("--sample", "1.5"), "--sample 값은"),
        (("--workers", "0"), "--workers 값은"),
        (("--conditions", "B,Z"), "알 수 없는 평가 조건입니다: Z"),
        (("--conditions", " , "), "평가 조건을 하나 이상"),
        (("--manifest", "m.json"), "--manifest는"),
    ],
)
def test_run_option_errors(args: tuple[str, ...], message: str, tmp_path: Path) -> None:
    code, _, err = run("eval", "run", *args, "--out", str(tmp_path / "never"))
    assert code == 2
    assert message in err
    assert not (tmp_path / "never").exists()


def test_condition_names() -> None:
    kit = main._evalkit()
    assert main._conditions(kit, "a, dr ,O") == (Condition.A_KEYWORD, Condition.DR_HYBRID, Condition.O_ORACLE)
    assert main._conditions(kit, "B,b_rule_graph,B") == (Condition.B_RULE_GRAPH,)
    with pytest.raises(CliError, match="가능한 값: A, B, C, D, DR, O"):
        main._conditions(kit, "E")


def test_report_needs_metrics(tmp_path: Path) -> None:
    code, _, err = run("eval", "report", "--metrics", str(tmp_path / "metrics.json"))
    assert code == 2
    assert "metrics.json이 없습니다" in err
    broken = tmp_path / "broken.json"
    broken.write_text("{}", encoding="utf-8")
    code, _, err = run("eval", "report", "--metrics", str(broken))
    assert code == 2
    assert "형식이 올바르지 않습니다" in err


def test_injection_selftest(tmp_path: Path) -> None:
    code, out, err = run("eval", "injection", "--n", "2")
    assert code == 0, err
    assert "무단 변경 0건" in out
    target = tmp_path / "injection.json"
    code, _, _ = run("eval", "injection", "--n", "2", "--format", "json", "--out", str(target))
    assert code == 0
    payload = json.loads(target.read_bytes())
    assert payload["n"] == 2
    assert payload["unauthorised_changes"] == 0
    code, _, err = run("eval", "injection", "--bundle", str(tmp_path / "missing"))
    assert code == 2
    assert "묶음 폴더를 찾을 수 없습니다" in err


def test_partial_runs_never_default_to_the_published_results(tmp_path: Path) -> None:
    assert main._results_dir(None, partial=False) == Path("docs/eval")
    assert main._results_dir(None, partial=True) == Path("docs/eval/sample")
    assert main._results_dir(tmp_path, partial=True) == tmp_path
    afile = tmp_path / "metrics.json"
    afile.write_text("{}", encoding="utf-8")
    with pytest.raises(CliError, match="결과 폴더"):
        main._results_dir(afile, partial=False)
