"""``jettae scan`` and ``jettae rehearse`` end to end on the 별빛상점 example (LLM off).

The CLI must show exactly what :class:`~jettae.orchestrator.JettaeService` concluded — the same
score, findings and rehearsals the REST API would serve — in every format, with evidence located
as ``file:line`` from the working directory, and its exit codes are a CI contract: 0 = done,
1 = gate failed, 2 = could not run (with a Korean message on stderr).
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from jettae.cli.main import app
from jettae.config import Settings
from jettae.domain import DiagnosisReport, FindingStatus, LlmMode, ScenarioOrigin, Severity
from jettae.ingest import load_bundle_dir
from jettae.orchestrator import JettaeService
from jettae.store import MemoryStore
from test_orchestrator_support import event_loop_runner

ROOT = Path(__file__).resolve().parents[1]
DEMO = Path("examples/dmc-shop")
PAYMENT_RUNBOOK = DEMO / "runbooks" / "01-결제-주문-장애대응.md"
ANSI = re.compile(r"\x1b\[")


@pytest.fixture(autouse=True)
def _repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every command from the repository root with the LLM off and no developer settings."""
    monkeypatch.chdir(ROOT)
    for name in ("JETTAE_LLM_MODE", "JETTAE_DATABASE_URL", "JETTAE_STORE_RAW_DOCUMENTS", "COLUMNS"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="module")
def expected() -> Iterator[DiagnosisReport]:
    """The same diagnosis through the service directly: what the CLI must reproduce."""
    service = JettaeService(MemoryStore(), Settings(_env_file=None, llm_mode=LlmMode.OFF))
    workspace = service.create_workspace("dmc-shop")
    service.load_bundle(
        workspace.id, load_bundle_dir(ROOT / DEMO), scenarios=(ROOT / DEMO / "scenarios.yaml").read_bytes()
    )
    with event_loop_runner() as runner:
        yield runner.run(service.diagnose(workspace.id, llm_mode=LlmMode.OFF, actor="test"))


def run(*args: str) -> tuple[int, str, str]:
    result = CliRunner().invoke(app, list(args))
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        raise result.exception
    return result.exit_code, result.stdout, result.stderr


def scan_json(*args: str) -> dict[str, object]:
    code, out, _ = run("scan", *args, "--format", "json")
    assert code == 0
    payload: dict[str, object] = json.loads(out)
    return payload


# --------------------------------------------------------------------------- formats


def test_text_summary_shows_score_services_findings_with_locations(expected: DiagnosisReport) -> None:
    code, out, err = run("scan", DEMO.as_posix())
    assert code == 0, err
    assert f"{expected.readiness.overall:.1f} / 100" in out
    assert "판정 경로 LLM 0회" in out
    assert "호출 0회" in out  # LLM calls/tokens disclosure
    assert "통과율" in out
    for service in expected.readiness.services:
        assert service.service_name in out
    # evidence located from the working directory, as a path the user can open
    assert re.search(r"위치 examples/dmc-shop/runbooks/0\d-[^:\s]+\.md:\d+", out)
    broken = [f for f in expected.findings if f.status is FindingStatus.BROKEN]
    assert f"끊긴 경로    {len(broken)}건" in out
    assert not ANSI.search(out), "a CLI run without a terminal must print plain text"


def test_json_is_the_service_report_plus_documents(expected: DiagnosisReport) -> None:
    payload = scan_json(DEMO.as_posix())
    report = DiagnosisReport.model_validate(payload["report"])
    assert report.readiness.overall == expected.readiness.overall
    assert report.graph_hash == expected.graph_hash
    assert [f.id for f in report.findings] == [f.id for f in expected.findings]
    assert report.llm.judgement_path_calls == 0
    documents = payload["documents"]
    assert isinstance(documents, dict)
    assert sorted(d["path"] for d in documents.values()) == sorted(
        f"examples/dmc-shop/runbooks/{p.name}" for p in (ROOT / DEMO / "runbooks").iterdir()
    )
    assert payload["gate"] == {"passed": True, "failures": []}
    summary = payload["summary"]
    assert isinstance(summary, dict)
    assert summary["critical"] == sum(1 for f in expected.findings if f.severity is Severity.CRITICAL)
    assert summary["judgement_path_llm_calls"] == 0


def test_sarif_has_runs_and_repository_relative_locations(expected: DiagnosisReport) -> None:
    code, out, _ = run("scan", DEMO.as_posix(), "--format", "sarif")
    assert code == 0
    sarif = json.loads(out)
    assert sarif["version"] == "2.1.0"
    (sarif_run,) = sarif["runs"]
    assert len(sarif_run["results"]) == len({f.id for f in expected.findings})
    uris = {
        location["physicalLocation"]["artifactLocation"]["uri"]
        for result in sarif_run["results"]
        for location in result["locations"]
    }
    assert uris
    assert all(uri.startswith("examples/dmc-shop/runbooks/") for uri in uris), uris
    assert sarif_run["properties"]["judgementPathLlmCalls"] == 0


def test_markdown_and_html_evidence_reports() -> None:
    code, md, _ = run("scan", DEMO.as_posix(), "--format", "md", "--no-rehearse")
    assert code == 0
    assert "준비도" in md
    assert md.startswith("# ")
    code, html, _ = run("scan", DEMO.as_posix(), "--format", "html", "--no-rehearse")
    assert code == 0
    assert html.lstrip().lower().startswith("<!doctype html")
    assert "준비도" in html


def test_out_writes_utf8_file_and_prints_summary(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "report.json"
    code, out, err = run("scan", DEMO.as_posix(), "--format", "json", "--out", str(target), "--no-rehearse")
    assert code == 0
    payload = json.loads(target.read_bytes().decode("utf-8"))
    assert payload["tool"]["name"] == "jettae"
    assert "전체 준비도" in out  # people still get the summary on stdout
    assert "저장했습니다" in err


def test_text_out_file_is_plain(tmp_path: Path) -> None:
    target = tmp_path / "summary.txt"
    code, _, _ = run("scan", DEMO.as_posix(), "--out", str(target), "--no-draft")
    assert code == 0
    text = target.read_bytes().decode("utf-8")
    assert "판정 경로 LLM 0회" in text
    assert not ANSI.search(text)
    assert all(line == line.rstrip() for line in text.splitlines())


def test_summary_md_appends_on_a_fresh_line(tmp_path: Path) -> None:
    summary = tmp_path / "step-summary.md"
    summary.write_bytes("# 이전 단계".encode())  # no trailing newline
    code, _, _ = run(
        "scan",
        DEMO.as_posix(),
        "--format",
        "sarif",
        "--out",
        str(tmp_path / "x.sarif"),
        "--summary-md",
        str(summary),
        "--no-rehearse",
    )
    assert code == 0
    text = summary.read_bytes().decode("utf-8")
    assert text.startswith("# 이전 단계\n# ")
    assert "준비도" in text


def test_no_rehearse_and_no_draft_are_reported() -> None:
    code, out, _ = run("scan", DEMO.as_posix(), "--no-rehearse", "--no-draft")
    assert code == 0
    assert "생략(--no-rehearse)" in out
    payload = scan_json(DEMO.as_posix(), "--no-rehearse", "--no-draft")
    report = DiagnosisReport.model_validate(payload["report"])
    assert report.rehearsals == ()
    assert report.patches == ()


def test_bundled_drills_are_rehearsed() -> None:
    report = DiagnosisReport.model_validate(scan_json(DEMO.as_posix(), "--no-draft")["report"])
    assert any(s.origin is ScenarioOrigin.HUMAN for s in report.scenarios)


def test_scan_is_reproducible() -> None:
    first = DiagnosisReport.model_validate(scan_json(DEMO.as_posix(), "--no-draft")["report"])
    second = DiagnosisReport.model_validate(scan_json(DEMO.as_posix(), "--no-draft")["report"])
    assert first.graph_hash == second.graph_hash
    assert first.findings == second.findings
    assert [r.result_hash() for r in first.rehearsals] == [r.result_hash() for r in second.rehearsals]


# --------------------------------------------------------------------------- gate


def test_fail_under_sets_exit_code_but_keeps_the_files(tmp_path: Path, expected: DiagnosisReport) -> None:
    sarif = tmp_path / "out.sarif"
    below = f"{expected.readiness.overall + 0.1:.1f}"
    code, _, err = run(
        "scan",
        DEMO.as_posix(),
        "--fail-under",
        below,
        "--format",
        "sarif",
        "--out",
        str(sarif),
        "--no-rehearse",
        "--no-draft",
    )
    assert code == 1
    assert "기준 미달" in err
    assert "--fail-under" in err
    assert json.loads(sarif.read_bytes())["runs"]  # written before the gate failed
    at = f"{expected.readiness.overall:.1f}"
    assert run("scan", DEMO.as_posix(), "--fail-under", at, "--no-rehearse", "--no-draft")[0] == 0
    assert run("scan", DEMO.as_posix(), "--fail-under", "0", "--no-rehearse", "--no-draft")[0] == 0


def test_fail_on_critical(expected: DiagnosisReport) -> None:
    assert any(f.severity is Severity.CRITICAL for f in expected.findings)
    code, _, err = run("scan", DEMO.as_posix(), "--fail-on-critical", "--no-rehearse", "--no-draft")
    assert code == 1
    assert "심각(critical)" in err
    payload = json.loads(
        run("scan", DEMO.as_posix(), "--fail-on-critical", "--format", "json", "--no-rehearse", "--no-draft")[
            1
        ]
    )
    assert payload["gate"]["passed"] is False


@pytest.mark.parametrize("value", ["150", "-1", "nan"])
def test_fail_under_must_be_a_score(value: str) -> None:
    code, _, err = run("scan", DEMO.as_posix(), "--fail-under", value)
    assert code == 2
    assert "0~100" in err


# --------------------------------------------------------------------------- targets and errors


def test_single_runbook_file(tmp_path: Path) -> None:
    payload = scan_json(PAYMENT_RUNBOOK.as_posix(), "--no-draft")
    report = DiagnosisReport.model_validate(payload["report"])
    assert report.findings
    documents = payload["documents"]
    assert isinstance(documents, dict)
    assert [d["path"] for d in documents.values()] == [PAYMENT_RUNBOOK.as_posix()]
    # the file's location is display only: a copy elsewhere yields the very same graph (P6)
    copy = tmp_path / "elsewhere" / PAYMENT_RUNBOOK.name
    copy.parent.mkdir()
    shutil.copyfile(ROOT / PAYMENT_RUNBOOK, copy)
    elsewhere = DiagnosisReport.model_validate(scan_json(str(copy), "--no-draft")["report"])
    assert elsewhere.graph_hash == report.graph_hash


def test_missing_path_is_a_korean_error() -> None:
    code, out, err = run("scan", "no/such/bundle")
    assert code == 2
    assert out == ""
    assert "경로를 찾을 수 없습니다: no/such/bundle" in err


def test_unsupported_file_and_empty_bundle(tmp_path: Path) -> None:
    sheet = tmp_path / "runbook.xlsx"
    sheet.write_bytes(b"PK")
    code, _, err = run("scan", str(sheet))
    assert code == 2
    assert "지원하지 않는 런북 형식" in err
    empty = tmp_path / "bundle"
    empty.mkdir()
    code, _, err = run("scan", str(empty))
    assert code == 2
    assert "런북 문서를 찾을 수 없습니다" in err


@pytest.mark.parametrize(("option", "value"), [("--format", "pdf"), ("--llm", "always")])
def test_invalid_choices_are_korean_errors(option: str, value: str) -> None:
    code, _, err = run("scan", DEMO.as_posix(), option, value)
    assert code == 2
    assert f"{option} 값은" in err


def test_bundle_with_two_drill_files_is_refused(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    shutil.copytree(ROOT / DEMO, bundle)
    shutil.copyfile(bundle / "scenarios.yaml", bundle / "scenarios.yml")
    code, _, err = run("scan", str(bundle))
    assert code == 2
    assert "시나리오 파일이 여러 개" in err


# --------------------------------------------------------------------------- rehearse


def test_rehearse_text_lists_every_drill(expected: DiagnosisReport) -> None:
    code, out, _ = run("rehearse", DEMO.as_posix())
    assert code == 0
    assert "통과율" in out
    assert "판정 경로 LLM 0회" in out
    human = [s.title for s in expected.scenarios if s.origin is ScenarioOrigin.HUMAN]
    assert human
    for title in human:
        assert title in out


def test_rehearse_json_carries_result_hashes(expected: DiagnosisReport) -> None:
    code, out, _ = run("rehearse", DEMO.as_posix(), "--format", "json")
    assert code == 0
    payload = json.loads(out)
    results = payload["results"]
    assert payload["total"] == len(results) == len({r.scenario_id for r in expected.rehearsals})
    assert payload["passed"] == sum(1 for r in results if r["result"]["outcome"] == "pass")
    assert all(re.fullmatch(r"[0-9a-f]{64}", r["result_hash"]) for r in results)
    outcomes = [r["result"]["outcome"] for r in results]
    assert outcomes == sorted(outcomes)  # failures first ("fail" < "pass")
    assert payload["judgement_path_llm_calls"] == 0


def test_rehearse_with_own_drill_file(tmp_path: Path) -> None:
    drills = tmp_path / "drills.yaml"
    drills.write_text(
        "- id: cli-drill\n  title: CLI 시험 — 슬랙 장애 중 주문 실패\n  service: 주문\n"
        "  perturbations:\n    - kind: channel_down\n      target: slack\n",
        encoding="utf-8",
    )
    code, out, _ = run("rehearse", DEMO.as_posix(), "--scenarios", str(drills), "--format", "json")
    assert code == 0
    titles = [r["scenario"]["title"] for r in json.loads(out)["results"]]
    assert "CLI 시험 — 슬랙 장애 중 주문 실패" in titles
    assert "추석 연휴 새벽 결제 장애 (flagd paymentFailure=1.0)" not in titles  # replaced, not merged


def test_rehearse_missing_drill_file_is_a_korean_error(tmp_path: Path) -> None:
    code, _, err = run("rehearse", DEMO.as_posix(), "--scenarios", str(tmp_path / "none.yaml"))
    assert code == 2
    assert "시나리오 파일을 찾을 수 없습니다" in err


def test_rehearse_out_file(tmp_path: Path) -> None:
    target = tmp_path / "rehearsal.txt"
    code, _, err = run("rehearse", PAYMENT_RUNBOOK.as_posix(), "--out", str(target))
    assert code == 0
    assert "시나리오별 결과" in target.read_text(encoding="utf-8")
    assert "저장했습니다" in err
