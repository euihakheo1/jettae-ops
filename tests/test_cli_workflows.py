"""The GitHub Action (``action.yml``) and the workflows that ship with the CLI.

These files run on machines nobody watches, so their contract is pinned here: the inputs the
README promises, the LLM key reaching only the scan step, no input pasted into a shell script
(script injection), every flag the Action passes actually accepted by ``jettae scan``, and the CI
quality gates of docs/ARCHITECTURE.md §7.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from jettae.cli.main import app

ROOT = Path(__file__).resolve().parents[1]
ACTION = ROOT / "action.yml"
CI = ROOT / ".github" / "workflows" / "ci.yml"
EXAMPLE = ROOT / ".github" / "workflows" / "jettae-scan-example.yml"
EXPRESSION = re.compile(r"\$\{\{\s*(.*?)\s*\}\}")


def load(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data


def triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    on: dict[str, Any] = workflow.get("on", workflow.get(True))  # YAML 1.1 reads a bare `on` as true
    return on


@pytest.fixture(scope="module")
def action() -> dict[str, Any]:
    return load(ACTION)


def steps(action: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = action["runs"]["steps"]
    return result


def step(action: dict[str, Any], step_id: str) -> dict[str, Any]:
    return next(s for s in steps(action) if s.get("id") == step_id)


# --------------------------------------------------------------------------- action.yml


def test_action_metadata_and_inputs(action: dict[str, Any]) -> None:
    assert action["name"] == "JETTAE Ops 런북 건강검진"
    assert action["runs"]["using"] == "composite"
    inputs = action["inputs"]
    assert {"path", "llm", "fail-under", "anthropic-api-key", "sarif-file"} <= set(inputs)
    assert inputs["llm"]["default"] == "off"
    assert inputs["fail-under"]["default"] == "0"
    assert inputs["anthropic-api-key"]["required"] is False
    assert "러너 밖으로 나가지 않습니다" in action["description"]
    assert {"score", "sarif-file", "passed"} <= set(action["outputs"])


def test_action_installs_itself_on_python_312(action: dict[str, Any]) -> None:
    setup = next(s for s in steps(action) if str(s.get("uses", "")).startswith("actions/setup-python@"))
    assert setup["uses"] == "actions/setup-python@v5"
    assert setup["with"]["python-version"] == "3.12"
    assert (
        'pip install --disable-pip-version-check --quiet "${{ github.action_path }}[anthropic]"'
        in step(action, "install")["run"]
    )


def test_llm_key_reaches_only_the_scan_step(action: dict[str, Any]) -> None:
    holders = [s.get("id", s.get("name")) for s in steps(action) if "anthropic-api-key" in yaml.safe_dump(s)]
    assert holders == ["scan"]
    assert step(action, "scan")["env"]["ANTHROPIC_API_KEY"] == "${{ inputs.anthropic-api-key }}"


def test_no_input_is_pasted_into_a_script(action: dict[str, Any]) -> None:
    for each in steps(action):
        for expression in EXPRESSION.findall(each.get("run", "")):
            assert not expression.startswith(("inputs.", "steps.")), (each.get("name"), expression)


def test_scan_step_uses_sarif_and_job_summary(action: dict[str, Any]) -> None:
    script = step(action, "scan")["run"]
    assert "--format sarif" in script
    assert '--out "$INPUT_SARIF"' in script
    assert '--summary-md "$GITHUB_STEP_SUMMARY"' in script
    assert '--llm "$INPUT_LLM"' in script
    upload = next(
        s for s in steps(action) if str(s.get("uses", "")).startswith("github/codeql-action/upload-sarif@")
    )
    assert upload["uses"] == "github/codeql-action/upload-sarif@v3"
    assert upload["with"]["category"] == "jettae"


def scan_flags(script: str) -> set[str]:
    """Every ``--flag`` the scan step passes to ``jettae``."""
    return {
        token for token in shlex.split(script.replace("(", " ").replace(")", " ")) if token.startswith("--")
    }


def test_every_flag_the_action_passes_exists(action: dict[str, Any]) -> None:
    result = CliRunner().invoke(app, ["scan", "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    flags = scan_flags(step(action, "scan")["run"])
    assert flags == {"--llm", "--format", "--out", "--fail-under", "--summary-md", "--fail-on-critical"}
    for flag in flags:
        assert flag in result.stdout, flag


def test_gate_step_fails_the_job_after_the_upload(action: dict[str, Any]) -> None:
    names = [s.get("name") for s in steps(action)]
    upload = next(i for i, s in enumerate(steps(action)) if "upload-sarif" in str(s.get("uses", "")))
    gate = names.index("준비도 기준 확인")
    assert upload < gate
    assert "exit 1" in steps(action)[gate]["run"]


# --------------------------------------------------------------------------- workflows


def all_runs(workflow: dict[str, Any]) -> str:
    return "\n".join(s.get("run", "") for job in workflow["jobs"].values() for s in job["steps"])


def test_ci_quality_gates() -> None:
    ci = load(CI)
    assert {"push", "pull_request"} <= set(triggers(ci))
    runs = all_runs(ci)
    for command in (
        "ruff check .",
        "ruff format --check .",
        "mypy src",
        "lint-imports",
        "pytest --cov=jettae",
    ):
        assert command in runs, command
    assert re.search(r"jettae eval run [^\n]*--conditions B,O", runs)
    assert ci["env"]["JETTAE_LLM_MODE"] == "off"
    assert ci["permissions"] == {"contents": "read"}


def test_ci_builds_the_web_only_when_it_exists() -> None:
    web = load(CI)["jobs"]["web"]
    commands = {s.get("run"): s for s in web["steps"] if "run" in s}
    for command in ("npm ci", "npm run build"):
        assert commands[command]["working-directory"] == "web"
        assert "hashFiles('web/package.json')" in commands[command]["if"]


def test_example_workflow_uses_the_action() -> None:
    example = load(EXAMPLE)
    assert example["permissions"]["security-events"] == "write"
    (job,) = example["jobs"].values()
    uses = next(s for s in job["steps"] if s.get("uses") == "./")
    assert uses["with"]["path"] == "examples/dmc-shop"
    assert uses["with"]["llm"] == "off"
    assert "anthropic-api-key" not in uses["with"]  # off by default: no key, no runbook text leaves
    for each in job["steps"]:
        assert not EXPRESSION.search(each.get("run", ""))
