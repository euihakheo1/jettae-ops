"""Adversarial cases for the ``jettae`` command line (independent review).

Each test pins a defect the first version had:

* a port another program holds on the *other* address of the family (``0.0.0.0`` vs
  ``127.0.0.1``) passed the free-port check on Windows, so ``serve`` would take over the other
  program's local connections;
* ``serve``/``demo`` in dev authentication on a reachable address said nothing, although anyone
  there can claim the ``lead`` role with one header;
* ``demo`` on an unusable database crashed with a traceback and exit status 1 — the code a CI job
  reads as "the gate failed" — and any internal error did the same;
* a bare ``KeyError`` from code was printed as if it were the user's mistake (no traceback);
* the organisation name taken from a file name printed the phone number that ingest masks in
  that very file name, and a malformed database URL printed its password.

No test binds port 8000 or starts a server: ports are ephemeral, sockets are bound and never
listen, and uvicorn is replaced by a recorder.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, NoReturn

import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st
from typer.testing import CliRunner

from jettae.cli import main
from jettae.cli.main import CliError, app
from jettae.domain.ports import NotFoundError

ROOT = Path(__file__).resolve().parents[1]
DEMO = Path("examples/dmc-shop")
PAYMENT_RUNBOOK = DEMO / "runbooks" / "01-결제-주문-장애대응.md"
SOCKET_ATTEMPTS = 8


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    for name in (
        "JETTAE_LLM_MODE",
        "JETTAE_DATABASE_URL",
        "JETTAE_STORE_RAW_DOCUMENTS",
        "JETTAE_AUTH_MODE",
        "JETTAE_API_TOKENS",
        "COLUMNS",
    ):
        monkeypatch.delenv(name, raising=False)


def run(*args: str) -> tuple[int, str, str]:
    result = CliRunner().invoke(app, list(args))
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        raise result.exception
    return result.exit_code, result.stdout, result.stderr


@pytest.fixture
def uvicorn_run(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    import uvicorn

    calls: list[dict[str, Any]] = []

    def record(*_args: Any, **kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(uvicorn, "run", record)
    return calls


@pytest.fixture
def api_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "_require_api", lambda: None)


@pytest.fixture
def no_port_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """For tests about something else: the default port is never bound, not even to probe it."""
    monkeypatch.setattr(main, "_ensure_port_free", lambda host, port: None)


@pytest.fixture
def hold_port() -> Iterator[Callable[[str], int]]:
    """Bind an ephemeral port on an IPv4 address the way another program would (bound, not
    listening — enough to own the port, and no firewall prompt); released after the test."""
    held: list[socket.socket] = []

    def hold(host: str) -> int:
        for _attempt in range(SOCKET_ATTEMPTS):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            held.append(sock)
            try:
                sock.bind((host, 0))
            except OSError:  # sandboxed hosts refuse a bind now and then (WinError 10014)
                continue
            return int(sock.getsockname()[1])
        pytest.fail(f"could not bind an ephemeral port on {host}")

    yield hold
    for sock in held:
        sock.close()


def _raise(exc: BaseException) -> Callable[..., NoReturn]:
    def fail(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise exc

    return fail


# --------------------------------------------------------------------------- port probe


@pytest.mark.parametrize(
    ("held", "requested"),
    [
        ("0.0.0.0", "127.0.0.1"),  # Windows let this bind and hijack the other program's localhost
        ("127.0.0.1", "0.0.0.0"),  # the printed http://127.0.0.1:P/ would reach the other program
        ("127.0.0.1", "127.0.0.1"),
        ("0.0.0.0", "0.0.0.0"),
    ],
)
def test_a_port_held_on_any_address_of_the_family_is_busy(
    hold_port: Callable[[str], int], held: str, requested: str
) -> None:
    port = hold_port(held)
    with pytest.raises(CliError, match=f"{requested}:{port} 포트를 다른 프로그램이 쓰고 있습니다"):
        main._ensure_port_free(requested, port)


def test_serve_refuses_a_port_held_on_the_wildcard_address(
    hold_port: Callable[[str], int], uvicorn_run: list[dict[str, Any]], api_present: None
) -> None:
    port = hold_port("0.0.0.0")
    code, _, err = run("serve", "--port", str(port))
    assert code == 2
    assert "다른 프로그램이 쓰고 있습니다" in err
    assert uvicorn_run == []


def test_a_released_port_is_free_again(hold_port: Callable[[str], int]) -> None:
    """The neighbour probes do not make every port look busy. Ephemeral ports are shared with
    every other process on the machine (a released one can be reused at once by someone's
    outgoing connection, and sandboxed hosts refuse a bind now and then), so fresh ports are
    tried until one passes; what must never happen is that none does."""
    passed = False
    for _attempt in range(4 * SOCKET_ATTEMPTS):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", 0))
            except OSError:
                continue
            port = int(sock.getsockname()[1])
        try:
            main._ensure_port_free("127.0.0.1", port)
            main._ensure_port_free("0.0.0.0", port)
        except CliError:
            time.sleep(0.1)
            continue
        passed = True
        break
    assert passed
    # ... and once another program holds it, the very same check refuses it
    held = hold_port("0.0.0.0")
    with pytest.raises(CliError):
        main._ensure_port_free("127.0.0.1", held)


def test_an_empty_host_means_every_interface() -> None:
    """``--host ""`` is the wildcard for asyncio, not this machine's own name (resolve only)."""
    addresses = main._listen_addresses("", main.DEFAULT_PORT)
    assert addresses
    assert {str(listen.address[0]) for listen in addresses} <= {"0.0.0.0", "::"}
    assert not any(listen.is_loopback for listen in addresses)
    assert all(listen.is_loopback for listen in main._listen_addresses("localhost", main.DEFAULT_PORT))


# --------------------------------------------------------------------------- serve / demo exposure


@pytest.mark.parametrize(
    ("host", "auth_mode", "warned"),
    [
        ("0.0.0.0", "dev", True),
        ("", "dev", True),
        ("127.0.0.1", "dev", False),
        ("localhost", "dev", False),
        ("0.0.0.0", "token", False),
        ("0.0.0.0", "public", False),
    ],
)
@pytest.mark.usefixtures("api_present", "no_port_probe")
def test_dev_auth_on_a_reachable_address_is_warned(
    monkeypatch: pytest.MonkeyPatch,
    uvicorn_run: list[dict[str, Any]],
    host: str,
    auth_mode: str,
    warned: bool,
) -> None:
    monkeypatch.setenv("JETTAE_AUTH_MODE", auth_mode)
    # token and public mode refuse to start without a valid token table (checked before serving)
    monkeypatch.setenv("JETTAE_API_TOKENS", '{"0123456789abcdef0123": {"user": "heo", "role": "admin"}}')
    code, _, err = run("serve", "--host", host)
    assert code == 0, err
    assert ("X-Jettae-Role" in err) is warned
    assert ("JETTAE_AUTH_MODE=token" in err) is warned
    assert len(uvicorn_run) == 1


def test_demo_warns_too(
    monkeypatch: pytest.MonkeyPatch,
    uvicorn_run: list[dict[str, Any]],
    api_present: None,
    no_port_probe: None,
) -> None:
    monkeypatch.setenv("JETTAE_DATABASE_URL", "memory://")
    code, _, err = run("demo", "--host", "0.0.0.0")
    assert code == 0, err
    assert "X-Jettae-Role" in err


def test_serve_reports_invalid_settings_before_starting(
    monkeypatch: pytest.MonkeyPatch, uvicorn_run: list[dict[str, Any]], api_present: None, no_port_probe: None
) -> None:
    """The server's factory reads the same settings; a bad one is a Korean error, not a traceback
    from inside uvicorn."""
    monkeypatch.setenv("JETTAE_AUTH_MODE", "everyone")
    code, _, err = run("serve")
    assert code == 2
    assert "환경 변수 설정이 올바르지 않습니다: JETTAE_AUTH_MODE" in err
    assert uvicorn_run == []


# --------------------------------------------------------------------------- unusable databases


def test_demo_on_a_file_that_is_not_a_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    uvicorn_run: list[dict[str, Any]],
    api_present: None,
    no_port_probe: None,
) -> None:
    garbage = tmp_path / "jettae.db"
    garbage.write_bytes(b"this is not a database, only text " * 8)
    monkeypatch.setenv("JETTAE_DATABASE_URL", f"sqlite:///{garbage.as_posix()}")
    code, _, err = run("demo")  # run() re-raises a crash, so reaching the asserts means none
    assert code == 2
    assert "오류: 데이터베이스를 열 수 없습니다" in err
    assert "file is not a database" in err
    assert "[SQL:" not in err  # one line, not SQLAlchemy's statement dump
    assert uvicorn_run == []


def test_demo_without_the_database_driver_masks_the_password(
    monkeypatch: pytest.MonkeyPatch,
    uvicorn_run: list[dict[str, Any]],
    api_present: None,
    no_port_probe: None,
) -> None:
    monkeypatch.setattr(main, "open_store", _raise(ModuleNotFoundError("no psycopg", name="psycopg")))
    monkeypatch.setenv("JETTAE_DATABASE_URL", "postgresql+psycopg://jettae:s3cret-pw@db/jettae")
    code, _, err = run("demo")
    assert code == 2
    assert "psycopg" in err
    assert "jettae-ops[postgres]" in err
    assert "jettae:***@db/jettae" in err
    assert "s3cret-pw" not in err
    assert uvicorn_run == []


def test_malformed_database_urls_never_print_the_password(
    monkeypatch: pytest.MonkeyPatch,
    uvicorn_run: list[dict[str, Any]],
    api_present: None,
    no_port_probe: None,
) -> None:
    malformed = "postgresql//jettae:s3cret-pw@db/jettae"  # "://" mistyped
    code, _, err = run("db", "upgrade", "--url", malformed)
    assert code == 2
    assert "URL 형식" in err
    assert "s3cret-pw" not in err
    monkeypatch.setenv("JETTAE_DATABASE_URL", malformed)
    code, _, err = run("demo")
    assert code == 2
    assert "s3cret-pw" not in err
    assert uvicorn_run == []


@given(
    user=st.text(alphabet="abc", min_size=1, max_size=8),
    password=st.text(alphabet="XYZ0123456789/", min_size=6, max_size=24),
    malformed=st.booleans(),
)
def test_a_database_password_is_never_shown(user: str, password: str, malformed: bool) -> None:
    url = f"postgresql{'//' if malformed else '+psycopg://'}{user}:{password}@db/jettae"
    try:
        shown = main._masked_url(url)
    except CliError as exc:
        shown = exc.message
    assert password not in shown


# --------------------------------------------------------------------------- exit codes of defects


def test_an_internal_error_exits_2_with_its_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Through the console entry point: Python's own status for a crash is 1, which the Action
    (and any CI job) reads as "the gate failed, results saved"."""
    monkeypatch.setattr(main, "_diagnose", _raise(RuntimeError("boom-in-the-pipeline")))
    with pytest.raises(SystemExit) as exit_info:
        app(["scan", PAYMENT_RUNBOOK.as_posix()])
    assert exit_info.value.code == main.EXIT_USAGE
    err = capsys.readouterr().err
    assert "Traceback" in err
    assert "boom-in-the-pipeline" in err
    assert "예기치 못한 내부 오류" in err


def test_a_failed_lookup_in_code_is_a_defect_not_a_user_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "_diagnose", _raise(KeyError("graph_hash")))
    result = CliRunner().invoke(app, ["scan", PAYMENT_RUNBOOK.as_posix()])
    assert isinstance(result.exception, KeyError)  # reaches the crash handler, traceback kept
    monkeypatch.setattr(main, "_diagnose", _raise(NotFoundError("워크스페이스 w-1을(를) 찾을 수 없습니다.")))
    code, _, err = run("scan", PAYMENT_RUNBOOK.as_posix())
    assert code == 2
    assert "오류: 워크스페이스 w-1을(를) 찾을 수 없습니다." in err


# --------------------------------------------------------------------------- personal data


def test_the_org_name_from_a_file_name_is_pii_masked(tmp_path: Path) -> None:
    runbook = tmp_path / "연락망_010-1234-5678.md"
    shutil.copyfile(ROOT / PAYMENT_RUNBOOK, runbook)
    code, md, err = run("scan", str(runbook), "--format", "md", "--no-rehearse", "--no-draft")
    assert code == 0, err
    assert "010-1234-5678" not in md  # the org name was printed unmasked next to the masked path
    assert "연락망_010-****-5678.md" in md  # the document, masked by ingest
    # an explicit --name is a label like a caller-chosen title: masked the same way
    target = main._load_target(PAYMENT_RUNBOOK, name="별빛 당직 010-9876-5432")
    assert target.name == "별빛 당직 010-****-5432"
    assert main._load_target(runbook).name == "연락망_010-****-5678"


@given(
    prefix=st.text(alphabet="가나다라 abcXYZ", min_size=1, max_size=12),
    middle=st.from_regex(r"[0-9]{4}", fullmatch=True),
    last=st.from_regex(r"[0-9]{4}", fullmatch=True),
)
def test_org_names_never_carry_a_mobile_number(prefix: str, middle: str, last: str) -> None:
    phone = f"010-{middle}-{last}"
    assert phone not in main._org_name(f"{prefix}_{phone}")


def test_org_name_keeps_ordinary_names() -> None:
    assert main._org_name("별빛상점") == "별빛상점"
    assert main._org_name("  dmc-shop  ") == "dmc-shop"
    assert main._org_name("   ") == "런북 진단"


# --------------------------------------------------------------------------- eval files


def test_eval_files_that_are_not_utf8_are_korean_errors(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    cases.write_bytes(b"\xff\xfe{not utf-8}\n")
    code, _, err = run("eval", "seal", "--cases", str(cases))
    assert code == 2
    assert "평가 사례 파일 형식이 올바르지 않습니다" in err


def test_eval_build_refuses_a_folder_before_generating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main, "_evalkit", _raise(AssertionError("the corpus was generated first")))
    code, _, err = run("eval", "build", "--out", str(tmp_path))
    assert code == 2
    assert "출력 경로가 폴더입니다" in err


# --------------------------------------------------------------------------- action


def test_action_job_summary_can_be_turned_off() -> None:
    action: dict[str, Any] = yaml.safe_load((ROOT / "action.yml").read_text(encoding="utf-8"))
    assert action["inputs"]["job-summary"]["default"] == "true"
    scan = next(s for s in action["runs"]["steps"] if s.get("id") == "scan")
    assert scan["env"]["INPUT_JOB_SUMMARY"] == "${{ inputs.job-summary }}"
    assert (
        'if [ "$INPUT_JOB_SUMMARY" = "true" ]; then args+=(--summary-md "$GITHUB_STEP_SUMMARY"); fi'
        in scan["run"]
    )
    # the documentation names where quotes of the runbooks end up
    header = (ROOT / "action.yml").read_text(encoding="utf-8").split("\nname:", 1)[0]
    assert "코드 스캐닝" in header
    assert "작업 요약" in header


# --------------------------------------------------------------------------- module entry points


@pytest.mark.parametrize("module", ["jettae.cli", "jettae.cli.main"])
def test_python_m_runs_the_command(module: str) -> None:
    """``python -m jettae.cli.main`` used to import the app and exit 0 without running anything —
    a CI gate written that way always passed."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    done = subprocess.run(
        [sys.executable, "-m", module, "scan", "no/such/bundle"],
        cwd=ROOT,
        capture_output=True,
        env=env,
        check=False,
    )
    assert done.returncode == main.EXIT_USAGE
    assert "경로를 찾을 수 없습니다: no/such/bundle" in done.stderr.decode("utf-8")
    version = subprocess.run(
        [sys.executable, "-m", module, "version"], cwd=ROOT, capture_output=True, env=env, check=False
    )
    assert version.returncode == 0
    assert "jettae" in version.stdout.decode("utf-8")
