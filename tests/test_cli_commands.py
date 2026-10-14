"""``jettae rules|version|serve|demo|db`` — everything around the diagnosis.

No test binds a real server: uvicorn is replaced by a recorder, and the only sockets opened are
ephemeral ports chosen by the OS (the API's default port is 8765; port 8000 belongs to other
local software and is never touched).
"""

from __future__ import annotations

import json
import os
import socket
import sys
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import pytest
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine
from typer.testing import CliRunner

from jettae import __version__
from jettae.check.registry import rule_catalog
from jettae.cli import main
from jettae.cli.loops import UVICORN_LOOP
from jettae.cli.main import API_APP, DEFAULT_PORT, app
from jettae.domain import RuleInfo
from jettae.orchestrator import DEMO_WORKSPACE_NAME
from jettae.store import SqlStore, alembic_config
from jettae.store.sql import MIGRATIONS_DIR

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    for name in (
        "JETTAE_LLM_MODE",
        "JETTAE_DATABASE_URL",
        "JETTAE_STORE_RAW_DOCUMENTS",
        # serve/demo validate the token table before starting: a developer's own must not leak in
        "JETTAE_AUTH_MODE",
        "JETTAE_API_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)


def run(*args: str) -> tuple[int, str, str]:
    result = CliRunner().invoke(app, list(args))
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        raise result.exception
    return result.exit_code, result.stdout, result.stderr


@dataclass
class Recorder:
    """Stands in for ``uvicorn.run``: remembers how the server would have been started."""

    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((args, kwargs))


@pytest.fixture
def uvicorn_run(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    import uvicorn

    recorder = Recorder()
    monkeypatch.setattr(uvicorn, "run", recorder)
    return recorder


@pytest.fixture
def api_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """The API package is written by another module; these tests only need its import string."""
    monkeypatch.setattr(main, "_require_api", lambda: None)


@pytest.fixture
def free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


# --------------------------------------------------------------------------- rules, version


def test_rules_table_lists_every_rule() -> None:
    code, out, _ = run("rules")
    assert code == 0
    for rule in rule_catalog():
        assert rule.rule_id in out
    assert f"고정 규칙 {len(rule_catalog())}개" in out


def test_rules_json_and_markdown() -> None:
    code, out, _ = run("rules", "--format", "json")
    assert code == 0
    parsed = [RuleInfo.model_validate(item) for item in json.loads(out)]
    assert {r.rule_id for r in parsed} == {r.rule_id for r in rule_catalog()}
    code, out, _ = run("rules", "--format", "md")
    assert code == 0
    assert out.startswith("# 제때옵스 고정 규칙 카탈로그")
    code, _, err = run("rules", "--format", "yaml")
    assert code == 2
    assert "--format 값은 table, json, md 중 하나" in err


@pytest.mark.parametrize("args", [("version",), ("--version",), ("-V",)])
def test_version(args: tuple[str, ...]) -> None:
    code, out, _ = run(*args)
    assert code == 0
    assert f"jettae {__version__}" in out


def test_help_is_korean() -> None:
    code, out, _ = run("--help")
    assert code == 0
    assert "끊긴 대응 경로" in out
    for command in ("scan", "rehearse", "rules", "serve", "demo", "eval", "db", "version"):
        assert command in out


def test_invalid_environment_is_a_korean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JETTAE_LLM_MODE", "sometimes")
    code, _, err = run("rules")
    assert code == 2
    assert "환경 변수 설정이 올바르지 않습니다: JETTAE_LLM_MODE" in err


@pytest.mark.parametrize("value", ["{'kim': 1}", "", "not json"])
def test_unparseable_json_environment_is_a_korean_error(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """Malformed JSON is pydantic-settings' SettingsError, not a ValidationError: it gets the same
    Korean message naming the variable, not the library's English text."""
    monkeypatch.setenv("JETTAE_API_TOKENS", value)
    code, _, err = run("rules")
    assert code == 2
    assert "환경 변수 설정이 올바르지 않습니다: JETTAE_API_TOKENS" in err
    assert "error parsing" not in err


def test_settings_error_without_a_field_name_still_reads_korean(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic_settings import SettingsError

    def broken() -> None:
        raise SettingsError("unexpected source failure")

    monkeypatch.setattr(main, "Settings", broken)
    code, _, err = run("rules")
    assert code == 2
    assert "환경 변수 설정이 올바르지 않습니다: JETTAE_*" in err


# --------------------------------------------------------------------------- serve


def test_serve_defaults(uvicorn_run: Recorder, api_present: None, monkeypatch: pytest.MonkeyPatch) -> None:
    probed: list[tuple[str, int]] = []
    monkeypatch.setattr(main, "_ensure_port_free", lambda host, port: probed.append((host, port)))
    code, _, err = run("serve")
    assert code == 0, err
    ((args, kwargs),) = uvicorn_run.calls
    assert args == (API_APP,)
    assert kwargs == {
        "factory": True,
        "host": "127.0.0.1",
        "port": DEFAULT_PORT,
        "reload": False,
        "loop": UVICORN_LOOP,
        # uvicorn's access log prints client IPs and query strings; the API logs requests itself
        "access_log": False,
    }
    assert DEFAULT_PORT == 8765
    assert probed == [("127.0.0.1", 8765)]
    assert "http://127.0.0.1:8765/" in err


def test_serve_on_a_free_port(uvicorn_run: Recorder, api_present: None, free_port: int) -> None:
    code, _, _ = run("serve", "--port", str(free_port), "--reload")
    assert code == 0
    ((_, kwargs),) = uvicorn_run.calls
    assert kwargs["port"] == free_port
    assert kwargs["reload"] is True


def test_serve_refuses_a_busy_port(uvicorn_run: Recorder, api_present: None) -> None:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        port = int(busy.getsockname()[1])
        code, _, err = run("serve", "--port", str(port))
    assert code == 2
    assert f"127.0.0.1:{port} 포트를 다른 프로그램이 쓰고 있습니다" in err
    assert uvicorn_run.calls == []


@pytest.mark.parametrize("port", ["0", "70000"])
def test_serve_port_range(uvicorn_run: Recorder, api_present: None, port: str) -> None:
    code, _, err = run("serve", "--port", port)
    assert code == 2
    assert "1~65535" in err
    assert uvicorn_run.calls == []


def test_missing_api_module_is_a_korean_error(uvicorn_run: Recorder, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "API_MODULE", "jettae.no_such_package.app")
    code, _, err = run("serve")
    assert code == 2
    assert "API 모듈(jettae.no_such_package.app)을 찾을 수 없습니다" in err
    assert uvicorn_run.calls == []
    monkeypatch.setattr(main, "API_MODULE", "jettae.cli.output")
    main._require_api()  # an importable module passes


def test_unresolvable_host(uvicorn_run: Recorder, api_present: None) -> None:
    code, _, err = run("serve", "--host", "no-such-host.invalid")
    assert code == 2
    assert "호스트 주소를 해석할 수 없습니다" in err


def test_browser_url_for_wildcard_hosts() -> None:
    assert main._url("0.0.0.0", 8765) == "http://127.0.0.1:8765/"
    assert main._url("::", 8765) == "http://[::1]:8765/"
    assert main._url("localhost", 9) == "http://localhost:9/"


# --------------------------------------------------------------------------- demo


def test_demo_bootstraps_into_the_configured_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, uvicorn_run: Recorder, api_present: None, free_port: int
) -> None:
    url = f"sqlite:///{(tmp_path / 'demo.db').as_posix()}"
    monkeypatch.setenv("JETTAE_DATABASE_URL", url)
    opened: list[tuple[str, int]] = []
    monkeypatch.setattr(main, "_open_browser_when_ready", lambda host, port: opened.append((host, port)))
    code, _, err = run("demo", "--port", str(free_port), "--open")
    assert code == 0, err
    assert "판정 경로 LLM 0회" in err
    assert opened == [("127.0.0.1", free_port)]
    ((_, kwargs),) = uvicorn_run.calls
    assert kwargs["port"] == free_port
    with closing(SqlStore(url)) as store:
        (workspace,) = store.workspaces.list_workspaces()
        assert workspace.name == DEMO_WORKSPACE_NAME
        assert store.reports.latest_report(workspace.id) is not None
    # running it again reuses the workspace instead of creating a second one
    assert run("demo", "--port", str(free_port))[0] == 0
    with closing(SqlStore(url)) as store:
        assert len(store.workspaces.list_workspaces()) == 1


def test_demo_with_memory_store_only_serves(
    monkeypatch: pytest.MonkeyPatch, uvicorn_run: Recorder, api_present: None, free_port: int
) -> None:
    monkeypatch.setenv("JETTAE_DATABASE_URL", "memory://")
    code, _, err = run("demo", "--port", str(free_port))
    assert code == 0
    assert "메모리 저장소" in err
    assert len(uvicorn_run.calls) == 1


_SHORT_TOKEN: Final = '{"short": {"user": "heo", "role": "admin"}}'
_ROLE_TYPO: Final = '{"0123456789abcdef0123": {"user": "heo", "role": "admni"}}'


@pytest.mark.usefixtures("api_present")
@pytest.mark.parametrize(
    "case",
    [
        ("public", _SHORT_TOKEN, "16자 이상"),
        ("token", _ROLE_TYPO, "알 수 없는 역할"),
        ("public", "{}", "공개 모드(JETTAE_AUTH_MODE=public)"),
    ],
)
def test_demo_refuses_a_bad_token_table_before_writing_the_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    uvicorn_run: Recorder,
    free_port: int,
    case: tuple[str, str, str],
) -> None:
    """The server would refuse the table only inside uvicorn — after the demo was bootstrapped, so a
    restarting container would write a new demo report on every attempt."""
    mode, tokens, message = case
    database = tmp_path / "demo.db"
    monkeypatch.setenv("JETTAE_DATABASE_URL", f"sqlite:///{database.as_posix()}")
    monkeypatch.setenv("JETTAE_AUTH_MODE", mode)
    monkeypatch.setenv("JETTAE_API_TOKENS", tokens)
    code, _, err = run("demo", "--port", str(free_port))
    assert code == 2
    assert message in err
    assert uvicorn_run.calls == []
    assert not database.exists()  # nothing was opened, let alone written


def test_serve_refuses_a_bad_token_table(
    monkeypatch: pytest.MonkeyPatch, uvicorn_run: Recorder, api_present: None, free_port: int
) -> None:
    monkeypatch.setenv("JETTAE_AUTH_MODE", "token")
    monkeypatch.setenv("JETTAE_API_TOKENS", _SHORT_TOKEN)
    code, _, err = run("serve", "--port", str(free_port))
    assert code == 2
    assert "JETTAE_API_TOKENS의 1번째 토큰" in err
    assert uvicorn_run.calls == []


def test_demo_without_example_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, uvicorn_run: Recorder, api_present: None, free_port: int
) -> None:
    monkeypatch.setenv("JETTAE_DEMO_BUNDLE", str(tmp_path / "missing"))
    code, _, err = run("demo", "--port", str(free_port))
    assert code == 2
    assert "예제 묶음" in err
    assert uvicorn_run.calls == []


# --------------------------------------------------------------------------- db


def test_db_upgrade_calls_the_migrations(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(main, "upgrade_schema", lambda url, revision: calls.append((url, revision)))
    monkeypatch.setenv("JETTAE_DATABASE_URL", "postgresql+psycopg://jettae:s3cret@db/jettae")
    code, _, err = run("db", "upgrade")
    assert code == 0
    assert calls == [("postgresql+psycopg://jettae:s3cret@db/jettae", "head")]
    assert "s3cret" not in err  # the password is never printed
    assert "jettae:***@db/jettae" in err


def test_db_upgrade_refuses_memory_and_reports_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    code, _, err = run("db", "upgrade", "--url", "memory://")
    assert code == 2
    assert "메모리 저장소" in err
    code, _, err = run(
        "db", "upgrade", "--url", f"sqlite:///{(tmp_path / 'x.db').as_posix()}", "--revision", "nope"
    )
    assert code == 2  # a migration that cannot run is "could not run", not a failed gate (1)
    assert "스키마를 올리지 못했습니다" in err
    code, _, err = run("db", "upgrade", "--url", "::not a url::")
    assert code == 2
    assert "URL 형식" in err


@pytest.mark.skipif(
    os.name == "nt" and len(str(MIGRATIONS_DIR / "versions" / "0001_initial.py")) >= 260,
    reason="installed package path exceeds Windows MAX_PATH (covered by test_store_migrations)",
)
def test_db_upgrade_builds_the_schema(tmp_path: Path) -> None:
    url = f"sqlite:///{(tmp_path / 'jettae.db').as_posix()}"
    code, _, err = run("db", "upgrade", "--url", url)
    assert code == 0, err
    with closing(SqlStore(url, create_schema=False)) as store:
        assert store.workspaces.list_workspaces() == []


@pytest.mark.skipif(
    os.name == "nt" and len(str(MIGRATIONS_DIR / "versions" / "0001_initial.py")) >= 260,
    reason="installed package path exceeds Windows MAX_PATH (covered by test_store_migrations)",
)
def test_container_start_upgrades_then_bootstraps_on_every_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, uvicorn_run: Recorder, api_present: None, free_port: int
) -> None:
    """The image runs ``jettae db upgrade && exec jettae demo`` against ``JETTAE_DATABASE_URL`` at
    every start: the second start must neither fail on the migrated schema (the store's
    ``create_all`` is then a no-op) nor create a second demo workspace."""
    url = f"sqlite:///{(tmp_path / 'render.db').as_posix()}"
    monkeypatch.setenv("JETTAE_DATABASE_URL", url)
    for _ in range(2):
        code, _, err = run("db", "upgrade")
        assert code == 0, err
        code, _, err = run("demo", "--host", "127.0.0.1", "--port", str(free_port))
        assert code == 0, err
    assert len(uvicorn_run.calls) == 2
    head = ScriptDirectory.from_config(alembic_config()).get_current_head()
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            assert MigrationContext.configure(conn).get_current_revision() == head
    finally:
        engine.dispose()
    with closing(SqlStore(url, create_schema=False)) as store:
        (workspace,) = store.workspaces.list_workspaces()
        assert workspace.name == DEMO_WORKSPACE_NAME


def test_utf8_streams_for_the_console_script(monkeypatch: pytest.MonkeyPatch) -> None:
    """The console entry point makes redirected streams UTF-8 before Typer prints anything."""
    import io

    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stream)
    with pytest.raises(SystemExit) as exit_info:
        app(["version"])
    assert exit_info.value.code == 0
    stream.flush()
    assert f"jettae {__version__}" in raw.getvalue().decode("utf-8")
    assert "제때옵스" in raw.getvalue().decode("utf-8")
