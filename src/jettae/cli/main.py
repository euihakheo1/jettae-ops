"""``jettae`` — the command line of 제때옵스(JETTAE Ops).

Why it is shaped this way:

* **One façade, no logic of its own.** Every diagnosis goes through
  :class:`~jettae.orchestrator.JettaeService` — the same use cases the REST API serves — on an
  in-memory store for the one-shot commands (``scan``, ``rehearse``, ``rules``): a CI run leaves no
  database behind and two runs never see each other. Only ``demo`` and ``db`` touch the configured
  database (``JETTAE_DATABASE_URL``).
* **Exit codes are a contract for CI.** 0 = done (and every gate passed), 1 = a gate failed
  (``--fail-under``, ``--fail-on-critical``, a seal mismatch, a failed self-test or reproducibility
  check), 2 = the command could not run (missing path, unreadable input, invalid option, an
  unusable database — and an unexpected internal error, which keeps its traceback: Python's own
  exit status for a crash is 1, which a CI job would read as "the gate failed, the results are
  saved"). Messages are Korean and go to stderr, so stdout stays clean for
  ``--format json|md|html|sarif``.
* **Late binding of siblings.** ``api``, ``cli`` and ``evalkit`` are independent siblings of the top
  layer (import-linter "Layers" contract), so none of them imports another. ``serve``/``demo`` hand
  uvicorn the API's *import string*, and ``eval`` loads :mod:`jettae.evalkit` *by name* when one of
  its commands runs. As a bonus ``jettae scan`` never loads a web server, the corpus generator or
  the evaluation harness.
* **LLM off unless asked** (P4). ``--llm`` overrides ``JETTAE_LLM_MODE``; with ``live`` runbook text
  is sent only to the provider of the user's own key. Whatever the mode, the judgement path —
  rules, scores, replays — never calls a model (P1), and every summary says so.
"""

from __future__ import annotations

import asyncio
import errno
import importlib
import importlib.util
import ipaddress
import math
import os
import re
import socket
import sys
import threading
import time
import traceback
import webbrowser
from collections.abc import Callable, Coroutine, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from typing import Annotated, Any, Final, NoReturn, TypeVar

import typer
from pydantic import ValidationError
from pydantic_settings import SettingsError
from rich.console import RenderableType
from rich.text import Text

from jettae import __version__
from jettae.cli.output import (
    DataFormat,
    RulesFormat,
    ScanFormat,
    ScanSummary,
    SourceLocator,
    data_table,
    display_prefix,
    ensure_utf8_streams,
    is_plain,
    key_value_table,
    located_documents,
    make_console,
    rehearsal_payload,
    rehearsal_renderables,
    render_plain,
    rules_json,
    rules_markdown,
    rules_table,
    scan_payload,
    scan_renderables,
    to_json,
    write_text,
)
from jettae.config import Settings
from jettae.domain import DiagnosisReport, InputBundle, LlmMode, OpsGraph, Severity
from jettae.domain.ports import NotFoundError
from jettae.ingest import (
    DEFAULT_MAX_BYTES,
    MEDIA_TYPES,
    load_bundle_dir,
    load_bytes,
    mask_pii,
    normalize_text,
)
from jettae.ingest.loaders import read_limited
from jettae.llm.factory import build_llm
from jettae.orchestrator import JettaeService, demo_bundle_path
from jettae.reporting import to_sarif
from jettae.reporting.evidence import KST
from jettae.store import MEMORY_URL, MemoryStore, SqlStore, open_store, upgrade_schema

__all__ = [
    "API_APP",
    "CLI_ACTOR",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "EXIT_GATE",
    "EXIT_OK",
    "EXIT_USAGE",
    "CliError",
    "app",
]

_T = TypeVar("_T")
_E = TypeVar("_E", bound=StrEnum)

EXIT_OK: Final = 0
#: A gate failed: the run itself worked, its result is below the bar the caller set.
EXIT_GATE: Final = 1
#: The command could not run as asked (Click uses 2 for usage errors too).
EXIT_USAGE: Final = 2

#: Audit-log actor of one-shot runs.
CLI_ACTOR: Final = "cli"
#: The API application factory, bound by uvicorn at run time (see the module docstring).
API_APP: Final = "jettae.api.app:create_app"
API_MODULE: Final = "jettae.api.app"
#: Holds the ``Authenticator`` the server builds at start-up (bound by name, like the app).
API_SECURITY_MODULE: Final = "jettae.api.security"
EVALKIT_MODULE: Final = "jettae.evalkit"
#: pydantic-settings names the field it could not parse: ``error parsing value for field "api_tokens" …``
_SETTINGS_FIELD: Final = re.compile(r'field "([A-Za-z0-9_]+)"')
DEFAULT_HOST: Final = "127.0.0.1"
#: Not 8000: that port is commonly taken by other local development servers.
DEFAULT_PORT: Final = 8765
#: Hand-written drill files a bundle may carry next to ``runbooks/`` (see ``examples/dmc-shop``).
SCENARIO_FILES: Final = ("scenarios.yaml", "scenarios.yml")
DEFAULT_EVAL_DIR: Final = Path("docs/eval")
DEFAULT_CASES: Final = DEFAULT_EVAL_DIR / "cases.jsonl"
#: Results of a smoke or sampled run go here by default: they must never overwrite the published
#: full-corpus numbers in ``docs/eval``.
DEFAULT_SAMPLE_DIR: Final = DEFAULT_EVAL_DIR / "sample"
DEFAULT_CONDITIONS: Final = "A,B,O"
#: Event-loop creation is retried on a transient ``OSError`` (see :func:`_run`).
_LOOP_ATTEMPTS: Final = 8
_LOOP_BACKOFF_SECONDS: Final = 0.05
#: Port probes are retried the same way, but fewer times: a real error must not stall the start.
_PROBE_ATTEMPTS: Final = 4
#: How long ``demo --open`` waits for the server before giving up on the browser.
_BROWSER_WAIT_SECONDS: Final = 30.0


class CliError(Exception):
    """A Korean message for the user and the exit code that goes with it."""

    def __init__(self, message: str, code: int = EXIT_USAGE) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class _JettaeApp(typer.Typer):
    """The Typer app behind the console entry point.

    * stdout/stderr become UTF-8 *before* anything — even ``--help`` — is printed (see
      :func:`~jettae.cli.output.ensure_utf8_streams`).
    * An exception no command turned into a Korean message is a defect. Left alone, Python would
      exit with 1 — the code that tells a CI job "the gate failed, the results are saved" — so a
      crash would pass for a low score. It exits with 2 ("could not run") instead, with the
      traceback kept for the bug report (locals are never printed: they hold runbook text).
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ensure_utf8_streams()
        try:
            return super().__call__(*args, **kwargs)
        except Exception as exc:
            traceback.print_exception(exc, file=sys.stderr)
            _stderr(
                "오류: 예기치 못한 내부 오류로 중단했습니다(제때옵스 결함). 위 오류 내용을 함께 알려 주세요.",
                style="bold red",
            )
            raise SystemExit(EXIT_USAGE) from None


_TYPER_SETTINGS: Final[dict[str, Any]] = {
    "no_args_is_help": True,
    "add_completion": False,
    # Help texts are Korean prose with brackets ("[off|live|replay]"); never parse them as markup.
    "rich_markup_mode": None,
    # A traceback must never print local variables: they hold runbook text and settings.
    "pretty_exceptions_show_locals": False,
    "context_settings": {"help_option_names": ["-h", "--help"]},
}

app = _JettaeApp(
    name="jettae",
    help=(
        "제때옵스(JETTAE Ops) — 장애 대응 런북을 운영 그래프로 바꿔, 끊긴 대응 경로를 장애 전에 찾습니다. "
        "끊김 판정·점수·리허설은 고정 규칙과 결정적 리플레이만 씁니다(LLM 0회)."
    ),
    **_TYPER_SETTINGS,
)
eval_app = typer.Typer(
    name="eval",
    help="봉인 결함 주입 평가셋: 코퍼스 만들기 → 봉인 → 조건별 실행 → 보고서, 프롬프트 인젝션 자기 시험.",
    **_TYPER_SETTINGS,
)
db_app = typer.Typer(name="db", help="데이터베이스(JETTAE_DATABASE_URL) 관리.", **_TYPER_SETTINGS)
app.add_typer(eval_app, name="eval")
app.add_typer(db_app, name="db")


# --------------------------------------------------------------------------- plumbing


def _stderr(message: str, *, style: str = "") -> None:
    """A message for people on stderr, never hard-wrapped (log search and CI annotations need
    whole lines)."""
    make_console(sys.stderr).print(Text(message, style=style), soft_wrap=True)


def _echo(text: str) -> None:
    """Machine output (JSON, Markdown, HTML, SARIF) to stdout, byte-for-byte as rendered."""
    sys.stdout.write(text if text.endswith("\n") else text + "\n")
    sys.stdout.flush()


def _show(renderables: Sequence[RenderableType]) -> None:
    """People's output: Rich on a terminal, plain text (trailing blanks trimmed) elsewhere."""
    console = make_console(sys.stdout)
    if is_plain(console):
        sys.stdout.write(render_plain(renderables, width=console.width))
        sys.stdout.flush()
        return
    for renderable in renderables:
        console.print(renderable)


def _plain() -> bool:
    return is_plain(make_console(sys.stdout))


def _exit(message: str, code: int) -> NoReturn:
    _stderr(f"오류: {message}" if code == EXIT_USAGE else message, style="bold red")
    raise typer.Exit(code)


def _os_message(exc: OSError) -> str:
    if exc.filename is None:
        return str(exc)
    name = Path(os.fsdecode(exc.filename)).as_posix()
    if isinstance(exc, FileNotFoundError):
        return f"파일이나 폴더를 찾을 수 없습니다: {name}"
    if isinstance(exc, IsADirectoryError):
        return f"파일이 아니라 폴더입니다: {name}"
    if isinstance(exc, PermissionError):
        return f"접근 권한이 없습니다: {name}"
    return f"파일을 다룰 수 없습니다({name}): {exc.strerror or exc}"


@contextmanager
def _errors() -> Iterator[None]:
    """Turn the errors a user can fix into a Korean message and an exit code.

    Domain and ingest errors are ``ValueError`` subclasses and :class:`NotFoundError` with Korean
    messages by contract (the API maps exactly these to HTTP 422/404), so they are shown as they
    are. A bare ``KeyError`` is a failed lookup in code, not something the user can fix: like
    anything else it is a defect and keeps its traceback (see :class:`_JettaeApp`).
    """
    try:
        yield
    except CliError as exc:
        _exit(exc.message, exc.code)
    except OSError as exc:
        _exit(_os_message(exc), EXIT_USAGE)
    except NotFoundError as exc:
        _exit(str(exc.args[0]) if exc.args else "찾을 수 없는 항목입니다.", EXIT_USAGE)
    except ValueError as exc:
        _exit(str(exc), EXIT_USAGE)


def _settings() -> Settings:
    """Settings from ``JETTAE_*`` environment variables (and ``.env``), read fresh per command.

    A value of the wrong shape is a ``ValidationError``; a value that is not even parseable — the
    malformed JSON of ``JETTAE_API_TOKENS``, say — is pydantic-settings' ``SettingsError``, whose
    English message names the field. Both become the same Korean message naming the variable.
    """
    try:
        return Settings()
    except ValidationError as exc:
        names = sorted({"JETTAE_" + str(error["loc"][0]).upper() for error in exc.errors() if error["loc"]})
        raise CliError(f"환경 변수 설정이 올바르지 않습니다: {', '.join(names) or exc.title}") from None
    except SettingsError as exc:
        field = _SETTINGS_FIELD.search(str(exc))
        name = f"JETTAE_{field.group(1).upper()}" if field else "JETTAE_* (JSON 형식 확인)"
        raise CliError(f"환경 변수 설정이 올바르지 않습니다: {name}") from None


def _choice(enum: type[_E], value: str, option: str) -> _E:
    try:
        return enum(value.strip().lower())
    except ValueError:
        allowed = ", ".join(member.value for member in enum)
        raise CliError(f"{option} 값은 {allowed} 중 하나여야 합니다: {value}") from None


def _llm_mode(value: str | None, settings: Settings) -> LlmMode:
    return settings.llm_mode if value is None else _choice(LlmMode, value, "--llm")


def _event_loop() -> asyncio.Runner:
    attempt = 1
    while True:
        runner = asyncio.Runner()
        try:
            runner.get_loop()
        except OSError:
            runner.close()
            if attempt >= _LOOP_ATTEMPTS:
                raise
            time.sleep(_LOOP_BACKOFF_SECONDS * attempt)
            attempt += 1
            continue
        return runner


def _run(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run ``coro`` on a fresh event loop.

    Opening a Windows event loop creates a loopback socket pair, which some sandboxed hosts refuse
    now and then (``WinError 10014``); the loop's *creation* is retried with a short back-off, the
    work itself never is.
    """
    try:
        runner = _event_loop()
    except BaseException:
        coro.close()  # never started: close it so it is not reported as "never awaited"
        raise
    with runner:
        return runner.run(coro)


def _shown(path: Path) -> str:
    return path.as_posix()


# --------------------------------------------------------------------------- targets


@dataclass(frozen=True, slots=True)
class _Target:
    """What one scan diagnoses, and how its documents are named from the working directory."""

    bundle: InputBundle
    scenarios: bytes | None
    name: str
    #: display/SARIF prefix of the bundle-relative document paths (see ``display_prefix``)
    prefix: str
    shown: str


def _bundled_drills(directory: Path) -> Path | None:
    found = [directory / name for name in SCENARIO_FILES if (directory / name).is_file()]
    if len(found) > 1:
        names = ", ".join(p.name for p in found)
        raise CliError(f"시나리오 파일이 여러 개 있습니다: {names}. 하나만 남겨 주세요.")
    return found[0] if found else None


def _read_drills(path: Path) -> bytes:
    if not path.is_file():
        raise CliError(f"시나리오 파일을 찾을 수 없습니다: {_shown(path)}")
    return path.read_bytes()


def _single_document(path: Path) -> InputBundle:
    """A bundle of one runbook file. The document is named by its file name only, exactly like a
    runbook inside a bundle is named relative to the bundle, so where the file lies on disk never
    changes the graph hash (P6); the display prefix adds the location back for people."""
    if path.suffix.lower() not in MEDIA_TYPES:
        supported = ", ".join(sorted(MEDIA_TYPES))
        raise CliError(f"지원하지 않는 런북 형식입니다: {_shown(path)} (지원 형식: {supported})")
    document = load_bytes(path.name, read_limited(path, path.name, DEFAULT_MAX_BYTES))
    return InputBundle(documents=(document,))


def _org_name(raw: str) -> str:
    """The organisation name printed on every report, PII-masked like a runbook's file name.

    By default it is the bundle folder's or the runbook file's name, and ingest masks personal
    data in file names because "연락망_010-1234-5678.md" would otherwise print the number in every
    report (see :func:`~jettae.ingest.load_bytes`); the name derived from that same file must not
    print it next to the masked path. An explicit ``--name`` is a label like a caller-chosen title
    and is masked the same way.
    """
    masked, _hits = mask_pii(normalize_text(raw))
    return masked.strip() or "런북 진단"


def _load_target(path: Path, *, scenarios: Path | None = None, name: str | None = None) -> _Target:
    """Load a bundle directory or a single runbook file (Korean errors for everything else)."""
    if not path.exists():
        raise CliError(f"경로를 찾을 수 없습니다: {_shown(path)} (런북 묶음 폴더나 런북 파일을 지정하세요)")
    if path.is_dir():
        bundle = load_bundle_dir(path)
        drills = scenarios if scenarios is not None else _bundled_drills(path)
        root, default_name = path, Path(os.path.abspath(path)).name
    elif path.is_file():
        bundle = _single_document(path)
        drills = scenarios
        root, default_name = path.parent, path.stem
    else:
        raise CliError(f"런북 폴더나 파일이 아닙니다: {_shown(path)}")
    return _Target(
        bundle=bundle,
        scenarios=_read_drills(drills) if drills is not None else None,
        name=_org_name(name or default_name),
        prefix=display_prefix(root, Path.cwd()),
        shown=_shown(path),
    )


@dataclass(frozen=True, slots=True)
class _Run:
    """One diagnosis on a private in-memory store."""

    service: JettaeService
    workspace_id: str
    report: DiagnosisReport
    graph: OpsGraph
    notes: tuple[str, ...]


def _diagnose(target: _Target, settings: Settings, mode: LlmMode, *, rehearse: bool, draft: bool) -> _Run:
    service = JettaeService(MemoryStore(), settings)
    workspace = service.create_workspace(target.name)
    service.load_bundle(workspace.id, target.bundle, scenarios=target.scenarios)
    report = _run(
        service.diagnose(workspace.id, llm_mode=mode, actor=CLI_ACTOR, rehearse=rehearse, draft=draft)
    )
    _version, graph = service.graph(workspace.id, report.graph_version)
    return _Run(
        service=service,
        workspace_id=workspace.id,
        report=report,
        graph=graph,
        notes=service.report_notes(workspace.id, report.run_id),
    )


def _check_score(value: float, option: str) -> float:
    if not math.isfinite(value) or not 0 <= value <= 100:
        raise CliError(f"{option} 값은 0~100 사이의 점수여야 합니다: {value:g}")
    return value


def _gate_failures(report: DiagnosisReport, *, fail_under: float, fail_on_critical: bool) -> list[str]:
    """Why the CI gate fails (empty = passes). Critical findings count whether broken or unknown:
    an unverifiable critical path is not a passing one (P3)."""
    failures: list[str] = []
    overall = report.readiness.overall
    if overall < fail_under:
        failures.append(f"전체 준비도 {overall:.1f}점이 기준 {fail_under:g}점보다 낮습니다(--fail-under).")
    if fail_on_critical:
        critical = sum(1 for f in report.findings if f.severity is Severity.CRITICAL)
        if critical:
            failures.append(
                f"심각(critical) 끊김이 {critical}건 있습니다(확인 불가 포함, --fail-on-critical)."
            )
    return failures


# --------------------------------------------------------------------------- scan


def _render_scan(
    scan: _Run, target: _Target, fmt: ScanFormat, summary: ScanSummary, failures: Sequence[str]
) -> str:
    report = scan.report
    if fmt is ScanFormat.TEXT:
        return render_plain(scan_renderables(summary, plain=True))
    if fmt is ScanFormat.MD:
        return scan.service.evidence_markdown(scan.workspace_id)
    if fmt is ScanFormat.HTML:
        return scan.service.evidence_html(scan.workspace_id)
    documents = located_documents(scan.graph.docs, target.prefix)
    if fmt is ScanFormat.SARIF:
        return to_json(to_sarif(report, scan.graph, documents=documents, tool_version=__version__))
    payload = scan_payload(
        report,
        target=target.shown,
        tool_version=__version__,
        documents=documents,
        notes=scan.notes,
        gate_failures=failures,
    )
    return to_json(payload)


@app.command(
    help=(
        "런북을 진단합니다: 운영 그래프를 만들고, 고정 규칙으로 끊긴 대응 경로를 찾고, 준비도(0~100)를 "
        "매기고, 모의 장애를 재생하고, 수정안 초안을 만듭니다. 기준 미달이면 종료 코드 1."
    )
)
def scan(
    *,
    path: Annotated[
        Path,
        typer.Argument(
            metavar="PATH",
            help="런북 묶음 폴더(runbooks/, roster.csv, catalog.yaml …) 또는 런북 파일 하나",
            show_default=False,
        ),
    ],
    llm: Annotated[
        str | None,
        typer.Option("--llm", metavar="off|live|replay", help="LLM 모드 (기본: JETTAE_LLM_MODE, 없으면 off)"),
    ] = None,
    output_format: Annotated[
        str, typer.Option("--format", "-f", metavar="text|json|md|html|sarif", help="출력 형식")
    ] = ScanFormat.TEXT.value,
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="결과를 이 파일에 저장 (화면에는 요약을 출력)", show_default=False),
    ] = None,
    fail_under: Annotated[
        float, typer.Option("--fail-under", help="전체 준비도가 이 점수보다 낮으면 종료 코드 1")
    ] = 0.0,
    fail_on_critical: Annotated[
        bool, typer.Option("--fail-on-critical", help="심각(critical) 끊김이 하나라도 있으면 종료 코드 1")
    ] = False,
    no_rehearse: Annotated[bool, typer.Option("--no-rehearse", help="모의 장애 리허설을 건너뜀")] = False,
    no_draft: Annotated[bool, typer.Option("--no-draft", help="수정안 초안을 만들지 않음")] = False,
    summary_md: Annotated[
        Path | None,
        typer.Option(
            "--summary-md",
            help="Markdown 증적 리포트를 이 파일 끝에 덧붙임 (예: $GITHUB_STEP_SUMMARY, 같은 진단 결과)",
            show_default=False,
        ),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option("--name", help="리포트에 적을 조직 이름 (기본: 폴더·파일 이름)", show_default=False),
    ] = None,
    top: Annotated[int, typer.Option("--top", help="요약에 보여 줄 끊김 수")] = 10,
) -> None:
    with _errors():
        settings = _settings()
        mode = _llm_mode(llm, settings)
        fmt = _choice(ScanFormat, output_format, "--format")
        _check_score(fail_under, "--fail-under")
        if top < 0:
            raise CliError(f"--top 값은 0 이상이어야 합니다: {top}")
        target = _load_target(path, name=name)
        result = _diagnose(target, settings, mode, rehearse=not no_rehearse, draft=not no_draft)
        failures = _gate_failures(result.report, fail_under=fail_under, fail_on_critical=fail_on_critical)
        summary = ScanSummary(
            report=result.report,
            target=target.shown,
            locator=SourceLocator.of(result.graph.docs, target.prefix),
            notes=result.notes,
            top=top,
            rehearsed=not no_rehearse,
            drafted=not no_draft,
        )
        # Files first: a CI job keeps its SARIF and job summary even when the gate fails below.
        if out is not None:
            write_text(out, _render_scan(result, target, fmt, summary, failures))
        if summary_md is not None:
            write_text(summary_md, result.service.evidence_markdown(result.workspace_id), append=True)
        if out is None and fmt is not ScanFormat.TEXT:
            _echo(_render_scan(result, target, fmt, summary, failures))
        else:
            _show(scan_renderables(summary, plain=_plain()))
        if out is not None:
            _stderr(f"{fmt.value} 결과를 저장했습니다: {_shown(out)}")
        if failures:
            for failure in failures:
                _stderr(f"기준 미달: {failure}", style="bold red")
            raise typer.Exit(EXIT_GATE)


# --------------------------------------------------------------------------- rehearse


@app.command(
    help=(
        "모의 장애 리허설: 런북 그래프 위에서 담당자 부재·메신저 다운·센터 장애 같은 시나리오를 "
        "결정적으로 재생합니다(실제 시스템은 건드리지 않음)."
    )
)
def rehearse(
    *,
    path: Annotated[
        Path, typer.Argument(metavar="PATH", help="런북 묶음 폴더 또는 런북 파일 하나", show_default=False)
    ],
    scenarios: Annotated[
        Path | None,
        typer.Option(
            "--scenarios",
            help="사람이 쓴 시나리오 파일(scenarios.yaml). 기본: 묶음 폴더의 scenarios.yaml",
            show_default=False,
        ),
    ] = None,
    llm: Annotated[
        str | None,
        typer.Option("--llm", metavar="off|live|replay", help="LLM 모드 (기본: JETTAE_LLM_MODE, 없으면 off)"),
    ] = None,
    output_format: Annotated[str, typer.Option("--format", "-f", metavar="text|json", help="출력 형식")] = (
        DataFormat.TEXT.value
    ),
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="결과를 이 파일에 저장", show_default=False)
    ] = None,
    name: Annotated[str | None, typer.Option("--name", help="조직 이름", show_default=False)] = None,
) -> None:
    with _errors():
        settings = _settings()
        mode = _llm_mode(llm, settings)
        fmt = _choice(DataFormat, output_format, "--format")
        target = _load_target(path, scenarios=scenarios, name=name)
        result = _diagnose(target, settings, mode, rehearse=True, draft=False)
        report = result.report
        if fmt is DataFormat.JSON:
            payload = to_json(
                rehearsal_payload(report, target=target.shown, tool_version=__version__, notes=result.notes)
            )
            if out is None:
                _echo(payload)
            else:
                write_text(out, payload)
        else:
            if out is not None:
                write_text(
                    out,
                    render_plain(rehearsal_renderables(report, target=target.shown, notes=result.notes)),
                )
            _show(rehearsal_renderables(report, target=target.shown, notes=result.notes))
        if out is not None:
            _stderr(f"{fmt.value} 결과를 저장했습니다: {_shown(out)}")


# --------------------------------------------------------------------------- rules, version


@app.command(help="판정에 쓰는 고정 규칙 전체를 보여 줍니다(판정 기준 공개 — 누구나 감사할 수 있게).")
def rules(
    *,
    output_format: Annotated[
        str, typer.Option("--format", "-f", metavar="table|json|md", help="출력 형식")
    ] = (RulesFormat.TABLE.value),
) -> None:
    with _errors():
        fmt = _choice(RulesFormat, output_format, "--format")
        catalog = JettaeService(MemoryStore(), _settings()).rule_catalog()
        if fmt is RulesFormat.JSON:
            _echo(rules_json(catalog))
        elif fmt is RulesFormat.MD:
            _echo(rules_markdown(catalog))
        else:
            _show([rules_table(catalog, plain=_plain())])


def _version_text() -> str:
    python = ".".join(str(part) for part in sys.version_info[:3])
    return f"제때옵스(JETTAE Ops) jettae {__version__} (Python {python})"


def _version_option(value: bool) -> None:
    if value:
        ensure_utf8_streams()
        _echo(_version_text())
        raise typer.Exit(EXIT_OK)


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option(
            "--version", "-V", is_eager=True, callback=_version_option, help="버전을 보여 주고 끝냄"
        ),
    ] = False,
) -> None:
    ensure_utf8_streams()


@app.command(help="버전을 보여 줍니다.")
def version() -> None:
    _echo(_version_text())


# --------------------------------------------------------------------------- serve, demo


def _check_port(port: int) -> None:
    if not 1 <= port <= 65535:
        raise CliError(f"--port 값은 1~65535 사이여야 합니다: {port}")


def _require_api() -> None:
    """Fail with a Korean message, not uvicorn's traceback, when the API module is not installed."""
    try:
        spec = importlib.util.find_spec(API_MODULE)
    except ModuleNotFoundError:
        spec = None
    if spec is None:
        raise CliError(f"API 모듈({API_MODULE})을 찾을 수 없습니다. 제때옵스를 다시 설치해 주세요.")


def _check_auth(settings: Settings) -> None:
    """Build the server's ``Authenticator`` once, here, before anything is written or bound.

    The app factory builds it again inside uvicorn, where a bad ``JETTAE_API_TOKENS`` (a short
    token, a misspelt role, public mode without a token) would only surface after ``demo`` had
    bootstrapped the database — and a container restarting on that error would write a new demo
    report on every attempt. Its ``ValueError`` carries a Korean message, which :func:`_errors`
    shows with exit code 2. The module is bound by name (see the module docstring).
    """
    security = importlib.import_module(API_SECURITY_MODULE)
    security.Authenticator(settings.auth_mode, settings.api_tokens)


#: ``bind`` failures that mean "someone else has this port" (Windows reports a port held by a
#: dual-stack or exclusive socket, or reserved by the OS, as access denied; its socket errors
#: carry the ``WSAE*`` numbers).
_PORT_TAKEN_ERRNOS: Final = frozenset(
    {
        errno.EADDRINUSE,
        errno.EACCES,
        getattr(errno, "WSAEADDRINUSE", errno.EADDRINUSE),
        getattr(errno, "WSAEACCES", errno.EACCES),
    }
)


@dataclass(frozen=True, slots=True)
class _ListenAddress:
    family: socket.AddressFamily
    address: tuple[Any, ...]

    @property
    def is_loopback(self) -> bool:
        try:
            return ipaddress.ip_address(str(self.address[0]).split("%", 1)[0]).is_loopback
        except ValueError:
            return False


def _listen_addresses(host: str, port: int) -> list[_ListenAddress]:
    """Every address the server will listen on: asyncio (under uvicorn) binds *each* address
    ``host`` resolves to — ``localhost`` is ``::1`` and ``127.0.0.1``, and an empty host means
    every interface (the wildcard addresses), not this machine's name."""
    try:
        infos = socket.getaddrinfo(host or None, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE)
    except socket.gaierror as exc:
        raise CliError(f"호스트 주소를 해석할 수 없습니다: {host} ({exc.strerror})") from None
    found: list[_ListenAddress] = []
    for family, _kind, _proto, _canonical, address in infos:
        candidate = _ListenAddress(family, tuple(address))
        if candidate not in found:
            found.append(candidate)
    return found


def _neighbours(listen: _ListenAddress) -> list[_ListenAddress]:
    """The wildcard and loopback addresses of ``listen``'s family on the same port (see
    :func:`_ensure_port_free`), without ``listen`` itself."""
    port = listen.address[1]
    candidates: list[tuple[Any, ...]]
    if listen.family == socket.AF_INET6:
        candidates = [("::", port, 0, 0), ("::1", port, 0, 0)]
    else:
        candidates = [("0.0.0.0", port), ("127.0.0.1", port)]
    return [
        neighbour
        for neighbour in (_ListenAddress(listen.family, address) for address in candidates)
        if neighbour != listen
    ]


def _bind_once(listen: _ListenAddress) -> OSError | None:
    try:
        probe = socket.socket(listen.family, socket.SOCK_STREAM)
    except OSError:
        return None
    with probe:
        if os.name != "nt":
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if listen.family == socket.AF_INET6 and hasattr(socket, "IPV6_V6ONLY"):
            probe.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        try:
            probe.bind(listen.address)
        except OSError as exc:
            return exc
    return None


def _bind_error(listen: _ListenAddress) -> OSError | None:
    """Bind and release ``listen`` the way the server will; the error, or ``None`` if it is free.

    ``SO_REUSEADDR`` is set on POSIX (a port in ``TIME_WAIT`` after a restart is not busy) but not
    on Windows, where it would let the bind steal a port in use; IPv6 is bound IPv6-only, as
    asyncio does. An address family this host cannot open is skipped, as asyncio skips it. A
    "taken" answer is final; any other failure is retried briefly, because the sandboxed hosts
    that refuse a loopback socket now and then (``WinError 10014``, see :func:`_run`) would
    otherwise turn a free port into an error.
    """
    error: OSError | None = None
    for attempt in range(1, _PROBE_ATTEMPTS + 1):
        error = _bind_once(listen)
        if error is None or error.errno in _PORT_TAKEN_ERRNOS:
            return error
        time.sleep(_LOOP_BACKOFF_SECONDS * attempt)
    return error


def _ensure_port_free(host: str, port: int) -> None:
    """Refuse, in Korean, a port another program is using — before uvicorn tries to bind it.

    Binding the requested address is not enough. Windows (and BSD with ``SO_REUSEADDR``) lets a
    socket bind ``127.0.0.1:P`` while another program listens on ``0.0.0.0:P`` — the new server
    then silently takes over every local connection to that program (a developer's other app on
    the same port would stop answering on localhost) — and ``0.0.0.0:P`` while another program
    holds ``127.0.0.1:P``, which leaves the printed ``http://127.0.0.1:P/`` pointing at the other
    program. So the wildcard and the loopback address of each family are probed too. Only the
    requested address may fail for another reason (that is a real error); for the neighbours
    only "taken" counts, so a host without IPv6 loopback can still serve on ``::``. This only
    binds and releases; nothing is ever sent to the other program.
    """
    busy = f"{host}:{port} 포트를 다른 프로그램이 쓰고 있습니다. --port로 다른 포트를 지정하세요."
    for listen in _listen_addresses(host, port):
        error = _bind_error(listen)
        if error is not None:
            if error.errno in _PORT_TAKEN_ERRNOS:
                raise CliError(busy)
            raise CliError(f"{host}:{port}에서 서버를 열 수 없습니다: {error.strerror or error}")
        for neighbour in _neighbours(listen):
            error = _bind_error(neighbour)
            if error is not None and error.errno in _PORT_TAKEN_ERRNOS:
                raise CliError(busy)


def _warn_if_exposed(host: str, port: int, settings: Settings) -> None:
    """Dev authentication trusts the ``X-Jettae-Role`` header, so anyone who can reach the server
    can claim ``lead`` and approve graph changes. That is fine on loopback (the developer's own
    machine) and dangerous anywhere else, so a reachable address gets a Korean warning."""
    if settings.auth_mode != "dev":
        return
    if all(listen.is_loopback for listen in _listen_addresses(host, port)):
        return
    shown = host or "모든 네트워크 주소"
    _stderr(
        f"주의: 개발용 인증 모드(JETTAE_AUTH_MODE=dev)로 {shown}에 엽니다. 이 주소로 접속할 수 있는 "
        "누구나 X-Jettae-Role 헤더만으로 승인 권한(lead)을 주장할 수 있습니다. 다른 컴퓨터에 열려면 "
        "JETTAE_AUTH_MODE=token(공개 웹사이트는 public)과 JETTAE_API_TOKENS를 설정하세요.",
        style="bold yellow",
    )


def _browser_host(host: str) -> str:
    if host in ("0.0.0.0", ""):
        return "127.0.0.1"
    if host == "::":
        return "::1"
    return host


def _url(host: str, port: int) -> str:
    shown = _browser_host(host)
    return f"http://[{shown}]:{port}/" if ":" in shown else f"http://{shown}:{port}/"


def _open_browser_when_ready(host: str, port: int) -> None:
    """Open the web UI once the server accepts connections (polled in a daemon thread, so a
    server that never comes up leaves no browser tab and blocks nothing)."""
    target = _browser_host(host)
    url = _url(host, port)

    def wait_and_open() -> None:
        deadline = time.monotonic() + _BROWSER_WAIT_SECONDS
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((target, port), timeout=0.5):
                    webbrowser.open(url)
                    return
            except OSError:
                time.sleep(0.25)

    threading.Thread(target=wait_and_open, name="jettae-open-browser", daemon=True).start()


def _serve(host: str, port: int, *, reload: bool) -> None:
    """Run uvicorn on the API factory.

    uvicorn's own access log is off: it prints each visitor's address (behind a proxy with
    ``FORWARDED_ALLOW_IPS``, the real client IP) and the full query string. The API writes its own
    structured access log without either (:mod:`jettae.api.middleware`).
    """
    import uvicorn

    from jettae.cli.loops import UVICORN_LOOP

    _stderr(f"제때옵스 API 서버: {_url(host, port)}  (끝내려면 Ctrl+C)")
    uvicorn.run(
        API_APP, factory=True, host=host, port=port, reload=reload, loop=UVICORN_LOOP, access_log=False
    )


@app.command(help="API 서버와 웹 화면을 띄웁니다(uvicorn).")
def serve(
    *,
    host: Annotated[str, typer.Option("--host", help="바인딩할 주소")] = DEFAULT_HOST,
    port: Annotated[int, typer.Option("--port", help="포트")] = DEFAULT_PORT,
    reload: Annotated[bool, typer.Option("--reload", help="코드가 바뀌면 다시 시작(개발용)")] = False,
) -> None:
    with _errors():
        # The server reads the same JETTAE_* settings in its factory; a bad value is reported here
        # in Korean rather than as a traceback from inside uvicorn.
        settings = _settings()
        _check_port(port)
        _require_api()
        _check_auth(settings)
        _ensure_port_free(host, port)
        _warn_if_exposed(host, port, settings)
    _serve(host, port, reload=reload)


def _db_reason(exc: BaseException) -> str:
    """Why a database could not be used, in one line: a missing module (usually the DB driver)
    says how to install one; any other error keeps its first line (for SQLAlchemy the rest is
    the SQL and a documentation link)."""
    if isinstance(exc, ImportError):
        return (
            f"필요한 모듈을 불러올 수 없습니다({exc.name or exc}). DB 드라이버라면 설치하세요"
            '(예: PostgreSQL은 pip install "jettae-ops[postgres]").'
        )
    lines = str(exc).strip().splitlines()
    return lines[0] if lines else type(exc).__name__


def _bootstrap_demo(settings: Settings, mode: LlmMode) -> tuple[str, DiagnosisReport]:
    """Load the 별빛상점 bundle into the configured database and diagnose it; (workspace id, report).

    A database that cannot be opened or written (no driver, a file that is not a database, a
    server that does not answer) is the user's to fix: a Korean one-line error with the URL's
    password masked, not a traceback reported as a defect of 제때옵스.
    """
    from sqlalchemy.exc import SQLAlchemyError

    shown = _masked_url(settings.database_url)
    try:
        store = open_store(settings.database_url)
    except (SQLAlchemyError, ImportError) as exc:
        raise CliError(f"데이터베이스를 열 수 없습니다({shown}): {_db_reason(exc)}") from None
    try:
        service = JettaeService(store, settings)
        report = _run(service.bootstrap_demo(llm_mode=mode))
    except SQLAlchemyError as exc:
        raise CliError(f"데모를 데이터베이스에 저장하지 못했습니다({shown}): {_db_reason(exc)}") from None
    finally:
        if isinstance(store, SqlStore):
            store.close()
    return report.workspace_id, report


@app.command(help="별빛상점(가상 회사) 데모를 설정된 DB에 만들고 서버를 띄웁니다.")
def demo(
    *,
    host: Annotated[str, typer.Option("--host", help="바인딩할 주소")] = DEFAULT_HOST,
    port: Annotated[int, typer.Option("--port", help="포트")] = DEFAULT_PORT,
    open_browser: Annotated[bool, typer.Option("--open", help="서버가 뜨면 브라우저로 엶")] = False,
    llm: Annotated[
        str | None,
        typer.Option("--llm", metavar="off|live|replay", help="데모 진단의 LLM 모드 (기본: JETTAE_LLM_MODE)"),
    ] = None,
) -> None:
    with _errors():
        settings = _settings()
        mode = _llm_mode(llm, settings)
        _check_port(port)
        _require_api()
        _check_auth(settings)  # before the bootstrap below writes anything
        _ensure_port_free(host, port)
        _warn_if_exposed(host, port, settings)
        demo_bundle_path()  # a missing example bundle is reported before anything is written
        if settings.database_url == MEMORY_URL:
            _stderr(
                "메모리 저장소(memory://)는 서버 프로세스와 공유되지 않아 데모를 미리 만들 수 없습니다. "
                "웹 화면의 예제 불러오기(POST /api/v1/demo/dmc-shop)를 쓰세요.",
                style="yellow",
            )
        else:
            workspace_id, report = _bootstrap_demo(settings, mode)
            _stderr(
                f"데모 워크스페이스 {workspace_id}: 준비도 {report.readiness.overall:.1f}/100 · "
                f"끊김 {len(report.findings)}건 · 판정 경로 LLM {report.llm.judgement_path_calls}회"
            )
    if open_browser:
        _open_browser_when_ready(host, port)
    _serve(host, port, reload=False)


# --------------------------------------------------------------------------- db


def _redacted_userinfo(url: str) -> str:
    """A URL SQLAlchemy could not parse, with everything between ``//`` (or the start) and the
    last ``@`` masked: a typo in the scheme must not print the password into a CI log, and an
    unparseable URL gives no reliable way to tell where the password ends (a raw ``/`` or ``@``
    in it), so the whole credentials part goes."""
    at = url.rfind("@")
    if at < 0:
        return url
    start = url.find("//")
    start = start + 2 if 0 <= start < at else 0
    return url[:start] + "***" + url[at:]


def _masked_url(url: str) -> str:
    """``url`` as it may be printed: the password masked (``user:***@host``)."""
    from sqlalchemy.engine import make_url
    from sqlalchemy.exc import ArgumentError

    try:
        parsed = make_url(url)
    except ArgumentError:
        raise CliError(f"데이터베이스 URL 형식이 올바르지 않습니다: {_redacted_userinfo(url)}") from None
    # Rendering re-quotes the database part (a SQLite path), so only URLs with a secret are rendered.
    return url if parsed.password is None else parsed.render_as_string(hide_password=True)


@db_app.command("upgrade", help="데이터베이스 스키마를 Alembic 마이그레이션으로 올립니다.")
def db_upgrade(
    *,
    url: Annotated[
        str | None, typer.Option("--url", help="대상 DB URL (기본: JETTAE_DATABASE_URL)", show_default=False)
    ] = None,
    revision: Annotated[str, typer.Option("--revision", help="목표 리비전")] = "head",
) -> None:
    from alembic.util import CommandError
    from sqlalchemy.exc import SQLAlchemyError

    with _errors():
        target = url or _settings().database_url
        if target == MEMORY_URL:
            raise CliError("메모리 저장소(memory://)는 스키마가 없어 마이그레이션할 수 없습니다.")
        shown = _masked_url(target)
        try:
            upgrade_schema(target, revision)
        except (CommandError, SQLAlchemyError, ImportError) as exc:  # ImportError: driver/migration
            # Exit 2, not 1: a migration that cannot run is not a gate that failed.
            raise CliError(f"스키마를 올리지 못했습니다({shown}): {_db_reason(exc)}") from None
        _stderr(f"데이터베이스 스키마를 {revision}까지 올렸습니다: {shown}")


# --------------------------------------------------------------------------- eval


def _evalkit() -> ModuleType:
    """The evaluation kit, bound by name when an ``eval`` command runs (see the module docstring)."""
    return importlib.import_module(EVALKIT_MODULE)


def _read_cases(kit: ModuleType, path: Path) -> list[Any]:
    if not path.is_file():
        raise CliError(f"평가 사례 파일이 없습니다: {_shown(path)} (먼저 jettae eval build로 만드세요)")
    try:
        cases: list[Any] = kit.read_cases_jsonl(path)
    except (ValidationError, UnicodeDecodeError):
        raise CliError(f"평가 사례 파일 형식이 올바르지 않습니다: {_shown(path)}") from None
    if not cases:
        raise CliError(f"평가 사례 파일이 비어 있습니다: {_shown(path)}")
    return cases


def _read_manifest(kit: ModuleType, path: Path) -> Any:
    if not path.is_file():
        raise CliError(f"봉인 파일이 없습니다: {_shown(path)} (먼저 jettae eval seal로 봉인하세요)")
    try:
        return kit.read_manifest(path)
    except (ValidationError, UnicodeDecodeError):
        raise CliError(f"봉인 파일 형식이 올바르지 않습니다: {_shown(path)}") from None


def _conditions(kit: ModuleType, text: str) -> tuple[Any, ...]:
    """``"A,B,O"`` (or full names such as ``B_rule_graph``, any case) → evalkit conditions."""
    known: dict[str, str] = {}
    for condition in kit.Condition:
        value = str(condition.value)
        known[value.casefold()] = value
        known[value.split("_", 1)[0].casefold()] = value
    chosen: list[str] = []
    for token in (t.strip() for t in text.split(",")):
        if not token:
            continue
        match = known.get(token.casefold())
        if match is None:
            allowed = ", ".join(sorted({v.split("_", 1)[0] for v in known.values()}))
            raise CliError(f"알 수 없는 평가 조건입니다: {token} (가능한 값: {allowed})")
        chosen.append(match)
    if not chosen:
        raise CliError("평가 조건을 하나 이상 지정하세요 (예: --conditions A,B,O)")
    return tuple(kit.parse_conditions(chosen))


def _today() -> str:
    return datetime.now(KST).date().isoformat()


def _progress(total_label: str) -> Callable[[int, int], None] | None:
    """In-place progress on an interactive stderr; nothing in logs (a CI log is not a terminal)."""
    if not sys.stderr.isatty():
        return None

    def show(done: int, total: int) -> None:
        sys.stderr.write(f"\r{total_label} {done:,}/{total:,}")
        sys.stderr.flush()
        if done >= total:
            sys.stderr.write("\n")

    return show


def _percent(rate: Any) -> str:
    value = rate.rate
    return "—" if value is None else f"{100 * value:.1f}% ({rate.hits}/{rate.n})"


@eval_app.command("build", help="결함을 심은 평가 사례(JSONL)를 만듭니다. 정답은 생성 과정에서 정해집니다.")
def eval_build(
    *,
    out: Annotated[Path, typer.Option("--out", "-o", help="사례 파일")] = DEFAULT_CASES,
    orgs: Annotated[int, typer.Option("--orgs", help="가상 조직 수 (작은 표본은 1~2)")] = 20,
    seed: Annotated[
        int | None, typer.Option("--seed", help="생성 시드 (기본: 평가 키트 기본값)", show_default=False)
    ] = None,
) -> None:
    with _errors():
        if orgs < 1:
            raise CliError(f"--orgs 값은 1 이상이어야 합니다: {orgs}")
        if out.is_dir():  # checked before the (minutes-long) generation, not after it
            raise CliError(f"출력 경로가 폴더입니다. 파일 이름까지 적어 주세요: {_shown(out)}")
        kit = _evalkit()
        used = kit.DEFAULT_SEED if seed is None else seed
        cases: list[Any] = kit.build_cases(kit.generate_gold_orgs(orgs, used), seed=used)
        out.parent.mkdir(parents=True, exist_ok=True)
        kit.write_cases_jsonl(cases, out)
        held_out = sum(1 for c in cases if c.held_out)
        clean = sum(1 for c in cases if c.is_clean)
        _show(
            [
                Text(f"평가 사례 {len(cases):,}건을 만들었습니다: {_shown(out)}", style="bold"),
                key_value_table(
                    [
                        ("조직", f"{orgs}곳 (시드 {used})"),
                        ("활성 결함", f"{len(cases) - held_out - clean:,}건"),
                        ("봉인 결함(held-out)", f"{held_out:,}건"),
                        ("깨끗한 대조군", f"{clean:,}건"),
                    ]
                ),
                Text(f"다음 단계: jettae eval seal --cases {_shown(out)}", style="dim"),
            ]
        )


@eval_app.command(
    "seal", help="사례 파일을 봉인합니다(사례별 SHA-256 + 머클 루트). --check는 봉인과 대조만 합니다."
)
def eval_seal(
    *,
    cases: Annotated[Path, typer.Option("--cases", help="사례 파일")] = DEFAULT_CASES,
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="봉인 파일 (기본: 사례 파일 옆 manifest.json)", show_default=False),
    ] = None,
    check: Annotated[
        bool, typer.Option("--check", help="새로 봉인하지 않고 기존 봉인과 일치하는지만 확인")
    ] = False,
) -> None:
    with _errors():
        kit = _evalkit()
        loaded = _read_cases(kit, cases)
        target = out if out is not None else cases.with_name("manifest.json")
        if check:
            manifest = _read_manifest(kit, target)
            if not kit.verify(loaded, manifest):
                raise CliError(
                    f"봉인 불일치: {_shown(cases)}가 봉인({_shown(target)})과 다릅니다. "
                    "사례가 바뀌었거나 다른 생성기 버전으로 만든 것입니다.",
                    EXIT_GATE,
                )
            _stderr(f"봉인 일치: 사례 {len(loaded):,}건 · 머클 루트 {manifest.merkle_root}", style="green")
            return
        manifest = kit.build_manifest(loaded)
        if target.is_dir():
            raise CliError(f"출력 경로가 폴더입니다. 파일 이름까지 적어 주세요: {_shown(target)}")
        target.parent.mkdir(parents=True, exist_ok=True)
        kit.write_manifest(manifest, target)
        _show(
            [
                Text(f"봉인했습니다: {_shown(target)}", style="bold"),
                key_value_table(
                    [
                        ("사례", f"{manifest.n_cases:,}건 ({manifest.generator_version})"),
                        ("머클 루트", str(manifest.merkle_root)),
                        ("봉인 결함 루트", str(manifest.held_out_merkle_root)),
                    ]
                ),
            ]
        )


@dataclass(frozen=True, slots=True)
class _Corpus:
    cases: list[Any]
    manifest: Any
    seed: int | None
    #: the manifest was computed here and belongs in the results folder (in-memory corpus)
    write_manifest: bool
    origin: str


def _corpus(
    kit: ModuleType, *, cases: Path | None, manifest: Path | None, orgs: int, seed: int | None
) -> _Corpus:
    """The corpus a run draws from: a sealed file (verified) or one built in memory."""
    if cases is None:
        if manifest is not None:
            raise CliError("--manifest는 --cases와 함께 쓸 때만 의미가 있습니다.")
        used = kit.DEFAULT_SEED if seed is None else seed
        built: list[Any] = kit.build_cases(kit.generate_gold_orgs(orgs, used), seed=used)
        return _Corpus(built, kit.build_manifest(built), used, True, f"즉석 생성(조직 {orgs}곳, 시드 {used})")
    if seed is not None:
        raise CliError("--seed는 사례를 즉석 생성할 때만 씁니다(--cases 파일에는 이미 정해져 있습니다).")
    loaded = _read_cases(kit, cases)
    seal_path = manifest if manifest is not None else cases.with_name("manifest.json")
    if manifest is None and not seal_path.is_file():
        _stderr(
            "봉인 파일이 없어 이 사례들로 봉인값을 즉석 계산합니다. "
            f"공개 전에는 jettae eval seal --cases {_shown(cases)}로 봉인하세요.",
            style="yellow",
        )
        return _Corpus(loaded, kit.build_manifest(loaded), None, False, _shown(cases))
    sealed = _read_manifest(kit, seal_path)
    if not kit.verify(loaded, sealed):
        raise CliError(f"봉인 불일치: {_shown(cases)}가 봉인({_shown(seal_path)})과 다릅니다.", EXIT_GATE)
    return _Corpus(loaded, sealed, None, False, f"{_shown(cases)} (봉인 확인됨)")


def _check_counts(**counts: int | None) -> None:
    for name, value in counts.items():
        if value is not None and value < 1:
            raise CliError(f"--{name} 값은 1 이상이어야 합니다: {value}")


def _results_dir(out: Path | None, *, partial: bool) -> Path:
    """Where a run writes; a smoke or sampled run never lands on the published numbers by default."""
    target = out if out is not None else DEFAULT_SAMPLE_DIR if partial else DEFAULT_EVAL_DIR
    if target.is_file():
        raise CliError(f"--out은 결과 폴더여야 합니다: {_shown(target)}")
    return target


def _paid_client(
    kit: ModuleType,
    cases: Sequence[Any],
    arms: Sequence[Any],
    settings: Settings,
    *,
    model: str,
    confirmed: bool,
) -> Any:
    """The live model for the LLM arms — only after the cost estimate was shown and confirmed.

    The estimate is computed offline from the very requests the run would send, so it is printed
    even when the run is then refused; a key is checked last, after the user has seen the cost.
    """
    estimate = kit.estimate_llm_cost(cases, arms, model=model, echo=None)
    _stderr(str(estimate.render()))
    if not confirmed:
        raise CliError("LLM 조건은 유료 호출입니다. 위 추정치를 확인한 뒤 --yes를 붙여 다시 실행하세요.")
    if settings.llm_provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise CliError("ANTHROPIC_API_KEY가 없어 LLM 조건을 실행할 수 없습니다.")
    return build_llm(settings, LlmMode.LIVE)


def _injection_bundle(bundle: Path | None) -> Path:
    root = bundle if bundle is not None else demo_bundle_path()
    if not root.is_dir():
        raise CliError(f"인젝션 자기 시험 묶음 폴더가 없습니다: {_shown(root)}")
    return root


@eval_app.command(
    "run",
    help=(
        "평가 조건을 실행하고 metrics.json·report.md·results.jsonl을 씁니다. A=키워드, B=규칙+그래프, "
        "O=정답 그래프(오프라인), C·D·DR=LLM(유료: 비용 추정을 보여 준 뒤 --yes가 있어야 실행)."
    ),
)
def eval_run(
    *,
    conditions: Annotated[
        str, typer.Option("--conditions", help="쉼표로 구분한 조건 (A,B,C,D,DR,O 또는 전체 이름)")
    ] = DEFAULT_CONDITIONS,
    cases: Annotated[
        Path | None,
        typer.Option("--cases", help="사례 파일 (없으면 --orgs만큼 즉석 생성)", show_default=False),
    ] = None,
    manifest: Annotated[
        Path | None,
        typer.Option("--manifest", help="봉인 파일 (기본: 사례 파일 옆 manifest.json)", show_default=False),
    ] = None,
    orgs: Annotated[int, typer.Option("--orgs", help="즉석 생성할 조직 수")] = 20,
    seed: Annotated[int | None, typer.Option("--seed", help="즉석 생성 시드", show_default=False)] = None,
    smoke: Annotated[
        bool, typer.Option("--smoke", help="빠른 점검: 조직 1곳 즉석 생성, 벤치마크 생략")
    ] = False,
    sample: Annotated[
        float | None,
        typer.Option(
            "--sample", help="사례 id 해시로 고르는 결정적 표본 비율 (0 초과 1 이하)", show_default=False
        ),
    ] = None,
    workers: Annotated[
        int | None,
        typer.Option("--workers", help="오프라인 조건 프로세스 수 (기본: CPU 수 - 2)", show_default=False),
    ] = None,
    concurrency: Annotated[
        int | None,
        typer.Option(
            "--concurrency", help="동시 LLM 요청 수 (기본: JETTAE_LLM_CONCURRENCY)", show_default=False
        ),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            "-o",
            help="결과 폴더 (기본: docs/eval, --smoke·--sample 실행은 docs/eval/sample)",
            show_default=False,
        ),
    ] = None,
    generated_at: Annotated[
        str | None,
        typer.Option(
            "--generated-at", help="보고서에 적을 생성 일자 (기본: 오늘, 한국 시간)", show_default=False
        ),
    ] = None,
    benchmarks: Annotated[
        bool, typer.Option("--benchmarks/--no-benchmarks", help="영향 범위 성능·인젝션 자기 시험 포함")
    ] = True,
    bundle: Annotated[
        Path | None,
        typer.Option("--bundle", help="인젝션 자기 시험에 쓸 묶음 (기본: 별빛상점 예제)", show_default=False),
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="LLM 조건의 유료 호출을 승인")] = False,
) -> None:
    with _errors():
        if smoke:
            if cases is not None:
                raise CliError("--smoke는 사례를 즉석 생성합니다. --cases와 함께 쓸 수 없습니다.")
            orgs, benchmarks = 1, False
        _check_counts(orgs=orgs, workers=workers, concurrency=concurrency)
        if sample is not None and not 0 < sample <= 1:
            raise CliError(f"--sample 값은 0 초과 1 이하여야 합니다: {sample:g}")
        results_dir = _results_dir(out, partial=smoke or sample is not None)
        settings = _settings()
        kit = _evalkit()
        chosen = _conditions(kit, conditions)
        corpus = _corpus(kit, cases=cases, manifest=manifest, orgs=orgs, seed=seed)
        selected = kit.reproducibility_sample(corpus.cases, sample) if sample is not None else corpus.cases
        llm_arms = [c for c in chosen if c in kit.LLM_CONDITIONS]
        model = settings.openai_model if settings.llm_provider == "openai" else settings.anthropic_model
        client = (
            _paid_client(kit, selected, llm_arms, settings, model=model, confirmed=yes) if llm_arms else None
        )
        injection_bundle = _injection_bundle(bundle) if benchmarks else None
        date = generated_at or _today()
        results = kit.run_eval(
            selected,
            chosen,
            llm=client,
            workers=workers,
            concurrency=settings.llm_concurrency if concurrency is None else concurrency,
            progress=_progress("평가 진행"),
        )
        reproducibility = (
            kit.check_reproducibility(results) if kit.Condition.B_RULE_GRAPH in results.conditions else None
        )
        results_dir.mkdir(parents=True, exist_ok=True)
        if corpus.write_manifest:
            kit.write_manifest(corpus.manifest, results_dir / "manifest.json")
        metrics = kit.write_report(
            results,
            results_dir,
            generated_at=date,
            manifest=corpus.manifest,
            seed=corpus.seed,
            reproducibility=reproducibility,
            impact=kit.impact_benchmark() if benchmarks else None,
            injection=kit.injection_selftest(injection_bundle) if injection_bundle is not None else None,
            cost_estimate=kit.estimate_llm_cost(selected, kit.LLM_CONDITIONS, model=model, echo=None),
        )
        problems = _eval_problems(kit, metrics)
        _show(
            _eval_summary(metrics, corpus=corpus, selected=len(selected), out=results_dir, problems=problems)
        )
        if problems:
            raise typer.Exit(EXIT_GATE)


def _eval_problems(kit: ModuleType, metrics: Any) -> list[str]:
    """What makes an offline evaluation untrustworthy: a run that does not reproduce byte for byte
    (P6), or an offline arm that crashed on a case (a defect, not a measurement)."""
    problems: list[str] = []
    reproducibility = metrics.reproducibility
    if reproducibility is not None and reproducibility.mismatched:
        problems.append(
            f"재현성 실패: 표본 {reproducibility.sample_size}건 중 {len(reproducibility.mismatched)}건이 "
            "새 프로세스에서 다른 결과를 냈습니다."
        )
    offline = {str(c.value) for c in kit.OFFLINE_CONDITIONS}
    for name, summary in sorted(metrics.conditions.items()):
        if name in offline and summary.errors:
            problems.append(f"{name}: 오프라인 조건에서 사례 {summary.errors}건이 오류로 끝났습니다.")
    return problems


def _eval_summary(
    metrics: Any, *, corpus: _Corpus, selected: int, out: Path, problems: Sequence[str]
) -> list[RenderableType]:
    plain = _plain()
    rows = [
        (
            str(name),
            _percent(summary.detection_active),
            _percent(summary.detection_held_out),
            _percent(summary.clean_fp),
            f"{summary.errors}",
        )
        for name, summary in metrics.conditions.items()
    ]
    reproducibility = metrics.reproducibility
    repro = (
        "측정 안 함(B 조건 없음)"
        if reproducibility is None
        else f"{reproducibility.identical}/{reproducibility.sample_size}건 동일 (새 프로세스 재실행)"
    )
    blocks: list[RenderableType] = [
        Text(f"평가 결과 ({metrics.generated_at})", style="bold"),
        key_value_table(
            [
                ("코퍼스", corpus.origin),
                ("실행 사례", f"{selected:,}건"),
                ("재현성", repro),
                ("결과", f"{_shown(out)}/metrics.json, report.md, results.jsonl"),
            ]
        ),
        data_table(("조건", "탐지율(활성)", "탐지율(봉인)", "오탐률(대조군)", "오류"), rows, plain=plain),
    ]
    blocks += [Text(problem, style="bold red") for problem in problems]
    return blocks


@eval_app.command(
    "report", help="metrics.json에서 report.md를 다시 씁니다(수치는 그대로, 문장은 수치에서 생성)."
)
def eval_report(
    *,
    metrics: Annotated[Path, typer.Option("--metrics", help="metrics.json")] = DEFAULT_EVAL_DIR
    / "metrics.json",
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="보고서 파일 (기본: metrics.json 옆 report.md)", show_default=False),
    ] = None,
) -> None:
    with _errors():
        if not metrics.is_file():
            raise CliError(f"metrics.json이 없습니다: {_shown(metrics)} (먼저 jettae eval run을 실행하세요)")
        kit = _evalkit()
        try:
            parsed = kit.EvalMetrics.model_validate_json(metrics.read_bytes().decode("utf-8"))
        except (ValidationError, UnicodeDecodeError):
            raise CliError(f"metrics.json 형식이 올바르지 않습니다: {_shown(metrics)}") from None
        target = out if out is not None else metrics.with_name("report.md")
        write_text(target, str(kit.render_report(parsed)))
        _stderr(f"보고서를 썼습니다: {_shown(target)}")


@eval_app.command(
    "injection",
    help="프롬프트 인젝션 자기 시험: 런북에 공격 문장을 심어도 그래프·끊김·점수가 그대로인지 확인(LLM 끔).",
)
def eval_injection(
    *,
    bundle: Annotated[
        Path | None, typer.Option("--bundle", help="시험할 묶음 (기본: 별빛상점 예제)", show_default=False)
    ] = None,
    n: Annotated[int, typer.Option("--n", help="공격 문장 수")] = 30,
    document: Annotated[
        str | None,
        typer.Option(
            "--document", help="공격을 심을 런북(묶음 기준 경로, 기본: 첫 텍스트 런북)", show_default=False
        ),
    ] = None,
    output_format: Annotated[str, typer.Option("--format", "-f", metavar="text|json", help="출력 형식")] = (
        DataFormat.TEXT.value
    ),
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="결과를 이 파일에 저장", show_default=False)
    ] = None,
) -> None:
    with _errors():
        fmt = _choice(DataFormat, output_format, "--format")
        if n < 1:
            raise CliError(f"--n 값은 1 이상이어야 합니다: {n}")
        root = bundle if bundle is not None else demo_bundle_path()
        if not root.is_dir():
            raise CliError(f"묶음 폴더를 찾을 수 없습니다: {_shown(root)}")
        kit = _evalkit()
        result = kit.injection_selftest(root, n, document=document)
        if fmt is DataFormat.JSON:
            payload = to_json(result.model_dump(mode="json"))
            if out is None:
                _echo(payload)
            else:
                write_text(out, payload)
        else:
            if out is not None:
                write_text(out, render_plain(_injection_view(result, plain=True)))
            _show(_injection_view(result, plain=_plain()))
        if result.unauthorised_changes:
            raise CliError(
                f"인젝션 자기 시험 실패: {result.unauthorised_changes}건이 그래프·끊김·점수를 바꿨습니다.",
                EXIT_GATE,
            )


def _injection_view(result: Any, *, plain: bool) -> list[RenderableType]:
    rows = [
        (
            str(case.case_id),
            str(case.kind),
            f"{case.line_no}",
            "예" if case.quarantined else "아니요",
            Text("변경됨", style="bold red") if case.unauthorised else Text("영향 없음", style="green"),
        )
        for case in result.cases
    ]
    verdict = (
        Text(f"무단 변경 {result.unauthorised_changes}건", style="bold red")
        if result.unauthorised_changes
        else Text("무단 변경 0건 — 문서 속 지시문이 판정에 영향을 주지 못했습니다", style="green")
    )
    return [
        Text(f"프롬프트 인젝션 자기 시험 — {result.bundle}", style="bold"),
        key_value_table(
            [
                ("대상 런북", str(result.document)),
                ("공격 문장", f"{result.n}건 (LLM {result.llm})"),
                ("격리", f"{result.quarantined}/{result.n}건"),
                ("기준 점수", f"{result.baseline_score:.1f}"),
                ("결과", verdict),
            ]
        ),
        data_table(("시험", "공격 유형", "줄", "격리", "판정 영향"), rows, plain=plain),
    ]


if __name__ == "__main__":  # `python -m jettae.cli.main` must run the command, never exit 0 silently
    app(prog_name="jettae")
