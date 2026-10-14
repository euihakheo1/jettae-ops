"""The deployment files (Dockerfile, render.yaml, constraints.txt, docs/deploy.md) agree with the code.

They are not executed by the test suite, so the facts they depend on are pinned here: the health
path the API really serves, the port variable the container honours, the settings names and values
the app accepts, the PostgreSQL URL a hosting provider injects, the pinned dependency set the image
installs, the commands the operator's guide tells a non-ops founder to run (including the check of
where the visitor's address comes from), and the limits and setting names the security docs state.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest
import yaml
from fastapi import FastAPI
from packaging.requirements import Requirement
from pydantic import ValidationError
from structlog.testing import capture_logs

from jettae.config import Settings
from jettae.store.sql import normalize_database_url
from test_api_support import new_app, open_client

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")
CONSTRAINTS = ROOT / "constraints.txt"
GUIDE = (ROOT / "docs" / "deploy.md").read_text(encoding="utf-8")
SECURITY = (ROOT / "docs" / "security.md").read_text(encoding="utf-8")
ARCHITECTURE = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
BLUEPRINT = (ROOT / "render.yaml").read_text(encoding="utf-8")
#: Every text that names settings or states a limit to an operator or a reviewer.
DOCS = {
    "docs/deploy.md": GUIDE,
    "docs/security.md": SECURITY,
    "docs/ARCHITECTURE.md": ARCHITECTURE,
    "README.md": README,
    "render.yaml": BLUEPRINT,
    "Dockerfile": DOCKERFILE,
}
#: The guide's address check: the endpoint, and the header carrying the visitor's address per field.
REQUEST_INFO = "/api/v1/admin/request-info"
_FIELD_HEADERS = {
    "cf_connecting_ip": "CF-Connecting-IP",
    "true_client_ip": "True-Client-IP",
    "x_real_ip": "X-Real-IP",
}
#: Documentation addresses (RFC 5737): the operator's own, a forged one, and a proxy hop.
_MINE, _FORGED, _HOP = "198.51.100.23", "203.0.113.7", "10.214.3.9"

#: ``name==version`` with an optional trailing ``# via …`` comment — nothing looser.
_PIN = re.compile(r"(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[0-9][A-Za-z0-9.+!-]*)(?:\s+#.*)?")
#: What a generated secret looks like (``secrets.token_urlsafe``): 32+ URL-safe characters mixing
#: lower case, upper case and digits — environment variable names and prose never match.
_URLSAFE = "[A-Za-z0-9_-]"
_TOKEN_LIKE = re.compile(
    rf"(?<!{_URLSAFE})(?={_URLSAFE}*[a-z])(?={_URLSAFE}*[A-Z])(?={_URLSAFE}*[0-9]){_URLSAFE}{{32,}}(?!{_URLSAFE})"
)
#: Packages ``uvicorn[standard]`` and ``psycopg[binary]`` add on Linux; the image needs them pinned too.
_EXTRA_PACKAGES = ("psycopg-binary", "uvloop", "httptools", "watchfiles", "websockets", "python-dotenv")
_LLM_SDKS = ("anthropic", "openai")


@pytest.fixture(scope="module")
def blueprint() -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load((ROOT / "render.yaml").read_text(encoding="utf-8"))
    return loaded


def _instructions(name: str) -> list[str]:
    """Every Dockerfile instruction ``name`` with its continuation lines joined."""
    joined = DOCKERFILE.replace("\\\n", " ")
    return [line for line in joined.splitlines() if line.startswith(f"{name} ")]


def _instruction(name: str) -> str:
    (line,) = _instructions(name)
    return line


def _canonical(name: str) -> str:
    """PEP 503 normalised project name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _pins() -> dict[str, str]:
    pins: dict[str, str] = {}
    for number, raw in enumerate(CONSTRAINTS.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _PIN.fullmatch(line)
        assert match, f"constraints.txt:{number} is not an exact 'name==version' pin: {raw!r}"
        name = _canonical(match["name"])
        assert name not in pins, f"constraints.txt:{number} pins {name} twice"
        pins[name] = match["version"]
    return pins


# --------------------------------------------------------------------------- container


def test_container_honours_port_and_upgrades_before_serving() -> None:
    command = json.loads(_instruction("CMD").removeprefix("CMD "))
    assert command == [
        "sh",
        "-c",
        'jettae db upgrade && exec jettae demo --host 0.0.0.0 --port "${PORT:-8765}"',
    ]
    healthcheck = _instruction("HEALTHCHECK")
    assert "http://127.0.0.1:${PORT:-8765}/healthz" in healthcheck
    assert re.search(r"^USER jettae$", DOCKERFILE, flags=re.MULTILINE)  # never root


def test_health_paths_are_served_at_the_root() -> None:
    with open_client(new_app()) as client:
        assert client.get("/healthz").json()["status"] == "ok"
        assert client.get("/api/v1/healthz").status_code == 404


def test_image_installs_the_pinned_set_without_llm_sdks() -> None:
    (install,) = [line for line in _instructions("RUN") if "pip install" in line]
    assert install == 'RUN pip install -c constraints.txt ".[postgres]"'
    copies = _instructions("COPY")
    assert any("constraints.txt" in line.split() for line in copies), "constraints.txt is not copied in"
    # the web stage installs exactly the committed lockfile
    assert "COPY web/package.json web/package-lock.json ./" in copies
    assert any(line.startswith("RUN npm ci ") for line in _instructions("RUN"))
    assert "npm install" not in DOCKERFILE
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").split()
    assert "constraints.txt" not in ignored
    assert not [pattern for pattern in ignored if "package-lock" in pattern]


def test_web_lockfile_matches_package_json() -> None:
    """``npm ci`` refuses a lockfile that disagrees with package.json — catch it before a deploy."""
    package = json.loads((ROOT / "web" / "package.json").read_text(encoding="utf-8"))
    lock = json.loads((ROOT / "web" / "package-lock.json").read_text(encoding="utf-8"))
    root = lock["packages"][""]
    for section in ("dependencies", "devDependencies"):
        assert root.get(section, {}) == package.get(section, {}), section


def test_constraints_pin_every_package_exactly() -> None:
    pins = _pins()
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    direct = [*pyproject["dependencies"], *pyproject["optional-dependencies"]["postgres"]]
    for spec in direct:
        requirement = Requirement(spec)
        name = _canonical(requirement.name)
        assert name in pins, f"{name} (pyproject.toml) is not pinned in constraints.txt"
        assert requirement.specifier.contains(pins[name], prereleases=True), (name, pins[name], spec)
    for name in _EXTRA_PACKAGES:
        assert name in pins, f"{name} is installed on Linux but not pinned in constraints.txt"
    # the public image runs with the LLM off: no SDKs (and none of their dependencies by name)
    assert not set(_LLM_SDKS) & set(pins)


def test_public_image_runs_without_the_llm_sdks() -> None:
    """The image installs ``.[postgres]`` only: the app factory, the routers and the demo diagnosis
    the container runs at start must work with neither ``anthropic`` nor ``openai`` importable."""
    script = (
        "import sys\n"
        "sys.modules['anthropic'] = None\n"
        "sys.modules['openai'] = None\n"
        "from jettae.api import create_app\n"
        "from jettae.cli.loops import resilient_event_loop\n"
        "from jettae.config import Settings\n"
        "from jettae.domain import LlmMode\n"
        "from jettae.orchestrator import JettaeService\n"
        "from jettae.store import MemoryStore\n"
        "token = 'deploy-test-token-' + 'x' * 32\n"
        "settings = Settings(_env_file=None, auth_mode='public', llm_mode='off', database_url='memory://',\n"
        "                    web_dist=None, api_tokens={token: {'user': 'heo', 'role': 'admin'}})\n"
        "assert create_app(settings).openapi()['paths']\n"
        "loop = resilient_event_loop()\n"
        "try:\n"
        "    report = loop.run_until_complete(\n"
        "        JettaeService(MemoryStore(), settings).bootstrap_demo(llm_mode=LlmMode.OFF))\n"
        "finally:\n"
        "    loop.close()\n"
        "assert report.findings\n"
        "loaded = [name for name in ('anthropic', 'openai') if sys.modules.get(name) is not None]\n"
        "assert not loaded, loaded\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        cwd=ROOT,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]


# --------------------------------------------------------------------------- blueprint


def test_blueprint_web_service(blueprint: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    (service,) = blueprint["services"]
    assert service["type"] == "web"
    assert service["name"] == "jettae-ops"
    assert service["runtime"] == "docker"
    assert service["region"] == "singapore"
    assert service["plan"] == "0.5c-512mb"
    assert service["healthCheckPath"] == "/healthz"
    env = {item["key"]: item for item in service["envVars"]}
    values = {key: item["value"] for key, item in env.items() if "value" in item}
    assert values == {
        "JETTAE_AUTH_MODE": "public",
        "JETTAE_LLM_MODE": "off",
        "JETTAE_LOG_JSON": "true",
        "PORT": "8765",
        "FORWARDED_ALLOW_IPS": "*",
        "JETTAE_CLIENT_IP_SOURCE": "header:CF-Connecting-IP",
        "JETTAE_DAILY_FREE_SCAN_LIMIT": "50",
        "JETTAE_FREE_SCANS_PER_CLIENT_PER_DAY": "5",
        "JETTAE_MAX_UPLOAD_MB": "2",
        "JETTAE_MAX_DOCUMENT_CHARS": "50000",
        "JETTAE_MAX_CONCURRENT_SCANS": "1",
        "JETTAE_EVENTS_PER_DAY": "20000",
        "JETTAE_FREE_SCAN_RETENTION_DAYS": "7",
        "JETTAE_SIGNUP_RETENTION_DAYS": "365",
    }
    assert env["JETTAE_API_TOKENS"] == {"key": "JETTAE_API_TOKENS", "sync": False}
    assert env["JETTAE_DATABASE_URL"]["fromDatabase"] == {"name": "jettae-db", "property": "connectionString"}
    # every JETTAE_* name is a real setting: unknown variables are silently ignored, so a typo
    # would leave the default (e.g. a 20 MB upload limit) on the public site
    ours = {key: value for key, value in values.items() if key.startswith("JETTAE_")}
    unknown = [key for key in ours if key.removeprefix("JETTAE_").lower() not in Settings.model_fields]
    assert not unknown, unknown
    # ... and the values are ones the app accepts, read the way Render passes them (environment)
    for key in [key for key in os.environ if key.upper().startswith("JETTAE_")]:
        monkeypatch.delenv(key)
    for key, value in ours.items():
        monkeypatch.setenv(key, value)
    settings = Settings(_env_file=None)
    assert (settings.auth_mode, settings.llm_mode.value, settings.log_json) == ("public", "off", True)
    assert settings.client_ip_source == "header:CF-Connecting-IP"
    assert settings.daily_free_scan_limit == 50
    assert settings.free_scans_per_client_per_day == 5
    assert settings.max_upload_mb == 2
    assert settings.max_document_chars == 50_000
    assert settings.max_concurrent_scans == 1
    assert settings.events_per_day == 20_000
    assert (settings.free_scan_retention_days, settings.signup_retention_days) == (7, 365)


def test_blueprint_points_to_the_address_check() -> None:
    """CF-Connecting-IP is an observation about Render's edge, not a documented contract: the
    blueprint says so where the value is set, and sends the operator to the guide's check."""
    comment = BLUEPRINT[: BLUEPRINT.index("- key: JETTAE_CLIENT_IP_SOURCE")].rsplit("- key:", 1)[1]
    for fragment in ("Cloudflare", "문서화되지 않음", REQUEST_INFO, "xff:<n>", "배포 직후 확인"):
        assert fragment in comment, fragment
    assert "## 4. 배포 직후 확인" in GUIDE
    assert "### 4-2." in GUIDE


def test_retention_matches_the_privacy_notice(blueprint: dict[str, Any]) -> None:
    """The deletion periods the server applies are the ones the visitor was promised."""
    (service,) = blueprint["services"]
    values = {item["key"]: item.get("value") for item in service["envVars"]}
    assert values["JETTAE_FREE_SCAN_RETENTION_DAYS"] == "7"
    assert values["JETTAE_SIGNUP_RETENTION_DAYS"] == "365"
    privacy = (ROOT / "web" / "src" / "pages" / "Privacy.tsx").read_text(encoding="utf-8")
    assert "7일 뒤 자동 삭제" in privacy
    assert "신청일로부터 1년" in privacy
    assert "싱가포르" in privacy  # the region both the service and the database run in
    assert service["region"] == "singapore"


def test_blueprint_database(blueprint: dict[str, Any]) -> None:
    (database,) = blueprint["databases"]
    # ipAllowList: [] blocks every connection from the internet; the web service in the same
    # region connects over the private network (fromDatabase connectionString is the internal URL)
    assert database == {"name": "jettae-db", "plan": "free", "region": "singapore", "ipAllowList": []}
    # Render's connectionString is a driver-less postgresql:// URL
    url = normalize_database_url("postgresql://jettae:pw@dpg-x-a.singapore-postgres.render.com/jettae")
    assert url.drivername == "postgresql+psycopg"


# --------------------------------------------------------------------------- operator's guide


def _guide_commands() -> list[tuple[str, str, dict[str, str]]]:
    """Each ``curl.exe`` line of the guide as (method, path, headers), with the PowerShell variables
    filled in the way the guide sets them (``$email`` from its ``EscapeDataString`` line)."""
    variables = {"$email": quote(_guide_email(), safe="")}
    parsed: list[tuple[str, str, dict[str, str]]] = []
    for line in GUIDE.splitlines():
        command = line.strip()
        if not command.startswith("curl.exe "):
            continue
        (target,) = re.findall(r'"\$app(/[^"]*)"', command)
        for name, value in variables.items():
            target = target.replace(name, value)
        headers = dict(header.split(": ", 1) for header in re.findall(r'-H "([^"]+)"', command))
        method = re.search(r"-X (\w+)", command)
        parsed.append((method.group(1) if method else "GET", target, headers))
    return parsed


def _guide_email() -> str:
    (email,) = re.findall(r'\$email = \[uri\]::EscapeDataString\("([^"]+)"\)', GUIDE)
    return str(email)


def _admin_app(token: str, **overrides: Any) -> FastAPI:
    return new_app(auth_mode="public", api_tokens={token: {"user": "heo", "role": "admin"}}, **overrides)


def test_deploy_guide_is_linked_and_matches_the_api() -> None:
    assert "(docs/deploy.md)" in README
    for fragment in (
        'py -c "import secrets; print(secrets.token_urlsafe(32))"',
        '{"<여기에 토큰>": {"user": "heo", "role": "admin"}}',
        "/api/v1/metrics/demand",
        "/api/v1/signups?format=csv",
        "/api/v1/me",
        REQUEST_INFO,
        "JETTAE_CLIENT_IP_SOURCE",
        "Save and deploy",
        "-X DELETE",
        "/healthz",
        "/privacy",
        "30일",
        "0.1c-256mb",
        "ipAllowList",
        "JETTAE_LLM_MODE=off",
    ):
        assert fragment in GUIDE, fragment


def test_docs_name_only_real_settings() -> None:
    """A setting the docs name but the app does not know is silently ignored when an operator
    sets it (pydantic drops unknown ``JETTAE_*`` variables) — e.g. ``JETTAE_STORE_RAW``."""
    for name, text in DOCS.items():
        for setting in sorted(set(re.findall(r"JETTAE_[A-Z][A-Z_]*[A-Z]", text))):
            assert setting.removeprefix("JETTAE_").lower() in Settings.model_fields, (name, setting)


def test_docs_state_the_blueprint_limits(blueprint: dict[str, Any]) -> None:
    """The numbers the operator's guide and the threat model quote are the ones deployed."""
    (service,) = blueprint["services"]
    values = {item["key"]: item.get("value") for item in service["envVars"]}
    scans = int(values["JETTAE_DAILY_FREE_SCAN_LIMIT"])
    chars = int(values["JETTAE_MAX_DOCUMENT_CHARS"])
    events = int(values["JETTAE_EVENTS_PER_DAY"])
    retained = scans * int(values["JETTAE_FREE_SCAN_RETENTION_DAYS"])
    for name in ("docs/deploy.md", "docs/security.md"):
        text = DOCS[name]
        assert f"하루 {scans}회" in text, name
        assert f"{chars:,}자" in text, name
        assert f"{events:,}건" in text, name
        assert f"{retained}건" in text, name
        assert values["JETTAE_CLIENT_IP_SOURCE"] in text, name
    assert f"{chars:,}" in ARCHITECTURE
    for fragment in (f"하루 {scans}회", f"{chars:,}자", f"{events:,}건"):
        assert fragment in README, fragment


def test_deploy_guide_commands_work_in_powershell() -> None:
    """``curl`` in Windows PowerShell is an alias of Invoke-WebRequest with other flags."""
    assert "```bash" not in GUIDE
    commands = [line.strip() for line in GUIDE.splitlines() if re.match(r"\s*curl(\.exe)?\s", line)]
    assert commands
    assert all(command.startswith("curl.exe ") for command in commands), commands
    # the e-mail of a deletion goes into the URL path encoded (PowerShell's own .NET call)
    assert '"$app/api/v1/signups/$email"' in GUIDE
    assert quote(_guide_email(), safe="") != _guide_email()  # the example needs the encoding (@)


def test_deploy_guide_commands_work_against_the_api() -> None:
    """Each ``curl.exe … "$app/api/v1/…"`` in the guide, sent as written with an admin token to a
    public-mode app, succeeds — the operator's address check, metrics, export and deletion steps
    really work — and the deletion request leaves the e-mail address nowhere in the log."""
    token = "deploy-guide-token-" + "x" * 32
    commands = _guide_commands()
    assert len(commands) >= 7
    assert [method for method, _, _ in commands].count("DELETE") == 1
    email = _guide_email()
    with capture_logs() as logs, open_client(_admin_app(token)) as client:
        signup = client.post("/api/v1/signups", json={"email": email, "consent_privacy": True})
        assert signup.status_code < 300, signup.text
        for method, target, written in commands:
            assert written.get("Authorization") == "Bearer $token", (method, target)
            headers = {**written, "Authorization": f"Bearer {token}"}
            response = client.request(method, target, headers=headers)
            assert response.status_code == 200, (method, target, response.status_code, response.text[:300])
            if method == "DELETE":
                assert response.json() == {"deleted": True}, target
        me = client.get("/api/v1/me", headers={"Authorization": f"Bearer {token}"}).json()
        assert me["role"] == "admin"
    logged = json.dumps(logs, ensure_ascii=False, default=str).casefold()
    for form in (email, quote(email, safe=""), quote(email, safe="@")):
        assert form.casefold() not in logged, form


def test_request_info_is_explained_in_the_guide() -> None:
    """The guide's example answer has the endpoint's fields, and its table explains every one."""
    token = "deploy-guide-token-" + "y" * 32
    with open_client(_admin_app(token)) as client:
        anonymous = client.get(REQUEST_INFO)
        info = client.get(REQUEST_INFO, headers={"Authorization": f"Bearer {token}"})
    assert anonymous.status_code in (401, 403)  # admin only: it echoes request headers back
    assert info.status_code == 200, info.text
    section = GUIDE[GUIDE.index("### 4-2.") : GUIDE.index("## 5.")]
    (example,) = [json.loads(line) for line in section.splitlines() if line.startswith('{"peer"')]
    assert set(example) == set(info.json())
    for field in info.json():
        assert re.search(rf"^\| `{field}` \|", section, flags=re.MULTILINE), field


def _decision_table() -> dict[str, str]:
    """The guide's step ③: the field where the operator's IP shows → the setting to choose."""
    section = GUIDE[GUIDE.index("**③ 설정 고르기.**") : GUIDE.index("**④ 바꾸기**")]
    return dict(re.findall(r"^\| `(\w+)` \| `(header:[^`]+)`", section, flags=re.MULTILINE))


def test_guide_decision_table_covers_the_proxy_headers() -> None:
    table = _decision_table()
    assert table == {field: f"header:{header}" for field, header in _FIELD_HEADERS.items()}
    assert next(iter(table.values())) == "header:CF-Connecting-IP"  # the blueprint's starting point


@pytest.mark.parametrize("field", sorted(_FIELD_HEADERS))
def test_guide_header_choice_resolves_the_visitor(field: str) -> None:
    """Choosing ``header:<name>`` as the guide says counts the visitor under that header's
    address, whatever the visitor writes into ``X-Forwarded-For`` (which ``peer`` would follow)."""
    source = _decision_table()[field]
    token = "deploy-guide-token-" + "z" * 32
    headers = {
        "Authorization": f"Bearer {token}",
        _FIELD_HEADERS[field]: _MINE,
        "X-Forwarded-For": f"{_FORGED}, {_MINE}, {_HOP}",
    }
    with open_client(_admin_app(token, client_ip_source=source)) as client:
        info = client.get(REQUEST_INFO, headers=headers).json()
    assert info[field] == _MINE
    assert info["client_ip_source"] == source
    assert info["resolved_client"] == _MINE


def test_guide_xff_example_resolves_the_visitor() -> None:
    """The guide's worked ``xff:<n>`` example (n counted from the right, rightmost = 1) picks the
    operator's address, and a forged entry the visitor puts in front does not move it."""
    ((chain, mine, source),) = re.findall(r"예: `([^`]+)`에서 내 IP가 `([^`]+)`이면 `(xff:\d+)`", GUIDE)
    token = "deploy-guide-token-" + "w" * 32
    auth = {"Authorization": f"Bearer {token}"}
    with open_client(_admin_app(token, client_ip_source=source)) as client:
        plain = client.get(REQUEST_INFO, headers={**auth, "X-Forwarded-For": chain}).json()
        forged = client.get(REQUEST_INFO, headers={**auth, "X-Forwarded-For": f"{_FORGED}, {chain}"}).json()
    assert plain["client_ip_source"] == source
    assert plain["resolved_client"] == forged["resolved_client"] == mine


def test_guide_ipv6_note_matches_the_key() -> None:
    """The guide tells the operator an IPv6 visitor shows as a /64 network, not the address."""
    assert "`2001:db8:1:2::/64`" in GUIDE
    token = "deploy-guide-token-" + "v" * 32
    headers = {"Authorization": f"Bearer {token}", "CF-Connecting-IP": "2001:db8:1:2:aaaa:bbbb:cccc:dddd"}
    with open_client(_admin_app(token, client_ip_source="header:CF-Connecting-IP")) as client:
        info = client.get(REQUEST_INFO, headers=headers).json()
    assert info["resolved_client"] == "2001:db8:1:2::/64"


@pytest.mark.parametrize("value", ["cf", "header:", "xff:0", "xff:x", "CF-Connecting-IP"])
def test_malformed_client_ip_source_stops_the_deploy(value: str) -> None:
    """A typo in the dashboard must fail the deploy (security.md), not fall back to ``peer``."""
    with pytest.raises(ValidationError, match="JETTAE_CLIENT_IP_SOURCE"):
        Settings(_env_file=None, client_ip_source=value)


def test_docs_show_no_usable_example_token() -> None:
    """A copied example must never work as an admin token: the docs carry placeholders only, and
    nothing that looks like a generated secret — least of all the OpenAPI examples' access key."""
    docs = {name: DOCS[name] for name in ("docs/deploy.md", "docs/security.md", "README.md", "render.yaml")}
    for name, text in docs.items():
        assert not _TOKEN_LIKE.findall(text), name
    schema = json.dumps(new_app().openapi(), ensure_ascii=False)
    examples = set(re.findall(r'"access_key": "([^"]{16,})"', schema))
    for name, text in docs.items():
        assert not [example for example in examples if example in text], name
