"""Application wiring: lifespan and store, error contract, SPA static serving, OpenAPI document.

The error contract is tested against a throwaway router that raises each domain error, so the
mapping (status, code, Korean detail) is pinned independently of which service call happens to
raise what today.
"""

from __future__ import annotations

import importlib.util
import json
import re
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import structlog
from fastapi import APIRouter, FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from hypothesis import given
from hypothesis import strategies as st
from pydantic import BaseModel, Field

from jettae import __version__
from jettae.api import OPENAPI_URL, create_app
from jettae.api.errors import translate_validation_errors
from jettae.api.middleware import WEB_CSP, _logged_path, configure_logging
from jettae.domain.ports import NotFoundError as PortNotFoundError
from jettae.domain.ports import VersionConflictError
from jettae.improve import PatchStateError
from jettae.llm.base import LLMError
from jettae.orchestrator import JettaeService
from jettae.store import AlreadyExistsError, MemoryStore, NotFoundError, StoreIntegrityError
from test_api_support import API, EDITOR, ROOT, new_app, new_service, open_client, settings

# --------------------------------------------------------------------------- lifespan


def test_lifespan_builds_and_closes_the_store_from_the_database_url(tmp_path: Path) -> None:
    url = f"sqlite:///{(tmp_path / 'jettae.db').as_posix()}"
    app = create_app(settings(database_url=url))
    assert app.state.service is None
    with open_client(app) as client:
        assert isinstance(app.state.service, JettaeService)
        assert client.get("/readyz").json() == {"status": "ready", "detail": None}
        created = client.post(f"{API}/workspaces", json={"name": "영속 확인"}, headers=EDITOR)
        assert created.status_code == 201
    assert app.state.service is None
    # a second app on the same database sees what the first one wrote
    with open_client(create_app(settings(database_url=url))) as again:
        assert [w["name"] for w in again.get(f"{API}/workspaces").json()] == ["영속 확인"]


def test_an_injected_service_is_used_and_left_open() -> None:
    service = new_service()
    app = create_app(settings(), service=service)
    with open_client(app) as client:
        assert app.state.service is service
        client.post(f"{API}/workspaces", json={"name": "주입"}, headers=EDITOR)
    assert [w.name for w in service.list_workspaces()] == ["주입"]


def test_settings_default_to_the_injected_service_settings() -> None:
    service = new_service(max_upload_mb=3)
    app = create_app(service=service)
    assert app.state.settings is service.settings


def test_requests_before_the_service_exists_are_503() -> None:
    app = new_app()
    with open_client(app) as client:
        service = app.state.service
        app.state.service = None
        try:
            response = client.get(f"{API}/workspaces")
            assert response.status_code == 503
            assert response.json()["code"] == "service_unavailable"
            assert client.get("/readyz").status_code == 503
        finally:
            app.state.service = service


class _BrokenWorkspaces:
    def get_workspace(self, workspace_id: str) -> None:
        raise ConnectionError("postgresql://admin:hunter2@db/jettae is down")


def _broken_store() -> MemoryStore:
    store = MemoryStore()
    store.workspaces = _BrokenWorkspaces()  # type: ignore[assignment]
    return store


def test_readiness_fails_without_leaking_connection_details() -> None:
    service = new_service(store=_broken_store())
    with open_client(create_app(settings(), service=service)) as client:
        assert client.get("/healthz").json() == {"status": "ok", "version": __version__}
        ready = client.get("/readyz")
    assert ready.status_code == 503
    assert ready.json()["status"] == "unavailable"
    assert "hunter2" not in ready.text
    assert "저장소" in ready.json()["detail"]


# --------------------------------------------------------------------------- error contract


class _Payload(BaseModel):
    count: int = Field(ge=1)


def _raising_app() -> FastAPI:
    errors: dict[str, BaseException] = {
        "not-found": PortNotFoundError("워크스페이스 ws-1을(를) 찾을 수 없습니다."),
        "store-not-found": NotFoundError("수정안 p-1을(를) 찾을 수 없습니다."),
        "permission": PermissionError("승인 권한이 없습니다."),
        # the OS flavour (errno set): the server cannot read a file — not the caller's role
        "os-permission": PermissionError(13, "Permission denied", "/srv/secret internals/cassette.jsonl"),
        "conflict": VersionConflictError("그래프 버전이 바뀌었습니다."),
        "exists": AlreadyExistsError("이미 있습니다."),
        "patch-state": PatchStateError("이미 결정된 수정안입니다."),
        "value-ko": ValueError("런북이 없습니다."),
        "value-en": ValueError("invalid literal for int() with base 10: 'x'"),
        "llm": LLMError("upstream 529 overloaded"),
        "integrity": StoreIntegrityError("저장된 그래프가 손상되었습니다."),
        "crash": RuntimeError("secret internals"),
    }
    router = APIRouter()

    @router.get("/raise/{name}")
    def raise_error(name: str) -> None:
        raise errors[name]

    @router.get("/invalid-model")
    def invalid_model() -> None:
        _Payload.model_validate({"count": 0})

    @router.post("/echo")
    def echo(body: _Payload) -> _Payload:
        return body

    app = new_app()
    app.include_router(router, prefix="/api/test")
    return app


@pytest.fixture(scope="module")
def raising() -> Iterator[TestClient]:
    with open_client(_raising_app()) as client:
        yield client


@pytest.mark.parametrize(
    ("name", "status", "code", "detail"),
    [
        ("not-found", 404, "not_found", "워크스페이스 ws-1을(를) 찾을 수 없습니다."),
        ("store-not-found", 404, "not_found", "수정안 p-1을(를) 찾을 수 없습니다."),
        ("permission", 403, "forbidden", "승인 권한이 없습니다."),
        ("os-permission", 500, "internal_error", None),
        ("conflict", 409, "version_conflict", "그래프 버전이 바뀌었습니다."),
        ("exists", 409, "already_exists", "이미 있습니다."),
        ("patch-state", 409, "patch_state", "이미 결정된 수정안입니다."),
        ("value-ko", 422, "invalid_input", "런북이 없습니다."),
        ("value-en", 422, "invalid_input", "입력값이 올바르지 않습니다."),
        ("llm", 502, "llm_failed", None),
        ("integrity", 500, "integrity_error", "저장된 그래프가 손상되었습니다."),
        ("crash", 500, "internal_error", None),
    ],
)
def test_domain_errors_map_to_status_code_and_korean_detail(
    raising: TestClient, name: str, status: int, code: str, detail: str | None
) -> None:
    response = raising.get(f"/api/test/raise/{name}")
    assert response.status_code == status
    body = response.json()
    assert body["code"] == code
    assert re.search(r"[가-힣]", body["detail"])
    if detail is not None:
        assert body["detail"] == detail
    assert body["request_id"] == response.headers["x-request-id"]
    assert "secret internals" not in response.text
    assert "upstream" not in response.text
    assert response.headers["x-content-type-options"] == "nosniff"


def test_validation_errors_inside_handlers_and_in_requests_are_korean(raising: TestClient) -> None:
    inner = raising.get("/api/test/invalid-model")
    assert inner.status_code == 422
    assert inner.json()["detail"] == "count: 1 이상이어야 합니다."
    bad_json = raising.post("/api/test/echo", content=b"{nope", headers={"Content-Type": "application/json"})
    assert bad_json.status_code == 422
    assert bad_json.json()["detail"] == "요청 본문: 올바른 JSON이 아닙니다."
    assert bad_json.json()["errors"][0]["field"] == "요청 본문"
    wrong_type = raising.post("/api/test/echo", json={"count": "many"})
    assert wrong_type.json()["errors"] == [
        {"field": "count", "message": "정수여야 합니다.", "type": "int_parsing"}
    ]


def test_framework_errors_are_korean_too(raising: TestClient) -> None:
    missing = raising.get(f"{API}/no-such-route")
    assert missing.status_code == 404
    assert missing.json() == {
        "detail": "요청한 경로를 찾을 수 없습니다.",
        "code": "not_found",
        "request_id": missing.headers["x-request-id"],
    }
    wrong_method = raising.delete(f"{API}/rules")
    assert wrong_method.status_code == 405
    assert wrong_method.json()["code"] == "method_not_allowed"
    assert "GET" in wrong_method.headers["allow"]


_ERROR_TYPES = [
    ("missing", {}),
    ("string_too_short", {"min_length": 1}),
    ("string_too_long", {"max_length": 5}),
    ("too_long", {"max_length": 3}),
    ("greater_than_equal", {"ge": 1}),
    ("less_than_equal", {"le": 9}),
    ("literal_error", {"expected": "'a', 'b' or 'c'"}),
    ("enum", {"expected": "'off', 'live' or 'replay'"}),
    ("value_error", {"error": ValueError("이메일 주소 형식이 올바르지 않습니다.")}),
    ("value_error", {"error": ValueError("plain english")}),
    ("json_invalid", {"error": "Expecting value"}),
    ("some_future_type", {}),
    ("string_too_long", {}),
]


@given(
    st.sampled_from(_ERROR_TYPES),
    st.lists(
        st.one_of(st.sampled_from(["body", "query", "path", "header"]), st.text(max_size=8), st.integers())
    ),
)
def test_every_translated_validation_error_is_korean(
    error: tuple[str, dict[str, Any]], loc: list[object]
) -> None:
    error_type, ctx = error
    (field_error,) = translate_validation_errors([{"type": error_type, "loc": tuple(loc), "ctx": ctx}])
    assert re.search(r"[가-힣]", field_error.message)
    assert field_error.type == error_type
    assert field_error.field
    if error_type == "json_invalid":
        # located by a character offset, not a field: it is about the body as a whole
        assert field_error.field == "요청 본문"
    elif loc and loc[0] in ("body", "query", "path", "header"):
        assert field_error.field == (".".join(str(part) for part in loc[1:]) or "요청 본문")


def test_literal_expectations_are_listed_without_english() -> None:
    (error,) = translate_validation_errors(
        [{"type": "literal_error", "loc": ("body", "name"), "ctx": {"expected": "'a' or 'b'"}}]
    )
    assert error.message == "허용되지 않은 값입니다(허용: 'a', 'b')."


# --------------------------------------------------------------------------- SPA


@pytest.fixture
def web_dist(tmp_path: Path) -> Path:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(
        "<!doctype html><title>제때옵스</title><div id=root></div>", encoding="utf-8"
    )
    (dist / "assets" / "index-abc123.js").write_text("console.log('jettae')", encoding="utf-8")
    (dist / "favicon.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("do not serve", encoding="utf-8")
    return dist


def test_spa_is_served_with_fallback_and_api_paths_stay_json(web_dist: Path) -> None:
    with open_client(new_app(web_dist=web_dist)) as client:
        root = client.get("/")
        assert root.status_code == 200
        assert "제때옵스" in root.text
        assert root.headers["cache-control"] == "no-cache"
        assert root.headers["content-security-policy"] == WEB_CSP
        deep = client.get("/app/w/ws-1/patches", params={"patch": "p-1"})
        assert deep.status_code == 200
        assert deep.text == root.text
        assert client.head("/app/w/ws-1").status_code == 200
        asset = client.get("/assets/index-abc123.js")
        assert asset.status_code == 200
        assert "immutable" in asset.headers["cache-control"]
        assert client.get("/favicon.svg").headers["content-type"].startswith("image/svg")
        for path in ("/assets/missing.js", f"{API}/nope", "/api", "/healthz/extra"):
            response = client.get(path)
            assert response.status_code == 404, path
            assert response.json()["code"] == "not_found"
        assert client.post("/app/w/ws-1").status_code == 405
        assert client.get("/healthz").json()["status"] == "ok"
        for escape in (
            "/%2e%2e/secret.txt",
            "/assets/..%2f..%2fsecret.txt",
            "/assets/%2e%2e/%2e%2e/secret.txt",
        ):
            traversal = client.get(escape)
            assert "do not serve" not in traversal.text, escape


def test_spa_mount_leaves_wrong_methods_on_api_paths_to_the_router(web_dist: Path) -> None:
    """The mount at "/" must not shadow the router's 405 for a known API path (it did: the mount
    matched fully and answered 404), and the answer must be the same with or without the SPA."""
    for app in (new_app(web_dist=web_dist), new_app()):
        with open_client(app) as client:
            for method, path, allowed in (
                ("DELETE", f"{API}/workspaces", "POST"),
                ("POST", f"{API}/rules", "GET"),
                ("GET", f"{API}/patches/p-1/approve", "POST"),
                ("POST", "/healthz", "GET"),
            ):
                response = client.request(method, path)
                assert response.status_code == 405, (method, path)
                assert response.json()["code"] == "method_not_allowed"
                assert allowed in response.headers["allow"]
            assert client.get(f"{API}/nope").json()["code"] == "not_found"


def test_spa_fallback_respects_a_proxy_root_path(web_dist: Path) -> None:
    """Behind a proxy prefix (``root_path``) the backend paths are still recognised as such."""
    with open_client(new_app(web_dist=web_dist), root_path="/jettae") as client:
        assert client.get("/jettae/app/w/ws-1").text.startswith("<!doctype html>")
        missing = client.get(f"/jettae{API}/nope")
        assert missing.status_code == 404
        assert missing.json()["code"] == "not_found"
        assert client.delete(f"/jettae{API}/workspaces").status_code == 405


def test_no_spa_without_a_built_bundle(tmp_path: Path) -> None:
    with open_client(new_app(web_dist=tmp_path / "missing")) as client:
        response = client.get("/")
    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


# --------------------------------------------------------------------------- OpenAPI

#: every operation of ARCHITECTURE §6 plus the read endpoints the web app needs
EXPECTED_OPERATIONS = {
    ("post", "/api/v1/workspaces"): "createWorkspace",
    ("get", "/api/v1/workspaces"): "listWorkspaces",
    ("get", "/api/v1/workspaces/{ws}"): "getWorkspace",
    ("get", "/api/v1/workspaces/{ws}/inputs"): "getInputs",
    ("post", "/api/v1/workspaces/{ws}/inputs"): "uploadInputs",
    ("post", "/api/v1/workspaces/{ws}/diagnose"): "diagnoseWorkspace",
    ("get", "/api/v1/workspaces/{ws}/report"): "getReport",
    ("get", "/api/v1/workspaces/{ws}/report.md"): "getReportMarkdown",
    ("get", "/api/v1/workspaces/{ws}/report.html"): "getReportHtml",
    ("get", "/api/v1/workspaces/{ws}/report.sarif"): "getReportSarif",
    ("get", "/api/v1/workspaces/{ws}/graph"): "getGraph",
    ("get", "/api/v1/workspaces/{ws}/versions"): "listVersions",
    ("get", "/api/v1/workspaces/{ws}/diff"): "diffVersions",
    ("post", "/api/v1/workspaces/{ws}/versions/{n}/revert"): "revertVersion",
    ("get", "/api/v1/workspaces/{ws}/scenarios"): "listScenarios",
    ("post", "/api/v1/workspaces/{ws}/rehearsals"): "runRehearsals",
    ("get", "/api/v1/workspaces/{ws}/patches"): "listPatches",
    ("post", "/api/v1/workspaces/{ws}/patches/draft"): "draftPatches",
    ("get", "/api/v1/patches/{patch_id}"): "getPatch",
    ("post", "/api/v1/patches/{patch_id}/approve"): "approvePatch",
    ("post", "/api/v1/patches/{patch_id}/reject"): "rejectPatch",
    ("get", "/api/v1/rules"): "listRules",
    ("post", "/api/v1/demo/dmc-shop"): "bootstrapDemo",
    ("post", "/api/v1/free-scans"): "createFreeScan",
    ("post", "/api/v1/signups"): "createSignup",
    ("get", "/api/v1/signups"): "listSignups",
    ("delete", "/api/v1/signups/{email}"): "deleteSignup",
    ("post", "/api/v1/events"): "recordEvent",
    ("get", "/api/v1/metrics/demand"): "getDemandMetrics",
    ("get", "/api/v1/me"): "getSession",
    ("get", "/api/v1/admin/request-info"): "getRequestInfo",
    ("get", "/healthz"): "healthz",
    ("get", "/readyz"): "readyz",
}


@pytest.fixture(scope="module")
def openapi() -> dict[str, Any]:
    with open_client(new_app()) as client:
        response = client.get(OPENAPI_URL)
    assert response.status_code == 200
    document: dict[str, Any] = response.json()
    return document


def test_openapi_lists_every_operation_with_a_stable_id(openapi: dict[str, Any]) -> None:
    operations = {
        (method, path): op["operationId"]
        for path, ops in openapi["paths"].items()
        for method, op in ops.items()
    }
    assert operations == EXPECTED_OPERATIONS
    assert all(re.fullmatch(r"[a-z][A-Za-z]+", op_id) for op_id in operations.values())


def test_openapi_metadata_tags_and_examples(openapi: dict[str, Any]) -> None:
    assert openapi["info"]["title"] == "제때옵스 API"
    assert openapi["info"]["version"] == __version__
    tags = [tag["name"] for tag in openapi["tags"]]
    assert all(re.search(r"[가-힣]", tag) for tag in tags)
    used = {tag for ops in openapi["paths"].values() for op in ops.values() for tag in op["tags"]}
    assert used == set(tags)
    schemas = openapi["components"]["schemas"]
    for name in (
        "ApiError",
        "ValidationApiError",
        "WorkspaceSummary",
        "CreateWorkspaceRequest",
        "DiagnoseRequest",
        "DraftRequest",
        "RehearsalRequest",
        "RejectRequest",
        "SignupRequest",
        "EventRequest",
        "FreeScanResult",
    ):
        assert schemas[name].get("examples"), name


def test_openapi_documents_our_error_bodies_only(openapi: dict[str, Any]) -> None:
    schemas = openapi["components"]["schemas"]
    assert "HTTPValidationError" not in schemas
    for path, ops in openapi["paths"].items():
        for method, op in ops.items():
            takes_input = bool(op.get("parameters")) or "requestBody" in op
            responses = op["responses"]
            if takes_input:
                assert "422" in responses, (method, path)
            if "422" in responses:
                ref = responses["422"]["content"]["application/json"]["schema"]["$ref"]
                assert ref.endswith("/ValidationApiError"), (method, path)


# --------------------------------------------------------------------------- export script


def _export_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("export_openapi", ROOT / "scripts" / "export_openapi.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_script_writes_and_checks(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    export = _export_module()
    target = tmp_path / "openapi.json"
    assert export.main(["--check", "--output", str(target)]) == 1
    assert export.main(["--output", str(target)]) == 0
    written = target.read_text(encoding="utf-8")
    assert written == export.openapi_json()
    assert json.loads(written)["info"]["title"] == "제때옵스 API"
    assert export.main(["--check", "--output", str(target)]) == 0
    target.write_text(written.replace("제때옵스 API", "다른 API"), encoding="utf-8")
    assert export.main(["--check", "--output", str(target)]) == 1
    assert "최신이 아닙니다" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("error_type", "ctx", "message"),
    [
        ("string_too_long", {}, "값의 범위가 올바르지 않습니다."),
        ("string_too_long", {"max_length": 5}, "5자 이하여야 합니다."),
        ("value_error", {"error": ValueError("plain english")}, "값이 올바르지 않습니다."),
        ("some_future_type", {}, "값이 올바르지 않습니다."),
        ("missing", {}, "필수 값이 빠졌습니다."),
    ],
)
def test_validation_reasons(error_type: str, ctx: dict[str, Any], message: str) -> None:
    (error,) = translate_validation_errors([{"type": error_type, "loc": ("body",), "ctx": ctx}])
    assert error.message == message
    assert error.field == "요청 본문"


def test_a_crash_after_the_response_started_reaches_the_server() -> None:
    router = APIRouter()

    @router.get("/stream")
    def stream() -> StreamingResponse:
        def chunks() -> Iterator[bytes]:
            yield b"partial"
            raise RuntimeError("mid-stream")

        return StreamingResponse(chunks(), media_type="text/plain")

    app = new_app()
    app.include_router(router, prefix="/api/test")
    with open_client(app) as client, pytest.raises(RuntimeError, match="mid-stream"):
        client.get("/api/test/stream")


# --------------------------------------------------------------------------- logging


@pytest.fixture
def fresh_structlog() -> Iterator[None]:
    structlog.reset_defaults()
    try:
        yield
    finally:
        structlog.reset_defaults()


@pytest.mark.usefixtures("fresh_structlog")
def test_logging_is_json_on_request_and_never_overrides_a_host_configuration() -> None:
    configure_logging(json_logs=True)
    processors = structlog.get_config()["processors"]
    assert isinstance(processors[-1], structlog.processors.JSONRenderer)
    configure_logging(json_logs=False)
    assert structlog.get_config()["processors"] is processors


@pytest.mark.usefixtures("fresh_structlog")
def test_console_logs_use_plain_tracebacks_without_local_variables() -> None:
    configure_logging(json_logs=False)
    renderer = structlog.get_config()["processors"][-1]
    assert isinstance(renderer, structlog.dev.ConsoleRenderer)
    try:
        token = "secret-token-value"
        raise RuntimeError(token[:3])
    except RuntimeError as exc:
        rendered = renderer(None, "error", {"event": "boom", "exc_info": exc})
    assert "RuntimeError" in str(rendered)
    assert "secret-token-value" not in str(rendered)


def test_logged_path_falls_back_to_the_decoded_path() -> None:
    assert _logged_path({"path": "/api/v1/workspaces/별빛"}) == "/api/v1/workspaces/%EB%B3%84%EB%B9%9B"
    assert _logged_path({"raw_path": b"/api/v1/x%20y", "path": "/api/v1/x y"}) == "/api/v1/x%20y"
