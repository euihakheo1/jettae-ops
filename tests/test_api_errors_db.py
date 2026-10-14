"""Database failures in the API's logs: class names only, never the values being written.

A database error's message quotes the statement's values — SQLAlchemy's ``[parameters: …]`` and the
driver's own text (PostgreSQL's ``DETAIL: Key (email)=(…)``) — and a traceback prints every message
of the exception chain. With JSON logs on (the public deployment), a failing sign-up insert would
otherwise put the person's e-mail and free-text message into the operator's log.
"""

from __future__ import annotations

import sqlite3
import traceback
from collections.abc import Iterator
from contextlib import closing

import pytest
import structlog
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from structlog.testing import capture_logs

from jettae.api.errors import failure_log_fields
from jettae.api.middleware import configure_logging
from jettae.store import SqlStore, StoreIntegrityError
from test_api_support import API, new_app, new_service, open_client

VICTIM = "victim@corp.example"
PHONE = "010-1234-5678"


@pytest.fixture
def fresh_structlog() -> Iterator[None]:
    structlog.reset_defaults()
    try:
        yield
    finally:
        structlog.reset_defaults()


def _db_error() -> IntegrityError:
    """What SQLAlchemy raises without ``hide_parameters``: the row's values in every message."""
    driver = sqlite3.IntegrityError(f"UNIQUE constraint failed: signups.email ({VICTIM})")
    error = IntegrityError("INSERT INTO signups (email, message) VALUES (?, ?)", (VICTIM, PHONE), driver)
    error.__cause__ = driver
    return error


@pytest.mark.usefixtures("fresh_structlog")
def test_a_failing_signup_insert_logs_no_personal_data(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(json_logs=True)  # the deployment's renderer, tracebacks included
    store = SqlStore("sqlite://")
    with store.engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TRIGGER refuse_signups BEFORE INSERT ON signups "
                "BEGIN SELECT RAISE(ABORT, 'disk full'); END"
            )
        )
    signup = {
        "email": "Victim@Corp.example",
        "organization": "에이크미",
        "message": f"연락처 {PHONE}",
        "consent_privacy": True,
    }
    with closing(store), open_client(new_app(service=new_service(store=store))) as client:
        response = client.post(f"{API}/signups", json=signup)
    assert response.status_code == 500
    assert response.json()["code"] == "internal_error"
    logged = capsys.readouterr().err
    assert "http.unhandled_error" in logged
    assert "IntegrityError" in logged  # the class names are what an operator gets
    for secret in (VICTIM, "Victim@Corp.example", PHONE, "에이크미", "INSERT", "Traceback"):
        assert secret not in logged, secret


def test_server_errors_caused_by_the_database_log_class_names_only() -> None:
    router = APIRouter()

    @router.get("/broken")
    def broken() -> None:
        raise StoreIntegrityError("저장된 데이터가 기록과 일치하지 않습니다.") from _db_error()

    app = new_app()
    app.include_router(router, prefix="/api/test")
    with open_client(app) as client, capture_logs() as logs:
        response = client.get("/api/test/broken")
    assert response.status_code == 500
    assert response.json()["code"] == "integrity_error"
    (entry,) = [entry for entry in logs if entry["event"] == "api.domain_error"]
    assert entry["error"] == "StoreIntegrityError"
    assert entry["db_error"] == "IntegrityError"
    assert entry["db_driver_error"] == "IntegrityError"
    assert "exc_info" not in entry
    assert VICTIM not in repr(logs)


def test_other_failures_keep_their_traceback() -> None:
    crash = RuntimeError("defect")
    assert failure_log_fields(crash) == {"error": "RuntimeError", "exc_info": crash}
    assert failure_log_fields(_db_error()) == {
        "error": "IntegrityError",
        "db_error": "IntegrityError",
        "db_driver_error": "IntegrityError",
    }


def test_a_database_error_after_the_response_started_reaches_the_server_without_values() -> None:
    """Nothing is left to replace once streaming began, so the error goes to the server — which
    prints the whole chain; a database error therefore leaves as a bare RuntimeError."""
    router = APIRouter()

    @router.get("/stream")
    def stream() -> StreamingResponse:
        def chunks() -> Iterator[bytes]:
            yield b"partial"
            raise _db_error()

        return StreamingResponse(chunks(), media_type="text/plain")

    app = new_app()
    app.include_router(router, prefix="/api/test")
    with open_client(app) as client, pytest.raises(RuntimeError, match="IntegrityError") as raised:
        client.get("/api/test/stream")
    rendered = "".join(traceback.format_exception(raised.value))
    assert VICTIM not in rendered
    assert PHONE not in rendered
