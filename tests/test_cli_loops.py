"""The API server's event-loop factory retries loop creation the OS refuses (sandboxed Windows)."""

from __future__ import annotations

import asyncio

import pytest

from jettae.cli import loops


def test_returns_a_working_loop() -> None:
    loop = loops.resilient_event_loop()
    try:
        assert loop.run_until_complete(asyncio.sleep(0, result=42)) == 42
    finally:
        loop.close()


def test_retries_until_the_os_accepts(monkeypatch: pytest.MonkeyPatch) -> None:
    real = asyncio.new_event_loop
    attempts: list[int] = []

    def flaky() -> asyncio.AbstractEventLoop:
        attempts.append(1)
        if len(attempts) < 3:
            raise OSError(10014, "socketpair refused")
        return real()

    monkeypatch.setattr(loops.asyncio, "new_event_loop", flaky)
    monkeypatch.setattr(loops.time, "sleep", lambda _s: None)
    loop = loops.resilient_event_loop()
    loop.close()
    assert len(attempts) == 3


def test_gives_up_after_the_last_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    def refused() -> asyncio.AbstractEventLoop:
        raise OSError(10014, "socketpair refused")

    monkeypatch.setattr(loops.asyncio, "new_event_loop", refused)
    monkeypatch.setattr(loops.time, "sleep", lambda _s: None)
    with pytest.raises(OSError, match="socketpair refused"):
        loops.resilient_event_loop()
