"""Shared helpers for the ``test_orchestrator_*`` suites (no tests here).

* :func:`event_loop_runner` — one ``asyncio.Runner`` per test module. Sandboxed Windows runners
  intermittently fail to create an event loop's self-pipe (``socket.socketpair``, WinError 10014),
  which has nothing to do with the code under test, so creation is retried. LangGraph runs
  synchronous nodes in the loop's worker threads, so a real loop is needed (a fake LLM that never
  suspends is not enough here).
* :func:`fake_llm` — a deterministic stand-in model for every task the pipeline may ask:
  ``extract`` (one grounded relation from the 결제 runbook), ``backtranslate`` (always equivalent),
  ``scenarios`` (one valid drill) and ``draft_fix`` (picks the template's own person and echoes the
  template sentence, so the LLM drafter accepts it).
* :func:`fixed_clock`, :func:`settings` and :func:`new_service` — reproducible services on a
  :class:`~jettae.store.MemoryStore`, with the LLM mode pinned OFF unless a test asks otherwise.
* :func:`demo_report_digest` — the demo report as one hash, for runs under other hash seeds.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from jettae.config import Settings
from jettae.domain import LlmMode
from jettae.extract.backtranslate import EquivalenceVerdict
from jettae.extract.llm_extractor import LlmGraphPayload
from jettae.llm.base import LLMClient, UntrustedDocument
from jettae.llm.fake import FakeHandler, FakeLLM, FakeOutcome
from jettae.orchestrator import JettaeService
from jettae.rehearse import LlmScenarioPayload
from jettae.store import MemoryStore, Store

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "dmc-shop"
PAYMENT = "service:결제"
CHUSEOK = "human:dmc-s1-chuseok-payment"
CLOCK = datetime(2026, 10, 28, 14, 0, tzinfo=UTC)

#: The 결제 runbook sentence the fake model "extracts" (verbatim in runbooks/01-…).
PAYMENT_QUOTE = "결제 장애가 계속되면 김대리에게 연락"

LOOP_ATTEMPTS = 8


@contextmanager
def event_loop_runner() -> Iterator[asyncio.Runner]:
    """An ``asyncio.Runner`` whose loop creation is retried on transient ``OSError``."""
    for attempt in range(LOOP_ATTEMPTS):
        runner = asyncio.Runner()
        try:
            runner.get_loop()
        except OSError:
            runner.close()
            if attempt == LOOP_ATTEMPTS - 1:
                raise
            continue
        break
    try:
        yield runner
    finally:
        runner.close()


def fixed_clock() -> datetime:
    return CLOCK


#: Pinned so a developer's ``JETTAE_*`` environment (say ``JETTAE_LLM_MODE=live``) cannot make a
#: test call a real model or keep raw text: explicit values win over the environment.
_PINNED: dict[str, Any] = {"llm_mode": LlmMode.OFF, "store_raw_documents": False, "llm_concurrency": 4}


def settings(**overrides: Any) -> Settings:
    """Settings independent of the developer's environment and ``.env`` file."""
    return Settings(_env_file=None, **{**_PINNED, **overrides})


def demo_report_digest() -> str:
    """SHA-256 of a fresh demo report minus its wall-clock timings (run under other hash seeds)."""
    with event_loop_runner() as runner:
        report = runner.run(new_service().bootstrap_demo())
    return hashlib.sha256(report.model_dump_json(exclude={"timings_ms"}).encode("utf-8")).hexdigest()


def new_service(
    *,
    store: Store | None = None,
    llm_factory: Callable[[Settings, LlmMode], LLMClient] | None = None,
    **overrides: Any,
) -> JettaeService:
    target = store if store is not None else MemoryStore(clock=fixed_clock)
    if llm_factory is None:
        return JettaeService(target, settings(**overrides), clock=fixed_clock)
    return JettaeService(target, settings(**overrides), clock=fixed_clock, llm_factory=llm_factory)


def _draft_choice(documents: tuple[UntrustedDocument, ...], context: str) -> dict[str, str]:
    """Pick the candidate the template sentence names and echo that sentence (a faithful rewording)."""
    text = documents[0].text.split("[템플릿 초안]\n", 1)[-1]
    candidates = re.findall(r"^- (person:\S+?):", context, flags=re.MULTILINE)
    named = [c for c in candidates if c.split(":", 1)[1] in text]
    choice = (named or candidates)[0]
    return {
        "choice": choice,
        "runbook_text": text,
        "rationale": "템플릿이 고른 사람과 같은 사람을 골랐습니다. 명부에서 활동 중인 사람입니다.",
    }


def demo_handler(
    task: str, schema: type[BaseModel], documents: tuple[UntrustedDocument, ...], context: str
) -> FakeOutcome:
    """Answers of the fake model (see the module docstring)."""
    if task == "extract":
        chunk = documents[0].text if documents else ""
        if PAYMENT_QUOTE not in chunk:
            return LlmGraphPayload()
        return {
            "entities": [],
            "relations": [
                {
                    "kind": "owned_by",
                    "src_kind": "service",
                    "src_name": "결제",
                    "dst_kind": "person",
                    "dst_name": "김대리",
                    "quote": PAYMENT_QUOTE,
                    "attrs": [],
                }
            ],
        }
    if task == "backtranslate":
        return EquivalenceVerdict(equivalent=True, reason="같은 뜻입니다.")
    if task == "scenarios":
        return LlmScenarioPayload.model_validate(
            {
                "scenarios": [
                    {
                        "title": "결제 장애 중 팀장 부재",
                        "description": "주말 오후 결제 오류가 이어지는데 박재현 팀장이 등산 중이다.",
                        "service_id": PAYMENT,
                        "incident_type": "outage",
                        "perturbations": [
                            {
                                "kind": "person_unavailable",
                                "target_id": "person:박재현",
                                "start_minute": 0,
                                "end_minute": None,
                            }
                        ],
                    }
                ]
            }
        )
    if task == "draft_fix":
        return _draft_choice(documents, context)
    raise AssertionError(f"unexpected task {task}")


def fake_llm(handler: FakeHandler = demo_handler) -> FakeLLM:
    return FakeLLM(handler)


def failing_llm(task: str, error: Exception) -> FakeLLM:
    """The demo model, except that every ``task`` request raises ``error``."""

    def handler(
        asked: str, schema: type[BaseModel], documents: tuple[UntrustedDocument, ...], context: str
    ) -> FakeOutcome:
        if asked == task:
            return error
        return demo_handler(asked, schema, documents, context)

    return FakeLLM(handler)


def factory_of(client: LLMClient) -> Callable[[Settings, LlmMode], LLMClient]:
    """An LLM factory that hands out ``client`` for LIVE/REPLAY (the pipeline ignores it when OFF)."""

    def build(_settings: Settings, _mode: LlmMode) -> LLMClient:
        return client

    return build
