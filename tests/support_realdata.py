"""Offline fakes for the real-data studies: SDK clients that answer from the request schema.

Nothing here reaches a network. The fake Anthropic SDK stands in for ``anthropic.AsyncAnthropic``
(``beta.messages.create`` and ``messages.count_tokens``), the fake OpenAI SDK for
``openai.AsyncOpenAI`` (``responses.create``); both build their answer from the JSON schema the
request asks for, so the real adapters, the spend recorder and the budget gate all run unchanged.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel

from jettae.evalkit.realdata.clients import StudyClients
from jettae.evalkit.realdata.pricing import ModelRef
from jettae.evalkit.realdata.providers import BudgetedLLM, SpendTracker
from jettae.llm.base import UntrustedDocument
from jettae.llm.fake import FakeLLM

PAYMENTS_RUNBOOK = """\
# Payments API runbook

The payments API is owned by the payments team.
When the error rate alarm fires, page the payments on-call in #payments-oncall.
To mitigate, roll back the last deploy.
"""

SEARCH_RUNBOOK = """\
# Search service runbook

The search service is owned by the search team.
Restart the search service if latency stays high.
"""

_REFS = re.compile(r"\[(F\d+)\]")
_UNITS = re.compile(r"\[(U\d+)\]")
_SENTENCES = re.compile(r"\[(S\d+)\]")


def write_runbook_folder(root: Path) -> Path:
    """A tiny GitLab-like input folder: two English runbooks + the folder's own notes."""
    (root / "docs" / "payments").mkdir(parents=True)
    (root / "docs" / "search").mkdir(parents=True)
    (root / "docs" / "payments" / "api.md").write_text(PAYMENTS_RUNBOOK, encoding="utf-8")
    (root / "docs" / "search" / "service.md").write_text(SEARCH_RUNBOOK, encoding="utf-8")
    (root / "SOURCE.md").write_text("고정 커밋: `" + "a" * 40 + "`\n", encoding="utf-8")
    (root / "LICENSE").write_text("MIT\n", encoding="utf-8")
    return root


def extraction_for(text: str) -> dict[str, Any]:
    """A plausible extraction of :data:`PAYMENTS_RUNBOOK` / :data:`SEARCH_RUNBOOK` chunks."""
    if "payments API" in text:
        return {
            "entities": [
                {"kind": "service", "name": "payments API", "quote": "payments API", "attrs": []},
                {"kind": "team", "name": "payments team", "quote": "payments team", "attrs": []},
                {
                    "kind": "action",
                    "name": "roll back the last deploy",
                    "quote": "roll back the last deploy",
                    "attrs": [{"key": "action_type", "value": "rollback", "quote": "roll back"}],
                },
            ],
            "relations": [
                {
                    "kind": "owned_by",
                    "src_kind": "service",
                    "src_name": "payments API",
                    "dst_kind": "team",
                    "dst_name": "payments team",
                    "quote": "The payments API is owned by the payments team.",
                    "attrs": [],
                },
                {
                    "kind": "mitigated_by",
                    "src_kind": "service",
                    "src_name": "payments API",
                    "dst_kind": "action",
                    "dst_name": "roll back the last deploy",
                    "quote": "To mitigate, roll back the last deploy.",
                    "attrs": [],
                },
            ],
        }
    if "search service" in text:
        return {
            "entities": [
                {"kind": "service", "name": "search service", "quote": "search service", "attrs": []},
                {"kind": "team", "name": "search team", "quote": "search team", "attrs": []},
            ],
            "relations": [
                {
                    "kind": "owned_by",
                    "src_kind": "service",
                    "src_name": "search service",
                    "dst_kind": "team",
                    "dst_name": "search team",
                    "quote": "The search service is owned by the search team.",
                    "attrs": [],
                }
            ],
        }
    if "order database" in text:
        return {
            "entities": [
                {"kind": "service", "name": "order database", "quote": "order database", "attrs": []},
                {"kind": "person", "name": "Dana Kim", "quote": "Dana Kim", "attrs": []},
            ],
            "relations": [
                {
                    "kind": "owned_by",
                    "src_kind": "service",
                    "src_name": "order database",
                    "dst_kind": "person",
                    "dst_name": "Dana Kim",
                    "quote": "The order database is owned by Dana Kim.",
                    "attrs": [],
                }
            ],
        }
    return {"entities": [], "relations": []}


def _label_for(ref: str, coder: str) -> dict[str, str]:
    """Deterministic two-question answers that disagree sometimes (so κ is not trivially 1).

    Derived labels: coder A correct unless the number is a multiple of 3 (premise "no"), coder B
    correct for odd numbers and unclear for even ones (claim "unclear").
    """
    number = int(ref[1:])
    if coder == "a":
        premise, claim = ("yes", "yes") if number % 3 else ("no", "yes")
    else:
        premise, claim = ("yes", "yes") if number % 2 else ("yes", "unclear")
    return {
        "finding_ref": ref,
        "premise_valid": premise,
        "claim_true": claim,
        "reason": f"{coder} 판정 {ref}",
    }


def answer(properties: set[str], user_text: str, coder: str) -> dict[str, Any]:  # noqa: PLR0911
    """The JSON a fake model returns for a request whose schema has ``properties``."""
    if "entities" in properties:
        return extraction_for(user_text)
    if "equivalent" in properties:
        return {"equivalent": True, "reason": "같은 사실"}
    if "labels" in properties and _REFS.search(user_text):
        return {"labels": [_label_for(ref, coder) for ref in dict.fromkeys(_REFS.findall(user_text))]}
    if "labels" in properties:
        units = dict.fromkeys(_UNITS.findall(user_text))
        return {"labels": [{"unit": u, "family": "PATH", "element": "fallback_channel"} for u in units]}
    if "runbooks" in properties:
        return reconstructed_bundle()
    if "flagged" in properties:
        sentences = _SENTENCES.findall(user_text)
        leaked = [s for s in sentences if _sentence_text(user_text, s).startswith("During this incident")]
        return {"flagged": [{"unit": s, "category": "incident_event", "reason": "사고 경과"} for s in leaked]}
    if "gaps" in properties:
        return gap_list(coder)
    raise AssertionError(f"unexpected schema: {sorted(properties)}")


def _sentence_text(user_text: str, unit: str) -> str:
    match = re.search(rf"\[{unit}\] (.*)", user_text)
    return match.group(1) if match else ""


def reconstructed_bundle() -> dict[str, Any]:
    return {
        "runbooks": [
            {
                "file_name": "order database.md",
                "markdown": (
                    "# Order database runbook\n\n"
                    "The order database is owned by Dana Kim.\n"
                    "When the write error rate alarm fires, page the database team in #db-oncall.\n"
                    "Failing over to the replica requires approval from the CTO.\n"
                    "During this incident Slack was down. Writes recovered at 00:45.\n"
                ),
            }
        ],
        "roster": [
            {
                "name": "Dana Kim",
                "status": "active",
                "title": None,
                "team": "database team",
                "roles": ["database owner"],
                "channels": ["slack"],
            },
            {
                "name": "CTO",
                "status": "unknown",
                "title": None,
                "team": None,
                "roles": ["CTO"],
                "channels": [],
            },
        ],
        "catalog": [
            {
                "name": "order database",
                "kind": "service",
                "tier": 1,
                "depends_on": [],
                "location": None,
                "owner": "database team",
            },
            {
                "name": "order database",
                "kind": "system",
                "tier": 0,
                "depends_on": [],
                "location": None,
                "owner": None,
            },
        ],
        "policy": {"critical_tiers": [1, 2], "terminal_roles": ["CTO"]},
        "not_stated": ["2차 당직 여부"],
    }


def gap_list(coder: str) -> dict[str, Any]:
    shared = [
        {
            "quote": (
                "The only channel the runbook gave for reaching the database team "
                "depended on the failed network."
            ),
            "description": "대체 연락 채널 없음",
            "family": "PATH",
            "element": "fallback_channel",
        },
        {
            "quote": "Only one person could approve a failover, and no deputy existed.",
            "description": "승인자 대리 없음",
            "family": "PATH",
            "element": "approver",
        },
        {
            "quote": "A schema migration locked the orders table",
            "description": "기술 원인",
            "family": "out_of_scope",
            "element": None,
        },
    ]
    if coder == "a":
        return {
            "gaps": [
                *shared,
                {
                    "quote": "There was no documented rollback for the schema migration",
                    "description": "롤백 절차 없음",
                    "family": "PATH",
                    "element": "rollback",
                },
                {
                    "quote": "This sentence is not in the report",
                    "description": "x",
                    "family": "ESC",
                    "element": None,
                },
            ]
        }
    return {
        "gaps": [
            *shared,
            {
                "quote": (
                    "Nobody was designated to take over when the primary on-call "
                    "could not reach the database owner."
                ),
                "description": "상향 경로 없음",
                "family": "ESC",
                "element": None,
            },
        ]
    }


def _schema_properties(schema: dict[str, Any]) -> set[str]:
    return set((schema.get("properties") or {}).keys())


# --------------------------------------------------------------------------- fake SDKs


class FakeAnthropicSDK:
    """``beta.messages.create`` + ``messages.count_tokens``; every response carries ``usage``."""

    def __init__(self, *, coder: str = "a", input_tokens: int = 1000, output_tokens: int = 500) -> None:
        self.coder = coder
        self.usage = (input_tokens, output_tokens)
        self.created: list[dict[str, Any]] = []
        self.counted: list[dict[str, Any]] = []
        self.closed = False
        outer = self

        class _Messages:
            async def create(self, **kwargs: Any) -> Any:
                return outer._create(kwargs)

            async def count_tokens(self, **kwargs: Any) -> Any:
                outer.counted.append(kwargs)
                return SimpleNamespace(input_tokens=len(json.dumps(kwargs, ensure_ascii=False)) // 4)

        self.beta = SimpleNamespace(messages=_Messages())
        self.messages = _Messages()

    def _create(self, kwargs: dict[str, Any]) -> Any:
        self.created.append(kwargs)
        schema = kwargs["output_config"]["format"]["schema"]
        user = kwargs["messages"][0]["content"]
        system = kwargs["system"][0]["text"]
        coder = "a" if "독립 평가자" in system else self.coder
        payload = answer(_schema_properties(schema), user, coder)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(payload, ensure_ascii=False))],
            stop_reason="end_turn",
            stop_details=None,
            usage=SimpleNamespace(
                input_tokens=self.usage[0],
                output_tokens=self.usage[1],
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
            model=kwargs["model"],
            _request_id="req_test",
        )

    async def close(self) -> None:
        self.closed = True


class FakeOpenAISDK:
    """``responses.create`` returning a completed structured-output response."""

    def __init__(self, *, input_tokens: int = 800, output_tokens: int = 300) -> None:
        self.usage = (input_tokens, output_tokens)
        self.created: list[dict[str, Any]] = []
        self.closed = False
        outer = self

        class _Responses:
            async def create(self, **kwargs: Any) -> Any:
                return outer._create(kwargs)

        self.responses = _Responses()

    def _create(self, kwargs: dict[str, Any]) -> Any:
        self.created.append(kwargs)
        schema = kwargs["text"]["format"]["schema"]
        user = kwargs["input"][0]["content"]
        text = json.dumps(answer(_schema_properties(schema), user, "b"), ensure_ascii=False)
        return SimpleNamespace(
            status="completed",
            incomplete_details=None,
            output=[
                SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text=text)])
            ],
            output_text=text,
            usage=SimpleNamespace(
                input_tokens=self.usage[0],
                output_tokens=self.usage[1],
                input_tokens_details=SimpleNamespace(cached_tokens=0),
            ),
            model=kwargs["model"],
        )

    async def close(self) -> None:
        self.closed = True


# --------------------------------------------------------------------------- fake LLM clients


def fake_handler(coder: str) -> Callable[[str, type[BaseModel], tuple[UntrustedDocument, ...], str], Any]:
    def handle(
        task: str, schema: type[BaseModel], documents: tuple[UntrustedDocument, ...], context: str
    ) -> Any:
        user = "\n".join(d.text for d in documents)
        return answer(set(schema.model_fields), user, coder)

    return handle


#: Price of the metered fakes' model per output token (claude-haiku-5-5: $0.50 / 1M).
_METERED_MODEL = "claude-haiku-5-5"
_METERED_USD_PER_OUTPUT_TOKEN = 0.50 / 1_000_000


class MeteredFake:
    """A fake model whose every call costs exactly ``usd``, booked on the tracker after the reply
    like a real response — so budget tests can be exact (``usd`` below the reservation)."""

    def __init__(self, inner: FakeLLM, tracker: SpendTracker, usd: float) -> None:
        self.inner = inner
        self.tracker = tracker
        self.output_tokens = round(usd / _METERED_USD_PER_OUTPUT_TOKEN)
        self.provider = inner.provider
        self.model = inner.model

    async def structured(self, **kwargs: Any) -> Any:
        result = await self.inner.structured(**kwargs)
        self.tracker.record(_METERED_MODEL, input_tokens=0, output_tokens=self.output_tokens)
        return result


def metered_clients(
    *,
    max_usd: float | None = None,
    max_usd_run: float | None = None,
    usd: float = 0.004,
    reserve: float = 0.01,
    tracker: SpendTracker | None = None,
) -> tuple[StudyClients, FakeLLM, FakeLLM, FakeLLM]:
    """Study clients behind the real budget gate: every call reserves ``reserve`` and costs ``usd``."""
    tracker = tracker or SpendTracker(max_usd=max_usd, max_usd_run=max_usd_run)
    clients, main, coder_a, coder_b = fake_clients(tracker)

    def gate(fake: FakeLLM) -> BudgetedLLM:
        return BudgetedLLM(MeteredFake(fake, tracker, usd), tracker, lambda *_: reserve)

    metered = StudyClients(
        main=gate(main),
        coder_a=gate(coder_a),
        coder_b=gate(coder_b),
        tracker=tracker,
        main_ref=clients.main_ref,
        coder_a_ref=clients.coder_a_ref,
        coder_b_ref=clients.coder_b_ref,
    )
    return metered, main, coder_a, coder_b


def write_reports(root: Path, count: int, *, source: Path) -> Path:
    """``count`` distinct incident reports (copies of the fictional fixture, each with its own id
    and content hash) in ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    text = source.read_text(encoding="utf-8")
    for index in range(1, count + 1):
        (root / f"incident-{index:02d}.md").write_text(
            text + f"\nReport copy {index} (test fixture).\n", encoding="utf-8"
        )
    return root


def fake_clients(tracker: SpendTracker | None = None) -> tuple[StudyClients, FakeLLM, FakeLLM, FakeLLM]:
    main = FakeLLM(fake_handler("main"), provider="anthropic", model="claude-opus-5-5")
    coder_a = FakeLLM(fake_handler("a"), provider="anthropic", model="claude-opus-5-5")
    coder_b = FakeLLM(fake_handler("b"), provider="openai", model="gpt-6.1-sol")
    clients = StudyClients(
        main=main,
        coder_a=coder_a,
        coder_b=coder_b,
        tracker=tracker or SpendTracker(),
        main_ref=ModelRef("anthropic", "claude-opus-5-5"),
        coder_a_ref=ModelRef("anthropic", "claude-opus-5-5"),
        coder_b_ref=ModelRef("openai", "gpt-6.1-sol"),
    )
    return clients, main, coder_a, coder_b
