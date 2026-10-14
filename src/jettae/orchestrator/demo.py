"""Finals demo helpers (결선 데모): where the example bundle lives, and how to record LLM answers.

The venue demo must work without a network: the presenter replays recorded model answers
(``JETTAE_LLM_MODE=replay``) instead of calling a live API in a hall with unreliable Wi-Fi. A cassette
records requests by content (task, instructions, schema, documents, context — see
:mod:`jettae.llm.replay`), so a recording made once with a real key replays byte-for-byte as long as
the bundle and the code that builds the prompts are unchanged. A miss takes the deterministic path
like LLM OFF (P4: rule parser for extraction, templates for wording), so the demo never breaks — it
only loses the LLM layer (an extraction fallback is noted in the run's notes).

Recording (once, with a key — e.g. ``ANTHROPIC_API_KEY`` and ``JETTAE_LLM_PROVIDER=anthropic``)::

    import asyncio
    from jettae.config import Settings
    from jettae.orchestrator import JettaeService, record_cassette
    from jettae.store import MemoryStore

    service = JettaeService(MemoryStore(), Settings())
    report = asyncio.run(record_cassette(service, "data/cassettes/dmc-shop.jsonl"))
    print(report.llm.calls, report.llm.by_task)

Replaying at the venue::

    JETTAE_LLM_MODE=replay JETTAE_CASSETTE_PATH=data/cassettes/dmc-shop.jsonl jettae demo

(or ``await service.bootstrap_demo(llm_mode=LlmMode.REPLAY)`` with those settings).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Final

import jettae
from jettae.config import Settings
from jettae.domain import DiagnosisReport, LlmMode
from jettae.llm.base import LLMClient
from jettae.llm.replay import CassetteLLM
from jettae.store import MemoryStore

if TYPE_CHECKING:
    from jettae.orchestrator.service import JettaeService

__all__ = [
    "DEMO_ACTOR",
    "DEMO_BUNDLE_ENV",
    "DEMO_BUNDLE_NAME",
    "DEMO_SCENARIOS_FILE",
    "DEMO_WORKSPACE_NAME",
    "demo_bundle_candidates",
    "demo_bundle_path",
    "record_cassette",
    "recording_factory",
]

#: Name of the demo workspace (가상 회사 별빛상점, 상암 DMC).
DEMO_WORKSPACE_NAME: Final = "별빛상점 (DMC 데모)"
#: Directory name of the example bundle under ``examples/``.
DEMO_BUNDLE_NAME: Final = "dmc-shop"
#: Hand-written drills shipped with the bundle (resolved by the first diagnosis).
DEMO_SCENARIOS_FILE: Final = "scenarios.yaml"
#: Actor recorded in the audit log for demo set-up.
DEMO_ACTOR: Final = "demo"
#: Environment variable that points at another copy of the bundle (e.g. on the venue laptop).
DEMO_BUNDLE_ENV: Final = "JETTAE_DEMO_BUNDLE"


def demo_bundle_candidates() -> tuple[Path, ...]:
    """Where the example bundle may live, in the order :func:`demo_bundle_path` tries them.

    1. ``$JETTAE_DEMO_BUNDLE`` (an explicit override always wins),
    2. the source checkout (``<repo>/examples/dmc-shop`` next to ``src/``),
    3. package data (``jettae/examples/dmc-shop``, for a wheel that ships the example),
    4. ``./examples/dmc-shop`` relative to the working directory.
    """
    package_dir = Path(jettae.__file__).resolve().parent
    candidates: list[Path] = []
    override = os.environ.get(DEMO_BUNDLE_ENV, "").strip()
    if override:
        candidates.append(Path(override))
    candidates += [
        package_dir.parents[1] / "examples" / DEMO_BUNDLE_NAME,
        package_dir / "examples" / DEMO_BUNDLE_NAME,
        Path.cwd() / "examples" / DEMO_BUNDLE_NAME,
    ]
    return tuple(dict.fromkeys(candidates))


def demo_bundle_path() -> Path:
    """The first candidate that is a bundle (a directory with a ``runbooks/`` folder).

    Raises ``FileNotFoundError`` (Korean message listing every place looked at) otherwise. An
    override that does not point at a bundle is an error, not a reason to silently use another copy.
    """
    candidates = demo_bundle_candidates()
    override = os.environ.get(DEMO_BUNDLE_ENV, "").strip()
    for index, candidate in enumerate(candidates):
        if (candidate / "runbooks").is_dir():
            return candidate
        if index == 0 and override:
            raise FileNotFoundError(
                f"{DEMO_BUNDLE_ENV}가 가리키는 곳에 예제 묶음(runbooks/ 폴더)이 없습니다: "
                f"{candidate.as_posix()}"
            )
    looked = ", ".join(c.as_posix() for c in candidates)
    raise FileNotFoundError(
        f"예제 묶음 examples/{DEMO_BUNDLE_NAME}을 찾을 수 없습니다(찾아본 곳: {looked}). "
        f"{DEMO_BUNDLE_ENV} 환경 변수로 위치를 알려 주세요."
    )


def recording_factory(
    factory: Callable[[Settings, LlmMode], LLMClient], path: Path
) -> Callable[[Settings, LlmMode], LLMClient]:
    """Wrap ``factory`` so every LIVE client also appends its answers to the cassette ``path``.

    Cassette mode ``record``: the live model is always asked and every answer is appended (a later
    line wins on replay). OFF and REPLAY clients pass through unchanged.
    """

    def build(settings: Settings, mode: LlmMode) -> LLMClient:
        client = factory(settings, mode)
        return CassetteLLM(client, path, "record") if LlmMode(mode) is LlmMode.LIVE else client

    return build


async def record_cassette(service: JettaeService, path: str | Path) -> DiagnosisReport:
    """Run the demo diagnosis with a LIVE model and append every answer to the cassette ``path``.

    The live model is whatever ``service`` builds for LIVE (by default the configured provider via
    :func:`~jettae.llm.factory.build_llm`); its answers are recorded on the way through.

    The recording runs on a fresh in-memory store, never on ``service``'s own: a cassette replays
    only the requests it recorded, and the scenario and draft requests describe the graph they
    were asked about. Recorded against an operator's demo workspace that already has approved
    patches, they would describe a repaired graph, and the venue's fresh ``bootstrap_demo`` would
    miss every one of them — no AI drafts and no LLM drills on stage, without a single note.
    The returned report belongs to that throwaway store; ``service``'s store is not touched.

    Raises ``RuntimeError`` when nothing was recorded — no key or no provider means the run
    silently fell back to rules, and a venue demo relying on that empty cassette would show no
    LLM layer at all.
    """
    target = Path(path)
    settings = service.settings.model_copy(update={"cassette_path": target})
    recorder = type(service)(
        MemoryStore(clock=service.clock),
        settings,
        clock=service.clock,
        llm_factory=recording_factory(service.llm_factory, target),
    )
    report = await recorder.bootstrap_demo(llm_mode=LlmMode.LIVE)
    if report.llm.calls == 0:
        raise RuntimeError(
            "LLM 응답을 하나도 녹화하지 못했습니다. API 키(예: ANTHROPIC_API_KEY)와 "
            "JETTAE_LLM_PROVIDER 설정을 확인한 뒤 다시 녹화하세요."
        )
    return report
