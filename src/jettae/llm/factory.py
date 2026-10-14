"""Build the :class:`LLMClient` for a run from settings + the requested :class:`LlmMode`.

The mode decides *whether* a model is consulted (OFF / LIVE / REPLAY); settings decide *which*.
Keeping that mapping in one place is what makes "LLM을 꺼도 동작" (P4) and deterministic replay
(P6) configuration rather than code paths scattered across callers.
"""

from __future__ import annotations

from jettae.config import Settings
from jettae.domain.inputs import LlmMode
from jettae.llm.anthropic_client import AnthropicLLM
from jettae.llm.base import LLMClient, MeteredLLM, NullLLM, UsageMeter
from jettae.llm.openai_compat import OpenAICompatLLM
from jettae.llm.replay import CassetteLLM


def build_llm(
    settings: Settings, mode: LlmMode, meter: UsageMeter | None = None, *, record: bool = False
) -> LLMClient:
    """Return the client for ``mode``, wrapped in :class:`MeteredLLM` when ``meter`` is given.

    * ``OFF``    → :class:`NullLLM` (every call raises ``LLMDisabledError`` → rule fallback).
    * ``LIVE``   → the configured provider; with ``record=True`` every live call is also
      appended to ``settings.cassette_path`` (mode ``record``: live answers always win).
    * ``REPLAY`` → :class:`CassetteLLM` over ``settings.cassette_path``; misses fall back.

    Construction never touches the network and never fails for missing credentials; that
    surfaces as ``LLMDisabledError`` on the first call so callers can fall back.
    """
    client = _client_for(settings, mode, record=record)
    return MeteredLLM(client, meter) if meter is not None else client


def _client_for(settings: Settings, mode: LlmMode, *, record: bool) -> LLMClient:
    # Coerce first: a raw "off" from an API payload or CLI flag must never fall through to the
    # live branch (that would send runbooks to a model the user switched off), and an unknown
    # value is a configuration error, not "live".
    match LlmMode(mode):
        case LlmMode.OFF:
            return NullLLM()
        case LlmMode.REPLAY:
            return CassetteLLM(None, settings.cassette_path, "replay")
        case LlmMode.LIVE:
            live = _live_client(settings)
            return CassetteLLM(live, settings.cassette_path, "record") if record else live


def _live_client(settings: Settings) -> LLMClient:
    if settings.llm_provider == "anthropic":
        return AnthropicLLM(
            settings.anthropic_model, settings.anthropic_effort, timeout=settings.llm_timeout_seconds
        )
    api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key is not None else None
    return OpenAICompatLLM(
        settings.openai_base_url, settings.openai_model, api_key, timeout=settings.llm_timeout_seconds
    )
