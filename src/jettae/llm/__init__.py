"""LLM adapters behind one port (:class:`LLMClient`). Generation only — never judgement (P1).

Every adapter takes trusted ``instructions`` and untrusted ``documents`` separately, forces a
strict JSON schema, and signals "no model available" with :class:`LLMDisabledError` so callers
fall back to deterministic code (P4).
"""

from jettae.llm.anthropic_client import DEFAULT_MODEL, FALLBACK_BETA, AnthropicLLM, Effort
from jettae.llm.base import (
    LLMClient,
    LLMDisabledError,
    LLMError,
    LLMOutputError,
    LLMRefusalError,
    LLMResult,
    LLMUsage,
    MeteredLLM,
    NullLLM,
    UntrustedDocument,
    UsageMeter,
)
from jettae.llm.factory import build_llm
from jettae.llm.fake import FakeCall, FakeHandler, FakeLLM
from jettae.llm.openai_compat import EMPTY_API_KEY, OpenAICompatLLM, json_schema_response_format
from jettae.llm.prompting import (
    SAFETY_PREAMBLE,
    Reply,
    build_system,
    complete_with_repair,
    describe_validation_error,
    document_nonce,
    neutralise_boundaries,
    render_repair_message,
    render_user_content,
    replace_lone_surrogates,
)
from jettae.llm.replay import CASSETTE_PROVIDER, CassetteLLM, CassetteMode, canonical_json, cassette_key
from jettae.llm.schema import (
    StrictSchemaError,
    drop_absent_nulls,
    parse_output,
    to_strict_json_schema,
    validate_output,
)

__all__ = [
    "CASSETTE_PROVIDER",
    "DEFAULT_MODEL",
    "EMPTY_API_KEY",
    "FALLBACK_BETA",
    "SAFETY_PREAMBLE",
    "AnthropicLLM",
    "CassetteLLM",
    "CassetteMode",
    "Effort",
    "FakeCall",
    "FakeHandler",
    "FakeLLM",
    "LLMClient",
    "LLMDisabledError",
    "LLMError",
    "LLMOutputError",
    "LLMRefusalError",
    "LLMResult",
    "LLMUsage",
    "MeteredLLM",
    "NullLLM",
    "OpenAICompatLLM",
    "Reply",
    "StrictSchemaError",
    "UntrustedDocument",
    "UsageMeter",
    "build_llm",
    "build_system",
    "canonical_json",
    "cassette_key",
    "complete_with_repair",
    "describe_validation_error",
    "document_nonce",
    "drop_absent_nulls",
    "json_schema_response_format",
    "neutralise_boundaries",
    "parse_output",
    "render_repair_message",
    "render_user_content",
    "replace_lone_surrogates",
    "to_strict_json_schema",
    "validate_output",
]
