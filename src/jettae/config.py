"""Runtime settings (env prefix ``JETTAE_``; ``.env`` supported)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from jettae.domain.inputs import LlmMode

__all__ = ["MAX_XFF_HOPS", "ClientIpSource", "Settings", "get_settings", "parse_client_ip_source"]

#: ``xff:<n>`` may count at most this many proxies from the right of ``X-Forwarded-For``.
MAX_XFF_HOPS: Final = 10
#: an HTTP header name (RFC 9110 token), kept short: it is only ever a proxy's client-IP header
_HEADER_NAME: Final = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]{1,64}")


@dataclass(frozen=True, slots=True)
class ClientIpSource:
    """Where the per-client limits read the visitor's address (``JETTAE_CLIENT_IP_SOURCE``).

    * ``peer`` — the transport's client address as uvicorn reports it (behind a proxy and with
      ``FORWARDED_ALLOW_IPS``, uvicorn's choice from ``X-Forwarded-For``);
    * ``header:<Header-Name>`` — one address from that header (e.g. ``header:CF-Connecting-IP``);
    * ``xff:<n>`` — the ``n``-th entry of ``X-Forwarded-For`` counted from the right (1 = the
      entry the nearest proxy appended, which a client cannot forge).
    """

    kind: Literal["peer", "header", "xff"]
    #: the header of ``header:<name>`` (as configured), else None
    header: str | None = None
    #: ``n`` of ``xff:<n>``, else None
    hops: int | None = None

    def __str__(self) -> str:
        if self.kind == "header":
            return f"header:{self.header}"
        if self.kind == "xff":
            return f"xff:{self.hops}"
        return "peer"


def parse_client_ip_source(value: str) -> ClientIpSource:
    """Parse ``JETTAE_CLIENT_IP_SOURCE`` (see :class:`ClientIpSource`); ``ValueError`` in Korean."""
    text = value.strip()
    kind, sep, arg = text.partition(":")
    kind = kind.strip().casefold()
    arg = arg.strip()
    if kind == "peer" and not sep:
        return ClientIpSource(kind="peer")
    if kind == "header" and _HEADER_NAME.fullmatch(arg):
        return ClientIpSource(kind="header", header=arg)
    if kind == "xff" and arg.isascii() and arg.isdigit() and 1 <= int(arg) <= MAX_XFF_HOPS:
        return ClientIpSource(kind="xff", hops=int(arg))
    shown = text[:40]
    raise ValueError(
        f"JETTAE_CLIENT_IP_SOURCE 값 '{shown}'을(를) 알 수 없습니다. "
        f"peer, header:<헤더 이름>(예: header:CF-Connecting-IP), xff:<1~{MAX_XFF_HOPS}>"
        "(X-Forwarded-For의 오른쪽에서 n번째) 중 하나로 적어 주세요."
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JETTAE_", env_file=".env", extra="ignore")

    # storage
    database_url: str = "sqlite:///./data/jettae.db"
    store_raw_documents: bool = Field(
        default=False, description="원문 비저장 기본값. True면 마스킹된 원문을 DB에 보관(근거 하이라이트용)"
    )

    # LLM
    llm_mode: LlmMode = LlmMode.OFF
    llm_provider: Literal["anthropic", "openai"] = "anthropic"
    anthropic_model: str = "claude-opus-5-5"
    anthropic_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    openai_base_url: str | None = Field(default=None, description="vLLM 등 OpenAI 호환 엔드포인트")
    openai_model: str = "local-model"
    openai_api_key: SecretStr | None = None
    cassette_path: Path = Path("./data/cassettes/llm.jsonl")
    llm_concurrency: int = 4
    llm_timeout_seconds: float = 120.0

    # API / auth
    #: dev(헤더로 역할 지정, 로컬 전용) · token(Bearer 토큰 필수) · public(공개 베타: 토큰 없으면 익명 viewer)
    auth_mode: Literal["dev", "token", "public"] = "dev"
    #: JSON mapping token -> {"user": "...", "role": "viewer|editor|lead|admin"}
    api_tokens: dict[str, dict[str, str]] = Field(default_factory=dict)
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    web_dist: Path | None = Path("web/dist")
    max_upload_mb: int = 20
    #: text length one uploaded runbook may have after conversion (None = no cap). Applies to the
    #: anonymous free scan (each runbook, and all of its runbooks together) and to uploads by
    #: callers below editor; the diagnosis cost grows faster than the text does.
    max_document_chars: int | None = Field(default=None, ge=1)
    daily_free_scan_limit: int = 200

    # public-site abuse limits (keyed by the client address JETTAE_CLIENT_IP_SOURCE picks)
    #: where the visitor's address comes from: peer · header:<Header-Name> · xff:<n> (see
    #: :class:`ClientIpSource`); verify on the host with GET /api/v1/admin/request-info
    client_ip_source: str = Field(
        default="peer",
        description="접속 주소를 읽을 곳: peer · header:<헤더 이름> · xff:<n>(X-Forwarded-For 오른쪽 n번째)",
    )
    #: diagnoses that may run at once for viewers and anonymous callers; more get 503 + Retry-After
    max_concurrent_scans: int = Field(default=2, ge=1)
    #: free scans one client address may use per Korean calendar day (inside the daily limit)
    free_scans_per_client_per_day: int = Field(default=20, ge=1)
    events_per_client_per_minute: int = Field(default=60, ge=1)
    signups_per_client_per_hour: int = Field(default=10, ge=1)
    #: demand events (POST /events) stored per Korean calendar day, site-wide; the rest are still
    #: answered 202 but dropped (0 = store none). Bounds the events table whatever the per-client
    #: limit lets through.
    events_per_day: int = Field(default=20_000, ge=0)

    # retention (0 = keep forever)
    #: days a free scan's private workspace is kept before the periodic purge deletes it
    free_scan_retention_days: int = Field(default=7, ge=0)
    #: days a pre-launch sign-up is kept (the consent text promises one year)
    signup_retention_days: int = Field(default=365, ge=0)

    # observability
    otel_enabled: bool = False
    log_json: bool = False

    @field_validator("client_ip_source")
    @classmethod
    def _client_ip_source(cls, value: str) -> str:
        """Refused at start-up (Korean message), stored normalised (``XFF:2`` → ``xff:2``)."""
        return str(parse_client_ip_source(value))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
