"""API keys for the paid studies: the environment first, then a ``.env`` file (no new dependency).

Keys are returned to the caller and handed straight to the SDK constructor — never written to
``os.environ``, never logged, never part of an exception message. A missing key is ``None``; the
caller decides whether that is fatal (a real run) or only a downgrade (a dry run then estimates
Claude input tokens with a character heuristic instead of ``count_tokens``).
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Final

__all__ = ["ANTHROPIC_KEY", "OPENAI_KEY", "default_dotenv_paths", "load_api_key", "parse_dotenv"]

ANTHROPIC_KEY: Final = "ANTHROPIC_API_KEY"
OPENAI_KEY: Final = "OPENAI_API_KEY"
_QUOTES: Final = ("'", '"')


def default_dotenv_paths() -> tuple[Path, ...]:
    """``.env`` in the working directory, then the one at the repository root (``jettae-ops/.env``)."""
    repo_root = Path(__file__).resolve().parents[4]
    candidates = (Path.cwd() / ".env", repo_root / ".env")
    unique: dict[str, Path] = {}
    for candidate in candidates:
        unique.setdefault(os.path.normcase(str(candidate.resolve())), candidate)
    return tuple(unique.values())


def parse_dotenv(text: str) -> dict[str, str]:
    """``KEY=value`` lines → dict. Supports ``export KEY=…``, quoted values and comments.

    Deliberately small: no interpolation, no multi-line values. The last assignment wins, like a
    shell sourcing the file. Malformed lines are skipped (a key file must not crash a dry run).
    """
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip().removeprefix("﻿")
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not key.replace("_", "").isalnum():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] in _QUOTES and value[-1] == value[0]:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        values[key] = value
    return values


def load_api_key(
    name: str,
    *,
    environ: Mapping[str, str] | None = None,
    dotenv_paths: Iterable[Path] | None = None,
) -> str | None:
    """The key ``name`` from the environment, else from the first ``.env`` that defines it."""
    env = os.environ if environ is None else environ
    value = env.get(name, "").strip()
    if value:
        return value
    for path in default_dotenv_paths() if dotenv_paths is None else dotenv_paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        found = parse_dotenv(text).get(name, "").strip()
        if found:
            return found
    return None
