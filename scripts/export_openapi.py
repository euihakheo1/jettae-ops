"""Write the API's OpenAPI document to ``web/openapi.json`` (input of the TypeScript client generator).

Why a file in the repository: the web app's client is generated from it, so a change to the API
contract shows up as a reviewable diff next to the code that caused it, and CI can refuse a stale
copy (``--check``).

Usage (from the project root)::

    python scripts/export_openapi.py            # write web/openapi.json
    python scripts/export_openapi.py --check    # exit 1 if the file is out of date (CI)
    python scripts/export_openapi.py --output other.json

The document does not depend on the environment: the app is built with fixed settings (in-memory
store, no web bundle), and the JSON is written with sorted keys, two-space indent, UTF-8 and a
trailing newline, so the same code always produces the same bytes (P6).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from jettae.api import create_app
from jettae.config import Settings

ROOT = Path(__file__).absolute().parents[1]
DEFAULT_OUTPUT = ROOT / "web" / "openapi.json"


def openapi_json() -> str:
    """The OpenAPI document of :func:`jettae.api.create_app`, as stable JSON text."""
    settings = Settings(_env_file=None, database_url="memory://", web_dist=None, auth_mode="dev")
    document = create_app(settings).openapi()
    return json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="제때옵스 API의 OpenAPI 문서를 파일로 씁니다.")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="쓸 파일 (기본: web/openapi.json)"
    )
    parser.add_argument(
        "--check", action="store_true", help="파일이 최신인지 확인만 합니다(다르면 종료 코드 1)"
    )
    args = parser.parse_args(argv)
    output: Path = args.output
    text = openapi_json()
    if args.check:
        current = output.read_text(encoding="utf-8") if output.is_file() else None
        if current != text:
            print(f"{output}이(가) 최신이 아닙니다. python scripts/export_openapi.py 로 다시 만드세요.")
            return 1
        print(f"{output}: 최신입니다.")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(text.encode("utf-8"))
    print(f"{output}에 OpenAPI 문서를 썼습니다({len(text.encode('utf-8')):,} 바이트).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
