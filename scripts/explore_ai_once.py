"""AI 위험 가설 비교 실험 (유료, 사용자 승인 2026-10-23) — 별빛상점 v5(결제 수정 4건 승인 후).

사용: python scripts/explore_ai_once.py <demo.db> [dry] [출력폴더]

A. 검증 엔진만: 고정 규칙 + 자동 전수 탐색 (LLM 0회)
B. 같은 그래프에 AI 위험 가설 1회 요청 (claude-sonnet-5-5, effort medium, n=5, max_tokens 4000)
C. 결정적 리허설로 검증 (확인됨 / 기각 / 확인 필요)
D. AI만의 추가 기여 = 검증 엔진이 보고하지 않았고 리허설로 확인된 가설 수

비용 상한: structured() 호출은 1번만 허용한다(내부 재요청 포함 최대 2회 API 호출).
키는 저장소의 .env에서 읽고 출력하지 않는다. ``dry``를 주면 LLM을 끄고 무료로 같은 경로를 확인한다.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from jettae.check.registry import run_checks
from jettae.config import Settings
from jettae.domain import SimConfig
from jettae.domain.inputs import LlmMode
from jettae.explore.ai import propose_hypotheses
from jettae.explore.explorer import finish
from jettae.explore.sweep import sweep
from jettae.llm.base import MeteredLLM, UsageMeter
from jettae.llm.factory import build_llm
from jettae.orchestrator.pipeline import build_context
from jettae.store.sql import SqlStore

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = "ws-527cfd5299cac19d"


class OneCall:
    """structured()를 한 번만 허용하는 비용 상한."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.provider = inner.provider
        self.model = inner.model
        self.n = 0

    async def structured(self, **kw: Any) -> Any:
        self.n += 1
        if self.n > 1:
            raise RuntimeError("cost cap: only one structured() call allowed")
        return await self.inner.structured(**kw)


def run(coro_factory: Any) -> Any:
    for _ in range(8):  # Windows 샌드박스에서 이벤트 루프 생성이 가끔 실패한다
        try:
            runner = asyncio.Runner()
            runner.get_loop()
        except OSError:
            continue
        with runner:
            return runner.run(coro_factory())
    raise RuntimeError("event loop could not be created")


def main() -> None:
    src_db = Path(sys.argv[1])  # 데모 DB(예: jettae demo가 만든 demo.db) — 사본만 연다
    dry = len(sys.argv) > 2 and sys.argv[2] == "dry"
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path.cwd()
    out.mkdir(parents=True, exist_ok=True)
    load_dotenv(REPO / ".env", override=False)  # ANTHROPIC_API_KEY 등 — 절대 출력하지 않음

    db = out / "demo_v5_copy.db"
    shutil.copyfile(src_db, db)
    store = SqlStore(f"sqlite:///{db.as_posix()}")
    inputs = store.inputs.load_inputs(WORKSPACE)
    versions = {v.number: v for v in store.graphs.list_versions(WORKSPACE)}
    graph = store.graphs.load_graph(versions[5].graph_hash)
    ctx = build_context(
        graph, roster=inputs.roster, oncall=inputs.oncall, catalog=inputs.catalog, policy=inputs.policy
    )
    findings = tuple(run_checks(ctx))
    sim = SimConfig()

    # A. 검증 엔진만
    swept = sweep(ctx, findings, config=sim)
    base = finish(ctx, swept, config=sim)

    settings = Settings(
        _env_file=None,
        llm_provider="anthropic",
        anthropic_model="claude-sonnet-5-5",
        anthropic_effort="medium",
    )
    meter = UsageMeter()
    mode = LlmMode.OFF if dry else LlmMode.LIVE
    llm = OneCall(MeteredLLM(build_llm(settings, mode), meter))

    started = time.perf_counter()
    proposals = run(lambda: propose_hypotheses(llm, ctx, findings, n=5, known=swept, config=sim))
    elapsed = time.perf_counter() - started
    # B + C. AI 가설까지 검증
    both = finish(ctx, swept, proposals=proposals, config=sim)

    base_ids = {h.id for h in base.hypotheses}
    ai_only = [h for h in both.hypotheses if h.id not in base_ids]
    result = {
        "model": settings.anthropic_model,
        "effort": settings.anthropic_effort,
        "n_requested": 5,
        "llm_calls": meter.calls,
        "input_tokens": meter.input_tokens,
        "output_tokens": meter.output_tokens,
        "cache_read_tokens": meter.cache_read_tokens,
        "seconds": round(elapsed, 1),
        "graph_hash": versions[5].graph_hash,
        "findings": len(findings),
        "A_engine_only": {
            "summary": base.summary.model_dump(mode="json"),
            "hypotheses": [h.model_dump(mode="json") for h in base.hypotheses],
        },
        "B_ai": {
            "status": str(proposals.status),
            "proposed": proposals.proposed,
            "kept": len(proposals.drafts),
            "dropped_invalid": proposals.dropped,
            "duplicates_of_engine": proposals.duplicates,
            "over_cap": proposals.over_cap,
            "drafts": [d.__dict__ for d in proposals.drafts],
        },
        "C_verified_ai_only": [h.model_dump(mode="json") for h in ai_only],
        "D_counts": {
            s: sum(1 for h in ai_only if str(h.status).endswith(s))
            for s in ("confirmed", "refuted", "needs_info")
        },
        "summary_with_ai": both.summary.model_dump(mode="json"),
    }
    name = "ai_v5_dry.json" if dry else "ai_v5.json"
    text = json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n"
    (out / name).write_text(text, encoding="utf-8")
    print(
        "calls", meter.calls, "in", meter.input_tokens, "out", meter.output_tokens, "sec", round(elapsed, 1)
    )
    print("status", proposals.status, "proposed", proposals.proposed, "kept", len(proposals.drafts))
    print("D", result["D_counts"])


if __name__ == "__main__":
    main()
