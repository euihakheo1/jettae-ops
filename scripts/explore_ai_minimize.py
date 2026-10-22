"""AI 위험 가설의 최소 실패 조건 자동 검증 — 기록된 AI 가설을 별빛상점 v5에서 다시 검증 (LLM 0회).

사용: python scripts/explore_ai_minimize.py <demo.db> [AI 기록 JSON] [출력 JSON]

* 데모 DB는 임시 폴더에 복사한 사본만 엽니다(원본은 열지 않음).
* AI 기록(기본: docs/eval/explore-ai/2026-10-23-dmc-v5.json)의 ``C_verified_ai_only``를 읽어, 승인 뒤
  재판정과 같은 경로(:func:`jettae.explore.explore_graph`)로 v5 그래프에서 다시 확인합니다. 확인됨 AI
  가설은 최소 실패 조건 자동 검증(:mod:`jettae.explore.minimize`)을 거칩니다.
* 출력의 ``verification_same_as_record``(AI 가설별)와 ``sweep_unchanged``(자동 전수 탐색, 기록의
  ``A_engine_only``와 비교)는 다시 확인해도 상태·리허설 기록이 바뀌지 않았는지를 적습니다.
* 이 스크립트는 LLM 클라이언트를 만들지 않습니다. 모델 호출은 0회입니다.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from jettae.check.registry import run_checks
from jettae.domain import HypothesisOrigin, HypothesisVerification, OpsGraph, Perturbation, RiskHypothesis
from jettae.explore import explore_graph
from jettae.explore.drafts import perturbation_label
from jettae.orchestrator.pipeline import build_context
from jettae.store.sql import SqlStore

ROOT = Path(__file__).absolute().parents[1]
DEFAULT_RECORD = ROOT / "docs" / "eval" / "explore-ai" / "2026-10-23-dmc-v5.json"
WORKSPACE = "ws-527cfd5299cac19d"  # 결선 데모 DB의 별빛상점 워크스페이스
VERSION = 5  # 결제 수정안 4건 승인 후


def _labels(graph: OpsGraph, perturbations: Sequence[Perturbation]) -> list[str]:
    return [perturbation_label(graph, p) for p in perturbations]


def _settled(hypothesis: RiskHypothesis) -> tuple[str, str, HypothesisVerification]:
    """What a re-check must not change: id, status and the replay record."""
    return hypothesis.id, hypothesis.status.value, hypothesis.verification


def _row(graph: OpsGraph, hypothesis: RiskHypothesis, recorded: RiskHypothesis | None) -> dict[str, Any]:
    v = hypothesis.verification
    row: dict[str, Any] = {
        "id": hypothesis.id,
        "service_id": hypothesis.service_id,
        "conditions": _labels(graph, hypothesis.perturbations),
        "assumptions": list(hypothesis.assumptions),
        "status_recorded": recorded.status.value if recorded is not None else None,
        "status": hypothesis.status.value,
        # 기록된 리허설 결과(재생 시나리오·결과 해시·끊긴 지점·사유·시한 전체)와 같은가
        "verification_same_as_record": recorded is not None and recorded.verification == v,
        "outcome": v.outcome.value,
        "failed_element": v.failed_element.value if v.failed_element is not None else None,
        "failed_at_id": v.failed_at_id,
        "reason": v.reason,
        "deadline_minutes": v.deadline_minutes,
        "minimization": None,
    }
    m = hypothesis.minimization
    if m is not None:
        row["minimization"] = {
            "replays": m.replays,
            "original_minimal": m.original_minimal,
            "minimal_sets": [
                {
                    "conditions": _labels(graph, s.perturbations),
                    "failed_element": s.failed_element.value if s.failed_element is not None else None,
                    "failed_at_id": s.failed_at_id,
                    "reason": s.reason,
                    "point_differs": s.point_differs,
                    "result_hash": s.result_hash,
                }
                for s in m.minimal_sets
            ],
            "unnecessary": _labels(graph, m.unnecessary),
            "other_failures": [
                {
                    "conditions": _labels(graph, s.perturbations),
                    "failed_element": s.failed_element.value if s.failed_element is not None else None,
                    "failed_at_id": s.failed_at_id,
                    "reason": s.reason,
                }
                for s in m.other_failures
            ],
            "raw": m.model_dump(mode="json"),
        }
    return row


def main(argv: Sequence[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    source = Path(argv[0])
    record_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_RECORD
    output = Path(argv[2]) if len(argv) > 2 else Path.cwd() / "minimize_v5.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    recorded = [RiskHypothesis.model_validate(h) for h in record["C_verified_ai_only"]]
    engine_only = [RiskHypothesis.model_validate(h) for h in record["A_engine_only"]["hypotheses"]]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        copy = Path(tmp) / "demo_copy.db"
        shutil.copyfile(source, copy)
        store = SqlStore(f"sqlite:///{copy.as_posix()}")
        try:
            inputs = store.inputs.load_inputs(WORKSPACE)
            version = {v.number: v for v in store.graphs.list_versions(WORKSPACE)}[VERSION]
            graph = store.graphs.load_graph(version.graph_hash)
        finally:
            store.close()
    ctx = build_context(
        graph, roster=inputs.roster, oncall=inputs.oncall, catalog=inputs.catalog, policy=inputs.policy
    )
    findings = tuple(run_checks(ctx))
    again = explore_graph(ctx, findings, previous=recorded)
    by_id = {h.id: h for h in recorded}
    ai = [h for h in again.hypotheses if h.origin is HypothesisOrigin.AI]
    swept = [h for h in again.hypotheses if h.origin is HypothesisOrigin.SWEEP]
    result = {
        "method": "최소 실패 조건 자동 검증 (jettae.explore.minimize, LLM 0회)",
        "graph_hash": version.graph_hash,
        "recorded_graph_hash": record.get("graph_hash"),
        "record": record_path.name,
        "findings": len(findings),
        # 자동 전수 탐색 가설이 기록(A. 검증 엔진만)과 id·상태·리허설 기록까지 같은가
        "sweep_hypotheses": len(swept),
        "sweep_unchanged": [_settled(h) for h in swept] == [_settled(h) for h in engine_only],
        "ai_hypotheses": [_row(graph, h, by_id.get(h.id)) for h in ai],
        "minimize_replays": sum(h.minimization.replays for h in ai if h.minimization is not None),
        "notes": list(again.summary.notes),
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in result["ai_hypotheses"]:
        m = row["minimization"]
        line = f"{row['status']:<10} {row['service_id']} | {' + '.join(row['conditions'])}"
        if m is not None:
            sets = " / ".join(" + ".join(s["conditions"]) for s in m["minimal_sets"])
            line += f" → 최소 실패 조건: {sets} (불필요: {', '.join(m['unnecessary']) or '없음'})"
        print(line)
    print(f"{output}에 썼습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
