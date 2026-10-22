"""AI가 확인한 '추천: 강현우 부재 + 상품 카탈로그 동시 장애'의 원인별 분리 실험 (LLM 0회, 결정적 리허설).

사용: python scripts/explore_ai_ablation.py <demo.db 사본> <ai_v5.json> [출력 파일]

A. 교란 없음(기준) · B. 강현우만 부재 · C. 상품 카탈로그 동시 장애만 · D. 둘 다(AI 가설)
E. 박재현만 부재 · F. 강현우 + 박재현 부재 (동시 장애가 실제로 끌어가는 사람으로 바꾼 조합)
같은 일을 자동으로 하는 것이 jettae.explore.minimize(최소 실패 조건 자동 검증)다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from jettae.domain import RiskHypothesis, SimConfig
from jettae.domain.rehearsal import Perturbation, PerturbationKind
from jettae.explore.drafts import HypothesisDraft, draft_from_hypothesis
from jettae.explore.verify import verify_hypotheses
from jettae.orchestrator.pipeline import build_context
from jettae.store.sql import SqlStore

WORKSPACE = "ws-527cfd5299cac19d"


def perturbation(kind: str, target: str) -> Perturbation:
    return Perturbation(kind=PerturbationKind(kind), target_id=target)


def person(name: str) -> Perturbation:
    return perturbation("person_unavailable", f"person:{name}")


def main() -> None:
    store = SqlStore("sqlite:///" + Path(sys.argv[1]).as_posix())
    inputs = store.inputs.load_inputs(WORKSPACE)
    v5 = {v.number: v for v in store.graphs.list_versions(WORKSPACE)}[5]
    graph = store.graphs.load_graph(v5.graph_hash)
    ctx = build_context(
        graph, roster=inputs.roster, oncall=inputs.oncall, catalog=inputs.catalog, policy=inputs.policy
    )
    with Path(sys.argv[2]).open(encoding="utf8") as fh:
        ai = json.load(fh)
    [confirmed] = [h for h in ai["C_verified_ai_only"] if h["status"] == "confirmed"]
    base = draft_from_hypothesis(RiskHypothesis.model_validate(confirmed))
    cases = {
        "B 강현우만 부재": (person("강현우"),),
        "C 상품 카탈로그 동시 장애만": (perturbation("concurrent_incident", "service:상품-카탈로그"),),
        "D 둘 다 (AI 가설)": base.perturbations,
        "E 박재현만 부재": (person("박재현"),),
        "F 강현우 + 박재현 부재": (person("강현우"), person("박재현")),
    }
    out: dict[str, object] = {"service": base.service_id, "graph_hash": v5.graph_hash}
    results: dict[str, dict[str, object]] = {}
    for label, perts in cases.items():
        draft = HypothesisDraft(
            origin=base.origin,
            service_id=base.service_id,
            title=label,
            explanation="",
            perturbations=tuple(perts),
            incident_type=base.incident_type,
        )
        [h] = verify_hypotheses(ctx, [draft], config=SimConfig())
        v = h.verification.model_dump(mode="json") if h.verification else {}
        results[label] = {
            "status": str(h.status),
            "outcome": v.get("outcome"),
            "baseline": v.get("baseline_outcome"),
            "failed_at": v.get("failed_at_id"),
            "reason": v.get("reason"),
            "deadline": v.get("deadline_minutes"),
            "time_to_mitigate": v.get("time_to_mitigate"),
        }
        print(label, "|", results[label])
    out["cases"] = results
    target = Path(sys.argv[3]) if len(sys.argv) > 3 else Path.cwd() / "ablation_v5.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
