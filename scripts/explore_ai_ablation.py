"""AI가 확인한 '추천: 강현우 부재 + 상품 카탈로그 동시 장애'의 원인별 분리 실험 (LLM 0회, 결정적 리허설).

A. 교란 없음(기준) · B. 강현우만 부재 · C. 상품 카탈로그 동시 장애만 · D. 둘 다(AI 가설)
E. 박재현만 부재 · F. 강현우 + 박재현 부재 (동시 장애가 실제로 끌어가는 사람으로 바꾼 조합)
"""
import json, sys
from pathlib import Path

from jettae.check.registry import run_checks
from jettae.domain import RiskHypothesis, SimConfig
from jettae.domain.rehearsal import Perturbation, PerturbationKind
from jettae.explore.drafts import HypothesisDraft, draft_from_hypothesis
from jettae.explore.verify import verify_hypotheses
from jettae.orchestrator.pipeline import build_context
from jettae.store.sql import SqlStore

HERE = Path(__file__).parent
store = SqlStore("sqlite:///" + (Path(sys.argv[1])).as_posix())
ws = "ws-527cfd5299cac19d"
inputs = store.inputs.load_inputs(ws)
v5 = {v.number: v for v in store.graphs.list_versions(ws)}[5]
graph = store.graphs.load_graph(v5.graph_hash)
ctx = build_context(graph, roster=inputs.roster, oncall=inputs.oncall, catalog=inputs.catalog, policy=inputs.policy)

ai = json.load(open(Path(sys.argv[2]), encoding="utf8"))
[confirmed] = [h for h in ai["C_verified_ai_only"] if h["status"] == "confirmed"]
base = draft_from_hypothesis(RiskHypothesis.model_validate(confirmed))
P = lambda kind, target: Perturbation(kind=PerturbationKind(kind), target_id=target)
person = lambda name: P("person_unavailable", f"person:{name}")
cases = {
    "B 강현우만 부재": (person("강현우"),),
    "C 상품 카탈로그 동시 장애만": (P("concurrent_incident", "service:상품-카탈로그"),),
    "D 둘 다 (AI 가설)": base.perturbations,
    "E 박재현만 부재": (person("박재현"),),
    "F 강현우 + 박재현 부재": (person("강현우"), person("박재현")),
}
out = {"service": base.service_id, "graph_hash": v5.graph_hash, "cases": {}}
for label, perts in cases.items():
    draft = HypothesisDraft(origin=base.origin, service_id=base.service_id, title=label, explanation="",
                            perturbations=tuple(perts), incident_type=base.incident_type)
    [h] = verify_hypotheses(ctx, [draft], config=SimConfig())
    v = h.verification.model_dump(mode="json") if h.verification else {}
    out["cases"][label] = {"status": str(h.status), "outcome": v.get("outcome"), "baseline": v.get("baseline_outcome"),
                           "failed_at": v.get("failed_at_id"), "reason": v.get("reason"), "deadline": v.get("deadline_minutes"),
                           "time_to_mitigate": v.get("time_to_mitigate")}
    print(label, "|", out["cases"][label])
(Path(sys.argv[3]) if len(sys.argv) > 3 else Path.cwd() / "ablation_v5.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
