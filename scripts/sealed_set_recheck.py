"""Re-check the sealed set (seed 20261012) against the published seal with the current code.

Usage (repository root, product venv):
    python -I scripts/sealed_set_recheck.py [--metrics <metrics.json from a fresh `jettae eval run`>]

The freeze (commit ccf3b4d) added a jurisdiction field to the policy model, so every generated case's
policy YAML now also states the default `jurisdictions: [KR]` / `jurisdiction: KR`. That changes the
case hashes (and the Merkle root) without changing what any case says. This script regenerates the
1,320 cases, drops exactly those two default lines, and checks the hashes against
docs/eval/manifest.json. With --metrics it also compares a fresh run's metrics with the published
docs/eval/metrics.json, ignoring latency. Standard library plus the jettae package; no model calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import jettae.evalkit as kit
from jettae.evalkit.seal import build_manifest, merkle_root

ROOT = Path(__file__).resolve().parents[1]
SEED, ORGS = 20261012, 20
ADDED_LINES = ("jurisdictions:\n- KR\n", "  jurisdiction: KR\n")


def legacy_hash(case: kit.EvalCase) -> str:
    body = case.model_dump(mode="json")
    for line in ADDED_LINES:
        body["policy_yaml"] = body["policy_yaml"].replace(line, "")
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def without_latency(metrics: dict) -> dict:
    conditions = {k: {f: v for f, v in c.items() if f != "latency"} for k, c in metrics["conditions"].items()}
    keep = ("by_style", "by_operator", "by_position", "clean_fp")
    keep += ("hallucination", "field_f1", "ungrounded_total")
    return {"conditions": conditions, **{k: metrics.get(k) for k in keep}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, help="metrics.json of a fresh A,B,O run on seed 20261012")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    sealed = json.loads((ROOT / "docs/eval/manifest.json").read_text(encoding="utf-8"))
    cases = kit.build_cases(kit.generate_gold_orgs(ORGS, SEED), seed=SEED)
    current = build_manifest(cases)
    legacy = {legacy_hash(c) for c in cases}
    same_cases = legacy == set(sealed["case_hashes"])
    print(f"cases generated: {len(cases)} (seed {SEED})")
    print(f"current-code Merkle root: {current.merkle_root[:16]}… (published {sealed['merkle_root'][:16]}…)")
    leaves_ok = merkle_root(sealed["case_hashes"]) == sealed["merkle_root"]
    print(f"published seal re-verified from its own leaves: {leaves_ok}")
    print(f"every case equals the sealed one once the two default policy lines are dropped: {same_cases}")
    ok = same_cases
    if a.metrics:
        fresh = json.loads(a.metrics.read_text(encoding="utf-8"))
        published = json.loads((ROOT / "docs/eval/metrics.json").read_text(encoding="utf-8"))
        same_metrics = json.dumps(without_latency(fresh), sort_keys=True) == json.dumps(
            without_latency(published), sort_keys=True
        )
        print(f"fresh metrics equal the published ones (latency ignored): {same_metrics}")
        ok = ok and same_metrics
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
