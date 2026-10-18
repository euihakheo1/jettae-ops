"""Score the blinded human review (10/22) against the two LLM coders.

Usage:
    python -I scripts/realdata_score_review.py <labels> <review_key.json> [--json out.json]

<labels> is either the JSON the review page copies ("판정 JSON 복사": {item_id: answers}) or a directory
of exported documents (one <item_id>.json per document, raw answers or wrapped as {"data": answers}).
Standard library only. Prints a Markdown summary; nothing here calls a model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

OUT = ("out_of_scope", "not_a_gap")
Labels = dict[str, dict]
Key = dict[str, dict]


def load_labels(path: Path) -> Labels:
    if not path.is_dir():
        return json.loads(path.read_text(encoding="utf-8"))
    labels = {}
    for f in sorted(path.rglob("*.json")):
        doc = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(doc, dict):
            labels[doc.get("id", f.stem)] = doc.get("data", doc)
    return labels


def wilson(hits: int, n: int, z: float = 1.959964) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = hits / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def s1_label(ans: dict) -> str | None:
    """Same rule as the coders (premise-claim-v4): both yes -> correct, any no -> incorrect."""
    p, c = ans.get("premise_valid"), ans.get("claim_true")
    if not p:
        return None
    if p == "no" or c == "no":
        return "incorrect"
    if not c:
        return None
    return "correct" if p == "yes" and c == "yes" else "unclear"


def fam(code: str | None) -> str | None:
    return code.split(":", 1)[0] if code else code


def rate(hits: int, n: int) -> str:
    ci = wilson(hits, n)
    if not ci:
        return f"{hits}/{n}"
    return f"{hits}/{n} = {hits / n:.1%} (95% CI {ci[0]:.1%}–{ci[1]:.1%})"


def agreement(pairs: list[tuple[str, str]]) -> str:
    k = kappa(pairs)
    same = sum(x == y for x, y in pairs)
    return f"{same}/{len(pairs)} · κ {'—' if k is None else f'{k:.3f}'}"


def score_s1(labels: Labels, key: Key) -> tuple[list[str], dict]:
    s1 = {i: k for i, k in key.items() if k["part"] == "S1"}
    done = {i: h for i in s1 if (h := s1_label(labels.get(i, {})))}
    c = Counter(done.values())
    decided = c["correct"] + c["incorrect"]
    lines = [
        "## S1 — 실증 ① test 지적",
        f"- 판정한 항목 {len(done)}/{len(s1)} · {dict(c)}",
        f"- 사람 기준 정밀도: {rate(c['correct'], decided)}",
    ]
    out: dict = {"answered": len(done), "counts": dict(c), "precision": [c["correct"], decided]}
    for who in ("a", "b"):
        coder = {i: s1[i][f"coder_{who}"] for i in done}
        pairs = [(done[i], coder[i]["label"]) for i in done if coder[i].get("label")]
        lines.append(f"- 사람–코더 {who.upper()} 판정 일치 {agreement(pairs)}")
        out[f"kappa_{who}"] = kappa(pairs)
        for q in ("premise_valid", "claim_true"):
            qp = [(labels[i][q], coder[i][q]) for i in done if labels[i].get(q) and coder[i].get(q)]
            out[f"kappa_{who}_{q}"] = kappa(qp)
    consensus = [
        i
        for i in done
        if s1[i]["coder_a"].get("label") == s1[i]["coder_b"].get("label") in ("correct", "incorrect")
    ]
    same = sum(done[i] == s1[i]["coder_a"]["label"] for i in consensus)
    lines.append(f"- 두 코더가 같은 판정(맞음/틀림)을 낸 {len(consensus)}건 중 사람도 같은 판정: {same}")
    out["agrees_with_coder_consensus"] = [same, len(consensus)]
    return lines, out


def score_s2(labels: Labels, key: Key) -> tuple[list[str], dict]:
    s2 = {i: k for i, k in key.items() if k["part"] == "S2"}
    ans = {i: labels.get(i, {}) for i in s2}
    fams = {i: x["a_family"] for i, x in ans.items() if x.get("a_family")}
    in_scope = [i for i, f in fams.items() if f not in OUT]
    strict = Counter(ans[i]["b_strict_match"] for i in s2 if ans[i].get("b_strict_match"))
    strict_in = Counter(ans[i]["b_strict_match"] for i in in_scope if ans[i].get("b_strict_match"))
    lenient = sum(1 for k in s2.values() if k["matched"])
    lines = [
        "## S2 — 실증 ② 범위 안 공백 41개",
        f"- (가) 계열 코딩 {len(fams)}/{len(s2)} · 사람도 범위 안으로 본 것 {len(in_scope)}",
        f"- (나) 엄격 일치(같은 서비스·같은 요소) 전체 {dict(strict)}: "
        f"{rate(strict['yes'], strict['yes'] + strict['no'])}",
        f"- (나) 사람이 범위 안으로 본 공백만: {rate(strict_in['yes'], strict_in['yes'] + strict_in['no'])}",
        f"- 참고: 자동 계열·요소 일치(느슨한 규칙) {lenient}/{len(s2)}",
    ]
    out: dict = {"family_answered": len(fams), "human_in_scope": len(in_scope)}
    out |= {"strict": dict(strict), "strict_in_scope": dict(strict_in)}
    for who in ("a", "b"):
        pairs = [(fam(fams[i]), s2[i][f"family_{who}"]) for i in fams if s2[i].get(f"family_{who}")]
        lines.append(f"- 사람–코더 {who.upper()} 계열 일치(계열 수준) {agreement(pairs)}")
        out[f"family_kappa_{who}"] = kappa(pairs)
    return lines, out


def score_s3(labels: Labels, key: Key) -> tuple[list[str], dict]:
    s3 = {i: k for i, k in key.items() if k["part"] == "S3"}
    fams = {i: f for i in s3 if (f := labels.get(i, {}).get("family"))}
    names = {"out_of_scope": "범위 밖", "not_a_gap": "공백 아님"}
    c = Counter(names.get(f, "범위 안") for f in fams.values())
    lines = [
        "## S3 — 범위 밖 표본 20개",
        f"- 답한 항목 {len(fams)}/{len(s3)} · {dict(c)}",
        f"- 사람이 범위 안으로 본 비율: {rate(c['범위 안'], len(fams))}",
    ]
    out: dict = {"answered": len(fams), "counts": dict(c)}
    for who in ("a", "b"):
        pairs = [(fam(fams[i]), s3[i][f"family_{who}"]) for i in fams if s3[i].get(f"family_{who}")]
        lines.append(f"- 사람–코더 {who.upper()} 계열 일치 {agreement(pairs)}")
        out[f"family_kappa_{who}"] = kappa(pairs)
    return lines, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", type=Path)
    ap.add_argument("key", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    labels = load_labels(a.labels)
    key: Key = json.loads(a.key.read_text(encoding="utf-8"))["items"]
    lines = ["# 사람 블라인드 검토 채점 (개발자 1인)", ""]
    out = {}
    for part, scorer in (("S1", score_s1), ("S2", score_s2), ("S3", score_s3)):
        part_lines, out[part] = scorer(labels, key)
        lines += [*part_lines, ""]
    missing = [i for i in key if i not in labels]
    shown = f": {', '.join(missing[:10])}" if missing else ""
    lines.append(f"- 저장되지 않은 항목 {len(missing)}개{shown}")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(lines))
    if a.json:
        a.json.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
