"""Chance baseline of study 2 (실증 ②): does m/k beat the findings of a *different* incident?

Reads a retro ``results.jsonl`` (one incident per line, as ``jettae eval retro`` writes it) and,
with the standard library only:

1. re-derives every gap unit's ``is_gap`` / ``in_scope`` / ``matched`` with the deterministic rule
   of ``jettae.evalkit.realdata.retro`` (``_finding_matches`` and ``match_units``, mirrored below
   line for line) and checks the result against the values stored in the file;
2. N, k, m overall, Korean (KR) vs overseas, and per rule family (grouped as the report does);
3. the cross-incident (chance) baseline: every in-scope unit is matched, with the same rule,
   against the findings of every OTHER incident. The mean share is what m/k would be if the
   findings had nothing to do with the incident they were made for. The primary comparison uses
   only the incidents the product actually diagnosed (an undiagnosed bundle has no findings and
   can never match); the harness m/k against a pool that still holds the undiagnosed incidents
   is printed after it, labelled as biased downward;
4. how widely the product flags the five PATH elements per bundle (why the baseline is high).

Usage (from the project root; no third-party packages, safe under ``python -I``)::

    python -I scripts/realdata_retro_baseline.py RESULTS_JSONL
    python -I scripts/realdata_retro_baseline.py RESULTS_JSONL --out docs/eval/realdata/retro-baseline.txt

The output is deterministic (sorted, no timestamps), Korean, UTF-8 with LF line endings. Exit code
1 means the re-derived values differ from the stored ones (the rule below no longer mirrors the
harness, or the file was edited).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

NOT_A_GAP = "not_a_gap"
OUT_OF_SCOPE = "out_of_scope"
#: the report's group for in-scope units the two coders put in different families
DISAGREE = "(코더 간 계열 불일치)"
PATH_ELEMENTS = ("detection", "owner", "approver", "fallback_channel", "rollback")
CHECKED_FIELDS = ("is_gap", "in_scope", "matched", "matched_broken")
#: per-incident totals and the unit decision each one counts
TOTALS = (("n", "is_gap"), ("k", "in_scope"), ("m", "matched"), ("m_broken", "matched_broken"))

Record = dict[str, Any]


# --------------------------------------------------------------------------- the harness rule


def finding_matches(finding: Record, family: str, element: str | None) -> bool:
    """Mirror of ``retro._finding_matches``: same family, and the elements agree or either is null."""
    return finding["family"] == family and (
        element is None or finding.get("element") is None or finding.get("element") == element
    )


def classify(unit: Record, findings: Sequence[Record]) -> dict[str, bool]:
    """Mirror of ``retro.match_units`` for one unit (both coders' codings must find a finding)."""
    labels = (unit["family_a"], unit["family_b"])
    is_gap = all(label is not None and label != NOT_A_GAP for label in labels)
    in_scope = is_gap and all(label not in (None, NOT_A_GAP, OUT_OF_SCOPE) for label in labels)
    matched = matched_broken = False
    if in_scope:
        codings = ((unit["family_a"], unit["element_a"]), (unit["family_b"], unit["element_b"]))
        per_coding = [[f for f in findings if finding_matches(f, str(fam), el)] for fam, el in codings]
        matched = all(per_coding)
        matched_broken = all(any(f["status"] == "broken" for f in hits) for hits in per_coding)
    return {"is_gap": is_gap, "in_scope": in_scope, "matched": matched, "matched_broken": matched_broken}


def findings_of(record: Record) -> list[Record]:
    """The product's findings for the incident (none when the bundle was not diagnosed)."""
    diagnosis = record.get("diagnosis")
    return list(diagnosis["findings"]) if diagnosis is not None else []


def family_group(unit: Record) -> str:
    """The report's per-family grouping of an in-scope unit."""
    same = unit["family_a"] == unit["family_b"] and unit["family_a"] is not None
    return str(unit["family_a"]) if same else DISAGREE


# --------------------------------------------------------------------------- analysis


def load(path: Path) -> list[Record]:
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return sorted(records, key=lambda r: r["incident_id"])


def derived_totals(record: Record) -> dict[str, int]:
    """n, k, m, m_broken of one incident, re-derived from its units and findings."""
    findings = findings_of(record)
    derived = [classify(unit, findings) for unit in record["units"]]
    return {field: sum(d[key] for d in derived) for field, key in TOTALS}


def verify(records: Sequence[Record]) -> list[str]:
    """Differences between the re-derived and the stored unit decisions and per-incident n/k/m."""
    problems: list[str] = []
    for record in records:
        findings = findings_of(record)
        where = record["incident_id"]
        for unit in record["units"]:
            mine = classify(unit, findings)
            problems.extend(
                f"{where} {unit['unit_id']}: {field} 저장 {unit[field]} ≠ 재계산 {mine[field]}"
                for field in CHECKED_FIELDS
                if unit[field] != mine[field]
            )
        problems.extend(
            f"{where}: {field} 저장 {record[field]} ≠ 재계산 {count}"
            for field, count in derived_totals(record).items()
            if record[field] != count
        )
    return problems


def pct(hits: float, n: int) -> str:
    return "—" if n == 0 else f"{100 * hits / n:.1f}%"


def totals_lines(records: Sequence[Record]) -> list[str]:
    groups = {"전체": list(records)}
    groups["한국(KR)"] = [r for r in records if r.get("jurisdiction") == "KR"]
    groups["해외(KR 아님)"] = [r for r in records if r.get("jurisdiction") != "KR"]
    lines = ["## 1. N · k · m (단위마다 같은 규칙으로 다시 계산)", ""]
    for name, group in groups.items():
        totals = [derived_totals(r) for r in group]
        n, k, m = (sum(t[f] for t in totals) for f in ("n", "k", "m"))
        lines.append(
            f"- {name}: 사고 {len(group)}건 · N={n} · k={k} (k/N {pct(k, n)}) · m={m} (m/k {pct(m, k)})"
        )
    undiagnosed = len(undiagnosed_ids(records))
    lines.append(
        f"- m/k는 하네스 값이다(진단하지 않은 {undiagnosed}건 포함, 느슨한 계열·요소 일치)."
        " 우연 기준과의 비교는 2절 — 이 값만으로 사고별 탐지를 말하지 않는다."
    )
    return lines


def chance_matches(
    records: Sequence[Record],
    compare_to: Sequence[str] | None = None,
    score: Sequence[str] | None = None,
) -> dict[str, list[tuple[bool, float]]]:
    """Per family group, (observed match, share of matching other incidents) for every in-scope unit.

    ``compare_to`` limits the 'other' incidents a unit is matched against and ``score`` the
    incidents whose units are counted (all incidents when None).
    """
    findings = {r["incident_id"]: findings_of(r) for r in records}
    pool = list(findings) if compare_to is None else [i for i in findings if i in set(compare_to)]
    scored = None if score is None else set(score)
    by_group: dict[str, list[tuple[bool, float]]] = defaultdict(list)
    for record in records:
        own = record["incident_id"]
        if scored is not None and own not in scored:
            continue
        others = [i for i in pool if i != own]
        for unit in record["units"]:
            mine = classify(unit, findings[own])
            if not mine["in_scope"]:
                continue
            share = sum(classify(unit, findings[o])["matched"] for o in others) / len(others)
            by_group[family_group(unit)].append((mine["matched"], share))
    return by_group


def undiagnosed_ids(records: Sequence[Record]) -> list[str]:
    """Incidents whose bundle was not diagnosed (no runbook survived reconstruction)."""
    return [r["incident_id"] for r in records if r.get("diagnosis") is None]


def summary(by_group: dict[str, list[tuple[bool, float]]]) -> tuple[int, int, float]:
    """k (in-scope units), observed m, and expected m under chance."""
    pairs = [pair for group in by_group.values() for pair in group]
    return len(pairs), sum(obs for obs, _ in pairs), sum(share for _, share in pairs)


def family_table(by_group: dict[str, list[tuple[bool, float]]], baseline_label: str) -> list[str]:
    lines = [f"| 계열 | k | m | 관측 m/k | {baseline_label} |", "|---|---:|---:|---:|---:|"]
    for name in sorted(by_group):
        group = by_group[name]
        hits = sum(obs for obs, _ in group)
        lines.append(
            f"| {name} | {len(group)} | {hits} | {pct(hits, len(group))} | "
            f"{pct(sum(s for _, s in group), len(group))} |"
        )
    return lines


def baseline_lines(records: Sequence[Record]) -> list[str]:
    """The primary (fair) comparison: diagnosed incidents only, on both sides.

    An undiagnosed incident (no runbook statement survived reconstruction, ``diagnosis`` is null)
    has no findings, so its own units can never match and, as an 'other' incident, it can never
    match anyone's units. Both would pull the comparison apart for reasons unrelated to the
    product, so they are left out of the observed units and of the comparison pool.
    """
    undiagnosed = undiagnosed_ids(records)
    diagnosed = [r["incident_id"] for r in records if r.get("diagnosis") is not None]
    by_group = chance_matches(records, compare_to=diagnosed, score=diagnosed)
    k, m, expected = summary(by_group)
    lines = [
        f"## 2. 우연 기준선 — 주 비교: 진단한 사고 {len(diagnosed)}건만",
        "",
    ]
    if undiagnosed:
        lines += [
            f"진단하지 않은 {len(undiagnosed)}건(복원 묶음에 런북 없음): {', '.join(undiagnosed)}",
            "재구성 묶음에 사고 전 런북 문장이 남지 않아 진단을 돌리지 않은 사고다.",
            "지적이 없으니 어떤 공백과도 일치할 수 없다. 그래서 공정한 비교는 관측 쪽 범위 안 단위와",
            "비교 대상 모두 진단한 사고만으로 한다.",
            "",
        ]
    lines += [
        f"진단한 사고의 범위 안 공백 단위마다, 같은 일치 규칙으로 진단한 '다른' 사고 {len(diagnosed) - 1}건",
        "각각의 제품 지적과 대조해 일치하는 비율을 구하고 단위들에 걸쳐 평균했다.",
        "지적이 그 사고와 무관해도 나오는 m/k 수준이다.",
        "",
        f"- 관측 m/k(진단한 사고의 범위 안 단위): {pct(m, k)} ({m}/{k})",
        f"- 우연 기준선(진단한 다른 사고 지적과의 일치 비율 평균): {pct(expected, k)}",
        f"- 우연일 때 기대되는 m: {expected:.1f} (관측 m={m}, 차이 {m - expected:+.1f})",
        "",
        *family_table(by_group, "우연 기준선"),
    ]
    return lines


def biased_lines(records: Sequence[Record]) -> list[str]:
    """The harness m/k against a pool that still holds the undiagnosed incidents (biased)."""
    undiagnosed = undiagnosed_ids(records)
    lines = [
        "## 2-1. 참고: 진단하지 않은 사고까지 넣은 비교 — 아래로 치우침, 쓰지 않는다",
        "",
    ]
    if not undiagnosed:
        return [*lines, "- 진단하지 않은 사고가 없어 위 2절과 같다."]
    by_group = chance_matches(records)
    k, m, expected = summary(by_group)
    left_out = Counter(
        r["incident_id"]
        for r in records
        if r["incident_id"] in undiagnosed
        for unit in r["units"]
        if classify(unit, [])["in_scope"]
    )
    counted = ", ".join(f"{incident} {count}개" for incident, count in sorted(left_out.items()))
    diagnosed = [r["incident_id"] for r in records if r.get("diagnosis") is not None]
    k2, _, expected2 = summary(chance_matches(records, compare_to=diagnosed))
    lines += [
        f"하네스 보고서(retro-report.md)의 m/k는 사고 {len(records)}건의 범위 안 단위 {k}개로 센 값이다.",
        f"진단하지 않은 사고의 범위 안 단위({counted or '없음'})는 불일치로 들어간다.",
        f"이 값을 진단하지 않은 사고까지 넣은 비교 대상(다른 사고 {len(records) - 1}건)의 기준선과 견주면,",
        "진단하지 않은 사고는 일치할 수 없어 기준선이 아래로 치우친다.",
        "이 비교로 하네스 m/k가 우연보다 높다고 말하지 않는다 — 위 2절의 비교를 쓴다.",
        "",
        f"- 하네스 m/k(사고 {len(records)}건, 범위 안 단위 {k}개): {pct(m, k)} ({m}/{k})",
        f"- 치우친 기준선(진단하지 않은 사고를 넣은 비교 대상): {pct(expected, k)} · 기대 m {expected:.1f}",
        f"- 비교 대상만 진단한 사고로(범위 안 단위 {k2}개 전체): 기준선 {pct(expected2, k2)}"
        f" · 기대 m {expected2:.1f}",
        "  (이번에는 관측 쪽에 일치할 수 없는 단위가 남아 관측이 아래로 치우친다)",
        "",
        *family_table(by_group, "치우친 기준선"),
    ]
    return lines


def coverage_lines(records: Sequence[Record]) -> list[str]:
    counts: dict[str, int] = {}
    for record in records:
        elements = {f.get("element") for f in findings_of(record) if f["family"] == "PATH"}
        counts[record["incident_id"]] = sum(e in elements for e in PATH_ELEMENTS)
    shares = [c / len(PATH_ELEMENTS) for c in counts.values()]
    undiagnosed = set(undiagnosed_ids(records))
    no_path = [incident for incident, count in counts.items() if count == 0]
    no_path_undiagnosed = sum(incident in undiagnosed for incident in no_path)
    lines = [
        "## 3. PATH 5요소 지적 범위 (기준선이 높은 이유)",
        "",
        "사고마다 PATH 5요소(detection·owner·approver·fallback_channel·rollback) 가운데 제품이 한 번이라도",
        "지적한(상태 무관) 요소의 비율.",
        "",
        f"- 평균: {100 * statistics.mean(shares):.0f}%",
        f"- 5요소 중 4개 이상을 지적한 사고: {sum(c >= 4 for c in counts.values())}/{len(counts)}건",
        f"- PATH 지적이 하나도 없는 사고: {len(no_path)}건 (그중 진단하지 않은 사고 {no_path_undiagnosed}건)",
        "",
        "| 사고 | 지적한 PATH 요소 |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {incident} | {count}/5{' (진단 안 함)' if incident in undiagnosed else ''} |"
        for incident, count in counts.items()
    )
    return lines


def findings_lines(records: Sequence[Record]) -> list[str]:
    per_incident = [len(findings_of(r)) for r in records]
    families = Counter(f["family"] for r in records for f in findings_of(r))
    undiagnosed = undiagnosed_ids(records)
    zero = [r["incident_id"] for r in records if r.get("diagnosis") is not None and not findings_of(r)]
    by_family = " · ".join(f"{name} {count}" for name, count in families.most_common())
    return [
        "## 4. 제품 지적",
        "",
        f"- 합계 {sum(per_incident)}건 ({by_family})",
        f"- 사고당 중앙값 {statistics.median(per_incident):g}건 (진단하지 않은 사고 포함)",
        f"- 진단하지 않은 {len(undiagnosed)}건(복원 묶음에 런북 없음): {', '.join(undiagnosed) or '없음'}",
        f"- 진단했지만 지적 0건인 사고 {len(zero)}건: {', '.join(zero) if zero else '없음'}",
    ]


def render(path: Path) -> tuple[str, bool]:
    records = load(path)
    problems = verify(records)
    lines = [
        "# 실증 ② 우연 기준선 — scripts/realdata_retro_baseline.py",
        "",
        f"입력: {path.name} · 사고 {len(records)}건 · 공백 단위 {sum(len(r['units']) for r in records)}개",
        "일치 규칙: retro._finding_matches / match_units와 같음 — 같은 계열이고, 요소가 같거나 어느 한쪽이",
        "null이면 일치. 두 코더의 코딩 모두에 맞는 지적이 있어야 한다. 같은 서비스·사람인지는 보지 않는다.",
        f"저장된 판정과 재계산 일치: {'예' if not problems else f'아니오 ({len(problems)}건)'}",
        "",
        *totals_lines(records),
        "",
        *baseline_lines(records),
        "",
        *biased_lines(records),
        "",
        *coverage_lines(records),
        "",
        *findings_lines(records),
    ]
    if problems:
        lines += ["", "## 불일치", "", *(f"- {p}" for p in problems)]
    return "\n".join(lines) + "\n", not problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("results", type=Path, help="retro 결과 폴더의 results.jsonl")
    parser.add_argument("--out", type=Path, help="결과를 이 파일에 저장 (UTF-8, LF)")
    args = parser.parse_args(argv)
    text, consistent = render(args.results)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8", newline="\n")
    else:
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")  # type: ignore[attr-defined]
        sys.stdout.write(text)
    return 0 if consistent else 1


if __name__ == "__main__":
    sys.exit(main())
