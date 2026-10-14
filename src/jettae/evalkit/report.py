"""Evaluation report (평가 보고서): metrics.json for machines, report.md for people.

``metrics.json`` is the single source of the published numbers (the pitch-deck generator reads
it); ``report.md`` renders the same object as Korean tables plus interpretation sentences that
are computed from the numbers, never written by hand, so the prose cannot drift from the data.
``results.jsonl`` keeps every (case, condition) verdict so anyone can recount a table.

Nothing here reads the clock: ``generated_at`` is passed in, and the files of the same run are
byte-identical apart from the measured timings.

Run the whole offline evaluation (corpus → seal → A/B/O → reproducibility → benchmarks →
report)::

    python -m jettae.evalkit --out docs/eval --generated-at 2026-10-13

LLM arms are refused unless asked for *and* confirmed (``--confirm-paid``); the cost estimate is
always printed first (``--estimate-only`` stops there).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from jettae.config import Settings, get_settings
from jettae.domain import LlmMode
from jettae.evalkit.benchmarks import (
    DEFAULT_BUNDLE,
    ImpactBenchmark,
    InjectionSelftest,
    impact_benchmark,
    injection_selftest,
)
from jettae.evalkit.conditions import (
    DEFECT_RULE_IDS,
    GRAPH_CONDITIONS,
    KEYWORD_SIGNALS,
    LLM_CONDITIONS,
    OFFLINE_CONDITIONS,
    CaseResult,
    Condition,
)
from jettae.evalkit.corpus import build_cases
from jettae.evalkit.gold import DEFAULT_SEED, generate_gold_orgs
from jettae.evalkit.harness import (
    REPRODUCIBILITY_FRACTION,
    CostEstimate,
    EvalResults,
    Reproducibility,
    check_reproducibility,
    estimate_llm_cost,
    parse_conditions,
    run_eval,
)
from jettae.evalkit.metrics import (
    DELETION_OPERATORS,
    ORG_LEVEL_RULE_IDS,
    EvalMetrics,
    OperatorRate,
    Prf,
    Rate,
    compute_metrics,
)
from jettae.evalkit.mutate import ALL_OPERATORS, EXPECTED_RULE_IDS, Operator
from jettae.evalkit.render import STYLES
from jettae.evalkit.seal import SealManifest, build_manifest, write_manifest
from jettae.llm.base import LLMClient
from jettae.llm.factory import build_llm

__all__ = [
    "CONDITION_LABEL_KO",
    "OPERATOR_LABEL_KO",
    "STYLE_LABEL_KO",
    "main",
    "render_report",
    "results_jsonl",
    "run_evaluation",
    "write_report",
]

CONDITION_LABEL_KO: dict[Condition, str] = {
    Condition.A_KEYWORD: "A 키워드 검색(그래프 없음)",
    Condition.B_RULE_GRAPH: "B 규칙 파서 + 그래프",
    Condition.C_LLM_DIRECT: "C LLM 직접 판정",
    Condition.D_LLM_GRAPH: "D LLM 추출 + 그래프",
    Condition.DR_HYBRID: "DR 규칙∪LLM 추출 + 그래프",
    Condition.O_ORACLE: "O 정답 그래프(상한)",
}
STYLE_LABEL_KO: dict[str, str] = {"table": "표", "explicit": "명시형 목록", "colloquial": "구어체"}
POSITION_LABEL_KO: dict[str, str] = {"front": "문서 앞", "back": "문서 뒤"}
OPERATOR_LABEL_KO: dict[Operator, str] = {
    Operator.DELETE_OWNER: "담당자 삭제",
    Operator.DEPARTED_CONTACT: "담당자 퇴사(명부만 변경)",
    Operator.DEADLINE_CONTRADICTION: "침해사고 신고 시한 48시간",
    Operator.ESCALATION_CYCLE: "상향 순환",
    Operator.DELETE_APPROVER: "승인자 삭제",
    Operator.DELETE_FALLBACK_CHANNEL: "대체 채널 삭제",
    Operator.DELETE_ROLLBACK: "롤백 삭제",
    Operator.DELETE_ESCALATION: "상향 경로 삭제",
    Operator.DELETE_DETECTION: "감지 삭제",
    Operator.DELETE_REPORTING_STEP: "KISA 신고 단계 삭제",
}
_DASH = "—"


# --------------------------------------------------------------------------- formatting


def _pct(value: float | None) -> str:
    return _DASH if value is None else f"{value * 100:.1f}%"


def _rate(rate: Rate, *, ci: bool = True) -> str:
    """``87.5% (84/96, 95% CI 79.4–92.7%)``; the dash when nothing was measured."""
    if rate.rate is None:
        return _DASH
    text = f"{_pct(rate.rate)} ({rate.hits:,}/{rate.n:,}"
    if ci and rate.ci95 is not None:
        lo, hi = rate.ci95
        text += f", 95% CI {lo * 100:.1f}–{hi * 100:.1f}%"
    return text + ")"


def _prf(prf: Prf) -> str:
    return f"{_pct(prf.precision)} / {_pct(prf.recall)} / {_pct(prf.f1)}"


def _pair_of(active: Rate, held_out: Rate) -> str:
    """``65.3% · 89.6%`` (active · held-out); one dash when neither was measured."""
    if active.rate is None and held_out.rate is None:
        return _DASH
    return f"{_pct(active.rate)} · {_pct(held_out.rate)}"


def _table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return out


def _label(condition: str) -> str:
    return CONDITION_LABEL_KO[Condition(condition)]


# --------------------------------------------------------------------------- sections


def _headline(m: EvalMetrics) -> list[str]:
    c = m.corpus
    lines = [
        "## 1. 한눈에 보기",
        "",
        f"- **코퍼스** {m.n_cases:,}건 = 결함 주입 {c.n_active + c.n_held_out:,}건"
        f"(활성 연산자 {c.n_active:,}건 + 봉인 연산자 {c.n_held_out:,}건) + 정상 대조군 {c.n_clean:,}건, "
        f"조직 {c.n_orgs}곳 × 문체 3종 × 배치 2종. 생성기 `{c.generator_version}`"
        + (f", 시드 `{c.seed}`." if c.seed is not None else "."),
    ]
    if m.manifest_root is not None:
        lines.append(
            f"- **봉인** Merkle 루트 `{m.manifest_root}` (봉인 연산자 루트 `{c.held_out_manifest_root}`), "
            "`manifest.json`에 사례별 SHA-256과 함께 공개."
        )
    lines += ["", "조건별 핵심 수치(괄호는 건수, 탐지율은 95% 신뢰구간과 함께 3장 이후 표 참고):", ""]
    rows = []
    for name, s in m.conditions.items():
        rows.append(
            [
                _label(name),
                _rate(s.detection_active, ci=False),
                _rate(s.detection_held_out, ci=False),
                _pair_of(s.detection_paired_active, s.detection_paired_held_out),
                _rate(s.clean_fp, ci=False),
                _prf(s.field) if s.field is not None else _DASH,
                str(s.ungrounded) if Condition(name) in GRAPH_CONDITIONS else _DASH,
                f"{s.latency.median_s * 1000:.0f} ms",
            ]
        )
    # operator counts come from the results, so a smoke run on a subset is not mislabelled
    operators = next(iter(m.by_operator.values()), {})
    n_active_ops = sum(not r.held_out for r in operators.values())
    n_held_out_ops = sum(r.held_out for r in operators.values())
    lines += _table(
        [
            "조건",
            f"탐지율(활성 {n_active_ops}종)",
            f"탐지율(봉인 {n_held_out_ops}종)",
            "대조군 보정 탐지율(활성 · 봉인)",
            "정상 대조군 오탐률",
            "필드 정밀도/재현율/F1",
            "근거 없는 요소",
            "사례당 지연(중앙값)",
        ],
        rows,
    )
    lines.append("")
    if m.reproducibility is not None:
        r = m.reproducibility
        lines.append(
            f"- **재현성** [{_label(r.condition)}] {r.sample_size}건({r.fraction * 100:.0f}% 표본) 재실행 시 "
            f"그래프 해시·발화 규칙이 모두 같은 사례 {r.identical}/{r.sample_size}건 ({_pct(r.rate)})."
        )
    lines.append(f"- **근거 없는 요소** 실행한 모든 그래프 조건 합계 {m.ungrounded_total}건 (목표 0건).")
    if m.impact_benchmark is not None:
        b = m.impact_benchmark
        lines.append(
            f"- **영향 범위 계산** 서비스 {b.n_services}개·시스템 {b.n_systems}개·위치 {b.n_locations}곳에서 "
            f"센터 1곳 장애 시 영향 서비스 목록 산출 {b.median_ms:.3f} ms (중앙값, {b.runs}회)."
        )
    if m.injection_selftest is not None:
        i = m.injection_selftest
        lines.append(
            f"- **프롬프트 인젝션 자기 시험(LLM 끔)** {i.n}건 중 무단 변경 {i.unauthorised_changes}건, "
            f"가드 격리 {i.quarantined}/{i.n}건."
        )
    if m.conditions_pending:
        pending = ", ".join(_label(c) for c in m.conditions_pending)
        lines.append(f"- **미실행** {pending} — API 키가 필요한 유료 조건입니다(13장 실행 방법).")
    return lines


def _design(m: EvalMetrics) -> list[str]:
    rows = [
        [CONDITION_LABEL_KO[c], text]
        for c, text in (
            (
                Condition.A_KEYWORD,
                "서비스 구간별 키워드 검색. 명부를 쓰지 않으므로 퇴사자·상향 순환은 원리상 못 찾음",
            ),
            (
                Condition.B_RULE_GRAPH,
                "LLM 끔 제품 경로: 수집 → 규칙 파서 → 조립 → 역번역 슬롯 대조 → 고정 규칙",
            ),
            (
                Condition.C_LLM_DIRECT,
                "LLM이 본문·명부·정책을 읽고 고정 10종 결함 목록에서 고름(그래프·고정 규칙 없음)",
            ),
            (Condition.D_LLM_GRAPH, "LLM 추출 → 조립 → 역번역 슬롯 대조 → 고정 규칙"),
            (Condition.DR_HYBRID, "규칙 추출과 LLM 추출을 합쳐 조립 → 고정 규칙 (LLM 켬 기본값)"),
            (Condition.O_ORACLE, "정답 사실로 만든 그래프 → 고정 규칙. 판정 규칙의 상한·건전성 확인"),
        )
    ]
    lines = ["## 2. 평가 설계", "", *_table(["조건", "구성"], rows), ""]
    lines += [
        "- **정답은 구성으로 정해집니다.** 사람이나 LLM이 정답을 매기지 않습니다. 완전한 조직 명세를 만들고, "
        "세 문체로 런북을 쓰고, 알려진 연산자로 서비스 하나를 깨뜨립니다. 연산자마다 걸려야 할 규칙이 "
        "공개 대응표로 고정되어 있습니다(4장).",
        "- **탐지(hit)** = 기대 규칙 중 하나가 대상 서비스에서 발화. 조직 전체의 의무가 될 수 있는 규칙"
        f"({', '.join(sorted(ORG_LEVEL_RULE_IDS))})만 조직 수준(서비스 미지정) 발화도 인정합니다. "
        "대상 서비스만 인정하는 엄격 기준도 함께 계산합니다(9장).",
        "- **대조군 보정 탐지** = 탐지 중에서, 같은 조직·문체·배치의 정상 대조군(쌍둥이)에서는 그 발화가 "
        "없었던 경우만 인정합니다. 결함이 없어도 늘 나는 발화(예: 승인자가 적혀 있는데도 '승인자 없음')로 "
        "얻은 탐지를 걸러 낸 값이며, 일반 탐지율과의 차이가 잡음으로 얻은 몫입니다. 쌍둥이 사례가 오류로 "
        "끝났으면 인정하지 않습니다.",
        "- **오탐** = 정상 대조군(같은 조직·같은 문체, 결함 없음)에서 HIGH·CRITICAL 결과가 하나라도 나오면 "
        "오탐.",
        "- **필드 단위 정확도** = 그래프 사실과 정답 사실의 정확 일치(속성 포함). 명부·카탈로그에서 그대로 "
        "가져오는 사실은 제외하고 런북만 말할 수 있는 관계(담당·감지·호출·상향·조치·승인·실행·대체 채널·"
        "신고)만 셉니다.",
        "- **누락 필드 환각** = 삭제 연산자로 문서에서 지운 사실을 그래프가 여전히 담고 있는 비율.",
        "- 오류가 난 사례는 불리하게 셉니다: 결함 사례는 놓침, 정상 대조군은 오탐, 정답 사실은 모두 누락.",
        "- 비율의 신뢰구간은 Wilson 95% 구간입니다.",
    ]
    return lines


def _by_style(m: EvalMetrics) -> list[str]:
    rows = []
    for name, styles in m.by_style.items():
        for style in STYLES:
            if style in styles:
                split = styles[style]
                rows.append([_label(name), STYLE_LABEL_KO[style], _rate(split.active), _rate(split.held_out)])
    return ["## 3. 문체별 탐지율", "", *_table(["조건", "문체", "활성 연산자", "봉인 연산자"], rows)]


def _by_operator(m: EvalMetrics) -> list[str]:
    names = list(m.by_operator)
    rows = []
    for op in ALL_OPERATORS:
        cells = [m.by_operator[n].get(op.value) for n in names]
        if all(c is None for c in cells):
            continue
        sealed = " (봉인)" if any(c is not None and c.held_out for c in cells) else ""
        rows.append(
            [
                f"{OPERATOR_LABEL_KO[op]}{sealed}",
                ", ".join(sorted(EXPECTED_RULE_IDS[op])),
                *(_DASH if c is None else _operator_cell(c) for c in cells),
            ]
        )
    header = ["결함 연산자", "기대 규칙", *(_label(n) for n in names)]
    return [
        "## 4. 결함 연산자별 탐지율",
        "",
        "칸마다 탐지율(건수)과, 정상 대조군에서도 같은 발화가 있던 탐지를 뺀 대조군 보정 탐지율(보정)을 "
        "함께 적었습니다.",
        "",
        *_table(header, rows),
    ]


def _operator_cell(rate: OperatorRate) -> str:
    text = _rate(rate, ci=False)
    if rate.paired is not None and rate.paired.rate is not None:
        text += f" · 보정 {_pct(rate.paired.rate)}"
    return text


def _by_position(m: EvalMetrics) -> list[str]:
    rows = []
    for name, positions in m.by_position.items():
        for position, split in positions.items():
            rows.append(
                [
                    _label(name),
                    POSITION_LABEL_KO.get(position, position),
                    _rate(split.active, ci=False),
                    _rate(split.held_out, ci=False),
                ]
            )
    return [
        "## 5. 결함 서비스 위치(문서 앞/뒤)별 탐지율",
        "",
        *_table(["조건", "위치", "활성 연산자", "봉인 연산자"], rows),
    ]


def _clean(m: EvalMetrics) -> list[str]:
    names = list(m.clean_fp)
    rows = [
        [
            _label(n),
            _rate(m.clean_fp[n].rate),
            *(
                _rate(m.clean_fp[n].by_style[s], ci=False) if s in m.clean_fp[n].by_style else _DASH
                for s in STYLES
            ),
        ]
        for n in names
    ]
    lines = [
        "## 6. 정상 대조군 오탐",
        "",
        *_table(["조건", "오탐률(HIGH·CRITICAL)", *(STYLE_LABEL_KO[s] for s in STYLES)], rows),
        "",
        "정상 대조군에서 한 번이라도 발화한 규칙(모든 심각도, 대조군 사례 수):",
        "",
    ]
    rules = sorted({r for n in names for r in m.clean_fp[n].per_rule})
    rule_rows = []
    for rule in rules:
        severity = next(
            m.clean_fp[n].per_rule[rule].severity for n in names if rule in m.clean_fp[n].per_rule
        )
        rule_rows.append(
            [
                rule,
                severity,
                *(
                    str(m.clean_fp[n].per_rule[rule].cases) if rule in m.clean_fp[n].per_rule else "0"
                    for n in names
                ),
            ]
        )
    if rule_rows:
        lines += _table(["규칙", "심각도", *(_label(n) for n in names)], rule_rows)
    else:
        lines.append("- 없음")
    return lines


def _fields(m: EvalMetrics) -> list[str]:
    if not m.field_f1:
        return []
    names = list(m.field_f1)
    style_rows = [
        [_label(n), STYLE_LABEL_KO[s], _prf(m.field_f1[n].by_style[s])]
        for n in names
        for s in STYLES
        if s in m.field_f1[n].by_style
    ]
    kinds = sorted({k for n in names for k in m.field_f1[n].by_kind})
    kind_rows = [
        [
            kind,
            *(_prf(m.field_f1[n].by_kind[kind]) if kind in m.field_f1[n].by_kind else _DASH for n in names),
        ]
        for kind in kinds
    ]
    return [
        "## 7. 필드 단위 정확도 (정밀도 / 재현율 / F1)",
        "",
        *_table(["조건", "문체", "정밀도 / 재현율 / F1"], style_rows),
        "",
        "관계 종류별(모든 문체):",
        "",
        *_table(["관계", *(_label(n) for n in names)], kind_rows),
    ]


def _hallucination(m: EvalMetrics) -> list[str]:
    if not m.hallucination:
        return []
    names = list(m.hallucination)

    def cell(name: str, op: Operator) -> str:
        by_operator = m.hallucination[name].by_operator
        return _rate(by_operator[op.value], ci=False) if op.value in by_operator else _DASH

    rows = [["전체", *(_rate(m.hallucination[n].rate) for n in names)]]
    rows += [[OPERATOR_LABEL_KO[op], *(cell(n, op) for n in names)] for op in DELETION_OPERATORS]
    return [
        "## 8. 누락 필드 환각률 (지운 사실을 그래프가 되살린 비율)",
        "",
        *_table(["삭제 연산자", *(_label(n) for n in names)], rows),
    ]


def _operations(m: EvalMetrics) -> list[str]:
    rows = [
        [
            _label(n),
            str(s.errors),
            _rate(s.detection_strict_active, ci=False),
            _rate(s.detection_strict_held_out, ci=False),
            _rate(s.detection_paired_active, ci=False),
            _rate(s.detection_paired_held_out, ci=False),
            str(s.ungrounded) if Condition(n) in GRAPH_CONDITIONS else _DASH,
            f"{s.latency.median_s * 1000:.1f} / {s.latency.p95_s * 1000:.1f} ms",
            f"{s.llm.calls_per_case:.2f}회 · 입력 {s.llm.input_tokens_per_case:,.0f} · 출력 "
            f"{s.llm.output_tokens_per_case:,.0f}",
        ]
        for n, s in m.conditions.items()
    ]
    lines = [
        "## 9. 근거·재현성·지연·LLM 사용량",
        "",
        *_table(
            [
                "조건",
                "오류 사례",
                "엄격 탐지율(활성)",
                "엄격 탐지율(봉인)",
                "대조군 보정 탐지율(활성)",
                "대조군 보정 탐지율(봉인)",
                "근거 없는 요소",
                "지연 중앙값 / p95",
                "사례당 LLM 호출·토큰",
            ],
            rows,
        ),
        "",
    ]
    if m.reproducibility is not None:
        r = m.reproducibility
        where = (
            "새로 띄운 별도 프로세스(문자열 해시 시드·캐시가 첫 실행과 다름)에서"
            if r.fresh_process
            else "첫 실행과 같은 프로세스에서(자기 일관성 확인일 뿐, 프로세스 간 재현성은 아님)"
        )
        lines.append(
            f"- 재현성: [{_label(r.condition)}] 사례 id의 SHA-256 순서로 고른 {r.sample_size}건을 {where} "
            "다시 실행해 그래프 해시·발화 규칙·사실 집합을 비교했습니다. "
            f"동일 {r.identical}건, 불일치 {len(r.mismatched)}건"
            + (f" ({', '.join(r.mismatched)})." if r.mismatched else ".")
        )
    return lines


def _impact(b: ImpactBenchmark | None) -> list[str]:
    if b is None:
        return []
    return [
        "## 10. 영향 범위(센터 1곳 장애) 계산 시간",
        "",
        f"- 합성 카탈로그: 서비스 {b.n_services}개, 시스템 {b.n_systems}개, 위치 {b.n_locations}곳 "
        f"(노드 {b.n_nodes}개, 간선 {b.n_edges}개, 시드 고정).",
        f"- `{b.failed}` 장애 시 영향 서비스 {b.impacted_services}개, 영향 시스템 {b.impacted_systems}개.",
        f"- `check.impact.blast_radius` {b.runs}회 실행: 중앙값 {b.median_ms:.3f} ms "
        f"(최소 {b.min_ms:.3f} ms, 최대 {b.max_ms:.3f} ms).",
    ]


def _injection(i: InjectionSelftest | None) -> list[str]:
    if i is None:
        return []
    kinds = sorted({c.kind for c in i.cases})
    rows = []
    for kind in kinds:
        cases = [c for c in i.cases if c.kind == kind]
        rows.append(
            [
                kind,
                str(len(cases)),
                str(sum(c.quarantined for c in cases)),
                str(sum(c.unauthorised for c in cases)),
                str(sum(not c.evidence_unchanged for c in cases)),
            ]
        )
    return [
        "## 11. 프롬프트 인젝션 자기 시험 (LLM 끔)",
        "",
        f"- 대상: `{i.bundle}`의 `{i.document}`에 공격 문장 한 줄씩 {i.n}건 삽입, 기준 점수 "
        f"{i.baseline_score}.",
        f"- **무단 변경 {i.unauthorised_changes}건**: 그래프 구조·끊김 목록·점수가 깨끗한 실행과 같고, "
        "삽입 줄에서 나온 요소와 심어 둔 인물(홍길동)이 없어야 통과입니다.",
        f"- 가드가 삽입 줄 전체를 격리한 사례 {i.quarantined}/{i.n}건.",
        f"- 근거 인용문만 달라진 사례 {i.evidence_changed}건(사실은 동일): 격리된 줄이 여러 줄 인용을 끊어 "
        "같은 사실을 더 짧은 인용으로 가리키게 된 경우이며 무단 변경으로 세지 않습니다.",
        "",
        *_table(["공격 유형", "건수", "격리", "무단 변경", "인용만 달라짐"], rows),
    ]


def _interpretation(m: EvalMetrics) -> list[str]:
    """Sentences computed from the numbers (no hand-written claims)."""
    lines = ["## 12. 해석 (수치에서 자동 생성)", ""]
    lines += _oracle_notes(m)
    lines += _paired_notes(m)
    lines += _style_notes(m)
    lines += _rule_field_notes(m)
    lines += _false_alarm_notes(m)
    lines += _keyword_notes(m)
    return lines


def _oracle_notes(m: EvalMetrics) -> list[str]:
    oracle = m.conditions.get(Condition.O_ORACLE.value)
    rule = m.conditions.get(Condition.B_RULE_GRAPH.value)
    if oracle is None:
        return []
    lines = [
        f"- 판정 규칙의 상한(O): 활성 {_pct(oracle.detection_active.rate)}, "
        f"봉인 {_pct(oracle.detection_held_out.rate)}, 정상 대조군 오탐 {_pct(oracle.clean_fp.rate)}. "
        "완전한 그래프라면 고정 규칙이 무엇을 잡는지 보여 주는 기준선입니다."
    ]
    gap = _gap(oracle.detection_active, rule.detection_active) if rule is not None else None
    if gap is not None:
        lines.append(
            f"- 추출 손실: 규칙 파서 그래프(B)의 활성 탐지율은 O보다 {gap:.1f}%p 낮습니다. "
            "같은 판정 규칙을 쓰므로 이 차이는 전부 문서를 그래프로 옮기는 단계에서 생깁니다."
        )
    return lines


def _paired_notes(m: EvalMetrics) -> list[str]:
    """How much of each arm's detection the clean twins would have produced anyway."""
    lines: list[str] = []
    for name, s in m.conditions.items():
        parts = [
            f"{what} 탐지 {plain.hits:,}건 중 {plain.hits - paired.hits:,}건"
            f"({_pct(plain.rate)} → {_pct(paired.rate)})"
            for what, plain, paired in (
                ("활성", s.detection_active, s.detection_paired_active),
                ("봉인", s.detection_held_out, s.detection_paired_held_out),
            )
            # without a twin for every case the two rates count different cases: no comparison
            if plain.rate is not None and paired.rate is not None and paired.n == plain.n
            if plain.hits > paired.hits
        ]
        if parts:
            lines.append(
                f"- {_label(name)}: {', '.join(parts)}은 같은 조직·문체·배치의 정상 대조군에서도 같은 "
                "규칙이 같은 대상에 발화했습니다. 결함이 없어도 나는 신호라 대조군 보정 탐지율에서는 "
                "빠집니다(괄호는 탐지율 → 대조군 보정 탐지율)."
            )
    return lines


def _style_notes(m: EvalMetrics) -> list[str]:
    styles = m.by_style.get(Condition.B_RULE_GRAPH.value, {})
    measured = [st for st in STYLES if st in styles and styles[st].active.rate is not None]
    if not measured:
        return []
    parts = ", ".join(f"{STYLE_LABEL_KO[st]} {_pct(styles[st].active.rate)}" for st in measured)
    lines = [f"- 문체별 B 활성 탐지율: {parts}."]
    if min(measured, key=lambda st: styles[st].active.rate or 0.0) == "colloquial":
        best = max(measured, key=lambda st: styles[st].active.rate or 0.0)
        if _separated(styles[best].active, styles["colloquial"].active):
            lines.append(
                "- 구어체에서 가장 낮고, 가장 높은 문체와 95% 신뢰구간이 겹치지 않습니다. 별칭·말투로 "
                "사람을 부르는 온보딩 메모를 규칙 파서만으로는 다 읽지 못한다는 신호이며, LLM 추출(D·DR)이 "
                "필요한 지점입니다. 판정은 여전히 고정 규칙이 합니다."
            )
        else:
            # overlapping intervals are not evidence of a style effect; say so instead of a claim
            lines.append(
                "- 구어체가 가장 낮지만 가장 높은 문체와 95% 신뢰구간이 겹쳐, 탐지율만으로는 문체 차이를 "
                "단정할 수 없습니다."
            )
            lines += _colloquial_recall_note(m)
    positions = m.by_position.get(Condition.B_RULE_GRAPH.value, {})
    if len(positions) > 1:
        splits = list(positions.values())
        parts = ", ".join(
            f"{POSITION_LABEL_KO.get(p, p)} {_pct(v.active.rate)}" for p, v in positions.items()
        )
        overlap = "겹치지 않음" if _separated(splits[0].active, splits[-1].active) else "겹침"
        lines.append(f"- 결함 서비스 위치별 B 활성 탐지율: {parts} (95% 신뢰구간 {overlap}).")
    return lines


def _colloquial_recall_note(m: EvalMetrics) -> list[str]:
    """The field-recall view of the style question, stated only when the numbers show it."""
    fields = m.field_f1.get(Condition.B_RULE_GRAPH.value)
    if fields is None:
        return []
    recalls = {st: p.recall for st, p in fields.by_style.items() if p.recall is not None}
    if "colloquial" not in recalls or min(recalls, key=lambda st: recalls[st]) != "colloquial":
        return []
    best = max(recalls, key=lambda st: recalls[st])
    return [
        f"- 다만 필드 재현율은 구어체가 가장 낮습니다(구어체 {_pct(recalls['colloquial'])}, "
        f"{STYLE_LABEL_KO[best]} {_pct(recalls[best])}). 규칙 파서가 별칭·말투로 쓴 문장에서 사실을 덜 "
        "읽는다는 신호이며, LLM 추출(D·DR)이 줄이려는 손실이 이것입니다. 판정은 여전히 고정 규칙이 합니다."
    ]


def _rule_field_notes(m: EvalMetrics) -> list[str]:
    fields = m.field_f1.get(Condition.B_RULE_GRAPH.value)
    if fields is None:
        return []
    lines: list[str] = []
    weak = sorted(k for k, prf in fields.by_kind.items() if prf.recall is not None and prf.recall < 0.5)
    if weak:
        lines.append(f"- B의 재현율이 50% 미만인 관계: {', '.join(weak)}.")
    recalls = ", ".join(
        f"{STYLE_LABEL_KO[st]} {_pct(fields.by_style[st].recall)}" for st in STYLES if st in fields.by_style
    )
    lines.append(f"- 문체별 B 필드 재현율: {recalls}.")
    return lines


def _false_alarm_notes(m: EvalMetrics) -> list[str]:
    rule = m.conditions.get(Condition.B_RULE_GRAPH.value)
    if rule is None:
        return []
    per_rule = m.clean_fp[Condition.B_RULE_GRAPH.value].per_rule
    top = sorted(per_rule.items(), key=lambda kv: (-kv[1].cases, kv[0]))[:3]
    if not top:
        return [f"- B의 정상 대조군 오탐률은 {_pct(rule.clean_fp.rate)}입니다."]
    listed = ", ".join(f"{rule_id}({count.cases}건, {count.severity})" for rule_id, count in top)
    line = f"- B의 정상 대조군 오탐률은 {_pct(rule.clean_fp.rate)}입니다. 가장 자주 발화한 규칙: {listed}."
    oracle = m.clean_fp.get(Condition.O_ORACLE.value)
    if oracle is not None and not any(rule_id in oracle.per_rule for rule_id, _ in top):
        line += (
            " 같은 대조군에서 O는 이 규칙들을 한 번도 발화하지 않았으므로, 원인은 규칙이 아니라 추출입니다."
        )
    return [line]


def _keyword_notes(m: EvalMetrics) -> list[str]:
    blind = [
        OPERATOR_LABEL_KO[Operator(op)]
        for op, rate in m.by_operator.get(Condition.A_KEYWORD.value, {}).items()
        if rate.rate == 0.0
    ]
    if not blind:
        return []
    return [
        f"- 키워드 검색(A)이 한 건도 못 찾은 결함: {', '.join(blind)}. "
        "명부와 경로가 없으면 보이지 않는 결함입니다."
    ]


def _separated(a: Rate, b: Rate) -> bool:
    """Whether the two Wilson 95% intervals do not overlap (a conservative difference test)."""
    if a.ci95 is None or b.ci95 is None:
        return False
    return a.ci95[1] < b.ci95[0] or b.ci95[1] < a.ci95[0]


def _gap(upper: Rate, lower: Rate) -> float | None:
    if upper.rate is None or lower.rate is None:
        return None
    return (upper.rate - lower.rate) * 100


def _llm_howto(m: EvalMetrics) -> list[str]:
    lines = [
        "## 13. LLM 조건(C·D·DR) 실행 방법",
        "",
        "C·D·DR은 실제 모델을 호출하므로 `ANTHROPIC_API_KEY`가 필요하고 비용이 듭니다. 이 저장소의 결과는 "
        "API 키 없이 만든 것이라 세 조건은 테스트(가짜 LLM)로만 검증되었고 수치가 없습니다. 하네스는 `llm` "
        "인자를 명시하지 않으면 LLM 조건을 거부하고, 실행 전에 호출 수·토큰 추정치를 먼저 출력합니다.",
        "",
        "```powershell",
        "# 1) 비용 추정만 (호출 없음)",
        "python -m jettae.evalkit --conditions C_llm_direct,D_llm_graph,DR_hybrid --estimate-only",
        "",
        "# 2) 실제 실행 (유료). 모델은 JETTAE_ANTHROPIC_MODEL(기본 claude-opus-5-5)",
        '$env:ANTHROPIC_API_KEY = "<발급받은 키>"',
        "python -m jettae.evalkit --out docs/eval --generated-at <YYYY-MM-DD> `",
        "  --conditions A_keyword,B_rule_graph,O_oracle,C_llm_direct,D_llm_graph,DR_hybrid --confirm-paid",
        "```",
        "",
        "- D와 DR은 사례마다 LLM 추출 한 번을 공유합니다(DR = 같은 LLM 출력 + 규칙 추출). 비교가 짝지어지고 "
        "비용은 한 번만 듭니다.",
        "- 역번역 대조는 모든 조건에서 슬롯 대조만 씁니다. LLM 동등성 판정은 간선을 보류할 수만 있고, 빼면 "
        "조건 간 차이가 추출에서만 생깁니다.",
        "- C는 고정 10종 결함 목록(스키마 열거형)에서만 고르고, 카탈로그에 없는 서비스 이름은 버립니다.",
    ]
    estimate = m.llm_cost_estimate
    if estimate is not None:
        lines += [
            "",
            f"전체 코퍼스 추정치(모델 `{estimate.model}`, 토큰 = 글자 수 / 2.5, 출력은 경험값):",
            "",
            *_table(
                ["조건", "호출", "입력 토큰", "출력 토큰"],
                [
                    [_label(c.condition.value), f"{c.calls:,}", f"{c.input_tokens:,}", f"{c.output_tokens:,}"]
                    for c in estimate.per_condition
                ]
                + [
                    [
                        "합계(D·DR 추출 공유)",
                        f"{estimate.total_calls:,}",
                        f"{estimate.total_input_tokens:,}",
                        f"{estimate.total_output_tokens:,}",
                    ]
                ],
            ),
        ]
    return lines


def _limitations(m: EvalMetrics) -> list[str]:
    keyword_rules = ", ".join(sorted(KEYWORD_SIGNALS))
    latency = (
        f"오프라인 조건을 작업 프로세스 {m.workers}개로 병렬 실행하며 잰 사례당 시간이라 단독 실행보다 "
        "길게 나옵니다."
        if m.workers > 1
        else "오프라인 조건을 한 프로세스에서 차례로 실행하며 잰 사례당 시간입니다."
    )
    fraction = m.reproducibility.fraction if m.reproducibility is not None else REPRODUCIBILITY_FRACTION
    return [
        "## 14. 한계",
        "",
        "- 생성된 코퍼스는 실제 사내 문서보다 규칙적입니다. 구어체도 생성기가 만든 변형이므로 "
        "실제 문서에서의 수치는 이보다 낮을 수 있습니다.",
        f"- 키워드 기준선(A)이 보는 규칙은 {keyword_rules} 입니다. 어휘는 일반적인 한국어 런북 "
        "표현으로 정했고 평가셋에 맞춰 조정하지 않았습니다. 서비스 구간은 정답(regions)에서 받으므로 "
        "실제보다 A에 유리합니다.",
        "- 조직 수준 발화도 탐지로 인정하는 기본 기준은 관대할 수 있어, 대상 서비스만 인정하는 엄격 기준을 "
        "9장에 함께 실었습니다.",
        "- 대조군 보정 탐지의 쌍둥이는 같은 조직·문체·배치의 정상 대조군 하나입니다. 1등급 규제 서비스가 "
        "둘인 조직에서는 결함 사례의 대상 서비스가 쌍둥이 문서에서 다른 자리(앞/뒤)에 놓일 수 있어, "
        "보정은 근사입니다.",
        "- 필드·환각 비율의 신뢰구간은 사실을 독립 표본으로 본 근사입니다(같은 사례 안의 사실은 서로 상관).",
        "- 봉인 연산자(감지 삭제, KISA 신고 단계 삭제)는 규칙 파서·규칙·프롬프트 개발과 조정에 쓰지 "
        "않았습니다.",
        f"- 지연 시간은 이 평가를 돌린 기기에서 {latency} 재현성 표본 비율은 {fraction * 100:g}%입니다.",
        "- C의 결함 유형 10종과 규칙 대응: "
        f"{', '.join(f'{d.value}→{r}' for d, r in DEFECT_RULE_IDS.items())}.",
    ]


def render_report(metrics: EvalMetrics) -> str:
    """report.md — Korean tables and computed interpretation of ``metrics``."""
    sections = [
        [
            f"# 제때옵스 봉인 평가 결과 ({metrics.generated_at})",
            "",
            "> 기억 말고 그래프로, 끊긴 곳은 장애 전에. 이 보고서의 모든 수치는 `metrics.json`에서 왔고, "
            "해석 문장도 수치로부터 자동 생성됩니다. 생성 일자는 실행자가 지정한 값입니다.",
        ],
        _headline(metrics),
        _design(metrics),
        _by_style(metrics),
        _by_operator(metrics),
        _by_position(metrics),
        _clean(metrics),
        _fields(metrics),
        _hallucination(metrics),
        _operations(metrics),
        _impact(metrics.impact_benchmark),
        _injection(metrics.injection_selftest),
        _interpretation(metrics),
        _llm_howto(metrics),
        _limitations(metrics),
    ]
    return "\n\n".join("\n".join(s) for s in sections if s) + "\n"


# --------------------------------------------------------------------------- files


def _record(result: CaseResult) -> dict[str, object]:
    def keys(values: frozenset[tuple[str | None, str]]) -> list[list[str | None]]:
        return [[s, r] for s, r in sorted(values, key=lambda k: (k[0] or "", k[1]))]

    return {
        "case_id": result.case_id,
        "condition": result.condition.value,
        "fired": keys(result.fired),
        "severe": keys(result.severe),
        "n_facts": None if result.facts is None else len(result.facts),
        "graph_hash": result.graph_hash,
        "ungrounded": result.ungrounded,
        "llm_calls": result.llm_calls,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "seconds": round(result.seconds, 6),
        "error": result.error,
    }


def results_jsonl(results: EvalResults) -> str:
    """One compact JSON verdict per (case, condition), in result order (facts as a count)."""
    return "".join(
        json.dumps(_record(r), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for r in results.results
    )


def write_report(
    results: EvalResults,
    out_dir: str | Path,
    *,
    generated_at: str,
    manifest: SealManifest | None = None,
    seed: int | None = None,
    reproducibility: Reproducibility | None = None,
    impact: ImpactBenchmark | None = None,
    injection: InjectionSelftest | None = None,
    cost_estimate: CostEstimate | None = None,
) -> EvalMetrics:
    """Write ``metrics.json``, ``report.md`` and ``results.jsonl`` into ``out_dir``."""
    metrics = compute_metrics(
        results,
        generated_at=generated_at,
        manifest=manifest,
        seed=seed,
        reproducibility=reproducibility,
        impact=impact,
        injection=injection,
        cost_estimate=cost_estimate,
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(metrics.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2)
    (out / "metrics.json").write_bytes((payload + "\n").encode("utf-8"))
    (out / "report.md").write_bytes(render_report(metrics).encode("utf-8"))
    (out / "results.jsonl").write_bytes(results_jsonl(results).encode("utf-8"))
    return metrics


def run_evaluation(
    out_dir: str | Path,
    *,
    generated_at: str,
    conditions: Iterable[Condition | str] = OFFLINE_CONDITIONS,
    llm: LLMClient | None = None,
    model: str | None = None,
    n_orgs: int = 20,
    seed: int = DEFAULT_SEED,
    workers: int | None = None,
    concurrency: int | None = None,
    benchmarks: bool = True,
    bundle: str | Path = DEFAULT_BUNDLE,
    progress: Callable[[int, int], None] | None = None,
) -> EvalMetrics:
    """The whole evaluation: corpus → ``manifest.json`` → conditions → reproducibility (when B
    runs) → benchmarks → report files. The LLM cost estimate of the full corpus is always
    computed (offline) and published so a paid run can be planned from the report."""
    chosen = parse_conditions(conditions)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = build_cases(generate_gold_orgs(n_orgs, seed), seed=seed)
    manifest = build_manifest(cases)
    write_manifest(manifest, out / "manifest.json")
    estimate = estimate_llm_cost(
        cases, LLM_CONDITIONS, model=model or get_settings().anthropic_model, echo=None
    )
    results = run_eval(
        cases,
        chosen,
        llm=llm,
        workers=workers,
        # ``is None``, not ``or``: an explicit 0 must be refused by run_eval, not silently replaced
        concurrency=get_settings().llm_concurrency if concurrency is None else concurrency,
        progress=progress,
    )
    reproducibility = check_reproducibility(results) if Condition.B_RULE_GRAPH in results.conditions else None
    return write_report(
        results,
        out,
        generated_at=generated_at,
        manifest=manifest,
        seed=seed,
        reproducibility=reproducibility,
        impact=impact_benchmark() if benchmarks else None,
        injection=injection_selftest(bundle) if benchmarks else None,
        cost_estimate=estimate,
    )


# --------------------------------------------------------------------------- command line


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m jettae.evalkit",
        description="제때옵스 봉인 평가: 코퍼스 생성·봉인 → 조건 실행 → metrics.json·report.md 작성",
    )
    parser.add_argument("--out", default="docs/eval", help="결과 폴더 (기본 docs/eval)")
    parser.add_argument("--generated-at", help="보고서에 적을 생성 일자 (예: 2026-10-13). 실행 시 필수")
    parser.add_argument(
        "--conditions",
        default=",".join(c.value for c in OFFLINE_CONDITIONS),
        help="쉼표로 구분한 조건 (기본 A_keyword,B_rule_graph,O_oracle)",
    )
    parser.add_argument("--orgs", type=int, default=20, help="조직 수 (기본 20, 스모크 시험은 1~2)")
    parser.add_argument("--workers", type=int, default=None, help="오프라인 조건 프로세스 수")
    parser.add_argument("--no-benchmarks", action="store_true", help="영향 범위·인젝션 시험 생략")
    parser.add_argument("--estimate-only", action="store_true", help="LLM 비용 추정만 출력하고 종료")
    parser.add_argument("--confirm-paid", action="store_true", help="LLM 조건의 유료 호출을 승인")
    return parser


def _parse(argv: Sequence[str] | None) -> tuple[argparse.Namespace, tuple[Condition, ...]] | str:
    """Arguments and conditions, or a Korean usage message (checked before any corpus is built)."""
    args = _parser().parse_args(argv)
    try:
        conditions = parse_conditions(c.strip() for c in args.conditions.split(",") if c.strip())
    except ValueError as exc:  # a typo in --conditions is a usage error, not a crash
        return str(exc)
    if args.orgs < 1:
        return f"--orgs는 1 이상이어야 합니다: {args.orgs}"
    if args.workers is not None and args.workers < 1:
        return f"--workers는 1 이상이어야 합니다: {args.workers}"
    return args, conditions


def _live_llm(args: argparse.Namespace, settings: Settings) -> LLMClient | str:
    """The paid client once every guard passed, else the Korean reason it may not be built."""
    if not args.confirm_paid:
        return "LLM 조건은 유료입니다. 위 추정치를 확인한 뒤 --confirm-paid로 다시 실행하세요."
    if settings.llm_provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        return "ANTHROPIC_API_KEY가 없어 LLM 조건을 실행할 수 없습니다."
    return build_llm(settings, LlmMode.LIVE)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code (2 = unusable arguments)."""
    parsed = _parse(argv)
    if isinstance(parsed, str):
        print(parsed, file=sys.stderr)
        return 2
    args, conditions = parsed
    llm_arms = [c for c in conditions if c in LLM_CONDITIONS]
    settings = get_settings()
    model = settings.openai_model if settings.llm_provider == "openai" else settings.anthropic_model
    if args.estimate_only or llm_arms:
        cases = build_cases(generate_gold_orgs(args.orgs))
        estimate_llm_cost(cases, llm_arms or LLM_CONDITIONS, model=model)
        if args.estimate_only:
            return 0
    if args.generated_at is None:
        print("--generated-at 날짜를 지정해 주세요 (보고서는 시계를 읽지 않습니다).", file=sys.stderr)
        return 2
    llm = _live_llm(args, settings) if llm_arms else None
    if isinstance(llm, str):
        print(llm, file=sys.stderr)
        return 2

    def progress(done: int, total: int) -> None:
        print(f"\r진행 {done:,}/{total:,}", end="", file=sys.stderr, flush=True)

    metrics = run_evaluation(
        args.out,
        generated_at=args.generated_at,
        conditions=conditions,
        llm=llm,
        model=model,
        n_orgs=args.orgs,
        workers=args.workers,
        benchmarks=not args.no_benchmarks,
        progress=progress,
    )
    print(file=sys.stderr)
    for name, summary in metrics.conditions.items():
        print(
            f"{name}: 탐지율(활성) {_pct(summary.detection_active.rate)}, "
            f"탐지율(봉인) {_pct(summary.detection_held_out.rate)}, 오탐률 {_pct(summary.clean_fp.rate)}, "
            f"오류 {summary.errors}건"
        )
    print(f"결과: {Path(args.out).as_posix()}/metrics.json, report.md, results.jsonl, manifest.json")
    return 0
