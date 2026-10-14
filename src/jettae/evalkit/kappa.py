"""Inter-rater agreement: Cohen's κ and Krippendorff's α (nominal), without scikit-learn.

Why we need agreement statistics in a set that has no human raters: the fault-injection corpus
is labelled by construction, but the *additional* real-world sample (and any LLM-as-reviewer
comparison) is not. Before such labels are trusted, two raters' agreement is reported. Both
statistics are computed with exact rational arithmetic (:class:`fractions.Fraction`), so the
result is identical on every platform and independent of category iteration order (P6).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Sequence
from fractions import Fraction


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Cohen's κ for two raters labelling the same items.

    κ = (p_o − p_e) / (1 − p_e). When both raters put every item into one and the same
    category, p_e = 1 and κ is 0/0; that is perfect agreement, so 1.0 is returned.

    Raises ``ValueError`` when the sequences differ in length or are empty.
    """
    if len(a) != len(b):
        raise ValueError(f"두 평가자의 판정 수가 다릅니다: {len(a)} ≠ {len(b)}")
    n = len(a)
    if n == 0:
        raise ValueError("판정이 하나도 없어 일치도를 계산할 수 없습니다")
    observed = Fraction(sum(1 for x, y in zip(a, b, strict=True) if x == y), n)
    count_a, count_b = Counter(a), Counter(b)
    expected = sum((Fraction(count_a[k] * count_b[k], n * n) for k in count_a if k in count_b), Fraction(0))
    if expected == 1:
        return 1.0
    return float((observed - expected) / (1 - expected))


def krippendorff_alpha_nominal(reliability_data: Sequence[Sequence[Hashable | None]]) -> float:
    """Krippendorff's α for nominal data; rows are raters, columns are units, ``None`` = missing.

    Built from the coincidence matrix (Krippendorff 2011, "Computing Krippendorff's
    Alpha-Reliability"): units with fewer than two values are not pairable and are skipped.
    α = 1 − (n − 1) · Σ_{c≠k} o_ck / Σ_{c≠k} n_c n_k. If only one category occurs among the
    pairable values there is nothing to disagree about and 1.0 is returned.

    Raises ``ValueError`` when rows differ in length or no unit has two values.
    """
    if not reliability_data:
        raise ValueError("평가자 데이터가 없습니다")
    n_units = len(reliability_data[0])
    if any(len(row) != n_units for row in reliability_data):
        raise ValueError("평가자마다 평가 단위 수가 다릅니다")
    #: coincidence matrix o_ck (ordered value pairs within units, weighted by 1 / (m_u − 1))
    weights: dict[tuple[Hashable, Hashable], Fraction] = {}
    for unit in range(n_units):
        values = Counter(row[unit] for row in reliability_data if row[unit] is not None)
        m_u = sum(values.values())
        if m_u < 2:
            continue
        for c, n_uc in values.items():
            for k, n_uk in values.items():
                pairs = n_uc * (n_uk - (1 if c == k else 0))
                if pairs:
                    weights[(c, k)] = weights.get((c, k), Fraction(0)) + Fraction(pairs, m_u - 1)
    if not weights:
        raise ValueError("두 개 이상의 값이 있는 평가 단위가 없어 α를 계산할 수 없습니다")
    marginals: dict[Hashable, Fraction] = {}
    for (c, _), weight in weights.items():
        marginals[c] = marginals.get(c, Fraction(0)) + weight
    n = sum(marginals.values(), Fraction(0))
    disagreement = sum((w for (c, k), w in weights.items() if c != k), Fraction(0))
    chance = sum(
        (marginals[c] * marginals[k] for c in marginals for k in marginals if c != k),
        Fraction(0),
    )
    if chance == 0:
        return 1.0
    return float(1 - (n - 1) * disagreement / chance)
