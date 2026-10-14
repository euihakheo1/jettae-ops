"""Cohen's κ and Krippendorff's α (nominal) against textbook values and algebraic properties."""

from __future__ import annotations

import math
from collections import Counter

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from jettae.evalkit.kappa import cohen_kappa, krippendorff_alpha_nominal

# Krippendorff (2011), "Computing Krippendorff's Alpha-Reliability", section C:
# 4 observers × 12 units, nominal values, "." = missing. Published α_nominal = 0.743.
KRIPPENDORFF_2011_NOMINAL: list[list[int | None]] = [
    [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None],
    [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3],
    [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None],
    [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None],
]
# Same paper, section A: 2 observers × 10 units, binary. Published α = 0.095.
KRIPPENDORFF_2011_BINARY: list[list[int | None]] = [
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 0, 0, 1, 0, 0, 0, 0],
]

labels = st.sampled_from(["pass", "broken", "unknown", "hold"])


def _paired(min_size: int = 1) -> st.SearchStrategy[tuple[list[str], list[str]]]:
    return st.lists(st.tuples(labels, labels), min_size=min_size, max_size=60).map(
        lambda pairs: ([a for a, _ in pairs], [b for _, b in pairs])
    )


def _confusion(yes_yes: int, yes_no: int, no_yes: int, no_no: int) -> tuple[list[str], list[str]]:
    a = ["y"] * (yes_yes + yes_no) + ["n"] * (no_yes + no_no)
    b = ["y"] * yes_yes + ["n"] * yes_no + ["y"] * no_yes + ["n"] * no_no
    return a, b


# --------------------------------------------------------------------------- Cohen's kappa


@pytest.mark.parametrize(
    ("cells", "expected"),
    [
        ((20, 5, 10, 15), 0.4),  # p_o = 0.70, p_e = 0.50
        ((45, 15, 25, 15), 0.1304),  # p_o = 0.60, p_e = 0.54
        ((25, 35, 5, 35), 0.2593),  # p_o = 0.60, p_e = 0.46
    ],
)
def test_cohen_kappa_textbook_tables(cells: tuple[int, int, int, int], expected: float) -> None:
    a, b = _confusion(*cells)
    assert cohen_kappa(a, b) == pytest.approx(expected, abs=1e-4)


def test_cohen_kappa_perfect_and_opposite_agreement() -> None:
    assert cohen_kappa(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert cohen_kappa([0, 1, 0, 1], [1, 0, 1, 0]) == -1.0


def test_cohen_kappa_degenerate_single_category_is_perfect_agreement() -> None:
    assert cohen_kappa(["pass"] * 5, ["pass"] * 5) == 1.0


def test_cohen_kappa_chance_level_is_zero() -> None:
    assert cohen_kappa(["y", "y", "n", "n"], ["y", "n", "y", "n"]) == 0.0


def test_cohen_kappa_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="판정 수가 다릅니다"):
        cohen_kappa([1, 2], [1])
    with pytest.raises(ValueError, match="하나도 없어"):
        cohen_kappa([], [])


@settings(max_examples=200, deadline=None)
@given(_paired())
def test_cohen_kappa_properties(pair: tuple[list[str], list[str]]) -> None:
    a, b = pair
    k = cohen_kappa(a, b)
    assert -1.0 - 1e-12 <= k <= 1.0 + 1e-12
    assert k == cohen_kappa(b, a)
    assert cohen_kappa(a, a) == 1.0
    relabel = {"pass": "통과", "broken": "끊김", "unknown": "확인 불가", "hold": "보류"}
    assert k == cohen_kappa([relabel[x] for x in a], [relabel[x] for x in b])
    order = sorted(range(len(a)), key=lambda i: (a[i], b[i], i))
    assert k == pytest.approx(cohen_kappa([a[i] for i in order], [b[i] for i in order]), abs=1e-12)


# --------------------------------------------------------------------------- Krippendorff's alpha


def test_alpha_matches_krippendorff_2011_nominal_example() -> None:
    assert krippendorff_alpha_nominal(KRIPPENDORFF_2011_NOMINAL) == pytest.approx(0.743, abs=5e-4)


def test_alpha_matches_krippendorff_2011_binary_example() -> None:
    assert krippendorff_alpha_nominal(KRIPPENDORFF_2011_BINARY) == pytest.approx(0.095, abs=5e-4)


def test_alpha_perfect_agreement_and_degenerate_category() -> None:
    assert krippendorff_alpha_nominal([[1, 2, 3], [1, 2, 3], [1, 2, None]]) == 1.0
    assert krippendorff_alpha_nominal([["x", "x"], ["x", "x"]]) == 1.0


def test_alpha_skips_units_without_pairable_values() -> None:
    with_lonely_units = [[1, 2, 7, None], [1, 1, None, None]]
    assert krippendorff_alpha_nominal(with_lonely_units) == krippendorff_alpha_nominal([[1, 2], [1, 1]])


def test_alpha_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="평가자 데이터가 없습니다"):
        krippendorff_alpha_nominal([])
    with pytest.raises(ValueError, match="단위 수가 다릅니다"):
        krippendorff_alpha_nominal([[1, 2], [1]])
    with pytest.raises(ValueError, match="계산할 수 없습니다"):
        krippendorff_alpha_nominal([[1, None], [None, 2]])


def test_alpha_hand_computed_systematic_disagreement() -> None:
    # units (1,2) and (2,1): o_12 = o_21 = 2, n_1 = n_2 = 2, n = 4 -> α = 1 − 3·4 / (2·2·2) = −0.5
    assert krippendorff_alpha_nominal([[1, 2], [2, 1]]) == -0.5


def test_alpha_rejects_empty_and_all_missing_units() -> None:
    with pytest.raises(ValueError, match="계산할 수 없습니다"):
        krippendorff_alpha_nominal([[]])
    with pytest.raises(ValueError, match="계산할 수 없습니다"):
        krippendorff_alpha_nominal([[None, None], [None, None], [None, None]])


def test_agreement_accepts_any_hashable_labels() -> None:
    labels_a = [("svc", "broken"), ("svc", "pass"), ("db", "unknown"), ("db", "pass")]
    labels_b = [("svc", "broken"), ("svc", "broken"), ("db", "unknown"), ("db", "pass")]
    as_text = [f"{x}/{y}" for x, y in labels_a], [f"{x}/{y}" for x, y in labels_b]
    assert cohen_kappa(labels_a, labels_b) == cohen_kappa(*as_text)
    assert krippendorff_alpha_nominal([labels_a, labels_b]) == krippendorff_alpha_nominal(list(as_text))


def test_cohen_kappa_three_way_hand_computed() -> None:
    # p_o = 3/6, marginals a = b = (2, 2, 2) -> p_e = 1/3 -> κ = (1/2 − 1/3) / (2/3) = 1/4
    assert cohen_kappa(list("aabbcc"), list("abbcca")) == 0.25
    # p_o = 0 with uniform marginals -> κ = (0 − 1/3) / (2/3) = −0.5
    assert cohen_kappa(list("abc"), list("bca")) == -0.5


@settings(max_examples=200, deadline=None)
@given(_paired(min_size=2))
def test_alpha_two_raters_relates_to_scotts_pi(pair: tuple[list[str], list[str]]) -> None:
    """For two complete raters: α = 1 − (2N−1)/(2N) · (1 − π) with Scott's π."""
    a, b = pair
    n = len(a)
    pooled = Counter(a) + Counter(b)
    p_e = sum((c / (2 * n)) ** 2 for c in pooled.values())
    assume(p_e < 1)
    p_o = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pi = (p_o - p_e) / (1 - p_e)
    expected = 1 - (2 * n - 1) / (2 * n) * (1 - pi)
    assert krippendorff_alpha_nominal([a, b]) == pytest.approx(expected, abs=1e-9)


@settings(max_examples=150, deadline=None)
@given(
    st.integers(2, 4).flatmap(
        lambda raters: st.lists(
            st.lists(st.one_of(st.none(), labels), min_size=raters, max_size=raters), min_size=2, max_size=25
        )
    )
)
def test_alpha_is_invariant_to_order_and_labels(units: list[list[str | None]]) -> None:
    data = [list(row) for row in zip(*units, strict=True)]  # raters × units
    try:
        alpha = krippendorff_alpha_nominal(data)
    except ValueError:
        assume(False)
        return
    assert math.isfinite(alpha)
    assert alpha <= 1.0 + 1e-12
    assert krippendorff_alpha_nominal(list(reversed(data))) == pytest.approx(alpha, abs=1e-12)
    assert krippendorff_alpha_nominal([list(reversed(r)) for r in data]) == pytest.approx(alpha, abs=1e-12)
    relabel = {"pass": 1, "broken": 2, "unknown": 3, "hold": 4}
    renamed = [[None if v is None else relabel[v] for v in row] for row in data]
    assert krippendorff_alpha_nominal(renamed) == pytest.approx(alpha, abs=1e-12)
