"""Metric tests on tiny hand-worked examples (the arithmetic is written out in each test)."""

import numpy as np
import pytest

from churn.metrics import (
    bootstrap_intervals,
    brier_score,
    calibration_table,
    evaluate,
    top_fraction_stats,
    top_k_count,
)

# Four customers. Churners score 0.9 and 0.7, non-churners 0.8 and 0.1.
Y4 = [1, 0, 1, 0]
S4 = [0.9, 0.8, 0.7, 0.1]


def test_roc_auc_and_pr_auc_by_hand():
    m = evaluate(Y4, S4, top_fractions=[0.5], n_bins=10, is_probability=True)
    # AUC: churner-vs-non-churner pairs ranked correctly: (0.9>0.8), (0.9>0.1), (0.7>0.1); not (0.7>0.8).
    assert m["roc_auc"] == pytest.approx(3 / 4)
    # Ranked: churner (precision 1, recall 1/2), non-churner, churner (precision 2/3, recall 1).
    # PR-AUC (average precision) = 1/2 * 1 + 1/2 * 2/3.
    assert m["pr_auc"] == pytest.approx(0.5 + 1 / 3)
    assert m["customers"] == 4 and m["churners"] == 2 and m["base_rate"] == 0.5


def test_brier_score_by_hand():
    # Squared errors: 0.1^2, 0.8^2, 0.3^2, 0.1^2 = 0.01 + 0.64 + 0.09 + 0.01 = 0.75; mean 0.1875.
    assert brier_score(Y4, S4) == pytest.approx(0.1875)
    assert evaluate(Y4, S4, [0.5], 10, True)["brier"] == pytest.approx(0.1875)


def test_mean_predicted_is_compared_with_base_rate():
    m = evaluate(Y4, S4, [0.5], 10, True)
    assert m["mean_predicted"] == pytest.approx((0.9 + 0.8 + 0.7 + 0.1) / 4)   # 0.625 vs base rate 0.5
    assert evaluate(Y4, [1, 1, 0, 0], [0.5], 10, False)["mean_predicted"] is None


@pytest.mark.parametrize("n, fraction, k", [(10, 0.1, 1), (10, 0.2, 2), (5, 0.1, 1),
                                            (4317, 0.1, 432), (4317, 0.2, 864), (100, 0.2, 20)])
def test_top_k_is_ceiling_of_fraction(n, fraction, k):
    assert top_k_count(n, fraction) == k


def test_precision_and_recall_in_top_fraction_by_hand():
    # Ten customers ranked by score; four churners at ranks 1, 3, 4, 7.
    y = [1, 0, 1, 1, 0, 0, 1, 0, 0, 0]
    s = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    top10 = top_fraction_stats(y, s, 0.1)   # k = 1: rank 1 is a churner
    assert (top10["k"], top10["churners_in_top"]) == (1, 1)
    assert top10["precision"] == 1.0 and top10["recall"] == 0.25
    top20 = top_fraction_stats(y, s, 0.2)   # k = 2: ranks 1-2 hold one churner
    assert (top20["k"], top20["churners_in_top"]) == (2, 1)
    assert top20["precision"] == 0.5 and top20["recall"] == 0.25
    top40 = top_fraction_stats(y, s, 0.4)   # k = 4: ranks 1-4 hold three churners
    assert top40["precision"] == 0.75 and top40["recall"] == 0.75


def test_ties_at_the_boundary_share_the_places():
    # k = ceil(0.4 x 5) = 2. One customer above the tie (a churner) takes one place; the other place
    # goes to the three customers tied at 0.5, one of whom is a churner: expected 1/3 churner.
    y = [1, 1, 0, 0, 0]
    s = [0.9, 0.5, 0.5, 0.5, 0.1]
    t = top_fraction_stats(y, s, 0.4)
    assert t["k"] == 2 and t["boundary_ties"] == 3
    assert t["churners_in_top"] == pytest.approx(4 / 3)
    assert t["precision"] == pytest.approx(2 / 3)   # (4/3) / 2
    assert t["recall"] == pytest.approx(2 / 3)      # (4/3) / 2 churners
    # Row order does not matter.
    order = [4, 2, 0, 3, 1]
    t2 = top_fraction_stats([y[i] for i in order], [s[i] for i in order], 0.4)
    assert t2["churners_in_top"] == pytest.approx(t["churners_in_top"])


def test_calibration_table_by_hand():
    p = [0.05, 0.1, 0.15, 0.95, 1.0]
    y = [0, 1, 0, 1, 1]
    table = calibration_table(y, p, n_bins=10)
    assert len(table) == 10
    assert table[0] == {"bin": 1, "lower": 0.0, "upper": 0.1, "customers": 1,
                        "mean_predicted": 0.05, "observed_churn_rate": 0.0}
    # 0.1 belongs to the second bin [0.1, 0.2), with 0.15.
    assert table[1]["customers"] == 2
    assert table[1]["mean_predicted"] == pytest.approx(0.125)
    assert table[1]["observed_churn_rate"] == 0.5
    # Empty bins are kept with a count of 0; 1.0 falls in the last bin [0.9, 1.0].
    assert table[5]["customers"] == 0 and table[5]["mean_predicted"] is None
    assert table[9]["customers"] == 2 and table[9]["observed_churn_rate"] == 1.0
    assert sum(r["customers"] for r in table) == 5


def test_calibration_rejects_values_outside_zero_one():
    with pytest.raises(ValueError):
        calibration_table([0, 1], [0.2, 1.2], n_bins=10)


def test_evaluate_lists_top_fractions_and_calibration():
    y = [1, 0, 1, 1, 0, 0, 1, 0, 0, 0]
    s = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    m = evaluate(y, s, top_fractions=[0.1, 0.2], n_bins=10, is_probability=True)
    assert [t["fraction"] for t in m["top"]] == [0.1, 0.2]
    assert m["top"][1]["precision"] == 0.5
    assert m["base_rate"] == 0.4
    assert len(m["calibration"]) == 10


def test_evaluate_skips_probability_metrics_for_yes_no_rule():
    m = evaluate([1, 0, 1, 0], [1, 1, 0, 0], [0.5], 10, is_probability=False)
    assert m["brier"] is None and m["calibration"] is None
    # Yes/no score: one of two flagged is a churner; AUC = (recall + specificity) / 2 = (0.5 + 0.5) / 2.
    assert m["roc_auc"] == pytest.approx(0.5)


def test_evaluate_needs_both_classes_and_equal_lengths():
    with pytest.raises(ValueError):
        evaluate([1, 1], [0.2, 0.3], [0.1], 10, True)
    with pytest.raises(ValueError):
        evaluate([1, 0], [0.2], [0.1], 10, True)


def test_bootstrap_is_reproducible_and_brackets_the_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    s = np.clip(y * 0.3 + rng.uniform(0, 0.7, 400), 0, 1)
    a = bootstrap_intervals(y, s, top_fractions=[0.1, 0.2], n_samples=200, seed=7)
    b = bootstrap_intervals(y, s, top_fractions=[0.1, 0.2], n_samples=200, seed=7)
    assert a == b
    assert set(a["intervals"]) == {"roc_auc", "pr_auc", "precision_top_10pct", "precision_top_20pct"}
    point = evaluate(y, s, [0.1, 0.2], 10, True)
    assert a["intervals"]["roc_auc"]["low"] <= point["roc_auc"] <= a["intervals"]["roc_auc"]["high"]
    assert a["intervals"]["pr_auc"]["low"] < a["intervals"]["pr_auc"]["high"]
    assert a["samples_used"] == 200 and a["level"] == 0.95


def test_bootstrap_of_perfect_ranking_is_exact():
    y = [0] * 50 + [1] * 50
    s = list(np.linspace(0, 1, 100))
    out = bootstrap_intervals(y, s, top_fractions=[0.1], n_samples=100, seed=1)
    assert out["intervals"]["roc_auc"] == {"low": pytest.approx(1.0), "high": pytest.approx(1.0)}
    assert out["intervals"]["precision_top_10pct"] == {"low": 1.0, "high": 1.0}
