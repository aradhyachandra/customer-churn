"""C6 summaries: churn by segment, predicted vs actual by group, near-best campaign sizes, value
concentration."""

import numpy as np
import pandas as pd
import pytest

from churn.findings import near_best_sizes, predicted_vs_actual, segment_churn, value_concentration

ROWS = pd.DataFrame({
    "cutoff": pd.to_datetime(["2011-01-01"] * 4 + ["2011-02-01"] * 4),
    "is_uk": [1, 1, 0, 0, 1, 1, 0, 0],
    "orders_365d": [1, 12, 3, 1, 1, 2, 10, 5],
    "churned": [1, 0, 1, 1, 1, 0, 0, 0],
})
SEGMENTS = {
    "is_uk": {"title": "Country group", "labels": {1: "United Kingdom", 0: "Rest of the world"}},
    "orders_365d": {"title": "Orders", "edges": [1, 2, 10, float("inf")], "labels": ["1", "2 to 9", "10 or more"]},
}


def test_segment_churn_counts_rates_and_per_cutoff_range():
    t = segment_churn(ROWS, SEGMENTS)
    uk = t[(t["segment"] == "is_uk") & (t["group"] == "United Kingdom")].iloc[0]
    assert (uk["rows"], uk["churned"]) == (4, 2)
    assert uk["churn_rate"] == pytest.approx(0.5)
    assert (uk["churn_rate_min"], uk["churn_rate_max"]) == (0.5, 0.5)
    one = t[(t["segment"] == "orders_365d") & (t["group"] == "1")].iloc[0]
    assert (one["rows"], one["churned"]) == (3, 3)
    ten = t[(t["segment"] == "orders_365d") & (t["group"] == "10 or more")].iloc[0]
    assert (ten["rows"], ten["churned"]) == (2, 0)       # 10 is in the band (lower edge included)
    assert list(t[t["segment"] == "orders_365d"]["group"]) == ["1", "2 to 9", "10 or more"]


def test_segment_bands_must_cover_every_row():
    bad = {"orders_365d": {"title": "Orders", "edges": [2, 10], "labels": ["2 to 9"]}}
    with pytest.raises(ValueError, match="outside"):
        segment_churn(ROWS, bad)


def test_predicted_vs_actual_by_group():
    p_model = np.array([0.9, 0.5, 0.4, 0.2])
    p_rescaled = np.array([0.8, 0.4, 0.3, 0.1])
    y = np.array([1, 0, 0, 0])
    chosen = np.array([True, True, False, False])
    t = predicted_vs_actual(p_model, p_rescaled, y, chosen).set_index("group")
    assert list(t.index) == ["everyone", "chosen", "not_chosen"]
    assert t.loc["everyone", "customers"] == 4
    assert t.loc["chosen", "mean_p_rescaled"] == pytest.approx(0.6)
    assert t.loc["chosen", "observed_churn_rate"] == pytest.approx(0.5)
    assert t.loc["chosen", "gap_points_rescaled"] == pytest.approx(10.0)
    assert t.loc["chosen", "ratio_rescaled"] == pytest.approx(1.2)
    assert t.loc["not_chosen", "churned"] == 0


def test_near_best_sizes_is_the_range_above_the_share_of_the_best():
    curve = np.array([0.0, 5.0, 9.0, 10.0, 9.5, 8.0, 2.0])
    r = near_best_sizes(curve, 0.9)
    assert r == {"best_size": 3, "best": 10.0, "share": 0.9, "min_size": 2, "max_size": 4}


def test_value_concentration():
    value = np.array([0.0] * 90 + [10.0] * 9 + [910.0])
    c = value_concentration(value, [0.01, 0.10])
    assert c[0] == {"top_share_of_customers": 0.01, "customers": 1, "value_share": pytest.approx(0.91)}
    assert c[1]["customers"] == 10 and c[1]["value_share"] == pytest.approx(1.0)
