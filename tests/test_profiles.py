"""Risk-list phrases: each customer's most unusual features compared with the median customer."""

import pandas as pd
import pytest

from churn.profiles import describe, unusual_features

SPECS = {
    "orders_365d": {"label": "orders in the past year", "format": "count"},
    "net_revenue_365d": {"label": "net revenue in the past 12 months", "format": "gbp"},
    "avg_gap_days": {"label": "usual days between purchases", "format": "days", "repeat_buyers_only": True},
}

# Five customers. Customer 5 has far more orders and revenue than anyone else; customer 1 bought
# on one day only (gap 365 is a fill value).
REF = pd.DataFrame({
    "customer_id": [1, 2, 3, 4, 5],
    "orders_365d": [1, 2, 2, 3, 40],
    "net_revenue_365d": [100.0, 200.0, 300.0, 400.0, 9000.0],
    "avg_gap_days": [365.0, 60.0, 50.0, 40.0, 5.0],
    "single_purchase_day_365d": [1, 0, 0, 0, 0],
})


def test_share_counts_customers_strictly_below_or_above():
    out = unusual_features(REF, REF, SPECS, n=3)
    top = {d["feature"]: d for d in out[4]}            # customer 5
    assert top["orders_365d"]["direction"] == "higher"
    assert top["orders_365d"]["share"] == pytest.approx(0.8)   # 4 of 5 strictly below
    assert top["orders_365d"]["median"] == 2


def test_ties_at_the_median_are_not_unusual():
    out = unusual_features(REF, REF, {"orders_365d": SPECS["orders_365d"]}, n=1)
    # Customer 2 has orders 2 = median: 1 of 5 below, 2 of 5 above -> share 0.4 ("lower than 40%").
    assert out[1][0]["share"] == pytest.approx(0.4) and out[1][0]["direction"] == "lower"


def test_returns_n_features_sorted_by_share_then_config_order():
    out = unusual_features(REF, REF, SPECS, n=2)
    assert len(out[4]) == 2
    # Customer 5: orders 0.8 higher, revenue 0.8 higher (tie: config order), gap 0.75 lower.
    assert [d["feature"] for d in out[4]] == ["orders_365d", "net_revenue_365d"]


def test_gap_features_skipped_for_one_day_buyers_and_compared_with_repeat_buyers_only():
    out = unusual_features(REF, REF, SPECS, n=3)
    assert "avg_gap_days" not in [d["feature"] for d in out[0]]
    assert len(out[0]) == 2
    gap = next(d for d in out[4] if d["feature"] == "avg_gap_days")
    # Customer 5 (gap 5) vs the 4 repeat buyers (60, 50, 40, 5): 3 of 4 strictly above.
    assert gap["direction"] == "lower" and gap["share"] == pytest.approx(0.75) and gap["median"] == 45


def test_medians_come_from_the_reference_rows():
    one = REF.iloc[[4]]
    out = unusual_features(one, REF, {"orders_365d": SPECS["orders_365d"]}, n=1)
    assert out[0][0]["median"] == 2 and out[0][0]["share"] == pytest.approx(0.8)


def test_describe_in_plain_words():
    item = {"feature": "net_revenue_365d", "value": 9000.0, "median": 300.0, "direction": "higher", "share": 0.8}
    assert describe(item, SPECS["net_revenue_365d"]) == (
        "net revenue in the past 12 months £9,000 (median £300; higher than 80% of customers)")
    item = {"feature": "orders_365d", "value": 1, "median": 2, "direction": "lower", "share": 0.6}
    assert describe(item, SPECS["orders_365d"]) == (
        "orders in the past year 1 (median 2; lower than 60% of customers)")
    pct = {"label": "share of value returned", "format": "pct"}
    item = {"feature": "x", "value": 0.25, "median": 0.0, "direction": "higher", "share": 0.9}
    assert describe(item, pct) == "share of value returned 25% (median 0%; higher than 90% of customers)"
