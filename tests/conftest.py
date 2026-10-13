"""Shared helpers for building small hand-made transaction tables."""

import pandas as pd

# One ordinary purchase line; tests override only the fields they care about.
DEFAULT_LINE = {
    "invoice": "500001",
    "stock_code": "85123A",
    "description": "WHITE HANGING HEART T-LIGHT HOLDER",
    "quantity": 6,
    "invoice_date": "2010-06-01 10:00",
    "price": 2.55,
    "customer_id": 13085,
    "country": "United Kingdom",
    "sheet": "Year 2009-2010",
}


def make_lines(*overrides):
    """Build a cleaned-schema DataFrame, one row per dict of overrides."""
    rows = [{**DEFAULT_LINE, **o} for o in overrides]
    df = pd.DataFrame(rows)
    df["invoice"] = df["invoice"].astype("str")
    df["stock_code"] = df["stock_code"].astype("str")
    df["description"] = df["description"].astype("str")
    df["quantity"] = df["quantity"].astype("int64")
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], format="ISO8601")
    df["price"] = df["price"].astype("float64")
    df["customer_id"] = df["customer_id"].astype("Int64")
    df["country"] = df["country"].astype("str")
    df["sheet"] = df["sheet"].astype("str")
    return df


def make_clean(*overrides):
    """Like make_lines, plus the two columns C1 adds (line_value, is_return)."""
    df = make_lines(*overrides)
    df["line_value"] = df["quantity"] * df["price"]
    df["is_return"] = df["invoice"].str.startswith("C") | (df["quantity"] < 0)
    return df


def make_snapshot(n, cutoff, seed=0):
    """A synthetic snapshot table with the C3 columns: plausible feature ranges, and churn more
    likely for customers with a long recency. For model tests only (no real data)."""
    import numpy as np

    from churn.snapshots import FEATURES

    rng = np.random.default_rng(seed)
    recency = rng.integers(1, 366, n)
    orders = rng.integers(1, 30, n)
    df = pd.DataFrame({name: rng.uniform(0, 5, n) for name in FEATURES})
    df["recency_days"] = recency
    df["orders_365d"] = orders
    df["orders_180d"] = np.minimum(orders, rng.integers(0, 15, n))
    df["orders_90d"] = np.minimum(df["orders_180d"], rng.integers(0, 8, n))
    df["net_revenue_365d"] = rng.normal(800, 2000, n)
    df["net_revenue_trend_90d"] = rng.normal(0, 500, n)
    df["orders_trend_90d"] = rng.integers(-5, 6, n)
    df["recency_gap_ratio"] = rng.exponential(2, n)
    for flag in ["single_purchase_day_365d", "first_seen_first_data_month", "is_uk", "bought_in_peak_365d"]:
        df[flag] = rng.integers(0, 2, n)
    df.insert(0, "customer_id", np.arange(10000, 10000 + n))
    df.insert(0, "cutoff", pd.Timestamp(cutoff))
    p_churn = 1 / (1 + np.exp(-(recency - 120) / 40))
    df["churned"] = (rng.uniform(0, 1, n) < p_churn).astype("int64")
    return df
