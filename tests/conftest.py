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
