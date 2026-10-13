"""Cleaning rules from PROJECT_PLAN.md Section 2.1, applied in a fixed order with a count at each step.

Order (a row is counted at the first step that removes it):
  1. drop_sheet_overlap            later-sheet lines identical to a line in the earlier sheet
  2. drop_within_sheet_duplicates  exact duplicate lines inside one sheet (first copy kept)
  3. drop_non_positive_price       price <= 0
  4. drop_non_product_codes        stock codes listed in config/cleaning.yaml
  5. drop_missing_customer_id      no Customer ID
Returns (cancellation invoices and negative quantities) are kept and flagged with is_return.
"""

import pandas as pd

# The eight original columns; two lines are "identical" when all of these match.
KEY_COLUMNS = [
    "invoice", "stock_code", "description", "quantity",
    "invoice_date", "price", "customer_id", "country",
]


def line_value(df):
    """Line revenue in GBP: quantity x price (negative for returns)."""
    return df["quantity"] * df["price"]


def _row_keys(df):
    """One 64-bit hash per row over KEY_COLUMNS; missing values hash consistently."""
    return pd.util.hash_pandas_object(df[KEY_COLUMNS], index=False)


def _ts(value):
    return str(pd.Timestamp(value))


def sheet_overlap(df, sheet_order):
    """Measure how the two sheets overlap. Returns (stats, mask of later-sheet rows to drop)."""
    earlier, later = sheet_order
    keys = _row_keys(df)
    in_earlier = df["sheet"] == earlier
    in_later = df["sheet"] == later

    stats = {"sheets": {}}
    for name, mask in [(earlier, in_earlier), (later, in_later)]:
        dates = df.loc[mask, "invoice_date"]
        stats["sheets"][name] = {
            "rows": int(mask.sum()),
            "first_date": _ts(dates.min()),
            "last_date": _ts(dates.max()),
            "within_sheet_duplicate_rows": int(keys[mask].duplicated().sum()),
        }

    start = max(df.loc[in_earlier, "invoice_date"].min(), df.loc[in_later, "invoice_date"].min())
    end = min(df.loc[in_earlier, "invoice_date"].max(), df.loc[in_later, "invoice_date"].max())
    if start > end:
        stats["overlap_date_range"] = None
        in_range = pd.Series(False, index=df.index)
    else:
        stats["overlap_date_range"] = {"start": _ts(start), "end": _ts(end)}
        in_range = df["invoice_date"].between(start, end)

    later_matches = in_later & keys.isin(keys[in_earlier].unique())
    earlier_matches = in_earlier & keys.isin(keys[in_later].unique())
    stats["rows_in_overlap_range"] = {
        earlier: int((in_earlier & in_range).sum()),
        later: int((in_later & in_range).sum()),
    }
    stats["later_rows_identical_to_earlier_sheet"] = int(later_matches.sum())
    stats["later_rows_identical_outside_overlap_range"] = int((later_matches & ~in_range).sum())
    stats["earlier_rows_in_range_without_match"] = int((in_earlier & in_range & ~earlier_matches).sum())
    stats["later_rows_in_range_without_match"] = int((in_later & in_range & ~later_matches).sum())
    return stats, later_matches


def within_sheet_duplicates_mask(df):
    return df.duplicated(subset=KEY_COLUMNS + ["sheet"], keep="first")


def non_positive_price_mask(df):
    return df["price"] <= 0


def non_product_mask(df, codes):
    return df["stock_code"].isin(list(codes))


def missing_customer_mask(df):
    return df["customer_id"].isna()


def is_return(df, cancellation_prefix):
    return df["invoice"].str.startswith(cancellation_prefix) | (df["quantity"] < 0)


def _money(x):
    return round(float(x), 2)


def _drop(df, mask, step, rule, steps, sheet_order):
    removed_values = line_value(df[mask])
    removed_sheets = df.loc[mask, "sheet"]
    steps.append({
        "step": step,
        "rule": rule,
        "rows_before": len(df),
        "rows_removed": int(mask.sum()),
        "rows_after": int((~mask).sum()),
        "rows_removed_by_sheet": {name: int((removed_sheets == name).sum()) for name in sheet_order},
        "net_value_removed": _money(removed_values.sum()),
        "sales_value_removed": _money(removed_values[removed_values > 0].sum()),
    })
    return df[~mask]


def _totals(df):
    values = line_value(df)
    return {
        "rows": len(df),
        "net_value": _money(values.sum()),
        "rows_by_sheet": {k: int(v) for k, v in df["sheet"].value_counts().sort_index().items()},
    }


def clean_transactions(raw, sheet_order, rules):
    """Apply the cleaning rules. Returns (clean DataFrame, list of step records, summary dict)."""
    codes = rules["non_product_stock_codes"]
    prefix = rules["cancellation_prefix"]
    steps = []

    overlap, overlap_mask = sheet_overlap(raw, sheet_order)
    df = _drop(raw, overlap_mask, "drop_sheet_overlap",
               "Line in the later sheet identical to a line in the earlier sheet", steps, sheet_order)
    df = _drop(df, within_sheet_duplicates_mask(df), "drop_within_sheet_duplicates",
               "Exact duplicate line within one sheet (first copy kept)", steps, sheet_order)
    df = _drop(df, non_positive_price_mask(df), "drop_non_positive_price",
               "Price is zero or negative", steps, sheet_order)
    df = _drop(df, non_product_mask(df, codes), "drop_non_product_codes",
               "Stock code is a non-product code listed in config", steps, sheet_order)
    df = _drop(df, missing_customer_mask(df), "drop_missing_customer_id",
               "No Customer ID", steps, sheet_order)

    df = df.reset_index(drop=True)
    df["line_value"] = line_value(df)
    df["is_return"] = is_return(df, prefix)

    returns = df[df["is_return"]]
    is_cancel = df["invoice"].str.startswith(prefix)
    purchases = df[~df["is_return"]]
    summary = {
        "overlap": overlap,
        "raw": _totals(raw),
        "returns": {
            "rows": len(returns),
            "net_value": _money(returns["line_value"].sum()),
            "cancellation_rows_with_positive_quantity": int((is_cancel & (df["quantity"] > 0)).sum()),
            "negative_quantity_rows_not_cancellation": int((~is_cancel & (df["quantity"] < 0)).sum()),
        },
        "final": {
            **_totals(df),
            "purchase_rows": len(purchases),
            "sales_value": _money(purchases["line_value"].sum()),
            "customers": int(df["customer_id"].nunique()),
            "purchase_invoices": int(purchases["invoice"].nunique()),
            "first_date": _ts(df["invoice_date"].min()),
            "last_date": _ts(df["invoice_date"].max()),
        },
    }
    return df, steps, summary
