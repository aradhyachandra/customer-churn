"""Snapshot tables: one row per active customer at a cutoff, with features and the churn label.

Cutoff convention (config/churn.yaml): a cutoff is 00:00 on its date.
  features  use only lines dated strictly before the cutoff
  label     churned = 1 when the customer has no purchase event in [cutoff, cutoff + horizon)
Active = at least one purchase event in [cutoff - lookback, cutoff), as in C2.

Purchase events follow config/churn.yaml (purchase_event_rule). With the same-day rule, a
purchase and the cancellation that reverses it fall on the same day, so both are on the same side
of any cutoff. Those cancellations are not counted as returns either. Feature settings and their
reasons are in config/features.yaml; the definitions below are what the docs and report print.
"""

import pandas as pd

from churn.events import full_cancellation_pairs, purchase_events

FEATURES = {
    "recency_days": "Whole days from the last purchase day to the cutoff date (1 = bought the day before).",
    "orders_90d": "Purchase events in the 90 days before the cutoff.",
    "orders_180d": "Purchase events in the 180 days before the cutoff.",
    "orders_365d": "Purchase events in the 365 days before the cutoff.",
    "net_revenue_90d": "Net value (£) of all lines, purchases minus returns, in the 90 days before the cutoff.",
    "net_revenue_180d": "Net value (£) of all lines in the 180 days before the cutoff.",
    "net_revenue_365d": "Net value (£) of all lines in the 365 days before the cutoff (trailing 12-month revenue).",
    "avg_order_value_365d": "Value (£) of purchase events in the last 365 days divided by their number.",
    "orders_trend_90d": "Purchase events in the last 90 days minus those in the 90 days before them.",
    "net_revenue_trend_90d": "Net value (£) in the last 90 days minus net value in the 90 days before them.",
    "single_purchase_day_365d": "1 if the customer bought on only one day in the last 365 days (gap features then use fill values).",
    "avg_gap_days": "Mean days between consecutive purchase days in the last 365 days (365 if only one purchase day).",
    "latest_gap_days": "Days between the last two purchase days in the last 365 days (365 if only one purchase day).",
    "latest_gap_ratio": "latest_gap_days / avg_gap_days (above 1: the last gap was longer than usual).",
    "recency_gap_ratio": "recency_days / avg_gap_days (above 1: already waiting longer than the usual gap).",
    "tenure_days": "Whole days from the first purchase day in the data to the cutoff date.",
    "first_seen_first_data_month": "1 if the first purchase is in the first data month (Dec 2009): tenure is a lower bound.",
    "distinct_products_365d": "Distinct stock codes (upper-cased, stripped) in purchase events in the last 365 days.",
    "return_rate_365d": "Value returned / value bought in the last 365 days; same-day full cancellations count on neither side. Can exceed 1 when earlier purchases are returned.",
    "is_uk": "1 if the country on the customer's latest line before the cutoff is the United Kingdom.",
    "bought_in_peak_365d": "1 if the customer bought in the November or December inside the last 365 days.",
}

CODE_NORMALISATIONS = ("upper_strip", "none")


def normalise_stock_codes(codes, method):
    if method not in CODE_NORMALISATIONS:
        raise ValueError(f"Unknown stock_code_normalisation {method!r}; expected one of {CODE_NORMALISATIONS}")
    return codes.str.strip().str.upper() if method == "upper_strip" else codes


def stock_code_merges(codes, method):
    """How many distinct codes disappear by merging into another code."""
    codes = pd.Series(codes).drop_duplicates()
    return len(codes) - normalise_stock_codes(codes, method).nunique()


def training_cutoffs(churn_cfg):
    t = churn_cfg["training_cutoffs"]
    if t["frequency"] != "monthly":
        raise ValueError("training_cutoffs.frequency must be monthly")
    return list(pd.date_range(t["first"], t["last"], freq="MS"))


def peak_months_inside_lookback(cutoff, lookback_days, peak_months):
    """First days of the peak months that lie completely inside [cutoff - lookback, cutoff)."""
    start = cutoff - pd.Timedelta(days=lookback_days)
    months = pd.date_range(start.to_period("M").to_timestamp(), cutoff, freq="MS")
    return [m for m in months
            if m >= start and m + pd.offsets.MonthBegin(1) <= cutoff and m.month in peak_months]


def history_inputs(lines, rule):
    """Purchase events and the cancellation invoices that reverse a purchase in full on the same day."""
    events = purchase_events(lines, rule)
    if rule == "drop_same_day_full_cancellations":
        cancels = set(full_cancellation_pairs(lines, same_day_only=True)["cancel_invoice"])
    else:
        cancels = set()
    return events, cancels


def _before(df, cutoff, days=None, gap_days=0):
    """Rows in [cutoff - days, cutoff - gap_days), or all rows before the cutoff when days is None."""
    d = df["invoice_date"]
    mask = d < cutoff - pd.Timedelta(days=gap_days)
    if days is not None:
        mask &= d >= cutoff - pd.Timedelta(days=days)
    return df[mask]


def snapshot_features(lines, events, same_day_cancels, cutoff, churn_cfg, feat_cfg):
    """Feature table indexed by customer_id (active customers at the cutoff), columns = FEATURES."""
    cutoff = pd.Timestamp(cutoff)
    lookback = churn_cfg["active_lookback_days"]
    windows, trend = feat_cfg["frequency_windows_days"], feat_cfg["trend_window_days"]
    if max(windows) > lookback or 2 * trend > lookback:
        raise ValueError("Feature windows must fit inside the active lookback")

    hist = _before(lines, cutoff)
    hist_events = _before(events, cutoff)
    year_events = _before(hist_events, cutoff, lookback)
    year_lines = _before(hist, cutoff, lookback)
    customers = pd.Index(sorted(year_events["customer_id"].unique()), name="customer_id", dtype="int64")
    out = pd.DataFrame(index=customers)

    def per_customer(series, fill):
        return series.reindex(customers, fill_value=fill)

    def orders(df):
        return per_customer(df.groupby("customer_id").size(), 0).astype("int64")

    def net(df):
        return per_customer(df.groupby("customer_id")["line_value"].sum(), 0.0)

    cutoff_day = cutoff.normalize()
    days = year_events.assign(day=year_events["invoice_date"].dt.normalize())[["customer_id", "day"]]
    days = days.drop_duplicates().sort_values(["customer_id", "day"])
    last_day = days.groupby("customer_id")["day"].max()
    out["recency_days"] = per_customer((cutoff_day - last_day).dt.days, 0).astype("int64")

    for w in windows:
        out[f"orders_{w}d"] = orders(_before(hist_events, cutoff, w))
    for w in windows:
        out[f"net_revenue_{w}d"] = net(_before(hist, cutoff, w))
    year_value = per_customer(year_events.groupby("customer_id")["value"].sum(), 0.0)
    out[f"avg_order_value_{lookback}d"] = year_value / out[f"orders_{lookback}d"]

    prev_orders = orders(_before(hist_events, cutoff, 2 * trend, gap_days=trend))
    prev_net = net(_before(hist, cutoff, 2 * trend, gap_days=trend))
    out[f"orders_trend_{trend}d"] = orders(_before(hist_events, cutoff, trend)) - prev_orders
    out[f"net_revenue_trend_{trend}d"] = net(_before(hist, cutoff, trend)) - prev_net

    fill = feat_cfg["single_purchase_gap_fill_days"]
    gaps = days.assign(gap=days.groupby("customer_id")["day"].diff().dt.days).dropna(subset=["gap"])
    has_gap = per_customer(gaps.groupby("customer_id").size(), 0) > 0
    out[f"single_purchase_day_{lookback}d"] = (~has_gap).astype("int64")
    out["avg_gap_days"] = per_customer(gaps.groupby("customer_id")["gap"].mean(), float(fill))
    out["latest_gap_days"] = per_customer(gaps.groupby("customer_id")["gap"].last(), fill).astype("int64")
    out["latest_gap_ratio"] = out["latest_gap_days"] / out["avg_gap_days"]
    out["recency_gap_ratio"] = out["recency_days"] / out["avg_gap_days"]

    first = hist_events.groupby("customer_id")["invoice_date"].min()
    out["tenure_days"] = per_customer((cutoff_day - first.dt.normalize()).dt.days, 0).astype("int64")
    censored = pd.Period(feat_cfg["tenure_censored_month"], "M")
    out["first_seen_first_data_month"] = per_customer(first.dt.to_period("M") == censored, False).astype("int64")

    bought = year_lines[year_lines["invoice"].isin(year_events["invoice"])]
    codes = normalise_stock_codes(bought["stock_code"], feat_cfg["stock_code_normalisation"])
    out[f"distinct_products_{lookback}d"] = per_customer(
        codes.groupby(bought["customer_id"]).nunique(), 0).astype("int64")

    returns = year_lines[year_lines["is_return"] & ~year_lines["invoice"].isin(same_day_cancels)]
    returned = per_customer(-returns.groupby("customer_id")["line_value"].sum(), 0.0)
    out[f"return_rate_{lookback}d"] = returned / year_value

    latest = hist.sort_values(["invoice_date", "invoice"]).groupby("customer_id")["country"].last()
    out["is_uk"] = per_customer(latest == feat_cfg["home_country"], False).astype("int64")

    peak = year_events["invoice_date"].dt.month.isin(feat_cfg["peak_months"])
    peak_customers = year_events.loc[peak, "customer_id"].unique()
    out[f"bought_in_peak_{lookback}d"] = customers.isin(peak_customers).astype("int64")

    return out[list(FEATURES)]


def churn_labels(events, customers, cutoff, horizon_days):
    """1 if the customer has no purchase event in [cutoff, cutoff + horizon), else 0."""
    d = events["invoice_date"]
    future = events[(d >= cutoff) & (d < cutoff + pd.Timedelta(days=horizon_days))]
    return pd.Series((~customers.isin(future["customer_id"])).astype("int64"), index=customers, name="churned")


def build_snapshot(lines, events, same_day_cancels, cutoff, churn_cfg, feat_cfg):
    """Columns: cutoff, customer_id, every feature in FEATURES, churned."""
    cutoff = pd.Timestamp(cutoff)
    feats = snapshot_features(lines, events, same_day_cancels, cutoff, churn_cfg, feat_cfg)
    feats["churned"] = churn_labels(events, feats.index, cutoff, churn_cfg["churn_horizon_days"])
    feats.insert(0, "cutoff", cutoff)
    return feats.reset_index()[["cutoff", "customer_id", *FEATURES, "churned"]]


def leakage_differences(lines, cutoff, churn_cfg, feat_cfg, truncate_at=None):
    """Build features from all lines and from lines dated before truncate_at (default: the cutoff),
    rebuilding purchase events from the truncated lines too. Returns the features that differ
    ("rows" if the set of customers differs); an empty list means no feature saw the future."""
    cutoff = pd.Timestamp(cutoff)
    truncate_at = cutoff if truncate_at is None else pd.Timestamp(truncate_at)
    rule = churn_cfg["purchase_event_rule"]
    full = snapshot_features(lines, *history_inputs(lines, rule), cutoff, churn_cfg, feat_cfg)
    cut_lines = lines[lines["invoice_date"] < truncate_at]
    cut = snapshot_features(cut_lines, *history_inputs(cut_lines, rule), cutoff, churn_cfg, feat_cfg)
    diffs = [] if full.index.equals(cut.index) else ["rows"]
    rows = full.index.union(cut.index)
    full, cut = full.reindex(rows), cut.reindex(rows)
    return diffs + [c for c in FEATURES if not full[c].equals(cut[c])]
