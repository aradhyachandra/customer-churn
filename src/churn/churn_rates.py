"""Descriptive churn at a cutoff (PROJECT_PLAN.md Section 3).

A cutoff is 00:00 on its date (config/churn.yaml, cutoff_convention).
  active   = at least one purchase event in [cutoff - lookback_days, cutoff)
  churned  = active, and no purchase event in [cutoff, cutoff + horizon_days)
Description only: no features, no model.
"""

import pandas as pd


def valid_monthly_cutoffs(first_ts, last_ts, lookback_days, horizon_days):
    """First-of-month cutoffs whose full lookback and outcome windows fit inside the data.
    The lookback may start on the first data day; the outcome window must end by the start of the
    last data day, because that day is only partly covered."""
    data_start = pd.Timestamp(first_ts).normalize()
    data_end = pd.Timestamp(last_ts).normalize()
    lookback, horizon = pd.Timedelta(days=lookback_days), pd.Timedelta(days=horizon_days)
    candidates = pd.date_range(data_start, data_end, freq="MS")
    return [c for c in candidates if c - lookback >= data_start and c + horizon <= data_end]


def churn_at_cutoff(events, cutoff, lookback_days, horizon_days):
    cutoff = pd.Timestamp(cutoff)
    dates = events["invoice_date"]
    before = events[(dates >= cutoff - pd.Timedelta(days=lookback_days)) & (dates < cutoff)]
    after = events[(dates >= cutoff) & (dates < cutoff + pd.Timedelta(days=horizon_days))]
    active = set(before["customer_id"])
    churned = active - set(after["customer_id"])
    return {
        "cutoff": cutoff.strftime("%Y-%m-%d"),
        "active": len(active),
        "churned": len(churned),
        "retained": len(active) - len(churned),
        "churn_rate": len(churned) / len(active) if active else None,
    }


def churn_table(events, cutoffs, lookback_days, horizon_days):
    rows = [churn_at_cutoff(events, c, lookback_days, horizon_days) for c in cutoffs]
    return pd.DataFrame(rows, columns=["cutoff", "active", "churned", "retained", "churn_rate"])
