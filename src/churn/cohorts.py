"""Customer-month table, first-purchase cohorts and cohort retention.

Cohort = calendar month of a customer's first purchase event. Retention in month k = share of the
cohort with at least one purchase event k months after the cohort month (k = 0 is always 100%).
Months are stored as the Timestamp of their first day.
"""

import pandas as pd


def _month(dates):
    return dates.dt.to_period("M").dt.to_timestamp()


def customer_months(df, events):
    """One row per customer and month with any line: purchase invoices and value (purchase events
    only), return value and net value (all lines, so cancelled purchases still net to zero)."""
    lines = df.assign(month=_month(df["invoice_date"]))
    lines["return_value"] = lines["line_value"].where(lines["is_return"], 0.0)
    money = lines.groupby(["customer_id", "month"]).agg(
        return_value=("return_value", "sum"), net_value=("line_value", "sum"))

    buys = events.assign(month=_month(events["invoice_date"]))
    bought = buys.groupby(["customer_id", "month"]).agg(
        purchase_invoices=("invoice", "nunique"), purchase_value=("value", "sum"))

    out = money.join(bought, how="outer")
    out["purchase_invoices"] = out["purchase_invoices"].fillna(0).astype("int64")
    out["purchase_value"] = out["purchase_value"].fillna(0.0)
    out[["return_value", "net_value"]] = out[["return_value", "net_value"]].fillna(0.0)
    out = out.reset_index()[["customer_id", "month", "purchase_invoices", "purchase_value",
                             "return_value", "net_value"]]
    return out.sort_values(["customer_id", "month"]).reset_index(drop=True)


def return_only_customers(df, events):
    """Customers in the clean data with no purchase event (sorted list of ids)."""
    ids = set(df["customer_id"].dropna()) - set(events["customer_id"])
    return sorted(int(i) for i in ids)


def first_purchase_months(events):
    """Cohort month per customer (Series indexed by customer_id)."""
    return _month(events.groupby("customer_id")["invoice_date"].min()).rename("cohort")


def retention_counts(events, last_month):
    """Rows = cohort month, columns = months since first purchase (0, 1, ...), values = number of
    cohort customers with a purchase event in that month. Months after last_month are missing (NaN)."""
    cohort = first_purchase_months(events)
    buys = events.assign(month=_month(events["invoice_date"]))
    buys = buys.join(cohort, on="customer_id")
    buys["k"] = ((buys["month"].dt.year - buys["cohort"].dt.year) * 12
                 + buys["month"].dt.month - buys["cohort"].dt.month)
    counts = buys.groupby(["cohort", "k"])["customer_id"].nunique().unstack("k")

    cohorts = pd.date_range(cohort.min(), last_month, freq="MS")
    counts = counts.reindex(index=cohorts, columns=range(len(cohorts))).fillna(0)
    months_in_data = (last_month.year - cohorts.year) * 12 + last_month.month - cohorts.month
    past_data = counts.columns.to_numpy()[None, :] > months_in_data.to_numpy()[:, None]
    counts = counts.mask(past_data)
    counts = counts[counts[0] > 0]
    counts.index.name = "cohort"
    counts.columns.name = "months_since_first"
    return counts


def retention_rates(counts):
    """Counts divided by cohort size (column 0)."""
    return counts.div(counts[0], axis=0)
