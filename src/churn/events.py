"""Purchase events: which purchase invoices count as a customer buying something.

The rule is set in config/churn.yaml (purchase_event_rule):
  all_purchase_invoices              every purchase invoice is an event
  drop_same_day_full_cancellations   a purchase invoice is not an event when a cancellation
                                     invoice from the same customer on the same calendar day
                                     reverses exactly its lines (stock code, quantity, price)

Matching is one to one: each cancellation reverses at most one purchase, and each purchase is
reversed at most once. When several identical purchases come before a cancellation, the latest
one is taken as reversed.
"""

import pandas as pd

RULES = ("all_purchase_invoices", "drop_same_day_full_cancellations")


def invoice_summary(df):
    """One row per invoice: customer, time, return flag, value and a signature of its lines."""
    line_key = (df["stock_code"] + "|" + df["quantity"].abs().astype(str) + "|"
                + df["price"].round(4).astype(str))
    grouped = df.assign(line_key=line_key).groupby("invoice", sort=False)
    out = grouped.agg(
        customer_id=("customer_id", "first"),
        invoice_date=("invoice_date", "min"),
        is_return=("is_return", "all"),
        value=("line_value", "sum"),
        signature=("line_key", lambda s: "#".join(sorted(s))),
    ).reset_index()
    out["day"] = out["invoice_date"].dt.normalize()
    return out


def full_cancellation_pairs(df, same_day_only):
    """Pair purchase invoices with the cancellation invoices that reverse them in full."""
    inv = invoice_summary(df)
    purchases = inv[~inv["is_return"]]
    cancels = inv[inv["is_return"]]
    cand = purchases.merge(cancels, on=["customer_id", "signature"], suffixes=("_p", "_c"))
    cand = cand[cand["invoice_date_c"] >= cand["invoice_date_p"]]
    if same_day_only:
        cand = cand[cand["day_c"] == cand["day_p"]]

    # One to one: walk cancellations in time order; each takes the latest unused earlier purchase.
    cand = cand.sort_values(["invoice_date_c", "invoice_c", "invoice_date_p", "invoice_p"],
                            ascending=[True, True, False, False])
    used_p, used_c, rows = set(), set(), []
    for r in cand.itertuples(index=False):
        if r.invoice_p in used_p or r.invoice_c in used_c:
            continue
        used_p.add(r.invoice_p)
        used_c.add(r.invoice_c)
        rows.append({
            "customer_id": r.customer_id,
            "purchase_invoice": r.invoice_p,
            "cancel_invoice": r.invoice_c,
            "purchase_date": r.invoice_date_p,
            "cancel_date": r.invoice_date_c,
            "value": round(float(r.value_p), 2),
        })
    columns = ["customer_id", "purchase_invoice", "cancel_invoice", "purchase_date", "cancel_date", "value"]
    return pd.DataFrame(rows, columns=columns)


def purchase_events(df, rule):
    """Purchase invoices that count as purchase events: invoice, customer_id, invoice_date, value."""
    if rule not in RULES:
        raise ValueError(f"Unknown purchase_event_rule {rule!r}; expected one of {RULES}")
    inv = invoice_summary(df)
    events = inv[~inv["is_return"]]
    if rule == "drop_same_day_full_cancellations":
        dropped = full_cancellation_pairs(df, same_day_only=True)["purchase_invoice"]
        events = events[~events["invoice"].isin(dropped)]
    events = events[["invoice", "customer_id", "invoice_date", "value"]]
    return events.sort_values(["invoice_date", "invoice"]).reset_index(drop=True)


def _pair_stats(pairs):
    return {
        "purchase_invoices": len(pairs),
        "customers": int(pairs["customer_id"].nunique()),
        "value": round(float(pairs["value"].sum()), 2),
    }


def cancellation_impact(df, rule):
    """How many purchase invoices and customers full cancellations affect, and what the rule changes."""
    all_events = purchase_events(df, "all_purchase_invoices")
    chosen = purchase_events(df, rule)
    first_all = all_events.groupby("customer_id")["invoice_date"].min().dt.to_period("M")
    first_chosen = chosen.groupby("customer_id")["invoice_date"].min().dt.to_period("M")
    still_buying = first_all.index.intersection(first_chosen.index)
    same_day = full_cancellation_pairs(df, same_day_only=True)
    any_date = full_cancellation_pairs(df, same_day_only=False)
    return {
        "purchase_invoices": len(all_events),
        "same_day": _pair_stats(same_day),
        "any_later_date": _pair_stats(any_date),
        "largest_same_day": [
            {"customer_id": int(r.customer_id), "purchase_invoice": r.purchase_invoice,
             "cancel_invoice": r.cancel_invoice, "value": r.value}
            for r in same_day.nlargest(5, "value").itertuples(index=False)
        ],
        "chosen_rule": {
            "rule": rule,
            "purchase_invoices_dropped": len(all_events) - len(chosen),
            "customers_left_with_no_purchase": int(len(first_all) - len(first_chosen)),
            "customers_first_purchase_month_changed": int(
                (first_all[still_buying] != first_chosen[still_buying]).sum()),
        },
    }
