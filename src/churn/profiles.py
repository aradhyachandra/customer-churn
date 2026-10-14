"""Describe each customer by the features where they differ most from the median customer.

Used for the dashboard's risk list. This describes the customer, not why the model gave its score.
Unusualness = the larger of the share of reference customers strictly below and strictly above
the customer's value; ties at the median therefore score low.
"""

import numpy as np


def unusual_features(df, reference, specs, n):
    """For each row of df: up to n dicts {feature, value, median, direction, share}, most unusual
    first (ties keep the order of specs). Features with repeat_buyers_only are skipped for
    one-day buyers and compared with the reference's repeat buyers only."""
    repeat = reference["single_purchase_day_365d"].to_numpy() == 0
    refs = {}
    for f, spec in specs.items():
        values = reference[f].to_numpy(dtype=float)
        values = np.sort(values[repeat] if spec.get("repeat_buyers_only") else values)
        refs[f] = (values, float(np.median(values)))

    one_day = df["single_purchase_day_365d"].to_numpy() == 1
    columns = {f: df[f].to_numpy(dtype=float) for f in specs}
    out = []
    for i in range(len(df)):
        items = []
        for f, spec in specs.items():
            if spec.get("repeat_buyers_only") and one_day[i]:
                continue
            values, median = refs[f]
            v = float(columns[f][i])
            below = np.searchsorted(values, v, side="left") / len(values)
            above = (len(values) - np.searchsorted(values, v, side="right")) / len(values)
            items.append({"feature": f, "value": v, "median": median,
                          "direction": "higher" if above <= below else "lower",
                          "share": float(max(below, above))})
        items.sort(key=lambda d: -d["share"])       # stable: ties keep the config order
        out.append(items[:n])
    return out


def _value(x, fmt):
    if fmt == "gbp":
        return f"{'-' if x < 0 else ''}£{abs(x):,.0f}"
    if fmt == "pct":
        return f"{x:.0%}"
    return f"{x:,.0f}"


def describe(item, spec):
    """One plain-language phrase, e.g. 'orders in the past year 1 (median 2; lower than 60% of customers)'."""
    fmt = spec["format"]
    return (f"{spec['label']} {_value(item['value'], fmt)} (median {_value(item['median'], fmt)}; "
            f"{item['direction']} than {item['share']:.0%} of customers)")
