"""Small summaries for the C6 report and dashboard. Descriptive only: nothing is fitted or tuned."""

import numpy as np
import pandas as pd


def segment_churn(rows, segments):
    """Churn rate per segment group, with the lowest and highest rate over the cutoffs in rows.
    segments: {feature: {"title", "labels"}} for a fixed value -> label map, or
    {feature: {"title", "edges", "labels"}} for bands [edge_i, edge_i+1)."""
    out = []
    for feature, spec in segments.items():
        if "edges" in spec:
            groups = pd.cut(rows[feature], bins=spec["edges"], labels=spec["labels"], right=False)
            if groups.isna().any():
                raise ValueError(f"{int(groups.isna().sum())} rows of {feature} fall outside the bands")
            order = list(spec["labels"])
        else:
            groups = rows[feature].map(spec["labels"])
            if groups.isna().any():
                raise ValueError(f"{int(groups.isna().sum())} rows of {feature} fall outside the labels")
            order = list(spec["labels"].values())
        frame = pd.DataFrame({"group": groups.astype(str), "cutoff": rows["cutoff"], "churned": rows["churned"]})
        for g in order:
            d = frame[frame["group"] == g]
            per_cutoff = d.groupby("cutoff")["churned"].mean()
            out.append({"segment": feature, "title": spec["title"], "group": g, "rows": len(d),
                        "churned": int(d["churned"].sum()),
                        "churn_rate": float(d["churned"].mean()) if len(d) else None,
                        "churn_rate_min": float(per_cutoff.min()) if len(d) else None,
                        "churn_rate_max": float(per_cutoff.max()) if len(d) else None})
    return pd.DataFrame(out)


def predicted_vs_actual(p_model, p_rescaled, y, chosen):
    """Mean predicted churn (model and rescaled) vs observed churn for everyone, the chosen
    campaign list and the rest. Observed churn is after-the-fact information."""
    p_model, p_rescaled, y = (np.asarray(a, dtype=float) for a in (p_model, p_rescaled, y))
    chosen = np.asarray(chosen, dtype=bool)
    rows = []
    for group, mask in (("everyone", np.ones(len(y), dtype=bool)), ("chosen", chosen), ("not_chosen", ~chosen)):
        observed = float(y[mask].mean())
        rows.append({"group": group, "customers": int(mask.sum()), "churned": int(y[mask].sum()),
                     "mean_p_model": float(p_model[mask].mean()), "mean_p_rescaled": float(p_rescaled[mask].mean()),
                     "observed_churn_rate": observed,
                     "gap_points_model": (float(p_model[mask].mean()) - observed) * 100,
                     "gap_points_rescaled": (float(p_rescaled[mask].mean()) - observed) * 100,
                     "ratio_rescaled": float(p_rescaled[mask].mean()) / observed if observed else None})
    return pd.DataFrame(rows)


def near_best_sizes(curve, share):
    """The best campaign size and the smallest and largest sizes whose total is at least share x best."""
    curve = np.asarray(curve, dtype=float)
    best_size = int(np.argmax(curve))
    best = float(curve[best_size])
    close = np.flatnonzero(curve >= share * best)
    return {"best_size": best_size, "best": best, "share": share,
            "min_size": int(close.min()), "max_size": int(close.max())}


def value_concentration(value, shares):
    """Share of total value held by the top fraction of customers (k = ceil(fraction x customers))."""
    v = np.sort(np.asarray(value, dtype=float))[::-1]
    out = []
    for s in shares:
        k = int(np.ceil(s * len(v)))
        out.append({"top_share_of_customers": s, "customers": k, "value_share": float(v[:k].sum() / v.sum())})
    return out
