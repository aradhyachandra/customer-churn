"""What drives churn risk (PROJECT_PLAN.md Section 5, explainability).

  permutation importance  drop in a score when one feature is shuffled (the model is not changed)
  partial dependence      average predicted probability with one feature set to each grid value
  group statements        churn rate inside vs outside a plain-language group, on training rows
"""

import operator

import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence, permutation_importance

OPERATORS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le,
             "==": operator.eq, "!=": operator.ne}


def permutation_importance_table(model, X, y, scoring, n_repeats, seed, spread=(2.5, 97.5)):
    """One row per feature, most important first: baseline score, mean drop, sd and percentile range."""
    result = permutation_importance(model, X, y, scoring=scoring, n_repeats=n_repeats,
                                    random_state=seed, n_jobs=1)
    drops = result.importances
    table = pd.DataFrame({
        "feature": list(X.columns),
        "mean": drops.mean(axis=1),
        "std": drops.std(axis=1),
        "low": np.percentile(drops, spread[0], axis=1),
        "high": np.percentile(drops, spread[1], axis=1),
        "repeats": n_repeats,
    })
    table = table.sort_values("mean", ascending=False, kind="stable").reset_index(drop=True)
    table.insert(0, "rank", range(1, len(table) + 1))
    return table


def partial_dependence_table(model, X, features, grid_resolution, percentiles):
    """Long table (feature, value, mean_predicted), brute force on predict_proba (probability scale)."""
    rows = []
    for f in features:
        pdp = partial_dependence(model, X, [f], grid_resolution=grid_resolution,
                                 percentiles=tuple(percentiles), method="brute",
                                 response_method="predict_proba", kind="average")
        for value, mean in zip(pdp["grid_values"][0], pdp["average"][0]):
            rows.append({"feature": f, "value": float(value), "mean_predicted": float(mean)})
    return pd.DataFrame(rows)


def _mask(df, condition):
    if condition["op"] not in OPERATORS:
        raise ValueError(f"Unknown operator {condition['op']!r}")
    return OPERATORS[condition["op"]](df[condition["feature"]], condition["value"])


def group_churn(df, condition):
    """Churn inside vs outside the group `feature op value`, pooled and per cutoff (min, max).
    With condition['among'] (another feature-op-value), only rows meeting it are compared."""
    if "among" in condition:
        df = df[_mask(df, condition["among"])]
    inside = _mask(df, condition)
    y = df["churned"]
    per = pd.DataFrame({"cutoff": df["cutoff"], "inside": inside, "churned": y})
    by = per.groupby(["cutoff", "inside"])["churned"].mean().unstack()
    churn_in, churn_out = float(y[inside].mean()), float(y[~inside].mean())
    return {
        "rows_in": int(inside.sum()), "churned_in": int(y[inside].sum()), "churn_in": churn_in,
        "rows_out": int((~inside).sum()), "churned_out": int(y[~inside].sum()), "churn_out": churn_out,
        "share_in": float(inside.mean()),
        "ratio": churn_in / churn_out if churn_out else None,
        "churn_in_min": float(by[True].min()), "churn_in_max": float(by[True].max()),
        "churn_out_min": float(by[False].min()), "churn_out_max": float(by[False].max()),
    }
