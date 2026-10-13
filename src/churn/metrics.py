"""Evaluation metrics for churn scores (PROJECT_PLAN.md Section 5).

y is 1 for churned, 0 for retained; a higher score means more likely to churn.
  ROC-AUC     chance a random churner scores above a random non-churner (ties count half)
  PR-AUC      average precision (scikit-learn): precision averaged over the recall steps
  top k%      the ceil(k% x customers) highest scores; customers tied at the boundary score share
              the remaining places equally (the average over random tie-breaks)
  Brier       mean squared difference between the probability and the 0/1 outcome
  calibration equal-width probability bins: mean predicted vs observed churn rate
"""

import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _arrays(y, score):
    y, score = np.asarray(y, dtype=int), np.asarray(score, dtype=float)
    if len(y) != len(score):
        raise ValueError(f"{len(y)} labels but {len(score)} scores")
    if len(np.unique(y)) != 2:
        raise ValueError("Need both churners and non-churners to compute the metrics")
    return y, score


def top_k_count(n, fraction):
    """ceil(fraction x n), robust to floating point (0.2 x 10 is 2, not 3)."""
    return max(1, math.ceil(round(fraction * n, 9)))


def _churners_in_top(y, score, k):
    """Expected churners among the k highest scores, sharing boundary ties; also the tie size."""
    threshold = np.sort(score)[::-1][k - 1]
    above, tied = score > threshold, score == threshold
    places_left = k - above.sum()
    return y[above].sum() + places_left * y[tied].mean(), int(tied.sum())


def top_fraction_stats(y, score, fraction):
    y, score = np.asarray(y, dtype=int), np.asarray(score, dtype=float)
    k = top_k_count(len(y), fraction)
    tp, ties = _churners_in_top(y, score, k)
    tp = float(tp)
    return {
        "fraction": fraction,
        "k": k,
        "churners_in_top": int(tp) if tp.is_integer() else tp,
        "precision": tp / k,
        "recall": tp / y.sum(),
        "boundary_ties": ties,
    }


def brier_score(y, prob):
    y, prob = np.asarray(y, dtype=float), np.asarray(prob, dtype=float)
    return float(np.mean((prob - y) ** 2))


def calibration_table(y, prob, n_bins):
    """One row per equal-width bin [i/n, (i+1)/n); the last bin includes 1.0. Empty bins are kept."""
    y, prob = np.asarray(y, dtype=int), np.asarray(prob, dtype=float)
    if prob.min() < 0 or prob.max() > 1:
        raise ValueError("Calibration needs probabilities between 0 and 1")
    bins = np.minimum(np.floor(prob * n_bins).astype(int), n_bins - 1)
    rows = []
    for i in range(n_bins):
        inside = bins == i
        count = int(inside.sum())
        rows.append({
            "bin": i + 1,
            "lower": round(i / n_bins, 10),
            "upper": round((i + 1) / n_bins, 10),
            "customers": count,
            "mean_predicted": float(prob[inside].mean()) if count else None,
            "observed_churn_rate": float(y[inside].mean()) if count else None,
        })
    return rows


def evaluate(y, score, top_fractions, n_bins, is_probability):
    """All metrics for one set of scores. Brier and calibration only for probabilities."""
    y, score = _arrays(y, score)
    return {
        "customers": len(y),
        "churners": int(y.sum()),
        "base_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "top": [top_fraction_stats(y, score, f) for f in top_fractions],
        "mean_predicted": float(score.mean()) if is_probability else None,
        "brier": brier_score(y, score) if is_probability else None,
        "calibration": calibration_table(y, score, n_bins) if is_probability else None,
    }


def top_key(fraction):
    return f"precision_top_{round(fraction * 100)}pct"


def bootstrap_intervals(y, score, top_fractions, n_samples, seed, level=0.95):
    """Percentile intervals from resampling customers with replacement. Resamples with only one
    class (where AUC is undefined) are skipped and the number used is reported."""
    y, score = _arrays(y, score)
    rng = np.random.default_rng(seed)
    n = len(y)
    names = ["roc_auc", "pr_auc"] + [top_key(f) for f in top_fractions]
    values = {name: [] for name in names}
    for _ in range(n_samples):
        idx = rng.integers(0, n, n)
        yb, sb = y[idx], score[idx]
        if yb.min() == yb.max():
            continue
        values["roc_auc"].append(roc_auc_score(yb, sb))
        values["pr_auc"].append(average_precision_score(yb, sb))
        for f in top_fractions:
            k = top_k_count(n, f)
            values[top_key(f)].append(_churners_in_top(yb, sb, k)[0] / k)
    tail = (1 - level) / 2 * 100
    intervals = {name: {"low": float(np.percentile(v, tail)), "high": float(np.percentile(v, 100 - tail))}
                 for name, v in values.items()}
    return {"level": level, "samples": n_samples, "samples_used": len(values["roc_auc"]),
            "seed": seed, "intervals": intervals}
