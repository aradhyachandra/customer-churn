"""Business layer (PROJECT_PLAN.md Section 6): revenue at risk and campaign targeting.

All money figures are ESTIMATES under the assumptions in config/business.yaml, not observed effects.

  customer value     v = trailing 12-month net revenue, floored at 0
  revenue at risk    sum of p(churn) x v
  expected benefit   p x save rate x margin x v - offer cost, per contacted customer
  campaign           contact customers in order of expected benefit; the best size maximises the
                     total, i.e. contacts everyone whose expected benefit is positive
"""

import itertools

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit


def customer_value(net_revenue_365d):
    """Trailing 12-month net revenue with negative values counted as 0."""
    return np.maximum(np.asarray(net_revenue_365d, dtype=float), 0.0)


def revenue_at_risk(p, value):
    return float(np.sum(np.asarray(p, dtype=float) * np.asarray(value, dtype=float)))


def expected_benefit(p, value, save_rate, margin, offer_cost):
    """Expected net benefit (GBP) of contacting each customer."""
    return np.asarray(p, dtype=float) * save_rate * margin * np.asarray(value, dtype=float) - offer_cost


def _order(benefit):
    """Indices from highest to lowest benefit; ties keep the original (customer) order."""
    return np.argsort(-np.asarray(benefit, dtype=float), kind="stable")


def campaign_curve(benefit):
    """Total expected benefit when contacting the top 0, 1, ..., n customers by benefit."""
    ordered = np.asarray(benefit, dtype=float)[_order(benefit)]
    return np.concatenate([[0.0], np.cumsum(ordered)])


def best_campaign(benefit):
    """The campaign size with the highest total (the smallest such size on a tie) and its total."""
    curve = campaign_curve(benefit)
    size = int(np.argmax(curve))
    return {"size": size, "total_benefit": float(curve[size])}


def chosen_customers(benefit, size):
    """Boolean mask of the `size` customers with the highest expected benefit."""
    mask = np.zeros(len(benefit), dtype=bool)
    mask[_order(benefit)[:size]] = True
    return mask


def rescale_to_mean(p, target):
    """Add one constant to every log-odds so the mean probability equals target. Ranking is kept."""
    if not 0 < target < 1:
        raise ValueError(f"target mean must be strictly between 0 and 1, got {target}")
    z = logit(np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12))
    shift = brentq(lambda b: expit(z + b).mean() - target, -50, 50, xtol=1e-14)
    return {"probabilities": expit(z + shift), "shift": float(shift)}


def reference_churn_rate(churn_by_cutoff, reference_cutoff, decision_day, horizon_days):
    """The churn rate at reference_cutoff, checked to be known on the decision day: its outcome window
    [c, c + horizon) must end on or before the decision day, and no later cutoff may also qualify."""
    table = churn_by_cutoff.assign(cutoff=pd.to_datetime(churn_by_cutoff["cutoff"]))
    ref, day, h = pd.Timestamp(reference_cutoff), pd.Timestamp(decision_day), pd.Timedelta(days=horizon_days)
    if ref + h > day:
        raise ValueError(f"{ref:%Y-%m-%d}: its window ends {ref + h:%Y-%m-%d}, so its churn rate was "
                         f"not known on {day:%Y-%m-%d}")
    known = table[table["cutoff"] + h <= day]
    latest = known["cutoff"].max()
    if ref != latest:
        raise ValueError(f"{ref:%Y-%m-%d} is not the most recent known cutoff ({latest:%Y-%m-%d})")
    row = table[table["cutoff"] == ref].iloc[0]
    return {"cutoff": f"{ref:%Y-%m-%d}", "window_end": f"{ref + h:%Y-%m-%d}",
            "active": int(row["active"]), "churned": int(row["churned"]),
            "churn_rate": float(row["churned"] / row["active"]), "known_on_decision_day": True}


def scenario_grid(cfg):
    """Every margin x save rate x offer cost combination, flagging the central case."""
    m, s, c = cfg["margin"], cfg["save_rate"], cfg["offer_cost_gbp"]
    return [{"margin": a, "save_rate": b, "offer_cost": d,
             "is_central": (a, b, d) == (m["central"], s["central"], c["central"])}
            for a, b, d in itertools.product(m["scenarios"], s["scenarios"], c["scenarios"])]
