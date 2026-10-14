"""Data loading and campaign maths for the dashboard (app/streamlit_app.py).

The dashboard reads committed files only (results/ and config/): no raw data, snapshot or model.
All money figures are ESTIMATES under the chosen assumptions, not observed effects.
"""

import json
from pathlib import Path

import pandas as pd
import yaml

from churn.business import best_campaign, campaign_curve, chosen_customers, expected_benefit
from churn.profiles import describe, unusual_features

SCORE_COLUMNS = ["customer_id", "p_model", "p_rescaled", "value", "expected_benefit_central", "rank_central",
                 "in_central_list", "single_purchase_day_365d", "churned_after_the_fact"]


def campaign_estimate(p, value, margin, save_rate, offer_cost):
    """Best campaign under one set of assumptions: contact customers in order of expected benefit
    and stop where the total is highest (the same maths as results/c5_campaign_scenarios.csv)."""
    p, value = pd.Series(p).to_numpy(dtype=float), pd.Series(value).to_numpy(dtype=float)
    benefit = expected_benefit(p, value, save_rate, margin, offer_cost)
    best = best_campaign(benefit)
    chosen = chosen_customers(benefit, best["size"])
    return {"best_size": best["size"], "total_benefit": best["total_benefit"],
            "offer_cost_total": best["size"] * offer_cost,
            "expected_saved_revenue": float((p[chosen] * save_rate * value[chosen]).sum()),
            "value_threshold": offer_cost / (save_rate * margin),
            "chosen": chosen, "curve": campaign_curve(benefit)}


def unusual_phrases(scores, specs, n):
    """Each customer's n most unusual features vs the median customer of the same file (the active
    customers at the test cutoff), as one plain-language string per customer."""
    items = unusual_features(scores, scores, specs, n)
    return ["; ".join(describe(d, specs[d["feature"]]) for d in row) for row in items]


def load_dashboard_data(root):
    root = Path(root)
    paths = yaml.safe_load((root / "config" / "data.yaml").read_text())["paths"]

    def csv(key, **kw):
        return pd.read_csv(root / paths[key], **kw)

    return {
        "business_config": yaml.safe_load((root / "config" / "business.yaml").read_text()),
        "report_config": yaml.safe_load((root / "config" / "report.yaml").read_text()),
        "business": json.loads((root / paths["c5_business"]).read_text()),
        "c6": json.loads((root / paths["c6_report"]).read_text()),
        "scores": csv("c6_test_scores"),
        "retention_rates": csv("c2_retention_rates", index_col="cohort"),
        "cohort_summary": csv("c2_cohort_summary"),
        "churn_by_cutoff": csv("c2_churn_by_cutoff", dtype={"cutoff": str}),
        "segments": csv("c6_segments_training"),
        "statements": csv("c5_driver_statements"),
        "retention_heatmap": root / paths["c2_heatmap"],
    }
