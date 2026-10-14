"""Business layer maths on hand-worked fixtures (PROJECT_PLAN.md Section 6)."""

import math

import numpy as np
import pandas as pd
import pytest

from churn.business import (
    best_campaign,
    campaign_curve,
    chosen_customers,
    customer_value,
    expected_benefit,
    reference_churn_rate,
    rescale_to_mean,
    revenue_at_risk,
    scenario_grid,
)
from churn.config import load_config

# Hand-worked case: p x v = [800, 200, 20]; with save 10% and margin 30%, x 0.03 = [24, 6, 0.6];
# minus an offer cost of 5 gives [19, 1, -4.4]. Best campaign: the first two, total 20.
P = np.array([0.8, 0.5, 0.2])
V = np.array([1000.0, 400.0, 100.0])


def test_customer_value_floors_negative_revenue_at_zero():
    assert customer_value(pd.Series([-50.0, 0.0, 120.5])).tolist() == [0.0, 0.0, 120.5]


def test_revenue_at_risk_hand_worked():
    assert revenue_at_risk(P, V) == pytest.approx(1020.0)


def test_expected_benefit_hand_worked():
    b = expected_benefit(P, V, save_rate=0.10, margin=0.30, offer_cost=5)
    assert b == pytest.approx([19.0, 1.0, -4.4])


def test_campaign_curve_starts_at_zero_and_ranks_by_benefit():
    b = np.array([1.0, -4.4, 19.0])            # deliberately unsorted
    assert campaign_curve(b) == pytest.approx([0.0, 19.0, 20.0, 15.6])


def test_best_campaign_hand_worked():
    best = best_campaign(np.array([19.0, 1.0, -4.4]))
    assert best["size"] == 2 and best["total_benefit"] == pytest.approx(20.0)


def test_best_campaign_is_empty_when_every_benefit_is_negative():
    best = best_campaign(np.array([-1.0, -2.0]))
    assert best["size"] == 0 and best["total_benefit"] == 0.0


def test_chosen_customers_are_the_top_by_benefit():
    chosen = chosen_customers(np.array([1.0, -4.4, 19.0]), 2)
    assert chosen.tolist() == [True, False, True]


def test_ties_keep_the_original_order():
    assert chosen_customers(np.array([3.0, 3.0, 3.0]), 2).tolist() == [True, True, False]


def test_rescale_hand_worked_logit_shift():
    # logits of 1/3 and 2/3 are -ln 2 and +ln 2; shifting both by +ln 2 gives sigmoid(0) = 0.5 and
    # sigmoid(2 ln 2) = 0.8, whose mean is 0.65.
    out = rescale_to_mean(np.array([1 / 3, 2 / 3]), 0.65)
    assert out["probabilities"] == pytest.approx([0.5, 0.8], abs=1e-9)
    assert out["shift"] == pytest.approx(math.log(2), abs=1e-9)


def test_rescale_of_equal_probabilities_hits_the_target_exactly():
    out = rescale_to_mean(np.full(4, 0.5), 0.2)
    assert out["probabilities"] == pytest.approx([0.2] * 4, abs=1e-9)


def test_rescale_keeps_order_and_bounds():
    p = np.array([0.05, 0.3, 0.9, 0.99])
    out = rescale_to_mean(p, 0.4)["probabilities"]
    assert out.mean() == pytest.approx(0.4, abs=1e-9)
    assert (np.diff(out) > 0).all() and out.min() > 0 and out.max() < 1


def test_rescale_rejects_targets_outside_zero_one():
    with pytest.raises(ValueError):
        rescale_to_mean(np.array([0.5]), 1.0)


CHURN_TABLE = pd.DataFrame({
    "cutoff": ["2011-05-01", "2011-06-01", "2011-07-01", "2011-09-01"],
    "active": [100, 200, 300, 400], "churned": [60, 120, 90, 100],
    "churn_rate": [0.6, 0.6, 0.3, 0.25],
})


def test_reference_rate_uses_a_window_that_ends_before_the_decision_day():
    ref = reference_churn_rate(CHURN_TABLE, "2011-06-01", "2011-09-01", 90)
    assert ref["churn_rate"] == pytest.approx(0.6)
    assert ref["window_end"] == "2011-08-30" and ref["known_on_decision_day"] is True


def test_reference_rate_refuses_a_window_still_open_on_the_decision_day():
    # 2011-07-01 + 90 days = 2011-09-29, after the decision day: its labels were not yet known.
    with pytest.raises(ValueError, match="not known"):
        reference_churn_rate(CHURN_TABLE, "2011-07-01", "2011-09-01", 90)


def test_reference_rate_must_be_the_latest_known_cutoff():
    with pytest.raises(ValueError, match="most recent"):
        reference_churn_rate(CHURN_TABLE, "2011-05-01", "2011-09-01", 90)


def test_scenario_grid_is_margin_by_save_rate_by_cost():
    cfg = {"margin": {"central": 0.3, "scenarios": [0.2, 0.3]},
           "save_rate": {"central": 0.1, "scenarios": [0.05, 0.1]},
           "offer_cost_gbp": {"central": 15, "scenarios": [5, 15, 40]}}
    grid = scenario_grid(cfg)
    assert len(grid) == 12
    assert sum(g["is_central"] for g in grid) == 1
    assert {"margin": 0.3, "save_rate": 0.1, "offer_cost": 15, "is_central": True} in grid


def test_business_config_has_rationales_and_central_values_inside_the_scenarios():
    cfg = load_config("business")
    for key in cfg:
        if not key.endswith("_rationale"):
            assert cfg.get(f"{key}_rationale"), f"{key} has no rationale"
    for key in ("margin", "save_rate", "offer_cost_gbp"):
        assert cfg[key]["central"] in cfg[key]["scenarios"]
        assert cfg[key]["central"] == sorted(cfg[key]["scenarios"])[1]   # low, central, high
    assert cfg["margin"]["scenarios"] == [0.20, 0.30, 0.40]
    assert cfg["save_rate"]["scenarios"] == [0.05, 0.10, 0.20]
    assert cfg["model"] == "gbm"
