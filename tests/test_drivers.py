"""Driver analysis: plain-language group statements (hand-worked), permutation importance and
partial dependence on synthetic fixtures."""

import numpy as np
import pandas as pd
import pytest

from churn.config import load_config
from churn.drivers import group_churn, partial_dependence_table, permutation_importance_table
from churn.models import feature_matrix, fit_final_models
from churn.snapshots import FEATURES
from tests.conftest import make_snapshot

# Cutoff A: 4 rows, 2 in the group (1 churned), 2 outside (0 churned).
# Cutoff B: 4 rows, 3 in the group (3 churned), 1 outside (1 churned).
# Pooled: in group 5 rows, 4 churned (80%); outside 3 rows, 1 churned (33.3%).
TABLE = pd.DataFrame({
    "cutoff": pd.to_datetime(["2011-01-01"] * 4 + ["2011-02-01"] * 4),
    "recency_gap_ratio": [3.0, 2.5, 1.0, 2.0, 4.0, 2.1, 9.0, 0.5],
    "churned": [1, 0, 0, 0, 1, 1, 1, 1],
})


def test_group_churn_hand_worked():
    s = group_churn(TABLE, {"feature": "recency_gap_ratio", "op": ">", "value": 2})
    assert s["rows_in"] == 5 and s["churned_in"] == 4 and s["churn_in"] == pytest.approx(0.8)
    assert s["rows_out"] == 3 and s["churned_out"] == 1 and s["churn_out"] == pytest.approx(1 / 3)
    assert s["ratio"] == pytest.approx(2.4)
    assert s["share_in"] == pytest.approx(5 / 8)
    # Per cutoff: in-group churn 50% (A) and 100% (B); outside 0% (A) and 100% (B).
    assert s["churn_in_min"] == pytest.approx(0.5) and s["churn_in_max"] == pytest.approx(1.0)
    assert s["churn_out_min"] == pytest.approx(0.0) and s["churn_out_max"] == pytest.approx(1.0)


def test_group_churn_among_a_subgroup_only():
    # Only rows with recency_gap_ratio < 5 (drops the 9.0 row): in group 4 rows, 3 churned; rest 3, 1.
    s = group_churn(TABLE, {"feature": "recency_gap_ratio", "op": ">", "value": 2,
                            "among": {"feature": "recency_gap_ratio", "op": "<", "value": 5}})
    assert s["rows_in"] == 4 and s["churned_in"] == 3 and s["rows_out"] == 3 and s["churned_out"] == 1
    assert s["share_in"] == pytest.approx(4 / 7)


def test_group_churn_equality_condition():
    s = group_churn(TABLE, {"feature": "recency_gap_ratio", "op": "==", "value": 2.0})
    assert s["rows_in"] == 1 and s["churned_in"] == 0


def test_group_churn_rejects_unknown_operator():
    with pytest.raises(ValueError):
        group_churn(TABLE, {"feature": "recency_gap_ratio", "op": "~", "value": 2})


@pytest.fixture(scope="module")
def fitted():
    cfg = load_config("model")
    selected = {"rule": {"threshold_days": 90}, "logistic": {"preprocessing": "log_clip"},
                "gbm": {"params": {"learning_rate": 0.1, "max_leaf_nodes": 8, "min_samples_leaf": 20}}}
    model = fit_final_models(make_snapshot(600, "2011-01-01", seed=21), selected, cfg)["gbm"]
    test = make_snapshot(400, "2011-09-01", seed=22)
    return model, test


def test_permutation_importance_finds_the_feature_that_drives_churn(fitted):
    model, test = fitted
    table = permutation_importance_table(model, feature_matrix(test), test["churned"].to_numpy(),
                                         scoring="average_precision", n_repeats=5, seed=1)
    assert list(table["feature"]) and set(table["feature"]) == set(FEATURES)
    assert table.iloc[0]["feature"] == "recency_days"      # the fixture's churn depends on recency only
    assert (table["rank"] == range(1, len(FEATURES) + 1)).all()
    assert (table["low"] <= table["mean"]).all() and (table["mean"] <= table["high"]).all()
    assert table["repeats"].eq(5).all()


def test_permutation_importance_is_reproducible(fitted):
    model, test = fitted
    args = dict(scoring="average_precision", n_repeats=3, seed=7)
    a = permutation_importance_table(model, feature_matrix(test), test["churned"].to_numpy(), **args)
    b = permutation_importance_table(model, feature_matrix(test), test["churned"].to_numpy(), **args)
    pd.testing.assert_frame_equal(a, b)


def test_partial_dependence_is_on_the_probability_scale(fitted):
    model, test = fitted
    pd_table = partial_dependence_table(model, feature_matrix(test), ["recency_days", "is_uk"],
                                        grid_resolution=10, percentiles=(0.05, 0.95))
    rec = pd_table[pd_table["feature"] == "recency_days"]
    assert len(rec) == 10 and rec["value"].is_monotonic_increasing
    assert pd_table["mean_predicted"].between(0, 1).all()
    assert rec["mean_predicted"].iloc[-1] > rec["mean_predicted"].iloc[0]   # longer recency, more churn
    assert sorted(pd_table[pd_table["feature"] == "is_uk"]["value"]) == [0, 1]
    # Brute force: the curve at a value equals the average prediction with the feature set to it.
    X = feature_matrix(test).copy()
    X["is_uk"] = 1.0
    row = pd_table[(pd_table["feature"] == "is_uk") & (pd_table["value"] == 1)]
    assert row["mean_predicted"].iloc[0] == pytest.approx(model.predict_proba(X)[:, 1].mean())


def test_drivers_config_has_rationales_and_valid_statements():
    cfg = load_config("drivers")
    for key in cfg:
        if not key.endswith("_rationale"):
            assert cfg.get(f"{key}_rationale"), f"{key} has no rationale"
    for s in cfg["statements"]:
        assert s["feature"] in FEATURES and s["op"] in {">", ">=", "<", "<=", "==", "!="}
        assert s["group"] and s["rationale"]
    assert np.isclose(cfg["partial_dependence"]["top_n"], 4)
