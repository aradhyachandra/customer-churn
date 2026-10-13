"""Model tests: validation scheme, logistic preprocessing fitted on training rows only, the rule,
the gradient boosting grid, reproducibility and frozen model files. Synthetic data only."""

import numpy as np
import pandas as pd
import pytest

from churn.config import load_config
from churn.models import (
    LogisticPreprocessor,
    RecencyRule,
    feature_matrix,
    fit_final_models,
    gbm_settings,
    load_models,
    load_snapshots,
    make_gbm,
    make_logistic,
    model_scores,
    rule_flag_stats,
    save_models,
    select_best,
    tune,
    tuning_cutoffs,
    validation_split,
)
from churn.snapshots import FEATURES
from tests.conftest import make_snapshot

T = pd.Timestamp
MONTHLY = [T("2010-12-01"), T("2011-01-01"), T("2011-02-01"), T("2011-03-01"), T("2011-04-01"),
           T("2011-05-01")]


# --- validation scheme -------------------------------------------------------------------------

def test_tuning_cutoffs_need_label_window_to_end_by_validation_cutoff():
    # 2010-12-01 + 90 days = 2011-03-01, 2011-01-01 + 90 = 2011-04-01, 2011-02-01 + 90 = 2011-05-02.
    assert tuning_cutoffs(MONTHLY, T("2011-05-01"), 90) == [T("2010-12-01"), T("2011-01-01")]
    # A window ending exactly on the validation cutoff qualifies.
    assert tuning_cutoffs(MONTHLY, T("2011-03-01"), 90) == [T("2010-12-01")]


def test_configured_validation_scheme():
    churn_cfg, model_cfg = load_config("churn"), load_config("model")
    split = validation_split(churn_cfg, model_cfg)
    assert split["tuning"] == [T("2010-12-01"), T("2011-01-01")]
    assert split["validation"] == T("2011-05-01")
    assert split["final"] == MONTHLY
    test = T(churn_cfg["test_cutoff"])
    assert test not in split["tuning"] + [split["validation"]] + split["final"]
    for c in split["tuning"]:
        assert c + pd.Timedelta(days=90) <= split["validation"]
    for c in split["final"]:
        assert c + pd.Timedelta(days=90) <= test


def test_validation_split_rejects_bad_settings():
    churn_cfg, model_cfg = load_config("churn"), load_config("model")
    with pytest.raises(ValueError, match="test cutoff"):
        validation_split({**churn_cfg, "test_cutoff": "2011-05-01"}, model_cfg)
    with pytest.raises(ValueError, match="training cutoff"):
        validation_split(churn_cfg, {**model_cfg, "validation_cutoff": "2011-06-01"})
    with pytest.raises(ValueError, match="no training cutoff"):
        validation_split(churn_cfg, {**model_cfg, "validation_cutoff": "2010-12-01"})


def test_load_snapshots_refuses_the_test_cutoff(tmp_path):
    with pytest.raises(ValueError, match="test cutoff"):
        load_snapshots(tmp_path, [T("2011-05-01"), T("2011-09-01")], T("2011-09-01"))


def test_load_snapshots_reads_and_stacks(tmp_path):
    for c in ["2010-12-01", "2011-01-01"]:
        make_snapshot(20, c).to_parquet(tmp_path / f"snapshot_{c}.parquet", index=False)
    df = load_snapshots(tmp_path, [T("2010-12-01"), T("2011-01-01")], T("2011-09-01"))
    assert len(df) == 40 and set(df["cutoff"]) == {T("2010-12-01"), T("2011-01-01")}


def test_model_config_has_rationales_and_complete_transforms():
    cfg = load_config("model")
    for key in cfg:
        if not key.endswith("_rationale"):
            assert cfg.get(f"{key}_rationale"), f"{key} has no rationale"
    listed = [f for group in cfg["feature_transforms"].values() for f in group]
    assert sorted(listed) == sorted(FEATURES)          # every feature exactly once
    assert len(gbm_settings(cfg["gbm_grid"])) <= 12
    assert cfg["bootstrap_samples"] == 1000
    assert cfg["top_fractions"] == [0.10, 0.20]


# --- rule ---------------------------------------------------------------------------------------

def test_recency_rule_flags_strictly_above_threshold():
    X = pd.DataFrame({"recency_days": [10, 90, 91, 300]})
    assert list(RecencyRule(90).fit(X).flag(X)) == [0.0, 0.0, 1.0, 1.0]


def test_recency_rule_ranks_by_recency_whatever_the_threshold():
    X = pd.DataFrame({"recency_days": [10, 90, 91, 300]})
    assert list(RecencyRule(90).fit(X).score(X)) == [10.0, 90.0, 91.0, 300.0]
    assert list(RecencyRule(30).score(X)) == list(RecencyRule(240).score(X))


def test_rule_flag_stats_are_worked_by_hand():
    X = pd.DataFrame({"recency_days": [10, 50, 100, 150, 200]})
    y = np.array([0, 1, 0, 1, 1])
    # flagged (> 90): rows 3-5, of which 2 churned; 3 churners in total
    stats = rule_flag_stats(RecencyRule(90), X, y)
    assert stats == {"threshold_days": 90, "flagged": 3, "flagged_share": 0.6,
                     "flag_precision": pytest.approx(2 / 3), "flag_recall": pytest.approx(2 / 3)}


# --- logistic preprocessing ---------------------------------------------------------------------

TRANSFORMS = {"log1p": ["a"], "signed": ["b"], "binary": ["c"]}


def small_frame(a, b, c):
    return pd.DataFrame({"a": a, "b": b, "c": c}, dtype=float)


def test_log_and_signed_log_before_standardising():
    e1 = np.e - 1
    prep = LogisticPreprocessor("log_signed_log", TRANSFORMS, [1, 99])
    shaped = prep.reshape(small_frame([0, e1], [-e1, e1], [0, 1]), clip=None)
    assert shaped[:, 0] == pytest.approx([0, 1])      # log1p
    assert shaped[:, 1] == pytest.approx([-1, 1])     # sign(x) * log1p(|x|)
    assert shaped[:, 2] == pytest.approx([0, 1])      # flags unchanged


def test_standardise_only_keeps_values():
    prep = LogisticPreprocessor("standardise_only", TRANSFORMS, [1, 99])
    shaped = prep.reshape(small_frame([0, 5], [-3, 7], [0, 1]), clip=None)
    assert shaped[:, 1] == pytest.approx([-3, 7])


def test_negative_value_in_log_feature_stops():
    prep = LogisticPreprocessor("log_signed_log", TRANSFORMS, [1, 99])
    with pytest.raises(ValueError, match="negative"):
        prep.fit(small_frame([1, -2], [0, 0], [0, 1]))


def test_preprocessing_is_fitted_on_training_rows_only():
    rng = np.random.default_rng(3)
    train = small_frame(rng.exponential(5, 200), rng.normal(0, 100, 200), rng.integers(0, 2, 200))
    val = small_frame(rng.exponential(500, 50), rng.normal(5000, 1e5, 50), rng.integers(0, 2, 50))
    for method in ["standardise_only", "log_signed_log", "log_clip"]:
        prep = LogisticPreprocessor(method, TRANSFORMS, [1, 99]).fit(train)
        alone = LogisticPreprocessor(method, TRANSFORMS, [1, 99]).fit(train.copy())
        # Fitted values are a function of the training rows alone ...
        assert np.array_equal(prep.mean_, alone.mean_) and np.array_equal(prep.scale_, alone.scale_)
        # ... and transforming validation rows does not change them.
        before = (prep.mean_.copy(), prep.scale_.copy(), prep.clip_low_, prep.clip_high_)
        out_val = prep.transform(val)
        assert np.array_equal(before[0], prep.mean_) and np.array_equal(before[1], prep.scale_)
        out_train = prep.transform(train)
        assert out_train.mean(axis=0) == pytest.approx([0, 0, 0], abs=1e-9)
        assert out_train.std(axis=0) == pytest.approx([1, 1, 1])
        assert np.abs(out_val.mean(axis=0)).max() > 0.5   # validation is not re-centred


def test_log_clip_uses_training_percentiles():
    train = small_frame(np.ones(101), np.arange(101), np.zeros(101))  # b: 0..100
    prep = LogisticPreprocessor("log_clip", TRANSFORMS, [1, 99]).fit(train)
    assert prep.clip_low_["b"] == pytest.approx(1) and prep.clip_high_["b"] == pytest.approx(99)
    shaped = prep.reshape(small_frame([1, 1], [-1e6, 1e6], [0, 0]), clip=(prep.clip_low_, prep.clip_high_))
    assert list(shaped[:, 1]) == pytest.approx([1, 99])


def test_logistic_pipeline_fits_preprocessing_inside():
    cfg = load_config("model")
    df = make_snapshot(300, "2011-01-01", seed=1)
    model = make_logistic("log_clip", cfg).fit(feature_matrix(df), df["churned"])
    prep = model.named_steps["prep"]
    expected = LogisticPreprocessor("log_clip", cfg["feature_transforms"], cfg["logistic_clip_percentiles"]
                                    ).fit(feature_matrix(df))
    assert np.array_equal(prep.mean_, expected.mean_)


# --- gradient boosting, tuning, reproducibility --------------------------------------------------

def test_gbm_grid_expands_to_distinct_settings():
    grid = {"learning_rate": [0.03, 0.1], "max_leaf_nodes": [8, 31], "min_samples_leaf": [20, 100, 300]}
    settings = gbm_settings(grid)
    assert len(settings) == 12
    assert len({tuple(sorted(s.items())) for s in settings}) == 12
    assert settings[0] == {"learning_rate": 0.03, "max_leaf_nodes": 8, "min_samples_leaf": 20}


def test_gbm_is_reproducible_with_fixed_seed():
    cfg = load_config("model")
    df = make_snapshot(400, "2011-01-01", seed=2)
    X, y = feature_matrix(df), df["churned"]
    params = {"learning_rate": 0.1, "max_leaf_nodes": 8, "min_samples_leaf": 20}
    a = make_gbm(params, cfg).fit(X, y).predict_proba(X)[:, 1]
    b = make_gbm(params, cfg).fit(X, y).predict_proba(X)[:, 1]
    assert np.array_equal(a, b)


def test_select_best_takes_highest_and_first_on_ties():
    rows = [{"setting": "a", "pr_auc": 0.5}, {"setting": "b", "pr_auc": 0.7}, {"setting": "c", "pr_auc": 0.7}]
    assert select_best(rows, "pr_auc")["setting"] == "b"


def test_tune_scores_every_candidate_on_validation():
    cfg = {**load_config("model"), "gbm_grid": {"learning_rate": [0.1], "max_leaf_nodes": [8],
                                                "min_samples_leaf": [20, 50]}}
    tuning = pd.concat([make_snapshot(300, "2010-12-01", 4), make_snapshot(300, "2011-01-01", 5)])
    val = make_snapshot(300, "2011-05-01", 6)
    out = tune(tuning, val, cfg)
    models = [r["model"] for r in out["candidates"]]
    assert models.count("rule") == len(cfg["rule_thresholds_days"])
    assert models.count("logistic") == 3 and models.count("gbm") == 2
    assert out["selected"]["rule"]["threshold_days"] in cfg["rule_thresholds_days"]
    assert out["selected"]["logistic"]["preprocessing"] in cfg["logistic_preprocessing_options"]
    best_gbm = max((r for r in out["candidates"] if r["model"] == "gbm"), key=lambda r: r["pr_auc"])
    assert out["selected"]["gbm"]["params"] == best_gbm["params"]
    # The validation models were fitted on tuning rows: their scores have one value per validation row.
    scores = model_scores(out["models"], val)
    assert all(len(s) == len(val) for s in scores.values())


# --- frozen models ------------------------------------------------------------------------------

def frozen(tmp_path):
    cfg = load_config("model")
    selected = {"rule": {"threshold_days": 90}, "logistic": {"preprocessing": "log_signed_log"},
                "gbm": {"params": {"learning_rate": 0.1, "max_leaf_nodes": 8, "min_samples_leaf": 20}}}
    models = fit_final_models(make_snapshot(300, "2011-01-01", 8), selected, cfg)
    return models, save_models(models, tmp_path)


def test_saved_models_round_trip(tmp_path):
    models, hashes = frozen(tmp_path)
    loaded = load_models(tmp_path, hashes)
    df = make_snapshot(50, "2011-05-01", 9)
    for name in ["rule", "logistic", "gbm"]:
        assert np.array_equal(model_scores(models, df)[name], model_scores(loaded, df)[name])


def test_changed_model_file_is_refused(tmp_path):
    _, hashes = frozen(tmp_path)
    (tmp_path / hashes["gbm"]["file"]).write_bytes(b"not the frozen model")
    with pytest.raises(ValueError, match="sha256"):
        load_models(tmp_path, hashes)
