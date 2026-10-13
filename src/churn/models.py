"""Churn models (PROJECT_PLAN.md Section 5) and the validation scheme (config/model.yaml).

  rule      ranks customers by recency_days (longest since last purchase first); the yes/no flag
            "recency_days > X" is reported alongside, with X chosen on validation
  logistic  preprocessing (log1p / signed log / clipping, then standardising) + logistic regression
  gbm       scikit-learn HistGradientBoostingClassifier

Tuning: fit on the tuning cutoffs (outcome window ends on or before the validation cutoff), score
the validation cutoff, keep the setting with the best validation PR-AUC. Final models: the chosen
settings refitted on all training cutoffs. The test cutoff is never read here.
"""

import hashlib
import itertools
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn.metrics import evaluate
from churn.snapshots import FEATURES, training_cutoffs

MODEL_NAMES = ("rule", "logistic", "gbm")
PREPROCESSING = ("standardise_only", "log_signed_log", "log_clip")


# --- validation scheme -------------------------------------------------------------------------

def tuning_cutoffs(train_cutoffs, validation_cutoff, horizon_days):
    """Training cutoffs whose outcome window [c, c + horizon) ends on or before the validation cutoff."""
    return [c for c in train_cutoffs if c + pd.Timedelta(days=horizon_days) <= validation_cutoff]


def validation_split(churn_cfg, model_cfg):
    """{'tuning': [...], 'validation': cutoff, 'final': [...]} with the checks the scheme relies on."""
    train = training_cutoffs(churn_cfg)
    validation = pd.Timestamp(model_cfg["validation_cutoff"])
    test = pd.Timestamp(churn_cfg["test_cutoff"])
    if validation >= test:
        raise ValueError("The validation cutoff must come before the test cutoff")
    if validation not in train:
        raise ValueError("The validation cutoff must be one of the training cutoffs")
    tuning = tuning_cutoffs(train, validation, churn_cfg["churn_horizon_days"])
    if not tuning:
        raise ValueError("no training cutoff has an outcome window ending by the validation cutoff")
    return {"tuning": tuning, "validation": validation, "final": train}


def load_snapshots(snapshots_dir, cutoffs, test_cutoff):
    """Stack the saved snapshot tables of the given cutoffs. Refuses the test cutoff."""
    if pd.Timestamp(test_cutoff) in [pd.Timestamp(c) for c in cutoffs]:
        raise ValueError("Refusing to load the test cutoff during model development")
    frames = [pd.read_parquet(Path(snapshots_dir) / f"snapshot_{pd.Timestamp(c):%Y-%m-%d}.parquet")
              for c in cutoffs]
    return pd.concat(frames, ignore_index=True)


def feature_matrix(df):
    """The 21 features in a fixed order (FEATURES), as floats."""
    return df[list(FEATURES)].astype(float)


# --- models -------------------------------------------------------------------------------------

class RecencyRule(BaseEstimator):
    """The simple baseline. score() ranks customers by recency_days, so every ranking metric
    (AUC, PR-AUC, top 10%/20%) uses days since the last purchase and does not depend on X.
    flag() is the yes/no rule "recency_days > threshold_days". Nothing is learned."""

    def __init__(self, threshold_days):
        self.threshold_days = threshold_days

    def fit(self, X, y=None):
        return self

    def score(self, X):
        return X["recency_days"].to_numpy().astype(float)

    def flag(self, X):
        return (X["recency_days"].to_numpy() > self.threshold_days).astype(float)


def rule_flag_stats(rule, X, y):
    """How the yes/no flag does: customers flagged, and its precision and recall."""
    flagged = rule.flag(X).astype(bool)
    y = np.asarray(y).astype(bool)
    return {
        "threshold_days": rule.threshold_days,
        "flagged": int(flagged.sum()),
        "flagged_share": float(flagged.mean()),
        "flag_precision": float(y[flagged].mean()) if flagged.any() else None,
        "flag_recall": float(flagged[y].mean()) if y.any() else None,
    }


class LogisticPreprocessor(BaseEstimator, TransformerMixin):
    """Reshape skewed features, then standardise. Every fitted value (clip bounds, means, standard
    deviations) comes from the rows passed to fit, i.e. training rows only.

    method: standardise_only | log_signed_log | log_clip
    transforms: {'log1p': [...], 'signed': [...], 'binary': [...]} (config/model.yaml)
    """

    def __init__(self, method, transforms, clip_percentiles):
        self.method = method
        self.transforms = transforms
        self.clip_percentiles = clip_percentiles

    def _columns(self):
        return [c for group in ("log1p", "signed", "binary") for c in self.transforms[group]]

    def reshape(self, X, clip):
        """Apply the method's per-feature transforms (before standardising). clip: (low, high) dicts."""
        X = X[self._columns()].astype(float).copy()
        if self.method not in PREPROCESSING:
            raise ValueError(f"Unknown preprocessing {self.method!r}; expected one of {PREPROCESSING}")
        if self.method == "standardise_only":
            return X.to_numpy()
        log_cols, signed_cols = self.transforms["log1p"], self.transforms["signed"]
        if (X[log_cols] < 0).any().any():
            bad = [c for c in log_cols if (X[c] < 0).any()]
            raise ValueError(f"log1p features must not be negative: {bad}")
        X[log_cols] = np.log1p(X[log_cols])
        if self.method == "log_signed_log":
            X[signed_cols] = np.sign(X[signed_cols]) * np.log1p(X[signed_cols].abs())
        elif clip is not None:
            low, high = clip
            for c in signed_cols:
                X[c] = X[c].clip(low[c], high[c])
        return X.to_numpy()

    def fit(self, X, y=None):
        self.clip_low_, self.clip_high_ = None, None
        if self.method == "log_clip":
            lo, hi = self.clip_percentiles
            signed = X[self.transforms["signed"]].astype(float)
            self.clip_low_ = {c: float(np.percentile(signed[c], lo)) for c in signed}
            self.clip_high_ = {c: float(np.percentile(signed[c], hi)) for c in signed}
        shaped = self.reshape(X, self._clip())
        self.mean_ = shaped.mean(axis=0)
        scale = shaped.std(axis=0)
        self.scale_ = np.where(scale > 0, scale, 1.0)   # a constant column stays 0 after centring
        return self

    def _clip(self):
        return None if self.clip_low_ is None else (self.clip_low_, self.clip_high_)

    def transform(self, X):
        return (self.reshape(X, self._clip()) - self.mean_) / self.scale_


def make_logistic(preprocessing, model_cfg):
    prep = LogisticPreprocessor(preprocessing, model_cfg["feature_transforms"],
                                model_cfg["logistic_clip_percentiles"])
    model = LogisticRegression(**model_cfg["logistic_params"], random_state=model_cfg["seed"])
    return Pipeline([("prep", prep), ("model", model)])


def make_gbm(params, model_cfg):
    return HistGradientBoostingClassifier(**model_cfg["gbm_fixed_params"], **params,
                                          random_state=model_cfg["seed"])


def gbm_settings(grid):
    """All combinations of the grid, in a fixed order (keys sorted, values as listed)."""
    keys = sorted(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def model_scores(models, df):
    """{name: churn scores} for a snapshot table: recency_days for the rule, probabilities otherwise."""
    X = feature_matrix(df)
    out = {}
    for name, model in models.items():
        out[name] = model.score(X) if isinstance(model, RecencyRule) else model.predict_proba(X)[:, 1]
    return out


# --- tuning and final fit -----------------------------------------------------------------------

def select_best(rows, metric):
    """The row with the highest metric; on a tie the earliest candidate wins."""
    best = rows[0]
    for r in rows[1:]:
        if r[metric] > best[metric]:
            best = r
    return best


def _validation_row(model, setting, label, y_val, score, model_cfg):
    m = evaluate(y_val, score, model_cfg["top_fractions"], model_cfg["calibration_bins"],
                 is_probability=model != "rule")
    return {"model": model, "setting": label, **setting, "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"],
            **{f"precision_top_{round(t['fraction'] * 100)}pct": t["precision"] for t in m["top"]}}


def tune(tuning_df, validation_df, model_cfg):
    """Fit every candidate on the tuning rows, score the validation rows, select by validation PR-AUC.
    Returns all candidate rows, the selected settings and the selected models (fitted on tuning rows)."""
    metric = model_cfg["selection_metric"]
    X_tr, y_tr = feature_matrix(tuning_df), tuning_df["churned"].to_numpy()
    X_val, y_val = feature_matrix(validation_df), validation_df["churned"].to_numpy()
    rows, fitted = {"rule": [], "logistic": [], "gbm": []}, {}

    for x in model_cfg["rule_thresholds_days"]:
        rule = RecencyRule(x).fit(X_tr, y_tr)
        # X only sets the yes/no flag, so candidates are compared on the flag (ranking ignores X).
        rows["rule"].append(_validation_row("rule", {"threshold_days": x}, f"recency_days > {x}",
                                            y_val, rule.flag(X_val), model_cfg))
        fitted[("rule", f"recency_days > {x}")] = rule
    for prep in model_cfg["logistic_preprocessing_options"]:
        model = make_logistic(prep, model_cfg).fit(X_tr, y_tr)
        rows["logistic"].append(_validation_row("logistic", {"preprocessing": prep}, prep,
                                                y_val, model.predict_proba(X_val)[:, 1], model_cfg))
        fitted[("logistic", prep)] = model
    for params in gbm_settings(model_cfg["gbm_grid"]):
        label = ", ".join(f"{k}={v}" for k, v in params.items())
        model = make_gbm(params, model_cfg).fit(X_tr, y_tr)
        rows["gbm"].append(_validation_row("gbm", {"params": params}, label,
                                           y_val, model.predict_proba(X_val)[:, 1], model_cfg))
        fitted[("gbm", label)] = model

    best = {name: select_best(rows[name], metric) for name in MODEL_NAMES}
    selected = {
        "rule": {"threshold_days": best["rule"]["threshold_days"]},
        "logistic": {"preprocessing": best["logistic"]["preprocessing"]},
        "gbm": {"params": best["gbm"]["params"]},
    }
    models = {name: fitted[(name, best[name]["setting"])] for name in MODEL_NAMES}
    return {"candidates": [r for name in MODEL_NAMES for r in rows[name]], "selected": selected,
            "models": models}


def fit_final_models(train_df, selected, model_cfg):
    """The selected settings refitted on the given (all training cutoffs) rows."""
    X, y = feature_matrix(train_df), train_df["churned"].to_numpy()
    return {
        "rule": RecencyRule(selected["rule"]["threshold_days"]).fit(X, y),
        "logistic": make_logistic(selected["logistic"]["preprocessing"], model_cfg).fit(X, y),
        "gbm": make_gbm(selected["gbm"]["params"], model_cfg).fit(X, y),
    }


# --- frozen model files -------------------------------------------------------------------------

def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_models(models, models_dir):
    """Save each model as <name>.joblib; return {name: {'file', 'sha256'}} for the manifest."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, model in models.items():
        path = models_dir / f"{name}.joblib"
        joblib.dump(model, path)
        out[name] = {"file": path.name, "sha256": _sha256(path)}
    return out


def load_models(models_dir, manifest_models):
    """Load the frozen models, refusing any file whose sha256 differs from the manifest."""
    models = {}
    for name, entry in manifest_models.items():
        path = Path(models_dir) / entry["file"]
        if _sha256(path) != entry["sha256"]:
            raise ValueError(f"{path.name}: sha256 differs from the frozen manifest")
        models[name] = joblib.load(path)
    return models
