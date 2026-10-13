"""Milestone C4, part A: tune on the validation cutoff, report validation metrics, freeze final models.

Run with:  python -m churn.run_c4      (needs the snapshot tables from python -m churn.run_c3)

Reads only the training snapshots (load_snapshots refuses the test cutoff). Validation metrics come
from models fitted on the tuning cutoffs; the final models are the chosen settings refitted on all
training cutoffs and are saved, with their sha256, for the one-shot test run (churn.run_c4_test).
"""

import sys
import warnings
from datetime import datetime, timezone

import pandas as pd
import sklearn

from churn.config import REPO_ROOT, load_config
from churn.metrics import bootstrap_intervals, evaluate, top_key
from churn.models import (
    MODEL_NAMES,
    feature_matrix,
    fit_final_models,
    load_snapshots,
    model_scores,
    rule_flag_stats,
    save_models,
    tune,
    validation_split,
)
from churn.plots import calibration_plot
from churn.report import markdown_table, render_template, save_report
from churn.run_c1 import git_state
from churn.snapshots import FEATURES

LABELS = {"rule": "Rule (rank by recency)", "logistic": "Logistic regression", "gbm": "Gradient boosting"}


def _day(ts):
    return f"{pd.Timestamp(ts):%Y-%m-%d}"


def base_rates(df):
    return {"customers": len(df), "churners": int(df["churned"].sum()), "churn_rate": float(df["churned"].mean())}


def cutoff_table(train_cutoffs, split, horizon):
    rows = []
    for c in train_cutoffs:
        end = c + pd.Timedelta(days=horizon)
        rows.append({"cutoff": _day(c), "label_window_end": _day(end),
                     "used_for_tuning": "yes" if c in split["tuning"] else "no",
                     "is_validation": "yes" if c == split["validation"] else "no"})
    return rows


def summary_row(name, m):
    row = {"model": name, "label": LABELS[name], "customers": m["customers"], "churners": m["churners"],
           "base_rate": m["base_rate"], "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"], "brier": m["brier"],
           "mean_predicted": m["mean_predicted"]}
    for t in m["top"]:
        pct = round(t["fraction"] * 100)
        row.update({f"top{pct}_k": t["k"], f"top{pct}_churners": t["churners_in_top"],
                    f"top{pct}_precision": t["precision"], f"top{pct}_recall": t["recall"]})
    iv = m["bootstrap"]["intervals"]
    for key in ["roc_auc", "pr_auc"] + [top_key(t["fraction"]) for t in m["top"]]:
        row[f"{key}_low"], row[f"{key}_high"] = iv[key]["low"], iv[key]["high"]
    return row


def main():
    data_cfg, churn_cfg, model_cfg = load_config("data"), load_config("churn"), load_config("model")
    paths = data_cfg["paths"]
    snap_dir, test = REPO_ROOT / paths["snapshots_dir"], churn_cfg["test_cutoff"]
    split = validation_split(churn_cfg, model_cfg)
    missing = [_day(c) for c in split["final"] if not (snap_dir / f"snapshot_{_day(c)}.parquet").exists()]
    if missing:
        sys.exit(f"Snapshots missing for {missing}. Run: python -m churn.run_c3")

    tuning_df = load_snapshots(snap_dir, split["tuning"], test)
    validation_df = load_snapshots(snap_dir, [split["validation"]], test)
    final_df = load_snapshots(snap_dir, split["final"], test)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tuned = tune(tuning_df, validation_df, model_cfg)
    fit_warnings = sorted({f"{w.category.__name__}: {str(w.message).splitlines()[0]}" for w in caught})

    # Validation metrics of the selected settings (fitted on the tuning cutoffs only).
    y_val = validation_df["churned"].to_numpy()
    metrics = {}
    for name, score in model_scores(tuned["models"], validation_df).items():
        m = evaluate(y_val, score, model_cfg["top_fractions"], model_cfg["calibration_bins"],
                     is_probability=name != "rule")
        m["bootstrap"] = bootstrap_intervals(y_val, score, model_cfg["top_fractions"],
                                             model_cfg["bootstrap_samples"], model_cfg["seed"])
        metrics[name] = m
        if name == "rule":
            m["flag"] = rule_flag_stats(tuned["models"]["rule"], feature_matrix(validation_df), y_val)

    # Final models: selected settings refitted on all training cutoffs, saved with their sha256.
    final_models = fit_final_models(final_df, tuned["selected"], model_cfg)
    hashes = save_models(final_models, REPO_ROOT / paths["models_dir"])
    git = git_state()
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git,
        "raw_sha256": data_cfg["raw"]["sha256"],
        "sklearn_version": sklearn.__version__,
        "features": list(FEATURES),
        "selected": tuned["selected"],
        "training_cutoffs": [_day(c) for c in split["final"]],
        "training_rows": base_rates(final_df),
        "models_dir": paths["models_dir"],
        "models": hashes,
    }
    save_report(manifest, REPO_ROOT / paths["c4_final_models"])

    # Saved tables and figure.
    candidates = pd.DataFrame(tuned["candidates"]).drop(columns=["params"], errors="ignore")
    candidates.to_csv(REPO_ROOT / paths["c4_validation_tuning"], index=False, float_format="%.4f")
    summary = [summary_row(name, metrics[name]) for name in MODEL_NAMES]
    pd.DataFrame(summary).to_csv(REPO_ROOT / paths["c4_validation_summary"], index=False, float_format="%.4f")
    calib = [{"model": name, **r} for name in ("logistic", "gbm") for r in metrics[name]["calibration"]]
    pd.DataFrame(calib).to_csv(REPO_ROOT / paths["c4_validation_calibration"], index=False, float_format="%.4f")
    calibration_plot({LABELS[n]: metrics[n]["calibration"] for n in ("logistic", "gbm")},
                     REPO_ROOT / paths["c4_calibration_plot"],
                     title=f"Calibration at the validation cutoff {_day(split['validation'])}")

    per_cutoff = {_day(c): base_rates(g) for c, g in final_df.groupby("cutoff")}
    report = {
        "generated_at": manifest["generated_at"],
        "git": git,
        "raw_sha256": data_cfg["raw"]["sha256"],
        "sklearn_version": sklearn.__version__,
        "test_cutoff": test,
        "feature_count": len(FEATURES),
        "test_cutoff_read": False,
        "scheme": {
            "validation_cutoff": _day(split["validation"]),
            "tuning_cutoffs": [_day(c) for c in split["tuning"]],
            "final_training_cutoffs": [_day(c) for c in split["final"]],
            "horizon_days": churn_cfg["churn_horizon_days"],
            "selection_metric": model_cfg["selection_metric"],
            "cutoffs": cutoff_table(split["final"], split, churn_cfg["churn_horizon_days"]),
        },
        "base_rates": {
            "tuning": base_rates(tuning_df),
            "validation": base_rates(validation_df),
            "final_training": base_rates(final_df),
            "per_training_cutoff": per_cutoff,
        },
        "candidates_count": {n: sum(r["model"] == n for r in tuned["candidates"]) for n in MODEL_NAMES},
        "selected": tuned["selected"],
        "fit_warnings": fit_warnings,
        "metrics": metrics,
        "summary": summary,
        "outputs": {k: paths[k] for k in ["c4_final_models", "c4_validation_summary", "c4_validation_tuning",
                                          "c4_validation_calibration", "c4_calibration_plot"]},
    }
    save_report(report, REPO_ROOT / paths["c4_validation_metrics"])
    render_doc(report, tuned["candidates"], paths)

    print(f"Tuning rows {len(tuning_df):,}, validation rows {len(validation_df):,}, "
          f"final training rows {len(final_df):,}")
    for r in summary:
        print(f"  {r['label']:<20} PR-AUC {r['pr_auc']:.3f}  ROC-AUC {r['roc_auc']:.3f}  "
              f"P@10% {r['top10_precision']:.3f}")
    print(f"Report: {paths['c4_validation_metrics']}   Doc: {paths['c4_doc_output']}")


def render_doc(report, candidates, paths):
    pct = "pct"
    s = report["selected"]
    report["selected_text"] = {
        "rule": f"recency_days > {s['rule']['threshold_days']}",
        "logistic": s["logistic"]["preprocessing"],
        "gbm": ", ".join(f"{k}={v}" for k, v in s["gbm"]["params"].items()),
    }
    rates = report["base_rates"]
    base_rows = [{"set": "Tuning (fit)", "cutoffs": ", ".join(report["scheme"]["tuning_cutoffs"]), **rates["tuning"]},
                 {"set": "Validation (score)", "cutoffs": report["scheme"]["validation_cutoff"], **rates["validation"]},
                 {"set": "Final training (refit)", "cutoffs": "all 6 training cutoffs", **rates["final_training"]}]
    summary = report["summary"]

    def ci(r, key):
        return f"{r[key]:.3f} ({r[key + '_low']:.3f}-{r[key + '_high']:.3f})"

    metric_rows = [{
        "model": r["label"], "roc": ci(r, "roc_auc"), "pr": ci(r, "pr_auc"),
        "p10": f"{r['top10_precision']:.1%} ({r['precision_top_10pct_low']:.1%}-{r['precision_top_10pct_high']:.1%})",
        "p20": f"{r['top20_precision']:.1%} ({r['precision_top_20pct_low']:.1%}-{r['precision_top_20pct_high']:.1%})",
        "brier": "n/a (not a probability)" if r["brier"] is None else f"{r['brier']:.3f}",
        "mean": "n/a" if r["mean_predicted"] is None else f"{r['mean_predicted']:.1%}",
    } for r in summary]
    count_rows = [{
        "model": r["label"],
        "k10": r["top10_k"], "c10": _count(r["top10_churners"]), "r10": r["top10_recall"],
        "k20": r["top20_k"], "c20": _count(r["top20_churners"]), "r20": r["top20_recall"],
    } for r in summary]
    calib_rows = []
    for name in ("logistic", "gbm"):
        for row in report["metrics"][name]["calibration"]:
            calib_rows.append({"model": {"logistic": "Logistic", "gbm": "Boosting"}[name],
                               "bin": f"{row['lower']:.0%}-{row['upper']:.0%}", "customers": row["customers"],
                               "pred": row["mean_predicted"], "obs": row["observed_churn_rate"]})
    cand_rows = [{**r, "model": LABELS[r["model"]], "pr_auc": f"{r['pr_auc']:.3f}",
                  "roc_auc": f"{r['roc_auc']:.3f}"} for r in candidates]

    tables = {
        "cutoffs": markdown_table(report["scheme"]["cutoffs"], [
            ("cutoff", "Training cutoff"), ("label_window_end", "90-day window ends"),
            ("used_for_tuning", "Used to fit during tuning"), ("is_validation", "Validation cutoff")]),
        "base_rates": markdown_table(base_rows, [("set", "Set"), ("cutoffs", "Cutoffs"),
                                                 ("customers", "Rows"), ("churners", "Churned"),
                                                 ("churn_rate", "Churn rate", pct)]),
        "per_cutoff": markdown_table(
            [{"cutoff": c, **v} for c, v in report["base_rates"]["per_training_cutoff"].items()],
            [("cutoff", "Cutoff"), ("customers", "Customers"), ("churners", "Churned"),
             ("churn_rate", "Churn rate", pct)]),
        "metrics": markdown_table(metric_rows, [
            ("model", "Model"), ("roc", "ROC-AUC (95% CI)"), ("pr", "PR-AUC (95% CI)"),
            ("p10", "Precision top 10% (95% CI)"), ("p20", "Precision top 20% (95% CI)"), ("brier", "Brier"),
            ("mean", "Mean predicted")]),
        "counts": markdown_table(count_rows, [
            ("model", "Model"), ("k10", "Top 10%: customers"), ("c10", "churners"), ("r10", "recall", pct),
            ("k20", "Top 20%: customers"), ("c20", "churners"), ("r20", "recall", pct)]),
        "calibration": markdown_table(calib_rows, [
            ("model", "Model"), ("bin", "Predicted bin"), ("customers", "Customers"),
            ("pred", "Mean predicted", pct), ("obs", "Observed churn", pct)]),
        "candidates": markdown_table(cand_rows, [
            ("model", "Model"), ("setting", "Setting"), ("pr_auc", "Validation PR-AUC"),
            ("roc_auc", "ROC-AUC"), ("precision_top_10pct", "Precision top 10%", pct)]),
    }
    template = (REPO_ROOT / paths["c4_doc_template"]).read_text()
    (REPO_ROOT / paths["c4_doc_output"]).write_text(render_template(template, report, tables))


def _count(x):
    return f"{x:,}" if float(x).is_integer() else f"{x:,.1f}"


if __name__ == "__main__":
    main()
