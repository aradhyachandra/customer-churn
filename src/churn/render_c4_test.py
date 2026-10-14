"""Milestone C4, part B (write-up): fill the test section of docs/C4_models.md from saved results.

Run with:  python -m churn.render_c4_test

Reads results/c4_test_metrics.json (written ONCE by churn.run_c4_test), results/c4_validation_metrics.json
and results/c2_churn_by_cutoff.csv. It loads no model and no snapshot and scores nothing; it only
formats saved numbers, draws the test calibration plot from them and replaces the
"## Test results (part B)" section of the doc.
"""

import json

import pandas as pd

from churn.config import REPO_ROOT, load_config
from churn.metrics import top_key
from churn.plots import calibration_plot
from churn.report import markdown_table, render_template, save_report
from churn.run_c4 import LABELS, _count, summary_row
from churn.snapshots import training_cutoffs

SECTION = "## Test results (part B)"
COMPARED = (("pr_auc", "PR-AUC"), ("precision_top_10pct", "precision in the top 10%"),
            ("precision_top_20pct", "precision in the top 20%"))
SEASON_MONTHS = (9, 10, 11)


def intervals_overlap(a, b):
    return a["low"] <= b["high"] and b["low"] <= a["high"]


def comparison_sentence(name_a, a, name_b, b):
    """'A beats B' only when the 95% intervals of all compared metrics are apart; otherwise not settled."""
    ia, ib = a["bootstrap"]["intervals"], b["bootstrap"]["intervals"]
    overlapping = [label for key, label in COMPARED if intervals_overlap(ia[key], ib[key])]
    names = ", ".join(label for _, label in COMPARED)
    if not overlapping:
        return f"{name_a} beats {name_b}: the 95% intervals of {names} do not overlap."
    return (f"{name_a} vs {name_b} is not settled: their 95% intervals overlap on "
            f"{', '.join(overlapping)}.")


def replace_section(doc, heading, new_section):
    """Replace everything from the heading line to the next heading of the same level (or the end)."""
    start = doc.find(heading + "\n")
    if start < 0:
        raise ValueError(f"Heading {heading!r} not found")
    level = heading.split(" ")[0] + " "
    nxt = doc.find("\n" + level, start + len(heading))
    end = len(doc) if nxt < 0 else nxt + 1
    return doc[:start] + new_section + doc[end:]


def season_overlap_days(cutoff, horizon_days, months):
    """Days of the outcome window [cutoff, cutoff + horizon) that fall in the given calendar months."""
    days = pd.date_range(cutoff, periods=horizon_days, freq="D")
    return int(days.month.isin(months).sum())


def _ci(m, key, pct=False):
    value = m[key] if key in ("roc_auc", "pr_auc") else next(
        t["precision"] for t in m["top"] if top_key(t["fraction"]) == key)
    iv = m["bootstrap"]["intervals"][key]
    f = "{:.1%}" if pct else "{:.3f}"
    return f"{f.format(value)} ({f.format(iv['low'])}-{f.format(iv['high'])})"


def side_by_side_rows(validation, test):
    rows = []
    metrics = [("roc_auc", "ROC-AUC", False), ("pr_auc", "PR-AUC", False),
               ("precision_top_10pct", "Precision top 10%", True),
               ("precision_top_20pct", "Precision top 20%", True)]
    for name in ("rule", "logistic", "gbm"):
        v, t = validation["metrics"][name], test["metrics"][name]
        for key, label, pct in metrics:
            rows.append({"model": LABELS[name], "metric": label, "validation": _ci(v, key, pct),
                         "test": _ci(t, key, pct)})
        if t["brier"] is not None:
            rows.append({"model": LABELS[name], "metric": "Brier", "validation": f"{v['brier']:.3f}",
                         "test": f"{t['brier']:.3f}"})
            rows.append({"model": LABELS[name], "metric": "Mean predicted",
                         "validation": f"{v['mean_predicted']:.1%}", "test": f"{t['mean_predicted']:.1%}"})
    return rows


def season_rows(churn_table, churn_cfg, horizon):
    train = {f"{c:%Y-%m-%d}" for c in training_cutoffs(churn_cfg)}
    rows = []
    for r in churn_table.itertuples():
        c = pd.Timestamp(r.cutoff)
        role = "test" if r.cutoff == churn_cfg["test_cutoff"] else ("training" if r.cutoff in train else "not used")
        rows.append({"cutoff": r.cutoff, "role": role,
                     "window": f"{c:%Y-%m-%d} to {c + pd.Timedelta(days=horizon - 1):%Y-%m-%d}",
                     "season_days": season_overlap_days(c, horizon, SEASON_MONTHS),
                     "active": int(r.active), "churned": int(r.churned),
                     "churn_rate": float(r.churned / r.active)})
    return rows


def build_report(validation, test, churn_table, churn_cfg):
    horizon = churn_cfg["churn_horizon_days"]
    base = test["metrics"]["gbm"]
    seasons = season_rows(churn_table, churn_cfg, horizon)
    training_season_days = [r["season_days"] for r in seasons if r["role"] == "training"]
    gap = {n: {"mean_predicted": test["metrics"][n]["mean_predicted"],
               "gap_points": (test["metrics"][n]["mean_predicted"] - base["base_rate"]) * 100}
           for n in ("logistic", "gbm")}
    return {
        "test": test,
        "validation_generated_at": validation["generated_at"],
        "base": {"customers": base["customers"], "churners": base["churners"], "churn_rate": base["base_rate"]},
        "rates": validation["base_rates"],
        "gap": {n: {**g, "gap_points": f"{g['gap_points']:.1f}"} for n, g in gap.items()},
        "comparison": {
            "gbm_vs_rule": comparison_sentence("Gradient boosting", test["metrics"]["gbm"],
                                               "the recency rule", test["metrics"]["rule"]),
            "gbm_vs_logistic": comparison_sentence("Gradient boosting", test["metrics"]["gbm"],
                                                   "logistic regression", test["metrics"]["logistic"]),
            "logistic_vs_rule": comparison_sentence("Logistic regression", test["metrics"]["logistic"],
                                                    "the recency rule", test["metrics"]["rule"]),
        },
        "season": {"rows": seasons, "max_training_days": max(training_season_days),
                   "test_days": next(r["season_days"] for r in seasons if r["role"] == "test")},
    }


def render_section(report, test, paths):
    pct = "pct"
    rates = report["rates"]
    base_rows = [
        {"set": "Final training (all 6 cutoffs)", **rates["final_training"]},
        {"set": "Validation (2011-05-01)", **rates["validation"]},
        {"set": f"Test ({test['test_snapshot']['cutoff']})", **report["base"]},
    ]
    summary = [summary_row(n, test["metrics"][n]) for n in ("rule", "logistic", "gbm")]
    count_rows = [{"model": r["label"], "k10": r["top10_k"], "c10": _count(r["top10_churners"]),
                   "r10": r["top10_recall"], "k20": r["top20_k"], "c20": _count(r["top20_churners"]),
                   "r20": r["top20_recall"]} for r in summary]
    calib_rows = [{"model": {"logistic": "Logistic", "gbm": "Boosting"}[n], "bin": f"{r['lower']:.0%}-{r['upper']:.0%}",
                   "customers": r["customers"], "pred": r["mean_predicted"], "obs": r["observed_churn_rate"]}
                  for n in ("logistic", "gbm") for r in test["metrics"][n]["calibration"]]
    tables = {
        "base_rates": markdown_table(base_rows, [("set", "Set"), ("customers", "Rows"), ("churners", "Churned"),
                                                 ("churn_rate", "Churn rate", pct)]),
        "side_by_side": markdown_table(side_by_side_rows(report["_validation"], test), [
            ("model", "Model"), ("metric", "Metric"), ("validation", "Validation 2011-05-01 (95% CI)"),
            ("test", f"Test {test['test_snapshot']['cutoff']} (95% CI)")]),
        "counts": markdown_table(count_rows, [
            ("model", "Model"), ("k10", "Top 10%: customers"), ("c10", "churners"), ("r10", "recall", pct),
            ("k20", "Top 20%: customers"), ("c20", "churners"), ("r20", "recall", pct)]),
        "calibration": markdown_table(calib_rows, [
            ("model", "Model"), ("bin", "Predicted bin"), ("customers", "Customers"),
            ("pred", "Mean predicted", pct), ("obs", "Observed churn", pct)]),
        "season": markdown_table(report["season"]["rows"], [
            ("cutoff", "Cutoff"), ("role", "Role"), ("window", "Outcome window"),
            ("season_days", "Days in Sep-Nov"), ("active", "Active"), ("churned", "Churned"),
            ("churn_rate", "Churn rate", pct)]),
    }
    template = (REPO_ROOT / paths["c4_test_doc_template"]).read_text()
    return render_template(template, report, tables), summary


def main():
    data_cfg, churn_cfg = load_config("data"), load_config("churn")
    paths = data_cfg["paths"]
    test = json.loads((REPO_ROOT / paths["c4_test_metrics"]).read_text())
    validation = json.loads((REPO_ROOT / paths["c4_validation_metrics"]).read_text())
    churn_table = pd.read_csv(REPO_ROOT / paths["c2_churn_by_cutoff"], dtype={"cutoff": str})

    report = build_report(validation, test, churn_table, churn_cfg)
    report["_validation"] = validation
    report["outputs"] = {k: paths[k] for k in ("c4_test_metrics", "c4_test_summary", "c4_test_calibration_plot")}
    calibration_plot({LABELS[n]: test["metrics"][n]["calibration"] for n in ("logistic", "gbm")},
                     REPO_ROOT / paths["c4_test_calibration_plot"],
                     title=f"Calibration at the test cutoff {test['test_snapshot']['cutoff']}")
    section, summary = render_section(report, test, paths)

    doc_path = REPO_ROOT / paths["c4_doc_output"]
    doc_path.write_text(replace_section(doc_path.read_text(), SECTION, section))
    pd.DataFrame(summary).to_csv(REPO_ROOT / paths["c4_test_summary"], index=False, float_format="%.4f")
    save_report({k: v for k, v in report.items() if k not in ("_validation", "test")},
                REPO_ROOT / paths["c4_test_writeup"])
    print(f"Test section written to {paths['c4_doc_output']}")


if __name__ == "__main__":
    main()
