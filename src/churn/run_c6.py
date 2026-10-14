"""Milestone C6: findings report, dashboard data and README, rendered from saved results.

Run with:  python -m churn.run_c6      (after run_c5)

Like run_c5, it loads the frozen gradient boosting model with its sha256 check only to predict at
the test cutoff, and refuses unless the model and test snapshot are exactly those of the one-shot
test and the scores, rescaling and central campaign equal the saved C5 results. Nothing is
fitted, tuned or changed, and churn.run_c4_test is not run. New in C6 (saved before use):
  - predicted vs actual churn for the central-case campaign list vs everyone (after the fact),
  - results/c6_test_scores.csv for the dashboard (scores, value, key features, labelled outcome),
  - churn by segment on the training cutoffs, near-best campaign sizes, value concentration.
docs/REPORT.md and README.md are filled from templates in docs/templates/; never edit them by hand.
"""

import hashlib
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from churn.business import customer_value, rescale_to_mean
from churn.config import REPO_ROOT, load_config
from churn.dashboard import campaign_estimate
from churn.findings import near_best_sizes, predicted_vs_actual, segment_churn, value_concentration
from churn.metrics import top_key
from churn.models import feature_matrix, load_models, load_snapshots
from churn.plots import predicted_vs_actual_plot
from churn.report import markdown_table, render_template, save_report
from churn.run_c1 import git_state
from churn.run_c5 import check_frozen
from churn.snapshots import training_cutoffs

GROUP_LABELS = {"everyone": "Everyone active", "chosen": "Central-case list", "not_chosen": "Not on the list"}
COHORT_MONTHS = (1, 3, 6, 12)


def _json(key, paths):
    return json.loads((REPO_ROOT / paths[key]).read_text())


def cohort_medians(rates, last_month, months, skip_first):
    """Median retention at month k over cohorts whose month k is complete (before the partly
    covered last data month), leaving out the first, left-censored cohort."""
    cohorts = pd.PeriodIndex(rates.index, freq="M")
    last = pd.Period(last_month, freq="M")
    out = {}
    for k in months:
        keep = [(c + k < last) and not (skip_first and i == 0) for i, c in enumerate(cohorts)]
        values = rates.loc[keep, str(k)].dropna()
        out[f"m{k}"] = {"median": float(values.median()), "cohorts": int(len(values)),
                        "min": float(values.min()), "max": float(values.max())}
    return out


# --- new numbers --------------------------------------------------------------------------------

def compute(paths, churn_cfg, business_cfg, report_cfg):
    test_cutoff = churn_cfg["test_cutoff"]
    test_metrics, manifest = _json("c4_test_metrics", paths), _json("c4_final_models", paths)
    c5 = _json("c5_business", paths)
    snap_path = REPO_ROOT / paths["snapshots_dir"] / f"snapshot_{test_cutoff}.parquet"
    check_frozen(test_metrics, manifest, hashlib.sha256(snap_path.read_bytes()).hexdigest(), test_cutoff)
    gbm = load_models(REPO_ROOT / paths["models_dir"], {"gbm": manifest["models"]["gbm"]})["gbm"]
    test_df = pd.read_parquet(snap_path)
    y = test_df["churned"].to_numpy()

    p_model = gbm.predict_proba(feature_matrix(test_df))[:, 1]
    if not np.isclose(p_model.mean(), test_metrics["metrics"]["gbm"]["mean_predicted"], atol=1e-12):
        raise ValueError("Probabilities differ from the one-shot test: the model or data changed")
    rescaled = rescale_to_mean(p_model, c5["reference"]["churn_rate"])
    if not np.isclose(rescaled["shift"], c5["rescale"]["shift_log_odds"], atol=1e-9):
        raise ValueError("The rescaling differs from results/c5_business.json")
    p_rescaled = rescaled["probabilities"]
    value = customer_value(test_df["net_revenue_365d"])

    m, s, c = (business_cfg[k]["central"] for k in ("margin", "save_rate", "offer_cost_gbp"))
    central = campaign_estimate(p_rescaled, value, m, s, c)
    saved = c5["central"]["b"]
    if central["best_size"] != saved["best_size"] or not np.isclose(central["total_benefit"], saved["total_benefit"]):
        raise ValueError("The central campaign differs from results/c5_business.json")
    chosen = central["chosen"]
    pva = predicted_vs_actual(p_model, p_rescaled, y, chosen)
    if int(pva.set_index("group").loc["chosen", "churned"]) != saved["chosen_churned"]:
        raise ValueError("The chosen list's churners differ from results/c5_business.json")

    benefit = p_rescaled * s * m * value - c
    rank = pd.Series(-benefit).rank(method="first").astype(int).to_numpy()
    scores = pd.DataFrame({"customer_id": test_df["customer_id"], "p_model": p_model, "p_rescaled": p_rescaled,
                           "value": value, "expected_benefit_central": benefit, "rank_central": rank,
                           "in_central_list": chosen})
    for f in [*report_cfg["profile_features"], "single_purchase_day_365d"]:
        scores[f] = test_df[f].to_numpy()
    scores["churned_after_the_fact"] = y
    scores = scores.sort_values("rank_central").reset_index(drop=True)

    train_df = load_snapshots(REPO_ROOT / paths["snapshots_dir"], training_cutoffs(churn_cfg), test_cutoff)
    segments = segment_churn(train_df, report_cfg["segments"])
    return {"scores": scores, "pva": pva, "segments": segments, "training_rows": len(train_df),
            "near_best": near_best_sizes(central["curve"], report_cfg["flat_optimum_share"]),
            "concentration": value_concentration(value, report_cfg["value_concentration_shares"]),
            "test_metrics": test_metrics, "manifest": manifest}


# --- doc context --------------------------------------------------------------------------------

def _gbp(x):
    return f"{'-' if x < 0 else ''}£{abs(x):,.0f}"


def _ci(metric, key, pct=False):
    iv = metric["bootstrap"]["intervals"][key]
    value = metric[key] if key in ("roc_auc", "pr_auc") else next(
        t["precision"] for t in metric["top"] if top_key(t["fraction"]) == key)
    f = "{:.1%}" if pct else "{:.3f}"
    return f"{f.format(value)} ({f.format(iv['low'])} to {f.format(iv['high'])})"


def _top(metric, fraction):
    t = next(t for t in metric["top"] if np.isclose(t["fraction"], fraction))
    n = t["churners_in_top"]
    count = f"{n:,.0f}" if float(n).is_integer() else f"{n:,.1f}"
    return f"{t['precision']:.1%} ({count} of {t['k']:,})"


def build_context(c6, paths, churn_cfg, business_cfg, report_cfg):
    c1, c2 = _json("cleaning_report", paths), _json("c2_report", paths)
    val, test, writeup = _json("c4_validation_metrics", paths), c6["test_metrics"], _json("c4_test_writeup", paths)
    drivers, business = _json("c5_drivers", paths), _json("c5_business", paths)
    vm, tm = val["metrics"], test["metrics"]
    names = {"rule": "Recency rule", "logistic": "Logistic regression", "gbm": "Gradient boosting"}
    cb = business["central"]["b"]
    pva = c6["pva"].set_index("group")
    rates = pd.read_csv(REPO_ROOT / paths["c2_retention_rates"], index_col="cohort")
    cohorts = cohort_medians(rates, c2["data"]["last_month"], COHORT_MONTHS, skip_first=True)
    seasons = writeup["season"]["rows"]
    train_rates = [r["churn_rate"] for r in seasons if r["role"] == "training"]
    near = c6["near_best"]
    imp = drivers["importance"]
    steps = {s["step"]: s for s in c1["steps"]}
    statements = [s for s in drivers["statements"] if s["is_top_driver"]]
    sens = {r["horizon_days"]: r for r in c2["sensitivity"]["rows"]}
    conc = {round(r["top_share_of_customers"] * 100): r for r in c6["concentration"]}
    first_cohort = next(r for r in c2["cohorts"]["rows"] if r["cohort"] == c2["cohorts"]["first"])

    t = {
        "raw_rows": f"{c1['raw']['rows']:,}", "clean_rows": f"{c1['final']['rows']:,}",
        "customers": f"{c2['data']['customers']:,}",
        "first_date": c1["final"]["first_date"][:10], "last_date": c1["final"]["last_date"][:10],
        "overlap_rows": f"{c1['overlap']['later_rows_identical_to_earlier_sheet']:,}",
        "missing_id_rows": f"{steps['drop_missing_customer_id']['rows_removed']:,}",
        "missing_id_value": _gbp(steps["drop_missing_customer_id"]["net_value_removed"]),
        "same_day_invoices": f"{c2['purchase_events']['same_day']['purchase_invoices']:,}",
        "return_only": f"{c2['return_only_customers']['customers']:,}",
        "first_cohort": f"{c2['cohorts']['first_cohort_size']:,}",
        "dec09_m1": f"{first_cohort['m1']:.1%}", "dec09_m12": f"{first_cohort['m12']:.1%}",
        "train_churn_min": f"{min(train_rates):.1%}", "train_churn_max": f"{max(train_rates):.1%}",
        "sens60": f"{sens[60]['churn_rate']:.1%}", "sens90": f"{sens[90]['churn_rate']:.1%}",
        "sens120": f"{sens[120]['churn_rate']:.1%}", "sens_cutoff": c2["sensitivity"]["cutoff"],
        **{f"cohort_m{k}": f"{cohorts[f'm{k}']['median']:.1%}" for k in COHORT_MONTHS},
        **{f"cohort_m{k}_n": str(cohorts[f"m{k}"]["cohorts"]) for k in COHORT_MONTHS},
        "test_customers": f"{tm['gbm']['customers']:,}", "test_churners": f"{tm['gbm']['churners']:,}",
        "test_rate": f"{tm['gbm']['base_rate']:.1%}",
        "val_rate": f"{writeup['rates']['validation']['churn_rate']:.1%}",
        "train_rate": f"{writeup['rates']['final_training']['churn_rate']:.1%}",
        "gbm_pr": _ci(tm["gbm"], "pr_auc"), "rule_pr": _ci(tm["rule"], "pr_auc"), "log_pr": _ci(tm["logistic"], "pr_auc"),
        "gbm_p10": _top(tm["gbm"], 0.10), "rule_p10": _top(tm["rule"], 0.10),
        "gbm_mean_p": f"{tm['gbm']['mean_predicted']:.1%}", "log_mean_p": f"{tm['logistic']['mean_predicted']:.1%}",
        "gbm_gap": writeup["gap"]["gbm"]["gap_points"], "log_gap": writeup["gap"]["logistic"]["gap_points"],
        "cmp_gbm_rule": writeup["comparison"]["gbm_vs_rule"],
        "cmp_gbm_log": writeup["comparison"]["gbm_vs_logistic"],
        "cmp_log_rule": writeup["comparison"]["logistic_vs_rule"],
        "rule_flag_days": str(test["selected"]["rule"]["threshold_days"]),
        "season_test_days": str(writeup["season"]["test_days"]),
        "season_train_days": str(writeup["season"]["max_training_days"]),
        "top_drivers": ", ".join(f"`{f}`" for f in drivers["top_drivers"]),
        "rank45": f"`{imp[3]['feature']}` {imp[3]['mean']:.4f} vs `{imp[4]['feature']}` {imp[4]['mean']:.4f}",
        "training_rows": f"{drivers['training_rows']:,}",
        "value_total": _gbp(business["value"]["total"]), "value_median": _gbp(business["value"]["median"]),
        "risk_a": _gbp(business["revenue_at_risk"][0]["revenue_at_risk"]),
        "risk_b": _gbp(business["revenue_at_risk"][1]["revenue_at_risk"]),
        "risk_share_b": f"{business['revenue_at_risk'][1]['share_of_value']:.1%}",
        "value_churners": _gbp(business["after_the_fact"]["value_of_churners"]),
        "value_churners_share": f"{business['after_the_fact']['value_share_of_churners']:.1%}",
        "ref_cutoff": business["reference"]["cutoff"], "ref_rate": f"{business['reference']['churn_rate']:.1%}",
        "median_aov": _gbp(business["training_median_order_value"]),
        "margin": f"{business_cfg['margin']['central']:.0%}", "save": f"{business_cfg['save_rate']['central']:.0%}",
        "cost": _gbp(business_cfg["offer_cost_gbp"]["central"]),
        "threshold": _gbp(cb["value_threshold"]),
        "central_size": f"{cb['best_size']:,}", "central_benefit": _gbp(cb["total_benefit"]),
        "central_cost": _gbp(cb["offer_cost_total"]), "central_saved": _gbp(cb["expected_saved_revenue"]),
        "central_chosen_value": _gbp(cb["chosen_value"]),
        "central_a_size": f"{business['central']['a']['best_size']:,}",
        "central_a_benefit": _gbp(business["central"]["a"]["total_benefit"]),
        "sizes_min": f"{business['best_sizes']['b']['min']:,}", "sizes_max": f"{business['best_sizes']['b']['max']:,}",
        "near_share": f"{near['share']:.0%}", "near_min": f"{near['min_size']:,}", "near_max": f"{near['max_size']:,}",
        "near_floor": _gbp(near["share"] * near["best"]),
        "chosen_n": f"{int(pva.loc['chosen', 'customers']):,}",
        "chosen_churned": f"{int(pva.loc['chosen', 'churned']):,}",
        "chosen_pred": f"{pva.loc['chosen', 'mean_p_rescaled']:.1%}",
        "chosen_obs": f"{pva.loc['chosen', 'observed_churn_rate']:.1%}",
        "chosen_gap": f"{pva.loc['chosen', 'gap_points_rescaled']:.1f}",
        "chosen_ratio": f"{pva.loc['chosen', 'ratio_rescaled']:.2f}",
        "all_pred": f"{pva.loc['everyone', 'mean_p_rescaled']:.1%}",
        "all_obs": f"{pva.loc['everyone', 'observed_churn_rate']:.1%}",
        "all_gap": f"{pva.loc['everyone', 'gap_points_rescaled']:.1f}",
        "all_ratio": f"{pva.loc['everyone', 'ratio_rescaled']:.2f}",
        "rest_pred": f"{pva.loc['not_chosen', 'mean_p_rescaled']:.1%}",
        "rest_obs": f"{pva.loc['not_chosen', 'observed_churn_rate']:.1%}",
        "same_size_by_p": f"{cb['same_size_by_p_churned']:,}",
        "hindsight": _gbp(cb["hindsight_benefit"]),
        "top1_share": f"{conc[1]['value_share']:.1%}", "top1_n": f"{conc[1]['customers']:,}",
        "top10_share": f"{conc[10]['value_share']:.1%}", "top10_n": f"{conc[10]['customers']:,}",
        "model_sha": c6["manifest"]["models"]["gbm"]["sha256"],
        "raw_sha": c1["raw_sha256"],
        "test_commit": test["git"]["commit"],
    }

    pct = "pct"
    step_rows = [{"step": s["rule"], "removed": s["rows_removed"], "after": s["rows_after"],
                  "value": _gbp(s["net_value_removed"])} for s in c1["steps"]]
    cutoff_rows = [{"cutoff": r["cutoff"], "role": r["role"], "active": r["active"], "churned": r["churned"],
                    "rate": r["churn_rate"], "season": r["season_days"]} for r in seasons]
    cohort_rows = [{"month": f"Month {k}", "median": cohorts[f"m{k}"]["median"],
                    "range": f"{cohorts[f'm{k}']['min']:.0%} to {cohorts[f'm{k}']['max']:.0%}",
                    "n": cohorts[f"m{k}"]["cohorts"]} for k in COHORT_MONTHS]
    base_rows = [{"set": "Training (6 cutoffs, 2010-12-01 to 2011-05-01)", **writeup["rates"]["final_training"]},
                 {"set": "Validation (2011-05-01)", **writeup["rates"]["validation"]},
                 {"set": f"Test ({churn_cfg['test_cutoff']}), touched once", **writeup["base"]}]
    perf_rows = [{"model": names[n], "val_pr": _ci(vm[n], "pr_auc"), "test_pr": _ci(tm[n], "pr_auc"),
                  "test_roc": _ci(tm[n], "roc_auc"), "p10": _top(tm[n], 0.10), "p20": _top(tm[n], 0.20),
                  "mean_p": "" if tm[n]["mean_predicted"] is None else f"{tm[n]['mean_predicted']:.1%}"}
                 for n in ("rule", "logistic", "gbm")]
    driver_rows = [{"rank": r["rank"], "feature": f"`{r['feature']}`", "mean": f"{r['mean']:.4f}",
                    "range": f"{r['low']:.4f} to {r['high']:.4f}", "definition": r["definition"]} for r in imp[:6]]
    statement_rows = [{"group": f"Customers who {s['group']}", "rank": s["driver_rank"], "rows": s["rows_in"],
                       "in": s["churn_in"], "out": s["churn_out"], "ratio": f"{s['ratio']:.2f}",
                       "after": "yes" if s["added_after_ranking"] else ""} for s in statements]
    seg_rows = [{"segment": r.title, "group": r.group, "rows": r.rows, "rate": r.churn_rate,
                 "range": f"{r.churn_rate_min:.0%} to {r.churn_rate_max:.0%}"} for r in c6["segments"].itertuples()]
    risk_rows = [{"label": f"({r['version']}) {r['label']}", "mean": r["mean_probability"],
                  "risk": _gbp(r["revenue_at_risk"]), "share": r["share_of_value"]} for r in business["revenue_at_risk"]]

    def bold(row, text):
        return f"**{text}**" if row["is_central"] else text

    scen_rows = [{k: bold(r, v) for k, v in {
        "margin": f"{r['margin']:.0%}", "save": f"{r['save_rate']:.0%}", "cost": _gbp(r["offer_cost"]),
        "threshold": _gbp(r["value_threshold"]), "size": f"{r['best_size']:,}",
        "benefit": _gbp(r["total_benefit"]), "spend": _gbp(r["offer_cost_total"])}.items()}
        for r in business["scenarios"] if r["version"] == "b"]
    pva_rows = [{"group": GROUP_LABELS[g], "customers": int(r.customers), "pred_a": r.mean_p_model,
                 "pred_b": r.mean_p_rescaled, "obs": r.observed_churn_rate, "churned": int(r.churned),
                 "gap": f"{r.gap_points_rescaled:+.1f}", "ratio": f"{r.ratio_rescaled:.2f}"}
                for g, r in pva.iterrows()]
    readme_rows = [{"model": names[n], "pr": _ci(tm[n], "pr_auc"), "roc": _ci(tm[n], "roc_auc"),
                    "p10": _top(tm[n], 0.10), "p20": _top(tm[n], 0.20)} for n in ("rule", "logistic", "gbm")]
    assumption_rows = [
        {"a": "Gross margin", "c": f"{business_cfg['margin']['central']:.0%}",
         "s": ", ".join(f"{x:.0%}" for x in business_cfg["margin"]["scenarios"]), "src": "Not in the data (plan default)"},
        {"a": "Save rate (share of contacted would-be churners kept)", "c": f"{business_cfg['save_rate']['central']:.0%}",
         "s": ", ".join(f"{x:.0%}" for x in business_cfg["save_rate"]["scenarios"]), "src": "Not observable here (plan default)"},
        {"a": "Offer cost per contacted customer", "c": _gbp(business_cfg["offer_cost_gbp"]["central"]),
         "s": ", ".join(_gbp(x) for x in business_cfg["offer_cost_gbp"]["scenarios"]),
         "src": "Author's judgement, adopted 2026-10-14; not checked with a retailer"},
        {"a": "Customer value", "c": "12-month net revenue, floored at £0", "s": "", "src": "`net_revenue_365d`"},
    ]
    tables = {
        "steps": markdown_table(step_rows, [("step", "Cleaning step (in order)"), ("removed", "Rows removed"),
                                            ("after", "Rows left"), ("value", "Net value removed")]),
        "cutoffs": markdown_table(cutoff_rows, [("cutoff", "Cutoff"), ("role", "Role"), ("active", "Active customers"),
                                                ("churned", "Churned"), ("rate", "Churn rate", pct),
                                                ("season", "Days of the 90-day window in Sep-Nov")]),
        "cohorts": markdown_table(cohort_rows, [("month", "Months after first purchase"),
                                                ("median", "Median share buying again", pct),
                                                ("range", "Lowest to highest cohort"), ("n", "Cohorts")]),
        "base_rates": markdown_table(base_rows, [("set", "Set"), ("customers", "Customers"),
                                                 ("churners", "Churned"), ("churn_rate", "Churn rate", pct)]),
        "performance": markdown_table(perf_rows, [
            ("model", "Model"), ("val_pr", "Validation PR-AUC (95% CI)"), ("test_pr", "Test PR-AUC (95% CI)"),
            ("test_roc", "Test ROC-AUC (95% CI)"), ("p10", "Test precision, top 10%"),
            ("p20", "Test precision, top 20%"), ("mean_p", "Test mean predicted")]),
        "drivers": markdown_table(driver_rows, [("rank", "Rank"), ("feature", "Feature"), ("mean", "Drop in PR-AUC"),
                                                ("range", "Spread over shuffles"), ("definition", "Meaning")]),
        "statements": markdown_table(statement_rows, [
            ("group", "Group (training rows)"), ("rank", "Driver rank"), ("rows", "Rows in group"),
            ("in", "Churn in group", pct), ("out", "Churn, everyone else", pct), ("ratio", "Ratio"),
            ("after", "Added after the ranking")]),
        "segments": markdown_table(seg_rows, [("segment", "Segment"), ("group", "Group"), ("rows", "Customer-cutoff rows"),
                                              ("rate", "Churn rate", pct), ("range", "Lowest to highest cutoff")]),
        "assumptions": markdown_table(assumption_rows, [("a", "Assumption"), ("c", "Central"), ("s", "Scenarios"),
                                                        ("src", "Source")]),
        "risk": markdown_table(risk_rows, [("label", "Probabilities"), ("mean", "Mean p(churn)", pct),
                                           ("risk", "Revenue at risk (estimate)"), ("share", "Share of 12-month revenue", pct)]),
        "scenarios": markdown_table(scen_rows, [
            ("margin", "Margin"), ("save", "Save rate"), ("cost", "Offer cost"), ("threshold", "Contact if p x value above"),
            ("size", "Best size"), ("benefit", "Expected net benefit"), ("spend", "Offer spend")]),
        "pva": markdown_table(pva_rows, [
            ("group", "Group"), ("customers", "Customers"), ("pred_a", "Mean predicted (model)", pct),
            ("pred_b", "Mean predicted (rescaled)", pct), ("obs", "Churned (after the fact)", pct),
            ("churned", "Churners"), ("gap", "Gap, points (rescaled)"), ("ratio", "Predicted / actual")]),
        "readme_results": markdown_table(readme_rows, [
            ("model", "Model"), ("pr", "PR-AUC (95% CI)"), ("roc", "ROC-AUC (95% CI)"),
            ("p10", "Precision, top 10%"), ("p20", "Precision, top 20%")]),
    }
    return {"t": t}, tables


# --- main ---------------------------------------------------------------------------------------

def main():
    data_cfg, churn_cfg = load_config("data"), load_config("churn")
    business_cfg, report_cfg = load_config("business"), load_config("report")
    paths = data_cfg["paths"]
    if not (REPO_ROOT / paths["c5_business"]).exists():
        sys.exit("results/c5_business.json is missing: run python -m churn.run_c5 first")
    c6 = compute(paths, churn_cfg, business_cfg, report_cfg)

    # Save every new number before it is used anywhere.
    c6["scores"].to_csv(REPO_ROOT / paths["c6_test_scores"], index=False)
    c6["pva"].to_csv(REPO_ROOT / paths["c6_predicted_vs_actual"], index=False, float_format="%.6f")
    c6["segments"].to_csv(REPO_ROOT / paths["c6_segments_training"], index=False, float_format="%.6f")
    stamp = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git": git_state(),
             "raw_sha256": data_cfg["raw"]["sha256"], "test_cutoff": churn_cfg["test_cutoff"],
             "model": {"name": "gbm", **c6["manifest"]["models"]["gbm"]}}
    save_report({**stamp, "after_the_fact_note": "observed churn columns use test outcomes, unknown on the decision day",
                 "predicted_vs_actual": c6["pva"].to_dict(orient="records"),
                 "near_best_sizes_central_b": c6["near_best"], "value_concentration": c6["concentration"],
                 "segments_training": c6["segments"].to_dict(orient="records"),
                 "segments_training_rows": c6["training_rows"],
                 "test_scores_file": paths["c6_test_scores"]}, REPO_ROOT / paths["c6_report"])
    predicted_vs_actual_plot(c6["pva"], GROUP_LABELS, REPO_ROOT / paths["c6_predicted_vs_actual_plot"],
                             title=f"Predicted vs actual churn, central-case list on {churn_cfg['test_cutoff']}")

    ctx, tables = build_context(c6, paths, churn_cfg, business_cfg, report_cfg)
    ctx["stamp"] = stamp
    for key in ("c6_report", "c6_readme"):
        text = render_template((REPO_ROOT / paths[f"{key}_template"]).read_text(), ctx, tables)
        (REPO_ROOT / paths[f"{key}_output"]).write_text(text)
    print(f"Central-case list: {ctx['t']['chosen_n']} customers, predicted {ctx['t']['chosen_pred']}, "
          f"actual {ctx['t']['chosen_obs']} (after the fact)")
    print(f"Wrote {paths['c6_report_output']}, {paths['c6_readme_output']}, {paths['c6_test_scores']}")


if __name__ == "__main__":
    main()
