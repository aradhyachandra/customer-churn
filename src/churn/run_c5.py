"""Milestone C5: drivers and business layer, as "what we would have done on 2011-09-01".

Run with:  python -m churn.run_c5      (after C4 part B: needs results/c4_test_metrics.json)

The frozen gradient boosting model is loaded with its sha256 check and only used to predict; the
test snapshot must be the one the one-shot test scored (same sha256). Nothing is fitted, tuned or
changed, and churn.run_c4_test is not run. Test outcomes are used for permutation importance
(held-out data, nothing changed afterwards) and for clearly labelled after-the-fact checks only.
Driver statements use the training cutoffs only.
"""

import hashlib
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

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
from churn.config import REPO_ROOT, load_config
from churn.drivers import group_churn, partial_dependence_table, permutation_importance_table
from churn.models import feature_matrix, load_models, load_snapshots
from churn.plots import campaign_plot, importance_plot, partial_dependence_plot
from churn.report import markdown_table, render_template, save_report
from churn.run_c1 import git_state
from churn.snapshots import FEATURES, training_cutoffs

VERSIONS = {"a": "Model probabilities as they are", "b": "Rescaled to the latest known churn rate"}


def check_frozen(test_metrics, manifest, snapshot_sha256, test_cutoff):
    """Refuse unless the models and snapshot are exactly those the one-shot test used."""
    if test_metrics["test_snapshot"]["cutoff"] != test_cutoff:
        raise ValueError(f"The one-shot test scored cutoff {test_metrics['test_snapshot']['cutoff']}, "
                         f"not {test_cutoff}")
    if test_metrics["test_snapshot"]["sha256"] != snapshot_sha256:
        raise ValueError("The test snapshot differs from the one the one-shot test scored")
    if test_metrics["models"] != manifest["models"] or test_metrics["training_cutoffs"] != manifest["training_cutoffs"]:
        raise ValueError("The frozen models differ from the models the one-shot test used")


def _day(ts):
    return f"{pd.Timestamp(ts):%Y-%m-%d}"


# --- drivers ------------------------------------------------------------------------------------

def run_drivers(gbm, test_df, train_df, cfg, seed, test_pr_auc):
    X, y = feature_matrix(test_df), test_df["churned"].to_numpy()
    p = gbm.predict_proba(X)[:, 1]
    baseline = float(average_precision_score(y, p))
    if not np.isclose(baseline, test_pr_auc, atol=1e-12):
        raise ValueError("Scores differ from the one-shot test: the model or data changed")
    perm = cfg["permutation"]
    importance = permutation_importance_table(gbm, X, y, perm["scoring"], perm["n_repeats"], seed,
                                              tuple(perm["spread_percentiles"]))
    importance["definition"] = importance["feature"].map(FEATURES)
    top = list(importance["feature"][:cfg["partial_dependence"]["top_n"]])
    pdc = cfg["partial_dependence"]
    pdp = partial_dependence_table(gbm, X, top, pdc["grid_resolution"], pdc["percentiles"])
    rank = dict(zip(importance["feature"], importance["rank"]))
    statements = []
    for s in cfg["statements"]:
        g = group_churn(train_df, s)
        among = s.get("among")
        condition = f"{s['feature']} {s['op']} {s['value']}" + (
            f" among {among['feature']} {among['op']} {among['value']}" if among else "")
        statements.append({"feature": s["feature"], "condition": condition,
                           "added_after_ranking": bool(s.get("added_after_ranking")),
                           "group": s["group"], "driver_rank": int(rank[s["feature"]]),
                           "is_top_driver": s["feature"] in top, **g})
    return {"baseline_pr_auc": baseline, "importance": importance, "top": top, "pdp": pdp,
            "statements": statements}


# --- business layer -----------------------------------------------------------------------------

def scenario_row(scenario, version, p, value, y):
    b = expected_benefit(p, value, scenario["save_rate"], scenario["margin"], scenario["offer_cost"])
    best = best_campaign(b)
    chosen = chosen_customers(b, best["size"])
    n = best["size"]
    hindsight = float(np.sum(y[chosen] * scenario["save_rate"] * scenario["margin"] * value[chosen])
                      - n * scenario["offer_cost"])
    return {
        **scenario, "version": version,
        "value_threshold": scenario["offer_cost"] / (scenario["save_rate"] * scenario["margin"]),
        "best_size": n, "total_benefit": best["total_benefit"], "offer_cost_total": n * scenario["offer_cost"],
        "expected_saved_revenue": float(np.sum(p[chosen] * scenario["save_rate"] * value[chosen])),
        "chosen_value": float(value[chosen].sum()),
        "chosen_mean_p": float(p[chosen].mean()) if n else None,
        # After the fact only (outcomes were unknown on the decision day):
        "chosen_churned": int(y[chosen].sum()),
        "chosen_churn_rate": float(y[chosen].mean()) if n else None,
        "hindsight_benefit": hindsight,
        "same_size_by_p_churned": int(y[chosen_customers(p, n)].sum()),   # ranking by p alone
    }


def run_business(gbm, test_df, cfg, churn_cfg, churn_table, test_metrics):
    X, y = feature_matrix(test_df), test_df["churned"].to_numpy()
    p_raw = gbm.predict_proba(X)[:, 1]
    if not np.isclose(p_raw.mean(), test_metrics["metrics"]["gbm"]["mean_predicted"], atol=1e-12):
        raise ValueError("Probabilities differ from the one-shot test: the model or data changed")
    value = customer_value(test_df["net_revenue_365d"])
    ref = reference_churn_rate(churn_table, cfg["rescale_reference_cutoff"], churn_cfg["test_cutoff"],
                               churn_cfg["churn_horizon_days"])
    rescaled = rescale_to_mean(p_raw, ref["churn_rate"])
    probs = {"a": p_raw, "b": rescaled["probabilities"]}

    risk = [{"version": v, "label": VERSIONS[v], "mean_probability": float(p.mean()),
             "revenue_at_risk": revenue_at_risk(p, value), "share_of_value": revenue_at_risk(p, value) / value.sum()}
            for v, p in probs.items()]
    grid = scenario_grid(cfg)
    rows = [scenario_row(s, v, p, value, y) for v, p in probs.items() for s in grid]

    curves = {}
    for v, p in probs.items():
        all_curves = np.array([campaign_curve(expected_benefit(p, value, s["save_rate"], s["margin"],
                                                               s["offer_cost"])) for s in grid])
        central = next(i for i, s in enumerate(grid) if s["is_central"])
        curves[v] = {"central": all_curves[central], "low": all_curves.min(axis=0), "high": all_curves.max(axis=0)}

    return {
        "customers": len(test_df),
        "value": {"total": float(value.sum()), "median": float(np.median(value)),
                  "floored_customers": int((test_df["net_revenue_365d"] < 0).sum()),
                  "floored_amount": float(-test_df.loc[test_df["net_revenue_365d"] < 0, "net_revenue_365d"].sum())},
        "reference": ref,
        "rescale": {"shift_log_odds": rescaled["shift"], "mean_before": float(p_raw.mean()),
                    "mean_after": float(rescaled["probabilities"].mean())},
        "revenue_at_risk": risk,
        "after_the_fact": {"churners": int(y.sum()), "churn_rate": float(y.mean()),
                           "value_of_churners": float(value[y == 1].sum()),
                           "value_share_of_churners": float(value[y == 1].sum() / value.sum())},
        "scenarios": rows,
        "curves": curves,
    }


# --- main ---------------------------------------------------------------------------------------

def main():
    data_cfg, churn_cfg, model_cfg = load_config("data"), load_config("churn"), load_config("model")
    drivers_cfg, business_cfg = load_config("drivers"), load_config("business")
    paths, test_cutoff = data_cfg["paths"], churn_cfg["test_cutoff"]
    test_path = REPO_ROOT / paths["c4_test_metrics"]
    if not test_path.exists():
        sys.exit("results/c4_test_metrics.json is missing: C5 needs the one-shot test (C4 part B)")
    test_metrics = json.loads(test_path.read_text())
    manifest = json.loads((REPO_ROOT / paths["c4_final_models"]).read_text())
    snap_path = REPO_ROOT / paths["snapshots_dir"] / f"snapshot_{test_cutoff}.parquet"
    check_frozen(test_metrics, manifest, hashlib.sha256(snap_path.read_bytes()).hexdigest(), test_cutoff)
    gbm = load_models(REPO_ROOT / paths["models_dir"], {"gbm": manifest["models"]["gbm"]})["gbm"]
    test_df = pd.read_parquet(snap_path)
    train_df = load_snapshots(REPO_ROOT / paths["snapshots_dir"], training_cutoffs(churn_cfg), test_cutoff)
    churn_table = pd.read_csv(REPO_ROOT / paths["c2_churn_by_cutoff"], dtype={"cutoff": str})

    drivers = run_drivers(gbm, test_df, train_df, drivers_cfg, model_cfg["seed"],
                          test_metrics["metrics"]["gbm"]["pr_auc"])
    business = run_business(gbm, test_df, business_cfg, churn_cfg, churn_table, test_metrics)
    git = git_state()
    stamp = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git": git,
             "raw_sha256": data_cfg["raw"]["sha256"], "test_cutoff": test_cutoff,
             "test_snapshot_sha256": test_metrics["test_snapshot"]["sha256"],
             "model": {"name": "gbm", **manifest["models"]["gbm"]}}

    # Saved tables.
    drivers["importance"].to_csv(REPO_ROOT / paths["c5_permutation_importance"], index=False, float_format="%.6f")
    drivers["pdp"].to_csv(REPO_ROOT / paths["c5_partial_dependence"], index=False, float_format="%.6f")
    pd.DataFrame(drivers["statements"]).to_csv(REPO_ROOT / paths["c5_driver_statements"], index=False,
                                               float_format="%.4f")
    pd.DataFrame(business["scenarios"]).to_csv(REPO_ROOT / paths["c5_campaign_scenarios"], index=False,
                                               float_format="%.4f")
    c = business["curves"]
    pd.DataFrame({"customers_contacted": np.arange(len(c["b"]["central"])),
                  "central_b": c["b"]["central"], "low_b": c["b"]["low"], "high_b": c["b"]["high"],
                  "central_a": c["a"]["central"], "low_a": c["a"]["low"], "high_a": c["a"]["high"]}
                 ).to_csv(REPO_ROOT / paths["c5_campaign_curve"], index=False, float_format="%.2f")

    train_aov = float(train_df["avg_order_value_365d"].median())
    drivers_report = {
        **stamp,
        "training_cutoffs": [_day(x) for x in training_cutoffs(churn_cfg)],
        "training_rows": len(train_df),
        "permutation": {**drivers_cfg["permutation"], "seed": model_cfg["seed"],
                        "baseline_pr_auc": drivers["baseline_pr_auc"]},
        "partial_dependence": drivers_cfg["partial_dependence"],
        "top_drivers": drivers["top"],
        "importance": drivers["importance"].to_dict(orient="records"),
        "partial_dependence_values": drivers["pdp"].to_dict(orient="records"),
        "statements": drivers["statements"],
    }
    save_report(drivers_report, REPO_ROOT / paths["c5_drivers"])
    central = {v: next(r for r in business["scenarios"] if r["is_central"] and r["version"] == v) for v in "ab"}
    business_report = {
        **stamp,
        "assumptions": {k: business_cfg[k] for k in ("margin", "save_rate", "offer_cost_gbp", "customer_value",
                                                     "rescale_method", "rescale_reference_cutoff")},
        "training_median_order_value": train_aov,
        **{k: v for k, v in business.items() if k != "curves"},
        "central": central,
        "best_sizes": {v: {"min": min(r["best_size"] for r in business["scenarios"] if r["version"] == v),
                           "max": max(r["best_size"] for r in business["scenarios"] if r["version"] == v)}
                       for v in "ab"},
    }
    save_report(business_report, REPO_ROOT / paths["c5_business"])

    # Figures.
    importance_plot(drivers["importance"], REPO_ROOT / paths["c5_importance_plot"],
                    title=f"Permutation importance, gradient boosting, test cutoff {test_cutoff}",
                    highlight=len(drivers["top"]),
                    xlabel=f"Drop in PR-AUC when the feature is shuffled (mean and 2.5-97.5% of "
                           f"{drivers_cfg['permutation']['n_repeats']} shuffles)")
    partial_dependence_plot(drivers["pdp"], drivers["top"], REPO_ROOT / paths["c5_pdp_plot"],
                            title=f"Partial dependence of the top {len(drivers['top'])} drivers, test cutoff {test_cutoff}")
    campaign_plot(c["b"]["central"], c["b"]["low"], c["b"]["high"], central["b"]["best_size"],
                  REPO_ROOT / paths["c5_campaign_plot"],
                  title=f"Expected net benefit by campaign size on {test_cutoff} (estimate)",
                  central_label="Central case (margin 30%, save rate 10%, offer £"
                                f"{business_cfg['offer_cost_gbp']['central']}), rescaled probabilities",
                  band_label="Range over all 27 scenarios")

    render_doc(drivers_report, business_report, paths, drivers_cfg)
    print(f"Top drivers: {', '.join(drivers['top'])}")
    for v in "ab":
        r = central[v]
        print(f"  ({v}) central: contact {r['best_size']:,}, expected net benefit £{r['total_benefit']:,.0f}")
    print(f"Doc: {paths['c5_doc_output']}")


# --- doc ----------------------------------------------------------------------------------------

def _gbp(x):
    return f"£{x:,.0f}"


def render_doc(drivers, business, paths, drivers_cfg):
    pct = "pct"
    imp_rows = [{"rank": r["rank"], "feature": f"`{r['feature']}`", "mean": f"{r['mean']:.4f}",
                 "range": f"{r['low']:.4f} to {r['high']:.4f}", "definition": r["definition"]}
                for r in drivers["importance"]]
    st_rows = [{"group": f"Customers who {s['group']}", "condition": f"`{s['condition']}`",
                "rank": s["driver_rank"], "share": s["share_in"], "rows_in": s["rows_in"],
                "churn_in": s["churn_in"], "churn_out": s["churn_out"],
                "ratio": "" if s["ratio"] is None else f"{s['ratio']:.2f}",
                "range_in": f"{s['churn_in_min']:.0%}-{s['churn_in_max']:.0%}",
                "range_out": f"{s['churn_out_min']:.0%}-{s['churn_out_max']:.0%}"}
               for s in drivers["statements"]]
    sentences = []
    for s in drivers["statements"]:
        tag = (" **(top-4 driver)**" if s["is_top_driver"] else "") + (
            " *(statement added after the ranking)*" if s["added_after_ranking"] else "")
        sentences.append(f"- Customers who {s['group']} churned **{s['churn_in']:.1%}** of the time vs "
                         f"**{s['churn_out']:.1%}** for everyone else (ratio {s['ratio']:.2f}; "
                         f"{s['rows_in']:,} of {s['rows_in'] + s['rows_out']:,} customer-cutoff rows; "
                         f"{s['churn_in_min']:.0%}-{s['churn_in_max']:.0%} vs "
                         f"{s['churn_out_min']:.0%}-{s['churn_out_max']:.0%} across the 6 cutoffs).{tag}")
    uncovered = [f for f in drivers["top_drivers"] if f not in {s["feature"] for s in drivers["statements"]}]

    risk_rows = [{"label": f"({r['version']}) {r['label']}", "mean": r["mean_probability"],
                  "risk": _gbp(r["revenue_at_risk"]), "share": r["share_of_value"]}
                 for r in business["revenue_at_risk"]]

    def bold(row, text):
        return f"**{text}**" if row["is_central"] else text

    by_key = {}
    for r in business["scenarios"]:
        by_key.setdefault((r["margin"], r["save_rate"], r["offer_cost"]), {})[r["version"]] = r
    sc_rows = []
    for (m, s, c), d in by_key.items():
        a, b = d["a"], d["b"]
        sc_rows.append({k: bold(b, v) for k, v in {
            "margin": f"{m:.0%}", "save": f"{s:.0%}", "cost": _gbp(c), "threshold": _gbp(b["value_threshold"]),
            "size_a": f"{a['best_size']:,}", "benefit_a": _gbp(a["total_benefit"]),
            "size_b": f"{b['best_size']:,}", "benefit_b": _gbp(b["total_benefit"])}.items()})

    pdp_rows = []
    for f in drivers["top_drivers"]:
        pts = [r for r in drivers["partial_dependence_values"] if r["feature"] == f]
        lo, hi = pts[0], pts[-1]
        pdp_rows.append({"feature": f"`{f}`", "low": f"{lo['value']:,.0f}", "p_low": lo["mean_predicted"],
                         "high": f"{hi['value']:,.0f}", "p_high": hi["mean_predicted"]})
    imp = drivers["importance"]
    business["text"] = {
        "baseline": f"{drivers['permutation']['baseline_pr_auc']:.3f}",
        "seed": str(drivers["permutation"]["seed"]),
        "rank45": (f"`{imp[3]['feature']}` {imp[3]['mean']:.4f} vs `{imp[4]['feature']}` "
                   f"{imp[4]['mean']:.4f}"),
        "uncovered": ("Every top-4 driver has at least one statement above." if not uncovered else
                      "No statement covers: " + ", ".join(f"`{f}`" for f in uncovered) + "."),
        "top_list": ", ".join(f"`{f}`" for f in drivers["top_drivers"]),
        "threshold_central": _gbp(business["central"]["b"]["value_threshold"]),
        "median_aov": _gbp(business["training_median_order_value"]),
        "value_total": _gbp(business["value"]["total"]),
        "value_median": _gbp(business["value"]["median"]),
        "floored_amount": _gbp(business["value"]["floored_amount"]),
        "value_of_churners": _gbp(business["after_the_fact"]["value_of_churners"]),
        "shift": f"{business['rescale']['shift_log_odds']:.3f}",
        "central": {v: {"benefit": _gbp(r["total_benefit"]), "cost": _gbp(r["offer_cost_total"]),
                        "saved": _gbp(r["expected_saved_revenue"]), "chosen_value": _gbp(r["chosen_value"]),
                        "hindsight": _gbp(r["hindsight_benefit"])}
                    for v, r in business["central"].items()},
    }
    report = {"d": drivers, "b": business}
    tables = {
        "importance": markdown_table(imp_rows, [("rank", "Rank"), ("feature", "Feature"),
                                                ("mean", "Mean drop in PR-AUC"), ("range", "2.5-97.5% of shuffles"),
                                                ("definition", "Definition")]),
        "statements": markdown_table(st_rows, [
            ("group", "Group"), ("condition", "Condition"), ("rank", "Driver rank"), ("share", "Share of rows", pct),
            ("rows_in", "Rows in group"), ("churn_in", "Churn in group", pct), ("churn_out", "Churn, rest", pct),
            ("ratio", "Ratio"), ("range_in", "Group, per cutoff"), ("range_out", "Rest, per cutoff")]),
        "pdp": markdown_table(pdp_rows, [("feature", "Feature"), ("low", "Lowest grid value"),
                                         ("p_low", "Average predicted churn there", pct),
                                         ("high", "Highest grid value"),
                                         ("p_high", "Average predicted churn there", pct)]),
        "risk": markdown_table(risk_rows, [("label", "Probabilities"), ("mean", "Mean p(churn)", pct),
                                           ("risk", "Revenue at risk (estimate)"),
                                           ("share", "Share of 12-month revenue", pct)]),
        "scenarios": markdown_table(sc_rows, [
            ("margin", "Margin"), ("save", "Save rate"), ("cost", "Offer cost"),
            ("threshold", "Contact if p x value above"),
            ("size_a", "(a) best size"), ("benefit_a", "(a) net benefit"),
            ("size_b", "(b) best size"), ("benefit_b", "(b) net benefit")]),
    }
    template = (REPO_ROOT / paths["c5_doc_template"]).read_text()
    text = render_template(template, report, tables)
    text = text.replace("<!-- statements -->", "\n".join(sentences))
    (REPO_ROOT / paths["c5_doc_output"]).write_text(text)


if __name__ == "__main__":
    main()
