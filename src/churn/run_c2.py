"""Milestone C2: purchase events, customer-month table, cohort retention and descriptive churn.

Run with:  python -m churn.run_c2      (needs data/processed/transactions_clean.parquet from run_c1)
"""

import sys
from datetime import datetime, timezone

import pandas as pd

from churn.churn_rates import churn_at_cutoff, churn_table, valid_monthly_cutoffs
from churn.cohorts import (
    customer_months,
    first_purchase_months,
    retention_counts,
    retention_rates,
    return_only_customers,
)
from churn.config import REPO_ROOT, load_config
from churn.events import cancellation_impact, purchase_events
from churn.plots import retention_heatmap
from churn.report import markdown_table, render_template, save_report
from churn.run_c1 import git_state

SUMMARY_MONTHS = (1, 3, 6, 12)


def _month_label(ts):
    return ts.strftime("%Y-%m")


def _retention_csv(table, path):
    out = table.copy()
    out.index = [_month_label(i) for i in out.index]
    out.index.name = "cohort"
    out.to_csv(path, float_format="%.4f")


def cohort_rows(counts, rates, flagged, partial_month):
    rows = []
    for cohort in counts.index:
        row = {"cohort": _month_label(cohort), "customers": int(counts.loc[cohort, 0]), "note": ""}
        for k in SUMMARY_MONTHS:
            v = rates.loc[cohort, k] if k in rates.columns else float("nan")
            row[f"m{k}"] = None if pd.isna(v) else float(v)
            if pd.notna(v) and cohort + pd.DateOffset(months=k) == partial_month:
                row["note"] = f"month {k} is the partly covered last month"
        if cohort == flagged:
            row["note"] = "includes customers who first bought before the data starts"
        rows.append(row)
    return rows


def main():
    paths = load_config("data")["paths"]
    cfg = load_config("churn")
    clean_path = REPO_ROOT / paths["clean_transactions"]
    if not clean_path.exists():
        sys.exit(f"{paths['clean_transactions']} not found. Run: python -m churn.run_c1")
    df = pd.read_parquet(clean_path)
    raw_cfg = load_config("data")["raw"]

    rule = cfg["purchase_event_rule"]
    events = purchase_events(df, rule)
    impact = cancellation_impact(df, rule)

    # Customers with only returns: counted, then left out of cohorts (they have no purchase event).
    return_only = return_only_customers(df, events)
    ro_lines = df[df["customer_id"].isin(return_only)]
    if not cfg["exclude_return_only_customers"]:
        sys.exit("exclude_return_only_customers must be true: such customers have no cohort month")

    cm = customer_months(df, events)
    cm_path = REPO_ROOT / paths["customer_months"]
    cm_path.parent.mkdir(parents=True, exist_ok=True)
    cm.to_parquet(cm_path, index=False)

    first_ts, last_ts = df["invoice_date"].min(), df["invoice_date"].max()
    last_month = last_ts.to_period("M").to_timestamp()
    last_month_partial = last_ts.normalize() < last_month + pd.offsets.MonthEnd(0)
    first_month = first_ts.to_period("M").to_timestamp()

    counts = retention_counts(events, last_month)
    rates = retention_rates(counts)
    sizes = counts[0].astype(int)
    _retention_csv(counts, REPO_ROOT / paths["c2_retention_counts"])
    _retention_csv(rates, REPO_ROOT / paths["c2_retention_rates"])
    retention_heatmap(rates, sizes, REPO_ROOT / paths["c2_heatmap"], flagged_cohort=first_month,
                      partial_month=last_month if last_month_partial else None)
    cohorts = cohort_rows(counts, rates, first_month, last_month if last_month_partial else None)
    pd.DataFrame(cohorts).to_csv(REPO_ROOT / paths["c2_cohort_summary"], index=False, float_format="%.4f")

    lookback, horizon = cfg["active_lookback_days"], cfg["churn_horizon_days"]
    cutoffs = valid_monthly_cutoffs(first_ts, last_ts, lookback, horizon)
    churn = churn_table(events, cutoffs, lookback, horizon)
    churn["is_test_cutoff"] = churn["cutoff"] == cfg["test_cutoff"]
    churn.to_csv(REPO_ROOT / paths["c2_churn_by_cutoff"], index=False, float_format="%.4f")

    s_cutoff = pd.Timestamp(cfg["sensitivity_cutoff"])
    sensitivity = []
    for h in sorted({*cfg["sensitivity_horizons_days"], horizon}):
        if s_cutoff not in valid_monthly_cutoffs(first_ts, last_ts, lookback, h):
            sys.exit(f"Sensitivity cutoff {cfg['sensitivity_cutoff']} does not fit a {h}-day window")
        sensitivity.append({"horizon_days": h, **churn_at_cutoff(events, s_cutoff, lookback, h)})
    pd.DataFrame(sensitivity).to_csv(REPO_ROOT / paths["c2_churn_sensitivity"], index=False,
                                     float_format="%.4f")

    later_sizes = sizes.drop(first_month)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "input_file": paths["clean_transactions"],
        "raw_sha256": raw_cfg["sha256"],
        "definitions": {k: v for k, v in cfg.items() if not k.endswith("_rationale")},
        "data": {
            "first_date": str(first_ts), "last_date": str(last_ts),
            "last_month": _month_label(last_month), "last_month_partial": bool(last_month_partial),
            "customers": int(df["customer_id"].nunique()),
        },
        "purchase_events": {**impact, "events_kept": len(events),
                            "customers_with_purchase": int(events["customer_id"].nunique())},
        "return_only_customers": {
            "customers": len(return_only),
            "lines": len(ro_lines),
            "net_value": round(float(ro_lines["line_value"].sum()), 2),
        },
        "customer_months": {"rows": len(cm), "file": paths["customer_months"],
                            "months_with_purchase": int((cm["purchase_invoices"] > 0).sum())},
        "cohorts": {
            "count": len(counts),
            "first": _month_label(counts.index[0]), "last": _month_label(counts.index[-1]),
            "customers": int(sizes.sum()),
            "first_cohort_size": int(sizes[first_month]),
            "later_cohorts_median_size": float(later_sizes.median()),
            "later_cohorts_max_size": int(later_sizes.max()),
            "rows": cohorts,
        },
        "churn_by_cutoff": {
            "lookback_days": lookback, "horizon_days": horizon,
            "first_cutoff": churn["cutoff"].iloc[0], "last_cutoff": churn["cutoff"].iloc[-1],
            "count": len(churn),
            "min_rate": float(churn["churn_rate"].min()), "max_rate": float(churn["churn_rate"].max()),
            "rows": churn.to_dict("records"),
        },
        "sensitivity": {"cutoff": cfg["sensitivity_cutoff"], "rows": sensitivity},
        "outputs": {k: paths[k] for k in ["c2_cohort_summary", "c2_retention_counts", "c2_retention_rates",
                                          "c2_churn_by_cutoff", "c2_churn_sensitivity", "c2_heatmap"]},
    }
    save_report(report, REPO_ROOT / paths["c2_report"])

    pct = "pct"
    tables = {
        "cohorts": markdown_table(cohorts, [
            ("cohort", "Cohort"), ("customers", "Customers"), ("m1", "Month 1", pct),
            ("m3", "Month 3", pct), ("m6", "Month 6", pct), ("m12", "Month 12", pct), ("note", "Note")]),
        "churn": markdown_table(
            [{**r, "note": "test cutoff (listed only)" if r["is_test_cutoff"] else ""}
             for r in report["churn_by_cutoff"]["rows"]],
            [("cutoff", "Cutoff"), ("active", "Active"), ("churned", "Churned"),
             ("retained", "Retained"), ("churn_rate", "Churn rate", pct), ("note", "Note")]),
        "sensitivity": markdown_table(sensitivity, [
            ("horizon_days", "Window (days)"), ("active", "Active"), ("churned", "Churned"),
            ("churn_rate", "Churn rate", pct)]),
        "largest_cancelled": markdown_table(
            [{**r, "customer_id": str(r["customer_id"])} for r in impact["largest_same_day"]], [
            ("customer_id", "Customer"), ("purchase_invoice", "Purchase invoice"),
            ("cancel_invoice", "Cancellation invoice"), ("value", "Value (£)")]),
    }
    template = (REPO_ROOT / paths["c2_doc_template"]).read_text()
    (REPO_ROOT / paths["c2_doc_output"]).write_text(render_template(template, report, tables))

    print(f"Purchase events: {len(events):,}  cohorts: {len(counts)}  cutoffs: {len(churn)}")
    print(f"Report: {paths['c2_report']}   Doc: {paths['c2_doc_output']}")


if __name__ == "__main__":
    main()
