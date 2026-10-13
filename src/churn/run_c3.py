"""Milestone C3: snapshot tables (features + churn label) for the training cutoffs and the test cutoff.

Run with:  python -m churn.run_c3      (needs data/processed/transactions_clean.parquet from run_c1)

The test cutoff's table is built and saved, but only its row count and churn rate are reported
(both already public in C2). Feature summaries use the training cutoffs only.
"""

import sys
from datetime import datetime, timezone

import pandas as pd

from churn.config import REPO_ROOT, load_config
from churn.load import load_raw
from churn.report import markdown_table, render_template, save_report
from churn.run_c1 import git_state
from churn.snapshots import (
    FEATURES,
    build_snapshot,
    history_inputs,
    normalise_stock_codes,
    peak_months_inside_lookback,
    stock_code_merges,
    training_cutoffs,
)

QUANTILES = {"p10": 0.10, "p25": 0.25, "median": 0.50, "p75": 0.75, "p90": 0.90}


def _day(ts):
    return ts.strftime("%Y-%m-%d")


def stock_code_report(raw, clean, method):
    """How many codes merge under the normalisation, in the raw file and in the clean data."""
    raw_codes = raw["stock_code"].dropna().astype(str)
    changed = raw_codes != normalise_stock_codes(raw_codes, method)
    raw_changed_lines = raw.loc[changed[changed].index]
    return {
        "method": method,
        "raw_distinct_codes": int(raw_codes.nunique()),
        "raw_codes_merged": stock_code_merges(raw_codes, method),
        "raw_lines_with_changed_code": int(changed.sum()),
        "raw_lines_with_changed_code_and_customer": int(raw_changed_lines["customer_id"].notna().sum()),
        "clean_distinct_codes": int(clean["stock_code"].nunique()),
        "clean_codes_merged": stock_code_merges(clean["stock_code"], method),
    }


def feature_summary(snapshots):
    """Pooled summary statistics per feature over the given (training) snapshot tables."""
    pooled = pd.concat(snapshots, ignore_index=True)
    rows = []
    for name in FEATURES:
        s = pooled[name]
        row = {"feature": name, "mean": float(s.mean()), "min": float(s.min())}
        row.update({k: float(s.quantile(q)) for k, q in QUANTILES.items()})
        row["max"] = float(s.max())
        rows.append(row)
    return rows


def main():
    paths = load_config("data")["paths"]
    raw_cfg = load_config("data")["raw"]
    churn_cfg, feat_cfg = load_config("churn"), load_config("features")
    clean_path = REPO_ROOT / paths["clean_transactions"]
    if not clean_path.exists():
        sys.exit(f"{paths['clean_transactions']} not found. Run: python -m churn.run_c1")
    df = pd.read_parquet(clean_path)

    raw = load_raw(REPO_ROOT / raw_cfg["path"], raw_cfg["sha256"], raw_cfg["sheets"],
                   REPO_ROOT / paths["cache_dir"])[["stock_code", "customer_id"]]
    codes = stock_code_report(raw, df, feat_cfg["stock_code_normalisation"])
    del raw

    events, cancels = history_inputs(df, churn_cfg["purchase_event_rule"])
    train = training_cutoffs(churn_cfg)
    test = pd.Timestamp(churn_cfg["test_cutoff"])
    if test in train or max(train) + pd.Timedelta(days=churn_cfg["churn_horizon_days"]) > test:
        sys.exit("Training outcome windows must end on or before the test cutoff")

    out_dir = REPO_ROOT / paths["snapshots_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows, train_tables, files = [], [], {}
    test_row = None
    for c in [*train, test]:
        snap = build_snapshot(df, events, cancels, c, churn_cfg, feat_cfg)
        if snap.isna().any().any():
            sys.exit(f"Snapshot {_day(c)} has missing values")
        path = out_dir / f"snapshot_{_day(c)}.parquet"
        snap.to_parquet(path, index=False)
        files[_day(c)] = str(path.relative_to(REPO_ROOT))
        row = {"cutoff": _day(c), "rows": len(snap), "churned": int(snap["churned"].sum()),
               "churn_rate": float(snap["churned"].mean())}
        if c == test:
            test_row = row  # row count and churn rate only: nothing about its features
            continue
        missing = snap[list(FEATURES)].isna().sum()
        peak = peak_months_inside_lookback(c, churn_cfg["active_lookback_days"], feat_cfg["peak_months"])
        row.update({
            "missing_values": int(missing.sum()),
            "missing_by_feature": {k: int(v) for k, v in missing.items()},
            "single_purchase_day_share": float(snap["single_purchase_day_365d"].mean()),
            "first_seen_first_data_month_share": float(snap["first_seen_first_data_month"].mean()),
            "peak_months_in_lookback": [m.strftime("%Y-%m") for m in peak],
        })
        train_rows.append(row)
        train_tables.append(snap)

    summary = feature_summary(train_tables)
    del train_tables
    pd.DataFrame(summary).to_csv(REPO_ROOT / paths["c3_feature_summary"], index=False, float_format="%.4f")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "input_file": paths["clean_transactions"],
        "raw_sha256": raw_cfg["sha256"],
        "definitions": {
            "churn": {k: v for k, v in churn_cfg.items() if not k.endswith("_rationale")},
            "features": {k: v for k, v in feat_cfg.items() if not k.endswith("_rationale")},
        },
        "features": [{"name": k, "definition": v} for k, v in FEATURES.items()],
        "feature_count": len(FEATURES),
        "stock_codes": codes,
        "purchase_events": {"events": len(events), "same_day_cancellation_invoices": len(cancels)},
        "training": {
            "cutoffs": [r["cutoff"] for r in train_rows],
            "count": len(train_rows),
            "rows": sum(r["rows"] for r in train_rows),
            "churned": sum(r["churned"] for r in train_rows),
            "missing_values": sum(r["missing_values"] for r in train_rows),
            "per_cutoff": train_rows,
            "feature_summary": summary,
        },
        "test": test_row,
        "files": files,
        "outputs": {k: paths[k] for k in ["c3_feature_summary", "snapshots_dir"]},
    }
    report["training"]["churn_rate"] = report["training"]["churned"] / report["training"]["rows"]
    save_report(report, REPO_ROOT / paths["c3_report"])

    pct = "pct"
    tables = {
        "features": markdown_table(report["features"], [("name", "Feature"), ("definition", "Definition")]),
        "cutoffs": markdown_table(
            [{**r, "peak": ", ".join(r["peak_months_in_lookback"])} for r in train_rows],
            [("cutoff", "Cutoff"), ("rows", "Customers"), ("churned", "Churned"),
             ("churn_rate", "Churn rate", pct), ("single_purchase_day_share", "One purchase day", pct),
             ("first_seen_first_data_month_share", "First seen Dec 2009", pct),
             ("missing_values", "Missing values"), ("peak", "Peak months in lookback")]),
        "summary": markdown_table(summary, [
            ("feature", "Feature"), ("mean", "Mean"), ("min", "Min"), ("p10", "P10"),
            ("median", "Median"), ("p90", "P90"), ("max", "Max")]),
    }
    template = (REPO_ROOT / paths["c3_doc_template"]).read_text()
    (REPO_ROOT / paths["c3_doc_output"]).write_text(render_template(template, report, tables))

    print(f"Snapshots: {len(files)}  training rows: {report['training']['rows']:,}  "
          f"test rows: {test_row['rows']:,}")
    print(f"Report: {paths['c3_report']}   Doc: {paths['c3_doc_output']}")


if __name__ == "__main__":
    main()
