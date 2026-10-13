"""Milestone C1: load both sheets, clean, save the clean parquet and the cleaning report, render the doc.

Run with:  python -m churn.run_c1
"""

import subprocess
from datetime import datetime, timezone

from churn.clean import clean_transactions
from churn.config import REPO_ROOT, load_config
from churn.load import load_raw
from churn.report import render_template, save_report


def git_state():
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "uncommitted_changes": bool(git("status", "--porcelain"))}


def main():
    data_cfg = load_config("data")
    cleaning_cfg = load_config("cleaning")
    raw_cfg, paths = data_cfg["raw"], data_cfg["paths"]
    sheets = raw_cfg["sheets"]

    raw = load_raw(REPO_ROOT / raw_cfg["path"], raw_cfg["sha256"], sheets, REPO_ROOT / paths["cache_dir"])

    rules = {
        "cancellation_prefix": cleaning_cfg["cancellation_prefix"],
        "non_product_stock_codes": [e["code"] for e in cleaning_cfg["non_product_stock_codes"]],
    }
    clean, steps, summary = clean_transactions(raw, sheets, rules)

    out = REPO_ROOT / paths["clean_transactions"]
    out.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(out, index=False)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "raw_file": raw_cfg["path"],
        "raw_sha256": raw_cfg["sha256"],
        "rules": rules,
        "raw": summary["raw"],
        "overlap": summary["overlap"],
        "steps": steps,
        "returns": summary["returns"],
        "final": summary["final"],
        "clean_file": paths["clean_transactions"],
    }
    save_report(report, REPO_ROOT / paths["cleaning_report"])

    template = (REPO_ROOT / paths["doc_template"]).read_text()
    (REPO_ROOT / paths["doc_output"]).write_text(render_template(template, report))

    print(f"Raw rows: {report['raw']['rows']:,}  ->  clean rows: {report['final']['rows']:,}")
    print(f"Report: {paths['cleaning_report']}   Doc: {paths['doc_output']}")


if __name__ == "__main__":
    main()
