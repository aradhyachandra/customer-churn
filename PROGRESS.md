# Progress

## C1: Repo setup, loading, cleaning (done 2026-10-13, awaiting review, not committed)

Built:
- Skeleton: `src/churn/`, `tests/`, `config/`, `docs/`, `data/raw/` and `data/processed/` and
  `results/` gitignored. `pyproject.toml` (src layout, pytest config), `requirements.txt` pinned.
- `src/churn/load.py`: checks the raw file's sha256 (stops with a clear error on mismatch), reads
  both sheets once with python-calamine, caches them as `data/processed/raw_sheets_<sha12>.parquet`
  and reuses the cache. Customer ID is a nullable integer (`Int64`); Invoice and StockCode are strings.
- `src/churn/clean.py`: sheet-overlap measurement and the Section 2.1 rules in a fixed order, with
  rows, net value and sales value removed at each step (overall and per sheet). Returns are kept and
  flagged (`is_return`); `line_value` = quantity x price.
- `config/data.yaml` (paths, sheet names, sha256) and `config/cleaning.yaml` (cancellation prefix,
  25 non-product codes, each with a reason).
- `src/churn/report.py` + `src/churn/run_c1.py`: saves `results/c1_cleaning_report.json` and renders
  `docs/C1_data_and_cleaning.md` from `docs/templates/C1_data_and_cleaning.md`, so every number in
  the doc comes from the report file.

Outputs (gitignored): `data/processed/transactions_clean.parquet`, `results/c1_cleaning_report.json`.

Tests: 36 unit tests on small hand-made tables (`pytest`), 5 real-file checks kept separate
(`pytest -m realdata`). All pass.

How to run:
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
.venv/bin/python -m churn.run_c1
.venv/bin/python -m pytest
```

Open items: see NOTES.md (TO VERIFY).

## C2: Cohorts, retention, descriptive churn (done 2026-10-13, committed)

Built:
- `config/churn.yaml`: purchase-event rule, return-only exclusion, 365-day lookback, 90-day
  horizon, 60/120-day sensitivity at 2011-05-01, test cutoff (listed only), cutoff convention
  (00:00 of the date; lookback `[c-365d, c)`, outcome `[c, c+h)`), cutoff fit rule. Each has a
  rationale; tests check that.
- `src/churn/events.py`: purchase events. A purchase invoice reversed in full by a cancellation
  invoice of the same customer on the same day (same stock codes, quantities, prices; one-to-one
  matching) is not a purchase event. Also measures later-day mirror cancellations (kept).
- `src/churn/cohorts.py`: customer-month table, return-only customers, first-purchase cohorts,
  retention counts and rates (cells past the data are empty, not zero).
- `src/churn/churn_rates.py`: monthly cutoffs whose full windows fit the data, active/churned
  counts per cutoff.
- `src/churn/plots.py`: retention heatmap (Dec 2009 cohort starred, partial Dec 2011 hatched).
- `src/churn/report.py`: now also fills `{{table:name}}` and `{{key|pct}}` placeholders.
- `src/churn/run_c2.py` renders `docs/C2_cohorts_and_churn.md` from
  `docs/templates/C2_cohorts_and_churn.md`.

Outputs (committable now): `results/c2_cohorts_and_churn.json`, `results/c2_cohort_summary.csv`,
`results/c2_retention_{counts,rates}.csv`, `results/c2_churn_by_cutoff.csv`,
`results/c2_churn_sensitivity.csv`, `results/figures/c2_retention_heatmap.png`, plus
`results/c1_cleaning_report.json` (C1 rerun). Gitignored: `data/processed/customer_months.parquet`.

Repo changes: `.gitignore` no longer ignores `results/`; only `results/**/*.parquet|pkl|joblib|zip`.
`matplotlib==3.11.2` (and its dependencies) pinned in `requirements.txt`.

Tests: 69 unit tests (`pytest`), 8 real-data checks (`pytest -m realdata`). All pass.

How to run:
```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m churn.run_c1
.venv/bin/python -m churn.run_c2
.venv/bin/python -m pytest
```

Decided after review: training cutoffs run monthly 2010-12-01..2011-05-01 (config/churn.yaml).
Open items: see NOTES.md (TO VERIFY).

## Next: C3 (snapshot builder + features + labels)
