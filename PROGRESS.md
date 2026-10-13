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

## Next: C2 (customer-month tables, cohorts, retention curves, descriptive churn)
