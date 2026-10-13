Read CLAUDE.md and PROJECT_PLAN.md fully. Do milestone C1 only.

Scope: repo skeleton (src/churn/, tests/, config/, data/raw gitignored, docs/), a Python venv with
pinned requirements, a loader for data/raw/online_retail_II.xlsx (both sheets), the cleaning rules in
PROJECT_PLAN.md Section 2.1 with every rule and code list in config, a cleaning report (rows and
revenue removed at each step), the sheet-overlap check, and a sha256 of the raw file recorded in
config. Tests first, including small hand-made fixtures for each cleaning rule. Save the cleaned
data as a parquet file under data/processed/ (gitignored).

When done, update PROGRESS.md, write docs/C1_data_and_cleaning.md in plain language, and list what I
should review. Commit nothing until I say.
