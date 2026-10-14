# Progress

## C1: Repo setup, loading, cleaning (done 2026-10-13, committed c8a14e4)

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

## C3: Snapshot builder, features, labels (done 2026-10-13, committed cfc6b97)

Built:
- `config/features.yaml`: frequency windows (90/180/365), trend window (90), stock-code
  normalisation (upper + strip), peak months (Nov, Dec inside the lookback), gap fill value for
  one-purchase-day customers (365), home country (UK), censored tenure month (2009-12). Each has a
  rationale; tests check that and that every window fits the lookback.
- `src/churn/snapshots.py`: `FEATURES` (21 features, each with a one-line definition),
  `snapshot_features` (one row per active customer, only lines strictly before the cutoff),
  `churn_labels` (no purchase event in [cutoff, cutoff + 90d)), `build_snapshot`, and
  `leakage_differences` (rebuilds features from data cut at the cutoff, with purchase events and
  same-day cancellations recomputed, and lists any feature that changes).
- `src/churn/run_c3.py`: builds and saves the 6 training snapshots and the 2011-09-01 test
  snapshot; reports only row count and churn rate for the test cutoff; summary statistics pooled
  over training cutoffs only. Renders `docs/C3_snapshots_and_features.md` from
  `docs/templates/C3_snapshots_and_features.md`.
- `config/data.yaml`: C3 output paths.

Outputs: `results/c3_features.json`, `results/c3_feature_summary_training.csv` (committable);
`data/processed/snapshots/snapshot_<cutoff>.parquet` x 7 (gitignored). Run: about 7 s, 0.7 GB peak.

Tests: 96 unit tests (`pytest`; 25 new in `tests/test_snapshots.py`, 2 in `tests/test_config.py`),
11 real-data checks (`pytest -m realdata`; 3 new in `tests/test_real_data_c3.py`: leakage at
2010-12-01 and 2011-05-01, rows and churn equal C2 at 2011-05-01). All pass. Snapshot rows and
churned counts equal `results/c2_churn_by_cutoff.csv` at all 7 cutoffs.

How to run:
```bash
.venv/bin/python -m churn.run_c3
.venv/bin/python -m pytest && .venv/bin/python -m pytest -m realdata
```

Open items: see NOTES.md (TO VERIFY).

## C4 part A: Models and validation (done 2026-10-13, reviewed and committed)

Built:
- `config/model.yaml`: seed, validation cutoff 2011-05-01, tuning gap rule, selection metric
  (PR-AUC), rule thresholds, logistic preprocessing options and per-feature transforms, logistic
  settings, boosting fixed settings and 12-setting grid, top fractions, calibration bins,
  bootstrap samples. Each has a rationale (tested).
- `src/churn/metrics.py`: ROC-AUC, PR-AUC, precision/recall in the top 10%/20% with counts (ties
  share places), Brier, 10-bin calibration table, mean predicted, base rate, 1,000-sample
  bootstrap 95% intervals for AUC, PR-AUC and precision@10/20%.
- `src/churn/models.py`: validation split (tuning cutoffs 2010-12-01 and 2011-01-01),
  `load_snapshots` (refuses the test cutoff), recency rule, logistic preprocessor (log1p, signed
  log or clipping, standardising; fitted on training rows only), boosting, tuning, final refit,
  model save/load with sha256 check.
- `src/churn/run_c4.py`: tunes, reports validation metrics, refits on all 6 training cutoffs,
  saves models and the frozen manifest, renders `docs/C4_models.md` from
  `docs/templates/C4_models.md` (test section left marked NOT RUN YET).
- `src/churn/run_c4_test.py`: the one-shot test run for part B. Tested on fixtures only; NOT run.
- `src/churn/plots.py`: calibration plot.
- `requirements.txt`: scikit-learn 1.9.1 and its dependencies pinned.

Outputs (committable): `results/c4_validation_metrics.json`, `results/c4_validation_summary.csv`,
`results/c4_validation_tuning.csv`, `results/c4_validation_calibration.csv`,
`results/c4_final_models.json` (frozen manifest), `results/figures/c4_validation_calibration.png`.
Gitignored: `data/processed/models/{rule,logistic,gbm}.joblib`. Run: about 17 s, 0.26 GB peak.

Review change (2026-10-13, before any test run): the rule now ranks by `recency_days` (validation
PR-AUC 0.794, was 0.732 as a yes/no score); `recency_days > 120` is kept as a reported yes/no flag.

Chosen settings: rule flag `recency_days > 120`; logistic `log_signed_log`; boosting
`learning_rate=0.03, max_leaf_nodes=8, min_samples_leaf=100` (all in docs/C4_models.md).

Tests: 143 unit tests (`pytest`; new: tests/test_metrics.py, tests/test_models.py,
tests/test_run_c4_test.py, one plot test), 12 real-data checks (`pytest -m realdata`; new:
tests/test_real_data_c4.py). All pass. The test cutoff (2011-09-01) was not read or scored.

How to run:
```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m churn.run_c4
.venv/bin/python -m pytest && .venv/bin/python -m pytest -m realdata
```

Open items: see NOTES.md (TO VERIFY, C4 entries).

## C4 part B: one-shot test (run once 2026-10-13 at 472fd3d, committed a7cacc0); write-up 2026-10-14

The test JSON was rendered into docs/C4_models.md by `python -m churn.render_c4_test` (new
template docs/templates/C4_test_results.md; validation part of the doc unchanged). It adds
validation vs test side by side, base rates, counts, test calibration
(results/figures/c4_test_calibration.png) and the seasonal TO VERIFY hypothesis. The test was not
rerun. New files: results/c4_test_summary.csv, results/c4_test_writeup.json.

## C5: Drivers and business layer (done 2026-10-14, uncommitted, awaiting review)

Built:
- `config/drivers.yaml`: permutation settings (PR-AUC, 30 shuffles), partial dependence (top 4,
  20 points, 5th-95th percentile), 11 plain-language statements (3 added after the ranking,
  flagged). `config/business.yaml`: model, decision day, customer value, margin, save rate, offer
  cost (TO VERIFY), rescale method (logit shift) and reference cutoff (2011-06-01). All with rationales.
- `src/churn/drivers.py`: permutation importance table, brute-force partial dependence on the
  probability scale, group churn statements (with an optional "among" subgroup).
- `src/churn/business.py`: customer value floor, revenue at risk, expected benefit, campaign curve
  and best size, rescaling to a mean, the "known on the day" reference-rate check, scenario grid.
- `src/churn/run_c5.py`: guards (frozen model/snapshot hashes, recomputed scores equal the saved
  test results), drivers on the test cutoff, statements on the 6 training cutoffs, business layer
  as of 2011-09-01, saved tables, 3 figures, renders docs/C5_drivers_and_business.md.
- `src/churn/plots.py`: importance, partial dependence and campaign plots.

Outputs (committable): results/c5_drivers.json, c5_permutation_importance.csv,
c5_partial_dependence.csv, c5_driver_statements.csv, c5_business.json, c5_campaign_scenarios.csv,
c5_campaign_curve.csv, results/figures/c5_{permutation_importance,partial_dependence,campaign_net_benefit}.png.
Run: about 11 s.

Headline results (all in docs/C5_drivers_and_business.md): top drivers orders_365d,
net_revenue_365d, tenure_days, recency_days (5th, latest_gap_days, tied with 4th). Revenue at risk
GBP 2.32m (a) / 2.28m (b), estimates. Central case (b): contact 1,567 customers, expected net
benefit GBP 22,621 (estimate); after the fact 562 of them churned (35.9%).

Tests: 179 unit tests (`pytest`; new: test_business.py with hand-worked benefit and rescaling
fixtures, test_drivers.py, test_render_c4_test.py, test_run_c5.py, 3 plot tests), 14 real-data
checks (`pytest -m realdata`; new: tests/test_real_data_c5.py). All pass.

How to run:
```bash
.venv/bin/python -m churn.render_c4_test
.venv/bin/python -m churn.run_c5
.venv/bin/python -m pytest && .venv/bin/python -m pytest -m realdata
```

Not done: nothing reran churn.run_c4_test, run_c4 or refit any model (model files unchanged, sha256
checked). Open items: NOTES.md (TO VERIFY, C4 part B and C5 entries).

## Next: C6 (after review and commit of C5)
Report, dashboard, README (PROJECT_PLAN.md Section 8).
