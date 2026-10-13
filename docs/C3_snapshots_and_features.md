# C3: Snapshots, features and churn labels

*Generated from `results/c3_features.json` on 2026-10-13T06:17:50+00:00 (git 8fb7edabe9b3a7c0ffaeab499bb0ab7e4a688584,
uncommitted changes: True). Do not edit by hand; edit
`docs/templates/C3_snapshots_and_features.md` and rerun `python -m churn.run_c3`.*

## What this step does

For each cutoff date it takes a snapshot of every active customer (at least one purchase in the
365 days before the cutoff) and describes them with
21 plain numbers built only from what was known before that date: how recently and
how often they bought, how much they spent, whether that is slowing down, how long they have been
a customer, and so on. It then adds the answer the model will learn: did they buy again in the next
90 days (churned = 0) or not (churned = 1)?

Tables are saved for the 6 training cutoffs and for the test cutoff. Settings and
their reasons are in `config/churn.yaml` and `config/features.yaml`.

## How leakage is ruled out

A cutoff is midnight at the start of its date. Every feature uses only lines dated strictly before
it; the label uses only purchases from the cutoff to 90 days
later. Purchases follow the C2 rule (`drop_same_day_full_cancellations`): an order cancelled
in full on the same day is not a purchase, and its cancellation is not counted as a return. Both
fall on the same day, so they are always on the same side of a cutoff. Other returns count only
when dated before the cutoff.

The key test rebuilds every feature after deleting all data from the cutoff on (purchase events and
same-day cancellations are recomputed from the shortened data) and checks that every value is
identical. It runs on small hand-made tables at five cutoffs (`tests/test_snapshots.py`) and on the
real data at 2010-12-01 and 2011-05-01 (`pytest -m realdata`). A companion test deletes data the
features are allowed to use and checks the comparison does report differences, so the check is
not passing by accident.

## Features

| Feature | Definition |
|---|---|
| recency_days | Whole days from the last purchase day to the cutoff date (1 = bought the day before). |
| orders_90d | Purchase events in the 90 days before the cutoff. |
| orders_180d | Purchase events in the 180 days before the cutoff. |
| orders_365d | Purchase events in the 365 days before the cutoff. |
| net_revenue_90d | Net value (£) of all lines, purchases minus returns, in the 90 days before the cutoff. |
| net_revenue_180d | Net value (£) of all lines in the 180 days before the cutoff. |
| net_revenue_365d | Net value (£) of all lines in the 365 days before the cutoff (trailing 12-month revenue). |
| avg_order_value_365d | Value (£) of purchase events in the last 365 days divided by their number. |
| orders_trend_90d | Purchase events in the last 90 days minus those in the 90 days before them. |
| net_revenue_trend_90d | Net value (£) in the last 90 days minus net value in the 90 days before them. |
| single_purchase_day_365d | 1 if the customer bought on only one day in the last 365 days (gap features then use fill values). |
| avg_gap_days | Mean days between consecutive purchase days in the last 365 days (365 if only one purchase day). |
| latest_gap_days | Days between the last two purchase days in the last 365 days (365 if only one purchase day). |
| latest_gap_ratio | latest_gap_days / avg_gap_days (above 1: the last gap was longer than usual). |
| recency_gap_ratio | recency_days / avg_gap_days (above 1: already waiting longer than the usual gap). |
| tenure_days | Whole days from the first purchase day in the data to the cutoff date. |
| first_seen_first_data_month | 1 if the first purchase is in the first data month (Dec 2009): tenure is a lower bound. |
| distinct_products_365d | Distinct stock codes (upper-cased, stripped) in purchase events in the last 365 days. |
| return_rate_365d | Value returned / value bought in the last 365 days; same-day full cancellations count on neither side. Can exceed 1 when earlier purchases are returned. |
| is_uk | 1 if the country on the customer's latest line before the cutoff is the United Kingdom. |
| bought_in_peak_365d | 1 if the customer bought in the November or December inside the last 365 days. |

Every feature is a plain number (counts, days, £, ratios, 0/1 flags); there is no target encoding.

### Decisions behind the features

- **Distinct products.** Stock codes are upper-cased and stripped of spaces before counting, so
  codes like 85123A and 85123a count once. In the raw file this merges
  174 of 5,305 codes
  (3,472 lines), but
  0 of those lines have a Customer ID, so
  C1 had already removed them all: in the clean data 0 of
  4,635 codes merge. The rule stays in place as a safeguard.
- **Seasonality.** "Bought in the peak" means bought in a November or December inside the
  365-day lookback. Every cutoff is the first of a month, so
  the lookback is exactly the 12 calendar months before it and always contains one complete
  November and one complete December, all inside the data (see the table below). "Last year's
  Nov-Dec" would not work at 2010-12-01, where Nov 2009 is missing from the data.
- **Tenure.** Days since the first purchase in the data. For customers first seen in December 2009
  (the first data month) the real first purchase may be earlier, so their tenure is a lower bound;
  `first_seen_first_data_month` marks them so a model can tell them apart.
- **One purchase day.** With only one purchase day in the lookback there is no gap between orders.
  Instead of a blank, the gap features take 365
  days (no repeat seen in the lookback), the ratios follow from that, and
  `single_purchase_day_365d` = 1 marks these customers.
- **Country.** UK vs rest, from the country on the customer's latest line before the cutoff.
- **Return rate.** Value returned divided by value bought in the lookback. It can exceed 1 when a
  customer returns goods bought before the window.

## Training cutoffs

| Cutoff | Customers | Churned | Churn rate | One purchase day | First seen Dec 2009 | Missing values | Peak months in lookback |
|---|---:|---:|---:|---:|---:|---:|---|
| 2010-12-01 | 4,236 | 2,831 | 66.8% | 35.2% | 22.5% | 0 | 2009-12, 2010-11 |
| 2011-01-01 | 4,202 | 2,814 | 67.0% | 35.1% | 20.0% | 0 | 2010-11, 2010-12 |
| 2011-02-01 | 4,218 | 2,760 | 65.4% | 35.5% | 19.8% | 0 | 2010-11, 2010-12 |
| 2011-03-01 | 4,258 | 2,687 | 63.1% | 36.6% | 19.5% | 0 | 2010-11, 2010-12 |
| 2011-04-01 | 4,316 | 2,686 | 62.2% | 38.1% | 19.0% | 0 | 2010-11, 2010-12 |
| 2011-05-01 | 4,317 | 2,647 | 61.3% | 38.1% | 18.7% | 0 | 2010-11, 2010-12 |

Together: 25,547 customer snapshots, 16,425 churned
(64.3%), 0 missing values. A customer active at
several cutoffs appears once per cutoff. Churn counts match C2 exactly (tested).

### Feature summary (all training snapshots pooled)

| Feature | Mean | Min | P10 | Median | P90 | Max |
|---|---:|---:|---:|---:|---:|---:|
| recency_days | 112.96 | 1.00 | 13.00 | 89.00 | 262.00 | 365.00 |
| orders_90d | 1.06 | 0.00 | 0.00 | 1.00 | 3.00 | 80.00 |
| orders_180d | 2.25 | 0.00 | 0.00 | 1.00 | 5.00 | 117.00 |
| orders_365d | 4.21 | 1.00 | 1.00 | 2.00 | 9.00 | 180.00 |
| net_revenue_90d | 467.76 | -11,491.46 | 0.00 | 60.17 | 1,017.73 | 83,567.69 |
| net_revenue_180d | 997.16 | -11,491.46 | 0.00 | 349.11 | 1,913.75 | 177,965.98 |
| net_revenue_365d | 1,852.33 | -1,663.06 | 146.48 | 649.32 | 3,474.87 | 318,751.34 |
| avg_order_value_365d | 377.49 | 1.55 | 118.91 | 285.38 | 670.34 | 14,844.77 |
| orders_trend_90d | -0.12 | -60.00 | -2.00 | 0.00 | 2.00 | 58.00 |
| net_revenue_trend_90d | -61.64 | -62,677.88 | -680.15 | 0.00 | 568.28 | 44,068.38 |
| single_purchase_day_365d | 0.36 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| avg_gap_days | 178.81 | 1.00 | 23.75 | 104.00 | 365.00 | 365.00 |
| latest_gap_days | 179.08 | 1.00 | 15.00 | 117.00 | 365.00 | 365.00 |
| latest_gap_ratio | 1.02 | 0.01 | 0.48 | 1.00 | 1.48 | 8.02 |
| recency_gap_ratio | 1.95 | 0.00 | 0.13 | 0.66 | 3.19 | 363.00 |
| tenure_days | 272.89 | 1.00 | 74.00 | 289.00 | 447.00 | 516.00 |
| first_seen_first_data_month | 0.20 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| distinct_products_365d | 61.29 | 1.00 | 7.00 | 37.00 | 142.00 | 1,712.00 |
| return_rate_365d | 0.02 | 0.00 | 0.00 | 0.00 | 0.05 | 3.18 |
| is_uk | 0.92 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| bought_in_peak_365d | 0.47 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |

Full table with quartiles: `results/c3_feature_summary_training.csv`.

## Test cutoff

The 2011-09-01 table is built and saved for C4, but nothing about its features is looked at
or reported here. Only the numbers already public in C2: 4,323 active customers, churn
rate 50.5%.

## Outputs

- Snapshot tables: `data/processed/snapshots/snapshot_<cutoff>.parquet` (not committed; rebuilt by
  the run). Columns: `cutoff`, `customer_id`, the features above, `churned`.
- `results/c3_features.json` (this report's numbers) and `results/c3_feature_summary_training.csv`.

## Reproduce

```bash
python -m churn.run_c1   # if data/processed/transactions_clean.parquet is missing
python -m churn.run_c3
```

Unit tests use small hand-made tables (`pytest`); checks on the real data: `pytest -m realdata`.
