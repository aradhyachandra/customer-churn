# C3: Snapshots, features and churn labels

*Generated from `results/c3_features.json` on {{generated_at}} (git {{git.commit}},
uncommitted changes: {{git.uncommitted_changes}}). Do not edit by hand; edit
`docs/templates/C3_snapshots_and_features.md` and rerun `python -m churn.run_c3`.*

## What this step does

For each cutoff date it takes a snapshot of every active customer (at least one purchase in the
{{definitions.churn.active_lookback_days}} days before the cutoff) and describes them with
{{feature_count}} plain numbers built only from what was known before that date: how recently and
how often they bought, how much they spent, whether that is slowing down, how long they have been
a customer, and so on. It then adds the answer the model will learn: did they buy again in the next
{{definitions.churn.churn_horizon_days}} days (churned = 0) or not (churned = 1)?

Tables are saved for the {{training.count}} training cutoffs and for the test cutoff. Settings and
their reasons are in `config/churn.yaml` and `config/features.yaml`.

## How leakage is ruled out

A cutoff is midnight at the start of its date. Every feature uses only lines dated strictly before
it; the label uses only purchases from the cutoff to {{definitions.churn.churn_horizon_days}} days
later. Purchases follow the C2 rule (`{{definitions.churn.purchase_event_rule}}`): an order cancelled
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

{{table:features}}

Every feature is a plain number (counts, days, £, ratios, 0/1 flags); there is no target encoding.

### Decisions behind the features

- **Distinct products.** Stock codes are upper-cased and stripped of spaces before counting, so
  codes like 85123A and 85123a count once. In the raw file this merges
  {{stock_codes.raw_codes_merged}} of {{stock_codes.raw_distinct_codes}} codes
  ({{stock_codes.raw_lines_with_changed_code}} lines), but
  {{stock_codes.raw_lines_with_changed_code_and_customer}} of those lines have a Customer ID, so
  C1 had already removed them all: in the clean data {{stock_codes.clean_codes_merged}} of
  {{stock_codes.clean_distinct_codes}} codes merge. The rule stays in place as a safeguard.
- **Seasonality.** "Bought in the peak" means bought in a November or December inside the
  {{definitions.churn.active_lookback_days}}-day lookback. Every cutoff is the first of a month, so
  the lookback is exactly the 12 calendar months before it and always contains one complete
  November and one complete December, all inside the data (see the table below). "Last year's
  Nov-Dec" would not work at 2010-12-01, where Nov 2009 is missing from the data.
- **Tenure.** Days since the first purchase in the data. For customers first seen in December 2009
  (the first data month) the real first purchase may be earlier, so their tenure is a lower bound;
  `first_seen_first_data_month` marks them so a model can tell them apart.
- **One purchase day.** With only one purchase day in the lookback there is no gap between orders.
  Instead of a blank, the gap features take {{definitions.features.single_purchase_gap_fill_days}}
  days (no repeat seen in the lookback), the ratios follow from that, and
  `single_purchase_day_365d` = 1 marks these customers.
- **Country.** UK vs rest, from the country on the customer's latest line before the cutoff.
- **Return rate.** Value returned divided by value bought in the lookback. It can exceed 1 when a
  customer returns goods bought before the window.

## Training cutoffs

{{table:cutoffs}}

Together: {{training.rows}} customer snapshots, {{training.churned}} churned
({{training.churn_rate|pct}}), {{training.missing_values}} missing values. A customer active at
several cutoffs appears once per cutoff. Churn counts match C2 exactly (tested).

### Feature summary (all training snapshots pooled)

{{table:summary}}

Full table with quartiles: `{{outputs.c3_feature_summary}}`.

## Test cutoff

The {{test.cutoff}} table is built and saved for C4, but nothing about its features is looked at
or reported here. Only the numbers already public in C2: {{test.rows}} active customers, churn
rate {{test.churn_rate|pct}}.

## Outputs

- Snapshot tables: `{{outputs.snapshots_dir}}/snapshot_<cutoff>.parquet` (not committed; rebuilt by
  the run). Columns: `cutoff`, `customer_id`, the features above, `churned`.
- `results/c3_features.json` (this report's numbers) and `{{outputs.c3_feature_summary}}`.

## Reproduce

```bash
python -m churn.run_c1   # if data/processed/transactions_clean.parquet is missing
python -m churn.run_c3
```

Unit tests use small hand-made tables (`pytest`); checks on the real data: `pytest -m realdata`.
