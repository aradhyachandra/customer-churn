# Notes and open questions

## TO VERIFY

- **PADS stock code (C1):** "PADS TO MATCH ALL CUSHIONS", priced at 0 or a fraction of a penny.
  Kept as a product. Its revenue is negligible either way; confirm it should not be a non-product code.
- **Within-sheet exact duplicates (C1):** dropped as the plan says. Some could be genuine repeat
  scans of the same item in one invoice at the same minute; the data cannot tell them apart.
- **Later-day full cancellations (C2):** kept as purchases (see docs/C2_cohorts_and_churn.md for
  counts). If C3 wants to drop them, it may only use cancellations dated before the cutoff.
- **Churn rate differs between cutoffs (C2):** the base rate is not constant across the 10
  cutoffs (docs/C2_cohorts_and_churn.md). C4 should report the training and test base rates side by
  side and keep this in mind when reading calibration.
- **Tenure is censored for the Dec 2009 cohort:** those customers' real first purchase may be
  earlier than the data, so "days since first purchase" is a lower bound. C3 adds the flag
  `first_seen_first_data_month`. Also, the largest possible tenure grows with the cutoff (about
  365 days at 2010-12-01, more later), so tenure is not on quite the same scale at every cutoff.
- **"Latest gap" meaning (C3):** `latest_gap_days` is the last *completed* gap (between the last
  two purchase days); the "overdue" signal (time since the last purchase vs the usual gap) is the
  separate `recency_gap_ratio`. PROJECT_PLAN.md Section 4 could be read either way; confirm.
- **Heavy tails (C3, matters for C4):** in the training snapshots `recency_gap_ratio` reaches 363
  (very frequent buyer who then stopped), `return_rate_365d` exceeds 1 and `net_revenue_*` can be
  negative (returns of goods bought before the window). Values are kept as they are; logistic
  regression in C4 may need log/clipping, decided on training data only.
- **Same customer at several cutoffs (C3, matters for C4):** training rows repeat customers across
  the 6 cutoffs, so any validation split inside training should group by customer or by cutoff.
- **Optional 2011-06-01 training cutoff (C2 review):** its 90-day window ends 2011-08-30, before the
  test cutoff, so it would be legitimate. Not added; training cutoffs stay 2010-12-01..2011-05-01.

## Decisions

- **Sheet overlap (C1):** measured, not assumed. The overlap window is an exact copy in both sheets
  (see docs/C1_data_and_cleaning.md), so the later sheet's identical lines are dropped.
- **Excel engine (C1):** python-calamine. Reading the workbook took about 8 s and about 1.2 GB peak
  memory on the 8 GB Mac; the parquet cache makes later runs about 2 s.
- **results/ is versioned (C2):** small JSON/CSV reports and figures are committed; only
  `results/**/*.parquet`, `*.pkl`, `*.joblib`, `*.zip` are ignored. Large tables go in
  data/processed/.
- **Purchase event (C2):** a purchase invoice cancelled in full by the same customer on the same
  day is not a purchase event (one-to-one matching on stock code, quantity and price). This covers
  customers 16446 and 12346 from the C1 review. Rule and reason in config/churn.yaml; counts in
  docs/C2_cohorts_and_churn.md.
- **Return-only customers (C2):** counted and left out of cohorts; they are never "active".
- **Cutoff convention (C2):** a cutoff is 00:00 on its date; purchases on the cutoff date belong
  to the outcome window.
- **Dec 2009 cohort (C2):** flagged as left-censored (it includes customers who first bought
  before the data starts) in the cohort table, plot and doc. Dec 2011 is a partial month.
- **Training cutoffs (decided after C2, 2026-10-13):** monthly from 2010-12-01 to 2011-05-01
  (6 cutoffs), not from 2010-06-01 as PROJECT_PLAN.md Section 3 suggested: earlier cutoffs lack a
  full 365-day lookback. Set in config/churn.yaml (training_cutoffs).
- **Stock code case/space variants (C3):** codes are upper-cased and stripped before counting
  distinct products (config/features.yaml). All raw lines with such variants have no Customer ID,
  so C1 removed them and no code merges in the clean data (counts in
  docs/C3_snapshots_and_features.md). The rule stays as a safeguard.
- **Seasonality flag (C3):** "bought in the Nov or Dec inside the 365-day lookback", which always
  holds one complete November and December inside the data (tested for every configured cutoff),
  unlike "last year's Nov-Dec", which is half missing at 2010-12-01.
- **One purchase day (C3):** gap features take 365 (the lookback length), ratios follow, and
  `single_purchase_day_365d` = 1 flags the rows. No NaNs in any snapshot.
- **Test cutoff (C3):** its snapshot is built and saved; only its row count and churn rate are
  reported. The run stops with an error if any snapshot, test included, has a missing value (an
  integrity check; nothing is summarised).
