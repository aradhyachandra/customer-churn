# Notes and open questions

## TO VERIFY

- **PADS stock code (C1):** "PADS TO MATCH ALL CUSHIONS", priced at 0 or a fraction of a penny.
  Kept as a product. Its revenue is negligible either way; confirm it should not be a non-product code.
- **Within-sheet exact duplicates (C1):** dropped as the plan says. Some could be genuine repeat
  scans of the same item in one invoice at the same minute; the data cannot tell them apart.
- **Lower-case stock code suffixes (C1, matters for C3):** some codes differ only in letter case
  (for example 85123A and 85123a). Not changed in C1. Before counting "distinct products" in C3,
  decide whether to treat them as one product.
- **One stock code with a trailing space (C1):** "47503J " is a product line and left as is.
- **Later-day full cancellations (C2):** kept as purchases (see docs/C2_cohorts_and_churn.md for
  counts). If C3 wants to drop them, it may only use cancellations dated before the cutoff.
- **Churn rate differs between cutoffs (C2):** the base rate is not constant across the 10
  cutoffs (docs/C2_cohorts_and_churn.md). C4 should report the training and test base rates side by
  side and keep this in mind when reading calibration.
- **Tenure is censored for the Dec 2009 cohort (matters for C3):** those customers' real first
  purchase may be earlier than the data, so "days since first purchase" is a lower bound for them.

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
