# C1: Data loading and cleaning

*Generated from `results/c1_cleaning_report.json` on 2026-10-13T05:27:40+00:00 (git ac8f20b3a7ae793a1a3cee8072ca91f1ce4bafa2,
uncommitted changes: True). Do not edit by hand; edit
`docs/templates/C1_data_and_cleaning.md` and rerun `python -m churn.run_c1`.*

## What this step does

It reads the retailer's transaction file, removes lines that would give wrong answers later
(copies, fees, bad prices, lines with no customer), and saves a clean table of invoice lines for
the churn analysis. Every number below is read from the saved cleaning report.

## The raw file

- File: `data/raw/online_retail_II.xlsx`, sha256 `bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980`. The loader recomputes this every run and stops
  with a clear error if the file is different.
- Two sheets: "Year 2009-2010" with 525,461 lines and
  "Year 2010-2011" with 541,910 lines, 1,067,371 in total.
- Customer ID is stored as a whole number with gaps allowed (so 13085 stays 13085, not 13085.0).
- The Excel file is read once and cached as parquet in `data/processed/`; later runs use the cache.

## Do the two sheets overlap?

Yes, and the overlap is an exact copy.

| Sheet | First line | Last line | Lines | Duplicate lines inside the sheet |
|---|---|---|---:|---:|
| Year 2009-2010 | 2009-12-01 07:45:00 | 2010-12-09 20:01:00 | 525,461 | 6,865 |
| Year 2010-2011 | 2010-12-01 08:26:00 | 2011-12-09 12:50:00 | 541,910 | 5,268 |

- Both sheets cover 2010-12-01 08:26:00 to 2010-12-09 20:01:00.
- In that window the first sheet has 22,523 lines and the
  second has 22,523.
- 22,523 lines in the second sheet are identical (all
  eight columns) to a line in the first sheet. Lines in the window with no identical partner:
  0 in the first sheet,
  0 in the second. Identical lines outside the window:
  0.
- So those days were exported twice. We keep the first sheet's copy and drop the second's.

## Cleaning steps

Steps run in this order. A line removed by one step is not counted again by a later one.
"Net value" is quantity x price summed (returns count as negative); "sales value" sums only the
positive lines.

| Step | Rows before | Rows removed | Rows after | Net value removed (£) | Sales value removed (£) |
|---|---:|---:|---:|---:|---:|
| drop_sheet_overlap | 1,067,371 | 22,523 | 1,044,848 | 377,488.45 | 438,852.65 |
| drop_within_sheet_duplicates | 1,044,848 | 11,812 | 1,033,036 | 54,228.42 | 57,481.47 |
| drop_non_positive_price | 1,033,036 | 6,019 | 1,027,017 | -158,676.14 | 0.00 |
| drop_non_product_codes | 1,027,017 | 5,745 | 1,021,272 | 86,810.77 | 832,772.38 |
| drop_missing_customer_id | 1,021,272 | 227,089 | 794,183 | 2,568,806.36 | 2,575,278.90 |

What each step means:

1. **Sheet overlap**: the double-exported days described above.
2. **Duplicate lines within one sheet**: lines identical in every column to another line in the
   same sheet; the first copy is kept. Before cleaning, the sheets held
   6,865 and
   5,268 such extra copies. This step
   removed 6,865 and
   4,947: the second sheet's remaining copies inside
   the overlap window had already gone in step 1. This number is kept separate from the overlap.
3. **Zero or negative price**: zero-price lines are mostly stock notes ("damaged", "missing",
   "?") with no customer and often no description; the negative prices are "Adjust bad debt"
   accounting entries.
4. **Non-product codes**: postage, carriage, fees, manual adjustments, discounts, samples, test
   items and gift vouchers. The full list, with a reason for each, is in `config/cleaning.yaml`.
   Real products sold through the dotcom gift shop (codes starting DCGS) are kept.
5. **No Customer ID**: these sales can't be tied to a customer, so they can't be used for churn.
   They are 227,089 lines worth £2,575,278.90 in sales after
   the earlier steps, so customer-level revenue in this project is lower than the shop's total.

## Returns

Cancellations (invoice starting "C") and negative quantities are **kept**, flagged as returns,
and will be netted against purchases per customer. They are not counted as purchases.

- Return lines in the clean data: 17,587, net value in GBP -709,990.03
  (negative because returns reduce revenue).
- Cancellation lines with a positive quantity: 0.
  Negative-quantity lines that are not cancellations: 0.
  So "cancellation" and "negative quantity" mean the same lines after cleaning.

## What is left

- 794,183 lines (776,596 purchase lines, 17,587 return lines)
  from 5,875 customers and 36,594 purchase invoices.
- Dates 2009-12-01 07:45:00 to 2011-12-09 12:50:00.
- Purchase sales £17,068,582.74; net of returns £16,358,592.71 (GBP, no conversion).
- Saved to `data/processed/transactions_clean.parquet` with two added columns: `line_value` (quantity x price) and `is_return`.

## Reproduce

```bash
python -m churn.run_c1
```

Unit tests use small hand-made tables only (`pytest`). Checks on the real file are separate:
`pytest -m realdata`.
