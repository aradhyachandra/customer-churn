# C1: Data loading and cleaning

*Generated from `results/c1_cleaning_report.json` on {{generated_at}} (git {{git.commit}},
uncommitted changes: {{git.uncommitted_changes}}). Do not edit by hand; edit
`docs/templates/C1_data_and_cleaning.md` and rerun `python -m churn.run_c1`.*

## What this step does

It reads the retailer's transaction file, removes lines that would give wrong answers later
(copies, fees, bad prices, lines with no customer), and saves a clean table of invoice lines for
the churn analysis. Every number below is read from the saved cleaning report.

## The raw file

- File: `{{raw_file}}`, sha256 `{{raw_sha256}}`. The loader recomputes this every run and stops
  with a clear error if the file is different.
- Two sheets: "Year 2009-2010" with {{raw.rows_by_sheet.Year 2009-2010}} lines and
  "Year 2010-2011" with {{raw.rows_by_sheet.Year 2010-2011}} lines, {{raw.rows}} in total.
- Customer ID is stored as a whole number with gaps allowed (so 13085 stays 13085, not 13085.0).
- The Excel file is read once and cached as parquet in `data/processed/`; later runs use the cache.

## Do the two sheets overlap?

Yes, and the overlap is an exact copy.

| Sheet | First line | Last line | Lines | Duplicate lines inside the sheet |
|---|---|---|---:|---:|
| Year 2009-2010 | {{overlap.sheets.Year 2009-2010.first_date}} | {{overlap.sheets.Year 2009-2010.last_date}} | {{overlap.sheets.Year 2009-2010.rows}} | {{overlap.sheets.Year 2009-2010.within_sheet_duplicate_rows}} |
| Year 2010-2011 | {{overlap.sheets.Year 2010-2011.first_date}} | {{overlap.sheets.Year 2010-2011.last_date}} | {{overlap.sheets.Year 2010-2011.rows}} | {{overlap.sheets.Year 2010-2011.within_sheet_duplicate_rows}} |

- Both sheets cover {{overlap.overlap_date_range.start}} to {{overlap.overlap_date_range.end}}.
- In that window the first sheet has {{overlap.rows_in_overlap_range.Year 2009-2010}} lines and the
  second has {{overlap.rows_in_overlap_range.Year 2010-2011}}.
- {{overlap.later_rows_identical_to_earlier_sheet}} lines in the second sheet are identical (all
  eight columns) to a line in the first sheet. Lines in the window with no identical partner:
  {{overlap.earlier_rows_in_range_without_match}} in the first sheet,
  {{overlap.later_rows_in_range_without_match}} in the second. Identical lines outside the window:
  {{overlap.later_rows_identical_outside_overlap_range}}.
- So those days were exported twice. We keep the first sheet's copy and drop the second's.

## Cleaning steps

Steps run in this order. A line removed by one step is not counted again by a later one.
"Net value" is quantity x price summed (returns count as negative); "sales value" sums only the
positive lines.

{{steps_table}}

What each step means:

1. **Sheet overlap**: the double-exported days described above.
2. **Duplicate lines within one sheet**: lines identical in every column to another line in the
   same sheet; the first copy is kept. Before cleaning, the sheets held
   {{overlap.sheets.Year 2009-2010.within_sheet_duplicate_rows}} and
   {{overlap.sheets.Year 2010-2011.within_sheet_duplicate_rows}} such extra copies. This step
   removed {{steps.1.rows_removed_by_sheet.Year 2009-2010}} and
   {{steps.1.rows_removed_by_sheet.Year 2010-2011}}: the second sheet's remaining copies inside
   the overlap window had already gone in step 1. This number is kept separate from the overlap.
3. **Zero or negative price**: zero-price lines are mostly stock notes ("damaged", "missing",
   "?") with no customer and often no description; the negative prices are "Adjust bad debt"
   accounting entries.
4. **Non-product codes**: postage, carriage, fees, manual adjustments, discounts, samples, test
   items and gift vouchers. The full list, with a reason for each, is in `config/cleaning.yaml`.
   Real products sold through the dotcom gift shop (codes starting DCGS) are kept.
5. **No Customer ID**: these sales can't be tied to a customer, so they can't be used for churn.
   They are {{steps.4.rows_removed}} lines worth £{{steps.4.sales_value_removed}} in sales after
   the earlier steps, so customer-level revenue in this project is lower than the shop's total.

## Returns

Cancellations (invoice starting "C") and negative quantities are **kept**, flagged as returns,
and will be netted against purchases per customer. They are not counted as purchases.

- Return lines in the clean data: {{returns.rows}}, net value in GBP {{returns.net_value}}
  (negative because returns reduce revenue).
- Cancellation lines with a positive quantity: {{returns.cancellation_rows_with_positive_quantity}}.
  Negative-quantity lines that are not cancellations: {{returns.negative_quantity_rows_not_cancellation}}.
  So "cancellation" and "negative quantity" mean the same lines after cleaning.

## What is left

- {{final.rows}} lines ({{final.purchase_rows}} purchase lines, {{returns.rows}} return lines)
  from {{final.customers}} customers and {{final.purchase_invoices}} purchase invoices.
- Dates {{final.first_date}} to {{final.last_date}}.
- Purchase sales £{{final.sales_value}}; net of returns £{{final.net_value}} (GBP, no conversion).
- Saved to `{{clean_file}}` with two added columns: `line_value` (quantity x price) and `is_return`.

## Reproduce

```bash
python -m churn.run_c1
```

Unit tests use small hand-made tables only (`pytest`). Checks on the real file are separate:
`pytest -m realdata`.
