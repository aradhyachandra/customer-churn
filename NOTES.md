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
- **Customers with only returns left after cleaning (C2):** check how many there are before
  computing first-purchase cohorts; they have no purchase event.
- **Purchases cancelled in full (C2/C3, found in C1 review):** the two largest lines in the clean
  data (80,995 x "PAPER CRAFT, LITTLE BIRDIE", customer 16446; 74,215 x "MEDIUM CERAMIC TOP
  STORAGE JAR", customer 12346) are each cancelled by the next invoice. They net to zero in
  revenue but still count as purchase events, which affects first purchase, recency and frequency.
  Decide in C2 whether a purchase cancelled in full still counts as a purchase.

## Decisions

- **Sheet overlap (C1):** measured, not assumed. The overlap window is an exact copy in both sheets
  (see docs/C1_data_and_cleaning.md), so the later sheet's identical lines are dropped.
- **Excel engine (C1):** python-calamine. Reading the workbook took about 8 s and about 1.2 GB peak
  memory on the 8 GB Mac; the parquet cache makes later runs about 2 s.
- **results/ is gitignored** (from the initial .gitignore), so `results/c1_cleaning_report.json`
  is not versioned; the rendered doc in docs/ is.
