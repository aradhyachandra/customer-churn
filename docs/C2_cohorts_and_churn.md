# C2: Cohorts, retention and descriptive churn

*Generated from `results/c2_cohorts_and_churn.json` on 2026-10-13T06:05:14+00:00 (git c8a14e4a80dd3484154333d664b35d1bb64f1a4b,
uncommitted changes: True). Do not edit by hand; edit
`docs/templates/C2_cohorts_and_churn.md` and rerun `python -m churn.run_c2`.*

## What this step does

It turns the clean invoice lines from C1 into "who bought when": one purchase event per counted
invoice, a table of each customer's activity per month, groups of customers by the month they
first bought (cohorts), how many of each group come back in later months, and how many active
customers stop buying after a set of dates (cutoffs). This is description only: no model yet.
Every number below is read from the saved report. Definitions and their reasons are in
`config/churn.yaml`.

## What counts as a purchase

Some orders are cancelled in full: a cancellation invoice reverses exactly the same lines
(stock codes, quantities, prices) for the same customer. The order nets to zero revenue but would
still look like "the customer bought something", which shifts first purchase, recency and
frequency.

How often this happens, out of 36,594 purchase invoices:

- **Cancelled on the same day:** 102 invoices from
  79 customers, worth £317,606.32.
- **Cancelled on the same day or any later day:** 239
  invoices from 183 customers, worth
  £378,883.37.

Each cancellation is matched to at most one purchase (the latest identical one before it).

The largest same-day cases:

| Customer | Purchase invoice | Cancellation invoice | Value (£) |
|---|---|---|---:|
| 16446 | 581483 | C581484 | 168,469.60 |
| 12346 | 541431 | C541433 | 77,183.60 |
| 16754 | 513653 | C513771 | 7,500.32 |
| 17450 | 529350 | C529352 | 6,325.08 |
| 12931 | 508096 | C508132 | 4,815.80 |

**Rule chosen (`drop_same_day_full_cancellations`):** a purchase invoice cancelled in full on
the same day is not a purchase event. Same-day reversals look like order-entry corrections, and
the rule never needs information from after that day, so it cannot leak future data into a
cutoff. Later-day matches are kept as purchases: when a customer repeats the same order, a later
cancellation can't be tied to one of them with confidence.

Effect of the rule: 102 invoices dropped,
36,492 purchase events left;
4 customers move to a later
first-purchase month, and 1 customer(s)
are left with no purchase at all. All lines, cancelled or not, still count in net revenue.

## Customers with only returns

24 customers have no purchase event in the data, only returns
(99 lines, net £-1,406.13). This includes the
customer(s) left with nothing by the rule above. They have no first purchase, so they are left
out of the cohorts and can never be "active". That leaves
5,851 customers with at least one purchase, out of
5,875 in the clean data.

## Customer-month table

One row per customer and calendar month with any line: number of purchase invoices, purchase
value, return value and net value. 26,797 rows, of which
25,479 have a purchase. Saved to `data/processed/customer_months.parquet`
(not committed; rebuilt by the run).

## Cohort retention

A cohort is the month of a customer's first purchase. Retention in month *k* is the share of the
cohort that bought anything *k* months later (month 0 is always 100%).

![Cohort retention heatmap](../results/figures/c2_retention_heatmap.png)

| Cohort | Customers | Month 1 | Month 3 | Month 6 | Month 12 | Note |
|---|---:|---:|---:|---:|---:|---|
| 2009-12 | 951 | 34.9% | 42.5% | 37.6% | 37.6% | includes customers who first bought before the data starts |
| 2010-01 | 368 | 21.2% | 31.5% | 26.9% | 22.8% |  |
| 2010-02 | 374 | 23.5% | 29.4% | 19.3% | 15.0% |  |
| 2010-03 | 441 | 19.0% | 24.3% | 24.7% | 20.0% |  |
| 2010-04 | 294 | 19.0% | 16.0% | 27.6% | 14.3% |  |
| 2010-05 | 253 | 15.4% | 17.8% | 21.3% | 15.4% |  |
| 2010-06 | 267 | 17.6% | 20.6% | 12.7% | 15.0% |  |
| 2010-07 | 186 | 15.6% | 29.6% | 11.3% | 12.9% |  |
| 2010-08 | 163 | 19.6% | 32.5% | 9.8% | 14.7% |  |
| 2010-09 | 238 | 23.1% | 13.0% | 13.9% | 22.7% |  |
| 2010-10 | 374 | 25.7% | 12.6% | 13.1% | 19.3% |  |
| 2010-11 | 327 | 17.4% | 9.5% | 13.1% | 25.1% |  |
| 2010-12 | 76 | 9.2% | 9.2% | 5.3% | 2.6% | month 12 is the partly covered last month |
| 2011-01 | 72 | 15.3% | 19.4% | 15.3% |  |  |
| 2011-02 | 126 | 15.9% | 19.0% | 15.9% |  |  |
| 2011-03 | 179 | 17.9% | 20.1% | 20.7% |  |  |
| 2011-04 | 106 | 24.5% | 19.8% | 17.9% |  |  |
| 2011-05 | 111 | 23.4% | 16.2% | 26.1% |  |  |
| 2011-06 | 108 | 23.1% | 26.9% | 8.3% |  | month 6 is the partly covered last month |
| 2011-07 | 101 | 20.8% | 27.7% |  |  |  |
| 2011-08 | 107 | 26.2% | 26.2% |  |  |  |
| 2011-09 | 188 | 27.1% | 14.9% |  |  | month 3 is the partly covered last month |
| 2011-10 | 222 | 31.5% |  |  |  |  |
| 2011-11 | 191 | 14.1% |  |  |  | month 1 is the partly covered last month |
| 2011-12 | 28 |  |  |  |  |  |

Full tables: `results/c2_retention_rates.csv` (shares), `results/c2_retention_counts.csv`
(customers) and `results/c2_cohort_summary.csv` (this table, with the notes).

Two things to keep in mind when reading it:

- **The first cohort (2009-12) is not a real cohort.** The data starts on
  2009-12-01 07:45:00, so everyone who bought that month lands in it, including long-standing
  customers whose real first purchase was earlier. It holds 951
  customers, against a median of 187.00 (largest
  441) for the later cohorts, and its retention is that of
  established customers, not new ones. It is marked with * in the plot.
- **The last month (2011-12) is only partly covered:** the data ends on
  2011-12-09 12:50:00. Shares in that month (hatched in the plot) are too low, and the
  2011-12 cohort has no later month yet.

## Churn at each cutoff

Following PROJECT_PLAN.md Section 3, at a cutoff date:

- **Active:** at least one purchase in the 365 days before the cutoff.
- **Churned:** an active customer with no purchase in the 90 days
  from the cutoff on.

A cutoff is midnight at the start of its date, so a purchase on the cutoff date counts as "after".
Cutoffs are the first of each month, used only when both windows fit inside the data:
10 cutoffs, from 2010-12-01 to
2011-09-01.

| Cutoff | Active | Churned | Retained | Churn rate | Note |
|---|---:|---:|---:|---:|---|
| 2010-12-01 | 4,236 | 2,831 | 1,405 | 66.8% |  |
| 2011-01-01 | 4,202 | 2,814 | 1,388 | 67.0% |  |
| 2011-02-01 | 4,218 | 2,760 | 1,458 | 65.4% |  |
| 2011-03-01 | 4,258 | 2,687 | 1,571 | 63.1% |  |
| 2011-04-01 | 4,316 | 2,686 | 1,630 | 62.2% |  |
| 2011-05-01 | 4,317 | 2,647 | 1,670 | 61.3% |  |
| 2011-06-01 | 4,324 | 2,741 | 1,583 | 63.4% |  |
| 2011-07-01 | 4,313 | 2,647 | 1,666 | 61.4% |  |
| 2011-08-01 | 4,325 | 2,461 | 1,864 | 56.9% |  |
| 2011-09-01 | 4,323 | 2,181 | 2,142 | 50.5% | test cutoff (listed only) |

Churn rates range from 50.5% to 67.0%. The
test cutoff is listed for completeness only; nothing else in this step looks at it.

### How much the window length matters

At the 2011-05-01 cutoff, with the same active customers:

| Window (days) | Active | Churned | Churn rate |
|---:|---:|---:|---:|
| 60 | 4,317 | 2,938 | 68.1% |
| 90 | 4,317 | 2,647 | 61.3% |
| 120 | 4,317 | 2,483 | 57.5% |

A shorter window labels more customers as churned, because fewer of them happen to buy again
inside it, so the reported churn rate depends strongly on this choice.

## Reproduce

```bash
python -m churn.run_c1   # if data/processed/transactions_clean.parquet is missing
python -m churn.run_c2
```

Unit tests use small hand-made tables (`pytest`); checks on the real data: `pytest -m realdata`.
