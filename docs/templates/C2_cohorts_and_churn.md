# C2: Cohorts, retention and descriptive churn

*Generated from `results/c2_cohorts_and_churn.json` on {{generated_at}} (git {{git.commit}},
uncommitted changes: {{git.uncommitted_changes}}). Do not edit by hand; edit
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

How often this happens, out of {{purchase_events.purchase_invoices}} purchase invoices:

- **Cancelled on the same day:** {{purchase_events.same_day.purchase_invoices}} invoices from
  {{purchase_events.same_day.customers}} customers, worth £{{purchase_events.same_day.value}}.
- **Cancelled on the same day or any later day:** {{purchase_events.any_later_date.purchase_invoices}}
  invoices from {{purchase_events.any_later_date.customers}} customers, worth
  £{{purchase_events.any_later_date.value}}.

Each cancellation is matched to at most one purchase (the latest identical one before it).

The largest same-day cases:

{{table:largest_cancelled}}

**Rule chosen (`{{definitions.purchase_event_rule}}`):** a purchase invoice cancelled in full on
the same day is not a purchase event. Same-day reversals look like order-entry corrections, and
the rule never needs information from after that day, so it cannot leak future data into a
cutoff. Later-day matches are kept as purchases: when a customer repeats the same order, a later
cancellation can't be tied to one of them with confidence.

Effect of the rule: {{purchase_events.chosen_rule.purchase_invoices_dropped}} invoices dropped,
{{purchase_events.events_kept}} purchase events left;
{{purchase_events.chosen_rule.customers_first_purchase_month_changed}} customers move to a later
first-purchase month, and {{purchase_events.chosen_rule.customers_left_with_no_purchase}} customer(s)
are left with no purchase at all. All lines, cancelled or not, still count in net revenue.

## Customers with only returns

{{return_only_customers.customers}} customers have no purchase event in the data, only returns
({{return_only_customers.lines}} lines, net £{{return_only_customers.net_value}}). This includes the
customer(s) left with nothing by the rule above. They have no first purchase, so they are left
out of the cohorts and can never be "active". That leaves
{{purchase_events.customers_with_purchase}} customers with at least one purchase, out of
{{data.customers}} in the clean data.

## Customer-month table

One row per customer and calendar month with any line: number of purchase invoices, purchase
value, return value and net value. {{customer_months.rows}} rows, of which
{{customer_months.months_with_purchase}} have a purchase. Saved to `{{customer_months.file}}`
(not committed; rebuilt by the run).

## Cohort retention

A cohort is the month of a customer's first purchase. Retention in month *k* is the share of the
cohort that bought anything *k* months later (month 0 is always 100%).

![Cohort retention heatmap](../{{outputs.c2_heatmap}})

{{table:cohorts}}

Full tables: `{{outputs.c2_retention_rates}}` (shares), `{{outputs.c2_retention_counts}}`
(customers) and `{{outputs.c2_cohort_summary}}` (this table, with the notes).

Two things to keep in mind when reading it:

- **The first cohort ({{cohorts.first}}) is not a real cohort.** The data starts on
  {{data.first_date}}, so everyone who bought that month lands in it, including long-standing
  customers whose real first purchase was earlier. It holds {{cohorts.first_cohort_size}}
  customers, against a median of {{cohorts.later_cohorts_median_size}} (largest
  {{cohorts.later_cohorts_max_size}}) for the later cohorts, and its retention is that of
  established customers, not new ones. It is marked with * in the plot.
- **The last month ({{data.last_month}}) is only partly covered:** the data ends on
  {{data.last_date}}. Shares in that month (hatched in the plot) are too low, and the
  {{cohorts.last}} cohort has no later month yet.

## Churn at each cutoff

Following PROJECT_PLAN.md Section 3, at a cutoff date:

- **Active:** at least one purchase in the {{churn_by_cutoff.lookback_days}} days before the cutoff.
- **Churned:** an active customer with no purchase in the {{churn_by_cutoff.horizon_days}} days
  from the cutoff on.

A cutoff is midnight at the start of its date, so a purchase on the cutoff date counts as "after".
Cutoffs are the first of each month, used only when both windows fit inside the data:
{{churn_by_cutoff.count}} cutoffs, from {{churn_by_cutoff.first_cutoff}} to
{{churn_by_cutoff.last_cutoff}}.

{{table:churn}}

Churn rates range from {{churn_by_cutoff.min_rate|pct}} to {{churn_by_cutoff.max_rate|pct}}. The
test cutoff is listed for completeness only; nothing else in this step looks at it.

### How much the window length matters

At the {{sensitivity.cutoff}} cutoff, with the same active customers:

{{table:sensitivity}}

A shorter window labels more customers as churned, because fewer of them happen to buy again
inside it, so the reported churn rate depends strongly on this choice.

## Reproduce

```bash
python -m churn.run_c1   # if data/processed/transactions_clean.parquet is missing
python -m churn.run_c2
```

Unit tests use small hand-made tables (`pytest`); checks on the real data: `pytest -m realdata`.
