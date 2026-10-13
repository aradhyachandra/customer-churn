# Customer Churn & Retention Analytics: Project Plan

Author: Aradhya Chandra. Status: plan only (no code yet). Target: about 1 week of build sessions.

## 0. One-paragraph summary

Using two years of real transactions from a UK online retailer (UCI "Online Retail II", about 1.07
million invoice lines, Dec 2009 - Dec 2011), this project (1) defines churn from purchase behaviour,
(2) measures retention by customer cohort, (3) predicts which active customers will stop buying in
the next 90 days, (4) explains what drives that risk, and (5) turns the scores into a costed
retention plan: whom to target, what it costs, and what revenue it could protect. The output is a
consulting-style findings report, a small dashboard and reproducible code with an honest,
out-of-time evaluation.

**Principle:** every business number is traceable to the data or to a clearly labelled assumption in
config. The model is evaluated on a later time period it never saw.

## 1. The business questions

1. How many customers does the retailer lose, and when? (churn rate, cohort retention curves)
2. Which currently active customers are likely to stop buying in the next 90 days?
3. What behaviour signals rising risk? (feature importance, plain-language drivers)
4. How much revenue is at risk, and which customers are worth a retention offer?
5. What would a retention campaign cost and return, under stated assumptions?

## 2. Data

- Source: UCI Machine Learning Repository, dataset 502, "Online Retail II", CC BY 4.0 (attribution
  required; cite in README). Download: https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip
  (one Excel file, two sheets: 2009-2010 and 2010-2011; about 43.5 MB).
- Columns: Invoice (starting "C" = cancellation), StockCode, Description, Quantity, InvoiceDate,
  Price (GBP), Customer ID, Country.
- Raw file lives in `data/raw/` (gitignored; the licence allows redistribution but the file is large).
  A checksum (sha256) of the raw file is recorded so results are reproducible.

### 2.1 Cleaning rules (decide and document; each with a count of rows affected)
- Rows without Customer ID: excluded from customer-level analysis (report how many and their revenue).
- Cancellations (Invoice starts with "C") and negative quantities: kept as returns, netted against
  purchases at customer level; not counted as purchase events.
- Non-product codes (postage, bank charges, manual adjustments, discounts, samples; for example
  StockCode POST, DOT, M, BANK CHARGES, D, AMAZONFEE, ADJUST): excluded from product revenue (list
  the final code set in config).
- Zero or negative prices: excluded.
- Exact duplicate lines: dropped (report count).
- Line revenue = Quantity x Price. Currency GBP throughout; no conversion.
- TO VERIFY during C1: overlap between the two sheets (the first sheet's last weeks and the second
  sheet's first weeks) so no transactions are double counted.

## 3. Definitions (the part that makes or breaks the project)

- **Active customer at a cutoff date:** at least one purchase in the 365 days before the cutoff.
- **Churn label:** an active customer with **no purchase in the 90 days after the cutoff**.
  The 90-day window is a choice; a sensitivity check uses 60 and 120 days.
- **Snapshot design (no leakage):** features use only data on or before the cutoff; the label uses
  only the 90 days after it. Several cutoffs build the training set (for example monthly from
  2010-06-01 to 2011-05-01). The final **test set is a later, untouched cutoff** (2011-09-01, whose
  90-day window ends inside the data). Nothing from the test period is used for training or tuning.
- **Cohort:** month of a customer's first purchase. Retention = share of a cohort buying again in
  each later month.

## 4. Features (per customer, per cutoff; all computed in code from transactions)

- Recency (days since last purchase), frequency (orders in 90/180/365 days), monetary (net revenue in
  the same windows), average order value.
- Trend: order count and revenue in the last 90 days vs the 90 days before.
- Gap behaviour: average and latest gap between orders; latest gap / average gap.
- Tenure (days since first purchase), number of distinct products, return rate (share of value
  returned), country group (UK vs rest), seasonality flag (bought in the Nov-Dec peak last year).

## 5. Models and evaluation

- Baselines first: (a) rule "recency > X days" with X chosen on training data; (b) logistic regression.
- Main model: gradient boosted trees (scikit-learn HistGradientBoosting or LightGBM, pinned).
- Metrics on the out-of-time test cutoff: ROC-AUC, PR-AUC, **precision and recall in the top 10% /
  top 20% of scores** (the business-relevant view), and calibration (do 70% scores churn ~70% of the
  time?). Report counts as well as rates, and the churn base rate.
- Explainability: permutation importance plus partial dependence on the top drivers; written as plain
  statements ("customers whose latest gap is twice their usual gap churn X times as often").
- Guardrails: no feature computed with post-cutoff data (tested); random seeds fixed; results saved
  with git SHA and data checksum.

## 6. Business layer (assumptions in config, never hidden)

- **Revenue at risk:** sum over test-cutoff active customers of churn probability x their trailing
  12-month net revenue.
- **Customer value:** trailing 12-month net revenue, optionally a simple expected-lifetime value
  (labelled as an estimate). Gross margin is unknown in the data: a config assumption (for example
  30%) with sensitivity 20-40%.
- **Targeting rule:** expected benefit of contacting a customer = p(churn) x save rate x 12-month
  margin - offer cost. Save rate (share of contacted churners kept) is unknown: a labelled assumption
  with a scenario table (5%, 10%, 20%). Rank customers by expected benefit; show the campaign size
  that maximises net benefit, and how the answer changes under each assumption.
- Wording rule: these are estimates under stated assumptions, not observed effects. No causal claims.

## 7. Deliverables

1. **Findings report** (consulting style, about 8-10 pages or slides): problem, data, cohort and
   churn picture, model performance (out-of-time), drivers, targeting plan with costs and benefits,
   limitations. Every number generated by code from saved results.
2. **Dashboard** (Streamlit): cohort retention heatmap, churn by segment, customer risk list with
   drivers, campaign calculator with the assumption sliders.
3. **Repo**: README with results table, how to reproduce, data licence and attribution; tests.

## 8. Milestones (each about one Claude Code session; tests first)

| ID | What | Done when |
|---|---|---|
| C1 | Repo setup, load both sheets, cleaning rules, cleaning report | Row counts at each cleaning step reported and tested; sheet overlap checked; checksum recorded |
| C2 | Customer-month tables, cohorts, retention curves, descriptive churn | Cohort retention table and plot reproduce from code; churn rate per cutoff |
| C3 | Snapshot builder + features + labels | A test proves no feature uses post-cutoff data; feature table per cutoff saved |
| C4 | Baselines + main model, out-of-time test | Metrics table (AUC, PR-AUC, precision@10/20%, calibration) for all models; test cutoff touched once |
| C5 | Drivers + business layer | Plain-language driver statements; revenue at risk; campaign table under each assumption |
| C6 | Report + dashboard + README | Report numbers generated from results; dashboard runs; clean-clone check passes |

## 9. Working rules for any coding tool

- Read this plan and PROGRESS.md first; one milestone per session; tests first.
- Never use data after a cutoff to build features for that cutoff. Never tune on the test cutoff.
- All assumptions (churn window, margin, save rate, offer cost, non-product codes) live in
  `config/`, each with a one-line rationale.
- No invented numbers: every figure in the report comes from saved results.
- Keep code plain and readable; pin dependencies; mark uncertain things TO VERIFY in NOTES.md.
- End each session by updating PROGRESS.md and adding a short plain-language note in `docs/`.

## 10. Risks

- **Leakage** (the classic churn mistake): handled by the snapshot design and a test.
- **Assumption-heavy money numbers:** shown as scenarios, never as single facts.
- **Wholesale-heavy customers** (many buyers are small businesses): segment results; note it.
- **Old data (2009-2011):** the method transfers; the numbers describe this retailer only.

## 11. README outline

What and why (3 lines) · Results table (out-of-time) · Key findings (5 bullets) · How it works
(diagram) · Reproduce · Data and licence (UCI, CC BY 4.0) · Assumptions · Limitations · Author.
