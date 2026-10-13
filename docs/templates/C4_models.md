# C4: Models and validation

*Generated from `results/c4_validation_metrics.json` on {{generated_at}} (git {{git.commit}},
uncommitted changes: {{git.uncommitted_changes}}). Do not edit by hand; edit
`docs/templates/C4_models.md` and rerun `python -m churn.run_c4`.*

## What this step does

It builds three ways of spotting customers who are about to stop buying, and compares them on a
month the models did not learn from:

1. **A simple rule:** rank customers by days since their last purchase, longest first. A yes/no
   version ("churn if no purchase for more than X days") is reported alongside.
2. **Logistic regression:** each of the {{feature_count}} features pushes the risk up or down by a fixed amount,
   after the features are put on a common scale.
3. **Gradient boosting** (scikit-learn `HistGradientBoostingClassifier`, version
   {{sklearn_version}}): many small decision trees, each correcting the previous ones.

**The test cutoff ({{test_cutoff}}) was not read in this step.** All settings were chosen
on the validation cutoff, and the final models are frozen for a single test run in part B.

## How the models were tuned without touching the test

The test asks: trained on earlier months, how well does a model rank customers at a later month?
Tuning asks the same question one step earlier. The validation cutoff is
{{scheme.validation_cutoff}}. A training cutoff may be used to fit models during tuning only if its
{{scheme.horizon_days}}-day outcome window ends on or before the validation cutoff, the same gap
rule the test uses, so no label used for fitting overlaps the validation period:

{{table:cutoffs}}

For each model, every candidate setting is fitted on the tuning cutoffs and scored on the
validation cutoff; the setting with the highest validation PR-AUC is kept. Then the chosen settings
are refitted on all six training cutoffs; those are the final models for the test.

Candidates (all in `config/model.yaml`, each with a reason):

- **Rule:** the ranking uses `recency_days` directly, so it has nothing to tune and all its
  ranking metrics below do not depend on X. X only sets the yes/no flag: chosen from
  {{candidates_count.rule}} values by the validation PR-AUC of the 1/0 flag. Chosen: `{{selected_text.rule}}`.
- **Logistic regression:** {{candidates_count.logistic}} ways of preparing the features. Chosen:
  `{{selected_text.logistic}}`. `log_signed_log` takes log(1 + x) of counts, days, averages and
  ratios that are never negative, and sign(x) x log(1 + |x|) of net revenue and trends, which can
  be negative. `log_clip` keeps log(1 + x) but clips the features that can be negative at their
  1st and 99th training percentiles. `standardise_only` does no reshaping. All three then put
  every feature on mean 0, standard deviation 1. Every value these steps learn (percentiles,
  means, standard deviations) comes from the training rows only (tested).
- **Gradient boosting:** {{candidates_count.gbm}} settings (learning rate x tree size x minimum leaf
  size), 200 trees each, no early stopping. Chosen: `{{selected_text.gbm}}`.

All seeds are fixed (`config/model.yaml`), so a rerun gives the same models and intervals.

## Churn base rates: training and validation side by side

The share of active customers who churn is not the same at every cutoff (NOTES.md), so a model's
probabilities are learned at one level and judged at another.

{{table:base_rates}}

Per training cutoff:

{{table:per_cutoff}}

## Validation results

Scores on {{scheme.validation_cutoff}} ({{base_rates.validation.customers}} active customers,
{{base_rates.validation.churners}} churned, base rate {{base_rates.validation.churn_rate|pct}}),
from models fitted on the tuning cutoffs only. 95% intervals come from
{{metrics.gbm.bootstrap.samples}} bootstrap resamples of the validation customers; they show
sampling noise, not how results move from one month to another.

{{table:metrics}}

How to read it: ROC-AUC is the chance that a random churner is scored above a random customer who
stays (0.5 = coin flip). PR-AUC summarises precision across all list lengths; a model that ranks
at random scores the base rate. Precision in the top 10% is the share of churners among the 10% of
customers with the highest scores, the group a retention campaign would contact first. Brier is
the mean squared gap between the probability and what happened (lower is better). Mean predicted
is the average churn probability, to compare with the base rate
({{base_rates.validation.churn_rate|pct}}).

**Counts behind the top-10% and top-20% figures:**

{{table:counts}}

Customers tied at the cut-off score share the remaining places equally, which is why a count can
have a decimal (many customers share the same recency_days).

The yes/no flag `{{selected_text.rule}}` says yes for {{metrics.rule.flag.flagged}} customers
({{metrics.rule.flag.flagged_share|pct}}); {{metrics.rule.flag.flag_precision|pct}} of them churned, and it catches
{{metrics.rule.flag.flag_recall|pct}} of all churners.

**Caution:** these are the scores of the settings that won on this same month, so they are a little
optimistic. The test cutoff gives the honest number.

### Calibration (do predicted probabilities match what happened?)

On average the logistic regression predicts {{metrics.logistic.mean_predicted|pct}} and gradient
boosting {{metrics.gbm.mean_predicted|pct}}, against an observed {{base_rates.validation.churn_rate|pct}}.
The models learned their probabilities on the tuning cutoffs, where the churn rate was
{{base_rates.tuning.churn_rate|pct}}, so a gap in this direction is expected when the churn rate falls.
Ranking (AUC, precision in the top groups) does not depend on that level; the probabilities do.

![Calibration at the validation cutoff](../{{outputs.c4_calibration_plot}})

{{table:calibration}}

The rule's score is a number of days, not a probability, so it has no Brier score or calibration table.

### Every candidate tried

{{table:candidates}}

Files: `{{outputs.c4_validation_summary}}` (all metrics with intervals),
`{{outputs.c4_validation_tuning}}`, `{{outputs.c4_validation_calibration}}`, and the frozen-model
manifest `{{outputs.c4_final_models}}` (chosen settings, training cutoffs and the sha256 of each
saved model file in `data/processed/models/`).

## Test results (part B)

> **NOT RUN YET.** This section stays empty until part B runs `python -m churn.run_c4_test` once,
> from a clean, committed working tree, with the frozen models above. That script writes
> `results/c4_test_metrics.json` (with the git SHA and data checksums) and refuses to run a second
> time. The test-cutoff metrics table, counts, calibration and base rate go here.
