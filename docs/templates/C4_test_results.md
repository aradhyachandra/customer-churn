## Test results (part B)

*Generated from `{{outputs.c4_test_metrics}}` (written once on {{test.generated_at}} at git
{{test.git.commit}}, uncommitted changes: {{test.git.uncommitted_changes}}) by
`python -m churn.render_c4_test`, which only formats saved numbers. Do not edit by hand; edit
`docs/templates/C4_test_results.md`.*

The frozen final models (sha256 in `results/c4_final_models.json`, checked before scoring) scored
the test cutoff {{test.test_snapshot.cutoff}} **once**: {{base.customers}} active customers,
{{base.churners}} churned, base rate **{{base.churn_rate|pct}}**. Nothing was tuned or refitted
afterwards. Test snapshot sha256: `{{test.test_snapshot.sha256}}`.

### Base rates

{{table:base_rates}}

The test month's churn rate is well below every training month's. That matters for the
probabilities (calibration, below), not for the ranking.

### Validation and test side by side

{{table:side_by_side}}

Intervals: 95% from {{test.metrics.gbm.bootstrap.samples}} bootstrap resamples of the customers
of that one cutoff (sampling noise only). Validation scores come from the models fitted on the
tuning cutoffs; test scores from the final models refitted on all six training cutoffs.

What the test supports:

- {{comparison.gbm_vs_rule}}
- {{comparison.gbm_vs_logistic}}
- {{comparison.logistic_vs_rule}}

"Do not overlap" is a cautious check: each interval is computed for one model on its own, so two
models can differ even when their intervals overlap a little. Overlap means "not shown", not
"equal".

**Counts behind the top-10% and top-20% figures at the test cutoff:**

{{table:counts}}

The yes/no flag `recency_days > {{test.metrics.rule.flag.threshold_days}}` says yes for
{{test.metrics.rule.flag.flagged}} customers ({{test.metrics.rule.flag.flagged_share|pct}});
{{test.metrics.rule.flag.flag_precision|pct}} of them churned, and it catches
{{test.metrics.rule.flag.flag_recall|pct}} of all churners.

### Calibration at the test cutoff: the probabilities are too high

On average the logistic regression predicts **{{gap.logistic.mean_predicted|pct}}** and gradient
boosting **{{gap.gbm.mean_predicted|pct}}**, against an observed **{{base.churn_rate|pct}}**: too
high by {{gap.logistic.gap_points}} and {{gap.gbm.gap_points}} percentage points. In almost every
bin below, fewer customers churned than predicted. The models learned their level on training
months where {{rates.final_training.churn_rate|pct}} churned. The ranking holds up (AUC and top-k
precision above); the probabilities should not be read as churn rates without adjustment
(C5 handles this for revenue at risk).

![Calibration at the test cutoff](../{{outputs.c4_test_calibration_plot}})

{{table:calibration}}

### Why is test churn lower? A hypothesis, not a finding

**TO VERIFY (NOTES.md).** No training cutoff has an outcome window in September-November, the
pre-Christmas buying season for a gift and homeware wholesaler; the test window lies entirely in
it ({{season.test_days}} of its 90 days, against at most {{season.max_training_days}} for any
training cutoff). If customers who would otherwise pause come back to stock up for Christmas,
fewer of them count as churned, which would explain part of the lower test churn rate.

The descriptive check with C2's churn-by-cutoff numbers (`results/c2_churn_by_cutoff.csv`):

{{table:season}}

The churn rate falls as more of the outcome window falls in September-November, which fits the
hypothesis. But churn was already drifting down across the training months, the data hold only
one autumn with a full lookback, and nothing here separates a seasonal effect from a trend or
from other changes in 2011. So it stays a hypothesis.

Files: `{{outputs.c4_test_metrics}}` (the one-shot result), `{{outputs.c4_test_summary}}` (flat
test metrics), `{{outputs.c4_test_calibration_plot}}`.
