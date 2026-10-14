# C5: What drives churn, and what we would have done on {{b.test_cutoff}}

*Generated from `results/c5_drivers.json` and `results/c5_business.json` on {{b.generated_at}}
(git {{b.git.commit}}, uncommitted changes: {{b.git.uncommitted_changes}}). Do not edit by hand;
edit `docs/templates/C5_drivers_and_business.md` and rerun `python -m churn.run_c5`.*

## What this step does, and what it does not touch

1. **Drivers:** which features the main model (gradient boosting) relies on, and plain-language
   statements about how churn differs between groups of customers.
2. **Business layer:** revenue at risk and a costed retention campaign, as if we had run it on
   the test cutoff {{b.test_cutoff}} with the frozen model.

Guardrails:

- The frozen gradient boosting model (`{{b.model.file}}`, sha256 `{{b.model.sha256}}`) is
  only used to predict. Nothing is refitted, tuned or changed, and the one-shot test
  (`churn.run_c4_test`) is not run again. Before anything else, the run checks that the model
  hashes and the test snapshot's sha256 (`{{b.test_snapshot_sha256}}`) match the one-shot test
  record, and that the recomputed PR-AUC and mean probability equal the saved test results.
- Test-cutoff outcomes are used for two things only: permutation importance (it measures how
  much the model's test score drops; nothing is changed afterwards), and after-the-fact checks
  that are labelled as such. The campaign itself uses only what was known on {{b.test_cutoff}}.
- Driver statements are computed on the {{d.training_rows}} training rows (6 training cutoffs),
  never on the test cutoff.
- **All money figures are estimates under the assumptions in `config/business.yaml`**, not
  observed effects. Nothing here shows that an offer causes anyone to stay.

## Part 1: Drivers

### Permutation importance (test cutoff, held-out data)

For each feature, its values are shuffled across the {{b.customers}} test customers and the drop
in the model's PR-AUC is measured (baseline PR-AUC {{b.text.baseline}}),
{{d.permutation.n_repeats}} times with seed {{b.text.seed}}. A larger drop means the model
leans on that feature more.

![Permutation importance](../results/figures/c5_permutation_importance.png)

{{table:importance}}

How to read it:

- The range is the spread over the {{d.permutation.n_repeats}} shuffles. It shows shuffle noise,
  not sampling noise or month-to-month change. A range that includes 0 means "no clear effect".
- **Related features share credit.** Shuffling one of several features that carry the same
  information hurts little, because the model can still use the others. Examples: the
  90/180/365-day order and revenue windows; recency and the gap features; and
  `single_purchase_day_365d`, which says almost the same as `orders_365d` = 1. A low rank
  therefore does not mean "irrelevant to churn". It means the model does not need this feature
  once it has the others.
- Ranks 4 and 5 are practically tied ({{b.text.rank45}}). "Top 4" is a reporting choice, not a sharp line.
- This explains the model, not the customers. It is not a causal ranking.

The top {{d.partial_dependence.top_n}} drivers: {{b.text.top_list}}.

### Partial dependence of the top {{d.partial_dependence.top_n}} drivers

Each curve sets one feature to each grid value for every test customer and averages the
predicted churn probability, holding the customers' other features as they are. Grid: 20 points
from the 5th to the 95th percentile of the test customers' values.

![Partial dependence](../results/figures/c5_partial_dependence.png)

{{table:pdp}}

The probabilities are on the model's own scale, which runs too high at the test cutoff
(docs/C4_models.md). Read the direction and shape, not the level. Partial dependence
averages over customers whose other features may not fit the set value (for example, a long
tenure next to a short recency), so it is a summary of the model, not of real customers.

### Plain-language statements (training cutoffs only)

On the {{d.training_rows}} customer-cutoff rows of the six training cutoffs
({{d.training_cutoffs.0}} to {{d.training_cutoffs.5}}; a customer can appear at several cutoffs):

<!-- statements -->

{{table:statements}}

"Per cutoff" gives the lowest and highest churn rate over the six cutoffs, to show the gap is
stable from month to month. These are associations: for example, frequent buyers churn less,
but nothing here says that making someone order more would keep them.

{{b.text.uncovered}} Statements marked "added after the ranking" were defined after the
importance ranking above had been computed (`config/drivers.yaml`). They use training rows only,
like the rest.

**On the "twice the usual gap" statement:** over all rows it barely separates churners (the
first statement). The comparison group includes the one-day buyers, whose gap is filled with
365 days, so their ratio is small even though they churn most. Among repeat buyers only, the
same statement does separate them.

## Part 2: Business layer on {{b.test_cutoff}}

### Assumptions (`config/business.yaml`)

| Assumption | Central | Scenarios | Source |
|---|---:|---|---|
| Gross margin | {{b.assumptions.margin.central|pct}} | 20%, 30%, 40% | Not in the data; PROJECT_PLAN.md Section 6 (TO VERIFY) |
| Save rate (share of contacted would-be churners kept) | {{b.assumptions.save_rate.central|pct}} | 5%, 10%, 20% | Not observable here; PROJECT_PLAN.md Section 6 (TO VERIFY) |
| Offer cost per contacted customer | £{{b.assumptions.offer_cost_gbp.central}} | £5, £15, £40 | Judgement, adopted by the author (not checked with a retailer), see below |
| Customer value | trailing 12-month net revenue, floored at £0 | | `net_revenue_365d` |

Offer cost: £15 is roughly a 5% discount on a typical order plus an emailed contact. The median
order value at the training cutoffs is {{b.text.median_aov}}. The cost is counted for every
contacted customer, which is cautious, since some never use the offer. £5 = an email with a small
voucher; £40 = a 10% discount plus a staff phone call.

Customer value at the test cutoff: {{b.customers}} active customers, total 12-month net revenue
{{b.text.value_total}}, median {{b.text.value_median}}. {{b.value.floored_customers}} customers
had negative net revenue (returns of earlier purchases, {{b.text.floored_amount}} in total) and
count as £0.

The value assumes a customer who is kept goes on buying at their last-12-month level for the next
12 months. That is generous: "churned" here means no purchase in 90 days, and some of those
customers come back later anyway (TO VERIFY).

### Revenue at risk

Revenue at risk = sum over customers of p(churn) x 12-month value. The C4 test showed the model's
probabilities run too high, so it is shown two ways:

{{table:risk}}

**Why (b) uses only information available on the day.** On {{b.test_cutoff}}, the most recent
cutoff whose 90-day outcome was complete was {{b.reference.cutoff}}: its window ended
{{b.reference.window_end}}, before the decision day. Its churn rate was
{{b.reference.churn_rate|pct}} ({{b.reference.churned}} of {{b.reference.active}}). The code
checks that this window ends on or before the decision day and that no later cutoff qualifies
(2011-07-01's window was still open). Version (b) adds one constant ({{b.text.shift}}) to every
customer's log-odds so the average probability equals that rate. Ranking is unchanged. Only that
one overall rate is used: no customer labels and no refitting.

**What this shows.** The rate known on the day ({{b.reference.churn_rate|pct}}) was close to the
model's own average ({{b.rescale.mean_before|pct}}), so rescaling barely changes anything. The
drop to the {{b.after_the_fact.churn_rate|pct}} actually seen at the test cutoff was not visible
on 2011-09-01 (see the seasonal hypothesis in docs/C4_models.md).

*After the fact only:* the customers who did churn held {{b.text.value_of_churners}} of 12-month
revenue ({{b.after_the_fact.value_share_of_churners|pct}} of the total), less than either
estimate. Both versions overstate the revenue at risk at this cutoff.

### Whom to contact

Expected benefit of contacting a customer = p(churn) x save rate x margin x 12-month value - offer
cost. Customers are ranked by it, and the best campaign contacts everyone with a positive expected
benefit. Within a scenario that is the same as: **contact a customer if p(churn) x value is above
offer cost / (save rate x margin)**. In the central case that bar is {{b.text.threshold_central}} of
expected revenue at risk. Margin and save rate enter only as a product, so for example 20% margin
x 10% save rate gives the same answer as 40% x 5%.

All 27 scenarios (central case in bold; (a) model probabilities, (b) rescaled):

{{table:scenarios}}

The best campaign size depends mostly on the assumptions: from
{{b.best_sizes.b.min}} to {{b.best_sizes.b.max}} customers under (b). The model ranks customers;
the assumptions decide how far down the list it pays to go.

![Net benefit by campaign size](../results/figures/c5_campaign_net_benefit.png)

The line is the central case with rescaled probabilities (b). The band is the lowest to highest
total over all 27 scenarios at each campaign size, each scenario using its own ranking. Every
curve rises while the next customer's expected benefit is positive and falls after that.

**Central case, "what we would have done on {{b.test_cutoff}}" (b):** contact
**{{b.central.b.best_size}}** customers. Offer cost {{b.text.central.b.cost}}; expected revenue kept
{{b.text.central.b.saved}}; expected net benefit (margin on revenue kept minus cost)
**{{b.text.central.b.benefit}}**, an estimate under the assumptions above. With (a) the answer is
{{b.central.a.best_size}} customers and {{b.text.central.a.benefit}}.

**After-the-fact check (not available on the day):** of the {{b.central.b.best_size}} customers
chosen in the central case (b), **{{b.central.b.chosen_churned}} actually churned
({{b.central.b.chosen_churn_rate|pct}})**. Their average predicted probability (b) was
{{b.central.b.chosen_mean_p|pct}}, and the test base rate was {{b.after_the_fact.churn_rate|pct}}.
The list leans towards high-value customers: the chosen customers hold {{b.text.central.b.chosen_value}}
of the {{b.text.value_total}} total. The same number of customers taken by p(churn) alone would
have held {{b.central.b.same_size_by_p_churned}} churners, so ranking by p x value trades
churners for value. Swapping each chosen
customer's predicted probability for what actually happened, with the same assumptions, gives a
net benefit of {{b.text.central.b.hindsight}} instead of {{b.text.central.b.benefit}}. That is
still positive under these assumptions, but the save rate and margin are untested.

### Limitations

- Save rate, margin and offer cost are assumptions, not measurements. Only a controlled campaign
  (contact a random half of the chosen list) could measure the save rate.
- Probabilities ran too high at this cutoff, and the on-the-day rescaling could not see the drop.
- "Churned" means no purchase in 90 days, not lost for good, while value counts 12 months.
- Customers are small businesses of very different sizes. A few large accounts weigh heavily in
  every £ total.
- Old data (2009-2011): the method transfers; the numbers describe this retailer only.

Files: `results/c5_drivers.json`, `results/c5_permutation_importance.csv`,
`results/c5_partial_dependence.csv`, `results/c5_driver_statements.csv`,
`results/c5_business.json`, `results/c5_campaign_scenarios.csv` (all 27 scenarios x 2 versions,
with after-the-fact columns), `results/c5_campaign_curve.csv`, and the three figures in
`results/figures/`.
