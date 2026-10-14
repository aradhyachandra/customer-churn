"""Churn and retention dashboard. Run with:  streamlit run app/streamlit_app.py

Reads committed files only (results/ and config/). No raw data, snapshot or model is needed, and
nothing is refitted. All money figures are ESTIMATES under the assumptions chosen on the page.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))      # works without `pip install -e .`

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from churn.dashboard import campaign_estimate, load_dashboard_data, unusual_phrases  # noqa: E402

st.set_page_config(page_title="Customer churn and retention", layout="wide")


@st.cache_data
def data():
    d = load_dashboard_data(ROOT)
    specs = d["report_config"]["profile_features"]
    d["scores"]["unusual"] = unusual_phrases(d["scores"], specs, d["report_config"]["profile_count"])
    return d


def gbp(x):
    return f"{'-' if x < 0 else ''}£{abs(x):,.0f}"


d = data()
business, cfg = d["business"], d["business_config"]
decision_day = business["test_cutoff"]

st.title("Customer churn and retention")
st.caption(f"UK online retailer (UCI Online Retail II, CC BY 4.0). Risk scores and the campaign are as on "
           f"{decision_day}, from the frozen gradient boosting model. Money figures are estimates under "
           "stated assumptions, not observed effects. Full write-up: docs/REPORT.md.")

retention_tab, segment_tab, risk_tab, calc_tab = st.tabs(
    ["Retention", "Churn by segment", f"Risk list at {decision_day}", "Campaign calculator"])

# --- Retention ----------------------------------------------------------------------------------
with retention_tab:
    st.subheader("Cohort retention")
    st.write("Each row is the month of a customer's first purchase; each cell the share of that cohort "
             "buying again *k* months later. Dec 2009 includes customers who first bought before the data "
             "starts; the last data month (Dec 2011) is only partly covered.")
    sizes = d["cohort_summary"].set_index("cohort")["customers"]
    long = (d["retention_rates"].reset_index().melt(id_vars="cohort", var_name="month", value_name="rate")
            .dropna().assign(month=lambda x: x["month"].astype(int), customers=lambda x: x["cohort"].map(sizes)))
    long = long[long["month"] > 0]
    heat = alt.Chart(long).mark_rect().encode(
        x=alt.X("month:O", title="Months after first purchase"),
        y=alt.Y("cohort:O", title="First-purchase month"),
        color=alt.Color("rate:Q", title="Buying again", scale=alt.Scale(scheme="blues"),
                        legend=alt.Legend(format=".0%")),
        tooltip=[alt.Tooltip("cohort:O", title="Cohort"), alt.Tooltip("customers:Q", title="Cohort size"),
                 alt.Tooltip("month:O", title="Month"), alt.Tooltip("rate:Q", title="Buying again", format=".1%")])
    st.altair_chart(heat, width="stretch")
    with st.expander("Table"):
        st.dataframe(d["retention_rates"].style.format("{:.1%}", na_rep=""), width="stretch")

# --- Churn by segment ---------------------------------------------------------------------------
with segment_tab:
    st.subheader("Churn by segment (training cutoffs only)")
    st.write("Share of active customers with no purchase in the next 90 days, pooled over the six "
             "training cutoffs (2010-12-01 to 2011-05-01). The test cutoff is not used here. "
             "These are associations, not causes.")
    seg = d["segments"]
    for title, g in seg.groupby("title", sort=False):
        bars = alt.Chart(g).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("churn_rate:Q", title="Churn rate", axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("group:N", title=None, sort=list(g["group"])),
            tooltip=[alt.Tooltip("group:N", title="Group"), alt.Tooltip("rows:Q", title="Customer-cutoff rows", format=","),
                     alt.Tooltip("churn_rate:Q", title="Churn rate", format=".1%"),
                     alt.Tooltip("churn_rate_min:Q", title="Lowest cutoff", format=".1%"),
                     alt.Tooltip("churn_rate_max:Q", title="Highest cutoff", format=".1%")])
        st.markdown(f"**{title}**")
        st.altair_chart(bars.properties(height=34 * len(g)), width="stretch")
    st.markdown("**Churn rate at each cutoff**")
    cut = d["churn_by_cutoff"].assign(role=lambda x: x["is_test_cutoff"].map({True: "test", False: "other"}))
    st.dataframe(cut[["cutoff", "active", "churned", "churn_rate", "role"]]
                 .style.format({"churn_rate": "{:.1%}", "active": "{:,}", "churned": "{:,}"}), hide_index=True)
    with st.expander("Plain-language driver statements (training rows)"):
        st.dataframe(d["statements"][["group", "condition", "rows_in", "churn_in", "churn_out", "ratio"]]
                     .style.format({"churn_in": "{:.1%}", "churn_out": "{:.1%}", "ratio": "{:.2f}", "rows_in": "{:,}"}),
                     hide_index=True, width="stretch")

# --- Risk list ----------------------------------------------------------------------------------
with risk_tab:
    st.subheader(f"Customers at risk on {decision_day}")
    st.write("Ranked by expected benefit of an offer in the central case (margin "
             f"{cfg['margin']['central']:.0%}, save rate {cfg['save_rate']['central']:.0%}, offer "
             f"£{cfg['offer_cost_gbp']['central']}). *p(churn)* is the model's score rescaled to the churn rate "
             "known on the day. **What stands out** compares each customer with the median active customer; "
             "it describes the customer, it is not the model's reason for the score.")
    s = d["scores"]
    c1, c2, c3 = st.columns(3)
    only_list = c1.checkbox("Only the central-case list", value=True)
    min_value = c2.number_input("Minimum 12-month revenue (£)", min_value=0, value=0, step=100)
    show_outcome = c3.checkbox("Show what actually happened (after the fact)", value=False,
                               help="Not known on the decision day. Shown only to judge the scores.")
    view = s[(s["in_central_list"] | (not only_list)) & (s["value"] >= min_value)]
    table = pd.DataFrame({
        "Rank": view["rank_central"], "Customer": view["customer_id"].astype(str),
        "p(churn)": view["p_rescaled"], "12-month revenue": view["value"],
        "Expected benefit (central, est.)": view["expected_benefit_central"],
        "Days since last purchase": view["recency_days"], "Orders past year": view["orders_365d"],
        "What stands out": view["unusual"]})
    fmt = {"p(churn)": "{:.0%}", "12-month revenue": "£{:,.0f}", "Expected benefit (central, est.)": "£{:,.0f}"}
    if show_outcome:
        table["Churned (after the fact)"] = view["churned_after_the_fact"].map({1: "yes", 0: "no"})
        st.warning("After-the-fact column: these outcomes were unknown on the decision day.")
    st.caption(f"{len(view):,} customers shown")
    st.dataframe(table.style.format(fmt), hide_index=True, width="stretch", height=520)

# --- Campaign calculator ------------------------------------------------------------------------
with calc_tab:
    st.subheader("Campaign calculator (estimates)")
    st.write("Choose the assumptions; the calculator contacts customers in order of expected benefit "
             "(p(churn) x save rate x margin x 12-month revenue - offer cost) and stops where the total is "
             "highest. None of these assumptions is in the data.")
    a, b, c, v = st.columns(4)
    margin = a.slider("Gross margin", 0.10, 0.60, float(cfg["margin"]["central"]), 0.05, format="%.2f", key="margin")
    save = b.slider("Save rate", 0.01, 0.30, float(cfg["save_rate"]["central"]), 0.01, format="%.2f", key="save")
    cost = c.slider("Offer cost (£ per contact)", 1, 60, int(cfg["offer_cost_gbp"]["central"]), 1, key="cost")
    version = v.radio("Probabilities", ["Rescaled (b)", "Model as is (a)"], key="version")
    p = s["p_rescaled"] if version.startswith("Rescaled") else s["p_model"]
    r = campaign_estimate(p, s["value"], margin, save, cost)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best number to contact", f"{r['best_size']:,}")
    m2.metric("Expected net benefit (estimate)", gbp(r["total_benefit"]))
    m3.metric("Offer spend", gbp(r["offer_cost_total"]))
    m4.metric("Expected revenue kept (estimate)", gbp(r["expected_saved_revenue"]))
    st.caption(f"Contact a customer if p(churn) x 12-month revenue is above {gbp(r['value_threshold'])}. "
               f"Customers: {business['customers']:,}. Grid points match results/c5_campaign_scenarios.csv.")
    curve = pd.DataFrame({"Customers contacted": range(len(r["curve"])), "Expected net benefit (£)": r["curve"]})
    line = alt.Chart(curve).mark_line(strokeWidth=2).encode(
        x="Customers contacted:Q", y=alt.Y("Expected net benefit (£):Q"),
        tooltip=[alt.Tooltip("Customers contacted:Q", format=","),
                 alt.Tooltip("Expected net benefit (£):Q", format=",.0f")])
    best = alt.Chart(pd.DataFrame({"x": [r["best_size"]]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")
    st.altair_chart(line + best, width="stretch")
    with st.expander("After the fact (not known on the decision day)"):
        chosen = r["chosen"]
        y = s["churned_after_the_fact"].to_numpy()
        hindsight = float((y[chosen] * save * margin * s["value"].to_numpy()[chosen]).sum() - r["best_size"] * cost)
        churned = int(y[chosen].sum())
        rate = churned / r["best_size"] if r["best_size"] else 0.0
        st.write(f"Of the {r['best_size']:,} chosen customers, {churned:,} ({rate:.1%}) actually churned, "
                 f"against a mean predicted {p[chosen].mean() if r['best_size'] else 0:.1%}. With actual outcomes "
                 f"in place of probabilities, the same assumptions give {gbp(hindsight)}.")
