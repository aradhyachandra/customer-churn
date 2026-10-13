"""REAL DATA CHECKS for C2. Skipped by default; run with:  pytest -m realdata

Read data/processed/transactions_clean.parquet (build it with: python -m churn.run_c1).
"""

import pandas as pd
import pytest

from churn.churn_rates import churn_table, valid_monthly_cutoffs
from churn.cohorts import retention_counts, return_only_customers
from churn.config import REPO_ROOT, load_config
from churn.events import purchase_events

pytestmark = pytest.mark.realdata


@pytest.fixture(scope="module")
def clean():
    path = REPO_ROOT / load_config("data")["paths"]["clean_transactions"]
    if not path.exists():
        pytest.skip("run python -m churn.run_c1 first")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def events(clean):
    return purchase_events(clean, load_config("churn")["purchase_event_rule"])


def test_known_full_cancellations_are_not_events(events):
    # NOTES.md: the two largest lines in the clean data are cancelled by the next invoice.
    assert not events["invoice"].isin(["581483", "541431"]).any()


def test_every_customer_is_in_a_cohort_or_return_only(clean, events):
    last_month = clean["invoice_date"].max().to_period("M").to_timestamp()
    counts = retention_counts(events, last_month)
    n_return_only = len(return_only_customers(clean, events))
    assert counts[0].sum() + n_return_only == clean["customer_id"].nunique()


def test_churn_arithmetic_and_cutoffs(clean, events):
    first, last = clean["invoice_date"].min(), clean["invoice_date"].max()
    cutoffs = valid_monthly_cutoffs(first, last, 365, 90)
    assert pd.Timestamp("2011-09-01") in cutoffs  # the plan's test cutoff fits
    table = churn_table(events, cutoffs, 365, 90)
    assert (table["churned"] + table["retained"] == table["active"]).all()
    assert table["churn_rate"].between(0, 1).all()
