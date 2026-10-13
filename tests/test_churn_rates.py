"""Descriptive churn tests: active and churned customers at a cutoff, and which cutoffs fit the data."""

import pandas as pd
import pytest

from churn.churn_rates import churn_at_cutoff, churn_table, valid_monthly_cutoffs

T = pd.Timestamp
CUTOFF = T("2011-01-01")


def events(*rows):
    """rows: (customer_id, invoice_date)."""
    return pd.DataFrame({
        "invoice": [str(i) for i in range(len(rows))],
        "customer_id": pd.array([r[0] for r in rows], dtype="Int64"),
        "invoice_date": pd.to_datetime([r[1] for r in rows], format="ISO8601"),
        "value": 1.0,
    })


def test_active_and_churned():
    ev = events(
        (1, "2010-06-01"), (1, "2011-02-01"),   # active, buys again within 90 days: retained
        (2, "2010-12-31 23:59"),                # active, never again: churned
        (3, "2009-06-01"),                      # last purchase > 365 days before: not active
        (4, "2011-01-15"),                      # first purchase after cutoff: not active
    )
    r = churn_at_cutoff(ev, CUTOFF, lookback_days=365, horizon_days=90)
    assert r == {"cutoff": "2011-01-01", "active": 2, "churned": 1, "retained": 1, "churn_rate": 0.5}


def test_lookback_boundaries():
    # Lookback is [cutoff - 365 days, cutoff): the start is included, the cutoff itself is not.
    ev = events((1, "2010-01-01 00:00"), (2, "2009-12-31 23:59"), (3, "2011-01-01 00:00"))
    assert churn_at_cutoff(ev, CUTOFF, 365, 90)["active"] == 1


def test_outcome_window_boundaries():
    # Outcome window is [cutoff, cutoff + 90 days): a purchase on the cutoff day counts as
    # retained; one exactly 90 days later is outside the window, so that customer churned.
    ev = events(
        (1, "2010-06-01"), (1, "2011-01-01 00:00"),
        (2, "2010-06-01"), (2, "2011-04-01 00:00"),
        (3, "2010-06-01"), (3, "2011-03-31 23:59"),
    )
    r = churn_at_cutoff(ev, CUTOFF, 365, 90)
    assert (r["active"], r["churned"]) == (3, 1)


def test_horizon_changes_churn():
    ev = events((1, "2010-06-01"), (1, "2011-03-15"))  # buys again 73 days after the cutoff
    assert churn_at_cutoff(ev, CUTOFF, 365, 60)["churned"] == 1
    assert churn_at_cutoff(ev, CUTOFF, 365, 90)["churned"] == 0


def test_no_active_customers_gives_no_rate():
    r = churn_at_cutoff(events((1, "2005-01-01")), CUTOFF, 365, 90)
    assert r["active"] == 0 and r["churn_rate"] is None


def test_valid_monthly_cutoffs_need_full_windows():
    # Data from 2009-12-01 07:45 to 2011-12-09 12:50, as in the real file.
    cutoffs = valid_monthly_cutoffs(T("2009-12-01 07:45"), T("2011-12-09 12:50"), 365, 90)
    assert cutoffs[0] == T("2010-12-01")   # 2010-12-01 minus 365 days = 2009-12-01
    assert cutoffs[-1] == T("2011-09-01")  # plus 90 days = 2011-11-30, before 2011-12-09
    assert len(cutoffs) == 10
    assert all(c.day == 1 for c in cutoffs)


def test_valid_cutoffs_exclude_partial_first_and_last_days():
    # Data starting after midnight on 2009-12-01 still covers that day for the lookback start;
    # the last day (partly covered) cannot be inside an outcome window.
    cutoffs = valid_monthly_cutoffs(T("2009-12-01 07:45"), T("2010-12-30 12:00"), 365, 29)
    assert cutoffs == [T("2010-12-01")]  # ends 2010-12-30 00:00, start of the last data day
    assert valid_monthly_cutoffs(T("2009-12-01 07:45"), T("2010-12-29 23:00"), 365, 29) == []


def test_churn_table_one_row_per_cutoff():
    ev = events((1, "2010-06-01"), (2, "2010-11-01"), (2, "2011-03-01"))
    table = churn_table(ev, [T("2011-01-01"), T("2011-02-01")], 365, 90)
    assert table["cutoff"].tolist() == ["2011-01-01", "2011-02-01"]
    assert table["active"].tolist() == [2, 2]
    assert table["churned"].tolist() == [1, 1]
    assert table["churn_rate"].tolist() == pytest.approx([0.5, 0.5])
