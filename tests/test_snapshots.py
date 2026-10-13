"""Snapshot tests: features, labels and the no-leakage check, on a small hand-made table.

Cutoff 2011-03-01 (00:00). Windows before it: 90 days from 2010-12-01, 180 days from 2010-09-02,
365 days from 2010-03-01. Outcome window: [2011-03-01, 2011-05-30).
"""

import pandas as pd
import pytest

from churn.churn_rates import churn_at_cutoff
from churn.config import load_config
from churn.snapshots import (
    FEATURES,
    build_snapshot,
    history_inputs,
    leakage_differences,
    normalise_stock_codes,
    peak_months_inside_lookback,
    snapshot_features,
    stock_code_merges,
    training_cutoffs,
)
from tests.conftest import make_clean

CUTOFF = pd.Timestamp("2011-03-01")
CHURN = load_config("churn")
FEAT = load_config("features")


def line(invoice, customer, date, code, qty, price, country="United Kingdom"):
    return {"invoice": invoice, "customer_id": customer, "invoice_date": date, "stock_code": code,
            "quantity": qty, "price": price, "country": country}


def fixture_lines():
    return make_clean(
        # Customer 1: regular UK buyer.
        line("1001", 1, "2010-02-15 09:00", "X1", 1, 10.0),          # before the lookback (tenure only)
        line("1002", 1, "2010-06-10 09:00", "85123A", 2, 5.0),       # same product as the next line
        line("1002", 1, "2010-06-10 09:00", "85123a ", 1, 10.0),
        line("1003", 1, "2010-11-20 09:00", "Y", 3, 10.0),           # November: peak month
        line("1004", 1, "2011-01-10 09:00", "Z", 4, 10.0),
        line("1005", 1, "2011-01-10 15:00", "W", 1, 10.0),           # same purchase day, second order
        line("C1006", 1, "2011-02-01 09:00", "Z", -1, 10.0),         # partial return
        line("1007", 1, "2011-02-10 10:00", "Q", 100, 1.0),          # cancelled in full the same day
        line("C1008", 1, "2011-02-10 11:00", "Q", -100, 1.0),
        line("1009", 1, "2011-03-01 00:00", "Z", 50, 20.0),          # on the cutoff date: future
        line("C1010", 1, "2011-03-05 09:00", "Y", -3, 10.0),         # later-day mirror of 1003: future
        line("C1011", 1, "2011-03-06 09:00", "Z", -4, 10.0),         # future return of 1004
        # Customer 2: one purchase, in Germany, in December; buys again just inside the window.
        line("2001", 2, "2010-12-24 09:00", "Y", 1, 8.0, "Germany"),
        line("2002", 2, "2011-05-29 23:00", "Y", 99, 8.0, "United Kingdom"),
        # Customer 3: first seen in Dec 2009; latest country UK; next purchase just after the window.
        line("3001", 3, "2009-12-05 09:00", "Y", 1, 8.0, "Germany"),
        line("3002", 3, "2010-04-01 09:00", "Y", 2, 8.0),
        line("3003", 3, "2011-05-30 00:00", "Y", 1, 8.0),
        # Not active at the cutoff: bought only before the lookback, only on the cutoff date,
        # or only an order cancelled in full the same day.
        line("4001", 4, "2010-02-20 09:00", "Y", 1, 8.0),
        line("5001", 5, "2011-03-01 08:00", "Y", 1, 8.0),
        line("6001", 6, "2011-01-05 09:00", "Y", 5, 8.0),
        line("C6002", 6, "2011-01-05 10:00", "Y", -5, 8.0),
    )


def features(lines=None, cutoff=CUTOFF, feat=FEAT):
    lines = fixture_lines() if lines is None else lines
    events, cancels = history_inputs(lines, CHURN["purchase_event_rule"])
    return snapshot_features(lines, events, cancels, cutoff, CHURN, feat)


# --- Who is in the snapshot ------------------------------------------------------------------

def test_rows_are_active_customers_only():
    assert features().index.tolist() == [1, 2, 3]


def test_rows_and_churn_match_descriptive_churn():
    lines = fixture_lines()
    events, cancels = history_inputs(lines, CHURN["purchase_event_rule"])
    snap = build_snapshot(lines, events, cancels, CUTOFF, CHURN, FEAT)
    expected = churn_at_cutoff(events, CUTOFF, CHURN["active_lookback_days"], CHURN["churn_horizon_days"])
    assert len(snap) == expected["active"]
    assert snap["churned"].sum() == expected["churned"]


def test_label_uses_outcome_window_boundaries():
    lines = fixture_lines()
    events, cancels = history_inputs(lines, CHURN["purchase_event_rule"])
    snap = build_snapshot(lines, events, cancels, CUTOFF, CHURN, FEAT).set_index("customer_id")
    # 1 buys on the cutoff date, 2 on the last day inside the window, 3 on the first day after it.
    assert snap["churned"].to_dict() == {1: 0, 2: 0, 3: 1}
    assert (snap["cutoff"] == CUTOFF).all()


def test_snapshot_columns_are_documented_and_complete():
    lines = fixture_lines()
    events, cancels = history_inputs(lines, CHURN["purchase_event_rule"])
    snap = build_snapshot(lines, events, cancels, CUTOFF, CHURN, FEAT)
    assert snap.columns.tolist() == ["cutoff", "customer_id", *FEATURES, "churned"]
    assert all(FEATURES.values())
    assert not snap.isna().any().any()


# --- Feature values (worked out by hand) --------------------------------------------------------

def test_frequency_and_revenue_windows():
    f = features().loc[1]
    assert (f["orders_90d"], f["orders_180d"], f["orders_365d"]) == (2, 3, 4)
    # 90 days: 40 + 10 - 10 (return) + 100 - 100 (same-day cancellation nets to zero).
    assert (f["net_revenue_90d"], f["net_revenue_180d"], f["net_revenue_365d"]) == (40.0, 70.0, 90.0)
    assert f["avg_order_value_365d"] == 25.0  # (20 + 30 + 40 + 10) / 4


def test_trend_is_last_90_days_minus_previous_90():
    f = features().loc[1]
    assert f["orders_trend_90d"] == 1          # 2 orders vs 1
    assert f["net_revenue_trend_90d"] == 10.0  # 40 vs 30


def test_recency_and_gaps_use_distinct_purchase_days():
    f = features().loc[1]
    # Purchase days in the lookback: 2010-06-10, 2010-11-20, 2011-01-10 -> gaps 163 and 51 days.
    assert f["recency_days"] == 50
    assert f["single_purchase_day_365d"] == 0
    assert f["avg_gap_days"] == 107.0
    assert f["latest_gap_days"] == 51
    assert f["latest_gap_ratio"] == pytest.approx(51 / 107)
    assert f["recency_gap_ratio"] == pytest.approx(50 / 107)


def test_single_purchase_day_gets_explicit_values_and_flag():
    f = features().loc[2]
    fill = FEAT["single_purchase_gap_fill_days"]
    assert f["single_purchase_day_365d"] == 1
    assert (f["avg_gap_days"], f["latest_gap_days"], f["latest_gap_ratio"]) == (fill, fill, 1.0)
    assert f["recency_days"] == 67
    assert f["recency_gap_ratio"] == pytest.approx(67 / fill)


def test_tenure_and_first_month_flag():
    f = features()
    assert f.loc[1, "tenure_days"] == 379
    assert f.loc[3, "tenure_days"] == 451
    assert f["first_seen_first_data_month"].to_dict() == {1: 0, 2: 0, 3: 1}


def test_distinct_products_merge_case_and_space_variants():
    assert features().loc[1, "distinct_products_365d"] == 4  # 85123A/85123a merge; Q not a purchase
    raw = features(feat={**FEAT, "stock_code_normalisation": "none"})
    assert raw.loc[1, "distinct_products_365d"] == 5


def test_stock_code_normalisation_and_merge_count():
    codes = pd.Series(["85123A", "85123a", "47503J ", "47503J", "X"])
    assert normalise_stock_codes(codes, "upper_strip").tolist() == ["85123A", "85123A", "47503J", "47503J", "X"]
    assert stock_code_merges(codes, "upper_strip") == 2
    assert stock_code_merges(codes, "none") == 0
    with pytest.raises(ValueError):
        normalise_stock_codes(codes, "lower")


def test_return_rate_excludes_same_day_full_cancellations():
    f = features()
    assert f.loc[1, "return_rate_365d"] == pytest.approx(0.1)  # 10 returned / 100 bought
    assert f.loc[2, "return_rate_365d"] == 0.0


def test_country_uses_latest_line_before_cutoff():
    assert features()["is_uk"].to_dict() == {1: 1, 2: 0, 3: 1}


def test_peak_flag():
    assert features()["bought_in_peak_365d"].to_dict() == {1: 1, 2: 1, 3: 0}


# --- Seasonality window means the same at every cutoff ---------------------------------------

def test_peak_window_at_first_training_cutoff():
    got = peak_months_inside_lookback(pd.Timestamp("2010-12-01"), 365, [11, 12])
    assert got == [pd.Timestamp("2009-12-01"), pd.Timestamp("2010-11-01")]


def test_every_configured_cutoff_holds_one_full_november_and_december():
    cutoffs = training_cutoffs(CHURN) + [pd.Timestamp(CHURN["test_cutoff"])]
    for c in cutoffs:
        months = peak_months_inside_lookback(c, CHURN["active_lookback_days"], FEAT["peak_months"])
        assert sorted(m.month for m in months) == [11, 12], c
        assert min(months) >= pd.Timestamp("2009-12-01")


def test_training_cutoffs_from_config():
    got = training_cutoffs(CHURN)
    assert got[0] == pd.Timestamp("2010-12-01") and got[-1] == pd.Timestamp("2011-05-01")
    assert len(got) == 6
    assert pd.Timestamp(CHURN["test_cutoff"]) not in got


# --- No leakage ------------------------------------------------------------------------------

@pytest.mark.parametrize("cutoff", ["2010-12-01", "2011-01-10", "2011-02-10", "2011-03-01", "2011-04-01"])
def test_features_identical_with_data_after_cutoff_removed(cutoff):
    # Includes cutoffs on a purchase day and on the day of a same-day full cancellation.
    assert leakage_differences(fixture_lines(), pd.Timestamp(cutoff), CHURN, FEAT) == []


def test_leakage_check_catches_missing_history():
    # Sanity check of the check: removing data the features legitimately use must be detected.
    diffs = leakage_differences(fixture_lines(), CUTOFF, CHURN, FEAT,
                                truncate_at=pd.Timestamp("2011-01-10"))
    assert "orders_90d" in diffs and "recency_days" in diffs


def test_future_lines_do_not_change_features():
    base = features()
    poison = make_clean(
        line("9001", 2, "2011-03-01 00:00", "NEW", 1000, 50.0, "France"),   # big order, new country
        line("C9002", 1, "2011-03-02 09:00", "85123A", -2, 5.0),           # return of a past purchase
        line("9003", 3, "2011-06-01 09:00", "Y", 1, 8.0),
    )
    lines = pd.concat([fixture_lines(), poison], ignore_index=True)
    pd.testing.assert_frame_equal(features(lines), base)
