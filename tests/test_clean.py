"""Cleaning-rule tests on small hand-made tables (see conftest.make_lines)."""

import pandas as pd
import pytest

from churn.clean import (
    clean_transactions,
    is_return,
    line_value,
    missing_customer_mask,
    non_positive_price_mask,
    non_product_mask,
    sheet_overlap,
    within_sheet_duplicates_mask,
)
from tests.conftest import make_lines

A, B = "Year 2009-2010", "Year 2010-2011"
SHEETS = [A, B]
RULES = {"cancellation_prefix": "C", "non_product_stock_codes": ["POST", "M", "BANK CHARGES"]}


# --- line value ---------------------------------------------------------------

def test_line_value_is_quantity_times_price():
    df = make_lines({"quantity": 3, "price": 2.5}, {"quantity": -2, "price": 1.0})
    assert list(line_value(df)) == [7.5, -2.0]


# --- sheet overlap -------------------------------------------------------------

def overlap_fixture():
    """Sheet A ends 2010-12-09, sheet B starts 2010-12-01; two lines appear in both."""
    return make_lines(
        {"sheet": A, "invoice": "1", "invoice_date": "2010-11-30 09:00"},
        {"sheet": A, "invoice": "2", "invoice_date": "2010-12-02 09:00"},  # also in B
        {"sheet": A, "invoice": "3", "invoice_date": "2010-12-09 09:00"},  # also in B
        {"sheet": A, "invoice": "4", "invoice_date": "2010-12-05 09:00"},  # only in A
        {"sheet": B, "invoice": "2", "invoice_date": "2010-12-02 09:00"},
        {"sheet": B, "invoice": "3", "invoice_date": "2010-12-09 09:00"},
        {"sheet": B, "invoice": "5", "invoice_date": "2010-12-01 09:00"},  # only in B
        {"sheet": B, "invoice": "6", "invoice_date": "2010-12-20 09:00"},
    )


def test_overlap_reports_date_ranges():
    stats, _ = sheet_overlap(overlap_fixture(), SHEETS)
    assert stats["sheets"][A]["first_date"] == "2010-11-30 09:00:00"
    assert stats["sheets"][A]["last_date"] == "2010-12-09 09:00:00"
    assert stats["sheets"][B]["first_date"] == "2010-12-01 09:00:00"
    assert stats["overlap_date_range"] == {
        "start": "2010-12-01 09:00:00",
        "end": "2010-12-09 09:00:00",
    }


def test_overlap_counts_identical_rows_and_unmatched_rows():
    stats, drop = sheet_overlap(overlap_fixture(), SHEETS)
    assert stats["later_rows_identical_to_earlier_sheet"] == 2
    assert stats["later_rows_identical_outside_overlap_range"] == 0
    assert stats["rows_in_overlap_range"] == {A: 3, B: 3}
    assert stats["earlier_rows_in_range_without_match"] == 1  # invoice 4
    assert stats["later_rows_in_range_without_match"] == 1  # invoice 5
    # Only the later sheet's copies are dropped.
    df = overlap_fixture()
    assert list(df.loc[drop, "sheet"]) == [B, B]
    assert sorted(df.loc[drop, "invoice"]) == ["2", "3"]


def test_overlap_rows_kept_apart_from_within_sheet_duplicates():
    # Line "2" appears twice in A (a genuine within-sheet duplicate) and once in B (overlap).
    df = make_lines(
        {"sheet": A, "invoice": "2", "invoice_date": "2010-12-02 09:00"},
        {"sheet": A, "invoice": "2", "invoice_date": "2010-12-02 09:00"},
        {"sheet": B, "invoice": "2", "invoice_date": "2010-12-02 09:00"},
    )
    stats, drop = sheet_overlap(df, SHEETS)
    assert stats["later_rows_identical_to_earlier_sheet"] == 1
    assert stats["sheets"][A]["within_sheet_duplicate_rows"] == 1
    assert stats["sheets"][B]["within_sheet_duplicate_rows"] == 0
    assert drop.sum() == 1


def test_overlap_matches_missing_customer_ids():
    df = make_lines(
        {"sheet": A, "customer_id": None, "invoice_date": "2010-12-02 09:00"},
        {"sheet": B, "customer_id": None, "invoice_date": "2010-12-02 09:00"},
    )
    stats, _ = sheet_overlap(df, SHEETS)
    assert stats["later_rows_identical_to_earlier_sheet"] == 1


def test_overlap_rows_differing_in_one_field_are_not_identical():
    df = make_lines(
        {"sheet": A, "quantity": 6, "invoice_date": "2010-12-02 09:00"},
        {"sheet": B, "quantity": 7, "invoice_date": "2010-12-02 09:00"},
    )
    stats, drop = sheet_overlap(df, SHEETS)
    assert stats["later_rows_identical_to_earlier_sheet"] == 0
    assert not drop.any()


# --- single rules ---------------------------------------------------------------

def test_within_sheet_duplicates_keep_first_copy():
    df = make_lines({}, {}, {"quantity": 7})
    assert list(within_sheet_duplicates_mask(df)) == [False, True, False]


def test_zero_and_negative_prices_are_flagged():
    df = make_lines({"price": 0.0}, {"price": -11062.06}, {"price": 0.01})
    assert list(non_positive_price_mask(df)) == [True, True, False]


def test_non_product_codes_match_exactly():
    df = make_lines(
        {"stock_code": "POST"},
        {"stock_code": "BANK CHARGES"},
        {"stock_code": "M"},
        {"stock_code": "85123A"},
        {"stock_code": "POSTX"},
    )
    assert list(non_product_mask(df, RULES["non_product_stock_codes"])) == [
        True, True, True, False, False,
    ]


def test_missing_customer_is_flagged():
    df = make_lines({"customer_id": None}, {"customer_id": 12346})
    assert list(missing_customer_mask(df)) == [True, False]


def test_returns_are_cancellations_or_negative_quantities():
    df = make_lines(
        {"invoice": "C500001", "quantity": -2},
        {"invoice": "500002", "quantity": -1},
        {"invoice": "500003", "quantity": 4},
    )
    assert list(is_return(df, "C")) == [True, True, False]


# --- full sequence --------------------------------------------------------------

def pipeline_fixture():
    """One row per rule, plus rows that two rules would both catch."""
    return make_lines(
        {"sheet": A, "invoice": "1", "invoice_date": "2010-12-02 09:00"},  # kept
        {"sheet": B, "invoice": "1", "invoice_date": "2010-12-02 09:00"},  # overlap
        {"sheet": A, "invoice": "2"},  # kept
        {"sheet": A, "invoice": "2"},  # within-sheet duplicate
        {"sheet": A, "invoice": "3", "price": 0.0, "customer_id": None},  # price (first match)
        {"sheet": A, "invoice": "4", "stock_code": "POST", "price": 18.0},  # non-product
        {"sheet": A, "invoice": "5", "customer_id": None, "quantity": 10},  # no customer
        {"sheet": A, "invoice": "C6", "quantity": -2},  # return, kept
    )


def test_pipeline_step_counts_and_order():
    clean, steps, _ = clean_transactions(pipeline_fixture(), SHEETS, RULES)
    removed = {s["step"]: s["rows_removed"] for s in steps}
    assert removed == {
        "drop_sheet_overlap": 1,
        "drop_within_sheet_duplicates": 1,
        "drop_non_positive_price": 1,
        "drop_non_product_codes": 1,
        "drop_missing_customer_id": 1,
    }
    assert len(clean) == 3
    assert sorted(clean["invoice"]) == ["1", "2", "C6"]


def test_pipeline_step_arithmetic_is_consistent():
    raw = pipeline_fixture()
    clean, steps, _ = clean_transactions(raw, SHEETS, RULES)
    assert steps[0]["rows_before"] == len(raw)
    for prev, nxt in zip(steps, steps[1:]):
        assert nxt["rows_before"] == prev["rows_after"]
    for s in steps:
        assert s["rows_after"] == s["rows_before"] - s["rows_removed"]
    assert steps[-1]["rows_after"] == len(clean)


def test_pipeline_reports_rows_removed_by_sheet():
    _, steps, _ = clean_transactions(pipeline_fixture(), SHEETS, RULES)
    by_step = {s["step"]: s for s in steps}
    assert by_step["drop_sheet_overlap"]["rows_removed_by_sheet"] == {A: 0, B: 1}
    assert by_step["drop_within_sheet_duplicates"]["rows_removed_by_sheet"] == {A: 1, B: 0}
    for s in steps:
        assert sum(s["rows_removed_by_sheet"].values()) == s["rows_removed"]


def test_pipeline_reports_value_removed():
    _, steps, _ = clean_transactions(pipeline_fixture(), SHEETS, RULES)
    by_step = {s["step"]: s for s in steps}
    assert by_step["drop_non_product_codes"]["net_value_removed"] == pytest.approx(6 * 18.0)
    assert by_step["drop_missing_customer_id"]["net_value_removed"] == pytest.approx(10 * 2.55)
    assert by_step["drop_non_positive_price"]["net_value_removed"] == 0.0


def test_pipeline_keeps_returns_and_flags_them():
    clean, _, summary = clean_transactions(pipeline_fixture(), SHEETS, RULES)
    assert clean.loc[clean["invoice"] == "C6", "is_return"].item()
    assert not clean.loc[clean["invoice"] == "1", "is_return"].item()
    assert summary["returns"]["rows"] == 1
    assert summary["returns"]["net_value"] == pytest.approx(-2 * 2.55)


def test_pipeline_adds_line_value_and_does_not_mutate_input():
    raw = pipeline_fixture()
    before = raw.copy()
    clean, _, _ = clean_transactions(raw, SHEETS, RULES)
    pd.testing.assert_frame_equal(raw, before)
    assert (clean["line_value"] == clean["quantity"] * clean["price"]).all()
