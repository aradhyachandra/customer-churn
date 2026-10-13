"""Purchase-event tests: which purchase invoices count, on small hand-made tables."""

import pytest

from churn.events import cancellation_impact, full_cancellation_pairs, purchase_events
from tests.conftest import make_clean


def big_order_cancelled_same_day():
    # Like customer 16446: one large line, reversed by the next invoice minutes later.
    return make_clean(
        {"invoice": "100", "customer_id": 1, "stock_code": "A", "quantity": 500, "price": 2.0,
         "invoice_date": "2011-01-18 10:01"},
        {"invoice": "C101", "customer_id": 1, "stock_code": "A", "quantity": -500, "price": 2.0,
         "invoice_date": "2011-01-18 10:17"},
        {"invoice": "102", "customer_id": 1, "stock_code": "B", "quantity": 3, "price": 1.0,
         "invoice_date": "2011-03-01 09:00"},
    )


def test_same_day_full_cancellation_is_paired():
    pairs = full_cancellation_pairs(big_order_cancelled_same_day(), same_day_only=True)
    assert pairs[["purchase_invoice", "cancel_invoice"]].values.tolist() == [["100", "C101"]]
    assert pairs["value"].tolist() == [1000.0]


def test_rule_drops_same_day_full_cancellation():
    events = purchase_events(big_order_cancelled_same_day(), "drop_same_day_full_cancellations")
    assert events["invoice"].tolist() == ["102"]


def test_rule_all_purchase_invoices_keeps_everything():
    events = purchase_events(big_order_cancelled_same_day(), "all_purchase_invoices")
    assert sorted(events["invoice"]) == ["100", "102"]


def test_unknown_rule_fails():
    with pytest.raises(ValueError, match="no_such_rule"):
        purchase_events(big_order_cancelled_same_day(), "no_such_rule")


def test_events_have_one_row_per_invoice_with_value():
    df = make_clean(
        {"invoice": "200", "stock_code": "A", "quantity": 2, "price": 1.5},
        {"invoice": "200", "stock_code": "B", "quantity": 1, "price": 4.0},
    )
    events = purchase_events(df, "drop_same_day_full_cancellations")
    assert len(events) == 1
    assert events.iloc[0]["value"] == 7.0
    assert set(events.columns) >= {"invoice", "customer_id", "invoice_date", "value"}


def test_partial_cancellation_does_not_remove_purchase():
    df = make_clean(
        {"invoice": "300", "stock_code": "A", "quantity": 5, "price": 1.0},
        {"invoice": "300", "stock_code": "B", "quantity": 5, "price": 1.0},
        {"invoice": "C301", "stock_code": "A", "quantity": -5, "price": 1.0},
    )
    assert full_cancellation_pairs(df, same_day_only=False).empty
    assert purchase_events(df, "drop_same_day_full_cancellations")["invoice"].tolist() == ["300"]


def test_different_quantity_or_price_or_customer_is_not_a_match():
    base = {"invoice": "400", "stock_code": "A", "quantity": 5, "price": 1.0, "customer_id": 1}
    for change in [{"quantity": -4}, {"price": 1.1}, {"customer_id": 2}]:
        df = make_clean(base, {**base, "invoice": "C401", "quantity": -5, **change})
        assert full_cancellation_pairs(df, same_day_only=False).empty, change


def test_later_day_mirror_is_measured_but_not_dropped():
    df = make_clean(
        {"invoice": "500", "stock_code": "A", "quantity": 5, "invoice_date": "2011-01-03 10:00"},
        {"invoice": "C501", "stock_code": "A", "quantity": -5, "invoice_date": "2011-01-10 10:00"},
    )
    assert full_cancellation_pairs(df, same_day_only=True).empty
    assert len(full_cancellation_pairs(df, same_day_only=False)) == 1
    assert purchase_events(df, "drop_same_day_full_cancellations")["invoice"].tolist() == ["500"]


def test_cancellation_before_purchase_is_not_a_match():
    df = make_clean(
        {"invoice": "C600", "stock_code": "A", "quantity": -5, "invoice_date": "2011-01-03 09:00"},
        {"invoice": "601", "stock_code": "A", "quantity": 5, "invoice_date": "2011-01-03 10:00"},
    )
    assert full_cancellation_pairs(df, same_day_only=False).empty


def test_matching_is_one_to_one():
    # Two identical orders, one cancellation: only one order is removed (the later one).
    df = make_clean(
        {"invoice": "700", "stock_code": "A", "quantity": 5, "invoice_date": "2011-01-03 09:00"},
        {"invoice": "701", "stock_code": "A", "quantity": 5, "invoice_date": "2011-01-03 10:00"},
        {"invoice": "C702", "stock_code": "A", "quantity": -5, "invoice_date": "2011-01-03 11:00"},
    )
    pairs = full_cancellation_pairs(df, same_day_only=True)
    assert pairs["purchase_invoice"].tolist() == ["701"]
    assert purchase_events(df, "drop_same_day_full_cancellations")["invoice"].tolist() == ["700"]


def test_line_order_within_invoice_does_not_matter():
    df = make_clean(
        {"invoice": "800", "stock_code": "A", "quantity": 1},
        {"invoice": "800", "stock_code": "B", "quantity": 2},
        {"invoice": "C801", "stock_code": "B", "quantity": -2},
        {"invoice": "C801", "stock_code": "A", "quantity": -1},
    )
    assert len(full_cancellation_pairs(df, same_day_only=True)) == 1


def test_cancellation_impact_counts():
    df = big_order_cancelled_same_day()
    impact = cancellation_impact(df, "drop_same_day_full_cancellations")
    assert impact["purchase_invoices"] == 2
    assert impact["same_day"]["purchase_invoices"] == 1
    assert impact["same_day"]["customers"] == 1
    assert impact["same_day"]["value"] == 1000.0
    assert impact["any_later_date"]["purchase_invoices"] == 1
    assert impact["chosen_rule"]["customers_first_purchase_month_changed"] == 1
    assert impact["chosen_rule"]["customers_left_with_no_purchase"] == 0
