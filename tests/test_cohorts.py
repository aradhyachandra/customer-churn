"""Customer-month, cohort and retention tests on small hand-made tables."""

import pandas as pd

from churn.cohorts import (
    customer_months,
    first_purchase_months,
    retention_counts,
    retention_rates,
    return_only_customers,
)
from churn.events import purchase_events
from tests.conftest import make_clean

RULE = "drop_same_day_full_cancellations"
M = pd.Timestamp


def three_customers():
    # Customer 1: Jan, Feb, Apr.  Customer 2: Jan only (two invoices).  Customer 3: Feb, Mar.
    return make_clean(
        {"invoice": "1", "customer_id": 1, "invoice_date": "2011-01-05", "quantity": 2, "price": 5.0},
        {"invoice": "2", "customer_id": 1, "invoice_date": "2011-02-10", "quantity": 1, "price": 5.0},
        {"invoice": "3", "customer_id": 1, "invoice_date": "2011-04-01", "quantity": 1, "price": 5.0},
        {"invoice": "4", "customer_id": 2, "invoice_date": "2011-01-20", "quantity": 1, "price": 3.0},
        {"invoice": "5", "customer_id": 2, "invoice_date": "2011-01-28", "quantity": 1, "price": 3.0},
        {"invoice": "C6", "customer_id": 2, "invoice_date": "2011-02-02", "stock_code": "Z",
         "quantity": -1, "price": 3.0},
        {"invoice": "7", "customer_id": 3, "invoice_date": "2011-02-01", "quantity": 1, "price": 4.0},
        {"invoice": "8", "customer_id": 3, "invoice_date": "2011-03-31 23:59", "quantity": 1, "price": 4.0},
    )


def test_customer_months_counts_invoices_and_nets_returns():
    df = three_customers()
    cm = customer_months(df, purchase_events(df, RULE)).set_index(["customer_id", "month"])
    jan2 = cm.loc[(2, M("2011-01-01"))]
    assert jan2["purchase_invoices"] == 2
    assert jan2["purchase_value"] == 6.0
    feb2 = cm.loc[(2, M("2011-02-01"))]  # return-only month: a row with no purchase
    assert feb2["purchase_invoices"] == 0
    assert feb2["return_value"] == -3.0
    assert feb2["net_value"] == -3.0
    assert cm.loc[(3, M("2011-03-01")), "purchase_invoices"] == 1  # 23:59 on the last day stays in March
    assert len(cm) == 7


def test_customer_months_net_value_includes_cancelled_purchase():
    df = make_clean(
        {"invoice": "10", "customer_id": 9, "stock_code": "A", "quantity": 5, "invoice_date": "2011-01-03 09:00"},
        {"invoice": "C11", "customer_id": 9, "stock_code": "A", "quantity": -5, "invoice_date": "2011-01-03 09:10"},
        {"invoice": "12", "customer_id": 9, "stock_code": "B", "quantity": 1, "price": 2.0,
         "invoice_date": "2011-01-04 09:00"},
    )
    row = customer_months(df, purchase_events(df, RULE)).iloc[0]
    assert row["purchase_invoices"] == 1      # the cancelled order is not a purchase event
    assert row["net_value"] == 2.0            # but every line is still netted


def test_return_only_customers_are_found():
    df = make_clean(
        {"invoice": "20", "customer_id": 1},
        {"invoice": "C21", "customer_id": 2, "quantity": -1},
        # customer 3's only purchase is cancelled the same day: no purchase event left
        {"invoice": "22", "customer_id": 3, "stock_code": "A", "quantity": 4, "invoice_date": "2011-02-01 10:00"},
        {"invoice": "C23", "customer_id": 3, "stock_code": "A", "quantity": -4, "invoice_date": "2011-02-01 11:00"},
    )
    assert return_only_customers(df, purchase_events(df, RULE)) == [2, 3]


def test_first_purchase_month():
    df = three_customers()
    first = first_purchase_months(purchase_events(df, RULE))
    assert first.to_dict() == {1: M("2011-01-01"), 2: M("2011-01-01"), 3: M("2011-02-01")}


def test_retention_counts_and_rates():
    df = three_customers()
    counts = retention_counts(purchase_events(df, RULE), last_month=M("2011-04-01"))
    # Jan cohort = {1, 2}: month 0: 2, month 1 (Feb): 1, month 2 (Mar): 0, month 3 (Apr): 1
    assert counts.loc[M("2011-01-01")].tolist() == [2, 1, 0, 1]
    # Feb cohort = {3}: month 0: 1, month 1 (Mar): 1, month 2 (Apr): 0, month 3 is past the data
    feb = counts.loc[M("2011-02-01")]
    assert feb.iloc[:3].tolist() == [1, 1, 0]
    assert pd.isna(feb.iloc[3])

    rates = retention_rates(counts)
    assert rates.loc[M("2011-01-01")].tolist() == [1.0, 0.5, 0.0, 0.5]
    assert pd.isna(rates.loc[M("2011-02-01")].iloc[3])


def test_retention_ignores_return_only_customers():
    df = make_clean(
        {"invoice": "30", "customer_id": 1, "invoice_date": "2011-01-05"},
        {"invoice": "C31", "customer_id": 2, "quantity": -1, "invoice_date": "2011-01-06"},
    )
    counts = retention_counts(purchase_events(df, RULE), last_month=M("2011-01-01"))
    assert counts.loc[M("2011-01-01"), 0] == 1
