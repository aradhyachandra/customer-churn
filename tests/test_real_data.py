"""REAL DATA CHECKS. Skipped by default; run with:  pytest -m realdata

These read data/raw/online_retail_II.xlsx (or its parquet cache) and check facts about the real file.
Unit tests elsewhere use only small hand-made tables.
"""

import pytest

from churn.clean import clean_transactions
from churn.config import REPO_ROOT, load_config
from churn.load import load_raw

pytestmark = pytest.mark.realdata


@pytest.fixture(scope="module")
def raw():
    cfg = load_config("data")
    return load_raw(
        REPO_ROOT / cfg["raw"]["path"], cfg["raw"]["sha256"], cfg["raw"]["sheets"],
        REPO_ROOT / cfg["paths"]["cache_dir"],
    )


@pytest.fixture(scope="module")
def cleaned(raw):
    sheets = load_config("data")["raw"]["sheets"]
    cleaning = load_config("cleaning")
    rules = {
        "cancellation_prefix": cleaning["cancellation_prefix"],
        "non_product_stock_codes": [e["code"] for e in cleaning["non_product_stock_codes"]],
    }
    return clean_transactions(raw, sheets, rules)


def test_row_counts_match_uci_total(raw):
    # UCI lists 1,067,371 instances for dataset 502.
    assert raw.groupby("sheet").size().to_dict() == {"Year 2009-2010": 525461, "Year 2010-2011": 541910}


def test_customer_id_is_nullable_integer(raw):
    assert str(raw["customer_id"].dtype) == "Int64"


def test_sheet_overlap_is_a_full_copy(cleaned):
    _, _, summary = cleaned
    ov = summary["overlap"]
    assert ov["later_rows_identical_to_earlier_sheet"] > 0
    assert ov["later_rows_identical_outside_overlap_range"] == 0
    assert ov["earlier_rows_in_range_without_match"] == 0
    assert ov["later_rows_in_range_without_match"] == 0


def test_step_arithmetic(raw, cleaned):
    clean, steps, _ = cleaned
    assert steps[0]["rows_before"] == len(raw)
    assert len(raw) - sum(s["rows_removed"] for s in steps) == len(clean)


def test_clean_data_obeys_rules(cleaned):
    clean, _, _ = cleaned
    assert (clean["price"] > 0).all()
    assert clean["customer_id"].notna().all()
    assert not clean.duplicated(subset=["invoice", "stock_code", "description", "quantity",
                                        "invoice_date", "price", "customer_id", "country"]).any()
