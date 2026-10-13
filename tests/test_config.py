"""Config tests: every rule and code lives in config/ with a rationale."""

from churn.config import load_config

PLAN_EXAMPLE_CODES = {"POST", "DOT", "M", "BANK CHARGES", "D", "AMAZONFEE", "ADJUST"}


def test_raw_checksum_recorded():
    data = load_config("data")
    assert data["raw"]["sha256"] == (
        "bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980"
    )
    assert data["raw"]["sheets"] == ["Year 2009-2010", "Year 2010-2011"]


def test_non_product_codes_cover_plan_examples_and_have_reasons():
    cleaning = load_config("cleaning")
    entries = cleaning["non_product_stock_codes"]
    codes = [e["code"] for e in entries]
    assert PLAN_EXAMPLE_CODES <= set(codes)
    assert len(codes) == len(set(codes))
    assert all(e.get("reason") for e in entries)


def test_every_cleaning_rule_has_a_rationale():
    cleaning = load_config("cleaning")
    for key, value in cleaning.items():
        if key.endswith("_rationale"):
            continue
        assert cleaning.get(f"{key}_rationale"), f"{key} has no rationale"


def test_every_churn_setting_has_a_rationale():
    churn = load_config("churn")
    for key in churn:
        if not key.endswith("_rationale"):
            assert churn.get(f"{key}_rationale"), f"{key} has no rationale"


def test_churn_definitions_match_plan():
    churn = load_config("churn")
    assert churn["active_lookback_days"] == 365
    assert churn["churn_horizon_days"] == 90
    assert churn["sensitivity_horizons_days"] == [60, 120]
    assert churn["test_cutoff"] == "2011-09-01"
    assert churn["sensitivity_cutoff"] != churn["test_cutoff"]
    assert churn["purchase_event_rule"] in {"drop_same_day_full_cancellations", "all_purchase_invoices"}


def test_training_cutoffs_have_full_lookback_and_precede_test():
    import pandas as pd
    churn = load_config("churn")
    first = pd.Timestamp(churn["training_cutoffs"]["first"])
    last = pd.Timestamp(churn["training_cutoffs"]["last"])
    assert first - pd.Timedelta(days=churn["active_lookback_days"]) >= pd.Timestamp("2009-12-01")
    assert last + pd.Timedelta(days=churn["churn_horizon_days"]) <= pd.Timestamp(churn["test_cutoff"])
