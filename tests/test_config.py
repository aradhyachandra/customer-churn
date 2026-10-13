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
