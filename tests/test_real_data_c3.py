"""REAL DATA CHECKS for C3. Skipped by default; run with:  pytest -m realdata

Read data/processed/transactions_clean.parquet (build it with: python -m churn.run_c1).
"""

import pandas as pd
import pytest

from churn.churn_rates import churn_at_cutoff
from churn.config import REPO_ROOT, load_config
from churn.snapshots import build_snapshot, history_inputs, leakage_differences, training_cutoffs

pytestmark = pytest.mark.realdata

CHURN = load_config("churn")
FEAT = load_config("features")


@pytest.fixture(scope="module")
def clean():
    path = REPO_ROOT / load_config("data")["paths"]["clean_transactions"]
    if not path.exists():
        pytest.skip("run python -m churn.run_c1 first")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def inputs(clean):
    return history_inputs(clean, CHURN["purchase_event_rule"])


@pytest.mark.parametrize("cutoff", ["2010-12-01", "2011-05-01"])
def test_no_feature_uses_data_from_the_cutoff_on(clean, cutoff):
    assert leakage_differences(clean, pd.Timestamp(cutoff), CHURN, FEAT) == []


def test_snapshot_rows_and_labels_match_c2_churn(clean, inputs):
    events, cancels = inputs
    c = training_cutoffs(CHURN)[-1]
    snap = build_snapshot(clean, events, cancels, c, CHURN, FEAT)
    expected = churn_at_cutoff(events, c, CHURN["active_lookback_days"], CHURN["churn_horizon_days"])
    assert len(snap) == expected["active"]
    assert int(snap["churned"].sum()) == expected["churned"]
    assert snap["customer_id"].is_unique
    assert not snap.isna().any().any()
