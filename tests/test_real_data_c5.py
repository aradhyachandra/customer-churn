"""REAL DATA CHECKS for C5. Skipped by default; run with:  pytest -m realdata

Read saved results and hash files only; nothing is scored and the one-shot test is not rerun.
"""

import hashlib
import json

import pandas as pd
import pytest

from churn.business import reference_churn_rate
from churn.config import REPO_ROOT, load_config
from churn.run_c5 import check_frozen

pytestmark = pytest.mark.realdata

PATHS, CHURN, BUSINESS = load_config("data")["paths"], load_config("churn"), load_config("business")


def test_rescale_reference_was_known_on_the_test_cutoff():
    table = pd.read_csv(REPO_ROOT / PATHS["c2_churn_by_cutoff"], dtype={"cutoff": str})
    ref = reference_churn_rate(table, BUSINESS["rescale_reference_cutoff"], CHURN["test_cutoff"],
                               CHURN["churn_horizon_days"])
    assert pd.Timestamp(ref["window_end"]) <= pd.Timestamp(CHURN["test_cutoff"])


def test_frozen_models_and_test_snapshot_match_the_one_shot_test():
    snap = REPO_ROOT / PATHS["snapshots_dir"] / f"snapshot_{CHURN['test_cutoff']}.parquet"
    if not snap.exists():
        pytest.skip("run python -m churn.run_c3 first")
    test = json.loads((REPO_ROOT / PATHS["c4_test_metrics"]).read_text())
    manifest = json.loads((REPO_ROOT / PATHS["c4_final_models"]).read_text())
    check_frozen(test, manifest, hashlib.sha256(snap.read_bytes()).hexdigest(), CHURN["test_cutoff"])
    for entry in manifest["models"].values():
        path = REPO_ROOT / PATHS["models_dir"] / entry["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
