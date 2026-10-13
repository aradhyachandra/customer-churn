"""REAL DATA CHECKS for C4 part A. Skipped by default; run with:  pytest -m realdata

Read the training snapshot tables (build them with: python -m churn.run_c3). Never the test cutoff.
"""

import pandas as pd
import pytest

import churn.models as models
from churn.config import REPO_ROOT, load_config
from churn.models import load_snapshots, validation_split

pytestmark = pytest.mark.realdata

CHURN, MODEL = load_config("churn"), load_config("model")
SNAP_DIR = REPO_ROOT / load_config("data")["paths"]["snapshots_dir"]


def test_split_reads_training_snapshots_only_and_matches_c2(monkeypatch):
    if not SNAP_DIR.exists():
        pytest.skip("run python -m churn.run_c3 first")
    read = []
    real_read = pd.read_parquet
    monkeypatch.setattr(models.pd, "read_parquet", lambda p, *a, **k: read.append(str(p)) or real_read(p, *a, **k))
    split = validation_split(CHURN, MODEL)
    val = load_snapshots(SNAP_DIR, [split["validation"]], CHURN["test_cutoff"])
    tuning = load_snapshots(SNAP_DIR, split["tuning"], CHURN["test_cutoff"])
    assert not any(CHURN["test_cutoff"] in p for p in read)
    c2 = pd.read_csv(REPO_ROOT / "results/c2_churn_by_cutoff.csv").set_index("cutoff")
    assert len(val) == c2.loc["2011-05-01", "active"] and val["churned"].sum() == c2.loc["2011-05-01", "churned"]
    assert len(tuning) == c2.loc[["2010-12-01", "2011-01-01"], "active"].sum()
