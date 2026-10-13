"""The one-shot test evaluation, run on fixtures only (never on the real test snapshot here)."""

import hashlib
import json

import pytest

from churn.config import load_config
from churn.models import fit_final_models, save_models
from churn.report import save_report
from churn.run_c4_test import RefusedToRun, evaluate_test
from tests.conftest import make_snapshot

CLEAN = {"commit": "abc123", "uncommitted_changes": False}
SELECTED = {"rule": {"threshold_days": 90}, "logistic": {"preprocessing": "log_clip"},
            "gbm": {"params": {"learning_rate": 0.1, "max_leaf_nodes": 8, "min_samples_leaf": 20}}}


@pytest.fixture
def setup(tmp_path):
    cfg = {**load_config("model"), "bootstrap_samples": 50}
    models = fit_final_models(make_snapshot(400, "2011-01-01", seed=11), SELECTED, cfg)
    models_dir = tmp_path / "models"
    manifest = {"selected": SELECTED, "models": save_models(models, models_dir),
                "training_cutoffs": ["2011-01-01"]}
    save_report(manifest, tmp_path / "manifest.json")
    snapshot = tmp_path / "snapshot_2011-09-01.parquet"
    make_snapshot(300, "2011-09-01", seed=12).to_parquet(snapshot, index=False)
    return {
        "snapshot_path": snapshot, "models_dir": models_dir, "manifest_path": tmp_path / "manifest.json",
        "out_path": tmp_path / "results" / "c4_test_metrics.json", "expected_cutoff": "2011-09-01",
        "model_cfg": cfg, "git": CLEAN, "raw_sha256": "rawsha",
    }


def test_writes_metrics_with_git_sha_and_checksums(setup):
    evaluate_test(**setup)
    out = json.loads(setup["out_path"].read_text())
    assert out["git"]["commit"] == "abc123"
    assert out["raw_sha256"] == "rawsha"
    expected = hashlib.sha256(setup["snapshot_path"].read_bytes()).hexdigest()
    assert out["test_snapshot"]["sha256"] == expected
    assert out["test_snapshot"]["cutoff"] == "2011-09-01" and out["test_snapshot"]["customers"] == 300
    assert set(out["metrics"]) == {"rule", "logistic", "gbm"}
    for name in ["logistic", "gbm"]:
        m = out["metrics"][name]
        assert 0 <= m["roc_auc"] <= 1 and len(m["calibration"]) == 10
        assert set(m["bootstrap"]["intervals"]) == {"roc_auc", "pr_auc", "precision_top_10pct",
                                                     "precision_top_20pct"}
    assert out["metrics"]["rule"]["brier"] is None
    assert set(out["metrics"]["rule"]["flag"]) == {"threshold_days", "flagged", "flagged_share",
                                                  "flag_precision", "flag_recall"}
    assert out["models"] == json.loads(setup["manifest_path"].read_text())["models"]


def test_refuses_when_output_exists(setup):
    evaluate_test(**setup)
    first = setup["out_path"].read_text()
    with pytest.raises(RefusedToRun, match="already exists"):
        evaluate_test(**setup)
    assert setup["out_path"].read_text() == first


def test_refuses_with_uncommitted_changes(setup):
    with pytest.raises(RefusedToRun, match="uncommitted"):
        evaluate_test(**{**setup, "git": {"commit": "abc123", "uncommitted_changes": True}})
    assert not setup["out_path"].exists()


def test_refuses_without_a_commit(setup):
    with pytest.raises(RefusedToRun, match="commit"):
        evaluate_test(**{**setup, "git": {"commit": "", "uncommitted_changes": False}})


def test_refuses_changed_model_file(setup):
    (setup["models_dir"] / "logistic.joblib").write_bytes(b"retrained elsewhere")
    with pytest.raises(RefusedToRun, match="sha256"):
        evaluate_test(**setup)
    assert not setup["out_path"].exists()


def test_refuses_snapshot_of_another_cutoff(setup, tmp_path):
    other = tmp_path / "other.parquet"
    make_snapshot(100, "2011-05-01", seed=13).to_parquet(other, index=False)
    with pytest.raises(RefusedToRun, match="cutoff"):
        evaluate_test(**{**setup, "snapshot_path": other})


def test_refuses_models_trained_on_the_test_cutoff(setup):
    manifest = json.loads(setup["manifest_path"].read_text())
    manifest["training_cutoffs"] = ["2011-01-01", "2011-09-01"]
    save_report(manifest, setup["manifest_path"])
    with pytest.raises(RefusedToRun, match="trained on"):
        evaluate_test(**setup)
