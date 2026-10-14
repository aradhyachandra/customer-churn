"""C5 reads the frozen models and the test snapshot only if they are exactly those of the one-shot test."""

import pytest

from churn.run_c5 import check_frozen

MODELS = {"gbm": {"file": "gbm.joblib", "sha256": "aaa"}}
TEST = {"test_snapshot": {"cutoff": "2011-09-01", "sha256": "snap"}, "models": MODELS,
        "training_cutoffs": ["2011-05-01"]}
MANIFEST = {"models": MODELS, "training_cutoffs": ["2011-05-01"]}


def test_accepts_the_frozen_inputs():
    check_frozen(TEST, MANIFEST, "snap", "2011-09-01")


def test_refuses_a_different_snapshot():
    with pytest.raises(ValueError, match="snapshot"):
        check_frozen(TEST, MANIFEST, "other", "2011-09-01")


def test_refuses_models_other_than_those_tested():
    changed = {"models": {"gbm": {"file": "gbm.joblib", "sha256": "bbb"}}, "training_cutoffs": ["2011-05-01"]}
    with pytest.raises(ValueError, match="models"):
        check_frozen(TEST, changed, "snap", "2011-09-01")


def test_refuses_another_test_cutoff():
    with pytest.raises(ValueError, match="cutoff"):
        check_frozen(TEST, MANIFEST, "snap", "2011-08-01")
