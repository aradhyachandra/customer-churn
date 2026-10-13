"""Milestone C4, part B: score the test cutoff (2011-09-01) ONCE with the frozen final models.

Run with:  python -m churn.run_c4_test      (only after C4 part A is reviewed and committed)

It refuses to run when results/c4_test_metrics.json already exists, when the working tree has
uncommitted changes, when a model file differs from the frozen manifest
(results/c4_final_models.json), when the models were trained on the test cutoff, or when the
snapshot is not the test cutoff. Nothing is tuned or refitted here.
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from churn.config import REPO_ROOT, load_config
from churn.metrics import bootstrap_intervals, evaluate
from churn.models import feature_matrix, load_models, model_scores, rule_flag_stats
from churn.run_c1 import git_state


class RefusedToRun(Exception):
    pass


def evaluate_test(snapshot_path, models_dir, manifest_path, out_path, expected_cutoff, model_cfg, git,
                  raw_sha256):
    out_path, expected = Path(out_path), pd.Timestamp(expected_cutoff)
    if out_path.exists():
        raise RefusedToRun(f"{out_path.name} already exists: the test cutoff is scored only once")
    if not git.get("commit"):
        raise RefusedToRun("No git commit found: commit the frozen code and settings first")
    if git.get("uncommitted_changes"):
        raise RefusedToRun("The working tree has uncommitted changes: commit or stash them first")

    manifest = json.loads(Path(manifest_path).read_text())
    if expected in [pd.Timestamp(c) for c in manifest["training_cutoffs"]]:
        raise RefusedToRun("The frozen models were trained on the test cutoff")
    try:
        models = load_models(models_dir, manifest["models"])
    except ValueError as e:
        raise RefusedToRun(str(e)) from e

    snapshot_bytes = Path(snapshot_path).read_bytes()
    df = pd.read_parquet(snapshot_path)
    if set(df["cutoff"]) != {expected}:
        raise RefusedToRun(f"The snapshot is not the test cutoff {expected:%Y-%m-%d}")

    y = df["churned"].to_numpy()
    metrics = {}
    for name, score in model_scores(models, df).items():
        m = evaluate(y, score, model_cfg["top_fractions"], model_cfg["calibration_bins"],
                     is_probability=name != "rule")
        m["bootstrap"] = bootstrap_intervals(y, score, model_cfg["top_fractions"],
                                             model_cfg["bootstrap_samples"], model_cfg["seed"])
        if name == "rule":
            m["flag"] = rule_flag_stats(models["rule"], feature_matrix(df), y)
        metrics[name] = m

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git,
        "raw_sha256": raw_sha256,
        "test_snapshot": {"file": Path(snapshot_path).name, "cutoff": f"{expected:%Y-%m-%d}",
                          "customers": len(df), "sha256": hashlib.sha256(snapshot_bytes).hexdigest()},
        "selected": manifest["selected"],
        "training_cutoffs": manifest["training_cutoffs"],
        "models": manifest["models"],
        "metrics": metrics,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "x") as f:   # "x": never overwrite, even if the file appeared meanwhile
        f.write(json.dumps(result, indent=2) + "\n")
    return result


def main():
    data_cfg, churn_cfg, model_cfg = load_config("data"), load_config("churn"), load_config("model")
    paths, test = data_cfg["paths"], churn_cfg["test_cutoff"]
    try:
        evaluate_test(
            snapshot_path=REPO_ROOT / paths["snapshots_dir"] / f"snapshot_{test}.parquet",
            models_dir=REPO_ROOT / paths["models_dir"],
            manifest_path=REPO_ROOT / paths["c4_final_models"],
            out_path=REPO_ROOT / paths["c4_test_metrics"],
            expected_cutoff=test,
            model_cfg=model_cfg,
            git=git_state(),
            raw_sha256=data_cfg["raw"]["sha256"],
        )
    except RefusedToRun as e:
        sys.exit(f"Refused: {e}")
    print(f"Test metrics written to {paths['c4_test_metrics']}")


if __name__ == "__main__":
    main()
