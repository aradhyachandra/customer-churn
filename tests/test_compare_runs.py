"""Clean-clone comparison: results files must match, ignoring run timestamps and git state."""

import json

from churn.compare_runs import compare_dirs, strip_run_info


def test_strip_run_info_drops_timestamps_and_git_at_any_depth():
    d = {"generated_at": "t", "git": {"commit": "x"}, "a": [{"validation_generated_at": "t", "b": 1}]}
    assert strip_run_info(d) == {"a": [{"b": 1}]}


def test_compare_dirs(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        (d / "figures").mkdir(parents=True)
    (a / "r.json").write_text(json.dumps({"generated_at": "1", "x": 1}))
    (b / "r.json").write_text(json.dumps({"generated_at": "2", "x": 1}))
    (a / "t.csv").write_text("x\n1\n")
    (b / "t.csv").write_text("x\n2\n")
    (a / "figures" / "f.png").write_bytes(b"png")
    (b / "figures" / "f.png").write_bytes(b"png")
    (a / "only_a.csv").write_text("x\n")
    result = {r["file"]: r["status"] for r in compare_dirs(a, b)}
    assert result == {"figures/f.png": "identical", "r.json": "same_ignoring_run_info",
                      "t.csv": "DIFFERENT", "only_a.csv": "MISSING_IN_SECOND"}
