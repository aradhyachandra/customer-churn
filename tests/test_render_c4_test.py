"""Rendering the C4 test section from the saved test JSON (no model is run here)."""

import pandas as pd
import pytest

from churn.render_c4_test import (
    comparison_sentence,
    intervals_overlap,
    replace_section,
    season_overlap_days,
)


def test_intervals_overlap():
    assert intervals_overlap({"low": 0.1, "high": 0.3}, {"low": 0.25, "high": 0.4})
    assert not intervals_overlap({"low": 0.1, "high": 0.2}, {"low": 0.21, "high": 0.4})
    assert intervals_overlap({"low": 0.1, "high": 0.2}, {"low": 0.2, "high": 0.4})   # touching counts


def _m(low10, high10, low20, high20, lowpr, highpr):
    return {"bootstrap": {"intervals": {
        "precision_top_10pct": {"low": low10, "high": high10},
        "precision_top_20pct": {"low": low20, "high": high20},
        "pr_auc": {"low": lowpr, "high": highpr}}}}


def test_comparison_sentence_says_beats_only_when_no_interval_overlaps():
    strong, weak = _m(0.8, 0.9, 0.75, 0.8, 0.72, 0.76), _m(0.7, 0.78, 0.68, 0.74, 0.65, 0.70)
    text = comparison_sentence("Boosting", strong, "the rule", weak)
    assert "beats" in text and "do not overlap" in text
    close = _m(0.75, 0.83, 0.73, 0.79, 0.70, 0.75)
    text = comparison_sentence("Boosting", strong, "logistic", close)
    assert "not settled" in text and "overlap" in text


def test_replace_section_swaps_only_the_named_section():
    doc = "# T\n\nintro\n\n## Test results (part B)\n\nNOT RUN YET\n"
    out = replace_section(doc, "## Test results (part B)", "## Test results (part B)\n\nnumbers\n")
    assert out == "# T\n\nintro\n\n## Test results (part B)\n\nnumbers\n"
    with pytest.raises(ValueError):
        replace_section(doc, "## Missing", "x")


def test_season_overlap_days_hand_worked():
    # 2011-09-01 + 90 days: Sep 1 .. Nov 29, all inside Sep-Nov.
    assert season_overlap_days(pd.Timestamp("2011-09-01"), 90, (9, 10, 11)) == 90
    # 2011-08-01: Aug 1 .. Oct 29 -> 30 Sep days + 29 Oct days.
    assert season_overlap_days(pd.Timestamp("2011-08-01"), 90, (9, 10, 11)) == 59
    assert season_overlap_days(pd.Timestamp("2011-05-01"), 90, (9, 10, 11)) == 0
