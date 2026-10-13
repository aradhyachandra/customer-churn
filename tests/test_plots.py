"""The retention heatmap is written as a PNG from a rates table."""

import pandas as pd

from churn.plots import retention_heatmap


def test_retention_heatmap_writes_png(tmp_path):
    months = pd.to_datetime(["2011-01-01", "2011-02-01"])
    rates = pd.DataFrame([[1.0, 0.5, 0.25], [1.0, 0.4, None]], index=months, columns=[0, 1, 2])
    sizes = pd.Series([4, 5], index=months)
    path = tmp_path / "fig" / "heatmap.png"
    retention_heatmap(rates, sizes, path, flagged_cohort=months[0], partial_month=months[1])
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
