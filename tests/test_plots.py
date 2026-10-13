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


def test_calibration_plot_writes_png(tmp_path):
    from churn.metrics import calibration_table
    from churn.plots import calibration_plot

    tables = {"Logistic regression": calibration_table([0, 1, 1, 0], [0.1, 0.8, 0.6, 0.3], 10),
              "Gradient boosting": calibration_table([0, 1, 1, 0], [0.2, 0.9, 0.7, 0.05], 10)}
    path = tmp_path / "fig" / "calibration.png"
    calibration_plot(tables, path, title="Validation cutoff 2011-05-01")
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
