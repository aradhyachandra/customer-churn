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


PNG = b"\x89PNG\r\n\x1a\n"


def test_importance_plot_writes_png(tmp_path):
    from churn.plots import importance_plot

    table = pd.DataFrame({"rank": [1, 2, 3], "feature": ["a", "b", "c"], "mean": [0.05, 0.01, 0.0],
                          "low": [0.04, 0.0, -0.001], "high": [0.06, 0.02, 0.001]})
    path = tmp_path / "fig" / "importance.png"
    importance_plot(table, path, title="t", highlight=2, xlabel="Drop in PR-AUC")
    assert path.read_bytes()[:8] == PNG


def test_partial_dependence_plot_writes_png(tmp_path):
    from churn.plots import partial_dependence_plot

    table = pd.DataFrame({"feature": ["a"] * 3 + ["b"] * 2, "value": [1, 2, 3, 0, 1],
                          "mean_predicted": [0.2, 0.4, 0.6, 0.5, 0.3]})
    path = tmp_path / "fig" / "pdp.png"
    partial_dependence_plot(table, ["a", "b"], path, title="t")
    assert path.read_bytes()[:8] == PNG


def test_campaign_plot_writes_png(tmp_path):
    from churn.plots import campaign_plot

    path = tmp_path / "fig" / "campaign.png"
    campaign_plot([0, 19, 20, 15.6], [0, 5, 4, -2], [0, 30, 40, 38], best_size=2, path=path, title="t",
                  central_label="central", band_label="range")
    assert path.read_bytes()[:8] == PNG


def test_predicted_vs_actual_plot_writes_png(tmp_path):
    from churn.findings import predicted_vs_actual
    from churn.plots import predicted_vs_actual_plot

    t = predicted_vs_actual([0.9, 0.5, 0.4, 0.2], [0.8, 0.4, 0.3, 0.1], [1, 0, 1, 0], [True, True, False, False])
    path = tmp_path / "fig" / "pva.png"
    predicted_vs_actual_plot(t, {"everyone": "Everyone", "chosen": "Chosen list", "not_chosen": "Not chosen"},
                             path, title="Predicted vs actual")
    assert path.read_bytes()[:8] == PNG
