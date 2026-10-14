"""Dashboard smoke tests: it loads its data from committed files only, the campaign calculator
matches results/c5_campaign_scenarios.csv at every scenario grid point, and the app runs."""

import numpy as np
import pandas as pd
import pytest

from churn.config import REPO_ROOT
from churn.dashboard import SCORE_COLUMNS, campaign_estimate, load_dashboard_data, unusual_phrases

APP = REPO_ROOT / "app" / "streamlit_app.py"


def test_campaign_estimate_by_hand():
    # Benefits: 0.5*0.1*0.3*1000-5 = 10, 0.8*0.1*0.3*100-5 = -2.6, 0.9*0.1*0.3*400-5 = 5.8
    p, value = np.array([0.5, 0.8, 0.9]), np.array([1000.0, 100.0, 400.0])
    r = campaign_estimate(p, value, margin=0.3, save_rate=0.1, offer_cost=5)
    assert r["best_size"] == 2
    assert r["total_benefit"] == pytest.approx(15.8)
    assert r["offer_cost_total"] == 10
    assert r["expected_saved_revenue"] == pytest.approx(0.5 * 0.1 * 1000 + 0.9 * 0.1 * 400)
    assert r["value_threshold"] == pytest.approx(5 / 0.03)
    assert list(r["curve"]) == pytest.approx([0, 10, 15.8, 13.2])


def test_load_dashboard_data_uses_committed_results_only():
    d = load_dashboard_data(REPO_ROOT)
    assert set(SCORE_COLUMNS) <= set(d["scores"].columns)
    assert d["scores"]["customer_id"].is_unique
    assert len(d["scores"]) == d["business"]["customers"]
    assert set(d["report_config"]["profile_features"]) <= set(d["scores"].columns)
    assert not d["retention_rates"].empty and not d["segments"].empty


def test_unusual_phrases_one_string_per_customer():
    d = load_dashboard_data(REPO_ROOT)
    rows = d["scores"].head(50)
    phrases = unusual_phrases(rows, d["report_config"]["profile_features"], 3)
    assert len(phrases) == 50
    assert all(p.count("than") == 3 for p in phrases)


def test_calculator_matches_saved_scenarios_at_every_grid_point():
    d = load_dashboard_data(REPO_ROOT)
    s, scenarios = d["scores"], pd.read_csv(REPO_ROOT / "results" / "c5_campaign_scenarios.csv")
    assert len(scenarios) == 54
    for row in scenarios.itertuples():
        p = s["p_model"] if row.version == "a" else s["p_rescaled"]
        r = campaign_estimate(p, s["value"], row.margin, row.save_rate, row.offer_cost)
        assert r["best_size"] == row.best_size, row
        assert r["total_benefit"] == pytest.approx(row.total_benefit, abs=0.01), row
        assert r["expected_saved_revenue"] == pytest.approx(row.expected_saved_revenue, abs=0.01), row


def test_app_runs_and_calculator_shows_the_central_case():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception
    assert len(at.tabs) == 4
    d = load_dashboard_data(REPO_ROOT)
    central = d["business"]["central"]["b"]
    shown = " ".join(str(m.value) for m in at.metric)
    assert f"{central['best_size']:,}" in shown
    assert f"£{central['total_benefit']:,.0f}" in shown


def test_app_toggles_and_sliders_run_without_errors():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.checkbox[0].uncheck().run()          # whole list, not only the central-case customers
    at.checkbox[1].check().run()            # after-the-fact column
    assert not at.exception
    assert any("After-the-fact" in w.value for w in at.warning)
    at.slider(key="save").set_value(0.2).run()
    at.slider(key="cost").set_value(40).run()
    assert not at.exception
    d = load_dashboard_data(REPO_ROOT)
    s = d["scores"]
    expected = campaign_estimate(s["p_rescaled"], s["value"], 0.3, 0.2, 40)
    assert f"{expected['best_size']:,}" in " ".join(str(m.value) for m in at.metric)
