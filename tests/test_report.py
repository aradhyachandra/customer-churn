"""Report tests: the saved report round-trips and docs are filled only from it."""

import json

import pytest

from churn.report import render_template, save_report


def sample_report():
    return {
        "final": {"rows": 1234567, "net_value": 8765432.123},
        "steps": [
            {"step": "drop_x", "rows_before": 10, "rows_removed": 3, "rows_after": 7,
             "net_value_removed": -1.5, "sales_value_removed": 2.0},
        ],
        "raw_sha256": "abc",
    }


def test_save_report_round_trips(tmp_path):
    path = tmp_path / "sub" / "report.json"
    save_report(sample_report(), path)
    assert json.loads(path.read_text()) == sample_report()


def test_render_formats_numbers_from_report():
    out = render_template("Rows {{final.rows}}, value £{{final.net_value}}, sha {{raw_sha256}}.",
                          sample_report())
    assert out == "Rows 1,234,567, value £8,765,432.12, sha abc."


def test_render_steps_table():
    out = render_template("{{steps_table}}", sample_report())
    assert "| drop_x | 10 | 3 | 7 | -1.50 | 2.00 |" in out


def test_render_indexes_into_lists():
    assert render_template("{{steps.0.rows_removed}}", sample_report()) == "3"


def test_render_unknown_placeholder_fails():
    with pytest.raises(KeyError, match="final.missing"):
        render_template("{{final.missing}}", sample_report())
