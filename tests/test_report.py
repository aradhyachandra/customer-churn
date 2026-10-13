"""Report tests: the saved report round-trips and docs are filled only from it."""

import json

import pytest

from churn.report import markdown_table, render_template, save_report


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


def test_render_named_tables():
    out = render_template("Before\n{{table:churn}}\nAfter", sample_report(), tables={"churn": "| a |"})
    assert out == "Before\n| a |\nAfter"


def test_render_unknown_table_fails():
    with pytest.raises(KeyError, match="table:missing"):
        render_template("{{table:missing}}", sample_report(), tables={})


def test_markdown_table_formats_values():
    rows = [{"cutoff": "2011-01-01", "active": 1234, "rate": 0.25, "note": None}]
    out = markdown_table(rows, [("cutoff", "Cutoff"), ("active", "Active"),
                                ("rate", "Rate", "pct"), ("note", "Note")])
    assert out.splitlines() == [
        "| Cutoff | Active | Rate | Note |",
        "|---|---:|---:|---|",
        "| 2011-01-01 | 1,234 | 25.0% |  |",
    ]


def test_render_percent_suffix():
    out = render_template("{{final.share|pct}}", {"final": {"share": 0.1234}})
    assert out == "12.3%"
