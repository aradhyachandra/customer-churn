"""Save the cleaning report and fill doc templates from it, so docs never hold hand-typed numbers."""

import json
import re
from pathlib import Path

PLACEHOLDER = re.compile(r"\{\{\s*([\w.\- ]+?)\s*\}\}")


def save_report(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


def _fmt(value):
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


def steps_table(steps):
    lines = [
        "| Step | Rows before | Rows removed | Rows after | Net value removed (£) | Sales value removed (£) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in steps:
        lines.append(
            f"| {s['step']} | {_fmt(s['rows_before'])} | {_fmt(s['rows_removed'])} | "
            f"{_fmt(s['rows_after'])} | {_fmt(s['net_value_removed'])} | {_fmt(s['sales_value_removed'])} |"
        )
    return "\n".join(lines)


def _lookup(report, dotted):
    if dotted == "steps_table":
        return steps_table(report["steps"])
    value = report
    for part in dotted.split("."):
        if isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        elif isinstance(value, dict) and part in value:
            value = value[part]
        else:
            raise KeyError(f"Placeholder {{{{{dotted}}}}} not found in report")
    return _fmt(value)


def render_template(text, report):
    """Replace {{a.b.c}} with the formatted value at report['a']['b']['c']."""
    return PLACEHOLDER.sub(lambda m: _lookup(report, m.group(1)), text)
