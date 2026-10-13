"""Save the cleaning report and fill doc templates from it, so docs never hold hand-typed numbers."""

import json
import re
from pathlib import Path

PLACEHOLDER = re.compile(r"\{\{\s*([\w.\-: |]+?)\s*\}\}")


def save_report(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


def _fmt(value, style=None):
    if value is None:
        return ""
    if style == "pct":
        return f"{value:.1%}"
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


def markdown_table(rows, columns):
    """rows: list of dicts. columns: (key, header) or (key, header, "pct"). Numbers align right."""
    def numeric(key):
        return any(isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool) for r in rows)

    lines = ["| " + " | ".join(c[1] for c in columns) + " |",
             "|" + "|".join("---:" if numeric(c[0]) else "---" for c in columns) + "|"]
    for r in rows:
        cells = [_fmt(r.get(c[0]), c[2] if len(c) > 2 else None) for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _lookup(report, dotted, tables):
    if dotted == "steps_table":
        return steps_table(report["steps"])
    if dotted.startswith("table:"):
        name = dotted[len("table:"):]
        if name not in tables:
            raise KeyError(f"Placeholder {{{{{dotted}}}}} has no table")
        return tables[name]
    dotted, _, style = dotted.partition("|")
    value = report
    for part in dotted.split("."):
        if isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        elif isinstance(value, dict) and part in value:
            value = value[part]
        else:
            raise KeyError(f"Placeholder {{{{{dotted}}}}} not found in report")
    return _fmt(value, style or None)


def render_template(text, report, tables=None):
    """Replace {{a.b.c}} with the formatted value at report['a']['b']['c'] ({{a.b|pct}} as a
    percentage) and {{table:name}} with tables[name], a ready-made markdown table."""
    return PLACEHOLDER.sub(lambda m: _lookup(report, m.group(1), tables or {}), text)
