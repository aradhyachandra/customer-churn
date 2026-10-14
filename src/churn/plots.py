"""Figures saved as PNG under results/figures/."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

# One blue hue, light (near zero) to dark (high retention).
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, MUTED, GRID = "#1f1f1e", "#6b6b66", "#ffffff"
LABELLED_MONTHS = (1, 3, 6, 12)


def retention_heatmap(rates, sizes, path, flagged_cohort=None, partial_month=None):
    """Cohort retention heatmap for months 1.. (month 0 is always 100% and is left out).
    flagged_cohort gets an asterisk; cells falling in partial_month are hatched."""
    data = rates.drop(columns=0)
    values = data.to_numpy(dtype=float)
    n_rows, n_cols = values.shape
    cmap = LinearSegmentedColormap.from_list("blues", BLUES).with_extremes(bad="#f4f4f2")
    vmax = np.nanmax(values)

    fig, ax = plt.subplots(figsize=(13, 8.5), dpi=150)
    im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")

    for i, cohort in enumerate(data.index):
        for j, k in enumerate(data.columns):
            v = values[i, j]
            if np.isnan(v):
                continue
            in_partial = partial_month is not None and cohort + pd.DateOffset(months=int(k)) == partial_month
            if in_partial:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///",
                                       edgecolor=MUTED, linewidth=0))
            if k in LABELLED_MONTHS:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > 0.55 * vmax else INK)

    labels = []
    for cohort in data.index:
        star = " *" if flagged_cohort is not None and cohort == flagged_cohort else ""
        labels.append(f"{cohort:%b %Y}{star}  (n={int(sizes[cohort]):,})")
    ax.set_yticks(range(n_rows), labels, fontsize=7.5, color=INK)
    ax.set_xticks(range(n_cols), [str(k) for k in data.columns], fontsize=7.5, color=INK)
    ax.set_xticks(np.arange(-0.5, n_cols), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows), minor=True)
    ax.grid(which="minor", color=GRID, linewidth=1)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlabel("Months since first purchase", fontsize=9, color=INK)
    ax.set_ylabel("Cohort (month of first purchase)", fontsize=9, color=INK)
    ax.set_title("Share of each first-purchase cohort buying again, by month", fontsize=12,
                 color=INK, loc="left", pad=12)
    note = "Numbers shown for months 1, 3, 6 and 12."
    if flagged_cohort is not None:
        note += f"  * {flagged_cohort:%b %Y} also holds customers who first bought before the data starts."
    if partial_month is not None:
        note += f"  Hatched: {partial_month:%b %Y}, only partly covered by the data."
    fig.text(0.01, 0.01, note, fontsize=7.5, color=MUTED)

    bar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, format=lambda x, _: f"{x:.0%}")
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=7.5, colors=INK, length=0)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# Categorical slots 1 and 2 of the reference palette (blue, orange), in fixed order.
SERIES = ["#2a78d6", "#eb6834"]


def calibration_plot(tables, path, title):
    """Reliability diagram: mean predicted probability vs observed churn rate per bin, one line per
    model (tables: {label: calibration_table rows}), with customers per bin in a panel below."""
    fig, (ax, ax_n) = plt.subplots(2, 1, figsize=(7.5, 8), dpi=150, sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", zorder=1)
    ax.text(0.30, 0.33, "perfect calibration", rotation=45, ha="left", va="bottom", fontsize=7.5,
            color=MUTED, rotation_mode="anchor")
    width = 0.1 / (len(tables) + 1)
    for i, (label, rows) in enumerate(tables.items()):
        filled = [r for r in rows if r["customers"]]
        x = [r["mean_predicted"] for r in filled]
        y = [r["observed_churn_rate"] for r in filled]
        ax.plot(x, y, color=SERIES[i], linewidth=2, marker="o", markersize=6,
                markeredgecolor="white", markeredgewidth=1.5, label=label, zorder=2 + i)
        centres = [r["lower"] + (i + 1) * width for r in rows]
        ax_n.bar(centres, [r["customers"] for r in rows], width=width * 0.9, color=SERIES[i])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Observed churn rate", fontsize=9, color=INK)
    ax.set_title(title, fontsize=12, color=INK, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK)
    ax_n.set_ylabel("Customers", fontsize=9, color=INK)
    ax_n.set_xlabel("Predicted churn probability (10 equal-width bins)", fontsize=9, color=INK)
    for a in (ax, ax_n):
        a.grid(color="#e6e6e3", linewidth=0.8)
        a.set_axisbelow(True)
        a.tick_params(labelsize=8, colors=INK, length=0)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color("#c9c9c4")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax_n.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax_n.set_xticks(np.linspace(0, 1, 11))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _tidy(ax):
    ax.grid(color="#e6e6e3", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, colors=INK, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c9c4")


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def importance_plot(table, path, title, highlight, xlabel):
    """Horizontal bars of mean permutation importance with a low-high whisker per feature, most
    important at the top; the first `highlight` features in the main blue, the rest light."""
    t = table.sort_values("rank")
    n = len(t)
    fig, ax = plt.subplots(figsize=(7.5, 0.28 * n + 1.2), dpi=150)
    y = np.arange(n)[::-1]
    colors = [SERIES[0] if r <= highlight else BLUES[1] for r in t["rank"]]
    ax.barh(y, t["mean"], height=0.7, color=colors)
    ax.errorbar(t["mean"], y, xerr=[t["mean"] - t["low"], t["high"] - t["mean"]], fmt="none",
                ecolor=INK, elinewidth=1, capsize=2)
    ax.set_yticks(y, t["feature"], fontsize=8, color=INK)
    ax.axvline(0, color="#c9c9c4", linewidth=1)
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.set_title(title, fontsize=12, color=INK, loc="left", pad=10)
    _tidy(ax)
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def partial_dependence_plot(table, features, path, title):
    """One small panel per feature (shared y axis): average predicted churn probability vs the value."""
    cols = 2
    rows = int(np.ceil(len(features) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(8, 3.2 * rows), dpi=150, sharey=True, squeeze=False)
    for ax, f in zip(axes.flat, features):
        d = table[table["feature"] == f].sort_values("value")
        ax.plot(d["value"], d["mean_predicted"], color=SERIES[0], linewidth=2, marker="o", markersize=4)
        ax.set_title(f, fontsize=10, color=INK, loc="left")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        _tidy(ax)
    for ax in axes.flat[len(features):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Average predicted churn", fontsize=9, color=INK)
    fig.suptitle(title, fontsize=12, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    _save(fig, path)


def campaign_plot(central, band_low, band_high, best_size, path, title, central_label, band_label):
    """Total expected net benefit (GBP) vs number of customers contacted: the central case as a line,
    the range over all scenarios as a shaded band, and a marker at the central best size."""
    x = np.arange(len(central))
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    ax.fill_between(x, band_low, band_high, color=BLUES[0], linewidth=0, label=band_label)
    ax.plot(x, central, color=SERIES[0], linewidth=2, label=central_label)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.plot([best_size], [central[best_size]], marker="o", markersize=8, color=SERIES[0],
            markeredgecolor="white", markeredgewidth=1.5, zorder=5)
    ax.annotate(f"best: {best_size:,} customers, £{central[best_size]:,.0f}", (best_size, central[best_size]),
                xytext=(10, 10), textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_xlabel("Customers contacted (highest expected benefit first)", fontsize=9, color=INK)
    ax.set_ylabel("Total expected net benefit (£, estimate)", fontsize=9, color=INK)
    ax.yaxis.set_major_formatter(lambda v, _: f"{'-' if v < 0 else ''}£{abs(v):,.0f}")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.set_xlim(0, len(central) - 1)
    ax.set_title(title, fontsize=12, color=INK, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK)
    _tidy(ax)
    _save(fig, path)


def predicted_vs_actual_plot(table, labels, path, title):
    """Paired bars per group: mean predicted churn (rescaled) next to the churn rate actually seen.
    table: rows of churn.findings.predicted_vs_actual; labels: {group: axis label}."""
    t = table.set_index("group").loc[list(labels)]
    x = np.arange(len(t))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=150)
    series = (("mean_p_rescaled", "Mean predicted churn (rescaled, known on the day)", SERIES[0], -1),
              ("observed_churn_rate", "Churn actually seen (after the fact)", SERIES[1], 1))
    for column, label, color, side in series:
        bars = ax.bar(x + side * (width / 2 + 0.01), t[column], width=width, color=color, label=label)
        ax.bar_label(bars, labels=[f"{v:.0%}" for v in t[column]], padding=3, fontsize=8.5, color=INK)
    ax.set_xticks(x, [f"{labels[g]}\n({n:,} customers)" for g, n in zip(t.index, t["customers"])],
                  fontsize=8.5, color=INK)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel("Share of customers", fontsize=9, color=INK)
    ax.set_title(title, fontsize=12, color=INK, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK)
    _tidy(ax)
    ax.grid(axis="x", visible=False)
    _save(fig, path)
