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
