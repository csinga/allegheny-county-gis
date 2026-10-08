"""Summary visualisation embedded in the layout: countywide income structure in two panels.

A) share of all county households in each ACS income bracket (the countywide distribution);
B) distribution of tract median incomes, bars coloured by the same classes as the map, so the
   chart doubles as a histogram legend.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import settings as S  # noqa: E402

MM = 1 / 25.4
INK = "#2b2b2b"
MUTED = "#6b6b6b"
GRID = "#dedede"


def _class_of(value: float, breaks: list[float]) -> int:
    return int(np.searchsorted(breaks, value, side="right"))


def summary_chart(
    county_dist: pd.DataFrame,
    tract_medians: np.ndarray,
    breaks: list[float],
    county_median: float,
    path: Path,
    width_mm: float = 146,
    height_mm: float = 70,
) -> Path:
    plt.rcParams.update(
        {
            "font.family": ["Liberation Sans", "Arial", "DejaVu Sans"],
            "font.size": 6.5,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "svg.fonttype": "none",
        }
    )
    colors = S.CLASSES.colors
    edge = "#8a8a8a"
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(width_mm * MM, height_mm * MM), gridspec_kw={"width_ratios": [1.05, 1], "wspace": 0.32}
    )

    # Panel A: countywide distribution of households across ACS brackets (horizontal bars).
    d = county_dist.copy()
    mid = np.where(d["upper_usd"].isna(), d["lower_usd"] * 1.25, (d["lower_usd"] + d["upper_usd"].fillna(0)) / 2)
    d["cls"] = [_class_of(m, breaks) for m in mid]
    short = [
        "<$10k", "$10-15k", "$15-20k", "$20-25k", "$25-30k", "$30-35k", "$35-40k", "$40-45k",
        "$45-50k", "$50-60k", "$60-75k", "$75-100k", "$100-125k", "$125-150k", "$150-200k", "$200k+",
    ]
    y = np.arange(len(d))
    ax1.barh(y, d["pct_households"], color=[colors[c] for c in d["cls"]], edgecolor=edge, linewidth=0.3, height=0.78)
    ax1.set_yticks(y, short)
    ax1.invert_yaxis()
    ax1.tick_params(axis="y", length=0, pad=2, labelsize=5.6)
    ax1.set_xlabel("Share of county households (%)", fontsize=6)
    for yi, v in zip(y, d["pct_households"]):
        ax1.text(v + 0.25, yi, f"{v:.1f}", va="center", fontsize=5, color=MUTED)
    ax1.set_xlim(0, d["pct_households"].max() * 1.22)
    ax1.set_title("A. Households by income bracket", loc="left", fontsize=7, fontweight="bold", color=INK, pad=4)

    # Panel B: histogram of tract medians, coloured by map class.
    v = tract_medians[np.isfinite(tract_medians)]
    width = 10_000
    bins = np.arange(0, np.ceil(v.max() / width) * width + width, width)
    counts, _ = np.histogram(v, bins)
    centers = bins[:-1] + width / 2
    ax2.bar(
        centers / 1000, counts, width=width / 1000 * 0.92,
        color=[colors[_class_of(c, breaks)] for c in centers], edgecolor=edge, linewidth=0.3,
    )
    ax2.axvline(county_median / 1000, color=INK, lw=0.8, ls=(0, (3, 2)))
    ax2.text(
        county_median / 1000 + 3, counts.max() * 0.97,
        f"County median\n${county_median:,.0f}", fontsize=5.6, va="top", color=INK,
    )
    ax2.set_xlabel("Tract median household income ($ thousands)", fontsize=6)
    ax2.set_ylabel("Number of census tracts", fontsize=6)
    ax2.set_xlim(0, bins[-1] / 1000)
    ax2.set_title("B. Tract medians by map class", loc="left", fontsize=7, fontweight="bold", color=INK, pad=4)

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_linewidth(0.5)
        ax.tick_params(width=0.5, length=2, labelsize=5.6)
        ax.set_axisbelow(True)
        ax.grid(axis="x" if ax is ax1 else "y", color=GRID, lw=0.4)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.9, bottom=0.17)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, transparent=False, facecolor="white")
    fig.savefig(path.with_suffix(".png"), dpi=300, facecolor="white")
    plt.close(fig)
    return path
