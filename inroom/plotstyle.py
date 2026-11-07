"""Shared figure style: palette, chrome and colormaps."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Categorical slots, in fixed order (validated: all pairs of the first three clear the CVD and
# normal-vision floors on a white surface).
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

CASE_COLORS = {"rigorous": SLOTS[0], "raw": SLOTS[1], "corrected": SLOTS[2]}
CASE_STYLES = {"rigorous": "-", "raw": ":", "corrected": "--"}
CASE_LABELS = {"rigorous": "rigorous (axis on baffle)", "raw": "centered, uncorrected",
               "corrected": "centered, corrected"}

DRIVER_COLORS = {"woofer": SLOTS[0], "mid": SLOTS[1], "tweeter": SLOTS[2], "system": "#0b0b0b"}
DRIVER_LABELS = {"woofer": "woofer", "mid": "midrange", "tweeter": "tweeter", "system": "3-way sum"}

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#f0efec"

# Sequential: one hue, light to dark.
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
         "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("blues", BLUES)
# Diverging: blue (too low) <-> gray <-> red (too high).
DIVERGING = LinearSegmentedColormap.from_list(
    "bluered", ["#0d366b", "#256abf", "#6da7ec", "#b7d3f6", NEUTRAL, "#f6c1bf", "#ec8584", "#d63b3a", "#7a1c1c"])


def apply():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.grid": True,
        "axes.grid.which": "both",
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 0,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "lines.linewidth": 1.2,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def freq_axis(ax, lo=20.0, hi=20_000.0, sparse=False):
    """Log frequency axis with plain tick labels (``sparse``: decades only, for narrow panels)."""
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    candidates = (20, 100, 1000, 10000) if sparse else (20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000)
    ticks = [t for t in candidates if lo <= t <= hi]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t / 1000:g}k" if t >= 1000 else f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())


def db10(p, floor=1e-30):
    return 10.0 * np.log10(np.maximum(p, floor))
