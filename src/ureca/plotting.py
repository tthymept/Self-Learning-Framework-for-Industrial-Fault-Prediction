"""Shared plot style for the notebooks: fixed colours and markers per class."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .data import CLASSES

NAMES = {"partial_discharge": "Partial discharge", "switching_noise": "Switching noise",
         "transient_noise": "Transient noise", "spark_lighter": "Spark lighter"}
COLORS = {"partial_discharge": "#2a78d6", "switching_noise": "#eb6834",
          "transient_noise": "#1baf7a", "spark_lighter": "#eda100"}
MARKERS = {"partial_discharge": "o", "switching_noise": "s",
           "transient_noise": "^", "spark_lighter": "D"}
FOLD_COLORS = ["#6250d6", "#e87ba4", "#008300"]   # validated as a separate palette

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def set_style():
    plt.rcParams.update({
        "figure.dpi": 110, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": INK2, "ytick.color": INK2, "font.size": 9, "axes.titlesize": 10,
        "lines.linewidth": 1.0, "legend.frameon": False,
    })


def legend_handles(classes=CLASSES, marker=True):
    return [Line2D([], [], color=COLORS[c], marker=MARKERS[c] if marker else None,
                   linestyle="" if marker else "-", markersize=7, label=NAMES[c]) for c in classes]


def strip(ax, values, labels, log=False):
    """One column of jittered points per class, black bar at the median."""
    rng = np.random.default_rng(0)
    values, labels = np.asarray(values), np.asarray(labels)
    for i, c in enumerate(CLASSES):
        v = values[labels == c]
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=16, color=COLORS[c],
                   marker=MARKERS[c], alpha=0.8, edgecolor=SURFACE, linewidth=0.4, zorder=3)
        if len(v):
            ax.hlines(np.median(v), i - 0.3, i + 0.3, color=INK, linewidth=1.2, zorder=4)
    ax.set_xticks(range(len(CLASSES)), [NAMES[c].replace(" ", "\n") for c in CLASSES])
    ax.grid(axis="x", visible=False)
    if log:
        ax.set_yscale("log")


def confusion_plot(ax, cm, title=""):
    """Confusion matrix: counts in the cells, colour = share of the true class (row)."""
    share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                    color="white" if share[i, j] > 0.55 else INK)
    names = [NAMES[c].replace(" ", "\n") for c in CLASSES]
    ax.set_xticks(range(len(CLASSES)), names, fontsize=8)
    ax.set_yticks(range(len(CLASSES)), names, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.grid(False); ax.tick_params(length=0)
    ax.set_title(title, loc="left")
