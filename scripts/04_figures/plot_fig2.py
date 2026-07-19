"""Full retrieval comparison figure (Fig. 2), 2x2 panels.

Top row - Foldseek-style sorted-sensitivity ROC1 under the Rfam ground truth:
    A: Family Sens@1FP (468 queries)
    B: Clan Sens@1FP, excluding same-family hits (361 queries)
Bottom row - retrieval under a structure-defined (US-align TM-score) ground
truth, grouped bars by TM-score threshold:
    C: Recall@1               D: Sens@1FP
US-align ranks by the truth-defining quantity in C/D, so it is shown as a
dashed ceiling (by construction) rather than a competing bar.

Writes figures/fig2.png.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from paths import DATA_ROOT, FIG_DIR, SRC_MODELS, SRC_SEARCH
_sys.path.insert(0, SRC_SEARCH); _sys.path.insert(0, SRC_MODELS)
_os.chdir(DATA_ROOT)
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(DATA_ROOT)
SRC = REPO / "files" / "production" / "retrieval" / "per_query_sens_test.json"
TM = REPO / "files" / "production" / "retrieval" / "structural_tmscore_metrics_test.json"
OUT = Path(FIG_DIR) / "fig2.png"

METHODS = ["RFoldseek", "US-align", "RMalign", "RiboSeek", "BLASTN"]
METHOD_COLOR = {
    "RFoldseek": "#1F9BD8",  # sky blue    — ours
    "US-align":  "#D62728",  # vivid red   — TM-score oracle
    "RMalign":   "#2CA02C",  # forest green
    "RiboSeek":  "#9467BD",  # purple      — fixed alphabet
    "BLASTN":    "#FFB000",  # amber       — sequence baseline
}
LINESTYLE = {"RFoldseek": "-", "US-align": "-", "RMalign": "-", "RiboSeek": "-", "BLASTN": "--"}
MARKER = {"RFoldseek": "o", "US-align": "s", "RMalign": "^", "RiboSeek": "v", "BLASTN": "D"}
BAR_METHODS = ["RFoldseek", "RMalign", "RiboSeek", "BLASTN"]  # US-align is the ceiling


def sorted_curve(arr):
    a = np.sort(np.asarray(arr))[::-1]
    x = np.arange(1, len(a) + 1) / len(a)
    return x, a


def curve_panel(ax, data, level, title):
    for m in METHODS:
        x, y = sorted_curve(data[m][level]["sens"])
        step = max(1, len(x) // 18)
        ax.plot(x, y, LINESTYLE[m], color=METHOD_COLOR[m], linewidth=2,
                marker=MARKER[m], markevery=step, markersize=7,
                markeredgecolor="white", markeredgewidth=0.9, label=m, zorder=3)
    ax.set_xlabel("Fraction of queries", fontsize=12)
    ax.set_ylabel("Sensitivity up to the 1st FP", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(title, fontsize=13, pad=8)
    ax.tick_params(axis="both", labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)


def bar_panel(ax, tm, metric, title, ylabel):
    thr = sorted(float(t) for t in tm["thresholds"])
    keys = [str(t) for t in thr]
    x = np.arange(len(thr))
    width = 0.21  # four adjacent bars per threshold group
    for i, m in enumerate(BAR_METHODS):
        off = (i - 1.5) * width
        vals = [tm["thresholds"][k][m][metric] for k in keys]
        bars = ax.bar(x + off, vals, width, color=METHOD_COLOR[m],
                      edgecolor="none", label=m, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7.0, color="#333333")
    ceil = tm["thresholds"][keys[0]]["US-align"][metric] \
        if "US-align" in tm["thresholds"][keys[0]] \
        else tm["thresholds"][keys[0]]["US-align*"][metric]
    ax.plot([-0.5, len(thr) - 0.5], [ceil, ceil], ls="--",
            color=METHOD_COLOR["US-align"], lw=1.5, zorder=2)
    ax.text(len(thr) - 0.52, ceil - 0.03, "US-align (ceiling)", ha="right",
            va="top", fontsize=8.5, color=METHOD_COLOR["US-align"])
    ax.set_xticks(x, [f"{t:.2f}" for t in thr])
    ax.set_xlabel("TM-score similarity threshold", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim(-0.5, len(thr) - 0.5)
    ax.set_ylim(0, 1.08)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_title(title, fontsize=13, pad=8)
    ax.tick_params(axis="both", labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    data = json.load(open(SRC))
    tm = json.load(open(TM))
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.6))
    curve_panel(axes[0, 0], data, "family", "Family (AUROC1)")
    curve_panel(axes[0, 1], data, "clan", "Clan (AUROC1)")
    bar_panel(axes[1, 0], tm, "Sens@1FP", "Structural ground truth, family level", "Sens@1FP")
    bar_panel(axes[1, 1], tm, "Sens@1FP_clan", "Structural ground truth, clan level", "Sens@1FP")
    for ax, lab in zip(axes.flat, "ABCD"):
        ax.text(-0.10, 1.04, lab, transform=ax.transAxes, fontsize=16,
                fontweight="bold", va="bottom", ha="right")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=12, bbox_to_anchor=(0.5, -0.01), handlelength=2.6,
               columnspacing=2.4)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
