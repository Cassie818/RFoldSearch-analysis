"""Supplementary retrieval-metrics figure: supplementary retrieval metrics for the benchmark methods on the
held-out test partition (same query sets as Fig 2A/B).

(A) Rfam family- and clan-level Recall@1 (grouped bars).
(B,C) Per-query-averaged precision-recall curves at the (B) family and (C) clan level: 
each method's mean interpolated precision over queries, with its mean average precision (AP) in the legend. Each query is
weighted equally, so the curves are not biased by family size or query length; AP corresponds to the per-query mean AUPR.
"""
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from paths import DATA_ROOT, FIG_DIR, SRC_MODELS, SRC_SEARCH, resolve

_sys.path.insert(0, SRC_SEARCH)
_sys.path.insert(0, SRC_MODELS)
_os.chdir(DATA_ROOT)
import json, numpy as np, matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
import matplotlib.pyplot as plt
from pathlib import Path

REPO = Path(DATA_ROOT)
RET = REPO / "files/production/retrieval"
r1 = json.load(open(RET / "recall1_rfam_test.json"))
wp = json.load(open(RET / "weighted_pr_rfam_test.json"))
ap_sig = json.load(open(RET / "bootstrap_figS12_ap.json"))

# Colours, markers and the RFoldSearch emphasis are shared with Fig. 2.
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from plot_fig2 import (CURVE_MUTED, HIGHLIGHT, LABEL, LINESTYLE, MARKER,  # noqa: E402
                       METHOD_BASE, METHOD_COLOR, METHODS)

# Two rows: family (top) and clan (bottom); left = Recall@1 bars, right = PR
# curve. Bars are coloured by method to match the curves on their right.
fig = plt.figure(figsize=(12.8, 9.0))
gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 1.0], wspace=0.22, hspace=0.55)
axA = fig.add_subplot(gs[0, 0])   # family Recall@1
axB = fig.add_subplot(gs[0, 1])   # family PR
axC = fig.add_subplot(gs[1, 0])   # clan Recall@1
axD = fig.add_subplot(gs[1, 1])   # clan PR

xs = np.arange(len(METHODS))


# ---- (A,C) Recall@1 bars, one level per panel, coloured by method ----
def recall_panel(ax, level, title):
    vals = np.array([r1[m][f"{level}_recall1"] for m in METHODS])
    colors = [METHOD_COLOR[m] for m in METHODS]
    bars = ax.bar(xs, vals, 0.68, color=colors, edgecolor="#666666",
                  linewidth=0.6, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.014, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8, color="#333333")
    ax.set_ylabel("Recall@1", fontsize=12)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL.get(m, m) for m in METHODS], fontsize=9.5, rotation=30, ha="right")
    ax.set_ylim(0, 1.16)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0.1, 1.0, 0.2), minor=True)
    ax.grid(axis="y", which="major", color="#DDDDDD", linewidth=0.8, zorder=0)
    ax.grid(axis="y", which="minor", color="#EEEEEE", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", which="minor", length=0)
    ax.set_xlim(-0.6, len(METHODS) - 0.4)
    ax.set_title(title, fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)


# ---- (B,D) per-query PR curves ----
def pr_panel(ax, level, title):
    for m in METHODS:
        rec = np.asarray(wp[m][level]["recall"])
        prec = np.asarray(wp[m][level]["precision"])
        ap = wp[m][level]["aupr"]
        star = "*" if ap_sig.get(level, {}).get(m, {}).get("sig") else ""
        hero = m in HIGHLIGHT
        step = max(1, len(rec) // 10)
        size = (10 if m == "RFoldSearch-TM" else 7) if hero else 5
        ax.plot(rec, prec, LINESTYLE[m], color=METHOD_BASE[m] if hero else CURVE_MUTED[m],
                linewidth=3.0 if hero else 1.9, marker=MARKER[m], markevery=step,
                markersize=size, markeredgecolor="white",
                markeredgewidth=0.9 if hero else 0.6,
                label=f"{LABEL.get(m, m)} ({ap:.2f}){star}", zorder=HIGHLIGHT.get(m, 3))
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_title(title, fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)
    # method + AP in the legend entries; no legend title
    ax.legend(loc="lower left", frameon=False, fontsize=8.4,
              handlelength=1.4, labelspacing=0.3)


recall_panel(axA, "family", "Family")
pr_panel(axB, "family", "Family")
recall_panel(axC, "clan", "Clan")
pr_panel(axD, "clan", "Clan")

# panel labels
for ax, lab in ((axA, "A"), (axB, "B"), (axC, "C"), (axD, "D")):
    ax.text(-0.12, 1.06, lab, transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")

fig.tight_layout(rect=(0, 0.02, 1, 1))
OUT = str(Path(FIG_DIR) / "supp_retrieval_metrics.png")
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
