"""Fig S12: supplementary retrieval metrics for the benchmark methods on the
held-out test partition (same query sets as Fig 2A/B).

(A) Rfam family- and clan-level Recall@1 (grouped bars).
(B,C) Per-query-averaged precision-recall curves at the (B) family and
      (C) clan level: each method's mean interpolated precision over queries,
      with its mean average precision (AP) in the legend. Each query is
      weighted equally, so the curves are not biased by family size or query
      length; AP corresponds to the per-query mean AUPR.
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

METHODS = ["RFoldseek", "GTalign", "RTM-align", "US-align", "RMalign", "RiboSeek", "BLASTN"]
METHOD_COLOR = {
    "RFoldseek": "#1F9BD8", "GTalign": "#9467BD", "RTM-align": "#FF7F0E",
    "US-align": "#D62728", "RMalign": "#2CA02C", "RiboSeek": "#E377C2", "BLASTN": "#FFC107",
}
MARKER = {"RFoldseek": "o", "GTalign": "X", "RTM-align": "P", "US-align": "s",
          "RMalign": "^", "RiboSeek": "v", "BLASTN": "D"}
# Morandi bar palette (panel A)
C1, C2, E1, E2 = "#9EB6C6", "#DFCE95", "#7E97A8", "#C4B274"

fig = plt.figure(figsize=(15.5, 4.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.0], wspace=0.28)
axA, axB, axC = (fig.add_subplot(gs[i]) for i in range(3))

# ---- (A) Recall@1 grouped bars ----
xs = np.arange(len(METHODS))
fam = np.array([r1[m]["family_recall1"] for m in METHODS])
clan = np.array([r1[m]["clan_recall1"] for m in METHODS])
w = 0.38
b1 = axA.bar(xs - w / 2, fam, w, color=C1, edgecolor=E1, linewidth=0.8, label="Family level", zorder=3)
b2 = axA.bar(xs + w / 2, clan, w, color=C2, edgecolor=E2, linewidth=0.8, label="Clan level", zorder=3)
for bars, vals in ((b1, fam), (b2, clan)):
    for b, v in zip(bars, vals):
        axA.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=7.5, color="#333333")
axA.set_ylabel("Recall@1", fontsize=12)
axA.set_xticks(xs)
axA.set_xticklabels(METHODS, fontsize=9.5, rotation=30, ha="right")
axA.set_ylim(0, 1.05)
axA.set_yticks(np.arange(0, 1.01, 0.2))
axA.set_xlim(-0.6, len(METHODS) - 0.4)
axA.spines[["top", "right"]].set_visible(False)
axA.tick_params(axis="both", labelsize=10)
axA.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="none",
           fontsize=9.5, handlelength=1.4, borderpad=0.4)

# ---- (B,C) per-query PR curves ----
def pr_panel(ax, level, title):
    for m in METHODS:
        rec = np.asarray(wp[m][level]["recall"])
        prec = np.asarray(wp[m][level]["precision"])
        ap = wp[m][level]["aupr"]
        step = max(1, len(rec) // 10)
        ax.plot(rec, prec, "-", color=METHOD_COLOR[m], linewidth=2.0,
                marker=MARKER[m], markevery=step, markersize=5.5,
                markeredgecolor="white", markeredgewidth=0.6,
                label=f"{m} ({ap:.2f})", zorder=3)
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

pr_panel(axB, "family", "Family level")
pr_panel(axC, "clan", "Clan level")

# panel labels
for ax, lab in ((axA, "A"), (axB, "B"), (axC, "C")):
    ax.text(-0.12, 1.06, lab, transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")

fig.tight_layout(rect=(0, 0.02, 1, 1))
OUT = str(Path(FIG_DIR) / "S12.png")
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
