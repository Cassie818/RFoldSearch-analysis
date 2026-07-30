"""Fig S12: Rfam family- and clan-level Recall@1 for the benchmark methods
on the held-out test partition (same query sets as Fig 2A/B). Grouped bars."""
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
d = json.load(open(REPO / "files/production/retrieval/recall1_rfam_test.json"))
METHODS = ["RFoldseek", "GTalign", "RTM-align", "US-align", "RMalign", "RiboSeek", "BLASTN"]
xs = np.arange(len(METHODS))
fam = np.array([d[m]["family_recall1"] for m in METHODS])
clan = np.array([d[m]["clan_recall1"] for m in METHODS])
C1 = "#9EB6C6"  # family level — Morandi muted blue
C2 = "#DFCE95"  # clan level   — Morandi muted yellow
E1 = "#7E97A8"  # muted-blue edge (defines the pale fill on a white surface)
E2 = "#C4B274"  # muted-yellow edge
fig, ax = plt.subplots(figsize=(7.2, 4.4))
w = 0.38
b1 = ax.bar(xs - w / 2, fam, w, color=C1, edgecolor=E1, linewidth=0.8, label="Family level", zorder=3)
b2 = ax.bar(xs + w / 2, clan, w, color=C2, edgecolor=E2, linewidth=0.8, label="Clan level", zorder=3)
for bars, vals in ((b1, fam), (b2, clan)):
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.012,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )
ax.set_ylabel("Recall@1", fontsize=12)
ax.set_xticks(xs)
ax.set_xticklabels(METHODS, fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_xlim(-0.6, len(METHODS) - 0.4)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="both", labelsize=10)
ax.legend(
    loc="lower center",
    frameon=False,
    fontsize=11,
    ncol=2,
    bbox_to_anchor=(0.5, -0.26),
    handlelength=1.6,
    columnspacing=2.5,
)
fig.tight_layout(rect=(0, 0.05, 1, 1))
OUT = str(Path(FIG_DIR) / "S12.png")
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
