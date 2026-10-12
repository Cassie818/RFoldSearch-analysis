"""Supplementary virtual-centre figure: virtual-centre weight ablation, family + clan Sens@1FP vs
(alpha,beta,gamma) weights, K=80 deployed pipeline, PDB-disjoint 5-fold CV."""
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from paths import DATA_ROOT, FIG_DIR, SRC_MODELS, SRC_SEARCH, resolve

_sys.path.insert(0, SRC_SEARCH)
_sys.path.insert(0, SRC_MODELS)
_os.chdir(DATA_ROOT)
from pathlib import Path
import json, numpy as np, matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
import matplotlib.pyplot as plt

d = json.load(open(resolve("files/ablation/cv_vc_clan_5fold.json")))
NAMES = [
    ("0_0_0", "(0, 0, 0)\nC4$'$-only"),
    ("0.5_0.5_0.5", "(0.5, 0.5, 0.5)"),
    ("1_0.5_0.25", "(1, 0.5, 0.25)"),
]
xs = np.arange(len(NAMES))
fam_m = np.array([d[k[0]]["fam_sens_mean"] for k in NAMES])
fam_s = np.array([d[k[0]]["fam_sens_std"] for k in NAMES])
clan_m = np.array([d[k[0]]["clan_sens_mean"] for k in NAMES])
clan_s = np.array([d[k[0]]["clan_sens_std"] for k in NAMES])
fig, ax = plt.subplots(figsize=(7, 4.8))
C1 = "#5B7C99"
C2 = "#B08D5C"
ax.plot(
    xs,
    fam_m,
    "o-",
    color=C1,
    linewidth=2,
    markersize=7,
    label="Family Sens@1FP",
    markerfacecolor=C1,
    markeredgecolor="white",
    markeredgewidth=1,
    zorder=3,
)
for x, v, e in zip(xs, fam_m, fam_s):
    ax.vlines(x, v, v + e, colors="#333333", linewidth=1.0, alpha=0.9, zorder=2)
    ax.hlines(v + e, x - 0.10, x + 0.10, colors="#333333", linewidth=1.0, alpha=0.9, zorder=2)
ax.plot(
    xs,
    clan_m,
    "s-",
    color=C2,
    linewidth=2,
    markersize=7,
    label="Clan Sens@1FP",
    markerfacecolor=C2,
    markeredgecolor="white",
    markeredgewidth=1,
    zorder=3,
)
for x, v, e in zip(xs, clan_m, clan_s):
    ax.vlines(x, v, v + e, colors="#333333", linewidth=1.0, alpha=0.9, zorder=2)
    ax.hlines(v + e, x - 0.10, x + 0.10, colors="#333333", linewidth=1.0, alpha=0.9, zorder=2)
for x, v, e in zip(xs, fam_m, fam_s):
    ax.text(x, v + e + 0.012, f"{v:.3f}", ha="center", fontsize=8, color=C1)
for x, v, e in zip(xs, clan_m, clan_s):
    ax.text(x, v + e + 0.012, f"{v:.3f}", ha="center", fontsize=8, color=C2)
ax.set_xlabel(r"Virtual-centre weights $(\alpha,\beta,\gamma)$", fontsize=12)
ax.xaxis.set_label_coords(0.5, -0.14)
ax.set_ylabel("Sens@1FP", fontsize=12)
ax.set_xticks(xs)
ax.set_xticklabels([n[1] for n in NAMES], fontsize=10)
ax.set_xlim(-0.4, len(NAMES) - 0.6)
ax.set_ylim(0, 0.6)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(
    loc="lower center",
    frameon=False,
    fontsize=11,
    ncol=2,
    bbox_to_anchor=(0.5, -0.34),
    handlelength=2.0,
    columnspacing=2.5,
)
fig.tight_layout(rect=(0, 0.07, 1, 1))
OUT = str(Path(FIG_DIR) / "supp_virtual_centre.png")
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
