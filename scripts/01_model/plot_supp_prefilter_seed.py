"""Supplementary prefilter seed-pattern figure: prefilter seed-pattern ablation (E4/G4/S4/SimE4), 5-fold CV,
deployed VQ-VAE + Structure-BLOSUM reranker. Line plot in the style of the other ablation figures."""
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

d = json.load(open(resolve("files/ablation/cv_s7_clan_5fold.json")))
NAMES = [
    ("E4", "E4\n1111"),
    ("G4", "G4\n11_11"),
    ("S4", "S4\n1_1_1_1"),
    ("SimE4", "SimE4\n1111 + expansion"),
]
xs = np.arange(len(NAMES))
fam_m = np.array([d[k[0]]["fam_sens_mean"] for k in NAMES])
fam_s = np.array([d[k[0]]["fam_sens_std"] for k in NAMES])
clan_m = np.array([d[k[0]]["clan_sens_mean"] for k in NAMES])
clan_s = np.array([d[k[0]]["clan_sens_std"] for k in NAMES])
fig, ax = plt.subplots(figsize=(8, 4.8))
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
    ax.hlines(v + e, x - 0.16, x + 0.16, colors="#333333", linewidth=1.0, alpha=0.9, zorder=2)
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
    ax.hlines(v + e, x - 0.16, x + 0.16, colors="#333333", linewidth=1.0, alpha=0.9, zorder=2)
for x, v, e in zip(xs, fam_m, fam_s):
    ax.text(x, v + e + 0.012, f"{v:.3f}", ha="center", fontsize=8, color=C1)
for x, v, e in zip(xs, clan_m, clan_s):
    ax.text(x, v + e + 0.012, f"{v:.3f}", ha="center", fontsize=8, color=C2)
ax.set_xlabel("Prefilter seed pattern", fontsize=12)
ax.set_ylabel("Sens@1FP", fontsize=12)
ax.set_xticks(xs)
ax.set_xticklabels([n[1] for n in NAMES], fontsize=10)
ax.set_ylim(0, 0.5)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(
    loc="lower center",
    frameon=False,
    fontsize=11,
    ncol=2,
    bbox_to_anchor=(0.5, -0.35),
    handlelength=2.0,
    columnspacing=2.5,
)
fig.tight_layout(rect=(0, 0.08, 1, 1))
OUT = str(Path(FIG_DIR) / "supp_prefilter_seed.png")
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
