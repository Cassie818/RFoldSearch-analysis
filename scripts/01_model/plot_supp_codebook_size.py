"""Supplementary codebook-size figure: VQ-VAE codebook-size sweep (deployed pipeline, PDB-disjoint 5-fold CV),
family-level and clan-level Sens@1FP vs K in {20,40,60,80}."""
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

d = json.load(open(resolve("files/ablation/cv_s4_clan_5fold.json")))
KS = [20, 40, 60, 80, 100]
fam_m = np.array([d[f"K{K}"]["fam_sens_mean"] for K in KS])
fam_s = np.array([d[f"K{K}"]["fam_sens_std"] for K in KS])
clan_m = np.array([d[f"K{K}"]["clan_sens_mean"] for K in KS])
clan_s = np.array([d[f"K{K}"]["clan_sens_std"] for K in KS])
fig, ax = plt.subplots(figsize=(7, 4.5))
C1 = "#5B7C99"
C2 = "#B08D5C"
ax.plot(
    KS,
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
for k, v, e in zip(KS, fam_m, fam_s):
    ax.vlines(k, v, v + e, colors="#333333", linewidth=1.0, alpha=0.9, zorder=2)
    ax.hlines(v + e, k - 2.0, k + 2.0, colors="#333333", linewidth=1.0, alpha=0.9, zorder=2)
ax.plot(
    KS,
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
for k, v, e in zip(KS, clan_m, clan_s):
    ax.vlines(k, v, v + e, colors="#333333", linewidth=1.0, alpha=0.9, zorder=2)
    ax.hlines(v + e, k - 2.0, k + 2.0, colors="#333333", linewidth=1.0, alpha=0.9, zorder=2)
for k, v, e in zip(KS, fam_m, fam_s):
    ax.text(k, v + e + 0.012, f"{v:.3f}", ha="center", fontsize=8, color=C1)
for k, v, e in zip(KS, clan_m, clan_s):
    ax.text(k, v + e + 0.012, f"{v:.3f}", ha="center", fontsize=8, color=C2)
ax.set_xlabel("Codebook size $K$", fontsize=12)
ax.set_ylabel("Sens@1FP", fontsize=12)
ax.set_xticks(KS)
ax.set_xticklabels([str(K) for K in KS], fontsize=11)
ax.set_ylim(0, 0.5)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(
    loc="lower center",
    frameon=False,
    fontsize=11,
    ncol=2,
    bbox_to_anchor=(0.5, -0.30),
    handlelength=2.0,
    columnspacing=2.5,
)
fig.tight_layout(rect=(0, 0.06, 1, 1))
OUT = str(Path(FIG_DIR) / "supp_codebook_size.png")
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
