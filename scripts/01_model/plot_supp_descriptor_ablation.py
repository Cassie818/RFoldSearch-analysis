"""Supplementary descriptor-ablation figure: single blocks (top row) and block combinations
(bottom row), 5-fold CV, family (left) and clan (right) Sens@1FP, with a shared bottom legend."""
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

ROWS = [
    (
        "files/ablation/cv_s2_clan_5fold.json",
        ["g_nn1", "g_nn2", "g_bp", "g_st", "g_ctx"],
        [
            "$g_{\\mathrm{nn}}^{(1)}$",
            "$g_{\\mathrm{nn}}^{(2)}$",
            "$g_{\\mathrm{bp}} + m_{\\mathrm{bp}}$",
            "$g_{\\mathrm{st}} + m_{\\mathrm{st}}$",
            "$g_{\\mathrm{ctx}}$",
        ],
        11,
    ),
    (
        "files/ablation/cv_s3_clan_5fold.json",
        [
            "ctx+nn1 (17D)",
            "ctx+nn1+nn2 (27D)",
            "ctx+nn1+bp (28D)",
            "ctx+nn1+nn2+bp (38D)",
            "ctx+nn1+bp+st (39D)",
        ],
        [
            "ctx + nn1\n(17D)",
            "ctx + nn1\n+ nn2 (27D)",
            "ctx + nn1\n+ bp (28D)",
            "ctx + nn1\n+ nn2 + bp (38D)",
            "ctx + nn1\n+ bp + st (39D)",
        ],
        9.5,
    ),
]
METRICS = [
    ("fam_sens_mean", "fam_sens_std", "Family Sens@1FP", 0.6),
    ("clan_sens_mean", "clan_sens_std", "Clan Sens@1FP", 0.18),
]
KS = [20, 40, 60, 80]
COLORS = ["#5B7C99", "#B08D5C", "#7DA087", "#9B6B8A"]
fig, axes = plt.subplots(2, 2, figsize=(13, 9.6))
for ri, (path, keys, labels, tick_size) in enumerate(ROWS):
    d = json.load(open(resolve(path)))
    for pi, (mkey, skey, ylab, ylim) in enumerate(METRICS):
        ax = axes[ri, pi]
        x = np.arange(len(keys))
        w = 0.21
        for ki, K in enumerate(KS):
            vals = np.array([d[f"{b}|K{K}"][mkey] for b in keys])
            errs = np.array([d[f"{b}|K{K}"][skey] for b in keys])
            yerr_asym = np.vstack([np.zeros_like(errs), errs])  # lower=0, upper=std
            bar_xs = x + (ki - 1.5) * w
            ax.bar(
                bar_xs,
                vals,
                w,
                yerr=yerr_asym,
                color=COLORS[ki],
                label=f"K = {K}",
                edgecolor="none",
                error_kw=dict(elinewidth=1.0, ecolor="#333333", capsize=0, alpha=0.9),
            )
            # draw the error-bar cap only on the upper side
            cap_half = w * 0.28
            for xb, v, e in zip(bar_xs, vals, errs):
                ax.hlines(
                    v + e, xb - cap_half, xb + cap_half, colors="#333333", linewidth=1.0, alpha=0.9
                )
            for j, v in enumerate(vals):
                ax.text(
                    x[j] + (ki - 1.5) * w,
                    v + errs[j] + ylim * 0.022,
                    f"{v:.3f}",
                    ha="center",
                    fontsize=5.5,
                    color=COLORS[ki],
                )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=tick_size)
        ax.set_ylabel(ylab, fontsize=11)
        ax.set_ylim(0, ylim)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(
            -0.09,
            1.04,
            "ABCD"[2 * ri + pi],
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            ha="right",
        )
# single shared legend along the bottom
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=4,
    frameon=False,
    fontsize=12,
    bbox_to_anchor=(0.5, -0.01),
    handlelength=1.4,
    columnspacing=2.2,
)
fig.tight_layout(rect=(0, 0.03, 1, 1), h_pad=2.5)
OUT = str(Path(FIG_DIR) / "supp_descriptor_ablation.png")
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
