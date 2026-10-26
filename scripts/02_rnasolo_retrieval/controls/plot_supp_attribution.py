"""Supplementary attribution figure: forward and reverse component replacement."""
import json
import os as _os
import sys as _sys
from pathlib import Path

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
from paths import DATA_ROOT, FIG_DIR, resolve

_os.chdir(DATA_ROOT)
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
import matplotlib.pyplot as plt

C_RF, C_RB = "#4FB3E3", "#EC9BD3"
STEP_FILL, STEP_EDGE, STEP_TEXT = "#C8D3DA", "#8FA3AE", "#4A6472"
NEG_FILL, NEG_EDGE, NEG_TEXT = "#E2E2E2", "#BDBDBD", "#888888"
ATTR = json.load(open(resolve("files/ablation/attribution_paths.json")))
ATTR_S = json.load(open(resolve("files/ablation/attribution_paths_structural.json")))
TAU = "0.45"

ORDERS = {
    "forward": (["A_ribo_published", "B_fwd_pipeline", "C_fwd_matrix", "D_3dr_sw", "D_rfoldsearch"],
                ["+ RFoldSearch pipeline", "+ learned matrix", "+ 3Dr alphabet", "+ structure score"],
                ["RiboSeek", "+ RFoldSearch\npipeline", "+ learned\nmatrix", "+ 3Dr\nalphabet",
                 "+ structure\nscore", "RFoldSearch"]),
    # the reverse pipeline step already includes structure-score ranking
    "reverse": (["A_ribo_published", "E0_rev_alphabet", "E_rev_alphabet", "D_rfoldsearch"],
                ["+ 3Dr alphabet", "+ learned matrix", "+ RFoldSearch pipeline"],
                ["RiboSeek", "+ 3Dr\nalphabet", "+ learned\nmatrix", "+ RFoldSearch\npipeline",
                 "RFoldSearch"]),
}
LEVELS = [
    (ATTR["levels"]["family"], "Family (Rfam)"),
    (ATTR["levels"]["clan"], "Clan (Rfam)"),
    (ATTR_S["thresholds"][TAU]["family"], f"All (TM $\\geq$ {TAU})"),
    (ATTR_S["thresholds"][TAU]["clan"], f"Beyond-family (TM $\\geq$ {TAU})"),
]


def waterfall_top(level, order):
    keys, steps, _ = ORDERS[order]
    means, estimates = level["means"], level[order]
    values = [means[key] for key in keys]
    return max(max(values), max(means[key] + max(estimates[label]["hi"] - estimates[label]["diff"], 0)
                                for key, label in zip(keys[1:], steps)))


def waterfall_panel(ax, level, order, title, ymax):
    keys, steps, ticklabels = ORDERS[order]
    means, estimates = level["means"], level[order]
    values = [means[key] for key in keys]
    ax.bar(0, values[0], 0.62, color=C_RB, edgecolor="#B05A94", linewidth=0.6, zorder=3)
    ax.text(0, values[0] + ymax * 0.008, f"{values[0]:.3f}", ha="center", va="bottom",
            fontsize=8, color="#E377C2")
    for index, label in enumerate(steps, start=1):
        previous, current = values[index - 1], values[index]
        low, high = min(previous, current), max(previous, current)
        estimate = estimates[label]
        positive = estimate["diff"] > 0
        ax.bar(index, max(high - low, ymax * 0.005), 0.62, bottom=low,
               color=STEP_FILL if positive else NEG_FILL,
               edgecolor=STEP_EDGE if positive else NEG_EDGE, linewidth=0.6, zorder=3)
        ax.errorbar(index, high,
                    yerr=[[max(estimate["diff"] - estimate["lo"], 0)],
                          [max(estimate["hi"] - estimate["diff"], 0)]],
                    fmt="none", ecolor="#555555", elinewidth=0.9, capsize=2.5, zorder=4)
        ax.text(index, high + max(estimate["hi"] - estimate["diff"], 0) + ymax * 0.02,
                f"{estimate['diff']:+.3f}{'*' if estimate['sig'] else ''}",
                ha="center", va="bottom", fontsize=7.5,
                color=STEP_TEXT if positive else NEG_TEXT)
        ax.plot([index - 1 + 0.31, index - 0.31], [previous, previous],
                color="#999999", lw=0.8, ls=(0, (3, 2)), zorder=2)
    last = len(steps) + 1
    ax.bar(last, values[-1], 0.62, color=C_RF, edgecolor="#15709C", linewidth=0.6, zorder=3)
    ax.text(last, values[-1] + ymax * 0.008, f"{values[-1]:.3f}", ha="center", va="bottom",
            fontsize=8, color="#1F9BD8")
    ax.plot([last - 0.69, last - 0.31], [values[-1], values[-1]], color="#999999", lw=0.8,
            ls=(0, (3, 2)), zorder=2)
    ax.set_xticks(range(last + 1))
    ax.set_xticklabels(ticklabels, fontsize=6.8)
    ax.set_xlim(-0.55, last + 0.55)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("Sens@1FP", fontsize=10)
    ax.set_title(title, fontsize=10.5, pad=5)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    fig, axes = plt.subplots(4, 2, figsize=(9.4, 10.0))
    for row, (level, title) in enumerate(LEVELS):
        ymax = 1.28 * max(waterfall_top(level, order) for order in ORDERS)
        waterfall_panel(axes[row, 0], level, "forward", title, ymax)
        waterfall_panel(axes[row, 1], level, "reverse", title, ymax)
    for ax, text in zip(axes[0], ("Forward replacement order", "Reverse replacement order")):
        ax.text(0.5, 1.22, text, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=11, color="#555555")
    for ax, label in zip(axes.flat, "ABCDEFGH"):
        ax.text(-0.11, 1.04, label, transform=ax.transAxes, fontsize=14,
                fontweight="bold", va="bottom", ha="right")
    fig.subplots_adjust(left=0.075, right=0.995, top=0.945, bottom=0.05, wspace=0.2, hspace=0.62)
    out_png = Path(FIG_DIR) / "supp_attribution.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
