"""Supplementary forward-attribution figure: forward attribution of the RiboSeek-to-RFoldSearch gap."""
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
STEP_FILL, STEP_TEXT = "#C8D3DA", "#4A6472"
ATTR = json.load(open(resolve("files/ablation/attribution_paths.json")))
ATTR_S = json.load(open(resolve("files/ablation/attribution_paths_structural.json")))
TAU = "0.45"


def forward_panel(ax, level, title):
    means, forward = level["means"], level["forward"]
    keys = ["A_ribo_published", "B_fwd_pipeline", "C_fwd_matrix", "D_3dr_sw", "D_rfoldsearch"]
    steps = ["+ RFoldSearch pipeline", "+ learned matrix", "+ 3Dr alphabet", "+ structure score"]
    values = [means[key] for key in keys]

    ax.bar(0, values[0], 0.62, color=C_RB, edgecolor="#B05A94", linewidth=0.6,
           zorder=3)
    ax.text(0, values[0] + 0.007, f"{values[0]:.3f}", ha="center", va="bottom",
            fontsize=8.5, color="#E377C2")

    for index, label in enumerate(steps, start=1):
        previous, current = values[index - 1], values[index]
        low, high = min(previous, current), max(previous, current)
        estimate = forward[label]
        span = max(high - low, max(values) * 0.006)
        fill = STEP_FILL if estimate["diff"] > 0 else "#E2E2E2"
        edge = "#8FA3AE" if estimate["diff"] > 0 else "#BDBDBD"
        ax.bar(index, span, 0.62, bottom=low, color=fill, edgecolor=edge,
               linewidth=0.6, zorder=3)
        ax.errorbar(index, high,
                    yerr=[[max(estimate["diff"] - estimate["lo"], 0)],
                          [max(estimate["hi"] - estimate["diff"], 0)]],
                    fmt="none", ecolor="#555555", elinewidth=0.9, capsize=2.5,
                    zorder=4)
        ax.text(index,
                high + max(estimate["hi"] - estimate["diff"], 0)
                + max(values) * 0.025,
                f"{estimate['diff']:+.3f}{'*' if estimate['sig'] else ''}",
                ha="center", va="bottom", fontsize=8,
                color=STEP_TEXT if estimate["diff"] > 0 else "#888888")
        ax.plot([index - 1 + 0.31, index - 0.31], [previous, previous],
                color="#999999", lw=0.8, ls=(0, (3, 2)), zorder=2)

    last = len(steps) + 1
    ax.bar(last, values[-1], 0.62, color=C_RF, edgecolor="#15709C", linewidth=0.6,
           zorder=3)
    ax.text(last, values[-1] + 0.007, f"{values[-1]:.3f}", ha="center", va="bottom",
            fontsize=8.5, color="#1F9BD8")
    ax.plot([last - 0.69, last - 0.31], [values[-1], values[-1]], color="#999999", lw=0.8,
            ls=(0, (3, 2)), zorder=2)

    ax.set_xticks(range(last + 1))
    ax.set_xticklabels(["RiboSeek", "+ RFoldSearch\npipeline", "+ learned\nmatrix",
                        "+ 3Dr\nalphabet", "+ structure\nscore", "RFoldSearch"], fontsize=8)
    ax.set_ylabel("Sens@1FP", fontsize=11)
    top = max(
        max(values),
        max(means[key] + max(forward[label]["hi"] - forward[label]["diff"], 0)
            for key, label in zip(keys[1:], steps)),
    )
    ax.set_ylim(0, top * 1.28)
    ax.set_xlim(-0.55, last + 0.55)
    ax.set_title(title, fontsize=12, pad=7)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.6))
    forward_panel(axes[0, 0], ATTR["levels"]["family"], "Family (Rfam)")
    forward_panel(axes[0, 1], ATTR["levels"]["clan"], "Clan (Rfam)")
    forward_panel(axes[1, 0], ATTR_S["thresholds"][TAU]["family"],
                  f"Family (TM $\\geq$ {TAU})")
    forward_panel(axes[1, 1], ATTR_S["thresholds"][TAU]["clan"],
                  f"Clan (TM $\\geq$ {TAU})")
    for ax, label in zip(axes.flat, "ABCD"):
        ax.text(-0.10, 1.04, label, transform=ax.transAxes, fontsize=15,
                fontweight="bold", va="bottom", ha="right")
    fig.tight_layout()
    out_png = Path(FIG_DIR) / "supp_attribution_forward.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
