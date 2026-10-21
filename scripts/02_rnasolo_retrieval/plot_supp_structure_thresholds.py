"""Supplementary figure: structure-defined truth at the TM-score thresholds not shown in
Fig. 2C/D (0.40 and 0.50).

    A/B  TM-score >= 0.40, family (A) and clan (B); RMscore truth at the quantile-matched
         threshold below each
    C/D  The same at TM-score >= 0.50
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_fig2 import (FIG_DIR, LABEL, LEGEND_ORDER, METHOD_COLOR, TM_THRESHOLD,  # noqa: E402
                       load_truths, plt, truth_pair)
from matplotlib.patches import Patch  # noqa: E402


def main():
    boot, boot_rm, rm_thresholds = load_truths()
    keys = [k for k in sorted(boot, key=float) if k != TM_THRESHOLD]

    fig = plt.figure(figsize=(14.0, 13.0))
    gs = fig.add_gridspec(len(keys), 2, hspace=0.16, wspace=0.16,
                          left=0.06, right=0.99, top=0.96, bottom=0.07)
    letter_axes = []
    for row, key in enumerate(keys):
        for col, level in enumerate(("family", "clan")):
            letter_axes.append(truth_pair(fig, gs[row, col], boot, boot_rm, rm_thresholds,
                                          key, level))
    for ax, lab in zip(letter_axes, "ABCD"):
        ax.text(-0.10, 1.04, lab, transform=ax.transAxes, fontsize=16, fontweight="bold",
                va="bottom", ha="right")

    handles = [Patch(facecolor=METHOD_COLOR[m], edgecolor="#666666", linewidth=0.6)
               for m in LEGEND_ORDER]
    fig.legend(handles, [LABEL.get(m, m) for m in LEGEND_ORDER], loc="lower center", ncol=5,
               frameon=False, fontsize=11.5, bbox_to_anchor=(0.5, -0.02), columnspacing=1.8)
    out_png = Path(FIG_DIR) / "supp_structure_thresholds.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
