"""Supplementary attribution-control figure: scoring controls and reverse-order attribution robustness."""
import json
import os as _os
import sys as _sys
from pathlib import Path

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
from paths import DATA_ROOT, FIG_DIR, resolve

_os.chdir(DATA_ROOT)
import matplotlib
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
import matplotlib.pyplot as plt

RF, RB = "RFoldSearch_tokens_same_pipeline", "RiboSeek_sa20_same_pipeline"
REGIMES = [("", "Native", "native"), ("_identity", "Identity", "identity"),
           ("_self_blosum", "Self-BLOSUM", "self_blosum")]
C_RF, C_RB = "#4FB3E3", "#EC9BD3"
STEP_FILL, STEP_TEXT = "#C8D3DA", "#4A6472"
BOOT = json.load(open(resolve("files/production/retrieval/bootstrap_controls.json")))
ATTR = json.load(open(resolve("files/ablation/attribution_paths.json")))
ATTR_S = json.load(open(resolve("files/ablation/attribution_paths_structural.json")))
TAU = "0.45"


def regime_panel(ax, key, title):
    x = np.arange(len(REGIMES)); width = 0.36
    values = {}
    for method, tag in (("RFoldSearch", RF), ("RiboSeek", RB)):
        values[method] = [json.load(open(resolve(
            f"files/ablation/riboseek_alphabet_same_pipeline_control{suffix}.json"
        )))[tag][key] for suffix, _, _ in REGIMES]
    for method, offset, colour in (("RFoldSearch", -width / 2, C_RF),
                                    ("RiboSeek", width / 2, C_RB)):
        ax.bar(x + offset, values[method], width, color=colour, edgecolor="#666666",
               linewidth=0.6, label=method, zorder=3)
        for xpos, value in zip(x + offset, values[method]):
            ax.text(xpos, value + 0.008, f"{value:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color="#1F9BD8" if method == "RFoldSearch" else "#E377C2")
    level = "family" if key.startswith("family") else "clan"
    for index, (_, _, regime_key) in enumerate(REGIMES):
        estimate = BOOT["alphabet_control"][regime_key][level]
        top = max(values["RFoldSearch"][index], values["RiboSeek"][index]) + 0.045
        ax.plot([x[index] - width / 2, x[index] - width / 2,
                 x[index] + width / 2, x[index] + width / 2],
                [top, top + 0.012, top + 0.012, top], lw=0.9,
                color="#777777", zorder=4)
        ax.text(x[index], top + 0.016,
                f"{estimate['diff_mean']:+.3f}{'*' if estimate['sig'] else ''}",
                ha="center", va="bottom", fontsize=8, color="#333333")
    ax.set_xticks(x, [label for _, label, _ in REGIMES], fontsize=9)
    ax.set_xlabel("Scoring regime", fontsize=11)
    ax.set_ylabel("Sens@1FP", fontsize=11)
    ax.set_ylim(0, max(max(v) for v in values.values()) * 1.42)
    ax.set_title(title, fontsize=12)
    ax.tick_params(labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)


def reverse_panel(ax, level, title):
    means, reverse = level["means"], level["reverse"]
    keys = ["A_ribo_published", "E0_rev_alphabet", "E_rev_alphabet", "D_rfoldsearch"]
    steps = ["+ 3Dr alphabet", "+ learned matrix", "+ RFoldSearch pipeline"]
    values = [means[key] for key in keys]
    ax.bar(0, values[0], 0.60, color=C_RB, edgecolor="#B05A94", linewidth=0.6,
           zorder=3)
    ax.text(0, values[0] + 0.006, f"{values[0]:.3f}", ha="center", va="bottom",
            fontsize=8, color="#E377C2")
    for index, label in enumerate(steps, start=1):
        low, high = sorted((values[index - 1], values[index]))
        estimate = reverse[label]
        ax.bar(index, high - low, 0.60, bottom=low, color=STEP_FILL,
               edgecolor="#8FA3AE", linewidth=0.6, zorder=3)
        ax.errorbar(index, high,
                    yerr=[[max(estimate["diff"] - estimate["lo"], 0)],
                          [max(estimate["hi"] - estimate["diff"], 0)]],
                    fmt="none", ecolor="#555555", elinewidth=0.9, capsize=2.5,
                    zorder=4)
        ax.text(index,
                high + max(estimate["hi"] - estimate["diff"], 0)
                + max(values) * 0.025,
                f"{estimate['diff']:+.3f}{'*' if estimate['sig'] else ''}",
                ha="center", va="bottom", fontsize=8, color=STEP_TEXT)
        ax.plot([index - 1 + 0.30, index - 0.30],
                [values[index - 1], values[index - 1]], color="#999999",
                lw=0.8, ls=(0, (3, 2)), zorder=2)
    last = len(steps) + 1
    ax.bar(last, values[-1], 0.60, color=C_RF, edgecolor="#15709C", linewidth=0.6,
           zorder=3)
    ax.text(last, values[-1] + 0.006, f"{values[-1]:.3f}", ha="center", va="bottom",
            fontsize=8, color="#1F9BD8")
    ax.plot([last - 0.70, last - 0.30], [values[-1], values[-1]], color="#999999", lw=0.8,
            ls=(0, (3, 2)), zorder=2)
    ax.set_xticks(range(last + 1))
    ax.set_xticklabels(["RiboSeek", "+ 3Dr\nalphabet", "+ learned\nmatrix",
                        "+ RFoldSearch\npipeline", "RFoldSearch"], fontsize=8)
    ax.set_ylabel("Sens@1FP", fontsize=11)
    top = max(max(values), max(means[key] + max(reverse[label]["hi"]
              - reverse[label]["diff"], 0) for key, label in zip(keys[1:], steps)))
    ax.set_ylim(0, top * 1.28)
    ax.set_title(title, fontsize=12)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    fig, axes = plt.subplots(3, 2, figsize=(10.6, 12.4))
    regime_panel(axes[0, 0], "family_sens_1fp", "Family")
    regime_panel(axes[0, 1], "clan_sens_1fp", "Clan")
    reverse_panel(axes[1, 0], ATTR["levels"]["family"], "Family (Rfam)")
    reverse_panel(axes[1, 1], ATTR["levels"]["clan"], "Clan (Rfam)")
    reverse_panel(axes[2, 0], ATTR_S["thresholds"][TAU]["family"],
                  f"All (TM $\\geq$ {TAU})")
    reverse_panel(axes[2, 1], ATTR_S["thresholds"][TAU]["clan"],
                  f"Beyond-family (TM $\\geq$ {TAU})")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.004))
    for ax, label in zip(axes.flat, "ABCDEF"):
        ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=15,
                fontweight="bold", va="top")
    fig.tight_layout(rect=(0, 0.024, 1, 1))
    out_png = Path(FIG_DIR) / "supp_attribution_controls.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
