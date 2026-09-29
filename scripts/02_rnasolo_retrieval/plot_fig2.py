"""Fig. 2: retrieval on the RNAsolo benchmark, 2x2.

    A/B  Rfam AUROC1 sorted-sensitivity curves (all methods; RFoldseek 95% band)
    C/D  Structure-defined (US-align TM-score) Sens@1FP bars, by TM threshold

Writes figures/fig2.png. Its colours, labels and markers are reused by the other
figures (scripts/common/figstyle.py, scripts/04_alignment_benchmark).
"""
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from paths import DATA_ROOT, FIG_DIR, SRC_SEARCH, resolve

_sys.path.insert(0, SRC_SEARCH)
_os.chdir(DATA_ROOT)
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
import matplotlib.pyplot as plt
import numpy as np

# ---- shared style (identical to plot_fig2.py) --------------------------------
METHODS = ["RFoldseek", "RFoldseek-TM", "GTalign", "RTM-align", "US-align", "RMalign",
           "RiboSeek", "RiboSeek RS-80", "BLASTN", "Infernal"]
# RiboSeek 0.2.0: RS-20 (geometry only; the "RiboSeek" key) and the default
# RS-80 (RS-20 x nucleotide identity) share one hue, RS-80 in the deeper shade.
METHOD_COLOR = {
    "RFoldseek": "#1F9BD8", "GTalign": "#9467BD", "RTM-align": "#FF7F0E",
    "RFoldseek-TM": "#009E9E", "US-align": "#D62728", "RMalign": "#2CA02C",
    "RiboSeek": "#E377C2", "RiboSeek RS-80": "#A8327F",
    "BLASTN": "#FFC107", "Infernal": "#BCBD22",
}
LABEL = {"RiboSeek": "RiboSeek (RS-20)", "RiboSeek RS-80": "RiboSeek (RS-80)"}


def _lighten(h, f=0.24):
    """Blend a hex colour toward white by fraction f (softer, lighter tone)."""
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02x%02x%02x" % tuple(round(c + (255 - c) * f) for c in (r, g, b))


METHOD_BASE = dict(METHOD_COLOR)          # text uses the unlightened tone
METHOD_COLOR = {k: _lighten(v) for k, v in METHOD_COLOR.items()}
# Keep Infernal at the exact olive specified for this baseline.
METHOD_COLOR["Infernal"] = METHOD_BASE["Infernal"]
LINESTYLE = {m: "-" for m in METHODS}
LINESTYLE["BLASTN"] = "--"
MARKER = {"RFoldseek": "o", "RFoldseek-TM": "*", "GTalign": "X", "RTM-align": "P", "US-align": "s",
          "RMalign": "^", "RiboSeek": "v", "RiboSeek RS-80": "v", "BLASTN": "D", "Infernal": "p"}
# C/D: US-align defines the structural truth (ceiling 1). GTalign and RTM-align
# rank by their own TM-scores and are near-oracles, shown for completeness.
# RFoldseek-TM ranks the RFoldseek prefilter candidates by the truth TM-score, so
# its Sens@1FP equals the prefilter recall (structure_truth_seeded.py); it is
# not drawn here.
BAR_METHODS = ["RFoldseek", "GTalign", "RTM-align", "RMalign",
               "RiboSeek", "RiboSeek RS-80", "BLASTN", "Infernal"]

def sorted_curve(arr):
    a = np.sort(np.asarray(arr))[::-1]
    return np.arange(1, len(a) + 1) / len(a), a


# ---- row 1: AUROC1 curves ----------------------------------------------------
def curve_panel(ax, data, level, title, band):
    for m in METHODS:
        x, y = sorted_curve(data[m][level]["sens"])
        step = max(1, len(x) // 18)
        marker_size = 11 if m == "RFoldseek-TM" else 7
        ax.plot(x, y, LINESTYLE[m], color=METHOD_COLOR[m], linewidth=2.4, marker=MARKER[m],
                markevery=step, markersize=marker_size, markeredgecolor="white", markeredgewidth=0.9,
                label=LABEL.get(m, m), zorder=3)
    ax.set_xlabel("Fraction of queries", fontsize=12)
    ax.set_ylabel("Sensitivity up to the 1st FP", fontsize=12)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title(title, fontsize=13, pad=8)
    ax.tick_params(axis="both", labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)


# ---- row 2: structural Sens@1FP bars -----------------------------------------
def bar_panel(ax, tm, metric, title, ylabel, boot, level):
    thr = sorted(float(t) for t in tm["thresholds"]); keys = [str(t) for t in thr]
    x = np.arange(len(thr)); width = 0.82 / len(BAR_METHODS)
    for i, m in enumerate(BAR_METHODS):
        off = (i - (len(BAR_METHODS) - 1) / 2) * width
        vals = [boot[k][level][m]["estimate"] if m in boot.get(k, {}).get(level, {})
                else tm["thresholds"][k][m][metric] for k in keys]
        yerr = None
        if all(m in boot.get(k, {}).get(level, {}) for k in keys):
            yerr = [[vals[j] - boot[keys[j]][level][m]["ci_lower"] for j in range(len(keys))],
                    [boot[keys[j]][level][m]["ci_upper"] - vals[j] for j in range(len(keys))]]
        bars = ax.bar(x + off, vals, width, color=METHOD_COLOR[m], edgecolor="#666666",
                      linewidth=0.6, label=LABEL.get(m, m), zorder=3, yerr=yerr, capsize=2.5,
                      error_kw={"elinewidth": 0.9, "ecolor": "#555555"})
        for j, (b, v) in enumerate(zip(bars, vals)):
            top = v + (yerr[1][j] if yerr else 0.0) + 0.012
            sig = boot.get(keys[j], {}).get(level, {}).get(m, {}).get("sig_vs_rf")
            # Only annotate RFoldseek numerically (others read from the y-axis);
            # significance stars sit on non-RFoldseek bars, visually detached,
            # and the "* vs RFoldseek" note in the corner disambiguates them.
            if m == "RFoldseek":
                ax.text(b.get_x() + b.get_width() / 2, top, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8.5,
                        color=METHOD_BASE["RFoldseek"])
            elif sig:
                ax.text(b.get_x() + b.get_width() / 2, top, "*",
                        ha="center", va="bottom", fontsize=10, color="#555555",
                        fontweight="bold")
    ceil = tm["thresholds"][keys[0]].get("US-align", tm["thresholds"][keys[0]].get("US-align*"))[metric]
    ax.plot([-0.5, len(thr) - 0.5], [ceil, ceil], ls="--", color=METHOD_COLOR["US-align"],
            lw=1.5, zorder=2)
    ax.text(-0.44, ceil + 0.012, "US-align (oracle ceiling)", ha="left",
            va="bottom", fontsize=8.5, color=METHOD_BASE["US-align"])
    ax.set_xticks(x, [f"{t:.2f}" for t in thr])
    ax.set_xlabel("TM-score similarity threshold", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim(-0.5, len(thr) - 0.5); ax.set_ylim(0, 1.08)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_title(title, fontsize=13, pad=8)
    ax.tick_params(axis="both", labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)



def main():
    data = json.load(open(resolve("files/production/retrieval/per_query_sens_test.json")))
    tm = json.load(open(resolve("files/production/retrieval/structural_tmscore_metrics_test.json")))
    boot = json.load(open(resolve("files/production/retrieval/bootstrap_fig2cd_results.json")))
    band = json.load(open(resolve("files/production/retrieval/bootstrap_fig2ab_band.json")))

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.4))
    curve_panel(axes[0, 0], data, "family", "Family (AUROC1)", band)
    curve_panel(axes[0, 1], data, "clan", "Clan (AUROC1)", band)
    bar_panel(axes[1, 0], tm, "Sens@1FP", "Family", "Sens@1FP", boot, "family")
    bar_panel(axes[1, 1], tm, "Sens@1FP_clan", "Clan", "Sens@1FP", boot, "clan")

    for ax, lab in zip(axes.flat, "ABCD"):
        ax.text(-0.10, 1.04, lab, transform=ax.transAxes, fontsize=16, fontweight="bold",
                va="bottom", ha="right")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Column-major, one pair per column: RFoldseek variants, structure aligners,
    # RiboSeek alphabets, sequence methods.
    order = ["RFoldseek", "RFoldseek-TM", "US-align", "GTalign", "RTM-align", "RMalign",
             "RiboSeek", "RiboSeek RS-80", "BLASTN", "Infernal"]
    by = dict(zip(labels, handles))
    labels = [LABEL.get(m, m) for m in order]
    handles = [by[l] for l in labels]
    legend = fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=11.5,
                        bbox_to_anchor=(0.5, -0.035), handlelength=2.6, columnspacing=1.8,
                        markerscale=1.45)
    for handle in legend.legend_handles:
        handle.set_linewidth(3.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_png = Path(FIG_DIR) / "fig2.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_png}")



if __name__ == "__main__":
    main()
