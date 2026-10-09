"""Fig. 2: retrieval on the RNAsolo benchmark, 3x2.

    A/B  Rfam AUROC1 sorted-sensitivity curves (all methods)
    C/D  Structure-defined truth, symmetric US-align TM-score: Sens@1FP bars by threshold
    E/F  Structure-defined truth, symmetric RMscore (thresholds quantile-matched to C/D)
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
METHODS = ["RFoldSearch", "RFoldSearch-TM", "GTalign", "RTM-align", "US-align", "RMalign",
           "RiboSeek", "RiboSeek RS-80", "BLASTN", "Infernal"]
# RiboSeek 0.2.0: RS-20 (geometry only; the "RiboSeek" key) and the default
# RS-80 (RS-20 x nucleotide identity) share one hue, RS-80 in the deeper shade.
METHOD_COLOR = {
    "RFoldSearch": "#1F9BD8", "GTalign": "#9467BD", "RTM-align": "#FF7F0E",
    "RFoldSearch-TM": "#009E9E", "US-align": "#D62728", "RMalign": "#2CA02C",
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
MARKER = {"RFoldSearch": "o", "RFoldSearch-TM": "*", "GTalign": "X", "RTM-align": "P", "US-align": "s",
          "RMalign": "^", "RiboSeek": "v", "RiboSeek RS-80": "v", "BLASTN": "D", "Infernal": "p"}
# C/D: US-align defines the structural truth (ceiling 1). GTalign and RTM-align
# rank by their own TM-scores and are near-oracles, shown for completeness.
# RFoldSearch-TM ranks the RFoldSearch prefilter candidates by the truth TM-score, so
# its Sens@1FP equals the prefilter recall (rankings/run_rfoldsearch.py); it is
# drawn but not tested against RFoldSearch.
BAR_METHODS = ["RFoldSearch", "RFoldSearch-TM", "GTalign", "RTM-align", "RMalign",
               "RiboSeek", "RiboSeek RS-80", "BLASTN", "Infernal"]

def sorted_curve(arr):
    a = np.sort(np.asarray(arr))[::-1]
    return np.arange(1, len(a) + 1) / len(a), a


# ---- row 1: AUROC1 curves ----------------------------------------------------
def curve_panel(ax, data, level, title, band):
    for m in METHODS:
        x, y = sorted_curve(data[m][level]["sens"])
        step = max(1, len(x) // 18)
        marker_size = 11 if m == "RFoldSearch-TM" else 7
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
def bar_panel(ax, boot, level, methods, oracle, xticks, xlabel, title):
    """Sens@1FP bars with 95% intervals; `oracle` (the method defining the truth), if any, is
    drawn as its ceiling line."""
    keys = sorted(boot, key=float)
    x = np.arange(len(keys)); width = 0.82 / len(methods)
    for i, m in enumerate(methods):
        off = (i - (len(methods) - 1) / 2) * width
        e = [boot[k][level][m] for k in keys]
        vals = [v["estimate"] for v in e]
        yerr = [[v["estimate"] - v["ci_lower"] for v in e], [v["ci_upper"] - v["estimate"] for v in e]]
        bars = ax.bar(x + off, vals, width, color=METHOD_COLOR[m], edgecolor="#666666",
                      linewidth=0.6, label=LABEL.get(m, m), zorder=3, yerr=yerr, capsize=2.5,
                      error_kw={"elinewidth": 0.9, "ecolor": "#555555"})
        for j, (b, v) in enumerate(zip(bars, e)):
            top = v["ci_upper"] + 0.012
            # Only annotate RFoldSearch numerically (others read from the y-axis);
            # significance stars sit on non-RFoldSearch bars.
            if m == "RFoldSearch":
                ax.text(b.get_x() + b.get_width() / 2, top, f"{v['estimate']:.2f}",
                        ha="center", va="bottom", fontsize=8.5, color=METHOD_BASE["RFoldSearch"])
            elif v.get("sig_vs_rf"):
                ax.text(b.get_x() + b.get_width() / 2, top, "*", ha="center", va="bottom",
                        fontsize=10, color="#555555", fontweight="bold")
    if oracle:
        ref = boot[keys[0]][level]
        ceil = (ref[oracle] if oracle in ref else ref[oracle + "*"])["estimate"]   # US-align* in the TM truth
        ax.plot([-0.5, len(keys) - 0.5], [ceil, ceil], ls="--", color=METHOD_COLOR[oracle], lw=1.5, zorder=2)
        ax.text(-0.44, ceil + 0.012, f"{oracle} (oracle ceiling)", ha="left", va="bottom", fontsize=8.5,
                color=METHOD_BASE[oracle])
    ax.set_xticks(x, xticks)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Sens@1FP", fontsize=12)
    ax.set_xlim(-0.5, len(keys) - 0.5); ax.set_ylim(0, 1.08)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_title(title, fontsize=13, pad=8)
    ax.tick_params(axis="both", labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)



def main():
    data = json.load(open(resolve("files/production/retrieval/per_query_sens_test.json")))
    boot = json.load(open(resolve("files/production/retrieval/bootstrap_fig2cd_results.json")))
    band = json.load(open(resolve("files/production/retrieval/bootstrap_fig2ab_band.json")))
    alt = json.load(open(resolve("files/production/retrieval/rmscore_truth.json")))
    boot_rm = alt["thresholds"]
    keys = sorted(boot, key=float)
    tm_ticks = [f"{float(k):.2f}" for k in keys]
    rm_ticks = [f"{alt['rmscore_thresholds'][k]:.2f}" for k in keys]
    rm_methods = [m for m in BAR_METHODS if m != "RMalign"]
    rm_methods.insert(rm_methods.index("RTM-align") + 1, "US-align")

    fig, axes = plt.subplots(3, 2, figsize=(14.0, 12.6))
    curve_panel(axes[0, 0], data, "family", "Family (AUROC1)", band)
    curve_panel(axes[0, 1], data, "clan", "Clan (AUROC1)", band)
    for col, level in enumerate(("family", "clan")):
        bar_panel(axes[1, col], boot, level, BAR_METHODS, "US-align", tm_ticks,
                  "TM-score similarity threshold", f"{level.capitalize()} (TM-score)")
        bar_panel(axes[2, col], boot_rm, level, rm_methods, "RMalign", rm_ticks,
                  "RMscore similarity threshold", f"{level.capitalize()} (RMscore)")

    for ax, lab in zip(axes.flat, "ABCDEF"):
        ax.text(-0.10, 1.04, lab, transform=ax.transAxes, fontsize=16, fontweight="bold",
                va="bottom", ha="right")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Column-major, one pair per column: RFoldSearch variants, structure aligners,
    # RiboSeek alphabets, sequence methods.
    order = ["RFoldSearch", "RFoldSearch-TM", "US-align", "GTalign", "RTM-align", "RMalign",
             "RiboSeek", "RiboSeek RS-80", "BLASTN", "Infernal"]
    by = dict(zip(labels, handles))
    labels = [LABEL.get(m, m) for m in order]
    handles = [by[l] for l in labels]
    legend = fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=11.5,
                        bbox_to_anchor=(0.5, -0.025), handlelength=2.6, columnspacing=1.8,
                        markerscale=1.45)
    for handle in legend.legend_handles:
        handle.set_linewidth(3.0)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out_png = Path(FIG_DIR) / "fig2.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_png}")



if __name__ == "__main__":
    main()
