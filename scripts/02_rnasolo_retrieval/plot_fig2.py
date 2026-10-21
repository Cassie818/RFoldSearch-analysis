"""Fig. 2: retrieval on the RNAsolo benchmark.

    A/B  Rfam AUROC1 sorted-sensitivity curves (all methods)
    C/D  Structure-defined truth at TM-score 0.45, family (C) and clan (D):
         top, symmetric US-align TM-score; bottom, symmetric RMscore at the
         quantile-matched threshold
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
TM_THRESHOLD = "0.45"   # the single threshold shown in C/D; 0.40 and 0.50 are supplementary


def sorted_curve(arr):
    a = np.sort(np.asarray(arr))[::-1]
    return np.arange(1, len(a) + 1) / len(a), a


# ---- row 1: AUROC1 curves ----------------------------------------------------
# Both RFoldSearch rankings are drawn on top in their full-strength tone (RFoldSearch
# uppermost); the baselines recede (lighter tone, thinner line, sparser and smaller markers).
HERO = "RFoldSearch"
HIGHLIGHT = {HERO: 5, "RFoldSearch-TM": 4}   # method -> zorder
CURVE_MUTED = {m: _lighten(METHOD_BASE[m], 0.42) for m in METHODS}


def curve_panel(ax, data, level, title, band):
    for m in METHODS:
        x, y = sorted_curve(data[m][level]["sens"])
        hero = m in HIGHLIGHT
        step = max(1, len(x) // (18 if hero else 10))
        marker_size = (11 if m == "RFoldSearch-TM" else 8.5) if hero else 5.5
        ax.plot(x, y, LINESTYLE[m], color=METHOD_BASE[m] if hero else CURVE_MUTED[m],
                linewidth=3.4 if hero else 2.1, marker=MARKER[m], markevery=step,
                markersize=marker_size, markeredgecolor="white",
                markeredgewidth=1.0 if hero else 0.7, label=LABEL.get(m, m),
                zorder=HIGHLIGHT.get(m, 3))
    ax.set_xlabel("Fraction of queries", fontsize=12)
    ax.set_ylabel("Sensitivity up to the 1st FP", fontsize=12)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title(title, fontsize=13, pad=8)
    ax.tick_params(axis="both", labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)


# ---- row 2: structural Sens@1FP bars -----------------------------------------
def bar_panel(ax, boot, key, level, methods, oracle, tag):
    """Sens@1FP bars with 95% intervals at one truth threshold `key`; `oracle` (the method
    defining the truth), if any, is drawn as its ceiling line. `tag` names the truth
    inside the panel, since the TM and RMscore panels are stacked on shared axes."""
    ref = boot[key][level]
    width = 0.72
    for i, m in enumerate(methods):
        v = ref[m]
        yerr = [[v["estimate"] - v["ci_lower"]], [v["ci_upper"] - v["estimate"]]]
        ax.bar(i, v["estimate"], width, color=METHOD_COLOR[m], edgecolor="#666666",
               linewidth=0.6, label=LABEL.get(m, m), zorder=3, yerr=yerr, capsize=3,
               error_kw={"elinewidth": 0.9, "ecolor": "#555555"})
        top = v["ci_upper"] + 0.012
        # Only annotate RFoldSearch numerically (others read from the y-axis);
        # significance stars sit on non-RFoldSearch bars.
        if m == "RFoldSearch":
            ax.text(i, top, f"{v['estimate']:.2f}", ha="center", va="bottom", fontsize=9,
                    color=METHOD_BASE["RFoldSearch"])
        elif v.get("sig_vs_rf"):
            ax.text(i, top, "*", ha="center", va="bottom", fontsize=10, color="#555555",
                    fontweight="bold")
    lo, hi = -0.6, len(methods) - 0.4
    if oracle:
        ceil = (ref[oracle] if oracle in ref else ref[oracle + "*"])["estimate"]   # US-align* in the TM truth
        ax.plot([lo, hi], [ceil, ceil], ls="--", color=METHOD_COLOR[oracle], lw=1.5, zorder=2)
        ax.text(hi, ceil + 0.02, f"{oracle} (reference)", ha="right", va="bottom",
                fontsize=8.5, color=METHOD_BASE[oracle])
    # Same size and baseline as the oracle label, in grey so it reads as a caption.
    ax.text(lo + 0.08, 1.02, tag, ha="left", va="bottom", fontsize=8.5, color="#666666")
    ax.set_xticks([])
    ax.set_xlim(lo, hi); ax.set_ylim(0, 1.12)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0.1, 1.0, 0.2), minor=True)
    ax.grid(axis="y", which="major", color="#DDDDDD", linewidth=0.8, zorder=0)
    ax.grid(axis="y", which="minor", color="#EEEEEE", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", which="minor", length=0)
    ax.tick_params(axis="both", labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)


# Column-major, one pair per column: RFoldSearch variants, structure aligners,
# RiboSeek alphabets, sequence methods.
LEGEND_ORDER = ["RFoldSearch", "RFoldSearch-TM", "US-align", "GTalign", "RTM-align", "RMalign",
                "RiboSeek", "RiboSeek RS-80", "BLASTN", "Infernal"]
RM_METHODS = [m for m in BAR_METHODS if m != "RMalign"]
RM_METHODS.insert(RM_METHODS.index("RTM-align") + 1, "US-align")


def load_truths():
    """Structure-truth bootstraps: TM-score (keyed by threshold) and the RMscore truth
    (same keys, with the quantile-matched RMscore threshold for each)."""
    boot = json.load(open(resolve("files/production/retrieval/bootstrap_fig2cd_results.json")))
    alt = json.load(open(resolve("files/production/retrieval/rmscore_truth.json")))
    return boot, alt["thresholds"], alt["rmscore_thresholds"]


# Under a structure truth the levels are named by what they measure rather than by Rfam:
# "family" counts every structurally similar target, "clan" only those from other Rfam families.
TRUTH_LEVEL_NAME = {"family": "All", "clan": "Beyond-family"}


def truth_pair(fig, cell, boot, boot_rm, rm_thresholds, key, level):
    """TM-score (top) and RMscore (bottom) truths stacked in `cell`, sharing x and y: same
    method slots, with the truth-defining aligner (RMalign / US-align) swapped in slot 5.
    Returns the top axes."""
    sub = cell.subgridspec(2, 1, hspace=0.06)
    ax_tm = fig.add_subplot(sub[0])
    ax_rm = fig.add_subplot(sub[1], sharex=ax_tm, sharey=ax_tm)
    bar_panel(ax_tm, boot, key, level, BAR_METHODS, "US-align",
              f"TM-score ≥ {float(key):.2f}")
    bar_panel(ax_rm, boot_rm, key, level, RM_METHODS, "RMalign",
              f"RMscore ≥ {rm_thresholds[key]:.2f}")
    ax_tm.set_title(f"{TRUTH_LEVEL_NAME[level]} (Sens@1FP)", fontsize=13, pad=8)
    ax_tm.set_ylabel("Sens@1FP", fontsize=12)
    ax_tm.yaxis.set_label_coords(-0.07, -0.03)   # centred on the stacked pair
    return ax_tm


def main():
    data = json.load(open(resolve("files/production/retrieval/per_query_sens_test.json")))
    band = json.load(open(resolve("files/production/retrieval/bootstrap_fig2ab_band.json")))
    boot, boot_rm, rm_thresholds = load_truths()

    fig = plt.figure(figsize=(14.0, 11.4))
    gs = fig.add_gridspec(2, 2, height_ratios=(1.25, 2), hspace=0.2, wspace=0.16,
                          left=0.06, right=0.99, top=0.96, bottom=0.08)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    curve_panel(ax_a, data, "family", "Family (AUROC1)", band)
    curve_panel(ax_b, data, "clan", "Clan (AUROC1)", band)
    letter_axes = [ax_a, ax_b]
    for col, level in enumerate(("family", "clan")):
        letter_axes.append(truth_pair(fig, gs[1, col], boot, boot_rm, rm_thresholds,
                                      TM_THRESHOLD, level))

    for ax, lab in zip(letter_axes, "ABCD"):
        ax.text(-0.10, 1.04, lab, transform=ax.transAxes, fontsize=16, fontweight="bold",
                va="bottom", ha="right")

    handles, labels = ax_a.get_legend_handles_labels()
    order = LEGEND_ORDER
    by = dict(zip(labels, handles))
    labels = [LABEL.get(m, m) for m in order]
    handles = [by[l] for l in labels]
    legend = fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=11.5,
                        bbox_to_anchor=(0.5, -0.025), handlelength=2.6, columnspacing=1.8,
                        markerscale=1.45)
    for handle, m in zip(legend.legend_handles, order):
        handle.set_linewidth(3.6 if m in HIGHLIGHT else 2.6)
    out_png = Path(FIG_DIR) / "fig2.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_png}")



if __name__ == "__main__":
    main()
