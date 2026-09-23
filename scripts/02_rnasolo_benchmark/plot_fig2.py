"""Fig. 2: retrieval and residue-level alignment benchmarks, 3x2.

    A/B  Rfam AUROC1 sorted-sensitivity curves (all methods; RFoldseek 95% band)
    C/D  Structure-defined (US-align TM-score) Sens@1FP bars, by TM threshold
    E    Exact and +/-1-column alignment F1 against Rfam seed alignments
    F    Paired family-level differences from RFoldseek, with bootstrap CIs

Writes figures/fig2.png.
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
METHODS = ["RFoldseek", "RFoldseek-TM", "GTalign", "RTM-align", "US-align", "RMalign", "RiboSeek", "BLASTN", "Infernal"]
METHOD_COLOR = {
    "RFoldseek": "#1F9BD8", "GTalign": "#9467BD", "RTM-align": "#FF7F0E",
    "RFoldseek-TM": "#009E9E", "US-align": "#D62728", "RMalign": "#2CA02C",
    "RiboSeek": "#E377C2", "BLASTN": "#FFC107", "Infernal": "#BCBD22",
}


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
          "RMalign": "^", "RiboSeek": "v", "BLASTN": "D", "Infernal": "p"}
# C/D bars now include GTalign and RTM-align; both rank by a TM-score, the same
# quantity that defines the structural ground truth, so they are near-oracles
# here (they hug the US-align ceiling) and are shown for completeness.
BAR_METHODS = ["RFoldseek", "RFoldseek-TM", "GTalign", "RTM-align", "RMalign", "RiboSeek", "BLASTN", "Infernal"]

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
                label=m, zorder=3)
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
        vals = [tm["thresholds"][k][m][metric] for k in keys]
        yerr = None
        if all(m in boot.get(k, {}).get(level, {}) for k in keys):
            yerr = [[vals[j] - boot[keys[j]][level][m]["ci_lower"] for j in range(len(keys))],
                    [boot[keys[j]][level][m]["ci_upper"] - vals[j] for j in range(len(keys))]]
        bars = ax.bar(x + off, vals, width, color=METHOD_COLOR[m], edgecolor="#666666",
                      linewidth=0.6, label=m, zorder=3, yerr=yerr, capsize=2.5,
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


# ---- row 3: reference residue-level alignment benchmark ---------------------
ALIGN_METHODS = ["RFoldseek", "RiboSeek", "US-align"]
ALIGN_KEY = {"RFoldseek": "rfoldseek", "RiboSeek": "riboseek", "US-align": "usalign"}


def _estimate(summary, key):
    item = summary[key]
    return item["family_macro_mean"], item["family_block_bootstrap_95ci"]


def alignment_panel(ax, alignment):
    """Family-macro alignment F1 with family-block-bootstrap 95% CIs."""
    summary = alignment["summary"]
    x = np.arange(len(ALIGN_METHODS))
    offsets = {"exact": -0.09, "tolerant": 0.09}

    for metric, label, filled in [("f1", "Exact", True),
                                  ("tolerant_f1", r"$\pm$1 nt", False)]:
        for i, method in enumerate(ALIGN_METHODS):
            mean, (lo, hi) = _estimate(summary, f"{ALIGN_KEY[method]}_{metric}")
            ax.errorbar(x[i] + offsets["exact" if filled else "tolerant"], mean,
                        yerr=[[mean - lo], [hi - mean]], fmt="o", ms=7.2,
                        mfc=METHOD_COLOR[method] if filled else "white",
                        mec=METHOD_BASE[method], mew=1.5, ecolor=METHOD_BASE[method],
                        elinewidth=1.35, capsize=3.2, zorder=4,
                        label=label if i == 0 else None)

    ax.set_xticks(x, ALIGN_METHODS)
    ax.set_ylabel("Alignment F1", fontsize=12)
    ax.set_ylim(0.60, 1.01)
    ax.set_yticks(np.arange(0.6, 1.01, 0.1))
    ax.set_xlim(-0.55, len(x) - 0.45)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right", handletextpad=0.3)
    ax.tick_params(axis="both", labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)


def alignment_difference_panel(ax, alignment):
    """Paired family-macro RFoldseek differences with block-bootstrap CIs."""
    summary = alignment["summary"]
    rows = [
        ("rfoldseek_minus_usalign_f1", "US-align", "Exact", True),
        ("rfoldseek_minus_riboseek_f1", "RiboSeek", "Exact", True),
        ("rfoldseek_minus_usalign_tolerant_f1", "US-align", r"$\pm$1 nt", False),
        ("rfoldseek_minus_riboseek_tolerant_f1", "RiboSeek", r"$\pm$1 nt", False),
    ]
    y = np.arange(len(rows))[::-1]
    for yy, (key, competitor, metric, filled) in zip(y, rows):
        mean, (lo, hi) = _estimate(summary, key)
        ax.errorbar(mean, yy, xerr=[[mean - lo], [hi - mean]], fmt="o", ms=7.2,
                    mfc=METHOD_COLOR[competitor] if filled else "white",
                    mec=METHOD_BASE[competitor], mew=1.5,
                    ecolor=METHOD_BASE[competitor], elinewidth=1.45, capsize=3.2,
                    zorder=4)
        star = "*" if lo > 0 or hi < 0 else ""
        ax.text(0.194, yy, f"{mean:+.3f}{star}", ha="right", va="center",
                fontsize=8.5, color=METHOD_BASE[competitor])

    ax.axvline(0, color="#777777", lw=1.1, ls="--", zorder=1)
    ax.set_yticks(y, [f"vs {competitor} ({metric})"
                      for _, competitor, metric, _ in rows])
    ax.set_xlabel(r"$\Delta$F1 (RFoldseek $-$ comparator)", fontsize=12)
    ax.set_xlim(-0.10, 0.20)
    ax.set_ylim(-0.65, len(rows) - 0.35)
    ax.tick_params(axis="both", labelsize=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def main():
    data = json.load(open(resolve("files/production/retrieval/per_query_sens_test.json")))
    tm = json.load(open(resolve("files/production/retrieval/structural_tmscore_metrics_test.json")))
    boot = json.load(open(resolve("files/production/retrieval/bootstrap_fig2cd_results.json")))
    band = json.load(open(resolve("files/production/retrieval/bootstrap_fig2ab_band.json")))
    alignment = json.load(open(resolve(
        "files/ablation/rfam3d_reference_disjoint_independent_3dseed/"
        "rfam3d_reference_alignment.json")))

    fig, axes = plt.subplots(3, 2, figsize=(14.0, 12.0))
    curve_panel(axes[0, 0], data, "family", "Family (AUROC1)", band)
    curve_panel(axes[0, 1], data, "clan", "Clan (AUROC1)", band)
    bar_panel(axes[1, 0], tm, "Sens@1FP", "Family", "Sens@1FP", boot, "family")
    bar_panel(axes[1, 1], tm, "Sens@1FP_clan", "Clan", "Sens@1FP", boot, "clan")
    alignment_panel(axes[2, 0], alignment)
    alignment_difference_panel(axes[2, 1], alignment)

    for ax, lab in zip(axes.flat, "ABCDEF"):
        ax.text(-0.10, 1.04, lab, transform=ax.transAxes, fontsize=16, fontweight="bold",
                va="bottom", ha="right")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=11.5,
                        bbox_to_anchor=(0.5, -0.038), handlelength=2.6, columnspacing=1.8,
                        markerscale=1.45)
    for handle in legend.legend_handles:
        handle.set_linewidth(3.0)
    fig.tight_layout(rect=(0, 0.026, 1, 1))
    out_png = Path(FIG_DIR) / "fig2.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
