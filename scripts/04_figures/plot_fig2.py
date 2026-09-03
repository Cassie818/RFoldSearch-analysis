"""Integrated Fig. 2: retrieval comparison + alphabet-control ablation, 3x2 grid.

Rows are three progressively stricter views of the same test library, columns are
the two ground-truth levels (left Family, right Clan). Colour is shared across
all six panels: RFoldseek is sky blue and RiboSeek is pink throughout, so the
alphabet control (row 3) reads against the same legend as the main benchmark.

    A/B  Rfam AUROC1 sorted-sensitivity curves (all methods; RFoldseek 95% band)
    C/D  Structure-defined (US-align TM-score) Sens@1FP bars, by TM threshold
    E/F  Attribution waterfall: the gap from RiboSeek as published to RFoldseek,
         split into the search pipeline, the learned substitution matrix and the
         3Dr alphabet, with block-bootstrap intervals on each step.

Writes figures/fig2.{png,pdf}.
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
METHODS = ["RFoldseek", "RFoldseek-TM", "GTalign", "RTM-align", "US-align", "RMalign", "RiboSeek", "BLASTN"]
METHOD_COLOR = {
    "RFoldseek": "#1F9BD8", "GTalign": "#9467BD", "RTM-align": "#FF7F0E",
    "RFoldseek-TM": "#009E9E", "US-align": "#D62728", "RMalign": "#2CA02C",
    "RiboSeek": "#E377C2", "BLASTN": "#FFC107",
}


def _lighten(h, f=0.24):
    """Blend a hex colour toward white by fraction f (softer, lighter tone)."""
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02x%02x%02x" % tuple(round(c + (255 - c) * f) for c in (r, g, b))


METHOD_BASE = dict(METHOD_COLOR)          # text uses the unlightened tone
METHOD_COLOR = {k: _lighten(v) for k, v in METHOD_COLOR.items()}
LINESTYLE = {m: "-" for m in METHODS}
LINESTYLE["BLASTN"] = "--"
MARKER = {"RFoldseek": "o", "RFoldseek-TM": "*", "GTalign": "X", "RTM-align": "P", "US-align": "s",
          "RMalign": "^", "RiboSeek": "v", "BLASTN": "D"}
# C/D bars now include GTalign and RTM-align; both rank by a TM-score, the same
# quantity that defines the structural ground truth, so they are near-oracles
# here (they hug the US-align ceiling) and are shown for completeness.
BAR_METHODS = ["RFoldseek", "RFoldseek-TM", "GTalign", "RTM-align", "RMalign", "RiboSeek", "BLASTN"]

# ---- alphabet-control constants ----------------------------------------------
RF = "RFoldseek_tokens_same_pipeline"
RB = "RiboSeek_sa20_same_pipeline"
REGIMES = [("", "Native"), ("_identity", "Identity"), ("_self_blosum", "Self-BLOSUM")]
REGIME_KEY = {"": "native", "_identity": "identity", "_self_blosum": "self_blosum"}
CTRL_WHO = [("RFoldseek", RF), ("RiboSeek", RB)]
CTRL_BOOT = json.load(open(resolve("files/production/retrieval/bootstrap_controls.json")))


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


# ---- row 3: alphabet-control bars --------------------------------------------
def load_ctrl(suffix):
    return json.load(open(resolve(
        f"files/ablation/riboseek_alphabet_same_pipeline_control{suffix}.json")))


def load_ctrl_pq(suffix):
    return json.load(open(resolve(
        f"files/ablation/riboseek_alphabet_same_pipeline_control{suffix}_per_query.json")))


ATTR = json.load(open(resolve("files/ablation/attribution_paths.json")))
STEPS = [("B_fwd_pipeline", "+ RFoldseek\npipeline", "+ RFoldseek pipeline"),
         ("C_fwd_matrix", "+ Structure-BLOSUM\nmatrix", "+ learned matrix"),
         ("D_rfoldseek", "+ 3Dr\nalphabet", "+ 3Dr alphabet")]
# Anchors reuse the method colours from rows 1-2; each contribution gets its own
# hue, checked pairwise for CVD separation against the anchors and each other.
STEP_COLOR = {"+ RFoldseek pipeline": "#C9A227",
              "+ learned matrix": "#556B7D",
              "+ 3Dr alphabet": "#B5651D"}
NULL_COLOR = "#BDBDBD"


def control_panel(ax, level, title):
    """Waterfall: how the RiboSeek-to-RFoldseek gap divides between the search
    pipeline, the learned substitution matrix and the 3Dr alphabet."""
    lv = ATTR["levels"][level]
    m, fwd = lv["means"], lv["forward"]
    keys = ["A_ribo_published"] + [k for k, _, _ in STEPS]
    vals = [m[k] for k in keys]
    up = METHOD_COLOR["RFoldseek"]

    ax.bar(0, vals[0], 0.62, color=METHOD_COLOR["RiboSeek"], edgecolor="#b05a94",
           linewidth=0.6, zorder=3)
    ax.text(0, vals[0] + 0.008, f"{vals[0]:.3f}", ha="center", va="bottom",
            fontsize=8.5, color=METHOD_BASE["RiboSeek"])
    for n, (_, _, lbl) in enumerate(STEPS, start=1):
        lo, hi = min(vals[n - 1], vals[n]), max(vals[n - 1], vals[n])
        e0 = fwd[[l for _, _, l in STEPS][n - 1]]
        lbl0 = [l for _, _, l in STEPS][n - 1]
        col = STEP_COLOR[lbl0] if e0["diff"] > 0 else NULL_COLOR
        edge = "#555555"
        span = max(hi - lo, max(vals) * 0.006)          # keep a null step visible
        ax.bar(n, span, 0.62, bottom=lo, color=col, edgecolor=edge,
               linewidth=0.6, zorder=3)
        e = fwd[lbl]
        ax.errorbar(n, hi, yerr=[[max(e["diff"] - e["lo"], 0)], [max(e["hi"] - e["diff"], 0)]],
                    fmt="none", ecolor="#555555", elinewidth=0.9, capsize=2.5, zorder=4)
        star = "*" if e["sig"] else ""
        ax.text(n, hi + max(e["hi"] - e["diff"], 0) + max(vals) * 0.022,
                f"{e['diff']:+.3f}{star}", ha="center", va="bottom", fontsize=8,
                color=STEP_COLOR[[l for _, _, l in STEPS][n - 1]] if e["diff"] > 0 else "#777777")
        ax.plot([n - 1 + 0.31, n - 0.31], [vals[n - 1], vals[n - 1]],
                color="#999999", lw=0.8, ls=(0, (3, 2)), zorder=2)
    ax.bar(4, vals[-1], 0.62, color=METHOD_COLOR["RFoldseek"], edgecolor="#15709c",
           linewidth=0.6, zorder=3)
    ax.text(4, vals[-1] + 0.008, f"{vals[-1]:.3f}", ha="center", va="bottom",
            fontsize=8.5, color=METHOD_BASE["RFoldseek"])
    ax.plot([3 + 0.31, 4 - 0.31], [vals[-1], vals[-1]], color="#999999", lw=0.8,
            ls=(0, (3, 2)), zorder=2)

    ax.set_xticks(range(5))
    ax.set_xticklabels(["RiboSeek"] + [t for _, t, _ in STEPS] + ["RFoldseek"],
                       fontsize=8)
    ax.set_ylabel("Sens@1FP", fontsize=12)
    top = max(max(vals),
              max(m[k] + max(fwd[l]["hi"] - fwd[l]["diff"], 0) for k, _, l in STEPS))
    ax.set_ylim(0, top * 1.30)
    ax.set_xlim(-0.55, 4.55)
    ax.set_title(title, fontsize=13, pad=8)
    ax.tick_params(axis="both", labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    data = json.load(open(resolve("files/production/retrieval/per_query_sens_test.json")))
    tm = json.load(open(resolve("files/production/retrieval/structural_tmscore_metrics_test.json")))
    boot = json.load(open(resolve("files/production/retrieval/bootstrap_fig2cd_results.json")))
    band = json.load(open(resolve("files/production/retrieval/bootstrap_fig2ab_band.json")))

    fig, axes = plt.subplots(3, 2, figsize=(14.0, 12.0))
    curve_panel(axes[0, 0], data, "family", "Family (AUROC1)", band)
    curve_panel(axes[0, 1], data, "clan", "Clan (AUROC1)", band)
    bar_panel(axes[1, 0], tm, "Sens@1FP", "Family", "Sens@1FP", boot, "family")
    bar_panel(axes[1, 1], tm, "Sens@1FP_clan", "Clan", "Sens@1FP", boot, "clan")
    control_panel(axes[2, 0], "family", "Family")
    control_panel(axes[2, 1], "clan", "Clan")

    for ax, lab in zip(axes.flat, "ABCDEF"):
        ax.text(-0.10, 1.04, lab, transform=ax.transAxes, fontsize=16, fontweight="bold",
                va="bottom", ha="right")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=12,
                        bbox_to_anchor=(0.5, -0.006), handlelength=2.8, columnspacing=2.4,
                        markerscale=1.45)
    for handle in legend.legend_handles:
        handle.set_linewidth(3.0)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    out_png = Path(FIG_DIR) / "fig2.png"
    out_pdf = Path(FIG_DIR) / "fig2.pdf"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
