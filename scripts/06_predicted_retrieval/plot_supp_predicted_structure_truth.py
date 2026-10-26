"""Supplementary figure: retrieval under the structure-defined ground truths, by threshold.

    A-D  TM-score truth (symmetric US-align TM-score >= 0.40, 0.45, 0.50): Sens@1FP (A, B) and
         Recall@1 (C, D) for experimental (left) and predicted (right) queries. US-align
         defines the positives (reference, 1.00); exhaustive RMalign is a competing method.
    E-H  RMscore truth at the quantile-matched thresholds, same layout; RMalign is the
         reference and US-align a competing method.
"""
import json
import os

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import evaluate as E
import figstyle as S
from figstyle import plt

FIG = os.path.join(E.REPO, "figures")
# Structure-based methods first and sequence-based after, as in Fig. 5.
STRUCT = ["RFoldSearch n500", "RFoldSearch-TM n500", "RiboSeek c500", "RiboSeek RS-80 c500"]
SEQ = ["BLASTN", "nhmmer", "Infernal"]
BAR_METHODS = STRUCT + SEQ
REF = "RFoldSearch n500"
SETS = [("exp", "Experimental queries"),
        ("p2p", "Predicted queries")]
COLS = ["exp", "p2p"]


def main():
    pq = json.load(open(os.path.join(E.HERE, "per_query.json")))
    report = json.load(open(os.path.join(E.HERE, "evaluation.json")))
    rthr = report.get("rmscore_thresholds", {})
    # Each truth draws the other exhaustive aligner in the first slot.
    truths = [("structure", [(f"structure t{t:.2f}", f"TM ≥ {t:.2f}") for t in E.TM_TAUS],
               ["RMalign (exhaustive)"] + BAR_METHODS),
              ("rmscore", [(f"rmscore t{t:.2f}", f"RM ≥ {rthr[f'{t:.2f}']:.2f}") for t in E.TM_TAUS],
               ["US-align (exhaustive)"] + BAR_METHODS)]
    fig, axes = plt.subplots(4, 2, figsize=(12.4, 14.4))
    for t, (truth_name, groups, methods) in enumerate(truths):
        for k, metric in enumerate(("Sens@1FP", "Recall@1")):
            plot_row(axes[2 * t + k], pq, truth_name, groups, methods, metric, title=(t == 0 and k == 0))
    for ax, lab in zip(axes.ravel(), "ABCDEFGH"):
        ax.text(-0.11, 1.04, lab, transform=ax.transAxes, fontsize=16, fontweight="bold", va="bottom")
    methods = ["US-align (exhaustive)", "RMalign (exhaustive)"] + BAR_METHODS
    S.shared_legend(fig, methods, ncol=5, y=-0.012, kind="bar")
    fig.tight_layout(rect=(0.02, 0.05, 1, 1), pad=0.5, w_pad=1.2, h_pad=1.6)
    for t, name in enumerate(("TM-score truth", "RMscore truth")):
        # one label per truth, centred on its two rows (after the layout is fixed)
        top, bottom = axes[2 * t, 0].get_position(), axes[2 * t + 1, 0].get_position()
        fig.text(top.x0 - 0.075, (top.y1 + bottom.y0) / 2, name, rotation=90, ha="center", va="center",
                 fontsize=12.5)
    out = os.path.join(FIG, "supp_predicted_structure_truth.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)


def plot_row(axes, pq, truth_name, groups, methods, metric, title):
    fams = pq["families"]
    oracle = E.ORACLE[truth_name]

    def cells(qs, truth, method):
        d = pq.get(qs, {}).get(f"{truth}|none", {}).get(method, {})
        return {q: v[metric] for q, v in d.items()}

    counts = {}
    width = 0.84 / len(methods)
    x = np.arange(len(groups))
    for ax, qs in zip(axes, COLS):
        ns = [len(cells(qs, t, REF)) for t, _ in groups]
        for i, m in enumerate(methods):
            off = (i - (len(methods) - 1) / 2) * width
            vals, lo, hi, sig = [], [], [], []
            for truth, _ in groups:
                v, rr = cells(qs, truth, m), cells(qs, truth, REF)
                if not v:
                    vals.append(np.nan); lo.append(0.0); hi.append(0.0); sig.append(False)
                    continue
                mean, l, h = E.block_boot(v, fams)
                vals.append(mean); lo.append(max(mean - l, 0)); hi.append(max(h - mean, 0))
                ids = [q for q in v if q in rr]
                if m == REF or len(ids) < 3:
                    sig.append(False)
                else:
                    _, dl, dh = E.block_boot({q: v[q] - rr[q] for q in ids}, fams)
                    sig.append(not (dl <= 0 <= dh))
            bars = S.bars(ax, x + off, vals, width, m, yerr=[lo, hi])
            for j, (bar, v) in enumerate(zip(bars, vals)):
                if np.isfinite(v) and sig[j]:
                    ax.text(bar.get_x() + bar.get_width() / 2, v + hi[j] + 0.012, "*", ha="center",
                            va="bottom", fontsize=10, color=S.GREY_TEXT, fontweight="bold")
        ax.axhline(1.0, color=S.COLOR[oracle], ls="--", lw=1.5, zorder=2)
        ax.text(-0.45, 1.13, f"{S.SHORT[oracle]} (reference)", ha="left", va="bottom", fontsize=8.5,
                color=S.TEXT[oracle])
        ax.set_xticks(x, [f"{lab}\n(n = {n})" for (_, lab), n in zip(groups, ns)], fontsize=9.5)
        counts[qs] = {lab.split("≥ ")[-1]: n for (_, lab), n in zip(groups, ns)}
        ax.set_xlim(-0.5, len(groups) - 0.5)
        ax.set_ylim(0, 1.22)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.grid(axis="y", color="#E6E6E6", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        S.style_axes(ax, ylabel=metric)
        if title:
            ax.set_title(dict(SETS)[qs], fontsize=12, pad=8)
    json.dump(counts, open(os.path.join(E.HERE, f"{truth_name}_truth_counts.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
