"""Supplementary figure: retrieval under the annotation (Rfam family) truth.

Sens@1FP (A, B) and Recall@1 (C, D) for experimental (A, C) and predicted (B, D) queries, with
all positives and after positives detected by BLASTN or nhmmer at E < 0.01 or E <= 10 are made
neutral. Methods in the order of Fig. 5: structure-based, then sequence-based; asterisks mark
a paired difference from RFoldSearch whose 95% interval excludes zero.
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
BAR_METHODS = ["US-align (exhaustive)", "RMalign (exhaustive)", "RFoldSearch n500", "RFoldSearch-TM n500",
               "RiboSeek c500", "RiboSeek RS-80 c500", "BLASTN", "nhmmer", "Infernal"]
REF = "RFoldSearch n500"
FILTERS = [("none", "None"), ("E<0.01", "$E<0.01$"), ("E<=10", "$E\\leq10$")]
COLS = [("exp", "Experimental queries"), ("p2p", "Predicted queries")]
ROWS = ["Sens@1FP", "Recall@1"]


def main():
    pq = json.load(open(os.path.join(E.HERE, "per_query.json")))
    fams = pq["families"]

    def cells(qs, filt, method, metric):
        d = pq.get(qs, {}).get(f"label|{filt}", {}).get(method, {})
        return {q: v[metric] for q, v in d.items()}

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.8))
    counts = {}
    width = 0.84 / len(BAR_METHODS)
    x = np.arange(len(FILTERS))
    for r_i, metric in enumerate(ROWS):
        for c_i, (qs, _) in enumerate(COLS):
            ax = axes[r_i, c_i]
            ns = [len(cells(qs, f, REF, metric)) for f, _ in FILTERS]
            for i, m in enumerate(BAR_METHODS):
                off = (i - (len(BAR_METHODS) - 1) / 2) * width
                vals, lo, hi, sig = [], [], [], []
                for filt, _ in FILTERS:
                    v, rr = cells(qs, filt, m, metric), cells(qs, filt, REF, metric)
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
            ax.set_xticks(x, [f"{lab}\n(n = {n})" for (_, lab), n in zip(FILTERS, ns)], fontsize=9.5)
            counts.setdefault(qs, {}).update({lab: n for (_, lab), n in zip(FILTERS, ns)})
            ax.set_xlim(-0.5, len(FILTERS) - 0.5)
            ax.set_ylim(0, 1.12)
            ax.set_yticks(np.arange(0, 1.01, 0.2))
            ax.grid(axis="y", color="#E6E6E6", lw=0.8, zorder=0)
            ax.set_axisbelow(True)
            S.style_axes(ax, ylabel=metric)
            if r_i == 0:
                ax.set_title(COLS[c_i][1], fontsize=12, pad=8)
    for ax, lab in zip(axes.ravel(), "ABCD"):
        ax.text(-0.11, 1.04, lab, transform=ax.transAxes, fontsize=16, fontweight="bold", va="bottom")
    S.shared_legend(fig, BAR_METHODS, ncol=5, y=-0.01, kind="bar")
    fig.tight_layout(rect=(0, 0.1, 1, 1), pad=0.5, w_pad=1.2, h_pad=1.6)
    fig.savefig(os.path.join(FIG, "supp_predicted_rfam_truth.png"), dpi=200, bbox_inches="tight")
    json.dump(counts, open(os.path.join(E.HERE, "annotation_truth_counts.json"), "w"), indent=1)
    print("wrote", os.path.join(FIG, "supp_predicted_rfam_truth.png"))


if __name__ == "__main__":
    main()
