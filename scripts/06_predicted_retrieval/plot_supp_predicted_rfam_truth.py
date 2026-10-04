"""Supplementary figure: retrieval under the annotation (Rfam family) truth."""
import json
import os

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import evaluate as E  # noqa: E402
import figstyle as S  # noqa: E402
from figstyle import plt  # noqa: E402

FIG = os.path.join(E.REPO, "figures")
BAR_METHODS = ["RFoldseek n500", "RFoldseek-TM n500", "RiboSeek c500", "RiboSeek RS-80 c500", "BLASTN", "nhmmer", "Infernal",
               "US-align (exhaustive)"]
REF = "RFoldseek n500"
FILTERS = [("none", "None"), ("E<0.01", "$E<0.01$"), ("E<=10", "$E\\leq10$")]
COLS = [("exp", "Experimental queries"), ("p2p", "Predicted queries")]
ROWS = ["Sens@1FP", "Recall@1"]


def main():
    pq = json.load(open(os.path.join(E.HERE, "per_query.json")))
    fams = pq["families"]

    def cells(qs, filt, method, metric):
        d = pq.get(qs, {}).get(f"label|{filt}", {}).get(method, {})
        return {q: v[metric] for q, v in d.items()}

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2))
    counts = {}
    width = 0.82 / len(BAR_METHODS)
    x = np.arange(len(FILTERS))
    for r_i, metric in enumerate(ROWS):
        for c_i, (qs, _) in enumerate(COLS):
            ax = axes[r_i, c_i]
            ns = [len(cells(qs, f, REF, metric)) for f, _ in FILTERS]
            for i, m in enumerate(BAR_METHODS):
                off = (i - (len(BAR_METHODS) - 1) / 2) * width
                vals, lo, hi, sig = [], [], [], []
                for filt, _ in FILTERS:
                    v = cells(qs, filt, m, metric)
                    rr = cells(qs, filt, REF, metric)
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
                for j, (b, v) in enumerate(zip(bars, vals)):
                    if not np.isfinite(v):
                        continue
                    top = v + hi[j] + 0.012
                    if m == REF:
                        ax.text(b.get_x() + b.get_width() / 2, top, f"{v:.2f}", ha="center",
                                va="bottom", fontsize=8, color=S.TEXT[REF])
                    elif sig[j]:
                        ax.text(b.get_x() + b.get_width() / 2, top, "*", ha="center", va="bottom",
                                fontsize=10, color=S.GREY_TEXT, fontweight="bold")
            ax.set_xticks(x, [lab for _, lab in FILTERS])
            counts.setdefault(qs, {}).update({lab: n for (_, lab), n in zip(FILTERS, ns)})
            ax.set_xlim(-0.5, len(FILTERS) - 0.5)
            ax.set_ylim(0, 1.12)
            ax.set_yticks(np.arange(0, 1.01, 0.2))
            S.style_axes(ax, ylabel=metric)
            if r_i == 0:                       # column header names the query set
                ax.set_title(COLS[c_i][1], fontsize=13, pad=10)
    S.panel_letters(axes.ravel(), "ABCD", x=-0.12)
    S.shared_legend(fig, BAR_METHODS, ncol=4, y=-0.01)
    fig.tight_layout(rect=(0, 0.08, 1, 1), pad=0.5, w_pad=0.6, h_pad=0.7)
    for ext in ("png",):
        fig.savefig(os.path.join(FIG, f"supp_predicted_rfam_truth.{ext}"), dpi=200,
                    bbox_inches="tight")
    json.dump(counts, open(os.path.join(E.HERE, "annotation_truth_counts.json"), "w"), indent=1)
    print("wrote", os.path.join(FIG, "supp_predicted_rfam_truth.png"))


if __name__ == "__main__":
    main()
