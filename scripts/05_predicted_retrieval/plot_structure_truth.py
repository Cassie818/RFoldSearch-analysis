"""Supplementary figure: retrieval under the structure-defined ground truth.

The structural truth follows Fig. 2 -- a positive is a non-self target whose
symmetric US-align TM-score reaches tau -- swept over tau in {0.40, 0.45, 0.50}.
Bars follow Fig. 2C,D: grouped bars per method, block-bootstrap intervals,
US-align as the dashed oracle ceiling rather than a competing method, RFoldseek
labelled numerically and stars marking methods that differ from it.

Two rows, two columns: the top row is Sens@1FP, the bottom row Recall@1;
columns are the experimental and predicted query sets. The method ordering is
stable across every threshold and both metrics.

Reads per_query.json (written by `evaluate.py`).
Writes figures/structure_truth.png and files/predicted_retrieval/structure_truth_table.tex
"""
import json
import os

import numpy as np

import evaluate as E
import figstyle as S
from figstyle import plt

FIG = os.path.join(E.REPO, "figures")
BAR_METHODS = ["RFoldseek n500", "RFoldseek-TM n500", "RiboSeek c500", "BLASTN", "nhmmer", "Infernal"]
ORACLE = "US-align (exhaustive)"
REF = "RFoldseek n500"
GROUPS = [(f"structure t{t:.2f}", f"{t:.2f}") for t in E.TM_TAUS]
SETS = [("exp", "Experimental queries"),
        ("p2p", "Predicted queries")]
COLS = ["exp", "p2p"]
ROWS = ["Sens@1FP", "Recall@1"]


def main():
    pq = json.load(open(os.path.join(E.HERE, "per_query.json")))
    fams = pq["families"]
    groups = list(GROUPS)

    def cells(qs, truth, method, metric):
        d = pq.get(qs, {}).get(f"{truth}|none", {}).get(method, {})
        return {q: v[metric] for q, v in d.items()}

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2))
    counts = {}
    width = 0.82 / len(BAR_METHODS)
    x = np.arange(len(groups))
    for r_i, metric in enumerate(ROWS):
        for c_i, qs in enumerate(COLS):
            ax = axes[r_i, c_i]
            ns = [len(cells(qs, t, REF, metric)) for t, _ in groups]
            for i, m in enumerate(BAR_METHODS):
                off = (i - (len(BAR_METHODS) - 1) / 2) * width
                vals, lo, hi, sig = [], [], [], []
                for truth, _ in groups:
                    v = cells(qs, truth, m, metric)
                    rr = cells(qs, truth, REF, metric)
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
            oracle = cells(qs, groups[0][0], ORACLE, metric)
            if oracle:
                S.ceiling(ax, float(np.mean(list(oracle.values()))), -0.5, len(groups) - 0.5)
            ax.set_xticks(x, [lab for _, lab in groups])
            counts.setdefault(qs, {}).update({lab: n for (_, lab), n in zip(groups, ns)})
            ax.set_xlim(-0.5, len(groups) - 0.5)
            ax.set_ylim(0, 1.12)
            ax.set_yticks(np.arange(0, 1.01, 0.2))
            S.style_axes(ax, ylabel=metric)
            if r_i == 0:
                ax.set_title(dict(SETS)[qs], fontsize=11, pad=6, color="black")
    S.panel_letters(axes.ravel(), "ABCD", x=-0.14)
    S.shared_legend(fig, BAR_METHODS, ncol=4, y=-0.004,
                    extra=[S.oracle_handle()])
    fig.tight_layout(rect=(0, 0.075, 1, 1), pad=0.5, w_pad=0.6, h_pad=0.7)
    for ext in ("png",):
        fig.savefig(os.path.join(FIG, f"structure_truth.{ext}"), dpi=200,
                    bbox_inches="tight")
    json.dump(counts, open(os.path.join(E.HERE, "structure_truth_counts.json"), "w"), indent=1)

    # ---------------------------------------------------------------- table
    report = json.load(open(os.path.join(E.HERE, "evaluation.json")))

    def cell(qs, truth, method, metric):
        d = report.get(qs, {}).get(f"{truth}|none", {}).get(method)
        return (d["mean"][metric], d["n"]) if d else (np.nan, 0)

    lines = ["\\begin{tabular}{l" + "cc" * len(groups) + "}", "\\toprule",
             "Method & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{h}}}" for _, h in groups) + " \\\\",
             " & " + " & ".join(["Sens@1FP & Recall@1"] * len(groups)) + " \\\\", "\\midrule"]
    for qs, caption in SETS:
        if qs not in report:
            continue
        ns = [cell(qs, t, REF, "Recall@1")[1] for t, _ in groups]
        lines.append("\\multicolumn{%d}{l}{\\textit{%s} ($n = %s$)} \\\\"
                     % (1 + 2 * len(groups), caption, ", ".join(str(n) for n in ns)))
        for m in BAR_METHODS + [ORACLE]:
            cs = []
            for truth, _ in groups:
                s1, _ = cell(qs, truth, m, "Sens@1FP")
                r1, _ = cell(qs, truth, m, "Recall@1")
                cs += [f"{s1:.3f}" if np.isfinite(s1) else "--",
                       f"{r1:.3f}" if np.isfinite(r1) else "--"]
            lines.append(f"\\quad {S.SHORT[m]} & " + " & ".join(cs) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    out = os.path.join(E.HERE, "structure_truth_table.tex")
    open(out, "w").write("\n".join(lines) + "\n")
    print("wrote", os.path.join(FIG, "structure_truth.png"))
    print("wrote", out)


if __name__ == "__main__":
    main()
