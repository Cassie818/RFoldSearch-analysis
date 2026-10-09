"""Supplementary figure: retrieval under the structure-defined ground truth."""
import json
import os

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import evaluate as E  # noqa: E402
import figstyle as S  # noqa: E402
from figstyle import plt  # noqa: E402

FIG = os.path.join(E.REPO, "figures")
BAR_METHODS = ["RFoldSearch n500", "RFoldSearch-TM n500", "RiboSeek c500", "RiboSeek RS-80 c500", "BLASTN", "nhmmer", "Infernal"]
REF = "RFoldSearch n500"
SETS = [("exp", "Experimental queries"),
        ("p2p", "Predicted queries")]
COLS = ["exp", "p2p"]
ROWS = ["Sens@1FP", "Recall@1"]


def main():
    pq = json.load(open(os.path.join(E.HERE, "per_query.json")))
    report = json.load(open(os.path.join(E.HERE, "evaluation.json")))
    rthr = report.get("rmscore_thresholds", {})
    plot(pq, report, "structure", [(f"structure t{t:.2f}", f"{t:.2f}") for t in E.TM_TAUS], BAR_METHODS)
    if rthr:
        methods = BAR_METHODS[:2] + ["US-align (exhaustive)"] + BAR_METHODS[2:]
        plot(pq, report, "rmscore", [(f"rmscore t{t:.2f}", f"{rthr[f'{t:.2f}']:.2f}") for t in E.TM_TAUS], methods)


def plot(pq, report, truth_name, groups, bar_methods):
    fams = pq["families"]
    ORACLE = E.ORACLE[truth_name]
    BAR_METHODS = bar_methods

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
                                va="bottom", fontsize=8, color=S.TEXT[REF], zorder=5,
                                bbox=dict(facecolor="white", edgecolor="none", pad=0.4, alpha=0.85))
                    elif sig[j]:
                        ax.text(b.get_x() + b.get_width() / 2, top, "*", ha="center", va="bottom",
                                fontsize=10, color=S.GREY_TEXT, fontweight="bold")
            oracle = cells(qs, groups[0][0], ORACLE, metric)
            if oracle:
                S.ceiling(ax, float(np.mean(list(oracle.values()))), -0.5, len(groups) - 0.5, ORACLE)
            ax.set_xticks(x, [lab for _, lab in groups])
            counts.setdefault(qs, {}).update({lab: n for (_, lab), n in zip(groups, ns)})
            ax.set_xlim(-0.5, len(groups) - 0.5)
            ax.set_ylim(0, 1.12)
            ax.set_yticks(np.arange(0, 1.01, 0.2))
            S.style_axes(ax, ylabel=metric)
            if r_i == 0:
                ax.set_title(dict(SETS)[qs], fontsize=11, pad=6, color="black")
    S.panel_letters(axes.ravel(), "ABCD", x=-0.14)
    S.shared_legend(fig, BAR_METHODS, ncol=5 if len(BAR_METHODS) > 7 else 4, y=-0.004,
                    extra=[S.oracle_handle(ORACLE)])
    fig.tight_layout(rect=(0, 0.075, 1, 1), pad=0.5, w_pad=0.6, h_pad=0.7)
    for ext in ("png",):
        fig.savefig(os.path.join(FIG, f"supp_predicted_{truth_name}_truth.{ext}"), dpi=200,
                    bbox_inches="tight")
    json.dump(counts, open(os.path.join(E.HERE, f"{truth_name}_truth_counts.json"), "w"), indent=1)

    # ---------------------------------------------------------------- table

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
    out = os.path.join(E.HERE, f"{truth_name}_truth_table.tex")
    open(out, "w").write("\n".join(lines) + "\n")
    print("wrote", os.path.join(FIG, f"supp_predicted_{truth_name}_truth.png"))
    print("wrote", out)


if __name__ == "__main__":
    main()
