"""Supplementary figure: residue-level alignment benchmark against Rfam 3D seed
alignments (313 within-family pairs, 24 families).

    A    Exact and +/-1-column alignment F1, including a global sequence baseline
    B    Paired family-level differences from RFoldseek on the same cohort
    C-E  Per-pair view: 3Dr token vs nucleotide identity, and RFoldseek exact F1
         against US-align and the global sequence baseline; cases 1-4 are the
         RF00015 examples of the alignment case figure (panels A-D there)
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "02_rnasolo_retrieval"))
from plot_fig2 import FIG_DIR, LABEL, METHOD_BASE, METHOD_COLOR, np, plt, resolve  # noqa: E402

ALIGN_METHODS = ["RFoldseek", "RiboSeek", "US-align", "Global sequence"]
ALIGN_KEY = {"RFoldseek": "rfoldseek", "RiboSeek": "riboseek",
             "US-align": "usalign", "Global sequence": "sequence_alignment"}
METHOD_COLOR["Global sequence"] = "#777777"
METHOD_BASE["Global sequence"] = "#555555"


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

    ax.set_xticks(x, [LABEL.get(m, m) for m in ALIGN_METHODS])
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
        ("rfoldseek_minus_sequence_alignment_f1", "Global sequence", "Exact", True),
        ("rfoldseek_minus_usalign_tolerant_f1", "US-align", r"$\pm$1 nt", False),
        ("rfoldseek_minus_riboseek_tolerant_f1", "RiboSeek", r"$\pm$1 nt", False),
        ("rfoldseek_minus_sequence_alignment_tolerant_f1", "Global sequence", r"$\pm$1 nt", False),
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
        ax.text(0.235, yy, f"{mean:+.3f}{star}", ha="right", va="center",
                fontsize=8.5, color=METHOD_BASE[competitor])

    ax.axvline(0, color="#777777", lw=1.1, ls="--", zorder=1)
    def row_label(competitor, metric):
        name = LABEL.get(competitor, competitor)
        return f"vs {name[:-1]}, {metric})" if name.endswith(")") else f"vs {name} ({metric})"
    ax.set_yticks(y, [row_label(competitor, metric) for _, competitor, metric, _ in rows])
    ax.set_xlabel(r"$\Delta$F1 (RFoldseek $-$ comparator)", fontsize=12)
    ax.set_xlim(-0.30, 0.25)
    ax.set_ylim(-0.65, len(rows) - 0.35)
    ax.tick_params(axis="both", labelsize=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


PAIRS = ("files/ablation/rfam3d_reference_disjoint_independent_3dseed/"
         "rfam3d_reference_alignment_with_sequence_baseline.pairs.tsv")
CASES = [("1", "2N7M_X"), ("2", "5GAN_V"), ("3", "6AH0_I"), ("4", "6QW6_4")]


def pair_scatter(ax, rows, cases, xkey, ykey, xlabel, ylabel, offsets=None):
    """Per-pair scatter on the unit square with the four RF00015 cases marked."""
    ax.plot([0, 1], [0, 1], color="#888888", lw=0.9, zorder=1)
    ax.scatter([float(r[xkey]) for r in rows], [float(r[ykey]) for r in rows],
               s=14, color="#79B7D9", alpha=0.5, lw=0, zorder=2)
    for lab, target in CASES:
        x, y = float(cases[target][xkey]), float(cases[target][ykey])
        ax.plot(x, y, "D", ms=7, mfc="#E6812B", mec="#20282D", mew=0.7, zorder=4)
        xy = (offsets or {}).get(lab, (-9 if x > 0.9 else 6, -11 if y > 0.95 else 5))
        ax.annotate(lab, (x, y), xytext=xy,
                    textcoords="offset points", fontsize=10, zorder=5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)


def load_pairs():
    import csv
    rows = list(csv.DictReader(open(resolve(PAIRS)), delimiter="\t"))
    assert len(rows) == 313 and len({r["family"] for r in rows}) == 24
    cases = {}
    for _, target in CASES:
        m = [r for r in rows if r["family"] == "RF00015" and {r["query_chain"], r["target_chain"]} == {"5O9Z_4", target}]
        assert len(m) == 1, target
        cases[target] = m[0]
    return rows, cases


def main():
    alignment = json.load(open(resolve(
        "files/ablation/rfam3d_reference_disjoint_independent_3dseed/"
        "rfam3d_reference_alignment_common_313.json")))
    rows, cases = load_pairs()
    fig = plt.figure(figsize=(14.0, 9.0))
    top, bottom = fig.subfigures(2, 1, height_ratios=[0.8, 1.0], hspace=0.0)
    axd = dict(zip("AB", top.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.62})))
    axd.update(zip("CDE", bottom.subplots(1, 3, gridspec_kw={"wspace": 0.38})))
    alignment_panel(axd["A"], alignment)
    alignment_difference_panel(axd["B"], alignment)
    pair_scatter(axd["C"], rows, cases, "token_identity", "sequence_identity",
                 "3Dr token identity", "Nucleotide sequence identity")
    pair_scatter(axd["D"], rows, cases, "rfoldseek_f1", "usalign_f1",
                 "RFoldseek exact F1", "US-align exact F1")
    pair_scatter(axd["E"], rows, cases, "rfoldseek_f1", "sequence_alignment_f1",
                 "RFoldseek exact F1", "Global sequence exact F1", offsets={"2": (8, -10), "3": (-4, 8)})
    for lab in "ABCDE":
        axd[lab].text(-0.10 if lab in "AB" else -0.22, 1.04, lab, transform=axd[lab].transAxes,
                      fontsize=16, fontweight="bold", va="bottom", ha="right")
    out_png = Path(FIG_DIR) / "supp_alignment_benchmark.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
