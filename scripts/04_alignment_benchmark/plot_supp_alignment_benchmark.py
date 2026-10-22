"""Supplementary figure: residue-level alignment benchmark against Rfam 3D seed
alignments (313 within-family pairs, 24 families).

    A    Exact and +/-1-column alignment F1, including a global sequence baseline
    B    Paired family-level differences from RFoldSearch on the same cohort
    C-E  Per-pair view: 3Dr token vs nucleotide identity, and RFoldSearch exact F1
         against US-align and the global sequence baseline; cases 1-4 are the
         RF00015 examples of the alignment case figure (panels A-D there)
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "02_rnasolo_retrieval"))
from plot_fig2 import FIG_DIR, LABEL, METHOD_BASE, METHOD_COLOR, np, plt, resolve  # noqa: E402

ALIGN_METHODS = ["RFoldSearch", "RiboSeek", "US-align", "Global sequence"]
ALIGN_KEY = {"RFoldSearch": "rfoldsearch", "RiboSeek": "riboseek",
             "US-align": "usalign", "Global sequence": "sequence_alignment"}
METHOD_COLOR["Global sequence"] = "#777777"
METHOD_BASE["Global sequence"] = "#555555"


GREY = "#666666"
METRICS = [("f1", "Exact", True, -0.14), ("tolerant_f1", r"$\pm$1 nt", False, 0.14)]


def _estimate(summary, key):
    item = summary[key]
    return item["family_macro_mean"], item["family_block_bootstrap_95ci"]


def _style(ax, xlabel):
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(axis="x", color="#E6E6E6", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(xlabel, fontsize=12)


def _metric_legend(ax):
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color="#555555", ms=8, label="Exact"),
                       plt.Line2D([], [], marker="o", ls="", color="#555555", mfc="white", mew=1.6, ms=8,
                                  label=r"$\pm$1 nt")],
              frameon=False, fontsize=9.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              labelcolor=GREY)


def _point(ax, mean, lo, hi, y, method, filled):
    ax.hlines(y, lo, hi, color=METHOD_COLOR[method], lw=2.0, zorder=2)
    ax.plot(mean, y, "o", ms=8, color=METHOD_COLOR[method], mfc=METHOD_COLOR[method] if filled else "white",
            mec="white" if filled else METHOD_COLOR[method], mew=0.8 if filled else 1.6, zorder=3)


def alignment_panel(ax, alignment):
    """Family-macro alignment F1 (point and family-block-bootstrap 95% CI), exact and +/-1 nt."""
    summary = alignment["summary"]
    for i, method in enumerate(ALIGN_METHODS):
        for metric, _, filled, dy in METRICS:
            mean, (lo, hi) = _estimate(summary, f"{ALIGN_KEY[method]}_{metric}")
            _point(ax, mean, lo, hi, i + dy, method, filled)
            ax.text(hi + 0.006, i + dy, f"{mean:.2f}", va="center", fontsize=9,
                    color=METHOD_BASE[method] if method == "RFoldSearch" else GREY)
    ax.set_yticks(range(len(ALIGN_METHODS)), [LABEL.get(m, m) for m in ALIGN_METHODS], fontsize=11)
    ax.set_ylim(len(ALIGN_METHODS) - 0.45, -0.55)
    ax.set_xlim(0.55, 1.0)
    ax.set_xticks(np.arange(0.6, 1.01, 0.1))
    _style(ax, "Alignment F1")
    _metric_legend(ax)


def alignment_difference_panel(ax, alignment):
    """Paired family-macro differences RFoldSearch - comparator, exact and +/-1 nt."""
    summary = alignment["summary"]
    key = {"US-align": "usalign", "RiboSeek": "riboseek", "Global sequence": "sequence_alignment"}
    comparators = list(key)
    for i, comp in enumerate(comparators):
        for metric, _, filled, dy in METRICS:
            mean, (lo, hi) = _estimate(summary, f"rfoldsearch_minus_{key[comp]}_{metric}")
            _point(ax, mean, lo, hi, i + dy, comp, filled)
            star = "*" if lo > 0 or hi < 0 else ""
            ax.text(max(hi, 0) + 0.008, i + dy, f"{mean:+.3f}{star}", va="center", fontsize=9, color=GREY)
    ax.axvline(0, color="#888888", lw=1.1, ls="--", zorder=1)
    ax.set_yticks(range(len(comparators)), [LABEL.get(m, m) for m in comparators], fontsize=11)
    ax.set_ylim(len(comparators) - 0.45, -0.55)
    ax.set_xlim(-0.3, 0.3)
    ax.set_xticks(np.arange(-0.3, 0.31, 0.1))
    _style(ax, r"$\Delta$F1 (RFoldSearch $-$ comparator)")
    _metric_legend(ax)


PAIRS = ("files/ablation/rfam3d_reference_disjoint_independent_3dseed/"
         "rfam3d_reference_alignment_with_sequence_baseline.pairs.tsv")
CASES = [("1", "2N7M_X"), ("2", "5GAN_V"), ("3", "6AH0_I"), ("4", "6QW6_4")]


def pair_scatter(ax, rows, cases, xkey, ykey, xlabel, ylabel, offsets=None):
    """Per-pair scatter on the unit square with the four RF00015 cases marked."""
    ax.plot([0, 1], [0, 1], color="#AAAAAA", lw=0.9, ls="--", zorder=1)
    ax.scatter([float(r[xkey]) for r in rows], [float(r[ykey]) for r in rows],
               s=16, color=METHOD_COLOR["RFoldSearch"], alpha=0.45, lw=0, zorder=2)
    for lab, target in CASES:
        x, y = float(cases[target][xkey]), float(cases[target][ykey])
        ax.plot(x, y, "D", ms=7, mfc="#444444", mec="white", mew=0.9, zorder=4)
        xy = (offsets or {}).get(lab, (-9 if x > 0.9 else 6, -11 if y > 0.95 else 5))
        ax.annotate(lab, (x, y), xytext=xy, textcoords="offset points", fontsize=10, color="#444444",
                    fontweight="bold", zorder=5)
    ax.grid(color="#EEEEEE", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
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
    fig = plt.figure(figsize=(14.0, 8.4))
    top, bottom = fig.subfigures(2, 1, height_ratios=[0.75, 1.0], hspace=0.0)
    axd = dict(zip("AB", top.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1], "wspace": 0.45})))
    axd.update(zip("CDE", bottom.subplots(1, 3, gridspec_kw={"wspace": 0.38})))
    alignment_panel(axd["A"], alignment)
    alignment_difference_panel(axd["B"], alignment)
    pair_scatter(axd["C"], rows, cases, "token_identity", "sequence_identity",
                 "3Dr token identity", "Nucleotide sequence identity")
    pair_scatter(axd["D"], rows, cases, "rfoldsearch_f1", "usalign_f1",
                 "RFoldSearch exact F1", "US-align exact F1")
    pair_scatter(axd["E"], rows, cases, "rfoldsearch_f1", "sequence_alignment_f1",
                 "RFoldSearch exact F1", "Global sequence exact F1", offsets={"2": (8, -10), "3": (-4, 8)})
    for lab in "ABCDE":
        axd[lab].text({"A": -0.30, "B": -0.30, "C": -0.50}.get(lab, -0.24), 1.06, lab, transform=axd[lab].transAxes,
                      fontsize=16, fontweight="bold", va="bottom", ha="right")
    out_png = Path(FIG_DIR) / "supp_alignment_benchmark.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
