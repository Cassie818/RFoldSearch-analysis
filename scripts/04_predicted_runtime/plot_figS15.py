"""Supplementary Fig S15: per-query cost against query length.

Sec 2.3 argues that mean and median per-query times diverge because prefilter
work grows with query length, so a minority of long queries carries the mean
while barely moving the median. Until the 300-query re-run recorded a length
per query that had to be inferred from the size of the gap; with 300 points
per corpus it can be drawn.

One point per query at full database size, both axes logarithmic, with the
mean and median of the same points as horizontal rules -- the mean sits among
the slowest tenth rather than among typical queries.

The trend line is a binned median, not a fitted power law, because at a given
length the cost spans up to 150-fold: length is a strong but incomplete
predictor (r^2 = 0.63 on RhoFold+), and a single exponent would imply a
tightness the data does not have.

What the bins show is a threshold rather than a smooth law. On RhoFold+ the
median holds between 12 and 30 ms from 35 to 109 tokens and then jumps
16-fold, to 475 ms, in the next bin: the point at which posting volume
overtakes the reranking pool, which is capped at top-N and therefore
length-bound rather than database-bound. NuFold crosses the same threshold
more gently, since its queries are shorter and most remain rerank-bound.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import runtime_common as F
import runtime_style as P

MM = F.MM
WIDTH_MM, HEIGHT_MM = 180.0, 78.0

PANELS = [
    ("A", "RhoFold+ database", "83,156 structures",
     "rhofold/kernel_timing_83156.json", (4e-3, 20), (20, 700)),
    ("B", "NuFold database", "531,888 structures",
     "nufold/nufold_timing_531888.json", (4e-3, 20), (15, 700)),
]


def load(rel):
    d = json.load(open(os.path.join(F.DATA_DIR, rel)))["per_query"]
    return (np.array([q["tokens"] for q in d], dtype=float),
            np.array([q["s"] for q in d], dtype=float))


def draw(ax, lengths, times, xlim, ylim):
    blue = P.COLOR["RFoldseek"]
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)

    for y in (1e-2, 1e-1, 1e0, 1e1):
        if ylim[0] <= y <= ylim[1]:
            ax.axhline(y, color="#E6E6E6", lw=0.45, zorder=0)

    mean, med = times.mean(), np.median(times)
    for v, lab, style in ((med, "median", (1.4, 1.6)), (mean, "mean", None)):
        ax.axhline(v, color=P.INK, lw=0.7, zorder=3,
                   **(dict(dashes=style) if style else {}))
        ax.text(xlim[1] * 0.94, v, f" {lab}\n {v * 1000:.0f} ms" if v < 1
                else f" {lab}\n {v:.2f} s", color=P.INK, fontsize=5.6,
                va="center", ha="right", linespacing=1.25,
                bbox=dict(fc="white", ec="none", pad=0.8), zorder=5)

    ax.scatter(lengths, times, s=5.5, color=blue, alpha=0.5, lw=0, zorder=5)

    # Binned median rather than a power-law fit: at a given length the cost
    # spans two orders of magnitude, so one exponent would imply a tightness
    # the data does not have. The exponent is still quoted as a summary.
    edges = np.logspace(np.log10(lengths.min()), np.log10(lengths.max()), 9)
    cx, cy = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (lengths >= lo) & ((lengths < hi) if hi < edges[-1] else (lengths <= hi))
        if m.sum() >= 5:
            cx.append(float(np.sqrt(lo * hi))); cy.append(float(np.median(times[m])))
    ax.plot(cx, cy, "-", color=blue, lw=1.7, marker="o", ms=3.4, mec="white",
            mew=0.5, zorder=7)

    slope, intercept = np.polyfit(np.log(lengths), np.log(times), 1)
    r = np.corrcoef(np.log(lengths), np.log(times))[0, 1]
    ax.text(0.035, 0.955,
            f"binned median\n$t \\propto L^{{{slope:.1f}}}$, $r^2={r ** 2:.2f}$",
            transform=ax.transAxes, fontsize=6.0, color=blue, va="top",
            linespacing=1.4)

    ax.set_xticks([20, 50, 100, 200, 500])
    ax.set_xticklabels(["20", "50", "100", "200", "500"])
    ax.set_yticks([1e-2, 1e-1, 1e0, 1e1])
    ax.set_yticklabels(["10 ms", "100 ms", "1 s", "10 s"])
    ax.minorticks_off()
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=2.0, pad=1.6)


def main():
    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM))
    gs = fig.add_gridspec(1, 2, left=0.085, right=0.985, top=0.845,
                          bottom=0.165, wspace=0.16)
    for i, (lab, title, sub, rel, ylim, xlim) in enumerate(PANELS):
        ax = fig.add_subplot(gs[0, i])
        L, T = load(rel)
        draw(ax, L, T, xlim, ylim)
        ax.set_xlabel("Query length (tokens)")
        if i == 0:
            ax.set_ylabel("Search time per query")
        else:
            ax.set_yticklabels([])
        ax.text(0.0, 1.115, f"{title} ({sub})", transform=ax.transAxes,
                fontsize=6.4, color=P.INK, va="baseline")
        ax.text(-0.155 if i == 0 else -0.09, 1.115, lab, transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="baseline", ha="left",
                color=P.INK)
        n_slow = max(1, len(T) // 10)
        share = np.sort(T)[::-1][:n_slow].sum() / T.sum()
        ax.text(0.035, 0.055,
                f"slowest {n_slow} of {len(T)} queries: {share * 100:.0f}% of total time",
                transform=ax.transAxes, fontsize=5.5, color=P.MUTED, va="bottom")

    for ext in ("png",):
        out = os.path.join(F.FIGDIR, f"S15.{ext}")
        fig.savefig(out, dpi=600 if ext == "png" else None)
        print("wrote", out)


if __name__ == "__main__":
    main()
