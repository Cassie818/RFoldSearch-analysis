"""Supplementary runtime figure (S14), 3 x 2: RhoFold+ left, NuFold right.

A-D  per-target cost relative to 1,000 structures, for the median (A, B) and mean (C, D)
     query: the Fig. 5A/B measurements as scaling shape alone.
E, F per-query RFoldSearch time against query length on the full databases.
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
WIDTH_MM, HEIGHT_MM = 180.0, 155.0
LENGTH_FILES = {"RhoFold+": "rhofold/kernel_timing_83156.json",
                "NuFold": "nufold/nufold_timing_531888.json"}


def load_lengths(rel):
    d = json.load(open(os.path.join(F.DATA_DIR, rel)))["per_query"]
    return (np.array([q["tokens"] for q in d], dtype=float),
            np.array([q["s"] for q in d], dtype=float))


def length_panel(ax, lengths, times, xlim, ylim):
    """Per-query time against query length: points, binned medians, mean and median."""
    blue = P.COLOR["RFoldSearch"]
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    for y in (1e-2, 1e-1, 1e0, 1e1):
        ax.axhline(y, color="#E6E6E6", lw=0.45, zorder=0)
    mean, med = times.mean(), np.median(times)
    for v, lab, style in ((mean, "Mean", None), (med, "Median", (1.4, 1.6))):
        ax.plot([xlim[0], 500], [v, v], color=P.INK, lw=0.7, zorder=3,
                **(dict(dashes=style) if style else {}))
        ax.text(480, v * 1.12, f"{lab} {v * 1000:.0f} ms", color=P.INK, fontsize=6.0,
                va="bottom", ha="right", zorder=5)
    ax.scatter(lengths, times, s=5.5, color=blue, alpha=0.5, lw=0, zorder=5)
    # binned medians: at one length the cost spans two orders of magnitude
    edges = np.logspace(np.log10(lengths.min()), np.log10(lengths.max()), 9)
    cx, cy = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (lengths >= lo) & ((lengths < hi) if hi < edges[-1] else (lengths <= hi))
        if m.sum() >= 5:
            cx.append(float(np.sqrt(lo * hi))); cy.append(float(np.median(times[m])))
    ax.plot(cx, cy, "-", color=blue, lw=1.5, marker="o", ms=3.2, mec="white", mew=0.5, zorder=7)
    slope = np.polyfit(np.log(lengths), np.log(times), 1)[0]
    r = np.corrcoef(np.log(lengths), np.log(times))[0, 1]
    n_slow = max(1, len(times) // 10)
    share = np.sort(times)[::-1][:n_slow].sum() / times.sum()
    print(f"length exponent {slope:.2f}, r2 {r ** 2:.2f}; slowest {n_slow}: {share * 100:.0f}% of time")
    ax.set_xticks([20, 50, 100, 200, 500])
    ax.set_xticklabels(["20", "50", "100", "200", "500"])
    ax.set_yticks([1e-2, 1e-1, 1e0, 1e1])
    ax.set_yticklabels(["10 ms", "100 ms", "1 s", "10 s"])
    ax.minorticks_off()
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=2.0, pad=1.6)


def main():
    rho, rho_iqr = F.load_rhofold()
    nu, nu_iqr = F.load_nufold()
    coord = [rho[m] for m in ("RMalign", "US-align", "RTM-align")]

    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM))
    gs = fig.add_gridspec(3, 2, left=0.095, right=0.885, top=0.95, bottom=0.07,
                          wspace=0.40, hspace=0.42)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(3)]

    RHO_YL, NU_YL = (1e-3, 2.0), (1e-3, 2.6)
    for row, (stat, name) in enumerate(((1, "median"), (0, "mean"))):
        a, b = axes[row]
        P.draw_shape(a, rho, P.CURVES, stat, F.SIZES, RHO_YL, iqr=rho_iqr, coord=coord)
        P.draw_shape(b, nu, P.CURVES, stat, F.NU_SIZES, NU_YL, iqr=nu_iqr)
        a.set_title(f"RhoFold+ ({name})", pad=4.0, color=P.INK)
        b.set_title(f"NuFold ({name})", pad=4.0, color=P.INK)
        a.set_ylabel("Cost per target,\nrelative to 1,000 structures")
        b.text(0.62, 0.955, "strictly linear", transform=b.transAxes, fontsize=5.8,
               color=P.MUTED, va="top")
    for ax in axes[1]:
        ax.set_xlabel("Database size")

    for ax, (name, rel), xlim in zip(axes[2], LENGTH_FILES.items(), ((20, 700), (15, 700))):
        lengths, times = load_lengths(rel)
        length_panel(ax, lengths, times, xlim, (4e-3, 20))
        ax.set_title(name, pad=4.0, color=P.INK)
        ax.set_xlabel("Query length")
    axes[2][0].set_ylabel("Search time per query")

    for ax, lab in zip([ax for row in axes for ax in row], "ABCDEF"):
        ax.text(-0.19, 1.08, lab, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="baseline", ha="left", color=P.INK)
    fig.align_ylabels([row[0] for row in axes])
    out = os.path.join(F.FIGDIR, "supp_runtime_scaling.png")
    fig.savefig(out, dpi=600)
    print("wrote", out)


if __name__ == "__main__":
    main()
