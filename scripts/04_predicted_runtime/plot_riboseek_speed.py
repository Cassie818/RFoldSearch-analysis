"""Supplementary figure: where RFoldseek's speed advantage over RiboSeek comes from.

The two methods share the pipeline shape -- k-mer prefilter then
Smith-Waterman rerank -- and differ in the alphabet, so the runtime
comparison is a scaling comparison rather than a constant factor.
Panel A shows that RiboSeek is the faster of the two on small
databases and that the ratio crosses one near 25,000 structures;
panel B shows the crossing is population-wide at full scale, not
carried by a subset of queries.

Data: the NuFold ladder, 300 queries held constant across every size
and both methods (files/predicted_runtime/nufold).
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import runtime_common as F          # noqa: E402
import runtime_style as P          # noqa: E402

MM = F.MM
SIZES = [1000, 5000, 10000, 20000, 50000, 100000, 200000, 531888]
NU = F.NU_DIR
BLUE, PINK = P.COLOR["RFoldseek"], P.COLOR["RiboSeek"]
INK, MUTED = P.INK, P.MUTED


def ladder(pat):
    return np.array([json.load(open(os.path.join(NU, pat.format(n=n))))["mean_s"]
                     for n in SIZES])


def main():
    rf = ladder("nufold_timing_{n}.json")
    rs = ladder("nufold_riboseek_{n}.json")
    N = np.array(SIZES, dtype=float)
    ratio = rs / rf
    a_rf = np.polyfit(np.log(N), np.log(rf), 1)[0]
    a_rs = np.polyfit(np.log(N), np.log(rs), 1)[0]

    # where the ratio crosses one, interpolated in log-log
    i = int(np.argmax(ratio > 1.0))
    cross = np.exp(np.log(N[i - 1]) + (np.log(1.0) - np.log(ratio[i - 1]))
                   * (np.log(N[i]) - np.log(N[i - 1]))
                   / (np.log(ratio[i]) - np.log(ratio[i - 1])))

    full = json.load(open(os.path.join(NU, "nufold_timing_531888.json")))["per_query"]
    rsq = json.load(open(os.path.join(NU, "nufold_riboseek_531888.json")))["per_query"]
    assert [q["id"] for q in full] == [q["id"] for q in rsq], "query sets differ"
    t_rf = np.array([q["s"] for q in full])
    t_rs = np.array([q["s"] for q in rsq])
    qlen = np.array([q["tokens"] for q in full], dtype=float)

    fig = plt.figure(figsize=(180 * MM, 64 * MM))
    gs = fig.add_gridspec(1, 2, left=0.075, right=0.955, top=0.86,
                          bottom=0.165, wspace=0.30)
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    # ---- A: ratio against database size
    a.axhspan(1.0, 12, color=BLUE, alpha=0.05, lw=0)
    a.axhline(1.0, color=INK, lw=0.8, dashes=(2.2, 2.0), zorder=3)
    a.plot(N, ratio, "-o", color=BLUE, lw=1.9, ms=4.0, mec="white", mew=0.6,
           zorder=5)
    a.axvline(cross, color=MUTED, lw=0.7, dashes=(1.4, 1.8), zorder=2)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_ylim(0.45, 12)
    a.set_xlabel("Database size (structures)")
    a.set_ylabel("RiboSeek time / RFoldseek time")
    a.set_xticks([1000, 10000, 100000, 531888])
    a.set_xticklabels(["1k", "10k", "100k", "532k"])
    a.set_yticks([0.5, 1, 2, 5, 10])
    a.set_yticklabels(["0.5", "1", "2", "5", "10"])
    a.text(cross * 0.86, 0.50, f"crossover\n{cross/1000:.0f}k structures",
           fontsize=5.8, color=MUTED, va="bottom", ha="right",
           linespacing=1.3)
    a.annotate(f"{ratio[-1]:.1f}×", xy=(N[-1], ratio[-1]),
               xytext=(-4, 7), textcoords="offset points", fontsize=7,
               color=BLUE, fontweight="bold", ha="right")
    a.text(0.035, 0.955,
           f"cost $\\propto N^{{{a_rf:.2f}}}$  RFoldseek\n"
           f"cost $\\propto N^{{{a_rs:.2f}}}$  RiboSeek",
           transform=a.transAxes, fontsize=6.0, va="top", color=INK,
           linespacing=1.5)
    # the shaded half is where RFoldseek wins; label each side of the 1.0 line
    a.text(0.985, 1.22, "RFoldseek faster", transform=a.get_yaxis_transform(),
           fontsize=5.8, color=BLUE, ha="right", va="bottom")
    a.text(0.985, 0.82, "RiboSeek faster", transform=a.get_yaxis_transform(),
           fontsize=5.8, color=PINK, ha="right", va="top")

    # ---- B: paired per-query cost at full scale
    lim = (10 ** np.floor(np.log10(min(t_rf.min(), t_rs.min()))),
           10 ** np.ceil(np.log10(max(t_rf.max(), t_rs.max()))))
    b.plot(lim, lim, "-", color=INK, lw=0.8, dashes=(2.2, 2.0), zorder=2)
    sc = b.scatter(t_rf, t_rs, c=qlen, cmap="viridis_r", s=7, lw=0,
                   alpha=0.85, zorder=5, norm=matplotlib.colors.LogNorm())
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlim(*lim); b.set_ylim(*lim)
    b.set_xlabel("RFoldseek time per query (s)")
    b.set_ylabel("RiboSeek time per query (s)")
    cb = fig.colorbar(sc, ax=b, pad=0.02, fraction=0.045)
    cb.set_label("Query length (tokens)", fontsize=6.0)
    cb.ax.tick_params(labelsize=5.6, length=2.0)
    n_above = int((t_rs > t_rf).sum())
    sp = t_rs / t_rf
    b.text(0.035, 0.955,
           f"{n_above} of {len(t_rf)} queries above the diagonal\n"
           f"median {np.median(sp):.0f}×, IQR "
           f"{np.percentile(sp, 25):.0f}–{np.percentile(sp, 75):.0f}×",
           transform=b.transAxes, fontsize=6.0, va="top", color=INK,
           linespacing=1.5)

    for ax, lab in ((a, "A"), (b, "B")):
        ax.text(-0.135, 1.06, lab, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="baseline", color=INK)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
        ax.minorticks_off()
        ax.tick_params(length=2.0, pad=1.6, labelsize=6)

    for ext in ("png",):
        out = os.path.join(F.FIGDIR, f"riboseek_speed.{ext}")
        fig.savefig(out, dpi=600 if ext == "png" else None)
        print("wrote", out)
    print(f"crossover {cross:,.0f}; alpha RF {a_rf:.3f} RS {a_rs:.3f}; "
          f"{n_above}/{len(t_rf)} above diagonal; median speedup {np.median(sp):.1f}x")


if __name__ == "__main__":
    main()
