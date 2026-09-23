"""Supplementary figure replacing Table S10: the two predicted databases.

Panel A: chain-length distributions of the RhoFold+ (83,156) and NuFold
(1-13-2026 catalogue, length >= 20) collections. Panel B: predicted
confidence against length, binned median with interquartile band.
RhoFold+ reports pLDDT on 0-1 and is shown x100 so both databases share
one axis. Data: data/nufold_catalog/ in the analysis workspace, pulled
from the AutoDL server copy of the NuFold 1-13-2026 catalogue (no
longer downloadable upstream) and the RhoFold+ fold metadata.
"""
import csv
import gzip
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import runtime_common as F          # noqa: E402

MM = F.MM
INK, MUTED = "#1A1A1A", "#6B6B6B"
C_RHO, C_NU = "#D55E00", "#56B4E9"    # db identities; distinct from method hues
# RhoFold+ metadata and NuFold catalog; not bundled (external input).
CAT = os.environ.get("RFOLDSEEK_NUFOLD_CATALOG",
                     "/Users/dylanyang/Documents/workspace/rfoldseek/data/nufold_catalog")


def load():
    rl, rp = [], []
    with gzip.open(f"{CAT}/rhofold_metadata.tsv.gz", "rt") as f:
        next(f)
        for line in f:
            c = line.split("\t")
            rl.append(int(c[2])); rp.append(float(c[4]) * 100.0)
    nl, np_ = [], []
    with gzip.open(f"{CAT}/data-1-13-2026.csv.gz", "rt") as f:
        for row in csv.DictReader(f):
            try:
                L = int(row["sequence_length"]); p = float(row["avg_plddt"])
            except (ValueError, TypeError):
                continue
            if L >= 20:
                nl.append(L); np_.append(p)
    return (np.array(rl), np.array(rp)), (np.array(nl), np.array(np_))


def binned(x, y, edges):
    cx, med, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        if m.sum() < 50:
            continue
        cx.append(np.sqrt(a * b))
        med.append(np.median(y[m]))
        lo.append(np.percentile(y[m], 25)); hi.append(np.percentile(y[m], 75))
    return map(np.array, (cx, med, lo, hi))


def main():
    (rl, rp), (nl, npl) = load()
    print(f"RhoFold+ n={len(rl):,}  NuFold(>=20nt) n={len(nl):,}")

    fig = plt.figure(figsize=(180 * MM, 62 * MM))
    gs = fig.add_gridspec(1, 2, left=0.065, right=0.985, top=0.86,
                          bottom=0.16, wspace=0.26)
    a = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gs[0, 1])

    bins = np.logspace(np.log10(20), np.log10(500), 36)
    for x, c, lab in ((rl, C_RHO, "RhoFold+ (83,156)"),
                      (nl, C_NU, f"NuFold ({len(nl):,})")):
        a.hist(x, bins=bins, density=True, histtype="step", lw=1.6,
               color=c, label=lab)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xticks([20, 50, 100, 200, 500])
    a.set_xticklabels(["20", "50", "100", "200", "500"])
    a.set_xlabel("Chain length (nt)")
    a.set_ylabel("Density")
    a.legend(frameon=False, fontsize=6.2, loc="upper right")
    for v, c in ((np.median(rl), C_RHO), (np.median(nl), C_NU)):
        a.axvline(v, color=c, lw=0.8, dashes=(1.5, 1.8), alpha=0.8)
    a.text(0.02, 1.06, "A", transform=a.transAxes, fontsize=8,
           fontweight="bold", va="baseline")

    edges = np.logspace(np.log10(20), np.log10(500), 12)
    for x, y, c, lab in ((nl, npl, C_NU, "NuFold"), (rl, rp, C_RHO, "RhoFold+")):
        cx, med, lo, hi = binned(x, y, edges)
        b.fill_between(cx, lo, hi, color=c, alpha=0.15, lw=0)
        b.plot(cx, med, "-o", color=c, lw=1.6, ms=3.0, mec="white",
               mew=0.5, label=lab)
    b.axhline(70, color="#BBBBBB", lw=0.6, zorder=0)
    b.set_xscale("log")
    b.set_xticks([20, 50, 100, 200, 500])
    b.set_xticklabels(["20", "50", "100", "200", "500"])
    b.set_ylim(0, 100)
    b.set_xlabel("Chain length (nt)")
    b.set_ylabel("Predicted confidence (pLDDT)")
    b.legend(frameon=False, fontsize=6.2, loc="lower left")
    b.text(0.02, 1.06, "B", transform=b.transAxes, fontsize=8,
           fontweight="bold", va="baseline")

    for ax in (a, b):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.minorticks_off()
        ax.tick_params(length=2.0, pad=1.6)

    for ext in ("png",):
        out = os.path.join(F.FIGDIR, f"dbstats.{ext}")
        fig.savefig(out, dpi=600 if ext == "png" else None)
        print("wrote", out)


if __name__ == "__main__":
    main()
