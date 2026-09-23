"""Supplementary Fig S14: the Fig 3 measurements as scaling shape alone.

Each curve is divided by its own value at 1,000 structures, so every method
starts at 1 and the panels show only how cost per target moves with database
size: horizontal is cost strictly proportional to database size, falling is
sublinear. That compresses the RhoFold+ row from 4.58 decades to 1.89 and the
NuFold row from 4.76 to 2.53, which is why the scaling claim is easiest to
check here.

It is supplementary rather than the main figure because normalising by each
method's own starting point ranks methods by how far they fell, not by what
they cost. BLASTN begins 39x more expensive than RFoldseek on NuFold, falls
336x against RFoldseek's 179x, and so ends up the lowest curve in every panel
-- while in fact costing 20.6x more per target at full scale. Fig 3 keeps the
absolute levels and is the figure to read for ranking.

The band is RFoldseek's interquartile range with each quartile normalised by
its own value at 1,000, so it shows the spread in scaling rather than the
spread in cost, which is what the band means in Fig 3.

The coordinate aligners collapse onto the 1.0 line here -- strict linearity is
exactly cost per target unchanged with scale -- so they are one gray band, and
their extrapolated points at 83,156 are omitted as elsewhere.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import runtime_common as F
import runtime_style as P

MM = F.MM
WIDTH_MM, HEIGHT_MM = 180.0, 104.0


def main():
    rho, rho_iqr = F.load_rhofold()
    nu, nu_iqr = F.load_nufold()
    coord = [rho[m] for m in ("RMalign", "US-align", "RTM-align")]

    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM))
    gs = fig.add_gridspec(2, 2, left=0.085, right=0.885, top=0.885,
                          bottom=0.105, wspace=0.40, hspace=0.46)
    a = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gs[0, 1])
    c = fig.add_subplot(gs[1, 0]); d = fig.add_subplot(gs[1, 1])

    RHO_YL, NU_YL = (9.0e-3, 2.0), (2.0e-3, 2.6)
    P.draw_shape(a, rho, P.CURVES, 1, F.SIZES, RHO_YL, iqr=rho_iqr, coord=coord)
    P.draw_shape(b, rho, P.CURVES, 0, F.SIZES, RHO_YL, iqr=rho_iqr, coord=coord)
    P.draw_shape(c, nu, P.CURVES, 1, F.NU_SIZES, NU_YL, iqr=nu_iqr)
    P.draw_shape(d, nu, P.CURVES, 0, F.NU_SIZES, NU_YL, iqr=nu_iqr)

    for ax in (a, c):
        ax.set_title("Median query", pad=4.0, color=P.INK)
    for ax in (b, d):
        ax.set_title("Mean query", pad=4.0, color=P.INK)
    for ax in (a, c):
        ax.set_ylabel("Cost per target,\nrelative to 1,000 structures")
    for ax in (c, d):
        ax.set_xlabel("Database size (structures)")

    for ax in (c, d):
        ax.text(0.035, 0.945, "strictly linear", transform=ax.transAxes,
                fontsize=5.8, color=P.MUTED, va="top")
    for ax in (a, c):
        ax.text(0.035, 0.055, "shaded: RFoldseek interquartile range",
                transform=ax.transAxes, fontsize=5.5,
                color=P.COLOR["RFoldseek"], alpha=0.75, va="bottom")

    a.text(0.0, 1.16, "RhoFold+ database (83,156 structures)",
           transform=a.transAxes, fontsize=6.4, color=P.INK, va="baseline")
    c.text(0.0, 1.16, "NuFold database (531,888 structures)",
           transform=c.transAxes, fontsize=6.4, color=P.INK, va="baseline")
    for ax, lab in ((a, "A"), (b, "B"), (c, "C"), (d, "D")):
        ax.text(-0.19, 1.16, lab, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="baseline", ha="left", color=P.INK)

    fig.align_ylabels([a, c])
    for ext in ("png",):
        out = os.path.join(F.FIGDIR, f"S14.{ext}")
        fig.savefig(out, dpi=600 if ext == "png" else None)
        print("wrote", out)


if __name__ == "__main__":
    main()
