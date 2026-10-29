"""Runtime panels: per-target search cost against database size, one panel per corpus,
with the full-database search-time ladder inset on the right of each panel."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import runtime_common as F
import runtime_style as P

MM = F.MM
WIDTH_MM, HEIGHT_MM = 180.0, 82.0
DATA_FRAC = 0.66          # share of the panel width used by the curves
# sequence methods dashed as in Fig. 2; fast presets dashed (GTalign) or dotted (Infernal,
# whose default mode is already dashed)
DASHED = {"GTalign-fast", "BLASTN", "Infernal"}
DOTTED = {"Infernal-fast"}


def linestyle(m):
    return ":" if m in DOTTED else "--" if m in DASHED else "-"


def setup_styles(tone=None):
    """`tone(m, colour)` recolours the marks, as Fig. 5 does for methods other than RFoldSearch."""
    P.LADDER_NAME["RFoldSearch-TM"] = "RFoldSearch-TM"
    if tone:
        for d in (F.COLOR, P.COLOR):
            for m in list(d):
                d[m] = tone(m, d[m])


def panel(ax, data, sizes, ylim, methods, ladder_ylim, title):
    """Mean curves on the left, search-time ladder inset on the right."""
    k = P.K
    for m in methods:
        xs, ys = P.per_thousand(data[m], 0)
        lw, ms, z = P.curve_style(m)
        ax.plot(xs, ys, linestyle(m), color=P.COLOR[m], lw=lw,
                marker=P.MARKER[m], ms=ms, mec="white", mew=0.4 * k, zorder=z)

    ax.set_xscale("log"); ax.set_yscale("log")
    x0, x1 = np.log10(sizes[0] * 0.82), np.log10(sizes[-1] * 1.15)
    ax.set_xlim(10 ** x0, 10 ** (x0 + (x1 - x0) / DATA_FRAC))
    ax.set_ylim(*ylim)
    decades = [n for n in (1000, 10000, 100000) if n < sizes[-1] * 0.8]
    ticks = decades + [sizes[-1]]
    ax.set_xticks(ticks)
    ax.set_xticklabels([F.db_tick_label(d) for d in ticks])
    ax.minorticks_off()
    ax.tick_params(length=2.0 * k, pad=1.6 * k, width=0.5 * k, labelsize=6.2 * k)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5 * k)
    ax.spines[["top", "right"]].set_visible(False)       # as in Figs 2 and 5
    ax.spines["bottom"].set_bounds(ax.get_xlim()[0], 10 ** x1)   # x axis ends with the curves
    ax.set_xlabel("Database size", x=DATA_FRAC / 2, fontsize=7.2 * k)
    ax.set_title(title, fontsize=7.2 * k, color=P.INK, pad=4 * k)

    ins = ax.inset_axes([DATA_FRAC + 0.01, 0.03, 0.99 - DATA_FRAC - 0.01, 0.86])
    ins.patch.set_alpha(0)
    P.ladder(ins, F.full_size_values(data, 0, sizes[-1]), "Search time",
             ladder_ylim, "", rail=0.36, ratio_x=0.60, name_x=None, step=0.07)
    return ins


LEGEND_LABEL = {"GTalign": "GTalign (GPU)", "GTalign-fast": "GTalign-fast (GPU)",
                "RiboSeek-RS20": "RiboSeek (RS-20)", "RiboSeek": "RiboSeek (RS-80)"}


def legend_label(m):
    return LEGEND_LABEL.get(m, m)


def legend_handles():
    order = ["RFoldSearch", "RFoldSearch-TM", "GTalign", "GTalign-fast", "RiboSeek-RS20",
             "RiboSeek", "BLASTN", "Infernal", "Infernal-fast", "US-align", "RTM-align", "RMalign"]
    handles = []
    for m in order:
        lab = legend_label(m)
        ls = "none" if m == "RFoldSearch-TM" else linestyle(m)
        handles.append(plt.Line2D([], [], color=F.COLOR[m], marker=F.MARKER[m], ls=ls,
                                  lw=1.1 * P.K, ms=(7.0 if m == "RFoldSearch-TM" else 4.0) * P.K,
                                  mec="white", mew=0.4 * P.K, label=lab))
    return handles


def speed_row(fig, cell, wspace=0.06):
    """The RhoFold+ and NuFold panels side by side in `cell`, sharing both y axes.
    Returns the two axes."""
    rho, _ = F.load_rhofold()
    nu, _ = F.load_nufold()
    tm_rho = json.load(open(os.path.join(F.RHO_DIR, "rhofold_tm_timing_83156.json")))
    tm_nu = json.load(open(os.path.join(F.NU_DIR, "nufold_tm_timing_531888.json")))
    # RFoldSearch-TM is timed at full scale only, so it appears on the ladder only
    rho["RFoldSearch-TM"] = {F.FULL: (tm_rho["mean_s"], tm_rho["median_s"])}
    nu["RFoldSearch-TM"] = {F.NU_FULL: (tm_nu["mean_s"], tm_nu["median_s"])}

    # timed on one corpus or in one mode only, so not part of the shared loaders
    def series(d, pattern, sizes):
        out = {}
        for n in sizes:
            r = json.load(open(os.path.join(d, pattern.format(n=n))))
            out[n] = (r["mean_s"], r["median_s"])
        return out
    rho["Infernal"] = series(F.RHO_DIR, "infernal_{n}.json", F.SIZES)
    rho["Infernal-fast"] = series(F.RHO_DIR, "infernal_fast_{n}.json", F.SIZES)
    rho["RiboSeek-RS20"] = series(F.RHO_DIR, "riboseek_rs20_{n}.json", F.SIZES)
    nu["RiboSeek-RS20"] = series(F.NU_DIR, "nufold_riboseek_rs20_{n}.json", F.NU_SIZES)

    sub = cell.subgridspec(1, 2, wspace=wspace)
    a = fig.add_subplot(sub[0]); b = fig.add_subplot(sub[1], sharey=a)
    ylim, ladder_ylim = (2.8e-5, 60), (5e-2, 1.2e4)       # shared by both panels
    panel(a, rho, F.SIZES, ylim,
          ("RMalign", "US-align", "RTM-align", "Infernal", "Infernal-fast", "GTalign",
           "GTalign-fast", "RiboSeek-RS20", "RiboSeek", "BLASTN", "RFoldSearch"),
          ladder_ylim, "RhoFold+ database")
    panel(b, nu, F.NU_SIZES, ylim,
          ("GTalign", "RiboSeek-RS20", "RiboSeek", "BLASTN", "RFoldSearch"),
          ladder_ylim, "NuFold database")
    a.set_ylabel("Search time per\n1,000 targets (s)", fontsize=7.2 * P.K)
    b.tick_params(labelleft=False)
    return a, b


def main():
    out_png = sys.argv[1] if len(sys.argv) > 1 else None
    setup_styles()
    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM))
    gs = fig.add_gridspec(1, 1, left=0.085, right=0.99, top=0.93, bottom=0.215)
    a, b = speed_row(fig, gs[0])
    for ax, lab, x in ((a, "A", -0.15), (b, "B", -0.04)):
        ax.text(x, 1.03, lab, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="baseline", ha="left", color=P.INK)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=6, frameon=False,
               fontsize=6.0, handlelength=2.2, columnspacing=1.6,
               bbox_to_anchor=(0.53, 0.0))

    out = out_png or os.path.join(F.FIGDIR, "fig4.png")
    fig.savefig(out, dpi=600)
    print("wrote", out)


if __name__ == "__main__":
    main()
