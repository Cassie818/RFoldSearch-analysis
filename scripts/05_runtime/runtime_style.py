"""Shared plotting style for the runtime figures. Originally:

Fig. 4: what a search costs, and how that cost scales.

Cost per target -- per-query time divided by database size -- so a horizontal
line is cost strictly proportional to database size and a falling line is
sublinear, while the vertical position of each curve stays its actual cost.

That last clause is the whole reason this view is the main figure. An earlier
draft normalised each curve by its own value at 1,000 structures, which shows
scaling shape cleanly but ranks methods by how far they fell rather than by
what they cost: BLASTN starts 39x more expensive than RFoldseek on NuFold, so
it falls 336x against RFoldseek's 179x and lands below it on the page, while
in fact costing 20.6x more per target at full scale. The normalised view is
kept as a supplementary figure (plot_supp_runtime_scaling.py), where the ranking cannot be misread as the
headline.

The RhoFold+ panels break the axis beneath the coordinate-aligner band: the
band sits at 8.4-26.7 s per 1,000 targets and the next method up, GTalign,
reaches 1.5, so three quarters of a decade in between holds nothing. Both
segments carry the same decades per unit height.

Panels C and F are the Riboseek search-time rail (Park et al. 2026, Fig. 2b/g)
anchored to whole-query time rather than cost per target, so they complement
the curves instead of restating them.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import runtime_common as F

MM = F.MM
WIDTH_MM, HEIGHT_MM = 180.0, 104.0
FIGDIR = F.FIGDIR

# Okabe-Ito, four hues, assigned in fixed order and never cycled.
COLOR = {"RFoldseek": "#0072B2", "GTalign": "#009E73", "GTalign-fast": "#009E73",
         "RiboSeek": "#CC79A7", "BLASTN": "#E69F00"}
MARKER = {"RFoldseek": "o", "GTalign": "X", "GTalign-fast": "X", "RiboSeek": "v", "BLASTN": "D"}
LABEL = {"RFoldseek": "RFoldseek", "GTalign": "GTalign (GPU)", "GTalign-fast": "GTalign (GPU)-fast",
         "RiboSeek": "RiboSeek", "BLASTN": "BLASTN"}
CURVES = ["GTalign", "RiboSeek", "BLASTN", "RFoldseek"]
BAND = "#8A8A8A"
INK, MUTED = "#1A1A1A", "#6B6B6B"

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.2, "axes.labelsize": 7.2, "axes.titlesize": 7.6,
    "xtick.labelsize": 6.2, "ytick.labelsize": 6.2,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def relative(pts, stat):
    xs = np.array(sorted(pts))
    ys = np.array([pts[n][stat] for n in xs]) / xs * 1000
    return xs, ys / ys[0]


def curve_style(m):
    if m == "RFoldseek":
        return 1.9, 3.2, 6
    return 1.1, 2.4, 4


def draw_shape(ax, data, order, stat, sizes, ylim, iqr=None, coord=None,
               label_gap=0.055):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(sizes[0] * 0.86, sizes[-1] * 1.06)
    ax.set_ylim(*ylim)

    for y in (1.0, 0.1, 0.01, 0.001):
        if ylim[0] <= y <= ylim[1]:
            ax.axhline(y, color="#E6E6E6", lw=0.45, zorder=0)

    # the coordinate aligners, as one band: they differ by <7% at every size.
    # Their extrapolated point at 83,156 is dropped, as in the absolute view --
    # a derived point on a fitted line can only restate the fit.
    if coord:
        ns = np.array([n for n in sorted(coord[0]) if n != F.FULL])
        stack = np.array([relative({n: v for n, v in c.items() if n != F.FULL},
                                   stat)[1] for c in coord])
        ax.fill_between(ns, stack.min(axis=0), stack.max(axis=0),
                        color=BAND, alpha=0.45, lw=0, zorder=2)
        ax.plot(ns, stack.mean(axis=0), "-", color=BAND, lw=1.0, zorder=2)
        ax.text(sizes[-1] * 1.14, stack.mean(axis=0)[-1], "coordinate\naligners",
                color="#5F5F5F", fontsize=6.0, va="center", ha="left",
                linespacing=1.15, clip_on=False)

    # where the coordinate-aligner band already sits on 1.0, a dashed rule
    # underneath it is one line too many
    if not coord:
        ax.axhline(1.0, color=MUTED, lw=0.6, dashes=(1.3, 1.5), zorder=1)

    if iqr and "RFoldseek" in iqr:
        ns = np.array(sorted(iqr["RFoldseek"]))
        lo = np.array([iqr["RFoldseek"][n][0] for n in ns]) / ns * 1000
        hi = np.array([iqr["RFoldseek"][n][1] for n in ns]) / ns * 1000
        ax.fill_between(ns, lo / lo[0], hi / hi[0], color=COLOR["RFoldseek"],
                        alpha=0.13, lw=0, zorder=3)

    ends = {}
    for m in order:
        xs, ys = relative(data[m], stat)
        lw, ms, z = curve_style(m)
        ax.plot(xs, ys, "-", color=COLOR[m], lw=lw, marker=MARKER[m], ms=ms,
                mec="white", mew=0.4, zorder=z)
        ends[m] = ys[-1]

    span = np.log10(ylim[1]) - np.log10(ylim[0])
    names = sorted(ends, key=lambda m: ends[m])
    placed = F.spread_labels([np.log10(ends[m]) for m in names],
                             min_gap=label_gap * span)
    for m, ly in zip(names, placed):
        ax.text(sizes[-1] * 1.14, 10 ** ly, LABEL[m], color=COLOR[m],
                fontsize=6.0, va="center", ha="left",
                fontweight="bold" if m == "RFoldseek" else "normal",
                clip_on=False)

    decades = [n for n in (1000, 10000, 100000) if n < sizes[-1] * 0.8]
    ticks = decades + [sizes[-1]]
    ax.set_xticks(ticks)
    ax.set_xticklabels([F.db_tick_label(d) for d in ticks])
    ax.set_yticks([1.0, 0.1, 0.01, 0.001])
    ax.set_yticklabels(["1", "0.1", "0.01", "0.001"])
    ax.minorticks_off()
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=2.0, pad=1.6)


def per_thousand(pts, stat, drop_extrapolated=False, method=None):
    ns = [n for n in sorted(pts)
          if not (drop_extrapolated and method in F.EXTRAPOLATED and n == F.FULL)]
    xs = np.array(ns, dtype=float)
    return xs, np.array([pts[n][stat] for n in ns]) / xs * 1000



def draw_absolute(ax, data, order, stat, sizes, ylim, coord=None, band_label_y=None,
         iqr=None, labels=True, xticks=True):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(sizes[0] * 0.86, sizes[-1] * 1.06)
    ax.set_ylim(*ylim)

    for e in range(-4, 2):
        if ylim[0] <= 10.0 ** e <= ylim[1]:
            ax.axhline(10.0 ** e, color="#E6E6E6", lw=0.45, zorder=0)

    if coord:
        ns = None
        rows = []
        for m, pts in coord:
            xs, ys = per_thousand(pts, stat, True, m)
            ns, _ = xs, rows.append(ys)
        stack = np.array(rows)
        ax.fill_between(ns, stack.min(axis=0), stack.max(axis=0),
                        color=BAND, alpha=0.45, lw=0, zorder=2)
        ax.plot(ns, stack.mean(axis=0), "-", color=BAND, lw=1.0, zorder=2)
        ax.text(ns[-1] * 1.12, band_label_y or stack.mean(axis=0)[-1],
                "coordinate\naligners", color="#5F5F5F", fontsize=6.0,
                va="center", ha="left", linespacing=1.15, clip_on=False)

    if iqr and "RFoldseek" in iqr:
        ns = np.array(sorted(iqr["RFoldseek"]))
        lo = np.array([iqr["RFoldseek"][n][0] for n in ns]) / ns * 1000
        hi = np.array([iqr["RFoldseek"][n][1] for n in ns]) / ns * 1000
        ax.fill_between(ns, lo, hi, color=COLOR["RFoldseek"], alpha=0.13,
                        lw=0, zorder=3)

    ends = {}
    for m in order:
        xs, ys = per_thousand(data[m], stat)
        lw, ms, z = curve_style(m)
        ax.plot(xs, ys, "-", color=COLOR[m], lw=lw, marker=MARKER[m], ms=ms,
                mec="white", mew=0.4, zorder=z)
        ends[m] = ys[-1]

    if labels:
        span = np.log10(ylim[1]) - np.log10(ylim[0])
        names = sorted(ends, key=lambda m: ends[m])
        placed = F.spread_labels([np.log10(ends[m]) for m in names],
                                 min_gap=0.055 * span)
        for m, ly in zip(names, placed):
            ax.text(sizes[-1] * 1.14, 10 ** ly, LABEL[m], color=COLOR[m],
                    fontsize=6.0, va="center", ha="left",
                    fontweight="bold" if m == "RFoldseek" else "normal",
                    clip_on=False)

    decades = [n for n in (1000, 10000, 100000) if n < sizes[-1] * 0.8]
    ticks = decades + [sizes[-1]]
    ax.set_xticks(ticks)
    if xticks:
        ax.set_xticklabels([F.db_tick_label(d) for d in ticks])
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
    ax.minorticks_off()
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=2.0, pad=1.6)



def draw_broken(ax_top, ax_bot, data, order, stat, sizes, ylim_top, ylim_bot,
                coord=None, iqr=None):
    """Split the axis across the empty decades.

    In this view the coordinate aligners sit at 8.4-26.7 s per 1,000 targets
    while the next method up is GTalign at 1.5, so three quarters of a decade
    in the middle of the panel holds nothing at all. Both segments carry the
    same number of decades per unit height, so nothing being compared is
    compressed relative to anything else.
    """
    draw_absolute(ax_top, data, order, stat, sizes, ylim_top, coord=coord,
         labels=False, xticks=False)
    draw_absolute(ax_bot, data, order, stat, sizes, ylim_bot, iqr=iqr)
    ax_top.spines["bottom"].set_visible(False)
    ax_top.tick_params(axis="x", length=0)   # after draw()'s blanket reset
    ax_top.set_yticks([10.0]); ax_top.set_yticklabels(["$10^{1}$"])
    kw = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=3.6, linestyle="none",
              color=INK, mec=INK, mew=0.55, clip_on=False)
    ax_top.plot([0], [0], transform=ax_top.transAxes, **kw)
    ax_bot.plot([0], [1], transform=ax_bot.transAxes, **kw)



LADDER_NAME = {"RFoldseek": "RFoldseek", "GTalign": "GTalign (GPU)",
               "RiboSeek": "RiboSeek", "BLASTN": "BLASTN",
               "RMalign": "RMalign", "US-align": "US-align",
               "RTM-align": "RTM-align",
               "US-align-fast": "US-align \u2212fast",
               "GTalign-fast": "GTalign (GPU)-fast"}


def ladder(ax, values, title, ylim, footer, ref="RFoldseek"):
    """Riboseek's search-time rail, but each rung names its method.

    With the legend gone, a marker on the rail is otherwise unidentifiable:
    the curve panels teach four colours, and the three coordinate aligners
    appear nowhere else.
    """
    ref_val = values[ref]
    lo, hi = ylim
    ax.set_yscale("log"); ax.set_xlim(0, 1); ax.set_ylim(lo, hi)
    ax.plot([0.26, 0.26], [lo, hi], color=INK, lw=0.7, zorder=2,
            solid_capstyle="butt")

    for v, lab in F.TIME_TICKS:
        if lo <= v <= hi:
            ax.plot([0.205, 0.26], [v, v], color=INK, lw=0.5, zorder=3)
            ax.text(0.175, v, lab, ha="right", va="center", fontsize=5.9,
                    color=MUTED)

    methods = sorted(values, key=lambda m: values[m])
    span = np.log10(hi) - np.log10(lo)
    label_y = F.spread_labels([np.log10(values[m]) for m in methods],
                              min_gap=0.058 * span)
    log_v = np.log10([values[m] for m in methods])
    mx = np.full(len(methods), 0.26)
    for i in range(1, len(methods)):
        if log_v[i] - log_v[i - 1] < 0.030 * span:
            mx[i] = 0.26 + (mx[i - 1] - 0.26 + 0.075)

    for m, ly, x in zip(methods, label_y, mx):
        v, c = values[m], F.COLOR[m]
        extra = m in F.EXTRAPOLATED
        ax.plot([x], [v], marker=F.MARKER[m], ms=4.2,
                mfc="white" if extra else c, mec=c, mew=0.9 if extra else 0.5,
                ls="none", zorder=6, clip_on=False)
        txt = (F.fmt_abs(v) if m == ref else F.fmt_ratio(v / ref_val)) + \
              ("†" if extra else "")
        ax.text(0.60, 10 ** ly, txt, ha="right", va="center", fontsize=6.3,
                color=c, fontweight="bold" if m == ref else "normal")
        ax.text(0.655, 10 ** ly, LADDER_NAME[m], ha="left", va="center",
                fontsize=5.2, color=MUTED)
        if abs(np.log10(v) - ly) > 0.012 * span:
            ax.plot([x + 0.02, 0.30], [v, 10 ** ly], color=c, lw=0.4,
                    alpha=0.55, zorder=4, clip_on=False)

    ax.set_title(title, pad=6.5, fontsize=6.8, fontweight="bold", color=INK)
    ax.text(0.5, -0.035, footer, transform=ax.transAxes, ha="center", va="top",
            fontsize=5.9, color=MUTED)
    ax.set_xticks([]); ax.set_yticks([]); ax.minorticks_off()
    for sp in ax.spines.values():
        sp.set_visible(False)
