"""Shared plotting style for the runtime figures. """
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import runtime_common as F

MM = F.MM
WIDTH_MM, HEIGHT_MM = 180.0, 104.0
FIGDIR = F.FIGDIR

# Fig. 2 method colours and markers, shared through runtime_common.
COLOR = {m: F.COLOR[m] for m in ("RFoldSearch", "GTalign", "GTalign-fast", "RiboSeek",
                                 "RiboSeek-RS20", "BLASTN", "Infernal", "Infernal-fast",
                                 "US-align", "RTM-align", "RMalign")}
MARKER = {m: F.MARKER[m] for m in COLOR}
LABEL = {"RFoldSearch": "RFoldSearch", "GTalign": "GTalign (GPU)", "GTalign-fast": "GTalign (GPU)-fast",
         "RiboSeek": "RiboSeek (RS-80)", "BLASTN": "BLASTN"}
CURVES = ["GTalign", "RiboSeek", "BLASTN", "RFoldSearch"]
BAND = "#8A8A8A"
INK, MUTED = "#1A1A1A", "#6B6B6B"
K = 1.0   # size scale for marks and text; raised when the panels are drawn into Fig. 5

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
    if m == "RFoldSearch":
        return 1.9 * K, 3.2 * K, 6
    return 1.1 * K, 2.4 * K, 4


def draw_shape(ax, data, order, stat, sizes, ylim, iqr=None, coord=None,
               label_gap=0.055):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(sizes[0] * 0.86, sizes[-1] * 1.06)
    ax.set_ylim(*ylim)

    for y in (1.0, 0.1, 0.01, 0.001):
        if ylim[0] <= y <= ylim[1]:
            ax.axhline(y, color="#E6E6E6", lw=0.45, zorder=0)

    # the coordinate aligners, as one band: their normalised curves differ little,
    # and none of them appears in the other curves' legend
    if coord:
        ns = np.array(sorted(coord[0]))
        stack = np.array([relative(c, stat)[1] for c in coord])
        ax.fill_between(ns, stack.min(axis=0), stack.max(axis=0),
                        color=BAND, alpha=0.45, lw=0, zorder=2)
        ax.plot(ns, stack.mean(axis=0), "-", color=BAND, lw=1.0, zorder=2)
        ax.text(sizes[0] * 1.25, stack.max(axis=0)[0] * 1.25, "Coordinate aligners",
                color="#5F5F5F", fontsize=6.0, va="bottom", ha="left")

    # where the coordinate-aligner band already sits on 1.0, a dashed rule
    # underneath it is one line too many
    if not coord:
        ax.axhline(1.0, color=MUTED, lw=0.6, dashes=(1.3, 1.5), zorder=1)

    if iqr and "RFoldSearch" in iqr:
        ns = np.array(sorted(iqr["RFoldSearch"]))
        lo = np.array([iqr["RFoldSearch"][n][0] for n in ns]) / ns * 1000
        hi = np.array([iqr["RFoldSearch"][n][1] for n in ns]) / ns * 1000
        ax.fill_between(ns, lo / lo[0], hi / hi[0], color=COLOR["RFoldSearch"],
                        alpha=0.13, lw=0, zorder=3)

    ends = {}
    for m in order:
        xs, ys = relative(data[m], stat)
        lw, ms, z = curve_style(m)
        ax.plot(xs, ys, "--" if m == "BLASTN" else "-", color=COLOR[m], lw=lw,
                marker=MARKER[m], ms=ms, mec="white", mew=0.4, zorder=z)
        ends[m] = ys[-1]

    span = np.log10(ylim[1]) - np.log10(ylim[0])
    names = sorted(ends, key=lambda m: ends[m])
    placed = F.spread_labels([np.log10(ends[m]) for m in names],
                             min_gap=label_gap * span)
    for m, ly in zip(names, placed):
        ax.text(sizes[-1] * 1.14, 10 ** ly, LABEL[m], color=COLOR[m],
                fontsize=6.0, va="center", ha="left",
                fontweight="bold" if m == "RFoldSearch" else "normal",
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

    if iqr and "RFoldSearch" in iqr:
        ns = np.array(sorted(iqr["RFoldSearch"]))
        lo = np.array([iqr["RFoldSearch"][n][0] for n in ns]) / ns * 1000
        hi = np.array([iqr["RFoldSearch"][n][1] for n in ns]) / ns * 1000
        ax.fill_between(ns, lo, hi, color=COLOR["RFoldSearch"], alpha=0.13,
                        lw=0, zorder=3)

    ends = {}
    for m in order:
        xs, ys = per_thousand(data[m], stat)
        lw, ms, z = curve_style(m)
        ax.plot(xs, ys, "--" if m == "BLASTN" else "-", color=COLOR[m], lw=lw,
                marker=MARKER[m], ms=ms, mec="white", mew=0.4, zorder=z)
        ends[m] = ys[-1]

    if labels:
        span = np.log10(ylim[1]) - np.log10(ylim[0])
        names = sorted(ends, key=lambda m: ends[m])
        placed = F.spread_labels([np.log10(ends[m]) for m in names],
                                 min_gap=0.055 * span)
        for m, ly in zip(names, placed):
            ax.text(sizes[-1] * 1.14, 10 ** ly, LABEL[m], color=COLOR[m],
                    fontsize=6.0, va="center", ha="left",
                    fontweight="bold" if m == "RFoldSearch" else "normal",
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



LADDER_NAME = {"RFoldSearch": "RFoldSearch", "GTalign": "GTalign (GPU)",
               "RiboSeek": "RiboSeek", "BLASTN": "BLASTN",
               "RMalign": "RMalign", "US-align": "US-align",
               "RTM-align": "RTM-align",
               "US-align-fast": "US-align \u2212fast",
               "GTalign-fast": "GTalign (GPU)-fast"}


def ladder(ax, values, title, ylim, footer, ref="RFoldSearch",
           rail=0.26, ratio_x=0.60, name_x=0.655, step=0.075):
    """Riboseek's search-time rail, but each rung names its method.

    With the legend gone, a marker on the rail is otherwise unidentifiable:
    the curve panels teach four colours, and the three coordinate aligners
    appear nowhere else.
    """
    ref_val = values[ref]
    lo, hi = ylim
    ax.set_yscale("log"); ax.set_xlim(0, 1); ax.set_ylim(lo, hi)
    ax.plot([rail, rail], [lo, hi], color=INK, lw=0.7 * K, zorder=2,
            solid_capstyle="butt")

    for v, lab in F.TIME_TICKS:
        if lo <= v <= hi:
            ax.plot([rail - 0.055, rail], [v, v], color=INK, lw=0.5 * K, zorder=3)
            ax.text(rail - 0.085, v, lab, ha="right", va="center", fontsize=5.9 * K,
                    color=MUTED)

    methods = sorted(values, key=lambda m: values[m])
    span = np.log10(hi) - np.log10(lo)
    label_y = F.spread_labels([np.log10(values[m]) for m in methods],
                              min_gap=0.058 * span)
    log_v = np.log10([values[m] for m in methods])
    mx = np.full(len(methods), rail)
    for i in range(1, len(methods)):
        if log_v[i] - log_v[i - 1] < 0.030 * span:
            mx[i] = mx[i - 1] + step

    for m, ly, x in zip(methods, label_y, mx):
        v, c = values[m], F.COLOR[m]
        extra = m in F.EXTRAPOLATED
        ax.plot([x], [v], marker=F.MARKER[m], ms=(6.0 if F.MARKER[m] == "*" else 4.2) * K,
                mfc="white" if extra else c, mec=c, mew=(0.9 if extra else 0.5) * K,
                ls="none", zorder=6, clip_on=False)
        txt = (F.fmt_abs(v) if m == ref else F.fmt_ratio(v / ref_val)) + \
              ("†" if extra else "")
        # without names the value sits left-aligned, joined to its marker
        ax.text(ratio_x, 10 ** ly, txt, ha="right" if name_x else "left",
                va="center", fontsize=6.3 * K,
                color=F.TEXT_COLOR.get(m, c), fontweight="bold" if m == ref else "normal")
        if name_x is not None:
            ax.text(name_x, 10 ** ly, LADDER_NAME[m], ha="left", va="center",
                    fontsize=5.2 * K, color=MUTED)
            if abs(np.log10(v) - ly) > 0.012 * span:
                ax.plot([x + 0.02, x + 0.05], [v, 10 ** ly], color=c, lw=0.4,
                        alpha=0.55, zorder=4, clip_on=False)
        else:
            ax.plot([x + 0.05, ratio_x - 0.03], [v, 10 ** ly], color=c, lw=0.4 * K,
                    alpha=0.45, zorder=4, clip_on=False)

    ax.set_title(title, pad=6.5 * K, fontsize=6.3 * K, color=MUTED)
    ax.text(0.5, -0.035, footer, transform=ax.transAxes, ha="center", va="top",
            fontsize=5.9 * K, color=MUTED)
    ax.set_xticks([]); ax.set_yticks([]); ax.minorticks_off()
    for sp in ax.spines.values():
        sp.set_visible(False)
