"""Shared data loading and constants for the runtime figures (Fig 3, S14, S15,
dbstats, riboseek_speed). Originally the ladder variant of Fig 3:

Fig. 3 with a Riboseek-style search-time ladder (Park et al. 2026, Fig. 2b/g).

Same data, same methods and the same four normalised scaling panels as
plot_fig3.py; what is added is a third column that answers the question the
normalised view deliberately suppresses -- what does one query actually cost at
full database size, and by what factor do the baselines differ.

The ladder follows the Riboseek idiom: a single logarithmic time axis, one
marker per method, the reference method annotated with its absolute time and
everything else annotated as a multiple of it. Ratios are mean per-query time,
matching the ratios quoted in the Results text; the median column is the
per-corpus median and is reported in the caption instead, because RFoldseek's
median is small enough that the coordinate-aligner ratios there run to five
figures and stop being readable as a comparison.

Coordinate aligners were measured to 50,000 targets only and least-squares
extrapolated to 83,156, so they are drawn with open markers and their labels
carry a dagger, exactly as the dashed segments do in panels A and B.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FIGDIR = os.path.join(REPO, "figures")
DATA_DIR = os.path.join(REPO, "files", "predicted_runtime")
RHO_DIR = os.path.join(DATA_DIR, "rhofold")   # RhoFold+ database timings
NU_DIR = os.path.join(DATA_DIR, "nufold")     # NuFold database timings

MM = 1 / 25.4
WIDTH_MM, HEIGHT_MM = 180.0, 118.0

SIZES = [1000, 5000, 10000, 20000, 50000, 83156]
FULL = 83156
NU_SIZES = [1000, 5000, 10000, 20000, 50000, 100000, 200000, 531888]
NU_FULL = 531888

COLOR = {"RFoldseek": "#0072B2", "GTalign": "#7E5AB6", "RTM-align": "#E69F00",
         "US-align": "#D55E00", "RMalign": "#009E73", "RiboSeek": "#CC79A7",
         "BLASTN": "#B58900"}
MARKER = {"RFoldseek": "o", "GTalign": "X", "RTM-align": "P", "US-align": "s",
          "RMalign": "^", "RiboSeek": "v", "BLASTN": "D"}
COLOR["US-align-fast"] = COLOR["US-align"]
COLOR["GTalign-fast"] = COLOR["GTalign"]
MARKER["US-align-fast"] = "s"
MARKER["GTalign-fast"] = "X"
LINES_RHO = ["RMalign", "US-align", "RTM-align", "GTalign", "GTalign-fast",
             "RiboSeek", "BLASTN", "RFoldseek"]
LINES_NU = ["GTalign", "GTalign-fast", "RiboSeek", "BLASTN", "RFoldseek"]
LEGEND_ORDER = ["RFoldseek", "GTalign", "RTM-align", "US-align",
                "RMalign", "RiboSeek", "BLASTN"]
LABEL = {m: (m + " (GPU)" if m == "GTalign" else m) for m in LEGEND_ORDER}
LABEL["US-align-fast"] = "US-align $-$fast"
LABEL["GTalign-fast"] = "GTalign (GPU)-fast"
BENCH_KEY = {"GTalign": "gtalign", "US-align": "usalign", "RMalign": "rmalign",
             "RTM-align": "rtmalign", "BLASTN": "blastn"}
# Fast presets are benchmarked as methods in their own right, as Foldseek's
# paper does, rather than mentioned only in Methods. US-align -fast is timed
# on the same 20-query set as its default mode; GTalign --speed=13 on the
# same 300-query NuFold ladder as --speed=8.
RHO_FILE_EXTRA = {"GTalign-fast": "gtalign_fast_{n}.json"}
NU_FILE = {"RFoldseek": "nufold_timing_{n}.json", "GTalign": "nufold_gtalign_{n}.json",
           "GTalign-fast": "nufold_gtalign13_{n}.json",
           "RiboSeek": "nufold_riboseek_{n}.json", "BLASTN": "nufold_blastn_{n}.json"}
EXTRAPOLATED = set()  # RhoFold+: coordinate aligners direct (2080), GTalign-fast measured at all sizes
COORDINATE_ALIGNERS = {"US-align", "US-align-fast", "RTM-align", "RMalign"}
REF = "RFoldseek"

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.2, "axes.labelsize": 7.2, "axes.titlesize": 7.6,
    "xtick.labelsize": 6.2, "ytick.labelsize": 6.2,
    "axes.linewidth": 0.55, "xtick.major.width": 0.55, "ytick.major.width": 0.55,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def quartiles(per_query):
    q = np.sort(np.array([x["s"] for x in per_query], dtype=float))
    return float(np.percentile(q, 25)), float(np.percentile(q, 75))


def load_rhofold(iqr=None):
    iqr = {} if iqr is None else iqr
    bench = {n: json.load(open(os.path.join(RHO_DIR, f"bench_{n}.json"))) for n in SIZES}
    _coord2080 = json.load(open(os.path.join(RHO_DIR, "coord2080_paired.json")))["sizes"]  # US/RTM/RMalign re-timed on 2080, 300q, scaled to reference RFoldseek baseline (paired ratio)
    out = {}
    for m in LINES_RHO:
        pts = {}
        for n in SIZES:
            if m == "RiboSeek":
                f = os.path.join(RHO_DIR, f"riboseek_{n}.json")
                if os.path.exists(f):
                    r = json.load(open(f))
                    pts[n] = (r["mean_s"], r["median_s"])
            elif m == "RFoldseek":
                r = json.load(open(os.path.join(RHO_DIR, f"kernel_timing_{n}.json")))
                assert r["top_n"] == 500 and r["n_database"] == n  # benchmark depth: 500, matching the competitors
                pts[n] = (r["mean_s"], r["median_s"])
                iqr.setdefault("RFoldseek", {})[n] = quartiles(r["per_query"])
            elif m in RHO_FILE_EXTRA:
                f = os.path.join(RHO_DIR, RHO_FILE_EXTRA[m].format(n=n))
                if os.path.exists(f):
                    r = json.load(open(f))
                    pts[n] = (r["mean_s"], r["median_s"])
            elif m in ("US-align", "RTM-align", "RMalign"):
                cd = _coord2080.get(str(n), {}).get(BENCH_KEY[m])
                if cd is not None:
                    pts[n] = (cd["mean_s"], cd["median_s"])
            else:
                v = bench[n]["methods"].get(BENCH_KEY[m])
                if v is not None:
                    pts[n] = (v["mean_s"], v["median_s"])
        if m in EXTRAPOLATED:
            measured = sorted(pts)
            xs = np.array(measured, dtype=float)
            fitted = []
            for stat in (0, 1):
                b, a = np.polyfit(xs, np.array([pts[n][stat] for n in measured]), 1)
                fitted.append(a + b * FULL)
            pts[FULL] = tuple(fitted)
        out[m] = pts
    return out, iqr


def load_nufold():
    out, iqr = {}, {}
    for m, pat in NU_FILE.items():
        pts = {}
        for n in NU_SIZES:
            f = os.path.join(NU_DIR, pat.format(n=n))
            if os.path.exists(f):
                r = json.load(open(f))
                pts[n] = (r["mean_s"], r["median_s"])
                if m == "RFoldseek":
                    iqr.setdefault(m, {})[n] = quartiles(r["per_query"])
        out[m] = pts
    return out, iqr


def db_tick_label(n):
    if n >= 100000:
        return f"{int(round(n / 1000)):,}k"
    return f"{int(n / 1000):,}k"


def style(method):
    lw, ms, z, alpha = 1.15, 2.5, 3, 0.88
    if method in COORDINATE_ALIGNERS:
        alpha = 0.72
    if method == "RFoldseek":
        lw, ms, z, alpha = 1.8, 3.1, 5, 1.0
    if method == "RTM-align":
        lw, ms, z, alpha = 0.8, 2.25, 4, 0.72
    return lw, ms, z, alpha


def scale_values(xs, ys, per_thousand):
    return ys / xs * 1000 if per_thousand else ys


# US-align and RTM-align agree to within 1.1-3.7% at every database size, which
# on a four-decade axis is narrower than a line width. Dashing one of them is
# what keeps seven methods countable instead of six.
DASHES = {"US-align": (4.6, 2.2)}


def plot_series(ax, data, order, stat, extrapolate=False, iqr=None,
                per_thousand=True):
    """Curves only; axis limits, ticks and spines are the caller's business."""
    if iqr and "RFoldseek" in iqr:
        ns = np.array(sorted(iqr["RFoldseek"]))
        lo = np.array([iqr["RFoldseek"][n][0] for n in ns])
        hi = np.array([iqr["RFoldseek"][n][1] for n in ns])
        lo = scale_values(ns, lo, per_thousand)
        hi = scale_values(ns, hi, per_thousand)
        ax.fill_between(ns, lo, hi, color=COLOR["RFoldseek"], alpha=0.15, lw=0, zorder=1)
    for m in order:
        pts = data[m]
        if not pts:
            continue
        xs = np.array(sorted(pts))
        ys = scale_values(xs, np.array([pts[n][stat] for n in xs]), per_thousand)
        c, mk = COLOR[m], MARKER[m]
        lw, ms, z, alpha = style(m)
        dash = DASHES.get(m)
        if extrapolate and m in EXTRAPOLATED:
            k = len(xs) - 1       # every point but the fitted one is measured
            solid = dict(dashes=dash) if dash else dict()
            ax.plot(xs[:k], ys[:k], "-", color=c, lw=lw, alpha=alpha,
                    marker=mk, ms=ms, mec="white", mew=0.35, zorder=z, **solid)
            ax.plot(xs[k - 1:], ys[k - 1:], "-", color=c, lw=lw * 0.9,
                    dashes=(2.0, 1.3), alpha=alpha, zorder=z)
            ax.plot(xs[k:], ys[k:], ls="none", marker=mk, ms=ms,
                    mfc="white", mec=c, mew=0.7, zorder=z)
        else:
            kw = dict(dashes=dash) if dash else dict()
            ax.plot(xs, ys, "-", color=c, lw=lw, alpha=alpha, marker=mk, ms=ms,
                    mec="white", mew=0.35, zorder=z, **kw)


def finish_axes(ax, sizes, ylim, xticks=True):
    ax.set_xscale("log"); ax.set_yscale("log")
    decades = [n for n in (1000, 10000, 100000) if n < sizes[-1] * 0.8]
    ticks = decades + [sizes[-1]]
    ax.set_xticks(ticks)
    if xticks:
        ax.set_xticklabels([db_tick_label(d) for d in ticks])
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
    ax.set_xlim(sizes[0] * 0.82, sizes[-1] * 1.30)
    ax.set_ylim(*ylim)
    ax.minorticks_off()
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=2.2, pad=1.6)


def draw(ax, data, order, stat, sizes, ylim, extrapolate=False, iqr=None,
         per_thousand=True):
    plot_series(ax, data, order, stat, extrapolate, iqr, per_thousand)
    finish_axes(ax, sizes, ylim)


def draw_broken(ax_top, ax_bot, data, order, stat, sizes, ylim_top, ylim_bot,
                extrapolate=False, iqr=None):
    """Split y axis: a compressed band for the strictly linear coordinate
    aligners, the rest of the height for the methods that actually compete.

    The two segments carry the same number of decades per millimetre, so the
    break compresses nothing that is being compared -- it only removes the
    empty decades between 1.5 and 8 s per 1,000 targets, where no method lies.
    """
    for ax, yl in ((ax_top, ylim_top), (ax_bot, ylim_bot)):
        plot_series(ax, data, order, stat, extrapolate,
                    iqr if ax is ax_bot else None)
        finish_axes(ax, sizes, yl, xticks=ax is ax_bot)
    ax_top.spines["bottom"].set_visible(False)
    ax_top.tick_params(axis="x", length=0)
    ax_top.set_yticks([10.0])
    ax_top.set_yticklabels(["10"])

    # break marks on the left spine, where the axis is actually cut
    kw = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=3.6, linestyle="none",
              color="#111111", mec="#111111", mew=0.55, clip_on=False)
    ax_top.plot([0], [0], transform=ax_top.transAxes, **kw)
    ax_bot.plot([0], [1], transform=ax_bot.transAxes, **kw)


# --------------------------------------------------------------------------
# Riboseek-style search-time ladder
# --------------------------------------------------------------------------

TIME_TICKS = [(0.001, "1 ms"), (0.01, "10 ms"), (0.1, "100 ms"), (1.0, "1 s"),
              (10.0, "10 s"), (60.0, "1 min"), (600.0, "10 min"),
              (3600.0, "1 h"), (86400.0, "24 h")]


def fmt_ratio(r):
    if r >= 100:
        return f"×{r:,.0f}"
    if r >= 10:
        return f"×{r:.1f}"
    return f"×{r:.2f}"


def fmt_abs(s):
    if s >= 60:
        return f"{s / 60:.0f} min"
    if s >= 1:
        return f"{s:.2f} s"
    ms = s * 1000
    if ms >= 100:
        return f"{ms:.0f} ms"          # 145 ms
    if ms >= 10:
        return f"{ms:.1f} ms"          # 13.5 ms
    return f"{ms:.2f} ms"              # 0.27 ms -- was rounding to "0 ms"


def spread_labels(y_log, min_gap):
    """Push overlapping label positions apart in log space, order preserved."""
    order = np.argsort(y_log)
    y = np.array(y_log, dtype=float)
    placed = y[order].copy()
    for i in range(1, len(placed)):
        if placed[i] - placed[i - 1] < min_gap:
            placed[i] = placed[i - 1] + min_gap
    out = np.empty_like(y)
    out[order] = placed
    return out


def ladder(ax, values, title, ylim, ref=REF):
    """One marker per method on a log time axis, annotated as x-fold of ref."""
    ref_val = values[ref]
    lo, hi = ylim
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_ylim(lo, hi)

    # the ladder rail
    ax.plot([0.30, 0.30], [lo, hi], color="#111111", lw=0.7, zorder=2,
            solid_capstyle="butt")

    ticks = [(v, lab) for v, lab in TIME_TICKS if lo <= v <= hi]
    for v, lab in ticks:
        ax.plot([0.245, 0.30], [v, v], color="#111111", lw=0.55, zorder=3)
        ax.text(0.215, v, lab, ha="right", va="center", fontsize=6.0,
                color="#111111")

    methods = sorted(values, key=lambda m: values[m])
    span = np.log10(hi) - np.log10(lo)
    label_y = spread_labels([np.log10(values[m]) for m in methods],
                            min_gap=0.070 * span)

    # Methods whose times differ by a few percent land on the same pixel of the
    # rail; step those markers sideways so all of them stay countable.
    log_v = np.log10([values[m] for m in methods])
    mx = np.full(len(methods), 0.30)
    for i in range(1, len(methods)):
        if log_v[i] - log_v[i - 1] < 0.030 * span:
            mx[i] = 0.30 + (mx[i - 1] - 0.30 + 0.075)

    for m, ly, x in zip(methods, label_y, mx):
        v = values[m]
        c, mk = COLOR[m], MARKER[m]
        extra = m in EXTRAPOLATED
        ax.plot([x], [v], marker=mk, ms=4.2, mfc="white" if extra else c,
                mec=c, mew=0.9 if extra else 0.5, ls="none", zorder=6,
                clip_on=False)
        txt = fmt_abs(v) if m == ref else fmt_ratio(v / ref_val)
        if extra:
            txt += "†"
        ax.text(0.52, 10 ** ly, txt, ha="left", va="center", fontsize=6.6,
                color=c, fontweight="bold" if m == ref else "normal")
        # leader line when the label had to be nudged off its marker
        if abs(np.log10(v) - ly) > 0.012 * span:
            ax.plot([x + 0.02, 0.50], [v, 10 ** ly], color=c, lw=0.4,
                    alpha=0.55, zorder=4, clip_on=False)

    ax.set_title(title, pad=4.5, fontsize=6.8, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.minorticks_off()
    for sp in ax.spines.values():
        sp.set_visible(False)


def full_size_values(data, stat, full):
    return {m: pts[full][stat] for m, pts in data.items() if pts and full in pts}
