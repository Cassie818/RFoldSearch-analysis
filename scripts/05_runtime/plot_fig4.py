"""Fig 3 variant: one curve panel per corpus, dual RFoldseek lines."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import runtime_common as F
import runtime_style as P

MM = F.MM
WIDTH_MM, HEIGHT_MM = 180.0, 110.0
CURVES = ["GTalign", "RiboSeek", "BLASTN", "RFoldseek"]


def dual_labels(ax, sizes, ends, ylim):
    span = np.log10(ylim[1]) - np.log10(ylim[0])
    names = sorted(ends, key=lambda m: ends[m])
    placed = F.spread_labels([np.log10(ends[m]) for m in names],
                             min_gap=0.052 * span)
    for m, ly in zip(names, placed):
        base = m.split(" ")[0]
        bold = base == "RFoldseek"
        ax.text(sizes[-1] * 1.14, 10 ** ly, m, color=P.COLOR[base],
                fontsize=6.0, va="center", ha="left",
                fontweight="bold" if bold else "normal", clip_on=False)


def rf_dual(ax, data, iqr, sizes):
    """RFoldseek mean (solid) + median (dashed) + IQR band."""
    if iqr and "RFoldseek" in iqr:
        ns = np.array(sorted(iqr["RFoldseek"]))
        lo = np.array([iqr["RFoldseek"][n][0] for n in ns]) / ns * 1000
        hi = np.array([iqr["RFoldseek"][n][1] for n in ns]) / ns * 1000
        ax.fill_between(ns, lo, hi, color=P.COLOR["RFoldseek"], alpha=0.13,
                        lw=0, zorder=3)
    out = {}
    for stat, style in ((0, "-"), (1, "--")):
        xs, ys = P.per_thousand(data["RFoldseek"], stat)
        ax.plot(xs, ys, style, color=P.COLOR["RFoldseek"], lw=1.9,
                marker="o" if stat == 0 else None, ms=3.2,
                mec="white", mew=0.4, zorder=8)
        out["RFoldseek (mean)" if stat == 0 else "RFoldseek (median)"] = ys[-1]
    return out


def curves_panel(ax_t, ax, data, iqr, sizes, top, bot, coord=None,
                 methods=("GTalign", "RiboSeek", "BLASTN")):
    """Mean curves for competitors, dual RFoldseek, optional broken top."""
    ends = {}
    for m in methods:
        if m not in data:
            continue
        xs, ys = P.per_thousand(data[m], 0)
        lw, ms, z = P.curve_style(m)
        style = "--" if m == "GTalign-fast" else "-"  # fast preset: dashed, same hue
        ax.plot(xs, ys, style, color=P.COLOR[m], lw=lw, marker=P.MARKER[m],
                ms=ms, mec="white", mew=0.4, zorder=z)
        ends[P.LABEL[m]] = ys[-1]
    ends.update(rf_dual(ax, data, iqr, sizes))
    if ax_t is not None and coord:
        P.draw_shape(ax_t, coord)          # gray band, broken top segment
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(sizes[0] * 0.82, sizes[-1] * 1.2)
    ax.set_ylim(*bot)
    for e in range(-4, 2):
        if bot[0] <= 10.0 ** e <= bot[1]:
            ax.axhline(10.0 ** e, color="#E6E6E6", lw=0.45, zorder=0)
    dual_labels(ax, sizes, ends, bot)
    decades = [n for n in (1000, 10000, 100000) if n < sizes[-1] * 0.8]
    ticks = decades + [sizes[-1]]
    ax.set_xticks(ticks)
    ax.set_xticklabels([F.db_tick_label(d) for d in ticks])
    ax.minorticks_off()
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=2.0, pad=1.6)


def main():
    out_png = sys.argv[1] if len(sys.argv) > 1 else None
    rho, rho_iqr = F.load_rhofold()
    nu, nu_iqr = F.load_nufold()

    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.75, 1.0],
                          left=0.075, right=0.985, top=0.910, bottom=0.075,
                          wspace=0.30, hspace=0.42)

    RHO_TOP, RHO_BOT = (7.0, 32.0), (2.5e-4, 1.9)
    hr = [(np.log10(RHO_TOP[1]) - np.log10(RHO_TOP[0]))
          / (np.log10(RHO_BOT[1]) - np.log10(RHO_BOT[0])), 1.0]
    sub_a = gs[0, 0].subgridspec(2, 1, height_ratios=hr, hspace=0.10)
    a_t = fig.add_subplot(sub_a[0]); a = fig.add_subplot(sub_a[1])
    coord = [(m, rho[m]) for m in ("RMalign", "US-align", "RTM-align")]

    # panel A: RhoFold+ curves with broken coordinate-aligner band
    a_t.set_xscale("log"); a_t.set_yscale("log")
    a_t.set_xlim(F.SIZES[0] * 0.82, F.SIZES[-1] * 1.2); a_t.set_ylim(*RHO_TOP)
    ns, rows = None, []
    for m, pts in coord:
        xs, ys = P.per_thousand(pts, 0, True, m)
        ns = xs; rows.append(ys)
    stack = np.array(rows)
    a_t.fill_between(ns, stack.min(axis=0), stack.max(axis=0),
                     color=P.BAND, alpha=0.45, lw=0, zorder=2)
    a_t.plot(ns, stack.mean(axis=0), "-", color=P.BAND, lw=1.0, zorder=2)
    a_t.text(ns[-1] * 1.12, stack.mean(axis=0)[-1], "coordinate\naligners",
             color="#5F5F5F", fontsize=6.0, va="center", ha="left",
             linespacing=1.15, clip_on=False)
    a_t.set_xticks([]); a_t.set_yticks([10])
    a_t.set_yticklabels([r"$10^{1}$"])
    a_t.minorticks_off()
    for sp in ("top", "right", "bottom"):
        a_t.spines[sp].set_visible(False)
    a_t.tick_params(length=2.0, pad=1.6)
    a_t.tick_params(axis="x", length=0)
    kw = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=3.6, linestyle="none",
              color=P.INK, mec=P.INK, mew=0.55, clip_on=False)
    a_t.plot([0], [0], transform=a_t.transAxes, **kw)
    curves_panel(None, a, rho, rho_iqr, F.SIZES, RHO_TOP, RHO_BOT,
                 methods=("GTalign", "GTalign-fast", "RiboSeek", "BLASTN"))
    a.plot([0], [1], transform=a.transAxes, **kw)

    # panel C: NuFold curves, no broken axis
    c = fig.add_subplot(gs[1, 0])
    curves_panel(None, c, nu, nu_iqr, F.NU_SIZES, None, (2.8e-5, 1.5))

    F.COLOR.update(P.COLOR); F.MARKER.update(P.MARKER)
    F.COLOR.update({"RMalign": "#2E2E2E", "US-align": "#545454",
                    "RTM-align": "#7A7A7A"})
    # a fast preset is the same tool at a different operating point, so it
    # keeps its parent's colour and is distinguished by the hollow marker
    F.COLOR["US-align-fast"] = F.COLOR["US-align"]
    F.COLOR["GTalign-fast"] = P.COLOR["GTalign"]
    # RFoldseek-TM: same prefilter, US-align rerank of the candidate pool.
    # Measured at full scale only (timing protocol, 300 queries, top_n 500).
    import json as _json
    _tmr = _json.load(open(os.path.join(F.RHO_DIR,
                                        "rhofold_tm_timing_83156.json")))
    _tmn = _json.load(open(os.path.join(F.NU_DIR,
                                        "nufold_tm_timing_531888.json")))
    rho["RFoldseek-TM"] = {F.FULL: (_tmr["mean_s"], _tmr["median_s"])}
    nu["RFoldseek-TM"] = {F.NU_FULL: (_tmn["mean_s"], _tmn["median_s"])}
    for m in (P.COLOR, F.COLOR):
        m["RFoldseek-TM"] = P.COLOR["RFoldseek"]
    for m in (P.MARKER, F.MARKER):
        m["RFoldseek-TM"] = "s"
    P.LADDER_NAME["RFoldseek-TM"] = "RFoldseek-TM"
    b = fig.add_subplot(gs[0, 1]); d = fig.add_subplot(gs[1, 1])
    P.ladder(b, F.full_size_values(rho, 0, F.FULL), "Search time",
             (5e-2, 1.2e4), "per query, at 83,156 targets")
    P.ladder(d, F.full_size_values(nu, 0, F.NU_FULL), "Search time",
             (7e-2, 6e1), "per query, at 531,888 targets")

    for ax in (a, c):
        ax.set_ylabel("Search time per\n1,000 targets (s)")
        ax.set_xlabel("Database size (structures)")
        ax.text(0.035, 0.055, "shaded: RFoldseek interquartile range",
                transform=ax.transAxes, fontsize=5.5,
                color=P.COLOR["RFoldseek"], alpha=0.75, va="bottom")
    a_t.text(0.0, 1.28, "RhoFold+ database (83,156 structures)",
             transform=a_t.transAxes, fontsize=6.4, color=P.INK, va="baseline")
    c.text(0.0, 1.045, "NuFold database (531,888 structures)",
           transform=c.transAxes, fontsize=6.4, color=P.INK, va="baseline")
    for ax, lab, ly in ((a_t, "A", 1.28), (b, "B", 1.13),
                        (c, "C", 1.045), (d, "D", 1.13)):
        ax.text(-0.105 if ax in (a_t, c) else -0.12, ly, lab,
                transform=ax.transAxes, fontsize=8, fontweight="bold",
                va="baseline", ha="left", color=P.INK)

    fig.align_ylabels([a, c])
    if out_png:                      # preview mode
        fig.savefig(out_png, dpi=600)
        print("wrote", out_png)
        return
    for ext in ("png",):       # canonical Fig 3, as included by the tex
        out = os.path.join(F.FIGDIR, f"fig4.{ext}")
        fig.savefig(out, dpi=600 if ext == "png" else None)
        print("wrote", out)


if __name__ == "__main__":
    main()
