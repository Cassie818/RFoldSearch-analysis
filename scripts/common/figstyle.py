"""Plot conventions shared with Fig. 2 (scripts/02_rnasolo_retrieval/plot_fig2.py).

Everything here is taken from that script so a method keeps its colour, its
marker and its typography across the paper: Arial, no grid, only the left and
bottom spines, bold panel letters outside the axes, a centred panel title, and
one shared legend at the foot of the figure. Methods that do not appear in
Fig. 2 (nhmmer, and RFoldseek at a deeper candidate pool) take a tone and a
marker that Fig. 2 leaves unused.
"""
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
import matplotlib.pyplot as plt
import numpy as np

# ---- colour and marker, as in Fig. 2 ----------------------------------------
BASE = {"RFoldseek n500": "#1F9BD8",      # Fig. 2 RFoldseek
        "RFoldseek-TM n500": "#009E9E",   # Fig. 2 RFoldseek-TM
        "RFoldseek-TM n200": "#009E9E",
        "RiboSeek c500": "#E377C2",       # Fig. 2 RiboSeek
        "RiboSeek c200": "#E377C2",
        "RiboSeek RS-80 c500": "#A8327F",  # RiboSeek 0.2.0 default RS-80: same hue, deeper shade
        "BLASTN": "#FFC107",              # Fig. 2 BLASTN
        "nhmmer": "#8C564B",              # not in Fig. 2: brown, absent from the Fig. 2 palette
        "Infernal": "#BCBD22",            # not in Fig. 2 or elsewhere: olive (unused tab10 slot)
        "US-align (exhaustive)": "#D62728",   # Fig. 2 US-align
        "RMalign (exhaustive)": "#2CA02C"}    # Fig. 2 RMalign
MARKER = {"RFoldseek n500": "o",
          "RFoldseek-TM n500": "*", "RFoldseek-TM n200": "*",
          "RiboSeek c500": "v", "RiboSeek c200": "v", "RiboSeek RS-80 c500": "v", "BLASTN": "D", "nhmmer": "h", "Infernal": "p",
          "US-align (exhaustive)": "s", "RMalign (exhaustive)": "^"}
LINESTYLE = {m: "-" for m in BASE}
LINESTYLE["BLASTN"] = "--"
OPEN = set()                              # lines: white face; bars: hatched
HATCH = {}                                # only where two depths share a panel
MSIZE = {"RFoldseek-TM n500": 11, "RFoldseek-TM n200": 11}         # the star needs to be bigger, as in Fig. 2
SHORT = {"RFoldseek n500": "RFoldseek", "RFoldseek-TM n500": "RFoldseek-TM",
         "RiboSeek c500": "RiboSeek (RS-20)", "RiboSeek RS-80 c500": "RiboSeek (RS-80)",
         "BLASTN": "BLASTN", "nhmmer": "nhmmer",
         "US-align (exhaustive)": "US-align", "RMalign (exhaustive)": "RMalign", "Infernal": "Infernal",
         # other candidate depths, named only where the depth itself is the point
         "RFoldseek-TM n200": "RFoldseek-TM (depth 200)",
         "RiboSeek c200": "RiboSeek (RS-20, depth 200)"}

# Query sets are never encoded by hue (hue is the method); they are encoded by
# fill, and where a panel is about the query sets themselves Fig. 2's two greys
# are used, so that no query set can be mistaken for a method.
NEUTRAL_DARK, NEUTRAL_LIGHT = "#555555", "#999999"
GREY_LINE, GREY_TEXT = "#999999", "#555555"


def lighten(h, f=0.24):
    """Blend a hex colour toward white by fraction f, as in Fig. 2."""
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02x%02x%02x" % tuple(round(c + (255 - c) * f) for c in (r, g, b))


COLOR = {m: lighten(c) for m, c in BASE.items()}   # lines, fills
TEXT = dict(BASE)                                   # annotations use the full tone


def style_axes(ax, title=None, xlabel=None, ylabel=None):
    """Fig. 2's axes: centred title, 12 pt labels, 10 pt ticks, no grid."""
    if title:
        ax.set_title(title, fontsize=13, pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)


def panel_letters(axes, labels, x=-0.09, y=1.03):
    for ax, lab in zip(axes, labels):
        ax.text(x, y, lab, transform=ax.transAxes, fontsize=15, fontweight="bold",
                va="bottom", ha="right")


def line(ax, x, y, method, yerr=None, label=None, **kw):
    """One method's series, in Fig. 2's line idiom."""
    face = "white" if method in OPEN else COLOR[method]
    common = dict(color=COLOR[method], linewidth=2.4, marker=MARKER[method],
                  markersize=MSIZE.get(method, 7), markerfacecolor=face,
                  markeredgecolor="white" if method not in OPEN else COLOR[method],
                  markeredgewidth=0.9, zorder=3,
                  label=SHORT[method] if label is None else label)
    common.update(kw)
    if yerr is None:
        return ax.plot(x, y, LINESTYLE[method], **common)
    common.pop("linewidth")
    return ax.errorbar(x, y, yerr=yerr, fmt=LINESTYLE[method], elinewidth=0.9,
                       capsize=2.5, ecolor=COLOR[method], linewidth=2.4, **common)


def bars(ax, x, vals, width, method, yerr=None, label=None):
    """One method's bar group, in Fig. 2's bar idiom."""
    return ax.bar(x, vals, width, color=COLOR[method], edgecolor="#666666",
                  linewidth=0.6, zorder=3, yerr=yerr, capsize=2.5,
                  hatch=HATCH.get(method), error_kw={"elinewidth": 0.9, "ecolor": GREY_TEXT},
                  label=SHORT[method] if label is None else label)


def ceiling(ax, value, x0, x1, oracle="US-align (exhaustive)"):
    """Fig. 2's dashed oracle line with its annotation, in the oracle's colour."""
    ax.plot([x0, x1], [value, value], ls="--", color=COLOR[oracle], lw=1.5, zorder=2)
    ax.text(x0 + 0.06, value + 0.012, f"{SHORT[oracle]} (oracle ceiling)", ha="left", va="bottom",
            fontsize=8.5, color=TEXT[oracle])


def bar_handle(method):
    """Legend swatch matching a bar group (hue plus hatch)."""
    from matplotlib.patches import Patch
    return Patch(facecolor=COLOR[method], edgecolor="#666666", linewidth=0.6,
                 hatch=HATCH.get(method), label=SHORT[method])


def oracle_handle(oracle="US-align (exhaustive)"):
    return plt.Line2D([], [], color=COLOR[oracle], lw=2.0, ls="--", label=f"{SHORT[oracle]} (oracle ceiling)")


def shared_legend(fig, methods, ncol=4, y=-0.004, extra=(), kind="line"):
    """One legend under the whole figure, as in Fig. 2."""
    if kind == "bar":
        handles = [bar_handle(m) for m in methods]
    else:
        handles = [plt.Line2D([], [], color=COLOR[m], lw=3.0, ls=LINESTYLE[m],
                              marker=MARKER[m], markersize=MSIZE.get(m, 7) + 1,
                              markerfacecolor="white" if m in OPEN else COLOR[m],
                              markeredgecolor="white" if m not in OPEN else COLOR[m],
                              markeredgewidth=0.9, label=SHORT[m]) for m in methods]
    handles += list(extra)
    legend = fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False,
                        fontsize=12, bbox_to_anchor=(0.5, y), handlelength=2.8,
                        columnspacing=2.4, markerscale=1.45)
    for h in legend.legend_handles:
        if hasattr(h, "set_linewidth") and not hasattr(h, "get_hatch"):
            h.set_linewidth(3.0)
    return legend


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)
