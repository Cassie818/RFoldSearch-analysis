"""Panels and data for the sequence-divergent homologues on the 885-chain RNAsolo benchmark."""

import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "common"))
import figstyle as S
from figstyle import plt

DATA = os.path.join(REPO, "files", "production", "retrieval", "sequence_divergent")
FIG_DIR = os.path.join(REPO, "figures")
B, SEED = 2000, 124

_KEY = {"RFoldSearch": "RFoldSearch n500", "RFoldSearch-TM": "RFoldSearch-TM n500", "US-align": "US-align (exhaustive)",
        "Infernal": "Infernal", "nhmmer": "nhmmer", "BLASTN": "BLASTN",
        "RS-20": "RiboSeek c500", "RS-80": "RiboSeek RS-80 c500"}
COLOR = {m: S.BASE[k] for m, k in _KEY.items()}
COLOR.update({"Composition": "#6E6E6E", "Length": "#8C8C8C", "Shuffled": "#ABABAB"})
MARK = {m: S.MARKER[k] for m, k in _KEY.items()}
MARK.update({"Composition": "D", "Length": "D", "Shuffled": "D"})
LABEL = {m: m for m in _KEY}
LABEL.update({"RS-20": "RiboSeek (RS-20)", "RS-80": "RiboSeek (RS-80)", "Composition": "Token composition",
              "Length": "Length", "Shuffled": "Shuffled tokens"})
LINE = {m: S.LINESTYLE.get(k, "-") for m, k in _KEY.items()}
MS = 7.5                                        # one marker size for every method
CLANS = [("CL00001", "tRNA"), ("CL00112", "LSU rRNA"), ("CL00111", "SSU rRNA")]


def boot(vals, blocks, reps=B, seed=SEED):
    """Hierarchical block bootstrap of a mean: resample blocks, keep their queries
    (used by sequence_divergent_zone_checks.py)."""
    by = defaultdict(list)
    for q, v in vals.items():
        by[blocks[q]].append(v)
    keys = sorted(by)
    rng = np.random.default_rng(seed)
    bs = [np.mean([x for k in rng.integers(0, len(keys), len(keys)) for x in by[keys[k]]]) for _ in range(reps)]
    return float(np.mean(list(vals.values()))), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def boot_methods(V, methods, blocks, reps=B, seed=SEED):
    """Mean and 95% CI of every method from one set of block resamples over the queries the
    methods share, so that all intervals in a panel come from the same bootstrap."""
    qs = sorted(set.intersection(*(set(V[m]) for m in methods)))
    by = defaultdict(list)
    for q in qs:
        by[blocks[q]].append(q)
    keys = sorted(by)
    rng = np.random.default_rng(seed)
    draws = [[q for k in rng.integers(0, len(keys), len(keys)) for q in by[keys[k]]] for _ in range(reps)]
    out = {}
    for m in methods:
        bs = [np.mean([V[m][q] for q in d]) for d in draws]
        out[m] = (float(np.mean([V[m][q] for q in qs])), float(np.percentile(bs, 2.5)),
                  float(np.percentile(bs, 97.5)))
    return out, qs


def marker_size(m):
    return S.MSIZE.get(_KEY.get(m, ""), MS) if m == "RFoldSearch-TM" else (MS - 1.5 if m == "Composition" else MS)


def _spread(values, gap, floor=None):
    """Nudge sorted label positions apart by at least gap, keeping their order; with a floor,
    labels pushed below it are stacked upwards from it instead."""
    order = np.argsort(values)
    pos = np.asarray(values, dtype=float)[order]
    for _ in range(50):
        moved = False
        for i in range(1, len(pos)):
            if pos[i] - pos[i - 1] < gap:
                shift = (gap - (pos[i] - pos[i - 1])) / 2
                pos[i - 1] -= shift
                pos[i] += shift
                moved = True
        if not moved:
            break
    if floor is not None:
        for i in range(len(pos)):
            pos[i] = max(pos[i], floor + i * gap if i == 0 else pos[i - 1] + gap)
    out = np.empty_like(pos)
    out[order] = pos
    return out


def line_panel(ax, V, blocks, strata, ticks, methods, bold=()):
    """Sens@1FP of each method across identity strata, on the queries the methods share in each
    stratum. The RFoldSearch rankings are drawn heavy and the others recede; every line is named
    at its right end instead of through a distant legend."""
    boots = {st: boot_methods(V[st], [m for m in methods if m in V[st]], blocks) for st in strata}
    est = {st: boots[st][0] for st in strata}
    span = 0.30
    step = span / max(1, len(methods) - 1)
    ends = {}
    for i, m in enumerate(methods):
        dx = -span / 2 + i * step
        pts = [(k + dx,) + est[st][m] for k, st in enumerate(strata) if m in est[st]]
        hero = m in bold
        color = COLOR[m] if hero else S.lighten(COLOR[m], 0.35)
        style = dict(marker=MARK[m], color=color, ms=marker_size(m) if hero else marker_size(m) - 1.5,
                     mec="white", mew=0.8, zorder=5 if hero else 3)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], LINE[m], lw=2.6 if hero else 1.4,
                label=LABEL[m], **style)
        ends[m] = (pts[-1][0], pts[-1][1], color, hero)
    label_y = _spread([ends[m][1] for m in methods], 0.052, floor=0.02)
    x_text = len(strata) - 1 + span / 2 + 0.12
    for m, y in zip(methods, label_y):
        x, y0, color, hero = ends[m]
        ax.plot([x + 0.03, x_text - 0.03], [y0, y], color=color, lw=0.6, alpha=0.7, zorder=1)
        ax.text(x_text, y, LABEL[m], va="center", fontsize=9.5, color=color,
                fontweight="bold" if hero else "normal")
    # each stratum's query count under its label, as in the clan panel
    ax.set_xticks(range(len(strata)),
                  [f"{t}\n(n = {len(boots[st][1])})" for t, st in zip(ticks, strata)], fontsize=10.5)
    ax.set_xlim(-0.4, len(strata) - 0.5 + 0.75)
    ax.grid(axis="y", color="#E6E6E6", lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def interval_panel(ax, V_all, V_filtered, blocks, rows, level="Family"):
    """Sens@1FP of each method with all positives (open) and with sequence-detectable positives
    neutralised (filled) on the same queries, one row per method so the names read directly."""
    shared = set.intersection(*(set(V_filtered[m]) for m in rows))
    V_all = {m: {q: V_all[m][q] for q in shared} for m in rows}
    est_all, qs = boot_methods(V_all, rows, blocks)
    est_filt, _ = boot_methods(V_filtered, rows, blocks)
    for i, m in enumerate(rows):
        v_all, v_filt = est_all[m][0], est_filt[m][0]
        ax.plot([v_filt, v_all], [i, i], "-", color=COLOR[m], lw=1.3, zorder=2)
        ax.plot(v_all, i, MARK[m], ms=marker_size(m), mfc="white", mec=COLOR[m], mew=1.5, zorder=3)
        ax.plot(v_filt, i, MARK[m], ms=marker_size(m), color=COLOR[m], mec="white", mew=0.8,
                zorder=4)
    ax.set_yticks(range(len(rows)), [LABEL[m] for m in rows], fontsize=10.5)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(-0.02, 1.0)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis="x", color="#E6E6E6", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    S.style_axes(ax, xlabel=f"{level}-level Sens@1FP\n(n = {len(qs)})")


def filter_handles():
    """Legend entries for the fill code shared by the interval and clan panels."""
    return [plt.Line2D([], [], marker="o", ls="", mfc="white", mec="#555555", mew=1.6, ms=MS + 1,
                       label="All positives"),
            plt.Line2D([], [], marker="o", ls="", color="#555555", ms=MS + 1,
                       label="Sequence-detectable positives neutralised")]


def clan_panel(ax, cl, method_legend=True, filter_legend=True):
    """Clan-level Sens@1FP per clan with all positives (open) and with the positives found by
    BLASTN or nhmmer (E <= 10) made neutral (filled), joined per method."""
    cv, clan = cl["values"], cl["clan"]
    f = "B/N E<=10"
    groups = CLANS + [("other", "Other")]

    def in_group(q, g):
        return clan[q] == g if g != "other" else clan[q] not in {x for x, _ in CLANS}
    cm = ["RFoldSearch", "RFoldSearch-TM", "US-align", "Infernal", "RS-20", "RS-80"]   # as in B
    step = 0.8 / len(cm)
    for k, (g, glab) in enumerate(groups):
        for j, m in enumerate(cm):
            xc = k - 0.4 + step * (j + 0.5)
            x0, x1 = xc - step * 0.13, xc + step * 0.13
            qs = [q for q in cv["none"][m] if in_group(q, g) and q in cv[f][m]]
            v0 = np.mean([cv["none"][m][q] for q in qs])
            v1 = np.mean([cv[f][m][q] for q in qs])
            ms = marker_size(m) if m != "RFoldSearch-TM" else marker_size(m) - 1
            ax.plot([x0, x1], [v0, v1], "-", color=COLOR[m], lw=1.3, zorder=2)
            ax.plot(x0, v0, MARK[m], ms=ms, mfc="white", mec=COLOR[m], mew=1.5, zorder=3)
            ax.plot(x1, v1, MARK[m], ms=ms, color=COLOR[m], mec="white", mew=0.8, zorder=4)
        if k:
            ax.axvline(k - 0.5, color="#EEEEEE", lw=1.0, zorder=0)
    labels = []
    for g, glab in groups:
        n = sum(in_group(q, g) for q in cv["none"]["RFoldSearch"])
        labels.append(f"{glab} clan{'s' if g == 'other' else ''}\n(n = {n})")
    ax.set_xticks(range(len(groups)), labels, fontsize=10.5)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(-0.02, 0.86)
    ax.set_yticks(np.arange(0, 0.81, 0.2))
    ax.grid(axis="y", color="#E6E6E6", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    S.style_axes(ax, ylabel="Clan-level Sens@1FP")
    # Two legends: hue and shape name the method, fill names the filtering.
    hm = [plt.Line2D([], [], marker=MARK[m], ls="", color=COLOR[m], ms=MS + 1, label=LABEL[m]) for m in cm]
    hf = filter_handles()
    kw = dict(frameon=False, fontsize=10.5, title_fontsize=10.5, handletextpad=0.3, columnspacing=1.4,
              alignment="left", labelcolor="#666666")
    legends = []
    if method_legend:
        legends.append(ax.legend(handles=hm, ncol=3, title="Method", loc="lower left",
                                 bbox_to_anchor=(0.0, 1.0), **kw))
        ax.add_artist(legends[0])
        legends.append(ax.legend(handles=hf, ncol=1, title="Filtering", loc="lower left",
                                 bbox_to_anchor=(0.62, 1.0), **kw))
    elif filter_legend:                     # the methods are named by a shared figure legend
        legends.append(ax.legend(handles=hf, ncol=1, title="Filtering", loc="upper right", **kw))
    for leg in legends:
        leg.get_title().set_color("#666666")


def identity_strata():
    """Identity strata of Fig. 2E from the identity of unrelated chains (different Rfam family
    and clan): above the 99th percentile (up to 80%), 50th-99th, below the 50th."""
    from sequence_divergent_rescore import background_identity, rescore
    _, bg = background_identity()
    p50, p99 = np.quantile(bg, [0.5, 0.99])
    strata = [("high", p99, 0.80), ("middle", p50, p99), ("low", 0.0, p50)]
    per, _ = rescore(strata)
    ticks = [f"{100 * p99:.0f}–80%", f"{100 * p50:.0f}–{100 * p99:.0f}%", f"<{100 * p50:.0f}%"]
    return per, [name for name, _, _ in strata], ticks
