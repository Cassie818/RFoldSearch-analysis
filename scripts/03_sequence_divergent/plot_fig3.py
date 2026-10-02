"""Sequence-divergent homologues on the 885-chain RNAsolo benchmark (Section 2.2 definitions).

Fig. 3:
  a  family-level Sens@1FP by pairwise sequence identity between query and
     same-family target (sequence_divergent_rescore.seq_identity), in zones
     defined from the identity of unrelated chains (background_identity), after
     Rost (1999): above background (> 99th percentile), twilight zone (50th-99th)
     and midnight zone (< 50th). Only same-family targets in the zone are
     positives; same-family targets outside it are neutral. Pairs above 80% are
     not shown.
     Query eligibility is determined separately in each zone, so cross-zone
     values are descriptive rather than paired measurements of a fixed cohort.
     95% block-bootstrap intervals for RFoldseek and Infernal only.
  b  RFoldseek minus each comparator after same-family targets detected by
     BLASTN (-task blastn, both strands) or nhmmer at E <= 10 are made neutral,
     all families and with tRNA (RF00005) excluded; homology-free controls rank
     targets by 3Dr token composition, by length, or by RFoldseek on the query's
     shuffled tokens
  c  clan-level Sens@1FP by clan without filtering (open) and after removing
     clan positives detected by BLASTN (-task blastn, both strands) or nhmmer at
     E <= 10 (filled); identity bins are uninformative here because cross-family
     pairs are almost all of low identity

RFoldseek and RFoldseek-TM are means over ten sha256 tie-breaking seeds. Unlabelled
chains are false positives. Means over queries; 95% intervals from the hierarchical
block bootstrap over Rfam families (2,000 replicates, seed 124), paired for differences.

Reads files/production/retrieval/sequence_divergent/{family,clan}_perquery.json.
Writes figures/fig3.png.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "common"))
import figstyle as S                                               # noqa: E402
from figstyle import plt                                           # noqa: E402

DATA = os.path.join(REPO, "files", "production", "retrieval", "sequence_divergent")
FIG_DIR = os.path.join(REPO, "figures")
B, SEED = 2000, 124

ZONES = ("above", "twilight", "midnight")      # identity zones, defined from the unrelated-pair background

_KEY = {"RFoldseek": "RFoldseek n500", "RFoldseek-TM": "RFoldseek-TM n500", "US-align": "US-align (exhaustive)",
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
LINE["Composition"] = "--"                     # a control, drawn apart from the methods
LW, MS = 2.0, 7.5                               # one line width and one marker size for every method
OWN = {"Infernal": "Infernal any", "BLASTN": "B/N", "nhmmer": "B/N"}   # neutralised by its own hits
CLANS = [("CL00001", "tRNA"), ("CL00112", "LSU rRNA"), ("CL00111", "SSU rRNA")]


def boot(vals, blocks, reps=B, seed=SEED):
    """Hierarchical block bootstrap of a mean: resample blocks, keep their queries."""
    by = defaultdict(list)
    for q, v in vals.items():
        by[blocks[q]].append(v)
    keys = sorted(by)
    rng = np.random.default_rng(seed)
    bs = [np.mean([x for k in rng.integers(0, len(keys), len(keys)) for x in by[keys[k]]]) for _ in range(reps)]
    return float(np.mean(list(vals.values()))), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def boot_p(vals, blocks, reps=B, seed=SEED):
    """Two-sided bootstrap p for a mean differing from 0, from the same resamples as boot()."""
    by = defaultdict(list)
    for q, v in vals.items():
        by[blocks[q]].append(v)
    keys = sorted(by)
    rng = np.random.default_rng(seed)
    bs = np.array([np.mean([x for k in rng.integers(0, len(keys), len(keys)) for x in by[keys[k]]])
                   for _ in range(reps)])
    return min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))


def nfam(qs, blocks):
    return len({blocks[q] for q in qs})


def marker_size(m):
    return S.MSIZE.get(_KEY.get(m, ""), MS) if m == "RFoldseek-TM" else (MS - 1.5 if m == "Composition" else MS)


def line_panel(ax, V, blocks, strata, ticks, methods, ci, light=(), bold=()):
    """Sens@1FP of each method across strata; `light` strata are drawn faded and dashed; if `bold` is
    given, those methods are drawn thick and the others thin."""
    dx = {"RFoldseek": -0.06, "RFoldseek-TM": -0.03, "US-align": 0.03, "Infernal": 0.06, "nhmmer": 0.02, "BLASTN": -0.02, "RS-80": 0.04}
    for m in methods:
        pts = []
        for k, st in enumerate(strata):
            if m not in V[st] or OWN.get(m, "#") in st:
                continue
            mean, lo, hi = boot(V[st][m], blocks)
            x = k + dx.get(m, 0.0)
            if m in ci:
                ax.vlines(x, lo, hi, color=COLOR[m], lw=1.2, alpha=0.2 if st in light else 0.45, zorder=2)
            pts.append((x, mean, st in light))
        solid = [p for p in pts if not p[2]]
        lw = LW if not bold else 2.2 if m in bold else 1.2
        kw = dict(marker=MARK[m], color=COLOR[m], lw=lw, ms=marker_size(m), mec="white", mew=0.8,
                  zorder=4 if m in ("RFoldseek",) + tuple(bold) else 3)
        ax.plot([p[0] for p in solid], [p[1] for p in solid], LINE[m], label=LABEL[m], **kw)
        faded = [p for p in pts if p[2]]
        if faded and solid:
            ax.plot([solid[-1][0], faded[0][0]], [solid[-1][1], faded[0][1]], ":", color=COLOR[m], lw=LW,
                    alpha=0.35, zorder=2)
            ax.plot([faded[0][0]], [faded[0][1]], ls="", marker=MARK[m], color=COLOR[m], ms=marker_size(m),
                    alpha=0.35, mec="white", mew=0.8, zorder=2)
    ax.set_xticks(range(len(strata)), ticks, fontsize=10.5)
    ax.set_xlim(-0.35, len(strata) - 0.65)
    ax.set_ylim(0, 0.82)
    ax.set_yticks(np.arange(0, 0.81, 0.2))
    ax.grid(axis="y", color="#E6E6E6", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, frameon=False, fontsize=9.5, loc="upper right", handlelength=2.2, columnspacing=1.0)


def forest_panel(ax, V, blocks, stratum, rows, xlabel):
    """RFoldseek minus each comparator in one stratum, all families (filled) and tRNA excluded (open)."""
    for i, m in enumerate(rows):
        for j, keep in enumerate((lambda q: True, lambda q: blocks[q] != "RF00005")):
            qs = [q for q in V[stratum]["RFoldseek"] if keep(q) and q in V[stratum][m]]
            diff = {q: V[stratum]["RFoldseek"][q] - V[stratum][m][q] for q in qs}
            d, lo, hi = boot(diff, blocks)
            y = i - 0.16 if j == 0 else i + 0.16
            ax.hlines(y, lo, hi, color=COLOR[m], lw=LW, zorder=2)
            if boot_p(diff, blocks) >= 0.05:
                ax.text(hi + 0.012, y, "n.s.", color=COLOR[m], fontsize=9.5, va="center")
            ax.plot(d, y, MARK[m], ms=MS + 1, color=COLOR[m], mfc=COLOR[m] if j == 0 else "white",
                    mec=COLOR[m], mew=1.6, zorder=3)
    ax.axvline(0, color="#888888", lw=1.1, ls="--", zorder=1)
    for v in (0.2, 0.4):
        ax.axvline(v, color="#E6E6E6", lw=0.8, zorder=0)
    ax.axhline(rows.index("Composition") - 0.5, color="#DDDDDD", lw=1.2, zorder=0)
    ax.set_yticks(range(len(rows)), [LABEL[m] for m in rows], fontsize=11)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(rows) - 0.45, -0.55)
    S.style_axes(ax, xlabel=xlabel)
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color="#555555", ms=MS + 1,
                                  label="all families"),
                       plt.Line2D([], [], marker="o", ls="", color="#555555", mfc="white", mew=1.6, ms=MS + 1,
                                  label="tRNA excluded")],
              frameon=False, fontsize=10, loc="lower right", bbox_to_anchor=(1.0, 0.99), ncol=2,
              handletextpad=0.4, columnspacing=1.4)


def clan_panel(ax, cl):
    cv, clan = cl["values"], cl["clan"]
    f = "B/N E<=10"
    groups = CLANS + [("other", "Other")]

    def in_group(q, g):
        return clan[q] == g if g != "other" else clan[q] not in {x for x, _ in CLANS}
    cm = ["RFoldseek", "Infernal", "RS-80"]
    for k, (g, glab) in enumerate(groups):
        for j, m in enumerate(cm):
            x = k + (j - 1) * 0.2
            q0 = [q for q in cv["none"][m] if in_group(q, g)]
            q1 = [q for q in cv[f][m] if in_group(q, g)]
            v0 = np.mean([cv["none"][m][q] for q in q0]) if q0 else np.nan
            v1 = np.mean([cv[f][m][q] for q in q1]) if q1 else np.nan
            if v0 - v1 > 0.03:
                ax.annotate("", xy=(x, v1 + 0.014), xytext=(x, v0 - 0.014),
                            arrowprops=dict(arrowstyle="-|>", color=COLOR[m], lw=LW, mutation_scale=12), zorder=2)
            ax.plot(x, v0, MARK[m], ms=MS + 2.5, mfc="white", mec=COLOR[m], mew=1.8, zorder=3)
            ax.plot(x, v1, MARK[m], ms=MS, color=COLOR[m], mec="white", mew=0.8, zorder=4)
    labels = []
    for g, glab in groups:
        labels.append(f"{glab} clan{'s' if g == 'other' else ''}")
    ax.set_xticks(range(len(groups)), labels, fontsize=10.5)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(-0.02, 0.64)
    ax.set_yticks(np.arange(0, 0.51, 0.1))
    ax.grid(axis="y", color="#E6E6E6", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    S.style_axes(ax, ylabel="Clan-level Sens@1FP")
    h = [plt.Line2D([], [], marker=MARK[m], ls="", color=COLOR[m], ms=MS + 1, label=LABEL[m]) for m in cm]
    h += [plt.Line2D([], [], marker="o", ls="", mfc="white", mec="#555555", mew=1.8, ms=MS + 1, label="all"),
          plt.Line2D([], [], marker="o", ls="", color="#555555", ms=MS + 1,
                     label="BLASTN/nhmmer $E$<10 removed")]
    ax.legend(handles=h, ncol=5, frameon=False, fontsize=10.5, loc="upper left", handletextpad=0.3,
              columnspacing=1.4)


def letters(axd, keys, big=()):
    for k in keys:
        axd[k].text(-0.055 if k in big else -0.11, 1.07, k.upper(), transform=axd[k].transAxes, fontsize=18,
                    fontweight="bold", va="bottom")


def main():
    fam = json.load(open(os.path.join(DATA, "family_perquery.json")))
    blocks, V = fam["blocks"], fam["values"]
    methods = ["RFoldseek", "RFoldseek-TM", "US-align", "Infernal", "nhmmer", "BLASTN", "RS-20", "RS-80",
               "Composition"]
    from sequence_divergent_rescore import background_identity, rescore
    _, bg = background_identity()
    p50, p99 = np.quantile(bg, [0.5, 0.99])  # identity between unrelated chains
    per, _ = rescore([("above", p99, 0.80), ("twilight", p50, p99), ("midnight", 0.0, p50)])
    V = dict(V, **{k: dict(v) for k, v in per.items()})
    bins = [("above", f"{100 * p99:.0f}–80%\nabove background"),
            ("twilight", f"{100 * p50:.0f}–{100 * p99:.0f}%\ntwilight zone"),
            ("midnight", f"<{100 * p50:.0f}%\nmidnight zone")]
    cl = json.load(open(os.path.join(DATA, "clan_perquery.json")))
    fig, axd = plt.subplot_mosaic([["a", "b"], ["c", "c"]], figsize=(13.5, 10),
                                  gridspec_kw={"height_ratios": [1, 0.8], "hspace": 0.38, "wspace": 0.42})
    ticks = [lab for _, lab in bins]
    line_panel(axd["a"], V, blocks, [b for b, _ in bins], ticks, methods, ci=("RFoldseek", "Infernal"),
               bold=("RFoldseek", "RFoldseek-TM"))
    S.style_axes(axd["a"], ylabel="Family-level Sens@1FP")
    axd["a"].set_ylim(0, 1.08)
    axd["a"].set_yticks(np.arange(0, 1.01, 0.2))
    axd["a"].legend(ncol=3, frameon=False, fontsize=9, loc="upper center", handlelength=1.8,
                    columnspacing=0.9, handletextpad=0.4)
    forest_panel(axd["b"], V, blocks, "B/N E<=10", ["Infernal", "RS-20", "RS-80", "Composition", "Length", "Shuffled"],
                 "RFoldseek − comparator (Δ Sens@1FP)")
    axd["b"].set_xlim(-0.25, 0.6)
    clan_panel(axd["c"], cl)
    letters(axd, "abc", big=("c",))
    out = os.path.join(FIG_DIR, "fig3.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
