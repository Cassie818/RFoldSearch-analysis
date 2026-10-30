"""Fig. 5: retrieval on the predicted-structure benchmark, and the numbers behind it."""
import csv
import json
import os
from collections import defaultdict

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import evaluate as E
import figstyle as S
from figstyle import plt

# the runtime panels (top row) are drawn by scripts/05_runtime; their style module sets
# rcParams on import, so it is imported inside a context that restores them
import matplotlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "05_runtime"))
with matplotlib.rc_context():
    import plot_fig4 as R
    import runtime_style as RS

HERE, ROOT = E.HERE, E.ROOT
FIG = os.path.join(E.REPO, "figures")
# Every structure method is held at the same candidate depth, 500, so that the
# comparison is of the methods and not of how deep each was allowed to look.
DEPTH = 500
METHODS = ["RFoldSearch n500", "RFoldSearch-TM n500", "RiboSeek c500", "RiboSeek RS-80 c500", "BLASTN", "nhmmer", "Infernal",
           "US-align (exhaustive)"]
BAR_METHODS = [m for m in METHODS if m != "US-align (exhaustive)"]
PLDDT_METHODS = ["RFoldSearch n500", "RFoldSearch-TM n500", "RiboSeek RS-80 c500", "nhmmer", "Infernal"]
MIN_BIN = 5
TARGET_METHODS = PLDDT_METHODS + ["US-align (exhaustive)"]
METRIC = "Sens@1FP"
MAIN_FILTER = "none"
FILTERS = [("none", "None"), ("E<0.01", "$E<0.01$"), ("E<=10", "$E\\leq10$")]   # sequence-detected positives removed
QS_NAME = {"exp": "Experimental queries", "p2p": "Predicted queries"}
BINS = [(0, 70), (70, 80), (80, 90), (90, 101)]
BIN_LABEL = ["<70", "70-79", "80-89", "$\\geq$90"]
STRUCT_TRUTH = f"structure t{E.TM_MAIN:.2f}"
RM_TRUTH = f"rmscore t{E.TM_MAIN:.2f}"          # RMscore threshold quantile-matched to TM_MAIN
# H: structure detectors compared on the homologues sequence search misses,
# each held to the same top-200 rule.
WHO = [f"RFoldSearch n{DEPTH}", "RFoldSearch-TM n500", "RiboSeek c500", "RiboSeek RS-80 c500"]


def collect():
    """Target-level facts the per-query metrics cannot give: the invisible
    homologues' structural similarity, the recovery of sequence-missed
    homologues (H), and the same-species comparison (F)."""
    rows = list(csv.DictReader(open(os.path.join(ROOT, "sample", "accuracy_db.tsv")), delimiter="\t"))
    n = len(rows)
    fam = np.array([r["rfam_family"] if r["part"] == "labelled" else "" for r in rows], dtype=object)
    tax = np.array([r["taxonomy_id"] for r in rows], dtype=object)
    urs = np.array([r["urs_id"].split("_")[0] for r in rows], dtype=object)
    plddt = np.array([float(r["avg_plddt"]) if r["avg_plddt"] not in ("", "N/A") else np.nan
                      for r in rows])
    row_of_file = {f"{r['urs_id']}_rank_1.pdb": i for i, r in enumerate(rows)}

    out = {"invisible": {},
           "recover": {qs: {w: [[] for _ in BINS] for w in WHO} for qs in ("exp", "p2p")},
           "species": defaultdict(dict), "families_p2p": {},
           "target_bins": defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))}
    for qs in ("exp", "p2p"):
        runs = json.load(open(os.path.join(HERE, "runs", f"{qs}.json")))
        if qs == "p2p":
            qlist = json.load(open(os.path.join(ROOT, "sample", "p2p_queries.json")))["queries"]
            queries = [dict(id=q["urs_id"], family=q["family"], taxon=q["taxonomy_id"],
                            urs=q["urs_id"].split("_")[0]) for q in qlist]
            out["families_p2p"] = {q["id"]: q["family"] for q in queries}
        else:
            qlist = json.load(open(os.path.join(ROOT, "sample", "exp_queries.json")))["queries"]
            queries = [dict(id=q["id"], family=q["family"], taxon=None, urs=None) for q in qlist]
        ident = E.identical_targets(queries) if qs == "exp" else {}
        inv = {b: [0, 0] for b in BINS}
        for q in queries:
            excl = (urs == q["urs"]) if q["urs"] else np.zeros(n, bool)
            if q["urs"] is None:
                excl[ident[q["id"]]] = True
            same = (fam == q["family"]) & ~excl
            pos = same & (tax != q["taxon"]) if q["taxon"] is not None else same
            seen01 = np.zeros(n, bool)
            seen10 = np.zeros(n, bool)
            for tool in ("BLASTN", "nhmmer"):
                r = runs[tool][q["id"]]
                seen10[r["ranking"]] = True
                seen01[[i for i, e in zip(r["ranking"], r["evalue"]) if e < 0.01]] = True
            tm1, tm2 = E.load_usalign(os.path.join(HERE, "usalign", qs, f"{q['id']}.tsv"),
                                      row_of_file, n)
            sym = E.sym_tm(tm1, tm2)
            miss = pos & ~seen10
            for i, b in enumerate(BINS):
                m = miss & (plddt >= b[0]) & (plddt < b[1])
                inv[b][0] += int(np.sum(sym[m] >= E.TM_MAIN))
                inv[b][1] += int(m.sum())
            if qs == "exp":                                   # panel E (target pLDDT)
                # Sens@1FP restricted to the positives in one target-pLDDT bin:
                # positives in other bins (true homologues, not errors) are made
                # neutral; negatives are kept whatever their pLDDT.
                rank = {m: [i for i in runs[m][q["id"]]["ranking"] if not excl[i]] for m in runs}
                mean_tm = sym.copy()
                mean_tm[excl] = -1
                rank["US-align (exhaustive)"] = [i for i in np.lexsort((np.arange(n), -mean_tm))
                                                 if not excl[i]]
                truths = {"label": (pos, fam == ""),
                          STRUCT_TRUTH: E.structure_truth(tm1, tm2, excl, E.TM_MAIN)}
                for truth, (p, nu) in truths.items():
                    for b in BINS:
                        inb = (plddt >= b[0]) & (plddt < b[1])
                        for m in TARGET_METHODS:
                            res = E.metrics(rank[m], p & inb, nu | (p & ~inb))
                            if res is not None:
                                out["target_bins"][truth][f"{b[0]}-{b[1]}"][m][q["id"]] = res[METRIC]
            # panel H: Rfam homologues that sequence search misses -- not reported
            # by BLASTN or nhmmer at E < 0.01, nor by Infernal at all (the
            # strongest sequence union, so the recovery is a conservative
            # estimate) -- and how many of them each structure detector has among
            # its first 200 hits. Kept per query, so that the rate can be
            # bootstrapped over families.
            seq_hit = seen01.copy()
            seq_hit[[i for i in runs["Infernal"][q["id"]]["ranking"] if not excl[i]]] = True
            missed = pos & ~seq_hit
            for w in WHO:
                str_hit = np.zeros(n, bool)
                str_hit[[i for i in runs[w][q["id"]]["ranking"] if not excl[i]][:200]] = True
                for i, b in enumerate(BINS):
                    m = missed & (plddt >= b[0]) & (plddt < b[1])
                    out["recover"][qs][w][i].append((q["family"], int(m.sum()), int(np.sum(m & str_hit))))
            if qs == "p2p":                                   # panel F
                neu_cross = (fam == "") | excl | (same & (tax == q["taxon"]))
                pos_same = same & (tax == q["taxon"])
                neu_same = (fam == "") | excl | (same & (tax != q["taxon"]))
                mean_tm = sym.copy()
                mean_tm[excl] = -1
                rank = {m: [i for i in runs[m][q["id"]]["ranking"] if not excl[i]] for m in runs}
                rank["US-align (exhaustive)"] = [i for i in np.lexsort((np.arange(n), -mean_tm))
                                                 if not excl[i]]
                for key, (p, nu) in (("cross", (pos, neu_cross)), ("same", (pos_same, neu_same))):
                    for m, r in rank.items():
                        res = E.metrics(r, p, nu)
                        if res is not None:
                            out["species"][key].setdefault(m, {})[q["id"]] = res[METRIC]
        out["invisible"][qs] = {f"{b[0]}-{b[1]}": inv[b] for b in BINS}
    return out


# Main-figure methods, structure-based first (exhaustive US-align leading) and sequence-based after. C leaves out the
# sequence methods whose own hits define the filter (BLASTN, nhmmer).
STRUCT_METHODS = ["US-align (exhaustive)", "RFoldSearch n500", "RFoldSearch-TM n500", "RiboSeek c500",
                  "RiboSeek RS-80 c500"]
SEQ_METHODS = ["BLASTN", "nhmmer", "Infernal"]
MAIN = STRUCT_METHODS + SEQ_METHODS
HERO = ("RFoldSearch n500", "RFoldSearch-TM n500")
SEQ_ONLY = ("BLASTN", "nhmmer", "Infernal")       # never read a structure: dashed in line panels
GREY = "#666666"
QS_SHORT = {"exp": "Experimental", "p2p": "Predicted"}


def tone(m):
    """As in Fig. 2: the RFoldSearch rankings in full tone, the others receding."""
    return S.COLOR[m] if m in HERO else S.lighten(S.COLOR[m], 0.35)


def letter(ax, lab, x):
    ax.text(x, 1.06, lab, transform=ax.transAxes, fontsize=16, fontweight="bold", va="bottom", ha="right")


def fill_legend(ax, labels, loc="lower center", anchor=(0.5, 1.0), ncol=2):
    h = [plt.Line2D([], [], marker="o", ls="", mfc="white", mec="#555555", mew=1.6, ms=8, label=labels[0]),
         plt.Line2D([], [], marker="o", ls="", color="#555555", ms=8, label=labels[1])]
    ax.legend(handles=h, frameon=False, fontsize=9.5, loc=loc, bbox_to_anchor=anchor, ncol=ncol,
              labelcolor=GREY, handletextpad=0.3, columnspacing=1.4)


def point(ax, x, y, lo, hi, m, filled, horizontal=True, ms=8):
    c = tone(m)
    if horizontal:
        ax.hlines(y, lo, hi, color=c, lw=2.0 if m in HERO else 1.4, zorder=2)
    else:
        ax.vlines(x, lo, hi, color=c, lw=1.8 if m in HERO else 1.1, zorder=2)
    ax.plot(x, y, S.MARKER[m], ms=ms + (2 if m == "RFoldSearch-TM n500" else 0), color=c,
            mfc=c if filled else "white", mec="white" if filled else c,
            mew=(0.4 if m == "RFoldSearch-TM n500" else 0.8) if filled else 1.6,
            zorder=4 if m in HERO else 3, clip_on=False)


def overall_panel(fig, cell, est, truth, methods, oracle, title, ylabels=True):
    """Two side-by-side axes in `cell`, experimental and predicted queries; rows are methods.
    Query sets are told apart by column, not by fill, so that open and filled markers keep
    one meaning across the paper (all positives / sequence-detectable positives neutral).
    Returns the left axes."""
    sub = cell.subgridspec(1, 2, wspace=0.12)
    axes = [fig.add_subplot(sub[0]), fig.add_subplot(sub[1])]
    for k, (ax, qs) in enumerate(zip(axes, ("exp", "p2p"))):
        for i, m in enumerate(methods):
            if m == oracle:
                ax.text(0.03, i, "1.00 (reference)", va="center", fontsize=8.5, color=GREY)
                continue
            mean, lo, hi, _ = est(qs, truth, MAIN_FILTER, m)
            point(ax, mean, i, mean - lo, mean + hi, m, True)
        ax.set_yticks(range(len(methods)), [S.SHORT[m] for m in methods] if ylabels and k == 0
                      else [""] * len(methods), fontsize=10.5)
        if k:
            ax.spines["left"].set_color("#BBBBBB")
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(len(methods) - 0.5, -0.5)
        ax.set_xlim(0, 1.0)
        ax.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1"])
        S.style_axes(ax, xlabel=METRIC)
        ax.set_title(f"{QS_SHORT[qs]} queries", fontsize=10, color=GREY, pad=4)
    left, right = axes[0].get_position(fig), axes[1].get_position(fig)
    fig.text((left.x0 + right.x1) / 2, left.y1 + 0.032 * 10.6 / fig.get_figheight(), title, ha="center", va="bottom", fontsize=11.5)
    return axes[0]


def plddt_panel(ax, est, qplddt, truth, methods, title):
    """Predicted queries by query pLDDT; bins with fewer than MIN_BIN evaluable queries are
    not drawn, and every bin is labelled with its number of evaluable queries."""
    ns = []
    span = 0.36
    for i, m in enumerate(methods):
        dx = -span / 2 + i * span / max(1, len(methods) - 1)
        xs, ys, los, his = [], [], [], []
        for k, b in enumerate(BINS):
            ids = {q for q in qplddt if b[0] <= qplddt[q] < b[1]}
            mean, lo, hi, n = est("p2p", truth, "none", m, ids=ids)
            if i == 0:
                ns.append(n)
            if n >= MIN_BIN:
                xs.append(k + dx); ys.append(mean); los.append(mean - lo); his.append(mean + hi)
        c = tone(m)
        ax.vlines(xs, los, his, color=c, lw=1.6 if m in HERO else 0.9, zorder=2)
        ax.plot(xs, ys, "--" if m in SEQ_ONLY else "-", marker=S.MARKER[m], color=c,
                lw=2.6 if m in HERO else 1.4, ms=(8 if m in HERO else 6) + (3 if m == "RFoldSearch-TM n500" else 0),
                mec="white", mew=0.8, zorder=4 if m in HERO else 3)
    ax.set_xticks(range(len(BINS)), [f"{lab}\n(n = {n})" for lab, n in zip(BIN_LABEL, ns)], fontsize=10)
    for k, n in enumerate(ns):
        if n < MIN_BIN:
            ax.text(k, 0.5, f"too few\nqueries", ha="center", va="center", fontsize=8.5, color=GREY)
    ax.set_xlim(-0.5, len(BINS) - 0.5)
    ax.set_ylim(0, 1.02)
    S.style_axes(ax, xlabel="Query pLDDT", ylabel=METRIC)
    ax.set_title(title, fontsize=11.5, pad=6)


def main():
    os.makedirs(FIG, exist_ok=True)
    pq = json.load(open(os.path.join(HERE, "per_query.json")))
    fams = pq["families"]
    data = collect()
    qplddt = {q["urs_id"]: q["plddt"] for q in
              json.load(open(os.path.join(ROOT, "sample", "p2p_queries.json")))["queries"]}
    order = ["exp", "p2p"]

    def est(qs, truth, filt, method, ids=None):
        d = pq.get(qs, {}).get(f"{truth}|{filt}", {}).get(method, {})
        v = {q: x[METRIC] for q, x in d.items() if ids is None or q in ids}
        if not v:
            return np.nan, 0.0, 0.0, 0
        mean, lo, hi = E.block_boot(v, fams)
        return mean, max(mean - lo, 0), max(hi - mean, 0), len(v)

    # Recovery of the Rfam homologues that sequence search misses (D): rate =
    # recovered / missed, pooled over queries; 95% interval from resampling families.
    def ratio_boot(recs, reps=2000, seed=0):
        by = defaultdict(lambda: np.zeros(2))
        for f, k_miss, k_rec in recs:
            by[f] += (k_miss, k_rec)
        arr = np.array(list(by.values()))
        if arr[:, 0].sum() == 0:
            return np.nan, 0.0, 0.0, 0, 0
        rate = arr[:, 1].sum() / arr[:, 0].sum()
        rng = np.random.default_rng(seed)
        bs = []
        for _ in range(reps):
            a = arr[rng.integers(0, len(arr), len(arr))]
            if a[:, 0].sum():
                bs.append(a[:, 1].sum() / a[:, 0].sum())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return rate, max(rate - lo, 0), max(hi - rate, 0), int(arr[:, 0].sum()), int(np.sum(arr[:, 1] > 0))

    h_values = {}
    for qs in order:
        for w in WHO:
            for i, b in enumerate(BINS):
                v = ratio_boot(data["recover"][qs][w][i])
                h_values.setdefault(qs, {}).setdefault(f"{b[0]}-{b[1]}", {})[w] = dict(
                    rate=v[0], lo=v[0] - v[1], hi=v[0] + v[2], missed=v[3], families_recovered=v[4])

    # ================= main figure =================
    fig = plt.figure(figsize=(16.0, 13.0))
    rows = fig.add_gridspec(3, 1, height_ratios=(1.05, 1, 1), hspace=0.44, left=0.12, right=0.99,
                            top=0.96, bottom=0.115)
    # A, B: runtime on the two predicted-structure databases, in this figure's type sizes
    # and with the same tones as the retrieval panels
    RS.K = 1.6
    R.setup_styles(tone=lambda m, c: c if m in ("RFoldSearch", "RFoldSearch-TM")
                   else S.lighten(c, 0.35))
    ax_sa, ax_sb = R.speed_row(fig, rows[0], wspace=0.05)
    # C's method names sit where the y-axis labels of the row below sit, so that the two
    # rows share their outer edges.
    r0 = fig.add_gridspec(1, 2, wspace=0.10, left=0.165, right=0.99,
                          top=rows[1].get_position(fig).y1, bottom=rows[1].get_position(fig).y0)
    ax_a1 = overall_panel(fig, r0[0], est, "label", MAIN, None, "Annotation truth (Rfam family)")
    title_c = fig.texts[-1]
    overall_panel(fig, r0[1], est, STRUCT_TRUTH, MAIN, "US-align (exhaustive)",
                  f"Structure truth (TM-score ≥ {E.TM_MAIN:.2f})", ylabels=False)
    title_d = fig.texts[-1]

    r1 = rows[2].subgridspec(1, 2, wspace=0.22)
    ax_b, ax_c = (fig.add_subplot(r1[k]) for k in range(2))
    plddt_panel(ax_b, est, qplddt, STRUCT_TRUTH, [m for m in MAIN if m != "US-align (exhaustive)"],
                "Predicted queries (structure truth)")

    # C: structure truth, before/after neutralising sequence-detectable positives.
    # BLASTN and nhmmer are left out (their own positives become neutral) and US-align (the oracle).
    c_methods = ["RFoldSearch n500", "RFoldSearch-TM n500", "RiboSeek c500", "RiboSeek RS-80 c500", "Infernal"]
    step = 0.7 / len(c_methods)
    for g, qs in enumerate(order):
        for j, m in enumerate(c_methods):
            xc = g - 0.35 + step * (j + 0.5)
            v0, v1 = est(qs, STRUCT_TRUTH, "none", m), est(qs, STRUCT_TRUTH, "E<=10", m)
            x0, x1 = xc - step * 0.18, xc + step * 0.18
            ax_c.plot([x0, x1], [v0[0], v1[0]], "-", color=tone(m), lw=1.3, zorder=2)
            point(ax_c, x0, v0[0], v0[0] - v0[1], v0[0] + v0[2], m, False, horizontal=False, ms=7)
            point(ax_c, x1, v1[0], v1[0] - v1[1], v1[0] + v1[2], m, True, horizontal=False, ms=7)
    ax_c.axvline(0.5, color="#DDDDDD", lw=0.9, zorder=0)
    ax_c.set_xticks(range(2), ["Experimental\nqueries", "Predicted\nqueries"], fontsize=10)
    ax_c.set_xlim(-0.5, 1.5)
    ax_c.set_ylim(0, 0.7)
    S.style_axes(ax_c, ylabel=METRIC)
    ax_c.set_title("Structure truth", fontsize=11.5, pad=6)
    fill_legend(ax_c, ("All positives", "Sequence-detectable positives neutral"), loc="upper center",
                anchor=(0.5, 1.0), ncol=2)
    ax_c.set_ylim(0, 0.75)

    # every letter level with its panel's title; A, C, E in one column, B, D, F in another
    pb, pc = ax_b.get_position(), ax_c.get_position()
    col_l, col_r = pb.x0 - 0.14 * pb.width, pc.x0 - 0.14 * pc.width
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    for title, lab, x in ((ax_sa.title, "A", col_l), (ax_sb.title, "B", col_r), (title_c, "C", col_l),
                          (title_d, "D", col_r), (ax_b.title, "E", col_l), (ax_c.title, "F", col_r)):
        bb = inv.transform(title.get_window_extent())
        fig.text(x, (bb[0][1] + bb[1][1]) / 2, lab, fontsize=16, fontweight="bold",
                 va="center", ha="right")
    # Legend: one row per kind of method, columns aligned across rows
    def handle(m):
        if m in MAIN:
            return plt.Line2D([], [], color=tone(m), lw=3.0 if m in HERO else 2.2,
                              ls="--" if m in SEQ_ONLY else "-", marker=S.MARKER[m],
                              ms=9 + (3 if m == "RFoldSearch-TM n500" else 0), mec="white",
                              mew=0.9, label=S.SHORT[m])
        # methods timed but not in the retrieval benchmark
        return plt.Line2D([], [], color=R.F.COLOR[m], lw=2.2, ls=R.linestyle(m),
                          marker=R.F.MARKER[m], ms=9, mec="white", mew=0.9,
                          label=R.legend_label(m))
    blank = plt.Line2D([], [], ls="none", label=" ")
    rows = [["RFoldSearch n500", "RFoldSearch-TM n500", "RiboSeek c500", "RiboSeek RS-80 c500", None],
            ["US-align (exhaustive)", "RTM-align", "RMalign", "GTalign", "GTalign-fast"],
            ["BLASTN", "nhmmer", "Infernal", "Infernal-fast", None]]
    handles = [handle(r[c]) if r[c] else blank for c in range(5) for r in rows]   # column-major
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=10.5,
               bbox_to_anchor=(0.5, -0.02), handlelength=2.6, columnspacing=2.2)
    fig.savefig(os.path.join(FIG, "fig5.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ================= supplementary figure =================
    fig = plt.figure(figsize=(14.0, 6.0))
    # The RMscore truth is in supp_predicted_structure_truth.png.
    gs = fig.add_gridspec(1, 2, width_ratios=(1, 2.1), wspace=0.22, left=0.12, right=0.99, top=0.88,
                          bottom=0.21)
    sd, sc = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # B: experimental queries by target pLDDT, both truths (US-align omitted under structure truth).
    fams_exp = dict(fams)
    blocks = [("label", 0), (STRUCT_TRUTH, len(BINS) + 1)]
    e_values = defaultdict(dict)
    for truth, x0 in blocks:
        for i, m in enumerate(MAIN):
            if truth != "label" and m == "US-align (exhaustive)":
                continue
            dx = -0.24 + i * 0.48 / (len(MAIN) - 1)          # dodge so the intervals stay apart
            xs, ys, los, his = [], [], [], []
            for k, b in enumerate(BINS):
                v = data["target_bins"][truth][f"{b[0]}-{b[1]}"].get(m, {})
                if len(v) < MIN_BIN:
                    continue
                mean, lo, hi = E.block_boot(v, fams_exp)
                e_values[f"{truth}|{b[0]}-{b[1]}"][m] = {"n": len(v), "mean": mean, "lo": lo, "hi": hi}
                xs.append(x0 + k + dx); ys.append(mean); los.append(lo); his.append(hi)
            c = tone(m)
            sc.vlines(xs, los, his, color=c, lw=1.4 if m in HERO else 0.9, zorder=2)
            sc.plot(xs, ys, "--" if m in SEQ_ONLY else "-", marker=S.MARKER[m], color=c,
                    lw=2.4 if m in HERO else 1.3, ms=7, mec="white", mew=0.8, zorder=4 if m in HERO else 3)
    sc.axvline(len(BINS), color="#DDDDDD", lw=0.9, zorder=1)
    for (truth, x0), name in zip(blocks, ("Annotation truth", "Structure truth")):
        sc.text(x0 + (len(BINS) - 1) / 2, 1.02, name, ha="center", va="bottom", fontsize=9.5, color=GREY)
    sc.set_xticks(np.r_[np.arange(len(BINS)), len(BINS) + 1 + np.arange(len(BINS))], BIN_LABEL + BIN_LABEL,
                  fontsize=9)
    sc.set_xlim(-0.5, 2 * len(BINS) + 0.5)
    sc.set_ylim(0, 1.0)
    sc.grid(axis="y", color="#E6E6E6", lw=0.8, zorder=0)
    sc.set_axisbelow(True)
    S.style_axes(sc, xlabel="Target pLDDT", ylabel=METRIC)
    sc.set_title("Experimental queries", fontsize=11.5, pad=20)

    # A: cost of crossing species (predicted queries): paired difference, different-species
    # minus same-species positives, with its 95% interval; no open/filled markers.
    both = sorted(set(data["species"]["cross"]["RFoldSearch n500"])
                  & set(data["species"]["same"]["RFoldSearch n500"]))
    for y, m in enumerate(MAIN):
        cross, same = data["species"]["cross"].get(m, {}), data["species"]["same"].get(m, {})
        ids = [q for q in both if q in cross and q in same]
        if not ids:
            continue
        d, lo, hi = E.block_boot({q: cross[q] - same[q] for q in ids}, data["families_p2p"])
        point(sd, d, y, lo, hi, m, True)
        sd.text(hi + 0.012, y, f"{d:+.2f}{'*' if not (lo <= 0 <= hi) else ''}", va="center",
                fontsize=9, color=GREY)
    sd.axvline(0, color="#888888", lw=1.1, ls="--", zorder=1)
    sd.set_yticks(range(len(MAIN)), [S.SHORT[m] for m in MAIN], fontsize=10.5)
    sd.tick_params(axis="y", length=0)
    sd.set_ylim(len(MAIN) - 0.45, -0.55)
    sd.set_xlim(-0.35, 0.1)
    sd.grid(axis="x", color="#E6E6E6", lw=0.8, zorder=0)
    sd.set_axisbelow(True)
    S.style_axes(sd, xlabel="Δ Sens@1FP (different − same species)")
    sd.set_title("Predicted queries (annotation truth)", fontsize=11.5, pad=6)
    for ax, lab, x in ((sd, "A", -0.62), (sc, "B", -0.065)):
        letter(ax, lab, x)
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, fontsize=10.5,
               bbox_to_anchor=(0.5, 0.0), handlelength=2.2, columnspacing=1.2)
    fig.savefig(os.path.join(FIG, "supp_predicted_breakdown.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    out = {"metric": METRIC, "main_filter": MAIN_FILTER, "candidate_depth": DEPTH,
           "invisible_by_plddt": data["invisible"], "H recovery": h_values,
           "caption_counts": {
               "A,B,G queries": {f"{qs}|{truth}|{filt}":
                                 est(qs, truth, filt, METHODS[0])[3]
                                 for qs in order for truth in ("label", STRUCT_TRUTH, RM_TRUTH)
                                 for filt, _ in FILTERS},
               "E queries per target bin": {t: {b: len(d.get("RFoldSearch n500", {})) for b, d in v.items()}
                                            for t, v in data["target_bins"].items()},
               "C,D query pLDDT bins": {f"{b[0]}-{b[1]}":
                                      len([q for q in qplddt if b[0] <= qplddt[q] < b[1]])
                                      for b in BINS},
               "F paired queries": len(both)}}
    for qs in order:
        for truth in ("label", STRUCT_TRUTH, RM_TRUTH):
            for filt, _ in FILTERS:
                key = f"{qs}|{truth}|{filt}"
                out[key] = {}
                for m in METHODS:
                    mean, lo, hi, k = est(qs, truth, filt, m)
                    if k:
                        out[key][m] = {"n": k, "mean": mean, "lo": mean - lo, "hi": mean + hi}
    out["exp target pLDDT"] = e_values
    json.dump(out, open(os.path.join(HERE, "fig5_data.json"), "w"), indent=1)
    print("wrote", os.path.join(FIG, "fig5.png"), "and", os.path.join(FIG, "supp_predicted_breakdown.png"))


if __name__ == "__main__":
    main()
