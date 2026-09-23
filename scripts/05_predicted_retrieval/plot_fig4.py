"""Main figure for the predicted-structure benchmark, and the numbers behind it.

Nine panels, three by three, in Fig. 2's idiom and with Fig. 2's primary metric
(Sens@1FP) so that the two benchmarks can be read against one another: the same
colours, the same markers, and the same sorted-sensitivity curve. Row 1 puts the
two ground truths side by side, as Fig. 2 puts Family and Clan side by side, and
ends with the distribution behind those means; the rows below take the finding
apart. Panels carry no titles -- only the bold letter and self-describing axes.

  A  Annotation truth. An experimental query searched against the predicted
     database retrieves homologues; a predicted query does not, and neither does
     exhaustive US-align, so the limit is the structures, not the search. Here
     US-align is a competing method (the truth is the Rfam family), so it is a
     bar like the rest.
  B  Structure truth (symmetric TM >= 0.45). The same contrast, with US-align as
     the oracle ceiling it becomes when it defines the positives.
  C  The distribution behind A: per-query sensitivity sorted from best to worst,
     as in Fig. 2A,B. A method's curve reaching far to the right means it works
     on many queries, not just on average.
  D  What structure search adds: sensitivity as sequence-detectable homologues
     are removed, first at E < 0.01 and then at the stricter E < 10.
  E  The accuracy knob: sensitivity against candidate depth.
  F  Why predicted queries fail: sensitivity against query pLDDT.
  G  Whether there is anything to find: how often a sequence-invisible homologue
     is structurally similar, by target pLDDT.
  H  Who finds what: every same-family positive of the experimental queries,
     split into found by sequence only, by structure only, by both, or by
     neither, by target pLDDT.
  I  The cost of crossing species, for predicted queries.

Detection in G means the target is among a method's first 200 evaluable hits;
sequence detection additionally requires E < 0.01. Means carry 95% intervals
from the hierarchical block bootstrap over Rfam families, fractions Wilson
intervals.

Reads per_query.json (written by `evaluate.py --band`) plus the raw runs.
Writes figures/fig4.png and files/predicted_retrieval/fig4_data.json.
"""
import csv
import json
import os
from collections import defaultdict

import numpy as np

import evaluate as E
import figstyle as S
from figstyle import plt

HERE, ROOT = E.HERE, E.ROOT
FIG = os.path.join(E.REPO, "figures")
# Every structure method is held at the same candidate depth, 500, so that the
# comparison is of the methods and not of how deep each was allowed to look.
DEPTH = 500
METHODS = ["RFoldseek n500", "RFoldseek-TM n500", "RiboSeek c500", "BLASTN", "nhmmer", "Infernal",
           "US-align (exhaustive)"]
BAR_METHODS = [m for m in METHODS if m != "US-align (exhaustive)"]
TREND = ["RFoldseek n500", "RFoldseek-TM n500", "nhmmer", "US-align (exhaustive)"]
DEPTHS = [("RFoldseek", [("RFoldseek n200", 200), ("RFoldseek n500", 500),
                         ("RFoldseek n2000", 2000)], "RFoldseek n500"),
          ("RFoldseek-TM", [("RFoldseek-TM n200", 200), ("RFoldseek-TM n500", 500)],
           "RFoldseek-TM n500"),
          ("RiboSeek", [("RiboSeek c200", 200), ("RiboSeek c500", 500)], "RiboSeek c500")]
METRIC = "Sens@1FP"
MAIN_FILTER = "E<0.01"
FILTERS = [("none", "All"), ("E<0.01", "$E<0.01$"), ("E<10", "$E<10$")]
QS_NAME = {"exp": "Experimental queries", "p2p": "Predicted queries"}
BINS = [(0, 70), (70, 80), (80, 90), (90, 101)]
BIN_LABEL = ["<70", "70-79", "80-89", "$\\geq$90"]
STRUCT_TRUTH = f"structure t{E.TM_MAIN:.2f}"
# G: hue follows the method that finds the target, so the panel reads against
# the same legend as the rest of the figure.
# categorical, deliberately NOT the method colours: purple = structure-only
# (the headline), orange = sequence-only, greys = both / neither.
BUCKETS = [("both", "Both", "#4D4D4D"),
           ("seq", "Sequence only", "#D9822B"),
           ("str", "Structure only", "#7B5EA7"),
           ("none", "Neither", "#E6E6E6")]


def collect():
    """Target-level facts the per-query metrics cannot give: the invisible
    homologues' structural similarity (F), the detection split (G), and the
    same-species comparison (H)."""
    rows = list(csv.DictReader(open(os.path.join(ROOT, "sample", "accuracy_db.tsv")), delimiter="\t"))
    n = len(rows)
    fam = np.array([r["rfam_family"] if r["part"] == "labelled" else "" for r in rows], dtype=object)
    tax = np.array([r["taxonomy_id"] for r in rows], dtype=object)
    urs = np.array([r["urs_id"].split("_")[0] for r in rows], dtype=object)
    plddt = np.array([float(r["avg_plddt"]) if r["avg_plddt"] not in ("", "N/A") else np.nan
                      for r in rows])
    row_of_file = {f"{r['urs_id']}_rank_1.pdb": i for i, r in enumerate(rows)}

    out = {"invisible": {}, "buckets": {b: [0] * len(BINS) for b, _, _ in BUCKETS},
           "species": defaultdict(dict), "families_p2p": {}}
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
        inv = {b: [0, 0] for b in BINS}
        for q in queries:
            excl = (urs == q["urs"]) if q["urs"] else np.zeros(n, bool)
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
            if qs == "exp":                                   # panel G (who-finds)
                str_hit = np.zeros(n, bool)
                str_hit[[i for i in runs[f"RFoldseek n{DEPTH}"][q["id"]]["ranking"][:200]
                         if not excl[i]]] = True
                # sequence-detectable = the strongest sequence baseline available:
                # BLASTN or nhmmer at E<0.01, OR any target Infernal (a profile CM,
                # the Rfam standard) retrieved. Using the strongest sequence union
                # makes "structure only" the most conservative possible estimate.
                seq_hit = seen01.copy()
                if "Infernal" in runs:
                    seq_hit[[i for i in runs["Infernal"][q["id"]]["ranking"] if not excl[i]]] = True
                for i, b in enumerate(BINS):
                    m = pos & (plddt >= b[0]) & (plddt < b[1])
                    s_, t_ = seq_hit & m, str_hit & m
                    out["buckets"]["both"][i] += int(np.sum(s_ & t_))
                    out["buckets"]["seq"][i] += int(np.sum(s_ & ~t_))
                    out["buckets"]["str"][i] += int(np.sum(t_ & ~s_))
                    out["buckets"]["none"][i] += int(np.sum(m & ~s_ & ~t_))
            if qs == "p2p":                                   # panel H
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


def main():
    os.makedirs(FIG, exist_ok=True)
    pq = json.load(open(os.path.join(HERE, "per_query.json")))
    fams = pq["families"]
    data = collect()
    qplddt = {q["urs_id"]: q["plddt"] for q in
              json.load(open(os.path.join(ROOT, "sample", "p2p_queries.json")))["queries"]}

    def est(qs, truth, filt, method, ids=None):
        d = pq.get(qs, {}).get(f"{truth}|{filt}", {}).get(method, {})
        v = {q: x[METRIC] for q, x in d.items() if ids is None or q in ids}
        if not v:
            return np.nan, 0.0, 0.0, 0
        mean, lo, hi = E.block_boot(v, fams)
        return mean, max(mean - lo, 0), max(hi - mean, 0), len(v)

    # reading order follows the text: two rulers then the filter sweep (row 1),
    # the pLDDT-gated trio (row 2), then the distribution detail and the general
    # cross-species cost (row 3). Keys are bound to content; letters are assigned
    # by position below. C = survival (old A detail), D = filter.
    layout = [["A", "A", "B", "B", "G", "G"],
              ["F", "F", "E", "E", "D", "D"],
              ["C", "C", "C", "H", "H", "H"]]
    fig, axd = plt.subplot_mosaic(layout, figsize=(14.4, 12.6))
    order = ["exp", "p2p"]

    def qtitle(ax, text):
        ax.set_title(text, fontsize=11, pad=6, color="black")

    def regime_panel(ax, truth, oracle_as_ceiling, title):
        # No per-bar numeric labels (values read from the axis / Supplementary
        # Table) so no method is singled out. Under structure truth US-align
        # equals 1.0 by construction; rather than stretch the axis to it, the
        # axis stays at the data's range and the oracle value is annotated.
        methods = BAR_METHODS if oracle_as_ceiling else METHODS
        width = 0.82 / len(methods)
        x = np.arange(2)
        for i, m in enumerate(methods):
            off = (i - (len(methods) - 1) / 2) * width
            e = [est(qs, truth, MAIN_FILTER, m) for qs in order]
            S.bars(ax, x + off, [v[0] for v in e], width, m,
                   yerr=[[v[1] for v in e], [v[2] for v in e]])
        top = 0.62 if oracle_as_ceiling else 0.75
        if oracle_as_ceiling:
            ax.text(0.5, 0.96, "US-align oracle $=$ 1.00", ha="center", va="top",
                    transform=ax.transAxes, fontsize=8.5,
                    color=S.TEXT["US-align (exhaustive)"])
        ax.axvline(0.5, color="#DDDDDD", lw=0.9, zorder=1)   # experimental | predicted
        ax.set_xticks(x, [QS_NAME[qs].split()[0] for qs in order])
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(0, top)
        S.style_axes(ax, xlabel="Query structures", ylabel=METRIC)
        qtitle(ax, title)

    # ---- A, B: the two ground truths ---------------------------------------
    # A (annotation truth): label RFoldseek and, via the ceiling branch, US-align
    # (a competing method here). B (structure truth): label RFoldseek-TM, whose
    # predicted-query bar (0.46) is the headline flip; US-align is the ceiling.
    regime_panel(axd["A"], "label", False, "Annotation truth")
    regime_panel(axd["B"], STRUCT_TRUTH, True, "Structure truth")

    # ---- C: the distribution behind A, as in Fig. 2A,B ---------------------
    ax = axd["C"]
    for m in METHODS:
        d = pq["exp"][f"label|{MAIN_FILTER}"].get(m)
        if not d:
            continue
        y = np.sort(np.array([v[METRIC] for v in d.values()]))[::-1]
        x = np.arange(1, len(y) + 1) / len(y)
        step = max(1, len(x) // 18)
        S.line(ax, x, y, m, markevery=step)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    S.style_axes(ax, xlabel="Fraction of queries", ylabel=METRIC)
    qtitle(ax, "Experimental queries")

    # ---- D: the sequence filter as an axis ---------------------------------
    ax = axd["D"]
    width = 0.82 / len(METHODS)
    x = np.arange(len(FILTERS))
    ns = [est("exp", "label", f, METHODS[0])[3] for f, _ in FILTERS]
    for i, m in enumerate(METHODS):
        off = (i - (len(METHODS) - 1) / 2) * width
        e = [est("exp", "label", f, m) for f, _ in FILTERS]
        S.bars(ax, x + off, [v[0] for v in e], width, m,
               yerr=[[v[1] for v in e], [v[2] for v in e]])
    ax.axvline(0.5, color="#DDDDDD", lw=0.9, zorder=1)
    ax.set_xticks(x, [lab for _, lab in FILTERS])
    ax.set_xlim(-0.5, len(FILTERS) - 0.5)
    ax.set_ylim(0, 0.75)
    S.style_axes(ax, xlabel="Positives counted", ylabel=METRIC)
    qtitle(ax, "Experimental queries")

    # ---- E: query confidence -----------------------------------------------
    ax = axd["E"]
    ns = []
    for m in TREND:
        mids, los, his, ns = [], [], [], []
        for b in BINS:
            ids = {q for q in qplddt if b[0] <= qplddt[q] < b[1]}
            mean, lo, hi, k = est("p2p", "label", "none", m, ids=ids)
            ns.append(k)
            mids.append(mean); los.append(lo); his.append(hi)
        S.line(ax, range(len(BINS)), mids, m, yerr=[los, his])
    ax.set_xticks(range(len(BINS)), BIN_LABEL)
    ax.set_xlim(-0.3, len(BINS) - 0.7)
    ax.set_ylim(0, 0.42)
    S.style_axes(ax, xlabel="Query pLDDT", ylabel=METRIC)
    qtitle(ax, "Predicted queries")

    # ---- F: is there anything to find? -------------------------------------
    ax = axd["F"]
    tick = {}
    # Query sets are told apart by colour + marker shape + line style, both markers
    # filled -- open vs filled markers are reserved for panel H (species).
    for qs, colr, mk, ls, xoff in (("exp", S.NEUTRAL_DARK, "o", "-", 0.03),
                                   ("p2p", S.NEUTRAL_LIGHT, "s", "--", -0.03)):
        ks = [data["invisible"][qs][f"{b[0]}-{b[1]}"][0] for b in BINS]
        nn = [data["invisible"][qs][f"{b[0]}-{b[1]}"][1] for b in BINS]
        e = [S.wilson(k, t) for k, t in zip(ks, nn)]
        ax.errorbar(np.arange(len(BINS)) + xoff, [v[0] for v in e],
                    yerr=[[v[0] - v[1] for v in e], [v[2] - v[0] for v in e]],
                    fmt=ls + mk, color=colr, ecolor=colr, lw=2.4,
                    markersize=8, markerfacecolor=colr, markeredgecolor="white",
                    markeredgewidth=0.9, elinewidth=0.9, capsize=2.5, zorder=3,
                    label=QS_NAME[qs])
        tick[qs] = nn
    ax.set_xticks(range(len(BINS)), BIN_LABEL)
    ax.set_xlim(-0.4, len(BINS) - 0.6)
    ax.set_ylim(0, 1.08)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    S.style_axes(ax, xlabel="Target pLDDT",
                 ylabel=f"Fraction with TM $\\geq$ {E.TM_MAIN}")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    # ---- G: who finds what -------------------------------------------------
    ax = axd["G"]
    tot = np.array([sum(data["buckets"][b][i] for b, _, _ in BUCKETS) for i in range(len(BINS))],
                   dtype=float)
    bottom = np.zeros(len(BINS))
    x = np.arange(len(BINS))
    for key, label, colr in BUCKETS:
        frac = np.array(data["buckets"][key], dtype=float) / np.where(tot > 0, tot, 1)
        ax.bar(x, frac, 0.62, bottom=bottom, color=colr, edgecolor="#666666", linewidth=0.6,
               label=label, zorder=3)
        for i, f in enumerate(frac):
            if f > 0.045:
                ax.text(i, bottom[i] + f / 2, f"{f:.0%}", ha="center", va="center",
                        fontsize=8, color="white" if key in ("both", "seq", "str") else "#555555")
        bottom += frac
    ax.set_xticks(x, BIN_LABEL)
    ax.set_xlim(-0.55, len(BINS) - 0.45)
    ax.set_ylim(0, 1.22)                      # white band so the legend clears the bars
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    S.style_axes(ax, xlabel="Target pLDDT", ylabel="Fraction of positives")
    qtitle(ax, "Experimental queries")
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2, columnspacing=1.2,
              handletextpad=0.6, handlelength=1.4, bbox_to_anchor=(0.5, 1.00))

    # ---- H: cost of crossing species ---------------------------------------
    ax = axd["H"]
    both = sorted(set(data["species"]["cross"]["RFoldseek n500"])
                  & set(data["species"]["same"]["RFoldseek n500"]))
    ypos = np.arange(len(METHODS))[::-1]
    for y, m in zip(ypos, METHODS):
        cross, same = data["species"]["cross"].get(m, {}), data["species"]["same"].get(m, {})
        ids = [q for q in both if q in cross and q in same]
        if not ids:
            continue
        s0 = float(np.mean([same[q] for q in ids]))
        c0 = float(np.mean([cross[q] for q in ids]))
        d, lo, hi = E.block_boot({q: cross[q] - same[q] for q in ids}, data["families_p2p"])
        ax.plot([s0, c0], [y, y], color=S.COLOR[m], lw=2.6, solid_capstyle="round", zorder=2)
        ax.scatter([s0], [y], s=42, marker=S.MARKER[m], facecolor="white",
                   edgecolor=S.COLOR[m], linewidth=1.1, zorder=3)
        ax.scatter([c0], [y], s=42, marker=S.MARKER[m], facecolor=S.COLOR[m],
                   edgecolor="white", linewidth=0.7, zorder=4)
        ax.annotate(f"{d:+.2f}{'*' if not (lo <= 0 <= hi) else ''}", (max(s0, c0), y),
                    xytext=(9, -3), textcoords="offset points", fontsize=8.5, color=S.TEXT[m])
    ax.set_yticks(ypos, [S.SHORT[m] for m in METHODS], fontsize=9)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.6, len(METHODS) + 0.6)
    S.style_axes(ax, xlabel=METRIC)
    qtitle(ax, "Predicted queries")
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", markerfacecolor="white",
                                  markeredgecolor=S.GREY_TEXT, color=S.GREY_TEXT,
                                  label="Same species"),
                       plt.Line2D([], [], marker="o", ls="", color=S.GREY_TEXT,
                                  markeredgecolor="white", label="Different species")],
              frameon=False, fontsize=8.5, loc="upper center", ncol=2, columnspacing=1.2,
              handletextpad=0.6, bbox_to_anchor=(0.5, 1.02))

    S.panel_letters([axd[k] for k in ["A", "B", "G", "F", "E", "D", "C", "H"]],
                    "ABCDEFGH", x=-0.08, y=1.03)
    S.shared_legend(fig, METHODS, ncol=7, y=-0.004)
    fig.tight_layout(rect=(0, 0.05, 1, 1), pad=0.5, w_pad=0.9, h_pad=1.6)
    for ext in ("png",):
        fig.savefig(os.path.join(FIG, f"fig4.{ext}"), dpi=200, bbox_inches="tight")

    out = {"metric": METRIC, "main_filter": MAIN_FILTER, "candidate_depth": DEPTH,
           "invisible_by_plddt": data["invisible"], "detection_buckets": data["buckets"],
           "caption_counts": {
               "A,B,D queries": {f"{qs}|{truth}|{filt}":
                                 est(qs, truth, filt, METHODS[0])[3]
                                 for qs in order for truth in ("label", STRUCT_TRUTH)
                                 for filt, _ in FILTERS},
               "F query pLDDT bins": {f"{b[0]}-{b[1]}":
                                      len([q for q in qplddt if b[0] <= qplddt[q] < b[1]])
                                      for b in BINS},
               "G invisible targets": data["invisible"],
               "H positives per bin": {f"{b[0]}-{b[1]}":
                                       sum(data["buckets"][k][i] for k, _, _ in BUCKETS)
                                       for i, b in enumerate(BINS)},
               "I paired queries": len(both)}}
    for qs in order:
        for truth in ("label", STRUCT_TRUTH):
            for filt, _ in FILTERS:
                key = f"{qs}|{truth}|{filt}"
                out[key] = {}
                for m in METHODS:
                    mean, lo, hi, k = est(qs, truth, filt, m)
                    if k:
                        out[key][m] = {"n": k, "mean": mean, "lo": mean - lo, "hi": mean + hi}
    json.dump(out, open(os.path.join(HERE, "fig4_data.json"), "w"), indent=1)
    print("wrote", os.path.join(FIG, "fig4.png"))


if __name__ == "__main__":
    main()
