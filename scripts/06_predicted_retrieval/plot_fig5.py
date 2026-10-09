"""Main figure for the predicted-structure benchmark, and the numbers behind it."""
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

    # Reading order follows the text: overall retrieval (A, B), prediction
    # confidence (C-E), species (F), complementarity with sequence search (G, H).
    # B is split into two stacked panels (TM-score truth above, RMscore truth below);
    # every other row spans two grid rows so that the panel heights are unchanged.
    layout = [["A", "A", "B", "B", "C", "C"],
              ["A", "A", "B2", "B2", "C", "C"],
              ["D", "D", "E", "E", "F", "F"],
              ["D", "D", "E", "E", "F", "F"],
              ["G", "G", "G", "G", "H", "H"],
              ["G", "G", "G", "G", "H", "H"]]
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
        top = 0.66 if oracle_as_ceiling else 0.9
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
    # A (annotation truth): label RFoldSearch and, via the ceiling branch, US-align
    # (a competing method here). B (structure truth): label RFoldSearch-TM, whose
    # predicted-query bar (0.53) is the headline flip; US-align is the ceiling.
    regime_panel(axd["A"], "label", False, "Annotation truth")

    # B: the two structure truths, stacked. Each oracle (US-align for TM-score,
    # RMalign for RMscore) is 1.00 by construction and annotated instead of drawn;
    # under the RMscore truth US-align is a competing method and is drawn.
    for key, truth, shown, oracle, title in (
            ("B", STRUCT_TRUTH, BAR_METHODS, "US-align (exhaustive)", "Structure truth (TM-score)"),
            ("B2", RM_TRUTH, METHODS, "RMalign (exhaustive)", "Structure truth (RMscore)")):
        ax = axd[key]
        width = 0.82 / len(METHODS)
        for g, qs in enumerate(order):
            for i, m in enumerate(shown):
                mean, lo, hi, _ = est(qs, truth, MAIN_FILTER, m)
                S.bars(ax, [g + (i - (len(shown) - 1) / 2) * width], [mean], width, m, yerr=[[lo], [hi]])
        S.ceiling(ax, 1.0, -0.5, 1.5, oracle)                 # Fig. 2's oracle line, in the oracle's colour
        ax.axvline(0.5, color="#DDDDDD", lw=0.9, zorder=1)   # experimental | predicted
        ax.set_xticks(range(2), [QS_NAME[qs].split()[0] for qs in order])
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(0, 1.18)
        ax.set_yticks(np.arange(0, 1.01, 0.5))
        S.style_axes(ax, xlabel="Query structures" if key == "B2" else None, ylabel=METRIC)
        qtitle(ax, title)

    # ---- C, D: the two ground truths by query pLDDT -------------------------
    # Predicted queries only (experimental queries carry no pLDDT); all positives
    # counted. A bin with fewer than MIN_BIN queries is not drawn.
    def plddt_panel(ax, truth, methods, title):
        for m in methods:
            mids, los, his = [], [], []
            for b in BINS:
                ids = {q for q in qplddt if b[0] <= qplddt[q] < b[1]}
                mean, lo, hi, k = est("p2p", truth, "none", m, ids=ids)
                if k < MIN_BIN:
                    mean, lo, hi = np.nan, 0.0, 0.0
                mids.append(mean); los.append(lo); his.append(hi)
            S.line(ax, range(len(BINS)), mids, m, yerr=[los, his])
        ax.set_xticks(range(len(BINS)), BIN_LABEL)
        ax.set_xlim(-0.3, len(BINS) - 0.7)
        ax.set_ylim(0, 1.02)
        S.style_axes(ax, xlabel="Query pLDDT", ylabel=METRIC)
        qtitle(ax, title)

    plddt_panel(axd["C"], "label", PLDDT_METHODS + ["US-align (exhaustive)"],
                "Annotation truth")
    plddt_panel(axd["D"], STRUCT_TRUTH, PLDDT_METHODS, "Structure truth")

    # ---- G: the sequence filter as an axis, both query sets ----------------
    # Structure truth, so that "what structure search adds" is measured against
    # structural homology for both query sets; US-align, the oracle, is omitted.
    ax = axd["G"]
    width = 0.82 / len(BAR_METHODS)
    gap = 0.6
    groups = [(qs, (len(FILTERS) + gap) * k + np.arange(len(FILTERS))) for k, qs in enumerate(order)]
    for i, m in enumerate(BAR_METHODS):
        off = (i - (len(BAR_METHODS) - 1) / 2) * width
        for qs, x in groups:
            e = [est(qs, STRUCT_TRUTH, f, m) for f, _ in FILTERS]
            S.bars(ax, x + off, [v[0] for v in e], width, m,
                   yerr=[[v[1] for v in e], [v[2] for v in e]])
    ax.axvline(len(FILTERS) - 0.5 + gap / 2, color="#DDDDDD", lw=0.9, zorder=1)
    for qs, x in groups:
        ax.text(x.mean(), 0.66, QS_NAME[qs], ha="center", va="bottom", fontsize=9, color=S.GREY_TEXT)
    ax.set_xticks(np.concatenate([x for _, x in groups]), [lab for _, lab in FILTERS] * len(groups))
    ax.set_xlim(-0.5, groups[-1][1][-1] + 0.5)
    ax.set_ylim(0, 0.7)
    S.style_axes(ax, ylabel=METRIC)
    qtitle(ax, "Structure truth")

    # ---- E: target confidence, both truths ----------------------------------
    # Experimental queries, so that the query carries no prediction error and the
    # trend is the targets' alone. Left block annotation truth, right block
    # structure truth (US-align, the oracle there, is omitted).
    ax = axd["E"]
    fams_exp = {q: f for q, f in fams.items()}
    blocks = [("label", 0), (STRUCT_TRUTH, len(BINS) + 1)]
    e_values = defaultdict(dict)
    for truth, x0 in blocks:
        for m in TARGET_METHODS:
            if truth != "label" and m == "US-align (exhaustive)":
                continue
            mids, los, his = [], [], []
            for b in BINS:
                v = data["target_bins"][truth][f"{b[0]}-{b[1]}"].get(m, {})
                if len(v) < MIN_BIN:
                    mids.append(np.nan); los.append(0.0); his.append(0.0)
                    continue
                mean, lo, hi = E.block_boot(v, fams_exp)
                e_values[f"{truth}|{b[0]}-{b[1]}"][m] = {"n": len(v), "mean": mean, "lo": lo, "hi": hi}
                mids.append(mean); los.append(max(mean - lo, 0)); his.append(max(hi - mean, 0))
            S.line(ax, x0 + np.arange(len(BINS)), mids, m, yerr=[los, his])
    ax.axvline(len(BINS), color="#DDDDDD", lw=0.9, zorder=1)
    for (truth, x0), name in zip(blocks, ("Annotation truth", "Structure truth")):
        ax.text(x0 + (len(BINS) - 1) / 2, 1.0, name, ha="center", va="bottom", fontsize=9,
                color=S.GREY_TEXT)
    ax.set_xticks(np.r_[np.arange(len(BINS)), len(BINS) + 1 + np.arange(len(BINS))],
                  BIN_LABEL + BIN_LABEL, fontsize=8.5)
    ax.set_xlim(-0.5, 2 * len(BINS) + 0.5)
    ax.set_ylim(0, 1.06)
    S.style_axes(ax, xlabel="Target pLDDT", ylabel=METRIC)
    qtitle(ax, "Experimental queries")

    # ---- F: cost of crossing species ---------------------------------------
    ax = axd["F"]
    both = sorted(set(data["species"]["cross"]["RFoldSearch n500"])
                  & set(data["species"]["same"]["RFoldSearch n500"]))
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

    # ---- H: recovery of the homologues sequence search misses ----------------
    # Rfam truth, which is independent of every structure score. Rate = recovered
    # / missed, pooled over queries; 95% interval from resampling families.
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

    # Every query set and detector is written to fig5_data.json; the panel
    # shows experimental queries (predicted queries have too few sequence-missed
    # homologues, from one or two families per bin).
    h_values = {}
    for qs in order:
        for w in WHO:
            for i, b in enumerate(BINS):
                v = ratio_boot(data["recover"][qs][w][i])
                h_values.setdefault(qs, {}).setdefault(f"{b[0]}-{b[1]}", {})[w] = dict(
                    rate=v[0], lo=v[0] - v[1], hi=v[0] + v[2], missed=v[3], families_recovered=v[4])
    ax = axd["H"]
    shown = WHO
    width = 0.8 / len(shown)
    x = np.arange(len(BINS))
    for j, w in enumerate(shown):
        v = [h_values["exp"][f"{b[0]}-{b[1]}"][w] for b in BINS]
        S.bars(ax, x + (j - (len(shown) - 1) / 2) * width, [d["rate"] for d in v], width, w,
               yerr=[[d["rate"] - d["lo"] for d in v], [d["hi"] - d["rate"] for d in v]])
    ax.set_xticks(x, BIN_LABEL)
    ax.set_xlim(-0.5, len(BINS) - 0.5)
    ax.set_ylim(0, 0.55)
    S.style_axes(ax, xlabel="Target pLDDT", ylabel="Fraction recovered")
    qtitle(ax, "Homologues missed by sequence search")

    S.shared_legend(fig, METHODS, ncol=4, y=-0.004)
    fig.tight_layout(rect=(0, 0.05, 1, 1), pad=0.5, w_pad=0.9, h_pad=1.6)
    # Letters sit a fixed distance left of each axis, so that panels of different
    # widths (G is wider than A and D) keep their letters in one column.
    w, h = fig.get_size_inches()
    for k in "ABCDEFGH":
        pos = axd[k].get_position()
        fig.text(pos.x0 - 0.62 / w, pos.y1 + 0.1 / h, k, fontsize=15, fontweight="bold",
                 va="bottom", ha="right")
    for ext in ("png",):
        fig.savefig(os.path.join(FIG, f"fig5.{ext}"), dpi=200, bbox_inches="tight")

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
    print("wrote", os.path.join(FIG, "fig5.png"))


if __name__ == "__main__":
    main()
