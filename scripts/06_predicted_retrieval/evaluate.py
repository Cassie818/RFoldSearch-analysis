"""Evaluation of the sampled NuFold 9-14-2026 benchmark.

Query sets
  p2p  predicted queries; CROSS-SPECIES label truth: a positive shares the
       query's Rfam family and comes from a different organism; same-family
       targets from the query organism and every entry with the query's
       RNAcentral sequence (URS, including the query itself) are neutral.
  exp  experimental queries; positives share the query's Rfam family. The
       queries are not in the benchmark, but targets whose sequence is
       identical to the query's are query-equivalent and neutral.
  exp_unlabelled  experimental queries without an Rfam family; structure truth
       only, each query its own bootstrap block.

"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
import sys as _sys                                             # noqa: E402
_sys.path.insert(0, REPO)
import paths                                                   # noqa: E402
HERE = os.path.join(REPO, "files", "predicted_retrieval")      # results; reruns write db/, usalign/, runs/ here
ROOT = HERE                                                    # holds sample/
TOPN = 200
TM_TAUS = (0.40, 0.45, 0.50)     # Fig. 2 thresholds on the symmetric TM-score
TM_MAIN = 0.45                    # tau used for the pLDDT breakdown
METRICS = ("Recall@1", "Prec@10", "Sens@1FP", "AP")
METHODS = ["RFoldseek n500", "RFoldseek-TM n200",
           "RFoldseek-TM n500", "RiboSeek c200", "RiboSeek c500", "RiboSeek RS-80 c500",
           "BLASTN", "nhmmer",
           "Infernal", "US-align (exhaustive)", "RMalign (exhaustive)"]
ORACLE = {"structure": "US-align (exhaustive)", "rmscore": "RMalign (exhaustive)"}
PAIRS = [("RFoldseek n500", "RiboSeek c500"),
         ("RFoldseek n500", "RiboSeek RS-80 c500"),
         ("RFoldseek n500", "BLASTN"), ("RFoldseek n500", "nhmmer"),
         ("RFoldseek-TM n500", "nhmmer"), ("RFoldseek-TM n500", "RFoldseek n500"),
         ("US-align (exhaustive)", "RFoldseek-TM n500")]
PLDDT_BINS = [(0, 70), (70, 80), (80, 90), (90, 101)]


def metrics(ranking, relevant, neutral):
    n_rel = int(relevant.sum())
    if n_rel == 0:
        return None
    denom = min(n_rel, TOPN)
    ranked = [i for i in ranking if not neutral[i]]
    r1 = float(relevant[ranked[0]]) if ranked else 0.0
    top10 = ranked[:10]
    p10 = float(np.mean(relevant[top10])) if top10 else 0.0
    tp = 0
    for i in ranked:
        if relevant[i]:
            tp += 1
        else:
            break
    hits, prec = 0, 0.0
    for k, i in enumerate(ranked[:TOPN], 1):
        if relevant[i]:
            hits += 1
            prec += hits / k
    return {"Recall@1": r1, "Prec@10": p10, "Sens@1FP": min(tp, denom) / denom, "AP": prec / denom}


def load_usalign(path, row_of_file, n):
    tm1 = np.full(n, np.nan)
    tm2 = np.full(n, np.nan)
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        f = line.rstrip("\n").split("\t")
        j = row_of_file.get(os.path.basename(f[1]).split(":")[0])
        if j is not None:
            tm1[j], tm2[j] = float(f[2]), float(f[3])
    return tm1, tm2


def load_rmalign(path, row_of_file, n):
    rq = np.full(n, np.nan)
    rt = np.full(n, np.nan)
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        j = row_of_file.get(f[0])
        if j is not None:
            rq[j], rt[j] = float(f[1]), float(f[2])
    return rq, rt


def rmscore_thresholds(row_of_file, n):
    """RMscore thresholds marking the same fraction of query-target pairs as each TM tau,
    over every query with both exhaustive outputs."""
    tms, rms = [], []
    for qs in ("p2p", "exp", "exp_unlabelled"):
        d = os.path.join(HERE, "rmalign", qs)
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            us = os.path.join(HERE, "usalign", qs, f)
            if not f.endswith(".tsv") or not os.path.exists(us):
                continue
            tm1, tm2 = load_usalign(us, row_of_file, n)
            rq, rt = load_rmalign(os.path.join(d, f), row_of_file, n)
            ok = ~np.isnan(tm1) & ~np.isnan(rq)
            tms.append(sym_tm(tm1, tm2)[ok])
            rms.append(sym_tm(rq, rt)[ok])
    if not tms:
        return {}
    tm, rm = np.concatenate(tms), np.concatenate(rms)
    return {tau: float(np.quantile(rm, 1 - np.mean(tm >= tau))) for tau in TM_TAUS}


def sym_tm(tm1, tm2):
    """Symmetric TM-score: mean of the query- and target-normalised scores."""
    return 0.5 * (np.nan_to_num(tm1) + np.nan_to_num(tm2))


def structure_truth(tm1, tm2, excl, tau):
    """Fig. 2 structural truth: symmetric TM >= tau is positive, everything
    else negative; unscored targets and the query's own URS are neutral."""
    valid = ~np.isnan(tm1)
    pos = (sym_tm(tm1, tm2) >= tau) & valid & ~excl
    return pos, (~valid) | excl


def identical_targets(queries):
    """For experimental queries (not in the benchmark): {qid: indices of targets
    whose sequence is identical to the query's}."""
    from run_searches import DB, NR80, seq_from_cif
    by_seq = defaultdict(list)
    seqs = [l.strip().upper().replace("T", "U") for l in open(os.path.join(DB, "sequences.fa"))
            if not l.startswith(">")]
    for i, s in enumerate(seqs):
        by_seq[s].append(i)
    return {q["id"]: by_seq.get(seq_from_cif(os.path.join(NR80, q["id"] + ".cif")), [])
            for q in queries}


def block_boot(values_by_query, families, reps=2000, seed=0):
    """Hierarchical block bootstrap of the mean: resample families, keep all
    their queries. values_by_query: {qid: value}. Returns (mean, lo, hi)."""
    rng = np.random.default_rng(seed)
    byfam = defaultdict(list)
    for q, v in values_by_query.items():
        byfam[families[q]].append(v)
    fams = list(byfam)
    point = float(np.mean(list(values_by_query.values())))
    boots = []
    for _ in range(reps):
        vals = []
        for i in rng.integers(0, len(fams), len(fams)):
            vals.extend(byfam[fams[i]])
        boots.append(np.mean(vals))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def fmt_ci(t):
    m, lo, hi = t
    return f"{m:+.3f}{'*' if not (lo <= 0 <= hi) else ' '}[{lo:+.3f},{hi:+.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", action="store_true", help="label truth only")
    ap.add_argument("--band", action="store_true",
                    help="also report the query-normalised TM_Q truth with a 0.30-0.45 neutral band")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(ROOT, "sample", "accuracy_db.tsv")), delimiter="\t"))
    n = len(rows)
    fam = np.array([r["rfam_family"] if r["part"] == "labelled" else "" for r in rows], dtype=object)
    tax = np.array([r["taxonomy_id"] for r in rows], dtype=object)
    urs = np.array([r["urs_id"].split("_")[0] for r in rows], dtype=object)
    plddt = np.array([float(r["avg_plddt"]) if r["avg_plddt"] not in ("", "N/A") else np.nan for r in rows])
    row_of = {r["urs_id"]: i for i, r in enumerate(rows)}
    row_of_file = {f"{r['urs_id']}_rank_1.pdb": i for i, r in enumerate(rows)}

    rthr = {} if args.label else rmscore_thresholds(row_of_file, n)
    for tau, r in rthr.items():
        print(f"TM >= {tau:.2f} <-> RMscore >= {r:.3f}")
    report = {"rmscore_thresholds": {f"{t:.2f}": r for t, r in rthr.items()}}
    per_query = {"families": {}}                       # qs -> "truth|filter" -> method -> qid -> metrics
    struct_pool = defaultdict(lambda: defaultdict(dict))  # experimental sets pooled, structure truth
    pool_families = {}
    for qs in ("p2p", "exp", "exp_unlabelled"):
        runs_path = os.path.join(HERE, "runs", f"{qs}.json")
        if not os.path.exists(runs_path):
            print(f"\n(no runs for {qs}, skipped)")
            continue
        runs = json.load(open(runs_path))
        if qs == "p2p":
            qlist = json.load(open(os.path.join(ROOT, "sample", "p2p_queries.json")))["queries"]
            queries = [dict(id=q["urs_id"], family=q["family"], taxon=q["taxonomy_id"],
                            urs=q["urs_id"].split("_")[0]) for q in qlist]
        elif qs == "exp":
            qlist = json.load(open(os.path.join(ROOT, "sample", "exp_queries.json")))["queries"]
            queries = [dict(id=q["id"], family=q["family"], taxon=None, urs=None) for q in qlist]
        else:
            qlist = json.load(open(os.path.join(ROOT, "sample", "exp_unlabelled_queries.json")))
            # no Rfam family: no label truth, and each query is its own bootstrap block
            queries = [dict(id=q["id"], family=q["id"], taxon=None, urs=None) for q in qlist]
        families = {q["id"]: q["family"] for q in queries}
        ident = identical_targets(queries) if qs != "p2p" else {}
        print(f"{qs}: {sum(map(len, ident.values()))} sequence-identical targets neutralised")
        pool_families.update(families)
        per_query["families"].update(families)

        per = defaultdict(lambda: defaultdict(dict))     # (truth, filter) -> method -> qid -> metrics
        invisible = defaultdict(lambda: [0, 0])           # pLDDT bin -> [TM-similar, total] (label positives, E<=10 filter)
        n_struct = 0
        for q in queries:
            qid = q["id"]
            excl = (urs == q["urs"]) if q["urs"] else np.zeros(n, bool)
            if q["urs"] is None:
                excl[ident[qid]] = True
            same = (fam == q["family"]) & ~excl
            if q["taxon"] is not None:
                label_pos = same & (tax != q["taxon"])
                label_neu = (fam == "") | excl | (same & (tax == q["taxon"]))
            else:
                label_pos, label_neu = same, (fam == "")
            seen01 = np.zeros(n, bool)
            seen10 = np.zeros(n, bool)
            for tool in ("BLASTN", "nhmmer"):
                r = runs[tool][qid]
                seen10[r["ranking"]] = True
                seen01[[i for i, e in zip(r["ranking"], r["evalue"]) if e < 0.01]] = True
            truths = {} if qs == "exp_unlabelled" else {"label": (label_pos, label_neu)}
            rank = {m: [i for i in runs[m][qid]["ranking"] if not excl[i]] for m in runs}

            us_path = os.path.join(HERE, "usalign", qs, f"{qid}.tsv")
            if not args.label and os.path.exists(us_path):
                n_struct += 1
                tm1, tm2 = load_usalign(us_path, row_of_file, n)
                for tau in TM_TAUS:
                    truths[f"structure t{tau:.2f}"] = structure_truth(tm1, tm2, excl, tau)
                if args.band:      # earlier variant, kept as a robustness check
                    truths["structure TM_Q band"] = (
                        (tm1 >= 0.45) & ~excl,
                        ((tm1 >= 0.30) & (tm1 < 0.45)) | np.isnan(tm1) | excl)
                mean_tm = sym_tm(tm1, tm2)
                mean_tm[excl] = -1
                rank["US-align (exhaustive)"] = [i for i in np.lexsort((np.arange(n), -mean_tm)) if not excl[i]]
                rm_path = os.path.join(HERE, "rmalign", qs, f"{qid}.tsv")
                if rthr and os.path.exists(rm_path):
                    rq, rt = load_rmalign(rm_path, row_of_file, n)
                    for tau, r in rthr.items():
                        truths[f"rmscore t{tau:.2f}"] = structure_truth(rq, rt, excl, r)
                    mean_rm = sym_tm(rq, rt)
                    mean_rm[excl] = -1
                    rank["RMalign (exhaustive)"] = [i for i in np.lexsort((np.arange(n), -mean_rm)) if not excl[i]]
                if qs != "exp_unlabelled":
                    inv = label_pos & ~seen10
                    for lo, hi in PLDDT_BINS:
                        m = inv & (plddt >= lo) & (plddt < hi)
                        invisible[(lo, hi)][0] += int(np.sum(sym_tm(tm1, tm2)[m] >= TM_MAIN))
                        invisible[(lo, hi)][1] += int(m.sum())

            for truth, (pos, neu) in truths.items():
                for filt, seen in (("none", None), ("E<0.01", seen01), ("E<=10", seen10)):
                    p, nu = (pos, neu) if seen is None else (pos & ~seen, neu | (pos & seen))
                    for m, r in rank.items():
                        res = metrics(r, p, nu)
                        if res is not None:
                            per[(truth, filt)][m][qid] = res
                            if qs.startswith("exp") and truth.startswith(("structure", "rmscore")):
                                struct_pool[(truth, filt)][m][qid] = res

        per_query[qs] = {f"{t}|{f}": {m: dict(v) for m, v in d.items()}
                         for (t, f), d in per.items()}
        print(f"\n{'#' * 100}\nQUERY SET {qs}: {len(queries)} queries, {len(set(families.values()))} families; "
              f"US-align available for {n_struct}\n{'#' * 100}")
        report[qs] = {}
        for (truth, filt), d in sorted(per.items()):
            ids_any = set().union(*[set(v) for v in d.values()]) if d else set()
            print(f"\n  truth={truth}  sequence filter={filt}: {len(ids_any)} evaluable queries "
                  f"from {len({families[i] for i in ids_any})} families")
            report[qs][f"{truth}|{filt}"] = {}
            for m in METHODS:
                pq = d.get(m)
                if not pq:
                    continue
                qmean = {k: float(np.mean([pq[i][k] for i in pq])) for k in METRICS}
                line = "  ".join(f"{k} {qmean[k]:.3f}" for k in METRICS)
                if qs == "exp":
                    byf = defaultdict(list)
                    for i in pq:
                        byf[families[i]].append(pq[i])
                    fm = {k: float(np.mean([np.mean([x[k] for x in v]) for v in byf.values()])) for k in ("Recall@1", "AP")}
                    line += f"   | family-macro Recall@1 {fm['Recall@1']:.3f} AP {fm['AP']:.3f}"
                tag = "  (near-circular)" if ORACLE.get(truth.split()[0]) == m else ""
                print(f"    {m:22s} n={len(pq):3d}  {line}{tag}")
                report[qs][f"{truth}|{filt}"][m] = dict(n=len(pq), mean=qmean)
            for a, b in PAIRS:
                if truth.startswith("structure") and a.startswith("US-align"):
                    continue
                pa, pb = d.get(a, {}), d.get(b, {})
                ids = [i for i in pa if i in pb]
                if len(ids) < 3:
                    continue
                cells = [f"{k} {fmt_ci(block_boot({i: pa[i][k] - pb[i][k] for i in ids}, families))}"
                         for k in ("Recall@1", "Sens@1FP", "AP")]
                print(f"      {a} - {b} (n={len(ids)}): " + "  ".join(cells))

        if any(v[1] for v in invisible.values()):
            print("\n  sequence-invisible label positives (no BLASTN/nhmmer hit at E<=10): "
                  f"fraction with symmetric TM >= {TM_MAIN} by target pLDDT")
            for (lo, hi), (sim, tot) in sorted(invisible.items()):
                name = f"<{hi}" if lo == 0 else f"{lo}-{hi-1}"
                print(f"    pLDDT {name:6s}: {sim:6d}/{tot:6d} = {sim/tot if tot else float('nan'):.3f}")

        print("\n  median search time per query")
        for m in METHODS:
            if m in runs:
                print(f"    {m:22s} {np.median([v['seconds'] for v in runs[m].values()])*1000:9.1f} ms")

    if struct_pool:
        print(f"\n{'#' * 100}\nEXPERIMENTAL QUERIES POOLED (annotated + unannotated), structure truth"
              f"\n{'#' * 100}")
        report["exp_pooled"] = {}
        for (truth, filt), d in sorted(struct_pool.items()):
            ids_any = set().union(*[set(v) for v in d.values()]) if d else set()
            print(f"\n  truth={truth}  sequence filter={filt}: {len(ids_any)} evaluable queries")
            report["exp_pooled"][f"{truth}|{filt}"] = {}
            for m in METHODS:
                pq = d.get(m)
                if not pq:
                    continue
                qmean = {k: float(np.mean([pq[i][k] for i in pq])) for k in METRICS}
                tag = "  (near-circular)" if ORACLE.get(truth.split()[0]) == m else ""
                print(f"    {m:22s} n={len(pq):3d}  " + "  ".join(f"{k} {qmean[k]:.3f}" for k in METRICS) + tag)
                report["exp_pooled"][f"{truth}|{filt}"][m] = dict(n=len(pq), mean=qmean)
            for a, b in PAIRS:
                if a.startswith("US-align"):
                    continue
                pa, pb = d.get(a, {}), d.get(b, {})
                ids = [i for i in pa if i in pb]
                if len(ids) < 3:
                    continue
                cells = [f"{k} {fmt_ci(block_boot({i: pa[i][k] - pb[i][k] for i in ids}, pool_families))}"
                         for k in ("Recall@1", "Sens@1FP", "AP")]
                print(f"      {a} - {b} (n={len(ids)}): " + "  ".join(cells))

    per_query["exp_pooled"] = {f"{t}|{f}": {m: dict(v) for m, v in d.items()}
                               for (t, f), d in struct_pool.items()}
    json.dump(report, open(os.path.join(HERE, "evaluation.json"), "w"), indent=1)
    json.dump(per_query, open(os.path.join(HERE, "per_query.json"), "w"))


if __name__ == "__main__":
    main()
