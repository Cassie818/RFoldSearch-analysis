"""Runtime ladder for the released RFoldSearch 1.0.2 package.

Re-measures the paper's Fig. 3 RFoldSearch curves with the package as
shipped, on databases re-encoded by that package. The protocol follows the
paper's scripts (verify_v3.py for RhoFold+, nufold_ladder.py for NuFold):

* RhoFold+: the 300 queries of db/queries_1000.json; every nested subset is
  the id list of the old pred_db_<n>.pt, and the query's own chain is
  excluded from its hits (self_idx), as before.
* NuFold: nested subsets are prefixes of one seeded shuffle; 300 native
  queries are drawn by length quantile and held out of every subset. Token
  lengths do not depend on the tokeniser, so the same chains are selected.
* One core (OMP/NUMBA/MKL threads = 1), compiled paths warmed before timing,
  --top_n 500, min_diag 2. Times cover the query path only; index
  construction is reported separately.

Modes: "sw" = prefilter + token Smith-Waterman (what the paper timed),
"tm_a" = the package default (structure score, S_SW x TM_a), "tm" =
RFoldSearch-TM (US-align over the candidate pool, full database only).

usage: bench_speed_v102.py rhofold|nufold DB.pt OUTDIR [--sizes a,b,..]
       [--modes sw,tm_a] [--tm-full] [--subset-ids DIR] [--queries FILE]
"""
import argparse
import json
import os
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

torch.set_num_threads(1)

from rfoldsearch.database import load_database
from rfoldsearch.prefilter import KmerPrefilter
from rfoldsearch.search import Searcher, load_matrix

RHO_SIZES = [1000, 5000, 10000, 20000, 50000, 83156]
NU_SIZES = [1000, 5000, 10000, 20000, 50000, 100000, 200000, 531888]
NU_SEED = 20260815
TOP_N, MIN_DIAG = int(os.environ.get("RFS_TOP_N", "500")), 2


def subset_searcher(db, ents, toks, ids, n_codes, sub):
    """A Searcher over an in-memory subset, without E-values (not timed here)."""
    s = Searcher.__new__(Searcher)
    s.db, s.ents, s.toks, s.ids, s.n_codes = db, ents, toks, ids, n_codes
    s.clusters = list(range(len(ids)))
    s.id2i = {cid: i for i, cid in enumerate(ids)}
    s.sub = sub
    t0 = time.perf_counter()
    s.prefilter = KmerPrefilter(toks, n_codes, sub)
    build_s = time.perf_counter() - t0
    s.n_clusters = len(ids)
    s.ev_par = s.power_ev = s.experimental_ev = s.predicted_ev = None
    s._tie_cache, s._flat = {}, None
    ix = s.prefilter.index
    index_mb = sum(v.nbytes for v in ix.values() if isinstance(v, np.ndarray)) / 1e6
    return s, build_s, index_mb


def nufold_queries(ids_all, toks_all):
    """nufold_ladder.py's query draw and nested order, verbatim."""
    order = np.random.default_rng(NU_SEED).permutation(len(toks_all))
    qn = 300
    lens_all = np.array([len(t) for t in toks_all])
    targets = np.quantile(lens_all[lens_all >= 20], np.linspace(0.02, 0.995, qn))
    rng = np.random.default_rng(NU_SEED + 1)
    qidx, used = [], set()
    for L in targets:
        cands = np.flatnonzero((lens_all >= L - 3) & (lens_all <= L + 3))
        cands = [int(c) for c in cands if int(c) not in used]
        if cands:
            j = int(rng.choice(cands))
            qidx.append(j)
            used.add(j)
    held = set(qidx)
    order = np.array([i for i in order if int(i) not in held])
    return qidx, order


def time_queries(s, queries, mode, usalign=None):
    """queries: list of (id, tokens, self_idx, entry). -> per-query seconds."""
    kw = dict(top_k=50, top_n=TOP_N, min_diag=MIN_DIAG, rerank=mode)
    if mode == "tm":
        kw["usalign"] = usalign
    if mode == "tm_a":
        s._flat_structures()             # one-off flattening, like a loaded server
    for _ in range(3):                   # warm every compiled path
        for _qid, qt, si, qe in queries[:3]:
            s.search(qt, self_idx=si, query_entry=qe, **kw)
    out = []
    for qid, qt, si, qe in queries:
        t0 = time.perf_counter()
        hits = s.search(qt, self_idx=si, query_entry=qe, **kw)
        out.append((qid, len(qt), time.perf_counter() - t0, len(hits)))
    return out


def summarise(rows):
    t = np.array([r[2] for r in rows])
    return {"mean_s": float(t.mean()), "median_s": float(np.median(t)),
            "q1_s": float(np.percentile(t, 25)), "q3_s": float(np.percentile(t, 75)),
            "p90_s": float(np.percentile(t, 90)), "max_s": float(t.max()),
            "per_query": [{"id": q, "tokens": n, "s": round(x, 5), "hits": h}
                          for q, n, x, h in rows]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", choices=["rhofold", "nufold"])
    ap.add_argument("db")
    ap.add_argument("outdir")
    ap.add_argument("--sizes")
    ap.add_argument("--modes", default="sw,tm_a")
    ap.add_argument("--tm-full", action="store_true", help="also time RFoldSearch-TM at the largest size")
    ap.add_argument("--tm-queries", type=int, default=300)
    ap.add_argument("--usalign")
    ap.add_argument("--subset-ids", help="dir with ids_<n>.txt (RhoFold+ nested subsets)")
    ap.add_argument("--queries", help="RhoFold+ queries_1000.json")
    ap.add_argument("--max-queries", type=int, default=None, help="smoke tests only")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    t0 = time.perf_counter()
    db, ents, toks, ids, n_codes, _cl = load_database(a.db)
    print(f"loaded {len(ids):,} chains in {time.perf_counter() - t0:.0f}s", flush=True)
    sub = load_matrix()
    toks = [np.asarray(t, np.int64) for t in toks]
    pos = {cid: i for i, cid in enumerate(ids)}

    if a.corpus == "rhofold":
        sizes = [int(x) for x in a.sizes.split(",")] if a.sizes else RHO_SIZES
        qlist = [q["id"] for q in json.load(open(a.queries))]
        subsets = {}
        for n in sizes:
            sid = [l.strip() for l in open(f"{a.subset_ids}/ids_{n}.txt") if l.strip()]
            subsets[n] = [pos[x] for x in sid]
    else:
        sizes = [int(x) for x in a.sizes.split(",")] if a.sizes else NU_SIZES
        qidx, order = nufold_queries(ids, toks)
        qlist = [ids[j] for j in qidx]
        subsets = {n: list(order[:n]) for n in sizes}
    if a.max_queries:
        qlist = qlist[:a.max_queries]
    print(f"{len(qlist)} queries, sizes {sizes}", flush=True)

    meta = {"corpus": a.corpus, "db": os.path.abspath(a.db), "top_n": TOP_N, "min_diag": MIN_DIAG,
            "threads": 1, "host": os.uname().nodename,
            "cpu": next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?"),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for n in sizes:
        idx = subsets[n]
        s, build_s, index_mb = subset_searcher(
            db, [ents[i] for i in idx], [toks[i] for i in idx], [ids[i] for i in idx], n_codes, sub)
        local = {cid: i for i, cid in enumerate(s.ids)}
        if a.corpus == "rhofold":      # queries live in every subset; drop the self hit
            queries = [(q, toks[pos[q]], local[q], None) for q in qlist]
        else:                          # held out of every subset
            queries = [(q, toks[pos[q]], -1, ents[pos[q]]) for q in qlist]
        modes = a.modes.split(",") + (["tm"] if a.tm_full and n == max(sizes) else [])
        for mode in modes:
            qs = queries[:a.tm_queries] if mode == "tm" else queries
            rows = time_queries(s, qs, mode, a.usalign)
            rec = dict(meta, n_database=n, mode=mode, build_s=round(build_s, 2),
                       index_mb=round(index_mb, 1), n_queries=len(rows), **summarise(rows))
            rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            path = f"{a.outdir}/{a.corpus}_{mode}_{n}.json"
            json.dump(rec, open(path, "w"), indent=1)
            print(f"{time.strftime('%T')} n={n:>7,} {mode:>4}: mean {rec['mean_s']:.4f}s "
                  f"median {rec['median_s']:.4f}s max {rec['max_s']:.2f}s  build {build_s:.1f}s "
                  f"index {index_mb:.0f}MB", flush=True)
        del s
    print("BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
