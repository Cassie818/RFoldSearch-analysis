"""Per-query time breakdown of the default RFoldSearch search on NuFold."""
import json, os, sys, time
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import bench_speed as B
from rfoldsearch import prefilter as PF, structure_score as SS
from rfoldsearch.search import GAP_OPEN, GAP_EXTEND
from rfoldsearch.smith_waterman import smith_waterman_numba

sizes = [int(x) for x in sys.argv[2].split(",")]
out = sys.argv[3]
db, ents, toks, ids, n_codes, _ = B.load_database(sys.argv[1])
sub = B.load_matrix(); toks = [np.asarray(t, np.int64) for t in toks]
qidx, order = B.nufold_queries(ids, toks)
res = {"cpu": next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")),
       "top_n": B.TOP_N, "min_diag": B.MIN_DIAG, "sizes": {}}
P = time.perf_counter
TIMES = ["expand", "kernel", "select", "sw", "tm_a", "search_sw", "search_tm_a"]
COUNTS = ["n_kmers", "postings", "n_cand", "cand_residues"]
for n in sizes:
    idx = list(order[:n])
    s, build_s, index_mb = B.subset_searcher(db, [ents[i] for i in idx], [toks[i] for i in idx],
                                             [ids[i] for i in idx], n_codes, sub)
    pf = s.prefilter; ix = pf.index
    s._flat_structures()

    def step(q, e):
        r = {}
        t = P(); codes, qps = PF.expand_query(q, ix["powers"], pf.sub, pf.diag, pf.thr); r["expand"] = P() - t
        r["n_kmers"] = int(codes.size)
        r["postings"] = int((ix["offsets"][codes + 1] - ix["offsets"][codes]).sum())
        t = P()
        best = PF._best_diagonal_counts(codes, qps.astype(np.int64), ix["offsets"], ix["targets"],
                                        ix["positions"], -1, ix["n_targets"], pf._span, pf._seg,
                                        pf._best, pf._buf)
        r["kernel"] = P() - t
        t = P()
        cand = np.flatnonzero(best >= B.MIN_DIAG); sc = best[cand]
        cand = cand[np.lexsort((cand, -sc))][:B.TOP_N].astype(np.int64)
        r["select"] = P() - t
        r["n_cand"] = int(len(cand)); r["cand_residues"] = int(sum(len(s.toks[c]) for c in cand))
        t = P()
        for c in cand:
            smith_waterman_numba(q, s.toks[c], s.sub, GAP_OPEN, GAP_EXTEND)
        r["sw"] = P() - t
        (ftoks, offs, xyz), qxyz = s._tm_a_inputs(q, e, -1)
        t = P()
        if len(cand):
            SS.score_candidates(q, qxyz, cand, ftoks, offs, xyz, s.sub,
                                np.float32(GAP_OPEN), np.float32(GAP_EXTEND))
        r["tm_a"] = P() - t
        for mode in ("sw", "tm_a"):
            t = P()
            s.search(q, top_k=50, top_n=B.TOP_N, min_diag=B.MIN_DIAG, rerank=mode, query_entry=e)
            r["search_" + mode] = P() - t
        return r

    qs = [(ids[j], toks[j], ents[j]) for j in qidx]
    for _ in range(3):
        for _, q, e in qs[:3]:
            step(q, e)
    rows = []
    for qid, q, e in qs:
        r = step(q, e); r["id"] = qid; r["tokens"] = len(q); rows.append(r)
    mean = {k: float(np.mean([r[k] for r in rows])) for k in TIMES + COUNTS}
    res["sizes"][n] = {"index_mb": round(index_mb, 1), "mean": mean, "per_query": rows}
    print(f"n={n:>7,}: " + "  ".join(f"{k} {mean[k] * 1e3:.1f}ms" for k in TIMES) + "  |  "
          + "  ".join(f"{k} {mean[k]:,.0f}" for k in COUNTS), flush=True)
    json.dump(res, open(out, "w"))
print("BREAKDOWN_DONE", flush=True)
