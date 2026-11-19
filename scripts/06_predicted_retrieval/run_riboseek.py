"""RiboSeek 0.2.1 on the predicted-structure benchmark: RS-20 at candidate depth 200 and 500 and
RS-80 (its default alphabet) at depth 500, for every query set (p2p, exp, exp_unlabelled).
"""
import csv
import json
import os
import sys
import time
from multiprocessing import Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from build_databases import MANIFEST, WORKERS, guard, path_of
from run_searches import DB, NR80, OUT, REPO, ROOT, load_queries

import torch

VERSION = "0.2.1"
DB020 = os.path.join(DB, "riboseek020_sa20_seq.json")
RUNS = (("sa20", 200, "RiboSeek c200"), ("sa20", 500, "RiboSeek c500"), ("rs80", 500, "RiboSeek RS-80 c500"))
_state = {}


def _init():
    from riboseek.alphabet import Alphabet
    _state["alpha"] = Alphabet.from_pretrained("sa20")


def _enc(job):
    i, urs = job
    from riboseek.features import pdb_to_features
    try:
        f = pdb_to_features(path_of(urs), keep_modified=False)
        key = f"{urs}_rank_1" + (f"_{f['chain_id']}" if f["chain_id"] else "")
        labels = _state["alpha"].encode(f["features"]).tolist()
        return i, key, {"labels": labels, "length": len(labels), "seq": f["sequence"]}, None
    except Exception as exc:
        return i, None, None, f"{urs}: {type(exc).__name__}: {exc}"[:300]


def build():
    """Encode the database (RS-20 labels and sequence); only needed if the cache is missing."""
    import riboseek
    ids = [r["urs_id"] for r in csv.DictReader(open(MANIFEST), delimiter="\t")]
    encoded, key_to_index, failures = {}, {}, []
    t0 = time.perf_counter()
    with Pool(WORKERS, initializer=_init) as pool:
        for n, (i, key, entry, err) in enumerate(
                pool.imap_unordered(_enc, list(enumerate(ids)), chunksize=16), 1):
            if err:
                failures.append(err)
            else:
                encoded[key] = entry
                key_to_index[key] = i
            guard(n, failures, pool)
    json.dump(encoded, open(DB020, "w"))
    json.dump(dict(key_to_index=key_to_index, failures=failures, riboseek=riboseek.__version__,
                   wall_seconds=time.perf_counter() - t0, workers=WORKERS),
              open(DB020 + ".meta.json", "w"))
    print(f"riboseek {riboseek.__version__} db: {len(encoded)} chains, {len(failures)} failures, "
          f"{(time.perf_counter()-t0)/60:.1f} min", flush=True)


def query_sets(paper):
    _, sets = load_queries(paper)
    unl = json.load(open(os.path.join(ROOT, "sample", "exp_unlabelled_queries.json")))
    sets["exp_unlabelled"] = [dict(id=q["id"], path=os.path.join(NR80, q["id"] + ".cif")) for q in unl]
    return sets


def search(alphabet, depths, sets, encoded, k2i):
    """{depth: {query set: {query: {seconds, ranking}}}} for one alphabet."""
    from riboseek import Searcher
    from riboseek.alphabet import Alphabet
    rs = Searcher(Alphabet.from_pretrained(alphabet), encoded, gap_penalty=-2.0)
    t0 = time.perf_counter()
    rs._ensure_prefilter()
    print(f"{alphabet}: prefilter index {time.perf_counter() - t0:.1f}s", flush=True)
    i2k = {v: k for k, v in k2i.items()}
    labels = {q["id"]: rs.encode(q["path"]) for qs in ("exp", "exp_unlabelled") for q in sets[qs]}
    labels.update({q["id"]: i2k[q["index"]] for q in sets["p2p"]})   # key: RiboSeek excludes the query
    rs.search(labels[sets["exp"][0]["id"]], top_n=5)
    res = {}
    for cand in depths:
        res[cand] = {}
        for qs, queries in sets.items():
            out = {}
            for q in queries:
                t = time.perf_counter()
                hits = rs.search(labels[q["id"]], top_n=cand, prefilter=True, prefilter_candidates=cand)
                out[q["id"]] = dict(seconds=time.perf_counter() - t, ranking=[k2i[h["chain"]] for h in hits])
            res[cand][qs] = out
            n = [len(v["ranking"]) for v in out.values()]
            print(f"  {qs} {alphabet} c{cand}: {len(n)} queries, {sum(n) / len(n):.0f} ranked per query",
                  flush=True)
    return res


def main():
    import riboseek
    if riboseek.__version__ != VERSION:
        raise SystemExit(f"expected riboseek {VERSION}, found {riboseek.__version__}")
    if not os.path.exists(DB020):
        build()
    encoded = json.load(open(DB020))
    k2i = json.load(open(DB020 + ".meta.json"))["key_to_index"]
    # RiboSeek breaks prefilter and score ties by database order; keep the order of the
    # original RS-20 database so every run sees the same tie-breaking.
    old = json.load(open(os.path.join(DB, "riboseek_sa20.json")))
    encoded = {k: encoded[k] for k in old if k in encoded} | \
              {k: v for k, v in encoded.items() if k not in old}

    paper = torch.load(os.path.join(REPO, "files", "models", "db.pt"),
                       map_location="cpu", weights_only=False)["entries"]
    sets = query_sets(paper)
    runs = {qs: json.load(open(os.path.join(OUT, f"{qs}.json"))) for qs in sets}
    for alphabet in ("sa20", "rs80"):
        depths = [c for a, c, _ in RUNS if a == alphabet]
        res = search(alphabet, depths, sets, encoded, k2i)
        for a, cand, method in RUNS:
            if a == alphabet:
                for qs in sets:
                    runs[qs][method] = res[cand][qs]
    for qs in sets:
        json.dump(runs[qs], open(os.path.join(OUT, f"{qs}.json"), "w"))
        print(f"wrote {qs}.json", flush=True)


if __name__ == "__main__":
    main()
