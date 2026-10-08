"""RiboSeek 0.2.0 RS-80 (its default alphabet) on the predicted-structure benchmark."""
import csv
import json
import os
import sys
import time
from multiprocessing import Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from build_databases import MANIFEST, WORKERS, guard, path_of
from run_searches import DB, OUT, REPO, load_queries

import torch

DB020 = os.path.join(DB, "riboseek020_sa20_seq.json")
METHOD = "RiboSeek RS-80 c500"
CAND = 500
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
    json.dump(dict(key_to_index=key_to_index, failures=failures, riboseek="0.2.0",
                   wall_seconds=time.perf_counter() - t0, workers=WORKERS),
              open(DB020 + ".meta.json", "w"))
    print(f"riboseek 0.2.0 db: {len(encoded)} chains, {len(failures)} failures, "
          f"{(time.perf_counter()-t0)/60:.1f} min", flush=True)


def search(alphabet, sets, encoded, k2i):
    from riboseek import Searcher
    from riboseek.alphabet import Alphabet
    rs = Searcher(Alphabet.from_pretrained(alphabet), encoded, gap_penalty=-2.0, prefilter_k=6)
    rs._ensure_prefilter()
    i2k = {v: k for k, v in k2i.items()}
    labels = {q["id"]: rs.encode(q["path"]) for q in sets["exp"]}
    labels.update({q["id"]: i2k[q["index"]] for q in sets["p2p"]})   # key: RiboSeek excludes the query
    rs.search(labels[sets["exp"][0]["id"]], top_n=5)
    res = {}
    for qs, queries in sets.items():
        out = {}
        for q in queries:
            t = time.perf_counter()
            hits = rs.search(labels[q["id"]], top_n=CAND, prefilter=True, prefilter_candidates=CAND)
            out[q["id"]] = dict(seconds=time.perf_counter() - t, ranking=[k2i[h["chain"]] for h in hits])
        res[qs] = out
    return res


def main():
    import riboseek
    if riboseek.__version__ != "0.2.0":
        raise SystemExit(f"expected riboseek 0.2.0, found {riboseek.__version__}")
    if not os.path.exists(DB020):
        build()
    encoded = json.load(open(DB020))
    k2i = json.load(open(DB020 + ".meta.json"))["key_to_index"]

    old = json.load(open(os.path.join(DB, "riboseek_sa20.json")))
    same = sum(1 for k, v in encoded.items() if old.get(k, {}).get("labels") == v["labels"])
    print(f"RS-20 labels identical to the existing database: {same}/{len(old)} "
          f"(re-encoded {len(encoded)})", flush=True)
    # RiboSeek breaks prefilter and score ties by database order; use the order
    # of the existing RS-20 database so both alphabets see identical tie-breaking.
    encoded = {k: encoded[k] for k in old if k in encoded} | \
              {k: v for k, v in encoded.items() if k not in old}

    paper = torch.load(os.path.join(REPO, "files", "models", "db.pt"),
                       map_location="cpu", weights_only=False)["entries"]
    _, sets = load_queries(paper)
    runs = {qs: json.load(open(os.path.join(OUT, f"{qs}.json"))) for qs in sets}

    rs20 = search("sa20", sets, encoded, k2i)
    for qs in sets:
        stored = runs[qs]["RiboSeek c500"]
        match = sum(rs20[qs][q]["ranking"] == stored[q]["ranking"] for q in stored)
        print(f"  {qs}: 0.2.0 RS-20 c500 reproduces stored RiboSeek c500 for {match}/{len(stored)} queries",
              flush=True)

    rs80 = search("rs80", sets, encoded, k2i)
    for qs in sets:
        runs[qs][METHOD] = rs80[qs]
        json.dump(runs[qs], open(os.path.join(OUT, f"{qs}.json"), "w"))
        n = [len(v["ranking"]) for v in rs80[qs].values()]
        print(f"  {qs}: added {METHOD} ({len(n)} queries, {sum(n)/len(n):.0f} ranked per query)", flush=True)


if __name__ == "__main__":
    main()
