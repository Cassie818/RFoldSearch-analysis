"""Searches and exhaustive US-align for the unlabelled experimental queries.

The 76 held-out test chains without an Rfam family assignment cannot be scored
against the annotation, but the structure-defined ground truth needs no
annotation, so the 54 of them within the database length range (30-400 nt) are
added to that evaluation. Same methods, same settings and same output format
as run_searches.py / run_usalign.py.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
import sys as _sys                                             # noqa: E402
_sys.path.insert(0, REPO)
import paths                                                   # noqa: E402
HERE = os.path.join(REPO, "files", "predicted_retrieval")      # results; reruns write db/, usalign/, runs/ here
ROOT = HERE                                                    # holds sample/
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, paths.RFOLDSEEK_REPO)
import run_searches as R                                      # noqa: E402

QUERIES = os.path.join(ROOT, "sample", "exp_unlabelled_queries.json")
OUT = os.path.join(HERE, "runs", "exp_unlabelled.json")
USDIR = os.path.join(HERE, "usalign", "exp_unlabelled")
WORKERS = 8


def usalign_one(job):
    qid, qpath, listfile, out = job
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return qid, 0.0, "cached"
    t0 = time.perf_counter()
    with open(out + ".part", "w") as fh:
        p = subprocess.run([R.USALIGN, qpath, "-dir2", os.path.join(paths.NUFOLD_ROOT, "structures", "rank1") + os.sep,
                            listfile, "-mol", "RNA", "-outfmt", "2"],
                           stdout=fh, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        return qid, time.perf_counter() - t0, f"FAILED: {p.stderr[:200]}"
    os.replace(out + ".part", out)
    return qid, time.perf_counter() - t0, "ok"


def main():
    paper = torch.load(os.path.join(REPO, "files", "models", "db.pt"),
                       map_location="cpu", weights_only=False)["entries"]
    qs = json.load(open(QUERIES))
    queries = [dict(id=q["id"], self_idx=-1, entry=paper[q["index"]],
                    path=os.path.join(R.NR80, q["id"] + ".cif")) for q in qs]
    for q in queries:
        q["seq"] = R.seq_from_cif(q["path"])
    missing = [q["id"] for q in queries if not os.path.exists(q["path"])]
    if missing:
        raise SystemExit(f"missing query files: {missing[:3]}")
    print(f"{len(queries)} unlabelled experimental queries", flush=True)

    sets = {"exp_unlabelled": queries}
    manifest = list(__import__("csv").DictReader(
        open(os.path.join(ROOT, "sample", "accuracy_db.tsv")), delimiter="\t"))
    res = R.run_token(sets, manifest)

    # RiboSeek and the sequence tools, as in run_searches.py
    from riboseek import Searcher as RS
    meta = json.load(open(os.path.join(R.DB, "riboseek_sa20.json.meta.json")))
    k2i = meta["key_to_index"]
    rs = RS.from_pretrained("sa20", db=os.path.join(R.DB, "riboseek_sa20.json"))
    rs._ensure_prefilter()
    labels = {q["id"]: rs.encode(q["path"]) for q in queries}
    rs.search(labels[queries[0]["id"]], top_n=5)
    for cand in (200, 500):
        out = {}
        for q in queries:
            t = time.perf_counter()
            hits = rs.search(labels[q["id"]], top_n=cand, prefilter=True, prefilter_candidates=cand)
            out[q["id"]] = dict(seconds=time.perf_counter() - t,
                                ranking=[k2i[h["chain"]] for h in hits])
        res["exp_unlabelled"][f"RiboSeek c{cand}"] = out
        print(f"  RiboSeek c{cand}: median {np.median([v['seconds'] for v in out.values()])*1000:.1f} ms", flush=True)
    R.run_sequence(sets, res)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res["exp_unlabelled"], open(OUT, "w"))
    print(f"wrote {OUT}", flush=True)

    # exhaustive US-align
    os.makedirs(USDIR, exist_ok=True)
    listfile = os.path.join(HERE, "usalign_targets.txt")
    jobs = [(q["id"], q["path"], listfile, os.path.join(USDIR, q["id"] + ".tsv"))
            for q in sorted(queries, key=lambda q: -len(q["entry"]["tokens"]))]
    print(f"exhaustive US-align: {len(jobs)} queries x {len(manifest)} targets", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(WORKERS) as ex:
        for n, fut in enumerate(as_completed([ex.submit(usalign_one, j) for j in jobs]), 1):
            qid, dt, status = fut.result()
            print(f"[{n}/{len(jobs)}] {qid} {dt:6.1f}s {status} "
                  f"(elapsed {(time.perf_counter()-t0)/60:.1f} min)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
