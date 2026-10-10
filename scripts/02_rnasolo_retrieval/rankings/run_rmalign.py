"""RMalign rankings on the RNAsolo benchmark, by the symmetric RMscore.

Every test query is aligned with all 885 chains; each pair is aligned once and both
RMscore lines of RNAalign are kept:

    RNAalign -A <i>.pdb -B <j>.pdb
      RMscore=... (normalized by length of structure B)   -> rm_j  (normalised by j)
      RMscore=... (normalized by length of structure A)   -> rm_i  (normalised by i)
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import benchmark as BM  # noqa: E402
from paths import NR80_DIR  # noqa: E402

RMALIGN = os.environ.get("RFS_RMALIGN_BIN", os.path.expanduser("~/Downloads/RMalign/RMalign/RNAalign"))
PDB_DIR = os.environ.get("RFS_RMALIGN_PDB", "/tmp/rmalign_pdb_bidir")
CACHE = os.path.join(BM.RET, "rmalign_bidirectional.cache.jsonl")
_RM_B = re.compile(r"RMscore=\s*([0-9.]+) \(if normalized by length of structure B")
_RM_A = re.compile(r"RMscore=\s*([0-9.]+) \(if normalized by length of structure A")


def to_pdb(tag):
    out = os.path.join(PDB_DIR, tag + ".pdb")
    if not os.path.exists(out):
        from Bio.PDB import MMCIFParser, PDBIO
        st = MMCIFParser(QUIET=True).get_structure(tag, os.path.join(NR80_DIR, tag + ".cif"))
        for m in list(st)[1:]:
            st.detach_child(m.id)
        for chain in next(iter(st)):
            if len(chain.id) != 1:
                chain.id = "A"
        io = PDBIO()
        io.set_structure(st)
        io.save(out + ".tmp")
        os.replace(out + ".tmp", out)
    return out


def align(job):
    i, j, a, b = job
    res = subprocess.run([RMALIGN, "-A", a, "-B", b], capture_output=True, text=True)
    mb, ma = _RM_B.search(res.stdout), _RM_A.search(res.stdout)
    if not (mb and ma):
        return {"i": i, "j": j, "error": "no RMscore"}
    return {"i": i, "j": j, "rm_i": float(ma.group(1)), "rm_j": float(mb.group(1))}


def validate(b, n_pairs, workers):
    old = [json.loads(l) for l in open(os.path.join(BM.RET, "rmalign_full.cache.jsonl")) if l.strip()]
    pick = [old[k] for k in np.random.default_rng(0).choice(len(old), n_pairs, replace=False)]
    jobs = [(r["i"], r["j"], to_pdb(b.tags[r["i"]]), to_pdb(b.tags[r["j"]])) for r in pick]
    t = time.perf_counter()
    with Pool(workers) as p:
        res = p.map(align, jobs)
    dt = time.perf_counter() - t
    d = np.array([abs(r["rm_j"] - o["rmscore"]) for r, o in zip(res, pick) if "rm_j" in r])
    print(f"{len(d)}/{len(pick)} pairs in {dt:.1f} s ({len(pick) / dt:.2f} pairs/s, {workers} workers); "
          f"rm_j equal to the cached RMscore (<1e-4) for {np.mean(d < 1e-4):.3f}, max |diff| {d.max():.5f}")


def align_missing(b, workers):
    pairs = sorted({(min(q, t), max(q, t)) for q in b.queries for t in range(b.n) if t != q})
    done = set()
    if os.path.exists(CACHE):
        done = {(r["i"], r["j"]) for r in map(json.loads, open(CACHE))}
    todo = [p for p in pairs if p not in done]
    print(f"{len(b.queries)} queries, {len(pairs)} pairs, {len(pairs) - len(todo)} cached, {len(todo)} to align",
          flush=True)
    if not todo:
        return
    jobs = [(i, j, to_pdb(b.tags[i]), to_pdb(b.tags[j])) for i, j in todo]
    t0 = time.perf_counter()
    with open(CACHE, "a", buffering=1) as out, Pool(workers) as p:
        for k, r in enumerate(p.imap_unordered(align, jobs, chunksize=16)):
            out.write(json.dumps(r) + "\n")
            if (k + 1) % 2000 == 0:
                rate = (k + 1) / (time.perf_counter() - t0)
                print(f"  {k + 1}/{len(jobs)}  {rate:.1f} pairs/s  ETA {(len(jobs) - k - 1) / rate / 3600:.1f} h",
                      flush=True)


def rankings(b):
    rm = np.full((b.n, b.n), np.nan, dtype=np.float32)
    for r in map(json.loads, open(CACHE)):
        if "rm_i" in r:
            rm[r["i"], r["j"]] = rm[r["j"], r["i"]] = 0.5 * (r["rm_i"] + r["rm_j"])
    BM.save_rankings("RMalign", {q: [BM.rank_by_score(rm[q], q)] for q in b.queries}, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", type=int, default=0, help="realign N pairs of the earlier cache and compare")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    b = BM.load()
    os.makedirs(PDB_DIR, exist_ok=True)
    if a.validate:
        validate(b, a.validate, a.workers)
        return
    align_missing(b, a.workers)
    rankings(b)


if __name__ == "__main__":
    main()
