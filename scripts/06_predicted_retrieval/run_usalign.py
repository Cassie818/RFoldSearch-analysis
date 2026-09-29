"""Exhaustive US-align: every query of both sets against all entries of the
sampled accuracy database (-mol RNA -outfmt 2). Longest queries first;
completed queries are skipped on restart.
Output: usalign/<queryset>/<query id>.tsv
"""
import csv
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
import sys as _sys                                             # noqa: E402
_sys.path.insert(0, REPO)
import paths                                                   # noqa: E402
HERE = os.path.join(REPO, "files", "predicted_retrieval")      # results; reruns write db/, usalign/, runs/ here
ROOT = HERE                                                    # holds sample/
STRUCTURES = os.path.join(paths.NUFOLD_ROOT, "structures", "rank1")
USALIGN = paths.USALIGN
NR80 = paths.NR80_DIR
WORKERS = 8


def run_one(job):
    qs, qid, qpath, listfile, out = job
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return qs, qid, 0.0, "cached"
    t0 = time.perf_counter()
    with open(out + ".part", "w") as fh:
        p = subprocess.run([USALIGN, qpath, "-dir2", STRUCTURES + os.sep, listfile, "-mol", "RNA",
                            "-outfmt", "2"], stdout=fh, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        return qs, qid, time.perf_counter() - t0, f"FAILED: {p.stderr[:200]}"
    os.replace(out + ".part", out)
    return qs, qid, time.perf_counter() - t0, "ok"


def main():
    manifest = list(csv.DictReader(open(os.path.join(ROOT, "sample", "accuracy_db.tsv")), delimiter="\t"))
    listfile = os.path.join(HERE, "usalign_targets.txt")
    with open(listfile, "w") as fh:
        fh.write("\n".join(f"{r['urs_id']}_rank_1.pdb" for r in manifest) + "\n")
    jobs = []
    p2p = json.load(open(os.path.join(ROOT, "sample", "p2p_queries.json")))["queries"]
    exp = json.load(open(os.path.join(ROOT, "sample", "exp_queries.json")))["queries"]
    for qs, qlist, pathf, idf in (("p2p", p2p, lambda q: os.path.join(STRUCTURES, f"{q['urs_id']}_rank_1.pdb"), lambda q: q["urs_id"]),
                                  ("exp", exp, lambda q: os.path.join(NR80, q["id"] + ".cif"), lambda q: q["id"])):
        d = os.path.join(HERE, "usalign", qs)
        os.makedirs(d, exist_ok=True)
        for q in qlist:
            jobs.append((q["length"], (qs, idf(q), pathf(q), listfile, os.path.join(d, idf(q) + ".tsv"))))
    jobs = [j for _, j in sorted(jobs, key=lambda x: -x[0])]
    print(f"{len(jobs)} queries x {len(manifest)} targets, {WORKERS} workers", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(WORKERS) as ex:
        for n, fut in enumerate(as_completed([ex.submit(run_one, j) for j in jobs]), 1):
            qs, qid, dt, status = fut.result()
            print(f"[{n}/{len(jobs)}] {qs} {qid} {dt:7.1f}s {status} (elapsed {(time.perf_counter()-t0)/60:.1f} min)", flush=True)
    print("DONE usalign", flush=True)


if __name__ == "__main__":
    main()
