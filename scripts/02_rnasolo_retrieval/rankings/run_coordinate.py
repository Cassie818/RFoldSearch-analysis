"""Rankings of US-align, GTalign and RTM-align on the RNAsolo benchmark (RMalign: run_rmalign.py).

Each aligner compares every query with all 885 chains; targets are ranked by the
aligner's symmetric score (mean of the query- and target-normalised scores), highest
first, ties by database index, the query excluded.

  usalign   symmetric TM-score from the all-vs-all cache (usalign_full.cache.jsonl)
  gtalign   GTalign 1.0.1 raw outputs (<query>__0.out), mean of the two TM-scores:
              gtalign --qrs=<query>.cif --rfs=<db dir> --pre-score=0
                      --pre-similarity=0 --nhits=885 --speed=8 -o <out>
  rtmalign  RTM-align raw outputs (<query>.tab), mean of TM1 and TM2:
              RTMalign <query>.cif -dir2 <db dir>/ <list> -outfmt 2

    python run_coordinate.py usalign
    python run_coordinate.py gtalign --out <dir>
    python run_coordinate.py rtmalign --out <dir>
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import benchmark as BM  # noqa: E402


def from_matrix(method, m):
    b = BM.load()
    BM.save_rankings(method, {q: [BM.rank_by_score(m[q], q)] for q in b.queries}, b)


def from_rows(method, files, query_of, rows_of):
    """Rankings from per-query raw outputs; rows_of yields (target tag, score1, score2)."""
    b = BM.load()
    runs = {}
    for f in files:
        q = b.tag2i.get(query_of(f))
        if q is None:
            continue
        rows = sorted(((b.tag2i[t], 0.5 * (s1 + s2)) for t, s1, s2 in rows_of(f) if t in b.tag2i),
                      key=lambda x: (-x[1], x[0]))
        runs[q] = [np.asarray([t for t, _ in rows if t != q], np.int64)]
    missing = [b.tags[q] for q in b.queries if q not in runs]
    assert not missing, f"no output for {missing}"
    BM.save_rankings(method, runs, b)


def gtalign_rows(path):
    row = re.compile(r"(PDB_[0-9A-Za-z_]+)\.cif")
    for line in open(path):
        m = row.search(line)
        nums = re.findall(r"\d+\.\d+", line)
        if m and len(nums) >= 2:
            yield m.group(1), float(nums[0]), float(nums[1])


def rtm_rows(path):
    for line in open(path):
        p = line.split()
        if line.startswith("#") or len(p) < 4:
            continue
        try:
            yield os.path.basename(p[1]).replace(".cif", ""), float(p[2]), float(p[3])
        except ValueError:      # trailing summary line ("Total ... time ...")
            continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=("usalign", "gtalign", "rtmalign"))
    ap.add_argument("--out", help="raw output directory (gtalign, rtmalign)")
    a = ap.parse_args()
    if a.method == "usalign":
        from_matrix("US-align", BM.load().tm)
    elif a.method == "gtalign":
        from_rows("GTalign", glob.glob(os.path.join(a.out, "*.out")),
                  lambda f: os.path.basename(f).split("__")[0], gtalign_rows)
    else:
        from_rows("RTM-align", glob.glob(os.path.join(a.out, "*.tab")),
                  lambda f: os.path.basename(f)[:-len(".tab")], rtm_rows)


if __name__ == "__main__":
    main()
