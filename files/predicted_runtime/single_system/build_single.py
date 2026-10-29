"""Single-system runtime data: US-align, RTM-align, RMalign, Infernal and Infernal-fast timed directly on the reference system (Xeon 8470Q,
one core per job; AutoDL bench2/single300), GTalign-fast directly on the RTX 4090."""
import glob
import json
import os
import shutil

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.expanduser("~/rfs_bench2/single300")
COORD = (1000, 5000, 10000, 83156)
INF = (1000, 5000, 10000, 20000, 50000, 83156)
SYS = "Intel Xeon Platinum 8470Q, one core per single-threaded job (AutoDL, 2026-10-29)"
NQ = 300


def per_query(m, n):
    recs = [json.load(open(f)) for f in glob.glob(f"{SRC}/results/{m}__{n}__*.json")]
    assert len(recs) == NQ, (m, n, len(recs))
    return sorted(recs, key=lambda r: r["q"])


def stats(recs):
    t = np.array([r["time_s"] for r in recs])
    return {"mean_s": float(t.mean()), "median_s": float(np.median(t)), "n": len(t)}


os.makedirs(os.path.join(HERE, "data_xsys"), exist_ok=True)
for f in glob.glob(os.path.join(HERE, "data", "infernal*_*.json")) + \
        glob.glob(os.path.join(HERE, "data", "gtalign_fast_*.json")):
    dst = os.path.join(HERE, "data_xsys", os.path.basename(f))
    if not os.path.exists(dst):
        shutil.copy(f, dst)

coord = {"note": f"Measured directly on the reference system: {SYS}; 300 queries x all targets",
         "sizes": {}}
for n in COORD:
    row = {}
    for m in ("usalign", "rtmalign", "rmalign"):
        recs = per_query(m, n)
        row[m] = stats(recs)
        if m == "rtmalign":
            row[m]["n_skip"] = int(sum(r["n_skip"] for r in recs))
        row[m]["per_query"] = [{"id": r["q"], "s": round(r["time_s"], 3)} for r in recs]
    coord["sizes"][str(n)] = row
json.dump(coord, open(os.path.join(HERE, "coord_8470.json"), "w"), indent=1)

for m, label in (("infernal", "Infernal cmsearch --toponly -T 0"),
                 ("infernal_fast", "Infernal cmsearch --toponly -T 0 --rfam")):
    for n in INF:
        recs = per_query(m, n)
        out = dict(method=label, n_database=n, **stats(recs), note=f"Measured directly: {SYS}",
                   n_hits_total=int(sum(r["n_hits"] for r in recs)),
                   per_query=[{"id": r["q"], "s": round(r["time_s"], 4), "n_hits": r["n_hits"]}
                              for r in recs])
        json.dump(out, open(os.path.join(HERE, "data", f"{m}_{n}.json"), "w"), indent=1)

for n in INF:
    g = json.load(open(f"{SRC}/results_gt/gtalign_s13_{n}.json"))
    assert g["n_queries"] == NQ
    g["note"] = "GTalign --speed=13 measured directly on the RTX 4090 (2026-10-29)"
    json.dump(g, open(os.path.join(HERE, "data", f"gtalign_fast_{n}.json"), "w"), indent=1)

for n in COORD:
    print(n, {m: round(v["mean_s"], 2) for m, v in coord["sizes"][str(n)].items()},
          "rtm skips", coord["sizes"][str(n)]["rtmalign"]["n_skip"])
for n in INF:
    print(n, {m: round(json.load(open(os.path.join(HERE, "data", f"{m}_{n}.json")))["mean_s"], 3)
              for m in ("infernal", "infernal_fast", "gtalign_fast")})
