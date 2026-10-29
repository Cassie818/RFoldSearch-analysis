"""GTalign-fast (--speed=13) on the RhoFold+ ladder, timed directly on the RTX 4090
(single-system rerun, Gitea #4 comment 503). Same loop, queries and flags as the
--speed=8 timing in rfoldseek/bench_all.py; replaces the 2080 Ti s13/s8 ratio.
A --speed=8 re-run at 83,156 checks the August default-mode time (4.326 s).
Run after the CPU jobs so nothing competes with the GPU host process.

usage: python gt13_rho.py"""
import json, subprocess, time
from pathlib import Path
import numpy as np

R = Path("/root/autodl-tmp/rfoldseek")
OUT = Path(__file__).resolve().parent / "results_gt"
OUT.mkdir(exist_ok=True)
qdir = R / "data/work/gt13_query"; qdir.mkdir(parents=True, exist_ok=True)
RUNS = [(13, n) for n in (1000, 5000, 10000, 20000, 50000, 83156)] + [(8, 83156)]

for speed, n in RUNS:
    path = OUT / f"gtalign_s{speed}_{n}.json"
    if path.exists():
        continue
    queries = json.loads((R / f"db/queries_{n}.json").read_text())
    per = []
    for q in queries:
        for f in qdir.glob("*.pdb"):
            f.unlink()
        subprocess.run(f"cp {R}/flat/{n}/{q['id']}.pdb {qdir}/", shell=True)
        t0 = time.perf_counter()
        subprocess.run(f"gtalign --qrs={qdir}/ --rfs={R}/flat/{n}/ --pre-score=0 --pre-similarity=0 "
                       f"--nhits=500 --speed={speed} -o {R}/data/work/gt13_out",
                       shell=True, capture_output=True)
        per.append(time.perf_counter() - t0)
    t = np.array(per)
    path.write_text(json.dumps({
        "method": "gtalign", "speed": speed, "n_database": n, "nhits": 500, "device": "RTX 4090",
        "n_queries": len(t), "mean_s": float(t.mean()), "median_s": float(np.median(t)),
        "p90_s": float(np.percentile(t, 90)), "max_s": float(t.max()),
        "per_query": [{"id": q["id"], "tokens": q["tokens"], "s": round(s, 4)}
                      for q, s in zip(queries, per)]}, indent=1))
    print(f"[{time.strftime('%m-%d %H:%M')}] speed={speed} n={n} mean {t.mean():.3f}s "
          f"median {np.median(t):.3f}s", flush=True)
print("=== GT13_RHO_COMPLETE ===", flush=True)
