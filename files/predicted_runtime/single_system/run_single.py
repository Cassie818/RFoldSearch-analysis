"""Single-system runtime benchmark (Ruyi, Gitea #4 comment 503, 2026-10-28): time the CPU
methods that were measured on the 2080 directly on the reference system (Xeon 8470Q), so
no cross-system factor is needed.

Same 300 RhoFold+ queries and nested target lists as the 2080 runs, same binaries and
command lines as run_xsys.py (bundled 2080 glibc/libstdc++ via the loader). Sizes follow
the 2080 protocol: coordinate aligners at 1k, 5k, 10k and 83,156; Infernal (default and
--rfam) at all six sizes. One single-threaded job per core; longest jobs start first.
Resumable: finished jobs are skipped on restart.

usage: python3 run_single.py FLAT_DIR NPROC"""
import json, os, subprocess, sys, time
from multiprocessing import Pool

H = os.path.dirname(os.path.abspath(__file__))
X = os.path.join(H, "..", "xsys30")
FLAT = sys.argv[1].rstrip("/")
NPROC = int(sys.argv[2])
LD = [f"{X}/lib/ld-linux-x86-64.so.2", "--library-path", f"{X}/lib"]
BIN = f"{X}/bin"
OUT = f"{H}/results"
QUERIES = [l.strip() for l in open(f"{H}/queries_300.txt") if l.strip()]
COORD_SIZES = (1000, 5000, 10000, 83156)
INF_SIZES = (1000, 5000, 10000, 20000, 50000, 83156)
# rough 8470Q seconds per query at 83,156 (xsys30), for longest-first ordering
COST = {"usalign": 1070, "rtmalign": 1120, "rmalign": 1480, "infernal": 45, "infernal_fast": 4}


def targets(n):
    return f"{H}/list_{n}.txt"


def usalign(q, n):
    t0 = time.perf_counter()
    subprocess.run(LD + [f"{BIN}/USalign", f"{FLAT}/{q}.pdb", "-dir2", FLAT + "/", targets(n),
                         "-suffix", ".pdb", "-mol", "RNA", "-outfmt", "2"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return time.perf_counter() - t0, {}


def rtmalign(q, n):
    # rtm_resilient.py: 1,000-target chunks; on a segfault skip the crashing target
    ids = [l.strip() for l in open(targets(n)) if l.strip()]
    total, skipped = 0.0, []
    for i in range(0, len(ids), 1000):
        sub = ids[i:i + 1000]
        while sub:
            tf = f"/tmp/rtm_{q}_{n}_{os.getpid()}.lst"
            open(tf, "w").write("\n".join(sub) + "\n")
            t0 = time.perf_counter()
            p = subprocess.run(LD + [f"{BIN}/RTMalign", f"{FLAT}/{q}.pdb", "-dir2", FLAT + "/", tf,
                                     "-suffix", ".pdb", "-mol", "RNA", "-outfmt", "2"],
                               capture_output=True, text=True)
            total += time.perf_counter() - t0
            os.remove(tf)
            done = sum(1 for ln in p.stdout.splitlines() if "URS" in ln and ".pdb" in ln)
            if done >= len(sub):
                break
            skipped.append(sub[done])
            sub = sub[done + 1:]
    return total, {"n_skip": len(skipped)}


def rmalign(q, n):
    # coordB.sh: one RNAalign process per pair, driven by a shell loop
    script = (f'while read tid; do {" ".join(LD)} {BIN}/RNAalign -A "{FLAT}/{q}.pdb" '
              f'-B "{FLAT}/$tid.pdb" >/dev/null 2>&1; done < {targets(n)}')
    t0 = time.perf_counter()
    subprocess.run(["bash", "-c", script])
    return time.perf_counter() - t0, {}


def infernal(q, n, extra):
    tbl = f"/tmp/inf_{q}_{n}_{os.getpid()}.tbl"
    t0 = time.perf_counter()
    subprocess.run(LD + [f"{BIN}/cmsearch", "--cpu", "0", "--noali", "--toponly", "-T", "0",
                         "--tblout", tbl, "-o", "/dev/null", *extra,
                         f"{H}/cm/{q}.cm", f"{H}/db/targets_{n}.fasta"], check=True,
                   stdout=subprocess.DEVNULL)
    dt = time.perf_counter() - t0
    nhit = sum(1 for l in open(tbl) if not l.startswith("#"))
    os.remove(tbl)
    return dt, {"n_hits": nhit}


METHODS = {"usalign": usalign, "rtmalign": rtmalign, "rmalign": rmalign,
           "infernal": lambda q, n: infernal(q, n, []),
           "infernal_fast": lambda q, n: infernal(q, n, ["--rfam"])}


def job(mqn):
    m, q, n = mqn
    dt, extra = METHODS[m](q, n)
    rec = dict(method=m, q=q, n_database=n, time_s=dt, host=os.uname().nodename, **extra)
    open(f"{OUT}/{m}__{n}__{q}.json", "w").write(json.dumps(rec))
    print(f"[{time.strftime('%m-%d %H:%M')}] {m:14s} {n:6d} {q} {dt:9.1f}s", flush=True)
    return rec


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    jobs = [(m, q, n) for m in METHODS
            for n in (INF_SIZES if m.startswith("infernal") else COORD_SIZES)
            for q in QUERIES if not os.path.exists(f"{OUT}/{m}__{n}__{q}.json")]
    jobs.sort(key=lambda j: -COST[j[0]] * j[2])
    est = sum(COST[m] * n / 83156 for m, _, n in jobs) / 3600
    print(f"{len(jobs)} jobs, {NPROC}-wide, est {est:.0f} core-h, ~{est / NPROC:.1f} h", flush=True)
    with Pool(NPROC) as pool:
        for _ in pool.imap_unordered(job, jobs):
            pass
    print("=== SINGLE300_COMPLETE ===", flush=True)
