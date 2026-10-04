"""Exhaustive RMalign: every structure-truth query against all entries of the sampled"""
import csv
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, REPO)
import paths  # noqa: E402

HERE = os.path.join(REPO, "files", "predicted_retrieval")
STRUCTURES = os.path.join(paths.NUFOLD_ROOT, "structures", "rank1")
RMALIGN = os.environ.get("RFS_RMALIGN_BIN", os.path.expanduser("~/Downloads/RMalign/RMalign/RNAalign"))
PDB_DIR = os.environ.get("RFS_RMALIGN_PDB", "/tmp/rmalign_pdb_predicted")
WORKERS = 8
_RM_B = re.compile(r"RMscore=\s*([0-9.]+) \(if normalized by length of structure B")
_RM_A = re.compile(r"RMscore=\s*([0-9.]+) \(if normalized by length of structure A")


def cif_to_pdb(qid, cif):
    out = os.path.join(PDB_DIR, qid + ".pdb")
    if not os.path.exists(out):
        from Bio.PDB import MMCIFParser, PDBIO
        st = MMCIFParser(QUIET=True).get_structure(qid, cif)
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


def short_link(name):
    """Short path to a predicted structure (RMalign truncates long file names)."""
    link = os.path.join(PDB_DIR, "t", name)
    if not os.path.lexists(link):
        os.symlink(os.path.join(STRUCTURES, name), link)
    return link


def run_one(job):
    qs, qid, qpath, targets, out = job
    if os.path.exists(out):
        return qs, qid, 0.0, "cached"
    t0 = time.perf_counter()
    n_ok = 0
    with open(out + ".part", "w") as fh:
        fh.write("target\tRM_query\tRM_target\n")
        for t in targets:
            res = subprocess.run([RMALIGN, "-A", qpath, "-B", os.path.join(PDB_DIR, "t", t)],
                                 capture_output=True, text=True).stdout
            mb, ma = _RM_B.search(res), _RM_A.search(res)
            if mb and ma:
                fh.write(f"{t}\t{ma.group(1)}\t{mb.group(1)}\n")
                n_ok += 1
    if n_ok == 0:
        return qs, qid, time.perf_counter() - t0, "FAILED: no target scored"
    os.replace(out + ".part", out)
    return qs, qid, time.perf_counter() - t0, f"ok ({n_ok} targets)"


def main():
    manifest = list(csv.DictReader(open(os.path.join(HERE, "sample", "accuracy_db.tsv")), delimiter="\t"))
    targets = [f"{r['urs_id']}_rank_1.pdb" for r in manifest]
    os.makedirs(os.path.join(PDB_DIR, "t"), exist_ok=True)
    for t in targets:
        short_link(t)
    sets = {"p2p": ("p2p_queries.json", lambda q: q["urs_id"],
                    lambda q: short_link(f"{q['urs_id']}_rank_1.pdb")),
            "exp": ("exp_queries.json", lambda q: q["id"],
                    lambda q: cif_to_pdb(q["id"], os.path.join(paths.NR80_DIR, q["id"] + ".cif"))),
            "exp_unlabelled": ("exp_unlabelled_queries.json", lambda q: q["id"],
                               lambda q: cif_to_pdb(q["id"], os.path.join(paths.NR80_DIR, q["id"] + ".cif")))}
    jobs = []
    for qs, (fname, idf, pathf) in sets.items():
        d = os.path.join(HERE, "rmalign", qs)
        os.makedirs(d, exist_ok=True)
        qlist = json.load(open(os.path.join(HERE, "sample", fname)))
        for q in qlist["queries"] if isinstance(qlist, dict) else qlist:
            if os.path.exists(os.path.join(HERE, "usalign", qs, idf(q) + ".tsv")):
                jobs.append((qs, idf(q), pathf(q), targets, os.path.join(d, idf(q) + ".tsv")))
    print(f"{len(jobs)} queries x {len(targets)} targets, {WORKERS} workers", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(WORKERS) as ex:
        for n, fut in enumerate(as_completed([ex.submit(run_one, j) for j in jobs]), 1):
            qs, qid, dt, status = fut.result()
            print(f"[{n}/{len(jobs)}] {qs} {qid} {dt:7.1f}s {status} (elapsed {(time.perf_counter() - t0) / 60:.1f} min)",
                  flush=True)
    print("DONE rmalign", flush=True)


if __name__ == "__main__":
    main()
