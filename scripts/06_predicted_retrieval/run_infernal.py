"""Infernal single-query-CM baseline for the NuFold benchmark."""
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
import sys as _sys
_sys.path.insert(0, REPO)
import paths
HERE = os.path.join(REPO, "files", "predicted_retrieval")
ROOT = HERE
sys.path.insert(0, paths.RFOLDSEARCH_REPO)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO, "src", "search"))
import run_searches as R
from build_databases import path_of
from secondary_structure import detect_pairs, WC_NITROGEN

INF = paths.INFERNAL_BIN         # Infernal 1.1.5 bin dir; empty -> PATH
CMBUILD, CMSEARCH = os.path.join(INF, "cmbuild"), os.path.join(INF, "cmsearch")
CMCALIBRATE = os.path.join(INF, "cmcalibrate")
DB_FA = os.path.join(R.DB, "sequences.fa")
STD = {"A", "C", "G", "U"}
CALIB_L = "0.01"        # tiny random-sequence length: cmsearch needs E-value
WORKERS = 8             # params to run at all, but we rank by bit score, so the
#                        calibration only has to exist, not be accurate.


def residue_atoms(path):
    """Per-residue (base, C1' xyz, WC-nitrogen xyz) for the first model/chain.

    Reads mmCIF (_atom_site loop) or PDB. Missing atoms give NaN rows."""
    ext = os.path.splitext(path)[1].lower()
    res = {}       # key -> {"base":.., "C1'":xyz, "N":xyz}, insertion-ordered
    order = []
    if ext == ".cif":
        cols, in_loop = {}, False
        model = chain = None
        for line in open(path):
            if line.startswith("_atom_site."):
                cols[line.strip().split(".", 1)[1]] = len(cols); in_loop = True; continue
            if in_loop and line.startswith(("ATOM", "HETATM")):
                f = line.split()
                cm, ca = f[cols["pdbx_PDB_model_num"]], f[cols["label_asym_id"]]
                if model is None:
                    model, chain = cm, ca
                if cm != model or ca != chain:
                    continue
                comp = f[cols["label_comp_id"]].strip('"')
                atom = f[cols["label_atom_id"]].strip('"')
                key = (f[cols["label_seq_id"]], f[cols["auth_seq_id"]],
                       f[cols.get("pdbx_PDB_ins_code", cols["auth_seq_id"])])
                xyz = [float(f[cols[c]]) for c in ("Cartn_x", "Cartn_y", "Cartn_z")]
                r = res.setdefault(key, {"base": comp});
                if key not in order:
                    order.append(key)
                if atom in ("C1'", "C1*"):
                    r["C1'"] = xyz
                if atom == WC_NITROGEN.get(comp if comp in WC_NITROGEN else "", ""):
                    r["N"] = xyz
            elif in_loop and res and line.startswith(("#", "loop_")):
                break
    else:   # PDB
        model_seen = False
        for line in open(path):
            if line.startswith("MODEL"):
                if model_seen:
                    break
                model_seen = True
            if line.startswith(("ATOM", "HETATM")):
                comp = line[17:20].strip()
                atom = line[12:16].strip()
                chain = line[21]
                key = (chain, line[22:27])           # chain + resSeq+icode
                if order and key[0] != order[0][0]:
                    continue
                try:
                    xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                except ValueError:
                    continue
                r = res.setdefault(key, {"base": comp})
                if key not in order:
                    order.append(key)
                if atom in ("C1'", "C1*"):
                    r["C1'"] = xyz
                if atom == WC_NITROGEN.get(comp if comp in WC_NITROGEN else "", ""):
                    r["N"] = xyz
    bases, c1, wc = [], [], []
    for key in order:
        r = res[key]
        b = r["base"] if r["base"] in STD else "N"
        bases.append(b)
        c1.append(r.get("C1'", [np.nan] * 3))
        wc.append(r.get("N", [np.nan] * 3))
    return bases, np.asarray(wc, float), np.asarray(c1, float)


def nested(pairs, n):
    """Drop crossing pairs (cmbuild needs a nested structure); greedy by span."""
    kept = []
    for i, j in sorted((tuple(p) for p in pairs), key=lambda p: p[1] - p[0]):
        if all(not (a < i < b < j or i < a < j < b) for a, b in kept):
            kept.append((i, j))
    ss = ["."] * n
    for i, j in kept:
        ss[i], ss[j] = "(", ")"
    return "".join(ss)


def build_cm(qid, seq, ss, tmp):
    sto = os.path.join(tmp, "q.sto")
    with open(sto, "w") as fh:
        fh.write("# STOCKHOLM 1.0\n")
        fh.write(f"{qid:<20s} {seq}\n")
        fh.write(f"{'#=GC SS_cons':<20s} {ss}\n//\n")
    cm = os.path.join(tmp, "q.cm")
    cmd = [CMBUILD, "-F", cm, sto] if "(" in ss else [CMBUILD, "-F", "--noss", cm, sto]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return None, p.stderr[:200]
    return cm, None


def one_query(job):
    """Build + calibrate a single-sequence CM and search the database.
    Runs in its own temp dir so many can run in parallel."""
    qid, path, self_idx = job
    try:
        bases, wc, c1 = residue_atoms(path)
        seq = "".join(bases).replace("T", "U")
        pairs = detect_pairs(bases, wc, c1)
        ss = nested(pairs, len(bases))
        with tempfile.TemporaryDirectory(prefix="infernal_") as tmp:
            cm, err = build_cm(qid, seq, ss, tmp)
            if cm is None:
                return qid, dict(seconds=0.0, ranking=[]), len(pairs), f"cmbuild: {err}"
            t = time.perf_counter()
            c = subprocess.run([CMCALIBRATE, "--cpu", "1", "-L", CALIB_L, cm],
                               capture_output=True, text=True)
            if c.returncode != 0:
                return qid, dict(seconds=0.0, ranking=[]), len(pairs), f"cmcalibrate: {c.stderr[:120]}"
            tbl = os.path.join(tmp, "o.tbl")
            s = subprocess.run([CMSEARCH, "--cpu", "1", "--toponly", "-T", "0", "--incT", "0",
                                "--tblout", tbl, "-o", os.devnull, cm, DB_FA],
                               capture_output=True, text=True)
            dt = time.perf_counter() - t
            if s.returncode != 0:
                return qid, dict(seconds=0.0, ranking=[]), len(pairs), f"cmsearch: {s.stderr[:120]}"
            best = {}
            for line in open(tbl):
                if line.startswith("#") or not line.strip():
                    continue
                f = line.split()
                i, bits = int(f[0][1:]), float(f[14])
                if i == self_idx:
                    continue
                if i not in best or bits > best[i]:
                    best[i] = bits
            order = sorted(best, key=lambda i: (-best[i], i))
            return qid, dict(seconds=dt, ranking=order), len(pairs), "ok"
    except Exception as exc:
        return qid, dict(seconds=0.0, ranking=[]), 0, f"{type(exc).__name__}: {exc}"[:150]


def search(sets):
    out = {}
    for qs, queries in sets.items():
        jobs = [(q["id"], q["path"], q["self_idx"]) for q in queries]
        res, npair = {}, []
        t0 = time.perf_counter()
        with ProcessPoolExecutor(WORKERS) as ex:
            for n, fut in enumerate(as_completed([ex.submit(one_query, j) for j in jobs]), 1):
                qid, r, np_, status = fut.result()
                res[qid] = r
                npair.append(np_)
                if status != "ok":
                    print(f"  [{n}/{len(jobs)}] {qid}: {status}", flush=True)
        print(f"{qs}: Infernal {len(res)} queries in {(time.perf_counter()-t0)/60:.1f} min, "
              f"median {np.median([v['seconds'] for v in res.values()])*1000:.0f} ms/search, "
              f"median {int(np.median(npair))} base pairs/query", flush=True)
        out[qs] = res
    return out


def main():
    import torch
    paper = torch.load(os.path.join(REPO, "files", "models", "db.pt"),
                       map_location="cpu", weights_only=False)["entries"]
    manifest = list(__import__("csv").DictReader(
        open(os.path.join(ROOT, "sample", "accuracy_db.tsv")), delimiter="\t"))
    row_of = {r["urs_id"]: i for i, r in enumerate(manifest)}
    exp = json.load(open(os.path.join(ROOT, "sample", "exp_queries.json")))["queries"]
    p2p = json.load(open(os.path.join(ROOT, "sample", "p2p_queries.json")))["queries"]
    sets = {
        "exp": [dict(id=q["id"], path=os.path.join(R.NR80, q["id"] + ".cif"), self_idx=-1) for q in exp],
        "p2p": [dict(id=q["urs_id"], path=path_of(q["urs_id"]), self_idx=row_of[q["urs_id"]])
                for q in p2p],
    }
    res = search(sets)
    for qs in res:
        json.dump(res[qs], open(os.path.join(HERE, "runs", f"infernal_{qs}.json"), "w"))
        runs = json.load(open(os.path.join(HERE, "runs", f"{qs}.json")))
        runs["Infernal"] = res[qs]
        json.dump(runs, open(os.path.join(HERE, "runs", f"{qs}.json"), "w"))
    print("DONE infernal", flush=True)


if __name__ == "__main__":
    main()
