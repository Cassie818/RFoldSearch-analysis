"""Infernal single-query-CM baseline for the main (Fig. 2) Rfam benchmark.

Same protocol as run_infernal.py, on the 886-chain experimental benchmark: each
of the 190 held-out test chains is turned into a single-sequence covariance
model built with its 3D-derived secondary structure, calibrated with a tiny
random-sequence length (we rank by bit score, not E-value), and searched
against the benchmark's own sequences with cmsearch. Targets are ranked by
descending bit score, self excluded.

Ground truth follows Fig. 2: a non-self target is a family positive if it shares
the query's Rfam family; at clan level, positives share the clan and same-family
targets are neutral. Sens@1FP = positives retrieved before the first false
positive, over min(#positives, 200). Chains without an Rfam family are neutral
as queries (dropped) and, as targets, are false positives.

Outputs runs_fig2/infernal.json {chain_id: {ranking:[chain_id...], seconds}}
and prints family/clan Sens@1FP.
"""
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PC = os.path.abspath(os.path.join(HERE, "..", ".."))            # repository root
sys.path.insert(0, PC)
import paths                                                     # noqa: E402
NR80 = paths.NR80_DIR          # experimental benchmark structures (external input)
sys.path.insert(0, os.path.join(PC, "src", "search"))
from secondary_structure import detect_pairs, WC_NITROGEN             # noqa: E402

INF = paths.INFERNAL_BIN       # Infernal 1.1.5 binaries; empty -> PATH
CMBUILD, CMSEARCH = os.path.join(INF, "cmbuild"), os.path.join(INF, "cmsearch")
CMCALIBRATE = os.path.join(INF, "cmcalibrate")
CALIB_L = "0.01"
STD = {"A", "C", "G", "U"}
RET = os.path.join(PC, "files", "production", "retrieval")
OUT = os.path.join(RET, "infernal_fig2_work")                   # scratch: FASTA database


def residue_atoms(path):
    """Per-residue (base, C1' xyz, WC-nitrogen xyz) for the first model/chain (CIF)."""
    cols, in_loop, model, chain = {}, False, None, None
    res, order = {}, []
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
            r = res.setdefault(key, {"base": comp})
            if key not in order:
                order.append(key)
            if atom in ("C1'", "C1*"):
                r["C1'"] = xyz
            if atom == WC_NITROGEN.get(comp if comp in WC_NITROGEN else "", ""):
                r["N"] = xyz
        elif in_loop and res and line.startswith(("#", "loop_")):
            break
    bases, c1, wc = [], [], []
    for key in order:
        r = res[key]
        bases.append(r["base"] if r["base"] in STD else "N")
        c1.append(r.get("C1'", [np.nan] * 3))
        wc.append(r.get("N", [np.nan] * 3))
    return bases, np.asarray(wc, float), np.asarray(c1, float)


def nested(pairs, n):
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
    return (cm, None) if p.returncode == 0 else (None, p.stderr[:200])
DB_FA = os.path.join(OUT, "chains.fa")
TOPN = 200


def prep_db(keep):
    """FASTA of the 885 benchmark chains (chain_sequences.tsv has one extra,
    PDB_00007SXP_1_A, absent from the labelled benchmark; `keep` is the label set)."""
    os.makedirs(OUT, exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(PC, "files/production/chain_sequences.tsv")),
                               delimiter="\t"))
    rows = [r for r in rows if r["chain_id"] in keep]
    with open(DB_FA, "w") as fh:
        for r in rows:
            fh.write(f">{r['chain_id']}\n{r['sequence'].replace('T', 'U')}\n")
    return [r["chain_id"] for r in rows]


def labels():
    fam, clan = {}, {}
    for r in csv.DictReader(open(os.path.join(PC, "Splits/labels.tsv")), delimiter="\t"):
        cid = os.path.splitext(os.path.basename(r["entry_id"]))[0]
        fam[cid] = r["rfam_acc"] or None
        clan[cid] = r["clan_acc"] or None
    return fam, clan


def one_query(job):
    qid, path = job
    try:
        bases, wc, c1 = residue_atoms(path)
        seq = "".join(bases).replace("T", "U")
        ss = nested(detect_pairs(bases, wc, c1), len(bases))
        with tempfile.TemporaryDirectory(prefix="inf2_") as tmp:
            cm, err = build_cm(qid, seq, ss, tmp)
            if cm is None:
                return qid, [], 0.0, f"cmbuild: {err}"
            t = time.perf_counter()
            c = subprocess.run([CMCALIBRATE, "--cpu", "1", "--tailp", "0.5", "-L", CALIB_L, cm],
                               capture_output=True, text=True)
            if c.returncode:
                return qid, [], 0.0, f"cmcalibrate: {c.stderr[:120]}"
            tbl = os.path.join(tmp, "o.tbl")
            s = subprocess.run([CMSEARCH, "--cpu", "1", "--toponly", "-T", "0", "--incT", "0",
                                "--tblout", tbl, "-o", os.devnull, cm, DB_FA],
                               capture_output=True, text=True)
            dt = time.perf_counter() - t
            if s.returncode:
                return qid, [], 0.0, f"cmsearch: {s.stderr[:120]}"
            best = {}
            for line in open(tbl):
                if line.startswith("#") or not line.strip():
                    continue
                f = line.split()
                tid, bits = f[0], float(f[14])
                if tid == qid:
                    continue
                if tid not in best or bits > best[tid]:
                    best[tid] = bits
            order = sorted(best, key=lambda t: (-best[t], t))
            return qid, order, dt, "ok"
    except Exception as exc:
        return qid, [], 0.0, f"{type(exc).__name__}: {exc}"[:150]


def sens1fp(ranking, pos, neu):
    p = [t for t in ranking if t not in neu]
    n_rel = len(pos)
    if n_rel == 0:
        return None
    tp = 0
    for t in p:
        if t in pos:
            tp += 1
        else:
            break
    denom = min(n_rel, TOPN)
    return min(tp, denom) / denom


def main():
    fam, clan = labels()
    ids = prep_db(set(fam))          # 885 labelled benchmark chains
    idset = set(ids)
    test = [json.loads(l)["id"] for l in open(os.path.join(PC, "Splits/pdb_out/test.json")) if l.strip()]
    queries = []
    for p in test:
        cid = os.path.splitext(os.path.basename(p))[0]
        cif = os.path.join(NR80, cid + ".cif")
        if os.path.exists(cif):
            queries.append((cid, cif))
    print(f"db {len(ids)} chains, {len(queries)} test queries with CIFs", flush=True)

    res, npair_fail = {}, 0
    t0 = time.perf_counter()
    with ProcessPoolExecutor(8) as ex:
        for n, fut in enumerate(as_completed([ex.submit(one_query, j) for j in queries]), 1):
            qid, order, dt, status = fut.result()
            res[qid] = dict(ranking=order, seconds=dt)
            if status != "ok":
                npair_fail += 1
                print(f"  [{n}/{len(queries)}] {qid}: {status}", flush=True)
    json.dump(res, open(os.path.join(RET, "infernal_rankings_test.json"), "w"))
    print(f"Infernal done: {len(res)} queries in {(time.perf_counter()-t0)/60:.1f} min, "
          f"median {np.median([v['seconds'] for v in res.values()])*1000:.0f} ms/search", flush=True)

    # quick Sens@1FP (family and clan), same-family neutral at clan level
    for level, lab in (("family", fam), ("clan", clan)):
        vals = []
        for qid, r in res.items():
            ql = lab.get(qid)
            if not ql:
                continue
            if level == "family":
                pos = {t for t in ids if fam.get(t) == ql and t != qid}
                neu = {t for t in ids if not fam.get(t)} | {qid}
            else:
                pos = {t for t in ids if clan.get(t) == ql and fam.get(t) != fam.get(qid) and t != qid}
                neu = ({t for t in ids if not clan.get(t)}
                       | {t for t in ids if fam.get(t) == fam.get(qid)} | {qid})
            s = sens1fp(r["ranking"], pos, neu)
            if s is not None:
                vals.append(s)
        print(f"  Infernal {level:6s} Sens@1FP = {np.mean(vals):.3f}  (n={len(vals)})", flush=True)


if __name__ == "__main__":
    main()
