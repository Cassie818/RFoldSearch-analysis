"""Sequence-method rankings on the RNAsolo benchmark: BLASTN and Infernal."""
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import benchmark as BM
from paths import BLAST_BIN, INFERNAL_BIN, NR80_DIR, SRC_SEARCH

sys.path.insert(0, SRC_SEARCH)
from secondary_structure import WC_NITROGEN, detect_pairs

CMBUILD, CMCALIBRATE, CMSEARCH = (os.path.join(INFERNAL_BIN, x) for x in ("cmbuild", "cmcalibrate", "cmsearch"))
CALIB_L = "0.01"
STD = {"A", "C", "G", "U"}
WORK = os.path.join(BM.RET, "infernal_fig2_work")
DB_FA = os.path.join(WORK, "chains.fa")


def sequences():
    """Sequences of the 885 benchmark chains (chain_sequences.tsv has one extra chain)."""
    b = BM.load()
    rows = csv.DictReader(open(os.path.join(BM.DATA_ROOT, "files", "production", "chain_sequences.tsv")),
                          delimiter="\t")
    seqs = {r["chain_id"]: r["sequence"] for r in rows}
    return {t: seqs[t] for t in b.tags}


def blastn():
    b = BM.load()
    exe = lambda name: os.path.join(BLAST_BIN, name) if BLAST_BIN else name  # noqa: E731
    work = tempfile.mkdtemp(prefix="blastn_")
    fasta = os.path.join(work, "db.fa")
    with open(fasta, "w") as f:
        for t, s in sequences().items():
            f.write(f">{t}\n{s}\n")
    subprocess.run([exe("makeblastdb"), "-in", fasta, "-dbtype", "nucl", "-out", os.path.join(work, "db")],
                   check=True, capture_output=True)
    queries = os.path.join(work, "queries.fa")
    with open(queries, "w") as f:
        for q in b.queries:
            f.write(f">{b.tags[q]}\n{sequences()[b.tags[q]]}\n")
    hits = os.path.join(work, "hits.tsv")
    subprocess.run([exe("blastn"), "-query", queries, "-db", os.path.join(work, "db"), "-evalue", "10",
                    "-out", hits, "-outfmt", "6 qseqid sseqid bitscore evalue", "-num_threads", "8",
                    "-max_target_seqs", str(b.n)], check=True, capture_output=True)
    best = {q: np.full(b.n, np.nan) for q in b.queries}
    for line in open(hits):
        qt, tt, bits, _ = line.rstrip("\n").split("\t")
        q, t = b.tag2i[qt], b.tag2i[tt]
        if np.isnan(best[q][t]) or float(bits) > best[q][t]:
            best[q][t] = float(bits)
    BM.save_rankings("BLASTN", {q: [BM.rank_by_score(best[q], q)] for q in b.queries}, b)


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


def infernal(retry_failed):
    b = BM.load()
    os.makedirs(WORK, exist_ok=True)
    with open(DB_FA, "w") as fh:
        for t, s in sequences().items():
            fh.write(f">{t}\n{s.replace('T', 'U')}\n")
    path = BM.rankings_path("Infernal")
    res = json.load(open(path)) if retry_failed else {}
    if retry_failed:
        todo = [t for t, r in res.items() if not r["ranking"] and float(r.get("seconds", 0)) == 0.0]
    else:
        todo = [b.tags[q] for q in sorted(b.test)]
    jobs = [(t, os.path.join(NR80_DIR, t + ".cif")) for t in todo]
    print(f"Infernal: {len(jobs)} queries", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(8) as ex:
        for fut in as_completed([ex.submit(one_query, j) for j in jobs]):
            qid, order, dt, status = fut.result()
            if status != "ok":
                print(f"  {qid}: {status}", flush=True)
            if status == "ok" or not retry_failed:
                res[qid] = dict(ranking=order, seconds=dt)
    os.makedirs(BM.RANKINGS, exist_ok=True)
    json.dump(res, open(path, "w"))
    print(f"wrote {path} ({len(res)} queries) in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=("blastn", "infernal"))
    ap.add_argument("--retry-failed", action="store_true")
    a = ap.parse_args()
    if a.method == "blastn":
        blastn()
    else:
        infernal(a.retry_failed)


if __name__ == "__main__":
    main()
