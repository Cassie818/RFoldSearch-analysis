"""Build the RFoldseek, RiboSeek and sequence databases for the sampled
NuFold 9-14-2026 accuracy database.
"""
import csv
import json
import os
import random
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
import sys as _sys
_sys.path.insert(0, REPO)
import paths
HERE = os.path.join(REPO, "files", "predicted_retrieval")
ROOT = HERE
MANIFEST = os.path.join(ROOT, "sample", "accuracy_db.tsv")
STRUCTURES = os.path.join(paths.NUFOLD_ROOT, "structures", "rank1")
UP = paths.RFOLDSEEK_REPO
CKPT = os.path.join(UP, "rfoldseek", "assets", "desc_vqvae_best.pt")
TOOLS = paths.BLAST_BIN          # NCBI BLAST+ 2.17 bin dir; empty -> PATH
MAKEBLASTDB = os.path.join(TOOLS, "makeblastdb")
WORKERS = 8
STD = {"A", "C", "G", "U"}
sys.path.insert(0, UP)

_state = {}


def path_of(urs_id):
    return os.path.join(STRUCTURES, f"{urs_id}_rank_1.pdb")


def guard(n, failures, pool):
    if n >= 500 and len(failures) > 0.01 * n:
        pool.terminate()
        raise SystemExit(f"aborting: {len(failures)} failures in first {n}; first: {failures[0]}")


# ------------------------------------------------------------------ RFoldseek

def _init_rfs():
    torch.set_num_threads(1)
    from rfoldseek.tokenizer import load_model
    model, ckpt, _, _ = load_model(CKPT, torch.device("cpu"))
    _state.update(model=model, nm=ckpt.get("norm_mean"), ns=ckpt.get("norm_std"),
                  ab=ckpt.get("active_blocks"))


def _enc_rfs(job):
    i, urs = job
    from rfoldseek.database import structure_to_entry
    from rfoldseek.tokenizer import encode_entry
    try:
        with torch.no_grad():
            enc = encode_entry(_state["model"], structure_to_entry(path_of(urs)),
                               torch.device("cpu"), _state["nm"], _state["ns"],
                               active_blocks=_state["ab"])
        return i, enc, None
    except Exception as exc:
        return i, None, f"{urs}: {type(exc).__name__}: {exc}"[:300]


def build_rfoldseek(ids, out):
    t0 = time.perf_counter()
    res, failures = [None] * len(ids), []
    with Pool(WORKERS, initializer=_init_rfs) as pool:
        for n, (i, enc, err) in enumerate(pool.imap_unordered(_enc_rfs, list(enumerate(ids)), chunksize=16), 1):
            if enc is None:
                failures.append(err or f"{ids[i]}: encode_entry returned None")
            else:
                res[i] = enc
            guard(n, failures, pool)
            if n % 5000 == 0:
                print(f"  rfoldseek {n}/{len(ids)} ({(time.perf_counter()-t0)/60:.1f} min)", flush=True)
    entries = [e for e in res if e is not None]
    for k, e in enumerate(entries):
        e["cluster"] = f"_singleton_{k}"
    torch.save({"n_codes": 80, "entries": entries, "checkpoint": CKPT,
                "n_clusters": len(entries), "clustered": False}, out)
    json.dump(dict(failures=failures, wall_seconds=time.perf_counter() - t0, workers=WORKERS),
              open(out + ".meta.json", "w"), indent=1)
    print(f"rfoldseek db: {len(entries)} entries, {len(failures)} failures, "
          f"{(time.perf_counter()-t0)/60:.1f} min", flush=True)
    return entries


def verify_rfoldseek(entries, k=20):
    from rfoldseek.database import structure_to_entry
    from rfoldseek.tokenizer import encode_entry, load_model
    model, ckpt, _, _ = load_model(CKPT, torch.device("cpu"))
    random.seed(0)
    ok = 0
    for e in random.sample(entries, k):
        with torch.no_grad():
            r = encode_entry(model, structure_to_entry(e["id"]), torch.device("cpu"),
                             ckpt.get("norm_mean"), ckpt.get("norm_std"),
                             active_blocks=ckpt.get("active_blocks"))
        ok += np.array_equal(np.asarray(r["tokens"]), np.asarray(e["tokens"]))
    print(f"rfoldseek verification: {ok}/{k} identical on independent re-encoding", flush=True)
    if ok != k:
        raise SystemExit("token verification failed")


# ------------------------------------------------------------------ RiboSeek

def _init_ribo():
    from riboseek.alphabet import Alphabet
    _state["alpha"] = Alphabet.from_pretrained("sa20")


def _enc_ribo(job):
    i, urs = job
    from riboseek.features import pdb_to_features
    try:
        f = pdb_to_features(path_of(urs))
        labels = _state["alpha"].encode(f["features"])
        key = f"{urs}_rank_1" + (f"_{f['chain_id']}" if f["chain_id"] else "")
        return i, key, labels.tolist(), None
    except Exception as exc:
        return i, None, None, f"{urs}: {type(exc).__name__}: {exc}"[:300]


def build_riboseek(ids, out):
    t0 = time.perf_counter()
    encoded, key_to_index, failures = {}, {}, []
    with Pool(WORKERS, initializer=_init_ribo) as pool:
        for n, (i, key, labels, err) in enumerate(pool.imap_unordered(_enc_ribo, list(enumerate(ids)), chunksize=16), 1):
            if err:
                failures.append(err)
            else:
                encoded[key] = {"labels": labels, "length": len(labels)}
                key_to_index[key] = i
            guard(n, failures, pool)
    json.dump(encoded, open(out, "w"))
    json.dump(dict(key_to_index=key_to_index, failures=failures,
                   wall_seconds=time.perf_counter() - t0, workers=WORKERS),
              open(out + ".meta.json", "w"))
    print(f"riboseek db: {len(encoded)} chains, {len(failures)} failures, "
          f"{(time.perf_counter()-t0)/60:.1f} min", flush=True)


# ------------------------------------------------------------------ sequences

def seq_from_pdb(path):
    out, last, chain = [], None, None
    for line in open(path):
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if chain is None:
            chain = line[21]
        elif line[21] != chain:
            break
        key = line[22:27]
        if key != last:
            last = key
            name = line[17:20].strip()
            out.append(name if name in STD else "N")
    return "".join(out)


def build_sequences(rows, out):
    mism = 0
    with open(out, "w") as fh:
        for i, r in enumerate(rows):
            s = seq_from_pdb(path_of(r["urs_id"]))
            mism += len(s) != int(float(r["sequence_length"]))
            fh.write(f">t{i}\n{s}\n")
    subprocess.run([MAKEBLASTDB, "-in", out, "-dbtype", "nucl", "-out", out],
                   check=True, capture_output=True)
    print(f"sequences: {len(rows)} written, {mism} length mismatches vs metadata; BLAST db built", flush=True)


def main():
    rows = list(csv.DictReader(open(MANIFEST), delimiter="\t"))
    ids = [r["urs_id"] for r in rows]
    missing = [u for u in ids if not os.path.exists(path_of(u))]
    if missing:
        raise SystemExit(f"{len(missing)} manifest entries have no structure file, e.g. {missing[:3]}")
    print(f"manifest: {len(ids)} entries, all structure files present", flush=True)

    out = os.path.join(HERE, "db")
    os.makedirs(out, exist_ok=True)
    rfs = os.path.join(out, "rfoldseek.pt")
    entries = build_rfoldseek(ids, rfs) if not os.path.exists(rfs + ".meta.json") \
        else torch.load(rfs, map_location="cpu", weights_only=False)["entries"]
    verify_rfoldseek(entries)
    rb = os.path.join(out, "riboseek_sa20.json")
    if not os.path.exists(rb + ".meta.json"):
        build_riboseek(ids, rb)
    fa = os.path.join(out, "sequences.fa")
    if not os.path.exists(fa + ".nin"):
        build_sequences(rows, fa)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
