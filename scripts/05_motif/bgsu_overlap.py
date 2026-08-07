"""BGSU 3D Motif Atlas overlap analysis for cross-family k-mer candidates.

For each k-mer candidate (from cross_family_kmer_mining), check whether
its occurrence (chain, residue range) overlaps with any annotated
internal-loop (IL_*) or hairpin-loop (HL_*) motif from the BGSU 3D
Motif Atlas. Report per-candidate overlap rate and aggregate statistics.

Inputs:
    /tmp/bgsu_il.csv, /tmp/bgsu_hl.csv  (downloaded BGSU motif lists)
    files/motifs_v2/kmer_candidates.tsv

Outputs:
    files/motifs_v2/kmer_bgsu_overlap.tsv  per-candidate overlap stats
    files/motifs_v2/chain_residue_map.json residue-number cache
    Console: summary statistics
"""
from __future__ import annotations
import argparse, csv, json, os, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import gemmi

REPO = Path(__file__).resolve().parents[2]
DB = REPO / "files" / "models" / "db.pt"
IL_CSV = Path("/tmp/bgsu_il.csv")
HL_CSV = Path("/tmp/bgsu_hl.csv")
CIF_DIR = Path("/Users/cassie/Desktop/NR80_local")
CAND_TSV = REPO / "files" / "motifs_v2" / "kmer_candidates.tsv"
OUT_TSV = REPO / "files" / "motifs_v2" / "kmer_bgsu_overlap.tsv"
CACHE_JSON = REPO / "files" / "motifs_v2" / "chain_residue_map.json"

RNA = {"A","C","G","U","PSU","5MU","4SU","H2U","UR3","5MC","OMC","5BU",
       "OMG","1MG","2MG","M2G","7MG","YG","YYG","1MA","6MZ","MA6","I"}

# stem format: PDB_00<6-char hex>_<model>_<chain>. The PDB id is the
# trailing 4 chars of the 6-char block (the leading "00" pads new-style
# 6-char PDB ids that the RCSB started issuing in 2024).
STEM_RE = re.compile(r"PDB_00([0-9A-Z]{6})_(\d+)_(.+)$", re.I)
TOKEN_RE = re.compile(r'"?([0-9A-Za-z]+)\|(\d+)\|([^|]+)\|([^|]+)\|(-?\d+[A-Za-z]?)"?')


def stem_parts(stem: str):
    m = STEM_RE.match(stem)
    if not m:
        return None
    raw = m.group(1).upper()
    # 4-char legacy PDB id padded with leading "00"; new-style 6-char ids
    # do not start with "00".
    pdb = raw[2:] if raw.startswith("00") else raw
    chain = m.group(3)
    return pdb, chain


def parse_bgsu(csv_path: Path, kind: str):
    """Yield (motif_id, list_of_occurrences). Each occurrence is a set of
    (pdb_lower, chain, residue_num_int)."""
    motifs = {}
    cur = None
    with open(csv_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith(">"):
                cur = line[1:].strip()
                motifs[cur] = []
                continue
            tokens = TOKEN_RE.findall(line)
            occ = set()
            for pdb, model, chain, nt, resnum in tokens:
                # strip insertion code suffix if present
                resnum_int = int(re.match(r"-?\d+", resnum).group(0))
                occ.add((pdb.upper(), chain, resnum_int))
            if occ:
                motifs[cur].append(occ)
    return motifs


def build_chain_residue_map(stems):
    """For each chain stem we have a CIF for, return a list of author
    residue numbers in the order produced by our parser. If cached, reuse."""
    if CACHE_JSON.exists():
        return json.load(open(CACHE_JSON))
    out = {}
    n_ok = n_miss = 0
    for stem in stems:
        cif = CIF_DIR / f"{stem}.cif"
        if not cif.exists():
            n_miss += 1
            continue
        try:
            st = gemmi.read_structure(str(cif))
            m = st[0]
            nums = []
            parts = stem_parts(stem)
            if parts is None:
                continue
            target_chain = parts[1]
            for c in m:
                if c.name != target_chain:
                    continue
                for r in c:
                    if r.name in RNA:
                        nums.append(int(r.seqid.num))
            out[stem] = nums
            n_ok += 1
        except Exception as e:
            n_miss += 1
    json.dump(out, open(CACHE_JSON, "w"))
    print(f"residue map: {n_ok} chains cached, {n_miss} skipped → {CACHE_JSON}")
    return out


def main():
    global CACHE_JSON

    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=CAND_TSV)
    parser.add_argument("--output", type=Path, default=OUT_TSV)
    parser.add_argument("--db", type=Path, default=DB)
    parser.add_argument("--il_csv", type=Path, default=IL_CSV)
    parser.add_argument("--hl_csv", type=Path, default=HL_CSV)
    parser.add_argument("--cache", type=Path, default=CACHE_JSON)
    args = parser.parse_args()

    print("parsing BGSU IL ...")
    il_motifs = parse_bgsu(args.il_csv, "IL")
    print(f"  {len(il_motifs)} IL motif groups")
    print("parsing BGSU HL ...")
    hl_motifs = parse_bgsu(args.hl_csv, "HL")
    print(f"  {len(hl_motifs)} HL motif groups")
    all_motifs = {**il_motifs, **hl_motifs}

    # Build chain -> [(motif_id, residue_set)]
    chain_to_motifs = defaultdict(list)
    for mid, occs in all_motifs.items():
        for occ in occs:
            chains_in_occ = {(p, c) for (p, c, _) in occ}
            for pc in chains_in_occ:
                residues_on_pc = {r for (p2, c2, r) in occ if (p2, c2) == pc}
                chain_to_motifs[pc].append((mid, residues_on_pc))
    print(f"  BGSU annotates {len(chain_to_motifs)} (PDB, chain) tuples")

    # Load our chains
    db = torch.load(args.db, map_location="cpu", weights_only=False)
    entries = db["entries"]
    stem2idx = {Path(e["id"]).stem: i for i, e in enumerate(entries)}
    print(f"  our db: {len(stem2idx)} chains")

    # Build residue-number mapping (cached)
    CACHE_JSON = args.cache
    residue_map = build_chain_residue_map(list(stem2idx.keys()))

    # How many of our chains have ANY BGSU annotation?
    annotated = 0
    for stem in stem2idx:
        p = stem_parts(stem)
        if p and p in chain_to_motifs:
            annotated += 1
    print(f"  our chains with BGSU annotation: {annotated} / {len(stem2idx)}")

    # ── Walk all k-mer candidates, compute overlap ──────────────────────
    # We re-derive each occurrence from db.pt + tokens, scanning the chain
    cand_rows = list(csv.DictReader(open(args.candidates), delimiter="\t"))
    rows_out = []
    K = len(cand_rows[0]["kmer"].split("-"))
    target_kmers = {tuple(int(t) for t in r["kmer"].split("-")): r
                    for r in cand_rows}

    # Per-occurrence overlap collection
    occ_overlap_per_kmer = defaultdict(list)  # kmer_str -> list[bool]
    motif_hits_per_kmer = defaultdict(list)   # kmer_str -> list of motif_id

    for stem, idx in stem2idx.items():
        e = entries[idx]
        toks = np.asarray(e["tokens"], dtype=np.int32)
        mask = np.asarray(e["mask"], dtype=bool)
        if len(toks) < K:
            continue
        parts = stem_parts(stem)
        if parts is None or parts not in chain_to_motifs:
            continue
        if stem not in residue_map:
            continue
        resnums = residue_map[stem]
        mask = np.asarray(e["mask"], dtype=bool)
        if len(resnums) != len(mask):
            continue
        # Tokens are stored only for valid residues, whereas residue numbers
        # and coordinates retain unresolved positions.
        valid_resnums = [resnum for resnum, valid in zip(resnums, mask) if valid]
        if len(valid_resnums) != len(toks):
            continue
        chain_motifs = chain_to_motifs[parts]

        for p in range(len(toks) - K + 1):
            km = tuple(int(t) for t in toks[p:p + K])
            if km not in target_kmers:
                continue
            km_str = target_kmers[km]["kmer"]
            kmer_residue_set = set(valid_resnums[p:p + K])
            hit = False
            hit_mid = None
            for mid, motif_res in chain_motifs:
                if kmer_residue_set & motif_res:  # any overlap
                    hit = True
                    hit_mid = mid
                    break
            occ_overlap_per_kmer[km_str].append(hit)
            if hit:
                motif_hits_per_kmer[km_str].append(hit_mid)

    # Aggregate
    for r in cand_rows:
        km_str = r["kmer"]
        ovs = occ_overlap_per_kmer.get(km_str, [])
        hits = motif_hits_per_kmer.get(km_str, [])
        n_total = len(ovs)
        n_hit = sum(ovs)
        if hits:
            mid_count = defaultdict(int)
            for h in hits:
                mid_count[h] += 1
            top_mid = max(mid_count.items(), key=lambda x: x[1])
            top_mid_str = f"{top_mid[0]} ({top_mid[1]})"
        else:
            top_mid_str = ""
        rows_out.append({
            "kmer": km_str,
            "n_occ_db": n_total,
            "n_occ_in_bgsu": n_hit,
            "frac_in_bgsu": (n_hit / n_total) if n_total else float("nan"),
            "top_bgsu_motif": top_mid_str,
            "cross_med_rmsd": r["cross_med_rmsd"],
            "n_fam": r["n_fam"],
            "n_clan": r["n_clan"],
        })

    rows_out.sort(key=lambda r: -r["frac_in_bgsu"]
                  if not np.isnan(r["frac_in_bgsu"]) else 1.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        w = csv.DictWriter(f, list(rows_out[0].keys()), delimiter="\t")
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print(f"wrote {args.output}  ({len(rows_out)} rows)")

    # ── Summary ─────────────────────────────────────────────────────────
    frs = [r["frac_in_bgsu"] for r in rows_out
           if not np.isnan(r["frac_in_bgsu"])]
    frs = np.asarray(frs)
    print("\n=== Summary ===")
    print(f"candidates with any BGSU-annotated chain occurrences: {len(frs)}")
    if len(frs):
        print(f"  frac in BGSU >= 0.50 : {(frs >= 0.50).sum()}")
        print(f"  frac in BGSU >= 0.25 : {(frs >= 0.25).sum()}")
        print(f"  frac in BGSU >  0    : {(frs > 0).sum()}")
        print(f"  median frac          : {np.median(frs):.3f}")


if __name__ == "__main__":
    main()
