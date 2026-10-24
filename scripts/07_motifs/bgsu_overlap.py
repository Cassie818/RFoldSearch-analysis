"""BGSU 3D Motif Atlas overlap analysis for cross-family k-mer candidates."""
from __future__ import annotations
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from residue_keys import (
    chain_key, cif_residue_keys, parse_bgsu as parse_bgsu_full,
    parse_key_string, parse_stem,
)

REPO = Path(__file__).resolve().parents[2]
DB = REPO / "files" / "models" / "db.pt"
IL_CSV = REPO / "files" / "motifs" / "bgsu_il_current_2026-08-03.csv"
HL_CSV = REPO / "files" / "motifs" / "bgsu_hl_current_2026-08-03.csv"
DEFAULT_CIF_DIR = Path.home() / "Desktop" / "NR80_local"
CAND_TSV = REPO / "files" / "motifs" / "kmer_candidates_production_new_k4_aa.tsv"
OUT_TSV = REPO / "files" / "motifs" / "kmer_bgsu_overlap_production_new_k4_aa.tsv"
CACHE_JSON = REPO / "files" / "motifs" / "chain_residue_key_map.json"
SUMMARY_TSV = REPO / "files" / "motifs" / "bgsu_overlap_production_new_k4_aa_summary.tsv"

sys.path.insert(0, str(SCRIPT_DIR / "word_controls"))
from common import load_db_pickle


def stem_parts(stem: str):
    return parse_stem(stem)


def parse_bgsu(csv_path: Path, kind: str):
    del kind  # Retained in the public API for compatibility.
    return parse_bgsu_full(csv_path)


def build_chain_residue_map(stems, cif_dir: Path, cache_json: Path):
    """Map each chain position to a full, insertion-aware residue key."""
    if cache_json.exists():
        return json.load(open(cache_json))
    out = {}
    n_ok = n_miss = 0
    for stem in stems:
        cif = cif_dir / f"{stem}.cif"
        if not cif.exists():
            n_miss += 1
            continue
        try:
            keys = cif_residue_keys(cif, stem)
            if not keys:
                n_miss += 1
                continue
            out[stem] = keys
            n_ok += 1
        except Exception as e:
            n_miss += 1
    json.dump(out, open(cache_json, "w"))
    print(f"residue map: {n_ok} chains cached, {n_miss} skipped → {cache_json}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=CAND_TSV)
    parser.add_argument("--output", type=Path, default=OUT_TSV)
    parser.add_argument("--db", type=Path, default=DB)
    parser.add_argument("--il_csv", type=Path, default=IL_CSV)
    parser.add_argument("--hl_csv", type=Path, default=HL_CSV)
    parser.add_argument("--cache", type=Path, default=CACHE_JSON)
    parser.add_argument("--cif-dir", type=Path, default=DEFAULT_CIF_DIR)
    parser.add_argument("--summary", type=Path, default=SUMMARY_TSV)
    args = parser.parse_args()

    print("parsing BGSU IL ...")
    il_motifs = parse_bgsu(args.il_csv, "IL")
    print(f"  {len(il_motifs)} IL motif groups")
    print("parsing BGSU HL ...")
    hl_motifs = parse_bgsu(args.hl_csv, "HL")
    print(f"  {len(hl_motifs)} HL motif groups")
    all_motifs = {**il_motifs, **hl_motifs}

    # Build full (PDB, model, chain) -> [(motif_id, residue-key set)].
    chain_to_motifs = defaultdict(list)
    for mid, occs in all_motifs.items():
        for occ in occs:
            chains_in_occ = {chain_key(parse_key_string(value)) for value in occ}
            for pc in chains_in_occ:
                residues_on_pc = {
                    value for value in occ
                    if chain_key(parse_key_string(value)) == pc
                }
                chain_to_motifs[pc].append((mid, residues_on_pc))
    print(f"  BGSU annotates {len(chain_to_motifs)} (PDB, model, chain) tuples")

    # Load our chains
    db = load_db_pickle(args.db)
    entries = db["entries"]
    stem2idx = {Path(e["id"]).stem: i for i, e in enumerate(entries)}
    print(f"  our db: {len(stem2idx)} chains")

    # Build full residue-key mapping (cached separately from the legacy integer map).
    residue_map = build_chain_residue_map(list(stem2idx.keys()), args.cif_dir, args.cache)

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
        residue_keys = residue_map[stem]
        mask = np.asarray(e["mask"], dtype=bool)
        if len(residue_keys) != len(mask):
            continue
        # Tokens are stored only for valid residues, whereas residue keys
        # and coordinates retain unresolved positions.
        valid_residue_keys = [key for key, valid in zip(residue_keys, mask) if valid]
        if len(valid_residue_keys) != len(toks):
            continue
        chain_motifs = chain_to_motifs[parts]

        for p in range(len(toks) - K + 1):
            km = tuple(int(t) for t in toks[p:p + K])
            if km not in target_kmers:
                continue
            km_str = target_kmers[km]["kmer"]
            kmer_residue_set = set(valid_residue_keys[p:p + K])
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
        w = csv.DictWriter(f, list(rows_out[0].keys()), delimiter="\t", lineterminator="\n")
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

    target = next(row for row in rows_out if row["kmer"] == "44-48-62-71")
    summary_rows = [
        ("source_il_url", "https://rna.bgsu.edu/rna3dhub/motifs/release/il/current/csv"),
        ("source_hl_url", "https://rna.bgsu.edu/rna3dhub/motifs/release/hl/current/csv"),
        ("download_date", "2026-08-03"),
        ("n_candidates", len(rows_out)),
        ("n_candidates_with_bgsu_annotated_occurrences", len(frs)),
        ("frac_ge_0.50", int((frs >= 0.50).sum())),
        ("frac_ge_0.25", int((frs >= 0.25).sum())),
        ("frac_gt_0", int((frs > 0).sum())),
        ("median_frac_in_bgsu", float(np.median(frs))),
        ("target_kmer", target["kmer"]),
        ("target_n_occ_db", target["n_occ_db"]),
        ("target_n_occ_in_bgsu", target["n_occ_in_bgsu"]),
        ("target_frac_in_bgsu", target["frac_in_bgsu"]),
        ("target_top_bgsu_motif", target["top_bgsu_motif"]),
        ("target_cross_med_rmsd", target["cross_med_rmsd"]),
        ("target_n_fam", target["n_fam"]),
        ("target_n_clan", target["n_clan"]),
    ]
    with args.summary.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["metric", "value"])
        writer.writerows(summary_rows)
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
