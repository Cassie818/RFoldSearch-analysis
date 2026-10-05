"""Random-window BGSU overlap background using the manuscript overlap rule."""
from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import DB_PATH, MOTIF_DIR, OUT_DIR, SEED, load_db_pickle, quantiles, read_tsv, write_metrics, write_tsv


def load_bgsu_module():
    for optional_dependency in ("torch", "gemmi"):
        try:
            __import__(optional_dependency)
        except (ImportError, OSError):
            sys.modules.setdefault(optional_dependency, types.ModuleType(optional_dependency))
    path = Path(__file__).resolve().parents[1] / "bgsu_overlap.py"
    spec = importlib.util.spec_from_file_location("bgsu_overlap_reference", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--hl", type=Path, default=MOTIF_DIR / "bgsu_hl_current_2026-08-03.csv")
    parser.add_argument("--il", type=Path, default=MOTIF_DIR / "bgsu_il_current_2026-08-03.csv")
    parser.add_argument("--residue-map", type=Path, default=MOTIF_DIR / "chain_residue_map.json")
    parser.add_argument("--candidate-overlap", type=Path, default=MOTIF_DIR / "kmer_bgsu_overlap_production_new_k4_aa.tsv")
    parser.add_argument("--candidate-summary", type=Path, default=MOTIF_DIR / "bgsu_overlap_production_new_k4_aa_summary.tsv")
    parser.add_argument("--output", type=Path, default=OUT_DIR / "bgsu_random_window_background.tsv")
    parser.add_argument("--summary", type=Path, default=OUT_DIR / "bgsu_random_window_background_summary.tsv")
    parser.add_argument("--repeats", type=int, default=25)
    args = parser.parse_args()

    bgsu = load_bgsu_module()
    import json

    motifs = {**bgsu.parse_bgsu(args.il, "IL"), **bgsu.parse_bgsu(args.hl, "HL")}
    chain_to_motif_residues: dict[tuple[str, str], list[set[int]]] = defaultdict(list)
    for occurrences in motifs.values():
        for occurrence in occurrences:
            for pdb_chain in {(pdb, chain) for pdb, chain, _residue in occurrence}:
                chain_to_motif_residues[pdb_chain].append({
                    residue for pdb, chain, residue in occurrence if (pdb, chain) == pdb_chain
                })

    residue_map = json.load(args.residue_map.open())
    entries = load_db_pickle(args.db)["entries"]
    windows: list[bool] = []
    n_covered_chains = 0
    n_mapping_skipped = 0
    for entry in entries:
        stem = Path(entry["id"]).stem
        parts = bgsu.stem_parts(stem)
        if parts is None or parts not in chain_to_motif_residues or stem not in residue_map:
            continue
        mask = np.asarray(entry["mask"], dtype=bool)
        resnums = residue_map[stem]
        if len(resnums) != len(mask):
            n_mapping_skipped += 1
            continue
        valid_resnums = [residue for residue, valid in zip(resnums, mask) if valid]
        if len(valid_resnums) != len(entry["tokens"]):
            n_mapping_skipped += 1
            continue
        n_covered_chains += 1
        motif_sets = chain_to_motif_residues[parts]
        for start in range(len(valid_resnums) - 3):
            residues = set(valid_resnums[start : start + 4])
            windows.append(any(residues & motif_residues for motif_residues in motif_sets))

    overlap_rows = read_tsv(args.candidate_overlap)
    n_candidate_occurrences_covered_chains = sum(int(row["n_occ_db"]) for row in overlap_rows)
    summary_source = {row["metric"]: row["value"] for row in read_tsv(args.candidate_summary)}
    candidate_median = float(summary_source["median_frac_in_bgsu"])
    rng = np.random.default_rng(SEED)
    boolean_windows = np.asarray(windows, dtype=np.int8)
    rows = []
    for repeat in range(args.repeats):
        indices = rng.integers(0, len(boolean_windows), size=n_candidate_occurrences_covered_chains)
        n_overlap = int(boolean_windows[indices].sum())
        rows.append({
            "repeat": repeat + 1,
            "n_sampled_windows": n_candidate_occurrences_covered_chains,
            "n_overlapping_bgsu": n_overlap,
            "background_overlap_fraction": n_overlap / n_candidate_occurrences_covered_chains,
            "random_seed": SEED,
        })
    write_tsv(args.output, rows)
    fractions = [float(row["background_overlap_fraction"]) for row in rows]
    p10, median, p90 = quantiles(fractions)
    write_metrics(args.summary, [
        ("random_seed", SEED),
        ("n_repeats", args.repeats),
        ("n_bgsu_covered_chains", n_covered_chains),
        ("n_chains_skipped_mapping_mismatch", n_mapping_skipped),
        ("n_eligible_random_windows", len(windows)),
        ("n_candidate_occurrences_on_bgsu_covered_chains", n_candidate_occurrences_covered_chains),
        ("background_overlap_median_fraction", median),
        ("background_overlap_p10_fraction", p10),
        ("background_overlap_p90_fraction", p90),
        ("candidate_word_median_overlap_fraction", candidate_median),
        ("candidate_minus_background_median_fraction", candidate_median - median),
        ("overlap_rule", "any residue intersection; imported from scripts/07_motifs/bgsu_overlap.py"),
        ("sampling_scope", "uniform 4-nt windows from BGSU-covered chains with valid residue mapping"),
    ])
    print(f"wrote {args.output}")
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
