"""Recompute recurrent-word RMSD with identity-independent base-inclusive atoms."""
from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common import (
    CANDIDATES_PATH,
    DB_PATH,
    MOTIF_DIR,
    OUT_DIR,
    SEED,
    BACKBONE_SUGAR,
    BASE_LOCATORS,
    expected_atoms,
    load_db_pickle,
    load_reference_rmsd_module,
    parse_word,
    parse_atom_site,
    read_tsv,
    stem_parts,
    word_string,
    write_metrics,
    write_tsv,
)


def sampled_pair_median(arrays: list[np.ndarray], rmsd, rng: random.Random, max_pairs: int) -> tuple[float, int]:
    n = len(arrays)
    if n < 2:
        return float("nan"), 0
    n_all = n * (n - 1) // 2
    if n_all <= max_pairs:
        pairs = list(itertools.combinations(range(n), 2))
    else:
        pairs = set()
        while len(pairs) < max_pairs:
            left, right = rng.sample(range(n), 2)
            pairs.add((min(left, right), max(left, right)))
    values = [rmsd.kabsch_rmsd(arrays[left], arrays[right]) for left, right in pairs]
    return float(np.median(values)), len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--candidates", type=Path, default=CANDIDATES_PATH)
    parser.add_argument("--residue-map", type=Path, default=MOTIF_DIR / "chain_residue_map.json")
    parser.add_argument("--cif-dir", type=Path, default=Path.home() / "Desktop" / "NR80_local")
    parser.add_argument("--frequency-controls", type=Path, default=OUT_DIR / "frequency_matched_random_words.tsv")
    parser.add_argument("--five-anchor-sequence-controls", type=Path, default=MOTIF_DIR / "global_sequence_matched_control_production_new_k4_aa.tsv")
    parser.add_argument("--five-anchor-control-summary", type=Path, default=MOTIF_DIR / "global_sequence_matched_control_production_new_k4_aa_summary.tsv")
    parser.add_argument("--output", type=Path, default=OUT_DIR / "base_inclusive_rmsd.tsv")
    parser.add_argument("--skip-log", type=Path, default=OUT_DIR / "base_inclusive_rmsd_skips.tsv")
    parser.add_argument("--summary", type=Path, default=OUT_DIR / "base_inclusive_rmsd_summary.tsv")
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--max-pairs", type=int, default=400)
    parser.add_argument("--max-control-pairs", type=int, default=25)
    args = parser.parse_args()

    if not args.cif_dir.is_dir():
        write_tsv(args.output, [], ["kmer", "status"])
        write_tsv(args.skip_log, [{"reason": "missing_mmcif_directory", "detail": str(args.cif_dir), "count": 1}])
        write_metrics(args.summary, [
            ("status", "blocked_missing_mmcif_directory"),
            ("random_seed", SEED),
            ("expected_mmcif_directory", args.cif_dir),
            ("n_mmcif_files_found", 0),
            ("base_inclusive_atoms", ",".join(BACKBONE_SUGAR) + "; purine:N9,C4,C8; pyrimidine:N1,C2,C6"),
        ])
        print(f"blocked: missing {args.cif_dir}")
        return

    rmsd = load_reference_rmsd_module()
    entries = load_db_pickle(args.db)["entries"]
    candidates = read_tsv(args.candidates)
    candidate_lookup = {parse_word(row["kmer"]): row for row in candidates}
    frequency = {row["kmer"]: row for row in read_tsv(args.frequency_controls)}
    five_anchor_sequence = {row["kmer"]: row for row in read_tsv(args.five_anchor_sequence_controls)}
    five_anchor_summary = {row["metric"]: row["value"] for row in read_tsv(args.five_anchor_control_summary)}
    residue_map = json.load(args.residue_map.open())
    rng = np.random.default_rng(SEED)
    pair_rng = random.Random(SEED)
    pools_by_sequence: dict[str, list[tuple[tuple[int, ...], np.ndarray]]] = defaultdict(list)
    observed: dict[tuple[int, ...], list[tuple[str, np.ndarray]]] = defaultdict(list)
    skips = Counter()

    for entry_number, entry in enumerate(entries, start=1):
        stem = Path(entry["id"]).stem
        cif_path = args.cif_dir / f"{stem}.cif"
        if not cif_path.exists() or stem not in residue_map:
            skips["missing_cif_or_residue_map"] += max(0, len(entry["tokens"]) - 3)
            continue
        _pdb, model, chain = stem_parts(stem)
        residues = parse_atom_site(cif_path, chain, model)
        mask = np.asarray(entry["mask"], dtype=bool)
        author_numbers = residue_map[stem]
        if len(mask) != len(author_numbers):
            skips["mask_residue_map_length_mismatch"] += max(0, len(entry["tokens"]) - 3)
            continue
        valid_numbers = [number for number, valid in zip(author_numbers, mask) if valid]
        tokens = np.asarray(entry["tokens"], dtype=int)
        for start in range(len(tokens) - 3):
            word = tuple(map(int, tokens[start : start + 4]))
            numbers = valid_numbers[start : start + 4]
            selected = [residues.get(number) for number in numbers]
            if any(value is None for value in selected):
                skips["residue_not_found_in_mmcif"] += 1
                continue
            bases = [value[0] for value in selected]
            if any(base not in BASE_LOCATORS for base in bases):
                skips["modified_or_noncanonical_residue"] += 1
                continue
            coordinates = []
            missing = False
            for base, (_name, atoms) in zip(bases, selected):
                names = expected_atoms(base)
                if any(name not in atoms for name in names):
                    missing = True
                else:
                    coordinates.extend(atoms[name] for name in names)
            if missing:
                skips["missing_base_inclusive_atoms"] += 1
                continue
            sequence = "".join(bases)
            coordinate_array = np.asarray(coordinates, dtype=np.float32)
            pools_by_sequence[sequence].append((word, coordinate_array))
            if word in candidate_lookup:
                observed[word].append((sequence, coordinate_array))
        if entry_number % 100 == 0:
            print(f"parsed {entry_number}/{len(entries)} mmCIF chains", flush=True)

    all_pool = [item for pool in pools_by_sequence.values() for item in pool]
    non_target_pool_cache: dict[tuple[str, tuple[int, ...]], list[tuple[tuple[int, ...], np.ndarray]]] = {}
    rows = []
    for word, candidate in candidate_lookup.items():
        occurrences = observed.get(word, [])
        arrays = [item[1] for item in occurrences]
        candidate_rmsd, n_pairs = sampled_pair_median(arrays, rmsd, pair_rng, args.max_pairs)
        length_values = []
        sequence_values = []
        for _repeat in range(args.repeats):
            if len(arrays) >= 2:
                sampled = [all_pool[int(rng.integers(len(all_pool)))][1] for _ in arrays]
                length_values.append(sampled_pair_median(sampled, rmsd, pair_rng, args.max_control_pairs)[0])
                sequence_sample = []
                valid = True
                for sequence, _coordinates in occurrences:
                    cache_key = (sequence, word)
                    if cache_key not in non_target_pool_cache:
                        non_target_pool_cache[cache_key] = [item for item in pools_by_sequence[sequence] if item[0] != word]
                    choices = non_target_pool_cache[cache_key]
                    if not choices:
                        valid = False
                        break
                    sequence_sample.append(choices[int(rng.integers(len(choices)))][1])
                if valid:
                    sequence_values.append(sampled_pair_median(sequence_sample, rmsd, pair_rng, args.max_control_pairs)[0])
        rows.append({
            "kmer": word_string(word),
            "n_occurrences_base_inclusive": len(arrays),
            "base_inclusive_candidate_sampled_pair_median_rmsd": candidate_rmsd,
            "base_inclusive_candidate_n_pairs": n_pairs,
            "base_inclusive_length_random_median_rmsd": float(np.median(length_values)) if length_values else float("nan"),
            "base_inclusive_sequence_matched_median_rmsd": float(np.median(sequence_values)) if sequence_values else float("nan"),
            "five_anchor_all_pair_median_rmsd": frequency[word_string(word)]["candidate_all_pair_median_rmsd"],
            "five_anchor_cross_family_median_rmsd": candidate["cross_med_rmsd"],
            "five_anchor_length_random_median_rmsd": five_anchor_summary["length_matched_random_median_rmsd"],
            "five_anchor_sequence_matched_median_rmsd": five_anchor_sequence[word_string(word)]["control_median_of_medians"],
            "random_seed": SEED,
            "n_repeats": args.repeats,
            "max_pairs_per_comparison": args.max_pairs,
            "max_pairs_per_control_replicate": args.max_control_pairs,
        })
        if len(rows) % 100 == 0:
            print(f"computed base-inclusive RMSD for {len(rows)}/{len(candidate_lookup)} words", flush=True)
    write_tsv(args.output, rows)
    write_tsv(args.skip_log, [{"reason": reason, "count": count, "detail": "."} for reason, count in sorted(skips.items())])
    complete = [row for row in rows if np.isfinite(float(row["base_inclusive_candidate_sampled_pair_median_rmsd"]))]
    metrics = [
        ("status", "complete"), ("random_seed", SEED), ("n_repeats", args.repeats),
        ("max_pairs_per_comparison", args.max_pairs),
        ("max_pairs_per_control_replicate", args.max_control_pairs),
        ("n_mmcif_files_found", len(list(args.cif_dir.glob("*.cif")))),
        ("n_candidates", len(rows)), ("n_candidates_with_base_inclusive", len(complete)),
        ("base_inclusive_candidate_median_rmsd", float(np.median([float(row["base_inclusive_candidate_sampled_pair_median_rmsd"]) for row in complete]))),
        ("base_inclusive_length_random_median_rmsd", float(np.median([float(row["base_inclusive_length_random_median_rmsd"]) for row in complete]))),
        ("base_inclusive_sequence_matched_median_rmsd", float(np.median([float(row["base_inclusive_sequence_matched_median_rmsd"]) for row in complete if np.isfinite(float(row["base_inclusive_sequence_matched_median_rmsd"]))]))),
        ("five_anchor_candidate_all_pair_median_rmsd", float(np.median([float(row["five_anchor_all_pair_median_rmsd"]) for row in complete]))),
        ("five_anchor_length_random_median_rmsd", five_anchor_summary["length_matched_random_median_rmsd"]),
        ("five_anchor_sequence_matched_median_rmsd", five_anchor_summary["global_sequence_matched_control_median_rmsd"]),
        ("rmsd_definition_source", "scripts/07_motifs/cross_family_kmer_mining.py:kabsch_rmsd/pairwise_rmsd"),
    ]
    for reason, count in sorted(skips.items()):
        metrics.append((f"skipped_{reason}", count))
    write_metrics(args.summary, metrics)
    print(f"wrote {args.output}")
    print(f"wrote {args.skip_log}")
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
