"""Recompute recurrent-word RMSD with identity-independent base-inclusive atoms.

As in the five-anchor frequency control, candidate and control RMSDs use only
occurrence pairs from different Rfam clans (an unclanned family is its own
clan), so homologous pairs do not enter either side of the comparison.
"""
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
    load_labels,
    load_reference_rmsd_module,
    parse_word,
    parse_atom_site,
    read_tsv,
    stem_parts,
    word_string,
    write_metrics,
    write_tsv,
)


def cross_clan_pair_median(
    arrays: list[np.ndarray], clans: list[str], rmsd, rng: random.Random, max_pairs: int,
) -> tuple[float, int]:
    """Median RMSD over sampled occurrence pairs whose clans differ."""
    n = len(arrays)
    n_all = n * (n - 1) // 2
    n_eligible = n_all - sum(count * (count - 1) // 2 for count in Counter(clans).values())
    if n_eligible <= 0:
        return float("nan"), 0
    if n_eligible <= max_pairs:
        pairs = [(left, right) for left, right in itertools.combinations(range(n), 2) if clans[left] != clans[right]]
    else:
        pairs = set()
        while len(pairs) < max_pairs:
            left, right = rng.sample(range(n), 2)
            if clans[left] != clans[right]:
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
    parser.add_argument("--frequency-control-summary", type=Path, default=OUT_DIR / "frequency_matched_random_words_summary.tsv")
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
    residue_map = json.load(args.residue_map.open())
    labels = load_labels()
    rng = np.random.default_rng(SEED)
    pair_rng = random.Random(SEED)
    # Pool items and observed occurrences carry the clan of their chain.
    pools_by_sequence: dict[str, list[tuple[tuple[int, ...], np.ndarray, str]]] = defaultdict(list)
    observed: dict[tuple[int, ...], list[tuple[str, np.ndarray, str]]] = defaultdict(list)
    skips = Counter()

    for entry_number, entry in enumerate(entries, start=1):
        stem = Path(entry["id"]).stem
        family, clan = labels.get(str(entry["id"]), labels.get(Path(entry["id"]).name, ("", "")))
        if not family:
            skips["unlabelled_chain"] += max(0, len(entry["tokens"]) - 3)
            continue
        clan = clan or f"family:{family}"
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
            pools_by_sequence[sequence].append((word, coordinate_array, clan))
            if word in candidate_lookup:
                observed[word].append((sequence, coordinate_array, clan))
        if entry_number % 100 == 0:
            print(f"parsed {entry_number}/{len(entries)} mmCIF chains", flush=True)

    all_pool = [item for pool in pools_by_sequence.values() for item in pool]
    non_target_pool_cache: dict[tuple[str, tuple[int, ...]], list[tuple[tuple[int, ...], np.ndarray, str]]] = {}
    frequency_summary = {row["metric"]: row["value"] for row in read_tsv(args.frequency_control_summary)}
    rows = []
    for word, candidate in candidate_lookup.items():
        occurrences = observed.get(word, [])
        arrays = [item[1] for item in occurrences]
        clans = [item[2] for item in occurrences]
        candidate_rmsd, n_pairs = cross_clan_pair_median(arrays, clans, rmsd, pair_rng, args.max_pairs)
        length_values = []
        sequence_values = []
        for _repeat in range(args.repeats):
            if np.isfinite(candidate_rmsd):
                sampled = [all_pool[int(rng.integers(len(all_pool)))] for _ in arrays]
                value = cross_clan_pair_median(
                    [item[1] for item in sampled], [item[2] for item in sampled], rmsd, pair_rng, args.max_control_pairs
                )[0]
                if np.isfinite(value):
                    length_values.append(value)
                sequence_sample = []
                valid = True
                for sequence, _coordinates, _clan in occurrences:
                    cache_key = (sequence, word)
                    if cache_key not in non_target_pool_cache:
                        non_target_pool_cache[cache_key] = [item for item in pools_by_sequence[sequence] if item[0] != word]
                    choices = non_target_pool_cache[cache_key]
                    if not choices:
                        valid = False
                        break
                    sequence_sample.append(choices[int(rng.integers(len(choices)))])
                if valid:
                    value = cross_clan_pair_median(
                        [item[1] for item in sequence_sample], [item[2] for item in sequence_sample],
                        rmsd, pair_rng, args.max_control_pairs,
                    )[0]
                    if np.isfinite(value):
                        sequence_values.append(value)
        rows.append({
            "kmer": word_string(word),
            "n_occurrences_base_inclusive": len(arrays),
            "base_inclusive_candidate_cross_clan_median_rmsd": candidate_rmsd,
            "base_inclusive_candidate_cross_clan_n_pairs": n_pairs,
            "base_inclusive_length_random_cross_clan_median_rmsd": float(np.median(length_values)) if length_values else float("nan"),
            "base_inclusive_length_random_n_replicates": len(length_values),
            "base_inclusive_sequence_matched_cross_clan_median_rmsd": float(np.median(sequence_values)) if sequence_values else float("nan"),
            "base_inclusive_sequence_matched_n_replicates": len(sequence_values),
            "five_anchor_candidate_cross_clan_median_rmsd": frequency[word_string(word)]["candidate_cross_clan_median_rmsd"],
            "five_anchor_frequency_matched_cross_clan_median_rmsd": frequency[word_string(word)]["cross_clan_matched_control_median_rmsd"],
            "random_seed": SEED,
            "n_repeats": args.repeats,
            "max_pairs_per_comparison": args.max_pairs,
            "max_pairs_per_control_replicate": args.max_control_pairs,
        })
        if len(rows) % 100 == 0:
            print(f"computed base-inclusive RMSD for {len(rows)}/{len(candidate_lookup)} words", flush=True)
    write_tsv(args.output, rows)
    write_tsv(args.skip_log, [{"reason": reason, "count": count, "detail": "."} for reason, count in sorted(skips.items())])
    complete = [row for row in rows if np.isfinite(float(row["base_inclusive_candidate_cross_clan_median_rmsd"]))]

    def median_of(column: str) -> float:
        values = [float(row[column]) for row in complete if np.isfinite(float(row[column]))]
        return float(np.median(values))

    metrics = [
        ("status", "complete"), ("random_seed", SEED), ("n_repeats", args.repeats),
        ("max_pairs_per_comparison", args.max_pairs),
        ("max_pairs_per_control_replicate", args.max_control_pairs),
        ("n_mmcif_files_found", len(list(args.cif_dir.glob("*.cif")))),
        ("n_candidates", len(rows)), ("n_candidates_with_base_inclusive", len(complete)),
        ("n_candidates_with_sequence_matched_control", sum(np.isfinite(float(row["base_inclusive_sequence_matched_cross_clan_median_rmsd"])) for row in complete)),
        ("base_inclusive_candidate_median_rmsd", median_of("base_inclusive_candidate_cross_clan_median_rmsd")),
        ("base_inclusive_length_random_median_rmsd", median_of("base_inclusive_length_random_cross_clan_median_rmsd")),
        ("base_inclusive_sequence_matched_median_rmsd", median_of("base_inclusive_sequence_matched_cross_clan_median_rmsd")),
        ("five_anchor_candidate_cross_clan_median_rmsd", frequency_summary["candidate_cross_clan_median_rmsd"]),
        ("five_anchor_frequency_matched_cross_clan_median_rmsd", frequency_summary["cross_clan_matched_control_median_rmsd"]),
        ("pair_scope", "different Rfam clans on candidate and control sides; unclanned families are singleton clans; unlabelled chains excluded"),
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
