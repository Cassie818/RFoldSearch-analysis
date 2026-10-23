"""Recurrent-word RMSD with five-anchor and base-inclusive atoms on one shared universe."""
from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from residue_keys import parse_key_string
from common import (  # noqa: E402
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
    occurrence_bin,
    parse_word,
    parse_atom_site,
    compact_coords,
    read_tsv,
    stratified_median_permutation_test,
    stem_parts,
    word_string,
    write_metrics,
    write_tsv,
)


def word_seed(word: tuple[int, ...], offset: int = 0) -> int:
    """Stable per-word seed, independent of dictionary iteration order."""
    return SEED + offset + sum((index + 1) * (token + 1) * 1009 for index, token in enumerate(word))


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


def paired_atom_medians(
    base_arrays: list[np.ndarray], five_arrays: list[np.ndarray], clans: list[str],
    rmsd, seed: int, max_pairs: int,
) -> tuple[float, float, int]:
    """Evaluate both atom representations on the same sampled occurrence pairs."""
    base_value, n_pairs = cross_clan_pair_median(
        base_arrays, clans, rmsd, random.Random(seed), max_pairs
    )
    five_value, five_n_pairs = cross_clan_pair_median(
        five_arrays, clans, rmsd, random.Random(seed), max_pairs
    )
    if n_pairs != five_n_pairs:
        raise RuntimeError("base-inclusive and five-anchor pair samples diverged")
    return base_value, five_value, n_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--candidates", type=Path, default=CANDIDATES_PATH)
    parser.add_argument("--residue-map", type=Path, default=MOTIF_DIR / "chain_residue_key_map.json")
    parser.add_argument("--cif-dir", type=Path, default=Path.home() / "Desktop" / "NR80_local")
    parser.add_argument("--output", type=Path, default=OUT_DIR / "base_inclusive_rmsd.tsv")
    parser.add_argument("--skip-log", type=Path, default=OUT_DIR / "base_inclusive_rmsd_skips.tsv")
    parser.add_argument("--summary", type=Path, default=OUT_DIR / "base_inclusive_rmsd_summary.tsv")
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--max-pairs", type=int, default=400)
    parser.add_argument("--max-control-pairs", type=int, default=25)
    parser.add_argument("--permutations", type=int, default=10_000,
                        help="within-frequency-bin unique-word label permutations (as in 01)")
    args = parser.parse_args()

    if not args.cif_dir.is_dir():
        raise SystemExit(
            f"blocked: missing {args.cif_dir}; existing outputs were left unchanged"
        )

    rmsd = load_reference_rmsd_module()
    entries = load_db_pickle(args.db)["entries"]
    candidates = read_tsv(args.candidates)
    candidate_lookup = {parse_word(row["kmer"]): row for row in candidates}
    residue_map = json.load(args.residue_map.open())
    labels = load_labels()
    rng = np.random.default_rng(SEED)
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
        # Residues are matched by (author number, insertion code), so that insertion-coded
        # residues such as 47 and 47A are not merged.
        residues = parse_atom_site(cif_path, chain, model, full_keys=True)
        mask = np.asarray(entry["mask"], dtype=bool)
        residue_keys = residue_map[stem]
        if len(mask) != len(residue_keys):
            skips["mask_residue_map_length_mismatch"] += max(0, len(entry["tokens"]) - 3)
            continue
        valid_numbers = [
            (key[4], key[5])
            for key, valid in zip(map(parse_key_string, residue_keys), mask) if valid
        ]
        tokens = np.asarray(entry["tokens"], dtype=int)
        anchors = compact_coords(entry)
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
            five = np.asarray(anchors[start : start + 4], dtype=np.float32).reshape(-1, 3)
            pools_by_sequence[sequence].append((word, coordinate_array, clan, five))
            if word in candidate_lookup:
                observed[word].append((sequence, coordinate_array, clan, five))
        if entry_number % 100 == 0:
            print(f"parsed {entry_number}/{len(entries)} mmCIF chains", flush=True)

    all_pool = [item for pool in pools_by_sequence.values() for item in pool]
    non_target_pool_cache: dict[tuple[str, tuple[int, ...]], list[tuple[tuple[int, ...], np.ndarray, str]]] = {}
    rows = []
    for word, candidate in candidate_lookup.items():
        occurrences = observed.get(word, [])
        arrays = [item[1] for item in occurrences]
        five_arrays = [item[3] for item in occurrences]
        clans = [item[2] for item in occurrences]
        candidate_rmsd, five_candidate_rmsd, n_pairs = paired_atom_medians(
            arrays, five_arrays, clans, rmsd, word_seed(word), args.max_pairs
        )
        length_values = []
        sequence_values = []
        five_length_values = []
        five_sequence_values = []
        for repeat in range(args.repeats):
            if np.isfinite(candidate_rmsd):
                sampled = [all_pool[int(rng.integers(len(all_pool)))] for _ in arrays]
                value, five_value, _ = paired_atom_medians(
                    [item[1] for item in sampled], [item[3] for item in sampled],
                    [item[2] for item in sampled], rmsd,
                    word_seed(word, 100_000 + repeat), args.max_control_pairs,
                )
                if np.isfinite(value): length_values.append(value)
                if np.isfinite(five_value): five_length_values.append(five_value)
                sequence_sample = []
                valid = True
                for sequence, _coordinates, _clan, _five in occurrences:
                    cache_key = (sequence, word)
                    if cache_key not in non_target_pool_cache:
                        non_target_pool_cache[cache_key] = [item for item in pools_by_sequence[sequence] if item[0] != word]
                    choices = non_target_pool_cache[cache_key]
                    if not choices:
                        valid = False
                        break
                    sequence_sample.append(choices[int(rng.integers(len(choices)))])
                if valid:
                    value, five_value, _ = paired_atom_medians(
                        [item[1] for item in sequence_sample], [item[3] for item in sequence_sample],
                        [item[2] for item in sequence_sample], rmsd,
                        word_seed(word, 200_000 + repeat), args.max_control_pairs,
                    )
                    if np.isfinite(value): sequence_values.append(value)
                    if np.isfinite(five_value): five_sequence_values.append(five_value)
        rows.append({
            "kmer": word_string(word),
            "n_occurrences_base_inclusive": len(arrays),
            "base_inclusive_candidate_cross_clan_median_rmsd": candidate_rmsd,
            "base_inclusive_candidate_cross_clan_n_pairs": n_pairs,
            "base_inclusive_length_random_cross_clan_median_rmsd": float(np.median(length_values)) if length_values else float("nan"),
            "base_inclusive_length_random_n_replicates": len(length_values),
            "base_inclusive_sequence_matched_cross_clan_median_rmsd": float(np.median(sequence_values)) if sequence_values else float("nan"),
            "base_inclusive_sequence_matched_n_replicates": len(sequence_values),
            "five_anchor_length_random_cross_clan_median_rmsd": float(np.median(five_length_values)) if five_length_values else float("nan"),
            "five_anchor_sequence_matched_cross_clan_median_rmsd": float(np.median(five_sequence_values)) if five_sequence_values else float("nan"),
            "five_anchor_candidate_cross_clan_median_rmsd": five_candidate_rmsd,
            "five_anchor_candidate_cross_clan_n_pairs": n_pairs,
            "random_seed": SEED,
            "n_repeats": args.repeats,
            "max_pairs_per_comparison": args.max_pairs,
            "max_pairs_per_control_replicate": args.max_control_pairs,
        })
        if len(rows) % 100 == 0:
            print(f"computed base-inclusive RMSD for {len(rows)}/{len(candidate_lookup)} words", flush=True)
    # ---- frequency-matched words in the shared complete-coordinate universe ------
    # Candidate, frequency-matched, random-window and sequence-matched values now all
    # use occurrences that passed the same mmCIF mapping and complete-atom checks.
    occurrences_by_word: dict[tuple[int, ...], list[tuple[np.ndarray, str, np.ndarray]]] = defaultdict(list)
    for pool in pools_by_sequence.values():
        for item_word, coordinates, item_clan, five in pool:
            occurrences_by_word[item_word].append((coordinates, item_clan, five))
    candidate_words = set(candidate_lookup)
    control_values: dict[tuple[int, ...], tuple[float, float]] = {}
    control_pools: dict[str, list[tuple[int, ...]]] = defaultdict(list)
    for control_word, items in sorted(occurrences_by_word.items()):
        if control_word in candidate_words or len(items) < 8:
            continue
        base_value, five_value, n_pairs = paired_atom_medians(
            [item[0] for item in items], [item[2] for item in items], [item[1] for item in items],
            rmsd, word_seed(control_word, 300_000), args.max_pairs,
        )
        if n_pairs:
            control_values[control_word] = (base_value, five_value)
            control_pools[occurrence_bin(len(items))].append(control_word)

    matching_rng = random.Random(SEED + 77)
    for row in rows:
        word = parse_word(row["kmer"])
        # occurrence_bin() places candidates with fewer than eight occurrences in the
        # universe in the 8-15 bin, the lowest bin with a control pool.
        frequency_bin = occurrence_bin(len(observed.get(word, [])))
        pool = control_pools[frequency_bin]
        if not pool:
            raise RuntimeError(f"no complete-coordinate controls for frequency bin {frequency_bin}")
        matched = [matching_rng.choice(pool) for _ in range(args.repeats)]
        base_values = [control_values[value][0] for value in matched]
        five_values = [control_values[value][1] for value in matched]
        row["base_inclusive_frequency_matched_cross_clan_median_rmsd"] = (
            float(np.median(base_values)) if base_values else float("nan"))
        row["five_anchor_frequency_matched_cross_clan_median_rmsd"] = (
            float(np.median(five_values)) if five_values else float("nan"))
        row["eligible_frequency_bin"] = frequency_bin
        row["eligible_frequency_matched_words"] = ";".join(map(word_string, matched))
        row["base_inclusive_frequency_matched_n_words"] = len(base_values)
    strata = {}
    five_strata = {}
    for frequency_bin in ("8-15", "16-31", ">=32"):
        strata[frequency_bin] = (
            [float(row["base_inclusive_candidate_cross_clan_median_rmsd"]) for row in rows
             if row["eligible_frequency_bin"] == frequency_bin],
            [control_values[word][0] for word in control_pools[frequency_bin]],
        )
        five_strata[frequency_bin] = (
            [float(row["five_anchor_candidate_cross_clan_median_rmsd"]) for row in rows
             if row["eligible_frequency_bin"] == frequency_bin],
            [control_values[word][1] for word in control_pools[frequency_bin]],
        )
    frequency_test = stratified_median_permutation_test(strata, args.permutations, SEED + 20)
    five_frequency_test = stratified_median_permutation_test(five_strata, args.permutations, SEED + 21)

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
        ("five_anchor_length_random_median_rmsd", median_of("five_anchor_length_random_cross_clan_median_rmsd")),
        ("five_anchor_sequence_matched_median_rmsd", median_of("five_anchor_sequence_matched_cross_clan_median_rmsd")),
        ("five_anchor_candidate_median_rmsd", median_of("five_anchor_candidate_cross_clan_median_rmsd")),
        ("five_anchor_frequency_matched_median_rmsd", median_of("five_anchor_frequency_matched_cross_clan_median_rmsd")),
        ("base_inclusive_frequency_matched_median_rmsd", median_of("base_inclusive_frequency_matched_cross_clan_median_rmsd")),
        ("base_inclusive_frequency_control_pool_words", len(control_values)),
        ("five_anchor_frequency_control_pool_words", len(control_values)),
        ("permutation_base_inclusive_candidate_vs_frequency_control_p", frequency_test["two_sided_p"]),
        ("permutation_base_inclusive_candidate_vs_frequency_control_n_permutations", frequency_test["n_permutations"]),
        ("permutation_five_anchor_candidate_vs_frequency_control_p", five_frequency_test["two_sided_p"]),
        ("permutation_five_anchor_candidate_vs_frequency_control_n_permutations", five_frequency_test["n_permutations"]),
        ("eligible_universe", "Rfam-labelled occurrences with unique full residue-key mapping and complete base-inclusive atoms"),
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
