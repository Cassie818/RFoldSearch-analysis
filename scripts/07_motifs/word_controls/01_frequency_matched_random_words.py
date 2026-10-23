"""Frequency-matched controls with known same-clan relationships excluded."""
from __future__ import annotations

import argparse
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import (
    CANDIDATES_PATH, DB_PATH, OUT_DIR, SEED, all_pair_median, compact_coords,
    fragment, load_db_pickle, load_labels, load_reference_rmsd_module,
    occurrence_bin, parse_word, quantiles, read_tsv, seeded_rng,
    stratified_median_permutation_test, wilcoxon_paired, word_occurrences,
    word_string, write_metrics, write_tsv,
)


def word_seed(word: tuple[int, ...]) -> int:
    return SEED + sum((index + 1) * (token + 1) * 1009 for index, token in enumerate(word))


def homology_metadata(occurrences, entry_families: list[str], entry_clans: list[str]) -> dict:
    labelled = [
        (entry_families[entry_idx], entry_clans[entry_idx])
        for entry_idx, _ in occurrences if entry_families[entry_idx]
    ]
    pairs = list(itertools.combinations(labelled, 2))
    cross_family = [(first, second) for first, second in pairs if first[0] != second[0]]
    cross_clan = sum(first[1] != second[1] for first, second in cross_family)
    return {
        "n_chains": len({entry_idx for entry_idx, _ in occurrences}),
        "n_labelled_families": len({family for family, _clan in labelled}),
        "n_labelled_clans": len({clan for _family, clan in labelled}),
        "same_family_pair_fraction": (len(pairs) - len(cross_family)) / len(pairs) if pairs else float("nan"),
        "available_cross_family_pairs": len(cross_family),
        "same_clan_fraction_of_cross_family_pairs": (
            (len(cross_family) - cross_clan) / len(cross_family) if cross_family else float("nan")
        ),
        "available_cross_clan_pairs": cross_clan,
    }


def between_group_median(
    word, occurrences, entry_groups, coordinates, rmsd_module, max_pairs: int,
) -> tuple[float, int]:
    """Median RMSD over occurrence pairs whose labelled groups differ."""
    labelled = [
        (entry_idx, start, entry_groups[entry_idx])
        for entry_idx, start in occurrences if entry_groups[entry_idx]
    ]
    pairs = [
        (left, right)
        for left in range(len(labelled)) for right in range(left + 1, len(labelled))
        if labelled[left][2] != labelled[right][2]
    ]
    if not pairs:
        return float("nan"), 0
    rng = np.random.default_rng(word_seed(word))
    if len(pairs) > max_pairs:
        pairs = [pairs[int(index)] for index in rng.choice(len(pairs), max_pairs, replace=False)]
    fragments = [
        coordinates[entry_idx][start : start + len(word)].reshape(-1, 3)
        for entry_idx, start, _group in labelled
    ]
    values = [rmsd_module.kabsch_rmsd(fragments[left], fragments[right]) for left, right in pairs]
    return float(np.median(values)), len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--candidates", type=Path, default=CANDIDATES_PATH)
    parser.add_argument("--output", type=Path, default=OUT_DIR / "frequency_matched_random_words.tsv")
    parser.add_argument("--catalog", type=Path, default=OUT_DIR / "frequency_control_word_catalog.tsv")
    parser.add_argument("--summary", type=Path, default=OUT_DIR / "frequency_matched_random_words_summary.tsv")
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--max-pairs", type=int, default=400)
    parser.add_argument("--min-control-pairs", type=int, default=1,
                        help="minimum cross-clan pairs for a control word to enter the primary pool")
    parser.add_argument("--sensitivity-min-control-pairs", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=10_000,
                        help="within-frequency-bin unique-word label permutations")
    args = parser.parse_args()

    rmsd = load_reference_rmsd_module()
    entries = load_db_pickle(args.db)["entries"]
    coordinates = [compact_coords(entry) for entry in entries]
    labels = load_labels()
    entry_labels = [
        labels.get(str(entry["id"]), labels.get(Path(entry["id"]).name, ("", "")))
        for entry in entries
    ]
    entry_families = [family for family, _clan in entry_labels]
    # Families without an Rfam clan are treated as singleton clans.
    entry_clans = [(clan or f"family:{family}") if family else "" for family, clan in entry_labels]
    candidates = read_tsv(args.candidates)
    candidate_words = {parse_word(row["kmer"]) for row in candidates}
    occurrences = word_occurrences(entries)
    eligible = {word: occs for word, occs in occurrences.items() if len(occs) >= 8}
    background = {word: occs for word, occs in eligible.items() if word not in candidate_words}

    all_pair_cache = {}
    cross_cache = {}
    clan_cache = {}

    def all_pair(word):
        if word not in all_pair_cache:
            fragments = [fragment(entries[entry_idx], start) for entry_idx, start in eligible[word]]
            all_pair_cache[word] = all_pair_median(fragments, rmsd)
        return all_pair_cache[word]

    def cross_pair(word):
        if word not in cross_cache:
            cross_cache[word] = between_group_median(
                word, eligible[word], entry_families, coordinates, rmsd, args.max_pairs
            )
        return cross_cache[word]

    def clan_pair(word):
        if word not in clan_cache:
            clan_cache[word] = between_group_median(
                word, eligible[word], entry_clans, coordinates, rmsd, args.max_pairs
            )
        return clan_cache[word]

    pools = defaultdict(list)
    sensitivity_pools = defaultdict(list)
    family_pools = defaultdict(list)
    catalog_rows = []
    for word, word_occurrences_ in sorted(background.items()):
        frequency_bin = occurrence_bin(len(word_occurrences_))
        clan_median, clan_n_pairs = clan_pair(word)
        cross_median, cross_n_pairs = cross_pair(word)
        all_median, all_n_pairs = all_pair(word)
        metadata = homology_metadata(word_occurrences_, entry_families, entry_clans)
        if clan_n_pairs >= args.min_control_pairs:
            pools[frequency_bin].append(word)
        if metadata["available_cross_clan_pairs"] >= args.sensitivity_min_control_pairs:
            sensitivity_pools[frequency_bin].append(word)
        if cross_n_pairs:
            family_pools[frequency_bin].append(word)
        catalog_rows.append({
            "kmer": word_string(word), "n_occ": len(word_occurrences_),
            "frequency_bin": frequency_bin, **metadata,
            "all_pair_median_rmsd": all_median, "all_pair_n_pairs": all_n_pairs,
            "cross_family_median_rmsd": cross_median,
            "cross_family_n_pairs_sampled": cross_n_pairs,
            "cross_family_eligible": int(cross_n_pairs > 0),
            "cross_clan_median_rmsd": clan_median,
            "cross_clan_n_pairs_sampled": clan_n_pairs,
            "cross_clan_eligible": int(clan_n_pairs >= args.min_control_pairs),
            "random_seed": word_seed(word),
        })
    write_tsv(args.catalog, catalog_rows)

    rng = seeded_rng()
    sensitivity_rng = seeded_rng(2)
    family_rng = seeded_rng(3)
    rows = []
    for candidate in candidates:
        word = parse_word(candidate["kmer"])
        n_occ = len(occurrences[word])
        frequency_bin = occurrence_bin(n_occ)
        candidate_all_median, candidate_all_pairs = all_pair(word)
        for name, pool_set in (("primary", pools), ("sensitivity", sensitivity_pools), ("cross-family", family_pools)):
            if not pool_set[frequency_bin]:
                raise RuntimeError(f"no {name} controls for frequency bin {frequency_bin}")
        controls = [rng.choice(pools[frequency_bin]) for _ in range(args.repeats)]
        control_clan_values = [clan_pair(control)[0] for control in controls]
        control_all_values = [all_pair(control)[0] for control in controls]
        clan_p10, clan_median, clan_p90 = quantiles(control_clan_values)
        all_p10, all_median, all_p90 = quantiles(control_all_values)
        sensitivity_controls = [sensitivity_rng.choice(sensitivity_pools[frequency_bin]) for _ in range(args.repeats)]
        sensitivity_median = quantiles([clan_pair(control)[0] for control in sensitivity_controls])[1]
        family_controls = [family_rng.choice(family_pools[frequency_bin]) for _ in range(args.repeats)]
        family_control_values = [cross_pair(control)[0] for control in family_controls]
        family_median = quantiles(family_control_values)[1]

        shuffled_word = None
        permutations = sorted(set(itertools.permutations(word)))
        rng.shuffle(permutations)
        for permutation in permutations:
            if permutation != word and permutation in eligible:
                shuffled_word = permutation
                break
        shuffled_median = all_pair(shuffled_word)[0] if shuffled_word is not None else float("nan")
        shuffled_n_occ = len(eligible[shuffled_word]) if shuffled_word is not None else 0
        # Recompute candidates with the same current database, family lookup,
        # pair cap and deterministic per-word sampler used for controls.  The
        # source candidate table is used to define the candidate set, not to
        # mix a separately sampled RMSD estimate into this comparison.
        candidate_clan, candidate_clan_pairs = clan_pair(word)
        candidate_cross, candidate_cross_pairs = cross_pair(word)
        rows.append({
            "kmer": candidate["kmer"], "n_occ": n_occ,
            "frequency_bin": frequency_bin,
            "candidate_cross_clan_median_rmsd": candidate_clan,
            "candidate_cross_clan_n_pairs": candidate_clan_pairs,
            "cross_clan_matched_control_median_rmsd": clan_median,
            "cross_clan_matched_control_p10_rmsd": clan_p10,
            "cross_clan_matched_control_p90_rmsd": clan_p90,
            "candidate_below_cross_clan_control": int(candidate_clan < clan_median),
            "sensitivity_cross_clan_control_median_rmsd": sensitivity_median,
            "candidate_cross_family_median_rmsd": candidate_cross,
            "candidate_cross_family_n_pairs": candidate_cross_pairs,
            "cross_family_matched_control_median_rmsd": family_median,
            "candidate_all_pair_median_rmsd": candidate_all_median,
            "candidate_all_pair_n_pairs": candidate_all_pairs,
            "matched_control_all_pair_median_rmsd": all_median,
            "matched_control_all_pair_p10_rmsd": all_p10,
            "matched_control_all_pair_p90_rmsd": all_p90,
            "shuffled_word": word_string(shuffled_word) if shuffled_word else "",
            "shuffled_n_occ": shuffled_n_occ,
            "shuffled_all_pair_median_rmsd": shuffled_median,
            "candidate_below_shuffled_control": int(candidate_all_median < shuffled_median) if np.isfinite(shuffled_median) else "",
            "matched_control_words": ";".join(word_string(control) for control in controls),
            "matched_control_cross_clan_rmsds": ";".join(f"{value:.9g}" for value in control_clan_values),
            "matched_control_all_pair_rmsds": ";".join(f"{value:.9g}" for value in control_all_values),
            "random_seed": SEED, "n_repeats": args.repeats,
        })
    write_tsv(args.output, rows)

    candidate_clan_values = [float(row["candidate_cross_clan_median_rmsd"]) for row in rows]
    control_clan_values = [float(row["cross_clan_matched_control_median_rmsd"]) for row in rows]
    sensitivity_values = [float(row["sensitivity_cross_clan_control_median_rmsd"]) for row in rows]
    candidate_cross_values = [float(row["candidate_cross_family_median_rmsd"]) for row in rows]
    control_cross_values = [float(row["cross_family_matched_control_median_rmsd"]) for row in rows]
    candidate_all_values = [float(row["candidate_all_pair_median_rmsd"]) for row in rows]
    control_all_values = [float(row["matched_control_all_pair_median_rmsd"]) for row in rows]
    shuffled_rows = [row for row in rows if np.isfinite(float(row["shuffled_all_pair_median_rmsd"]))]
    shuffled_candidates = [float(row["candidate_all_pair_median_rmsd"]) for row in shuffled_rows]
    shuffled_values = [float(row["shuffled_all_pair_median_rmsd"]) for row in shuffled_rows]
    kc10, kcmed, kc90 = quantiles(candidate_clan_values)
    kx10, kxmed, kx90 = quantiles(control_clan_values)
    ccmed = quantiles(candidate_cross_values)[1]
    xcmed = quantiles(control_cross_values)[1]
    ca10, camed, ca90 = quantiles(candidate_all_values)
    xa10, xamed, xa90 = quantiles(control_all_values)
    s10, smed, s90 = quantiles(shuffled_values)
    p_shuffled, method_shuffled, n_shuffled = wilcoxon_paired(shuffled_candidates, shuffled_values)
    def finite_median(column: str) -> float:
        values = [float(row[column]) for row in catalog_rows if np.isfinite(float(row[column]))]
        return float(np.median(values))

    def pool_counts(pool_set) -> str:
        return ";".join(f"{name}:{len(pool_set[name])}" for name in ("8-15", "16-31", ">=32"))

    primary_pool = [row for row in catalog_rows if int(row["cross_clan_eligible"])]
    sensitivity_pool = [
        row for row in catalog_rows
        if int(row["available_cross_clan_pairs"]) >= args.sensitivity_min_control_pairs
    ]
    family_pool = [row for row in catalog_rows if int(row["cross_family_eligible"])]

    def stratified_test(candidate_column: str, control_rows_: list[dict], control_column: str, seed_offset: int):
        strata = {}
        for frequency_bin in ("8-15", "16-31", ">=32"):
            candidate_values_ = [
                float(row[candidate_column]) for row in rows if row["frequency_bin"] == frequency_bin
            ]
            control_values_ = [
                float(row[control_column]) for row in control_rows_
                if row["frequency_bin"] == frequency_bin
            ]
            strata[frequency_bin] = (candidate_values_, control_values_)
        return stratified_median_permutation_test(strata, args.permutations, SEED + seed_offset)

    primary_test = stratified_test(
        "candidate_cross_clan_median_rmsd", primary_pool, "cross_clan_median_rmsd", 10
    )
    sensitivity_test = stratified_test(
        "candidate_cross_clan_median_rmsd", sensitivity_pool, "cross_clan_median_rmsd", 11
    )
    family_test = stratified_test(
        "candidate_cross_family_median_rmsd", family_pool, "cross_family_median_rmsd", 12
    )
    all_pair_test = stratified_test(
        "candidate_all_pair_median_rmsd", primary_pool, "all_pair_median_rmsd", 13
    )
    write_metrics(args.summary, [
        ("random_seed", SEED), ("n_repeats", args.repeats),
        ("max_pairs_per_word", args.max_pairs),
        ("min_cross_clan_pairs_per_control_word", args.min_control_pairs),
        ("frequency_bins", "8-15;16-31;>=32"), ("n_candidates", len(rows)),
        ("n_background_words_occ_ge_8", len(background)),
        ("n_background_words_cross_family_eligible", sum(int(row["cross_family_eligible"]) for row in catalog_rows)),
        ("n_background_words_cross_clan_eligible", len(primary_pool)),
        ("background_fraction_without_cross_family_pairs", float(np.mean([not int(row["cross_family_eligible"]) for row in catalog_rows]))),
        ("background_fraction_without_cross_clan_pairs", float(np.mean([not int(row["available_cross_clan_pairs"]) for row in catalog_rows]))),
        ("background_same_family_pair_fraction_median", finite_median("same_family_pair_fraction")),
        ("background_same_clan_fraction_of_cross_family_pairs_median", finite_median("same_clan_fraction_of_cross_family_pairs")),
        ("primary_control_pool_by_bin", pool_counts(pools)),
        ("primary_control_cross_clan_pairs_median", float(np.median([int(row["cross_clan_n_pairs_sampled"]) for row in primary_pool]))),
        ("candidate_cross_clan_pairs_median", float(np.median([int(row["candidate_cross_clan_n_pairs"]) for row in rows]))),
        ("candidate_cross_clan_median_rmsd", kcmed),
        ("candidate_cross_clan_p10_rmsd", kc10), ("candidate_cross_clan_p90_rmsd", kc90),
        ("cross_clan_matched_control_median_rmsd", kxmed),
        ("cross_clan_matched_control_p10_rmsd", kx10),
        ("cross_clan_matched_control_p90_rmsd", kx90),
        ("fraction_candidates_below_cross_clan_control", float(np.mean(np.asarray(candidate_clan_values) < np.asarray(control_clan_values)))),
        ("permutation_candidate_vs_cross_clan_control_p", primary_test["two_sided_p"]),
        ("permutation_candidate_lower_one_sided_p", primary_test["one_sided_p_candidates_lower"]),
        ("permutation_candidate_vs_cross_clan_control_statistic", primary_test["statistic_control_minus_candidate_rmsd"]),
        ("permutation_candidate_vs_cross_clan_control_method", primary_test["method"]),
        ("permutation_candidate_vs_cross_clan_control_n_permutations", primary_test["n_permutations"]),
        ("permutation_candidate_words", primary_test["n_candidate_words"]),
        ("permutation_control_words", primary_test["n_control_words"]),
        ("permutation_frequency_strata", primary_test["n_strata"]),
        ("sensitivity_min_cross_clan_pairs_per_control_word", args.sensitivity_min_control_pairs),
        ("sensitivity_control_pool_by_bin", pool_counts(sensitivity_pools)),
        ("sensitivity_cross_clan_control_median_rmsd", float(np.median(sensitivity_values))),
        ("sensitivity_fraction_candidates_below_control", float(np.mean(np.asarray(candidate_clan_values) < np.asarray(sensitivity_values)))),
        ("sensitivity_permutation_p", sensitivity_test["two_sided_p"]),
        ("sensitivity_permutation_statistic", sensitivity_test["statistic_control_minus_candidate_rmsd"]),
        ("sensitivity_permutation_method", sensitivity_test["method"]),
        ("sensitivity_permutation_control_words", sensitivity_test["n_control_words"]),
        ("diagnostic_candidate_cross_family_median_rmsd", ccmed),
        ("diagnostic_cross_family_matched_control_median_rmsd", xcmed),
        ("diagnostic_fraction_candidates_below_cross_family_control", float(np.mean(np.asarray(candidate_cross_values) < np.asarray(control_cross_values)))),
        ("diagnostic_permutation_candidate_vs_cross_family_control_p", family_test["two_sided_p"]),
        ("diagnostic_permutation_candidate_vs_cross_family_control_statistic", family_test["statistic_control_minus_candidate_rmsd"]),
        ("diagnostic_permutation_cross_family_control_words", family_test["n_control_words"]),
        ("candidate_all_pair_median_rmsd", camed),
        ("candidate_all_pair_p10_rmsd", ca10), ("candidate_all_pair_p90_rmsd", ca90),
        ("matched_control_all_pair_median_rmsd", xamed),
        ("matched_control_all_pair_p10_rmsd", xa10),
        ("matched_control_all_pair_p90_rmsd", xa90),
        ("permutation_candidate_vs_all_pair_control_p", all_pair_test["two_sided_p"]),
        ("permutation_candidate_vs_all_pair_control_statistic", all_pair_test["statistic_control_minus_candidate_rmsd"]),
        ("n_candidates_with_existing_shuffled_word", len(shuffled_rows)),
        ("shuffled_control_median_rmsd", smed), ("shuffled_control_p10_rmsd", s10),
        ("shuffled_control_p90_rmsd", s90),
        ("wilcoxon_candidate_vs_shuffled_p", p_shuffled),
        ("wilcoxon_candidate_vs_shuffled_method", method_shuffled),
        ("wilcoxon_candidate_vs_shuffled_n", n_shuffled),
        ("rmsd_definition_source", "scripts/07_motifs/cross_family_kmer_mining.py:kabsch_rmsd"),
        ("primary_pair_scope", "different Rfam clans on both candidate and control sides; unclanned families are singleton clans"),
    ])
    print(f"wrote {args.output}")
    print(f"wrote {args.catalog}")
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
