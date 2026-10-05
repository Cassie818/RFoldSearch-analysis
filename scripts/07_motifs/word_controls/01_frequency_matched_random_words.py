"""Frequency-matched and letter-shuffled controls for recurrent 3Dr words."""
from __future__ import annotations

import argparse
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import (
    CANDIDATES_PATH,
    DB_PATH,
    OUT_DIR,
    SEED,
    all_pair_median,
    fragment,
    load_db_pickle,
    load_reference_rmsd_module,
    occurrence_bin,
    parse_word,
    quantiles,
    read_tsv,
    seeded_rng,
    wilcoxon_paired,
    word_occurrences,
    word_string,
    write_metrics,
    write_tsv,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--candidates", type=Path, default=CANDIDATES_PATH)
    parser.add_argument("--output", type=Path, default=OUT_DIR / "frequency_matched_random_words.tsv")
    parser.add_argument("--summary", type=Path, default=OUT_DIR / "frequency_matched_random_words_summary.tsv")
    parser.add_argument("--repeats", type=int, default=25)
    args = parser.parse_args()

    rmsd = load_reference_rmsd_module()
    entries = load_db_pickle(args.db)["entries"]
    candidates = read_tsv(args.candidates)
    candidate_words = {parse_word(row["kmer"]) for row in candidates}
    occurrences = word_occurrences(entries)
    eligible = {word: occs for word, occs in occurrences.items() if len(occs) >= 8}
    background = {word: occs for word, occs in eligible.items() if word not in candidate_words}
    pools = defaultdict(list)
    for word, occs in background.items():
        pools[occurrence_bin(len(occs))].append(word)

    medians: dict[tuple[int, ...], tuple[float, int]] = {}

    def median_for(word: tuple[int, ...]) -> tuple[float, int]:
        if word not in medians:
            frags = [fragment(entries[entry_idx], start) for entry_idx, start in eligible[word]]
            medians[word] = all_pair_median(frags, rmsd)
        return medians[word]

    rng = seeded_rng()
    rows = []
    for candidate in candidates:
        word = parse_word(candidate["kmer"])
        n_occ = len(occurrences[word])
        candidate_median, candidate_pairs = median_for(word)
        pool = pools[occurrence_bin(n_occ)]
        controls = [rng.choice(pool) for _ in range(args.repeats)]
        control_values = [median_for(control)[0] for control in controls]
        p10, control_median, p90 = quantiles(control_values)

        shuffled_word = None
        permutations = sorted(set(itertools.permutations(word)))
        rng.shuffle(permutations)
        for permutation in permutations:
            if permutation != word and permutation in eligible:
                shuffled_word = permutation
                break
        shuffled_median = float("nan")
        shuffled_n_occ = 0
        if shuffled_word is not None:
            shuffled_median = median_for(shuffled_word)[0]
            shuffled_n_occ = len(eligible[shuffled_word])

        row = {
            "kmer": candidate["kmer"],
            "n_occ": n_occ,
            "frequency_bin": occurrence_bin(n_occ),
            "candidate_all_pair_median_rmsd": candidate_median,
            "candidate_all_pair_n_pairs": candidate_pairs,
            "candidate_cross_family_median_rmsd": candidate["cross_med_rmsd"],
            "matched_control_median_rmsd": control_median,
            "matched_control_p10_rmsd": p10,
            "matched_control_p90_rmsd": p90,
            "candidate_below_matched_control": int(candidate_median < control_median),
            "shuffled_word": word_string(shuffled_word) if shuffled_word else "",
            "shuffled_n_occ": shuffled_n_occ,
            "shuffled_all_pair_median_rmsd": shuffled_median,
            "candidate_below_shuffled_control": int(candidate_median < shuffled_median) if np.isfinite(shuffled_median) else "",
            "matched_control_words": ";".join(word_string(control) for control in controls),
            "matched_control_rmsds": ";".join(f"{value:.9g}" for value in control_values),
            "random_seed": SEED,
            "n_repeats": args.repeats,
        }
        rows.append(row)

    write_tsv(args.output, rows)
    candidate_values = [float(row["candidate_all_pair_median_rmsd"]) for row in rows]
    matched_values = [float(row["matched_control_median_rmsd"]) for row in rows]
    shuffled_rows = [row for row in rows if np.isfinite(float(row["shuffled_all_pair_median_rmsd"]))]
    shuffled_candidates = [float(row["candidate_all_pair_median_rmsd"]) for row in shuffled_rows]
    shuffled_values = [float(row["shuffled_all_pair_median_rmsd"]) for row in shuffled_rows]
    c10, cmed, c90 = quantiles(candidate_values)
    m10, mmed, m90 = quantiles(matched_values)
    s10, smed, s90 = quantiles(shuffled_values) if shuffled_values else (float("nan"),) * 3
    p_matched, method_matched, n_matched = wilcoxon_paired(candidate_values, matched_values)
    p_shuffled, method_shuffled, n_shuffled = wilcoxon_paired(shuffled_candidates, shuffled_values)
    write_metrics(args.summary, [
        ("random_seed", SEED),
        ("n_repeats", args.repeats),
        ("frequency_bins", "8-15;16-31;>=32"),
        ("n_candidates", len(rows)),
        ("n_background_words_occ_ge_8", len(background)),
        ("n_background_words_bin_8_15", len(pools["8-15"])),
        ("n_background_words_bin_16_31", len(pools["16-31"])),
        ("n_background_words_bin_ge_32", len(pools[">=32"])),
        ("candidate_all_pair_median_rmsd", cmed),
        ("candidate_all_pair_p10_rmsd", c10),
        ("candidate_all_pair_p90_rmsd", c90),
        ("frequency_matched_control_median_rmsd", mmed),
        ("frequency_matched_control_p10_rmsd", m10),
        ("frequency_matched_control_p90_rmsd", m90),
        ("fraction_candidates_below_matched_control", float(np.mean([int(row["candidate_below_matched_control"]) for row in rows]))),
        ("wilcoxon_candidate_vs_matched_p", p_matched),
        ("wilcoxon_candidate_vs_matched_method", method_matched),
        ("wilcoxon_candidate_vs_matched_n", n_matched),
        ("n_candidates_with_existing_shuffled_word", len(shuffled_rows)),
        ("shuffled_control_median_rmsd", smed),
        ("shuffled_control_p10_rmsd", s10),
        ("shuffled_control_p90_rmsd", s90),
        ("fraction_candidates_below_shuffled_control", float(np.mean([int(row["candidate_below_shuffled_control"]) for row in shuffled_rows])) if shuffled_rows else float("nan")),
        ("wilcoxon_candidate_vs_shuffled_p", p_shuffled),
        ("wilcoxon_candidate_vs_shuffled_method", method_shuffled),
        ("wilcoxon_candidate_vs_shuffled_n", n_shuffled),
        ("rmsd_definition_source", "scripts/07_motifs/cross_family_kmer_mining.py:pairwise_rmsd"),
        ("occurrence_scope", "all 885 database chains"),
    ])
    print(f"wrote {args.output}")
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
