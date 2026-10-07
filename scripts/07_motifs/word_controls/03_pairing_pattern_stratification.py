"""Pairing-pattern stratification with category-matched cross-clan controls."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common import (
    CANDIDATES_PATH, DB_PATH, MOTIF_DIR, OUT_DIR, SEED, load_db_pickle,
    load_labels, load_sources, parse_word, quantiles, read_tsv, seeded_rng,
    stratified_median_permutation_test, word_string, write_metrics, write_tsv,
)


CATEGORY_NAMES = {
    "a": "standard_helix_PPPP_cis_antiparallel",
    "b": "all_paired_nonstandard",
    "c": "partly_unpaired_1_to_3_U",
    "d": "all_unpaired_UUUU",
}


def classify(pattern: str, orientations: tuple[int, ...], partners: tuple[int, ...]) -> str:
    if pattern == "PPPP":
        if all(value == 1 for value in orientations) and partners == tuple(range(partners[0], partners[0] - 4, -1)):
            return "a"
        return "b"
    if pattern == "UUUU":
        return "d"
    return "c"


def modal_category(signatures: list[tuple]) -> tuple[str, tuple, int, int, int]:
    pattern_counts = Counter(signature[0] for signature in signatures)
    pattern, pattern_count = pattern_counts.most_common(1)[0]
    modal_pattern_rows = [signature for signature in signatures if signature[0] == pattern]
    category_counts = Counter(signature[3] for signature in modal_pattern_rows)
    category, category_count = category_counts.most_common(1)[0]
    modal_geometry_rows = [signature for signature in modal_pattern_rows if signature[3] == category]
    representative, geometry_count = Counter(modal_geometry_rows).most_common(1)[0]
    return category, representative, pattern_count, category_count, geometry_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--candidates", type=Path, default=CANDIDATES_PATH)
    parser.add_argument("--frequency-controls", type=Path, default=OUT_DIR / "frequency_matched_random_words.tsv")
    parser.add_argument("--control-catalog", type=Path, default=OUT_DIR / "frequency_control_word_catalog.tsv")
    parser.add_argument("--bgsu-overlap", type=Path, default=MOTIF_DIR / "kmer_bgsu_overlap_production_new_k4_aa.tsv")
    parser.add_argument("--occurrences", type=Path, default=OUT_DIR / "pairing_pattern_occurrences.tsv")
    parser.add_argument("--words", type=Path, default=OUT_DIR / "pairing_pattern_words.tsv")
    parser.add_argument("--control-words", type=Path, default=OUT_DIR / "pairing_pattern_control_words.tsv")
    parser.add_argument("--summary", type=Path, default=OUT_DIR / "pairing_pattern_stratification_summary.tsv")
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--permutations", type=int, default=10_000,
                        help="unique-word label permutations within matching strata")
    args = parser.parse_args()

    candidate_rows = read_tsv(args.candidates)
    candidate_words = {parse_word(row["kmer"]): row for row in candidate_rows}
    frequency = {row["kmer"]: row for row in read_tsv(args.frequency_controls)}
    control_catalog = {
        parse_word(row["kmer"]): row for row in read_tsv(args.control_catalog)
        if row["cross_clan_eligible"] == "1"
    }
    target_words = set(candidate_words) | set(control_catalog)
    bgsu = {row["kmer"]: row for row in read_tsv(args.bgsu_overlap)}
    sources, labels = load_sources(), load_labels()
    entries = load_db_pickle(args.db)["entries"]

    occurrence_rows = []
    signatures = defaultdict(list)
    n_entries_missing_pair_annotations = 0
    n_occurrences_missing_pair_annotations = 0
    n_occurrences_with_pair_annotations = 0
    total_candidate_occurrences_table = sum(int(row["n_occ"]) for row in candidate_rows)
    for entry in entries:
        entry_id, portable_id = str(entry["id"]), Path(entry["id"]).name
        if not labels.get(entry_id, labels.get(portable_id, ("", "")))[0]:
            continue
        tokens = np.asarray(entry["tokens"], dtype=np.int32)
        starts = [
            start for start in range(len(tokens) - 3)
            if tuple(map(int, tokens[start : start + 4])) in target_words
        ]
        if not starts:
            continue
        source = sources.get(entry_id, sources.get(portable_id))
        if source is None or "pair_idx" not in source or "pair_orient" not in source:
            candidate_starts = [s for s in starts if tuple(map(int, tokens[s : s + 4])) in candidate_words]
            n_entries_missing_pair_annotations += 1
            n_occurrences_missing_pair_annotations += len(candidate_starts)
            continue
        mask = np.asarray(source.get("mask", []), dtype=bool)
        raw_indices = np.flatnonzero(mask) if mask.size and int(mask.sum()) == len(tokens) else np.arange(len(tokens))
        pair_idx = np.asarray(source["pair_idx"], dtype=int).reshape(-1)
        pair_orient = np.asarray(source["pair_orient"], dtype=int).reshape(-1)
        for start in starts:
            word = tuple(map(int, tokens[start : start + 4]))
            raw = tuple(map(int, raw_indices[start : start + 4]))
            partners = tuple(int(pair_idx[index]) for index in raw)
            orientations = tuple(int(pair_orient[index]) for index in raw)
            pattern = "".join("P" if partner >= 0 else "U" for partner in partners)
            category = classify(pattern, orientations, partners)
            paired_partners = [partner for partner in partners if partner >= 0]
            origin = paired_partners[0] if paired_partners else 0
            normalized = tuple(partner - origin if partner >= 0 else -999 for partner in partners)
            signatures[word].append((pattern, orientations, normalized, category))
            if word in candidate_words:
                n_occurrences_with_pair_annotations += 1
                occurrence_rows.append({
                    "kmer": word_string(word), "entry_id": portable_id,
                    "token_start_1based": start + 1,
                    "raw_indices_0based": ",".join(map(str, raw)),
                    "pairing_pattern": pattern,
                    "partner_indices_0based": ",".join(map(str, partners)),
                    "pair_orient_codes": ",".join(map(str, orientations)),
                    "occurrence_category": category, "annotation_status": "annotated",
                    "random_seed": SEED,
                })
    write_tsv(args.occurrences, occurrence_rows)

    category_by_word = {}
    details_by_word = {}
    for word in target_words:
        if not signatures[word]:
            continue
        category, representative, pattern_count, category_count, geometry_count = modal_category(signatures[word])
        category_by_word[word] = category
        details_by_word[word] = (representative, pattern_count, category_count, geometry_count)

    control_rows = []
    pools = defaultdict(list)
    for word, control in control_catalog.items():
        if word not in category_by_word:
            continue
        category = category_by_word[word]
        pools[(control["frequency_bin"], category)].append(word)
        control_rows.append({**control, "word_category": category, "word_category_name": CATEGORY_NAMES[category]})
    write_tsv(args.control_words, control_rows)

    rng = seeded_rng(1)
    word_rows = []
    for word, candidate in candidate_words.items():
        category = category_by_word[word]
        representative, pattern_count, category_count, geometry_count = details_by_word[word]
        pattern, orientations, normalized_partners, _ = representative
        pool = pools[(frequency[word_string(word)]["frequency_bin"], category)]
        if not pool:
            raise RuntimeError(f"no category-matched controls for {word_string(word)} ({category})")
        controls = [rng.choice(pool) for _ in range(args.repeats)]
        values = [float(control_catalog[control]["cross_clan_median_rmsd"]) for control in controls]
        p10, control_median, p90 = quantiles(values)
        bgsu_fraction = (
            float(bgsu[word_string(word)]["frac_in_bgsu"])
            if word_string(word) in bgsu and bgsu[word_string(word)]["frac_in_bgsu"] not in {"", "nan"}
            else float("nan")
        )
        candidate_cross = float(frequency[word_string(word)]["candidate_cross_clan_median_rmsd"])
        word_rows.append({
            "kmer": word_string(word), "word_category": category,
            "word_category_name": CATEGORY_NAMES[category], "majority_pairing_pattern": pattern,
            "majority_pair_orient_codes": ",".join(map(str, orientations)),
            "majority_normalized_partner_pattern": ",".join(map(str, normalized_partners)),
            "majority_pattern_count": pattern_count,
            "n_occurrences_with_pair_annotations": len(signatures[word]),
            "n_occurrences_total": candidate["n_occ"],
            "majority_pattern_fraction": pattern_count / len(signatures[word]),
            "category_count_within_majority_pattern": category_count,
            "category_fraction_within_majority_pattern": category_count / pattern_count,
            "representative_geometry_count": geometry_count,
            "candidate_cross_clan_median_rmsd": candidate_cross,
            "category_matched_control_cross_clan_median_rmsd": control_median,
            "category_matched_control_cross_clan_p10_rmsd": p10,
            "category_matched_control_cross_clan_p90_rmsd": p90,
            "candidate_below_category_matched_control": int(candidate_cross < control_median),
            "category_matched_control_words": ";".join(word_string(control) for control in controls),
            "bgsu_overlap_fraction": bgsu_fraction, "random_seed": SEED,
        })
    write_tsv(args.words, word_rows)

    metrics = [
        ("random_seed", SEED), ("n_repeats", args.repeats),
        ("n_permutations", args.permutations),
        ("n_candidates", len(word_rows)),
        ("n_cross_clan_control_words_with_pairing_category", len(control_rows)),
        ("n_candidate_occurrences_total_from_candidate_table", total_candidate_occurrences_table),
        ("n_candidate_occurrences_enumerated_in_current_db", len(occurrence_rows)),
        ("n_occurrences_with_pair_annotations", n_occurrences_with_pair_annotations),
        ("n_occurrences_missing_pair_annotations", n_occurrences_missing_pair_annotations),
        ("n_entries_missing_pair_annotations", n_entries_missing_pair_annotations),
        ("pairing_annotation_source", "Splits/pdb_out train+val+test_with_edges Biotite pair_idx/pair_orient arrays"),
    ]

    def category_permutation(categories: set[str], seed_offset: int):
        strata = {}
        for frequency_bin in ("8-15", "16-31", ">=32"):
            for category_ in sorted(categories):
                candidate_values_ = [
                    float(row["candidate_cross_clan_median_rmsd"])
                    for row in word_rows
                    if row["word_category"] == category_
                    and frequency[row["kmer"]]["frequency_bin"] == frequency_bin
                ]
                control_values_ = [
                    float(row["cross_clan_median_rmsd"])
                    for row in control_rows
                    if row["word_category"] == category_
                    and row["frequency_bin"] == frequency_bin
                ]
                if candidate_values_ and control_values_:
                    strata[(frequency_bin, category_)] = (candidate_values_, control_values_)
        return stratified_median_permutation_test(
            strata, args.permutations, SEED + seed_offset
        )

    category_tests = {
        category: category_permutation({category}, 20 + ord(category))
        for category in "abcd"
    }
    ordered = sorted("abcd", key=lambda category: float(category_tests[category]["two_sided_p"]))
    holm_adjusted = {}
    running_max = 0.0
    for rank, category in enumerate(ordered):
        adjusted = min(1.0, (len(ordered) - rank) * float(category_tests[category]["two_sided_p"]))
        running_max = max(running_max, adjusted)
        holm_adjusted[category] = running_max

    for category in "abcd":
        subset = [row for row in word_rows if row["word_category"] == category]
        candidate_values = [float(row["candidate_cross_clan_median_rmsd"]) for row in subset]
        control_values = [float(row["category_matched_control_cross_clan_median_rmsd"]) for row in subset]
        permutation = category_tests[category]
        metrics.extend([
            (f"category_{category}_name", CATEGORY_NAMES[category]),
            (f"category_{category}_n_words", len(subset)),
            (f"category_{category}_candidate_cross_clan_median_rmsd", float(np.median(candidate_values))),
            (f"category_{category}_control_cross_clan_median_rmsd", float(np.median(control_values))),
            (f"category_{category}_fraction_candidate_below_control", float(np.mean(np.asarray(candidate_values) < np.asarray(control_values)))),
            (f"category_{category}_permutation_p", permutation["two_sided_p"]),
            (f"category_{category}_permutation_p_holm", holm_adjusted[category]),
            (f"category_{category}_permutation_statistic", permutation["statistic_control_minus_candidate_rmsd"]),
            (f"category_{category}_permutation_method", permutation["method"]),
            (f"category_{category}_permutation_control_words", permutation["n_control_words"]),
            (f"category_{category}_control_pool_n", sum(
                len(pools[(frequency_bin, category)])
                for frequency_bin in ("8-15", "16-31", ">=32")
            )),
        ])

    nonhelix = [row for row in word_rows if row["word_category"] != "a"]
    nonhelix_candidate = [float(row["candidate_cross_clan_median_rmsd"]) for row in nonhelix]
    nonhelix_control = [float(row["category_matched_control_cross_clan_median_rmsd"]) for row in nonhelix]
    nonhelix_permutation = category_permutation({"b", "c", "d"}, 30)
    metrics.extend([
        ("nonhelix_n_words", len(nonhelix)),
        ("nonhelix_candidate_cross_clan_median_rmsd", float(np.median(nonhelix_candidate))),
        ("nonhelix_control_cross_clan_median_rmsd", float(np.median(nonhelix_control))),
        ("nonhelix_fraction_candidate_below_control", float(np.mean(np.asarray(nonhelix_candidate) < np.asarray(nonhelix_control)))),
        ("nonhelix_permutation_p", nonhelix_permutation["two_sided_p"]),
        ("nonhelix_permutation_statistic", nonhelix_permutation["statistic_control_minus_candidate_rmsd"]),
        ("nonhelix_permutation_method", nonhelix_permutation["method"]),
        ("nonhelix_permutation_control_words", nonhelix_permutation["n_control_words"]),
        ("primary_control_matching", "occurrence-frequency bin plus modal pairing category; different-clan pairs only"),
    ])
    write_metrics(args.summary, metrics)
    print(f"wrote {args.occurrences}")
    print(f"wrote {args.control_words}")
    print(f"wrote {args.words}")
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
