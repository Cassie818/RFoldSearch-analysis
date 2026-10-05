"""Stratify recurrent 3Dr words by pipeline-derived pairing patterns."""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common import (
    CANDIDATES_PATH,
    DB_PATH,
    MOTIF_DIR,
    OUT_DIR,
    SEED,
    load_db_pickle,
    load_labels,
    load_sources,
    parse_word,
    quantiles,
    read_tsv,
    wilcoxon_paired,
    word_string,
    write_metrics,
    write_tsv,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--candidates", type=Path, default=CANDIDATES_PATH)
    parser.add_argument("--frequency-controls", type=Path, default=OUT_DIR / "frequency_matched_random_words.tsv")
    parser.add_argument("--bgsu-overlap", type=Path, default=MOTIF_DIR / "kmer_bgsu_overlap_production_new_k4_aa.tsv")
    parser.add_argument("--occurrences", type=Path, default=OUT_DIR / "pairing_pattern_occurrences.tsv")
    parser.add_argument("--words", type=Path, default=OUT_DIR / "pairing_pattern_words.tsv")
    parser.add_argument("--summary", type=Path, default=OUT_DIR / "pairing_pattern_stratification_summary.tsv")
    args = parser.parse_args()

    candidate_rows = read_tsv(args.candidates)
    candidate_words = {parse_word(row["kmer"]): row for row in candidate_rows}
    frequency = {row["kmer"]: row for row in read_tsv(args.frequency_controls)}
    bgsu = {row["kmer"]: row for row in read_tsv(args.bgsu_overlap)}
    sources = load_sources()
    labels = load_labels()
    entries = load_db_pickle(args.db)["entries"]

    occurrence_rows = []
    signatures: dict[tuple[int, ...], list[tuple]] = defaultdict(list)
    n_entries_missing_pair_annotations = 0
    n_occurrences_missing_pair_annotations = 0
    n_occurrences_with_pair_annotations = 0
    total_candidate_occurrences_table = sum(int(row["n_occ"]) for row in candidate_rows)
    for entry in entries:
        entry_id = str(entry["id"])
        portable_id = Path(entry_id).name
        if not labels.get(entry_id, labels.get(portable_id, ("", "")))[0]:
            continue
        tokens = np.asarray(entry["tokens"], dtype=np.int32)
        source = sources.get(entry_id, sources.get(portable_id))
        candidate_starts = [
            start for start in range(len(tokens) - 3)
            if tuple(map(int, tokens[start : start + 4])) in candidate_words
        ]
        if not candidate_starts:
            continue
        if source is None or "pair_idx" not in source or "pair_orient" not in source:
            n_entries_missing_pair_annotations += 1
            n_occurrences_missing_pair_annotations += len(candidate_starts)
            for start in candidate_starts:
                word = tuple(map(int, tokens[start : start + 4]))
                occurrence_rows.append({
                    "kmer": word_string(word),
                    "entry_id": portable_id,
                    "token_start_1based": start + 1,
                    "raw_indices_0based": "",
                    "pairing_pattern": "",
                    "partner_indices_0based": "",
                    "pair_orient_codes": "",
                    "occurrence_category": "",
                    "annotation_status": "missing_pair_arrays",
                    "random_seed": SEED,
                })
            continue
        mask = np.asarray(source.get("mask", []), dtype=bool)
        raw_indices = np.flatnonzero(mask) if mask.size and int(mask.sum()) == len(tokens) else np.arange(len(tokens))
        pair_idx = np.asarray(source["pair_idx"], dtype=int).reshape(-1)
        pair_orient = np.asarray(source["pair_orient"], dtype=int).reshape(-1)
        for start in candidate_starts:
            word = tuple(map(int, tokens[start : start + 4]))
            raw = tuple(map(int, raw_indices[start : start + 4]))
            partners = tuple(int(pair_idx[index]) for index in raw)
            orientations = tuple(int(pair_orient[index]) for index in raw)
            pattern = "".join("P" if partner >= 0 else "U" for partner in partners)
            category = classify(pattern, orientations, partners)
            paired_partners = [partner for partner in partners if partner >= 0]
            origin = paired_partners[0] if paired_partners else 0
            normalized_partners = tuple(partner - origin if partner >= 0 else -999 for partner in partners)
            signature = (pattern, orientations, normalized_partners, category)
            signatures[word].append(signature)
            n_occurrences_with_pair_annotations += 1
            occurrence_rows.append({
                "kmer": word_string(word),
                "entry_id": portable_id,
                "token_start_1based": start + 1,
                "raw_indices_0based": ",".join(map(str, raw)),
                "pairing_pattern": pattern,
                "partner_indices_0based": ",".join(map(str, partners)),
                "pair_orient_codes": ",".join(map(str, orientations)),
                "occurrence_category": category,
                "annotation_status": "annotated",
                "random_seed": SEED,
            })
    write_tsv(args.occurrences, occurrence_rows)

    word_rows = []
    for word, candidate in candidate_words.items():
        observed = signatures[word]
        pattern_counts = Counter(signature[0] for signature in observed)
        pattern, pattern_count = pattern_counts.most_common(1)[0]
        modal_pattern_rows = [signature for signature in observed if signature[0] == pattern]
        category_counts = Counter(signature[3] for signature in modal_pattern_rows)
        category, category_count = category_counts.most_common(1)[0]
        modal_geometry_rows = [signature for signature in modal_pattern_rows if signature[3] == category]
        representative, geometry_count = Counter(modal_geometry_rows).most_common(1)[0]
        _pattern, orientations, normalized_partners, _category = representative
        bgsu_fraction = float(bgsu[word_string(word)]["frac_in_bgsu"]) if word_string(word) in bgsu and bgsu[word_string(word)]["frac_in_bgsu"] not in {"", "nan"} else float("nan")
        control = frequency[word_string(word)]
        word_rows.append({
            "kmer": word_string(word),
            "word_category": category,
            "word_category_name": CATEGORY_NAMES[category],
            "majority_pairing_pattern": pattern,
            "majority_pair_orient_codes": ",".join(map(str, orientations)),
            "majority_normalized_partner_pattern": ",".join(map(str, normalized_partners)),
            "majority_pattern_count": pattern_count,
            "n_occurrences_with_pair_annotations": len(observed),
            "n_occurrences_total": candidate["n_occ"],
            "majority_pattern_fraction": pattern_count / len(observed),
            "category_count_within_majority_pattern": category_count,
            "category_fraction_within_majority_pattern": category_count / pattern_count,
            "representative_geometry_count": geometry_count,
            "candidate_cross_family_median_rmsd": candidate["cross_med_rmsd"],
            "candidate_all_pair_median_rmsd": control["candidate_all_pair_median_rmsd"],
            "frequency_matched_control_median_rmsd": control["matched_control_median_rmsd"],
            "candidate_below_matched_control": control["candidate_below_matched_control"],
            "bgsu_overlap_fraction": bgsu_fraction,
            "random_seed": SEED,
        })
    write_tsv(args.words, word_rows)

    metrics: list[tuple[str, object]] = [
        ("random_seed", SEED),
        ("n_candidates", len(word_rows)),
        ("n_candidate_occurrences_total_from_candidate_table", total_candidate_occurrences_table),
        ("n_candidate_occurrences_enumerated_in_current_db", len(occurrence_rows)),
        ("n_occurrences_with_pair_annotations", n_occurrences_with_pair_annotations),
        ("n_occurrences_missing_pair_annotations", n_occurrences_missing_pair_annotations),
        ("n_entries_missing_pair_annotations", n_entries_missing_pair_annotations),
        ("pairing_annotation_source", "Splits/pdb_out train+val+test_with_edges Biotite pair_idx/pair_orient arrays"),
    ]
    for category in "abcd":
        subset = [row for row in word_rows if row["word_category"] == category]
        cross = [float(row["candidate_cross_family_median_rmsd"]) for row in subset]
        controls = [float(row["frequency_matched_control_median_rmsd"]) for row in subset]
        bgsu_values = [float(row["bgsu_overlap_fraction"]) for row in subset if np.isfinite(float(row["bgsu_overlap_fraction"]))]
        metrics.extend([
            (f"category_{category}_name", CATEGORY_NAMES[category]),
            (f"category_{category}_n_words", len(subset)),
            (f"category_{category}_cross_family_median_rmsd", float(np.median(cross))),
            (f"category_{category}_matched_control_median_rmsd", float(np.median(controls))),
            (f"category_{category}_bgsu_overlap_median_fraction", float(np.median(bgsu_values)) if bgsu_values else float("nan")),
            (f"category_{category}_n_words_with_bgsu_denominator", len(bgsu_values)),
        ])

    nonhelix = [row for row in word_rows if row["word_category"] != "a"]
    nonhelix_candidate = [float(row["candidate_all_pair_median_rmsd"]) for row in nonhelix]
    nonhelix_control = [float(row["frequency_matched_control_median_rmsd"]) for row in nonhelix]
    p_value, p_method, p_n = wilcoxon_paired(nonhelix_candidate, nonhelix_control)
    metrics.extend([
        ("nonstandard_after_removing_a_n_words", len(nonhelix)),
        ("nonstandard_after_removing_a_candidate_all_pair_median_rmsd", float(np.median(nonhelix_candidate))),
        ("nonstandard_after_removing_a_matched_control_median_rmsd", float(np.median(nonhelix_control))),
        ("nonstandard_after_removing_a_fraction_candidate_below_control", float(np.mean(np.asarray(nonhelix_candidate) < np.asarray(nonhelix_control)))),
        ("nonstandard_after_removing_a_wilcoxon_p", p_value),
        ("nonstandard_after_removing_a_wilcoxon_method", p_method),
        ("nonstandard_after_removing_a_wilcoxon_n", p_n),
    ])
    write_metrics(args.summary, metrics)
    print(f"wrote {args.occurrences}")
    print(f"wrote {args.words}")
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
