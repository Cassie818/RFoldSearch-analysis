"""Assign recurrent words their modal Biotite pairing-pattern category."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common import (
    CANDIDATES_PATH, DB_PATH, OUT_DIR, SEED, load_db_pickle, load_labels,
    load_sources, parse_word, read_tsv, word_string, write_metrics, write_tsv,
)

CATEGORY_NAMES = {
    "a": "standard_helix_PPPP_cis_antiparallel",
    "b": "all_paired_nonstandard",
    "c": "partly_unpaired_1_to_3_U",
    "d": "all_unpaired_UUUU",
}


def classify(pattern: str, orientations: tuple[int, ...], partners: tuple[int, ...]) -> str:
    if pattern == "PPPP":
        antiparallel = partners == tuple(range(partners[0], partners[0] - 4, -1))
        if all(value == 1 for value in orientations) and antiparallel:
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
    parser.add_argument("--occurrences", type=Path, default=OUT_DIR / "pairing_pattern_occurrences.tsv")
    parser.add_argument("--words", type=Path, default=OUT_DIR / "pairing_pattern_words.tsv")
    parser.add_argument("--summary", type=Path, default=OUT_DIR / "pairing_pattern_stratification_summary.tsv")
    args = parser.parse_args()

    candidate_rows = read_tsv(args.candidates)
    candidate_words = {parse_word(row["kmer"]): row for row in candidate_rows}
    sources, labels = load_sources(), load_labels()
    entries = load_db_pickle(args.db)["entries"]

    occurrence_rows = []
    signatures = defaultdict(list)
    n_entries_missing_pair_annotations = 0
    n_occurrences_missing_pair_annotations = 0
    n_occurrences_with_pair_annotations = 0
    total_candidate_occurrences = sum(int(row["n_occ"]) for row in candidate_rows)

    for entry in entries:
        entry_id, portable_id = str(entry["id"]), Path(entry["id"]).name
        if not labels.get(entry_id, labels.get(portable_id, ("", "")))[0]:
            continue
        tokens = np.asarray(entry["tokens"], dtype=np.int32)
        starts = [
            start for start in range(len(tokens) - 3)
            if tuple(map(int, tokens[start:start + 4])) in candidate_words
        ]
        if not starts:
            continue
        source = sources.get(entry_id, sources.get(portable_id))
        if source is None or "pair_idx" not in source or "pair_orient" not in source:
            n_entries_missing_pair_annotations += 1
            n_occurrences_missing_pair_annotations += len(starts)
            continue
        mask = np.asarray(source.get("mask", []), dtype=bool)
        raw_indices = np.flatnonzero(mask) if mask.size and int(mask.sum()) == len(tokens) else np.arange(len(tokens))
        pair_idx = np.asarray(source["pair_idx"], dtype=int).reshape(-1)
        pair_orient = np.asarray(source["pair_orient"], dtype=int).reshape(-1)
        for start in starts:
            word = tuple(map(int, tokens[start:start + 4]))
            raw = tuple(map(int, raw_indices[start:start + 4]))
            partners = tuple(int(pair_idx[index]) for index in raw)
            orientations = tuple(int(pair_orient[index]) for index in raw)
            pattern = "".join("P" if partner >= 0 else "U" for partner in partners)
            category = classify(pattern, orientations, partners)
            paired_partners = [partner for partner in partners if partner >= 0]
            origin = paired_partners[0] if paired_partners else 0
            normalized = tuple(partner - origin if partner >= 0 else -999 for partner in partners)
            signatures[word].append((pattern, orientations, normalized, category))
            n_occurrences_with_pair_annotations += 1
            occurrence_rows.append({
                "kmer": word_string(word), "entry_id": portable_id,
                "token_start_1based": start + 1,
                "raw_indices_0based": ",".join(map(str, raw)),
                "pairing_pattern": pattern,
                "partner_indices_0based": ",".join(map(str, partners)),
                "pair_orient_codes": ",".join(map(str, orientations)),
                "occurrence_category": category,
                "annotation_status": "annotated", "random_seed": SEED,
            })

    missing_words = sorted(word_string(word) for word in candidate_words if not signatures[word])
    if missing_words:
        raise RuntimeError(
            f"{len(missing_words)} candidate words lack Biotite pairing annotations; "
            f"first examples: {', '.join(missing_words[:10])}"
        )

    word_rows = []
    for word, candidate in candidate_words.items():
        category, representative, pattern_count, category_count, geometry_count = modal_category(signatures[word])
        pattern, orientations, normalized_partners, _ = representative
        word_rows.append({
            "kmer": word_string(word), "word_category": category,
            "word_category_name": CATEGORY_NAMES[category],
            "majority_pairing_pattern": pattern,
            "majority_pair_orient_codes": ",".join(map(str, orientations)),
            "majority_normalized_partner_pattern": ",".join(map(str, normalized_partners)),
            "majority_pattern_count": pattern_count,
            "n_occurrences_with_pair_annotations": len(signatures[word]),
            "n_occurrences_total": candidate["n_occ"],
            "majority_pattern_fraction": pattern_count / len(signatures[word]),
            "category_count_within_majority_pattern": category_count,
            "category_fraction_within_majority_pattern": category_count / pattern_count,
            "representative_geometry_count": geometry_count,
        })

    write_tsv(args.occurrences, occurrence_rows)
    write_tsv(args.words, word_rows)
    metrics = [
        ("n_candidates", len(word_rows)),
        ("n_candidate_occurrences_total_from_candidate_table", total_candidate_occurrences),
        ("n_candidate_occurrences_enumerated_in_current_db", len(occurrence_rows)),
        ("n_occurrences_with_pair_annotations", n_occurrences_with_pair_annotations),
        ("n_occurrences_missing_pair_annotations", n_occurrences_missing_pair_annotations),
        ("n_entries_missing_pair_annotations", n_entries_missing_pair_annotations),
        ("pairing_annotation_source", "Splits/pdb_out train+val+test_with_edges Biotite pair_idx/pair_orient arrays"),
    ]
    for category in "abcd":
        metrics.extend([
            (f"category_{category}_name", CATEGORY_NAMES[category]),
            (f"category_{category}_n_words", sum(row["word_category"] == category for row in word_rows)),
        ])
    write_metrics(args.summary, metrics)
    print(f"wrote {args.occurrences}")
    print(f"wrote {args.words}")
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
