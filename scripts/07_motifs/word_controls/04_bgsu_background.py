"""Word-size-matched BGSU null distributions and overlap sensitivity checks.

The inferential unit is a recurrent word. Each null replicate draws the same
number of random four-nucleotide windows as each word contributes on
BGSU-covered chains, then takes the median overlap fraction across words.
Pooled occurrence fractions are reported separately as diagnostics.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import (
    CANDIDATES_PATH, DB_PATH, MOTIF_DIR, OUT_DIR, SEED, load_db_pickle,
    parse_word, quantiles, read_tsv, write_metrics, write_tsv,
)


THRESHOLDS = (1, 3, 4)


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


def empirical_p_ge(null_values: list[float], observed: float) -> float:
    """One-sided Monte Carlo P value with the standard plus-one correction."""
    return (1 + sum(value >= observed for value in null_values)) / (len(null_values) + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--candidates", type=Path, default=CANDIDATES_PATH)
    parser.add_argument("--hl", type=Path, default=MOTIF_DIR / "bgsu_hl_current_2026-08-03.csv")
    parser.add_argument("--il", type=Path, default=MOTIF_DIR / "bgsu_il_current_2026-08-03.csv")
    parser.add_argument("--residue-map", type=Path, default=MOTIF_DIR / "chain_residue_map.json")
    parser.add_argument("--pairing-words", type=Path, default=OUT_DIR / "pairing_pattern_words.tsv")
    parser.add_argument("--output", type=Path, default=OUT_DIR / "bgsu_random_window_background.tsv")
    parser.add_argument("--candidate-output", type=Path, default=OUT_DIR / "bgsu_candidate_overlap_sensitivity.tsv")
    parser.add_argument("--summary", type=Path, default=OUT_DIR / "bgsu_random_window_background_summary.tsv")
    parser.add_argument("--repeats", type=int, default=1000)
    args = parser.parse_args()

    bgsu = load_bgsu_module()
    motifs = {**bgsu.parse_bgsu(args.il, "IL"), **bgsu.parse_bgsu(args.hl, "HL")}
    chain_to_motif_residues: dict[tuple[str, str], list[set[int]]] = defaultdict(list)
    for occurrences in motifs.values():
        for occurrence in occurrences:
            for pdb_chain in {(pdb, chain) for pdb, chain, _residue in occurrence}:
                chain_to_motif_residues[pdb_chain].append({
                    residue for pdb, chain, residue in occurrence if (pdb, chain) == pdb_chain
                })

    candidate_rows = read_tsv(args.candidates)
    candidates = {parse_word(row["kmer"]) for row in candidate_rows}
    residue_map = json.load(args.residue_map.open())
    entries = load_db_pickle(args.db)["entries"]
    random_window_scores: list[int] = []
    candidate_scores: dict[tuple[int, ...], list[int]] = defaultdict(list)
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
        tokens = np.asarray(entry["tokens"], dtype=np.int32)
        if len(valid_resnums) != len(tokens):
            n_mapping_skipped += 1
            continue
        n_covered_chains += 1
        motif_sets = chain_to_motif_residues[parts]
        for start in range(len(valid_resnums) - 3):
            residues = set(valid_resnums[start : start + 4])
            score = max((len(residues & motif_residues) for motif_residues in motif_sets), default=0)
            random_window_scores.append(score)
            word = tuple(map(int, tokens[start : start + 4]))
            if word in candidates:
                candidate_scores[word].append(score)

    candidate_output_rows = []
    for candidate in candidate_rows:
        word = parse_word(candidate["kmer"])
        scores = candidate_scores.get(word, [])
        row: dict[str, object] = {"kmer": candidate["kmer"], "n_occ_on_bgsu_chains": len(scores)}
        for threshold in THRESHOLDS:
            hits = sum(score >= threshold for score in scores)
            row[f"hits_ge_{threshold}"] = hits
            row[f"fraction_ge_{threshold}"] = hits / len(scores) if scores else float("nan")
        candidate_output_rows.append(row)
    write_tsv(args.candidate_output, candidate_output_rows)

    finite_rows = [row for row in candidate_output_rows if int(row["n_occ_on_bgsu_chains"]) > 0]
    sizes = np.asarray([int(row["n_occ_on_bgsu_chains"]) for row in finite_rows], dtype=int)
    pairing = {row["kmer"]: row["word_category"] for row in read_tsv(args.pairing_words)}
    group_indices = {
        "a": np.asarray([i for i, row in enumerate(finite_rows) if pairing.get(row["kmer"]) == "a"], dtype=int),
        "nonhelix": np.asarray([i for i, row in enumerate(finite_rows) if pairing.get(row["kmer"]) in {"b", "c", "d"}], dtype=int),
    }

    rng = np.random.default_rng(SEED)
    window_scores = np.asarray(random_window_scores, dtype=np.int8)
    background_rows = []
    null_word_medians: dict[int, list[float]] = defaultdict(list)
    null_pooled: dict[int, list[float]] = defaultdict(list)
    null_group_medians: dict[tuple[int, str], list[float]] = defaultdict(list)
    for repeat in range(args.repeats):
        sampled_scores = [rng.choice(window_scores, size=n, replace=True) for n in sizes]
        for threshold in THRESHOLDS:
            fractions = np.asarray([np.mean(scores >= threshold) for scores in sampled_scores])
            pooled_hits = sum(int(np.sum(scores >= threshold)) for scores in sampled_scores)
            word_median = float(np.median(fractions))
            pooled_fraction = pooled_hits / int(sizes.sum())
            null_word_medians[threshold].append(word_median)
            null_pooled[threshold].append(pooled_fraction)
            row = {
                "repeat": repeat + 1, "overlap_threshold_residues": threshold,
                "n_words": len(finite_rows), "n_sampled_windows": int(sizes.sum()),
                "null_word_median_fraction": word_median,
                "null_pooled_overlap_fraction": pooled_fraction, "random_seed": SEED,
            }
            for group, indices in group_indices.items():
                value = float(np.median(fractions[indices]))
                null_group_medians[(threshold, group)].append(value)
                row[f"null_{group}_word_median_fraction"] = value
            background_rows.append(row)
    write_tsv(args.output, background_rows)

    metrics: list[tuple[str, object]] = [
        ("random_seed", SEED), ("n_repeats", args.repeats),
        ("n_bgsu_covered_chains", n_covered_chains),
        ("n_chains_skipped_mapping_mismatch", n_mapping_skipped),
        ("n_eligible_random_windows", len(window_scores)),
        ("n_candidate_words_with_bgsu_denominator", len(finite_rows)),
        ("n_candidate_occurrences_on_bgsu_covered_chains", int(sizes.sum())),
        ("overlap_thresholds_residues", ";".join(map(str, THRESHOLDS))),
        ("null_sampling_unit", "for each word, n random windows where n equals that word's BGSU-covered occurrence count"),
        ("sampling_scope", "uniform 4-nt windows with replacement from BGSU-covered chains with valid residue mapping"),
    ]
    for threshold in THRESHOLDS:
        fractions = np.asarray([float(row[f"fraction_ge_{threshold}"]) for row in finite_rows])
        hits = sum(int(row[f"hits_ge_{threshold}"]) for row in finite_rows)
        observed_median = float(np.median(fractions))
        null10, null50, null90 = quantiles(null_word_medians[threshold])
        pooled10, pooled50, pooled90 = quantiles(null_pooled[threshold])
        prefix = f"threshold_{threshold}"
        metrics.extend([
            (f"{prefix}_eligible_window_fraction", float(np.mean(window_scores >= threshold))),
            (f"{prefix}_candidate_word_median_fraction", observed_median),
            (f"{prefix}_candidate_pooled_fraction", hits / int(sizes.sum())),
            (f"{prefix}_null_word_median_p10_fraction", null10),
            (f"{prefix}_null_word_median_fraction", null50),
            (f"{prefix}_null_word_median_p90_fraction", null90),
            (f"{prefix}_word_median_empirical_p_ge", empirical_p_ge(null_word_medians[threshold], observed_median)),
            (f"{prefix}_null_pooled_p10_fraction", pooled10),
            (f"{prefix}_null_pooled_median_fraction", pooled50),
            (f"{prefix}_null_pooled_p90_fraction", pooled90),
        ])
        for group, indices in group_indices.items():
            observed = float(np.median(fractions[indices]))
            group10, group50, group90 = quantiles(null_group_medians[(threshold, group)])
            metrics.extend([
                (f"{prefix}_group_{group}_n_words", len(indices)),
                (f"{prefix}_group_{group}_observed_word_median_fraction", observed),
                (f"{prefix}_group_{group}_null_word_median_p10_fraction", group10),
                (f"{prefix}_group_{group}_null_word_median_fraction", group50),
                (f"{prefix}_group_{group}_null_word_median_p90_fraction", group90),
                (f"{prefix}_group_{group}_empirical_p_ge", empirical_p_ge(null_group_medians[(threshold, group)], observed)),
            ])
    write_metrics(args.summary, metrics)
    print(f"wrote {args.candidate_output}")
    print(f"wrote {args.output}")
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
