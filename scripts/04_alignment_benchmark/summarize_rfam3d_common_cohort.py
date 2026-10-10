"""Build the manuscript's canonical summary on the verified common pair cohort."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


METHODS = ("rfoldsearch", "riboseek", "usalign", "sequence_alignment")
METRICS = ("f1", "tolerant_f1")


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(rows: list[dict], reps: int, seed: int,
              expected_pairs: int | None = None) -> dict:
    selected = [row for row in rows if row["sequence_baseline_reconstruction_verified"]]
    grouped = defaultdict(list)
    for row in selected:
        grouped[row["family"]].append(row)
    families = sorted(grouped)
    if (expected_pairs is not None and len(selected) != expected_pairs) or len(families) != 24:
        raise ValueError(f"Expected {expected_pairs} pairs from 24 families; got "
                         f"{len(selected)} pairs from {len(families)} families")

    family_means = {}
    for method in METHODS:
        for metric in METRICS:
            key = f"{method}_{metric}"
            family_means[key] = [mean(row[key] for row in grouped[family])
                                 for family in families]
    for comparator in METHODS[1:]:
        for metric in METRICS:
            key = f"rfoldsearch_minus_{comparator}_{metric}"
            family_means[key] = [
                mean(row[f"rfoldsearch_{metric}"] - row[f"{comparator}_{metric}"]
                     for row in grouped[family]) for family in families
            ]

    rng = random.Random(seed)
    draws = [[rng.randrange(len(families)) for _ in families]
             for _ in range(reps)]
    summary = {
        "manuscript_canonical_summary": True,
        "n_evaluable_pairs": len(selected),
        "n_evaluable_families": len(families),
        "n_pairs_below_sequence_identity_0.8": sum(
            row["sequence_identity"] < 0.8 for row in selected),
        "family_pair_counts": dict(sorted(Counter(row["family"] for row in selected).items())),
        "cohort_rule": (
            "Only pairs with reconstructed reference-pair count, US-align family-region "
            "lengths, and sequence identity matching the saved 313-pair benchmark"
        ),
        "bootstrap": {"unit": "Rfam family", "replicates": reps, "seed": seed},
    }
    for key, values in family_means.items():
        estimates = [mean(values[index] for index in draw) for draw in draws]
        summary[key] = {
            "n_families": len(families),
            "family_macro_mean": mean(values),
            "family_block_bootstrap_95ci": [
                percentile(estimates, 0.025), percentile(estimates, 0.975)
            ],
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original-313", type=Path)
    parser.add_argument("--robustness-tsv", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--expected-pairs", type=int)
    args = parser.parse_args()
    rows = json.loads(args.input.read_text())["pairs"]
    result = summarize(rows, args.bootstrap, args.seed, args.expected_pairs)
    args.output.write_text(json.dumps({"summary": result,
                                       "pairs": [row for row in rows if row[
                                           "sequence_baseline_reconstruction_verified"]]},
                                      indent=2) + "\n")
    if args.robustness_tsv is not None:
        if args.original_313 is None:
            parser.error("--robustness-tsv requires --original-313")
        original = json.loads(args.original_313.read_text())["summary"]
        table = []
        for cohort, count, source in (("Original", 313, original),
                                      ("Common", result["n_evaluable_pairs"], result)):
            for method in METHODS:
                if cohort == "Original" and method == "sequence_alignment":
                    continue
                for metric in METRICS:
                    key = f"{method}_{metric}"
                    estimate = source[key]
                    table.append({
                        "cohort": cohort, "n_pairs": count,
                        "n_families": source.get("n_evaluable_families", 24),
                        "method": method, "metric": metric,
                        "family_macro_f1": estimate["family_macro_mean"],
                        "ci_lower": estimate["family_block_bootstrap_95ci"][0],
                        "ci_upper": estimate["family_block_bootstrap_95ci"][1],
                    })
        with args.robustness_tsv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(table)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
