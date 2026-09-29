#!/usr/bin/env python3
"""Fit the final family-weighted empirical map for predicted-database E-values."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.input.read_text())
    records = data["records"]
    by_family = defaultdict(list)
    for record in records:
        by_family[record["family"]].append(record)
    families = sorted(by_family)
    lengths = [int(record["length"]) for record in records]
    entropies = [float(record["token_entropy"]) for record in records]
    training_domain = {
        "length_min": min(lengths),
        "length_max": max(lengths),
        "log_length_min": math.log(min(lengths)),
        "log_length_max": math.log(max(lengths)),
        "entropy_min": min(entropies),
        "entropy_max": max(entropies),
    }

    values, weights = [], []
    n_unresolved = 0
    for family in families:
        query_weight = 1.0 / (len(families) * len(by_family[family]))
        for record in by_family[family]:
            for candidate in record["candidates"]:
                if candidate.get("tm") is None:
                    n_unresolved += 1
                if candidate.get("background"):
                    values.append(float(candidate["raw_evalue"]))
                    weights.append(query_weight)
    if not values:
        raise ValueError("no resolved background candidates")

    order = np.argsort(values, kind="stable")
    knots = np.asarray(values, dtype=float)[order]
    sorted_weights = np.asarray(weights, dtype=float)[order]
    cumulative = np.cumsum(sorted_weights)
    finite_sample_floor = 0.5 * float(sorted_weights.min())

    # Store one cumulative value per distinct right-continuous step.
    final_at_knot = np.r_[knots[1:] != knots[:-1], True]
    knots = knots[final_at_knot]
    cumulative = cumulative[final_at_knot]
    output = {
        "scheme": "predicted-query-conditional-clustermax-empirical",
        "database": "NuFold predicted-structure database",
        "score": data["score"],
        "ranking_score": "raw Structure-BLOSUM Smith-Waterman score",
        "n_clusters": int(data["n_clusters"]),
        "cluster_definition": "CD-HIT-EST at 80% nucleotide-sequence identity",
        "n_calibration_queries": len(records),
        "n_calibration_families": len(families),
        "query_parameter_model": {
            "distribution": "Gumbel",
            "responses": ["mu", "log_beta"],
            "features": [
                "intercept", "standardized_log_length",
                "standardized_log_length_squared", "standardized_token_entropy",
                "standardized_token_entropy_squared",
                "standardized_log_length_by_token_entropy",
            ],
            **data["deployment_model"],
            "training_domain": training_domain,
        },
        "empirical_calibration": {
            "definition": "family-weighted mean number of unrelated clusters per query",
            "weighting": "equal Rfam-family weight; equal query weight within family",
            "related_target_exclusion": {
                "annotation": "query family/clan excluded during background scoring",
                "symmetric_usalign_tm_score_threshold": float(data["tm_threshold"]),
                "tm_score_definition": "mean of query- and target-normalized US-align TM-scores",
            },
            "evaluation": "right-continuous empirical step function",
            "candidate_raw_evalue_ceiling": float(data["e_max"]),
            "finite_sample_floor": finite_sample_floor,
            "raw_evalue_knots": knots.tolist(),
            "calibrated_evalues": cumulative.tolist(),
            "largest_raw_evalue_knot": float(knots[-1]),
            "largest_resolved_calibrated_evalue": float(cumulative[-1]),
            "n_background_observations": len(values),
            "n_unresolved_candidates": n_unresolved,
        },
        "validation": {
            "scheme": "leave-one-Rfam-family-out",
            "crossfit_summary": "entropy_conditional_empirical_lofo_summary_E100.json",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "n_calibration_queries": len(records),
        "n_calibration_families": len(families),
        "n_background_observations": len(values),
        "n_unresolved_candidates": n_unresolved,
        "training_domain": training_domain,
        "finite_sample_floor": finite_sample_floor,
        "largest_raw_evalue_knot": float(knots[-1]),
        "largest_resolved_calibrated_evalue": float(cumulative[-1]),
    }, indent=2))


if __name__ == "__main__":
    main()
