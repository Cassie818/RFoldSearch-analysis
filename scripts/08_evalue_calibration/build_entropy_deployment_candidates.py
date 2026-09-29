#!/usr/bin/env python3
"""Apply the all-query entropy model and enumerate deployment candidates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def design(length, entropy, model):
    centre = model["centre"]
    x = (math.log(length) - centre["log_length_mean"]) / centre["log_length_scale"]
    h = (entropy - centre["entropy_mean"]) / centre["entropy_scale"]
    return np.asarray([1.0, x, x * x, h, h * h, x * h], dtype=float)


def evalues(scores, mu, beta, n_clusters):
    tail = -np.expm1(-np.exp(-(np.asarray(scores, dtype=float) - mu) / beta))
    return n_clusters * tail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--raw-shards", type=Path, nargs="+", required=True)
    parser.add_argument("--tm-cache", type=Path)
    parser.add_argument("--e-max", type=float, default=100.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fitted = json.loads(args.model.read_text())
    deployment = fitted["deployment_model"]
    coefficients = np.asarray(deployment["coefficients_mu_logbeta"], dtype=float)
    n_clusters = int(fitted["n_clusters"])
    cache = {}
    if args.tm_cache:
        previous = json.loads(args.tm_cache.read_text())
        for record in previous["records"]:
            for candidate in record["candidates"]:
                if candidate.get("tm") is not None:
                    cache[(record["query"], int(candidate["target_index"]))] = float(candidate["tm"])

    records = []
    for json_path in args.raw_shards:
        metadata = json.loads(json_path.read_text())
        arrays = np.load(json_path.with_suffix(".npz"))
        offsets = arrays["offsets"]
        for index, record in enumerate(metadata["records"]):
            start, end = int(offsets[index]), int(offsets[index + 1])
            scores = np.asarray(arrays["scores"][start:end], dtype=float)
            clusters = arrays["cluster_ids"][start:end]
            targets = arrays["target_indices"][start:end]
            response = design(record["length"], record["token_entropy"], deployment) @ coefficients
            mu, beta = float(response[0]), float(math.exp(response[1]))
            raw_evalues = evalues(scores, mu, beta, n_clusters)
            selected = np.flatnonzero(raw_evalues <= args.e_max)
            selected = selected[np.argsort(-scores[selected], kind="stable")]
            candidates = []
            for position in selected:
                target_index = int(targets[position])
                candidate = {
                    "cluster": int(clusters[position]),
                    "target_index": target_index,
                    "score": float(scores[position]),
                    "raw_evalue": float(raw_evalues[position]),
                }
                cached_tm = cache.get((record["query"], target_index))
                if cached_tm is not None:
                    candidate["tm"] = cached_tm
                    candidate["background"] = cached_tm < 0.45
                candidates.append(candidate)
            records.append({
                **record,
                "deployment_mu": mu,
                "deployment_beta": beta,
                "n_candidates_e_le_max": len(candidates),
                "candidates": candidates,
            })
    records.sort(key=lambda record: record["global_position"])
    output = {
        "scheme": "predicted-entropy-conditional-deployment-candidates",
        "score": "raw Structure-BLOSUM Smith-Waterman score",
        "n_queries": len(records),
        "n_families": len({record["family"] for record in records}),
        "n_clusters": n_clusters,
        "e_max": args.e_max,
        "deployment_model": deployment,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "n_queries": len(records),
        "total_candidates": sum(len(record["candidates"]) for record in records),
        "queries_with_candidates": sum(bool(record["candidates"]) for record in records),
        "cached_tm": sum(
            sum("tm" in candidate for candidate in record["candidates"])
            for record in records
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
