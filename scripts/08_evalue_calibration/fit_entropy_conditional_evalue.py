#!/usr/bin/env python3
"""Cross-fit query-conditional Gumbel parameters from length and token entropy."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import numpy as np


def load_evalue(upstream: Path):
    package = types.ModuleType("rfoldseek")
    package.__path__ = [str(upstream / "rfoldseek")]
    sys.modules["rfoldseek"] = package
    for short in ("smith_waterman", "evalue"):
        name = f"rfoldseek.{short}"
        spec = importlib.util.spec_from_file_location(
            name, upstream / "rfoldseek" / f"{short}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules["rfoldseek.evalue"]


def features(lengths, entropies, centre=None):
    log_length = np.log(np.asarray(lengths, dtype=float))
    entropy = np.asarray(entropies, dtype=float)
    if centre is None:
        centre = {
            "log_length_mean": float(log_length.mean()),
            "log_length_scale": float(log_length.std(ddof=0)),
            "entropy_mean": float(entropy.mean()),
            "entropy_scale": float(entropy.std(ddof=0)),
        }
    x = (log_length - centre["log_length_mean"]) / centre["log_length_scale"]
    h = (entropy - centre["entropy_mean"]) / centre["entropy_scale"]
    design = np.column_stack([np.ones(len(x)), x, x * x, h, h * h, x * h])
    return design, centre


def ridge_fit(design, response, penalty=1e-3):
    gram = design.T @ design
    gram[1:, 1:] += np.eye(design.shape[1] - 1) * penalty
    return np.linalg.solve(gram, design.T @ response)


def evalues(scores, mu, beta, n_clusters):
    tail = -np.expm1(-np.exp(-(np.asarray(scores, dtype=float) - mu) / beta))
    return n_clusters * tail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--e-max", type=float, default=10.0)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ev = load_evalue(args.upstream)
    records = []
    score_rows = []
    cluster_rows = []
    target_rows = []
    n_clusters = None
    for json_path in args.shards:
        metadata = json.loads(json_path.read_text())
        arrays = np.load(json_path.with_suffix(".npz"))
        offsets = arrays["offsets"]
        if n_clusters is None:
            n_clusters = int(metadata["n_clusters"])
        elif n_clusters != int(metadata["n_clusters"]):
            raise ValueError("inconsistent cluster count")
        for index, record in enumerate(metadata["records"]):
            records.append(record)
            start, end = int(offsets[index]), int(offsets[index + 1])
            score_rows.append(np.asarray(arrays["scores"][start:end], dtype=float))
            cluster_rows.append(np.asarray(arrays["cluster_ids"][start:end], dtype=np.int32))
            target_rows.append(np.asarray(arrays["target_indices"][start:end], dtype=np.int32))

    order = np.argsort([record["global_position"] for record in records])
    records = [records[index] for index in order]
    score_rows = [score_rows[index] for index in order]
    cluster_rows = [cluster_rows[index] for index in order]
    target_rows = [target_rows[index] for index in order]
    lengths = np.asarray([record["length"] for record in records], dtype=float)
    entropies = np.asarray([record["token_entropy"] for record in records], dtype=float)
    families = np.asarray([record["family"] for record in records], dtype=object)

    fitted = np.asarray([ev._gumbel_mle(scores) for scores in score_rows], dtype=float)
    response = np.column_stack([fitted[:, 0], np.log(fitted[:, 1])])
    predicted = np.zeros_like(response)
    fold_coefficients = {}
    for held_family in sorted(set(families.tolist())):
        held = families == held_family
        train = ~held
        train_design, centre = features(lengths[train], entropies[train])
        coefficients = ridge_fit(train_design, response[train], args.ridge)
        # Apply the same boundary policy as deployment, using only the
        # training families to determine each held-out fold's domain.
        domain = {
            "length_min": float(lengths[train].min()),
            "length_max": float(lengths[train].max()),
            "entropy_min": float(entropies[train].min()),
            "entropy_max": float(entropies[train].max()),
        }
        held_design, _ = features(
            np.clip(lengths[held], domain["length_min"], domain["length_max"]),
            np.clip(entropies[held], domain["entropy_min"], domain["entropy_max"]),
            centre,
        )
        predicted[held] = held_design @ coefficients
        fold_coefficients[held_family] = {
            "centre": centre,
            "training_domain": domain,
            "coefficients_mu_logbeta": coefficients.tolist(),
        }

    all_design, all_centre = features(lengths, entropies)
    all_coefficients = ridge_fit(all_design, response, args.ridge)
    rmse = np.sqrt(np.mean((predicted - response) ** 2, axis=0))
    output_records = []
    for index, record in enumerate(records):
        mu = float(predicted[index, 0])
        beta = float(math.exp(predicted[index, 1]))
        raw_evalues = evalues(score_rows[index], mu, beta, n_clusters)
        selected = np.flatnonzero(raw_evalues <= args.e_max)
        selected = selected[np.argsort(-score_rows[index][selected], kind="stable")]
        output_records.append({
            **record,
            "fitted_mu": float(fitted[index, 0]),
            "fitted_beta": float(fitted[index, 1]),
            "crossfit_mu": mu,
            "crossfit_beta": beta,
            "n_candidates_e_le_max": int(len(selected)),
            "candidates": [
                {
                    "cluster": int(cluster_rows[index][position]),
                    "target_index": int(target_rows[index][position]),
                    "score": float(score_rows[index][position]),
                    "raw_evalue": float(raw_evalues[position]),
                }
                for position in selected
            ],
        })

    output = {
        "scheme": "predicted-entropy-conditional-clustermax-crossfit",
        "score": "raw Structure-BLOSUM Smith-Waterman score",
        "features": ["log_length", "log_length_squared", "token_entropy",
                     "token_entropy_squared", "log_length_by_entropy"],
        "ridge": args.ridge,
        "n_queries": len(records),
        "n_families": len(set(families.tolist())),
        "n_clusters": n_clusters,
        "e_max": args.e_max,
        "rmse_mu": float(rmse[0]),
        "rmse_log_beta": float(rmse[1]),
        "deployment_model": {
            "centre": all_centre,
            "coefficients_mu_logbeta": all_coefficients.tolist(),
        },
        "fold_models": fold_coefficients,
        "records": output_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "scheme": output["scheme"],
        "n_queries": output["n_queries"],
        "n_families": output["n_families"],
        "rmse_mu": output["rmse_mu"],
        "rmse_log_beta": output["rmse_log_beta"],
        "total_candidates": sum(r["n_candidates_e_le_max"] for r in output_records),
        "queries_with_candidates": sum(r["n_candidates_e_le_max"] > 0 for r in output_records),
    }, indent=2))


if __name__ == "__main__":
    main()
