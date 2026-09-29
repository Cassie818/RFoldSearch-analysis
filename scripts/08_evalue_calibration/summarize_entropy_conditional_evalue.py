#!/usr/bin/env python3
"""Family-level LOFO validation of entropy-conditional predicted E-values."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


LEVELS = np.logspace(-2, 1, 24)


def fit_map(records, indices):
    families = sorted({records[index]["family"] for index in indices})
    by_family = defaultdict(list)
    for index in indices:
        by_family[records[index]["family"]].append(index)
    values, weights = [], []
    for index in indices:
        family = records[index]["family"]
        weight = 1.0 / (len(families) * len(by_family[family]))
        background = [
            candidate["raw_evalue"] for candidate in records[index]["candidates"]
            if candidate["background"]
        ]
        values.extend(background)
        weights.extend([weight] * len(background))
    order = np.argsort(values, kind="stable")
    return (np.asarray(values)[order], np.cumsum(np.asarray(weights)[order]),
            0.5 * min(weights))


def apply_map(values, fitted):
    knots, cumulative, floor = fitted
    positions = np.searchsorted(knots, values, side="right")
    result = np.full(len(values), floor, dtype=float)
    present = positions > 0
    result[present] = cumulative[positions[present] - 1]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    records = data["records"]
    by_family = defaultdict(list)
    for index, record in enumerate(records):
        by_family[record["family"]].append(index)
    raw = {float(level): [] for level in LEVELS}
    calibrated = {float(level): [] for level in LEVELS}
    details = []
    for family, held_indices in sorted(by_family.items()):
        training = [index for index in range(len(records)) if records[index]["family"] != family]
        mapping = fit_map(records, training)
        detail = {"family": family, "n_queries": len(held_indices), "levels": {}}
        for level in LEVELS:
            raw_counts, calibrated_counts = [], []
            for index in held_indices:
                values = np.asarray([
                    candidate["raw_evalue"] for candidate in records[index]["candidates"]
                    if candidate["background"]
                ], dtype=float)
                raw_counts.append(int(np.count_nonzero(values <= level)))
                calibrated_counts.append(int(np.count_nonzero(apply_map(values, mapping) <= level)))
            raw_mean = float(np.mean(raw_counts))
            calibrated_mean = float(np.mean(calibrated_counts))
            raw[float(level)].append(raw_mean)
            calibrated[float(level)].append(calibrated_mean)
            detail["levels"][str(float(level))] = {
                "raw": raw_mean, "calibrated": calibrated_mean,
            }
        details.append(detail)
    output = {
        "scheme": "predicted-entropy-conditional-empirical-lofo",
        "n_queries": len(records),
        "n_families": len(by_family),
        "n_clusters": data["n_clusters"],
        "n_candidates": sum(len(record["candidates"]) for record in records),
        "n_background": data["n_background"],
        "rmse_mu": data["rmse_mu"],
        "rmse_log_beta": data["rmse_log_beta"],
        "levels": [float(level) for level in LEVELS],
        "raw": {"observed": [float(np.mean(raw[float(level)])) for level in LEVELS]},
        "calibrated": {
            "observed": [float(np.mean(calibrated[float(level)])) for level in LEVELS]
        },
        "family_details": details,
        "deployment_model": data["deployment_model"],
    }
    rng = np.random.default_rng(20260924)
    for name, values_by_level in (("raw", raw), ("calibrated", calibrated)):
        family_values = np.asarray([
            values_by_level[float(level)] for level in LEVELS
        ], dtype=float).T
        draws = rng.integers(0, len(family_values), (10000, len(family_values)))
        bootstrap = family_values[draws].mean(axis=1)
        intervals = np.percentile(bootstrap, [2.5, 97.5], axis=0).T
        output[name]["family_block_95ci"] = intervals.tolist()
        output[name]["observed_to_nominal"] = (
            np.asarray(output[name]["observed"]) / LEVELS
        ).tolist()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: output[key] for key in (
        "scheme", "n_queries", "n_families", "n_candidates", "n_background",
        "rmse_mu", "rmse_log_beta", "levels", "raw", "calibrated",
    )}, indent=2))


if __name__ == "__main__":
    main()
