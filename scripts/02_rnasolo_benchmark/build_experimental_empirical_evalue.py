"""Fit and embed the final empirical E-value map for the 885-chain database."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import pickle
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = Path("/Users/cassie/Desktop/RFoldseek-upstream")

# Load only the E-value module and its lightweight Smith--Waterman dependency.
# Importing the package initializer also imports optional structure libraries,
# which are irrelevant here and can have a different NumPy ABI.
package = types.ModuleType("rfoldseek")
package.__path__ = [str(UPSTREAM / "rfoldseek")]
sys.modules["rfoldseek"] = package


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name, UPSTREAM / "rfoldseek" / filename
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


load_module("rfoldseek.smith_waterman", "smith_waterman.py")
EV = load_module("rfoldseek.evalue", "evalue.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=Path(
            "/Users/cassie/Desktop/RFoldSeek/data/experimental/"
            "nr80_885_evalue_structure_adjudication/qevd_nr80_885.pkl"
        ),
    )
    parser.add_argument(
        "--adjudication",
        type=Path,
        default=Path(
            "/Users/cassie/Desktop/RFoldSeek/data/experimental/"
            "nr80_885_database_adaptive_calibration/adjudication_rawE1.tsv"
        ),
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=UPSTREAM / "rfoldseek/assets/evalue_params.json",
    )
    parser.add_argument(
        "--database", type=Path, default=ROOT / "files/models/db.pt"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "files/production/evalue_experimental",
    )
    parser.add_argument("--tm-threshold", type=float, default=0.45)
    args = parser.parse_args()

    evaluation = pickle.loads(args.evaluation.read_bytes())
    rows = evaluation["rows"]
    n_clusters = int(evaluation["n_clusters"])
    params = json.loads(args.params.read_text())
    if int(params["n_clusters"]) != n_clusters:
        raise ValueError(
            f"parameter/database cluster mismatch: {params['n_clusters']} != {n_clusters}"
        )

    related = {}
    with args.adjudication.open(newline="") as handle:
        for record in csv.DictReader(handle, delimiter="\t"):
            tm = min(
                float(record["tm_query_normalized"]),
                float(record["tm_target_normalized"]),
            )
            related[(int(record["query_index"]), int(record["cluster_position"]))] = tm

    by_family = defaultdict(list)
    for index, row in enumerate(rows):
        by_family[str(row["fam"])].append(index)

    values, weights = [], []
    excluded_related = 0
    for family, indices in sorted(by_family.items()):
        family_weight = 1.0 / len(by_family)
        for index in indices:
            row = rows[index]
            tags = np.asarray(row["tag"])
            labelled_background = tags == "bg"
            denominator = int(labelled_background.sum())
            if denominator == 0:
                continue
            scores = np.asarray(row["scores"], dtype=float)
            mu, beta = EV.clustermax_params_for_length(params, float(row["len"]))
            raw_evalues = EV.evalue_clustermax(scores, mu, beta, n_clusters)
            query = int(row["i"])
            structurally_related = np.asarray(
                [
                    related.get((query, position), -np.inf) >= args.tm_threshold
                    for position in range(len(tags))
                ],
                dtype=bool,
            )
            background = labelled_background & ~structurally_related
            excluded_related += int((labelled_background & structurally_related).sum())
            observation_weight = (
                family_weight * n_clusters / (len(indices) * denominator)
            )
            values.extend(raw_evalues[background].tolist())
            weights.extend([observation_weight] * int(background.sum()))

    calibration = EV.fit_empirical_calibration(
        np.asarray(values), np.asarray(weights), n_clusters=n_clusters
    )
    calibration.update(
        {
            "weighting": "equal-family, equal-query-within-family; scaled to target cluster count",
            "n_families": len(by_family),
            "n_calibration_queries": len(rows),
            "tm_threshold": args.tm_threshold,
            "structurally_related_background_excluded": excluded_related,
            "source": str(args.evaluation),
            "adjudication": str(args.adjudication),
        }
    )
    deployed = dict(params)
    deployed["empirical_calibration"] = calibration

    database = torch.load(args.database, map_location="cpu", weights_only=False)
    database_clusters = len(
        {entry.get("cluster", f"_singleton_{i}") for i, entry in enumerate(database["entries"])}
    )
    if database_clusters != n_clusters:
        raise ValueError(
            f"database has {database_clusters} clusters; calibration has {n_clusters}"
        )
    database["evalue_params"] = deployed

    args.output_dir.mkdir(parents=True, exist_ok=True)
    params_path = args.output_dir / "evalue_params_experimental_empirical.json"
    database_path = args.output_dir / "db_885_empirical_evalue.pt"
    summary_path = args.output_dir / "deployment_summary.json"
    params_path.write_text(json.dumps(deployed, indent=2) + "\n")
    torch.save(database, database_path)
    summary = {
        "database": str(args.database),
        "deployed_database": str(database_path),
        "parameters": str(params_path),
        "n_chains": len(database["entries"]),
        "n_clusters": n_clusters,
        "n_calibration_queries": len(rows),
        "n_families": len(by_family),
        "n_background_observations": len(values),
        "excluded_structurally_related_background": excluded_related,
        "total_background_weight": calibration["total_weight"],
        "finite_sample_floor": calibration["finite_sample_floor"],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
