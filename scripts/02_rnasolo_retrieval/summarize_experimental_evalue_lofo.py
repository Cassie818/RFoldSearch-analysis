"""Summarize experimental-database calibration over every calibration query."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
LEVELS = np.asarray([0.01, 0.1, 1.0, 10.0, 100.0])


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
        "--cross-validated",
        type=Path,
        default=(
            ROOT
            / "files/production/evalue_experimental/lofo_validation/"
            "cross_validated_evalues.npz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "files/production/evalue_experimental/lofo_validation/"
            "all_query_background_summary.json"
        ),
    )
    parser.add_argument("--tm-threshold", type=float, default=0.45)
    parser.add_argument("--dense", action="store_true",
                        help="Evaluate 24 log-spaced thresholds from 0.01 to 10")
    args = parser.parse_args()
    levels = np.logspace(-2, 1, 24) if args.dense else LEVELS

    evaluation = pickle.loads(args.evaluation.read_bytes())
    rows = evaluation["rows"]
    n_clusters = int(evaluation["n_clusters"])
    cross = np.load(args.cross_validated, allow_pickle=True)
    raw = cross["raw"]
    calibrated = cross["calibrated"]

    related = {}
    with args.adjudication.open(newline="") as handle:
        for record in csv.DictReader(handle, delimiter="\t"):
            related[(int(record["query_index"]), int(record["cluster_position"]))] = 0.5 * (
                float(record["tm_query_normalized"])
                + float(record["tm_target_normalized"])
            )

    result = {
        "n_queries": len(rows),
        "n_families": len({str(row["fam"]) for row in rows}),
        "n_clusters": n_clusters,
        "tm_score_definition": "mean of query- and target-normalized US-align TM-scores",
        "levels": levels.tolist(),
        "models": {},
    }
    for name, arrays in (("raw_clustermax", raw), ("empirical_lofo", calibrated)):
        by_family = defaultdict(list)
        for row, evalues in zip(rows, arrays):
            tags = np.asarray(row["tag"])
            labelled_background = tags == "bg"
            query = int(row["i"])
            related_mask = np.asarray(
                [
                    related.get((query, position), -np.inf) >= args.tm_threshold
                    for position in range(len(tags))
                ]
            )
            background = labelled_background & ~related_mask
            scale = n_clusters / int(labelled_background.sum())
            by_family[str(row["fam"])].append(
                np.asarray(
                    [float((np.asarray(evalues)[background] <= level).sum()) * scale
                     for level in levels]
                )
            )
        family_values = np.stack(
            [np.mean(values, axis=0) for _, values in sorted(by_family.items())]
        )
        observed = family_values.mean(axis=0)
        rng = np.random.default_rng(20260923)
        draws = rng.integers(
            0, len(family_values), (10000, len(family_values))
        )
        ci = np.percentile(family_values[draws].mean(axis=1), [2.5, 97.5], axis=0)
        result["models"][name] = {
            "observed_clusters_per_query": observed.tolist(),
            "observed_to_nominal": (observed / levels).tolist(),
            "family_block_95ci": ci.T.tolist(),
        }

    args.output.write_text(json.dumps(result, indent=2) + "\n")
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.linewidth": 0.9,
        }
    )
    fig, axis = plt.subplots(figsize=(4.4, 4.0))
    axis.plot(levels, levels, "--", color="#888888", linewidth=1.2,
              label="Perfect calibration")
    for name, color, label in (
        ("raw_clustermax", "#D97732", r"Parametric $E_0$"),
        ("empirical_lofo", "#2A9DCE", r"Empirical $E_{\mathrm{cal}}$"),
    ):
        values = np.asarray(result["models"][name]["observed_clusters_per_query"])
        axis.plot(levels, values, "o-", color=color, linewidth=1.5,
                  markersize=4.5, label=label)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Reported E-value")
    axis.set_ylabel("Observed unrelated clusters per query")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    stem = args.output.with_suffix("")
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
