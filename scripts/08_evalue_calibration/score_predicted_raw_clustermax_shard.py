#!/usr/bin/env python3
"""Save complete raw-score cluster maxima for predicted calibration queries."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch


UNKNOWN = {"", "N/A", "NA", "None", "none", ".", "-"}


def load_sw(upstream: Path):
    package = types.ModuleType("rfoldseek")
    package.__path__ = [str(upstream / "rfoldseek")]
    sys.modules["rfoldseek"] = package
    name = "rfoldseek.smith_waterman"
    spec = importlib.util.spec_from_file_location(name, upstream / "rfoldseek/smith_waterman.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_clusters(path: Path, n_chains: int) -> np.ndarray:
    labels = np.full(n_chains, -1, dtype=np.int32)
    cluster = -1
    with path.open() as handle:
        for line in handle:
            if line.startswith(">Cluster"):
                cluster = int(line.split()[1])
            else:
                match = re.search(r">t(\d+)\.\.\.", line)
                if match:
                    labels[int(match.group(1))] = cluster
    if np.any(labels < 0):
        raise ValueError("cluster file does not cover every chain")
    return labels


def token_entropy(tokens: np.ndarray, n_states: int) -> float:
    counts = np.bincount(tokens, minlength=n_states).astype(float)
    probabilities = counts[counts > 0] / len(tokens)
    return float(-(probabilities * np.log2(probabilities)).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--n-shards", type=int, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--cluster-file", type=Path, required=True)
    parser.add_argument("--legacy-background", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--structures", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    sw = load_sw(args.upstream)
    database = torch.load(args.db, map_location="cpu", weights_only=False)
    entries = database["entries"]
    ids = [Path(str(entry["id"])).stem for entry in entries]
    id_to_index = {chain_id: index for index, chain_id in enumerate(ids)}
    tokens = [np.asarray(entry["tokens"], dtype=np.int64) for entry in entries]
    lengths = np.asarray([len(value) for value in tokens])
    clusters = read_clusters(args.cluster_file, len(entries))
    matrix = np.asarray(torch.load(
        args.upstream / "rfoldseek/assets/sub_matrix.pt",
        map_location="cpu", weights_only=False,
    )["matrix"].numpy(), dtype=np.float32)
    family_by_urs = {
        row["urs_id"]: row["rfam_family"]
        for row in csv.DictReader(args.metadata.open())
    }
    family = np.asarray([
        family_by_urs.get("_".join(chain_id.split("_")[:2]), "")
        for chain_id in ids
    ], dtype=object)
    clan_by_family = {
        row["rfam_acc"]: row["clan_acc"]
        for row in csv.DictReader(args.labels.open(), delimiter="\t")
        if row.get("rfam_acc") and row.get("clan_acc")
    }
    clan = np.asarray([clan_by_family.get(value, "") for value in family], dtype=object)
    exists = np.asarray([
        (args.structures / f"{chain_id}.pdb").is_file() for chain_id in ids
    ])
    legacy = json.loads(args.legacy_background.read_text())
    queries = [record for record in legacy["records"] if record["family"] not in UNKNOWN]
    selected = [
        (position, record) for position, record in enumerate(queries)
        if position % args.n_shards == args.shard_index
    ]

    sw.smith_waterman_numba(
        np.zeros(4, dtype=np.int64), np.zeros(4, dtype=np.int64),
        matrix, -2.0, -0.5,
    )
    records = []
    scores, cluster_ids, target_indices = [], [], []
    offsets = [0]
    started = time.monotonic()
    for number, (global_position, prior) in enumerate(selected, 1):
        query_id = prior["query"]
        qi = id_to_index[query_id]
        query = tokens[qi]
        qfamily, qclan, qcluster = family[qi], clan[qi], int(clusters[qi])
        lo, hi = len(query) / 3.0, len(query) * 3.0
        best = {}
        for j, target in enumerate(tokens):
            cluster = int(clusters[j])
            if cluster == qcluster or not exists[j] or not (lo <= lengths[j] <= hi):
                continue
            if qfamily not in UNKNOWN and family[j] == qfamily:
                continue
            if qclan not in UNKNOWN and clan[j] == qclan:
                continue
            score = float(sw.smith_waterman_numba(query, target, matrix, -2.0, -0.5))
            if cluster not in best or score > best[cluster][0]:
                best[cluster] = (score, j)
        ordered = sorted(best.items())
        cluster_ids.extend(cluster for cluster, _ in ordered)
        scores.extend(value[0] for _, value in ordered)
        target_indices.extend(value[1] for _, value in ordered)
        offsets.append(len(scores))
        records.append({
            "global_position": global_position,
            "query": query_id,
            "query_index": qi,
            "family": str(qfamily),
            "length": int(len(query)),
            "token_entropy": token_entropy(query, matrix.shape[0]),
            "n_unique_tokens": int(len(np.unique(query))),
            "n_eligible_clusters": len(best),
        })
        print(
            f"shard {args.shard_index} {number}/{len(selected)}: {query_id}, "
            f"{len(best)} maxima, {time.monotonic()-started:.0f}s",
            flush=True,
        )

    metadata = {
        "scheme": "predicted-raw-clustermax-full-shard",
        "shard_index": args.shard_index,
        "n_shards": args.n_shards,
        "n_chains": len(entries),
        "n_clusters": len(set(clusters.tolist())),
        "records": records,
    }
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    np.savez_compressed(
        args.output_prefix.with_suffix(".npz"),
        offsets=np.asarray(offsets, dtype=np.int64),
        scores=np.asarray(scores, dtype=np.float32),
        cluster_ids=np.asarray(cluster_ids, dtype=np.int32),
        target_indices=np.asarray(target_indices, dtype=np.int32),
    )
    print(
        f"wrote {args.output_prefix.with_suffix('.json')} and "
        f"{args.output_prefix.with_suffix('.npz')} in {time.monotonic()-started:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
