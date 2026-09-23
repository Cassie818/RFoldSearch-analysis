#!/usr/bin/env python3
"""Add Infernal rankings to the existing Fig. 2 retrieval summaries.

The input is the raw per-query output produced by ``_infernal_runner.py``.
This script deliberately re-scores those rankings with the exact relevance,
neutral-hit, query-selection, and TM-score definitions already used by Fig. 2,
rather than importing the quick summaries printed by the Infernal runner.

It updates:
  files/production/retrieval/per_query_sens_test.json
  files/production/retrieval/structural_tmscore_per_query_test.json
  files/production/retrieval/structural_tmscore_metrics_test.json
  files/production/retrieval/bootstrap_fig2cd_results.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RET = ROOT / "files/production/retrieval"
B = 2000
SEED = 123


def read_database_order() -> tuple[list[str], np.ndarray, np.ndarray]:
    labels = {}
    with open(ROOT / "Splits/labels.tsv") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            labels[Path(row["entry_id"]).stem] = row

    tags = []
    with open(ROOT / "files/production/chain_sequences.tsv") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["chain_id"] in labels:
                tags.append(row["chain_id"])
    if len(tags) != 885 or len(set(tags)) != len(tags):
        raise RuntimeError(f"Expected 885 unique benchmark chains, found {len(tags)}")

    fam = np.asarray([labels[t].get("rfam_acc", "") or "" for t in tags], dtype=object)
    clan = np.asarray([labels[t].get("clan_acc", "") or "" for t in tags], dtype=object)
    return tags, fam, clan


def first_fp(ranking: np.ndarray, relevant: np.ndarray,
             neutral: np.ndarray | None = None) -> float:
    n_rel = int(relevant.sum())
    if n_rel == 0:
        return 0.0
    if neutral is None:
        neutral = np.zeros(len(relevant), dtype=bool)
    tp = 0
    for target in ranking:
        if relevant[target]:
            tp += 1
        elif neutral[target]:
            continue
        else:
            break
    return tp / n_rel


def rfam_masks(q: int, fam: np.ndarray, clan: np.ndarray,
               level: str) -> tuple[np.ndarray, np.ndarray]:
    valid = np.ones(len(fam), dtype=bool)
    valid[q] = False
    neutral = ~valid
    if level == "family":
        relevant = (fam == fam[q]) & valid if fam[q] else np.zeros(len(fam), bool)
        if clan[q]:
            neutral |= (clan == clan[q]) & (fam != fam[q]) & valid
    elif level == "clan":
        relevant = ((clan == clan[q]) & (fam != fam[q]) & valid
                    if clan[q] else np.zeros(len(fam), bool))
        if fam[q]:
            neutral |= (fam == fam[q]) & valid
    else:
        raise ValueError(level)
    return relevant, neutral


def load_rankings(path: Path, tags: list[str]) -> dict[int, np.ndarray]:
    raw = json.load(open(path))
    id2i = {tag: i for i, tag in enumerate(tags)}
    unknown_queries = sorted(set(raw) - set(id2i))
    if unknown_queries:
        raise RuntimeError(f"Unknown Infernal query IDs: {unknown_queries[:3]}")
    ranks = {}
    for query_id, result in raw.items():
        q = id2i[query_id]
        seen = {q}
        ordered = []
        for target_id in result.get("ranking", []):
            target = id2i.get(target_id)
            if target is not None and target not in seen:
                ordered.append(target)
                seen.add(target)
        ranks[q] = np.asarray(ordered, dtype=np.int64)
    return ranks


def update_rfam(ranks: dict[int, np.ndarray], fam: np.ndarray,
                clan: np.ndarray) -> None:
    path = RET / "per_query_sens_test.json"
    data = json.load(open(path))
    method = {}
    for level in ("family", "clan"):
        values = []
        for q in data["_query_ids"][level]:
            relevant, neutral = rfam_masks(q, fam, clan, level)
            values.append(first_fp(ranks.get(q, np.zeros(0, np.int64)), relevant, neutral))
        method[level] = {"sens": values, "n": len(values), "mean": float(np.mean(values))}
    data["Infernal"] = method
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)


def load_tm(n: int) -> np.ndarray:
    tm = np.full((n, n), np.nan, dtype=np.float32)
    with open(RET / "usalign_full.cache.jsonl") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            i, j = int(row["i"]), int(row["j"])
            score = 0.5 * (float(row["tm_query"]) + float(row["tm_target"]))
            tm[i, j] = score
            tm[j, i] = score
    return tm


def update_structural(ranks: dict[int, np.ndarray], fam: np.ndarray,
                      clan: np.ndarray) -> None:
    pq_path = RET / "structural_tmscore_per_query_test.json"
    metric_path = RET / "structural_tmscore_metrics_test.json"
    per_query = json.load(open(pq_path))
    metrics = json.load(open(metric_path))
    tm = load_tm(len(fam))

    for threshold_key in sorted(per_query, key=float):
        threshold = float(threshold_key)
        reference_method = next(iter(per_query[threshold_key]))
        family_queries = [int(q) for q in per_query[threshold_key][reference_method]["family"]]
        clan_queries = [int(q) for q in per_query[threshold_key][reference_method]["clan"]]
        family_scores = {}
        clan_scores = {}
        recall1 = []

        for q in family_queries:
            valid = np.ones(len(fam), dtype=bool)
            valid[q] = False
            positive = (~np.isnan(tm[q])) & (tm[q] >= threshold) & valid
            ranking = ranks.get(q, np.zeros(0, np.int64))
            family_scores[str(q)] = first_fp(ranking, positive)
            recall1.append(int(len(ranking) > 0 and positive[ranking[0]]))

        for q in clan_queries:
            valid = np.ones(len(fam), dtype=bool)
            valid[q] = False
            positive = (~np.isnan(tm[q])) & (tm[q] >= threshold) & valid
            relevant = positive & (fam != fam[q])
            neutral = (fam == fam[q]) & valid
            ranking = ranks.get(q, np.zeros(0, np.int64))
            clan_scores[str(q)] = first_fp(ranking, relevant, neutral)

        per_query[threshold_key]["Infernal"] = {
            "family": family_scores,
            "clan": clan_scores,
        }
        metrics["thresholds"][threshold_key]["Infernal"] = {
            "Recall@1": float(np.mean(recall1)),
            "Sens@1FP": float(np.mean(list(family_scores.values()))),
            "Sens@1FP_clan": float(np.mean(list(clan_scores.values()))),
            "n_queries": len(family_scores),
            "n_clan": len(clan_scores),
        }

    with open(pq_path, "w") as handle:
        json.dump(per_query, handle, indent=2)
    with open(metric_path, "w") as handle:
        json.dump(metrics, handle, indent=2)
    update_bootstrap(per_query, fam, clan)


def update_bootstrap(per_query: dict, fam: np.ndarray, clan: np.ndarray) -> None:
    rng = np.random.default_rng(SEED)
    output = {}
    for threshold in sorted(per_query, key=float):
        output[threshold] = {"family": {}, "clan": {}}
        for level in ("family", "clan"):
            block_labels = fam if level == "family" else clan
            methods = [m for m in per_query[threshold] if per_query[threshold][m][level]]
            qset = list(per_query[threshold][methods[0]][level])
            blocks = defaultdict(list)
            for q in qset:
                if block_labels[int(q)]:
                    blocks[block_labels[int(q)]].append(q)
            names = list(blocks)
            bootstrap = {m: [] for m in methods}
            for _ in range(B):
                sampled = [q for block in rng.choice(names, size=len(names), replace=True)
                           for q in blocks[block]]
                for method in methods:
                    bootstrap[method].append(np.mean([
                        per_query[threshold][method][level][q] for q in sampled
                    ]))
            bootstrap = {m: np.asarray(v) for m, v in bootstrap.items()}
            for method in methods:
                lo, hi = np.quantile(bootstrap[method], [0.025, 0.975])
                significant = None
                if method != "RFoldseek":
                    delta = bootstrap["RFoldseek"] - bootstrap[method]
                    dlo, dhi = np.quantile(delta, [0.025, 0.975])
                    significant = bool(dlo > 0 or dhi < 0)
                output[threshold][level][method] = {
                    "estimate": float(np.mean([
                        per_query[threshold][method][level][q] for q in qset
                    ])),
                    "ci_lower": float(lo),
                    "ci_upper": float(hi),
                    "sig_vs_rf": significant,
                    "n_blocks": len(names),
                }
    with open(RET / "bootstrap_fig2cd_results.json", "w") as handle:
        json.dump(output, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--infernal-json", type=Path, default=RET / "infernal_rankings_test.json")
    args = parser.parse_args()
    tags, fam, clan = read_database_order()
    ranks = load_rankings(args.infernal_json, tags)
    update_rfam(ranks, fam, clan)
    update_structural(ranks, fam, clan)
    print(f"Added Infernal from {args.infernal_json} ({len(ranks)} queries)")


if __name__ == "__main__":
    main()
