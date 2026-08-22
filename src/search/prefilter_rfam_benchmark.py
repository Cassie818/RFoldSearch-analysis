"""Shared Rfam label and retrieval-metric helpers."""
from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np


def load_labels(tsv_path: str, label_column: str = "rfam_acc") -> dict[str, str]:
    labels = {}
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            labels[row["entry_id"]] = row.get(label_column, "") or ""
    return labels


def attach_labels(entries: list[dict], labels: dict[str, str]) -> np.ndarray:
    """Return per-entry class labels; empty string means unlabeled distractor."""
    return np.array([labels.get(str(e["id"]), "") for e in entries], dtype=object)


def first_fp_sensitivity(
    ranking: np.ndarray,
    is_relevant: np.ndarray,
    is_neutral: np.ndarray | None = None,
) -> float:
    """Foldseek-style sensitivity up to the first false positive."""
    n_rel = int(is_relevant.sum())
    if n_rel == 0:
        return 0.0
    rel = is_relevant[ranking]
    neu = is_neutral[ranking] if is_neutral is not None else np.zeros(len(ranking), dtype=bool)
    tp = 0
    for r, nu in zip(rel, neu):
        if r:
            tp += 1
        elif nu:
            continue
        else:
            break
    return float(tp) / float(n_rel)


def evaluable_queries(fams: np.ndarray) -> np.ndarray:
    """Use labelled entries with at least one other same-class target as queries."""
    fam_counts = defaultdict(int)
    for fam in fams:
        if fam:
            fam_counts[fam] += 1
    return np.array(
        [i for i, fam in enumerate(fams) if fam and fam_counts[fam] >= 2],
        dtype=np.int64,
    )


def rfam_relevance_masks(
    q_idx: int,
    fams: np.ndarray,
    clans: np.ndarray,
    level: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Rfam true-positive and neutral masks for the benchmark hierarchy."""
    valid = np.ones(len(fams), dtype=bool)
    valid[q_idx] = False
    rel = np.zeros(len(fams), dtype=bool)
    neu = ~valid

    if level == "family":
        if fams[q_idx]:
            rel = (fams == fams[q_idx]) & valid
            if clans[q_idx]:
                neu |= (clans == clans[q_idx]) & (fams != fams[q_idx]) & valid
        return rel, neu

    if level == "clan":
        if clans[q_idx]:
            rel = (clans == clans[q_idx]) & (fams != fams[q_idx]) & valid
            if fams[q_idx]:
                neu |= (fams == fams[q_idx]) & valid
        return rel, neu

    raise ValueError(f"level must be 'family' or 'clan', got {level!r}")


def per_query_average_precision(
    ranked: np.ndarray,
    relevant: np.ndarray,
    neutral: np.ndarray,
    recall_grid: np.ndarray,
) -> tuple[float, np.ndarray] | None:
    """Average precision and interpolated PR curve for one query ranking."""
    n_rel = int(relevant.sum())
    if n_rel == 0:
        return None
    tp_flags = []
    for t in ranked:
        t = int(t)
        if t < 0 or t >= len(relevant) or neutral[t]:
            continue
        tp_flags.append(1.0 if relevant[t] else 0.0)
    if not tp_flags:
        return 0.0, np.zeros(len(recall_grid))

    tp_flags = np.asarray(tp_flags, dtype=np.float64)
    cum_tp = np.cumsum(tp_flags)
    ranks = np.arange(1, len(tp_flags) + 1)
    precision = cum_tp / ranks
    recall = cum_tp / n_rel
    ap = float((precision * tp_flags).sum() / n_rel)
    interp = np.array(
        [precision[recall >= r].max() if (recall >= r).any() else 0.0 for r in recall_grid]
    )
    return ap, interp


def rfam_per_query_pr_curve(
    query_ids,
    ranked_targets,
    fams: np.ndarray,
    clans: np.ndarray,
    level: str,
    n_recall_points: int = 101,
) -> dict:
    """Per-query-averaged precision-recall curve and mean AP."""
    grid = np.linspace(0.0, 1.0, n_recall_points)
    aps = []
    curves = []
    for q in query_ids:
        q = int(q)
        rel, neu = rfam_relevance_masks(q, fams, clans, level)
        if not rel.any():
            continue
        ranked = np.asarray(ranked_targets.get(q, np.zeros(0, dtype=np.int64)), dtype=np.int64)
        res = per_query_average_precision(ranked, rel, neu, grid)
        if res is None:
            continue
        ap, interp = res
        aps.append(ap)
        curves.append(interp)

    if not aps:
        return {
            "recall": grid.tolist(),
            "precision": [0.0] * n_recall_points,
            "aupr": 0.0,
            "aupr_std": 0.0,
            "per_query_ap": [],
            "n_queries": 0,
            "level": level,
            "weighting": "per-query mean average precision (each query weighted equally)",
        }

    return {
        "recall": grid.tolist(),
        "precision": np.mean(np.asarray(curves), axis=0).tolist(),
        "aupr": float(np.mean(aps)),
        "aupr_std": float(np.std(aps)),
        "per_query_ap": [float(a) for a in aps],
        "n_queries": int(len(aps)),
        "level": level,
        "weighting": "per-query mean average precision (each query weighted equally)",
    }
