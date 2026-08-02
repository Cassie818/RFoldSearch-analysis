"""
Shared Rfam label/metric helpers, plus an early exploratory prefilter benchmark.

Most of the package imports only the helper functions from this module —
`attach_labels`, `load_labels`, `first_fp_sensitivity`, `evaluable_queries` —
which define the Rfam family/clan labels and the Sens@1FP metric used
everywhere.

WARNING: the standalone benchmark in this file's `main()` is NOT the deployed
prefilter. Its `prefilter_scores()` ranks candidates by whole-chain token
*composition* similarity (histogram cosine / intersection / Jaccard), an early
exploration. The deployed pipeline instead uses a 4-mer inverted-index prefilter
with Structure-BLOSUM similar-k-mer expansion, implemented inline in the
`scripts/02_benchmark/` evaluation scripts (see also
`scripts/03_ablations/ablation_figS9_prefilter_scoring.py`). Running this file
to reproduce the README's "4-mer + similar-k-mer expansion" will not match.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smith_waterman import smith_waterman_batch, smith_waterman_fast, smith_waterman_numba


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
    ranking: np.ndarray, is_relevant: np.ndarray, is_neutral: np.ndarray | None = None
) -> float:
    """Foldseek-style sensitivity up to the first false positive (ROC1).

    Walk the ranked list from the top, counting same-family hits (TP) until the
    first false positive, then stop. The denominator is the TOTAL number of
    relevant (same-family) targets in the *whole* database, so homologs missed
    by any prefilter still count against sensitivity (the Foldseek/SCOP
    convention, not relevant-in-shortlist).

    ``is_neutral`` marks targets that are neither TP nor FP -- the Rfam analogue
    of SCOP same-superfamily-different-family hits, i.e. same clan but different
    family. They are skipped: not counted, and do not trigger the first-FP stop.
    Only a cross-clan (truly unrelated) hit is the first FP.
    """
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
    """Use labeled entries with at least one other same-class target as queries."""
    fam_counts = defaultdict(int)
    for fam in fams:
        if fam:
            fam_counts[fam] += 1
    return np.array(
        [i for i, fam in enumerate(fams) if fam and fam_counts[fam] >= 2], dtype=np.int64
    )


def rfam_relevance_masks(
    q_idx: int, fams: np.ndarray, clans: np.ndarray, level: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return Rfam TP and neutral masks for the Foldseek-style hierarchy.

    family:
        TP = same Rfam family; neutral = same clan but different family.
    clan:
        TP = same clan but different family; neutral = same family.

    Cross-clan labeled and unlabeled targets are false positives. Self-hits are
    always neutral so callers can pass rankings that have not removed them.
    """
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

    raise ValueError("level must be 'family' or 'clan', got %r" % (level,))


def per_query_average_precision(
    ranked: np.ndarray,
    relevant: np.ndarray,
    neutral: np.ndarray,
    recall_grid: np.ndarray,
) -> tuple[float, np.ndarray] | None:
    """Average precision and interpolated PR curve for one query's ranking.

    Walks the ranked target list (skipping neutral targets); a relevant target
    is a TP, anything else a FP. AP is the mean precision at the ranks where a
    TP is retrieved, normalized by the total number of relevant targets -- so
    relevant targets missing from the list (a prefilter returns only its top
    hits) lower AP. The curve is the standard right-interpolated precision on a
    fixed recall grid so that per-query curves can be averaged. Returns ``None``
    when the query has no relevant targets.
    """
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
    """Per-query-averaged PR curve and mean AP for Rfam family/clan retrieval.

    Each query contributes one average-precision value and one interpolated PR
    curve, weighted equally regardless of family size or query length. This is
    deliberately *not* a pooled curve: RNA query lengths span two orders of
    magnitude and long rRNAs share real local structure motifs with non-
    homologs, so a pooled curve (threshold swept across all queries at once) is
    dominated by the longest queries. Equal-per-query weighting removes that
    bias and matches the Sens@1FP metric (within-query comparison, across-query
    averaging). Ranking is taken directly from each method's own hit order, so
    no cross-query-comparable score is required.
    """
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
        "aupr": float(np.mean(aps)),  # mean average precision, length/size-equal
        "aupr_std": float(np.std(aps)),
        "per_query_ap": [float(a) for a in aps],
        "n_queries": int(len(aps)),
        "level": level,
        "weighting": "per-query mean average precision (each query weighted equally)",
    }


def parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def load_sub_matrix(path: str, n_codes: int) -> np.ndarray:
    sm = torch.load(path, map_location="cpu", weights_only=False)
    matrix = sm["matrix"] if isinstance(sm, dict) else sm
    if hasattr(matrix, "numpy"):
        matrix = matrix.numpy()
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (n_codes, n_codes):
        raise ValueError(
            "sub_matrix shape %s does not match database n_codes=%d" % (matrix.shape, n_codes)
        )
    return matrix


def token_histograms(entries: list[dict], n_codes: int) -> np.ndarray:
    hist = np.zeros((len(entries), n_codes), dtype=np.float64)
    for i, e in enumerate(entries):
        toks = e["tokens"].astype(np.int64)
        hist[i] = np.bincount(toks, minlength=n_codes)
    return hist


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def token_set_matrix(entries: list[dict], n_codes: int) -> np.ndarray:
    mat = np.zeros((len(entries), n_codes), dtype=bool)
    for i, e in enumerate(entries):
        mat[i, np.unique(e["tokens"].astype(np.int64))] = True
    return mat


def prefilter_scores(entries: list[dict], n_codes: int, method: str) -> np.ndarray:
    if method == "hist_cosine":
        hist = normalize_rows(token_histograms(entries, n_codes))
        return hist @ hist.T

    if method == "hist_intersection":
        hist = token_histograms(entries, n_codes)
        denom = np.maximum(hist.sum(axis=1, keepdims=True), 1.0)
        hist_norm = hist / denom
        scores = np.zeros((len(entries), len(entries)), dtype=np.float64)
        for i in range(len(entries)):
            scores[i] = np.minimum(hist_norm[i], hist_norm).sum(axis=1)
        return scores

    if method == "jaccard":
        sets = token_set_matrix(entries, n_codes)
        inter = sets.astype(np.int16) @ sets.T.astype(np.int16)
        counts = sets.sum(axis=1)
        union = counts[:, None] + counts[None, :] - inter
        return inter / np.maximum(union, 1)

    raise ValueError("Unknown prefilter method: %s" % method)


def build_padded(entries: list[dict], indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.array([len(entries[i]["tokens"]) for i in indices], dtype=np.int64)
    max_len = int(lengths.max()) if len(lengths) else 0
    padded = np.zeros((len(indices), max_len), dtype=np.int64)
    for row, idx in enumerate(indices):
        toks = entries[idx]["tokens"].astype(np.int64)
        padded[row, : len(toks)] = toks
    return padded, lengths


def evaluate_prefilter(
    entries: list[dict],
    fams: np.ndarray,
    evaluable_idx: np.ndarray,
    sub_matrix: np.ndarray,
    pre_scores: np.ndarray,
    top_n: int,
    gap_open: float,
    gap_extend: float,
    rerank_mode: str,
    clans: np.ndarray | None = None,
) -> dict:
    r1_hits = []
    fp1_sens = []
    per_fam = defaultdict(lambda: {"n": 0, "r1": 0, "fp1": 0.0})

    n_targets = len(entries)
    n_dp_pairs = 0
    t_dp = 0.0

    for qi, q_idx in enumerate(evaluable_idx):
        q_tokens = entries[q_idx]["tokens"].astype(np.int64)
        q_fam = fams[q_idx]
        candidates = np.argsort(-pre_scores[q_idx])
        candidates = candidates[candidates != q_idx][: min(top_n, n_targets - 1)]
        n_dp_pairs += len(candidates)

        t0 = time.time()
        if rerank_mode == "batch":
            padded, lengths = build_padded(entries, candidates)
            sw_scores = smith_waterman_batch(
                q_tokens,
                padded,
                lengths,
                sub_matrix,
                gap_open=gap_open,
                gap_extend=gap_extend,
            )
        elif rerank_mode == "fast_loop":
            sw_scores = np.array(
                [
                    smith_waterman_fast(
                        q_tokens,
                        entries[idx]["tokens"].astype(np.int64),
                        sub_matrix,
                        gap_open=gap_open,
                        gap_extend=gap_extend,
                    )
                    for idx in candidates
                ],
                dtype=np.float32,
            )
        elif rerank_mode == "numba_loop":
            sw_scores = np.array(
                [
                    smith_waterman_numba(
                        q_tokens,
                        entries[idx]["tokens"].astype(np.int64),
                        sub_matrix,
                        gap_open=gap_open,
                        gap_extend=gap_extend,
                    )
                    for idx in candidates
                ],
                dtype=np.float32,
            )
        else:
            raise ValueError("Unknown rerank_mode: %s" % rerank_mode)
        t_dp += time.time() - t0

        order = np.argsort(-sw_scores)
        ranking = candidates[order]
        valid = np.arange(n_targets) != q_idx
        is_rel = (fams == q_fam) & valid
        # Same-clan-different-family hits are neutral (Foldseek convention).
        if clans is not None and clans[q_idx]:
            is_neutral = (clans == clans[q_idx]) & (fams != q_fam) & valid
        else:
            is_neutral = None

        r1 = int(is_rel[ranking[:1]].any()) if len(ranking) else 0
        fp1 = first_fp_sensitivity(ranking, is_rel, is_neutral)

        r1_hits.append(r1)
        fp1_sens.append(fp1)
        per_fam[q_fam]["n"] += 1
        per_fam[q_fam]["r1"] += r1
        per_fam[q_fam]["fp1"] += fp1

    out = {
        "Recall@1": float(np.mean(r1_hits)),
        "FirstFP_sensitivity": float(np.mean(fp1_sens)),
        "dp_pairs": int(n_dp_pairs),
        "dp_time_sec": float(t_dp),
    }
    out["per_family"] = {
        fam: {
            "n": st["n"],
            "R@1": st["r1"] / st["n"],
            "FirstFP_sensitivity": st["fp1"] / st["n"],
        }
        for fam, st in per_fam.items()
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parent = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    parser.add_argument("--db", default=os.path.join(parent, "files", "search", "pdb_out", "db.pt"))
    parser.add_argument(
        "--sub_matrix",
        default=os.path.join(parent, "files", "codebook", "sub_matrix_codebook_cosine.pt"),
        help="Substitution matrix for SW reranking. Defaults to the codebook-cosine "
        "matrix from build_matrix_codebook_cosine.py; pass the deployed "
        "files/models/sub_matrix.pt to rerank with Structure-BLOSUM.",
    )
    parser.add_argument("--labels", default=os.path.join(parent, "Splits", "labels.tsv"))
    parser.add_argument(
        "--label_column",
        default="rfam_acc",
        help="TSV column to use as 'same-class' label. "
        "rfam_acc = family-level retrieval (default); "
        "clan_acc = Rfam-clan-level retrieval.",
    )
    parser.add_argument(
        "--output", default=os.path.join(parent, "files", "search", "prefilter_results.json")
    )
    parser.add_argument("--methods", default="hist_cosine,jaccard,hist_intersection")
    parser.add_argument("--top_ns", default="10,20,30,50")
    parser.add_argument("--gap_open", type=float, default=-2.0)
    parser.add_argument("--gap_extend", type=float, default=-0.5)
    parser.add_argument(
        "--rerank_mode", choices=["batch", "fast_loop", "numba_loop"], default="numba_loop"
    )
    args = parser.parse_args()

    print("Loading DB: %s" % args.db, flush=True)
    db = torch.load(args.db, map_location="cpu", weights_only=False)
    entries_all = db["entries"]
    n_codes = int(db["n_codes"])

    print("Loading labels: %s" % args.labels, flush=True)
    labels = load_labels(args.labels, label_column=args.label_column)
    entries = list(entries_all)
    fams = attach_labels(entries, labels)
    evaluable_idx = evaluable_queries(fams)
    # For family-level retrieval, load clans so that same-clan-different-family
    # hits can be treated as neutral in Sens@1FP (Foldseek convention).
    if args.label_column == "rfam_acc":
        clans = attach_labels(entries, load_labels(args.labels, label_column="clan_acc"))
    else:
        clans = None

    print("Loading sub_matrix: %s" % args.sub_matrix, flush=True)
    sub_matrix = load_sub_matrix(args.sub_matrix, n_codes)

    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    top_ns = parse_ints(args.top_ns)

    labeled_mask = fams != ""
    labeled_fams = set(fams[labeled_mask])
    print(
        "  targets: %d entries (%d labeled, %d unlabeled distractors), %d unique classes"
        % (
            len(entries),
            int(labeled_mask.sum()),
            int((~labeled_mask).sum()),
            len(labeled_fams),
        ),
        flush=True,
    )
    print(
        "  queries: %d entries, %d classes"
        % (len(evaluable_idx), len(set(fams[evaluable_idx])) if len(evaluable_idx) else 0),
        flush=True,
    )
    print("  full DP pairs would be: %d" % (len(evaluable_idx) * (len(entries) - 1)), flush=True)

    all_results = {}
    for method in methods:
        print("\nPrefilter: %s" % method, flush=True)
        t0 = time.time()
        scores = prefilter_scores(entries, n_codes, method)
        pre_time = time.time() - t0
        method_results = {}
        for top_n in top_ns:
            t0 = time.time()
            metrics = evaluate_prefilter(
                entries,
                fams,
                evaluable_idx,
                sub_matrix,
                scores,
                top_n,
                gap_open=args.gap_open,
                gap_extend=args.gap_extend,
                rerank_mode=args.rerank_mode,
                clans=clans,
            )
            total_time = time.time() - t0 + pre_time
            metrics["prefilter_time_sec"] = float(pre_time)
            metrics["total_time_sec"] = float(total_time)
            metrics["top_n"] = int(top_n)
            metrics["method"] = method
            metrics["rerank_mode"] = args.rerank_mode
            method_results[str(top_n)] = metrics
            print(
                "  top%-4d R@1=%.3f FP1-sens=%.3f "
                "time=%.2fs dp_pairs=%d"
                % (
                    top_n,
                    metrics["Recall@1"],
                    metrics["FirstFP_sensitivity"],
                    metrics["total_time_sec"],
                    metrics["dp_pairs"],
                ),
                flush=True,
            )
        all_results[method] = method_results

    payload = {
        "db": args.db,
        "sub_matrix": args.sub_matrix,
        "labels": args.labels,
        "label_column": args.label_column,
        "n_queries": int(len(evaluable_idx)),
        "n_targets": int(len(entries)),
        "n_labeled_targets": int(labeled_mask.sum()),
        "n_unlabeled_distractors": int((~labeled_mask).sum()),
        "n_classes_queried": int(len(set(fams[evaluable_idx])) if len(evaluable_idx) else 0),
        "n_families_queried": int(len(set(fams[evaluable_idx])) if len(evaluable_idx) else 0),
        "full_dp_pairs": int(len(evaluable_idx) * (len(entries) - 1)),
        "rerank_mode": args.rerank_mode,
        "gap_open": float(args.gap_open),
        "gap_extend": float(args.gap_extend),
        "top_ns": top_ns,
        "methods": methods,
        "results": all_results,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print("\nSaved: %s" % args.output, flush=True)


if __name__ == "__main__":
    main()
