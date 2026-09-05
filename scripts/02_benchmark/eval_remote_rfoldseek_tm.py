"""Evaluate RFoldseek-TM for the remote-homologue benchmark.

RFoldseek-TM is the Foldseek-style coordinate-reranking variant used in the
NuFold remote-homologue analysis: RFoldseek supplies a token prefilter candidate
pool, then US-align reranks only those candidates by the mean of its two
length-normalised TM-scores.

The script is intentionally format-tolerant because the full NuFold rankings are
large external artifacts. It accepts JSON, JSONL, or TSV candidate files and
US-align score files, evaluates the query sets stored in filter_state.json, and
writes per-query and summary metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


def _normalise_id(value: Any) -> str:
    """Convert IDs from JSON/TSV payloads into the string IDs used for scoring."""
    if value is None:
        return ""
    return str(value).strip()


def _read_json_or_jsonl(path: Path) -> Any:
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows


def load_labels(path: Path, label_column: str) -> dict[str, str]:
    """Load Rfam family labels keyed by chain/database entry ID."""
    labels: dict[str, str] = {}
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "entry_id" not in reader.fieldnames:
            raise ValueError(f"{path} must contain an entry_id column")
        if label_column not in reader.fieldnames:
            raise ValueError(f"{path} must contain a {label_column!r} column")
        for row in reader:
            labels[_normalise_id(row["entry_id"])] = _normalise_id(row.get(label_column, ""))
    return labels


def _targets_from_payload(payload: Any) -> list[str]:
    """Extract ordered target IDs from a candidate payload for one query."""
    if isinstance(payload, dict):
        for key in ("candidates", "ranking", "ranked", "targets", "hits"):
            if key in payload:
                return _targets_from_payload(payload[key])
        for key in ("target", "target_id", "chain", "id", "sseqid"):
            if key in payload:
                return [_normalise_id(payload[key])]
        return []

    if not isinstance(payload, list):
        return [_normalise_id(payload)]

    out: list[str] = []
    for item in payload:
        if isinstance(item, dict):
            for key in ("target", "target_id", "chain", "id", "sseqid"):
                if key in item:
                    target = _normalise_id(item[key])
                    if target:
                        out.append(target)
                    break
        elif isinstance(item, (list, tuple)) and item:
            target = _normalise_id(item[0])
            if target:
                out.append(target)
        else:
            target = _normalise_id(item)
            if target:
                out.append(target)
    return out


def load_candidates(path: Path) -> dict[str, list[str]]:
    """Load prefilter candidates keyed by query ID.

    Supported inputs:
      * JSON dict: {"query_id": ["target1", ...]} or values with a candidates
        / ranking / hits field.
      * JSON list/JSONL: rows with query/q/qseqid and target/t/target_id/chain.
      * TSV: columns query/q/qseqid and target/t/target_id/chain.
    """
    if path.suffix.lower() in {".json", ".jsonl"}:
        data = _read_json_or_jsonl(path)
        if isinstance(data, dict):
            if "queries" in data and isinstance(data["queries"], dict):
                data = data["queries"]
            return {
                _normalise_id(query): _targets_from_payload(payload)
                for query, payload in data.items()
                if _normalise_id(query)
            }
        if isinstance(data, list):
            by_query: dict[str, list[tuple[int, str]]] = {}
            for pos, row in enumerate(data):
                if not isinstance(row, dict):
                    continue
                query = _first_present(row, ("query", "query_id", "q", "qseqid"))
                target = _first_present(row, ("target", "target_id", "t", "chain", "sseqid"))
                if not query or not target:
                    continue
                rank = _safe_int(row.get("rank"), pos)
                by_query.setdefault(query, []).append((rank, target))
            return {
                query: [target for _, target in sorted(rows, key=lambda x: x[0])]
                for query, rows in by_query.items()
            }
        raise ValueError(f"Unsupported candidate JSON structure in {path}")

    with path.open() as f:
        sample = f.read(4096)
        f.seek(0)
        has_header = any(name in sample.splitlines()[0].lower().split("\t") for name in ("query", "qseqid"))
        by_query: dict[str, list[tuple[int, str]]] = {}
        if has_header:
            reader = csv.DictReader(f, delimiter="\t")
            for pos, row in enumerate(reader):
                query = _first_present(row, ("query", "query_id", "q", "qseqid"))
                target = _first_present(row, ("target", "target_id", "t", "chain", "sseqid"))
                if not query or not target:
                    continue
                rank = _safe_int(row.get("rank"), pos)
                by_query.setdefault(query, []).append((rank, target))
        else:
            reader = csv.reader(f, delimiter="\t")
            for pos, row in enumerate(reader):
                if len(row) < 2:
                    continue
                by_query.setdefault(_normalise_id(row[0]), []).append((pos, _normalise_id(row[1])))
    return {
        query: [target for _, target in sorted(rows, key=lambda x: x[0])]
        for query, rows in by_query.items()
    }


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in row and _normalise_id(row[key]):
            return _normalise_id(row[key])
    return ""


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _score_from_row(row: dict[str, Any]) -> float | None:
    for key in ("score", "tm", "tm_mean", "tmscore", "tm_score"):
        if key in row:
            return _safe_float(row[key])
    if "tm_query" in row and "tm_target" in row:
        q = _safe_float(row["tm_query"])
        t = _safe_float(row["tm_target"])
        if q is not None and t is not None:
            return 0.5 * (q + t)
    return None


def _safe_float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def load_pair_scores(path: Path) -> dict[tuple[str, str], float]:
    """Load symmetric query-target coordinate-alignment scores."""
    rows: list[dict[str, Any]]
    if path.suffix.lower() in {".json", ".jsonl"}:
        data = _read_json_or_jsonl(path)
        if isinstance(data, dict) and "pairs" in data:
            data = data["pairs"]
        if isinstance(data, dict):
            rows = []
            for query, payload in data.items():
                if isinstance(payload, dict):
                    for target, score_payload in payload.items():
                        score = (
                            _score_from_row(score_payload)
                            if isinstance(score_payload, dict)
                            else _safe_float(score_payload)
                        )
                        rows.append({"query": query, "target": target, "score": score})
                else:
                    for target in _targets_from_payload(payload):
                        rows.append({"query": query, "target": target})
        elif isinstance(data, list):
            rows = [row for row in data if isinstance(row, dict)]
        else:
            raise ValueError(f"Unsupported score JSON structure in {path}")
    else:
        with path.open() as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)

    scores: dict[tuple[str, str], float] = {}
    for row in rows:
        query = _first_present(row, ("query", "query_id", "q", "qseqid", "i"))
        target = _first_present(row, ("target", "target_id", "t", "sseqid", "j"))
        score = _score_from_row(row)
        if not query or not target or score is None:
            continue
        scores[(query, target)] = score
        scores[(target, query)] = score
    return scores


def is_true_positive(query: str, target: str, labels: dict[str, str], neutral: set[str]) -> bool:
    q_label = labels.get(query, "")
    return bool(q_label and labels.get(target, "") == q_label and target != query and target not in neutral)


def total_true_positives(query: str, labels: dict[str, str], neutral: set[str]) -> int:
    q_label = labels.get(query, "")
    if not q_label:
        return 0
    return sum(
        1
        for target, label in labels.items()
        if target != query and target not in neutral and label == q_label
    )


def rerank_candidates(
    query: str,
    candidates: list[str],
    scores: dict[tuple[str, str], float],
    neutral: set[str],
    missing_score: str,
) -> tuple[list[str], int]:
    """Remove neutral/self hits and rerank candidates by US-align score."""
    scored: list[tuple[float, int, str]] = []
    missing = 0
    for rank, target in enumerate(candidates):
        if target == query or target in neutral:
            continue
        score = scores.get((query, target))
        if score is None:
            missing += 1
            if missing_score == "drop":
                continue
            if missing_score == "error":
                raise KeyError(f"Missing US-align score for {query!r} -> {target!r}")
            score = float("-inf")
        scored.append((score, rank, target))
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    return [target for _, _, target in scored], missing


def evaluate(
    *,
    query_ids: list[str],
    candidates: dict[str, list[str]],
    scores: dict[tuple[str, str], float],
    labels: dict[str, str],
    neutral_by_query: dict[str, set[str]],
    top_k: int,
    candidate_depth: int | None,
    missing_score: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    per_query: dict[str, dict[str, Any]] = {}
    for query in query_ids:
        neutral = neutral_by_query.get(query, set())
        cand = candidates.get(query, [])
        if candidate_depth is not None:
            cand = cand[:candidate_depth]
        pool_tp_ranks = [
            rank
            for rank, target in enumerate(cand, start=1)
            if is_true_positive(query, target, labels, neutral)
        ]
        ranked, missing = rerank_candidates(query, cand, scores, neutral, missing_score)
        total_tp = total_true_positives(query, labels, neutral)
        per_query[query] = {
            "recall1": bool(ranked and is_true_positive(query, ranked[0], labels, neutral)),
            "top10": any(is_true_positive(query, t, labels, neutral) for t in ranked[:top_k]),
            "pool_has_tp": bool(pool_tp_ranks),
            "pool_first_tp_rank": min(pool_tp_ranks) if pool_tp_ranks else None,
            "total_tp": total_tp,
            "n_candidates": len(cand),
            "n_ranked": len(ranked),
            "missing_scores": missing,
            "top_target": ranked[0] if ranked else None,
        }

    summary = summarise(per_query)
    return per_query, summary


def summarise(per_query: dict[str, dict[str, Any]]) -> dict[str, Any]:
    n = len(per_query)
    recall1 = sum(1 for x in per_query.values() if x["recall1"])
    top10 = sum(1 for x in per_query.values() if x["top10"])
    pool = sum(1 for x in per_query.values() if x["pool_has_tp"])
    ranks = [x["pool_first_tp_rank"] for x in per_query.values() if x["pool_first_tp_rank"] is not None]
    return {
        "n_queries": n,
        "recall1": recall1 / n if n else 0.0,
        "recall1_count": recall1,
        "top10": top10 / n if n else 0.0,
        "top10_count": top10,
        "pool_coverage": pool / n if n else 0.0,
        "pool_coverage_count": pool,
        "pool_first_tp_median_rank": median(ranks) if ranks else None,
        "missing_scores": sum(x["missing_scores"] for x in per_query.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter-state", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--usalign-scores", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--label-column", default="rfam_acc")
    parser.add_argument("--method-name", default="RFoldseek-TM")
    parser.add_argument("--query-set", default="evalq", help="filter_state query list to evaluate")
    parser.add_argument("--blind-query-set", default="blind10", help="optional blind-set list to summarise")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-depth", type=int, default=None)
    parser.add_argument(
        "--missing-score",
        choices=("error", "drop", "last"),
        default="error",
        help="How to handle a candidate without a US-align score.",
    )
    args = parser.parse_args()

    filter_state = json.loads(args.filter_state.read_text())
    if args.query_set not in filter_state:
        raise KeyError(f"{args.query_set!r} was not found in {args.filter_state}")

    query_ids = [_normalise_id(x) for x in filter_state[args.query_set]]
    blind_query_ids = [
        _normalise_id(x)
        for x in filter_state.get(args.blind_query_set, [])
        if _normalise_id(x) in set(query_ids)
    ]
    neutral_by_query = {
        _normalise_id(query): {_normalise_id(target) for target in targets}
        for query, targets in filter_state.get("neutral", {}).items()
    }

    labels = load_labels(args.labels, args.label_column)
    candidates = load_candidates(args.candidates)
    scores = load_pair_scores(args.usalign_scores)

    per_query, summary = evaluate(
        query_ids=query_ids,
        candidates=candidates,
        scores=scores,
        labels=labels,
        neutral_by_query=neutral_by_query,
        top_k=args.top_k,
        candidate_depth=args.candidate_depth,
        missing_score=args.missing_score,
    )
    blind_per_query = {q: per_query[q] for q in blind_query_ids if q in per_query}
    blind_summary = summarise(blind_per_query)

    out = {
        "method": args.method_name,
        "definition": (
            "RFoldseek token prefilter candidates reranked by US-align, using "
            "the mean of query- and target-normalised TM-scores."
        ),
        "candidate_depth": args.candidate_depth,
        "query_set": args.query_set,
        "blind_query_set": args.blind_query_set,
        "summary": summary,
        "blind_summary": blind_summary,
        "per_query": per_query,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True))

    def fmt(s: dict[str, Any]) -> str:
        n = s["n_queries"]
        return (
            f"Recall@1 {s['recall1_count']}/{n} = {s['recall1']:.3f}; "
            f"top-{args.top_k} {s['top10_count']}/{n} = {s['top10']:.3f}; "
            f"pool {s['pool_coverage_count']}/{n} = {s['pool_coverage']:.3f}; "
            f"median pool rank {s['pool_first_tp_median_rank']}"
        )

    print(f"{args.method_name} all:   {fmt(summary)}", flush=True)
    print(f"{args.method_name} blind: {fmt(blind_summary)}", flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
