"""Exact-sequence-matched RMSD control for token word 44-48-62-71.

This script reproduces the strict sequence-matched control used in Section 2.7.
For each observed occurrence of 44-48-62-71, it samples one non-target
four-nucleotide window with the same exact RNA sequence, computes all-pair
all-anchor Kabsch RMSD, and repeats this procedure for 2,000 replicates.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "files" / "models" / "db.pt"
MOTIF_DIR = REPO / "files" / "motifs"
KMER = "44-48-62-71"
KMER_TUPLE = tuple(int(value) for value in KMER.split("-"))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def compact_coords(entry: dict) -> np.ndarray:
    coords = np.asarray(entry["coords"], dtype=np.float64)
    mask = np.asarray(entry["mask"], dtype=bool)
    coords = coords[mask]
    if len(coords) != len(entry["tokens"]):
        raise ValueError(f"coordinate/token length mismatch for {entry['id']}")
    return coords


def kabsch_rmsd(points: np.ndarray, reference: np.ndarray) -> float:
    pc = points - points.mean(0, keepdims=True)
    qc = reference - reference.mean(0, keepdims=True)
    h = pc.T @ qc
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    aligned = pc @ rotation.T
    return float(np.sqrt(np.mean(np.sum((aligned - qc) ** 2, axis=1))))


def all_pair_rmsd(fragments: list[np.ndarray]) -> np.ndarray:
    values = [
        kabsch_rmsd(first, second)
        for first, second in itertools.combinations(fragments, 2)
    ]
    return np.asarray(values, dtype=float)


def load_db() -> tuple[list[dict], dict[str, dict]]:
    db = torch.load(DB_PATH, map_location="cpu", weights_only=False)
    sources = {}
    for data_path in db["data"]:
        with open(REPO / data_path if not Path(data_path).is_absolute() else data_path) as handle:
            for line in handle:
                source = json.loads(line)
                sources[str(source["id"])] = source
    return db["entries"], sources


def observed_fragments(
    entries_by_stem: dict[str, dict],
    context_rows: list[dict[str, str]],
) -> list[np.ndarray]:
    fragments = []
    for row in context_rows:
        entry = entries_by_stem[row["pdb_chain"]]
        start = int(row["token_start_1based"]) - 1
        fragments.append(compact_coords(entry)[start : start + 4].reshape(-1, 3))
    return fragments


def build_sequence_pool(
    entries: list[dict],
    sources: dict[str, dict],
    target_sequences: set[str],
) -> dict[str, list[tuple[str, np.ndarray]]]:
    pool = defaultdict(list)
    k = len(KMER_TUPLE)
    for entry in entries:
        source = sources.get(str(entry["id"]))
        if source is None:
            continue
        seq = str(source.get("seq", ""))
        mask = np.asarray(source.get("mask", entry["mask"]), dtype=bool)
        raw_indices = np.flatnonzero(mask)
        tokens = np.asarray(entry["tokens"], dtype=np.int32)
        coords = compact_coords(entry)
        if len(raw_indices) != len(tokens):
            continue
        stem = Path(entry["id"]).stem
        for start in range(len(tokens) - k + 1):
            token_window = tuple(int(value) for value in tokens[start : start + k])
            if token_window == KMER_TUPLE:
                continue
            raw_window = raw_indices[start : start + k]
            if len(raw_window) != k or raw_window[-1] >= len(seq):
                continue
            window_seq = "".join(seq[index] for index in raw_window)
            if window_seq not in target_sequences:
                continue
            fragment = coords[start : start + k].reshape(-1, 3)
            label = f"{window_seq}:{stem}:{start + 1}"
            pool[window_seq].append((label, fragment))
    return pool


def write_summary(
    path: Path,
    variant_counts: Counter[str],
    observed_values: np.ndarray,
    control_medians: list[float],
    n_reps: int,
    seed: int,
) -> None:
    observed_median = float(np.median(observed_values))
    n_le = int(np.sum(np.asarray(control_medians) <= observed_median))
    rows = [
        ("kmer", KMER),
        ("n_observed_occurrences", sum(variant_counts.values())),
        ("observed_sequence_counts", ";".join(
            f"{seq}:{count}" for seq, count in variant_counts.most_common()
        )),
        ("observed_all_pair_median_rmsd", observed_median),
        ("observed_all_pair_p90_rmsd", float(np.percentile(observed_values, 90))),
        ("observed_n_pairs", len(observed_values)),
        ("n_replicates", n_reps),
        ("control_median_of_medians", float(np.median(control_medians))),
        ("control_p10_median_rmsd", float(np.percentile(control_medians, 10))),
        ("control_p90_median_rmsd", float(np.percentile(control_medians, 90))),
        ("control_min_median_rmsd", float(np.min(control_medians))),
        ("control_max_median_rmsd", float(np.max(control_medians))),
        ("empirical_one_sided_p_le_observed", n_le / n_reps),
        ("n_replicates_le_observed", n_le),
        (
            "sampling_rule",
            "for each observed occurrence, sample one random non-target "
            "4-nt window with the exact same 4-nt sequence",
        ),
        ("random_seed", seed),
    ]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=Path,
                        default=MOTIF_DIR / f"candidate_context_core_{KMER}.tsv")
    parser.add_argument("--out", type=Path,
                        default=MOTIF_DIR / f"sequence_matched_control_{KMER}_exact_strict.tsv")
    parser.add_argument("--summary", type=Path,
                        default=MOTIF_DIR / f"sequence_matched_control_{KMER}_exact_strict_summary.tsv")
    parser.add_argument("--n_reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=260807)
    args = parser.parse_args()

    rows = read_tsv(args.context)
    variant_counts = Counter(row["sequence"] for row in rows)
    entries, sources = load_db()
    entries_by_stem = {Path(entry["id"]).stem: entry for entry in entries}
    observed_values = all_pair_rmsd(observed_fragments(entries_by_stem, rows))
    pool = build_sequence_pool(entries, sources, set(variant_counts))
    missing = [seq for seq in variant_counts if not pool.get(seq)]
    if missing:
        raise RuntimeError(f"no non-target sequence-matched windows for: {missing}")

    draw_order = []
    for seq, count in variant_counts.items():
        draw_order.extend([seq] * count)

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    control_medians = []
    with args.out.open("w", newline="") as handle:
        fieldnames = [
            "replicate", "median_rmsd", "p90_rmsd", "n_fragments", "n_pairs",
            "sampled_windows",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for rep in range(1, args.n_reps + 1):
            sampled = [rng.choice(pool[seq]) for seq in draw_order]
            labels = [item[0] for item in sampled]
            fragments = [item[1] for item in sampled]
            values = all_pair_rmsd(fragments)
            median = float(np.median(values))
            control_medians.append(median)
            writer.writerow({
                "replicate": rep,
                "median_rmsd": median,
                "p90_rmsd": float(np.percentile(values, 90)),
                "n_fragments": len(fragments),
                "n_pairs": len(values),
                "sampled_windows": ";".join(labels),
            })

    write_summary(
        args.summary, variant_counts, observed_values, control_medians,
        args.n_reps, args.seed,
    )
    print(f"wrote {args.out}")
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
