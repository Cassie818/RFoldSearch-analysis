"""Global exact-sequence-matched RMSD control for recurrent token words.

For each recurrent length-four RFoldseek token word, this script samples
non-target four-nucleotide windows with the same exact 4-nt RNA sequences as
the observed occurrences. It computes all-anchor Kabsch RMSD medians for the
sampled windows and summarizes the control distribution across all candidates.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


REPO = Path(__file__).resolve().parents[2]
SOURCE_REPO = Path("/Users/cassie/Desktop/RFoldSeek")
DB_PATH = REPO / "files" / "models" / "db.pt"
MOTIF_DIR = REPO / "files" / "motifs"
FIG_DIR = REPO / "figures" / "motifs"
LABELS_PATH = Path("/Users/cassie/Desktop/RFoldSeek/Splits/labels.tsv")


def read_labels(path: Path) -> dict[str, tuple[str, str]]:
    if not path.exists():
        return {}
    labels: dict[str, tuple[str, str]] = {}
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            chain_id = row.get("entry_id") or row.get("id") or row.get("chain") or row.get("pdb_chain")
            family = row.get("family") or row.get("rfam") or row.get("rfam_acc")
            clan = row.get("clan_acc") or row.get("clan") or row.get("rfam_clan") or ""
            if chain_id and family:
                labels[str(chain_id)] = (str(family), str(clan))
                labels[Path(str(chain_id)).stem] = (str(family), str(clan))
    return labels


def load_db(db_path: Path) -> tuple[list[dict], dict[str, dict]]:
    db = torch.load(db_path, map_location="cpu", weights_only=False)
    sources: dict[str, dict] = {}
    for data_path in db["data"]:
        path = Path(data_path)
        if not path.is_absolute():
            candidates = [REPO / path, SOURCE_REPO / path, Path("/Users/cassie/Desktop") / path]
            path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        with path.open() as handle:
            for line in handle:
                source = json.loads(line)
                sources[str(source["id"])] = source
                sources[Path(str(source["id"])).stem] = source
    return db["entries"], sources


def compact_coords(entry: dict) -> np.ndarray:
    coords = np.asarray(entry["coords"], dtype=np.float64)
    tokens = np.asarray(entry["tokens"], dtype=np.int32)
    mask = np.asarray(entry.get("mask", []), dtype=bool)
    if coords.shape[0] == mask.shape[0] and int(mask.sum()) == len(tokens):
        coords = coords[mask]
    if len(coords) != len(tokens):
        raise ValueError(f"coordinate/token length mismatch for {entry['id']}")
    return coords.reshape(len(tokens), -1, 3)


def source_raw_indices(source: dict, n_tokens: int) -> np.ndarray | None:
    mask = np.asarray(source.get("mask", []), dtype=bool)
    if mask.size:
        raw_indices = np.flatnonzero(mask)
        if len(raw_indices) == n_tokens:
            return raw_indices
    seq = str(source.get("seq", ""))
    if len(seq) == n_tokens:
        return np.arange(n_tokens, dtype=int)
    return None


def kabsch_rmsd(points: np.ndarray, reference: np.ndarray) -> float:
    pc = points - points.mean(0, keepdims=True)
    qc = reference - reference.mean(0, keepdims=True)
    h = pc.T @ qc
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    aligned = pc @ rotation.T
    return float(np.sqrt(np.mean(np.sum((aligned - qc) ** 2, axis=1))))


def kabsch_rmsd_batch(points: np.ndarray, references: np.ndarray) -> np.ndarray:
    pc = points - points.mean(axis=1, keepdims=True)
    qc = references - references.mean(axis=1, keepdims=True)
    h = np.einsum("bni,bnj->bij", pc, qc)
    u, _, vt = np.linalg.svd(h)
    det = np.linalg.det(np.einsum("bij,bjk->bik", np.swapaxes(vt, 1, 2), np.swapaxes(u, 1, 2)))
    signs = np.sign(det)
    signs[signs == 0] = 1.0
    diag = np.zeros((len(points), 3, 3), dtype=np.float64)
    diag[:, 0, 0] = 1.0
    diag[:, 1, 1] = 1.0
    diag[:, 2, 2] = signs
    rotation = np.einsum("bij,bjk,bkl->bil", np.swapaxes(vt, 1, 2), diag, np.swapaxes(u, 1, 2))
    aligned = np.einsum("bni,bji->bnj", pc, rotation)
    return np.sqrt(np.mean(np.sum((aligned - qc) ** 2, axis=2), axis=1))


def sampled_pair_median(
    fragments: list[np.ndarray],
    rng: random.Random,
    max_pairs: int,
) -> tuple[float, int]:
    n = len(fragments)
    if n < 2:
        return float("nan"), 0
    n_all = n * (n - 1) // 2
    if n_all <= max_pairs:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        seen: set[tuple[int, int]] = set()
        while len(seen) < max_pairs:
            i = rng.randrange(n)
            j = rng.randrange(n - 1)
            if j >= i:
                j += 1
            if i > j:
                i, j = j, i
            seen.add((i, j))
        pairs = list(seen)
    first = np.stack([fragments[i] for i, _j in pairs]).astype(np.float64, copy=False)
    second = np.stack([fragments[j] for _i, j in pairs]).astype(np.float64, copy=False)
    values = kabsch_rmsd_batch(first, second)
    return float(np.median(values)), len(values)


def scan_windows(
    entries: list[dict],
    sources: dict[str, dict],
    labels: dict[str, tuple[str, str]],
    target_kmers: set[str],
    k: int,
    exclude_clans: set[str] | None = None,
) -> tuple[dict[str, list[tuple[str, np.ndarray]]], dict[str, list[tuple[str, np.ndarray]]]]:
    pool_by_sequence: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
    observed_by_kmer: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
    exclude_clans = exclude_clans or set()
    for entry in entries:
        entry_id = str(entry["id"])
        stem = Path(entry_id).stem
        label = labels.get(entry_id) or labels.get(stem)
        if label and exclude_clans and label[1] in exclude_clans:
            continue
        source = sources.get(entry_id) or sources.get(stem)
        if source is None:
            continue
        seq = str(source.get("seq", "")).upper().replace("T", "U")
        tokens = np.asarray(entry["tokens"], dtype=np.int32)
        raw_indices = source_raw_indices(source, len(tokens))
        if raw_indices is None:
            continue
        coords = compact_coords(entry)
        has_family = label is not None
        for start in range(0, len(tokens) - k + 1):
            raw = raw_indices[start : start + k]
            if len(raw) != k or raw[-1] >= len(seq):
                continue
            sequence = "".join(seq[int(idx)] for idx in raw)
            if len(sequence) != k or any(base not in "ACGU" for base in sequence):
                continue
            kmer = "-".join(str(int(value)) for value in tokens[start : start + k])
            fragment = coords[start : start + k].reshape(-1, 3).astype(np.float32, copy=True)
            pool_by_sequence[sequence].append((kmer, fragment))
            if has_family and kmer in target_kmers:
                observed_by_kmer[kmer].append((sequence, fragment))
    return pool_by_sequence, observed_by_kmer


def summarize_controls(
    candidates: pd.DataFrame,
    pool_by_sequence: dict[str, list[tuple[str, np.ndarray]]],
    observed_by_kmer: dict[str, list[tuple[str, np.ndarray]]],
    n_reps: int,
    max_pairs: int,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for _, row in candidates.iterrows():
        kmer = str(row["kmer"])
        observed = observed_by_kmer.get(kmer, [])
        sequence_order = [sequence for sequence, _ in observed]
        unique_sequences = sorted(set(sequence_order))
        eligible_by_sequence = {
            sequence: [fragment for token_word, fragment in pool_by_sequence.get(sequence, []) if token_word != kmer]
            for sequence in unique_sequences
        }
        missing_sequences = [sequence for sequence in unique_sequences if not eligible_by_sequence[sequence]]
        if len(observed) < 2 or missing_sequences:
            rows.append({
                "kmer": kmer,
                "status": "skipped_missing_sequence_pool" if missing_sequences else "skipped_too_few_observed",
                "n_observed_scanned": len(observed),
                "n_unique_sequences": len(unique_sequences),
                "missing_sequences": ";".join(missing_sequences),
                "n_replicates": 0,
                "control_median_of_medians": np.nan,
                "control_p10_median_rmsd": np.nan,
                "control_p90_median_rmsd": np.nan,
                "control_min_median_rmsd": np.nan,
                "control_max_median_rmsd": np.nan,
                "n_replicates_le_candidate_cross_median": np.nan,
                "candidate_cross_med_rmsd": float(row["cross_med_rmsd"]),
                "candidate_n_occ": int(row["n_occ"]),
                "candidate_n_fam": int(row["n_fam"]),
                "candidate_n_clan": int(row["n_clan"]),
                "candidate_n_chain": int(row["n_chain"]),
            })
            continue
        medians = []
        n_pairs_used = []
        for _rep in range(n_reps):
            fragments = [rng.choice(eligible_by_sequence[sequence]) for sequence in sequence_order]
            median, n_pairs = sampled_pair_median(fragments, rng, max_pairs)
            medians.append(median)
            n_pairs_used.append(n_pairs)
        medians_array = np.asarray(medians, dtype=float)
        rows.append({
            "kmer": kmer,
            "status": "ok",
            "n_observed_scanned": len(observed),
            "n_unique_sequences": len(unique_sequences),
            "missing_sequences": "",
            "n_replicates": n_reps,
            "control_median_of_medians": float(np.median(medians_array)),
            "control_p10_median_rmsd": float(np.percentile(medians_array, 10)),
            "control_p90_median_rmsd": float(np.percentile(medians_array, 90)),
            "control_min_median_rmsd": float(np.min(medians_array)),
            "control_max_median_rmsd": float(np.max(medians_array)),
            "control_median_n_pairs": float(np.median(n_pairs_used)),
            "n_replicates_le_candidate_cross_median": int(np.sum(medians_array <= float(row["cross_med_rmsd"]))),
            "candidate_cross_med_rmsd": float(row["cross_med_rmsd"]),
            "candidate_n_occ": int(row["n_occ"]),
            "candidate_n_fam": int(row["n_fam"]),
            "candidate_n_clan": int(row["n_clan"]),
            "candidate_n_chain": int(row["n_chain"]),
        })
    return pd.DataFrame(rows)


def write_summary(path: Path, controls: pd.DataFrame, random_control: pd.DataFrame, n_reps: int, max_pairs: int, seed: int, k: int) -> None:
    ok = controls[controls["status"] == "ok"].copy()
    random_rmsd = random_control["rmsd"].dropna().astype(float).to_numpy()
    rows = [
        ("n_candidates_total", len(controls)),
        ("n_candidates_evaluated", len(ok)),
        ("n_candidates_skipped", int((controls["status"] != "ok").sum())),
        ("candidate_cross_median_rmsd_all", float(controls["candidate_cross_med_rmsd"].median())),
        ("candidate_cross_median_rmsd_evaluated", float(ok["candidate_cross_med_rmsd"].median())),
        ("length_matched_random_median_rmsd", float(np.median(random_rmsd))),
        ("global_sequence_matched_control_median_rmsd", float(ok["control_median_of_medians"].median())),
        ("global_sequence_matched_control_p10_rmsd", float(ok["control_median_of_medians"].quantile(0.10))),
        ("global_sequence_matched_control_p90_rmsd", float(ok["control_median_of_medians"].quantile(0.90))),
        ("fraction_candidates_below_own_sequence_matched_control", float((ok["candidate_cross_med_rmsd"] < ok["control_median_of_medians"]).mean())),
        ("median_sequence_matched_minus_candidate_rmsd", float((ok["control_median_of_medians"] - ok["candidate_cross_med_rmsd"]).median())),
        ("n_replicates_per_candidate", n_reps),
        ("max_pairs_per_replicate", max_pairs),
        ("random_seed", seed),
        ("sampling_rule", f"for each observed target occurrence, sample one non-target {k}-nt window with the same exact {k}-nt sequence"),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def plot_control(path: Path, controls: pd.DataFrame, random_control: pd.DataFrame) -> None:
    ok = controls[controls["status"] == "ok"].copy()
    observed = controls["candidate_cross_med_rmsd"].dropna().astype(float).to_numpy()
    length_random = random_control["rmsd"].dropna().astype(float).to_numpy()
    seq_random = ok["control_median_of_medians"].dropna().astype(float).to_numpy()
    colors = {
        "observed": "#7895a7",
        "length": "#d6d2ca",
        "sequence": "#b98b7f",
    }
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    bins = np.linspace(0, 4.5, 70)
    ax.hist(length_random, bins=bins, density=True, color=colors["length"], alpha=0.65, label=f"length-matched random, median {np.median(length_random):.2f} Å")
    ax.hist(seq_random, bins=bins, density=True, histtype="step", linewidth=2.1, color=colors["sequence"], label=f"sequence-matched non-target, median {np.median(seq_random):.2f} Å")
    ax.hist(observed, bins=bins, density=True, histtype="step", linewidth=2.1, color=colors["observed"], label=f"token words, median {np.median(observed):.2f} Å")
    for values, color in [(observed, colors["observed"]), (length_random, "#aaa39a"), (seq_random, colors["sequence"])]:
        ax.axvline(float(np.median(values)), color=color, linestyle="--", linewidth=1.5, alpha=0.95)
    ax.set_xlim(0, 4.5)
    ax.set_xlabel("cross-family median RMSD / control median RMSD (Å)")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=MOTIF_DIR / "kmer_candidates_production_new_k4_aa.tsv")
    parser.add_argument("--random_control", type=Path, default=MOTIF_DIR / "random_control_rmsd_production_new_k4_aa.tsv")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--out", type=Path, default=MOTIF_DIR / "global_sequence_matched_control_production_new_k4_aa.tsv")
    parser.add_argument("--summary", type=Path, default=MOTIF_DIR / "global_sequence_matched_control_production_new_k4_aa_summary.tsv")
    parser.add_argument("--figure", type=Path, default=FIG_DIR / "global_sequence_matched_control_fig4a.png")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--exclude_clans_pool", default="",
                        help="comma-separated Rfam clans to exclude from both "
                             "observed target windows and sequence-matched "
                             "non-target control pools")
    parser.add_argument("--n_reps", type=int, default=25)
    parser.add_argument("--max_pairs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=260810)
    args = parser.parse_args()

    candidates = pd.read_csv(args.candidates, sep="\t")
    random_control = pd.read_csv(args.random_control, sep="\t")
    labels = read_labels(args.labels)
    entries, sources = load_db(args.db)
    exclude_clans = {clan.strip() for clan in args.exclude_clans_pool.split(",") if clan.strip()}
    pool_by_sequence, observed_by_kmer = scan_windows(
        entries=entries,
        sources=sources,
        labels=labels,
        target_kmers=set(candidates["kmer"].astype(str)),
        k=args.k,
        exclude_clans=exclude_clans,
    )
    controls = summarize_controls(
        candidates=candidates,
        pool_by_sequence=pool_by_sequence,
        observed_by_kmer=observed_by_kmer,
        n_reps=args.n_reps,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    controls.to_csv(args.out, sep="\t", index=False)
    write_summary(args.summary, controls, random_control, args.n_reps, args.max_pairs, args.seed, args.k)
    plot_control(args.figure, controls, random_control)
    ok = controls[controls["status"] == "ok"]
    print(f"candidates total: {len(controls)}")
    print(f"candidates evaluated: {len(ok)}")
    print(f"observed median: {controls['candidate_cross_med_rmsd'].median():.3f} Å")
    print(f"sequence-matched control median: {ok['control_median_of_medians'].median():.3f} Å")
    print(f"wrote {args.out}")
    print(f"wrote {args.summary}")
    print(f"wrote {args.figure}")


if __name__ == "__main__":
    main()
