#!/usr/bin/env python3
"""Rfam-3D reference benchmark for RNA structural alphabets and alignments."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from Bio.Align import PairwiseAligner


MISSING = {"", "-", "."}


def parse_stockholm(path: Path) -> dict[str, dict]:
    families: dict[str, dict] = {}
    family = None
    sequences: dict[str, list[str]] = defaultdict(list)
    mappings: list[tuple[str, str]] = []
    rf_annotation: list[str] = []
    opener = (gzip.open(path, "rt", encoding="latin-1") if path.suffix == ".gz"
              else path.open(encoding="latin-1"))
    with opener as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith("#=GF AC"):
                family = line.split()[2]
            elif line == "//":
                if family:
                    families[family] = {
                        "sequences": {key: "".join(value) for key, value in sequences.items()},
                        "mappings": list(mappings),
                        "rf_annotation": "".join(rf_annotation),
                    }
                family = None
                sequences = defaultdict(list)
                mappings = []
                rf_annotation = []
            elif line.startswith("#=GR"):
                fields = line.split()
                if len(fields) >= 4 and fields[2].endswith("_SS"):
                    mappings.append((fields[1], fields[2][:-3].upper()))
            elif line.startswith("#=GC RF"):
                fields = line.split()
                if len(fields) >= 3:
                    rf_annotation.append(fields[2])
            elif line and not line.startswith("#"):
                fields = line.split()
                if len(fields) >= 2:
                    sequences[fields[0]].append(fields[1])
    return families


def discover_bgsu_chains(directories: list[Path]) -> dict[str, Path]:
    chains = {}
    pattern = re.compile(r"PDB_0*([0-9A-Z]{4})_\d+_(.+)$")
    for directory in directories:
        for path in sorted(directory.glob("*.cif")):
            match = pattern.match(path.stem.upper())
            if match:
                chains[f"{match.group(1)}_{match.group(2)}"] = path
    return chains


def clean_sequence(sequence: str) -> str:
    return "".join(base if base in "ACGU" else "N"
                   for base in sequence.upper().replace("T", "U"))


def ungapped(sequence: str) -> str:
    return clean_sequence("".join(char for char in sequence if char not in ".-"))


def residue_columns(aligned: str) -> list[int]:
    return [column for column, char in enumerate(aligned) if char not in ".-"]


def consensus_match_columns(rf_annotation: str) -> set[int]:
    """Return Rfam consensus match states; lowercase and dots are inserts."""
    return {column for column, char in enumerate(rf_annotation) if char.isupper()}


def pdb_id(value: str) -> str:
    match = re.search(r"PDB_0*([0-9A-Z]{4})_", value.upper())
    if match:
        return match.group(1)
    return value.upper().split("_", 1)[0].lstrip("0")


def load_excluded_pdbs(paths: list[Path]) -> set[str]:
    excluded = set()
    for path in paths:
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                excluded.add(pdb_id(str(record["id"])))
    return excluded


def map_reference_to_structure(reference: str, structure: str) -> tuple[dict[int, int], dict]:
    reference = ungapped(reference)
    structure = clean_sequence(structure)
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -3.0
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(reference, structure)[0]
    mapping: dict[int, int] = {}
    matches = 0
    for ref_block, struct_block in zip(alignment.aligned[0], alignment.aligned[1]):
        ref_start, ref_end = map(int, ref_block)
        struct_start, struct_end = map(int, struct_block)
        width = min(ref_end - ref_start, struct_end - struct_start)
        for offset in range(width):
            ri, si = ref_start + offset, struct_start + offset
            mapping[ri] = si
            matches += reference[ri] == structure[si]
    n = len(mapping)
    return mapping, {
        "reference_length": len(reference),
        "structure_length": len(structure),
        "mapped": n,
        "coverage": n / len(reference) if reference else 0.0,
        "identity": matches / n if n else 0.0,
    }


def global_sequence_pairs(query: str, target: str) -> np.ndarray:
    """Align two independently delimited RNA regions without Rfam constraints."""
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -3.0
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(clean_sequence(query), clean_sequence(target))[0]
    pairs = []
    for query_block, target_block in zip(*alignment.aligned):
        query_start, query_end = map(int, query_block)
        target_start, target_end = map(int, target_block)
        if query_end - query_start != target_end - target_start:
            raise ValueError("Unequal aligned sequence block widths")
        pairs.extend(zip(range(query_start, query_end),
                         range(target_start, target_end)))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def local_linear_traceback(first: np.ndarray, second: np.ndarray,
                           matrix: np.ndarray, gap: float) -> tuple[float, np.ndarray]:
    """Smith-Waterman traceback with the linear gap used by RiboSeek."""
    n, m = len(first), len(second)
    score = np.zeros((n + 1, m + 1), dtype=np.float64)
    pointer = np.zeros((n + 1, m + 1), dtype=np.int8)
    best = 0.0
    best_pos = (0, 0)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            values = (0.0,
                      score[i - 1, j - 1] + matrix[first[i - 1], second[j - 1]],
                      score[i - 1, j] + gap,
                      score[i, j - 1] + gap)
            state = int(np.argmax(values))
            score[i, j] = values[state]
            pointer[i, j] = state
            if score[i, j] > best:
                best = float(score[i, j])
                best_pos = (i, j)
    pairs = []
    i, j = best_pos
    while i and j and pointer[i, j]:
        state = int(pointer[i, j])
        if state == 1:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif state == 2:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return best, np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def metrics(predicted: np.ndarray, reference: np.ndarray,
            tolerance: int = 0) -> tuple[float, float, float]:
    pred = {tuple(map(int, pair)) for pair in predicted}
    ref = {tuple(map(int, pair)) for pair in reference}
    if tolerance == 0:
        shared = len(pred & ref)
        precision = shared / len(pred) if pred else 0.0
        recall = shared / len(ref) if ref else 0.0
    else:
        precision = (np.mean([
            any(abs(q - rq) <= tolerance and abs(t - rt) <= tolerance
                for rq, rt in ref) for q, t in pred
        ]) if pred else 0.0)
        recall = (np.mean([
            any(abs(q - pq) <= tolerance and abs(t - pt) <= tolerance
                for pq, pt in pred) for q, t in ref
        ]) if ref else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def restrict_to_evaluable_positions(predicted: np.ndarray,
                                    reference: np.ndarray) -> np.ndarray:
    """Treat predicted pairs outside the common match-state positions as neutral."""
    query_positions = {int(pair[0]) for pair in reference}
    target_positions = {int(pair[1]) for pair in reference}
    return np.asarray([
        (int(query), int(target))
        for query, target in predicted
        if int(query) in query_positions and int(target) in target_positions
    ], dtype=np.int64).reshape(-1, 2)


def subset_pairs(pairs: np.ndarray, first_positions: np.ndarray,
                 second_positions: np.ndarray) -> np.ndarray:
    return np.asarray([(first_positions[i], second_positions[j]) for i, j in pairs],
                      dtype=np.int64).reshape(-1, 2)


def contiguous_family_region(column_to_position: dict[int, int],
                             n_positions: int) -> np.ndarray:
    """Return an independently defined family span, including insert positions."""
    positions = sorted(set(map(int, column_to_position.values())))
    if not positions:
        return np.empty(0, dtype=np.int64)
    if positions[0] < 0 or positions[-1] >= n_positions:
        raise ValueError(
            f"mapped family position outside sequence: "
            f"[{positions[0]}, {positions[-1]}] versus length {n_positions}"
        )
    return np.arange(positions[0], positions[-1] + 1, dtype=np.int64)


def subset_entry(entry: dict, positions: np.ndarray) -> dict:
    """Restrict a coordinate entry to an independently defined family span."""
    return {
        "coords": np.asarray(entry["coords"])[positions],
        "mask": np.asarray(entry["mask"])[positions],
    }


def write_trace_pdb(entry: dict, path: Path) -> np.ndarray:
    """Write C4-prime trace coordinates using US-align's accepted RNA atom label."""
    valid = np.flatnonzero(np.asarray(entry["mask"], bool))
    c4 = np.asarray(entry["coords"], float)[valid, 1]
    with path.open("w") as handle:
        for serial, (x, y, z) in enumerate(c4, 1):
            handle.write(
                f"ATOM  {serial:5d}  C3'   G A{serial:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          C\n"
            )
        handle.write("END\n")
    return valid


def parse_usalign(text: str, query_map: np.ndarray,
                  target_map: np.ndarray) -> dict:
    lines = text.splitlines()
    marker = next(i for i, line in enumerate(lines)
                  if line.startswith('(":" denotes'))
    query_alignment, target_alignment = lines[marker + 1], lines[marker + 3]
    if len(query_alignment) != len(target_alignment):
        raise ValueError("US-align alignment lines have different lengths")
    query_position = target_position = 0
    pairs = []
    for query_char, target_char in zip(query_alignment, target_alignment):
        qi = query_map[query_position] if query_char != "-" else None
        tj = target_map[target_position] if target_char != "-" else None
        query_position += query_char != "-"
        target_position += target_char != "-"
        if qi is not None and tj is not None:
            pairs.append((int(qi), int(tj)))
    match = re.search(r"Aligned length=\s*(\d+), RMSD=\s*([0-9.]+)", text)
    scores = re.findall(r"TM-score=\s*([0-9.]+)", text)
    if match is None or len(scores) < 2:
        raise ValueError("Could not parse US-align summary")
    return {
        "pairs": pairs,
        "aligned_length_reported": int(match.group(1)),
        "rmsd_reported": float(match.group(2)),
        "tm_query_reported": float(scores[0]),
        "tm_target_reported": float(scores[1]),
    }


def run_usalign(binary: Path, query: dict, target: dict) -> dict:
    with tempfile.TemporaryDirectory(prefix="rfam3d_reference_") as directory:
        directory = Path(directory)
        query_map = write_trace_pdb(query, directory / "query.pdb")
        target_map = write_trace_pdb(target, directory / "target.pdb")
        result = subprocess.run(
            [str(binary), str(directory / "query.pdb"),
             str(directory / "target.pdb"), "-mol", "RNA"],
            check=True, capture_output=True, text=True,
        )
        return parse_usalign(result.stdout, query_map, target_map)


def family_bootstrap(rows: list[dict], key: str, reps: int, seed: int) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(float(row[key]))
    families = sorted(grouped)
    family_means = np.asarray([np.mean(grouped[fam]) for fam in families], float)
    rng = np.random.default_rng(seed)
    draws = np.asarray([
        np.mean(rng.choice(family_means, len(family_means), replace=True))
        for _ in range(reps)
    ])
    return {
        "n_families": len(families),
        "family_macro_mean": float(family_means.mean()),
        "family_block_bootstrap_95ci": [float(x) for x in np.percentile(draws, [2.5, 97.5])],
    }


def pooled_alignment_metrics(rows: list[dict], method: str) -> dict:
    predicted = sum(int(row[f"{method}_pairs"]) for row in rows)
    reference = sum(int(row["reference_pairs"]) for row in rows)
    correct = sum(
        float(row[f"{method}_precision"]) * int(row[f"{method}_pairs"])
        for row in rows
    )
    precision = correct / predicted if predicted else 0.0
    sensitivity = correct / reference if reference else 0.0
    f1 = (2.0 * precision * sensitivity / (precision + sensitivity)
          if precision + sensitivity else 0.0)
    return {
        "n_pairs": len(rows),
        "mean_pair_f1": float(np.mean([row[f"{method}_f1"] for row in rows])),
        "pooled_precision": float(precision),
        "pooled_sensitivity": float(sensitivity),
        "pooled_f1": float(f1),
        "empty_alignments": sum(int(row[f"{method}_pairs"]) == 0 for row in rows),
    }


def plot_results(rows: list[dict], output: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))
    color = "#4C9CCB"
    edge = "white"
    panels = [
        ("sequence_conservation", "token_conservation",
         "Nucleotide conservation above background",
         "3Dr conservation above background", "A"),
        ("rfoldsearch_f1", "usalign_f1", "RFoldSearch alignment $F_1$",
         "US-align alignment $F_1$", "B"),
        ("rfoldsearch_f1", "riboseek_f1", "RFoldSearch alignment $F_1$",
         "RiboSeek alignment $F_1$", "C"),
    ]
    for ax, (xkey, ykey, xlabel, ylabel, label) in zip(axes, panels):
        valid = [row for row in rows if np.isfinite(row.get(xkey, np.nan))
                 and np.isfinite(row.get(ykey, np.nan))]
        x = [row[xkey] for row in valid]
        y = [row[ykey] for row in valid]
        ax.scatter(x, y, s=26, color=color, alpha=0.55, edgecolor=edge, linewidth=0.35)
        ax.plot([0, 1], [0, 1], color="#777777", linewidth=1.2)
        ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel=xlabel, ylabel=ylabel)
        ax.set_aspect("equal", adjustable="box")
        ax.text(-0.14, 1.04, label, transform=ax.transAxes, fontsize=14,
                fontweight="bold", va="top")
        ax.grid(color="#dddddd", linewidth=0.5, alpha=0.55)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    import riboseek

    parser = argparse.ArgumentParser()
    parser.add_argument("--rfam-seed", type=Path, required=True)
    parser.add_argument("--structure-dir", type=Path, required=True, action="append")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--usalign", type=Path, required=True)
    parser.add_argument("--rfoldsearch-repo", "--rfoldseek-repo",
                        dest="rfoldsearch_repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-reference-pairs", type=int, default=20)
    parser.add_argument("--min-map-coverage", type=float, default=0.70)
    parser.add_argument("--min-map-identity", type=float, default=0.70)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument(
        "--exclude-split", type=Path, action="append", default=[],
        help="JSON-lines split whose PDB entries are excluded before pair construction",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.rfoldsearch_repo))
    paper_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(paper_root / "src" / "search"))
    from rfoldseek.database import structure_to_entry
    from rfoldseek.tokenizer import encode_entry, load_model
    from token_alignment import smith_waterman_traceback
    args.output_dir.mkdir(parents=True, exist_ok=True)
    families = parse_stockholm(args.rfam_seed)
    local_chains = discover_bgsu_chains(args.structure_dir)
    excluded_pdbs = load_excluded_pdbs(args.exclude_split)
    candidates = []
    for family, data in sorted(families.items()):
        by_row = defaultdict(list)
        for seed_row, pdb_chain in data["mappings"]:
            if pdb_chain in local_chains and seed_row in data["sequences"]:
                by_row[seed_row].append(pdb_chain)
        for seed_row, pdb_chains in sorted(by_row.items()):
            for pdb_chain in sorted(set(pdb_chains)):
                if pdb_id(pdb_chain) in excluded_pdbs:
                    continue
                candidates.append((family, seed_row, pdb_chain,
                                   data["sequences"][seed_row], data["rf_annotation"],
                                   local_chains[pdb_chain]))

    device = torch.device("cpu")
    model, checkpoint, _, _ = load_model(str(args.checkpoint), device)
    mean, std = checkpoint.get("norm_mean"), checkpoint.get("norm_std")
    active = checkpoint.get("active_blocks")
    matrix_obj = torch.load(args.matrix, map_location="cpu", weights_only=False)
    rfoldsearch_matrix = np.asarray(matrix_obj["matrix"] if isinstance(matrix_obj, dict)
                                  else matrix_obj, dtype=np.float64)
    alphabet = riboseek.Alphabet.from_pretrained("sa20")
    ribo_aligner = riboseek.NWAligner(alphabet.score_matrix, gap_penalty=-2.0)

    structures = {}
    failures = []
    for number, (family, seed_row, pdb_chain, aligned_sequence,
                 rf_annotation, path) in enumerate(candidates, 1):
        key = (family, seed_row)
        if key in structures:
            continue
        try:
            raw = structure_to_entry(str(path))
            encoded = encode_entry(model, raw, device, mean, std, active_blocks=active)
            if encoded is None:
                raise ValueError("RFoldSearch encoding returned None")
            encoded["seq"] = raw["seq"]
            mapping, quality = map_reference_to_structure(aligned_sequence, raw["seq"])
            if quality["coverage"] < args.min_map_coverage or quality["identity"] < args.min_map_identity:
                raise ValueError(f"weak Rfam-to-chain map: {quality}")
            columns = residue_columns(aligned_sequence)
            structure_by_column = {
                columns[ri]: si for ri, si in mapping.items() if ri < len(columns)
            }
            valid_full_positions = np.flatnonzero(np.asarray(encoded["mask"], bool))
            full_to_token = {
                int(full_position): token_position
                for token_position, full_position in enumerate(valid_full_positions)
            }
            token_by_column = {
                column: full_to_token[full_position]
                for column, full_position in structure_by_column.items()
                if full_position in full_to_token
            }
            ribo_data = riboseek.pdb_to_features(str(path), keep_modified=True)
            ribo_sequence = ribo_data.get("sequence", "")
            ribo_labels = alphabet.encode(ribo_data["features"])
            ribo_mapping, ribo_quality = map_reference_to_structure(aligned_sequence, ribo_sequence)
            ribo_by_column = {
                columns[ri]: si for ri, si in ribo_mapping.items()
                if ri < len(columns) and si < len(ribo_labels)
            }
            structures[key] = {
                "family": family, "seed_row": seed_row, "pdb_chain": pdb_chain,
                "path": str(path), "aligned_sequence": aligned_sequence,
                "match_columns": consensus_match_columns(rf_annotation),
                "entry": encoded, "sequence": clean_sequence(raw["seq"]),
                "structure_by_column": structure_by_column,
                "rfoldsearch_by_column": token_by_column, "map_quality": quality,
                "riboseek_labels": np.asarray(ribo_labels, np.int64),
                "riboseek_by_column": ribo_by_column,
                "riboseek_map_quality": ribo_quality,
            }
        except Exception as exc:
            failures.append({"family": family, "seed_row": seed_row,
                             "pdb_chain": pdb_chain, "path": str(path),
                             "error": f"{type(exc).__name__}: {exc}"})
        if number % 20 == 0:
            print(f"prepared {number}/{len(candidates)} candidates; "
                  f"retained {len(structures)} rows", flush=True)

    by_family = defaultdict(list)
    for item in structures.values():
        by_family[item["family"]].append(item)

    rows = []
    score_deltas = []
    total_pairs = sum(len(list(combinations(items, 2))) for items in by_family.values())
    done = 0
    for family, items in sorted(by_family.items()):
        for first, second in combinations(sorted(items, key=lambda x: x["seed_row"]), 2):
            done += 1
            evaluable_columns = sorted(
                set(first["structure_by_column"])
                & set(second["structure_by_column"])
                & set(first["rfoldsearch_by_column"])
                & set(second["rfoldsearch_by_column"])
                & set(first["riboseek_by_column"])
                & set(second["riboseek_by_column"])
            )
            common_match_columns = sorted(
                set(evaluable_columns) & first["match_columns"] & second["match_columns"]
            )
            if len(common_match_columns) < args.min_reference_pairs:
                continue

            # Rfam mappings define each chain's family boundaries independently.
            # All positions inside those boundaries, including insert states, are
            # retained for alignment. Shared Rfam columns are used only below to
            # construct the post hoc reference correspondence.
            first_pos = contiguous_family_region(
                first["rfoldsearch_by_column"], len(first["entry"]["tokens"]))
            second_pos = contiguous_family_region(
                second["rfoldsearch_by_column"], len(second["entry"]["tokens"]))
            first_tokens = np.asarray(first["entry"]["tokens"], np.int64)[first_pos]
            second_tokens = np.asarray(second["entry"]["tokens"], np.int64)[second_pos]
            rfs_score, rfs_local = smith_waterman_traceback(
                first_tokens, second_tokens, rfoldsearch_matrix, -2.0, -0.5)
            rfs_pairs = subset_pairs(rfs_local, first_pos, second_pos)

            first_ribo_pos = contiguous_family_region(
                first["riboseek_by_column"], len(first["riboseek_labels"]))
            second_ribo_pos = contiguous_family_region(
                second["riboseek_by_column"], len(second["riboseek_labels"]))
            ribo_first = first["riboseek_labels"][first_ribo_pos]
            ribo_second = second["riboseek_labels"][second_ribo_pos]
            ribo_raw_score, ribo_local = local_linear_traceback(
                ribo_first, ribo_second, alphabet.score_matrix, -2.0)
            compiled_score = ribo_aligner.align(ribo_first, ribo_second, local=True)
            ribo_score = (ribo_raw_score / min(len(ribo_first), len(ribo_second))
                          if len(ribo_first) and len(ribo_second) else 0.0)
            score_deltas.append(abs(ribo_score - compiled_score))
            ribo_pairs = subset_pairs(ribo_local, first_ribo_pos, second_ribo_pos)

            first_entry = first["entry"]
            second_entry = second["entry"]
            first_structure_pos = contiguous_family_region(
                first["structure_by_column"], len(first_entry["coords"]))
            second_structure_pos = contiguous_family_region(
                second["structure_by_column"], len(second_entry["coords"]))
            sequence_local = global_sequence_pairs(
                "".join(first["sequence"][int(i)] for i in first_structure_pos),
                "".join(second["sequence"][int(i)] for i in second_structure_pos),
            )
            sequence_pairs = subset_pairs(
                sequence_local, first_structure_pos, second_structure_pos)
            try:
                usa = run_usalign(
                    args.usalign,
                    subset_entry(first_entry, first_structure_pos),
                    subset_entry(second_entry, second_structure_pos),
                )
                usa_local = np.asarray(usa["pairs"], np.int64).reshape(-1, 2)
            except (OSError, subprocess.SubprocessError, ValueError):
                usa = {"pairs": [], "tm_query_reported": np.nan,
                       "tm_target_reported": np.nan}
                usa_local = np.empty((0, 2), dtype=np.int64)
            usa_pairs = subset_pairs(
                usa_local, first_structure_pos, second_structure_pos)

            common_reference = {
                "rfoldsearch": np.asarray([
                    (first["rfoldsearch_by_column"][col], second["rfoldsearch_by_column"][col])
                    for col in common_match_columns
                ], dtype=np.int64),
                "riboseek": np.asarray([
                    (first["riboseek_by_column"][col], second["riboseek_by_column"][col])
                    for col in common_match_columns
                ], dtype=np.int64),
                "usalign": np.asarray([
                    (first["structure_by_column"][col], second["structure_by_column"][col])
                    for col in common_match_columns
                ], dtype=np.int64),
            }
            rfs_evaluable = restrict_to_evaluable_positions(
                rfs_pairs, common_reference["rfoldsearch"])
            ribo_evaluable = restrict_to_evaluable_positions(
                ribo_pairs, common_reference["riboseek"])
            usa_evaluable = restrict_to_evaluable_positions(
                usa_pairs, common_reference["usalign"])
            sequence_evaluable = restrict_to_evaluable_positions(
                sequence_pairs, common_reference["usalign"])
            rfs_metric = metrics(rfs_evaluable, common_reference["rfoldsearch"])
            rfs_tolerant = metrics(
                rfs_evaluable, common_reference["rfoldsearch"], tolerance=1)
            usa_metric = metrics(usa_evaluable, common_reference["usalign"])
            usa_tolerant = metrics(
                usa_evaluable, common_reference["usalign"], tolerance=1)
            ribo_metric = metrics(ribo_evaluable, common_reference["riboseek"])
            ribo_tolerant = metrics(
                ribo_evaluable, common_reference["riboseek"], tolerance=1)
            sequence_metric = metrics(
                sequence_evaluable, common_reference["usalign"])
            sequence_tolerant = metrics(
                sequence_evaluable, common_reference["usalign"], tolerance=1)
            first_indices = [first["structure_by_column"][col]
                             for col in common_match_columns]
            second_indices = [second["structure_by_column"][col]
                              for col in common_match_columns]
            seq_identity = float(np.mean([
                first["sequence"][i] == second["sequence"][j]
                for i, j in zip(first_indices, second_indices)
            ]))
            token_columns = common_match_columns
            token_identity = float(np.mean([
                first["entry"]["tokens"][first["rfoldsearch_by_column"][column]] ==
                second["entry"]["tokens"][second["rfoldsearch_by_column"][column]]
                for column in token_columns
            ]))
            rows.append({
                "family": family,
                "query_seed_row": first["seed_row"],
                "target_seed_row": second["seed_row"],
                "query_chain": first["pdb_chain"],
                "target_chain": second["pdb_chain"],
                "reference_pairs": len(common_match_columns),
                "rfoldsearch_query_region_length": len(first_pos),
                "rfoldsearch_target_region_length": len(second_pos),
                "riboseek_query_region_length": len(first_ribo_pos),
                "riboseek_target_region_length": len(second_ribo_pos),
                "usalign_query_region_length": len(first_structure_pos),
                "usalign_target_region_length": len(second_structure_pos),
                "sequence_identity": seq_identity,
                "token_identity": token_identity,
                "sequence_alignment_pairs": len(sequence_evaluable),
                "sequence_alignment_pairs_raw": len(sequence_pairs),
                "sequence_alignment_precision": sequence_metric[0],
                "sequence_alignment_recall": sequence_metric[1],
                "sequence_alignment_f1": sequence_metric[2],
                "sequence_alignment_tolerant_f1": sequence_tolerant[2],
                "rfoldsearch_score": float(rfs_score),
                "rfoldsearch_pairs": len(rfs_evaluable),
                "rfoldsearch_pairs_raw": len(rfs_pairs),
                "rfoldsearch_precision": rfs_metric[0],
                "rfoldsearch_recall": rfs_metric[1],
                "rfoldsearch_f1": rfs_metric[2],
                "rfoldsearch_tolerant_f1": rfs_tolerant[2],
                "riboseek_score": float(ribo_score),
                "riboseek_pairs": len(ribo_evaluable),
                "riboseek_pairs_raw": len(ribo_pairs),
                "riboseek_precision": ribo_metric[0],
                "riboseek_recall": ribo_metric[1],
                "riboseek_f1": ribo_metric[2],
                "riboseek_tolerant_f1": ribo_tolerant[2],
                "usalign_pairs": len(usa_evaluable),
                "usalign_pairs_raw": len(usa_pairs),
                "usalign_precision": usa_metric[0],
                "usalign_recall": usa_metric[1],
                "usalign_f1": usa_metric[2],
                "usalign_tolerant_f1": usa_tolerant[2],
                "usalign_tm_query": float(usa["tm_query_reported"]),
                "usalign_tm_target": float(usa["tm_target_reported"]),
            })
            if done % 25 == 0:
                print(f"evaluated {done}/{total_pairs} candidate pairs; "
                      f"retained {len(rows)}", flush=True)

    if not rows:
        raise RuntimeError("No evaluable reference pairs")

    sequence_values = []
    token_values = []
    for item in structures.values():
        for position in set(item["structure_by_column"].values()):
            sequence_values.append(item["sequence"][position])
        for position in set(item["rfoldsearch_by_column"].values()):
            token_values.append(int(item["entry"]["tokens"][position]))
    sequence_counts = np.bincount(
        np.asarray(["ACGUN".index(value) for value in sequence_values], np.int64),
        minlength=5,
    ).astype(float)
    token_counts = np.bincount(np.asarray(token_values, np.int64), minlength=80).astype(float)
    sequence_frequency = sequence_counts / sequence_counts.sum()
    token_frequency = token_counts / token_counts.sum()
    sequence_background = float(np.sum(sequence_frequency ** 2))
    token_background = float(np.sum(token_frequency ** 2))
    for row in rows:
        row["sequence_conservation"] = float(
            (row["sequence_identity"] - sequence_background) / (1.0 - sequence_background))
        row["token_conservation"] = float(
            (row["token_identity"] - token_background) / (1.0 - token_background))

    difference_keys = {
        "token_minus_sequence_identity": lambda r: r["token_identity"] - r["sequence_identity"],
        "token_minus_sequence_conservation": (
            lambda r: r["token_conservation"] - r["sequence_conservation"]),
        "rfoldsearch_minus_usalign_f1": lambda r: r["rfoldsearch_f1"] - r["usalign_f1"],
        "rfoldsearch_minus_riboseek_f1": lambda r: r["rfoldsearch_f1"] - r["riboseek_f1"],
        "rfoldsearch_minus_usalign_tolerant_f1": (
            lambda r: r["rfoldsearch_tolerant_f1"] - r["usalign_tolerant_f1"]),
        "rfoldsearch_minus_riboseek_tolerant_f1": (
            lambda r: r["rfoldsearch_tolerant_f1"] - r["riboseek_tolerant_f1"]),
        "rfoldsearch_minus_sequence_alignment_f1": (
            lambda r: r["rfoldsearch_f1"] - r["sequence_alignment_f1"]),
        "rfoldsearch_minus_sequence_alignment_tolerant_f1": (
            lambda r: r["rfoldsearch_tolerant_f1"] - r["sequence_alignment_tolerant_f1"]),
    }
    augmented = []
    for row in rows:
        copy = dict(row)
        for key, function in difference_keys.items():
            copy[key] = function(row)
        augmented.append(copy)

    summary = {
        "protocol": {
            "reference": "Rfam.3d.seed alignment columns mapped to experimental PDB residues",
            "rfoldsearch_repository": str(args.rfoldsearch_repo),
            "rfoldsearch_commit": subprocess.run(
                ["git", "-C", str(args.rfoldsearch_repo), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True,
            ).stdout.strip(),
            "checkpoint": str(args.checkpoint),
            "substitution_matrix": str(args.matrix),
            "structure_sources": [str(path) for path in args.structure_dir],
            "excluded_split_files": [str(path) for path in args.exclude_split],
            "excluded_pdb_entries": len(excluded_pdbs),
            "training_exposure": {
                "rfoldsearch": (
                    "PDB entries in the supplied split files were excluded"
                    if args.exclude_split else
                    "no split exclusion was applied; training exposure was not assessed"
                ),
                "usalign": "not learned",
                "riboseek": "unverified for the bundled pretrained RS-20 alphabet; no training-chain manifest was available",
            },
            "one_structure_per_distinct_seed_row": True,
            "alignment_region": (
                "independently bounded contiguous Rfam family span; all internal "
                "residues, including inserts, retained"
            ),
            "reference_use": (
                "shared Rfam match columns used only for post hoc scoring, not "
                "as alignment constraints"
            ),
            "sequence_alignment": (
                "global affine-gap alignment of independently delimited structure-chain "
                "nucleotide sequences; match +2, mismatch -1, gap open -3, "
                "gap extension -0.5; no Rfam constraints"
            ),
            "reference_columns": "uppercase Rfam #=GC RF consensus match states",
            "insert_state_pairs": "neutral; excluded from both metric denominators",
            "shared_valid_positions": True,
            "evaluable_positions": (
                "intersection of positions mapped for every compared structural "
                "method in both structures"
            ),
            "tolerant_match": (
                "a pair matches if query and target indices are each within 1 nt "
                "of any reference pair; precision and recall are evaluated analogously"
            ),
            "missing_alignment_score": 0.0,
            "minimum_reference_pairs": args.min_reference_pairs,
            "minimum_mapping_coverage": args.min_map_coverage,
            "minimum_mapping_identity": args.min_map_identity,
            "aggregation": "pair values averaged within family; families weighted equally",
        },
        "n_seed_families": len(families),
        "n_local_seed_rows": len(structures),
        "n_evaluable_pairs": len(rows),
        "n_evaluable_families": len({row["family"] for row in rows}),
        "n_preparation_failures": len(failures),
        "n_pairs_below_sequence_identity": {
            str(cutoff): sum(row["sequence_identity"] < cutoff for row in rows)
            for cutoff in (0.4, 0.5, 0.6, 0.7, 0.8)
        },
        "max_riboseek_traceback_score_difference": max(score_deltas) if score_deltas else None,
        "sequence_random_match_background": sequence_background,
        "token_random_match_background": token_background,
        "sequence_states_observed": int(np.count_nonzero(sequence_counts)),
        "token_states_observed": int(np.count_nonzero(token_counts)),
        "token_effective_states_simpson": float(1.0 / token_background),
        "token_top_frequencies": sorted(
            ((int(index), float(frequency))
             for index, frequency in enumerate(token_frequency) if frequency > 0),
            key=lambda item: item[1], reverse=True,
        )[:10],
    }
    for key in ("sequence_identity", "token_identity", "sequence_conservation",
                "token_conservation", "rfoldsearch_f1", "riboseek_f1", "usalign_f1",
                "sequence_alignment_f1", "sequence_alignment_tolerant_f1",
                "rfoldsearch_tolerant_f1", "riboseek_tolerant_f1",
                "usalign_tolerant_f1", *difference_keys):
        summary[key] = family_bootstrap(augmented, key, args.bootstrap,
                                        args.seed + len(summary))
    summary["pair_level"] = {
        method: pooled_alignment_metrics(rows, method)
        for method in ("rfoldsearch", "riboseek", "usalign", "sequence_alignment")
    }

    output_json = args.output_dir / "rfam3d_reference_alignment.json"
    output_json.write_text(json.dumps({"summary": summary, "pairs": rows,
                                       "failures": failures}, indent=2) + "\n")
    output_tsv = args.output_dir / "rfam3d_reference_alignment.pairs.tsv"
    with output_tsv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    plot_results(rows, args.output_dir / "rfam3d_reference_alignment")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
