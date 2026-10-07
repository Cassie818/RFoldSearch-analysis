#!/usr/bin/env python3
"""Alignment rows of the four RF00015 case pairs (query 5O9Z_4)."""

from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from Bio.Align import PairwiseAligner


BLUE = "#79B7D9"
ORANGE = "#E6812B"
INK = "#20282D"
MUTED = "#6D7982"
REFERENCE_COLOR = "#4D87AD"
TOKEN_MATCH_COLOR = "#318F91"
SEQUENCE_MATCH_COLOR = "#C47C35"

FAMILY = "RF00015"
QUERY = ("5O9Z_4", "URS0000200C70_9606/1-145", "PDB_00005O9Z_1_4.cif")
CASES = [
    ("A", "2N7M_X", "URS00008FED30_32630/1-92", "PDB_00002N7M_1_X.cif"),
    ("B", "5GAN_V", "URS00001143F5_4932/1-160", "PDB_00005GAN_1_V.cif"),
    ("C", "6AH0_I", "URS00003F07BD_9606/1-144", "PDB_00006AH0_1_I.cif"),
    ("D", "6QW6_4", "URS0000E60BB1_9606/2-146", "PDB_00006QW6_1_4.cif"),
]


def args_parser() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    import paths
    parser = argparse.ArgumentParser()
    parser.add_argument("--rfoldseek-repo", type=Path,
                        default=Path(paths.RFOLDSEEK_REPO))
    parser.add_argument("--structure-dir", type=Path,
                        default=Path(paths.RFAM3D_STRUCTURES))
    parser.add_argument("--rfam-seed", type=Path,
                        default=(root / "files/ablation/"
                                 "rfam3d_reference_disjoint_independent_3dseed/"
                                 "rfam_3d_seed.gz"))
    parser.add_argument("--pairs", type=Path,
                        default=(root / "files/ablation/"
                                 "rfam3d_reference_disjoint_independent_3dseed/"
                                 "rfam3d_reference_alignment_with_sequence_baseline.pairs.tsv"))
    parser.add_argument("--dump", type=Path,
                        default=(root / "files/ablation/"
                                 "rfam3d_reference_disjoint_independent_3dseed/"
                                 "alignment_cases.json"),
                        help="alignment rows of the four cases, read by plot_supp_alignment_cases.py")
    parser.add_argument("--dump-width", type=int, default=80)
    return parser.parse_args()


def load_helper(root: Path):
    path = root / "scripts/04_alignment_benchmark/alignment_helpers.py"
    spec = importlib.util.spec_from_file_location("alignment_example_helpers", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_family(path: Path, family: str) -> tuple[dict[str, str], str]:
    rows: dict[str, list[str]] = defaultdict(list)
    rf_annotation: list[str] = []
    active = False
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="latin-1") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith("#=GF AC"):
                active = line.split()[2] == family
            elif active and line == "//":
                break
            elif active and line.startswith("#=GC RF"):
                fields = line.split()
                if len(fields) >= 3:
                    rf_annotation.append(fields[2])
            elif active and line and not line.startswith("#"):
                fields = line.split()
                if len(fields) >= 2:
                    rows[fields[0]].append(fields[1])
    return ({key: "".join(value) for key, value in rows.items()},
            "".join(rf_annotation))


def clean_sequence(sequence: str) -> str:
    return "".join(base if base in "ACGU" else "N"
                   for base in sequence.upper().replace("T", "U"))


def ungapped(sequence: str) -> str:
    return clean_sequence("".join(char for char in sequence if char not in ".-"))


def reference_columns(aligned: str) -> list[int]:
    return [column for column, char in enumerate(aligned) if char not in ".-"]


def map_reference(reference: str, structure: str) -> dict[int, int]:
    reference = ungapped(reference)
    structure = clean_sequence(structure)
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -3.0
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(reference, structure)[0]
    mapping = {}
    for first, second in zip(alignment.aligned[0], alignment.aligned[1]):
        rs, re = map(int, first)
        ss, se = map(int, second)
        for offset in range(min(re - rs, se - ss)):
            mapping[rs + offset] = ss + offset
    return mapping


def prepare(row: str, path: Path, model, device, mean, std, active,
            structure_to_entry, encode_entry) -> dict:
    raw = structure_to_entry(str(path))
    encoded = encode_entry(model, raw, device, mean, std, active_blocks=active)
    if encoded is None:
        raise ValueError(f"Could not encode {path}")
    mapping = map_reference(row, raw["seq"])
    columns = reference_columns(row)
    structure_by_column = {
        columns[ri]: si for ri, si in mapping.items() if ri < len(columns)
    }
    valid_full = np.flatnonzero(np.asarray(encoded["mask"], bool))
    full_to_token = {int(full): index for index, full in enumerate(valid_full)}
    token_by_column = {
        column: full_to_token[position]
        for column, position in structure_by_column.items()
        if position in full_to_token
    }
    mapped_token_positions = sorted(set(token_by_column.values()))
    if not mapped_token_positions:
        raise ValueError(f"No mapped token positions for {path}")
    region_start, region_stop = mapped_token_positions[0], mapped_token_positions[-1] + 1
    positions = np.arange(region_start, region_stop, dtype=np.int64)
    full_positions = valid_full[positions]
    sequence = clean_sequence(raw["seq"])
    return {
        "aligned_seed": row,
        "tokens": np.asarray(encoded["tokens"], dtype=np.int64)[positions],
        "coords": np.asarray(encoded["coords"], dtype=float)[full_positions, 1],
        "bases": np.asarray([sequence[i] for i in full_positions]),
        "residue_numbers": full_positions + 1,
        "structure_by_column": structure_by_column,
        "token_by_column": token_by_column,
        "local_by_column": {
            column: position - region_start
            for column, position in token_by_column.items()
            if region_start <= position < region_stop
        },
        "full_coords": np.asarray(encoded["coords"], dtype=float)[:, 1],
    }


def pair_row(rows: list[dict[str, str]], query: str, target: str) -> dict[str, str]:
    for row in rows:
        if row["family"] != FAMILY:
            continue
        if row["query_chain"] == query and row["target_chain"] == target:
            return row
        if row["query_chain"] == target and row["target_chain"] == query:
            return row
    raise ValueError(f"Pair not found: {query}, {target}")


def rmsd(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 3:
        return float("nan")
    first = first - first.mean(axis=0)
    second = second - second.mean(axis=0)
    u, _, vt = np.linalg.svd(second.T @ first)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    fitted = second @ rotation.T
    return float(np.sqrt(np.mean(np.sum((fitted - first) ** 2, axis=1))))


def trim_reference(first: str, second: str) -> tuple[str, str]:
    first = first.upper().replace(".", "-")
    second = second.upper().replace(".", "-")
    keep = [i for i, (a, b) in enumerate(zip(first, second)) if a != "-" or b != "-"]
    lo, hi = min(keep), max(keep) + 1
    return first[lo:hi], second[lo:hi]


def start_from_row(row: str) -> int:
    match = re.search(r"/(\d+)-\d+$", row)
    return int(match.group(1)) if match else 1


def numbered_chunks(first: str, second: str, width: int,
                    first_start: int, second_start: int):
    qpos, tpos = first_start, second_start
    chunks = []
    for start in range(0, len(first), width):
        q = first[start:start + width]
        t = second[start:start + width]
        q0, t0 = qpos, tpos
        qpos += sum(char != "-" for char in q)
        tpos += sum(char != "-" for char in t)
        match = "".join("|" if a == b and a != "-" else " " for a, b in zip(q, t))
        chunks.append((q, t, match, q0, qpos - 1, t0, tpos - 1))
    return chunks



def case_rows(case: dict, width: int) -> dict:
    """Numbered alignment rows (query, 3Dr, Seq, target) of the reference and
    RFoldseek alignments, cut into lines of `width` columns."""
    reference = []
    for index, (q, t, m, q0, q1, t0, t1) in enumerate(numbered_chunks(
            case["ref_q"], case["ref_t"], width,
            start_from_row(QUERY[1]), start_from_row(case["target_row"]))):
        reference.append(dict(q=q, s=case["ref_struct"][index * width:(index + 1) * width], n=m, t=t,
                              q0=q0, q1=q1, t0=t0, t1=t1))
    rfoldseek = []
    for start in range(0, len(case["rf_q"]), width):
        stop = start + width
        qv = [x for x in case["qnums"][start:stop] if x is not None]
        tv = [x for x in case["tnums"][start:stop] if x is not None]
        rfoldseek.append(dict(q=case["rf_q"][start:stop], s=case["rf_s"][start:stop],
                              n=case["rf_n"][start:stop], t=case["rf_t"][start:stop],
                              q0=int(qv[0]) if qv else 0, q1=int(qv[-1]) if qv else 0,
                              t0=int(tv[0]) if tv else 0, t1=int(tv[-1]) if tv else 0))
    keys = ("label", "target", "target_row", "seq_id", "tok_id", "precision", "recall", "f1",
            "tol_f1", "n_ref", "n_rf", "ref_rmsd", "rf_rmsd")
    out = {k: (float(case[k]) if isinstance(case[k], (float, np.floating)) else
               int(case[k]) if isinstance(case[k], (int, np.integer)) else case[k]) for k in keys}
    return dict(out, reference=reference, rfoldseek=rfoldseek)


def main() -> None:
    args = args_parser()
    root = Path(__file__).resolve().parents[2]
    helper = load_helper(root)
    rows = list(csv.DictReader(args.pairs.open(), delimiter="\t"))
    seed, rf_annotation = parse_family(args.rfam_seed, FAMILY)
    match_columns = {index for index, char in enumerate(rf_annotation) if char.isupper()}

    sys.path.insert(0, str(args.rfoldseek_repo))
    from rfoldseek.database import structure_to_entry
    from rfoldseek.tokenizer import encode_entry, load_model

    device = torch.device("cpu")
    checkpoint_path = args.rfoldseek_repo / "rfoldseek/assets/desc_vqvae_best.pt"
    matrix_path = args.rfoldseek_repo / "rfoldseek/assets/sub_matrix.pt"
    model, checkpoint, _, _ = load_model(str(checkpoint_path), device)
    mean, std = checkpoint.get("norm_mean"), checkpoint.get("norm_std")
    active = checkpoint.get("active_blocks")
    matrix_obj = torch.load(matrix_path, map_location="cpu", weights_only=False)
    matrix = np.asarray(matrix_obj["matrix"] if isinstance(matrix_obj, dict)
                        else matrix_obj, dtype=float)

    query = prepare(seed[QUERY[1]], args.structure_dir / QUERY[2], model, device,
                    mean, std, active, structure_to_entry, encode_entry)
    prepared_cases = []
    for label, target_name, target_row, filename in CASES:
        target = prepare(seed[target_row], args.structure_dir / filename, model,
                         device, mean, std, active, structure_to_entry, encode_entry)
        table = pair_row(rows, QUERY[0], target_name)
        common = sorted(
            set(query["structure_by_column"])
            & set(target["structure_by_column"])
            & set(query["token_by_column"])
            & set(target["token_by_column"])
        )
        pair_query = query
        pair_target = target
        score, path = helper.affine_local_traceback(
            pair_query["tokens"], pair_target["tokens"], matrix)
        rf_q, rf_t, rf_s, rf_n, qnums, tnums, pairs = helper.alignment_strings(
            path, pair_query, pair_target, matrix)
        if not np.isclose(score, float(table["rfoldseek_score"]), atol=1e-5):
            raise ValueError(f"{label}: reconstructed score differs from benchmark")

        reference_columns = [col for col in common if col in match_columns]
        qref = np.asarray([query["structure_by_column"][col]
                           for col in reference_columns])
        tref = np.asarray([target["structure_by_column"][col]
                           for col in reference_columns])
        reference_start, reference_stop = reference_columns[0], reference_columns[-1] + 1
        ref_query = seed[QUERY[1]][reference_start:reference_stop]
        ref_target = seed[target_row][reference_start:reference_stop]
        ref_q, ref_t = trim_reference(ref_query, ref_target)
        keep = [i for i, (a, b) in enumerate(zip(ref_query, ref_target))
                if a not in ".-" or b not in ".-"]
        trim_start, trim_stop = min(keep), max(keep) + 1
        ref_struct_full = []
        for col in range(reference_start, reference_stop):
            qi = query["local_by_column"].get(col)
            ti = target["local_by_column"].get(col)
            if qi is None or ti is None:
                ref_struct_full.append(" ")
            else:
                qt = int(query["tokens"][qi])
                tt = int(target["tokens"][ti])
                ref_struct_full.append(
                    "|" if qt == tt else ("+" if matrix[qt, tt] > 0 else "."))
        ref_struct = "".join(ref_struct_full)[trim_start:trim_stop]
        prepared_cases.append({
            "label": label, "target": target_name, "target_row": target_row,
            "seq_id": float(table["sequence_identity"]),
            "tok_id": float(table["token_identity"]),
            "precision": float(table["rfoldseek_precision"]),
            "recall": float(table["rfoldseek_recall"]),
            "f1": float(table["rfoldseek_f1"]),
            "tol_f1": float(table["rfoldseek_tolerant_f1"]),
            "n_ref": len(reference_columns), "n_rf": len(pairs),
            "ref_rmsd": rmsd(query["full_coords"][qref],
                             target["full_coords"][tref]),
            "rf_rmsd": rmsd(pair_query["coords"][pairs[:, 0]],
                            pair_target["coords"][pairs[:, 1]]),
            "ref_q": ref_q, "ref_t": ref_t, "ref_struct": ref_struct,
            "rf_q": rf_q, "rf_t": rf_t, "rf_s": rf_s, "rf_n": rf_n,
            "qnums": qnums, "tnums": tnums,
            "table": table,
        })

    import json
    json.dump(dict(query=QUERY[0], width=args.dump_width,
                   cases=[case_rows(c, args.dump_width) for c in prepared_cases]),
              open(args.dump, "w"), indent=1)
    print(f"wrote {args.dump}")



if __name__ == "__main__":
    main()
