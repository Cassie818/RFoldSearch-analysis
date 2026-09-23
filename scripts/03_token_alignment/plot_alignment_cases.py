#!/usr/bin/env python3
"""Foldseek-style population plots and RNA alignment case series.

All correspondences are evaluated against external Rfam 3D seed columns.
The highlighted cases fix one query chain and span increasing nucleotide
identity within a single Rfam family.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from Bio.Align import PairwiseAligner


BLUE = "#79B7D9"
ORANGE = "#E6812B"
INK = "#20282D"
MUTED = "#6D7982"

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
                        default=root / "files/rebuilt_subsets_2305f07/rfam_3d_seed.gz")
    parser.add_argument("--pairs", type=Path,
                        default=(root / "files/ablation/"
                                 "rfam3d_reference_disjoint_independent_3dseed/"
                                 "rfam3d_reference_alignment.pairs.tsv"))
    parser.add_argument("--figures", type=Path, default=root / "figures",
                        help="writes alignment.png (overview) and cases.png (case series)")
    parser.add_argument("--line-width", type=int, default=92)
    return parser.parse_args()


def load_helper(root: Path):
    path = root / "scripts/03_token_alignment/alignment_helpers.py"
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


def case_title(case: dict) -> str:
    return (
        f"({case['label']}) {QUERY[0]} - {case['target']} "
        f"(sensitivity = {case['recall']:.2f}, precision = {case['precision']:.2f}, "
        f"F1 score = {case['f1']:.2f})"
    )


def block_text(case: dict, width: int) -> str:
    query_name, target_name = QUERY[0], case["target"]
    lines = [
        f"Rfam reference alignment: (RMSD={case['ref_rmsd']:.1f} A for "
        f"{case['n_ref']} nucleotides, seq. id. = {100 * case['seq_id']:.0f}%, "
        f"3Dr id. = {100 * case['tok_id']:.0f}%)",
    ]
    for q, t, m, q0, q1, t0, t1 in numbered_chunks(
            case["ref_q"], case["ref_t"], width,
            start_from_row(QUERY[1]), start_from_row(case["target_row"])):
        lines.extend([
            f"{query_name:<9} {q0:>3} {q} {q1:>3}",
            f"{'':13}{m}",
            f"{target_name:<9} {t0:>3} {t} {t1:>3}",
        ])
    lines.append("")
    lines.append(
        f"RFoldseek alignment: (RMSD={case['rf_rmsd']:.1f} A for "
        f"{case['n_rf']} nucleotides)")
    qnums, tnums = case["qnums"], case["tnums"]
    for start in range(0, len(case["rf_q"]), width):
        stop = start + width
        q, t = case["rf_q"][start:stop], case["rf_t"][start:stop]
        s, n = case["rf_s"][start:stop], case["rf_n"][start:stop]
        qv = [x for x in qnums[start:stop] if x is not None]
        tv = [x for x in tnums[start:stop] if x is not None]
        q0, q1 = (qv[0], qv[-1]) if qv else (0, 0)
        t0, t1 = (tv[0], tv[-1]) if tv else (0, 0)
        lines.extend([
            f"{query_name:<9} {q0:>3} {q} {q1:>3}",
            f"{'3Dr':<9} {'':>3} {s}",
            f"{'Seq':<9} {'':>3} {n}",
            f"{target_name:<9} {t0:>3} {t} {t1:>3}",
        ])
    return "\n".join(lines)


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
            "ref_q": ref_q, "ref_t": ref_t,
            "rf_q": rf_q, "rf_t": rf_t, "rf_s": rf_s, "rf_n": rf_n,
            "qnums": qnums, "tnums": tnums,
            "table": table,
        })

    by_family = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
    })

    def add_scatter(ax, x, y, xlabel, ylabel):
        ax.scatter(x, y, s=8, color=BLUE, alpha=0.52, edgecolors="none")
        ax.plot([0, 1], [0, 1], color="#777777", lw=0.8, zorder=0)
        ax.set(xlim=(0, 1.02), ylim=(0, 1.02), xlabel=xlabel, ylabel=ylabel)
        ax.set_xticks(np.arange(0.2, 1.01, 0.2))
        ax.set_yticks(np.arange(0.2, 1.01, 0.2))
        ax.tick_params(labelsize=7, length=3)
        ax.xaxis.label.set_size(7.5)
        ax.yaxis.label.set_size(7.5)
        ax.set_aspect("equal", adjustable="box")

    def add_case_points(ax_a, ax_b):
        for case in prepared_cases:
            table = case["table"]
            xa = float(table["token_identity"])
            ya = float(table["sequence_identity"])
            xb = float(table["rfoldseek_f1"])
            yb = float(table["usalign_f1"])
            for ax, x, y in ((ax_a, xa, ya), (ax_b, xb, yb)):
                ax.scatter([x], [y], marker="d", s=26, color=ORANGE,
                           edgecolor="black", linewidth=0.45, zorder=5)
                ax.annotate(f"({case['label']})", (x, y), xytext=(-3, 7),
                            textcoords="offset points", fontsize=7, ha="center")

    title_fontsize = 8.5
    body_fontsize = 6.9
    body_offset = 0.018
    body_linespacing = 1.08

    def add_case_block(fig, case, left, top):
        fig.text(left, top, case_title(case), va="top", ha="left",
                 fontsize=title_fontsize, fontweight="semibold", color="black")
        fig.text(left, top - body_offset, block_text(case, args.line_width),
                 va="top", ha="left", family="DejaVu Sans Mono",
                 fontsize=body_fontsize, linespacing=body_linespacing,
                 color="black")

    overview = plt.figure(figsize=(7.0, 3.35))
    ax_a = overview.add_axes([0.085, 0.16, 0.39, 0.73])
    ax_b = overview.add_axes([0.545, 0.16, 0.39, 0.73])
    add_scatter(
        ax_a,
        np.asarray([float(row["token_identity"]) for row in rows]),
        np.asarray([float(row["sequence_identity"]) for row in rows]),
        "3Dr sequence identity", "Nucleotide sequence identity")
    add_scatter(
        ax_b,
        np.asarray([float(row["rfoldseek_f1"]) for row in rows]),
        np.asarray([float(row["usalign_f1"]) for row in rows]),
        "RFoldseek alignment $F_1$ score", "US-align alignment $F_1$ score")
    add_case_points(ax_a, ax_b)
    ax_a.text(-0.18, 1.04, "A", transform=ax_a.transAxes, fontsize=11,
              fontweight="bold", va="top", ha="left")
    ax_b.text(-0.18, 1.04, "B", transform=ax_b.transAxes, fontsize=11,
              fontweight="bold", va="top", ha="left")

    case_page = plt.figure(figsize=(8.27, 11.69))
    page_height_points = case_page.get_figheight() * 72.0
    block_heights = []
    for case in prepared_cases:
        line_count = block_text(case, args.line_width).count("\n") + 1
        body_height = line_count * body_fontsize * body_linespacing / page_height_points
        block_heights.append(body_offset + body_height + 0.004)
    equal_gap = 0.025
    group_height = sum(block_heights) + equal_gap * 3.0
    case_top = 0.5 + group_height / 2.0
    for case, block_height in zip(prepared_cases, block_heights):
        add_case_block(case_page, case, 0.065, case_top)
        case_top -= block_height + equal_gap

    args.figures.mkdir(parents=True, exist_ok=True)
    overview.savefig(args.figures / "alignment.png", dpi=300,
                     bbox_inches="tight", facecolor="white")
    case_page.savefig(args.figures / "cases.png", dpi=240, facecolor="white")
    plt.close(overview)
    plt.close(case_page)
    print(f"wrote {args.figures / 'alignment.png'}")
    print(f"wrote {args.figures / 'cases.png'}")


if __name__ == "__main__":
    main()
