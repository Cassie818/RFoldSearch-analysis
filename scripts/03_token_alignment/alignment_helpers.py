#!/usr/bin/env python3
"""Draw a Foldseek-style RFoldseek alignment example from experimental RNA.

The figure combines a C4'-trace superposition with the exact affine-gap
Smith--Waterman traceback.  Structural-token similarity and nucleotide
identity are shown on separate rows so that the two are not conflated.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import pickle
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from Bio.Align import PairwiseAligner


QUERY_COLOUR = "#159EAE"
TARGET_COLOUR = "#E58A2B"
MUTED = "#66727C"
LIGHT = "#D9DEE2"


def read_seed_rows(path: Path, family: str, wanted: set[str]) -> dict[str, str]:
    rows: dict[str, list[str]] = defaultdict(list)
    active = False
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith("#=GF AC"):
                active = line.split()[2] == family
            elif active and line == "//":
                break
            elif active and line and not line.startswith("#"):
                fields = line.split()
                if len(fields) >= 2 and fields[0] in wanted:
                    rows[fields[0]].append(fields[1])
    found = {key: "".join(parts) for key, parts in rows.items()}
    missing = wanted - set(found)
    if missing:
        raise ValueError(f"Rfam rows not found: {sorted(missing)}")
    return found


def clean_sequence(sequence: str) -> str:
    return "".join(base if base in "ACGU" else "N"
                   for base in sequence.upper().replace("T", "U"))


def ungapped(sequence: str) -> str:
    return clean_sequence("".join(char for char in sequence if char not in ".-"))


def reference_columns(aligned: str) -> list[int]:
    return [column for column, char in enumerate(aligned) if char not in ".-"]


def map_reference_to_structure(reference: str, structure: str) -> dict[int, int]:
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
    for ref_block, structure_block in zip(alignment.aligned[0], alignment.aligned[1]):
        rs, re = map(int, ref_block)
        ss, se = map(int, structure_block)
        for offset in range(min(re - rs, se - ss)):
            mapping[rs + offset] = ss + offset
    return mapping


def torch_pickle_payload(path: Path):
    """Load torch archives whose data.pkl contains only Python/NumPy data."""
    with zipfile.ZipFile(path) as archive:
        name = next(item for item in archive.namelist() if item.endswith("data.pkl"))
        return pickle.loads(archive.read(name))


def load_float_matrix(path: Path, shape=(80, 80)) -> np.ndarray:
    """Read the single contiguous FloatStorage in RFoldseek's matrix archive."""
    with zipfile.ZipFile(path) as archive:
        stores = [name for name in archive.namelist()
                  if re.search(r"/data/\d+$", name)]
        if len(stores) != 1:
            raise ValueError(f"Expected one tensor storage in {path}, found {stores}")
        values = np.frombuffer(archive.read(stores[0]), dtype="<f4").copy()
    if values.size != int(np.prod(shape)):
        raise ValueError(f"Unexpected substitution-matrix size: {values.size}")
    return values.reshape(shape)


def database_entry(database: dict, chain_name: str) -> dict:
    pattern = re.compile(r"PDB_0*([0-9A-Z]{4})_\d+_(.+)$")
    for entry in database["entries"]:
        stem = os.path.splitext(os.path.basename(str(entry.get("id", "")).upper()))[0]
        match = pattern.match(stem)
        if match and f"{match.group(1)}_{match.group(2)}" == chain_name.upper():
            return entry
    raise ValueError(f"{chain_name} not found in database")


def encode_family_region(encoded: dict, sequence: str, aligned_seed: str) -> dict:
    mapping = map_reference_to_structure(aligned_seed, sequence)
    columns = reference_columns(aligned_seed)
    structure_by_column = {
        columns[ri]: si for ri, si in mapping.items() if ri < len(columns)
    }
    # The shipped database stores one token and one coordinate row per valid
    # nucleotide, so the parsed valid-residue sequence indexes both directly.
    if len(encoded["tokens"]) != len(sequence):
        raise ValueError(
            f"Database/structure length mismatch: {len(encoded['tokens'])} vs {len(sequence)}")
    full_to_token = {index: index for index in range(len(sequence))}
    token_by_column = {
        column: full_to_token[full]
        for column, full in structure_by_column.items()
        if full in full_to_token
    }
    token_positions = np.asarray(sorted(token_by_column.values()), dtype=np.int64)
    token_to_full = token_positions
    coords = np.asarray(encoded["coords"], dtype=float)[token_to_full, 1]
    return {
        "tokens": np.asarray(encoded["tokens"], dtype=np.int64)[token_positions],
        "coords": coords,
        "bases": np.asarray([sequence[i] for i in token_to_full]),
        "residue_numbers": token_to_full + 1,
    }


def affine_local_traceback(query: np.ndarray, target: np.ndarray,
                           matrix: np.ndarray, gap_open: float = -2.0,
                           gap_extend: float = -0.5) -> tuple[float, list[tuple[str, int | None, int | None]]]:
    """Return the exact local affine-gap path as M, D, and I columns."""
    nq, nt = len(query), len(target)
    h = np.zeros((nq + 1, nt + 1), dtype=float)
    e = np.full((nq + 1, nt + 1), -np.inf)
    f = np.full((nq + 1, nt + 1), -np.inf)
    source = np.zeros((nq + 1, nt + 1), dtype=np.int8)
    e_open = np.zeros_like(source, dtype=bool)
    f_open = np.zeros_like(source, dtype=bool)
    best, end = 0.0, (0, 0)
    for i in range(1, nq + 1):
        for j in range(1, nt + 1):
            oe, xe = h[i, j - 1] + gap_open, e[i, j - 1] + gap_extend
            e[i, j], e_open[i, j] = (oe, True) if oe >= xe else (xe, False)
            of, xf = h[i - 1, j] + gap_open, f[i - 1, j] + gap_extend
            f[i, j], f_open[i, j] = (of, True) if of >= xf else (xf, False)
            values = (0.0,
                      h[i - 1, j - 1] + matrix[query[i - 1], target[j - 1]],
                      e[i, j], f[i, j])
            state = int(np.argmax(values))
            h[i, j], source[i, j] = values[state], state
            if h[i, j] > best:
                best, end = float(h[i, j]), (i, j)

    path: list[tuple[str, int | None, int | None]] = []
    i, j = end
    state = 0
    while i > 0 and j > 0:
        if state == 0:
            state = int(source[i, j])
            if state == 0:
                break
            if state == 1:
                path.append(("M", i - 1, j - 1))
                i -= 1
                j -= 1
                state = 0
            elif state == 2:
                state = 2
            else:
                state = 3
        elif state == 2:
            path.append(("I", None, j - 1))
            opened = e_open[i, j]
            j -= 1
            state = 0 if opened else 2
        else:
            path.append(("D", i - 1, None))
            opened = f_open[i, j]
            i -= 1
            state = 0 if opened else 3
    path.reverse()
    return best, path


def kabsch_target_to_query(query: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    q_center = query.mean(axis=0)
    t_center = target.mean(axis=0)
    q = query - q_center
    t = target - t_center
    u, _, vt = np.linalg.svd(t.T @ q)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    transformed = t @ rotation.T + q_center
    rmsd = float(np.sqrt(np.mean(np.sum((transformed - query) ** 2, axis=1))))
    return transformed, rmsd


def transform_all(points: np.ndarray, paired_target: np.ndarray,
                  paired_query: np.ndarray) -> np.ndarray:
    q_center = paired_query.mean(axis=0)
    t_center = paired_target.mean(axis=0)
    q = paired_query - q_center
    t = paired_target - t_center
    u, _, vt = np.linalg.svd(t.T @ q)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    return (points - t_center) @ rotation.T + q_center


def equal_3d_axes(ax, arrays: list[np.ndarray]) -> None:
    xyz = np.concatenate(arrays)
    centre = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
    radius = max((xyz.max(axis=0) - xyz.min(axis=0)).max() / 2, 1.0)
    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_zlim(centre[2] - radius, centre[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def pair_metadata(path: Path, query: str, target: str) -> dict[str, str]:
    with path.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["query_chain"] == query and row["target_chain"] == target:
                return row
    return {}


def alignment_strings(path, query, target, matrix):
    qbase, tbase, struct, seq = [], [], [], []
    qnums, tnums = [], []
    pairs = []
    for op, qi, ti in path:
        if op == "M":
            qb, tb = str(query["bases"][qi]), str(target["bases"][ti])
            qt, tt = int(query["tokens"][qi]), int(target["tokens"][ti])
            qbase.append(qb)
            tbase.append(tb)
            struct.append("|" if qt == tt else ("+" if matrix[qt, tt] > 0 else "."))
            seq.append("|" if qb == tb else " ")
            qnums.append(int(query["residue_numbers"][qi]))
            tnums.append(int(target["residue_numbers"][ti]))
            pairs.append((int(qi), int(ti)))
        elif op == "D":
            qbase.append(str(query["bases"][qi])); tbase.append("-")
            struct.append(" "); seq.append(" ")
            qnums.append(int(query["residue_numbers"][qi])); tnums.append(None)
        else:
            qbase.append("-"); tbase.append(str(target["bases"][ti]))
            struct.append(" "); seq.append(" ")
            qnums.append(None); tnums.append(int(target["residue_numbers"][ti]))
    return ("".join(qbase), "".join(tbase), "".join(struct), "".join(seq),
            qnums, tnums, np.asarray(pairs, dtype=np.int64))


def endpoint(numbers: list[int | None], first: bool) -> str:
    values = [n for n in numbers if n is not None]
    return str(values[0 if first else -1]) if values else ""
