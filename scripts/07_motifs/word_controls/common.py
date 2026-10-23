"""Shared utilities for the recurrent-word control analyses."""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import pickle
import random
import shlex
import sys
import types
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[3]
MOTIF_DIR = REPO / "files" / "motifs"
OUT_DIR = MOTIF_DIR / "word_controls"
SCRIPT_DIR = REPO / "scripts" / "07_motifs"
DB_PATH = REPO / "files" / "models" / "db.pt"
CANDIDATES_PATH = MOTIF_DIR / "kmer_candidates_production_new_k4_aa.tsv"
LABELS_PATH = REPO / "Splits" / "labels.tsv"
SOURCE_DIR = REPO / "Splits" / "pdb_out"
SEED = 260810
BACKBONE_SUGAR = ("P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'")
BASE_LOCATORS = {
    "A": ("N9", "C4", "C8"),
    "G": ("N9", "C4", "C8"),
    "C": ("N1", "C2", "C6"),
    "U": ("N1", "C2", "C6"),
}


def atom_name(value: str) -> str:
    return value.strip().strip('"').replace("*", "'")


def stem_parts(stem: str) -> tuple[str, int, str]:
    parts = stem.split("_")
    raw = parts[1][2:] if parts[1].startswith("00") else parts[1]
    return raw.upper(), int(parts[2]), "_".join(parts[3:])


def expected_atoms(base: str) -> tuple[str, ...]:
    return BACKBONE_SUGAR + BASE_LOCATORS[base]


def parse_atom_site(
    path: Path,
    chain: str,
    model: int,
    full_keys: bool = False,
) -> dict[object, tuple[str, dict[str, np.ndarray]]]:
    """Read standard heavy-atom coordinates from a PDB mmCIF atom_site loop."""
    lines = path.read_text(errors="replace").splitlines()
    residues: dict[object, tuple[str, dict[str, tuple[float, np.ndarray]]]] = {}
    i = 0
    while i < len(lines):
        if lines[i].strip() != "loop_":
            i += 1
            continue
        i += 1
        columns = []
        while i < len(lines) and lines[i].lstrip().startswith("_atom_site."):
            columns.append(lines[i].strip())
            i += 1
        if not columns:
            continue
        lookup = {name: index for index, name in enumerate(columns)}
        required = ["_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z"]
        if any(name not in lookup for name in required):
            continue
        while i < len(lines):
            stripped = lines[i].strip()
            if not stripped or stripped == "#" or stripped == "loop_" or stripped.startswith("_"):
                break
            values = shlex.split(stripped, posix=True)
            i += 1
            if len(values) < len(columns):
                continue
            get = lambda *names, default="": next((values[lookup[name]] for name in names if name in lookup), default)
            if get("_atom_site.group_PDB", default="ATOM") not in {"ATOM", "HETATM"}:
                continue
            if int(float(get("_atom_site.pdbx_PDB_model_num", default="1"))) != model:
                continue
            if get("_atom_site.auth_asym_id", "_atom_site.label_asym_id") != chain:
                continue
            try:
                residue_number = int(float(get("_atom_site.auth_seq_id", "_atom_site.label_seq_id")))
                coordinate = np.asarray([float(get(name)) for name in required], dtype=np.float32)
            except (ValueError, TypeError):
                continue
            if get("_atom_site.type_symbol").upper() in {"H", "D"}:
                continue
            alt = get("_atom_site.label_alt_id", default=".")
            if alt not in {".", "?", "A", "1"}:
                continue
            occupancy = float(get("_atom_site.occupancy", default="1") or 1)
            residue_name = get("_atom_site.auth_comp_id", "_atom_site.label_comp_id").upper()
            name = atom_name(get("_atom_site.auth_atom_id", "_atom_site.label_atom_id"))
            insertion = get("_atom_site.pdbx_PDB_ins_code", default="")
            insertion = "" if insertion in {"", ".", "?"} else insertion
            residue_key = (residue_number, insertion) if full_keys else residue_number
            current_name, atoms = residues.setdefault(residue_key, (residue_name, {}))
            if current_name == residue_name and (name not in atoms or occupancy > atoms[name][0]):
                atoms[name] = (occupancy, coordinate)
    return {
        number: (name, {atom: value[1] for atom, value in atoms.items()})
        for number, (name, atoms) in residues.items()
    }


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_metrics(path: Path, rows: list[tuple[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def load_db_pickle(path: Path = DB_PATH) -> dict:
    """Load the numpy-only torch archive without requiring a torch runtime."""
    # Archives written by NumPy 2 reference ``numpy._core``.  NumPy 1.x uses
    # the public ``numpy.core`` path; alias it so the same artifact remains
    # readable in the paper's older plotting environment.
    if not hasattr(np, "_core"):
        sys.modules.setdefault("numpy._core", np.core)
    with zipfile.ZipFile(path) as archive:
        payload = next(name for name in archive.namelist() if name.endswith("/data.pkl"))
        return pickle.loads(archive.read(payload))


def load_reference_rmsd_module():
    """Import the manuscript RMSD implementation, stubbing torch if absent.

    The analysis functions are imported rather than copied so Kabsch alignment
    and RMSD remain definitionally identical to cross_family_kmer_mining.py.
    """
    try:
        import torch  # noqa: F401
    except (ImportError, OSError):
        sys.modules.setdefault("torch", types.ModuleType("torch"))
    path = SCRIPT_DIR / "cross_family_kmer_mining.py"
    spec = importlib.util.spec_from_file_location("cross_family_kmer_mining_reference", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compact_coords(entry: dict) -> np.ndarray:
    coords = np.asarray(entry["coords"], dtype=np.float32)
    mask = np.asarray(entry.get("mask", []), dtype=bool)
    tokens = np.asarray(entry["tokens"], dtype=np.int32)
    if mask.size == len(coords) and int(mask.sum()) == len(tokens):
        coords = coords[mask]
    if len(coords) != len(tokens):
        raise ValueError(f"coordinate/token mismatch for {entry['id']}")
    return coords


def word_occurrences(entries: list[dict], k: int = 4) -> dict[tuple[int, ...], list[tuple[int, int]]]:
    index: dict[tuple[int, ...], list[tuple[int, int]]] = defaultdict(list)
    for entry_idx, entry in enumerate(entries):
        tokens = np.asarray(entry["tokens"], dtype=np.int32)
        for start in range(len(tokens) - k + 1):
            index[tuple(map(int, tokens[start : start + k]))].append((entry_idx, start))
    return index


def fragment(entry: dict, start: int, k: int = 4) -> np.ndarray:
    return compact_coords(entry)[start : start + k].reshape(-1, 3)


def all_pair_median(fragments: list[np.ndarray], rmsd_module) -> tuple[float, int]:
    if len(fragments) < 2:
        return float("nan"), 0
    values = rmsd_module.pairwise_rmsd(fragments)
    return float(np.median(values)), int(len(values))


def quantiles(values: list[float] | np.ndarray) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    return tuple(map(float, np.percentile(array, [10, 50, 90])))


def stratified_median_permutation_test(
    strata: dict[object, tuple[list[float], list[float]]],
    n_permutations: int,
    seed: int,
) -> dict[str, float | int | str]:
    """Unique-word permutation test for a candidate/control median difference.

    Candidate and control labels are permuted independently within each
    matching stratum while preserving the observed group sizes.  The test
    statistic is the candidate-count-weighted mean of the stratum-specific
    ``median(control) - median(candidate)`` differences.  Positive values mean
    that candidates have lower RMSD.  Every input value is used once per
    permutation; repeated descriptive control draws are deliberately excluded.
    """
    prepared = []
    for name, (candidate_values, control_values) in strata.items():
        candidate = np.asarray(candidate_values, dtype=float)
        control = np.asarray(control_values, dtype=float)
        candidate = candidate[np.isfinite(candidate)]
        control = control[np.isfinite(control)]
        if not len(candidate) or not len(control):
            raise ValueError(f"empty candidate or control group in stratum {name!r}")
        prepared.append((name, candidate, control))
    total_candidates = sum(len(candidate) for _name, candidate, _control in prepared)
    weights = [len(candidate) / total_candidates for _name, candidate, _control in prepared]

    def statistic(groups) -> float:
        return float(sum(
            weight * (float(np.median(control)) - float(np.median(candidate)))
            for weight, (_name, candidate, control) in zip(weights, groups)
        ))

    observed = statistic(prepared)
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=float)
    for repeat in range(n_permutations):
        permuted = []
        for name, candidate, control in prepared:
            pooled = np.concatenate((candidate, control)).copy()
            rng.shuffle(pooled)
            split = len(candidate)
            permuted.append((name, pooled[:split], pooled[split:]))
        null[repeat] = statistic(permuted)
    two_sided_p = (1 + int(np.sum(np.abs(null) >= abs(observed)))) / (n_permutations + 1)
    one_sided_p = (1 + int(np.sum(null >= observed))) / (n_permutations + 1)
    return {
        "statistic_control_minus_candidate_rmsd": observed,
        "two_sided_p": float(two_sided_p),
        "one_sided_p_candidates_lower": float(one_sided_p),
        "n_permutations": n_permutations,
        "n_candidate_words": total_candidates,
        "n_control_words": sum(len(control) for _name, _candidate, control in prepared),
        "n_strata": len(prepared),
        "method": "unique_word_stratified_permutation_two_sided",
    }


def occurrence_bin(n_occ: int) -> str:
    if n_occ <= 15:
        return "8-15"
    if n_occ <= 31:
        return "16-31"
    # No non-candidate word occurs >=64 times in this database, so the final
    # bin is kept open to retain a valid background pool for high-frequency
    # candidates.
    return ">=32"


def wilcoxon_paired(x: list[float], y: list[float]) -> tuple[float, str, int]:
    """Two-sided Wilcoxon signed-rank test with a dependency-free fallback."""
    pairs = [(float(a), float(b)) for a, b in zip(x, y) if np.isfinite(a) and np.isfinite(b)]
    differences = np.asarray([a - b for a, b in pairs], dtype=float)
    differences = differences[differences != 0]
    n = len(differences)
    if not n:
        return float("nan"), "undefined_no_nonzero_pairs", 0
    try:
        from scipy.stats import wilcoxon

        result = wilcoxon(differences, alternative="two-sided", zero_method="wilcox")
        return float(result.pvalue), "scipy.stats.wilcoxon_two_sided", n
    except ImportError:
        absolute = np.abs(differences)
        order = np.argsort(absolute, kind="stable")
        ranks = np.empty(n, dtype=float)
        sorted_abs = absolute[order]
        tie_sizes = []
        left = 0
        while left < n:
            right = left + 1
            while right < n and sorted_abs[right] == sorted_abs[left]:
                right += 1
            ranks[order[left:right]] = (left + 1 + right) / 2.0
            tie_sizes.append(right - left)
            left = right
        w_plus = float(ranks[differences > 0].sum())
        mean = n * (n + 1) / 4.0
        variance = n * (n + 1) * (2 * n + 1) / 24.0
        variance -= sum(t**3 - t for t in tie_sizes if t > 1) / 48.0
        if variance <= 0:
            return float("nan"), "normal_approximation_tie_corrected", n
        z = (abs(w_plus - mean) - 0.5) / math.sqrt(variance)
        p = math.erfc(max(0.0, z) / math.sqrt(2.0))
        return float(min(1.0, p)), "normal_approximation_tie_corrected", n


def load_sources() -> dict[str, dict]:
    sources = {}
    for path in sorted(SOURCE_DIR.glob("*.json")):
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                identifier = str(row["id"])
                sources[identifier] = row
                sources[Path(identifier).name] = row
    return sources


def load_labels() -> dict[str, tuple[str, str]]:
    labels = {}
    for row in read_tsv(LABELS_PATH):
        identifier = row["entry_id"]
        value = (row.get("rfam_acc", ""), row.get("clan_acc", ""))
        labels[identifier] = value
        labels[Path(identifier).name] = value
    return labels


def word_string(word: tuple[int, ...]) -> str:
    return "-".join(map(str, word))


def parse_word(value: str) -> tuple[int, ...]:
    return tuple(map(int, value.split("-")))


def seeded_rng(offset: int = 0) -> random.Random:
    return random.Random(SEED + offset)
