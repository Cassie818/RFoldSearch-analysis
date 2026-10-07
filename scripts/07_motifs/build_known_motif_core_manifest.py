"""Build the strict BGSU motif-core manifest used by the motif analysis."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
MOTIF_DIR = REPO / "files" / "motifs"
COMMON_DIR = Path(__file__).resolve().parent / "word_controls"
sys.path.insert(0, str(COMMON_DIR))

from common import load_db_pickle, load_labels, load_sources  # noqa: E402


DEFAULT_TARGETS = {
    (18, 48, 62, 71): "HL_34789.6",
    (22, 22, 18, 32): "HL_35677.5",
    (44, 48, 62, 71): "HL_34789.6",
    (48, 24, 39, 39): "HL_33597.6",
    (62, 71, 66, 12): "HL_34789.6",
}
TOKEN_RE = re.compile(
    r'"?([0-9A-Za-z]+)\|(\d+)\|([^|]+)\|([^|]+)\|(-?\d+[A-Za-z]?)"?'
)
STEM_RE = re.compile(r"PDB_00([0-9A-Z]{6})_(\d+)_(.+)$", re.I)
FIELDS = [
    "kmer", "entry_id", "pdb_chain", "family", "clan",
    "token_start_1based", "raw_indices_1based", "author_residues",
    "sequence", "pair_sep_pattern", "pair_orient_pattern",
    "stack_sep_pattern", "n_external_pair", "n_internal_pair",
    "dssr_status", "dssr_json", "strict_clean", "strict_clean_reason",
]


def parse_stem(stem: str) -> tuple[str, str] | None:
    match = STEM_RE.fullmatch(stem)
    if not match:
        return None
    raw = match.group(1).upper()
    pdb_id = raw[2:] if raw.startswith("00") else raw
    return pdb_id, match.group(3)


def parse_bgsu(path: Path) -> dict[str, list[set[tuple[str, str, int]]]]:
    motifs: dict[str, list[set[tuple[str, str, int]]]] = {}
    current: str | None = None
    with path.open() as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith(">"):
                current = line[1:].strip()
                motifs[current] = []
                continue
            if not line.strip() or current is None:
                continue
            occurrence = {
                (pdb.upper(), chain, int(re.match(r"-?\d+", residue).group(0)))
                for pdb, _model, chain, _nt, residue in TOKEN_RE.findall(line)
            }
            if occurrence:
                motifs[current].append(occurrence)
    return motifs


def chain_motif_index(
    paths: list[Path], target_ids: set[str]
) -> dict[tuple[str, str], list[tuple[str, set[int]]]]:
    index: dict[tuple[str, str], list[tuple[str, set[int]]]] = defaultdict(list)
    for path in paths:
        for motif_id, occurrences in parse_bgsu(path).items():
            if motif_id not in target_ids:
                continue
            for occurrence in occurrences:
                for pdb_chain in {(pdb, chain) for pdb, chain, _ in occurrence}:
                    residues = {
                        residue for pdb, chain, residue in occurrence
                        if (pdb, chain) == pdb_chain
                    }
                    index[pdb_chain].append((motif_id, residues))
    return index


def read_targets(path: Path | None) -> dict[tuple[int, ...], str]:
    if path is None:
        return dict(DEFAULT_TARGETS)
    targets = {}
    with path.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            targets[tuple(map(int, row["kmer"].split("-")))] = row["motif_id"]
    if not targets:
        raise ValueError(f"no targets in {path}")
    return targets


def vector(source: dict, name: str) -> np.ndarray:
    return np.asarray(source[name], dtype=np.int64).reshape(-1)


def occurrence_context(
    entry: dict,
    source: dict,
    residue_numbers: list[int],
    start: int,
    length: int,
) -> dict[str, str | int]:
    mask = np.asarray(source["mask"], dtype=bool)
    raw_indices = np.flatnonzero(mask)
    if len(raw_indices) != len(entry["tokens"]):
        raise ValueError(f"source/token length mismatch for {entry['id']}")
    window = raw_indices[start : start + length]
    if len(window) != length or max(window) >= len(residue_numbers):
        raise ValueError(f"invalid token window for {entry['id']}:{start + 1}")

    pair_index = vector(source, "pair_idx")
    pair_orient = vector(source, "pair_orient")
    stack_index = vector(source, "stack_idx")
    core = set(map(int, window))
    pair_sep, orientations, stack_sep = [], [], []
    external_pairs: set[tuple[int, int]] = set()
    internal_pairs: set[tuple[int, int]] = set()
    for raw_index in window:
        partner = int(pair_index[raw_index])
        pair_sep.append("." if partner < 0 else str(partner - int(raw_index)))
        orientations.append({1: "cis", 2: "trans"}.get(int(pair_orient[raw_index]), "."))
        stack = int(stack_index[raw_index])
        stack_sep.append("." if stack < 0 else str(stack - int(raw_index)))
        if partner >= 0:
            edge = tuple(sorted((int(raw_index), partner)))
            (internal_pairs if partner in core else external_pairs).add(edge)

    sequence = str(source["seq"])
    return {
        "raw_indices_1based": ",".join(str(int(i) + 1) for i in window),
        "author_residues": ",".join(str(residue_numbers[int(i)]) for i in window),
        "sequence": "".join(sequence[int(i)] for i in window),
        "pair_sep_pattern": ",".join(pair_sep),
        "pair_orient_pattern": ",".join(orientations),
        "stack_sep_pattern": ",".join(stack_sep),
        "n_external_pair": len(external_pairs),
        "n_internal_pair": len(internal_pairs),
    }


def build_rows(
    db_path: Path,
    residue_map_path: Path,
    labels_path: Path,
    source_dir: Path,
    bgsu_paths: list[Path],
    targets: dict[tuple[int, ...], str],
    dssr_dir: Path,
) -> tuple[list[dict], dict[str, int]]:
    # The common loaders use repository defaults.  Keep the explicit arguments
    # in the public CLI, while loading alternative inputs locally when given.
    db = load_db_pickle(db_path)
    if labels_path == REPO / "Splits" / "labels.tsv":
        labels = load_labels()
    else:
        labels = {}
        with labels_path.open() as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                labels[row["entry_id"]] = (row.get("rfam_acc", ""), row.get("clan_acc", ""))
                labels[Path(row["entry_id"]).name] = labels[row["entry_id"]]
    if source_dir == REPO / "Splits" / "pdb_out":
        sources = load_sources()
    else:
        sources = {}
        for path in sorted(source_dir.glob("*.json")):
            with path.open() as handle:
                for line in handle:
                    source = json.loads(line)
                    sources[str(source["id"])] = source
                    sources[Path(source["id"]).name] = source

    residue_map = json.loads(residue_map_path.read_text())
    motif_index = chain_motif_index(bgsu_paths, set(targets.values()))
    rows: list[dict] = []
    counters = defaultdict(int)
    word_length = {len(word) for word in targets}
    if len(word_length) != 1:
        raise ValueError("all target words must have the same length")
    k = word_length.pop()

    for entry in db["entries"]:
        stem = Path(entry["id"]).stem
        pdb_chain = parse_stem(stem)
        source = sources.get(str(entry["id"])) or sources.get(Path(entry["id"]).name)
        if pdb_chain is None or source is None or stem not in residue_map:
            continue
        chain_motifs = motif_index.get(pdb_chain, [])
        if not chain_motifs:
            continue
        residues = list(map(int, residue_map[stem]))
        tokens = np.asarray(entry["tokens"], dtype=np.int64)
        raw_indices = np.flatnonzero(np.asarray(source["mask"], dtype=bool))
        if len(raw_indices) != len(tokens) or len(residues) != len(source["mask"]):
            counters["length_mismatch"] += 1
            continue

        for start in range(len(tokens) - k + 1):
            word = tuple(map(int, tokens[start : start + k]))
            motif_id = targets.get(word)
            if motif_id is None:
                continue
            author = [residues[int(i)] for i in raw_indices[start : start + k]]
            if len(set(author)) != k:
                counters["non_injective_author_mapping"] += 1
                continue
            if not any(motif == motif_id and set(author) <= motif_residues
                       for motif, motif_residues in chain_motifs):
                continue

            context = occurrence_context(entry, source, residues, start, k)
            family, clan = labels.get(str(entry["id"]), labels.get(Path(entry["id"]).name, ("", "")))
            dssr_json = dssr_dir / f"{stem}.json"
            rows.append({
                "kmer": "-".join(map(str, word)),
                "entry_id": str(entry["id"]),
                "pdb_chain": stem,
                "family": family,
                "clan": clan,
                "token_start_1based": start + 1,
                **context,
                "dssr_status": "cached" if dssr_json.exists() else "missing",
                "dssr_json": str(dssr_json),
                "strict_clean": 1,
                "strict_clean_reason": "",
            })
            counters["strict_hits"] += 1
    return rows, dict(counters)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows({name: row.get(name, "") for name in FIELDS} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=REPO / "files" / "models" / "db.pt")
    parser.add_argument("--labels", type=Path, default=REPO / "Splits" / "labels.tsv")
    parser.add_argument("--source-dir", type=Path, default=REPO / "Splits" / "pdb_out")
    parser.add_argument("--residue-map", type=Path, default=MOTIF_DIR / "chain_residue_map.json")
    parser.add_argument("--bgsu-hl", type=Path, default=MOTIF_DIR / "bgsu_hl_current_2026-08-03.csv")
    parser.add_argument("--bgsu-il", type=Path, default=MOTIF_DIR / "bgsu_il_current_2026-08-03.csv")
    parser.add_argument("--targets", type=Path, help="TSV with kmer and motif_id columns")
    parser.add_argument("--dssr-dir", type=Path, default=MOTIF_DIR / "dssr_known_motif_cores")
    parser.add_argument("--output", type=Path, default=MOTIF_DIR / "known_motif_core_matches_production_new_relabel_manifest.tsv")
    args = parser.parse_args()

    rows, counters = build_rows(
        args.db, args.residue_map, args.labels, args.source_dir,
        [args.bgsu_hl, args.bgsu_il], read_targets(args.targets), args.dssr_dir,
    )
    write_rows(args.output, rows)
    counts = defaultdict(int)
    for row in rows:
        counts[row["kmer"]] += 1
    print(f"wrote {args.output} ({len(rows)} rows)")
    print("counts: " + ", ".join(f"{word}={counts[word]}" for word in sorted(counts)))
    if counters.get("non_injective_author_mapping"):
        print(f"excluded non-injective residue maps: {counters['non_injective_author_mapping']}")


if __name__ == "__main__":
    main()
