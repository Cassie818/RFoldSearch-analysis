"""Build six-residue instances formed by two offset RFoldSearch words."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
MOTIF_DIR = REPO / "files" / "motifs"
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "word_controls"))

from build_known_motif_core_manifest import (
    chain_motif_index,
    occurrence_context,
    parse_stem,
)
from common import load_db_pickle, load_labels, load_sources


LEFT = (44, 48, 62, 71)
RIGHT = (62, 71, 66, 12)
OFFSET = 2
FIELDS = [
    "entry_id", "pdb_chain", "family", "clan", "token_start_1based",
    "raw_indices_1based", "author_residues", "residue_keys", "sequence", "left_kmer",
    "right_kmer", "left_pair_sep_pattern", "left_pair_orient_pattern",
    "right_pair_sep_pattern", "right_pair_orient_pattern",
    "dssr_lw_signature", "dssr_exact_bgsu_hl_34789_6",
]


def read_dssr_rows(path: Path) -> dict[tuple[str, int, str], dict[str, str]]:
    index = {}
    with path.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = (row["pdb_chain"], int(row["token_start_1based"]), row["kmer"])
            index[key] = row
    return index


def join_signature(left: str, right: str) -> str:
    left_values = left.split(",")
    right_values = right.split(",")
    if len(left_values) != 4 or len(right_values) != 4:
        raise ValueError(f"expected two four-position signatures: {left!r}, {right!r}")
    return ",".join(left_values + right_values[2:])


def build_rows(
    db_path: Path,
    residue_map_path: Path,
    bgsu_hl_path: Path,
    dssr_instances_path: Path,
) -> list[dict]:
    db = load_db_pickle(db_path)
    sources = load_sources()
    labels = load_labels()
    residue_map = json.loads(residue_map_path.read_text())
    dssr = read_dssr_rows(dssr_instances_path)
    motif_index = chain_motif_index([bgsu_hl_path], {"HL_34789.6"})
    rows = []

    for entry in db["entries"]:
        stem = Path(entry["id"]).stem
        parsed = parse_stem(stem)
        source = sources.get(str(entry["id"])) or sources.get(Path(entry["id"]).name)
        if parsed is None or source is None or stem not in residue_map:
            continue
        tokens = np.asarray(entry["tokens"], dtype=np.int64)
        residue_keys = list(map(str, residue_map[stem]))
        mask = np.asarray(source["mask"], dtype=bool)
        raw_indices = np.flatnonzero(mask)
        if len(raw_indices) != len(tokens) or len(residue_keys) != len(mask):
            continue

        for start in range(len(tokens) - 5):
            if tuple(map(int, tokens[start : start + 4])) != LEFT:
                continue
            if tuple(map(int, tokens[start + OFFSET : start + OFFSET + 4])) != RIGHT:
                continue
            six_context = occurrence_context(entry, source, residue_keys, start, 6)
            left_context = occurrence_context(entry, source, residue_keys, start, 4)
            right_context = occurrence_context(entry, source, residue_keys, start + OFFSET, 4)
            family, clan = labels.get(str(entry["id"]), labels.get(Path(entry["id"]).name, ("", "")))

            window_keys = [residue_keys[int(i)] for i in raw_indices[start : start + 6]]
            exact_bgsu = len(set(window_keys)) == 6 and any(
                motif_id == "HL_34789.6" and set(window_keys) <= motif_residues
                for motif_id, motif_residues in motif_index.get(parsed, [])
            )
            left_row = dssr.get((stem, start + 1, "-".join(map(str, LEFT))))
            right_row = dssr.get((stem, start + OFFSET + 1, "-".join(map(str, RIGHT))))
            signature = ""
            if exact_bgsu and left_row is not None and right_row is not None:
                signature = join_signature(left_row["dssr_lw_pattern"], right_row["dssr_lw_pattern"])

            rows.append({
                "entry_id": Path(entry["id"]).name,
                "pdb_chain": stem,
                "family": family,
                "clan": clan,
                "token_start_1based": start + 1,
                "raw_indices_1based": six_context["raw_indices_1based"],
                "author_residues": six_context["author_residues"],
                "residue_keys": six_context["residue_keys"],
                "sequence": six_context["sequence"],
                "left_kmer": "-".join(map(str, LEFT)),
                "right_kmer": "-".join(map(str, RIGHT)),
                "left_pair_sep_pattern": left_context["pair_sep_pattern"],
                "left_pair_orient_pattern": left_context["pair_orient_pattern"],
                "right_pair_sep_pattern": right_context["pair_sep_pattern"],
                "right_pair_orient_pattern": right_context["pair_orient_pattern"],
                "dssr_lw_signature": signature,
                "dssr_exact_bgsu_hl_34789_6": int(bool(signature)),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=REPO / "files" / "models" / "db.pt")
    parser.add_argument("--residue-map", type=Path, default=MOTIF_DIR / "chain_residue_key_map.json")
    parser.add_argument("--bgsu-hl", type=Path, default=MOTIF_DIR / "bgsu_hl_current_2026-08-03.csv")
    parser.add_argument(
        "--dssr-instances", type=Path,
        default=MOTIF_DIR / "known_motif_core_dssr_pair_conservation_instances_production_new.tsv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=MOTIF_DIR / "overlapping_hairpin_44-48-62-71-66-12_instances.tsv",
    )
    args = parser.parse_args()

    rows = build_rows(args.db, args.residue_map, args.bgsu_hl, args.dssr_instances)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output} ({len(rows)} rows; "
          f"{sum(int(row['dssr_exact_bgsu_hl_34789_6']) for row in rows)} exact DSSR/BGSU matches)")


if __name__ == "__main__":
    main()
