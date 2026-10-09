#!/usr/bin/env python3
"""Reconstruct the sequence baseline on the saved 313 Rfam-3D structure pairs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import types
from pathlib import Path

import numpy as np

from rfam3d_reference_alignment import (
    consensus_match_columns,
    contiguous_family_region,
    family_bootstrap,
    global_sequence_pairs,
    map_reference_to_structure,
    metrics,
    parse_stockholm,
    residue_columns,
    restrict_to_evaluable_positions,
    subset_pairs,
)


def load_backbone_extractor(upstream: Path):
    # Import the original extraction module without importing the upstream
    # package's unrelated database/prefilter modules.
    package_name = "rfoldsearch" if (upstream / "rfoldsearch").is_dir() else "rfoldseek"
    package = types.ModuleType(package_name)
    package.__path__ = [str(upstream / package_name)]
    sys.modules[package_name] = package
    backbone = __import__(f"{package_name}.backbone", fromlist=["*"])
    RNA_RESIDUE_NAMES = backbone.RNA_RESIDUE_NAMES
    extract_backbone_coords = backbone.extract_backbone_coords
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx

    def extract(path: Path):
        try:
            return extract_backbone_coords(str(path))
        except ValueError as exc:
            if "No RNA residues in selected chain" not in str(exc):
                raise
            # The PDBe author-chain filter can retain an RNA polymer whose
            # label chain ID differs from the original BGSU filename suffix.
            atoms = pdbx.get_structure(
                pdbx.CIFFile.read(str(path)), model=1, use_author_fields=False)
            starts = struc.get_residue_starts(atoms)
            counts = {}
            for index in starts:
                if str(atoms.res_name[index]).strip().upper() in RNA_RESIDUE_NAMES:
                    chain = str(atoms.chain_id[index])
                    counts[chain] = counts.get(chain, 0) + 1
            if not counts:
                raise
            selected = max(sorted(counts), key=counts.get)
            return extract_backbone_coords(str(path), chain_id=selected)

    return extract


def chain_path(directory: Path, chain: str) -> Path:
    pdb, asym = chain.split("_", 1)
    return directory / f"PDB_0000{pdb}_1_{asym}.cif"


def prepare_row(family: str, seed_row: str, chain: str, families: dict,
                sequences: dict[str, tuple[str, np.ndarray]]) -> dict:
    aligned = families[family]["sequences"][seed_row]
    sequence, mask = sequences[chain]
    mapping, quality = map_reference_to_structure(aligned, sequence)
    if quality["coverage"] < 0.70 or quality["identity"] < 0.70:
        raise ValueError(f"Weak reconstructed mapping for {family} {seed_row} {chain}: {quality}")
    columns = residue_columns(aligned)
    by_column = {columns[ri]: si for ri, si in mapping.items() if ri < len(columns)}
    valid_columns = {col for col, si in by_column.items() if mask[si]}
    return {"sequence": sequence, "by_column": by_column,
            "valid_columns": valid_columns, "quality": quality}


def reconstruct(original: dict, families: dict,
                sequences: dict[str, tuple[str, np.ndarray]]) -> dict:
    saved_rows = original["pairs"]
    prepared = {}
    for row in saved_rows:
        family = row["family"]
        for prefix in ("query", "target"):
            key = family, row[f"{prefix}_seed_row"], row[f"{prefix}_chain"]
            if key not in prepared:
                prepared[key] = prepare_row(*key, families, sequences)

    rows = []
    mismatches = []
    for row in saved_rows:
        family = row["family"]
        query = prepared[family, row["query_seed_row"], row["query_chain"]]
        target = prepared[family, row["target_seed_row"], row["target_chain"]]
        match_columns = consensus_match_columns(families[family]["rf_annotation"])
        columns = sorted(query["valid_columns"] & target["valid_columns"] & match_columns)
        reference = np.asarray([
            (query["by_column"][col], target["by_column"][col])
            for col in columns
        ], dtype=np.int64).reshape(-1, 2)
        query_region = contiguous_family_region(query["by_column"], len(query["sequence"]))
        target_region = contiguous_family_region(target["by_column"], len(target["sequence"]))
        identity = float(np.mean([
            query["sequence"][i] == target["sequence"][j]
            for i, j in reference
        ])) if len(reference) else 0.0
        checks = {
            "reference_pairs": len(reference),
            "usalign_query_region_length": len(query_region),
            "usalign_target_region_length": len(target_region),
        }
        issues = [f"{key}: saved={row[key]} reconstructed={value}"
                  for key, value in checks.items() if row[key] != value]
        if not np.isclose(identity, row["sequence_identity"], atol=1e-12, rtol=0):
            issues.append(f"sequence_identity: saved={row['sequence_identity']} "
                          f"reconstructed={identity}")
        if issues:
            mismatches.append({"family": family, "query_chain": row["query_chain"],
                               "target_chain": row["target_chain"], "issues": issues})

        local = global_sequence_pairs(
            "".join(query["sequence"][int(i)] for i in query_region),
            "".join(target["sequence"][int(i)] for i in target_region),
        )
        predicted = subset_pairs(local, query_region, target_region)
        evaluable = restrict_to_evaluable_positions(predicted, reference)
        precision, recall, f1 = metrics(evaluable, reference)
        _, _, tolerant_f1 = metrics(evaluable, reference, tolerance=1)
        rows.append({**row,
                     "sequence_baseline_reconstruction_verified": not issues,
                     "sequence_alignment_pairs": len(evaluable),
                     "sequence_alignment_pairs_raw": len(predicted),
                     "sequence_alignment_precision": precision,
                     "sequence_alignment_recall": recall,
                     "sequence_alignment_f1": f1,
                     "sequence_alignment_tolerant_f1": tolerant_f1})

    return {"rows": rows, "mismatches": mismatches,
            "n_saved_pairs": len(saved_rows),
            "n_strictly_verified_pairs": len(saved_rows) - len(mismatches),
            "n_reconstructed_pairs": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-json", type=Path, required=True)
    parser.add_argument("--rfam-seed", type=Path, required=True)
    parser.add_argument("--chain-dir", type=Path, required=True)
    parser.add_argument("--fallback-chain-dir", type=Path)
    parser.add_argument(
        "--primary-chain", action="append", default=[],
        help="When set, use --chain-dir only for these chain IDs; all others use fallback",
    )
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()

    original = json.loads(args.benchmark_json.read_text())
    families = parse_stockholm(args.rfam_seed)
    extractor = load_backbone_extractor(args.upstream)
    chains = sorted({row[key] for row in original["pairs"]
                     for key in ("query_chain", "target_chain")})
    sequences = {}
    sources = {}
    preferred = set(args.primary_chain)
    for number, chain in enumerate(chains, 1):
        if preferred and chain not in preferred:
            if args.fallback_chain_dir is None:
                raise ValueError("--primary-chain requires --fallback-chain-dir")
            path = chain_path(args.fallback_chain_dir, chain)
        else:
            path = chain_path(args.chain_dir, chain)
            if not path.is_file() and args.fallback_chain_dir is not None:
                path = chain_path(args.fallback_chain_dir, chain)
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            _, mask, sequence = extractor(path)
        except Exception as exc:
            raise RuntimeError(f"Could not parse reconstructed chain {chain}: {path}") from exc
        sequences[chain] = (sequence, np.asarray(mask, dtype=bool))
        sources[chain] = str(path)
        if number % 20 == 0:
            print(f"parsed {number}/{len(chains)} chains", flush=True)

    result = reconstruct(original, families, sequences)
    result["chain_sources"] = sources
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sequence_baseline_reconstruction_audit.json").write_text(
        json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2)
        + "\n")
    rows = result["rows"]
    verified = [row for row in rows if row["sequence_baseline_reconstruction_verified"]]
    summary = dict(original["summary"])
    summary["sequence_baseline_reconstruction"] = {
        "structure_source": "chain-level first-model mmCIF; exact paths in reconstruction audit",
        "primary_chains": sorted(preferred),
        "n_chains": len(chains),
        "n_pairs_verified": len(verified),
        "n_pairs_with_source_differences": len(rows) - len(verified),
        "verification": "reference-pair count, family-region lengths, and sequence identity",
        "alignment": "global affine-gap nucleotide alignment; match +2, mismatch -1, "
                     "gap open -3, gap extension -0.5; no Rfam constraints",
    }
    for key in ("sequence_alignment_f1", "sequence_alignment_tolerant_f1"):
        summary[f"{key}_strictly_verified"] = family_bootstrap(
            verified, key, args.bootstrap, args.seed)
        summary[f"{key}_all_reconstructed"] = family_bootstrap(
            rows, key, args.bootstrap, args.seed)
    for key, left, right in (
        ("rfoldsearch_minus_sequence_alignment_f1", "rfoldsearch_f1", "sequence_alignment_f1"),
        ("rfoldsearch_minus_sequence_alignment_tolerant_f1",
         "rfoldsearch_tolerant_f1", "sequence_alignment_tolerant_f1"),
    ):
        augmented = [{**row, key: row[left] - row[right]} for row in verified]
        summary[f"{key}_strictly_verified"] = family_bootstrap(
            augmented, key, args.bootstrap, args.seed)
    output = args.output_dir / "rfam3d_reference_alignment_with_sequence_baseline.json"
    output.write_text(json.dumps({"summary": summary, "pairs": rows,
                                  "failures": original.get("failures", [])}, indent=2) + "\n")
    with (args.output_dir / "rfam3d_reference_alignment_with_sequence_baseline.pairs.tsv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"n_pairs": len(rows), "n_strictly_verified": len(verified),
                      "sequence_alignment_f1_strictly_verified":
                          summary["sequence_alignment_f1_strictly_verified"],
                      "sequence_alignment_f1_all_reconstructed":
                          summary["sequence_alignment_f1_all_reconstructed"],
                      "rfoldsearch_minus_sequence_alignment_f1_strictly_verified":
                          summary["rfoldsearch_minus_sequence_alignment_f1_strictly_verified"]},
                     indent=2))


if __name__ == "__main__":
    main()
