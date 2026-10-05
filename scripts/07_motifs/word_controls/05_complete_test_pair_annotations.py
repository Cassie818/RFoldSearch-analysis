"""Recreate full test-set preprocessing records without replacing test.json."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
PREPROCESSING = REPO / "preprocessing"
sys.path.insert(0, str(PREPROCESSING))

from add_edges import find_base_pair_and_stack_idx
from extract_backbone import extract_backbone_coords


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=REPO / "Splits" / "pdb_out" / "test.json")
    parser.add_argument("--cif-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=REPO / "Splits" / "pdb_out" / "test_with_edges.json")
    parser.add_argument("--report", type=Path, default=REPO / "files" / "motifs" / "word_controls" / "test_with_edges_report.tsv")
    args = parser.parse_args()

    records = [json.loads(line) for line in args.input.open() if line.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    report_rows = []
    n_success = 0

    with temporary.open("w") as handle:
        for index, original in enumerate(records, start=1):
            original_id = str(original["id"])
            cif_path = args.cif_dir / Path(original_id).name
            try:
                if not cif_path.is_file():
                    raise FileNotFoundError(str(cif_path))
                edges = find_base_pair_and_stack_idx(str(cif_path))
                coords, mask, sequence = extract_backbone_coords(str(cif_path), chain_id=None, max_len=None)
                lengths = {
                    "coords": len(coords),
                    "mask": len(mask),
                    "sequence": len(sequence),
                    "pair_idx": len(edges["pair_idx"]),
                    "pair_orient": len(edges["pair_orient"]),
                    "stack_idx": len(edges["stack_idx"]),
                }
                if len(set(lengths.values())) != 1:
                    raise ValueError("length mismatch: " + ", ".join(f"{key}={value}" for key, value in lengths.items()))
                output = {
                    "id": Path(original_id).name,
                    "coords": coords.tolist(),
                    "mask": mask.tolist(),
                    "seq": sequence,
                    "pair_idx": edges["pair_idx"].tolist(),
                    "pair_orient": edges["pair_orient"].tolist(),
                    "stack_idx": edges["stack_idx"].tolist(),
                }
                handle.write(json.dumps(output) + "\n")
                report_rows.append({"id": Path(original_id).name, "status": "ok", "n_residues": len(coords), "reason": "."})
                n_success += 1
            except Exception as error:
                report_rows.append({
                    "id": Path(original_id).name,
                    "status": "failed",
                    "n_residues": "",
                    "reason": f"{type(error).__name__}: {error}",
                })
            if index % 25 == 0 or index == len(records):
                print(f"processed {index}/{len(records)} test chains; successful={n_success}", flush=True)

    temporary.replace(args.output)
    with args.report.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "status", "n_residues", "reason"], delimiter="\t")
        writer.writeheader()
        writer.writerows(report_rows)
    print(f"wrote {args.output}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
