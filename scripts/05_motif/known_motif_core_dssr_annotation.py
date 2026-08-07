"""Extract DSSR interaction signatures for known motif-core token words.

The input manifest lists clean BGSU motif-core matches. For each occurrence,
this script reads the DSSR JSON output, extracts base-pair annotations for the
four token residues, and summarizes dominant position-specific signatures.

If a DSSR JSON file is missing and DSSR_BIN is set, the script runs DSSR with
JSON output on the source mmCIF file. Otherwise it fails with a clear message.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
MOTIF_DIR = REPO / "files" / "motifs_v2"
DEFAULT_MANIFEST = MOTIF_DIR / "known_motif_core_matches_strict_clean_manifest.tsv"
DEFAULT_INSTANCES = MOTIF_DIR / "known_motif_core_dssr_pair_conservation_instances_strict_clean.tsv"
DEFAULT_SUMMARY = MOTIF_DIR / "known_motif_core_dssr_pair_conservation_summary_strict_clean.tsv"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def residue_number(value: str) -> int | None:
    match = re.search(r"(-?\d+)(?!.*\d)", str(value))
    return int(match.group(1)) if match else None


def load_or_run_dssr(row: dict[str, str]) -> tuple[dict, str, Path]:
    json_path = Path(row["dssr_json"])
    if json_path.exists():
        return json.load(json_path.open()), "cached", json_path

    dssr_bin = os.environ.get("DSSR_BIN")
    if not dssr_bin:
        raise FileNotFoundError(
            f"missing DSSR JSON {json_path}; set DSSR_BIN to regenerate it"
        )

    json_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        dssr_bin,
        f"--input={row['entry_id']}",
        "--json",
        f"--output={json_path}",
    ]
    subprocess.run(command, check=True)
    return json.load(json_path.open()), "computed", json_path


def pair_lookup(dssr: dict) -> dict[int, dict]:
    lookup: dict[int, dict] = {}
    for pair in dssr.get("pairs", []):
        n1 = residue_number(pair.get("nt1", ""))
        n2 = residue_number(pair.get("nt2", ""))
        if n1 is None or n2 is None:
            continue
        record_1 = dict(pair)
        record_1["core_nt"] = pair.get("nt1", ".")
        record_1["partner_nt"] = pair.get("nt2", ".")
        record_1["partner_resnum"] = n2
        record_2 = dict(pair)
        record_2["core_nt"] = pair.get("nt2", ".")
        record_2["partner_nt"] = pair.get("nt1", ".")
        record_2["partner_resnum"] = n1
        lookup.setdefault(n1, record_1)
        lookup.setdefault(n2, record_2)
    return lookup


def stack_links(dssr: dict, core_residues: list[int]) -> str:
    core_to_pos = {residue: pos for pos, residue in enumerate(core_residues, start=1)}
    links = set()
    for stack in dssr.get("stacks", []):
        nts_long = str(stack.get("nts_long", ""))
        residues = [residue_number(item) for item in nts_long.split(",")]
        positions = sorted(
            core_to_pos[residue]
            for residue in residues
            if residue in core_to_pos
        )
        for left, right in zip(positions, positions[1:]):
            if right == left + 1:
                links.add(f"{left}{right}")
    return ",".join(sorted(links)) if links else "."


def annotate_row(row: dict[str, str]) -> dict[str, str]:
    dssr, status, json_path = load_or_run_dssr(row)
    core_residues = [
        residue_number(value)
        for value in row["author_residues"].split(",")
    ]
    if any(value is None for value in core_residues):
        raise ValueError(f"cannot parse author residues: {row['author_residues']}")
    core_residues = [int(value) for value in core_residues]
    pairs = pair_lookup(dssr)

    pair_mask = []
    partner_delta = []
    lw_pattern = []
    dssr_pattern = []
    name_pattern = []
    bp_pattern = []
    partner_nt_pattern = []
    pair_index_pattern = []
    local_pairs = []

    for pos, residue in enumerate(core_residues, start=1):
        pair = pairs.get(residue)
        if not pair:
            pair_mask.append("U")
            partner_delta.append(".")
            lw_pattern.append(".")
            dssr_pattern.append(".")
            name_pattern.append(".")
            bp_pattern.append(".")
            partner_nt_pattern.append(".")
            pair_index_pattern.append(".")
            continue

        pair_mask.append("P")
        partner_delta.append(str(pair["partner_resnum"] - residue))
        lw_pattern.append(str(pair.get("LW", ".")))
        dssr_pattern.append(str(pair.get("DSSR", ".")))
        name_pattern.append(str(pair.get("name", ".")))
        bp_pattern.append(str(pair.get("bp", ".")))
        partner_nt_pattern.append(str(pair.get("partner_nt", ".")))
        pair_index_pattern.append(str(pair.get("index", ".")))
        local_pairs.append(
            "pos{pos}:{core}-{partner}:{bp}:{lw}:{dssr}:{name}".format(
                pos=pos,
                core=pair.get("core_nt", "."),
                partner=pair.get("partner_nt", "."),
                bp=pair.get("bp", "."),
                lw=pair.get("LW", "."),
                dssr=pair.get("DSSR", "."),
                name=pair.get("name", "."),
            )
        )

    out = dict(row)
    out["dssr_status"] = status
    out["dssr_json"] = str(json_path)
    out["dssr_stack_links_between_core_neighbors"] = stack_links(dssr, core_residues)
    out["dssr_local_pairs"] = " || ".join(local_pairs) if local_pairs else "."
    out["dssr_pair_mask"] = "".join(pair_mask)
    out["dssr_partner_delta_pattern"] = ",".join(partner_delta)
    out["dssr_lw_pattern"] = ",".join(lw_pattern)
    out["dssr_dssr_pattern"] = ",".join(dssr_pattern)
    out["dssr_name_pattern"] = ",".join(name_pattern)
    out["dssr_bp_pattern"] = ",".join(bp_pattern)
    out["dssr_partner_nt_pattern"] = ",".join(partner_nt_pattern)
    out["dssr_pair_index_pattern"] = ",".join(pair_index_pattern)
    out.pop("strict_clean", None)
    out.pop("strict_clean_reason", None)
    return out


def state_summary(rows: list[dict[str, str]], metric: str) -> dict[str, str]:
    counts = Counter(row[metric] for row in rows)
    dominant_state, dominant_count = counts.most_common(1)[0]
    all_states = "; ".join(f"{state} ({count})" for state, count in counts.most_common())
    return {
        "metric": metric,
        "n_instances": str(len(rows)),
        "n_states": str(len(counts)),
        "dominant_state": dominant_state,
        "dominant_count": str(dominant_count),
        "dominant_fraction": f"{dominant_count / len(rows):.4f}",
        "all_states": all_states,
    }


def write_instances(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "kmer", "entry_id", "pdb_chain", "family", "clan",
        "token_start_1based", "raw_indices_1based", "author_residues",
        "sequence", "pair_sep_pattern", "pair_orient_pattern",
        "stack_sep_pattern", "n_external_pair", "n_internal_pair",
        "dssr_status", "dssr_json",
        "dssr_stack_links_between_core_neighbors", "dssr_local_pairs",
        "dssr_pair_mask", "dssr_partner_delta_pattern", "dssr_lw_pattern",
        "dssr_dssr_pattern", "dssr_name_pattern", "dssr_bp_pattern",
        "dssr_partner_nt_pattern", "dssr_pair_index_pattern",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    rows_by_kmer = defaultdict(list)
    for row in rows:
        rows_by_kmer[row["kmer"]].append(row)

    metrics = [
        "dssr_pair_mask",
        "dssr_partner_delta_pattern",
        "dssr_lw_pattern",
        "dssr_dssr_pattern",
        "dssr_name_pattern",
        "dssr_bp_pattern",
        "dssr_stack_links_between_core_neighbors",
        "sequence",
    ]
    fieldnames = [
        "kmer", "metric", "n_instances", "n_states", "dominant_state",
        "dominant_count", "dominant_fraction", "all_states",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for kmer in sorted(rows_by_kmer):
            kmer_rows = rows_by_kmer[kmer]
            n = str(len(kmer_rows))
            writer.writerow({
                "kmer": kmer,
                "metric": "n_motif_core_matches",
                "n_instances": n,
            })
            writer.writerow({
                "kmer": kmer,
                "metric": "n_strict_clean_core_matches",
                "n_instances": n,
            })
            for metric in metrics:
                record = state_summary(kmer_rows, metric)
                record["kmer"] = kmer
                writer.writerow(record)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--instances", type=Path, default=DEFAULT_INSTANCES)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    manifest_rows = [
        row for row in read_tsv(args.manifest)
        if str(row.get("strict_clean", "1")) == "1"
    ]
    annotated = [annotate_row(row) for row in manifest_rows]
    args.instances.parent.mkdir(parents=True, exist_ok=True)
    write_instances(args.instances, annotated)
    write_summary(args.summary, annotated)
    print(f"wrote {args.instances}")
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
