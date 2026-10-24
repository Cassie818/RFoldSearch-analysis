"""Extract DSSR interaction signatures for known motif-core token words."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
MOTIF_DIR = REPO / "files" / "motifs"
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from residue_keys import make_key, parse_key_string

DEFAULT_MANIFEST = MOTIF_DIR / "known_motif_core_matches_production_new_relabel_manifest.tsv"
DEFAULT_INSTANCES = MOTIF_DIR / "known_motif_core_dssr_pair_conservation_instances_production_new.tsv"
DEFAULT_SUMMARY = MOTIF_DIR / "known_motif_core_dssr_pair_conservation_summary_production_new.tsv"
REQUIRED_DSSR_VERSION = "DSSR v2.9.2"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def residue_number(value: str) -> int | None:
    match = re.search(r"(-?\d+)(?!.*\d)", str(value))
    return int(match.group(1)) if match else None


def nucleotide_chain(value: str) -> str | None:
    """Extract the chain identifier from common DSSR nucleotide IDs."""
    text = str(value)
    if re.fullmatch(r"-?\d+[A-Za-z]?", text):
        return None
    parts = text.split(".")
    # Current DSSR JSON commonly uses ``..CHAIN.BASE.RESNUM.``.
    if len(parts) >= 5 and parts[2]:
        return parts[2]
    # Older output commonly uses ``CHAIN.BASE123`` or ``MODEL:CHAIN.BASE123``.
    prefix = parts[0]
    if ":" in prefix:
        prefix = prefix.rsplit(":", 1)[-1]
    return prefix or None


def nucleotide_insertion_code(value: str) -> str:
    """Extract a DSSR insertion code from dotted or caret notation."""
    text = str(value)
    caret = re.search(r"-?\d+\^([A-Za-z])", text)
    if caret:
        return caret.group(1)
    parts = text.split(".")
    for index, part in enumerate(parts):
        if re.fullmatch(r"-?\d+", part):
            for following in parts[index + 1:]:
                if following:
                    return following if re.fullmatch(r"[A-Za-z]", following) else ""
                if following == "":
                    continue
    suffix = re.search(r"-?\d+([A-Za-z])(?=[.\s]*$)", text)
    return suffix.group(1) if suffix else ""


def nucleotide_identity(value: str) -> str | None:
    """Extract the residue identity from compact or dotted DSSR nucleotide IDs."""
    text = str(value)
    if re.fullmatch(r"-?\d+[A-Za-z]?", text):
        return None
    parts = text.split(".")
    # Current dotted output: ``..CHAIN.BASE.RESNUM.``.
    if len(parts) >= 5 and parts[3]:
        return parts[3].upper()
    # Compact output: ``CHAIN.BASE123`` or ``MODEL:CHAIN.BASE123^A``.
    compact = parts[-1].split("^", 1)[0]
    match = re.fullmatch(r"(.+?)-?\d+", compact)
    return match.group(1).upper() if match else None


def residue_reference(
    value: str,
    default_chain: str | None = None,
    pdb: str | None = None,
    model: int | None = None,
) -> tuple | None:
    """Return a DSSR residue reference, using the canonical six-field key when possible."""
    number = residue_number(value)
    if number is None:
        return None
    chain = nucleotide_chain(value) or default_chain or ""
    insertion = nucleotide_insertion_code(value)
    nucleotide = nucleotide_identity(value)
    if pdb is not None and model is not None and nucleotide is not None:
        return make_key(pdb, model, chain, nucleotide, number, insertion)
    # Compatibility path for legacy manifests that contain only author numbers.
    return chain, number, insertion


def reference_number(reference: tuple) -> int:
    """Return the author residue number from a legacy or canonical reference."""
    return int(reference[4] if len(reference) == 6 else reference[1])


def validate_dssr_version(dssr: dict, json_path: Path) -> None:
    version = str(dssr.get("metadata", {}).get("version", ""))
    if not version.startswith(REQUIRED_DSSR_VERSION):
        raise RuntimeError(
            f"{json_path}: expected {REQUIRED_DSSR_VERSION}, found {version or 'missing metadata.version'}"
        )


def load_or_run_dssr(row: dict[str, str]) -> tuple[dict, str, Path]:
    json_path = Path(row["dssr_json"])
    if json_path.exists():
        dssr = json.load(json_path.open())
        validate_dssr_version(dssr, json_path)
        return dssr, "cached", json_path

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
    dssr = json.load(json_path.open())
    validate_dssr_version(dssr, json_path)
    return dssr, "computed", json_path


def pair_lookup(
    dssr: dict,
    core_chain: str | None = None,
    pdb: str | None = None,
    model: int | None = None,
) -> dict[tuple, list[dict]]:
    """Return every DSSR pairing record associated with each residue.

    A nucleotide can participate in more than one DSSR-annotated interaction.
    Keeping a list is important: using ``setdefault`` here previously retained
    only the first record and made the reported signature depend on JSON order.
    """
    lookup: dict[tuple, list[dict]] = defaultdict(list)
    for pair in dssr.get("pairs", []):
        nt1 = pair.get("nt1", "")
        nt2 = pair.get("nt2", "")
        ref1 = residue_reference(nt1, core_chain, pdb, model)
        ref2 = residue_reference(nt2, core_chain, pdb, model)
        if ref1 is None or ref2 is None:
            continue
        record_1 = dict(pair)
        record_1["core_nt"] = pair.get("nt1", ".")
        record_1["partner_nt"] = pair.get("nt2", ".")
        record_1["partner_resnum"] = reference_number(ref2)
        record_1["partner_ref"] = ref2
        record_2 = dict(pair)
        record_2["core_nt"] = pair.get("nt2", ".")
        record_2["partner_nt"] = pair.get("nt1", ".")
        record_2["partner_resnum"] = reference_number(ref1)
        record_2["partner_ref"] = ref1
        chain1 = nucleotide_chain(nt1)
        chain2 = nucleotide_chain(nt2)
        if core_chain is None or chain1 is None or chain1 == core_chain:
            lookup[ref1].append(record_1)
        if core_chain is None or chain2 is None or chain2 == core_chain:
            lookup[ref2].append(record_2)
    return dict(lookup)


def ordered_unique(values: list[str]) -> list[str]:
    """Return deterministic unique values without relying on DSSR order."""
    return sorted(set(values))


def joined_pair_value(records: list[dict], key: str) -> str:
    values = ordered_unique([str(record.get(key, ".")) for record in records])
    return "&".join(values) if values else "."


def joined_partner_deltas(records: list[dict], residue: int) -> str:
    values = sorted({int(record["partner_resnum"]) - residue for record in records})
    return "&".join(str(value) for value in values) if values else "."


def pair_scope(records: list[dict], core_residues: set[tuple]) -> str:
    scopes = {
        "I" if tuple(record["partner_ref"]) in core_residues else "E"
        for record in records
    }
    return "&".join(sorted(scopes)) if scopes else "."


def lw_pattern_contains(actual: str, required: str) -> bool:
    """Test for required LW interactions while allowing additional pair records.

    A dot in the required pattern imposes no constraint.  Non-dot positions may
    contain one or more ``&``-joined DSSR annotations and pass when the required
    annotation is among them.
    """
    actual_values = str(actual).split(",")
    required_values = str(required).split(",")
    if len(actual_values) != len(required_values):
        return False
    return all(
        expected == "." or expected in set(observed.split("&"))
        for observed, expected in zip(actual_values, required_values)
    )


def stack_links(
    dssr: dict,
    core_residues: list[tuple],
    core_chain: str | None,
    pdb: str | None = None,
    model: int | None = None,
) -> str:
    core_to_pos = {residue: pos for pos, residue in enumerate(core_residues, start=1)}
    links = set()
    for stack in dssr.get("stacks", []):
        nts_long = str(stack.get("nts_long", ""))
        residues = [
            residue_reference(item, core_chain, pdb, model)
            for item in nts_long.split(",")
        ]
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
    core_chain = row.get("chain")
    if not core_chain and row.get("pdb_chain"):
        core_chain = row["pdb_chain"].rsplit("_", 1)[-1]
    if row.get("residue_keys"):
        parsed_keys = [parse_key_string(value) for value in row["residue_keys"].split(",")]
        contexts = {(key[0], key[1], key[2]) for key in parsed_keys}
        if len(contexts) != 1:
            raise ValueError("core residue keys do not belong to one PDB/model/chain")
        core_pdb, core_model, key_chain = contexts.pop()
        if core_chain and core_chain != key_chain:
            raise ValueError(
                f"manifest chain {core_chain!r} disagrees with residue keys {key_chain!r}"
            )
        core_chain = key_chain
        core_residues = parsed_keys
    else:
        core_pdb = None
        core_model = None
        core_residues = [
            residue_reference(value, core_chain)
            for value in row["author_residues"].split(",")
        ]
        if any(value is None for value in core_residues):
            raise ValueError(f"cannot parse author residues: {row['author_residues']}")
        core_residues = [value for value in core_residues if value is not None]
    pairs = pair_lookup(
        dssr, core_chain=core_chain, pdb=core_pdb, model=core_model
    )

    pair_mask = []
    partner_delta = []
    lw_pattern = []
    dssr_pattern = []
    name_pattern = []
    bp_pattern = []
    partner_nt_pattern = []
    pair_index_pattern = []
    pair_count_pattern = []
    partner_scope_pattern = []
    local_pairs = []
    core_residue_set = set(core_residues)

    for pos, residue_ref in enumerate(core_residues, start=1):
        residue = reference_number(residue_ref)
        residue_pairs = pairs.get(residue_ref, [])
        pair_count_pattern.append(str(len(residue_pairs)))
        if not residue_pairs:
            pair_mask.append("U")
            partner_delta.append(".")
            lw_pattern.append(".")
            dssr_pattern.append(".")
            name_pattern.append(".")
            bp_pattern.append(".")
            partner_nt_pattern.append(".")
            pair_index_pattern.append(".")
            partner_scope_pattern.append(".")
            continue

        pair_mask.append("P")
        partner_delta.append(joined_partner_deltas(residue_pairs, residue))
        lw_pattern.append(joined_pair_value(residue_pairs, "LW"))
        dssr_pattern.append(joined_pair_value(residue_pairs, "DSSR"))
        name_pattern.append(joined_pair_value(residue_pairs, "name"))
        bp_pattern.append(joined_pair_value(residue_pairs, "bp"))
        partner_nt_pattern.append(joined_pair_value(residue_pairs, "partner_nt"))
        pair_index_pattern.append(joined_pair_value(residue_pairs, "index"))
        partner_scope_pattern.append(pair_scope(residue_pairs, core_residue_set))
        for pair in sorted(
            residue_pairs,
            key=lambda record: (
                str(record.get("partner_nt", ".")),
                str(record.get("LW", ".")),
                str(record.get("index", ".")),
            ),
        ):
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
    out["dssr_stack_links_between_core_neighbors"] = stack_links(
        dssr, core_residues, core_chain, core_pdb, core_model
    )
    out["dssr_local_pairs"] = " || ".join(local_pairs) if local_pairs else "."
    out["dssr_pair_mask"] = "".join(pair_mask)
    out["dssr_pair_count_pattern"] = ",".join(pair_count_pattern)
    out["dssr_n_multiply_paired_core_nts"] = str(
        sum(int(count) > 1 for count in pair_count_pattern)
    )
    out["dssr_partner_scope_pattern"] = ",".join(partner_scope_pattern)
    out["dssr_n_internal_partner_records"] = str(sum(
        tuple(record["partner_ref"]) in core_residue_set
        for residue in core_residues
        for record in pairs.get(residue, [])
    ))
    out["dssr_n_external_partner_records"] = str(sum(
        tuple(record["partner_ref"]) not in core_residue_set
        for residue in core_residues
        for record in pairs.get(residue, [])
    ))
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
        "token_start_1based", "raw_indices_1based", "author_residues", "residue_keys",
        "sequence", "pair_sep_pattern", "pair_orient_pattern",
        "stack_sep_pattern", "n_external_pair", "n_internal_pair",
        "dssr_status", "dssr_json",
        "dssr_stack_links_between_core_neighbors", "dssr_local_pairs",
        "dssr_pair_mask", "dssr_pair_count_pattern",
        "dssr_n_multiply_paired_core_nts", "dssr_partner_scope_pattern",
        "dssr_n_internal_partner_records", "dssr_n_external_partner_records",
        "dssr_partner_delta_pattern", "dssr_lw_pattern",
        "dssr_dssr_pattern", "dssr_name_pattern", "dssr_bp_pattern",
        "dssr_partner_nt_pattern", "dssr_pair_index_pattern",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    rows_by_kmer = defaultdict(list)
    for row in rows:
        rows_by_kmer[row["kmer"]].append(row)

    metrics = [
        "dssr_pair_mask",
        "dssr_pair_count_pattern",
        "dssr_n_multiply_paired_core_nts",
        "dssr_partner_scope_pattern",
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
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
        )
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
