"""Canonical residue identifiers shared by the BGSU motif analyses."""
from __future__ import annotations

import csv
import re
import shlex
from pathlib import Path


ResidueKey = tuple[str, int, str, str, int, str]

RNA_NAMES = {
    "A", "C", "G", "U", "PSU", "5MU", "4SU", "H2U", "UR3", "5MC",
    "OMC", "5BU", "OMG", "1MG", "2MG", "M2G", "7MG", "YG", "YYG",
    "1MA", "6MZ", "MA6", "I",
}
STEM_RE = re.compile(r"PDB_00([0-9A-Z]{6})_(\d+)_(.+)$", re.I)


def normalize_insertion_code(value: str | None) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text in {"", ".", "?"} else text


def parse_number_and_insertion(value: str, explicit: str = "") -> tuple[int, str]:
    match = re.fullmatch(r"\s*(-?\d+)([A-Za-z]?)\s*", str(value))
    if not match:
        raise ValueError(f"invalid author residue identifier: {value!r}")
    suffix = normalize_insertion_code(explicit) or normalize_insertion_code(match.group(2))
    return int(match.group(1)), suffix


def parse_stem(stem: str) -> tuple[str, int, str] | None:
    match = STEM_RE.fullmatch(stem)
    if not match:
        return None
    raw = match.group(1).upper()
    pdb_id = raw[2:] if raw.startswith("00") else raw
    return pdb_id, int(match.group(2)), match.group(3)


def make_key(
    pdb: str,
    model: int | str,
    chain: str,
    nucleotide: str,
    residue_number: int | str,
    insertion_code: str = "",
) -> ResidueKey:
    number, insertion = parse_number_and_insertion(
        str(residue_number), explicit=insertion_code
    )
    return (
        str(pdb).upper(),
        int(model),
        str(chain),
        str(nucleotide).upper(),
        number,
        insertion,
    )


def key_string(key: ResidueKey) -> str:
    pdb, model, chain, nucleotide, number, insertion = key
    return f"{pdb}|{model}|{chain}|{nucleotide}|{number}|{insertion}"


def parse_key_string(value: str) -> ResidueKey:
    fields = str(value).split("|")
    if len(fields) != 6:
        raise ValueError(f"invalid canonical residue key: {value!r}")
    return make_key(*fields)


def residue_label(key: ResidueKey) -> str:
    return f"{key[4]}{key[5]}"


def chain_key(key: ResidueKey) -> tuple[str, int, str]:
    return key[0], key[1], key[2]


def parse_bgsu_unit_id(value: str) -> ResidueKey:
    """Parse an FR3D/BGSU unit ID, including its optional insertion code.

    Unit IDs normally contain five pipe-separated fields. FR3D insertion-coded
    residues may contain empty atom/alternate-ID fields before the insertion
    code, for example ``3W3S|1|B|C|47|||F``.
    """
    fields = str(value).strip().strip('"').split("|")
    if len(fields) < 5:
        raise ValueError(f"invalid BGSU unit identifier: {value!r}")
    pdb, model, chain, nucleotide, number = fields[:5]
    insertion = fields[7] if len(fields) > 7 else ""
    return make_key(pdb, model, chain, nucleotide, number, insertion)


def parse_bgsu(path: Path) -> dict[str, list[set[str]]]:
    """Read a BGSU motif atlas file as sets of canonical residue keys."""
    motifs: dict[str, list[set[str]]] = {}
    current: str | None = None
    with path.open(newline="") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.startswith(">"):
                current = line[1:].strip()
                motifs[current] = []
                continue
            if not line.strip() or current is None:
                continue
            values = next(csv.reader([line]))
            occurrence = {key_string(parse_bgsu_unit_id(value)) for value in values}
            if occurrence:
                motifs[current].append(occurrence)
    return motifs


def _atom_site_rows(path: Path):
    """Yield atom_site rows as dictionaries using only the Python stdlib."""
    lines = path.read_text(errors="replace").splitlines()
    index = 0
    while index < len(lines):
        if lines[index].strip() != "loop_":
            index += 1
            continue
        index += 1
        columns = []
        while index < len(lines) and lines[index].lstrip().startswith("_atom_site."):
            columns.append(lines[index].strip())
            index += 1
        if not columns:
            continue
        while index < len(lines):
            stripped = lines[index].strip()
            if not stripped or stripped == "#" or stripped == "loop_" or stripped.startswith("_"):
                break
            values = shlex.split(stripped, posix=True)
            index += 1
            if len(values) >= len(columns):
                yield dict(zip(columns, values))


def cif_residue_keys(path: Path, stem: str) -> list[str]:
    """Return full residue keys in the chain order used by the token mask."""
    parsed = parse_stem(stem)
    if parsed is None:
        raise ValueError(f"cannot parse chain stem: {stem}")
    pdb, model, chain = parsed
    keys: list[str] = []
    seen: set[str] = set()
    for row in _atom_site_rows(path):
        try:
            row_model = int(float(row.get("_atom_site.pdbx_PDB_model_num", "1")))
        except ValueError:
            continue
        row_chain = row.get("_atom_site.auth_asym_id", row.get("_atom_site.label_asym_id", ""))
        if row_model != model or row_chain != chain:
            continue
        nucleotide = row.get("_atom_site.auth_comp_id", row.get("_atom_site.label_comp_id", "")).upper()
        if nucleotide not in RNA_NAMES:
            continue
        number = row.get("_atom_site.auth_seq_id", row.get("_atom_site.label_seq_id", ""))
        insertion = row.get("_atom_site.pdbx_PDB_ins_code", "")
        try:
            key = key_string(make_key(pdb, model, chain, nucleotide, number, insertion))
        except ValueError:
            continue
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys
