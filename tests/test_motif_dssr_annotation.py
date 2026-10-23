from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "07_motifs"
    / "known_motif_core_dssr_annotation.py"
)
SPEC = importlib.util.spec_from_file_location("known_motif_core_dssr_annotation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def annotate(monkeypatch, pairs, author_residues, pdb_chain="PDB_00000TST_1_A"):
    dssr = {"pairs": pairs, "stacks": []}
    monkeypatch.setattr(
        MODULE,
        "load_or_run_dssr",
        lambda row: (dssr, "test", Path("test.json")),
    )
    return MODULE.annotate_row({
        "author_residues": author_residues,
        "pdb_chain": pdb_chain,
    })


def test_all_pair_records_are_retained_independent_of_json_order(monkeypatch):
    pairs = [
        {
            "index": 1,
            "nt1": "..A.G.21.",
            "nt2": "..A.A.24.",
            "LW": "tSH",
            "DSSR": "tm-M",
            "name": "Sheared",
            "bp": "G-A",
        },
        {
            "index": 2,
            "nt1": "..A.A.24.",
            "nt2": "..C.G.2.",
            "LW": "tWS",
            "DSSR": "tW+m",
            "name": "Linker",
            "bp": "A+G",
        },
    ]

    forward = annotate(monkeypatch, pairs, "20,21,22,23,24,25")
    reverse = annotate(monkeypatch, list(reversed(pairs)), "20,21,22,23,24,25")

    assert forward["dssr_pair_count_pattern"] == "0,1,0,0,2,0"
    assert forward["dssr_lw_pattern"] == ".,tSH,.,.,tSH&tWS,."
    assert forward["dssr_partner_scope_pattern"] == ".,I,.,.,E&I,."
    assert forward["dssr_n_multiply_paired_core_nts"] == "1"
    for field in (
        "dssr_local_pairs",
        "dssr_partner_delta_pattern",
        "dssr_lw_pattern",
        "dssr_dssr_pattern",
        "dssr_name_pattern",
        "dssr_bp_pattern",
        "dssr_partner_nt_pattern",
        "dssr_pair_index_pattern",
    ):
        assert forward[field] == reverse[field]


def test_tloop_single_pair_signature_and_external_scope_are_unchanged(monkeypatch):
    pairs = [
        {"index": 1, "nt1": "..A.U.54.", "nt2": "..A.A.58.", "LW": "tWH"},
        {"index": 2, "nt1": "..A.U.55.", "nt2": "..A.G.18.", "LW": "tWS"},
        {"index": 3, "nt1": "..A.C.56.", "nt2": "..A.G.19.", "LW": "cWW"},
    ]

    row = annotate(monkeypatch, pairs, "54,55,56,57")

    assert row["dssr_pair_mask"] == "PPPU"
    assert row["dssr_pair_count_pattern"] == "1,1,1,0"
    assert row["dssr_lw_pattern"] == "tWH,tWS,cWW,."
    assert row["dssr_partner_scope_pattern"] == "E,E,E,."
    assert row["dssr_n_internal_partner_records"] == "0"
    assert row["dssr_n_external_partner_records"] == "3"
    assert row["dssr_n_multiply_paired_core_nts"] == "0"


def test_full_structure_json_is_restricted_to_the_core_chain(monkeypatch):
    pairs = [
        # Same residue number on another chain must not become a record for A24.
        {"index": 1, "nt1": "..B.A.24.", "nt2": "..C.G.2.", "LW": "tWS"},
        {"index": 2, "nt1": "..A.G.21.", "nt2": "..A.A.24.", "LW": "tSH"},
        # This inter-chain interaction belongs to A24 and must be retained.
        {"index": 3, "nt1": "..A.A.24.", "nt2": "..C.G.2.", "LW": "tWS"},
    ]

    row = annotate(
        monkeypatch,
        pairs,
        "20,21,22,23,24,25",
        pdb_chain="PDB_00003NPQ_1_A",
    )

    assert row["dssr_pair_count_pattern"] == "0,1,0,0,2,0"
    assert row["dssr_lw_pattern"] == ".,tSH,.,.,tSH&tWS,."


def test_insertion_coded_residues_remain_distinct(monkeypatch):
    pairs = [
        {
            "index": 1,
            "nt1": "..B.C.47.F",
            "nt2": "..B.A.47.I",
            "LW": "cWW",
        },
        {
            "index": 2,
            "nt1": "..B.G.47.G",
            "nt2": "..B.C.47.H",
            "LW": "tSH",
        },
    ]
    dssr = {"pairs": pairs, "stacks": []}
    monkeypatch.setattr(
        MODULE,
        "load_or_run_dssr",
        lambda row: (dssr, "test", Path("test.json")),
    )
    row = MODULE.annotate_row({
        "pdb_chain": "PDB_00003W3S_1_B",
        "author_residues": "47F,47G,47H,47I",
        "residue_keys": (
            "3W3S|1|B|C|47|F,3W3S|1|B|G|47|G,"
            "3W3S|1|B|C|47|H,3W3S|1|B|A|47|I"
        ),
    })

    assert row["dssr_pair_count_pattern"] == "1,1,1,1"
    assert row["dssr_lw_pattern"] == "cWW,tSH,tSH,cWW"
    assert row["dssr_partner_scope_pattern"] == "I,I,I,I"


def test_canonical_key_rejects_wrong_nucleotide_identity(monkeypatch):
    pairs = [
        {"index": 1, "nt1": "B.U47^F", "nt2": "B.A47^I", "LW": "cWW"},
    ]
    dssr = {"pairs": pairs, "stacks": []}
    monkeypatch.setattr(
        MODULE,
        "load_or_run_dssr",
        lambda row: (dssr, "test", Path("test.json")),
    )
    row = MODULE.annotate_row({
        "pdb_chain": "PDB_00003W3S_1_B",
        "author_residues": "47F",
        "residue_keys": "3W3S|1|B|C|47|F",
    })

    assert row["dssr_pair_count_pattern"] == "0"
    assert row["dssr_lw_pattern"] == "."


def test_required_lw_pattern_allows_additional_pairings():
    assert MODULE.lw_pattern_contains(
        "cWW,tSH&tWS,.,cWH,tSH,cWW&tWW",
        "cWW,tSH,.,.,tSH,cWW",
    )
    assert not MODULE.lw_pattern_contains(
        "cWW,tWH,.,.,tWH,cWW",
        "cWW,tSH,.,.,tSH,cWW",
    )
