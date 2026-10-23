from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "07_motifs"
    / "residue_keys.py"
)
SPEC = importlib.util.spec_from_file_location("motif_residue_keys", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_bgsu_insertion_codes_are_preserved(tmp_path):
    atlas = tmp_path / "hl.csv"
    atlas.write_text(
        '>HL_34789.6\n'
        '"3W3S|1|B|C|47|||F","3W3S|1|B|G|47|||G",'
        '"3W3S|1|B|C|47|||H","3W3S|1|B|A|47|||I"\n'
    )

    occurrence = MODULE.parse_bgsu(atlas)["HL_34789.6"][0]

    assert occurrence == {
        "3W3S|1|B|C|47|F",
        "3W3S|1|B|G|47|G",
        "3W3S|1|B|C|47|H",
        "3W3S|1|B|A|47|I",
    }


def test_cif_map_retains_model_identity_and_insertion_codes(tmp_path):
    cif = tmp_path / "PDB_00003W3S_1_B.cif"
    cif.write_text(
        "loop_\n"
        "_atom_site.group_PDB\n"
        "_atom_site.auth_comp_id\n"
        "_atom_site.auth_asym_id\n"
        "_atom_site.auth_seq_id\n"
        "_atom_site.pdbx_PDB_ins_code\n"
        "_atom_site.pdbx_PDB_model_num\n"
        "ATOM C B 47 F 1\n"
        "ATOM C B 47 F 1\n"
        "ATOM G B 47 G 1\n"
        "ATOM A B 47 I 1\n"
        "ATOM U B 47 I 2\n"
        "ATOM U C 47 I 1\n"
        "#\n"
    )

    keys = MODULE.cif_residue_keys(cif, "PDB_00003W3S_1_B")

    assert keys == [
        "3W3S|1|B|C|47|F",
        "3W3S|1|B|G|47|G",
        "3W3S|1|B|A|47|I",
    ]
