"""mmCIF reader returning every atom the structure scores need.

Residue validity mirrors rfoldsearch/backbone.py: a nucleotide is kept when
C3', C4' and O4' are all present. Indices in the returned arrays refer to the
kept nucleotides only, so they line up with a token alignment.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

REQUIRED = ("C3'", "C4'", "O4'")
CANONICAL = {"A": "A", "C": "C", "G": "G", "U": "U"}
WC_NITROGEN = {"A": "N1", "G": "N1", "C": "N3", "U": "N3"}
NAN3 = (np.nan, np.nan, np.nan)


def read_chain(path):
    """(sequence, C4' coords, C1' coords, WC nitrogen coords) or None."""
    cols, rows, in_loop = {}, [], False
    with open(path) as fh:
        for line in fh:
            if line.startswith("_atom_site."):
                cols[line.strip().split(".", 1)[1]] = len(cols)
                in_loop = True
                continue
            if in_loop and (line.startswith("ATOM") or line.startswith("HETATM")):
                rows.append(line.split())
            elif in_loop and rows and (line.startswith("#") or line.startswith("loop_")):
                break
    need = ("label_atom_id", "label_comp_id", "label_asym_id", "label_seq_id",
            "auth_seq_id", "Cartn_x", "Cartn_y", "Cartn_z", "pdbx_PDB_model_num")
    if not rows or any(k not in cols for k in need):
        return None
    i = {k: cols[k] for k in need}

    model, chain = rows[0][i["pdbx_PDB_model_num"]], rows[0][i["label_asym_id"]]
    residues, order = defaultdict(dict), []
    for r in rows:
        if r[i["pdbx_PDB_model_num"]] != model or r[i["label_asym_id"]] != chain:
            continue
        key = (r[i["label_seq_id"]], r[i["auth_seq_id"]])
        if key not in residues:
            order.append((key, r[i["label_comp_id"]].strip().upper()))
        try:
            residues[key][r[i["label_atom_id"]].strip('"')] = (
                float(r[i["Cartn_x"]]), float(r[i["Cartn_y"]]), float(r[i["Cartn_z"]]))
        except ValueError:
            continue

    seq, c4, c1, wc = [], [], [], []
    for key, comp in order:
        atoms = residues[key]
        if not all(a in atoms for a in REQUIRED):
            continue
        base = CANONICAL.get(comp, "N")
        seq.append(base)
        c4.append(atoms["C4'"])
        c1.append(atoms.get("C1'", NAN3))
        wc.append(atoms.get(WC_NITROGEN.get(base, ""), NAN3))
    if not seq:
        return None
    return ("".join(seq),
            np.asarray(c4, dtype=np.float64),
            np.asarray(c1, dtype=np.float64),
            np.asarray(wc, dtype=np.float64))
