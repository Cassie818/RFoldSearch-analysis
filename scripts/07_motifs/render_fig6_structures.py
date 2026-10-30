"""Fig. 6F/G structure renders: superposed fragments drawn with ChimeraX."""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_fig6 as F

OUT = F.RENDER_DIR
VIEW = (-62, 24)        # azimuth, elevation (degrees), as in the earlier matplotlib panels
BACKBONE = {"P", "OP1", "OP2", "OP3", "O5'", "C5'", "O3'", "O2'"}


def camera(azimuth: float, elevation: float) -> np.ndarray:
    az, el = np.radians([azimuth, elevation])
    right = np.array([-np.sin(az), np.cos(az), 0.0])
    toward = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return np.vstack([right, np.cross(toward, right), toward])


def prepare() -> None:
    data = F.preflight(F.DEFAULT_CIF_DIR)
    view = camera(*VIEW)
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    manifest = {"panels": {}, "camera_rotation": view.tolist()}
    max_extent = 0.0
    for letter, rows, fragments, n in (("F", data["exact_hairpin"], data["gnra_coords"], 6),
                                       ("G", data["tloop"], data["tloop_coords"], 4)):
        # As in the earlier panels: the T-loop representative is the medoid among the tRNAs
        # numbered in the standard way (U54-G57), so that partners read as A58, G18 and G19.
        allowed = ([i for i, row in enumerate(rows) if row["author_residues"].split(",")[0] == "54"]
                   if letter == "G" else None)
        medoid = F.representative_index(fragments, allowed)
        order = [medoid] + [i for i in range(len(rows)) if i != medoid]
        panel = {"representative": rows[medoid]["pdb_chain"], "count": len(rows), "n_residues": n,
                 "colors": F.POSITION_COLORS if letter == "F" else [F.TLOOP_COLOR] * n,
                 "models": [], "links": []}
        for rank, index in enumerate(order):
            row = rows[index]
            _pdb, model, chain = F.stem_parts(row["pdb_chain"])
            residues = F.parse_atom_site(F.DEFAULT_CIF_DIR / f"{row['pdb_chain']}.cif", chain, model,
                                         full_keys=True)
            keys = [(key[4], key[5]) for key in map(F.parse_key_string, row["residue_keys"].split(","))][:n]
            rotation, mu = F.superposition(fragments[index], fragments[medoid])
            selected = [("A", j + 1, key) for j, key in enumerate(keys)]
            partners = data["tloop_partners"][index] if letter == "G" else []
            for j, partner in enumerate(partners):
                number, insertion = re.fullmatch(r"[A-Z]+(-?\d+)(.*)", partner["label"]).groups()
                selected.append(("B", j + 1, (int(number), insertion)))
            atoms, centres = [], []
            for out_chain, number, key in selected:
                base, xyz = residues[key]
                for name, point in xyz.items():
                    atoms.append({"chain": out_chain, "number": number, "base": base, "name": name,
                                  "xyz": (((np.asarray(point) - mu) @ rotation.T) @ view.T).tolist()})
                ring = [point for name, point in xyz.items() if "'" not in name and name not in BACKBONE]
                centres.append((((np.mean(ring, axis=0) - mu) @ rotation.T) @ view.T).tolist())
            panel["models"].append({"source": row["pdb_chain"], "rank": rank, "atoms": atoms})
            if rank == 0:
                panel["centres"] = centres[:n]
                panel["labels"] = [f"{residues[key][0]}{key[0]}{key[1]}" for key in keys]
                panel["links"] = [{"start": centres[p["position"] - 1], "end": centres[n + j],
                                   "label": p["label"], "lw": p["lw"], "position": p["position"]}
                                  for j, p in enumerate(partners)]
        # centre the ensemble in the view; a translation only
        points = np.array([atom["xyz"] for m in panel["models"] for atom in m["atoms"]])
        offset = (points.min(axis=0) + points.max(axis=0)) / 2
        for m in panel["models"]:
            for atom in m["atoms"]:
                atom["xyz"] = (np.asarray(atom["xyz"]) - offset).tolist()
        panel["centres"] = [(np.asarray(c) - offset).tolist() for c in panel["centres"]]
        for link in panel["links"]:
            link["start"] = (np.asarray(link["start"]) - offset).tolist()
            link["end"] = (np.asarray(link["end"]) - offset).tolist()
        max_extent = max(max_extent, float(np.abs((points - offset)[:, :2]).max()))
        for m in panel["models"]:
            path = OUT / "models" / f"{letter}_{m['rank']:02}.pdb"
            lines = [f"ATOM  {serial:5d} {a['name']:>4s} {a['base']:>3s} {a['chain']}{a['number']:4d}    "
                     f"{a['xyz'][0]:8.3f}{a['xyz'][1]:8.3f}{a['xyz'][2]:8.3f}{1.0:6.2f}{0.0:6.2f}"
                     f"          {a['name'][0]:>2s}\n" for serial, a in enumerate(m["atoms"], 1)]
            path.write_text("".join(lines) + "END\n")
            m["path"] = str(path.relative_to(F.REPO))
        manifest["panels"][letter] = panel
    # one field width for both panels, so F and G share a physical scale
    manifest["camera_width_angstrom"] = float(math.ceil(2 * max_extent + 8))
    (OUT / "aligned_models.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {OUT / 'aligned_models.json'}")


def render(session, letter: str) -> None:
    from chimerax.core.commands import run
    from chimerax.geometry import Place
    manifest = json.loads((OUT / "aligned_models.json").read_text())
    panel = manifest["panels"][letter]
    run(session, "close all"); run(session, "set bgColor white"); run(session, "camera ortho")
    for m in panel["models"]:
        model = run(session, f'open "{F.REPO / m["path"]}"')[0]
        spec = model.atomspec
        # the PDB carries topology only; restore full-precision coordinates
        lookup = {(a["chain"], a["number"], a["name"]): a["xyz"] for a in m["atoms"]}
        for atom in model.atoms:
            atom.coord = lookup[(atom.residue.chain_id, atom.residue.number, atom.name)]
        for bond in list(model.bonds):             # external partners are separate residues
            a1, a2 = bond.atoms
            if a1.residue != a2.residue and "B" in {a1.residue.chain_id, a2.residue.chain_id}:
                bond.delete()
        for group in list(model.pbg_map.values()):
            group.delete()
        run(session, f"hide {spec} cartoons"); run(session, f"show {spec} atoms")
        run(session, f"style {spec} stick"); run(session, f"color {spec} #DDDDDD")
        run(session, f"size {spec} stickRadius 0.025")
        if m["rank"] == 0:                          # representative: filled rings, position colours
            run(session, f"nucleotides {spec} fill"); run(session, f"size {spec} stickRadius 0.13")
            for j, colour in enumerate(panel["colors"], 1):
                run(session, f"color {spec}/A:{j} {colour}")
            if letter == "G":
                run(session, f"color {spec}/B #858585")
        for atom in model.atoms:
            if atom.residue.chain_id == "B" and atom.name in BACKBONE:
                atom.display = False
    run(session, "lighting simple"); run(session, "lighting depthCue false")
    run(session, "graphics silhouettes false"); run(session, "view")
    session.main_view.camera.position = Place(origin=(0, 0, 100))
    session.main_view.camera.field_width = manifest["camera_width_angstrom"]
    run(session, "windowsize 1000 1000"); run(session, "wait 2")
    run(session, f'save "{OUT / f"molecule_{letter}.png"}" width 1600 height 1600 supersample 3')


if __name__ == "__main__" or "session" in globals():
    stage = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    if stage == "prepare":
        prepare()
    elif stage == "render":
        render(session, sys.argv[2])  # noqa: F821  (ChimeraX provides `session`)
    else:
        raise SystemExit(f"unknown stage {stage}")
