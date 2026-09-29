#!/usr/bin/env python3
"""Alignment case figures for the four RF00015 pairs (query 5O9Z_4).

For each pair: the C4' traces superposed on the Rfam reference correspondences
(not the RFoldseek traceback), and the Rfam reference and RFoldseek alignments.
Alignment rows are read from alignment_cases.json, written by
build_alignment_cases.py (which reconstructs the benchmark alignments and checks
their scores). Everything is drawn with matplotlib, so the PDFs are vector.

Writes figures/supp_alignment_cases.{png,pdf}.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "files/ablation/rfam3d_reference_disjoint_independent_3dseed"
STRUCTURES = DATA / "reconstructed_chains"
SEED = DATA / "rfam_3d_seed.gz"
ROWS = DATA / "alignment_cases.json"
FIGURES = ROOT / "figures"

QUERY = ("5O9Z_4", "URS0000200C70_9606/1-145", "PDB_00005O9Z_1_4.cif")
CASES = [
    ("A", "2N7M_X", "URS00008FED30_32630/1-92", "PDB_00002N7M_1_X.cif", 327, 720),
    ("B", "5GAN_V", "URS00001143F5_4932/1-160", "PDB_00005GAN_1_V.cif", 832, 1325),
    ("C", "6AH0_I", "URS00003F07BD_9606/1-144", "PDB_00006AH0_1_I.cif", 1436, 1930),
    ("D", "6QW6_4", "URS0000E60BB1_9606/2-146", "PDB_00006QW6_1_4.cif", 2040, 2535),
]

QUERY_COLOR = "#5B62A9"
TARGET_COLOR = "#D28644"
TOKEN_COLOR = "#2A8C8E"
SEQ_COLOR = "#B07A45"
INK = "#20282D"
MUTED = "#6D7982"
MONO = "DejaVu Sans Mono"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})

FIG_W = 7.0                 # inches
STRUCT_W = 1.35             # structure panel width, inches
TEXT_X = 1.6                # left edge of the alignment text, inches
FS = 6.2                    # alignment font size, pt
LINE = FS * 1.2 / 72        # line height, inches
HEADER = 0.34               # case header height, inches
GAP = 0.2                   # space between cases, inches


def parse_seed() -> tuple[dict[str, str], str]:
    rows: dict[str, list[str]] = {}
    rf: list[str] = []
    active = False
    with gzip.open(SEED, "rt", encoding="latin-1") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith("#=GF AC"):
                active = line.split()[2] == "RF00015"
            elif active and line == "//":
                break
            elif active and line.startswith("#=GC RF"):
                rf.append(line.split()[2])
            elif active and line and not line.startswith("#"):
                fields = line.split()
                if len(fields) > 1:
                    rows.setdefault(fields[0], []).append(fields[1])
    return {key: "".join(parts) for key, parts in rows.items()}, "".join(rf)


def read_chain(path: Path) -> tuple[str, dict[int, np.ndarray]]:
    sequence: dict[int, str] = {}
    c4: dict[int, np.ndarray] = {}
    atom_fields: list[str] = []
    in_polymer = False
    in_atom = False
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line == "loop_":
            in_polymer = False
            in_atom = False
            atom_fields = []
            continue
        if line.startswith("_entity_poly_seq."):
            in_polymer = True
            continue
        if line.startswith("_atom_site."):
            in_atom = True
            atom_fields.append(line)
            continue
        if line == "#":
            in_polymer = False
            in_atom = False
            continue
        fields = line.split()
        if in_polymer and len(fields) >= 4 and fields[3].isdigit():
            sequence[int(fields[3]) - 1] = fields[2].upper().replace("T", "U")
        if in_atom and fields and fields[0] == "ATOM":
            record = dict(zip(atom_fields, fields))
            if record.get("_atom_site.label_atom_id") == "C4'":
                index = int(record["_atom_site.label_seq_id"]) - 1
                c4[index] = np.array([
                    float(record["_atom_site.Cartn_x"]),
                    float(record["_atom_site.Cartn_y"]),
                    float(record["_atom_site.Cartn_z"]),
                ])
    if not sequence or not c4:
        raise ValueError(f"No polymer sequence or C4' coordinates in {path}")
    polymer = "".join(sequence.get(i, "N") for i in range(max(sequence) + 1))
    return polymer, c4


def map_sequence(reference: str, structure: str) -> dict[int, int]:
    """Global affine alignment, using the seed-to-structure mapping scores."""
    reference = reference.upper().replace("T", "U")
    structure = structure.upper().replace("T", "U")
    n, m = len(reference), len(structure)
    score = np.full((3, n + 1, m + 1), -1e9, dtype=float)
    back = np.zeros((3, n + 1, m + 1), dtype=np.int8)
    score[0, 0, 0] = 0
    for i in range(1, n + 1):
        score[1, i, 0] = -3 - .5 * (i - 1)
        back[1, i, 0] = 0 if i == 1 else 1
    for j in range(1, m + 1):
        score[2, 0, j] = -3 - .5 * (j - 1)
        back[2, 0, j] = 0 if j == 1 else 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            prior = score[:, i - 1, j - 1]
            state = int(np.argmax(prior))
            score[0, i, j] = prior[state] + (2 if reference[i - 1] == structure[j - 1] else -1)
            back[0, i, j] = state
            prior = score[:, i - 1, j] + np.array([-3, -.5, -3])
            state = int(np.argmax(prior))
            score[1, i, j], back[1, i, j] = prior[state], state
            prior = score[:, i, j - 1] + np.array([-3, -3, -.5])
            state = int(np.argmax(prior))
            score[2, i, j], back[2, i, j] = prior[state], state
    mapping = {}
    i, j = n, m
    state = int(np.argmax(score[:, i, j]))
    while i or j:
        previous = int(back[state, i, j])
        if state == 0:
            mapping[i - 1] = j - 1
            i -= 1
            j -= 1
        elif state == 1:
            i -= 1
        else:
            j -= 1
        state = previous
    return mapping


def column_coordinates(aligned: str, sequence: str,
                       coordinates: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    ungapped = "".join(base for base in aligned if base not in ".-")
    residue_map = map_sequence(ungapped, sequence)
    output = {}
    index = 0
    for column, base in enumerate(aligned):
        if base not in ".-":
            residue = residue_map.get(index)
            if residue in coordinates:
                output[column] = coordinates[residue]
            index += 1
    return output


def fitted_structures(query: dict[int, np.ndarray],
                      target: dict[int, np.ndarray], match_columns: set[int]
                      ) -> tuple[np.ndarray, np.ndarray, float, int]:
    common = sorted(set(query) & set(target) & match_columns)
    q_fit = np.array([query[i] for i in common])
    t_fit = np.array([target[i] for i in common])
    q_mean, t_mean = q_fit.mean(axis=0), t_fit.mean(axis=0)
    u, _, vt = np.linalg.svd((t_fit - t_mean).T @ (q_fit - q_mean))
    sign = np.linalg.det(u @ vt)
    rotation = u @ np.diag([1.0, 1.0, np.sign(sign)]) @ vt
    transformed_fit = (t_fit - t_mean) @ rotation + q_mean
    rmsd = float(np.sqrt(np.mean(np.sum((transformed_fit - q_fit) ** 2, axis=1))))
    q_all = np.array([point for _, point in sorted(query.items())])
    t_all = np.array([(point - t_mean) @ rotation + q_mean
                      for _, point in sorted(target.items())])
    return q_all, t_all, rmsd, len(common)


def trace_xy(points: np.ndarray, centre: np.ndarray, view: np.ndarray) -> np.ndarray:
    """2-D projection with NaN breaks at chain gaps (C4'-C4' > 13 A)."""
    xy = (points - centre) @ view
    out = [xy[0]]
    for first, second, p in zip(points[:-1], points[1:], xy[1:]):
        if np.linalg.norm(first - second) > 13:
            out.append([np.nan, np.nan])
        out.append(p)
    return np.array(out)


def structure_axes(ax, query: np.ndarray, target: np.ndarray) -> None:
    # A fixed, query-derived PCA view is applied to both fitted structures.
    centre = query.mean(axis=0)
    _, _, axes = np.linalg.svd(query - centre, full_matrices=False)
    view = axes[:2].T
    for pts, color, z in ((query, QUERY_COLOR, 2), (target, TARGET_COLOR, 3)):
        xy = trace_xy(pts, centre, view)
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1.6, alpha=0.9, solid_capstyle="round",
                solid_joinstyle="round", zorder=z)
    ax.set_aspect("equal")
    ax.axis("off")


def alignment_lines(case: dict, query: str) -> list:
    """(kind, payload) items: section headings and the four rows of each chunk."""
    items = [("head", f"Rfam reference alignment  (RMSD {case['ref_rmsd']:.1f} \u00c5 over {case['n_ref']} nt)")]
    for c in case["reference"]:
        items += [("q", c, query), ("s", c, "3Dr"), ("n", c, "Seq"), ("t", c, case["target"])]
    items += [("space", None), ("head", f"RFoldseek alignment  (RMSD {case['rf_rmsd']:.1f} \u00c5 over {case['n_rf']} nt)")]
    for c in case["rfoldseek"]:
        items += [("q", c, query), ("s", c, "3Dr"), ("n", c, "Seq"), ("t", c, case["target"])]
    return items


def case_height(case: dict, query: str) -> float:
    return HEADER + len(alignment_lines(case, query)) * LINE


def draw_case(fig, case: dict, query: str, q_trace, t_trace, top: float, H: float) -> None:
    """Draw one case with its top edge at `top` inches from the figure top."""
    y = lambda inch: 1 - inch / H
    x = lambda inch: inch / FIG_W
    fig.text(x(0.08), y(top + 0.05), case["label"], fontsize=12, fontweight="bold", va="top", color=INK)
    fig.text(x(0.34), y(top + 0.07), f"{query} vs {case['target']}", fontsize=9, fontweight="bold",
             va="top", color=INK)
    fig.text(x(FIG_W - 0.1), y(top + 0.08),
             f"seq. id. {100 * case['seq_id']:.0f}%    3Dr id. {100 * case['tok_id']:.0f}%    "
             f"RFoldseek exact F1 {case['f1']:.2f}", fontsize=7.5, color=MUTED, va="top", ha="right")
    lines = alignment_lines(case, query)
    body = len(lines) * LINE
    side = min(STRUCT_W, body)
    ax = fig.add_axes([x(0.1), y(top + HEADER + (body + side) / 2), x(STRUCT_W), side / H])
    structure_axes(ax, q_trace, t_trace)
    cw = FS * 0.602 / 72                       # DejaVu Sans Mono advance, inches
    for i, item in enumerate(lines):
        yy = y(top + HEADER + (i + 0.5) * LINE)
        kind = item[0]
        if kind == "head":
            fig.text(x(TEXT_X), yy, item[1], fontsize=6.8, fontweight="bold", color=INK, va="center")
            continue
        if kind == "space":
            continue
        c, name = item[1], item[2]
        color = {"q": QUERY_COLOR, "t": TARGET_COLOR, "s": TOKEN_COLOR, "n": SEQ_COLOR}[kind]
        fig.text(x(TEXT_X), yy, name, fontsize=FS, color=color, va="center",
                 fontweight="bold" if kind in "qt" else "normal")
        x0 = TEXT_X + 0.52
        if kind in "qt":
            a, b = (c["q0"], c["q1"]) if kind == "q" else (c["t0"], c["t1"])
            fig.text(x(x0 + 3 * cw), yy, f"{a}", fontsize=FS, family=MONO, color=MUTED, va="center", ha="right")
            fig.text(x(x0 + (5 + len(c[kind])) * cw), yy, f"{b}", fontsize=FS, family=MONO, color=MUTED,
                     va="center")
        fig.text(x(x0 + 4 * cw), yy, c[kind], fontsize=FS, family=MONO, color=color, va="center")


def compose(cases, query, traces, out: Path) -> None:
    heights = [case_height(c, query) for c in cases]
    legend = 0.42
    H = sum(heights) + GAP * (len(cases) - 1) + legend + 0.05
    fig = plt.figure(figsize=(FIG_W, H))
    top = 0.05
    for i, (case, h) in enumerate(zip(cases, heights)):
        draw_case(fig, case, query, *traces[case["target"]], top, H)
        top += h
        if i < len(cases) - 1:
            fig.add_artist(plt.Line2D([0.08 / FIG_W, 1 - 0.1 / FIG_W], [1 - (top + GAP / 2) / H] * 2,
                                      color="#DDE2E4", lw=0.8))
            top += GAP
    handles = [plt.Line2D([], [], color=QUERY_COLOR, lw=2, label=f"{query} (query)"),
               plt.Line2D([], [], color=TARGET_COLOR, lw=2, label="target"),
               plt.Line2D([], [], color=TOKEN_COLOR, lw=0, marker="$|$", ms=6,
                          label="3Dr: | identical, + positive score, . other"),
               plt.Line2D([], [], color=SEQ_COLOR, lw=0, marker="$|$", ms=6, label="Seq: identical nucleotide")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.08 / FIG_W, 0.02 / H), ncol=4,
               frameon=False, fontsize=7, handlelength=1.6, columnspacing=1.6)
    for suffix in (".png", ".pdf"):
        fig.savefig(out.with_suffix(suffix), dpi=300, facecolor="white")
    plt.close(fig)
    print(f"wrote {out.with_suffix('.png')}")


def main() -> None:
    seed, annotation = parse_seed()
    match_columns = {i for i, char in enumerate(annotation) if char.isupper()}
    rows = json.load(open(ROWS))
    query_seq, query_xyz = read_chain(STRUCTURES / QUERY[2])
    query_columns = column_coordinates(seed[QUERY[1]], query_seq, query_xyz)
    traces = {}
    for label, name, seed_row, filename, *_ in CASES:
        target_seq, target_xyz = read_chain(STRUCTURES / filename)
        target_columns = column_coordinates(seed[seed_row], target_seq, target_xyz)
        q_trace, t_trace, rmsd, n_ref = fitted_structures(query_columns, target_columns, match_columns)
        traces[name] = (q_trace, t_trace)
        print(f"{label}: {name}, {n_ref} reference C4' pairs, RMSD {rmsd:.2f} \u00c5")
    cases = rows["cases"]
    compose(cases, rows["query"], traces, FIGURES / "supp_alignment_cases")


if __name__ == "__main__":
    main()
