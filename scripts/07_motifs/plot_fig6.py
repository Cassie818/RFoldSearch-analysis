"""Draw Fig. 6 from the recurrent-word control tables."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, PathPatch, Rectangle
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
from mpl_toolkits.mplot3d import proj3d


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "common"))
import figstyle as S  # noqa: E402  (palette and type sizes shared with Figs 2-5)
sys.path.insert(0, str(REPO / "scripts" / "07_motifs" / "word_controls"))
from common import expected_atoms, parse_atom_site, stem_parts  # noqa: E402
MOTIF = REPO / "files" / "motifs"
CONTROLS = MOTIF / "word_controls"
FIGURES = REPO / "figures"
DEFAULT_CIF_DIR = Path.home() / "Desktop" / "NR80_local"

FREQUENCY = CONTROLS / "frequency_matched_random_words.tsv"
FREQUENCY_SUMMARY = CONTROLS / "frequency_matched_random_words_summary.tsv"
BASE = CONTROLS / "base_inclusive_rmsd.tsv"
BASE_SUMMARY = CONTROLS / "base_inclusive_rmsd_summary.tsv"
RANDOM = MOTIF / "random_control_rmsd_production_new_k4_aa.tsv"
SEQUENCE = MOTIF / "global_sequence_matched_control_production_new_k4_aa.tsv"
PAIRING = CONTROLS / "pairing_pattern_words.tsv"
BACKGROUND_SUMMARY = CONTROLS / "bgsu_random_window_background_summary.tsv"
HAIRPIN = MOTIF / "overlapping_hairpin_44-48-62-71-66-12_instances.tsv"
DSSR = MOTIF / "known_motif_core_dssr_pair_conservation_instances_production_new.tsv"

# Palette of Figs 2-5 (figstyle): RFoldseek blue for the 3Dr words, Fig. 2's greys for
# controls, and the same tab10-derived hues, lightened, for everything else.
INK = "#333333"
GRID = "#DDDDDD"
BLUE, ORANGE = S.BASE["RFoldseek n500"], "#FF7F0E"

# A/B: the 3Dr words in RFoldseek blue, controls in neutral greys.
CANDIDATE = S.COLOR["RFoldseek n500"]
CONTROL_GREYS = {"frequency": "#8C8C8C", "random": "#B3B3B3", "sequence": "#D9D9D9"}

# C/E: the two overlapping words and the six GNRA positions, blue to orange.
WORD_COLORS = {"left": BLUE, "right": ORANGE}
POSITION_COLORS = [S.lighten(c, 0.1) for c in ("#1F9BD8", "#4FA8C9", "#86B4A6", "#C9A86B", "#F0913A", "#D9661F")]

# F: Leontis-Westhof classes.
LW_COLORS = {"cWW": S.lighten("#9467BD"), "tSH": S.lighten("#E377C2"), "tWH": S.lighten("#17BECF"),
             "tWS": S.lighten("#BCBD22"), ".": "#E6E6E6"}
TLOOP_COLOR = S.lighten("#2CA02C", 0.1)   # G: one hue, so its positions are not read as the GNRA's
LOGO_COLOR = "#A6A6A6"                     # D: one colour; the letters name the nucleotides


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_metrics(path: Path) -> dict[str, str]:
    return {row["metric"]: row["value"] for row in read_tsv(path)}


def numeric(rows: list[dict[str, str]], column: str) -> np.ndarray:
    return np.asarray([float(row[column]) for row in rows if row[column] not in {"", "nan", "NaN"}], dtype=float)


def panel_label(ax, label: str, top=None, left=None, dy=0.0) -> None:
    """Panel letter a fixed distance left of the axes' top-left corner, as in Figs 2-5;
    `top`/`left` take that edge from another axes, `dy` shifts the letter (inches)."""
    fig = ax.figure
    w, h = fig.get_size_inches()
    x0 = (left or ax).get_position().x0
    y1 = (top or ax).get_position().y1
    fig.text(x0 - 0.62 / w, y1 + (0.14 + dy) / h, label, fontsize=15, fontweight="bold",
             va="bottom", ha="right")


def style_axis(ax, title=None, xlabel=None, ylabel=None) -> None:
    S.style_axes(ax, title=title, xlabel=xlabel, ylabel=ylabel)


def subhead(ax, x, y, text, transform=None) -> None:
    """Grey in-panel heading, as in Fig. 5G."""
    ax.text(x, y, text, ha="center", va="bottom", fontsize=9.5, color=S.GREY_TEXT,
            transform=transform or ax.transAxes)


def atom_a_fragment(row: dict[str, str], n_residues: int, cif_dir: Path) -> np.ndarray:
    _pdb, model, chain = stem_parts(row["pdb_chain"])
    residues = parse_atom_site(cif_dir / f"{row['pdb_chain']}.cif", chain, model)
    coordinates = []
    for number in [int(value) for value in row["author_residues"].split(",")][:n_residues]:
        base, atoms = residues[number]
        names = expected_atoms(base)
        missing = [name for name in names if name not in atoms]
        if missing:
            raise RuntimeError(f"{row['pdb_chain']} residue {number}: missing {','.join(missing)}")
        coordinates.extend(atoms[name] for name in names)
    return np.asarray(coordinates, dtype=np.float64)


def kabsch_align(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    target = reference - reference.mean(axis=0, keepdims=True)
    u, _, vt = np.linalg.svd(centered.T @ target)
    reflection = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, reflection]) @ u.T
    return centered @ rotation.T


def preflight(cif_dir: Path) -> dict:
    frequency = read_tsv(FREQUENCY)
    base = read_tsv(BASE)
    random_rows = read_tsv(RANDOM)
    sequence_rows = [row for row in read_tsv(SEQUENCE) if row["status"] == "ok"]
    pairing = read_tsv(PAIRING)
    background = read_metrics(BACKGROUND_SUMMARY)
    hairpin = read_tsv(HAIRPIN)
    dssr = read_tsv(DSSR)

    a_distributions = {
        "five_candidate": numeric(frequency, "candidate_all_pair_median_rmsd"),
        "five_frequency": numeric(frequency, "matched_control_median_rmsd"),
        "five_random": numeric(random_rows, "rmsd"),
        "five_sequence": numeric(sequence_rows, "control_median_of_medians"),
        "atom_candidate": numeric(base, "atom_set_A_candidate_sampled_pair_median_rmsd"),
        "atom_random": numeric(base, "atom_set_A_length_random_median_rmsd"),
        "atom_sequence": numeric(base, "atom_set_A_sequence_matched_median_rmsd"),
    }
    a_medians = {key: float(np.median(values)) for key, values in a_distributions.items()}
    frequency_summary = read_metrics(FREQUENCY_SUMMARY)
    base_summary = read_metrics(BASE_SUMMARY)
    expected_a = {
        "five_candidate": float(frequency_summary["candidate_all_pair_median_rmsd"]),
        "five_frequency": float(frequency_summary["frequency_matched_control_median_rmsd"]),
        "five_random": float(base_summary["five_anchor_length_random_median_rmsd"]),
        "five_sequence": float(base_summary["five_anchor_sequence_matched_median_rmsd"]),
        "atom_candidate": float(base_summary["atom_set_A_candidate_median_rmsd"]),
        "atom_random": float(base_summary["atom_set_A_length_random_median_rmsd"]),
        "atom_sequence": float(base_summary["atom_set_A_sequence_matched_median_rmsd"]),
    }
    for key in a_medians:
        if not np.isclose(a_medians[key], expected_a[key], atol=1e-12, rtol=1e-10):
            raise SystemExit(f"STOP: panel A {key} median {a_medians[key]} != summary {expected_a[key]}")

    helix = [row for row in pairing if row["word_category"] == "a" and row["bgsu_overlap_fraction"] not in {"", "nan", "NaN"}]
    other = [row for row in pairing if row["word_category"] in {"b", "c", "d"} and row["bgsu_overlap_fraction"] not in {"", "nan", "NaN"}]
    if (len(helix), len(other)) != (514, 419):
        raise SystemExit(f"STOP: panel B n={(len(helix), len(other))}, expected (514, 419)")

    sequences = [row["sequence"].upper().replace("T", "U") for row in hairpin]
    n_gnra = sum(len(seq) == 6 and seq[1] == "G" and seq[3] in "AG" and seq[4] == "A" for seq in sequences)
    n_gc = sum(len(seq) == 6 and seq[0] + seq[5] in {"GC", "CG"} for seq in sequences)
    if (n_gnra, len(sequences), n_gc) != (44, 48, 37):
        raise SystemExit(f"STOP: panel D counts={(n_gnra, len(sequences), n_gc)}, expected (44, 48, 37)")

    exact_hairpin = [row for row in hairpin if row["dssr_lw_signature"]]
    n_gnra_signature = sum(row["dssr_lw_signature"] == "cWW,tSH,.,.,tSH,cWW" for row in exact_hairpin)
    tloop = [row for row in dssr if row["kmer"] == "48-24-39-39"]
    n_tloop_signature = sum(row["dssr_lw_pattern"] == "tWH,tWS,cWW,." for row in tloop)
    if (n_gnra_signature, len(exact_hairpin), n_tloop_signature, len(tloop)) != (11, 12, 10, 10):
        raise SystemExit("STOP: panel G signature counts do not equal 11/12 and 10/10")

    topology_count = sum(
        row["left_pair_orient_pattern"].split(",")[0] == "cis"
        and row["left_pair_orient_pattern"].split(",")[1] == "trans"
        and row["right_pair_orient_pattern"].split(",")[2] == "trans"
        and row["right_pair_orient_pattern"].split(",")[3] == "cis"
        for row in hairpin
    )
    if topology_count != len(hairpin):
        raise SystemExit(f"STOP: panel C topology count {topology_count}/{len(hairpin)}")

    if not cif_dir.is_dir():
        raise SystemExit(f"STOP: missing mmCIF directory {cif_dir}")
    gnra_coords = [atom_a_fragment(row, 6, cif_dir) for row in exact_hairpin]
    tloop_coords = [atom_a_fragment(row, 4, cif_dir) for row in tloop]

    return {
        "a_distributions": a_distributions,
        "a_medians": a_medians,
        "helix_values": np.asarray([float(row["bgsu_overlap_fraction"]) for row in helix]),
        "other_values": np.asarray([float(row["bgsu_overlap_fraction"]) for row in other]),
        "background": float(background["background_overlap_median_fraction"]),
        "jitter_seed": int(background["random_seed"]),
        "hairpin": hairpin,
        "sequences": sequences,
        "n_gnra": n_gnra,
        "n_gc": n_gc,
        "topology_count": topology_count,
        "exact_hairpin": exact_hairpin,
        "n_gnra_signature": n_gnra_signature,
        "tloop": tloop,
        "n_tloop_signature": n_tloop_signature,
        "gnra_coords": gnra_coords,
        "tloop_coords": tloop_coords,
    }


POOLED = {"five_random"}   # A: a single pooled value, not one value per word


def boxplot(ax, values: list[np.ndarray], positions: list[float], colors: list[str],
            widths: float = 0.55, alpha: float = 0.90) -> None:
    parts = ax.boxplot(values, positions=positions, widths=widths, patch_artist=True,
                       showfliers=False, whis=(10, 90),
                       medianprops={"color": INK, "linewidth": 1.7},
                       whiskerprops={"color": INK, "linewidth": 0.9},
                       capprops={"color": INK, "linewidth": 0.9},
                       boxprops={"edgecolor": INK, "linewidth": 0.9})
    for patch, color in zip(parts["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(alpha)


def panel_a(ax, data: dict) -> None:
    keys = ["five_candidate", "five_frequency", "five_random", "five_sequence",
            "atom_candidate", "atom_random", "atom_sequence"]
    labels = ["3Dr\nword", "Freq.\nmatched", "Random", "Seq.\nmatched",
              "3Dr\nword", "Random", "Seq.\nmatched"]
    positions = [0.8, 2.0, 3.2, 4.4, 6.2, 7.4, 8.6]
    colors = [CANDIDATE, CONTROL_GREYS["frequency"], CONTROL_GREYS["random"], CONTROL_GREYS["sequence"],
              CANDIDATE, CONTROL_GREYS["random"], CONTROL_GREYS["sequence"]]
    # The five-anchor random control is one pooled median over fragment pairs, not a
    # per-word distribution, so it is drawn as a single point.
    boxed = [i for i, key in enumerate(keys) if key not in POOLED]
    boxplot(ax, [data["a_distributions"][keys[i]] for i in boxed], [positions[i] for i in boxed],
            [colors[i] for i in boxed])
    for i, key in enumerate(keys):
        if key in POOLED:
            ax.scatter([positions[i]], [data["a_medians"][key]], s=46, marker="D", color=colors[i],
                       edgecolors=INK, linewidths=0.9, zorder=3)
    ax.axvline(5.3, color=GRID, lw=0.9, zorder=1)
    subhead(ax, 2.6, 0.95, "Five anchors", ax.get_xaxis_transform())
    subhead(ax, 7.4, 0.95, "Atom set A", ax.get_xaxis_transform())
    for x, key in zip(positions, keys):
        top = data["a_medians"][key] if key in POOLED else float(np.percentile(data["a_distributions"][key], 90))
        ax.text(x, top + 0.14, f"{data['a_medians'][key]:.2f}",
                color=S.TEXT["RFoldseek n500"] if "candidate" in key else S.GREY_TEXT,
                ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8.5, linespacing=1.0)
    ax.set_xlim(0.15, 9.15)
    ax.set_ylim(0, 4.8)
    style_axis(ax, ylabel=r"Median RMSD, $\AA$")


def panel_b(ax, data: dict) -> None:
    groups = [data["helix_values"], data["other_values"]]
    rng = np.random.default_rng(data["jitter_seed"])
    ax.axhline(data["background"], color=S.NEUTRAL_LIGHT, lw=1.4, ls="--", zorder=0)
    for x, values in zip([1, 2], groups):
        jitter = rng.uniform(-0.28, 0.28, len(values))
        ax.scatter(x + jitter, values, s=9, color=CANDIDATE, alpha=0.30, edgecolors="none", rasterized=True, zorder=1)
        median = float(np.median(values))
        ax.plot([x - 0.36, x + 0.36], [median, median], color=INK, lw=2.2, solid_capstyle="butt", zorder=3)
        ax.text(x + 0.40, median, f"{median * 100:.0f}%", ha="left", va="center",
                color=S.TEXT["RFoldseek n500"], fontsize=9)
    ax.text(1.5, data["background"] + 0.015, f"background\n{data['background'] * 100:.1f}%",
            ha="center", va="bottom", fontsize=8.5, color=S.GREY_TEXT)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Standard helix", "All other words"])
    ax.set_xlim(0.55, 2.68)
    ax.set_ylim(-0.04, 1.1)
    style_axis(ax, ylabel="BGSU motif overlap fraction")


def panel_c(ax, data: dict) -> None:
    """Arc diagram of the two overlapping 3Dr words on the six-nucleotide GNRA hairpin."""
    from matplotlib.patches import Arc
    first = data["hairpin"][0]
    left, right = first["left_kmer"].split("-"), first["right_kmer"].split("-")
    tokens = left + right[2:]
    w, h = 0.76, 0.62
    for x, token in enumerate(tokens):
        ax.add_patch(FancyBboxPatch((x - w / 2, -h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                    facecolor="white", edgecolor=INK, lw=1.1, zorder=3))
        ax.text(x, 0, token, ha="center", va="center", fontsize=11, color=INK, zorder=4)
        ax.text(x, -h / 2 - 0.16, str(x + 1), ha="center", va="top", fontsize=9, color=S.GREY_TEXT)
    # the two words: brackets above the token track
    for (start, end), y, color, word, ha in (((0, 3), 0.62, WORD_COLORS["left"], first["left_kmer"], "left"),
                                             ((2, 5), 1.36, WORD_COLORS["right"], first["right_kmer"], "right")):
        x0, x1 = start - w / 2, end + w / 2
        ax.plot([x0, x0, x1, x1], [y - 0.14, y, y, y - 0.14], color=color, lw=1.6, solid_capstyle="butt")
        ax.text(x0 if ha == "left" else x1, y + 0.10, word.replace("-", "–"), ha=ha, va="bottom",
                fontsize=9.5, color=color)
    # base pairs: arcs below the track
    y0 = -h / 2 - 0.55
    for i, j, label in ((0, 5, "cis"), (1, 4, "trans")):
        span = j - i
        ax.add_patch(Arc(((i + j) / 2, y0), span, span * 0.62, theta1=180, theta2=360, color=INK, lw=1.3))
        ax.text((i + j) / 2, y0 - span * 0.31 - 0.12, label, ha="center", va="top",
                fontsize=9, color=INK)
    ax.set_xlim(-0.75, 5.75)
    ax.set_ylim(y0 - 5 * 0.31 - 0.55, 2.05)
    ax.set_aspect("equal")
    ax.set_anchor("C")
    ax.axis("off")


def draw_logo(ax, probabilities: list[dict[str, float]], colors: dict[str, str], width: float = 0.9) -> None:
    """Probability sequence logo: letters stacked by probability, the most frequent at the bottom."""
    font = FontProperties(family="Arial", weight="bold")
    for position, column in enumerate(probabilities):
        y = 0.0
        for base, p in sorted(column.items(), key=lambda kv: -kv[1]):
            if p <= 0:
                continue
            path = TextPath((0, 0), base, size=1, prop=font)
            box = path.get_extents()
            shape = (Affine2D().translate(-box.x0, -box.y0)
                     .scale(width / box.width, p / box.height)
                     .translate(position - width / 2, y))
            ax.add_patch(PathPatch(shape.transform_path(path), facecolor=colors[base], edgecolor="white", lw=0.8))
            y += p


def panel_d(ax, data: dict) -> None:
    n = len(data["sequences"])
    probabilities = [{base: sum(seq[position] == base for seq in data["sequences"]) / n for base in "ACGU"}
                     for position in range(6)]
    draw_logo(ax, probabilities, dict.fromkeys("ACGU", LOGO_COLOR))
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(6))
    ax.set_xticklabels(["1", "2", "3", "4", "5", "6"])
    ax.set_yticks([0, 0.5, 1.0])
    style_axis(ax, xlabel="Hairpin position", ylabel="Probability")


def panel_e(ax, data: dict) -> None:
    rows = [
        (["cWW", "tSH", ".", ".", "tSH", "cWW"],
         f"{data['n_gnra_signature']}/{len(data['exact_hairpin'])}", "HL_34789.6"),
        (["tWH", "tWS", "cWW", "."],
         f"{data['n_tloop_signature']}/{len(data['tloop'])}", "HL_33597.6"),
    ]
    ax.set_xlim(0, 8.7)
    ax.set_ylim(-0.32, 1.32)
    ax.invert_yaxis()
    for y, (signature, count, label) in enumerate(rows):
        for x, state in enumerate(signature):
            ax.add_patch(Rectangle((x + 0.05, y - 0.27), 0.90, 0.54,
                                   facecolor=LW_COLORS[state], edgecolor="white", lw=1.2))
            ax.text(x + 0.50, y, state, ha="center", va="center", fontsize=8.5, color=INK)
        ax.text(6.35, y - 0.12, count, ha="left", va="center", fontsize=10)
        ax.text(6.35, y + 0.16, label, ha="left", va="center", fontsize=8.5, color=S.GREY_TEXT)
    ax.set_xticks(np.arange(6) + 0.50)
    ax.set_xticklabels(["1", "2", "3", "4", "5", "6"])
    ax.set_yticks([])
    style_axis(ax, xlabel="Position")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    handles = [Patch(facecolor=LW_COLORS[state], edgecolor="none", label=state) for state in ("cWW", "tSH", "tWH", "tWS")]
    ax.legend(handles=handles, loc="center right", bbox_to_anchor=(1.0, 0.5), ncol=1,
              frameon=False, fontsize=9.5, columnspacing=1.2, handlelength=1.1)


def align_to_medoid(fragments: list[np.ndarray]) -> np.ndarray:
    """Superpose every instance on the medoid (lowest mean RMSD to the others); medoid first."""
    n = len(fragments)
    rmsd = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            diff = kabsch_align(fragments[i], fragments[j]) - (fragments[j] - fragments[j].mean(axis=0))
            rmsd[i, j] = rmsd[j, i] = np.sqrt((diff ** 2).sum(axis=1).mean())
    medoid = int(np.argmin(rmsd.sum(axis=1)))
    order = [medoid] + [i for i in range(n) if i != medoid]
    return np.asarray([kabsch_align(fragments[i], fragments[medoid]) for i in order])


# Atom set A per residue: P OP1 OP2 O5' C5' C4' O4' C3' O3' C2' O2' C1', then three base locators.
P, O5, C5, C4, O4, C3, O3, C2, C1 = 0, 3, 4, 5, 6, 7, 8, 9, 11
BASE_TRI = slice(12, 15)


BASE_SCALE = 2.2   # base triangles enlarged about the glycosidic atom to roughly the base's size


def base_triangle(residue: np.ndarray) -> np.ndarray:
    tri = residue[BASE_TRI]
    return tri[0] + (tri - tri[0]) * BASE_SCALE


def backbone_trace(residues: np.ndarray) -> np.ndarray:
    """P-O5'-C5'-C4'-C3'-O3' through every residue."""
    return np.vstack([r[[P, O5, C5, C4, C3, O3]] for r in residues])


def draw_atom_a_overlay(ax, aligned: np.ndarray, title: str, colors: list[str]) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    n_residues = aligned.shape[1] // 15
    for fragment in aligned[1:]:                       # other instances: thin translucent lines
        residues = fragment.reshape(n_residues, 15, 3)
        trace = backbone_trace(residues)
        ax.plot(*trace.T, color=S.NEUTRAL_LIGHT, alpha=0.45, lw=0.6)
        for position, residue in enumerate(residues):
            color = colors[position]
            tri = base_triangle(residue)
            tri = np.vstack([tri, tri[:1]])
            ax.plot(*np.vstack([residue[C1], residue[12]]).T, color=color, alpha=0.35, lw=0.6)
            ax.plot(*tri.T, color=color, alpha=0.35, lw=0.6)
    medoid = aligned[0].reshape(n_residues, 15, 3)
    for position, residue in enumerate(medoid):        # medoid: thick backbone, filled base triangles
        color = colors[position]
        segment = residue[[P, O5, C5, C4, C3, O3]]
        if position + 1 < n_residues:
            segment = np.vstack([segment, medoid[position + 1][P]])
        ax.plot(*segment.T, color=color, lw=2.6, solid_capstyle="round")
        ax.plot(*residue[[C4, O4, C1, C2, C3]].T, color=color, lw=1.2)
        ax.plot(*np.vstack([residue[C1], residue[12]]).T, color=color, lw=1.8)
        ax.add_collection3d(Poly3DCollection([base_triangle(residue)], facecolor=color, edgecolor=INK,
                                             linewidths=0.6, alpha=0.85))
    ax.text2D(0.5, 0.98, title, transform=ax.transAxes, ha="center", va="top", fontsize=10.5)
    if len(set(colors)) > 1:                           # position key only when positions are coloured
        legend_x0 = 0.5 - (n_residues - 1) * 0.042
        for position in range(n_residues):
            ax.text2D(legend_x0 + position * 0.084, 0.90, str(position + 1), transform=ax.transAxes,
                      ha="center", va="center", fontsize=9, color=colors[position])
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1), zoom=1.75)
    ax.view_init(elev=24, azim=-62)


def same_physical_limits(axes: list, aligned_sets: list[np.ndarray]) -> None:
    full_span = max(float(np.ptp(aligned.reshape(-1, 3), axis=0).max()) for aligned in aligned_sets)
    half_range = full_span * 0.55
    for ax, aligned in zip(axes, aligned_sets):
        center = aligned.reshape(-1, 3).mean(axis=0)
        ax.set_xlim(center[0] - half_range, center[0] + half_range)
        ax.set_ylim(center[1] - half_range, center[1] + half_range)
        ax.set_zlim(center[2] - half_range, center[2] + half_range)


def add_scale_bar(ax, length: float = 2.0) -> None:
    center = np.asarray([np.mean(ax.get_xlim3d()), np.mean(ax.get_ylim3d()), np.mean(ax.get_zlim3d())])
    projected = proj3d.proj_transform(*center, ax.get_proj())
    origin = ax.transData.transform(projected[:2])
    pixels = []
    for delta in np.eye(3) * length:
        endpoint = proj3d.proj_transform(*(center + delta), ax.get_proj())
        pixels.append(np.linalg.norm(ax.transData.transform(endpoint[:2]) - origin))
    fraction = max(pixels) / ax.get_window_extent().width
    x0, y0 = 0.08, 0.10
    ax.add_line(Line2D([x0, x0 + fraction], [y0, y0], transform=ax.transAxes, color=INK, lw=2.1))
    ax.text2D(x0 + fraction / 2, y0 - 0.025, r"2 $\AA$", transform=ax.transAxes, ha="center", va="top", fontsize=9)


def panel_f(fig, cells, data: dict) -> tuple[list, list[np.ndarray]]:
    """The two Atom-set-A superpositions, one per grid cell."""
    gnra = align_to_medoid(data["gnra_coords"])
    tloop = align_to_medoid(data["tloop_coords"])
    axes = [fig.add_subplot(cells[0], projection="3d"), fig.add_subplot(cells[1], projection="3d")]
    draw_atom_a_overlay(axes[0], gnra, "GNRA hairpin", POSITION_COLORS)
    draw_atom_a_overlay(axes[1], tloop, "T-loop core", [TLOOP_COLOR] * 4)
    same_physical_limits(axes, [gnra, tloop])
    return axes, [gnra, tloop]


def write_numbers(data: dict) -> Path:
    rows = []
    def add(panel, item, value, source, column, statistic):
        rows.append({"panel": panel, "item": item, "value": value, "source_file": str(source.relative_to(REPO)),
                     "source_column": column, "statistic_or_rule": statistic})
    a_sources = {
        "five_candidate": (FREQUENCY, "candidate_all_pair_median_rmsd"),
        "five_frequency": (FREQUENCY, "matched_control_median_rmsd"),
        "five_random": (RANDOM, "rmsd"),
        "five_sequence": (SEQUENCE, "control_median_of_medians"),
        "atom_candidate": (BASE, "atom_set_A_candidate_sampled_pair_median_rmsd"),
        "atom_random": (BASE, "atom_set_A_length_random_median_rmsd"),
        "atom_sequence": (BASE, "atom_set_A_sequence_matched_median_rmsd"),
    }
    for key, value in data["a_medians"].items():
        source, column = a_sources[key]
        add("A", key, value, source, column, "median")
    add("B", "standard_helix_n", len(data["helix_values"]), PAIRING, "word_category,bgsu_overlap_fraction", "category a with finite BGSU denominator")
    add("B", "other_words_n", len(data["other_values"]), PAIRING, "word_category,bgsu_overlap_fraction", "categories b+c+d with finite BGSU denominator")
    add("B", "standard_helix_median_percent", np.median(data["helix_values"]) * 100, PAIRING, "bgsu_overlap_fraction", "median × 100")
    add("B", "other_words_median_percent", np.median(data["other_values"]) * 100, PAIRING, "bgsu_overlap_fraction", "median × 100")
    add("B", "random_background_percent", data["background"] * 100, BACKGROUND_SUMMARY, "value where metric=background_overlap_median_fraction", "× 100")
    add("C", "six_token_string", data["hairpin"][0]["left_kmer"] + "-" + "-".join(data["hairpin"][0]["right_kmer"].split("-")[2:]), HAIRPIN, "left_kmer,right_kmer", "overlap positions 3–4")
    add("C", "cis_trans_topology_count", data["topology_count"], HAIRPIN, "left_pair_orient_pattern,right_pair_orient_pattern", "cis 1↔6 and trans 2↔5")
    add("C", "topology_denominator", len(data["hairpin"]), HAIRPIN, "sequence", "row count")
    add("D", "GNRA_loop_count", data["n_gnra"], HAIRPIN, "sequence", "positions 2–5 match GNRA")
    add("D", "GC_closing_pair_count", data["n_gc"], HAIRPIN, "sequence", "positions 1 and 6 are G–C/C–G")
    add("D", "sequence_count", len(data["sequences"]), HAIRPIN, "sequence", "row count")
    add("E", "GNRA_superposition_n", len(data["gnra_coords"]), HAIRPIN, "pdb_chain,author_residues,dssr_lw_signature", "successfully parsed Atom set A")
    add("E", "scale_bar_angstrom", 2, HAIRPIN, "pdb_chain,author_residues", "physical coordinate scale")
    add("F", "T_loop_superposition_n", len(data["tloop_coords"]), DSSR, "pdb_chain,author_residues,kmer", "successfully parsed Atom set A")
    add("F", "scale_bar_angstrom", 2, DSSR, "pdb_chain,author_residues", "physical coordinate scale")
    add("G", "GNRA_signature_count", data["n_gnra_signature"], HAIRPIN, "dssr_lw_signature", "equals cWW,tSH,.,.,tSH,cWW")
    add("G", "GNRA_signature_denominator", len(data["exact_hairpin"]), HAIRPIN, "dssr_lw_signature", "non-empty")
    add("G", "T_loop_signature_count", data["n_tloop_signature"], DSSR, "kmer,dssr_lw_pattern", "kmer 48-24-39-39; equals tWH,tWS,cWW,.")
    add("G", "T_loop_signature_denominator", len(data["tloop"]), DSSR, "kmer", "kmer 48-24-39-39 row count")
    path = FIGURES / "fig6_numbers.tsv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cif-dir", type=Path, default=DEFAULT_CIF_DIR)
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "Arial", "font.sans-serif": ["Arial"], "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    data = preflight(args.cif_dir)
    FIGURES.mkdir(parents=True, exist_ok=True)

    # Row 1: A (wide) and B. Row 2, the GNRA example: word diagram, sequences, structure.
    # Row 3: base-pairing signatures of both motifs and the T-loop structure.
    fig = plt.figure(figsize=(14.4, 9.2), constrained_layout=False)
    grid = fig.add_gridspec(3, 3, width_ratios=[1.0, 1.0, 1.0], height_ratios=[1.0, 1.0, 0.55],
                            hspace=0.30, wspace=0.26)
    ax_a = fig.add_subplot(grid[0, 0:2])
    ax_b = fig.add_subplot(grid[0, 2])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])
    ax_e = fig.add_subplot(grid[2, 0:2])
    right = grid[1:3, 2].subgridspec(2, 1, hspace=0.05)      # the two structures share the right column equally
    f_axes, _aligned = panel_f(fig, (right[0], right[1]), data)
    panel_a(ax_a, data)
    panel_b(ax_b, data)
    panel_c(ax_c, data)
    panel_d(ax_d, data)
    panel_e(ax_e, data)

    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.06, top=0.97)
    shift = 0.035                            # right column moved in towards A and D
    pb = ax_b.get_position()                 # a narrower B
    ax_b.set_position([pb.x1 - pb.width * 0.8 - shift, pb.y0, pb.width * 0.8, pb.height])
    for ax in f_axes:
        p = ax.get_position(original=True)
        ax.set_position([p.x0 - shift, p.y0, p.width, p.height])
    fig.canvas.draw()
    for ax in f_axes:
        add_scale_bar(ax, 2.0)
    # letters in reading order, row by row
    panel_label(ax_a, "A")
    panel_label(ax_b, "B")
    panel_label(ax_c, "C", top=ax_d, dy=-0.12)
    panel_label(ax_d, "D", dy=-0.12)
    panel_label(f_axes[0], "E", top=ax_d, left=ax_b, dy=-0.12)   # GNRA structure
    panel_label(f_axes[1], "F", left=ax_b)                        # T-loop structure
    panel_label(ax_e, "G")                                       # interaction signatures

    png = FIGURES / "fig6.png"
    pdf = FIGURES / "fig6.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    numbers = write_numbers(data)
    print(f"FIG6_CAPTION_N_GNRA={len(data['gnra_coords'])}")
    print(f"FIG6_CAPTION_N_TLOOP={len(data['tloop_coords'])}")
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    print(f"Wrote {numbers}")


if __name__ == "__main__":
    main()
