"""Draw Fig. 6 from the recurrent-word control tables."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Arc, PathPatch, Rectangle
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "common"))
import figstyle as S  # (palette and type sizes shared with Figs 2-5)
sys.path.insert(0, str(REPO / "scripts" / "07_motifs" / "word_controls"))
from common import expected_atoms, parse_atom_site, stem_parts
sys.path.insert(0, str(REPO / "scripts" / "07_motifs"))
from known_motif_core_dssr_annotation import lw_pattern_contains, pair_lookup  
from residue_keys import parse_key_string, parse_number_and_insertion 
MOTIF = REPO / "files" / "motifs"
CONTROLS = MOTIF / "word_controls"
FIGURES = REPO / "figures"
DEFAULT_CIF_DIR = MOTIF / "fig6_structures"   # RNAsolo chains of the F/G fragments
RENDER_DIR = MOTIF / "fig6_render"            # ChimeraX renders of F/G (render_fig6_structures.py)

BASE = CONTROLS / "base_inclusive_rmsd.tsv"
BASE_SUMMARY = CONTROLS / "base_inclusive_rmsd_summary.tsv"
PAIRING = CONTROLS / "pairing_pattern_words.tsv"
BGSU_SENSITIVITY = CONTROLS / "bgsu_candidate_overlap_sensitivity.tsv"
BACKGROUND_SUMMARY = CONTROLS / "bgsu_random_window_background_summary.tsv"
BACKGROUND_REPLICATES = CONTROLS / "bgsu_random_window_background.tsv"
HAIRPIN = MOTIF / "overlapping_hairpin_44-48-62-71-66-12_instances.tsv"
DSSR = MOTIF / "known_motif_core_dssr_pair_conservation_instances_production_new.tsv"

# Palette of Figs 2-5 (figstyle): RFoldSearch blue for the 3Dr words, Fig. 2's greys for
# controls, and the same tab10-derived hues, lightened, for everything else.
INK = "#333333"
GRID = "#DDDDDD"
BLUE, ORANGE = S.BASE["RFoldSearch n500"], "#FF7F0E"

# A/B: the 3Dr words in RFoldSearch blue, controls in neutral greys.
CANDIDATE = S.COLOR["RFoldSearch n500"]
CONTROL_GREYS = {"frequency": "#8C8C8C", "random": "#B3B3B3", "sequence": "#D9D9D9"}

# C/E: the two overlapping words and the six GNRA positions, blue to orange.
WORD_COLORS = {"left": BLUE, "right": ORANGE}
POSITION_COLORS = [S.lighten(c, 0.1) for c in ("#1F9BD8", "#4FA8C9", "#86B4A6", "#C9A86B", "#F0913A", "#D9661F")]

# G/H: Leontis-Westhof classes, read from each token nucleotide to its partner; the two
# readings of one pair (tSH/tHS) share a colour.
LW_COLORS = {"cWW": S.lighten("#9467BD"), "tSH": S.lighten("#E377C2"), "tWH": S.lighten("#17BECF"),
             "tWS": S.lighten("#BCBD22"), ".": "#E6E6E6"}
LW_COLORS["tHS"], LW_COLORS["tSW"] = LW_COLORS["tSH"], LW_COLORS["tWS"]
GNRA_SIGNATURE = "cWW,tSH,.,.,tHS,cWW"
TLOOP_SIGNATURE = "tWH,tSW,cWW,."
TLOOP_COLOR = S.lighten("#2CA02C", 0.1)
PARTNER_COLOR = "#7A7A7A"   # F: residues outside the token window that pair with it   # G: one hue, so its positions are not read as the GNRA's
LOGO_COLOR = "#A6A6A6"      # D: one colour; the letters name the nucleotides


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


def base_inclusive_fragment(row: dict[str, str], n_residues: int, cif_dir: Path) -> np.ndarray:
    _pdb, model, chain = stem_parts(row["pdb_chain"])
    use_full_keys = bool(row.get("residue_keys"))
    residues = parse_atom_site(cif_dir / f"{row['pdb_chain']}.cif", chain, model, full_keys=True)
    if use_full_keys:
        keys = [
            (key[4], key[5])
            for key in map(parse_key_string, row["residue_keys"].split(","))
        ][:n_residues]
    else:
        keys = [
            parse_number_and_insertion(value)
            for value in row["author_residues"].split(",")
        ][:n_residues]
    coordinates = []
    for residue_key in keys:
        base, atoms = residues[residue_key]
        names = expected_atoms(base)
        missing = [name for name in names if name not in atoms]
        if missing:
            raise RuntimeError(
                f"{row['pdb_chain']} residue {residue_key}: missing {','.join(missing)}"
            )
        coordinates.extend(atoms[name] for name in names)
    return np.asarray(coordinates, dtype=np.float64)


def pairing_partners(row: dict[str, str], n_residues: int, cif_dir: Path) -> list[dict]:
    """DSSR pairing records of the token residues whose partner lies outside the window:
    token position, LW label, partner label and the partner's base-inclusive atoms."""
    keys = [parse_key_string(value) for value in row["residue_keys"].split(",")][:n_residues]
    pdb, model, chain = keys[0][0], keys[0][1], keys[0][2]
    pairs = pair_lookup(json.loads(Path(row["dssr_json"]).read_text()), core_chain=chain, pdb=pdb, model=model)
    residues = parse_atom_site(cif_dir / f"{row['pdb_chain']}.cif", chain, model, full_keys=True)
    window = set(keys)
    partners = []
    for position, key in enumerate(keys, start=1):
        for record in pairs.get(key, []):
            partner = tuple(record["partner_ref"])
            if partner in window:
                continue
            base, atoms = residues[(partner[4], partner[5])]
            partners.append({
                "position": position, "lw": str(record.get("LW", ".")),
                "label": f"{partner[3]}{partner[4]}{partner[5]}",
                "coords": np.asarray([atoms[name] for name in expected_atoms(base)], dtype=np.float64),
            })
    return partners


def superposition(points: np.ndarray, reference: np.ndarray):
    """Rotation and centres that superpose `points` on `reference` (as kabsch_align)."""
    mu, ref_mu = points.mean(axis=0), reference.mean(axis=0)
    u, _, vt = np.linalg.svd((points - mu).T @ (reference - ref_mu))
    reflection = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, reflection]) @ u.T, mu


def kabsch_align(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    target = reference - reference.mean(axis=0, keepdims=True)
    u, _, vt = np.linalg.svd(centered.T @ target)
    reflection = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, reflection]) @ u.T
    return centered @ rotation.T


def preflight(cif_dir: Path) -> dict:
    base = read_tsv(BASE)
    pairing = read_tsv(PAIRING)
    bgsu_sensitivity = read_tsv(BGSU_SENSITIVITY)
    background = read_metrics(BACKGROUND_SUMMARY)
    hairpin = read_tsv(HAIRPIN)
    dssr = read_tsv(DSSR)

    # Both atom sets carry the same three controls: frequency-matched words, length-matched
    # random windows and sequence-matched windows.
    a_distributions = {
        # All eight distributions come from 02 and share one window universe: labelled
        # occurrences with a unique residue-key mapping and complete base-inclusive atoms.
        "five_candidate": numeric(base, "five_anchor_candidate_cross_clan_median_rmsd"),
        "five_frequency": numeric(base, "five_anchor_frequency_matched_cross_clan_median_rmsd"),
        "five_random": numeric(base, "five_anchor_length_random_cross_clan_median_rmsd"),
        "five_sequence": numeric(base, "five_anchor_sequence_matched_cross_clan_median_rmsd"),
        "base_candidate": numeric(base, "base_inclusive_candidate_cross_clan_median_rmsd"),
        "base_frequency": numeric(base, "base_inclusive_frequency_matched_cross_clan_median_rmsd"),
        "base_random": numeric(base, "base_inclusive_length_random_cross_clan_median_rmsd"),
        "base_sequence": numeric(base, "base_inclusive_sequence_matched_cross_clan_median_rmsd"),
    }
    a_medians = {key: float(np.median(values)) for key, values in a_distributions.items()}
    base_summary = read_metrics(BASE_SUMMARY)
    expected_a = {
        "five_candidate": float(base_summary["five_anchor_candidate_median_rmsd"]),
        "five_frequency": float(base_summary["five_anchor_frequency_matched_median_rmsd"]),
        "five_random": float(base_summary["five_anchor_length_random_median_rmsd"]),
        "five_sequence": float(base_summary["five_anchor_sequence_matched_median_rmsd"]),
        "base_candidate": float(base_summary["base_inclusive_candidate_median_rmsd"]),
        "base_frequency": float(base_summary["base_inclusive_frequency_matched_median_rmsd"]),
        "base_random": float(base_summary["base_inclusive_length_random_median_rmsd"]),
        "base_sequence": float(base_summary["base_inclusive_sequence_matched_median_rmsd"]),
    }
    for key in a_medians:
        if not np.isclose(a_medians[key], expected_a[key], atol=1e-12, rtol=1e-10):
            raise SystemExit(f"STOP: panel A {key} median {a_medians[key]} != summary {expected_a[key]}")

    category = {row["kmer"]: row["word_category"] for row in pairing}
    helix = [row for row in bgsu_sensitivity if category.get(row["kmer"]) == "a" and row["fraction_ge_1"] not in {"", "nan", "NaN"}]
    other = [row for row in bgsu_sensitivity if category.get(row["kmer"]) in {"b", "c", "d"} and row["fraction_ge_1"] not in {"", "nan", "NaN"}]
    expected_b = (int(background["threshold_1_group_a_n_words"]), int(background["threshold_1_group_nonhelix_n_words"]))
    if (len(helix), len(other)) != expected_b:
        raise SystemExit(f"STOP: panel B n={(len(helix), len(other))}, expected {expected_b}")

    sequences = [row["sequence"].upper().replace("T", "U") for row in hairpin]
    n_gnra = sum(len(seq) == 6 and seq[1] == "G" and seq[3] in "AG" and seq[4] == "A" for seq in sequences)
    n_gc = sum(len(seq) == 6 and seq[0] + seq[5] in {"GC", "CG"} for seq in sequences)
    if (n_gnra, len(sequences), n_gc) != (44, 48, 37):
        raise SystemExit(f"STOP: panel D counts={(n_gnra, len(sequences), n_gc)}, expected (44, 48, 37)")

    exact_hairpin = [
        row for row in hairpin
        if row["dssr_exact_bgsu_hl_34789_6"] == "1"
    ]
    # Signature rule: every listed interaction must be among the DSSR records at its
    # position (additional records are allowed); "." positions are not constrained.
    n_gnra_signature = sum(
        lw_pattern_contains(row["dssr_lw_signature"], GNRA_SIGNATURE)
        for row in exact_hairpin
    )
    tloop = [row for row in dssr if row["kmer"] == "48-24-39-39"]
    n_tloop_signature = sum(
        lw_pattern_contains(row["dssr_lw_pattern"], TLOOP_SIGNATURE)
        for row in tloop
    )
    if (n_gnra_signature, len(exact_hairpin), n_tloop_signature, len(tloop)) != (12, 13, 10, 10):
        raise SystemExit("STOP: panel H signature counts do not equal 12/13 and 10/10")

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
    gnra_coords = [base_inclusive_fragment(row, 6, cif_dir) for row in exact_hairpin]
    tloop_coords = [base_inclusive_fragment(row, 4, cif_dir) for row in tloop]
    tloop_partners = [pairing_partners(row, 4, cif_dir) for row in tloop]

    return {
        "a_distributions": a_distributions,
        "a_medians": a_medians,
        "a_permutation_p": float(base_summary["permutation_five_anchor_candidate_vs_frequency_control_p"]),
        "a_permutations": int(base_summary["permutation_five_anchor_candidate_vs_frequency_control_n_permutations"]),
        "a_base_permutation_p": float(base_summary["permutation_base_inclusive_candidate_vs_frequency_control_p"]),
        "a_base_permutations": int(base_summary["permutation_base_inclusive_candidate_vs_frequency_control_n_permutations"]),
        "helix_values": np.asarray([float(row["fraction_ge_1"]) for row in helix]),
        "other_values": np.asarray([float(row["fraction_ge_1"]) for row in other]),
        "bgsu_null": {
            "a": (
                float(background["threshold_1_group_a_null_word_median_p10_fraction"]),
                float(background["threshold_1_group_a_null_word_median_fraction"]),
                float(background["threshold_1_group_a_null_word_median_p90_fraction"]),
            ),
            "nonhelix": (
                float(background["threshold_1_group_nonhelix_null_word_median_p10_fraction"]),
                float(background["threshold_1_group_nonhelix_null_word_median_fraction"]),
                float(background["threshold_1_group_nonhelix_null_word_median_p90_fraction"]),
            ),
        },
        "bgsu_null_ci95": {
            group: tuple(float(value) for value in np.percentile([
                float(row[f"null_{group}_word_median_fraction"])
                for row in read_tsv(BACKGROUND_REPLICATES) if row["overlap_threshold_residues"] == "1"
            ], [2.5, 97.5]))
            for group in ("a", "nonhelix")
        },
        "bgsu_p": {
            "a": float(background["threshold_1_group_a_empirical_p_ge"]),
            "nonhelix": float(background["threshold_1_group_nonhelix_empirical_p_ge"]),
        },
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
        "tloop_partners": tloop_partners,
    }



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
    labels = ["3Dr\nword", "Frequency\nmatched", "Sequence\nmatched", "Random"]
    groups = [("five", 0.0), ("base", 5.0)]
    offsets = [0.9, 2.0, 3.1, 4.2]
    colors = [CANDIDATE, CONTROL_GREYS["frequency"], CONTROL_GREYS["sequence"], CONTROL_GREYS["random"]]
    suffixes = ["candidate", "frequency", "sequence", "random"]
    ticks, ticklabels = [], []
    for prefix, x0 in groups:
        keys = [f"{prefix}_{suffix}" for suffix in suffixes]
        positions = [x0 + off for off in offsets]
        boxplot(ax, [data["a_distributions"][key] for key in keys], positions, colors)
        for x, key in zip(positions, keys):
            top = float(np.percentile(data["a_distributions"][key], 90))
            ax.text(x, top + 0.14, f"{data['a_medians'][key]:.2f}",
                    color=S.TEXT["RFoldSearch n500"] if "candidate" in key else S.GREY_TEXT,
                    ha="center", va="bottom", fontsize=8.5)
        # Unique-word permutation test of the 3Dr word against its frequency-matched control.
        p_value, n_permutations = ((data["a_permutation_p"], data["a_permutations"]) if prefix == "five"
                                   else (data["a_base_permutation_p"], data["a_base_permutations"]))
        floor = 1 / (n_permutations + 1)
        exponent = int(round(np.log10(n_permutations)))
        p_label = rf"P < $10^{{-{exponent}}}$" if p_value <= floor + 1e-12 else f"P = {p_value:.2g}"
        bracket = max(float(np.percentile(data["a_distributions"][key], 90)) for key in keys[:2]) + 0.55
        ax.plot([positions[0], positions[0], positions[1], positions[1]],
                [bracket, bracket + 0.08, bracket + 0.08, bracket], color=INK, lw=0.8)
        ax.text((positions[0] + positions[1]) / 2, bracket + 0.12, p_label, ha="center", va="bottom",
                fontsize=8.2, color=S.GREY_TEXT)
        ticks += positions
        ticklabels += labels
    ax.axvline(5.0, color=GRID, lw=0.9, zorder=1)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels, fontsize=8.5, linespacing=1.0)
    ax.set_xlim(0.3, 9.8)
    ax.set_ylim(0, 4.8)
    style_axis(ax, ylabel="Median pairwise RMSD (Å)")


def panel_b(ax, data: dict) -> None:
    """Observed per-word median overlap versus the word-size-matched random expectation."""
    groups = ["a", "nonhelix"]
    observed = [float(np.median(data["helix_values"])), float(np.median(data["other_values"]))]
    expected = [data["bgsu_null"][group][1] for group in groups]
    low, high = zip(*(data["bgsu_null_ci95"][group] for group in groups))
    yerr = np.asarray([np.subtract(expected, low), np.subtract(high, expected)])
    x = np.arange(2)
    width = 0.36
    bar_style = dict(edgecolor="#666666", linewidth=0.6, zorder=3)
    ax.bar(x - width / 2, observed, width, color=CANDIDATE, label="3Dr words", **bar_style)
    ax.bar(x + width / 2, expected, width, color=CONTROL_GREYS["random"], label="Random expectation",
           yerr=yerr, capsize=2.5, error_kw={"elinewidth": 0.9, "ecolor": S.GREY_TEXT}, **bar_style)
    for xi, value in zip(x - width / 2, observed):
        ax.text(xi, value + 0.015, f"{value * 100:.0f}%", ha="center", va="bottom", fontsize=8.5,
                color=S.TEXT["RFoldSearch n500"])
    for xi, group, top_obs, top_exp in zip(x, groups, observed, high):
        top = max(top_obs + 0.08, top_exp + 0.04)
        p_value = data["bgsu_p"][group]
        if p_value >= 0.05:
            p_label = "n.s."
        else:
            p_label = f"P = {p_value:.3f}" if p_value >= 0.001 else "P < 0.001"
        ax.plot([xi - width / 2, xi - width / 2, xi + width / 2, xi + width / 2],
                [top, top + 0.02, top + 0.02, top], color=INK, lw=0.8)
        ax.text(xi, top + 0.03, p_label, ha="center", va="bottom", fontsize=8.2, color=S.GREY_TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(["Standard helix", "Other words"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 1.25)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0.0, 1.02))
    style_axis(ax, ylabel="Fraction overlapping BGSU motifs")


def panel_c(ax, data: dict) -> None:
    """Arc diagram of the two overlapping 3Dr words on the six-nucleotide GNRA hairpin; the
    strip under each token carries its position colour, as in E and F."""
    first = data["hairpin"][0]
    left, right = first["left_kmer"].split("-"), first["right_kmer"].split("-")
    tokens = left + right[2:]
    w = 0.74
    for x, token in enumerate(tokens):
        ax.add_patch(Rectangle((x - w / 2, -0.13), w, 0.32, facecolor="white", edgecolor="#555555", lw=0.8))
        ax.add_patch(Rectangle((x - w / 2, -0.13), w, 0.045, facecolor=POSITION_COLORS[x], edgecolor="none"))
        ax.text(x, 0.03, token, ha="center", va="center", fontsize=12)
    # the two words: brackets above the token track
    for (start, end), y, color, word in (((0, 3), 0.46, WORD_COLORS["left"], first["left_kmer"]),
                                         ((2, 5), 0.91, WORD_COLORS["right"], first["right_kmer"])):
        ax.plot([start - w / 2, start - w / 2, end + w / 2, end + w / 2], [y - 0.05, y, y, y - 0.05],
                color=color, lw=1.5)
        ax.text((start + end) / 2, y + 0.05, word.replace("-", "–"), ha="center", va="bottom",
                color=color, fontsize=10)
    # base pairs: arcs below the track
    for i, j, label in ((0, 5, "cis"), (1, 4, "trans")):
        span = j - i
        ax.add_patch(Arc(((i + j) / 2, -0.22), span, 0.25 * span, theta1=180, theta2=360, color="#555555", lw=1))
        ax.text((i + j) / 2, -0.22 - 0.125 * span - 0.03, label, ha="center", va="top", fontsize=9)
    ax.set_xlim(-0.65, 5.65)
    ax.set_ylim(-1.02, 1.22)
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
    ax.set_xlim(-0.65, 5.65)
    ax.set_ylim(0, 1.02)
    for x, color in enumerate(POSITION_COLORS):
        ax.add_patch(Rectangle((x - 0.37, -0.035), 0.74, 0.025, facecolor=color, edgecolor="none",
                               clip_on=False))
    ax.set_xticks(np.arange(6))
    ax.set_xticklabels(["1", "2", "3", "4", "5", "6"])
    ax.set_yticks([0, 0.5, 1.0])
    style_axis(ax, xlabel="Position in the six-token pattern", ylabel="Nucleotide frequency")


NO_REQUIRED_INTERACTION = "–"
OUTSIDE = "N/A"


def panel_e(ax, data: dict) -> None:
    """Panel H: required DSSR Leontis-Westhof label at each token position, read from the token
    nucleotide to its partner. A row counts the fragments whose records include every required
    label; "–" positions are unconstrained and N/A lies outside the four-nucleotide fragment."""
    rows = [
        ([label if label != "." else NO_REQUIRED_INTERACTION for label in GNRA_SIGNATURE.split(",")],
         "GNRA hairpin\nHL_34789.6  (F)", (data["n_gnra_signature"], len(data["exact_hairpin"]))),
        ([label if label != "." else NO_REQUIRED_INTERACTION for label in TLOOP_SIGNATURE.split(",")],
         "T-loop fragment\nHL_33597.6  (G)", (data["n_tloop_signature"], len(data["tloop"]))),
    ]
    for y, (labels, name, (n_match, n)) in enumerate(rows):
        ax.text(-0.8, y, name, ha="right", va="center", fontsize=11)
        for x in range(6):
            label = labels[x] if x < len(labels) else OUTSIDE
            outside = label == OUTSIDE
            face = "white" if outside else LW_COLORS["."] if label == NO_REQUIRED_INTERACTION else LW_COLORS[label]
            ax.add_patch(Rectangle((x - 0.45, y - 0.24), 0.9, 0.48, facecolor=face,
                                   edgecolor="#BBBBBB" if outside else "white", lw=0.7))
            ax.text(x, y, label, ha="center", va="center", fontsize=9 if outside else 11,
                    color="#888888" if outside else "#222222")
        ax.text(6, y, f"{n_match}/{n}", ha="center", va="center", fontsize=11)
    ax.text(6, -0.55, "Matching\nfragments", ha="center", va="center", fontsize=10)
    ax.set_xlim(-2.5, 6.65)
    ax.set_ylim(1.65, -0.85)
    ax.set_xticks(range(6))
    ax.set_xticklabels(range(1, 7))
    ax.set_yticks([])
    style_axis(ax, xlabel="Token position")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.5, -0.35, f"{NO_REQUIRED_INTERACTION}  Unconstrained label    {OUTSIDE}  Outside the fragment",
            transform=ax.transAxes, ha="center", fontsize=9, color=S.GREY_TEXT)
    ax.text(0.5, -0.47, "Additional interactions are allowed", transform=ax.transAxes, ha="center",
            fontsize=9, color=S.GREY_TEXT)


def representative_index(fragments: list[np.ndarray], allowed: list[int] | None = None) -> int:
    """Instance with the lowest summed RMSD to all others, optionally among `allowed`."""
    n = len(fragments)
    rmsd = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            diff = kabsch_align(fragments[i], fragments[j]) - (fragments[j] - fragments[j].mean(axis=0))
            rmsd[i, j] = rmsd[j, i] = np.sqrt((diff ** 2).sum(axis=1).mean())
    candidates = allowed if allowed else list(range(n))
    return min(candidates, key=lambda i: rmsd[i].sum())


VIEW_HALF_WIDTH = 13.0   # Å; the same crop for F and G keeps one physical scale


def panel_structure(ax, letter: str, manifest: dict) -> None:
    """A ChimeraX render from render_fig6_structures.py, placed in Å so that F and G share a
    scale: the representative coloured, the other fragments thin grey sticks. G adds the
    representative's external partners (grey) with a dashed connector per DSSR pairing class."""
    panel = manifest["panels"][letter]
    half = manifest["camera_width_angstrom"] / 2
    ax.imshow(plt.imread(RENDER_DIR / f"molecule_{letter}.png"), extent=(-half, half, -half, half))
    h = min(half, VIEW_HALF_WIDTH)
    ax.set_xlim(-h, h)
    ax.set_ylim(-h, h)
    ax.axis("off")
    # label positions (Å) for the T-loop representative, chosen clear of the stacked bases
    tloop_label_xy = [(-10, -5), (-3, -11), (10, -3), (10, 1)]
    partner_label_xy = {"A58": (-9, 5), "G18": (-1, 11), "G19": (9, 7)}
    for j, centre in enumerate(panel["centres"]):
        xy = np.asarray(centre[:2])
        if letter == "G":
            pos, label = tloop_label_xy[j], f"{j + 1} {panel['labels'][j]}"
        else:
            pos, label = xy + xy / (np.linalg.norm(xy) + 1e-9) * 3, str(j + 1)
        ax.annotate(label, xy=xy, xytext=pos, fontsize=9, color=panel["colors"][j], ha="center",
                    va="center", fontweight="bold", arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.5},
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.9})
    for link in panel["links"]:
        start, end = np.asarray(link["start"][:2]), np.asarray(link["end"][:2])
        delta = end - start
        ax.plot(*np.transpose([start + delta * 0.25, end - delta * 0.25]), "--", color=LW_COLORS[link["lw"]], lw=1.2)
        ax.annotate(f"{link['label']}\n{link['lw']}", xy=end, xytext=partner_label_xy[link["label"]], fontsize=8,
                    ha="center", va="center", arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.5},
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.9})
    ax.plot([-h + 1, -h + 3], [-h + 1, -h + 1], color="#333333", lw=2)
    ax.text(-h + 2, -h + 0.25, "2 Å", ha="center", va="top", fontsize=9)


def write_numbers(data: dict) -> Path:
    rows = []
    def add(panel, item, value, source, column, statistic):
        rows.append({"panel": panel, "item": item, "value": value, "source_file": str(source.relative_to(REPO)),
                     "source_column": column, "statistic_or_rule": statistic})
    a_sources = {
        "five_candidate": (BASE, "five_anchor_candidate_cross_clan_median_rmsd"),
        "five_frequency": (BASE, "five_anchor_frequency_matched_cross_clan_median_rmsd"),
        "five_random": (BASE, "five_anchor_length_random_cross_clan_median_rmsd"),
        "five_sequence": (BASE, "five_anchor_sequence_matched_cross_clan_median_rmsd"),
        "base_candidate": (BASE, "base_inclusive_candidate_cross_clan_median_rmsd"),
        "base_frequency": (BASE, "base_inclusive_frequency_matched_cross_clan_median_rmsd"),
        "base_random": (BASE, "base_inclusive_length_random_cross_clan_median_rmsd"),
        "base_sequence": (BASE, "base_inclusive_sequence_matched_cross_clan_median_rmsd"),
    }
    for key, value in data["a_medians"].items():
        source, column = a_sources[key]
        add("B" if key.startswith("base") else "A", key, value, source, column, "median")
    add("A", "five_anchor_permutation_p", data["a_permutation_p"], BASE_SUMMARY,
        "permutation_five_anchor_candidate_vs_frequency_control_p", "two-sided unique-word stratified permutation P")
    add("B", "base_inclusive_permutation_p", data["a_base_permutation_p"], BASE_SUMMARY,
        "permutation_base_inclusive_candidate_vs_frequency_control_p", "two-sided unique-word stratified permutation P")
    add("C", "standard_helix_n", len(data["helix_values"]), BGSU_SENSITIVITY, "kmer,fraction_ge_1", "pairing category a with finite denominator")
    add("C", "other_words_n", len(data["other_values"]), BGSU_SENSITIVITY, "kmer,fraction_ge_1", "pairing categories b+c+d with finite denominator")
    add("C", "standard_helix_median_percent", np.median(data["helix_values"]) * 100, BGSU_SENSITIVITY, "fraction_ge_1", "median × 100")
    add("C", "other_words_median_percent", np.median(data["other_values"]) * 100, BGSU_SENSITIVITY, "fraction_ge_1", "median × 100")
    for group in ("a", "nonhelix"):
        low95, high95 = data["bgsu_null_ci95"][group]
        add("C", f"{group}_matched_null_ci95_low_percent", low95 * 100, BACKGROUND_REPLICATES,
            f"null_{group}_word_median_fraction (threshold 1)", "2.5th percentile × 100; error bar")
        add("C", f"{group}_matched_null_ci95_high_percent", high95 * 100, BACKGROUND_REPLICATES,
            f"null_{group}_word_median_fraction (threshold 1)", "97.5th percentile × 100; error bar")
        add("C", f"{group}_empirical_p", data["bgsu_p"][group], BACKGROUND_SUMMARY,
            f"threshold_1_group_{group}_empirical_p_ge", "one-sided Monte Carlo P")
        p10, median, p90 = data["bgsu_null"][group]
        add("C", f"{group}_matched_null_median_percent", median * 100, BACKGROUND_SUMMARY,
            f"threshold_1_group_{group}_null_word_median_fraction", "word-size-matched null median × 100")
        add("C", f"{group}_matched_null_p10_percent", p10 * 100, BACKGROUND_SUMMARY,
            f"threshold_1_group_{group}_null_word_median_p10_fraction", "null P10 × 100")
        add("C", f"{group}_matched_null_p90_percent", p90 * 100, BACKGROUND_SUMMARY,
            f"threshold_1_group_{group}_null_word_median_p90_fraction", "null P90 × 100")
    add("D", "six_token_string", data["hairpin"][0]["left_kmer"] + "-" + "-".join(data["hairpin"][0]["right_kmer"].split("-")[2:]), HAIRPIN, "left_kmer,right_kmer", "overlap positions 3–4")
    add("D", "cis_trans_topology_count", data["topology_count"], HAIRPIN, "left_pair_orient_pattern,right_pair_orient_pattern", "cis 1↔6 and trans 2↔5")
    add("D", "topology_denominator", len(data["hairpin"]), HAIRPIN, "sequence", "row count")
    add("E", "GNRA_loop_count", data["n_gnra"], HAIRPIN, "sequence", "positions 2–5 match GNRA")
    add("E", "GC_closing_pair_count", data["n_gc"], HAIRPIN, "sequence", "positions 1 and 6 are G–C/C–G")
    add("E", "sequence_count", len(data["sequences"]), HAIRPIN, "sequence", "row count")
    add("F", "GNRA_superposition_n", len(data["gnra_coords"]), HAIRPIN, "pdb_chain,author_residues,dssr_lw_signature", "successfully parsed base-inclusive atoms")
    add("F", "scale_bar_angstrom", 2, HAIRPIN, "pdb_chain,author_residues", "physical coordinate scale")
    add("G", "T_loop_superposition_n", len(data["tloop_coords"]), DSSR, "pdb_chain,author_residues,kmer", "successfully parsed base-inclusive atoms")
    add("G", "scale_bar_angstrom", 2, DSSR, "pdb_chain,author_residues", "physical coordinate scale")
    add("H", "GNRA_signature_count", data["n_gnra_signature"], HAIRPIN, "dssr_lw_signature", f"contains {GNRA_SIGNATURE} ('.' unconstrained); labels read from each token nucleotide")
    add("H", "GNRA_signature_denominator", len(data["exact_hairpin"]), HAIRPIN,
        "dssr_exact_bgsu_hl_34789_6", "six residues exactly match one six-nucleotide HL_34789.6 occurrence")
    add("H", "T_loop_signature_count", data["n_tloop_signature"], DSSR, "kmer,dssr_lw_pattern", f"kmer 48-24-39-39; contains {TLOOP_SIGNATURE} ('.' unconstrained)")
    add("H", "T_loop_signature_denominator", len(data["tloop"]), DSSR, "kmer", "kmer 48-24-39-39 row count")
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
    f_axes = [fig.add_subplot(right[0]), fig.add_subplot(right[1])]
    manifest = json.loads((RENDER_DIR / "aligned_models.json").read_text())
    for ax, letter in zip(f_axes, "FG"):
        panel_structure(ax, letter, manifest)
    panel_a(ax_a, data)
    panel_b(ax_b, data)
    panel_c(ax_c, data)
    panel_d(ax_d, data)
    panel_e(ax_e, data)

    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.06, top=0.97)
    shift = 0.055                            # right column moved in towards A and D
    for ax in f_axes:
        p = ax.get_position(original=True)
        ax.set_position([p.x0 - shift, p.y0, p.width, p.height])
        ax.set_aspect("equal")                # equal cells and crops: one Å scale for F and G
        ax.set_anchor("C")
        ax.apply_aspect()
    pb = ax_b.get_position()                 # a narrower B, left-aligned with E and F
    ax_b.set_position([f_axes[0].get_position().x0, pb.y0, pb.width * 0.8, pb.height])
    # letters in reading order, row by row
    panel_label(ax_a, "A")
    # B is the base-inclusive half of the shared axes, lettered just right of the divider.
    box = ax_a.get_position()
    divider = box.x0 + box.width * (5.0 - ax_a.get_xlim()[0]) / np.ptp(ax_a.get_xlim())
    fig.text(divider + 0.12 / fig.get_size_inches()[0], box.y1 + 0.14 / fig.get_size_inches()[1], "B",
             fontsize=15, fontweight="bold", va="bottom", ha="left")
    panel_label(ax_b, "C")
    panel_label(ax_c, "D", top=ax_d, dy=-0.12)
    panel_label(ax_d, "E", dy=-0.12)
    panel_label(f_axes[0], "F", top=ax_d, left=ax_b, dy=-0.12)   # GNRA structure
    panel_label(f_axes[1], "G", left=ax_b)                        # T-loop structure
    panel_label(ax_e, "H")                                       # interaction signatures

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
