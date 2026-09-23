"""Plot the Fig. 5 motif analysis for the production token database."""
from __future__ import annotations

import csv
import itertools
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Patch, Rectangle
from mpl_toolkits.mplot3d import proj3d
import math
from scipy.stats import beta


REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("RFOLDSEEK_DATA", "/Users/cassie/Desktop/RFoldSeek"))
MOTIF_DIR = REPO / "files" / "motifs"
FIG_DIR = REPO / "figures" / "motifs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = REPO / "files" / "models" / "db.pt"
KMER_CANDIDATES = MOTIF_DIR / "kmer_candidates_production_new_k4_aa.tsv"
RANDOM_CONTROL = MOTIF_DIR / "random_control_rmsd_production_new_k4_aa.tsv"
GLOBAL_SEQ_CONTROL = MOTIF_DIR / "global_sequence_matched_control_production_new_k4_aa.tsv"
DSSR_SUMMARY = MOTIF_DIR / "known_motif_core_dssr_pair_conservation_summary_production_new.tsv"
STRICT_CLEAN_MANIFEST = MOTIF_DIR / "known_motif_core_matches_production_new_relabel_manifest.tsv"
SEQ_CONTROL = MOTIF_DIR / "sequence_matched_control_44-48-62-71_exact_strict.tsv"
SEQ_CONTROL_SUMMARY = MOTIF_DIR / "sequence_matched_control_44-48-62-71_exact_strict_summary.tsv"

PRIMARY_KMER = "44-48-62-71"
TARGETS = [
    ("44-48-62-71", "HL_34789.6"),
    ("62-71-66-12", "HL_34789.6"),
    ("48-24-39-39", "HL_33597.6"),
]
C4_INDEX = 1

KMER_COLORS = {
    "44-48-62-71": "#5B7C99",
    "62-71-66-12": "#7DA087",
    "48-24-39-39": "#9B6B8A",
}
GREY_FILL, GREY_LINE = "#D9D9D9", "#7F7F7F"
INK = "#333333"
SOFT_BG = "#F5F5F5"
GRID = "#E6E6E6"
POS_COLORS = ["#5B7C99", "#B76E5A", "#C4A35A", "#7DA087"]
LW_COLORS = {
    "cWW": "#5B7C99",
    "tSH": "#B76E5A",
    "tWH": "#9B6B8A",
    "tWS": "#C4A35A",
    ".": "#E6E6E6",
}


def read_column(path: Path, column: str) -> np.ndarray:
    return np.asarray([float(row[column]) for row in csv.DictReader(path.open(), delimiter="\t")])


def read_key_value(path: Path) -> dict[str, str]:
    return {row["metric"]: row["value"] for row in csv.DictReader(path.open(), delimiter="\t")}


def read_tsv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(), delimiter="\t"))


def load_db():
    db = torch.load(DB_PATH, map_location="cpu", weights_only=False)
    sources = {}
    # Recent compact production databases omit the original JSONL source list;
    # all panels can still be redrawn from entries plus the cached controls.
    for data_path in db.get("data", []):
        path = REPO / data_path if not Path(data_path).is_absolute() else Path(data_path)
        if not path.exists() and not Path(data_path).is_absolute():
            path = DATA_ROOT / data_path
        with path.open() as handle:
            for line in handle:
                source = json.loads(line)
                sources[str(source["id"])] = source
    entries = db["entries"]
    return entries, {Path(entry["id"]).stem: entry for entry in entries}, sources


def compact_coords(entry: dict) -> np.ndarray:
    coords = np.asarray(entry["coords"], dtype=np.float32)
    mask = np.asarray(entry["mask"], dtype=bool)
    coords = coords[mask]
    if len(coords) != len(entry["tokens"]):
        raise ValueError(f"coordinate/token length mismatch for {entry['id']}")
    return coords


def kabsch_transform(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    pc = points - points.mean(0, keepdims=True)
    qc = reference - reference.mean(0, keepdims=True)
    u, _, vt = np.linalg.svd(pc.T @ qc)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return pc @ rotation.T


def kabsch_rmsd(points: np.ndarray, reference: np.ndarray) -> float:
    ref = reference - reference.mean(0, keepdims=True)
    aligned = kabsch_transform(points, reference)
    return float(np.sqrt(np.mean(np.sum((aligned - ref) ** 2, axis=1))))


def all_pair_rmsd(fragments: list[np.ndarray]) -> np.ndarray:
    arr = np.asarray(fragments, dtype=np.float64)
    n = len(arr)
    if n < 2:
        return np.asarray([], dtype=float)
    left, right = np.triu_indices(n, k=1)
    p = arr[left]
    q = arr[right]
    pc = p - p.mean(axis=1, keepdims=True)
    qc = q - q.mean(axis=1, keepdims=True)
    h = np.einsum("bni,bnj->bij", pc, qc)
    u, _, vt = np.linalg.svd(h)
    det = np.linalg.det(np.einsum("bij,bjk->bik", np.swapaxes(vt, 1, 2), np.swapaxes(u, 1, 2)))
    sign = np.where(det < 0, -1.0, 1.0)
    d = np.zeros((len(h), 3, 3), dtype=np.float64)
    d[:, 0, 0] = 1.0
    d[:, 1, 1] = 1.0
    d[:, 2, 2] = sign
    rotation = np.einsum(
        "bij,bjk,bkl->bil",
        np.swapaxes(vt, 1, 2),
        d,
        np.swapaxes(u, 1, 2),
    )
    aligned = np.einsum("bni,bji->bnj", pc, rotation)
    return np.sqrt(np.mean(np.sum((aligned - qc) ** 2, axis=2), axis=1))


def scan_occurrences(kmer: str, entries: list[dict], sources: dict[str, dict]) -> list[dict[str, object]]:
    target = tuple(int(value) for value in kmer.split("-"))
    rows = []
    for entry in entries:
        tokens = np.asarray(entry["tokens"], dtype=np.int32)
        source = sources.get(str(entry["id"]))
        if source is None:
            continue
        seq = str(source.get("seq", ""))
        mask = np.asarray(source.get("mask", entry["mask"]), dtype=bool)
        raw_indices = np.flatnonzero(mask)
        coords = compact_coords(entry)
        if len(raw_indices) != len(tokens):
            continue
        stem = Path(entry["id"]).stem
        for start in range(max(0, len(tokens) - len(target) + 1)):
            if tuple(int(value) for value in tokens[start:start + len(target)]) != target:
                continue
            raw_window = raw_indices[start:start + len(target)]
            rows.append({
                "pdb_chain": stem,
                "start": start,
                "sequence": "".join(seq[index] for index in raw_window if index < len(seq)),
                "fragment": coords[start:start + len(target)].reshape(-1, 3),
            })
    return rows


def strict_clean_rows(kmer: str) -> list[dict[str, str]]:
    return [
        row for row in csv.DictReader(STRICT_CLEAN_MANIFEST.open(), delimiter="\t")
        if row["kmer"] == kmer and row["strict_clean"] == "1"
    ]


def motif_fragments(rows: list[dict[str, str]], entries_by_stem: dict[str, dict]) -> list[np.ndarray]:
    fragments = []
    for row in rows:
        entry = entries_by_stem[row["pdb_chain"]]
        start = int(row["token_start_1based"]) - 1
        fragments.append(compact_coords(entry)[start:start + 4])
    return fragments


def median_all_pair_rmsd(fragments: list[np.ndarray]) -> float:
    return float(np.median(all_pair_rmsd([fragment.reshape(-1, 3) for fragment in fragments])))


def ensure_sequence_control(entries: list[dict], sources: dict[str, dict]) -> None:
    if SEQ_CONTROL.exists() and SEQ_CONTROL_SUMMARY.exists():
        return
    occurrences = scan_occurrences(PRIMARY_KMER, entries, sources)
    observed_fragments = [row["fragment"] for row in occurrences]
    observed_values = all_pair_rmsd(observed_fragments)
    variant_counts = Counter(str(row["sequence"]) for row in occurrences)
    target = tuple(int(value) for value in PRIMARY_KMER.split("-"))
    pool = defaultdict(list)
    for entry in entries:
        source = sources.get(str(entry["id"]))
        if source is None:
            continue
        seq = str(source.get("seq", ""))
        mask = np.asarray(source.get("mask", entry["mask"]), dtype=bool)
        raw_indices = np.flatnonzero(mask)
        tokens = np.asarray(entry["tokens"], dtype=np.int32)
        coords = compact_coords(entry)
        if len(raw_indices) != len(tokens):
            continue
        stem = Path(entry["id"]).stem
        for start in range(max(0, len(tokens) - 4 + 1)):
            token_window = tuple(int(value) for value in tokens[start:start + 4])
            if token_window == target:
                continue
            raw_window = raw_indices[start:start + 4]
            if len(raw_window) != 4 or raw_window[-1] >= len(seq):
                continue
            window_seq = "".join(seq[index] for index in raw_window)
            if window_seq in variant_counts:
                label = f"{window_seq}:{stem}:{start + 1}"
                pool[window_seq].append((label, coords[start:start + 4].reshape(-1, 3)))
    missing = [seq for seq in variant_counts if not pool.get(seq)]
    if missing:
        raise RuntimeError(f"missing sequence-matched pools for {missing}")
    draw_order = []
    for seq, count in variant_counts.items():
        draw_order.extend([seq] * count)
    rng = random.Random(260807)
    medians = []
    SEQ_CONTROL.parent.mkdir(parents=True, exist_ok=True)
    with SEQ_CONTROL.open("w", newline="") as handle:
        fieldnames = ["replicate", "median_rmsd", "p90_rmsd", "n_fragments", "n_pairs", "sampled_windows"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for rep in range(1, 2001):
            sampled = [rng.choice(pool[seq]) for seq in draw_order]
            values = all_pair_rmsd([item[1] for item in sampled])
            medians.append(float(np.median(values)))
            writer.writerow({
                "replicate": rep,
                "median_rmsd": medians[-1],
                "p90_rmsd": float(np.percentile(values, 90)),
                "n_fragments": len(sampled),
                "n_pairs": len(values),
                "sampled_windows": ";".join(item[0] for item in sampled),
            })
    n_le = int(np.sum(np.asarray(medians) <= np.median(observed_values)))
    rows = [
        ("kmer", PRIMARY_KMER),
        ("n_observed_occurrences", len(occurrences)),
        ("observed_sequence_counts", ";".join(f"{seq}:{count}" for seq, count in variant_counts.most_common())),
        ("observed_all_pair_median_rmsd", float(np.median(observed_values))),
        ("observed_all_pair_p90_rmsd", float(np.percentile(observed_values, 90))),
        ("observed_n_pairs", len(observed_values)),
        ("n_replicates", 2000),
        ("control_median_of_medians", float(np.median(medians))),
        ("control_p10_median_rmsd", float(np.percentile(medians, 10))),
        ("control_p90_median_rmsd", float(np.percentile(medians, 90))),
        ("empirical_one_sided_p_le_observed", n_le / 2000),
        ("n_replicates_le_observed", n_le),
        ("random_seed", 260807),
    ]
    with SEQ_CONTROL_SUMMARY.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def dssr_rows() -> list[dict[str, object]]:
    rows = read_tsv(DSSR_SUMMARY)
    out = []
    for kmer, bgsu in TARGETS:
        rec = next(row for row in rows if row["kmer"] == kmer and row["metric"] == "dssr_lw_pattern")
        count = int(float(rec["dominant_count"]))
        n = int(rec["n_instances"])
        lo = 0.0 if count == 0 else beta.ppf(0.025, count, n - count + 1)
        hi = 1.0 if count == n else beta.ppf(0.975, count + 1, n - count)
        out.append({
            "kmer": kmer,
            "bgsu": bgsu,
            "signature": rec["dominant_state"],
            "count": count,
            "n": n,
            "frac": count / n,
            "lo": lo,
            "hi": hi,
        })
    return out


def set_panel_label(ax, label: str) -> None:
    if hasattr(ax, "text2D"):
        ax.text2D(-0.08, 1.06, label, transform=ax.transAxes, fontsize=14, fontweight="bold")
    else:
        ax.text(-0.08, 1.06, label, transform=ax.transAxes, fontsize=14, fontweight="bold")


def set_panel_title(ax, title: str) -> None:
    """Consistent, compact within-panel heading."""
    if hasattr(ax, "text2D"):
        ax.text2D(0.0, 1.035, title, transform=ax.transAxes, fontsize=10.5,
                  fontweight="bold", color=INK, ha="left", va="bottom")
    else:
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold",
                     color=INK, pad=9)


def panel_a(ax) -> None:
    token_rmsd = read_column(KMER_CANDIDATES, "cross_med_rmsd")
    random_rmsd = read_column(RANDOM_CONTROL, "rmsd")
    seq_rmsd = read_column(GLOBAL_SEQ_CONTROL, "control_median_of_medians")
    seq_rmsd = seq_rmsd[np.isfinite(seq_rmsd)]
    rng = np.random.default_rng(7)
    random_sample = rng.choice(random_rmsd, size=min(30000, len(random_rmsd)), replace=False)
    bins = np.linspace(0, 4.5, 70)
    blue = KMER_COLORS[PRIMARY_KMER]
    seq_color = "#B76E5A"
    ax.hist(random_sample, bins=bins, density=True, color=GREY_FILL, alpha=0.72,
            edgecolor="none", label="length-matched random")
    ax.hist(seq_rmsd, bins=bins, density=True, histtype="step", linewidth=1.8,
            color=seq_color, label="sequence-matched non-target")
    ax.hist(token_rmsd, bins=bins, density=True, histtype="step", linewidth=2.0,
            color=blue, label="token words")
    med_token, med_random, med_seq = (np.median(token_rmsd), np.median(random_rmsd),
                                      np.median(seq_rmsd))
    ax.axvline(med_token, color=blue, linewidth=1.15)
    ax.axvline(med_random, color=GREY_LINE, linewidth=1.15, linestyle=(0, (4, 3)))
    ax.axvline(med_seq, color=seq_color, linewidth=1.15, linestyle=(0, (2, 2)))
    ymax = ax.get_ylim()[1]
    ax.text(med_token + 0.05, ymax * 0.83, rf"{med_token:.2f} $\AA$", color=blue,
            fontsize=8.6, fontweight="bold")
    ax.text(med_random + 0.06, ymax * 0.74, rf"{med_random:.2f} $\AA$", color=GREY_LINE,
            fontsize=8.6, fontweight="bold")
    ax.text(med_seq + 0.06, ymax * 0.64, rf"{med_seq:.2f} $\AA$", color=seq_color,
            fontsize=8.6, fontweight="bold")
    ax.set_xlim(0, 4.5)
    ax.set_xlabel(r"Median RMSD ($\AA$)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=7.8, loc="upper right", handlelength=2.3,
              labelspacing=0.35)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    set_panel_label(ax, "A")


def draw_superposition(ax, ax_txt, frags: list[np.ndarray], tag: str, note: str) -> None:
    ref = frags[0].reshape(-1, 3)
    for frag in frags:
        aligned = kabsch_transform(frag.reshape(-1, 3), ref).reshape(4, 5, 3)
        c4 = aligned[:, C4_INDEX, :]
        ax.plot(c4[:, 0], c4[:, 1], c4[:, 2], color="#A6ADB6", alpha=0.4, linewidth=1.0)
        ax.scatter(c4[:, 0], c4[:, 1], c4[:, 2], c=POS_COLORS, s=13, depthshade=False, edgecolors="white", linewidths=0.3)
    ax.view_init(elev=21, azim=-58)
    ax.set_axis_off()
    ax_txt.axis("off")
    ax_txt.text(0.0, 0.62, tag, transform=ax_txt.transAxes, fontsize=9.2, color=KMER_COLORS[tag], fontweight="bold", va="center", ha="left")
    ax_txt.text(0.0, 0.34, note, transform=ax_txt.transAxes, fontsize=8.0, va="center", ha="left", linespacing=1.35)


def add_horizontal_scale_bar(ax, length_ang: float = 2.0) -> None:
    """Draw a horizontal 2D scale bar at the bottom-left of a 3D axes.

    Length is chosen so the bar spans `length_ang` in world units along the
    world axis that projects most strongly to the screen for the current
    view — this keeps the bar horizontal on screen while still faithfully
    representing length_ang angstroms in the plot's visible plane.
    """
    xlim = ax.get_xlim3d(); ylim = ax.get_ylim3d(); zlim = ax.get_zlim3d()
    cx = 0.5 * (xlim[0] + xlim[1])
    cy = 0.5 * (ylim[0] + ylim[1])
    cz = 0.5 * (zlim[0] + zlim[1])
    M = ax.get_proj()
    p0 = proj3d.proj_transform(cx, cy, cz, M)
    d_center = ax.transData.transform((p0[0], p0[1]))
    pixel_lengths = []
    for vx, vy, vz in [(length_ang, 0, 0), (0, length_ang, 0), (0, 0, length_ang)]:
        p1 = proj3d.proj_transform(cx + vx, cy + vy, cz + vz, M)
        d_end = ax.transData.transform((p1[0], p1[1]))
        pixel_lengths.append(math.hypot(d_end[0] - d_center[0], d_end[1] - d_center[1]))
    pixel_length = max(pixel_lengths)
    bbox = ax.get_window_extent()
    frac_length = pixel_length / bbox.width if bbox.width > 0 else 0.15
    x0, y0 = 0.06, 0.10
    ax.add_line(Line2D(
        [x0, x0 + frac_length], [y0, y0],
        transform=ax.transAxes, color=INK, linewidth=2.2,
        solid_capstyle="butt", clip_on=False, zorder=10,
    ))
    text_fn = ax.text2D if hasattr(ax, "text2D") else ax.text
    text_fn(
        x0 + frac_length / 2, y0 - 0.045,
        rf"{length_ang:g} $\AA$",
        transform=ax.transAxes, color=INK, fontsize=7.5,
        ha="center", va="top", clip_on=False, zorder=10,
    )


def add_lw_color_legend(ax_e) -> None:
    """Base-pair geometry color legend, placed below panel E."""
    handles = [
        Patch(facecolor=LW_COLORS[k], edgecolor="white", label=k)
        for k in ("cWW", "tSH", "tWH", "tWS")
    ]
    ax_e.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        frameon=False,
        fontsize=8.0,
        handlelength=1.2,
        handleheight=0.9,
        columnspacing=1.8,
        borderpad=0.0,
    )


def panel_b(rows, entries, entries_by_stem, sources) -> None:
    frags_18 = motif_fragments(strict_clean_rows(PRIMARY_KMER), entries_by_stem)
    frags_62 = motif_fragments(strict_clean_rows("62-71-66-12"), entries_by_stem)
    frags_48 = motif_fragments(strict_clean_rows("48-24-39-39"), entries_by_stem)
    specs = [
        (frags_18, PRIMARY_KMER, f"{len(frags_18)} clean matches\n" + rf"median = {median_all_pair_rmsd(frags_18):.2f} $\AA$"),
        (frags_62, "62-71-66-12", f"{len(frags_62)} clean matches\n" + rf"median = {median_all_pair_rmsd(frags_62):.2f} $\AA$"),
        (frags_48, "48-24-39-39", f"{len(frags_48)} clean matches\n" + rf"median = {median_all_pair_rmsd(frags_48):.2f} $\AA$"),
    ]
    for (ax3d, ax_txt), (frags, tag, note) in zip(rows, specs):
        draw_superposition(ax3d, ax_txt, frags, tag, note)
    rows[0][0].text2D(-0.10, 1.06, "B", transform=rows[0][0].transAxes,
                      fontsize=14, fontweight="bold", ha="left", va="bottom")


def panel_c(ax) -> None:
    vals = read_column(SEQ_CONTROL, "median_rmsd")
    summary = read_key_value(SEQ_CONTROL_SUMMARY)
    observed = float(summary["observed_all_pair_median_rmsd"])
    parts = ax.violinplot([vals], positions=[1], widths=0.7, showmeans=False, showmedians=False, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(GREY_FILL)
        body.set_edgecolor(GREY_LINE)
        body.set_alpha(0.95)
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    lo, hi = np.percentile(vals, [5, 95])
    ax.plot([0.82, 1.18], [med, med], color=INK, linewidth=2)
    ax.add_patch(Rectangle((0.88, q1), 0.24, q3 - q1, facecolor="none", edgecolor=INK, linewidth=1))
    ax.plot([1, 1], [lo, hi], color=INK, linewidth=1)
    color = KMER_COLORS[PRIMARY_KMER]
    ax.scatter([1], [observed], s=72, color=color, zorder=5, edgecolors="white", linewidths=0.8)
    ax.plot([1, 1], [observed + 0.08, med - 0.08], color=color, linewidth=1.0, linestyle=(0, (3, 3)), alpha=0.8)
    ax.text(1.055, observed, rf"Observed  {observed:.2f} $\AA$", color=color,
            fontsize=8.5, fontweight="bold", va="center", ha="left")
    ax.text(1.47, med + 0.18, rf"Control median  {med:.2f} $\AA$", color=INK,
            fontsize=8.0, va="bottom", ha="right")
    ax.set_xlim(0.45, 1.55)
    ax.set_xticks([1])
    ax.set_xticklabels(["Sequence-matched\nnon-target windows"])
    ax.set_ylabel(r"Median all-pair RMSD ($\AA$)")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    set_panel_label(ax, "C")


def draw_motif_core(ax, y: float, label: str, core_start: int, motif_len: int, signature: str, color: str) -> None:
    x0 = 2.18
    spacing = 0.40 if motif_len > 6 else 0.52
    radius = 0.12 if motif_len > 6 else 0.14
    for i in range(motif_len):
        x = x0 + i * spacing
        face = color if core_start <= i < core_start + 4 else "white"
        edge = color if core_start <= i < core_start + 4 else "#5b6470"
        circ = Circle((x, y), radius, facecolor=face, edgecolor=edge, linewidth=1.3)
        ax.add_patch(circ)
        ax.text(x, y, str(i + 1), ha="center", va="center", fontsize=8, color="white" if face != "white" else "#4b5563")
    ax.add_patch(FancyBboxPatch((x0 + core_start * spacing - radius - 0.09, y - 0.31), spacing * 3 + 2 * radius + 0.18, 0.62, boxstyle="round,pad=0.02,rounding_size=0.04", facecolor="none", edgecolor=color, linewidth=1.2))
    ax.text(0.05, y, label, ha="left", va="center", fontsize=8.5, color=color, fontweight="bold")
    signature_x = x0 + (motif_len - 1) * spacing + 0.34
    ax.text(signature_x, y, signature, ha="left", va="center", fontsize=8.2,
            family="monospace")


def panel_d(ax) -> None:
    ax.set_xlim(0, 9.6)
    ax.set_ylim(0, 4.25)
    ax.axis("off")
    # symbol legend: filled circle / open circle / rounded box, spacing handled by legend layout
    open_edge = "#5b6470"
    legend_handles = [
        Line2D([], [], marker="o", markerfacecolor=INK, markeredgecolor=INK,
               markersize=7, linestyle="None", label="token-covered residue"),
        Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=open_edge,
               markeredgewidth=1.4, markersize=7, linestyle="None", label="other motif residue"),
        Patch(facecolor="none", edgecolor=INK, linewidth=1.2, label="recovered core"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
        fontsize=7.8,
        handlelength=1.1,
        handletextpad=0.5,
        columnspacing=2.6,
        borderpad=0.0,
    )
    ax.text(0.02, 3.67, "HL_34789.6 annotated motif  ·  6 nt", fontsize=8.8,
            color=INK, fontweight="bold")
    draw_motif_core(ax, 2.95, PRIMARY_KMER, 0, 6, "PPUU | cWW,tSH,.,.", KMER_COLORS[PRIMARY_KMER])
    draw_motif_core(ax, 2.15, "62-71-66-12", 2, 6, "UUPP | .,.,tSH,cWW", KMER_COLORS["62-71-66-12"])
    ax.plot([0.0, 9.35], [1.69, 1.69], color=GRID, lw=0.8, zorder=0)
    ax.text(0.02, 1.38, "HL_33597.6 annotated motif  ·  9 nt", fontsize=8.8,
            color=INK, fontweight="bold")
    draw_motif_core(ax, 0.75, "48-24-39-39", 0, 9, "PPPU | tWH,tWS,cWW,.", KMER_COLORS["48-24-39-39"])
    set_panel_label(ax, "D")


def panel_e(ax) -> None:
    rows = dssr_rows()
    ax.set_xlim(0, 5.2)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.set_xticks(np.arange(4) + 0.5)
    ax.set_xticklabels(["1", "2", "3", "4"])
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([row["kmer"] for row in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("core position")
    for y, row in enumerate(rows):
        sig = str(row["signature"]).split(",")
        for x, value in enumerate(sig):
            ax.add_patch(Rectangle((x, y - 0.36), 0.88, 0.72, facecolor=LW_COLORS.get(value, "#bfc5cc"), edgecolor="white", linewidth=1.2))
            ax.text(x + 0.44, y, value, ha="center", va="center", fontsize=8, color="white" if value != "." else "#5b6470")
        ax.text(4.2, y, f"{row['count']}/{row['n']}", va="center", fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    set_panel_label(ax, "E")


def panel_f(ax) -> None:
    rows = dssr_rows()
    y = np.arange(len(rows))
    frac = np.asarray([float(row["frac"]) for row in rows])
    lo = np.asarray([float(row["lo"]) for row in rows])
    hi = np.asarray([float(row["hi"]) for row in rows])
    ax.barh(y, frac, color=[KMER_COLORS[str(row["kmer"])] for row in rows], height=0.54, zorder=2)
    ax.errorbar(frac, y, xerr=np.vstack([frac - lo, hi - frac]), fmt="none", ecolor=INK, capsize=3, linewidth=1.2, zorder=4)
    for yi, row in enumerate(rows):
        ax.text(max(float(row["frac"]), float(row["hi"])) + 0.035, yi, f"{float(row['frac']) * 100:.1f}%", va="center", fontsize=9, ha="left", clip_on=False)
    ax.set_xlim(0, 1.16)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{row['kmer']}\n{row['bgsu']}" for row in rows], fontsize=7.7)
    ax.invert_yaxis()
    ax.set_xlabel("dominant signature frequency")
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    set_panel_label(ax, "F")


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.edgecolor": "#4b5563",
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    entries, entries_by_stem, sources = load_db()
    ensure_sequence_control(entries, sources)
    fig = plt.figure(figsize=(12.0, 8.25), constrained_layout=False)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.18, 1.0, 1.08],
                          height_ratios=[1.17, 1.0], hspace=0.30, wspace=0.25)
    ax_a = fig.add_subplot(gs[0, 0])
    gs_b = gs[0, 1].subgridspec(3, 2, width_ratios=[1.0, 0.8], hspace=0.12, wspace=0.0)
    b_rows = [(fig.add_subplot(gs_b[r, 0], projection="3d"), fig.add_subplot(gs_b[r, 1])) for r in range(3)]
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])
    panel_a(ax_a)
    panel_b(b_rows, entries, entries_by_stem, sources)
    panel_c(ax_c)
    panel_d(ax_d)
    panel_e(ax_e)
    panel_f(ax_f)
    fig.subplots_adjust(left=0.065, right=0.975, bottom=0.095, top=0.93,
                        wspace=0.25, hspace=0.30)
    # LW color legend below panel E
    add_lw_color_legend(ax_e)
    # horizontal 2 Å scale bars for the three Kabsch superpositions in panel B
    # (needs a first draw so axes bboxes are final)
    fig.canvas.draw()
    for ax3d, _ in b_rows:
        add_horizontal_scale_bar(ax3d, 2.0)
    out_png = FIG_DIR / "known_motif_interaction_cores.png"
    root_png = REPO / "figures" / "fig5.png"
    for path, kwargs in ((out_png, {"dpi": 350}), (root_png, {"dpi": 350})):
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
    print(f"Wrote {root_png}")


if __name__ == "__main__":
    main()
