"""Supplementary alphabet-states figure: the 80-state RFoldSearch structural alphabet, drawn from real representatives.

For each token, the script selects representative occurrences closest to the
token centroid in the normalised 17-D production descriptor space. Their
central nucleotides and nearest non-sequential spatial neighbours are aligned
in the central nucleotide's local C4' frame and overlaid.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import DATA_ROOT, FIG_DIR, SRC_MODELS  # noqa: E402

REPO = Path(DATA_ROOT)
sys.path.insert(0, SRC_MODELS)

from descriptors import compute_descriptor  # noqa: E402
from descriptors._utils import build_local_frames  # noqa: E402

ACTIVE_DIMS = np.r_[0:10, 42:49]
STATE_COLORS = ["#2A9D8F", "#E76F51", "#457B9D", "#E9C46A", "#B565A7"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(REPO / "files/models/db.pt"))
    p.add_argument("--checkpoint", default=str(REPO / "files/models/desc_vqvae_best.pt"))
    p.add_argument(
        "--data",
        nargs="+",
        default=[str(REPO / f"Splits/pdb_out/{s}.json") for s in ("train", "val", "test")],
    )
    p.add_argument(
        # NOTE: only annotates the side TSV (frequency/role columns); the
        # figure itself does not use it. The bundled table predates the
        # final retrain, so its token indices refer to the earlier codebook.
        "--profile",
        default=str(REPO / "files/motifs/token_profile_summary.tsv"),
    )
    p.add_argument("--output", default=str(Path(FIG_DIR) / "supp_alphabet_states.png"))
    p.add_argument(
        # Companion table listing the chosen representatives. Written next to
        # the figure; note that figures/ is gitignored, so this is a local
        # artifact rather than a bundled result.
        "--table",
        default=str(Path(FIG_DIR) / "token_state_representatives.tsv"),
    )
    p.add_argument("--n_examples", type=int, default=5)
    return p.parse_args()


def load_profiles(path: str):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return {int(r["token"]): r for r in rows}


def load_sources(paths, wanted):
    entries = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                e = json.loads(line)
                if e.get("id") in wanted:
                    entries[e["id"]] = e
    missing = wanted.difference(entries)
    if missing:
        raise RuntimeError(f"Could not find {len(missing)} database entries in source JSON files")
    return entries


def tensors(entry):
    coords = torch.as_tensor(entry["coords"], dtype=torch.float32).unsqueeze(0)
    mask = torch.as_tensor(entry["mask"], dtype=torch.bool).unsqueeze(0)
    pair = torch.as_tensor(
        entry.get("pair_idx", entry.get("pair_edges")), dtype=torch.long
    ).unsqueeze(0)
    stack = torch.as_tensor(
        entry.get("stack_idx", entry.get("stack_edges")), dtype=torch.long
    ).unsqueeze(0)
    orient = entry.get("pair_orient")
    if orient is not None:
        orient = torch.as_tensor(orient, dtype=torch.long).unsqueeze(0)
    if pair.ndim == 3:
        pair = pair[..., 0]
    if stack.ndim == 3:
        stack = stack[..., 0]
    if orient is not None and orient.ndim == 3:
        orient = orient[..., 0]
    return coords, mask, pair, stack, orient


@torch.no_grad()
def descriptor_for_entry(entry, norm_mean, norm_std):
    coords, mask, pair, stack, orient = tensors(entry)
    desc, info = compute_descriptor(coords, mask, pair, stack, pair_orient=orient)
    norm = (desc[0] - norm_mean) / norm_std
    valid = np.flatnonzero(mask[0].numpy())
    active = norm[valid][:, ACTIVE_DIMS].numpy()
    nn_idx = info["nn1_idx"][0].numpy()
    return active, valid, nn_idx


def select_representatives(db_entries, source_entries, norm_mean, norm_std, n_codes, n_examples=5):
    sums = np.zeros((n_codes, len(ACTIVE_DIMS)), dtype=np.float64)
    counts = np.zeros(n_codes, dtype=np.int64)

    cached = {}
    for db_e in db_entries:
        active, valid, nn_idx = descriptor_for_entry(
            source_entries[db_e["id"]], norm_mean, norm_std
        )
        tokens = np.asarray(db_e["tokens"], dtype=np.int64)
        if len(tokens) != len(valid):
            raise RuntimeError(f"Token/mask mismatch for {db_e['id']}")
        np.add.at(sums, tokens, active)
        np.add.at(counts, tokens, 1)
        cached[db_e["id"]] = (active, valid, nn_idx)

    if np.any(counts == 0):
        raise RuntimeError(f"Unused production tokens: {np.flatnonzero(counts == 0).tolist()}")
    centroids = sums / counts[:, None]
    # Keep the closest occurrence from each RNA chain for every token. This
    # prevents one highly repetitive structure from dominating an overlay.
    candidates = [dict() for _ in range(n_codes)]

    for db_e in db_entries:
        active, valid, nn_idx = cached[db_e["id"]]
        tokens = np.asarray(db_e["tokens"], dtype=np.int64)
        d2 = ((active - centroids[tokens]) ** 2).mean(axis=1)
        source = source_entries[db_e["id"]]
        seq = source.get("seq", "")
        for token, raw_i, dist in zip(tokens, valid, d2):
            token = int(token)
            raw_i = int(raw_i)
            raw_j = int(nn_idx[raw_i])
            old = candidates[token].get(db_e["id"])
            if raw_j < 0 or (old is not None and dist >= old["centroid_mse"]):
                continue
            candidates[token][db_e["id"]] = {
                "token": token,
                "id": db_e["id"],
                "raw_i": raw_i,
                "raw_j": raw_j,
                "seq_i": seq[raw_i] if raw_i < len(seq) else "?",
                "seq_j": seq[raw_j] if raw_j < len(seq) else "?",
                "seq_sep": abs(raw_i - raw_j),
                "centroid_mse": float(dist),
            }

    representatives = []
    for token in range(n_codes):
        selected = sorted(candidates[token].values(), key=lambda r: r["centroid_mse"])[:n_examples]
        if not selected:
            representatives.append([])
            continue
        for rep in selected:
            coords = np.asarray(source_entries[rep["id"]]["coords"], dtype=np.float32)
            rot, origin = build_local_frames(
                torch.from_numpy(coords[rep["raw_i"], :3]).unsqueeze(0)
            )
            rot = rot[0].numpy()
            origin = origin[0].numpy()
            rep["local_i"] = (coords[rep["raw_i"]] - origin) @ rot
            rep["local_j"] = (coords[rep["raw_j"]] - origin) @ rot
        representatives.append(selected)
    return representatives, counts


def project(points):
    # Fixed axonometric view after alignment to the central nucleotide frame.
    matrix = np.array([[0.866, -0.500], [0.250, 0.433], [0.866, 0.866]])
    return np.asarray(points) @ matrix


def draw_nucleotide(ax, points, color, alpha=1.0, zorder=3):
    p = project(points)
    # Some modified residues have no extracted base anchor (stored as zero).
    # Keep the resolved sugar atoms and omit only the unavailable anchor.
    base_ok = np.linalg.norm(points[4] - points[1]) < 8.0
    bonds = [(0, 1), (1, 2), (1, 3)]
    if base_ok:
        bonds.append((3, 4))
    for a, b in bonds:
        ax.plot(
            p[[a, b], 0],
            p[[a, b], 1],
            color=color,
            lw=1.45,
            alpha=alpha,
            solid_capstyle="round",
            zorder=zorder,
        )


def draw_state(ax, reps, token):
    for rank, rep in enumerate(reps):
        color = STATE_COLORS[rank % len(STATE_COLORS)]
        oi, oj = project(np.stack([rep["local_i"][1], rep["local_j"][1]]))
        ax.plot([oi[0], oj[0]], [oi[1], oj[1]], color=color, lw=0.8, alpha=0.64, zorder=1)
        draw_nucleotide(ax, rep["local_j"], color, alpha=0.84, zorder=2)
        draw_nucleotide(ax, rep["local_i"], color, alpha=0.94, zorder=3)

    ax.text(
        0.5,
        -0.02,
        str(token),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.5,
        color="#111111",
        fontfamily="Arial",
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(-13.5, 13.5)
    ax.set_ylim(-12.5, 15.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")


def write_table(path, reps, profiles, counts):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "token",
        "representative_rank",
        "n",
        "frequency",
        "role",
        "structure_id",
        "residue_index_1based",
        "nucleotide",
        "neighbour_index_1based",
        "neighbour_nucleotide",
        "sequence_separation",
        "centroid_mse",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for token_reps in reps:
            for rank, rep in enumerate(token_reps, start=1):
                token = rep["token"]
                p = profiles[token]
                w.writerow(
                    {
                        "token": token,
                        "representative_rank": rank,
                        "n": int(counts[token]),
                        "frequency": p["freq"],
                        "role": p["role"],
                        "structure_id": Path(rep["id"]).name,
                        "residue_index_1based": rep["raw_i"] + 1,
                        "nucleotide": rep["seq_i"],
                        "neighbour_index_1based": rep["raw_j"] + 1,
                        "neighbour_nucleotide": rep["seq_j"],
                        "sequence_separation": rep["seq_sep"],
                        "centroid_mse": f"{rep['centroid_mse']:.6g}",
                    }
                )


def plot(output, reps):
    fig, axes = plt.subplots(10, 8, figsize=(12.8, 16.0))
    for token, ax in enumerate(axes.flat):
        draw_state(ax, reps[token], token)

    fig.subplots_adjust(left=0.012, right=0.988, top=0.995, bottom=0.018, wspace=0.03, hspace=0.16)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    db = torch.load(args.db, map_location="cpu", weights_only=False)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    n_codes = int(db["n_codes"])
    if n_codes != 80:
        raise ValueError(f"Expected the production 80-state model, found K={n_codes}")
    norm_mean = ckpt["norm_mean"].float()
    norm_std = ckpt["norm_std"].float().clamp_min(1e-8)
    wanted = {e["id"] for e in db["entries"]}
    sources = load_sources(args.data, wanted)
    profiles = load_profiles(args.profile)

    reps, counts = select_representatives(
        db["entries"],
        sources,
        norm_mean,
        norm_std,
        n_codes,
        n_examples=args.n_examples,
    )
    if any(not r for r in reps):
        raise RuntimeError("Representative selection failed for one or more tokens")
    write_table(args.table, reps, profiles, counts)
    plot(args.output, reps)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.table}")


if __name__ == "__main__":
    main()
