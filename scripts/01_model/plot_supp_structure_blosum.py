"""Supplementary figure: the deployed Structure-BLOSUM substitution matrix.

Reads the deployed matrix from files/models/sub_matrix.pt (log-odds
scores, not cosine).

Panel A : 80x80 token-to-token log-odds heatmap, with token order permuted by
          hierarchical clustering of the matrix itself so that structurally
          similar tokens form visible block-diagonal groups.
Panel B : off-diagonal log-odds distribution (negative entries = structurally
          dissimilar pairs; positive entries = structurally equivalent tokens
          that the matrix lets Smith--Waterman trade).

Writes: figures/supp_structure_blosum.png
"""
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from paths import DATA_ROOT, FIG_DIR, SRC_MODELS, SRC_SEARCH, resolve

_sys.path.insert(0, SRC_SEARCH)
_sys.path.insert(0, SRC_MODELS)
_os.chdir(DATA_ROOT)
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

REPO = Path(__file__).resolve().parent.parent
SRC = Path(resolve("files/models/sub_matrix.pt"))
OUT = Path(FIG_DIR) / "supp_structure_blosum.png"

# Muted blue--ivory--coral diverging palette: negative substitutions remain
# visually distinct without the very pale cyan cast of RdBu_r.
STRUCTURE_BLOSUM_CMAP = LinearSegmentedColormap.from_list(
    "structure_blosum",
    ["#244C66", "#6F9FB5", "#F7F3EA", "#E89A78", "#A63D4D"],
    N=256,
)


def main():
    d = torch.load(SRC, map_location="cpu", weights_only=False)
    M = d["matrix"].numpy() if isinstance(d["matrix"], torch.Tensor) else np.asarray(d["matrix"])
    K = M.shape[0]
    assert M.shape == (K, K), f"unexpected matrix shape {M.shape}"

    # Hierarchical clustering on the matrix as a similarity → distance
    dist = float(M.max()) - M
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2  # symmetrise to silence FP noise
    np.clip(dist, 0, None, out=dist)
    cond = squareform(dist, checks=False)
    order = leaves_list(linkage(cond, method="average"))
    M_clust = M[np.ix_(order, order)]

    off = M[np.triu_indices(K, k=1)]

    fig = plt.figure(figsize=(12.6, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.30)

    # --- Panel A : clustered heatmap ---
    ax = fig.add_subplot(gs[0, 0])
    vext = float(np.abs([M.min(), M.max()]).max())
    im = ax.imshow(
        M_clust,
        cmap=STRUCTURE_BLOSUM_CMAP,
        vmin=-vext,
        vmax=vext,
        aspect="equal",
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Substitution score (log-odds)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    ax.set_xlabel("Token (clustered)", fontsize=11)
    ax.set_ylabel("Token (clustered)", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        -0.10,
        1.04,
        "A",
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="bottom",
        ha="right",
    )

    # --- Panel B : off-diagonal distribution ---
    ax = fig.add_subplot(gs[0, 1])
    counts, edges = np.histogram(off, bins=30)
    centres = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    score_norm = TwoSlopeNorm(vmin=float(off.min()), vcenter=0.0, vmax=float(off.max()))
    ax.bar(
        centres,
        counts,
        width=widths * 1.01,
        color=STRUCTURE_BLOSUM_CMAP(score_norm(centres)),
        edgecolor="none",
        align="center",
    )
    ax.set_xlabel("Substitution score (off-diagonal)", fontsize=11)
    ax.set_ylabel("Token-pair count", fontsize=11)
    _pad = 0.05 * (off.max() - off.min())
    ax.set_xlim(off.min() - _pad, off.max() + _pad)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        -0.10,
        1.04,
        "B",
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="bottom",
        ha="right",
    )

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    print(f"wrote {OUT}  ({K}×{K}, off-diag mean {off.mean():.3f}, " f"std {off.std():.3f})")


if __name__ == "__main__":
    main()
