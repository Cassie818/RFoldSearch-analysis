"""
Build a VQ-VAE token substitution matrix from codebook embeddings.

The substitution score between token i and token j is the cosine similarity
of their codebook embedding vectors.  No alignment data required.

Outputs
-------
  sub_matrix.pt   : (N, N) float32 tensor of cosine similarities [-1, 1]
  sub_matrix.tsv  : tab-separated N×N matrix (human-readable)
  sub_matrix.png  : heatmap visualization

Usage
-----
  python build_substitution_matrix.py --checkpoint ../files/pdb_out/desc_vqvae_best.pt
  python build_substitution_matrix.py --checkpoint ../files/pdb_out/desc_vqvae_best.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models"))
from vqvae import (
    DescriptorVQVAE,
    infer_decoder_out_dim,
    infer_n_dec_layers,
    infer_n_enc_layers,
)


def build_matrix(checkpoint_path: str, save_dir: str, figures_dir: str | None = None):
    os.makedirs(save_dir, exist_ok=True)

    # ── Load checkpoint ──
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sd = ckpt["model"]

    d_descriptor = ckpt.get("d_descriptor", infer_decoder_out_dim(sd))
    d_hidden = ckpt.get("d_hidden", sd["encoder.net.0.weight"].shape[0])
    d_latent = ckpt.get("d_latent", sd["codebook.embeddings"].shape[1])
    n_codes = sd["codebook.embeddings"].shape[0]

    n_enc = ckpt.get("n_enc_layers", infer_n_enc_layers(sd))
    n_dec = ckpt.get("n_dec_layers", infer_n_dec_layers(sd))
    ema_decay = ckpt.get("ema_decay", 0.99)
    model = DescriptorVQVAE(
        d_descriptor=d_descriptor,
        d_hidden=d_hidden,
        d_latent=d_latent,
        n_codes=n_codes,
        n_enc_layers=n_enc,
        n_dec_layers=n_dec,
        ema_decay=ema_decay,
    )
    model.load_state_dict(sd)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    print(
        "Loaded checkpoint: epoch %s, %d codes, d_latent=%d, %dD"
        % (epoch, n_codes, d_latent, d_descriptor)
    )

    # ── Extract codebook embeddings ──
    # shape: (N_CODES, D_LATENT)
    embeddings = model.codebook.embeddings.detach()  # (N, D)
    print("Codebook embeddings: %s" % str(embeddings.shape))

    # ── Cosine similarity matrix ──
    norm_emb = F.normalize(embeddings, dim=-1)  # (N, D), unit vectors
    sim_matrix = norm_emb @ norm_emb.T  # (N, N), values in [-1, 1]
    sim_np = sim_matrix.numpy()

    # ── Save .pt ──
    pt_path = os.path.join(save_dir, "sub_matrix.pt")
    torch.save(
        {
            "matrix": sim_matrix,
            "n_codes": n_codes,
            "method": "codebook-cosine",
            "d_latent": d_latent,
            "epoch": epoch,
            "checkpoint": checkpoint_path,
        },
        pt_path,
    )
    print("Saved: %s" % pt_path)

    # ── Save .tsv ──
    tsv_path = os.path.join(save_dir, "sub_matrix.tsv")
    np.savetxt(tsv_path, sim_np, delimiter="\t", fmt="%.4f")
    print("Saved: %s" % tsv_path)

    # ── Statistics ──
    # Mask diagonal (self-similarity = 1.0 by definition)
    off_diag = sim_np[~np.eye(n_codes, dtype=bool)]
    print("\nSubstitution matrix statistics:")
    print(
        "  Diagonal (self):   min=%.4f  max=%.4f  mean=%.4f"
        % (sim_np.diagonal().min(), sim_np.diagonal().max(), sim_np.diagonal().mean())
    )
    print(
        "  Off-diagonal:      min=%.4f  max=%.4f  mean=%.4f  std=%.4f"
        % (off_diag.min(), off_diag.max(), off_diag.mean(), off_diag.std())
    )

    # How many pairs are "similar" (sim > 0.5)?
    sim_pairs = (off_diag > 0.5).sum()
    total_pairs = len(off_diag)
    print(
        "  Pairs with sim>0.5: %d / %d (%.1f%%)"
        % (sim_pairs, total_pairs, 100.0 * sim_pairs / total_pairs)
    )
    print(
        "  Pairs with sim>0.8: %d / %d (%.1f%%)"
        % ((off_diag > 0.8).sum(), total_pairs, 100.0 * (off_diag > 0.8).sum() / total_pairs)
    )

    # Token neighborhood: for each token, how many others are "close"?
    neighbor_counts = (sim_matrix - torch.eye(n_codes) > 0.5).sum(dim=1).numpy()
    print(
        "  Avg neighbors (sim>0.5) per token: %.1f  (min=%d, max=%d)"
        % (neighbor_counts.mean(), neighbor_counts.min(), neighbor_counts.max())
    )

    # ── Plot heatmap ──
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        "RNA VQ-VAE Token Substitution Matrix  (epoch %s, %d codes)" % (epoch, n_codes), fontsize=14
    )

    # (a) Full heatmap
    ax = axes[0]
    im = ax.imshow(sim_np, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cosine Similarity")
    ax.set_xlabel("Token index")
    ax.set_ylabel("Token index")
    ax.set_title("Full %d×%d Substitution Matrix" % (n_codes, n_codes))

    # (b) Off-diagonal similarity distribution
    ax = axes[1]
    ax.hist(off_diag, bins=80, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1.2, label="sim=0")
    ax.axvline(x=0.5, color="orange", linestyle="--", linewidth=1.2, label="sim=0.5")
    ax.axvline(x=0.8, color="red", linestyle="--", linewidth=1.2, label="sim=0.8")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    ax.set_title(
        "Off-diagonal Similarity Distribution\nmean=%.3f  std=%.3f"
        % (off_diag.mean(), off_diag.std())
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    png_dir = figures_dir or save_dir
    os.makedirs(png_dir, exist_ok=True)
    png_path = os.path.join(png_dir, "sub_matrix.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: %s" % png_path)

    # ── Top similar pairs ──
    print("\nTop 10 most similar token pairs (off-diagonal):")
    upper = np.triu(sim_np, k=1)
    flat = upper.flatten()
    top_k = np.argsort(flat)[::-1][:10]
    for rank, idx in enumerate(top_k):
        i, j = divmod(idx, n_codes)
        print("  %2d. token %3d <-> token %3d  sim=%.4f" % (rank + 1, i, j, sim_np[i, j]))

    print("\nDone. Files saved to: %s" % save_dir)
    return sim_matrix


def main():
    parent = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    parser = argparse.ArgumentParser(
        description="Build substitution matrix from codebook embeddings"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(parent, "files", "pdb_out", "desc_vqvae_best.pt"),
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join(parent, "files", "codebook"),
        help="Where to write .pt and .tsv outputs",
    )
    parser.add_argument(
        "--figures_dir",
        type=str,
        default=os.path.join(parent, "figures", "codebook"),
        help="Where to write the PNG figure",
    )
    args = parser.parse_args()

    build_matrix(args.checkpoint, args.save_dir, args.figures_dir)


if __name__ == "__main__":
    main()
