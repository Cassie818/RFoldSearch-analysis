"""
Build a searchable token database from a JSON of RNA structures.

Encodes every structure to a VQ-VAE token sequence and stores it with
metadata (id, length, C4' coords for later RMSD verification).

Output: a single .pt file loadable as:
    {
      "n_codes":  int,
      "entries":  [ { "id": str, "tokens": np.ndarray (L,),
                      "coords": np.ndarray (L_raw, 3, 3),
                      "mask":   np.ndarray (L_raw,),
                      "n_valid": int }, ... ],
      "checkpoint": str,
    }

Usage
-----
    python build_database.py \\
      --checkpoint ../files/pdb_out/desc_vqvae_best.pt \\
      --data ../Splits/pdb_out/test.json \\
      --output ../files/search/pdb_out/db.pt

    # Build one database from multiple partitions.
    python build_database.py \\
      --checkpoint ../files/pdb_out/desc_vqvae_best.pt \\
      --data Splits/pdb_out/train.json Splits/pdb_out/val.json Splits/pdb_out/test.json \\
      --output ../files/search/pdb_out/all_db.pt
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models"))
from vqvae import (
    compute_descriptor,
    DescriptorVQVAE,
    infer_decoder_out_dim,
    infer_n_dec_layers,
    infer_n_enc_layers,
)


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
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
    ).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model, ckpt, n_codes, d_descriptor


def prepare_sample(entry, device="cpu"):
    coords = torch.tensor(entry["coords"], dtype=torch.float32).unsqueeze(0).to(device)
    mask = torch.tensor(entry["mask"], dtype=torch.bool).unsqueeze(0).to(device)
    # New extract_backbone.py emits "pair_idx" / "stack_idx"; older
    # snapshots used "pair_edges" / "stack_edges". Accept either.
    pair_key = "pair_idx" if "pair_idx" in entry else "pair_edges"
    stack_key = "stack_idx" if "stack_idx" in entry else "stack_edges"
    pair_edges = torch.tensor(entry[pair_key], dtype=torch.long).unsqueeze(0).to(device)
    stack_edges = torch.tensor(entry[stack_key], dtype=torch.long).unsqueeze(0).to(device)
    pair_orient = None
    if "pair_orient" in entry:
        pair_orient = torch.tensor(entry["pair_orient"], dtype=torch.long).unsqueeze(0).to(device)

    if mask.sum() == 0:
        return None
    if pair_edges.ndim == 3:
        pair_edges = pair_edges[..., 0]
    if stack_edges.ndim == 3:
        stack_edges = stack_edges[..., 0]
    if pair_orient is not None and pair_orient.ndim == 3:
        pair_orient = pair_orient[..., 0]
    if (
        coords.shape[1] != mask.shape[1]
        or pair_edges.shape[1] != mask.shape[1]
        or stack_edges.shape[1] != mask.shape[1]
        or (pair_orient is not None and pair_orient.shape[1] != mask.shape[1])
    ):
        return None
    return coords, mask, pair_edges, stack_edges, pair_orient


@torch.no_grad()
def encode_entry(model, entry, device, norm_mean, norm_std, zero_dim=-1, zero_dims=None):
    sample = prepare_sample(entry, device)
    if sample is None:
        return None
    coords, mask, pair_edges, stack_edges, pair_orient = sample
    descriptor, _info = compute_descriptor(
        coords,
        mask,
        pair_edges,
        stack_edges,
        pair_orient=pair_orient,
    )

    if norm_mean is not None and norm_std is not None:
        descriptor = (descriptor - norm_mean.to(device)) / norm_std.to(device)
    if zero_dims:
        descriptor = descriptor.clone()
        for d in zero_dims:
            if 0 <= d < descriptor.shape[-1]:
                descriptor[..., d] = 0.0
    elif zero_dim is not None and zero_dim >= 0 and zero_dim < descriptor.shape[-1]:
        descriptor = descriptor.clone()
        descriptor[..., zero_dim] = 0.0

    output = model(descriptor)
    valid_mask = mask[0].cpu()
    tokens = output["indices"][0].cpu()[valid_mask].numpy().astype(np.int16)
    return {
        "id": entry.get("id", ""),
        "tokens": tokens,
        "coords": coords[0].cpu().numpy(),
        "mask": mask[0].cpu().numpy(),
        "n_valid": int(valid_mask.sum()),
    }


def main():
    parent = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    parser = argparse.ArgumentParser(description="Build searchable token database")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(parent, "files", "pdb_out", "desc_vqvae_best.pt"),
    )
    parser.add_argument(
        "--data",
        type=str,
        nargs="+",
        default=[os.path.join(parent, "Splits", "pdb_out", "test.json")],
    )
    parser.add_argument(
        "--output", type=str, default=os.path.join(parent, "files", "search", "pdb_out", "db.pt")
    )
    parser.add_argument("--min_len", type=int, default=20)
    parser.add_argument(
        "--zero_dim",
        type=int,
        default=-1,
        help="If >=0, zero this dim of the normalized descriptor "
        "before encoding (per-dim ablation). -1 disables.",
    )
    parser.add_argument(
        "--zero_dims",
        type=str,
        default="",
        help="Comma-separated normalized descriptor dims to zero "
        "before encoding (e.g. 0,1,2). Overrides --zero_dim.",
    )
    args = parser.parse_args()

    device = torch.device("cpu")

    # ── Load model ──
    print("Loading model from %s ..." % args.checkpoint)
    model, ckpt, n_codes, d_descriptor = load_model(args.checkpoint, device)
    norm_mean = ckpt.get("norm_mean", None)
    norm_std = ckpt.get("norm_std", None)
    zero_dims = []
    if args.zero_dims.strip():
        zero_dims = [int(x.strip()) for x in args.zero_dims.split(",") if x.strip()]
    print("  %d codes, %dD" % (n_codes, d_descriptor))

    # ── Load data ──
    data_paths = args.data if isinstance(args.data, list) else [args.data]
    print("Loading structures from %s ..." % ", ".join(data_paths))
    structures = []
    seen_ids = set()
    n_read, n_short, n_duplicate = 0, 0, 0
    for data_path in data_paths:
        with open(data_path) as f:
            for line in f:
                e = json.loads(line)
                n_read += 1
                entry_id = str(e.get("id", ""))
                if entry_id in seen_ids:
                    n_duplicate += 1
                    continue
                seen_ids.add(entry_id)
                if sum(e["mask"]) < args.min_len:
                    n_short += 1
                    continue
                e["_source_json"] = data_path
                structures.append(e)
    print(
        "  %d structures (read=%d, short=%d, duplicate=%d)"
        % (len(structures), n_read, n_short, n_duplicate)
    )

    # ── Encode ──
    entries = []
    for i, e in enumerate(structures):
        ent = encode_entry(
            model, e, device, norm_mean, norm_std, zero_dim=args.zero_dim, zero_dims=zero_dims
        )
        if ent is not None:
            entries.append(ent)
        if (i + 1) % 100 == 0:
            print("  encoded %d/%d" % (i + 1, len(structures)), flush=True)
    print("  total entries: %d" % len(entries))

    # ── Save ──
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(
        {
            "n_codes": n_codes,
            "entries": entries,
            "checkpoint": args.checkpoint,
            "data": data_paths,
        },
        args.output,
    )
    lengths = [e["n_valid"] for e in entries]
    print("\nSaved: %s" % args.output)
    print(
        "  Entries: %d  (length: mean=%.0f, median=%.0f, max=%d)"
        % (len(entries), np.mean(lengths), np.median(lengths), max(lengths))
    )


if __name__ == "__main__":
    main()
