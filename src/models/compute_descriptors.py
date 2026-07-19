"""Pre-compute the 49-D structural descriptor for VQ-VAE training.

Descriptor layout (see `descriptors/compose.py` for details):
  [FRAME]   g_nn (k=1)  10D    nearest spatial neighbour
            g_nn (k=2)  10D    second-nearest neighbour
            g_bp        10D    base-pair partner xform
            m_bp         1D    base-pair availability mask
            g_st        10D    stacking partner xform
            m_st         1D    stacking availability mask
  [SCALAR]  g_ctx         7D    seq-sep × 3 + cis/trans + contact density
  ────────────────────────────
  Total:               49D

Saves:
  - descriptors: (N_total, 49) flat tensor
  - stats: {mean, std} computed from training set (used for z-score
    normalisation at training time)

Usage:
    python compute_descriptors.py --input train.json --output train_desc.pt --save_stats
    python compute_descriptors.py --input test.json  --output test_desc.pt
    python compute_descriptors.py --split_dir /path/to/Splits
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from descriptors import DESCRIPTOR_WIDTH, compute_descriptor  


def compute_descriptors(input_path: str, min_len: int) -> torch.Tensor:
    all_descriptors = []
    n_structures = 0
    n_skipped = 0
    n_bad_shape = 0

    with open(input_path) as f:
        for line in f:
            entry = json.loads(line)
            if sum(entry["mask"]) < min_len:
                n_skipped += 1
                continue

            coords = torch.tensor(entry["coords"], dtype=torch.float32).unsqueeze(0)
            mask = torch.tensor(entry["mask"], dtype=torch.bool).unsqueeze(0)
            pair_key = "pair_idx" if "pair_idx" in entry else "pair_edges"
            stack_key = "stack_idx" if "stack_idx" in entry else "stack_edges"
            pair_edges = torch.tensor(entry[pair_key], dtype=torch.long).unsqueeze(0)
            stack_edges = torch.tensor(entry[stack_key], dtype=torch.long).unsqueeze(0)
            pair_orient = None
            if "pair_orient" in entry:
                pair_orient = torch.tensor(
                    entry["pair_orient"], dtype=torch.long
                ).unsqueeze(0)

            if pair_edges.ndim == 3:
                pair_edges = pair_edges[..., 0]
            if stack_edges.ndim == 3:
                stack_edges = stack_edges[..., 0]
            if pair_orient is not None and pair_orient.ndim == 3:
                pair_orient = pair_orient[..., 0]

            L_mask = mask.shape[1]
            if (coords.shape[1] != L_mask
                    or pair_edges.shape[1] != L_mask
                    or stack_edges.shape[1] != L_mask
                    or (pair_orient is not None and pair_orient.shape[1] != L_mask)):
                n_bad_shape += 1
                print("  [SKIP shape] %s coords=%d mask=%d" %
                      (entry.get("id", "?"), coords.shape[1], L_mask), flush=True)
                continue

            try:
                with torch.no_grad():
                    descriptor, _info = compute_descriptor(
                        coords, mask, pair_edges, stack_edges,
                        pair_orient=pair_orient,
                    )
            except Exception as e:
                n_bad_shape += 1
                print("  [SKIP error] %s: %s" % (entry.get("id", "?"), e),
                      flush=True)
                continue

            if descriptor.shape[-1] != DESCRIPTOR_WIDTH:
                n_bad_shape += 1
                print("  [SKIP descriptor] %s expected %dD, got %dD" % (
                    entry.get("id", "?"), DESCRIPTOR_WIDTH, descriptor.shape[-1]),
                    flush=True)
                continue

            valid = mask[0]
            desc_valid = descriptor[0][valid]
            if torch.isnan(desc_valid).any() or torch.isinf(desc_valid).any():
                desc_valid = torch.nan_to_num(
                    desc_valid, nan=0.0, posinf=0.0, neginf=0.0
                )

            all_descriptors.append(desc_valid)
            n_structures += 1

            if n_structures % 50 == 0:
                total_res = sum(d.shape[0] for d in all_descriptors)
                print("  %d structures, %d residues so far..." %
                      (n_structures, total_res), flush=True)

    if not all_descriptors:
        raise RuntimeError("No valid structures found; no descriptors were written")

    all_desc = torch.cat(all_descriptors, dim=0)
    print("\nDone.")
    print("  Structures: %d processed, %d skipped (short), %d skipped (bad shape)" %
          (n_structures, n_skipped, n_bad_shape))
    print("  Total residues: %d" % all_desc.shape[0])
    print("  Descriptor shape: %s" % str(all_desc.shape))
    return all_desc


# Dim positions in the 49-D layout — m_bp and m_st are always kept
# binary (mean=0, std=1) so that the BCE mask loss is well-defined.
MASK_DIMS = [30, 41]
BP_DIMS = list(range(20, 30))    # g_bp
ST_DIMS = list(range(31, 41))    # g_st

DIM_NAMES = (
    ["nn1_" + s for s in ["dist", "dx", "dy", "dz", "r00", "r10", "r20", "r01", "r11", "r21"]] +
    ["nn2_" + s for s in ["dist", "dx", "dy", "dz", "r00", "r10", "r20", "r01", "r11", "r21"]] +
    ["bp_" + s for s in ["dist", "dx", "dy", "dz", "r00", "r10", "r20", "r01", "r11", "r21"]] +
    ["m_bp"] +
    ["st_" + s for s in ["dist", "dx", "dy", "dz", "r00", "r10", "r20", "r01", "r11", "r21"]] +
    ["m_st"] +
    ["ctx_" + s for s in ["nn_seqsep", "bp_seqsep", "st_seqsep",
                          "bp_cis", "bp_trans", "contact8", "contact12"]]
)
assert len(DIM_NAMES) == DESCRIPTOR_WIDTH


def save_stats(all_desc: torch.Tensor, stats_path: str, conditional_stats: bool):
    mean = all_desc.mean(dim=0)
    std = all_desc.std(dim=0).clamp(min=1e-6)

    if conditional_stats:
        bp_active = all_desc[:, MASK_DIMS[0]] > 0.5
        st_active = all_desc[:, MASK_DIMS[1]] > 0.5
        if bp_active.any():
            mean[BP_DIMS] = all_desc[bp_active][:, BP_DIMS].mean(dim=0)
            std[BP_DIMS] = all_desc[bp_active][:, BP_DIMS].std(dim=0).clamp(min=1e-6)
            print("  Conditional bp stats from %d active residues" %
                  int(bp_active.sum().item()))
        if st_active.any():
            mean[ST_DIMS] = all_desc[st_active][:, ST_DIMS].mean(dim=0)
            std[ST_DIMS] = all_desc[st_active][:, ST_DIMS].std(dim=0).clamp(min=1e-6)
            print("  Conditional stack stats from %d active residues" %
                  int(st_active.sum().item()))

    for d in MASK_DIMS:
        mean[d] = 0.0
        std[d] = 1.0
    print("  Mask dims (m_bp=%d, m_st=%d) kept binary (mean=0, std=1)" %
          (MASK_DIMS[0], MASK_DIMS[1]))

    torch.save({
        "mean": mean,
        "std": std,
        "d_descriptor": int(all_desc.shape[1]),
        "conditional_stats": bool(conditional_stats),
    }, stats_path)
    print("  Stats saved to: %s" % stats_path)

    print("\n  Per-dim stats:")
    for i in range(all_desc.shape[1]):
        print("    %-12s  mean=%7.3f  std=%7.3f" % (DIM_NAMES[i], mean[i], std[i]))


def run_single(input_path: str, output_path: str, min_len: int,
               save_train_stats: bool, conditional_stats: bool):
    print("\nComputing descriptors: %s" % input_path)
    all_desc = compute_descriptors(input_path, min_len)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(all_desc, output_path)
    print("  Saved to: %s" % output_path)
    if save_train_stats:
        stats_path = os.path.join(os.path.dirname(output_path), "desc_stats.pt")
        save_stats(all_desc, stats_path, conditional_stats)


def run_split_dir(split_dir: str, out_dir: str, min_len: int,
                  conditional_stats: bool):
    split_names = []
    for name in ("pdb_out",):
        if os.path.exists(os.path.join(split_dir, name, "train.json")):
            split_names.append(name)
    if not split_names:
        raise FileNotFoundError(
            "No split subdirectories with train.json under %s" % split_dir
        )
    for split_name in split_names:
        for input_name, output_name, save_train_stats in (
            ("train.json", "train_desc.pt", True),
            ("val.json", "val_desc.pt", False),
            ("test.json", "test_desc.pt", False),
        ):
            input_path = os.path.join(split_dir, split_name, input_name)
            if not os.path.exists(input_path):
                print("  [SKIP] %s not found in %s" % (input_name, split_name))
                continue
            output_path = os.path.join(out_dir, split_name, output_name)
            run_single(input_path, output_path, min_len,
                       save_train_stats, conditional_stats)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute the 49-D structural descriptor for VQ-VAE training."
    )
    parser.add_argument("--input", type=str,
                        help="Input JSON for single-file mode")
    parser.add_argument("--output", type=str,
                        help="Output .pt file for single-file mode")
    parser.add_argument("--split_dir", type=str,
                        help="Directory containing split JSONs")
    parser.add_argument("--out_dir", type=str,
                        help="Descriptor output directory for --split_dir mode")
    parser.add_argument("--save_stats", action="store_true",
                        help="Single-file mode: compute and save mean/std")
    parser.add_argument("--conditional_stats", action="store_true",
                        help="Compute bp/stack geometry stats only on active "
                             "masked residues (default: marginal over all)")
    parser.add_argument("--min_len", type=int, default=20,
                        help="Skip structures with fewer valid residues")
    args = parser.parse_args()

    if args.split_dir:
        out_dir = args.out_dir or args.split_dir
        run_split_dir(args.split_dir, out_dir, args.min_len, args.conditional_stats)
        return

    if not args.input or not args.output:
        parser.error("single-file mode requires --input and --output, "
                     "or use --split_dir")
    run_single(args.input, args.output, args.min_len,
               args.save_stats, args.conditional_stats)


if __name__ == "__main__":
    main()
