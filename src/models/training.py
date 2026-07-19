"""
Training script for RNA Structure Descriptor VQ-VAE.

Usage:
       python models/training.py --split pdb_out
"""

import os
import csv
import math
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset

from descriptors import DESCRIPTOR_WIDTH
from vqvae import DescriptorVQVAE, descriptor_loss


def parse_cli_into(cfg):
    """Override Config defaults via CLI for split-specific runs."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--split", default=cfg.SPLIT_NAME, help="Which prepared split under --split_dir to train on"
    )
    p.add_argument(
        "--split_dir",
        default=cfg.SPLIT_DIR,
        help="Root directory containing prepared split folders",
    )
    p.add_argument("--train_desc", default=None, help="Override train descriptor path")
    p.add_argument("--val_desc", default=None, help="Override val descriptor path")
    p.add_argument("--test_desc", default=None, help="Override test descriptor path")
    p.add_argument("--stats", default=None, help="Override descriptor normalization stats path")
    p.add_argument("--save_dir", default=None, help="Override model output directory")
    p.add_argument("--epochs", type=int, default=cfg.NUM_EPOCHS)
    p.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--n_codes", type=int, default=cfg.N_CODES)
    p.add_argument("--n_enc_layers", type=int, default=cfg.N_ENC_LAYERS)
    p.add_argument("--n_dec_layers", type=int, default=cfg.N_DEC_LAYERS)
    p.add_argument("--d_hidden", type=int, default=cfg.D_HIDDEN)
    p.add_argument("--d_latent", type=int, default=cfg.D_LATENT)
    p.add_argument("--beta_commit", type=float, default=cfg.BETA_COMMIT)
    p.add_argument("--ema_decay", type=float, default=cfg.EMA_DECAY)
    p.add_argument("--freeze_codebook_after", type=int, default=cfg.FREEZE_CODEBOOK_AFTER)
    p.add_argument("--lambda_geom", type=float, default=cfg.LAMBDA_GEOM)
    p.add_argument("--lambda_pair", type=float, default=cfg.LAMBDA_PAIR)
    p.add_argument("--lambda_stack", type=float, default=cfg.LAMBDA_STACK)
    p.add_argument("--lambda_mask", type=float, default=cfg.LAMBDA_MASK)
    p.add_argument("--pair_stack_reduce", choices=["sum", "mean"], default=cfg.PAIR_STACK_REDUCE)
    p.add_argument("--lambda_latent_norm", type=float, default=cfg.LAMBDA_LATENT_NORM)
    p.add_argument("--lr", type=float, default=cfg.LR)
    p.add_argument(
        "--active_blocks",
        default=None,
        help="Comma-separated subset of "
        "{g_nn1,g_nn2,g_bp,g_st,g_ctx} indicating which "
        "descriptor blocks should contribute to the loss. "
        "Used in feature-ablation runs to avoid scoring "
        "decoder output on zeroed dimensions. Defaults to "
        "all blocks active.",
    )
    a = p.parse_args()

    # ----------------------------------------------------------------------------------------------#
    cfg.SPLIT_NAME = a.split
    cfg.SPLIT_DIR = a.split_dir
    split_path = os.path.join(cfg.SPLIT_DIR, cfg.SPLIT_NAME)
    cfg.TRAIN_DESC_PATH = a.train_desc or os.path.join(split_path, "train_desc.pt")
    cfg.VAL_DESC_PATH = a.val_desc or os.path.join(split_path, "val_desc.pt")
    cfg.TEST_DESC_PATH = a.test_desc or os.path.join(split_path, "test_desc.pt")
    cfg.STATS_PATH = a.stats or os.path.join(split_path, "desc_stats.pt")
    cfg.MODEL_SAVE_PATH = a.save_dir or os.path.join(cfg.FILES_DIR, cfg.SPLIT_NAME)
    cfg.NUM_EPOCHS = a.epochs
    cfg.BATCH_SIZE = a.batch_size
    cfg.N_CODES = a.n_codes
    cfg.N_ENC_LAYERS = a.n_enc_layers
    cfg.N_DEC_LAYERS = a.n_dec_layers
    cfg.D_HIDDEN = a.d_hidden
    cfg.D_LATENT = a.d_latent
    cfg.BETA_COMMIT = a.beta_commit
    cfg.EMA_DECAY = a.ema_decay
    cfg.FREEZE_CODEBOOK_AFTER = a.freeze_codebook_after
    cfg.LAMBDA_GEOM = a.lambda_geom
    cfg.LAMBDA_PAIR = a.lambda_pair
    cfg.LAMBDA_STACK = a.lambda_stack
    cfg.LAMBDA_MASK = a.lambda_mask
    cfg.PAIR_STACK_REDUCE = a.pair_stack_reduce
    cfg.LAMBDA_LATENT_NORM = a.lambda_latent_norm
    cfg.LR = a.lr
    if a.active_blocks:
        cfg.ACTIVE_BLOCKS = {b.strip() for b in a.active_blocks.split(",") if b.strip()}
    else:
        cfg.ACTIVE_BLOCKS = None
    return cfg


class Config:
    # Data — pre-computed descriptor .pt files
    SPLIT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Splits")
    SPLIT_NAME = "pdb_out"
    TRAIN_DESC_PATH = None
    VAL_DESC_PATH = None
    TEST_DESC_PATH = None
    STATS_PATH = None
    MODEL_SAVE_PATH = None
    FILES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "files")

    # Descriptor dims. The loader below still overwrites these from the
    # descriptor tensor shape, but the default should match the canonical
    # descriptor to keep logs and ad-hoc runs honest.
    D_STRUCTURAL = DESCRIPTOR_WIDTH
    D_DESCRIPTOR = D_STRUCTURAL
    D_HIDDEN = 128
    D_LATENT = 128
    N_CODES = 60
    N_ENC_LAYERS = 2
    N_DEC_LAYERS = 2

    # Training
    BATCH_SIZE = 512
    NUM_EPOCHS = 50
    LR = 3e-4
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 200
    CLIP_GRAD = 1.0
    SAVE_EVERY = 50
    BETA_COMMIT = 1.0  # z is L2-normalized → commit MSE bounded; standard β
    EMA_DECAY = 0.99
    FREEZE_CODEBOOK_AFTER = 0  # 0 disables freezing
    LAMBDA_GEOM = 1.0
    LAMBDA_PAIR = 1.0
    LAMBDA_STACK = 1.0
    LAMBDA_MASK = 1.0
    PAIR_STACK_REDUCE = "sum"
    LAMBDA_LATENT_NORM = 0.0  # redundant once z is L2-normalized
    ACTIVE_BLOCKS = None

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize(desc, mean, std):
    """Z-score normalize: (x - mean) / std."""
    return (desc - mean) / std


def denormalize(desc, mean, std):
    """Inverse z-score: x * std + mean."""
    return desc * std + mean


LOSS_KEYS = [
    "loss",
    "L_recon",
    "L_geom",
    "L_pair",
    "L_stack",
    "L_mask",
    "L_commit",
    "L_latent",
]


def compute_loss(output, batch_norm, cfg):
    return descriptor_loss(
        output,
        batch_norm,
        beta_commit=cfg.BETA_COMMIT,
        lambda_geom=cfg.LAMBDA_GEOM,
        lambda_pair=cfg.LAMBDA_PAIR,
        lambda_stack=cfg.LAMBDA_STACK,
        lambda_mask=cfg.LAMBDA_MASK,
        pair_stack_reduce=cfg.PAIR_STACK_REDUCE,
        lambda_latent_norm=cfg.LAMBDA_LATENT_NORM,
        active_blocks=cfg.ACTIVE_BLOCKS,
    )


def train_epoch(model, dataloader, optimizer, scheduler, cfg, mean, std):
    model.train()
    totals = {k: 0.0 for k in LOSS_KEYS}
    num_batches = 0

    for (batch_desc,) in dataloader:
        batch_desc = batch_desc.to(cfg.DEVICE)

        # Normalize structural descriptor
        batch_norm = normalize(batch_desc, mean, std)

        output = model(batch_norm)
        losses = compute_loss(output, batch_norm, cfg)

        if torch.isnan(losses["loss"]) or torch.isinf(losses["loss"]):
            continue

        optimizer.zero_grad()
        losses["loss"].backward()
        if cfg.CLIP_GRAD > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_GRAD)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        for k in LOSS_KEYS:
            totals[k] += losses[k].item()
        num_batches += 1

    n = max(1, num_batches)
    return {k: totals[k] / n for k in LOSS_KEYS}


@torch.no_grad()
def evaluate(model, dataloader, cfg, mean, std):
    model.eval()
    totals = {k: 0.0 for k in LOSS_KEYS}
    num_batches = 0
    all_indices = []

    for (batch_desc,) in dataloader:
        batch_desc = batch_desc.to(cfg.DEVICE)
        batch_norm = normalize(batch_desc, mean, std)

        output = model(batch_norm)
        losses = compute_loss(output, batch_norm, cfg)

        for k in LOSS_KEYS:
            totals[k] += losses[k].item()
        num_batches += 1
        all_indices.append(output["indices"].cpu().flatten())

    n = max(1, num_batches)
    all_indices = torch.cat(all_indices, dim=0)
    n_used = all_indices.unique().numel()

    result = {k: totals[k] / n for k in LOSS_KEYS}
    result["codebook_usage"] = n_used
    result["codebook_usage_pct"] = n_used / cfg.N_CODES * 100
    return result


def save_checkpoint(model, optimizer, scheduler, epoch, loss, cfg, stats, filename=None):
    os.makedirs(cfg.MODEL_SAVE_PATH, exist_ok=True)
    if filename is None:
        filename = "desc_vqvae_epoch%d.pt" % epoch
    save_path = os.path.join(cfg.MODEL_SAVE_PATH, filename)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "loss": float(loss),
            "n_codes": cfg.N_CODES,
            "d_descriptor": cfg.D_STRUCTURAL,
            "d_hidden": cfg.D_HIDDEN,
            "d_latent": cfg.D_LATENT,
            "n_enc_layers": cfg.N_ENC_LAYERS,
            "n_dec_layers": cfg.N_DEC_LAYERS,
            "beta_commit": cfg.BETA_COMMIT,
            "ema_decay": cfg.EMA_DECAY,
            "freeze_codebook_after": cfg.FREEZE_CODEBOOK_AFTER,
            "lambda_geom": cfg.LAMBDA_GEOM,
            "lambda_pair": cfg.LAMBDA_PAIR,
            "lambda_stack": cfg.LAMBDA_STACK,
            "lambda_mask": cfg.LAMBDA_MASK,
            "pair_stack_reduce": cfg.PAIR_STACK_REDUCE,
            "lambda_latent_norm": cfg.LAMBDA_LATENT_NORM,
            "split_name": cfg.SPLIT_NAME,
            "norm_mean": stats["mean"],
            "norm_std": stats["std"],
            "l2_normalize_encoder": True,
            "l2_normalize_codebook": True,
        },
        save_path,
    )
    print("  Saved checkpoint: %s" % save_path)


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Linear warmup + cosine decay."""

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    cfg = parse_cli_into(Config())
    print("Device: %s" % cfg.DEVICE)
    print("  split:      %s" % cfg.SPLIT_NAME)
    print("  split_dir:  %s" % cfg.SPLIT_DIR)
    print("  train_desc: %s" % cfg.TRAIN_DESC_PATH)
    print("  val_desc:   %s" % cfg.VAL_DESC_PATH)
    print("  test_desc:  %s" % cfg.TEST_DESC_PATH)
    print("  stats:      %s" % cfg.STATS_PATH)
    print("  save_dir:   %s" % cfg.MODEL_SAVE_PATH)
    print(
        "  batch_size: %d, epochs: %d, n_codes: %d, lr: %g"
        % (cfg.BATCH_SIZE, cfg.NUM_EPOCHS, cfg.N_CODES, cfg.LR)
    )
    print("  layers: encoder=%d, decoder=%d" % (cfg.N_ENC_LAYERS, cfg.N_DEC_LAYERS))
    print(
        "  beta_commit: %g, ema_decay: %g, freeze_codebook_after: %d"
        % (cfg.BETA_COMMIT, cfg.EMA_DECAY, cfg.FREEZE_CODEBOOK_AFTER)
    )
    print(
        "  loss: geom=%g pair=%g stack=%g mask=%g pair_stack_reduce=%s latent_norm=%g"
        % (
            cfg.LAMBDA_GEOM,
            cfg.LAMBDA_PAIR,
            cfg.LAMBDA_STACK,
            cfg.LAMBDA_MASK,
            cfg.PAIR_STACK_REDUCE,
            cfg.LAMBDA_LATENT_NORM,
        )
    )
    print("Codebook size: %d" % cfg.N_CODES)
    print("Descriptor dim: %d before loading descriptors" % cfg.D_DESCRIPTOR)

    # ── Load normalization stats ──
    stats = torch.load(cfg.STATS_PATH, map_location=cfg.DEVICE)
    mean = stats["mean"].to(cfg.DEVICE)
    std = stats["std"].to(cfg.DEVICE)
    print("Loaded normalization stats from %s" % cfg.STATS_PATH)
    print(
        "  d_descriptor=%s conditional_stats=%s"
        % (
            stats.get("d_descriptor", "?"),
            stats.get("conditional_stats", False),
        )
    )

    # ── Load pre-computed descriptors ──
    train_desc = torch.load(cfg.TRAIN_DESC_PATH, map_location="cpu")
    val_desc_path = cfg.VAL_DESC_PATH
    has_val = os.path.isfile(val_desc_path)
    if has_val:
        val_desc = torch.load(val_desc_path, map_location="cpu")
    else:
        val_desc = None
        print("  [WARN] No val_desc.pt found; falling back to test set for validation.")
    test_desc = torch.load(cfg.TEST_DESC_PATH, map_location="cpu")
    cfg.D_STRUCTURAL = int(train_desc.shape[1])
    cfg.D_DESCRIPTOR = cfg.D_STRUCTURAL
    print("Train: %d residues" % train_desc.shape[0])
    if has_val:
        print("Val:   %d residues" % val_desc.shape[0])
    print("Test:  %d residues" % test_desc.shape[0])
    print("Descriptor dim: %d" % cfg.D_DESCRIPTOR)

    train_ds = TensorDataset(train_desc)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=False)
    if has_val:
        val_ds = TensorDataset(val_desc)
        val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, drop_last=False)
    else:
        val_loader = None
    test_ds = TensorDataset(test_desc)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, drop_last=False)
    print("Batch size: %d residues" % cfg.BATCH_SIZE)
    print("Train batches/epoch: %d" % len(train_loader))
    os.makedirs(cfg.MODEL_SAVE_PATH, exist_ok=True)

    # ── Model ──
    # encoder and decoder both structural descriptor only (no nucleotide one-hot)
    model = DescriptorVQVAE(
        d_descriptor=cfg.D_STRUCTURAL,
        d_hidden=cfg.D_HIDDEN,
        d_latent=cfg.D_LATENT,
        n_codes=cfg.N_CODES,
        n_enc_layers=cfg.N_ENC_LAYERS,
        n_dec_layers=cfg.N_DEC_LAYERS,
        ema_decay=cfg.EMA_DECAY,
    ).to(cfg.DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model parameters: %s" % format(n_params, ","))

    # ── Optimizer + scheduler ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    total_steps = len(train_loader) * cfg.NUM_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.WARMUP_STEPS, total_steps)

    # ── Training log ──
    log_path = os.path.join(cfg.MODEL_SAVE_PATH, "training_log.csv")
    log_fields = [
        "epoch",
        "train_loss",
        "train_L_recon",
        "train_L_geom",
        "train_L_pair",
        "train_L_stack",
        "train_L_mask",
        "train_L_commit",
        "train_L_latent",
        "val_loss",
        "val_L_recon",
        "val_L_geom",
        "val_L_pair",
        "val_L_stack",
        "val_L_mask",
        "val_L_commit",
        "val_L_latent",
        "val_codebook_usage",
        "val_codebook_usage_pct",
        "lr",
    ]
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()

    # ── Training loop ──
    # Validation set is used for best-checkpoint selection.
    # Test set is evaluated only once after training completes.
    print("\nStarting training for %d epochs...\n" % cfg.NUM_EPOCHS)

    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        if cfg.FREEZE_CODEBOOK_AFTER > 0 and epoch == cfg.FREEZE_CODEBOOK_AFTER + 1:
            model.codebook.freeze_codebook = True
            print("  Freezing codebook EMA updates after epoch %d" % cfg.FREEZE_CODEBOOK_AFTER)

        train_losses = train_epoch(model, train_loader, optimizer, scheduler, cfg, mean, std)

        # ── Validation (for model selection) ──
        if val_loader is not None:
            val_losses = evaluate(model, val_loader, cfg, mean, std)
            val_tag = "val"
        else:
            # Fallback: use test set if no val set is available (not recommended)
            val_losses = evaluate(model, test_loader, cfg, mean, std)
            val_tag = "test"

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            "Epoch %3d/%d  "
            "Train: %.4f (geom=%.3f pair=%.3f stack=%.3f mask=%.3f commit=%.4f)  "
            "%s: %.4f  "
            "Codebook: %d/%d (%.1f%%)"
            % (
                epoch,
                cfg.NUM_EPOCHS,
                train_losses["loss"],
                train_losses["L_geom"],
                train_losses["L_pair"],
                train_losses["L_stack"],
                train_losses["L_mask"],
                train_losses["L_commit"],
                val_tag,
                val_losses["loss"],
                val_losses["codebook_usage"],
                cfg.N_CODES,
                val_losses["codebook_usage_pct"],
            )
        )

        row = {
            "epoch": epoch,
            "train_loss": train_losses["loss"],
            "train_L_recon": train_losses["L_recon"],
            "train_L_geom": train_losses["L_geom"],
            "train_L_pair": train_losses["L_pair"],
            "train_L_stack": train_losses["L_stack"],
            "train_L_mask": train_losses["L_mask"],
            "train_L_commit": train_losses["L_commit"],
            "train_L_latent": train_losses["L_latent"],
            "val_loss": val_losses["loss"],
            "val_L_recon": val_losses["L_recon"],
            "val_L_geom": val_losses["L_geom"],
            "val_L_pair": val_losses["L_pair"],
            "val_L_stack": val_losses["L_stack"],
            "val_L_mask": val_losses["L_mask"],
            "val_L_commit": val_losses["L_commit"],
            "val_L_latent": val_losses["L_latent"],
            "val_codebook_usage": val_losses["codebook_usage"],
            "val_codebook_usage_pct": val_losses["codebook_usage_pct"],
            "lr": current_lr,
        }
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writerow(row)

        # ── Best-by-val checkpoint ──
        if val_losses["loss"] < best_val_loss:
            best_val_loss = val_losses["loss"]
            best_epoch = epoch
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                val_losses["loss"],
                cfg,
                stats,
                filename="desc_vqvae_best.pt",
            )
            print(
                "  ✓ new best (%s=%.4f) saved to desc_vqvae_best.pt" % (val_tag, val_losses["loss"])
            )

        if epoch % cfg.SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, train_losses["loss"], cfg, stats)

    save_checkpoint(model, optimizer, scheduler, cfg.NUM_EPOCHS, train_losses["loss"], cfg, stats)

    # ── Final test evaluation (only once, after training completes) ──
    print("\n" + "=" * 64)
    print("Final test set evaluation (best checkpoint from val epoch %d)" % best_epoch)
    print("=" * 64)

    # Load best checkpoint for final test evaluation
    best_ckpt_path = os.path.join(cfg.MODEL_SAVE_PATH, "desc_vqvae_best.pt")
    if os.path.isfile(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location=cfg.DEVICE)
        model.load_state_dict(best_ckpt["model"])
        print("Loaded best checkpoint from epoch %d" % best_ckpt.get("epoch", best_epoch))

    test_losses = evaluate(model, test_loader, cfg, mean, std)
    print("  Test loss: %.4f" % test_losses["loss"])
    print(
        "  Test L_recon: %.4f  L_geom: %.4f  L_pair: %.4f  L_stack: %.4f  L_mask: %.4f"
        % (
            test_losses["L_recon"],
            test_losses["L_geom"],
            test_losses["L_pair"],
            test_losses["L_stack"],
            test_losses["L_mask"],
        )
    )
    print("  Test L_commit: %.4f" % test_losses["L_commit"])
    print(
        "  Codebook usage: %d/%d (%.1f%%)"
        % (test_losses["codebook_usage"], cfg.N_CODES, test_losses["codebook_usage_pct"])
    )

    # Save test results
    test_result_path = os.path.join(cfg.MODEL_SAVE_PATH, "test_results.pt")
    torch.save(
        {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "test_losses": {
                k: float(v) if torch.is_tensor(v) else v for k, v in test_losses.items()
            },
        },
        test_result_path,
    )

    print("\nTraining complete! Log: %s" % log_path)
    print("Best val loss: %.4f at epoch %d (desc_vqvae_best.pt)" % (best_val_loss, best_epoch))
    print("Final test loss: %.4f" % test_losses["loss"])


if __name__ == "__main__":
    main()
