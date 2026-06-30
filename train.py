#!/usr/bin/env python
"""
Training entry point for seq2seq mutation rate prediction.

Usage:
    python mural_s2s/train.py \
        --fasta /path/to/hg19.fa \
        --intervals /path/to/segments.bed \
        --target-dir /path/to/bigwig_dir \
        --mask-bw /path/to/mask.bw \
        --output-dir ./output
"""

import argparse
import math
import os
import random
import sys
import time

import numpy as np
import torch

# Allow running this script directly from anywhere without installing mural_s2s
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mural_s2s.config import TrainingConfig
from mural_s2s.data.genome import Genome
from mural_s2s.data.targets import GenomicSignalFeatures
from mural_s2s.data.sampler import IntervalsSampler
from mural_s2s.data.dataloader import build_dataloader
from mural_s2s.model import PuffinD, count_parameters
from mural_s2s.training.trainer import Trainer, _mask_no_mut
from mural_s2s.training.observer import LossMinor, GradMinor
from mural_s2s.evaluation.metrics import calc_regional_correlation_grouped
from mural_s2s.utils.helpers import EarlyStopping, save_model


def parse_args():
    p = argparse.ArgumentParser(description="Train seq2seq mutation rate predictor")

    p.add_argument("--fasta", required=True, help="Reference FASTA")
    p.add_argument("--intervals", required=True, help="BED intervals file")
    p.add_argument("--target-dir", required=True, help="Directory with BigWig files")
    p.add_argument("--mask-bw", required=True, help="Coverage mask BigWig")
    p.add_argument("--output-dir", default="./output", help="Output directory")

    p.add_argument("--val-chroms", nargs="+", default=["chr1"])
    p.add_argument("--test-chroms", nargs="+", default=["chr2"])
    p.add_argument("--sequence-length", type=int, default=10000)

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--lr-gamma", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--total-weight", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=436)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--reverse-complement-aug", action="store_true", default=False)
    p.add_argument("--disable-reverse", action="store_false", dest="use_reverse",
                   default=True, help="Disable reverse-complement module in model")
    p.add_argument("--progress-every", type=int, default=100,
                   help="Print progress every N batches")

    p.add_argument("--device", type=str, default="cuda:0")

    return p.parse_args()


def _format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{s:.0f}s"


def _print_config_summary(config, device, sampler):
    """Print a comprehensive configuration summary."""
    target_features = config.target_features + ["mask"]
    target_bw_paths = config.target_bw_paths + [config.mask_bw]

    train_n = len(sampler.get_mode_indices("train"))
    val_n = len(sampler.get_mode_indices("validate"))
    test_n = len(sampler.get_mode_indices("test")) \
        if "test" in sampler.modes else 0

    train_batches = math.ceil(train_n / config.batch_size)
    val_batches = math.ceil(val_n / config.batch_size)
    test_batches = math.ceil(test_n / config.batch_size)

    n_out = len(config.target_features)
    gamma_step = config.lr_gamma ** (config.batch_size / max(train_n, 1))

    lines = []
    L = lines.append
    L("=" * 70)
    L("Configuration Summary")
    L("=" * 70)
    L(f"  Seed: {config.seed}")
    L(f"  Device: {device}")
    L("")
    L("  Data:")
    L(f"    FASTA:          {config.fasta}")
    L(f"    Intervals:      {config.intervals}")
    L(f"    Seq length:     {config.sequence_length}")
    L(f"    Val chroms:     {', '.join(config.val_chroms)}")
    L(f"    Test chroms:    {', '.join(config.test_chroms)}")
    L("    Target BigWigs:")
    for feat, path in zip(target_features, target_bw_paths):
        L(f"      {feat:16s} → {path}")
    L("")
    L("  Data split:")
    L(f"    Train:    {train_n:>8,} intervals → {train_batches:>6,} batches/epoch")
    L(f"    Validate: {val_n:>8,} intervals → {val_batches:>6,} batches/epoch")
    if test_n > 0:
        L(f"    Test:     {test_n:>8,} intervals → {test_batches:>6,} batches/epoch")
    L("")
    L("  Model:")
    L(f"    Architecture:   PuffinD")
    L(f"    Output channels: {n_out} ({', '.join(config.target_features)})")
    L(f"    Use reverse module: {config.use_reverse}")
    L("")
    L("  Training:")
    L(f"    Epochs:        {config.epochs}")
    L(f"    Batch size:    {config.batch_size}")
    L(f"    Optimizer:     Adam (lr={config.learning_rate}, weight_decay={config.weight_decay})")
    L(f"    LR schedule:   StepLR per-batch (gamma={gamma_step:.6f}, min_lr={config.min_lr})")
    L(f"    Loss:          Poisson_PseudoKL (total_weight={config.total_weight})")
    L(f"    Grad clip:     {config.gradient_clip_norm}")
    L(f"    Rev-comp aug:  {config.reverse_complement_aug}")
    L(f"    Patience:      {config.patience}")
    L(f"    Num workers:   {config.num_workers}")
    L(f"    Output dir:    {config.output_dir}")
    L("=" * 70)

    print("\n".join(lines))
    return train_n, val_n, test_n, train_batches, val_batches, test_batches, gamma_step


def _write_log_header(log_path, config, device, train_n, val_n):
    """Write config as commented header lines in the training log."""
    with open(log_path, 'w') as f:
        f.write(f"# Training log — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# fasta={config.fasta}\n")
        f.write(f"# intervals={config.intervals}\n")
        f.write(f"# val_chroms={','.join(config.val_chroms)}\n")
        f.write(f"# test_chroms={','.join(config.test_chroms)}\n")
        f.write(f"# seq_length={config.sequence_length}\n")
        f.write(f"# batch_size={config.batch_size} epochs={config.epochs}\n")
        f.write(f"# lr={config.learning_rate} min_lr={config.min_lr} lr_gamma={config.lr_gamma}\n")
        f.write(f"# weight_decay={config.weight_decay} grad_clip={config.gradient_clip_norm}\n")
        f.write(f"# total_weight={config.total_weight} seed={config.seed}\n")
        f.write(f"# use_reverse={config.use_reverse}\n")
        f.write(f"# rev_complement_aug={config.reverse_complement_aug}\n")
        f.write(f"# device={device}\n")
        cols = ["epoch", "train_loss", "val_loss", "lr", "duration_s"]
        for nuc in ['A', 'C', 'G', 'T']:
            for ch in range(len(config.target_features)):
                cols.append(f"corr_{nuc}_{config.target_features[ch]}")
        f.write("# " + "\t".join(cols) + "\n")


def main():
    args = parse_args()

    config = TrainingConfig(
        fasta=args.fasta,
        intervals=args.intervals,
        target_dir=args.target_dir,
        mask_bw=args.mask_bw,
        val_chroms=args.val_chroms,
        test_chroms=args.test_chroms,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        lr_gamma=args.lr_gamma,
        gradient_clip_norm=args.grad_clip,
        total_weight=args.total_weight,
        patience=args.patience,
        seed=args.seed,
        num_workers=args.num_workers,
        reverse_complement_aug=args.reverse_complement_aug,
        use_reverse=args.use_reverse,
        output_dir=args.output_dir,
        progress_every_n_batches=args.progress_every,
    )

    os.makedirs(config.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    np.random.seed(config.seed)
    random.seed(config.seed + 1)
    torch.manual_seed(config.seed)

    # --- Data pipeline ---
    genome = Genome(config.fasta)

    target_features = config.target_features + ["mask"]
    target_bw_paths = config.target_bw_paths + [config.mask_bw]
    tfeature = GenomicSignalFeatures(target_bw_paths, target_features)

    sampler = IntervalsSampler(
        reference_sequence=genome,
        target=tfeature,
        intervals_path=config.intervals,
        sequence_length=config.sequence_length,
        validation_holdout=config.val_chroms,
        test_holdout=config.test_chroms,
        seed=config.seed,
    )

    config.train_size = len(sampler.get_mode_indices("train"))

    # --- Print config & init log ---
    train_n, val_n, test_n, train_batches, val_batches, test_batches, gamma_step = \
        _print_config_summary(config, device, sampler)

    log_path = os.path.join(config.output_dir, "training.log")
    _write_log_header(log_path, config, device, train_n, val_n)

    # --- Model ---
    n_output_channels = len(config.target_features)
    model = PuffinD(n_output_channels=n_output_channels, use_reverse=args.use_reverse).to(device)
    print(f"\nModel parameters: {count_parameters(model):,}")

    # --- Trainer ---
    loss_observer = LossMinor(out_after_n_batch=500)
    observers = [loss_observer, GradMinor(out_after_n_batch=500)]
    trainer = Trainer(model, config, device, observers=observers)

    early_stopping = EarlyStopping(patience=config.patience)
    best_val_loss = float('inf')
    best_epoch = -1
    progress_n = config.progress_every_n_batches

    # --- Training loop ---
    for epoch in range(1, config.epochs + 1):
        current_lr = trainer.optimizer.param_groups[0]['lr']

        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{config.epochs} | "
              f"Train batches: {train_batches:,} | "
              f"Val batches: {val_batches:,} | "
              f"LR: {current_lr:.2e}")
        print(f"{'='*70}")

        # --- Train ---
        train_loader = build_dataloader(
            sampler, config.batch_size, mode="train",
            num_workers=config.num_workers, seed=config.seed,
            reverse_complement_aug=config.reverse_complement_aug,
            shuffle=True,
        )
        model.train()
        t0 = time.time()

        for batch_idx, (sequence, target, _) in enumerate(train_loader):
            trainer.train_step(sequence, target)
            b = batch_idx + 1
            if b % progress_n == 0:
                elapsed = time.time() - t0
                pct = 100.0 * b / train_batches
                print(f"  [train] {b:>5d}/{train_batches} ({pct:4.0f}%) | "
                      f"{_format_time(elapsed)} | "
                      f"loss {loss_observer.current_loss:.6f} | "
                      f"LR {trainer.optimizer.param_groups[0]['lr']:.2e}")

        train_metrics = trainer.epoch_finish("train")
        train_time = time.time() - t0
        train_loss = train_metrics.get('train_mean_loss', 0.0)
        print(f"  [train] done in {_format_time(train_time)} — "
              f"mean loss: {train_loss:.6f}")

        # --- Save checkpoint ---
        checkpoint_dir = os.path.join(config.output_dir, f"checkpoint_{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_path = os.path.join(checkpoint_dir, "model")
        save_model(model, config, save_path)
        print(f"Checkpoint saved to {save_path}")

        # --- Validate ---
        val_loader = build_dataloader(
            sampler, config.batch_size, mode="validate",
            num_workers=config.num_workers, seed=config.seed,
            shuffle=False,
        )
        model.eval()

        valid_preds = []
        valid_targets = []
        valid_seq_indices = []
        valid_bg_rates = []
        valid_n_valid = []
        t0_val = time.time()

        for batch_idx, (sequence, target, _) in enumerate(val_loader):
            pred, bg_rates, n_valid = trainer.valid_step(sequence, target)
            mask = target[:, -1, :]
            target_values = _mask_no_mut(sequence, target[:, :-1, :])

            pred_masked = pred.cpu() * mask.unsqueeze(1)
            target_masked = target_values.cpu() * mask.unsqueeze(1)

            valid_preds.append(pred_masked.numpy())
            valid_targets.append(target_masked.numpy())
            valid_seq_indices.append(sequence.argmax(dim=1).cpu().numpy())
            valid_bg_rates.append(bg_rates.cpu().numpy())
            valid_n_valid.append(n_valid.cpu().numpy())

            b = batch_idx + 1
            if b % progress_n == 0:
                elapsed = time.time() - t0_val
                pct = 100.0 * b / val_batches
                print(f"  [valid] {b:>5d}/{val_batches} ({pct:4.0f}%) | "
                      f"{_format_time(elapsed)}")

        val_metrics = trainer.epoch_finish("validate")
        val_time = time.time() - t0_val
        val_loss = val_metrics.get('validate_mean_loss', 0.0)

        # --- Metrics ---
        pred_all = np.concatenate(valid_preds, axis=0)
        true_all = np.concatenate(valid_targets, axis=0)
        seq_all = np.concatenate(valid_seq_indices, axis=0)
        bg_all = np.concatenate(valid_bg_rates, axis=0)       # (N_val, 4)
        nv_all = np.concatenate(valid_n_valid, axis=0)         # (N_val, 4)

        corr_grouped = calc_regional_correlation_grouped(pred_all, true_all, seq_all)

        # Per-sample, per-channel mean prediction over valid positions
        pred_mean_all = pred_all.sum(axis=2) / np.maximum(nv_all, 1.0)

        # Fold change: pred_mean / bg_rate (where bg_rate > 0)
        fold_all = np.where(bg_all > 0, pred_mean_all / bg_all, np.nan)

        # --- Epoch summary ---
        base_labels = ['A', 'C', 'G', 'T']
        ch_names = config.target_features

        # --- Track best ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            improved = True
        else:
            improved = False

        print(f"\n--- Epoch {epoch} Summary ---")
        print(f"Train loss: {train_loss:.6f} | "
              f"Val loss: {val_loss:.8f} | "
              f"LR: {current_lr:.2e} | "
              f"Train time: {_format_time(train_time)} | "
              f"Val time: {_format_time(val_time)}")
        if improved:
            print(f"  *** Best model updated: epoch {best_epoch}, "
                  f"val_loss = {best_val_loss:.8f} ***")
        print("Per-nucleotide-grouped Pearson r:")
        header = "      " + "  ".join(f"{n:>8s}" for n in ch_names)
        print(header)
        for nuc_idx, nuc_label in enumerate(base_labels):
            vals = "  ".join(f"{corr_grouped[nuc_idx, ch]:8.4f}"
                             for ch in range(len(ch_names)))
            print(f"  {nuc_label}  {vals}")

        print("Background rate & fold change per channel:")
        print(f"  {'Channel':<12s} {'bg_mean':>10s} {'pred_mean':>10s} "
              f"{'fold p25':>8s} {'fold p50':>8s} {'fold p75':>8s}")
        for ch, name in enumerate(ch_names):
            bg_mean = bg_all[:, ch].mean()
            pred_mean = pred_mean_all[:, ch].mean()
            fc_ch = fold_all[:, ch]
            fc_valid = fc_ch[~np.isnan(fc_ch)]
            if len(fc_valid) > 0:
                p25, p50, p75 = np.percentile(fc_valid, [25, 50, 75])
                print(f"  {name:<12s} {bg_mean:10.2e} {pred_mean:10.2e} "
                      f"{p25:8.4f} {p50:8.4f} {p75:8.4f}")
            else:
                print(f"  {name:<12s} {bg_mean:10.2e} {pred_mean:10.2e} "
                      f"{'N/A':>8s} {'N/A':>8s} {'N/A':>8s}")

        # --- Write log ---
        log_cols = [str(epoch), f"{train_loss:.6f}", f"{val_loss:.8f}",
                    f"{current_lr:.6e}", f"{train_time:.1f}"]
        for nuc_idx in range(4):
            for ch in range(len(ch_names)):
                log_cols.append(f"{corr_grouped[nuc_idx, ch]:.6f}"
                                if not np.isnan(corr_grouped[nuc_idx, ch]) else "nan")
        with open(log_path, 'a') as lf:
            lf.write("\t".join(log_cols) + "\n")

        # --- Early stopping ---
        if early_stopping(val_loss, model):
            print(f"\nEarly stopping triggered (patience={config.patience}). "
                  f"Best val_loss={early_stopping.best_loss:.8f}")
            break

        print(f"Checkpoint saved to {save_path}")

    print(f"\nTraining finished.")
    print(f"Best model: checkpoint_{best_epoch}/ (val_loss = {best_val_loss:.8f})")
    print(f"Outputs saved to {config.output_dir}")


if __name__ == "__main__":
    main()
