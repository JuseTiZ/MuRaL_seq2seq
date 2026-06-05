#!/usr/bin/env python
"""
Prediction entry point for seq2seq mutation rate prediction.

Usage:
    python mural_s2s/predict.py \
        --fasta /path/to/hg19.fa \
        --intervals /path/to/segments.bed \
        --target-dir /path/to/bigwig_dir \
        --mask-bw /path/to/mask.bw \
        --model /path/to/model_checkpoint \
        --output predictions.tsv.gz \
        --mode test
"""

import argparse
import os
import sys

# Allow running this script directly from anywhere without installing mural_s2s
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys

import numpy as np
import pandas as pd
import torch

from mural_s2s.config import TrainingConfig
from mural_s2s.data.genome import Genome
from mural_s2s.data.targets import GenomicSignalFeatures
from mural_s2s.data.sampler import IntervalsSampler
from mural_s2s.data.dataloader import build_dataloader
from mural_s2s.model import PuffinD


def parse_args():
    p = argparse.ArgumentParser(description="Predict mutation rates with seq2seq model")

    # Data
    p.add_argument("--fasta", required=True, help="Reference FASTA")
    p.add_argument("--intervals", required=True, help="BED intervals file")
    p.add_argument("--target-dir", required=True, help="Directory with BigWig files")
    p.add_argument("--mask-bw", required=True, help="Coverage mask BigWig")

    # Model
    p.add_argument("--model", required=True, help="Model checkpoint path")

    # Config (must match training)
    p.add_argument("--val-chroms", nargs="+", default=["chr1"])
    p.add_argument("--test-chroms", nargs="+", default=["chr2"])
    p.add_argument("--sequence-length", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=436)

    # Output
    p.add_argument("--output", required=True, help="Output TSV path (.gz supported)")
    p.add_argument("--mode", default="test", choices=["train", "validate", "test"],
                   help="Which data split to predict on")

    # GPU
    p.add_argument("--device", default="cuda:0")

    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build config with minimal settings
    config = TrainingConfig(
        fasta=args.fasta,
        intervals=args.intervals,
        target_dir=args.target_dir,
        mask_bw=args.mask_bw,
        val_chroms=args.val_chroms,
        test_chroms=args.test_chroms,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # --- Data ---
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

    # --- Model ---
    n_output_channels = len(config.target_features)
    model = PuffinD(n_output_channels=n_output_channels)
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Model loaded from {args.model}")

    # --- Predict ---
    sampler.mode = args.mode
    loader = build_dataloader(sampler, config.batch_size, num_workers=0, seed=config.seed)

    bases = ['A', 'C', 'G', 'T']
    all_rows = []

    with torch.no_grad():
        for sequence, target, metadatas in loader:
            sequence = sequence.to(device)
            preds = model(sequence).cpu().numpy()  # (B, 4, L)

            for i, meta in enumerate(metadatas):
                chrom = meta.chroms
                start = int(meta.bin_starts)
                positions = np.arange(start, start + preds.shape[2])

                for j, base in enumerate(bases):
                    rows = pd.DataFrame({
                        'chrom': chrom,
                        'pos': positions,
                        'base': base,
                        'mut_rate': preds[i, j, :],
                    })
                    all_rows.append(rows)

    df = pd.concat(all_rows, axis=0, ignore_index=True)

    # Pivot: rows = positions, columns = mut_rate per base
    df_wide = df.pivot_table(
        index=['chrom', 'pos'], columns='base', values='mut_rate'
    ).reset_index()
    df_wide.columns.name = None
    df_wide.rename(columns={
        'A': 'mut_rate_A', 'C': 'mut_rate_C',
        'G': 'mut_rate_G', 'T': 'mut_rate_T',
    }, inplace=True)

    df_wide.to_csv(args.output, index=False, sep='\t',
                   compression='gzip' if args.output.endswith('.gz') else None)
    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
