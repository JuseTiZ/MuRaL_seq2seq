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
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

# Allow running this script directly from anywhere without installing mural_s2s
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mural_s2s.config import TrainingConfig
from mural_s2s.data.genome import Genome
from mural_s2s.data.targets import GenomicSignalFeatures
from mural_s2s.data.sampler import IntervalsSampler
from mural_s2s.data.dataloader import build_dataloader
from mural_s2s.model import PuffinD


def _format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{s:.0f}s"


def parse_args():
    p = argparse.ArgumentParser(description="Predict mutation rates with seq2seq model")

    p.add_argument("--fasta", required=True, help="Reference FASTA")
    p.add_argument("--intervals", required=True, help="BED intervals file")
    p.add_argument("--target-dir", required=True, help="Directory with BigWig files")
    p.add_argument("--mask-bw", required=True, help="Coverage mask BigWig")
    p.add_argument("--model", required=True, help="Model checkpoint path")

    p.add_argument("--val-chroms", nargs="+", default=["chr1"])
    p.add_argument("--test-chroms", nargs="+", default=["chr2"])
    p.add_argument("--sequence-length", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=436)
    p.add_argument("--progress-every", type=int, default=100,
                   help="Print progress every N batches")

    p.add_argument("--output", required=True, help="Output TSV path (.gz supported)")
    p.add_argument("--mode", default="test", choices=["train", "validate", "test"],
                   help="Which data split to predict on")
    p.add_argument("--device", default="cuda:0")

    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

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

    sampler.mode = args.mode
    n_intervals = sampler.n_samples
    n_batches = math.ceil(n_intervals / config.batch_size)

    # --- Model ---
    n_output_channels = len(config.target_features)
    model = PuffinD(n_output_channels=n_output_channels)
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # --- Config summary ---
    use_gzip = args.output.endswith('.gz')
    print("=" * 70)
    print("Prediction Configuration")
    print("=" * 70)
    print(f"  Model:      {args.model}")
    print(f"  Device:     {device}")
    print(f"  FASTA:      {config.fasta}")
    print(f"  Intervals:  {config.intervals}")
    print(f"  Mode:        {args.mode}")
    print(f"  Seq length:  {config.sequence_length}")
    print(f"  Batch size:  {config.batch_size}")
    print(f"  Intervals:   {n_intervals:,}")
    print(f"  Batches:     {n_batches:,}")
    print(f"  Output:      {args.output}")
    print("=" * 70)

    # --- Predict ---
    loader = build_dataloader(sampler, config.batch_size, num_workers=0, seed=config.seed)
    progress_n = args.progress_every

    bases = ['A', 'C', 'G', 'T']
    col_names = ['chrom', 'pos'] + [f'mut_rate_{b}' for b in bases]
    header_line = '\t'.join(col_names) + '\n'

    # Write uncompressed temp file, compress at end if needed
    tmp_output = args.output + ".tmp"

    t0 = time.time()
    n_positions = 0

    with open(tmp_output, 'w') as f:
        f.write(header_line)

        with torch.no_grad():
            for batch_idx, (sequence, target, metadatas) in enumerate(loader):
                sequence = sequence.to(device)
                preds = model(sequence).cpu().numpy()  # (B, 4, L)

                lines = []
                for i, meta in enumerate(metadatas):
                    chrom = meta.chroms
                    start = int(meta.bin_starts)
                    end = start + preds.shape[2]
                    for pos in range(start, end):
                        p = pos - start
                        vals = '\t'.join(f'{preds[i, j, p]:.8f}' for j in range(4))
                        lines.append(f'{chrom}\t{pos}\t{vals}\n')

                f.write(''.join(lines))
                n_positions += len(lines)

                b = batch_idx + 1
                if b % progress_n == 0:
                    elapsed = time.time() - t0
                    pct = 100.0 * b / n_batches
                    print(f"  [predict] {b:>5d}/{n_batches} ({pct:4.0f}%) | "
                          f"{_format_time(elapsed)}")

    # Compress if requested, then rename
    if use_gzip:
        os.system(f"gzip -f {tmp_output}")
        tmp_output = tmp_output + ".gz"

    os.rename(tmp_output, args.output)

    total_time = time.time() - t0
    print(f"\nPrediction finished in {_format_time(total_time)}")
    print(f"Positions predicted: {n_positions:,}")
    print(f"Output saved to {args.output}")


if __name__ == "__main__":
    main()
