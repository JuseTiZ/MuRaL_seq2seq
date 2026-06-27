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
import gzip
import math
import os
import pickle
import sys
import time
from collections import namedtuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Allow running this script directly from anywhere without installing mural_s2s
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mural_s2s.config import TrainingConfig
from mural_s2s.data.genome import Genome
from mural_s2s.data.targets import GenomicSignalFeatures
from mural_s2s.data.sampler import IntervalsSampler
from mural_s2s.data.dataloader import build_dataloader
from mural_s2s.model import PuffinD


CropMetadata = namedtuple(
    "CropMetadata", ["chrom", "output_start", "output_end", "crop_start"]
)


class CenterCropPredictionDataset(Dataset):
    """Build fixed-length inference windows around smaller output tiles."""

    def __init__(self, sampler, mode, sequence_length, output_length):
        self.genome = sampler.reference_sequence
        self.sequence_length = sequence_length
        self.windows = []

        for interval_idx in sampler.get_mode_indices(mode):
            chrom, interval_start, interval_end = sampler.sample_from_intervals[
                interval_idx
            ]
            interval_length = interval_end - interval_start
            if interval_length != sequence_length:
                raise ValueError(
                    f"BED interval {chrom}:{interval_start}-{interval_end} has length "
                    f"{interval_length}, expected --sequence-length {sequence_length}"
                )

            for output_start in range(interval_start, interval_end, output_length):
                output_end = min(output_start + output_length, interval_end)
                tile_length = output_end - output_start
                total_flank = sequence_length - tile_length
                left_flank = total_flank // 2
                right_flank = total_flank - left_flank
                input_start = output_start - left_flank
                input_end = output_end + right_flank
                self.windows.append(
                    (
                        chrom,
                        input_start,
                        input_end,
                        CropMetadata(chrom, output_start, output_end, left_flank),
                    )
                )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        chrom, input_start, input_end, meta = self.windows[idx]
        seq = self.genome.get_encoding_from_coords(
            chrom, input_start, input_end, pad=True
        )
        if seq.shape[0] != self.sequence_length:
            raise RuntimeError(
                f"Failed to retrieve inference window "
                f"{chrom}:{input_start}-{input_end}"
            )
        return torch.from_numpy(seq).float().permute(1, 0), meta


def _collate_center_crop(batch):
    return torch.stack([item[0] for item in batch]), [item[1] for item in batch]


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
    p.add_argument(
        "--center-output-length",
        type=int,
        default=None,
        help=(
            "Only write the center N bases from each inference window and tile "
            "them across every BED interval (default: disabled)"
        ),
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=436)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--disable-reverse", action="store_false", dest="use_reverse",
                   default=True, help="Disable reverse-complement module in model")
    p.add_argument("--progress-every", type=int, default=100,
                   help="Print progress every N batches")

    p.add_argument("--output", required=True, help="Output TSV path (.gz supported)")
    p.add_argument("--mode", default="test", choices=["train", "validate", "test"],
                   help="Which data split to predict on")
    p.add_argument("--device", default="cuda:0")

    return p.parse_args()


def main():
    args = parse_args()

    if args.center_output_length is not None:
        if not 0 < args.center_output_length <= args.sequence_length:
            raise ValueError(
                "--center-output-length must be greater than zero and no larger "
                "than --sequence-length"
            )
        if (args.sequence_length - args.center_output_length) % 2 != 0:
            raise ValueError(
                "--sequence-length minus --center-output-length must be even so "
                "the output tile has symmetric flanking context"
            )

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

    n_intervals = len(sampler.get_mode_indices(args.mode))
    n_expected_positions = sum(
        sampler.sample_from_intervals[idx][2]
        - sampler.sample_from_intervals[idx][1]
        for idx in sampler.get_mode_indices(args.mode)
    )

    if args.center_output_length is None:
        loader = build_dataloader(
            sampler,
            config.batch_size,
            mode=args.mode,
            num_workers=args.num_workers,
            seed=config.seed,
            shuffle=False,
        )
        n_inference_windows = n_intervals
    else:
        dataset = CenterCropPredictionDataset(
            sampler,
            args.mode,
            config.sequence_length,
            args.center_output_length,
        )
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=_collate_center_crop,
        )
        n_inference_windows = len(dataset)

    n_batches = math.ceil(n_inference_windows / config.batch_size)

    # --- Model ---
    # Try to auto-detect use_reverse from checkpoint config
    config_path = args.model + ".config.pkl"
    if os.path.exists(config_path):
        with open(config_path, 'rb') as f:
            saved_config = pickle.load(f)
        use_reverse = getattr(saved_config, 'use_reverse', True)
    else:
        use_reverse = args.use_reverse

    n_output_channels = len(config.target_features)
    model = PuffinD(n_output_channels=n_output_channels, use_reverse=use_reverse)
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
    print(f"  use_reverse:{use_reverse}")
    print(f"  Device:     {device}")
    print(f"  FASTA:      {config.fasta}")
    print(f"  Intervals:  {config.intervals}")
    print(f"  Mode:        {args.mode}")
    print(f"  Seq length:  {config.sequence_length}")
    if args.center_output_length is None:
        print("  Inference:   full-window output")
    else:
        flank = (config.sequence_length - args.center_output_length) // 2
        print("  Inference:   center-cropped tiled output")
        print(f"  Center size: {args.center_output_length}")
        print(f"  Min flank:   {flank} per side")
    print(f"  Batch size:  {config.batch_size}")
    print(f"  Intervals:   {n_intervals:,}")
    print(f"  Windows:     {n_inference_windows:,}")
    print(f"  Batches:     {n_batches:,}")
    print(f"  Positions:   {n_expected_positions:,}")
    print(f"  Output:      {args.output}")
    print("=" * 70)

    # --- Predict ---
    progress_n = args.progress_every

    bases = ['A', 'C', 'G', 'T']
    col_names = ['chrom', 'start', 'end'] + [f'mut_rate_{b}' for b in bases]
    header_line = '\t'.join(col_names) + '\n'

    t0 = time.time()
    n_positions = 0

    open_func = gzip.open if use_gzip else open
    with open_func(args.output, 'wt') as f:
        f.write(header_line)

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if args.center_output_length is None:
                    sequence, _, metadatas = batch
                else:
                    sequence, metadatas = batch
                sequence = sequence.to(device)
                preds = model(sequence)  # (B, 4, L)

                # Hard-mask self-mutation channels: A->A, C->C, G->G, T->T are not mutations
                preds = preds * (1 - sequence)

                preds = preds.cpu().numpy()

                lines = []
                for i, meta in enumerate(metadatas):
                    if args.center_output_length is None:
                        chrom = meta.chroms
                        output_start = int(meta.bin_starts)
                        output_end = output_start + preds.shape[2]
                        crop_start = 0
                    else:
                        chrom = meta.chrom
                        output_start = int(meta.output_start)
                        output_end = int(meta.output_end)
                        crop_start = int(meta.crop_start)

                    for pos in range(output_start, output_end):
                        p = crop_start + pos - output_start
                        vals = '\t'.join(f'{preds[i, j, p]:.8f}' for j in range(4))
                        lines.append(f'{chrom}\t{pos}\t{pos+1}\t{vals}\n')

                f.write(''.join(lines))
                n_positions += len(lines)

                b = batch_idx + 1
                if b % progress_n == 0:
                    elapsed = time.time() - t0
                    pct = 100.0 * b / n_batches
                    print(f"  [predict] {b:>5d}/{n_batches} ({pct:4.0f}%) | "
                          f"{_format_time(elapsed)}")

    total_time = time.time() - t0
    print(f"\nPrediction finished in {_format_time(total_time)}")
    print(f"Positions predicted: {n_positions:,}")
    print(f"Output saved to {args.output}")


if __name__ == "__main__":
    main()
