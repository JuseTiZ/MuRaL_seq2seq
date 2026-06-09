import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class Seq2SeqDataset(Dataset):
    """
    Map-style PyTorch Dataset. Each item is one genomic interval.

    Uses idx to directly look up the interval from the sampler's interval list.
    Compatible with PyTorch DataLoader random/shuffle sampling.
    """

    def __init__(self, sampler, mode, reverse_complement_aug=False):
        self.sampler = sampler
        self.mode = mode
        self.indices = sampler.get_mode_indices(mode)
        self.reverse_complement_aug = reverse_complement_aug

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        interval_idx = self.indices[idx]
        chrom, start, end = self.sampler.sample_from_intervals[interval_idx]

        result = self.sampler.retrieve(chrom, start, end)
        if result is None:
            raise RuntimeError(
                f"Failed to retrieve interval {interval_idx}: "
                f"{chrom}:{start}-{end} in mode={self.mode}"
            )

        seq, target = result  # (L, 4), (C, L)

        if self.reverse_complement_aug and random.random() < 0.5:
            seq, target = _reverse_complement(seq, target)

        # Transpose seq from (L, 4) → (4, L) for Conv1d
        seq = torch.from_numpy(seq).float().permute(1, 0)
        target = torch.from_numpy(target).float()

        from mural_s2s.data.sampler import Metadata
        meta = Metadata(chrom, start, end)
        return seq, target, meta


def _reverse_complement(seq, target):
    """Reverse-complement a single (L, 4) sequence and its (C, L) target.

    Target channels: [mut_to_A, mut_to_C, mut_to_G, mut_to_T, mask].
    After reverse-complement, mut_to_A <-> mut_to_T and mut_to_C <-> mut_to_G.
    """
    seq_rc = seq[::-1, ::-1].copy()
    target_rc = target.copy()
    target_rc[:4] = target[:4][::-1, ::-1]  # swap A<->T (0<->3), C<->G (1<->2), and reverse pos
    target_rc[4] = target[4, ::-1]          # mask: position-reverse only
    return seq_rc, target_rc


def worker_init_fn(worker_id, seed=436):
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id + 1)


def build_dataloader(sampler, batch_size, mode, num_workers=1, seed=436,
                     reverse_complement_aug=False, shuffle=False):
    """Build a finite PyTorch DataLoader for a given data split mode."""
    dataset = Seq2SeqDataset(sampler, mode=mode,
                             reverse_complement_aug=reverse_complement_aug)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=lambda wid: worker_init_fn(wid, seed) if num_workers > 0 else None,
        collate_fn=_collate_batch,
    )
    return dataloader


def _collate_batch(batch):
    """Collate list of (seq, target, Metadata) into batched tensors."""
    sequences = torch.stack([item[0] for item in batch])  # (B, 4, L)
    targets = torch.stack([item[1] for item in batch])    # (B, C, L)
    metadatas = [item[2] for item in batch]
    return sequences, targets, metadatas
