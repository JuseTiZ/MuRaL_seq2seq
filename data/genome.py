import numpy as np
import pyfaidx

BASES_ARR = ['A', 'C', 'G', 'T']
UNK_BASE = 'N'
BASE_TO_INDEX = {b: i for i, b in enumerate(BASES_ARR)}


def sequence_to_encoding(seq_str: str) -> np.ndarray:
    """Convert a DNA string to one-hot encoding (L, 4)."""
    encoding = np.zeros((len(seq_str), 4), dtype=np.float32)
    for i, base in enumerate(BASES_ARR):
        encoding[:, i] = np.array([1.0 if c == base else 0.0 for c in seq_str], dtype=np.float32)
    # Unknown bases (N) get 0.25 for each channel
    unk_mask = np.array([c == UNK_BASE or c not in 'ACGT' for c in seq_str])
    if unk_mask.any():
        encoding[unk_mask] = 0.25
    return encoding


class Genome:
    """Reference genome loader wrapping pyfaidx.Fasta."""

    def __init__(self, fasta_path: str):
        self.fasta_path = fasta_path
        self._fasta = None
        self.chroms = {}
        self.len_chroms = {}

    def _init(self):
        if self._fasta is None:
            self._fasta = pyfaidx.Fasta(self.fasta_path)
            self.chroms = {c: c for c in self._fasta.keys()}
            self.len_chroms = {c: len(self._fasta[c]) for c in self._fasta.keys()}

    def get_encoding_from_coords(self, chrom, start, end, pad=True):
        """
        Get one-hot encoded sequence for a genomic interval.

        Args:
            chrom: chromosome name
            start: 0-based start
            end: 0-based end (exclusive)
            pad: whether to handle out-of-bounds with N-padding

        Returns:
            numpy array of shape (L, 4)
        """
        self._init()
        length = end - start

        if not pad:
            if chrom not in self.len_chroms or start < 0 or end > self.len_chroms[chrom]:
                return np.array([])
            seq_str = self._fasta[chrom][start:end].seq.upper()
        else:
            chr_len = self.len_chroms.get(chrom, 0)
            eff_start = max(start, 0)
            eff_end = min(end, chr_len)
            if eff_start < eff_end and chrom in self.len_chroms:
                seq_str = self._fasta[chrom][eff_start:eff_end].seq.upper()
            else:
                seq_str = ''

            left_pad = eff_start - start
            right_pad = end - eff_end
            seq_str = 'N' * left_pad + seq_str + 'N' * right_pad

        return sequence_to_encoding(seq_str)
