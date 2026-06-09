import random
from collections import namedtuple

import numpy as np

SampleIndices = namedtuple("SampleIndices", ["indices"])
Metadata = namedtuple("Metadata", ["chroms", "bin_starts", "bin_ends"])


class IntervalsSampler:
    """
    Manages genomic intervals partitioned into train/validate/test by chromosome.

    Does NOT handle sampling iteration — that is delegated to PyTorch DataLoader
    via map-style Seq2SeqDataset.
    """

    def __init__(
        self,
        reference_sequence,
        target,
        intervals_path,
        sequence_length=10000,
        validation_holdout=None,
        test_holdout=None,
        seed=436,
    ):
        self.reference_sequence = reference_sequence
        self.target = target
        self.sequence_length = sequence_length
        self.seed = seed

        np.random.seed(self.seed)
        random.seed(self.seed + 1)

        self.validation_holdout = [str(c) for c in (validation_holdout or [])]
        self.test_holdout = [str(c) for c in (test_holdout or [])]

        self.modes = ["train", "validate"]
        if self.test_holdout:
            self.modes.append("test")

        self._sample_from_mode = {m: SampleIndices([]) for m in self.modes}
        self.sample_from_intervals = []
        self.interval_lengths = []

        self._load_intervals(intervals_path)

    def _load_intervals(self, intervals_path):
        with open(intervals_path, 'r') as fh:
            for index, line in enumerate(fh):
                cols = line.strip().split('\t')
                chrom = cols[0]
                start = int(cols[1])
                end = int(cols[2])

                if chrom in self.validation_holdout:
                    self._sample_from_mode["validate"].indices.append(index)
                elif self.test_holdout and chrom in self.test_holdout:
                    self._sample_from_mode["test"].indices.append(index)
                else:
                    self._sample_from_mode["train"].indices.append(index)

                self.sample_from_intervals.append((chrom, start, end))
                self.interval_lengths.append(end - start)

    def get_mode_indices(self, mode):
        """Return the list of interval indices for a given mode."""
        return self._sample_from_mode[mode].indices

    def retrieve(self, chrom, start, end):
        """Fetch sequence and target data for one genomic interval.

        Returns (seq, targets) as numpy arrays, or None if the interval is invalid.
        """
        retrieved_targets = self.target.get_feature_data(chrom, start, end)

        if retrieved_targets.shape[1] != self.sequence_length:
            return None

        retrieved_seq = self.reference_sequence.get_encoding_from_coords(
            chrom, start, end, pad=True
        )

        if retrieved_seq.shape[0] == 0:
            return None
        if np.sum(retrieved_seq) / float(retrieved_seq.shape[0]) < 0.60:
            return None

        return retrieved_seq, retrieved_targets
