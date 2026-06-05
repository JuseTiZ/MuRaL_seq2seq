import random
from collections import namedtuple

import numpy as np

SampleIndices = namedtuple("SampleIndices", ["indices"])
Batch = namedtuple("Batch", ["sequences", "targets", "metadatas"])
Metadata = namedtuple("Metadata", ["chroms", "bin_starts", "bin_ends"])


class IntervalsSampler:
    """
    Samples fixed-size genomic windows from a BED intervals file,
    partitioned into train/validate/test by chromosome holdout.
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

        self._mode = "train"
        self._sample_from_mode = {m: SampleIndices([]) for m in self.modes}
        self._randcache = {m: {"cache_indices": None, "sample_next": 0} for m in self.modes}

        self.sample_from_intervals = []
        self.interval_lengths = []

        self._load_intervals(intervals_path)
        for mode in self.modes:
            self._update_randcache(mode)

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

    def _update_randcache(self, mode=None):
        if mode is None:
            mode = self._mode
        indices = self._sample_from_mode[mode].indices
        self._randcache[mode]["cache_indices"] = np.random.choice(
            indices, size=len(indices), replace=False
        )
        self._randcache[mode]["sample_next"] = 0

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        if value not in self.modes:
            raise ValueError(f"Mode must be one of {self.modes}, got '{value}'")
        self._mode = value

    @property
    def n_samples(self):
        return len(self._sample_from_mode[self._mode].indices)

    def _retrieve(self, chrom, start, end):
        retrieved_targets = self.target.get_feature_data(chrom, start, end)

        if retrieved_targets.shape[1] != self.sequence_length:
            return None

        retrieved_seq = self.reference_sequence.get_encoding_from_coords(
            chrom, start, end, pad=True
        )

        if retrieved_seq.shape[0] == 0:
            return None
        if np.sum(retrieved_seq) / float(retrieved_seq.shape[0]) < 0.60:
            # >40% ambiguous bases — reject
            return None

        return retrieved_seq, retrieved_targets

    def sample(self, batch_size=1, mode=None):
        """Draw a mini-batch. Returns Batch(sequences, targets, metadatas)."""
        mode = mode if mode is not None else self._mode

        sequences = np.zeros((batch_size, self.sequence_length, 4), dtype=np.float32)
        targets = np.zeros(
            (batch_size, self.target.n_features, self.sequence_length), dtype=np.float32
        )
        metadatas = [None] * batch_size

        n_drawn = 0
        while n_drawn < batch_size:
            sample_index = self._randcache[mode]["sample_next"]
            if sample_index >= len(self._sample_from_mode[mode].indices):
                self._update_randcache(mode)
                sample_index = 0

            interval_idx = self._randcache[mode]["cache_indices"][sample_index]
            self._randcache[mode]["sample_next"] += 1

            chrom, bin_start, bin_end = self.sample_from_intervals[interval_idx]
            result = self._retrieve(chrom, bin_start, bin_end)
            if result is None:
                continue

            seq, seq_targets = result
            sequences[n_drawn] = seq
            targets[n_drawn] = seq_targets
            metadatas[n_drawn] = Metadata(chrom, bin_start, bin_end)
            n_drawn += 1

        return Batch(sequences, targets, metadatas)
