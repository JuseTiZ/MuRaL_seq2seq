import numpy as np
import pyBigWig


class GenomicSignalFeatures:
    """Loads BigWig files and provides per-position feature data."""

    def __init__(self, input_paths, feature_names):
        self.input_paths = input_paths
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.feature_index_dict = {name: i for i, name in enumerate(feature_names)}
        self._initialized = False
        self._bw_handles = None

    def _init(self):
        if not self._initialized:
            self._bw_handles = [pyBigWig.open(p) for p in self.input_paths]
            self._initialized = True

    def get_feature_data(self, chrom, start, end):
        """Return (n_features, length) numpy array of BigWig values."""
        self._init()
        wigmat = np.vstack([
            bw.values(chrom, start, end, numpy=True) for bw in self._bw_handles
        ])
        wigmat[np.isnan(wigmat)] = 0
        wigmat[wigmat < 0] = 0
        return wigmat.astype(np.float32)
