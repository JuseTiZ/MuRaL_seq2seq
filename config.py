from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainingConfig:
    # --- Data paths ---
    fasta: str = ""
    intervals: str = ""
    target_dir: str = ""
    mask_bw: str = ""

    # --- Target specification ---
    target_features: List[str] = field(default_factory=lambda: [
        "mut_to_A", "mut_to_C", "mut_to_G", "mut_to_T",
    ])
    # BigWig filenames corresponding to each feature
    target_bw_files: List[str] = field(default_factory=lambda: [
        "genome.mut_to_A.bw",
        "genome.mut_to_C.bw",
        "genome.mut_to_G.bw",
        "genome.mut_to_T.bw",
    ])

    # --- Chromosome split ---
    val_chroms: List[str] = field(default_factory=lambda: ["chr1"])
    test_chroms: List[str] = field(default_factory=lambda: ["chr2"])

    # --- Sequence ---
    sequence_length: int = 10000
    center_bin_to_predict: Optional[int] = None  # None = predict full window

    # --- Training ---
    seed: int = 436
    batch_size: int = 32
    epochs: int = 20
    num_workers: int = 0

    # --- Optimizer ---
    learning_rate: float = 5e-3
    min_lr: float = 1e-4
    weight_decay: float = 1e-6
    gradient_clip_norm: float = 10.0

    # --- LR scheduler: exponential decay per step ---
    lr_gamma: float = 0.1  # raw gamma, auto-scaled to per-step

    # --- Loss ---
    total_weight: float = 1.0
    loss: str = "poisson_total_kl_emd"
    emd_weight: float = 0.01

    # --- Early stopping ---
    patience: int = 5

    # --- Augmentation ---
    reverse_complement_aug: bool = False

    # --- Model ---
    use_reverse: bool = True

    # --- Derived / runtime ---
    train_size: int = 0  # set by training script after sampler init
    progress_every_n_batches: int = 100

    # --- Output ---
    output_dir: str = "./output"

    @property
    def target_bw_paths(self) -> List[str]:
        import os
        return [os.path.join(self.target_dir, f) for f in self.target_bw_files]
