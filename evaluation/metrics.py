import numpy as np
from scipy.stats import pearsonr


def apply_coverage_mask(predictions, targets, mask):
    """Zero out masked positions in both predictions and targets."""
    masked_preds = predictions * mask[:, np.newaxis, :]
    masked_targets = targets * mask[:, np.newaxis, :]
    return masked_preds, masked_targets


def calc_regional_correlation(pred, true):
    """
    Pearson correlation of regional mean mutation rates.

    Args:
        pred: (N, C, L) numpy array
        true: (N, C, L) numpy array

    Returns:
        list of correlation coefficients, one per channel
    """
    if isinstance(pred, np.ndarray):
        pass
    else:
        pred = pred.cpu().numpy()
        true = true.cpu().numpy()

    pred_regional = pred.mean(axis=2)  # (N, C)
    true_regional = true.mean(axis=2)

    corrs = []
    for i in range(pred_regional.shape[1]):
        corr, _ = pearsonr(pred_regional[:, i], true_regional[:, i])
        corrs.append(corr)
    return corrs


def calc_regional_correlation_grouped(pred, true, seq_indices):
    """
    Per-nucleotide-class grouped Pearson correlation.

    For each output channel and each nucleotide (A/C/G/T), compute the
    Pearson correlation of regional mean predictions vs. regional mean
    targets, restricted to positions of that nucleotide.

    Args:
        pred: (N, C, L) numpy array
        true: (N, C, L) numpy array
        seq_indices: (N, L) numpy array of nucleotide indices (0=A, 1=C, 2=G, 3=T)

    Returns:
        numpy array of shape (4, C) — rows are nucleotides, cols are channels
    """
    if not isinstance(pred, np.ndarray):
        pred = pred.cpu().numpy()
        true = true.cpu().numpy()
        seq_indices = seq_indices.cpu().numpy()

    N, C, L = pred.shape
    corrs = np.full((4, C), np.nan)

    for nuc in range(4):
        for ch in range(C):
            pred_means = []
            true_means = []
            for n in range(N):
                mask = (seq_indices[n] == nuc)
                if mask.sum() > 0:
                    pred_means.append(pred[n, ch, mask].mean())
                    true_means.append(true[n, ch, mask].mean())
            if len(pred_means) > 1:
                corr, _ = pearsonr(pred_means, true_means)
                corrs[nuc, ch] = corr

    return corrs
