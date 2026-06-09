import torch


def poisson(y_pred, y_true, epsilon: float = 1e-7, mask=None):
    """Poisson deviance: rate - count * log(rate), divided by effective length."""
    rate = y_pred.sum(dim=-1)
    count = y_true.sum(dim=-1)
    eff_len = mask.sum(dim=-1).clamp_min(1.0) if mask is not None else y_pred.shape[-1]
    return (rate - count * torch.log(rate + epsilon)) / eff_len


def PseudoPoissonKL(y_pred, y_true, epsilon: float = 1e-10):
    """Element-wise: y_true * log(y_true / y_pred) + y_pred - y_true."""
    return y_true * torch.log((y_true + epsilon) / (y_pred + epsilon)) + y_pred - y_true


def Poisson_PseudoKL(y_pred, y_true, total_weight: float = 1.0, mask=None):
    """
    Combined Poisson + Pseudo-KL loss.

    Args:
        y_pred: (B, C, L) — predicted mutation rates
        y_true: (B, C, L) — target mutation rates
        total_weight: weight of the Poisson term relative to KL
        mask: (B, 1, L) or (B, C, L) — per-position mask; divides by effective length

    Returns:
        scalar loss averaged over batch
    """
    if mask is not None:
        eff_len = mask.sum(dim=-1).clamp_min(1.0)  # (B, 1) or (B, C)
    else:
        eff_len = y_true.shape[-1]

    poisson_term = poisson(y_pred, y_true, mask=mask)

    kl_values = PseudoPoissonKL(y_pred, y_true)  # (B, C, L)
    kl_term = kl_values.sum(dim=-1) / eff_len  # (B, C) or (B, 1) → broadcast

    loss_raw = kl_term + total_weight * poisson_term  # (B, C)
    return loss_raw.mean()
