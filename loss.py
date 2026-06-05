import torch


def poisson(y_pred, y_true, epsilon: float = 1e-7):
    """Poisson deviance: rate - count * log(rate), divided by seq_len."""
    seq_len = y_pred.shape[-1]
    rate = y_pred.sum(dim=-1)   # predicted total mutation count
    count = y_true.sum(dim=-1)  # observed total mutation count
    return (rate - count * torch.log(rate + epsilon)) / seq_len


def PseudoPoissonKL(y_pred, y_true, epsilon: float = 1e-10):
    """Element-wise: y_true * log(y_true / y_pred) + y_pred - y_true."""
    return y_true * torch.log((y_true + epsilon) / (y_pred + epsilon)) + y_pred - y_true


def Poisson_PseudoKL(y_pred, y_true, total_weight: float = 1.0):
    """
    Combined Poisson + Pseudo-KL loss.

    Args:
        y_pred: (B, C, L) — predicted mutation rates
        y_true: (B, C, L) — target mutation rates
        total_weight: weight of the Poisson term relative to KL

    Returns:
        scalar loss averaged over batch
    """
    seq_len = y_true.shape[-1]

    poisson_term = poisson(y_pred, y_true)  # (B, C)

    kl_values = PseudoPoissonKL(y_pred, y_true)  # (B, C, L)
    kl_term = kl_values.sum(dim=-1) / seq_len  # (B, C)

    loss_raw = kl_term + total_weight * poisson_term  # (B, C)
    return loss_raw.mean()
