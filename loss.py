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


def emd_loss_1d(y_pred, y_true, mask=None, emd_mask=None):
    """
    Per-channel Earth Mover's Distance on CDFs.

    For each sample and channel, gathers unmasked positions, normalizes to
    sum=1, computes CDFs via cumsum, and takes L1 distance.

    Args:
        y_pred:   (B, C, L) predicted rates (already masked)
        y_true:   (B, C, L) target rates (already masked)
        mask:     (B, 1, L) or (B, C, L) coverage mask (used as fallback)
        emd_mask: (B, C, L) mask for EMD — excludes self-mutation positions
                  in addition to coverage. If None, falls back to mask.

    Returns:
        scalar — mean EMD over batch and channels
    """
    B, C, L = y_pred.shape
    total_emd = 0.0
    n_valid = 0

    for b in range(B):
        for c in range(C):
            if emd_mask is not None:
                m = emd_mask[b, c, :] > 0
            elif mask is not None:
                if mask.dim() == 3 and mask.shape[1] == 1:
                    m = mask[b, 0, :] > 0
                else:
                    m = mask[b, c, :] > 0
            else:
                m = torch.ones(L, dtype=torch.bool, device=y_pred.device)

            p = y_pred[b, c, :][m]
            t = y_true[b, c, :][m]

            if p.numel() < 2:
                continue
            if t.sum() < 1e-8:
                continue

            p_norm = p / (p.sum() + 1e-8)
            t_norm = t / (t.sum() + 1e-8)

            p_cdf = torch.cumsum(p_norm, dim=0)
            t_cdf = torch.cumsum(t_norm, dim=0)

            total_emd += torch.abs(p_cdf - t_cdf).mean()
            n_valid += 1

    if n_valid == 0:
        return torch.tensor(0.0, device=y_pred.device)
    return total_emd / n_valid


def Poisson_PseudoKL(y_pred, y_true, total_weight: float = 1.0, mask=None,
                     emd_weight: float = 0.01, emd_mask=None,
                     return_components: bool = False):
    """
    Combined Poisson + Pseudo-KL + EMD loss.

    Args:
        y_pred:     (B, C, L) — predicted mutation rates
        y_true:     (B, C, L) — target mutation rates
        total_weight: weight of the Poisson term relative to KL
        mask:       (B, 1, L) or (B, C, L) — per-position coverage mask
        emd_weight: weight of the EMD auxiliary term (set to 0 to disable)
        emd_mask:   (B, C, L) — mask for EMD term (excludes self-mutation
                    positions). If None, EMD falls back to mask.
        return_components: if True, also return a dict with individual loss terms

    Returns:
        scalar loss, or (loss, dict) if return_components is True
    """
    if mask is not None:
        eff_len = mask.sum(dim=-1).clamp_min(1.0)  # (B, 1) or (B, C)
    else:
        eff_len = y_true.shape[-1]

    poisson_term = poisson(y_pred, y_true, mask=mask)

    kl_values = PseudoPoissonKL(y_pred, y_true)  # (B, C, L)
    kl_term = kl_values.sum(dim=-1) / eff_len  # (B, C) or (B, 1) → broadcast

    loss_raw = kl_term + total_weight * poisson_term  # (B, C)
    poisson_kl = loss_raw.mean()

    if emd_weight > 0:
        emd_raw = emd_loss_1d(y_pred, y_true, mask, emd_mask=emd_mask)
        loss = poisson_kl + emd_weight * emd_raw
    else:
        emd_raw = torch.tensor(0.0, device=y_pred.device)
        loss = poisson_kl

    if return_components:
        return loss, {
            'poisson_kl': poisson_kl.item(),
            'emd_raw': emd_raw.item(),
            'emd_weighted': (emd_weight * emd_raw).item(),
        }
    return loss
