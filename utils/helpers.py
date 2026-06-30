import copy
import pickle

import numpy as np
import torch


def compute_background_rates(target_values, sequence, mask):
    """
    Compute per-channel window background mutation rates.

    Args:
        target_values: (B, 4, L) — targets AFTER _mask_no_mut (self-mutation
                       positions already zeroed).
        sequence:      (B, 4, L) — one-hot DNA [A, C, G, T].
        mask:          (B, 1, L) — coverage mask (1=valid, 0=invalid).

    Returns:
        bg_rates: (B, 4) — background rate per channel, detached.
        n_valid:  (B, 4) — effective position count per channel.
    """
    # T_c: sum of coverage-valid targets. Self-mutation positions are
    # already zero in target_values (_mask_no_mut applied upstream).
    valid_targets = target_values * mask  # (B, 4, L)
    t_sum = valid_targets.sum(dim=-1)     # (B, 4)

    # N_c: positions that are coverage-valid AND not self-mutation.
    # Self-mutation for channel c: sequence[:, c, :] == 1.
    not_self = (sequence < 1.0).float()   # (B, 4, L), 0 at self-mutation pos
    n_valid = (mask * not_self).sum(dim=-1)  # (B, 4)

    # b_c = T_c / N_c; force zero where no valid positions.
    bg_rates = t_sum / n_valid.clamp_min(1.0)
    bg_rates[n_valid < 0.5] = 0.0

    return bg_rates.detach(), n_valid


class EarlyStopping:
    """Early stopping: tracks best loss and restores best model weights."""

    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.best_epoch = -1
        self.best_state = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
            self.early_stop = False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.best_state is not None:
                    model.load_state_dict(self.best_state)
        return self.early_stop


def save_model(model, config, save_path):
    torch.save(model.state_dict(), save_path)
    with open(save_path + '.config.pkl', 'wb') as f:
        pickle.dump(config, f)


def load_model(model, load_path):
    state_dict = torch.load(load_path, map_location='cpu')
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        missing = [k for k in model.state_dict().keys()
                   if k not in state_dict]
        if any('bg_mlp' in k for k in missing):
            raise RuntimeError(
                "Old checkpoint detected (no bg_mlp parameters). "
                "This branch requires a model trained with background "
                "mutation rate conditioning. Please retrain from scratch."
            ) from e
        raise


def reverse_complement_batch(sequence, target):
    """
    Reverse-complement a batch: flip (B, 4, L) sequence and (B, C, L) target.

    sequence: (B, 4, L) with channels [A, C, G, T]
    target: (B, C, L) with channels [mut_to_A, mut_to_C, mut_to_G, mut_to_T, mask]
    After reverse-complement, mut_to_A <-> mut_to_T and mut_to_C <-> mut_to_G.
    """
    seq_rc = torch.flip(sequence, dims=[1, 2])
    target_rc = target.clone()
    target_rc[:, :4] = torch.flip(target[:, :4], dims=[1, 2])  # swap A<->T, C<->G, and reverse pos
    target_rc[:, 4:] = torch.flip(target[:, 4:], dims=[2])     # mask: position-reverse only
    return seq_rc, target_rc
