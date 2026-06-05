import copy
import pickle

import numpy as np
import torch


class EarlyStopping:
    """Early stopping: tracks best loss and restores best model weights."""

    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
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
    model.load_state_dict(torch.load(load_path, map_location='cpu'))


def reverse_complement_batch(sequence, target):
    """
    Reverse-complement a batch: flip (B, 4, L) sequence and (B, C, L) target.

    sequence: (B, 4, L) with channels [A, C, G, T]
    Returns (B, 4, L) with channels [T, G, C, A] reversed along L axis.
    """
    seq_rc = torch.flip(sequence, dims=[1, 2])  # reverse base order AND position
    target_rc = torch.flip(target, dims=[2])    # reverse position only
    return seq_rc, target_rc
