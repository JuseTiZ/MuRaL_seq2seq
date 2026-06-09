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
    model.load_state_dict(torch.load(load_path, map_location='cpu'))


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
