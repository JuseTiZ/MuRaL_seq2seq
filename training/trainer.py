import time
import sys

import torch


class Trainer:
    """Training loop for seq2seq mutation rate prediction."""

    def __init__(self, model, config, device, observers=None):
        self.model = model
        self.config = config
        self.device = device
        self.observers = observers or []

        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # LR scheduler: per-step exponential decay
        gamma_step = config.lr_gamma ** (config.batch_size / config.train_size)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=gamma_step
        )
        print(f"LR gamma per step: {gamma_step:.6f}")

        self.total_weight = config.total_weight
        self.printer = print

    def train_step(self, sequence, target):
        self.model.train()
        t_start = time.time()

        mask = target[:, -1, :].unsqueeze(1)  # (B, 1, L)  coverage mask
        target_values = target[:, :-1, :]      # (B, C_out, L)  mutation rate labels

        sequence = sequence.to(self.device)
        target_values = target_values.to(self.device)
        mask = mask.to(self.device)

        preds = self.model(sequence)

        # Mask out no-mutation positions in target: e.g. at an A site,
        # mut_to_A label is a ref-base indicator (1.0), not a real rate.
        # Only zero the target (not preds) so the model learns to
        # suppress predictions at matched-base positions via the loss.
        target_values = _mask_no_mut(sequence, target_values)

        preds = preds * mask
        target_values = target_values * mask

        from mural_s2s.loss import Poisson_PseudoKL
        loss = Poisson_PseudoKL(preds, target_values, total_weight=self.total_weight)

        if not torch.isfinite(loss):
            self.printer("WARNING: non-finite loss, ending training", file=sys.stderr)
            sys.exit(1)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.gradient_clip_norm,
            error_if_nonfinite=False,
        )
        self.optimizer.step()
        self.scheduler.step()

        # LR floor
        if self.optimizer.param_groups[0]['lr'] < self.config.min_lr:
            for g in self.optimizer.param_groups:
                g['lr'] = self.config.min_lr

        train_time = time.time() - t_start
        loss_val = loss.item()

        for obs in self.observers:
            obs.update(
                model=self.model,
                loss=loss_val,
                sample_number=sequence.size(0),
                train_time=train_time,
                mode="train",
            )

    def valid_step(self, sequence, target):
        self.model.eval()
        with torch.no_grad():
            mask = target[:, -1, :].unsqueeze(1)
            target_values = target[:, :-1, :]

            sequence = sequence.to(self.device)
            target_values = target_values.to(self.device)
            mask = mask.to(self.device)

            preds = self.model(sequence)

            target_values = _mask_no_mut(sequence, target_values)

            preds_masked = preds * mask
            target_masked = target_values * mask

            from mural_s2s.loss import Poisson_PseudoKL
            loss = Poisson_PseudoKL(preds_masked, target_masked, total_weight=self.total_weight)
            loss_val = loss.item()

            for obs in self.observers:
                obs.update(
                    loss=loss_val,
                    sample_number=sequence.size(0),
                    mode="validate",
                )

        return preds

    def epoch_finish(self, mode):
        results = {}
        for obs in self.observers:
            ret = obs.update(epoch_finish=True, mode=mode)
            if ret:
                results.update(ret)
        return results


def _mask_no_mut(sequence, targets):
    """
    Zero out target labels at positions where the reference base matches the
    mutation target channel (A→A, C→C, G→G, T→T are not real mutations).

    The BigWig labels store 1.0 as a ref-base indicator at these positions.
    By only masking targets (not predictions), the model is penalized via the
    loss for outputting non-zero values at matched-base positions, which forces
    it to learn to suppress predictions there.

    Args:
        sequence: (B, 4, L) one-hot DNA
        targets:  (B, C_out, L) label values

    Returns:
        targets with no-mutation positions zeroed.
    """
    B, C, L = targets.shape
    keep_mask = torch.ones(B, C, L, device=targets.device, dtype=targets.dtype)
    for ch in range(C):
        base_idx = ch % 4
        keep_mask[:, ch, :] = (sequence[:, base_idx, :] != 1).float().to(targets.dtype)
    return targets * keep_mask
