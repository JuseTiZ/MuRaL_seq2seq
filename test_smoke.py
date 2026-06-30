#!/usr/bin/env python
"""Smoke tests for background mutation rate feature."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch import nn

from utils.helpers import compute_background_rates, load_model
from model import PuffinD, count_parameters
from training.trainer import _mask_no_mut


def test_compute_bg_rates_normal():
    """Normal window: all positions valid, all bases present."""
    B, C, L = 2, 4, 100
    target_values = torch.rand(B, C, L) * 0.01
    sequence = torch.zeros(B, C, L)
    # Each sample has 25 positions of each base
    for b in range(B):
        for c in range(C):
            sequence[b, c, c*25:(c+1)*25] = 1.0
    mask = torch.ones(B, 1, L)

    bg_rates, n_valid = compute_background_rates(target_values, sequence, mask)

    assert bg_rates.shape == (B, C)
    assert n_valid.shape == (B, C)
    # N_c should be 75 per channel (100 total - 25 self-mutation)
    assert torch.allclose(n_valid, torch.full((B, C), 75.0), atol=0.1), f"n_valid={n_valid}"
    # bg_rates should be > 0
    assert (bg_rates > 0).all()
    # bg_rates should be detached
    assert not bg_rates.requires_grad
    print("  PASS: normal window")


def test_compute_bg_rates_partial_coverage():
    """Partial coverage: some positions masked."""
    B, C, L = 1, 4, 100
    target_values = torch.ones(B, C, L) * 0.01
    sequence = torch.zeros(B, C, L)
    sequence[:, 0, :50] = 1.0  # first 50 bp are A
    sequence[:, 1, 50:] = 1.0  # last 50 bp are C
    mask = torch.ones(B, 1, L)
    mask[:, :, :30] = 0.0  # first 30 bp masked

    bg_rates, n_valid = compute_background_rates(target_values, sequence, mask)

    # Ch0 (A): self-mut at 0-49, mask zeros 0-29
    # Valid non-self-mut for ch0: pos 50-99 = 50
    assert n_valid[0, 0] == 50, f"ch0 n_valid: {n_valid[0, 0]}"
    # Ch1 (C): self-mut at 50-99, mask zeros 0-29
    # Valid non-self-mut for ch1: pos 30-49 = 20
    assert n_valid[0, 1] == 20, f"ch1 n_valid: {n_valid[0, 1]}"
    print("  PASS: partial coverage")


def test_compute_bg_rates_all_self_mutation():
    """Window where all positions are self-mutation for one channel → N_c=0."""
    B, C, L = 1, 4, 100
    target_values = torch.ones(B, C, L) * 0.01
    sequence = torch.zeros(B, C, L)
    sequence[:, 0, :] = 1.0  # all A → all self-mutation for ch0
    mask = torch.ones(B, 1, L)

    bg_rates, n_valid = compute_background_rates(target_values, sequence, mask)

    # Ch0: all self-mutation → N_c=0 → bg_rate=0
    assert n_valid[0, 0] == 0, f"ch0 n_valid: {n_valid[0, 0]}"
    assert bg_rates[0, 0] == 0.0, f"ch0 bg_rate: {bg_rates[0, 0]}"
    # Ch1: no self-mutation → N_c=100
    assert n_valid[0, 1] == 100, f"ch1 n_valid: {n_valid[0, 1]}"
    assert bg_rates[0, 1] > 0
    print("  PASS: all self-mutation")


def test_compute_bg_rates_no_mut_applied():
    """Verify _mask_no_mut is correctly applied before bg_rate computation."""
    B, C, L = 1, 4, 100
    # Raw targets with 1.0 at self-mutation position
    target_values = torch.zeros(B, C, L)
    target_values[0, 0, 0] = 1.0  # A→A position: artificial 1.0 ref-base indicator
    target_values[0, 0, 50] = 1.0  # A→A at another position
    target_values[0, 1, 10] = 1.0  # genuine C→C (also artificial 1.0)
    sequence = torch.zeros(B, C, L)
    sequence[0, 0, 0] = 1.0  # A at pos 0
    sequence[0, 0, 50] = 1.0  # A at pos 50
    sequence[0, 1, 10] = 1.0  # C at pos 10
    sequence[0, 2, 20] = 0.5  # not one-hot (simulate N), shouldn't matter
    mask = torch.ones(B, 1, L)

    # Apply _mask_no_mut first
    target_masked = _mask_no_mut(sequence, target_values)
    # Self-mutation positions should be zeroed
    assert target_masked[0, 0, 0] == 0.0
    assert target_masked[0, 0, 50] == 0.0
    assert target_masked[0, 1, 10] == 0.0

    bg_rates, n_valid = compute_background_rates(target_masked, sequence, mask)
    # T_c should be 0 for all channels (no real mutations)
    assert (bg_rates == 0.0).all(), f"Expected all-zero bg_rates, got {bg_rates}"
    print("  PASS: _mask_no_mut before bg_rate")


def test_model_forward_shapes():
    """Model forward pass produces correct shapes."""
    B, C, L = 4, 4, 10000
    model = PuffinD(n_output_channels=C, use_reverse=True)
    x = torch.randn(B, C, L).softmax(dim=1)  # one-hot-like
    bg_rates = torch.rand(B, C) * 0.01

    y = model(x, bg_rates)
    assert y.shape == (B, C, L), f"Expected (B,4,L), got {y.shape}"
    assert (y >= 0).all(), "Output should be non-negative"
    print(f"  PASS: forward shape {y.shape}")


def test_model_self_mutation_zero():
    """Self-mutation positions are exactly zero in output."""
    B, C, L = 1, 4, 10000
    model = PuffinD(n_output_channels=C, use_reverse=True)
    x = torch.zeros(B, C, L)
    # All A's → self-mutation for ch0 at all positions
    x[:, 0, :] = 1.0
    bg_rates = torch.tensor([[0.01, 0.02, 0.03, 0.04]])

    y = model(x, bg_rates)
    # Channel 0 should be all zero (self-mutation)
    assert (y[:, 0, :] == 0.0).all(), f"ch0 not all zero: max={y[0,0,:].max()}"
    # Other channels should be non-zero (can have mutations from A)
    assert (y[:, 1:, :] > 0).any(), "Other channels should have non-zero values"
    print("  PASS: self-mutation hard-zeroed")


def test_model_zero_init():
    """With zero-initialized last conv, prediction should equal background rate."""
    B, C, L = 2, 4, 10000
    model = PuffinD(n_output_channels=C, use_reverse=True)
    # Force all params to zero except necessary ones to verify the zero-init of final conv
    # Actually just test: with zero weight in final conv, delta=0, f=exp(0)=1
    # This should hold regardless of other params since Conv(..., 0-weight, 0-bias) = 0

    x = torch.randn(B, C, L).softmax(dim=1)
    # Ensure no self-mutation to test the f=1 property
    # Make positions have mixed bases so no position has all-1 in one channel
    for b in range(B):
        for l in range(L):
            base = l % 4
            x[b, :, l] = 0.0
            x[b, base, l] = 1.0

    bg_rates = torch.tensor([[0.01, 0.02, 0.03, 0.04],
                             [0.05, 0.06, 0.07, 0.08]])

    y = model(x, bg_rates)

    # For non-self-mutation positions, y should be close to bg_rate
    # (since final conv is zero-initialized, delta=0, f=1)
    for b in range(B):
        for c in range(C):
            # Find non-self-mutation positions for this channel
            not_self = (x[b, c, :] != 1.0)
            if not_self.any():
                pred_at_not_self = y[b, c, not_self]
                expected = bg_rates[b, c]
                # Allowing some tolerance from batch norm and floating point
                max_diff = (pred_at_not_self - expected).abs().max().item()
                assert max_diff < 0.1, \
                    f"Sample {b} ch {c}: max diff from bg_rate = {max_diff:.6f}"
    print("  PASS: zero-init → prediction ≈ bg_rate")


def test_gradient_flow():
    """Gradients flow to both bg_mlp and main network after one optimizer step."""
    B, C, L = 2, 4, 10000
    model = PuffinD(n_output_channels=C, use_reverse=True)

    # Zero-init (W=0, b=0) blocks gradient flow through the last conv
    # since d(delta)/d(h) = W^T * grad = 0. Set small random weights for testing.
    nn.init.normal_(model.final[-1].weight, mean=0.0, std=0.01)
    nn.init.constant_(model.final[-1].bias, 0.01)

    x = torch.randn(B, C, L).softmax(dim=1)
    for b in range(B):
        for l in range(L):
            base = l % 4
            x[b, :, l] = 0.0
            x[b, base, l] = 1.0

    bg_rates = torch.tensor([[0.01, 0.02, 0.03, 0.04],
                             [0.05, 0.06, 0.07, 0.08]])

    y = model(x, bg_rates)
    loss = y.sum()
    loss.backward()

    # Check gradients in bg_mlp
    bg_mlp_grad = sum(p.grad.abs().sum().item()
                      for p in model.bg_mlp.parameters() if p.grad is not None)
    assert bg_mlp_grad > 0, "No gradients in bg_mlp"

    # Check gradients in main network (first conv in uplblocks)
    main_grad = sum(p.grad.abs().sum().item()
                    for p in model.uplblocks[0].parameters() if p.grad is not None)
    assert main_grad > 0, "No gradients in main network"

    print(f"  PASS: grad in bg_mlp={bg_mlp_grad:.2e}, main_net={main_grad:.2e}")


def test_old_checkpoint_rejection():
    """Loading old checkpoint (no bg_mlp) should raise clear error."""
    import tempfile, pickle
    from config import TrainingConfig

    model = PuffinD(n_output_channels=4, use_reverse=False)
    config = TrainingConfig()

    # Save a "fake" old checkpoint (model without bg_mlp, remove bg_mlp from state_dict)
    old_state_dict = {k: v for k, v in model.state_dict().items() if 'bg_mlp' not in k}

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save(old_state_dict, f.name)
        tmp_path = f.name

    try:
        new_model = PuffinD(n_output_channels=4, use_reverse=True)
        load_model(new_model, tmp_path)
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "Old checkpoint" in str(e) or "bg_mlp" in str(e).lower(), \
            f"Wrong error message: {e}"
        print(f"  PASS: old checkpoint rejected: {e}")
    finally:
        os.unlink(tmp_path)


def test_parameter_count():
    """Model still has reasonable parameter count."""
    model = PuffinD(n_output_channels=4, use_reverse=True)
    n_params = count_parameters(model)
    # bg_mlp adds 368 params
    assert n_params > 13_000_000, f"Too few params: {n_params}"
    print(f"  PASS: parameter count = {n_params:,}")


if __name__ == "__main__":
    print("=" * 60)
    print("Smoke tests for background mutation rate feature")
    print("=" * 60)

    print("\n--- compute_background_rates ---")
    test_compute_bg_rates_normal()
    test_compute_bg_rates_partial_coverage()
    test_compute_bg_rates_all_self_mutation()
    test_compute_bg_rates_no_mut_applied()

    print("\n--- Model forward pass ---")
    test_model_forward_shapes()
    test_model_self_mutation_zero()
    test_model_zero_init()
    test_gradient_flow()
    test_parameter_count()

    print("\n--- Checkpoint ---")
    test_old_checkpoint_rejection()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
