# BRANCH.md — feature/emd-loss

## Summary

Adds an Earth Mover's Distance (EMD) auxiliary loss to the existing `Poisson_PseudoKL` loss function. The EMD term compares per-channel CDFs of predicted vs observed mutation rates along the sequence dimension, providing a weak distribution-shape regularization signal alongside the position-specific Poisson + KL loss.

## Decisions

| # | Topic | Decision |
|---|-------|----------|
| 1 | Motivation | Exploratory — test whether CDF-shape comparison helps |
| 2 | EMD role | Auxiliary term only, always combined with Poisson_PseudoKL |
| 3 | Multi-channel handling | Per-channel EMD, averaged over 4 mutation channels |
| 4 | Mask handling | Exclude masked (low-coverage) positions; per-sample loop |
| 5 | Default weight | `emd_weight = 0.01` (set to 0 to disable entirely) |
| 6 | Monitoring | Individual loss terms fed via `LossMinor.loss_tasks` |
| 7 | Implementation style | Plain function, consistent with rest of `loss.py` |
| 8 | Opt-in vs always-on | Always-on, gated by `emd_weight` |
| 9 | Config label | `config.loss = "poisson_total_kl_emd"` |

## Files changed

| File | Change |
|------|--------|
| `config.py` | `loss` default → `"poisson_total_kl_emd"`; added `emd_weight: float = 0.01` |
| `loss.py` | Added `emd_loss_1d()` (mask-aware per-channel CDF comparison); updated `Poisson_PseudoKL` with `emd_weight` and `return_components` params |
| `training/trainer.py` | `Trainer` stores `emd_weight`, passes to loss, feeds `loss_tasks=[poisson_kl, emd]` to observer |
| `train.py` | Added `--emd-weight` CLI arg (default 0.01); wired into config, summary, and log header |

## Assessment: will EMD loss improve performance?

**Likely no significant improvement. The reasons:**

1. **Loss of positional signal.** EMD normalizes the entire sequence to a probability distribution and compares CDFs — it penalizes the model for wrong distribution *shape*, not wrong *positions*. Two predictions with identical CDFs but mutations at completely wrong loci get the same EMD loss. The core value of a seq2seq model is position-specific prediction; EMD does not reinforce that.

2. **Extreme sparsity.** Mutation targets are binary and extremely sparse (a few 1.0 values among 10,000 positions, all others 0). Normalizing by `sum` divides by near-zero, making the CDF dominated by the zeros. The signal from actual mutation positions is diluted.

3. **The Poisson term already handles rate calibration.** The `rate - count * log(rate)` term captures total mutation burden per region, and the KL term captures per-position accuracy. There is no clear gap that EMD fills.

**Possible (marginal) benefit:** EMD might act as a very weak regularizer that prevents the model from collapsing to a uniform prediction in regions with no mutations. However, the masking and KL loss already handle this.

**Recommendation:** Run an A/B comparison (`--emd-weight 0` vs `--emd-weight 0.01`) on a full training run. If validation loss and per-nucleotide Pearson correlation are indistinguishable, abandon the branch. If there is a small but consistent improvement, EMD may be providing a useful regularization effect worth keeping at a low weight.
