# BRANCH.md — feature/emd-loss

## Summary

Adds an Earth Mover's Distance (EMD) auxiliary loss to the existing `Poisson_PseudoKL` loss function. The EMD term compares per-channel CDFs of predicted vs observed mutation rates along the sequence dimension, providing a weak distribution-shape regularization signal alongside the position-specific Poisson + KL loss.

## Decisions

| # | Topic | Decision |
|---|-------|----------|
| 1 | Motivation | Exploratory — test whether CDF-shape comparison helps |
| 2 | EMD role | Auxiliary term only, always combined with Poisson_PseudoKL |
| 3 | Multi-channel handling | Per-channel EMD, averaged over 4 mutation channels |
| 4 | Mask handling | Coverage mask for Poisson/KL; separate `emd_mask` (coverage × mutation-possible) for EMD, excluding self-mutation positions |
| 5 | Default weight | `emd_weight = 0.01` (set to 0 to disable entirely) |
| 6 | Monitoring | Three loss components fed via `LossMinor.loss_tasks`: `poisson_kl`, `emd_raw`, `emd_weighted` |
| 7 | Implementation style | Plain function, consistent with rest of `loss.py` |
| 8 | Opt-in vs always-on | Always-on, gated by `emd_weight` |
| 9 | Config label | `config.loss = "poisson_total_kl_emd"` |
| 10 | Zero-target channels | Skipped when `t.sum() < 1e-8` to avoid noisy gradients on empty targets |
| 11 | EMD mask vs Poisson/KL mask | Separate `emd_mask` parameter — Poisson/KL path completely untouched |

## Files changed

| File | Change |
|------|--------|
| `config.py` | `loss` default → `"poisson_total_kl_emd"`; added `emd_weight: float = 0.01` |
| `loss.py` | Added `emd_loss_1d()` (mask-aware per-channel CDF comparison, `emd_mask` param, zero-target gate); updated `Poisson_PseudoKL` with `emd_weight`, `emd_mask`, and `return_components` (`poisson_kl`/`emd_raw`/`emd_weighted`) |
| `training/trainer.py` | Added `_get_no_mut_mask()` helper; `Trainer` stores `emd_weight`, computes `emd_mask`, passes to loss, feeds 3-component `loss_tasks` to observer |
| `training/observer.py` | Fixed `loss_tasks` accumulation to weight by `sample_number` (consistent with main loss averaging) |
| `train.py` | Added `--emd-weight` CLI arg (default 0.01); wired into config, summary, and log header |

## Code review findings and fixes

A code review identified 6 potential issues. After verification:

| # | Issue | Severity | Disposition |
|---|-------|----------|-------------|
| 1 | EMD included ref→ref self-mutation positions in CDF comparison | HIGH | **Fixed.** Added `emd_mask` parameter — EMD now excludes self-mutation positions. Poisson/KL path unchanged. |
| 2 | Zero-target channel-windows produce unstable/noisy EMD gradients | HIGH | **Fixed.** Added `t.sum() < 1e-8` gate in `emd_loss_1d` — degenerate channels are skipped. |
| 3 | EMD normalizes away total mutation burden, may conflict with Poisson term | — | **By design.** EMD is a distribution-shape regularizer only; the Poisson term handles total rate. |
| 4 | `components["emd"]` reported weighted value, not raw EMD | MEDIUM | **Fixed.** Components dict now returns `emd_raw` and `emd_weighted` separately. |
| 5 | `LossMinor.loss_tasks` averaged without sample-number weighting | LOW | **Fixed.** Now uses `np.asarray(loss_tasks) * sample_number`. |
| 6 | Validation correlation uses targets without `_mask_no_mut` | MEDIUM (pre-existing) | **Deferred.** Model selection is based on loss, not correlation. Correlation is for reference only. |

## Assessment: will EMD loss improve performance?

**Likely no significant improvement.** The reasons:

1. **Loss of positional signal.** EMD normalizes the entire sequence to a probability distribution and compares CDFs — it penalizes the model for wrong distribution *shape*, not wrong *positions*. Two predictions with identical CDFs but mutations at completely wrong loci get the same EMD loss.

2. **Extreme sparsity.** Mutation targets are binary and extremely sparse. After the zero-target gate (Issue 2 fix), many channel-windows are skipped entirely. The remaining windows with real mutations produce a meaningful CDF comparison, but the signal is still dominated by zero-rate background positions.

3. **The Poisson term already handles rate calibration.** The `rate - count * log(rate)` term captures total mutation burden, and the KL term captures per-position accuracy. There is no clear gap that EMD fills.

**After the fixes above**, the EMD term is now computing a clean signal: it compares CDFs only on real-mutation-possible positions (excluding ref→ref) and only on windows with actual target mass. This eliminates the two largest sources of spurious gradient. However, the fundamental limitation remains — EMD does not reward positional accuracy, which is the core task.

**Recommendation:** Run A/B comparison (`--emd-weight 0` vs `--emd-weight 0.01`) on a full training run. Monitor `loss_task_0` (poisson_kl) and `loss_task_1` (emd_raw) separately — if poisson_kl is not lower with EMD enabled, the auxiliary term is not helping. If there is a small but consistent improvement, EMD may provide useful regularization at a low weight.
