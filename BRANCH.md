# BRANCH.md — feature/background-mutation-rate

## Summary

Modifies the PuffinD seq2seq model to use per-window background mutation rates as conditional input. Instead of predicting absolute rates from sequence alone, the model predicts per-position **correction factors** relative to a 4-channel window background rate `b_c = T_c / N_c` (computed from targets within the current window). This allows the model to focus on learning sequence-specific rate deviations from the regional average.

Key equation: `ŷ_{c,i} = b_c × exp(δ_{c,i})`, where δ is the model's output logits (zero-initialized, so initial prediction = background rate).

## Decisions

| # | Topic | Decision | Rationale |
|---|-------|----------|-----------|
| 1 | Background rate formula | `b_c = T_c / N_c` (N_c=0 → b_c=0) | Simple, computed in-batch from targets |
| 2 | Computation order | `_mask_no_mut` → `compute_background_rates` | Self-mutation positions hold artificial 1.0 in BigWig — must zero them first to avoid inflating T_c |
| 3 | Correction factor form | `f = exp(δ)` | Symmetric: 2× up and 1/2× down are equal-|δ|. δ=0 → f=1 when last conv is zero-initialized |
| 4 | Zero-initialization | Last conv weights & bias zeroed | Ensures initial prediction = background rate |
| 5 | Background MLP | `log(b+1e-10) → Linear(4,16) → SiLU → Linear(16,16) → SiLU` | Log handles wide dynamic range; 368 params; SiLU consistent with ConvBlock activations |
| 6 | bg_emb injection point | Before output head: `[seq_emb (68); bg_emb (16)] → output head` | Full interaction through non-linearity. Alternative (inject post-ReLU, before final conv) noted for future ablation |
| 7 | `total_weight` default | 0.0 (was 1.0) | Background rate provides base level; window Poisson is redundant. Kept as CLI option for ablation |
| 8 | Invalid channel handling (N_c=0) | Excluded from loss averaging in trainer | Prevents zero-contribution channels from diluting loss |
| 9 | Old checkpoint loading | Catch `RuntimeError` from `load_state_dict`, check for `bg_mlp` keys | Clear error message; no backward compatibility |
| 10 | RC augmentation compatibility | bg_rate computed after RC (in trainer) | RC-transformed targets have channels already swapped (A↔T, C↔G), so bg_rate is consistent |
| 11 | Self-mutation in model output | Hard-zeroed in forward: `y * (1 - x)` | Model never needs to learn to suppress these positions |
| 12 | `_mask_no_mut` retained | Yes — for numerical stability | Protects loss from NaN when y_pred=0 but y_true=1.0 at self-mutation positions |
| 13 | Validation fold change logging | Quartiles (p25, p50, p75) per channel | Mean can mask heterogeneity |
| 14 | Validation additional metrics | Per-channel bg_mean, pred_mean, fold quartiles | Tracks how much model adjusts relative to background |
| 15 | Background rate gradient | Detached (`b_c.detach()`) | Conditioning input, not learnable |
| 16 | Center-crop prediction | bg_rate from full input window (including flanks), N_c uses full-window effective positions | Consistent with training; full context used for background |
| 17 | predict.py redundant hard mask | Removed (`preds * (1-x)` already in model forward) | Double-application incorrectly dampens N positions |

## Files changed

| File | Change |
|------|--------|
| `config.py` | `total_weight` default: 1.0 → 0.0 |
| `model.py` | Added `self.bg_mlp` (4→16→16 with SiLU); modified `self.final` input channels: `final_dim` → `final_dim+16`, removed Softplus, zero-initialized last conv; `forward(self, x, bg_rates)` — computes bg_emb, concats, predicts δ, returns `bg_rate × exp(δ) × (1-x)` |
| `utils/helpers.py` | Added `compute_background_rates(target_values, sequence, mask)` → (bg_rates, n_valid); updated `load_model()` with old checkpoint detection |
| `training/trainer.py` | `train_step`/`valid_step`: `_mask_no_mut` → `compute_background_rates` → `model(seq, bg_rates)`; `channel_valid` loss correction for N_c=0 channels; `valid_step` returns `(preds, bg_rates, n_valid)` |
| `train.py` | Validation: collects `bg_rates`, computes per-channel fold change quartiles (p25, p50, p75); `--total-weight` CLI default: 0.0 |
| `predict.py` | `CenterCropPredictionDataset` now returns targets for bg_rate computation (via `sampler.target`); `_collate_center_crop` batches targets; prediction loop computes bg_rates and passes to model; removed redundant hard mask; uses `load_model()` for old checkpoint detection |
| `BRANCH.md` | This file |

## Assessment: will background rate conditioning improve performance?

**Expected improvement: moderate.** The reasoning:

1. **Simplified learning problem.** Instead of predicting absolute rates (which requires learning regional baselines from sequence alone), the model only needs to predict position-specific deviations. The background rate encodes all non-sequence regional factors (replication timing, chromatin state, etc.) that would otherwise need to be inferred from sequence.

2. **Beneficial inductive bias.** The `f = exp(δ)` formulation with zero-init means the model starts from the simple "constant background rate" hypothesis and learns to add sequence-specific modulation. This is a natural curriculum.

3. **Robust to noisy background.** The background rate may be biased (target labels are binary indicators, not true rates), but the model can learn to correct systematic biases through the learned correction factor. The `exp(δ)` form allows unlimited up/down scaling.

4. **Potential concern — information leakage?** The background rate is computed from targets, so the model receives information derived from what it's trying to predict. However, b_c is a single scalar per channel — it carries no per-position information. The model still needs DNA sequence to make position-specific predictions.

5. **Loss of总量 constraint.** With `total_weight=0`, there is no explicit penalty on the total predicted count deviating from the observed count. The Pseudo-KL provides a weak总量 constraint, but models might learn systematic scaling biases. The optional `total_weight` parameter allows re-enabling Poisson regularization if needed.

**Recommendation:** Train with `total_weight=0` as baseline. If validation fold change distributions show systematic deviation from 1.0 (median fold ≠ 1), try small positive `total_weight` values (0.01, 0.1) to see if constraining总量 improves metrics.
