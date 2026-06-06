# CLAUDE.md — mural_s2s

Seq2seq germline mutation rate predictor. Pure DNA sequence (one-hot A/C/G/T) → per-position mutation probabilities (4 channels: mut_to_A/C/G/T). Built from scratch, referencing `notebook/unet_model*.ipynb`.

## Environment

```bash
conda activate mural
# Python 3.8.5, PyTorch 1.10.2 + CUDA 11.3
# pyfaidx, pyBigWig, pytabix, scipy
```

## Architecture

**Model**: `PuffinD` U-Net (~13M params), no reverse-complement module.
- Input: `(B, 4, 10000)` one-hot DNA
- Output: `(B, 4, 10000)` via Softplus
- Two encoder-decoder passes with skip connections.

**Loss**: `Poisson_PseudoKL = kl_term + total_weight * poisson_term`
- KL term: `y_true * log(y_true/y_pred) + y_pred - y_true`, element-wise, then mean over L
- Poisson term: `(rate - count * log(rate)) / L` where rate=sum(y_pred), count=sum(y_true)
- Both terms use epsilon protection to prevent log(0)

**Data flow**:
1. `Genome` (pyfaidx) — FASTA → one-hot `(L, 4)`, N→0.25, edge-pad with N
2. `GenomicSignalFeatures` (pyBigWig) — BigWig → `(5, L)`, NaN→0, neg→0
3. `IntervalsSampler` — BED intervals, chr-partitioned (chr1=val, chr2=test)
4. `build_dataloader` → finite PyTorch Dataset, num_workers=0

**Pre-loss masking** (trainer.py `_mask_no_mut`):
1. No-mutation mask: zeros out A→A, C→C, G→G, T→T positions in both preds & targets
2. Coverage mask: zeros out low-coverage positions (from mask_coverage_15_45.bw)
3. Applied in both train_step and valid_step

## Key data facts (critical — verified from actual BigWig inspection)

- **`genome.mut_to_{A,C,G,T}.bw`**: BINARY — only `1.0` and `nan`. `1.0` at matched-ref-base positions (A→A for mut_to_A channel = 1.0) AND at genuine mutation sites. After `_mask_no_mut`, only genuine mutation sites remain as 1.0, everything else → 0.
- **`genome.us1_mid.to_X.bw`**, **`genome.mid_ds1.to_X.bw`**: CONTINUOUS — values 0~0.996. These are k-mer based flanking-region mutation rates. Currently NOT used (model only uses the 4 basic mut_to_X channels).
- **`genome.mask_coverage_15_45.bw`**: Coverage mask (15–45×)

## Bugs fixed during development

1. **Poisson formula was swapped** (loss.py): wrote `count - rate * log(count)` instead of `rate - count * log(rate)`. Caused loss to be negative and decrease without bound. Fixed by standardizing variable names to `y_pred, y_true`.

2. **Inplace modification broke autograd** (trainer.py `_mask_no_mut`): `preds[:, ch, :][ref_mask] = 0` inplace-mutated Softplus output tensor, breaking backward graph. Fixed by using multiplication mask instead.

3. **Model not moved to GPU** (train.py): `model = PuffinD(...)` without `.to(device)`. Fixed.

4. **Module import error**: `python mural_s2s/train.py` fails with `ModuleNotFoundError` because project root not in sys.path. Fixed by adding `sys.path.insert(0, ...)` at top of train.py and predict.py.

## Training behavior (first completed run, 20 epochs)

- LR decays from 5e-3 to min_lr=1e-4 by epoch 3 (too fast — 17 epochs with no LR movement)
- val_loss stabilizes around epoch 10 (~0.00625), very little improvement thereafter
- Early stopping (patience=5) never triggered because loss kept micro-improving
- Best model: checkpoint_19 (val_loss=0.0062439632), though epochs 10–20 nearly identical
- Per-nucleotide-grouped Pearson correlation: ~0.05–0.34 for most channel-nucleotide pairs
- NaN correlations on diagonal (A→A, C→C, etc.) — expected since those positions are masked out

## Design decisions (from grill session)

| Decision | Choice |
|----------|--------|
| Output channels | 4 (basic mut_to_A/C/G/T) |
| Reverse-complement | Model: none. Data aug: configurable, off by default |
| Loss | `poisson_total_kl` (simple, no base-grouped variant) |
| Mask scope | Coverage mask everywhere (loss + evaluation) |
| No-mutation masking | Active (needed: BigWig stores 1.0 at ref-base positions) |
| Epoch definition | Finite Dataset: one pass through all intervals |
| Scheduler | StepLR per-batch exponential decay + min_lr floor |
| Checkpointing | Every epoch + best-epoch tracking (no separate best_model.pt) |

## File map

```
mural_s2s/
├── config.py          TrainingConfig dataclass
├── model.py           PuffinD(n_output_channels), ConvBlock, count_parameters
├── loss.py            poisson(), PseudoPoissonKL(), Poisson_PseudoKL()
├── data/
│   ├── genome.py      Genome(fasta_path) — get_encoding_from_coords()
│   ├── targets.py     GenomicSignalFeatures(bw_paths, names) — get_feature_data()
│   ├── sampler.py     IntervalsSampler — chr-partitioned sampling
│   └── dataloader.py  Seq2SeqDataset, build_dataloader(), collate
├── training/
│   ├── trainer.py     Trainer — Adam+StepLR, train_step(), valid_step(), _mask_no_mut()
│   └── observer.py    LossMinor, GradMinor — loss tracking with current_loss property
├── evaluation/
│   └── metrics.py     calc_regional_correlation(), calc_regional_correlation_grouped()
├── utils/
│   └── helpers.py     EarlyStopping, save_model(), load_model(), reverse_complement_batch()
├── train.py           CLI: --fasta --intervals --target-dir --mask-bw --output-dir ...
└── predict.py         CLI: --model --output --mode {train,validate,test}
```

## Git

```
git@github.com:JuseTiZ/MuRaL_seq2seq.git  (master)
```

## Training command (example)

```bash
conda activate mural
python mural_s2s/train.py \
    --fasta /public5/home/songhui/data/hg19/hg19_ucsc_ordered.fa \
    --intervals /public5/home/songhui/mural_snv/s2m/segments/segments.win10k.step10k.valid_sites_8k.bed \
    --target-dir /public5_data/home/songhui/s2m \
    --mask-bw /public5_data/home/songhui/s2m/genome.mask_coverage_15_45.bw \
    --output-dir ./output \
    --batch-size 32 --epochs 20
```

## Data sources

| Resource | Path |
|----------|------|
| hg19 FASTA | `/public5/home/songhui/data/hg19/hg19_ucsc_ordered.fa` |
| Mutation BigWigs (mut_to_X) | `/public5_data/home/songhui/s2m/genome.mut_to_{A,C,G,T}.bw` |
| Coverage mask | `/public5_data/home/songhui/s2m/genome.mask_coverage_15_45.bw` |
| Flanking BigWigs (unused) | `/public5_data/home/songhui/s2m/genome.{us1_mid,mid_ds1,...}.to_{A,C,G,T}.bw` |
| 10kb intervals BED | `/public5/home/songhui/mural_snv/s2m/segments/segments.win10k.step10k.valid_sites_8k.bed` |
