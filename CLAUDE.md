# CLAUDE.md — mural_s2s

Seq2seq germline mutation rate predictor. Pure DNA sequence (one-hot A/C/G/T) → per-position mutation probabilities (4 channels: mut_to_A/C/G/T). Built from scratch, referencing `notebook/unet_model*.ipynb`.

## Environment

```bash
conda activate mural
# Python 3.8.5, PyTorch 1.10.2 + CUDA 11.3
# pyfaidx, pyBigWig, pytabix, scipy
```

## Architecture

**Model**: `PuffinD(n_output_channels=4, use_reverse=True)` (~13M params).
- Input: `(B, 4, 10000)` one-hot DNA
- Output: `(B, 4, 10000)` via Softplus
- Two encoder-decoder passes with skip connections.
- `use_reverse=True` (default): reverse-complement symmetric conv on input (`conv(x) + conv(flip(x))`) + AT/CG embedding (2-class, 4-dim, grouped as `{A,T}` / `{C,G}`) concatenated to final layer (`final_dim = 64 + 4 = 68`). The AT/CG grouping respects the reverse-complement symmetry of the conv layer.
- `collapse_map = [0, 1, 1, 0]` maps one-hot `[A,C,G,T]` → AT/CG classes (A/T→0, C/G→1).
- `use_reverse=False`: no reverse-complement module, no AT/CG embedding (`final_dim = 64`). Compatible with old checkpoints.
- `use_reverse` is stored in `TrainingConfig` and persisted in checkpoint `.config.pkl`.

**Loss**: `Poisson_PseudoKL = kl_term + total_weight * poisson_term`
- Both terms normalize by effective mask length (`mask.sum()`) instead of fixed `seq_len`, for consistent scaling across intervals with different coverage fractions.
- KL term: `y_true * log(y_true/y_pred) + y_pred - y_true`, element-wise, then mean over effective L
- Poisson term: `(rate - count * log(rate)) / effective_L` where rate=sum(y_pred), count=sum(y_true)
- Both terms use epsilon protection to prevent log(0)

**Data flow**:
1. `Genome` (pyfaidx) — FASTA → one-hot `(L, 4)`, N→0.25, edge-pad with N
2. `GenomicSignalFeatures` (pyBigWig) — BigWig → `(5, L)`, NaN→0, neg→0
3. `IntervalsSampler` — reads BED intervals, partitions by chromosome (chr1=val, chr2=test). Purely a data manager; does NOT handle iteration/shuffling.
4. `Seq2SeqDataset` — map-style PyTorch Dataset. `__getitem__(idx)` directly looks up the interval from the sampler's interval list and calls `sampler.retrieve()`. Compatible with PyTorch DataLoader `shuffle` and `num_workers`.
5. `build_dataloader(sampler, batch_size, mode, ...)` — mode-aware loader factory. Training uses `shuffle=True`, validation/prediction uses `shuffle=False`.

**Pre-loss masking** (trainer.py `_mask_no_mut`):
1. No-mutation mask: zeros out A→A, C→C, G→G, T→T positions in targets only (not preds);
   this forces the model to learn to suppress predictions at matched-base positions via the loss
2. Coverage mask: zeros out low-coverage positions (from mask_coverage_15_45.bw)
3. Applied in both train_step and valid_step

## Predict output format

TSV (gzipped or plain) with columns:
```
chrom, start, end, mut_rate_A, mut_rate_C, mut_rate_G, mut_rate_T
```
- `start` is 0-based, `end = start + 1` (single-base intervals)
- Self-mutation channels are hard-masked to 0 (A→A at A sites = 0, etc.)
- Output is streamed via `gzip.open` (no temp file)
- `--center-output-length N` enables context-aware tiled inference. Each BED
  interval is divided into output tiles of at most `N` bases; every tile is
  predicted from a fixed `--sequence-length` input centered on that tile, and
  only the center is written. The minimum flank on each side is
  `(sequence_length - center_output_length) / 2`.
- Center-cropped inference preserves the original BED coordinates and emits
  each position once. Without `--center-output-length`, full-window prediction
  remains unchanged.

## Key data facts (critical — verified from actual BigWig inspection)

- **`genome.mut_to_{A,C,G,T}.bw`**: BINARY — only `1.0` and `nan`. `1.0` at matched-ref-base positions (A→A for mut_to_A channel = 1.0) AND at genuine mutation sites. After `_mask_no_mut`, only genuine mutation sites remain as 1.0, everything else → 0.
- **`genome.us1_mid.to_X.bw`**, **`genome.mid_ds1.to_X.bw`**: CONTINUOUS — values 0~0.996. These are k-mer based flanking-region mutation rates. Currently NOT used (model only uses the 4 basic mut_to_X channels).
- **`genome.mask_coverage_15_45.bw`**: Coverage mask (15–45×)

## Bugs fixed during development

1. **Poisson formula was swapped** (loss.py): wrote `count - rate * log(count)` instead of `rate - count * log(rate)`. Caused loss to be negative and decrease without bound. Fixed by standardizing variable names to `y_pred, y_true`.

2. **Inplace modification broke autograd** (trainer.py `_mask_no_mut`): `preds[:, ch, :][ref_mask] = 0` inplace-mutated Softplus output tensor, breaking backward graph. Fixed by using multiplication mask instead.

3. **Model not moved to GPU** (train.py): `model = PuffinD(...)` without `.to(device)`. Fixed.

4. **Module import error**: `python mural_s2s/train.py` fails with `ModuleNotFoundError` because project root not in sys.path. Fixed by adding `sys.path.insert(0, ...)` at top of train.py and predict.py.

5. **`collapse_map` was amino/keto instead of AT/CG** (model.py): `[0,0,1,1]` mapped `{A,C},{G,T}` (amino/keto), but the AT/CG grouping `[0,1,1,0]` mapping `{A,T},{C,G}` is required for reverse-complement symmetry with the conv layer (`conv(x) + conv(flip(x))`). Fixed.

6. **Reverse-complement aug didn't swap target channels** (dataloader.py, helpers.py): `_reverse_complement` only position-reversed targets, but `mut_to_A`↔`mut_to_T` and `mut_to_C`↔`mut_to_G` must swap. Fixed.

7. **`LossMinor` epoch mean loss was off by batch_size** (observer.py): accumulated batch-mean loss without weighting by sample count, dividing the mean again. Fixed with `self.loss += loss * sample_number`.

8. **Loss normalized by fixed seq_len** (loss.py): Poisson and KL terms divided by `L=10000` regardless of coverage fraction. Fixed by passing mask and using `mask.sum()` as effective length.

9. **Map-style Dataset refactor** (dataloader.py, sampler.py): `Seq2SeqDataset.__getitem__` previously ignored `idx`, using internal sampler state — causing duplicate sampling with `num_workers > 0`. Refactored to true map-style: `__getitem__` uses `idx` to fetch intervals, DataLoader `shuffle` controls randomization. Removed `sample()`, `_randcache`, `sample_next`, `mode` setter from `IntervalsSampler`.

## Design decisions (from grill session)

| Decision | Choice |
|----------|--------|
| Output channels | 4 (basic mut_to_A/C/G/T) |
| Reverse-complement | AT/CG embedding in model (use_reverse=True). Data aug: configurable, off by default |
| Loss | `poisson_total_kl` with effective-length normalization |
| Mask scope | Coverage mask everywhere (loss + evaluation). Self-mutation hard mask in predict output |
| No-mutation masking | Active in targets (needed: BigWig stores 1.0 at ref-base positions) |
| Epoch definition | Finite Dataset: one pass through all intervals |
| Scheduler | StepLR per-batch exponential decay + min_lr floor |
| Checkpointing | Every epoch + best-epoch tracking. Config saved as `.config.pkl` |
| Predict output | `chrom, start, end, mut_rate_*` (start=0-based), gzip streaming |
| Dataset style | Map-style (idx-based), DataLoader shuffle for training |

## File map

```
mural_s2s/             Root path (the directory name which project locates in)
├── config.py          TrainingConfig dataclass (includes use_reverse)
├── model.py           PuffinD(n_output_channels, use_reverse), ConvBlock, count_parameters
├── loss.py            poisson(), PseudoPoissonKL(), Poisson_PseudoKL() — mask-aware
├── data/
│   ├── genome.py      Genome(fasta_path) — get_encoding_from_coords()
│   ├── targets.py     GenomicSignalFeatures(bw_paths, names) — get_feature_data()
│   ├── sampler.py     IntervalsSampler — chr-partitioned interval manager, retrieve()
│   └── dataloader.py  Seq2SeqDataset (map-style), build_dataloader(mode=...), collate
├── training/
│   ├── trainer.py     Trainer — Adam+StepLR, train_step(), valid_step(), _mask_no_mut()
│   └── observer.py    LossMinor, GradMinor — loss tracking with current_loss property
├── evaluation/
│   └── metrics.py     calc_regional_correlation(), calc_regional_correlation_grouped()
├── utils/
│   └── helpers.py     EarlyStopping, save_model(), load_model(), reverse_complement_batch()
├── train.py           CLI: --fasta --intervals --target-dir --mask-bw --output-dir ...
│                      --disable-reverse --reverse-complement-aug
└── predict.py         CLI: --model --output --mode {train,validate,test}
                        gzip.open streaming output with chrom/start/end/mut_rate_* columns,
                        self-mutation hard mask, auto-detect use_reverse from checkpoint config,
                        shuffle=False for genomic-order output, optional context-aware tiled
                        inference via --center-output-length,
                        --progress-every N batches.
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

## Prediction command (example)

```bash
conda activate mural
python mural_s2s/predict.py \
    --fasta /public5/home/songhui/data/hg19/hg19_ucsc_ordered.fa \
    --intervals /public5/home/songhui/mural_snv/s2m/segments/segments.win10k.step10k.valid_sites_8k.bed \
    --target-dir /public5_data/home/songhui/s2m \
    --mask-bw /public5_data/home/songhui/s2m/genome.mask_coverage_15_45.bw \
    --model ./output/checkpoint_19/model \
    --output predictions_chr2.tsv.gz \
    --sequence-length 10000 \
    --center-output-length 5000 \
    --mode test
```

Output: gzipped TSV in genomic-position order, columns: `chrom, start, end, mut_rate_A, mut_rate_C, mut_rate_G, mut_rate_T`.

With the settings above, each 10 kb BED interval is covered by two predictions.
Each prediction receives a 10 kb sequence, retains only its central 5 kb, and
therefore provides at least 2.5 kb of context on each side of every retained
tile. For a 100 kb-trained model, the analogous settings are
`--sequence-length 100000 --center-output-length 50000`, giving 25 kb flanks.

## Data sources

| Resource | Path |
|----------|------|
| hg19 FASTA | `/public5/home/songhui/data/hg19/hg19_ucsc_ordered.fa` |
| Mutation BigWigs (mut_to_X) | `/public5_data/home/songhui/s2m/genome.mut_to_{A,C,G,T}.bw` |
| Coverage mask | `/public5_data/home/songhui/s2m/genome.mask_coverage_15_45.bw` |
| Flanking BigWigs (unused) | `/public5_data/home/songhui/s2m/genome.{us1_mid,mid_ds1,...}.to_{A,C,G,T}.bw` |
| 10kb intervals BED | `/public5/home/songhui/mural_snv/s2m/segments/segments.win10k.step10k.valid_sites_8k.bed` |
