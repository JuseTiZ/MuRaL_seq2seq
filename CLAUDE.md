# CLAUDE.md — mural_s2s

Seq2seq germline mutation rate predictor. Pure DNA sequence (one-hot A/C/G/T) → per-position mutation probabilities (4 channels: mut_to_A/C/G/T). Built from scratch, referencing `notebook/unet_model*.ipynb`.

## Environment

```bash
conda activate mural
# Python 3.8.5, PyTorch 1.10.2 + CUDA 11.3
# pyfaidx, pyBigWig, pytabix, scipy
```

## Architecture

**Model**: `PuffinD` U-Net (no reverse-complement module — kept for later ablation).
- Input: `(B, 4, 10000)` one-hot DNA
- Output: `(B, 4, 10000)` per-position mutation probabilities via Softplus
- ~13M trainable params
- Two encoder-decoder passes with skip connections. Strided convs for downsampling (4×, 5×), Upsample+Conv for upsampling.

**Loss**: `Poisson_PseudoKL` — KL divergence (element-wise) + Poisson term (window-total), weighted 1:1.

**Data flow**:
1. `Genome` (pyfaidx) loads FASTA → one-hot `(L, 4)`
2. `GenomicSignalFeatures` (pyBigWig) loads 4 mutation rate + 1 coverage mask BigWig → `(5, L)`
3. `IntervalsSampler` reads BED intervals, partitions by chromosome holdout (chr1=val, chr2=test, rest=train). No-replacement shuffle per epoch.
4. `build_dataloader` → finite PyTorch Dataset (one epoch = one pass through all intervals)

**Trainer**: Adam + StepLR (per-step exponential decay with floor `min_lr`). Gradient clipping at 10.0. Coverage mask applied everywhere (loss + evaluation).

**Metrics**: Per-nucleotide-grouped Pearson correlation — regional mean predictions vs targets, stratified by A/C/G/T positions.

## Key design decisions (from grill session)

| Decision | Choice |
|----------|--------|
| Output channels | 4 (basic mut_to_A/C/G/T, no population labels) |
| Reverse-complement | Not in model; optional data augmentation (off by default) |
| Loss | `poisson_total_kl` (simple, no base-grouped variant) |
| Mask scope | Coverage mask applied in loss AND evaluation |
| Target masking | Coverage mask only (no A→A masking; labels already 0) |
| Epoch definition | Finite Dataset: one pass through all intervals |
| Prediction window | Full 10kb (configurable `center_bin_to_predict`) |
| Scheduler | Standard exponential decay with min_lr floor (no cyclic restart) |
| Checkpointing | Every epoch |
| Edge padding (-1) | Zeroed out and included |

## File map

```
mural_s2s/
├── config.py          TrainingConfig dataclass — all paths + hyperparams
├── model.py           PuffinD(n_output_channels), ConvBlock, count_parameters
├── loss.py            poisson(), PseudoPoissonKL(), Poisson_PseudoKL()
├── data/
│   ├── genome.py      Genome(fasta_path) — get_encoding_from_coords()
│   ├── targets.py     GenomicSignalFeatures(bw_paths, names) — get_feature_data()
│   ├── sampler.py     IntervalsSampler — chromosome-partitioned sampling
│   └── dataloader.py  Seq2SeqDataset, build_dataloader(), _collate_batch()
├── training/
│   ├── trainer.py     Trainer — Adam+StepLR, train_step(), valid_step()
│   └── observer.py    LossMinor, GradMinor
├── evaluation/
│   └── metrics.py     calc_regional_correlation(), calc_regional_correlation_grouped()
├── utils/
│   └── helpers.py     EarlyStopping, save_model(), load_model(), reverse_complement_batch()
├── train.py           CLI: --fasta --intervals --target-dir --mask-bw --output-dir ...
└── predict.py         CLI: --model --output --mode {train,validate,test}
```

## Training

```bash
python mural_s2s/train.py \
    --fasta /public5/home/songhui/data/hg19/hg19_ucsc_ordered.fa \
    --intervals /public5/home/songhui/mural_snv/s2m/segments/segments.win10k.step10k.valid_sites_8k.bed \
    --target-dir /public5_data/home/songhui/s2m \
    --mask-bw /public5_data/home/songhui/s2m/genome.mask_coverage_15_45.bw \
    --output-dir ./output \
    --batch-size 32 --epochs 20 --lr 5e-3
```

All paths are CLI parameters — no hardcoded paths in the library code.

## Prediction

```bash
python mural_s2s/predict.py \
    --fasta ... --intervals ... --target-dir ... --mask-bw ... \
    --model ./output/checkpoint_20/model \
    --output predictions.tsv.gz \
    --mode test
```

Output: gzipped TSV with columns `chrom, pos, mut_rate_A, mut_rate_C, mut_rate_G, mut_rate_T`.

## Data sources (for reference)

| Resource | Path |
|----------|------|
| hg19 FASTA | `/public5/home/songhui/data/hg19/hg19_ucsc_ordered.fa` |
| Mutation rate BigWigs | `/public5_data/home/songhui/s2m/genome.mut_to_{A,C,G,T}.bw` |
| Coverage mask | `/public5_data/home/songhui/s2m/genome.mask_coverage_15_45.bw` |
| 10kb intervals | `/public5/home/songhui/mural_snv/s2m/segments/segments.win10k.step10k.valid_sites_8k.bed` |
