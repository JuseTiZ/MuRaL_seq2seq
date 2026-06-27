# CODEX.md - mural_seq2seq

This repository contains `mural_s2s`, a PyTorch seq2seq model for predicting
per-base germline mutation rates from DNA sequence.

## Project Overview

The model consumes one-hot encoded reference DNA:

```text
sequence: (B, 4, L)    # channels are A, C, G, T
```

and predicts mutation-rate tracks:

```text
prediction: (B, 4, L)  # mut_to_A, mut_to_C, mut_to_G, mut_to_T
```

Training targets are loaded from BigWig files. The default target tensor has
five channels:

```text
target: (B, 5, L)
target[:, :4, :]  # mutation target channels
target[:, 4, :]   # coverage mask
```

## Directory Map

```text

├── config.py              TrainingConfig dataclass
├── model.py               PuffinD model and ConvBlock
├── loss.py                Poisson + Pseudo-KL loss
├── train.py               training CLI entry point
├── predict.py             prediction CLI entry point
├── data/
│   ├── genome.py          FASTA loading and DNA one-hot encoding
│   ├── targets.py         BigWig feature loading
│   ├── sampler.py         BED interval split and retrieval
│   └── dataloader.py      PyTorch Dataset/DataLoader construction
├── training/
│   ├── trainer.py         train/validation steps, optimizer, scheduler
│   └── observer.py        loss and gradient observers
├── evaluation/
│   └── metrics.py         regional Pearson correlation metrics
└── utils/
    └── helpers.py         early stopping, checkpoint helpers, RC helpers
```

## Main Execution Flow

`train.py` is the training entry point.

1. Parse CLI arguments into `TrainingConfig`.
2. Load FASTA through `Genome`.
3. Load four mutation BigWigs plus one coverage mask through
   `GenomicSignalFeatures`.
4. Read BED intervals with `IntervalsSampler`.
5. Split intervals by chromosome into train/validate/test.
6. Build `Seq2SeqDataset` and `DataLoader` with `build_dataloader`.
7. Train `PuffinD` with `Trainer`.
8. Save one checkpoint per epoch under `output_dir/checkpoint_<epoch>/model`.
9. Log training loss, validation loss, learning rate, duration, and grouped
   Pearson correlations to `training.log`.

`predict.py` is the prediction entry point. It loads a saved model,
auto-detects `use_reverse` from `<model>.config.pkl` when available, and writes
per-base predictions as TSV or gzipped TSV. Optional center-cropped tiled
inference adds flanking sequence context while preserving the original BED
coordinates.

## Model

`PuffinD` is defined in `model.py`.

- It is a 1D convolutional U-Net-like model with two encoder-decoder passes.
- `ConvBlock` is an inverted residual block with residual addition.
- The default `use_reverse=True` adds:
  - reverse-complement symmetric input convolution:
    `conv(x) + conv(x.flip([1, 2])).flip([2])`
  - an AT/CG reference-base embedding concatenated before the final layer.
- Output uses `Softplus`, so predicted rates are non-negative.

Default output channels are:

```text
mut_to_A, mut_to_C, mut_to_G, mut_to_T
```

## Data Pipeline

`Genome` wraps `pyfaidx.Fasta` and lazily opens the FASTA file. Unknown bases
are encoded as `[0.25, 0.25, 0.25, 0.25]`. Out-of-bound sequence requests are
padded with `N`.

`GenomicSignalFeatures` wraps `pyBigWig`, lazily opens BigWig files, stacks
features into `(n_features, L)`, and converts `NaN` or negative values to zero.

`IntervalsSampler` reads BED intervals and partitions them by chromosome:

- `val_chroms` go to `validate`
- `test_chroms` go to `test`
- all other chromosomes go to `train`

The sampler is only a data manager. Actual iteration and shuffling are handled
by PyTorch `DataLoader`.

`Seq2SeqDataset` is map-style. `__getitem__(idx)` directly maps to a BED
interval, retrieves `(seq, target)`, optionally applies reverse-complement
augmentation, and returns:

```text
sequence: (4, L)
target:   (C, L)
meta:     Metadata(chrom, start, end)
```

The collate function stacks these into batch tensors.

## Loss and Masking

The loss is `Poisson_PseudoKL` in `loss.py`:

```text
loss = pseudo_kl + total_weight * poisson
```

Important masking behavior is in `training/trainer.py`.

1. Coverage mask:
   - `target[:, -1, :]` is used as a per-position mask.
   - Predictions and target values are multiplied by this mask before loss.
   - Loss normalization uses effective mask length instead of fixed sequence
     length.

2. No-mutation mask:
   - At reference A positions, `mut_to_A` target is zeroed.
   - At reference C positions, `mut_to_C` target is zeroed.
   - At reference G positions, `mut_to_G` target is zeroed.
   - At reference T positions, `mut_to_T` target is zeroed.
   - This is applied to targets only during training/validation so the model is
     penalized for predicting self-mutation rates.

Prediction additionally hard-masks self-mutation channels with:

```python
preds = preds * (1 - sequence)
```

## Evaluation

`evaluation/metrics.py` provides:

- `calc_regional_correlation`: Pearson correlation over interval-level mean
  rates per channel.
- `calc_regional_correlation_grouped`: the main validation metric used by
  training. It groups positions by reference nucleotide A/C/G/T, then computes
  interval-level mean prediction-vs-target Pearson correlation for each output
  channel.

## Checkpoints

Use `save_model(model, config, save_path)` from `utils/helpers.py`.
It writes:

```text
<save_path>              # torch state_dict
<save_path>.config.pkl   # pickled TrainingConfig
```

Prediction uses the config pickle to recover model options such as
`use_reverse`.

## Common Commands

Training example:

```bash
python train.py \
    --fasta /path/to/hg19.fa \
    --intervals /path/to/segments.bed \
    --target-dir /path/to/bigwig_dir \
    --mask-bw /path/to/mask.bw \
    --output-dir ./output \
    --batch-size 32 \
    --epochs 20
```

Prediction example:

```bash
python predict.py \
    --fasta /path/to/hg19.fa \
    --intervals /path/to/segments.bed \
    --target-dir /path/to/bigwig_dir \
    --mask-bw /path/to/mask.bw \
    --model ./output/checkpoint_1/model \
    --output predictions.tsv.gz \
    --sequence-length 10000 \
    --center-output-length 5000 \
    --mode test
```

Prediction output columns:

```text
chrom, start, end, mut_rate_A, mut_rate_C, mut_rate_G, mut_rate_T
```

Coordinates are 0-based, one row per genomic position.

### Context-aware tiled prediction

When `--center-output-length` is set, each BED interval is divided into output
tiles. Every tile is centered in a fixed `--sequence-length` FASTA window, the
model predicts that complete window, and only the center tile is written. The
minimum flanking context per side is:

```text
(sequence_length - center_output_length) / 2
```

For `--sequence-length 10000 --center-output-length 5000`, a 10 kb BED interval
produces two 10 kb model inputs. Their central 5 kb predictions are concatenated
to recover the original interval exactly, with 2.5 kb of context on both sides
of each tile. The corresponding 100 kb/50 kb settings provide 25 kb flanks.
Without `--center-output-length`, prediction retains the original full-window
behavior. BED interval length must equal `--sequence-length`.

## Development Notes

- Prefer keeping the data shape convention explicit in new code:
  - sequence tensors are `(B, 4, L)`
  - target/prediction tensors are `(B, C, L)`
- Keep target channel order aligned with `[mut_to_A, mut_to_C, mut_to_G, mut_to_T]`.
- If adding target channels, inspect `_mask_no_mut`, reverse-complement logic,
  metrics, and prediction output formatting.
- Avoid in-place edits on tensors that participate in autograd.
- Keep checkpoint compatibility in mind when changing `PuffinD` constructor
  arguments or `TrainingConfig` fields.
- The current package is designed to run directly via `python train.py`
  and `python predict.py`; both scripts add the project root to
  `sys.path`.
