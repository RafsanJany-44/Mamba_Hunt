# A4 L3/L4 Loss-Weight Tuning

This experiment keeps the RhythmMamba architecture, A4 sampling, official
online augmentation, 80/20 source split, optimizer, seed, and training policy
fixed. Only the concentration and harmonic coefficients change.

## Jobs per source dataset

| Experiment code | Loss | Pearson | CE | Concentration | Harmonic |
|---|---|---:|---:|---:|---:|
| `L3_C010` | L3 | 0.2 | 0 | 0.10 | 0 |
| `L3_C025` | L3 | 0.2 | 0 | 0.25 | 0 |
| `L3_C050` | L3 | 0.2 | 0 | 0.50 | 0 |
| `L4_C025_H100` | L4 | 0.2 | 0 | 0.25 | 1.0 |
| `L4_C025_H200` | L4 | 0.2 | 0 | 0.25 | 2.0 |
| `L4_C050_H100` | L4 | 0.2 | 0 | 0.50 | 1.0 |
| `L4_C050_H200` | L4 | 0.2 | 0 | 0.50 | 2.0 |
| `L4_C100_H200` | L4 | 0.2 | 0 | 1.00 | 2.0 |

The already completed v9 experiments provide the excluded unit-weight
references: `L3` with concentration 1.0 and `L4` with concentration 1.0 plus
harmonic 1.0.

## Verification

Run before training:

```bash
python verify_a4_loss_weight_tuning.py
```

The verifier checks all eight definitions, finite nonzero gradients, A4 loader
availability, and the expected PURE and UBFC training/validation counts.

## Two-GPU training

Terminal 1, UBFC on GPU 0:

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python train_ubfc_a4_loss_weight_tuning.py 2>&1 \
  | tee results/logs/UBFC_A4_loss_weight_tuning.log
```

Terminal 2, PURE on GPU 1:

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=1 python train_pure_a4_loss_weight_tuning.py 2>&1 \
  | tee results/logs/PURE_A4_loss_weight_tuning.log
```

Each source runs eight models sequentially. Training uses at most 100 epochs,
at least 30 epochs, early-stopping patience 10, seed 100, clean validation, and
best-checkpoint-only saving.

Outputs are written under:

```text
results/models/a4_loss_weight_tuning/
```

An existing completed model is skipped only when its completion record matches
the expected loss variant, weights, and seed. An incomplete directory causes a
hard failure for that job and is preserved for diagnosis.
