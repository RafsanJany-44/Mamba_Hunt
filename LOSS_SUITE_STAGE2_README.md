# RhythmMamba Stage-2 loss suite

This package adds four controlled loss variants without modifying the model,
data split, A0/A2 augmentation, optimizer, scheduler, or evaluation code.

## New variants

| Code | Loss |
|---|---|
| L2 | `0.2 Pearson + 0.1 Harmonic` |
| L3 | `0.2 Pearson + 1.0 Concentration` |
| L4 | `0.2 Pearson + 1.0 Concentration + 0.1 Harmonic` |
| L5 | `0.2 Pearson + 1.0 CE + 1.0 Concentration` |

L0 (`0.2 Pearson + CE`) and L1 (`L0 + 0.1 Harmonic`) already exist and are
not retrained.

Concentration is calculated from the official normalized sinusoidal spectral
projection over 45--149 BPM. It is the negative log of predicted power inside
the ground-truth HR +/- 3 BPM region. Harmonic competition uses the same
spectrum and compares that correct region with valid 0.5x, 1.5x and 2x bands.

## Files

- `loss_suite_stage2.py`: loss definitions and fixed variant registry.
- `trainer_loss_suite.py`: shared best-checkpoint-only trainer.
- `verify_loss_suite.py`: L0 parity, spectral behavior and gradient checks.
- `train_pure_loss_suite.py`: eight sequential PURE jobs.
- `train_ubfc_loss_suite.py`: eight sequential UBFC jobs.

## Run

First verify:

```bash
python verify_loss_suite.py
```

Then use two terminals:

```bash
CUDA_VISIBLE_DEVICES=0 python train_pure_loss_suite.py
```

```bash
CUDA_VISIBLE_DEVICES=1 python train_ubfc_loss_suite.py
```

Each source runner trains L2--L5 with A0 and A2: eight new models per GPU.
Completed models are verified and skipped on rerun. A failed model is reported
with its full traceback, and the suite continues so one failure does not waste
the remaining overnight GPU time.

Outputs are written under:

```text
results/models/loss_suite_stage2/
```

Each successful experiment contains only its best `.pth` checkpoint, training
history CSV, configuration JSON and completion JSON.
