# RhythmMamba harmonic-aware loss experiment

Place all five Python files directly inside `Mamba_Hunt`. This experiment does
not replace or edit the verified `loss.py`.

## Models

| Source | A0 official augmentation | A2 offline augmentation |
|---|---|---|
| PURE | PURE_A0_HARMONIC | PURE_A2_HARMONIC |
| UBFC | UBFC_A0_HARMONIC | UBFC_A2_HARMONIC |

The loss is the verified HybridLoss plus a fixed weight 0.1 competition between
power near the ground-truth HR and valid 0.5x, 1.5x and 2x harmonic bands. Each
band has a fixed half-width of 2 BPM.

## Mandatory verification

```bash
cd /media/data/rPPG/Code/GitHub/Mamba_Hunt
conda activate mamba_hunting
python verify_harmonic_loss.py
```

Do not train unless every check reports `PASSED`.

## Parallel training

Terminal 1:

```bash
CUDA_VISIBLE_DEVICES=0 python train_pure_harmonic_loss.py
```

Terminal 2:

```bash
CUDA_VISIBLE_DEVICES=1 python train_ubfc_harmonic_loss.py
```

Each script trains A0 first and A2 second. Completed models are skipped safely
on rerun. Incomplete output directories cause a stop rather than overwrite.

## Fixed schedule

- Maximum epochs: 100
- Minimum epochs before early stopping: 30
- Early-stopping patience: 10
- OneCycle learning-rate schedule configured for 100 epochs
- Best checkpoint selected by clean validation total loss
- Best checkpoint only; base and harmonic loss components logged separately

Outputs are written through the existing `results` link under:

```text
results/models/harmonic_loss_stage1/
```
