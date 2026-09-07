# PURE A4 L0 seed-100 pilot

This package also includes backward-compatible replacements for the generic A4
generator and trainer used by UBFC. Existing UBFC results are not changed.

Generate PURE A4 data:

```bash
python generate_pure_a4_offline_augmentation.py
```

Verify:

```bash
python verify_pure_a4_training.py
```

Train the pilot on GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 python train_pure_a4_l0_seed100.py
```

The best checkpoint is saved under:

```text
results/models/pure_a4_pilot/PURE_A4_L0_SEED100/
PURE_A4_L0_SEED100_RhythmMamba_Best.pth
```
