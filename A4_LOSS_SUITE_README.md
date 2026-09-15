# A4 + L1-L4 controlled loss experiment

The experiment trains four loss variants independently for PURE and UBFC.
All new experimental loss terms initially have weight 1.0. The official
Pearson coefficient remains 0.2.

| Code | Definition |
|---|---|
| L1 | `0.2 Pearson + 1.0 CE + 1.0 harmonic` |
| L2 | `0.2 Pearson + 1.0 harmonic` |
| L3 | `0.2 Pearson + 1.0 concentration` |
| L4 | `0.2 Pearson + 1.0 concentration + 1.0 harmonic` |

Every model uses A4 sampling: 50% original and 12.5% for each of JPEG, blur,
gamma, and contrast. Official RhythmMamba online augmentation remains enabled.
Validation uses the clean source 0.8-1.0 split. Maximum training is 100 epochs,
minimum training is 30 epochs, and early-stopping patience is 10 epochs. Only
the lowest-clean-validation-loss checkpoint is retained.

Run verification first:

```bash
python verify_a4_loss_suite.py
```

Then run the two suites in separate terminals:

```bash
CUDA_VISIBLE_DEVICES=0 python train_ubfc_a4_loss_suite.py \
  2>&1 | tee results/logs/UBFC_A4_L1_L4_training.log
```

```bash
CUDA_VISIBLE_DEVICES=1 python train_pure_a4_loss_suite.py \
  2>&1 | tee results/logs/PURE_A4_L1_L4_training.log
```
